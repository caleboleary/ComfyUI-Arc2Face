import os
import logging
import torch
import numpy as np
from PIL import Image
import cv2
import folder_paths
from insightface.app import FaceAnalysis
from insightface.utils import storage
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler, AutoPipelineForImage2Image
import json
from safetensors.torch import load_file
from .utils import project_face_embs
from .models import CLIPTextModelWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Arc2FaceFaceExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "average_method": (["average", "median", "trimmed_mean", "ensemble_average", "ensemble_median", "max_pooling", "min_pooling", "rounded_mode", "rounded_mode_averaging", "random_sampling"],),
                "n_outliers": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("FACE_EMBEDDING",)
    FUNCTION = "extract_face_embedding"
    CATEGORY = "Arc2Face"

    def __init__(self):
        def custom_download(sub_dir, name, force, root='~/.insightface'):
            return os.path.join(folder_paths.models_dir, "antelopev2")

        storage.download = custom_download
        self.app = FaceAnalysis(name='antelopev2', root=os.path.join(folder_paths.models_dir, ""), providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_face_embedding(self, images, average_method, n_outliers):
        try:
            img_np = images.cpu().numpy()
            if img_np.ndim == 3:
                img_np = np.expand_dims(img_np, axis=0)
            if img_np.ndim != 4:
                raise ValueError(f"Invalid image dimensions: {img_np.shape}")
            
            all_embeddings = []
            
            for img in img_np:
                img = np.clip(255. * img, 0, 255).astype(np.uint8)
                if img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]
                elif img.shape[2] != 3:
                    raise ValueError(f"Unexpected number of channels: {img.shape[2]}")

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                min_size = 64
                if img.shape[0] < min_size or img.shape[1] < min_size:
                    scale = min_size / min(img.shape[0], img.shape[1])
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                faces = self.app.get(img)
                
                if faces:
                    embeddings = [torch.tensor(face.embedding, dtype=torch.float32) for face in faces]
                    all_embeddings.extend(embeddings)
            
            if not all_embeddings:
                raise ValueError("No faces detected in any of the images")

            all_embeddings = self.remove_outliers(all_embeddings, n_outliers)
            avg_embedding = self.average_embeddings(all_embeddings, method=average_method)
            
            return (avg_embedding,)
        
        except Exception as e:
            logger.error(f"Error in Arc2FaceFaceExtractor: {str(e)}")
            return (torch.zeros(512, dtype=torch.float32),)

    def remove_outliers(self, embeddings, n_outliers):
        if n_outliers == 0 or n_outliers >= len(embeddings):
            return embeddings

        centroid = torch.mean(torch.stack(embeddings), dim=0)
        distances = [torch.norm(e - centroid, p=2).item() for e in embeddings]
        outlier_indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[:n_outliers]
        return [e for i, e in enumerate(embeddings) if i not in outlier_indices]

    def average_embeddings(self, embeddings, method="average"):
        if not embeddings:
            raise ValueError("No valid embeddings to average")

        logger.info(f"Combining {len(embeddings)} embeddings using method: {method}")

        embeddings_stack = torch.stack(embeddings)

        if method == "average":
            return torch.mean(embeddings_stack, dim=0)
        elif method == "median":
            return torch.median(embeddings_stack, dim=0).values
        elif method == "trimmed_mean":
            lower_bound, upper_bound = int(0.15 * len(embeddings)), int(0.85 * len(embeddings))
            sorted_embeddings = embeddings_stack.sort(dim=0).values
            trimmed_embeddings = sorted_embeddings[lower_bound:upper_bound, :]
            return torch.mean(trimmed_embeddings, dim=0)
        elif method == "max_pooling":
            return torch.max(embeddings_stack, dim=0).values
        elif method == "min_pooling":
            return torch.min(embeddings_stack, dim=0).values
        elif method == "rounded_mode":
            mode_embeddings = torch.zeros_like(embeddings_stack[0])
            for dim in range(embeddings_stack.size(1)):
                current_dim_values = embeddings_stack[:, dim]
                rounded_dim_values = torch.round(current_dim_values * 100) / 100
                values, counts = rounded_dim_values.unique(return_counts=True)
                mode_embeddings[dim] = values[counts.argmax()]
            return mode_embeddings
        elif method == "rounded_mode_averaging":
            averaged_mode_embeddings = torch.zeros_like(embeddings_stack[0])
            for dim in range(embeddings_stack.size(1)):
                current_dim_values = embeddings_stack[:, dim]
                rounded_dim_values = torch.round(current_dim_values * 100) / 100
                mode_value, _ = rounded_dim_values.mode()
                contributing_indices = (rounded_dim_values == mode_value).nonzero(as_tuple=True)[0]
                contributing_values = current_dim_values[contributing_indices]
                averaged_mode_embeddings[dim] = torch.mean(contributing_values)
            return averaged_mode_embeddings
        elif method == "ensemble_average":
            mean_embedding = torch.mean(embeddings_stack, dim=0)
            median_embedding = torch.median(embeddings_stack, dim=0).values
            lower_bound, upper_bound = int(0.15 * len(embeddings)), int(0.85 * len(embeddings))
            sorted_embeddings = embeddings_stack.sort(dim=0).values
            trimmed_embeddings = sorted_embeddings[lower_bound:upper_bound, :]
            trimmed_mean_embedding = torch.mean(trimmed_embeddings, dim=0)
            all_averages = torch.stack([mean_embedding, median_embedding, trimmed_mean_embedding])
            return torch.mean(all_averages, dim=0)
        elif method == "ensemble_median":
            mean_embedding = torch.mean(embeddings_stack, dim=0)
            median_embedding = torch.median(embeddings_stack, dim=0).values
            lower_bound, upper_bound = int(0.15 * len(embeddings)), int(0.85 * len(embeddings))
            sorted_embeddings = embeddings_stack.sort(dim=0).values
            trimmed_embeddings = sorted_embeddings[lower_bound:upper_bound, :]
            trimmed_mean_embedding = torch.mean(trimmed_embeddings, dim=0)
            all_averages = torch.stack([mean_embedding, median_embedding, trimmed_mean_embedding])
            return torch.median(all_averages, dim=0).values
        elif method == "random_sampling":
            randomly_sampled_embedding = torch.empty(embeddings_stack.shape[1], dtype=embeddings_stack.dtype).to(embeddings_stack.device)
            for dim in range(embeddings_stack.shape[1]):
                random_index = torch.randint(0, embeddings_stack.shape[0], (1,)).item()
                randomly_sampled_embedding[dim] = embeddings_stack[random_index, dim]
            return randomly_sampled_embedding
        else:
            raise ValueError("Unsupported averaging method.")

class Arc2FaceUNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_path": ("STRING", {"default": "diffusion_pytorch_model.safetensors"})}}

    RETURN_TYPES = ("ARC2FACE_UNET",)
    FUNCTION = "load_unet"
    CATEGORY = "Arc2Face"

    def load_unet(self, model_path):
        arc2face_path = os.path.join(folder_paths.models_dir, "arc2face_checkpoints")
        full_path = os.path.join(arc2face_path, model_path)
        
        with open(os.path.join(arc2face_path, "config.json"), 'r') as f:
            config = json.load(f)

        unet = UNet2DConditionModel(**config)
        unet_state_dict = load_file(full_path)
        unet.load_state_dict(unet_state_dict)
        return (unet,)

class Arc2FaceEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"encoder_path": ("STRING", {"default": "encoder"})}}

    RETURN_TYPES = ("ARC2FACE_ENCODER",)
    FUNCTION = "load_encoder"
    CATEGORY = "Arc2Face"

    def load_encoder(self, encoder_path):
        arc2face_path = os.path.join(folder_paths.models_dir, "arc2face_checkpoints")
        full_path = os.path.join(arc2face_path, encoder_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        encoder = CLIPTextModelWrapper.from_pretrained(
            full_path,
            torch_dtype=dtype
        )
        return (encoder,)

class Arc2FaceGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "face_embedding": ("FACE_EMBEDDING",),
            "unet": ("ARC2FACE_UNET",),
            "encoder": ("ARC2FACE_ENCODER",),
            "negative_prompt": ("STRING", {"default": "ugly, deformed, noisy, blurry, low contrast, split image"}),
            "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "guidance_scale": ("FLOAT", {"default": 2.7, "min": 0.1, "max": 30.0, "step": 0.1}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            "seed": ("INT", {"default": -1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Arc2Face"

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def load_pipeline(self, unet, encoder):
        if self.pipe is None:
            base_model = 'runwayml/stable-diffusion-v1-5'
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                base_model,
                text_encoder=encoder,
                unet=unet,
                torch_dtype=self.dtype,
                safety_checker=None
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)

    def generate(self, face_embedding, unet, encoder, negative_prompt, num_inference_steps, guidance_scale, num_images, width, height, seed):
        self.load_pipeline(unet, encoder)

        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        face_embedding = face_embedding.to(dtype=torch.float16, device=self.device)
        face_embedding = face_embedding / torch.norm(face_embedding, dim=0, keepdim=True)

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            id_emb = project_face_embs(self.pipe, face_embedding.unsqueeze(0))
            
            output = self.pipe(
                prompt_embeds=id_emb,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                width=width,
                height=height,
                generator=generator,
            )

        images = output.images
        comfy_images = []
        for img in images:
            img = img.convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            comfy_images.append(img)

        if len(comfy_images) == 1:
            return (comfy_images[0],)
        return (torch.cat(comfy_images, dim=0),)

class Arc2FaceImg2ImgGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "face_embedding": ("FACE_EMBEDDING",),
            "unet": ("ARC2FACE_UNET",),
            "encoder": ("ARC2FACE_ENCODER",),
            "initial_image": ("IMAGE",),
            "negative_prompt": ("STRING", {"default": "ugly, deformed, noisy, blurry, low contrast, split image"}),
            "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            "guidance_scale": ("FLOAT", {"default": 2.7, "min": 0.1, "max": 30.0, "step": 0.1}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            "seed": ("INT", {"default": -1}),
            "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "extra_param": ("STRING", {"default": ""})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Arc2Face"

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def load_pipeline(self, unet, encoder):
        if self.pipe is None:
            base_model = 'runwayml/stable-diffusion-v1-5'
            
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                base_model,
                text_encoder=encoder,
                unet=unet,
                torch_dtype=self.dtype,
                safety_checker=None
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)

    def generate(self, face_embedding, unet, encoder, initial_image, negative_prompt, num_inference_steps, guidance_scale, num_images, seed, denoise_strength, extra_param):
        self.load_pipeline(unet, encoder)

        logger.info(f"Initial image shape: {initial_image.shape}")

        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        face_embedding = face_embedding.to(dtype=torch.float16, device=self.device)
        face_embedding = face_embedding / torch.norm(face_embedding, dim=0, keepdim=True)

        initial_image_np = initial_image.squeeze().cpu().numpy()
        logger.info(f"Initial image shape after squeeze: {initial_image_np.shape}")
        
        initial_image_pil = Image.fromarray((initial_image_np * 255).astype(np.uint8), mode='RGB')

        logger.info(f"Initial image size: {initial_image_pil.size}")

        width, height = initial_image_pil.size

        if width < 64 or height < 64:
            raise ValueError(f"Initial image is too small, width is {width} and height is {height}")
        
        if initial_image_pil is None:
            raise ValueError("No initial image provided")

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            id_emb = project_face_embs(self.pipe, face_embedding.unsqueeze(0), additional_text=extra_param)
            
            output = self.pipe(
                image=initial_image_pil,
                prompt_embeds=id_emb,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
                strength=denoise_strength,
            )

        images = output.images
        comfy_images = []
        for img in images:
            img = img.convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img)[None,]
            comfy_images.append(img)

        if len(comfy_images) == 1:
            return (comfy_images[0],)
        return (torch.cat(comfy_images, dim=0),)

class Arc2FaceImageGridGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "max_images": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "max_size": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 64}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_grid"
    CATEGORY = "Arc2Face"

    def resize_image(self, img, max_size):
        width, height = img.size
        if width > height:
            if width > max_size:
                height = int(max_size * height / width)
                width = max_size
        else:
            if height > max_size:
                width = int(max_size * width / height)
                height = max_size
        return img.resize((width, height), Image.LANCZOS)

    def generate_grid(self, directory, max_images, max_size):
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if not image_files:
            raise ValueError("No image files found in the specified directory.")
        
        image_files = image_files[:max_images]
        
        images = [self.resize_image(Image.open(os.path.join(directory, img)).convert('RGB'), max_size) for img in image_files]
        
        n = len(images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        grid_width = cols * max_size
        grid_height = rows * max_size
        grid_img = Image.new('RGB', (grid_width, grid_height))
        
        for i, img in enumerate(images):
            x = (i % cols) * max_size
            y = (i // cols) * max_size
            grid_img.paste(img, (x, y))
        
        grid_np = np.array(grid_img).astype(np.float32) / 255.0
        grid_tensor = torch.from_numpy(grid_np)[None,]
        
        logger.info(f"Grid tensor shape: {grid_tensor.shape}")
        logger.info(f"Grid tensor dtype: {grid_tensor.dtype}")
        
        return (grid_tensor,)

NODE_CLASS_MAPPINGS = {
    "Arc2FaceFaceExtractor": Arc2FaceFaceExtractor,
    "Arc2FaceUNetLoader": Arc2FaceUNetLoader,
    "Arc2FaceEncoderLoader": Arc2FaceEncoderLoader,
    "Arc2FaceGenerator": Arc2FaceGenerator,
    "Arc2FaceImg2ImgGenerator": Arc2FaceImg2ImgGenerator,
    "Arc2FaceImageGridGenerator": Arc2FaceImageGridGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Arc2FaceFaceExtractor": "Arc2Face Face Extractor",
    "Arc2FaceUNetLoader": "Arc2Face UNet Loader",
    "Arc2FaceEncoderLoader": "Arc2Face Encoder Loader",
    "Arc2FaceGenerator": "Arc2Face Generator",
    "Arc2FaceImg2ImgGenerator": "Arc2Face Img2Img Generator",
    "Arc2FaceImageGridGenerator": "Arc2Face Image Grid Generator"
}