# import torch
# import torch.nn.functional as F

# @torch.no_grad()
# def project_face_embs(pipeline, face_embs):
#     '''
#     face_embs: (N, 512) normalized ArcFace embeddings
#     '''
#     # Ensure face_embs is on the correct device and has the correct dtype
#     face_embs = face_embs.to(device=pipeline.device, dtype=torch.float16)

#     arcface_token_id = pipeline.tokenizer.encode("id", add_special_tokens=False)[0]

#     input_ids = pipeline.tokenizer(
#         "photo of a id person",
#         truncation=True,
#         padding="max_length",
#         max_length=pipeline.tokenizer.model_max_length,
#         return_tensors="pt",
#     ).input_ids.to(pipeline.device)

#     face_embs_padded = F.pad(face_embs, (0, pipeline.text_encoder.config.hidden_size-512), "constant", 0)
#     token_embs = pipeline.text_encoder(input_ids=input_ids.repeat(len(face_embs), 1), return_token_embs=True)
    
#     # Ensure token_embs is on the correct device and has the correct dtype
#     token_embs = token_embs.to(device=pipeline.device, dtype=torch.float16)
    
#     # Use masked_scatter_ to avoid the dtype mismatch error
#     mask = (input_ids == arcface_token_id).unsqueeze(-1).expand_as(token_embs)
#     token_embs = token_embs.masked_scatter_(mask, face_embs_padded)

#     prompt_embeds = pipeline.text_encoder(
#         input_ids=input_ids,
#         input_token_embs=token_embs
#     )[0]

#     return prompt_embeds.to(dtype=torch.float16)

import torch
import torch.nn.functional as F

@torch.no_grad()
def project_face_embs(pipeline, face_embs, additional_text=None):

    '''
    face_embs: (N, 512) normalized ArcFace embeddings
    '''

    arcface_token_id = pipeline.tokenizer.encode("id", add_special_tokens=False)[0]

    input_ids = pipeline.tokenizer(
            "photo of a id person" + (' ' + additional_text if additional_text else ''),
            truncation=True,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(pipeline.device)

    face_embs_padded = F.pad(face_embs, (0, pipeline.text_encoder.config.hidden_size-512), "constant", 0)
    token_embs = pipeline.text_encoder(input_ids=input_ids.repeat(len(face_embs), 1), return_token_embs=True)
    token_embs[input_ids==arcface_token_id] = face_embs_padded

    prompt_embeds = pipeline.text_encoder(
        input_ids=input_ids,
        input_token_embs=token_embs
    )[0]

    return prompt_embeds