import torch

dtype = torch.bfloat16
device = "cuda:0"

from diffusers import HunyuanVideo15Pipeline, attention_backend
from diffusers.utils import export_to_video

pipe = HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v", torch_dtype=dtype)

for name in ['vae', 'text_encoder', 'text_encoder_2', 'transformer', 'scheduler', 'guider']:
    module = getattr(pipe, name, None)
    if module is None:
        continue
    if hasattr(module, 'config'):
        print(f"--- {name} ---")
        print(module.config)
    else:
        print(f"--- {name} ---")
        print(module)  # e.g., guider or tokenizer


pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# generator = torch.Generator(device=device).manual_seed(seed)

# with attention_backend("_flash_3_hub"): # or `"flash_hub"` if you are not on H100/H800
#     video = pipe(
#         prompt=prompt,
#         generator=generator,
#         num_frames=121,
#         num_inference_steps=50,
#     ).frames[0]
#     export_to_video(video, "output.mp4", fps=24)
