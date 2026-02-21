import torch
from diffusers.utils import export_to_video

from optimum.onnxruntime.modeling_diffusion import ORTPipelineForText2Video

model_id = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

device = "cpu"
seed=1

prompt = "A cat walks on the grass, realistic"
generator = torch.Generator(device=device).manual_seed(seed)
num_frames=121
num_inference_steps=50

inf_kwargs = {
    "prompt": prompt,
    "generator": generator,
    "num_frames": num_frames,
    "num_inference_steps": num_inference_steps,
}

module_arch_fields = {
    "text_encoder": ["hidden_size", "vocab_size"],
    "text_encoder_2": ["d_model", "vocab_size"],
    "transformer": ["in_channels", "out_channels","text_embed_dim", "text_embed_2_dim", "image_embed_dim"],
    "vae_encoder": ["in_channels", "out_channels", "latent_channels"],
    "vae_decoder": ["in_channels", "out_channels", "latent_channels"],
}

pipe = ORTPipelineForText2Video.from_pretrained(
    model_id,
    provider=providers[1],  # Force GPU
    torch_dtype=torch.float16,
    inf_kwargs = inf_kwargs,
    module_arch_fields = module_arch_fields,
)