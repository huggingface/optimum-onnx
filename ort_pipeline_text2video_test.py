import torch
from optimum.onnxruntime.modeling_diffusion import ORTPipelineForText2Video

wan_list = [
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
]

providers = [
    "CUDAExecutionProvider", 
    "CPUExecutionProvider"
]

pipe = ORTPipelineForText2Video.from_pretrained(
    wan_list[0],
    provider=providers[0],   # Force GPU
    torch_dtype=torch.float16
)
print("Loaded successfully on:", pipe.device)
# Ensure weights move to GPU
