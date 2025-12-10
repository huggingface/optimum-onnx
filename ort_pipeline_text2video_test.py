import torch
from optimum.onnxruntime.modeling_diffusion import ORTPipelineForText2Video

wan_list = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "ali-vilab/text-to-video-ms-1.7b",
]

providers = [
    "CUDAExecutionProvider", 
    "CPUExecutionProvider"
]

pipe = ORTPipelineForText2Video.from_pretrained(
    wan_list[2],
    provider=providers[0],   # Force GPU
    torch_dtype=torch.float16
)
print("Loaded successfully on:", pipe.device)
# Ensure weights move to GPU
