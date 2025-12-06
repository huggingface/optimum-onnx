import torch
from optimum.onnxruntime.modeling_diffusion import ORTPipelineForText2Video

wan_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
hunyuan_id = "hunyuanvideo-community/HunyuanVideo"
cogvideo = "THUDM/CogVideoX-5b"

pipe = ORTPipelineForText2Video.from_pretrained(wan_id, variant="fp16")