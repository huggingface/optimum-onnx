
import torch
from diffusers.utils import export_to_video

from optimum.onnxruntime.modeling_diffusion import ORTPipelineForText2Video

wan_list = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "ali-vilab/text-to-video-ms-1.7b",
]

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

pipe = ORTPipelineForText2Video.from_pretrained(
    wan_list[0],
    provider=providers[0],  # Force GPU
    torch_dtype=torch.float16,
)
print("Loaded successfully on:", pipe.device)
prompt = "A cat walks on the grass grass grass"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

negative_prompt = ""

output = pipe(prompt=prompt, negative_prompt=negative_prompt, height=240, width=416, num_frames=21).frames[0]
export_to_video(output, "output.mp4", fps=15)
