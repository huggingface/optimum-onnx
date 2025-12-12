import torch
from diffusers.utils import export_to_video

from optimum.onnxruntime.modeling_diffusion import ORTPipelineForText2Video


wan_list = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "ali-vilab/text-to-video-ms-1.7b",
    "Wan-AI/Wan2.2-Animate-14B-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
]

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

pipe = ORTPipelineForText2Video.from_pretrained(
    wan_list[2],
    provider=providers[0],  # Force GPU
    torch_dtype=torch.float16,
)
print("Loaded successfully on:", pipe.device)
# Ensure weights move to GPU

# prompt = "Spiderman is surfing"
# video_frames = pipe(prompt, num_inference_steps=25).frames
# print(video_frames.shape)
# import numpy as np
# # tensor: (1,16,256,256,3)
# video = video_frames[0]        # -> (16,256,256,3)
# # ensure numpy
# if hasattr(video, "cpu"):
#     video = video.cpu().numpy()
# frames = [frame for frame in video]  # list of 16 arrays (256,256,3)
# video_path = export_to_video(frames)

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(prompt=prompt, negative_prompt=negative_prompt, height=128, width=128, num_frames=2).frames[0]

export_to_video(output, "output.mp4", fps=15)
