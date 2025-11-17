import torch

from optimum.onnxruntime.modeling_diffusion import ORTDiffusionPipeline


pipe = ORTDiffusionPipeline.from_pretrained("Efficient-Large-Model/Sana_600M_512px_diffusers", variant="fp16")
prompt = 'a cyberpunk cat in neon style'
image = pipe(
    prompt=prompt,
    generator=torch.Generator(device="cpu").manual_seed(142),
)[0]

image[0].save("sana_fp16.png")