# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from PIL import Image
import requests
import torch

from huggingface_hub import login

ckpt = "google/gemma-3-4b-it"
processor = AutoProcessor.from_pretrained(ckpt)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/password.jpg"},
            {"type": "text", "text": "What is the password?"}
        ]
    }
]

module_arch_fields = {
    "transformer": ["in_channels", "text_dim"],
}

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
)

input_len = inputs["input_ids"].shape[-1]
inf_kwargs = dict(inputs)

for key, value in inf_kwargs.items():
    print(key, value.shape)
    
model = ORTModelForSeq2SeqLM.from_pretrained(
    ckpt, inf_kwargs=inf_kwargs, export_by_inference = True, module_arch_fields=module_arch_fields, torch_dtype=torch.bfloat16,
    skip_random_generation=True,
).eval()

inputs = inputs.to(model.device, dtype=torch.bfloat16)
# input_len = inputs["input_ids"].shape[-1]

# with torch.inference_mode():
#     generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]

# decoded = processor.decode(generation, skip_special_tokens=True)
# print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene, 
# focusing on a cluster of pink cosmos flowers and a busy bumblebee. 
# It has a slightly soft, natural feel, likely captured in daylight.
