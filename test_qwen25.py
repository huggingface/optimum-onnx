from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/DeepCaption-VLA-V2.0-7B", torch_dtype="auto", device_map="auto"
)

model.config.vision_config_hidden_size = model.config.vision_config.hidden_size
model.config.vision_config_in_channels = model.config.vision_config.in_channels
model.config.vision_config_in_chans = model.config.vision_config.in_chans

model_id = "prithivMLmods/DeepCaption-VLA-V2.0-7B"

print(model.config)

processor = AutoProcessor.from_pretrained("prithivMLmods/DeepCaption-VLA-V2.0-7B")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image with detailed attributes and properties."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
# inputs = inputs.to("cuda")

# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

module_arch_fields = {
    "transformer": ["hidden_size", "vision_config_hidden_size", "vision_config_in_channels", "vision_config_in_chans", "vocab_size"],
}

input_len = inputs["input_ids"].shape[-1]
inf_kwargs = dict(inputs)

for key, value in inf_kwargs.items():
    print(key, value.shape)
    
model = ORTModelForSeq2SeqLM.from_pretrained(
    model_id, inf_kwargs=inf_kwargs, export_by_inference=True, module_arch_fields=module_arch_fields, torch_dtype=torch.bfloat16,
    skip_random_generation=True,
).eval()