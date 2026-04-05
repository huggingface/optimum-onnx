from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

ckpt = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

# inference
input_ids = tokenizer(
    "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1

inf_kwargs = {"input_ids": input_ids}
module_arch_fields = {
    "transformer": ["d_model", "vocab_size"]
}

# not work
model = ORTModelForSeq2SeqLM.from_pretrained(
    ckpt, 
    inf_kwargs=inf_kwargs,
    export_by_inference=True,
    module_arch_fields=module_arch_fields,
    skip_random_generation=True,
    export=True,
)

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# studies have shown that owning a dog is good for you.

