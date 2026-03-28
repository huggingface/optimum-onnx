from transformers import AutoTokenizer
from transformers import ORTModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = ORTModelForCausalLM.from_pretrained("google-t5/t5-small")

# inference
input_ids = tokenizer(
    "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1

inf_kwargs = {"input_ids": input_ids}
module_arch_fields = {
    "transformer": ["d_model", "vocab_size"]
}

model = ORTModelForCausalLM.from_pretrained(
    ckpt, 
    inf_kwargs=inf_kwargs,
    export_by_inference=True,
    module_arch_fields=module_arch_fields,
    torch_dtype=torch.bfloat16,
    skip_random_generation=False,
).eval()

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# studies have shown that owning a dog is good for you.

