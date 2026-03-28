from transformers import GPT2Tokenizer
from transformers import ORTModelForCausalLM

ckpt = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(ckpt)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

inf_kwargs = dict(encoded_input)
module_arch_fields = {
	"transformer": ["n_ctx", "n_embd"]
}

model = ORTModelForCausalLM.from_pretrained(
	ckpt, 
	inf_kwargs=inf_kwargs,
	export_by_inference=True,
	module_arch_fields=module_arch_fields,
	torch_dtype=torch.bfloat16,
	skip_random_generation=False,
).eval()

output_ids = model.generate(**encoded_input)
text_output = tokenizer.decode(output_ids[0])
print(text_output)
