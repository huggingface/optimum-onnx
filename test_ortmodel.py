from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModel

ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(ckpt)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

inf_kwargs = dict(encoded_input)
module_arch_fields = {
	"transformer": ["hidden_size", "intermediate_size", "max_position_embeddings", "vocab_size"]
}

model = ORTModel.from_pretrained(
	ckpt, 
	inf_kwargs=inf_kwargs, 
	export_by_inference=True, 
	module_arch_fields=module_arch_fields, 
	torch_dtype=torch.bfloat16,
	skip_random_generation=False).eval()

output = model(**encoded_input)

print(output)