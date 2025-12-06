from transformers import AutoTokenizer, UMT5EncoderModel
import torch

# 1. Load tokenizer & model
model_name = "google/umt5-small"   # or another UMT5 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = UMT5EncoderModel.from_pretrained(model_name)

# 2. Sample text
text = ["Hello world", "How are you?"]

# 3. Tokenize
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

print("=== Tokenizer Output Shapes ===")
print("input_ids:", inputs["input_ids"].shape)
print("attention_mask:", inputs["attention_mask"].shape)

# 4. Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# print("\n=== Model Output Shapes ===")
# print("last_hidden_state:", outputs.last_hidden_state.shape)   # (batch, seq_len, hidden_dim)

# # Optional: check embedding for first token
# print("\nSample embedding vector (first token of first sentence):")
# print(outputs.last_hidden_state[0, 0])

print("output keys: ", outputs.keys())