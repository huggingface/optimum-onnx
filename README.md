<div align="center">

<img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/onnxruntime/logo/optimum-onnx.png" width=80%>

# 🤗 Optimum ONNX

**Export your Hugging Face models to ONNX**

[Documentation](https://huggingface.co/docs/optimum/index) | [ONNX](https://onnx.ai/) | [Hub](https://huggingface.co/onnx)

</div>


### Installation

Before you begin, make sure you have all the necessary libraries installed :

```bash
pip install git+https://github.com/huggingface/optimum-onnx.git
```

### ONNX export

It is possible to export 🤗 Transformers, Diffusers, Timm and Sentence Transformers models to the [ONNX](https://onnx.ai/) format and perform graph optimization as well as quantization easily:

```bash
optimum-cli export onnx --model meta-llama/Llama-3.2-1B onnx_llama/
```
The model can also be optimized and quantized with `onnxruntime`.

For more information on the ONNX export, please check the [documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model).

#### Inference

Once the model is exported to the ONNX format, we provide Python classes enabling you to run the exported ONNX model in a seemless manner using [ONNX Runtime](https://onnxruntime.ai/) in the backend:


```diff

  from transformers import AutoTokenizer, pipeline
- from transformers import AutoModelForCausalLM
+ from optimum.onnxruntime import ORTModelForCausalLM

- model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B") # PyTorch checkpoint
+ model = ORTModelForCausalLM.from_pretrained("onnx-community/Llama-3.2-1B", subfolder="onnx") # ONNX checkpoint
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
  result = pipe("He never went out without a book under his arm")
```

More details on how to run ONNX models with `ORTModelForXXX` classes [here](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models).
