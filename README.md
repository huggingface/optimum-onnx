
<div align="center">

# 🤗 Optimum ONNX

**Export your Hugging Face models to ONNX**

[Documentation](https://huggingface.co/docs/optimum/index) | [ONNX](https://onnx.ai/) | [Hub](https://huggingface.co/onnx)

</div>

---

## Installation

Before you begin, make sure you have **Python 3.9 or higher** installed.

### 1. Create a virtual environment (recommended)
```
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.venv\Scripts\activate     # Windows
```

### 2. Install Optimum ONNX (CPU version)

```
pip install optimum-onnx[onnxruntime]
```

### 3. Install Optimum ONNX (GPU version)

Before installing, ensure your CUDA and cuDNN versions match [ONNX Runtime GPU requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

```
pip uninstall onnxruntime  # avoid conflicts
pip install optimum-onnx[onnxruntime-gpu]
```

---

## ONNX Export

It is possible to export 🤗 Transformers, Diffusers, Timm, and Sentence Transformers models to the [ONNX](https://onnx.ai/) format and perform graph optimization as well as quantization easily.

Example: Export **Llama 3.2–1B** to ONNX:

```
optimum-cli export onnx --model meta-llama/Llama-3.2-1B onnx_llama/
```

The model can also be optimized and quantized with `onnxruntime`.

### Additional Examples

**DistilBERT for text classification**

```
optimum-cli export onnx --model distilbert-base-uncased-finetuned-sst-2-english distilbert_onnx/
```

**Whisper for speech-to-text**

```
optimum-cli export onnx --model openai/whisper-small whisper_onnx/
```

**Gemma for general-purpose LLM tasks**

```
optimum-cli export onnx --model google/gemma-2b gemma_onnx/
```

For more information on the ONNX export, please check the [documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model).

---

## Inference

Once the model is exported to the ONNX format, we provide Python classes enabling you to run the exported ONNX model seamlessly using [ONNX Runtime](https://onnxruntime.ai/) in the backend.

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

---

## Troubleshooting

**1. `ModuleNotFoundError: No module named 'onnxruntime'`**
Ensure you have installed either `onnxruntime` (CPU) or `onnxruntime-gpu` (GPU):

```
pip install "optimum-onnx[onnxruntime]"      # CPU
pip install "optimum-onnx[onnxruntime-gpu]"  # GPU
```

---

**2. CUDA/cuDNN not found**
Verify your `nvcc --version` output matches ONNX Runtime GPU requirements.
Install the correct CUDA and cuDNN versions before retrying.

---

**3. Out-of-memory errors**
Use smaller models (e.g., `distilbert-base-uncased`) or enable model quantization:

```
optimum-cli export onnx --model distilbert-base-uncased --quantize int8 distilbert_quant/
```

---

**4. `onnxruntime` and `onnxruntime-gpu` conflict**
Uninstall the CPU version before installing the GPU version:

```
pip uninstall onnxruntime
```

---
