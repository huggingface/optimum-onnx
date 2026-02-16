# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers.utils import is_torch_available

from optimum.exporters.base import ExporterConfig
from optimum.exporters.tasks import TasksManager
from optimum.exporters.utils import _get_submodels_and_export_configs
from optimum.utils.import_utils import is_diffusers_available, is_transformers_version
from optimum.exporters.onnx.model_configs import DummyOnnxConfig


if TYPE_CHECKING:
    if is_diffusers_available():
        from diffusers import DiffusionPipeline


if TYPE_CHECKING:
    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel


MODEL_TYPES_REQUIRING_POSITION_IDS = {
    "arcee",
    "codegen",
    "deepseek_v3",
    "cohere",
    "falcon",
    "glm",
    "gpt2",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "granite",
    "helium",
    "imagegpt",
    "internlm2",
    "llama",
    "mistral",
    "phi",
    "phi3",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "smollm3",
    "stablelm",
    "olmo2",
    "olmo",
}


if is_transformers_version(">=", "4.46.0"):
    MODEL_TYPES_REQUIRING_POSITION_IDS.add("opt")


def recursive_to_device(value: tuple | list | torch.Tensor, device: str):
    if isinstance(value, tuple):
        value = list(value)
        for i, val in enumerate(value):
            value[i] = recursive_to_device(val, device)
        value = tuple(value)
    elif isinstance(value, list):
        for i, val in enumerate(value):
            value[i] = recursive_to_device(val, device)
    elif isinstance(value, torch.Tensor):
        value = value.to(device)

    return value


def recursive_to_dtype(
    value: tuple | list | torch.Tensor, dtype: torch.dtype | None, start_dtype: torch.dtype | None = None
):
    if dtype is None:
        return value

    if isinstance(value, tuple):
        value = list(value)
        for i, val in enumerate(value):
            value[i] = recursive_to_dtype(val, dtype)
        value = tuple(value)
    elif isinstance(value, list):
        for i, val in enumerate(value):
            value[i] = recursive_to_dtype(val, dtype)
    elif isinstance(value, torch.Tensor):
        if start_dtype is None or (start_dtype is not None and value.dtype == start_dtype):
            value = value.to(dtype=dtype)

    return value


# Copied from https://github.com/microsoft/onnxruntime/issues/7846#issuecomment-850217402
class PickableInferenceSession:  # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, sess_options, providers):
        import onnxruntime as ort

        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers
        self.sess = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)

    def run(self, *args):
        return self.sess.run(*args)

    def get_outputs(self):
        return self.sess.get_outputs()

    def get_inputs(self):
        return self.sess.get_inputs()

    def __getstate__(self):
        return {"model_path": self.model_path}

    def __setstate__(self, values):
        import onnxruntime as ort

        self.model_path = values["model_path"]
        self.sess = ort.InferenceSession(self.model_path, sess_options=self.sess_options, providers=self.providers)


def _get_submodels_for_export_metaclip_2(model, variant):
    models_for_export = {}

    if variant == "monolith":
        models_for_export["model"] = model
    else:
        # We rather use the model patcher to patch their forward method.
        models_for_export["vision_model"] = model
        models_for_export["text_model"] = model

    return models_for_export


def get_metaclip_2_models_for_export(model: PreTrainedModel, config: ExporterConfig):
    models_for_export = _get_submodels_for_export_metaclip_2(model, config.variant)

    if config.variant == "monolith":
        export_config = config.__class__(model.config, task=config.task, variant=config.variant)
        models_for_export["model"] = (models_for_export["model"], export_config)
    else:
        vision_model_export_config = config.__class__(
            model.config, task=config.task, variant=config.variant, vision_model=True
        )
        text_model_export_config = config.__class__(
            model.config, task=config.task, variant=config.variant, vision_model=False
        )
        models_for_export["vision_model"] = (models_for_export["vision_model"], vision_model_export_config)
        models_for_export["text_model"] = (models_for_export["text_model"], text_model_export_config)

    return models_for_export


def get_sana_models_for_export(pipeline: DiffusionPipeline, int_dtype: str = "int64", float_dtype: str = "fp32"):
    import copy

    models_for_export = {}
    text_encoder = pipeline.text_encoder
    text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=text_encoder,
        exporter="onnx",
        library_name="diffusers",
        task="feature-extraction",
        model_type="gemma2-text-encoder",
    )
    text_encoder_export_config = text_encoder_config_constructor(
        pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["text_encoder"] = (text_encoder, text_encoder_export_config)

    transformer = pipeline.transformer
    transformer.config.vocab_size = pipeline.text_encoder.config.vocab_size
    transformer.config.text_encoder_projection_dim = transformer.config.caption_channels
    transformer.config.requires_aesthetics_score = False
    transformer.config.time_cond_proj_dim = None
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=transformer,
        exporter="onnx",
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="sana-transformer",
    )
    transformer_export_config = export_config_constructor(
        pipeline.transformer.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["transformer"] = (transformer, transformer_export_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample).latent}
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter="onnx",
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="dcae-encoder",
    )
    vae_encoder_export_config = vae_config_constructor(
        vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["vae_encoder"] = (vae_encoder, vae_encoder_export_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter="onnx",
        library_name="diffusers",
        task="semantic-segmentation",
        model_type="dcae-decoder",
    )
    vae_decoder_export_config = vae_config_constructor(
        vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype
    )
    models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_export_config)

    return models_for_export

def get_dynamic_models_for_export(
    pipeline: DiffusionPipeline,
    models_and_inputs: dict | None = None,
    models_and_outputs: dict | None = None,
    int_dtype: str = "int64", 
    float_dtype: str = "fp32"
):
    import copy
    config_dim = {"in_channels": 16, "d_model": 4096}

    models_for_export = {}
    text_encoder = pipeline.text_encoder
    text_encoder_config = DummyOnnxConfig(config=text_encoder.config, 
                                          task="text-encoding", 
                                          preprocessors=None, 
                                          int_dtype=int_dtype,
                                          float_dtype=float_dtype,
                                          model_inputs=models_and_inputs["text_encoder"],
                                          model_outputs=models_and_outputs["text_encoder"],
                                          config_dim=config_dim)
    models_for_export["text_encoder"] = (text_encoder, text_encoder_config) 

    transformer = pipeline.transformer
    transformer_config = DummyOnnxConfig(config=transformer.config, 
                                          task="backbone", 
                                          preprocessors=None, 
                                          int_dtype=int_dtype,
                                          float_dtype=float_dtype,
                                          model_inputs=models_and_inputs["transformer"],
                                          model_outputs=models_and_outputs["transformer"],
                                          config_dim=config_dim)
    models_for_export["transformer"] = (transformer, transformer_config)

    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    vae_decoder_config = DummyOnnxConfig(config=vae_decoder.config, 
                                          task="latent_decode", 
                                          preprocessors=None, 
                                          int_dtype=int_dtype,
                                          float_dtype=float_dtype,
                                          model_inputs=models_and_inputs["vae_decoder"],
                                          model_outputs=models_and_outputs["vae_decoder"],
                                          config_dim=config_dim)
    models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_config)
    return models_for_export

def _get_submodels_and_onnx_configs(
    model: PreTrainedModel,
    task: str,
    monolith: bool,
    custom_onnx_configs: dict,
    custom_architecture: bool,
    _variant: str,
    library_name: str,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    fn_get_submodels: Callable | None = None,
    preprocessors: list[Any] | None = None,
    model_kwargs: dict | None = None,
    models_and_inputs: dict | None = None,
    models_and_outputs: dict | None = None,
):
    if library_name == "transformers" and model.config.model_type == "metaclip_2":
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="onnx", task=task, library_name="transformers"
        )
        export_config = export_config_constructor(
            model.config,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        export_config.variant = _variant
        return export_config, get_metaclip_2_models_for_export(model, export_config)

    if library_name == "diffusers" and model.__class__.__name__.startswith("Sana"):
        return None, get_sana_models_for_export(model, int_dtype, float_dtype)

    ## 
    if library_name == "diffusers" and models_and_inputs is not None and models_and_outputs is not None:
        return None, get_dynamic_models_for_export(model, models_and_inputs, models_and_outputs, int_dtype, float_dtype)

    return _get_submodels_and_export_configs(
        model,
        task,
        monolith,
        custom_onnx_configs,
        custom_architecture,
        _variant,
        library_name,
        int_dtype,
        float_dtype,
        fn_get_submodels,
        preprocessors,
        model_kwargs,
        exporter="onnx",
    )

def make_positional_hook(dummy_inputs, module_name):
    import inspect
    def hook(module, args):
        sig = inspect.signature(module.forward)
        # Bind positional args to real parameter names
        bound = sig.bind_partial(*args)

        named_shapes = {}
        for name, value in bound.arguments.items():
            if torch.is_tensor(value):
                named_shapes[name] = tuple(value.shape)

        dummy_inputs[module_name] = named_shapes
        return None  # do not modify inputs
    return hook

def get_output_name_and_shape(output, name):
    from dataclasses import fields, is_dataclass

    named_shapes = {}
    if torch.is_tensor(output):
        named_shapes[name] = tuple(output.shape)
    elif is_dataclass(output):
        for f in fields(output):
            val = getattr(output, f.name)
            if torch.is_tensor(val):
                named_shapes[f.name] = tuple(val.shape)
    elif isinstance(output, (tuple, list)):
        for i, x in enumerate(output):
            if torch.is_tensor(x):
                named_shapes[f"{name}_{i}"] = tuple(x.shape)
    elif isinstance(output, dict):
        for k, v in output.items():
            if torch.is_tensor(v):
                named_shapes[k] = tuple(v.shape)
    return named_shapes
    

def make_dataclass_output_hook(dummy_outputs, module_name):
    def hook(module, args, output):
        dummy_outputs[module_name] = get_output_name_and_shape(output, "sample")
        return None  # don't modify output
    return hook

def _get_submodels_and_tensors_(
    model: PreTrainedModel | DiffusionPipeline,
    inf_kwargs: dict[str, Any] | None = None):
    # key: module_name, value: {input_name: tensor_shape}
    dummy_inputs = {}
    dummy_outputs = {}

    import torch.nn as nn
    for name, module in model.components.items():
        if isinstance(module, nn.Module):
            dummy_inputs[name] = {}
            dummy_outputs[name] = {}

    if "text_encoder" in dummy_inputs.keys():
        model.text_encoder.register_forward_pre_hook(
            make_positional_hook(dummy_inputs, "text_encoder")
        )
        model.text_encoder.register_forward_hook(
            make_dataclass_output_hook(dummy_outputs, "text_encoder")
        )

    if "transformer" in dummy_inputs.keys():
        original_forward = model.transformer.forward
        def wrapped_forward(*args, **kwargs):
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    dummy_inputs["transformer"][key] = tuple(value.shape)
            return original_forward(*args, **kwargs)
        
        model.transformer.forward = wrapped_forward
        model.transformer.register_forward_hook(
            make_dataclass_output_hook(dummy_outputs, "transformer")
        )

    if "vae" in dummy_inputs.keys():
        import inspect
        import types

        dummy_inputs["vae_encoder"] = {}
        dummy_inputs["vae_decoder"] = {}
        # hook encoder
        wrap_encode = model.vae.encode
        orig_encode = None
        for cell in wrap_encode.__closure__:
            if inspect.isfunction(cell.cell_contents):
                orig_decode = cell.cell_contents
                break
        if orig_encode is None:
            sig = None
        else:
            sig = inspect.signature(orig_encode)
        def hooked_encode(self, *args, **kwargs):
            if sig is not None:
                bound = sig.bind_partial(self, *args, **kwargs)
                for name, value in bound.arguments.items():
                    if torch.is_tensor(value):
                        dummy_inputs["vae_encoder"][name] = tuple(value.shape)
            output = wrap_encode(*args, **kwargs)
            dummy_output["vae_encoder"] = get_output_name_and_shape(output, "latent_dist")
            return output
        model.vae.encode = types.MethodType(hooked_encode, model.vae)

        wrap_decode = model.vae.decode
        orig_decode = None
        for cell in wrap_decode.__closure__:
            if inspect.isfunction(cell.cell_contents):
                orig_decode = cell.cell_contents
                break
        if orig_decode is None:
            sig = None
        else:
            sig = inspect.signature(orig_decode)
        def hooked_decode(self, *args, **kwargs):
            if sig is not None:
                bound = sig.bind_partial(self, *args, **kwargs)
                for name, value in bound.arguments.items():
                    if torch.is_tensor(value):
                        dummy_inputs["vae_decoder"][name] = tuple(value.shape)
            output = wrap_decode(*args, **kwargs)
            dummy_outputs["vae_decoder"] = get_output_name_and_shape(output, "sample")
            return output
        model.vae.decode = types.MethodType(hooked_decode, model.vae)

    output = model(**inf_kwargs).frames[0]  # yes, we can inference

    filtered_inputs = {k: v for k, v in dummy_inputs.items() if v}
    filtered_outputs = {k: v for k, v in dummy_outputs.items() if v}

    print("dummy_inputs: ", filtered_inputs)
    print("dummy_outputs: ", filtered_outputs)
    
    return filtered_inputs, filtered_outputs
    
    

