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

import copy
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers.utils import is_torch_available

from optimum.exporters.base import ExporterConfig
from optimum.exporters.tasks import TasksManager
from optimum.exporters.utils import _get_submodels_and_export_configs
from optimum.utils.import_utils import is_diffusers_available, is_transformers_version


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


_DIFFUSERS_CLASS_NAME_TO_SUBMODEL_TYPE = {
    "CLIPTextModel": "clip-text",
    "CLIPTextModelWithProjection": "clip-text-with-projection",
    "FluxTransformer2DModel": "flux-transformer-2d",
    "SD3Transformer2DModel": "sd3-transformer-2d",
    "UNet2DConditionModel": "unet-2d-condition",
    "T5EncoderModel": "t5-encoder",
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


def get_diffusion_models_for_export(
    pipeline: DiffusionPipeline,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    exporter: str = "onnx",
) -> dict[str, tuple[torch.nn.Module, ExporterConfig]]:
    """Return diffusion submodels and ONNX configs across Transformers 4 and 5.

    Transformers 5 flattened the CLIP text model and removed its ``text_model``
    container. Keeping component discovery in the ONNX integration avoids
    relying on either internal layout.
    """
    models_for_export = {}
    is_sdxl = pipeline.__class__.__name__.startswith("StableDiffusionXL")
    is_sd3 = pipeline.__class__.__name__.startswith("StableDiffusion3")

    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        submodel = getattr(pipeline, name, None)
        if submodel is None:
            continue
        if name != "text_encoder_3" and (is_sdxl or is_sd3):
            submodel.config.output_hidden_states = True
            nested_text_model = getattr(submodel, "text_model", None)
            if nested_text_model is not None:
                nested_text_model.config.output_hidden_states = True
        submodel.config.export_model_type = _DIFFUSERS_CLASS_NAME_TO_SUBMODEL_TYPE.get(submodel.__class__.__name__)
        models_for_export[name] = submodel

    unet = getattr(pipeline, "unet", None)
    if unet is not None:
        unet.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
        unet.config.time_cond_proj_dim = getattr(unet.config, "time_cond_proj_dim", None)
        unet.config.text_encoder_projection_dim = (
            pipeline.text_encoder_2.config.projection_dim if is_sdxl else pipeline.text_encoder.config.projection_dim
        )
        unet.config.export_model_type = _DIFFUSERS_CLASS_NAME_TO_SUBMODEL_TYPE.get(unet.__class__.__name__)
        models_for_export["unet"] = unet

    transformer = getattr(pipeline, "transformer", None)
    if transformer is not None:
        transformer.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
        transformer.config.time_cond_proj_dim = getattr(transformer.config, "time_cond_proj_dim", None)
        transformer.config.text_encoder_projection_dim = pipeline.text_encoder.config.projection_dim
        transformer.config.export_model_type = _DIFFUSERS_CLASS_NAME_TO_SUBMODEL_TYPE.get(
            transformer.__class__.__name__
        )
        models_for_export["transformer"] = transformer

    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_parameters": vae_encoder.encode(x=sample)["latent_dist"].parameters}
    models_for_export["vae_encoder"] = vae_encoder

    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    models_for_export["vae_decoder"] = vae_decoder

    tasks = {
        "text_encoder": ("feature-extraction", None),
        "text_encoder_2": ("feature-extraction", None),
        "text_encoder_3": ("feature-extraction", None),
        "unet": ("semantic-segmentation", None),
        "transformer": ("semantic-segmentation", None),
        "vae_encoder": ("semantic-segmentation", "vae-encoder"),
        "vae_decoder": ("semantic-segmentation", "vae-decoder"),
    }
    for name, submodel in models_for_export.items():
        task, model_type = tasks[name]
        config_constructor = TasksManager.get_exporter_config_constructor(
            model=submodel,
            exporter=exporter,
            library_name="diffusers",
            task=task,
            model_type=model_type,
        )
        export_config = config_constructor(submodel.config, int_dtype=int_dtype, float_dtype=float_dtype)
        models_for_export[name] = (submodel, export_config)

    return models_for_export


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

    if library_name == "diffusers":
        return None, get_diffusion_models_for_export(model, int_dtype, float_dtype)

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
