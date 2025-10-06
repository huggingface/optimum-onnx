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

from optimum.exporters.utils import _get_submodels_and_export_configs
from optimum.utils.import_utils import is_transformers_version


if TYPE_CHECKING:
    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel


MODEL_TYPES_REQUIRING_POSITION_IDS = {
    "arcee",
    "codegen",
    "deepseek_v3",
    "cohere",
    "falcon",
    "gemma",
    "glm",
    "gpt2",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "gpt_oss",
    "gptj",
    "granite",
    "helium",
    "imagegpt",
    "internlm2",
    "llama",
    "mistral",
    "nemotron",
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
