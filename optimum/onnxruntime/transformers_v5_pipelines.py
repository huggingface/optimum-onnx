# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Pipelines removed in Transformers 5 but still supported by Optimum ORT models."""

from __future__ import annotations

from enum import Enum

from transformers import AutoModelForSeq2SeqLM, GenerationConfig
from transformers.image_utils import load_image
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import Pipeline
from transformers.tokenization_utils_base import TruncationStrategy


class _ReturnType(Enum):
    TENSORS = 0
    TEXT = 1


class ORTText2TextGenerationPipeline(Pipeline):
    """Transformers 4 compatible text-to-text generation pipeline."""

    _pipeline_calls_generate = True
    _load_processor = False
    _load_image_processor = False
    _load_video_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True
    _default_generation_config = GenerationConfig(max_new_tokens=256, num_beams=4)
    return_name = "generated"

    def _sanitize_parameters(
        self,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        truncation=None,
        stop_sequence=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        if return_tensors is not None and return_type is None:
            return_type = _ReturnType.TENSORS if return_tensors else _ReturnType.TEXT
        postprocess_params = {}
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]
        return preprocess_params, generate_kwargs, postprocess_params

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if (
            args
            and isinstance(args[0], list)
            and all(isinstance(item, str) for item in args[0])
            and all(len(item) == 1 for item in result)
        ):
            return [item[0] for item in result]
        return result

    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        prefix = self.prefix if self.prefix is not None else ""
        if isinstance(inputs, list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("The tokenizer must define pad_token_id for batch inputs.")
            inputs = [prefix + item for item in inputs]
            padding = True
        elif isinstance(inputs, str):
            inputs = prefix + inputs
            padding = False
        else:
            raise TypeError("Text-to-text pipeline inputs must be a string or a list of strings.")
        model_inputs = self.tokenizer(inputs, padding=padding, truncation=truncation, return_tensors="pt")
        model_inputs.pop("token_type_ids", None)
        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_batch = model_inputs["input_ids"].shape[0]
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        output_batch = output_ids.shape[0]
        output_ids = output_ids.reshape(input_batch, output_batch // input_batch, *output_ids.shape[1:])
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, return_type=_ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            if return_type == _ReturnType.TENSORS:
                records.append({f"{self.return_name}_token_ids": output_ids})
            else:
                records.append(
                    {
                        f"{self.return_name}_text": self.tokenizer.decode(
                            output_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                        )
                    }
                )
        return records


class ORTSummarizationPipeline(ORTText2TextGenerationPipeline):
    pass


class ORTTranslationPipeline(ORTText2TextGenerationPipeline):
    pass


class ORTImageToTextPipeline(Pipeline):
    """Transformers 4 compatible image-to-text generation pipeline."""

    _pipeline_calls_generate = True
    _load_processor = False
    _load_image_processor = True
    _load_video_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True
    _default_generation_config = GenerationConfig(max_new_tokens=256)

    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, prompt=None, timeout=None):
        preprocess_params = {}
        if prompt is not None:
            preprocess_params["prompt"] = prompt
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        forward_params = dict(generate_kwargs or {})
        if max_new_tokens is not None:
            if "max_new_tokens" in forward_params:
                raise ValueError("max_new_tokens was provided twice.")
            forward_params["max_new_tokens"] = max_new_tokens
        return preprocess_params, forward_params, {}

    def __call__(self, inputs=None, **kwargs):
        inputs = kwargs.pop("images", inputs)
        if inputs is None:
            raise ValueError("Cannot call the image-to-text pipeline without an inputs argument.")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, image, prompt=None, timeout=None):
        image = load_image(image, timeout=timeout)
        model_inputs = self.image_processor(images=image, return_tensors="pt")
        if prompt is not None:
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string.")
            text_inputs = self.tokenizer(prompt, return_tensors="pt")
            model_inputs.update(text_inputs)
        if self.model.config.model_type == "git" and prompt is None:
            model_inputs["input_ids"] = None
        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config
        inputs = model_inputs.pop(self.model.main_input_name)
        return self.model.generate(inputs, **model_inputs, **generate_kwargs)

    def postprocess(self, model_outputs):
        return [
            {"generated_text": self.tokenizer.decode(output_ids, skip_special_tokens=True)}
            for output_ids in model_outputs
        ]


def register_transformers_v5_ort_pipelines(defaults: dict[str, tuple[str, str]]) -> None:
    """Register only tasks that Transformers 5 no longer provides."""
    task_classes = {
        "text2text-generation": ORTText2TextGenerationPipeline,
        "summarization": ORTSummarizationPipeline,
        "translation": ORTTranslationPipeline,
        "image-to-text": ORTImageToTextPipeline,
    }
    for task, pipeline_class in task_classes.items():
        if task not in PIPELINE_REGISTRY.supported_tasks:
            PIPELINE_REGISTRY.register_pipeline(
                task,
                pipeline_class=pipeline_class,
                pt_model=AutoModelForSeq2SeqLM,
                default={"model": defaults[task]},
                type="text" if task != "image-to-text" else "multimodal",
            )
            # These are built-in compatibility pipelines from Optimum's point of
            # view. Do not serialize them as Hub custom code on save_pretrained().
            del pipeline_class._registered_impl
