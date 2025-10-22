#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Classes handling causal-lm related architectures in ONNX Runtime."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast

from onnxruntime import InferenceSession, SessionOptions
from optimum.exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime.constants import (
    DECODER_MERGED_ONNX_FILE_PATTERN,
    DECODER_ONNX_FILE_PATTERN,
    DECODER_WITH_PAST_ONNX_FILE_PATTERN,
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_FILE_PATTERN,
    ONNX_WEIGHTS_NAME,
)
from optimum.onnxruntime.modeling import ONNX_MODEL_END_DOCSTRING, ORTModel
from optimum.onnxruntime.utils import prepare_providers_and_provider_options
from optimum.utils import is_transformers_version
from optimum.utils.file_utils import find_files_matching_pattern
from optimum.utils.save_utils import maybe_save_preprocessors


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if is_transformers_version(">=", "4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin  # type: ignore


logger = logging.getLogger(__name__)

DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

CAUSALLM_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

_TOKENIZER_FOR_DOC = "AutoTokenizer"

TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    >>> text = "My name is Arthur and I live in"
    >>> gen = onnx_gen(text)
    ```
"""


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForCausalLM(ORTModel, GenerationMixin):
    """ONNX model with a causal language modeling head for ONNX Runtime inference. This class officially supports bloom, codegen, falcon, gpt2, gpt-bigcode, gpt_neo, gpt_neox, gptj, llama."""

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"
    _supports_cache_class = False
    _is_stateful = False

    def __init__(
        self,
        *,
        config: PretrainedConfig = None,
        session: InferenceSession = None,
        use_io_binding: bool | None = None,
        generation_config: GenerationConfig | None = None,
        model_save_dir: str | Path | TemporaryDirectory | None = None,
    ):
        super().__init__(config=config, session=session, use_io_binding=use_io_binding, model_save_dir=model_save_dir)

        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        self.can_use_cache = len(self.key_value_input_names) > 0 and len(self.key_value_output_names) > 0
        self.is_merged = "use_cache_branch" in self.input_names
        self.generation_config = generation_config

        # Reference: https://github.com/huggingface/optimum/pull/1381
        if self.config.model_type in MODEL_TYPES_REQUIRING_POSITION_IDS and "position_ids" not in self.input_names:
            logger.warning(
                f"ORTModelForCausalLM loaded a legacy ONNX model with no `position_ids` input, although the model type `{self.config.model_type}` requires it. "
                "We strongly encourage to re-export the model with Optimum-ONNX for better performance and more reliable text generation. "
                "To re-export your model, simply set `export=True` as in `from_pretrained(..., export=True, use_cache=True)`. "
                "Please note that support for legacy models will be removed in a future version of Optimum-ONNX."
            )

        if not self.can_use_cache and self.config.use_cache:
            logger.warning(
                "`model.config.use_cache=True` but the loaded model does not support using the past key values cache."
                "Please re-export the original model once again with `use_cache=True` to be able to use it during generation. "
                "To re-export your model, simply set `export=True` in `from_pretrained(... , export=True, use_cache=True)`."
            )

        self.old_bloom_modeling = self.config.model_type == "bloom" and (
            len(self.input_shapes.get("past_key_values.0.key", ())) == 3
            or len(self.output_shapes.get("past_key_values.0.key", ())) == 3
            or is_transformers_version("<", "4.44.0")
        )  # Old Bloom style
        if self.can_use_cache and self.old_bloom_modeling:
            logger.warning(
                "The loaded Bloom ONNX model uses an old cache format that squeezes the batch_size and num_key_value_heads dimensions into one. "
                "We strongly encourage to re-export the model with a newer version of Optimum and Transformers for better performance and more reliable generation. "
                "To re-export your model, simply set `export=True` as in `from_pretrained(..., export=True, use_cache=True)`."
            )

        self.old_gpt_bigcode_modeling = self.config.model_type == "gpt_bigcode" and (
            self.input_shapes.get("past_key_values.0.key_value", None) is not None
            or self.output_shapes.get("past_key_values.0.key_value", None) is not None
            or is_transformers_version("<", "4.54.0")
        )  # Old GPT BigCode style
        if self.can_use_cache and self.old_gpt_bigcode_modeling:
            logger.warning(
                "The loaded GPT BigCode ONNX model uses an old cache format that fuses keys and values in one tensor. "
                "We strongly encourage to re-export the model with a newer version of Optimum and Transformers for better performance and more reliable generation. "
                "To re-export your model, simply set `export=True` as in `from_pretrained(..., export=True, use_cache=True)`."
            )

        if self.config.model_type in {"gemma", "gemma3_text", "gpt_oss", "nemotron"}:
            self.embed_size_per_head = self.config.head_dim
        elif self.config.model_type == "gemma3":
            self.embed_size_per_head = self.config.text_config.head_dim
        elif self.old_gpt_bigcode_modeling:
            # (before v4.54) GPT BigCode fuses keys and values in one tensor, doubling the head dimension
            self.embed_size_per_head = self.config.hidden_size // self.config.num_attention_heads * 2
        elif self.config.model_type == "deepseek_v3":
            # For deepseek_v3, keys and values have different head dimensions
            self.qk_head_dim = self.config.qk_rope_head_dim + self.config.qk_nope_head_dim
            self.v_head_dim = self.config.v_head_dim
        else:
            self.embed_size_per_head = self.config.hidden_size // self.config.num_attention_heads

        if self.config.model_type in {
            "arcee",
            "deepseek_v3",
            "cohere",
            "gemma",
            "gemma3_text",
            "glm",
            "granite",
            "gpt_oss",
            "helium",
            "mistral",
            "llama",
            "nemotron",
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "smollm3",
            "stablelm",
        }:
            self.num_key_value_heads = self.config.num_key_value_heads
        elif self.config.model_type == "gemma3":
            self.num_key_value_heads = self.config.text_config.num_key_value_heads
        elif self.config.model_type == "falcon":
            if self.config.new_decoder_architecture or not self.config.multi_query:
                self.num_key_value_heads = self.config.num_kv_heads
            else:
                self.num_key_value_heads = 1
        elif self.config.model_type == "gpt_bigcode":
            if self.config.multi_query:
                self.num_key_value_heads = 1
            else:
                self.num_key_value_heads = self.config.num_attention_heads
        else:
            self.num_key_value_heads = self.config.num_attention_heads

    @property
    def use_cache(self):
        logger.warning(
            "The `ORTModelForCausalLM.use_cache` property is deprecated and will be removed in a future version. "
            "Please rather use `ORTModelForCausalLM.can_use_cache` to check if a model supports using cache during generation. "
            "And use `ORTModelForCausalLM.config.use_cache` to check if the model is configured to use cache during generation."
        )
        return self.can_use_cache

    @property
    def use_merged(self):
        logger.warning(
            "The `ORTModelForCausalLM.use_merged` property is deprecated and will be removed in a future version. "
            "Please rather use `ORTModelForCausalLM.is_merged` to check if the underlying model is merged or not."
        )
        return self.is_merged

    @add_start_docstrings_to_model_forward(
        CAUSALLM_ONNX_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForCausalLM",
            checkpoint="optimum/gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                "`use_cache=True` was passed to the model but the loaded model does not support pkv cache reuse. "
                "Please load your current model with `use_cache=True` or re-export the original model "
                "once again with `use_cache=True` when calling the `from_pretrained` method. "
                "To re-export your model, simply set `export=True` in `from_pretrained`."
            )

        # Get the input/output dimensions
        batch_size, in_seq_len = input_ids.shape
        if past_key_values is not None:
            if self.old_gpt_bigcode_modeling:
                # (before v4.54) GPT BigCode fuses keys and values in one tensor
                past_seq_len = past_key_values[0].shape[-2]
            else:
                # We use the past value and not key to be compatible with old bloom cache
                past_seq_len = past_key_values[0][1].shape[-2]
        else:
            past_seq_len = 0

        out_seq_len = past_seq_len + in_seq_len

        # Prepare position_ids
        if position_ids is None and "position_ids" in self.input_names:
            if self.config.model_type == "opt":
                if attention_mask is not None:
                    # OPT models use a different way to infer position_ids from attention_mask
                    position_ids = attention_mask.cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, -1)
                    position_ids = position_ids[:, past_seq_len:]
                else:
                    raise ValueError(
                        "The model OPT requires position_ids for batched generation but none were provided. "
                        "Please provide position_ids or attention_mask (from which position_ids can be inferred)."
                    )
            elif self.old_gpt_bigcode_modeling:
                if attention_mask is not None:
                    # GPT BigCode models use a different way to infer position_ids from attention_mask
                    position_ids = attention_mask.cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids[:, past_seq_len:]
                else:
                    raise ValueError(
                        "The model gpt_bigcode requires position_ids for batched generation but none were provided. "
                        "Please provide position_ids or attention_mask (from which position_ids can be inferred)."
                    )
            else:
                # Create position_ids from input_ids
                position_ids = (
                    torch.arange(past_seq_len, out_seq_len, dtype=torch.long, device=input_ids.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )

        use_cache_branch = None
        if self.is_merged:
            # NOTE: This needs to be evaluated here before creating dummy past_key_values in the following block
            # Uses cache branch of merged decoders depending on whether real past key values are passed
            use_cache_branch = torch.full((1,), past_key_values is not None, dtype=torch.bool, device=self.device)

        if len(self.key_value_input_names) > 0 and past_key_values is None:
            # Generates the input pkv for the first forward of the model (merged or with past)
            if self.old_bloom_modeling:
                # (before v4.44) Bloom squeezes the batch_size and num_key_value_heads dimensions into one
                k_shape = (batch_size * self.num_key_value_heads, self.embed_size_per_head, 0)
                v_shape = (batch_size * self.num_key_value_heads, 0, self.embed_size_per_head)
            elif self.old_gpt_bigcode_modeling and self.config.multi_query:
                # (before v4.54) When multi_query is True, GPT BigCode squeezes the num_key_value_heads dimension
                k_shape = v_shape = (batch_size, 0, self.embed_size_per_head)
            elif self.config.model_type == "deepseek_v3":
                # Deepseek V3 uses different head dimensions for keys and values
                k_shape = (batch_size, self.num_key_value_heads, 0, self.qk_head_dim)
                v_shape = (batch_size, self.num_key_value_heads, 0, self.v_head_dim)
            else:
                k_shape = v_shape = (batch_size, self.num_key_value_heads, 0, self.embed_size_per_head)
            k_tensor = torch.zeros(k_shape, dtype=self.dtype, device=self.device)
            v_tensor = torch.zeros(v_shape, dtype=self.dtype, device=self.device)
            past_key_values = tuple(k_tensor if ".key" in name else v_tensor for name in self.key_value_input_names)
        elif isinstance(past_key_values, tuple) and isinstance(past_key_values[0], tuple):
            # Flattens the past_key_values to a single tuple if it is a tuple of tuples
            past_key_values = sum(past_key_values, ())

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "use_cache_branch": use_cache_branch,
        }
        if len(self.key_value_input_names) > 0:
            model_inputs.update(zip(self.key_value_input_names, past_key_values))

        known_output_shapes = {}
        outputs_to_not_bind = set()
        if use_cache and self.use_io_binding:
            # Infers the shape of the output pkv
            batch_size, _ = input_ids.shape
            if self.old_bloom_modeling:
                num_key_value_heads_batch_size, embed_size_per_head = past_key_values[0].shape[:2]
                k_shape = (num_key_value_heads_batch_size, embed_size_per_head, out_seq_len)
                v_shape = (num_key_value_heads_batch_size, out_seq_len, embed_size_per_head)
            elif self.old_gpt_bigcode_modeling and self.config.multi_query:
                # (before v4.54) GPT BigCode squeezes the num_key_value_heads dimension when multi_query is True
                embed_size_per_head = past_key_values[0].shape[-1]
                k_shape = v_shape = (batch_size, out_seq_len, embed_size_per_head)
            elif self.config.model_type == "deepseek_v3":
                # Deepseek V3 uses different head dimensions for keys and values
                k_shape = (batch_size, self.num_key_value_heads, out_seq_len, self.qk_head_dim)
                v_shape = (batch_size, self.num_key_value_heads, out_seq_len, self.v_head_dim)
            else:
                k_shape = v_shape = (batch_size, self.num_key_value_heads, out_seq_len, self.embed_size_per_head)
            known_output_shapes = {
                name: k_shape if ".key" in name else v_shape for name in self.key_value_output_names
            }
        else:
            # we don't bind the key/values if they are not gonna be returned/used
            outputs_to_not_bind.update(self.key_value_output_names)

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, outputs_to_not_bind=outputs_to_not_bind, known_output_shapes=known_output_shapes
            )

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers.pop("logits").view(output_shapes["logits"])

            if use_cache:
                past_key_values = tuple(
                    output_buffers.pop(name).view(output_shapes[name]) for name in self.key_value_output_names
                )
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs.pop("logits")

            if use_cache:
                past_key_values = tuple(model_outputs.pop(name) for name in self.key_value_output_names)

        if use_cache:
            if self.old_gpt_bigcode_modeling:
                # `n_layers` fused key-value tensors
                past_key_values = past_key_values
            else:
                # `n_layers` tuples of key and value tensors
                past_key_values = tuple(past_key_values[i : i + 2] for i in range(0, len(past_key_values), 2))
        else:
            past_key_values = None

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if is_transformers_version("<", "4.46.0"):
            return self._prepare_inputs_for_generation_legacy(*args, **kwargs)
        else:
            return super().prepare_inputs_for_generation(*args, **kwargs)

    # Adapted from transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeForCausalLM.prepare_inputs_for_generation
    def _prepare_inputs_for_generation_legacy(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        use_cache=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if self.old_gpt_bigcode_modeling:
                # (before v4.54) GPT BigCode fuses keys and values in one tensor
                past_seq_len = past_key_values[0].shape[-2]
            else:
                # We use the past value and not key to be compatible with bloom cache
                past_seq_len = past_key_values[0][1].shape[-2]

            if input_ids.shape[1] > past_seq_len:
                remove_prefix_length = past_seq_len
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # falcon, gpt_bigcode, and other models used to override the prepare_inputs_for_generation method to add this logic
        # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L1186
        if "position_ids" in self.input_names and position_ids is None and attention_mask is not None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(
        past_key_values: tuple[tuple[torch.Tensor]] | tuple[torch.Tensor],
        beam_idx: torch.LongTensor,
    ) -> tuple[tuple[torch.Tensor]]:
        if (
            isinstance(past_key_values, tuple)
            and isinstance(past_key_values[0], tuple)
            and isinstance(past_key_values[0][0], torch.Tensor)
            and past_key_values[0][0].ndim == 3
        ):
            # Old Bloom style
            batch_size_times_num_heads, head_dim, seq_length = past_key_values[0][0].shape
            num_heads = batch_size_times_num_heads // beam_idx.shape[0]
            batch_size = beam_idx.shape[0]
            return tuple(
                (
                    layer_past[0]
                    .view(batch_size, num_heads, head_dim, seq_length)
                    .index_select(0, beam_idx.to(layer_past[0].device))
                    .view(batch_size_times_num_heads, head_dim, seq_length),
                    layer_past[1]
                    .view(batch_size, num_heads, seq_length, head_dim)
                    .index_select(0, beam_idx.to(layer_past[1].device))
                    .view(batch_size_times_num_heads, seq_length, head_dim),
                )
                for layer_past in past_key_values
            )
        elif (
            isinstance(past_key_values, tuple)
            and isinstance(past_key_values[0], tuple)
            and isinstance(past_key_values[0][0], torch.Tensor)
            and past_key_values[0][0].ndim == 4
        ):
            # GPT2 style
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                for layer_past in past_key_values
            )
        elif (
            isinstance(past_key_values, tuple)
            and isinstance(past_key_values[0], torch.Tensor)
            and past_key_values[0].ndim in {3, 4}
        ):
            # GPT BigCode style (Multi-Query=False/Multi-Query=True)
            return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)
        else:
            raise TypeError(
                f"Unexpected past_key_values: {past_key_values}. "
                "Expected tuple of tuples of 3D tensors (old Bloom style), "
                "tuple of tuples of 4D tensors (GPT2 style), or tuple of 3D tensors (GPT BigCode style)."
            )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str | Path,
        config: PretrainedConfig,
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: bool | str | None = None,
        # file options
        file_name: str | None = None,
        # session options
        provider: str = "CPUExecutionProvider",
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        session_options: SessionOptions | None = None,
        # inference options
        use_cache: bool = True,
        use_io_binding: bool | None = None,
        generation_config: GenerationConfig | None = None,
        dtype: torch.dtype = torch.float32,
        # other arguments
        model_save_dir: str | Path | TemporaryDirectory | None = None,
    ) -> ORTModelForCausalLM:
        onnx_files = find_files_matching_pattern(
            model_id,
            ONNX_FILE_PATTERN,
            glob_pattern="**/*.onnx",
            subfolder=subfolder,
            revision=revision,
            token=token,
        )
        if len(onnx_files) == 0:
            raise FileNotFoundError(f"Could not find any ONNX model file in {model_id}")
        if Path(model_id).is_dir():
            onnx_files = [f.relative_to(model_id) for f in onnx_files]

        file_path = cls._infer_file_path(
            ONNX_FILE_PATTERN,
            onnx_files=onnx_files,
            standard_file_name=ONNX_WEIGHTS_NAME,
            target_file_name=file_name,
        )

        # TODO: remove this block once legacy merged/unmerged models are no longer supported (definitely)
        # if the inferred file_path is neither the user-provided file_name nor the standard_file_name,
        # we try to infer the file path a second time, prioritizing merged models (kinda like seq2seq models)
        if file_path.name not in {file_name, ONNX_WEIGHTS_NAME}:
            # we disable logging to avoid meaningless warnings from _infer_file_path in this block
            original_logging_level = logger.level
            logger.setLevel(logging.ERROR)
            use_merged = False
            legacy = False
            try:
                file_path = cls._infer_file_path(
                    DECODER_MERGED_ONNX_FILE_PATTERN,
                    onnx_files=onnx_files,
                    standard_file_name=ONNX_DECODER_MERGED_NAME,
                    target_file_name=file_name,
                )
                use_merged = True
                legacy = True
            except FileNotFoundError:
                try:
                    # if use_merged is None or False, we try to load a non-merged model
                    file_path = cls._infer_file_path(
                        DECODER_WITH_PAST_ONNX_FILE_PATTERN if use_cache else DECODER_ONNX_FILE_PATTERN,
                        onnx_files=onnx_files,
                        standard_file_name=ONNX_DECODER_WITH_PAST_NAME if use_cache else ONNX_DECODER_NAME,
                        target_file_name=file_name,
                    )
                    legacy = True
                except FileNotFoundError:
                    # we keep the initially inferred file_path
                    pass
            logger.setLevel(original_logging_level)

            if legacy:
                logger.warning(
                    f"You didn't pass a `file_name` to `from_pretrained` and we have inferred a legacy {'merged' if use_merged else 'non-merged'} ONNX model in {file_path}. "
                    "We strongly encourage you to either specify `file_name` or re-export the model with a newer version of Optimum for better performance and more reliable generation. "
                    "To re-export your model, simply set `export=True` as in `from_pretrained(..., export=True, use_cache=True)`. "
                    "Please note that support for legacy models will be removed in a future version of Optimum-ONNX."
                )

        model_path = cls._cached_file(
            model_id,
            filename=file_path.name,
            subfolder=file_path.parent.as_posix(),
            force_download=force_download,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance,
        # in which case we want to keep it instead.
        if model_save_dir is None:
            model_save_dir = model_path.parent

        # Important: for encoder-decoder models used with CausalLM, we need to set the is_decoder flag to True
        # and the is_encoder_decoder flag to False. This is needed for the model to work correctly with generation logic.
        config.use_cache = use_cache
        if hasattr(config, "is_decoder"):
            config.is_decoder = True
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "onnxruntime"

        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except OSError:
                logger.info("Generation config file not found, creating a new one from model config.")
                generation_config = GenerationConfig.from_model_config(config)

        generation_config.use_cache = use_cache
        if hasattr(generation_config, "cache_implementation"):
            generation_config.cache_implementation = None

        if is_transformers_version(">=", "4.45.0"):
            misplaced_generation_parameters = config._get_non_default_generation_parameters()
            if len(misplaced_generation_parameters) > 0:
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(generation_config, param_name, param_value)
                    setattr(config, param_name, None)

        providers, provider_options = prepare_providers_and_provider_options(
            provider=provider, providers=providers, provider_options=provider_options
        )
        session = InferenceSession(
            model_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )

        return cls(
            config=config,
            session=session,
            use_io_binding=use_io_binding,
            generation_config=generation_config,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _export(
        cls,
        model_id: str | Path,
        config: PretrainedConfig,
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: bool | str | None = None,
        # inference options
        use_cache: bool = True,
        **kwargs,
    ) -> ORTModelForCausalLM:
        # this is guaranteed to work since we it uses a mapping from model classes to task names
        # instead of relying on the hub metadata or the model configuration
        task = TasksManager._infer_task_from_model_or_model_class(model_class=cls.auto_model_class)
        if use_cache:
            task += "-with-past"

        if kwargs.get("task") is not None:
            raise ValueError(
                f"The `task` argument is not needed when exporting a model with `{cls.__name__}`. "
                f"The `task` is automatically inferred from the class as `{task}`."
            )

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=False,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_cache=use_cache,
            model_save_dir=save_dir,
            **kwargs,
        )

    def _save_config(self, save_directory):
        """Save the model and generation configs to the specified directory.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the model and generation configs will be saved.
        """
        self.config.save_pretrained(save_directory)
        self.generation_config.save_pretrained(save_directory)

    def _prepare_cache_for_generation(self, *args, **kwargs):
        return
