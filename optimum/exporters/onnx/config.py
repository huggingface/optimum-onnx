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
"""Common ONNX configuration classes that handle most of the features for building model specific configurations."""

from __future__ import annotations

import enum
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from optimum.exporters.onnx.base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from optimum.exporters.onnx.constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from optimum.exporters.onnx.model_patcher import ModelPatcher
from optimum.exporters.tasks import TasksManager
from optimum.utils import (
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    is_diffusers_available,
    logging,
)


# TODO : moved back onnx imports applied in https://github.com/huggingface/optimum/pull/2114/files after refactorization


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin

logger = logging.get_logger(__name__)


class TextEncoderOnnxConfig(OnnxConfig):
    """Handles encoder-based text architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)


class TextDecoderOnnxConfig(OnnxConfigWithPast):
    """Handles decoder-based text architectures."""

    PAD_ATTENTION_MASK_TO_PAST = True
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        preprocessors: list[Any] | None = None,
        legacy: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            preprocessors=preprocessors,
            legacy=legacy,
        )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.use_past_in_inputs:
            common_inputs = {"input_ids": {0: "batch_size", 1: "sequence_length"}}
            common_inputs["attention_mask"] = {0: "batch_size", 1: "past_sequence_length + sequence_length"}
            self.add_past_key_values(common_inputs, direction="inputs")
        else:
            common_inputs = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }
        return common_inputs

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.is_merged is False:
            common_outputs = super().outputs
        else:
            # in the merged case, we need to allow the `sequence_length` to be variable, as it is not 1
            # during the first pass without past key values
            common_outputs = OrderedDict({"logits": {0: "batch_size", 1: "sequence_length"}})
            self.add_past_key_values(common_outputs, direction="outputs")
        return common_outputs

    def post_process_exported_models(
        self,
        path: Path,
        models_and_onnx_configs: dict[str, tuple[PreTrainedModel | ModelMixin, OnnxConfig]],
        onnx_files_subpaths: list[str],
    ):
        models_and_onnx_configs, onnx_files_subpaths = super().post_process_exported_models(
            path, models_and_onnx_configs, onnx_files_subpaths
        )

        # Attempt to merge only if the decoder-only was exported separately without/with past
        if self.use_past is True and len(models_and_onnx_configs) == 2:
            from optimum.onnx import merge_decoders

            decoder_path = Path(path, onnx_files_subpaths[0])
            decoder_with_past_path = Path(path, onnx_files_subpaths[1])
            decoder_merged_path = Path(path, ONNX_DECODER_MERGED_NAME + ".onnx")
            try:
                merge_decoders(
                    decoder=decoder_path,
                    decoder_with_past=decoder_with_past_path,
                    save_path=decoder_merged_path,
                )
            except Exception as e:
                raise RuntimeError("Unable to merge decoders") from e

            # In order to do the validation of the two branches on the same file
            onnx_files_subpaths = [decoder_merged_path.name, decoder_merged_path.name]

            # We validate the two branches of the decoder model then
            models_and_onnx_configs[ONNX_DECODER_NAME][1].is_merged = True
            models_and_onnx_configs[ONNX_DECODER_NAME][1].use_cache_branch = False

            # Past key values won't be generated by default, but added in the input
            models_and_onnx_configs[ONNX_DECODER_NAME][1].use_past_in_inputs = True

            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1].use_cache_branch = True
            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1].is_merged = True

        return models_and_onnx_configs, onnx_files_subpaths


class TextDecoderWithPositionIdsOnnxConfig(TextDecoderOnnxConfig):
    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = super().inputs

        # Decoders based on GPT2 require a position_ids input to avoid generating wrong position_ids in the model itself:
        # https://github.com/huggingface/transformers/blob/v4.33.1/src/transformers/models/gpt2/modeling_gpt2.py#L802
        if not self.legacy and self.task in {"text-generation", "feature-extraction"}:
            common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs


class TextSeq2SeqOnnxConfig(OnnxSeq2SeqConfigWithPast):
    """Handles encoder-decoder-based text architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )

    @property
    def torch_to_onnx_input_map(self) -> dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_ids"] = {0: "batch_size", 1: "encoder_sequence_length"}

        common_inputs["attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                # TODO: validate the axis name for attention_mask
                # common_inputs["attention_mask"][1] = "past_encoder_sequence_length + sequence_length"
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
                self.add_past_key_values(common_inputs, direction="inputs")
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        return common_inputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> list[DummyInputGenerator]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](
            self.task,
            self._normalized_config,
            **kwargs,
        )
        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2](
            self.task,
            self._normalized_config,
            encoder_sequence_length=dummy_text_input_generator.sequence_length,
            **kwargs,
        )
        dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

        return dummy_inputs_generators


class VisionOnnxConfig(OnnxConfig):
    """Handles vision architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)


class TextAndVisionOnnxConfig(OnnxConfig):
    """Handles multi-modal text and vision architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyVisionInputGenerator, DummyBboxInputGenerator)


class AudioOnnxConfig(OnnxConfig):
    """Handles audio architectures."""

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {"input_values": {0: "batch_size", 1: "sequence_length"}}


class AudioToTextOnnxConfig(OnnxSeq2SeqConfigWithPast):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyAudioInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {}

        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_features"] = {0: "batch_size", 1: "feature_size", 2: "encoder_sequence_length"}

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
                self.add_past_key_values(common_inputs, direction="inputs")
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        return common_inputs

    @property
    def torch_to_onnx_input_map(self) -> dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}


class EncoderDecoderBaseOnnxConfig(OnnxSeq2SeqConfigWithPast):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        behavior: ConfigBehavior = ConfigBehavior.MONOLITH,
        preprocessors: list[Any] | None = None,
        legacy: bool = False,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            behavior=behavior,
            preprocessors=preprocessors,
            legacy=legacy,
        )

        from optimum.exporters.tasks import TasksManager

        self.is_decoder_with_past = False

        # Set up the encoder ONNX config.
        encoder_onnx_config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            task="feature-extraction",
            model_type=config.encoder.model_type,
            library_name="transformers",
        )
        self._encoder_onnx_config = encoder_onnx_config_constructor(
            config.encoder, int_dtype=int_dtype, float_dtype=float_dtype, preprocessors=preprocessors
        )
        self._normalized_config.ENCODER_NORMALIZED_CONFIG_CLASS = self._encoder_onnx_config._normalized_config

        # Set up the decoder ONNX config.
        decoder_onnx_config_constructor = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            task="feature-extraction",
            model_type=config.decoder.model_type,
            library_name="transformers",
        )
        kwargs = {}
        if issubclass(decoder_onnx_config_constructor.func, OnnxConfigWithPast):
            self.is_decoder_with_past = True
            kwargs["use_past"] = use_past
        else:
            self.use_past = False

        if use_past and not self.is_decoder_with_past:
            raise ValueError(
                f"The decoder part of the encoder-decoder model is {config.decoder.model_type} which does not need "
                "past key values."
            )

        self._decoder_onnx_config = decoder_onnx_config_constructor(
            config.decoder, int_dtype=int_dtype, float_dtype=float_dtype, preprocessors=preprocessors, **kwargs
        )
        if issubclass(decoder_onnx_config_constructor.func, OnnxSeq2SeqConfigWithPast):
            self._decoder_onnx_config = self._decoder_onnx_config.with_behavior(
                self._behavior, use_past=kwargs["use_past"], use_past_in_inputs=use_past_in_inputs
            )

        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS = self._decoder_onnx_config._normalized_config
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS = self._decoder_onnx_config._normalized_config
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.encoder_num_attention_heads = (
            self._decoder_onnx_config._normalized_config.num_attention_heads
        )
        self._normalized_config.DECODER_NORMALIZED_CONFIG_CLASS.decoder_num_attention_heads = (
            self._decoder_onnx_config._normalized_config.num_attention_heads
        )

        if isinstance(self._decoder_onnx_config, OnnxSeq2SeqConfigWithPast):
            self._past_key_values_generator = (
                DummySeq2SeqDecoderTextInputGenerator,
                DummySeq2SeqPastKeyValuesGenerator,
            )
        else:
            self._past_key_values_generator = (
                DummySeq2SeqDecoderTextInputGenerator,
                DummyPastKeyValuesGenerator,
            )

        self.DUMMY_INPUT_GENERATOR_CLASSES += self._past_key_values_generator

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        common_inputs = {}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs["input_ids"] = {0: "batch_size", 1: "encoder_sequence_length"}

        common_inputs["attention_mask"] = {0: "batch_size", 1: "encoder_sequence_length"}

        if self._behavior is not ConfigBehavior.ENCODER:
            # TODO: it is likely this pop() is unwanted as we then always hit
            # https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/t5/modeling_t5.py#L965-L969
            common_inputs.pop("attention_mask")

            if self.use_past_in_inputs:
                # TODO: validate the axis name for attention_mask
                # common_inputs["attention_mask"][1] = "past_encoder_sequence_length + sequence_length"
                common_inputs["decoder_input_ids"] = {0: "batch_size"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch_size", 1: "decoder_sequence_length"}

            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction="inputs")

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["encoder_outputs"] = {0: "batch_size", 1: "encoder_sequence_length"}

        return common_inputs

    @property
    def torch_to_onnx_input_map(self) -> dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "encoder_outputs": "encoder_hidden_states",
                "attention_mask": "encoder_attention_mask",
            }
        return {}

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        if self.is_decoder_with_past:
            return self._decoder_onnx_config.add_past_key_values(inputs_or_outputs, direction)

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        if self.is_decoder_with_past:
            return self._decoder_onnx_config.flatten_past_key_values(flattened_output, name, idx, t)

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> dict[str, Any]:
        return self._decoder_onnx_config.flatten_output_collection_property(name, field)

    def generate_dummy_inputs_for_validation(
        self, reference_model_inputs: dict[str, Any], onnx_input_names: list[str] | None = None
    ) -> dict[str, Any]:
        if self._behavior is ConfigBehavior.ENCODER:
            return self._encoder_onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)
        else:
            if self._behavior is ConfigBehavior.DECODER:
                reference_model_inputs["input_ids"] = reference_model_inputs.pop("decoder_input_ids")

            if "encoder_outputs" in reference_model_inputs:
                if "encoder_hidden_states" in onnx_input_names:
                    reference_model_inputs["encoder_hidden_states"] = reference_model_inputs.pop("encoder_outputs")[0]
                else:
                    reference_model_inputs.pop("encoder_outputs")

            return self._decoder_onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)

    def post_process_exported_models(
        self,
        path: Path,
        models_and_onnx_configs: dict[str, tuple[PreTrainedModel | ModelMixin, OnnxConfig]],
        onnx_files_subpaths: list[str],
    ):
        models_and_onnx_configs, onnx_files_subpaths = super().post_process_exported_models(
            path, models_and_onnx_configs, onnx_files_subpaths
        )
        if self.use_past is True and len(models_and_onnx_configs) == 3:
            models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.is_merged = True
            models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.use_cache_branch = False

            # Past key values won't be generated by default, but added in the input
            models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.use_past_in_inputs = True

            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1]._decoder_onnx_config.use_cache_branch = True
            models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1]._decoder_onnx_config.is_merged = True

        return models_and_onnx_configs, onnx_files_subpaths


class VLMConfigBehavior(str, enum.Enum):
    """Specifies the behavior of the [`~exporters.onnx.base.VLMDecoderOnnxConfig`].

    - MONOLITH: the config can be used to export the entire multimodal model as a single file.
    - VISION_ENCODER: the config can be used to export the underlying vision encoder.
    - MULTIMODAL_PROJECTOR: the config can be used to export the underlying multimodal projector.
    - TEXT_ENCODER: the config can be used to export the underlying text encoder, mapping inputs ids to embeddings.
    - LANGUAGE_MODEL: the config can be used to export the underlying language model. Note: this does not
        include the language model head.
    """

    MONOLITH = "monolith"
    VISION_ENCODER = "vision_encoder"
    MULTIMODAL_PROJECTOR = "multimodal_projector"
    TEXT_ENCODER = "text_encoder"
    LANGUAGE_MODEL = "language_model"
    LANGUAGE_MODEL_WITH_HEAD = "language_model_with_head"


class VLMDecoderOnnxConfig(TextDecoderOnnxConfig):
    """Base config for decoder-based vision language models."""

    DUMMY_INPUT_GENERATOR_CLASSES = TextAndVisionOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    SUPPORTED_BEHAVIORS: ClassVar[list[VLMConfigBehavior]] = list(VLMConfigBehavior)

    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "feature-extraction",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        use_past: bool = False,
        use_past_in_inputs: bool = False,
        preprocessors: list[Any] | None = None,
        legacy: bool = False,
        behavior: VLMConfigBehavior = VLMConfigBehavior.MONOLITH,
    ):
        super().__init__(
            config=config,
            task=task,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            use_past=use_past,
            use_past_in_inputs=use_past_in_inputs,
            preprocessors=preprocessors,
            legacy=legacy,
        )
        self._behavior = behavior

    @property
    def behavior(self) -> VLMConfigBehavior:
        """The behavior property."""
        return self._behavior

    @behavior.setter
    def behavior(self, value: str | VLMConfigBehavior) -> None:
        if isinstance(value, str):
            try:
                value = VLMConfigBehavior(value)
            except ValueError:
                raise ValueError(
                    f"behavior must be one of {self.SUPPORTED_BEHAVIORS}, but got {value} instead."
                ) from None

        self._behavior = value

    def get_supported_behaviors(self, task: str) -> list[VLMConfigBehavior]:
        """Get supported behaviors for this model.

        The supported behaviors are task-dependent. For instance, "text-generation" is handled by
        the language model and associated head.
        """
        if "image-text-to-text" in task:
            # All parts of the model
            return [
                VLMConfigBehavior.VISION_ENCODER,
                VLMConfigBehavior.MULTIMODAL_PROJECTOR,
                VLMConfigBehavior.TEXT_ENCODER,
                VLMConfigBehavior.LANGUAGE_MODEL_WITH_HEAD,
            ]

        elif "text-generation" in task:
            # Only text-related components needed
            return [
                VLMConfigBehavior.TEXT_ENCODER,
                VLMConfigBehavior.LANGUAGE_MODEL_WITH_HEAD,
            ]

        elif "feature-extraction" in task:
            # Same as image-text-to-text but without the LM head
            return [
                VLMConfigBehavior.VISION_ENCODER,
                VLMConfigBehavior.MULTIMODAL_PROJECTOR,
                VLMConfigBehavior.TEXT_ENCODER,
                VLMConfigBehavior.LANGUAGE_MODEL,
            ]

        else:
            message = f"Invalid task for {self.__class__.__name__}: {task}"
            raise ValueError(message)

    def with_behavior(self, behavior: VLMConfigBehavior) -> Self:
        if behavior == VLMConfigBehavior.LANGUAGE_MODEL_WITH_HEAD:
            model_config = self._config.text_config
            model_type = model_config.model_type

            if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
                raise ValueError(
                    f"Unsupported language model type provided `{model_type}`. Please define custom export config"
                )

            lm_task = "text-generation-with-past" if self.use_past else "text-generation"
            exporter_config_constructor = TasksManager.get_exporter_config_constructor(
                exporter="onnx",
                model_type=model_type,
                task=lm_task,
            )
            return exporter_config_constructor(
                model_config,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                use_past=self.use_past,
                use_past_in_inputs=self.use_past_in_inputs,
            )

        elif behavior in [
            VLMConfigBehavior.MONOLITH,
            VLMConfigBehavior.TEXT_ENCODER,
            VLMConfigBehavior.VISION_ENCODER,
            VLMConfigBehavior.MULTIMODAL_PROJECTOR,
            VLMConfigBehavior.LANGUAGE_MODEL,
        ]:
            # TODO: check if we need to handle vision encoder part similarly, with config.vision_config
            return type(self)(
                config=self._config,
                task=self.task,
                int_dtype=self.int_dtype,
                float_dtype=self.float_dtype,
                use_past=self.use_past,
                use_past_in_inputs=self.use_past_in_inputs,
                preprocessors=self._preprocessors,
                legacy=self.legacy,
                behavior=behavior,
            )

        message = f"Behavior must be one of {self.SUPPORTED_BEHAVIORS}, but got {behavior} instead."
        raise ValueError(message)

    def get_model_for_behavior(self, model: PreTrainedModel, behavior: VLMConfigBehavior):
        if behavior != self.behavior:
            raise ValueError(
                f"Config behavior {self.behavior} does not match the requested behavior {behavior}. Please run `.with_behavior` first."
            )

        if behavior == VLMConfigBehavior.LANGUAGE_MODEL:
            return model.language_model

        if behavior == VLMConfigBehavior.LANGUAGE_MODEL_WITH_HEAD:
            # No default way to get just the LM and LM head, so we get entire model.
            return model

        if behavior == VLMConfigBehavior.VISION_ENCODER:
            vision_encoder = model.vision_tower
            vision_encoder.config = model.config.vision_config
            return vision_encoder

        if behavior == VLMConfigBehavior.MULTIMODAL_PROJECTOR:
            multi_modal_projector = model.multi_modal_projector
            # TODO: check if multimodal projector actually acceps the base config, not config.vision_config
            multi_modal_projector.config = model.config
            return multi_modal_projector

        if behavior == VLMConfigBehavior.MONOLITH:
            return model

        if behavior == VLMConfigBehavior.TEXT_ENCODER:
            return model.get_input_embeddings()

        message = f"Behavior must be one of {self.SUPPORTED_BEHAVIORS}, but got {behavior} instead."
        raise ValueError(message)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        if self.behavior == VLMConfigBehavior.VISION_ENCODER:
            return {"pixel_values": {0: "batch_size"}}

        if self.behavior == VLMConfigBehavior.MULTIMODAL_PROJECTOR:
            # Should be batch_size, number of tokens per image, and hidden size of the vision encoder
            return {
                "vision_outputs": {
                    0: "batch_size",
                    1: "num_patch_tokens",
                    2: "hidden_size",
                }
            }

        if self.behavior in (
            VLMConfigBehavior.LANGUAGE_MODEL,
            VLMConfigBehavior.LANGUAGE_MODEL_WITH_HEAD,
        ):
            return super().inputs

        if self.behavior == VLMConfigBehavior.MONOLITH:
            inputs = super().inputs

            # text-generation task should not include images.
            if "image-text-to-text" in self.task:
                # No need to add channel and image dimensions
                inputs["pixel_values"] = {0: "batch_size"}

            return inputs

        if self.behavior == VLMConfigBehavior.TEXT_ENCODER:
            return super().inputs

        message = f"Behavior must be one of {self.SUPPORTED_BEHAVIORS}, but got {self.behavior} instead."
        raise ValueError(message)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        if self.behavior in (
            VLMConfigBehavior.VISION_ENCODER,
            VLMConfigBehavior.LANGUAGE_MODEL,
        ):
            return {"last_hidden_state": {0: "batch_size"}}

        if self.behavior == VLMConfigBehavior.MULTIMODAL_PROJECTOR:
            return {
                "image_features": {
                    0: "batch_size",
                    1: "mm_tokens_per_image",
                    2: "text_hidden_size",
                }
            }

        if self.behavior == VLMConfigBehavior.TEXT_ENCODER:
            return {"inputs_embeds": {0: "batch_size", 1: "sequence_length"}}

        return super().outputs

    def patch_model_for_export(
        self, model: PreTrainedModel, model_kwargs: dict[str, Any] | None = None
    ) -> ModelPatcher:
        if self.behavior in (
            VLMConfigBehavior.LANGUAGE_MODEL,
            VLMConfigBehavior.LANGUAGE_MODEL_WITH_HEAD,
            VLMConfigBehavior.MONOLITH,
        ):
            if model_kwargs is None:
                model_kwargs = {}
            model_kwargs["use_cache"] = self.use_past

        return super().patch_model_for_export(model=model, model_kwargs=model_kwargs)
