# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models import GenerationConfig

from optimum.onnxruntime.base import ORTSessionMixin
from optimum.onnxruntime.modeling_decoder import ORTModelForCausalLM

from optimum.onnxruntime.modeling_ort import ORTModel


logger = logging.getLogger(__name__)


# TODO: to be implemented
class ORTVisionEncoder(ORTSessionMixin):
    pass


class ORTMultiModalProjector(ORTSessionMixin):
    pass


class ORTModelForVisualCausalLM(ORTModel):
    def __init__(
        self,
        language_model_with_head,
        text_embeddings,
        vision_embeddings,
        multimodal_projector,
        config: PretrainedConfig,
        device: str = "CPU",
        dynamic_shapes: bool | None = None,
        model_save_dir: Union[str, Path, TemporaryDirectory] | None = None,
        **kwargs,
    ):
        if dynamic_shapes is not None:
            logger.warning(
                f"`dynamic_shapes` was set to {dynamic_shapes}, but this value will be ignored as only dynamic shapes are supported."
            )

        self.is_dynamic = True
        self.config = config
        self.use_cache = kwargs.get("use_cache", True)
        self._model_save_dir = model_save_dir
        self._device = device.upper()

        self.preprocessors = kwargs.get("preprocessors", [])
        self.vision_embeddings_model = vision_embeddings
        self.multimodal_projector = multimodal_projector
        self.text_embeddings_model = text_embeddings
        self.lm_model = language_model_with_head

        self.language_model = ORTModelForCausalLM(
            config=config,
            device=device,
            ov_config=ov_config,
            model_save_dir=model_save_dir,
            quantization_config=quantization_config,
            compile=self._compile_only or enable_compilation,
            compile_only=self._compile_only,
        )
        self.vision_embeddings = OVVisionEmbedding(self.vision_embeddings_model, self)

        self.main_input_name = "input_ids"
        self.generation_config = kwargs.get("generation_config", GenerationConfig.from_model_config(config))
        for part in self.additional_parts:
            model_part = getattr(self, f"{part}_model", None)
            if model_part is not None:
                model_part = MODEL_PARTS_CLS_MAPPING[part](model_part, self)
            setattr(self, part, model_part)

        if enable_compilation and not self._compile_only:
            self.compile()

        # Avoid warnings when creating a transformers pipeline
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        self.auto_model_class.register(AutoConfig, self.__class__)


class _ORTGemma3ForCausalLM(ORTModelForVisualCausalLM):
    def get_vision_embeddings(self, pixel_values, input_ids=None, **kwargs):
        if input_ids is not None and input_ids.shape[1] == 1:
            return None
        return self.vision_embeddings(pixel_values).last_hidden_state

    def merge_vision_text_embeddings(
        self,
        vision_embeds,
        inputs_embeds,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        # Adopted from https://github.com/huggingface/transformers/blob/v4.49.0-Gemma-3/src/transformers/models/gemma3/modeling_gemma3.py#L1323-L1339
        image_features = torch.from_numpy(vision_embeds) if isinstance(vision_embeds, np.ndarray) else vision_embeds
        inputs_embeds = torch.from_numpy(inputs_embeds) if isinstance(inputs_embeds, np.ndarray) else inputs_embeds
        if input_ids is None:
            special_image_mask = inputs_embeds == torch.from_numpy(
                self.get_text_embeddings(torch.tensor([[self.config.image_token_index]], dtype=torch.long))[0]
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds)

            image_features = image_features.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds, attention_mask, position_ids

    @staticmethod
    def preprocess_inputs(
        text: str,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = processor(images=image, text=text_prompt, videos=video, return_tensors="pt")
        return inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        # Token type ids used only for first inference mask generation
        model_kwargs.pop("token_type_ids", None)

        return model_kwargs


MODEL_PARTS_CLS_MAPPING = {
    "vision_encoder": ORTVisionEncoder,
    "multimodal_projector": ORTMultiModalProjector,
    "language_model_with_head": ORTModelForCausalLM,
    "vision_language_model": ORTModelForVisualCausalLM,
}
MODEL_TYPE_TO_CLS_MAPPING = {
    "gemma3": _ORTGemma3ForCausalLM,
}
