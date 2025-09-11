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
"""Custom input generators for ONNX export configs."""

from typing import Optional, cast

import torch

from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    DummyTextInputGenerator,
    NormalizedTextConfig,
)


class Gemma3DummyInputGenerator(DummyTextInputGenerator):
    """Dummy input generator for Gemma3."""

    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "vision_outputs",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
        num_choices: int = DEFAULT_DUMMY_SHAPES["num_choices"],
        random_batch_size_range: Optional[tuple[int, int]] = None,
        random_sequence_length_range: Optional[tuple[int, int]] = None,
        random_num_choices_range: Optional[tuple[int, int]] = None,
        padding_side: str = "right",
        **kwargs,
    ):
        super().__init__(
            task,
            normalized_config,
            batch_size,
            sequence_length,
            num_choices,
            random_batch_size_range,
            random_sequence_length_range,
            random_num_choices_range,
            padding_side,
            **kwargs,
        )

        # Gemma3 default image size
        self.height = self.width = 896
        self.n_channels = 3
        self.padding = "left"
        self.image_token_index = int(self.normalized_config.image_token_index)
        self.mm_tokens_per_image = int(self.normalized_config.mm_tokens_per_image)
        self.boi_token_index = self.normalized_config.boi_token_index
        self.eoi_token_index = self.normalized_config.eoi_token_index
        self.image_size = self.normalized_config.vision_config.image_size
        self.patch_size = self.normalized_config.vision_config.patch_size

    def _generate_pixel_values(
        self,
        framework: str,
        float_dtype: str,
    ):
        """Generate random pixel values."""
        shape = [self.batch_size, self.n_channels, self.height, self.width]
        min_value = -1
        # See Gemma3ImageProcessor's `rescale_factor`.
        max_value = 1 / 255
        return self.random_float_tensor(
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            framework=framework,
            dtype=float_dtype,
        )

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        if input_name == "pixel_values":
            # Vision encoder input
            return self._generate_pixel_values(framework, float_dtype)

        elif input_name == "vision_outputs":
            # Multimodal projector input
            patches_per_image = int(self.image_size // self.patch_size) ** 2
            shape = [self.batch_size, patches_per_image, self.sequence_length]
            return self.random_float_tensor(shape=shape, framework=framework, dtype=float_dtype)

        elif input_name == "image_features":
            shape = [
                self.batch_size,
                self.mm_tokens_per_image,
                self.normalized_config.text_config.hidden_size,
            ]
            return self.random_float_tensor(
                shape=shape,
                framework=framework,
                dtype=float_dtype,
            )

        generated_inputs = super().generate(input_name, framework, int_dtype, float_dtype)
        image_size = (self.batch_size, self.mm_tokens_per_image)
        if input_name == "input_ids":
            # Add image tokens corresponding to mm_tokens_per_image's per image
            if framework == "pt":
                input_ids = cast(torch.Tensor, generated_inputs)
                image_tokens = torch.full(
                    size=image_size,
                    fill_value=self.image_token_index,
                    dtype=input_ids.dtype,
                )
                return torch.cat((image_tokens, input_ids), dim=1)
            elif framework == "tf":
                import tensorflow as tf

                input_ids = cast(tf.Tensor, generated_inputs)

                image_tokens = tf.fill(
                    dims=image_size,
                    value=self.image_token_index,
                )
                return tf.concat((image_tokens, generated_inputs), axis=1)
            elif framework == "np":
                import numpy as np

                input_ids = cast(np.ndarray, generated_inputs)
                image_tokens = np.full(
                    shape=image_size,
                    fill_value=self.image_token_index,
                    dtype=input_ids.dtype,
                )
                return np.concatenate((image_tokens, generated_inputs), axis=1)

        elif input_name == "attention_mask":
            # Add attention mask for image tokens
            if framework == "pt":
                attention_mask = cast(torch.Tensor, generated_inputs)
                image_attention_mask = torch.ones(
                    size=image_size,
                    dtype=attention_mask.dtype,
                )
                if self.padding == "right":
                    return torch.cat((attention_mask, image_attention_mask), dim=1)
                else:
                    return torch.cat((image_attention_mask, attention_mask), dim=1)
            elif framework == "tf":
                import tensorflow as tf

                attention_mask = cast(tf.Tensor, generated_inputs)

                image_attention_mask = tf.ones(
                    image_size,
                    dtype=attention_mask.dtype,
                )
                if self.padding == "right":
                    return tf.concat((attention_mask, image_attention_mask), axis=1)
                else:
                    return tf.concat((image_attention_mask, attention_mask), axis=1)
            elif framework == "np":
                import numpy as np

                attention_mask = cast(np.ndarray, generated_inputs)

                image_attention_mask = np.ones(
                    image_size,
                    dtype=generated_inputs.dtype,
                )
                if self.padding == "right":
                    return np.concatenate((attention_mask, image_attention_mask), axis=1)
                else:
                    return np.concatenate((image_attention_mask, attention_mask), axis=1)

        else:
            raise ValueError(f"Input name {input_name} not supported.")
