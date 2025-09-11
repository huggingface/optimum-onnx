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
import copy
import gc

import requests
import torch
from parameterized import parameterized
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, set_seed
from transformers.configuration_utils import PretrainedConfig
from transformers.models import GenerationConfig

from optimum.exporters.onnx.model_patcher import patch_update_causal_mask
from optimum.onnxruntime.modeling_vlm import (
    MODEL_PARTS_CLS_MAPPING,
    MODEL_TYPE_TO_CLS_MAPPING,
    ORTModelForVisualCausalLM,
)
from optimum.utils import is_transformers_version
from tests.onnxruntime.testing_utils import MODEL_NAMES, ORTModelTestMixin


SEED = 42
TEST_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


class ORTModelForImageTextToTextIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = []

    if is_transformers_version(">", "4.52.0"):
        SUPPORTED_ARCHITECTURES += ["gemma3"]

    TASK = "image-text-to-text"

    IMAGE = Image.open(
        requests.get(
            TEST_IMAGE_URL,
            stream=True,
        ).raw
    )

    def get_transformer_model_class(self, model_arch):
        if is_transformers_version(">=", "4.52.0") and model_arch in [
            "gemma3",
        ]:
            from transformers import Gemma3ForConditionalGeneration

            return Gemma3ForConditionalGeneration

        return AutoModelForCausalLM

    def get_preprocessors(self, model_arch: str) -> dict:
        model_id = MODEL_NAMES[model_arch]
        config = AutoConfig.from_pretrained(model_id)

        return {
            "processor": AutoProcessor.from_pretrained(model_id),
            "tokenizer": None,
            "config": config,
        }

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch: str) -> None:
        prompt = "What is shown in this image?"
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)

        transformers_model = self.get_transformer_model_class(model_arch).from_pretrained(model_id).eval()
        preprocessors = self.get_preprocessors(model_arch)
        ort_model = ORTModelForVisualCausalLM.from_pretrained(model_id, export=True)
        self.assertIsInstance(ort_model, MODEL_TYPE_TO_CLS_MAPPING[ort_model.config.model_type])
        for component_name, component in ort_model.components.items():
            self.assertIsInstance(component, MODEL_PARTS_CLS_MAPPING[component_name])
        self.assertIsInstance(ort_model.config, PretrainedConfig)

        inputs = ort_model.preprocess_inputs(**preprocessors, text=prompt, image=self.IMAGE.resize((600, 600)))
        transformers_inputs = copy.deepcopy(inputs)

        # Check logits
        set_seed(SEED)
        ov_outputs = ort_model(**inputs)
        set_seed(SEED)
        with torch.no_grad():
            transformers_outputs = transformers_model(**transformers_inputs)
        self.assertTrue(
            torch.allclose(ov_outputs.logits, transformers_outputs.logits, atol=4e-3),
            f"Max abs diff {(torch.abs(ov_outputs.logits - transformers_outputs.logits).max())}",
        )

        additional_inputs = {}

        # gemma3 does not support dynamic cache, it is unfair to compare dynamic cache result vs hybrid cache,
        # align cache representation in torch model
        if model_arch == "gemma3":
            patch_update_causal_mask(
                (transformers_model if is_transformers_version("<", "4.52.0") else transformers_model.language_model),
                "4.43.0",
            )
            transformers_model._supports_cache_class = True
            transformers_model.generation_config.cache_implementation = None
            from transformers.cache_utils import DynamicCache

            additional_inputs = {"past_key_values": DynamicCache()}

        # Compare generation
        gen_config = GenerationConfig(
            max_new_tokens=30,
            min_new_tokens=30,
            do_sample=False,
            eos_token_id=None,
        )
        set_seed(SEED)
        onnx_outputs = ort_model.generate(**inputs, generation_config=gen_config, **additional_inputs)
        set_seed(SEED)
        with torch.no_grad():
            torch_outputs = transformers_model.generate(
                **transformers_inputs,
                generation_config=gen_config,
                **additional_inputs,
            )
        torch.testing.assert_close(onnx_outputs, torch_outputs, atol=self.ATOL, rtol=self.RTOL)

        del ort_model
        del transformers_model
        gc.collect()
