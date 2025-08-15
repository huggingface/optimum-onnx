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
import os
import tempfile
import unittest
from typing import Optional

import torch
from onnxruntime import InferenceSession
from parameterized import parameterized
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, set_seed
from transformers.cache_utils import Cache
from transformers.generation import GenerationConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.config import TextDecoderWithPositionIdsOnnxConfig
from optimum.exporters.onnx.model_configs import (
    ArceeOnnxConfig,
    BloomOnnxConfig,
    GemmaOnnxConfig,
    GraniteOnnxConfig,
    InternLM2OnnxConfig,
    MPTOnnxConfig,
    Olmo2OnnxConfig,
    OlmoOnnxConfig,
    OPTOnnxConfig,
    Phi3OnnxConfig,
    PhiOnnxConfig,
    Qwen2OnnxConfig,
    Qwen3MoeOnnxConfig,
    Qwen3OnnxConfig,
    SmolLM3OnnxConfig,
)
from optimum.exporters.onnx.utils import MODEL_TYPES_REQUIRING_POSITION_IDS
from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime import (
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_WEIGHTS_NAME,
    ORTModelForCausalLM,
)
from optimum.pipelines import pipeline as optimum_pipeline
from optimum.utils.import_utils import is_transformers_version
from optimum.utils.logging import get_logger
from optimum.utils.testing_utils import grid_parameters, remove_directory, require_hf_token


logger = get_logger(__name__)


class ORTModelForCausalLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "bart",
        "bigbird_pegasus",
        "blenderbot-small",
        "blenderbot",
        "codegen",
        "falcon",
        "falcon-alibi-True",
        "gpt2",
        "gpt_bigcode",
        "gpt_bigcode-multi_query-False",
        "gpt_neo",
        "gpt_neox",
        "gptj",
        "llama",
        "marian",
        "mbart",
        "mistral",
        "pegasus",
    ]

    if is_transformers_version(">=", str(ArceeOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("arcee")
    if is_transformers_version(">=", str(OPTOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("opt")
    if is_transformers_version(">=", str(PhiOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("phi")
    if is_transformers_version(">=", str(BloomOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("bloom")
    if is_transformers_version(">=", str(OlmoOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("olmo")
    if is_transformers_version(">=", str(Olmo2OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("olmo2")
    if is_transformers_version(">=", str(Qwen2OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("qwen2")
    if is_transformers_version(">=", str(GemmaOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("gemma")
    if is_transformers_version(">=", str(MPTOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("mpt")
    if is_transformers_version(">=", str(GraniteOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("granite")
    if is_transformers_version(">=", str(Phi3OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("phi3")
    if is_transformers_version(">=", str(Qwen3OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("qwen3")
    if is_transformers_version(">=", str(Qwen3MoeOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("qwen3_moe")
    if is_transformers_version(">=", str(InternLM2OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("internlm2")
    if is_transformers_version(">=", str(SmolLM3OnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("smollm3")

    TRUST_REMOTE_CODE_MODELS = {"internlm2"}  # noqa: RUF012

    # base generation kwargs
    GEN_KWARGS = {  # noqa: RUF012
        "num_beams": 1,  # we test beam search in a separate test
        "do_sample": True,  # to avoid the model returning the same id repeatedly
        "max_new_tokens": 10,
        "min_new_tokens": 10,
    }

    TASK = "text-generation"
    ORTMODEL_CLASS = ORTModelForCausalLM
    AUTOMODEL_CLASS = AutoModelForCausalLM

    # UTILITIES
    def get_tokenizer(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            else:
                raise ValueError(
                    f"Tokenizer for model {model_id} does not have a defined `pad_token`, `eos_token`, or `bos_token`."
                )
        tokenizer.padding_side = "left"
        return tokenizer

    def get_inputs(
        self, model_arch: str, for_generation: bool = False, for_pipeline: bool = False, batched: bool = True
    ):
        if batched:
            texts = ["This is me", "Today is a nice day and I am longer"]
        else:
            texts = "Today is a nice day."

        if for_pipeline:
            return texts

        tokenizer = self.get_tokenizer(model_arch)
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        if for_generation and is_transformers_version(">=", "4.51.0"):
            inputs["use_model_defaults"] = False

        return inputs

    def mask_logits(self, logits, attention_mask):
        """Mask the logits based on the attention mask."""
        mask = attention_mask.unsqueeze(-1)
        logits.masked_fill_(mask == 0, 0)
        return logits

    def mask_past_key_values(self, onnx_model, past_key_values, attention_mask):
        """Mask the past key values based on the attention mask."""
        if onnx_model.config.model_type == "gpt_bigcode":
            if onnx_model.config.multi_query:
                mask = attention_mask.unsqueeze(-1)
            else:
                mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            for i in range(len(past_key_values)):
                past_key_values[i].masked_fill_(mask == 0, 0)
        elif onnx_model.config.model_type == "bloom" and onnx_model.old_bloom_modeling:
            num_key_value_heads = onnx_model.num_key_value_heads
            key_mask = attention_mask.repeat_interleave(num_key_value_heads, dim=0).unsqueeze(1)
            value_mask = attention_mask.repeat_interleave(num_key_value_heads, dim=0).unsqueeze(-1)
            for i in range(len(past_key_values)):
                past_key_values[i][0].masked_fill_(key_mask == 0, 0)
                past_key_values[i][1].masked_fill_(value_mask == 0, 0)
        else:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            for i in range(len(past_key_values)):
                past_key_values[i][0].masked_fill_(mask == 0, 0)
                past_key_values[i][1].masked_fill_(mask == 0, 0)

    def check_onnx_model_attributes(
        self,
        onnx_model,
        use_cache: bool = True,
        use_merged: Optional[bool] = None,
        use_io_binding: Optional[bool] = None,
    ):
        self.assertIsInstance(onnx_model, self.ORTMODEL_CLASS)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)
        self.assertIsInstance(onnx_model.session, InferenceSession)
        self.assertIsInstance(onnx_model.generation_config, GenerationConfig)

        self.assertEqual(onnx_model.generation_config.use_cache, use_cache)
        self.assertEqual(onnx_model.config.use_cache, use_cache)

        if use_cache or use_merged:
            self.assertTrue(onnx_model.can_use_cache)
        else:
            self.assertFalse(onnx_model.can_use_cache)

        if use_merged is not None:
            self.assertEqual(onnx_model.is_merged, use_merged)
        else:
            self.assertFalse(onnx_model.is_merged)

        if use_io_binding is not None:
            self.assertEqual(onnx_model.use_io_binding, use_io_binding)

    def compare_logits(
        self,
        inputs,
        outputs1,
        outputs2,
        onnx_model: ORTModelForCausalLM,
        use_cache: Optional[bool] = None,
    ):
        self.assertTrue("logits" in outputs1)
        self.assertTrue("logits" in outputs2)
        self.assertIsInstance(outputs1.logits, torch.Tensor)
        self.assertIsInstance(outputs2.logits, torch.Tensor)

        if is_transformers_version("<", "4.39.0") and "attention_mask" in inputs:
            self.mask_logits(outputs1.logits, inputs["attention_mask"])
            self.mask_logits(outputs2.logits, inputs["attention_mask"])

        torch.testing.assert_close(outputs1.logits, outputs2.logits, atol=self.ATOL, rtol=self.RTOL)

        if use_cache:
            self.assertTrue("past_key_values" in outputs1)
            self.assertTrue("past_key_values" in outputs2)
            self.assertIsInstance(outputs1.past_key_values, (tuple, list, Cache))
            self.assertIsInstance(outputs2.past_key_values, (tuple, list, Cache))

            if isinstance(outputs1.past_key_values, Cache):
                outputs1.past_key_values = outputs1.past_key_values.to_legacy_cache()
            if isinstance(outputs2.past_key_values, Cache):
                outputs2.past_key_values = outputs2.past_key_values.to_legacy_cache()

            if is_transformers_version("<", "4.39.0") and "attention_mask" in inputs:
                self.mask_past_key_values(onnx_model, outputs1.past_key_values, inputs["attention_mask"])
                self.mask_past_key_values(onnx_model, outputs2.past_key_values, inputs["attention_mask"])

            torch.testing.assert_close(
                outputs1.past_key_values, outputs2.past_key_values, atol=self.ATOL, rtol=self.RTOL
            )

    # INTEGRATION TESTS
    def test_find_untested_architectures(self):
        if len(self.SUPPORTED_ARCHITECTURES) != len(set(self.SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"For the task `{self.TASK}`, some architectures are duplicated in the list of tested architectures: "
                f"{self.SUPPORTED_ARCHITECTURES}.\n"
            )

        tested_architectures = set(self.SUPPORTED_ARCHITECTURES)
        transformers_architectures = set(CONFIG_MAPPING_NAMES.keys())
        onnx_architectures = set(TasksManager.get_supported_model_type_for_task(task=self.TASK, exporter="onnx"))
        supported_architectures = onnx_architectures & transformers_architectures
        untested_architectures = supported_architectures - tested_architectures

        if len(untested_architectures) > 0:
            raise ValueError(
                f"For the task `{self.TASK}`, the ONNX exporter supports {supported_architectures} but some of them are not "
                f"tested: {untested_architectures}.\n"
            )

    def test_all_models_requiring_position_ids(self):
        for model_type in TasksManager.get_supported_model_type_for_task(task=self.TASK, exporter="onnx"):
            model_type_requires_position_ids = model_type in MODEL_TYPES_REQUIRING_POSITION_IDS
            onnx_config_class = TasksManager._SUPPORTED_MODEL_TYPE[model_type]["onnx"][self.TASK].func
            onnx_config_class_with_position_ids = issubclass(onnx_config_class, TextDecoderWithPositionIdsOnnxConfig)

            if model_type_requires_position_ids ^ onnx_config_class_with_position_ids:
                raise ValueError(
                    f"Model type {model_type} {'requires' if model_type_requires_position_ids else 'does not require'} position ids, "
                    f"but the ONNX config class {onnx_config_class} {'is' if onnx_config_class_with_position_ids else 'is not'} "
                    f"subclassed from TextDecoderWithPositionIdsOnnxConfig.\n"
                )

    def test_load_model_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["vit"], export=True)
        self.assertIn("only supports the tasks", str(context.exception))

    def test_load_model_from_hub(self):
        model = self.ORTMODEL_CLASS.from_pretrained("fxmarty/onnx-tiny-random-gpt2-without-merge")
        self.check_onnx_model_attributes(model, use_cache=True, use_merged=False)

        model = self.ORTMODEL_CLASS.from_pretrained("fxmarty/onnx-tiny-random-gpt2-with-merge")
        self.check_onnx_model_attributes(model, use_cache=True, use_merged=True)

        model = self.ORTMODEL_CLASS.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        self.check_onnx_model_attributes(model, use_cache=True)

    @parameterized.expand(
        grid_parameters({"use_cache": [False, True], "use_merged": [False, True]}, add_test_name=False)
    )
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_load_model_with_external_data(self, use_cache: bool, use_merged: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_id = MODEL_NAMES["gpt2"]
            # export=True because there's a folder with onnx model in hf-internal-testing/tiny-random-GPT2LMHeadModel
            model = self.ORTMODEL_CLASS.from_pretrained(
                model_id, use_cache=use_cache, use_merged=use_merged, export=True
            )
            model.save_pretrained(tmpdirname)
            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(ONNX_WEIGHTS_NAME + "_data" in folder_contents)
            # verify loading from local folder works
            model = self.ORTMODEL_CLASS.from_pretrained(tmpdirname, use_cache=use_cache, use_merged=use_merged)
            model.generate(**self.GEN_KWARGS)
            remove_directory(tmpdirname)

    @require_hf_token
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_push_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_id = MODEL_NAMES["gpt2"]
            repo_dir = model_id.split("/")[-1] + "-onnx"
            token = os.environ.get("HF_AUTH_TOKEN", None)
            model = self.ORTMODEL_CLASS.from_pretrained(model_id, export=True)
            # verify the model can be pushed to the hub
            model.save_pretrained(tmpdirname, token=token, repository_id=repo_dir, push_to_hub=True)
            # verify pulling from hub works
            model = self.ORTMODEL_CLASS.from_pretrained(repo_dir, token=token, export=False)
            model.generate(**self.GEN_KWARGS)
            remove_directory(tmpdirname)

    def test_trust_remote_code(self):
        model_id = "optimum-internal-testing/tiny-testing-gpt2-remote-code"

        inputs = self.get_inputs("gpt2")
        model = self.AUTOMODEL_CLASS.from_pretrained(model_id, trust_remote_code=True).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(model_id, export=True, trust_remote_code=True)

        outputs = model(**inputs)
        onnx_outputs = onnx_model(**inputs)
        self.compare_logits(inputs, outputs, onnx_outputs, onnx_model=onnx_model)

    def test_load_model_from_hub_infer_onnx_model(self):
        model_id = "optimum-internal-testing/tiny-random-llama"
        # export from hub
        model = self.ORTMODEL_CLASS.from_pretrained(model_id)
        self.assertEqual(model.path.name, "model.onnx")
        # load from hub
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx")
        self.assertEqual(model.path.name, "model.onnx")
        # load from hub with revision
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx")
        self.assertEqual(model.path.name, "decoder_model_merged.onnx")
        # load from hub with revision and subfolder
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx", subfolder="subfolder")
        self.assertEqual(model.path.name, "model.onnx")
        # load from hub with revision and file_name
        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx", file_name="model_optimized.onnx")
        self.assertEqual(model.path.name, "model_optimized.onnx")
        # revision + file_name (decoder with past)
        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", file_name="decoder_with_past_model.onnx"
        )
        self.assertEqual(model.path.name, "decoder_with_past_model.onnx")

        # TODO: something went wrong here
        # revision + subfolder + file_name (target file exists but it loaded a different one)
        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", subfolder="subfolder", file_name="optimized_model.onnx"
        )
        self.assertEqual(model.path.name, "model.onnx")

        with self.assertRaises(FileNotFoundError):
            self.ORTMODEL_CLASS.from_pretrained(
                "hf-internal-testing/tiny-random-LlamaForCausalLM", file_name="doesnt_exist.onnx"
            )

    # NUMERICAL CONSISTENCY WITH TRANSFORMERS
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_logits_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        inputs = self.get_inputs(model_arch)

        set_seed(SEED)
        model = self.AUTOMODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch], use_cache=use_cache, trust_remote_code=trust_remote_code
        ).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, trust_remote_code=trust_remote_code
        )
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=use_cache)
        onnx_outputs = onnx_model(**inputs, use_cache=use_cache)
        self.compare_logits(inputs, outputs, onnx_outputs, onnx_model=onnx_model, use_cache=use_cache)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test, so we only test use_cache=True
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_generation_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        inputs = self.get_inputs(model_arch, for_generation=True)
        set_seed(SEED)
        model = self.AUTOMODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch], use_cache=use_cache, trust_remote_code=trust_remote_code
        ).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, trust_remote_code=trust_remote_code
        )
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache)

        set_seed(SEED)
        outputs = model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test, so we only test use_cache=True
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_beam_search_to_transformers(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        inputs = self.get_inputs(model_arch, for_generation=True)

        set_seed(SEED)
        model = self.AUTOMODEL_CLASS.from_pretrained(
            MODEL_NAMES[model_arch], use_cache=use_cache, trust_remote_code=trust_remote_code
        ).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, trust_remote_code=trust_remote_code
        )
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache)

        # beam search with random sampling
        gen_config = GenerationConfig(num_beams=4, max_new_tokens=10, min_new_tokens=10, do_sample=True)
        set_seed(SEED)
        outputs = model.generate(**inputs, generation_config=gen_config)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, generation_config=gen_config)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

        # group beam search with diversity penalty
        gen_config = GenerationConfig(
            num_beams=4,
            max_new_tokens=10,
            min_new_tokens=10,
            diversity_penalty=0.0001,
            num_beam_groups=2,
            do_sample=False,
        )
        outputs = model.generate(**inputs, generation_config=gen_config)
        onnx_outputs = onnx_model.generate(**inputs, generation_config=gen_config)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_generation_with_and_without_past_key_values(self, model_arch):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        model_args = {
            "test_name": model_arch + "_False",
            "model_arch": model_arch,
            "use_cache": False,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)
        model_args = {
            "test_name": model_arch + "_True",
            "model_arch": model_arch,
            "use_cache": True,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        model_with_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True, trust_remote_code=trust_remote_code
        )
        self.check_onnx_model_attributes(model_with_pkv, use_cache=True)
        model_without_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False, trust_remote_code=trust_remote_code
        )
        self.check_onnx_model_attributes(model_without_pkv, use_cache=False)

        inputs = self.get_inputs(model_arch, for_generation=True)
        set_seed(SEED)
        outputs_model_with_pkv = model_with_pkv.generate(**inputs, **self.GEN_KWARGS, use_cache=True)
        set_seed(SEED)
        outputs_model_without_pkv = model_without_pkv.generate(**inputs, **self.GEN_KWARGS, use_cache=False)
        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_logits_with_and_without_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_cache=use_cache,
            use_io_binding=False,
            trust_remote_code=trust_remote_code,
        )
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_io_binding=False)
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_cache=use_cache,
            use_io_binding=True,
            trust_remote_code=trust_remote_code,
        )
        self.check_onnx_model_attributes(io_model, use_cache=use_cache, use_io_binding=True)

        inputs = self.get_inputs(model_arch)
        set_seed(SEED)
        io_outputs = io_model(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        onnx_outputs = onnx_model(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        self.compare_logits(inputs, io_outputs, onnx_outputs, onnx_model=onnx_model, use_cache=use_cache)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test, so we only test use_cache=True
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_generation_with_and_without_io_binding(self, test_name: str, model_arch: str, use_cache: bool):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS
        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "trust_remote_code": trust_remote_code,
        }
        self._setup(model_args)

        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name],
            trust_remote_code=trust_remote_code,
            use_io_binding=False,
            use_cache=use_cache,
        )
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_io_binding=False)
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name],
            trust_remote_code=trust_remote_code,
            use_io_binding=True,
            use_cache=use_cache,
        )
        self.check_onnx_model_attributes(io_model, use_cache=use_cache, use_io_binding=True)

        inputs = self.get_inputs(model_arch, for_generation=True)
        set_seed(SEED)
        io_outputs = io_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(io_outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    # PIPELINE TESTS
    @parameterized.expand(grid_parameters({"use_cache": [True, False]}))
    def test_pipeline_with_default_model(self, test_name: str, use_cache: bool):
        texts = self.get_inputs("llama", for_pipeline=True)
        pipe = optimum_pipeline("text-generation", model_kwargs={"use_cache": use_cache})
        self.check_onnx_model_attributes(pipe.model, use_cache=use_cache)

        set_seed(SEED)
        outputs = pipe(texts, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0][0], dict)
        self.assertIn("generated_text", outputs[0][0])
        self.assertIsInstance(outputs[0][0]["generated_text"], str)
        self.assertGreater(len(outputs[0][0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = optimum_pipeline("text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache})
            set_seed(SEED)
            outputs_local_model = pipe(texts, **self.GEN_KWARGS)
            self.assertEqual(outputs, outputs_local_model)

    @parameterized.expand(grid_parameters({"model_arch": ["llama"], "use_cache": [True, False]}))
    def test_pipeline_with_onnx_model(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        tokenizer = self.get_tokenizer(model_arch)
        texts = self.get_inputs(model_arch, for_pipeline=True)
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache)

        pipe = optimum_pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)
        set_seed(SEED)
        outputs = pipe(texts, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0][0], dict)
        self.assertIn("generated_text", outputs[0][0])
        self.assertIsInstance(outputs[0][0]["generated_text"], str)
        self.assertGreater(len(outputs[0][0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = optimum_pipeline("text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache})
            set_seed(SEED)
            local_pipe_outputs = pipe(texts, **self.GEN_KWARGS)
            self.assertEqual(outputs, local_pipe_outputs)

    @parameterized.expand(grid_parameters({"model_arch": ["llama"], "use_cache": [True, False]}, add_test_name=False))
    def test_pipeline_with_hub_model_id(self, model_arch: str, use_cache: bool):
        texts = self.get_inputs(model_arch, for_pipeline=True)
        pipe = optimum_pipeline(
            "text-generation", model=MODEL_NAMES[model_arch], model_kwargs={"use_cache": use_cache}
        )

        set_seed(SEED)
        outputs = pipe(texts, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0][0], dict)
        self.assertIn("generated_text", outputs[0][0])
        self.assertIsInstance(outputs[0][0]["generated_text"], str)
        self.assertGreater(len(outputs[0][0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = optimum_pipeline("text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache})
            set_seed(SEED)
            outputs_local_model = pipe(texts, **self.GEN_KWARGS)
            self.assertEqual(outputs, outputs_local_model)

    @parameterized.expand([(False,), (True,)])
    def test_inference_with_old_onnx_model(self, use_cache):
        # old onnx model can't handle batched inputs (missing position_ids)
        inputs = self.get_inputs("gpt2", batched=False)
        model = self.AUTOMODEL_CLASS.from_pretrained("gpt2").eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained("optimum/gpt2", use_cache=use_cache)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache)

        self.assertEqual(onnx_model.use_cache, use_cache)
        if use_cache:
            self.assertEqual(onnx_model.path.name, ONNX_DECODER_WITH_PAST_NAME)
        else:
            self.assertEqual(onnx_model.path.name, ONNX_DECODER_NAME)

        with torch.no_grad():
            outputs = model(**inputs)
        onnx_outputs = onnx_model(**inputs)
        self.compare_logits(inputs, outputs, onnx_outputs, onnx_model=onnx_model, use_cache=use_cache)

        # old onnx model can't handle batched inputs (missing position_ids)
        inputs = self.get_inputs("gpt2", for_generation=True, batched=False)
        set_seed(SEED)
        outputs = model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(outputs, onnx_outputs, atol=self.ATOL, rtol=self.RTOL)

    # LEGACY MODELS
    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_merged_and_not_merged_models_outputs(self, model_arch: str):
        trust_remote_code = model_arch in self.TRUST_REMOTE_CODE_MODELS

        with tempfile.TemporaryDirectory() as tmpdir:
            # legacy models can't handle batched inputs (missing position_ids)
            inputs = self.get_inputs(model_arch, batched=False)
            model_id = MODEL_NAMES[model_arch]
            task = "text-generation-with-past"

            set_seed(SEED)
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id, trust_remote_code=trust_remote_code).eval()

            set_seed(SEED)
            main_export(
                model_id,
                output=tmpdir,
                task=task,
                legacy=True,
                no_post_processing=False,
                do_validation=False,
                trust_remote_code=trust_remote_code,
            )

            # not merged model, without cache
            not_merged_without_cache_model = self.ORTMODEL_CLASS.from_pretrained(
                tmpdir, use_cache=False, use_merged=False, trust_remote_code=trust_remote_code
            )
            self.check_onnx_model_attributes(not_merged_without_cache_model, use_cache=False, use_merged=False)

            # merged model
            merged_model = self.ORTMODEL_CLASS.from_pretrained(
                tmpdir, use_cache=True, use_merged=True, trust_remote_code=trust_remote_code
            )
            self.check_onnx_model_attributes(merged_model, use_cache=True, use_merged=True)

            # forward
            logits = model(**inputs).logits
            merged_logits = merged_model(**inputs).logits
            not_merged_without_cache_logits = not_merged_without_cache_model(**inputs).logits

            # compare merged to transformers
            torch.testing.assert_close(logits, merged_logits, atol=self.ATOL, rtol=self.RTOL)
            # compare not merged without cache to transformers
            torch.testing.assert_close(logits, not_merged_without_cache_logits, atol=self.ATOL, rtol=self.RTOL)

            # legacy models can't handle batched inputs (missing position_ids)
            inputs = self.get_inputs(model_arch, for_generation=True, batched=False)
            # generate
            set_seed(SEED)
            outputs = model.generate(**inputs, **self.GEN_KWARGS)
            set_seed(SEED)
            merged_outputs = merged_model.generate(**inputs, **self.GEN_KWARGS)
            set_seed(SEED)
            not_merged_without_cache_outputs = not_merged_without_cache_model.generate(**inputs, **self.GEN_KWARGS)

            # compare merged to transformers
            torch.testing.assert_close(outputs, merged_outputs, atol=self.ATOL, rtol=self.RTOL)
            # compare not merged without cache to transformers
            torch.testing.assert_close(outputs, not_merged_without_cache_outputs, atol=self.ATOL, rtol=self.RTOL)

            # load and save (only merged is loaded and saved)
            loaded_model = self.ORTMODEL_CLASS.from_pretrained(tmpdir, trust_remote_code=trust_remote_code)
            self.assertEqual(loaded_model.path.name, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(loaded_model.use_merged)
            self.assertTrue(loaded_model.use_cache)
            save_dir = os.path.join(tmpdir, "save")
            loaded_model.save_pretrained(save_dir)
            self.assertNotIn(ONNX_DECODER_NAME, os.listdir(save_dir))
            self.assertNotIn(ONNX_DECODER_WITH_PAST_NAME, os.listdir(save_dir))
            self.assertIn(ONNX_DECODER_MERGED_NAME, os.listdir(save_dir))
            reloaded_model = self.ORTMODEL_CLASS.from_pretrained(save_dir, trust_remote_code=trust_remote_code)
            self.assertEqual(reloaded_model.path.name, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(reloaded_model.use_merged)
            self.assertTrue(reloaded_model.use_cache)
