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
import os
import tempfile
import unittest
from typing import Optional

import pytest
import torch
from parameterized import parameterized
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, PretrainedConfig, set_seed
from transformers.cache_utils import Cache
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from optimum.exporters import TasksManager
from optimum.onnx.utils import has_onnx_input
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.modeling_seq2seq import ORTDecoderForSeq2Seq, ORTEncoder
from optimum.onnxruntime.utils import (
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
)
from optimum.pipelines import pipeline as optimum_pipeline
from optimum.utils.import_utils import is_tensorrt_available, is_transformers_version
from optimum.utils.testing_utils import grid_parameters, remove_directory, require_hf_token


TORCH_DEVICE = "cpu"
EXECUTION_ROVIDER = "CPUExecutionProvider"

if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
    if is_tensorrt_available():
        EXECUTION_ROVIDER = "TensorrtExecutionProvider"
    elif torch.version.hip is not None:
        EXECUTION_ROVIDER = "ROCMExecutionProvider"
    else:
        EXECUTION_ROVIDER = "CUDAExecutionProvider"


class ORTModelForSeq2SeqLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "bart",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        "encoder-decoder",
        "encoder-decoder-bert-bert",
        "longt5",
        "m2m_100",
        "marian",
        "mbart",
        "mt5",
        "pegasus",
        "t5",
    ]

    GEN_KWARGS = {"max_new_tokens": 10, "min_new_tokens": 10, "num_beams": 1, "do_sample": False}  # noqa: RUF012
    if is_transformers_version(">=", "4.51.0"):
        GEN_KWARGS["use_model_defaults"] = False

    TASK = "text2text-generation"
    ORTMODEL_CLASS = ORTModelForSeq2SeqLM
    AUTOMODEL_CLASS = AutoModelForSeq2SeqLM

    def get_batched_inputs(self):
        return ["This is me", "Today is a nice day and I am longer"]

    def get_tokenizer(self, model_id: str, trust_remote_code: bool = False):
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
        return tokenizer

    def check_onnx_model_sanity(
        self,
        onnx_model,
        use_cache: bool = True,
        use_merged: Optional[bool] = None,
        use_io_binding: Optional[bool] = None,
    ):
        self.assertIsInstance(onnx_model, self.ORTMODEL_CLASS)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)
        self.assertIsInstance(onnx_model.generation_config, GenerationConfig)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
        if use_cache and use_merged is not True:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
        else:
            self.assertIsNone(onnx_model.decoder_with_past)

        self.assertEqual(onnx_model.config.use_cache, use_cache)
        if use_merged is not None:
            self.assertEqual(onnx_model.is_merged, use_merged)
        if use_io_binding is not None:
            self.assertEqual(onnx_model.use_io_binding, use_io_binding)

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

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    def test_load_model_from_hub(self):
        # TODO: create, export and push a tiny random t5 models to the hub
        # one merged, one not merged, and without cache support
        pass

    @parameterized.expand(
        grid_parameters({"use_cache": [False, True], "use_merged": [False, True]}, add_test_name=False)
    )
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_load_model_with_external_data(self, use_cache: bool, use_merged: bool):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_id = MODEL_NAMES["t5"]
            model = self.ORTMODEL_CLASS.from_pretrained(
                model_id, use_cache=use_cache, use_merged=use_merged, export=True
            )
            model.save_pretrained(tmpdirname)
            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(ONNX_ENCODER_NAME, folder_contents)
            self.assertIn(ONNX_ENCODER_NAME + "_data", folder_contents)

            if use_merged:
                merged_path = os.path.join(tmpdirname, ONNX_DECODER_MERGED_NAME)
                self.assertTrue(has_onnx_input(merged_path, "use_cache_branch"))
                self.assertIn(ONNX_DECODER_MERGED_NAME, folder_contents)
                self.assertIn(ONNX_DECODER_MERGED_NAME + "_data", folder_contents)
            else:
                not_merged_path = os.path.join(tmpdirname, ONNX_DECODER_NAME)
                self.assertFalse(has_onnx_input(not_merged_path, "use_cache_branch"))
                self.assertIn(ONNX_DECODER_NAME, folder_contents)
                self.assertIn(ONNX_DECODER_NAME + "_data", folder_contents)

                if use_cache:
                    with_cache_path = os.path.join(tmpdirname, ONNX_DECODER_WITH_PAST_NAME)
                    self.assertFalse(has_onnx_input(with_cache_path, "use_cache_branch"))
                    self.assertIn(ONNX_DECODER_WITH_PAST_NAME, folder_contents)
                    self.assertIn(ONNX_DECODER_WITH_PAST_NAME + "_data", folder_contents)

            # verify loading from local folder works
            model = self.ORTMODEL_CLASS.from_pretrained(tmpdirname, use_cache=use_cache, use_merged=use_merged)
            model.generate(**self.GEN_KWARGS)
            remove_directory(tmpdirname)

    @require_hf_token
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_push_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_id = MODEL_NAMES["t5"]
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
        # TODO: create a t5 model with custom remote code and test it
        pass

    def test_load_model_from_hub_infer_onnx_model(self):
        # TODO: add a test for the different arguments of from_pretrained like subfolder, revision, filename, etc.
        pass

    # NUMERICAL CONSISTENCY WITH TRANSFORMERS
    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False], "use_merged": [False, True]}
        )
    )
    def test_compare_logits_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if model_arch == "encoder-decoder-bert-bert":
            if use_cache:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support returning past key values (use_cache=True)."
                )
            elif use_merged:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
                )

        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        texts = self.get_batched_inputs()
        tokenizer = self.get_tokenizer(model_id)
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        inputs["decoder_input_ids"] = torch.ones((inputs.input_ids.shape[0], 1), dtype=torch.long)

        set_seed(SEED)
        if model_arch.startswith("encoder-decoder"):
            # EnocderDecoderModel does not take `use_cache` during instantiation
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        else:
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id, use_cache=use_cache).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged
        )
        self.check_onnx_model_sanity(onnx_model, use_cache=use_cache, use_merged=use_merged)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=use_cache)
        onnx_outputs = onnx_model(**inputs, use_cache=use_cache)

        self.assertTrue("logits" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.logits, torch.Tensor)
        torch.testing.assert_close(onnx_outputs.logits, outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        self.assertTrue("encoder_last_hidden_state" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.encoder_last_hidden_state, torch.Tensor)
        torch.testing.assert_close(
            onnx_outputs.encoder_last_hidden_state, outputs.encoder_last_hidden_state, atol=self.ATOL, rtol=self.RTOL
        )

        if use_cache:
            self.assertTrue("past_key_values" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.past_key_values, tuple)
            self.assertIsInstance(onnx_outputs.past_key_values[0], tuple)

            if isinstance(outputs.past_key_values, Cache):
                outputs.past_key_values = outputs.past_key_values.to_legacy_cache()

            torch.testing.assert_close(
                onnx_outputs.past_key_values, outputs.past_key_values, atol=self.ATOL, rtol=self.RTOL
            )

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_merged": [False, True], "use_cache": [True]})
    )
    # Generation is slow without pkv, and we do compare with/without pkv in a different test, so we only test use_cache=True
    def test_compare_generation_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        if model_arch == "encoder-decoder-bert-bert":
            if use_merged:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
                )
            elif use_cache:
                # The encoder-decoder-bert-bert model does not support using pkv cache, so we test it with use_cache=False.
                use_cache = False

        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        texts = self.get_batched_inputs()
        tokenizer = self.get_tokenizer(model_id)
        inputs = tokenizer(texts, return_tensors="pt", padding=True)

        set_seed(SEED)
        if model_arch.startswith("encoder-decoder"):
            # EnocderDecoderModel does not take `use_cache` during instantiation
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        else:
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id, use_cache=use_cache).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged
        )
        self.check_onnx_model_sanity(onnx_model, use_cache=use_cache, use_merged=use_merged)

        if model_arch == "encoder-decoder-bert-bert":
            model.config.decoder_start_token_id = 1
            onnx_model.config.decoder_start_token_id = 1
            model.generation_config.decoder_start_token_id = 1
            onnx_model.generation_config.decoder_start_token_id = 1

        outputs = model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

    # Beam search generation is slow without pkv, and we do compare with/without pkv in a different test, so we only test use_cache=True
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_beam_search_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        if model_arch == "encoder-decoder-bert-bert":
            if use_merged:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
                )
            elif use_cache:
                # The encoder-decoder-bert-bert model does not support using pkv cache, so we test it with use_cache=False.
                use_cache = False

        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        texts = self.get_batched_inputs()
        tokenizer = self.get_tokenizer(model_id)
        inputs = tokenizer(texts, return_tensors="pt", padding=True)

        set_seed(SEED)
        if model_arch.startswith("encoder-decoder"):
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id).eval()
        else:
            model = self.AUTOMODEL_CLASS.from_pretrained(model_id, use_cache=use_cache).eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged
        )
        self.check_onnx_model_sanity(onnx_model, use_cache=use_cache, use_merged=use_merged)

        gen_kwargs = {"use_cache": use_cache}
        if is_transformers_version(">=", "4.51.0"):
            gen_kwargs["use_model_defaults"] = False

        if model_arch == "bigbird_pegasus":
            # bigbird_pegasus is exported with original_full attention to avoid
            # issues with switching attention type inbetween inference iterations
            model.model.encoder.set_attention_type("original_full")

        if model_arch == "encoder-decoder-bert-bert":
            model.config.decoder_start_token_id = 1
            onnx_model.config.decoder_start_token_id = 1
            model.generation_config.decoder_start_token_id = 1
            onnx_model.generation_config.decoder_start_token_id = 1

        if model_arch == "encoder-decoder":
            model._reorder_cache = onnx_model._reorder_cache

        # beam search with random sampling
        gen_config = GenerationConfig(
            num_beams=4,
            do_sample=True,
            max_new_tokens=10,
            min_new_tokens=10,
        )
        set_seed(SEED)
        outputs = model.generate(**inputs, **gen_kwargs, generation_config=gen_config)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **gen_kwargs, generation_config=gen_config)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

        # group beam search with diversity penalty
        gen_config = GenerationConfig(
            num_beams=4,
            do_sample=False,
            max_new_tokens=10,
            min_new_tokens=10,
            num_beam_groups=2,
            diversity_penalty=0.0001,
        )
        outputs = model.generate(**inputs, **gen_kwargs, generation_config=gen_config)
        onnx_outputs = onnx_model.generate(**inputs, **gen_kwargs, generation_config=gen_config)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)

    # NUMERICAL CONSISTENCY WITH PAST KEY VALUES
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_merged": [False, True]}))
    def test_compare_generation_with_and_without_past_key_values(
        self, test_name: str, model_arch: str, use_merged: bool
    ):
        if model_arch == "encoder-decoder-bert-bert":
            pytest.skip(
                "The encoder-decoder-bert-bert model does not support returning past key values (use_cache=True)."
            )

        model_args = {
            "test_name": model_arch + "_False",
            "model_arch": model_arch,
            "use_merged": use_merged,
            "use_cache": False,
        }
        self._setup(model_args)
        model_args = {
            "test_name": model_arch + "_True",
            "model_arch": model_arch,
            "use_merged": use_merged,
            "use_cache": True,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        texts = self.get_batched_inputs()
        tokenizer = self.get_tokenizer(model_id)
        tokens = tokenizer(texts, return_tensors="pt", padding=True)

        model_with_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        self.check_onnx_model_sanity(model_with_pkv, use_cache=True)
        model_without_pkv = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        self.check_onnx_model_sanity(model_without_pkv, use_cache=False)

        outputs_model_with_pkv = model_with_pkv.generate(**tokens, **self.GEN_KWARGS, use_cache=True)
        outputs_model_without_pkv = model_without_pkv.generate(**tokens, **self.GEN_KWARGS, use_cache=False)
        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL)

    # NUMERICAL CONSISTENCY WITH DECODER MERGING
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    # Generation is slow without pkv, and we do compare with/without pkv in a different test, so we only test use_cache=True
    def test_compare_merged_and_not_merged(self, test_name: str, model_arch: str, use_cache: bool):
        if model_arch == "encoder-decoder-bert-bert":
            pytest.skip(
                "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
            )

        model_args = {
            "test_name": test_name + "_True",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": True,
        }
        self._setup(model_args)
        model_args = {
            "test_name": test_name + "_False",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": False,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        texts = self.get_batched_inputs()
        tokenizer = self.get_tokenizer(model_id)
        tokens = tokenizer(texts, return_tensors="pt", padding=True)
        tokens["decoder_input_ids"] = torch.ones((tokens.input_ids.shape[0], 1), dtype=torch.long)

        model_merged = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name + "_True"])
        self.check_onnx_model_sanity(model_merged, use_cache=use_cache, use_merged=True)
        model_not_merged = self.ORTMODEL_CLASS.from_pretrained(self.onnx_model_dirs[test_name + "_False"])
        self.check_onnx_model_sanity(model_not_merged, use_cache=use_cache, use_merged=False)

        outputs_model_merged = model_merged(**tokens, use_cache=use_cache)
        outputs_model_not_merged = model_not_merged(**tokens, use_cache=use_cache)

        self.assertTrue("logits" in outputs_model_merged)
        self.assertIsInstance(outputs_model_merged.logits, torch.Tensor)
        torch.testing.assert_close(
            outputs_model_not_merged.logits, outputs_model_merged.logits, rtol=self.RTOL, atol=self.ATOL
        )

        self.assertTrue("encoder_last_hidden_state" in outputs_model_merged)
        self.assertIsInstance(outputs_model_merged.encoder_last_hidden_state, torch.Tensor)
        torch.testing.assert_close(
            outputs_model_not_merged.encoder_last_hidden_state,
            outputs_model_merged.encoder_last_hidden_state,
            rtol=self.RTOL,
            atol=self.ATOL,
        )

        if use_cache:
            self.assertTrue("past_key_values" in outputs_model_merged)
            self.assertIsInstance(outputs_model_merged.past_key_values, tuple)
            self.assertIsInstance(outputs_model_merged.past_key_values[0], tuple)

            if isinstance(outputs_model_not_merged.past_key_values, Cache):
                outputs_model_not_merged.past_key_values = outputs_model_not_merged.past_key_values.to_legacy_cache()

            torch.testing.assert_close(
                outputs_model_not_merged.past_key_values,
                outputs_model_merged.past_key_values,
                rtol=self.RTOL,
                atol=self.ATOL,
            )

        tokens.pop("decoder_input_ids")
        outputs_model_merged = model_merged.generate(**tokens, **self.GEN_KWARGS, use_cache=use_cache)
        outputs_model_not_merged = model_not_merged.generate(**tokens, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged, rtol=self.RTOL, atol=self.ATOL)

    # NUMERICAL CONSISTENCY WITH IO BINDING
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_merged": [False, True], "use_cache": [True]})
    )
    # Generation is slow without pkv, and we do compare with/without pkv in a different test, so we only test use_cache=True
    def test_compare_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_merged: bool, use_cache: bool
    ):
        if model_arch == "encoder-decoder-bert-bert":
            if use_merged:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
                )
            elif use_cache:
                # The encoder-decoder-bert-bert model does not support using pkv cache, so we test it with use_cache=False.
                use_cache = False

        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, use_cache=use_cache
        )
        io_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, use_cache=use_cache
        )
        self.check_onnx_model_sanity(onnx_model, use_cache=use_cache, use_merged=use_merged, use_io_binding=False)
        self.check_onnx_model_sanity(io_model, use_cache=use_cache, use_merged=use_merged, use_io_binding=True)

        inputs = self.get_batched_inputs()
        tokenizer = self.get_tokenizer(model_id)
        tokens = tokenizer(inputs, return_tensors="pt", padding=True)
        tokens["decoder_input_ids"] = torch.ones((tokens.input_ids.shape[0], 1), dtype=torch.long)

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        self.assertTrue("encoder_last_hidden_state" in io_outputs)
        self.assertIsInstance(io_outputs.encoder_last_hidden_state, torch.Tensor)
        torch.testing.assert_close(
            onnx_outputs.encoder_last_hidden_state,
            io_outputs.encoder_last_hidden_state,
            atol=self.ATOL,
            rtol=self.RTOL,
        )

        if use_cache:
            self.assertTrue("past_key_values" in io_outputs)
            self.assertIsInstance(io_outputs.past_key_values, tuple)
            self.assertIsInstance(io_outputs.past_key_values[0], tuple)

            if isinstance(onnx_outputs.past_key_values, Cache):
                onnx_outputs.past_key_values = onnx_outputs.past_key_values.to_legacy_cache()

            torch.testing.assert_close(
                onnx_outputs.past_key_values, io_outputs.past_key_values, atol=self.ATOL, rtol=self.RTOL
            )

        if model_arch == "encoder-decoder-bert-bert":
            io_model.config.decoder_start_token_id = 1
            onnx_model.config.decoder_start_token_id = 1
            io_model.generation_config.decoder_start_token_id = 1
            onnx_model.generation_config.decoder_start_token_id = 1

        tokens.pop("decoder_input_ids")
        io_outputs = io_model.generate(**tokens, **self.GEN_KWARGS, use_cache=use_cache)
        onnx_outputs = onnx_model.generate(**tokens, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)

    # PIPELINE TESTS
    @parameterized.expand(grid_parameters({"use_cache": [True, False], "use_merged": [False, True]}))
    def test_pipeline_with_default_model(self, tste_name: str, use_cache: bool, use_merged: bool):
        text = "The capital of France is"

        # Text2Text generation
        pipe = optimum_pipeline(
            "text2text-generation", model_kwargs={"use_cache": use_cache, "use_merged": use_merged, "export": True}
        )
        self.check_onnx_model_sanity(pipe.model, use_cache=use_cache, use_merged=use_merged)
        set_seed(SEED)
        outputs = pipe(text, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertGreater(len(outputs[0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = optimum_pipeline(
                "text2text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache, "use_merged": use_merged}
            )
            self.check_onnx_model_sanity(pipe.model, use_cache=use_cache, use_merged=use_merged)
            set_seed(SEED)
            outputs_local_model = pipe(text, **self.GEN_KWARGS)
            self.assertEqual(outputs[0]["generated_text"], outputs_local_model[0]["generated_text"])

        # Summarization
        pipe = optimum_pipeline(
            "summarization", model_kwargs={"use_cache": use_cache, "use_merged": use_merged, "export": True}
        )
        self.check_onnx_model_sanity(pipe.model, use_cache=use_cache, use_merged=use_merged)
        set_seed(SEED)
        outputs = pipe(text, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("summary_text", outputs[0])
        self.assertIsInstance(outputs[0]["summary_text"], str)
        self.assertGreater(len(outputs[0]["summary_text"]), 0)

        # Translation
        pipe = optimum_pipeline(
            "translation_en_to_de", model_kwargs={"use_cache": use_cache, "use_merged": use_merged, "export": True}
        )
        self.check_onnx_model_sanity(pipe.model, use_cache=use_cache, use_merged=use_merged)
        set_seed(SEED)
        outputs = pipe(text, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("translation_text", outputs[0])
        self.assertIsInstance(outputs[0]["translation_text"], str)
        self.assertGreater(len(outputs[0]["translation_text"]), 0)

    @parameterized.expand(
        grid_parameters({"model_arch": ["t5"], "use_cache": [True, False], "use_merged": [False, True]})
    )
    def test_pipeline_with_onnx_model(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "use_merged": use_merged,
            "model_arch": model_arch,
        }
        self._setup(model_args)

        text = "The capital of France is"
        model_id = MODEL_NAMES[model_arch]
        tokenizer = self.get_tokenizer(model_id)
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged
        )
        self.check_onnx_model_sanity(onnx_model, use_cache=use_cache, use_merged=use_merged)

        # Text2Text generation
        pipe = optimum_pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer)
        set_seed(SEED)
        outputs = pipe(text, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = optimum_pipeline(
                "text2text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache, "use_merged": use_merged}
            )
            set_seed(SEED)
            outputs_local_model = pipe(text, **self.GEN_KWARGS)
            self.assertEqual(outputs[0]["generated_text"], outputs_local_model[0]["generated_text"])

        # Summarization
        pipe = optimum_pipeline("summarization", model=onnx_model, tokenizer=tokenizer)
        set_seed(SEED)
        outputs = pipe(text, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("summary_text", outputs[0])
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = optimum_pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
        set_seed(SEED)
        outputs = pipe(text, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("translation_text", outputs[0])
        self.assertIsInstance(outputs[0]["translation_text"], str)

    @parameterized.expand(grid_parameters({"use_cache": [False, True]}, add_test_name=False))
    def test_inference_old_seq2seq_onnx_model(self, use_cache):
        model = self.AUTOMODEL_CLASS.from_pretrained("t5-small").eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained("optimum/t5-small", use_cache=use_cache)
        self.check_onnx_model_sanity(onnx_model, use_cache=use_cache, use_merged=False)

        texts = self.get_batched_inputs()
        tokenizer = self.get_tokenizer("t5-small")
        tokens = tokenizer(texts, return_tensors="pt", padding=True)

        outputs = model.generate(**tokens, **self.GEN_KWARGS, use_cache=use_cache)
        onnx_outputs = onnx_model.generate(**tokens, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(onnx_outputs, outputs, atol=self.ATOL, rtol=self.RTOL)
