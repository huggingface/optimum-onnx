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

import numpy as np
import pytest
import torch
from onnxruntime import InferenceSession
from parameterized import parameterized
from PIL import Image
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    set_seed,
)
from transformers.cache_utils import Cache
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from optimum.exporters import TasksManager
from optimum.exporters.onnx.model_configs import MoonshineOnnxConfig
from optimum.onnx.utils import has_onnx_input
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTModelForSpeechSeq2Seq, ORTModelForVision2Seq, pipeline
from optimum.onnxruntime.modeling_seq2seq import ORTDecoderForSeq2Seq, ORTEncoder
from optimum.onnxruntime.utils import (
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
)
from optimum.utils.import_utils import is_transformers_version
from optimum.utils.testing_utils import grid_parameters, remove_directory, require_hf_token


class ORTSeq2SeqTestMixin(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = None

    MODEL_ATOL = {}  # noqa: RUF012
    MODEL_RTOL = {}  # noqa: RUF012

    GEN_KWARGS = {  # noqa: RUF012
        "num_beams": 1,  # we test beam search in a separate test
        "do_sample": True,  # to avoid the model returning the same id repeatedly
        "max_new_tokens": 10,
        "min_new_tokens": 10,
    }

    def get_inputs(self, model_arch: str, for_generation: bool = False, for_pipeline: bool = False):
        raise NotImplementedError(f"Please implement the `get_inputs` method in the {self.__class__.__name__} class.")

    def get_transformers_model(
        self, test_name: str, model_arch: str, use_cache: bool = True, use_merged: Optional[bool] = None, **kwargs
    ):
        raise NotImplementedError(
            f"Please implement the `get_transformers_model` method in the {self.__class__.__name__} class."
        )

    def get_onnx_model(
        self, test_name: str, model_arch: str, use_cache: bool = True, use_merged: Optional[bool] = None, **kwargs
    ):
        raise NotImplementedError(
            f"Please implement the `get_onnx_model` method in the {self.__class__.__name__} class."
        )

    def check_onnx_model_attributes(
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
        self.assertIsInstance(onnx_model.encoder.session, InferenceSession)
        self.assertIsInstance(onnx_model.decoder.session, InferenceSession)
        if use_cache and use_merged is not True:
            # if a model is exported with use_cache=True and use_merged=False/None
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
            self.assertIsInstance(onnx_model.decoder_with_past.session, InferenceSession)
        else:
            self.assertIsNone(onnx_model.decoder_with_past)

        self.assertEqual(onnx_model.config.use_cache, use_cache)

        if use_cache or use_merged:
            self.assertTrue(onnx_model.can_use_cache)
        else:
            self.assertFalse(onnx_model.can_use_cache)

        if use_merged is not None:
            self.assertEqual(onnx_model.decoder.is_merged, use_merged)
        else:
            self.assertFalse(onnx_model.decoder.is_merged)

        if use_io_binding is not None:
            self.assertEqual(onnx_model.use_io_binding, use_io_binding)

    def compare_logits(self, model_arch: str, outputs1, outputs2, use_cache: bool = True):
        atol = self.MODEL_ATOL.get(model_arch, self.ATOL)
        rtol = self.MODEL_RTOL.get(model_arch, self.RTOL)

        self.assertTrue("logits" in outputs1)
        self.assertTrue("logits" in outputs2)
        self.assertIsInstance(outputs1.logits, torch.Tensor)
        self.assertIsInstance(outputs2.logits, torch.Tensor)
        torch.testing.assert_close(outputs1.logits, outputs2.logits, atol=atol, rtol=rtol)

        self.assertTrue("encoder_last_hidden_state" in outputs1)
        self.assertTrue("encoder_last_hidden_state" in outputs2)
        self.assertIsInstance(outputs1.encoder_last_hidden_state, torch.Tensor)
        self.assertIsInstance(outputs2.encoder_last_hidden_state, torch.Tensor)
        torch.testing.assert_close(
            outputs1.encoder_last_hidden_state, outputs2.encoder_last_hidden_state, atol=atol, rtol=rtol
        )

        if use_cache:
            self.assertTrue("past_key_values" in outputs1)
            self.assertTrue("past_key_values" in outputs2)
            self.assertIsInstance(outputs1.past_key_values, (tuple, list, Cache))
            self.assertIsInstance(outputs2.past_key_values, (tuple, list, Cache))
            self.assertIsInstance(outputs1.past_key_values[0], tuple)
            self.assertIsInstance(outputs2.past_key_values[0], tuple)

            if isinstance(outputs1.past_key_values, Cache):
                outputs1.past_key_values = outputs1.past_key_values.to_legacy_cache()
            if isinstance(outputs2.past_key_values, Cache):
                outputs2.past_key_values = outputs2.past_key_values.to_legacy_cache()

            torch.testing.assert_close(outputs1.past_key_values, outputs2.past_key_values, atol=atol, rtol=rtol)

    # INTEGRATION TESTS
    def _test_find_untested_architectures(self):
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

    def _test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = self.ORTMODEL_CLASS.from_pretrained(MODEL_NAMES["bert"], export=True)
        self.assertIn("only supports the tasks", str(context.exception))

    # NUMERICAL CONSISTENCY WITH TRANSFORMERS
    def _test_compare_logits_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool = True, use_merged: Optional[bool] = None
    ):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(setup_args)

        inputs = self.get_inputs(model_arch)
        model = self.get_transformers_model(**setup_args)
        onnx_model = self.get_onnx_model(**setup_args)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged)

        outputs = model(**inputs, use_cache=use_cache)
        onnx_outputs = onnx_model(**inputs, use_cache=use_cache)
        self.compare_logits(model_arch, outputs, onnx_outputs, use_cache=use_cache)

    def _test_compare_generation_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool = True, use_merged: Optional[bool] = None
    ):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(setup_args)

        inputs = self.get_inputs(model_arch, for_generation=True)
        model = self.get_transformers_model(**setup_args)
        onnx_model = self.get_onnx_model(**setup_args)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged)

        set_seed(SEED)
        outputs = model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(outputs, onnx_outputs)

    def _test_compare_beam_search_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool = True, use_merged: bool = False
    ):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(setup_args)

        inputs = self.get_inputs(model_arch, for_generation=True)
        model = self.get_transformers_model(**setup_args)
        onnx_model = self.get_onnx_model(**setup_args)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged)

        # beam search with random sampling
        gen_config = GenerationConfig(
            num_beams=4,
            do_sample=True,
            max_new_tokens=10,
            min_new_tokens=10,
            use_cache=use_cache,
        )
        set_seed(SEED)
        outputs = model.generate(**inputs, generation_config=gen_config)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, generation_config=gen_config)
        torch.testing.assert_close(outputs, onnx_outputs)

        # group beam search with diversity penalty
        gen_config = GenerationConfig(
            num_beams=4,
            do_sample=False,
            max_new_tokens=10,
            min_new_tokens=10,
            num_beam_groups=2,
            diversity_penalty=0.0001,
            use_cache=use_cache,
        )
        outputs = model.generate(**inputs, generation_config=gen_config)
        onnx_outputs = onnx_model.generate(**inputs, generation_config=gen_config)
        torch.testing.assert_close(outputs, onnx_outputs)

    # NUMERICAL CONSISTENCY WITH DECODER MERGING
    def _test_compare_logits_merged_and_not_merged(self, model_arch: str, use_cache: bool = True):
        merged_setup_args = {
            "test_name": f"{model_arch}_{use_cache}_True",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": True,
        }
        self._setup(merged_setup_args)
        not_merged_setup_args = {
            "test_name": f"{model_arch}_{use_cache}_False",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": False,
        }
        self._setup(not_merged_setup_args)

        inputs = self.get_inputs(model_arch)
        model_merged = self.get_onnx_model(**merged_setup_args)
        model_not_merged = self.get_onnx_model(**not_merged_setup_args)
        self.check_onnx_model_attributes(model_merged, use_cache=use_cache, use_merged=True)
        self.check_onnx_model_attributes(model_not_merged, use_cache=use_cache, use_merged=False)

        outputs_model_merged = model_merged(**inputs, use_cache=use_cache)
        outputs_model_not_merged = model_not_merged(**inputs, use_cache=use_cache)
        self.compare_logits(model_arch, outputs_model_not_merged, outputs_model_merged, use_cache=use_cache)

    def _test_compare_generation_merged_and_not_merged(self, model_arch: str, use_cache: bool = True):
        merged_setup_args = {
            "test_name": f"{model_arch}_{use_cache}_True",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": True,
        }
        self._setup(merged_setup_args)
        not_merged_setup_args = {
            "test_name": f"{model_arch}_{use_cache}_False",
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": False,
        }
        self._setup(not_merged_setup_args)

        inputs = self.get_inputs(model_arch, for_generation=True)
        model_merged = self.get_onnx_model(**merged_setup_args)
        model_not_merged = self.get_onnx_model(**not_merged_setup_args)
        self.check_onnx_model_attributes(model_merged, use_cache=use_cache, use_merged=True)
        self.check_onnx_model_attributes(model_not_merged, use_cache=use_cache, use_merged=False)

        set_seed(SEED)
        outputs_model_merged = model_merged.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        outputs_model_not_merged = model_not_merged.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged)

    # NUMERICAL CONSISTENCY WITH IOBINDING
    def _test_compare_logits_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool = True, use_merged: bool = False
    ):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(setup_args)

        inputs = self.get_inputs(model_arch)
        onnx_model = self.get_onnx_model(**setup_args, use_io_binding=False)
        io_model = self.get_onnx_model(**setup_args, use_io_binding=True)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged, use_io_binding=False)
        self.check_onnx_model_attributes(io_model, use_cache=use_cache, use_merged=use_merged, use_io_binding=True)

        onnx_outputs = onnx_model(**inputs, use_cache=use_cache)
        io_outputs = io_model(**inputs, use_cache=use_cache)
        self.compare_logits(model_arch, onnx_outputs, io_outputs, use_cache=use_cache)

    def _test_compare_generation_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool = True, use_merged: bool = False
    ):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "model_arch": model_arch,
            "use_merged": use_merged,
        }
        self._setup(setup_args)

        inputs = self.get_inputs(model_arch, for_generation=True)
        onnx_model = self.get_onnx_model(**setup_args, use_io_binding=False)
        io_model = self.get_onnx_model(**setup_args, use_io_binding=True)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged, use_io_binding=False)
        self.check_onnx_model_attributes(io_model, use_cache=use_cache, use_merged=use_merged, use_io_binding=True)

        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        io_outputs = io_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(onnx_outputs, io_outputs)

    # NUMERICAL CONSISTENCY WITH PAST KEY VALUES
    def _test_compare_generation_with_and_without_past_key_values(self, model_arch: str, use_merged: bool = False):
        with_pkv_setup_args = {
            "test_name": f"{model_arch}_True_{use_merged}",
            "model_arch": model_arch,
            "use_merged": use_merged,
            "use_cache": True,
        }
        self._setup(with_pkv_setup_args)
        without_pkv_setup_args = {
            "test_name": f"{model_arch}_False_{use_merged}",
            "model_arch": model_arch,
            "use_merged": use_merged,
            "use_cache": False,
        }
        self._setup(without_pkv_setup_args)

        inputs = self.get_inputs(model_arch, for_generation=True)
        model_with_pkv = self.get_onnx_model(**with_pkv_setup_args)
        model_without_pkv = self.get_onnx_model(**without_pkv_setup_args)
        self.check_onnx_model_attributes(model_with_pkv, use_cache=True, use_merged=use_merged)
        self.check_onnx_model_attributes(model_without_pkv, use_cache=False, use_merged=use_merged)

        set_seed(SEED)
        outputs_with_pkv = model_with_pkv.generate(**inputs, **self.GEN_KWARGS, use_cache=True)
        set_seed(SEED)
        outputs_without_pkv = model_without_pkv.generate(**inputs, **self.GEN_KWARGS, use_cache=False)
        torch.testing.assert_close(outputs_with_pkv, outputs_without_pkv)


class ORTModelForSeq2SeqLMIntegrationTest(ORTSeq2SeqTestMixin):
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

    TASK = "text2text-generation"
    ORTMODEL_CLASS = ORTModelForSeq2SeqLM
    AUTOMODEL_CLASS = AutoModelForSeq2SeqLM

    # UTILITIES
    def get_tokenizer(self, model_arch: str):
        model_id = MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

    def get_inputs(self, model_arch: str, for_generation: bool = False, for_pipeline: bool = False):
        set_seed(SEED)
        texts = ["This is me", "Today is a nice day and I am longer"]

        if for_pipeline:
            return texts

        tokenizer = self.get_tokenizer(model_arch)
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        if for_generation and is_transformers_version(">=", "4.51.0"):
            inputs["use_model_defaults"] = False
        if not for_generation:
            size = (next(iter(inputs.values())).shape[0], 10)
            inputs["decoder_input_ids"] = torch.randint(0, 100, size)

        return inputs

    def get_transformers_model(self, model_arch: str, use_cache: bool = True, **kwargs):
        set_seed(SEED)
        if model_arch.startswith("encoder-decoder"):
            # EnocderDecoderModel does not take `use_cache` during instantiation
            model = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch]).eval()
            model.decoder.config.use_cache = use_cache
        else:
            model = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], use_cache=use_cache).eval()

        if model_arch == "bigbird_pegasus":
            # bigbird_pegasus is exported with original_full attention to avoid
            # issues with switching attention type inbetween inference iterations
            model.model.encoder.set_attention_type("original_full")

        if model_arch == "encoder-decoder-bert-bert":
            # The encoder-decoder-bert-bert model is missing these attributes
            model.generation_config.decoder_start_token_id = 1

        if model_arch == "encoder-decoder":
            # EncoderDecoderModel does not implement the `_reorder_cache` method
            # So we use the one defined in the ORTModelForSeq2SeqLM class
            model._reorder_cache = self.ORTMODEL_CLASS._reorder_cache

        if model_arch == "m2m_100":
            # madness -_-, I spent 2 days trying to figure out why the sequences didn't
            # even when the logits and scores and past key values were all matching,
            # Apparently the M2M100 model consumes the random state during the forward pass
            # because of an ungarded call to `torch.rand()` (even with eval() mode).
            def rng_preserving_wrapper(func):
                def rng_preserving_func(*args, **kwargs):
                    with torch.random.fork_rng():
                        return func(*args, **kwargs)

                return rng_preserving_func

            model.model.encoder.forward = rng_preserving_wrapper(model.model.encoder.forward)
            model.model.decoder.forward = rng_preserving_wrapper(model.model.decoder.forward)

        return model

    def get_onnx_model(
        self,
        test_name: str,
        model_arch: str,
        use_cache: bool = True,
        use_merged: Optional[bool] = None,
        use_io_binding: Optional[bool] = None,
    ):
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged, use_io_binding=use_io_binding
        )

        if model_arch == "encoder-decoder-bert-bert":
            # The encoder-decoder-bert-bert model is missing these attributes
            onnx_model.generation_config.decoder_start_token_id = 1

        return onnx_model

    # INTEGRATION TESTS
    def test_find_untested_architectures(self):
        self._test_find_untested_architectures()

    def test_load_vanilla_transformers_which_is_not_supported(self):
        self._test_load_vanilla_transformers_which_is_not_supported()

    @parameterized.expand(
        grid_parameters({"model_arch": ["t5"], "use_cache": [False, True], "use_merged": [False, True]})
    )
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_load_model_with_external_data(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = self.ORTMODEL_CLASS.from_pretrained(
                MODEL_NAMES[model_arch], use_cache=use_cache, use_merged=use_merged, export=True
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

    def test_load_model_from_hub(self):
        # TODO: create, export and push a tiny random t5 model to the hub
        # one merged, one not merged, and one without cache support
        # see test_decoder.py for an example
        pass

    def test_trust_remote_code(self):
        # TODO: create a t5 model with custom remote code and test it
        # see test_decoder.py for an example
        pass

    def test_load_model_from_hub_infer_onnx_model(self):
        # TODO: add a test for the different arguments of from_pretrained like subfolder, revision, filename, etc.
        # see test_decoder.py for an example
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

        self._test_compare_logits_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_generation_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        if model_arch == "encoder-decoder-bert-bert":
            if use_merged:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
                )
            elif use_cache:
                # The encoder-decoder-bert-bert model does not support using pkv cache,
                # so we test it with use_cache=False instead.
                use_cache = False

        self._test_compare_generation_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Beam search generation is slow without pkv, and we do compare with/without pkv in a different test
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
                # The encoder-decoder-bert-bert model does not support using pkv cache,
                # so we test it with use_cache=False instead.
                use_cache = False

        self._test_compare_beam_search_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # NUMERICAL CONSISTENCY WITH PAST KEY VALUES
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_merged": [False, True]}))
    def test_compare_generation_with_and_without_past_key_values(
        self, test_name: str, model_arch: str, use_merged: bool
    ):
        if model_arch == "encoder-decoder-bert-bert":
            pytest.skip(
                "The encoder-decoder-bert-bert model does not support returning past key values (use_cache=True)."
            )

        self._test_compare_generation_with_and_without_past_key_values(model_arch=model_arch, use_merged=use_merged)

    # NUMERICAL CONSISTENCY WITH DECODER MERGING
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_logits_merged_and_not_merged(self, test_name: str, model_arch: str, use_cache: bool):
        if model_arch == "encoder-decoder-bert-bert":
            pytest.skip(
                "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
            )

        self._test_compare_logits_merged_and_not_merged(model_arch=model_arch, use_cache=use_cache)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_generation_merged_and_not_merged(self, test_name: str, model_arch: str, use_cache: bool):
        if model_arch == "encoder-decoder-bert-bert":
            pytest.skip(
                "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
            )

        self._test_compare_generation_merged_and_not_merged(model_arch=model_arch, use_cache=use_cache)

    # NUMERICAL CONSISTENCY WITH IO BINDING
    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False], "use_merged": [False, True]}
        )
    )
    def test_compare_logits_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        if model_arch == "encoder-decoder-bert-bert":
            if use_merged:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support merging decoders (because there's only one)."
                )
            elif use_cache:
                pytest.skip(
                    "The encoder-decoder-bert-bert model does not support returning past key values (use_cache=True)."
                )

        self._test_compare_logits_with_and_without_io_binding(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_generation_with_and_without_io_binding(
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

        self._test_compare_generation_with_and_without_io_binding(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # PIPELINE TESTS
    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"use_cache": [True], "use_merged": [False, True]}))
    def test_pipeline_with_default_model(self, test_name: str, use_cache: bool, use_merged: bool):
        texts = self.get_inputs("t5", for_pipeline=True)

        # Text2Text generation
        pipe = pipeline("text2text-generation", model_kwargs={"use_cache": use_cache, "use_merged": use_merged})
        self.check_onnx_model_attributes(pipe.model, use_cache=use_cache, use_merged=use_merged)
        set_seed(SEED)
        outputs = pipe(texts, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertGreater(len(outputs[0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline(
                "text2text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache, "use_merged": use_merged}
            )
            self.check_onnx_model_attributes(pipe.model, use_cache=use_cache, use_merged=use_merged)
            set_seed(SEED)
            local_outputs = pipe(texts, **self.GEN_KWARGS)
            self.assertEqual(outputs, local_outputs)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"model_arch": ["t5"], "use_cache": [True], "use_merged": [False, True]}))
    def test_pipeline_with_onnx_model(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "use_merged": use_merged,
            "model_arch": model_arch,
        }
        self._setup(setup_args)

        tokenizer = self.get_tokenizer(model_arch)
        texts = self.get_inputs(model_arch, for_pipeline=True)
        onnx_model = self.get_onnx_model(**setup_args)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged)

        # Text2Text generation
        pipe = pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer)
        set_seed(SEED)
        outputs = pipe(texts, **self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("generated_text", outputs[0])
        self.assertIsInstance(outputs[0]["generated_text"], str)
        self.assertGreater(len(outputs[0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline(
                "text2text-generation", model=tmpdir, model_kwargs={"use_cache": use_cache, "use_merged": use_merged}
            )
            self.check_onnx_model_attributes(pipe.model, use_cache=use_cache, use_merged=use_merged)
            set_seed(SEED)
            outputs_local_model = pipe(texts, **self.GEN_KWARGS)
            self.assertEqual(outputs, outputs_local_model)

    # OLD SEQ2SEQ ONNX MODEL TESTS
    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"use_cache": [True]}))
    def test_inference_old_onnx_model(self, test_name: str, use_cache: bool):
        model = self.AUTOMODEL_CLASS.from_pretrained("t5-small").eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained("optimum/t5-small", use_cache=use_cache)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache)

        inputs = self.get_inputs("t5")
        with torch.no_grad():
            outputs = model(**inputs, use_cache=use_cache)
        onnx_outputs = onnx_model(**inputs, use_cache=use_cache)
        self.compare_logits("t5", outputs, onnx_outputs, use_cache=use_cache)

        inputs = self.get_inputs("t5", for_generation=True)
        set_seed(SEED)
        outputs = model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(outputs, onnx_outputs)


class ORTModelForSpeechSeq2SeqIntegrationTest(ORTSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "speech_to_text",
        "whisper",
    ]

    if is_transformers_version(">=", str(MoonshineOnnxConfig.MIN_TRANSFORMERS_VERSION)):
        SUPPORTED_ARCHITECTURES.append("moonshine")

    TASK = "automatic-speech-recognition"
    ORTMODEL_CLASS = ORTModelForSpeechSeq2Seq
    AUTOMODEL_CLASS = AutoModelForSpeechSeq2Seq

    MODEL_ATOL = {  # noqa: RUF012
        "moonshine": 1e-2,  # Moonshine model has a lot of numerical noise from the convolutional layers
    }
    MODEL_RTOL = {  # noqa: RUF012
        "moonshine": 1e-2,  # Moonshine model has a lot of numerical noise from the convolutional layers
    }

    def get_tokenizer(self, model_arch: str):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_arch])
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            else:
                raise ValueError(
                    f"Tokenizer for model {MODEL_NAMES[model_arch]} does not have a defined `pad_token`, `eos_token`, or `bos_token`."
                )
        return tokenizer

    def get_feature_extractor(self, model_arch: str):
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAMES[model_arch])
        return feature_extractor

    def get_inputs(
        self, model_arch: str, for_generation: bool = False, for_pipeline: bool = False, batched: bool = True
    ):
        set_seed(SEED)
        if batched:
            audios = [np.random.randn(5 * 16000), np.random.randn(60 * 16000)]  # 5 seconds, 60 seconds
        else:
            audios = np.random.randn(5 * 16000)

        if for_pipeline:
            return audios

        feature_extractor = self.get_feature_extractor(model_arch)

        if model_arch == "whisper":
            inputs = feature_extractor(audios, return_tensors="pt", return_attention_mask=True)
            if inputs.input_features.shape[-1] < 3000:
                inputs = feature_extractor(audios, return_tensors="pt", padding=True, return_attention_mask=True)
            elif inputs.input_features.shape[-1] > 3000:
                inputs = feature_extractor(audios, return_tensors="pt", truncation=True, return_attention_mask=True)
        else:
            inputs = feature_extractor(audios, return_tensors="pt", padding=True, return_attention_mask=True)

        if for_generation and is_transformers_version(">=", "4.51.0"):
            inputs["use_model_defaults"] = False
        if not for_generation:
            size = (next(iter(inputs.values())).shape[0], 10)
            inputs["decoder_input_ids"] = torch.randint(0, 100, size)

        return inputs

    def get_transformers_model(self, model_arch: str, use_cache: bool = True, **kwargs):
        set_seed(SEED)
        model = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch], use_cache=use_cache).eval()
        return model

    def get_onnx_model(
        self,
        test_name: str,
        use_cache: bool = True,
        use_merged: Optional[bool] = None,
        use_io_binding: Optional[bool] = None,
        **kwargs,
    ):
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged, use_io_binding=use_io_binding
        )
        return onnx_model

    # INTEGRATION TESTS
    # The task automatic-speech-recognition contains models like hubert, mctct, sew, etc. that are not supported by the
    # ORTForSpeechSeq2Seq class, but rather by the ORTModelForCTC class.
    # def test_find_untested_architectures(self):
    #     self._test_find_untested_architectures()

    def test_load_vanilla_transformers_which_is_not_supported(self):
        self._test_load_vanilla_transformers_which_is_not_supported()

    # NUMERICAL CONSISTENCY WITH TRANSFORMERS
    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False], "use_merged": [False, True]}
        )
    )
    def test_compare_logits_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        self._test_compare_logits_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_generation_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_generation_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Beam search generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_beam_search_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_beam_search_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # NUMERICAL CONSISTENCY WITH DECODER MERGING
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_logits_merged_and_not_merged(self, test_name: str, model_arch: str, use_cache: bool):
        self._test_compare_logits_merged_and_not_merged(model_arch=model_arch, use_cache=use_cache)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_generation_merged_and_not_merged(self, test_name: str, model_arch: str, use_cache: bool):
        self._test_compare_generation_merged_and_not_merged(model_arch=model_arch, use_cache=use_cache)

    # NUMERICAL CONSISTENCY WITH AND WITHOUT PAST KEY VALUES
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_merged": [False, True]}))
    def test_compare_generation_with_and_without_past_key_values(
        self, test_name: str, model_arch: str, use_merged: bool
    ):
        self._test_compare_generation_with_and_without_past_key_values(model_arch=model_arch, use_merged=use_merged)

    # NUMERICAL CONSISTENCY WITH IO BINDING
    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False], "use_merged": [False, True]}
        )
    )
    def test_compare_logits_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_logits_with_and_without_io_binding(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_generation_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_generation_with_and_without_io_binding(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # PIPELINE TESTS
    @parameterized.expand(grid_parameters({"use_cache": [True], "use_merged": [False, True]}))
    def test_pipeline_with_default_model(self, test_name: str, use_cache: bool, use_merged: bool):
        pytest.skip("Skipping because the default model for ASR in pipelines is wav2vec2, which is a CTC model.")

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": ["whisper"], "use_cache": [True], "use_merged": [False, True]})
    )
    def test_pipeline_with_onnx_model(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "use_merged": use_merged,
            "model_arch": model_arch,
        }
        self._setup(setup_args)

        tokenizer = self.get_tokenizer(model_arch)
        feature_extractor = self.get_feature_extractor(model_arch)
        onnx_model = self.get_onnx_model(**setup_args)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged)
        audios = self.get_inputs(model_arch, for_pipeline=True)

        # Automatic Speech Recognition
        pipe = pipeline(
            "automatic-speech-recognition", model=onnx_model, tokenizer=tokenizer, feature_extractor=feature_extractor
        )
        set_seed(SEED)
        outputs = pipe(audios, generate_kwargs=self.GEN_KWARGS, return_timestamps=True)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0], dict)
        self.assertIn("text", outputs[0])
        self.assertIsInstance(outputs[0]["text"], str)
        self.assertGreater(len(outputs[0]["text"]), 0)
        self.assertIn("chunks", outputs[0])
        self.assertIsInstance(outputs[0]["chunks"], list)
        self.assertGreater(len(outputs[0]["chunks"]), 0)

        if not hasattr(pipe, "image_processor"):
            # Error in pipelines in transformers 4.36
            pipe.image_processor = None

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=tmpdir,
                model_kwargs={"use_cache": use_cache, "use_merged": use_merged},
            )
            self.check_onnx_model_attributes(pipe.model, use_cache=use_cache, use_merged=use_merged)
            set_seed(SEED)
            outputs_local_model = pipe(audios, generate_kwargs=self.GEN_KWARGS, return_timestamps=True)
            self.assertEqual(outputs, outputs_local_model)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"use_cache": [True]}))
    def test_inference_old_onnx_model(self, test_name: str, use_cache: bool):
        model = self.AUTOMODEL_CLASS.from_pretrained("openai/whisper-tiny.en").eval()
        onnx_model = self.ORTMODEL_CLASS.from_pretrained("optimum/whisper-tiny.en", use_cache=use_cache)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=False)

        # TODO: the optimum model doesn't output the right logits for padding tokens,
        # we should probably update it with the newest version of optimum
        inputs = self.get_inputs("whisper", batched=False)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=use_cache)
        onnx_outputs = onnx_model(**inputs, use_cache=use_cache)
        self.compare_logits("whisper", outputs, onnx_outputs, use_cache=use_cache)

        inputs = self.get_inputs("whisper", for_generation=True)
        set_seed(SEED)
        outputs = model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        set_seed(SEED)
        onnx_outputs = onnx_model.generate(**inputs, **self.GEN_KWARGS, use_cache=use_cache)
        torch.testing.assert_close(outputs, onnx_outputs)


class ORTModelForVision2SeqIntegrationTest(ORTSeq2SeqTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "pix2struct",
        "vision-encoder-decoder",
        "vision-encoder-decoder-donut",
        "vision-encoder-decoder-trocr",
    ]

    TASK = "image-to-text"
    ORTMODEL_CLASS = ORTModelForVision2Seq
    AUTOMODEL_CLASS = AutoModelForVision2Seq

    def get_tokenizer(self, model_arch: str):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAMES[model_arch])
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            else:
                raise ValueError(
                    f"Tokenizer for model {MODEL_NAMES[model_arch]} does not have a defined `pad_token`, `eos_token`, or `bos_token`."
                )
        return tokenizer

    def get_image_processor(self, model_arch: str):
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAMES[model_arch])
        return image_processor

    def get_inputs(self, model_arch: str, for_generation: bool = False, for_pipeline: bool = False):
        set_seed(SEED)
        images = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(2)]

        if for_pipeline:
            return [Image.fromarray((image * 255).astype(np.uint8)) for image in images]

        image_processor = self.get_image_processor(model_arch)

        inputs = image_processor(images, return_tensors="pt")

        if for_generation and is_transformers_version(">=", "4.51.0"):
            inputs["use_model_defaults"] = False
        if not for_generation:
            size = (next(iter(inputs.values())).shape[0], 10)
            inputs["decoder_input_ids"] = torch.randint(0, 100, size)

        return inputs

    def get_transformers_model(self, model_arch: str, use_cache: bool = True, **kwargs):
        set_seed(SEED)
        # vision-encoder-decoders and pix2struct models do not support use_cache=True at instantiation
        model = self.AUTOMODEL_CLASS.from_pretrained(MODEL_NAMES[model_arch]).eval()
        model.decoder.config.use_cache = use_cache

        if model_arch == "vision-encoder-decoder":
            # VisionEncoderDecoderModel does not implement the `_reorder_cache` method
            # So we use the one defined in the ORT class
            model._reorder_cache = self.ORTMODEL_CLASS._reorder_cache

        if model_arch == "pix2struct" and is_transformers_version("<", "4.50.0"):
            # Pix2StructModel does not implement the `_reorder_cache` method in transformers < 4.50.0
            # So we use the one defined in the ORT class
            model._reorder_cache = self.ORTMODEL_CLASS._reorder_cache

        return model

    def get_onnx_model(
        self,
        test_name: str,
        use_cache: bool = True,
        use_merged: Optional[bool] = None,
        use_io_binding: Optional[bool] = None,
        **kwargs,
    ):
        onnx_model = self.ORTMODEL_CLASS.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged, use_io_binding=use_io_binding
        )
        return onnx_model

    # INTEGRATION TESTS
    def test_find_untested_architectures(self):
        self._test_find_untested_architectures()

    def test_load_vanilla_transformers_which_is_not_supported(self):
        self._test_load_vanilla_transformers_which_is_not_supported()

    # NUMERICAL CONSISTENCY WITH TRANSFORMERS
    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False], "use_merged": [False, True]}
        )
    )
    def test_compare_logits_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        self._test_compare_logits_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_generation_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_generation_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Beam search generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_beam_search_to_transformers(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_beam_search_to_transformers(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # NUMERICAL CONSISTENCY WITH DECODER MERGING
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False]}))
    def test_compare_logits_merged_and_not_merged(self, test_name: str, model_arch: str, use_cache: bool):
        self._test_compare_logits_merged_and_not_merged(model_arch=model_arch, use_cache=use_cache)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_generation_merged_and_not_merged(self, test_name: str, model_arch: str, use_cache: bool):
        self._test_compare_generation_merged_and_not_merged(model_arch=model_arch, use_cache=use_cache)

    # NUMERICAL CONSISTENCY WITH AND WITHOUT PAST KEY VALUES
    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_merged": [False, True]}))
    def test_compare_generation_with_and_without_past_key_values(
        self, test_name: str, model_arch: str, use_merged: bool
    ):
        self._test_compare_generation_with_and_without_past_key_values(model_arch=model_arch, use_merged=use_merged)

    # NUMERICAL CONSISTENCY WITH IO BINDING
    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True, False], "use_merged": [False, True]}
        )
    )
    def test_compare_logits_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_logits_with_and_without_io_binding(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    def test_compare_generation_with_and_without_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool
    ):
        self._test_compare_generation_with_and_without_io_binding(
            test_name=test_name, model_arch=model_arch, use_cache=use_cache, use_merged=use_merged
        )

    # PIPELINE TESTS
    @parameterized.expand(grid_parameters({"use_cache": [True], "use_merged": [False, True]}))
    def test_pipeline_with_default_model(self, test_name: str, use_cache: bool, use_merged: bool):
        images = self.get_inputs("vision-encoder-decoder", for_pipeline=True)

        # Image-to-Text generation
        pipe = pipeline("image-to-text", model_kwargs={"use_cache": use_cache, "use_merged": use_merged})
        self.check_onnx_model_attributes(pipe.model, use_cache=use_cache, use_merged=use_merged)
        set_seed(SEED)
        outputs = pipe(images, generate_kwargs=self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0][0], dict)
        self.assertIn("generated_text", outputs[0][0])
        self.assertIsInstance(outputs[0][0]["generated_text"], str)
        self.assertGreater(len(outputs[0][0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe = pipeline(
                "image-to-text", model=tmpdir, model_kwargs={"use_cache": use_cache, "use_merged": use_merged}
            )
            self.check_onnx_model_attributes(pipe.model, use_cache=use_cache, use_merged=use_merged)
            set_seed(SEED)
            local_outputs = pipe(images, generate_kwargs=self.GEN_KWARGS)
            self.assertEqual(outputs, local_outputs)

    # Generation is slow without pkv, and we do compare with/without pkv in a different test
    @parameterized.expand(
        grid_parameters({"model_arch": ["vision-encoder-decoder"], "use_cache": [True], "use_merged": [False, True]})
    )
    def test_pipeline_with_onnx_model(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        setup_args = {
            "test_name": test_name,
            "use_cache": use_cache,
            "use_merged": use_merged,
            "model_arch": model_arch,
        }
        self._setup(setup_args)

        tokenizer = self.get_tokenizer(model_arch)
        image_processor = self.get_image_processor(model_arch)
        onnx_model = self.get_onnx_model(**setup_args)
        self.check_onnx_model_attributes(onnx_model, use_cache=use_cache, use_merged=use_merged)
        images = self.get_inputs(model_arch, for_pipeline=True)

        # Image-to-Text generation
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            feature_extractor=image_processor,
        )
        set_seed(SEED)
        outputs = pipe(images, generate_kwargs=self.GEN_KWARGS)
        self.assertIsInstance(outputs, list)
        self.assertIsInstance(outputs[0][0], dict)
        self.assertIn("generated_text", outputs[0][0])
        self.assertIsInstance(outputs[0][0]["generated_text"], str)
        self.assertGreater(len(outputs[0][0]["generated_text"]), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)

            pipe = pipeline(
                "image-to-text", model=tmpdir, model_kwargs={"use_cache": use_cache, "use_merged": use_merged}
            )
            self.check_onnx_model_attributes(pipe.model, use_cache=use_cache, use_merged=use_merged)
            set_seed(SEED)
            outputs_local_model = pipe(images, generate_kwargs=self.GEN_KWARGS)
            self.assertEqual(outputs, outputs_local_model)
