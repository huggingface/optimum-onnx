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
import gc
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import requests
import torch
from parameterized import parameterized
from PIL import Image
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import (
    AutoImageProcessor,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    AutoTokenizer,
    Pix2StructForConditionalGeneration,  # Pix2Struct does not work with AutoModel
    PretrainedConfig,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import require_torch_gpu

from optimum.exporters import TasksManager
from optimum.exporters.onnx import main_export
from optimum.onnx.utils import has_onnx_input
from optimum.onnxruntime import (
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    ORTModelForPix2Struct,
    ORTModelForSeq2SeqLM,
    ORTModelForSpeechSeq2Seq,
    ORTModelForVision2Seq,
    pipeline,
)
from optimum.onnxruntime.modeling_seq2seq import ORTDecoderForSeq2Seq, ORTEncoder
from optimum.utils import logging
from optimum.utils.save_utils import maybe_load_preprocessors
from optimum.utils.testing_utils import grid_parameters, require_ort_rocm


logger = logging.get_logger()


class ORTModelForSeq2SeqLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "bart",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        "encoder-decoder",
        "longt5",
        "m2m_100",
        "marian",
        "mbart",
        "mt5",
        "pegasus",
        "t5",
    ]

    FULL_GRID = {  # noqa: RUF012
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForSeq2SeqLM
    TASK = "text2text-generation"

    GENERATION_LENGTH = 100

    def _get_model_ids(self, model_arch):
        model_ids = MODEL_NAMES[model_arch]
        if isinstance(model_ids, dict):
            model_ids = list(model_ids.keys())
        else:
            model_ids = [model_ids]
        return model_ids

    def _get_onnx_model_dir(self, model_id, model_arch, test_name):
        onnx_model_dir = self.onnx_model_dirs[test_name]
        if isinstance(MODEL_NAMES[model_arch], dict):
            onnx_model_dir = onnx_model_dir[model_id]

        return onnx_model_dir

    @parameterized.expand([(True,)])  # old exported model ouputs gibberish when use_cache=False
    @pytest.mark.run_in_series
    def test_inference_old_seq2seq_onnx_model(self, use_cache):
        tokenizer = get_preprocessor("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            "optimum/t5-small", use_cache=use_cache, use_io_binding=False, use_merged=False
        )

        self.assertEqual(onnx_model.use_cache, use_cache)
        self.assertEqual(onnx_model.encoder.path.name, ONNX_ENCODER_NAME)
        self.assertEqual(onnx_model.decoder.path.name, ONNX_DECODER_NAME)
        if use_cache:
            self.assertEqual(onnx_model.decoder_with_past.path.name, ONNX_DECODER_WITH_PAST_NAME)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")

        onnx_outputs = onnx_model.generate(**tokens, min_new_tokens=30, max_new_tokens=30, do_sample=False)
        outputs = model.generate(**tokens, min_new_tokens=30, max_new_tokens=30, do_sample=False)
        onnx_text_outputs = tokenizer.decode(onnx_outputs[0], skip_special_tokens=True)
        text_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertEqual(onnx_text_outputs, text_outputs)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and use_cache is True
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)
            text = "This is a sample output"
            tokens = tokenizer(text, return_tensors="pt")

            # General case
            outputs = model.generate(**tokens)
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.assertIsInstance(res[0], str)

            # With input ids
            outputs = model.generate(input_ids=tokens["input_ids"])
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_transformers_and_save(self, model_arch):
        if "text2text-generation-with-past" not in TasksManager.get_supported_tasks_for_model_type(
            model_arch, exporter="onnx", library_name="transformers"
        ):
            self.skipTest("Unsupported -with-past export case")

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_merged=True is not supported for bert as a decoder")
                continue

            model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True, use_merged=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_pretrained(tmpdir)
                save_path = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
                self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

                folder_contents = os.listdir(tmpdir)
                self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
                self.assertTrue(ONNX_DECODER_NAME not in folder_contents)
                self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch):
        task = "text2text-generation-with-past"

        if task not in TasksManager.get_supported_tasks_for_model_type(model_arch, exporter="onnx"):
            self.skipTest("Unsupported export case", library_name="transformers")

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_merged=True is not supported for bert as a decoder")
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                main_export(model_id, tmpdir, task=task)

                model = ORTModelForSeq2SeqLM.from_pretrained(tmpdir)

                self.assertTrue(model.use_merged)
                self.assertTrue(model.decoder_with_past is None)

                model.save_pretrained(tmpdir + "_save")
                save_path = os.path.join(tmpdir + "_save", ONNX_DECODER_MERGED_NAME)
                self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

                folder_contents = os.listdir(tmpdir + "_save")
                self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
                self.assertFalse(ONNX_DECODER_NAME in folder_contents)
                self.assertFalse(ONNX_DECODER_WITH_PAST_NAME in folder_contents)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and use_cache is True
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            self.assertIsInstance(onnx_model.encoder, ORTEncoder)
            if use_merged is False:
                model_path = Path(onnx_model_dir, ONNX_DECODER_NAME)
                self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
                self.assertEqual(onnx_model.use_merged, False)
            else:
                model_path = Path(onnx_model_dir, ONNX_DECODER_MERGED_NAME)
                self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
                self.assertEqual(onnx_model.use_merged, True)

            self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
            if onnx_model.use_cache is True and onnx_model.use_merged is False:
                self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
            if onnx_model.use_cache is True and onnx_model.use_merged is True:
                self.assertTrue(onnx_model.decoder_with_past is None)

            self.assertIsInstance(onnx_model.config, PretrainedConfig)

            set_seed(SEED)
            transformers_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            tokenizer = get_preprocessor(model_id)
            inputs = "This is a sample output"
            tokens = tokenizer(inputs, return_tensors="pt", padding=True)
            decoder_start_token_id = transformers_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id

            decoder_inputs = {"decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id}

            with torch.no_grad():
                transformers_outputs = transformers_model(**tokens, **decoder_inputs)

            for input_type in ["pt", "np"]:
                tokens = tokenizer(inputs, return_tensors=input_type, padding=True)

                if input_type == "np":
                    decoder_inputs = {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id}

                onnx_outputs = onnx_model(**tokens, **decoder_inputs)

                self.assertTrue("logits" in onnx_outputs)
                self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

                # Compare tensor outputs
                torch.testing.assert_close(
                    torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
                )

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_text_generation(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and use_cache is True
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)

            decoder_start_token_id = onnx_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id

            # Text2Text generation
            pipe = pipeline("text2text-generation", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id, min_new_tokens=10, max_new_tokens=10)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["generated_text"], str)

            # Summarization
            pipe = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id, min_new_tokens=10, max_new_tokens=10)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["summary_text"], str)

            # Translation
            pipe = pipeline("translation_en_to_de", model=onnx_model, tokenizer=tokenizer)
            text = "This is a test"
            outputs = pipe(text, decoder_start_token_id=decoder_start_token_id, min_new_tokens=10, max_new_tokens=10)
            self.assertEqual(pipe.device, onnx_model.device)
            self.assertIsInstance(outputs[0]["translation_text"], str)

            if model_arch == "t5":
                with tempfile.TemporaryDirectory() as tmpdir:
                    pipe.save_pretrained(tmpdir)
                    model_kwargs = {"use_cache": use_cache}
                    pipe = pipeline(
                        "translation_en_to_de",
                        model=tmpdir,
                        model_kwargs=model_kwargs,
                        accelerator="ort",
                    )
                    outputs_local_model = pipe(text, min_new_tokens=10, max_new_tokens=10)
                    self.assertEqual(outputs[0]["translation_text"], outputs_local_model[0]["translation_text"])

        gc.collect()

    def test_load_pipeline(self):
        pipe = pipeline(
            "text2text-generation",
            model="echarlaix/t5-small-onnx",
            accelerator="ort",
        )
        outputs = pipe("this is an example input")
        self.assertIsInstance(outputs[0]["generated_text"], str)

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        # Text2text generation
        pipe = pipeline("text2text-generation")
        text = "This is a test"
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["generated_text"], str)

        # Summarization
        pipe = pipeline("summarization")
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["summary_text"], str)

        # Translation
        pipe = pipeline("translation_en_to_de")
        outputs = pipe(text, min_length=1, max_length=2)
        # compare model output class
        self.assertIsInstance(outputs[0]["translation_text"], str)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder"
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)
            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=False, device=0
            )
            text = "My Name is Philipp and i live"
            outputs = pipe(text, max_length=2 * len(text) + 1)
            # check model device
            self.assertEqual(pipe.model.device.type.lower(), "cuda")
            # compare model output class
            self.assertTrue(isinstance(outputs[0]["translation_text"], str))

            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=True, device=0
            )

            outputs = pipe(text, min_length=len(text) + 1, max_length=2 * len(text) + 1)
            self.assertTrue(isinstance(outputs[0]["translation_token_ids"], torch.Tensor))
            self.assertTrue(len(outputs[0]["translation_token_ids"]) > len(text))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder"
                continue

            onnx_model_dir = self._get_onnx_model_dir(model_id, model_arch, test_name)
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(onnx_model_dir, use_cache=use_cache)

            tokenizer = get_preprocessor(model_id)
            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=False, device=0
            )
            text = "My Name is Philipp and i live"
            outputs = pipe(text, max_length=2 * len(text) + 1)
            # check model device
            self.assertEqual(pipe.model.device.type.lower(), "cuda")
            # compare model output class
            self.assertTrue(isinstance(outputs[0]["translation_text"], str))

            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=True, device=0
            )

            outputs = pipe(text, min_length=len(text) + 1, max_length=2 * len(text) + 1)
            self.assertTrue(isinstance(outputs[0]["translation_token_ids"], torch.Tensor))
            self.assertTrue(len(outputs[0]["translation_token_ids"]) > len(text))

    # TRT EP compile time can be long, so we don't test all archs
    @parameterized.expand(grid_parameters({"model_arch": ["t5"], "use_cache": [True, False]}))
    @require_torch_gpu
    @pytest.mark.trt_ep_test
    def test_pipeline_on_trt_execution_provider(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        with tempfile.TemporaryDirectory() as engine_cache_dir:
            provider_options = {"trt_engine_cache_enable": True, "trt_engine_cache_path": engine_cache_dir}

            model_id = MODEL_NAMES[model_arch]
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self.onnx_model_dirs[test_name],
                provider="TensorrtExecutionProvider",
                provider_options=provider_options,
                use_cache=use_cache,
            )

            tokenizer = get_preprocessor(model_id)

            decoder_inputs = {
                "decoder_input_ids": torch.ones((1, 1), dtype=torch.long) * onnx_model.config.decoder_start_token_id
            }

            # build engine for a short sequence
            text = ["short"]
            encoded_input = tokenizer(text, return_tensors="pt").to("cuda")
            _ = onnx_model(**encoded_input, **decoder_inputs)

            # build engine for a long sequence
            text = [" a very long input just for demo purpose, this is very long" * 10]
            encoded_input = tokenizer(text, return_tensors="pt").to("cuda")
            _ = onnx_model(**encoded_input, **decoder_inputs)

            pipe = pipeline(
                "translation_en_to_de", model=onnx_model, tokenizer=tokenizer, return_tensors=True, device=0
            )
            text = "My Name is Philipp and i live"
            outputs = pipe(text, min_length=len(text) + 1, max_length=2 * len(text) + 1)
            self.assertTrue(isinstance(outputs[0]["translation_token_ids"], torch.Tensor))
            self.assertTrue(len(outputs[0]["translation_token_ids"]) > len(text))

            encoded_input = tokenizer("Please continue this", return_tensors="pt").to("cuda")
            _ = onnx_model.generate(**encoded_input)

            gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        if model_arch == "m2m_100":
            generation_length = 20  # model's predefined maximum length
        else:
            generation_length = self.GENERATION_LENGTH

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            tokenizer = get_preprocessor(model_id)
            text = "This is a sample output"
            tokens = tokenizer(text, return_tensors="pt")
            model_with_pkv = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, model_arch + "_True"), use_cache=True
            )

            outputs_model_with_pkv = model_with_pkv.generate(
                **tokens, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
            )

            model_without_pkv = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, model_arch + "_False"), use_cache=False
            )

            outputs_model_without_pkv = model_without_pkv.generate(
                **tokens, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
            )

            torch.testing.assert_close(
                outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL
            )
            self.assertEqual(outputs_model_with_pkv.shape[1], generation_length + 1)
            self.assertEqual(outputs_model_without_pkv.shape[1], generation_length + 1)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_merged_and_not_merged_models_outputs(self, test_name: str, model_arch: str, use_cache: bool):
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

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            tokenizer = get_preprocessor(model_id)
            text = "My Name is Philipp and i live"
            tokens = tokenizer(text, return_tensors="pt")

            model_not_merged_dir = self._get_onnx_model_dir(model_id, model_arch, test_name + "_False")
            model_merged_dir = self._get_onnx_model_dir(model_id, model_arch, test_name + "_True")

            model_not_merged = ORTModelForSeq2SeqLM.from_pretrained(model_not_merged_dir)
            not_merged_onnx_path = Path(model_not_merged_dir, ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
            self.assertEqual(model_not_merged.use_merged, False)

            model_merged = ORTModelForSeq2SeqLM.from_pretrained(model_merged_dir)
            merged_onnx_path = Path(model_merged_dir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
            self.assertEqual(model_merged.decoder_with_past, None)
            self.assertEqual(model_merged.use_merged, True)

            outputs_model_not_merged = model_not_merged.generate(**tokens)
            outputs_model_merged = model_merged.generate(**tokens)

            torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=False,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )
            io_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=True,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )

            self.assertFalse(onnx_model.use_io_binding)
            self.assertTrue(io_model.use_io_binding)

            tokenizer = get_preprocessor(model_id)
            tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")
            decoder_start_token_id = onnx_model.config.decoder_start_token_id if model_arch != "mbart" else 2
            if model_arch == "encoder-decoder":
                decoder_start_token_id = tokenizer.cls_token_id
            decoder_inputs = {
                "decoder_input_ids": torch.ones((2, 1), dtype=torch.long).to("cuda") * decoder_start_token_id
            }

            onnx_outputs = onnx_model(**tokens, **decoder_inputs)
            io_outputs = io_model(**tokens, **decoder_inputs)

            self.assertTrue("logits" in io_outputs)
            self.assertIsInstance(io_outputs.logits, torch.Tensor)

            # compare tensor outputs
            torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self,
        test_name: str,
        model_arch: str,
        use_cache: bool,
        use_merged: bool,
        num_beams: int,
    ):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }

        self._setup(model_args)

        model_ids = self._get_model_ids(model_arch)
        for model_id in model_ids:
            if (
                model_arch == "encoder-decoder"
                and "text2text-generation-with-past" not in MODEL_NAMES[model_arch][model_id]
            ):
                # The model with use_cache=True is not supported for bert as a decoder")
                continue

            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=False,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )
            io_model = ORTModelForSeq2SeqLM.from_pretrained(
                self._get_onnx_model_dir(model_id, model_arch, test_name),
                use_io_binding=True,
                use_cache=use_cache,
                provider="CUDAExecutionProvider",
            )

            self.assertFalse(onnx_model.use_io_binding)
            self.assertTrue(io_model.use_io_binding)

            tokenizer = get_preprocessor(model_id)
            tokens = tokenizer("This is a sample output", return_tensors="pt").to("cuda")

            onnx_outputs = onnx_model.generate(**tokens, num_beams=num_beams)
            io_outputs = io_model.generate(**tokens, num_beams=num_beams)

            # compare tensor outputs
            torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForSpeechSeq2SeqIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["whisper", "speech_to_text"]  # noqa: RUF012

    FULL_GRID = {  # noqa: RUF012
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForSpeechSeq2Seq
    TASK = "automatic-speech-recognition"

    GENERATION_LENGTH = 100

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 18736), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)

        return audio_data

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_transformers_and_save(self, model_arch):
        if "automatic-speech-recognition-with-past" not in TasksManager.get_supported_tasks_for_model_type(
            model_arch, exporter="onnx", library_name="transformers"
        ):
            self.skipTest("Unsupported -with-past export case")

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, use_merged=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            save_path = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir)
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME not in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        task = "automatic-speech-recognition-with-past"

        if task not in TasksManager.get_supported_tasks_for_model_type(model_arch, exporter="onnx"):
            self.skipTest("Unsupported export case", library_name="transformers")

        with tempfile.TemporaryDirectory() as tmpdir:
            main_export(model_id, tmpdir, task=task)

            model = ORTModelForSpeechSeq2Seq.from_pretrained(tmpdir)

            self.assertTrue(model.use_merged)
            self.assertTrue(model.decoder_with_past is None)

            model.save_pretrained(tmpdir + "_save")
            save_path = os.path.join(tmpdir + "_save", ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir + "_save")
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_WITH_PAST_NAME in folder_contents)

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name])
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        outputs = model.generate(inputs=features["input_features"])
        res = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        if use_merged is False:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, False)
        else:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, True)

        self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is False:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is True:
            self.assertTrue(onnx_model.decoder_with_past is None)

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        features = {
            "np": processor.feature_extractor(data, return_tensors="np"),
            "pt": processor.feature_extractor(data, return_tensors="pt"),
        }

        decoder_start_token_id = transformers_model.config.decoder_start_token_id
        decoder_inputs = {
            "np": {"decoder_input_ids": np.ones((1, 1), dtype=np.int64) * decoder_start_token_id},
            "pt": {"decoder_input_ids": torch.ones((1, 1), dtype=torch.int64) * decoder_start_token_id},
        }

        with torch.no_grad():
            transformers_outputs = transformers_model(**features["pt"], **decoder_inputs["pt"])

        for input_type in ["pt", "np"]:
            onnx_outputs = onnx_model(**features[input_type], **decoder_inputs[input_type])

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        if model_arch == "speech_to_text":
            generation_length = 20
        else:
            generation_length = self.GENERATION_LENGTH

        with torch.no_grad():
            transformers_outputs = transformers_model.generate(
                **features["pt"],
                max_new_tokens=generation_length,
                min_new_tokens=generation_length,
                do_sample=False,
                num_beams=1,
            )

        onnx_outputs = onnx_model.generate(
            **features["pt"],
            max_new_tokens=generation_length,
            min_new_tokens=generation_length,
            do_sample=False,
            num_beams=1,
        )

        torch.testing.assert_close(torch.Tensor(onnx_outputs), transformers_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_speech_recognition(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        tokenizer, _, feature_extractor = maybe_load_preprocessors(model_id)
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_merged=use_merged
        )
        # Speech recogition generation
        pipe = pipeline(
            "automatic-speech-recognition",
            model=onnx_model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        data = self._generate_random_audio_data()
        outputs = pipe(data)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs["text"], str)

        if model_arch == "whisper":
            outputs = pipe(data, return_timestamps=True)
            self.assertTrue("chunks" in outputs)

            outputs = pipe(data, return_timestamps=False)
            self.assertTrue("chunks" not in outputs)

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        processor = get_preprocessor(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=onnx_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0,
        )

        data = self._generate_random_audio_data()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs["text"], str))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        processor = get_preprocessor(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=onnx_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0,
        )

        data = self._generate_random_audio_data()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs["text"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        if model_arch == "speech_to_text":
            generation_length = 20  # maximum length for the model
        else:
            generation_length = self.GENERATION_LENGTH

        model_with_pkv = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        outputs_model_with_pkv = model_with_pkv.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )

        model_without_pkv = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )
        outputs_model_without_pkv = model_without_pkv.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )

        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_merged_and_not_merged_models_outputs(self, test_name: str, model_arch: str, use_cache: bool):
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
        processor = get_preprocessor(model_id)

        data = self._generate_random_audio_data()
        features = processor.feature_extractor(data, return_tensors="pt")

        model_not_merged_dir = self.onnx_model_dirs[test_name + "_False"]
        model_merged_dir = self.onnx_model_dirs[test_name + "_True"]

        model_not_merged = ORTModelForSpeechSeq2Seq.from_pretrained(model_not_merged_dir)
        not_merged_onnx_path = Path(model_not_merged_dir, ONNX_DECODER_NAME)
        self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_not_merged.use_merged, False)

        model_merged = ORTModelForSpeechSeq2Seq.from_pretrained(model_merged_dir)
        merged_onnx_path = Path(model_merged_dir, ONNX_DECODER_MERGED_NAME)
        self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_merged.decoder_with_past, None)
        self.assertEqual(model_merged.use_merged, True)

        generation_length = 10

        outputs_model_not_merged = model_not_merged.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )
        outputs_model_merged = model_merged.generate(
            **features, min_new_tokens=generation_length, max_new_tokens=generation_length, num_beams=1
        )

        torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={
                "cudnn_conv_algo_search": "DEFAULT",
            },
        )
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={
                "cudnn_conv_algo_search": "DEFAULT",
            },
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        inputs = processor([data] * 2, return_tensors="pt").to("cuda")
        inputs["decoder_input_ids"] = torch.ones((2, 1), dtype=torch.long).to("cuda")

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self,
        test_name: str,
        model_arch: str,
        use_cache: bool,
        use_merged: bool,
        num_beams: int,
    ):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        processor = get_preprocessor(model_id)
        data = self._generate_random_audio_data()
        features = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model.generate(**features, num_beams=num_beams)
        io_outputs = io_model.generate(**features, num_beams=num_beams)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForVision2SeqIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["vision-encoder-decoder", "trocr", "donut"]  # noqa: RUF012

    FULL_GRID = {  # noqa: RUF012
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForVision2Seq

    TASK = "image-to-text"

    GENERATION_LENGTH = 100

    ATOL = 1e-3
    RTOL = 1e-3

    def _get_sample_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def _get_preprocessors(self, model_id):
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        return image_processor, tokenizer

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForVision2Seq.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_generate_utils(self, test_name: str, model_arch: str, use_cache: str):
        model_args = {"test_name": test_name, "model_arch": model_arch, "use_cache": use_cache}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        image_processor, tokenizer = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        outputs = model.generate(inputs=features["pixel_values"])
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertIsInstance(res[0], str)

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        if use_merged is False:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, False)
        else:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
            self.assertEqual(onnx_model.use_merged, True)

        self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is False:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
        if onnx_model.use_cache is True and onnx_model.use_merged is True:
            self.assertTrue(onnx_model.decoder_with_past is None)

        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        image_processor, tokenizer = self._get_preprocessors(model_id)
        transformers_model = AutoModelForVision2Seq.from_pretrained(model_id)

        data = self._get_sample_image()
        inputs = image_processor(data, return_tensors="pt")
        inputs["decoder_input_ids"] = tokenizer("This is a sample output", return_tensors="pt").input_ids

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, use_cache=use_cache)

        for input_type in ["pt", "np"]:
            inputs = image_processor(data, return_tensors=input_type)
            inputs["decoder_input_ids"] = tokenizer("This is a sample output", return_tensors=input_type).input_ids

            onnx_outputs = onnx_model(**inputs, use_cache=use_cache)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

            if use_cache:
                self.assertEqual(
                    len(onnx_outputs["past_key_values"]),
                    len(transformers_outputs["past_key_values"]),
                )
                for i in range(len(onnx_outputs["past_key_values"])):
                    self.assertEqual(
                        len(onnx_outputs["past_key_values"][i]),
                        len(transformers_outputs["past_key_values"][i]),
                    )
                    for j in range(len(onnx_outputs["past_key_values"][i])):
                        torch.testing.assert_close(
                            torch.Tensor(onnx_outputs["past_key_values"][i][j]),
                            transformers_outputs["past_key_values"][i][j],
                            atol=self.ATOL,
                            rtol=self.RTOL,
                        )

        gc.collect()

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_pipeline_image_to_text(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)
        image_processor, tokenizer = self._get_preprocessors(model_id)

        # Speech recogition generation
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            feature_extractor=image_processor,  # for older versions of transformers
        )
        data = self._get_sample_image()
        outputs = pipe(data, max_new_tokens=10)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs[0]["generated_text"], str)

        gc.collect()

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=False
        )
        image_processor, tokenizer = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, use_cache: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_cache=use_cache, use_io_binding=False
        )
        image_processor, tokenizer = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-text",
            model=onnx_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(isinstance(outputs[0]["generated_text"], str))

    @parameterized.expand(SUPPORTED_ARCHITECTURES[:1])
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        image_processor, _ = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        model_with_pkv = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )

        outputs_model_with_pkv = model_with_pkv.generate(
            **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )

        model_without_pkv = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )

        outputs_model_without_pkv = model_without_pkv.generate(
            **features, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )

        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL)
        self.assertEqual(outputs_model_with_pkv.shape[1], self.GENERATION_LENGTH + 1)
        self.assertEqual(outputs_model_without_pkv.shape[1], self.GENERATION_LENGTH + 1)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )
        io_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._get_sample_image()
        image_processor, _ = self._get_preprocessors(model_id)
        pixel_values = image_processor([data] * 2, return_tensors="pt").pixel_values.to("cuda")
        decoder_start_token_id = onnx_model.config.decoder.bos_token_id
        decoder_input_ids = torch.full((2, 1), decoder_start_token_id, dtype=torch.long).to("cuda")

        onnx_outputs = onnx_model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        io_outputs = io_model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool, num_beams: int
    ):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForVision2Seq.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._get_sample_image()
        image_processor, _ = self._get_preprocessors(model_id)
        features = image_processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model.generate(**features, num_beams=num_beams)
        io_outputs = io_model.generate(**features, num_beams=num_beams)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForPix2StructTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["pix2struct"]  # noqa: RUF012

    FULL_GRID = {  # noqa: RUF012
        "model_arch": SUPPORTED_ARCHITECTURES,
        "use_cache": [False, True],
        "use_merged": [False, True],
    }

    ORTMODEL_CLASS = ORTModelForPix2Struct
    TASK = "image-to-text"  # is it fine as well with visual-question-answering?

    GENERATION_LENGTH = 100

    IMAGE = Image.open(
        requests.get(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
            stream=True,
        ).raw
    )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForPix2Struct.from_pretrained(MODEL_NAMES["bert"], export=True)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_transformers_and_save(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        model = ORTModelForPix2Struct.from_pretrained(model_id, export=True, use_merged=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            save_path = os.path.join(tmpdir, ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir)
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertTrue(ONNX_DECODER_NAME not in folder_contents)
            self.assertTrue(ONNX_DECODER_WITH_PAST_NAME not in folder_contents)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_merge_from_onnx_and_save(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        task = "image-to-text-with-past"

        with tempfile.TemporaryDirectory() as tmpdir:
            main_export(model_id, tmpdir, task=task)

            model = ORTModelForPix2Struct.from_pretrained(tmpdir)

            self.assertTrue(model.use_merged)
            self.assertTrue(model.decoder_with_past is None)

            model.save_pretrained(tmpdir + "_save")
            save_path = os.path.join(tmpdir + "_save", ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(save_path, "use_cache_branch"))

            folder_contents = os.listdir(tmpdir + "_save")
            self.assertTrue(ONNX_ENCODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_NAME in folder_contents)
            self.assertFalse(ONNX_DECODER_WITH_PAST_NAME in folder_contents)

    @parameterized.expand(grid_parameters(FULL_GRID))
    def test_compare_to_transformers(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        if use_cache is False and use_merged is True:
            self.skipTest("use_cache=False, use_merged=True are uncompatible")

        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name], use_cache=use_cache)

        self.assertIsInstance(onnx_model.encoder, ORTEncoder)
        if use_merged is False:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_NAME)
            self.assertFalse(has_onnx_input(model_path, "use_cache_branch"))
            self.assertFalse(onnx_model.use_merged)
        else:
            model_path = Path(self.onnx_model_dirs[test_name], ONNX_DECODER_MERGED_NAME)
            self.assertTrue(has_onnx_input(model_path, "use_cache_branch"))
            self.assertTrue(onnx_model.use_merged)

        self.assertIsInstance(onnx_model.decoder, ORTDecoderForSeq2Seq)
        if use_cache is True and use_merged is False:
            self.assertIsInstance(onnx_model.decoder_with_past, ORTDecoderForSeq2Seq)
        if use_cache is True and use_merged is True:
            self.assertTrue(onnx_model.decoder_with_past is None)

        set_seed(SEED)
        transformers_model = Pix2StructForConditionalGeneration.from_pretrained(model_id)

        preprocessor = get_preprocessor(model_id)
        questions = [
            "Who am I?",
            "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud and this is long long very long and super long my dear",
        ]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=questions, padding=True, return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(
                images=[self.IMAGE, self.IMAGE], text=questions, padding=True, return_tensors=input_type
            )

            onnx_outputs = onnx_model(**inputs)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_with_and_without_past_key_values(self, model_arch: str):
        model_args = {"test_name": model_arch + "_False", "model_arch": model_arch, "use_cache": False}
        self._setup(model_args)
        model_args = {"test_name": model_arch + "_True", "model_arch": model_arch, "use_cache": True}
        self._setup(model_args)

        model_with_pkv = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[model_arch + "_True"], use_cache=True
        )
        model_without_pkv = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[model_arch + "_False"], use_cache=False
        )

        model_id = MODEL_NAMES[model_arch]
        preprocessor = get_preprocessor(model_id)
        question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")

        outputs_model_with_pkv = model_with_pkv.generate(
            **inputs, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )
        outputs_model_without_pkv = model_without_pkv.generate(
            **inputs, min_new_tokens=self.GENERATION_LENGTH, max_new_tokens=self.GENERATION_LENGTH, num_beams=1
        )

        self.assertEqual(
            (outputs_model_with_pkv.shape[1], outputs_model_without_pkv.shape[1]),
            (
                inputs["decoder_input_ids"].shape[1] + self.GENERATION_LENGTH + 1,
                inputs["decoder_input_ids"].shape[1] + self.GENERATION_LENGTH + 1,
            ),
        )

        torch.testing.assert_close(outputs_model_with_pkv, outputs_model_without_pkv, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True]}))
    def test_compare_merged_and_not_merged_models_outputs(self, test_name: str, model_arch: str, use_cache: bool):
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

        model_not_merged = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name + "_False"])
        not_merged_onnx_path = Path(self.onnx_model_dirs[test_name + "_False"], ONNX_DECODER_NAME)
        self.assertFalse(has_onnx_input(not_merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_not_merged.use_merged, False)

        model_merged = ORTModelForPix2Struct.from_pretrained(self.onnx_model_dirs[test_name + "_True"])
        merged_onnx_path = Path(self.onnx_model_dirs[test_name + "_True"], ONNX_DECODER_MERGED_NAME)
        self.assertTrue(has_onnx_input(merged_onnx_path, "use_cache_branch"))
        self.assertEqual(model_merged.decoder_with_past, None)
        self.assertEqual(model_merged.use_merged, True)

        model_id = MODEL_NAMES[model_arch]
        preprocessor = get_preprocessor(model_id)
        question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
        inputs = preprocessor(images=self.IMAGE, text=question, return_tensors="pt")

        outputs_model_not_merged = model_not_merged.generate(
            **inputs, max_new_tokens=self.GENERATION_LENGTH, min_new_tokens=self.GENERATION_LENGTH
        )
        outputs_model_merged = model_merged.generate(
            **inputs, max_new_tokens=self.GENERATION_LENGTH, min_new_tokens=self.GENERATION_LENGTH
        )

        torch.testing.assert_close(outputs_model_not_merged, outputs_model_merged, rtol=self.RTOL, atol=self.ATOL)

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "use_cache": [True], "use_merged": [False, True]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        question = ["What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud", "Who are you?"]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=question, padding=True, return_tensors="pt").to(
            "cuda"
        )

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertTrue("encoder_last_hidden_state" in io_outputs)

        self.assertIsInstance(io_outputs.logits, torch.Tensor)
        self.assertIsInstance(io_outputs.encoder_last_hidden_state, torch.Tensor)

        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

    @parameterized.expand(
        grid_parameters(
            {
                "model_arch": SUPPORTED_ARCHITECTURES,
                "use_cache": [True],
                "use_merged": [False, True],
                "num_beams": [1, 3],
            }
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_generation_to_io_binding(
        self, test_name: str, model_arch: str, use_cache: bool, use_merged: bool, num_beams: int
    ):
        model_args = {
            "test_name": test_name,
            "model_arch": model_arch,
            "use_cache": use_cache,
            "use_merged": use_merged,
        }
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForPix2Struct.from_pretrained(
            self.onnx_model_dirs[test_name], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        question = ["What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud", "Who are you?"]
        inputs = preprocessor(images=[self.IMAGE, self.IMAGE], text=question, padding=True, return_tensors="pt").to(
            "cuda"
        )

        onnx_outputs = onnx_model.generate(**inputs, num_beams=num_beams)
        io_outputs = io_model.generate(**inputs, num_beams=num_beams)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs, io_outputs, atol=self.ATOL, rtol=self.RTOL)
