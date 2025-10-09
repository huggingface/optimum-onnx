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
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import requests
import torch
from huggingface_hub import HfApi
from huggingface_hub.constants import default_cache_path
from parameterized import parameterized
from PIL import Image
from testing_utils import MODEL_NAMES, SEED, ORTModelTestMixin
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForImageToImage,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    set_seed,
)
from transformers.modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from transformers.models.swin2sr.configuration_swin2sr import Swin2SRConfig
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import get_gpu_count, require_torch_gpu
from transformers.utils import http_user_agent

from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime import (
    ONNX_WEIGHTS_NAME,
    ORTModelForAudioClassification,
    ORTModelForAudioFrameClassification,
    ORTModelForAudioXVector,
    ORTModelForCTC,
    ORTModelForCustomTasks,
    ORTModelForFeatureExtraction,
    ORTModelForImageClassification,
    ORTModelForImageToImage,
    ORTModelForMaskedLM,
    ORTModelForMultipleChoice,
    ORTModelForQuestionAnswering,
    ORTModelForSemanticSegmentation,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    pipeline,
)
from optimum.onnxruntime.modeling import ORTModel
from optimum.utils import CONFIG_NAME, logging
from optimum.utils.save_utils import maybe_load_preprocessors
from optimum.utils.testing_utils import grid_parameters, remove_directory, require_hf_token, require_ort_rocm


logger = logging.get_logger()


class ORTModelIntegrationTest(unittest.TestCase):
    ORTMODEL_CLASS = ORTModel
    AUTOMODEL_CLASS = AutoModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LOCAL_MODEL_PATH = "tests/assets/onnx"
        self.ONNX_MODEL_ID = "philschmid/distilbert-onnx"
        self.TINY_ONNX_MODEL_ID = "fxmarty/resnet-tiny-beans"
        self.FAIL_ONNX_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"

    def test_load_model_from_hub_infer_onnx_model(self):
        model_id = "optimum-internal-testing/tiny-random-llama"
        file_name = "model_optimized.onnx"

        model = self.ORTMODEL_CLASS.from_pretrained(model_id)
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx")
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="onnx", file_name=file_name)
        self.assertEqual(model.path.name, file_name)

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx", file_name=file_name)
        self.assertEqual(model.path.name, file_name)

        model = self.ORTMODEL_CLASS.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(model_id, revision="merged-onnx", subfolder="subfolder")
        self.assertEqual(model.path.name, "model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", subfolder="subfolder", file_name=file_name
        )
        self.assertEqual(model.path.name, file_name)

        model = self.ORTMODEL_CLASS.from_pretrained(
            model_id, revision="merged-onnx", file_name="decoder_with_past_model.onnx"
        )
        self.assertEqual(model.path.name, "decoder_with_past_model.onnx")

        model = self.ORTMODEL_CLASS.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.assertEqual(model.path.name, "model.onnx")

        with self.assertRaises(FileNotFoundError):
            self.ORTMODEL_CLASS.from_pretrained(
                "hf-internal-testing/tiny-random-LlamaForCausalLM", file_name="test.onnx"
            )

    def test_load_model_from_local_path(self):
        model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        model = ORTModel.from_pretrained(
            "fxmarty/tiny-bert-sst2-distilled-subfolder",
            subfolder="my_subfolder",
        )
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = ORTModel.from_pretrained("fxmarty/tiny-bert-sst2-distilled-onnx-subfolder", subfolder="my_subfolder")
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_cache(self):
        _ = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID)  # caching

        model = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)

        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.TINY_ONNX_MODEL_ID.replace("/", "--"))
        remove_directory(dirpath)

        with self.assertRaises(Exception):  # noqa: B017
            _ = ORTModel.from_pretrained(self.TINY_ONNX_MODEL_ID, local_files_only=True)

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_load_model_cuda_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    @require_torch_gpu
    @pytest.mark.trt_ep_test
    def test_load_model_tensorrt_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="TensorrtExecutionProvider")
        self.assertListEqual(
            model.providers, ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_load_model_rocm_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="ROCMExecutionProvider")
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cuda:0"))

    def test_load_model_cpu_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CPUExecutionProvider")
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])
        self.assertListEqual(model.model.get_providers(), model.providers)
        self.assertEqual(model.device, torch.device("cpu"))

    def test_load_model_unknown_provider(self):
        with self.assertRaises(ValueError):
            ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="FooExecutionProvider")

    def test_load_model_from_hub_without_onnx_model(self):
        ORTModel.from_pretrained(self.FAIL_ONNX_MODEL_ID)

    def test_model_on_cpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        cpu = torch.device("cpu")
        model.to(cpu)
        self.assertEqual(model.device, cpu)
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    # test string device input for to()
    def test_model_on_cpu_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        cpu = torch.device("cpu")
        model.to("cpu")
        self.assertEqual(model.device, cpu)
        self.assertListEqual(model.providers, ["CPUExecutionProvider"])

    def test_missing_execution_provider(self):
        with self.assertRaises(ValueError) as cm:
            ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="ThisProviderDoesNotExist")

        self.assertTrue("but the available execution providers" in str(cm.exception))

        is_onnxruntime_gpu_installed = (
            subprocess.run("pip list | grep onnxruntime-gpu", shell=True, capture_output=True).stdout.decode("utf-8")
            != ""
        )
        is_onnxruntime_installed = "onnxruntime " in subprocess.run(
            "pip list | grep onnxruntime", shell=True, capture_output=True
        ).stdout.decode("utf-8")

        if not is_onnxruntime_gpu_installed:
            for provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
                with self.assertRaises(ValueError) as cm:
                    _ = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider=provider)

            self.assertTrue("but the available execution providers" in str(cm.exception))

        else:
            logger.info("Skipping CUDAExecutionProvider/TensorrtExecutionProvider without `onnxruntime-gpu` test")

        # need to install first onnxruntime-gpu, then onnxruntime for this test to pass,
        # thus overwriting onnxruntime/capi/_ld_preload.py
        if is_onnxruntime_installed and is_onnxruntime_gpu_installed:
            for provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"]:
                with self.assertRaises(ValueError) as cm:
                    _ = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider=provider)

                self.assertTrue("but the available execution providers" in str(cm.exception))
        else:
            logger.info("Skipping double onnxruntime + onnxruntime-gpu install test")

        # despite passing CUDA_PATH='' LD_LIBRARY_PATH='', this test does not pass in nvcr.io/nvidia/tensorrt:22.08-py3
        # It does pass locally.
        """
        # LD_LIBRARY_PATH can't be set at runtime,
        # see https://stackoverflow.com/questions/856116/changing-ld-library-path-at-runtime-for-ctypes
        # testing only for TensorRT as having ORT_CUDA_UNAVAILABLE is hard
        if is_onnxruntime_gpu_installed:
            commands = [
                "from optimum.onnxruntime import ORTModel",
                "model = ORTModel.from_pretrained('philschmid/distilbert-onnx', provider='TensorrtExecutionProvider')",
            ]

            full_command = json.dumps(";".join(commands))

            out = subprocess.run(
                f"CUDA_PATH='' LD_LIBRARY_PATH='' python -c {full_command}", shell=True, capture_output=True
            )
            self.assertTrue("requirements could not be loaded" in out.stderr.decode("utf-8"))
        else:
            logger.info("Skipping broken CUDA/TensorRT install test")
        """

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_model_on_gpu(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_model_on_rocm_ep(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        gpu = torch.device("cuda")
        model.to(gpu)
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    # test string device input for to()
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_model_on_gpu_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_model_on_rocm_ep_str(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to("cuda")
        self.assertEqual(model.device, torch.device("cuda:0"))
        self.assertListEqual(model.providers, ["ROCMExecutionProvider", "CPUExecutionProvider"])

    def test_passing_session_options(self):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 3
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, session_options=options)
        self.assertEqual(model.model.get_session_options().intra_op_num_threads, 3)

    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_passing_provider_options(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="CUDAExecutionProvider")
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "1")

        model = ORTModel.from_pretrained(
            self.ONNX_MODEL_ID,
            provider="CUDAExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
        )
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["do_copy_in_default_stream"], "0")

        # two providers case
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="TensorrtExecutionProvider")
        self.assertEqual(
            model.model.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "0"
        )

        model = ORTModel.from_pretrained(
            self.ONNX_MODEL_ID,
            provider="TensorrtExecutionProvider",
            provider_options={"trt_engine_cache_enable": True},
        )
        self.assertEqual(
            model.model.get_provider_options()["TensorrtExecutionProvider"]["trt_engine_cache_enable"], "1"
        )

    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_passing_provider_options_rocm_provider(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID, provider="ROCMExecutionProvider")
        self.assertEqual(model.model.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "1")

        model = ORTModel.from_pretrained(
            self.ONNX_MODEL_ID,
            provider="ROCMExecutionProvider",
            provider_options={"do_copy_in_default_stream": 0},
        )
        self.assertEqual(model.model.get_provider_options()["ROCMExecutionProvider"]["do_copy_in_default_stream"], "0")

    @unittest.skipIf(get_gpu_count() <= 1, "this test requires multi-gpu")
    def test_model_on_gpu_id(self):
        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to(torch.device("cuda:1"))
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to(1)
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

        model = ORTModel.from_pretrained(self.ONNX_MODEL_ID)
        model.to("cuda:1")
        self.assertEqual(model.model.get_provider_options()["CUDAExecutionProvider"]["device_id"], "1")

    def test_load_model_from_hub_private(self):
        token = os.environ.get("HF_TOKEN", None)

        if not token:
            self.skipTest(
                "Test requires a read access token for optimum-internal-testing in the environment variable `HF_TOKEN`."
            )

        model = ORTModelForCustomTasks.from_pretrained(
            "optimum-internal-testing/tiny-random-phi-private", revision="onnx", token=token
        )
        self.assertIsInstance(model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and ONNX exported model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(ONNX_WEIGHTS_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_save_load_ort_model_with_external_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], export=True)
            model.save_pretrained(tmpdirname)

            # verify external data is exported
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(ONNX_WEIGHTS_NAME, folder_contents)
            self.assertIn(ONNX_WEIGHTS_NAME + "_data", folder_contents)

            # verify loading from local folder works
            model = ORTModelForSequenceClassification.from_pretrained(tmpdirname, export=False)
            remove_directory(tmpdirname)

    @require_hf_token
    def test_save_model_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(
                tmpdirname,
                token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id=self.HUB_REPOSITORY,
                private=True,
            )

    @require_hf_token
    @unittest.mock.patch.dict(os.environ, {"FORCE_ONNX_EXTERNAL_DATA": "1"})
    def test_push_ort_model_with_external_data_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["bert"], export=True)
            model.save_pretrained(
                tmpdirname + "/onnx",
                token=os.environ.get("HF_AUTH_TOKEN", None),
                repository_id=MODEL_NAMES["bert"].split("/")[-1] + "-onnx",
                private=True,
                push_to_hub=True,
            )

            # verify loading from hub works
            model = ORTModelForSequenceClassification.from_pretrained(
                MODEL_NAMES["bert"] + "-onnx",
                export=False,
                token=os.environ.get("HF_AUTH_TOKEN", None),
            )

    @parameterized.expand(("", "onnx"))
    def test_loading_with_config_not_from_subfolder(self, subfolder):
        # config.json file in the root directory and not in the subfolder
        model_id = "sentence-transformers-testing/stsb-bert-tiny-onnx"
        # hub model
        ORTModelForFeatureExtraction.from_pretrained(model_id, subfolder=subfolder, export=subfolder == "")
        with tempfile.TemporaryDirectory() as tmpdirname:
            local_dir = Path(tmpdirname) / "model"
            HfApi(user_agent=http_user_agent()).snapshot_download(repo_id=model_id, local_dir=local_dir)
            ORTModelForFeatureExtraction.from_pretrained(local_dir, subfolder=subfolder, export=subfolder == "")
            remove_directory(tmpdirname)


class ORTModelForQuestionAnsweringIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "albert",
        "bart",
        "bert",
        "big_bird",
        "bigbird_pegasus",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        # "flaubert", # currently fails for some reason (squad multiprocessing),
        # but also couldn't find any real qa checkpoints on the hub for this model
        "gptj",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in ORTModel
        # "layoutlm"
        # "layoutlmv3",
        "mbart",
        "mobilebert",
        "nystromformer",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm-qa",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForQuestionAnswering
    TASK = "question-answering"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForQuestionAnswering.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        tokens = tokenizer("This is a sample output", return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer("This is a sample output", return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("start_logits", onnx_outputs)
            self.assertIn("end_logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.start_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(onnx_outputs.end_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.start_logits),
                transformers_outputs.start_logits,
                atol=self.ATOL,
                rtol=self.RTOL,
            )
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.end_logits), transformers_outputs.end_logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        self.assertEqual(pipe.device, pipe.model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("question-answering")
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer, device=0)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        provider = "ROCMExecutionProvider"
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer, device=0)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertTrue(isinstance(outputs["answer"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForQuestionAnswering.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForQuestionAnswering.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("start_logits" in io_outputs)
        self.assertTrue("end_logits" in io_outputs)
        self.assertIsInstance(io_outputs.start_logits, torch.Tensor)
        self.assertIsInstance(io_outputs.end_logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            torch.Tensor(io_outputs.start_logits), onnx_outputs.start_logits, atol=self.ATOL, rtol=self.RTOL
        )
        torch.testing.assert_close(
            torch.Tensor(io_outputs.end_logits), onnx_outputs.end_logits, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class ORTModelForMaskedLMIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "albert",
        "bert",
        "big_bird",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "mobilebert",
        "mpnet",
        "perceiver_text",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForMaskedLM
    TASK = "fill-mask"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForMaskedLM.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = f"The capital of France is {tokenizer.mask_token}."
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer)
        MASK_TOKEN = tokenizer.mask_token  # noqa: N806
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("fill-mask")
        text = f"The capital of France is {pipe.tokenizer.mask_token}."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token  # noqa: N806
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer, device=0)
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["token_str"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token  # noqa: N806
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer, device=0)
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["token_str"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMaskedLM.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForMaskedLM.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer([f"The capital of France is {tokenizer.mask_token}."] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(io_outputs.logits, onnx_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()

    def test_load_sentence_transformers_model_as_fill_mask(self):
        model_id = "sparse-encoder-testing/splade-bert-tiny-nq"
        onnx_model = ORTModelForMaskedLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer)
        text = f"The capital of France is {tokenizer.mask_token}."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

        gc.collect()


class ORTModelForSequenceClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "albert",
        "bart",
        "bert",
        "big_bird",
        "bigbird_pegasus",
        "bloom",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        # "gpt2",  # see tasks.py
        # "gpt_neo",  # see tasks.py
        # "gptj",  # see tasks.py
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in ORTModel
        # "layoutlm"
        # "layoutlmv3",
        "mbart",
        "mobilebert",
        "nystromformer",
        "perceiver_text",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForSequenceClassification
    TASK = "text-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSequenceClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("text-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    def test_pipeline_zero_shot_classification(self):
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            "typeform/distilbert-base-uncased-mnli", export=True
        )
        tokenizer = get_preprocessor("typeform/distilbert-base-uncased-mnli")
        pipe = pipeline("zero-shot-classification", model=onnx_model, tokenizer=tokenizer)
        sequence_to_classify = "Who are you voting for in 2020?"
        candidate_labels = ["Europe", "public health", "politics", "elections"]
        hypothesis_template = "This text is about {}."
        outputs = pipe(
            sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template
        )

        # compare model output class
        self.assertTrue(all(score > 0.0 for score in outputs["scores"]))
        self.assertTrue(all(isinstance(label, str) for label in outputs["labels"]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSequenceClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForTokenClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "albert",
        "bert",
        "big_bird",
        "bloom",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "gpt2",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in ORTModel
        # "layoutlm"
        # "layoutlmv3",
        "mobilebert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForTokenClassification
    TASK = "token-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForTokenClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            onnx_outputs = onnx_model(**tokens)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("token-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForTokenClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForFeatureExtractionIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "albert",
        "bert",
        "camembert",
        "distilbert",
        "electra",
        "mpnet",
        "roberta",
        "xlm-roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForFeatureExtraction
    TASK = "feature-extraction"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            # Test default behavior (return_dict=True)
            onnx_outputs = onnx_model(**tokens)
            self.assertIsInstance(onnx_outputs, BaseModelOutput)
            self.assertIn("last_hidden_state", onnx_outputs)
            self.assertIsInstance(onnx_outputs.last_hidden_state, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Test return_dict=False
            onnx_outputs_dict = onnx_model(**tokens, return_dict=False)
            self.assertIsInstance(onnx_outputs_dict, tuple)
            self.assertIsInstance(onnx_outputs_dict[0], self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.last_hidden_state),
                transformers_outputs.last_hidden_state,
                atol=self.ATOL,
                rtol=self.RTOL,
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("feature-extraction")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("last_hidden_state" in io_outputs)
        self.assertIsInstance(io_outputs.last_hidden_state, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            onnx_outputs.last_hidden_state, io_outputs.last_hidden_state, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()

    def test_default_token_type_ids(self):
        model_id = MODEL_NAMES["bert"]
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("this is a simple input", return_tensors="np")
        self.assertTrue("token_type_ids" in model.input_names)
        token_type_ids = tokens.pop("token_type_ids")
        outs = model(token_type_ids=token_type_ids, **tokens)
        outs_without_token_type_ids = model(**tokens)
        torch.testing.assert_close(
            outs.last_hidden_state, outs_without_token_type_ids.last_hidden_state, atol=self.ATOL, rtol=self.RTOL
        )
        gc.collect()


class ORTModelForFeatureExtractionFromImageModelsIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["vit", "dinov2", "visual_bert"]  # noqa: RUF012

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForFeatureExtraction
    TASK = "feature-extraction"

    def get_raw_image(self):
        # Create a simple 200x300 RGB image with random colors
        np.random.seed(42)  # For reproducibility
        image_array = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
        return Image.fromarray(image_array)

    def get_input(self, model_arch, return_tensors="pt"):
        model_id = MODEL_NAMES[model_arch]
        processor = get_preprocessor(model_id)
        text = "This is a sample output"

        if model_arch == "visual_bert":
            tokens = processor(text, return_tensors=return_tensors)

            np.random.seed(SEED)
            shared_visual_embeds = np.random.randn(1, 10, 20).astype(np.float32)

            if return_tensors == "pt":
                visual_embeds = torch.from_numpy(shared_visual_embeds)
                visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
                visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
            elif return_tensors == "np":
                visual_embeds = shared_visual_embeds
                visual_token_type_ids = np.ones(visual_embeds.shape[:-1], dtype=np.int64)
                visual_attention_mask = np.ones(visual_embeds.shape[:-1], dtype=np.float32)

            tokens.update(
                {
                    "visual_embeds": visual_embeds,
                    "visual_token_type_ids": visual_token_type_ids,
                    "visual_attention_mask": visual_attention_mask,
                }
            )
            return tokens

        raw_input = self.get_raw_image()
        return processor(images=raw_input, return_tensors=return_tensors)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        inputs = self.get_input(model_arch, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = self.get_input(model_arch, return_tensors=input_type)
            onnx_outputs = onnx_model(**inputs)

            self.assertIn("last_hidden_state", onnx_outputs)
            self.assertIsInstance(onnx_outputs.last_hidden_state, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.last_hidden_state),
                transformers_outputs.last_hidden_state,
                atol=self.ATOL,
                rtol=self.RTOL,
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_ort_model_inference(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        onnx_model = ORTModelForFeatureExtraction.from_pretrained(self.onnx_model_dirs[model_arch])
        processed_inputs = self.get_input(model_arch, return_tensors="pt")
        outputs = onnx_model(**processed_inputs)

        # Check device and output format
        assert onnx_model.device.type == "cpu"
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        features = outputs.last_hidden_state.detach().cpu().numpy().tolist()
        assert all(isinstance(item, float) for row in features for inner in row for item in inner)
        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_inference_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        ).to("cuda")
        processed_inputs = self.get_input(model_arch, return_tensors="pt").to("cuda")
        outputs = onnx_model(**processed_inputs)

        # Check device and output format
        assert onnx_model.device.type == "cuda"
        assert isinstance(outputs.last_hidden_state, torch.Tensor)
        features = outputs.last_hidden_state.detach().cpu().numpy().tolist()
        assert all(isinstance(item, float) for row in features for inner in row for item in inner)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokens = self.get_input(model_arch, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("last_hidden_state" in io_outputs)
        self.assertIsInstance(io_outputs.last_hidden_state, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            onnx_outputs.last_hidden_state, io_outputs.last_hidden_state, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class ORTModelForMultipleChoiceIntegrationTest(ORTModelTestMixin):
    # Multiple Choice tests are conducted on different models due to mismatch size in model's classifier
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "albert",
        "bert",
        "big_bird",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "mobilebert",
        "nystromformer",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
        "rembert",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForMultipleChoice
    TASK = "multiple-choice"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMultipleChoice.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        num_choices = 4
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        start = "The color of the sky is"
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

        pt_inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
        with torch.no_grad():
            transformers_outputs = transformers_model(**pt_inputs)

        for input_type in ["pt", "np"]:
            inps = dict(inputs.convert_to_tensors(tensor_type=input_type))
            onnx_outputs = onnx_model(**inps)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForMultipleChoice.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForMultipleChoice.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        num_choices = 4
        start = "The color of the sky is"
        tokenizer = get_preprocessor(model_id)
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)
        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
        inputs = dict(inputs.convert_to_tensors(tensor_type="pt").to("cuda"))

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(io_outputs.logits, onnx_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForImageClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "beit",
        "convnext",
        "convnextv2",
        "data2vec-vision",
        "deit",
        "dinov2",
        "efficientnet",
        "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "perceiver_vision",
        "poolformer",
        "resnet",
        "segformer",
        "swin",
        "swin-window",
        "vit",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForImageClassification
    TASK = "image-classification"

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

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForImageClassification.from_pretrained(MODEL_NAMES["t5"])
        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = maybe_load_preprocessors(model_id)[-1]
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")

        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)

            onnx_outputs = onnx_model(**inputs)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), trtfs_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        preprocessor = maybe_load_preprocessors(model_id)[-1]
        pipe = pipeline("image-classification", model=onnx_model, image_processor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("image-classification")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-classification", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )
        io_model = ORTModelForImageClassification.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForSemanticSegmentationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ("segformer", "dpt")

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForSemanticSegmentation
    TASK = "semantic-segmentation"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForSemanticSegmentation.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        trfs_model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            trtfs_outputs = trfs_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)

            onnx_outputs = onnx_model(**inputs)

            self.assertIn("logits", onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), trtfs_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(self.onnx_model_dirs[model_arch])
        preprocessor = maybe_load_preprocessors(model_id)[-1]
        pipe = pipeline("image-segmentation", model=onnx_model, image_processor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertEqual(pipe.device, onnx_model.device)
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("image-segmentation")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # compare model output class
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

    # TODO: enable TensorrtExecutionProvider test once https://github.com/huggingface/optimum/issues/798 is fixed
    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider"]})
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-segmentation", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        preprocessor = get_preprocessor(model_id)
        pipe = pipeline("image-segmentation", model=onnx_model, feature_extractor=preprocessor, device=0)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")

        # compare model output class
        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForSemanticSegmentation.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        preprocessor = get_preprocessor(model_id)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=[image] * 2, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForAudioClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "audio-spectrogram-transformer",
        "data2vec-audio",
        "hubert",
        "sew",
        "sew-d",
        "unispeech",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
        "whisper",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForAudioClassification
    TASK = "audio-classification"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForAudioClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(self.onnx_model_dirs[model_arch])
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=onnx_model, feature_extractor=processor, sampling_rate=220)
        data = self._generate_random_audio_data()
        outputs = pipe(data)

        self.assertEqual(pipe.device, onnx_model.device)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("audio-classification")
        data = self._generate_random_audio_data()
        outputs = pipe(data)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    @pytest.mark.trt_ep_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=onnx_model, feature_extractor=processor, device=0)
        data = self._generate_random_audio_data()
        outputs = pipe(data)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(
        grid_parameters({"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["ROCMExecutionProvider"]})
    )
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, test_name: str, model_arch: str, provider: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], provider=provider
        )
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=onnx_model, feature_extractor=processor, device=0)
        data = self._generate_random_audio_data()
        outputs = pipe(data)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForAudioClassification.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._generate_random_audio_data()
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)

        gc.collect()


class ORTModelForCTCIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "data2vec-audio",
        "hubert",
        "mctct",
        "sew",
        "sew-d",
        "unispeech",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForCTC
    TASK = "ctc"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForCTC.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCTC.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForCTC.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForCTC.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=False,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )
        io_model = ORTModelForCTC.from_pretrained(
            self.onnx_model_dirs[model_arch],
            use_io_binding=True,
            provider="CUDAExecutionProvider",
            provider_options={"cudnn_conv_algo_search": "DEFAULT"},
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._generate_random_audio_data()
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            torch.Tensor(onnx_outputs.logits), io_outputs.logits, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class ORTModelForAudioXVectorIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "data2vec-audio",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForAudioXVector
    TASK = "audio-xvector"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForAudioXVector.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioXVector.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioXVector.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)
        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(onnx_outputs.embeddings, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.embeddings), transformers_outputs.embeddings, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioXVector.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=False, provider="CUDAExecutionProvider"
        )
        io_model = ORTModelForAudioXVector.from_pretrained(
            self.onnx_model_dirs[model_arch], use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        data = self._generate_random_audio_data()
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(data, return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**input_values)
        io_outputs = io_model(**input_values)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)
        self.assertIsInstance(io_outputs.embeddings, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(onnx_outputs.logits, io_outputs.logits, atol=self.ATOL, rtol=self.RTOL)
        torch.testing.assert_close(onnx_outputs.embeddings, io_outputs.embeddings, atol=self.ATOL, rtol=self.RTOL)
        gc.collect()


class ORTModelForAudioFrameClassificationIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = [  # noqa: RUF012
        "data2vec-audio",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}  # noqa: RUF012
    ORTMODEL_CLASS = ORTModelForAudioFrameClassification
    TASK = "audio-frame-classification"

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForAudioFrameClassification.from_pretrained(MODEL_NAMES["t5"])

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForAudioFrameClassification.from_pretrained(self.onnx_model_dirs[model_arch])

        self.assertIsInstance(onnx_model.model, onnxruntime.InferenceSession)
        self.assertIsInstance(onnx_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            onnx_outputs = onnx_model(**input_values)

            self.assertTrue("logits" in onnx_outputs)
            self.assertIsInstance(onnx_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            torch.testing.assert_close(
                torch.Tensor(onnx_outputs.logits), transformers_outputs.logits, atol=self.ATOL, rtol=self.RTOL
            )

        gc.collect()


class ORTModelForImageToImageIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["swin2sr"]  # noqa: RUF012

    ORTMODEL_CLASS = ORTModelForImageToImage

    TASK = "image-to-image"

    def _get_sample_image(self):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def _get_preprocessors(self, model_id):
        image_processor = AutoImageProcessor.from_pretrained(model_id)

        return image_processor

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = ORTModelForImageToImage.from_pretrained(MODEL_NAMES["bert"], export=True)
        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        self.assertIsInstance(onnx_model.config, Swin2SRConfig)
        set_seed(SEED)

        transformers_model = AutoModelForImageToImage.from_pretrained(model_id)
        image_processor = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**features)

        onnx_outputs = onnx_model(**features)
        self.assertIsInstance(onnx_outputs, ImageSuperResolutionOutput)
        self.assertTrue("reconstruction" in onnx_outputs)
        self.assertIsInstance(onnx_outputs.reconstruction, torch.Tensor)
        torch.testing.assert_close(
            onnx_outputs.reconstruction, transformers_outputs.reconstruction, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_generate_utils(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)

        data = self._get_sample_image()
        features = image_processor(data, return_tensors="pt")

        outputs = onnx_model(**features)
        self.assertIsInstance(outputs, ImageSuperResolutionOutput)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_image_to_image(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-image",
            model=onnx_model,
            image_processor=image_processor,
        )
        data = self._get_sample_image()
        outputs = pipe(data)
        self.assertEqual(pipe.device, onnx_model.device)
        self.assertIsInstance(outputs, Image.Image)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-image",
            model=onnx_model,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        self.assertIsInstance(outputs, Image.Image)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm(self, model_arch: str):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)
        model_id = MODEL_NAMES[model_arch]
        onnx_model = ORTModelForImageToImage.from_pretrained(self.onnx_model_dirs[model_arch])
        image_processor = self._get_preprocessors(model_id)
        pipe = pipeline(
            "image-to-image",
            model=onnx_model,
            image_processor=image_processor,
            device=0,
        )

        data = self._get_sample_image()
        outputs = pipe(data)

        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        self.assertIsInstance(outputs, Image.Image)


class ORTModelForCustomTasksIntegrationTest(ORTModelTestMixin):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {  # noqa: RUF012
        "sbert": "optimum/sbert-all-MiniLM-L6-with-pooler",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_model_call(self, *args, **kwargs):
        model_arch, model_id = args
        model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        for input_type in ["pt", "np"]:
            tokens = tokenizer("This is a sample output", return_tensors=input_type)
            outputs = model(**tokens)
            self.assertIsInstance(outputs.pooler_output, self.TENSOR_ALIAS_TO_TYPE[input_type])

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_pipeline_ort_model(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_pipeline_on_gpu(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    @require_ort_rocm
    @pytest.mark.rocm_ep_test
    def test_pipeline_on_rocm_ep(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(any(any(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_default_pipeline_and_model_device(self, *args, **kwargs):
        model_arch, model_id = args
        onnx_model = ORTModelForCustomTasks.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=onnx_model, tokenizer=tokenizer)
        self.assertEqual(pipe.device, onnx_model.device)

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    @require_torch_gpu
    @pytest.mark.cuda_ep_test
    def test_compare_to_io_binding(self, *args, **kwargs):
        _, model_id = args

        set_seed(SEED)
        onnx_model = ORTModelForCustomTasks.from_pretrained(
            model_id, use_io_binding=False, provider="CUDAExecutionProvider"
        )
        set_seed(SEED)
        io_model = ORTModelForCustomTasks.from_pretrained(
            model_id, use_io_binding=True, provider="CUDAExecutionProvider"
        )

        self.assertFalse(onnx_model.use_io_binding)
        self.assertTrue(io_model.use_io_binding)

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer("This is a sample output", return_tensors="pt").to("cuda")

        onnx_outputs = onnx_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("pooler_output" in io_outputs)
        self.assertIsInstance(io_outputs.pooler_output, torch.Tensor)

        # compare tensor outputs
        torch.testing.assert_close(
            onnx_outputs.pooler_output, io_outputs.pooler_output, atol=self.ATOL, rtol=self.RTOL
        )

        gc.collect()


class TestBothExportersORTModel(unittest.TestCase):
    @parameterized.expand(
        [
            ["question-answering", ORTModelForQuestionAnsweringIntegrationTest],
            ["text-classification", ORTModelForSequenceClassificationIntegrationTest],
            ["token-classification", ORTModelForTokenClassificationIntegrationTest],
            ["feature-extraction", ORTModelForFeatureExtractionIntegrationTest],
            ["multiple-choice", ORTModelForMultipleChoiceIntegrationTest],
            ["image-classification", ORTModelForImageClassificationIntegrationTest],
            ["semantic-segmentation", ORTModelForSemanticSegmentationIntegrationTest],
            ["audio-classification", ORTModelForAudioClassificationIntegrationTest],
            ["automatic-speech-recognition", ORTModelForCTCIntegrationTest],
            ["audio-xvector", ORTModelForAudioXVectorIntegrationTest],
            ["audio-frame-classification", ORTModelForAudioFrameClassificationIntegrationTest],
            ["image-to-image", ORTModelForImageToImageIntegrationTest],
        ]
    )
    def test_find_untested_architectures(self, task: str, test_class):
        supported_export_models = TasksManager.get_supported_model_type_for_task(task=task, exporter="onnx")
        tested_architectures = set(test_class.SUPPORTED_ARCHITECTURES)

        untested_architectures = set(supported_export_models) - tested_architectures
        if len(untested_architectures) > 0:
            logger.warning(
                f"For the task `{task}`, the ONNX export supports {supported_export_models}, but only {tested_architectures} are tested.\n"
                f"    Missing {untested_architectures}."
            )
