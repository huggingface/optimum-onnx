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
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import onnx
import torch
from huggingface_hub import HfApi
from onnx import load as onnx_load
from onnxruntime import InferenceSession
from parameterized import parameterized
from transformers import AutoModel, AutoTokenizer
from transformers.utils import http_user_agent

from optimum.exporters.onnx import main_export
from optimum.onnx.graph_transformations import (
    cast_slice_nodes_inputs_to_int32,
    merge_decoders,
    remove_duplicate_weights,
)


class WeightSharingTestCase(TestCase):
    def test_weight_sharing_output_match(self):
        with torch.no_grad():
            for model_id in {"albert-base-v1", "albert-base-v2"}:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id)

                task = "feature-extraction"
                with TemporaryDirectory() as tmpdir:
                    subprocess.run(
                        f"python3 -m optimum.exporters.onnx --model {model_id} --task {task} {tmpdir}",
                        shell=True,
                        check=True,
                    )

                    original_albert_ir = onnx_load(os.path.join(tmpdir, "model.onnx"))
                    compressed_albert_ir = remove_duplicate_weights(original_albert_ir, inplace=False)
                    compressed_albert_session = InferenceSession(
                        compressed_albert_ir.SerializeToString(), providers=["CPUExecutionProvider"]
                    )

                original_outputs = model(**tokenizer("Hello from Hugging Face", return_tensors="pt"))
                compressed_outputs = compressed_albert_session.run(
                    None, dict(tokenizer("Hello from Hugging Face", return_tensors="np"))
                )

            self.assertTrue(
                np.allclose(original_outputs.last_hidden_state.cpu().numpy(), compressed_outputs[0], atol=1e-4)
            )


class OnnxMergingTestCase(TestCase):
    SUPPORTED_ARCHITECTURES_WITH_MODEL_ID = {  # noqa: RUF012
        "hf-internal-testing/tiny-random-t5": "text2text-generation-with-past",
        "hf-internal-testing/tiny-random-BartModel": "text2text-generation-with-past",
        "optimum-internal-testing/tiny-random-whisper": "automatic-speech-recognition-with-past",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_MODEL_ID.items())
    def test_merge_decoders(self, *args):
        model_id, task = args

        with TemporaryDirectory() as tmpdir:
            main_export(model_id, tmpdir, task=task, no_post_process=True)

            decoder = onnx.load(os.path.join(tmpdir, "decoder_model.onnx"))
            decoder_with_past = onnx.load(os.path.join(tmpdir, "decoder_with_past_model.onnx"))
            merged_path = os.path.join(tmpdir, "decoder_model_merged.onnx")

            merge_decoders(decoder, decoder_with_past, save_path=merged_path, strict=False)

            # ONNX Runtime does additional validity checks compared to onnx.checker.check_model
            InferenceSession(merged_path, providers=["CPUExecutionProvider"])

    def test_merge_decoders_with_zero_sized_output_dimension(self):
        float_type = onnx.TensorProto.FLOAT
        hidden = onnx.helper.make_tensor_value_info("hidden", float_type, [1])
        logits = onnx.helper.make_tensor_value_info("logits", float_type, [1])
        cache = onnx.helper.make_tensor_value_info("cache", float_type, [0, "sequence"])
        empty_cache = onnx.helper.make_tensor("empty_cache", float_type, [0, 2], [])
        opsets = [onnx.helper.make_opsetid("", 17)]

        decoder = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node("Identity", ["hidden"], ["logits"]),
                    onnx.helper.make_node("Constant", [], ["cache"], value=empty_cache),
                ],
                "decoder",
                [hidden],
                [logits, cache],
            ),
            opset_imports=opsets,
        )
        decoder_with_past = onnx.helper.make_model(
            onnx.helper.make_graph(
                [onnx.helper.make_node("Neg", ["hidden"], ["logits"])],
                "decoder_with_past",
                [hidden],
                [logits],
            ),
            opset_imports=opsets,
        )

        with TemporaryDirectory() as tmpdir:
            merged_path = Path(tmpdir) / "merged.onnx"
            merge_decoders(decoder, decoder_with_past, save_path=merged_path, strict=False)
            session = InferenceSession(merged_path, providers=["CPUExecutionProvider"])

            for use_cache, expected_logit, expected_cache_shape in [(False, 3.0, (0, 2)), (True, -3.0, (0, 0))]:
                with self.subTest(use_cache=use_cache):
                    logits_output, cache_output = session.run(
                        None,
                        {"hidden": np.array([3.0], dtype=np.float32), "use_cache_branch": np.array([use_cache])},
                    )
                    np.testing.assert_array_equal(logits_output, np.array([expected_logit], dtype=np.float32))
                    self.assertEqual(cache_output.shape, expected_cache_shape)


class OnnxToInt32Test(TestCase):
    def test_to_int32(self):
        model_id = "fxmarty/gpt2-tiny-onnx"

        with TemporaryDirectory() as tmpdir:
            repo_path = HfApi(user_agent=http_user_agent()).snapshot_download(model_id, cache_dir=tmpdir)
            path = str(Path(repo_path, "decoder_model.onnx"))
            save_path = str(Path(repo_path, "decoder_model_int32.onnx"))
            model = onnx.load(path)

            model = cast_slice_nodes_inputs_to_int32(model)

            onnx.save(
                model,
                save_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=Path(save_path).name + "_data",
                convert_attribute=True,
            )

            onnx.checker.check_model(save_path)

            model = InferenceSession(save_path, providers=["CPUExecutionProvider"])

            inputs = {
                "input_ids": np.array([[12, 54, 290, 314, 823, 287, 287]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.int64),
            }

            model.run(None, inputs)
