#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import gc
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from onnx import load as onnx_load
from onnx import save as onnx_save
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType, quantize_dynamic
from packaging.version import Version, parse
from parameterized import parameterized
from transformers import AutoTokenizer

from optimum.onnxruntime import (
    AutoCalibrationConfig,
    AutoQuantizationConfig,
    ORTConfig,
    ORTModelForCausalLM,
    ORTModelForFeatureExtraction,
    ORTModelForSeq2SeqLM,
    ORTModelForSequenceClassification,
    ORTQuantizer,
    QuantizationConfig,
)
from optimum.utils.testing_utils import grid_parameters


class ORTQuantizerTest(unittest.TestCase):
    LOAD_CONFIGURATION = {  # noqa: RUF012
        "local_asset": {
            "model_or_path": "tests/assets/onnx",
        },
        "local_asset_different_name": {
            "model_or_path": "tests/assets/onnx",
            "file_name": "different_name.onnx",
        },
        "ort_model_class": {
            "model_or_path": ORTModelForSequenceClassification.from_pretrained(
                "optimum/distilbert-base-uncased-finetuned-sst-2-english"
            )
        },
        "ort_model_with_onnx_model_in_subfolder": {
            "model_or_path": ORTModelForFeatureExtraction.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                subfolder="onnx",
                file_name="model.onnx",
            )
        },
    }

    @parameterized.expand(LOAD_CONFIGURATION.items())
    def test_from_pretrained_method(self, *args):
        _, args = args
        quantizer = ORTQuantizer.from_pretrained(**args)
        self.assertIsInstance(quantizer, ORTQuantizer)

    def test_fail_from_pretrained_method(self):
        with self.assertRaises(Exception) as context:
            ORTQuantizer.from_pretrained("bert-base-cased")
        self.assertIn("Could not find any ONNX model file in bert-base-cased", str(context.exception))

        with self.assertRaises(Exception) as context:
            model = ORTModelForSeq2SeqLM.from_pretrained("optimum/t5-small")
            ORTQuantizer.from_pretrained(model)
        self.assertIn("ORTQuantizer does not support multi-file quantization.", str(context.exception))


class ORTDynamicQuantizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bert", 30),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-roberta", 30),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-distilbert", 30),
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-BartModel", 32),
    )

    SUPPORTED_DECODER_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        (ORTModelForCausalLM, "hf-internal-testing/tiny-random-gpt2", 22),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_dynamic_quantization(self, model_cls, model_name, expected_quantized_matmuls):
        qconfig = QuantizationConfig(
            is_static=False,
            format=QuantFormat.QOperator,
            mode=QuantizationMode.IntegerOps,
            activations_dtype=QuantType.QUInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = model_cls.from_pretrained(model_name, export=True)
            model.save_pretrained(tmp_dir)

            quantizer = ORTQuantizer.from_pretrained(model)
            quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
            expected_ort_config = ORTConfig(quantization=qconfig)
            ort_config = ORTConfig.from_pretrained(tmp_dir)
            # Verify the ORTConfig was correctly created and saved
            self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())

            quantized_model = onnx_load(output_dir.joinpath("model_quantized.onnx"))
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "MatMul" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)
            gc.collect()

    # NOTE: Will be fixed in 1.17.1, reference: https://github.com/microsoft/onnxruntime/pull/19421
    @unittest.skipIf(parse(ort_version) == Version("1.17.0"), "not supported with this onnxruntime version")
    def test_dynamic_quantization_subgraphs(self):
        qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)
        tmp_dir = tempfile.mkdtemp()
        output_dir = Path(tmp_dir)
        model = ORTModelForCausalLM.from_pretrained("fxmarty/onnx-tiny-random-gpt2-with-merge")
        model.save_pretrained(tmp_dir)

        quantizer = ORTQuantizer.from_pretrained(model)
        quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
        expected_ort_config = ORTConfig(quantization=qconfig)
        ort_config = ORTConfig.from_pretrained(tmp_dir)
        # Verify the ORTConfig was correctly created and saved
        self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())

        quantized_model = onnx_load(output_dir.joinpath("decoder_model_merged_quantized.onnx"))
        num_quantized_matmul = 0
        for initializer in quantized_model.graph.initializer:
            if "weight" in initializer.name and "quantized" in initializer.name:
                num_quantized_matmul += 1

        self.assertTrue(num_quantized_matmul > 0)
        gc.collect()

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_DECODER_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS, "use_cache": [True, False]}
        )
    )
    def test_decoder_quantization_with_and_without_cache(self, test_name, model_info, use_cache):
        model_cls, model_name, expected_quantized_matmuls = model_info
        qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)
        model = model_cls.from_pretrained(model_name, export=True, use_cache=use_cache, use_io_binding=use_cache)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            output_dir = Path(tmp_dir)
            quantizer = ORTQuantizer.from_pretrained(model)
            quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
            expected_ort_config = ORTConfig(quantization=qconfig)
            ort_config = ORTConfig.from_pretrained(tmp_dir)

            # Verify the ORTConfig was correctly created and saved
            self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())
            quantized_model = onnx_load(output_dir.joinpath("model_quantized.onnx"))
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "weight" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)
            gc.collect()

    @staticmethod
    def _create_single_head_attention_model(model_path: Path):
        # Q = X @ Wq, K = X @ Wk, V = X @ Wv, Y = softmax(Q @ K^T) @ V @ Wo
        # 4 MatMuls with a constant B (weights) and 2 activation x activation MatMuls.
        seq_len, hidden_size = 8, 16
        rng = np.random.default_rng(42)
        weights = [
            numpy_helper.from_array(rng.standard_normal((hidden_size, hidden_size)).astype(np.float32), name)
            for name in ("Wq", "Wk", "Wv", "Wo")
        ]
        nodes = [
            helper.make_node("MatMul", ["X", "Wq"], ["Q"], name="q_proj"),
            helper.make_node("MatMul", ["X", "Wk"], ["K"], name="k_proj"),
            helper.make_node("MatMul", ["X", "Wv"], ["V"], name="v_proj"),
            helper.make_node("Transpose", ["K"], ["K_t"], perm=[1, 0]),
            helper.make_node("MatMul", ["Q", "K_t"], ["scores"], name="attn_scores"),
            helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
            helper.make_node("MatMul", ["probs", "V"], ["context"], name="attn_context"),
            helper.make_node("MatMul", ["context", "Wo"], ["Y"], name="o_proj"),
        ]
        graph = helper.make_graph(
            nodes,
            "single_head_attention",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq_len, hidden_size])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [seq_len, hidden_size])],
            initializer=weights,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx_save(model, model_path.as_posix())

    @parameterized.expand(
        [
            ("ort_default", None, {"q_proj", "k_proj", "v_proj", "o_proj"}),
            ("const_b_only", True, {"q_proj", "k_proj", "v_proj", "o_proj"}),
            (
                "all_matmuls",
                False,
                {"q_proj", "k_proj", "v_proj", "o_proj", "attn_scores", "attn_context"},
            ),
        ]
    )
    def test_dynamic_quantization_matmul_const_b_only(self, _, matmul_const_b_only, expected_quantized_matmuls):
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        qconfig.matmul_const_b_only = matmul_const_b_only
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir).joinpath("model.onnx")
            self._create_single_head_attention_model(model_path)
            output_dir = Path(tmp_dir).joinpath("quantized")

            quantizer = ORTQuantizer(model_path)
            quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)
            ort_config = ORTConfig.from_pretrained(output_dir)
            self.assertEqual(ort_config.to_dict(), ORTConfig(quantization=qconfig).to_dict())

            quantized_model = onnx_load(output_dir.joinpath("model_quantized.onnx"))
            float_matmuls = {node.name for node in quantized_model.graph.node if node.op_type == "MatMul"}
            num_matmul_integer = sum(node.op_type == "MatMulInteger" for node in quantized_model.graph.node)
            self.assertEqual(num_matmul_integer, len(expected_quantized_matmuls))
            self.assertEqual(
                float_matmuls,
                {"q_proj", "k_proj", "v_proj", "o_proj", "attn_scores", "attn_context"} - expected_quantized_matmuls,
            )

            if matmul_const_b_only is None:
                # Same nodes as onnxruntime's own dynamic quantization with its default options
                ort_quantized_path = Path(tmp_dir).joinpath("model_ort_quantized.onnx")
                quantize_dynamic(model_path, ort_quantized_path, per_channel=True, weight_type=QuantType.QInt8)
                ort_quantized_model = onnx_load(ort_quantized_path.as_posix())
                self.assertEqual(
                    sorted(node.op_type for node in quantized_model.graph.node),
                    sorted(node.op_type for node in ort_quantized_model.graph.node),
                )
            gc.collect()


class ORTStaticQuantizationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS = (
        (ORTModelForSequenceClassification, "hf-internal-testing/tiny-random-bert", 30),
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_EXPECTED_QUANTIZED_MATMULS)
    def test_static_quantization(self, model_cls, model_name, expected_quantized_matmuls):
        qconfig = QuantizationConfig(
            is_static=True,
            format=QuantFormat.QDQ,
            mode=QuantizationMode.QLinearOps,
            activations_dtype=QuantType.QInt8,
            weights_dtype=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
            operators_to_quantize=["MatMul"],
        )

        def preprocess_function(examples, tokenizer):
            return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = model_cls.from_pretrained(model_name, export=True)
            model.save_pretrained(tmp_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            quantizer = ORTQuantizer.from_pretrained(model)

            calibration_dataset = quantizer.get_calibration_dataset(
                "glue",
                dataset_config_name="sst2",
                preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
                num_samples=40,
                dataset_split="train",
            )
            calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
            ranges = quantizer.fit(dataset=calibration_dataset, calibration_config=calibration_config)
            quantizer.quantize(
                save_dir=output_dir,
                calibration_tensors_range=ranges,
                quantization_config=qconfig,
            )

            expected_ort_config = ORTConfig(quantization=qconfig)
            ort_config = ORTConfig.from_pretrained(tmp_dir)
            # Verify the ORTConfig was correctly created and saved
            self.assertEqual(ort_config.to_dict(), expected_ort_config.to_dict())

            quantized_model = onnx_load(output_dir.joinpath("model_quantized.onnx"))
            num_quantized_matmul = 0
            for initializer in quantized_model.graph.initializer:
                if "MatMul" in initializer.name and "quantized" in initializer.name:
                    num_quantized_matmul += 1
            self.assertEqual(expected_quantized_matmuls, num_quantized_matmul)
            gc.collect()
