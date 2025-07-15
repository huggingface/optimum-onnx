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

import inspect
import subprocess
import tempfile
import unittest
from pathlib import Path

import optimum.commands
from optimum.onnxruntime.configuration import AutoQuantizationConfig, ORTConfig


CLI_WIH_CUSTOM_COMMAND_PATH = Path(__file__).parent / "cli_with_custom_command.py"
OPTIMUM_COMMANDS_DIR = Path(inspect.getfile(optimum.commands)).parent
REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH = OPTIMUM_COMMANDS_DIR / "register" / "cli_with_custom_command.py"


class TestCLI(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        REGISTERED_CLI_WITH_CUSTOM_COMMAND_PATH.unlink(missing_ok=True)

    def test_helps_no_raise(self):
        commands = [
            "optimum-cli --help",
            "optimum-cli export --help",
            "optimum-cli export onnx --help",
            "optimum-cli env --help",
            "optimum-cli onnxruntime quantize --help",
            "optimum-cli onnxruntime optimize --help",
        ]

        for command in commands:
            subprocess.run(command, shell=True, check=True)

    def test_env_commands(self):
        subprocess.run("optimum-cli env", shell=True, check=True)

    def test_export_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            commands = [
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-vision_perceiver_conv --task image-classification {tempdir}/onnx",
            ]

            for command in commands:
                subprocess.run(command, shell=True, check=True)

    def test_optimize_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # First export a tiny encoder, decoder only and encoder-decoder
            export_commands = [
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-BertModel {tempdir}/encoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-gpt2 {tempdir}/decoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-bart {tempdir}/encoder-decoder",
            ]
            optimize_commands = [
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/encoder -O1 -o {tempdir}/optimized_encoder",
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/decoder -O1 -o {tempdir}/optimized_decoder",
                f"optimum-cli onnxruntime optimize --onnx_model {tempdir}/encoder-decoder -O1 -o {tempdir}/optimized_encoder_decoder",
            ]

            for export, optimize in zip(export_commands, optimize_commands):
                subprocess.run(export, shell=True, check=True)
                subprocess.run(optimize, shell=True, check=True)

    def test_quantize_commands(self):
        with tempfile.TemporaryDirectory() as tempdir:
            ort_config = ORTConfig(quantization=AutoQuantizationConfig.avx2(is_static=False))
            ort_config.save_pretrained(tempdir)

            # First export a tiny encoder, decoder only and encoder-decoder
            export_commands = [
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-bert {tempdir}/encoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-gpt2 {tempdir}/decoder",
                f"optimum-cli export onnx --model hf-internal-testing/tiny-random-t5 {tempdir}/encoder-decoder",
            ]
            quantize_commands = [
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/encoder --avx2 -o {tempdir}/quantized_encoder",
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/decoder --avx2 -o {tempdir}/quantized_decoder",
                f"optimum-cli onnxruntime quantize --onnx_model {tempdir}/encoder-decoder --avx2 -o {tempdir}/quantized_encoder_decoder",
            ]

            quantize_with_config_commands = [
                f"optimum-cli onnxruntime quantize --onnx_model hf-internal-testing/tiny-random-bert --c {tempdir}/ort_config.json -o {tempdir}/quantized_encoder_with_config",
                f"optimum-cli onnxruntime quantize --onnx_model hf-internal-testing/tiny-random-gpt2 --c {tempdir}/ort_config.json -o {tempdir}/quantized_decoder_with_config",
                f"optimum-cli onnxruntime quantize --onnx_model hf-internal-testing/tiny-random-t5 --c {tempdir}/ort_config.json -o {tempdir}/quantized_encoder_decoder_with_config",
            ]

            for export, quantize, quantize_with_config in zip(
                export_commands, quantize_commands, quantize_with_config_commands
            ):
                subprocess.run(export, shell=True, check=True)
                subprocess.run(quantize, shell=True, check=True)
                subprocess.run(quantize_with_config, shell=True, check=True)
