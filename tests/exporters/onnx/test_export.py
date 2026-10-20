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
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import onnx
import pytest
from parameterized import parameterized
from transformers import (
    AutoConfig,
    is_torch_available,
)
from transformers.testing_utils import require_onnx, require_torch, require_torch_gpu, require_vision, slow
from utils_tests import (
    PYTORCH_DIFFUSION_MODEL,
    PYTORCH_EXPORT_MODELS_TINY,
    PYTORCH_SENTENCE_TRANSFORMERS_MODEL,
    PYTORCH_TIMM_MODEL,
    VALIDATE_EXPORT_ON_SHAPES_SLOW,
)

from optimum.exporters.error_utils import AtolError
from optimum.exporters.onnx import (
    OnnxConfig,
    OnnxConfigWithPast,
    export_models,
    get_metaclip_2_models_for_export,
    main_export,
    validate_models_outputs,
)
from optimum.exporters.onnx.base import ConfigBehavior
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
from optimum.exporters.onnx.utils import get_sana_models_for_export
from optimum.exporters.tasks import TasksManager
from optimum.exporters.utils import (
    get_decoder_models_for_export,
    get_diffusion_models_for_export,
    get_encoder_decoder_models_for_export,
    get_speecht5_models_for_export,
)
from optimum.utils import DummyPastKeyValuesGenerator, DummyTextInputGenerator, NormalizedTextConfig
from optimum.utils.testing_utils import grid_parameters, require_diffusers


SEED = 42


@require_onnx
class OnnxUtilsTestCase(TestCase):
    """Covers all the utilities involved to export ONNX models."""

    def test_flatten_output_collection_property(self):
        """This test ensures we correctly flatten nested collection such as the one we use when returning past_keys.
        past_keys = Tuple[Tuple].

        ONNX exporter will export nested collections as ${collection_name}.${level_idx_0}.${level_idx_1}...${idx_n}
        """
        self.assertEqual(
            OnnxConfig.flatten_output_collection_property("past_key", [[0], [1], [2]]),
            {
                "past_key.0": 0,
                "past_key.1": 1,
                "past_key.2": 2,
            },
        )


def _get_models_to_test(export_models_dict: dict, library_name: str = "transformers"):
    models_to_test = []

    for model_type, model_names_tasks in export_models_dict.items():
        task_config_mapping = TasksManager.get_supported_tasks_for_model_type(
            model_type, "onnx", library_name=library_name
        )

        if isinstance(model_names_tasks, str):  # test export of all tasks on the same model
            tasks = list(task_config_mapping.keys())
            model_tasks = {model_names_tasks: tasks}
        else:
            unique_tasks = set()
            for tasks in model_names_tasks.values():
                for task in tasks:
                    unique_tasks.add(task)
            n_tested_tasks = len(unique_tasks)
            if n_tested_tasks != len(task_config_mapping):
                raise ValueError(f"Not all tasks are tested for {model_type}.")
            model_tasks = model_names_tasks  # possibly, test different tasks on different models

        for model_name, tasks in model_tasks.items():
            for task in tasks:
                if model_type == "encoder-decoder" and task == "seq2seq-lm-with-past":
                    # The model uses bert as decoder and does not support past key values
                    continue

                onnx_config_constructor = TasksManager.get_exporter_config_constructor(
                    model_type=model_type,
                    exporter="onnx",
                    task=task,
                    model_name=model_name,
                    library_name=library_name,
                )

                models_to_test.append(
                    (f"{model_type}_{task}", model_type, model_name, task, onnx_config_constructor, False)
                )

                if any(
                    task == ort_special_task
                    for ort_special_task in [
                        "text-generation",
                        "text2text-generation",
                        "automatic-speech-recognition",
                        "image-to-text",
                    ]
                ):
                    models_to_test.append(
                        (
                            f"{model_type}_{task}_monolith",
                            model_type,
                            model_name,
                            task,
                            onnx_config_constructor,
                            True,
                        )
                    )
    return sorted(models_to_test)


# TODO: TOO MUCH HACKING FOR TESTING
class OnnxExportTestCase(TestCase):
    """Integration tests ensuring supported models are correctly exported."""

    def _onnx_export(
        self,
        test_name: str,
        model_type: str,
        model_name: str,
        task: str,
        onnx_config_class_constructor,
        shapes_to_validate: dict,
        monolith: bool,
        device="cpu",
    ):
        library_name = TasksManager.infer_library_from_model(model_name)

        if library_name == "timm":
            model_class = TasksManager.get_model_class_for_task(task, library=library_name)
            model = model_class(f"hf_hub:{model_name}", pretrained=True, exportable=True)
            TasksManager.standardize_model_attributes(model, library_name=library_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            model_class = TasksManager.get_model_class_for_task(task, model_type=config.model_type)
            model = model_class.from_config(config)

        # Dynamic axes aren't supported for YOLO-like models. This means they cannot be exported to ONNX on CUDA devices.
        # See: https://github.com/ultralytics/yolov5/pull/8378
        if model.__class__.__name__.startswith("Yolos") and device != "cpu":
            return

        onnx_config = onnx_config_class_constructor(model.config)

        # We need to set this to some value to be able to test the outputs values for batch size > 1.
        if (
            isinstance(onnx_config, OnnxConfigWithPast)
            and getattr(model.config, "pad_token_id", None) is None
            and task == "text-classification"
        ):
            model.config.pad_token_id = 0

        if is_torch_available():
            from optimum.utils.import_utils import _torch_version, _transformers_version

            if not onnx_config.is_transformers_support_available:
                pytest.skip(
                    "Skipping due to incompatible Transformers version. Minimum required is"
                    f" {onnx_config.MIN_TRANSFORMERS_VERSION}, got: {_transformers_version}"
                )

            if not onnx_config.is_torch_support_available:
                pytest.skip(
                    "Skipping due to incompatible PyTorch version. Minimum required is"
                    f" {onnx_config.MIN_TORCH_VERSION}, got: {_torch_version}"
                )

        atol = onnx_config.ATOL_FOR_VALIDATION
        if isinstance(atol, dict):
            atol = atol[task.replace("-with-past", "")]

        model_kwargs = None
        if (
            model.config.is_encoder_decoder
            and task.startswith(
                (
                    "text2text-generation",
                    "automatic-speech-recognition",
                    "image-to-text",
                    "feature-extraction-with-past",
                )
            )
            and monolith is False
        ):
            models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
        elif task.startswith("text-generation") and monolith is False:
            models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
        elif model.config.model_type == "speecht5":
            model_kwargs = {"vocoder": "fxmarty/speecht5-hifigan-tiny"}
            models_and_onnx_configs = get_speecht5_models_for_export(model, onnx_config, model_kwargs)
        elif model.config.model_type == "metaclip_2":
            models_and_onnx_configs = get_metaclip_2_models_for_export(model, onnx_config)
        else:
            models_and_onnx_configs = {"model": (model, onnx_config)}

        with TemporaryDirectory() as tmpdirname:
            _, onnx_outputs = export_models(
                models_and_onnx_configs=models_and_onnx_configs,
                opset=onnx_config.DEFAULT_ONNX_OPSET,
                output_dir=Path(tmpdirname),
                device=device,
                model_kwargs=model_kwargs,
            )
            input_shapes_iterator = grid_parameters(shapes_to_validate, yield_dict=True, add_test_name=False)
            for input_shapes in input_shapes_iterator:
                skip = False
                for model_onnx_conf in models_and_onnx_configs.values():
                    if (
                        hasattr(model_onnx_conf[0].config, "max_position_embeddings")
                        and input_shapes["sequence_length"] >= model_onnx_conf[0].config.max_position_embeddings
                    ):
                        skip = True
                        break
                    if (
                        model_type == "groupvit"
                        and input_shapes["sequence_length"]
                        >= model_onnx_conf[0].config.text_config.max_position_embeddings
                    ):
                        skip = True
                        break
                if skip:
                    continue

                try:
                    validate_models_outputs(
                        models_and_onnx_configs=models_and_onnx_configs,
                        onnx_named_outputs=onnx_outputs,
                        atol=atol,
                        output_dir=Path(tmpdirname),
                        input_shapes=input_shapes,
                        device=device,
                        model_kwargs=model_kwargs,
                    )
                except AtolError as e:
                    print(f"The ONNX export succeeded with the warning: {e}")

                gc.collect()

    def _onnx_export_diffusion_models(self, model_type: str, model_name: str, device="cpu"):
        pipeline = TasksManager.get_model_from_task(model_type, model_name, device=device)

        if model_type == "sana":
            models_and_onnx_configs = get_sana_models_for_export(pipeline)
            atol = 2e-3
        else:
            models_and_onnx_configs = get_diffusion_models_for_export(pipeline)
            atol = None

        with TemporaryDirectory() as tmpdirname:
            _, onnx_outputs = export_models(
                models_and_onnx_configs=models_and_onnx_configs,
                output_dir=Path(tmpdirname),
                device=device,
            )
            validate_models_outputs(
                models_and_onnx_configs=models_and_onnx_configs,
                onnx_named_outputs=onnx_outputs,
                output_dir=Path(tmpdirname),
                use_subprocess=False,
                atol=atol,
            )

    def test_all_models_tested(self):
        # make sure we test all models
        missing_models_set = (
            TasksManager._SUPPORTED_CLI_MODEL_TYPE
            - set(PYTORCH_EXPORT_MODELS_TINY.keys())
            - set(PYTORCH_TIMM_MODEL.keys())
            - set(PYTORCH_SENTENCE_TRANSFORMERS_MODEL.keys())
        )
        if len(missing_models_set) > 0:
            self.fail(f"Not testing all models. Missing models: {missing_models_set}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    # TODO: I just enabled it and it seems to hang for some reason, investigate later
    @slow
    @pytest.mark.run_slow
    def test_pytorch_export_on_cpu(
        self, test_name, model_type, model_name, task, onnx_config_class_constructor, monolith
    ):
        if model_type == "speecht5" and monolith:
            self.skipTest("unsupported export")

        self._onnx_export(
            test_name,
            model_type,
            model_name,
            task,
            onnx_config_class_constructor,
            shapes_to_validate=VALIDATE_EXPORT_ON_SHAPES_SLOW,
            monolith=monolith,
            device="cpu",
        )

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS_TINY))
    @require_torch
    @require_vision
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_pytorch_export_on_cuda(
        self, test_name, model_type, model_name, task, onnx_config_class_constructor, monolith
    ):
        if model_type == "speecht5" and monolith:
            self.skipTest("unsupported export")

        self._onnx_export(
            test_name,
            model_type,
            model_name,
            task,
            onnx_config_class_constructor,
            shapes_to_validate=VALIDATE_EXPORT_ON_SHAPES_SLOW,
            monolith=monolith,
            device="cuda",
        )

    @parameterized.expand(PYTORCH_DIFFUSION_MODEL.items())
    @require_torch
    @require_vision
    @require_diffusers
    def test_pytorch_export_for_diffusion_models(self, model_type, model_name):
        self._onnx_export_diffusion_models(model_type, model_name)

    @parameterized.expand(PYTORCH_DIFFUSION_MODEL.items())
    @require_torch
    @require_vision
    @require_diffusers
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_pytorch_export_for_diffusion_models_cuda(self, model_type, model_name):
        self._onnx_export_diffusion_models(model_type, model_name, device="cuda")


class CustomWhisperOnnxConfig(WhisperOnnxConfig):
    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        common_outputs = super().outputs

        if self._behavior is ConfigBehavior.ENCODER:
            for i in range(self._config.encoder_layers):
                common_outputs[f"encoder_attentions.{i}"] = {0: "batch_size"}
        elif self._behavior is ConfigBehavior.DECODER:
            for i in range(self._config.decoder_layers):
                common_outputs[f"decoder_attentions.{i}"] = {0: "batch_size", 3: "decoder_sequence_length"}
            for i in range(self._config.decoder_layers):
                common_outputs[f"cross_attentions.{i}"] = {0: "batch_size", 3: "cross_attention_length"}

        return common_outputs

    @property
    def torch_to_onnx_output_map(self):
        if self._behavior is ConfigBehavior.ENCODER:
            # The encoder export uses WhisperEncoder that returns the key "attentions"
            return {"attentions": "encoder_attentions"}
        else:
            return {}


class MPTDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """MPT swaps the two last dimensions for the key cache compared to usual transformers
    decoder models, thus the redefinition here.
    """

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        past_key_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.sequence_length,
        )
        past_value_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


class CustomMPTOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, MPTDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = MPTDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        hidden_size="d_model", num_layers="n_layers", num_attention_heads="n_heads"
    )

    def add_past_key_values(self, inputs_or_outputs: dict[str, dict[int, str]], direction: str):
        """Adapted from https://github.com/huggingface/optimum/blob/v1.9.0/optimum/exporters/onnx/base.py#L625."""
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 3: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: decoder_sequence_name}


def fn_get_submodels_custom(model):
    return {"model": model}


class OnnxCustomExport(TestCase):
    @parameterized.expand([False, True])
    def test_custom_export_official_model_whisper(self, dynamo: bool):
        model_id = "openai/whisper-tiny.en"
        config = AutoConfig.from_pretrained(model_id)
        custom_onnx_config = CustomWhisperOnnxConfig(config=config, task="automatic-speech-recognition")

        encoder_config = custom_onnx_config.with_behavior("encoder")
        decoder_config = custom_onnx_config.with_behavior("decoder", use_past=False)
        decoder_with_past_config = custom_onnx_config.with_behavior("decoder", use_past=True)

        custom_onnx_configs = {
            "encoder_model": encoder_config,
            "decoder_model": decoder_config,
            "decoder_with_past_model": decoder_with_past_config,
        }

        with TemporaryDirectory() as tmpdirname:
            main_export(
                model_id,
                output=tmpdirname,
                no_post_process=True,
                model_kwargs={"output_attentions": True},
                custom_onnx_configs=custom_onnx_configs,
                dynamo=dynamo,
            )

            model = onnx.load(os.path.join(tmpdirname, "decoder_model.onnx"))

            output_names = [outp.name for outp in model.graph.output]
            assert "decoder_attentions.0" in output_names
            assert "cross_attentions.0" in output_names

    @parameterized.expand([False, True])
    def test_custom_export_official_model_tiny_llm(self, dynamo: bool):
        model_id = "arnir0/Tiny-LLM"

        with TemporaryDirectory() as tmpdirname:
            main_export(
                model_id,
                output=tmpdirname,
                no_post_process=True,
                model_kwargs={"output_attentions": True},
                dynamo=dynamo,
            )

            model = onnx.load(os.path.join(tmpdirname, "model.onnx"))

            output_names = [outp.name for outp in model.graph.output]
            assert "logits" in output_names
            assert "present.0.key" in output_names

    @parameterized.expand([(None,), (fn_get_submodels_custom,)])
    @mock.patch.dict("sys.modules", triton_pre_mlir=mock.Mock())
    def test_custom_export_trust_remote(self, fn_get_submodels):
        model_id = "echarlaix/tiny-mpt-random-remote-code"
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        onnx_config_with_past = CustomMPTOnnxConfig(config, task="text-generation", use_past=True)
        custom_onnx_configs = {"model": onnx_config_with_past}

        with TemporaryDirectory() as tmpdirname:
            main_export(
                model_id,
                output=tmpdirname,
                task="text-generation-with-past",
                trust_remote_code=True,
                custom_onnx_configs=custom_onnx_configs,
                no_post_process=True,
                fn_get_submodels=partial(fn_get_submodels) if fn_get_submodels else None,
                opset=14,
            )

    def test_custom_export_trust_remote_error(self):
        model_id = "optimum-internal-testing/tiny-random-arctic"

        with self.assertRaises(ValueError) as context, TemporaryDirectory() as tmpdirname:
            main_export(
                model_id,
                output=tmpdirname,
                task="text-generation-with-past",
                trust_remote_code=True,
                no_post_process=True,
            )

        self.assertIn("custom or unsupported architecture", str(context.exception))


def _make_traceable_sdpa_layer_norm(dtype, mask):
    import torch

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm(4).to(dtype=dtype)
            if mask is not None:
                self.register_buffer("attn_mask", mask)
            else:
                self.attn_mask = None

        def forward(self, query, key, value):
            from optimum.exporters.onnx.model_patcher import traceable_scaled_dot_product_attention

            attended = traceable_scaled_dot_product_attention(query, key, value, attn_mask=self.attn_mask)
            return attended, self.layer_norm(attended)

    return Module().eval()


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "float32"])
@require_torch
def test_traceable_sdpa_layer_norm_onnx_dtype(precision, mask_kind, tmp_path):
    """SDPA keeps its query dtype through a following LayerNorm in legacy ONNX export."""
    import numpy as np
    import onnxruntime as ort
    import torch

    from optimum.exporters.onnx.model_patcher import (
        original_scaled_dot_product_attention,
        traceable_scaled_dot_product_attention,
    )

    dtype = torch.float16 if precision == "fp16" else torch.float32
    atol = rtol = 1e-3 if dtype == torch.float16 else 1e-5
    query = torch.zeros((1, 2, 3, 4), dtype=dtype)
    key = torch.zeros_like(query)
    pattern = torch.tensor([1, -1, 1, -1], dtype=dtype).reshape(1, 1, 1, 4)
    value = (torch.arange(1, 4, dtype=dtype).reshape(1, 1, 3, 1) * pattern).expand(1, 2, 3, 4).contiguous()
    mask = None
    if mask_kind == "bool":
        mask = torch.ones((1, 1, 3, 3), dtype=torch.bool)
        mask[..., -1, :] = False
    elif mask_kind == "float32":
        mask = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        # Keep this additive-mask control finite; the boolean case covers a fully masked query row.
        mask[..., -1, -1] = -2.0

    module = _make_traceable_sdpa_layer_norm(dtype, mask)
    with torch.inference_mode():
        expected_attention = original_scaled_dot_product_attention(query, key, value, attn_mask=mask)
        wrapped_attention = traceable_scaled_dot_product_attention(query, key, value, attn_mask=mask)
        expected_normalized = module.layer_norm(expected_attention)
        module_attention, module_normalized = module(query, key, value)
    assert wrapped_attention.dtype == dtype
    torch.testing.assert_close(wrapped_attention, expected_attention, atol=atol, rtol=rtol)
    torch.testing.assert_close(module_attention, expected_attention, atol=atol, rtol=rtol)
    torch.testing.assert_close(module_normalized, expected_normalized, atol=atol, rtol=rtol)

    output_path = tmp_path / "sdpa-layer-norm.onnx"
    torch.onnx.export(
        module,
        (query, key, value),
        str(output_path),
        input_names=["query", "key", "value"],
        output_names=["attention", "normalized"],
        opset_version=20,
        dynamo=False,
        do_constant_folding=False,
    )
    onnx.checker.check_model(str(output_path), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output_path), options, providers=["CPUExecutionProvider"])
    attention, normalized = session.run(
        ["attention", "normalized"], {"query": query.numpy(), "key": key.numpy(), "value": value.numpy()}
    )
    for output, expected_tensor in [(attention, expected_attention), (normalized, expected_normalized)]:
        expected_np = expected_tensor.numpy()
        assert output.dtype == expected_np.dtype
        assert np.isfinite(output).all()
        np.testing.assert_allclose(output, expected_np, atol=atol, rtol=rtol)


@pytest.mark.parametrize("with_past", [False, True], ids=["without-cache", "with-cache"])
@require_torch
def test_tiny_whisper_sdpa_fp16_cpu_onnx(with_past, tmp_path):
    """An offline tiny Whisper model runs encoder and decoder in CPU ONNX Runtime at FP16."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    from optimum.exporters.onnx import onnx_export_from_model

    config = WhisperConfig(
        vocab_size=32,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=16,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = WhisperForConditionalGeneration(config).eval().to(dtype=torch.float16)
    features = torch.linspace(-1, 1, 8 * 32, dtype=torch.float16).reshape(1, 8, 32)
    decoder_ids = torch.tensor([[1, 3]], dtype=torch.int64)
    task = "automatic-speech-recognition-with-past" if with_past else "automatic-speech-recognition"
    onnx_export_from_model(
        model,
        output=tmp_path / "whisper-onnx",
        task=task,
        opset=20,
        monolith=False,
        no_post_process=True,
        do_validation=False,
        device="cpu",
        dynamo=False,
        batch_size=1,
        feature_size=8,
        nb_max_frames=32,
        sequence_length=2,
    )

    model_dir = tmp_path / "whisper-onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    encoder_path = model_dir / "encoder_model.onnx"
    decoder_path = model_dir / "decoder_model.onnx"
    for path in [encoder_path, decoder_path] + ([model_dir / "decoder_with_past_model.onnx"] if with_past else []):
        onnx.checker.check_model(str(path), full_check=True)

    encoder = ort.InferenceSession(str(encoder_path), options, providers=["CPUExecutionProvider"])
    encoder_input = encoder.get_inputs()[0].name
    encoder_result = encoder.run(None, {encoder_input: features.numpy()})
    encoder_outputs = {item.name: value for item, value in zip(encoder.get_outputs(), encoder_result)}
    hidden_states = encoder_outputs.get("last_hidden_state", encoder_result[0])

    decoder = ort.InferenceSession(str(decoder_path), options, providers=["CPUExecutionProvider"])
    decoder_inputs = {item.name for item in decoder.get_inputs()}
    decoder_feed = {}
    if "input_ids" in decoder_inputs:
        decoder_feed["input_ids"] = decoder_ids.numpy()
    if "encoder_hidden_states" in decoder_inputs:
        decoder_feed["encoder_hidden_states"] = hidden_states
    decoder_result = decoder.run(None, decoder_feed)
    decoder_outputs = {item.name: value for item, value in zip(decoder.get_outputs(), decoder_result)}
    logits = decoder_outputs["logits"]

    with torch.inference_mode():
        reference = model(input_features=features, decoder_input_ids=decoder_ids).logits.numpy()
    assert logits.dtype == np.float16
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(logits, reference, atol=1e-3, rtol=1e-3)

    if with_past:
        cached_ids = torch.tensor([[4]], dtype=torch.int64)
        past_decoder = ort.InferenceSession(
            str(model_dir / "decoder_with_past_model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        past_feed = {"input_ids": cached_ids.numpy()}
        for item in past_decoder.get_inputs():
            if item.name.startswith("past_key_values."):
                present_name = item.name.replace("past_key_values.", "present.", 1)
                past_feed[item.name] = decoder_outputs[present_name]
        past_result = past_decoder.run(None, past_feed)
        past_outputs = {item.name: value for item, value in zip(past_decoder.get_outputs(), past_result)}
        cached_logits = past_outputs["logits"]
        with torch.inference_mode():
            full_ids = torch.tensor([[1, 3, 4]], dtype=torch.int64)
            cached_reference = model(input_features=features, decoder_input_ids=full_ids).logits[:, -1:, :].numpy()
        assert cached_logits.dtype == np.float16
        assert np.isfinite(cached_logits).all()
        np.testing.assert_allclose(cached_logits, cached_reference, atol=1e-3, rtol=1e-3)
