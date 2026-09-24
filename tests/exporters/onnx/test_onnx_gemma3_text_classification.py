# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Regression tests for issue #2488:
    gemma3_text + text-classification ONNX export support.

These tests verify that:
1. The `gemma3_text` model type is registered for the `text-classification` task
   in the ONNX exporter TasksManager.
2. The exporter config constructor can be retrieved for the task.
3. A minimal end-to-end ONNX export produces valid output.

Run with:
    pytest tests/exporters/test_onnx_gemma3_text_classification.py -v
"""

import unittest

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tasks_manager():
    """Import TasksManager, triggering ONNX config registration as a side-effect."""
    from optimum.exporters.tasks import TasksManager
    try:
        import optimum.exporters.onnx  # noqa: F401 – registers configs
    except ImportError:
        pass
    return TasksManager


def _onnx_available():
    try:
        import optimum.exporters.onnx  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Unit tests (no model download required)
# ---------------------------------------------------------------------------

class TestGemma3TextOnnxTaskRegistration(unittest.TestCase):
    """Verify that the ONNX exporter exposes text-classification for gemma3_text."""

    @classmethod
    def setUpClass(cls):
        if not _onnx_available():
            raise unittest.SkipTest("optimum.exporters.onnx not available")
        cls.TasksManager = _get_tasks_manager()

    def test_gemma3_text_registered(self):
        """gemma3_text must appear in _SUPPORTED_MODEL_TYPE."""
        self.assertIn(
            "gemma3_text",
            self.TasksManager._SUPPORTED_MODEL_TYPE,
            "gemma3_text not registered in TasksManager._SUPPORTED_MODEL_TYPE",
        )

    def test_gemma3_text_onnx_tasks_include_text_classification(self):
        """The ONNX backend for gemma3_text must list text-classification."""
        model_info = self.TasksManager._SUPPORTED_MODEL_TYPE.get("gemma3_text", {})
        onnx_tasks = list(model_info.get("onnx", {}).keys())
        self.assertIn(
            "text-classification",
            onnx_tasks,
            f"text-classification missing from gemma3_text ONNX tasks. "
            f"Found: {onnx_tasks}",
        )

    def test_gemma3_text_onnx_generation_tasks_still_present(self):
        """Existing text-generation tasks must not have been removed."""
        model_info = self.TasksManager._SUPPORTED_MODEL_TYPE.get("gemma3_text", {})
        onnx_tasks = list(model_info.get("onnx", {}).keys())
        for expected_task in (
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
        ):
            self.assertIn(
                expected_task,
                onnx_tasks,
                f"Task {expected_task!r} missing from gemma3_text ONNX tasks",
            )

    def test_gemma3_text_config_constructor_retrievable(self):
        """TasksManager.get_exporter_config_constructor must succeed for text-classification."""
        from transformers import Gemma3TextConfig

        pretrained_config = Gemma3TextConfig()
        constructor = self.TasksManager.get_exporter_config_constructor(
            model_type="gemma3_text",
            exporter="onnx",
            task="text-classification",
            model_type_for_task=pretrained_config,
        )
        self.assertIsNotNone(constructor, "Config constructor should not be None")

    def test_gemma3_text_config_instantiation(self):
        """The retrieved config constructor must produce a valid OnnxConfig object."""
        from transformers import Gemma3TextConfig

        pretrained_config = Gemma3TextConfig()
        constructor = self.TasksManager.get_exporter_config_constructor(
            model_type="gemma3_text",
            exporter="onnx",
            task="text-classification",
            model_type_for_task=pretrained_config,
        )
        onnx_config = constructor(pretrained_config)
        # Should have inputs / outputs properties
        self.assertTrue(
            hasattr(onnx_config, "inputs"),
            "OnnxConfig must expose an 'inputs' property",
        )
        self.assertTrue(
            hasattr(onnx_config, "outputs"),
            "OnnxConfig must expose an 'outputs' property",
        )

    def test_get_supported_tasks_for_model_type(self):
        """get_supported_tasks_for_model must return text-classification for gemma3_text."""
        supported = self.TasksManager.get_supported_tasks_for_model_type(
            "gemma3_text", "onnx"
        )
        self.assertIn(
            "text-classification",
            supported,
            f"text-classification not in supported tasks: {supported}",
        )

    def test_no_regression_text_generation(self):
        """text-generation-with-past must still work for gemma3_text after the fix."""
        supported = self.TasksManager.get_supported_tasks_for_model_type(
            "gemma3_text", "onnx"
        )
        self.assertIn("text-generation-with-past", supported)


# ---------------------------------------------------------------------------
# End-to-end export test (requires model download + onnxruntime)
# Skipped automatically if not available.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _onnx_available(), reason="optimum.exporters.onnx not available")
class TestGemma3TextClassificationExport(unittest.TestCase):
    """Lightweight end-to-end export test using a tiny synthetic Gemma3Text model.

    We deliberately do NOT download google/gemma-3-270m (gated model) and instead
    construct the smallest possible in-memory model that shares the same
    model_type='gemma3_text' config, verifying that:
      - The ONNX export pipeline accepts the task
      - The resulting ONNX model produces `logits` output
      - ONNX Runtime can load and run the model

    Requires: onnxruntime, transformers >= 4.53
    """

    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime  # noqa: F401
            from transformers import Gemma3TextConfig, Gemma3TextForSequenceClassification
        except ImportError as e:
            raise unittest.SkipTest(f"Required package not available: {e}")

        # Build smallest possible config
        cls.config = Gemma3TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            num_labels=2,  # binary classification
        )
        cls.model = Gemma3TextForSequenceClassification(cls.config)
        cls.model.eval()

    def test_export_produces_logits(self):
        """Export the tiny model and verify the ONNX output contains 'logits'."""
        import os
        import tempfile

        import onnxruntime as ort

        from optimum.exporters.onnx.convert import export_models
        from optimum.exporters.tasks import TasksManager

        with tempfile.TemporaryDirectory() as export_dir:
            # Export to ONNX
            onnx_config = TasksManager.get_exporter_config_constructor('onnx', self.model, task='text-classification')(self.config)
            export_models(
                models_and_onnx_configs={'model.onnx': (self.model, onnx_config)},
                opset=14,
                output_dir=export_dir,
            )

            # Check files were created
            exported_file = os.path.join(export_dir, 'model.onnx')
            self.assertTrue(os.path.exists(exported_file), f'{exported_file} was not generated')

            # Verify the model has 'logits' as an output
            session = ort.InferenceSession(exported_file, providers=['CPUExecutionProvider'])
            output_names = [out.name for out in session.get_outputs()]
            self.assertIn('logits', output_names, f'Expected logits in output, got {output_names}')
