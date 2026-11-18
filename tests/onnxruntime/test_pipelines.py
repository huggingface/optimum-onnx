import logging
import tempfile
import unittest

from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    GPT2Config,
    PreTrainedTokenizerFast,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTImageProcessor,
)
from transformers import pipeline as transformers_pipeline

from optimum.onnxruntime import ORTModelForVision2Seq
from optimum.onnxruntime import pipeline as ort_pipeline
from optimum.onnxruntime.pipelines import ORT_TASKS_MAPPING, get_ort_model_class


def _create_tiny_vision_encoder_decoder_dir(tmpdir: str) -> tuple[ViTImageProcessor, PreTrainedTokenizerFast]:
    tok_obj = Tokenizer(models.WordLevel(vocab={"<unk>": 0, "<s>": 1, "</s>": 2, "cat": 3}, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()

    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        ViTConfig(
            image_size=16,
            patch_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        ),
        GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=4, bos_token_id=1, eos_token_id=2),
    )
    cfg.decoder_start_token_id = 1
    cfg.eos_token_id = 2
    cfg.pad_token_id = 0

    VisionEncoderDecoderModel(cfg).save_pretrained(tmpdir)
    image_processor = ViTImageProcessor(size={"height": 16, "width": 16})
    image_processor.save_pretrained(tmpdir)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok_obj, bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tokenizer.save_pretrained(tmpdir)
    return image_processor, tokenizer


class ORTPipelineTest(unittest.TestCase):
    def test_vision2seq_transformers_and_ort_pipeline_no_unsupported_error(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as ort_dir:
            image_processor, tokenizer = _create_tiny_vision_encoder_decoder_dir(tmpdir)
            ort_model = ORTModelForVision2Seq.from_pretrained(tmpdir, export=True)
            ort_model.save_pretrained(ort_dir)
            image_processor.save_pretrained(ort_dir)
            tokenizer.save_pretrained(ort_dir)

            image = Image.new("RGB", (16, 16))
            pipeline_logger = logging.getLogger("transformers.pipelines.base")

            with self.assertNoLogs(pipeline_logger, level=logging.ERROR):
                pipe_tf = transformers_pipeline(
                    "image-to-text",
                    model=ort_model,
                    tokenizer=tokenizer,
                    feature_extractor=image_processor,
                    image_processor=image_processor,
                    device="cpu",
                )
            outputs_tf = pipe_tf(image, max_new_tokens=3)
            self.assertIsInstance(outputs_tf, list)
            self.assertIn("generated_text", outputs_tf[0])

            with self.assertNoLogs(pipeline_logger, level=logging.ERROR):
                pipe_ort = ort_pipeline("image-to-text", model=ort_dir, device="cpu")
            outputs_ort = pipe_ort(image, max_new_tokens=3)
            self.assertIsInstance(outputs_ort, list)
            self.assertIn("generated_text", outputs_ort[0])

    def test_image_text_to_text_task_registered(self):
        self.assertIn("image-text-to-text", ORT_TASKS_MAPPING)
        self.assertIs(get_ort_model_class("image-text-to-text"), ORTModelForVision2Seq)
