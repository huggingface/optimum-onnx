#  Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""ORTModelForXXX classes related to seq2seq, allowing to run ONNX Models with ONNX Runtime using the same API as Transformers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForVision2Seq,
    GenerationConfig,
    GenerationMixin,
    Pix2StructForConditionalGeneration,
    WhisperForConditionalGeneration,
)
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES

from onnxruntime import InferenceSession, SessionOptions
from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime.base import ORTParentMixin, ORTSessionMixin
from optimum.onnxruntime.constants import (
    DECODER_MERGED_ONNX_FILE_PATTERN,
    DECODER_ONNX_FILE_PATTERN,
    DECODER_WITH_PAST_ONNX_FILE_PATTERN,
    ENCODER_ONNX_FILE_PATTERN,
    ONNX_DECODER_MERGED_NAME,
    ONNX_DECODER_NAME,
    ONNX_DECODER_WITH_PAST_NAME,
    ONNX_ENCODER_NAME,
    ONNX_FILE_PATTERN,
)
from optimum.onnxruntime.modeling import ORTModel
from optimum.onnxruntime.utils import DummyWhisperModel, prepare_providers_and_provider_options
from optimum.utils import is_transformers_version
from optimum.utils.file_utils import find_files_matching_pattern
from optimum.utils.logging import get_logger
from optimum.utils.save_utils import maybe_save_preprocessors


if is_transformers_version(">=", "4.48.0"):
    from transformers import MoonshineForConditionalGeneration
else:
    MoonshineForConditionalGeneration = None

if is_transformers_version(">=", "4.49.0"):
    # Because of some type hint logic added in PreTrainedModel we use
    # WhisperGenerationMixin instead of always using WhisperForConditionalGeneration
    from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
else:
    WhisperGenerationMixin = WhisperForConditionalGeneration


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = get_logger(__name__)

ONNX_MODEL_END_DOCSTRING = r"""
    This model inherits from [`~onnxruntime.modeling.ORTModelForConditionalGeneration`], check its documentation for the generic methods the
    library implements for all its model (such as downloading or saving).

    This class should be initialized using the [`onnxruntime.modeling.ORTModelForConditionalGeneration.from_pretrained`] method.
"""

SEQ2SEQ_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
"""

SPEECH_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor`):
            Mel / fbank features extracted from the raw speech waveform. `(batch_size, feature_size, encoder_sequence_length)`.
"""

MOONSHINE_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor`):
            Float values of the raw speech waveform. `(batch_size, audio_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
"""

VISION_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor`):
            Features extracted from an Image. This tensor should be of shape `(batch_size, num_channels, height, width)`.
"""

PIX2STRUCT_INPUTS_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_channels x patch_height x patch_width)`):
            Flattened and padded pixel values.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding pixel values.
"""

DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder `input_ids`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of input sequence tokens in the vocabulary of shape `(batch_size, encoder_sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, encoder_sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor`):
            Mel features extracted from the raw speech waveform.
            `(batch_size, feature_size, encoder_sequence_length)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

MOONSHINE_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor`):
            Float values of the raw speech waveform. `(batch_size, audio_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

VISION_ENCODER_DECODER_SEQ2SEQ_ONNX_MODEL_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor`):
            Features extracted from an Image. This tensor should be of shape
            `(batch_size, num_channels, height, width)`.
        decoder_input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, decoder_sequence_length)`.
        encoder_outputs (`torch.FloatTensor`):
            The encoder `last_hidden_state` of shape `(batch_size, encoder_sequence_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

PIX2STRUCT_ONNX_MODEL_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, seq_length, hidden_size)`):
            Flattened pixel patches. the `hidden_size` is obtained by the following formula: `hidden_size` =
            `num_channels` * `patch_size` * `patch_size`
            The process of flattening the pixel patches is done by `Pix2StructProcessor`.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.
            Pix2StructText uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, decoder_sequence_length, embed_size_per_head)` and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
"""

_TOKENIZER_FOR_DOC = "AutoTokenizer"
_PROCESSOR_FOR_DOC = "AutoProcessor"
_IMAGE_PROCESSOR_FOR_DOC = "AutoImageProcessor"

TRANSLATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Eustache and I like to", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_translation = pipeline("translation_en_to_de", model=model, tokenizer=tokenizer)

    >>> text = "My name is Eustache."
    >>> pred = onnx_translation(text)
    ```
"""


AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor.feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> gen_tokens = model.generate(inputs=inputs.input_features)
    >>> outputs = processor.tokenizer.batch_decode(gen_tokens)
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}
    >>> from datasets import load_dataset

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> speech_recognition = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> pred = speech_recognition(ds[0]["audio"]["array"])
    ```
"""


IMAGE_TO_TEXT_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}, {tokenizer_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from PIL import Image
    >>> import requests


    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> tokenizer = {tokenizer_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> inputs = processor(image, return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, {tokenizer_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}
    >>> from PIL import Image
    >>> import requests


    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> tokenizer = {tokenizer_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_to_text = pipeline("image-to-text", model=model, tokenizer=tokenizer, feature_extractor=processor, image_processor=processor)
    >>> pred = image_to_text(image)
    ```
"""

PIX2STRUCT_EXAMPLE = r"""
    Example of pix2struct:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> from PIL import Image
    >>> import requests

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True, use_io_binding=True)

    >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
    >>> inputs = processor(images=image, text=question, return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs)
    >>> outputs = processor.batch_decode(gen_tokens, skip_special_tokens=True)
    ```
"""


class ORTEncoder(ORTSessionMixin):
    """Encoder of an encoder-decoder model for ONNX Runtime inference."""

    main_input_name = "input_ids"

    def __init__(
        self,
        config: PretrainedConfig,
        session: InferenceSession,
        use_io_binding: bool | None = None,
    ):
        self.config = config
        self.initialize_ort_attributes(session, use_io_binding)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> BaseModelOutput:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTEncoderForSpeech(ORTEncoder):
    """Encoder model for ONNX Runtime inference for Whisper model.

    Args:
        session (`InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    main_input_name = "input_features"

    @add_start_docstrings_to_model_forward(SPEECH_ENCODER_INPUTS_DOCSTRING)
    def forward(
        self, input_features: torch.FloatTensor, attention_mask: torch.LongTensor, **kwargs
    ) -> BaseModelOutput:
        use_torch = isinstance(input_features, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTEncoderForMoonshine(ORTEncoder):
    """Encoder model for ONNX Runtime inference for Moonshine model.

    Args:
        session (`InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    main_input_name = "input_values"

    @add_start_docstrings_to_model_forward(MOONSHINE_ENCODER_INPUTS_DOCSTRING)
    def forward(self, input_values: torch.FloatTensor, attention_mask: torch.LongTensor, **kwargs) -> BaseModelOutput:
        use_torch = isinstance(input_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "input_values": input_values,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTEncoderForVision(ORTEncoder):
    """Encoder model for ONNX Runtime inference for VisionEncoderDecoder models.

    Args:
        session (`InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    main_input_name = "pixel_values"

    @add_start_docstrings_to_model_forward(VISION_ENCODER_INPUTS_DOCSTRING)
    def forward(self, pixel_values: torch.FloatTensor, **kwargs) -> BaseModelOutput:
        use_torch = isinstance(pixel_values, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "pixel_values": pixel_values,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTEncoderForPix2Struct(ORTEncoder):
    """Encoder model for ONNX Runtime inference for Pix2Struct.

    Args:
        session (`InferenceSession`):
            The ONNX Runtime inference session associated to the encoder.
    """

    main_input_name = "flattened_patches"

    @add_start_docstrings_to_model_forward(PIX2STRUCT_INPUTS_DOCSTRING)
    def forward(
        self, flattened_patches: torch.FloatTensor, attention_mask: torch.LongTensor, **kwargs
    ) -> BaseModelOutput:
        use_torch = isinstance(flattened_patches, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        model_inputs = {
            "flattened_patches": flattened_patches,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(model_inputs)

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            last_hidden_state = output_buffers["last_hidden_state"].view(output_shapes["last_hidden_state"])
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            last_hidden_state = model_outputs["last_hidden_state"]

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoderForSeq2Seq(ORTSessionMixin):
    """Decoder of an encoder-decoder model for ONNX Runtime inference."""

    main_input_name = "input_ids"

    def __init__(
        self,
        config: PretrainedConfig,
        session: InferenceSession,
        use_io_binding: bool | None = None,
    ):
        self.config = config
        self.initialize_ort_attributes(session, use_io_binding)

        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        self.self_attention_output_names = [key for key in self.key_value_output_names if "encoder" not in key]
        self.cross_attention_output_names = [key for key in self.key_value_output_names if "encoder" in key]
        self.can_use_cache = len(self.key_value_input_names) > 0 and len(self.key_value_output_names) > 0
        self.is_merged = "use_cache_branch" in self.input_names

        if self.config.model_type == "pix2struct":
            self.vocab_size = getattr(self.config, "text_config", self.config).vocab_size
            self.num_attention_heads = getattr(self.config, "text_config", self.config).num_heads
            self.embed_size_per_head = (
                getattr(self.config, "text_config", self.config).hidden_size // self.num_attention_heads
            )
        else:
            self.vocab_size = getattr(self.config, "decoder", self.config).vocab_size
            self.num_attention_heads = getattr(self.config, "decoder", self.config).num_attention_heads
            self.embed_size_per_head = (
                getattr(self.config, "decoder", self.config).hidden_size // self.num_attention_heads
            )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> Seq2SeqLMOutput:
        use_torch = isinstance(input_ids, torch.Tensor)
        self.raise_on_numpy_input_io_binding(use_torch)

        batch_size, in_seq_len = input_ids.shape
        encoder_seq_len = encoder_hidden_states.shape[1]
        past_seq_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        out_seq_len = past_seq_len + in_seq_len

        use_cache_branch = None
        if self.is_merged:
            # NOTE: This needs to be evaluated here before creating dummy past_key_values in the following block
            # Uses cache branch of merged decoders depending on whether real past key values are passed
            use_cache_branch = torch.full((1,), past_key_values is not None, dtype=torch.bool, device=self.device)

        # Save the precomputed cross-attention key/values
        cross_attention_key_values = None
        if past_key_values is not None:
            if len(past_key_values[0]) == 4:
                cross_attention_key_values = tuple(past_key_values[i][2:4] for i in range(len(past_key_values)))

        # Generate dummy past for the first forward pass
        if len(self.key_value_input_names) > 0 and past_key_values is None:
            shape = (batch_size, self.num_attention_heads, 0, self.embed_size_per_head)
            key_or_value = torch.zeros(shape, dtype=self.dtype, device=self.device)
            past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))
        elif isinstance(past_key_values, tuple) and isinstance(past_key_values[0], tuple):
            past_key_values = sum(past_key_values, ())
            if cross_attention_key_values is not None:
                cross_attention_key_values = sum(cross_attention_key_values, ())

        # Generate dummy position cache for the first forward pass
        if "cache_position" in self.input_names and cache_position is None:
            cache_position = torch.arange(past_seq_len, out_seq_len, dtype=torch.int64, device=self.device)

        # Generate dummy attention mask for Pix2Struct text model
        if "attention_mask" in self.input_names and attention_mask is None:
            attention_mask = torch.ones((batch_size, out_seq_len), dtype=torch.int64, device=self.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "use_cache_branch": use_cache_branch,
            "cache_position": cache_position,
        }
        if past_key_values is not None:
            model_inputs.update(zip(self.key_value_input_names, past_key_values))

        outputs_to_not_bind = set()
        known_output_shapes = {"logits": (batch_size, in_seq_len, self.vocab_size)}
        if use_cache:
            # Infers the shape of the output pkv
            self_attn_shape = (batch_size, self.num_attention_heads, out_seq_len, self.embed_size_per_head)
            known_output_shapes.update(dict.fromkeys(self.self_attention_output_names, self_attn_shape))
            if cross_attention_key_values is None:
                cross_attn_shape = (batch_size, self.num_attention_heads, encoder_seq_len, self.embed_size_per_head)
                known_output_shapes.update(dict.fromkeys(self.cross_attention_output_names, cross_attn_shape))
        else:
            # we don't bind the key/values if they are not gonna be returned/used
            outputs_to_not_bind.update(self.key_value_output_names)

        if cross_attention_key_values is not None:
            # we don't bind the cross-attention key/values if they are already provided/computed
            outputs_to_not_bind.update(self.cross_attention_output_names)

        if self.use_io_binding:
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, outputs_to_not_bind=outputs_to_not_bind, known_output_shapes=known_output_shapes
            )

            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()

            logits = output_buffers.pop("logits").view(output_shapes["logits"])

            if use_cache:
                self_attention_key_values = tuple(
                    output_buffers.pop(name).view(output_shapes[name]) for name in self.self_attention_output_names
                )
                if cross_attention_key_values is None:
                    cross_attention_key_values = tuple(
                        output_buffers.pop(name).view(output_shapes[name])
                        for name in self.cross_attention_output_names
                    )
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

            logits = model_outputs.pop("logits")

            if use_cache:
                self_attention_key_values = tuple(model_outputs.pop(name) for name in self.self_attention_output_names)
                if cross_attention_key_values is None:
                    cross_attention_key_values = tuple(
                        model_outputs.pop(name) for name in self.cross_attention_output_names
                    )

        if use_cache:
            # At this point we should definitely have both self-attention and cross-attention key/values :)
            past_key_values = tuple(
                self_attention_key_values[i : i + 2] + cross_attention_key_values[i : i + 2]
                for i in range(0, len(self_attention_key_values), 2)
            )
        else:
            past_key_values = None

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)


class ORTModelForConditionalGeneration(ORTParentMixin, ORTModel, GenerationMixin):
    """Sequence-to-sequence model with a language modeling head for ONNX Runtime inference.

    Important attributes:
        config ([`PretrainedConfig`]):
            Instance of the configuration associated to the model. Initializing with a config file does
            not load the weights associated with the model, only the configuration.
        use_io_binding (`Optional[bool]`, defaults to `None`):
            Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to `True`
            if the device is CUDA, otherwise defaults to `False`.
        use_cache (`bool`):
            Whether or not past key/values cache should be used. It is determined by whether an InferenceSession for
            that was provided or not.
        providers (`List[str`]):
            The list of execution providers the model is running on.
        encoder (`ORTEncoder`):
            The encoder model.
        decoder (`ORTDecoderForSeq2Seq`):
            The decoder model.
        decoder_with_past (`Optional[ORTDecoderForSeq2Seq]`):
            The decoder model handling the past key/values if `use_cache=True`, else `None`.

    Other attributes:
        encoder_file_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_ENCODER_NAME`):
            The name of the ONNX file containing the encoder part of the model.
        decoder_file_name (`str`,  defaults to `optimum.onnxruntime.utils.ONNX_DECODER_NAME`):
            The name of the ONNX file containing the decoder part of the model.
        decoder_file_with_past_name (`str`, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_WITH_PAST_NAME`):
            The name of the ONNX file containing the decoder with past key/values part of the model.
        model_save_dir (`str`, defaults to `""`):
            The directory under which the model exported to ONNX was saved.

    """

    _is_stateful = False
    _supports_cache_class = False

    _library_name = "transformers"
    _ort_encoder_class = ORTEncoder
    _ort_decoder_class = ORTDecoderForSeq2Seq

    def __init__(
        self,
        *,
        config: PretrainedConfig = None,
        encoder_session: InferenceSession = None,
        decoder_session: InferenceSession = None,
        decoder_with_past_session: InferenceSession | None = None,
        use_io_binding: bool | None = None,
        generation_config: GenerationConfig | None = None,
        model_save_dir: str | Path | TemporaryDirectory | None = None,
    ):
        """Initialize an `ORTModelForConditionalGeneration` instance.

        Args:
            config ([`PretrainedConfig`]):
                `config` is an instance of the configuration associated to the model. Initializing with a config file
                does not load the weights associated with the model, only the configuration.
            encoder_session (`InferenceSession`):
                The ONNX Runtime inference session associated to the encoder.
            decoder_session (`InferenceSession`):
                The ONNX Runtime inference session associated to the decoder.
            decoder_with_past_session (`Optional[InferenceSession]`, *optional*, defaults to `None`):
                The ONNX Runtime inference session associated to the decoder with past key values.
            use_io_binding (``Optional[bool]`, *optional*, defaults to `None`):
                Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            generation_config (`Optional[GenerationConfig]`, *optional*, defaults to `None`):
                The generation configuration used by default when calling `generate()`.
                Refer to https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate.
            model_save_dir (``Optional[Union[str, Path, TemporaryDirectory]]`, *optional*, defaults to `None`):
                The directory under which the model exported to ONNX was saved.
        """
        super(ORTModel, self).__init__(model=encoder_session, config=config)
        self.generation_config = generation_config

        self.encoder = self._ort_encoder_class(config=config, session=encoder_session, use_io_binding=use_io_binding)
        self.decoder = self._ort_decoder_class(config=config, session=decoder_session, use_io_binding=use_io_binding)
        self.decoder_with_past = None
        if decoder_with_past_session is not None:
            self.decoder_with_past = self._ort_decoder_class(
                config=config, session=decoder_with_past_session, use_io_binding=use_io_binding
            )

        self.initialize_ort_attributes(parts=list(filter(None, {self.encoder, self.decoder, self.decoder_with_past})))

        # This attribute is needed to keep one reference on the temporary directory, since garbage collecting it
        # would end-up removing the directory containing the underlying ONNX model.
        self._model_save_dir_tempdirectory_instance = None
        if model_save_dir is None:
            self.model_save_dir = Path(encoder_session._model_path).parent
        elif isinstance(model_save_dir, TemporaryDirectory):
            self._model_save_dir_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir

        # Registers the ORTModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def _save_pretrained(self, save_directory: str | Path):
        """Saves the encoder, decoder and decoder_with_past ONNX files to the save directory.

        Args:
            save_directory (`Union[str, Path`]):
                The directory under which the models will be saved.
        """
        self.encoder.save_session(save_directory)
        self.decoder.save_session(save_directory)
        if self.decoder_with_past is not None:
            self.decoder_with_past.save_session(save_directory)

    def _save_config(self, save_directory):
        """Saves the model and generation configs to the save directory.

        Args:
            save_directory (`Union[str, Path`]):
                The directory under which the configs will be saved.
        """
        self.config.save_pretrained(save_directory)
        self.generation_config.save_pretrained(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str | Path,
        config: PretrainedConfig,
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: bool | str | None = None,
        # file options
        encoder_file_name: str | None = None,
        decoder_file_name: str | None = None,
        decoder_with_past_file_name: str | None = None,
        # session options
        provider: str = "CPUExecutionProvider",
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        session_options: SessionOptions | None = None,
        # inference options
        use_cache: bool = True,
        use_merged: bool | None = None,
        use_io_binding: bool | None = None,
        generation_config: GenerationConfig | None = None,
        dtype: torch.dtype = torch.float32,
        # other arguments
        model_save_dir: str | Path | TemporaryDirectory | None = None,
    ) -> ORTModelForConditionalGeneration:
        onnx_files = find_files_matching_pattern(
            model_id,
            ONNX_FILE_PATTERN,
            glob_pattern="**/*.onnx",
            subfolder=subfolder,
            revision=revision,
            token=token,
        )
        if len(onnx_files) == 0:
            raise FileNotFoundError(f"Could not find any ONNX model file in {model_id}")
        if Path(model_id).is_dir():
            onnx_files = [f.relative_to(model_id) for f in onnx_files]

        # we start with encoder to fail fast if something is wrong
        encoder_path = cls._infer_file_path(
            ENCODER_ONNX_FILE_PATTERN,
            onnx_files=onnx_files,
            target_file_name=encoder_file_name,
            standard_file_name=ONNX_ENCODER_NAME,
        )

        decoder_path = None
        decoder_with_past_path = None
        # We default to looking for merged decoder if user didn't disable it explicitly.
        if use_merged is not False:
            try:
                decoder_path = cls._infer_file_path(
                    DECODER_MERGED_ONNX_FILE_PATTERN,
                    onnx_files=onnx_files,
                    target_file_name=decoder_file_name,
                    standard_file_name=ONNX_DECODER_MERGED_NAME,
                )
                use_merged = True
            except FileNotFoundError:
                if use_merged is True:
                    raise
                use_merged = False

        # if the user disabled merged explicitly, or if we didn't find it, or if the user provided file names
        if use_merged is False or (decoder_file_name is not None or decoder_with_past_file_name is not None):
            decoder_path = cls._infer_file_path(
                DECODER_ONNX_FILE_PATTERN,
                onnx_files=onnx_files,
                target_file_name=decoder_file_name,
                standard_file_name=ONNX_DECODER_NAME,
            )
            if use_cache or (decoder_with_past_file_name is not None):
                decoder_with_past_path = cls._infer_file_path(
                    DECODER_WITH_PAST_ONNX_FILE_PATTERN,
                    onnx_files=onnx_files,
                    target_file_name=decoder_with_past_file_name,
                    standard_file_name=ONNX_DECODER_WITH_PAST_NAME,
                )

        encoder_path = cls._cached_file(
            model_id,
            filename=encoder_path.name,
            subfolder=encoder_path.parent.as_posix(),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
        decoder_path = cls._cached_file(
            model_id,
            filename=decoder_path.name,
            subfolder=decoder_path.parent.as_posix(),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
        if decoder_with_past_path is not None:
            decoder_with_past_path = cls._cached_file(
                model_id,
                filename=decoder_with_past_path.name,
                subfolder=decoder_with_past_path.parent.as_posix(),
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance,
        # in which case we want to keep it instead.
        if model_save_dir is None:
            model_save_dir = encoder_path.parent

        # Important: for encoder-decoder models used with ConditionalGeneration, we need to set the is_decoder flag to False
        # and the is_encoder_decoder flag to True. This is needed for the model to work correctly with generation logic.
        config.use_cache = use_cache
        if hasattr(config, "is_decoder"):
            config.is_decoder = False
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = True
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "onnxruntime"

        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except OSError:
                logger.info("Generation config file not found, creating a new one from model config.")
                generation_config = GenerationConfig.from_model_config(config)

        generation_config.use_cache = use_cache

        if is_transformers_version(">=", "4.45.0"):
            misplaced_generation_parameters = config._get_non_default_generation_parameters()
            if len(misplaced_generation_parameters) > 0:
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(generation_config, param_name, param_value)
                    setattr(config, param_name, None)

        providers, provider_options = prepare_providers_and_provider_options(
            provider=provider, providers=providers, provider_options=provider_options
        )
        encoder_session = InferenceSession(
            encoder_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )
        decoder_session = InferenceSession(
            decoder_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )

        decoder_with_past_session = None
        if decoder_with_past_path is not None:
            decoder_with_past_session = InferenceSession(
                decoder_with_past_path,
                providers=providers,
                provider_options=provider_options,
                sess_options=session_options,
            )

        return cls(
            config=config,
            encoder_session=encoder_session,
            decoder_session=decoder_session,
            decoder_with_past_session=decoder_with_past_session,
            generation_config=generation_config,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _export(
        cls,
        model_id: str | Path,
        config: PretrainedConfig,
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: bool | str | None = None,
        # inference options
        use_merged: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> ORTModelForConditionalGeneration:
        # this is guaranteed to work since we it uses a mapping from model classes to task names
        # instead of relying on the hub metadata or the model configuration
        task = TasksManager._infer_task_from_model_or_model_class(model_class=cls.auto_model_class)
        if use_cache or use_merged:
            task += "-with-past"

        if kwargs.get("task") is not None:
            raise ValueError(
                f"The `task` argument is not needed when exporting a model with `{cls.__name__}`. "
                f"The `task` is automatically inferred from the class as `{task}`."
            )

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=not use_merged,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            library_name=cls._library_name,
        )
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_cache=use_cache,
            use_merged=use_merged,
            model_save_dir=save_dir,
            **kwargs,
        )

    def get_encoder(self) -> ORTEncoder:
        return self.encoder

    def get_decoder(self) -> ORTDecoderForSeq2Seq:
        return self.decoder

    # Adapted from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    @staticmethod
    def _reorder_cache(
        past_key_values: tuple[tuple[torch.Tensor]], beam_idx: torch.LongTensor
    ) -> tuple[tuple[torch.Tensor]]:
        if (
            isinstance(past_key_values, tuple)
            and isinstance(past_key_values[0], tuple)
            and len(past_key_values[0]) == 4
        ):
            # Cached cross_attention states don't have to be reordered -> they are always the same
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:]
                for layer_past in past_key_values
            )
        elif (
            isinstance(past_key_values, tuple)
            and isinstance(past_key_values[0], tuple)
            and len(past_key_values[0]) == 2
        ):
            return tuple(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
                for layer_past in past_key_values
            )
        else:
            raise TypeError(
                f"Unexpected past_key_values: {past_key_values}. "
                "Expected tuple of tuples of length 2 or 4, "
                "where each tuple contains the past key and value tensors for each layer."
            )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        if is_transformers_version("<", "4.46.0"):
            return self._prepare_inputs_for_generation_legacy(*args, **kwargs)
        else:
            return super().prepare_inputs_for_generation(*args, **kwargs)

    def _prepare_inputs_for_generation_legacy(
        self, input_ids, past_key_values: tuple[tuple[torch.Tensor]] | None = None, **kwargs
    ) -> dict:
        if past_key_values is not None:
            past_seq_len = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_seq_len:
                remove_prefix_length = past_seq_len
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            **kwargs,
        }

    @property
    def can_use_cache(self) -> bool:
        return self.decoder.can_use_cache or (
            self.decoder_with_past is not None and self.decoder_with_past.can_use_cache
        )

    @property
    def use_merged(self) -> bool:
        logger.warning(
            f"The `{self.__class__.__name__}.use_merged` property is deprecated and will be removed in a future version. "
            "Use `model.decoder.is_merged` instead."
        )
        return self.decoder.is_merged

    @property
    def use_cache(self) -> bool:
        logger.warning(
            f"The `{self.__class__.__name__}.use_cache` property is deprecated and will be removed in a future version. "
            "Use `model.config.use_cache` to know whether the model will use cache or not during inference, and "
            "`model.can_use_cache` to know whether the model supports cache use or not."
        )
        return self.can_use_cache

    def _prepare_cache_for_generation(self, *args, **kwargs):
        return


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForSeq2SeqLM(ORTModelForConditionalGeneration):
    """Sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports bart, blenderbot, blenderbot-small, longt5, m2m_100, marian, mbart, mt5, pegasus, t5."""

    main_input_name = "input_ids"
    auto_model_class = AutoModelForSeq2SeqLM

    _ort_encoder_class = ORTEncoder

    @add_start_docstrings_to_model_forward(
        SEQ2SEQ_ONNX_MODEL_DOCSTRING
        + TRANSLATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForSeq2SeqLM",
            checkpoint="optimum/t5-small",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: BaseModelOutput | list[torch.FloatTensor] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                "`use_cache=True` was passed to the model but the loaded model does not support pkv cache reuse. "
                "Please load your current model with `use_cache=True` or re-export the original model "
                "once again with `use_cache=True` when calling the `from_pretrained` method. "
                "To re-export your model, simply set `export=True` in `from_pretrained`."
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        decoder = (
            self.decoder_with_past
            if (past_key_values is not None and self.decoder_with_past is not None)
            else self.decoder
        )
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForSpeechSeq2Seq(ORTModelForConditionalGeneration):
    """Speech sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports whisper, speech_to_text."""

    main_input_name = "input_features"
    auto_model_class = AutoModelForSpeechSeq2Seq

    _ort_encoder_class = ORTEncoderForSpeech

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Following a breaking change in transformers that relies directly on the mapping name and not on the
        # greedy model mapping (that can be extended), we need to hardcode the ortmodel in this dictionary.
        # Other pipelines do not seem to have controlflow depending on the mapping name.
        # See: https://github.com/huggingface/transformers/pull/24960/files
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES["ort_speechseq2seq"] = self.__class__.__name__

    @classmethod
    def _from_pretrained(cls, model_id: str | Path, config: PretrainedConfig, **kwargs):
        if config.model_type == "whisper":
            return ORTModelForWhisper._from_pretrained(model_id, config, **kwargs)
        elif config.model_type == "moonshine":
            return ORTModelForMoonshine._from_pretrained(model_id, config, **kwargs)
        else:
            return super()._from_pretrained(model_id, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        SPEECH_SEQ2SEQ_ONNX_MODEL_DOCSTRING
        + AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForSpeechSeq2Seq",
            checkpoint="optimum/whisper-tiny.en",
        )
    )
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                "`use_cache=True` was passed to the model but the loaded model does not support pkv cache reuse. "
                "Please load your current model with `use_cache=True` or re-export the original model "
                "once again with `use_cache=True` when calling the `from_pretrained` method. "
                "To re-export your model, simply set `export=True` in `from_pretrained`."
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_features=input_features, attention_mask=attention_mask)

        decoder = (
            self.decoder_with_past
            if (past_key_values is not None and self.decoder_with_past is not None)
            else self.decoder
        )
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForWhisper(ORTModelForSpeechSeq2Seq, WhisperGenerationMixin):
    """Whisper sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports whisper."""

    main_input_name = "input_features"
    auto_model_class = WhisperForConditionalGeneration

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = DummyWhisperModel()

    # force the use of the WhisperForConditionalGeneration generate and prepare_inputs_for_generation methods
    generate = WhisperGenerationMixin.generate
    # force the use of the WhisperForConditionalGeneration prepare_inputs_for_generation method
    prepare_inputs_for_generation = WhisperGenerationMixin.prepare_inputs_for_generation

    # this is needed to avoid circular calls
    @classmethod
    def _from_pretrained(cls, model_id: str | Path, config: PretrainedConfig, **kwargs):
        return super(ORTModelForSpeechSeq2Seq, cls)._from_pretrained(model_id, config, **kwargs)


class ORTModelForMoonshine(ORTModelForSpeechSeq2Seq):
    """Moonshine sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports moonshine."""

    main_input_name = "input_values"
    auto_model_class = MoonshineForConditionalGeneration

    _ort_encoder_class = ORTEncoderForMoonshine

    # this is needed to avoid circular calls
    @classmethod
    def _from_pretrained(cls, model_id: str | Path, config: PretrainedConfig, **kwargs):
        return super(ORTModelForSpeechSeq2Seq, cls)._from_pretrained(model_id, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        MOONSHINE_ONNX_MODEL_DOCSTRING
        + AUTOMATIC_SPEECH_RECOGNITION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForMoonshine",
            checkpoint="UsefulSensors/moonshine-base",
        )
    )
    def forward(
        self,
        input_values: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                "`use_cache=True` was passed to the model but the loaded model does not support pkv cache reuse. "
                "Please load your current model with `use_cache=True` or re-export the original model "
                "once again with `use_cache=True` when calling the `from_pretrained` method. "
                "To re-export your model, simply set `export=True` in `from_pretrained`."
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)

        decoder = (
            self.decoder_with_past
            if (past_key_values is not None and self.decoder_with_past is not None)
            else self.decoder
        )
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForVision2Seq(ORTModelForConditionalGeneration):
    """Vision sequence-to-sequence model with a language modeling head for ONNX Runtime inference. This class officially supports vision encoder-decoder and pix2struct."""

    main_input_name = "pixel_values"
    auto_model_class = AutoModelForVision2Seq

    _ort_encoder_class = ORTEncoderForVision

    @classmethod
    def _from_pretrained(cls, model_id: str | Path, config: PretrainedConfig, **kwargs):
        if config.model_type == "pix2struct":
            return ORTModelForPix2Struct._from_pretrained(model_id, config, **kwargs)
        else:
            return super()._from_pretrained(model_id, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        VISION_ENCODER_DECODER_SEQ2SEQ_ONNX_MODEL_DOCSTRING
        + IMAGE_TO_TEXT_EXAMPLE.format(
            processor_class=_IMAGE_PROCESSOR_FOR_DOC,
            tokenizer_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForVision2Seq",
            checkpoint="nlpconnect/vit-gpt2-image-captioning",
        )
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        encoder_outputs: BaseModelOutput | list[torch.FloatTensor] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                "`use_cache=True` was passed to the model but the loaded model does not support pkv cache reuse. "
                "Please load your current model with `use_cache=True` or re-export the original model "
                "once again with `use_cache=True` when calling the `from_pretrained` method. "
                "To re-export your model, simply set `export=True` in `from_pretrained`."
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(pixel_values=pixel_values)

        decoder = (
            self.decoder_with_past
            if (past_key_values is not None and self.decoder_with_past is not None)
            else self.decoder
        )
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


@add_end_docstrings(ONNX_MODEL_END_DOCSTRING)
class ORTModelForPix2Struct(ORTModelForVision2Seq):
    """Pix2Struct model with a language modeling head for ONNX Runtime inference. This class officially supports pix2struct."""

    main_input_name = "flattened_patches"
    auto_model_class = Pix2StructForConditionalGeneration

    _ort_encoder_class = ORTEncoderForPix2Struct

    # this is needed to avoid circular calls when ORTModelForVision2Seq is called to instantiate a ORTModelForPix2Struct
    @classmethod
    def _from_pretrained(cls, model_id: str | Path, config: PretrainedConfig, **kwargs):
        return super(ORTModelForVision2Seq, cls)._from_pretrained(model_id, config, **kwargs)

    @add_start_docstrings_to_model_forward(
        PIX2STRUCT_ONNX_MODEL_DOCSTRING
        + PIX2STRUCT_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_DOC,
            model_class="ORTModelForPix2Struct",
            checkpoint="google/pix2struct-ai2d-base",
        )
    )
    def forward(
        self,
        flattened_patches: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.BoolTensor | None = None,
        encoder_outputs: BaseModelOutput | list[torch.FloatTensor] | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                "`use_cache=True` was passed to the model but the loaded model does not support pkv cache reuse. "
                "Please load your current model with `use_cache=True` or re-export the original model "
                "once again with `use_cache=True` when calling the `from_pretrained` method. "
                "To re-export your model, simply set `export=True` in `from_pretrained`."
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(flattened_patches=flattened_patches, attention_mask=attention_mask)

        decoder = (
            self.decoder_with_past
            if (past_key_values is not None and self.decoder_with_past is not None)
            else self.decoder
        )
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )
