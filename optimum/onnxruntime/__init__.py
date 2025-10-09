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

# ruff: noqa: F401

from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule

from optimum.utils import is_diffusers_available


_import_structure = {
    "configuration": [
        "CalibrationConfig",
        "AutoCalibrationConfig",
        "QuantizationMode",
        "AutoQuantizationConfig",
        "OptimizationConfig",
        "AutoOptimizationConfig",
        "ORTConfig",
        "QuantizationConfig",
    ],
    "constants": [
        "DECODER_MERGED_ONNX_FILE_PATTERN",
        "DECODER_ONNX_FILE_PATTERN",
        "DECODER_WITH_PAST_ONNX_FILE_PATTERN",
        "ENCODER_ONNX_FILE_PATTERN",
        "ONNX_DECODER_MERGED_NAME",
        "ONNX_DECODER_NAME",
        "ONNX_DECODER_WITH_PAST_NAME",
        "ONNX_ENCODER_NAME",
        "ONNX_FILE_PATTERN",
        "ONNX_WEIGHTS_NAME",
    ],
    "modeling": [
        "ORTModel",
        "ORTModelForAudioClassification",
        "ORTModelForAudioFrameClassification",
        "ORTModelForAudioXVector",
        "ORTModelForCustomTasks",
        "ORTModelForCTC",
        "ORTModelForFeatureExtraction",
        "ORTModelForImageClassification",
        "ORTModelForZeroShotImageClassification",
        "ORTModelForMaskedLM",
        "ORTModelForMultipleChoice",
        "ORTModelForQuestionAnswering",
        "ORTModelForSemanticSegmentation",
        "ORTModelForSequenceClassification",
        "ORTModelForTokenClassification",
        "ORTModelForImageToImage",
    ],
    "modeling_seq2seq": [
        "ORTModelForSeq2SeqLM",
        "ORTModelForSpeechSeq2Seq",
        "ORTModelForVision2Seq",
        "ORTModelForPix2Struct",
    ],
    "modeling_decoder": ["ORTModelForCausalLM"],
    "optimization": ["ORTOptimizer"],
    "pipelines": ["pipeline"],
    "quantization": ["ORTQuantizer"],
    "utils": ["ORTQuantizableOperator"],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()  # noqa: TRY301
except OptionalDependencyNotAvailable:
    _import_structure["dummy_objects"] = [
        "ORTDiffusionPipeline",
        "ORTPipelineForText2Image",
        "ORTPipelineForImage2Image",
        "ORTPipelineForInpainting",
        # flux
        "ORTFluxPipeline",
        # lcm
        "ORTLatentConsistencyModelImg2ImgPipeline",
        "ORTLatentConsistencyModelPipeline",
        # sd3
        "ORTStableDiffusion3Img2ImgPipeline",
        "ORTStableDiffusion3InpaintPipeline",
        "ORTStableDiffusion3Pipeline",
        # sd
        "ORTStableDiffusionImg2ImgPipeline",
        "ORTStableDiffusionInpaintPipeline",
        "ORTStableDiffusionPipeline",
        # xl
        "ORTStableDiffusionXLImg2ImgPipeline",
        "ORTStableDiffusionXLInpaintPipeline",
        "ORTStableDiffusionXLPipeline",
    ]
else:
    _import_structure["modeling_diffusion"] = [
        "ORTDiffusionPipeline",
        "ORTPipelineForText2Image",
        "ORTPipelineForImage2Image",
        "ORTPipelineForInpainting",
        # flux
        "ORTFluxPipeline",
        # lcm
        "ORTLatentConsistencyModelImg2ImgPipeline",
        "ORTLatentConsistencyModelPipeline",
        # sd3
        "ORTStableDiffusion3Img2ImgPipeline",
        "ORTStableDiffusion3InpaintPipeline",
        "ORTStableDiffusion3Pipeline",
        # sd
        "ORTStableDiffusionImg2ImgPipeline",
        "ORTStableDiffusionInpaintPipeline",
        "ORTStableDiffusionPipeline",
        # xl
        "ORTStableDiffusionXLImg2ImgPipeline",
        "ORTStableDiffusionXLInpaintPipeline",
        "ORTStableDiffusionXLPipeline",
    ]


# Direct imports for type-checking
if TYPE_CHECKING:
    from optimum.onnxruntime.configuration import ORTConfig, QuantizationConfig
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
        ONNX_WEIGHTS_NAME,
    )
    from optimum.onnxruntime.modeling import (
        ORTModel,
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
        ORTModelForZeroShotImageClassification,
    )
    from optimum.onnxruntime.modeling_decoder import ORTModelForCausalLM
    from optimum.onnxruntime.modeling_seq2seq import (
        ORTModelForPix2Struct,
        ORTModelForSeq2SeqLM,
        ORTModelForSpeechSeq2Seq,
        ORTModelForVision2Seq,
    )
    from optimum.onnxruntime.optimization import ORTOptimizer
    from optimum.onnxruntime.pipelines import pipeline
    from optimum.onnxruntime.quantization import ORTQuantizer
    from optimum.onnxruntime.utils import ORTQuantizableOperator

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()  # noqa: TRY301
    except OptionalDependencyNotAvailable:
        from optimum.onnxruntime.dummy_objects import (
            # generic entrypoint
            ORTDiffusionPipeline,
            # flux
            ORTFluxPipeline,
            # lcm
            ORTLatentConsistencyModelImg2ImgPipeline,
            ORTLatentConsistencyModelPipeline,
            # task-specific entrypoints
            ORTPipelineForImage2Image,
            ORTPipelineForInpainting,
            ORTPipelineForText2Image,
            # sd3
            ORTStableDiffusion3Img2ImgPipeline,
            ORTStableDiffusion3InpaintPipeline,
            ORTStableDiffusion3Pipeline,
            # sd
            ORTStableDiffusionImg2ImgPipeline,
            ORTStableDiffusionInpaintPipeline,
            ORTStableDiffusionPipeline,
            # xl
            ORTStableDiffusionXLImg2ImgPipeline,
            ORTStableDiffusionXLInpaintPipeline,
            ORTStableDiffusionXLPipeline,
        )
    else:
        from optimum.onnxruntime.modeling_diffusion import (
            # generic entrypoint
            ORTDiffusionPipeline,
            # flux
            ORTFluxPipeline,
            # lcm
            ORTLatentConsistencyModelImg2ImgPipeline,
            ORTLatentConsistencyModelPipeline,
            # task-specific entrypoints
            ORTPipelineForImage2Image,
            ORTPipelineForInpainting,
            ORTPipelineForText2Image,
            # sd3
            ORTStableDiffusion3Img2ImgPipeline,
            ORTStableDiffusion3InpaintPipeline,
            ORTStableDiffusion3Pipeline,
            # sd
            ORTStableDiffusionImg2ImgPipeline,
            ORTStableDiffusionInpaintPipeline,
            ORTStableDiffusionPipeline,
            # xl
            ORTStableDiffusionXLImg2ImgPipeline,
            ORTStableDiffusionXLInpaintPipeline,
            ORTStableDiffusionXLPipeline,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
