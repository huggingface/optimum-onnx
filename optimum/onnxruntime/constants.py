#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model.onnx"
ONNX_DECODER_MERGED_NAME = "decoder_model_merged.onnx"

ENCODER_ONNX_FILE_PATTERN = r"(.*)?encoder(.*)?\.onnx"
DECODER_ONNX_FILE_PATTERN = r"(.*)?decoder((?!(with_past|merged)).)*?\.onnx"
DECODER_WITH_PAST_ONNX_FILE_PATTERN = r"(.*)?decoder(.*)?with_past(.*)?\.onnx"
DECODER_MERGED_ONNX_FILE_PATTERN = r"(.*)?decoder(.*)?merged(.*)?\.onnx"
ONNX_FILE_PATTERN = r".*\.onnx$"
