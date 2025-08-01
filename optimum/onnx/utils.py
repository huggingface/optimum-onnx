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
from __future__ import annotations

from pathlib import Path

import onnx
from onnx.external_data_helper import ExternalDataInfo, _get_initializer_tensors, uses_external_data


def _get_onnx_external_constants(model: onnx.ModelProto) -> list[str]:
    external_constants = []

    for node in model.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                external_datas = attribute.t.external_data
                for external_data in external_datas:
                    external_constants.append(external_data.value)

    return external_constants


def _get_onnx_external_data_tensors(model: onnx.ModelProto) -> list[str]:
    """Gets the paths of the external data tensors in the model.
    Note: make sure you load the model with load_external_data=False.
    """
    model_tensors = _get_initializer_tensors(model)
    model_tensors_ext = [ExternalDataInfo(tensor).location for tensor in model_tensors if uses_external_data(tensor)]
    return model_tensors_ext


def _get_external_data_paths(src_paths: list[Path], dst_paths: list[Path]) -> tuple[list[Path], list[str]]:
    """Gets external data paths from the model and add them to the list of files to copy."""
    model_paths = src_paths.copy()
    for idx, model_path in enumerate(model_paths):
        model = onnx.load(str(model_path), load_external_data=False)
        model_tensors = _get_initializer_tensors(model)
        # filter out tensors that are not external data
        model_tensors_ext = [
            ExternalDataInfo(tensor).location
            for tensor in model_tensors
            if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
        ]
        if len(set(model_tensors_ext)) == 1:
            # if external data was saved in a single file
            src_paths.append(model_path.parent / model_tensors_ext[0])
            dst_paths.append(dst_paths[idx].parent / model_tensors_ext[0])
        else:
            # if external data doesnt exist or was saved in multiple files
            src_paths.extend([model_path.parent / tensor_name for tensor_name in model_tensors_ext])
            dst_paths.extend(dst_paths[idx].parent / tensor_name for tensor_name in model_tensors_ext)
    return src_paths, dst_paths


def _get_model_external_data_paths(model_path: Path) -> list[Path]:
    """Gets external data paths from the model."""
    onnx_model = onnx.load(str(model_path), load_external_data=False)
    model_tensors = _get_initializer_tensors(onnx_model)
    # filter out tensors that are not external data
    model_tensors_ext = [
        ExternalDataInfo(tensor).location
        for tensor in model_tensors
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    return list({model_path.parent / tensor_name for tensor_name in model_tensors_ext})


def check_model_uses_external_data(model: onnx.ModelProto | Path | str) -> bool:
    """Checks if the model uses external data."""
    if isinstance(model, (str, Path)):
        model = Path(model).as_posix()
        model = onnx.load(model, load_external_data=False)

    initializer_tensors = _get_initializer_tensors(model)
    return any(uses_external_data(tensor) for tensor in initializer_tensors)


def has_onnx_input(model: onnx.ModelProto | Path | str, input_name: str) -> bool:
    """Checks if the model has a specific input."""
    if isinstance(model, (str, Path)):
        model = Path(model).as_posix()
        model = onnx.load(model, load_external_data=False)

    return any(input.name == input_name for input in model.graph.input)
