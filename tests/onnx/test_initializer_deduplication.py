# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import onnx
import pytest
from onnxruntime import InferenceSession

from optimum.onnx.graph_transformations import merge_decoders, remove_duplicate_weights


def make_model(initializers, nodes, inputs, outputs):
    graph = onnx.helper.make_graph(nodes, "initializer_deduplication", inputs, outputs, initializer=initializers)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    return model


def run_model(model, inputs):
    return InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, inputs)


def assert_outputs_equal(expected, actual):
    assert len(expected) == len(actual)
    for expected_output, actual_output in zip(expected, actual):
        assert expected_output.shape == actual_output.shape
        assert expected_output.dtype == actual_output.dtype
        np.testing.assert_array_equal(expected_output, actual_output)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("include_weights", [False, True])
@pytest.mark.parametrize(
    "constant",
    [
        np.array(2, dtype=np.float32),
        np.array(2, dtype=np.float64),
        np.array(2, dtype=np.int32),
        np.array(2, dtype=np.int64),
        np.array(True, dtype=np.bool_),
        np.array([2, 2], dtype=np.int32),
        np.array([2, 2], dtype=np.int64),
    ],
    ids=["float32_scalar", "float64_scalar", "int32_scalar", "int64_scalar", "bool_scalar", "int32_1d", "int64_1d"],
)
def test_remove_duplicate_weights_preserves_excluded_initializers(constant, include_weights, inplace):
    initializers = []
    nodes = []
    outputs = []
    # Equal constants with distinct names must both survive, including when the sharing map is empty.
    for name in ["constant", "constant_copy"]:
        initializer = onnx.numpy_helper.from_array(constant, name=name)
        initializers.append(initializer)
        nodes.append(onnx.helper.make_node("Identity", [name], [name + "_output"]))
        outputs.append(onnx.helper.make_tensor_value_info(name + "_output", initializer.data_type, constant.shape))
    inputs = []
    if include_weights:
        inputs.append(onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2]))
        for name in ["weight", "weight_copy"]:
            initializers.append(onnx.numpy_helper.from_array(np.eye(2, dtype=np.float32), name=name))
            nodes.append(onnx.helper.make_node("MatMul", ["input", name], [name + "_output"]))
            outputs.append(onnx.helper.make_tensor_value_info(name + "_output", onnx.TensorProto.FLOAT, [2, 2]))
    model = make_model(initializers, nodes, inputs, outputs)
    original_bytes = model.SerializeToString()
    feeds = {"input": np.array([[1, 2], [3, 4]], dtype=np.float32)} if include_weights else {}
    expected = run_model(model, feeds)

    optimized = remove_duplicate_weights(model, inplace=inplace)

    assert (optimized is model) == inplace
    if not inplace:
        assert model.SerializeToString() == original_bytes
    onnx.checker.check_model(optimized)
    names = {initializer.name for initializer in optimized.graph.initializer}
    assert {"constant", "constant_copy"}.issubset(names)
    assert len(names) == (3 if include_weights else 2)
    assert list(optimized.graph.node[0].input) == ["constant"]
    assert list(optimized.graph.node[1].input) == ["constant_copy"]
    if include_weights:
        assert optimized.graph.node[2].input[1] == optimized.graph.node[3].input[1]
    assert_outputs_equal(expected, run_model(optimized, feeds))
    # A second pass must also preserve all references and constants.
    second_pass = remove_duplicate_weights(optimized)
    onnx.checker.check_model(second_pass)
    assert second_pass.SerializeToString() == optimized.SerializeToString()


@pytest.mark.parametrize(
    "other, expected_count",
    [
        (np.array([[1, 2], [3, 4]], dtype=np.float32), 1),
        (np.array([[1, 2], [3, 5]], dtype=np.float32), 2),
        (np.array([1, 2, 3, 4], dtype=np.float32), 2),
        # Equal raw bytes must not cause tensors of different dtypes to share a name.
        (np.array([[1, 2], [3, 4]], dtype=np.float32).view(np.int32), 2),
    ],
    ids=["identical", "different_values", "different_shape", "different_dtype"],
)
def test_remove_duplicate_weights_compares_data_type_and_shape(other, expected_count):
    initializers = [
        onnx.numpy_helper.from_array(np.array([[1, 2], [3, 4]], dtype=np.float32), name="weight"),
        onnx.numpy_helper.from_array(other, name="other_weight"),
    ]
    model = make_model(
        initializers,
        [
            onnx.helper.make_node("Identity", [initializer.name], [initializer.name + "_output"])
            for initializer in initializers
        ],
        [],
        [
            onnx.helper.make_tensor_value_info(initializer.name + "_output", initializer.data_type, initializer.dims)
            for initializer in initializers
        ],
    )
    expected = run_model(model, {})
    optimized = remove_duplicate_weights(model)
    onnx.checker.check_model(optimized)
    assert len(optimized.graph.initializer) == expected_count
    assert_outputs_equal(expected, run_model(optimized, {}))


def make_decoder(weight, weight_name, shift, indices):
    initializers = [
        onnx.numpy_helper.from_array(weight, name=weight_name),
        onnx.numpy_helper.from_array(np.array(shift, dtype=np.float32), name="shift"),
        onnx.numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="shape"),
        onnx.numpy_helper.from_array(np.array(indices, dtype=np.int32), name="indices"),
    ]
    return make_model(
        initializers,
        [
            onnx.helper.make_node("Cast", [weight_name], ["cast_weight"], to=onnx.TensorProto.FLOAT),
            onnx.helper.make_node("Reshape", ["cast_weight", "shape"], ["reshaped_weight"]),
            onnx.helper.make_node("MatMul", ["input", "reshaped_weight"], ["product"]),
            onnx.helper.make_node("Add", ["product", "shift"], ["output"]),
            onnx.helper.make_node("Identity", ["indices"], ["indices_output"]),
        ],
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])],
        [
            onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2]),
            onnx.helper.make_tensor_value_info("indices_output", onnx.TensorProto.INT32, [2]),
        ],
    )


@pytest.mark.parametrize(
    "mode", ["identical", "different_name", "different_values", "different_shape", "different_dtype"]
)
def test_remove_duplicate_weights_then_merge_decoders_preserves_local_constants(mode, tmp_path):
    weight = np.array([[1, 2], [3, 4]], dtype=np.float32)
    other_weight = weight.copy()
    if mode == "different_values":
        other_weight[0, 0] = 5
    elif mode == "different_shape":
        other_weight = other_weight.reshape(4)
    elif mode == "different_dtype":
        other_weight = other_weight.view(np.int32)
    decoder = make_decoder(weight, "weight", 1, [0, 1])
    decoder_with_past = make_decoder(other_weight, "other_weight" if mode == "different_name" else "weight", 2, [1, 0])
    feeds = {"input": np.array([[1, 2], [3, 4]], dtype=np.float32)}
    expected = [run_model(decoder, feeds), run_model(decoder_with_past, feeds)]
    merged = merge_decoders(
        remove_duplicate_weights(decoder),
        remove_duplicate_weights(decoder_with_past),
        save_path=tmp_path / "merged.onnx",
    )
    onnx.checker.check_model(merged)
    assert len(merged.graph.initializer) == (1 if mode in ["identical", "different_name"] else 2)
    for attribute in merged.graph.node[0].attribute:
        assert {initializer.name for initializer in attribute.g.initializer} == {"shift", "shape", "indices"}
    for use_cache, branch_outputs in zip([False, True], expected):
        assert_outputs_equal(
            branch_outputs, run_model(merged, {**feeds, "use_cache_branch": np.array([use_cache], dtype=np.bool_)})
        )
