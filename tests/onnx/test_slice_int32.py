# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from optimum.onnx import cast_slice_nodes_inputs_to_int32


def constant(name, value):
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=name.removesuffix("_output_0") if name.endswith("_output_0") else f"producer_{name}",
        value=helper.make_tensor("", TensorProto.INT64, [1], [value]),
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_shared_slice_constants(reverse, dynamic):
    nodes = [
        constant("Constant_zero_output_0", 0),
        constant("Constant_end_output_0", np.iinfo(np.int64).max),
        constant("Constant_end_output_0_slice_int32", 7),
    ]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])]
    feeds = {"x": np.arange(3, dtype=np.float32).reshape(1, 3)}
    starts = "Constant_zero_output_0"
    if dynamic:
        starts = "starts"
        inputs.append(helper.make_tensor_value_info(starts, TensorProto.INT64, [1]))
        feeds[starts] = np.array([0], dtype=np.int64)
    consumers = [
        helper.make_node("Squeeze", ["x", "Constant_zero_output_0"], ["squeezed"], name="squeeze"),
        helper.make_node(
            "Slice", ["x", starts, "Constant_end_output_0", "Constant_zero_output_0"], ["sliced"], name="slice"
        ),
        helper.make_node(
            "Slice",
            ["x", "Constant_zero_output_0", "Constant_end_output_0", "Constant_zero_output_0"],
            ["other_slice"],
            name="other_slice",
        ),
    ]
    nodes.extend(reversed(consumers) if reverse else consumers)
    outputs = [
        helper.make_tensor_value_info("squeezed", TensorProto.FLOAT, [3]),
        helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [1, 3]),
        helper.make_tensor_value_info("other_slice", TensorProto.FLOAT, [1, 3]),
        helper.make_tensor_value_info("Constant_end_output_0", TensorProto.INT64, [1]),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "shared", inputs, outputs), opset_imports=[helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model, full_check=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    assert cast_slice_nodes_inputs_to_int32(model) is model
    onnx.checker.check_model(model, full_check=True)
    actual = ReferenceEvaluator(model).run(None, feeds)
    for result, original in zip(actual, expected):
        np.testing.assert_array_equal(result, original)
    constants = {node.output[0]: node.attribute[0].t for node in model.graph.node if node.op_type == "Constant"}
    for node in model.graph.node:
        if node.op_type == "Slice":
            expected_type = TensorProto.INT64 if node.output[0] == "sliced" and dynamic else TensorProto.INT32
            assert constants[node.input[2]].data_type == expected_type
            assert onnx.numpy_helper.to_array(constants[node.input[2]])[0] == np.iinfo(np.int32).max
    # The shared graph output must retain its original value, not merely its dtype.
    assert actual[-1][0] == np.iinfo(np.int64).max
    assert onnx.numpy_helper.to_array(constants["Constant_end_output_0_slice_int32"])[0] == 7
    serialized = model.SerializeToString()
    cast_slice_nodes_inputs_to_int32(model)
    assert model.SerializeToString() == serialized


def test_slice_optional_empty_axes():
    nodes = [constant("start", 0), constant("stop", 2), constant("step", 1)]
    nodes.append(helper.make_node("Slice", ["x", "start", "stop", "", "step"], ["y"]))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "optional",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    cast_slice_nodes_inputs_to_int32(model)
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, {"x": np.arange(3, dtype=np.float32)})[0], [0, 1]
    )
    slice_node = next(node for node in model.graph.node if node.op_type == "Slice")
    constants = {node.output[0]: node.attribute[0].t for node in model.graph.node if node.op_type == "Constant"}
    assert slice_node.input[3] == ""
    assert all(constants[name].data_type == TensorProto.INT32 for name in slice_node.input[1:] if name)
    # A second pass leaves an already converted graph unchanged.
    serialized = model.SerializeToString()
    cast_slice_nodes_inputs_to_int32(model)
    assert model.SerializeToString() == serialized
