import numpy as np
import onnx

from optimum.onnx.graph_transformations import remove_duplicate_weights


def _model(initializers, nodes, outputs=None):
    if outputs is None:
        outputs = [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)]
    graph = onnx.helper.make_graph(
        nodes,
        "graph",
        [],
        outputs,
        initializer=initializers,
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    return model


def test_scalar_initializer_is_kept():
    value = np.array(1, dtype=np.float32)
    constant = onnx.numpy_helper.from_array(value, name="constant")
    model = _model(
        [constant],
        [onnx.helper.make_node("Identity", ["constant"], ["output"])],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [])],
    )
    updated = remove_duplicate_weights(model)
    assert [item.name for item in updated.graph.initializer] == ["constant"]
    assert updated.graph.node[0].input[0] == "constant"


def test_int64_vector_is_kept_and_duplicate_floats_are_shared():
    weight = np.arange(4, dtype=np.float32).reshape(2, 2)
    shared_a = onnx.numpy_helper.from_array(weight, name="weight_a")
    shared_b = onnx.numpy_helper.from_array(weight.copy(), name="weight_b")
    indices = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="indices")
    model = _model(
        [shared_a, indices, shared_b],
        [
            onnx.helper.make_node("Identity", ["weight_a"], ["hidden"]),
            onnx.helper.make_node("Identity", ["weight_b"], ["output"]),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("hidden", onnx.TensorProto.FLOAT, [2, 2]),
            onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2]),
        ],
    )
    updated = remove_duplicate_weights(model, inplace=True)
    names = [item.name for item in updated.graph.initializer]
    assert "indices" in names
    assert len(names) == 2
    kept = next(name for name in names if name != "indices")
    assert updated.graph.node[0].input[0] == kept
    assert updated.graph.node[1].input[0] == kept
