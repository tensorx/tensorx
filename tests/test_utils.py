import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
import tensorflow as tf
import tensorx as tx
from tensorx.utils import *
import numpy as np
import functools


def test_graph_as_function():
    data = [[1., 2.]]

    in1 = tx.Input(n_units=1, name="in1", dtype=tf.float32, constant=False)
    in2 = tx.Input(n_units=1, name="in1", dtype=tf.float32, constant=False)
    in3 = tx.Constant(tf.ones(shape=[1], dtype=tf.float32))
    in12 = tx.Add(in1, in2, in3)
    graph = tx.Graph.build(inputs=[in1, in2, in3], outputs=in12)

    fn = graph.as_function_v2(ord_inputs=[in1, in2, in3], stateful_inputs=True, compile=False, fn_name="add")
    # fn = graph.as_function_v2(stateful_inputs=True, compile=False)

    # TODO I should make sure the function converts the inputs to tensors
    # to make sure I don't pass lists around
    assert fn(np.array([[1.]], dtype=np.float), np.array([[1.]], dtype=np.float)) == [[3]]
    assert fn() == [[3]]
    assert fn([[1.]], [[2.]]) == [[4]]
    assert fn() == [[4]]
    assert fn([[2.]]) == [[5]]


def test_graph_input_order():
    in1 = tx.Input(n_units=1, name="in1", dtype=tf.float32, constant=False)
    in2 = tx.Input(n_units=1, name="in2", dtype=tf.float32, constant=False)
    in12 = tx.Add(in1, in2)
    in3 = tx.Constant(tf.ones(shape=[1], dtype=tf.float32))
    in123 = tx.Add(in12, in3)
    graph = tx.Graph.build(inputs=None, outputs=in123)

    # print("\n")
    # for layer,p in graph.dependency_iter().items():
    #     print(layer.name)
    #     print(p)
    print(list(map(lambda x: x.name, graph.in_nodes)))


def test_linear_graph_module_integration(tmp_path):
    tmp_path = tmp_path.joinpath("linear")
    save_path = str(tmp_path)

    x = tx.Input(init_value=tf.ones([2, 2], dtype=tf.float32))
    # x = tx.Constant(tf.constant([[32.]]), n_units=1)
    x = tx.Linear(x, n_units=x.n_units)
    linear = tx.Linear(x, n_units=4)
    graph = tx.Graph.build(inputs=None, outputs=linear)
    module = tx.Module(inputs=None, output=linear)

    assert len(module.inputs) == 1
    assert module.inputs == list(graph.in_nodes)
    assert len(graph.in_nodes) == 1

    tf.saved_model.save(module, save_path)
    module_loaded = tf.saved_model.load(save_path)
    assert tx.tensor_equal(module_loaded(), module())

    tf.saved_model.save(linear, save_path)
    linear_loaded = tf.saved_model.load(save_path)
    assert tx.tensor_equal(module_loaded(), linear_loaded())


def test_layer_graph():
    data = [[1., 2.]]

    in1 = tx.Input(n_units=2, name="in1", constant=False)
    in2 = tx.Input(n_units=2, name="in2", constant=False)
    linear = tx.Linear(in1, 1, add_bias=False)
    graph = tx.Graph.build(inputs=in1, outputs=linear)

    assert in1 in graph.in_nodes

    with pytest.raises(ValueError):
        tx.Graph.build(inputs=[in1, in2], outputs=linear)
        pytest.fail("Expected ValueError: some inputs are not connected to anything")

    with pytest.raises(ValueError):
        tx.Graph.build(inputs=[in2], outputs=linear)
        pytest.fail("Expected ValueError: inputs specified but dependencies are missing")

    w = tf.matmul(data, linear.weights)

    in1.value = data
    r1 = linear()
    r2 = graph(data)

    assert tx.tensor_equal(r2[0], w)
    assert tx.tensor_equal(r1, w)


def test_graph_no_inputs():
    in1 = tx.Input(n_units=2, constant=False)
    lin1 = tx.Linear(in1, n_units=4)

    graph = tx.Graph.build(inputs=None, outputs=lin1)
    assert len(graph.nodes) == 2
    assert in1 in graph.nodes
    assert lin1 in graph.nodes

    graph = tx.Graph.build(inputs=None, outputs=lin1, add_missing_inputs=True)


def test_multi_output_graph():
    data1 = [[1., 1.]]
    data2 = [[2., 1.]]

    in1 = tx.Input(data1, 2, name="in1", constant=False)
    in2 = tx.Input(data2, 2, name="in2")

    linear1 = tx.Linear(in1, 1)
    linear2 = tx.Linear(tx.Add(in1, in2), 1)

    graph = tx.Graph.build(inputs=None, outputs=[linear1, linear2])

    result1 = graph()
    assert len(result1) == 2

    graph2 = tx.Graph.build(inputs=None, outputs=[linear2])
    result2 = graph2()
    assert len(result2) == 1
    assert tx.tensor_equal(result2[0], result1[-1])


def test_graph_build():
    x = tx.Input([[1]])
    g = Graph.build(None, x)

    assert len(g.in_nodes) == len(g.out_nodes)
    assert len(g.in_nodes) == 1

    l1 = tx.Linear(x, n_units=2)
    l2 = tx.Linear(x, n_units=2)
    l3 = tx.Linear(x, n_units=2)

    g1 = Graph.build(None, l1)

    assert len(g1.in_nodes) == len(g1.out_nodes)
    assert set.isdisjoint(set(g1.in_nodes), g1.out_nodes)
    assert l1 in g1.out_nodes
    assert x in g1.in_nodes

    g2 = Graph.build(x, l1)

    assert not set.isdisjoint(set(g1.in_nodes), g2.in_nodes)
    assert not set.isdisjoint(set(g1.out_nodes), g2.out_nodes)

    with pytest.raises(ValueError):
        Graph.build([l2, l3], l1)
        pytest.fail("ValueError Expected: invalid graph")

    g = Graph.build(x, [l2, l3])

    assert len(g.edges_out[x]) == 2
    assert l2 in g.edges_out[x]
    assert l3 in g.edges_out[x]
    assert x == g.edges_in[l2][0]


def test_graph_draw(tmpdir):
    x = tx.Input([[1]])
    x2 = tx.Input([[1]])
    l1 = tx.Linear(x, 2, name="l1")
    l2 = tx.Linear(x, 2, name="l2")
    l3 = tx.layer(n_units=2, name="l3")(lambda a, b: tf.add(a, b))(l1, l2)
    l4 = l2.reuse_with(x2)

    graph = Graph.build(inputs=[x, x2], outputs=[l3, l4])
    str_path = str(tmpdir.join("test.pdf"))
    graph.draw(path=str_path)

    assert os.path.exists(str_path)
    # import webbrowser
    # webbrowser.open(str_path)


def test_graph_repeated():
    x = tx.Input([[1]])
    l1 = tx.Linear(x, 2, name="l1")
    l2 = tx.Linear(x, 2, name="l2")
    l3 = tx.layer(n_units=2, name="l3")(lambda a, b: tf.add(a, b))(l1, l2)

    g = Graph.build(l1, l3, add_missing_inputs=True)
    assert set([x, l1]) == set(g.in_nodes)


def test_sp_variable():
    x = tx.sparse_ones([[0, 2], [1, 1], [2, 0]], dense_shape=[3, 3])
    x2 = x * 2
    x3 = tx.sparse_ones([[0, 1], [0, 2], [1, 1], [2, 0]], dense_shape=[3, 3])
    v = tx.SparseVariable(x, validate_shape=False)

    v.assign(x2)
    assert tx.tensor_equal(tf.sparse.to_dense(v.value()), tf.sparse.to_dense(x2))

    v.assign(x3)
    assert tx.tensor_equal(tf.sparse.to_dense(v.value()), tf.sparse.to_dense(x3))


def test_override_out_nodes():
    x = tx.Input(n_units=2, name="x", constant=False)
    y = tx.Linear(x, 2, name="y")
    out1 = tx.Activation(y, tf.nn.softmax, name="out1")
    out2 = tx.Activation(out1, tf.nn.softmax, name="out2")

    graph = Graph.build(inputs=x, outputs=[out1, out2])
    assert out1 in graph.out_nodes
    assert out2 in graph.out_nodes

    graph = Graph.build(inputs=x, outputs=out1)
    assert out1 in graph.out_nodes
    assert out2 not in graph.out_nodes


def test_dependency_iter():
    """ Dependency iterator after adding leaves to the graph
    """

    x1 = tx.Input(n_units=2, name="x1", constant=False)
    x2 = tx.Input(n_units=2, name="x2", constant=False)

    y1 = tx.Linear(x2, 2, name="y1")
    y2 = tx.Linear(y1, 2, name="y2")
    y3 = tx.Linear(x1, 2, name="y3")

    graph = Graph.build(inputs=[x1, x2], outputs=[y2, y3])
    dep = graph.dependency_iter()
    dep_iter = list(dep)

    assert sorted(dep.values())

    assert dep_iter[0] is x1
    assert dep_iter[1] is x2
    assert y1 in dep_iter[-2:]
    assert y2 in dep_iter[-2:]

    # ANOTHER GRAPH
    x1 = tx.Input(n_units=1, name="x1")
    x2 = tx.Input(n_units=1, name="x2")
    x3 = tx.Input(n_units=1, name="x3")

    h = tx.Add(x1, x2, name="h")
    y = tx.Add(x3, h, name="y")

    g = Graph.build(inputs=None, outputs=y)

    priorities = g.dependency_iter()

    assert priorities[y] == (2, 0)
    assert priorities[x1] == (0, 1)
    assert priorities[y] > priorities[h]
