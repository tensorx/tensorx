import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
import tensorflow as tf
import tensorx as tx
from tensorx.utils import *
import numpy as np
import functools


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


def test_graph_merge():
    x = tx.Input([[1]])
    l1 = tx.Linear(x, n_units=2)
    l2 = tx.Linear(x, n_units=2)
    l3 = tx.Linear(l2, n_units=2)

    g1 = Graph.build(None, l1)
    g2 = Graph.build(None, l3)

    assert len(set.difference(set(g1.in_nodes), g2.in_nodes)) == 0
    assert len(set.difference(set(g1.out_nodes), g2.out_nodes)) != 0

    g3 = Graph.merge(g1, g2)
    assert set.intersection(set(g1.in_nodes), g3.in_nodes) == set(g1.in_nodes)
    assert set.intersection(set(g1.out_nodes), g3.out_nodes) == set(g1.out_nodes)


def test_graph_repeated():
    x = tx.Input([[1]])
    l1 = tx.Linear(x, 2, name="l1")
    l2 = tx.Linear(x, 2, name="l2")
    l3 = tx.layer(n_units=2, name="l3")(lambda a, b: tf.add(a, b))(l1, l2)

    g = Graph.build(l1, l3, missing_inputs=True)
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

    graph.append_layer(out2)
    assert out1 in graph.out_nodes
    assert out2 in graph.out_nodes


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
