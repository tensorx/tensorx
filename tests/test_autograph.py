import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '10'

import tensorflow as tf
import tensorx as tx
from tensorx.utils import Graph


def test_autograph_to_graph():
    inputs = tx.Input(tf.ones([2, 4]))

    def fn(x):
        return tf.multiply(x, 2)

    # only works in functions that expose __code__
    g = tf.autograph.to_graph(fn)
    y = g(inputs)

    assert tx.tensor_equal(y, inputs() * 2)


def test_tensor_layer():
    x = tx.Input([[2]], n_units=1, constant=False)

    fn = x.as_function()
    x.value = [[4]]
    y = fn()

    assert y.numpy().flatten() == 4


def test_build_graph():
    x1 = tx.Input(n_units=1000, constant=False, dtype=tf.float32)
    x2 = tx.Input(init_value=tf.ones([1, 3]), dtype=tf.float32, constant=True)

    y10 = tx.Linear(x1, n_units=3)
    y11 = tx.Activation(y10)
    y1 = tx.Module(x1, y11)
    y2 = tx.Add(y1, x2)
    output = y2

    graph = Graph.build(inputs=None, outputs=[y1, y2])
    # module condenses 2 nodes so it's 4 and not 6
    assert len(graph.nodes) == 4

    @tf.function
    def simple_graph(in0):
        x1.value = in0
        return y2()

    simple_graph_2 = Graph.build(inputs=[x1, x2], outputs=y2)
    simple_graph_2 = tf.function(simple_graph_2)
    g = Graph.build(inputs=[x1, x2], outputs=y2)
    y2fn = y2.as_function()
    data = tf.ones([256, 1000])
    x1.value = data

    compiled_fn = g.as_function(ord_inputs=x1,
                                ord_outputs=output)

    assert tx.tensor_equal(compiled_fn(data), y2fn())
    assert tx.tensor_equal(compiled_fn(data), simple_graph_2()[0])

    from timeit import timeit

    def update_run():
        x1.value = tf.random.uniform([256, 1000])
        return y2fn()

    n = 1000
    t_update_run = timeit(update_run, number=n)
    t_generated = timeit(lambda: compiled_fn(tf.random.uniform([256, 1000])), number=n)
    t_compile_value_set = timeit(lambda: simple_graph(tf.random.uniform([256, 1000])), number=n)
    t_graph_call_tf = timeit(lambda: simple_graph_2(tf.random.uniform([256, 1000])), number=n)

    assert t_generated < t_update_run
    assert t_generated < t_compile_value_set
    assert t_generated < t_graph_call_tf
    assert t_update_run > t_compile_value_set

    o1 = compiled_fn(tf.random.uniform([256, 1000]))
    o2 = compiled_fn(tf.random.uniform([256, 1000]))
    assert not tx.tensor_equal(o1, o2)
