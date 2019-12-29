import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '10'

import unittest
import tensorflow as tf
import tensorx as tx
from datetime import datetime
from tensorx.utils import Graph

from tensorflow.python.ops import summary_ops_v2

import numpy as np


class TestAutoGraph(unittest.TestCase):
    def test_autograph_to_graph(self):
        x = tx.Input(tf.ones([2, 4]))

        def fn(x):
            return tf.tile(x, [1, 2])

        # this generates a function that is compiled into a graph
        g = tf.autograph.to_graph(fn)
        # this generates the autograph code
        # c = tf.autograph.to_code(fn)

        y = g(x)

        # print(y)

    def test_tensorboard(self):
        logdir = "/home/davex32/tmp/logs/test"  # + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = summary_ops_v2.create_file_writer(logdir)

        # Call only one tf.function when tracing.

        x = tx.Input([[1, 1]], dtype=tf.float32)
        y1 = tx.Linear(x, 2, name="y1")
        y2 = tx.Linear(y1, 3, name="y2")
        v2 = tx.VariableLayer(y2)
        y3 = y2.reuse_with(x, name='y3')
        y4 = tx.layer(3, 'add')(lambda x1, x2: tf.add(x1, x2))(v2, y3)

        # z = tf.function(y4.compute)
        z = y4.as_function()

        tf.summary.trace_on(graph=True, profiler=True)
        z()

        with writer.as_default():
            tf.summary.trace_export(
                name="layers",
                step=0,
                profiler_outdir=logdir)

        writer.flush()

    def test_tensor_layer(self):
        x = tx.Input([[2]], n_units=1, constant=False)

        logdir = "/home/davex32/tmp/logs/test"  # + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = summary_ops_v2.create_file_writer(logdir)

        fn = x.as_function()
        x.value = [[4]]

        tf.summary.trace_on(graph=True, profiler=True)
        y = fn()

        self.assertEqual(y.numpy().flatten(), 4)

        with writer.as_default():
            tf.summary.trace_export(
                name="default",
                step=0,
                profiler_outdir=logdir)

        writer.flush()

    def test_default_values(self):
        class Default:
            def __init__(self, value):
                self._value = tf.convert_to_tensor(value)
                self.var = tf.Variable(initial_value=value,
                                       trainable=False,
                                       validate_shape=False,
                                       shape=tf.TensorShape([None, None]))

            @property
            def value(self):
                # return self._value
                return self.compute()

            @value.setter
            def value(self, value):
                self.var.assign(value)
                # self._value = value

            def compute(self):
                return self.var.value()

        x = Default([[5]])

        # x = tx.TensorLayer(value=[[5]], constant=False)

        logdir = "/home/davex32/tmp/logs/test"  # + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = summary_ops_v2.create_file_writer(logdir)

        self.assertEqual(x.compute().numpy().flatten(), 5)
        x.value = tf.convert_to_tensor([[2]])

        g = tf.function(x.compute)

        @tf.function
        def test():
            return tf.multiply(g(), 2)

        g2 = test

        # g2 = test
        tf.summary.trace_on(graph=True, profiler=True)
        x.value = [[3]]
        # self.assertEqual(fn().numpy(), 10)
        #  x.value = 4
        #  self.assertEqual(fn().numpy(), 8)
        # self.assertEqual(fn(3).numpy(), 6)

        with writer.as_default():
            tf.summary.trace_export(
                name="default",
                step=0,
                profiler_outdir=logdir)

        writer.flush()

        # annotated functions become a special callable
        # we can get the python code as follows
        # x.compute.python_function

        # print(tf.autograph.to_code(x.compute.python_function))

    def test_build_graph(self):
        x1 = tx.Input(n_units=1000, constant=False, dtype=tf.float32)
        x2 = tx.Input(init_value=tf.ones([1, 3]), dtype=tf.float32, constant=True)

        y10 = tx.Linear(x1, n_units=3)
        y11 = tx.Activation(y10)
        y1 = tx.Module(x1, y11)
        y2 = tx.Add(y1, x2)
        output = y2

        test_graph = Graph.build(inputs=None, outputs=[y1, y2])

        # print(test_graph.out_nodes)
        # print(test_graph.in_nodes)

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

        # compiled_recursive = g.compile_recursive(x1)
        # print(compiled_recursive)
        # print(compiled_recursive(tf.random.uniform([256, 1000])))

        np.testing.assert_array_equal(compiled_fn(data), y2fn())
        np.testing.assert_array_equal(compiled_fn(data), simple_graph_2()[0])

        from timeit import timeit
        def update_run():
            x1.value = tf.random.uniform([256, 1000])
            return y2fn()

        n = 1000
        t1 = timeit(update_run, number=n)
        t2 = timeit(lambda: compiled_fn(tf.random.uniform([256, 1000])), number=n)
        t3 = timeit(lambda: simple_graph(tf.random.uniform([256, 1000])), number=n)
        t4 = timeit(lambda: simple_graph_2(tf.random.uniform([256, 1000])), number=n)
        # t5 = timeit(lambda: compiled_recursive(tf.random.uniform([256, 1000])), number=n)

        print(f"{t1}\tupdate input and run")
        print(f"{t2}\tgenerated function")
        print(f"{t3}\tcompile value change and graph call")
        print(f"{t4}\tgraph call with autograph")
        # TODO I'm almost sure this slow down is due to reference to outside collections
        # print(f"{t5}\trecursive autograph")
        # TODO the problem with simple graph is that if we want to create a graph for
        #   inputs that start at the middle of a neural network (e.g. a module), this
        #   would not work unless we created the inputs first, but we would loose access to the
        #   variable slots in the process

        # TODO problems here
        # o1 = y2()
        # x1.value = np.random.uniform(size=[256, 1000])
        # o2 = y2()
        # self.assertFalse(np.array_equal(o1, o2))
        # print(graph)

        o1 = compiled_fn(tf.random.uniform([256, 1000]))
        o2 = compiled_fn(tf.random.uniform([256, 1000]))
        self.assertFalse(np.array_equal(o1, o2))

        # source = compile(fn_def + fn_compute,mode="eval")

        # print(source)
        # print(locals()["graph"](2))
        # {"graph": locals()["graph"], "layers": layers}
        # print(
        #    eval("""graph(2)"""))

        # TODO to analyse the code we can use the dis module
        # dis.dis(some function)
        #

    def test_autograph_dynamic_code(self):
        # the string bellow detects a line break as a line break and a tab as a tab just as
        # typical python code
        a = '''def fn(arg):
                res = tf.multiply(arg,2)
                print(arg)
                return res
            '''

        exec(a, globals(), globals())

        g = tf.function(eval("fn"))
        # print(g(2))
        # print(eval("fn"))
        # print(eval("fn(2)"))

    def test_model(self):
        import numpy as np

        x = tx.Input(n_units=200, constant=False)
        h = tx.Linear(x, n_units=100)
        h = tx.Activation(h, tf.nn.relu)
        y = tx.Linear(h, n_units=6)
        y = tx.Activation(y, tf.nn.softmax)

        m = tx.Module(inputs=x, output=y)
        # print(list(m.graph.in_nodes))
        # print(m.inputs)
        # print(list(m.graph.out_nodes))

        x2 = tx.Input(n_units=200, constant=False)
        m = m.reuse_with(x2)

        x.value = np.ones([4, 200])
        # print(y())
        # print(m())
        x2.value = np.ones([4, 200])
        # print(m())

        # g = Graph.build_graph(None, output_layers=y)

        # print(str(g.edges_in[y][0]))
