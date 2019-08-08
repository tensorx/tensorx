import unittest
import tensorflow as tf
import tensorx as tx
from datetime import datetime
from tensorx.utils import Graph

from tensorflow.python.ops import summary_ops_v2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np


class TestAutoGraph(unittest.TestCase):
    def test_autograph_to_graph(self):
        x = tx.TensorLayer(tf.ones([2, 4]))

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

        x = tx.TensorLayer([[1, 1]])
        y1 = tx.Linear(x, 2, name="y1")
        y2 = tx.Linear(y1, 3, name="y2")
        v2 = tx.VariableLayer(y2)
        y3 = y2.reuse_with(x, name='y3')
        y4 = tx.layer(3, 'add')(lambda x1, x2: tf.add(x1, x2))(v2, y3)

        # z = tf.function(y4.compute)
        z = y4.compile_graph()

        tf.summary.trace_on(graph=True, profiler=True)
        z()

        with writer.as_default():
            tf.summary.trace_export(
                name="layers",
                step=0,
                profiler_outdir=logdir)

        writer.flush()

    def test_tensor_layer(self):
        x = tx.TensorLayer([[2]], n_units=1, constant=False)

        logdir = "/home/davex32/tmp/logs/test"  # + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = summary_ops_v2.create_file_writer(logdir)

        fn = x.compile_graph()
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
                # the problem with this is that it doesn't show the graph in tensorboard
                # the alternative is to assign the value on setter
                # return tf.py_function(lambda: tf.cast(self.value, tf.float32), inp=[], Tout=tf.float32)

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
        print(g2())
        x.value = [[3]]
        print(g2())
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
        x1 = tx.TensorLayer(n_units=1000, constant=False, dtype=tf.float32)
        x2 = tx.TensorLayer(value=tf.ones([1, 3]), dtype=tf.float32)

        y1 = tx.Linear(x1, n_units=3)
        y2 = tx.Add(y1, x2)

        g = Graph.build_graph(None, output_layers=y2)

        var_inputs = list(filter(lambda x: not x.constant, g.in_nodes))
        var_input_map = {in_layer: f"in_{i}"
                         for i, in_layer in
                         enumerate(var_inputs)
                         }
        input_args = ", ".join(var_input_map.values())

        fn_def = f"def graph({input_args}):\n"  # + "\tprint(globals())\n"

        fn_compute = []

        cons_inputs = list(filter(lambda x: x.constant, g.in_nodes))

        nv = len(var_input_map)
        nc = len(cons_inputs)

        cons_input_map = {in_layer: f"in_{i}" for
                          i, in_layer in
                          zip(range(nv, nv + nc), cons_inputs)}

        # compute constant tensors
        # requires outer access to layers var
        for cons_layer in cons_inputs:
            name = cons_input_map[cons_layer]
            fn_compute.append(f"\t{name} = layers[\"{name}\"].compute()")

        # TODO this exposes the API outside the compute function
        # I should convert to layers inside compute with a util fn as_layer
        # node_map = dict({in_node: "tx.TensorLayer()"})
        node_map = dict(var_input_map)
        node_map.update(cons_input_map)

        # for each layer not in inputs
        node_i = nv + nc
        visited = set(cons_inputs + var_inputs)
        prev_nodes = visited
        print(visited.difference(g.out_nodes))
        while len(visited) < len(g.nodes) - len(g.out_nodes):
            # advance in the graph
            next_nodes = [node for in_node in prev_nodes for node in g.edges_out[in_node]]
            # for each next node that is not an output node
            for node in next_nodes:
                if node not in g.out_nodes:
                    name = f"layer_{node_i}"
                    node_map[node] = name
                    node_i += 1

                    # TODO basically we can't feed values to compute because compute acts on layers
                    # TODO not results, what about layers that need access to state methods, e.g. Tensor
                    # Layers
                    # TODO should I make computations on tensors directly ? extracting tensors from inputs
                    # for in_node in g.edges_in[node]:
                    #     if in_node in var_input_map:
                    #         # TODO this or convert inside compute
                    #         name = node_map[in_node]
                    #         node_map[in_node] = f"tx.TensorLayer({name})"
                    #     else:

                    in_args = ", ".join([node_map[in_node] for in_node in g.edges_in[node]])
                    fn_compute.append(f"\t{name} = layers[\"{name}\"].compute({in_args})")
                    visited.add(node)

            prev_nodes = next_nodes

        # TODO sets are not ordered but we could do this based on an ordered list of output nodes
        for i, node in enumerate(g.out_nodes):
            name = f"out_{i}"
            node_map[node] = name
            in_args = ", ".join([node_map[in_node] for in_node in g.edges_in[node]])
            fn_compute.append(f"\t{name} = layers[\"{name}\"].compute({in_args})")

        outputs = ", ".join([node_map[node] for node in g.out_nodes])
        fn_compute.append(f"\treturn {outputs}\n")

        fn_compute = "\n".join(fn_compute)
        import pprint
        # reverse nodemap
        # node_map = {v: k for k, v in node_map.items()}
        # node_map = {v: str(k) for k, v in node_map.items()}

        print(fn_def + fn_compute)

        import inspect

        # print(eval("graph",globals(),locals()))
        # reverse node_map

        # global layers
        layers = {v: k for k, v in node_map.items()}
        pprint.pprint(layers)

        exec(fn_def + fn_compute, locals())
        fn = eval("graph")
        print(fn)

        out = tf.function(fn)

        y2 = y2.compile_graph()

        data = tf.ones([256, 1000])

        x1.value = data

        # print()
        # print(y2())

        np.testing.assert_array_equal(out(data), y2())

        from timeit import timeit
        def update_run():
            x1.value = tf.random.uniform([256, 1000])
            return y2()

        t1 = timeit(lambda: out(tf.random.uniform([256, 1000])), number=1000)
        t2 = timeit(update_run, number=1000)
        print(t1)
        print(t2)

        # TODO problems here
        # o1 = y2()
        # x1.value = np.random.uniform(size=[256, 1000])
        # o2 = y2()
        # self.assertFalse(np.array_equal(o1, o2))
        # print(graph)

        o1 = out(tf.random.uniform([256, 1000]))
        o2 = out(tf.random.uniform([256, 1000]))
        self.assertFalse(np.array_equal(o1, o2))

        # source = compile(fn_def + fn_compute,mode="eval")

        # print(source)
        # print(locals()["graph"](2))
        # {"graph": locals()["graph"], "layers": layers}
        # print(
        #    eval("""graph(2)"""))

        #TODO to analyse the code we can use the dis module
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
        print(g(2))
        # print(eval("fn"))
        # print(eval("fn(2)"))

    def test_model(self):
        import numpy as np

        x = tx.TensorLayer(n_units=200, constant=False)
        x2 = tx.TensorLayer(n_units=200, constant=False)
        h = tx.Linear(x, n_units=100)
        h = tx.Activation(h, tf.nn.relu)
        y = tx.Linear(h, n_units=6)
        y = tx.Activation(y, tf.nn.softmax)

        m = tx.Module(inputs=None, output=y)
        m = m.reuse_with(x2)

        x.value = np.ones([4, 200])
        print(y())
        print(m())
        x2.value = np.ones([4, 200])
        print(m())

        # g = Graph.build_graph(None, output_layers=y)

        # print(str(g.edges_in[y][0]))
