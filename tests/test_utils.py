import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import tensorflow as tf
import tensorx as tx
from tensorx.utils import *
import numpy as np
import functools
from tensorx.test_utils import TestCase


class TestUtils(TestCase):

    def test_graph_build(self):
        x = tx.Input([[1]])
        g = Graph.build(None, x)

        self.assertEqual(len(g.in_nodes), len(g.out_nodes))
        self.assertEqual(len(g.in_nodes), 1)

        l1 = tx.Linear(x, n_units=2)
        l2 = tx.Linear(x, n_units=2)
        l3 = tx.Linear(x, n_units=2)

        g1 = Graph.build(None, l1)
        self.assertEqual(len(g1.in_nodes), len(g1.out_nodes))
        self.assertTrue(set.isdisjoint(set(g1.in_nodes), g1.out_nodes))
        self.assertIn(l1, g1.out_nodes)
        self.assertIn(x, g1.in_nodes)

        g2 = Graph.build(x, l1)
        self.assertFalse(set.isdisjoint(set(g1.in_nodes), g2.in_nodes))
        self.assertFalse(set.isdisjoint(set(g1.out_nodes), g2.out_nodes))

        try:
            g = Graph.build([l2, l3], l1)
            self.fail("Invalid graph should have raised an exception")
        except ValueError:
            pass

        g = Graph.build(x, [l2, l3])

        self.assertEqual(len(g.edges_out[x]), 2)
        self.assertIn(l2, g.edges_out[x])
        self.assertIn(l3, g.edges_out[x])
        self.assertEqual(x, g.edges_in[l2][0])

    def test_graph_merge(self):
        x = tx.Input([[1]])

        l1 = tx.Linear(x, n_units=2)
        l2 = tx.Linear(x, n_units=2)
        l3 = tx.Linear(l2, n_units=2)

        g1 = Graph.build(None, l1)
        g2 = Graph.build(None, l3)

        self.assertEqual(len(set.difference(set(g1.in_nodes), g2.in_nodes)), 0)
        self.assertNotEqual(len(set.difference(set(g1.out_nodes), g2.out_nodes)), 0)

        g3 = Graph.merge(g1, g2)
        self.assertEqual(set.intersection(set(g1.in_nodes), g3.in_nodes), set(g1.in_nodes))
        self.assertEqual(set.intersection(set(g1.out_nodes), g3.out_nodes), set(g1.out_nodes))

    def test_graph_repeated(self):
        x = tx.Input([[1]])
        l1 = tx.Linear(x, 2, name="l1")
        l2 = tx.Linear(x, 2, name="l2")

        l3 = tx.layer(n_units=2, name="l3")(lambda a, b: tf.add(a, b))(l1, l2)

        g = Graph.build(l1, l3, missing_inputs=True)

        self.assertEqual(set([x, l1]), set(g.in_nodes))

    def test_sp_variable(self):
        x = tx.sparse_ones([[0, 2], [1, 1], [2, 0]], dense_shape=[3, 3])
        x2 = x * 2
        x3 = tx.sparse_ones([[0, 1], [0, 2], [1, 1], [2, 0]], dense_shape=[3, 3])
        v = tx.SparseVariable(x, validate_shape=False)

        v.assign(x2)
        self.assertArrayEqual(tf.sparse.to_dense(v.value()), tf.sparse.to_dense(x2))

        v.assign(x3)
        self.assertArrayEqual(tf.sparse.to_dense(v.value()), tf.sparse.to_dense(x3))

    def test_override_out_nodes(self):
        x = tx.Input(n_units=2, name="x", constant=False)
        y = tx.Linear(x, 2, name="y")
        out1 = tx.Activation(y, tf.nn.softmax, name="out1")
        out2 = tx.Activation(out1, tf.nn.softmax, name="out2")

        graph = Graph.build(inputs=x, outputs=[out1, out2])
        self.assertIn(out1, graph.out_nodes)
        self.assertIn(out2, graph.out_nodes)

        graph = Graph.build(inputs=x, outputs=out1)
        self.assertIn(out1, graph.out_nodes)
        self.assertNotIn(out2, graph.out_nodes)

        graph.append_layer(out2)
        self.assertIn(out1, graph.out_nodes)
        self.assertIn(out2, graph.out_nodes)

    def test_dependency_iter(self):
        """ test dependency iterator after adding leafs to the graph
        """

        x1 = tx.Input(n_units=2, name="x1", constant=False)
        x2 = tx.Input(n_units=2, name="x2", constant=False)

        y1 = tx.Linear(x2, 2, name="y1")
        y2 = tx.Linear(y1, 2, name="y2")
        y3 = tx.Linear(x1, 2, name="y3")

        graph = Graph.build(inputs=[x1, x2], outputs=[y2, y3])
        dep = graph.dependency_iter()
        dep_iter = list(dep)

        self.assertTrue(sorted(dep.values()))

        self.assertIs(dep_iter[0], x1)
        self.assertIs(dep_iter[1], x2)
        self.assertIn(y1, dep_iter[-2:])
        self.assertIn(y2, dep_iter[-2:])

        # ANOTHER GRAPH

        x1 = tx.Input(n_units=1, name="x1")
        x2 = tx.Input(n_units=1, name="x2")
        x3 = tx.Input(n_units=1, name="x3")

        h = tx.Add(x1, x2, name="h")
        y = tx.Add(x3, h, name="y")

        g = Graph.build(inputs=None, outputs=y)

        priorities = g.dependency_iter()

        self.assertEqual(priorities[y], (2, 0))
        self.assertEqual(priorities[x1], (0, 1))
        self.assertGreater(priorities[y], priorities[h])


if __name__ == '__main__':
    unittest.main()
