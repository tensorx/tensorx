import unittest
import tensorflow as tf
import tensorx as tx
from tensorx.utils import *
import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestUtils(unittest.TestCase):

    def assertArrayEqual(self, actual, desired, verbose=True):
        if isinstance(actual, tx.Layer):
            actual = actual.tensor()
        if isinstance(desired, tx.Layer):
            desired = desired.tensor()

        self.assertTrue(np.array_equal(actual, desired))

    def assertArrayNotEqual(self, actual, desired):
        if isinstance(actual, tx.Layer):
            actual = actual.tensor()
        if isinstance(desired, tx.Layer):
            desired = desired.tensor()

        self.assertFalse(np.array_equal(actual, desired))

    def test_graph_build(self):
        x = tx.Input([[1]])
        g = Graph.build_graph(None, x)

        self.assertEqual(len(g.in_nodes), len(g.out_nodes))
        self.assertEqual(len(g.in_nodes), 1)

        l1 = tx.Linear(x, n_units=2)
        l2 = tx.Linear(x, n_units=2)
        l3 = tx.Linear(x, n_units=2)

        g1 = Graph.build_graph(None, l1)
        self.assertEqual(len(g1.in_nodes), len(g1.out_nodes))
        self.assertTrue(set.isdisjoint(g1.in_nodes, g1.out_nodes))
        self.assertIn(l1, g1.out_nodes)
        self.assertIn(x, g1.in_nodes)

        g2 = Graph.build_graph(x, l1)
        self.assertFalse(set.isdisjoint(g1.in_nodes, g2.in_nodes))
        self.assertFalse(set.isdisjoint(g1.out_nodes, g2.out_nodes))

        try:
            g = Graph.build_graph([l2, l3], l1)
            self.fail("Invalid graph should have raised an exception")
        except ValueError:
            pass

        g = Graph.build_graph(x, [l2, l3])

        self.assertEqual(len(g.edges_out[x]), 2)
        self.assertIn(l2, g.edges_out[x])
        self.assertIn(l3, g.edges_out[x])
        self.assertEqual(x, g.edges_in[l2][0])

    def test_graph_merge(self):
        x = tx.Input([[1]])

        l1 = tx.Linear(x, n_units=2)
        l2 = tx.Linear(x, n_units=2)
        l3 = tx.Linear(l2, n_units=2)

        g1 = Graph.build_graph(None, l1)
        g2 = Graph.build_graph(None, l3)

        self.assertEqual(len(set.difference(g1.in_nodes, g2.in_nodes)), 0)
        self.assertNotEqual(len(set.difference(g1.out_nodes, g2.out_nodes)), 0)

        g3 = Graph.merge(g1, g2)
        self.assertEqual(set.intersection(g1.in_nodes, g3.in_nodes), g1.in_nodes)
        self.assertEqual(set.intersection(g1.out_nodes, g3.out_nodes), g1.out_nodes)

    def test_graph_repeated(self):
        x = tx.Input([[1]])
        l1 = tx.Linear(x, 2, name="l1")
        l2 = tx.Linear(x, 2, name="l2")

        l3 = tx.layer(n_units=2, name="l3")(lambda a, b: tf.add(a, b))(l1, l2)

        g = Graph.build_graph(l1, l3)

        # for a, b in g.edges_out.items():
        #    for out in b:
        #        print("{}==>{}".format(a.name, out.name))

    def test_sp_variable(self):
        x = tx.sparse_ones([[0, 2], [1, 1], [2, 0]], dense_shape=[3, 3])
        x2 = x * 2
        x3 = tx.sparse_ones([[0, 1], [0, 2], [1, 1], [2, 0]], dense_shape=[3, 3])
        v = tx.SparseVariable(x, validate_shape=False)

        v.assign(x2)
        self.assertArrayEqual(tf.sparse.to_dense(v.value()), tf.sparse.to_dense(x2))

        v.assign(x3)
        self.assertArrayEqual(tf.sparse.to_dense(v.value()), tf.sparse.to_dense(x3))


if __name__ == '__main__':
    unittest.main()
