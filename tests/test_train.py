import unittest
import tensorflow as tf
import tensorx as tx
import numpy as np

from tensorx.train import LayerGraph

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestTrain(unittest.TestCase):

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

    def test_layer_graph(self):
        data = [[1., 2.]]

        in1 = tx.TensorLayer(n_units=2, name="in1", constant=False)
        in2 = tx.TensorLayer(n_units=2, name="in2", constant=False)
        linear = tx.Linear(in1, 1)
        graph = LayerGraph(linear)

        try:
            LayerGraph(inputs=[in1, in2], outputs=linear)
            self.fail("should have raised an exception: some inputs are not connected to anything")
        except ValueError:
            pass

        try:
            LayerGraph(inputs=[in2], outputs=linear)
            self.fail("should have raised an error: inputs specified but dependencies are missing")
        except ValueError:
            pass

        w = tf.matmul(data, linear.weights)
        result2 = graph.eval()
        result1 = graph.eval(feed={in1: data})

        other_fetches = tx.TensorLayer(tf.constant([[0]]), dtype=tf.int32)
        result3 = graph.eval(feed={in1: data},
                             other_tensors=other_fetches,
                             )

        self.assertTrue(len(result3), 2)
        self.assertEqual(result3[-1], [[0]])
        self.assertArrayEqual(result3[0], w)
        self.assertArrayEqual(result1, w)
        self.assertArrayNotEqual(result2, w)

    def test_multi_output_graph(self):
        data1 = [[1, 1]]
        data2 = [[2, 1]]

        in1 = tx.TensorLayer(data1, 2, name="in1", constant=False)
        in2 = tx.TensorLayer(data2, 2, name="in2")

        linear1 = tx.Linear(in1, 1)
        linear2 = tx.Linear(tx.Add(in1, in2), 1)

        graph = LayerGraph(outputs=[linear1, linear2])

        result1 = graph.eval()
        self.assertEqual(len(result1), 2)

        result2 = graph.eval(target_outputs=linear2)
        self.assertArrayEqual(result2, result1[-1])
        result4 = graph.eval(feed={in1: data2}, target_outputs=linear2)

        # not the same because we use data2
        self.assertArrayNotEqual(result4, result2)
