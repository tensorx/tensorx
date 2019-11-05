import unittest
import tensorx as tx
import tensorx.callbacks as tc
from functools import partial
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MyTestCase(unittest.TestCase):
    def test_gradient_sparse_var(self):
        """
        https://www.tensorflow.org/beta/guide/effective_tf2
        Returns:

        """
        target = tf.constant([[1., 0., 0.], [1., 0., 0.]])

        v = tf.Variable([0.5, 0.5])

        # TODO lambda taking 0 inputs can't call the function
        #   because that would require the function to take an empty list of inputs
        x = tx.Lambda([],
                      fn=lambda _: tf.SparseTensor([[0, 0], [1, 1]], v, [2, 3]),
                      n_units=3,
                      var_list=v)
        print(x())
        print(x.trainable_variables)
        y = tx.Linear(x, n_units=3)

        # a graph without inputs needs to have missing inputs declared
        # otherwise it will try to add the inputs detected to inputs
        graph = tx.Graph.build(inputs=None,
                               outputs=y)
        fn = graph.as_function()

        @tf.function
        def loss(labels):
            return tf.reduce_mean(tf.pow(labels - fn(), 2))

        with tf.GradientTape() as tape:
            loss_val = loss(target)

        print(tape.gradient(loss_val, v))
        # grad_vals = tape.gradient(loss_val, v.values)
        # grad_vals = tape.gradient(loss_val, v2)
        # print(loss_val)
        # print(grad_vals)

    def test_to_sparse_gradient(self):
        target = tf.constant([[1., 0.], [1., 0.]])
        x = tx.Tensor(tf.ones([1, 4], dtype=tf.float32), n_units=4)
        h = tx.Linear(x, n_units=2)
        y = tx.ToSparse(h)
        y = tx.ToDense(y)

        @tf.function
        def loss_fn(labels, pred):
            return tf.reduce_mean(tf.pow(labels - pred, 2))

        with tf.GradientTape() as tape:
            pred = y()
            loss = loss_fn(target, pred)

        gradients = tape.gradient(loss, h.trainable_variables)
        print(gradients)

        # print(y.weights)
        # print(tape.gradient(y.weights, loss_val))


if __name__ == '__main__':
    unittest.main()
