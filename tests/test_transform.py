import unittest

import tensorflow as tf
import numpy as np

import tensorx.transform as tx


class TestTransform(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_sparse_put(self):
        tensor = tf.SparseTensor([[0, 0], [1, 0]], [2, 0.2], [2, 2])
        sp_values = tf.SparseTensor([[0, 0], [0, 1]], [3.0, 32.0], [2, 2])

        expected = tf.constant([[3., 32.], [0.2, 0.]])

        result = tx.sparse_put(tensor, sp_values)
        result = tf.sparse_tensor_to_dense(result)

        np.testing.assert_array_equal(expected.eval(), result.eval())

    def test_dense_put(self):
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[3, 1], [1, 1]])

        sp_values = tf.SparseTensor(indices=[[0, 0]], values=[3], dense_shape=[2, 2])
        result = tx.dense_put(tensor, sp_values)

        np.testing.assert_array_equal(expected.eval(), result.eval())

        # updates are cast to the given tensor type
        # 0 as update should also work since sparse tensors have a set of indices used to
        # make the updated indices explicit
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[0, 1], [1, 1]])
        sp_values = tf.SparseTensor(indices=[[0, 0]], values=tf.constant([0],dtype=tf.float32), dense_shape=[2, 2])

        result = tx.dense_put(tensor, sp_values)
        np.testing.assert_array_equal(expected.eval(), result.eval())

    def test_enum_row(self):
        indices = tf.constant([[0,1],[1,2]],dtype=tf.int64)
        result = tx.enum_row(indices)

        ss = tf.Session()
        print(ss.run(result))
        print(ss.run(tf.shape(result)))


if __name__ == '__main__':
    unittest.main()
