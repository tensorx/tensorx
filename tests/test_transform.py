"""Tests for tensor transformations module"""
import unittest

import tensorflow as tf
import numpy as np

import tensorx.transform as transform


class TestTransform(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_empty_sparse_tensor(self):
        dense_shape = [2, 2]
        empty = transform.empty_sparse_tensor(dense_shape)
        dense_empty = tf.sparse_tensor_to_dense(empty)
        zeros = tf.zeros(dense_shape)
        np.testing.assert_array_equal(dense_empty.eval(), np.zeros(dense_shape))
        np.testing.assert_array_equal(zeros.eval(), dense_empty.eval())

        dense_shape = [4]
        empty = transform.empty_sparse_tensor(dense_shape)
        dense_empty = tf.sparse_tensor_to_dense(empty)
        zeros = tf.zeros(dense_shape)
        np.testing.assert_array_equal(dense_empty.eval(), np.zeros(dense_shape))
        self.assertTrue(tf.reduce_all(tf.equal(zeros, dense_empty)).eval())

    def test_pairs(self):
        tensor1 = [[0], [1]]
        tensor2 = [1, 2]
        expected = [[0, 1], [1, 1], [0, 2], [1, 2]]

        result = transform.pairs(tensor1, tensor2)
        np.testing.assert_array_equal(expected, result.eval())

    def test_batch_to_matrix_indices(self):
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        data = [[0, 1, 3], [1, 2, 3]]
        expected = [[0, 0], [0, 1], [0, 3], [1, 1], [1, 2], [1, 3]]

        result = transform.batch_to_matrix_indices(ph, dtype=tf.int64)
        result = result.eval({ph: data})
        np.testing.assert_array_equal(expected, result)

        ph = tf.placeholder(dtype=tf.int64, shape=[None, 3])
        result = transform.batch_to_matrix_indices(ph, dtype=tf.int64)
        result = result.eval({ph: data})
        np.testing.assert_array_equal(expected, result)

        ph = tf.placeholder(dtype=tf.int64, shape=[None, 3, 1])
        self.assertRaises(ValueError, transform.batch_to_matrix_indices, ph)

    def test_sparse_put(self):
        tensor = tf.SparseTensor([[0, 0], [1, 0]], [2, 0.2], [2, 2])
        sp_values = tf.SparseTensor([[0, 0], [0, 1]], [3.0, 0], [2, 2])

        expected = tf.constant([[3., 0], [0.2, 0.]])

        result = transform.sparse_put(tensor, sp_values)
        result = tf.sparse_tensor_to_dense(result)

        np.testing.assert_array_equal(expected.eval(), result.eval())

    def test_dense_put(self):
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[3, 1], [1, 1]])

        sp_values = tf.SparseTensor(indices=[[0, 0]], values=[3], dense_shape=[2, 2])
        result = transform.dense_put(tensor, sp_values)

        np.testing.assert_array_equal(expected.eval(), result.eval())

        # updates are cast to the given tensor type
        # 0 as update should also work since sparse tensors have a set of indices used to
        # make the updated indices explicit
        tensor = tf.ones([2, 2], dtype=tf.int32)
        expected = tf.constant([[0, 1], [1, 1]])
        sp_values = tf.SparseTensor(indices=[[0, 0]], values=tf.constant([0], dtype=tf.float32), dense_shape=[2, 2])

        result = transform.dense_put(tensor, sp_values)
        np.testing.assert_array_equal(expected.eval(), result.eval())

        # test with unknown input batch dimension
        ph = tf.placeholder(dtype=tf.int32, shape=[None, 2])
        data = np.ones([2, 2])

        sp_values = tf.SparseTensor(indices=[[0, 0]], values=[0], dense_shape=[2, 2])
        result = transform.dense_put(ph, sp_values)
        np.testing.assert_array_equal(expected.eval(), result.eval({ph: data}))

    def test_fill_sp_ones(self):
        indices = [[0, 0], [1, 0]]
        dense_shape = [2, 2]
        expected = tf.SparseTensorValue(indices=indices, values=[1, 1], dense_shape=dense_shape)

        fill = transform.sparse_ones(indices, dense_shape, dtype=tf.float32)

        fill_dense = tf.sparse_tensor_to_dense(fill)
        expected_dense = tf.sparse_tensor_to_dense(expected)
        expected_dense = tf.cast(expected_dense, tf.float32)

        np.testing.assert_array_equal(fill_dense.eval(), expected_dense.eval())

    def test_to_sparse(self):
        c = [[1, 0], [2, 3]]

        sparse_tensor = transform.to_sparse(c)

        dense_shape = tf.shape(c, out_type=tf.int64)
        indices = tf.where(tf.not_equal(c, 0))

        flat_values = tf.reshape(c, [-1])
        flat_indices = tf.where(tf.not_equal(flat_values, 0))
        flat_indices = tf.squeeze(flat_indices)
        flat_indices = tf.mod(flat_indices, dense_shape[1])

        values = tf.gather_nd(c, indices)

        sp_indices = transform.sp_indices_from_sp_tensor(sparse_tensor)
        np.testing.assert_array_equal(sparse_tensor.indices.eval(), indices.eval())

        np.testing.assert_array_equal(sp_indices.values.eval(), flat_indices.eval())
        np.testing.assert_array_equal(sparse_tensor.values.eval(), values.eval())

    def test_to_sparse_zero(self):
        shape = [2, 3]
        data_zero = np.zeros(shape)
        sparse_tensor = transform.to_sparse(data_zero)

        self.assertEqual(sparse_tensor.eval().indices.shape[0], 0)

        dense = tf.sparse_tensor_to_dense(sparse_tensor)
        np.testing.assert_array_equal(dense.eval(), np.zeros(shape))

    def test_profile_one_hot_conversions(self):
        ph = tf.placeholder(dtype=tf.int64, shape=[2, 3])
        data = [[0, 1, 3],
                [1, 2, 3]]

        dense_shape = [2, 4]

        expected_dense = [[1, 1, 0, 1],
                          [0, 1, 1, 1]]

        sp_one_hot = transform.sparse_one_hot(ph, dense_shape)
        dense_one_hot = transform.dense_one_hot(ph, dense_shape)

        dense1 = self.ss.run(tf.sparse_tensor_to_dense(sp_one_hot), feed_dict={ph: data})
        dense2 = self.ss.run(dense_one_hot, feed_dict={ph: data})

        np.testing.assert_array_equal(dense1, expected_dense)
        np.testing.assert_array_equal(dense1, dense2)

    def test_l2_normalise(self):
        v1 = tf.constant([2., 0., -1.])
        v1s = transform.to_sparse(v1)

        self.assertIsInstance(v1s, tf.SparseTensor)

        v1_norm = transform.l2_normalize(v1, -1)
        v1s_norm = transform.l2_normalize(v1s, -1)
        v1s_norm_dense = tf.sparse_tensor_to_dense(v1s_norm)

        self.assertTrue(np.array_equal(v1_norm.eval(), v1s_norm_dense.eval()))


if __name__ == '__main__':
    unittest.main()
