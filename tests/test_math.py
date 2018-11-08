import unittest
import tensorflow as tf
import numpy as np
from tensorx import test_utils

from tensorx import math
from tensorx import transform
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MyTestCase(test_utils.TestCase):

    def test_l2_norm(self):
        v1 = tf.constant([[2., 0., -1.], [2., 0., -1.]])
        v1s = transform.to_sparse(v1)

        norm1 = tf.norm(v1, axis=1)
        norm1s = math.sparse_l2_norm(v1s, axis=1)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(norm1, norm1s)

    def test_sparse_dot(self):
        v1 = tf.constant([[2., 0., -1.], [2., 0., -1.]])
        v1s = transform.to_sparse(v1)

        dot_sparse_dense1 = math.sparse_dot(v1s, v1)
        sp_sp_dots = math.sparse_dot(v1s, v1s)

        v2 = tf.constant(np.random.uniform(-1, 1, [2, 3]))
        v3 = tf.constant(np.random.uniform(-1, 1, [2, 3]))
        v2s = transform.to_sparse(v2)
        v3s = transform.to_sparse(v3)

        dot_dense = tf.reduce_sum(tf.multiply(v2, v3), axis=-1)
        dot_sparse_dense2 = math.sparse_dot(v2s, v3)
        sp_sp_dots2 = math.sparse_dot(v2s, v3s)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(dot_sparse_dense1, sp_sp_dots)
            # dense dot
            self.assertArrayEqual(dot_dense, dot_sparse_dense2)
            self.assertArrayEqual(dot_dense, sp_sp_dots2)

    def test_sparse_sparse_multiply(self):
        sp_tensor1 = tf.SparseTensor([[0, 0], [1, 0]], [2., 0.5], [2, 2])
        sp_tensor2 = tf.SparseTensor([[0, 0], [0, 1]], [4., 4.], [2, 2])
        dense_tensor = tf.convert_to_tensor([[4., 4.], [0., 0.]])

        result1 = math.sparse_multiply(sp_tensor1, sp_tensor2)
        expected_values = [8]

        result2 = math.sparse_multiply(sp_tensor1, dense_tensor)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(result1.values, expected_values)
            self.assertArrayEqual(result1.indices, result2.indices)
            self.assertArrayEqual(result1.values, result2.values)

    def test_sparse_multiply(self):
        v1_data = np.random.normal(-1., 1., [4])
        v2_data = np.random.normal(-1., 1., [4])
        v1 = tf.constant(v1_data, dtype=tf.float32)
        v2 = tf.constant(v2_data, dtype=tf.float32)
        v1s = transform.to_sparse(v1)

        dense_values = tf.gather_nd(v2, v1s.indices)
        dense_mul = tf.multiply(v1s.values, dense_values)

        result = math.sparse_multiply(v1s, v2)
        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(result.values, dense_mul)

    def test_logit(self):
        x_val = 0.2
        x = tf.constant(x_val, dtype=tf.float32)
        sigmoid = tf.nn.sigmoid(x)
        logits = math.logit(sigmoid)

        with self.cached_session(use_gpu=True):
            self.assertAlmostEqual(x_val, logits, places=6)

    def test_logit_sparse(self):
        x_val = 0.2
        x = tf.SparseTensor(indices=[[0, 0], [0, 1]], values=[x_val, x_val], dense_shape=[2, 2])
        sigmoid = tf.SparseTensor(indices=[[0, 0], [0, 1]], values=tf.nn.sigmoid([x_val, x_val]), dense_shape=[2, 2])
        logit = tf.SparseTensor(indices=[[0, 0], [0, 1]], values=math.logit(sigmoid.values), dense_shape=[2, 2])

        with self.cached_session(use_gpu=True):
            self.assertAllClose(x.values, logit.values, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
