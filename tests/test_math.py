import unittest
import tensorflow as tf
import numpy as np

from tensorx import math as mathx
from tensorx import transform


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_l2_norm(self):
        v1 = tf.constant([[2., 0., -1.], [2., 0., -1.]])
        v1s = transform.to_sparse(v1)

        norm1 = tf.norm(v1, axis=1)
        norm1s = mathx.sparse_l2_norm(v1s, axis=1)

        self.assertTrue(np.array_equal(norm1.eval(), norm1s.eval()))

    def test_sparse_dot(self):
        v1 = tf.constant([[2., 0., -1.], [2., 0., -1.]])
        v1s = transform.to_sparse(v1)

        dots = mathx.sparse_dot(v1s, v1)
        dotss = mathx.sparse_dot(v1s, v1s)

        self.assertTrue(np.array_equal(dots.eval(), dotss.eval()))

        v2 = tf.constant(np.random.uniform(-1, 1,[2, 3]))
        v3 = tf.constant(np.random.uniform(-1, 1,[2, 3]))
        v2s = transform.to_sparse(v2)
        v3s = transform.to_sparse(v3)

        # dense dot
        dotd = tf.reduce_sum(tf.multiply(v2, v3), axis=-1)
        dots = mathx.sparse_dot(v2s, v3)
        dotss = mathx.sparse_dot(v2s, v3s)

        self.assertTrue(np.array_equal(dotd.eval(), dots.eval()))
        self.assertTrue(np.array_equal(dotd.eval(), dotss.eval()))

    def test_sparse_sparse_multiply(self):
        sp_tensor1 = tf.SparseTensor([[0, 0], [1, 0]], [2., 0.5], [2, 2])
        sp_tensor2 = tf.SparseTensor([[0, 0], [0, 1]], [4., 4.], [2, 2])
        dense_tensor = tf.convert_to_tensor([[4., 4.], [0., 0.]])

        result1 = mathx.sparse_multiply(sp_tensor1, sp_tensor2)
        expected_values = [8]
        self.assertTrue(np.array_equal(expected_values, result1.values.eval()))

        result2 = mathx.sparse_multiply(sp_tensor1, dense_tensor)
        self.assertTrue(np.array_equal(result1.indices.eval(), result2.indices.eval()))
        self.assertTrue(np.array_equal(result1.values.eval(), result2.values.eval()))

    def test_sparse_multiply(self):
        v1_data = np.random.normal(-1., 1., [4])
        v2_data = np.random.normal(-1., 1., [4])
        v1 = tf.constant(v1_data, dtype=tf.float32)
        v2 = tf.constant(v2_data, dtype=tf.float32)
        v1s = transform.to_sparse(v1)

        dense_values = tf.gather_nd(v2, v1s.indices)
        dense_mul = tf.multiply(v1s.values, dense_values)

        result = mathx.sparse_multiply(v1s,v2)
        self.assertTrue(np.array_equal(result.values.eval(),dense_mul.eval()))

if __name__ == '__main__':
    unittest.main()
