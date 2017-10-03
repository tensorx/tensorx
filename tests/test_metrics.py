import unittest
import tensorflow as tf
from tensorx import transform, metrics
import numpy as np
from numpy.linalg import norm

import time


def np_cosine_dist(u, v):
    dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
    return dist


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_cosine_distance(self):
        v1_data = np.random.normal(0., 1., [2])
        v2_data = np.random.normal(0., 1., [2])
        v3_data = np.random.normal(0., 1., [2])

        v1 = tf.constant(v1_data, dtype=tf.float32)
        v2 = tf.constant(v2_data, dtype=tf.float32)

        dist0 = metrics.cosine_distance(v1, v1, 0)
        self.assertAlmostEqual(dist0.eval(), 0, places=6)

        v1s = transform.to_sparse(v1)
        self.assertIsInstance(v1s, tf.SparseTensor)

        dist1_dense = metrics.cosine_distance(v1, v2, 0)
        dist1_sparse = metrics.cosine_distance(v1s, v2, 0)
        self.assertEqual(dist1_dense.eval(), dist1_sparse.eval())

        v3 = tf.constant(v3_data, dtype=tf.float32)

        dist2_dense = metrics.cosine_distance(v1, v3, 0)
        dist2_sparse = metrics.cosine_distance(v1s, v3, 0)
        self.assertEqual(dist2_dense.eval(), dist2_sparse.eval())

    def test_sparse_cosine_distance(self):
        dense_shape = [100, 1000]

        v4_data = np.random.normal(0., 1., dense_shape)
        v4 = tf.constant(v4_data, dtype=tf.float32)
        v5_data = np.random.normal(0., 1., dense_shape)
        v5 = tf.constant(v4_data, dtype=tf.float32)

        # we want to compute the cosine distance along dimension 1 because we have shape [1,2]

        dist4 = metrics.cosine_distance(v4, v4, dim=1)
        start = time.time()
        dist4.eval()
        end = time.time()
        self.assertTrue(np.allclose(dist4.eval(), 0., atol=1e-6))

        v6s = tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1]], values=[1., 2.,1.], dense_shape=dense_shape)
        v6 = tf.sparse_tensor_to_dense(v6s)

        dist5_dense = metrics.cosine_distance(v6, v5, 1)
        dist5_sparse = metrics.cosine_distance(v6s, v5, 1)
        #print(dist5_dense.eval())
        #print(dist5_sparse.eval())

        self.assertTrue(np.array_equal(dist5_dense.eval(), dist5_sparse.eval()))

        dist60 = metrics.cosine_distance(v6s, v6, 1)
        #print(v6.eval())
        print(dist60.eval())
        #self.assertTrue(np.allclose(dist60.eval(), 0))

        start = time.time()
        metrics.cosine_distance(v6, v6, 1).eval()
        end = time.time()
        print("time dense ", end - start)

        start = time.time()
        metrics.cosine_distance(v6s, v6, 1).eval()
        end = time.time()
        print("time sparse ", end - start)

        start = time.time()
        metrics.cosine_distance_v2(v6, v6, 1).eval()
        end = time.time()
        print("time dense v2 ", end - start)

        start = time.time()
        metrics.cosine_distance_v2(v6s, v6, 1).eval()
        end = time.time()
        print("time sparse v2 ", end - start)


if __name__ == '__main__':
    unittest.main()
