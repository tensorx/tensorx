""" Tests for TensorX random ops"""
import unittest

import tensorflow as tf

from tensorx import random as random
import numpy as np


class TestRandom(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_sample(self):
        range_max = 10
        num_sampled = 2
        batch_size = 2

        samples = random.sample(range_max, num_sampled, unique=True)

        y, _ = tf.unique(samples)
        unique_set = y.eval()
        self.assertEqual(len(unique_set), num_sampled)

        samples = random.sample(range_max, num_sampled, batch_size, unique=True)
        for i in range(batch_size):
            y, _ = tf.unique(tf.squeeze(tf.gather(samples, [i])))
            unique_set = y.eval()
            self.assertEqual(len(unique_set), num_sampled)

    def test_dynamic_sample(self):

        shape = [1]
        input_ph = tf.placeholder(tf.int32, shape=shape)

        # fails because we can't find the exact value for num_sampled
        # at graph-building time
        with self.assertRaises(ValueError):
            random.sample(range_max=10, num_sampled=input_ph, unique=True)

        # THIS SUCCEEDS
        samples = random.sample(range_max=10, num_sampled=1, unique=True)
        np.testing.assert_array_equal(tf.shape(samples).eval(), shape)

    def test_sample_range_max(self):
        range_max = 10
        num_sampled = 11

        try:
            sample = random.sample(range_max, num_sampled, unique=True)
            sample.eval()
        except:
            pass

        try:
            sample = random.sample(range_max, num_sampled, unique=False)
            sample.eval()
        except Exception:
            self.fail("should have not raised an exception since the number of samples > range_max but unique == False")

    def test_salt_pepper_noise(self):
        batch_size = 8
        dim = 12
        noise_amount = 0.5

        noise_tensor = random.salt_pepper_noise([batch_size, dim], density=noise_amount, max_value=1, min_value=0,
                                                dtype=tf.float32)
        sum_tensor = tf.sparse_reduce_sum(noise_tensor)
        max_tensor = tf.sparse_reduce_max(noise_tensor)

        self.assertEqual(sum_tensor.eval(), int(dim * noise_amount) // 2 * batch_size)
        self.assertEqual(max_tensor.eval(), 1)
        # self.assertEqual(min_tensor,0)

        # use negative pepper
        noise_tensor = random.salt_pepper_noise([batch_size, dim], density=noise_amount, max_value=1, min_value=-1,
                                                dtype=tf.float32)
        sum_tensor = tf.sparse_reduce_sum(noise_tensor)
        self.assertEqual(sum_tensor.eval(), 0)

        dim = 10
        noise_tensor = random.salt_pepper_noise([batch_size, dim], density=noise_amount, max_value=1, min_value=-1,
                                                dtype=tf.float32)
        sum_tensor = tf.sparse_reduce_sum(noise_tensor)
        self.assertEqual(sum_tensor.eval(), 0)

    def test_sparse_random_normal(self):
        batch_size = 1000
        dim = 103
        density = 0.1

        sp_random = random.sparse_random_normal(dense_shape=[batch_size, dim], density=density)
        result = sp_random.eval()

        self.assertEqual(len(result.indices), int(density * dim) * batch_size)
        self.assertAlmostEqual(np.mean(result.values), 0, places=1)

    def test_sparse_random_mask(self):
        batch_size = 2
        dim = 10
        density = 0.3
        mask_values = [1, -1]

        sp_mask = random.sparse_random_mask([batch_size, dim], density, mask_values, symmetrical=False)
        dense_mask = tf.sparse_tensor_to_dense(sp_mask, validate_indices=False)
        dense_result = dense_mask.eval()
        self.assertNotEqual(np.sum(dense_result), 0)

        density = 0.5
        mask_values = [1, -1, 2]
        sp_mask = random.sparse_random_mask([batch_size, dim], density, mask_values, symmetrical=False)
        dense_mask = tf.sparse_tensor_to_dense(sp_mask, validate_indices=False)
        print(dense_mask.eval())


if __name__ == '__main__':
    unittest.main()
