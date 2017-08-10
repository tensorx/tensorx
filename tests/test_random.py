""" Tests for TensorX random ops"""
import unittest

import tensorflow as tf

from tensorx import random as tr


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

        samples = tr.sample(range_max, [num_sampled], unique=True)

        y, _ = tf.unique(samples)
        unique_set = y.eval()
        self.assertEqual(len(unique_set), num_sampled)

        samples = tr.sample(range_max, [batch_size, num_sampled], unique=True)
        for i in range(batch_size):
            y, _ = tf.unique(tf.squeeze(tf.gather(samples, [i])))
            unique_set = y.eval()
            self.assertEqual(len(unique_set), num_sampled)

    def test_sample_range_max(self):
        range_max = 10
        num_sampled = 11

        try:
            sample = tr.sample(range_max, [num_sampled], unique=True)
            sample.eval()
        except:
            pass

        try:
            sample = tr.sample(range_max, [num_sampled], unique=False)
            sample.eval()
        except Exception:
            self.fail("should have not raised an exception since the number of samples > range_max but unique == False")

    def test_salt_pepper_noise(self):
        batch_size = 8
        dim = 12
        noise_amount = 0.5

        noise_tensor = tr.salt_pepper_noise([batch_size, dim], noise_amount=noise_amount, max_value=1, min_value=0,
                                            dtype=tf.float32)
        sum_tensor = tf.sparse_reduce_sum(noise_tensor)
        self.assertEqual(sum_tensor.eval(), (dim * noise_amount // 2) * batch_size)

        # use negative pepper
        noise_tensor = tr.salt_pepper_noise([batch_size, dim], noise_amount=noise_amount, max_value=1, min_value=-1,
                                            dtype=tf.float32)
        sum_tensor = tf.sparse_reduce_sum(noise_tensor)
        self.assertEqual(sum_tensor.eval(), 0)

        # diff number of salt and pepper
        # 10 // 2 == 5
        # 5 // 2 == 2
        # num_pepper = 5 - 2 == 3
        dim = 10
        noise_tensor = tr.salt_pepper_noise([batch_size, dim], noise_amount=noise_amount, max_value=1, min_value=-1,
                                            dtype=tf.float32)
        sum_tensor = tf.sparse_reduce_sum(noise_tensor)
        self.assertEqual(sum_tensor.eval(), -1 * batch_size)


if __name__ == '__main__':
    unittest.main()
