""" Tests for TensorX random ops"""
import unittest

import tensorflow as tf
from tensorx import test_utils
from tensorx import random as random
from tensorx import transform
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestRandom(test_utils.TestCase):

    def test_sample(self):
        range_max = 10
        num_sampled = 2
        batch_size = 2
        samples = random.sample(range_max, num_sampled, unique=True)
        y, _ = tf.unique(samples)

        samples = random.sample(range_max, num_sampled, batch_size, unique=True)

        with self.cached_session(use_gpu=True):
            unique_set = self.eval(y)
            self.assertEqual(len(unique_set), num_sampled)

            for i in range(batch_size):
                y, _ = tf.unique(tf.squeeze(tf.gather(samples, [i])))
                unique_set = self.eval(y)
                self.assertEqual(len(unique_set), num_sampled)

    def test_dynamic_sample(self):
        shape = [1]
        input_ph = tf.placeholder(tf.int32, shape=shape)

        # fails because we can't find the exact value for num_sampled
        # at graph-building time
        with self.assertRaises(ValueError):
            random.sample(range_max=10, num_sampled=input_ph, unique=True)

        samples = random.sample(range_max=10, num_sampled=1, unique=True)
        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(tf.shape(samples), shape)

    def test_sample_range_max(self):
        range_max = 10
        num_sampled = 11
        sample = random.sample(range_max, num_sampled=num_sampled, unique=False)

        with self.cached_session(use_gpu=True):
            result = self.eval(sample)
            self.assertEqual(len(result), num_sampled)

    def test_salt_pepper_noise(self):
        batch_size = 8
        dim = 12
        noise_amount = 0.5

        noise_positive = random.salt_pepper_noise(batch_size, dim,
                                                  density=noise_amount,
                                                  salt_value=1,
                                                  pepper_value=0,
                                                  dtype=tf.float32)

        sum_tensor = tf.sparse_reduce_sum(noise_positive)
        max_tensor = tf.sparse_reduce_max(noise_positive)

        noise_symmetric = random.salt_pepper_noise(batch_size, dim,
                                                   density=noise_amount,
                                                   salt_value=1,
                                                   pepper_value=-1,
                                                   dtype=tf.float32)
        sum_symmetric = tf.sparse_reduce_sum(noise_symmetric)

        with self.cached_session(use_gpu=True):
            self.assertEqual(sum_tensor, int(dim * noise_amount) // 2 * batch_size)
            self.assertEqual(max_tensor, 1)
            self.assertEqual(sum_symmetric, 0)

    def test_sparse_random_normal(self):
        batch_size = 1000
        dim = 103
        density = 0.1

        sp_random = random.sparse_random_normal(dense_shape=[batch_size, dim], density=density)

        with self.cached_session(use_gpu=True):
            self.assertEqual(tf.shape(sp_random.indices)[0], int(density * dim) * batch_size)
            self.assertAlmostEqual(tf.reduce_mean(sp_random.values, axis=-1), 0, places=1)

    def test_random_bernoulli(self):
        n = 20
        batch_size = 1000
        prob = 0.5
        binary = random.random_bernoulli(shape=[batch_size, n], prob=prob)

        with self.cached_session(use_gpu=True):
            self.assertAlmostEqual(tf.reduce_mean(tf.reduce_mean(binary, -1)), prob, 1)

    def test_sparse_random_mask(self):
        batch_size = 2
        dim = 10
        density = 0.3
        mask_values = [1, -1]

        sp_mask = random.sparse_random_mask(dim=dim,
                                            batch_size=batch_size,
                                            density=density,
                                            mask_values=mask_values,
                                            symmetrical=False)
        dense_mask = tf.sparse.to_dense(sp_mask)

        sp_symmetrical_mask = random.sparse_random_mask(dim=dim,
                                                        batch_size=batch_size,
                                                        density=density,
                                                        mask_values=mask_values,
                                                        symmetrical=True)
        dense_symmetrical_mask = tf.sparse.to_dense(sp_symmetrical_mask)

        density = 0.5
        mask_values = [1, -1, 2, -2]
        sp_mask = random.sparse_random_mask(dim=dim,
                                            batch_size=batch_size,
                                            density=density,
                                            mask_values=mask_values,
                                            symmetrical=False)
        dense_mask_multiple_values = tf.sparse.to_dense(sp_mask)

        with self.cached_session(use_gpu=True):
            self.assertNotEqual(tf.reduce_sum(dense_mask), 0.0)
            self.assertNotEqual(tf.reduce_sum(dense_mask_multiple_values), 0.0)
            self.assertEqual(tf.reduce_sum(dense_symmetrical_mask), 0.0)

    def test_sample_sigmoid(self):
        shape = [2, 4]
        n_samples = 2
        v = np.random.uniform(size=shape)
        v = tf.nn.sigmoid(v)

        sample = random.sample_sigmoid_from_logits(v, n_samples)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(tf.shape(sample), [n_samples] + shape)


if __name__ == '__main__':
    test_utils.main()
