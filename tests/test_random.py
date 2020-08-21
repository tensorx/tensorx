import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
import tensorflow as tf
import pytest


def test_gumbel_sample():
    shape = [4, 10]
    logits = tf.random.uniform(shape)
    sample = tx.gumbel_top(logits, 2)
    all_unique = tf.map_fn(lambda x: tf.equal(tf.unique(x)[0].shape[-1], 2), sample, dtype=tf.bool)
    all_unique = tf.reduce_all(all_unique, axis=0)
    assert all_unique
    assert sample.shape == [4, 2]


def test_random_bernoulli():
    prob = 0.2
    binary = tx.random.bernoulli(shape=[10, 1000], prob=prob)
    ones_prob = tf.reduce_mean(binary, axis=-1)
    ones_expected = tf.constant(prob, shape=tf.shape(ones_prob))

    assert tx.tensor_all_close(ones_prob,
                               ones_expected,
                               atol=4e-2)


def test_sample_sigmoid():
    shape = [4, 4]
    n_samples = 2
    # get some random logits
    logits = tf.random.uniform(shape)
    p = tf.nn.sigmoid(logits)

    # expensive to sample
    # u ~ Uniform(0,1)
    # Y = sigmoid(x) > u
    # equivalent to fast x > logit(u).

    sample = tx.sample_sigmoid(logits, n_samples)

    assert tx.tensor_equal(tf.shape(sample), [n_samples] + shape)
