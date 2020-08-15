import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pytest import approx
import tensorflow as tf
import tensorx as tx
import numpy as np


def test_dropout():
    """ Dropout with scaling preserves average input activation
    """
    x = tf.ones([100, 100])
    drop_probability = 0.2

    drop_x = tx.dropout(x, probability=drop_probability, scale=True)

    min_x = tf.reduce_min(drop_x)

    non_zero = tf.math.count_nonzero(drop_x)

    assert min_x == 0
    # approx doesn't work with Tensor
    assert non_zero.numpy() == approx(8000, abs=1000)


def test_dropout_random_tensor():
    x = tf.ones([100, 100])
    drop_probability = 0.2

    mask = np.random.uniform(size=x.shape)
    drop_x1 = tx.dropout(x, probability=drop_probability,
                         random_mask=mask,
                         scale=True)

    # probability becomes irrelevant
    drop_x2 = tx.dropout(x,
                         probability=0.9,
                         random_mask=mask,
                         scale=True)

    nonzero_indices1 = tf.where(tf.not_equal(drop_x1, 0))
    nonzero_indices2 = tf.where(tf.not_equal(drop_x2, 0))

    assert tx.tensor_equal(nonzero_indices1, nonzero_indices2)


def test_dropout_unscaled():
    x = tf.ones([100, 100])
    keep_prob = 0.5

    drop_x = tx.dropout(x, probability=keep_prob, scale=False)

    actual_avg = tf.reduce_mean(drop_x)
    expected_avg = tf.reduce_mean(x)

    assert actual_avg < expected_avg


def test_empty_sparse_tensor():
    dense_shape = [2, 2]
    empty = tx.empty_sparse_tensor(dense_shape)
    dense_empty = tf.sparse.to_dense(empty)
    zeros = tf.zeros(dense_shape)

    assert tf.reduce_all(tf.equal(zeros, dense_empty))


def test_sort_by_first():
    v1 = tf.constant([[3, 1], [2, 1]])
    sorted1 = [[1, 3], [1, 2]]
    v2 = tf.constant([[1, 2], [1, 2]])
    sorted2 = [[2, 1], [2, 1]]

    s1, s2 = tx.sort_by_first(v1, v2, ascending=True)

    assert tx.tensor_equal(s1, sorted1)
    assert tx.tensor_equal(s2, sorted2)

    s1, s2 = tx.sort_by_first([2, 1, 3], [1, 2, 3])
    sorted1 = [1, 2, 3]
    sorted2 = [2, 1, 3]
    assert tx.tensor_equal(s1, sorted1)
    assert tx.tensor_equal(s2, sorted2)
