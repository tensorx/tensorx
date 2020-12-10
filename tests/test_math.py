import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
import numpy as np
import pytest
import tensorflow as tf


def test_sparse_dense_mul():
    x = tf.random.uniform([2, 4])
    s = tx.to_sparse(x)
    v = tf.random.uniform([4, 3])

    # matmul
    mul0 = tf.matmul(x, v)
    mul1 = tf.sparse.sparse_dense_matmul(s, v)

    assert tx.tensor_all_close(mul0, mul1)

    # element-wise
    mul2 = tx.sparse_dense_multiply(s, x)
    mul3 = tf.multiply(x, x)

    assert tx.tensor_all_close(mul2, mul3)
