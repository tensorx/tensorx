"""Tests gradient propagation for different ops

"""
import unittest
import os

import tensorflow as tf
import numpy as np
import tensorx as tx

from tensorflow.python.ops import linalg_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import array_ops, variables, math_ops
from tensorflow.python.framework import sparse_tensor, dtypes, ops
from tensorflow.python.ops.nn import embedding_lookup
import logging

from tensorx import transform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestTransform(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)

    def tearDown(self):
        self.ss.close()

    def test_sparse_tile(self):
        n = 4
        dim = 10000
        weight_dim = 3

        sp1 = tf.SparseTensorValue([[0, 0], [1, 1], [2, 3]], [1., 2., -7.], [3, dim])
        sp2 = tf.SparseTensorValue([[0, 0], [1, 2], [2, 2]], [0.5, 5., 2.], [3, dim])
        sp1 = tf.convert_to_tensor_or_sparse_tensor(sp1)
        sp2 = tf.convert_to_tensor_or_sparse_tensor(sp2)


        tsp1 = transform.sparse_tile(sp1, n)
        # print(tsp1.eval())

        tsp2 = transform.sparse_tile(sp2, n)

        weights = tf.get_variable("test_weights", shape=[dim, weight_dim])
        tf.global_variables_initializer().run()

        ids1 = tx.sparse_indices(sp1)
        ids2 = tx.sparse_indices(sp2)
        #sp1 = None
        #sp2 = None

        v1 = tx.embedding_lookup_sparse(weights, ids1, sp1, combiner="sqrtn")
        v2 = tx.embedding_lookup_sparse(weights, ids2, sp2, combiner="sqrtn")

        v1_o = tf.nn.embedding_lookup_sparse(weights, ids1, sp1, combiner="sqrtn")
        v2_o = tf.nn.embedding_lookup_sparse(weights, ids2, sp2, combiner="sqrtn")

        # v1nd = tf.gather_nd(weights,ids1.indices)
        # v1nd = tf.

        tv1 = tx.embedding_lookup_sparse(weights, tx.sparse_indices(tsp1), tsp1, combiner="sqrtn")
        tv1 = tx.embedding_lookup_sparse(weights, tx.sparse_indices(tsp1), tsp1, combiner="sqrtn")
        tv2 = tx.embedding_lookup_sparse(weights, tx.sparse_indices(tsp2), tsp2, combiner="sqrtn")

        loss1 = v1 - v2
        loss2 = v1_o - v2_o
        loss3 = tv1 - tv2

        # loss1 = tf.reduce_sum(v1 - v2)
        # loss2 = tf.reduce_sum(tv1 - tv2)

        train1 = self.opt.minimize(loss1)
        train2 = self.opt.minimize(loss2)
        train3 = self.opt.minimize(loss3)

        train1.run()
        train2.run()
        train3.run()

        print(v1.eval())
        print(v1_o.eval())
        # print(v1nd.eval())
        # print(v1.eval())
        # train2.run()


if __name__ == '__main__':
    unittest.main()
