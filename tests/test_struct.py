from unittest import TestCase
import tensorx as tx
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestStruct(TestCase):

    def setUp(self):
        self.ss = tf.InteractiveSession()

    def test_index_value_table(self):
        i = [[1, 0], [2, 3], [5, 4]]
        v = [[-1, 1], [1, -1], [-1, 1]]

        table = tx.IndexValueTable.from_list(i, v, 6)

        sp_tensor = table.to_sparse_tensor()

        self.assertTrue(np.array_equal(tf.sparse_tensor_to_dense(sp_tensor).eval(),
                                       [[1, -1, 0, 0, 0, 0],
                                        [0, 0, 1, -1, 0, 0],
                                        [0, 0, 0, 0, 1, -1]]))
