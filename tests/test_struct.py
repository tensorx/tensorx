from tensorx import test_utils
import tensorx as tx
import tensorflow as tf
import os
from tensorx.struct import IndexValueTable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestStruct(test_utils.TestCase):

    def test_index_value_table(self):
        i = [[1, 0], [2, 3], [5, 4]]
        v = [[-1, 1], [1, -1], [-1, 1]]

        expected = [[1, -1, 0, 0, 0, 0],
                    [0, 0, 1, -1, 0, 0],
                    [0, 0, 0, 0, 1, -1]]

        table = IndexValueTable.from_list(i, v, 6)

        sp_tensor = table.to_sparse_tensor()
        result = tf.sparse.to_dense(sp_tensor)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(result, expected)
