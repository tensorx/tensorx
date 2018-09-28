""" TensorX data structures

Includes sparse tables, dynamic variables, and related ops

"""
from tensorx.utils import to_tensor_cast
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.nn import top_k


def sorted_indices_values(indices, values):
    """ Sort indices and values according to the index sorting

    Args:
        indices: indices to be sorted
        values: values to be sorted according to the indices

    Returns:
        (indices, values) sorted according to the indices

    """
    indices, values = zip(*[(i, v) for i, v in sorted(zip(indices, values), key=lambda pair: pair[0])])

    return list(indices), list(values)


class SparseTable:
    """ Sparse Table
    Implements a Sparse Tensor that can be stored as two tables of (indices, values),
    the difference between this and sparse tensor being that rows in the table must have

    """

    def __init__(self, indices, values, k, s, dtype=dtypes.float32):
        self.indices = to_tensor_cast(indices, dtypes.int64)
        self.values = to_tensor_cast(values)
        self.k = k
        self.s = s

    @staticmethod
    def from_ri_list(ri_list, k, s, dtype=tf.float32):
        iv = [ri.sorted_indices_values() for ri in ri_list]
        indices, values = zip(*iv)
        return SparseTable(indices, values, k, s, dtype)

    def to_sparse_tensor(self, reorder=False):
        with tf.name_scope("to_sparse"):
            """
            [[0,2],[0,4]] ---> [[0,0],[0,2],[1,0],[1,4]]
            """
            indices = tx.column_indices_to_matrix_indices(self.indices, dtype=tf.int64)
            values = tf.reshape(self.values, [-1])

            # num_rows = self.indices.get_shape().as_list()[0]
            num_rows = tf.shape(self.indices, out_type=tf.int64)[0]
            num_cols = tf.convert_to_tensor(self.k, dtype=tf.int64)

            dense_shape = tf.stack([num_rows, num_cols])

            sp = tf.SparseTensor(indices, values, dense_shape)
            if reorder:
                sp = tf.sparse_reorder(sp)

        return sp

    def gather(self, ids):
        with tf.name_scope("gather"):
            ids = to_tensor_cast(ids, tf.int64)
            ids = tf.reshape(ids, [-1])
            indices = tf.gather(self.indices, ids)
            values = tf.gather(self.values, ids)

        return RandomIndexTensor(indices, values, self.k, self.s)
