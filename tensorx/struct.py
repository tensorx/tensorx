""" TensorX data structures

Includes sparse tables, dynamic variables, and related ops

"""
from tensorx.utils import to_tensor_cast
from tensorflow.python.framework import dtypes, ops
from tensorx.transform import sort_by_first, to_matrix_indices
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.ops import array_ops, sparse_ops, math_ops, state_ops
from tensorflow.python.ops.variables import Variable, VariableMetaclass
from tensorflow.python.framework import tensor_util
# import tensorflow as tf
import six
from tensorflow.python.training.checkpointable import base as checkpointable


class IndexValueTable:
    """ Index-Value Table

    Implements a Sparse Tensor that can be stored as two tables of (indices, values),
    the difference between this and sparse tensor being that rows in the table must have

    Args:
            indices: a tensor with the indices with values < k
            values: a tensor with the same shape as indices with the values corresponding to each index
            k: the dimension of the table, all indices must be lower than this value

    """

    def __init__(self, indices, values, k):
        self.indices = to_tensor_cast(indices)
        self.values = to_tensor_cast(values)
        self.k = to_tensor_cast(k)

    @staticmethod
    def from_list(indices, values, k):
        indices, values = sort_by_first(indices, values)
        return IndexValueTable(indices, values, k)

    def to_sparse_tensor(self, reorder=False):
        with name_scope("to_sparse"):
            # [[0,2],[0,4]] --> [[0,0],[0,2],[1,0],[1,4]]
            indices = to_matrix_indices(self.indices, dtype=dtypes.int64)
            values = array_ops.reshape(self.values, [-1])

            # num_rows = self.indices.get_shape().as_list()[0]
            num_rows = array_ops.shape(self.indices, out_type=dtypes.int64)[0]
            num_cols = math_ops.cast(self.k, dtype=dtypes.int64)

            dense_shape = array_ops.stack([num_rows, num_cols])

            sp = SparseTensor(indices, values, dense_shape)
            if reorder:
                sp = sparse_ops.sparse_reorder(sp)

        return sp

    def gather(self, ids):
        with name_scope("gather"):
            ids = to_tensor_cast(ids, dtypes.int64)
            ids = array_ops.reshape(ids, [-1])
            indices = array_ops.gather(self.indices, ids)
            values = array_ops.gather(self.values, ids)

        return IndexValueTable(indices, values, self.k)


__all__ = ["IndexValueTable"]
