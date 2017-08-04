""" TensorFlow tensor Transformation
Utilities to convert between and combine tensors
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import sparse_ops as sp_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework.sparse_tensor import SparseTensor, SparseTensorValue
from tensorflow.python.ops import functional_ops as fn_ops

from numpy import array as np_array


def sparse_put(sp_tensor, sp_updates):
    """Sparse Put
    Changes a given SparseTensor according to the updates specified in a SparseTensor


    Args:
        sp_tensor: a sparse tensor we wich to set some indices to given values
        sp_updates: a sparse tensor with the indices to be changed and the respective values

    The resulting tensor will have the same values as the input tensor, except for the indices
    overlapping with update tensor which will be getting the updates.
    """

    # 1 concat indices and establish final tensor shape
    update_shape = sp_updates.values.get_shape()
    zero_updates = SparseTensor(sp_updates.indices,
                                array_ops.zeros(update_shape, dtype=dtypes.float32),
                                sp_updates.dense_shape)
    proto_result = sp_ops.sparse_add(sp_tensor, zero_updates)

    # shape of resulting values tensor
    proto_shape = array_ops.shape(proto_result.values)

    # 2 get mask for input tensor
    proto_ones = SparseTensor(proto_result.indices,
                              array_ops.ones(proto_shape, dtypes.int8),
                              proto_result.dense_shape)

    # mask_ones = math_ops.scalar_mul(-1, array_ops.ones(update_shape))
    sp_mask = SparseTensor(sp_updates.indices,
                           array_ops.constant(-1, dtypes.int8, update_shape),
                           sp_updates.dense_shape)

    to_retain = sp_ops.sparse_add(proto_ones, sp_mask)
    to_retain = math_ops.not_equal(to_retain.values, 0)

    # get tensor with masked values
    tensor_masked = sp_ops.sparse_retain(proto_result, to_retain)

    # add values to entries previously set to 0
    return sp_ops.sparse_add(tensor_masked, sp_updates)


def dense_put(tensor, sp_updates):
    """ Dense Put

    Changes a given tensor according to the updates specified in a SparseTensor

    Args:
        tensor: a dense tensor we want to change
        sp_updates: a sparse tensor with the indices to be changed and the respective values

    The resulting tensor will have the same values as the input tensor, except for the indices
    overlapping with update tensor which will be getting the updates.
    """
    tensor = ops.convert_to_tensor(tensor)
    if sp_updates.dtype != tensor.dtype:
        sp_updates = math_ops.cast(sp_updates,tensor.dtype)

    markers = array_ops.ones(shape=array_ops.shape(sp_updates.values))
    sparse_marker_tensor = SparseTensor(indices=sp_updates.indices,values=markers,dense_shape=sp_updates.dense_shape)
    dense_update_marker = sp_ops.sparse_tensor_to_dense(sparse_marker_tensor)
    dense_updates = sp_ops.sparse_tensor_to_dense(sp_updates)
    return array_ops.where(math_ops.not_equal(dense_update_marker, 0),
                           dense_updates,
                           tensor)


def to_sparse(tensor):
    """
    Returns a sparse representation for a given multi-dimensional tensor

    Args:
        tensor a dense tensor to be converted
    Return:
        (sp_indices,sp_values)
        sp_indices is a sparse tensor with the values for the indices to be returned
        sp_values is a sparse tensor with the values to be attributed to each index
    """

    indices = array_ops.where(math_ops.not_equal(tensor, 0))
    dense_shape = tensor.get_shape()

    # Sparse Tensor for sp_indices
    flat_layer = array_ops.reshape(tensor, [-1])
    values = math_ops.mod(array_ops.squeeze(array_ops.where(math_ops.not_equal(flat_layer, 0))), dense_shape[1])

    sp_indices = SparseTensor(indices, values, dense_shape)

    # Sparse Tensor for values
    values = array_ops.gather_nd(tensor, indices)
    sp_values = SparseTensor(indices, values, dense_shape)

    return sp_indices, sp_values


def pairs(tensor1, tensor2):
    """
    Returns a tensor resulting from the pairwise combination of the elements of each tensor

    t1 = [0,1]
    t2 = [2,3,4]
    pairs(t1,t2) == [[[0,2],[0,3],[0,4]],[[1,2],[1,3],[1,4]],...]
    """
    return array_ops.squeeze(array_ops.stack(array_ops.meshgrid(tensor1, tensor2), axis=-1), name="pairs")


def enum_row(tensor, name="row_enum", dtype=dtypes.int64):
    with ops.name_scope(name):
        """ Converts a tensor with an equal amount of values per row
        e.g. [[1,2],
              [2,5]] to a rank 2 tensor with the enumeration of
        (row index, value) pairs
    
        e.g. [[0,1],[0,2],
              [1,2],[1,5]]
    
        Args:
            tensor: the tensor to be converted
    
        Returns:
            a rank-2 tensor with (row index,value) for each element in the given tensor
    
        """
        tensor = ops.convert_to_tensor(tensor)
        shape = tensor.get_shape()

        # for each coordinate
        row_i = math_ops.range(0, shape[0], dtype=dtype)
        enum = fn_ops.map_fn(lambda i: pairs(i, tensor[i]), elems=row_i, dtype=dtype)

        enum = array_ops.reshape(enum, shape=[-1, 2], name="ids")

        return enum


""" Prepare TensorFlow Inputs
Utilities to prepare inputs for a TensorFlow graph
    e.g. create sparse tensor values, etc
"""


def index_list_to_sparse(indices, shape):
    """
    Converts a list of lists of indexes to a sparse tensor value with the given shape

    example:

    idx =[[0,5],[0,2,7],[1]]

    we want to transform this into:

    SparseTensorValue(indices=[[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]],
                 values=[0,5,0,2,7,1],
                 dense_shape=[3,10])

    this can be then fed to a tf.sparse_placeholder

    if any index value >= shape[1] it raises an exception

    Args:
        indices: list of lists of indexes
        shape: the given shape, typically [BATCH_SIZE, MAX_INDEX]
        a sparse tensor with the sparse indexes
    """
    idx = []
    for row, indexes in enumerate(indices):
        for i in indexes:
            if i >= shape[1]:
                raise Exception("Invalid shape: index value " + i + " >= ", shape[1])
            idx.append([row, i])
    idx = np_array(idx)
    values = np_array(sum(indices, []))

    return SparseTensorValue(indices=idx, values=values, dense_shape=shape)


def value_list_to_sparse(values, sp_indices, shape):
    """ Converts a list of value vectors to a sparse tensor value, maps each index in
    the given sp_indices to each value.

    sp_indices have the form of an array [[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]]

    Args:
        values: values to be encapsulated by the sparse tensor value
        sp_indices: indices to be mapped to each value
        shape: given shape of the sparse tensor value

    Returns:
        A sparse tensor value with each index mapping to the given values
    """
    if len(sp_indices) != len(values):
        raise Exception(
            "Number of indices doesn't match number of values: %d != %d".format(len(sp_indices), len(values)))

    return SparseTensorValue(indices=sp_indices, values=values, dense_shape=shape)
