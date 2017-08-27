""" TensorFlow tensor Transformation
Utilities to convert between and combine tensors
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops, tensor_util
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import sparse_ops as sp_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework.sparse_tensor import SparseTensor, SparseTensorValue
from tensorflow.python.ops.nn import dropout

from numpy import array as np_array


def empty_sparse_tensor(dense_shape, dtype=dtypes.float32):
    """ Creates an empty SparseTensor

    TODO make this work for dense_shape values with shape [1] as well

    Args:
        dtype: the dtype of the values for the empty SparseTensor
        dense_shape: a 1-D tensor of type int64 and shape [2] with the dense_shape for the sparse tensor

    Returns:
        an empty sparse tensor with a given shape

    """
    if not tensor_util.is_tensor(dense_shape):
        dense_shape = ops.convert_to_tensor(dense_shape, dtypes.int32)

    empty_indices = array_ops.ones([0, 2], dtype=dtypes.int64)
    empty_values = array_ops.ones([0], dtype=dtype)

    if dense_shape.dtype != dtypes.int64:
        dense_shape = math_ops.cast(dense_shape, dtypes.int64)
    return SparseTensor(empty_indices, empty_values, dense_shape)


def pairs(tensor1, tensor2):
    """
    Returns a tensor resulting from the pairwise combination of the elements of each tensor

    t1 = [0,1]
    t2 = [2,3,4]
    pairs(t1,t2) == [[0,2],[1,2],[0,3],[1,3],...]
    """
    x, y = array_ops.meshgrid(tensor1, tensor2)
    result = array_ops.stack([x, y], axis=-1)
    return array_ops.reshape(result, [-1, 2], name="pairs")


def enum_row(tensor, name="enum_row", dtype=dtypes.int64):
    """ Converts a tensor of int32 or int64 to a 2-D tensor with row-value pairs.

    For each row `r` with `d` columns, each value `i` is considered an index for a 1-D tensor
    and converted to pairs [[r,i1],[r,i2],[r,id]].

    Example:

            [[1,2],
             [2,5]] is converted to a `SparseTensor` with

             indices = [[0,1],
                        [0,2],
                        [1,2],
                        [1,5]]

    Use Case:
        Convert a batch of indices (used to slice another tensor with embedding lookup or gather)
        to be used in a SparseTensor, so that we can change the weights of each slice.

    Args:
        dtype: tensor type
        name: tensor name to create a scope for the op
        tensor: the tensor to be converted

        Returns:
            a 2-D tensor with (row index,value) for each element in the given tensor

    """
    with ops.name_scope(name):
        tensor = ops.convert_to_tensor(tensor, dtype=dtype)
        shape = array_ops.shape(tensor)

        rows = math_ops.range(math_ops.cast(shape[0], dtype))
        rows = array_ops.expand_dims(rows, 1)

        multiples = array_ops.stack([1, shape[1]])
        rows = array_ops.tile(rows, multiples)

        enum = array_ops.stack([rows, tensor], axis=-1)
        enum = array_ops.reshape(enum, shape=[-1, 2])
        return enum


def to_tensor_cast(x, dtype):
    """ Converts to tensor and casts to a given type if possible

    Args:
        x: an input Tensor.
        dtype: the type we which to cast the input tensor into

    Returns:
        a tensor of type dtype
    """
    x = ops.convert_to_tensor(x)
    if x.dtype != dtype:
        x = math_ops.cast(x, dtype)
    return x


def sparse_put(sp_tensor, sp_updates):
    """Sparse Put
    Changes a given SparseTensor according to the updates specified in a SparseTensor


    Args:
        sp_tensor: a sparse tensor we which to set some indices to given values
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
        sp_updates = math_ops.cast(sp_updates, tensor.dtype)

    markers = array_ops.ones(shape=array_ops.shape(sp_updates.values))
    sparse_marker_tensor = SparseTensor(indices=sp_updates.indices, values=markers, dense_shape=sp_updates.dense_shape)
    dense_update_marker = sp_ops.sparse_tensor_to_dense(sparse_marker_tensor)
    dense_updates = sp_ops.sparse_tensor_to_dense(sp_updates)
    return array_ops.where(math_ops.not_equal(dense_update_marker, 0),
                           dense_updates,
                           tensor)


def sparse_dropout(sp_tensor, keep_prob=0.2, seed=None):
    """Performs a dropout computation on sparse tensors

    Implementation Note:
        if sp_indices comes from a sparse_placeholder, and the batch_size is unknown
        the values in this sparse tensor have a dynamic shape that is only computed once
        the values are fed, this means we have to supply unstack with num, otherwise this cannot be inferred
    """
    dense_shape = sp_tensor.dense_shape

    drop_values = dropout(sp_tensor.values, keep_prob, seed=seed)
    not_zero = math_ops.not_equal(drop_values, 0)

    # new indices (after dropout)
    # not_zero_indices = array_ops.where(not_zero)
    # indices = array_ops.gather_nd(sp_indices.indices, not_zero_indices)

    values = array_ops.boolean_mask(drop_values, not_zero)
    indices = array_ops.boolean_mask(sp_tensor.indices, not_zero)
    return SparseTensor(indices, values, dense_shape)

"""
TODO this caused a problem with the constant because the shape of the input was unknown
in these cases I can use broadcasting or fill to create a tensor with dynamic shape
I need to test all of these operations with inputs of unknown shape, meaning the shape will be dynamic
another option is to use fill
"""


def default_sp_values(sp_indices, dtype=dtypes.float32):
    values = array_ops.ones_like(sp_indices.values, dtype=dtype)
    return SparseTensor(sp_indices.indices, values, sp_indices.dense_shape)


def fill_sp_ones(sp_indices, dtype=dtypes.float32):
    """ Replaces the given sparse tensor values with 1.

    Args:
        sp_indices: a ``SparseTensor`` or ``SparseTensorValue``
        dtype: the tensor type for the values

    Returns:
        ``SparseTensor``: a new SparseTensor with the values set to 1.
    """
    n_active = array_ops.shape(sp_indices.indices)[0]

    values = array_ops.ones([n_active], dtype)
    return SparseTensor(sp_indices.indices, values, sp_indices.dense_shape)


def to_dense(sp_tensor):
    if not isinstance(sp_tensor, SparseTensor):
        raise TypeError("Expected sp_indices to be a SparseTensor {} received instead.".format(type(sp_tensor)))

    return sp_ops.sparse_tensor_to_dense(sp_tensor, name="to_dense")


def flat_indices_to_dense(indices, dense_shape):
    """Converts a batch of flat indexes to a dense tensor
    Args:
        indices: a batch of indices (flat indices, each row has the indices up to a given dense_shape[1])
        dense_shape: a `TensorShape` or Tensor with the desired shape for the conversion.

    Returns:
        A tensor (one hot encoding for the given indices on each row)
    """
    indices = ops.convert_to_tensor(indices)
    if indices.dtype != dtypes.int64:
        indices = math_ops.cast(indices, dtypes.int64)

    if isinstance(dense_shape, TensorShape):
        depth = dense_shape.as_list()[1]
    else:
        dense_shape = ops.convert_to_tensor(dense_shape)
        depth = dense_shape[1]

    encoding = array_ops.one_hot(indices, depth=depth)
    one_hot_dense = math_ops.reduce_sum(encoding, axis=1)

    return one_hot_dense


def complete_shape(tensor, shape, dtype=dtypes.int64):
    """ Completes a given shape if not fully defined

    Use Case:
        Sometimes we have networks with an unknown batch size at the time of graph creation,
        this method can be used to complete that same shape based on the first dimension
        of the given tensor and the second dimension of the given shape
    """
    tensor_shape = TensorShape(shape)
    if tensor_shape.is_fully_defined():
        return shape
    else:
        dim = shape.as_list()[1]
        batch_size = array_ops.shape(tensor)[0]
        return math_ops.cast([batch_size, dim], dtype)


def sp_indices_from_sp_values(sp_values):
    """ Returns the a SparseTensor with the indices that go with the given sparse value tensor

    Use Case:
        sometimes we might want to modify a sparse tensor, but to use the new values with a sparse lookup, we need
        new sparse indices

    Args:
        sp_values: a 2-D matrix with the sparse value tensor

    Returns:
        a SparseTensor with the indices that go with the given sparse values tensor

    """
    # new sparse indices after put
    _, flat_indices = array_ops.unstack(sp_values.indices, axis=-1)
    return SparseTensor(sp_values.indices, flat_indices, sp_values.dense_shape)


def flat_to_sparse_indices(indices, dense_shape):
    """Transforms a batch of flat indices to a sparse tensor with the same indices

    Example:
        [[0,1],[1,2]] -> SparseTensor(indices=[[0,0],[0,1],[1,1],[1,2]], values=[0,1,1,2], dense_shape=dense_shape)
    Args:
        indices:
        dense_shape: a list or tensor with the desired dense shape for the flat indices

    Returns:
        a `SparseTensor`
    """
    indices = ops.convert_to_tensor(indices)
    if indices.dtype != dtypes.int64:
        indices = math_ops.cast(indices, dtypes.int64)
    sp_indices = enum_row(indices, dtype=dtypes.int64)

    if dense_shape[0] is None:
        dense_shape = complete_shape(indices, dense_shape, dtypes.int64)
    else:
        dense_shape = ops.convert_to_tensor(dense_shape, dtypes.int64)

    return SparseTensor(sp_indices, array_ops.reshape(indices, shape=[-1]), dense_shape)


def flat_indices_to_sparse_tensor(indices, dense_shape, default_value=1, dtype=dtypes.float32):
    """Transforms a batch of flat indices to a sparse tensor with values set to 1

        Example:
            [[0,1],[1,2]] -> SparseTensor(indices=[[0,0],[0,1],[1,1],[1,2]], values=[1,1,1,1], dense_shape=dense_shape)
        Args:
            indices: a dense ``Tensor`` with the indices to be active for each sample (row)
            dense_shape: a list or tensor with the desired dense shape for the flat indices

        Returns:
            a `SparseTensor`
    """
    indices = ops.convert_to_tensor(indices)
    if indices.dtype != dtypes.int64:
        indices = math_ops.cast(indices, dtypes.int64)
    sp_indices = enum_row(indices, dtype=dtypes.int64)
    if isinstance(dense_shape, TensorShape):
        dense_shape = complete_shape(indices, dense_shape, dtypes.int64)
    else:
        dense_shape = ops.convert_to_tensor(dense_shape, dtypes.int64)

    n_values = array_ops.shape(sp_indices)[0]
    values = array_ops.fill([n_values], default_value, name="default_values")
    if values.dtype != dtype:
        values = math_ops.cast(values, dtype)

    return SparseTensor(sp_indices, values, dense_shape)


def to_sparse(tensor):
    """ Returns a sparse representation for a given tensor

    Example:
        For a dense ``Tensor`` such as::

            tensor = [[1,0],
                      [2,3]]

        this returns an op that creates the following two ``SparseTensor``::

            SparseTensor(indices = [[0,0],[1,0],[1,1]],
                                    values = [1,2,3],
                                    dense_shape = [2,2])

    Args:
        tensor: a dense ``Tensor``

    Returns:
        ``SparseTensor``: a sparse tensor with sparse index and value tensors
        with the non-zero entries of the given input.

    """
    indices = array_ops.where(math_ops.not_equal(tensor, 0))
    dense_shape = array_ops.shape(tensor, out_type=dtypes.int64)

    values = array_ops.gather_nd(tensor, indices)
    sp_tensor = SparseTensor(indices, values, dense_shape)

    return sp_tensor


""" Prepare TensorFlow Inputs
Utilities to prepare inputs for a TensorFlow graph
    e.g. create sparse tensor values, etc
"""


def index_list_to_sparse(indices, dense_shape):
    """
    Converts a list of lists of indexes to a sparse tensor value with the given shape

    Example:

    ..transforms a python list of indices::

        idx =[[0,5],[0,2,7],[1]]

    into a ``SparseTensorValue`` as follows::

        SparseTensorValue(indices=[[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]],
                          values=[0,5,0,2,7,1],
                          dense_shape=[3,10])

    this can be then fed to a ``tf.sparse_placeholder``

    Args:
        indices: python list of int indexes
        dense_shape: a python list or tuple with the shape for the ``SparseTensorValue``, typically ``[BATCH_SIZE, MAX_INDEX]``.

    Raises:
        ``ValueError`` exception if any index ``i`` in the list ``value >= shape[1]``

    Returns:
        ``SparseTensorValue``: a SparseTensorValue with the sparse indices and the indices for each row as values.

    """
    idx = []
    for row, indexes in enumerate(indices):
        for i in indexes:
            if i >= dense_shape[1]:
                raise ValueError("Invalid shape: index value {} >= {}".format(i, dense_shape[1]))
            idx.append([row, i])
    idx = np_array(idx)
    values = np_array(sum(indices, []))

    return SparseTensorValue(indices=idx, values=values, dense_shape=dense_shape)


def value_list_to_sparse(values, sp_indices, dense_shape):
    """ Converts a list of `values` to a sparse tensor value and maps each index in
    `sp_indices` to each value.

    Args:
        values: values to be encapsulated by the sparse tensor value
        sp_indices:indices of the form::

            [[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]]

        dense_shape: shape for resulting ``SparseTensorValue``

    Returns:
        ``SparseTensorValue``: a SparseTensorValue with each index mapping to the given values
    """
    if len(sp_indices) != len(values):
        raise Exception(
            "Number of indices doesn't match number of values: %d != %d".format(len(sp_indices), len(values)))

    return SparseTensorValue(indices=sp_indices, values=values, dense_shape=dense_shape)
