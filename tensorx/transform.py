""" TensorFlow tensor Transformation

Utilities to create, convert between, and combine tensors
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.framework.sparse_tensor import SparseTensor, SparseTensorValue, \
    convert_to_tensor_or_sparse_tensor
from tensorflow.python.ops.nn import dropout

import numpy as np

from tensorx.utils import to_tensor_cast, complete_shape


def empty_sparse_tensor(dense_shape, dtype=dtypes.float32, name="empty_sp_tensor"):
    """ Creates an empty SparseTensor.

    Note:
        ``shape = [10]``

        ``empty = tf.sparse_tensor_to_dense(transform.empty_sparse_tensor(shape))``

        is equivalent to:

        ``zero = tf.zeros(shape)``

       meaning that:

       ``tf.reduce_all(tf.equal(zeros,empty)).eval()``

       returns True


    Args:
        dense_shape: a 1-D tensor, python list, or numpy array with the output shape for the sparse tensor
        dtype: the dtype of the values for the empty SparseTensor
        name: a name for this operation (optional)

    Returns:
        ``SparseTensor``: an empty sparse tensor with a given shape

    """
    with ops.name_scope(name):
        dense_shape = ops.convert_to_tensor(dense_shape, name="dense_shape", dtype=dtypes.int64)

        index_shape = dense_shape.get_shape().with_rank(1)
        empty_indices = array_ops.ones([0, index_shape[0]], dtype=dtypes.int64)
        empty_values = array_ops.ones([0], dtype=dtype)

        return SparseTensor(empty_indices, empty_values, dense_shape)


def pairs(tensor1, tensor2, name="pairs"):
    """Pairwise combination of elements from the two tensors.

    Example::

        t1 = [[0],[1]]
        t2 = [2,3,4]
        t12 = [[0,2],[1,2],[0,3],[1,3],[0,4],[1,4]]

        tf.reduce_all(tf.equal(pairs(t1,t2),t12))

    Args:
        tensor1(``Tensor``): a tensor, python list, or numpy array
        tensor2(``Tensor``): a tensor, python list, or numpy array
        name: name for this operation (optional)

    Returns:
        ``Tensor``: a ``Tensor`` of rank 2
    """
    with ops.name_scope(name):
        tensor1 = ops.convert_to_tensor(tensor1)
        tensor2 = ops.convert_to_tensor(tensor2)

        x, y = array_ops.meshgrid(tensor1, tensor2)

        result = array_ops.stack([x, y], axis=-1)
        result = array_ops.reshape(result, [-1, 2])
        return result


def batch_to_matrix_indices(tensor, name="batch_to_matrix", dtype=dtypes.int32):
    """ Converts a batch of indices to a 2-D tensor of matrix indices.

    For a given batch of indices of shape [n,m] this op outputs a 2-D ``Tensor``
    with the `row-index pairs`::

        [[r1,i1],[r1,i2],...,[r1,im],
         [r2,i1],[r2,i2],...,[r2,im],
         ...
         [rn,i1],[rn,i2],...,[rn,im]]


    Example::

            tensor = [[1,2],
                      [2,5]]


            batch_to_matrix(tensor)

            [[0,1],
             [0,2],
             [1,2],
             [1,5]]

    Use Case:
        Convert a batch of indices (used to slice another tensor with embedding lookup or gather)
        to be used in a SparseTensor, so that we can change the weights of each slice.

    Args:
        dtype: int32 or int64, the output tensor type
        name: name for batch_to_matrix_indices op
        tensor: the tensor to be converted

        Returns:
            ``Tensor``: a 2-D tensor with (row index,value) for each index in the input tensor.

    """
    with ops.name_scope(name):
        tensor = to_tensor_cast(tensor, dtype)
        if tensor.dtype != dtypes.int32 and tensor.dtype != dtypes.int64:
            raise TypeError("Invalid tensor type: expected {t1} or {t2}, found {t3}".format(t1=dtypes.int32,
                                                                                            t2=dtypes.int64,
                                                                                            t3=tensor.dtype))
        shape = tensor.get_shape().with_rank(2)
        if shape.is_fully_defined():
            shape = shape.as_list()
        else:
            shape = array_ops.shape(tensor)

        rows = math_ops.range(math_ops.cast(shape[0], tensor.dtype))
        rows = array_ops.expand_dims(rows, 1)

        multiples = array_ops.stack([1, shape[1]])
        rows = array_ops.tile(rows, multiples)

        enum = array_ops.stack([rows, tensor], axis=-1)
        enum = array_ops.reshape(enum, shape=[-1, 2])
        return enum


def sparse_put(sp_tensor, sp_updates, name="sparse_put"):
    """Changes a given SparseTensor according to the updates specified in a SparseTensor.

    Creates a new tensor where the values of the updates override the
    values in the original tensor. The input tensors must have the same
    ``dense_shape``.

    Args:
        sp_tensor: a ``SparseTensor`` we which to set some indices to given values
        sp_updates: a ``SparseTensor`` with the indices to be changed and the respective values
        name: name for this operation (optional)

    Returns:
        ``SparseTensor``: a ``SparseTensor`` with the updated values.
    """
    with ops.name_scope(name):
        if sp_updates.dtype != sp_tensor.dtype:
            sp_updates = math_ops.cast(sp_updates, sp_tensor.dtype)

        # 1 concat indices and establish final tensor shape
        update_shape = sp_updates.values.get_shape()
        zero_updates = SparseTensor(sp_updates.indices,
                                    array_ops.zeros(update_shape, dtype=dtypes.float32),
                                    sp_updates.dense_shape)
        proto_result = sparse_ops.sparse_add(sp_tensor, zero_updates)

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

        to_retain = sparse_ops.sparse_add(proto_ones, sp_mask)
        to_retain = math_ops.not_equal(to_retain.values, 0)

        # get tensor with masked values
        tensor_masked = sparse_ops.sparse_retain(proto_result, to_retain)

        # add values to entries previously set to 0
        new_tensor = sparse_ops.sparse_add(tensor_masked, sp_updates)
        return new_tensor


def dense_put(tensor, sp_updates, name="dense_put"):
    """ Changes a given dense ``Tensor`` according to the updates specified in a ``SparseTensor``.

    Creates a new ``Tensor`` where the values of the updates override the
    values in the original tensor. The tensor `shape` must be the same as the updates `dense_shape`.

    Args:
        tensor: a ``Tensor`` we want to change.
        sp_updates: a ``SparseTensor`` with the indices to be changed and the respective values.
        name: the name for this operation (optional).

    Returns:
        ``Tensor``: a ``Tensor`` with the updated values.
    """
    with ops.name_scope(name):
        tensor = ops.convert_to_tensor(tensor)
        if sp_updates.dtype != tensor.dtype:
            sp_updates = math_ops.cast(sp_updates, tensor.dtype)

        markers = array_ops.ones(shape=array_ops.shape(sp_updates.values))
        sparse_marker_tensor = SparseTensor(indices=sp_updates.indices, values=markers,
                                            dense_shape=sp_updates.dense_shape)
        dense_update_marker = sparse_ops.sparse_tensor_to_dense(sparse_marker_tensor)
        dense_updates = sparse_ops.sparse_tensor_to_dense(sp_updates)

        new_tensor = array_ops.where(math_ops.not_equal(dense_update_marker, 0),
                                     dense_updates, tensor)
        return new_tensor


def sparse_dropout(sp_tensor, keep_prob=0.2, seed=None, name="sparse_dropout"):
    """Performs a dropout on a ``SparseTensor``.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    Args:
        sp_tensor: a ``SparseTensor`` on which the dropout is performed.
        keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
        seed: A Python integer. Used to create random seeds. (See `TensorFlow` documentation
        for ``tf.set_random_seed`` for behavior.)
        name: A name for this operation (optional).

    """
    with ops.name_scope(name):
        dense_shape = sp_tensor.dense_shape

        drop_values = dropout(sp_tensor.values, keep_prob, seed=seed)
        not_zero = math_ops.not_equal(drop_values, 0)

        # new indices (after dropout)
        # not_zero_indices = array_ops.where(not_zero)
        # indices = array_ops.gather_nd(sp_indices.indices, not_zero_indices)

        values = array_ops.boolean_mask(drop_values, not_zero)
        indices = array_ops.boolean_mask(sp_tensor.indices, not_zero)

        new_tensor = SparseTensor(indices, values, dense_shape)
        return new_tensor


def sparse_ones(indices, dense_shape, dtype=dtypes.float32):
    """ Creates a new ``SparseTensor`` where the values are 1

    Args:
        indices: a 2-D ``Tensor`` with the indices for the resulting sparse tensor
        dense_shape: the ``SparseTensor`` dense shape
        dtype: the tensor type for the values

    Returns:
        ``SparseTensor``: a new SparseTensor with the values set to 1.
    """
    indices = to_tensor_cast(indices, dtypes.int64)
    dense_shape = to_tensor_cast(dense_shape, dtypes.int64)
    indices_shape = complete_shape(indices)
    values = array_ops.ones([indices_shape[0]], dtype)
    return SparseTensor(indices, values, dense_shape)


def sparse_one_hot(indices, dense_shape, dtype=dtypes.float32):
    """Transforms a batch of indices to a one-hot encoding ``SparseTensor``.

        Example::

            indices = [[0,1,4],
                       [1,2,6]]

            dense_shape = [2,10]

            sp_one_hot = sparse_one_hot(indices,dense_shape)

            expected = SparseTensor(indices=[[0,0],[0,1],[0,4],[1,1],[1,2],[1,6]],
                                    values=[1,1,1,1,1,1],
                                    dense_shape=[2,10])

        Args:
            indices: a dense ``Tensor`` with the indices to be active for each sample (row)
            dense_shape: a list, array or `Tensor` with the shape for output dense one_hot encoding tensor.
            dtype: the type for the output values.

        Returns:
            `SparseTensor`: a ``Sparse Tensor`` with the one hot encoding for the given indices
    """
    flat_indices = to_tensor_cast(indices, dtypes.int64)
    indices = batch_to_matrix_indices(flat_indices, dtype=dtypes.int64)

    return sparse_ones(indices, dense_shape, dtype)


def dense_one_hot(indices, dense_shape, dtype=dtypes.float32):
    """Transforms a batch of indices to a dense ``Tensor`` by adding the `one-hot` encoding for each index.

    Example::

        indices = [[0],[1]]
        dense_shape = [2,2]

        dense_one_hot = [[1,0],[0,1]]

    Args:
        indices: a dense ``Tensor`` with the active indices for each sample (row).
        dense_shape: a list, array or `Tensor` with the shape for the output dense one_hot encoding tensor.
        dtype: the type for the output tensor.

    Returns:
        ``Tensor``: A dense ``Tensor`` with a `one-hot encoding` for the given indices.
    """
    indices = to_tensor_cast(indices, dtypes.int64)
    dense_shape = ops.convert_to_tensor(dense_shape)

    encoding = array_ops.one_hot(indices, depth=dense_shape[1], dtype=dtype)
    one_hot_dense = math_ops.reduce_sum(encoding, axis=1)

    return one_hot_dense


def sp_indices_from_sp_tensor(sp_values):
    """ Returns the a ``SparseTensor`` with the indices for the active values on a given ``SparseTensor`` .

    Use Case:
        To be used with ``embedding_lookup_sparse`` when we need two ``SparseTensor`` : one with the indices and
        one with the values.

    Args:
        sp_values: a ``SparseTensor`` for which we extract the active indices.

    Returns:
        ``SparseTensor``: a ``SparseTensor`` with the indices of the active elements of another ``SparseTensor`` .
    """
    _, flat_indices = array_ops.unstack(sp_values.indices, num=2, axis=-1)
    return SparseTensor(sp_values.indices, flat_indices, sp_values.dense_shape)


def to_sparse(tensor, name="to_sparse"):
    """ Returns a ``SparseTensor` for a`given dense ``Tensor``.

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
        name: name for this operation (optional)

    Returns:
        ``SparseTensor``: a sparse tensor with sparse index and value tensors
        with the non-zero entries of the given input.

    """
    with ops.name_scope(name):
        indices = array_ops.where(math_ops.not_equal(tensor, 0))
        dense_shape = array_ops.shape(tensor, out_type=dtypes.int64)

        values = array_ops.gather_nd(tensor, indices)
        sp_tensor = SparseTensor(indices, values, dense_shape)

    return sp_tensor


def sparse_tensor_value_one_hot(indices, dense_shape):
    """ Converts a python or numpy array of indices to a ``SparseTensorValue``.

    Example:

    ..transforms a python list of indices::

        idx =[[0,5],[0,2,7],[1]]

    into a ``SparseTensorValue`` as follows::

        SparseTensorValue(indices=[[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]],
                          values=[1,1,1,1,1,1],
                          dense_shape=[3,10])

    this can be then fed to a ``tf.sparse_placeholder``

    Args:
        indices: a 2-D list or array of indices
        dense_shape: a python list or tuple with the shape for the ``SparseTensorValue``,
                     typically ``[BATCH_SIZE, MAX_INDEX]``.

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
    idx = np.array(idx)
    values = np.ones([len(idx)])

    return SparseTensorValue(indices=idx, values=values, dense_shape=dense_shape)


def sparse_l2_norm(sp_tensor, axis, name=None, keep_sparse=False):
    with ops.name_scope(name, "l2_norm", [sp_tensor]) as name:
        square = math_ops.square(sp_tensor)
        if not keep_sparse:
            square_sum = sparse_ops.sparse_reduce_sum(square, axis=axis)
        else:
            square_sum = sparse_ops.sparse_reduce_sum_sparse(square, axis=axis, keep_dims=True)
        l2_norm = math_ops.sqrt(square_sum)
        return l2_norm


def sparse_dot(sp_tensor1, tensor2, dim, name=None):
    """

    Args:
        sp_tensor1: a ``SparseTensor``
        tensor2: a ``Tensor
        dim: the dimension along which the dot product is computed
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with ops.name_scope(name, "l2_norm", [sp_tensor1, tensor2]):
        dense_values = array_ops.gather_nd(tensor2, sp_tensor1.indices)
        radial_dif = math_ops.multiply(sp_tensor1.values, dense_values)
        radial_dif_sp = SparseTensor(indices=sp_tensor1.indices, values=radial_dif, dense_shape=sp_tensor1.dense_shape)
        dot_prod = sparse_ops.sparse_reduce_sum(radial_dif_sp, axis=dim)

        return dot_prod


__all__ = ["empty_sparse_tensor",
           "to_sparse",
           "batch_to_matrix_indices",
           "dense_one_hot",
           "sparse_one_hot",
           "dense_put",
           "sparse_put",
           "sparse_tensor_value_one_hot",
           "sp_indices_from_sp_tensor",
           "sparse_ones",
           "sparse_dropout",
           "pairs",
           "sparse_dot",
           "sparse_l2_norm"]
