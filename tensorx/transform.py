""" TensorFlow tensor Transformation
Utilities to convert between and combine tensors
"""

import tensorflow as tf
import numpy as np


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
    update_shape = tf.shape(sp_updates.values)
    sp_zeros = tf.SparseTensor(sp_updates.indices,
                               tf.zeros(update_shape, dtype=tf.float32),
                               sp_updates.dense_shape)
    concat_indices = tf.sparse_add(sp_tensor, sp_zeros)

    # shape of resulting values tensor
    value_shape = tf.shape(concat_indices.values)

    # 2 get mask for input tensor
    ones_values = tf.ones(value_shape, dtype=tf.float32)
    sp_ones = tf.SparseTensor(concat_indices.indices,
                              ones_values,
                              concat_indices.dense_shape)
    mask_ones = tf.scalar_mul(-1, tf.ones(update_shape))
    sp_mask = tf.SparseTensor(sp_updates.indices, mask_ones, sp_updates.dense_shape)

    to_retain = tf.sparse_add(sp_ones, sp_mask)
    to_retain = tf.not_equal(to_retain.values, 0)

    # get tensor with masked values
    tensor_masked = tf.sparse_retain(concat_indices, to_retain)

    # add values to entries previously set to 0
    return tf.sparse_add(tensor_masked, sp_updates)



def dense_put(tensor,sp_updates):
    """ Dense Put

    Changes a given tensor according to the updates specified in a SparseTensor

    Args:
        tensor: a dense tensor we want to change
        sp_updates: a sparse tensor with the indices to be changed and the respective values

    The resulting tensor will have the same values as the input tensor, except for the indices
    overlapping with update tensor which will be getting the updates.
    """
    dense_values = tf.sparse_tensor_to_dense(sp_updates)
    return tf.where(tf.not_equal(dense_values,0),dense_values,tensor)


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


    indices = tf.where(tf.not_equal(tensor, 0))
    dense_shape = tf.shape(tensor, out_type=tf.int64)

    # Sparse Tensor for sp_indices
    flat_layer = tf.reshape(tensor, [-1])
    values = tf.mod(tf.squeeze(tf.where(tf.not_equal(flat_layer, 0))), dense_shape[1])

    sp_indices = tf.SparseTensor(indices, values, dense_shape)

    # Sparse Tensor for values
    values = tf.gather_nd(tensor, indices)
    sp_values = tf.SparseTensor(indices, values, dense_shape)

    return (sp_indices, sp_values)


def pairs(tensor1, tensor2):
    """
    Returns a tensor resulting from the pairwise combination of the elements of each tensor

    t1 = [0,1]
    t2 = [2,3,4]
    pairs(t1,t2) == [[[0,2],[0,3],[0,4]],[[1,2],[1,3],[1,4]],...]
    """
    return tf.squeeze(tf.stack(tf.meshgrid(tensor1, tensor2), axis=-1))


def enum_row(tensor):
    """
    Converts
    :param tensor:
    :return:
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.shape(tensor)

    # for each coordinate
    row_i = tf.range(0, shape[0])
    enum = tf.map_fn(lambda i: pairs(i, tensor[i]), elems=row_i)
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
    idx = np.array(idx)
    values = np.array(sum(indices, []))

    return tf.SparseTensorValue(indices=idx, values=values, dense_shape=shape)


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
        raise Exception("Number of indices doesn't match number of values: " + len(sp_indices) + "!=" + len(values))

    return tf.SparseTensorValue(indices=sp_indices, values=values, dense_shape=shape)
