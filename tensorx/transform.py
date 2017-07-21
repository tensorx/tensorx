""" TensorFlow tensor Transformation
Utilities to convert between and combine tensors
"""

import tensorflow as tf
import numpy as np


def sparse_mask(tensor, mask):
    pass


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
