import tensorflow as tf
from tensorx.utils import as_tensor
from typing import Union
from tensorflow import Tensor, SparseTensor


def tensor_equal(first, second):
    """ returns a `bool` tensor with value `True` if tensors are equal,
    `False` otherwise.

    Args:
        first (`Union[Tensor, SparseTensor]`): a tensor
        second (`Union[Tensor, SparseTensor]`): second tensor

    Returns:
        tensor (`Tensor`): a tensor with dtype `tf.bool`

    """
    first: Union[Tensor, SparseTensor] = as_tensor(first)
    second: Union[Tensor, SparseTensor] = as_tensor(second)
    if first.dtype != second.dtype:
        return tf.constant(False)
    if type(first) != type(second):
        if isinstance(first, SparseTensor):
            first = tf.sparse.to_dense(first)
        elif isinstance(second, SparseTensor):
            second = tf.sparse.to_dense(second)
    elif isinstance(first, SparseTensor):
        diff: SparseTensor = tf.abs(tf.sparse.add(first, second * -1))
        return tf.reduce_all(tf.equal(diff.values, tf.zeros_like(diff.values)))

    shapes_equal = tf.reduce_all(tf.equal(tf.shape(first), tf.shape(second)))
    return tf.logical_and(shapes_equal,
                          tf.reduce_all(tf.equal(first, second)))
