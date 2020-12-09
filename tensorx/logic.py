import tensorflow as tf
from tensorx.utils import as_tensor
from typing import Union, List
from tensorflow import Tensor, SparseTensor


def tensor_equal(first, second):
    """ returns a `bool` tensor with value `True` if tensors are equal,
    `False` otherwise.

    Args:
        first: An object whose type has a registered `Tensor` conversion function.
        second: Another object whose type has a registered `Tensor` conversion function.

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
        diff: SparseTensor = tf.abs(tf.sparse.add(first, second * tf.cast(-1, second.dtype)))
        first = diff.values
        second = tf.zeros_like(diff.values)

    if len(first.shape.as_list()) != len(second.shape.as_list()):
        return tf.constant(False)

    # logic_and evaluates everything so we use cond instead
    return tf.cond(
        tf.reduce_all(tf.equal(tf.shape(first), tf.shape(second))),
        true_fn=lambda: tf.reduce_all(tf.equal(first, second)),
        false_fn=lambda: tf.constant(False)
    )


def same_shape(tensor1, tensor2):
    return tensor_equal(tf.shape(tensor1), tf.shape(tensor2))


def shape_equal(shape1, shape2):
    shape1 = tf.TensorShape(shape1)
    shape2 = tf.TensorShape(shape2)
    return all([d1 == d2 for d1, d2 in zip(shape1, shape2)])


def tensor_close(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):  # pylint: disable=missing-docstring
    rtol_ = as_tensor(rtol)
    atol_ = as_tensor(atol)
    result = (tf.math.abs(a - b) <= atol_ + rtol_ * tf.math.abs(b))
    if equal_nan:
        result = tf.logical_or(result, tf.logical_and(tf.math.is_nan(a), tf.math.is_nan(b)))
    return result


def tensor_all_close(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return tf.reduce_all(tensor_close(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
