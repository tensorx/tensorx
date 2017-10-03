""" Metrics module

measures different properties of and between tensors
"""
from tensorflow.python.ops import math_ops, array_ops, linalg_ops, sparse_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.sparse_tensor import convert_to_tensor_or_sparse_tensor

from tensorx.transform import sparse_l2_norm, sparse_dot, l2_normalize


def cosine_distance_v2(tensor1, tensor2, dim, dtype=dtypes.float32):
    sp_tensor = convert_to_tensor_or_sparse_tensor(tensor1, dtype)
    tensor2 = ops.convert_to_tensor(tensor2, dtype)

    if isinstance(tensor1, Tensor):
        return _cosine_distance(sp_tensor, tensor2, dim)

    sp_tensor = l2_normalize(sp_tensor, dim)
    tensor2 = l2_normalize(tensor2, dim)

    distance = 1 - sparse_dot(sp_tensor, tensor2, dim)

    return distance


def _cosine_distance_v2(tensor1, tensor2, dim):
    """ Computes the cosine distance between two tensors

    Args:
        tensor1: a ``Tensor``
        tensor2: a ``Tensor``


    Returns:
        ``Tensor``: a ``Tensor`` with the cosine distances between the two tensors

    """
    tensor1 = ops.convert_to_tensor(tensor1)
    tensor2 = ops.convert_to_tensor(tensor2)
    tensor1.get_shape().assert_is_compatible_with(tensor2.get_shape())

    tensor1 = l2_normalize(tensor1, dim)
    tensor2 = l2_normalize(tensor2, dim)

    dist = 1 - math_ops.reduce_sum(math_ops.multiply(tensor1, tensor2), axis=dim)

    return dist


def cosine_distance(tensor1, tensor2, dim, dtype=dtypes.float32):
    """ Computes the cosine distance between a `Tensor` or `SparseTensor` and a `Tensor`

    Args:
        tensor1 : a ``Tensor`` or ``SparseTensor``
        tensor2: a ``Tensor``
        dim: dimension along which the
        dtype: casts the input tensors to the given type

    """
    sp_tensor = convert_to_tensor_or_sparse_tensor(tensor1, dtype)
    tensor2 = ops.convert_to_tensor(tensor2, dtype)

    if isinstance(tensor1, Tensor):
        return _cosine_distance(sp_tensor, tensor2, dim)

    epsilon = 1e-12
    norm1 = math_ops.maximum(sparse_l2_norm(sp_tensor, dim), epsilon)
    norm2 = math_ops.maximum(linalg_ops.norm(tensor2, axis=dim), epsilon)

    distance = 1 - math_ops.div(sparse_dot(sp_tensor, tensor2, dim), (norm1 * norm2))

    return distance


def _cosine_distance(tensor1, tensor2, dim):
    """ Computes the cosine distance between two tensors

    Args:
        tensor1: a ``Tensor``
        tensor2: a ``Tensor``


    Returns:
        ``Tensor``: a ``Tensor`` with the cosine distances between the two tensors

    """
    tensor1 = ops.convert_to_tensor(tensor1)
    tensor2 = ops.convert_to_tensor(tensor2)
    tensor1.get_shape().assert_is_compatible_with(tensor2.get_shape())

    # because the norm can be zero
    epsilon = 1e-12
    norm1 = math_ops.maximum(linalg_ops.norm(tensor1, axis=dim), epsilon)
    norm2 = math_ops.maximum(linalg_ops.norm(tensor2, axis=dim), epsilon)

    dist = 1 - math_ops.div(math_ops.reduce_sum(math_ops.multiply(tensor1, tensor2), axis=dim), norm1 * norm2)

    return dist


def euclidean_distance(tensor1, tensor2, dim):
    """ Computes the euclidean distance between two tensors

        Args:
            tensor1: a ``Tensor``
            tensor2: a ``Tensor``
            dim: dimension along which the euclidean distance is computed

        Returns:
            ``Tensor``: a ``Tensor`` with the euclidean distances between the two tensors

        """
    tensor1 = ops.convert_to_tensor(tensor1)
    tensor2 = ops.convert_to_tensor(tensor2)

    distance = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(tensor1 - tensor2), axis=dim))

    return distance
