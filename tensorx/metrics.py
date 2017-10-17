""" Metrics.

This module contains metrics or distance functions defining a distance between each pair of elements of a set.

"""
from tensorflow.python.ops import math_ops, array_ops, linalg_ops, sparse_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.math import sparse_l2_norm, batch_sparse_dot, sparse_dot
from tensorx.utils import to_tensor_cast


def pairwise_sparse_cosine_distance(sp_tensor, tensor2, dtype=dtypes.float32, keep_dims=False):
    """ Computes the cosine distance between two non-zero `SparseTensor` and `Tensor`

        Args:
            keep_dims: keeps the original dimension of the input tensor
            sp_tensor: a `SparseTensor`
            tensor2: a `Tensor`
            dim: the dimension along which the distance is computed
            dtype:

        Returns:
            a `Tensor` with the cosine distance between two tensors
        """
    tensor1 = SparseTensor.from_value(sp_tensor)
    if tensor1.values.dtype != dtype:
        tensor1.values = math_ops.cast(tensor1.values, dtype)
    tensor2 = ops.convert_to_tensor(tensor2, dtype)

    dot_prod = batch_sparse_dot(tensor1, tensor2, keep_dims=keep_dims)

    norm1 = sparse_l2_norm(tensor1, axis=-1, keep_dims=True)
    norm2 = linalg_ops.norm(tensor2, axis=-1)

    norm12 = norm1 * norm2
    if keep_dims:
        norm12 = array_ops.expand_dims(norm12, -1)

    cos12 = dot_prod / norm12
    distance = 1 - cos12

    distance = array_ops.where(math_ops.is_nan(distance), array_ops.zeros_like(distance), distance)

    return distance


def pairwise_cosine_distance(tensor1, tensor2, dtype=dtypes.float32, keep_dims=False):
    """ Computes the pairwise cosine distance between two non-zero `Tensor`s

    Args:
        tensor1: a `Tensor`
        tensor2: a `Tensor`
        dim: the dimension along which the distance is computed
        dtype:

    Returns:
        a `Tensor` with the cosine distance between two tensors
    """
    tensor1 = ops.convert_to_tensor(tensor1, dtype)
    tensor2 = ops.convert_to_tensor(tensor2, dtype)
    tensor1 = array_ops.expand_dims(tensor1, 1)
    dot_prod = math_ops.reduce_sum(math_ops.multiply(tensor1, tensor2), -1, keep_dims=keep_dims)

    norm1 = linalg_ops.norm(tensor1, axis=-1)
    norm2 = linalg_ops.norm(tensor2, axis=-1)
    norm12 = norm1 * norm2
    if keep_dims:
        norm12 = array_ops.expand_dims(norm12, -1)

    cos12 = dot_prod / norm12
    distance = 1 - cos12
    distance = array_ops.where(math_ops.is_nan(distance), array_ops.zeros_like(distance), distance)

    return distance


def sparse_cosine_distance(sp_tensor, tensor2, dtype=dtypes.float32):
    """ Computes the cosine distance between two non-zero `SparseTensor` and `Tensor`

        Args:
            keep_dims: keeps the original dimension of the input tensor
            sp_tensor: a `SparseTensor`
            tensor2: a `Tensor`
            dim: the dimension along which the distance is computed
            dtype:

        Returns:
            a `Tensor` with the cosine distance between two tensors
        """
    tensor1 = SparseTensor.from_value(sp_tensor)
    if tensor1.values.dtype != dtype:
        tensor1.values = math_ops.cast(tensor1.values, dtype)
    tensor2 = ops.convert_to_tensor(tensor2, dtype)

    dot_prod = sparse_dot(tensor1, tensor2)
    norm1 = sparse_l2_norm(tensor1, axis=-1)
    norm2 = linalg_ops.norm(tensor2, axis=-1)

    norm12 = norm1 * norm2

    cos12 = dot_prod / norm12
    distance = 1 - cos12

    distance = array_ops.where(math_ops.is_nan(distance), array_ops.zeros_like(distance), distance)
    return distance


def cosine_distance(tensor1, tensor2, dtype=dtypes.float32):
    """ Computes the pairwise cosine distance between two non-zero `Tensor`s

    Args:
        tensor1: a `Tensor`
        tensor2: a `Tensor`
        dim: the dimension along which the distance is computed
        dtype:

    Returns:
        a `Tensor` with the cosine distance between two tensors
    """
    tensor1 = ops.convert_to_tensor(tensor1, dtype)
    tensor2 = ops.convert_to_tensor(tensor2, dtype)

    dot_prod = math_ops.reduce_sum(math_ops.multiply(tensor1, tensor2), -1)

    norm1 = linalg_ops.norm(tensor1, axis=-1)
    norm2 = linalg_ops.norm(tensor2, axis=-1)
    norm12 = norm1 * norm2

    cos12 = dot_prod / norm12
    distance = 1 - cos12
    distance = array_ops.where(math_ops.is_nan(distance), array_ops.zeros_like(distance), distance)

    return distance


def euclidean_distance(tensor1, tensor2, dim):
    """ Computes the euclidean distance between two tensors.

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


def torus_1d_l1_distance(point, size):
    """ Computes the l1 distance between a given point or batch of points and a all points in a 1D torus

    Args:
        point: a rank 0 tensor with a single point or a rank 2 tensor with a batch of points.
        size: the size of the 1d torus

    Returns:
        `Tensor`: a rank 1 or 2 `Tensor` with the distances between each point in the 1D torus and the given points

    Example:

    ..distance for a single point::

        torus_1d_l1_distance(1,4).eval()

    ..or::

        torus_1d_l1_distance([1],4).eval()

        array([ 1.,  0.,  1.,  2.], dtype=float32)

    ..distance for multiple points::

        torus_1d_l1_distance([[2],[3]],4).eval()

        array([[ 2.,  1.,  0.,  1.],
               [ 1.,  2.,  1.,  0.]], dtype=float32)

    """
    point = to_tensor_cast(point, dtypes.float32)
    other = math_ops.range(0, size, 1, dtype=dtypes.float32)

    size = other.get_shape()[-1].value
    return math_ops.minimum(math_ops.abs(point - other), math_ops.mod(-(math_ops.abs(point - other)), size))


__all__ = ["batch_sparse_dot",
           "cosine_distance",
           "sparse_cosine_distance",
           "euclidean_distance",
           "torus_1d_l1_distance",
           "pairwise_cosine_distance",
           "pairwise_sparse_cosine_distance"
           ]
