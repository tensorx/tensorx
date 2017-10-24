""" Metrics.

This module contains metrics or distance functions defining a distance between each pair of elements of a set.

"""
from tensorflow.python.ops import math_ops, array_ops, linalg_ops, sparse_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.math import sparse_l2_norm, batch_sparse_dot, sparse_dot
from tensorx.utils import to_tensor_cast
from tensorx.transform import indices


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


def torus_l1_distance(point, shape):
    """ Computes the l1 distance between a given point or batch of points and a all points in a 1D torus

    Args:
        point: a rank 0 or rank 1 tensor with the coordinates for a point or a rank 2 tensor with a batch of points.
        shape: a :obj:`list` with the shape for the torus - either 1D or 2D

    Returns:
        `Tensor`: a rank 1 or 2 `Tensor` with the distances between each point in the 1D torus and each unique
        coordinate in the shape

    Example:

    ..distance for a single point::

        torus_l1_distance(1,[4]).eval()

    ..or::

        torus_1d_l1_distance([1],[4]).eval()

        array([ 1.,  0.,  1.,  2.], dtype=float32)

    ..distance for multiple points::

        torus_l1_distance([[2],[3]],[4]).eval()

        array([[ 2.,  1.,  0.,  1.],
               [ 1.,  2.,  1.,  0.]], dtype=float32)

    ..distance for 2d torus::

        torus_l1_distance([[0,0]],[2,2]).eval()

        array([[ 0.,  1.,  1.,  2.]], dtype=float32)

    """
    point = to_tensor_cast(point, dtypes.float32)
    if len(shape) == 1:
        max_x = shape[0]
        coor_x = math_ops.range(0, max_x, 1, dtype=dtypes.float32)
        dx = math_ops.abs(point - coor_x)
        distance = math_ops.minimum(dx, math_ops.mod(-dx, max_x))
    elif len(shape) == 2:
        max_x = shape[0]
        max_y = shape[1]

        xys = indices(shape)
        xys = math_ops.cast(xys, dtypes.float32)

        xs, ys = array_ops.unstack(xys, num=2, axis=-1)

        px, py = array_ops.unstack(point, num=2, axis=-1)
        px = array_ops.expand_dims(px, 1)
        py = array_ops.expand_dims(py, 1)

        dx = math_ops.abs(px - xs)
        dy = math_ops.abs(py - ys)

        dx = math_ops.minimum(dx, math_ops.mod(-dx, max_x))

        dy = math_ops.minimum(dy, math_ops.mod(-dy, max_y))

        distance = dx + dy
    else:
        raise ValueError("Invalid shape parameter, shape must have len 1 or 2")

    return distance


__all__ = ["batch_sparse_dot",
           "cosine_distance",
           "sparse_cosine_distance",
           "euclidean_distance",
           "torus_l1_distance",
           "pairwise_cosine_distance",
           "pairwise_sparse_cosine_distance"
           ]
