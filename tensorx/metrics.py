""" Metrics.

This module contains metrics or distance functions defining a distance between each pair of elements of a set.

"""
from tensorflow.python.ops import math_ops, array_ops, linalg_ops, clip_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.math import sparse_l2_norm, batch_sparse_dot, sparse_dot
from tensorx.utils import to_tensor_cast

import tensorx.transform as txf
from tensorflow import sparse

from tensorflow.python.framework.sparse_tensor import convert_to_tensor_or_sparse_tensor


def batch_sparse_cosine_distance(sp_tensor, tensor2, dtype=dtypes.float32, keepdims=False):
    """ Computes the cosine dsitance between two non-zero `SparseTensor` and `Tensor`

        Warning:
            1 - cosine similarity is not a proper distance metric, but any use where only the relative ordering of
            similarity or distance within a set of vectors is important, the resulting order will be unaffected
            by the choice.

        Args:
            keepdims: keeps the original dimension of the input tensor
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

    dot_prod = batch_sparse_dot(tensor1, tensor2, keepdims=keepdims)

    norm1 = sparse_l2_norm(tensor1, axis=-1, keepdims=True)
    norm2 = linalg_ops.norm(tensor2, axis=-1)

    norm12 = norm1 * norm2
    if keepdims:
        norm12 = array_ops.expand_dims(norm12, -1)

    cos12 = dot_prod / norm12

    sim = array_ops.where(math_ops.is_nan(cos12), array_ops.zeros_like(cos12), cos12)
    sim = clip_ops.clip_by_value(sim, -1., 1.)

    return 1 - sim


def batch_cosine_distance(tensor1, tensor2, dtype=dtypes.float32, keepdims=False):
    """ Computes the pairwise cosine similarity between two non-zero `Tensor`s

    Warning:
            1 - cosine similarity is not a proper distance metric, but any use where only the relative ordering of
            similarity or distance within a set of vectors is important, the resulting order will be unaffected
            by the choice.

    Args:

        tensor1: a `Tensor`
        tensor2: a `Tensor`
        keepdims: if true maintains the original dims for tensor1
        dtype: the type for the distance values

    Returns:
        a `Tensor` with the cosine distance between two tensors
    """
    tensor1 = ops.convert_to_tensor(tensor1, dtype)
    tensor2 = ops.convert_to_tensor(tensor2, dtype)
    tensor1 = array_ops.expand_dims(tensor1, 1)
    dot_prod = math_ops.reduce_sum(math_ops.multiply(tensor1, tensor2), -1, keepdims=keepdims)

    norm1 = linalg_ops.norm(tensor1, axis=-1)
    norm2 = linalg_ops.norm(tensor2, axis=-1)
    norm12 = norm1 * norm2

    if keepdims:
        norm12 = array_ops.expand_dims(norm12, -1)

    cos12 = dot_prod / norm12

    sim = array_ops.where(math_ops.is_nan(cos12), array_ops.zeros_like(cos12), cos12)
    sim = clip_ops.clip_by_value(sim, -1., 1.)
    return 1 - sim


def sparse_cosine_distance(sp_tensor, tensor2, dtype=dtypes.float32):
    """ Computes the cosine distance between two non-zero `SparseTensor` and `Tensor`

        Warning:
            1 - cosine similarity is not a proper distance metric, but any use where only the relative ordering of
            similarity or distance within a set of vectors is important, the resulting order will be unaffected
            by the choice.

        Args:
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

    sim = array_ops.where(math_ops.is_nan(cos12), array_ops.zeros_like(cos12), cos12)
    sim = clip_ops.clip_by_value(sim, -1., 1.)
    return 1 - sim


def cosine_distance(tensor1, tensor2, dtype=dtypes.float32):
    """ Computes the pairwise cosine distance between two non-zero `Tensor`s

    Computed on -1 dim

    Computed as 1 - cosine_similarity

    Args:
        tensor1: a `Tensor`
        tensor2: a `Tensor`
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

    sim = array_ops.where(math_ops.is_nan(cos12), array_ops.zeros_like(cos12), cos12)

    # because of floating point accuracy (if we need to correct this to angular distance, acos(1.000001) is nan)
    sim = clip_ops.clip_by_value(sim, -1., 1.)
    return 1 - sim


def euclidean_distance(tensor1, tensor2):
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

    distance = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(tensor1 - tensor2), axis=-1))

    return distance


def sparse_euclidean_distance(sp_tensor, tensor2):
    """ Computes the euclidean distance between two tensors.

        Args:
            tensor1: a ``Tensor`` or ``SparseTensor``
            tensor2: a ``Tensor``
            dim: dimension along which the euclidean distance is computed

        Returns:
            ``Tensor``: a ``Tensor`` with the euclidean distances between the two tensors

        """
    tensor1 = SparseTensor.from_value(sp_tensor)
    if tensor1.values.dtype != dtypes.float32:
        tensor1.values = math_ops.cast(tensor1.values, dtypes.float32)
    tensor2 = ops.convert_to_tensor(tensor2)

    distance = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(tensor1 - tensor2), axis=-1))

    return distance


def pairwise_euclidean_distance(tensor1, tensor2, keepdims=False):
    """ Computes the euclidean distance between two tensors.

        Args:
            tensor1: a ``Tensor``
            tensor2: a ``Tensor``
            keepdims: if True, the result maintains the dimensions of the original result


        Returns:
            ``Tensor``: a ``Tensor`` with the euclidean distances between the two tensors

        """
    tensor1 = ops.convert_to_tensor(tensor1)
    tensor2 = ops.convert_to_tensor(tensor2)
    tensor1 = array_ops.expand_dims(tensor1, 1)

    distance = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(tensor1 - tensor2), axis=-1, keepdims=keepdims))

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

        xys = txf.grid(shape)
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


def batch_manhattan_distance(tensor1, tensor2, keepdims=False):
    """ Compute the manhattan distance between a batch of tensors and a matrix

    If any tensor is a ``SparseTensor``, it is converted to

    Args:
        tensor1: ``Tensor`` or ``SparseTensor``
        tensor2: ``Tensor`` or ``SparseTensor``

    Returns:
        The hamming distance between the two tensors

    """
    tensor1 = convert_to_tensor_or_sparse_tensor(tensor1)
    tensor2 = convert_to_tensor_or_sparse_tensor(tensor2)

    if isinstance(tensor1, SparseTensor):
        tensor1 = sparse.to_dense(tensor1)
    if isinstance(tensor2, SparseTensor):
        tensor2 = sparse.to_dense(tensor2)

    tensor1 = array_ops.expand_dims(tensor1, 1)
    abs_diff = math_ops.abs(math_ops.subtract(tensor1, tensor2))
    return math_ops.reduce_sum(abs_diff, axis=-1, keepdims=keepdims)


__all__ = ["batch_sparse_dot",
           "cosine_distance",
           "sparse_cosine_distance",
           "euclidean_distance",
           "torus_l1_distance",
           "batch_cosine_distance",
           "batch_sparse_cosine_distance"
           ]
