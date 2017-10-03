""" Metrics module

measures different properties of and between tensors
"""
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.transform import l2_normalize


def cosine_distance(tensor1, tensor2, dim):
    """ Computes the cosine distance between two tensors

    Args:
        tensor1: a ``Tensor``
        tensor2: a ``Tensor``
        normalize: if ``True``, normalises (l2) the tensors before computing their distance

    Returns:
        ``Tensor``: a ``Tensor`` with the cosine distances between the two tensors

    """
    if not isinstance(tensor1, (ops.Tensor, SparseTensor)):
        tensor1 = ops.convert_to_tensor(tensor1)
    if not isinstance(tensor1, (ops.Tensor, SparseTensor)):
        tensor2 = ops.convert_to_tensor(tensor2)

    tensor1.get_shape().assert_is_compatible_with(tensor2.get_shape())

    tensor1 = l2_normalize(tensor1, -1)
    tensor2 = l2_normalize(tensor2, -1)

    radial_diffs = math_ops.multiply(tensor1, tensor2)

    distance = 1 - math_ops.reduce_sum(radial_diffs, axis=[dim, ])

    return distance


def euclidean_distance(tensor1, tensor2):
    """ Computes the euclidean distance between two tensors

        Args:
            tensor1: a ``Tensor``
            tensor2: a ``Tensor``

        Returns:
            ``Tensor``: a ``Tensor`` with the euclidean distances between the two tensors

        """
    tensor1 = ops.convert_to_tensor(tensor1)
    tensor2 = ops.convert_to_tensor(tensor2)

    distance = math_ops.sqrt(math_ops.reduce_sum(math_ops.square(tensor1 - tensor2), axis=1))

    return distance
