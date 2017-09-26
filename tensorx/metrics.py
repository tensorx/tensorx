""" Metrics module

measures different properties of and between tensors
"""
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn import l2_normalize
from tensorflow.python.framework import ops


def cosine_distance(tensor1, tensor2, normalize=True):
    """ Computes the cosine distance between two tensors

    Args:
        tensor1: a ``Tensor``
        tensor2: a ``Tensor``
        normalize: if ``True``, normalises (l2) the tensors before computing their distance

    Returns:
        ``Tensor``: a ``Tensor`` with the cosine distances between the two tensors

    """
    tensor1 = ops.convert_to_tensor(tensor1)
    tensor2 = ops.convert_to_tensor(tensor2)

    if normalize:
        tensor1 = l2_normalize(tensor1, 0)
        tensor2 = l2_normalize(tensor2, 0)

    distance = 1 - math_ops.reduce_sum(math_ops.multiply(tensor1, tensor2), axis=1)
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
