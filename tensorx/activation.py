""" Activation Functions

To be used by :py:class:`~tensorx.layers.Activation` or with any other :py:class:`~tensorflow.Tensor`.
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn import top_k
from tensorflow.python.ops import nn
from tensorflow.python.ops import clip_ops

from tensorx.utils import to_tensor_cast


def identity(x, name=None):
    """ Returns a tensor with the same content as the input tensor.

    Args:
        x: The input tensor.

    Returns:
        A tensor of the same shape, type and content of the input tensor.
    """
    return array_ops.identity(x, name=name)


def sigmoid(x):
    """Element-wise sigmoid.

        Args
            x: A tensor or variable.

        Returns:
            A tensor.
    """
    return nn.sigmoid(x)


def hard_sigmoid(x, name="hard_sigmoid"):
    """Segment-wise linear approximation of sigmoid.
    Faster than sigmoid.

    Note: Approximates in 3 parts: 0, scaled linear, 1.

    returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
    In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    Args:
        name: name for this op
        x: A tensor or variable.

    Returns:
        a float32 tensor resulting in an approximated element-wise sigmoid applied to x

    """
    x = ops.convert_to_tensor(x)
    with ops.name_scope(name):
        slope = to_tensor_cast(0.2, x.dtype)
        shift = to_tensor_cast(0.5, x.dtype)
        x = (slope * x) + shift
        zero = to_tensor_cast(0., x.dtype)
        one = to_tensor_cast(1., x.dtype)
        x = clip_ops.clip_by_value(x, zero, one)

    return x


def tanh(x):
    """Element-wise hyperbolic tangent (tanh).

    Args:
        x: A tensor or variable.


    Returns: A tensor.

    """
    return nn.tanh(x)


def relu(x):
    """ Rectifier linear unit [1].

    References:
        [1] (Vinod & Hinton, 2010) "Rectified linear units improve restricted boltzmann machines."

    Args:
        x (Tensor):  Tensor or variable to compute the activation function for.
    Returns:
        ´´Tensor´´: A tensor that results in element-wise rectifier applied to x.
    """
    return nn.relu(x)


def leaky_relu(x, alpha=0.2):
    """ Leaky Rectifier Linear Unit

    Args:
        x: (Tensor):  Tensor or variable to which the ReLU will be applied
        alpha: Slope of the activation function at x < 0.

    Returns:
        ´´Tensor´´: A tensor that results in element-wise rectifier applied to x.
    """
    return nn.leaky_relu(x, alpha)


def elu(x, alpha=1.0):
    """Exponential Linear Unit

    Reference:

        (Clevert et al. 2015) Fast and accurate deep network learning by exponential linear units (ELUs).

    Args:
        x: A tenor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.

    Returns:
        A tensor
    """
    y = nn.elu(x)
    if alpha == 1:
        return y
    else:
        return array_ops.where(x > 0, y, x * y)


def softmax(x):
    """Softmax activation function

    applies the softmax function to the input tensor last dimension

    Args:
        x: a 2D Tensor of variable

    Returns:
        a 2D tensor whose ijth element is computed from the softmax function

    """
    return nn.softmax(x)


def sparsemax(logits, name=None):
    """Computes the sparsemax activation function [1]

    For each batch `i` and class `j` we have
      sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)

    References:
        [1]: https://arxiv.org/abs/1602.02068

    Args:
      logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
        `float64`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `logits`.
    """

    with ops.name_scope(name, "sparsemax", [logits]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        obs = array_ops.shape(logits)[0]
        dims = array_ops.shape(logits)[1]

        z = logits - math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

        # sort z
        z_sorted, _ = top_k(z, k=dims)

        # calculate k(z)
        z_cumsum = math_ops.cumsum(z_sorted, axis=1)
        k = math_ops.range(
            1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

        # calculate tau(z)
        indices = array_ops.stack([math_ops.range(0, obs), k_z - 1], axis=1)
        tau_sum = array_ops.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

        # calculate p
        return math_ops.maximum(
            math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])


__all__ = ["relu",
           "leaky_relu",
           "sigmoid",
           "hard_sigmoid",
           "tanh",
           "elu",
           "identity",
           "softmax",
           "sparsemax"]
