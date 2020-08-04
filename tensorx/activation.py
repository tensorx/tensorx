import tensorflow as tf
import math
from tensorx.utils import as_tensor


def identity(x, name: str = None) -> tf.Tensor:
    """ Identity function

    Returns a tensor with the same content as the input tensor.

    Args:
        x (`Tensor`): The input tensor.
        name (`str`): name for this op

    Returns:
        tensor (`Tensor`): of the same shape, type and content of the input tensor.
    """
    return tf.identity(x, name=name)


def sigmoid(x):
    """ Sigmoid function

    Element-wise sigmoid function, defined as:

    $$
    f(x) = \\frac{1}{1 + \\exp(-x)}
    $$

    Args:
        x (`Tensor`): A tensor or variable.

    Returns:
        A tensor (`Tensor`): with the result of applying the sigmoid function to the input tensor.
    """
    return tf.nn.sigmoid(x)


def hard_sigmoid(x, name="hard_sigmoid"):
    """ Hard Sigmoid

    Segment-wise linear approximation of sigmoid. (Faster than sigmoid)

    !!! note
        Approximates the sigmoid function in 3 parts: 0, scaled linear, 1.

        returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
        In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    Args:
        x (`Tensor`): input tensor
        name (`str`): name for this op

    Returns:
        tensor (`Tensor): the result of applying an approximated element-wise sigmoid to the input tensor

    """
    x = as_tensor(x)
    with tf.name_scope(name):
        slope = as_tensor(0.2, x.dtype)
        shift = as_tensor(0.5, x.dtype)
        x = (slope * x) + shift
        zero = as_tensor(0., x.dtype)
        one = as_tensor(1., x.dtype)
        x = tf.clip_by_value(x, zero, one)

    return x


def tanh(x):
    """ Hyperbolic tangent (tanh) function.

    The element-wise hyperbolic tangent function is essentially a rescaled
    sigmoid function. The sigmoid function with range $[0,1]$ is defined as follows:

    $$
    f(x) = \\frac{1}{1 + \\exp(-x)}
    $$

    the hyperbolic tangent is a re-scaled function such that it's outputs range $[-1,1]$ defined as:
    $$
    tanh(x) = 2f(2x)−1
    $$

    which leads us to the standard definition of hyperbolic tangent

    $$
    tanh(x)=\\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
    $$

    Args:
        x (`Tensor`): an input tensor

    Returns:
        tensor (`Tensor`): a tensor with the result of applying the element-wise hyperbolic tangent to the input

    """
    return tf.nn.tanh(x)


def relu(x):
    """ relu activation

    A Rectifier linear unit [1] is defined as:

    $$
    f(x)= \\max(0, x)
    $$

    !!! cite "References"
        1. (Vinod & Hinton, 2010) [Rectified linear units improve restricted boltzmann machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

    Args:
        x (`Tensor`):  input tensor

    Returns:
        tensor (`Tensor`) that results in element-wise rectifier applied to x.
    """
    return tf.nn.relu(x)


def elu(x, alpha=1.0):
    """ elu activation

    An Exponential Linear Unit (ELU) is defined as:

    $$
    f(x)=\\left\\{\\begin{array}{cc}x & x>0 \\\\
    \\alpha \\cdot \\left(e^{x}-1\\right) & x<=0
    \\end{array}\\right\\}
    $$

    !!! cite "References"
        1. (Clevert et al. 2015) [Fast and accurate deep network learning by exponential linear units (ELUs)](https://arxiv.org/abs/1511.07289).

    Args:
        x (`Tensor`): an input tensor
        alpha (`float`): A scalar, slope of positive section.

    Returns:
        tensor (`Tensor`): resulting from the application of the elu activation to the input tensor.
    """
    y = tf.nn.elu(x)
    if alpha == 1:
        return y
    else:
        return tf.where(x > 0, y, x * y)


def gelu(x, approximate: bool = True) -> tf.Tensor:
    """ Gaussian Error Linear Unit.

    Computes gaussian error linear:
        `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))` or
        `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, where P(X) ~ N(0, 1),
        depending on whether approximation is enabled.

    !!! cite "References"
        1. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
        2. [BERT](https://arxiv.org/abs/1810.04805).
    Args:
        x (`Tensor`):  Must be one of the following types:
            `float16`, `float32`, `float64`.
        approximate (bool): whether to enable approximation.

    Returns:
        tensor (`Tensor`): with the same type as `x`
    """
    x = tf.convert_to_tensor(x)
    if approximate:
        pi = tf.cast(tf.constant(math.pi), x.dtype)
        coefficient = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coefficient * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))


def selu(x):
    """ The Scaled Exponential Linear Unit (SELU)

    `scale * x` if `x > 0` and `scale * alpha * (exp(x) - 1)` if `x < 0`
    where alpha and scale are pre-defined constants (`alpha = 1.67326324` and `scale = 1.05070098`).

    The values of alpha and scale are chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized correctly
    (see `variance_scaling_init`).

    To be used together with initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN').
    For correct dropout, use tf.contrib.nn.alpha_dropout.

    !!! cite "References"
        1. [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf)

    Args:
        x (`Tensor`): input tensor

    Returns:
        tensor (`Tensor`): results in `scale * x` if `x > 0` and `scale * alpha * (exp(x) - 1)` if `x < 0`
    """
    return tf.nn.selu(x)


def softmax(x, axis=None, name=None):
    """ softmax activation

    Softmax activation function, is equivalent to `softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)`
    and it is defined as:

    $$
    \\sigma(\\mathbf{z})_{i}=\\frac{e^{z_{i}}}{\\sum_{j=1}^{K} e^{z_{j}}}
    $$

    Args:
        x (`Tensor`): input tensor
        axis (`int`): the dimension softmax would be performed on. The default is -1 which indicates the last dimension.
        name (`str`): name for this op

    Returns:
        tensor (`Tensor`): output resulting from the application of the softmax function to the input tensor

    """
    return tf.nn.softmax(x, axis=axis, name=name)


def sparsemax(logits, name: str = None) -> tf.Tensor:
    """Computes the sparsemax activation function [1]

    For each batch `i` and class `j` we have
      sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)

    References:
        [1]: https://arxiv.org/abs/1602.02068

    Args:
        logits (`Tensor`): tensor with dtype: `half`, `float32`,`float64`.
        name (`str`): A name for the operation (optional).

    Returns:
        tensor (`Tensor`): with the same type as the input logits.
    """

    with tf.name_scope(name, "sparsemax"):
        logits = tf.convert_to_tensor(logits, name="logits")
        obs = tf.shape(logits)[0]
        dims = tf.shape(logits)[1]

        z = logits - tf.reduce_mean(logits, axis=1)[:, tf.newaxis]

        # sort z
        z_sorted, _ = tf.nn.top_k(z, k=dims)

        # calculate k(z)
        z_cumsum = tf.cumsum(z_sorted, axis=1)
        k = tf.range(
            1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = tf.reduce_sum(tf.cast(z_check, tf.int32), axis=1)

        # calculate tau(z)
        indices = tf.stack([tf.range(0, obs), k_z - 1], axis=1)
        tau_sum = tf.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

        # calculate p
        return tf.maximum(
            tf.cast(0, logits.dtype), z - tau_z[:, tf.newaxis])


__all__ = [
    "identity",
    "sigmoid",
    "hard_sigmoid",
    "tanh",
    "relu",
    "selu",
    "elu",
    "gelu",
    "softmax",
    "sparsemax"
]
