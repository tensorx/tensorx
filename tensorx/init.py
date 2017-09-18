""" Initialisation Functions

Functions that return weight initialisation tensors for different use cases.
"""

from tensorflow.python.framework import dtypes as dt
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import init_ops


def zero_init():
    return init_ops.zeros_initializer()


def random_uniform(shape, maxval=1, minval=-1, dtype=dt.float32):
    """ Random Uniform Initialisation between minval and maxval.

    Wrapper around TensorFlow random_uniform function between minval and maxval

    Args:
        minval: minimum value for random uniform distribution values
        maxval: maximum value for random uniform distribution values
        shape: shape of the tensor to be generated
        dtype: TensorFlow data type

    Returns:
        ``Tensor``: a TensorFlow ``Tensor`` used to initialise a ``Variable``.
    """
    return random_ops.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype)


def xavier_init(shape, dtype=dt.float32):
    """ "Xavier Initialisation" - Normalised Weight Initialisation [1]

    This initialisation keeps the scale of the gradients roughly the same in all layers to
    mitigate `vanishing` and `exploding gradients` see [1].


    References:
        [1] (Glorot and Bengio 2010), "Understanding the difficulty of training deep
        feedforward neural networks".

    Args:
        shape: [fan_in, fan_out]
        dtype: TensorFlow data type

    Returns:
        Tensor: a TensorFlow tensor used to initialise variable
    """
    [fan_in, fan_out] = shape
    low = -math_ops.sqrt(6.0 / (fan_in + fan_out))
    high = math_ops.sqrt(6.0 / (fan_in + fan_out))

    return random_ops.random_uniform((fan_in, fan_out), low, high, dtype)


def relu_init(shape, dtype=dt.float32):
    """ ReLU Weight Initialisation [1].


    Initialisation tensor for weights used as inputs to ReLU activations. Initialises the weights with
    a `Gaussian Distribution`::

        mu: 0
        sigma: sqrt(2/fan_in)

    Liner Neuron Assumption: immediately after initialisation, the parts of tanh and sigm
    that are being explored are close to zero --- the gradient is close to one.
    This doesn't hold for rectifying non-linearities.

    References:
        [1] (He, Rang, Zhen and Sun 2015), "Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification".

    Args:
        shape: [fan_in, fan_out]
        dtype: TensorFlow dtype

    Returns:
        Tensor: a TensorFlow tensor used to initialise variable
    """
    [fan_in, fan_out] = shape
    mu = 0
    sigma = math_ops.sqrt(2.0 / fan_in)
    return random_ops.random_normal((fan_in, fan_out), mu, sigma, dtype)
