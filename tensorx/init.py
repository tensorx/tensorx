""" Initialisation Functions

Functions that return weight initialisation tensors for different use cases.
"""

from tensorflow.python.framework import dtypes as dt
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import init_ops


def zero_init(dtype=dt.float32):
    return init_ops.zeros_initializer(dtype=dtype)


def ones_init(dtype=dt.float32):
    return init_ops.ones_initializer(dtype=dtype)


def random_normal(mean=0.0, stdev=1.0, seed=None, dtype=dt.float32):
    return init_ops.random_normal_initializer(mean, stdev, seed, dtype)


def random_uniform(maxval=1, minval=-1, seed=None, dtype=dt.float32):
    """ Random Uniform Initialisation between minval and maxval.

    Wrapper around TensorFlow random_uniform function between minval and maxval

    Args:
        seed: seed for random number generator
        minval: minimum value for random uniform distribution values
        maxval: maximum value for random uniform distribution values
        dtype: TensorFlow data type

    Returns:
        ``Tensor``: a TensorFlow ``Tensor`` used to initialise a ``Variable``.
    """
    return init_ops.random_uniform_initializer(minval, maxval, seed=seed, dtype=dtype)


def xavier_init(seed=None, dtype=dt.float32):
    """ "Xavier Initialisation" - Normalised Weight Initialisation [1]

    This initialisation keeps the scale of the gradients roughly the same in all layers to
    mitigate `vanishing` and `exploding gradients` see [1].


    References:
        [1] (Glorot and Bengio 2010), "Understanding the difficulty of training deep
        feedforward neural networks".

    Args:
        seed: seed for random number generator
        dtype: TensorFlow data type

    Returns:
        Tensor: a TensorFlow tensor used to initialise variable
    """

    # [fan_in, fan_out] = shape
    # low = -math_ops.sqrt(6.0 / (fan_in + fan_out))
    # high = math_ops.sqrt(6.0 / (fan_in + fan_out))

    # return random_ops.random_uniform((fan_in, fan_out), low, high, dtype)
    return init_ops.glorot_uniform_initializer(seed, dtype)


class HeNormal(init_ops.Initializer):
    """ ReLU Weight Initialisation [1].


        Initialisation tensor for weights used as inputs to ReLU activations. Initialises the weights with
        a `Gaussian Distribution`::

            mu: 0
            sigma: sqrt(2/fan_in)

        Liner Neuron Assumption: immediately after initialisation, the parts of tanh and sigm
        that are being explored are close to zero --- the gradient is close to one.
        This doesn't hold for rectifying non-linearities.

        References:
            [1] (He, Rang, Zhen and Sun 2015), "Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet
            Classification".

        Args:

            dtype: TensorFlow dtype
    """

    def __init__(self, dtype=dt.float32):
        self.dtype = dt.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        [fan_in, fan_out] = shape
        mu = 0
        sigma = math_ops.sqrt(2.0 / fan_in)
        return random_ops.random_normal((fan_in, fan_out), mu, sigma, dtype)

    def get_config(self):
        return {"dtype": self.dtype.name}


he_normal_init = HeNormal

__all__ = ["zero_init", "ones_init", "random_normal", "random_uniform", "he_normal_init", "xavier_init"]
