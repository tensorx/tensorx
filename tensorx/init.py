""" Initialisation Functions

Functions that return weight initialisation tensors for different use cases.
"""

from tensorflow.python.framework import dtypes as dt
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.linalg_ops import qr


def zeros_init(dtype=dt.float32):
    return init_ops.zeros_initializer(dtype=dtype)


def ones_init(dtype=dt.float32):
    return init_ops.ones_initializer(dtype=dtype)


def const_init(value, dtype=dt.float32):
    """ returns an initializer that can be called with a shape to return a tensor value

    Args:
        value: the value to be used in the initializer
        dtype: the initialization value dtype

    Returns:
        an initialization function that takes a shape and returns a tensor filled with the given value

    """
    return init_ops.constant_initializer(value, dtype)


def random_normal(mean=0.0, stddev=1.0, seed=None, dtype=dt.float32):
    return init_ops.random_normal_initializer(mean, stddev, seed, dtype)


def random_uniform(minval=-1, maxval=1, seed=None, dtype=dt.float32):
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


def glorot_uniform(seed=None, dtype=dt.float32):
    """ "glorot uniform initialisation" - Normalised Weight Initialisation [1]

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
    return init_ops.glorot_uniform_initializer(seed, dtype)


def orthogonal_init(gain=1.0, seed=None, dtype=dt.float32):
    """Initializer that generates an orthogonal matrix.

    Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    dtype: The type of the output.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    """

    return init_ops.orthogonal_initializer(gain, seed, dtype)


def he_normal_init(seed=None):
    """ ReLU Weight Initialisation [1].


        Initialisation tensor for weights used as inputs to ReLU activations. Initialises the weights with
        a `Gaussian Distribution`::

            mu: 0
            sigma: sqrt(2/fan_in)

        Liner Neuron Assumption: immediately after initialisation, the parts of tanh and sigm
        that are being explored are close to zero --- the gradient is close to one.
        This doesn't hold for rectifying non-linearities.

        References:
            [1] (He, Rang, Zhen and Sun 2015), "Delving Deep into Rectifiers:Surpassing Human-Level Performance on
            ImageNet Classification".
    """
    return init_ops.he_normal(seed=seed)


__all__ = ["zeros_init",
           "ones_init",
           "random_normal",
           "random_uniform",
           "he_normal_init",
           "glorot_uniform"]
