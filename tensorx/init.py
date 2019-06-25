""" Initialisation Functions

Functions that return weight initialisation tensors for different use cases.
"""

from tensorflow.python.ops.init_ops import zeros_initializer, ones_initializer, constant_initializer, \
    random_normal_initializer, random_uniform_initializer, glorot_uniform_initializer, orthogonal_initializer, \
    he_normal, he_uniform


def zeros_init():
    return zeros_initializer()


def ones_init():
    return ones_initializer()


def const_init(value):
    """ returns an initializer that can be called with a shape to return a tensor value

    Args:
        value: the value to be used in the initializer
        dtype: the initialization value dtype

    Returns:
        an initialization function that takes a shape and returns a tensor filled with the given value

    """
    return constant_initializer(value)


def random_normal(mean=0.0, stddev=1.0, seed=None):
    return random_normal_initializer(mean, stddev, seed)


def random_uniform(minval=-1, maxval=1, seed=None):
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
    return random_uniform_initializer(minval, maxval, seed=seed)


def glorot_uniform(seed=None):
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
    return glorot_uniform_initializer(seed)


def orthogonal_init(gain=1.0, seed=None):
    """Initializer that generates an orthogonal matrix.

    Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    dtype: The type of the output.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    """

    return orthogonal_initializer(gain, seed)


__all__ = ["zeros_init",
           "ones_init",
           "random_normal",
           "random_uniform",
           "he_uniform",
           "he_normal",
           "glorot_uniform"]
