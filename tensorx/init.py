"""
    Initializers allow you to pre-specify an initialization strategy, encoded in
    the Initializer object, without knowing the shape and dtype of the variable
    being initialized.

    !!! Note
        Despite most initializers being accessible through Tensorflow, we decided to gather
        them here for the sake of namespace consistency (not all TF initializers are in tf.initializers)
"""
import tensorflow as tf
from typing import Callable, Optional


def zeros_init():
    """ Zeroes Initializer

    Initializer that generates tensors initialized to 0.

    Returns:
        initializer (Callable): an initializer that returns a tensor filled with 0 when called on a given shape.

    """
    return tf.zeros_initializer()


def ones_init():
    """ Ones Initializer

    Initializer that generates tensors initialized to 1.

    Returns:
        initializer (Callable): an initializer that returns a tensor filled with 1 when called on a given shape.

    """
    return tf.ones_initializer()


def constant_init(value=0):
    """ Constant Initializer

    The resulting tensor is populated with values of type dtype, as specified by arguments value
    following the desired shape.

    The argument value can be a constant value, or a list of values of type dtype. If value is a list, then the length
    of the list must be less than or equal to the number of elements implied by the desired shape of the tensor.
    In the case where the total number of elements in value is less than the number of elements required by the tensor
    shape, the last element in value will be used to fill the remaining entries. If the total number of elements in
    value is greater than the number of elements required by the tensor shape, the initializer will raise a ValueError.

    Args:
        value: A Python scalar, list or tuple of values, or a N-dimensional numpy array. All elements of
        the initialized variable will be set to the corresponding value in the value argument.

    Returns:
        initializer (Callable): an initializer that returns a tensor from the given specification and a given shape
    """
    return tf.constant_initializer(value)


def uniform_init(minval: float = -0.05, maxval: float = 0.05, seed=None):
    """ Random Uniform Initializer

    Initializer that generates tensors with a uniform distribution.

    Args:
        minval:  Lower bound of the range of random values to generate.
        maxval: Upper bound of the range of random values to generate. Defaults to 1 for float types.
        seed (int32/int64): seed for random number generator

    Returns:
         initializer (Callable): an initializer that returns a tensor from the given specification and a given shape

    """
    return tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=seed)


def normal_init(mean: float = 0.0, stddev=0.05, seed=None):
    """ Random Normal Initializer

    Initializer that generates tensors with a normal distribution.

    Args:
        mean: Mean of the random values to generate.
        stddev: Standard deviation of the random values to generate.
        seed (int32/int64): seed for random number generator

    Returns:
         initializer (Callable): an initializer that returns a tensor from the given specification and a given shape

    """
    return tf.random_normal_initializer(mean=mean, stddev=stddev, seed=seed)


def glorot_uniform_init(seed: Optional = None) -> Callable:
    """ Glorot Uniform Initializer

    This initialisation keeps the scale of the gradients roughly the same in all layers to
    mitigate `vanishing` and `exploding gradients` see [1].
    
    References:
        [1] (Glorot and Bengio 2010), "Understanding the difficulty of training deep
        feedforward neural networks".
    
    Args:
        seed (int32/int64): seed for random number generator
    
    Returns:
        initializer (Callable): callable that creates an initial value from a given shape
    """
    return tf.initializers.glorot_uniform(seed)


def glorot_normal_init(seed: Optional = None) -> Callable:
    """ Glorot Normal Initializer

    This initialisation keeps the scale of the gradients roughly the same in all layers to
    mitigate `vanishing` and `exploding gradients` see [1].

    Draws samples from a truncated normal distribution.

    References:
        [1] (Glorot and Bengio 2010), "Understanding the difficulty of training deep
        feedforward neural networks".

    Args:
        seed (int32/int64): seed for random number generator

    Returns:
        initializer (Callable): callable that creates an initial value from a given shape
    """
    return tf.initializers.glorot_normal(seed)


def orthogonal_init(gain: float = 1.0, seed=None) -> Callable:
    """ Orthogonal initializer

    If the shape of the tensor to initialize is two-dimensional, it is initialized
    with an orthogonal matrix obtained from the QR decomposition of a matrix of
    random numbers drawn from a normal distribution.

    If the matrix has fewer rows than columns then the output will have orthogonal
    rows. Otherwise, the output will have orthogonal columns.

    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.

    !!! cite "References"
        1. [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://openreview.net/forum?id=_wzZwKpTDF_9C)

    Args:
        gain (float): multiplicative factor to apply to the orthogonal matrix
        seed (int32/int64): seed for random number generator

    Returns:
        initializer (Callable): callable that creates an orthogonal matrix from a given shape
    """

    return tf.initializers.orthogonal(gain=gain, seed=seed)


def identity_init(gain: float = 1.0):
    """ Identity Initializer

    creates an identity matrix for a 2D shape

    Args:
        gain (float): multiplicative factor to be applied to the identity matrix

    Returns:
        initializer (Callable): callable that creates an identity matrix from a given 2D shape
    """
    return tf.initializers.identity(gain=gain)


def he_uniform_init(seed=None):
    """ He Uniform Initializer

    also known as `MSRA` initialization

    It draws samples from a uniform distribution within $[-l, l]$ where $l = \\sqrt{\\frac{6}{fan_{in}}}$ where
    $fan_{in}$ is the number of input units in the weight tensor.

    !!! Cite "References"
        1. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

    Args:
        seed (int32/int64): seed for random number generator

    Returns:
        initializer (Callable): callable that returns a tensor value from a given shape

    """
    return tf.initializers.he_uniform(seed=seed)


def he_normal_init(seed=None):
    """ He Normal Initializer

    also known as `MSRA` initialization

    It draws samples from a truncated normal distribution centered on $0$ with
    $stddev = \\sqrt{\\frac{2}{fan_{in}}} where $fan_{in}$ is the number of input units in the weight tensor.


    !!! Cite "References"
        1. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

    Args:
        seed (int32/int64): seed for random number generator

    Returns:
        initializer (Callable): callable that returns a tensor value from a given shape

    """
    return tf.initializers.he_normal(seed=seed)


def variance_scaling_init(scale=1.0, mode="fan_in", uniform=False, seed=None):
    """ variance scaling init

    A generalization of `glorot_uniform_init` and `he_uniform_init`.
    The principle behind this is that it's good to to keep the scale of the input variance constant,
    so it does not explode or diminish by reaching the final layer.

    ```
    if mode='fan_in': # Count only number of input connections.
        n = fan_in
    elif mode='fan_out': # Count only number of output connections.
        n = fan_out
    elif mode='fan_avg': # Average number of inputs and output connections.
        n = (fan_in + fan_out)/2.0

    truncated_normal(shape, 0.0, stddev=sqrt(scale / n))
    ```

    !!! note "Example Configurations"
        * `factor=2.0, mode='FAN_IN', uniform=False`: MSRA / He normal initialization
        * `factor=1.0 mode='FAN_AVG' uniform=True/False`: glorot / xavier initialization

    Args:
        mode (`str`): "fan_in", "fan_out" or "fan_avg".
        scale (`float`): positive scaling factor
        uniform (`bool`): if True samples from uniform distribution, else from the truncated normal
        seed (int32/int64): seed for random number generator

    Returns:
        initializer (Callable): callable that returns a tensor value from a given shape

    """
    return tf.initializers.VarianceScaling(scale=scale, mode=mode,
                                           distribution="uniform" if uniform else "truncated_normal", seed=seed)
