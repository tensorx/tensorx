""" Initialisation Functions

Provides functions that return weight initialisation tensors for different use cases.

"""

import tensorflow as tf


def random_uniform(shape, dtype=tf.float32):
    """ Random Uniform Initialisation

    Wrapper around TensorFlow random_uniform function between -1 and 1

    Args:
        shape: shape of the tensor to be generated
        dtype: TensorFlow data type

    Returns:
        Tensor: a TensorFlow tensor used to initialise variable
    """
    return tf.random_uniform(shape, minval=-1, maxval=1, dtype=dtype)


def xavier_init(shape, dtype=tf.float32):
    """ "Xavier Initialisation" - Normalised Weight Initialisation

    This initialisation keeps the scale of the gradients roughly the same in all layers.

    The idea is to try and mitigate:
                vanishing gradient: the gradient decreases exponentially
                    (multiplication throughout each layer) and the front
                    layers train very slowly. (Affects deep networks and
                    recurrent networks --more multiplications);

                exploring gradient: when we use activation functions
                    whose derivative can take larger values,gradients
                    can grow exponentially.

    Reference:
        (Glorot and Bengio 2010),
        "Understanding the difficulty of training deep feedforward neural networks".

    Args:
        shape: [fan_in, fan_out]
        dtype: TensorFlow data type

    Returns:
        Tensor: a TensorFlow tensor used to initialise variable
    """
    [fan_in, fan_out] = shape
    # TODO needs testing: not sure if tf.sqrt works here or if I need to use np.sqrt as the original
    low = -tf.sqrt(6.0 / (fan_in + fan_out))
    high = tf.sqrt(6.0 / (fan_in + fan_out))

    return tf.random_uniform((fan_in, fan_out), low, high, dtype)


def relu_init(shape, dtype=tf.float32):
    """ ReLU Weight Initialisation

    Initialiser tensor for weights to be used as inputs to ReLU activations. Initialises the weights with
    a Gaussian distribution:
        mu: 0
        sigma: sqrt(2/fan_in)

    Liner Neuron Assumption: immediately after initialisation, the parts of tanh and sigm
    that are being explored are close to zero --- the gradient is close to one.
    This doesn't hold for rectifying non-linearities.

    Reference:
        (He, Rang, Zhen and Sun 2015),
        "Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification".

    Use Case:
        With deep networks using ReLU and PReLU activations.

    Args:
        shape: [fan_in, fan_out]
        dtype: TensorFlow data type

    Returns:
        Tensor: a TensorFlow tensor used to initialise variable
    """
    [fan_in, fan_out] = shape
    mu = 0
    # TODO needs testing: not sure if tf.sqrt works here or if I need to use np.sqrt as the original
    sigma = tf.sqrt(2.0 / fan_in)
    return tf.random_normal((fan_in, fan_out), mu, sigma, dtype)
