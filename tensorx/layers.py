""" Neural Network Layers.

All layers contain a certain number of units, its shape, name and a tensor member
which gives us a handle for a TensorFlow tensor that can be evaluated.

Types of layers:
    inputs: wrap around TensorFlow placeholders.

    dense:  a layer encapsulating a dense matrix of weights,
            possibly including biases and an activation function.

    sparse: a dense matrix of weights accessed through a list of indexes,
            (e.g. by being connected to an IndexInput layer)

"""

import tensorflow as tf


class Layer:
    def __init__(self, n_units, shape, dtype=tf.float32, name="layer"):
        self.n_units = n_units
        self.name = name
        self.tensor = None
        self.dtype = dtype

        if shape is None:
            self.shape = [None, n_units]
        elif shape[1] != n_units:
            raise Exception("Shape must match [,n_units], was " + shape)

    def tensor(self):
        return self.tensor


class Input(Layer):
    """ Input Layer:
    creates placeholders to receive tensors with a given shape
    [batch_size, n_units] and a given data type
    """
    def __init__(self, n_unsts, batch_size=None, dtype=tf.float32, name="input"):
        shape = [batch_size, n_units]
        super().__init__(n_units, shape, dtype, name)
        self.tensor = tf.placeholder(self.dtype, self.shape, self.name)


class IndexInput(Layer):
    """

    """
    def __init__(self, n_units, n_active, batch_size=None, name="index_input"):
        shape = [batch_size, n_active]
        super().__init__(n_units,shape,tf.int32,name)

        self.n_active = n_active
        self.tensor = tf.placeholder(self.dtype,self.shape,self.name)

    def to_dense(self):
        """Converts the output tensor
        to a dense s with n_units
        """
        return tf.one_hot(self.tensor, self.n_units)

