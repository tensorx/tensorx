import tensorflow as tf
from tensorflow.python.framework.tensor_util import constant_value

static_value = constant_value


def to_tensor_cast(x, dtype=None):
    """ Converts to tensor and casts to a given type if possible

    Args:
        x: an input ``Tensor``.
        dtype: the type we which to cast the input tensor into

    Returns:
        ``Tensor``: a tensor with the given dtype
    """
    x = tf.convert_to_tensor_or_sparse_tensor(x)

    if dtype is not None:
        if x.dtype != dtype:
            x = tf.cast(x, dtype)
    return x


def complete_shape(tensor, dtype=None):
    """ Returns the complete shape of a tensor if not fully defined. If
    dtype is given, casts the shape to that dtype.

    Note: dtype can only be int32 or int64. int64 shapes are needed to create a ``SparseTensor`` .

    Args:
        tensor: a ``Tensor`` whose shape we wish to know
        dtype: the expected output shape type

    Returns:
        ``Tensor``: a ``Tensor`` with the complete shape of a tiven ``Tensor``.
    """
    shape = tensor.get_shape()
    if shape.is_fully_defined():
        shape = shape.as_list()
    else:
        shape = tf.shape(tensor)

    if dtype is not None:
        if dtype != tf.int32 and dtype != tf.int64:
            raise ValueError("Invalid dtype provided: must be int32 or int64")
        shape = to_tensor_cast(shape, dtype)

    return shape


def as_list(elems):
    """ Returns a list from one or multiple elements.

    if one element is passed, returns a list with one element,
    if a list or tuple of elements is passed, returns a list with the elements

    Note: we exclude SparseTensorValue because it is a named tuple
    and we want to feed the whole object as a single data sample if needed

    Args:
        elems: one element, a tuple of elements or a list of elements

    Returns:
        a :obj:`list` with the elements in elems
    """
    if elems is None:
        elems = []
    elif isinstance(elems, (list, tuple)) and not isinstance(elems, (
            tf.SparseTensorValue, tf.SparseTensor)):
        elems = list(elems)
    else:
        elems = [elems]
    return elems


class Graph:
    """ Simple append only graph"""

    def __init__(self):
        self.nodes = set()
        self.edges_in = dict()
        self.edges_out = dict()

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges_in[node] = []
            self.edges_out[node] = []

    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        self.edges_out[node1].append(node2)
        self.edges_in[node2].append(node1)


__all__ = ["constant_value",
           "Graph",
           "as_list",
           "complete_shape",
           "to_tensor_cast"]
