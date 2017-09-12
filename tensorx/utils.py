from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.framework import dtypes


def to_tensor_cast(x, dtype):
    """ Converts to tensor and casts to a given type if possible

    Args:
        x: an input ``Tensor``.
        dtype: the type we which to cast the input tensor into

    Returns:
        ``Tensor``: a tensor with the given dtype
    """
    x = ops.convert_to_tensor(x)
    if x.dtype != dtype:
        x = math_ops.cast(x, dtype)
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
        shape = array_ops.shape(tensor)

    if dtype is not None:
        if dtype != dtypes.int32 and dtype != dtypes.int64:
            raise ValueError("Invalid dtype provided: must be int32 or int64")
        shape = to_tensor_cast(shape, dtype)

    return shape
