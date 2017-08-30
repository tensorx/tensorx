from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def to_tensor_cast(x, dtype):
    """ Converts to tensor and casts to a given type if possible

    Args:
        x: an input Tensor.
        dtype: the type we which to cast the input tensor into

    Returns:
        a tensor of type dtype
    """
    x = ops.convert_to_tensor(x)
    if x.dtype != dtype:
        x = math_ops.cast(x, dtype)
    return x
