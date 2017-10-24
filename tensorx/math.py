""" TensorX Math.

Arithmetic operators, linear algebra operators, etc.

"""

from tensorflow.python.ops import math_ops, array_ops, sparse_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor


def safe_div(numerator, denominator, name="value"):
    """Computes a safe divide which returns 0 if the denominator is zero.
    Note that the function contains an additional conditional check that is
    necessary for avoiding situations where the loss is zero causing NaNs to
    creep into the gradient computation.
    Args:
      numerator: An arbitrary `Tensor`.
      denominator: `Tensor` whose shape matches `numerator` and whose values are
        assumed to be non-negative.
      name: An optional name for the returned op.
    Returns:
      The element-wise value of the numerator divided by the denominator.
    """
    res = math_ops.div(numerator,
                       array_ops.where(math_ops.equal(denominator, 0), array_ops.ones_like(denominator), denominator)),
    res = array_ops.where(math_ops.is_finite(res), res, array_ops.zeros_like(res))
    return res


def gaussian(x, sigma=0.5):
    """ Computes the application of a gaussian function to a given input tensor

    the function is of the form:

    .. math::

        e^(-x^2)/sigma^2

    Args:
        x: an input tensor
        sigma: an input tensor

    Returns:
        a `Tensor` with the result of the operation

    """
    x = ops.convert_to_tensor(x, dtype=dtypes.float32)
    sigma = ops.convert_to_tensor(sigma, dtype=dtypes.float32)
    sigma = array_ops.expand_dims(sigma, -1)

    gauss = math_ops.exp(safe_div(-math_ops.pow(x, 2), math_ops.pow(sigma, 2)))
    gauss = array_ops.squeeze(gauss, 0)
    return gauss


def sparse_l2_norm(sp_tensor, axis, name=None, keep_sparse=False, keep_dims=False):
    with ops.name_scope(name, "l2_norm", [sp_tensor]) as name:
        square = math_ops.square(sp_tensor)
        if not keep_sparse:
            square_sum = sparse_ops.sparse_reduce_sum(square, axis=axis, keep_dims=keep_dims)
        else:
            square_sum = sparse_ops.sparse_reduce_sum_sparse(square, axis=axis, keep_dims=keep_dims)
        l2_norm = math_ops.sqrt(square_sum)
        return l2_norm


def batch_sparse_dot(sp_tensor1, tensor2, name=None, keep_dims=True):
    """

    Args:
        sp_tensor1: a ``SparseTensor``
        tensor2: a ``Tensor
        name: the name for this op
        keep_dims: if true keeps the dimensions of the dot product:
         tensor1.shape[0] x tensor2.shape[0] x tensor2.shape[1]

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with ops.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        dot_prod = sparse_ops.sparse_tensor_dense_matmul(sp_tensor1, array_ops.transpose(tensor2))

        sp_shape = math_ops.cast(sp_tensor1.dense_shape, dtypes.int32)
        dense_shape = array_ops.shape(tensor2)

        if keep_dims:
            dot_prod = array_ops.reshape(dot_prod, [sp_shape[0], dense_shape[0], 1])

        return dot_prod


def sparse_dot(sp_tensor1, tensor2, name=None):
    """ Returns the dot product between two tensors with the same shape

    Args:
        sp_tensor1: a ``SparseTensor``
        tensor2: a ``Tensor
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with ops.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        dense_values = array_ops.gather_nd(tensor2, sp_tensor1.indices)
        radial_dif = math_ops.multiply(sp_tensor1.values, dense_values)
        radial_dif_sp = SparseTensor(indices=sp_tensor1.indices, values=radial_dif, dense_shape=sp_tensor1.dense_shape)
        dot_prod = sparse_ops.sparse_reduce_sum(radial_dif_sp, axis=-1)

        return dot_prod


def sparse_mul(sp_tensor, dense_tensor):
    """ Multiply a `Sparse Tensor` by a `Tensor`.

    Args:
        sp_tensor: a `SparseTensor`
        dense_tensor: a `Tensor` with the same shape as the sp_tensor.dense_shape

    Returns:
        a `SparseTensor` with the result of the multiplication

    """
    dense_values = array_ops.gather_nd(dense_tensor, sp_tensor.indices)
    dense_mul = math_ops.multiply(sp_tensor.values, dense_values)
    return SparseTensor(sp_tensor.indices, dense_mul, sp_tensor.dense_shape)



__all__ = ["safe_div",
           "gaussian",
           "sparse_l2_norm",
           "sparse_dot",
           "batch_sparse_dot",
           "sparse_mul"]
