""" TensorX Math.

Arithmetic operators, linear algebra operators, etc.

"""

from tensorflow.python.ops import math_ops, array_ops, sparse_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.sparse_tensor import SparseTensor, convert_to_tensor_or_sparse_tensor

from tensorx.transform import sparse_overlap


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


def sparse_l2_norm(sp_tensor, axis, name=None, keep_sparse=False, keepdims=False):
    with ops.name_scope(name, "l2_norm", [sp_tensor]) as name:
        square = math_ops.square(sp_tensor)
        if not keep_sparse:
            square_sum = sparse_ops.sparse_reduce_sum(square, axis, keepdims)
        else:
            square_sum = sparse_ops.sparse_reduce_sum_sparse(square, axis, keepdims)
        l2_norm = math_ops.sqrt(square_sum)
        return l2_norm


def batch_sparse_dot(sp_tensor1, tensor2, name=None, keepdims=True):
    """

    Args:
        sp_tensor1: a ``SparseTensor``
        tensor2: a ``Tensor
        name: the name for this op
        keepdims: if true keeps the dimensions of the dot product:
         tensor1.shape[0] x tensor2.shape[0] x tensor2.shape[1]

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with ops.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        dot_prod = sparse_ops.sparse_tensor_dense_matmul(sp_tensor1, array_ops.transpose(tensor2))

        sp_shape = math_ops.cast(sp_tensor1.dense_shape, dtypes.int32)
        dense_shape = array_ops.shape(tensor2)

        if keepdims:
            dot_prod = array_ops.reshape(dot_prod, [sp_shape[0], dense_shape[0], 1])

        return dot_prod


def dot(tensor1, tensor2, name=None):
    return math_ops.reduce_sum(math_ops.multiply(tensor1, tensor2), axis=-1)


def sparse_dot(sp_tensor1, tensor2, name=None):
    """ Returns the dot product between two tensors with the same shape

    Args:
        sp_tensor1: a ``SparseTensor``
        tensor2: a ``Tensor`` or ``SparseTensor``
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with ops.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        if isinstance(tensor2, ops.Tensor):
            # sp_radial_dif = sparse_multiply(sp_tensor1,tensor2)
            dense_values = array_ops.gather_nd(tensor2, sp_tensor1.indices)
            radial_dif = math_ops.multiply(sp_tensor1.values, dense_values)
            sp_radial_dif = SparseTensor(indices=sp_tensor1.indices, values=radial_dif,
                                         dense_shape=sp_tensor1.dense_shape)
            dot_prod = sparse_ops.sparse_reduce_sum(sp_radial_dif, axis=-1)
            return dot_prod
        elif isinstance(tensor2, SparseTensor):
            return sparse_sparse_dot(sp_tensor1, tensor2)
        else:
            raise TypeError(
                "inputs must be of type Tensor or SparseTensor: tensor2 == {t} found".format(t=type(tensor2)))


def sparse_sparse_dot(sp_tensor1, sp_tensor2, name=None):
    """ Returns the dot product between two tensors with the same shape

    Args:
        sp_tensor1: a ``SparseTensor``
        sp_tensor2: a ``SparseTensor``
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with ops.name_scope(name, "sparse_dot", [sp_tensor1, sp_tensor2]):
        # sparse multiply computes the overlap between two sparse tensors
        radial_dif = sparse_sparse_multiply(sp_tensor1, sp_tensor2)
        dot_prod = sparse_ops.sparse_reduce_sum(radial_dif, axis=-1)
        return dot_prod


def sparse_multiply(sp_tensor1, tensor2, name="sparse_multiply"):
    """ Element-wise multiplication of a `Sparse Tensor` by a `Tensor` or a `SparseTensor`

    Args:
        sp_tensor1: a `SparseTensor`
        tensor2: a `Tensor` with the same shape as the sp_tensor.dense_shape

    Returns:
        a `SparseTensor` with the result of the multiplication

    """
    with ops.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        sp_tensor1 = convert_to_tensor_or_sparse_tensor(sp_tensor1)
        assert (isinstance(sp_tensor1, SparseTensor))

        tensor2 = convert_to_tensor_or_sparse_tensor(tensor2)

        if isinstance(tensor2, ops.Tensor):
            dense_values = array_ops.gather_nd(tensor2, sp_tensor1.indices)
            dense_mul = math_ops.multiply(sp_tensor1.values, dense_values)
            result = SparseTensor(sp_tensor1.indices, dense_mul, sp_tensor1.dense_shape)
            result = sparse_ops.sparse_retain(result, math_ops.not_equal(dense_mul, 0.))

            return result
        else:
            return sparse_sparse_multiply(sp_tensor1, tensor2)


def sparse_multiply_dense(sp_tensor1, tensor2, name="sparse_multiply"):
    """ Uses an operation from  Tensorflow that seems faster and supports broadcasting
    but returns a dense result.

    Note:
        also reshapes the result to match the shape of sp_tensor1

    """
    with ops.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        mul = sparse_ops.sparse_dense_cwise_mul(sp_tensor1.indices,
                                                sp_tensor1.values,
                                                sp_tensor1.dense_shape,
                                                tensor2)

        mul = array_ops.reshape(mul, array_ops.shape(sp_tensor1))
        return mul


def sparse_sparse_multiply(sp_tensor1, sp_tensor2):
    """ Element-wise multiplication of two sparse tensors

    Note:
        if the two sparse tensors don't overlap, returns an empty sparse tensor.

    Args:
        sp_tensor1: a `SparseTensor`
        sp_tensor2: a `SparseTensor`

    Returns:
        a `SparseTensor` with the element-wise multiplication of the two sparse tensors

    """
    overlap1 = sparse_overlap(sp_tensor1, sp_tensor2)
    overlap2 = sparse_overlap(sp_tensor2, sp_tensor1)

    values = math_ops.multiply(overlap1.values, overlap2.values)
    return SparseTensor(overlap1.indices, values, overlap1.dense_shape)


def logit(x, dtype=dtypes.float32):
    """
    The logit function is defined as logit(p) = log(p/(1-p)). Note that logit(0) = -inf, logit(1) = inf, and logit(p) for p<0 or p>1 yields nan.

    Args:
        dtype: input tensor dtype
        x: an input Tensor

    Returns:
        A Tensor of the same shape as x. Its entries are logit of the corresponding entry of x.
    """
    x = ops.convert_to_tensor(x, dtype)

    x = math_ops.div(x, 1 - x)
    return math_ops.log(x)


__all__ = ["safe_div",
           "gaussian",
           "sparse_l2_norm",
           "sparse_dot",
           "batch_sparse_dot",
           "sparse_multiply",
           "sparse_multiply_dense",
           "logit"]
