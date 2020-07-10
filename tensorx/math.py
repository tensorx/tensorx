import tensorflow as tf
from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul
from tensorx.transform import sparse_ones, sparse_zeros


def rms(x):
    """ Root mean square (RMS)

    Also known as quadratic mean is defined as:

    $x_{\\mathrm{RMS}}=\\sqrt{\\frac{x_{1}^{2}+x_{2}^{2}+\\ldots+x_{n}^{2}}{n}}$

    In estimation theory, the root-mean-square deviation of an estimator is a measure of the imperfection of the fit of
    the estimator to the data.

    Args:
        x (`Tensor`): input tensor

    Returns:
        result (`Tensor`): scalar tensor with the result of applying the root mean square to the input tensor

    """
    return tf.sqrt(tf.reduce_mean(tf.square(x)))


def sparse_multiply_dense(sp_tensor, dense_tensor, name="sparse_multiply_dense"):
    """ sparse_multiply_dense

    !!! info
        Uses `sparse_dense_cwise_mul` from Tensorflow but returns a dense result
        and reshapes the result to match the shape of `sp_tensor`

    Args:
        sp_tensor (SparseTensor): a sparse tensor
        dense_tensor (Tensor): a dense tensor
        name (str): op name

    Returns:
        A dense tensor (Tensor): the result for the multiplication between the sparse and dense tensors
      

    """
    with tf.name_scope(name):
        mul = sparse_dense_cwise_mul(sp_tensor.indices,
                                     sp_tensor.values,
                                     sp_tensor.dense_shape,
                                     dense_tensor)

        mul = tf.reshape(mul, tf.shape(sp_tensor))
        return mul


def sparse_sparse_dot(sp_tensor1, sp_tensor2, name="sparse_sparse_dot"):
    """ Returns the dot product between two tensors with the same shape

    Args:
        sp_tensor1: a ``SparseTensor``
        sp_tensor2: a ``SparseTensor``
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with tf.name_scope(name):
        # sparse multiply computes the overlap between two sparse tensors
        radial_dif = sparse_sparse_multiply(sp_tensor1, sp_tensor2)
        dot_prod = tf.sparse.reduce_sum(radial_dif, axis=-1)
        return dot_prod


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
    with tf.name_scope(name):
        dot_prod = tf.sparse.sparse_dense_matmul(sp_tensor1, tensor2, adjoint_b=True)

        sp_shape = tf.cast(sp_tensor1.dense_shape, tf.int32)
        dense_shape = tf.shape(tensor2)

        if keepdims:
            dot_prod = tf.reshape(dot_prod, [sp_shape[0], dense_shape[0], 1])

        return dot_prod


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

    values = tf.math.multiply(overlap1.values, overlap2.values)
    return tf.SparseTensor(overlap1.indices, values, overlap1.dense_shape)


def sparse_overlap(sp_tensor1, sp_tensor2, name="sparse_overlap"):
    """ Returns a `SparseTensor` where the indices of the two tensors overlap returning a ``SparseTensor``
    with the values of the first one

    Args:
        name: name for this op
        sp_tensor1: a `SparseTensor`
        sp_tensor2: a `SparseTensor`

    Returns:
        `SparseTensor`, `SparseTensor`: sp1, sp2 - sparse tensors with the overlapping indices
    """
    with tf.name_scope(name):
        ones1 = sparse_ones(sp_tensor1.indices, sp_tensor1.dense_shape)
        ones2 = sparse_ones(sp_tensor2.indices, sp_tensor2.dense_shape)

        index_union = tf.sparse_add(ones1, ones2)

        index_filter = tf.math.equal(index_union.values, 2.)

        zeros1 = sparse_zeros(index_union.indices, index_union.dense_shape, sp_tensor1.values.dtype)
        expand1 = tf.sparse_add(zeros1, sp_tensor1)

        filtered = tf.sparse_retain(expand1, index_filter)
        return filtered


def sparse_dot(sp_tensor1, tensor2, name=None):
    """ Returns the dot product between two tensors with the same shape

    Args:
        sp_tensor1: a ``SparseTensor``
        tensor2: a ``Tensor`` or ``SparseTensor``
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with tf.name_scope(name):
        if isinstance(tensor2, tf.Tensor):
            dense_values = tf.gather_nd(tensor2, sp_tensor1.indices)
            radial_dif = tf.math.multiply(sp_tensor1.values, dense_values)
            sp_radial_dif = tf.SparseTensor(indices=sp_tensor1.indices, values=radial_dif,
                                            dense_shape=sp_tensor1.dense_shape)
            dot_prod = tf.sparse.reduce_sum(sp_radial_dif, axis=-1)
            return dot_prod
        elif isinstance(tensor2, tf.SparseTensor):
            return sparse_sparse_dot(sp_tensor1, tensor2)
        else:
            raise TypeError(
                "inputs must be of type Tensor or SparseTensor: tensor2 == {t} found".format(t=type(tensor2)))


def sparse_l2_norm(sp_tensor, axis=-1, keep_sparse=False, keepdims=False, name="sparse_l2_norm"):
    with tf.name_scope(name):
        square = tf.math.square(sp_tensor)
        square_sum = tf.sparse.reduce_sum(square, axis=axis, output_is_sparse=keep_sparse, keepdims=keepdims)
        l2_norm = tf.math.sqrt(square_sum)
        return l2_norm


__all__ = [
    "sparse_multiply_dense",
    "sparse_dot",
    "sparse_sparse_multiply",
    "batch_sparse_dot",
    "sparse_l2_norm",
    "rms"
]
