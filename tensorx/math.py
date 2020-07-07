import tensorflow as tf
from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul


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


__all__ = ["sparse_multiply_dense","rms"]
