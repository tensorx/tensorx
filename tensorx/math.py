import tensorflow as tf
from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul


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


__all__ = ["sparse_multiply_dense"]
