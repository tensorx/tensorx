import tensorflow as tf

from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul


def sparse_multiply_dense(sp_tensor1, tensor2, name="sparse_multiply"):
    """ Uses an operation from  Tensorflow that seems faster and supports broadcasting
    but returns a dense result.

    Note:
        also reshapes the result to match the shape of sp_tensor1

    """
    with tf.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        mul = sparse_dense_cwise_mul(sp_tensor1.indices,
                                     sp_tensor1.values,
                                     sp_tensor1.dense_shape,
                                     tensor2)

        mul = tf.reshape(mul, tf.shape(sp_tensor1))
        return mul


__all__ = ["sparse_multiply_dense"]
