import tensorflow as tf

from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul

from tensorflow.python.framework import ops


# TODO do I need this, can't I use range?


def sparse_multiply_dense(sp_tensor, dense_tensor, name="sparse_multiply_dense"):
    """ sparse_multiply_dense

    !!! info
        Uses an `sparse_dense_cwise_mul` from Tensorflow but returns a dense result
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


def _clip(params, ids, max_norm):
    """Helper function for _embedding_lookup_and_transform.
    This function optionally clips embeddings to an l2-norm of max_norm.

    Args:
        params: A `Tensor` of embeddings retrieved by `gather`.

        ids: The `ids` argument that was passed to `gather`.

        max_norm: If not `None`, each embedding is clipped if its l2-norm is larger than this value.
    Returns:
        A `Tensor` with the same type as `params`.
    """

    def _rank(x):
        """Helper function to retrieve the rank of a tensor.
        Args:
            x: Something convertible to `Tensor`.
        Returns:
            Either a pair `(rank, True)` where `rank` is an integer or a pair
            `(rank, False)` where `rank` is an integer `Tensor`. In either case,
            `rank` is the rank of `x`.
        """
        rank = ops.convert_to_tensor(x).get_shape().ndims
        if rank:
            return rank, True
        else:
            return tf.rank(x), False

    if max_norm is None:
        return params
    ids_rank, ids_static = _rank(ids)
    params_rank, params_static = _rank(params)
    return tf.clip_by_norm(
        params,
        max_norm,
        axes=(list(range(ids_rank, params_rank)) if ids_static and params_static
              else tf.range(ids_rank, params_rank)))


__all__ = ["sparse_multiply_dense"]
