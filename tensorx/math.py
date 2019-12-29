import tensorflow as tf

from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul

from tensorflow.python.ops.variables import PartitionedVariable
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

# TODO do I need this, can't I use range?
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.ops import data_flow_ops


def sparse_multiply_dense(sp_tensor1, tensor2, name="sparse_multiply"):
    """ sparse_multiply_dense

    Uses an operation from  Tensorflow that seems faster and supports broadcasting
    but returns a dense result.

    Note:
        also reshapes the result to match the shape of sp_tensor1

    Args:
      sp_tensor1 (SparseTensor): a sparse tensor
      tensor2 (Tensor): a dense tensor
      name (str): op name

    Returns:
      A dense Tensor with resulting from the multiplication of the sparse by 
      the dense tensor
      

    """
    with tf.name_scope(name):
        mul = sparse_dense_cwise_mul(sp_tensor1.indices,
                                     sp_tensor1.values,
                                     sp_tensor1.dense_shape,
                                     tensor2)

        mul = tf.reshape(mul, tf.shape(sp_tensor1))
        return mul


def _clip(params, ids, max_norm):
    """Helper function for _embedding_lookup_and_transform.
    This function optionally clips embeddings to an l2-norm of max_norm.
    Args:
      params: A `Tensor` of embeddings retrieved by `gather`.
      ids: The `ids` argument that was passed to `gather`.
      max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
        than this value.
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


def embedding_lookup_sparse(params,
                            sp_tensor,
                            partition_strategy="mod",
                            name=None,
                            combiner=None,
                            max_norm=None):
    """Computes embeddings for the given ids and weights.

    This op assumes that there is at least one id for each row in the dense tensor
    represented by sp_ids (i.e. there are no rows with empty features), and that
    all the indices of sp_ids are in canonical row-major order.
    It also assumes that all id values lie in the range [0, p0), where p0
    is the sum of the size of params along dimension 0.

    None:
      The difference between this and tensorflow's implementation is that in the original 
      implementation the sparse gradients do not propagate through gather. 
      This was included to reduce the number of ids gathered from remote parameter servers
      in a distributed setting. But it defeats the purpose of using sparse Tensors for
      large dense shapes. 

    Args:
      params: A single tensor representing the complete embedding tensor, or a
        list of P tensors all of same shape except for the first dimension,
        representing sharded embedding tensors.  Alternatively, a
        `PartitionedVariable`, created by partitioning along dimension 0. Each
        element must be appropriately sized for the given `partition_strategy`.
      sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
        and M is arbitrary.
      sp_weights: either a `SparseTensor` of float / double weights, or `None` to
        indicate all weights should be taken to be 1. If specified, `sp_weights`
        must have exactly the same shape and indices as `sp_ids`.
      partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
        is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: Optional name for the op.
      combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
        and "sum" are supported. "sum" computes the weighted sum of the embedding
        results for each row. "mean" is the weighted sum divided by the total
        weight. "sqrtn" is the weighted sum divided by the square root of the sum
        of the squares of the weights.
      max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
        than this value, before combining.
    Returns:
      A dense tensor representing the combined embeddings for the
      sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
      looks up the embeddings for all ids in that row, multiplies them by the
      corresponding weight, and combines these embeddings as specified.
      In other words, if
        `shape(combined params) = [p0, p1, ..., pm]`
      and
        `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`
      then
        `shape(output) = [d0, d1, ..., dn-1, p1, ..., pm]`.
      For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are
        ```python
        [0, 0]: id 1, weight 2.0
        [0, 1]: id 3, weight 0.5
        [1, 0]: id 0, weight 1.0
        [2, 3]: id 1, weight 3.0
        ```
      with `combiner`="mean", then the output will be a 3x20 matrix where
        ```python
        output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
        output[1, :] = (params[0, :] * 1.0) / 1.0
        output[2, :] = (params[1, :] * 3.0) / 3.0
        ```
    Raises:
      TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
        neither `None` nor `SparseTensor`.
      ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
    """
    if combiner is None:
        logging.warn("The default value of combiner will change from \"mean\" "
                     "to \"sqrtn\" after 2016/11/01.")
        combiner = "mean"
    if combiner not in ("mean", "sqrtn", "sum"):
        raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
    if isinstance(params, PartitionedVariable):
        params = list(params)  # Iterate to get the underlying Variables.
    if not isinstance(params, list):
        params = [params]
    if not isinstance(sp_tensor, tf.SparseTensor):
        raise TypeError("sp_ids must be SparseTensor")

    with ops.name_scope(name, "embedding_lookup_sparse",
                        params + [sp_tensor]) as name:
        segment_ids = sp_tensor.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        ids = sp_tensor.indices[:, -1]
        # ids, idx = tf.unique(ids)

        embeddings = tf.nn.embedding_lookup(
            params=params,
            ids=ids,
            max_norm=max_norm)

        # ***
        # this second lookup causes problems because sparse gradients don't propagate though gather
        # embeddings = embedding_lookup(embeddings, idx)
        # embeddings, _ = gather_dynamic(embeddings, idx)
        # ***

        weights = sp_tensor.values
        if weights.dtype != embeddings.dtype:
            weights = tf.cast(weights, embeddings.dtype)

        # Reshape weights to allow broadcast
        ones = tf.fill(
            tf.expand_dims(tf.rank(embeddings) - 1, 0), 1)
        bcast_weights_shape = tf.concat(
            [tf.shape(weights), ones], 0)

        orig_weights_shape = weights.get_shape()
        weights = tf.reshape(weights, bcast_weights_shape)

        # Set the weight shape, since after reshaping to bcast_weights_shape,
        # the shape becomes None.
        if embeddings.get_shape().ndims is not None:
            weights.set_shape(orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

        embeddings *= weights

        if combiner == "sum":
            embeddings = tf.math.segment_sum(embeddings, segment_ids, name=name)
        elif combiner == "mean":
            embeddings = tf.math.segment_sum(embeddings, segment_ids)
            weight_sum = tf.math.segment_sum(weights, segment_ids)
            embeddings = tf.math.divide_no_nan(embeddings, weight_sum, name=name)
        elif combiner == "sqrtn":
            embeddings = tf.math.segment_sum(embeddings, segment_ids)
            weights_squared = tf.math.pow(weights, 2)
            weight_sum = tf.math.segment_sum(weights_squared, segment_ids)
            weight_sum_sqrt = tf.math.sqrt(weight_sum)
            embeddings = tf.math.divide_no_nan(embeddings, weight_sum_sqrt, name=name)
        else:
            assert False, "Unrecognized combiner"

        return embeddings


__all__ = ["sparse_multiply_dense", "embedding_lookup_sparse"]
