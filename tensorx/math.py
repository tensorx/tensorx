""" TensorX Math.

Arithmetic operators, linear algebra operators, etc.

"""
import tensorflow as tf
import logging
from tensorx.transform import sparse_overlap
from tensorflow.python.ops.variables import PartitionedVariable
from tensorflow.python.ops.sparse_ops import sparse_dense_cwise_mul


def safe_div(numerator, denominator, name=None):
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
    with tf.name_scope(name, "safe_div", [numerator, denominator]):
        res = tf.math.divide(numerator,
                             tf.where(tf.math.equal(denominator, 0), tf.ones_like(denominator),
                                      denominator)),
        res = tf.where(tf.math.is_finite(res), res, tf.zeros_like(res))
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
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    sigma = tf.expand_dims(sigma, -1)

    gauss = tf.math.exp(safe_div(-tf.math.pow(x, 2), tf.math.pow(sigma, 2)))
    gauss = tf.squeeze(gauss, 0)
    return gauss


def sparse_l2_norm(sp_tensor, axis, name=None, keep_sparse=False, keepdims=False):
    with tf.name_scope(name, "l2_norm", [sp_tensor]) as name:
        square = tf.math.square(sp_tensor)
        if not keep_sparse:
            square_sum = tf.sparse_reduce_sum(square, axis, keepdims)
        else:
            square_sum = tf.sparse_reduce_sum_sparse(square, axis, keepdims)
        l2_norm = tf.math.sqrt(square_sum)
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
    with tf.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        dot_prod = tf.sparse_tensor_dense_matmul(sp_tensor1, tensor2, adjoint_b=True)

        sp_shape = tf.cast(sp_tensor1.dense_shape, tf.int32)
        dense_shape = tf.shape(tensor2)

        if keepdims:
            dot_prod = tf.reshape(dot_prod, [sp_shape[0], dense_shape[0], 1])

        return dot_prod


def dot(tensor1, tensor2, name=None):
    return tf.math.reduce_sum(tf.math.multiply(tensor1, tensor2), axis=-1)


def sparse_dot(sp_tensor1, tensor2, name=None):
    """ Returns the dot product between two tensors with the same shape

    Args:
        sp_tensor1: a ``SparseTensor``
        tensor2: a ``Tensor`` or ``SparseTensor``
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with tf.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        if isinstance(tensor2, tf.Tensor):
            # sp_radial_dif = sparse_multiply(sp_tensor1,tensor2)
            dense_values = tf.gather_nd(tensor2, sp_tensor1.indices)
            radial_dif = tf.math.multiply(sp_tensor1.values, dense_values)
            sp_radial_dif = tf.SparseTensor(indices=sp_tensor1.indices, values=radial_dif,
                                            dense_shape=sp_tensor1.dense_shape)
            dot_prod = tf.sparse_reduce_sum(sp_radial_dif, axis=-1)
            return dot_prod
        elif isinstance(tensor2, tf.SparseTensor):
            return sparse_sparse_dot(sp_tensor1, tensor2)
        else:
            raise TypeError(
                "inputs must be of type Tensor or SparseTensor: tensor2 == {t} found".format(t=type(tensor2)))


def sparse_sparse_dot(sp_tensor1, sp_tensor2, name="sparse_sparse_dot"):
    """ Returns the dot product between two tensors with the same shape

    Args:
        sp_tensor1: a ``SparseTensor``
        sp_tensor2: a ``SparseTensor``
        name: the name for this op

    Returns:
        ``Tensor``: a ``Tensor`` with the result of the dot product

    """
    with tf.name_scope(name, "sparse_sparse_dot", [sp_tensor1, sp_tensor2]):
        # sparse multiply computes the overlap between two sparse tensors
        radial_dif = sparse_sparse_multiply(sp_tensor1, sp_tensor2)
        dot_prod = tf.sparse_reduce_sum(radial_dif, axis=-1)
        return dot_prod


def sparse_multiply(sp_tensor1, tensor2, name="sparse_multiply"):
    """ Element-wise multiplication of a `Sparse Tensor` by a `Tensor` or a `SparseTensor`

    Args:
        sp_tensor1: a `SparseTensor`
        tensor2: a `Tensor` with the same shape as the sp_tensor.dense_shape

    Returns:
        a `SparseTensor` with the result of the multiplication

    """
    with tf.name_scope(name, "sparse_dot", [sp_tensor1, tensor2]):
        sp_tensor1 = tf.convert_to_tensor_or_sparse_tensor(sp_tensor1)
        assert (isinstance(sp_tensor1, tf.SparseTensor))

        tensor2 = tf.convert_to_tensor_or_sparse_tensor(tensor2)

        if isinstance(tensor2, tf.Tensor):
            dense_values = tf.gather_nd(tensor2, sp_tensor1.indices)
            dense_mul = tf.math.multiply(sp_tensor1.values, dense_values)
            result = tf.SparseTensor(sp_tensor1.indices, dense_mul, sp_tensor1.dense_shape)
            result = tf.sparse_retain(result, tf.math.not_equal(dense_mul, 0.))

            return result
        else:
            return sparse_sparse_multiply(sp_tensor1, tensor2)


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
    

def logit(x, dtype=tf.float32):
    """
    logit 
    
    ..math::
        logit(p) = log(p/(1-p)). Note that logit(0) = -inf, logit(1) = inf, and logit(p) for p<0 or p>1 yields nan.

    Args:
        dtype: input tensor dtype
        x: an input Tensor

    Returns:
        A Tensor of the same shape as x. Its entries are logit of the corresponding entry of x.
    """
    x = tf.convert_to_tensor(x, dtype)

    x = tf.math.divide(x, 1 - x)
    return tf.math.log(x)


def lookup_sparse(params, sp_tensor,
                  partition_strategy="mod",
                  name=None,
                  combiner=None,
                  max_norm=None):
    """Computes embeddings for the given ids and weights.

    This op assumes that there is at least one id for each row in the dense tensor
    represented by sp_mat (i.e. there are no rows with empty features, if so,
    put 0.0 in sp_mat entry), and that all the indices of sp_mat are in
    canonical row-major order.

    It also assumes that all id values lie in the range [0, p0), where p0
    is the sum of the size of params along dimension 0.

    Args:
      params: A single tensor representing the complete embedding tensor,
        or a list of P tensors all of same shape except for the first dimension,
        representing sharded embedding tensors.  Alternatively, a
        `PartitionedVariable`, created by partitioning along dimension 0. Each
        element must be appropriately sized for the given `partition_strategy`.
      sp_tensor: N x M SparseTensor of zero or non-zero weights,
        where N is typically batch size and M is the embedding table size.
      partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
        is `"mod"`. See `tf.nn.embedding_lookup` for more details.
      name: Optional name for the op.
      combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
        and "sum" are supported.
        "sum" computes the weighted sum of the embedding results for each row.
        "mean" is the weighted sum divided by the total weight.
        "sqrtn" is the weighted sum divided by the square root of the sum of the
        squares of the weights.
      max_norm: If not None, each embedding is normalized to have l2 norm equal
        to max_norm before combining.

    Returns:
      A dense tensor representing the combined embeddings for the sparse ids.
      For each row in the dense tensor represented by sp_mat, the op looks up
      the embeddings for all (non-zero) ids in that row, multiplies them by the
      corresponding weight, and combines these embeddings as specified.

      In other words, if

        shape(combined params) = [p0, p1, ..., pm]

      and

        shape(sp_mat) = [d0, d1, ..., dn]

      then

        shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].

      For instance, if params is a 10x20 matrix, and sp_mat is

        [0, 0]: 1.0
        [0, 1]: 3.0
        [1, 0]: 0.0
        [2, 3]: 1.0

      with `combiner`="mean", then the output will be a 3x20 matrix where

        output[0, :] = (params[0, :] * 1.0 + params[1, :] * 3.0) / (1.0 + 3.0)
        output[1, :] = params[0, :] * 0.0 / div_protect
        output[2, :] = params[3, :] * 1.0 / 1.0

    Raises:
      TypeError: If sp_mat is not a SparseTensor.
      ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
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
        raise TypeError("sp_mat must be SparseTensor")

    with tf.name_scope(name, "embedding_lookup_sparse",
                       params + [sp_tensor]) as name:
        segment_ids = sp_tensor.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        ids = sp_tensor.indices[:, -1]

        embeddings = tf.nn.embedding_lookup(
            params=params,
            ids=ids,
            partition_strategy=partition_strategy,
            max_norm=max_norm)

        # ***
        # this second lookup causes problems
        # embeddings = embedding_lookup(embeddings, idx)
        # embeddings, _ = gather_dynamic(embeddings, idx)
        # ***

        weights = sp_tensor.values
        if weights.dtype != embeddings.dtype:
            weights = tf.cast(weights, embeddings.dtype)

        # Reshape weights to allow broadcast
        ones = tf.fill(
            tf.expand_dims(tf.rank(embeddings) - 1, 0), 1)
        bcast_weights_shape = tf.concat_v2(
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


def embedding_lookup_sparse(params,
                            sp_ids,
                            sp_weights,
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
    Args:
      params: A single tensor representing the complete embedding tensor,
        or a list of P tensors all of same shape except for the first dimension,
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
        and "sum" are supported.
        "sum" computes the weighted sum of the embedding results for each row.
        "mean" is the weighted sum divided by the total weight.
        "sqrtn" is the weighted sum divided by the square root of the sum of the
        squares of the weights.
      max_norm: If not `None`, each embedding is clipped if its l2-norm is
        larger than this value, before combining.
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
    if not isinstance(sp_ids, tf.SparseTensor):
        raise TypeError("sp_ids must be SparseTensor")
    ignore_weights = sp_weights is None
    if not ignore_weights:
        if not isinstance(sp_weights, tf.SparseTensor):
            raise TypeError("sp_weights must be either None or SparseTensor")
        sp_ids.values.get_shape().assert_is_compatible_with(
            sp_weights.values.get_shape())
        sp_ids.indices.get_shape().assert_is_compatible_with(
            sp_weights.indices.get_shape())
        sp_ids.dense_shape.get_shape().assert_is_compatible_with(
            sp_weights.dense_shape.get_shape())

    with tf.name_scope(name, "embedding_lookup_sparse",
                       params + [sp_ids]) as name:
        segment_ids = sp_ids.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        ids = sp_ids.values
        unique_ids, idx = tf.unique(ids)

        if not ignore_weights:
            embeddings = tf.nn.embedding_lookup(
                params, ids, partition_strategy=partition_strategy, max_norm=max_norm)

            if embeddings.dtype in (tf.float16, tf.bfloat16):
                embeddings = tf.math.to_float(embeddings)

            weights = sp_weights.values
            if weights.dtype != embeddings.dtype:
                weights = tf.cast(weights, embeddings.dtype)

            # Reshape weights to allow broadcast
            ones = tf.fill(
                tf.expand_dims(tf.rank(embeddings) - 1, 0), 1)
            bcast_weights_shape = tf.concat([tf.shape(weights), ones],
                                            0)

            orig_weights_shape = weights.get_shape()
            weights = tf.reshape(weights, bcast_weights_shape)

            # Set the weight shape, since after reshaping to bcast_weights_shape,
            # the shape becomes None.
            if embeddings.get_shape().ndims is not None:
                weights.set_shape(
                    orig_weights_shape.concatenate(
                        [1 for _ in range(embeddings.get_shape().ndims - 1)]))

            embeddings *= weights

            if combiner == "sum":
                embeddings = tf.math.segment_sum(embeddings, segment_ids, name=name)
            elif combiner == "mean":
                embeddings = tf.math.segment_sum(embeddings, segment_ids)
                weight_sum = tf.math.segment_sum(weights, segment_ids)
                embeddings = tf.math.divide(embeddings, weight_sum, name=name)
            elif combiner == "sqrtn":
                embeddings = tf.math.segment_sum(embeddings, segment_ids)
                weights_squared = tf.math.pow(weights, 2)
                weight_sum = tf.math.segment_sum(weights_squared, segment_ids)
                weight_sum_sqrt = tf.math.sqrt(weight_sum)
                embeddings = tf.math.divide(embeddings, weight_sum_sqrt, name=name)
            else:
                assert False, "Unrecognized combiner"
        else:
            embeddings = tf.nn.embedding_lookup(
                params, unique_ids, partition_strategy=partition_strategy, max_norm=max_norm)

            if embeddings.dtype in (tf.float16, tf.bfloat16):
                embeddings = tf.math.to_float(embeddings)

            assert idx is not None
            if combiner == "sum":
                embeddings = tf.math.sparse_segment_sum(
                    embeddings, idx, segment_ids, name=name)
            elif combiner == "mean":
                embeddings = tf.math.sparse_segment_mean(
                    embeddings, idx, segment_ids, name=name)
            elif combiner == "sqrtn":
                embeddings = tf.math.sparse_segment_sqrt_n(
                    embeddings, idx, segment_ids, name=name)
            else:
                assert False, "Unrecognized combiner"
    return embeddings


__all__ = ["safe_div",
           "gaussian",
           "sparse_l2_norm",
           "sparse_dot",
           "batch_sparse_dot",
           "sparse_multiply",
           "sparse_multiply_dense",
           "logit",
           "lookup_sparse",
           "embedding_lookup_sparse"]
