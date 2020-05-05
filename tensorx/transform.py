import tensorflow as tf
from tensorflow.python import PartitionedVariable
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

from tensorx.utils import as_tensor
from tensorflow.python.framework import tensor_shape, tensor_util
from tensorx.math import sparse_multiply_dense


def sparse_indices(sp_values, name="sparse_indices"):
    """ Returns the a ``SparseTensor`` with the indices for the active values on a given ``SparseTensor`` .

    Use Case:
        To be used with ``embedding_lookup_sparse`` when we need two ``SparseTensor`` : one with the indices and
        one with the values.

    Args:
        sp_values: a ``SparseTensor`` for which we extract the active indices.
        name: name for this op

    Returns:
        ``SparseTensor``: a ``SparseTensor`` with the indices of the active elements of another ``SparseTensor`` .
    """
    with tf.name_scope(name=name):
        if len(sp_values.get_shape().dims) == 1:
            [flat_indices] = tf.unstack(sp_values.indices, num=1, axis=-1)
        else:
            _, flat_indices = tf.unstack(sp_values.indices, num=2, axis=-1)
        sp_indices = tf.SparseTensor(sp_values.indices, flat_indices, sp_values.dense_shape)

        return sp_indices


def repeat(x, n, name="repeat"):
    """ Repeats the values of a tensor along the last dimension

    Args:
        x: ``Tensor``
        n: :obj:`int` number of repetitions of each element
        name: name for the operation

    Returns:
        A `Tensor` with shape [shape[:-1, ], shape[-1:, ] * n]
    """
    with tf.name_scope(name):
        x = tf.convert_to_tensor(x)
        n = as_tensor(n, dtype=x.dtype)

        shape = tf.shape(x, out_type=x.dtype)
        flat_x = tf.reshape(x, [-1])

        rep_x = tf.tile(tf.expand_dims(flat_x, -1), tf.stack([1, n]))

        new_shape = tf.concat([shape[:-1, ], shape[-1:, ] * n], axis=-1)
        rep_x = tf.reshape(rep_x, new_shape)

    return rep_x


def matrix_indices(index_tensor, dtype=tf.int64, sort_indices=True, name="matrix_indices"):
    """ transforms a batch of column indices to a batch of matrix indices, if the indices are out of order and sorted is
     `True`, it returns, the resulting indices are sorted in canonical row-major order.

    Args:
        index_tensor (Tensor): a tensor with shape `[b,n]` with a batch of `n` column indices.
        dtype (DType): the output dtype for the indices. Defaults to `int64`.
        sort_indices (bool): if `True`, output indices are sorted in canonical row-major order.
        name (str): name for this op.

    Returns:
         tensor (Tensor): tensor with shape `[b,2]` for each index in the input tensor with the corresponding matrix
         indices.

    """
    with tf.name_scope(name):
        index_tensor = as_tensor(index_tensor, dtype)

        shape = tf.shape(index_tensor, out_type=dtype)
        row_indices = tf.range(0, shape[0])
        row_indices = repeat(row_indices, shape[1])

        # sort ascending
        if sort_indices:
            sorted_indices, _ = tf.nn.top_k(tf.cast(index_tensor, tf.int32),
                                            k=tf.cast(shape[1], tf.int32))
            sorted_indices = tf.reverse(sorted_indices, axis=[-1])
            col_indices = sorted_indices
        else:
            col_indices = index_tensor

        col_indices = tf.reshape(col_indices, [-1])
        col_indices = tf.cast(col_indices, dtype)

        indices = tf.stack([row_indices, col_indices], axis=-1)

        return indices


def sparse_ones(indices, dense_shape, dtype=tf.float32, name="sparse_ones"):
    """ Creates a new ``SparseTensor`` with the given indices having value 1

    Args:
        indices: a rank 2 ``Tensor`` with the (row,column) indices for the resulting sparse tensor
        dense_shape: the ``SparseTensor`` dense shape
        dtype: the tensor type for the values
        name: name for this op

    Returns:
        ``SparseTensor``: a new tf.SparseTensor with the values set to 1.
    """
    with tf.name_scope(name=name):
        indices = as_tensor(indices, tf.int64)
        dense_shape = as_tensor(dense_shape, tf.int64)
        indices_shape = indices.shape
        values = tf.ones([indices_shape[0]], dtype)
        return tf.SparseTensor(indices, values, dense_shape)


def sparse_one_hot(column_indices, num_cols, dtype=tf.float32, name="sparse_one_hot"):
    """Transforms a batch of column indices to a one-hot encoding ``SparseTensor``.

        Example::

            indices = [[0,1,4],
                       [1,2,6]]

            dense_shape = [2,10]

            sp_one_hot = sparse_one_hot(indices,dense_shape)

            expected = tf.SparseTensor(indices=[[0,0],[0,1],[0,4],[1,1],[1,2],[1,6]],
                                    values=[1,1,1,1,1,1],
                                    dense_shape=[2,10])

        Args:
            column_indices: a dense ``Tensor`` with the indices to be active for each sample (row)
            num_cols: number of columns for the one-hot encoding
            dtype: the type for the output values.
            name: name for this op

        Returns:
            `SparseTensor`: a ``Sparse Tensor`` with the one hot encoding for the given indices
    """
    with tf.name_scope(name=name):
        column_indices = as_tensor(column_indices, tf.int64)
        indices = matrix_indices(column_indices, dtype=tf.int64)

        dense_shape = tf.cast([tf.shape(column_indices)[0], num_cols], dtype=tf.int64)

        return sparse_ones(indices, dense_shape, dtype)


def _get_noise_shape(x, noise_shape):
    # If noise_shape is none return immediately.
    if noise_shape is None:
        return tf.shape(x)

    try:
        noise_shape_ = tensor_shape.as_shape(noise_shape)
    except (TypeError, ValueError):
        return noise_shape

    if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
        new_dims = []
        for i, dim in enumerate(x.shape.dims):
            if noise_shape_.dims[i].value is None and dim.value is not None:
                new_dims.append(dim.value)
            else:
                new_dims.append(noise_shape_.dims[i].value)
        return tf.TensorShape(new_dims)

    return noise_shape


def dropout(tensor,
            noise_shape=None,
            random_mask=None,
            probability=0.1,
            scale=True,
            seed=None,
            return_mask=False,
            name="dropout"):
    """ dropout

    With probability `probability`, outputs `0`  otherwise outputs the input element. If ``scale`` is True, the
    input elements are scaled up by `1 / (1-probability)` so that the expected sum of the activations is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Args:
        noise_shape: A 1-D `Tensor` of type `int32`, representing the shape for randomly generated drop flags.
        return_mask: if true, returns the random mask used
        random_mask: a tensor used to create the random bernoulli mask
        tensor: A floating point tensor.
        probability: A scalar `Tensor` with the same type as x. The probability that each element is kept.
        scale: if true rescales the non-zero elements to 1 / (1-drop_probability)
        seed: A Python integer with the random number generator seed
        name: a name for this operation (optional)

    Returns:
        a Tensor of the same shape of tensor

    Raises:
        ValueError: If `probability` is not in `[0, 1]` or if `x` is not a floating point tensor.
    """
    with tf.name_scope(name):
        tensor = tf.convert_to_tensor(tensor, name="x")
        if random_mask is not None:
            random_mask = as_tensor(random_mask, tensor.dtype)

        if not tensor.dtype.is_floating:
            try:
                tensor = tf.cast(tensor, tf.float32)
            except Exception as e:
                raise ValueError("x has to be a floating point tensor since it might be scaled"
                                 "Got a %s tensor instead. and could not cast it" % tensor.dtype)

        if not 0 <= probability < 1:
            raise ValueError("drop probability must be a scalar tensor or a float in the "
                             "range [0, 1), got %g" % probability)

        # Early return if nothing needs to be dropped.
        if isinstance(probability, float) and probability == 0:
            if return_mask:
                return tensor, None
            else:
                return tensor
        elif isinstance(probability, float) and probability == 1:
            zeros = tf.zeros_like(tensor)
            if return_mask:
                return zeros, None
            else:
                return zeros

        probability = tf.convert_to_tensor(
            probability, dtype=tensor.dtype, name="drop_probability")
        probability.get_shape().assert_is_compatible_with(tf.TensorShape([]))

        # Do nothing if we know drop_probability == 0
        const_val = tensor_util.constant_value(probability)
        if const_val == 0:
            if return_mask:
                return tensor, None
            else:
                return tensor
        elif const_val == 1:
            zeros = tf.zeros_like(tensor)
            if return_mask:
                return zeros, None
            else:
                return zeros

        noise_shape = _get_noise_shape(tensor, noise_shape)

        if random_mask is None:
            with tf.name_scope(name="random_mask"):
                keep_prob = 1 - probability
                random_state = tf.random.uniform(noise_shape, seed=seed, dtype=tensor.dtype)
                mask = keep_prob + random_state
                random_mask = tf.math.floor(mask, name="binary_mask")

        if scale:
            ret = tf.math.divide(tensor, tf.math.maximum(1 - probability, 1e-10)) * random_mask
        else:
            ret = tensor * random_mask
        if not tf.executing_eagerly():
            ret.set_shape(tensor.get_shape())

        if return_mask:
            return ret, random_mask
        else:
            return ret


def binary_mask(tensor, mask_probability=0.0, seed=None):
    with tf.name_scope(name="random_mask"):
        tensor = as_tensor(tensor)
        noise_shape = _get_noise_shape(tensor, None)
        keep_prob = 1 - mask_probability
        random_state = tf.random.uniform(noise_shape, seed=seed, dtype=tensor.dtype)
        mask = keep_prob + random_state
        random_mask = tf.math.floor(mask, name="binary_mask")

        return random_mask


def sparse_dropout(sp_tensor,
                   probability=0.2,
                   scale=True,
                   seed=None,
                   mask=None,
                   return_mask=False,
                   name="sparse_dropout"):
    """Performs a dropout on a ``SparseTensor``.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    Args:
        mask: a random_mask to be applied to the values of this tensor
        return_mask: if true returns the random_mask used to perform dropout (result,random_mask)
        sp_tensor: a ``SparseTensor`` on which the dropout is performed.
        probability: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
        scale: if True rescales the input to 1 / keep_prob else simply drops without rescaling
        seed: A Python integer. Used to create random seeds. (See `TensorFlow` documentation
        for ``tf.set_random_seed`` for behavior.)
        name: A name for this operation (optional).

    """
    with tf.name_scope(name=name):
        dense_shape = sp_tensor.dense_shape

        if not sp_tensor.values.dtype.is_floating:
            raise ValueError("sp_tensor has to be a floating point tensor since its values are going to"
                             " be scaled. Got a %s tensor instead." % sp_tensor.dtype)
        if not 0 <= probability < 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % probability)
        probability = tf.convert_to_tensor(probability,
                                           dtype=sp_tensor.dtype,
                                           name="drop_probability")

        drop_values = dropout(tensor=sp_tensor.values,
                              random_mask=mask,
                              probability=probability,
                              scale=scale,
                              return_mask=return_mask,
                              seed=seed)

        if return_mask is not None:
            drop_values, mask = drop_values

        not_zero = tf.math.not_equal(drop_values, 0)
        values = tf.boolean_mask(drop_values, not_zero)
        indices = tf.boolean_mask(sp_tensor.indices, not_zero)

        new_tensor = tf.SparseTensor(indices, values, dense_shape)

        if return_mask is not None:
            return new_tensor, mask
        else:
            return new_tensor


def apply_gate(x, gate):
    with tf.name_scope("apply_gate"):
        x = as_tensor(x)
        gate = as_tensor(gate)

        n_gates = tf.shape(gate)[-1]
        n_units = tf.shape(x)[-1]
        feature_dim = n_units // n_gates

        if isinstance(x, tf.SparseTensor):
            tensor_in = tf.sparse_reshape(x, [-1, n_gates, feature_dim])
            gated = sparse_multiply_dense(tensor_in, tf.expand_dims(gate, -1))
        else:
            tensor_in = tf.reshape(x, [-1, n_gates, feature_dim])
            gated = tensor_in * tf.expand_dims(gate, -1)

        out_shape = tf.stack([-1, n_units])
        output = tf.reshape(gated, out_shape)

        # since n_units is taken from a tensor, we need to set the shape manually
        # otherwise this can't be determined
        # this was required in the non-eager version
        # output.set_shape(tf.TensorShape([None, n_units]))

        return output


def empty_sparse_tensor(dense_shape, dtype=tf.float32, name="empty_sp_tensor"):
    """ Creates an empty tf.SparseTensor.

    Note:
        ``shape = [10]``

        ``empty = tf.sparse.to_dense(transform.empty_sparse_tensor(shape))``

        is equivalent to:

        ``zero = tf.zeros(shape)``

       meaning that:

       ``tf.reduce_all(tf.equal(zeros,empty)).eval()``

       returns True


    Args:
        dense_shape: a 1-D tensor, python list, or numpy array with the output shape for the sparse tensor
        dtype: the dtype of the values for the empty tf.SparseTensor
        name: a name for this operation (optional)

    Returns:
        ``SparseTensor``: an empty sparse tensor with a given shape

    """
    with tf.name_scope(name):
        dense_shape = tf.convert_to_tensor(dense_shape, name="dense_shape", dtype=tf.int64)

        index_shape = dense_shape.get_shape().with_rank(1)
        empty_indices = tf.ones([0, index_shape[0]], dtype=tf.int64)
        empty_values = tf.ones([0], dtype=dtype)

        return tf.SparseTensor(empty_indices, empty_values, dense_shape)


class SparseVariable:
    """

    TODO: check this problem for values
        https://github.com/tensorflow/tensorflow/issues/32215
    """

    def __init__(self, initial_value: tf.SparseTensor, trainable=True, validate_shape=True, dtype=None,
                 name="sparsevar"):
        self.indices = tf.Variable(initial_value=initial_value.indices,
                                   trainable=False,
                                   dtype=tf.int64,
                                   shape=tf.TensorShape([None] * len(initial_value.shape[:-1]) +
                                                        [initial_value.indices.shape[-1]]),
                                   validate_shape=validate_shape,
                                   name=f"{name}_indices")

        self.values = tf.Variable(initial_value=initial_value.values,
                                  dtype=dtype,
                                  trainable=trainable,
                                  validate_shape=validate_shape,
                                  shape=tf.TensorShape([None]),
                                  name=f"{name}_values"
                                  )

        self.shape = tf.Variable(initial_value=initial_value.shape.as_list(),
                                 trainable=False,
                                 dtype=tf.int64,
                                 validate_shape=validate_shape,
                                 shape=tf.TensorShape([None]),
                                 name=f"{name}_shape")

    def assign(self, sp_value):
        if len(sp_value.shape) != len(self.shape.value()):
            raise ValueError("cannot assign SparseTensor with Different dimensions")

        self.indices.assign(sp_value.indices)
        self.values.assign(sp_value.values)
        self.shape.assign(sp_value.dense_shape)

    def value(self):
        sp = tf.SparseTensor(self.indices, self.values, self.shape)
        return tf.sparse.reorder(sp)


def to_sparse(tensor, name="to_sparse"):
    """ Returns a ``SparseTensor` for a`given dense ``Tensor``.

    Example:

        For a dense ``Tensor`` such as::

            tensor = [[1,0],
                      [2,3]]

        this returns an op that creates the following two ``SparseTensor``::

            tf.SparseTensor(indices = [[0,0],[1,0],[1,1]],
                                    values = [1,2,3],
                                    dense_shape = [2,2])

    Args:
        tensor: a dense ``Tensor``
        name: name for this operation (optional)

    Returns:
        ``SparseTensor``: a sparse tensor with sparse index and value tensors
        with the non-zero entries of the given input.

    """
    with tf.name_scope(name=name):
        indices = tf.where(tf.math.not_equal(tensor, 0))
        dense_shape = tf.shape(tensor, out_type=tf.int64)

        values = tf.gather_nd(tensor, indices)
        sp_tensor = tf.SparseTensor(indices, values, dense_shape)

        return sp_tensor


def embedding_lookup_sparse(params,
                            sp_tensor,
                            partition_strategy="mod",
                            combiner=None,
                            max_norm=None,
                            name="embedding_lookup_sparse"):
    """Computes embeddings for the given ids and weights.

    !!! info
        assumes that there is at least one id for each row in the dense tensor
        represented by sp_ids (i.e. there are no rows with empty features), and that
        all the indices of sp_ids are in canonical row-major order.
        It also assumes that all id values lie in the range [0, p0), where p0
        is the sum of the size of params along dimension 0.

    !!! note
        in tensorflow's implementation, sparse gradients do not propagate through gather.

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

    with tf.name_scope(name) as name:
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


def dense_one_hot(column_indices, num_cols, dtype=tf.float32, name="dense_one_hot"):
    """transforms a batch of indices into a dense `Tensor` where each row represents a `one-hot` encoding for the
    indices.

    Examples:
        ```python
        indices = [0,1]
        dense_one_hot(indices,num_cols=2)

        [[1,0],
         [0,1]]
        ```

        If a multiple indices are passed for each row, their one-hot encodings are summed.

        ```python
        indices = [[0,1],
                   [1,1]]
        dense_one_hot(indices,num_cols=2)

        [[1,1],
         [0,2]]
        ```

    Args:
        column_indices (Tensor): a dense `Tensor` with the active indices for each row.
        num_cols (int): total number of columns for the one-hot encoding
        dtype (Dtype): output tensor dtype
        name (str): name for this op

    Returns:
        dense_tensor (Tensor): A dense ``Tensor`` with a [one-hot encoding](https://en.wikipedia.org/wiki/One-hot)
        for the given indices.
    """
    with tf.name_scope(name=name):
        column_indices = as_tensor(column_indices, tf.int64)
        one_hot_dense = tf.one_hot(column_indices, depth=num_cols, dtype=dtype)

        if column_indices.get_shape().ndims == 2:
            one_hot_dense = tf.math.reduce_sum(one_hot_dense, axis=1)

        return one_hot_dense


def sparsemax(logits, name=None):
    """Computes the sparsemax activation function [1]

    TODO needs tests

    For each batch `i` and class `j` we have
      sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)

    References:
        [1]: https://arxiv.org/abs/1602.02068

    Args:
      logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
        `float64`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `logits`.
    """

    with ops.name_scope(name, "sparsemax", [logits]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        obs = tf.shape(logits)[0]
        dims = tf.shape(logits)[1]

        z = logits - tf.reduce_mean(logits, axis=1)[:, tf.newaxis]

        # sort z
        z_sorted, _ = tf.nn.top_k(z, k=dims)

        # calculate k(z)
        z_cumsum = tf.cumsum(z_sorted, axis=1)
        k = tf.range(
            1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = tf.reduce_sum(tf.cast(z_check, tf.int32), axis=1)

        # calculate tau(z)
        indices = tf.stack([tf.range(0, obs), k_z - 1], axis=1)
        tau_sum = tf.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

        # calculate p
        return tf.maximum(
            tf.cast(0, logits.dtype), z - tau_z[:, tf.newaxis])


__all__ = ["apply_gate",
           "sparse_ones",
           "sparse_indices",
           "sparse_one_hot",
           "matrix_indices",
           "dropout",
           "sparse_dropout",
           "binary_mask",
           "empty_sparse_tensor",
           "SparseVariable",
           "to_sparse",
           "embedding_lookup_sparse",
           "dense_one_hot",
           "sparsemax"]
