import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops.variables import PartitionedVariable
from tensorx.utils import as_tensor
from tensorflow.python.framework import tensor_shape, tensor_util
from tensorx import math as mx


def sparse_ones(indices, dense_shape, dtype=tf.float32, name="sparse_ones"):
    """ Creates a new `SparseTensor` with the given indices having value 1

    Args:
        indices (`Tensor`): a rank 2 tensor with the `(row,column)` indices for the resulting sparse tensor
        dense_shape (`Tensor` or `TensorShape`): the output dense shape
        dtype (`tf.DType`): the tensor type for the values
        name (`str`): sparse_ones op

    Returns:
        sp_tensor (`SparseTensor`): a new sparse tensor with values set to 1
    """
    with tf.name_scope(name=name):
        indices = as_tensor(indices, tf.int64)
        dense_shape = as_tensor(dense_shape, tf.int64)
        indices_shape = indices.shape
        values = tf.ones([indices_shape[0]], dtype)
        return tf.SparseTensor(indices, values, dense_shape)


def sparse_zeros(indices, dense_shape, dtype=tf.float32, name="sparse_zeros"):
    """ Creates a new `SparseTensor` with the given indices having value 0

    Args:
        indices (`Tensor`): a rank 2 tensor with the `(row,column)` indices for the resulting sparse tensor
        dense_shape (`Tensor` or `TensorShape`): the output dense shape
        dtype (`tf.DType`): the tensor type for the values
        name (`str`): sparse_ones op

    Returns:
        sp_tensor (`SparseTensor`): a new sparse tensor with values set to 0
    """
    with tf.name_scope(name=name):
        indices = as_tensor(indices, tf.int64)
        dense_shape = as_tensor(dense_shape, tf.int64)
        indices_shape = tf.shape(indices)
        values = tf.zeros([indices_shape[0]], dtype)
        return tf.SparseTensor(indices, values, dense_shape)


def sparse_indices(sp_values, name="sparse_indices"):
    """ Returns a `SparseTensor` with the values containing column indices for the active values on a
    given `SparseTensor`.

    !!! example "Use Case"
        To be used with ``embedding_lookup_sparse`` when we need two `SparseTensor` objects with the indices and
        values

    Args:
        sp_values (`SparseTensor`): a sparse tensor for which we extract the active indices.
        name (`str`): name for sparse_indices op

    Returns:
        sp_indices (`SparseTensor`): a sparse tensor with the column indices
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
        x (`Tensor`): input tensor
        n (`int`): number of repetitions of each element
        name (`str`): name for the repeat op

    Returns:
        tensor (`Tensor`): tensor with shape [shape[:-1, ], shape[-1:, ] * n]
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
    """ Transforms a batch of column indices into a batch of matrix indices

    Args:
        index_tensor (`Tensor`): a tensor with shape `(b,n)` with a batch of `n` column indices.
        dtype (`DType`): the output dtype for the indices. Defaults to `int64`.
        sort_indices (`bool`): if `True`, output indices are sorted in canonical row-major order.
        name (`str`): name for this op.

    Returns:
        tensor (`Tensor`): tensor with shape `[b,2]` for each index in the input tensor with the corresponding matrix
        indices
    """
    with tf.name_scope(name):
        index_tensor = as_tensor(index_tensor, dtype)
        if len(index_tensor.shape) < 2:
            index_tensor = tf.expand_dims(index_tensor, 0)

        shape = tf.shape(index_tensor, out_type=dtype)
        row_indices = tf.range(0, shape[0])
        row_indices = repeat(row_indices, shape[-1])

        # sort ascending
        if sort_indices:
            sorted_indices, _ = tf.nn.top_k(tf.cast(index_tensor, tf.int32),
                                            k=tf.cast(shape[-1], tf.int32))
            sorted_indices = tf.reverse(sorted_indices, axis=[-1])
            col_indices = sorted_indices
        else:
            col_indices = index_tensor

        col_indices = tf.reshape(col_indices, [-1])
        col_indices = tf.cast(col_indices, dtype)

        indices = tf.stack([row_indices, col_indices], axis=-1)

        return indices


def dense_one_hot(column_indices, num_cols, dtype=tf.float32, reduce=True, name="dense_one_hot"):
    """Transforms a batch of indices to a dense `Tensor` by adding the `one-hot` encoding for each index.

    Example:
        ```python
        indices = [[0],[1]]
        dense_shape = [2,2]

        dense_one_hot = [[1,0],[0,1]]
        ```

    Args:
        column_indices: a dense `Tensor` with the active indices for each sample (row).
        num_cols: number of columns for the one-hot encoding
        dtype: the type for the output tensor.
        reduce (`bool`): if true applies reduce sum on last dimension,
        name: name for this op

    Returns:
        `Tensor`: A dense `Tensor` with a `one-hot encoding` for the given indices.
    """
    with tf.name_scope(name=name):
        column_indices = as_tensor(column_indices, tf.int64)
        one_hot_dense = tf.one_hot(column_indices, depth=num_cols, dtype=dtype)

        if column_indices.get_shape().ndims >= 2 and reduce:
            one_hot_dense = tf.math.reduce_sum(one_hot_dense, axis=1)

        return one_hot_dense


def sparse_matrix_indices(column_indices, num_cols, dtype=tf.float32, name="sparse_one_hot"):
    """Transforms a batch of column indices to a one-hot encoding `SparseTensor`.

        Example:
            ``` python
            indices = [[0,1,4],
                       [1,2,6]]
            dense_shape = [2,10]
            sp_one_hot = sparse_one_hot(indices,dense_shape)
            expected = tf.SparseTensor(indices=[[0,0],
                                                [0,1],
                                                [0,4],
                                                [1,1],
                                                [1,2],
                                                [1,6]],
                                    values=[1,1,1,1,1,1],
                                    dense_shape=[2,10])
            ```

        Args:
            column_indices (`Tensor`): a dense tensor with the indices to be active for each sample (row)
            num_cols (`int`): number of columns for the one-hot encoding
            dtype (`tf.DType`): the type for the output values.
            name (`str`): name for this op

        Returns:
            sp_tensor (`SparseTensor`): a sparse tensor with the one hot encoding for the given indices
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
    """ With probability `probability`, outputs `0`  otherwise outputs the input element. If ``scale`` is True, the
    input elements are scaled up by `1 / (1-probability)` so that the expected sum of the activations is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Args:
        tensor (`Tensor`): an input tensor
        noise_shape (`Tensor`): A 1-D `Tensor` of type `int32`, representing the shape for randomly generated drop flags
        return_mask (`bool`): if `True`, returns the random mask used
        random_mask (`Tensor`): a tensor used to create the random bernoulli mask
        probability (`float` or `Tensor`): A scalar `Tensor` with the same type as x. The probability that each element
            is kept.
        scale (`bool`): if true rescales the non-zero elements to 1 / (1-drop_probability)
        seed (`int`): A Python integer with the random number generator seed
        name (`str`): a name for this operation

    Returns:
        tensor (`Tensor`): output tensor with the same `DType` as the input

    Raises:
        ValueError: if `probability` is not in `[0, 1]` or if `x` is not a floating point tensor.
    """
    with tf.name_scope(name):
        tensor = tf.convert_to_tensor(tensor, name="x")
        if random_mask is not None:
            random_mask = as_tensor(random_mask, tensor.dtype)

        if not tensor.dtype.is_floating:
            try:
                tensor = tf.cast(tensor, tf.float32)
            except Exception:
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


def alpha_dropout(tensor,
                  noise_shape=None,
                  random_mask=None,
                  probability=0.1,
                  seed=None,
                  return_mask=False,
                  name="dropout"):
    """ Alpha Dropout keeps mean and variance of inputs in order to ensure the self-normalization after dropout.
    Alpha dropout is proposed for Scaled Exponential Linear Units (SELUs) because it randomly sets activations to the
    negative saturation value rather than 0.

    The multiplicative noise will have standard deviation $\\sqrt{\\frac{probability}{(1-probability)}}

    !!! cite "References"
        1. [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf)

    Args:
        tensor (`Tensor`): A floating point tensor.
        noise_shape (`Tensor`): A 1-D `Tensor` of type `int32`, representing the shape for randomly generated drop flags
        return_mask (`bool`): if true, returns the random mask used
        random_mask (`Tensor`): a tensor used to create the random bernoulli mask
        probability (`float` or `Tensor`): A scalar `Tensor` with the same type as x. The probability that each element
            is kept.
        seed (`int`): A Python integer with the random number generator seed
        name (`str`): a name for this operation (optional)

    Returns:
        result (`Tensor`): a tensor with the same shape as the input with the dropped units set to negative values

    """
    tensor = tf.convert_to_tensor(tensor, name="x")
    with tf.name_scope(name):
        if random_mask is not None:
            random_mask = as_tensor(random_mask, tensor.dtype)

        if not tensor.dtype.is_floating:
            try:
                tensor = tf.cast(tensor, tf.float32)
            except Exception:
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

        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        alpha_p = -alpha * scale

        # Get affine transformation params
        a = ((1 - probability) * (1 + probability * alpha_p ** 2)) ** -0.5
        b = -a * alpha_p * probability

        # Apply mask
        x = tensor * random_mask + alpha_p * (1 - random_mask)

        # Do affine transformation
        return a * x + b


def binary_random_mask(tensor, mask_probability=0.0, seed=None):
    """ Creates a binary mask with the same shape as the given tensor, randomly generated from
    the given mask probability.

    Args:
        tensor (`Tensor`): tensor for which we would like to create a mask
        mask_probability (`float`, `Tensor`): scalar tensor or float with probability of masking a given value
        seed (`int`): seed for random number generator

    Returns:
        binary_mask (`Tensor`): a tensor with values `0` or `1` with the same shape as the input tensor

    """
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
                   alpha=False,
                   name="sparse_dropout"):
    """ Performs a dropout on a `SparseTensor`.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    Args:
        sp_tensor (`SparseTensor`): a sparse tensor on which the dropout is performed.
        mask (`Tensor`): a binary random mask to be applied to the values of this tensor
        return_mask (`bool`): if true returns the random_mask used to perform dropout (result,random_mask)
        probability (`float`, `Tensor`): A scalar tensor with the same type as x. The probability
            that each element is kept.
        scale (`bool`): if True rescales the input to 1 / keep_prob else simply drops without rescaling
        seed (`int): A Python integer used as seed. (See `TensorFlow` documentation
            for ``tf.set_random_seed`` for behavior.)
        alpha (`bool`): if True uses `alpha_dropout` instead of `dropout` in the inputs
        name (`str`): A name for this operation (optional).

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
        if alpha:
            drop_values = alpha_dropout(tensor=sp_tensor.values,
                                        random_mask=mask,
                                        probability=probability,
                                        return_mask=return_mask,
                                        seed=seed
                                        )
        else:
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


def apply_gate(tensor, gate):
    """ Applies a gate tensor to the given input

    if input tensor outer dimension is a multiple of gate outer dimension we use broadcasting to apply the gate evenly
    across the input tensor.

    Example:
        ```python
        tx.apply_gate(tf.ones([1,4]),[1.,0.])

        [[1., 1., 0., 0.]]
        ```

    Args:
        tensor (`Tensor`): an input tensor
        gate (`Tensor`): float tensor that is multiplied by the input tensor. The outer dimension of the input tensor
            should either match the gate tensor or be a multiple of gate tensor.

    Returns:
        gated (`Tensor`): input tensor gated using the given gate weights

    """
    with tf.name_scope("apply_gate"):
        tensor = as_tensor(tensor)
        gate = as_tensor(gate)

        n_gates = tf.shape(gate)[-1]
        n_units = tf.shape(tensor)[-1]
        feature_dim = n_units // n_gates

        if isinstance(tensor, tf.SparseTensor):
            tensor_in = tf.sparse.reshape(tensor, [-1, n_gates, feature_dim])
            gate = tf.expand_dims(gate, -1)
            gated = mx.sparse_dense_multiply(tensor_in, gate)
        else:
            tensor_in = tf.reshape(tensor, [-1, n_gates, feature_dim])
            gated = tensor_in * tf.expand_dims(gate, -1)

        out_shape = tf.stack([-1, n_units])
        output = tf.reshape(gated, out_shape)

        return output


def empty_sparse_tensor(dense_shape, dtype=tf.float32, name="empty_sp_tensor"):
    """ Creates an empty `SparseTensor`

    Args:
        dense_shape (`TensorShape`): a 1-D tensor, python list, or numpy array with the output shape for the sparse tensor
        dtype (`DType`): the dtype of the values for the empty tf.SparseTensor
        name (`str`): a name for this operation

    Returns:
        sp_tensor (`SparseTensor`): an empty sparse tensor with a given shape

    """
    with tf.name_scope(name):
        dense_shape = tf.convert_to_tensor(dense_shape, name="dense_shape", dtype=tf.int64)

        index_shape = dense_shape.get_shape().with_rank(1)
        empty_indices = tf.ones([0, index_shape[0]], dtype=tf.int64)
        empty_values = tf.ones([0], dtype=dtype)

        return tf.SparseTensor(empty_indices, empty_values, dense_shape)


# TODO: check this problem for values
#        https://github.com/tensorflow/tensorflow/issues/32215
class SparseVariable:
    """ SparseVariable is the equivalent of `tf.Variable` but `SparseTensor` values can be assigned directly
    for an update.

    Args:
        initial_value (`SparseValue`): sparse variable initial value
        trainable (`bool`): if `True` sets the values tensor variable as trainable
        validate_shape (`bool`): if False, the shape of the variables is not checked
        dtype (`DType`): data type
        name (`str`): name for sparse variable
    """

    def __init__(self,
                 initial_value: tf.SparseTensor,
                 trainable=True,
                 validate_shape=True,
                 dtype=None,
                 name="sparse_var"):
        with tf.name_scope(name):
            self.indices = tf.Variable(initial_value=initial_value.indices,
                                       trainable=False,
                                       dtype=tf.int64,
                                       shape=tf.TensorShape([None] * len(initial_value.shape[:-1]) +
                                                            [initial_value.indices.shape[-1]]),
                                       validate_shape=validate_shape,
                                       name=f"indices")

            self.values = tf.Variable(initial_value=initial_value.values,
                                      dtype=dtype,
                                      trainable=trainable,
                                      validate_shape=validate_shape,
                                      shape=tf.TensorShape([None]),
                                      name=f"values"
                                      )

            self.shape = tf.Variable(initial_value=initial_value.shape.as_list(),
                                     trainable=False,
                                     dtype=tf.int64,
                                     validate_shape=validate_shape,
                                     shape=tf.TensorShape([None]),
                                     name=f"shape")

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
    """Converts a given `Tensor` in a `SparseTensor`

    Example:
        For a dense `Tensor` such as:
        ```python
        tensor = [[1,0],
                  [2,3]]
        ```
        this returns an op that creates the following two `SparseTensor`:

        ```python
        tf.SparseTensor(indices = [[0,0],
                                   [1,0],
                                   [1,1]],
                        values = [1,2,3],
                        dense_shape = [2,2])
        ```
    Args:
        tensor (`Tensor`): a dense tensor
        name (`str`): name for to_sparse op

    Returns:
        sp_tensor (`SparseTensor`): a sparse tensor with sparse index and value tensors
        with the non-zero entries of the given input.

    """
    with tf.name_scope(name=name):
        indices = tf.where(tf.math.not_equal(tensor, 0))
        dense_shape = tf.shape(tensor, out_type=tf.int64)

        values = tf.gather_nd(tensor, indices)
        sp_tensor = tf.SparseTensor(indices, values, dense_shape)

        return sp_tensor


# TODO check if this has been fixed from previous versions
def embedding_lookup_sparse(params,
                            sp_tensor,
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
        sp_tensor:
        params: A single tensor representing the complete embedding tensor, or a
        list of P tensors all of same shape except for the first dimension,
        representing sharded embedding tensors.  Alternatively, a
        `PartitionedVariable`, created by partitioning along dimension 0. Each
        element must be appropriately sized for the given `partition_strategy`.

        sp_tensor (`SparseTensor`):  N x M `SparseTensor` with the ids and weights
            where N is typically batch size and M is arbitrary.

        combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
        and "sum" are supported. "sum" computes the weighted sum of the embedding
        results for each row. "mean" is the weighted sum divided by the total
        weight. "sqrtn" is the weighted sum divided by the square root of the sum
        of the squares of the weights.

        max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
        than this value, before combining.

        name (`str`): op name

    Returns:
        tensor (`Tensor`): dense tensor representing the combined embeddings for the
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


def sparse_overlap(sp_tensor1, sp_tensor2, name="sparse_overlap"):
    """sparse overlap

    Returns a `SparseTensor` where the indices of the overlapping indices in the two
    sparse tensors with the values of the first one.

    Args:
        sp_tensor1 (`SparseTensor`): a sparse tensor
        sp_tensor2 (`SparseTensor`): another sparse tensor
        name (`str`): name for sparse_overlap op

    Returns:
        sp_tensor (`SparseTensor`): sparse tensor with the overlapping indices and the values of `sp_tensor1`
    """
    with tf.name_scope(name):
        ones1 = mx.sparse_ones(sp_tensor1.indices, sp_tensor1.dense_shape)
        ones2 = mx.sparse_ones(sp_tensor2.indices, sp_tensor2.dense_shape)

        index_union = tf.sparse.add(ones1, ones2)

        index_filter = tf.equal(index_union.values, 2.)

        zeros1 = sparse_zeros(index_union.indices, index_union.dense_shape, sp_tensor1.values.dtype)
        expand1 = tf.sparse.add(zeros1, sp_tensor1)

        filtered = tf.sparse.retain(expand1, index_filter)
        return filtered


def sort_by_first(tensor1, tensor2, ascending=True, name="sort_by_first"):
    """sort_by_first

    Sorts two tensors. Sorts the second by the changes in the first sort

    Args:
        tensor1 (`Tensor`): tensor to determine the oder by which the second is sorted
        tensor2 (`Tensor`): tensor to be sorted according to the sorting of the first
        ascending (`Bool`): if True sorts by ascending order of value
        name (`str`): name of the op

    Returns:
        tensor1, tensor2 (`Tensor`,`Tensor`): sorted first tensor, second tensor sorted according to the indices of the
        first tensor sorting

    """

    with tf.name_scope(name=name):
        tensor1 = as_tensor(tensor1)
        tensor2 = as_tensor(tensor2)

        sorted_tensor1, sorted_tensor1_indices = tf.nn.top_k(tensor1, k=tf.shape(tensor1)[-1])
        if ascending:
            sorted_tensor1 = tf.reverse(sorted_tensor1, axis=[-1])
            sorted_tensor1_indices = tf.reverse(sorted_tensor1_indices, axis=[-1])

        # TODO not sure what the performance implication of this check is when converted to graph
        if len(tensor1.shape.as_list()) == 1:
            sorted_tensor1_indices = tf.expand_dims(sorted_tensor1_indices, 1)
        else:
            sorted_tensor1_indices = matrix_indices(sorted_tensor1_indices, sort_indices=False)

        sorted_values = tf.gather_nd(tensor2, sorted_tensor1_indices)
        sorted_values = tf.reshape(sorted_values, tf.shape(tensor2))

        return sorted_tensor1, sorted_values


def ranges(range_sizes, name="ranges"):
    """ ranges

    similar to concatenating multiple `tf.range` calls applied
    to each element of a given 1D tensor with range sizes.

    Example:

        ```python
        ranges([1,2,4])

        [0,0,1,0,1,2,3]
        ```

        the enums are `[0]`, `[0,1]`, `[0,1,2,3]`

    Args:
        range_sizes (`Tensor`): 1D tensor with range sizes
        name (`str`): ranges op name

    Returns:
        ranges (`Tensor`): a 1D `Tensor` with `tf.reduce_sum(range_sizes)` dimensions
    """
    with tf.name_scope(name):
        range_sizes = tf.convert_to_tensor(range_sizes)

        tf.ensure_shape(range_sizes, tf.TensorShape([None]))

        tf.debugging.assert_greater(tf.shape(range_sizes)[0], 0,
                                    message="range_sizes cannot be empty")

        num_ranges = tf.shape(range_sizes)[0]

        # get maximum repeat length in x
        max_len = tf.math.reduce_max(range_sizes)
        x = tf.range(max_len)

        # tile it to the maximum repeat length [maxlen x maxlen] now
        x_repeat = tf.stack([num_ranges, 1], axis=0)
        x_tiled = tf.tile(tf.expand_dims(x, 0), x_repeat)

        # create a sequence mask using x
        # this will create a boolean matrix of shape [xlen, max_len]
        # where result[i,j] is true if j < x[i].
        mask = tf.sequence_mask(range_sizes, max_len)

        # mask the elements based on the sequence mask
        return tf.boolean_mask(x_tiled, mask)


def gather_sparse(sp_tensor, ids, name="gather_sparse"):
    """ gather_sparse

    gather rows from a sparse tensor by the given ids and returns a sparse tensor

    !!! warning
        gathering from a `SparseTensor` is inefficient


    Example:
        ```python
        gather_sparse(sp_tensor,[1,1,4])
        ```
        returns a `[3,sp_tensor.dense_shape[-1]]` `SparseTensor`

    Args:
        sp_tensor (`SparseTensor`): sparse tensor
        ids (`Tensor`): an int tensor with the ids of the rows to be returned
        name (`str`): on name

    Returns:
        sp_gathered (`SparseTensor`): a sparse tensor with the gathered rows.

    """
    with tf.name_scope(name=name):
        ids = tf.cast(ids, tf.int64)
        ids = tf.reshape(ids, [-1])

        # count columns and compute row coordinates
        sp_column_ones = sparse_ones(sp_tensor.indices, sp_tensor.dense_shape, dtype=tf.int64)
        col_count = tf.sparse.reduce_sum(sp_column_ones, axis=-1)
        # sparse_reduce_sum sets shape to unknown
        col_count.set_shape([sp_tensor.get_shape().as_list()[0]])
        col_count_cs = tf.math.cumsum(col_count)
        row_start_coor = col_count_cs - col_count

        g_col_count = tf.gather(col_count, ids)
        g_row_start_coor = tf.gather(row_start_coor, ids)

        row_start_coor = tf.repeat(g_row_start_coor, g_col_count)
        # col_counts = repeat_each(g_col_count, g_col_count)

        offset = ranges(g_col_count)

        # use modular arithmetic to make sure we get incremental coordinates
        # gather_ids = row_start_coor + offset % col_counts
        gather_ids = row_start_coor + offset

        num_ids = tf.cast(tf.shape(ids)[0], tf.int64)
        new_rows = tf.repeat(tf.range(num_ids), g_col_count)

        sp_cols = sp_tensor.indices[:, -1]
        new_cols = tf.gather(sp_cols, gather_ids)
        new_indices = tf.stack([new_rows, new_cols], axis=-1)
        new_values = tf.gather(sp_tensor.values, gather_ids)

        new_shape = tf.concat([tf.expand_dims(tf.cast(num_ids, tf.int64), -1),
                               sp_tensor.dense_shape[1:]],
                              axis=-1)

        sp = tf.SparseTensor(new_indices, new_values, new_shape)
        return sp


def grid_2d(shape, name="grid_2d"):
    """ creates a tensor with a grid 2d coordinates
    
    Args:
        shape (`Tensor`): an Tensor of tf.int32 with a 2D shape for the grid
        name (`str`): grid_2d op name

    Returns:
        grid_coordinates (`Tensor`): 2D tensor with grid coordinates

    """
    shape = as_tensor(shape, tf.int32)
    with tf.name_scope(name):
        x = tf.range(shape[0])
        y = tf.range(shape[1])
        x = x[tf.newaxis, :, tf.newaxis]
        y = y[:, tf.newaxis, tf.newaxis]

        return tf.reshape(tf.concat([x + tf.zeros_like(y),
                                     tf.zeros_like(x) + y], axis=2), [-1, 2])


def sparse_tile(sp_tensor, num, name="sparse_tile"):
    """ Constructs a `SparseTensor` by replicating the input sparse tensor `num` times

    Args:
        sp_tensor (`SparseTensor`): a sparse input tensor to be tiled
        num (`int`): number of repetitions
        name (`str`): name for the op

    Returns:
        sp_tile (`SparseTensor`): result sparse tensor
    """
    with tf.name_scope(name):
        sp_tensor = as_tensor(sp_tensor)
        values = tf.tile(sp_tensor.values, [num])
        num = as_tensor(num, tf.int64)

        indices = tf.tile(sp_tensor.indices, [num, 1])
        row_indices, col_indices = tf.unstack(indices, num=2, axis=-1)

        # fix row indices
        num_values = tf.shape(sp_tensor.values, out_type=tf.int64)[0]
        batch_size = tf.shape(sp_tensor, out_type=tf.int64)[0]

        # this is preferable to using dense shape directly because we need the num cols to be known
        dim = sp_tensor.dense_shape[-1]
        offset = tf.range(start=0, limit=num * batch_size, delta=batch_size, dtype=tf.int64)

        row_offset = repeat(x=offset, n=num_values)
        row_indices = row_indices + row_offset
        indices = tf.stack([row_indices, col_indices], axis=-1)

        tile_batch_size = batch_size * num
        tiled_dense_shape = tf.stack([tile_batch_size, dim], axis=0)
        sp_tilled = tf.SparseTensor(indices=indices,
                                    values=values,
                                    dense_shape=tiled_dense_shape)

        return sp_tilled


def pairs(tensor1, tensor2, name="pairs"):
    """Pairwise combination of elements from the two tensors.

    Example:
        ```python
        t1 = [[0],[1]]
        t2 = [2,3,4]
        t12 = [[0,2],[1,2],[0,3],[1,3],[0,4],[1,4]]

        p12 = tx.pairs(t1,t2)
        tf.reduce_all(tf.equal(p12,t12))
        ```

    Args:
        tensor1 (`Tensor`): a tensor, python list, or numpy array
        tensor2 (`Tensor`): a tensor, python list, or numpy array
        name (`str`): name for pairs op)

    Returns:
        tensor (`Tensor`): a tensor with the pairwise combination of input tensors
    """
    tensor1 = tf.convert_to_tensor(tensor1)
    tensor2 = tf.convert_to_tensor(tensor2)

    with tf.name_scope(name):
        x, y = tf.meshgrid(tensor1, tensor2)
        result = tf.stack([x, y], axis=-1)
        result = tf.reshape(result, [-1, 2])
        return result


def sparse_put(sp_tensor, sp_updates, name="sparse_put"):
    """ sparse_put

    Changes a given tf.SparseTensor according to the updates specified in a tf.SparseTensor.

    Creates a new tensor where the values of the updates override the
    values in the original tensor. The input tensors must have the same
    `dense_shape`.

    Args:
        sp_tensor (`SparseTensor`): a sparse tensor we which to set some indices to given values
        sp_updates (`SparseTensor): a ``SparseTensor`` with the indices to be changed and the respective values
        name (`str`): sparse_put op name

    Returns:
        sparse_tensor (`SparseTensor`): a sparse tensor with the updated values.
    """
    with tf.name_scope(name=name):
        if sp_updates.dtype != sp_tensor.dtype:
            sp_updates = tf.cast(sp_updates, sp_tensor.dtype)

        # 1 concat indices and establish final tensor shape
        update_shape = tf.shape(sp_updates.values)
        zero_updates = tf.SparseTensor(sp_updates.indices,
                                       tf.zeros(update_shape, dtype=tf.float32),
                                       sp_updates.dense_shape)
        proto_result = tf.sparse.add(sp_tensor, zero_updates)

        # shape of resulting values tensor
        proto_shape = tf.shape(proto_result.values)

        # 2 get mask for input tensor
        proto_ones = tf.SparseTensor(proto_result.indices,
                                     tf.ones(proto_shape, tf.int32),
                                     proto_result.dense_shape)

        # mask_ones = tf.math.scalar_mul(-1, tf.ones(update_shape))
        sp_mask = tf.SparseTensor(sp_updates.indices,
                                  tf.ones_like(sp_updates.values, dtype=tf.int32) * -1,
                                  sp_updates.dense_shape)

        to_retain = tf.sparse.add(proto_ones, sp_mask)
        to_retain = tf.not_equal(to_retain.values, 0)

        # get tensor with masked values
        tensor_masked = tf.sparse.retain(proto_result, to_retain)

        # add values to entries previously set to 0
        new_tensor = tf.sparse.add(tensor_masked, sp_updates)
        return new_tensor


def put(tensor, sp_updates, name="put"):
    """ put

    Changes a given dense ``Tensor`` according to the updates specified in a ``SparseTensor``.

    Creates a new ``Tensor`` where the values of the updates override the
    values in the original tensor. The tensor `shape` must be the same as the updates `dense_shape`.

    Args:
        tensor (`Tensor`): tensor to be updated
        sp_updates (`SparseTensor`): sparse tensor with the indices to be changed and the respective values.
        name (`str`): put op name

    Returns:
        tensor (`Tensor`): a tensor with the updated values.
    """
    tensor = as_tensor(tensor)

    with tf.name_scope(name=name):
        if sp_updates.dtype != tensor.dtype:
            sp_updates = tf.cast(sp_updates, tensor.dtype)

        markers = tf.ones(shape=tf.shape(sp_updates.values))
        sparse_marker_tensor = tf.SparseTensor(indices=sp_updates.indices,
                                               values=markers,
                                               dense_shape=sp_updates.dense_shape)
        dense_update_marker = tf.sparse.to_dense(sparse_marker_tensor)
        dense_updates = tf.sparse.to_dense(sp_updates)

        new_tensor = tf.where(tf.not_equal(dense_update_marker, 0),
                              dense_updates,
                              tensor)
        return new_tensor


def filter_nd(condition, params, name="filter_nd"):
    """ filter_nd
    Filters a given tensor based on a condition tensor
    condition and params must have the same shape

    Args:
        condition (`Tensor`): a `bool` tensor used to filter params
        params (`Tensor`): the tensor to be filtered
        name (`str`): name for filter_nd op
    Returns:
        sp_tensor (`SparseTensor`): a sparse tensor with the values in params filtered according to condition
    """
    with tf.name_scope(name=name):
        indices = tf.cast(tf.where(condition), dtype=tf.int64)
        values = tf.gather_nd(params, indices)
        dense_shape = tf.cast(tf.shape(params), tf.int64)
        sp_result = tf.SparseTensor(indices, values, dense_shape)
        return sp_result


def sparse_overlap(sp_tensor1, sp_tensor2, name="sparse_overlap"):
    """ Returns a `SparseTensor` where the indices of the two tensors overlap returning a `SparseTensor`
    with the values of the first one

    Args:
        sp_tensor1 (`SparseTensor`): a `SparseTensor`
        sp_tensor2 (`SparseTensor`): another `SparseTensor`
        name (`str`): name for this op

    Returns:
        sp1 (`SparseTensor`): sparse tensor with the overlapping indices and values of the first tensor
    """
    with tf.name_scope(name):
        ones1 = sparse_ones(sp_tensor1.indices, sp_tensor1.dense_shape)
        ones2 = sparse_ones(sp_tensor2.indices, sp_tensor2.dense_shape)

        index_union = tf.sparse.add(ones1, ones2)

        index_filter = tf.math.equal(index_union.values, 2.)

        zeros1 = sparse_zeros(index_union.indices, index_union.dense_shape, sp_tensor1.values.dtype)
        expand1 = tf.sparse.add(zeros1, sp_tensor1)

        filtered = tf.sparse.retain(expand1, index_filter)
        return filtered


__all__ = [
    "matrix_indices",
    "empty_sparse_tensor",
    "sparse_ones",
    "sparse_zeros",
    "sparse_overlap",
    "apply_gate",
    "sparse_indices",
    "dense_one_hot",
    "sparse_matrix_indices",
    "dropout",
    "alpha_dropout",
    "sparse_dropout",
    "binary_random_mask",
    "SparseVariable",
    "to_sparse",
    "embedding_lookup_sparse",
    "sparse_overlap",
    "sort_by_first",
    "ranges",
    "grid_2d",
    "gather_sparse",
    "sparse_tile",
    "pairs",
    "sparse_put",
    "put",
    "filter_nd",
    "repeat"
]
