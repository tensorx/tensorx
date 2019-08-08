import tensorflow as tf
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
    """ converts a batch of column indices to 2d matrix indices, if the indices are out of order
    it and sorted is True, returns a batch of sorted matrix indices

    Args:
        index_tensor: a tensor with shape [b,n] with a batch of n column indices
        dtype: the out dtype for the indices
        sort_indices: if true sorts the indices on each row
        name: name for this op

    Returns:
         ``Tensor``: a tensor with (row,column) for each index in the input tensor.

    """
    with tf.name_scope(name=name):
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
        probability.get_shape().assert_is_compatible_with(tensor_shape.scalar())

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
    with tf.name_scope(name=name, values=[sp_tensor]):
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
    def __init__(self, initial_value: tf.SparseTensor, trainable=True, validate_shape=True, dtype=None):
        self.indices = tf.Variable(initial_value=initial_value.indices,
                                   trainable=False,
                                   dtype=tf.int64,
                                   shape=tf.TensorShape([None] * len(initial_value.shape[:-1]) +
                                                        [initial_value.indices.shape[-1]]),
                                   validate_shape=validate_shape)

        self.values = tf.Variable(initial_value=initial_value.values,
                                  dtype=dtype,
                                  trainable=trainable,
                                  validate_shape=validate_shape,
                                  shape=tf.TensorShape([None])
                                  )

        self.shape = tf.Variable(initial_value=initial_value.shape.as_list(),
                                 trainable=False,
                                 dtype=tf.int64,
                                 validate_shape=validate_shape,
                                 shape=tf.TensorShape([None]))

    def assign(self, sp_value):
        if len(sp_value.shape) != len(self.shape.value()):
            raise ValueError("cannot assign SparseTensor with Different dimensions")

        self.indices.assign(sp_value.indices)
        self.values.assign(sp_value.values)
        self.shape.assign(sp_value.dense_shape)

    def value(self):
        sp = tf.SparseTensor(self.indices, self.values, self.shape)
        return tf.sparse.reorder(sp)


__all__ = ["apply_gate",
           "sparse_ones",
           "sparse_indices",
           "sparse_one_hot",
           "matrix_indices",
           "dropout",
           "sparse_dropout",
           "binary_mask",
           "empty_sparse_tensor",
           "SparseVariable"]
