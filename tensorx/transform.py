""" TensorFlow tensor Transformation

Utilities to create, convert between, and combine tensors
"""
import numbers

import numpy as np

from tensorx.utils import to_tensor_cast, complete_shape
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, tensor_util


def to_tensor(tensor, dtype=None):
    """
    to_tensor Converts the input to tensor or sparse tensor and casts to the given type
    
    Args:
        tensor (obj): an object convertible to tensor or sparse tensor (e.g. numpy array)
        dtype (dtype, optional): Defaults to None. dtype for the resulting tensor
    
    Returns:
        Tensor,SparseTensor: Tensor or SparseTensor
    """

    return to_tensor_cast(tensor, dtype)


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


def repeat(x, n, name="repeat"):
    """ Repeats the values of a tensor along the last dimension

    Args:
        x: ``Tensor``
        n: :obj:`int` number of repetitions of each element
        name: name for the operation

    Returns:
        A `Tensor` with shape [shape[:-1, ], shape[-1:, ] * n]
    """
    with tf.name_scope(name, values=[x, n]):
        x = tf.convert_to_tensor(x)
        n = to_tensor_cast(n, dtype=x.dtype)

        shape = tf.shape(x, out_type=x.dtype)
        flat_x = tf.reshape(x, [-1])

        rep_x = tf.tile(tf.expand_dims(flat_x, -1), tf.stack([1, n]))

        new_shape = tf.concat([shape[:-1, ], shape[-1:, ] * n], axis=-1)
        rep_x = tf.reshape(rep_x, new_shape)

    return rep_x


# TODO check boolean mask here and see if it can be replaced by dynamic partition
def repeat_each(x, repeats, name="repeat_each"):
    """ Repeats each element in x its corresponding number of repetitions


    Args:
        x: Tensor with the same shape as repeats
        repeats: Tensor with the same shape as x
        name: name for this op

    Returns:

    """
    with tf.name_scope(name, values=[x, repeats]):
        x = tf.convert_to_tensor(x)
        repeats = tf.convert_to_tensor(repeats)

        # get maximum repeat length in x
        maxlen = tf.math.reduce_max(repeats)

        # tile it to the maximum repeat length, it should be of shape [xlen, maxlen] now
        x_repeat = tf.stack([1, maxlen], axis=0)
        x_tiled = tf.tile(tf.expand_dims(x, 1), x_repeat)

        # create a sequence mask using x
        # this will create a boolean matrix of shape [xlen, maxlen]
        # where result[i,j] is true if j < x[i].
        mask = tf.sequence_mask(repeats, maxlen)

        # mask the elements based on the sequence mask
        return tf.boolean_mask(x_tiled, mask)


def sparse_tile(sp_tensor, num, name="sparse_tile"):
    with tf.name_scope(name, values=[sp_tensor, num]):
        sp_tensor = to_tensor_cast(sp_tensor)
        values = tf.tile(sp_tensor.values, [num])
        num = to_tensor_cast(num, tf.int64)

        indices = tf.tile(sp_tensor.indices, [num, 1])
        row_indices, col_indices = tf.unstack(indices, num=2, axis=-1)

        # fix row indices
        num_values = tf.shape(sp_tensor.values, out_type=tf.int64)[0]
        batch_size = tf.shape(sp_tensor, out_type=tf.int64)[0]

        # this is preferable to using dense shape directly because we need the num cols to be known
        dim = sp_tensor.get_shape().as_list()[-1]
        if dim is None:
            raise ValueError("Could not determine the last dimension of input sp_tensor")

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


def enum_each(enum_sizes, name="repeat_each"):
    """ creates an enumeration for each repeat
    and concatenates the results because we can't have
    a tensor with different row or column sizes

    Example:

        enum_each([1,2,4])

        Returns

        [0,0,0,1,0,1,2,3]

        the enums are [0], [0,1], [0,1,2,3]

    Args:
        enum_sizes: Tensor with the size for each enum
        name: name for this op

    Returns:
        A 1-D Tensor with reduce_sum(enum_sizes) dimension

    """
    with tf.name_scope(name, values=[enum_sizes]):
        enum_sizes = tf.convert_to_tensor(enum_sizes)
        num_enums = tf.shape(enum_sizes)[0]

        # get maximum repeat length in x
        maxlen = tf.math.reduce_max(enum_sizes)
        x = tf.range(maxlen)

        # tile it to the maximum repeat length, it should be of shape [maxlen x maxlen] now
        x_repeat = tf.stack([num_enums, 1], axis=0)
        x_tiled = tf.tile(tf.expand_dims(x, 0), x_repeat)

        # create a sequence mask using x
        # this will create a boolean matrix of shape [xlen, maxlen]
        # where result[i,j] is true if j < x[i].
        mask = tf.sequence_mask(enum_sizes, maxlen)

        # mask the elements based on the sequence mask
        return tf.boolean_mask(x_tiled, mask)


def grid(shape, name="grid"):
    with tf.name_scope(name):
        if len(shape) == 1:
            return tf.expand_dims(tf.range(0, shape[0], 1), -1)
        elif len(shape) == 2:
            max_x = shape[0]
            max_y = shape[1]

            ys = tf.range(0, max_y, 1)
            ys = tf.tile(ys, [max_x])
            ys = tf.reshape(ys, shape)

            xys = to_matrix_indices_2d(ys)
            return xys
        else:
            raise ValueError("Invalid shape: shape should have len 1 or 2")


def pairs(tensor1, tensor2, name="pairs"):
    """Pairwise combination of elements from the two tensors.

    Example::

        t1 = [[0],[1]]
        t2 = [2,3,4]
        t12 = [[0,2],[1,2],[0,3],[1,3],[0,4],[1,4]]

        tf.reduce_all(tf.equal(pairs(t1,t2),t12))

    Args:
        tensor1(``Tensor``): a tensor, python list, or numpy array
        tensor2(``Tensor``): a tensor, python list, or numpy array
        name: name for this operation (optional)

    Returns:
        ``Tensor``: a ``Tensor`` of rank 2
    """
    with tf.name_scope(name, values=[tensor1, tensor2]):
        tensor1 = tf.convert_to_tensor(tensor1)
        tensor2 = tf.convert_to_tensor(tensor2)

        x, y = tf.meshgrid(tensor1, tensor2)

        result = tf.stack([x, y], axis=-1)
        result = tf.reshape(result, [-1, 2])
        return result


def to_matrix_indices_2d(index_tensor, dtype=tf.int64, sort_indices=True, name="matrix_indices"):
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
    with tf.name_scope(name=name, values=[index_tensor]):
        index_tensor = to_tensor_cast(index_tensor, dtype)

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


def to_matrix_indices(tensor, dtype=tf.int64, name="matrix_indices"):
    """ Converts batches of column indices to batches of [row,column] indices

    For a given batch of indices of shape [n,m] or [b,n,m] this op outputs a 2-D ``Tensor``
    with the `row-index pairs`::

        [[r1,i1],[r1,i2],...,[r1,im],
         [r2,i1],[r2,i2],...,[r2,im],
         ...
         [rn,i1],[rn,i2],...,[rn,im]]


    Rank 2 Example::

            tensor = [[1,2],
                      [2,5]]


            batch_to_matrix(tensor)

            [[0,1],
             [0,2],
             [1,2],
             [1,5]]



    Rank 3 Example::

            tensor = [[[1,2],
                       [3,4]],

                       [5,6],
                       [7,8]]]

            batch_to_matrix(tensor)

             [[[0,1],
               [0,2],
               [1,3],
               [1,4]],

               [[0,5],
               [0,6],
               [1,7],
               [1,8]]]

    Use Case:
        Convert a batch of indices (used to slice another tensor with embedding lookup or gather)
        to be used in a tf.SparseTensor, so that we can change the weights of each slice.

    Args:
        dtype: int32 or int64, the output tensor type
        name: name for batch_to_matrix_indices op
        tensor: a ``Tensor`` with rank 2 or rank 3

        Returns:
            ``Tensor``: a tensor with (row,column) for each index in the input tensor. If the input is a tensor with
            rank 3 it outputs a rank 3 tensor. It considers the last two dimensions as the ones to be converted.

    """
    with tf.name_scope(name, values=[tensor]):
        tensor = to_tensor_cast(tensor, dtype)
        if tensor.dtype != tf.int32 and tensor.dtype != tf.int64:
            raise TypeError("Invalid tensor type: expected {t1} or {t2}, found {t3}".format(t1=tf.int32,
                                                                                            t2=tf.int64,
                                                                                            t3=tensor.dtype))

        static_shape = tensor.get_shape().with_rank_at_most(3)
        shape = tf.shape(tensor)
        rows = tf.range(tf.cast(shape[-2], tensor.dtype))

        # [0, 1] -> [[0,0,...],[1,1,...]] -> [0,0,...,1,1,...]
        multiples = tf.stack([1, shape[-1]])
        rows = tf.tile(tf.expand_dims(rows, -1), multiples)
        rows = tf.reshape(rows, [-1])

        if static_shape.ndims == 3:
            multiples = tf.stack([1, shape[-3]])
            rows = tf.tile(tf.expand_dims(rows, 0), multiples)
            rows = tf.reshape(rows, [-1])

        # determine shape for final reshape
        s = []
        if static_shape.ndims == 3:
            s.append(shape[0])
        s += [shape[-2] * shape[-1], 2]
        s = tf.stack(s)

        # align rows, transpose and reshape to the final shape
        flat_tensor = tf.reshape(tensor, shape=[-1])
        enum = tf.stack([rows, flat_tensor])
        enum = tf.transpose(enum)
        enum = tf.reshape(enum, shape=s)

        return enum


def sparse_put(sp_tensor, sp_updates, name="sparse_put"):
    """Changes a given tf.SparseTensor according to the updates specified in a tf.SparseTensor.

    Creates a new tensor where the values of the updates override the
    values in the original tensor. The input tensors must have the same
    ``dense_shape``.

    Args:
        sp_tensor: a ``SparseTensor`` we which to set some indices to given values
        sp_updates: a ``SparseTensor`` with the indices to be changed and the respective values
        name: name for this operation (optional)

    Returns:
        ``SparseTensor``: a ``SparseTensor`` with the updated values.
    """
    with tf.name_scope(name, values=[sp_tensor, sp_updates]):
        if sp_updates.dtype != sp_tensor.dtype:
            sp_updates = tf.cast(sp_updates, sp_tensor.dtype)

        # 1 concat indices and establish final tensor shape
        update_shape = tf.shape(sp_updates.values)
        zero_updates = tf.SparseTensor(sp_updates.indices,
                                       tf.zeros(update_shape, dtype=tf.float32),
                                       sp_updates.dense_shape)
        proto_result = tf.sparse_add(sp_tensor, zero_updates)

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

        to_retain = tf.sparse_add(proto_ones, sp_mask)
        to_retain = tf.math.not_equal(to_retain.values, 0)

        # get tensor with masked values
        tensor_masked = tf.sparse_retain(proto_result, to_retain)

        # add values to entries previously set to 0
        new_tensor = tf.sparse_add(tensor_masked, sp_updates)
        return new_tensor


def dense_put(tensor, sp_updates, name="dense_put"):
    """ Changes a given dense ``Tensor`` according to the updates specified in a ``SparseTensor``.

    Creates a new ``Tensor`` where the values of the updates override the
    values in the original tensor. The tensor `shape` must be the same as the updates `dense_shape`.

    Args:
        tensor: a ``Tensor`` we want to change.
        sp_updates: a ``SparseTensor`` with the indices to be changed and the respective values.
        name: the name for this operation (optional).

    Returns:
        ``Tensor``: a ``Tensor`` with the updated values.
    """
    with tf.name_scope(name, values=[tensor, sp_updates]):
        tensor = tf.convert_to_tensor(tensor)
        if sp_updates.dtype != tensor.dtype:
            sp_updates = tf.cast(sp_updates, tensor.dtype)

        markers = tf.ones(shape=tf.shape(sp_updates.values))
        sparse_marker_tensor = tf.SparseTensor(indices=sp_updates.indices, values=markers,
                                               dense_shape=sp_updates.dense_shape)
        dense_update_marker = tf.sparse.to_dense(sparse_marker_tensor)
        dense_updates = tf.sparse.to_dense(sp_updates)

        new_tensor = tf.where(tf.math.not_equal(dense_update_marker, 0),
                              dense_updates, tensor)
        return new_tensor


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
    with tf.name_scope(name, "dropout", [tensor]):
        tensor = tf.convert_to_tensor(tensor, name="x")
        if random_mask is not None:
            random_mask = to_tensor_cast(random_mask, tensor.dtype)

        if not tensor.dtype.is_floating:
            try:
                tensor = tf.cast(tensor, tf.float32)
            except Exception as e:
                raise ValueError("x has to be a floating point tensor since it might be scaled"
                                 "Got a %s tensor instead. and could not cast it" % tensor.dtype)

        if isinstance(probability, numbers.Real) and not 0 <= probability < 1:
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

        if tf.executing_eagerly():
            if isinstance(probability, ops.EagerTensor):
                if probability.numpy() == 0:
                    if return_mask:
                        return tensor, None
                    else:
                        return tensor
                elif probability.numpy() == 1:
                    zeros = tf.zeros_like(tensor)
                    if return_mask:
                        return zeros, None
                    else:
                        return zeros
        else:
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
            with tf.name_scope("random_mask"):
                keep_prob = 1 - probability
                random_state = tf.random_uniform(noise_shape, seed=seed, dtype=tensor.dtype)
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


def zoneout(tensor, zoneout_tensor, noise_shape=None, probability=0.1, seed=None, name="dropout"):
    """
    """
    with tf.name_scope(name, "dropout", [tensor]):
        tensor = tf.convert_to_tensor(tensor, name="x")
        if not tensor.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % tensor.dtype)
        if isinstance(probability, numbers.Real) and not 0 <= probability < 1:
            raise ValueError("dropout probability must be a scalar tensor or a float in the "
                             "range [0, 1), got %g" % probability)

        # Early return if nothing needs to be dropped.
        if isinstance(probability, float) and probability == 0:
            return tensor
        if tf.executing_eagerly():
            if isinstance(probability, ops.EagerTensor):
                if probability.numpy() == 0:
                    return tensor
        else:
            probability = tf.convert_to_tensor(
                probability, dtype=tensor.dtype, name="drop_probability")
            probability.get_shape().assert_is_compatible_with(tensor_shape.scalar())

            # Do nothing if we know keep_prob == 1
            if tensor_util.constant_value(probability) == 0:
                return tensor

        noise_shape = _get_noise_shape(tensor, noise_shape)

        keep_prob = 1 - probability
        random_tensor = keep_prob + tf.random_uniform(noise_shape, seed=seed, dtype=tensor.dtype)
        mask = tf.math.floor(random_tensor)

        kept = tensor * (1 - mask)
        zoned = zoneout_tensor * mask
        ret = kept + zoned

        if not tf.executing_eagerly():
            ret.set_shape(tensor.get_shape())
        return ret


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
    with tf.name_scope(name, values=[sp_tensor]):
        dense_shape = sp_tensor.dense_shape

        if not sp_tensor.values.dtype.is_floating:
            raise ValueError("sp_tensor has to be a floating point tensor since its values are going to"
                             " be scaled. Got a %s tensor instead." % sp_tensor.dtype)
        if isinstance(probability, numbers.Real) and not 0 <= probability < 1:
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


def sparse_ones(matrix_indices, dense_shape, dtype=tf.float32, name="sparse_ones"):
    """ Creates a new ``SparseTensor`` with the given indices having value 1

    Args:
        matrix_indices: a rank 2 ``Tensor`` with the (row,column) indices for the resulting sparse tensor
        dense_shape: the ``SparseTensor`` dense shape
        dtype: the tensor type for the values
        name: name for this op

    Returns:
        ``SparseTensor``: a new tf.SparseTensor with the values set to 1.
    """
    with tf.name_scope(name, values=[matrix_indices, dense_shape]):
        matrix_indices = to_tensor_cast(matrix_indices, tf.int64)
        dense_shape = to_tensor_cast(dense_shape, tf.int64)
        indices_shape = complete_shape(matrix_indices)
        values = tf.ones([indices_shape[0]], dtype)
        return tf.SparseTensor(matrix_indices, values, dense_shape)


def sparse_zeros(matrix_indices, dense_shape, dtype=tf.float32, name="sparse_zeros"):
    """ Creates a new ``SparseTensor`` with the given indices having value 1

    Args:
        matrix_indices: a rank 2 ``Tensor`` with the indices for the resulting sparse tensor
        dense_shape: the ``SparseTensor`` dense shape
        dtype: the tensor type for the values
        name: name for this op

    Returns:
        ``SparseTensor``: a new tf.SparseTensor with the values set to 1.
    """
    with tf.name_scope(name, values=[matrix_indices, dense_shape]):
        matrix_indices = to_tensor_cast(matrix_indices, tf.int64)
        dense_shape = to_tensor_cast(dense_shape, tf.int64)
        indices_shape = complete_shape(matrix_indices)
        values = tf.zeros([indices_shape[0]], dtype)
        return tf.SparseTensor(matrix_indices, values, dense_shape)


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
    with tf.name_scope(name, values=[column_indices, num_cols]):
        column_indices = to_tensor_cast(column_indices, tf.int64)
        matrix_indices = to_matrix_indices_2d(column_indices, dtype=tf.int64)

        dense_shape = tf.cast([tf.shape(column_indices)[0], num_cols], dtype=tf.int64)

        return sparse_ones(matrix_indices, dense_shape, dtype)


def dense_one_hot(column_indices, num_cols, dtype=tf.float32, name="dense_one_hot"):
    """Transforms a batch of indices to a dense ``Tensor`` by adding the `one-hot` encoding for each index.

    Example::

        indices = [[0],[1]]
        dense_shape = [2,2]

        dense_one_hot = [[1,0],[0,1]]

    Args:
        column_indices: a dense ``Tensor`` with the active indices for each sample (row).
        num_cols: number of columns for the one-hot encoding
        dtype: the type for the output tensor.
        name: name for this op

    Returns:
        ``Tensor``: A dense ``Tensor`` with a `one-hot encoding` for the given indices.
    """
    with tf.name_scope(name, values=[column_indices, num_cols]):
        column_indices = to_tensor_cast(column_indices, tf.int64)
        one_hot_dense = tf.one_hot(column_indices, depth=num_cols, dtype=dtype)

        if column_indices.get_shape().ndims == 2:
            one_hot_dense = tf.math.reduce_sum(one_hot_dense, axis=1)

        return one_hot_dense


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
    with tf.name_scope(name, values=[sp_values]):
        if len(sp_values.get_shape().dims) == 1:
            [flat_indices] = tf.unstack(sp_values.indices, num=1, axis=-1)
        else:
            _, flat_indices = tf.unstack(sp_values.indices, num=2, axis=-1)
        sp_indices = tf.SparseTensor(sp_values.indices, flat_indices, sp_values.dense_shape)

        return sp_indices


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
    with tf.name_scope(name, values=[tensor]):
        indices = tf.where(tf.math.not_equal(tensor, 0))
        dense_shape = tf.shape(tensor, out_type=tf.int64)

        values = tf.gather_nd(tensor, indices)
        sp_tensor = tf.SparseTensor(indices, values, dense_shape)

        return sp_tensor


def sparse_tensor_value_one_hot(indices, dense_shape):
    """ Converts a python or numpy array of indices to a ``SparseTensorValue``.

    Example:

    ..transforms a python list of indices::

        idx =[[0,5],[0,2],[1,7]]

    into a ``SparseTensorValue`` as follows::

        tf.SparseTensorValue(indices=[[0,0],[0,5],[1,0],[1,2],[1,7],[2,1]],
                          values=[1,1,1,1,1,1],
                          dense_shape=[3,10])

    this can be then fed to a ``tf.sparse_placeholder``

    Args:
        indices: a 2-D list or array of indices
        dense_shape: a python list or tuple with the shape for the ``SparseTensorValue``,
                     typically ``[BATCH_SIZE, MAX_INDEX]``.

    Raises:
        ``ValueError`` exception if any index ``i`` in the list ``value >= shape[1]``

    Returns:
        ``SparseTensorValue``: a tf.SparseTensorValue with the sparse indices and the indices for each row as values.

    """
    idx = []
    for row, indexes in enumerate(indices):
        for i in indexes:
            if i >= dense_shape[1]:
                raise ValueError("Invalid shape: index value {} >= {}".format(i, dense_shape[1]))
            idx.append([row, i])
    idx = np.array(idx)
    values = np.ones([len(idx)])

    return tf.SparseTensorValue(indices=idx, values=values, dense_shape=dense_shape)


def filter_nd(condition, params, name="filter_nd"):
    """ Filters a given tensor based on a condition tensor

    condition and params must have the same shape

    Args:
        condition: a boolean tensor used to filter params
        params: the tensor to be filtered
    Returns:
        ``SparseTensor``: a `SparseTensor` with the values in params filtered according to condition
    """
    with tf.name_scope(name, [condition, params]):
        indices = tf.cast(tf.where(condition), dtype=tf.int64)
        values = tf.gather_nd(params, indices)
        dense_shape = tf.cast(tf.shape(params), tf.int64)
        sp_result = tf.SparseTensor(indices, values, dense_shape)
        return sp_result


def gather_sparse(sp_tensor, ids, name="gather_sparse_v2"):
    """ gather_sparse

    Warning:
        very inefficient for obvious reasons.

    Performs gather on sparse tensors.

    Example:
        gather_sparse(sp_tensor,[1,1,4])

        returns a [3,sp_tensor.dense_shape[-1]] ``SparseTensor``


    Args:
        sp_tensor: a ``SparseTensor`` with the rows from which we will slice
        ids: ``Tensor`` with row ids to be gathered from the sparse tensor
        name: name for this op

    Returns:
        a ``SparseTensor`` with a number of rows equal to the number of ids to be gathered.

    """
    with tf.name_scope(name, [sp_tensor, ids]):
        ids = tf.cast(ids, tf.int64)
        ids = tf.reshape(ids, [-1])

        # count columns and compute row coordinates
        sp_column_ones = sparse_ones(sp_tensor.indices, sp_tensor.dense_shape, dtype=tf.int64)
        col_count = tf.sparse_reduce_sum(sp_column_ones, axis=-1)
        # sparse_reduce_sum sets shape to unknown
        col_count.set_shape([sp_tensor.get_shape().as_list()[0]])
        col_count_cs = tf.math.cumsum(col_count)
        row_start_coor = col_count_cs - col_count

        g_col_count = tf.gather(col_count, ids)
        g_row_start_coor = tf.gather(row_start_coor, ids)

        row_start_coor = repeat_each(g_row_start_coor, g_col_count)
        # col_counts = repeat_each(g_col_count, g_col_count)

        offset = enum_each(g_col_count)

        # use modular arithmetic to make sure we get incremental coordinates
        # gather_ids = row_start_coor + offset % col_counts
        gather_ids = row_start_coor + offset

        num_ids = tf.cast(tf.shape(ids)[0], tf.int64)
        new_rows = repeat_each(tf.range(num_ids), g_col_count)

        sp_cols = sp_tensor.indices[:, -1]
        new_cols = tf.gather(sp_cols, gather_ids)
        new_indices = tf.stack([new_rows, new_cols], axis=-1)
        new_values = tf.gather(sp_tensor.values, gather_ids)

        new_shape = tf.concat([tf.expand_dims(tf.cast(num_ids, tf.int64), -1),
                               sp_tensor.dense_shape[1:]],
                              axis=-1)

        sp = tf.SparseTensor(new_indices, new_values, new_shape)
        return sp


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
    with tf.name_scope(name, [sp_tensor1, sp_tensor2]):
        ones1 = sparse_ones(sp_tensor1.indices, sp_tensor1.dense_shape)
        ones2 = sparse_ones(sp_tensor2.indices, sp_tensor2.dense_shape)

        index_union = tf.sparse_add(ones1, ones2)

        index_filter = tf.math.equal(index_union.values, 2.)

        zeros1 = sparse_zeros(index_union.indices, index_union.dense_shape, sp_tensor1.values.dtype)
        expand1 = tf.sparse_add(zeros1, sp_tensor1)

        filtered = tf.sparse_retain(expand1, index_filter)
        return filtered


def flatten(tensor):
    return tf.reshape(tensor, [-1])


def sort_by_first(tensor1, tensor2, ascending=True, name="sort_by_first"):
    """ Sorts two tensors by the first tensor

    Args:
        tensor1: tensor to determine the oder by which the second is sorted
        tensor2: tensor to be sorted according to the sorting of the first
        ascending: if True sorts by ascending order of value
        name: name of the op

    Returns:
        tensor1, tensor2 sorted according to the

    """

    with tf.name_scope(name, values=[tensor1, tensor2]):
        tensor1 = to_tensor_cast(tensor1)
        tensor2 = to_tensor_cast(tensor2)

        sorted_tensor1, sorted_tensor1_indices = tf.nn.top_k(tensor1, k=tf.shape(tensor1)[-1])
        if ascending:
            sorted_tensor1 = tf.reverse(sorted_tensor1, axis=[-1])
            sorted_tensor1_indices = tf.reverse(sorted_tensor1_indices, axis=[-1])
        sorted_tensor1_indices = to_matrix_indices_2d(sorted_tensor1_indices, sort_indices=False)

        sorted_values = tf.gather_nd(tensor2, sorted_tensor1_indices)
        sorted_values = tf.reshape(sorted_values, tf.shape(tensor2))

        return sorted_tensor1, sorted_values


__all__ = ["sparse_overlap",
           "empty_sparse_tensor",
           "to_sparse",
           "gather_sparse",
           "to_matrix_indices",
           "dense_one_hot",
           "sparse_one_hot",
           "dense_put",
           "sparse_put",
           "sparse_tensor_value_one_hot",
           "sparse_indices",
           "sparse_ones",
           "dropout",
           "sparse_dropout",
           "pairs",
           "grid",
           "filter_nd",
           "repeat",
           "repeat_each",
           "sort_by_first",
           "sparse_tile",
           "to_matrix_indices_2d"
           ]
