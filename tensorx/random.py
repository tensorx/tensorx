import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, random_ops, math_ops
from tensorflow.python.framework import ops, tensor_util, tensor_shape
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.transform import enum_row_v2


def _shape_tensor(shape, dtype=dtypes.int32):
    """Convert to an int32 or int64 tensor, defaulting to int32 if empty."""
    if isinstance(shape, (tuple, list)):
        shape = ops.convert_to_tensor(shape, dtype=dtype, name="shape")
    elif isinstance(shape, ops.Tensor) and dtype.is_compatible_with(dtype):
        shape = math_ops.cast(shape, dtype)
    else:
        shape = ops.convert_to_tensor(shape)
    return shape


def _sample(range_max, num_sampled, unique=True, seed=None):
    candidates, _, _ = tf.nn.uniform_candidate_sampler(
        [[0]],  # this used just for its shape     (num columns must match next arg)
        1,  # this is not used either (minimum 1 required)
        num_sampled,
        unique,
        range_max,
        seed,
    )
    return candidates


def sample(range_max, shape=[1], unique=True, seed=None, name="sample"):
    """
    Samples a set of integers from [0,range_max) using the uniform distribution
    if unique is True, it samples without replacement.

    None:
        although the true class parameter is not used, its shape is used
        so if we want to provide

    Args:

        dtype: type for values generated (default: int32) (change to int64 if needed)
        range_max: ints are sampled from [0,max_range)
        shape shape: A 1-D integer Tensor or Python array. The shape of the output tensor:
            note that if unique == True, n must be <= range_max
        unique: if True, samples without replacement
        seed: optional seed to be used with the uniform candidate sampler

    #TODO shape should be a tensor?

    Returns:
        a tensor with the shape [num_sampled] of type

    """
    with ops.name_scope(name):
        if len(shape) == 1 and shape[0] > 0:
            num_sampled = shape[0]
            return _sample(range_max, num_sampled, unique, seed)
        elif len(shape) == 2 and shape[0] > 0 and shape[1] > 0:
            num_sampled = shape[1]
            batch_size = shape[0]

            i = tf.range(0, batch_size)
            fn = lambda _: _sample(range_max, num_sampled)
            return tf.map_fn(fn, i, dtypes.int64)
        else:
            raise ValueError("Invalid Shape: expect a shape with rank 1 or 2 with positive dimensions")


def sparse_random_normal(dense_shape, density=0.1, mean=0.0, stddev=1, dtype=dtypes.float32, seed=None):
    """ Creates a NxM `SparseTensor` with `density` * NXM non-zero entries

    The values for the sparse tensor come from a random normal distribution.

    Args:
        dense_shape: a list of integers, a tuple of integers
        density: the proportion of non-zero entries, 1 means that all the entries have a value sampled from a
        random normal distribution.

        mean: normal distribution mean
        stddev: normal distribution standard deviation
        seed: the seed used for the random number generator
        seed: the seed used for the random number generator
    Returns:
        A `SparseTensor` with a given density with values sampled from the normal distribution
    """
    num_noise = int(density * dense_shape[1])
    noise_shape = [dense_shape[0], num_noise]

    flat_indices = sample(range_max=dense_shape[1], shape=noise_shape, unique=True, seed=seed)
    indices = enum_row_v2(flat_indices, dtype=dtypes.int64)

    value_shape = tensor_shape.as_shape([dense_shape[0] * num_noise])

    values = random_ops.random_normal(shape=value_shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

    # dense_shape = tensor_shape.as_shape(dense_shape)
    return SparseTensor(indices, values, dense_shape)


def salt_pepper_noise(shape, noise_amount=0.5, max_value=1, min_value=-1, seed=None, dtype=dtypes.float32):
    """ Creates a noise tensor with a given shape [N,M]


    Args:
        seed:
        dtype:
        shape:
        noise_amount: the amount of dimensions corrupted, 1.0 means every index is corrupted and set to
        one of two values: `max_value` or `min_value`
        max_value: the maximum noise constant (salt)
        min_value: the minimum noise constant (pepper)

    Returns:
        A noise tensor with the given shape where a given ammount
        (noise_amount * M) of indices is corrupted with
        salt or pepper values (max_value, min_value)

    """
    # for we corrupt (n_units * noise_amount) for each training example
    num_noise = noise_amount * shape[1]
    num_salt = num_noise // 2
    num_pepper = num_noise - num_salt

    noise_shape = [shape[0], num_noise]
    samples = sample(shape[1], shape=noise_shape, unique=True, seed=seed)

    # constant values to attribute to salt or pepper
    salt_tensor = array_ops.constant(max_value, dtype, shape=[noise_shape[0], num_salt])
    pepper_tensor = array_ops.constant(min_value, dtype, shape=[noise_shape[0], num_pepper])
    """
    [[1,1,-1,-1],
     [1,1,-1,-1]]
    ===================== 
    [1,1,-1,-1,1,1,-1,-1]
    """
    values = array_ops.concat([salt_tensor, pepper_tensor], axis=-1)
    values = array_ops.reshape(values, [-1])

    indices = enum_row_v2(samples, dtype=dtypes.int64)

    dense_shape = ops.convert_to_tensor(shape, dtype=dtypes.int64)

    return SparseTensor(indices=indices,
                        values=values,
                        dense_shape=dense_shape)
