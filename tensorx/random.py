import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, random_ops, math_ops
from tensorflow.python.framework import ops, tensor_util, tensor_shape
from tensorflow.python.framework.sparse_tensor import SparseTensor

from tensorx.transform import enum_row


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
    """
    Samples using a uni
    Args:
        range_max:
        num_sampled:
        unique:
        seed:

    Returns:

    """
    if tensor_util.is_tensor(range_max):
        range_max = tensor_util.constant_value(range_max)

    if tensor_util.is_tensor(num_sampled):
        num_sampled = tensor_util.constant_value(num_sampled)

    if tensor_util.is_tensor(seed):
        seed = tensor_util.constant_value(seed)

    candidates, _, _ = tf.nn.uniform_candidate_sampler(
        [[0]],  # this used just for its shape     (num columns must match next arg)
        1,  # this is not used either (minimum 1 required)
        num_sampled,
        unique,
        range_max,
        seed,
    )
    return candidates


def sample(range_max, shape, unique=True, seed=None, name="sample"):
    """

    Args:
        range_max: an int32 or int64 scalar with the maximum range for the int samples
        shape: a 1-D integer Tensor or Python array. The shape of the output tensor
        unique: boolean
        seed: a python integer. Used to create a random seed for the distribution.
        See @{tf.set_random_seed} for behaviour

        name: a name for the operation (optional)
    Returns:
        A tensor of the specified shape filled with int values between 0 and max_range from the
        uniform distribution. If unique=True, samples values without repetition
    """
    with ops.name_scope(name):
        if tensor_util.is_tensor(shape):
            print(shape)
            shape = tensor_util.constant_value(shape)
            if shape is None:
                raise ValueError("Shape could not be converted to constant array")

        if len(shape) == 1 and shape[0] > 0:
            return _sample(range_max, shape[0], unique, seed)
        elif len(shape) == 2 and shape[0] > 0 and shape[1] > 0:
            num_sampled = shape[1]
            batch_size = shape[0]

            i = tf.range(0, batch_size)
            fn = lambda _: _sample(range_max, num_sampled)
            return tf.map_fn(fn, i, dtypes.int64)
        else:
            raise ValueError("Invalid Shape: expect 1-D tensor or array with positive dimensions")

def sample2(range_max, shape, unique=True, seed=None, name="sample"):
    """

    Args:
        range_max: an int32 or int64 scalar with the maximum range for the int samples
        shape: a 1-D integer Tensor or Python array. The shape of the output tensor
        unique: boolean
        seed: a python integer. Used to create a random seed for the distribution.
        See @{tf.set_random_seed} for behaviour

        name: a name for the operation (optional)
    Returns:
        A tensor of the specified shape filled with int values between 0 and max_range from the
        uniform distribution. If unique=True, samples values without repetition
    """
    with ops.name_scope(name):
        num_sampled = shape[1]
        batch_size = shape[0]

        i = tf.range(0, batch_size)
        fn = lambda _: _sample(range_max, num_sampled)
        return tf.map_fn(fn, i, dtypes.int64)



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
    indices = enum_row(flat_indices, dtype=dtypes.int64)

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

    indices = enum_row(samples, dtype=dtypes.int64)

    dense_shape = ops.convert_to_tensor(shape, dtype=dtypes.int64)

    return SparseTensor(indices=indices,
                        values=values,
                        dense_shape=dense_shape)
