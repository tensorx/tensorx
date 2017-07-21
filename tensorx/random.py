import tensorflow as tf


def _sample(range_max, num_sampled, unique=True, seed=None):
    candidates, _, _ = tf.nn.uniform_candidate_sampler(
        [[0]],  # this used just for its shape     (num columns must match next arg)
        1,  # this is not used either (minimum 1 required)
        num_sampled,
        unique,
        range_max, seed,
    )
    return candidates


def sample(range_max, shape=[1], unique=True, seed=None):
    """
    Samples a set of integers from [0,range_max) using the uniform distribution
    if unique is True, it samples without replacement.

    None:
        although the true class parameter is not used, its shape is used
        so if we want to provide

    Args:

        range_max: ints are sampled from [0,max_range)
        shape:
            - if 1D (e.g. [n]) returns a 1D tensor with a shape [1] and n sampled integers
            - if 2D (e.g. [batch_size,n]) returns a rank 2 tensor with batch_size independent samples of n integers
            note that if unique == True, n must be <= range_max
        unique: if True, samples without replacement
        seed: optional seed to be used with the uniform candidate sampler

    #TODO shape should be a tensor?

    Returns:
        a tensor with the shape [num_sampled] of type

    """
    if len(shape) == 1 and shape[0] > 0:
        num_sampled = shape[0]
        return _sample(range_max, num_sampled, unique, seed)
    elif len(shape) == 2 and shape[0] > 0 and shape[1] > 0:
        num_sampled = shape[1]
        batch_size = shape[0]

        i = tf.range(0, batch_size, dtype=tf.int64)
        fn = lambda i: _sample(range_max, num_sampled)
        return tf.map_fn(fn, i)
    else:
        raise ValueError("Invalid Shape: expect a shape with rank 1 or 2 with positive dimensions")

    return sample
