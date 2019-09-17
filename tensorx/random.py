import tensorflow as tf


def gumbel_sample(range_max, num_samples, batch_size=None, dtype=tf.int64, seed=None):
    """ samples without replacement using a gumbel distribution

    Args:
        range_max: maximum value for the range from which to sample
        num_samples: number of unique samples to draw from the range
        batch_size: number of independent samples
        dtype: output dtype

    Returns:
        a tensor with type ``dtype`` with shape [batch_size,num_samples]
    """
    if batch_size is None:
        shape = [range_max]
    else:
        shape = [batch_size, range_max]
    u1 = tf.random.uniform(shape, minval=0, maxval=1, seed=seed)
    u2 = tf.random.uniform(shape, minval=0, maxval=1, seed=seed)
    gumbel = -tf.math.log(-tf.math.log(u1))
    dist = u2 + gumbel
    _, indices = tf.nn.top_k(dist, k=num_samples)
    if indices.dtype != dtype:
        indices = tf.cast(indices, dtype)
    return indices


__all__ = ["gumbel_sample"]
