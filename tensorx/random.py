import tensorflow as tf


def gumbel_top(logits, num_samples, dtype=tf.int32, seed=None):
    """ gumbel_top sampling

    uses the Gumbel-Top trick to sample without replacement from a discrete probability distribution
    parameterized by given (possibly unnormalized) log-probabilities `logits`.

    Args:
        logits (`Tensor`): log probabilities parameterizing the discrete distribution
        num_samples (int): number of unique samples to draw from the distribution
        dtype (`DType`): output dtype
        seed (int): random seed

    Returns:
        samples (int): a tensor with the indices sampled from the target distribution with shape
            `[shape(logits)[0],num_samples]`.
    """

    with tf.name_scope("gumbel_top"):
        shape = tf.shape(logits)
        u = tf.random.uniform(shape, minval=0, maxval=1, seed=seed)
        g = -tf.math.log(-tf.math.log(u))
        z = u + g
        _, indices = tf.nn.top_k(z, k=num_samples)
        if indices.dtype != dtype:
            indices = tf.cast(indices, dtype)
        return indices


__all__ = ["gumbel_top"]
