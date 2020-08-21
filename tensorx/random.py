import tensorflow as tf
from typing import Optional, List
from tensorx.utils import as_tensor
from tensorx.math import logit


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


def bernoulli(shape, prob=0.5, seed=None, dtype=tf.float32, name="random_bernoulli"):
    """ Random bernoulli tensor.

    A bernoulli tensor sampled from a random uniform distribution.

    !!! info
        The Bernoulli distribution takes the value _1_ with probability p and the value 0 with probability _q = 1 − p_

    Args:
        shape (`List[int]`): output tensor shape
        prob (`float`): probability of each element being 1
        seed (`int`): A Python `int`. Used in combination with `tf.random.set_seed` to create a reproducible sequence of
            tensors across multiple calls.
        dtype (`DType`): output tensor type
        name (`Optional[str]`): name for random_bernoulli op

    Returns:
        tensor (`Tensor`): tensor with the given input shape
    """
    with tf.name_scope(name=name):
        # uniform [prob, 1.0 + prob)
        random_tensor = prob
        random_tensor += tf.random.uniform(
            shape, seed=seed, dtype=dtype)
        return tf.math.floor(random_tensor)


def sample_sigmoid(logits, n, dtype=None, seed=None, name="sample_sigmoid"):
    """ sample_sigmoid

    Efficient sampling Bernoulli random variable from a sigmoid defined distribution

    !!! info
        This can be applied to the output layer of a neural net if this represents a bernoulli
        distribution defined using a parameterized sigmoid-activated layer

    Args:
        logits (`Tensor`): logits
        n (`int`): number of samples per row of logits
        dtype (`tf.DType`): input Tensor dtype
        seed (`int`): random number generator seed
        name (`Optional[str]`): name for sample sigmoid op (optional)

    Returns:
        samples (`Tensor`): a tensor with samples
    """
    with tf.name_scope(name):
        logits = as_tensor(logits)
        if dtype is not None:
            logits = tf.cast(logits, dtype)
        n = as_tensor(n)

        shape = tf.shape(logits)
        sample_shape = tf.concat([[n], shape], axis=-1)

        uniform_sample = tf.random.uniform(sample_shape,
                                           minval=0,
                                           maxval=1,
                                           dtype=logits.dtype, seed=seed)
        z = logit(uniform_sample, dtype=logits.dtype)

        return tf.cast(tf.greater(logits, z), tf.float32)


__all__ = [
    "gumbel_top",
    "bernoulli",
    "sample_sigmoid"
]
