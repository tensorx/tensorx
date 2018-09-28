""" Loss Functions

To be used with to optimise neural network models, some of these are forwards from the `TensorFlow` API with some
additional documentation.
"""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.ops import array_ops, sparse_ops
from tensorflow.python.ops import math_ops, variables
from tensorflow.python.ops.losses.losses import mean_squared_error
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow.python.ops.nn import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.nn import embedding_lookup_sparse, embedding_lookup
from tensorflow.python.ops.nn import embedding_lookup
from tensorflow.python.ops import candidate_sampling_ops as candidate_sampling
from tensorflow.python.ops.losses.losses import hinge_loss as tf_hinge_loss
from tensorflow.python.ops.candidate_sampling_ops import uniform_candidate_sampler as uniform_sampler
import tensorx.transform as tx_fn
import tensorx.random as tx_rnd
from tensorflow.python.ops import random_ops


def mse(labels, predictions, weights=1.0):
    """ Mean Squared Error (MSE)

    Measures the average of the squares of the errors - the difference between an estimator and what is estimated.
    This is a risk function, corresponding to the expected value of the quadratic loss. Like variance, mean squared
    error has the disadvantage of heavily weighting outliers.


    Args:
        weights: Optional `Tensor` whose rank is either 0, or the same rank as `labels`, and must be broadcastable to
        `labels` (i.e., all dimensions must be either `1`, or the same as the corresponding `losses` dimension).
        predictions: a tensor with the estimated target values
        labels: ground truth, correct values

    Returns:
        ``Tensor``: a float ``Tensor``.

    """
    return mean_squared_error(labels, predictions, weights)


def binary_cross_entropy(labels, logits, name="binary_cross_entropy"):
    """ Binary Cross Entropy

    Measures the probability error in discrete binary classification tasks in which each class is independent and
    not mutually exclusive. The cross entropy between two distributions p and q is defined as:


    Warning:
        This is to be used on the logits of a model, not on the predicted labels.
        See ``tf.nn.sigmoid_cross_entropy_with_logits``.

    Args:
        labels: ground truth, correct values
        logits: a tensor with the unscaled log probabilities used to predict the labels with sigmoid(logits)


    Returns:
        ``Tensor``: a float ``Tensor``.

    """
    return sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)


def categorical_cross_entropy(labels, logits, dim=-1):
    """ Categorical Cross entropy

    Measures the probability error in discrete classification tasks in which the classes are mutually exclusive.

    Warning:
        This is to be used on the logits of a model, not on the predicted labels.
        See ``tf.nn.softmax_cross_entropy_with_logits``.

    Args:
        labels: ground truth, correct values with a one-hot encoding. Each row labels[i] must be a valid probability
        distribution.
        logits: a tensor with the unscaled log probabilities used to predict the labels with softmax(logits)
        dim: The class dimension. Defaulted to -1 which is the last dimension.
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).

    Returns:
        ``Tensor``: a float ``Tensor``.

    """
    return softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, dim=dim)


def binary_hinge(labels, logits, weights=1.0):
    """ Binary Hinge Loss

    Measures the classification error for maximum-margin classification. Margin classifiers like
    Support Vector Machines (SVM) maximise the distance between the closest examples and the decision boundary
    separating the binary classes.
    The hinge loss is defined as:

    .. math::

        \ell(y) = \max(0, 1-t \cdot y),

    where :math:`t` is the intended output (labels) and :math:`y` is the `raw` output (logits) from the
    classification decision function, not the predicted class label.

    Args:
        labels: The ground truth output tensor with values 0.0 or 1.0. Its shape should match the shape of logits.
        logits: The unscaled log probabilities, a float tensor.
        weights: Optional Tensor whose rank is either 0, or the same rank as labels, and must be broadcastable to
        labels (i.e., all dimensions must be either 1, or the same as the corresponding losses dimension).

    Returns:
        ``Tensor``: a float ``Tensor``.
    """
    return tf_hinge_loss(labels, logits, weights)


def sparsemax_loss(logits, sparsemax, labels, name="sparsemax_loss"):
    """Sparsemax loss function.

    References:
        [1]: From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
        https://arxiv.org/abs/1602.02068

    Args:
      logits: A `Tensor`, before sparsemax is applied
      sparsemax: A `Tensor` resulting from applying a sparsemax activation. Must have the same type as `logits`.
      labels: A `Tensor`. Must have the same type as `logits`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `logits`.
    """

    with ops.name_scope(name, [logits, sparsemax, labels]):
        logits = ops.convert_to_tensor(logits)
        sparsemax = ops.convert_to_tensor(sparsemax)
        labels = ops.convert_to_tensor(labels, name="labels")

        shifted_logits = logits - math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

        # sum over support (support = predicted labels)
        support = math_ops.cast(sparsemax > 0, sparsemax.dtype)
        sum_s = support * sparsemax * (shifted_logits - 0.5 * sparsemax)

        # - z_k + ||q||^2
        q_part = labels * (0.5 * labels - shifted_logits)

        return math_ops.reduce_sum(sum_s + q_part, axis=1)


def _sum_rows(x):
    with ops.name_scope("row_sum"):
        """Returns a vector summing up each row of the matrix x."""
        # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
        # a matrix.  The gradient of _sum_rows(x) is more efficient than
        # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
        # we use _sum_rows(x) in the nce_loss() computation since the loss
        # is mostly used for training.
        cols = array_ops.shape(x)[1]
        ones_shape = array_ops.stack([cols, 1])
        ones = array_ops.ones(ones_shape, x.dtype)
        return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def nce_loss(labels,
             model_prediction,
             weights,
             bias,
             scaling_var,
             num_samples,
             num_classes,
             num_true=1,
             class_sampler=None,
             labels_to_features=None):
    if class_sampler is None:
        class_sampler = uniform_sampler(
            true_classes=labels,
            num_true=num_true,
            num_sampled=num_samples,
            unique=True,
            range_max=num_classes,
            seed=None)

    labels_flat = array_ops.reshape(labels, [-1])

    # label_sample_prob is the probability of label to appear in the sampled set
    # sampled_classes_prob is the probability of each sampled class to appear in the sampled_classes
    sampled_classes, label_sample_prob, sampled_classes_prob = (
        array_ops.stop_gradient(s) for s in class_sampler)
    all_ids = array_ops.concat([labels_flat, sampled_classes], 0)

    all_ids = labels_to_features(all_ids)
    if isinstance(all_ids, SparseTensor):
        sp_values = all_ids
        sp_indices = tx_fn.sparse_indices(sp_values)

        all_w = embedding_lookup_sparse(
            params=weights,
            sp_ids=sp_indices,
            sp_weights=sp_values,
            combiner="sum",
            partition_strategy="mod")

        if bias is not None:
            all_b = embedding_lookup_sparse(
                params=bias,
                sp_ids=sp_indices,
                sp_weights=sp_values,
                combiner="sum",
                partition_strategy="mod")

    else:
        all_w = embedding_lookup(
            params=weights,
            ids=all_ids,
            partition_strategy="mod")

        if bias is not None:
            all_b = embedding_lookup(
                params=bias,
                ids=all_ids,
                combiner="sum",
                partition_strategy="mod")

    # true_w shape is [batch_size * num_true, m] with m being the weights dim
    true_w = array_ops.slice(all_w, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))
    sampled_w = array_ops.slice(all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])

    # [batch_size, num_samples]
    sampled_logits = math_ops.matmul(model_prediction, sampled_w, transpose_b=True)

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(model_prediction, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

    if bias is not None:
        true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
        true_b = array_ops.reshape(true_b, [-1, num_true])
        sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

        true_logits += true_b
        sampled_logits += sampled_b

    # add trainable scaling var
    true_logits += scaling_var
    sampled_logits += scaling_var

    # subtract log(k*p_noise(w_j))
    true_logits -= math_ops.log(label_sample_prob)
    sampled_logits -= math_ops.log(sampled_classes_prob)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
    ], 1)

    return binary_cross_entropy(labels=out_labels, logits=out_logits)


def sparse_cnce_loss(labels,
                     model_prediction,
                     weights,
                     num_classes,
                     num_samples=1,
                     noise_ratio=0.1,
                     labels_to_sparse_features=None):
    """
    Sparse Conditional Noise-Contrastive Estimation

    A variation of Conditional Noise-Contrastive Estimation where
    we use sparse additive symmetric noise sampled from an hypercube

    References:
        Ceylan, Gutmann 2018 - Conditional Noise-Contrastive Estimation of Unnormalised Models
        https://arxiv.org/abs/1806.03664

    Args:
        labels_to_sparse_features: function that transforms label indices into sparse tensor values possibly
        with multiple features

    """
    labels_flat = array_ops.reshape(labels, [-1])

    labels_tile = array_ops.tile(labels_flat, [num_samples])
    features_tile = labels_to_sparse_features(labels_tile)

    features = labels_to_sparse_features(labels_flat)
    if not isinstance(features, SparseTensor):
        raise TypeError("labels_to_sparse_features did not convert labels to a SparseTensor")

    batch_size = array_ops.shape(labels_flat)[0]
    dim = features.get_shape().as_list()[-1]

    noise = tx_rnd.sparse_random_mask(dim=dim,
                                      batch_size=batch_size * num_samples,
                                      density=noise_ratio,
                                      mask_values=[-1, 1],
                                      symmetrical=True,
                                      dtype=dtypes.float32)

    # noise_features = sparse_ops.sparse_add(math_ops.cast(features_tile, dtypes.float32), noise)
    # noise_features = noise
    noise_features = SparseTensor(indices=noise.indices,
                                  values=noise.values * random_ops.random_normal(array_ops.shape(noise.values)),
                                  dense_shape=noise.dense_shape)

    true_w = embedding_lookup_sparse(
        params=weights,
        sp_ids=tx_fn.sparse_indices(features_tile),
        sp_weights=features_tile,
        combiner="sum",
        partition_strategy="mod")

    noise_w = embedding_lookup_sparse(
        params=weights,
        sp_ids=tx_fn.sparse_indices(noise_features),
        sp_weights=noise_features,
        combiner="sum",
        partition_strategy="mod")

    # p_m(y=1|m)
    true_logits = math_ops.matmul(model_prediction, true_w, transpose_b=True)
    noise_logits = math_ops.matmul(model_prediction, noise_w, transpose_b=True)

    true_logits = math_ops.exp(true_logits)
    noise_logits = num_samples * math_ops.exp(noise_logits)

    # true_logits = math_ops.exp(true_logits)
    # noise_logits = math_ops.exp(noise_logits)

    true_logits = math_ops.log(1 / (1 + noise_logits / true_logits))
    noise_logits = math_ops.log(1 / (1 + true_logits / noise_logits))

    true_logits = true_logits - noise_logits
    noise_logits = noise_logits - true_logits
    # true_logits = true_logits - noise_logits
    # noise_logits = noise_logits - true_logits

    # this returns a smooth curve to perplexity 3, for 3 classes it should be 1 because I'm not repeating
    # loss = -2*math_ops.reduce_mean(math_ops.log(1 + math_ops.exp(true_logits + noise_logits)))
    # loss = -2*math_ops.reduce_mean(math_ops.log(1 + math_ops.exp(true_logits + noise_logits)))
    # loss = -math_ops.reduce_mean(math_ops.log(math_ops.exp(true_logits + noise_logits)))
    # noise_loss = -2 * math_ops.reduce_sum(math_ops.log(1 + math_ops.exp(-true_logits + num_samples*noise_logits)))
    # also reduces to a smooth curve
    # loss = -2*math_ops.reduce_mean(math_ops.log(1+math_ops.exp(true_logits + noise_logits)))
    # loss = -2 * math_ops.reduce_mean(math_ops.log(math_ops.exp(1 + math_ops.log(true_logits)-math_ops.log(noise_logits))))

    # loss = nce_loss(labels, model_prediction, weights, None, 1, num_samples, num_classes,
    #                labels_to_features=labels_to_sparse_features)

    # return loss + 0.1 * noise_loss

    # return loss
    out_logits = array_ops.concat([true_logits, noise_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits),
        array_ops.zeros_like(noise_logits)
    ], 1)

    return binary_cross_entropy(labels=out_labels, logits=out_logits)


__all__ = ["mse",
           "sparsemax_loss",
           "binary_cross_entropy",
           "categorical_cross_entropy",
           "binary_hinge",
           "sparse_cnce_loss"]
