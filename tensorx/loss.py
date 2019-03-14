""" Loss Functions

To be used with to optimise neural network models, some of these are forwards from the `TensorFlow` API with some
additional documentation.
"""

from tensorx.math import embedding_lookup_sparse
import tensorx.random as tx_rnd

import tensorflow as tf
from tensorflow.python.ops.nn import uniform_candidate_sampler as uniform_sampler
import tensorx.transform as txf


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
    return tf.losses.mean_squared_error(labels, predictions, weights)


def binary_cross_entropy(labels, logits, name="binary_cross_entropy"):
    """ Binary Cross Entropy

    Measures the probability error in discrete binary classification tasks in which each class is independent and
    not mutually exclusive. The cross entropy between two distributions p and q is defined as:


    Warning:
        This is to be used on the logits of a model, not on the predicted labels.
        See ``tf.nn.sigmoid_cross_entropy_with_logits``.

    Args:
        name: function name
        labels: ground truth, correct values
        logits: a tensor with the unscaled log probabilities used to predict the labels with sigmoid(logits)


    Returns:
        ``Tensor``: a float ``Tensor``.

    """
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name=name)


def categorical_cross_entropy(labels, logits, axis=-1, name="categorical_cross_entropy"):
    """ Categorical Cross entropy

    Measures the probability error in discrete classification tasks in which the classes are mutually exclusive.

    Warning:
        This is to be used on the logits of a model, not on the predicted labels.
        See ``tf.nn.softmax_cross_entropy_with_logits``.

    Args:
        labels: ground truth, correct values with a one-hot encoding. Each row labels[i] must be a valid probability
        distribution.
        logits: a tensor with the unscaled log probabilities used to predict the labels with softmax(logits)
        axis: The class dimension. Defaulted to -1 which is the last dimension.
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).

    Returns:
        ``Tensor``: a float ``Tensor``.

    """
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, axis=axis, name=name)


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
    return tf.losses.hinge_loss(labels, logits, weights)


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

    with tf.name_scope(name, [logits, sparsemax, labels]):
        logits = tf.convert_to_tensor(logits)
        sparsemax = tf.convert_to_tensor(sparsemax)
        labels = tf.convert_to_tensor(labels, name="labels")

        shifted_logits = logits - tf.math.reduce_mean(logits, axis=1)[:, tf.newaxis]

        # sum over support (support = predicted labels)
        support = tf.cast(sparsemax > 0, sparsemax.dtype)
        sum_s = support * sparsemax * (shifted_logits - 0.5 * sparsemax)

        # - z_k + ||q||^2
        q_part = labels * (0.5 * labels - shifted_logits)

        return tf.math.reduce_sum(sum_s + q_part, axis=1)


def _sum_rows(x):
    with tf.name_scope("row_sum"):
        """Returns a vector summing up each row of the matrix x."""
        # _sum_rows(x) is equivalent to tf.math.reduce_sum(x, 1) when x is
        # a matrix.  The gradient of _sum_rows(x) is more efficient than
        # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
        # we use _sum_rows(x) in the nce_loss() computation since the loss
        # is mostly used for training.
        cols = tf.shape(x)[1]
        ones_shape = tf.stack([cols, 1])
        ones = tf.ones(ones_shape, x.dtype)
        return tf.reshape(tf.math.matmul(x, ones), [-1])


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

    labels_flat = tf.reshape(labels, [-1])

    # label_sample_prob is the probability of label to appear in the sampled set
    # sampled_classes_prob is the probability of each sampled class to appear in the sampled_classes
    sampled_classes, label_sample_prob, sampled_classes_prob = (
        tf.stop_gradient(s) for s in class_sampler)
    all_ids = tf.concat([labels_flat, sampled_classes], 0)

    all_ids = labels_to_features(all_ids)
    if isinstance(all_ids, tf.SparseTensor):
        sp_values = all_ids
        sp_indices = txf.sparse_indices(sp_values)

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
        all_w = tf.nn.embedding_lookup(
            params=weights,
            ids=all_ids,
            partition_strategy="mod")

        if bias is not None:
            all_b = tf.nn.embedding_lookup(
                params=bias,
                ids=all_ids,
                combiner="sum",
                partition_strategy="mod")

    # true_w shape is [batch_size * num_true, m] with m being the weights dim
    true_w = tf.slice(all_w, [0, 0], tf.stack([tf.shape(labels_flat)[0], -1]))
    sampled_w = tf.slice(all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])

    # [batch_size, num_samples]
    sampled_logits = tf.math.matmul(model_prediction, sampled_w, transpose_b=True)

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = tf.shape(true_w)[1:2]
    new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
    row_wise_dots = tf.math.multiply(
        tf.expand_dims(model_prediction, 1),
        tf.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = tf.reshape(row_wise_dots,
                                tf.concat([[-1], dim], 0))
    true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])

    if bias is not None:
        true_b = tf.slice(all_b, [0], tf.shape(labels_flat))
        true_b = tf.reshape(true_b, [-1, num_true])
        sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

        true_logits += true_b
        sampled_logits += sampled_b

    # add trainable scaling var
    true_logits += scaling_var
    sampled_logits += scaling_var

    # subtract log(k*p_noise(w_j))
    true_logits -= tf.math.log(label_sample_prob)
    sampled_logits -= tf.math.log(sampled_classes_prob)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = tf.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    out_labels = tf.concat([
        tf.ones_like(true_logits) / num_true,
        tf.zeros_like(sampled_logits)
    ], 1)

    return binary_cross_entropy(labels=out_labels, logits=out_logits)


def sparse_cnce_loss(label_features,
                     model_prediction,
                     weights,
                     noise_features=None,
                     num_samples=1,
                     noise_ratio=0.1,
                     corrupt_labels=False,
                     gaussian_corruption=False):
    """
    Sparse Conditional Noise-Contrastive Estimation

    A variation of Conditional Noise-Contrastive Estimation where
    we use sparse additive symmetric noise sampled from an hypercube

    References:
        Ceylan, Gutmann 2018 - Conditional Noise-Contrastive Estimation of Unnormalised Models
        https://arxiv.org/abs/1806.03664

    Args:
        noise_features (SparseTensor) [Optional]: noise_features must be a sparse_tensor with shape
        [dim,shape(label_features)[0]*num_samples]. It none, the noise is generated by the loss function
        but it might be slow due to sample using map in tensorx.
        label_features: the labels to be transformed into sparse features according to the given function
        gaussian_corruption: if True the corrupted entries of the sparse noise are taken from a
        random.normal * noise_values
        corrupt_labels: if True, noise is added to the current sparse feature labels, if False, random noise is used
        without the current labels
        noise_ratio: the ratio of noise according to the number of sparse features
        num_samples: number of counter examples to use for each true label
        weights: the weight table (embeddings) from which we draw the features
        model_prediction: the predicted embedding representation for the next class to be predicted

    """
    with tf.name_scope("sparse_cnce_loss"):
        label_features = tf.convert_to_tensor_or_sparse_tensor(label_features)
        num_samples = tf.convert_to_tensor_or_sparse_tensor(num_samples)

        if not isinstance(label_features, tf.SparseTensor):
            raise TypeError("label_features is must be a SparseTensor: {} found".format(type(label_features)))

        tiled_label_features = txf.sparse_tile(label_features, num_samples)

        if noise_features is None:
            dim = label_features.get_shape().as_list()[-1]
            batch_size = tf.shape(tiled_label_features)[0]

            noise = tx_rnd.sparse_random_mask(dim=dim,
                                              batch_size=batch_size,
                                              density=noise_ratio,
                                              mask_values=[-1, 1],
                                              symmetrical=True,
                                              dtype=tf.float32)

            if corrupt_labels:
                noise = tf.sparse_add(tiled_label_features, noise)
                noise = tf.SparseTensor(noise.indices, noise.values, tiled_label_features.dense_shape)

            if gaussian_corruption:
                sp_noise_values = noise.values * tf.random_normal(tf.shape(noise.values))
            else:
                sp_noise_values = noise.values

            noise_features = tf.SparseTensor(indices=noise.indices,
                                             values=sp_noise_values,
                                             dense_shape=noise.dense_shape)

        true_w = embedding_lookup_sparse(
            params=weights,
            sp_ids=txf.sparse_indices(tiled_label_features),
            sp_weights=tiled_label_features,
            combiner="sum",
            partition_strategy="mod",
            name="true_weights"
        )

        noise_w = embedding_lookup_sparse(
            params=weights,
            sp_ids=txf.sparse_indices(noise_features),
            sp_weights=noise_features,
            combiner="sum",
            partition_strategy="mod",
            name="noise_weights"

        )

        # p_m(y=1|m)
        true_logits = tf.matmul(model_prediction, true_w, transpose_b=True)
        noise_logits = tf.matmul(model_prediction, noise_w, transpose_b=True)

        # log(exp(a)/exp(b)) =  -log(exp(b)/exp(a))
        logit_ratio = true_logits - noise_logits

        # logit_ratio = array_tf.reshape(logit_ratio, [batch_size, -1])

        # log(exp(features) + 1) is the softplus, and softplus from tf already deals with numerical instability
        return tf.math.reduce_mean(tf.nn.softplus(-logit_ratio))


__all__ = ["mse",
           "sparsemax_loss",
           "binary_cross_entropy",
           "categorical_cross_entropy",
           "binary_hinge",
           "sparse_cnce_loss"]
