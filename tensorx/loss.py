""" Loss Functions

To be used with to optimise neural network models, some of these are forwards from the `TensorFlow` API with some
additional documentation.
"""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses.losses import mean_squared_error, sigmoid_cross_entropy
from tensorflow.python.ops.nn import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.losses.losses import hinge_loss as tf_hinge_loss


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


def binary_cross_entropy(labels, logits, weights=1.0):
    """ Binary Cross Entropy

    Measures the probability error in discrete binary classification tasks in which each class is independent and
    not mutually exclusive. The cross entropy between two distributions p and q is defined as:


    Warning:
        This is to be used on the logits of a model, not on the predicted labels.
        See ``tf.nn.sigmoid_cross_entropy_with_logits``.

    Args:
        labels: ground truth, correct values
        logits: a tensor with the unscaled log probabilities used to predict the labels with sigmoid(logits)
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).

    Returns:
        ``Tensor``: a float ``Tensor``.

    """
    return sigmoid_cross_entropy(labels, logits)


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


__all__ = ["mse",
           "sparsemax_loss",
           "binary_cross_entropy",
           "categorical_cross_entropy",
           "binary_hinge"]
