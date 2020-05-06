import tensorflow as tf
import tensorx as tx


def binary_cross_entropy(labels, logits, name="binary_cross_entropy"):
    """ Binary Cross Entropy

    Measures the probability error in discrete binary classification tasks in which each class is independent and
    not mutually exclusive.

    !!! note "On Entropy and Cross-Entropy"

        Entropy refers to the number of bits required to transmit a randomly selected event from a probability
        distribution. A skewed distribution has a low entropy, whereas a distribution where events have equal
        probability has a larger entropy.

        The entropy of a random variable with a set $x \\in X$ discrete states and their
        probability $P(x)$, can be computed as:

        $$
            H(X) = –\\sum_{x \\in X} P(x) * log(P(x))
        $$

        Cross-entropy builds upon this idea to compute the number of bits required to represent or
        transmit an average event from one distribution compared to another distribution. if we consider a target
        distribution $P$ and an approximation of the target distribution $Q$, the cross-entropy of $Q$ from $P$
        is the number of additional bits to represent an event using Q instead of P:

        $$
            H(P, Q) = –\\sum_{x \\in X} P(x) * log(Q(x))
        $$


    !!! warning
        This is to be used on the **logits** of a model, not on the predicted labels.
        See also [from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits).

    Args:
        labels (`Tensor`): empiric probability values (labels that occurred for a given sample)
        logits (`Tensor`): unscaled log probabilities used to predict the labels with `sigmoid(logits)`
        name (str): op name


    Returns:
        tensor (`Tensor`): binary (sigmoid) cross-entropy loss.

    """
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name=name)


def categorical_cross_entropy(labels, logits, axis=-1, name="categorical_cross_entropy"):
    """ Categorical Cross entropy

    Measures the probability error in discrete classification tasks in which the classes are mutually exclusive.

   !!! warning
        This is to be used on the **logits** of a model, not on the predicted labels. Do not call this loss with the
        output of softmax.
        See also [from TensorFlow](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits).

    Args:
        labels (Tensor): empiric probability distribution. Each row labels[i] must be a valid probability distribution
        (integrate to 1).
        logits (Tensor): unscaled log probabilities used to predict the labels with `softmax(logits)`
        axis (int): The class dimension. Defaulted to -1 which is the last dimension.
        name (str): op name

    Returns:
        tensor (`Tensor`): categorical (softmax) cross-entropy loss.

    """
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, axis=axis, name=name)


def binary_hinge(labels, logits):
    """ Binary Hinge Loss

    Measures the classification error for maximum-margin classification. Margin classifiers like
    Support Vector Machines (SVM) maximise the distance between the closest examples and the decision boundary
    separating the binary classes. The hinge loss is defined as:

    $$
    \\ell(y) = \\max(0, 1-t \\cdot y),
    $$

    where $t$ is the intended output (labels) and $y$ are the output logits from the
    classification decision function, not the predicted class label.

    Args:
        labels (`Tensor`): tensor with values -1 or 1. Binary (0 or 1) labels are converted to -1 or 1.
        logits (`Tensor`): unscaled log probabilities.

    Returns:
        tensor (`Tensor`): hinge loss float tensor
    """
    return tf.losses.hinge(labels, logits)


def mse(target, predicted):
    """ Mean Squared Error (MSE) Loss

    Measures the average of the squares of the errors - the difference between an estimator and what is estimated.
    This is a risk function, corresponding to the expected value of the quadratic loss:

    $$
    MSE =\\frac{1}{N}​\\sum^{N}_{i=0}​(y-\\hat{y})^2
    $$

    !!! info
        MSE is sensitive towards outliers and given several examples with the same input feature values,
        the optimal prediction will be their mean target value. This should be compared with _Mean Absolute
        Error_, where the optimal prediction is the median. MSE is thus good to use if you believe that your
        target data, conditioned on the input, is normally distributed around a mean value --and when it's
        important to penalize outliers.

    Args:
        predicted (`Tensor`): estimated target values
        target (`Tensor`): ground truth, correct values

    Returns:
        tensor (`Tensor`): mean squared error value

    """
    return tf.losses.mean_squared_error(target, predicted)


def sparsemax_loss(logits, labels, name="sparsemax_loss"):
    """ Sparsemax Loss

    A loss function for the sparsemax activation function. This is similar to `tf.nn.softmax`, but able to output s
    parse probabilities.

    !!! info
        Applicable to multi-label classification problems and attention-based neural networks
        (e.g. for natural language inference)

    !!! cite "References"
        [1] [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068)


    Args:
        logits (`Tensor`): unnormalized log probabilities
        name (str): op name

    Returns:
        loss (`Tensor`): sparsemax loss
    """

    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits)
        sparsemax = tx.sparsemax(logits)
        labels = tf.convert_to_tensor(labels, name="labels")

        shifted_logits = logits - tf.math.reduce_mean(logits, axis=1)[:, tf.newaxis]

        # sum over support (support = predicted labels)
        support = tf.cast(sparsemax > 0, sparsemax.dtype)
        sum_s = support * sparsemax * (shifted_logits - 0.5 * sparsemax)

        # - z_k + ||q||^2
        q_part = labels * (0.5 * labels - shifted_logits)

        return tf.math.reduce_sum(sum_s + q_part, axis=1)


def kld(target, predicted):
    """ Kullback–Leibler Divergence Loss

    Kullback–Leibler divergence (also called relative entropy) is a measure of how one probability distribution is
    different from a second, reference probability distribution.

    $$
    D_{KL}(P || Q) = - \\sum_{x \\in X}P(x) log\\left(\\frac{Q(x)}{P(x)}\\right)
    $$

    it is the expectation of the logarithmic difference between the probabilities $P$ and $Q$, where the
    expectation is taken using the probabilities $P$.

    Args:
        target (`Tensor`): target probability distribution
        predicted (`Tensor`): distribution predicted by the model

    Returns:
        kld (`Tensor`): LK divergence between the target and predicted distributions
    """
    return tf.losses.kullback_leibler_divergence(target, predicted)
