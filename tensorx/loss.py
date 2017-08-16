from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def sparsemax_loss(logits, sparsemax, labels, name=None):
    """Computes sparsemax loss function [1].

    References:
        [1]: https://arxiv.org/abs/1602.02068

    Args:
      logits: A `Tensor`, before sparsemax is applied
      sparsemax: A `Tensor` resulting from applying a sparsemax activation. Must have the same type as `logits`.
      labels: A `Tensor`. Must have the same type as `logits`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `logits`.
    """

    with ops.name_scope(name, "sparsemax_loss",
                        [logits, sparsemax, labels]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        sparsemax = ops.convert_to_tensor(sparsemax, name="sparsemax")
        labels = ops.convert_to_tensor(labels, name="labels")

        shifted_logits = logits - \
                         math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

        # sum over support (support = predicted labels)
        support = math_ops.cast(sparsemax > 0, sparsemax.dtype)
        sum_s = support * sparsemax * (shifted_logits - 0.5 * sparsemax)

        # - z_k + ||q||^2
        q_part = labels * (0.5 * labels - shifted_logits)

        return math_ops.reduce_sum(sum_s + q_part, axis=1)
