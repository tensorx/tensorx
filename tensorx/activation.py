from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn import top_k


def sparsemax(logits, name=None):
    """Computes the sparsemax activation function [1]

    For each batch `i` and class `j` we have
      sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)

    References:
        [1]: https://arxiv.org/abs/1602.02068

    Args:
      logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
        `float64`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `logits`.
    """

    with ops.name_scope(name, "sparsemax", [logits]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        obs = array_ops.shape(logits)[0]
        dims = array_ops.shape(logits)[1]

        z = logits - math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

        # sort z
        z_sorted, _ = top_k(z, k=dims)

        # calculate k(z)
        z_cumsum = math_ops.cumsum(z_sorted, axis=1)
        k = math_ops.range(
            1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

        # calculate tau(z)
        indices = array_ops.stack([math_ops.range(0, obs), k_z - 1], axis=1)
        tau_sum = array_ops.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

        # calculate p
        return math_ops.maximum(
            math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])
