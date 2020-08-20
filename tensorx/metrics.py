import tensorflow as tf
from typing import Union, List
from tensorx.utils import as_tensor
from tensorx.math import batch_sparse_dot, sparse_l2_norm
from tensorx.ops import grid_2d


def cosine_distance(tensor1, tensor2, dtype=tf.float32):
    """ cosine_distance

    Computes the pairwise cosine distance between two non-zero tensors on their last dimension.
    The cosine distance is defined as 1 - cosine similarity. With the cosine similarity defined as:

    $$
    similarity =\\cos (\\theta)=\\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\|\\|\\mathbf{B}\\|}=\\frac{
    \\sum_{i=1}^{n} A_{i} B_{i}}{\\sqrt{\\sum_{i=1}^{n} A_{i}^{2}} \\sqrt{\\sum_{i=1}^{n} B_{i}^{2}}}
    $$

    Args:
        tensor1 (`Tensor`): first tensor
        tensor2 (`Tensor`): second tensor
        dtype (`DType`): assumed type of both tensors

    Returns:
        distance (`Tensor`): the pairwise cosine distance between two tensors
    """
    tensor1 = tf.convert_to_tensor(tensor1, dtype)
    tensor2 = tf.convert_to_tensor(tensor2, dtype)

    dot_prod = tf.reduce_sum(tf.multiply(tensor1, tensor2), -1)
    norm1 = tf.norm(tensor1, axis=-1)
    norm2 = tf.norm(tensor2, axis=-1)

    norm12 = norm1 * norm2
    cos12 = dot_prod / norm12

    sim = tf.where(tf.math.is_nan(cos12), tf.zeros_like(cos12), cos12)

    # if we need to correct this to angular distance, acos(1.000001) is nan)
    sim = tf.clip_by_value(sim, -1., 1.)
    return 1 - sim


def euclidean_distance(tensor1, tensor2):
    """ Computes the euclidean distance between two tensors.

    The euclidean distance or $L^2$ distance between points $p$ and $q$ is the length of the line segment
    connecting them.

    $$
    distance(q,p) =\\sqrt{\\sum_{i=1}^{n}\\left(q_{i}-p_{i}\\right)^{2}}
    $$

    Args:
        tensor1: a ``Tensor``
        tensor2: a ``Tensor``
        dim: dimension along which the euclidean distance is computed

    Returns:
        ``Tensor``: a ``Tensor`` with the euclidean distances between the two tensors

    """
    tensor1 = tf.convert_to_tensor(tensor1)
    tensor2 = tf.convert_to_tensor(tensor2)

    distance = tf.sqrt(tf.reduce_sum(tf.square(tensor1 - tensor2), axis=-1))

    return distance


def sparse_euclidean_distance(sp_tensor, tensor2):
    """ Computes the euclidean distance between two tensors.

        Args:
            sp_tensor (`Union[Tensor,SparseTensor]`): a tensor or sparse tensor
            tensor2 (`Tensor`): a dense tensor

        Returns:
            distance (`Tensor`): euclidean distances between the two tensors

        """
    tensor1 = tf.SparseTensor.from_value(sp_tensor)
    if tensor1.values.dtype != tf.float32:
        tensor1.values = tf.cast(tensor1.values, tf.float32)
    tensor2 = tf.convert_to_tensor(tensor2)

    distance = tf.sqrt(tf.reduce_sum(tf.square(tensor1 - tensor2), axis=-1))

    return distance


def pairwise_euclidean_distance(tensor1, tensor2, keepdims=False):
    """ Computes the euclidean distance between two tensors.

    Args:
        tensor1 (`Tensor`): a dense tensor
        tensor2 (`Tensor`): a dense tensor
        keepdims (`Bool`): if True, the result maintains the dimensions of the original result

    Returns:
        distance (`Tensor`): euclidean distances between the two tensors
    """
    tensor1 = tf.convert_to_tensor(tensor1)
    tensor2 = tf.convert_to_tensor(tensor2)
    tensor1 = tf.expand_dims(tensor1, 1)

    distance = tf.sqrt(tf.reduce_sum(tf.square(tensor1 - tensor2), axis=-1, keepdims=keepdims))

    return distance


def torus_l1_distance(point, shape):
    """ Computes the l1 distance between a given point or batch of points and all other points in a torus

    Args:
        point (`Tensor`): a rank 0 or rank 1 tensor with the coordinates for a point or a rank 2 tensor with a batch of points.
        shape (`List`): a list with the shape for the torus - either 1D or 2D

    Returns:
        distances (`Tensor`): a rank 1 or 2 tensor with the distances between each point in the 1D torus and each unique
        coordinate in the shape

    Examples:
        * distance for a single point
        `torus_l1_distance(1,[4])`
        or
        `torus_1d_l1_distance([1],[4])`

        ```python
        [ 1.,  0.,  1.,  2.]
        ```

        * distance for multiple points `torus_l1_distance([[2],[3]],[4])`

        ```python
        [[ 2.,  1.,  0.,  1.],
         [ 1.,  2.,  1.,  0.]]
        ```

        * distance between a point and other coordinates in a 2D torus

        ```python
        r = torus_l1_distance([[1,1],[1,2]],[3,3])
        np.reshape(r,[-1,3,3])

        [[[ 2.,  1.,  2.],
          [ 1.,  0.,  1.],
          [ 2.,  1.,  2.]],

         [[ 2.,  2.,  1.],
          [ 1.,  1.,  0.],
          [ 2.,  2.,  1.]]]
        ```

    """
    point = as_tensor(point, tf.float32)
    if len(shape) == 1:
        max_x = shape[0]
        coor_x = tf.range(0, max_x, 1, dtype=tf.float32)
        dx = tf.abs(point - coor_x)
        distance = tf.minimum(dx, tf.math.mod(-dx, max_x))
    elif len(shape) == 2:
        max_x = shape[0]
        max_y = shape[1]

        xys = grid_2d(shape)
        xys = tf.cast(xys, tf.float32)

        xs, ys = tf.unstack(xys, num=2, axis=-1)

        px, py = tf.unstack(point, num=2, axis=-1)
        px = tf.expand_dims(px, 1)
        py = tf.expand_dims(py, 1)

        dx = tf.abs(px - xs)
        dy = tf.abs(py - ys)

        dx = tf.minimum(dx, tf.math.mod(-dx, max_x))

        dy = tf.minimum(dy, tf.math.mod(-dy, max_y))

        distance = dx + dy
    else:
        raise ValueError("Invalid shape parameter, shape must have len 1 or 2")

    return distance


def batch_manhattan_distance(tensor1, tensor2, keepdims=False):
    """ Compute the pairwise manhattan distance between a batch of tensors and a second tensor

    If any tensor is a ``SparseTensor``, it is converted to

    Args:
        tensor1 (`Union[Tensor,SparseTensor]`): a batch of tensors or sparse tensor
        tensor2 (`Union[Tensor,SparseTensor]`): another tensor or a sparse tensor
        keepdims (`Bool`): if True keeps the dimensions of the original tensors

    Returns:
        distance (`Tensor`): the manhattan distance between the two tensors

    """
    tensor1 = as_tensor(tensor1)
    tensor2 = as_tensor(tensor2)

    if isinstance(tensor1, tf.SparseTensor):
        tensor1 = tf.sparse.to_dense(tensor1)
    if isinstance(tensor2, tf.SparseTensor):
        tensor2 = tf.sparse.to_dense(tensor2)

    tensor1 = tf.expand_dims(tensor1, 1)
    abs_diff = tf.abs(tf.subtract(tensor1, tensor2))
    return tf.reduce_sum(abs_diff, axis=-1, keepdims=keepdims)


def batch_sparse_cosine_distance(sp_tensor, tensor, dtype=tf.float32, keepdims=False):
    """ Computes the cosine distance between two non-zero `SparseTensor` and `Tensor`

        Warning:
            1 - cosine similarity is not a proper distance metric, to repair the triangle inequality property while
            maintaining the same ordering, it is necessary to convert to angular distance

        Args:
            sp_tensor: a `SparseTensor`
            tensor: a `Tensor`
            dtype:
            keepdims: keeps the original dimension of the input tensor

        Returns:
            a `Tensor` with the cosine distance between two tensors
        """
    sp_tensor = as_tensor(sp_tensor, dtype)
    tensor = tf.convert_to_tensor(tensor, dtype)

    dot_prod = batch_sparse_dot(sp_tensor, tensor, keepdims=keepdims)

    norm1 = sparse_l2_norm(sp_tensor, axis=-1, keepdims=True)
    norm2 = tf.norm(tensor, axis=-1)

    norm12 = norm1 * norm2
    if keepdims:
        norm12 = tf.expand_dims(norm12, -1)

    cos12 = dot_prod / norm12

    sim = tf.where(tf.math.is_nan(cos12), tf.zeros_like(cos12), cos12)
    sim = tf.clip_by_value(sim, -1., 1.)

    return 1 - sim


def sinkhorn(tensor1, tensor2, epsilon, n_iter, cost_fn=None):
    """ Sinkhorn Distance

    !!! info
        Optimal Transport (OT) provides a framework from which one can define a more powerful geometry to compare
        probability distributions. This power comes, however, with a heavy computational price. The cost of computing OT
        distances scales at least in $O(d^3 log(d))$ when comparing two histograms of dimension $d$. Sinkhorn algorithm
        alleviate this problem by solving an regularized OT in linear time.

    Given two measures with n points each with locations x and y
    outputs an approximation of the Optimal Transport (OT) cost with regularization
    parameter epsilon, niter is the maximum number of steps in sinkhorn loop

    !!! cite "References" 1. [Concerning nonnegative matrices and doubly stochastic matrices](
    https://msp.org/pjm/1967/21-2/p14.xhtml) 2. [Sinkhorn Distances:Lightspeed Computation of Optimal Transport](
    https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf)

    Args:
        tensor1 (`Tensor`): a tensor representing a distribution
        tensor2 (`Tensor`): other tensor with another distribution
        epsilon (float): regularization term >0
        n_iter (`int`): number of sinkhorn iterations
        cost_fn (`Callable`): function that returns the cost matrix between y_pred and y_true, defaults to $|x_i-y_j|^p$.

    Returns:
        cost (`Tensor`): sinkhorn cost of moving from the mass from the model distribution `y_pred` to the empirical
        distribution `y_true`.
    """

    def cost_matrix(x, y, p=2):
        """ cost matrix of $|x_i-y_j|^p$.
        """
        xc = tf.expand_dims(x, 1)
        yr = tf.expand_dims(y, 0)
        d = tf.math.pow(tf.abs(xc - yr), p)
        return tf.reduce_sum(d, axis=-1)

    # n x n Wasserstein cost function
    if cost_fn is None:
        cost_m = cost_matrix(tensor1, tensor2)
    else:
        cost_m = cost_fn(tensor1, tensor2)
    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * tf.ones([n], dtype=tf.float32))
    # nu = Variable(1. / n * tf.ones([n], dtype=tf.float32))

    n = tf.shape(tensor1)[0]

    init_v = tf.cast(n, tf.float32) * tf.ones([n], dtype=tf.float32)

    mu = 1. / init_v
    nu = 1. / init_v

    # Parameters of the sinkhorn algorithm.
    rho = 1  # (.5) **2 # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 0.1  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        # Barycenter subroutine, used by kinetic acceleration through extrapolation.
        return tau * u + (1 - tau) * u1

    def M(u, v):
        # Modified cost for logarithmic updates $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        return (-cost_m + tf.expand_dims(u, 1) + tf.expand_dims(v, 0)) / epsilon

    def lse(A):
        # log-sum-exp
        return tf.reduce_logsumexp(A, axis=1, keepdims=True)

    # Actual Sinkhorn loop ......................................................................
    init_u, init_v, init_err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    def body(i, u, v, _):
        u0 = u  # to check the error threshold
        new_u = epsilon * (tf.math.log(mu) - tf.squeeze(lse(M(u, v)))) + u
        new_v = epsilon * (tf.math.log(nu) - tf.squeeze(lse(tf.transpose(M(new_u, v))))) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        error = tf.reduce_sum(tf.abs(new_u - u0))

        return i + 1, new_u, new_v, error

    def cond(i, u, v, err):
        return tf.logical_and(tf.less(err, thresh), tf.less(i, n_iter))

    i, u, v, err = tf.while_loop(cond=cond,
                                 body=body,
                                 loop_vars=(0, init_u, init_v, init_err))

    pi = tf.exp(M(u, v))  # Transport plan p_i = diag(a)*K*diag(b)

    return tf.reduce_sum(pi * cost_m)  # pi * cost_m  # tf.reduce_sum(pi * cost_m)


__all__ = [
    "torus_l1_distance",
    "sparse_euclidean_distance",
    "euclidean_distance",
    "cosine_distance",
    "batch_manhattan_distance",
    "batch_sparse_cosine_distance",
    "sinkhorn",
    "pairwise_euclidean_distance"
]
