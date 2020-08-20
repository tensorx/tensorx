import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorx as tx
import pytest
import numpy as np


def test_dropout():
    """ Dropout with scaling preserves average input activation
    """
    x = tf.ones([100, 100])
    drop_probability = 0.2

    drop_x = tx.dropout(x, probability=drop_probability, scale=True)

    min_x = tf.reduce_min(drop_x)

    non_zero = tf.math.count_nonzero(drop_x)

    assert min_x == 0
    # approx doesn't work with Tensor
    assert non_zero.numpy() == pytest.approx(8000, abs=1000)


def test_dropout_random_tensor():
    x = tf.ones([100, 100])
    drop_probability = 0.2

    mask = np.random.uniform(size=x.shape)
    drop_x1 = tx.dropout(x, probability=drop_probability,
                         random_mask=mask,
                         scale=True)

    # probability becomes irrelevant
    drop_x2 = tx.dropout(x,
                         probability=0.9,
                         random_mask=mask,
                         scale=True)

    nonzero_indices1 = tf.where(tf.not_equal(drop_x1, 0))
    nonzero_indices2 = tf.where(tf.not_equal(drop_x2, 0))

    assert tx.tensor_equal(nonzero_indices1, nonzero_indices2)


def test_dropout_unscaled():
    x = tf.ones([100, 100])
    keep_prob = 0.5

    drop_x = tx.dropout(x, probability=keep_prob, scale=False)

    actual_avg = tf.reduce_mean(drop_x)
    expected_avg = tf.reduce_mean(x)

    assert actual_avg < expected_avg


def test_empty_sparse_tensor():
    dense_shape = [2, 2]
    empty = tx.empty_sparse_tensor(dense_shape)
    dense_empty = tf.sparse.to_dense(empty)
    zeros = tf.zeros(dense_shape)

    assert tf.reduce_all(tf.equal(zeros, dense_empty))


def test_sort_by_first():
    v1 = tf.constant([[3, 1], [2, 1]])
    sorted1 = [[1, 3], [1, 2]]
    v2 = tf.constant([[1, 2], [1, 2]])
    sorted2 = [[2, 1], [2, 1]]

    s1, s2 = tx.sort_by_first(v1, v2, ascending=True)

    assert tx.tensor_equal(s1, sorted1)
    assert tx.tensor_equal(s2, sorted2)

    s1, s2 = tx.sort_by_first([2, 1, 3], [1, 2, 3])
    sorted1 = [1, 2, 3]
    sorted2 = [2, 1, 3]
    assert tx.tensor_equal(s1, sorted1)
    assert tx.tensor_equal(s2, sorted2)


def test_ranges():
    lens = [2, 1, 3]
    rs = tx.ranges(lens)
    assert tf.shape(rs)[-1] == tf.reduce_sum(lens)

    lens = [[2, 1], [2, 1]]
    with pytest.raises(tf.errors.InvalidArgumentError):
        tx.ranges(lens)
        pytest.fail("InvalidArgumentError expected: ranges only works for 1D tensors")

    with pytest.raises(tf.errors.InvalidArgumentError):
        tx.ranges([])


def test_gather_sparse():
    v = tf.constant([[1, 0, 1], [0, 0, 2], [3, 0, 3]], tf.float32)
    sp = tx.to_sparse(v)

    indices = np.array([[0, 1], [2, 0]], dtype=np.int64)

    gather_sp = tx.gather_sparse(sp, indices)
    gather = tf.sparse.to_dense(gather_sp)
    expected = tf.constant([[1., 0., 1.],
                            [0., 0., 2.],
                            [3., 0., 3.],
                            [1., 0, 1.]])

    assert tx.tensor_equal(gather, expected)


def test_sparse_overlapping():
    tensor1 = tf.SparseTensor([[1, 0]], [1], [2, 3])
    tensor2 = tf.SparseTensor([[0, 0], [1, 0], [1, 1]], [3, 4, 5], [2, 3])
    tensor3 = tf.SparseTensor([[0, 1], [1, 1]], [3, 4], [2, 3])

    overlap12 = tx.sparse_overlap(tensor1, tensor2)
    overlap21 = tx.sparse_overlap(tensor2, tensor1)
    overlap23 = tx.sparse_overlap(tensor2, tensor3)
    no_overlap = tx.sparse_overlap(tensor1, tensor3)
    expected_overlap = tf.constant([1])
    expected_overlap23 = tf.constant([5])
    expected_no_overlap = tf.constant([], dtype=tf.int32)

    assert tx.tensor_equal(overlap12.values, expected_overlap)
    assert tx.tensor_equal(overlap21.indices, overlap12.indices)
    # sparse overlapping with no overlapping
    assert tx.tensor_equal(no_overlap.values, expected_no_overlap)
    assert tx.tensor_equal(overlap23.values, expected_overlap23)


def test_grid():
    shape_2d = [4, 4]
    xys = tx.grid_2d(shape_2d)
    shape_xys = tf.shape(xys)

    assert tf.rank(xys) == 2
    assert tx.tensor_equal(shape_xys, [shape_2d[0] * shape_2d[1], 2])


def test_sparse_tile():
    n = 4
    sp = tf.SparseTensor([[0, 0], [0, 1], [1, 2], [2, 3]], [1, 1, 2, 3], [3, 10])
    tilled_sp = tx.sparse_tile(sp, n)

    dense = tf.sparse.to_dense(sp)
    tilled_dense = tf.tile(dense, [n, 1])

    assert tx.tensor_equal(tilled_dense, tf.sparse.to_dense(tilled_sp))

    shape = tf.shape(tilled_sp, tf.int64)
    assert shape[0] == sp.dense_shape[0] * n
    assert shape[-1] == sp.dense_shape[-1]


def test_pairs():
    tensor1 = [[0], [1]]
    tensor2 = [1, 2, 3]
    expected = [[0, 1],
                [1, 1],
                [0, 2],
                [1, 2],
                [0, 3],
                [1, 3]]
    result = tx.pairs(tensor1, tensor2)

    assert tx.tensor_equal(result, expected)


def test_sparse_put():
    tensor = tf.SparseTensor([[0, 0], [1, 0]], [2, 0.2], [2, 2])
    sp_values = tf.SparseTensor([[0, 0], [0, 1]], [3.0, 0], [2, 2])
    expected = tf.constant([[3., 0], [0.2, 0.]])

    result = tx.sparse_put(tensor, sp_values)
    result = tf.sparse.to_dense(result)

    assert tx.tensor_equal(result, expected)


def test_put():
    x = tf.ones([2, 2], tf.int32)
    expected = tf.constant([[3, 1], [1, 1]])

    sp_values = tf.SparseTensor(indices=[[0, 0]], values=[3], dense_shape=[2, 2])
    result = tx.put(x, sp_values)

    assert tx.tensor_equal(result, expected)


def test_dense_put_zero():
    x = tf.ones([2, 2], tf.int32)
    expected = tf.constant([[0, 1], [1, 1]])
    sp_values = tf.SparseTensor(indices=[[0, 0]], values=[0.], dense_shape=[2, 2])
    result = tx.put(x, sp_values)

    assert tx.tensor_equal(result, expected)


def test_sparse_matrix_indices():
    x = tf.constant([[0, 1, 3],
                     [1, 2, 3]])
    num_cols = 4
    expected_dense = [[1, 1, 0, 1],
                      [0, 1, 1, 1]]

    sp_one_hot = tx.sparse_matrix_indices(x, num_cols, dtype=tf.int32)
    dense1 = tf.sparse.to_dense(sp_one_hot)

    assert tx.tensor_equal(dense1, expected_dense)

    sp_one_hot = tx.sparse_matrix_indices(x, num_cols)
    dense1 = tf.sparse.to_dense(sp_one_hot)

    assert not tx.tensor_equal(dense1, expected_dense)


def test_to_sparse():
    c = [[1, 0], [2, 3]]

    sparse_tensor = tx.to_sparse(c)

    dense_shape = tf.shape(c, out_type=tf.int64)
    indices = tf.where(tf.not_equal(c, 0))

    flat_values = tf.reshape(c, [-1])
    flat_indices = tf.where(tf.not_equal(flat_values, 0))
    flat_indices = tf.squeeze(flat_indices)
    flat_indices = tf.math.mod(flat_indices, dense_shape[1])

    values = tf.gather_nd(c, indices)

    sp_indices = tx.sparse_indices(sparse_tensor)

    assert tx.tensor_equal(sparse_tensor.indices, indices)
    assert tx.tensor_equal(sp_indices.values, flat_indices)
    assert tx.tensor_equal(sparse_tensor.values, values)


def test_to_sparse_zero():
    shape = [2, 3]
    data_zero = tf.zeros(shape)
    sparse_tensor = tx.to_sparse(data_zero)
    dense = tf.sparse.to_dense(sparse_tensor)
    num_indices = tf.shape(sparse_tensor.indices)[0]

    assert num_indices == 0
    assert tx.tensor_equal(dense, data_zero)


def test_filter_nd():
    inputs = tf.constant([[1], [2], [0], [3], [4]])
    sp_result = tx.filter_nd(tf.greater(inputs, 0), inputs)
    assert isinstance(sp_result, tf.SparseTensor)
    assert tx.tensor_equal(sp_result.values, [1, 2, 3, 4])
