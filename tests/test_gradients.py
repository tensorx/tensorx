import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
import tensorflow as tf


def test_gradient_sparse_var():
    """
    https://www.tensorflow.org/beta/guide/effective_tf2
    """
    target = tf.constant([[1., 0., 0.], [1., 0., 0.]])

    v = tf.Variable([0.5, 0.5])

    x = tx.Lambda([],
                  fn=lambda _: tf.SparseTensor([[0, 0], [1, 1]], v, [2, 3]),
                  n_units=3,
                  var_list=v)

    assert isinstance(x(), tf.SparseTensor)
    assert len(x.trainable_variables) == 1

    y = tx.Linear(x, n_units=3)

    # a graph without inputs needs to have missing inputs declared
    # otherwise it will try to add the inputs detected to inputs
    graph = tx.Graph.build(inputs=None,
                           outputs=y)
    fn = graph.as_function()

    @tf.function
    def loss(labels):
        return tf.reduce_mean(tf.pow(labels - fn(), 2))

    with tf.GradientTape() as tape:
        loss_val = loss(target)

    assert tx.shape_equal(tape.gradient(loss_val, v), v.value())


def test_to_sparse_gradient():
    target = tf.constant([[1., 0.], [1., 0.]])
    x = tx.Constant(tf.ones([1, 4], dtype=tf.float32), n_units=4)
    h = tx.Linear(x, n_units=2)
    y = tx.ToSparse(h)
    y = tx.ToDense(y)

    @tf.function
    def loss_fn(labels, prediction):
        return tf.reduce_mean(tf.pow(labels - prediction, 2))

    with tf.GradientTape() as tape:
        pred = y()
        loss = loss_fn(target, pred)

    gradients = tape.gradient(loss, h.trainable_variables)
    assert len(gradients) == 2
    assert tx.shape_equal(gradients[0], h.weights)
    assert tx.shape_equal(gradients[1], h.bias)
