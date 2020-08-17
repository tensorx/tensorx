import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
import tensorflow as tf
import numpy as np


def test_model_vars():
    target = tx.Constant([[1.]])
    inputs = tx.Input(n_units=2, name="inputs", constant=False)
    output = tx.Linear(inputs, n_units=1, name="y")
    loss = tx.Lambda(target, output, fn=tf.nn.softmax_cross_entropy_with_logits, name="xent")

    m = tx.Model(run_outputs=output, train_inputs=[inputs, target], train_loss=loss)
    assert m.trainable_variables == output.trainable_variables


def test_add_optimizer():
    target = tx.Constant([[1.]])
    inputs = tx.Input(n_units=2, name="inputs")
    output = tx.Linear(inputs, n_units=1, name="y")

    loss = tx.Lambda(target, output, fn=tf.nn.softmax_cross_entropy_with_logits, name="xent")

    m = tx.Model(run_outputs=output, train_inputs=[inputs, target], train_loss=loss)

    lr = tx.Param(init_value=0.2, name="lr")
    optimizer1 = m.set_optimizer(tf.optimizers.SGD, lr=lr)
    optimizer2: tf.optimizers.Optimizer = m.optimizer

    assert optimizer1 == optimizer2

    # optimizer is hashable
    opt_dict = {optimizer1: 0, optimizer2: 1}
    assert optimizer1 in opt_dict
    assert opt_dict[optimizer1] == 1

    lr.value = 0.3
    assert np.float32(0.3) == optimizer1.lr.numpy()
