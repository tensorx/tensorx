import unittest
import tensorx as tx
import tensorx.callbacks as tc
from functools import partial
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MyTestCase(unittest.TestCase):
    def test_model_vars(self):
        target = tx.Tensor([[1.]])
        inputs = tx.Input(n_units=2, name="inputs", constant=False)
        output = tx.Linear(inputs, n_units=1, name="y")
        loss = tx.Lambda(target, output, fn=tf.nn.softmax_cross_entropy_with_logits, name="xent")

        m = tx.Model(run_outputs=output, train_inputs=[inputs, target], train_loss=loss)
        self.assertListEqual(m.trainable_variables, output.trainable_variables)

    def test_add_optimizer(self):
        target = tx.Tensor([[1.]])

        inputs = tx.Input(n_units=2, name="inputs")
        output = tx.Linear(inputs, n_units=1, name="y")

        loss = tx.Lambda(target, output, fn=tf.nn.softmax_cross_entropy_with_logits, name="xent")

        # TODO are train inputs exclusively from training ? or do we have to specify all of them?
        m = tx.Model(run_outputs=output, train_inputs=[inputs, target], train_loss=loss)

        lr = tx.Param(value=0.2, name="lr")
        optimizer1 = m.add_optimizer(tf.optimizers.SGD, lr=lr)
        optimizer2: tf.optimizers.Optimizer = m.optimizers[0]

        self.assertEqual(optimizer1, optimizer2)
        try:
            opt_dict = {optimizer1: 0, optimizer2: 1}
            self.assertEqual(opt_dict[optimizer1], 1)
        except Exception as e:
            raise ValueError(f"optimizers should be hashable but: {str(e)}")
        lr.value = 0.3
        self.assertEqual(np.float32(0.3), optimizer1.lr.numpy())


if __name__ == '__main__':
    unittest.main()
