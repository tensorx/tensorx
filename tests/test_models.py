import unittest
from tensorx.model import Model
from tensorx.layers import Input, Linear, Activation, Merge
from tensorx.activation import tanh, sigmoid
from tensorx.loss import binary_cross_entropy

import numpy as np
import tensorflow as tf

"""TODO 
    - consider how models can be visualised on each layer
    - consider how model can be used to facilitate debugging
    - consider if it is worth it to change the layer api to have a "build_graph" method 
    to create reusable layers that can be cloned and wired afterwards in a model or using
    something similar to the functional API of keras, or using something similar to tensorfold blocks
"""


class MyTestCase(unittest.TestCase):
    def test_model_session(self):
        input_layer = Input(1)

        model = Model(inputs=[input_layer], outputs=[input_layer])
        self.assertEqual(model.session, None)

        model.run([[1]])

        session1 = model.session

        with tf.Session() as session2:
            self.assertNotEqual(session1, session2)
            model.run([[1]])
            self.assertEqual(model.session, session2)

    def test_model_run(self):
        inputs = Input(4)
        linear = Linear(inputs, 2)
        h = Activation(linear, fn=tanh)
        logits = Linear(h, 4)
        out = Activation(logits, fn=sigmoid)

        model = Model(inputs, out)

        data1 = [[1, -1, 1, -1]]
        data2 = [[1, -1, 1, -1]]

        result = model.run(data1)
        self.assertIsInstance(result,np.ndarray)
        self.assertTrue(np.ndim(result),2)


    def test_model_io(self):
        inputs = Input(1)
        model = Model(inputs, inputs)
        result = model.run([[1], [1]])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, np.ndarray)

        linear1 = Linear(inputs, 1)
        linear2 = Linear(inputs, 2)

        model = Model(inputs, outputs=[linear1, linear2])
        result = model.run([[1]])

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

        data1 = [[1]]
        data2 = [[-1]]
        with self.assertRaises(ValueError):
            model.run(data1, data2)

        inputs1 = Input(1)
        inputs2 = Input(1)

        merge = Merge([inputs1,inputs2])






    def test_model_train(self):
        input_layer = Input(4)
        linear = Linear(input_layer, 2)
        h = Activation(linear, fn=tanh)
        logits = Linear(h, 4)
        out = Activation(logits, fn=sigmoid)

        optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        labels = Input(4)
        losses = binary_cross_entropy(labels.tensor, logits.tensor)

        model = Model(input_layer, out)
        model.config(optimiser, losses, labels)

        data = [[[1, 1, 1, 1]]]
        target = [[[1, 0, 1, 0]]]
        model.train(data, target)


if __name__ == '__main__':
    unittest.main()
