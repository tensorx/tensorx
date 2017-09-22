import unittest
from tensorx.model import Model
from tensorx.layers import Input, Linear, Activation, Merge, Add
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
        data = [[1]]
        inputs = Input(1)
        linear = Linear(inputs, 1)
        model = Model(inputs, linear)

        # creates a new session or uses the default session
        self.assertIsNone(model.session)
        self.assertFalse(model.vars_inited())
        model.run(data)
        self.assertIsNotNone(model.session)
        self.assertTrue(model.vars_inited())

        # consecutive runs do not change the session
        session1 = model.session
        result1 = model.run(data)
        # weights1 = session1.run(linear.weights, {inputs.tensor: data})
        self.assertEqual(model.session, session1)

        # this creates a new session
        session2 = model.set_session()
        self.assertEqual(session2, model.session)
        self.assertIsNotNone(model.session)
        self.assertNotEqual(session1, session2)
        # setting a new session resets the variables
        self.assertFalse(model.vars_inited())

        # different sessions init the variables again
        result2 = model.run(data)
        self.assertFalse(np.array_equal(result1, result2))

        session3 = model.set_session()
        # explicitly initialise variables with the new session
        model.init_vars()
        self.assertTrue(model.vars_inited())
        result31 = model.run(data)
        # if the session doesn't change and variables are not re-initialised, the result should be the same
        result32 = model.run(data)
        model.init_vars()
        result33 = model.run(data)
        self.assertTrue(np.array_equal(result31, result32))
        self.assertFalse(np.array_equal(result31, result33))

        # to use the model in a new session, either call reset or model.set_session(session)
        model.reset_session()

        with tf.Session() as session4:
            model.run(data)
            self.assertEqual(model.session, session4)

        model.reset_session()
        self.assertIsNone(model.session)
        session5 = tf.InteractiveSession()
        model.run(data)
        self.assertEqual(model.session, session5)

    def test_model_var_init(self):
        inputs = Input(1)
        linear = Linear(inputs, 2)
        model = Model(inputs, linear)

        with tf.Session() as session1:
            self.assertFalse(session1.run(tf.is_variable_initialized(linear.bias)))
            model.init_vars()
            self.assertTrue(session1.run(tf.is_variable_initialized(linear.bias)))
            model.run([[1.]])

        # if reset is not called, init vars tries to use session1
        model.reset_session()
        session2 = tf.Session()
        model.set_session(session2)
        model.init_vars()
        self.assertTrue(session2.run(tf.is_variable_initialized(linear.bias)))

        session2.close()

        # self.assertFalse(model.vars_initialised)

        # model.init_vars()
        # self.assertTrue(session2.run(tf.is_variable_initialized(linear.bias)))

    def test_model_run(self):
        inputs = Input(4)
        linear = Linear(inputs, 2)
        h = Activation(linear, fn=tanh)
        logits = Linear(h, 4)
        out = Activation(logits, fn=sigmoid)

        model = Model(inputs, out)

        data1 = [[1, -1, 1, -1]]

        result = model.run(data1)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.ndim(result), 2)

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

        merge = Add([inputs1, inputs2])

        model = Model([inputs1, inputs2], merge)

        with self.assertRaises(ValueError):
            model.run(data1)

        result = model.run(data1, data2)
        self.assertEqual(result[0][0], 0)

    def test_model_train(self):
        input_layer = Input(4,name="x")
        linear = Linear(input_layer, 2)
        h = Activation(linear, fn=tanh)

        # configure training
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        labels = Input(2,name="y_")
        losses = binary_cross_entropy(labels.tensor, h.tensor)

        model = Model(input_layer, h)
        model.config(optimiser, losses, labels)

        data = [[[1, 1, 1, 1]]]
        target = [[[1, 0]]]

        # session = model.session
        # weights = session.run(linear.weights)
        model.init_vars()
        weights1 = model.session.run(linear.weights)

        for i in range(100):
            model.train(data, target, n_epochs=1)

        weights2 = model.session.run(linear.weights)

        self.assertFalse(np.array_equal(weights1,weights2))


if __name__ == '__main__':
    unittest.main()
