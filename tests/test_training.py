import unittest
from tensorx.training import ModelRunner, Model
from tensorx.layers import Input, Linear, Activation, Add, TensorLayer

from tensorx.activation import tanh, sigmoid
from tensorx.loss import binary_cross_entropy
from tensorx import init

import numpy as np
import tensorflow as tf

import os
import glob

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
        model = Model(inputs=inputs, outputs=linear)
        runner = ModelRunner(model)

        # creates a new session or uses the default session
        self.assertIsNone(runner.session)
        self.assertFalse(runner.vars_inited())
        runner.run(data)
        self.assertIsNotNone(runner.session)
        self.assertTrue(runner.vars_inited())

        # consecutive runs do not change the session
        session1 = runner.session
        result1 = runner.run(data)
        # weights1 = session1.run(linear.weights, {inputs.tensor: data})
        self.assertEqual(runner.session, session1)

        # this creates a new session
        session2 = runner.set_session()
        self.assertEqual(session2, runner.session)
        self.assertIsNotNone(runner.session)
        self.assertNotEqual(session1, session2)
        # setting a new session resets the variables
        self.assertFalse(runner.vars_inited())

        # different sessions init the variables again
        result2 = runner.run(data)
        self.assertFalse(np.array_equal(result1, result2))

        runner.set_session()
        # explicitly initialise variables with the new session
        runner.init_vars()
        self.assertTrue(runner.vars_inited())
        result31 = runner.run(data)
        # if the session doesn't change and variables are not re-initialised, the result should be the same
        result32 = runner.run(data)
        runner.init_vars()
        result33 = runner.run(data)
        self.assertTrue(np.array_equal(result31, result32))
        self.assertFalse(np.array_equal(result31, result33))

        # to use the model in a new session, either call reset or model.set_session(session)
        runner.reset_session()

        with tf.Session() as session4:
            runner.run(data)
            self.assertEqual(runner.session, session4)

        runner.reset_session()
        self.assertIsNone(runner.session)
        session5 = tf.InteractiveSession()
        runner.run(data)
        self.assertEqual(runner.session, session5)

    def test_model_var_init(self):
        inputs = Input(1)
        linear = Linear(inputs, 2)
        model = Model(inputs=inputs, outputs=linear)
        runner = ModelRunner(model)

        with tf.Session() as session1:
            self.assertFalse(session1.run(tf.is_variable_initialized(linear.bias)))
            runner.init_vars()
            self.assertTrue(session1.run(tf.is_variable_initialized(linear.bias)))
            runner.run([[1.]])

        # if reset is not called, init vars tries to use session1
        runner.reset_session()
        session2 = tf.Session()
        runner.set_session(session2)
        runner.init_vars()
        self.assertTrue(session2.run(tf.is_variable_initialized(linear.bias)))

        session2.close()

    def test_model_run(self):
        inputs = Input(4)
        linear = Linear(inputs, 2)
        h = Activation(linear, fn=tanh)
        logits = Linear(h, 4)
        out = Activation(logits, fn=sigmoid)

        model = Model(inputs, out)
        runner = ModelRunner(model)

        data1 = [[1, -1, 1, -1]]

        result = runner.run(data1)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.ndim(result), 2)

    def test_model_graph_save(self):
        tf.reset_default_graph()
        const = tf.ones([1, 10], name="const")
        wrap = TensorLayer(const, 10)
        model = Model(inputs=wrap, outputs=wrap)

        all_ops = tf.get_default_graph().get_operations()
        self.assertEqual(len(all_ops), 1)

        run = ModelRunner(model)

        result = run.run()

        run.save_model(save_dir="/tmp/", model_name="graph_only", save_graph=True)

        tf.reset_default_graph()
        all_ops = tf.get_default_graph().get_operations()
        self.assertEqual(len(all_ops), 0)

        tf.train.import_meta_graph("/tmp/graph_only.meta")
        all_ops = tf.get_default_graph().get_operations()

        self.assertEqual(len(all_ops), 1)
        t = tf.get_default_graph().get_tensor_by_name("const:0")
        with tf.Session() as sess:
            result2 = sess.run(t)
            np.testing.assert_array_equal(result, result2)

    def test_model_save(self):

        session = tf.Session()

        def build_model():
            inputs = Input(4)
            linear = Linear(inputs, 2)
            h = Activation(linear, fn=tanh)
            logits = Linear(h, 4)
            outputs = Activation(logits, fn=sigmoid)
            return Model(inputs, outputs)

        model = build_model()
        runner = ModelRunner(model)
        runner.set_session(session)
        runner.init_vars()
        # for layer in model.layers:
        #    print(layer)
        self.assertEqual(len(model.layers), 5)

        w1, w2 = model.layers[1].weights, model.layers[3].weights

        w1 = session.run(w1)
        w2 = session.run(w2)

        save_dir = "/tmp"
        model_name = "test.ckpt"
        model_path = os.path.join(save_dir, model_name)

        self.assertFalse(os.path.exists(model_path))
        runner.save_model(save_dir, model_name)

        model_files = glob.glob('{model}*'.format(model=model_path))
        self.assertTrue(len(model_files) != 0)
        runner.init_vars()

        runner.load_model(save_dir, model_name)
        w3, w4 = model.layers[1].weights, model.layers[3].weights
        w3 = session.run(w3)
        w4 = session.run(w4)

        self.assertTrue(np.array_equal(w1, w3))
        self.assertTrue(np.array_equal(w2, w4))

        os.remove(os.path.join(save_dir, "checkpoint"))
        for file in model_files:
            os.remove(file)

    def test_model_io(self):
        inputs = Input(1)
        layer = Linear(inputs, n_units=1, init=init.ones_init())

        model = Model(inputs, layer)

        model_runner = ModelRunner(model)
        result = model_runner.run([[1], [1]])
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, np.ndarray)

        linear1 = Linear(inputs, 1)
        linear2 = Linear(inputs, 2)

        model = Model(inputs, outputs=[linear1, linear2])
        model_runner = ModelRunner(model)
        result = model_runner.run([[1]])

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

        data1 = [[1]]
        data2 = [[-1]]
        with self.assertRaises(ValueError):
            model_runner.run(data1, data2)

        inputs1 = Input(1)
        inputs2 = Input(1)

        merge = Add([inputs1, inputs2])

        model = Model([inputs1, inputs2], merge)
        model_runner = ModelRunner(model)

        with self.assertRaises(ValueError):
            model_runner.run(data1)

        result = model_runner.run(data1, data2)
        self.assertEqual(result[0][0], 0)

    def test_model_train(self):
        input_layer = Input(4, name="x")
        linear = Linear(input_layer, 2)
        h = Activation(linear, fn=tanh)

        # configure training
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        labels = Input(2, name="y_")
        losses = binary_cross_entropy(labels.tensor, h.tensor)

        model = Model(input_layer, h)
        runner = ModelRunner(model)
        runner.config_training(optimiser, losses, labels)

        data = np.array([[1, 1, 1, 1]])
        target = np.array([[1, 0]])

        # session = runner.session
        # weights = session.run(linear.weights)
        runner.init_vars()
        weights1 = runner.session.run(linear.weights)

        for i in range(100):
            runner.train(data, target)

        weights2 = runner.session.run(linear.weights)

        self.assertFalse(np.array_equal(weights1, weights2))


if __name__ == '__main__':
    unittest.main()
