from tensorx import test_utils
from tensorx.train import ModelRunner, Model, EvalStepDecayParam
from tensorx.layers import Input, Linear, Activation, Add, TensorLayer

from tensorx.activation import tanh, sigmoid
from tensorx.loss import binary_cross_entropy
from tensorx import init

import numpy as np
import tensorflow as tf

import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ModelRunnerTest(test_utils.TestCase):
    def test_model_session(self):
        data = [[1]]
        inputs = Input(1)
        # feed = {inputs.placeholder: data}
        linear = Linear(inputs, 1)
        model = Model(run_in_layers=inputs, run_out_layers=linear)
        runner = ModelRunner(model)

        # runner.set_session()

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

        with tf.Session() as new_session:
            # this creates a new session
            session2 = runner.set_session()
            self.assertIsNotNone(runner.session)
            self.assertNotEqual(session1, session2)
            self.assertEqual(session2, new_session)
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
        tf.reset_default_graph()

    def test_model_var_init(self):
        inputs = Input(1)
        linear = Linear(inputs, 2)
        model = Model(run_in_layers=inputs, run_out_layers=linear)
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
        with self.cached_session(use_gpu=True):
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

    def test_model_train(self):
        with self.cached_session(use_gpu=True):
            input_layer = Input(4, name="x")
            linear = Linear(input_layer, 2)
            h = Activation(linear, fn=sigmoid)

            # configure training
            labels = Input(2, name="y_")
            losses = binary_cross_entropy(labels.tensor, h.tensor)

            model = Model(input_layer, h,
                          train_loss_tensors=losses,
                          train_loss_in=labels,
                          eval_tensors=losses,
                          eval_tensors_in=labels)
            runner = ModelRunner(model)

            optimiser = tf.train.AdadeltaOptimizer(learning_rate=0.5)
            runner.config_optimizer(optimiser)

            data = np.array([[1, 1, 1, 1]])
            target = np.array([[1.0, 0.0]])

            # session = runner.session
            # weights = session.run(linear.weights)
            runner.init_vars()
            weights1 = runner.session.run(linear.weights)

            for i in range(10):
                runner.train(data, target)

            weights2 = runner.session.run(linear.weights)

            self.assertFalse(np.array_equal(weights1, weights2))

    def test_eval_step_decay_param(self):
        v1 = 4
        decay_rate = 0.5
        param = EvalStepDecayParam(v1,
                                   decay_rate=decay_rate,
                                   improvement_threshold=1.0,
                                   less_is_better=True)
        param.update(evaluation=1)
        v2 = param.value
        self.assertEqual(v1, v2)

        # eval does not improve
        param.update(evaluation=10)
        v3 = param.value
        self.assertNotEqual(v2, v3)
        self.assertEqual(v3, v2 * decay_rate)
        self.assertEqual(param.eval_improvement(), -9)

        # eval improves but not more than threshold
        v4 = param.value
        param.update(evaluation=9)
        self.assertEqual(v4, v3)

        # eval does not improve
        v5 = param.value
        param.update(evaluation=10)
        self.assertEqual(v5, v4 * decay_rate)

        # eval improves but within threshold
        v6 = param.value
        param.update(evaluation=8.9)
        self.assertEqual(v6, v5 * decay_rate)

        # INCREASING EVAL
        param = EvalStepDecayParam(v1, decay_rate=decay_rate, improvement_threshold=1.0, less_is_better=False)
        param.update(evaluation=5)
        v2 = param.value
        self.assertEqual(v1, v2)

        # eval does not improve
        param.update(evaluation=4)
        v3 = param.value
        self.assertEqual(v3, v2 * decay_rate)

        # improvement within threshold / did not improve
        param.update(evaluation=5)
        v4 = param.value
        self.assertEqual(v4, v3 * decay_rate)

        # improvement more than threshold
        param.update(evaluation=6.1)
        v5 = param.value
        self.assertEqual(v5, v4)


if __name__ == '__main__':
    test_utils.main()
