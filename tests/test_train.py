import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorx as tx
from tensorx.test_utils import TestCase
import numpy as np
import logging


class TestTrain(TestCase):
    def test_layer_graph(self):
        data = [[1., 2.]]

        in1 = tx.Input(n_units=2, name="in1", constant=False)
        in2 = tx.Input(n_units=2, name="in2", constant=False)
        linear = tx.Linear(in1, 1, add_bias=False)
        graph = tx.Graph.build(inputs=in1, outputs=linear)

        try:
            tx.Graph.build(inputs=[in1, in2], outputs=linear)
            self.fail("should have raised an exception: some inputs are not connected to anything")
        except ValueError:
            pass

        try:
            tx.Graph.build(inputs=[in2], outputs=linear)
            self.fail("should have raised an error: inputs specified but dependencies are missing")
        except ValueError:
            pass

        w = tf.matmul(data, linear.weights)

        in1.value = data
        r1 = linear()
        r2 = graph(data)

        self.assertArrayEqual(r2[0], w)
        self.assertArrayEqual(r1, w)

    def test_multi_output_graph(self):
        data1 = [[1., 1.]]
        data2 = [[2., 1.]]

        in1 = tx.Input(data1, 2, name="in1", constant=False)
        in2 = tx.Input(data2, 2, name="in2")

        linear1 = tx.Linear(in1, 1)
        linear2 = tx.Linear(tx.Add(in1, in2), 1)

        graph = tx.Graph.build(inputs=None, outputs=[linear1, linear2])

        result1 = graph()
        self.assertEqual(len(result1), 2)

        graph2 = tx.Graph.build(inputs=None, outputs=[linear2])
        result2 = graph2()
        self.assertEqual(len(result2), 1)
        self.assertArrayEqual(result2[0], result1[-1])

    def test_model_run(self):
        data1 = [[1., 1.]]

        x = tx.Input(n_units=2, name="x", constant=False)
        labels = tx.Input(n_units=2, name="y_", constant=False)
        y = tx.Linear(x, 2, name="y")
        out1 = tx.Activation(y, tf.nn.softmax)
        out2 = tx.Activation(y, tf.nn.softmax)

        @tx.layer(n_units=2, name="loss")
        def loss(pred, labels):
            return tf.losses.categorical_crossentropy(labels, pred)

        model = tx.Model(run_inputs=x,
                         run_outputs=[out1, out2],
                         train_inputs=[x, labels],
                         train_outputs=out1,
                         train_loss=loss(out1, labels)
                         )

        result1 = model.run({x: data1})
        result2 = model.run([data1])

        self.assertArrayEqual(result1[0], result2[0])
        self.assertArrayEqual(result1[1], result2[1])

        result3 = model.run({x: data1}, compiled_graph=True)
        self.assertArrayEqual(result3[0], result2[0])
        self.assertArrayEqual(result3[1], result2[1])

    def test_loss_model_dependencies(self):
        x = tx.Input(n_units=2, name="x", constant=False)
        labels = tx.Input(n_units=2, name="y_", constant=False)
        y = tx.Linear(x, 2, name="y")
        out1 = tx.Activation(y, tf.nn.softmax, name="out1")
        out2 = tx.Activation(y, tf.nn.softmax, name="out2")

        @tx.layer(n_units=2, name="loss")
        def loss(pred, labels):
            return tf.losses.categorical_crossentropy(labels, pred)

        logging.basicConfig(level=logging.DEBUG)

        model = tx.Model(run_inputs=x,
                         run_outputs=[out1, out2],
                         train_inputs=[x, labels],
                         train_outputs=[out2, out1],
                         train_loss=loss(out1, labels)
                         )

        lr = tx.Param(0.5)
        opt = model.set_optimizer(tf.optimizers.SGD, lr=lr)

        it = model.train_graph.dependency_iter()

    def test_set_optimizer(self):
        x = tx.Input(n_units=2, name="x", constant=False)
        labels = tx.Input(n_units=2, name="labels", constant=False)
        y = tx.Linear(x, 2, name="y")
        out1 = tx.Activation(y, tf.nn.softmax)
        out2 = tx.Activation(y, tf.nn.softmax)

        @tx.layer(n_units=2, name="loss")
        def loss(pred, labels):
            return tf.losses.categorical_crossentropy(labels, pred)

        model = tx.Model(run_inputs=x,
                         run_outputs=[out1, out2],
                         train_inputs=[x, labels],
                         train_outputs=[out2, out1],
                         train_loss=loss(out1, labels)
                         )

        lr = tx.Param(0.5)
        opt = model.set_optimizer(tf.optimizers.SGD,
                                  learning_rate=lr,
                                  clipnorm=0.1)

        self.assertIsInstance(opt, tf.optimizers.Optimizer)

        self.assertEqual(model.optimizer.lr, 0.5)

        data1 = [[1., 1.], [1., 1.]]
        data2 = [[0., 1.], [0., 1.]]
        params = model.optimizer_params[model.optimizer]
        data_dict, params_dict = tx.Model.parse_input({x: data1,
                                                       "learning_rate": 0.2},
                                                      model.run_graph,
                                                      params)
        self.assertEqual(len(data_dict), 1)
        self.assertEqual(len(params_dict), 1)
        self.assertIs(model.optimizer_params[opt]["learning_rate"], lr)

        result1 = model.train_step({x: data1, labels: data2})
        result2 = model.train_step([data1, data2])

        self.assertEqual(len(result1), 3)
        self.assertEqual(len(result2), 3)
        np.testing.assert_array_less(result2[-1], result1[-1])

        result1 = model.run({x: np.array(data1, dtype=np.float32)})
        result2 = model.run([data1])
        result3 = model.run(np.array(data1, np.float32))

        x.value = data1
        o2 = out2()
        o1 = out1()

        result4 = (o2, o1)

        for i in range(2):
            self.assertArrayEqual(result1[i], result2[i])
            self.assertArrayEqual(result1[i], result3[i])
            self.assertArrayEqual(result1[i], result4[i])
