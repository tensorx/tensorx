import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorx as tx
from tensorx.test_utils import TestCase
import numpy as np


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

    def test_model(self):
        data1 = [[1., 1.]]
        data2 = [[0., 1.]]

        x = tx.Input(n_units=2, name="x", constant=False)
        labels = tx.Input(n_units=2, name="y_", constant=False)
        y = tx.Linear(x, 2, name="y")
        out = tx.Activation(y, tf.nn.softmax)
        out2 = tx.Activation(y, tf.nn.softmax)

        @tx.layer(n_units=2, name="loss")
        def loss(pred, labels):
            return tf.losses.categorical_crossentropy(labels, pred)

        model = tx.Model(run_inputs=x,
                         run_outputs=[out,out2],
                         train_inputs=[x, labels],
                         train_outputs=out,
                         train_loss=loss(out, labels)
                         )

        result1 = model.run({x: data1})
        result2 = model.run([data1])
        result3 = model.run({x: data1}, compiled_graph=True)

        print(result1)
        print(result2)
        print(result3)
