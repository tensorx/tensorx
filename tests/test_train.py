import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorx as tx
from tensorx.test_utils import TestCase
import numpy as np
import logging

class TestTrain(TestCase):

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

        model.set_optimizer(tf.optimizers.SGD, lr=0.5)

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
                                                      model.run_graph.in_nodes,
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

    def test_model_var_inputs(self):
        # wanted to test when our train graph has more inputs that do not need to be fed (e.g. variable state)
        n_features = 5
        embed_size = 4
        hdim = 3
        seq_size = 3
        out_size = 2
        batch_size = 2

        x = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        y = tx.Input(np.random.random([batch_size, out_size]), n_units=out_size, dtype=tf.float32)
        lookup = tx.Lookup(x, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        # seq = lookup.permute_batch_time()
        seq = tx.Transpose(lookup, [1, 0, 2])

        # zero_state = tf.zeros([batch_size, hdim])
        rnn1 = tx.RNN(seq, cell_proto=tx.RNNCell.proto(n_units=hdim))
        y_ = tx.Linear(rnn1[seq_size - 1], n_units=out_size)

        # y_ = tx.Linear(tx.SeqConcat(lookup, seq_size=seq_size), n_units=out_size)

        @tx.layer(n_units=2, dtype=tf.float32, name="loss")
        def loss(pred, labels):
            return tx.mse(pred, labels)

        model = tx.Model(run_inputs=x,
                         run_outputs=y_,
                         train_inputs=[x, y],
                         train_outputs=y_,
                         train_loss=loss(y_, y)
                         )

        model.draw("test.pdf")


        model.set_optimizer(tf.optimizers.SGD, lr=0.5)

        data1 = [[0, 1, 2], [2, 1, 0]]
        data2 = [[0., 1.], [1., 0.]]

        out = model.train_step(input_feed={x: data1, y: data2})

    def test_model_train(self):
        x = tx.Input(n_units=2, name="x", constant=False)
        labels = tx.Input(n_units=2, name="labels", constant=False)
        y = tx.Linear(x, 2, name="y1", add_bias=False)
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

        data1 = [[1., 1.], [1., 1.]]
        data2 = [[0., 1.], [0., 1.]]

        w1 = y.weights.numpy()

        epochs = 100
        prog = tx.Progress()
        # plot = tx.Plot(monitor=["train_loss","last_loss"], keep_open=True)
        model.train(train_data=[{x: data1, labels: data2}], epochs=epochs, callbacks=[prog])

        w2 = y.weights.numpy()

        y.weights.assign(w1)

        for _ in range(epochs):
            model.train_step(input_feed={x: data1, labels: data2})

        w3 = y.weights.numpy()

        self.assertArrayEqual(w2, w3)
