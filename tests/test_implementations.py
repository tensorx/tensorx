# suppressing messages only works if set before tensorflow is imported
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorx as tx
import numpy as np

import unittest
from tensorx.test_utils import TestCase
from tensorflow.keras.layers import SimpleRNNCell, LSTMCell, GRUCell, Attention, Conv1D


class MyTestCase(TestCase):
    def test_rnn_cell(self):
        n_inputs = 3
        n_units = 4
        batch_size = 2
        inputs = tx.Input(n_units=n_inputs)

        rnn0 = tx.RNNCell(inputs, n_units)

        # Keras RNN cell
        rnn1 = SimpleRNNCell(n_units)
        state = rnn1.get_initial_state(inputs, batch_size=1)
        self.assertArrayEqual(state, rnn0.previous_state[0]())

        inputs.value = tf.ones([batch_size, n_inputs])
        res1 = rnn1(inputs, (state,))

        rnn1.kernel = rnn0.layer_state.w.weights
        rnn1.bias = rnn0.layer_state.w.bias
        rnn1.recurrent_kernel = rnn0.layer_state.u.weights

        res2 = rnn1(inputs, (state,))
        self.assertArrayNotEqual(res1, res2)

        res0 = rnn0()
        self.assertArrayEqual(res2[0], res0)

    def test_lstm_cell(self):
        from tensorflow.keras.backend import dot
        n_inputs = 3
        n_units = 4
        batch_size = 1
        inputs = tx.Input(n_units=n_inputs)

        lstm0 = tx.LSTMCell(inputs, n_units,
                            activation=tf.tanh,
                            gate_activation=tf.sigmoid,
                            forget_bias_init=tf.initializers.ones(),
                            # y_dropout=0.5,
                            # regularized=True
                            )

        lstm1 = LSTMCell(n_units,
                         activation='tanh',
                         recurrent_activation='sigmoid',
                         unit_forget_bias=True,
                         implementation=2)

        state0 = [s() for s in lstm0.previous_state]
        # Note: get_initial_state from keras returns either a tuple or a single
        #  state see `test_rnn_cell`, but the __call__ API requires an iterable
        state1 = lstm1.get_initial_state(inputs, batch_size=1)

        self.assertArrayEqual(state1, state0)

        inputs.value = tf.ones([batch_size, n_inputs])
        res1 = lstm1(inputs, state0)
        res1_ = lstm1(inputs, state0)

        for r1, r2 in zip(res1, res1_):
            self.assertArrayEqual(r1, r2)

        # the only difference is that keras kernels are fused together
        kernel = tf.concat([w.weights.value() for w in lstm0.layer_state.w], axis=-1)
        w_i, _, _, _ = tf.split(kernel, 4, axis=1)
        self.assertArrayEqual(w_i, lstm0.w[0].weights.value())

        recurrent_kernel = tf.concat([u.weights for u in lstm0.layer_state.u], axis=-1)
        bias = tf.concat([w.bias for w in lstm0.layer_state.w], axis=-1)

        self.assertArrayEqual(tf.shape(kernel), tf.shape(lstm1.kernel))
        self.assertArrayEqual(tf.shape(recurrent_kernel), tf.shape(lstm1.recurrent_kernel))
        self.assertArrayEqual(tf.shape(bias), tf.shape(lstm1.bias))

        lstm1.kernel = kernel
        lstm1.recurrent_kernel = recurrent_kernel
        lstm1.bias = bias

        res2 = lstm1(inputs, state0)
        self.assertArrayNotEqual(res1, res2)
        res0 = lstm0()
        self.assertArrayEqual(res0, res2[0])

    def test_gru_cell(self):
        n_inputs = 3
        n_units = 4
        batch_size = 1
        inputs = tx.Input(n_units=n_inputs)

        gru0 = tx.GRUCell(inputs, n_units,
                          activation=tf.tanh,
                          gate_activation=tf.sigmoid)

        # after applies gate after matrix multiplication and uses
        # recurrent biases, this makes it compatible with cuDNN implementation
        gru1 = GRUCell(n_units,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       reset_after=False,
                       implementation=1,
                       use_bias=True)
        try:
            gru1.kernel
        except AttributeError as e:
            pass
            # keras layers are not initialized without running for the first time

        state0 = [s() for s in gru0.previous_state]
        # Note: get_initial_state from keras returns either a tuple or a single
        #  state see `test_rnn_cell`, but the __call__ API requires an iterable
        state1 = gru1.get_initial_state(inputs, batch_size=1)

        self.assertArrayEqual((state1,), state0)

        inputs.value = tf.ones([batch_size, n_inputs])

        res1 = gru1(inputs, state0)
        res1_ = gru1(inputs, state0)

        for r1, r2 in zip(res1, res1_):
            self.assertArrayEqual(r1, r2)

        # the only difference is that keras kernels are fused together
        kernel = tf.concat([w.weights.value() for w in gru0.layer_state.w], axis=-1)
        recurrent_kernel = tf.concat([u.weights for u in gru0.layer_state.u], axis=-1)
        bias = tf.concat([w.bias for w in gru0.layer_state.w], axis=-1)

        self.assertArrayEqual(tf.shape(kernel), tf.shape(gru1.kernel))
        self.assertArrayEqual(tf.shape(recurrent_kernel), tf.shape(gru1.recurrent_kernel))
        self.assertArrayEqual(tf.shape(bias), tf.shape(gru1.bias))

        gru1.kernel = kernel
        gru1.recurrent_kernel = recurrent_kernel
        gru1.bias = bias

        res2 = gru1(inputs, state0)
        self.assertArrayNotEqual(res1, res2)
        res0 = gru0()
        res0_ = gru0.state[0]()
        self.assertArrayEqual(res0, res2[0])

    def test_conv1d(self):
        n_features = 3
        embed_size = 128
        seq_size = 3
        batch_size = 2

        inputs = tx.Tensor(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        emb = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = emb()

        n_units = 100
        filter_size = 4
        cnn = tf.keras.layers.Conv1D(
            filters=n_units,
            kernel_size=filter_size,
            padding='same')

        res = cnn(seq)

        cnn2 = tx.Conv1D(emb, n_units=100, filter_size=filter_size)
        res2 = cnn2(seq)

        self.assertTrue(len(cnn.variables), len(cnn.variables))

        cnn.kernel = cnn2.filters
        cnn.bias = cnn2.bias

        res3 = cnn(seq)

        self.assertArrayNotEqual(res, res2)
        self.assertArrayEqual(res2, res3)

    def test_attention(self):
        n_features = 3
        embed_size = 8
        seq_size = 3
        batch_size = 2

        inputs = tx.Tensor(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        emb = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = emb()

        # keras attention doesn't have multiple heads
        attention = Attention(use_scale=False)

        res = attention([seq, seq, seq])
        # print(np.shape(res))
        # print(attention.variables)

        attention2 = tx.MHAttention(emb, emb, emb, n_units=embed_size, n_heads=1)
        self.assertTrue(len(attention2.variables), 3)

        attention2.wq = tx.Linear(emb, n_units=None,
                                  shared_weights=tf.linalg.eye(embed_size, embed_size),
                                  add_bias=False)
        attention2.wk = tx.Linear(emb, n_units=None,
                                  shared_weights=tf.linalg.eye(embed_size, embed_size),
                                  add_bias=False)
        attention2.wv = tx.Linear(emb, n_units=None,
                                  shared_weights=tf.linalg.eye(embed_size, embed_size),
                                  add_bias=False)
        # print(attention2.layer_state.wq.weights)
        # print(attention2.wq.weights)
        self.assertArrayEqual(attention2.wq(seq), seq)

        res2 = attention2()

        self.assertArrayEqual(res, res2)


if __name__ == '__main__':
    unittest.main()
