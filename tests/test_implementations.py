import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNNCell, LSTMCell, GRUCell, Attention


def test_rnn_cell():
    n_inputs = 3
    n_units = 4
    batch_size = 2
    inputs = tx.Input(n_units=n_inputs)

    rnn0 = tx.RNNCell(inputs, n_units)

    # Keras RNN cell
    rnn1 = SimpleRNNCell(n_units)
    state = rnn1.get_initial_state(inputs, batch_size=1)
    assert tx.tensor_equal(state, rnn0.previous_state[0]())

    inputs.value = tf.ones([batch_size, n_inputs])
    res1 = rnn1(inputs, (state,))

    rnn1.kernel = rnn0.layer_state.w.weights
    rnn1.bias = rnn0.layer_state.w.bias
    rnn1.recurrent_kernel = rnn0.layer_state.u.weights

    res2 = rnn1(inputs, (state,))
    assert not tx.tensor_equal(res1[0], res2[0])
    assert not tx.tensor_equal(res1[1], res2[1])

    res0 = rnn0()
    assert tx.tensor_equal(res2[0], res0)


def test_lstm_cell():
    n_inputs = 3
    n_units = 4
    batch_size = 1
    inputs = tx.Input(n_units=n_inputs)

    lstm0 = tx.LSTMCell(inputs, n_units,
                        activation=tf.tanh,
                        gate_activation=tf.sigmoid,
                        forget_bias_init=tf.initializers.ones(),
                        )

    lstm1 = LSTMCell(n_units,
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     unit_forget_bias=True,
                     implementation=2)

    state0 = [s() for s in lstm0.previous_state]
    #  get_initial_state from keras returns either a tuple or a single
    #  state see `test_rnn_cell`, but the __call__ API requires an iterable
    state1 = lstm1.get_initial_state(inputs, batch_size=1)

    assert tx.tensor_equal(state1, state0)

    inputs.value = tf.ones([batch_size, n_inputs])
    res1 = lstm1(inputs, state0)
    res1_ = lstm1(inputs, state0)

    for r1, r2 in zip(res1, res1_):
        assert tx.tensor_equal(r1, r2)

    # the only difference is that keras kernels are fused together
    kernel = tf.concat([w.weights.value() for w in lstm0.layer_state.w], axis=-1)
    w_i, _, _, _ = tf.split(kernel, 4, axis=1)
    assert tx.tensor_equal(w_i, lstm0.w[0].weights.value())

    recurrent_kernel = tf.concat([u.weights for u in lstm0.layer_state.u], axis=-1)
    bias = tf.concat([w.bias for w in lstm0.layer_state.w], axis=-1)

    assert tx.tensor_equal(tf.shape(kernel), tf.shape(lstm1.kernel))
    assert tx.tensor_equal(tf.shape(recurrent_kernel), tf.shape(lstm1.recurrent_kernel))
    assert tx.tensor_equal(tf.shape(bias), tf.shape(lstm1.bias))

    lstm1.kernel = kernel
    lstm1.recurrent_kernel = recurrent_kernel
    lstm1.bias = bias

    res2 = lstm1(inputs, state0)
    for i in range(len(res1)):
        assert not tx.tensor_equal(res1[i], res2[i])
    res0 = lstm0()
    assert tx.tensor_equal(res0, res2[0])


def test_lstm_rnn_stateful():
    n_units = 4
    batch_size = 12
    seq_size = 3
    n_features = 16
    embed_size = 6

    feature_indices = np.random.randint(0, high=n_features, size=[batch_size, seq_size])

    inputs = tx.Input(init_value=feature_indices,
                      n_units=seq_size,
                      dtype=tf.int32)
    lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
    seq = lookup.permute_batch_time()

    # (N, T, M)
    # print(np.shape(seq()))

    lstm_cell = tx.LSTMCell.proto(n_units=n_units,
                                  activation=tf.tanh,
                                  gate_activation=tf.sigmoid,
                                  forget_bias_init=tf.initializers.ones()
                                  )

    # state0 = [s() for s in lstm0.previous_state]

    # inputs.value = tf.ones([batch_size, n_features])
    # res1 = lstm1(inputs, state0)
    # res1_ = lstm1(inputs, state0)

    lstm_layer = tx.RNN(input_seq=seq, cell_proto=lstm_cell, stateful=True, return_state=True)
    state0 = [s() for s in lstm_layer.previous_state]
    lstm_layer()
    state1 = [s() for s in lstm_layer.previous_state]

    for i in range(len(state0)):
        assert not tx.tensor_equal(state0[i], state1[i])

    assert np.shape(state1[0]) == (batch_size, n_units)

    tx_cell = lstm_layer.cell
    kernel = tf.concat([w.weights.value() for w in tx_cell.w], axis=-1)
    recurrent_kernel = tf.concat([u.weights.value() for u in tx_cell.u], axis=-1)
    bias = tf.concat([w.bias.value() for w in tx_cell.w], axis=-1)

    # create keras lstm and update with the same cell state
    # since LSTM initializes the cell state internally this was
    # the only way to initializing that state from the tensorx state
    class FromOther(tf.keras.initializers.Initializer):
        def __init__(self, value):
            self.value = value

        def __call__(self, shape, dtype=None):
            if not tf.TensorShape(shape).is_compatible_with(tf.shape(self.value)):
                raise Exception(f"init called with shape {shape} != value shape {tf.shape(self.value)}")
            else:
                return self.value

    # seq = lookup()
    # seq = tf.transpose(seq, [1, 0, 2])

    # lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=n_units)
    # lstm_cell.build(np.shape(seq[0]))

    # full_kernel = tf.concat([kernel, recurrent_kernel], axis=0)
    # lstm_cell = (full_kernel, bias)
    # lstm_cell.weights[0] = full_kernel
    # lstm_cell.weights[1] = bias
    # print(type())

    # print(lstm_cell(seq[0],state=tuple(state1)))
    # rnn = tf.keras.layers.RNN(cell=lstm_cell,
    #                         dtype=tf.float32,
    #                         return_sequences=True,
    #                         time_major=True,
    #                         unroll=False)
    # print(rnn(seq))
    # print(lstm_layer())
    # tf_lstm_output = rnn(seq, tuple(state1))
    # tx_lstm_output = lstm_layer()

    keras_lstm = tf.keras.layers.LSTM(units=n_units,
                                      activation=tf.tanh,
                                      kernel_initializer=FromOther(kernel.numpy()),
                                      recurrent_initializer=FromOther(recurrent_kernel.numpy()),
                                      bias_initializer=FromOther(bias.numpy()),
                                      recurrent_activation=tf.sigmoid,
                                      unit_forget_bias=False,
                                      implementation=2,
                                      time_major=True,
                                      unroll=True,
                                      return_sequences=True,
                                      stateful=False)

    #
    # lookup is of form [batch x features x input_dim] instead of [features x batch x input_dim]
    keras_lstm_output = keras_lstm(seq(), initial_state=tuple(state1))

    assert tx.tensor_equal(keras_lstm.cell.kernel.value(), kernel)
    assert tx.tensor_equal(keras_lstm.cell.recurrent_kernel.value(), recurrent_kernel)
    assert tx.tensor_equal(keras_lstm.cell.bias.value(), bias)

    tx_lstm_output = lstm_layer()[0]
    assert tx.tensor_all_close(keras_lstm_output, tx_lstm_output)


def test_gru_cell():
    n_inputs = 3
    n_units = 4
    batch_size = 1
    inputs = tx.Input(n_units=n_inputs)

    gru0 = tx.GRUCell(inputs, n_units,
                      activation=tf.tanh,
                      gate_activation=tf.sigmoid)

    # applies gate after matrix multiplication and uses
    # recurrent biases, this makes it compatible with cuDNN
    # implementation
    gru1 = GRUCell(n_units,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   reset_after=False,
                   implementation=1,
                   use_bias=True)

    assert not hasattr(gru1, "kernel")

    state0 = [s() for s in gru0.previous_state]
    #  get_initial_state from keras returns either a tuple or a single
    #  state see test_rnn_cell, but the __call__ API requires an iterable
    state1 = gru1.get_initial_state(inputs, batch_size=1)

    assert tx.tensor_equal(state1, state0[0])

    inputs.value = tf.ones([batch_size, n_inputs])

    res1 = gru1(inputs, state0)
    res1_ = gru1(inputs, state0)

    for r1, r2 in zip(res1, res1_):
        assert tx.tensor_equal(r1, r2)

    # the only difference is that keras kernels are fused together
    kernel = tf.concat([w.weights.value() for w in gru0.layer_state.w], axis=-1)
    recurrent_kernel = tf.concat([u.weights for u in gru0.layer_state.u], axis=-1)
    bias = tf.concat([w.bias for w in gru0.layer_state.w], axis=-1)

    assert tx.shape_equal(kernel, gru1.kernel)
    assert tx.shape_equal(recurrent_kernel, gru1.recurrent_kernel)
    assert tx.shape_equal(bias, gru1.bias)

    gru1.kernel = kernel
    gru1.recurrent_kernel = recurrent_kernel
    gru1.bias = bias

    res2 = gru1(inputs, state0)
    for i in range(len(res1)):
        assert not tx.tensor_equal(res1[i], res2[i])
    res0 = gru0()
    # res0_ = gru0.state[0]()
    assert tx.tensor_equal(res0, res2[0])


def test_conv1d():
    n_features = 3
    embed_size = 128
    seq_size = 3
    batch_size = 2

    inputs = tx.Constant(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
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

    assert len(cnn.variables) == len(cnn.variables)

    cnn.kernel = cnn2.filters
    cnn.bias = cnn2.bias
    res3 = cnn(seq)

    assert not tx.tensor_equal(res, res2)
    assert tx.tensor_equal(res2, res3)


def test_attention():
    n_features = 3
    embed_size = 8
    seq_size = 3
    batch_size = 2

    inputs = tx.Constant(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
    emb = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
    seq = emb()

    # keras attention doesn't have multiple heads
    attention = Attention(use_scale=False)

    res = attention([seq, seq, seq])

    attention2 = tx.MHAttention(emb, emb, emb, n_units=embed_size, n_heads=1)
    assert len(attention2.variables) == 3

    attention2.wq = tx.Linear(emb, n_units=None,
                                          weights=tf.linalg.eye(embed_size, embed_size),
                                          add_bias=False)
    attention2.wk = tx.Linear(emb, n_units=None,
                                          weights=tf.linalg.eye(embed_size, embed_size),
                                          add_bias=False)
    attention2.wv = tx.Linear(emb, n_units=None,
                                          weights=tf.linalg.eye(embed_size, embed_size),
                                          add_bias=False)

    assert tx.tensor_equal(attention2.wq(seq), seq)

    res2 = attention2()

    g = tx.Graph.build(inputs=emb, outputs=attention2)
    g = g.as_function(ord_inputs=emb, ord_outputs=attention2)

    res3 = g(seq)

    assert tx.tensor_equal(res, res2)
    assert tx.tensor_equal(res, res3)
