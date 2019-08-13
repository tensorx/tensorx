import unittest
import tensorflow as tf
import tensorx as tx
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestLayers(unittest.TestCase):
    def assertArrayEqual(self, actual, desired, verbose=True):
        if isinstance(actual, tx.Layer):
            actual = actual.tensor()
        if isinstance(desired, tx.Layer):
            desired = desired.tensor()

        self.assertTrue(np.array_equal(actual, desired))

    def assertArrayNotEqual(self, actual, desired):
        if isinstance(actual, tx.Layer):
            actual = actual.tensor()
        if isinstance(desired, tx.Layer):
            desired = desired.tensor()

        self.assertFalse(np.array_equal(actual, desired))

    def test_linear(self):
        inputs = np.ones([2, 4])
        inputs2 = inputs * 2

        linear = tx.Linear(inputs, 8)

        w = linear.weights
        b = linear.bias

        self.assertEqual(w.shape, [4, 8])
        self.assertEqual(b.shape, [8])
        self.assertEqual(len(linear.trainable_vars), 2)

        t1 = linear.tensor()
        t2 = linear.tensor()

        self.assertTrue(np.array_equal(t1.numpy(), t2.numpy()))

        linear2 = tx.Linear(linear.input_layers[0], 8, share_vars_with=linear)
        t3 = linear2.tensor()
        self.assertTrue(np.array_equal(t1.numpy(), t3.numpy()))

        linear = tx.Linear(inputs, 8)
        linear2 = linear.reuse_with(inputs2)

        self.assertEqual(linear.weights, linear2.weights)
        self.assertEqual(linear.bias, linear2.bias)

        # with eager we can do this in the tests
        self.assertTrue(tf.reduce_all(tf.equal(linear.compute() * 2, linear2.compute())))
        # in alternative to
        # self.assertTrue(np.array_equal(linear.compute().numpy()*2, linear2.compute().numpy())

    def test_linear_rank3(self):
        x = tx.TensorLayer([[[1], [1]], [[2], [2]]], dtype=tf.float32)

        x2 = tx.Transpose(x)
        x_flat = tx.Reshape(x, [-1, 1])
        linear1 = tx.Linear(x, n_units=2)
        linear2 = tx.Linear(x2,
                            shape=[2, 1],
                            shared_weights=linear1.weights,
                            transpose_weights=True)

        # we cant do this because it changes the definition of the layer (n_units etc)
        try:
            linear3 = linear1.reuse_with(x2, transpose_weights=True)
            self.fail("we can't reuse with transpose weights while changing the layer definition")
        except ValueError as e:
            pass

        linear_flat = linear1.reuse_with(x_flat)
        linear_flat = tx.Reshape(linear_flat, x.compute().get_shape().as_list()[:-1] + [2])

        self.assertTrue(np.array_equal(linear1.compute(), linear_flat.compute()))
        self.assertTrue(np.array_equal(tf.shape(linear2.compute()), [1, 2, 1]))

    def test_dynamic_input_graph(self):
        x = tx.TensorLayer(tf.zeros([2, 2]), n_units=2, constant=False)
        g = x.compile_graph()

        out1 = g()
        x.value = tf.ones([2, 2])
        out2 = g()

        self.assertArrayNotEqual(out1, out2)

    def test_transpose_reshape(self):
        x = tf.reshape(tf.range(9), [3, 3])
        x2 = tx.Reshape(tf.range(9), [3, 3])

        self.assertArrayEqual(x2(), x)
        self.assertArrayEqual(x2(tf.range(9)), x)
        t = tf.transpose(x)

        y = tx.Transpose(t)
        self.assertArrayEqual(y(), x)
        self.assertArrayEqual(y(x), t)

    def test_wrap_transpose(self):
        x = tf.reshape(tf.range(9), [3, 3])
        t = tf.transpose(x)
        y = tx.Transpose(t, n_units=3)
        w = tx.WrapLayer(y, lambda x: x * 2, apply_to_layer=False)
        self.assertArrayEqual(w(), x * 2)
        self.assertArrayEqual(w(x), t * 2)

    def test_input(self):
        inputs = tx.TensorLayer(n_units=4, dtype=tf.int32, constant=False)
        self.assertArrayEqual(inputs.value, tf.zeros([1, 4]))
        try:
            inputs.tensor()
        except TypeError:
            pass

        try:
            inputs.value = np.ones([2, 3])
            self.fail("should have thrown exception with invalid shape")
        except ValueError as e:
            pass
        inputs.value = np.ones([2, 4])
        self.assertIsNotNone(inputs.value)
        self.assertIsNotNone(inputs.compute())
        self.assertEqual(inputs.compute().dtype, tf.int32)

        # test sparse input
        inputs = tx.TensorLayer(n_units=4, n_active=2, dtype=tf.int64, constant=False)
        self.assertArrayEqual(inputs.value, tf.zeros([0, 2]))

        try:
            inputs.value = [[0, 2, 2]]
        except ValueError as e:
            self.assertTrue("Invalid shape" in str(e))

        inputs.value = [[0, 2]]
        # create an equivalent sparse input
        sp_input = inputs.compute()
        self.assertIsInstance(sp_input, tf.SparseTensor)
        inputs2 = tx.TensorLayer(n_units=4, value=sp_input)

        dense_value = tf.sparse.to_dense(inputs.compute())
        dense_value2 = tf.sparse.to_dense(inputs2.compute())
        expected = np.array([[1, 0, 1, 0]], dtype=np.float32)
        self.assertTrue(np.array_equal(expected, dense_value))
        self.assertTrue(np.array_equal(dense_value, dense_value2))

    def test_variable_layer(self):
        input_layer = tx.TensorLayer([[1]], n_units=1, dtype=tf.float32)
        # print(input_layer.tensor())
        var_layer = tx.VariableLayer(input_layer, dtype=tf.float32)
        # print(var_layer.tensor())

        init_value = var_layer.variable.numpy()
        after_update = var_layer.compute().numpy()

        self.assertArrayNotEqual(init_value, after_update)
        self.assertArrayEqual(after_update, var_layer.variable.numpy())

    def test_variable_init_from_input(self):
        input_layer = tx.TensorLayer(n_units=1, constant=False)
        layer_once = tx.VariableLayer(input_layer, update_once=True)
        layer_var = tx.VariableLayer(input_layer, update_once=False)

        var_layer_fwd2 = layer_once.reuse_with(init_from_input=False)

        data1 = np.array([[1]])
        data2 = np.array([[2]])
        data3 = np.array([[3]])

        input_layer.value = data1
        self.assertEqual(layer_once.counter.numpy(), 0)
        input_layer.value = data2
        y1 = layer_once.compute()
        self.assertEqual(layer_once.counter.numpy(), 1)
        self.assertTrue(np.array_equal(layer_once.variable.value(), y1))
        input_layer.value = data3
        y2 = layer_once.compute()
        self.assertEqual(layer_once.counter.numpy(), 1)
        self.assertTrue(np.array_equal(y1, y2))
        self.assertTrue(np.array_equal(y1, layer_once.variable.value()))

        # dynamic var layer
        input_layer.value = data1
        self.assertEqual(layer_var.counter.numpy(), 0)
        y1 = layer_var.compute()
        self.assertEqual(layer_var.counter.numpy(), 1)
        self.assertTrue(np.array_equal(layer_var.variable.value(), y1))

        input_layer.value = data2
        y2 = layer_var.compute()
        self.assertEqual(layer_var.counter.numpy(), 2)
        self.assertFalse(np.array_equal(y1, y2))

    def test_variable_layer_reuse(self):
        input_layer = tx.TensorLayer([[1]], n_units=1, dtype=tf.float32)
        input_layer2 = tx.TensorLayer([[1], [2]], n_units=1, dtype=tf.float32)
        var1 = tx.VariableLayer(shape=[2, 1])

        var2 = var1.reuse_with(input_layer)
        var3 = var1.reuse_with(input_layer2)

        v0 = var1.compute()
        v1 = var2.compute()
        self.assertFalse(np.array_equal(v0, v1))

        # v0 inner variable changed when we evaluate v1
        v2 = var1.compute()
        self.assertFalse(np.array_equal(v0, v1))

        v3 = var3.compute()
        self.assertFalse(np.array_equal(v2, v3))
        v4 = var1.compute()
        self.assertTrue(np.array_equal(v3, v4))

        # variable batch dimension is dynamic its shape will be different
        self.assertFalse(np.array_equal(np.shape(v4), np.shape(v1)))
        self.assertTrue(np.array_equal(np.shape(v2), np.shape(v1)))

    def test_standalone_variable_layer(self):
        var_layer = tx.VariableLayer(shape=[10])
        self.assertTrue(np.array_equal(np.zeros([10]), var_layer.compute()))

    def test_module(self):
        l1 = tx.TensorLayer([[1]], n_units=1, dtype=tf.float32)
        l2 = tx.TensorLayer([[1]], n_units=1, dtype=tf.float32)
        l3 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l1, l2)
        l4 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l1, l2)
        l5 = tx.Linear(l4, 1)
        in1 = tx.TensorLayer([[1]], n_units=1, dtype=tf.float32)
        l7 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l3, in1)
        l8 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l7, l5)

        in2 = tx.TensorLayer([[1]], n_units=1, dtype=tf.float32, constant=False)
        in3 = tx.TensorLayer([[1]], n_units=1, dtype=tf.float32)

        m = tx.Module([l1, l2, in1], l8)
        with tf.name_scope("module_reuse"):
            m2 = m.reuse_with(in2, in3, in1)

        self.assertTrue(np.array_equal(m.compute(), m2.compute()))
        in2.value = [[3]]
        self.assertFalse(np.array_equal(m.compute(), m2.compute()))

    def test_rnn_cell(self):
        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = tx.TensorLayer(tf.ones([batch_size, n_inputs]))

        rnn_1 = tx.RNNCell(inputs, n_hidden)

        rnn_2 = rnn_1.reuse_with(inputs, rnn_1.state[0].compute())
        rnn_3 = rnn_1.reuse_with(inputs)

        try:
            tx.RNNCell(inputs, n_hidden, share_state_with=inputs)
            self.fail("should have thrown exception because inputs cannot share state with RNNCell")
        except TypeError:
            pass

        res1 = rnn_1.compute()
        res2 = rnn_2.compute()
        res3 = rnn_3.compute()

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertTrue(np.array_equal(res1, res3))
        self.assertFalse(np.array_equal(res1, res2))

    def test_rnn_cell_drop(self):

        n_hidden = 4
        inputs1 = tx.TensorLayer(np.ones([2, 100]), dtype=tf.float32)
        inputs2 = tx.TensorLayer(np.ones([2, 100]), dtype=tf.float32)

        with tf.name_scope("wtf"):
            rnn1 = tx.RNNCell(inputs1, n_hidden,
                              x_dropout=0.5,
                              r_dropout=0.5,
                              u_dropconnect=0.5,
                              w_dropconnect=0.5,
                              regularized=True
                              )
        rnn2 = rnn1.reuse_with(inputs2, previous_state=rnn1)
        rnn3 = rnn1.reuse_with(inputs2, previous_state=rnn1)
        rnn4 = rnn1.reuse_with(inputs2, previous_state=None, regularized=False)
        rnn5 = rnn4.reuse_with(inputs2, previous_state=None, regularized=True)

        r1, r2, r3, r4, r5 = rnn1.compute(), rnn2.compute(), rnn3.compute(), rnn4.compute(), rnn5.compute()
        # w is a linear layer from the input but a regularized layer applies dropout to the input, so we have a dropout
        # in between

        # without a shared state object, we couldn't rewire graphs, in the case of non-eager we can share a tensor
        # that is already wired with something (it takes the shape of the input of one layer and creates a mask tensor
        # shared across dropout instances
        # Linear layers should have shared states as well, in this case sharing the weights
        dropout_state1 = rnn1.w.input_layers[0].layer_state
        dropout_state2 = rnn2.w.input_layers[0].layer_state
        dropout_state3 = rnn3.w.input_layers[0].layer_state

        mask1, mask2, mask3 = dropout_state1.mask, dropout_state2.mask, dropout_state3

        # TODO MASKS are probabilistic so this could fail
        # self.assertArrayEqual(mask1, mask2)
        # self.assertArrayNotEqual(r1, r2)

        self.assertArrayEqual(r2, r3)
        self.assertArrayNotEqual(r2, r4)
        self.assertArrayNotEqual(r4, r5)

        mask1, mask2 = rnn1.w.layer_state, rnn2.w.layer_state
        self.assertArrayEqual(mask1, mask2)

    def test_gru_cell(self):
        n_inputs = 4
        n_hidden = 2
        batch_size = 2
        data = np.ones([batch_size, 4])

        inputs = tx.TensorLayer(value=data, n_units=n_inputs, dtype=tf.float32)

        rnn_1 = tx.GRUCell(inputs, n_hidden)

        rnn_2 = rnn_1.reuse_with(input_layer=inputs,
                                 previous_state=rnn_1)

        # if we don't wipe the memory it reuses it
        # print([batch_size, rnn_1.n_units])
        rnn_3 = rnn_1.reuse_with(inputs,
                                 previous_state=tx.GRUCell.zero_state(rnn_1.n_units))

        res1 = rnn_1.tensor()
        res2 = rnn_2.tensor()
        res3 = rnn_3.tensor()

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertArrayEqual(res1, res3)
        self.assertArrayNotEqual(res1, res2)

    def test_module_gate(self):
        l1 = tx.TensorLayer(n_units=4, name="in1", constant=False)
        l2 = tx.TensorLayer(n_units=2, name="in2", constant=False)

        gate = tx.Gate(layer=l1, gate_input=l2, gate_fn=tf.sigmoid)
        gate_module = tx.Module([l1, l2], gate)

        t1 = tx.TensorLayer([[1, 1, 1, 1]], n_units=4, dtype=tf.float32)
        t2 = tx.TensorLayer([[1, 1]], n_units=2, dtype=tf.float32)

        m2 = gate_module.reuse_with(t1, t2)

        l1.value = t1.tensor()
        l2.value = t2.tensor()

        result1 = gate_module.compute()
        result2 = m2.compute()
        result3 = gate_module.compute(t1, t2)

        self.assertArrayEqual(result1, result2)
        self.assertArrayEqual(result2, result3)

    def test_lstm_cell(self):
        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = tx.TensorLayer(np.ones([batch_size, n_inputs], np.float32), n_units=n_inputs)
        rnn1 = tx.LSTMCell(inputs, n_hidden, gate_activation=tf.sigmoid)
        previous_state = (None, rnn1.state[-1].tensor())
        rnn2 = rnn1.reuse_with(inputs, previous_state=previous_state)

        # if we don't wipe the memory it reuses it
        previous_state = (None, tx.LSTMCell.zero_state(rnn1.n_units))
        rnn3 = rnn1.reuse_with(inputs, previous_state=previous_state)

        res1 = rnn1.tensor()
        res2 = rnn2.tensor()
        res3 = rnn3.tensor()

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertArrayEqual(res1, res3)
        self.assertArrayNotEqual(res1, res2)

    def test_lstm_cell_regularization(self):

        n_inputs = 8
        n_hidden = 2
        batch_size = 2

        inputs = tx.TensorLayer(n_units=n_inputs, constant=False)

        rnn1 = tx.LSTMCell(inputs, n_hidden,
                           u_dropconnect=0.1,
                           w_dropconnect=0.1,
                           name="lstm1")

        rnn2 = rnn1.reuse_with(inputs,
                               previous_state=rnn1.state,
                               regularized=True,
                               name="lstm2"
                               )

        rnn3 = rnn2.reuse_with(inputs,
                               previous_state=rnn1.state,
                               name="lstm3"
                               )

        data = np.ones([batch_size, n_inputs])

        inputs.value = data

        self.assertArrayEqual(rnn2, rnn3)
        self.assertArrayNotEqual(rnn1, rnn3)
        self.assertArrayNotEqual(rnn1, rnn3)

        state2, state3 = rnn2.w_f.weight_mask, rnn3.w_f.weight_mask
        self.assertArrayEqual(state2, state3)

        w2, w3 = rnn2.w_f, rnn3.w_f
        self.assertArrayEqual(w2, w3)
        w2, w3 = rnn2.w_i, rnn3.w_i
        self.assertArrayEqual(w2, w3)
        w2, w3 = rnn2.w_o, rnn3.w_o
        self.assertArrayEqual(w2, w3)
        w2, w3 = rnn2.w_c, rnn3.w_c
        self.assertArrayEqual(w2, w3)

    def test_rnn_layer(self):
        n_features = 5
        embed_size = 4
        hdim = 3
        seq_size = 3
        batch_size = 2

        inputs = tx.TensorLayer(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = lookup.permute_batch_time()

        ones_state = tf.ones([batch_size, hdim])
        zero_state = (tf.zeros([batch_size, hdim]))

        def rnn_proto(x, **kwargs):
            return tx.RNNCell(x, n_units=hdim, **kwargs)

        rnn1 = tx.RNN(seq, cell_proto=rnn_proto, previous_state=ones_state)
        rnn2 = rnn1.reuse_with(seq)

        out1, last1 = rnn1.tensor()
        out2, last2 = rnn2.tensor()

        self.assertArrayEqual(out1, out2)
        self.assertArrayEqual(last1, last2)

        rnn3 = rnn1.reuse_with(seq, previous_state=zero_state)
        rnn4 = rnn3.reuse_with(seq)
        rnn5 = rnn4.reuse_with(seq, previous_state=ones_state)

        self.assertArrayEqual(rnn2.previous_state, rnn1.previous_state)
        self.assertArrayEqual(rnn3.previous_state, rnn4.previous_state)

        out3, last3 = rnn3.tensor()
        out4, last4 = rnn4.tensor()

        self.assertArrayEqual(out3, out4)
        self.assertArrayEqual(last3, last4)

        cell_state1 = rnn1.cell.previous_state[0].tensor()
        cell_state2 = rnn2.cell.previous_state[0].tensor()
        cell_state3 = rnn3.cell.previous_state[0].tensor()
        cell_state4 = rnn4.cell.previous_state[0].tensor()

        self.assertEqual(len(rnn1.cell.previous_state), 1)

        self.assertArrayEqual(cell_state1, cell_state2)
        self.assertArrayEqual(cell_state3, cell_state4)

        self.assertArrayNotEqual(out1, out3)

        out5, last5 = rnn5.tensor()

        self.assertArrayEqual(out1, out5)
        self.assertArrayEqual(last1, last5)

    def test_stateful_rnn_layer(self):
        n_features = 5
        embed_size = 4
        hdim = 3
        seq_size = 3
        batch_size = 2

        inputs = tx.TensorLayer(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = lookup.permute_batch_time()

        def rnn_proto(x, **kwargs):
            return tx.RNNCell(x, n_units=hdim, **kwargs)

        # print("slice", seq.tensor()[0])
        rnn1 = tx.RNN(seq, cell_proto=rnn_proto, stateful=True)
        lstm1 = tx.RNN(seq, cell_proto=tx.LSTMCell.proto(n_units=hdim), stateful=True)

        zero_state0 = [layer.tensor() for layer in rnn1.previous_state]

        #print(zero_state0[0])
        self.assertEqual(len(zero_state0), 1)
        self.assertArrayEqual(zero_state0[0], np.zeros([1, hdim]))

        # g = rnn1.compile_graph()
        # run once get output and last state
        out1, state1 = rnn1()

        out2, state2, memory2 = lstm1()

        # state after single run
        # zero_state1 = [layer.tensor() for layer in ]
        zero_state1 = rnn1.previous_state[0].tensor()
        self.assertArrayEqual(zero_state1, state1)

        rnn1.reset()
        reset_state = rnn1.previous_state[0].tensor()
        self.assertArrayEqual(reset_state, zero_state0[0])

    def test_lookup_sequence_dense(self):
        input_dim = 4
        embed_dim = 3
        seq_size = 2
        batch_size = 3

        inputs = tx.TensorLayer(np.array([[2, 0], [1, 2]]), 2, dtype=tf.int64)
        tensor_input = tx.TensorLayer(tf.constant([2]), 1, dtype=tf.int64)

        lookup = tx.Lookup(inputs, seq_size,
                           embedding_shape=[input_dim, embed_dim],
                           batch_size=batch_size,
                           batch_padding=True)

        lookup_from_tensor = lookup.reuse_with(tensor_input)

        v1 = lookup.tensor()
        v2 = lookup_from_tensor.tensor()

        self.assertEqual(np.shape(v1), (batch_size, seq_size, embed_dim))
        self.assertEqual(np.shape(v2), (batch_size, seq_size, embed_dim))

    def test_lookup_dynamic_sequence(self):
        seq1 = [[1, 2], [3, 4]]
        seq2 = [[1, 2, 3], [4, 5, 6]]

        n = 10
        m = 4

        inputs = tx.TensorLayer(dtype=tf.int32, constant=False)

        lookup = tx.Lookup(inputs, seq_size=None, embedding_shape=[n, m])
        concat = lookup.as_concat()

        inputs.value = seq1
        r1 = inputs.tensor()

        inputs.value = seq2
        r2 = inputs.tensor()

        inputs.value = seq1
        l1 = lookup.tensor()
        inputs.value = seq2
        l2 = lookup.tensor()

        inputs.value = seq1
        c1 = concat.tensor()
        inputs.value = seq2
        c2 = concat.tensor()

        self.assertEqual(np.shape(l1)[-1], m)
        self.assertEqual(np.shape(l2)[-1], m)

        self.assertEqual(np.shape(c1)[-1], m * 2)
        self.assertEqual(np.shape(c2)[-1], m * 3)

    def test_lookup_dynamic_sparse_sequence(self):
        k = 8
        m = 3
        seq1 = tf.SparseTensor(
            indices=[[0, 1], [1, 2],
                     [2, 3], [3, 4]],
            values=[1, 2, 3, 4],
            dense_shape=[4, k]
        )
        seq2 = tf.SparseTensor(
            indices=[[0, 1], [1, 2], [2, 3],
                     [3, 3], [4, 4], [5, 5]],
            values=[1, 2, 3, 3, 4, 5],
            dense_shape=[6, k]
        )

        inputs = tx.TensorLayer(n_units=k, sparse=True, dtype=tf.int32, constant=False)
        # print(inputs.value)
        seq_len = tx.TensorLayer(value=2, shape=[], constant=False)
        lookup = tx.Lookup(inputs, seq_size=seq_len, embedding_shape=[k, m])
        # concat = lookup.as_concat()

        inputs.value = seq1
        in1 = inputs.tensor()
        # set seq_len to 2
        l1 = lookup.tensor()

        # set seq len to 3
        inputs.value = seq2
        seq_len.value = 3
        l2 = lookup.tensor()

        # c1 = concat.tensor, {inputs.placeholder: seq1, seq_len.placeholder: 2})
        # c2 = self.eval(concat.tensor, {inputs.placeholder: seq2, seq_len.placeholder: 3})
        # print(in1)

        # self.assertEqual(np.shape(c1)[-1], m * 2)
        # self.assertEqual(np.shape(c2)[-1], m * 3)


def test_lookup_sequence_sparse(self):
    input_dim = 10
    embed_dim = 3
    seq_size = 2
    batch_size = 3

    sparse_input = tf.SparseTensor([[0, 2], [1, 0], [2, 1]], [1, 1, 1], [3, input_dim])
    sparse_input_1d = tf.SparseTensor([[2], [0], [1]], [1, 1, 1], [input_dim])
    tensor_input = TensorLayer(sparse_input, input_dim)
    tensor_input_1d = TensorLayer(sparse_input_1d, input_dim)

    lookup = Lookup(tensor_input, seq_size,
                    lookup_shape=[input_dim, embed_dim],
                    batch_size=batch_size,
                    batch_padding=False)

    lookup_padding = Lookup(tensor_input, seq_size,
                            lookup_shape=[input_dim, embed_dim],
                            batch_size=batch_size,
                            batch_padding=True)

    lookup_1d = Lookup(tensor_input_1d, seq_size,
                       lookup_shape=[input_dim, embed_dim],
                       batch_size=batch_size,
                       batch_padding=True)

    init = tf.global_variables_initializer()
    with self.cached_session(use_gpu=True):
        self.eval(init)

        result = self.eval(lookup.tensor)
        result_padding = self.eval(lookup_padding.tensor)
        result_1d = self.eval(lookup_1d.tensor)

        self.assertEqual(np.shape(result), (2, seq_size, embed_dim))
        self.assertEqual(np.shape(result_padding), (batch_size, seq_size, embed_dim))
        self.assertEqual(np.shape(result_1d), (batch_size, seq_size, embed_dim))


def test_lookup_sparse_padding(self):
    input_dim = 6
    embed_dim = 3
    seq_size = 1

    sparse_input = tf.SparseTensor([[0, 1], [0, 3], [1, 0]], [1, 1, 1], [2, input_dim])
    sparse_input = TensorLayer(sparse_input, input_dim)

    lookup = Lookup(sparse_input,
                    seq_size=seq_size,
                    lookup_shape=[input_dim, embed_dim],
                    batch_size=None,
                    batch_padding=False)

    init = tf.global_variables_initializer()
    with self.cached_session(use_gpu=True):
        self.eval(init)
        self.eval(lookup.tensor)


def test_lookup_sequence_bias(self):
    vocab_size = 4
    n_features = 3
    seq_size = 2

    inputs = Input(seq_size, dtype=tf.int32)
    input_data = np.array([[2, 0], [1, 2], [0, 2]])
    lookup = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features], bias=True)

    init = tf.global_variables_initializer()
    with self.cached_session(use_gpu=True):
        self.eval(init)
        v1 = self.eval(lookup.tensor, {inputs.placeholder: input_data})
        self.assertEqual(np.shape(v1), (np.shape(input_data)[0], seq_size, n_features))


def test_lookup_sequence_transform(self):
    vocab_size = 4
    embed_dim = 2
    seq_size = 2

    inputs = Input(seq_size, dtype=tf.int32)
    input_data = np.array([[2, 0], [1, 2], [0, 2]])
    lookup = Lookup(inputs, seq_size, lookup_shape=[vocab_size, embed_dim], bias=True)
    concat_lookup = lookup.as_concat()
    seq_lookup = lookup.permute_batch_time()

    self.assertTrue(hasattr(lookup, "seq_size"))

    init = tf.global_variables_initializer()
    with self.cached_session(use_gpu=True):
        self.eval(init)

        v1 = self.eval(lookup.tensor, {inputs.placeholder: input_data})
        v2 = self.eval(concat_lookup.tensor, {inputs.placeholder: input_data})
        v3 = self.eval(seq_lookup.tensor, {inputs.placeholder: input_data})

        self.assertEqual(np.shape(v1), (np.shape(input_data)[0], seq_size, embed_dim))
        self.assertEqual(np.shape(v2), (np.shape(input_data)[0], seq_size * embed_dim))

        self.assertEqual(np.shape(v3), (seq_size, np.shape(input_data)[0], embed_dim))
        self.assertTrue(np.array_equal(v1[:, 0], v3[0]))


if __name__ == '__main__':
    unittest.main()
