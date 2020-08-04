# suppressing messages only works if set before tensorflow is imported

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

import tensorx as tx
import numpy as np
import unittest
from tensorx.testing import TestCase
from tensorx.layers import LayerProto
import tensorflow as tf


class TestLayers(TestCase):

    def test_input(self):
        inputs = tx.Input(n_units=4, dtype=tf.int32, constant=False)
        self.assertArrayEqual(inputs.value, tf.zeros([1, 4]))
        try:
            inputs.tensor()
        except TypeError:
            pass

        try:
            inputs.value = np.ones([2, 3], dtype=np.int32)
            self.fail("should have thrown exception with invalid shape")
        except ValueError as e:
            pass
        inputs.value = np.ones([2, 4], dtype=np.int32)
        self.assertIsNotNone(inputs.value)
        self.assertIsNotNone(inputs())
        self.assertEqual(inputs().dtype, tf.int32)

        # test sparse input
        inputs = tx.Input(n_units=4, n_active=2, dtype=tf.int64, constant=False)
        self.assertArrayEqual(inputs.value, tf.zeros([0, 2]))

        try:
            inputs.value = [[0, 2, 2]]
        except ValueError as e:
            self.assertTrue("Invalid shape" in str(e))

        inputs.value = [[0, 2]]
        # create an equivalent sparse input
        sp_input = inputs()
        self.assertIsInstance(sp_input, tf.SparseTensor)
        inputs2 = tx.Input(n_units=4, init_value=sp_input)

        dense_value = tf.sparse.to_dense(inputs())
        dense_value2 = tf.sparse.to_dense(inputs2())
        expected = np.array([[1, 0, 1, 0]], dtype=np.float32)
        self.assertTrue(np.array_equal(expected, dense_value))
        self.assertTrue(np.array_equal(dense_value, dense_value2))

    def test_input_3d(self):
        # we either create a 3d input specifying the shape
        data = np.ones([2, 2, 2], dtype=np.float32)
        x = tx.Input(shape=[None, None, 2], dtype=tf.float32, n_units=2)
        x.value = data
        x.value = x() * 2
        self.assertArrayEqual(data * 2, x())

        x2 = tx.Input(data)
        self.assertEqual(x2.n_units, np.shape(data)[-1])
        self.assertEqual(x2.shape[-1], np.shape(data)[-1])
        x2.value = x2() * 2
        self.assertArrayEqual(data * 2, x2())

        try:
            x3 = tx.Input(n_units=2)
            x3.value = data
        except ValueError:
            # TODO can we delay Input slots to set value ?
            #  to do that we would have to create it if we called compute() without setting the value
            #  otherwise it would be created on value set
            pass

    def test_dynamic_input_graph(self):
        # tf.autograph.set_verbosity(
        #    10,
        #    alsologtostdout=True
        # )
        # issue with python 3.8
        # https://github.com/tensorflow/tensorflow/issues/34433

        x = tx.Input(tf.zeros([2, 2]), n_units=2, constant=False)
        # p = x.proto
        g = x.as_function(input_signature=None)

        out1 = g()
        x.value = tf.ones([2, 2])
        out2 = g()

        self.assertArrayNotEqual(out1, out2)

    def test_layer_proto(self):
        inputs = tx.Input(init_value=tf.ones([2, 2]), n_units=2)
        inputs_proto = inputs.proto
        l2 = inputs_proto()
        self.assertArrayEqual(inputs(), l2())

        # linear = tx.Linear(inputs,n_units=3)
        # cfg1 = linear.proto

        rnncell = tx.RNNCell(input_layer=inputs, n_units=3)

        class KWClass:
            def __init__(self, param1, **kwargs):
                self.param1 = param1

        proto = LayerProto(KWClass, param2=1, param1=2)
        p = proto()
        self.assertEqual(p.param1, 2)

    def test_shared_state(self):
        inputs = np.ones([2, 4])
        l1 = tx.Linear(inputs, 8)

        l2 = tx.Linear(inputs, 8, share_state_with=l1)

        p = tx.Linear.proto(n_units=8, share_state_with=l1)
        l3 = p(inputs)

        self.assertIs(l1.weights, l2.weights)
        self.assertIs(l1.bias, l2.bias)
        self.assertIs(l1.weights, l3.weights)
        self.assertIs(l1.bias, l3.bias)

    def test_linear(self):
        inputs = np.ones([2, 4])
        inputs2 = inputs * 2

        linear = tx.Linear(inputs, 8)

        w = linear.weights
        b = linear.bias

        self.assertEqual(w.shape, [4, 8])
        self.assertEqual(b.shape, [8])
        self.assertEqual(len(linear.trainable_variables), 2)

        t1 = linear.tensor()
        t2 = linear.tensor()

        self.assertTrue(np.array_equal(t1.numpy(), t2.numpy()))

        linear2 = tx.Linear(linear.input_layers[0], 8, share_state_with=linear)
        t3 = linear2.tensor()
        self.assertTrue(np.array_equal(t1.numpy(), t3.numpy()))

        linear = tx.Linear(inputs, 8)
        linear2 = linear.reuse_with(inputs2)

        # Can't use equals this because they changed the equality operator
        # self.assertTrue(linear.weights == linear2.weights)
        self.assertIs(linear.weights, linear2.weights)
        self.assertIs(linear.bias, linear2.bias)

        # with eager we can do this in the tests
        self.assertArrayEqual(linear() * 2, linear2())
        # in alternative to
        # self.assertTrue(np.array_equal(linear().numpy()*2, linear2().numpy())

    def test_linear_rank3(self):
        x = tx.Input([[[1], [1]], [[2], [2]]], dtype=tf.float32)

        x2 = tx.Transpose(x)
        x_flat = tx.Reshape(x, [-1, 1])
        linear1 = tx.Linear(x, n_units=2)
        linear2 = tx.Linear(x2,
                            shape=[2, 1],
                            weights=linear1.weights,
                            transpose_weights=True)

        # we cant do this because it changes the definition of the layer (n_units etc)
        try:
            linear3 = linear1.reuse_with(x2, transpose_weights=True)
            self.fail("we can't reuse with transpose weights while changing the layer definition")
        except ValueError as e:
            pass

        linear_flat = linear1.reuse_with(x_flat)
        linear_flat = tx.Reshape(linear_flat, x().get_shape().as_list()[:-1] + [2])

        self.assertTrue(np.array_equal(linear1(), linear_flat()))
        self.assertTrue(np.array_equal(tf.shape(linear2()), [1, 2, 1]))

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

        trans = tx.Transpose(t, n_units=3)
        # using this or a module is practically the same (in this case) except we
        # don't have to create the graph and then the module
        w = tx.Wrap(trans, lambda layer: tx.Lambda(layer, fn=lambda x: x * 2))
        w2 = w.reuse_with(x)

        self.assertArrayEqual(w2(), t * 2)
        self.assertArrayEqual(w(x), t * 2)

        self.assertArrayEqual(w(t), w())

        self.assertArrayEqual(w.compute(t), tf.transpose(t) * 2)
        self.assertArrayEqual(trans.compute(t), x)
        self.assertArrayEqual(w2.compute(x), w2())

    def test_variable_layer(self):
        input_layer = tx.Input([[1]], n_units=1, dtype=tf.float32)
        var_layer = tx.VariableLayer(input_layer, dtype=tf.float32)

        init_value = var_layer.variable.numpy()
        after_update = var_layer().numpy()

        self.assertArrayNotEqual(init_value, after_update)
        self.assertArrayEqual(after_update, var_layer.variable.numpy())

    def test_variable_init_from_input(self):
        input_layer = tx.Input(n_units=1, constant=False)
        layer_once = tx.VariableLayer(input_layer, update_once=True)
        layer_var = tx.VariableLayer(input_layer, update_once=False)

        var_layer_fwd2 = layer_once.reuse_with(init_from_input=False)

        data1 = np.array([[1]])
        data2 = np.array([[2]])
        data3 = np.array([[3]])

        input_layer.value = data1
        self.assertEqual(layer_once.counter.numpy(), 0)
        input_layer.value = data2
        y1 = layer_once()
        self.assertEqual(layer_once.counter.numpy(), 1)
        self.assertTrue(np.array_equal(layer_once.variable.value(), y1))
        input_layer.value = data3
        y2 = layer_once()
        self.assertEqual(layer_once.counter.numpy(), 1)
        self.assertTrue(np.array_equal(y1, y2))
        self.assertTrue(np.array_equal(y1, layer_once.variable.value()))

        # dynamic var layer
        input_layer.value = data1
        self.assertEqual(layer_var.counter.numpy(), 0)
        y1 = layer_var()
        self.assertEqual(layer_var.counter.numpy(), 1)
        self.assertTrue(np.array_equal(layer_var.variable.value(), y1))

        input_layer.value = data2
        y2 = layer_var()
        self.assertEqual(layer_var.counter.numpy(), 2)
        self.assertFalse(np.array_equal(y1, y2))

    def test_variable_layer_reuse(self):
        input_layer = tx.Input([[1]], n_units=1, dtype=tf.float32)
        input_layer2 = tx.Input([[1], [2]], n_units=1, dtype=tf.float32)
        var1 = tx.VariableLayer(shape=[2, 1])

        var2 = var1.reuse_with(input_layer)
        var3 = var1.reuse_with(input_layer2)

        v0 = var1()
        v1 = var2()
        self.assertFalse(np.array_equal(v0, v1))

        # v0 inner variable changed when we evaluate v1
        v2 = var1()
        self.assertFalse(np.array_equal(v0, v1))

        v3 = var3()
        self.assertFalse(np.array_equal(v2, v3))
        v4 = var1()
        self.assertTrue(np.array_equal(v3, v4))

        # variable batch dimension is dynamic its shape will be different
        self.assertFalse(np.array_equal(np.shape(v4), np.shape(v1)))
        self.assertTrue(np.array_equal(np.shape(v2), np.shape(v1)))

    def test_standalone_variable_layer(self):
        var_layer = tx.VariableLayer(shape=[10])
        self.assertTrue(np.array_equal(np.zeros([10]), var_layer()))

    def test_module_reuse_order(self):
        x1 = tx.Input([[2.]], n_units=1, name="x1")
        x2 = tx.Input([[2.]], n_units=1, name="x2")
        x3 = tx.Input([[1.]], n_units=1, name="x3")

        h = tx.Add(x2, x3)
        y = tx.Add(x1, h)

        m = tx.Module(inputs=[x1, x2, x3], output=y)

        x1_ = tx.Constant([[2.]], name="x1b")
        x2_ = tx.Constant([[2.]], name="x2b")
        x3_ = tx.Constant([[1.]], name="x2b")

        m2 = m.reuse_with(x1_, x2_)

        m1 = m()
        m2 = m2()

        self.assertArrayEqual(m1, m2)

    def test_module_rnn(self):
        # test wrapping module around RNN because it has input dependencies that might not be given in the constructor
        x1 = tx.Input(tf.ones([1, 2, 3]), n_units=3, name="x1")
        x2 = tx.Input(tf.ones([1, 2, 3]), n_units=3, name="x2")
        rnn1 = tx.RNN(x1, cell_proto=tx.LSTMCell.proto(n_units=4), n_units=4, stateful=False)
        rnn2 = tx.RNN(x1, cell_proto=tx.LSTMCell.proto(n_units=4), n_units=4, stateful=False)

        out = tx.Concat(rnn1, rnn2)

        # we need to add previous state as a dependency to a module
        m = tx.Module(inputs=x1, output=out, dependencies=rnn1.previous_state + rnn2.previous_state)

        m2 = m.reuse_with(x2)
        var_layers = set()
        for node in m2.graph.dependency_iter():
            if isinstance(node, tx.VariableLayer):
                var_layers.add(node)
        self.assertSetEqual(var_layers, set(rnn1.previous_state + rnn2.previous_state))

        self.assertArrayEqual(m(), m2())

    def test_module_with_attention(self):
        # logger = logging.getLogger('tensorx')
        # logger.setLevel(logging.DEBUG)
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG)
        # logger.addHandler(ch)

        x1 = tx.Input(tf.ones([1, 2, 3]), n_units=3, name="x1")
        x2 = tx.Input(tf.ones([1, 2, 3]), n_units=3, name="x2")
        rnn1 = tx.RNN(x1, cell_proto=tx.LSTMCell.proto(n_units=4), n_units=4, stateful=False)
        att = tx.MHAttention(rnn1, rnn1, rnn1, n_units=3)
        m = tx.Module(inputs=x1, output=att, dependencies=rnn1.previous_state)
        #m.graph.draw("test.pdf")
        g = tx.Graph.build(inputs=x1, outputs=m, missing_inputs=True)
        # list(map(print, g.in_nodes))
        fn = g.as_function(ord_inputs=x1, ord_outputs=m)
        # this returns a tuple
        out1 = g.compute(tf.ones([1, 2, 3]))
        # this returns the function result
        out2 = fn(tf.ones([1, 2, 3]))

        self.assertArrayEqual(out1[0], out2)
        # list(map(print, m.trainable_variables))

    def test_module(self):
        l1 = tx.Input([[1]], n_units=1, dtype=tf.float32)
        l2 = tx.Input([[1]], n_units=1, dtype=tf.float32)
        l3 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l1, l2)
        l4 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l1, l2)
        l5 = tx.Linear(l4, 1)
        in1 = tx.Input([[1]], n_units=1, dtype=tf.float32)
        l7 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l3, in1)
        l8 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l7, l5)

        in2 = tx.Input([[1]], n_units=1, dtype=tf.float32, constant=False)
        in3 = tx.Input([[1]], n_units=1, dtype=tf.float32)

        m = tx.Module([l1, l2, in1], l8)
        with tf.name_scope("module_reuse"):
            m2 = m.reuse_with(in2, in3, in1)

        self.assertTrue(np.array_equal(m(), m2()))
        in2.value = [[3]]
        self.assertFalse(np.array_equal(m(), m2()))

    def test_rnn_cell(self):
        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = tx.Input(tf.ones([batch_size, n_inputs]))

        rnn1 = tx.RNNCell(inputs, n_hidden)

        state = rnn1.state
        state = state[0]()

        rnn_2 = rnn1.reuse_with(inputs, state)
        rnn_3 = rnn1.reuse_with(inputs)

        try:
            tx.RNNCell(inputs, n_hidden, share_state_with=inputs)
            self.fail("should have thrown exception because inputs cannot share state with RNNCell")
        except TypeError:
            pass

        res1 = rnn1()
        res2 = rnn_2()
        res3 = rnn_3()

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertTrue(np.array_equal(res1, res3))
        self.assertFalse(np.array_equal(res1, res2))

    def test_rnn_cell_graph(self):
        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        data1 = tf.ones([batch_size, n_inputs])
        inputs = tx.Input(data1)
        rnn1 = tx.RNNCell(inputs, n_hidden)
        # if I use missing_inputs=True, it will just add the input to the graph without
        try:
            g = tx.Graph.build(inputs=inputs, outputs=rnn1, missing_inputs=False)
            self.fail("should have raised Value Error for missing inputs")
        except ValueError:
            pass
        try:
            g = tx.Graph.build(inputs=inputs, outputs=rnn1, missing_inputs=True)
            f = g.as_function(ord_inputs=inputs)
            f(data1)
        except Exception as e:
            self.fail(str(e))

    def test_rnn_cell_drop(self):

        n_hidden = 4
        inputs1 = tx.Input(np.ones([2, 100]), dtype=tf.float32)
        inputs2 = tx.Input(np.ones([2, 100]), dtype=tf.float32)

        with tf.name_scope("wtf"):
            rnn1 = tx.RNNCell(inputs1, n_hidden,
                              x_dropout=0.5,
                              r_dropout=0.5,
                              u_dropconnect=0.5,
                              w_dropconnect=0.5,
                              regularized=True
                              )
        rnn2 = rnn1.reuse_with(inputs2, rnn1)
        rnn3 = rnn1.reuse_with(inputs2, rnn1)
        rnn4 = rnn1.reuse_with(inputs2, None, regularized=False)
        rnn5 = rnn4.reuse_with(inputs2, None, regularized=True)

        r1, r2, r3, r4, r5 = rnn1(), rnn2(), rnn3(), rnn4(), rnn5()
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

        inputs = tx.Input(init_value=data, n_units=n_inputs, dtype=tf.float32)

        rnn_1 = tx.GRUCell(inputs, n_hidden)

        rnn_2 = rnn_1.reuse_with(inputs, rnn_1)

        # if we don't wipe the memory it reuses it
        rnn_3 = rnn_1.reuse_with(inputs, tx.GRUCell.zero_state(rnn_1.n_units))

        res1 = rnn_1()
        res2 = rnn_2()
        res3 = rnn_3()

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertArrayEqual(res1, res3)
        self.assertArrayNotEqual(res1, res2)

    def test_module_gate(self):
        x1 = tx.Input([[1, 1, 1, 1]], n_units=4, dtype=tf.float32)
        x2 = tx.Input([[1, 1]], n_units=2, dtype=tf.float32)
        x1 = tx.Add(x1, x1)

        gate = tx.Gate(input_layer=x1, gate_input=x2, gate_fn=tf.sigmoid)
        gate_module = tx.Module([x1, x2], gate)

        x3 = tx.Input([[1, 1, 1, 1]], n_units=4, dtype=tf.float32)
        x4 = tx.Input([[1, 1]], n_units=2, dtype=tf.float32)

        m2 = gate_module.reuse_with(x3, x4)

        result1 = gate_module()
        result2 = m2()
        result3 = gate_module.compute(x3, x4)

        self.assertArrayEqual(result1, result2 * 2)
        self.assertArrayEqual(result2, result3)

    def test_lstm_cell(self):
        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = tx.Input(np.ones([batch_size, n_inputs], np.float32), n_units=n_inputs, constant=True)
        rnn1 = tx.LSTMCell(inputs, n_hidden, gate_activation=tf.sigmoid)
        previous_state = (None, rnn1.state[-1]())
        rnn2 = rnn1.reuse_with(inputs, *previous_state)

        # if we don't wipe the memory it reuses it
        previous_state = (None, tx.LSTMCell.zero_state(rnn1.n_units))
        rnn3 = rnn1.reuse_with(inputs, *previous_state)
        rnn4 = rnn1.reuse_with(inputs)

        res1 = rnn1()
        res2 = rnn2()
        res3 = rnn3()
        res4 = rnn4()

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertArrayEqual(res1, res3)
        self.assertArrayNotEqual(res1, res2)
        self.assertArrayEqual(res1, res4)

    def test_lstm_cell_regularization(self):

        n_inputs = 8
        n_hidden = 2
        batch_size = 2

        inputs = tx.Input(n_units=n_inputs, constant=False)

        rnn1 = tx.LSTMCell(inputs, n_hidden,
                           u_dropconnect=0.1,
                           w_dropconnect=0.1,
                           name="lstm1")

        rnn2 = rnn1.reuse_with(inputs,
                               *rnn1.state,
                               regularized=True,
                               name="lstm2"
                               )

        rnn3 = rnn2.reuse_with(inputs,
                               *rnn1.state,
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

    def test_lstm_cell_state(self):
        n_inputs = 8
        n_hidden = 2
        batch = 3

        x = tf.ones([batch, n_inputs], dtype=tf.float32)

        cell = tx.LSTMCell(x, n_hidden,
                           u_dropconnect=0.1,
                           w_dropconnect=0.1,
                           name="cell")

        state = cell.previous_state
        state = [s() for s in state]

        state = tx.Graph.build(inputs=None,
                               outputs=cell.state)

        x = tf.random.uniform([batch, n_inputs])
        s = state.compute(x)
        s_ = state.compute(x, *s)

    def test_rnn_layer(self):
        n_features = 5
        embed_size = 4
        hdim = 3
        seq_size = 3
        batch_size = 2

        inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = lookup.permute_batch_time()

        ones_state = tf.ones([batch_size, hdim])
        zero_state = (tf.zeros([batch_size, hdim]))

        rnn_proto = tx.RNNCell.proto(n_units=hdim)

        rnn1 = tx.RNN(seq, cell_proto=rnn_proto, previous_state=ones_state, return_state=True)
        rnn2 = rnn1.reuse_with(seq)

        # TODO problem with RNN layer is that it uses modules that require
        #  all the params to output the right answer
        #  we need to supply the default values for the rest or all the inputs
        out1, last1 = rnn1()
        out2, last2 = rnn2()

        self.assertArrayEqual(out1, out2)
        self.assertArrayEqual(last1, last2)

        rnn3 = rnn1.reuse_with(seq, zero_state)
        rnn4 = rnn3.reuse_with(seq)
        rnn5 = rnn4.reuse_with(seq, ones_state)

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

    def test_biRNN(self):
        # bidirectional RNN
        n_features = 5
        embed_size = 4
        hdim = 3
        seq_size = 6
        batch_size = 2

        inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = lookup.permute_batch_time()

        rnn_proto = tx.RNNCell.proto(n_units=hdim)
        rnn0 = tx.RNN(seq, cell_proto=rnn_proto, stateful=False, return_state=True)

        # because a stateful rnn0 has a variable layer as input as well
        rnn_m0 = tx.Module(inputs=rnn0.input_layers, output=rnn0)

        rnn1 = rnn0.reuse_with(seq, reverse=True, stateful=False, return_state=True)
        # TODO this solves rnn output multiple tensors

        r01 = rnn_m0.compute(seq(), rnn0.previous_state[0]())
        rnn0.reset()
        r02 = rnn0()

        self.assertArrayEqual(r01[0], r02[0])

        rnn0_ = rnn0[0]
        rnn1_ = rnn1[0]
        rnn0 = tx.Wrap(rnn0, wrap_fn=lambda y: y[0], n_units=rnn0.n_units)
        rnn1 = tx.Wrap(rnn1, wrap_fn=lambda y: y[0], n_units=rnn1.n_units)

        self.assertArrayEqual(tf.shape(rnn0()), tf.shape(rnn1()))
        self.assertArrayEqual(tf.shape(rnn0()), tf.shape(rnn0_()))
        self.assertArrayEqual(tf.shape(rnn1()), tf.shape(rnn1_()))

        # print(tf.shape(rnn0()))
        r0 = rnn0()
        r1 = rnn1()
        c = tx.Concat(rnn0, rnn1, axis=-1)
        # print(tf.shape(c()))

        # concat = tx.Con
        # print(tf.shape())
        # self.assertArrayEqual()

    def test_stateful_rnn_layer(self):
        n_features = 5
        embed_size = 4
        hdim = 3
        seq_size = 3
        batch_size = 2

        inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = lookup.permute_batch_time()

        rnn_proto = tx.RNNCell.proto(n_units=hdim)

        rnn1 = tx.RNN(seq, cell_proto=rnn_proto, stateful=True, return_state=True)
        lstm1 = tx.RNN(seq, cell_proto=tx.LSTMCell.proto(n_units=hdim), stateful=True, return_state=True)

        zero_state0 = [layer() for layer in rnn1.previous_state]

        self.assertEqual(len(zero_state0), 1)
        self.assertArrayEqual(zero_state0[0], np.zeros([1, hdim]))

        import logging
        logging.getLogger("tensorx").setLevel(logging.DEBUG)

        out1, state1 = rnn1()

        layers = tx.Graph.build(inputs=None, outputs=lstm1)
        out2, state2 = lstm1()

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

        inputs = tx.Input(np.array([[2, 0], [1, 2]]), 2, dtype=tf.int64)
        tensor_input = tx.Input(tf.constant([2]), 1, dtype=tf.int64)

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

        inputs = tx.Input(dtype=tf.int32, constant=False)

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

    def test_dynamic_concat(self):
        seq1 = [[1, 2], [3, 4]]
        seq2 = [[1, 2, 3], [4, 5, 6]]

        n = 10
        m = 4

        inputs = tx.Input(seq2, shape=[None, None], dtype=tf.int32, constant=False)
        inputs2 = tx.Input(seq2, dtype=tf.int32, constant=True)

        lookup = tx.Lookup(inputs, seq_size=None, embedding_shape=[n, m])
        lookup2 = tx.Lookup(inputs2, seq_size=3, embedding_shape=[n, m])
        concat1 = lookup.as_concat()
        concat2 = lookup2.as_concat()

        self.assertFalse(concat1.n_units)
        self.assertTrue(concat2.n_units)

        concat3 = tx.SeqConcat(lookup, time_major=False)
        concat4 = tx.SeqConcat(lookup, seq_size=3, time_major=False)

        c1, c2 = concat1(), concat3()
        self.assertArrayEqual(c1, c2)
        self.assertFalse(concat3.n_units)
        self.assertEqual(concat4.n_units, 3 * lookup.n_units)

        inputs.value = seq1
        l1 = lookup()
        inputs.value = seq2
        l2 = lookup()

        self.assertEqual(np.shape(l1)[-1], m)
        self.assertEqual(np.shape(l2)[-1], m)

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

        inputs = tx.Input(n_units=k, sparse=True, dtype=tf.int32, constant=False)
        seq_len = tx.Input(init_value=2, shape=[], constant=False)
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

        # self.assertEqual(np.shape(c1)[-1], m * 2)
        # self.assertEqual(np.shape(c2)[-1], m * 3)

    def test_lookup_sequence_sparse(self):
        input_dim = 10
        embed_dim = 3
        seq_size = 2
        batch_size = 3

        sparse_input = tf.SparseTensor([[0, 2], [1, 0], [2, 1]], [1, 1, 1], [3, input_dim])
        sparse_input_1d = tf.SparseTensor([[2], [0], [1]], [1, 1, 1], [input_dim])
        tensor_input = tx.Constant(sparse_input, input_dim)
        tensor_input_1d = tx.Constant(sparse_input_1d, input_dim)

        lookup = tx.Lookup(tensor_input, seq_size,
                           embedding_shape=[input_dim, embed_dim],
                           batch_size=batch_size,
                           batch_padding=False)

        lookup_padding = tx.Lookup(tensor_input, seq_size,
                                   embedding_shape=[input_dim, embed_dim],
                                   batch_size=batch_size,
                                   batch_padding=True)

        lookup_1d = tx.Lookup(tensor_input_1d, seq_size,
                              embedding_shape=[input_dim, embed_dim],
                              batch_size=batch_size,
                              batch_padding=True)

        result = lookup()
        result_padding = lookup_padding()
        result_1d = lookup_1d()

        self.assertEqual(np.shape(result), (2, seq_size, embed_dim))
        self.assertEqual(np.shape(result_padding), (batch_size, seq_size, embed_dim))
        self.assertEqual(np.shape(result_1d), (batch_size, seq_size, embed_dim))

    def test_lookup_sparse_padding(self):
        input_dim = 6
        embed_dim = 3
        seq_size = 1

        sparse_input = tf.SparseTensor([[0, 1], [0, 3], [1, 0]], [1, 1, 1], [2, input_dim])
        sparse_input = tx.Constant(sparse_input, input_dim)

        lookup = tx.Lookup(sparse_input,
                           seq_size=seq_size,
                           embedding_shape=[input_dim, embed_dim],
                           batch_size=None,
                           batch_padding=False)

        result = lookup()

    def test_lookup_sequence_bias(self):
        vocab_size = 4
        n_features = 3
        seq_size = 2

        inputs = tx.Input(n_units=seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2], [0, 2]])
        lookup = tx.Lookup(input_layer=inputs,
                           seq_size=seq_size,
                           embedding_shape=[vocab_size, n_features],
                           add_bias=True)

        inputs.value = input_data
        v1 = lookup()
        self.assertEqual(np.shape(v1), (np.shape(input_data)[0], seq_size, n_features))

    def test_lookup_sequence_transform(self):
        vocab_size = 4
        embed_dim = 2
        seq_size = 2

        inputs = tx.Input(n_units=seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2], [0, 2]])
        lookup = tx.Lookup(inputs,
                           seq_size=seq_size,
                           embedding_shape=[vocab_size, embed_dim],
                           add_bias=True)
        concat_lookup = lookup.as_concat()
        seq_lookup = lookup.permute_batch_time()

        self.assertTrue(hasattr(lookup, "seq_size"))

        inputs.value = input_data

        v1 = lookup()
        v2 = concat_lookup()
        v3 = seq_lookup()

        self.assertEqual(np.shape(v1), (np.shape(input_data)[0], seq_size, embed_dim))
        self.assertEqual(np.shape(v2), (np.shape(input_data)[0], seq_size * embed_dim))

        self.assertEqual(np.shape(v3), (seq_size, np.shape(input_data)[0], embed_dim))
        self.assertTrue(np.array_equal(v1[:, 0], v3[0]))

    def test_reuse_dropout(self):
        x1 = tx.Constant(np.ones(shape=[2, 4]), dtype=tf.float32)
        x2 = tx.Activation(x1)

        drop1 = tx.Dropout(x2, probability=0.5, locked=True)

        self.assertEqual(len(drop1.input_layers), 2)
        self.assertIs(drop1.input_layers[0], x2)
        self.assertIs(drop1.input_layers[-1], drop1.layer_state.mask)

        # shared state overrides mask?
        _, mask = tx.dropout(x2, return_mask=True)
        drop2 = drop1.reuse_with(x2, mask)

        self.assertEqual(len(drop2.input_layers), 2)
        self.assertIs(drop2.input_layers[0], x2)
        self.assertIs(drop2.input_layers[-1], drop2.layer_state.mask)

        self.assertArrayNotEqual(drop1(), drop2())

        graph = tx.Graph.build(inputs=None, outputs=[drop1, drop2])

        out1, out2 = graph()
        self.assertArrayEqual(out1, out2)

        drop1 = tx.Dropout(x2, probability=0.5)
        drop2 = drop1.reuse_with(x1)

        graph.eval(drop1, drop2)
        #
        # d1, d2, d3 = drop1(),drop2(),drop3()
        #
        # self.assertIs(drop1.mask, drop3.mask)
        # self.assertIsNot(drop1.mask, drop2.mask)
        #
        # self.assertArrayEqual(d1, d3)
        # self.assertArrayNotEqual(d1, d2)

    def test_drop_lookup(self):
        seq_size = 4
        vocab_size = 10
        embed_dim = 4
        input_data = np.array([[2, 0, 2, 0], [1, 2, 2, 1], [0, 2, 0, 2]])
        inputs = tx.Input(init_value=input_data, n_units=seq_size, dtype=tf.int32)
        lookup = tx.Lookup(inputs,
                           seq_size=seq_size,
                           embedding_shape=[vocab_size, embed_dim],
                           add_bias=True)

        drop = tx.DropLookup(lookup, probability=0.5)
        # TODO this works but need to finish the tests for it

    def test_residual(self):
        x1 = tx.Input([[1., 1., 1., 1.]], 4)
        x2 = tx.Input([[1., 1., 1., 1.]], 4)

        h1 = tx.FC(x1, 4, activation=tf.sigmoid)
        h2 = tx.FC(x1, 2, activation=tf.sigmoid)
        h3 = tx.FC(x2, 2, activation=tf.sigmoid)

        residual = tx.Residual(x1, h1)
        residual2 = tx.Residual(x1, h2)

        try:
            residual3 = tx.Residual(x1, h3)
        except ValueError:
            pass

        self.assertArrayEqual(tf.shape(h1()), tf.shape(residual()))
        self.assertFalse(hasattr(residual, "projection"))
        self.assertTrue(hasattr(residual2, "projection"))
        self.assertEqual(len(residual.trainable_variables), 0)
        self.assertEqual(len(residual2.trainable_variables), 1)

    def test_fully_connected(self):
        x1 = tx.Input(init_value=[[1., 1., 1., 1.]], n_units=4, dtype=tf.float32, constant=True)
        x2 = tx.Input(init_value=np.random.uniform(size=[2, 4]), dtype=tf.float32, n_units=4, constant=True)

        y1 = tx.FC(x1, 4, add_bias=True, activation=tf.sigmoid)

        y2 = tx.Linear(x1, 4, add_bias=True, weights=y1.linear.weights, bias=y1.linear.bias)
        a2 = tx.Activation(y2, fn=tf.sigmoid)

        w = y2.weights
        b = y2.bias

        self.assertIs(y1.linear.weights, w)
        self.assertIs(y1.linear.bias, b)

        x = x1()
        y = tf.matmul(x, w) + b
        a = tf.sigmoid(y)

        self.assertArrayEqual(y2(), y)
        self.assertArrayEqual(y1(), a)
        self.assertArrayEqual(y1(), a2())
        self.assertArrayEqual(a2(), a)

        y1 = y1.reuse_with(x2)
        y2 = y2.reuse_with(x2)
        a2 = a2.reuse_with(y2)

        self.assertIs(y2.weights, w)
        self.assertIs(y2.bias, b)

        self.assertIs(y1.linear.weights, w)
        self.assertIs(y1.linear.bias, b)

        # print(y1())

        # self.assertArrayEqual(y1(), a2())

    def test_conv1d(self):
        num_filters = 2
        input_dim = 4
        seq_size = 3
        batch_size = 2
        filter_size = 2

        filter_shape = [filter_size, input_dim, num_filters]

        x = tf.ones([batch_size, seq_size, input_dim])
        x_layer = tx.Constant(x, input_dim)

        filters = tf.ones(filter_shape)
        conv_layer = tx.Conv1D(x_layer, num_filters, filter_size, shared_filters=filters)
        conv = tf.nn.conv1d(input=x,
                            filters=filters,
                            stride=1,
                            padding="SAME",
                            data_format="NWC")

        output = conv_layer()
        self.assertArrayEqual(conv, output)
        self.assertArrayEqual(tf.shape(conv_layer.filters), (filter_size, input_dim, num_filters))
        self.assertArrayEqual(tf.shape(output), (batch_size, seq_size, num_filters))

    def test_map_seq(self):
        n_features = 5
        embed_size = 4
        hdim = 3
        seq_size = 3
        batch_size = 2

        inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
        seq = lookup.permute_batch_time()

        n_units = 2
        linear_fn = tx.Linear.proto(n_units=n_units)
        self.assertArrayEqual(tf.shape(seq()), [seq_size, batch_size, embed_size])

        seq_map = tx.SeqMap(seq, n_units=2, layer_proto=linear_fn)
        self.assertArrayEqual(tf.shape(seq_map), [seq_size, batch_size, n_units])

    def test_multihead_attention(self):
        n_features = 3
        embed_size = 128
        seq_size = 3
        batch_size = 2
        n_heads = 8

        inputs = tx.Constant(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        emb = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])

        attention = tx.MHAttention(query=emb,
                                   key=emb,
                                   value=emb,
                                   n_units=embed_size,
                                   n_heads=n_heads,
                                   causality=False,
                                   attention_dropout=0.1,
                                   regularized=False)

        self.assertEqual(len(attention.input_layers), 3)

        # 3 "kernels" + bias
        self.assertEqual(len(attention.variables), 3)

        attention_reg = attention.reuse_with(emb, emb, emb, regularized=True)
        attention_2 = attention.reuse_with(emb, emb, emb, regularized=False)
        attention_causal = attention.reuse_with(emb, emb, emb, causality=True)

        res = attention_causal()

        result = attention()
        result_reg = attention_reg()
        result2 = attention_2()

        self.assertArrayEqual(tf.shape(result), tf.shape(result_reg))
        self.assertArrayEqual(result, result2)

        vars1 = map(lambda v: v.ref(), attention.variables)
        vars2 = map(lambda v: v.ref(), attention_2.variables)

        self.assertSetEqual(set(vars1), set(vars2))


if __name__ == '__main__':
    unittest.main()
