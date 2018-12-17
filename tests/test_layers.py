from tensorx import test_utils
import tensorflow as tf
import numpy as np
from tensorx.layers import *
from tensorx.init import *
from tensorx.loss import *
from tensorx.train import LayerGraph
from tensorx.activation import *
from tensorx.transform import sparse_tensor_value_one_hot
from tensorflow import sparse
import math
import os
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestLayers(test_utils.TestCase):
    def test_var_scope(self):
        linear = Linear([[1]], 2)
        linear2 = linear.reuse_with([[2]])
        with self.cached_session(use_gpu=True):
            self.eval(tf.global_variables_initializer())
            self.assertArrayEqual(linear.variables, linear2.variables)
            result1 = linear.eval()
            result2 = linear2.eval()

            self.assertArrayEqual(result2, result1 * 2)

    def test_fn_layer(self):
        in1 = Input(1)
        in2 = Input(1)
        in3 = TensorLayer(tf.constant([[2.]]))

        txfn = FnLayer(in1, in2, apply_fn=binary_cross_entropy)
        txfn2 = txfn.reuse_with(in1, in3)

        graph = LayerGraph(outputs=[txfn, txfn2])

        with self.cached_session(use_gpu=True):
            data1 = [[0.23]]
            data2 = [[2.]]
            result = graph.eval({in1: data1, in2: data1}, target_outputs=txfn)
            result2 = self.eval(binary_cross_entropy(data1, data1))
            result3 = graph.eval({in1: data1, in2: data2}, target_outputs=txfn)

            self.assertArrayEqual(result, result2)
            self.assertArrayNotEqual(result, result3)

            result4 = graph.eval({in1: data1}, target_outputs=txfn2)
            self.assertArrayEqual(result4, result3)

    def test_variable_layer(self):
        input_layer = TensorLayer([[1]], n_units=1)
        var_layer = VariableLayer(input_layer)

        init = tf.global_variables_initializer()
        with self.cached_session(use_gpu=True):
            self.eval(init)
            var_layer_init_value = self.eval(var_layer.variable)
            var_value = self.eval(var_layer.tensor)
            self.assertArrayNotEqual(var_layer_init_value, var_value)
            self.assertArrayEqual(var_layer.tensor, var_layer.variable)

    def test_variable_layer_reuse(self):
        input_layer = TensorLayer([[1]], n_units=1, dtype=tf.float32)
        input_layer2 = TensorLayer([[1], [2]], n_units=1, dtype=tf.float32)
        var1 = VariableLayer(var_shape=[1, 1])

        var2 = var1.reuse_with(input_layer)

        var3 = var1.reuse_with(input_layer2)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            r1, r2 = self.eval([var1.tensor, var2.tensor])
            r3 = self.eval(var1.tensor)
            r4 = self.eval(var3.tensor)
            self.assertArrayEqual(r2, r3)
            self.assertNotEqual(r1, r2)
            # since the variable batch dimension is dynamic its shape will be different
            self.assertArrayNotEqual(np.shape(r3), np.shape(r4))

    def test_standalone_variable_layer(self):
        var_layer = VariableLayer(var_shape=[10])
        with self.cached_session(use_gpu=True):
            self.eval(tf.global_variables_initializer())
            value = var_layer.eval()
            self.assertArrayEqual(np.zeros([10]), value)

    def test_variable_layer_broadcasting(self):
        layer1 = TensorLayer([[1], [1]], dtype=tf.float32)

        var1 = VariableLayer(n_units=layer1.n_units)
        var2 = var1.reuse_with(layer1)

        linear1 = Linear(var1, n_units=2)
        linear2 = Linear(layer1, n_units=2)

        add1 = Add(var1, layer1)
        add2 = Add(linear1, linear2)

        init = tf.global_variables_initializer()
        with self.cached_session(use_gpu=True):
            self.eval(init)
            in1 = self.eval(layer1.tensor)
            r1 = self.eval(var1.tensor)
            l1 = self.eval(linear1.tensor)
            r5 = self.eval(add2.tensor)
            r2 = self.eval(add1.tensor)
            r3 = self.eval(var2.tensor)
            r4 = self.eval(add1.tensor)
            l2 = self.eval(linear1.tensor)
            r6 = self.eval(add2.tensor)

            self.assertEqual(np.shape(l1)[0], 1)
            self.assertEqual(np.shape(r1)[0], 1)
            self.assertEqual(np.shape(r5)[0], np.shape(in1)[0])
            self.assertEqual(np.shape(r2)[0], 2)
            self.assertEqual(np.shape(r3)[0], 2)

            self.assertEqual(np.shape(r4)[0], np.shape(r2)[0])
            self.assertNotEqual(np.shape(l1)[0], np.shape(l2)[0])
            self.assertEqual(np.shape(r5), np.shape(r6))

    def test_variable_layer_dynamic_batch(self):
        input_layer = Input(n_units=1)
        var_layer = VariableLayer(input_layer)

        init = tf.global_variables_initializer()
        with self.cached_session(use_gpu=True):
            self.eval(init)
            var_layer_init_value = self.eval(var_layer.variable)
            var_value = self.eval(var_layer.tensor, {input_layer.placeholder: [[1]]})
            self.assertArrayNotEqual(var_layer_init_value, var_value)
            self.assertArrayEqual(var_value, var_layer.variable)

    def test_variable_layer_multiply_dynamic(self):
        n_units = 3
        input_layer = Input(n_units=n_units)

        var_layer = VariableLayer(input_layer, resource=True)
        # print(var_layer.variable.get_shape())

        input_layer_2 = Input(n_units=n_units)

        # print(tensor_util.constant_value_as_shape(tf.shape(var_layer.variable)))
        # ok this is probably calling tf.shape so I just need to figure what value this uses

        mul = tf.matmul(input_layer_2, var_layer.variable, transpose_b=True)
        # print(mul.get_shape())

        res = tf.nn.softmax(mul)

        init = tf.global_variables_initializer()
        with self.cached_session(use_gpu=True):
            self.eval(init)
            init_value = self.eval(var_layer.variable)
            self.assertArrayEqual(init_value, np.zeros(shape=[0, n_units]))

            val1 = np.ones([10, n_units]) * 2
            self.eval(var_layer.tensor, {input_layer.placeholder: val1})

            assign1 = self.eval(var_layer.variable)
            self.assertArrayNotEqual(assign1, init_value)

            val2 = np.ones([2, n_units]) * 2
            result = self.eval(tf.shape(mul), {input_layer_2.placeholder: val2})
            self.assertArrayEqual(result, [2, 10])

            assign2 = self.eval(var_layer.variable)
            # value didn't change because we used the variable value not the tensor
            self.assertArrayEqual(assign2, assign1)

            val3 = np.ones([5, n_units]) * 3
            assign3 = self.eval(var_layer.tensor, {input_layer.placeholder: val3})
            self.assertArrayNotEqual(assign3, assign2)
            self.assertArrayEqual(assign3, var_layer.variable)

    def test_convert_layer(self):
        constant = [[2., 3.]]
        layer = TensorLayer(tensor=constant, n_units=2)
        tensor = tf.convert_to_tensor_or_sparse_tensor(layer)
        with self.cached_session():
            result = self.eval(tensor)
            self.assertArrayEqual(result, constant)

    def test_compose(self):
        in1 = Input(1)
        in2 = TensorLayer([[1.]], 1)

        l1 = Linear(in1, 4)
        l2 = Activation(l1, relu)

        comp = Compose(l1, l2)
        comp2 = comp.reuse_with(in2)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            res1 = self.eval(l2.tensor, {in1.placeholder: [[1.]]})
            res2 = self.eval(comp.tensor, {in1.placeholder: [[1.]]})
            res3 = self.eval(comp2.tensor)

            self.assertArrayEqual(res1, res2)
            self.assertTrue(np.array_equal(res1, res3))

    def test_fn_compose(self):
        in1 = Input(1)
        in2 = TensorLayer([[1.]], 1)

        l1 = Linear(in1, 4, add_bias=True)
        l2 = Activation(l1, relu)

        comp = Compose(l1, l2)
        comp2 = comp.reuse_with(in2)

        fn1 = FC(in1, 4, fn=relu, share_vars_with=l1)
        fn2 = fn1.reuse_with(in2, name="fn2")
        fn3 = fn2.reuse_with(in2, name="fn3")
        fn4 = FC(in2, 4, fn=relu, shared_weights=l1.weights, bias_init=tf.initializers.constant(value=34))

        init = tf.global_variables_initializer()

        feed = {in1.placeholder: [[1.]]}

        with self.cached_session(use_gpu=True):
            self.eval(init)

            res1 = self.eval(l2.tensor, feed)
            res2 = self.eval(comp.tensor, feed)
            res3 = self.eval(comp2.tensor)

            self.assertArrayEqual(res1, res2)
            self.assertArrayEqual(res1, res3)

            res_fn1 = self.eval(fn1.tensor, feed)
            res_fn2 = self.eval(fn2.tensor)
            res_fn3 = self.eval(fn3.tensor)
            res_fn4 = self.eval(fn4.tensor)

            self.assertArrayEqual(res_fn1, res_fn2)
            self.assertArrayEqual(res_fn2, res_fn3)
            self.assertArrayNotEqual(res_fn4, res_fn2)

    def test_compose_merge(self):
        in1 = Input(1)
        in2 = TensorLayer([[1.]], 1)
        in3 = Input(1)

        a1 = Add(in1, in2)
        l1 = Linear(a1, 4)

        comp = Compose(a1, l1)
        comp2 = comp.reuse_with(in1, in3)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)

            res1 = self.eval(comp.tensor, {in1.placeholder: [[1.]]})
            res2 = self.eval(comp2.tensor, {in1.placeholder: [[1.]], in3.placeholder: [[1.]]})
            self.assertArrayEqual(res1, res2)

    def test_highway(self):
        x = TensorLayer([[1., 1., 1., 1.]], 4)
        x2 = TensorLayer([[1., 1., 1., 1.]], 4)

        h = FC(x, 4, fn=sigmoid)
        highway = Highway(x, h)

        with self.assertRaises(ValueError):
            Highway(x2, h)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertArrayEqual(x.shape, highway.shape)

    def test_residual(self):
        x = TensorLayer([[1., 1., 1., 1.]], 4)
        x2 = TensorLayer([[1., 1., 1., 1.]], 4)

        h = FC(x, 4, fn=sigmoid)
        h2 = FC(x, 2, fn=sigmoid)

        residual = Residual(x, h)
        residual_2 = Residual(x, h2)

        with self.assertRaises(ValueError):
            Residual(x2, h)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertArrayEqual(h.shape, residual.shape)
            self.assertTrue(residual.projection == residual.input_layers[0])
            self.assertIsInstance(residual.projection, TensorLayer)
            self.assertEqual(len(residual.variables), 0)

            self.assertTrue(residual_2.projection != residual.input_layers[0])
            self.assertIsInstance(residual_2.projection, Linear)
            self.assertEqual(len(residual_2.variables), 1)

    def test_conv1d(self):
        num_filters = 2
        input_dim = 4
        seq_size = 3
        batch_size = 2
        filter_size = 2

        filter_shape = [filter_size, input_dim, num_filters]

        x = tf.ones([batch_size, seq_size, input_dim])
        x_layer = TensorLayer(x, input_dim)

        filters = tf.ones(filter_shape)
        conv_layer = Conv1D(x_layer, num_filters, filter_size, shared_filters=filters)
        conv = tf.nn.conv1d(value=x,
                            filters=filters,
                            stride=1,
                            padding="SAME",
                            use_cudnn_on_gpu=True,
                            data_format="NWC")

        init = tf.global_variables_initializer()

        self.assertSequenceEqual(conv_layer.filter_shape, (filter_size, input_dim, num_filters))
        self.assertSequenceEqual(conv_layer.output_shape, (batch_size, seq_size, num_filters))

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertArrayEqual(conv, conv_layer.tensor)

    def test_causal_conv(self):
        num_filters = 1
        input_dim = 1
        seq_size = 6
        batch_size = 2
        filter_size = 3
        dilation_rate = 2

        filter_shape = [filter_size, input_dim, num_filters]

        x = tf.ones([batch_size, seq_size, input_dim])

        x_layer = TensorLayer(x, input_dim)

        conv_filter = tf.ones(filter_shape)
        conv_layer = CausalConv(x_layer, num_filters, filter_size,
                                shared_filters=conv_filter,
                                dilation_rate=dilation_rate)

        left_pad = dilation_rate * (filter_size - 1)
        padding = [[0, 0], [left_pad, 0], [0, 0]]
        x = tf.pad(x, padding)

        conv = tf.nn.convolution(input=x,
                                 filter=conv_filter,
                                 dilation_rate=(dilation_rate,),
                                 strides=(1,),
                                 padding="VALID",
                                 data_format="NWC")

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertSequenceEqual(conv_layer.filter_shape, (filter_size, input_dim, num_filters))
            self.assertSequenceEqual(conv_layer.output_shape, (batch_size, seq_size, num_filters))
            self.assertArrayEqual(conv, conv_layer.tensor)

    def test_conv2d(self):
        # simple dummy data with 10 examples of mnist digit and class data
        # digits are 28x28 data
        local_path = (os.path.dirname(__file__))

        x = np.load(local_path + "/data/mnist_10x.npy")
        y = np.load(local_path + "/data/mnist_10y.npy")

        # we only have one channel so we need to reshape the data
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        x_layer = TensorLayer(x, 1)
        # f = Flatten(x_layer)

        conv = Conv2D(layer=x_layer,
                      n_units=2,
                      filter_size=5,
                      stride=(1, 1),
                      dilation_rate=(1, 1),
                      same_padding=True,
                      bias=True)

        with self.cached_session(use_gpu=True):
            self.eval(tf.global_variables_initializer())
            self.assertArrayEqual(tf.shape(x), (10, 28, 28, 1))
            self.eval(conv.tensor)

    def test_qrnn(self):
        num_filters = 2
        input_dim = 1000
        seq_size = 2
        batch_size = 2
        filter_size = 2
        dilation_rate = 1

        x = tf.ones([batch_size, seq_size, input_dim])
        x_layer = TensorLayer(x, input_dim)
        qrnn = QRNN(layer=x_layer,
                    n_units=num_filters,
                    filter_size=filter_size,
                    dilation_rate=dilation_rate,
                    input_gate=True)

        qrnn2 = qrnn.reuse_with(x_layer)
        qrnn_zoneout = qrnn.reuse_with(x_layer, zoneout=True)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)

            self.assertArrayEqual(tf.shape(qrnn.tensor), (batch_size, seq_size, num_filters))
            self.assertArrayEqual(tf.shape(qrnn_zoneout.tensor), (batch_size, seq_size, num_filters))

            self.assertArrayEqual(qrnn.tensor, qrnn2.tensor)
            # this might fail, zoneout is stochastic
            # self.assertFalse(np.array_equal(qrnn, qrnn_zoneout))

    def test_bias_reuse(self):
        in1 = TensorLayer([[1.]], 1)
        in2 = TensorLayer([[1.]], 1)

        b1 = Bias(in1)
        b2 = b1.reuse_with(in2)

        self.assertListEqual(b1.variable_names, b2.variable_names)

    def test_reusable_layers(self):
        in1 = TensorLayer([[1.]], 1)
        in2 = TensorLayer([[1.]], 1)
        in3 = TensorLayer([[-1.]], 1)

        layer1 = Linear(in1, 2, add_bias=True)

        layer2 = layer1.reuse_with(in2)
        layer3 = layer2.reuse_with(in3)

        self.assertListEqual(layer1.variable_names, layer2.variable_names)
        self.assertListEqual(layer2.variable_names, layer3.variable_names)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertArrayEqual(layer1.tensor, layer2.tensor)
            self.assertArrayEqual(layer3.tensor, layer2.tensor * -1)

    def test_input(self):
        """ Test Input layer - creates a TensorFlow Placeholder

        this corresponds to an input layer with n_units in the input
        and a shape corresponding to [batch_size, n_units]
        """
        in_layer = Input(n_units=10)
        self.assertIsInstance(in_layer.tensor, tf.Tensor)

        ones = np.ones(shape=(2, 10))
        ones_wrong_shape = np.ones([2, 11])

        with self.cached_session(use_gpu=True):
            result = self.eval(in_layer.tensor, {in_layer.tensor: ones})
            self.assertArrayEqual(ones, result)

            variables = in_layer.variable_names
            self.assertEqual(len(variables), 0)

            self.assertRaises(ValueError, self.eval, in_layer.tensor, {in_layer.tensor: ones_wrong_shape})

    def test_input_default_values(self):
        in_layer = Input(n_units=1, value=[[2]], dtype=tf.float32)

        with self.cached_session(use_gpu=True) as ss:
            res1 = self.eval(in_layer.tensor, feed_dict={in_layer.placeholder: in_layer.value})
            res2 = in_layer.eval()
            res3 = in_layer.eval(session=ss, feed_dict={in_layer.placeholder: in_layer.value})

            self.assertArrayEqual(res1, res2)
            self.assertArrayEqual(res1, res3)

    def test_input_default_sparse(self):
        in_layer = Input(n_units=4, n_active=1, value=[[2], [3]], dtype=tf.float32)
        with self.cached_session(use_gpu=True):
            result = in_layer.eval()
            self.assertArrayEqual([[0, 2], [1, 3]], result.indices)

    def test_sparse_input_default_values(self):
        sp_value = tf.SparseTensorValue(indices=[[0, 1], [1, 3]], values=[1., 2.], dense_shape=[2, 4])
        in_layer = SparseInput(n_units=4, value=sp_value, dtype=tf.float32)

        with self.cached_session(use_gpu=True):
            result = in_layer.eval()
            self.assertArrayEqual(result.indices, sp_value.indices)
            self.assertArrayEqual(result.values, sp_value.values)

    def test_flat_sparse_input(self):
        """ Create a Sparse Input by providing
        a n_active parameter
        """
        dim = 4
        index = [[0]]

        input_layer = Input(n_units=dim, n_active=1, dtype=tf.int64)
        self.assertEqual(input_layer.tensor.values.dtype, tf.int64)

        with self.cached_session(use_gpu=True):
            result = self.eval(input_layer.tensor, {input_layer.placeholder: index})
            self.assertEqual(len(result.values), 1)

    def test_sparse_input(self):
        indices = [[0, 1], [1, 1]]
        values = [1, 1]
        dense_shape = [4, 4]
        sp_data = tf.SparseTensorValue(indices, values, dense_shape)

        sp_input = SparseInput(n_units=4)

        with self.cached_session(use_gpu=True):
            result = self.eval(sparse.to_dense(sp_input.tensor), {sp_input.placeholder: sp_data})
            self.assertArrayEqual(result[tuple(sp_data.indices)], [1, 1])
            self.assertArrayEqual(result.shape, dense_shape)

    def test_tensor_input(self):
        indices = [[0, 1], [1, 1]]
        values = [1, 1]
        dense_shape = [4, 4]
        sp_data = tf.SparseTensorValue(indices, values, dense_shape)

        # test with sparse tensor value
        tensor_input = TensorLayer(tensor=sp_data, n_units=4)
        sparse_input = SparseInput(n_units=4)

        self.assertTrue(tensor_input.is_sparse())
        self.assertTrue(sparse_input.is_sparse())

        with self.cached_session(use_gpu=True):
            result_tensor = self.eval(sparse.to_dense(tensor_input.tensor))
            result_sparse = self.eval(sparse.to_dense(sparse_input.tensor), {sparse_input.tensor: sp_data})

            self.assertArrayEqual(result_tensor, result_sparse)

            dense_input = TensorLayer(result_tensor, n_units=4)
            self.assertArrayEqual(dense_input.tensor, result_tensor)

    def test_linear_equal_sparse_dense(self):
        index = 0
        dim = 10

        # x1 = dense input / x2 = sparse input / x3 = sparse input (explicit)
        x1 = Input(n_units=dim)
        x2 = Input(n_units=dim, n_active=1, dtype=tf.int64)
        x3 = SparseInput(10)

        # two layers with shared weights, one uses a sparse input layer, the other the dense one
        y1 = Linear(x1, 4, name="linear1")
        y2 = Linear(x2, 4, shared_weights=y1.weights, name="linear2")
        y3 = Linear(x3, 4, shared_weights=y1.weights, name="linear3")

        init = tf.global_variables_initializer()

        # dummy input data
        input1 = np.zeros([1, dim])
        input1[0, index] = 1
        input2 = [[index]]
        input3 = sparse_tensor_value_one_hot(input2, [1, dim])
        self.assertIsInstance(input3, tf.SparseTensorValue)

        with self.cached_session(use_gpu=True):
            self.eval(init)
            # one evaluation performs a embedding lookup and reduce sum, the other uses a matmul
            y1_output = self.eval(y1.tensor, {x1.placeholder: input1})
            y2_output = self.eval(y2.tensor, {x2.placeholder: input2})
            y3_output = self.eval(y3.tensor, {x3.placeholder: input3})

            self.assertArrayEqual(y1_output, y2_output)
            self.assertArrayEqual(y2_output, y3_output)

    def test_linear_dropconnect(self):

        inputs = TensorLayer(np.ones([4, 6]), dtype=tf.float32)
        inputs2 = TensorLayer(np.ones([4, 6]), dtype=tf.float32)

        layer = Linear(inputs, 6)
        drop = DropConnect(layer, keep_prob=0.5)
        drop2 = drop.reuse_with(inputs2)

        drop3 = DropConnect(layer, keep_prob=0.5)

        init = tf.global_variables_initializer()
        with self.cached_session(use_gpu=True):
            self.eval(init)
            w, b, drop_w, drop_b = self.eval(tensors=[layer.weights, layer.bias, drop.weights, drop.bias])
            self.assertArrayEqual(w[np.nonzero(drop_w)], drop_w[np.nonzero(drop_w)])
            self.assertArrayEqual(b[np.nonzero(drop_b)], b[np.nonzero(drop_b)])

            out_drop, out_drop2, out_drop3 = self.eval(tensors=[drop.tensor, drop2.tensor, drop3.tensor])
            self.assertArrayEqual(out_drop, out_drop2, out_drop3)
            self.assertArrayNotEqual(out_drop, out_drop3)

            inner_drop, inner_drop2 = self.eval(tensors=[drop.inner_layer.tensor, drop3.inner_layer.tensor])
            self.assertArrayEqual(inner_drop, inner_drop2)

    def test_linear_variable_names(self):
        inputs = TensorLayer([[1]], 1, dtype=tf.float32)
        layer = Linear(inputs, 10, add_bias=True, name="layer1")
        layer2 = Linear(inputs, 10, add_bias=True, name="layer2")
        layer_shared = Linear(inputs, 10, shared_weights=layer.weights, add_bias=True, name="layer3")

        var_names = layer.variable_names
        var_names_2 = layer2.variable_names
        shared_names = layer_shared.variable_names

        self.assertEqual(var_names[0], shared_names[0])
        self.assertNotEqual(var_names_2[0], shared_names[0])

        self.assertNotEqual(var_names[1], var_names_2[1])
        self.assertNotEqual(var_names[1], shared_names[1])

        with self.cached_session():
            with tf.variable_scope("", reuse=True):
                weights1 = layer.weights
                weights2 = layer_shared.weights

                self.assertIs(weights1, weights2)

                # print(tf.trainable_variables())

    def test_linear_none_n_units(self):
        in1 = TensorLayer([[2], [2]], 1)
        w = VariableLayer(in1)
        l1 = Linear(in1, n_units=None, shared_weights=w.tensor, transpose_weights=True)

        with self.cached_session(use_gpu=True):
            self.eval(tf.global_variables_initializer())
            res1 = l1.eval()
            res2 = l1.eval()
            # the result should be the same because we
            # call linear multiplication on tensor which saves the value of the input to the variable
            # in this case it doesn't change, but WE DO assign it multiple times by using tensor
            # instead of variable
            self.assertArrayEqual(res1, res2)

    def test_linear_shared(self):
        in1 = TensorLayer([[-1.]], 1)
        in2 = TensorLayer([[2.]], 1)

        l1 = Linear(in1, 1, weight_init=ones_init(), add_bias=True)
        l2 = l1.reuse_with(in2, name="shared")
        l3 = Linear(in1, 1, weight_init=ones_init(), shared_weights=l1.weights, add_bias=True)
        l4 = l3.reuse_with(in2)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertArrayEqual(l1.tensor, l3.tensor)
            self.assertArrayEqual(l2.tensor, l4.tensor)

            self.assertListEqual(l1.variable_names, l2.variable_names)

            self.assertFalse(l3.variable_names == l1.variable_names)
            self.assertListEqual(l3.variable_names, l4.variable_names)

    def test_to_sparse(self):
        index = 0
        dim = 10

        # dense input
        x1 = Input(n_units=dim)
        x2 = Input(n_units=dim, n_active=1, dtype=tf.int64)
        x3 = SparseInput(10)

        # dummy input data
        input1 = np.zeros([1, dim])
        input1[0, index] = 1
        input2 = [[index]]
        input3 = sparse_tensor_value_one_hot(input2, [1, dim])

        s1 = ToSparse(x1)
        s2 = ToSparse(x2)
        s3 = ToSparse(x3)

        with self.cached_session(use_gpu=True):
            y1_sp_tensor = self.eval(s1.tensor, {x1.placeholder: input1})
            self.assertEqual(len(y1_sp_tensor.values), 1)

            y2_sp_tensor = self.eval(s2.tensor, {x2.placeholder: input2})
            self.assertEqual(len(y1_sp_tensor.values), 1)

            self.assertArrayEqual(y1_sp_tensor.indices, y2_sp_tensor.indices)
            self.assertArrayEqual(y1_sp_tensor.values, y2_sp_tensor.values)

            y3_sp_tensor = self.eval(s3.tensor, {x3.placeholder: input3})
            self.assertEqual(len(y2_sp_tensor.values), 1)
            self.assertEqual(y2_sp_tensor.values, 1)

            self.assertArrayEqual(y1_sp_tensor.indices, y3_sp_tensor.indices)
            self.assertArrayEqual(y1_sp_tensor.values, y3_sp_tensor.values)

    def test_to_dense(self):
        dim = 10
        n_active = 1
        index = 0

        x1 = Input(n_units=dim, n_active=n_active, dtype=tf.int64)
        x2 = SparseInput(10)

        data1 = [[index]]
        data2 = sparse_tensor_value_one_hot(data1, [1, dim])

        expected = np.zeros([1, dim])
        expected[0, index] = 1

        to_dense1 = ToDense(x1)
        to_dense2 = ToDense(x2)

        with self.cached_session(use_gpu=True):
            result1 = self.eval(to_dense1.tensor, {x1.placeholder: data1})
            result2 = self.eval(to_dense2.tensor, {x2.placeholder: data2})
            self.assertArrayEqual(result1, expected)
            self.assertArrayEqual(result2, expected)

    def test_reuse_dropout(self):
        x1 = TensorLayer(np.ones(shape=[2, 4]), dtype=tf.float32)

        x2 = Activation(x1)

        drop1 = Dropout(x2, keep_prob=0.5)
        drop2 = Dropout(x2, keep_prob=0.5)
        drop3 = drop1.reuse_with(x1)

        with self.cached_session(use_gpu=True):
            # self.assertArrayEqual(drop1.dropout_mask, drop3.dropout_mask)
            d1, d2, d3 = self.eval([drop1.tensor, drop2.tensor, drop3.tensor])

            self.assertIs(drop1.random_state, drop3.random_state)
            self.assertIsNot(drop1.random_state, drop2.random_state)

            self.assertArrayEqual(d1, d3)
            self.assertArrayNotEqual(d1, d2)

    def test_dropout_layer(self):
        dim = 100
        keep_prob = 0.5
        num_iter = 50

        dense_input = Input(dim)
        data = np.ones([1, dim], dtype=np.float32)
        dropout = Dropout(dense_input, keep_prob)

        with self.cached_session(use_gpu=True):
            # TEST DROPOUT WITH DENSE INPUTS
            final_count = 0
            for _ in range(0, num_iter):
                result = self.eval(dropout.tensor, {dense_input.placeholder: data})
                final_count += np.count_nonzero(result)

                # test the scaling
                sorted_result = np.unique(np.sort(result))
                if len(sorted_result) > 1:
                    self.assertAllClose(1 / keep_prob, sorted_result[1])

            # Check that we are in the 10% error range
            expected_count = dim * keep_prob * num_iter
            rel_error = math.fabs(math.fabs(final_count - expected_count) / expected_count)
            self.assertLess(rel_error, 0.1)

            # TEST DROPOUT WITH keep_prob = 1
            drop_dense = Dropout(dense_input, keep_prob=1)
            result = self.eval(drop_dense.tensor, {dense_input.placeholder: data})
            self.assertArrayEqual(result, data)

            # TEST FLAT INDEX SPARSE INPUT
            n_active = 2
            data = [list(range(0, n_active, 1))]
            flat_sparse = Input(dim, n_active)
            self.assertTrue(flat_sparse.is_sparse())

            dropout = Dropout(flat_sparse, keep_prob)
            self.assertTrue(dropout.is_sparse())

            result = self.eval(dropout.tensor, {flat_sparse.placeholder: data})
            self.assertAllClose(1 / keep_prob, result.values)
            self.assertLessEqual(len(result.values), len(data[0]))

            # test for keep_prob == 1
            dropout = Dropout(flat_sparse, keep_prob=1)
            after_dropout = self.eval(dropout.tensor, {flat_sparse.placeholder: data})
            after_input = self.eval(flat_sparse.tensor, {flat_sparse.placeholder: data})
            self.assertArrayEqual(after_input.indices, after_dropout.indices)

            # TEST DROPOUT ON SPARSE INPUT
            sparse_data = sparse_tensor_value_one_hot(data, [1, dim])
            sparse_input = SparseInput(dim)
            dropout = Dropout(sparse_input, keep_prob=keep_prob)

            # feed sparse tensor values with indices
            after_dropout = self.eval(dropout.tensor, {sparse_input.placeholder: sparse_data})
            np.testing.assert_allclose(1 / keep_prob, after_dropout.values)
            self.assertLessEqual(len(after_dropout.indices), len(sparse_data.indices))

            dropout = Dropout(sparse_input, keep_prob=1)
            before_dropout = self.eval(sparse_input.tensor, {sparse_input.placeholder: sparse_data})
            after_dropout = self.eval(dropout.tensor, {sparse_input.placeholder: sparse_data})
            self.assertArrayEqual(before_dropout.indices, after_dropout.indices)
            self.assertArrayEqual(before_dropout.values, after_dropout.values)

    def test_dropout_noise_mask(self):
        embed_dim = 4
        seq_size = 2
        input_dim = 1000

        tensor_input = TensorLayer(tf.constant([[0, 1], [0, 1]]), 2)

        lookup = Lookup(tensor_input, seq_size, lookup_shape=[input_dim, embed_dim], batch_padding=False)

        dropped = Dropout(lookup, keep_prob=0.5, noise_shape=[2, seq_size, embed_dim])

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)

            w, d = self.eval([lookup.weights, dropped.tensor])

    def test_dropout_reuse(self):
        x1 = TensorLayer(np.ones(shape=[2, 4]) * 2, dtype=tf.float32)
        x2 = TensorLayer(np.ones(shape=[2, 4]), dtype=tf.float32)

        zone1 = ZoneOut(x2, x1, keep_prob=0.5)
        zone2 = ZoneOut(x2, x1, keep_prob=0.5)
        zone3 = zone1.reuse_with(x2, x1)

        with self.cached_session(use_gpu=True):
            # self.assertArrayEqual(drop1.dropout_mask, drop3.dropout_mask)
            d1, d2, d3 = self.eval([zone1.tensor, zone2.tensor, zone3.tensor])

            self.assertIs(zone1.mask, zone3.mask)
            self.assertIsNot(zone1.mask, zone2.mask)

            self.assertArrayEqual(d1, d3)
            self.assertArrayNotEqual(d1, d2)

    def test_zoneout_layer(self):
        dim = 100
        batch_size = 1000
        keep_prob = 0.5

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        with self.cached_session(use_gpu=True):
            self.assertAlmostEqual(tf.reduce_mean(tf.reduce_sum(zoneout.tensor, -1), -1), 0., delta=1.0)

        # test keep_prob = 1
        keep_prob = 1.0

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        with self.cached_session(use_gpu=True):
            self.assertEqual(tf.reduce_mean(tf.reduce_sum(zoneout.tensor, -1), -1), dim)

        # test keep_prob = 0
        keep_prob = 0.0

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        with self.cached_session(use_gpu=True):
            self.assertEqual(tf.reduce_mean(tf.reduce_sum(zoneout.tensor, -1), -1), -dim)

        # test keep_prob = 0
        keep_prob = np.random.rand()

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        with self.cached_session(use_gpu=True):
            expected = (2 * dim * keep_prob) - dim
            self.assertAlmostEqual(tf.reduce_mean(tf.reduce_sum(zoneout.tensor, -1), -1), expected, delta=1.0)

    def test_gaussian_noise(self):
        dim = 1000
        # for sparse inputs
        n_active = 10

        dense_input = Input(dim)
        dense_data = np.ones([1, dim], dtype=np.float32)
        noise_layer = GaussianNoise(dense_input)

        # test that expected average tensor is approximately the same
        with self.cached_session(use_gpu=True):
            result = self.eval(noise_layer.tensor, {dense_input.placeholder: dense_data})
            mean_result = np.mean(result)
            mean_data = np.mean(dense_data)
            self.assertAlmostEqual(mean_data, mean_result, delta=0.1)

            # sparse input with flat indices
            flat_indices = [list(range(0, n_active, 1))]
            flat_input = Input(dim, n_active, dtype=tf.int64)
            noise_layer = GaussianNoise(flat_input)
            result = self.eval(noise_layer.tensor, {flat_input.placeholder: flat_indices})

            dense_input = np.zeros([1, dim])
            dense_input[0, flat_indices[0]] = 1
            mean_data = np.mean(dense_input)
            mean_result = np.mean(result)
            self.assertAlmostEqual(mean_data, mean_result, delta=0.1)

            sparse_input = SparseInput(dim)
            noise_layer = GaussianNoise(sparse_input)
            sparse_data = sparse_tensor_value_one_hot(flat_indices, [1, dim])
            result = self.eval(noise_layer.tensor, {sparse_input.placeholder: sparse_data})
            mean_result = np.mean(result)
            self.assertAlmostEqual(mean_data, mean_result, delta=0.1)

    def test_sp_noise(self):
        # PARAMS
        noise_amount = 0.5
        batch_size = 4
        dim = 100

        dense_input = Input(dim)
        dense_data = np.zeros([batch_size, dim], dtype=np.float32)
        noise_layer = SaltPepperNoise(dense_input, noise_amount)

        with self.cached_session(use_gpu=True):
            result = self.eval(noise_layer.tensor, {dense_input.placeholder: dense_data})
            mean_result = np.mean(result)
            self.assertEqual(mean_result, 0)

    def test_sp_noise_sp(self):
        noise_density = 0.5
        batch_size = 4
        dim = 1000

        data = np.random.randint(0, dim, [batch_size, 1])

        x = Input(dim, n_active=1)
        n = SaltPepperNoise(x, density=noise_density, pepper_value=1)

        feed = {x.placeholder: data}

        with  self.cached_session(use_gpu=True):
            sum_res = tf.sparse_reduce_sum(n.tensor, axis=-1)
            expected = [dim * noise_density] * batch_size
            self.assertAllClose(self.eval(sum_res, feed), expected, atol=1)

    def test_activation_with_params(self):
        inputs = Input(1)
        act = Activation(inputs, leaky_relu, alpha=0.)

        with self.cached_session(use_gpu=True):
            r0 = self.eval(act.tensor, {inputs.tensor: [[-1]]})
            r1 = self.eval(act.tensor, {inputs.tensor: [[1]]})
            r2 = self.eval(act.tensor, {inputs.tensor: [[3]]})

            self.assertEqual(r0[0], 0)
            self.assertEqual(r1[0], 1)
            self.assertEqual(r2[0], 3)

    def test_layers_to_list(self):
        """ layers_to_list returns the layers without repetition using a breadth first search from the last layer
        and then reversing the layers found.
        """
        l11 = Input(1, name="in1")
        l12 = Input(1, name="in2")
        l121 = WrapLayer(l12, n_units=l12.n_units, wrap_fn=lambda x: tf.identity(x))
        l2 = Add(l11, l121)

        l3 = Linear(l2, 1)
        l4 = Add(l3, l12)

        l41 = Activation(l4, fn=sigmoid, name="act1")
        l42 = Activation(l4, fn=hard_sigmoid, name="act2")

        l5 = ToSparse(l41)

        g1 = Module(l3, l5)
        g2 = Module(l3, l42)

        all_layers = set(g1.graph.nodes).union(g2.graph.nodes)

        self.assertEqual(len(all_layers), 6)
        self.assertIn(l12, all_layers)
        self.assertNotIn(l2, all_layers)
        self.assertNotIn(l11, all_layers)

    def test_wrap_layer(self):
        data = np.random.uniform(-1, 1, [1, 4])

        input_layer = Input(4)
        wrap_layer = WrapLayer(input_layer, n_units=4, wrap_fn=lambda layer: tf.multiply(layer, 2))
        self.assertIs(input_layer.placeholder, wrap_layer.placeholder)

        with self.cached_session(use_gpu=True):
            t1 = self.eval(input_layer.tensor, {input_layer.placeholder: data})
            t2 = self.eval(wrap_layer.tensor, {wrap_layer.placeholder: data})

            self.assertAllClose(t1 * 2, t2, atol=1e-6)

    def test_wrap_reuse(self):
        """

                     +---------------------------------------+
                     | +----------------------------+        |
                     | | +------------+             |        |
                     | | |            | WRAP        | WRAP   |
                     | | |   INPUT    |             |        |
            +--------------> LAYER    |             |        +------->
                     | | |            |             |        |
                     | | +------------+             |        |
                     | +----------------------------+        |
                     +---------------------------------------+

        """
        input1 = TensorLayer(np.array([1, 1, 1, 1]), 4)
        input2 = TensorLayer(np.array([0, 1, 0, 1]), 4)

        wrap1 = WrapLayer(input1, n_units=input1.n_units, wrap_fn=lambda x: tf.multiply(x, 2))
        wrap2 = WrapLayer(wrap1, n_units=wrap1.n_units, wrap_fn=lambda x: tf.multiply(x, 2))

        with self.assertRaises(AttributeError):
            wrap1.reuse_with(input2)
            # this will try to call reuse on wrap1 which will call reuse in TensorLayer
            wrap2.reuse_with(input2)

        """
        

                         +---------------------------------------+
                         | +----------------------------+        |
                         | | +------------+             |        |
                         | | |            | WRAP        | WRAP   |
                         | | | ACTIVATION |             |        |
           INPUT +--------------> LAYER   |             |        +------->
                         | | |            |             |        |
                         | | +------------+             |        |
                         | +----------------------------+        |
                         +---------------------------------------+

        """

        input1 = TensorLayer(np.array([1, 1, 1, 1]), 4)
        input2 = TensorLayer(np.array([0, 1, 0, 1]), 4)

        input1_act = Activation(input1, fn=lambda x: tf.identity(x))

        # since there's an attribute forward wrap_fn cannot be fn internally if other layers use that attribute
        wrap1 = WrapLayer(input1_act, n_units=input1_act.n_units, wrap_fn=lambda x: tf.multiply(x, 2), attr_fwd="fn")
        wrap2 = WrapLayer(wrap1, n_units=wrap1.n_units, wrap_fn=lambda x: tf.multiply(x, 2))

        # this is ok because we're wrapping the activation
        wrap2_r1 = wrap2.reuse_with(input2)
        wrap2_r2 = wrap2_r1.reuse_with(input2)

        self.assertTrue(hasattr(wrap2_r2, "fn"))

        with self.cached_session(use_gpu=True):
            self.assertArrayNotEqual(tf.reduce_sum(wrap2.tensor), tf.reduce_sum(wrap2_r2.tensor))
            self.assertArrayEqual(tf.reduce_sum(wrap2.tensor), tf.reduce_sum(wrap2_r2.tensor * 2))

    def test_lookup_sequence_dense(self):
        input_dim = 4
        embed_dim = 3
        seq_size = 2
        batch_size = 3

        inputs = Input(2, dtype=tf.int64)
        input_data = np.array([[2, 0], [1, 2]])

        tensor_input = TensorLayer(tf.constant([2]), 1)

        lookup = Lookup(inputs, seq_size, lookup_shape=[input_dim, embed_dim], batch_size=batch_size,
                        batch_padding=True)

        lookup_from_tensor = lookup.reuse_with(tensor_input)

        init = tf.global_variables_initializer()
        with self.cached_session(use_gpu=True):
            self.eval(init)

            v1 = self.eval(lookup.tensor, {inputs.placeholder: input_data})
            v2 = self.eval(lookup_from_tensor.tensor)

            self.assertEqual(np.shape(v1), (batch_size, seq_size, embed_dim))
            self.assertEqual(np.shape(v2), (batch_size, seq_size, embed_dim))

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
        seq_lookup = lookup.as_seq()

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

    def test_gating(self):
        vocab_size = 4
        n_features = 3
        seq_size = 2

        inputs = Input(seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2]])

        features = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features]).as_concat()
        sp_features = ToSparse(features)

        gate_w = Linear(features, n_units=seq_size, add_bias=True)
        gate1 = Gate(features, gate_w)
        gate2 = gate1.reuse_with(sp_features)

        init = tf.global_variables_initializer()
        feed = {inputs.placeholder: input_data}

        with self.cached_session(use_gpu=True):
            self.eval(init)
            r1 = self.eval(gate1.tensor, feed)
            r2 = self.eval(gate2.tensor, feed)

            self.assertArrayEqual(r1, r2)

    def test_coupled_gate(self):

        vocab_size = 4
        n_features = 3
        seq_size = 2

        inputs = Input(seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2]])

        features1 = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features]).as_concat()
        features2 = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features]).as_concat()

        sp_features1 = ToSparse(features1)

        gate_w = Linear(features1, seq_size, add_bias=True)
        coupled_gate = CoupledGate(features1, features2, gate_w)

        coupled_gate2 = coupled_gate.reuse_with(sp_features1, features2)

        init = tf.global_variables_initializer()

        feed = {inputs.placeholder: input_data}

        with self.cached_session(use_gpu=True):
            self.eval(init)
            r1 = self.eval(coupled_gate.tensor, feed)
            r2 = self.eval(coupled_gate2.tensor, feed)

            self.assertArrayEqual(r1, r2)

    def test_rnn_cell(self):

        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = Input(n_inputs)
        rnn_1 = RNNCell(inputs, n_hidden)
        rnn_2 = rnn_1.reuse_with(inputs, rnn_1)

        rnn_3 = rnn_1.reuse_with(inputs)

        init = tf.global_variables_initializer()

        data = np.ones([batch_size, 4])

        with self.cached_session(use_gpu=True):
            self.eval(init)
            res1 = self.eval(rnn_1.tensor, {inputs.placeholder: data})
            res2 = self.eval(rnn_2.tensor, {inputs.placeholder: data})
            res3 = self.eval(rnn_3.tensor, {inputs.placeholder: data})

            self.assertEqual((batch_size, n_hidden), np.shape(res1))
            self.assertArrayEqual(res1, res3)
            self.assertArrayNotEqual(res1, res2)

    def test_rnn_cell_drop(self):

        n_hidden = 4
        inputs1 = TensorLayer(np.ones([2, 4]), dtype=tf.float32)
        inputs2 = TensorLayer(np.ones([2, 4]), dtype=tf.float32)

        rnn1 = RNNCell(inputs1, n_hidden,
                       u_regularizer=partial(DropConnect, keep_prob=0.5),
                       w_regularizer=partial(Dropout, keep_prob=0.5),
                       regularized=True
                       )
        rnn2 = rnn1.reuse_with(inputs2, previous_state=rnn1)
        rnn3 = rnn1.reuse_with(inputs2, previous_state=rnn1)
        rnn4 = rnn1.reuse_with(inputs2, previous_state=None, regularized=False)
        rnn5 = rnn4.reuse_with(inputs2, previous_state=None, regularized=True)

        with self.cached_session(use_gpu=True):
            self.eval(tf.global_variables_initializer())

            r1, r2, r3, r4 = self.eval([rnn1.tensor, rnn2.tensor, rnn3.tensor, rnn4.tensor])
            mask1, mask2 = self.eval([rnn1.w.random_state, rnn2.w.random_state])

            self.assertArrayEqual(mask1, mask2)
            self.assertArrayNotEqual(r1, r2)
            self.assertArrayEqual(r2, r3)
            self.assertArrayNotEqual(r2, r4)

            self.assertTrue(isinstance(rnn1.w, ViewLayer))
            self.assertTrue(isinstance(rnn2.w, ViewLayer))
            self.assertTrue(isinstance(rnn3.w, ViewLayer))
            self.assertFalse(isinstance(rnn4.w, ViewLayer))
            self.assertTrue(isinstance(rnn5.w, ViewLayer))

    def test_lstm_cell(self):

        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = Input(n_inputs)
        rnn_1 = LSTMCell(inputs, n_hidden)
        rnn_2 = rnn_1.reuse_with(inputs,
                                 previous_state=rnn_1)

        # if we don't wipe the memory it reuses it
        rnn_3 = rnn_1.reuse_with(inputs,
                                 previous_state=None,
                                 memory_state=LSTMCell.zero_state(inputs, rnn_1.n_units))

        init = tf.global_variables_initializer()

        data = np.ones([batch_size, 4])

        with self.cached_session(use_gpu=True):
            self.eval(init)
            res1 = self.eval(rnn_1.tensor, {inputs.placeholder: data})
            res2 = self.eval(rnn_2.tensor, {inputs.placeholder: data})
            res3 = self.eval(rnn_3.tensor, {inputs.placeholder: data})

            self.assertEqual((batch_size, n_hidden), np.shape(res1))
            self.assertArrayEqual(res1, res3)
            self.assertArrayNotEqual(res1, res2)

    def test_lstm_layer(self):
        n_features = 10
        embed_size = 4
        lstm_size = 5
        seq_size = 3
        batch_size = 2

        inputs = TensorLayer(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
        lookup = Lookup(inputs, seq_size=seq_size, lookup_shape=[n_features, embed_size])
        seq = lookup.as_seq()

        lstm = LSTM(seq, seq_size=3, n_units=lstm_size)
        lstm2 = lstm.reuse_with(seq)

        with self.cached_session(use_gpu=True):
            self.eval(tf.global_variables_initializer())

            self.assertArrayEqual(lstm.tensor, lstm2.tensor)

    def test_gru_cell(self):
        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = Input(n_inputs)
        rnn_1 = GRUCell(inputs, n_hidden)
        rnn_2 = rnn_1.reuse_with(inputs,
                                 previous_state=rnn_1)

        # if we don't wipe the memory it reuses it
        rnn_3 = rnn_1.reuse_with(inputs,
                                 previous_state=GRUCell.zero_state(inputs, rnn_1.n_units))

        init = tf.global_variables_initializer()

        data = np.ones([batch_size, 4])

        with self.cached_session(use_gpu=True):
            self.eval(init)

            res1 = self.eval(rnn_1.tensor, {inputs.placeholder: data})
            res2 = self.eval(rnn_2.tensor, {inputs.placeholder: data})
            res3 = self.eval(rnn_3.tensor, {inputs.placeholder: data})

            self.assertEqual((batch_size, n_hidden), np.shape(res1))
            self.assertArrayEqual(res1, res3)
            self.assertArrayNotEqual(res1, res2)

    def test_module(self):
        l1 = Input(1, name="in1")
        l2 = Input(1, name="in2")
        l3 = Add(l1, l2)
        l4 = Add(l1, l2)
        l5 = Linear(l4, 1)
        t1 = TensorLayer([[1]], n_units=1, dtype=tf.float32)
        l6 = Add(l3, t1)
        l7 = Add(l6, l5)

        t2 = TensorLayer([[1]], n_units=1, dtype=tf.float32)
        t3 = TensorLayer([[1]], n_units=1, dtype=tf.float32)

        m = Module([l1, l2, t1], l7)
        with tf.name_scope("module_reuse"):
            m2 = m.reuse_with(t2, t3, t1)

        init = tf.global_variables_initializer()

        feed = {l1.placeholder: [[1]], l2.placeholder: [[1]]}

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.eval(m.tensor, feed)
            self.eval(m2.tensor)

    def test_module_gate(self):
        l1 = Input(4, name="in1")
        l2 = Input(2, name="in2")

        gate = Gate(layer=l1, gate_input=l2)
        gate_module = Module([l1, l2], gate)

        # model = Model(run_in_layers=gate_module.input_layers, run_out_layers=gate_module)
        # runner = ModelRunner(model)
        # runner.log_graph("/tmp")

        t1 = TensorLayer([[1, 1, 1, 1]], n_units=4, dtype=tf.float32)
        t2 = TensorLayer([[1, 1]], n_units=2, dtype=tf.float32)

        with tf.name_scope("module_reuse"):
            m2 = gate_module.reuse_with(t1, t2)

        # model = Model(m2.input_layers, m2)
        # runner = ModelRunner(model)
        # runner.log_graph("/tmp/")

    def test_reshape(self):
        v = np.array([[[1], [2]], [[3], [4]]])
        x = TensorLayer(v)

        fl = Reshape(x, [-1, 2])
        fl2 = fl.reuse_with(x)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(fl.tensor, fl2.tensor)

    def test_flatten(self):
        v = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        x = TensorLayer(v, n_units=2)
        fl = Flatten(x)

        rs = Reshape(x, [2, -1])
        fl2 = fl.reuse_with(x)

        with self.cached_session(use_gpu=True):
            self.assertArrayEqual(fl.tensor, rs.tensor)
            self.assertArrayEqual(fl.tensor, fl2.tensor)

            self.assertArrayEqual(x.shape, [2])
            self.assertSequenceEqual(fl.shape, [2, 6])

    def test_batch_norm(self):

        v = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [-1, 1, -1, -1]])
        x = TensorLayer(v, n_units=4, dtype=tf.float32)
        xs = ToSparse(x)

        inputs_shape = v.shape
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]

        # during training
        decay = 0.999
        epsilon = 0.001

        # this can be params of the layer
        beta = tf.get_variable('beta',
                               shape=params_shape,
                               dtype=x.dtype,
                               initializer=tf.zeros_initializer,
                               trainable=True)

        gamma = tf.get_variable('gamma',
                                shape=params_shape,
                                dtype=x.dtype,
                                initializer=tf.ones_initializer,
                                trainable=True)

        # these are not trainable but updated each time we compute mean and variance
        moving_mean = tf.get_variable("moving_mean",
                                      shape=params_shape,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_var = tf.get_variable("moving_var",
                                     shape=params_shape,
                                     initializer=tf.zeros_initializer(),
                                     trainable=False)

        # Calculate the moments based on the individual batch.
        mean, variance = tf.nn.moments(x.tensor, axis, shift=moving_mean)

        """Some training algorithms, such as GradientDescent and Momentum often benefit 
            from maintaining a moving average of variables during optimization. 
            Using the moving averages for evaluations often improve results significantly.
        """
        from tensorflow.python.training.moving_averages import assign_moving_average

        """  The moving average of 'variable' updated with 'value' is:
            variable * decay + value * (1 - decay)
        """

        update_mv_avg = assign_moving_average(moving_mean, mean, decay)
        update_mv_var = assign_moving_average(moving_var, variance, decay)

        with tf.control_dependencies([update_mv_avg, update_mv_var]):
            outputs = tf.nn.batch_normalization(
                x.tensor, mean, variance, beta, gamma, epsilon)

        # if not training instead of using mean and variance we use an estimate
        # for the pop ulation mean and variance computed for example from the
        # exponential moving averages

        # outputs = nn.batch_normalization(
        #    inputs, moving_mean, moving_variance, beta, gamma, epsilon)
        # outputs.set_shape(inputs.get_shape())

        bn = BatchNorm(x, beta=beta, gamma=gamma, center=True, scale=True)
        bn2 = BatchNorm(x, gamma=gamma, center=True, scale=True, beta_init=random_uniform(0, 1))
        bn3 = BatchNorm(x, beta=beta, gamma=gamma, center=True, scale=True, beta_init=random_uniform(0, 1))

        bn_simple = BatchNorm(x, scale=False, center=False)

        bn_inference = bn.reuse_with(x, training=False)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)

            # test updated moving avg
            before = self.eval(bn.moving_mean)
            self.eval(bn.tensor)
            after = self.eval(bn.moving_mean)

            self.assertArrayNotEqual(before, after)
            before = self.eval(bn.moving_mean)
            self.eval(bn.tensor)
            after = self.eval(bn.moving_mean)
            self.assertArrayNotEqual(before, after)

            self.assertArrayEqual(bn.moving_mean, bn_inference.moving_mean)
            before = self.eval(bn_inference.moving_mean)
            self.eval(bn_inference.tensor)
            after = self.eval(bn_inference.moving_mean)
            self.assertArrayEqual(before, after)

            self.assertArrayEqual(outputs, bn.tensor)
            self.assertArrayNotEqual(outputs, bn_inference.tensor)
            # ignores init because we pass beta and gamma
            self.assertArrayEqual(outputs, bn3.tensor)
            self.assertArrayNotEqual(outputs, bn2.tensor)

            self.assertArrayEqual(outputs, bn_simple.tensor)

    def test_batch_norm_sparse(self):

        v = np.array([[1, 1], [2, 3], [-1, 6]])
        x = TensorLayer(v, n_units=2, dtype=tf.float32)
        xs = ToSparse(x)

        bn = BatchNorm(x, training=False, name="bn_infer")
        bns = bn.reuse_with(xs, name="bns_infer")

        # print(bn.moving_mean.op.name)
        # print(bns.moving_mean.op.name)

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            self.assertArrayEqual(bn.tensor, bns.tensor)

            bn = bn.reuse_with(x, training=True, name="bn_train")
            bns = bn.reuse_with(xs, name="bns_train")

            moving_mean_before = self.eval(bn.moving_mean)
            self.assertArrayEqual(bn.tensor, bns.tensor)
            moving_mean_after = self.eval(bn.moving_mean)
            self.assertArrayNotEqual(moving_mean_before, moving_mean_after)

            bn = bn.reuse_with(x, training=False)
            bns = bn.reuse_with(xs)

            self.assertArrayEqual(bn.tensor, bns.tensor)

    def test_batch_norm_mv_average(self):
        t1 = np.array([[1, 1], [2, 3], [-1, 6]])
        t2 = np.array([[5, 5], [1, 2], [-6, -10]])

        x = Input(n_units=2, dtype=tf.float32)
        xs = ToSparse(x)

        bn = BatchNorm(x, training=True, name="bn")
        bns = bn.reuse_with(xs)

        bn_infer = bn.reuse_with(x, training=False, name="bn_infer")

        init = tf.global_variables_initializer()

        with self.cached_session(use_gpu=True):
            self.eval(init)
            r0_infer = self.eval(bn_infer.tensor, {x.placeholder: t1})

            mv0 = self.eval(bn.moving_mean)
            r1 = self.eval(bn.tensor, {x.placeholder: t1})
            mv1 = self.eval(bn.moving_mean)

            r1_infer = self.eval(bn_infer.tensor, {x.placeholder: t1})

            # moving average and variance are updated so they can't be the same
            self.assertArrayNotEqual(mv0, mv1)

            # the result with the same data can't be the same because it uses the
            # estimate for population mean and variance which is updated by the training step
            self.assertArrayNotEqual(r0_infer, r1_infer)

            r2 = self.eval(bn.tensor, {x.placeholder: t2})
            r2_infer = self.eval(bn_infer.tensor, {x.placeholder: t1})

            rs1 = self.eval(bns.tensor, {x.placeholder: t1})
            r3_infer = self.eval(bn_infer.tensor, {x.placeholder: t1})
            rs2 = self.eval(bns.tensor, {x.placeholder: t2})

            # the results should be the same because they are computed based on the current
            # mini-batch mean and variance
            self.assertArrayEqual(r1, rs1)
            self.assertArrayEqual(r2, rs2)

            # again can't be the same because the moving avg changed
            self.assertArrayNotEqual(r1_infer, r2_infer)

            # the reused layer should also update the moving average
            # so the inference step will give a different value again
            self.assertArrayNotEqual(r2_infer, r3_infer)

    if __name__ == '__main__':
        test_utils.main()
