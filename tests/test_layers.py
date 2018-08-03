import unittest
import tensorflow as tf
import numpy as np
from tensorx.layers import *
from tensorx.init import *

from tensorx.layers import layers_to_list
from tensorx.activation import *
from tensorx.transform import sparse_tensor_value_one_hot
from tensorx.train import Model, ModelRunner
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestLayers(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def reset(self):
        tf.reset_default_graph()
        self.ss.close()
        self.ss = tf.InteractiveSession()

    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_compose(self):
        in1 = Input(1)
        in2 = TensorLayer([[1.]], 1)

        l1 = Linear(in1, 4)
        l2 = Activation(l1, relu)

        comp = Compose(l1, l2)
        comp2 = comp.reuse_with(in2)

        tf.global_variables_initializer().run()

        res1 = l2.tensor.eval({in1.placeholder: [[1.]]})
        res2 = comp.tensor.eval({in1.placeholder: [[1.]]})

        res3 = comp2.tensor.eval()

        self.assertTrue(np.array_equal(res1, res2))
        self.assertTrue(np.array_equal(res1, res3))

    def test_fn_compose(self):
        in1 = Input(1)
        in2 = TensorLayer([[1.]], 1)

        l1 = Linear(in1, 4)
        l2 = Activation(l1, relu)

        comp = Compose(l1, l2)
        comp2 = comp.reuse_with(in2)

        fn1 = Fn(in1, 4, fn=relu, share_vars_with=l1)
        fn2 = fn1.reuse_with(in2, name="fn2")

        tf.global_variables_initializer().run()

        feed = {in1.placeholder: [[1.]]}
        res1 = l2.tensor.eval(feed)
        res2 = comp.tensor.eval(feed)
        res3 = comp2.tensor.eval()

        self.assertTrue(np.array_equal(res1, res2))
        self.assertTrue(np.array_equal(res1, res3))

        res_fn1 = fn1.tensor.eval(feed)
        res_fn2 = fn2.tensor.eval()

        self.assertTrue(np.array_equal(res_fn1, res_fn2))

        # m = Model(in1, fn1)
        # r = ModelRunner(m)
        # r.log_graph("/tmp")

    def test_compose_merge(self):
        in1 = Input(1)
        in2 = TensorLayer([[1.]], 1)
        in3 = Input(1)

        a1 = Add(in1, in2)
        l1 = Linear(a1, 4)

        comp = Compose(a1, l1)
        comp2 = comp.reuse_with(in1, in3)

        tf.global_variables_initializer().run()

        res1 = comp.tensor.eval({in1.placeholder: [[1.]]})
        res2 = comp2.tensor.eval({in1.placeholder: [[1.]], in3.placeholder: [[1.]]})

        self.assertTrue(np.array_equal(res1, res2))

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
        conv = tf.nn.conv1d(x, filters, stride=1, padding="SAME", use_cudnn_on_gpu=True, data_format="NWC")

        tf.global_variables_initializer().run()

        self.assertSequenceEqual(conv_layer.filter_shape, (filter_size, input_dim, num_filters))
        self.assertSequenceEqual(conv_layer.shape, (batch_size, seq_size, num_filters))
        self.assertTrue(np.array_equal(conv.eval(), conv_layer.tensor.eval()))

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

        filter = tf.ones(filter_shape)
        conv_layer = CausalConv(x_layer, num_filters, filter_size,
                                shared_filters=filter,
                                dilation_rate=dilation_rate)

        left_pad = dilation_rate * (filter_size - 1)
        padding = [[0, 0], [left_pad, 0], [0, 0]]
        x = tf.pad(x, padding)

        conv = tf.nn.convolution(input=x,
                                 filter=filter,
                                 dilation_rate=(dilation_rate,),
                                 strides=(1,),
                                 padding="VALID",
                                 data_format="NWC")

        tf.global_variables_initializer().run()

        self.assertSequenceEqual(conv_layer.filter_shape, (filter_size, input_dim, num_filters))
        self.assertSequenceEqual(conv_layer.shape, (batch_size, seq_size, num_filters))
        self.assertTrue(np.array_equal(conv.eval(), conv_layer.tensor.eval()))

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

        tf.global_variables_initializer().run()

        res1 = qrnn.tensor.eval()
        res2 = qrnn2.tensor.eval()
        res3 = qrnn_zoneout.tensor.eval()

        self.assertSequenceEqual(np.shape(res1), (batch_size, seq_size, num_filters))
        self.assertSequenceEqual(np.shape(res3), (batch_size, seq_size, num_filters))

        self.assertTrue(np.array_equal(res1, res2))
        self.assertFalse(np.array_equal(res1, res3))

        m = Model(x_layer, qrnn)
        r = ModelRunner(m)
        r.log_graph("/tmp")

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

        layer1 = Linear(in1, 2)

        layer2 = layer1.reuse_with(in2)
        layer3 = layer2.reuse_with(in3)

        self.assertListEqual(layer1.variable_names, layer2.variable_names)
        self.assertListEqual(layer2.variable_names, layer3.variable_names)

        tf.global_variables_initializer().run()

        result1 = layer1.tensor.eval()
        result2 = layer2.tensor.eval()
        result3 = layer3.tensor.eval()

        np.testing.assert_array_equal(result1, result2)

        expected = result2 * -1
        np.testing.assert_array_equal(expected, result3)

    def test_input(self):
        """ Test Input layer - creates a TensorFlow Placeholder

        this corresponds to an input layer with n_units in the input
        and a shape corresponding to [batch_size, n_units]
        """
        in_layer = Input(n_units=10)
        self.assertIsInstance(in_layer.tensor, tf.Tensor)

        ones = np.ones(shape=(2, 10))
        result = self.ss.run(in_layer.tensor, feed_dict={in_layer.tensor: ones})
        np.testing.assert_array_equal(ones, result)

        variables = in_layer.variable_names
        self.assertEqual(len(variables), 0)

        ones_wrong_shape = np.ones(shape=(2, 11))
        try:
            self.ss.run(in_layer.tensor, feed_dict={in_layer.tensor: ones_wrong_shape})
            self.fail("Should have raised an exception since shapes don't match")
        except ValueError:
            pass

    def test_flat_sparse_input(self):
        """ Create a Sparse Input by providing
        a n_active parameter
        """
        dim = 4
        index = [[0]]

        input_layer = Input(n_units=dim, n_active=1, dtype=tf.int64)
        self.assertEqual(input_layer.tensor.values.dtype, tf.int64)

        result = self.ss.run(input_layer.tensor, feed_dict={input_layer.placeholder: index})
        self.assertEqual(len(result.values), 1)

    def test_sparse_input(self):
        indices = [[0, 1], [1, 1]]
        values = [1, 1]
        dense_shape = [4, 4]
        sp_data = tf.SparseTensorValue(indices, values, dense_shape)

        sp_input = SparseInput(n_units=4)
        result = tf.sparse_tensor_to_dense(sp_input.tensor).eval({sp_input.placeholder: sp_data})

        np.testing.assert_array_equal(result[sp_data.indices], [1, 1])
        np.testing.assert_array_equal(result.shape, dense_shape)

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

        result_tensor = tf.sparse_tensor_to_dense(tensor_input.tensor).eval()
        result_sparse = tf.sparse_tensor_to_dense(sparse_input.tensor).eval({sparse_input.tensor: sp_data})

        np.testing.assert_array_equal(result_sparse, result_tensor)

        dense_input = TensorLayer(result_tensor, n_units=4)
        np.testing.assert_array_equal(dense_input.tensor.eval(), result_tensor)

        # np.testing.assert_array_equal(result_tensor[sp_data.indices], [1, 1])
        # np.testing.assert_array_equal(result.shape, dense_shape)

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

        self.ss.run(tf.global_variables_initializer())

        # dummy input data
        input1 = np.zeros([1, dim])
        input1[0, index] = 1
        input2 = [[index]]
        input3 = sparse_tensor_value_one_hot(input2, [1, dim])
        self.assertIsInstance(input3, tf.SparseTensorValue)

        # one evaluation performs a embedding lookup and reduce sum, the other uses a matmul
        y1_output = y1.tensor.eval({x1.placeholder: input1})
        y2_output = y2.tensor.eval({x2.placeholder: input2})
        y3_output = y3.tensor.eval({x3.placeholder: input3})

        # the result should be the same
        np.testing.assert_array_equal(y1_output, y2_output)
        np.testing.assert_array_equal(y2_output, y3_output)

    def test_linear_variable_names(self):
        self.reset()

        inputs = TensorLayer([[1]], 1, dtype=tf.float32)
        layer = Linear(inputs, 10)
        layer2 = Linear(inputs, 10)
        layer_shared = Linear(inputs, 10, shared_weights=layer.weights)

        var_names = layer.variable_names
        var_names_2 = layer2.variable_names
        shared_names = layer_shared.variable_names

        self.assertEqual(var_names[0], shared_names[0])
        self.assertNotEqual(var_names_2[0], shared_names[0])

        self.assertNotEqual(var_names[1], var_names_2[1])
        self.assertNotEqual(var_names[1], shared_names[1])

        with tf.variable_scope("", reuse=True):
            weights1 = tf.get_variable("linear/w")
            weights2 = tf.get_variable(shared_names[0])

            self.assertIs(weights1, weights2)

    def test_linear_shared(self):
        in1 = TensorLayer([[-1.]], 1)
        in2 = TensorLayer([[2.]], 1)

        l1 = Linear(in1, 1, init=ones_init())
        l2 = l1.reuse_with(in2, name="shared")
        l3 = Linear(in1, 1, init=ones_init(), shared_weights=l1.weights, bias=True)
        l4 = l3.reuse_with(in2)

        self.ss.run(tf.global_variables_initializer())

        res1 = l1.tensor.eval()
        res2 = l2.tensor.eval()
        res3 = l3.tensor.eval()
        res4 = l4.tensor.eval()

        self.assertTrue(np.array_equal(res1, res3))
        self.assertTrue(np.array_equal(res2, res4))

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

        y1_sp_tensor = self.ss.run(s1.tensor, {x1.placeholder: input1})

        self.assertEqual(len(y1_sp_tensor.values), 1)

        y2_sp_tensor = self.ss.run(s2.tensor, {x2.placeholder: input2})
        self.assertEqual(len(y1_sp_tensor.values), 1)
        np.testing.assert_array_equal(y1_sp_tensor.indices, y2_sp_tensor.indices)
        np.testing.assert_array_equal(y1_sp_tensor.values, y2_sp_tensor.values)

        y3_sp_tensor = self.ss.run(s3.tensor, {x3.placeholder: input3})
        self.assertEqual(len(y2_sp_tensor.values), 1)
        self.assertEqual(y2_sp_tensor.values, 1)
        np.testing.assert_array_equal(y1_sp_tensor.indices, y3_sp_tensor.indices)
        np.testing.assert_array_equal(y1_sp_tensor.values, y3_sp_tensor.values)

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

        result1 = to_dense1.tensor.eval({x1.placeholder: data1})
        result2 = to_dense2.tensor.eval({x2.placeholder: data2})

        np.testing.assert_array_equal(expected, result1)
        np.testing.assert_array_equal(expected, result2)

    def test_dropout_layer(self):
        dim = 100
        keep_prob = 0.5
        num_iter = 50

        dense_input = Input(dim)
        data = np.ones([1, dim], dtype=np.float32)
        dropout = Dropout(dense_input, keep_prob)

        # TEST DROPOUT WITH DENSE INPUTS
        final_count = 0
        for _ in range(0, num_iter):
            result = dropout.tensor.eval({dense_input.placeholder: data})
            final_count += np.count_nonzero(result)

            # test the scaling
            sorted_result = np.unique(np.sort(result))
            if len(sorted_result) > 1:
                np.testing.assert_allclose(1 / keep_prob, sorted_result[1])

        # Check that we are in the 10% error range
        expected_count = dim * keep_prob * num_iter
        rel_error = math.fabs(math.fabs(final_count - expected_count) / expected_count)
        self.assertLess(rel_error, 0.1)

        # TEST DROPOUT WITH keep_prob = 1
        drop_dense = Dropout(dense_input, keep_prob=1)
        result = drop_dense.tensor.eval({dense_input.placeholder: data})
        np.testing.assert_array_equal(result, data)

        # TEST FLAT INDEX SPARSE INPUT
        n_active = 2
        data = [list(range(0, n_active, 1))]
        flat_sparse = Input(dim, n_active)
        self.assertTrue(flat_sparse.is_sparse())

        dropout = Dropout(flat_sparse, keep_prob)
        self.assertTrue(dropout.is_sparse())

        result = self.ss.run(dropout.tensor, {flat_sparse.placeholder: data})
        np.testing.assert_allclose(1 / keep_prob, result.values)
        self.assertLessEqual(len(result.values), len(data[0]))

        # test for keep_prob == 1
        dropout = Dropout(flat_sparse, keep_prob=1)
        after_dropout = self.ss.run(dropout.tensor, {flat_sparse.placeholder: data})
        after_input = flat_sparse.tensor.eval({flat_sparse.placeholder: data})
        np.testing.assert_array_equal(after_input.indices, after_dropout.indices)

        # TEST DROPOUT ON SPARSE INPUT
        sparse_data = sparse_tensor_value_one_hot(data, [1, dim])
        sparse_input = SparseInput(dim)
        dropout = Dropout(sparse_input, keep_prob=keep_prob)

        # feed sparse tensor values with indices
        after_dropout = self.ss.run(dropout.tensor, {sparse_input.placeholder: sparse_data})
        np.testing.assert_allclose(1 / keep_prob, after_dropout.values)
        self.assertLessEqual(len(after_dropout.indices), len(sparse_data.indices))

        dropout = Dropout(sparse_input, keep_prob=1)
        before_dropout = self.ss.run(sparse_input.tensor, {sparse_input.placeholder: sparse_data})
        after_dropout = self.ss.run(dropout.tensor, {sparse_input.placeholder: sparse_data})
        np.testing.assert_array_equal(before_dropout.indices, after_dropout.indices)
        np.testing.assert_array_equal(before_dropout.values, after_dropout.values)

    def test_zoneout_layer(self):
        dim = 100
        batch_size = 1000
        keep_prob = 0.5

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        mean_sum = np.mean(np.sum(zoneout.tensor.eval(), axis=-1))
        self.assertAlmostEqual(mean_sum, 0., delta=1.0)

        # test keep_prob = 1
        keep_prob = 1.0

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        mean_sum = np.mean(np.sum(zoneout.tensor.eval(), axis=-1))
        self.assertEqual(mean_sum, dim)

        # test keep_prob = 0
        keep_prob = 0.0

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        mean_sum = np.mean(np.sum(zoneout.tensor.eval(), axis=-1))
        self.assertEqual(mean_sum, -dim)

        # test keep_prob = 0
        keep_prob = np.random.rand()

        current_data = np.full([batch_size, dim], fill_value=1.)
        previous_data = np.full([batch_size, dim], fill_value=-1.)

        current_layer = TensorLayer(tensor=current_data, n_units=dim)
        previous_layer = TensorLayer(tensor=previous_data, n_units=dim)

        zoneout = ZoneOut(current_layer, previous_layer, keep_prob=keep_prob)

        mean_sum = np.mean(np.sum(zoneout.tensor.eval(), axis=-1))
        expected = (2 * dim * keep_prob) - dim

        self.assertAlmostEqual(mean_sum, expected, delta=1.0)

    def test_gaussian_noise(self):
        dim = 1000
        # for sparse inputs
        n_active = 10

        dense_input = Input(dim)
        dense_data = np.ones([1, dim], dtype=np.float32)
        noise_layer = GaussianNoise(dense_input)

        # test that expected average tensor is approximately the same
        result = noise_layer.tensor.eval({dense_input.placeholder: dense_data})
        mean_result = np.mean(result)
        mean_data = np.mean(dense_data)
        self.assertAlmostEqual(mean_data, mean_result, delta=0.1)

        # sparse input with flat indices
        flat_indices = [list(range(0, n_active, 1))]
        flat_input = Input(dim, n_active, dtype=tf.int64)
        noise_layer = GaussianNoise(flat_input)
        result = noise_layer.tensor.eval({flat_input.placeholder: flat_indices})

        dense_input = np.zeros([1, dim])
        dense_input[0, flat_indices[0]] = 1
        mean_data = np.mean(dense_input)
        mean_result = np.mean(result)
        self.assertAlmostEqual(mean_data, mean_result, delta=0.1)

        sparse_input = SparseInput(dim)
        noise_layer = GaussianNoise(sparse_input)
        sparse_data = sparse_tensor_value_one_hot(flat_indices, [1, dim])
        result = noise_layer.tensor.eval({sparse_input.placeholder: sparse_data})
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
        result = noise_layer.tensor.eval({dense_input.placeholder: dense_data})
        mean_result = np.mean(result)
        self.assertEqual(mean_result, 0)

    def test_activation_with_params(self):
        inputs = Input(1)
        act = Activation(inputs, leaky_relu, alpha=0.)

        r0 = act.tensor.eval({inputs.tensor: [[-1]]})
        r1 = act.tensor.eval({inputs.tensor: [[1]]})
        r2 = act.tensor.eval({inputs.tensor: [[3]]})

        self.assertEqual(r0[0], 0)
        self.assertEqual(r1[0], 1)
        self.assertEqual(r2[0], 3)

    def test_layers_to_list(self):
        """ layers_to_list returns the layers without repetition using a breadth first search from the last layer
        and then reversing the layers found.
        """
        l11 = Input(1, name="in1")
        l12 = Input(1, name="in2")
        l121 = WrapLayer(l12, l12.n_units, tf_fn=lambda x: tf.identity(x))
        l2 = Add(l11, l121)

        l3 = Linear(l2, 1)
        l4 = Add(l3, l12)

        l41 = Activation(l4, fn=sigmoid, name="act1")
        l42 = Activation(l4, fn=hard_sigmoid, name="act2")

        l5 = ToSparse(l41)

        outs = [l5, l42]
        layers = layers_to_list(outs, l3)
        # for layer in layers:
        #    print(layer)

        self.assertEqual(len(layers), 6)
        self.assertEqual(layers[0], l3)
        self.assertEqual(layers[-1], l5)
        self.assertIn(l12, layers)
        self.assertNotIn(l2, layers)
        self.assertNotIn(l11, layers)

    def test_wrap_layer(self):
        data = np.random.uniform(-1, 1, [1, 4])

        input_layer = Input(4)
        wrap_layer = WrapLayer(input_layer, 4, lambda layer: tf.identity(layer))
        self.assertIs(input_layer.placeholder, wrap_layer.placeholder)

        with tf.Session() as sess:
            t1 = sess.run(input_layer.tensor, feed_dict={input_layer.placeholder: data})
            t2 = sess.run(wrap_layer.tensor, feed_dict={wrap_layer.placeholder: data})

            np.testing.assert_array_almost_equal(t1, data, decimal=6)
            np.testing.assert_array_almost_equal(t1, t2, decimal=6)

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

        var_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(var_init)

            v1 = sess.run(lookup.tensor, {inputs.placeholder: input_data})
            v2 = sess.run(lookup_from_tensor.tensor)

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

        lookup = Lookup(tensor_input, seq_size, lookup_shape=[input_dim, embed_dim], batch_size=batch_size,
                        batch_padding=False)

        lookup_padding = Lookup(tensor_input, seq_size, lookup_shape=[input_dim, embed_dim], batch_size=batch_size,
                                batch_padding=True)

        lookup_1d = Lookup(tensor_input_1d, seq_size, lookup_shape=[input_dim, embed_dim], batch_size=batch_size,
                           batch_padding=True)

        var_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(var_init)

            result = sess.run(lookup.tensor)
            result_padding = sess.run(lookup_padding.tensor)
            result_1d = sess.run(lookup_1d.tensor)

            self.assertEqual(np.shape(result), (2, seq_size, embed_dim))
            self.assertEqual(np.shape(result_padding), (batch_size, seq_size, embed_dim))
            self.assertEqual(np.shape(result_1d), (batch_size, seq_size, embed_dim))

    def test_lookup_sequence_bias(self):
        vocab_size = 4
        n_features = 3
        seq_size = 2

        inputs = Input(seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2], [0, 2]])
        lookup = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features], bias=True)

        var_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(var_init)

            v1 = sess.run(lookup.tensor, {inputs.placeholder: input_data})

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

        var_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(var_init)

            v1 = sess.run(lookup.tensor, {inputs.placeholder: input_data})
            v2 = sess.run(concat_lookup.tensor, {inputs.placeholder: input_data})
            v3 = sess.run(seq_lookup.tensor, {inputs.placeholder: input_data})

            self.assertEqual(np.shape(v1), (np.shape(input_data)[0], seq_size, embed_dim))
            self.assertEqual(np.shape(v2), (np.shape(input_data)[0], seq_size * embed_dim))

            self.assertEqual(np.shape(v3), (seq_size, np.shape(input_data)[0], embed_dim))
            self.assertTrue(np.array_equal(v1[:, 0], v3[0]))

    def test_gating(self):
        self.reset()

        vocab_size = 4
        n_features = 3
        seq_size = 2

        inputs = Input(seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2]])

        features = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features]).as_concat()
        sp_features = ToSparse(features)

        gate_w = Linear(features, seq_size)
        gate1 = Gate(features, gate_w)
        gate2 = gate1.reuse_with(sp_features)

        init = tf.global_variables_initializer()
        init.run()

        feed = {inputs.placeholder: input_data}

        r1 = gate1.tensor.eval(feed)
        r2 = gate2.tensor.eval(feed)

        self.assertTrue(np.array_equal(r1, r2))

    def test_coupled_gate(self):
        self.reset()

        vocab_size = 4
        n_features = 3
        seq_size = 2
        batch_size = 4

        inputs = Input(seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2]])

        features1 = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features]).as_concat()
        features2 = Lookup(inputs, seq_size, lookup_shape=[vocab_size, n_features]).as_concat()

        sp_features1 = ToSparse(features1)

        gate_w = Linear(features1, seq_size)
        coupled_gate = CoupledGate(features1, features2, gate_w)

        coupled_gate2 = coupled_gate.reuse_with(sp_features1, features2)

        init = tf.global_variables_initializer()
        init.run()

        feed = {inputs.placeholder: input_data}

        r1 = coupled_gate.tensor.eval(feed)
        r2 = coupled_gate2.tensor.eval(feed)

        self.assertTrue(np.array_equal(r1, r2))

    def test_rnn_cell(self):
        self.reset()

        n_inputs = 4
        n_hidden = 2
        batch_size = 2

        inputs = Input(n_inputs)
        rnn_1 = RNNCell(inputs, n_hidden)
        rnn_2 = rnn_1.reuse_with(inputs, rnn_1)

        rnn_3 = rnn_1.reuse_with(inputs)

        tf.global_variables_initializer().run()

        data = np.ones([batch_size, 4])

        res1 = rnn_1.tensor.eval({inputs.placeholder: data})
        res2 = rnn_2.tensor.eval({inputs.placeholder: data})
        res3 = rnn_3.tensor.eval({inputs.placeholder: data})

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertTrue(np.array_equal(res1, res3))
        self.assertFalse(np.array_equal(res1, res2))

        m = Model(inputs, rnn_2)
        # r = ModelRunner(m)
        # r.log_graph("/tmp")

    def test_lstm_cell(self):
        self.reset()

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

        tf.global_variables_initializer().run()

        data = np.ones([batch_size, 4])

        res1 = rnn_1.tensor.eval({inputs.placeholder: data})
        res2 = rnn_2.tensor.eval({inputs.placeholder: data})
        res3 = rnn_3.tensor.eval({inputs.placeholder: data})

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertTrue(np.array_equal(res1, res3))
        self.assertFalse(np.array_equal(res1, res2))

        m = Model(inputs, rnn_2)
        # r = ModelRunner(m)
        # r.log_graph("/tmp")

    def test_gru_cell(self):
        self.reset()

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

        tf.global_variables_initializer().run()

        data = np.ones([batch_size, 4])

        res1 = rnn_1.tensor.eval({inputs.placeholder: data})
        res2 = rnn_2.tensor.eval({inputs.placeholder: data})
        res3 = rnn_3.tensor.eval({inputs.placeholder: data})

        self.assertEqual((batch_size, n_hidden), np.shape(res1))
        self.assertTrue(np.array_equal(res1, res3))
        self.assertFalse(np.array_equal(res1, res2))

        m = Model(inputs, rnn_2)
        r = ModelRunner(m)
        r.log_graph("/tmp")

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

        tf.global_variables_initializer().run()

        feed = {l1.placeholder: [[1]], l2.placeholder: [[1]]}
        res1 = m.tensor.eval(feed)
        res2 = m2.tensor.eval()

        # model = Model(m2.input_layers, m2)
        # runner = ModelRunner(model)
        # runner.log_graph("/tmp")

    def test_module_gate(self):
        l1 = Input(4, name="in1")
        l2 = Input(2, name="in2")

        gate = Gate(layer=l1, gate_input=l2)
        gate_module = Module([l1, l2], gate)

        model = Model(run_in_layers=gate_module.input_layers, run_out_layers=gate_module)
        runner = ModelRunner(model)
        runner.log_graph("/tmp")

        t1 = TensorLayer([[1, 1, 1, 1]], n_units=4, dtype=tf.float32)
        t2 = TensorLayer([[1, 1]], n_units=2, dtype=tf.float32)

        with tf.name_scope("module_reuse"):
            m2 = gate_module.reuse_with(t1, t2)

        model = Model(m2.input_layers, m2)
        runner = ModelRunner(model)
        runner.log_graph("/tmp/")


if __name__ == '__main__':
    unittest.main()
