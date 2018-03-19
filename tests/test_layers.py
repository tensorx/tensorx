import unittest
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from tensorx.layers import *
from tensorx.layers import Lookup
from tensorx.init import *

from tensorx.layers import layers_to_list
from tensorx.activation import *
from tensorx.transform import sparse_tensor_value_one_hot
import math


class TestLayers(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_compose(self):
        in1 = Input(1)
        in2 = TensorLayer([[1.]], 1)

        l1 = Linear(in1, 4)
        l2 = Activation(l1, relu)

        comp = Compose([l1, l2])
        comp2 = comp.reuse_with(in2)

        tf.global_variables_initializer().run()

        """
        for layer in layers_to_list(l2):
            print(layer.full_str())

        print("=" * 10)
        for layer in layers_to_list(comp):
            print(layer.full_str())

        print("=" * 10)
        for layer in layers_to_list(comp2):
            print(layer.full_str())
        """

        res1 = l2.tensor.eval({in1.placeholder: [[1.]]})
        res2 = comp.tensor.eval({in1.placeholder: [[1.]]})

        res3 = comp2.tensor.eval()

        self.assertTrue(np.array_equal(res1, res2))
        self.assertTrue(np.array_equal(res1, res3))

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
        tf.reset_default_graph()

        inputs = TensorLayer([[1]], 1, batch_size=1)
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
        np.testing.assert_array_equal(after_input.indices, after_input.indices)

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
        l11 = Input(1)
        l12 = Input(1)
        l2 = Add([l11, l12])

        l3 = Linear(l2, 1)

        l41 = Activation(l3, fn=sigmoid)
        l42 = Activation(l3, fn=hard_sigmoid)

        l5 = ToSparse(l41)

        outs = [l5, l42]
        layers = layers_to_list(outs)

        self.assertEqual(len(layers), 7)
        self.assertEqual(layers[0], l11)
        self.assertEqual(layers[1], l12)
        self.assertEqual(layers[2], l2)
        self.assertEqual(layers[3], l3)
        self.assertEqual(layers[4], l41)
        self.assertEqual(layers[5], l42)
        self.assertEqual(layers[6], l5)

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

    def test_lookup_layer(self):
        vocab_size = 4
        n_features = 3
        seq_size = 2
        batch_size = 4

        inputs = Input(seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2]])

        # DENSE LOOKUP
        lookup = Lookup(inputs, seq_size, [vocab_size, n_features], batch_size)

        sp_inputs = SparseInput(n_features, dtype=tf.int64)

        # INPUT DATA
        sp_indices = np.array([[0, 2], [1, 0], [2, 1], [3, 2]], np.int64)
        sp_values = np.array([1, 1, 1, 1], np.int64)
        dense_shape = np.array([4, vocab_size], np.int64)
        sp_values = tf.SparseTensorValue(indices=sp_indices, values=sp_values, dense_shape=dense_shape)

        # SPARSE LOOKUP
        # reuse on sparse
        sp_lookup = lookup.reuse_with(sp_inputs)

        # reuse on dense again
        shared_lookup = sp_lookup.reuse_with(inputs)

        self.assertListEqual(lookup.variable_names, sp_lookup.variable_names)
        self.assertListEqual(lookup.variable_names, shared_lookup.variable_names)

        var_init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(var_init)
            v1 = sess.run(sp_lookup.tensor, {sp_inputs.placeholder: sp_values})
            v2 = sess.run(lookup.tensor, {inputs.placeholder: input_data})
            v3 = sess.run(shared_lookup.tensor, {inputs.placeholder: input_data})

            self.assertTrue(np.array_equal(v1, v2))
            self.assertTrue(np.array_equal(v2, v3))

            # testing dynamic batch sizes
            input_data = np.array([[2, 0], [1, 2], [1, 2]])
            sp_indices = np.array([[0, 2], [1, 0],
                                   [2, 1], [3, 2],
                                   [4, 1], [5, 2]], np.int64)
            sp_values = np.array([1, 1, 1, 1, 1, 1], np.int64)
            sp_values = np.array([1, 1, 1, 1, 1, 1], np.int64)
            dense_shape = np.array([6, vocab_size], np.int64)
            sp_values = tf.SparseTensorValue(indices=sp_indices, values=sp_values, dense_shape=dense_shape)

            v1 = sess.run(lookup.tensor, {inputs.placeholder: input_data})
            v2 = sess.run(sp_lookup.tensor, {sp_inputs.placeholder: sp_values})

            self.assertTrue(np.array_equal(v1, v2))

    def test_lookup_padding(self):
        tf.reset_default_graph()
        self.ss = tf.InteractiveSession()

        ri_size = 4
        vocab_size = 4
        embed_dim = 1
        seq_size = 2
        batch_size = 2

        inputs = TensorLayer(tf.constant([[0, 2]]), 2, dtype=tf.int32)
        sp_inputs = TensorLayer(
            tf.SparseTensorValue(indices=[[0, 0], [1, 2]], values=[1, 1],
                                 dense_shape=[2, ri_size]),
            ri_size)
        sp_inputs2 = Input(n_units=4, n_active=1, dtype=tf.int32)

        lookup_dense = Lookup(inputs, seq_size=seq_size, feature_shape=[vocab_size, embed_dim], batch_size=batch_size)

        lookup_sparse = lookup_dense.reuse_with(sp_inputs)

        lookup_sparse2 = lookup_dense.reuse_with(sp_inputs2)

        tf.global_variables_initializer().run()

        res1 = lookup_sparse.tensor.eval()
        res2 = lookup_dense.tensor.eval()
        res3 = lookup_sparse2.tensor.eval({sp_inputs2.placeholder: [[0], [2]]})

        self.assertTrue(np.array_equal(res1, res2))
        self.assertTrue(np.array_equal(res2, res3))

    def test_gating(self):
        tf.reset_default_graph()
        self.ss = tf.InteractiveSession()

        vocab_size = 4
        n_features = 3
        seq_size = 2
        batch_size = 4
        h_dim = 10

        inputs = Input(seq_size, dtype=tf.int32)
        input_data = np.array([[2, 0], [1, 2]])

        features = Lookup(inputs, seq_size, feature_shape=[vocab_size, n_features])
        features2 = Lookup(inputs, seq_size, feature_shape=[vocab_size, n_features])

        gated = Gate(features, n_gates=seq_size, h_dim=h_dim)

        # gates with custom hidden layer
        hl = Linear(features, h_dim)
        ha = Activation(hl, sigmoid)
        h = Compose([hl, ha])

        gated2 = Gate(features, n_gates=seq_size, gate_input=h)

        gate2_features2 = gated2.reuse_with(features2)

        # back to features 1 but with same params
        gate2_copy = gate2_features2.reuse_with(features)

        init = tf.global_variables_initializer()
        init.run()

        feed = {inputs.placeholder: input_data}

        print(gated.full_str())
        print("inner hidden layer:\n", gated.gate_input.full_str())
        print(tf.shape(gated.tensor).eval(feed))

        print(gated2.full_str())
        # print(h.full_str())
        print(tf.shape(gated2.tensor).eval(feed))

        print(gate2_features2.full_str())
        print("inner hidden layer:\n", gated2.gate_input.full_str())
        # print(h.full_str())
        print(tf.shape(gate2_features2.tensor).eval(feed))

        out1 = gated2.tensor.eval(feed)
        out2 = gate2_features2.tensor.eval(feed)
        out3 = gate2_copy.tensor.eval(feed)

        # they gate different features
        self.assertFalse(np.array_equal(out1, out2))
        self.assertTrue(np.array_equal(out1, out3))


if __name__ == '__main__':
    unittest.main()
