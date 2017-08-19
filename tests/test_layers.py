from unittest import TestCase
import tensorflow as tf
import numpy as np
from tensorx.layers import Input, SparseInput, Linear, ToSparse, ToDense
from tensorx.transform import index_list_to_sparse


class TestLayers(TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_input(self):
        """ Test Input layer - creates a TensorFlow Placeholder

        this corresponds to an input layer with n_units in the input
        and a shape corresponding to [batch_size, n_units]
        """
        in_layer = Input(n_units=10)
        self.assertIsInstance(in_layer.output, tf.Tensor)

        ones = np.ones(shape=(2, 10))
        result = self.ss.run(in_layer.output, feed_dict={in_layer.output: ones})

        np.testing.assert_array_equal(ones, result)

        ones_wrong_shape = np.ones(shape=(2, 11))
        try:
            self.ss.run(in_layer.output, feed_dict={in_layer.output: ones_wrong_shape})
            self.fail("Should have raised an exception since shapes don't match")
        except ValueError:
            pass

    def test_index_input(self):
        """ Create a Sparse Input by providing
        a n_active parameter
        """
        dim = 10
        index = np.random.randint(0, 10)
        index = [[index]]

        input_layer = Input(n_units=dim, n_active=1, dtype=tf.int64)

        result = self.ss.run(input_layer.output, feed_dict={input_layer.output: index})
        s = np.shape(result)
        self.assertEqual(s[1], 1)

    def test_linear_equal_sparse_dense(self):
        index = 0
        dim = 10

        # x1 = dense input / x2 = sparse input / x3 = sparse input (explicit)
        x1 = Input(n_units=dim)
        x2 = Input(n_units=dim, n_active=1, dtype=tf.int64)
        x3 = SparseInput(10, n_active=1)

        # two layers with shared weights, one uses a sparse input layer, the other the dense one
        y1 = Linear(x1, 4)
        y2 = Linear(x2, 4, weights=y1.weights)
        y3 = Linear(x3, 4, weights=y1.weights)

        self.ss.run(tf.global_variables_initializer())

        # dummy input data
        input1 = np.zeros([1, dim])
        input1[0, index] = 1
        input2 = [[index]]
        input3 = index_list_to_sparse(input2, [1, dim])
        self.assertIsInstance(input3, tf.SparseTensorValue)

        # one evaluation performs a embedding lookup and reduce sum, the other uses a matmul
        y1_output = y1.output.eval({x1.key: input1})
        y2_output = y2.output.eval({x2.key: input2})
        y3_output = y3.output.eval({x3.key[0]: input3})

        # the result should be the same
        np.testing.assert_array_equal(y1_output, y2_output)
        np.testing.assert_array_equal(y2_output, y3_output)

    def test_to_sparse(self):
        index = 0
        dim = 10

        # dense input
        x1 = Input(n_units=dim)
        x2 = Input(n_units=dim, n_active=1, dtype=tf.int64)
        x3 = SparseInput(10, n_active=1)

        # dummy input data
        input1 = np.zeros([1, dim])
        input1[0, index] = 1
        input2 = [[index]]
        input3 = index_list_to_sparse(input2, [1, dim])

        s1 = ToSparse(x1)
        s2 = ToSparse(x2)
        s3 = ToSparse(x3)

        y1_sp_indices, y1_sp_values = self.ss.run(s1.output, {x1.key: input1})

        self.assertEqual(len(y1_sp_indices.values), 1)
        self.assertEqual(len(y1_sp_values.values), 1)
        self.assertEqual(y1_sp_values.values, 1)

        y2_sp_indices, y2_sp_values = self.ss.run(s2.output, {x2.key: input2})
        self.assertEqual(len(y1_sp_indices.values), 1)
        self.assertEqual(len(y1_sp_values.values), 1)
        self.assertEqual(y1_sp_values.values, 1)
        np.testing.assert_array_equal(y1_sp_indices.indices, y2_sp_indices.indices)
        np.testing.assert_array_equal(y1_sp_indices.values, y2_sp_indices.values)

        y3_sp_indices, y3_sp_values = self.ss.run(s3.output, {x3.key[0]: input3})
        self.assertEqual(len(y3_sp_indices.values), 1)
        self.assertEqual(len(y3_sp_values.values), 1)
        self.assertEqual(y3_sp_values.values, 1)
        np.testing.assert_array_equal(y1_sp_indices.indices, y3_sp_indices.indices)
        np.testing.assert_array_equal(y1_sp_indices.values, y3_sp_indices.values)

    def test_to_dense(self):
        dim = 10
        n_active = 1
        index = 0

        x1 = Input(n_units=dim, n_active=n_active, dtype=tf.int64)
        x2 = SparseInput(10, n_active=1)

        data1 = [[index]]
        data2 = index_list_to_sparse(data1, [1, dim])

        expected = np.zeros([1, dim])
        expected[0, index] = 1

        to_dense1 = ToDense(x1)
        to_dense2 = ToDense(x2)

        result1 = to_dense1.output.eval({x1.key: data1})
        result2 = to_dense2.output.eval({x2.key[0]: data2})

        np.testing.assert_array_equal(expected, result1)
        np.testing.assert_array_equal(expected, result2)
