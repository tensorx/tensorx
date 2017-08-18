from unittest import TestCase
import tensorflow as tf
import numpy as np
from tensorx.layers import Input, Linear


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
        self.assertIsInstance(in_layer.y, tf.Tensor)

        ones = np.ones(shape=(2, 10))
        result = self.ss.run(in_layer.y, feed_dict={in_layer.y: ones})

        np.testing.assert_array_equal(ones, result)

        ones_wrong_shape = np.ones(shape=(2, 11))
        try:
            self.ss.run(in_layer.y, feed_dict={in_layer.y: ones_wrong_shape})
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

        result = self.ss.run(input_layer.y, feed_dict={input_layer.y: index})
        s = np.shape(result)
        self.assertEqual(s[1], 1)

    def test_linear_equal_sparse_dense(self):
        index = 0
        dim = 10

        # x1 = sparse input / x2 = dense input
        x1 = Input(n_units=dim, n_active=1, dtype=tf.int64,name="x1")
        x2 = Input(n_units=dim,name="x2")

        # two layers with shared weights, one uses a sparse input layer, the other the dense one
        y1 = Linear(x1, 4)
        y2 = Linear(x2, 4, weights=y1.weights)

        self.ss.run(tf.global_variables_initializer())

        input1 = [[index]]
        input2 = np.zeros([1, dim])
        input2[0, index] = 1

        # one evaluation performs a embedding lookup and reduce sum, the other uses a matmul
        y1_output = y1.y.eval({x1.y: input1})
        y2_output = y2.y.eval({x2.y: input2})

        # the result should be the same
        np.testing.assert_array_equal(y1_output, y2_output)
