from unittest import TestCase
import tensorflow as tf
import numpy as np
from tensorx.layers import Input
from tensorx.layers import Activation as Act


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
        self.assertIsInstance(in_layer.tensor, tf.Tensor)

        with tf.Session() as sess:
            ones = np.ones(shape=(2, 10))

            result = sess.run(in_layer.tensor, feed_dict={in_layer.tensor: ones})
            np.testing.assert_array_equal(ones, result)

            ones_wrong_shape = np.ones(shape=(2, 11))
            try:
                sess.run(in_layer.tensor, feed_dict={in_layer.tensor: ones_wrong_shape})
                self.fail("Input can only receive a tensor with the shape: ", in_layer.shape)
            except ValueError:
                pass

    def test_index_input(self):
        """ Test IndexInput layer - creates TensorFlow int placeholder

        keeps track of original shape so it can be linked with other layers
        """
        dim = 10
        index = np.random.randint(0, 10)
        index = [[index]]

        input_layer = IndexInput(n_units=dim, n_active=1)

        with tf.Session() as ss:
            result = ss.run(input_layer.tensor, feed_dict={input_layer.tensor: index})

            s = np.shape(result)
            self.assertEqual(s[1], 1)

    def test_activation(self):
        pass
        # Act.Fn.sigmoid


        # fn = Act(None,sg)
