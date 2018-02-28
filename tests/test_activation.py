import unittest
from tensorx.activation import hard_sigmoid, relu
import tensorflow as tf
import numpy as np


class TestActivations(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_hard_sigmoid(self):
        value = -2.5
        result = hard_sigmoid(value)
        self.assertEqual(result.eval(), 0.)
        value = 2.5
        result = hard_sigmoid(value)
        self.assertEqual(result.eval(), 1.)

        value = 1
        result = relu(value)
        self.assertEqual(type(result.eval()), np.int32)

    def test_relu(self):
        value = -1
        result = relu(value)
        self.assertEqual(result.eval(), 0.)
        self.assertEqual(type(result.eval()), np.int32)
        value = 1.
        result = relu(value)
        self.assertEqual(type(result.eval()), np.float32)
        self.assertEqual(result.eval(), 1.)


if __name__ == '__main__':
    unittest.main()
