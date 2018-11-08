from tensorx import test_utils
from tensorx.activation import hard_sigmoid, relu
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestActivations(test_utils.TestCase):
    def test_hard_sigmoid(self):
        with self.cached_session(use_gpu=True):
            value = -2.5
            result = hard_sigmoid(value)
            self.assertEqual(result, 0.)
            value = 2.5
            result = hard_sigmoid(value)
            self.assertEqual(result, 1.)

            value = 1
            result = relu(value)
            self.assertEqual(result.dtype, tf.int32)

    def test_relu(self):
        with self.cached_session(use_gpu=True):
            value = -1
            result = relu(value)
            self.assertEqual(result, 0.)
            self.assertEqual(result.dtype, tf.int32)
            value = 1.
            result = relu(value)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result, 1.)


if __name__ == '__main__':
    test_utils.main()
