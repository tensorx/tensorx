""" Scopes were changed in Tensorflow 2.0
The idea is that variables are not managed in an object-oriented fashion, and garbage collected if lost
The new scopes are simply pushing a prefix when creating new tensorflow ops:::

def my_op(a, b, c, name=None):
  with tf.name_scope("MyOp") as scope:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    c = tf.convert_to_tensor(c, name="c")
    # Define some computation that uses `a`, `b`, and `c`.
    return foo_op(..., name=scope)


"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import tensorflow as tf
from tensorx.layers import layer_scope
import tensorx as tx


class TestScopes(unittest.TestCase):

    def test_layer_scope(self):
        name = "test_name"
        with tf.name_scope(name) as scope:
            x1 = tx.Input(tf.ones([2, 2]), name="x")
            x2 = tx.Input(tf.ones([2, 2]), name="x")
            print(x1.slot.name)
            print(x2.slot.name)
            print(x1.scoped_name)
            print(x2.name)
            name1 = scope

        with tf.name_scope(name) as scope:
            name2 = scope

        # unique names are only present when not executing eagerly
        self.assertEqual(name1, name2)

    def test_name_scope(self):
        with tf.name_scope("testing_scope") as scope:
            v1 = tf.Variable(1, True, name="test_var")

        with tf.name_scope("testing_scope"):
            v2 = tf.Variable(2, True, name="test_var")
            v2.assign(3)

        # in tf2 variables with the same name will reference different objects
        self.assertEqual(v1.name, v2.name)
        self.assertNotEqual(v1, v2)
        self.assertNotEqual(v1.value(), v2.value())


if __name__ == '__main__':
    unittest.main()
