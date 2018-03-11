from unittest import TestCase
import tensorflow as tf


class TestVarExtend(TestCase):
    def test_extend(self):
        ss = tf.InteractiveSession()

        # if validate_shape=True var ends up with shape [1,4] until we call set shape
        var = tf.Variable(tf.zeros([1, 4], tf.float32), validate_shape=False)
        # setting the shape to None,dim, makes so that we don't have to adjust it later
        # we treat the var like a placeholder
        var.set_shape([None, 4])
        comp = tf.multiply(var, 2)

        new_row = tf.placeholder(shape=[1, 4], dtype=tf.float32)

        # if validate_shape is true tries to validate shape of new value against the current shape
        # we want to expand the var so this would fail
        extend = tf.assign(var, tf.concat([var, new_row], axis=0), validate_shape=False)

        ss.run(tf.global_variables_initializer())
        shape1 = tf.shape(var).eval()
        comp_shape1 = tf.shape(comp).eval()
        print(shape1)
        print(var.get_shape())
        print(comp.get_shape())

        extend.eval({new_row: [[1, 1, 1, 1]]})

        shape2 = tf.shape(var).eval()
        comp_shape2 = tf.shape(comp).eval()

        print(shape2)
        print(var.get_shape())
        var.set_shape(tf.shape(var).eval())
