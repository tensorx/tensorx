from unittest import TestCase
import tensorflow as tf

class TestVarExtend(TestCase):
    def test_extend(self):
        ss = tf.InteractiveSession()

        var = tf.Variable(tf.zeros([1, 4], tf.float32), validate_shape=False, expected_shape=[None, 4])
        comp = tf.multiply(var,2)

        new_row = tf.placeholder(shape=[1,4],dtype=tf.float32)
        extend = tf.assign(var, tf.concat([var,new_row], axis=0), validate_shape=False)

        ss.run(tf.global_variables_initializer())
        shape1 = tf.shape(var).eval()
        comp_shape1 = tf.shape(comp).eval()
        print(shape1)

        extend.eval({new_row:[[1,1,1,1]]})

        shape2 = tf.shape(var).eval()
        comp_shape2 = tf.shape(comp).eval()

        print(shape2)



