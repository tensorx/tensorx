from tensorx import test_utils
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestVarExtend(test_utils.TestCase):
    def test_extend(self):
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

        with self.cached_session(use_gpu=True):
            self.eval(tf.global_variables_initializer())
            shape = tf.shape(var)
            comp_shape = tf.shape(comp)

            self.eval(extend, {new_row: [[1, 1, 1, 1]]})

            self.assertEqual(shape[0], 2)
            self.assertEqual(comp_shape[0], 2)

            var.set_shape(self.eval(tf.shape(var)))


if __name__ == "__main__":
    test_utils.main()
