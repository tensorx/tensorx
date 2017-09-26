import unittest
import tensorflow as tf
from tensorx.metrics import cosine_distance
import functools


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_stage_implementation(self):
        n_inputs = 2
        n_partitions = 3
        n_hidden = 1
        batch_size = 1

        feature_shape = [n_partitions, n_inputs, n_hidden]

        inputs = tf.placeholder(tf.float32, [None, n_inputs], "x")
        som_w = tf.get_variable("p", [n_partitions, n_inputs], tf.float32, tf.random_uniform_initializer(-1., 1.))
        feature_w = tf.get_variable("w", feature_shape, tf.float32, tf.random_uniform_initializer(-1., 1.))

        # bmu = tf.reduce_sum(tf.sqrt(som_w-inputs),axis=1)
        # bmu = tf.reduce_sum(tf.sqrt(som_w-inputs),axis=1)
        # bmu_dist = tf.sqrt(tf.reduce_sum(tf.pow((som_w - inputs), 2), 1))
        # som_distances = tf.sqrt(tf.reduce_sum(tf.pow((som_w - inputs), 2), 1))

        # som_dist = cosine_distance(inputs,som_w)
        # bmu = tf.argmin(som_dist, axis=0)
        # bmu_slice = feature_w[bmu]

        # dummy dense example
        indices = tf.constant([0, 2])
        feature_w_slice = tf.gather(feature_w, indices=indices)

        # ext_inputs = tf.expand_dims(inputs,0)
        # features = tf.squeeze(feature_w_slice, [-1])
        features = feature_w_slice

        h = tf.tensordot(inputs, features, axes=[[0, 1], [1, 2]])

        def neighbourhood(x,weights,n_neurons,bmu,p):
            neurons = tf.range(0,n_neurons)
            # TODO 


        # didn't know but calling global variable int must be done after adding the vars to the graph
        init_vars = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_vars)

            feed_dict = {inputs: [[1., -1.]]}

            print("som vars:\n", som_w.eval())
            print("feature vars:\n", feature_w.eval())
            # print("dist: \n", som_dist.eval(feed_dict))
            # print(bmu.eval(feed_dict))

            # print("slices:\n", feature_w_slice.eval(feed_dict))
            print("features:\n", features.eval(feed_dict))
            # hidden
            print("result:\n", h.eval(feed_dict))

            print("result iterative")
            for i in range(2):
                features = feature_w_slice[i]
                op = tf.matmul(inputs, features)
                print(op.eval(feed_dict))
                # print("h:\n",h.eval(feed_dict))
                # print("slice: {}:{}".format(slice_begin.eval(feed_dict), slice_end.eval(feed_dict)))
                # print(feature_w_slice.eval(feed_dict))


if __name__ == '__main__':
    unittest.main()
