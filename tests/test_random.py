import unittest
import tensorflow as tf
from tensorx import random as tr


class TestRandom(unittest.TestCase):
    # setup and close TensorFlow sessions before and after the tests (so we can use tensor.eval())
    def setUp(self):
        self.ss = tf.InteractiveSession()

    def tearDown(self):
        self.ss.close()

    def test_sample(self):
        ss = tf.Session()

        range_max = 10
        num_sampled = 2
        batch_size = 2

        samples = tr.sample(range_max, [num_sampled], unique=True)

        y, _ = tf.unique(samples)
        unique_set = y.eval()
        self.assertEqual(len(unique_set), num_sampled)

        samples = tr.sample(range_max, [batch_size, num_sampled], unique=True)
        for i in range(batch_size):
            y, _ = tf.unique(tf.squeeze(tf.gather(samples, [i])))
            unique_set = y.eval()
            self.assertEqual(len(unique_set), num_sampled)

    def test_sample_range_max(self):
        ss = tf.Session()
        range_max = 10
        num_sampled = 11


        try:
            sample = tr.sample(range_max, [num_sampled], unique=True)
            sample.eval()
        except:
            pass

        try:
            sample = tr.sample(range_max, [num_sampled], unique=False)
            sample.eval()
        except Exception:
            self.fail("should have not raised an exception since the number of samples > range_max but unique == False")


    def test_salt_pepper_noise(self):

        some_tensor = tf.constant(0.5,shape=[4,4])
        indices,values = tr.salt_pepper_noise([4,4],dtype=tf.float32)

        print(indices.eval())



if __name__ == '__main__':
    unittest.main()
