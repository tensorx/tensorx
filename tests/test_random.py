import unittest
import tensorflow as tf
from tensorx import random as tx_rand


class TestRandom(unittest.TestCase):
    def test_sample(self):
        ss = tf.Session()

        range_max = 10
        num_sampled = 2
        batch_size = 2

        sample = tx_rand.sample(range_max, [num_sampled], unique=True)

        y, _ = tf.unique(sample)
        unique_set = ss.run(y)
        self.assertEqual(len(unique_set), num_sampled)

        samples = tx_rand.sample(range_max, [batch_size, num_sampled], unique=True)
        for i in range(batch_size):
            y, _ = tf.unique(tf.squeeze(tf.gather(samples, [i])))
            unique_set = ss.run(y)
            self.assertEqual(len(unique_set), num_sampled)

    def test_sample_range_max(self):
        ss = tf.Session()
        range_max = 10
        num_sampled = 11

        sample = tx_rand.sample(range_max, [num_sampled], unique=True)
        self.assertRaises(Exception, ss.run, sample)

        try:
            sample = tx_rand.sample(range_max, [num_sampled], unique=False)
            ss.run(sample)
        except Exception:
            self.fail("should have not raised an exception since the number of samples > range_max but unique == False")


if __name__ == '__main__':
    unittest.main()
