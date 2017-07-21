import unittest
import tensorflow as tf
from tensorx import random as tx_rand

class TestRandom(unittest.TestCase):
    def test_sample(self):
        ss = tf.Session()

        range_max = 10
        num_sampled = 2

        sample = tx_rand.sample(range_max,[4],unique=True)
        print(ss.run(sample))

        #y, _ = tf.unique(sample)
        #unique_set = ss.run(y)
        #self.assertEqual(len(unique_set),num_sampled)


if __name__ == '__main__':
    unittest.main()
