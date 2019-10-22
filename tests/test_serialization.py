import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorx.test_utils import TestCase
import unittest
import tensorx as tx
import tensorflow as tf
from tensorflow.python.util import object_identity

# checkpoint manager expexts trackable
from tensorflow.python.training.tracking import base as trackable


# there are two trackable base classes
# AutoTrackable which ovewrides setattr to add dependencies between trackable objects
# and Trackable, where dependencies should be added manually

class MyTestCase(TestCase):
    def test_variable_storage(self):
        x = tf.ones([2, 4])
        l1 = tx.Linear(x, 3, add_bias=True, name="l1")
        l2 = tx.Linear(x, 3, add_bias=False, name="l1")

        w1 = l1.weights
        w2 = l2.weights

        self.assertIsNot(w1, w2)

        print(w1)
        print(w2)

        ckpt = tf.train.Checkpoint(l1=l1)
        manager = tf.train.CheckpointManager(ckpt, './ckpts', max_to_keep=1)
        manager.save(1)
        # manager.save(2)

        l1.weights.assign(l2.weights.value())
        print(w1)
        print(w2)

        status = ckpt.restore(manager.latest_checkpoint)

        print(w1)
        print(w2)

        #print(status.assert_existing_objects_matched())
        print(tf.train.list_variables(manager.latest_checkpoint))

        # d = dict({w1: 0, w2: 1})


if __name__ == '__main__':
    unittest.main()
