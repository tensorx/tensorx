import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorx.test_utils import TestCase
import unittest
import tensorx as tx
import tensorflow as tf
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.training.tracking import util
import shutil


# checkpoint manager expects trackable

# there are two trackable base classes
# AutoTrackable which overrides setattr to add dependencies between trackable objects
# and Trackable, where dependencies should be added manually


class MyTestCase(TestCase):
    def test_variable_checkpoint(self):
        x = tf.ones([2, 4])
        l1 = tx.Linear(x, 3, add_bias=True, name="l1")
        l2 = tx.Linear(x, 3, add_bias=False, name="l1")

        w1 = l1.weights
        w2 = l2.weights

        self.assertIsNot(w1, w2)

        # print(w1)
        # print(w2)

        track: AutoTrackable = l1.layer_state
        print(util.list_objects(track))

        ckpt = tf.train.Checkpoint(l1=l1)
        manager = tf.train.CheckpointManager(ckpt, './ckpts', max_to_keep=1)
        manager.save(1)
        # manager.save(2)

        l1.weights.assign(l2.weights.value())

        status = ckpt.restore(manager.latest_checkpoint)
        status.assert_existing_objects_matched()

        # print()
        print(tf.train.list_variables(manager.latest_checkpoint))

        shutil.rmtree('./ckpts')


if __name__ == '__main__':
    unittest.main()
