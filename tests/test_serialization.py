import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def test_variable_checkpoint(tmp_path):
    inputs = tx.Constant(tf.ones([2, 4]))
    l1 = tx.Linear(inputs, 3, add_bias=True, name="l1")
    l2 = tx.Linear(inputs, 3, add_bias=False, name="l1")

    # track: AutoTrackable = l1.layer_state

    checkpoint = tf.train.Checkpoint(l1=l1)
    manager = tf.train.CheckpointManager(checkpoint, tmp_path / 'ckpts',
                                         max_to_keep=1)
    manager.save(1)
    # manager.save(2)

    l1.weights.assign(l2.weights.value())

    status = checkpoint.restore(manager.latest_checkpoint)
    status.assert_existing_objects_matched()

    checkpoint_vars = tf.train.list_variables(manager.latest_checkpoint)
    assert len(checkpoint_vars) == 4
    assert checkpoint_vars[0][0] == '_CHECKPOINTABLE_OBJECT_GRAPH'
    assert "l1/bias" in checkpoint_vars[1][0]
    assert "l1/weights" in checkpoint_vars[2][0]
    assert "save_counter" in checkpoint_vars[3][0]
