import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
import tensorflow as tf


def test_gumbel_sample():
    shape = [4, 10]
    logits = tf.random.uniform(shape)
    sample = tx.gumbel_top(logits, 2)
    all_unique = tf.map_fn(lambda x: tf.equal(tf.unique(x)[0].shape[-1], 2), sample, dtype=tf.bool)
    all_unique = tf.reduce_all(all_unique, axis=0)
    assert all_unique
    assert sample.shape == [4, 2]
