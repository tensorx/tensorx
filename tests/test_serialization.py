import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import tensorx as tx
import tensorflow as tf
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking.tracking import AutoTrackable
from tensorflow.python.training.tracking import util
import shutil
from tensorflow.python.training.tracking.tracking import AutoTrackable


# checkpoint manager expects trackable

# there are two trackable base classes
# AutoTrackable which overrides setattr to add dependencies between trackable objects
# and Trackable, where dependencies should be added manually

def test_layer_save(tmp_path):
    class CustomLayer(tf.Module):

        def __init__(self):
            super(CustomLayer, self).__init__()
            self.v = tf.Variable(2.)

        def __call__(self, x):
            return x * self.v

        @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
        def mutate(self, new_v):
            self.v.assign(new_v)

        def functions_to_serialize(self, serialization_cache):
            """

            Args:
                serialization_cache:

            Returns:
                A dictionary mapping attribute names to `Function` or
                `ConcreteFunction`.

            """
            signatures = [tf.TensorSpec([], tf.float32)]
            concrete_function = tf.function(input_signature=signatures)(self.__call__).get_concrete_function()
            return dict({"__call__": concrete_function})

        def _list_functions_for_serialization(self, serialization_cache):
            print("\nsuck my D")
            fns = self.functions_to_serialize(serialization_cache)

            # AutoTrackable class saves tf.functions
            # add these functions to the dict.
            fns.update(
                AutoTrackable._list_functions_for_serialization(  # pylint:disable=protected-access
                    self, serialization_cache))
            return fns

    custom = CustomLayer()
    custom(tf.constant(0.))

    tmp_path.joinpath("custom")
    save_path = str(tmp_path)

    tf.saved_model.save(custom, save_path)

    loaded = tf.saved_model.load(save_path)
    assert loaded(tf.constant(2.)) == 4.
    loaded.mutate(3.)
    assert loaded(tf.constant(2.)) == 6.
    # loaded object has the variables
    assert loaded.v.value() == tf.constant(3.)

    # the graph imported is a copy of the current one
    assert not custom.v.value() == tf.constant(3.)
    assert custom.v.value() == tf.constant(2.)

    loaded = tf.saved_model.load(save_path)
    assert loaded(tf.constant(2.)) == 4.


def test_keras_signatures(tmp_path):
    tmp_path = tmp_path.joinpath("keras_dense")
    save_path = str(tmp_path)

    data = tf.ones([2, 2])
    x = tf.keras.layers.Input(shape=(2,))
    y = tf.keras.layers.Dense(units=4, input_dim=2)(x)

    model = tf.keras.models.Model(x, y)
    print(type(model))
    # call comes from base layer, how is the fucking model callable then
    print(type(model.__call__))
    tf.saved_model.save(model, save_path)

    loaded = tf.saved_model.load(save_path)
    print(loaded.__dict__)
    # print(loaded.__call__)
    # print(loaded.__call__.concrete_functions)
    # print(loaded.__call__.__dict__)
    assert tx.tensor_equal(loaded(data), model(data))
    # print(model.__call__)


def test_constant_save(tmp_path):
    tmp_path = tmp_path.joinpath("custom")
    save_path = str(tmp_path)

    l1 = tx.Constant(tf.constant(42.))

    assert l1() == 42.
    tf.saved_model.save(l1, save_path)

    loaded = tf.saved_model.load(save_path)
    assert loaded() == 42.


def test_input_save(tmp_path):
    tmp_path = tmp_path.joinpath("input")
    save_path = str(tmp_path)

    inputs = tx.Input(init_value=tf.constant(2.0), constant=False, name="x")
    assert inputs() == 2.
    tf.saved_model.save(inputs, save_path)
    # inputs.value = 3.
    inputs.value = 3.
    assert inputs() == 3.
    loaded = tf.saved_model.load(save_path)
    assert loaded() == 2.
    # TODO should I just use value as a variable for the input state?
    #  then we could no longer do inputs.value = something
    hasattr(loaded, "slot")
    loaded.slot.assign(3.)
    assert loaded() == 3.


def test_linear_save(tmp_path):
    tmp_path = tmp_path.joinpath("linear")
    save_path = str(tmp_path)

    x = tx.Input(init_value=tf.ones([2, 2]), n_units=2)
    linear = tx.Linear(x, n_units=4)
    module = tx.Module(x, linear)

    # I could build the graph for linear and export the function from
    # the graph
    tf.saved_model.save(module, save_path)
    loaded = tf.saved_model.load(save_path)

    assert tx.tensor_equal(loaded(), linear())
    assert tx.tensor_equal(loaded(), module())

    # TODO problem with linear tries to convert Layer
    #  while module builds a graph that works as a layer
    # tf.saved_model.save(linear, save_path)
    # loaded = tf.saved_model.load(save_path)
    #
    # assert tx.tensor_equal(loaded(), linear())
    # assert tx.tensor_equal(loaded(), module())


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
