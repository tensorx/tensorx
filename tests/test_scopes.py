from unittest import TestCase
import tensorx as tx
import tensorflow as tf
import numpy as np


class TestScopes(TestCase):
    """
    Issues with Layer Scopes:
        Reuse layer: calling reuse will try to use the name of the previous layer
        which if defined inside a given scope will be altered to scope/name
        if we call reuse inside another scope, the new layer will try to use the name scope/name
        and the result will be scope2/scope/name, if we specify name in the reuse, this is mitigated
        because the name of the original layer will be be used

    Possible Solutions:
        Distinguish between given layer names and scope names, when the layer is created it should be
        created with a given name, this name is altered depending on the outter scope, we could
        preserve the original name (which will be used as a basis for a scope anyway. We would need to
        maintain two values, name, and scope_name, name is the given name, scope_name is the complete
        name including the current scope. Basically this is done within layer_scope. Instead of altering
        the name it could just add a scope_name and save the unique name without the current scope.


    Scope on Compose Layers:
        Compose layers do not impose a scope on the inner layers, instead they serve as containers
        that forward certain properties and call reuse on an entire block
    """

    def test_nested_layers(self):
        data = np.random.uniform(-1., 1., size=[1, 4])

        with tf.name_scope("scope1"):
            inputs = tx.TensorLayer(tf.constant(data), 4)
            h1 = tx.Linear(inputs, 3)
            h2 = tx.Activation(h1, tx.elu)
            h = tx.Compose([h1, h2])

            layer = tx.Gate(inputs, 4, gate_input=h)

        with tf.name_scope("scope2"):
            h = h.reuse_with(inputs)

            layer2 = layer.reuse_with(tx.TensorLayer(tf.constant(data), 4), name="gate",gate_input=h)
            layer3 = tx.Gate(inputs, 4, gate_input=h)

        model = tx.Model(run_in_layers=inputs,run_out_layers=layer, train_out_layers=[layer2,layer3])
        runner = tx.ModelRunner(model)
        runner.log_graph(save_dir="/tmp/")

        print(layer.full_str())
        print(layer.gate_input.full_str())

        print(layer2.full_str())
        print(layer2.gate_input.full_str())

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
