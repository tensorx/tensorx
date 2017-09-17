import unittest
from tensorx.model import Model
from tensorx.layers import Input, Linear, Activation
from tensorx.activation import relu, sigmoid
import tensorflow as tf

"""TODO 
    - consider how models can be visualised on each layer
    - consider how model can be used to facilitate debugging
    - consider if it is worth it to change the layer api to have a "build_graph" method 
    to create reusable layers that can be cloned and wired afterwards in a model or using
    something similar to the functional API of keras, or using something similar to tensorfold blocks
"""


class MyTestCase(unittest.TestCase):
    def test_model_run(self):
        input_layer = Input(4)
        linear = Linear(input_layer, 2)
        h = Activation(linear, fn=relu)
        logits = Linear(h, 4)
        out = Activation(logits, fn=sigmoid)

        self.assertNotEqual(linear.weights.name, logits.weights.name)

        model = Model([input_layer], [h])
        data = [[1, 1, 1, 1]]

        with tf.Session():
            result = model.run(data)
            print(result)


if __name__ == '__main__':
    unittest.main()
