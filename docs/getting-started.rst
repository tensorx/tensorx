Getting Started
###############

The main component in **TensorX** is a ``Layer``. A layer is essentially a function that transforms the output of its
input layer into a TensorFlow tensor. The only layer that does not have an input is the ``Input``. Let's take a simple
linear regression model as an example.

.. code-block:: ruby

    from tensorx.layers import Input, Linear

    inputs = Input(1)
    out = Linear(inputs, 1)

``inputs`` creates a TensorFlow ``placeholder`` for the given number of units. ``out`` is a ``Linear`` layer that takes
the output of the input layer and transforms it with a graph that outputs a linear transformation of the form :math:`y = Wx + b`.
