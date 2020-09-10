# TensorX Tutorial
This tutorial introduces basic TensorX concepts.

## Prerequisites

TensorX is a machine learning library to build neural network models written in Python and it works as a complement to 
[Tensorflow](https://tensorflow.org), therefore, to make the most out of this tutorial (and this library), 
readers should be familiarized with the following:

* **Python 3**: if you're new to the Python language or need to refresh some concepts check out the
[Python Tutorial](https://docs.python.org/3/tutorial/).
* **Tensorflow**: [Tensorflow](https://tensorflow.org) is a high-performance machine learning library that allows 
numerical computation on GPUs and [TPUs](https://en.wikipedia.org/wiki/Tensor_processing_unit). It was originally 
designed to build auto-differentiable dataflow graphs. Although it has adopted an eager execution model in version 2, 
[computation graphs](https://www.tensorflow.org/guide/function) are still an integral concept of high performance 
tensorflow program definition and deployment. Unfortunately most tensorflow tutorials and guides, focus on Keras, a high
level interface similar to TensorX. For a primer on Tensorflow, I recommend taking a look at 
[Tensorflow basics](https://www.tensorflow.org/guide/eager) guide section instead.

* **NumPy**: similarly to the core of Tensorflow, [Numpy](https://numpy.org) is a numerical computation library with 
a focus in multi-dimensional array transformations. Given that Tensorflow tensors are converted to and from NumPy 
arrays, and TensorX also depends on NumPy (mostly for testing), it is recommended that the reader is familiarized with  
[NumPy basics](https://numpy.org/doc/stable/user/quickstart.html). For more details, check the 
[NumPy documentation](https://numpy.org/doc/stable/).

## Installation 
You can install `tensorx` with `pip` as follows:

```shell
pip install tensorflow
pip install tensorx
```

for more details see the [installation documentation](https://tensorx.org/start/install/).

## Layers
In TensorX, a `Layer` is the basic building block of a neural network. A layer performs computations on its inputs and
can have some state. An `Input` layer is a layer without inputs. 

### Common Layer properties and methods

* `input_layers`: `list` of `Layer` inputs for the current layer.
* `n_units`: number of output units or neurons, this is the last dimension of the output tensor resulting from this 
layer's computation.
* `compute(*input_tensors)`: applies the layer transformation to a list of inputs with the same length as `input_layers`.
* `__call__`: all layers are [`Callable`](https://docs.python.org/3/reference/datamodel.html#object.__call__) and the 
result is the computation of the entire network taking the current `Layer` as an endpoint or terminal node.
* `reuse_with`: create a new layer object that **shares the state** with the current layer but is connected to different
inputs. The new layer is the end-point node of a new layer graph.

### Using existing Layers
TensorX ships with a number of built in Layers that you can easily use to compose layer graphs that perform various computations.
All layers are accessible from the global namespace `tensorx.Linear` or from the `tensorx.layers` module.
The following example shows how to use a simple [`Linear`](/layers/core/#linear) layer that performs the computation $y=Wx+b$:
```python 
import tensorflow as tf
import tensorx as tx

x = tf.random.uniform([2, 2], dtype=tf.float32)
# y = Wx + b
y = tx.Linear(x, n_units=3)

result = y()

assert tx.tensor_equal(tf.shape(result), [2, 3])
assert len(y.input_layers) == 1

in_layer = y.input_layers[0]
assert isinstance(in_layer, tx.Input)
assert in_layer.constant
```
Note that we can pass a `Tensor` object to `Linear` (or any other layer), and it will be automatically converted to a
`Layer`, in this case a constant `Input`. The layer `y` has exactly 1 input layer and `__call__` will return the result 
of its computation on this input.

### Dynamic stateful Input
The `Input` layer allows us to add a dynamic input to a layer graph. Suppose we're doing the previous operation, but 
would like to change the value of the input.

```python 
value = tf.random.uniform([2, 2], dtype=tf.float32)
x = tx.Input(init_value=value)
# y = Wx + b
y = tx.Linear(x, n_units=3)

result1 = y()
# x is stateful and its value can be changed e.g. to a new random value
x.value = tf.random.uniform([2,2], dtype=tf.float32)
result2 = y()
result3 = y.compute(value)

assert not tx.tensor_equal(result1, result2)
assert not y.input_layers[0].constant
# compute returns the layer computation independently from its current graph
assert tx.tensor_equal(result1, result3)

print(result1)
```
```shell
tf.Tensor(
    [[ 0.8232075   0.2716378  -0.33215973]
     [ 0.34996247 -0.02594224 -0.05033442]], 
    shape=(2, 3), 
    dtype=float32)
```

`Input` allows the creation of dynamic input layers with a value property that 
can be changed, we can see that the value at the end-point of this graph changes as well. Moreover, 
the `compute` method is distinct from `__call__` as it only depends on the layer current state and 
not on the current graph.

!!! important
    if `n_units` is not set to `None` on a dynamic `Input` layer, it will take the last dimension of the initial value, 
    henceforth, any tensor assigned to `value` must match the `n_units` in its last dimension. This means that the batch
    dimension can be variable for example.
    
!!! warning
    what you **can't** do is switch the number of dimension in a dynamic `Input`. Without an initial value or shape, it
    defaults to a shape `(0, 0)` (an empty tensor with 2 dimensions). An error is thrown if you try to assign a tensor 
    with a mismatching number of dimensions. For example, if you create an input as follows 
    `Input(shape=[None,None,None])`, an error is thrown if you assign a tensor with a mismatching number of dimensions 
    like `input.value = tf.ones([2,2])`.
 
### Re-Using Layers
When you create a new `Layer` object, usually you will pass it its input layers which will then make it the end-node of
a graph connected to those input layers. This will also call the `init_state` method which initializes any `tf.Variable`
objects that might be part of the layers' state. If you want to re-use this layer with a different set of input layers, 
you can use the `reuse_with` method. This creates a new layer with all the same parameters, additionally
this new layer will share it's state with the previous one.

```python
import tensorflow as tf
import tensorx as tx

x1 = tx.Constant(tf.ones([2,2]))
x2 = tx.Constant(tf.zeros([2,2]))
y1 = tx.Linear(x1,4)
# this shared state with y1 but new graph x2 ==> y2
y2 = y1.reuse_with(x2)
```

!!! warning
    Any changes to the state of one layer will affect the state of the second.

### Re-Using Modules
A `Module` is a special layer which creates a single `Layer` from a given _layer graph_. A _layer graph_ is a 
set of layers connected to each other. For example: 

```python 
x = tx.Input(tf.ones([2,2]))
y1 = tx.Linear(x,3)
y2 = tx.Linear(y1,4)

m = tx.Module(inputs=x,output=y2)

assert tx.tensor_equal(m(),y2())
```
You can take the two `Linear` layers and create a single module with a state shared with both layers. Like with any 
other layer you can also call `reuse_with` on a module and in this case, the entire state of the two `Linear` layers 
will again be shared with the newly created `Module`.

## Training
