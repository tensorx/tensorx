# Tutorial
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
<img align="right" width="20%" src="/img/layer_graph.png"> 

In TensorX, a [Layer](/api/layers/Layer/) is the basic building block of a **neural network**. Semantically speaking, a 
layer is an object that can have multiple inputs, an inner state, and a computation function applied to its inputs (that 
depends on the current state). Each layer has a single output. In essence,we can say that a Layer instance is a stateful 
function. Connecting a series of layers results in a **layer graph**. In TensorX, each layer is the end-node of a 
subgraph, and executing it will result in the execution of all layers in the subgraph with the current layer as output.
Layer subclasses can range from simple linear transformations (e.g. [Layer](/api/layers/Linear/)) to more complex 
layers used to build recurrent neural networks such as long short-term memory (LSTM) cells 
(e.g. [LSTMCell](/api/layers/rnn/LSTMCell/)) or attention mechanisms such as [MHAttention](/api/layers/MHAttention/).

### Layer properties and methods

* **`inputs`**: list of input layers for the current layer;
* **`input`**: syntax sugar for `inputs[0]`;
* **`n_units`**: number of output units or neurons, this is the last dimension of the output tensor resulting from this 
  layer's computation;
* **`shape`**: the inferred shape for the layer output;
* **`compute(*tensors)`**: layer computation applied to its input layers or input tensors if any is given. 
* **`__call__`**: all layers are [Callable](https://docs.python.org/3/reference/datamodel.html#object.__call__) and the 
  result is the computation of the entire layer graph taking the current layer as the terminal node.
* **`reuse_with`**: create a new layer object that **shares the state** with the current layer but is connected to 
  different inputs. The new layer is the end-point node of a new layer graph.
* **`variables`**: a `list` of `tf.Variable` objects that handled by the current layer
* **`trainable_variables`**: a `list` of `tf.Variable` objects that are _trainable_, this is, that are changed by an 
  optimizer during training.*
* **`config`**: a layer configuration ([LayerConfig](/api/layers/utils/LayerConfig/)) with the arguments used in the 
  current layer instance constructor.

### Using existing Layers
TensorX ships with a number of built in Layers that you can easily use to compose layer graphs that perform various 
computations. All layers are accessible from the global namespace `tensorx.Linear` or from the `tensorx.layers` module.
The following example shows how to use a simple [`Linear`](/api/layers/Linear) layer that performs the computation 
$y=Wx+b$:

```python 
import tensorflow as tf
import tensorx as tx

x = tf.random.uniform([2, 2], dtype=tf.float32)
# y = Wx + b
y = tx.Linear(x, n_units=3)
result = y()

assert tx.tensor_equal(tf.shape(result), [2, 3])
assert len(y.inputs) == 1
assert isinstance(y.input, tx.Constant)
```

Note that we can pass a `Tensor` object to `Linear` (or any other layer), and it will be automatically converted to a
`Layer`, to a `Constant` layer to be more precise. The layer `y` has exactly 1 input layer and `__call__` will return 
the result of its computation on this input.

### Dynamic stateful Input
The `Input` layer allows us to add a dynamic input to a layer graph:

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
assert not y.input.constant
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
    You **can't** switch the number of dimension in a dynamic `Input`. Without an initial value or shape, it
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

<div style="width:100%;">
  <div style="width:49%; float:left;">
```python
import tensorflow as tf
import tensorx as tx

# stateful input placeholder
x1 = tx.Input(n_units=2)
x1.value = tf.random.uniform([2, 2])
#y = Wx + b
l1 = tx.Linear(x1, n_units=3)
a1 = tx.Activation(l1, tx.relu)
l2 = tx.Linear(a1,n_units=4)

d1 = tx.Dropout(a1,probability=0.4)
l3 = l2.reuse_with(d1)
```
  </div>
  <div style="width:49%;padding-top:50px; float:right;">
  <img width="80%;" src="/img/reuse_with.svg"> 
  </div>
</div>
<div style="clear: both;"></div>


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

## Gradients and Autodiff
Automatic differentiation is a cornerstone of most deep learning frameworks. TensorFlow remembers what operations 
happen and in what order during the forward pass, then, during the backpropagation pass, TensorFlow traverses this list 
of operations in reverse order to compute gradients --usually with respect to some input like a `tf.Variable`. 
Automatic differentiation can be accessed in Tensorflow using the [`tf.GradientTape`](https://www.tensorflow.org/guide/autodiff)
context. Whatever is executed inside the `GradientTape` context, gets tracked so that the gradients with respect to some 
variables can be computed:

```python
import tensorflow as tf
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
  y = x**2

# dy = 2x * dx
dy_dx = tape.gradient(y, x)
```

TensorX Layers describe operations over tensors in terms of tensorflow operations, and store their state in `tf.Variable`
objects, so layers executed inside the `tf.GradientTape` context are tracked just like any other Tensorflow operation. 
With this in mind, we can then compute the gradients of a particular value with respect to the `trainable_variables` 
used in the computation. For example: 

```python
import tensorflow as tf
import tensorx as tx

x = tx.Input(n_units=3)
# y = Wx + b
y = tx.Linear(x, 3, add_bias=True)
loss = tx.Lambda(y, fn=lambda v: tf.reduce_mean(v ** 2))
x.value = [[1., 2., 3.]]

with tf.GradientTape() as tape:  
    loss_value = loss()
    
    # we could have done this as well
    # v = y()
    # loss_value = tf.reduce_mean(v ** 2)

grads = tape.gradient(loss_value, y.trainable_variables)

assert len(y.trainable_variables) == 2
assert len(grads) == 2
assert grads[0].shape == y.weights.shape
assert grads[1].shape == y.bias.shape
```

In this case, only the `weights`, and `bias` of the `Linear` layer are trainable variables, so we can take the gradient 
of `loss_value` with respect to these variables, the result is a list of tensors with the same shape as the variables 
used as targets.

!!! tip
    In these examples we're still using an eager execution model from Tensorflow, as we will see, this is good for 
    debugging, but not very efficient. Next in this tutorial, we will show how we can compile TensorX layer graphs 
    into Tensorflow graphs using the [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).  

## Graph Compilation


## Models

## Callbacks

## Serialization 


