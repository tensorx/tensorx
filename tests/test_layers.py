# suppressing messages only works if set before tensorflow is imported

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
import numpy as np
import pytest
import tensorflow as tf


def test_input_spec():
    x = tx.Input()
    assert x.shape.as_list() == [None, None]
    x = tx.Input(n_units=1, shape=None)
    assert x.shape.as_list() == [None, 1]

    # n_units does not match shape
    with pytest.raises(ValueError):
        tx.Input(n_units=1, shape=[])

    x = tx.Input(n_units=1, shape=[1])
    assert x.shape.as_list() == [1]
    assert tx.tensor_equal(x(), tf.zeros([1]))

    with pytest.raises(ValueError):
        tx.Input(n_units=1, shape=[None, 2])
    with pytest.raises(ValueError):
        tx.Input(n_units=0, shape=[None, 1])

    x = tx.Input(n_units=2, shape=[None, None, 2])
    assert x.n_units == 2
    assert x.shape.as_list() == [None, None, 2]
    assert tx.tensor_equal(x(), tf.zeros([1, 1, 2]))

    value = tf.ones([2, 2])

    x = tx.Input(value)
    assert x.shape.as_list() == [None, 2]

    with pytest.raises(ValueError):
        tx.Input(value, n_units=3)

    with pytest.raises(ValueError):
        tx.Input(value, shape=[None])
    with pytest.raises(ValueError):
        tx.Input(value, shape=[None, 3])
    with pytest.raises(ValueError):
        tx.Input(value, shape=[None, None, 2])

    x = tx.Input(value, shape=[None, 2])
    assert x.shape.as_list() == [None, 2]

    x = tx.Input(value)
    assert x.dtype == value.dtype
    assert x.dtype == tf.float32

    x = tx.Input(value, dtype=tf.int32)
    assert x.dtype == tf.int32
    assert x().dtype == tf.int32

    x = tx.Input(value, dtype=tf.int32, cast=False)
    with pytest.raises(TypeError):
        x.value = value

    x = tx.Input(value, n_active=2, n_units=10)
    assert x.dtype == tf.int64
    assert isinstance(x(), tf.SparseTensor)

    with pytest.raises(ValueError):
        # [2,2] not compatible with [None,3]
        tx.Input(value, n_active=3, n_units=10)


def test_input_config():
    in1 = tx.Input(init_value=tf.ones([2, 2]), n_units=2)
    cfg1 = in1.config
    in2 = cfg1()

    assert tx.tensor_equal(in1(), in2())


def test_input_value():
    inputs = tx.Input(n_units=4, dtype=tf.int32, constant=False)
    assert tx.tensor_equal(inputs.value, tf.zeros([1, 4], dtype=tf.int32))

    with pytest.raises(ValueError):
        inputs.value = np.ones([2, 3], dtype=np.int32)

    inputs.value = np.ones([2, 4], dtype=np.int32)
    assert inputs.value is not None
    assert inputs() is not None
    assert inputs().dtype == tf.int32

    # test sparse input
    inputs = tx.Input(n_units=4, n_active=2, dtype=tf.int64, constant=False)
    assert tx.tensor_equal(inputs.value, tf.zeros([0, 2], dtype=tf.int64))

    with pytest.raises(ValueError) as ve:
        inputs.value = [[0, 2, 2]]
        assert "Invalid shape" in str(ve)

    inputs.value = [[0, 2]]
    # create an equivalent sparse input
    sp_input = inputs()
    assert isinstance(sp_input, tf.SparseTensor)
    inputs2 = tx.Input(n_units=4, init_value=sp_input)

    dense_value = tf.sparse.to_dense(inputs())
    dense_value2 = tf.sparse.to_dense(inputs2())
    expected = tf.constant([[1, 0, 1, 0]], dtype=np.int64)
    assert tx.tensor_equal(expected, dense_value)
    assert tx.tensor_equal(dense_value, dense_value2)


def test_input_compile():
    inputs = tx.Input(n_units=4, dtype=tf.int32, constant=False)
    fn = tf.function()(inputs.__call__)
    assert tx.tensor_equal(fn(), tf.zeros([1, 4], dtype=tf.int32))


def test_input_3d():
    # we either create a 3d input or specify the shape
    data = np.ones([2, 2, 2], dtype=np.float32)
    x = tx.Input(shape=[None, None, 2], dtype=tf.float32, n_units=2)
    x.value = data
    x.value = x() * 2
    assert tx.tensor_equal(data * 2, x())

    x2 = tx.Input(data)
    assert x2.n_units == np.shape(data)[-1]
    assert x2.shape[-1] == np.shape(data)[-1]
    x2.value = x2() * 2
    assert tx.tensor_equal(data * 2, x2())

    with pytest.raises(ValueError, match="Invalid shape"):
        x3 = tx.Input(n_units=2)
        x3.value = data
        pytest.fail("Value Error Expected: invalid shape for value set")


def test_dynamic_input_graph():
    """
    When we freeze the graph function with a dynamic input,
    the function includes a variable value read operation, that
    reads from the variable defined in the Input layer
    """
    x = tx.Input(tf.zeros([2, 2]), n_units=2, constant=False)
    y = tx.Linear(x, 2, add_bias=False)
    graph_function = y.as_function()
    out1 = graph_function()

    assert tx.tensor_equal(out1, tf.zeros([2, 2]))

    x.value = tf.ones([2, 2])
    out2 = graph_function()

    assert tx.tensor_equal(out2, tf.matmul(tf.ones([2, 2]), y.weights))
    assert not tx.tensor_equal(out1, out2)


def test_activation():
    inputs = tx.Input(init_value=tf.ones([2, 2]), n_units=2)
    output = tx.Activation(inputs, tf.sigmoid)
    assert tx.shape_same(inputs.shape, output.shape)


def test_shared_state():
    inputs = tf.ones([2, 4])
    l1 = tx.Linear(inputs, 8)
    l2 = tx.Linear(inputs, 8, share_state_with=l1)
    proto = tx.Linear.config(n_units=8, share_state_with=l1)
    l3 = proto(inputs)

    assert l1.weights is l2.weights
    assert l1.bias is l2.bias
    assert l1.weights is l3.weights
    assert l1.bias is l3.bias


def test_mul():
    # also tests graphs with constants
    inputs = tx.Constant(tf.constant(2), dtype=tf.float64)
    inputs2 = inputs * 2
    assert tx.tensor_equal(inputs2(), inputs() * 2)

    inputs2_fn = tf.function(inputs2.__call__)
    assert inputs2_fn() == inputs2()


def test_linear_function():
    inputs = tx.Constant(tf.ones([2, 4]), dtype=tf.float64)
    linear = tx.Linear(inputs, n_units=8, dtype=tf.float64)
    fn = tf.function(linear.__call__)
    assert tx.tensor_equal(fn(), linear())


def test_linear():
    inputs = tx.Constant(tf.ones([2, 4]), dtype=tf.float64)
    inputs2 = inputs * 2

    linear = tx.Linear(inputs, n_units=8, dtype=tf.float64)

    w = linear.weights
    b = linear.bias

    assert w.shape == [4, 8]
    assert b.shape == [8]
    assert len(linear.trainable_variables) == 2

    t1 = linear()
    t2 = linear()

    assert tx.tensor_equal(t1, t2)

    linear2 = tx.Linear(linear.inputs[0], 8, share_state_with=linear, dtype=tf.float64)
    t3 = linear2()
    assert tx.tensor_equal(t1, t3)

    linear = tx.Linear(inputs, 8, dtype=tf.float64)
    linear2 = linear.reuse_with(inputs2)

    assert linear.weights is linear2.weights
    assert linear.bias is linear2.bias

    assert tx.tensor_equal(linear() * 2, linear2())


def test_linear_rank3():
    val = tf.constant([[[1], [1]], [[2], [2]]])
    x1 = tx.Input(val, dtype=tf.float32)
    x2 = tx.Transpose(x1)

    assert val.shape[1:] == x1.shape[1:]

    x1_flat = tx.Reshape(x1, [-1, 1])

    linear1 = tx.Linear(x1, n_units=2)
    linear2 = tx.Linear(x2,
                        weights_shape=[2, 1],
                        weights=linear1.weights,
                        transpose_weights=True)

    # we cant do this because it changes the definition
    # of the layer (n_units etc)
    with pytest.raises(ValueError):
        linear1.reuse_with(x2, transpose_weights=True)
        pytest.fail("can't reuse with transpose weights while changing the layer definition")

    linear_flat = linear1.reuse_with(x1_flat)
    linear_flat = tx.Reshape(linear_flat, x1().get_shape().as_list()[:-1] + [2])

    assert tx.tensor_equal(linear1(), linear_flat())
    assert tx.tensor_equal(tf.shape(linear2()), [1, 2, 1])


def test_constant_shape():
    tensor = tf.ones([3, 3])
    const_layer = tx.Constant(tensor)

    assert const_layer.shape == tensor.shape


def test_transpose():
    tensor = tf.ones([3, 3])
    trans_tensor = tf.transpose(tensor)
    trans_layer = tx.Transpose(tensor, n_units=3)

    assert trans_layer.input.shape == [3, 3]
    assert trans_layer.shape == trans_tensor.shape

    tensor = tf.ones([2, 3, 4])
    perm = [2, 0, 1]
    trans_tensor = tf.transpose(tensor, perm)
    trans_layer = tx.Transpose(tensor, perm)

    assert trans_layer.input.n_units == tensor.shape[-1]
    assert trans_layer.shape == trans_tensor.shape
    assert trans_layer.n_units == tensor.shape[perm[-1]]

    inputs = tx.Input(shape=tf.TensorShape([None, 3]))
    trans = tx.Transpose(inputs)
    assert trans.shape[-1] is None
    assert trans.shape[0] == 3


def test_reshape_shape():
    x = tf.reshape(tf.range(9), [3, 3, 1])
    x = tx.Input(x, dtype=tf.float32)
    flat = tx.Reshape(x, [-1, 1])
    assert flat.shape[0] is None
    assert flat.shape[-1] == 1

    x = tx.Input(x, shape=[3, 3, 1], dtype=tf.float32)
    print(x.shape)
    flat = tx.Reshape(x, [-1, 1])
    print(flat.shape)


def test_transpose_reshape():
    x = tf.reshape(tf.range(9), [3, 3])
    x2 = tx.Reshape(tf.range(9), [3, 3])

    assert tx.tensor_equal(x2(), x)
    assert tx.tensor_equal(x2.compute(tf.range(9)), x)

    t = tf.transpose(x)
    y = tx.Transpose(t)
    assert tx.tensor_equal(y(), x)
    assert tx.tensor_equal(y.compute(x), t)

    x = tf.reshape(tf.ones([18]), [-1, 3, 2])

    x2 = tx.Reshape(tf.ones([18]), [-1, 3, 2])

    assert x.shape == [3, 3, 2]
    assert x.shape == x2.shape


def test_mul_shape():
    x = tx.Input(n_units=3)
    m = x * 2
    assert m.shape[0] is None
    assert m.shape[-1] is 3

    t = tx.Transpose(x)
    assert t.shape[-1] is None
    t = tx.Transpose(x, n_units=3)
    assert t.shape == [3, 3]
    m = t * 2
    assert m.shape == [3, 3]

    x = tx.Input(n_units=3)  # [None,3]
    t = tx.Transpose(x)  # [None,None]
    assert t.shape[0] == 3
    assert t.shape[-1] is None

    m = t * 2  # [None,None]

    # TensorShape([3,None]) != TensorShape([3,None])
    # because we don't know what None is
    assert m.shape[0] == 3
    assert m.shape[-1] is None


def test_module_shape():
    x = tx.Input(n_units=3)
    t = tx.Transpose(x, n_units=3)
    mul = t * 2
    assert mul.shape == [3, 3]
    m = tx.Module(output=mul, inputs=x)
    assert m.n_units == 3
    m()


def test_wrap_shape():
    x = tx.Input(n_units=3)
    t = tx.Transpose(x, n_units=3)
    assert t.shape[-1] == 3

    w = tx.Wrap(t, wrap_fn=lambda layer: layer * 2)
    assert w.shape == [3, 3]


def test_wrap_transpose():
    tensor = tf.reshape(tf.range(9), [3, 3])
    t = tf.transpose(tensor)

    t_layer = tx.Transpose(t, n_units=3)
    assert t_layer.shape == (3, 3)

    mul2 = tx.Wrap(t_layer, wrap_fn=lambda layer: layer * 2)
    mul2_2 = mul2.reuse_with(tensor)

    assert tx.tensor_equal(mul2_2(), t * 2)
    assert tx.tensor_equal(mul2(tensor), t * 2)
    assert tx.tensor_equal(mul2(t), mul2())
    assert tx.tensor_equal(mul2.compute(t), mul2())
    assert tx.tensor_equal(mul2.compute(t), tf.transpose(t) * 2)
    assert tx.tensor_equal(t_layer.compute(t), tensor)
    assert tx.tensor_equal(mul2_2.compute(tensor), mul2_2())


def test_variable_layer():
    input_layer = tx.Input([[1]], n_units=1, dtype=tf.float32)
    var_layer = tx.VariableLayer(input_layer, dtype=tf.float32)

    init_value = var_layer.variable.value()
    after_update = var_layer()

    assert not tx.tensor_equal(init_value, after_update)
    assert tx.tensor_equal(after_update, var_layer.variable.value())


def test_variable_init_from_input():
    input_layer = tx.Input(n_units=1, constant=False)
    layer_once = tx.VariableLayer(input_layer, update_once=True)
    layer_var = tx.VariableLayer(input_layer, update_once=False)

    layer_once.reuse_with(init_from_input=False)

    data1 = np.array([[1]])
    data2 = np.array([[2]])
    data3 = np.array([[3]])

    input_layer.value = data1
    # counter is a tf.Variable
    assert layer_once.counter.value() == 0
    input_layer.value = data2
    y1 = layer_once()

    assert layer_once.counter.value() == 1
    assert tx.tensor_equal(layer_once.variable.value(), y1)
    input_layer.value = data3
    y2 = layer_once()
    assert layer_once.counter.value() == 1
    assert tx.tensor_equal(y1, y2)
    assert tx.tensor_equal(y1, layer_once.variable.value())

    # dynamic var layer
    input_layer.value = data1
    assert layer_var.counter.value() == 0
    y1 = layer_var()
    assert layer_var.counter.value() == 1
    assert tx.tensor_equal(layer_var.variable.value(), y1)

    input_layer.value = data2
    y2 = layer_var()
    assert layer_var.counter.value() == 2
    assert not tx.tensor_equal(y1, y2)


def test_variable_layer_reuse():
    input_layer = tx.Input([[1]], n_units=1, dtype=tf.float32)
    input_layer2 = tx.Input([[1], [2]], n_units=1, dtype=tf.float32)
    var1 = tx.VariableLayer(shape=[2, 1])

    var2 = var1.reuse_with(input_layer)
    var3 = var1.reuse_with(input_layer2)

    v0 = var1()
    v1 = var2()
    assert not tx.tensor_equal(v0, v1)

    # v0 inner variable changed when we evaluate v1
    v2 = var1()
    assert not tx.tensor_equal(v0, v1)

    v3 = var3()
    assert not tx.tensor_equal(v2, v3)
    v4 = var1()
    assert tx.tensor_equal(v3, v4)

    # variable batch dimension is dynamic its shape will be different
    assert not tx.shape_equal(v4, v1)
    assert tx.shape_equal(v2, v1)


def test_standalone_variable_layer():
    var_layer = tx.VariableLayer(shape=[4])

    assert tx.tensor_equal(np.zeros([4], dtype=np.float32), var_layer())
    assert not tx.tensor_equal(np.zeros([4]), var_layer())
    assert tx.tensor_equal(tf.zeros([4]), var_layer())


def test_merge_add_shape():
    x1 = tx.Input([[2.]], n_units=1, name="x1")
    x2 = tx.Input([[2.]], n_units=1, name="x2")

    add = tx.Add(x1, x2)
    assert len(add.shape) == 2
    assert add.shape[-1] == 1
    assert add.shape[0] is None


def test_module_reuse_order():
    x1 = tx.Input([[2.]], n_units=1, name="x1")
    x2 = tx.Input([[2.]], n_units=1, name="x2")
    x3 = tx.Input([[1.]], n_units=1, name="x3")

    h = tx.Add(x2, x3)
    y = tx.Add(x1, h)

    module = tx.Module(inputs=[x1, x2, x3], output=y)

    x1_ = tx.Constant([[2.]], name="x1b")
    x2_ = tx.Constant([[2.]], name="x2b")

    m2 = module.reuse_with(x1_, x2_)

    m1 = module()
    m2 = m2()

    assert tx.tensor_equal(m1, m2)


def test_module_rnn():
    """ Module + RNN integration
    """
    # test wrapping module around RNN because it has input dependencies that might not be given in the constructor
    x1 = tx.Input(tf.ones([1, 2, 3]), n_units=3, name="x1")
    x2 = tx.Input(tf.ones([1, 2, 3]), n_units=3, name="x2")
    rnn1 = tx.RNN(x1, cell_config=tx.LSTMCell.config(n_units=4), n_units=4, stateful=False)
    rnn2 = tx.RNN(x1, cell_config=tx.LSTMCell.config(n_units=4), n_units=4, stateful=False)

    out = tx.Concat(rnn1, rnn2)

    # add previous state as a dependency to a module
    m = tx.Module(inputs=x1, output=out,
                  dependencies=rnn1.previous_state + rnn2.previous_state)

    m2 = m.reuse_with(x2)
    var_layers = set()
    for node in m2.graph.dependency_iter():
        if isinstance(node, tx.VariableLayer):
            var_layers.add(node)

    assert var_layers == set(rnn1.previous_state + rnn2.previous_state)
    assert tx.tensor_equal(m(), m2())


def test_module_with_attention():
    """ Module + Attention integration
    This also tests Graph indirectly to check if we can add layers
    whose input layers are the same object (e.g. in self-attention)
    """

    x1 = tx.Input(tf.ones([1, 2, 3]), n_units=3, name="x1")
    rnn1 = tx.RNN(x1, cell_config=tx.LSTMCell.config(n_units=4), n_units=4, stateful=False)
    att = tx.MHAttention(rnn1, rnn1, rnn1, n_units=3)
    m = tx.Module(inputs=x1, output=att, dependencies=rnn1.previous_state)
    g = tx.Graph.build(inputs=x1, outputs=m, add_missing_inputs=True)
    fn = g.as_function(ord_inputs=x1, ord_outputs=m)
    # this returns a tuple
    out1 = g.compute(tf.ones([1, 2, 3]))
    # this returns the function result
    out2 = fn(tf.ones([1, 2, 3]))

    assert tx.tensor_equal(out1[0], out2)


def test_module():
    l1 = tx.Input([[1]], n_units=1, dtype=tf.float32)
    l2 = tx.Input([[1]], n_units=1, dtype=tf.float32)
    l3 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l1, l2)
    l4 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l1, l2)
    l5 = tx.Linear(l4, 1)
    in1 = tx.Input([[1]], n_units=1, dtype=tf.float32)
    l7 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l3, in1)
    l8 = tx.layer(n_units=1)(lambda x1, x2: tf.add(x1, x2))(l7, l5)

    in2 = tx.Input([[1]], n_units=1, dtype=tf.float32, constant=False)
    in3 = tx.Input([[1]], n_units=1, dtype=tf.float32)

    m = tx.Module([l1, l2, in1], l8)
    with tf.name_scope("module_reuse"):
        m2 = m.reuse_with(in2, in3, in1)

    assert tx.tensor_equal(m(), m2())
    in2.value = [[3]]
    assert not tx.tensor_equal(m(), m2())


def test_rnn_cell():
    n_inputs = 4
    n_hidden = 2
    batch_size = 2

    x = tx.Input(init_value=tf.ones([batch_size, n_inputs]), constant=False)
    rnn1 = tx.RNNCell(x, n_hidden)

    assert rnn1.shape[0] == x.shape[0]
    assert rnn1.shape[-1] == rnn1.n_units

    state = rnn1.state
    state = state[0]()

    rnn_2 = rnn1.reuse_with(x, state)
    rnn_3 = rnn1.reuse_with(x)

    with pytest.raises(TypeError):
        tx.RNNCell(x, n_hidden, share_state_with=x)
        pytest.fail("Type Error Expected: inputs cannot share state with RNNCell")

    res1 = rnn1()
    res2 = rnn_2()
    res3 = rnn_3()

    assert (batch_size, n_hidden) == np.shape(res1)
    assert tx.tensor_equal(res1, res3)
    assert not tx.tensor_equal(res1, res2)


def test_rnn_layer_config():
    x1 = tx.Input(init_value=tf.ones([2, 2]), n_units=2)
    x_config = x1.config
    x2 = x_config()
    assert tx.tensor_equal(x1(), x2())

    rnn_cell = tx.RNNCell(input_layer=x1, n_units=3)
    rnn_proto = rnn_cell.config
    rnn_cell2 = rnn_proto(x1)

    assert tx.shape_equal(rnn_cell(), rnn_cell2())
    assert not tx.tensor_equal(rnn_cell(), rnn_cell2())


def test_gru_cell_module():
    n_inputs = 4
    n_hidden = 2
    batch_size = 2
    data = tf.ones([batch_size, 4])

    inputs = tx.Input(init_value=data, n_units=n_inputs, dtype=tf.float32)
    rnn_1 = tx.GRUCell(inputs, n_hidden)
    fn = rnn_1.as_function(compile=True)
    fn()

    # rnn_2 = rnn_1.reuse_with(inputs, rnn_1)
    #
    # rnn_3 = rnn_1.reuse_with(inputs, tx.GRUCell.zero_state(rnn_1.n_units))
    #
    # res1 = rnn_1()
    # res2 = rnn_2()
    # res3 = rnn_3()
    #
    # assert (batch_size, n_hidden) == np.shape(res1)
    # assert tx.tensor_equal(res1, res3)
    # assert not tx.tensor_equal(res1, res2)


def test_rnn_cell_graph():
    n_inputs = 4
    n_hidden = 2
    batch_size = 2

    data1 = tf.ones([batch_size, n_inputs])
    inputs = tx.Input(data1)
    rnn1 = tx.RNNCell(inputs, n_hidden)
    # if we use missing_inputs=True, extra inputs might be added
    with pytest.raises(ValueError):
        tx.Graph.build(inputs=inputs,
                       outputs=rnn1,
                       add_missing_inputs=False)
        pytest.fail("Value Error Expected: missing inputs")

    g = tx.Graph.build(inputs=inputs,
                       outputs=rnn1,
                       add_missing_inputs=True)
    f = g.as_function(ord_inputs=inputs)
    f(data1)


def test_rnn_cell_drop():
    n_hidden = 4
    inputs1 = tx.Input(np.ones([2, 100]), dtype=tf.float32)
    inputs2 = tx.Input(np.ones([2, 100]), dtype=tf.float32)

    with tf.name_scope("wtf"):
        rnn1 = tx.RNNCell(inputs1, n_hidden,
                          x_dropout=0.5,
                          r_dropout=0.5,
                          u_dropconnect=0.5,
                          w_dropconnect=0.5,
                          regularized=True
                          )
    rnn2 = rnn1.reuse_with(inputs2, rnn1)
    rnn3 = rnn1.reuse_with(inputs2, rnn1)
    rnn4 = rnn1.reuse_with(inputs2, None, regularized=False)
    rnn5 = rnn4.reuse_with(inputs2, None, regularized=True)

    r1, r2, r3, r4, r5 = rnn1(), rnn2(), rnn3(), rnn4(), rnn5()
    # w is a linear layer from the input but a regularized layer applies dropout to the input, so we have a dropout
    # in between

    # without a shared state object, we couldn't rewire graphs, in the case of non-eager we can share a tensor
    # that is already wired with something (it takes the shape of the input of one layer and creates a mask tensor
    # shared across dropout instances
    # Linear layers should have shared states as well, in this case sharing the weights
    # dropout_state1 = rnn1.w.input_layers[0].layer_state
    # dropout_state2 = rnn2.w.input_layers[0].layer_state
    # dropout_state3 = rnn3.w.input_layers[0].layer_state

    # mask1, mask2, mask3 = dropout_state1.mask, dropout_state2.mask, dropout_state3

    assert tx.tensor_equal(r2, r3)
    assert not tx.tensor_equal(r2, r4)
    assert not tx.tensor_equal(r4, r5)

    assert rnn1.dropout_locked
    assert rnn2.dropout_locked

    assert hasattr(rnn1, "w")
    assert hasattr(rnn2, "w")

    w1: tx.Layer = getattr(rnn1, "w")
    w2: tx.Layer = getattr(rnn2, "w")

    assert isinstance(w1, tx.DropConnect)

    state1, state2 = w1.layer_state, w2.layer_state

    assert hasattr(state1, "weight_mask")
    assert hasattr(state2, "weight_mask")

    # dropout locked == true
    mask1 = getattr(state1, "weight_mask")
    mask2 = getattr(state2, "weight_mask")

    assert tx.tensor_equal(mask1, mask2)


def test_to_sparse():
    inputs = tx.Input(init_value=tf.ones([2, 100]))
    linear = tx.Linear(inputs, n_units=100)
    relu = tx.Activation(linear, tx.relu)
    sparse = tx.ToSparse(relu)

    assert tx.shape_same(sparse.shape, linear.shape)
    assert tx.shape_same(sparse.shape, relu.shape)


def test_gate():
    inputs = tx.Input(init_value=tf.ones([2, 3]))
    linear = tx.Linear(inputs, n_units=4)
    nop = tx.Activation(linear, fn=tx.identity)
    gate_w = tx.Linear(linear, n_units=4, add_bias=True)
    gate1 = tx.Gate(linear, gate_w)
    gate2 = gate1.reuse_with(nop)

    assert tx.shape_same(gate1.shape, gate2.shape)

    r1 = gate1()
    r2 = gate2()

    assert tx.tensor_equal(r1, r2)


def test_coupled_gate(self):
    vocab_size = 4
    n_features = 3
    seq_size = 2

    inputs = tx.Input(init_value=np.array([[2, 0], [1, 2]]),
                      n_units=seq_size,
                      dtype=tf.int32,
                      constant=True)

    features1 = tx.Lookup(inputs, seq_size, embedding_shape=[vocab_size, n_features]).as_concat()
    features2 = tx.Lookup(inputs, seq_size, embedding_shape=[vocab_size, n_features]).as_concat()

    sp_features1 = tx.ToSparse(features1)

    gate_w = tx.Linear(features1, seq_size, add_bias=True)
    coupled_gate = tx.CoupledGate(features1, features2, gate_w)

    coupled_gate2 = coupled_gate.reuse_with(sp_features1, features2)

    r1 = coupled_gate()
    r2 = coupled_gate2()

    assert tx.tensor_equal(r1, r2)


def test_module_gate():
    """ Module + Gate Integration
    """
    x1 = tx.Input([[1, 1, 1, 1]], n_units=4, dtype=tf.float32)
    x2 = tx.Input([[1, 1]], n_units=2, dtype=tf.float32)
    x1 = tx.Add(x1, x1)

    gate = tx.Gate(input_layer=x1, gate_input=x2, gate_fn=tf.sigmoid)
    gate_module = tx.Module([x1, x2], gate)

    x3 = tx.Input([[1, 1, 1, 1]], n_units=4, dtype=tf.float32)
    x4 = tx.Input([[1, 1]], n_units=2, dtype=tf.float32)

    m2 = gate_module.reuse_with(x3, x4)

    result1 = gate_module()
    result2 = m2()
    result3 = gate_module.compute(x3, x4)

    assert tx.tensor_equal(result1, result2 * 2)
    assert tx.tensor_equal(result2, result3)


def test_gru_cell():
    n_inputs = 4
    n_hidden = 2
    batch_size = 2
    data = tf.ones([batch_size, 4])

    inputs = tx.Input(init_value=data, n_units=n_inputs, dtype=tf.float32)
    rnn_1 = tx.GRUCell(inputs, n_hidden)

    rnn_2 = rnn_1.reuse_with(inputs, rnn_1)

    # if we don't wipe the memory it reuses it
    rnn_3 = rnn_1.reuse_with(inputs, tx.GRUCell.zero_state(rnn_1.n_units))

    res1 = rnn_1()
    res2 = rnn_2()
    res3 = rnn_3()

    assert (batch_size, n_hidden) == np.shape(res1)
    assert tx.tensor_equal(res1, res3)
    assert not tx.tensor_equal(res1, res2)


def test_lstm_cell():
    n_inputs = 4
    n_hidden = 2
    batch_size = 2

    inputs = tx.Input(np.ones([batch_size, n_inputs], np.float32), n_units=n_inputs, constant=True)
    rnn1 = tx.LSTMCell(inputs, n_hidden, gate_activation=tf.sigmoid)
    previous_state = (None, rnn1.state[-1]())
    rnn2 = rnn1.reuse_with(inputs, *previous_state)

    # if we don't wipe the memory, memory will be reused
    previous_state = (None, tx.LSTMCell.zero_state(rnn1.n_units))
    rnn3 = rnn1.reuse_with(inputs, *previous_state)
    rnn4 = rnn1.reuse_with(inputs)

    res1 = rnn1()
    res2 = rnn2()
    res3 = rnn3()
    res4 = rnn4()

    assert (batch_size, n_hidden) == np.shape(res1)
    assert tx.tensor_equal(res1, res3)
    assert not tx.tensor_equal(res1, res2)
    assert tx.tensor_equal(res1, res4)


def test_lstm_cell_regularization():
    n_inputs = 8
    n_hidden = 2
    batch_size = 2

    inputs = tx.Input(n_units=n_inputs, constant=False)

    rnn1 = tx.LSTMCell(inputs, n_hidden,
                       u_dropconnect=0.1,
                       w_dropconnect=0.1,
                       name="lstm1")

    rnn2 = rnn1.reuse_with(inputs,
                           *rnn1.state,
                           regularized=True,
                           name="lstm2"
                           )

    rnn3 = rnn2.reuse_with(inputs,
                           *rnn1.state,
                           name="lstm3"
                           )

    data = np.ones([batch_size, n_inputs])

    inputs.value = data

    assert tx.tensor_equal(rnn2, rnn3)
    assert not tx.tensor_equal(rnn1, rnn3)
    assert not tx.tensor_equal(rnn1, rnn3)

    state2, state3 = rnn2.w_f.weight_mask, rnn3.w_f.weight_mask
    assert tx.tensor_equal(state2, state3)

    w2, w3 = rnn2.w_f, rnn3.w_f
    assert tx.tensor_equal(w2, w3)
    w2, w3 = rnn2.w_i, rnn3.w_i
    assert tx.tensor_equal(w2, w3)
    w2, w3 = rnn2.w_o, rnn3.w_o
    assert tx.tensor_equal(w2, w3)
    w2, w3 = rnn2.w_c, rnn3.w_c
    assert tx.tensor_equal(w2, w3)


def test_lstm_cell_state():
    n_inputs = 8
    n_hidden = 2
    batch = 3

    x = tf.ones([batch, n_inputs], dtype=tf.float32)

    cell = tx.LSTMCell(x, n_hidden,
                       u_dropconnect=0.1,
                       w_dropconnect=0.1,
                       name="cell")

    state = cell.previous_state
    assert len(state) == 2
    # state = [s() for s in state]

    state = tx.Graph.build(inputs=None,
                           outputs=cell.state)

    x = tf.random.uniform([batch, n_inputs])
    s = state.compute(x)
    state.compute(x, *s)


def test_rnn_layer():
    n_features = 5
    embed_size = 4
    hidden_dim = 3
    seq_size = 3
    batch_size = 2

    inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
    lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
    seq = lookup.permute_batch_time()

    ones_state = tf.ones([batch_size, hidden_dim])
    zero_state = (tf.zeros([batch_size, hidden_dim]))

    rnn_proto = tx.RNNCell.config(n_units=hidden_dim)

    rnn1 = tx.RNN(seq, cell_config=rnn_proto, previous_state=ones_state, return_state=True)
    rnn2 = rnn1.reuse_with(seq)

    #  problem with RNN layer is that it uses modules that require
    #  all the params to output the right answer
    #  we need to supply the default values for the rest or all the inputs
    out1, last1 = rnn1()
    out2, last2 = rnn2()

    assert tx.tensor_equal(out1, out2)
    assert tx.tensor_equal(last1, last2)

    rnn3 = rnn1.reuse_with(seq, zero_state)
    rnn4 = rnn3.reuse_with(seq)
    rnn5 = rnn4.reuse_with(seq, ones_state)

    assert tx.tensor_equal(rnn2.previous_state, rnn1.previous_state)
    assert tx.tensor_equal(rnn3.previous_state, rnn4.previous_state)

    out3, last3 = rnn3()
    out4, last4 = rnn4()

    assert tx.tensor_equal(out3, out4)
    assert tx.tensor_equal(last3, last4)

    cell_state1 = rnn1.cell.previous_state[0]()
    cell_state2 = rnn2.cell.previous_state[0]()
    cell_state3 = rnn3.cell.previous_state[0]()
    cell_state4 = rnn4.cell.previous_state[0]()

    assert len(rnn1.cell.previous_state) == 1

    assert tx.tensor_equal(cell_state1, cell_state2)
    assert tx.tensor_equal(cell_state3, cell_state4)

    assert not tx.tensor_equal(out1, out3)

    out5, last5 = rnn5()

    assert tx.tensor_equal(out1, out5)
    assert tx.tensor_equal(last1, last5)


def test_biRNN():
    # bidirectional RNN
    n_features = 5
    embed_size = 4
    hidden_dim = 3
    seq_size = 6
    batch_size = 2

    inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
    lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
    seq = lookup.permute_batch_time()

    rnn_proto = tx.RNNCell.config(n_units=hidden_dim)
    rnn0 = tx.RNN(seq, cell_config=rnn_proto, stateful=False, return_state=True)

    # because a stateful rnn0 has a variable layer as input as well
    rnn_m0 = tx.Module(inputs=rnn0.inputs, output=rnn0)

    rnn1 = rnn0.reuse_with(seq, reverse=True, stateful=False, return_state=True)
    # this solves rnn output multiple tensors

    r01 = rnn_m0.compute(seq(), rnn0.previous_state[0]())
    rnn0.reset()
    r02 = rnn0()

    assert tx.tensor_equal(r01[0], r02[0])

    rnn0_ = rnn0[0]
    rnn1_ = rnn1[0]
    rnn0 = tx.Wrap(rnn0, wrap_fn=lambda y: y[0], n_units=rnn0.n_units)
    rnn1 = tx.Wrap(rnn1, wrap_fn=lambda y: y[0], n_units=rnn1.n_units)

    assert tx.shape_equal(rnn0(), rnn1())
    assert tx.shape_equal(rnn0(), rnn0_())
    assert tx.shape_equal(rnn1(), rnn1_())


def test_stateful_rnn_layer():
    n_features = 5
    embed_size = 4
    hidden_dim = 3
    seq_size = 3
    batch_size = 2

    inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
    lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
    seq = lookup.permute_batch_time()

    rnn_proto = tx.RNNCell.config(n_units=hidden_dim)

    rnn1 = tx.RNN(seq, cell_config=rnn_proto, stateful=True, return_state=True)
    lstm1 = tx.RNN(seq, cell_config=tx.LSTMCell.config(n_units=hidden_dim), stateful=True, return_state=True)

    zero_state0 = [layer() for layer in rnn1.previous_state]

    assert len(zero_state0) == 1
    expected_state = tf.zeros([1, hidden_dim], dtype=tf.float32)
    assert tx.tensor_equal(zero_state0[0], expected_state)

    # import logging
    # logging.getLogger("tensorx").setLevel(logging.DEBUG)

    out1, state1 = rnn1()

    tx.Graph.build(inputs=None, outputs=lstm1)
    # out2, state2 = lstm1()
    lstm1()

    # state after single run
    # zero_state1 = [layer() for layer in ]
    zero_state1 = rnn1.previous_state[0]()
    assert tx.tensor_equal(zero_state1, state1)

    rnn1.reset()
    reset_state = rnn1.previous_state[0]()
    assert tx.tensor_equal(reset_state, zero_state0[0])


def test_lookup_sequence_dense():
    input_dim = 4
    embed_dim = 3
    seq_size = 2
    batch_size = 3

    inputs = tx.Input(np.array([[2, 0], [1, 2]]), 2, dtype=tf.int64)
    tensor_input = tx.Input(tf.constant([2]), 1, dtype=tf.int64)

    lookup = tx.Lookup(inputs, seq_size,
                       embedding_shape=[input_dim, embed_dim],
                       batch_size=batch_size,
                       batch_padding=True)

    lookup_from_tensor = lookup.reuse_with(tensor_input)

    v1 = lookup()
    v2 = lookup_from_tensor()

    assert np.shape(v1) == (batch_size, seq_size, embed_dim)
    assert np.shape(v2) == (batch_size, seq_size, embed_dim)


def test_lookup_dynamic_sequence():
    seq1 = [[1, 2], [3, 4]]
    seq2 = [[1, 2, 3], [4, 5, 6]]

    n = 10
    m = 4

    inputs = tx.Input(dtype=tf.int32, constant=False)

    lookup = tx.Lookup(inputs, seq_size=None, embedding_shape=[n, m])
    concat = lookup.as_concat()

    inputs.value = seq1
    inputs()

    inputs.value = seq2
    inputs()

    inputs.value = seq1
    l1 = lookup()
    inputs.value = seq2
    l2 = lookup()

    inputs.value = seq1
    c1 = concat()
    inputs.value = seq2
    c2 = concat()

    assert np.shape(l1)[-1] == m
    assert np.shape(l2)[-1] == m

    assert np.shape(c1)[-1] == m * 2
    assert np.shape(c2)[-1] == m * 3


def test_dynamic_concat():
    seq1 = [[1, 2], [3, 4]]
    seq2 = [[1, 2, 3], [4, 5, 6]]

    n = 10
    m = 4

    inputs = tx.Input(seq2, shape=[None, None], dtype=tf.int32, constant=False)
    inputs2 = tx.Input(seq2, dtype=tf.int32, constant=True)

    lookup = tx.Lookup(inputs, seq_size=None, embedding_shape=[n, m])
    lookup2 = tx.Lookup(inputs2, seq_size=3, embedding_shape=[n, m])
    concat1 = lookup.as_concat()
    concat2 = lookup2.as_concat()

    assert concat1.n_units is None
    assert concat2.n_units is not None

    concat3 = tx.SeqConcat(lookup, time_major=False)
    concat4 = tx.SeqConcat(lookup, seq_size=3, time_major=False)

    c1, c2 = concat1(), concat3()
    assert tx.tensor_equal(c1, c2)
    assert concat3.n_units is None
    assert concat4.n_units == 3 * lookup.n_units

    inputs.value = seq1
    l1 = lookup()
    inputs.value = seq2
    l2 = lookup()

    assert np.shape(l1)[-1] == m
    assert np.shape(l2)[-1] == m


def test_lookup_dynamic_sparse_sequence():
    """ Testing Sparse Inputs to Lookup with dynamic
    seq_len passed through Input layer that acts as
    a parameter (scalar, this is n_units = 0)
    """
    k = 8
    m = 3
    seq1 = tf.SparseTensor(
        indices=[[0, 1], [1, 2],
                 [2, 3], [3, 4]],
        values=[1, 2, 3, 4],
        dense_shape=[4, k]
    )
    seq2 = tf.SparseTensor(
        indices=[[0, 1], [1, 2], [2, 3],
                 [3, 3], [4, 4], [5, 5]],
        values=[1, 2, 3, 3, 4, 5],
        dense_shape=[6, k]
    )

    inputs = tx.Input(n_units=k, sparse=True, dtype=tf.int32, constant=False)
    seq_len = tx.Input(init_value=2, shape=[], constant=False)
    assert seq_len.n_units == 0

    lookup = tx.Lookup(inputs, seq_size=seq_len, embedding_shape=[k, m])
    # concat = lookup.as_concat()

    inputs.value = seq1
    inputs()
    # set seq_len to 4
    seq_len.value = 4
    lookup_4 = lookup()
    # (batch, seq_len, embed_dim)
    assert lookup_4.numpy().shape == (1, 4, m)

    # set seq len to 3
    inputs.value = seq2
    seq_len.value = 3
    lookup_4 = lookup()
    # (batch, seq_len, embed_dim)
    assert lookup_4.numpy().shape == (2, 3, 3)


def test_lookup_sequence_sparse():
    input_dim = 10
    embed_dim = 3
    seq_size = 2
    batch_size = 3

    sparse_input = tf.SparseTensor([[0, 2], [1, 0], [2, 1]], [1, 1, 1], [3, input_dim])
    sparse_input_1d = tf.SparseTensor([[2], [0], [1]], [1, 1, 1], [input_dim])
    tensor_input = tx.Constant(sparse_input, input_dim)
    tensor_input_1d = tx.Constant(sparse_input_1d, input_dim)

    lookup = tx.Lookup(tensor_input, seq_size,
                       embedding_shape=[input_dim, embed_dim],
                       batch_size=batch_size,
                       batch_padding=False)

    lookup_padding = tx.Lookup(tensor_input, seq_size,
                               embedding_shape=[input_dim, embed_dim],
                               batch_size=batch_size,
                               batch_padding=True)

    lookup_1d = tx.Lookup(tensor_input_1d, seq_size,
                          embedding_shape=[input_dim, embed_dim],
                          batch_size=batch_size,
                          batch_padding=True)

    result = lookup()
    result_padding = lookup_padding()
    result_1d = lookup_1d()

    assert np.shape(result) == (2, seq_size, embed_dim)
    assert np.shape(result_padding) == (batch_size, seq_size, embed_dim)
    assert np.shape(result_1d) == (batch_size, seq_size, embed_dim)


def test_lookup_sparse_padding():
    """ Sparse Lookup Padding
    Lookup adds padding if seq_size is greater than the max row indice
    in the input SparseTensor

    """
    input_dim = 6
    embed_dim = 4
    seq_size = 3

    sparse_input = tf.SparseTensor(indices=[[0, 1], [0, 3],
                                            [1, 0]],
                                   values=[1, 1, 1],
                                   dense_shape=[2, input_dim])
    sparse_input = tx.Constant(sparse_input, input_dim)

    lookup = tx.Lookup(sparse_input,
                       seq_size=seq_size,
                       embedding_shape=[input_dim, embed_dim],
                       batch_size=None,
                       batch_padding=False)

    result = lookup()
    assert tf.sparse.to_dense(sparse_input()).shape == (2, input_dim)
    assert tx.tensor_equal(result[0][-1], tf.zeros([embed_dim]))


def test_lookup_sequence_bias():
    vocab_size = 4
    n_features = 3
    seq_size = 2

    inputs = tx.Input(n_units=seq_size, dtype=tf.int32)
    input_data = np.array([[2, 0], [1, 2], [0, 2]])
    lookup = tx.Lookup(input_layer=inputs,
                       seq_size=seq_size,
                       embedding_shape=[vocab_size, n_features],
                       add_bias=True)

    inputs.value = input_data
    v1 = lookup()
    assert np.shape(v1) == (np.shape(input_data)[0], seq_size, n_features)


def test_lookup_sequence_transform():
    vocab_size = 4
    embed_dim = 2
    seq_size = 2

    inputs = tx.Input(n_units=seq_size, dtype=tf.int32)
    input_data = np.array([[2, 0], [1, 2], [0, 2]])
    lookup = tx.Lookup(inputs,
                       seq_size=seq_size,
                       embedding_shape=[vocab_size, embed_dim],
                       add_bias=True)
    concat_lookup = lookup.as_concat()
    seq_lookup = lookup.permute_batch_time()

    assert hasattr(lookup, "seq_size")

    inputs.value = input_data

    v1 = lookup()
    v2 = concat_lookup()
    v3 = seq_lookup()

    assert np.shape(v1) == (np.shape(input_data)[0], seq_size, embed_dim)
    assert np.shape(v2) == (np.shape(input_data)[0], seq_size * embed_dim)

    assert np.shape(v3) == (seq_size, np.shape(input_data)[0], embed_dim)
    assert tx.tensor_equal(v1[:, 0], v3[0])


def test_reuse_dropout():
    x1 = tx.Constant(np.ones(shape=[2, 4]), dtype=tf.float32)
    x2 = tx.Activation(x1)
    drop1 = tx.Dropout(x2, probability=0.5, locked=True)

    assert len(drop1.inputs) == 2
    assert drop1.inputs[0] is x2
    assert drop1.inputs[-1] is drop1.layer_state.mask

    # shared state overrides mask?
    _, mask = tx.dropout(x2, return_mask=True)
    drop2 = drop1.reuse_with(x2, mask)

    assert len(drop2.inputs) == 2
    assert drop2.inputs[0] is x2
    assert drop2.inputs[-1] is drop2.layer_state.mask

    assert not tx.tensor_equal(drop1(), drop2())

    graph = tx.Graph.build(inputs=None, outputs=[drop1, drop2])

    out1, out2 = graph()
    assert tx.tensor_equal(out1, out2)

    drop1 = tx.Dropout(x2, probability=0.5)
    drop2 = drop1.reuse_with(x1)

    graph.eval(drop1, drop2)


def test_drop_lookup():
    """ Embedding Dropout
    TODO finish test
    """
    seq_size = 4
    vocab_size = 10
    embed_dim = 3
    input_data = tf.constant([[2, 0, 2, 0], [1, 2, 2, 3], [0, 3, 0, 2]])
    inputs = tx.Input(init_value=input_data, n_units=seq_size, dtype=tf.int32)
    lookup = tx.Lookup(inputs,
                       seq_size=seq_size,
                       embedding_shape=[vocab_size, embed_dim],
                       add_bias=True)

    tx.DropLookup(lookup, probability=0.5)


def test_residual():
    x1 = tx.Input([[1., 1., 1., 1.]], 4)
    x2 = tx.Input([[1., 1., 1., 1.]], 4)

    h1 = tx.FC(x1, 4, activation=tf.sigmoid)
    h2 = tx.FC(x1, 2, activation=tf.sigmoid)
    h3 = tx.FC(x2, 2, activation=tf.sigmoid)

    residual = tx.Residual(x1, h1)
    residual2 = tx.Residual(x1, h2)

    with pytest.raises(ValueError):
        tx.Residual(x1, h3)
        pytest.fail("ValueError Expected: invalid module x1 not connected to h3")

    assert tx.shape_equal(h1(), residual())
    assert not hasattr(residual, "projection")
    assert hasattr(residual2, "projection")
    assert len(residual.trainable_variables) == 0
    assert len(residual2.trainable_variables) == 1


def test_fully_connected():
    x1 = tx.Input(init_value=[[1., 1., 1., 1.]], n_units=4, dtype=tf.float32, constant=True)
    x2 = tx.Input(init_value=np.random.uniform(size=[2, 4]), dtype=tf.float32, n_units=4, constant=True)

    y1 = tx.FC(x1, 4, add_bias=True, activation=tf.sigmoid)

    y2 = tx.Linear(x1, 4, add_bias=True, weights=y1.linear.weights, bias=y1.linear.bias)
    a2 = tx.Activation(y2, fn=tf.sigmoid)

    w = y2.weights
    b = y2.bias

    assert y1.linear.weights is w
    assert y1.linear.bias is b

    x = x1()
    y = tf.matmul(x, w) + b
    a = tf.sigmoid(y)

    assert tx.tensor_equal(y2(), y)
    assert tx.tensor_equal(y1(), a)
    assert tx.tensor_equal(y1(), a2())
    assert tx.tensor_equal(a2(), a)

    y1 = y1.reuse_with(x2)
    y2 = y2.reuse_with(x2)

    assert y2.weights is w
    assert y2.bias is b

    assert y1.linear.weights is w
    assert y1.linear.bias is b


def test_conv1d():
    num_filters = 2
    input_dim = 4
    seq_size = 3
    batch_size = 2
    filter_size = 2

    filter_shape = [filter_size, input_dim, num_filters]

    x = tf.ones([batch_size, seq_size, input_dim])
    x_layer = tx.Constant(x, input_dim)

    filters = tf.ones(filter_shape)
    conv_layer = tx.Conv1D(x_layer, num_filters, filter_size, filters=filters)
    conv = tf.nn.conv1d(input=x,
                        filters=filters,
                        stride=1,
                        padding="SAME",
                        data_format="NWC")

    output = conv_layer()
    assert tx.tensor_equal(conv, output)
    assert tx.tensor_equal(tf.shape(conv_layer.filters),
                           [filter_size, input_dim, num_filters])
    assert tx.tensor_equal(tf.shape(output),
                           [batch_size, seq_size, num_filters])


def test_map_seq():
    n_features = 5
    embed_size = 4
    seq_size = 3
    batch_size = 2

    inputs = tx.Input(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
    lookup = tx.Lookup(inputs, seq_size=seq_size, embedding_shape=[n_features, embed_size])
    seq = lookup.permute_batch_time()

    n_units = 2
    linear_fn = tx.Linear.config(n_units=n_units)
    assert tx.tensor_equal(tf.shape(seq()), [seq_size, batch_size, embed_size])

    seq_map = tx.SeqMap(seq, n_units=2, layer_config=linear_fn)
    assert tx.tensor_equal(tf.shape(seq_map), [seq_size, batch_size, n_units])


def test_multihead_attention():
    """
    TODO check causality

    """
    n_features = 3
    embed_size = 128
    seq_size = 3
    batch_size = 2
    n_heads = 8

    inputs = tx.Constant(np.random.random([batch_size, seq_size]), n_units=seq_size, dtype=tf.int32)
    emb = tx.Lookup(inputs,
                    seq_size=seq_size,
                    embedding_shape=[n_features, embed_size])

    attention = tx.MHAttention(query=emb,
                               key=emb,
                               value=emb,
                               n_units=embed_size,
                               n_heads=n_heads,
                               causality=False,
                               attention_dropout=0.1,
                               regularized=False)

    assert len(attention.inputs) == 3

    # 3 "kernels" + bias
    assert len(attention.variables) == 3

    attention_reg = attention.reuse_with(emb, emb, emb, regularized=True)
    attention_2 = attention.reuse_with(emb, emb, emb, regularized=False)
    attention_causal = attention.reuse_with(emb, emb, emb, causality=True)

    attention_causal()

    result = attention()
    result_reg = attention_reg()
    result2 = attention_2()

    assert tx.shape_equal(result, result_reg)
    assert tx.tensor_equal(result, result2)

    vars1 = map(lambda v: v.ref(), attention.variables)
    vars2 = map(lambda v: v.ref(), attention_2.variables)

    assert set(vars1) == set(vars2)
