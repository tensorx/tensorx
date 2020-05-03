![tensorx](tensorx.png?raw=true "TensorX")
-----------------
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Docs](https://readthedocs.org/projects/tensorx/badge/?version=latest)](http://tensorx.readthedocs.io/en/latest/?badge=latest)

![](/assets/images/logo_full.svg)

**TensorX** is a high-level neural network library written in Python
that simplifies model specification, training, and execution using 
[TensorFlow](https://www.tensorflow.org/). It was designed for fast 
prototyping with minimum verbose and provides a set of modular 
components with a user-centric consistent API.

## Design Philosophy

It's design philosophy is somewhere between functional **dataflow 
computation graphs** and an **object-oriented** neural network **layers**. 
TensorX provides a set of components that can be debugged eagerly in an 
imperative fashion while taking full advantage of compiled computation 
graphs. Stateful neural network functions can be composed into larger 
layer graphs that can then be reused as new components in a "Monoidal" fashion.

## Summary

* **Neural Network** layer building blocks
* **Additional TensorFlow ops**  for tensor manipulation
* **Model class**:
    - simplified inference, training, and evaluation
    - Customizable training loop with a ``Callback`` system
    

**TensorX** is an free and open-source minimalistic high-level library to build, train, and run neural network models in [TensorFlow](https://github.com/tensorflow/tensorflow). This is currently under heavy development, as such I don't recommend it's usage yet. Tensorflow is approaching a new release and there will be a split in this library as well that will accomodate both static graphs and eager execution (the new default mode).

**Documentation** will be available at [tensorx.readthedocs.io](http://tensorx.readthedocs.io/en/latest/?badge=latest).


## Installation
TensorX is compatible with **Python 3.5+**. It depends on Tensorflow and pygraphviz but will not be shipping with tensorflow as a dependency. The reason for this is that tensorflow might come under different packages depending on whether or not the user is using GPU hardware. Moreover, you might want to use a custom tensorflow build. For these reasons Tensorflow should be installed separately from tensorx.

``` bash
pip install tensorflow 
# pip install tensorflow-gpu
pip install tensorx
```

**virtualenv**: much like tensorflow, virtualenv is recommended to install tensorx in its own environment to avoid interfering with system-wide packages.

## Getting Started
Documentation and tutorials will be coming soon

## Author
* **[Davide Nunes](https://github.com/davidenunes)**: get in touch [@davidelnunes](https://twitter.com/davidelnunes)

## License

[Apache License 2.0](LICENSE)
