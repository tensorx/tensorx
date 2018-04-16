![tensorx](tensorx.png?raw=true "TensorX")
-----------------
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Docs](https://readthedocs.org/projects/tensorx/badge/?version=latest)](http://tensorx.readthedocs.io/en/latest/?badge=latest)
[![dev build](https://travis-ci.org/davidenunes/tensorx.svg?branch=dev)](https://travis-ci.org/davidenunes/tensorx)

**TensorX** is an free and open-source minimalistic high-level library to build, train, and run neural network models in [TensorFlow](https://github.com/tensorflow/tensorflow). 
This library aims to provide simple, extensible, and modular components, without unnecessary levels of abstraction or verbose.

You can read the **documentation** on [tensorx.readthedocs.io](http://tensorx.readthedocs.io/en/latest/?badge=latest).

## Development

Both this library and its documentation are a work in progress, any input is welcome. 
You can follow the project on [Github](https://github.com/davidenunes/tensorx) and read its 
documentation on [ReadTheDocs](http://tensorx.readthedocs.io/en/latest/?badge=latest). I’m focusing 
on features I myself use in my research, so I’ll add components as I need them. 
If more people get interested in the project, I’ll create some contribution guidelines.


## Installation
TensorX is compatible with **Python 3.5+**. You can install it directly from this repository using `pip` as follows:

``` bash
sudo pip3 install --upgrade git+https://github.com/davidenunes/tensorx.git
```

This will install the `tensorx` module along with all the required dependencies. If you wish to install a particular version of
`tensorflow` (or build it from source) you can either install it first, or install `tensorx` with the ``--no-dependencies``
switch and install `tensorflow` afterwards. Instructions on how to install TensorFlow
can be found [here](<https://www.tensorflow.org/install/>).

**virtualenv**: much like tensorflow, virtualenv is recommended to install tensorx in its own environment to avoid interfering with system-wide packages.

## Getting Started
Coming soon.
```python 
import tensorx as tx
```

## Philosophy

* **Consistent API**: simple intuitive API focused on modular neural networks with multiple layers. 

* **Pragmatic Code**: verbose-free code is more readable, reproducible,
  and easier to debug and experiment with. Make it easy to use for
  common use cases.

* **Transparency**: the main goal is not to replace the use of
  TensorFlow or hide it behind abstractions, but to complement it with
  easy-to-use modular API to create and manipulate tensors.  

* **Focus**: this is not a library to create every single "Deep
  Learning" model one might read about. Its about taking advantage of
  TensorFlow flexibility while compensating for some of its
  shortcomings.

## Author
* **[Davide Nunes](https://github.com/davidenunes)**: get in touch [@davidelnunes](https://twitter.com/davidelnunes)

## License

[Apache License 2.0](LICENSE)
