## 

<p align="center">
<img src="https://raw.githubusercontent.com/davidenunes/tensorx/master/docs/theme/assets/images/logo_full.svg" width="80%" alt="Tensor X Logo">
</p>

<p align="center">
  <a href="http://www.apache.org/licenses/LICENSE-2.0.html">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2 Licence"/>
  </a>
  <a href="https://travis-ci.org/github/davidenunes/tensorx">
    <img src="https://travis-ci.org/davidenunes/tensorx.svg" alt="Travis CI"/>
  </a>
  <a href="https://pypi.org/project/tensorx">
    <img src="https://img.shields.io/pypi/v/tensorx.svg" alt="Python Package Index"/>
  </a>
  <a href="https://pypistats.org/packages/tensorx">
    <img src="https://img.shields.io/pypi/dm/tensorx.svg" alt="Downloads"/>
  </a>
</p>


**TensorX** is a high-level deep neural network library written in Python
that simplifies model specification, training, and execution using 
[TensorFlow](https://www.tensorflow.org/). It was designed for fast 
prototyping with minimum verbose and provides a set of modular 
components with a user-centric consistent API.

## Design Philosophy

TensorX aims to be **simple but sophisticated** without a code base plagued by unnecessary abstractions and 
over-engineering and **without sacrificing performance**. It uses Tensorflow without hiding it completely behind a new namespace, it's mean to be a complement
instead of a complete abstraction. The design mixes functional **dataflow computation graphs** with **object-oriented** 
neural network **layer** building blocks that are **easy to add to and extend**. 

## Feature Summary

* **Neural Network** layer building blocks like `Input`, `Linear`, `Lookup`;
* **New TensorFlow ops**:  `gumbel_top`, `logit`, `sinkhorn`, etc;
* **`Graph` Utils**: allow for validation and compilation of layer graphs;
* **`Model` Class**: for easy _inference_, _training_, and _evaluation_;
* **Training Loop**: easily customizable with a ``Callback`` system;

## Installation
TensorX is written in pure python but **depends on Tensorflow**, which needs to be installed from the `tensorflow` package.
The reason for this is that you might want to install Tensorflow builds optimized for your machine (see 
[these](https://github.com/davidenunes/tensorflow-wheels)). Additionally, TensorX has **optional
dependencies** like `matplotlib` or `pygraphviz` for certain functionality.

## Pip installation
Install using `pip` with the following commands:

```shell
pip install tensorflow 
pip install tensorx
```

For more details about the installation, check the [documentation](https://tensorx.org/start/install/).

## Test your installation
```python
import tensorflow as tf
import tensorx as tx
```


## Documentation
For details about TensorX API, tutorials, and other documentation, see [https://tensorx.org](https://tensorx.org).
You can help by trying the project out, reporting bugs, suggest features, and by letting me know what you think. 
If you want to help, please read the [contribution guide](https://tensorx.org/contributing/).


## Author
* **[Davide Nunes](https://github.com/davidenunes)**: get in touch [@davidelnunes](https://twitter.com/davidelnunes) 
or by [e-mail](mailto:davidenunes@pm.me)

## License

[Apache License 2.0](LICENSE)
