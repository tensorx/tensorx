## 
![](docs/theme/assets/images/logo_full.svg)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![build](https://travis-ci.org/davidenunes/tensorx.svg?branch=master)](https://travis-ci.org/davidenunes/tensorx)
<!--[![Docs](https://readthedocs.org/projects/tensorx/badge/?version=latest)](http://tensorx.readthedocs.io/en/latest/?badge=latest)-->

**TensorX** is a high-level deep neural network library written in Python
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
    

<!--**Documentation** will be available at [tensorx.readthedocs.io](http://tensorx.readthedocs.io/en/latest/?badge=latest).-->
## Documentation
Documentation will be available soon, I'm switching from sphinx to markdown files and mkdocs using my custom generator ([mkgendocs](https://pypi.org/project/mkgendocs/)) to create the API pages for Google style docstrings. I'll also make available guides for how to contribute documentation. For now, a tentative version of a contribution guide can be found in [CONTRIBUTING.md](https://github.com/davidenunes/tensorx/blob/master/CONTRIBUTING.md).

## Installation
TensorX is compatible with **Python 3.7+**. It depends on Tensorflow and pygraphviz but will not be shipping with tensorflow as a dependency. The reason for this is that tensorflow might come under different packages depending on whether or not the user is using GPU hardware. Moreover, you might want to use a custom tensorflow build. For these reasons Tensorflow should be installed separately from tensorx.

``` bash
pip install tensorflow 
pip install tensorx
```
The only reason I don't add Tensorflow to the dependencies of TensorX is because one might want to install custom TF wheels (optimized for
certain hardware, etc)

## Getting Started
Documentation and tutorials will be coming soon

## Author
* **[Davide Nunes](https://github.com/davidenunes)**: get in touch [@davidelnunes](https://twitter.com/davidelnunes) or by [e-mail](mailto:davidenunes@pm.me)

## License

[Apache License 2.0](LICENSE)
