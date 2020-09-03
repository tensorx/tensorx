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


<!--[![Docs](https://readthedocs.org/projects/tensorx/badge/?version=latest)](http://tensorx.readthedocs.io/en/latest/?badge=latest)-->

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
