![tensorx](tensorx.png?raw=true "TensorX")
-----------------
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Docs](https://readthedocs.org/projects/tensorx/badge/?version=latest)](http://tensorx.readthedocs.io/en/latest/?badge=latest)
[![dev build](https://travis-ci.org/davidenunes/tensorx.svg?branch=dev)](https://travis-ci.org/davidenunes/tensorx)

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
