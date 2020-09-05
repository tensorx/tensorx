# Installation
TensorX is written in pure python but it's made to complement [Tensorflow](https://tensorflow.org), so it **depends on 
Tensorflow**, which needs to be installed from the `tensorflow` package.
The reason for this is that you might want to install Tensorflow builds optimized for your machine (see 
[these](https://github.com/davidenunes/tensorflow-wheels)). Additionally, TensorX has **optional
dependencies** like `matplotlib` or `pygraphviz` for certain functionality.

## with pip
Install using `pip` with the following commands
```shell
pip install tensorflow 
pip install tensorx
```

You can install a [custom tensorflow wheel](https://github.com/davidenunes/tensorflow-wheels) using it's URL like so:
```shell
pip install https://github.com/davidenunes/tensorflow-wheels/releases/download/r2.3.cp38.gpu/tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl
```

## with Poetry
poetry add myprivaterepo --git ssh://git@github.com/myorganization/myprivaterepo.git

## Developer Installation
I made the project easy to install using 
[Poetry](https://python-poetry.org/) which manages the `virtualenv` for you. 
The git repository has a `pyproject.toml` and a `poetry.lock` files, which allows for the development installation 
to be reproducible (meaning Poetry will install the exact same versions of dependencies I use when developing TensorX).

See [Poetry documentation](https://python-poetry.org/docs/) for details on how to install it and run it. 
One you install it you can clone the repository and run create a development environment installation as follows:

````shell 
# Clone TensorX git repository
git clone https://github.com/davidenunes/tensorx.git
cd tensorx

# Install dependencies 
poetry install
````

Again, running `install` when a `poetry.lock file is present resolves and installs all dependencies that you listed `
in `pyproject.toml`, but Poetry uses the exact versions listed in poetry.lock`