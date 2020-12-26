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
# adding pre because tensorx is currently in alpha
pip install tensorx --pre
```

You can install a [custom tensorflow wheel](https://github.com/davidenunes/tensorflow-wheels) using it's URL like so:
```shell
pip install https://github.com/davidenunes/tensorflow-wheels/releases/download/r2.3.cp38.gpu/tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl
```

## with Poetry 
TensorX is easy to install using [Poetry](https://python-poetry.org/) which manages the dependencies and a `virtualenv` 
for you. See [Poetry documentation](https://python-poetry.org/docs/) for details on installation and usage. 
Once Poetry is installed, you can use it to install TensorX either from _PyPI_, or directly from _git_.

If you want to **create a new project** using poetry, simply run: 
```bash
poetry new myproject
cd myproject
```
if you already have a project, move to it's directory and run the following commands:
```bash
poetry add tensorflow
poetry add tensorx
# or, if you want to have tensorx as a dependency from the git repository
poetry add git+https://github.com/davidenunes/tensorx.git 
```

## Developer Installation
For a developer installation of TensorX, simply clone the git repository and install it with Poetry.
The git repository has a `pyproject.toml` and `poetry.lock` files. These allow for the 
installation to be reproducible --meaning that Poetry will install the exact versions of dependencies that were being 
used on a specific commit.

```shell 
# Clone TensorX git repository
git clone https://github.com/davidenunes/tensorx.git
cd tensorx

# Install dependencies 
poetry install

# You need Tensorflow which is not specified as a dependency
# to install it without adding it as a dependency simple run
poetry run pip install tensorflow
```
`poetry run` will run a given command inside the project current `virtualenv`
