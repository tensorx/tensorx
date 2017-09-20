Installation
############

TensorX only depends on `TensorFlow <https://www.tensorflow.org/>`_ and `NumPy <http://www.numpy.org/>`_. In any case
this library can be installed using `pip` with the `setup.py` declaring all the dependencies and version requirements.

Requirements
============

    * **python**: `tensorx` supports python version **3.5+**.
    * **pip**: python includes _pip_ for installing additional modules that are not shipped with your operating system.
    * **virtualenv**: `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ is recommended to install `tensorx` in its own environment to avoid interfering with system-wide packages.

Pip Installation
================

You can install `tensorx` directly using `pip` as follows:

.. code-block:: bash

    sudo pip3 install --upgrade git+https://github.com/davidenunes/tensorx.git


This will install all `tensorx` along with all the required dependencies. If you wish to install a particular version of
`tensorflow` (or build it from source) you can either install it first, or install `tensorx` with the ``--no-dependencies``
switch and install `tensorflow` afterwards. For instructions on how to install TensorFlow
see `this <https://www.tensorflow.org/install/>`_.


