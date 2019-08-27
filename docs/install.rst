Installation
============

Requirements
------------
    * **python3**: the ``tensorx`` package supports Python version **3.7+**.
    * **virtualenv**: `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ is recommended to install ``tensorx`` in its own environment.

Dependencies
------------

* `tensorflow <https://www.tensorflow.org>`_ : ``tensorflow`` or ``tensorflow-gpu``: not declared in the package setup because this depends on the version of TensorFlow you wish to use with TensorX.

* `numpy <https://www.numpy.org/>`_ : this already a dependency of TensorFlow

* `graphviz <https://www.tensorflow.org>`_ *[Optional]*: optional dependency for quick network diagram visualization


Via Python Package
------------------
Install the package (or add it to your ``Pipfile`` or ``requirements.txt`` file). TensorX depends
on ``tensorflow`` package, so this should be installed first.

.. code-block:: bash

    pip install tensorflow # or tensorflow-gpu
    pip install tensorx



