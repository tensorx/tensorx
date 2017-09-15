.. title:: Home

.. TensorX documentation master file, created by
   sphinx-quickstart on Thu Aug 24 23:56:40 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: _static/img/logo.png


TensorX is a minimalistic high-level library to build and train neural network
models in `TensorFlow <https://www.tensorflow.org/>`_. It was developed to
design simple neural network models with minimum verbose. 

Development
===========
Both this library and its documentation are a work in progress, any input is
welcome. You can follow the project on `Github
<https://github.com/davidenunes/tensorx>`_ and read its documentation on
`readthedocs <http://tensorx.readthedocs.io>`_. I'm focusing on features I myself
use in my research, so I'll add components as I need them. If more people get
interested in the project, I'll create some contribution guidelines.

Philosophy
==========

        * **Consistent API**: `tensorx` is designed to have a simple intuitive
          API focused on modular neural networks with multiple layers. 

        * **Pragmatic Code**: verbose-free code is more readable, reproducible,
          and easier to debug and experiment with. Make it easy to use for
          common use cases. Do not overwhelm users with features they will not
          use.

        * **Transparency**: the main goal is not to replace the use of
          `TensorFlow` or hide it behind abstractions, but to complement it with
          easy-to-use modular API to create and manipulate tensors.  

        * **Focus**: this is not a library to create every single "Deep
          Learning" model one might read about. Its about taking advantage of
          `TensorFlow` flexibility while compensating for some of its
          shortcomings.


.. toctree::
   :caption: Introduction
   :maxdepth: 2
   
   install 
   getting-started

.. toctree::
   :caption: API 
   :maxdepth: 2
   :hidden:
   
   api

