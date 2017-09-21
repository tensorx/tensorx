#!/bin/bash

# builds api documentation from scratch
sphinx-apidoc -f -o . ../tensorx
make clean html
