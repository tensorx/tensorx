from distutils.core import setup

_VERSION = '0.2'

setup(
    name='tensorx',
    version=_VERSION,
    packages=['tensorx'],
    url='https://github.com/davidenunes/tensorx',
    license='Apache 2.0',
    author='Davide Nunes',
    author_email='davidelnunes@gmail.com',
    description='TensorX is a minimalistic utility library to '
                'build neural network models in TensorFlow',

    install_requires=[
        'tensorflow >= 1.3.0',
        'numpy >= 1.13'
    ]
)