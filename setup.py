from distutils.core import setup

_VERSION = '0.4'

install_requires = [
    'tensorflow >= 1.12.0'
]

setup(
    name='tensorx',
    version=_VERSION,
    packages=['tensorx'],
    url='https://github.com/davidenunes/tensorx',
    license='Apache 2.0',
    author='Davide Nunes',
    author_email='mail@davidenunes.com',
    description='TensorX is a minimalistic utility library to '
                'build neural network models in TensorFlow',

    python_requires='>=3.6',
    install_requires=install_requires
)
