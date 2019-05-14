#!/usr/bin/env python
import os
from setuptools import find_packages, setup, Command
import codecs
import sys
from shutil import rmtree

about = {}

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "tensorx", "__version__.py")) as f:
    exec(f.read(), about)

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel upload")
    sys.exit()


class UploadCommand(Command):
    """Support setup.py publish."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except FileNotFoundError:
            pass
        self.status("Building Source distribution…")
        os.system("{0} setup.py sdist bdist_wheel".format(sys.executable))
        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")
        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")
        sys.exit()


install_requires = [

]

extras_require = {
    'graphviz': ['pygraphviz'],
    'tqdm': ['tqdm']
}

setup(
    name='tensorx',
    version=about["__version__"],
    packages=['tensorx', 'tensorx.data'],
    url='https://github.com/davidenunes/tensorx',
    license='Apache 2.0',
    author='Davide Nunes',
    author_email='mail@davidenunes.com',
    description='TensorX is a minimalistic utility library to '
                'build neural network models in TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={
        "": ["LICENSE"],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.6',
    cmdclass={"upload": UploadCommand},
    install_requires=install_requires
)
