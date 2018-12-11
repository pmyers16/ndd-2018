import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_output, call
from Cython.Build import cythonize
from sys import platform
import numpy


PACKAGE_NAME = 'patrick-sprint1'
DESCRIPTION = 'A sandbox repository to contain the code and output for the data analysis done in sprint1'
with open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = 'Patrick Myers'
AUTHOR_EMAIL = 'pmyers16@jhu.edu'
URL = 'https://github.com/pmyers16/ndd-2018'
MINIMUM_PYTHON_VERSION = 3, 4  # Minimum of Python 3.4

REQUIRED_PACKAGES = ["numpy>=1.14.5"]
VERSION = '0.0.1'


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    include_dirs=[numpy.get_include()],
    packages=find_packages()
)
~                                                                                                                     
~                            
