#!/usr/bin/env python3

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring

import codecs
import os
from setuptools import setup, find_packages, Extension

import numpy as np

################################################################################

def read_rel(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as ifh:
        return ifh.read()


def get_version(rel_path):
    for line in read_rel(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


################################################################################

# Comma separated list of names and emails
authors = "Josep Pon Farreny"


emails = "jponfarreny@gmail.com"


# Short description
description = "Utility functions to extract data from documents"


# Long description
with open("README.md", encoding="utf-8") as ifh:
    readme = ifh.read()


# Requirements
with open("requirements.txt", encoding="utf-8") as ifh:
    requirements = [x for x in map(str.strip, ifh.read().splitlines())
                    if x and not x.startswith("#")]


# Additional (keyword) arguments
kwargs = {
    "entry_points": {
        "console_scripts": []
    }
}

####

module_name = "nst"
version = get_version(os.path.join(module_name, "__init__.py"))


################################################################################

setup(
    name=module_name,
    version=version,
    description=description,
    long_description=readme,
    author=authors,
    author_email=emails,
    url="https://github.com/jponf/neural-style-transfer",
    license="",
    keywords=["neural", "style", "transfer"],
    install_requires=requirements,
    packages=find_packages(),
    package_data={},
    include_package_data=True,  # include MANIFEST.in data
    platforms="any",
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    **kwargs
)
