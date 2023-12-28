#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on 2022-Dec-31

@author: matt-chv
"""
import re
from os.path import abspath, join, pardir
from semver import VersionInfo
from setuptools import setup

fp_readme = abspath(join(__file__, pardir, "README.md"))
with open(fp_readme, "r", encoding="utf-8") as fi:
    long_description = fi.read()

with open("mmWrt/__init__.py", "r") as fi:
    package_version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fi.read(), re.MULTILINE
    ).group(1)

assert VersionInfo.is_valid(package_version)

setup(
    name='mmWrt',
    version=package_version,
    author='matt-chv',
    author_email="contact@matthieuchevrier.com",
    description='minimal raytracing code example for MIMO FMCW radar',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/matt-chv/mmWrt',
    project_urls={
        "Bug Tracker": "https://github.com/matt-chv/mmWrt/issues",
    },
    license='LICENSE',
    keywords='radar MIMO FMCW raytracing',
    packages=['mmWrt'],
    package_dir={'mmWrt': abspath(join(__file__, pardir, "mmWrt"))},
    python_requires='>=3.8',
    classifiers=[
                "Development Status :: 3 - Alpha",
                "Topic :: Utilities",
                "License :: OSI Approved :: MIT License",
                'Environment :: Console',
                'Intended Audience :: End Users/Desktop',
                'Intended Audience :: Developers',
                "Operating System :: OS Independent",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
                "Topic :: Utilities",
    ],
    install_requires=["numpy", "scipy", "matplotlib", "semver"],
    extra_require= {
        "dev": [
            "recommonmark",
            "pyroma",
            "darglint",
            "flake8",
            "pytest",
            "coverage",
            "tox",
            "sphinx",
            "nbsphinx",
            "myst-parser",
            "sphinx_markdown_builder" ]}
)
