# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, "mmWrt")))

from mmWrt import __version__ as mmWrt_ver

project = 'mmWrt'
copyright = '2023, matt-chv'
author = 'matt-chv'
release = mmWrt_ver

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# for specifics on dollarmath
# check https://myst-parser.readthedocs.io/en/v0.15.1/syntax/optional.html#syntax-math
extensions = ['myst_parser', 'sphinx.ext.napoleon', 'sphinx_markdown_builder',
              'nbsphinx','dollarmath','html_admonition']

templates_path = ['_templates']
exclude_patterns = []

add_module_names = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
