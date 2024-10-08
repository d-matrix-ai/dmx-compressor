# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dmx-compressor'
copyright = '2023, d-Matrix'
author = '"Tristan Webb <twebb@d-matrix.ai>", "Xin Wang <xwang@d-matrix.ai>", "Michael Klachko <mklachko@d-matrix.ai>", "Sayeh Sharify <sayehs@d-matrix.ai>", "Wanzin Yazar <wyazar@d-matrix.ai>"'
release = '0.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary']

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'


import pathlib
import sys
import os
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.insert(0, os.path.abspath('../../src/dmx'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))
