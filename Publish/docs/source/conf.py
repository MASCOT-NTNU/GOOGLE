# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# html_baseurl = '/GOOGLE/Publish/docs/build/html'

import sphinx_readable_theme
html_theme = 'readable'
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GOOGLE'
copyright = '2023, Yaolin Ge'
author = 'Yaolin Ge'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# autodoc_mock_imports = ['numpy', 'scipy', 'pandas', 'shapely', 'matplotlib', 'joblib', 'tqdm', 'geopandas', 'sklearn']
autodoc_mock_imports = ['numpy', 'shapely', 'pandas', 'matplotlib', 'scipy']

templates_path = ['_templates']
exclude_patterns = []

# extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_jekyll_builder']
# exclude_patterns = [
#     'build/*'
# ]

html_static_path = ['static']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']
