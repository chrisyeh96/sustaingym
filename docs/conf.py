# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# sphinx.ext.viewcode relies on having the main package in the system path
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SustainGym'
copyright = '2023, Christopher Yeh'
author = 'Christopher Yeh'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # allow Markdown files instead of ReST
    # this also imports 'sphinx.ext.mathjax'
    'myst_parser',

    # parse Google style docstrings
    'sphinx.ext.napoleon',

    # add [source] link to API Reference
    'sphinx.ext.viewcode',

    # add copy button to code blocks
    'sphinx_copybutton',

    # automatically generate API Reference
    # this also imports 'sphinx.ext.autodoc'
    'autoapi.extension',
]

# allow using `single ticks` for code
default_role = 'any'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# templates_path = ['_templates']


# -- Extension configuration ---------------------------------------------------

# sphinx.ext.napoleon
# - document multiple return variables
napoleon_custom_sections = [("Returns", "params_style")]

# myst_parser
# - use $...$ for inline, $$...$$ for display, and \$ for an actual dollar sign
myst_enable_extensions = [
    "dollarmath"
]

# sphinx.ext.mathjax
# - add common LaTeX macros
mathjax3_config = {
    'tex': {
        'macros': {
            'E': "{\\mathbb{E}}",
            'R': "{\\mathbb{R}}",
            'Z': "{\\mathbb{Z}}",
        },
        # Use AMS numbering rules and support labeled equations.
        # See https://docs.mathjax.org/en/latest/input/tex/eqnumbers.html
        'tags': 'ams'
      }
}

# autoapi.extension
# - due to an existing issue (https://github.com/readthedocs/sphinx-autoapi/issues/298)
#   we do not use implicit namespace packages
autoapi_options = [
    'members', 'undoc-members', 'show-inheritance', 'show-module-summary',
    'imported-members'
]
autoapi_root = 'api'
autoapi_dirs = ['../sustaingym']
autoapi_add_toctree_entry = False
# autoapi_python_use_implicit_namespaces = True
autodoc_typehints = 'both'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# left-sidebar title
html_title = "SustainGym"

# include custom CSS
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_theme = 'furo'

# add edit button
# see https://pradyunsg.me/furo/customisation/edit-button/
html_theme_options = {
    "source_repository": "https://github.com/chrisyeh96/sustaingym/",
    "source_branch": "main",
    "source_directory": "docs/",
}
