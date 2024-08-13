# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("./caqtus"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "caqtus"
copyright = "2023, Caqtus"
author = "Caqtus"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "sphinx.ext.inheritance_diagram",
    "nbsphinx",
]

autodoc_typehints = "signature"
autodoc_preserve_defaults = True
autodoc_member_order = "bysource"
maximum_signature_line_length = 80


autodoc_type_aliases = {
    "LaneFactory": "LaneFactory",
    "JSON": "JSON",
    "Step": "Step",
    "AnalogValue": "AnalogValue",
    "Image": "Image",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "PySide6": ("https://doc.qt.io/qtforpython-6", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["reference/device/sequencer/sequencer_instruction_example.ipynb"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "navigation_depth": 8,
}
