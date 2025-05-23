# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# Define the canonical URL if you are using a custom domain on Read the Docs
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True  # type: ignore[reportPossiblyUnboundVariable]

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
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "nbsphinx",
]

templates_path = ["_templates"]

autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
autodoc_member_order = "bysource"
maximum_signature_line_length = 80
autosectionlabel_prefix_document = True


autodoc_type_aliases = {
    "LaneFactory": "LaneFactory",
    "JSON": "JSON",
    "Step": "Step",
    "AnalogValue": "AnalogValue",
    "Image": "Image",
    "Data": "Data",
    "Parameter": "Parameter",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "PySide6": ("https://doc.qt.io/qtforpython-6", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

exclude_patterns = [
    "reference/device/sequencer/sequencer_instruction_example.ipynb",
    "_templates",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "navigation_depth": 8,
}

rst_prolog = """
.. |project| replace:: Condetrol
.. |sequence| replace:: :ref:`sequence <concepts_sequence>`
.. |shot| replace:: :ref:`shot <concepts_shot>`
.. |expression| replace:: :ref:`expression <concepts_expression>`
"""
