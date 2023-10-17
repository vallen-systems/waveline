# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
from importlib import metadata

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "waveline"
copyright = f"{date.today().year}, Vallen Systeme GmbH"  # noqa
author = "Lukas Berbuer (Vallen Systeme GmbH)"
release = metadata.version("waveline")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = [".rst", ".md"]

autosummary_generate = True
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "serial": ("https://pyserial.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

default_role = "autolink"
add_function_parentheses = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_static_path = ["_static"]
