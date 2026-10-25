"""Sphinx configuration for the guide-rs Python documentation."""

import shutil
from importlib import metadata
from pathlib import Path

project = "guide-rs"
author = "Brandon Greenwell"
copyright = "2026, Brandon Greenwell"
release = metadata.version("guide-rs")

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]
exclude_patterns = ["_build", "jupyter_execute"]
html_theme = "furo"
html_title = "guide-rs"

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_xref_param_type = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# The notebooks are committed with outputs in python/examples; render them
# as they are rather than re-running them during the build.
nb_execution_mode = "off"
examples = Path(__file__).parent / "examples"
examples.mkdir(exist_ok=True)
for notebook in (Path(__file__).parents[1] / "examples").glob("*.ipynb"):
    shutil.copyfile(notebook, examples / notebook.name)
