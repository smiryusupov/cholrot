# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.metadata
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "cholrot"
author = "Shohruh Miryusupov"
release = importlib.metadata.version("cholrot")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = "cholrot"

# PDF manual via Sphinx LaTeX builder.
latex_engine = "xelatex"
latex_documents = [
    ("index", "cholrot-manual.tex", "cholrot Manual", author, "manual"),
]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
\usepackage{amsmath,amssymb}
\usepackage{microtype}
""",
}
