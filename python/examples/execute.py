"""Executes the example notebooks in place, refreshing their outputs.

Run from python/: uv run --group examples python examples/execute.py
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

here = Path(__file__).parent
for path in sorted(here.glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=900, resources={"metadata": {"path": str(here)}}).execute()
    nbformat.write(notebook, path)
    print(f"executed {path.name}")
