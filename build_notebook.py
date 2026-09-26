"""Assemble self_play_pretraining.ipynb from nbsrc/ (the notebook is a build artifact).

nbsrc/NN_name.md   -> a markdown cell
nbsrc/NN_name.py   -> a code cell
A code cell whose first line is  # @writefile <path>  becomes  %%writefile <path>  followed by
the CURRENT contents of that file, so the notebook can never drift from the package.

    python build_notebook.py
"""
import glob
import json
import os

cells = []
for path in sorted(glob.glob("nbsrc/*")):
    src = open(path, encoding="utf-8").read().rstrip("\n")
    if path.endswith(".md"):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        continue
    first, _, rest = src.partition("\n")
    if first.startswith("# @writefile "):
        target = first.split(" ", 2)[2].strip()
        src = f"%%writefile {target}\n" + open(target, encoding="utf-8").read().rstrip("\n")
    else:
        compile(src, path, "exec")                    # every plain code cell must at least parse
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "accelerator": "GPU", "colab": {"provenance": []}}}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
json.dump(nb, open("self_play_pretraining.ipynb", "w", encoding="utf-8", newline="\n"), indent=1)
print(f"wrote self_play_pretraining.ipynb: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code)")
