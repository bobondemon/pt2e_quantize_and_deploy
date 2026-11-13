# M00 — torch.export and the fx Graph

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m00_export_basics/](../en/m00_export_basics/)

Start with the overview, then open the four notebooks in order. On GitHub you can see the executed results directly; to change parameters and run it yourself, see "Running on Your Own Machine" below.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: why we need a graph, why everything on the graph is an ATen op, what a node is made of, the four graph-editing actions |
| [01_export_and_inspect.ipynb](01_export_and_inspect.ipynb) | M00-1: ExportedProgram and GraphModule, node meta, what ATen is |
| [02_eval_vs_train.ipynb](02_eval_vs_train.ipynb) | M00-2: how eval/train mode is written into the graph |
| [03_decomposition.ipynb](03_decomposition.ipynb) | M00-3: decomposition, and why PT2E annotates before decomposing |
| [04_fx_graph_editing.ipynb](04_fx_graph_editing.ipynb) | M00-4: the six graph-editing steps (insert, rewire, recompile, replace, constants, clean up) |

## Running on Your Own Machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); graphviz subgraphs need the system `dot` executable. Run from the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; only needed once for the whole project
uv run jupyter lab m00_export_basics/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything automatically:
press Shift+Enter cell by cell, or use the menu Run → Run All Cells. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and choose `.venv` as the kernel.
