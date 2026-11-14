# M01 — Your First Quantization

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m01_hello_pt2e/](../en/m01_hello_pt2e/)

Start with the overview, then open the three notebooks in order. You can see the executed results directly on GitHub; to change parameters and run them yourself, see "Running on Your Own Machine" below.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: the three questions quantization has to answer, scale / zero point, observers, the Q/DQ graph |
| [01_minimal_ptq.ipynb](01_minimal_ptq.ipynb) | M01-1: the four steps end to end, how the graph changes at each step and how much accuracy drops |
| [02_graph_diff.ipynb](02_graph_diff.ipynb) | M01-2: what prepare / convert change in the graph, which ops the quantizer actually covers |
| [03_config_matrix.ipynb](03_config_matrix.ipynb) | M01-3: what the three switches per-channel, QAT, and dynamic each change |

## Running on Your Own Machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`), and graphviz subgraphs need the system `dot` executable. Run from the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; only needed once for the whole project
uv run jupyter lab m01_hello_pt2e/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything automatically:
press Shift+Enter cell by cell, or use the menu Run → Run All Cells. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick the `.venv` kernel.
