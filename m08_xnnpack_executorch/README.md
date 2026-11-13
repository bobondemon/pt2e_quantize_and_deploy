# M08 — XNNPACK / ExecuTorch: Turning the Q/DQ Graph into a `.pte`

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m08_xnnpack_executorch/](../en/m08_xnnpack_executorch/)

Start from the overview, then open the notebooks in order. You can see the execution results directly on GitHub; to change parameters and run it yourself, see "Running on your own machine" below.

This chapter follows Route A of the [M07 overview](../m07_representation_lowering/00_overview.ipynb); for which layer ExecuTorch and XNNPACK each sit at, read the first half of the M07 overview.
Numbers and speed depend on the platform. The outputs saved in the notebooks come from macOS arm64, torch 2.14.0, executorch 1.5.0; when you run them on your machine, the commentary is printed from what is actually measured.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: conv + relu through PT2E → lowering → `.pte` → runtime with the raw APIs; Edge dialect, partitioner, delegate, memory planning; where q/dq go during lowering; numerical differences from the Q/DQ graph |
| [01_executorch_vs_torchao_copy.ipynb](01_executorch_vs_torchao_copy.ipynb) | M08-1: how the two same-named `XNNPACKQuantizer`s differ; the static and dynamic specs; `set_module_name` has no effect on conv |
| [02_export_to_pte.ipynb](02_export_to_pte.ipynb) | M08-2: graphs before and after lowering, the nodes XNNPACK actually executes inside the delegate, `.pte` size, running with the ExecuTorch runtime, why changing the batch fails |
| [03_unsupported_ops.ipynb](03_unsupported_ops.ipynb) | M08-3: an op XNNPACK does not take splits the graph into two delegates, and fp32 is passed across the boundary |
| [04_latency.ipynb](04_latency.ipynb) | M08-4: latency of eager, the Q/DQ graph, the fp32 `.pte`, and the int8 `.pte`; the gain splits into switching runtime and quantizing |

## Running on your own machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); graphviz subgraphs need the system `dot` executable. This chapter additionally needs executorch, so `uv sync` needs `--group executorch`. Run from the project root:

```bash
brew install graphviz                  # Ubuntu: sudo apt install graphviz
uv sync --group executorch             # create .venv and install executorch
uv run jupyter lab m08_xnnpack_executorch/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything automatically:
press Shift+Enter cell by cell, or use the menu Run → Run All Cells. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
