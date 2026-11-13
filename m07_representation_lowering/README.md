# M07 — Backend Deployment Overview: After the Q/DQ Graph

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m07_representation_lowering/](../en/m07_representation_lowering/)

Start with the overview, then open the notebooks in order. You can see the executed results directly on GitHub; to change parameters and run them yourself, see "Running on your own machine" below.

The numbers in this chapter depend on the platform. The outputs stored in the notebooks come from macOS arm64, torch 2.14.0, an environment without mkldnn; when you run them on your machine, the notes under the tables are printed according to the measured results.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview (the start of M07 to M10): the first half is background shared by backend deployment: the layers of inference framework, hardware extension unit, and kernel library, the three routes, and why the Quantizer must match the backend; the second half covers the "dq → op → q" units on the Q/DQ graph, what reference representation, lowering, and Inductor each are and how they relate, and the rules for comparing backend results |
| [01_fp32_numeric_floor.ipynb](01_fp32_numeric_floor.ipynb) | M07-1: how much the fp32 output differs without quantization when only the kernel changes; seeing kernels with the profiler; switching by hand with `torch.backends` |
| [02_representation_and_lowering.ipynb](02_representation_and_lowering.ipynb) | M07-2: what op the same Q/DQ graph becomes through reference representation, `lower_pt2e_quantized_to_x86`, and `torch.compile` |

## Running on your own machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); graphviz subgraphs need the system `dot` executable. From the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; only needed once for the whole project
uv run jupyter lab m07_representation_lowering/00_overview.ipynb
```

The last line launches JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything:
press Shift+Enter cell by cell, or use the menu Run → Run All Cells. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
