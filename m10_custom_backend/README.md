# M10 — Custom Backend: Writing Hardware Constraints as a Quantizer

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m10_custom_backend/](../en/m10_custom_backend/)

Start with the overview, then open the notebooks in order. GitHub renders the saved outputs directly; to change parameters and rerun, see "Running on your own machine" below.

This chapter invents a VPU and stands on the backend author's side: write the hardware constraints as a Quantizer, check the Q/DQ graph, and fall back unsupported ops to CPU.
The VPU is fictional; the constraints are common requirements of this kind of hardware (general knowledge), and this project has no real device.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: the Q/DQ graphs of the same conv + relu on the VPU and on XNNPACK; which PT2E mechanism each of the eight constraints uses; why the hardware wants these constraints (the zero-point cross terms); the whole flow |
| [01_vpu_quantizer.ipynb](01_vpu_quantizer.ipynb) | M10-1: `MyVPUQuantizer` section by section; the cost of pow2 scales; `validate()` rejecting dynamic batch; which tensors in TinyConvNet are forced to share a scale |
| [02_constraint_checker.ipynb](02_constraint_checker.ipynb) | M10-2: the checker pass `check_vpu` that reads only the Q/DQ graph; which rules the XNNPACK graph violates; the requant right-shift amounts |
| [03_fallback_report.ipynb](03_fallback_report.ipynb) | M10-3: a minimal partitioner `partition`; the partitioned graph, boundaries and data moved (TinyConvNet, OpZooNet) |
| [vpu_quantizer.py](vpu_quantizer.py) | The backend itself (`VPU_OPS`, the Pow2 observers, `MyVPUQuantizer`, `vpu_ptq`). It is not a notebook and is not run on its own; the notebooks import it with `from vpu_quantizer import …` (Jupyter's working directory is the notebook's folder) |

## Running on your own machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); graphviz subgraphs need the system `dot` executable. From the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; needed once for the whole project
uv run jupyter lab m10_custom_backend/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything:
press Shift+Enter cell by cell, or choose Run → Run All Cells from the menu. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
