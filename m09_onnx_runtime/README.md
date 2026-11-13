# M09 — ONNX Runtime: QDQ ONNX and the Rewrites at Session Creation

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m09_onnx_runtime/](../en/m09_onnx_runtime/)

Start with the overview, then open the notebooks in order. You can see the executed results directly on GitHub; to change parameters and run them yourself, see "Running on Your Own Machine" below.

This chapter follows Route B of the [M07 overview](../m07_representation_lowering/00_overview.ipynb); for which layer ONNX Runtime and the Execution Provider each sit in, read the first half of the M07 overview first.
The graph ORT rewrites depends on the platform (x86 differs from arm). The outputs stored in the notebooks come from macOS arm64, torch 2.14.0, onnxruntime 1.30.0; when you run them on your machine, the explanations are printed according to what is actually measured.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: conv + relu through PT2E → QDQ ONNX → ORT with the raw APIs; what each of the five graph optimization levels does in the ORT 1.30 source; the three groups of QDQ node fusion; platform-dependent rewrites; comparison with M08; CPU vs. VPU/NPU |
| [01_onnx_qdq_export.ipynb](01_onnx_qdq_export.ipynb) | M09-1: exporting the Q/DQ graph as QDQ ONNX, what each op becomes; what the file stores; dynamic batch has to be stated again |
| [02_ort_qdq_fusion.ipynb](02_ort_qdq_fusion.ipynb) | M09-2: what TinyConvNet is rewritten into at each of the five levels; int32 bias; why Relu disappears; `ReduceMean` stays in fp32 |
| [03_ort_quant_compare.ipynb](03_ort_quant_compare.ipynb) | M09-3: ORT's `quantize_static` vs PT2E; QDQ format vs QOperator format |

## Running on Your Own Machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); the graphviz subgraphs need the system `dot` executable. `onnx`, `onnxscript`, and `onnxruntime` are main dependencies, so no separate install is needed. Run from the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; needed only once for the whole project
uv run jupyter lab m09_onnx_runtime/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything automatically:
press Shift+Enter cell by cell, or choose Run → Run All Cells from the menu. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick the `.venv` kernel.
