# M05 — Debugging PTQ Accuracy

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m05_ptq_pipeline/](../en/m05_ptq_pipeline/)

Start with the overview, then open the notebooks in order (M05-5 is optional). You can view the executed results directly on GitHub; to change parameters and rerun them yourself, see "Running on Your Own Machine" below.

| File | Content |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: why we need an order of investigation; definitions of output SQNR, layerwise SQNR, and mixed precision |
| [01_ptq_full.ipynb](01_ptq_full.ipynb) | M05-1: PTQ on a trained model; which of accuracy and output SQNR shows quantization error; per-channel vs per-tensor |
| [02_calibration_study.ipynb](02_calibration_study.ipynb) | M05-2: number of calibration samples and the spread of results; whether BN affects results during calibration |
| [03_layerwise_sqnr.ipynb](03_layerwise_sqnr.ipynb) | M05-3: layerwise SQNR with the numeric debugger; how to read it; which graph to use as the reference |
| [04_mixed_precision.ipynb](04_mixed_precision.ipynb) | M05-4: excluding some modules from the Quantizer; which exclusions actually help; how error accumulates |
| [05_weight_norm_ptq.ipynb](05_weight_norm_ptq.ipynb) | M05-5 (optional): what `weight_norm` becomes after export; why it errors with BN; two fixes |

## Running on Your Own Machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`), and graphviz subgraphs need the system `dot` executable. From the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; only needed once for the whole project
uv run jupyter lab m05_ptq_pipeline/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything automatically:
press Shift+Enter cell by cell, or choose Run → Run All Cells from the menu. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
