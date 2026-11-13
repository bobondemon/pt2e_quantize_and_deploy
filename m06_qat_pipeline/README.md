# M06 — The QAT Training Pipeline

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m06_qat_pipeline/](../en/m06_qat_pipeline/)

Start with the overview, then open the notebooks in order (M06-5 and M06-6 are optional). You can see the executed results directly on GitHub; to change parameters and run them yourself, see "Running on your own machine" below.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: QAT winning back the accuracy PTQ loses at int4; definitions of fake quant and STE; the principle of the whole chapter, "the training graph = the graph after convert" |
| [01_qat_minimal.ipynb](01_qat_minimal.ipynb) | M06-1: the gradient of STE; a minimal QAT loop written out line by line; same shape as PTQ after convert |
| [02_train_eval_switch.ipynb](02_train_eval_switch.ipynb) | M06-2: the two switches to flip before evaluation (BN, observer); `allow_exported_model_train_eval` |
| [03_fused_fakequant_mismatch.ipynb](03_fused_fakequant_mismatch.ipynb) | M06-3: XNNPACK's default weight fused fake quant makes the training graph ≠ the convert graph; the scale formulas and the fix |
| [04_qat_vs_ptq_int4.ipynb](04_qat_vs_ptq_int4.ipynb) | M06-4: whether QAT beats PTQ at int4; compared with "PTQ after retraining" to confirm the benefit comes from quantization adaptation (5 seeds, about 30 seconds) |
| [05_learnable_fake_quant.ipynb](05_learnable_fake_quant.ipynb) | M06-5 (optional): how to plug in `LearnableFakeQuantize`, the gradient and trajectory of scale, the effect (about 2 minutes) |
| [06_weight_norm_qat.ipynb](06_weight_norm_qat.ipynb) | M06-6 (optional): QAT that keeps `weight_norm` to train g and v; the convert error with BN and the fix |

## Running on your own machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); the graphviz subgraphs need the system `dot` executable. Run from the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # create .venv; only needed once for the whole project
uv run jupyter lab m06_qat_pipeline/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything automatically:
press Shift+Enter cell by cell, or choose Run → Run All Cells from the menu. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
