# M04 — Conv + BN Folding

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m04_conv_bn_folding/](../en/m04_conv_bn_folding/)

Start with the overview, then open the three notebooks in order. On GitHub you can see the execution results directly; to change parameters and run it yourself, see "Running on your own machine" below.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: why fold BN, the folding formulas, at which step PTQ and QAT each fold; why QAT's fake quant quantizes $W_{inf}$, the slow version and the approximate version |
| [01_bn_folding_trace.ipynb](01_bn_folding_trace.ipynb) | M04-1: PTQ's BN disappears at `prepare_pt2e`; how weight and bias are rewritten; what it looks like after convert when there is an extra quantization point between conv and relu |
| [02_qat_graph_structure.ipynb](02_qat_graph_structure.ipynb) | M04-2: what each op of the conv + BN subgraph from `prepare_qat_pt2e` computes; how convert folds, and when folding fails |
| [03_qat_vs_article.ipynb](03_qat_vs_article.ipynb) | M04-3: PT2E's QAT subgraph is bit-for-bit identical to PyTorch's `_forward_approximate` (approximate version); how much it differs from the whitepaper's slow version |

## Running on your own machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`), and the graphviz subgraphs need the system `dot` executable. Run from the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; only needed once for the whole project
uv run jupyter lab m04_conv_bn_folding/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything automatically:
press Shift+Enter cell by cell, or use the menu Run → Run All Cells. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
