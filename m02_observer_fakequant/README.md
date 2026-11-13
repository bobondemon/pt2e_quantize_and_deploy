# M02 — Observers and Fake Quant

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

Start with the overview, then open the three notebooks in order. GitHub shows the executed results directly; to change parameters and run them yourself, see "Running on Your Own Machine" below.

English notebooks: [../en/m02_observer_fakequant/](../en/m02_observer_fakequant/)

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: split prepare and convert apart and do everything by hand once; the steps of observer, FakeQuantize, and convert |
| [01_observer_zoo.ipynb](01_observer_zoo.ipynb) | M02-1: symmetric vs. affine, how MinMax / Histogram / MovingAverage choose the range, per-channel, PlaceholderObserver |
| [02_manual_insert.ipynb](02_manual_insert.ipynb) | M02-2: insert observers by hand without a Quantizer and compare with `prepare_pt2e`; switch to FakeQuantize and its two switches |
| [03_observer_to_qdq.ipynb](03_observer_to_qdq.ipynb) | M02-3: do `convert_pt2e` by hand: observer → q/dq, `DuplicateDQPass`, constant folding of the weight |

## Running on Your Own Machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); graphviz subgraphs need the system `dot` executable. Run from the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # create .venv, needed only once for the whole project
uv run jupyter lab m02_observer_fakequant/00_overview.ipynb
```

The last line starts JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything:
press Shift+Enter cell by cell, or use the menu Run → Run All Cells. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
