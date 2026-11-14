# M03 — Annotations and the Quantizer

> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)

English notebooks: [../en/m03_quantizer_api/](../en/m03_quantizer_api/)

Start with the overview, then open the seven notebooks in order. You can see the executed results directly on GitHub; to change parameters and run them yourself, see "Running on your own machine" below.

| File | Contents |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | Overview: who decides where observers go; annotations, `QuantizationSpec` and the four specs, the Quantizer's three methods |
| [01_annotate_then_prepare.ipynb](01_annotate_then_prepare.ipynb) | M03-1: annotate by hand, and `prepare_pt2e` inserts observers according to the annotations; edge annotations and output annotations |
| [02_read_annotations.ipynb](02_read_annotations.ipynb) | M03-2: read the annotations XNNPACKQuantizer attaches; what fusion looks like in annotations; ops that were not annotated |
| [03_my_first_quantizer.ipynb](03_my_first_quantizer.ipynb) | M03-3: write a Quantizer that quantizes only conv2d using `node.target`; `_annotated` and `validate` |
| [04_shared_and_implicit_sharing.ipynb](04_shared_and_implicit_sharing.ipynb) | M03-4: `SharedQuantizationSpec`; when implicit sharing merges and when it chains |
| [05_derived_and_fixed_qparams.ipynb](05_derived_and_fixed_qparams.ipynb) | M03-5: `DerivedQuantizationSpec` for the int32 bias; `FixedQParamsQuantizationSpec` for sigmoid |
| [06_partial_quantization.ipynb](06_partial_quantization.ipynb) | M03-6: quantizing only part of a model, `ComposableQuantizer`, annotation order |
| [07_backend_config_api.ipynb](07_backend_config_api.ipynb) | M03-7: the config priority order of `X86InductorQuantizer`; the `set_module_name` that XNNPACKQuantizer silently ignores |

## Running on your own machine

The whole project shares one [uv](https://docs.astral.sh/uv/) environment (Python 3.12, in `.venv/`); graphviz subgraphs need the system `dot` executable. From the project root:

```bash
brew install graphviz          # Ubuntu: sudo apt install graphviz
uv sync                        # creates .venv; only needed once for the whole project
uv run jupyter lab m03_quantizer_api/00_overview.ipynb
```

The last line launches JupyterLab with the `.venv` Python and opens this chapter's overview in the browser. It only opens it and does not run anything:
press Shift+Enter cell by cell, or use the menu Run → Run All Cells. Open the other notebooks from the file browser on the left.
You can also open the `.ipynb` in VS Code and pick `.venv` as the kernel.
