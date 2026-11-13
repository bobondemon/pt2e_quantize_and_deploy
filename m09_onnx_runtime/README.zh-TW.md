# M09 — ONNX Runtime：QDQ ONNX 與建 session 時的改寫

> ### 🌐 [English](README.md) ｜ **繁體中文**

從概覽開始讀，再依序打開 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

這一章走 [M07 概覽](../m07_representation_lowering/00_overview.ipynb)的路線 B；ONNX Runtime、Execution Provider 各在哪一層，先讀 M07 概覽的前半。
ORT 改寫出來的圖和平台有關（x86 與 arm 不同）。notebook 裡存的輸出來自 macOS arm64、torch 2.14.0、onnxruntime 1.30.0；在你的機器上執行時，說明會依實測印出。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：conv + relu 用原始 API 走完 PT2E → QDQ ONNX → ORT；五個圖最佳化等級在 ORT 1.30 原始碼裡各做什麼；QDQ node fusion 的三組；與平台有關的改寫；和 M08 的對照；CPU 與 VPU／NPU 的差別 |
| [01_onnx_qdq_export.ipynb](01_onnx_qdq_export.ipynb) | M09-1：Q/DQ 圖匯出成 QDQ ONNX，每個 op 變成什麼；檔案裡存了什麼；動態 batch 要再講一次 |
| [02_ort_qdq_fusion.ipynb](02_ort_qdq_fusion.ipynb) | M09-2：TinyConvNet 在五個等級各被改成什麼；int32 bias；Relu 為什麼消失；`ReduceMean` 留在 fp32 |
| [03_ort_quant_compare.ipynb](03_ort_quant_compare.ipynb) | M09-3：ORT 的 `quantize_static` vs PT2E；QDQ format 與 QOperator format |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。`onnx`、`onnxscript`、`onnxruntime` 是主依賴，不需要另外安裝。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m09_onnx_runtime/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
