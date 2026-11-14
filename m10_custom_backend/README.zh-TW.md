# M10 — 自訂後端：把硬體約束寫成 Quantizer

> ### 🌐 [English](README.md) ｜ **繁體中文**

從概覽開始讀，再依序打開 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

這一章虛構一顆 VPU，站在後端作者那一側：把硬體約束寫成 Quantizer、檢查 Q/DQ 圖、把不支援的 op 切回 CPU。
VPU 是虛構的；約束是這類硬體常見的要求（一般說法），本專案沒有實機。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：同一個 conv + relu 在 VPU 與 XNNPACK 的 Q/DQ 圖；八條約束各用 PT2E 的哪個機制；硬體為什麼要這些約束（zero point 的交叉項）；整條流程 |
| [01_vpu_quantizer.ipynb](01_vpu_quantizer.ipynb) | M10-1：`MyVPUQuantizer` 逐段說明；pow2 scale 的代價；`validate()` 擋下動態 batch；TinyConvNet 哪些 tensor 被迫同 scale |
| [02_constraint_checker.ipynb](02_constraint_checker.ipynb) | M10-2：只讀 Q/DQ 圖的檢查 pass `check_vpu`；XNNPACK 的圖違反哪些規則；requant 的右移位數 |
| [03_fallback_report.ipynb](03_fallback_report.ipynb) | M10-3：極簡 partitioner `partition`；分區圖、邊界與搬移量（TinyConvNet、OpZooNet） |
| [vpu_quantizer.py](vpu_quantizer.py) | 後端本體（`VPU_OPS`、Pow2 observer、`MyVPUQuantizer`、`vpu_ptq`）。不是 notebook，也不單獨執行；notebook 用 `from vpu_quantizer import …` 匯入（Jupyter 的工作目錄就是 notebook 所在的資料夾） |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m10_custom_backend/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
