# M08 — XNNPACK／ExecuTorch：把 Q/DQ 圖變成 `.pte`

> ### 🌐 [English](README.md) ｜ **繁體中文**

從概覽開始讀，再依序打開 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

這一章走 [M07 概覽](../m07_representation_lowering/00_overview.ipynb)的路線 A；ExecuTorch、XNNPACK 各在哪一層，先讀 M07 概覽的前半。
數值與速度和平台有關。notebook 裡存的輸出來自 macOS arm64、torch 2.14.0、executorch 1.5.0；在你的機器上執行時，說明會依實測印出。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：conv + relu 用原始 API 走完 PT2E → lowering → `.pte` → runtime；Edge dialect、partitioner、delegate、memory planning；lowering 時 q/dq 去哪了；和 Q/DQ 圖的數值差異 |
| [01_executorch_vs_torchao_copy.ipynb](01_executorch_vs_torchao_copy.ipynb) | M08-1：兩份同名的 `XNNPACKQuantizer` 差在哪；static 與 dynamic 的規格；`set_module_name` 對 conv 不生效 |
| [02_export_to_pte.ipynb](02_export_to_pte.ipynb) | M08-2：lowering 前後的圖、delegate 裡 XNNPACK 實際執行的 node、`.pte` 的大小、用 ExecuTorch runtime 執行、換 batch 為什麼失敗 |
| [03_unsupported_ops.ipynb](03_unsupported_ops.ipynb) | M08-3：XNNPACK 不收的 op 把圖切成兩個 delegate，邊界上傳 fp32 |
| [04_latency.ipynb](04_latency.ipynb) | M08-4：eager、Q/DQ 圖、fp32 `.pte`、int8 `.pte` 的 latency；收益分成換 runtime 與量化兩段 |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。這一章另外需要 executorch，所以 `uv sync` 要加 `--group executorch`。在專案根目錄執行：

```bash
brew install graphviz                  # Ubuntu：sudo apt install graphviz
uv sync --group executorch             # 建立 .venv，並裝入 executorch
uv run jupyter lab m08_xnnpack_executorch/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
