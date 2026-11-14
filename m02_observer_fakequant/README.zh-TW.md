# M02 — observer 與 fake quant

> ### 🌐 [English](README.md) ｜ **繁體中文**

從概覽開始讀，再依序打開三本 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：把 prepare 與 convert 拆開，全部手做一次；observer、FakeQuantize、convert 的步驟 |
| [01_observer_zoo.ipynb](01_observer_zoo.ipynb) | M02-1：symmetric 與 affine、MinMax／Histogram／MovingAverage 怎麼選範圍、per-channel、PlaceholderObserver |
| [02_manual_insert.ipynb](02_manual_insert.ipynb) | M02-2：不用 Quantizer 手插 observer，與 `prepare_pt2e` 比對；換成 FakeQuantize 與兩個開關 |
| [03_observer_to_qdq.ipynb](03_observer_to_qdq.ipynb) | M02-3：手做 `convert_pt2e`：observer → q/dq、`DuplicateDQPass`、weight 的 constant folding |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m02_observer_fakequant/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
