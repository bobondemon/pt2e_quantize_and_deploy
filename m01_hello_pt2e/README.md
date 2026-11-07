# M01 — 第一次量化

從概覽開始讀，再依序打開三本 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：量化要回答哪三個問題、scale／zero point、observer、Q/DQ 圖 |
| [01_minimal_ptq.ipynb](01_minimal_ptq.ipynb) | M01-1：四個步驟走一遍，每一步圖怎麼變、精度掉多少 |
| [02_graph_diff.ipynb](02_graph_diff.ipynb) | M01-2：prepare／convert 改了圖的哪裡、quantizer 實際管了哪些 op |
| [03_config_matrix.ipynb](03_config_matrix.ipynb) | M01-3：per-channel、QAT、dynamic 三個開關各改變了什麼 |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m01_hello_pt2e/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
