# M00 — torch.export 與 fx 圖

從概覽開始讀，再依序打開四本 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：為什麼需要一張圖、為什麼圖上都是 ATen op、node 的組成、改圖的四個動作 |
| [01_export_and_inspect.ipynb](01_export_and_inspect.ipynb) | M00-1：ExportedProgram 與 GraphModule、node 的 meta、ATen 是什麼 |
| [02_eval_vs_train.ipynb](02_eval_vs_train.ipynb) | M00-2：eval／train 模式怎麼寫進圖裡 |
| [03_decomposition.ipynb](03_decomposition.ipynb) | M00-3：decomposition，以及 PT2E 為什麼在拆之前標註 |
| [04_fx_graph_editing.ipynb](04_fx_graph_editing.ipynb) | M00-4：六個改圖步驟（插入、改接、recompile、取代、常數、清理） |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m00_export_basics/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
