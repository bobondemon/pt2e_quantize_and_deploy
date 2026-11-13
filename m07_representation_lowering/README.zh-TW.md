# M07 — 後端部署總覽：Q/DQ 圖之後

從概覽開始讀，再依序打開 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

這一章的數字與平台有關。notebook 裡存的輸出來自 macOS arm64、torch 2.14.0、沒有 mkldnn 的環境；在你的機器上執行時，表格下的說明會依實測印出。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽（M07～M10 的開端）：前半是後端部署共用的先備知識——推論框架、硬體擴充單位、kernel 函式庫的分層，三條路線，Quantizer 要配後端；後半是 Q/DQ 圖上的「dq → op → q」單元，reference representation、lowering、Inductor 各是什麼、彼此的關係，以及比較後端結果的規矩 |
| [01_fp32_numeric_floor.ipynb](01_fp32_numeric_floor.ipynb) | M07-1：不量化、只換 kernel 時 fp32 輸出差多少；profiler 看 kernel；用 `torch.backends` 手動切換 |
| [02_representation_and_lowering.ipynb](02_representation_and_lowering.ipynb) | M07-2：同一張 Q/DQ 圖經 reference representation、`lower_pt2e_quantized_to_x86`、`torch.compile` 各變成什麼 op |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m07_representation_lowering/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
