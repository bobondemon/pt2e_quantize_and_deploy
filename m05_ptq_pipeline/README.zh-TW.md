> ### 🌐 [English](README.md) ｜ **繁體中文**

# M05 — PTQ 的精度除錯

從概覽開始讀，再依序打開 notebook（M05-5 是選讀）。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：為什麼需要一套查問題的順序；輸出 SQNR、逐層 SQNR、mixed precision 的定義 |
| [01_ptq_full.ipynb](01_ptq_full.ipynb) | M05-1：訓練好的模型做 PTQ；accuracy 與輸出 SQNR 哪個看得出量化誤差；per-channel 與 per-tensor |
| [02_calibration_study.ipynb](02_calibration_study.ipynb) | M05-2：校準樣本數與結果的波動；校準時 BN 會不會影響結果 |
| [03_layerwise_sqnr.ipynb](03_layerwise_sqnr.ipynb) | M05-3：用 numeric debugger 看逐層 SQNR；怎麼讀；參考圖用哪一張 |
| [04_mixed_precision.ipynb](04_mixed_precision.ipynb) | M05-4：排除某些 module 的 Quantizer；排除哪裡真的有用；誤差怎麼累積 |
| [05_weight_norm_ptq.ipynb](05_weight_norm_ptq.ipynb) | M05-5（選讀）：`weight_norm` export 後變成什麼；有 BN 時為什麼報錯；兩個解法 |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m05_ptq_pipeline/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
