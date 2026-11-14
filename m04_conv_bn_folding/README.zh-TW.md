# M04 — Conv + BN 折疊

> ### 🌐 [English](README.md) ｜ **繁體中文**

從概覽開始讀，再依序打開三本 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：為什麼折 BN、折疊公式、PTQ 與 QAT 各在哪一步折；QAT 的 fake quant 為什麼量化 $W_{inf}$、慢版與近似版 |
| [01_bn_folding_trace.ipynb](01_bn_folding_trace.ipynb) | M04-1：PTQ 的 BN 在 `prepare_pt2e` 消失；weight 與 bias 怎麼改寫；conv 與 relu 之間多一個量化點時 convert 後的樣子 |
| [02_qat_graph_structure.ipynb](02_qat_graph_structure.ipynb) | M04-2：`prepare_qat_pt2e` 的 conv + BN 子圖每個 op 在算什麼；convert 時怎麼折、什麼情況下折不掉 |
| [03_qat_vs_article.ipynb](03_qat_vs_article.ipynb) | M04-3：PT2E 的 QAT 子圖與 PyTorch `_forward_approximate`（近似版）逐位元相同；和白皮書的慢版差多少 |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m04_conv_bn_folding/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
