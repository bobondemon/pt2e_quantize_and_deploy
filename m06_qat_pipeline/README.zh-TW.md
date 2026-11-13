# M06 — QAT 的訓練流程

> ### 🌐 [English](README.md) ｜ **繁體中文**

從概覽開始讀，再依序打開 notebook（M06-5、M06-6 是選讀）。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：int4 下 QAT 追回 PTQ 的精度；fake quant 與 STE 的定義；全章的原則「訓練時的圖 = convert 之後的圖」 |
| [01_qat_minimal.ipynb](01_qat_minimal.ipynb) | M06-1：STE 的梯度；逐行寫出的最小 QAT 迴圈；convert 之後和 PTQ 同形 |
| [02_train_eval_switch.ipynb](02_train_eval_switch.ipynb) | M06-2：評估前要切的兩個開關（BN、observer）；`allow_exported_model_train_eval` |
| [03_fused_fakequant_mismatch.ipynb](03_fused_fakequant_mismatch.ipynb) | M06-3：XNNPACK 預設的 weight fused fake quant 讓訓練圖 ≠ convert 圖；scale 公式與解法 |
| [04_qat_vs_ptq_int4.ipynb](04_qat_vs_ptq_int4.ipynb) | M06-4：int4 下 QAT 是否比 PTQ 好；和「再訓練後 PTQ」對照，確認效益來自量化適應（5 個 seed，約 30 秒） |
| [05_learnable_fake_quant.ipynb](05_learnable_fake_quant.ipynb) | M06-5（選讀）：`LearnableFakeQuantize` 的接法、scale 的梯度與軌跡、效果（約 2 分鐘） |
| [06_weight_norm_qat.ipynb](06_weight_norm_qat.ipynb) | M06-6（選讀）：QAT 保留 `weight_norm` 訓練 g、v；有 BN 時 convert 報錯與解法 |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m06_qat_pipeline/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
