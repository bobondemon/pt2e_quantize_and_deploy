# M03 — 標籤與 Quantizer

從概覽開始讀，再依序打開七本 notebook。GitHub 上可以直接看到執行結果；要自己改參數再跑，見下方「在自己的電腦上執行」。

| 檔案 | 內容 |
|---|---|
| [00_overview.ipynb](00_overview.ipynb) | 概覽：observer 的位置由誰決定；標籤、`QuantizationSpec` 與四種 spec、Quantizer 的三個方法 |
| [01_annotate_then_prepare.ipynb](01_annotate_then_prepare.ipynb) | M03-1：手貼標籤，`prepare_pt2e` 照標籤插 observer；邊標籤與輸出標籤 |
| [02_read_annotations.ipynb](02_read_annotations.ipynb) | M03-2：讀懂 XNNPACKQuantizer 貼的標籤；fusion 在標籤上的樣子；沒被標到的 op |
| [03_my_first_quantizer.ipynb](03_my_first_quantizer.ipynb) | M03-3：用 `node.target` 寫一個只量化 conv2d 的 Quantizer；`_annotated` 與 `validate` |
| [04_shared_and_implicit_sharing.ipynb](04_shared_and_implicit_sharing.ipynb) | M03-4：`SharedQuantizationSpec`；隱式共用何時合併、何時串接 |
| [05_derived_and_fixed_qparams.ipynb](05_derived_and_fixed_qparams.ipynb) | M03-5：int32 bias 的 `DerivedQuantizationSpec`；sigmoid 的 `FixedQParamsQuantizationSpec` |
| [06_partial_quantization.ipynb](06_partial_quantization.ipynb) | M03-6：只量化一部分、`ComposableQuantizer`、標註順序 |
| [07_backend_config_api.ipynb](07_backend_config_api.ipynb) | M03-7：`X86InductorQuantizer` 的設定優先順序；XNNPACKQuantizer 默默不生效的 `set_module_name` |

## 在自己的電腦上執行

全專案共用一個 [uv](https://docs.astral.sh/uv/) 環境（Python 3.12，在 `.venv/`），graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz          # Ubuntu：sudo apt install graphviz
uv sync                        # 建立 .venv，全專案只需做一次
uv run jupyter lab m03_quantizer_api/00_overview.ipynb
```

最後一行用 `.venv` 的 Python 啟動 JupyterLab，並在瀏覽器打開本章的概覽。它只負責打開，不會自動執行：
要逐格按 Shift+Enter，或選選單的 Run → Run All Cells。其他 notebook 從左側的檔案瀏覽器打開。
也可以改用 VS Code 開 `.ipynb`，kernel 選 `.venv`。
