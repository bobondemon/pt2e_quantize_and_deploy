# PT2E 端到端：模型量化、後端部署與演算法／部署的分工

用 PyTorch 2 Export（PT2E）把模型量化成 int8，再交給 ExecuTorch + XNNPACK、ONNX Runtime 這類後端執行的動手教材。

**📖 請到網站閱讀：<https://bobondemon.github.io/pt2e_quantize_and_deploy/>**

這個 repo 是教材的原始檔。網站把各章的 notebook、圖與表格排成一本書，有側邊欄目錄與搜尋，比在 GitHub 上逐檔點開好讀。

## 這本教材的目的

<!-- 作者之後手動撰寫 -->

## 適合誰

- 用過 PyTorch 訓練模型，想把模型量化成 int8、部署到 CPU 或加速器上的人。
- 讀過 `torch.ao.quantization` 時代的文章，想知道 PT2E 現在怎麼用的人：API 在 torch 2.14 已搬到 torchao，舊文章的 import 路徑多半失效。
- 負責模型部署、需要寫或讀 Quantizer 的後端工程師。

不需要量化的背景；需要知道 `nn.Module` 怎麼寫、怎麼訓練。PT2E 的每一步都在改 `torch.fx` 的圖，
不熟 fx 可以先讀 [PyTorch FX 的 Graph／Node 一些基礎概念](https://bobondemon.github.io/2025/07/29/%E7%AD%86%E8%A8%98-PyTorch-FX-%E7%9A%84-Graph-Node-%E4%B8%80%E4%BA%9B%E5%9F%BA%E7%A4%8E%E6%A6%82%E5%BF%B5/)，M00 也會用小例子帶過。

## 兩個部分、兩種角色

教材以 `convert_pt2e` 產生的 **Q/DQ 圖**為界分成兩部分，實務上通常各由一種角色負責：

| | 第一部分（M01～M06） | 第二部分（M07～M10） |
|---|---|---|
| 做什麼 | fp32 模型 → PTQ／QAT → Q/DQ 圖 | Q/DQ 圖 → 後端的 int8 kernel → 裝置 |
| 角色 | 演算法人員：建模、訓練、量化 | 後端部署人員：選定平台、提供 Quantizer、lowering 與驗證 |

兩種角色不是各做各的：Quantizer 是後端照自己的 kernel 設計的，所以要先選後端、再量化。
交接的內容、合作的順序、精度掉了先查哪一段，詳見網站首頁（[index.qmd](index.qmd)）。

## 怎麼讀、怎麼執行

- **讀**：在網站上依側邊欄的順序讀。每一章先有一篇概覽，講這一章要回答的問題與核心概念，之後的 notebook 逐一動手驗證。
  notebook 的輸出都已經跑好，不執行也能看到結果。
- **自己執行**：點每本 notebook 頂端的 **Open in Colab** 按鈕，在 Colab 逐格執行、改參數再跑。
  第一個 code cell 會下載本教材的程式並安裝固定版本的套件（第一次約需幾分鐘），之後照順序執行即可。
  模型都很小、資料是程式產生的，只用 CPU 就跑得完，不必下載資料集。
- **在自己的電腦上執行**：用 [uv](https://docs.astral.sh/uv/) 建環境，指令寫在每一章資料夾的 `README.md`。

notebook 裡存的輸出來自 macOS arm64、Python 3.12、torch 2.14.0、torchao 0.18.0、executorch 1.5.0、onnxruntime 1.30.0。
數值與速度和平台有關的 notebook 會先印出執行環境，再依實測結果說明。

## 目錄結構

```text
.
├── index.qmd                  網站首頁：故事線、學習路徑、兩種角色
├── part1_overview.qmd         第一部分概覽（M01～M06）
├── part2_overview.qmd         第二部分概覽（M07～M10）
├── m00_export_basics/ … m10_custom_backend/
│   ├── 00_overview.ipynb      每章的概覽
│   ├── 01_….ipynb …           依序閱讀的 notebook
│   └── README.md              該章的索引與執行方式
├── common/                    notebook 共用的小工具（畫圖、小模型、合成資料、PT2E API 的單一入口）
├── _quarto.yml                網站設定
└── .github/workflows/publish.yml   push 到 main 時自動建置網站、發佈到 GitHub Pages
```

## 環境與指令（給想修改教材的人）

環境用 [uv](https://docs.astral.sh/uv/) 管理（Python 3.12，venv 在 `.venv/`），指令一律用 `uv run`，不必 activate。
graphviz 的子圖需要系統的 `dot` 執行檔。在專案根目錄執行：

```bash
brew install graphviz                          # Ubuntu：sudo apt install graphviz
uv sync --group executorch --group site        # executorch 只有 M08 用；site 是 pip 版 quarto
uv run python check_env.py                     # 版本、PT2E API 的位置，並跑一次端到端量化
uv run --group site quarto preview             # 在瀏覽器預覽網站
uv run --group site quarto render              # 建網站到 _site/
```

網站建置時不執行 notebook，直接用存在檔案裡的輸出；所以改了 notebook 要先重新執行再建網站。
重新執行一本 notebook（`--inplace` 會把新的輸出寫回檔案）：

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ClearMetadataPreprocessor.enabled=True m10_custom_backend/01_vpu_quantizer.ipynb
```

### 編寫 notebook

`.ipynb` 是唯一的原始檔。比較好編輯的方式是用 [jupytext](https://jupytext.readthedocs.io/) 的 percent 格式
（`# %%` 是 code cell，`# %% [markdown]` 是說明），在 repo 外的暫存資料夾編輯，再轉回 notebook：

```bash
# notebook → percent .py（放在 repo 外）
uv run jupytext --to py:percent --output /tmp/draft.py m01_hello_pt2e/01_minimal_ptq.ipynb

# percent .py → notebook，再執行一次把輸出寫進去
uv run jupytext --to ipynb --output m01_hello_pt2e/01_minimal_ptq.ipynb \
  --update-metadata '{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"jupytext":null}' \
  /tmp/draft.py
```

repo 裡不要留和 notebook 同名的 `.py`：同一個資料夾有同名的 `.py` 時，Quarto 會略過那本 `.ipynb`。

### 驗證

沒有另外的測試套件。notebook 裡的 `assert` 會檢查教材的每個主張，所以驗證就是重新執行所有 notebook
（在 macOS arm64 的 CPU 上全部約 10 分鐘）：

```bash
uv run python check_env.py
uv run jupyter nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True m*/*.ipynb
```

再確認輸出裡沒有 error 或 stderr（torch 的 `W…` 警告會出現在 stderr），下面的指令應該什麼都不印：

```bash
uv run python -c "import json,glob; [print(p) for p in glob.glob('m*/*.ipynb') if any(o['output_type']=='error' or o.get('name')=='stderr' for c in json.load(open(p))['cells'] for o in c.get('outputs',[]))]"
```

型別檢查用 pyright，要 0 error。notebook 先用上面的 jupytext 指令轉成 `.py` 放到 repo 外的資料夾再檢查；
M10 的 notebook 會 `import vpu_quantizer`，檢查時要用 `pyrightconfig.json` 把 `m10_custom_backend/` 加進 `extraPaths`。

```bash
uvx pyright --pythonpath .venv/bin/python common/ <轉出的 .py 所在資料夾>
```
