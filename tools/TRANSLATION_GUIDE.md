# English translation guide (zh-TW → en)

Source of truth is the Chinese tree (repo root `m*/`, `*.qmd`). English tree is `en/`.
Keep the two in sync chapter by chapter. Do **not** change the Chinese files except where noted.

## Voice
Plain, precise technical English; keep the author's slightly conversational tone (e.g. the home-page aside).
Keep structure identical: same headings (translated), same tables, same cell boundaries, same numbering ("M03-2", "A.", "B.").
Do not add or drop content. Do not "improve" claims.

## Do not translate
Code identifiers, API names, file names, op names (`aten.conv2d`), math, URLs, numbers, units, Mermaid ids.
Notebook cell directives such as `#| include: false` stay as the first line of the cell.
Links to the author's Chinese blog posts stay as-is; append "(in Chinese)" after the link text.

## Glossary (use consistently)
| zh | en |
|---|---|
| 演算法人員 | algorithm engineer |
| 後端部署人員 | backend deployment engineer |
| 概覽 | overview |
| 標籤 / 貼標籤 | annotation / annotate |
| 校準 | calibration / calibrate |
| 折疊 (BN) | fold / folding |
| 改圖 | graph editing / edit the graph |
| 單元 (dq→op→q) | unit |
| 底線 (fp32 numeric) | floor |
| 切回 CPU | fall back to CPU |
| 分區圖 | partitioned graph |
| 交付 | hand over / deliver |
| 伏筆 | foreshadowing → "sets up", "a preview of" |
| 動手驗證 | verify hands-on |
| 第一部分 / 第二部分 | Part 1 / Part 2 |
| 〈量化之前，先選後端〉 | "Before Quantizing, Choose the Backend" |
| 〈比較後端之前的規矩〉 | "Rules Before Comparing Backends" |
Keep English technical terms the Chinese text already uses (observer, fake quant, Quantizer, qparams, scale, zero point, per-channel, lowering, delegate, fallback, SQNR, STE, QAT, PTQ, Q/DQ graph, fx, ATen, ExecuTorch, XNNPACK, ORT ...).

## Notebooks (`en/mXX_*/NN_*.ipynb`)
1. Copy the Chinese notebook to the same relative path under `en/`.
2. Translate every markdown cell, every Chinese comment / docstring / `print` string / f-string label / assert message / plot title & axis label / table header in code cells.
3. Page title = first H1 cell; translate it (it becomes the sidebar label).
4. Colab badge URL: `.../blob/main/en/<path>.ipynb`.
5. Colab bootstrap cell: translate comments; change `os.chdir(f"{repo}/mXX_...")` to `os.chdir(f"{repo}/en/mXX_...")`. Keep `#| include: false`.
6. Internal relative links stay valid inside `en/` (same layout).
7. Module 10 notebooks `import vpu_quantizer`: copy `m10_custom_backend/vpu_quantizer.py` to `en/m10_custom_backend/` and translate its comments/docstrings/messages.
8. Re-execute so outputs and images are English (`common/` is already English):
   `uv run jupyter nbconvert --to notebook --execute --inplace --ClearMetadataPreprocessor.enabled=True en/mXX_*/NN_*.ipynb`
   (M08 needs `uv sync --group executorch`.) If an `assert` fails because numbers differ from macOS arm64, do not weaken it silently: adjust only if the Chinese notebook's claim is platform-dependent, and report it.
   Prose that cites specific measured numbers/timings must match the re-executed outputs; if the numbers moved, update the English prose and list the change in your report.
9. Remaining CJK check: `python3 tools/check_cjk.py en/mXX_*` must print nothing (the allowed exceptions are listed inside the script).
10. Keep notebook JSON style (indent=1, ensure_ascii=False, trailing newline) so diffs stay small. Never leave a `.py` next to a notebook.

## Chapter README (`mXX_*/README.md`, English; Chinese one is `README.zh-TW.md`)
Translate `README.zh-TW.md` in the same folder. First line under the H1: `> ### 🌐 **English** ｜ [繁體中文](README.zh-TW.md)`. File links stay relative to the Chinese notebooks, and add a line pointing to the English notebooks: `English notebooks: [../en/mXX_.../](../en/mXX_.../)`. Also add the matching switch line to the top of the Chinese file: `> ### 🌐 [English](README.md) ｜ **繁體中文**`.
