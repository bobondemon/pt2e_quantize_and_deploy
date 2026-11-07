"""用 graphviz 畫 fx 圖的子圖：只畫要解釋的那幾個 node。

整張圖動輒幾十個 node，真正要看的往往只有三五個。這裡的兩個函式都先決定
「保留哪些 node」，其餘的 node 只用虛線的「…」表示「外面還有東西」：

  draw(gm, focus=..., hops=1)          畫一張圖的局部
  draw_stages([(gm1, "export"), ...])  把同一個局部在多個階段並排，標出新增／消失的 node
  draw_onnx(onnx_model, focus=...)     同 draw，但畫的是 ONNX 圖（onnx.ModelProto，M09）

回傳的 graphviz.Digraph 在 notebook 裡會直接顯示成 SVG。
另附 `table()`：notebook 裡的表格用 HTML 顯示（中英混排的等寬文字表在瀏覽器裡會錯位）；
`annotation_notes()`：把 Quantizer 貼的標籤寫成 draw(notes=...) 的說明，標在 node 底下。
需要系統的 graphviz（`dot` 執行檔）：macOS `brew install graphviz`、Ubuntu `apt install graphviz`。
"""

from collections.abc import Callable, Iterable

import graphviz
import torch
from IPython.display import HTML
from tabulate import tabulate

from common.graph_utils import QDQ_PREFIXES, qualified_name, short

FONT = "PingFang TC, Noto Sans CJK TC, Helvetica, sans-serif"


# GitHub 的 notebook 預覽會把 SVG 裡的 UTF-8 文字（中文、…、→）顯示成 \xe5… 這種亂碼。
# 這裡讓 graphviz 物件在 notebook 輸出的 SVG 改用 XML 字元參照（&#21069; 這類），整份 SVG 只剩 ASCII；
# 瀏覽器、Jupyter、Quarto 顯示的結果不變。只影響 notebook 的顯示，不影響 graphviz 的其他用法。
_svg_utf8 = graphviz.jupyter_integration.JupyterIntegration._repr_image_svg_xml


def ascii_svg(svg: str) -> str:
    return svg.encode("ascii", "xmlcharrefreplace").decode("ascii")


def _repr_image_svg_xml_ascii(self) -> str:
    return ascii_svg(_svg_utf8(self))


graphviz.jupyter_integration.JupyterIntegration._repr_image_svg_xml = _repr_image_svg_xml_ascii

# 依 node 種類決定外觀：(shape, 填色, 邊框色)
STYLE = {
    "io": ("ellipse", "#E7F5FF", "#1C7ED6"),
    "op": ("box", "#FFFFFF", "#495057"),
    "observer": ("box", "#FFF4E6", "#E8590C"),
    "fake_quant": ("box", "#FFE8CC", "#D9480F"),
    "quantize": ("box", "#E5DBFF", "#7048E8"),
    "dequantize": ("box", "#F3F0FF", "#7048E8"),
    "param": ("note", "#F8F9FA", "#868E96"),
}
ADDED, REMOVED, HIGHLIGHT = "#2B8A3E", "#C92A2A", "#D6336C"

Focus = None | str | Iterable[str] | Callable[[torch.fx.Node], bool]


# ---------------------------------------------------------------------------
# node 分類與標籤
# ---------------------------------------------------------------------------
def _is_guard(gm, n):
    """torch.export 自動插入的 shape guard（_guards_fn），與量化無關，一律不畫。"""
    return n.op == "call_module" and type(gm.get_submodule(n.target)).__name__ == "GuardsFn"


def _kind(gm, n):
    if n.op in ("placeholder", "output"):
        return "io"
    if n.op == "get_attr":
        return "param"
    if n.op == "call_module":
        mod = gm.get_submodule(n.target)
        if "FakeQuantize" in type(mod).__name__:
            return "fake_quant"
        # observer 都有 calculate_qparams；其他 call_module（例如 M00-4 自己掛上去的模組）當一般 op 畫
        return "observer" if hasattr(mod, "calculate_qparams") else "op"
    name = qualified_name(n.target)
    if name.startswith(QDQ_PREFIXES):
        return "dequantize" if "dequantize" in name else "quantize"
    return "op"


def _key(gm, n):
    """跨階段比對 node 用的鍵。一般 node 用名字；observer／fake quant 用它觀察的對象，
    因為 prepare_pt2e 依插入順序替 observer 編號，同一個位置的 observer 在不同階段可能叫不同名字。"""
    if _kind(gm, n) in ("observer", "fake_quant"):
        return f"{_key(gm, n.args[0])}→observer"
    return n.name


def _param_text(gm, n):
    """get_attr 的短描述；不是 fp32 的 tensor 附上 dtype（例如 convert 後的 int8 weight）。"""
    try:
        t = _fetch(gm, n.target)
    except AttributeError:
        return n.name
    if isinstance(t, torch.Tensor) and t.dtype != torch.float32:
        return f"{n.name}: {str(t.dtype).removeprefix('torch.')}"
    return n.name


def _fetch(gm, target):
    obj = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _qparams_text(gm, n):
    """observer 校準後的 qparams；q/dq node 的 scale／zp（per-tensor 時是常數參數）。"""
    if n.op == "call_module":
        mod = gm.get_submodule(n.target)
        min_val = getattr(mod, "min_val", None)
        if not isinstance(min_val, torch.Tensor) or min_val.numel() == 0 or not torch.isfinite(min_val).all():
            return ""  # 還沒校準，不呼叫 calculate_qparams
        s, z = mod.calculate_qparams()
        if s.numel() > 1:
            return f"per-channel × {s.numel()}"
        return f"scale={float(s):.4g} zp={int(z)}"
    if _kind(gm, n) in ("quantize", "dequantize"):
        s, z = n.args[1], n.args[2]
        if isinstance(s, float) and isinstance(z, int):
            return f"scale={s:.4g} zp={z}"
        if "per_channel" in str(n.target):
            return "per-channel"
        return "scale／zp 執行時計算" if isinstance(s, torch.fx.Node) else ""  # dynamic：qparams 來自 choose_qparams
    return ""


def _label(gm, n, show_params, note):
    kind = _kind(gm, n)
    if kind == "io":
        title = n.name if n.op == "placeholder" else "output"
    elif n.op == "call_module":
        title = type(gm.get_submodule(n.target)).__name__
    elif kind == "param":
        title = _param_text(gm, n)
    else:
        title = short(n.target)
    rows = [f"<b>{_esc(title)}</b>"]
    if kind not in ("io", "param") and n.name != title:
        rows.append(f'<font point-size="9" color="#868E96">{_esc(n.name)}</font>')
    if not show_params:
        params = [_param_text(gm, a) for a in n.all_input_nodes if a.op == "get_attr"]
        if params:
            text = ", ".join(params[:2]) + (f", +{len(params) - 2}" if len(params) > 2 else "")
            rows.append(f'<font point-size="9" color="#495057">‹{_esc(text)}›</font>')
    qp = _qparams_text(gm, n)
    if qp:
        rows.append(f'<font point-size="9" color="#5F3DC4">{_esc(qp)}</font>')
    for line in (note or "").split("\n"):  # 一則說明可以有好幾行
        if line:
            rows.append(f'<font point-size="10" color="{HIGHLIGHT}"><i>{_esc(line)}</i></font>')
    return "<" + "<br/>".join(rows) + ">"


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# 選出要畫的 node
# ---------------------------------------------------------------------------
def _neighbors(gm, n, show_params):
    """資料流上的鄰居（輸入 + 使用者）；get_attr 預設內嵌在標籤裡，不算鄰居。"""
    ins = [a for a in n.all_input_nodes if show_params or a.op != "get_attr"]
    return [m for m in [*ins, *n.users] if not _is_guard(gm, m)]


def _drawable(gm, show_params):
    return [n for n in gm.graph.nodes if not _is_guard(gm, n) and (show_params or n.op != "get_attr")]


def _select(gm, focus: Focus, hops, show_params):
    nodes = _drawable(gm, show_params)
    if focus is None:
        return set(nodes)
    if callable(focus):
        seeds = {n for n in nodes if focus(n)}
    else:
        names = {focus} if isinstance(focus, str) else set(focus)
        seeds = {n for n in nodes if n.name in names}
        if not seeds:  # 部分名稱不在圖上沒關係（例如 prepare 之後 batch_norm 已被折掉），全都找不到才報錯
            raise ValueError(f"圖上找不到這些 node：{sorted(names)}")
    kept, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        frontier = {m for n in frontier for m in _neighbors(gm, n, show_params)} - kept
        kept |= frontier
    return kept


def _carry_over(gm, prev_keys, kept_prev_keys, show_params):
    """下一個階段要保留的 node：上一階段保留的同一批 node（以 _key 比對），加上和它們相連的新 node。

    新 node（上一階段沒有，例如 prepare 插的 observer、convert 插的 q/dq）
    只要透過一串新 node 接到保留的 node，就一起保留。
    """
    nodes = _drawable(gm, show_params)
    kept = {n for n in nodes if _key(gm, n) in kept_prev_keys}
    new = {n for n in nodes if _key(gm, n) not in prev_keys}
    frontier = set(kept)
    while frontier:
        frontier = {m for n in frontier for m in _neighbors(gm, n, show_params) if m in new} - kept
        kept |= frontier
    return kept, {n.name for n in new}


# ---------------------------------------------------------------------------
# 畫圖
# ---------------------------------------------------------------------------
def _add_graph(g, gm, kept, *, prefix, show_params, show_shapes, notes, highlight, added=(), removed=()):
    """把 kept 的 node 與邊加進 g，回傳 (起點 id, 終點 id)：沒有入邊／出邊的 node，draw_stages 換行時要用。"""
    kept = [n for n in gm.graph.nodes if n in kept]  # 依圖上的順序輸出，版面才會固定
    ids, edges = [prefix + n.name for n in kept], []
    for n in kept:
        shape, fill, border = STYLE[_kind(gm, n)]
        attrs = dict(shape=shape, style="filled,rounded" if shape == "box" else "filled",
                     fillcolor=fill, color=border, penwidth="1.2")
        if n.name in removed:  # 先套「下一步消失」，若同時是新增的，下面的綠框蓋過紅色
            attrs.update(style=attrs["style"] + ",dashed", color=REMOVED, penwidth="2.5")
        if n.name in added:
            attrs.update(color=ADDED, penwidth="3")
        if n.name in highlight:
            attrs.update(color=HIGHLIGHT, penwidth="3")
        g.node(prefix + n.name, _label(gm, n, show_params, notes.get(n.name)), **attrs)

    stub = dict(shape="plaintext", label="…", fontcolor="#ADB5BD")
    edge_stub = dict(style="dashed", color="#ADB5BD")
    for n in kept:
        ins = [a for a in n.all_input_nodes if (show_params or a.op != "get_attr") and not _is_guard(gm, a)]
        for a in dict.fromkeys(ins):
            if a in kept:
                g.edge(prefix + a.name, prefix + n.name, label=_shape_text(a) if show_shapes else "")
                edges.append((prefix + a.name, prefix + n.name))
        if any(a not in kept for a in ins):
            g.node(f"{prefix}{n.name}__in", **stub)
            g.edge(f"{prefix}{n.name}__in", prefix + n.name, **edge_stub)
            ids.append(f"{prefix}{n.name}__in")
            edges.append((f"{prefix}{n.name}__in", prefix + n.name))
        if any(u not in kept and not _is_guard(gm, u) for u in n.users):
            g.node(f"{prefix}{n.name}__out", **stub)
            g.edge(prefix + n.name, f"{prefix}{n.name}__out", **edge_stub)
            ids.append(f"{prefix}{n.name}__out")
            edges.append((prefix + n.name, f"{prefix}{n.name}__out"))
    srcs, dsts = {a for a, _ in edges}, {b for _, b in edges}
    return [i for i in ids if i not in dsts], [i for i in ids if i not in srcs]


def _shape_text(n):
    val = n.meta.get("val")
    if not isinstance(val, torch.Tensor):
        # convert_pt2e 插入的 q/dq 沒有 meta["val"]，型別從參數推：quantize 輸出最後一個參數的 dtype，dequantize 輸出 fp32
        target = qualified_name(n.target)
        if target.startswith("quantized_decomposed.quantize") and isinstance(n.args[-1], torch.dtype):
            return f" {str(n.args[-1]).removeprefix('torch.')}"
        if target.startswith("quantized_decomposed.dequantize"):
            return " float32"
        return ""
    dims = ",".join(str(d) if isinstance(d, int) else "N" for d in val.shape)  # 動態維度印成 N
    dtype = "" if val.dtype == torch.float32 else f" {str(val.dtype).removeprefix('torch.')}"
    return f" [{dims}]{dtype}"


def _new_digraph(title, rankdir):
    g = graphviz.Digraph(format="svg")
    g.attr(rankdir=rankdir, bgcolor="white", fontname=FONT, nodesep="0.25", ranksep="0.35")
    g.attr("node", fontname=FONT, fontsize="11", margin="0.12,0.05", height="0.3")
    g.attr("edge", fontname=FONT, fontsize="9", color="#495057", arrowsize="0.6")
    if title:
        g.attr(label=title, labelloc="t", fontsize="14")
    return g


def draw(gm, focus: Focus = None, hops=1, *, title=None, notes=None, highlight=(),
         show_params=False, show_shapes=False, rankdir="TB") -> graphviz.Digraph:
    """畫 gm 的局部。

    focus：要看的 node 名稱（字串或一組字串），或 `lambda node: ...`；None 畫整張圖。
    hops：從 focus 往輸入、往使用者各擴張幾步。
    notes：{node 名稱: 一句說明}，畫在該 node 底下。highlight：要加粗外框的 node 名稱。
    show_params：weight／bias 這類 get_attr 另外畫成 node（預設內嵌在使用它的 op 標籤裡）。
    show_shapes：邊上標 tensor shape（取自 node.meta["val"]）。
    """
    g = _new_digraph(title, rankdir)
    kept = _select(gm, focus, hops, show_params)
    _add_graph(g, gm, kept, prefix="", show_params=show_params, show_shapes=show_shapes,
               notes=notes or {}, highlight=set(highlight))
    return g


def draw_stages(stages, focus: Focus = None, hops=1, *, title=None, notes=None, per_row=2,
                show_params=False, show_shapes=False, rankdir="TB") -> graphviz.Digraph:
    """把同一個局部在多個階段並排（例如 export → prepare → convert）。

    stages：[(gm, "階段名稱"), ...]。focus／hops 只用在第一個階段；之後每個階段保留
    上一階段的同一批 node 與新接上的 node。一般 node 依名字比對，observer 依它觀察的對象比對。新增的 node 用綠框，下一階段消失的 node 用虛線
    （紅色；若同時是新增的則維持綠色）。圖的標題下方會自動加上這段圖例。
    notes：{node 名稱: 說明}，同名 node 在每個階段都會標；也可以傳一個 list，每個階段各一個 dict。
    per_row：一個 row 放幾個階段，超過就換到下一個 row。預設 2：三個階段並排時，在網站的內文寬度裡會縮到看不清，
    所以第三個階段（通常是 convert）畫在第二個 row。
    """
    legend = "綠框：這一步新增的 node　虛線：下一步會消失的 node"
    g = _new_digraph(f"{title}\n{legend}" if title else legend, rankdir)
    gms = [gm for gm, _ in stages]
    kept_sets, added_sets = [_select(gms[0], focus, hops, show_params)], [set()]
    for prev, gm in zip(gms, gms[1:]):
        prev_keys = {_key(prev, n) for n in prev.graph.nodes}
        kept, added = _carry_over(gm, prev_keys, {_key(prev, n) for n in kept_sets[-1]}, show_params)
        kept_sets.append(kept)
        added_sets.append(added)
    ends = []  # 每個階段的 (起點, 終點)
    for i, ((gm, label), kept) in enumerate(zip(stages, kept_sets)):
        nxt = {_key(gms[i + 1], n) for n in gms[i + 1].graph.nodes} if i + 1 < len(gms) else None
        removed = {n.name for n in kept if nxt is not None and _key(gm, n) not in nxt}
        c = graphviz.Digraph(name=f"cluster_{i}")  # graphviz 以 cluster_ 開頭的子圖畫成一個框
        c.attr(label=label, fontsize="13", style="rounded", color="#CED4DA", labeljust="l")
        ends.append(_add_graph(c, gm, kept, prefix=f"s{i}__", show_params=show_params, show_shapes=show_shapes,
                               notes=(notes[i] if isinstance(notes, (list, tuple)) else notes) or {},
                               highlight=set(), added=added_sets[i], removed=removed))
        g.subgraph(c)
    # 換 row：從上一個 row 每個階段的終點，拉看不見的邊到下一個 row 每個階段的起點，graphviz 就會把下一個 row 排在下面
    for i in range(per_row, len(stages)):
        row_above = range((i // per_row - 1) * per_row, i // per_row * per_row)
        for j in row_above:
            for a in ends[j][1]:
                for b in ends[i][0]:
                    g.edge(a, b, style="invis")
    return g


def _spec_text(spec):
    """一份 qspec 的短描述，畫在 node 底下用。"""
    if spec is None:
        return "不量化"
    name = type(spec).__name__
    if name == "SharedQuantizationSpec":
        e = spec.edge_or_node
        return f"與 {e[0].name}→{e[1].name} 共用" if isinstance(e, tuple) else f"與 {e.name} 的輸出共用"
    if name == "DerivedQuantizationSpec":
        return f"Derived {str(spec.dtype).removeprefix('torch.')}"
    if name == "FixedQParamsQuantizationSpec":
        return f"固定 scale={spec.scale:.4g} zp={spec.zero_point}"
    scheme = str(spec.qscheme).removeprefix("torch.")  # per_tensor_affine、per_channel_symmetric…
    granularity, kind = scheme.rsplit("_", 1)
    return f"{str(spec.dtype).removeprefix('torch.')} {granularity.replace('_', '-')} {kind.replace('symmetric', 'sym')}"


def annotation_notes(gm, unlabeled=False) -> dict[str, str]:
    """把每個 node 的標籤（QuantizationAnnotation）寫成 draw(notes=...) 用的說明。

    每條輸入邊一行「邊 上游：規格」，輸出一行「輸出：規格」；有標籤但沒有輸出標籤時寫「輸出：沒有標籤」。
    unlabeled=True 時，完全沒有標籤的 op 也標上「沒有標籤」。
    """
    notes = {}
    for n in gm.graph.nodes:
        if n.op != "call_function":  # get_attr 也可能帶空標籤（XNNPACKQuantizer 用來記錄「已標過」），不畫
            continue
        ann = n.meta.get("quantization_annotation")
        if ann is None:
            if unlabeled:
                notes[n.name] = "沒有標籤"
            continue
        lines = [f"邊 {a.name}：{_spec_text(s)}" for a, s in ann.input_qspec_map.items()]
        lines.append(f"輸出：{_spec_text(ann.output_qspec)}" if ann.output_qspec is not None else "輸出：沒有標籤")
        notes[n.name] = "\n".join(lines)
    return notes


def table(rows_or_html, headers=()) -> HTML:
    """在 notebook 裡顯示表格：傳 rows + headers，或傳 graph_utils 用 tablefmt="html" 產生的字串。"""
    if isinstance(rows_or_html, str):
        return HTML(rows_or_html)
    return HTML(tabulate(rows_or_html, headers=list(headers), tablefmt="html", disable_numparse=True))


def _onnx_op(node) -> str:
    """ONNX node 的 op 名稱；不是標準 domain 的加上 domain，例如 com.microsoft.QLinearAdd。"""
    return f"{node.domain}.{node.op_type}" if node.domain not in ("", "ai.onnx") else node.op_type


def _onnx_const_text(arr) -> str:
    """ONNX 常數的短描述：純量印數值，其餘印 dtype[shape]。"""
    if arr.size == 1:
        v = arr.item()
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return f"{arr.dtype}[{'×'.join(str(d) for d in arr.shape)}]"


def draw_onnx(model, focus=None, hops=1, *, title=None, notes=None, highlight=(), rankdir="TB") -> graphviz.Digraph:
    """畫 ONNX 圖（onnx.ModelProto）的局部，用法與外觀同 draw()（M09）。

    focus：ONNX node 的 name（例如 node_conv2d）、一組 name，或 `lambda node: ...`；None 畫整張圖。
    ORT 的 QDQ node fusion 沿用原本的 name（例如 node_conv2d 從 Conv 變成 QLinearConv），同一個 focus 可以跨等級比較；
    LAYOUT 換成 NHWC 版本的 node 會換名字。
    只吃常數的 node（例如 weight 的 DequantizeLinear）與 initializer 不另畫，內嵌在使用它的 op 標籤的 ‹…› 裡；
    QuantizeLinear／DequantizeLinear 的 scale、zero_point 是純量時直接印數值。
    """
    from onnx import numpy_helper  # 只有 M09 用到，不讓 viz 一定要依賴 onnx

    notes = notes or {}
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    const_text = {}  # 只吃常數的 node 的輸出 → 描述
    for n in model.graph.node:
        if n.op_type == "Constant":
            const_text[n.output[0]] = "Constant"
        elif n.input and all(not i or i in inits for i in n.input):
            first = inits[n.input[0]]
            const_text[n.output[0]] = f"{_onnx_op(n)}({_onnx_const_text(first)})"
    nodes = [n for n in model.graph.node if n.output[0] not in const_text]
    inputs, outputs = [i.name for i in model.graph.input], [o.name for o in model.graph.output]
    producer = {o: n.name for n in nodes for o in n.output} | {i: i for i in inputs}
    preds = {n.name: list(dict.fromkeys(producer[i] for i in n.input if i in producer)) for n in nodes}
    preds |= {f"{o}__output": [producer[o]] for o in outputs if o in producer}
    succs: dict[str, list[str]] = {}
    for k, ps in preds.items():
        for p in ps:
            succs.setdefault(p, []).append(k)
    order = inputs + [n.name for n in nodes] + [f"{o}__output" for o in outputs]

    if focus is None:
        kept = set(order)
    else:
        if callable(focus):
            seed = {n.name for n in nodes if focus(n)}
        else:
            wanted = {focus} if isinstance(focus, str) else set(focus)
            seed = wanted & set(order)
        if not seed:
            raise ValueError(f"focus 在圖上找不到任何 node：{focus}")
        kept, frontier = set(seed), set(seed)
        for _ in range(hops):
            frontier = {m for k in frontier for m in preds.get(k, []) + succs.get(k, [])} - kept
            kept |= frontier

    g = _new_digraph(title, rankdir)
    by_name = {n.name: n for n in nodes}
    for k in (k for k in order if k in kept):
        if k in by_name:
            n = by_name[k]
            kind = {"QuantizeLinear": "quantize", "DequantizeLinear": "dequantize"}.get(n.op_type, "op")
            rows = [f"<b>{_esc(_onnx_op(n))}</b>"]
            if n.name != n.op_type:
                rows.append(f'<font point-size="9" color="#868E96">{_esc(n.name)}</font>')
            consts = [i for i in n.input if i in inits or i in const_text]
            if kind != "op" and consts and all(c in inits and inits[c].size == 1 for c in consts):
                zp = _onnx_const_text(inits[consts[1]]) if len(consts) > 1 else "0"
                rows.append(f'<font point-size="9" color="#5F3DC4">scale={_onnx_const_text(inits[consts[0]])} zp={zp}</font>')
            else:
                big = [const_text.get(c) or _onnx_const_text(inits[c]) for c in consts
                       if c in const_text or inits[c].size > 1]
                if big:
                    text = ", ".join(big[:4]) + (f", +{len(big) - 4}" if len(big) > 4 else "")
                    rows.append(f'<font point-size="9" color="#495057">‹{_esc(text)}›</font>')
            for line in (notes.get(k) or "").split("\n"):
                if line:
                    rows.append(f'<font point-size="10" color="{HIGHLIGHT}"><i>{_esc(line)}</i></font>')
            label = "<" + "<br/>".join(rows) + ">"
        else:
            kind, label = "io", f"<<b>{_esc(k if k in inputs else 'output')}</b>>"
        shape, fill, border = STYLE[kind]
        attrs = dict(shape=shape, style="filled,rounded" if shape == "box" else "filled",
                     fillcolor=fill, color=border, penwidth="1.2")
        if k in highlight:
            attrs.update(color=HIGHLIGHT, penwidth="3")
        g.node(k, label, **attrs)

    stub = dict(shape="plaintext", label="…", fontcolor="#ADB5BD")
    edge_stub = dict(style="dashed", color="#ADB5BD")
    for k in (k for k in order if k in kept):
        for p in preds.get(k, []):
            if p in kept:
                g.edge(p, k)
        if any(p not in kept for p in preds.get(k, [])):
            g.node(f"{k}__in", **stub)
            g.edge(f"{k}__in", k, **edge_stub)
        if any(s not in kept for s in succs.get(k, [])):
            g.node(f"{k}__out", **stub)
            g.edge(k, f"{k}__out", **edge_stub)
    return g
