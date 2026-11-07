"""看圖的工具。PT2E 學習過程 90% 的時間都在盯著 graph 看，所以先把工具做順手。"""

from collections import Counter

import torch
from tabulate import tabulate

# convert_pt2e 產生的 q/dq op 都在這個 namespace 底下
QDQ_PREFIXES = ("quantized_decomposed.quantize", "quantized_decomposed.dequantize")


def qualified_name(target):
    """node.target 的完整名稱，例如 aten.conv2d.default。

    executorch 的 Edge dialect op（EdgeOpOverload，M08）的 str() 是 `<EdgeOpOverload: aten.relu.default>: schema = …`，
    名稱要從 __name__ 取。
    """
    name = str(target)
    if name.startswith("<EdgeOpOverload"):
        return target.__name__
    return name


def short(target):
    """把 node.target 印成短名字，例如 aten.conv2d.default -> conv2d。"""
    if isinstance(target, str):
        return target
    name = qualified_name(target)
    if name.startswith("aten."):
        return name[len("aten.") :].rsplit(".", 1)[0]
    if name.startswith("quantized_decomposed."):
        return name[len("quantized_decomposed.") :].rsplit(".", 1)[0]
    return getattr(target, "__name__", name)


def _fmt_arg(a):
    if isinstance(a, torch.fx.Node):
        return a.name
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(_fmt_arg(x) for x in a) + "]"
    if isinstance(a, float):
        return f"{a:.4g}"
    return str(a)


def node_rows(gm, show_args=True):
    rows = []
    for i, n in enumerate(gm.graph.nodes):
        args = ", ".join(_fmt_arg(a) for a in n.args) if show_args else ""
        if len(args) > 60:
            args = args[:57] + "..."
        rows.append([i, n.op, n.name, short(n.target), args])
    return rows


def graph_table(gm, title=None, show_args=True):
    tbl = tabulate(
        node_rows(gm, show_args),
        headers=["#", "op", "name", "target", "args"],
        tablefmt="simple",
    )
    if title:
        bar = "=" * max(len(title) + 8, 40)
        return f"\n{bar}\n=== {title}\n{bar}\n{tbl}"
    return tbl


def print_graph(gm, title=None, show_args=True):
    print(graph_table(gm, title, show_args))


def _node_label(gm, n, show_literals=False):
    """graph_tree 用的單行描述：op 名稱 + 內嵌的 get_attr（‹…›）+（可選）非 node 的常數參數。"""
    if n.op in ("placeholder", "get_attr"):
        return str(n.target)
    if n.op == "call_module":
        tname = type(gm.get_submodule(n.target)).__name__
        base = "fake_quant" if "FakeQuantize" in tname else "observer" if "Observer" in tname else tname
    else:
        base = short(n.target)
    attrs = [str(a.target) for a in n.all_input_nodes if a.op == "get_attr"]
    lits = [_fmt_arg(a) for a in n.args if not isinstance(a, (torch.fx.Node, type(None)))] if show_literals else []
    out = base
    if attrs:
        out += " ‹" + ", ".join(attrs) + "›"
    if lits:
        out += " (" + ", ".join(lits) + ")"
    return out


def graph_tree(gm, notes=None, names=None, show_literals=False):
    """把圖畫成『從 output 往回展開的資料流樹』，比逐 node 的表格容易看出結構。

    * 只畫 output 真正依賴的 node（`_guards_fn`、BN 的 num_batches_tracked 更新這類旁支自然不見）。
    * get_attr 內嵌在使用它的 node 後面，用 ‹…› 包起來，不另佔一個 row。
    * 被多個 node 使用的中間結果（分叉）先在最上面用 `[名字]` 宣告一次，樹裡以 `[名字]` 引用。
    * notes: {node.name: 說明}，附在該行尾端；names: {node.name: 別名}，用在分叉的 `[名字]`。
    * show_literals: 是否顯示 stride、eps 這類非 node 的常數參數（預設不顯示，避免淹沒結構）。
    """
    notes, names = notes or {}, names or {}
    out = next(n for n in gm.graph.nodes if n.op == "output")

    def inputs(n):  # 不含 get_attr（它內嵌在 label 裡）
        return [a for a in n.all_input_nodes if a.op != "get_attr"]

    users, seen = Counter(), set()

    def visit(n):
        for a in inputs(n):
            users[a] += 1
            if a not in seen:
                seen.add(a)
                visit(a)

    visit(out)
    shared = [n for n in gm.graph.nodes if users[n] > 1 and n.op in ("call_function", "call_module")]
    alias = {n: names.get(n.name, n.name) for n in shared}

    def render(n, prefix, last, lines, expand):
        tag = f"[{alias[n]}]" if n in alias and not expand else _node_label(gm, n, show_literals)
        note = f"   ← {notes[n.name]}" if n.name in notes and not (n in alias and not expand) else ""
        branch = "" if prefix is None else prefix + ("└─ " if last else "├─ ")
        lines.append((branch or "") + tag + note)
        if n in alias and not expand:
            return
        kids = inputs(n)
        child_prefix = "" if prefix is None else prefix + ("   " if last else "│  ")
        for i, k in enumerate(kids):
            render(k, child_prefix, i == len(kids) - 1, lines, False)

    lines = []
    for n in shared:
        sub = []
        render(n, None, True, sub, True)
        lines.append(f"[{alias[n]}] = {sub[0]}")
        lines.extend("  " + ln for ln in sub[1:])
        lines.append("")
    lines.append("output")
    for i, k in enumerate(inputs(out)):
        render(k, "", i == len(inputs(out)) - 1, lines, False)
    return "\n".join(lines)


def op_counts(gm):
    """統計圖上各種 op 出現次數（call_module 用模組類別名）。"""
    c = Counter()
    for n in gm.graph.nodes:
        if n.op == "call_function":
            c[short(n.target)] += 1
        elif n.op == "call_module":
            mod = gm.get_submodule(n.target)
            tname = type(mod).__name__
            if tname == "GuardsFn":  # torch.export 自動插入的 shape guard，與量化無關
                continue
            c[f"<module:{tname}>"] += 1
    return c


def diff_ops(before, after, label_before="before", label_after="after", tablefmt="simple"):
    """回傳可讀的 op 增減表，用來回答『prepare 到底做了什麼』。"""
    cb, ca = op_counts(before), op_counts(after)
    keys = sorted(set(cb) | set(ca))
    rows = [[k, cb.get(k, 0), ca.get(k, 0), ca.get(k, 0) - cb.get(k, 0)] for k in keys]
    rows = [r for r in rows if r[3] != 0] + [r for r in rows if r[3] == 0]
    return tabulate(rows, headers=["op", label_before, label_after, "delta"], tablefmt=tablefmt)


def observer_nodes(gm):
    """prepare_pt2e 之後，圖上插入的 observer / fake-quant（都是 call_module）。

    回傳 [(node, module), ...]
    """
    out = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        mod = gm.get_submodule(n.target)
        if hasattr(mod, "calculate_qparams") or "FakeQuantize" in type(mod).__name__:
            out.append((n, mod))
    return out


def qdq_nodes(gm):
    """convert_pt2e 之後圖上的 quantize / dequantize node。"""
    return [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and str(n.target).startswith(QDQ_PREFIXES)
    ]


def observer_placement_table(gm, tablefmt="simple"):
    """把每個 observer 對應到『它掛在哪條邊上』：producer -> observer -> consumers。"""
    rows = []
    for n, mod in observer_nodes(gm):
        producer = n.args[0].name if n.args and isinstance(n.args[0], torch.fx.Node) else "-"
        consumers = ", ".join(u.name for u in n.users) or "-"
        qparams = ""
        try:
            s, z = mod.calculate_qparams()
            qparams = f"scale={_fmt_arg(float(s.flatten()[0]))} zp={int(z.flatten()[0])}"
            if s.numel() > 1:
                qparams += f" (per-channel, n={s.numel()})"
        except Exception:
            qparams = "(not calibrated)"
        rows.append([n.name, type(mod).__name__, producer, consumers, qparams])
    return tabulate(
        rows,
        headers=["observer", "type", "輸入來自", "輸出給", "qparams"],
        tablefmt=tablefmt,
    )


def sqnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio (dB)。越大越好，> 30dB 通常沒問題。"""
    ref, test = ref.detach().float(), test.detach().float()
    noise = ref - test
    p_ref = ref.pow(2).mean()
    p_noise = noise.pow(2).mean()
    if p_noise == 0:
        return float("inf")
    return float(10 * torch.log10(p_ref / p_noise))


@torch.no_grad()
def compare(model_a, model_b, inputs, name_a="fp32", name_b="quant"):
    """跑同一份 input，比較兩個模型輸出的 SQNR / 最大絕對誤差。"""
    ya, yb = model_a(*inputs), model_b(*inputs)
    return {
        "sqnr_db": sqnr(ya, yb),
        "max_abs_err": float((ya - yb).abs().max()),
        "names": (name_a, name_b),
    }


def annotation_table(gm, tablefmt="simple"):
    """列出圖上每個 node 的 QuantizationAnnotation（quantizer.annotate() 之後才有東西）。

    這是回答「這個 quantizer 到底管了哪些 op」最直接的方法——
    不要相信文章，自己 dump 出來看。
    """
    rows = []
    for n in gm.graph.nodes:
        ann = n.meta.get("quantization_annotation")
        if ann is None:
            continue
        ins = ", ".join(
            f"{k.name}:{type(v).__name__.replace('QuantizationSpec', 'Spec')}"
            for k, v in ann.input_qspec_map.items()
        )
        out = type(ann.output_qspec).__name__.replace("QuantizationSpec", "Spec")
        rows.append([n.name, short(n.target), ins or "-", out])
    return tabulate(rows, headers=["node", "target", "input_qspec_map", "output_qspec"],
                    tablefmt=tablefmt)


def unannotated_ops(gm):
    """圖上沒有被 quantizer 標註到的 call_function node（會留在 fp32）。"""
    return [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.meta.get("quantization_annotation") is None
    ]


def qspec_summary(spec):
    """把一份 qspec 印成一行：dtype、qscheme、整數範圍、observer、ch_axis。"""
    if spec is None:
        return "None（不量化）"
    name = type(spec).__name__
    if name == "SharedQuantizationSpec":
        edge = spec.edge_or_node
        label = f"{edge[0].name} -> {edge[1].name}" if isinstance(edge, tuple) else f"{edge.name} (output)"
        return f"Shared（與 {label} 共用）"
    if name == "DerivedQuantizationSpec":
        return f"Derived（由 {len(spec.derived_from)} 個位置推導，{str(spec.dtype).split('.')[-1]}）"
    if name == "FixedQParamsQuantizationSpec":
        return f"FixedQParams（scale={spec.scale:g}, zp={spec.zero_point}）"
    ctr = spec.observer_or_fake_quant_ctr
    ctr = getattr(getattr(ctr, "p", None), "func", ctr)  # with_args 包過的話，剝開
    scheme = str(spec.qscheme).split(".")[-1].replace("per_", "per-").replace("_", " ")
    axis = f", ch_axis={spec.ch_axis}" if spec.ch_axis is not None else ""
    return f"{str(spec.dtype).split('.')[-1]} {scheme} [{spec.quant_min},{spec.quant_max}]\n{ctr.__name__}{axis}"
