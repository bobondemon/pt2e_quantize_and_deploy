"""Tools for reading graphs. About 90% of learning PT2E is spent staring at the graph, so these tools come first."""

from collections import Counter

import torch
from tabulate import tabulate

# The q/dq ops produced by convert_pt2e all live under this namespace
QDQ_PREFIXES = ("quantized_decomposed.quantize", "quantized_decomposed.dequantize")


def qualified_name(target):
    """Full name of node.target, e.g. aten.conv2d.default.

    The str() of an executorch Edge dialect op (EdgeOpOverload, M08) is `<EdgeOpOverload: aten.relu.default>: schema = …`,
    so the name has to be taken from __name__.
    """
    name = str(target)
    if name.startswith("<EdgeOpOverload"):
        return target.__name__
    return name


def short(target):
    """Print node.target as a short name, e.g. aten.conv2d.default -> conv2d."""
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
    """One-line description used by graph_tree: op name + inlined get_attr (‹…›) + (optionally) non-node constant arguments."""
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
    """Draw the graph as a dataflow tree expanded backwards from the output; the structure is easier to see than in a node-by-node table.

    * Only nodes the output actually depends on are drawn (side branches such as `_guards_fn` and BN's num_batches_tracked update naturally disappear).
    * get_attr is inlined after the node that uses it, wrapped in ‹…›, and does not take a row of its own.
    * Intermediate results used by several nodes (forks) are declared once at the top as `[name]` and referenced as `[name]` inside the tree.
    * notes: {node.name: text}, appended at the end of that line; names: {node.name: alias}, used for the `[name]` of forks.
    * show_literals: whether to show non-node constant arguments such as stride and eps (off by default so they do not drown the structure).
    """
    notes, names = notes or {}, names or {}
    out = next(n for n in gm.graph.nodes if n.op == "output")

    def inputs(n):  # excludes get_attr (it is inlined in the label)
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
    """Count how often each op appears in the graph (call_module uses the module class name)."""
    c = Counter()
    for n in gm.graph.nodes:
        if n.op == "call_function":
            c[short(n.target)] += 1
        elif n.op == "call_module":
            mod = gm.get_submodule(n.target)
            tname = type(mod).__name__
            if tname == "GuardsFn":  # shape guard inserted automatically by torch.export; unrelated to quantization
                continue
            c[f"<module:{tname}>"] += 1
    return c


def diff_ops(before, after, label_before="before", label_after="after", tablefmt="simple"):
    """Return a readable table of op additions and removals, to answer what exactly prepare did."""
    cb, ca = op_counts(before), op_counts(after)
    keys = sorted(set(cb) | set(ca))
    rows = [[k, cb.get(k, 0), ca.get(k, 0), ca.get(k, 0) - cb.get(k, 0)] for k in keys]
    rows = [r for r in rows if r[3] != 0] + [r for r in rows if r[3] == 0]
    return tabulate(rows, headers=["op", label_before, label_after, "delta"], tablefmt=tablefmt)


def observer_nodes(gm):
    """The observers / fake-quants inserted into the graph after prepare_pt2e (all call_module).

    Returns [(node, module), ...]
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
    """The quantize / dequantize nodes in the graph after convert_pt2e."""
    return [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and str(n.target).startswith(QDQ_PREFIXES)
    ]


def observer_placement_table(gm, tablefmt="simple"):
    """Map each observer to the edge it sits on: producer -> observer -> consumers."""
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
        headers=["observer", "type", "input from", "output to", "qparams"],
        tablefmt=tablefmt,
    )


def sqnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio (dB). Higher is better; above 30 dB is usually fine."""
    ref, test = ref.detach().float(), test.detach().float()
    noise = ref - test
    p_ref = ref.pow(2).mean()
    p_noise = noise.pow(2).mean()
    if p_noise == 0:
        return float("inf")
    return float(10 * torch.log10(p_ref / p_noise))


@torch.no_grad()
def compare(model_a, model_b, inputs, name_a="fp32", name_b="quant"):
    """Run the same input and compare the two models' outputs by SQNR / max absolute error."""
    ya, yb = model_a(*inputs), model_b(*inputs)
    return {
        "sqnr_db": sqnr(ya, yb),
        "max_abs_err": float((ya - yb).abs().max()),
        "names": (name_a, name_b),
    }


def annotation_table(gm, tablefmt="simple"):
    """List the QuantizationAnnotation of every node in the graph (only populated after quantizer.annotate()).

    This is the most direct way to answer "which ops does this quantizer actually cover?" --
    don't trust articles; dump it and look for yourself.
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
    """The call_function nodes in the graph that the quantizer did not annotate (they stay in fp32)."""
    return [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.meta.get("quantization_annotation") is None
    ]


def qspec_summary(spec):
    """Print a qspec as one line: dtype, qscheme, integer range, observer, ch_axis."""
    if spec is None:
        return "None (not quantized)"
    name = type(spec).__name__
    if name == "SharedQuantizationSpec":
        edge = spec.edge_or_node
        label = f"{edge[0].name} -> {edge[1].name}" if isinstance(edge, tuple) else f"{edge.name} (output)"
        return f"Shared (shared with {label})"
    if name == "DerivedQuantizationSpec":
        return f"Derived (derived from {len(spec.derived_from)} locations, {str(spec.dtype).split('.')[-1]})"
    if name == "FixedQParamsQuantizationSpec":
        return f"FixedQParams (scale={spec.scale:g}, zp={spec.zero_point})"
    ctr = spec.observer_or_fake_quant_ctr
    ctr = getattr(getattr(ctr, "p", None), "func", ctr)  # unwrap if it was wrapped by with_args
    scheme = str(spec.qscheme).split(".")[-1].replace("per_", "per-").replace("_", " ")
    axis = f", ch_axis={spec.ch_axis}" if spec.ch_axis is not None else ""
    return f"{str(spec.dtype).split('.')[-1]} {scheme} [{spec.quant_min},{spec.quant_max}]\n{ctr.__name__}{axis}"
