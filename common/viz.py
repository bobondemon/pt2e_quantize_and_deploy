"""Draw sub-graphs of an fx graph with graphviz: only the few nodes that need explaining.

A whole graph easily has dozens of nodes, and only three or five of them are really worth looking at. Both functions here first decide
which nodes to keep; all other nodes are represented only by a dashed "…" meaning "there is more outside":

  draw(gm, focus=..., hops=1)          draw part of one graph
  draw_stages([(gm1, "export"), ...])  put the same part from several stages side by side, marking added / disappearing nodes
  draw_onnx(onnx_model, focus=...)     like draw, but for an ONNX graph (onnx.ModelProto, M09)

The returned graphviz.Digraph is displayed directly as SVG in a notebook.
Also included: `table()`, which shows tables in a notebook as HTML (monospace text tables mixing CJK and Latin text misalign in browsers);
and `annotation_notes()`, which writes the annotations a Quantizer attached as notes for draw(notes=...), placed under each node.
Requires the system graphviz (the `dot` executable): macOS `brew install graphviz`, Ubuntu `apt install graphviz`.
"""

from collections.abc import Callable, Iterable

import graphviz
import torch
from IPython.display import HTML
from tabulate import tabulate

from common.graph_utils import QDQ_PREFIXES, qualified_name, short

FONT = "PingFang TC, Noto Sans CJK TC, Helvetica, sans-serif"


# GitHub's notebook preview shows the UTF-8 text in an SVG (CJK, …, →) as mojibake like \xe5….
# Here graphviz objects emit their SVG in notebook output with XML character references (such as &#21069;), so the whole SVG is pure ASCII;
# what browsers, Jupyter and Quarto display is unchanged. It only affects notebook display, not other uses of graphviz.
_svg_utf8 = graphviz.jupyter_integration.JupyterIntegration._repr_image_svg_xml


def ascii_svg(svg: str) -> str:
    return svg.encode("ascii", "xmlcharrefreplace").decode("ascii")


def _repr_image_svg_xml_ascii(self) -> str:
    return ascii_svg(_svg_utf8(self))


graphviz.jupyter_integration.JupyterIntegration._repr_image_svg_xml = _repr_image_svg_xml_ascii

# Look is decided by node kind: (shape, fill color, border color)
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
# Node classification and labels
# ---------------------------------------------------------------------------
def _is_guard(gm, n):
    """Shape guard inserted automatically by torch.export (_guards_fn); unrelated to quantization and never drawn."""
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
        # Every observer has calculate_qparams; other call_modules (e.g. the module attached by hand in M00-4) are drawn as ordinary ops
        return "observer" if hasattr(mod, "calculate_qparams") else "op"
    name = qualified_name(n.target)
    if name.startswith(QDQ_PREFIXES):
        return "dequantize" if "dequantize" in name else "quantize"
    return "op"


def _key(gm, n):
    """Key used to match nodes across stages. Ordinary nodes use their name; observers / fake quants use the thing they observe,
    because prepare_pt2e numbers observers in insertion order, so the observer at the same position may have different names in different stages."""
    if _kind(gm, n) in ("observer", "fake_quant"):
        return f"{_key(gm, n.args[0])}→observer"
    return n.name


def _param_text(gm, n):
    """Short description of a get_attr; a tensor that is not fp32 gets its dtype appended (e.g. the int8 weight after convert)."""
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
    """The qparams of an observer after calibration; the scale / zp of a q/dq node (constant arguments when per-tensor)."""
    if n.op == "call_module":
        mod = gm.get_submodule(n.target)
        min_val = getattr(mod, "min_val", None)
        if not isinstance(min_val, torch.Tensor) or min_val.numel() == 0 or not torch.isfinite(min_val).all():
            return ""  # not calibrated yet, so calculate_qparams is not called
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
        return "scale/zp computed at runtime" if isinstance(s, torch.fx.Node) else ""  # dynamic: qparams come from choose_qparams
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
    for line in (note or "").split("\n"):  # one note can span several lines
        if line:
            rows.append(f'<font point-size="10" color="{HIGHLIGHT}"><i>{_esc(line)}</i></font>')
    return "<" + "<br/>".join(rows) + ">"


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Choose the nodes to draw
# ---------------------------------------------------------------------------
def _neighbors(gm, n, show_params):
    """Neighbors on the dataflow (inputs + users); get_attr is inlined in the label by default and does not count as a neighbor."""
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
        if not seeds:  # It is fine if some names are not on the graph (e.g. batch_norm is already folded away after prepare); only error if none are found
            raise ValueError(f"could not find these nodes on the graph: {sorted(names)}")
    kept, frontier = set(seeds), set(seeds)
    for _ in range(hops):
        frontier = {m for n in frontier for m in _neighbors(gm, n, show_params)} - kept
        kept |= frontier
    return kept


def _carry_over(gm, prev_keys, kept_prev_keys, show_params):
    """The nodes to keep in the next stage: the same nodes kept in the previous stage (matched by _key), plus the new nodes connected to them.

    A new node (absent from the previous stage, e.g. observers inserted by prepare, q/dq inserted by convert)
    is kept as well as long as a chain of new nodes connects it to a kept node.
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
# Drawing
# ---------------------------------------------------------------------------
def _add_graph(g, gm, kept, *, prefix, show_params, show_shapes, notes, highlight, added=(), removed=()):
    """Add the kept nodes and edges to g and return (start ids, end ids): the nodes with no incoming / outgoing edge, which draw_stages needs when wrapping to a new row."""
    kept = [n for n in gm.graph.nodes if n in kept]  # emit in graph order so the layout stays stable
    ids, edges = [prefix + n.name for n in kept], []
    for n in kept:
        shape, fill, border = STYLE[_kind(gm, n)]
        attrs = dict(shape=shape, style="filled,rounded" if shape == "box" else "filled",
                     fillcolor=fill, color=border, penwidth="1.2")
        if n.name in removed:  # Apply "disappears in the next step" first; if the node is also new, the green border below overrides the red
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
        # The q/dq inserted by convert_pt2e have no meta["val"], so the type is inferred from the arguments: quantize outputs the dtype in its last argument, dequantize outputs fp32
        target = qualified_name(n.target)
        if target.startswith("quantized_decomposed.quantize") and isinstance(n.args[-1], torch.dtype):
            return f" {str(n.args[-1]).removeprefix('torch.')}"
        if target.startswith("quantized_decomposed.dequantize"):
            return " float32"
        return ""
    dims = ",".join(str(d) if isinstance(d, int) else "N" for d in val.shape)  # dynamic dimensions are printed as N
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
    """Draw part of gm.

    focus: the node name(s) to look at (a string or a collection of strings), or `lambda node: ...`; None draws the whole graph.
    hops: how many steps to expand from focus toward the inputs and toward the users.
    notes: {node name: one-line note}, drawn under that node. highlight: names of nodes whose border is made bold.
    show_params: draw get_attr nodes such as weight / bias as separate nodes (by default they are inlined in the label of the op that uses them).
    show_shapes: label edges with the tensor shape (taken from node.meta["val"]).
    """
    g = _new_digraph(title, rankdir)
    kept = _select(gm, focus, hops, show_params)
    _add_graph(g, gm, kept, prefix="", show_params=show_params, show_shapes=show_shapes,
               notes=notes or {}, highlight=set(highlight))
    return g


def draw_stages(stages, focus: Focus = None, hops=1, *, title=None, notes=None, per_row=2,
                show_params=False, show_shapes=False, rankdir="TB") -> graphviz.Digraph:
    """Put the same part of the graph from several stages side by side (e.g. export → prepare → convert).

    stages: [(gm, "stage name"), ...]. focus / hops apply only to the first stage; each later stage keeps
    the same nodes as the previous stage plus the newly attached nodes. Ordinary nodes are matched by name, observers by the thing they observe. New nodes get a green border, and nodes that disappear in the next stage are dashed
    (red; if also new, they stay green). This legend is added automatically under the graph title.
    notes: {node name: note}, applied to nodes of that name in every stage; a list is also accepted, with one dict per stage.
    per_row: how many stages go in a row before wrapping to the next row. Default 2: three stages side by side shrink too small to read at the website's body width,
    so the third stage (usually convert) goes on the second row.
    """
    legend = "Green border: node added in this step    Dashed: node that disappears in the next step"
    g = _new_digraph(f"{title}\n{legend}" if title else legend, rankdir)
    gms = [gm for gm, _ in stages]
    kept_sets, added_sets = [_select(gms[0], focus, hops, show_params)], [set()]
    for prev, gm in zip(gms, gms[1:]):
        prev_keys = {_key(prev, n) for n in prev.graph.nodes}
        kept, added = _carry_over(gm, prev_keys, {_key(prev, n) for n in kept_sets[-1]}, show_params)
        kept_sets.append(kept)
        added_sets.append(added)
    ends = []  # (start, end) of each stage
    for i, ((gm, label), kept) in enumerate(zip(stages, kept_sets)):
        nxt = {_key(gms[i + 1], n) for n in gms[i + 1].graph.nodes} if i + 1 < len(gms) else None
        removed = {n.name for n in kept if nxt is not None and _key(gm, n) not in nxt}
        c = graphviz.Digraph(name=f"cluster_{i}")  # graphviz draws a subgraph whose name starts with cluster_ as a box
        c.attr(label=label, fontsize="13", style="rounded", color="#CED4DA", labeljust="l")
        ends.append(_add_graph(c, gm, kept, prefix=f"s{i}__", show_params=show_params, show_shapes=show_shapes,
                               notes=(notes[i] if isinstance(notes, (list, tuple)) else notes) or {},
                               highlight=set(), added=added_sets[i], removed=removed))
        g.subgraph(c)
    # Wrap to a new row: draw invisible edges from the end of each stage in the previous row to the start of each stage in the next row, and graphviz places the next row below
    for i in range(per_row, len(stages)):
        row_above = range((i // per_row - 1) * per_row, i // per_row * per_row)
        for j in row_above:
            for a in ends[j][1]:
                for b in ends[i][0]:
                    g.edge(a, b, style="invis")
    return g


def _spec_text(spec):
    """Short description of a qspec, drawn under a node."""
    if spec is None:
        return "not quantized"
    name = type(spec).__name__
    if name == "SharedQuantizationSpec":
        e = spec.edge_or_node
        return f"shared with {e[0].name}→{e[1].name}" if isinstance(e, tuple) else f"shared with output of {e.name}"
    if name == "DerivedQuantizationSpec":
        return f"Derived {str(spec.dtype).removeprefix('torch.')}"
    if name == "FixedQParamsQuantizationSpec":
        return f"fixed scale={spec.scale:.4g} zp={spec.zero_point}"
    scheme = str(spec.qscheme).removeprefix("torch.")  # per_tensor_affine, per_channel_symmetric…
    granularity, kind = scheme.rsplit("_", 1)
    return f"{str(spec.dtype).removeprefix('torch.')} {granularity.replace('_', '-')} {kind.replace('symmetric', 'sym')}"


def annotation_notes(gm, unlabeled=False) -> dict[str, str]:
    """Write each node's annotation (QuantizationAnnotation) as notes for draw(notes=...).

    One line per input edge, "edge upstream: spec", and one line for the output, "output: spec"; if the node is annotated but has no output annotation, it says "output: no annotation".
    With unlabeled=True, ops with no annotation at all are also marked "no annotation".
    """
    notes = {}
    for n in gm.graph.nodes:
        if n.op != "call_function":  # get_attr can also carry an empty annotation (XNNPACKQuantizer uses it to record "already annotated"); not drawn
            continue
        ann = n.meta.get("quantization_annotation")
        if ann is None:
            if unlabeled:
                notes[n.name] = "no annotation"
            continue
        lines = [f"edge {a.name}: {_spec_text(s)}" for a, s in ann.input_qspec_map.items()]
        lines.append(f"output: {_spec_text(ann.output_qspec)}" if ann.output_qspec is not None else "output: no annotation")
        notes[n.name] = "\n".join(lines)
    return notes


def table(rows_or_html, headers=()) -> HTML:
    """Show a table in a notebook: pass rows + headers, or a string produced by graph_utils with tablefmt="html"."""
    if isinstance(rows_or_html, str):
        return HTML(rows_or_html)
    return HTML(tabulate(rows_or_html, headers=list(headers), tablefmt="html", disable_numparse=True))


def _onnx_op(node) -> str:
    """Op name of an ONNX node; non-standard domains are prefixed with the domain, e.g. com.microsoft.QLinearAdd."""
    return f"{node.domain}.{node.op_type}" if node.domain not in ("", "ai.onnx") else node.op_type


def _onnx_const_text(arr) -> str:
    """Short description of an ONNX constant: scalars print their value, everything else prints dtype[shape]."""
    if arr.size == 1:
        v = arr.item()
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return f"{arr.dtype}[{'×'.join(str(d) for d in arr.shape)}]"


def draw_onnx(model, focus=None, hops=1, *, title=None, notes=None, highlight=(), rankdir="TB") -> graphviz.Digraph:
    """Draw part of an ONNX graph (onnx.ModelProto), used and styled like draw() (M09).

    focus: the name of an ONNX node (e.g. node_conv2d), a collection of names, or `lambda node: ...`; None draws the whole graph.
    ORT's QDQ node fusion keeps the original name (e.g. node_conv2d goes from Conv to QLinearConv), so the same focus can be compared across levels;
    nodes swapped for the NHWC version at the LAYOUT level get new names.
    Nodes that only consume constants (e.g. the weight's DequantizeLinear) and initializers are not drawn separately; they are inlined in the ‹…› of the label of the op that uses them;
    when the scale and zero_point of QuantizeLinear / DequantizeLinear are scalars, their values are printed directly.
    """
    from onnx import numpy_helper  # only M09 uses this, so viz does not have to depend on onnx

    notes = notes or {}
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    const_text = {}  # output of a constants-only node -> description
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
            raise ValueError(f"focus matches no node on the graph: {focus}")
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
