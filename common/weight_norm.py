"""Two weight_norm tools (shared by M05-5 and M06-6).

`add_weight_norm` works on an nn.Module; `strip_weight_norm` works on the GraphModule after export,
using the graph-editing moves from M00-4: create a new node -> make downstream consume the new one -> delete the old one.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

# The ATen op that weight_norm becomes after export. This is its name on the graph (used only for matching), not a replaceable Python API
WEIGHT_NORM_OP = torch.ops.aten._weight_norm.default


def add_weight_norm(model, types=(nn.Conv2d, nn.Linear)):
    """Apply weight_norm(dim=0) in place to every module in model matching types, and return those modules' names."""
    names = [name for name, m in model.named_modules() if isinstance(m, types)]
    for name in names:
        weight_norm(model.get_submodule(name), name="weight", dim=0)
    return names


def strip_weight_norm(gm):
    """Replace aten._weight_norm in the graph with a get_attr, storing the weight back as <module>.weight. Edits in place and returns the names of the weights stripped.

    Numerics are unchanged: the weight is exactly what this op computes from the current g and v. Afterwards the weight on the graph is a constant and g, v are no longer trained,
    so PTQ calls this after export and QAT calls it after training. Only weight_norm is handled; other parametrizations are unverified.
    """
    graph, stripped = gm.graph, []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not WEIGHT_NORM_OP:
            continue
        v_node, g_node = node.all_input_nodes
        # the target of g looks like conv.parametrizations.weight.original0; when the module itself is the root there is no "conv." prefix
        prefix, _, rest = str(g_node.target).partition("parametrizations.")
        name = rest.split(".")[0]
        with torch.no_grad():
            value = WEIGHT_NORM_OP(gm.get_parameter(str(v_node.target)), gm.get_parameter(str(g_node.target)),
                                   *node.args[2:], **node.kwargs)
        gm.get_submodule(prefix.removesuffix(".")).register_parameter(name, nn.Parameter(value))

        with graph.inserting_before(node):
            weight_node = graph.get_attr(f"{prefix}{name}")
        weight_node.meta = dict(node.meta)
        node.replace_all_uses_with(weight_node)  # The order must not be reversed: the old node can be deleted only after downstream consumes the new one
        graph.erase_node(node)
        for param_node in (v_node, g_node):
            if not param_node.users:
                graph.erase_node(param_node)

        gm.delete_submodule(f"{prefix}parametrizations.{name}")
        if not list(gm.get_submodule(f"{prefix}parametrizations").children()):
            gm.delete_submodule(f"{prefix}parametrizations")
        stripped.append(f"{prefix}{name}")
    graph.lint()
    gm.recompile()
    return stripped
