"""weight_norm 的兩個工具（M05-5、M06-6 共用）。

`add_weight_norm` 作用在 nn.Module；`strip_weight_norm` 作用在 export 之後的 GraphModule，
用的就是 M00-4 的改圖動作：建新 node -> 下游改吃新的 -> 刪舊的。
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

# weight_norm 被 export 進圖之後的 ATen op。這是圖上的名字（只能用它比對），不是可替換的 Python API
WEIGHT_NORM_OP = torch.ops.aten._weight_norm.default


def add_weight_norm(model, types=(nn.Conv2d, nn.Linear)):
    """對 model 裡符合 types 的 module 原地套 weight_norm(dim=0)，回傳這些 module 的名稱。"""
    names = [name for name, m in model.named_modules() if isinstance(m, types)]
    for name in names:
        weight_norm(model.get_submodule(name), name="weight", dim=0)
    return names


def strip_weight_norm(gm):
    """把圖上的 aten._weight_norm 換成 get_attr，權重存回 <module>.weight。原地修改，回傳拔掉的 weight 名稱。

    數值不變：weight 就是這顆 op 用目前的 g、v 算出來的結果。之後圖上的 weight 是常數，訓練的不再是 g、v，
    所以 PTQ 在 export 之後呼叫、QAT 在訓練之後呼叫。只處理 weight_norm，其他 parametrization 未驗證。
    """
    graph, stripped = gm.graph, []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not WEIGHT_NORM_OP:
            continue
        v_node, g_node = node.all_input_nodes
        # g 的 target 形如 conv.parametrizations.weight.original0；module 本身是根時沒有 "conv." 前綴
        prefix, _, rest = str(g_node.target).partition("parametrizations.")
        name = rest.split(".")[0]
        with torch.no_grad():
            value = WEIGHT_NORM_OP(gm.get_parameter(str(v_node.target)), gm.get_parameter(str(g_node.target)),
                                   *node.args[2:], **node.kwargs)
        gm.get_submodule(prefix.removesuffix(".")).register_parameter(name, nn.Parameter(value))

        with graph.inserting_before(node):
            weight_node = graph.get_attr(f"{prefix}{name}")
        weight_node.meta = dict(node.meta)
        node.replace_all_uses_with(weight_node)  # 順序不能反：下游改吃新的之後，舊的才能刪
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
