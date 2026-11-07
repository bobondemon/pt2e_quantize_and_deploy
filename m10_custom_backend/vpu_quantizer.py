"""M10 的後端本體：一個虛構 VPU 的 Quantizer（MyVPUQuantizer）。

M10-1 逐段說明這個檔案，M10-2、M10-3 匯入使用（不是腳本，不單獨執行）。
VPU 是虛構的，但每條約束都是 VPU／NPU 類硬體常見的要求（一般說法，本專案沒有實機）：

    (1) 只有 VPU_OPS 裡的 op 有硬體 kernel；其餘 op 不貼標籤 → 留在 fp32，fallback 到 CPU
    (2) activation：int8、per-tensor symmetric（zero_point 固定 0）
    (3) weight：int8 [-127, 127]、per-channel symmetric
    (4) 所有 scale 都是 2 的冪次：requant 只要 shift，不需要乘法器
    (5) bias：int32，scale = 輸入 scale × weight scale
    (6) add／cat／max_pool2d／flatten／relu：輸入與輸出同一個 scale（硬體在這些 op 上不做 requant）
    (7) conv／linear／add 後面的 relu 融合成同一顆 kernel：中間沒有量化點
    (8) 只吃 static shape

註解裡的 (1)～(8) 對應上面的約束。
"""

import torch
from torch.fx import Node

from common.data import calibration_batches
from common.pt2e import (
    ANNOTATION_KEY,
    DerivedQuantizationSpec,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
    annotate_input_qspec_map,
    annotate_output_qspec,
    convert_pt2e,
    export_module,
    prepare_pt2e,
)

aten = torch.ops.aten
CONV2D, LINEAR, RELU = aten.conv2d.default, aten.linear.default, aten.relu.default
ADD, CAT = aten.add.Tensor, aten.cat.default
MAX_POOL2D, FLATTEN = aten.max_pool2d.default, aten.flatten.using_ints

# (1) 硬體 kernel 清單。quantizer 只標這些 op；M10-3 的 partitioner 也讀同一份清單
VPU_OPS = {CONV2D, LINEAR, RELU, ADD, CAT, MAX_POOL2D, FLATTEN}


# --- (4) scale 是 2 的冪次：自訂 observer --------------------------------------------
def round_up_pow2(scale: torch.Tensor) -> torch.Tensor:
    """把 scale 往上取到 2^k。往上取：可表示範圍只會變大，不會多 clip，代價是解析度變粗（最多 2 倍）。"""
    return torch.exp2(torch.ceil(torch.log2(scale)))


class Pow2MinMaxObserver(MinMaxObserver):
    """先照 MinMaxObserver 算 scale，再往上取到 2 的冪次。convert_pt2e 呼叫的就是 calculate_qparams()。"""

    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        return round_up_pow2(scale), zero_point


class Pow2PerChannelMinMaxObserver(PerChannelMinMaxObserver):
    """weight 用的 per-channel 版本：每個 output channel 各自往上取到 2^k。"""

    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        return round_up_pow2(scale), zero_point


# --- (2)(3)(5) 三種 spec ----------------------------------------------------------------
def act_spec() -> QuantizationSpec:
    """(2) activation：int8、per-tensor symmetric → zero_point = 0。"""
    return QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127,
                            qscheme=torch.per_tensor_symmetric, observer_or_fake_quant_ctr=Pow2MinMaxObserver)


def weight_spec() -> QuantizationSpec:
    """(3) weight：int8 [-127, 127]、per-channel symmetric，沿 output channel（軸 0）。"""
    return QuantizationSpec(dtype=torch.int8, quant_min=-127, quant_max=127, qscheme=torch.per_channel_symmetric,
                            ch_axis=0, observer_or_fake_quant_ctr=Pow2PerChannelMinMaxObserver)


def derive_bias_qparams(obs_or_fqs):
    """(5) bias scale = 輸入 scale × weight scale。兩者都是 2^k，乘積也是 2^k。"""
    s_in, _ = obs_or_fqs[0].calculate_qparams()
    s_w, _ = obs_or_fqs[1].calculate_qparams()
    scale = s_in * s_w
    return scale, torch.zeros_like(scale, dtype=torch.int32)


def bias_spec(act: Node, weight: Node, op: Node) -> DerivedQuantizationSpec:
    """(5) bias：int32，qparams 由 act → op、weight → op 兩條邊上的 observer 推導（M03-5）。"""
    return DerivedQuantizationSpec(derived_from=[(act, op), (weight, op)], derive_qparams_fn=derive_bias_qparams,
                                   dtype=torch.int32, quant_min=-(2**31), quant_max=2**31 - 1,
                                   qscheme=torch.per_channel_symmetric, ch_axis=0)


# --- 小工具 ---------------------------------------------------------------------------
def _done(*nodes: Node) -> bool:
    """有任何一個 node 已經被標過（_annotated）就不再動（M03-3）。"""
    return any((a := n.meta.get(ANNOTATION_KEY)) is not None and a._annotated for n in nodes)


def _mark(*nodes: Node):
    for n in nodes:
        n.meta[ANNOTATION_KEY]._annotated = True


def fused_relu(node: Node) -> Node | None:
    """(7) node 唯一的使用者是 relu 時，回傳那個 relu：兩者融合成一顆 kernel。"""
    users = list(node.users)
    if len(users) == 1 and users[0].op == "call_function" and users[0].target == RELU:
        return users[0]
    return None


class MyVPUQuantizer(Quantizer):
    """把 (1)～(8) 寫成 PT2E 的標籤。prepare_pt2e 會先把 BN 折進 conv，再呼叫 annotate、validate。"""

    def annotate(self, model):
        ops = [n for n in model.graph.nodes if n.op == "call_function"]
        # 標註順序：先大 pattern（conv+relu、add+relu），relu 才不會先被當成單獨的 op 標走（M03-6）
        for n in ops:
            if n.target in (CONV2D, LINEAR):
                self._conv_or_linear(n)
        for n in ops:
            if n.target == ADD:
                self._add(n)
        for n in ops:
            if n.target == CAT:
                self._cat(n)
            elif n.target in (MAX_POOL2D, FLATTEN, RELU):
                self._same_scale(n)
        # (1) 其餘 op（不在 VPU_OPS）不貼標籤 → convert 後在 dq 與 q 之間用 fp32 跑
        return model

    def _conv_or_linear(self, op: Node):
        out = fused_relu(op) or op  # (7) 有 relu 時，輸出標籤貼在 relu 上
        if _done(op, out):
            return
        act, weight = op.args[0], op.args[1]
        assert isinstance(act, Node) and isinstance(weight, Node)
        annotate_input_qspec_map(op, act, act_spec())         # (2)
        annotate_input_qspec_map(op, weight, weight_spec())   # (3)(4)
        bias = op.args[2] if len(op.args) > 2 else None
        if isinstance(bias, Node):
            annotate_input_qspec_map(op, bias, bias_spec(act, weight, op))  # (5)
        annotate_output_qspec(out, act_spec())
        _mark(op, out)

    def _add(self, add: Node):
        a, b = add.args[0], add.args[1]
        if not (isinstance(a, Node) and isinstance(b, Node)):  # x + 純量 不是硬體的 add，留給 CPU
            return
        out = fused_relu(add) or add  # (7)
        if _done(add, out):
            return
        same = SharedQuantizationSpec((a, add))  # (6) 第二個輸入與輸出都跟第一個輸入共用
        annotate_input_qspec_map(add, a, act_spec())
        annotate_input_qspec_map(add, b, same)
        annotate_output_qspec(out, same)
        _mark(add, out)

    def _cat(self, cat: Node):
        inputs = cat.args[0]
        assert isinstance(inputs, (list, tuple)) and all(isinstance(x, Node) for x in inputs)
        if _done(cat):
            return
        first = inputs[0]
        assert isinstance(first, Node)
        same = SharedQuantizationSpec((first, cat))  # (6) 所有輸入與輸出共用
        annotate_input_qspec_map(cat, first, act_spec())
        for x in inputs[1:]:
            assert isinstance(x, Node)
            annotate_input_qspec_map(cat, x, same)
        annotate_output_qspec(cat, same)
        _mark(cat)

    def _same_scale(self, op: Node):
        """max_pool2d／flatten／單獨的 relu：輸出與輸入共用 scale（6）。"""
        x = op.args[0]
        assert isinstance(x, Node)
        if _done(op):
            return
        annotate_input_qspec_map(op, x, act_spec())
        annotate_output_qspec(op, SharedQuantizationSpec((x, op)))
        _mark(op)

    def validate(self, model):
        """(8) prepare_pt2e 在 annotate 之後呼叫：任何輸入有動態維度（SymInt）就在校準前報錯。"""
        for n in model.graph.nodes:
            val = n.meta.get("val")
            if n.op == "placeholder" and isinstance(val, torch.Tensor):
                if not all(isinstance(d, int) for d in val.shape):
                    raise ValueError(f"輸入 {n.name} 的 shape {tuple(val.shape)} 有動態維度；"
                                     "這個 VPU 只吃 static shape，export 時不要給 dynamic_shapes")


def vpu_ptq(model, example_inputs, n_calib=64):
    """VPU 的 PTQ：torch.export.export（static shape）→ prepare_pt2e(…, MyVPUQuantizer()) → 校準 → convert_pt2e。"""
    gm = export_module(model, example_inputs, dynamic_batch=False)
    prepared = prepare_pt2e(gm, MyVPUQuantizer())
    batch = example_inputs[0].shape[0]  # static shape：校準資料的 batch 必須和 export 時相同
    with torch.no_grad():
        for b in calibration_batches(n=n_calib // batch * batch, batch_size=batch):
            prepared(*b)
    return convert_pt2e(prepared)
