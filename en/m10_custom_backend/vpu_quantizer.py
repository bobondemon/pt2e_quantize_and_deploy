"""The backend itself for M10: a Quantizer (MyVPUQuantizer) for a fictional VPU.

M10-1 walks through this file section by section; M10-2 and M10-3 import it (it is not a script and is not run on its own).
The VPU is fictional, but every constraint is a common requirement of VPU/NPU-class hardware (general knowledge; this project has no real device):

    (1) only the ops in VPU_OPS have a hardware kernel; other ops get no annotation -> stay in fp32 and fall back to CPU
    (2) activation: int8, per-tensor symmetric (zero_point fixed at 0)
    (3) weight: int8 [-127, 127], per-channel symmetric
    (4) every scale is a power of 2: requant needs only a shift, no multiplier
    (5) bias: int32, scale = input scale x weight scale
    (6) add/cat/max_pool2d/flatten/relu: input and output share one scale (the hardware does no requant on these ops)
    (7) a relu after conv/linear/add is fused into the same kernel: no quantization point in between
    (8) static shapes only

The (1)-(8) in the comments refer to the constraints above.
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

# (1) The list of hardware kernels. The quantizer annotates only these ops; the partitioner in M10-3 reads the same list
VPU_OPS = {CONV2D, LINEAR, RELU, ADD, CAT, MAX_POOL2D, FLATTEN}


# --- (4) scale is a power of 2: custom observers --------------------------------------------
def round_up_pow2(scale: torch.Tensor) -> torch.Tensor:
    """Round scale up to 2^k. Rounding up only enlarges the representable range, so nothing extra is clipped; the cost is coarser resolution (at most 2x)."""
    return torch.exp2(torch.ceil(torch.log2(scale)))


class Pow2MinMaxObserver(MinMaxObserver):
    """Compute scale as MinMaxObserver does, then round it up to a power of 2. convert_pt2e calls calculate_qparams()."""

    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        return round_up_pow2(scale), zero_point


class Pow2PerChannelMinMaxObserver(PerChannelMinMaxObserver):
    """The per-channel version for weights: each output channel is rounded up to 2^k on its own."""

    def calculate_qparams(self):
        scale, zero_point = super().calculate_qparams()
        return round_up_pow2(scale), zero_point


# --- (2)(3)(5) the three specs ----------------------------------------------------------------
def act_spec() -> QuantizationSpec:
    """(2) activation: int8, per-tensor symmetric -> zero_point = 0."""
    return QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127,
                            qscheme=torch.per_tensor_symmetric, observer_or_fake_quant_ctr=Pow2MinMaxObserver)


def weight_spec() -> QuantizationSpec:
    """(3) weight: int8 [-127, 127], per-channel symmetric, along the output channel (axis 0)."""
    return QuantizationSpec(dtype=torch.int8, quant_min=-127, quant_max=127, qscheme=torch.per_channel_symmetric,
                            ch_axis=0, observer_or_fake_quant_ctr=Pow2PerChannelMinMaxObserver)


def derive_bias_qparams(obs_or_fqs):
    """(5) bias scale = input scale x weight scale. Both are 2^k, so the product is 2^k too."""
    s_in, _ = obs_or_fqs[0].calculate_qparams()
    s_w, _ = obs_or_fqs[1].calculate_qparams()
    scale = s_in * s_w
    return scale, torch.zeros_like(scale, dtype=torch.int32)


def bias_spec(act: Node, weight: Node, op: Node) -> DerivedQuantizationSpec:
    """(5) bias: int32, with qparams derived from the observers on the two edges act -> op and weight -> op (M03-5)."""
    return DerivedQuantizationSpec(derived_from=[(act, op), (weight, op)], derive_qparams_fn=derive_bias_qparams,
                                   dtype=torch.int32, quant_min=-(2**31), quant_max=2**31 - 1,
                                   qscheme=torch.per_channel_symmetric, ch_axis=0)


# --- helpers ---------------------------------------------------------------------------
def _done(*nodes: Node) -> bool:
    """If any of the nodes has already been annotated (_annotated), leave them alone (M03-3)."""
    return any((a := n.meta.get(ANNOTATION_KEY)) is not None and a._annotated for n in nodes)


def _mark(*nodes: Node):
    for n in nodes:
        n.meta[ANNOTATION_KEY]._annotated = True


def fused_relu(node: Node) -> Node | None:
    """(7) If the only user of node is a relu, return that relu: the two fuse into one kernel."""
    users = list(node.users)
    if len(users) == 1 and users[0].op == "call_function" and users[0].target == RELU:
        return users[0]
    return None


class MyVPUQuantizer(Quantizer):
    """Writes (1)-(8) as PT2E annotations. prepare_pt2e first folds BN into conv, then calls annotate and validate."""

    def annotate(self, model):
        ops = [n for n in model.graph.nodes if n.op == "call_function"]
        # Annotation order: larger patterns first (conv+relu, add+relu), so the relu is not annotated first as a standalone op (M03-6)
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
        # (1) Other ops (not in VPU_OPS) get no annotation -> after convert they run in fp32 between dq and q
        return model

    def _conv_or_linear(self, op: Node):
        out = fused_relu(op) or op  # (7) with a relu, the output annotation goes on the relu
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
        if not (isinstance(a, Node) and isinstance(b, Node)):  # x + scalar is not a hardware add; leave it to the CPU
            return
        out = fused_relu(add) or add  # (7)
        if _done(add, out):
            return
        same = SharedQuantizationSpec((a, add))  # (6) the second input and the output both share with the first input
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
        same = SharedQuantizationSpec((first, cat))  # (6) all inputs and the output share
        annotate_input_qspec_map(cat, first, act_spec())
        for x in inputs[1:]:
            assert isinstance(x, Node)
            annotate_input_qspec_map(cat, x, same)
        annotate_output_qspec(cat, same)
        _mark(cat)

    def _same_scale(self, op: Node):
        """max_pool2d / flatten / a standalone relu: output and input share a scale (6)."""
        x = op.args[0]
        assert isinstance(x, Node)
        if _done(op):
            return
        annotate_input_qspec_map(op, x, act_spec())
        annotate_output_qspec(op, SharedQuantizationSpec((x, op)))
        _mark(op)

    def validate(self, model):
        """(8) prepare_pt2e calls this after annotate: any input with a dynamic dimension (SymInt) raises an error before calibration."""
        for n in model.graph.nodes:
            val = n.meta.get("val")
            if n.op == "placeholder" and isinstance(val, torch.Tensor):
                if not all(isinstance(d, int) for d in val.shape):
                    raise ValueError(f"Input {n.name} has shape {tuple(val.shape)} with a dynamic dimension; "
                                     "this VPU takes static shapes only, so do not pass dynamic_shapes when exporting")


def vpu_ptq(model, example_inputs, n_calib=64):
    """PTQ for the VPU: torch.export.export (static shape) -> prepare_pt2e(..., MyVPUQuantizer()) -> calibration -> convert_pt2e."""
    gm = export_module(model, example_inputs, dynamic_batch=False)
    prepared = prepare_pt2e(gm, MyVPUQuantizer())
    batch = example_inputs[0].shape[0]  # static shape: the calibration batch must equal the batch used at export
    with torch.no_grad():
        for b in calibration_batches(n=n_calib // batch * batch, batch_size=batch):
            prepared(*b)
    return convert_pt2e(prepared)
