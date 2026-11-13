"""Shared by M09: export a GraphModule to ONNX and run it with ONNX Runtime (ORT); draw ONNX graphs with common.viz.draw_onnx.

onnx / onnxscript / onnxruntime are main dependencies and need no separate install.
"""

import common.pt2e  # noqa: F401  # Must come before torchao: silences noisy logs during import first

import copy
import logging
import tempfile
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, quantize_static
from torch.export import Dim

from common.data import calibration_batches
from common.pt2e import convert_pt2e, export_module, get_xnnpack_quantizer, prepare_pt2e

__all__ = ["LEVELS", "onnx_op_counts", "optimized_model", "ort_quantize", "ort_run", "ort_session",
           "to_onnx", "xnnpack_qdq"]

# ORT's five graph optimization levels (chosen when creating a session; the default is ORT_ENABLE_ALL). Each level includes the previous one.
# They map to ORT's internal TransformerLevel: Default / Level1 / Level2 / Level3 / MaxLevel (from reading the ORT 1.30 source, abi_session_options.cc)
LEVELS = {
    "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ORT_ENABLE_LAYOUT": ort.GraphOptimizationLevel.ORT_ENABLE_LAYOUT,
    "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def xnnpack_qdq(model, example_inputs, n_calib=128):
    """M09's PTQ: export (dynamic batch) → torchao test-copy XNNPACKQuantizer (per-channel) → calibration → convert.

    Returns the Q/DQ graph (GraphModule) from convert_pt2e, written the same way as ptq() in M05-1.
    """
    gm = export_module(model, example_inputs)
    quantizer, _ = get_xnnpack_quantizer(is_per_channel=True)
    prepared = prepare_pt2e(gm, quantizer)
    with torch.no_grad():
        for batch in calibration_batches(n_calib):
            prepared(*batch)
    return convert_pt2e(prepared)


def to_onnx(gm, example_inputs, dynamic_batch=True, optimize=True) -> onnx.ModelProto:
    """nn.Module or GraphModule (fp32 or Q/DQ graph) → ONNX ModelProto.

    torch.onnx.export(dynamo=True) runs torch.export again, and the dynamic batch the GraphModule originally had is not carried over,
    so it has to be stated again through dynamic_shapes (M09-1, part C).
    optimize=False turns off torch.onnx's own graph optimizations (e.g. merging identical dq nodes, folding BN into Conv first); M09-1 uses it to count the dq nodes before merging.
    """
    dynamic_shapes = tuple({0: Dim("batch")} for _ in example_inputs) if dynamic_batch else None
    prog = torch.onnx.export(gm, example_inputs, dynamo=True, verbose=False, dynamic_shapes=dynamic_shapes,
                             optimize=optimize)
    assert prog is not None
    return prog.model_proto


def ort_session(model: onnx.ModelProto, level="ORT_ENABLE_ALL", providers=("CPUExecutionProvider",),
                optimized_path=None):
    """Build an ORT InferenceSession.

    level          : a key of LEVELS. ORT rewrites the graph at this level when it creates the session (M09-2).
    optimized_path : if given, ORT saves the rewritten graph as an .onnx file (used by optimized_model()).
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = LEVELS[level]
    so.log_severity_level = 3  # errors only: when saving the optimized graph ORT warns that it "contains hardware-specific optimizations", which does not matter here
    if optimized_path:
        so.optimized_model_filepath = str(optimized_path)
    return ort.InferenceSession(model.SerializeToString(), so, providers=list(providers))


def optimized_model(model: onnx.ModelProto, level) -> onnx.ModelProto:
    """The graph ORT rewrites at this optimization level when creating a session (saved to a temp file and read back)."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "optimized.onnx"
        ort_session(model, level, optimized_path=path)
        return onnx.load(str(path))


def ort_run(sess, x: torch.Tensor) -> torch.Tensor:
    """Run one input through an ORT session (torch → numpy → torch)."""
    name = sess.get_inputs()[0].name
    return torch.from_numpy(sess.run(None, {name: x.numpy()})[0])


class CalibReader(CalibrationDataReader):
    """Feed the same calibration data PT2E uses (calibration_batches, 128 samples) to ORT's quantization tool."""

    def __init__(self, input_name):
        self.batches = iter([{input_name: x.numpy()} for (x,) in calibration_batches()])

    def get_next(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        # Return None once the data runs out and the tool stops calibrating (ORT's type annotation does not mention None)
        return next(self.batches, None)


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def ort_quantize(fp32_onnx, **kwargs):
    """quantize_static(fp32 ONNX) → (quantized ModelProto, warnings the tool printed).

    * The tool can only write to a file, so the output goes through a temp file.
    * The tool modifies the ModelProto passed in (it turns weights into external data pointing at the temp file), so a copy is passed each time.
    * The tool's warnings go through Python's root logger; they are collected first and the caller decides whether to print them (M09-3, part C).
    """
    handler, root = _Collect(), logging.getLogger()
    only_ours: list[logging.Handler] = [handler]
    saved, root.handlers = root.handlers, only_ours
    try:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "ort_quant.onnx"
            quantize_static(copy.deepcopy(fp32_onnx), out, CalibReader(fp32_onnx.graph.input[0].name), **kwargs)
            return onnx.load(str(out)), handler.messages
    finally:
        root.handlers = saved


def op_name(node) -> str:
    """Op name of an ONNX node; non-standard domains are prefixed with the domain, e.g. com.microsoft.QGemm。"""
    return f"{node.domain}.{node.op_type}" if node.domain not in ("", "ai.onnx") else node.op_type


def onnx_op_counts(model: onnx.ModelProto) -> Counter:
    return Counter(op_name(n) for n in model.graph.node)
