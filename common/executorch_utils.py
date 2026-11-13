"""Shared by M08: hand a GraphModule to ExecuTorch's XNNPACK backend, produce a .pte, then run it with the ExecuTorch runtime.

Requires `uv sync --group executorch`; only M08 imports this file.
"""

import common.pt2e  # noqa: F401  # Must come before executorch (which imports torchao): silences noisy logs during import first

import torch
import executorch.backends.xnnpack.xnnpack_preprocess as _xnn_preprocess  # pyright: ignore[reportMissingImports]
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner  # pyright: ignore[reportMissingImports]
from executorch.devtools.backend_debug import get_delegation_info  # pyright: ignore[reportMissingImports]
from executorch.exir import to_edge_transform_and_lower  # pyright: ignore[reportMissingImports]
from executorch.runtime import Runtime  # pyright: ignore[reportMissingImports]

from common.data import calibration_batches
from common.env import capture_stderr
from common.pt2e import convert_pt2e, export_module, get_xnnpack_quantizer, prepare_pt2e

__all__ = ["delegation_summary", "get_delegation_info", "lower_to_xnnpack", "PteRunner", "xnnpack_nodes", "xnnpack_ptq"]


def xnnpack_ptq(model, example_inputs, n_calib=128):
    """M08's PTQ: static-shape export → executorch's XNNPACKQuantizer → calibration → convert; returns the Q/DQ graph.

    Edge backends need static shapes, so dynamic batch is turned off at export (set up in M00); the graph only accepts the batch size
    of example_inputs, so calibration data must be split to the same size before feeding it.
    """
    gm = export_module(model, example_inputs, dynamic_batch=False)
    quantizer, _ = get_xnnpack_quantizer("executorch", is_per_channel=True)
    prepared = prepare_pt2e(gm, quantizer)
    bs = example_inputs[0].shape[0]
    with torch.no_grad():
        for (x,) in calibration_batches(n_calib):
            for i in range(0, len(x), bs):
                prepared(x[i : i + bs])
    return convert_pt2e(prepared)


def lower_to_xnnpack(gm, example_inputs):
    """GraphModule (fp32 or the Q/DQ graph after convert) → (edge program manager, bytes of the .pte).

    Three steps:
      torch.export.export          : re-export as an ExportedProgram (shapes fixed by example_inputs)
      to_edge_transform_and_lower  : convert to the Edge dialect; XnnpackPartitioner cuts out the subgraphs it can handle and hands them to XNNPACK to compile
      to_executorch                : do memory planning on the remaining graph and serialize it to a .pte
    """
    ep = torch.export.export(gm, example_inputs)
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    return edge, edge.to_executorch().buffer


def delegation_summary(edge):
    """Return (number of delegates, nodes handed to XNNPACK, nodes left in ExecuTorch, {op: (handed to XNNPACK, left behind)}).

    Op names use the Edge dialect spelling, e.g. aten_convolution_default (conv2d is already decomposed into convolution in Edge).
    """
    info = get_delegation_info(edge.exported_program().graph_module)
    by_op = {k: (v.delegated, v.non_delegated) for k, v in info.delegation_by_operator.items()}
    return info.num_delegated_subgraphs, info.num_delegated_nodes, info.num_non_delegated_nodes, by_op


def xnnpack_nodes(gm, example_inputs):
    """Lower once more and return the XNNPACK nodes in each delegate: [[(node, input dtype, output dtype, output clamp), ...], ...].

    During lowering the XNNPACK backend rebuilds each delegate's subgraph as its own graph (XNNGraph) and then serializes it into the .pte;
    here that graph is intercepted just before serialization. dtype uses XNNPACK's spelling: qint8 (per-tensor int8), qcint8 / qcint32
    (per-channel int8 weight / int32 bias), fp32. clamp is the output range fused into the node (e.g. relu → [0, +inf]).

    Note: this depends on executorch 1.5.0 internals (`xnnpack_preprocess.serialize_xnnpack_binary` and the XNNGraph fields),
    which are not public API and may need changes on other versions.
    """
    graphs = []
    original = _xnn_preprocess.serialize_xnnpack_binary

    def spy(xnn_graph, *args, **kwargs):
        graphs.append(xnn_graph)
        return original(xnn_graph, *args, **kwargs)

    _xnn_preprocess.serialize_xnnpack_binary = spy
    try:
        lower_to_xnnpack(gm, example_inputs)
    finally:
        _xnn_preprocess.serialize_xnnpack_binary = original
    return [_xnn_rows(g) for g in graphs]


def _xnn_rows(xnn_graph):
    dtype = {}
    for v in xnn_graph.xvalues:
        tensor = getattr(v.xvalue_union, "tensor_value", v.xvalue_union)  # quantized tensors are wrapped one level deeper
        dtype[tensor.id_out] = tensor.datatype.name.removeprefix("xnn_datatype_")
    rows = []
    for node in xnn_graph.xnodes:
        op = node.xnode_union
        ins = [dtype[v] for k, v in vars(op).items() if k.endswith("_id") and k != "output_id" and isinstance(v, int)]
        mm = node.output_min_max
        clamp = f"[{mm.output_min}, {mm.output_max}]" if mm is not None else ""
        rows.append((type(op).__name__, ", ".join(ins), dtype[op.output_id], clamp))
    return rows


class PteRunner:
    """Load a .pte and run forward.

    Note: keep a reference to the Program returned by `Runtime.load_program()` for as long as the runner lives. If only the Method from `load_method()`
    is kept and the Program gets garbage-collected, the Method reads freed memory when it runs (this project hit a puzzling resize error while prototyping).
    """

    def __init__(self, pte):
        """pte: the bytes of a .pte, or the path to a .pte file.

        The first time the XNNPACK delegate is loaded, ExecuTorch's C++ layer prints a few `[cpuinfo_utils.cpp:…]` lines to stderr
        (CPU core detection); they are unrelated to the result, so they are swallowed here.
        """
        with capture_stderr():
            self.program = Runtime.get().load_program(pte)
            method = self.program.load_method("forward")
        assert method is not None, "no forward method in the .pte"
        self.method = method

    def __call__(self, *inputs):
        return self.method.execute(list(inputs))[0]
