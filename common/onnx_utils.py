"""M09 共用：把 GraphModule 匯出成 ONNX、交給 ONNX Runtime（ORT）執行（畫 ONNX 圖用 common.viz.draw_onnx）。

onnx／onnxscript／onnxruntime 是主依賴，不需另外安裝。
"""

import common.pt2e  # noqa: F401  # 必須先於 torchao：先壓掉 import 期間的雜訊 log

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

# ORT 的五個圖最佳化等級（建 session 時選；預設是 ORT_ENABLE_ALL）。每一級都包含前一級的內容。
# 對應 ORT 內部的 TransformerLevel：Default／Level1／Level2／Level3／MaxLevel（讀 ORT 1.30 原始碼 abi_session_options.cc）
LEVELS = {
    "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ORT_ENABLE_LAYOUT": ort.GraphOptimizationLevel.ORT_ENABLE_LAYOUT,
    "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def xnnpack_qdq(model, example_inputs, n_calib=128):
    """M09 的 PTQ：export（動態 batch）→ torchao 測試版 XNNPACKQuantizer（per-channel）→ 校準 → convert。

    回傳 convert_pt2e 的 Q/DQ 圖（GraphModule），寫法同 M05-1 的 ptq()。
    """
    gm = export_module(model, example_inputs)
    quantizer, _ = get_xnnpack_quantizer(is_per_channel=True)
    prepared = prepare_pt2e(gm, quantizer)
    with torch.no_grad():
        for batch in calibration_batches(n_calib):
            prepared(*batch)
    return convert_pt2e(prepared)


def to_onnx(gm, example_inputs, dynamic_batch=True, optimize=True) -> onnx.ModelProto:
    """nn.Module 或 GraphModule（fp32 或 Q/DQ 圖）→ ONNX ModelProto。

    torch.onnx.export(dynamo=True) 會重新 torch.export 一次，GraphModule 上原本的動態 batch 不會帶過來，
    所以要用 dynamic_shapes 再講一次（M09-1 C 段）。
    optimize=False 關掉 torch.onnx 自己的圖最佳化（例如先把 BN 折進 Conv），M09-3 用來看 ORT 自己折 BN。
    """
    dynamic_shapes = tuple({0: Dim("batch")} for _ in example_inputs) if dynamic_batch else None
    prog = torch.onnx.export(gm, example_inputs, dynamo=True, verbose=False, dynamic_shapes=dynamic_shapes,
                             optimize=optimize)
    assert prog is not None
    return prog.model_proto


def ort_session(model: onnx.ModelProto, level="ORT_ENABLE_ALL", providers=("CPUExecutionProvider",),
                optimized_path=None):
    """建 ORT 的 InferenceSession。

    level          : LEVELS 的 key。ORT 在建 session 時依這個等級改寫圖（M09-2）。
    optimized_path : 給路徑的話，ORT 把改寫後的圖存成 .onnx（optimized_model() 用）。
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = LEVELS[level]
    so.log_severity_level = 3  # 只印 error：存最佳化圖時 ORT 會警告「含硬體相關的最佳化」，這裡用不到
    if optimized_path:
        so.optimized_model_filepath = str(optimized_path)
    return ort.InferenceSession(model.SerializeToString(), so, providers=list(providers))


def optimized_model(model: onnx.ModelProto, level) -> onnx.ModelProto:
    """ORT 在這個最佳化等級下、建 session 時改寫出來的圖（存到暫存檔再讀回）。"""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "optimized.onnx"
        ort_session(model, level, optimized_path=path)
        return onnx.load(str(path))


def ort_run(sess, x: torch.Tensor) -> torch.Tensor:
    """用 ORT session 跑一個輸入（torch → numpy → torch）。"""
    name = sess.get_inputs()[0].name
    return torch.from_numpy(sess.run(None, {name: x.numpy()})[0])


class CalibReader(CalibrationDataReader):
    """把 PT2E 用的同一份校準資料（calibration_batches，128 筆）餵給 ORT 的量化工具。"""

    def __init__(self, input_name):
        self.batches = iter([{input_name: x.numpy()} for (x,) in calibration_batches()])

    def get_next(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        # 資料讀完回傳 None，工具就停止校準（ORT 的型別標註沒寫 None）
        return next(self.batches, None)


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def ort_quantize(fp32_onnx, **kwargs):
    """quantize_static(fp32 ONNX) → (量化後的 ModelProto, 工具印出的警告)。

    * 工具只能寫檔，所以輸出經過暫存檔。
    * 工具會改動傳進去的 ModelProto（把 weight 改成指向暫存檔的 external data），所以每次傳一份拷貝。
    * 工具的警告走 Python 的 root logger；先收起來，由呼叫端決定要不要印（M09-4 C 段）。
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
    """ONNX node 的 op 名稱；不是標準 domain 的加上 domain，例如 com.microsoft.QGemm。"""
    return f"{node.domain}.{node.op_type}" if node.domain not in ("", "ai.onnx") else node.op_type


def onnx_op_counts(model: onnx.ModelProto) -> Counter:
    return Counter(op_name(n) for n in model.graph.node)
