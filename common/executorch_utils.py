"""M08 共用：把 GraphModule 交給 ExecuTorch 的 XNNPACK 後端、產生 .pte，再用 ExecuTorch runtime 執行。

需要 `uv sync --group executorch`；只有 M08 會 import 這個檔案。
"""

import common.pt2e  # noqa: F401  # 必須先於 executorch（它會 import torchao）：先壓掉 import 期間的雜訊 log

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner  # pyright: ignore[reportMissingImports]
from executorch.devtools.backend_debug import get_delegation_info  # pyright: ignore[reportMissingImports]
from executorch.exir import to_edge_transform_and_lower  # pyright: ignore[reportMissingImports]
from executorch.runtime import Runtime  # pyright: ignore[reportMissingImports]

from common.data import calibration_batches
from common.env import capture_stderr
from common.pt2e import convert_pt2e, export_module, get_xnnpack_quantizer, prepare_pt2e

__all__ = ["delegation_summary", "get_delegation_info", "lower_to_xnnpack", "PteRunner", "xnnpack_ptq"]


def xnnpack_ptq(model, example_inputs, n_calib=128):
    """M08 的 PTQ：static shape export → executorch 版 XNNPACKQuantizer → 校準 → convert，回傳 Q/DQ 圖。

    edge 後端要 static shape，所以 export 時關掉動態 batch（M00 的伏筆）；圖只吃 example_inputs 的 batch 大小，
    校準資料要切成同樣大小再餵。
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
    """GraphModule（fp32 或 convert 後的 Q/DQ 圖）→ (edge program manager, .pte 的 bytes)。

    三步：
      torch.export.export          : 重新 export 成 ExportedProgram（shape 依 example_inputs 寫死）
      to_edge_transform_and_lower  : 轉成 Edge dialect，XnnpackPartitioner 把它吃得下的子圖切出來、交給 XNNPACK 編譯
      to_executorch                : 剩下的圖做 memory planning，序列化成 .pte
    """
    ep = torch.export.export(gm, example_inputs)
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    return edge, edge.to_executorch().buffer


def delegation_summary(edge):
    """回傳 (delegate 個數, 交給 XNNPACK 的 node 數, 留在 ExecuTorch 的 node 數, {op: (交給 XNNPACK, 留下)})。

    op 名稱是 Edge dialect 的寫法，例如 aten_convolution_default（conv2d 在 Edge 已 decompose 成 convolution）。
    """
    info = get_delegation_info(edge.exported_program().graph_module)
    by_op = {k: (v.delegated, v.non_delegated) for k, v in info.delegation_by_operator.items()}
    return info.num_delegated_subgraphs, info.num_delegated_nodes, info.num_non_delegated_nodes, by_op


class PteRunner:
    """載入 .pte、執行 forward。

    注意：`Runtime.load_program()` 回傳的 Program 必須一直留著參照。只留 `load_method()` 的 Method、
    讓 Program 被回收的話，Method 執行時會讀到已釋放的記憶體（本專案試作時出現莫名的 resize 錯誤）。
    """

    def __init__(self, pte):
        """pte：.pte 的 bytes，或 .pte 檔案路徑。

        第一次載入 XNNPACK delegate 時，ExecuTorch 的 C++ 層會在 stderr 印幾行 `[cpuinfo_utils.cpp:…]`
        （偵測 CPU 核心），與結果無關，這裡收起來。
        """
        with capture_stderr():
            self.program = Runtime.get().load_program(pte)
            method = self.program.load_method("forward")
        assert method is not None, ".pte 裡沒有 forward method"
        self.method = method

    def __call__(self, *inputs):
        return self.method.execute(list(inputs))[0]
