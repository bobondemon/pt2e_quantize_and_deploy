"""PT2E API 的單一入口（compat layer）。

PT2E 的 API 這兩年搬過家：`torch.ao.quantization.*` -> `torchao.quantization.pt2e.*`。
全專案只在這裡 import，以後 API 再搬只要改這一個檔案。
"""

import logging
import warnings
from typing import Any

import torch


def _quiet_noisy_dependency_loggers():
    """關掉 torch/torchao 匯入時會出現、但不影響 demo 的已知訊息。"""
    logging.getLogger(
        "torch.distributed.elastic.multiprocessing.redirects"
    ).setLevel(logging.ERROR)
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)


# 這兩則訊息會在下方 import torchao 時產生，必須先套用 filter；
# 等各支 demo 呼叫 quiet() 已經太晚。
_quiet_noisy_dependency_loggers()

# --- 主流程三兄弟 -------------------------------------------------------------
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

# --- convert_pt2e 內部的 pass（M02-3 手動重做 convert 時用）--------------------------
from torchao.quantization.pt2e.quantizer.duplicate_dq_pass import DuplicateDQPass  # noqa: F401

# --- annotation 用的 spec 們 ---------------------------------------------------
from torchao.quantization.pt2e.quantizer import (  # noqa: F401
    ComposableQuantizer,
    DerivedQuantizationSpec,
    EdgeOrNode,
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationConfig,
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
    annotate_input_qspec_map,
    annotate_output_qspec,
    get_module_name_filter,
)

# --- observer / fake quant ----------------------------------------------------
from torchao.quantization.pt2e import (  # noqa: F401
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    HistogramObserver,
    LearnableFakeQuantize,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    ObserverBase,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
    allow_exported_model_train_eval,
    find_sequential_partitions,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)

# --- QAT 的開關（M02-2、M06）：對 model.apply(...) 用 ------------------------------------
from torchao.quantization.pt2e import (  # noqa: F401
    disable_fake_quant,
    disable_observer,
    disable_range_learning,
    enable_fake_quant,
    enable_observer,
    enable_range_learning,
)

# --- 數值除錯（M05）：逐 node 比較量化前後的輸出 ---------------------------------------
from torchao.quantization.pt2e import (  # noqa: F401
    compare_results,
    extract_results_from_loggers,
    prepare_for_propagation_comparison,
)

# --- 後端提供的 Quantizer（含 global / module / op type 設定 API）--------------------
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (  # noqa: E402,F401
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

# --- Lowering（M07）：把 Q/DQ 圖交給 x86 Inductor 的 int8 pattern -------------------------
from torchao.quantization.pt2e.lowering import lower_pt2e_quantized_to_x86  # noqa: E402,F401

ANNOTATION_KEY = "quantization_annotation"


def export_module(model, example_inputs, train=False, dynamic_batch=True):
    """把 nn.Module export 成可以餵給 prepare_pt2e 的 GraphModule。

    train=True     : 用 training IR（QAT 需要，BN 才會是 training 版本的 op）。
    dynamic_batch  : 讓 batch 維度是動態的。export 預設把所有 shape 寫死，
                     校準時餵不同 batch size 會直接 assert failed。
                     注意：edge 後端（ExecuTorch/多數 NPU）反而要求 static shape，
                     M08 會刻意關掉這個選項（M08-2）。
    """
    from torch.export import Dim, export

    model = model.train() if train else model.eval()
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = tuple({0: Dim.AUTO} for _ in example_inputs)
    return export(model, example_inputs, dynamic_shapes=dynamic_shapes).module()


def get_xnnpack_api(source="torchao") -> tuple[Any, Any, str]:
    """取得 (XNNPACKQuantizer 類別, get_symmetric_quantization_config 函式, 來源字串)。

    PT2E 有兩份同名的 XNNPACKQuantizer，程式碼幾乎相同、行為有差（M08-1）：
      source="torchao"    : torchao 測試版，`torchao.testing.pt2e._xnnpack_quantizer`。
                            torchao 為自己的單元測試留的拷貝（會噴 deprecated warning，這裡壓掉）。
                            M01～M07 一律用這份，結果才不會隨「有沒有裝 executorch」改變；
                            它的 cat annotator 在 torch 2.14 靜默失效，接在 cat 後的 max_pool2d 跟著沒標（M01-2、M08-1）。
      source="executorch" : executorch 版，`executorch.backends.xnnpack.quantizer.xnnpack_quantizer`。
                            ExecuTorch 的 XNNPACK 後端附的 quantizer，要匯出 .pte 就用這份（M08）。
                            需要 `uv sync --group executorch`。
    兩份的 QuantizationConfig 是不同類別，回傳型別標成 Any，讓呼叫端不必分兩種型別。
    """
    if source == "executorch":
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (  # pyright: ignore[reportMissingImports]
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        return XNNPACKQuantizer, get_symmetric_quantization_config, "executorch"
    if source != "torchao":
        raise ValueError(f"source 只能是 'torchao' 或 'executorch'，收到 {source!r}")
    from torchao.testing.pt2e._xnnpack_quantizer import (  # type: ignore
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    return XNNPACKQuantizer, get_symmetric_quantization_config, "torchao.testing"


def get_xnnpack_quantizer(source="torchao", /, **config_kwargs):
    """取得已 set_global 的 XNNPACKQuantizer。回傳 (quantizer, source_str)。source 見 get_xnnpack_api()。"""
    quantizer_cls, config_fn, src = get_xnnpack_api(source)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*deprecated.*")
        q = quantizer_cls().set_global(config_fn(**config_kwargs))
    return q, src


def quiet():
    """把跑 demo 時吵雜但無害的 warning 關掉。"""
    _quiet_noisy_dependency_loggers()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # 等同 torch._logging.set_logs(dynamo=40, inductor=40)（那是私有 API，也會被 pyright 擋），
    # 實測兩者只差在這兩個 logger 的 level，所以直接用標準 logging 設定
    for name in ("torch._dynamo", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)
