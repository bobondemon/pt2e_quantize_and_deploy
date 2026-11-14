"""Single entry point for the PT2E API (compat layer).

The PT2E API has moved house over the past two years: `torch.ao.quantization.*` -> `torchao.quantization.pt2e.*`.
The whole project imports it only here, so if the API moves again only this one file needs to change.
"""

import logging
import warnings
from typing import Any

import torch


def _quiet_noisy_dependency_loggers():
    """Silence known messages that appear while importing torch/torchao but do not affect the demos."""
    logging.getLogger(
        "torch.distributed.elastic.multiprocessing.redirects"
    ).setLevel(logging.ERROR)
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
    # On Linux (e.g. Colab) without triton, importing torchao prints "Detected no triton"; this course only uses CPU, so it is not needed.
    logging.getLogger("torchao.kernel.intmm").setLevel(logging.ERROR)


# These two messages are produced when torchao is imported below, so the filter must be applied first;
# by the time each demo calls quiet() it is too late.
_quiet_noisy_dependency_loggers()

# --- The three main-flow functions -------------------------------------------------------------
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

# --- Internal pass of convert_pt2e (used when M02-3 redoes convert by hand) --------------------------
from torchao.quantization.pt2e.quantizer.duplicate_dq_pass import DuplicateDQPass  # noqa: F401

# --- Specs used for annotation ---------------------------------------------------
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

# --- QAT switches (M02-2, M06): used with model.apply(...) ------------------------------------
from torchao.quantization.pt2e import (  # noqa: F401
    disable_fake_quant,
    disable_observer,
    disable_range_learning,
    enable_fake_quant,
    enable_observer,
    enable_range_learning,
)

# --- Numeric debugging (M05): compare outputs before and after quantization node by node ---------------------------------------
from torchao.quantization.pt2e import (  # noqa: F401
    compare_results,
    extract_results_from_loggers,
    prepare_for_propagation_comparison,
)

# --- Quantizer provided by a backend (with global / module / op type config API) --------------------
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (  # noqa: E402,F401
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

# --- Lowering (M07): hand the Q/DQ graph to the x86 Inductor int8 pattern -------------------------
from torchao.quantization.pt2e.lowering import lower_pt2e_quantized_to_x86  # noqa: E402,F401

ANNOTATION_KEY = "quantization_annotation"


def export_module(model, example_inputs, train=False, dynamic_batch=True):
    """Export an nn.Module into a GraphModule that can be fed to prepare_pt2e.

    train=True     : use the training IR (needed for QAT, so BN becomes the training version of the op).
    dynamic_batch  : make the batch dimension dynamic. export fixes every shape by default,
                     so feeding a different batch size during calibration fails an assert outright.
                     Note: edge backends (ExecuTorch / most NPUs) actually require static shapes,
                     so M08 deliberately turns this option off (M08-2).
    """
    from torch.export import Dim, export

    model = model.train() if train else model.eval()
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = tuple({0: Dim.AUTO} for _ in example_inputs)
    return export(model, example_inputs, dynamic_shapes=dynamic_shapes).module()


def get_xnnpack_api(source="torchao") -> tuple[Any, Any, str]:
    """Get (XNNPACKQuantizer class, get_symmetric_quantization_config function, source string).

    PT2E has two XNNPACKQuantizers with the same name; the code is nearly identical but the behavior differs (M08-1):
      source="torchao"    : the torchao test copy, `torchao.testing.pt2e._xnnpack_quantizer`.
                            A copy torchao keeps for its own unit tests (it emits a deprecated warning, which is suppressed here).
                            M01 to M07 always use this one so results do not change depending on whether executorch is installed;
                            its cat annotator silently fails on torch 2.14, so the max_pool2d after cat goes unannotated too (M01-2, M08-1).
      source="executorch" : the executorch copy, `executorch.backends.xnnpack.quantizer.xnnpack_quantizer`.
                            The quantizer shipped with ExecuTorch's XNNPACK backend; use this one to export a .pte (M08).
                            Requires `uv sync --group executorch`.
    The two QuantizationConfig classes are different types, so the return type is annotated as Any to spare callers from handling two types.
    """
    if source == "executorch":
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (  # pyright: ignore[reportMissingImports]
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        return XNNPACKQuantizer, get_symmetric_quantization_config, "executorch"
    if source != "torchao":
        raise ValueError(f"source must be 'torchao' or 'executorch', got {source!r}")
    from torchao.testing.pt2e._xnnpack_quantizer import (  # type: ignore
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    return XNNPACKQuantizer, get_symmetric_quantization_config, "torchao.testing"


def get_xnnpack_quantizer(source="torchao", /, **config_kwargs):
    """Get an XNNPACKQuantizer that already has set_global applied. Returns (quantizer, source_str). See get_xnnpack_api() for source."""
    quantizer_cls, config_fn, src = get_xnnpack_api(source)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*deprecated.*")
        q = quantizer_cls().set_global(config_fn(**config_kwargs))
    return q, src


def quiet():
    """Silence warnings that are noisy but harmless when running demos."""
    _quiet_noisy_dependency_loggers()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Equivalent to torch._logging.set_logs(dynamo=40, inductor=40) (that is a private API and pyright rejects it too);
    # in practice the two differ only in the level of these two loggers, so standard logging is used directly
    for name in ("torch._dynamo", "torch._inductor"):
        logging.getLogger(name).setLevel(logging.ERROR)
