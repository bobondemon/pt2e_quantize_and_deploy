"""Environment self-check: if it passes, every prerequisite of the whole course project is OK."""

import importlib
import importlib.metadata
import platform
import sys

import torch

# Must come before any torchao import: when common.pt2e loads it first silences noisy logs emitted while torchao is imported
import common.pt2e  # noqa: F401


def main():
    import torchao

    print("=" * 56)
    print("Environment")
    print("=" * 56)
    print(f"  python      : {sys.version.split()[0]} ({platform.machine()})")
    print(f"  torch       : {torch.__version__}")
    print(f"  torchao     : {torchao.__version__}")

    print("\n" + "=" * 56)
    print("PT2E API locations")
    print("=" * 56)
    checks = [
        ("Main flow", "torchao.quantization.pt2e.quantize_pt2e",
         ["prepare_pt2e", "prepare_qat_pt2e", "convert_pt2e"]),
        ("Quantizer/Spec", "torchao.quantization.pt2e.quantizer",
         ["Quantizer", "QuantizationSpec", "SharedQuantizationSpec", "DerivedQuantizationSpec"]),
        ("Observer", "torchao.quantization.pt2e",
         ["MinMaxObserver", "FakeQuantize", "move_exported_model_to_eval"]),
        ("X86 Inductor", "torchao.quantization.pt2e.quantizer.x86_inductor_quantizer",
         ["X86InductorQuantizer"]),
        ("Arm Inductor", "torchao.quantization.pt2e.quantizer.arm_inductor_quantizer",
         ["ArmInductorQuantizer"]),
    ]
    for label, mod, names in checks:
        try:
            m = importlib.import_module(mod)
            missing = [n for n in names if not hasattr(m, n)]
            status = "OK" if not missing else f"missing {missing}"
        except Exception as e:  # pragma: no cover
            status = f"FAIL {type(e).__name__}"
        print(f"  [{status:>4}] {label:<16} {mod}")

    from common.pt2e import get_xnnpack_quantizer

    for source, label, note in [("torchao", "XNNPACK torchao", "used by M01 to M07"), ("executorch", "XNNPACK ET", "used by M08")]:
        try:
            _, src = get_xnnpack_quantizer(source)
            print(f"  [  OK] {label:<16} {src} ({note})")
        except Exception:
            print(f"  [ --- ] {label:<16} not installed ({note}; uv sync --group executorch)")

    for opt in ["onnx", "onnxruntime", "onnxscript", "executorch"]:
        try:
            m = importlib.import_module(opt)
            print(f"  [  OK] {opt:<16} {getattr(m, "__version__", None) or importlib.metadata.version(opt)}")
        except Exception:
            note = "  (needed by M08: uv sync --group executorch)" if opt == "executorch" else ""
            print(f"  [ --- ] {opt:<16} not installed{note}")

    print("\n" + "=" * 56)
    print("End-to-end smoke test (export -> prepare -> calibrate -> convert)")
    print("=" * 56)
    from common.data import calibration_batches
    from common.graph_utils import qdq_nodes, sqnr
    from common.models import example_inputs, get_model
    from common.pt2e import convert_pt2e, export_module, prepare_pt2e, quiet

    quiet()
    ex = example_inputs()
    fp32 = get_model("tiny").eval()
    gm = export_module(fp32, ex)
    q, _ = get_xnnpack_quantizer(is_per_channel=True)
    prepared = prepare_pt2e(gm, q)
    for batch in calibration_batches(n=32, batch_size=16):
        prepared(*batch)
    converted = convert_pt2e(prepared)
    y_fp, y_q = fp32(*ex), converted(*ex)
    print(f"  q/dq node count : {len(qdq_nodes(converted))}")
    print(f"  output SQNR     : {sqnr(y_fp, y_q):.2f} dB")
    print("\nAll checks passed ✅  You can start M00.")


if __name__ == "__main__":
    main()
