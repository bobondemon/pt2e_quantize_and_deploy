"""Train the course's TinyConvNet once and save its weights to common/tiny_net.pt.

Why: training is not bit-reproducible across platforms (or even thread counts), but the notebooks' numbers
(SQNR, per-layer rankings, ...) depend on the exact weights. Committing the weights makes every platform
quantize the same model. `common.data.trained_tiny_net()` loads this file when it exists.

Run it on the machine whose outputs are stored in the notebooks (macOS arm64), from the repo root:

    uv run python tools/make_tiny_net_weights.py            # trains from scratch, writes common/tiny_net.pt
    uv run python tools/make_tiny_net_weights.py --check    # only loads the file and prints the check numbers

Then paste the printed "check numbers" back so they can be compared with the stored outputs
(M05-00 expects an output SQNR of about 27.35 dB).
"""
import argparse
import copy
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable without installing

from common.data import (TINY_NET_WEIGHTS, _train_tiny_net, calibration_batches, evaluate, evaluate_loss,
                         make_loader, trained_tiny_net)
from common.graph_utils import sqnr
from common.models import example_inputs
from common.pt2e import convert_pt2e, export_module, get_xnnpack_quantizer, prepare_pt2e, quiet

EPOCHS, SEED = 20, 0  # the defaults of trained_tiny_net(); the notebooks rely on them


def check_numbers(model):
    """Numbers to compare with the stored notebook outputs (same recipe as M05-00 cell 5)."""
    quiet()
    prepared = prepare_pt2e(export_module(model, example_inputs()), get_xnnpack_quantizer(is_per_channel=True)[0])
    with torch.no_grad():
        for batch in calibration_batches(128):
            prepared(*batch)
    converted = convert_pt2e(copy.deepcopy(prepared))
    test_loader = make_loader(n=1024, seed=2, shuffle=False)
    (x_eval,) = next(calibration_batches(64, seed=9))
    with torch.no_grad():
        out_sqnr = sqnr(model(x_eval), converted(x_eval))
    return {
        "fp32 accuracy": round(evaluate(model, test_loader), 4),
        "int8 accuracy": round(evaluate(converted, test_loader), 4),
        "fp32 test loss": round(evaluate_loss(model, test_loader), 4),
        "M05-00 output SQNR (dB)": round(float(out_sqnr), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="do not train; load the existing file and print the check numbers")
    args = ap.parse_args()

    if args.check:
        if not TINY_NET_WEIGHTS.exists():
            sys.exit(f"{TINY_NET_WEIGHTS} does not exist; run without --check first")
        model = trained_tiny_net()
        print(f"loaded {TINY_NET_WEIGHTS}")
    else:
        if TINY_NET_WEIGHTS.exists():
            print(f"note: overwriting {TINY_NET_WEIGHTS}")
        model = _train_tiny_net(EPOCHS, SEED)  # always train from scratch, ignoring any existing file
        payload = {
            "state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            "meta": {"epochs": EPOCHS, "seed": SEED, "torch": str(torch.__version__),
                     "platform": f"{platform.system()} {platform.machine()}", "threads": torch.get_num_threads()},
        }
        torch.save(payload, TINY_NET_WEIGHTS)
        print(f"wrote {TINY_NET_WEIGHTS} ({TINY_NET_WEIGHTS.stat().st_size} bytes)")
        model = trained_tiny_net()  # reload through the normal path, so the numbers below are what the notebooks will see

    print(f"{platform.system()} {platform.machine()} | torch {torch.__version__} | threads={torch.get_num_threads()}")
    print("check numbers:")
    for k, v in check_numbers(model).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
