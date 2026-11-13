"""One-line summary of the runtime environment, plus a quiet CPU profiler. Chapters whose numbers vary with platform and kernel (M07 on) print the environment at the top."""

import contextlib
import io
import os
import platform
import sys
import tempfile
import types

import torch
from torch.profiler import ProfilerActivity, profile


def env_summary() -> str:
    """OS / CPU architecture, torch version, thread count, and whether the two CPU backends that affect the conv implementation are available."""
    return (f"{platform.system()} {platform.machine()} | torch {torch.__version__} | threads={torch.get_num_threads()}"
            f" | mkldnn={torch.backends.mkldnn.is_available()} | nnpack={torch.backends.nnpack.is_available()}")


@contextlib.contextmanager
def capture_stderr():
    """Capture everything written to stderr inside the block; after leaving the `with`, `.text` holds the captured text.

    The ExecuTorch runtime's C++ log (e.g. `[cpuinfo_utils.cpp:…]`, `[tensor_impl.cpp:…]`) shows up as
    stderr output in a notebook. Its Python binding writes through `sys.stderr` in Jupyter, but writes
    straight to file descriptor 2 in a plain Python program, so both paths must be caught: `sys.stderr` is
    swapped for a string buffer, and fd 2 is redirected to a temporary file.
    Usage: `with capture_stderr() as log: ...`, then read `log.text`. Python warnings raised inside the block are captured too.
    """
    log = types.SimpleNamespace(text="")
    buf = io.StringIO()
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as f:
        os.dup2(f.fileno(), 2)
        try:
            with contextlib.redirect_stderr(buf):
                yield log
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            f.seek(0)
            log.text = buf.getvalue() + f.read().decode(errors="replace")


@contextlib.contextmanager
def cpu_profile():
    """`torch.profiler.profile(activities=[CPU])`, but with stderr (file descriptor 2) redirected to /dev/null inside the block.

    Kineto prints `USDT: ... profiler_start/stop` to stderr from the C layer, and no environment variable turns it off, so in a
    notebook it appears as stderr output. Used like profile: `with cpu_profile() as prof: ...`, then read `prof.events()`.
    """
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            yield prof
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
