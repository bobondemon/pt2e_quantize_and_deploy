"""執行環境的一行摘要，以及安靜的 CPU profiler。數值會隨平台與 kernel 不同的教材（M07 起）在開頭印出環境。"""

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
    """作業系統／CPU 架構、torch 版本、執行緒數，以及 CPU 上兩個會影響 conv 實作的後端是否可用。"""
    return (f"{platform.system()} {platform.machine()}｜torch {torch.__version__}｜threads={torch.get_num_threads()}"
            f"｜mkldnn={torch.backends.mkldnn.is_available()}｜nnpack={torch.backends.nnpack.is_available()}")


@contextlib.contextmanager
def capture_stderr():
    """期間收下所有寫到 stderr 的內容；離開 `with` 後，`.text` 是收到的文字。

    ExecuTorch runtime 的 C++ log（例如 `[cpuinfo_utils.cpp:…]`、`[tensor_impl.cpp:…]`）在 notebook 裡會變成
    stderr 輸出。它的 Python binding 在 Jupyter 裡經由 `sys.stderr` 轉寫，在一般的 Python 程式裡則直接寫
    file descriptor 2，所以兩條都要接：`sys.stderr` 換成字串緩衝，fd 2 導到暫存檔。
    用法：`with capture_stderr() as log: ...`，之後讀 `log.text`。期間的 Python 警告也會被收下。
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
    """`torch.profiler.profile(activities=[CPU])`，但期間把 stderr（file descriptor 2）導到 /dev/null。

    Kineto 在 C 層對 stderr 印 `USDT: ... profiler_start/stop`，環境變數關不掉，在 notebook 裡會變成
    stderr 輸出。用法和 profile 相同：`with cpu_profile() as prof: ...`，之後讀 `prof.events()`。
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
