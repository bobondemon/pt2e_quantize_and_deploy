"""M06 各支腳本共用的 QAT 設定與訓練迴圈。

M06-1 把訓練迴圈逐行寫在 notebook 裡；之後的 notebook 改用這裡的 `train_qat`，內容相同。
M06-3 說明為什麼 weight 的 fake quant 預設不用 `FusedMovingAvgObsFakeQuantize`。
"""

from typing import Any

import torch

from common.data import make_loader
from common.models import example_inputs
from common.pt2e import (
    FakeQuantize,
    LearnableFakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QuantizationSpec,
    disable_observer,
    get_xnnpack_quantizer,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)

EPS = 2**-12  # 與 XNNPACKQuantizer 的設定一致

# int4：activation [-8, 7]、weight [-7, 7]（對稱）。XNNPACK 的 config 函式接受這四個參數，dtype 仍是 int8 容器。
INT4_RANGE = dict(act_qmin=-8, act_qmax=7, weight_qmin=-7, weight_qmax=7)


def _respec(spec, ctr, **extra):
    """保留 spec 的 dtype／範圍／qscheme，只換掉 observer_or_fake_quant_ctr。"""
    kwargs = dict(dtype=spec.dtype, quant_min=spec.quant_min, quant_max=spec.quant_max, qscheme=spec.qscheme,
                  observer_or_fake_quant_ctr=ctr.with_args(eps=EPS, **extra))
    if spec.ch_axis is not None:
        kwargs["ch_axis"] = spec.ch_axis
    return QuantizationSpec(**kwargs)


def qat_quantizer(bits=8, weight_fq="plain", act_fq="fused", grad_scaling=False):
    """XNNPACK 的 QAT quantizer（per-channel weight），可換 fake quant 的種類與位元數。

    bits      : 8 或 4（4 = 用 INT4_RANGE 縮小整數範圍）。
    weight_fq : "plain"     非 fused 的 FakeQuantize（預設；與 convert 一致，M06-3）
                "fused"     XNNPACK 預設的 FusedMovingAvgObsFakeQuantize（per-channel symmetric 與 convert 不一致）
                "learnable" LearnableFakeQuantize（M06-5）
    act_fq    : "fused"（預設，per-tensor affine 與 convert 公式一致，scale 可差 1 ULP，見 M06-3 C）或 "learnable"。
    """
    kw: dict[str, Any] = dict(is_per_channel=True, is_qat=True)
    if bits == 4:
        kw.update(INT4_RANGE)
    quantizer, _ = get_xnnpack_quantizer(**kw)
    cfg = quantizer.global_config
    assert cfg is not None and cfg.input_activation is not None and cfg.weight is not None
    act, weight = cfg.input_activation, cfg.weight
    if weight_fq == "plain":
        weight = _respec(weight, FakeQuantize, observer=MovingAveragePerChannelMinMaxObserver)
    elif weight_fq == "learnable":
        weight = _respec(weight, LearnableFakeQuantize, observer=MovingAveragePerChannelMinMaxObserver,
                         ch_axis=0, use_grad_scaling=grad_scaling)
    if act_fq == "learnable":
        act = _respec(act, LearnableFakeQuantize, observer=MovingAverageMinMaxObserver, use_grad_scaling=grad_scaling)
    # 用 cfg 自己的類別重建：executorch 版與 torchao 測試版的 QuantizationConfig 是兩個同形的 dataclass
    return quantizer.set_global(type(cfg)(act, act, weight, None, is_qat=True))


def make_schedule(observer_off_at=None, bn_eval_at=None):
    """回傳給 train_qat 的 on_step(step, model)：到了指定 step 就凍結 observer／把 BN 切到 eval。"""

    def on_step(step, model):
        if observer_off_at is not None and step == observer_off_at:
            model.apply(disable_observer)
        if bn_eval_at is not None and step >= bn_eval_at:
            move_exported_model_to_eval(model)  # 每個 epoch 開頭會被切回 train，所以要持續壓住

    return on_step


def clamp_learnable_scales(model, eps=EPS):
    """把 LearnableFakeQuantize 學出來的 scale 夾回正數。

    torchao 只在 calculate_qparams()（convert 時）才 clamp；訓練時梯度可以把 scale 推過 0，forward 照常
    （round(x/s)·s 不在乎 s 的正負），convert 卻會把它夾成 eps，訓練圖與部署圖就不一致了（M06-5）。
    """
    for m in model.modules():
        if isinstance(m, LearnableFakeQuantize) and m.scale is not None:
            m.scale.data.clamp_(min=eps)


def train_qat(prepared, epochs=5, lr=1e-3, on_step=None, seed=1, qparams_lr=None, log=None, clamp_scale=True):
    """QAT 訓練迴圈（與 M06-1 逐行相同）。回傳 prepared 本身（已原地訓練，最後處於 eval 模式）。

    on_step(step, model) 在每個 step 的 forward 之前呼叫，排程用。
    qparams_lr : LearnableFakeQuantize 的 scale／zero_point 另給 lr（None = 與 weight 相同）。
    log        : 可選的 callback log(step, model)，每個 step 結束後呼叫（M06-5 用來記 scale 軌跡）。
    clamp_scale: 每步更新後把 LearnableFakeQuantize 的 scale 夾回正數（M06-5 示範關掉會怎樣）。
    """
    torch.manual_seed(seed)
    loader = make_loader(n=1024, seed=1)
    lossf = torch.nn.CrossEntropyLoss()
    # LearnableFakeQuantize 採 lazy initialization：建構時 scale／zero_point 是 None，
    # 第一次 forward 才根據 observer 統計建立成 nn.Parameter。optimizer 只記住建立當下
    # 已存在的 Parameter，不會自動收進之後新增的參數，所以必須先 forward 再建 optimizer。
    # 這次 forward 只為初始化 qparams，不需要建立 autograd graph。
    with torch.no_grad():
        prepared(*example_inputs())
    if qparams_lr is None:
        opt = torch.optim.SGD(prepared.parameters(), lr=lr, momentum=0.9)
    else:
        is_q = lambda name: name.endswith((".scale", ".zero_point"))  # noqa: E731
        groups = [{"params": [p for n, p in prepared.named_parameters() if not is_q(n)]},
                  {"params": [p for n, p in prepared.named_parameters() if is_q(n)], "lr": qparams_lr}]
        opt = torch.optim.SGD(groups, lr=lr, momentum=0.9)
    step = 0
    for _ in range(epochs):
        move_exported_model_to_train(prepared)  # export 後的圖沒有 .train()：BN 要用這個切
        for x, y in loader:
            if on_step is not None:
                on_step(step, prepared)
            opt.zero_grad()
            lossf(prepared(x), y).backward()
            opt.step()
            if clamp_scale:
                clamp_learnable_scales(prepared)
            step += 1
            if log is not None:
                log(step, prepared)
        move_exported_model_to_eval(prepared)
    return prepared


def freeze_for_eval(prepared):
    """評估或 convert 前的固定動作：BN 用 running stats、observer 不再更新（M06-2）。"""
    move_exported_model_to_eval(prepared)
    prepared.apply(disable_observer)
    return prepared
