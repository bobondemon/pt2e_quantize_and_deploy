"""QAT setup and training loop shared by the M06 scripts.

M06-1 writes the training loop out line by line in the notebook; later notebooks use `train_qat` from here instead, with the same content.
M06-3 explains why the weight fake quant does not use `FusedMovingAvgObsFakeQuantize` by default.
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

EPS = 2**-12  # matches the XNNPACKQuantizer setting

# int4: activation [-8, 7], weight [-7, 7] (symmetric). XNNPACK's config function accepts these four parameters; dtype is still an int8 container.
INT4_RANGE = dict(act_qmin=-8, act_qmax=7, weight_qmin=-7, weight_qmax=7)


def _respec(spec, ctr, **extra):
    """Keep the spec's dtype / range / qscheme and replace only observer_or_fake_quant_ctr."""
    kwargs = dict(dtype=spec.dtype, quant_min=spec.quant_min, quant_max=spec.quant_max, qscheme=spec.qscheme,
                  observer_or_fake_quant_ctr=ctr.with_args(eps=EPS, **extra))
    if spec.ch_axis is not None:
        kwargs["ch_axis"] = spec.ch_axis
    return QuantizationSpec(**kwargs)


def qat_quantizer(bits=8, weight_fq="plain", act_fq="fused", grad_scaling=False):
    """XNNPACK's QAT quantizer (per-channel weight), with a choice of fake-quant type and bit width.

    bits      : 8 or 4 (4 = shrink the integer range with INT4_RANGE).
    weight_fq : "plain"     non-fused FakeQuantize (default; consistent with convert, M06-3)
                "fused"     XNNPACK's default FusedMovingAvgObsFakeQuantize (per-channel symmetric is inconsistent with convert)
                "learnable" LearnableFakeQuantize (M06-5)
    act_fq    : "fused" (default; per-tensor affine matches the convert formula, scale may differ by 1 ULP, see M06-3 C) or "learnable".
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
    # Rebuild with cfg's own class: the executorch and torchao-test QuantizationConfig are two dataclasses of the same shape
    return quantizer.set_global(type(cfg)(act, act, weight, None, is_qat=True))


def make_schedule(observer_off_at=None, bn_eval_at=None):
    """Return an on_step(step, model) for train_qat: at the given step, freeze the observers / switch BN to eval."""

    def on_step(step, model):
        if observer_off_at is not None and step == observer_off_at:
            model.apply(disable_observer)
        if bn_eval_at is not None and step >= bn_eval_at:
            move_exported_model_to_eval(model)  # It is switched back to train at the start of every epoch, so it must be held down continuously

    return on_step


def clamp_learnable_scales(model, eps=EPS):
    """Clamp the scales learned by LearnableFakeQuantize back to positive values.

    torchao only clamps in calculate_qparams() (at convert time); during training the gradient can push scale past 0 and forward still works
    (round(x/s)·s does not care about the sign of s), but convert clamps it to eps, so the training graph and the deployed graph disagree (M06-5).
    """
    for m in model.modules():
        if isinstance(m, LearnableFakeQuantize) and m.scale is not None:
            m.scale.data.clamp_(min=eps)


def train_qat(prepared, epochs=5, lr=1e-3, on_step=None, seed=1, qparams_lr=None, log=None, clamp_scale=True):
    """QAT training loop (identical to M06-1 line by line). Returns prepared itself (trained in place, left in eval mode at the end).

    on_step(step, model) is called before each step's forward, for scheduling.
    qparams_lr : a separate lr for LearnableFakeQuantize's scale / zero_point (None = same as the weights).
    log        : optional callback log(step, model), called at the end of every step (M06-5 uses it to record the scale trajectory).
    clamp_scale: after every update, clamp LearnableFakeQuantize's scale back to positive (M06-5 shows what happens when this is turned off).
    """
    torch.manual_seed(seed)
    loader = make_loader(n=1024, seed=1)
    lossf = torch.nn.CrossEntropyLoss()
    # LearnableFakeQuantize uses lazy initialization: scale / zero_point are None at construction and
    # only become nn.Parameter (built from observer statistics) on the first forward. The optimizer only remembers the
    # Parameters that exist when it is created and does not pick up ones added later, so forward must run before building the optimizer.
    # This forward only initializes the qparams, so no autograd graph is needed.
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
        move_exported_model_to_train(prepared)  # graphs after export have no .train(): use this to switch BN
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
    """Fixed steps before evaluation or convert: BN uses running stats and observers stop updating (M06-2)."""
    move_exported_model_to_eval(prepared)
    prepared.apply(disable_observer)
    return prepared
