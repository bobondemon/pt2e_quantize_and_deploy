"""M03 起各支腳本共用的 QuantizationSpec。

一份 qspec = 「這個位置要怎麼量化」：dtype、整數範圍、量化方案，以及用哪種 observer 收集統計。
這裡只放兩個最常用的，其餘（Shared / Derived / FixedQParams）由各腳本依需要自己建。
"""

import torch

from common.pt2e import MinMaxObserver, PerChannelMinMaxObserver, QuantizationSpec


def act_spec() -> QuantizationSpec:
    """activation：int8、per-tensor affine（zero_point 可非 0）。"""
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MinMaxObserver,
    )


def weight_spec() -> QuantizationSpec:
    """weight：int8、per-channel symmetric（zero_point = 0），沿 output channel 軸 0。

    範圍用 [-127, 127] 而不是 [-128, 127]，讓正負對稱（常見於硬體實作）。
    """
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    )
