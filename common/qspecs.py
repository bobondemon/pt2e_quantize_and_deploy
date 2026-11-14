"""QuantizationSpecs shared by the scripts from M03 on.

A qspec = "how to quantize this location": dtype, integer range, quantization scheme, and which observer collects the statistics.
Only the two most common ones live here; the rest (Shared / Derived / FixedQParams) are built by each script as needed.
"""

import torch

from common.pt2e import MinMaxObserver, PerChannelMinMaxObserver, QuantizationSpec


def act_spec() -> QuantizationSpec:
    """Activation: int8, per-tensor affine (zero_point may be non-zero)."""
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MinMaxObserver,
    )


def weight_spec() -> QuantizationSpec:
    """Weight: int8, per-channel symmetric (zero_point = 0), along output channel axis 0.

    The range is [-127, 127] rather than [-128, 127], keeping positive and negative sides symmetric (common in hardware implementations).
    """
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver,
    )
