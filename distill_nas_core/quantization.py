from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import QuantizedLinear


def quantize_linear(linear: nn.Linear, num_bits: int = 8) -> QuantizedLinear:
    return QuantizedLinear.from_linear(linear, num_bits=num_bits)


def quantization_mse(
    linear: nn.Linear,
    inputs: Iterable[torch.Tensor],
    num_bits: int = 8,
    device: torch.device | str = "cpu",
) -> float:
    device = torch.device(device)
    linear = linear.to(device).eval()
    quantized = quantize_linear(linear, num_bits=num_bits).to(device).eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in inputs:
            batch = batch.to(device)
            reference = linear(batch)
            candidate = quantized(batch)
            losses.append(float(F.mse_loss(candidate.float(), reference.float()).detach().cpu()))
    if not losses:
        raise ValueError("quantization_mse requires at least one input batch")
    return sum(losses) / len(losses)


def should_fallback_to_dense(
    linear: nn.Linear,
    inputs: Iterable[torch.Tensor],
    max_mse: float,
    num_bits: int = 8,
    device: torch.device | str = "cpu",
) -> bool:
    return quantization_mse(linear, inputs, num_bits=num_bits, device=device) > max_mse
