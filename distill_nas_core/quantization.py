from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import QuantizedLinear


@dataclass(frozen=True)
class QuantizationDecision:
    plan: str
    use_quantized: bool
    mse: float | None = None
    num_bits: int = 8
    max_mse: float | None = None
    metadata: dict[str, Any] | None = None


QuantizationPlanner = Callable[[nn.Linear, Iterable[torch.Tensor], dict[str, Any]], QuantizationDecision]
QUANTIZATION_PLANS: dict[str, QuantizationPlanner] = {}


def register_quantization_plan(name: str, planner: QuantizationPlanner) -> None:
    if not name:
        raise ValueError("quantization plan name must be non-empty")
    QUANTIZATION_PLANS[name] = planner


def get_quantization_plan(name: str) -> QuantizationPlanner:
    try:
        return QUANTIZATION_PLANS[name]
    except KeyError as exc:
        raise ValueError(f"unknown quantization plan: {name}") from exc


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


def evaluate_quantization_plan(
    linear: nn.Linear,
    inputs: Iterable[torch.Tensor],
    plan: str = "mse_threshold",
    **options: Any,
) -> QuantizationDecision:
    return get_quantization_plan(plan)(linear, inputs, options)


def apply_quantization_plan(
    linear: nn.Linear,
    inputs: Iterable[torch.Tensor],
    plan: str = "mse_threshold",
    **options: Any,
) -> nn.Module:
    decision = evaluate_quantization_plan(linear, inputs, plan=plan, **options)
    if decision.use_quantized:
        return quantize_linear(linear, num_bits=decision.num_bits)
    return linear


def _always_quantize_plan(
    _linear: nn.Linear,
    _inputs: Iterable[torch.Tensor],
    options: dict[str, Any],
) -> QuantizationDecision:
    num_bits = int(options.get("num_bits", 8))
    return QuantizationDecision(plan="always", use_quantized=True, num_bits=num_bits)


def _mse_threshold_plan(
    linear: nn.Linear,
    inputs: Iterable[torch.Tensor],
    options: dict[str, Any],
) -> QuantizationDecision:
    num_bits = int(options.get("num_bits", 8))
    max_mse = float(options.get("max_mse", 1e-4))
    device = options.get("device", "cpu")
    mse = quantization_mse(linear, inputs, num_bits=num_bits, device=device)
    return QuantizationDecision(
        plan="mse_threshold",
        use_quantized=mse <= max_mse,
        mse=mse,
        num_bits=num_bits,
        max_mse=max_mse,
    )


register_quantization_plan("always", _always_quantize_plan)
register_quantization_plan("mse_threshold", _mse_threshold_plan)
