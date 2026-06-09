from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from torch import nn

from .blocks import CausalSelfAttention, TransformerBlock


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def effective_parameter_elements(module: nn.Module) -> int:
    quantized = 0
    for submodule in module.modules():
        weight_q = getattr(submodule, "weight_q", None)
        if isinstance(weight_q, torch.Tensor):
            quantized += weight_q.numel()
    return count_parameters(module) + quantized


def parameter_memory_bytes(module: nn.Module, dtype_bytes: int = 2) -> int:
    quantized = 0
    quantized_module_ids: set[int] = set()
    for submodule in module.modules():
        if hasattr(submodule, "quantized_memory_bytes") and not list(submodule.children()):
            quantized += int(submodule.quantized_memory_bytes(dtype_bytes))
            quantized_module_ids.add(id(submodule))
    dense_parameters = sum(
        parameter.numel()
        for submodule in module.modules()
        if id(submodule) not in quantized_module_ids
        for parameter in submodule.parameters(recurse=False)
    )
    return dense_parameters * dtype_bytes + quantized


def kv_cache_memory_bytes(block: nn.Module, seq_len: int, dtype_bytes: int = 2) -> int:
    if isinstance(block, TransformerBlock):
        attention = block.attention
    else:
        attention = block
    if hasattr(attention, "kv_cache_elements"):
        return int(attention.kv_cache_elements(seq_len) * dtype_bytes)
    if not isinstance(attention, CausalSelfAttention):
        return 0
    return 2 * attention.num_kv_heads * attention.head_dim * seq_len * dtype_bytes


def rough_runtime_cost(block: nn.Module, seq_len: int, batch_size: int) -> float:
    """Analytical runtime proxy used when target-hardware profiling is unavailable."""

    params = effective_parameter_elements(block)
    kv = kv_cache_memory_bytes(block, seq_len, dtype_bytes=1)
    return float((params * batch_size * seq_len) + (kv * batch_size)) / 1e9


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def profile_block_runtime(
    block: nn.Module,
    hidden_size: int,
    seq_len: int,
    batch_size: int,
    device: torch.device | str = "cpu",
    warmup: int = 2,
    steps: int = 8,
) -> float:
    """Measure average forward latency for a block on the selected device."""

    device = torch.device(device)
    block = block.to(device).eval()
    inputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            block(inputs)
        _sync(device)
        start = time.perf_counter()
        for _ in range(steps):
            block(inputs)
        _sync(device)
        elapsed = time.perf_counter() - start
    return elapsed / max(steps, 1)


def candidate_runtime_map(
    block: nn.Module,
    hidden_size: int,
    seq_len: int,
    batch_sizes: Iterable[int],
    measured: bool = False,
    device: torch.device | str = "cpu",
) -> dict[int, float]:
    runtimes: dict[int, float] = {}
    for batch_size in batch_sizes:
        if measured:
            runtimes[batch_size] = profile_block_runtime(block, hidden_size, seq_len, batch_size, device=device)
        else:
            runtimes[batch_size] = rough_runtime_cost(block, seq_len, batch_size)
    return runtimes
