from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .artifacts import load_torch_artifact
from .evaluation import load_toy_model_from_artifact
from .resources import parameter_memory_bytes
from .toy import TinyConfig, random_token_batches


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _config_from_dict(raw: dict[str, Any]) -> TinyConfig:
    return TinyConfig(**raw)


def profile_model_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    warmup: int = 1,
    steps: int = 5,
) -> dict[str, float]:
    device = input_ids.device
    model = model.to(device).eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(steps):
            model(input_ids)
        _sync(device)
        elapsed = time.perf_counter() - started
    result = {
        "latency_seconds": elapsed / max(steps, 1),
        "throughput_tokens_per_second": float(input_ids.numel() * max(steps, 1) / max(elapsed, 1e-12)),
    }
    if device.type == "cuda":
        result["peak_memory_bytes"] = float(torch.cuda.max_memory_allocated(device))
    return result


def profile_toy_artifact(
    artifact_pth: str | Path,
    device: torch.device | str = "cpu",
    batch_sizes: list[int] | None = None,
    seq_len: int | None = None,
    warmup: int = 1,
    steps: int = 5,
) -> dict[str, Any]:
    payload = load_torch_artifact(artifact_pth)
    if "model_state_dict" not in payload:
        raise ValueError("toy profiling requires a model_state_dict artifact")
    config = _config_from_dict(payload["config"])
    model = load_toy_model_from_artifact(payload)
    resolved_device = torch.device(device)
    batch_sizes = batch_sizes or [1, 2, 4]
    resolved_seq_len = int(seq_len or payload.get("seq_len") or min(config.max_seq_len, 16))
    profiles: dict[str, dict[str, float]] = {}
    for batch_size in batch_sizes:
        batch = random_token_batches(
            config.vocab_size,
            batch_size,
            resolved_seq_len,
            num_batches=1,
            seed=202,
        )[0].to(resolved_device)
        profiles[str(batch_size)] = profile_model_forward(model, batch, warmup=warmup, steps=steps)
    return {
        "stage": payload.get("stage"),
        "backend": payload.get("backend", "toy"),
        "device": str(resolved_device),
        "seq_len": resolved_seq_len,
        "batch_sizes": batch_sizes,
        "parameter_memory_bytes": parameter_memory_bytes(model),
        "profiles": profiles,
    }


def profile_artifact(
    artifact_pth: str | Path,
    backend: str = "auto",
    device: torch.device | str = "cpu",
    batch_sizes: list[int] | None = None,
    seq_len: int | None = None,
    warmup: int = 1,
    steps: int = 5,
) -> dict[str, Any]:
    payload = load_torch_artifact(artifact_pth)
    resolved_backend = backend
    if backend == "auto":
        resolved_backend = str(payload.get("backend") or "toy")
    if resolved_backend == "toy" and "model_state_dict" in payload:
        return profile_toy_artifact(
            artifact_pth,
            device=device,
            batch_sizes=batch_sizes,
            seq_len=seq_len,
            warmup=warmup,
            steps=steps,
        )
    return {
        "stage": payload.get("stage"),
        "backend": resolved_backend,
        "profiled": False,
        "reason": "metadata-only artifact profiling is available for non-toy backends",
    }
