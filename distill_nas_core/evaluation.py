from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .artifacts import load_torch_artifact
from .resources import parameter_memory_bytes
from .scoring import causal_lm_loss
from .toy import TinyCausalLM, TinyConfig, random_token_batches
from .toy_assembly import assemble_student_from_bld


def _config_from_dict(raw: dict[str, Any]) -> TinyConfig:
    return TinyConfig(**raw)


def _load_plain_toy_model(payload: dict[str, Any]) -> TinyCausalLM:
    if "config" not in payload:
        raise ValueError("toy artifact is missing config")
    if "model_state_dict" not in payload:
        raise ValueError("toy artifact is missing model_state_dict")
    model = TinyCausalLM(_config_from_dict(payload["config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model


def load_toy_model_from_artifact(payload: dict[str, Any]) -> TinyCausalLM:
    try:
        return _load_plain_toy_model(payload)
    except RuntimeError:
        pass

    if "assembled_pth" in payload:
        assembled = load_torch_artifact(payload["assembled_pth"])
        bld = load_torch_artifact(assembled["bld_pth"])
        _teacher, student = assemble_student_from_bld(bld, assembled["architecture_config"])
    elif "bld_pth" in payload and "architecture_config" in payload:
        bld = load_torch_artifact(payload["bld_pth"])
        _teacher, student = assemble_student_from_bld(bld, payload["architecture_config"])
    else:
        raise RuntimeError("artifact has a transformed toy state_dict but no BLD/assembly metadata")
    student.load_state_dict(payload["model_state_dict"])
    return student


def evaluate_lm_model(
    model: nn.Module,
    token_batches: list[torch.Tensor],
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    device = torch.device(device)
    model = model.to(device).eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in token_batches:
            batch = batch.to(device)
            output = model(batch)
            loss = causal_lm_loss(output.logits, batch)
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("evaluation requires at least one batch")
    mean_loss = sum(losses) / len(losses)
    return {
        "lm_loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 50.0))),
        "num_batches": float(len(losses)),
    }


def evaluate_action_mse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    if predicted.shape != target.shape:
        raise ValueError(f"action shape mismatch: {tuple(predicted.shape)} vs {tuple(target.shape)}")
    return float(torch.mean((predicted.float() - target.float()) ** 2).detach().cpu())


def evaluate_toy_artifact(
    artifact_pth: str | Path,
    device: torch.device | str = "cpu",
    seq_len: int | None = None,
    batch_size: int | None = None,
    num_batches: int = 4,
    seed: int = 101,
) -> dict[str, Any]:
    payload = load_torch_artifact(artifact_pth)
    model = load_toy_model_from_artifact(payload)
    config = _config_from_dict(payload["config"])
    eval_seq_len = int(seq_len or payload.get("seq_len") or min(config.max_seq_len, 16))
    eval_batch_size = int(batch_size or payload.get("batch_size") or 2)
    batches = random_token_batches(
        config.vocab_size,
        eval_batch_size,
        eval_seq_len,
        num_batches=num_batches,
        seed=seed,
    )
    metrics = evaluate_lm_model(model, batches, device=device)
    metrics.update(
        {
            "stage": payload.get("stage"),
            "backend": payload.get("backend", "toy"),
            "seq_len": eval_seq_len,
            "batch_size": eval_batch_size,
            "parameter_memory_bytes": parameter_memory_bytes(model),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    if payload.get("losses"):
        losses = [float(item) for item in payload["losses"]]
        metrics["gkd_first_loss"] = losses[0]
        metrics["gkd_last_loss"] = losses[-1]
    return metrics


def evaluate_metadata_artifact(artifact_pth: str | Path) -> dict[str, Any]:
    payload = load_torch_artifact(artifact_pth)
    return {
        "stage": payload.get("stage"),
        "backend": payload.get("backend", "unknown"),
        "has_model_state_dict": "model_state_dict" in payload,
        "has_architecture_config": "architecture_config" in payload,
        "num_losses": len(payload.get("losses", [])) if isinstance(payload.get("losses"), list) else 0,
    }


def evaluate_artifact(
    artifact_pth: str | Path,
    backend: str = "auto",
    device: torch.device | str = "cpu",
    seq_len: int | None = None,
    batch_size: int | None = None,
    num_batches: int = 4,
    seed: int = 101,
) -> dict[str, Any]:
    payload = load_torch_artifact(artifact_pth)
    resolved_backend = backend
    if backend == "auto":
        resolved_backend = str(payload.get("backend") or "toy")
    if resolved_backend == "toy" and "model_state_dict" in payload and "config" in payload:
        return evaluate_toy_artifact(
            artifact_pth,
            device=device,
            seq_len=seq_len,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
        )
    return evaluate_metadata_artifact(artifact_pth)
