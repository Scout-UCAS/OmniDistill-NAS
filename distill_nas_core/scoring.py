from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .distill import batch_input_ids, forward_batch, logits_kl_loss, move_batch_to_device


@contextmanager
def replace_block(model: nn.Module, layer_idx: int, block: nn.Module):
    original = model.blocks[layer_idx]
    model.blocks[layer_idx] = block
    try:
        yield
    finally:
        model.blocks[layer_idx] = original


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def score_replace_one_block(
    parent_model: nn.Module,
    layer_idx: int,
    candidate_block: nn.Module,
    token_batches: Iterable[torch.Tensor],
    metric: str = "kl",
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> float:
    """Evaluate the paper's replace-1-block score for a single candidate."""

    if metric not in {"kl", "lm_loss"}:
        raise ValueError("metric must be 'kl' or 'lm_loss'")

    device = torch.device(device)
    parent_model = parent_model.to(device).eval()
    candidate_block = candidate_block.to(device).eval()
    values: list[float] = []
    original_block = parent_model.blocks[layer_idx]

    with torch.no_grad():
        try:
            for batch_idx, batch in enumerate(token_batches):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = move_batch_to_device(batch, device)
                input_ids = batch_input_ids(batch)

                parent_model.blocks[layer_idx] = original_block
                parent_out = forward_batch(parent_model, batch) if metric == "kl" else None

                parent_model.blocks[layer_idx] = candidate_block
                candidate_out = forward_batch(parent_model, batch)

                if metric == "lm_loss":
                    if input_ids is None:
                        raise ValueError("lm_loss scoring requires input_ids in each batch")
                    score = causal_lm_loss(candidate_out.logits, input_ids)
                else:
                    assert parent_out is not None
                    score = logits_kl_loss(parent_out.logits, candidate_out.logits)
                values.append(float(score.detach().cpu()))
        finally:
            parent_model.blocks[layer_idx] = original_block

    if not values:
        raise ValueError("token_batches produced no batches")
    return sum(values) / len(values)


def score_candidates(
    parent_model: nn.Module,
    candidates: Iterable[tuple[int, str, nn.Module]],
    token_batches: list[torch.Tensor],
    metric: str = "kl",
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for layer_idx, name, block in candidates:
        scores[name] = score_replace_one_block(
            parent_model,
            layer_idx,
            block,
            token_batches,
            metric=metric,
            device=device,
            max_batches=max_batches,
        )
    return scores
