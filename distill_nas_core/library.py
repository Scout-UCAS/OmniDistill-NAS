from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .blocks import TransformerBlock, copy_layer_norms
from .distill import local_distill_block
from .search_space import AttentionSpec, BlockSpec, FFNSpec, make_block_variant


@dataclass
class TrainedBlock:
    layer_idx: int
    spec: BlockSpec
    block: TransformerBlock
    losses: list[float]

    @property
    def name(self) -> str:
        return f"L{self.layer_idx}:{self.spec.name}"


def train_coupled_block_library(
    layer_idx: int,
    parent_block: TransformerBlock,
    block_specs: Iterable[BlockSpec],
    hidden_batches: Iterable[torch.Tensor],
    ffn_channel_order: torch.Tensor | None = None,
    steps: int = 100,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> list[TrainedBlock]:
    """Train every attention/FFN pair directly, the coupled BLD baseline."""

    trained: list[TrainedBlock] = []
    for spec in block_specs:
        block = make_block_variant(parent_block, spec, ffn_channel_order)
        losses: list[float] = []
        if steps > 0 and spec.name != "parent_attn+parent_ffn":
            losses = local_distill_block(parent_block, block, hidden_batches, steps=steps, lr=lr, device=device)
        trained.append(TrainedBlock(layer_idx, spec, block, losses))
    return trained


def train_decoupled_block_library(
    layer_idx: int,
    parent_block: TransformerBlock,
    attention_specs: Iterable[AttentionSpec],
    ffn_specs: Iterable[FFNSpec],
    hidden_batches: Iterable[torch.Tensor],
    ffn_channel_order: torch.Tensor | None = None,
    steps: int = 100,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> list[TrainedBlock]:
    """Train |A| + |F| subblocks, then compose |A| * |F| full blocks."""

    attention_specs = list(attention_specs)
    ffn_specs = list(ffn_specs)
    parent_attention_spec = _find_parent_attention(attention_specs)
    parent_ffn_spec = _find_parent_ffn(ffn_specs)

    attention_modules: dict[str, nn.Module] = {}
    ffn_modules: dict[str, nn.Module] = {}
    attention_losses: dict[str, list[float]] = {}
    ffn_losses: dict[str, list[float]] = {}

    for attention_spec in attention_specs:
        spec = BlockSpec(attention_spec, parent_ffn_spec)
        block = make_block_variant(parent_block, spec, ffn_channel_order)
        losses: list[float] = []
        if steps > 0 and attention_spec.kind != "parent":
            losses = local_distill_block(
                parent_block,
                block,
                hidden_batches,
                steps=steps,
                lr=lr,
                device=device,
                trainable_modules=[block.attention],
            )
        attention_modules[attention_spec.name] = copy.deepcopy(block.attention).cpu()
        attention_losses[attention_spec.name] = losses

    for ffn_spec in ffn_specs:
        spec = BlockSpec(parent_attention_spec, ffn_spec)
        block = make_block_variant(parent_block, spec, ffn_channel_order)
        losses = []
        if steps > 0 and ffn_spec.kind != "parent":
            losses = local_distill_block(
                parent_block,
                block,
                hidden_batches,
                steps=steps,
                lr=lr,
                device=device,
                trainable_modules=[block.ffn],
            )
        ffn_modules[ffn_spec.name] = copy.deepcopy(block.ffn).cpu()
        ffn_losses[ffn_spec.name] = losses

    trained: list[TrainedBlock] = []
    for attention_spec in attention_specs:
        for ffn_spec in ffn_specs:
            spec = BlockSpec(attention_spec, ffn_spec)
            block = TransformerBlock(
                parent_block.hidden_size,
                copy.deepcopy(attention_modules[attention_spec.name]),
                copy.deepcopy(ffn_modules[ffn_spec.name]),
                norm_eps=parent_block.ln_1.eps,
            )
            copy_layer_norms(parent_block, block)
            losses = attention_losses.get(attention_spec.name, []) + ffn_losses.get(ffn_spec.name, [])
            trained.append(TrainedBlock(layer_idx, spec, block, losses))
    return trained


def _find_parent_attention(specs: list[AttentionSpec]) -> AttentionSpec:
    for spec in specs:
        if spec.kind == "parent":
            return spec
    raise ValueError("attention_specs must include a parent spec")


def _find_parent_ffn(specs: list[FFNSpec]) -> FFNSpec:
    for spec in specs:
        if spec.kind == "parent":
            return spec
    raise ValueError("ffn_specs must include a parent spec")
