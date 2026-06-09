from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .blocks import TransformerBlock, make_parent_block


@dataclass
class TinyConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 128
    max_seq_len: int = 128


@dataclass
class TinyOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] = ()


class TinyCausalLM(nn.Module):
    """Small causal Transformer used to exercise the distillation NAS pipeline."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                make_parent_block(config.hidden_size, config.num_heads, config.intermediate_size)
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False) -> TinyOutput:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = [hidden] if output_hidden_states else []
        for block in self.blocks:
            hidden = block(hidden)
            if output_hidden_states:
                hidden_states.append(hidden)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        return TinyOutput(logits=logits, hidden_states=tuple(hidden_states))

    def clone(self) -> "TinyCausalLM":
        return copy.deepcopy(self)


def random_token_batches(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    seed: int = 0,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
        for _ in range(num_batches)
    ]


def collect_layer_inputs(
    model: TinyCausalLM,
    token_batches: Iterable[torch.Tensor],
    layer_idx: int,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    device = torch.device(device)
    model = model.to(device).eval()
    collected: list[torch.Tensor] = []
    with torch.no_grad():
        for input_ids in token_batches:
            input_ids = input_ids.to(device)
            batch, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
            hidden = model.token_embedding(input_ids) + model.position_embedding(positions)
            for block_idx in range(layer_idx):
                hidden = model.blocks[block_idx](hidden)
            collected.append(hidden.detach().cpu())
    return collected


def collect_ffn_norm_inputs(
    block: TransformerBlock,
    hidden_batches: Iterable[torch.Tensor],
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    device = torch.device(device)
    block = block.to(device).eval()
    normalized: list[torch.Tensor] = []
    with torch.no_grad():
        for hidden in hidden_batches:
            hidden = hidden.to(device)
            after_attention = hidden + block.attention(block.ln_1(hidden))
            normalized.append(block.ln_2(after_attention).detach().cpu())
    return normalized

