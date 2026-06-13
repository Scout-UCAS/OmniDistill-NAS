from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Callable
from typing import Iterable, cast

import torch
from torch import nn

from .blocks import (
    CausalSelfAttention,
    NoOpSubblock,
    SwiGLUFFN,
    TransformerBlock,
    copy_layer_norms,
    gqa_from_attention,
    infer_block_config,
    keep_count,
    kernel_linear_from_attention,
    local_sparse_from_attention,
    linear_from_attention,
    linear_from_ffn,
    mfa_from_attention,
    mha_from_attention,
    mka_from_attention,
    mla_from_attention,
    mqa_from_attention,
    pruned_ffn_from_parent,
    quantized_mha_from_attention,
    retention_from_attention,
)


@dataclass(frozen=True)
class AttentionSpec:
    name: str
    kind: str
    num_kv_heads: int | None = None
    latent_dim: int | None = None


@dataclass(frozen=True)
class FFNSpec:
    name: str
    kind: str
    ratio: float | None = None


@dataclass(frozen=True)
class BlockSpec:
    attention: AttentionSpec
    ffn: FFNSpec
    alias: str | None = None

    @property
    def name(self) -> str:
        if self.alias is not None:
            return self.alias
        return f"{self.attention.name}+{self.ffn.name}"


AttentionSpecFactory = Callable[[str, int], AttentionSpec]
AttentionModuleFactory = Callable[[CausalSelfAttention, AttentionSpec], nn.Module]
_CUSTOM_ATTENTION_SPEC_FACTORIES: dict[str, AttentionSpecFactory] = {}
_CUSTOM_ATTENTION_MODULE_FACTORIES: dict[str, AttentionModuleFactory] = {}
_CUSTOM_ATTENTION_ALIASES: dict[str, tuple[str, ...]] = {}


def register_attention_variant(
    name: str,
    spec_factory: AttentionSpecFactory,
    module_factory: AttentionModuleFactory,
    aliases: Iterable[str] = (),
) -> None:
    """Register an attention candidate without editing the built-in search space."""

    if not name:
        raise ValueError("attention variant name must be non-empty")
    _CUSTOM_ATTENTION_SPEC_FACTORIES[name] = spec_factory
    _CUSTOM_ATTENTION_MODULE_FACTORIES[name] = module_factory
    for alias in aliases:
        if not alias:
            raise ValueError("attention variant aliases must be non-empty")
        _CUSTOM_ATTENTION_ALIASES[alias] = (name,)


def registered_attention_variants() -> tuple[str, ...]:
    return tuple(_CUSTOM_ATTENTION_SPEC_FACTORIES)


QWEN_ATTENTION_VARIANT_NAMES = (
    "parent_attn",
    "mha_attn",
    "quant_mha_attn",
    "mqa_attn",
    "gqa_kv2",
    "mfa_kv2",
    "mla_kv2",
    "mka_attn",
    "linear_attn",
    "noop_attn",
)
FLA_LINEAR_ATTENTION_VARIANT_NAMES = (
    "fla_linear_attn",
    "fla_gated_linear_attn",
    "fla_based_linear_attn",
    "fla_rebased_linear_attn",
    "fla_deltanet_attn",
    "fla_gated_deltanet_attn",
    "fla_kimi_delta_attn",
)
FLA_STRUCTURED_ATTENTION_VARIANT_NAMES = (
    "fla_multiscale_retention_attn",
    "fla_mla_attn",
    "fla_native_sparse_attn",
    "fla_moba_attn",
)
FLA_ATTENTION_VARIANT_NAMES = FLA_LINEAR_ATTENTION_VARIANT_NAMES + FLA_STRUCTURED_ATTENTION_VARIANT_NAMES
LINEAR_ATTENTION_VARIANT_NAMES = ("linear_attn",) + FLA_LINEAR_ATTENTION_VARIANT_NAMES
CORE_ATTENTION_VARIANT_NAMES = (
    "parent_attn",
    "mha_attn",
    "quant_mha_attn",
    "mqa_attn",
    "gqa_kv2",
    "mfa_kv2",
    "mla_kv2",
    "mka_attn",
) + LINEAR_ATTENTION_VARIANT_NAMES + ("noop_attn",)
ALL_ATTENTION_VARIANT_NAMES = CORE_ATTENTION_VARIANT_NAMES + FLA_STRUCTURED_ATTENTION_VARIANT_NAMES
LAYER_VARIANT_NAMES = (
    "parent",
    "skip_attn",
    "skip_mlp",
    "skip_both",
)
ATTENTION_VARIANT_ALIASES = {
    "parent": ("parent_attn",),
    "all_qwen_attn": QWEN_ATTENTION_VARIANT_NAMES,
    "all_linear_attn": LINEAR_ATTENTION_VARIANT_NAMES,
    "all_core_attn": CORE_ATTENTION_VARIANT_NAMES,
    "all_fla": FLA_ATTENTION_VARIANT_NAMES,
    "all_attention": ALL_ATTENTION_VARIANT_NAMES,
    "fla_gla_attn": ("fla_gated_linear_attn",),
    "fla_based_attn": ("fla_based_linear_attn",),
    "fla_rebased_attn": ("fla_rebased_linear_attn",),
    "fla_delta_net_attn": ("fla_deltanet_attn",),
    "fla_gated_delta_net_attn": ("fla_gated_deltanet_attn",),
    "fla_kda_attn": ("fla_kimi_delta_attn",),
    "fla_retention_attn": ("fla_multiscale_retention_attn",),
    "fla_multihead_latent_attn": ("fla_mla_attn",),
    "fla_nsa_attn": ("fla_native_sparse_attn",),
}


def default_attention_specs(
    parent_num_kv_heads: int,
    parent_num_heads: int | None = None,
    include_fla: bool = False,
) -> list[AttentionSpec]:
    parent_num_heads = parent_num_kv_heads if parent_num_heads is None else parent_num_heads
    candidates = [
        AttentionSpec("parent_attn", "parent", parent_num_kv_heads),
        AttentionSpec("mha_attn", "mha", parent_num_heads),
        AttentionSpec("quant_mha_attn", "quant_mha", parent_num_heads),
        AttentionSpec("mqa_attn", "mqa", 1),
    ]
    for heads in [8, 4, 2, 1]:
        if 1 < heads < parent_num_heads and parent_num_heads % heads == 0:
            candidates.append(AttentionSpec(f"gqa_kv{heads}", "gqa", heads))
    candidates.extend(
        [
            AttentionSpec("mfa_attn", "mfa", max(1, parent_num_heads // 2)),
            AttentionSpec("mla_attn", "mla", max(1, parent_num_heads // 2)),
            AttentionSpec("mka_attn", "mka", 1),
            AttentionSpec("linear_attn", "linear", None),
            AttentionSpec("noop_attn", "noop", None),
        ]
    )
    if include_fla:
        candidates.extend(attention_specs_from_names("all_fla", parent_num_heads))
    return candidates


def expand_attention_variant_names(raw_variants: str | Iterable[str]) -> list[str]:
    if isinstance(raw_variants, str):
        names = [name.strip() for name in raw_variants.split(",") if name.strip()]
    else:
        names = [name.strip() for name in raw_variants if name.strip()]
    variants: list[str] = []
    for name in names:
        expanded = _CUSTOM_ATTENTION_ALIASES.get(name, ATTENTION_VARIANT_ALIASES.get(name, (name,)))
        for variant in expanded:
            if variant in {"skip_attn", "skip_mlp", "skip_both"}:
                raise ValueError(f"{variant} is a layer variant; use --layer-variants instead")
            if variant not in variants:
                variants.append(variant)
    return variants


def attention_spec_from_name(name: str, parent_num_heads: int) -> AttentionSpec:
    if name in _CUSTOM_ATTENTION_SPEC_FACTORIES:
        return _CUSTOM_ATTENTION_SPEC_FACTORIES[name](name, parent_num_heads)
    if name == "parent_attn":
        return AttentionSpec(name, "parent", parent_num_heads)
    if name == "mha_attn":
        return AttentionSpec(name, "mha", parent_num_heads)
    if name == "quant_mha_attn":
        return AttentionSpec(name, "quant_mha", parent_num_heads)
    if name == "mqa_attn":
        return AttentionSpec(name, "mqa", 1)
    if name.startswith("gqa_kv"):
        return AttentionSpec(name, "gqa", parse_kv_suffix(name))
    if name == "mfa_attn":
        return AttentionSpec(name, "mfa", max(1, parent_num_heads // 2))
    if name.startswith("mfa_kv"):
        return AttentionSpec(name, "mfa", parse_kv_suffix(name))
    if name == "mla_attn":
        return AttentionSpec(name, "mla", max(1, parent_num_heads // 2))
    if name.startswith("mla_kv"):
        return AttentionSpec(name, "mla", parse_kv_suffix(name))
    if name == "mka_attn":
        return AttentionSpec(name, "mka", 1)
    if name == "linear_attn":
        return AttentionSpec(name, "linear", None)
    if name == "noop_attn":
        return AttentionSpec(name, "noop", None)
    if name in FLA_ATTENTION_VARIANT_NAMES:
        return AttentionSpec(name, name.removesuffix("_attn"), None)
    raise ValueError(f"unknown attention variant: {name}")


def attention_specs_from_names(raw_variants: str | Iterable[str], parent_num_heads: int) -> list[AttentionSpec]:
    return [attention_spec_from_name(name, parent_num_heads) for name in expand_attention_variant_names(raw_variants)]


def parse_kv_suffix(name: str) -> int:
    try:
        value = int(name.rsplit("kv", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"expected variant ending with kv<count>, got {name}") from exc
    if value < 1:
        raise ValueError(f"KV head count must be positive, got {name}")
    return value


def layer_variant_specs_from_names(raw_variants: str | Iterable[str], parent_num_heads: int) -> list[BlockSpec]:
    if isinstance(raw_variants, str):
        names = [name.strip() for name in raw_variants.split(",") if name.strip()]
    else:
        names = [name.strip() for name in raw_variants if name.strip()]
    specs: list[BlockSpec] = []
    for name in names:
        if name == "parent":
            specs.append(
                BlockSpec(
                    attention_spec_from_name("parent_attn", parent_num_heads),
                    FFNSpec("parent_ffn", "parent", 1.0),
                    alias="parent",
                )
            )
        elif name == "skip_attn":
            specs.append(
                BlockSpec(
                    attention_spec_from_name("noop_attn", parent_num_heads),
                    FFNSpec("parent_ffn", "parent", 1.0),
                    alias="skip_attn",
                )
            )
        elif name == "skip_mlp":
            specs.append(
                BlockSpec(
                    attention_spec_from_name("parent_attn", parent_num_heads),
                    FFNSpec("noop_ffn", "noop", None),
                    alias="skip_mlp",
                )
            )
        elif name == "skip_both":
            specs.append(
                BlockSpec(
                    attention_spec_from_name("noop_attn", parent_num_heads),
                    FFNSpec("noop_ffn", "noop", None),
                    alias="skip_both",
                )
            )
        else:
            raise ValueError(f"unknown layer variant: {name}")
    return specs


def default_ffn_specs() -> list[FFNSpec]:
    return [
        FFNSpec("parent_ffn", "parent", 1.0),
        FFNSpec("ffn_87", "pruned", 0.875),
        FFNSpec("ffn_75", "pruned", 0.75),
        FFNSpec("ffn_50", "pruned", 0.50),
        FFNSpec("ffn_25", "pruned", 0.25),
        FFNSpec("ffn_20", "pruned", 0.20),
        FFNSpec("ffn_10", "pruned", 0.10),
        FFNSpec("linear_ffn", "linear", None),
        FFNSpec("noop_ffn", "noop", None),
    ]


def iter_block_specs(
    attention_specs: Iterable[AttentionSpec],
    ffn_specs: Iterable[FFNSpec],
) -> list[BlockSpec]:
    return [BlockSpec(attn, ffn) for attn in attention_specs for ffn in ffn_specs]


def ffn_channel_contribution_order(
    ffn: SwiGLUFFN,
    normalized_inputs: Iterable[torch.Tensor],
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rank FFN channels by average |activation| * ||W_down[channel]||_2."""

    device = torch.device(device)
    ffn = ffn.to(device)
    totals = torch.zeros(ffn.intermediate_size, device=device)
    seen = 0
    was_training = ffn.training
    ffn.eval()
    with torch.no_grad():
        down_norm = ffn.down_proj.weight.norm(dim=0)
        for inputs in normalized_inputs:
            inputs = inputs.to(device)
            intermediate = ffn.intermediate(inputs)
            contribution = intermediate.abs() * down_norm
            totals += contribution.reshape(-1, ffn.intermediate_size).sum(dim=0)
            seen += contribution.numel() // ffn.intermediate_size
    if was_training:
        ffn.train()
    if seen == 0:
        raise ValueError("normalized_inputs produced no tokens")
    scores = totals / seen
    return torch.argsort(scores, descending=True).cpu()


def make_attention_variant(parent_attention: CausalSelfAttention, spec: AttentionSpec) -> nn.Module:
    if spec.name in _CUSTOM_ATTENTION_MODULE_FACTORIES:
        return _CUSTOM_ATTENTION_MODULE_FACTORIES[spec.name](parent_attention, spec)
    if spec.kind == "parent":
        return copy.deepcopy(parent_attention)
    if spec.kind == "mha":
        return mha_from_attention(parent_attention)
    if spec.kind == "quant_mha":
        return quantized_mha_from_attention(parent_attention, num_bits=8)
    if spec.kind == "mqa":
        return mqa_from_attention(parent_attention)
    if spec.kind == "gqa":
        if spec.num_kv_heads is None:
            raise ValueError("gqa attention spec requires num_kv_heads")
        return gqa_from_attention(parent_attention, spec.num_kv_heads)
    if spec.kind == "mfa":
        return mfa_from_attention(parent_attention, spec.num_kv_heads, spec.latent_dim)
    if spec.kind == "mla":
        return mla_from_attention(parent_attention, spec.num_kv_heads, spec.latent_dim)
    if spec.kind == "mka":
        return mka_from_attention(parent_attention, spec.latent_dim)
    if spec.kind == "linear":
        return linear_from_attention(parent_attention)
    if spec.kind == "noop":
        return NoOpSubblock()
    if spec.kind == "fla_linear":
        return kernel_linear_from_attention(parent_attention, feature_map="elu", output_gate=False)
    if spec.kind == "fla_gated_linear":
        return kernel_linear_from_attention(parent_attention, feature_map="elu", output_gate=True)
    if spec.kind == "fla_based_linear":
        return kernel_linear_from_attention(parent_attention, feature_map="relu", output_gate=False)
    if spec.kind == "fla_rebased_linear":
        return kernel_linear_from_attention(parent_attention, feature_map="relu_squared", output_gate=False)
    if spec.kind == "fla_deltanet":
        return kernel_linear_from_attention(parent_attention, feature_map="elu", output_gate=False)
    if spec.kind == "fla_gated_deltanet":
        return kernel_linear_from_attention(parent_attention, feature_map="elu", output_gate=True)
    if spec.kind == "fla_kimi_delta":
        return kernel_linear_from_attention(parent_attention, feature_map="silu_elu", output_gate=True)
    if spec.kind == "fla_multiscale_retention":
        return retention_from_attention(parent_attention)
    if spec.kind == "fla_mla":
        return mla_from_attention(parent_attention, max(1, parent_attention.num_heads // 2), parent_attention.hidden_size // 4)
    if spec.kind == "fla_native_sparse":
        return local_sparse_from_attention(parent_attention, window_size=max(1, parent_attention.head_dim))
    if spec.kind == "fla_moba":
        return local_sparse_from_attention(parent_attention, window_size=max(1, parent_attention.hidden_size // 2))
    raise ValueError(f"unknown attention kind: {spec.kind}")


def make_ffn_variant(
    parent_ffn: SwiGLUFFN,
    spec: FFNSpec,
    channel_order: torch.Tensor | None = None,
) -> nn.Module:
    if spec.kind == "parent":
        return copy.deepcopy(parent_ffn)
    if spec.kind == "pruned":
        if spec.ratio is None:
            raise ValueError("pruned FFN spec requires ratio")
        if channel_order is None:
            channel_order = torch.arange(parent_ffn.intermediate_size)
        count = keep_count(parent_ffn.intermediate_size, spec.ratio)
        return pruned_ffn_from_parent(parent_ffn, channel_order[:count])
    if spec.kind == "linear":
        return linear_from_ffn(parent_ffn)
    if spec.kind == "noop":
        return NoOpSubblock()
    raise ValueError(f"unknown FFN kind: {spec.kind}")


def make_block_variant(
    parent_block: TransformerBlock,
    spec: BlockSpec,
    ffn_channel_order: torch.Tensor | None = None,
) -> TransformerBlock:
    config = infer_block_config(parent_block)
    parent_attention = cast(CausalSelfAttention, parent_block.attention)
    parent_ffn = cast(SwiGLUFFN, parent_block.ffn)
    attention = make_attention_variant(parent_attention, spec.attention)
    ffn = make_ffn_variant(parent_ffn, spec.ffn, ffn_channel_order)
    target = TransformerBlock(config.hidden_size, attention, ffn, norm_eps=parent_block.ln_1.eps)
    copy_layer_norms(parent_block, target)
    return target
