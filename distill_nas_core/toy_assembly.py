from __future__ import annotations

from typing import Any, cast

import torch

from .blocks import TransformerBlock
from .search_space import AttentionSpec, BlockSpec, FFNSpec, make_block_variant
from .toy import TinyCausalLM, TinyConfig


def config_from_dict(raw: dict[str, Any]) -> TinyConfig:
    return TinyConfig(**raw)


def spec_from_dict(raw: dict[str, Any]) -> BlockSpec:
    return BlockSpec(
        attention=AttentionSpec(**raw["attention"]),
        ffn=FFNSpec(**raw["ffn"]),
        alias=raw.get("alias"),
    )


def build_parent_model(artifact: dict[str, Any]) -> TinyCausalLM:
    model = TinyCausalLM(config_from_dict(artifact["config"]))
    model.load_state_dict(artifact["parent_state_dict"])
    return model


def block_key(layer_idx: int, spec: BlockSpec) -> tuple[int, str, str]:
    return (layer_idx, spec.attention.name, spec.ffn.name)


def block_library_map(artifact: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    records = {}
    for item in artifact["blocks"]:
        spec = spec_from_dict(item["spec"])
        records[block_key(int(item["layer_idx"]), spec)] = item
    return records


def reconstruct_block(
    parent: TinyCausalLM,
    layer_idx: int,
    spec: BlockSpec,
    state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    parent_block = cast(TransformerBlock, parent.blocks[layer_idx])
    block = make_block_variant(parent_block, spec, ffn_channel_order=None)
    block.load_state_dict(state_dict)
    return block


def assemble_student_from_bld(
    artifact: dict[str, Any],
    selected_config: dict[str, Any],
) -> tuple[TinyCausalLM, TinyCausalLM]:
    parent = build_parent_model(artifact)
    student = build_parent_model(artifact)
    library = block_library_map(artifact)
    for selected in selected_config["selected"]:
        layer_idx = int(selected["layer_idx"])
        selected_spec = spec_from_dict(selected["spec"])
        base_spec = BlockSpec(selected_spec.attention, selected_spec.ffn)
        record = library[block_key(layer_idx, base_spec)]
        block = reconstruct_block(parent, layer_idx, base_spec, record["state_dict"])
        student.blocks[layer_idx] = block
    return parent, student
