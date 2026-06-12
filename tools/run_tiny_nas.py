from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from distill_nas_core.library import train_decoupled_block_library
from distill_nas_core.mip import SearchCandidate, SearchConstraints, solve_nas_mip
from distill_nas_core.resources import candidate_runtime_map, kv_cache_memory_bytes, parameter_memory_bytes
from distill_nas_core.scoring import score_replace_one_block
from distill_nas_core.search_space import (
    FFNSpec,
    attention_specs_from_names,
    ffn_channel_contribution_order,
    layer_variant_specs_from_names,
)
from distill_nas_core.toy import (
    TinyCausalLM,
    TinyConfig,
    collect_ffn_norm_inputs,
    collect_layer_inputs,
    random_token_batches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Distillation NAS pipeline.")
    parser.add_argument("--quick", action="store_true", help="Use the fastest settings.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, gpu, cuda, mps, or cpu.")
    parser.add_argument("--bld-steps", type=int, default=None, help="Local distillation steps per block.")
    parser.add_argument("--score-batches", type=int, default=2)
    parser.add_argument(
        "--attention-variants",
        default="all_attention",
        help="Comma-separated attention variants or aliases: all_qwen_attn, all_linear_attn, all_core_attn, all_fla, all_attention.",
    )
    parser.add_argument(
        "--layer-variants",
        default="parent,skip_attn,skip_mlp,skip_both",
        help="Comma-separated layer-level variants: parent, skip_attn, skip_mlp, skip_both.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if normalized == "gpu":
            raise RuntimeError("requested GPU device, but neither CUDA nor MPS is available")
        return torch.device("cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(7)

    config = TinyConfig(
        vocab_size=96,
        hidden_size=32 if args.quick else 64,
        num_layers=2 if args.quick else 4,
        num_heads=4,
        intermediate_size=64 if args.quick else 128,
        max_seq_len=64,
    )
    seq_len = 16 if args.quick else 32
    batch_size = 2
    batch_sizes = [1, 2, 4]
    bld_steps = args.bld_steps if args.bld_steps is not None else (1 if args.quick else 4)

    parent = TinyCausalLM(config)
    token_batches = random_token_batches(config.vocab_size, batch_size, seq_len, num_batches=4, seed=13)
    score_batches = token_batches[: args.score_batches]

    attention_specs = attention_specs_from_names(args.attention_variants, config.num_heads)
    layer_variant_specs = layer_variant_specs_from_names(args.layer_variants, config.num_heads)
    attention_spec_names = {spec.name for spec in attention_specs}
    for spec in layer_variant_specs:
        if spec.attention.name not in attention_spec_names:
            attention_specs.append(spec.attention)
            attention_spec_names.add(spec.attention.name)
    ffn_specs = [
        FFNSpec("parent_ffn", "parent", 1.0),
        FFNSpec("ffn_50", "pruned", 0.5),
        FFNSpec("linear_ffn", "linear", None),
        FFNSpec("noop_ffn", "noop", None),
    ]
    candidates_by_layer: list[list[SearchCandidate]] = []
    for layer_idx, parent_block in enumerate(parent.blocks):
        hidden_batches = collect_layer_inputs(parent, token_batches, layer_idx, device=device)
        ffn_inputs = collect_ffn_norm_inputs(parent_block, hidden_batches, device=device)
        channel_order = ffn_channel_contribution_order(parent_block.ffn, ffn_inputs, device=device)
        layer_candidates: list[SearchCandidate] = []

        trained_blocks = train_decoupled_block_library(
            layer_idx,
            parent_block,
            attention_specs,
            ffn_specs,
            hidden_batches,
            ffn_channel_order=channel_order,
            steps=bld_steps,
            lr=3e-3,
            device=device,
        )

        candidate_by_combo: dict[tuple[str, str], SearchCandidate] = {}
        for trained in trained_blocks:
            spec = trained.spec
            block = trained.block
            name = f"L{layer_idx}:{spec.name}"
            score = score_replace_one_block(
                parent,
                layer_idx,
                block,
                score_batches,
                metric="kl",
                device=device,
                max_batches=args.score_batches,
            )
            runtimes = candidate_runtime_map(
                block,
                config.hidden_size,
                seq_len,
                batch_sizes,
                measured=False,
                device=device,
            )
            candidate = SearchCandidate(
                layer_idx=layer_idx,
                name=name,
                score=score,
                param_memory=parameter_memory_bytes(block, dtype_bytes=2),
                kv_cache_memory=kv_cache_memory_bytes(block, seq_len, dtype_bytes=2),
                runtimes=runtimes,
                payload=spec,
            )
            layer_candidates.append(candidate)
            candidate_by_combo[(spec.attention.name, spec.ffn.name)] = candidate

        for spec in layer_variant_specs:
            base = candidate_by_combo[(spec.attention.name, spec.ffn.name)]
            layer_candidates.append(
                SearchCandidate(
                    layer_idx=layer_idx,
                    name=f"L{layer_idx}:{spec.name}",
                    score=base.score,
                    param_memory=base.param_memory,
                    kv_cache_memory=base.kv_cache_memory,
                    runtimes=dict(base.runtimes),
                    payload=spec,
                )
            )

        candidates_by_layer.append(layer_candidates)

    parent_memory = sum(
        candidate.param_memory
        for layer in candidates_by_layer
        for candidate in layer
        if candidate.name.endswith("parent_attn+parent_ffn")
    )
    parent_runtime_b2 = sum(
        candidate.runtimes[2]
        for layer in candidates_by_layer
        for candidate in layer
        if candidate.name.endswith("parent_attn+parent_ffn")
    )

    constraints = SearchConstraints(
        seq_len=seq_len,
        batch_sizes=batch_sizes,
        memory_max=parent_memory * 0.82,
        latency_max=parent_runtime_b2 * 0.82,
        score_direction="minimize",
    )
    solution = solve_nas_mip(candidates_by_layer, constraints)

    print(f"generated_candidates={sum(len(layer) for layer in candidates_by_layer)}")
    print(f"device={device}")
    print(f"selected_batch_size={solution.batch_size}")
    print(f"total_kl_score={solution.total_score:.6f}")
    print(f"total_memory_bytes={solution.total_memory:.0f}")
    print(f"total_runtime_proxy={solution.total_runtime:.8f}")
    print(f"throughput_proxy={solution.throughput:.2f}")
    print("architecture:")
    for candidate in solution.selected:
        print(f"  {candidate.name} score={candidate.score:.6f}")


if __name__ == "__main__":
    main()
