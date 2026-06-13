from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.distill import global_knowledge_distillation
from distill_nas_core.library import train_decoupled_block_library
from distill_nas_core.mip import SearchCandidate, SearchConstraints, solve_nas_mip
from distill_nas_core.resources import candidate_runtime_map, kv_cache_memory_bytes, parameter_memory_bytes
from distill_nas_core.scoring import score_replace_one_block
from distill_nas_core.search_space import (
    AttentionSpec,
    BlockSpec,
    FFNSpec,
    attention_specs_from_names,
    ffn_channel_contribution_order,
    layer_variant_specs_from_names,
)
from distill_nas_core.toy_assembly import (
    assemble_student_from_bld,
    block_key,
    build_parent_model,
    config_from_dict,
    reconstruct_block,
    spec_from_dict,
)
from distill_nas_core.toy import (
    TinyCausalLM,
    TinyConfig,
    collect_ffn_norm_inputs,
    collect_layer_inputs,
    random_token_batches,
)


DEFAULT_WORKFLOW_OUTPUT_DIR = Path("outputs/distill_nas_workflow")
DEFAULT_BLD_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "04_bld_block_library"
DEFAULT_NAS_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "05_nas_layer_scoring"
DEFAULT_MIP_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "06_mip_topk_architecture_configs"
DEFAULT_ASSEMBLY_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "07_model_assembly"
DEFAULT_GKD_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "08_global_knowledge_distillation"
DEFAULT_ATTENTION_VARIANTS = "parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,linear_attn,noop_attn"
DEFAULT_LAYER_VARIANTS = "parent,skip_attn,skip_mlp,skip_both"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def save_pth(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved)
    return resolved


def load_pth(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    try:
        return torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(resolved, map_location="cpu")


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


def make_tiny_config(quick: bool) -> TinyConfig:
    return TinyConfig(
        vocab_size=96,
        hidden_size=32 if quick else 64,
        num_layers=2 if quick else 4,
        num_heads=4,
        intermediate_size=64 if quick else 128,
        max_seq_len=64 if quick else 128,
    )


def spec_to_dict(spec: BlockSpec) -> dict[str, Any]:
    return {
        "attention": asdict(spec.attention),
        "ffn": asdict(spec.ffn),
        "alias": spec.alias,
        "name": spec.name,
    }


def ffn_specs() -> list[FFNSpec]:
    return [
        FFNSpec("parent_ffn", "parent", 1.0),
        FFNSpec("ffn_50", "pruned", 0.5),
        FFNSpec("linear_ffn", "linear", None),
        FFNSpec("noop_ffn", "noop", None),
    ]


def prepare_attention_and_layer_specs(
    attention_variants: str,
    layer_variants: str,
    num_heads: int,
) -> tuple[list[AttentionSpec], list[BlockSpec]]:
    attention_specs = attention_specs_from_names(attention_variants, num_heads)
    layer_specs = layer_variant_specs_from_names(layer_variants, num_heads)
    attention_names = {spec.name for spec in attention_specs}
    for layer_spec in layer_specs:
        if layer_spec.attention.name not in attention_names:
            attention_specs.append(layer_spec.attention)
            attention_names.add(layer_spec.attention.name)
    return attention_specs, layer_specs


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def make_token_batches(artifact: dict[str, Any], count: int | None = None) -> list[torch.Tensor]:
    config = config_from_dict(artifact["config"])
    num_batches = count if count is not None else int(artifact.get("num_token_batches", 4))
    return random_token_batches(
        config.vocab_size,
        int(artifact.get("batch_size", 2)),
        int(artifact["seq_len"]),
        num_batches=num_batches,
        seed=int(artifact.get("data_seed", 13)),
    )


def command_bld(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    torch.manual_seed(args.model_seed)
    config = make_tiny_config(args.quick)
    parent = TinyCausalLM(config)
    token_batches = random_token_batches(
        config.vocab_size,
        args.batch_size,
        args.seq_len,
        num_batches=args.num_batches,
        seed=args.data_seed,
    )
    attention_specs, layer_specs = prepare_attention_and_layer_specs(
        args.attention_variants,
        args.layer_variants,
        config.num_heads,
    )

    records: list[dict[str, Any]] = []
    for layer_idx, parent_block in enumerate(parent.blocks):
        hidden_batches = collect_layer_inputs(parent, token_batches, layer_idx, device=device)
        ffn_inputs = collect_ffn_norm_inputs(parent_block, hidden_batches, device=device)
        channel_order = ffn_channel_contribution_order(parent_block.ffn, ffn_inputs, device=device)
        trained_blocks = train_decoupled_block_library(
            layer_idx,
            parent_block,
            attention_specs,
            ffn_specs(),
            hidden_batches,
            ffn_channel_order=channel_order,
            steps=args.bld_steps,
            lr=args.lr,
            device=device,
        )
        for trained in trained_blocks:
            records.append(
                {
                    "layer_idx": trained.layer_idx,
                    "name": trained.name,
                    "spec": spec_to_dict(trained.spec),
                    "losses": trained.losses,
                    "state_dict": cpu_state_dict(trained.block),
                }
            )

    artifact = {
        "stage": "bld",
        "format_version": 1,
        "config": asdict(config),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "num_token_batches": args.num_batches,
        "model_seed": args.model_seed,
        "data_seed": args.data_seed,
        "attention_variants": args.attention_variants,
        "layer_variants": args.layer_variants,
        "bld_steps": args.bld_steps,
        "device": str(device),
        "parent_state_dict": cpu_state_dict(parent),
        "layer_variant_specs": [spec_to_dict(spec) for spec in layer_specs],
        "blocks": records,
    }
    pth_path = save_pth(args.output_pth, artifact)
    summary = {
        "stage": "bld",
        "pth": str(pth_path),
        "num_layers": config.num_layers,
        "num_blocks": len(records),
        "attention_variants": args.attention_variants,
        "layer_variants": args.layer_variants,
        "bld_steps": args.bld_steps,
    }
    summary_path = write_json(args.summary_json, summary)
    print(f"wrote_bld_pth={pth_path}")
    print(f"wrote_summary={summary_path}")


def score_records_from_bld(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = load_pth(args.bld_pth)
    device = resolve_device(args.device)
    parent = build_parent_model(artifact)
    token_batches = make_token_batches(artifact, count=args.score_batches)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    scores: list[dict[str, Any]] = []
    combo_scores: dict[tuple[int, str, str], dict[str, Any]] = {}

    for item in artifact["blocks"]:
        layer_idx = int(item["layer_idx"])
        spec = spec_from_dict(item["spec"])
        block = reconstruct_block(parent, layer_idx, spec, item["state_dict"])
        score = score_replace_one_block(
            parent,
            layer_idx,
            block,
            token_batches,
            metric="kl",
            device=device,
            max_batches=args.score_batches,
        )
        runtimes = candidate_runtime_map(
            block,
            artifact["config"]["hidden_size"],
            artifact["seq_len"],
            batch_sizes,
            measured=False,
            device=device,
        )
        record = {
            "layer_idx": layer_idx,
            "name": f"L{layer_idx}:{spec.name}",
            "variant": spec.name,
            "score": score,
            "param_memory": parameter_memory_bytes(block, dtype_bytes=2),
            "kv_cache_memory": kv_cache_memory_bytes(block, artifact["seq_len"], dtype_bytes=2),
            "runtimes": {str(key): value for key, value in runtimes.items()},
            "spec": spec_to_dict(spec),
            "source": "bld_block",
        }
        scores.append(record)
        combo_scores[block_key(layer_idx, spec)] = record

    for layer_spec_raw in artifact["layer_variant_specs"]:
        layer_spec = spec_from_dict(layer_spec_raw)
        base_spec = BlockSpec(layer_spec.attention, layer_spec.ffn)
        for layer_idx in range(artifact["config"]["num_layers"]):
            base = combo_scores.get(block_key(layer_idx, base_spec))
            if base is None:
                continue
            alias_record = dict(base)
            alias_record["name"] = f"L{layer_idx}:{layer_spec.name}"
            alias_record["variant"] = layer_spec.name
            alias_record["spec"] = spec_to_dict(layer_spec)
            alias_record["source"] = "layer_alias"
            scores.append(alias_record)

    return artifact, scores


def command_score(args: argparse.Namespace) -> None:
    artifact, scores = score_records_from_bld(args)
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for record in scores:
        by_layer.setdefault(int(record["layer_idx"]), []).append(record)

    layer_importance = []
    for layer_idx, layer_scores in sorted(by_layer.items()):
        best = min(layer_scores, key=lambda item: item["score"])
        worst = max(layer_scores, key=lambda item: item["score"])
        parent = next((item for item in layer_scores if item["variant"] == "parent"), None)
        layer_importance.append(
            {
                "layer_idx": layer_idx,
                "num_candidates": len(layer_scores),
                "best_candidate": best["name"],
                "best_score": best["score"],
                "worst_candidate": worst["name"],
                "worst_score": worst["score"],
                "score_range_importance": worst["score"] - best["score"],
                "parent_score": None if parent is None else parent["score"],
                "parent_to_best_delta": None if parent is None else parent["score"] - best["score"],
            }
        )

    output = {
        "stage": "nas_layer_importance",
        "bld_pth": str(resolve_path(args.bld_pth)),
        "config": artifact["config"],
        "seq_len": artifact["seq_len"],
        "batch_sizes": [int(item) for item in args.batch_sizes.split(",") if item.strip()],
        "score_batches": args.score_batches,
        "scores": scores,
        "layer_importance": layer_importance,
    }
    output_path = write_json(args.output_json, output)
    print(f"wrote_importance_json={output_path}")


def candidates_by_layer_from_scores(scores_payload: dict[str, Any]) -> list[list[SearchCandidate]]:
    layers: dict[int, list[SearchCandidate]] = {}
    for record in scores_payload["scores"]:
        runtimes = {int(key): float(value) for key, value in record["runtimes"].items()}
        candidate = SearchCandidate(
            layer_idx=int(record["layer_idx"]),
            name=record["name"],
            score=float(record["score"]),
            param_memory=float(record["param_memory"]),
            kv_cache_memory=float(record["kv_cache_memory"]),
            runtimes=runtimes,
            payload={"spec": record["spec"], "variant": record["variant"]},
        )
        layers.setdefault(candidate.layer_idx, []).append(candidate)
    return [layers[index] for index in sorted(layers)]


def solution_to_config(solution, rank: int) -> dict[str, Any]:
    selected = []
    for candidate in solution.selected:
        selected.append(
            {
                "layer_idx": candidate.layer_idx,
                "name": candidate.name,
                "variant": candidate.payload["variant"],
                "score": candidate.score,
                "param_memory": candidate.param_memory,
                "kv_cache_memory": candidate.kv_cache_memory,
                "runtimes": {str(key): value for key, value in candidate.runtimes.items()},
                "spec": candidate.payload["spec"],
            }
        )
    return {
        "rank": rank,
        "selected_batch_size": solution.batch_size,
        "total_score": solution.total_score,
        "total_memory": solution.total_memory,
        "total_runtime": solution.total_runtime,
        "throughput": solution.throughput,
        "objective": solution.objective,
        "objective_components": solution.objective_components,
        "selected": selected,
    }


def command_mip(args: argparse.Namespace) -> None:
    scores_path = resolve_path(args.scores_json)
    scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
    candidates_by_layer = candidates_by_layer_from_scores(scores_payload)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]

    parent_candidates = []
    for layer in candidates_by_layer:
        parent = next((candidate for candidate in layer if candidate.payload["variant"] == "parent"), None)
        if parent is None:
            parent = next(candidate for candidate in layer if candidate.name.endswith("parent_attn+parent_ffn"))
        parent_candidates.append(parent)
    parent_memory_by_batch = {
        batch_size: sum(candidate.param_memory + batch_size * candidate.kv_cache_memory for candidate in parent_candidates)
        for batch_size in batch_sizes
    }
    parent_runtime_by_batch = {
        batch_size: sum(candidate.runtimes[batch_size] for candidate in parent_candidates)
        for batch_size in batch_sizes
    }

    configs = []
    previous: list[list[str]] = []
    for rank in range(args.top_k):
        constraints = SearchConstraints(
            seq_len=int(scores_payload["seq_len"]),
            batch_sizes=batch_sizes,
            memory_max_by_batch={
                batch_size: value * args.memory_fraction
                for batch_size, value in parent_memory_by_batch.items()
            },
            latency_max_by_batch={
                batch_size: value * args.runtime_fraction
                for batch_size, value in parent_runtime_by_batch.items()
            },
            score_direction="minimize",
            objective_mode=args.objective_mode,
            score_weight=args.score_weight,
            memory_weight=args.memory_weight,
            runtime_weight=args.runtime_weight,
            normalize_objectives=not args.no_normalize_objectives,
            diversity_alpha=args.diversity_alpha if previous else None,
            previous_solutions=previous,
        )
        try:
            solution = solve_nas_mip(candidates_by_layer, constraints)
        except RuntimeError as exc:
            if rank == 0:
                raise
            print(f"stopped_topk_at_rank={rank} reason={exc}")
            break
        config = solution_to_config(solution, rank)
        configs.append(config)
        previous.append(solution.selected_names)

    output = {
        "stage": "mip_topk_configs",
        "scores_json": str(scores_path),
        "top_k_requested": args.top_k,
        "memory_fraction": args.memory_fraction,
        "runtime_fraction": args.runtime_fraction,
        "memory_max_by_batch": {
            str(batch_size): value * args.memory_fraction
            for batch_size, value in parent_memory_by_batch.items()
        },
        "latency_max_by_batch": {
            str(batch_size): value * args.runtime_fraction
            for batch_size, value in parent_runtime_by_batch.items()
        },
        "objective_mode": args.objective_mode,
        "objective_weights": {
            "score": args.score_weight,
            "memory": args.memory_weight,
            "runtime": args.runtime_weight,
        },
        "normalize_objectives": not args.no_normalize_objectives,
        "diversity_alpha": args.diversity_alpha,
        "configs": configs,
    }
    output_path = write_json(args.output_json, output)
    config_dir = resolve_path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    for stale in config_dir.glob("config_rank_*.json"):
        stale.unlink()
    for config in configs:
        write_json(config_dir / f"config_rank_{config['rank']:02d}.json", config)
    print(f"wrote_topk_configs={output_path}")
    print(f"wrote_config_dir={config_dir}")


def load_selected_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config_path:
        return json.loads(resolve_path(args.config_path).read_text(encoding="utf-8"))
    payload = json.loads(resolve_path(args.configs_json).read_text(encoding="utf-8"))
    configs = payload.get("configs", [])
    if not configs:
        raise ValueError("configs JSON contains no configs")
    return configs[args.config_rank]


def command_assemble(args: argparse.Namespace) -> None:
    artifact = load_pth(args.bld_pth)
    selected_config = load_selected_config(args)
    _, student = assemble_student_from_bld(artifact, selected_config)

    output = {
        "stage": "assembled_model",
        "bld_pth": str(resolve_path(args.bld_pth)),
        "config": artifact["config"],
        "architecture_config": selected_config,
        "parent_state_dict": artifact["parent_state_dict"],
        "model_state_dict": cpu_state_dict(student),
    }
    pth_path = save_pth(args.output_pth, output)
    summary_path = write_json(
        args.summary_json,
        {
            "stage": "assembled_model",
            "pth": str(pth_path),
            "rank": selected_config.get("rank"),
            "selected": [item["name"] for item in selected_config["selected"]],
        },
    )
    print(f"wrote_assembled_pth={pth_path}")
    print(f"wrote_summary={summary_path}")


def command_gkd(args: argparse.Namespace) -> None:
    artifact = load_pth(args.assembled_pth)
    device = resolve_device(args.device)
    bld_artifact = load_pth(artifact["bld_pth"])
    teacher, student = assemble_student_from_bld(bld_artifact, artifact["architecture_config"])
    student.load_state_dict(artifact["model_state_dict"])
    config = config_from_dict(artifact["config"])
    token_batches = random_token_batches(
        config.vocab_size,
        args.batch_size,
        args.seq_len,
        num_batches=args.num_batches,
        seed=args.data_seed,
    )
    losses = global_knowledge_distillation(
        teacher,
        student,
        token_batches,
        steps=args.gkd_steps,
        lr=args.lr,
        device=device,
        include_lm_loss=args.include_lm_loss,
        opd_weight=args.opd_weight,
        opd_max_new_tokens=args.opd_max_new_tokens,
        opd_temperature=args.opd_temperature,
        opd_top_k=args.opd_top_k,
    )
    output = {
        "stage": "gkd_model",
        "assembled_pth": str(resolve_path(args.assembled_pth)),
        "config": artifact["config"],
        "architecture_config": artifact["architecture_config"],
        "losses": losses,
        "parent_state_dict": artifact["parent_state_dict"],
        "model_state_dict": cpu_state_dict(student),
    }
    pth_path = save_pth(args.output_pth, output)
    summary_path = write_json(
        args.summary_json,
        {
            "stage": "gkd_model",
            "pth": str(pth_path),
            "num_losses": len(losses),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "opd_weight": args.opd_weight,
            "opd_max_new_tokens": args.opd_max_new_tokens,
        },
    )
    print(f"wrote_gkd_pth={pth_path}")
    print(f"wrote_summary={summary_path}")


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--data-seed", type=int, default=13)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the staged toy Distillation NAS workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bld = subparsers.add_parser("bld", help="Stage 04: train BLD block library and save a .pth artifact.")
    add_common_model_args(bld)
    bld.add_argument("--bld-steps", type=int, default=1)
    bld.add_argument("--lr", type=float, default=3e-3)
    bld.add_argument("--attention-variants", default=DEFAULT_ATTENTION_VARIANTS)
    bld.add_argument("--layer-variants", default=DEFAULT_LAYER_VARIANTS)
    bld.add_argument("--output-pth", default=str(DEFAULT_BLD_OUTPUT_DIR / "block_library.pth"))
    bld.add_argument("--summary-json", default=str(DEFAULT_BLD_OUTPUT_DIR / "summary.json"))
    bld.set_defaults(func=command_bld)

    score = subparsers.add_parser("score", help="Stage 05: score candidates and write layer importance JSON.")
    score.add_argument("--bld-pth", default=str(DEFAULT_BLD_OUTPUT_DIR / "block_library.pth"))
    score.add_argument("--device", default="auto")
    score.add_argument("--score-batches", type=int, default=2)
    score.add_argument("--batch-sizes", default="1,2,4")
    score.add_argument("--output-json", default=str(DEFAULT_NAS_OUTPUT_DIR / "layer_importance.json"))
    score.set_defaults(func=command_score)

    mip = subparsers.add_parser("mip", help="Stage 06: solve MIP and write top-K architecture configs.")
    mip.add_argument("--scores-json", default=str(DEFAULT_NAS_OUTPUT_DIR / "layer_importance.json"))
    mip.add_argument("--output-json", default=str(DEFAULT_MIP_OUTPUT_DIR / "topk_architecture_configs.json"))
    mip.add_argument("--config-dir", default=str(DEFAULT_MIP_OUTPUT_DIR / "configs"))
    mip.add_argument("--top-k", type=int, default=3)
    mip.add_argument("--batch-sizes", default="1,2,4")
    mip.add_argument("--memory-fraction", type=float, default=0.82)
    mip.add_argument("--runtime-fraction", type=float, default=0.82)
    mip.add_argument("--diversity-alpha", type=float, default=0.75)
    mip.add_argument("--objective-mode", choices=["score", "weighted"], default="score")
    mip.add_argument("--score-weight", type=float, default=1.0)
    mip.add_argument("--memory-weight", type=float, default=0.0)
    mip.add_argument("--runtime-weight", type=float, default=0.0)
    mip.add_argument("--no-normalize-objectives", action="store_true")
    mip.set_defaults(func=command_mip)

    assemble = subparsers.add_parser("assemble", help="Stage 07: assemble a student model from a selected config.")
    assemble.add_argument("--bld-pth", default=str(DEFAULT_BLD_OUTPUT_DIR / "block_library.pth"))
    assemble.add_argument("--configs-json", default=str(DEFAULT_MIP_OUTPUT_DIR / "topk_architecture_configs.json"))
    assemble.add_argument("--config-path", default=None)
    assemble.add_argument("--config-rank", type=int, default=0)
    assemble.add_argument("--output-pth", default=str(DEFAULT_ASSEMBLY_OUTPUT_DIR / "assembled_model.pth"))
    assemble.add_argument("--summary-json", default=str(DEFAULT_ASSEMBLY_OUTPUT_DIR / "summary.json"))
    assemble.set_defaults(func=command_assemble)

    gkd = subparsers.add_parser("gkd", help="Stage 08: run global knowledge distillation on the assembled model.")
    gkd.add_argument("--assembled-pth", default=str(DEFAULT_ASSEMBLY_OUTPUT_DIR / "assembled_model.pth"))
    gkd.add_argument("--device", default="auto")
    gkd.add_argument("--seq-len", type=int, default=16)
    gkd.add_argument("--batch-size", type=int, default=2)
    gkd.add_argument("--num-batches", type=int, default=4)
    gkd.add_argument("--data-seed", type=int, default=17)
    gkd.add_argument("--gkd-steps", type=int, default=2)
    gkd.add_argument("--lr", type=float, default=1e-4)
    gkd.add_argument("--include-lm-loss", action="store_true")
    gkd.add_argument("--opd-weight", type=float, default=0.0)
    gkd.add_argument("--opd-max-new-tokens", type=int, default=0)
    gkd.add_argument("--opd-temperature", type=float, default=None)
    gkd.add_argument("--opd-top-k", type=int, default=None)
    gkd.add_argument("--output-pth", default=str(DEFAULT_GKD_OUTPUT_DIR / "gkd_model.pth"))
    gkd.add_argument("--summary-json", default=str(DEFAULT_GKD_OUTPUT_DIR / "summary.json"))
    gkd.set_defaults(func=command_gkd)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
