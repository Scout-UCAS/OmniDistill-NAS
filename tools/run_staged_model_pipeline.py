from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.distill import AUXILIARY_BATCH_KEYS, global_knowledge_distillation
from distill_nas_core.mip import SearchCandidate, SearchConstraints, solve_nas_mip
from tools.run_qwen3_attention_search import (
    DEFAULT_HF_CACHE,
    DEFAULT_VARIANTS,
    FLA_REPO,
    VENDOR,
    QwenCandidateLayer,
    dtype_from_name,
    dtype_nbytes,
    effective_param_memory_bytes,
    expand_variants,
    find_decoder_layers,
    kv_cache_bytes,
    load_model_bundle,
    load_prompts_from_args,
    make_batches,
    precompute_parent_targets,
    resolve_device,
    runtime_proxy,
    score_target_distance,
)


DEFAULT_WORKFLOW_OUTPUT_DIR = Path("outputs/distill_nas_workflow")
DEFAULT_BLD_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "04_bld_block_library"
DEFAULT_NAS_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "05_nas_layer_scoring"
DEFAULT_MIP_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "06_mip_topk_architecture_configs"
DEFAULT_ASSEMBLY_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "07_model_assembly"
DEFAULT_GKD_OUTPUT_DIR = DEFAULT_WORKFLOW_OUTPUT_DIR / "08_global_knowledge_distillation"


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
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


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    elif hasattr(config, "__dict__"):
        raw = dict(config.__dict__)
    else:
        raw = {}
    return {
        key: value
        for key, value in raw.items()
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict))
    }


def make_context(args: argparse.Namespace):
    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype, device)
    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_model_bundle(args, device=device, dtype=dtype, cache_dir=cache_dir)
    examples, prompt_metadata = load_prompts_from_args(args, loaded.model_kind)
    batches = make_batches(loaded, examples, args.seq_len, device, args)
    return loaded, batches, prompt_metadata, device, dtype, cache_dir


def forward_model(model: nn.Module, batch: dict[str, Any], **kwargs):
    try:
        return model(**batch, **kwargs)
    except TypeError as exc:
        if not kwargs:
            filtered = {key: value for key, value in batch.items() if key not in AUXILIARY_BATCH_KEYS}
            if len(filtered) == len(batch):
                raise
            try:
                return model(**filtered)
            except TypeError:
                raise exc
        try:
            return model(**batch)
        except TypeError:
            filtered = {key: value for key, value in batch.items() if key not in AUXILIARY_BATCH_KEYS}
            if len(filtered) == len(batch):
                raise exc
            filtered_merged = dict(filtered)
            filtered_merged.update(kwargs)
            try:
                return model(**filtered_merged)
            except TypeError:
                try:
                    return model(**filtered)
                except TypeError:
                    raise exc


def iter_layer_indices(num_layers: int, max_layers: int, layer_stride: int) -> list[int]:
    indices = list(range(0, num_layers, max(layer_stride, 1)))
    if max_layers > 0:
        indices = indices[:max_layers]
    return indices


def detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    return value


def capture_layer_samples(model: nn.Module, layer: nn.Module, batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    def hook(_module, inputs, kwargs, output):
        hidden_states = kwargs.get("hidden_states") if isinstance(kwargs, dict) else None
        if hidden_states is None and inputs:
            hidden_states = inputs[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError("failed to capture decoder hidden_states for BLD")
        output_hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(output_hidden, torch.Tensor):
            raise RuntimeError("decoder layer output is not a tensor")
        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("hidden_states", None)
        samples.append(
            {
                "hidden_states": hidden_states.detach(),
                "kwargs": detach_tree(layer_kwargs),
                "target": output_hidden.detach(),
            }
        )

    try:
        handle = layer.register_forward_hook(hook, with_kwargs=True)
    except TypeError as exc:  # pragma: no cover - old PyTorch only
        raise RuntimeError("Qwen BLD requires a PyTorch version with forward hooks that capture kwargs") from exc
    try:
        with torch.inference_mode():
            for batch in batches:
                forward_model(model, batch, use_cache=False)
    finally:
        handle.remove()
    return samples


def replacement_attr(candidate: QwenCandidateLayer) -> str | None:
    if candidate.qwen_attn is not None:
        return "qwen_attn"
    if candidate.fla_attn is not None:
        return "fla_attn"
    return None


def replacement_state(candidate: QwenCandidateLayer) -> tuple[str | None, dict[str, torch.Tensor] | None]:
    attr = replacement_attr(candidate)
    if attr is None:
        return None, None
    module = getattr(candidate, attr)
    return attr, {key: value.detach().cpu() for key, value in module.state_dict().items()}


def load_state_dict_checked(module: nn.Module, state: dict[str, torch.Tensor], strict: bool, context: str) -> None:
    result = module.load_state_dict(state, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if not strict and (missing or unexpected):
        print(
            f"partial_checkpoint_load context={context} missing={missing} unexpected={unexpected}",
            file=sys.stderr,
        )


def load_replacement_state(candidate: QwenCandidateLayer, record: dict[str, Any] | None, strict: bool = True) -> None:
    if not record:
        return
    attr = record.get("replacement_attr")
    state = record.get("replacement_state_dict")
    if not attr or not state:
        return
    module = getattr(candidate, attr, None)
    if module is None:
        raise RuntimeError(f"candidate {candidate.variant} has no replacement module {attr!r}")
    load_state_dict_checked(module, state, strict=strict, context=f"{candidate.variant}.{attr}")


def restore_selected_replacements(layers: nn.ModuleList, replacements: list[dict[str, Any]], strict: bool = True) -> None:
    for record in replacements:
        layer_idx = int(record["layer_idx"])
        layer = layers[layer_idx]
        if not isinstance(layer, QwenCandidateLayer):
            if record.get("replacement_state_dict"):
                raise RuntimeError(
                    f"cannot restore replacement weights for layer {layer_idx}: assembled layer is not a QwenCandidateLayer"
                )
            continue
        load_replacement_state(layer, record, strict=strict)


def train_candidate(
    candidate: QwenCandidateLayer,
    samples: list[dict[str, Any]],
    steps: int,
    lr: float,
) -> list[float]:
    attr = replacement_attr(candidate)
    if steps <= 0 or attr is None:
        return []
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    module = getattr(candidate, attr)
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        return []

    candidate.train()
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    losses: list[float] = []
    for step in range(steps):
        sample = samples[step % len(samples)]
        output = candidate(sample["hidden_states"], **sample["kwargs"])[0]
        target = sample["target"].to(device=output.device, dtype=output.dtype)
        denom = target.float().pow(2).mean().clamp_min(1e-8)
        loss = F.mse_loss(output.float(), target.float()) / denom
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def record_key(layer_idx: int, variant: str) -> tuple[int, str]:
    return int(layer_idx), variant


def records_by_key(artifact: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {record_key(item["layer_idx"], item["variant"]): item for item in artifact.get("records", [])}


def make_candidate(
    base_layer: nn.Module,
    variant: str,
    language_config: Any,
    layer_idx: int,
    fla_mode: str,
    fla_feature_map: str,
    record: dict[str, Any] | None = None,
    strict_checkpoint: bool = True,
) -> QwenCandidateLayer:
    candidate = QwenCandidateLayer(
        base_layer,
        variant,
        config=language_config,
        layer_idx=layer_idx,
        fla_mode=fla_mode,
        fla_feature_map=fla_feature_map,
    )
    load_replacement_state(candidate, record, strict=strict_checkpoint)
    return candidate


def apply_architecture(
    layers: nn.ModuleList,
    language_config: Any,
    selected_config: dict[str, Any],
    artifact: dict[str, Any],
    strict_checkpoint: bool = True,
) -> list[dict[str, Any]]:
    record_map = records_by_key(artifact)
    applied: list[dict[str, Any]] = []
    for selected in selected_config["selected"]:
        layer_idx = int(selected["layer_idx"])
        variant = selected["variant"]
        record = record_map.get(record_key(layer_idx, variant))
        candidate = make_candidate(
            layers[layer_idx],
            variant,
            language_config,
            layer_idx=layer_idx,
            fla_mode=artifact.get("fla", {}).get("mode", "chunk"),
            fla_feature_map=artifact.get("fla", {}).get("feature_map", "elu"),
            record=record,
            strict_checkpoint=strict_checkpoint,
        )
        layers[layer_idx] = candidate
        attr, state = replacement_state(candidate)
        applied.append(
            {
                "layer_idx": layer_idx,
                "variant": variant,
                "replacement_attr": attr,
                "replacement_state_dict": state,
            }
        )
    return applied


def score_candidate(
    model: nn.Module,
    layers: nn.ModuleList,
    layer_idx: int,
    variant: str,
    batches: list[dict[str, Any]],
    parent_targets,
    model_kind: str,
    language_config: Any,
    fla_mode: str,
    fla_feature_map: str,
    record: dict[str, Any] | None,
) -> tuple[float, float]:
    import time

    original_layer = layers[layer_idx]
    layers[layer_idx] = make_candidate(
        original_layer,
        variant,
        language_config,
        layer_idx=layer_idx,
        fla_mode=fla_mode,
        fla_feature_map=fla_feature_map,
        record=record,
    )
    values: list[float] = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch, teacher_target in zip(batches, parent_targets, strict=True):
                child_output = forward_model(model, batch, use_cache=False)
                score = score_target_distance(teacher_target, child_output, model_kind=model_kind)
                values.append(float(score.detach().cpu()))
    finally:
        layers[layer_idx] = original_layer
    return sum(values) / len(values), time.perf_counter() - started


def command_bld(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(args.bld_steps > 0)
    loaded, batches, prompt_metadata, device, dtype, cache_dir = make_context(args)
    layer_indices = iter_layer_indices(len(loaded.layers), args.max_layers, args.layer_stride)
    variants = expand_variants(args.variants)
    records: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}

    for layer_idx in layer_indices:
        samples = []
        if args.bld_steps > 0:
            samples = capture_layer_samples(loaded.model, loaded.layers[layer_idx], batches)
        for variant in variants:
            if variant in skipped:
                continue
            try:
                candidate = make_candidate(
                    loaded.layers[layer_idx],
                    variant,
                    loaded.language_config,
                    layer_idx=layer_idx,
                    fla_mode=args.fla_mode,
                    fla_feature_map=args.fla_feature_map,
                )
                losses = train_candidate(candidate, samples, args.bld_steps, args.lr)
                attr, state = replacement_state(candidate)
            except Exception as exc:
                if variant.startswith("fla_") and not args.no_skip_unavailable_fla:
                    skipped[variant] = f"{type(exc).__name__}: {exc}"
                    print(f"skipped_variant={variant} reason={skipped[variant]}", file=sys.stderr)
                    continue
                raise
            records.append(
                {
                    "layer_idx": layer_idx,
                    "variant": variant,
                    "name": f"L{layer_idx}:{variant}",
                    "replacement_attr": attr,
                    "replacement_state_dict": state if args.bld_steps > 0 else None,
                    "bld_losses": losses,
                }
            )

    artifact = {
        "stage": "bld_block_library",
        "backend": "qwen_vlm_vla",
        "format_version": 1,
        "model_id": args.model_id,
        "model_kind": loaded.model_kind,
        "decoder_layer_path": loaded.layer_path,
        "language_config": config_to_dict(loaded.language_config),
        "device": str(device),
        "dtype": str(dtype),
        "cache_dir": str(cache_dir),
        "vendor_dir": str(VENDOR),
        "fla_repo_dir": str(FLA_REPO),
        "seq_len": args.seq_len,
        "max_prompts": args.max_prompts,
        "prompt_source": prompt_metadata,
        "searched_layers": layer_indices,
        "variants": variants,
        "fla": {"mode": args.fla_mode, "feature_map": args.fla_feature_map, "skipped_variants": skipped},
        "bld_steps": args.bld_steps,
        "records": records,
    }
    pth_path = save_pth(args.output_pth, artifact)
    summary_path = write_json(
        args.summary_json,
        {
            "stage": "bld_block_library",
            "backend": "qwen_vlm_vla",
            "pth": str(pth_path),
            "model_id": args.model_id,
            "model_kind": loaded.model_kind,
            "num_layers": len(layer_indices),
            "num_records": len(records),
            "variants": variants,
            "skipped_variants": skipped,
        },
    )
    print(f"wrote_bld_pth={pth_path}")
    print(f"wrote_summary={summary_path}")


def command_score(args: argparse.Namespace) -> None:
    artifact = load_pth(args.bld_pth)
    loaded, batches, prompt_metadata, device, dtype, cache_dir = make_context(args)
    batches = batches[: args.score_batches]
    parent_targets = precompute_parent_targets(loaded.model, batches, model_kind=loaded.model_kind)
    record_map = records_by_key(artifact)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    scores: list[dict[str, Any]] = []
    skipped = dict(artifact.get("fla", {}).get("skipped_variants", {}))

    for layer_idx in artifact["searched_layers"]:
        layer = loaded.layers[int(layer_idx)]
        for variant in artifact["variants"]:
            if variant in skipped:
                continue
            try:
                score, measured_seconds = score_candidate(
                    loaded.model,
                    loaded.layers,
                    int(layer_idx),
                    variant,
                    batches,
                    parent_targets,
                    loaded.model_kind,
                    loaded.language_config,
                    artifact.get("fla", {}).get("mode", "chunk"),
                    artifact.get("fla", {}).get("feature_map", "elu"),
                    record_map.get(record_key(int(layer_idx), variant)),
                )
                dtype_bytes = dtype_nbytes(dtype)
                param_memory = effective_param_memory_bytes(layer, variant, dtype_bytes)
                kv_memory = kv_cache_bytes(loaded.language_config, variant, args.seq_len, dtype_bytes)
                proxy = runtime_proxy(layer, loaded.language_config, variant, args.seq_len, dtype_bytes)
            except Exception as exc:
                if variant.startswith("fla_") and not args.no_skip_unavailable_fla:
                    skipped[variant] = f"{type(exc).__name__}: {exc}"
                    print(f"skipped_variant={variant} reason={skipped[variant]}", file=sys.stderr)
                    continue
                raise
            scores.append(
                {
                    "layer_idx": int(layer_idx),
                    "name": f"L{layer_idx}:{variant}",
                    "variant": variant,
                    "score": score,
                    "metric": parent_targets[0].metric,
                    "target_name": parent_targets[0].name,
                    "param_memory": float(param_memory),
                    "kv_cache_memory": float(kv_memory),
                    "runtimes": {str(batch_size): float(proxy) * batch_size for batch_size in batch_sizes},
                    "measured_seconds": measured_seconds,
                    "source": "bld_block",
                }
            )

    by_layer: dict[int, list[dict[str, Any]]] = {}
    for score in scores:
        by_layer.setdefault(int(score["layer_idx"]), []).append(score)
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

    payload = {
        "stage": "nas_layer_importance",
        "backend": "qwen_vlm_vla",
        "bld_pth": str(resolve_path(args.bld_pth)),
        "model_id": args.model_id,
        "model_kind": loaded.model_kind,
        "decoder_layer_path": loaded.layer_path,
        "device": str(device),
        "dtype": str(dtype),
        "cache_dir": str(cache_dir),
        "seq_len": args.seq_len,
        "prompt_source": prompt_metadata,
        "batch_sizes": batch_sizes,
        "score_batches": len(batches),
        "scores": scores,
        "layer_importance": layer_importance,
        "skipped_variants": skipped,
    }
    output_path = write_json(args.output_json, payload)
    print(f"wrote_importance_json={output_path}")


def candidates_by_layer_from_scores(scores_payload: dict[str, Any]) -> list[list[SearchCandidate]]:
    layers: dict[int, list[SearchCandidate]] = {}
    for record in scores_payload["scores"]:
        candidate = SearchCandidate(
            layer_idx=int(record["layer_idx"]),
            name=record["name"],
            score=float(record["score"]),
            param_memory=float(record["param_memory"]),
            kv_cache_memory=float(record["kv_cache_memory"]),
            runtimes={int(key): float(value) for key, value in record["runtimes"].items()},
            payload={"variant": record["variant"]},
        )
        layers.setdefault(candidate.layer_idx, []).append(candidate)
    return [layers[index] for index in sorted(layers)]


def solution_to_config(solution, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "selected_batch_size": solution.batch_size,
        "total_score": solution.total_score,
        "total_memory": solution.total_memory,
        "total_runtime": solution.total_runtime,
        "throughput": solution.throughput,
        "objective": solution.objective,
        "objective_components": solution.objective_components,
        "selected": [
            {
                "layer_idx": candidate.layer_idx,
                "name": candidate.name,
                "variant": candidate.payload["variant"],
                "score": candidate.score,
                "param_memory": candidate.param_memory,
                "kv_cache_memory": candidate.kv_cache_memory,
                "runtimes": {str(key): value for key, value in candidate.runtimes.items()},
            }
            for candidate in solution.selected
        ],
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
            parent = next((candidate for candidate in layer if candidate.payload["variant"] == "parent_attn"), None)
        if parent is None:
            raise RuntimeError("each searched layer must include a parent or parent_attn candidate")
        parent_candidates.append(parent)
    parent_memory_by_batch = {
        batch_size: sum(candidate.param_memory + batch_size * candidate.kv_cache_memory for candidate in parent_candidates)
        for batch_size in batch_sizes
    }
    parent_runtime_by_batch = {
        batch_size: sum(candidate.runtimes[batch_size] for candidate in parent_candidates)
        for batch_size in batch_sizes
    }
    previous: list[list[str]] = []
    configs = []
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

    payload = {
        "stage": "mip_topk_configs",
        "backend": scores_payload.get("backend", "qwen_vlm_vla"),
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
    output_path = write_json(args.output_json, payload)
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
    loaded, batches, _prompt_metadata, _device, _dtype, _cache_dir = make_context(args)
    selected_replacements = apply_architecture(
        loaded.layers,
        loaded.language_config,
        selected_config,
        artifact,
        strict_checkpoint=not args.allow_partial_checkpoint_load,
    )
    if not args.skip_forward_check and batches:
        with torch.inference_mode():
            forward_model(loaded.model, batches[0], use_cache=False)
    output = {
        "stage": "assembled_model",
        "backend": "qwen_vlm_vla",
        "bld_pth": str(resolve_path(args.bld_pth)),
        "model_id": args.model_id,
        "model_kind": loaded.model_kind,
        "decoder_layer_path": loaded.layer_path,
        "architecture_config": selected_config,
        "selected_replacements": selected_replacements,
    }
    if args.save_full_state_dict:
        output["model_state_dict"] = {key: value.detach().cpu() for key, value in loaded.model.state_dict().items()}
    pth_path = save_pth(args.output_pth, output)
    summary_path = write_json(
        args.summary_json,
        {
            "stage": "assembled_model",
            "backend": "qwen_vlm_vla",
            "pth": str(pth_path),
            "rank": selected_config.get("rank"),
            "selected": [item["name"] for item in selected_config["selected"]],
            "save_full_state_dict": args.save_full_state_dict,
            "strict_checkpoint_load": not args.allow_partial_checkpoint_load,
        },
    )
    print(f"wrote_assembled_pth={pth_path}")
    print(f"wrote_summary={summary_path}")


def command_gkd(args: argparse.Namespace) -> None:
    assembled = load_pth(args.assembled_pth)
    bld_artifact = load_pth(assembled["bld_pth"])
    teacher_args = copy.copy(args)
    student_args = copy.copy(args)
    base_device = getattr(args, "device", "auto")
    teacher_args.device = args.teacher_device or base_device
    student_args.device = args.student_device or base_device
    teacher_loaded, batches, _prompt_metadata, teacher_device, _dtype, _cache_dir = make_context(teacher_args)
    student_loaded, _student_batches, _student_prompt_metadata, student_device, _student_dtype, _student_cache_dir = make_context(student_args)
    apply_architecture(
        student_loaded.layers,
        student_loaded.language_config,
        assembled["architecture_config"],
        bld_artifact,
        strict_checkpoint=not args.allow_partial_checkpoint_load,
    )
    restore_selected_replacements(
        student_loaded.layers,
        assembled.get("selected_replacements", []),
        strict=not args.allow_partial_checkpoint_load,
    )
    if "model_state_dict" in assembled:
        load_state_dict_checked(
            student_loaded.model,
            assembled["model_state_dict"],
            strict=not args.allow_partial_checkpoint_load,
            context="assembled_model_state_dict",
        )
    losses = global_knowledge_distillation(
        teacher_loaded.model,
        student_loaded.model,
        batches,
        steps=args.gkd_steps,
        lr=args.lr,
        device=student_device,
        include_lm_loss=args.include_lm_loss,
        opd_weight=args.opd_weight,
        opd_max_new_tokens=args.opd_max_new_tokens,
        opd_temperature=args.opd_temperature,
        opd_top_k=args.opd_top_k,
        teacher_device=teacher_device,
        student_device=student_device,
        strict_action_opd=args.strict_action_opd,
    )
    selected_replacements = []
    for selected in assembled["architecture_config"]["selected"]:
        layer = student_loaded.layers[int(selected["layer_idx"])]
        if isinstance(layer, QwenCandidateLayer):
            attr, state = replacement_state(layer)
        else:
            attr, state = None, None
        selected_replacements.append(
            {
                "layer_idx": int(selected["layer_idx"]),
                "variant": selected["variant"],
                "replacement_attr": attr,
                "replacement_state_dict": state,
            }
        )
    output = {
        "stage": "gkd_model",
        "backend": "qwen_vlm_vla",
        "assembled_pth": str(resolve_path(args.assembled_pth)),
        "model_id": args.model_id,
        "model_kind": teacher_loaded.model_kind,
        "architecture_config": assembled["architecture_config"],
        "losses": losses,
        "selected_replacements": selected_replacements,
    }
    if args.save_full_state_dict:
        output["model_state_dict"] = {key: value.detach().cpu() for key, value in student_loaded.model.state_dict().items()}
    pth_path = save_pth(args.output_pth, output)
    summary_path = write_json(
        args.summary_json,
        {
            "stage": "gkd_model",
            "backend": "qwen_vlm_vla",
            "pth": str(pth_path),
            "num_losses": len(losses),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "opd_weight": args.opd_weight,
            "opd_max_new_tokens": args.opd_max_new_tokens,
            "teacher_device": str(teacher_device),
            "student_device": str(student_device),
            "strict_action_opd": args.strict_action_opd,
            "strict_checkpoint_load": not args.allow_partial_checkpoint_load,
            "save_full_state_dict": args.save_full_state_dict,
        },
    )
    print(f"wrote_gkd_pth={pth_path}")
    print(f"wrote_summary={summary_path}")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-kind", default="auto", choices=["auto", "text", "vlm", "vla"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE / "models"))
    parser.add_argument("--prompt-source", default="built_in")
    parser.add_argument("--mmlu-dataset", default="cais/mmlu")
    parser.add_argument("--mmlu-subject", default="abstract_algebra")
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-task", default="auto", choices=["auto", "llm", "vlm", "vla"])
    parser.add_argument("--dataset-image-root", default=None)
    parser.add_argument("--include-dataset-target", action="store_true")
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--vlm-blank-image-size", type=int, default=224)
    parser.add_argument("--allow-blank-image", action="store_true")
    parser.add_argument("--no-vlm-generation-prompt", action="store_true")
    parser.add_argument("--dataset-cache-dir", default=str(DEFAULT_HF_CACHE / "datasets"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=2)


def add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-layers", type=int, default=2, help="0 means all layers.")
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--fla-mode", default="chunk")
    parser.add_argument("--fla-feature-map", default="elu")
    parser.add_argument("--no-skip-unavailable-fla", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged BLD/NAS/MIP/assembly/GKD for Qwen-style LLM/VLM/VLA models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bld = subparsers.add_parser("bld", help="Stage 04: build or train a real-model candidate block library.")
    add_model_args(bld)
    add_search_args(bld)
    bld.add_argument("--bld-steps", type=int, default=0)
    bld.add_argument("--lr", type=float, default=1e-4)
    bld.add_argument("--output-pth", default=str(DEFAULT_BLD_OUTPUT_DIR / "block_library.pth"))
    bld.add_argument("--summary-json", default=str(DEFAULT_BLD_OUTPUT_DIR / "summary.json"))
    bld.set_defaults(func=command_bld)

    score = subparsers.add_parser("score", help="Stage 05: score real-model candidates and layer importance.")
    add_model_args(score)
    score.add_argument("--bld-pth", default=str(DEFAULT_BLD_OUTPUT_DIR / "block_library.pth"))
    score.add_argument("--score-batches", type=int, default=2)
    score.add_argument("--batch-sizes", default="1,2,4")
    score.add_argument("--no-skip-unavailable-fla", action="store_true")
    score.add_argument("--output-json", default=str(DEFAULT_NAS_OUTPUT_DIR / "layer_importance.json"))
    score.set_defaults(func=command_score)

    mip = subparsers.add_parser("mip", help="Stage 06: solve MIP and export top-K architecture configs.")
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

    assemble = subparsers.add_parser("assemble", help="Stage 07: assemble a real-model student from a selected config.")
    add_model_args(assemble)
    assemble.add_argument("--bld-pth", default=str(DEFAULT_BLD_OUTPUT_DIR / "block_library.pth"))
    assemble.add_argument("--configs-json", default=str(DEFAULT_MIP_OUTPUT_DIR / "topk_architecture_configs.json"))
    assemble.add_argument("--config-path", default=None)
    assemble.add_argument("--config-rank", type=int, default=0)
    assemble.add_argument("--skip-forward-check", action="store_true")
    assemble.add_argument("--save-full-state-dict", action="store_true")
    assemble.add_argument("--allow-partial-checkpoint-load", action="store_true")
    assemble.add_argument("--output-pth", default=str(DEFAULT_ASSEMBLY_OUTPUT_DIR / "assembled_model.pth"))
    assemble.add_argument("--summary-json", default=str(DEFAULT_ASSEMBLY_OUTPUT_DIR / "summary.json"))
    assemble.set_defaults(func=command_assemble)

    gkd = subparsers.add_parser("gkd", help="Stage 08: run GKD/OPD on the real assembled student.")
    add_model_args(gkd)
    gkd.add_argument("--assembled-pth", default=str(DEFAULT_ASSEMBLY_OUTPUT_DIR / "assembled_model.pth"))
    gkd.add_argument("--gkd-steps", type=int, default=2)
    gkd.add_argument("--lr", type=float, default=1e-4)
    gkd.add_argument("--include-lm-loss", action="store_true")
    gkd.add_argument("--opd-weight", type=float, default=0.0)
    gkd.add_argument("--opd-max-new-tokens", type=int, default=0)
    gkd.add_argument("--opd-temperature", type=float, default=None)
    gkd.add_argument("--opd-top-k", type=int, default=None)
    gkd.add_argument("--teacher-device", default=None)
    gkd.add_argument("--student-device", default=None)
    gkd.add_argument("--strict-action-opd", action="store_true")
    gkd.add_argument("--allow-partial-checkpoint-load", action="store_true")
    gkd.add_argument("--save-full-state-dict", action="store_true")
    gkd.add_argument("--output-pth", default=str(DEFAULT_GKD_OUTPUT_DIR / "gkd_model.pth"))
    gkd.add_argument("--summary-json", default=str(DEFAULT_GKD_OUTPUT_DIR / "summary.json"))
    gkd.set_defaults(func=command_gkd)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
