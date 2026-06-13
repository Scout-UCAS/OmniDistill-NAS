from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.mip import SearchCandidate, SearchConstraints
from distill_nas_core.multi_objective import (
    combination_count,
    enumerate_feasible_solutions,
    pareto_front,
    pareto_svg,
    parse_weight_grid,
    run_weight_sweep,
    solution_to_config,
    unique_solutions,
    write_multi_objective_report,
    write_pareto_configs,
)


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def candidates_by_layer_from_scores(scores_payload: dict[str, Any]) -> list[list[SearchCandidate]]:
    layers: dict[int, list[SearchCandidate]] = {}
    for record in scores_payload["scores"]:
        payload = {"variant": record.get("variant")}
        if "spec" in record:
            payload["spec"] = record["spec"]
        candidate = SearchCandidate(
            layer_idx=int(record["layer_idx"]),
            name=record["name"],
            score=float(record["score"]),
            param_memory=float(record["param_memory"]),
            kv_cache_memory=float(record["kv_cache_memory"]),
            runtimes={int(key): float(value) for key, value in record["runtimes"].items()},
            payload=payload,
        )
        layers.setdefault(candidate.layer_idx, []).append(candidate)
    return [layers[index] for index in sorted(layers)]


def find_parent_candidates(candidates_by_layer: list[list[SearchCandidate]]) -> list[SearchCandidate]:
    parent_candidates = []
    for layer in candidates_by_layer:
        parent = next(
            (
                candidate
                for candidate in layer
                if isinstance(candidate.payload, dict)
                and candidate.payload.get("variant") in {"parent", "parent_attn"}
            ),
            None,
        )
        if parent is None:
            parent = next((candidate for candidate in layer if candidate.name.endswith("parent_attn+parent_ffn")), None)
        if parent is None:
            raise RuntimeError("each searched layer must include a parent candidate")
        parent_candidates.append(parent)
    return parent_candidates


def build_constraints(
    scores_payload: dict[str, Any],
    candidates_by_layer: list[list[SearchCandidate]],
    batch_sizes: list[int],
    memory_fraction: float | None,
    runtime_fraction: float | None,
    score_direction: str,
    normalize_objectives: bool,
) -> SearchConstraints:
    parent_candidates = find_parent_candidates(candidates_by_layer)
    parent_memory_by_batch = {
        batch_size: sum(candidate.param_memory + batch_size * candidate.kv_cache_memory for candidate in parent_candidates)
        for batch_size in batch_sizes
    }
    parent_runtime_by_batch = {
        batch_size: sum(candidate.runtimes[batch_size] for candidate in parent_candidates)
        for batch_size in batch_sizes
    }
    memory_max_by_batch = None
    if memory_fraction is not None:
        memory_max_by_batch = {
            batch_size: value * memory_fraction
            for batch_size, value in parent_memory_by_batch.items()
        }
    latency_max_by_batch = None
    if runtime_fraction is not None:
        latency_max_by_batch = {
            batch_size: value * runtime_fraction
            for batch_size, value in parent_runtime_by_batch.items()
        }
    return SearchConstraints(
        seq_len=int(scores_payload["seq_len"]),
        batch_sizes=batch_sizes,
        memory_max_by_batch=memory_max_by_batch,
        latency_max_by_batch=latency_max_by_batch,
        score_direction=score_direction,
        normalize_objectives=normalize_objectives,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run weight sweep and Pareto frontier search from NAS score JSON.")
    parser.add_argument("--scores-json", default="outputs/distill_nas_workflow/05_nas_layer_scoring/layer_importance.json")
    parser.add_argument("--output-json", default="outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_search.json")
    parser.add_argument("--config-dir", default="outputs/distill_nas_workflow/06_mip_topk_architecture_configs/pareto_configs")
    parser.add_argument("--report-md", default="outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_report.md")
    parser.add_argument("--plot-svg", default="outputs/distill_nas_workflow/06_mip_topk_architecture_configs/pareto_front.svg")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--memory-fraction", type=float, default=0.82)
    parser.add_argument("--runtime-fraction", type=float, default=0.82)
    parser.add_argument("--no-memory-cap", action="store_true")
    parser.add_argument("--no-runtime-cap", action="store_true")
    parser.add_argument("--score-direction", choices=["minimize", "maximize"], default="minimize")
    parser.add_argument("--weight-grid", default=None, help="Semicolon-separated triples, e.g. '1,0,0;1,0.25,0.25'.")
    parser.add_argument("--grid-resolution", type=int, default=4)
    parser.add_argument("--pareto-mode", choices=["auto", "exact", "sweep"], default="auto")
    parser.add_argument("--max-exhaustive-combinations", type=int, default=200_000)
    parser.add_argument("--no-normalize-objectives", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scores_path = resolve_path(args.scores_json)
    scores_payload = json.loads(scores_path.read_text(encoding="utf-8"))
    candidates_by_layer = candidates_by_layer_from_scores(scores_payload)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    constraints = build_constraints(
        scores_payload,
        candidates_by_layer,
        batch_sizes,
        None if args.no_memory_cap else args.memory_fraction,
        None if args.no_runtime_cap else args.runtime_fraction,
        args.score_direction,
        not args.no_normalize_objectives,
    )
    weight_grid = parse_weight_grid(args.weight_grid, resolution=args.grid_resolution)
    sweep_records = run_weight_sweep(candidates_by_layer, constraints, weight_grid)
    sweep_solutions = unique_solutions([record["solution"] for record in sweep_records])

    exact_solutions = []
    pareto_source = "sweep"
    total_combinations = combination_count(candidates_by_layer, batch_sizes)
    if args.pareto_mode in {"auto", "exact"}:
        try:
            exact_solutions = enumerate_feasible_solutions(
                candidates_by_layer,
                constraints,
                max_combinations=args.max_exhaustive_combinations,
            )
            pareto_source = "exact"
        except RuntimeError:
            if args.pareto_mode == "exact":
                raise

    solution_pool = exact_solutions if pareto_source == "exact" else sweep_solutions
    front = pareto_front(solution_pool, score_direction=args.score_direction)
    pareto_configs = [solution_to_config(solution, rank=index, source=pareto_source) for index, solution in enumerate(front)]
    sweep_configs = [solution_to_config(solution, rank=index, source="sweep") for index, solution in enumerate(sweep_solutions)]
    all_plot_solutions = unique_solutions([*sweep_solutions, *front])

    payload = {
        "stage": "multi_objective_search",
        "backend": scores_payload.get("backend", "toy"),
        "scores_json": str(scores_path),
        "batch_sizes": batch_sizes,
        "memory_fraction": None if args.no_memory_cap else args.memory_fraction,
        "runtime_fraction": None if args.no_runtime_cap else args.runtime_fraction,
        "score_direction": args.score_direction,
        "memory_max_by_batch": {
            str(key): value for key, value in (constraints.memory_max_by_batch or {}).items()
        },
        "latency_max_by_batch": {
            str(key): value for key, value in (constraints.latency_max_by_batch or {}).items()
        },
        "weight_grid": weight_grid,
        "normalize_objectives": not args.no_normalize_objectives,
        "pareto_mode": args.pareto_mode,
        "pareto_source": pareto_source,
        "total_combinations": total_combinations,
        "num_sweep_solutions": len(sweep_configs),
        "num_pareto_solutions": len(pareto_configs),
        "sweep_solutions": sweep_configs,
        "pareto_front": pareto_configs,
        "plot_svg": pareto_svg(all_plot_solutions, front, score_direction=args.score_direction),
    }

    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({key: value for key, value in payload.items() if key != "plot_svg"}, indent=2, default=str), encoding="utf-8")
    config_dir = write_pareto_configs(pareto_configs, args.config_dir)
    report_path, plot_path = write_multi_objective_report(payload, args.report_md, args.plot_svg)
    print(f"wrote_multi_objective_json={output_path}")
    print(f"wrote_pareto_config_dir={config_dir}")
    print(f"wrote_multi_objective_report={report_path}")
    print(f"wrote_pareto_svg={plot_path}")


if __name__ == "__main__":
    main()
