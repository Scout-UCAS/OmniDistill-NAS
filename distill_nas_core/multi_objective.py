from __future__ import annotations

import html
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .artifacts import resolve_path
from .mip import NasSolution, SearchCandidate, SearchConstraints, solve_nas_mip


def generate_weight_grid(resolution: int = 4) -> list[dict[str, float]]:
    if resolution < 1:
        raise ValueError("resolution must be positive")
    weights: list[dict[str, float]] = []
    for score in range(resolution + 1):
        for memory in range(resolution + 1 - score):
            runtime = resolution - score - memory
            if score == memory == runtime == 0:
                continue
            weights.append(
                {
                    "score": score / resolution,
                    "memory": memory / resolution,
                    "runtime": runtime / resolution,
                }
            )
    return sorted(weights, key=lambda item: (-item["score"], item["memory"], item["runtime"]))


def parse_weight_grid(raw: str | None, resolution: int = 4) -> list[dict[str, float]]:
    if raw is None or not raw.strip():
        return generate_weight_grid(resolution)
    weights: list[dict[str, float]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        values = {"score": 0.0, "memory": 0.0, "runtime": 0.0}
        if "=" in chunk:
            for part in chunk.split(","):
                name, value = part.split("=", 1)
                key = name.strip().lower()
                if key not in values:
                    raise ValueError(f"unknown objective weight name: {key}")
                values[key] = float(value)
        else:
            parts = [float(part.strip()) for part in chunk.split(",") if part.strip()]
            if len(parts) != 3:
                raise ValueError("weight triples must contain score,memory,runtime")
            values = {"score": parts[0], "memory": parts[1], "runtime": parts[2]}
        if values["score"] + values["memory"] + values["runtime"] <= 0:
            raise ValueError("each weight triple must contain at least one positive value")
        weights.append(values)
    if not weights:
        raise ValueError("weight grid is empty")
    return weights


def solution_signature(solution: NasSolution) -> tuple[int, tuple[str, ...]]:
    return solution.batch_size, tuple(solution.selected_names)


def is_dominated(candidate: NasSolution, other: NasSolution, score_direction: str = "minimize") -> bool:
    if score_direction not in {"minimize", "maximize"}:
        raise ValueError("score_direction must be 'minimize' or 'maximize'")
    if score_direction == "minimize":
        score_no_worse = other.total_score <= candidate.total_score
        score_better = other.total_score < candidate.total_score
    else:
        score_no_worse = other.total_score >= candidate.total_score
        score_better = other.total_score > candidate.total_score
    memory_no_worse = other.total_memory <= candidate.total_memory
    runtime_no_worse = other.total_runtime <= candidate.total_runtime
    strictly_better = (
        score_better
        or other.total_memory < candidate.total_memory
        or other.total_runtime < candidate.total_runtime
    )
    return score_no_worse and memory_no_worse and runtime_no_worse and strictly_better


def pareto_front(solutions: Sequence[NasSolution], score_direction: str = "minimize") -> list[NasSolution]:
    front: list[NasSolution] = []
    for solution in solutions:
        if not any(
            is_dominated(solution, other, score_direction=score_direction)
            for other in solutions
            if other is not solution
        ):
            front.append(solution)
    reverse_score = score_direction == "maximize"
    return sorted(front, key=lambda item: (item.total_score * (-1 if reverse_score else 1), item.total_memory, item.total_runtime))


def _is_feasible(solution: NasSolution, constraints: SearchConstraints) -> bool:
    memory_cap = _batch_cap(constraints.memory_max_by_batch, solution.batch_size)
    if memory_cap is None:
        memory_cap = constraints.memory_max
    if memory_cap is not None and solution.total_memory > memory_cap + 1e-8:
        return False
    latency_cap = _batch_cap(constraints.latency_max_by_batch, solution.batch_size)
    if latency_cap is None:
        latency_cap = constraints.latency_max
    if latency_cap is not None and solution.total_runtime > latency_cap + 1e-8:
        return False
    if constraints.throughput_min is not None and solution.throughput + 1e-8 < constraints.throughput_min:
        return False
    return True


def _batch_cap(caps: dict[int, float] | None, batch_size: int) -> float | None:
    if not caps:
        return None
    if batch_size in caps:
        return float(caps[batch_size])
    key = str(batch_size)
    if key in caps:  # type: ignore[operator]
        return float(caps[key])  # type: ignore[index]
    return None


def _make_solution(selected: list[SearchCandidate], constraints: SearchConstraints, batch_size: int) -> NasSolution:
    total_score = float(sum(candidate.score for candidate in selected))
    total_memory = float(sum(candidate.param_memory + batch_size * candidate.kv_cache_memory for candidate in selected))
    total_runtime = float(sum(candidate.runtimes[batch_size] for candidate in selected))
    throughput = float("inf") if total_runtime == 0 else batch_size * constraints.seq_len / total_runtime
    objective = total_score if constraints.score_direction == "minimize" else -total_score
    return NasSolution(
        batch_size=batch_size,
        selected=selected,
        total_score=total_score,
        total_memory=total_memory,
        total_runtime=total_runtime,
        throughput=throughput,
        objective=objective,
        objective_components={
            "score": total_score,
            "memory": total_memory,
            "runtime": total_runtime,
            "weighted_score": objective,
        },
    )


def combination_count(candidates_by_layer: Sequence[Sequence[SearchCandidate]], batch_sizes: Sequence[int]) -> int:
    total = len(batch_sizes)
    for layer in candidates_by_layer:
        total *= len(layer)
    return total


def enumerate_feasible_solutions(
    candidates_by_layer: Sequence[Sequence[SearchCandidate]],
    constraints: SearchConstraints,
    max_combinations: int = 200_000,
) -> list[NasSolution]:
    total = combination_count(candidates_by_layer, constraints.batch_sizes)
    if total > max_combinations:
        raise RuntimeError(f"exact Pareto enumeration would inspect {total} combinations")
    solutions: list[NasSolution] = []
    for batch_size in constraints.batch_sizes:
        for selected_tuple in itertools.product(*candidates_by_layer):
            solution = _make_solution(list(selected_tuple), constraints, int(batch_size))
            if _is_feasible(solution, constraints):
                solutions.append(solution)
    return solutions


def run_weight_sweep(
    candidates_by_layer: Sequence[Sequence[SearchCandidate]],
    constraints: SearchConstraints,
    weight_grid: Sequence[dict[str, float]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[str, ...], float, float, float]] = set()
    for weights in weight_grid:
        weighted_constraints = replace(
            constraints,
            objective_mode="weighted",
            score_weight=float(weights["score"]),
            memory_weight=float(weights["memory"]),
            runtime_weight=float(weights["runtime"]),
        )
        solution = solve_nas_mip(candidates_by_layer, weighted_constraints)
        key = (
            *solution_signature(solution),
            float(weights["score"]),
            float(weights["memory"]),
            float(weights["runtime"]),
        )
        if key in seen:
            continue
        seen.add(key)
        records.append({"weights": dict(weights), "solution": solution})
    return records


def unique_solutions(solutions: Sequence[NasSolution]) -> list[NasSolution]:
    seen: set[tuple[int, tuple[str, ...]]] = set()
    unique: list[NasSolution] = []
    for solution in solutions:
        key = solution_signature(solution)
        if key in seen:
            continue
        seen.add(key)
        unique.append(solution)
    return unique


def solution_to_config(solution: NasSolution, rank: int, source: str = "pareto") -> dict[str, Any]:
    return {
        "rank": rank,
        "source": source,
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
                "variant": candidate.payload.get("variant") if isinstance(candidate.payload, dict) else None,
                "score": candidate.score,
                "param_memory": candidate.param_memory,
                "kv_cache_memory": candidate.kv_cache_memory,
                "runtimes": {str(key): value for key, value in candidate.runtimes.items()},
                **({"spec": candidate.payload["spec"]} if isinstance(candidate.payload, dict) and "spec" in candidate.payload else {}),
            }
            for candidate in solution.selected
        ],
    }


def _metric_range(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        high = low + 1.0
    return low, high


def pareto_svg(solutions: Sequence[NasSolution], pareto_solutions: Sequence[NasSolution], width: int = 720, height: int = 460) -> str:
    margin = 58
    plot_width = width - margin * 2
    plot_height = height - margin * 2
    memory_values = [solution.total_memory for solution in solutions]
    score_values = [solution.total_score for solution in solutions]
    memory_min, memory_max = _metric_range(memory_values)
    score_min, score_max = _metric_range(score_values)
    pareto_keys = {solution_signature(solution) for solution in pareto_solutions}

    def x_pos(memory: float) -> float:
        return margin + ((memory - memory_min) / (memory_max - memory_min)) * plot_width

    def y_pos(score: float) -> float:
        return height - margin - ((score - score_min) / (score_max - score_min)) * plot_height

    points = []
    for solution in solutions:
        is_front = solution_signature(solution) in pareto_keys
        color = "#d92d20" if is_front else "#667085"
        radius = 5 if is_front else 3
        title = html.escape(
            f"score={solution.total_score:.6g}, memory={solution.total_memory:.6g}, runtime={solution.total_runtime:.6g}"
        )
        points.append(
            f'<circle cx="{x_pos(solution.total_memory):.2f}" cy="{y_pos(solution.total_score):.2f}" '
            f'r="{radius}" fill="{color}" opacity="0.88"><title>{title}</title></circle>'
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#344054"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#344054"/>',
            f'<text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" font-size="13" fill="#344054">memory bytes, lower is better</text>',
            f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" font-size="13" fill="#344054" transform="rotate(-90 18 {height / 2:.1f})">score, lower is better</text>',
            f'<text x="{margin}" y="{margin - 20}" font-size="16" fill="#101828">Pareto search candidates</text>',
            *points,
            "</svg>",
        ]
    )


def write_multi_objective_report(
    payload: dict[str, Any],
    report_md: str | Path,
    plot_svg: str | Path,
) -> tuple[Path, Path]:
    report_path = resolve_path(report_md)
    plot_path = resolve_path(plot_svg)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text(payload["plot_svg"], encoding="utf-8")

    lines = [
        "# Multi-Objective NAS Report",
        "",
        f"- Scores JSON: `{payload.get('scores_json')}`",
        f"- Pareto source: `{payload.get('pareto_source')}`",
        f"- Sweep solutions: `{len(payload.get('sweep_solutions', []))}`",
        f"- Pareto solutions: `{len(payload.get('pareto_front', []))}`",
        "",
        f"![Pareto plot]({plot_path.name})",
        "",
        "| Rank | Score | Memory | Runtime | Batch | Selected |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in payload.get("pareto_front", []):
        selected = ", ".join(selected_item["name"] for selected_item in item.get("selected", []))
        lines.append(
            f"| {item['rank']} | {item['total_score']:.6g} | {item['total_memory']:.6g} | "
            f"{item['total_runtime']:.6g} | {item['selected_batch_size']} | {selected} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, plot_path


def write_pareto_configs(configs: Sequence[dict[str, Any]], config_dir: str | Path) -> Path:
    target = resolve_path(config_dir)
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("pareto_rank_*.json"):
        stale.unlink()
    for config in configs:
        path = target / f"pareto_rank_{int(config['rank']):02d}.json"
        path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return target
