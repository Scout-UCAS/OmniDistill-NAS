from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class SearchCandidate:
    layer_idx: int
    name: str
    score: float
    param_memory: float
    kv_cache_memory: float
    runtimes: dict[int, float]
    payload: Any = None


@dataclass(frozen=True)
class SearchConstraints:
    seq_len: int
    batch_sizes: Sequence[int]
    memory_max: float | None = None
    throughput_min: float | None = None
    latency_max: float | None = None
    score_direction: str = "minimize"
    diversity_alpha: float | None = None
    previous_solutions: Sequence[Sequence[str]] = field(default_factory=tuple)
    time_limit_seconds: float | None = 30.0


@dataclass(frozen=True)
class NasSolution:
    batch_size: int
    selected: list[SearchCandidate]
    total_score: float
    total_memory: float
    total_runtime: float
    throughput: float
    objective: float

    @property
    def selected_names(self) -> list[str]:
        return [candidate.name for candidate in self.selected]


def solve_nas_mip(
    candidates_by_layer: Sequence[Sequence[SearchCandidate]],
    constraints: SearchConstraints,
) -> NasSolution:
    """Solve the grouped knapsack MILP over one or more batch sizes."""

    if constraints.score_direction not in {"minimize", "maximize"}:
        raise ValueError("score_direction must be 'minimize' or 'maximize'")
    if not candidates_by_layer:
        raise ValueError("candidates_by_layer must not be empty")

    solutions: list[NasSolution] = []
    errors: list[str] = []
    for batch_size in constraints.batch_sizes:
        try:
            solutions.append(_solve_for_batch(candidates_by_layer, constraints, batch_size))
        except RuntimeError as exc:
            errors.append(f"batch={batch_size}: {exc}")

    if not solutions:
        joined = "; ".join(errors) if errors else "no feasible solution"
        raise RuntimeError(joined)

    reverse = constraints.score_direction == "maximize"
    return sorted(solutions, key=lambda item: item.total_score, reverse=reverse)[0]


def _solve_for_batch(
    candidates_by_layer: Sequence[Sequence[SearchCandidate]],
    constraints: SearchConstraints,
    batch_size: int,
) -> NasSolution:
    flat: list[SearchCandidate] = [candidate for layer in candidates_by_layer for candidate in layer]
    offsets = np.cumsum([0] + [len(layer) for layer in candidates_by_layer])
    num_vars = len(flat)

    c = np.array([candidate.score for candidate in flat], dtype=float)
    if constraints.score_direction == "maximize":
        c = -c

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for layer_idx in range(len(candidates_by_layer)):
        row = np.zeros(num_vars)
        row[offsets[layer_idx] : offsets[layer_idx + 1]] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    if constraints.memory_max is not None:
        rows.append(
            np.array(
                [
                    candidate.param_memory + batch_size * candidate.kv_cache_memory
                    for candidate in flat
                ],
                dtype=float,
            )
        )
        lower.append(-np.inf)
        upper.append(float(constraints.memory_max))

    runtime_row = np.array([candidate.runtimes[batch_size] for candidate in flat], dtype=float)
    runtime_cap = None
    if constraints.latency_max is not None:
        runtime_cap = float(constraints.latency_max)
    if constraints.throughput_min is not None:
        throughput_cap = batch_size * constraints.seq_len / float(constraints.throughput_min)
        runtime_cap = throughput_cap if runtime_cap is None else min(runtime_cap, throughput_cap)
    if runtime_cap is not None:
        rows.append(runtime_row)
        lower.append(-np.inf)
        upper.append(runtime_cap)

    if constraints.diversity_alpha is not None:
        for previous in constraints.previous_solutions:
            previous_set = set(previous)
            row = np.array([1.0 if candidate.name in previous_set else 0.0 for candidate in flat])
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float(constraints.diversity_alpha) * len(candidates_by_layer))

    matrix = np.vstack(rows)
    linear_constraint = LinearConstraint(matrix, np.array(lower), np.array(upper))
    result = milp(
        c=c,
        integrality=np.ones(num_vars),
        bounds=Bounds(np.zeros(num_vars), np.ones(num_vars)),
        constraints=linear_constraint,
        options={"time_limit": constraints.time_limit_seconds} if constraints.time_limit_seconds else None,
    )

    if not result.success:
        return _solve_exhaustive(candidates_by_layer, constraints, batch_size)

    chosen_indices = np.flatnonzero(result.x > 0.5).tolist()
    selected = [flat[index] for index in chosen_indices]
    return _make_solution(selected, constraints, batch_size)


def _solve_exhaustive(
    candidates_by_layer: Sequence[Sequence[SearchCandidate]],
    constraints: SearchConstraints,
    batch_size: int,
) -> NasSolution:
    total_combinations = 1
    for layer in candidates_by_layer:
        total_combinations *= len(layer)
    if total_combinations > 500_000:
        raise RuntimeError("MILP failed and exhaustive fallback is too large")

    best: NasSolution | None = None
    for selected_tuple in itertools.product(*candidates_by_layer):
        selected = list(selected_tuple)
        solution = _make_solution(selected, constraints, batch_size)
        if not _is_feasible(solution, constraints):
            continue
        if best is None:
            best = solution
        elif constraints.score_direction == "minimize" and solution.total_score < best.total_score:
            best = solution
        elif constraints.score_direction == "maximize" and solution.total_score > best.total_score:
            best = solution

    if best is None:
        raise RuntimeError("no feasible solution")
    return best


def _make_solution(
    selected: list[SearchCandidate],
    constraints: SearchConstraints,
    batch_size: int,
) -> NasSolution:
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
    )


def _is_feasible(solution: NasSolution, constraints: SearchConstraints) -> bool:
    if constraints.memory_max is not None and solution.total_memory > constraints.memory_max + 1e-8:
        return False
    if constraints.latency_max is not None and solution.total_runtime > constraints.latency_max + 1e-8:
        return False
    if constraints.throughput_min is not None and solution.throughput + 1e-8 < constraints.throughput_min:
        return False
    if constraints.diversity_alpha is not None:
        selected = set(solution.selected_names)
        for previous in constraints.previous_solutions:
            overlap = sum(1 for name in previous if name in selected)
            if overlap > constraints.diversity_alpha * len(solution.selected) + 1e-8:
                return False
    return True
