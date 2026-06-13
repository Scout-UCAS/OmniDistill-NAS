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
    memory_max_by_batch: dict[int, float] | None = None
    throughput_min: float | None = None
    latency_max: float | None = None
    latency_max_by_batch: dict[int, float] | None = None
    score_direction: str = "minimize"
    objective_mode: str = "score"
    score_weight: float = 1.0
    memory_weight: float = 0.0
    runtime_weight: float = 0.0
    normalize_objectives: bool = True
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
    objective_components: dict[str, float] = field(default_factory=dict)

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
    if constraints.objective_mode not in {"score", "weighted"}:
        raise ValueError("objective_mode must be 'score' or 'weighted'")
    if any(
        weight < 0
        for weight in (constraints.score_weight, constraints.memory_weight, constraints.runtime_weight)
    ):
        raise ValueError("objective weights must be non-negative")
    if constraints.objective_mode == "weighted" and (
        constraints.score_weight + constraints.memory_weight + constraints.runtime_weight <= 0
    ):
        raise ValueError("weighted objective requires at least one positive objective weight")
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

    return sorted(solutions, key=lambda item: item.objective)[0]


def _solve_for_batch(
    candidates_by_layer: Sequence[Sequence[SearchCandidate]],
    constraints: SearchConstraints,
    batch_size: int,
) -> NasSolution:
    flat: list[SearchCandidate] = [candidate for layer in candidates_by_layer for candidate in layer]
    offsets = np.cumsum([0] + [len(layer) for layer in candidates_by_layer])
    num_vars = len(flat)

    c = _objective_coefficients(flat, constraints, batch_size)

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for layer_idx in range(len(candidates_by_layer)):
        layer_row = np.zeros(num_vars, dtype=float)
        layer_row[offsets[layer_idx] : offsets[layer_idx + 1]] = 1.0
        rows.append(layer_row)
        lower.append(1.0)
        upper.append(1.0)

    memory_cap = _memory_cap(constraints, batch_size)
    if memory_cap is not None:
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
        upper.append(float(memory_cap))

    runtime_row = np.array([candidate.runtimes[batch_size] for candidate in flat], dtype=float)
    runtime_cap = _latency_cap(constraints, batch_size)
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
            diversity_row = np.array([1.0 if candidate.name in previous_set else 0.0 for candidate in flat], dtype=float)
            rows.append(diversity_row)
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
    return _make_solution(selected, constraints, batch_size, objective_candidates=flat)


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
    flat: list[SearchCandidate] = [candidate for layer in candidates_by_layer for candidate in layer]
    for selected_tuple in itertools.product(*candidates_by_layer):
        selected = list(selected_tuple)
        solution = _make_solution(selected, constraints, batch_size, objective_candidates=flat)
        if not _is_feasible(solution, constraints):
            continue
        if best is None:
            best = solution
        elif solution.objective < best.objective:
            best = solution

    if best is None:
        raise RuntimeError("no feasible solution")
    return best


def _make_solution(
    selected: list[SearchCandidate],
    constraints: SearchConstraints,
    batch_size: int,
    objective_candidates: Sequence[SearchCandidate] | None = None,
) -> NasSolution:
    total_score = float(sum(candidate.score for candidate in selected))
    total_memory = float(sum(candidate.param_memory + batch_size * candidate.kv_cache_memory for candidate in selected))
    total_runtime = float(sum(candidate.runtimes[batch_size] for candidate in selected))
    throughput = float("inf") if total_runtime == 0 else batch_size * constraints.seq_len / total_runtime
    objective, objective_components = _solution_objective(
        selected,
        constraints,
        batch_size,
        objective_candidates=objective_candidates,
    )
    return NasSolution(
        batch_size=batch_size,
        selected=selected,
        total_score=total_score,
        total_memory=total_memory,
        total_runtime=total_runtime,
        throughput=throughput,
        objective=objective,
        objective_components=objective_components,
    )


def _is_feasible(solution: NasSolution, constraints: SearchConstraints) -> bool:
    memory_cap = _memory_cap(constraints, solution.batch_size)
    if memory_cap is not None and solution.total_memory > memory_cap + 1e-8:
        return False
    latency_cap = _latency_cap(constraints, solution.batch_size)
    if latency_cap is not None and solution.total_runtime > latency_cap + 1e-8:
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


def _batch_cap(caps: dict[int, float] | None, batch_size: int) -> float | None:
    if not caps:
        return None
    if batch_size in caps:
        return float(caps[batch_size])
    key = str(batch_size)
    if key in caps:  # type: ignore[operator]
        return float(caps[key])  # type: ignore[index]
    return None


def _memory_cap(constraints: SearchConstraints, batch_size: int) -> float | None:
    batch_cap = _batch_cap(constraints.memory_max_by_batch, batch_size)
    return batch_cap if batch_cap is not None else constraints.memory_max


def _latency_cap(constraints: SearchConstraints, batch_size: int) -> float | None:
    batch_cap = _batch_cap(constraints.latency_max_by_batch, batch_size)
    return batch_cap if batch_cap is not None else constraints.latency_max


def _scale(values: np.ndarray, normalize: bool) -> float:
    if not normalize:
        return 1.0
    value = float(np.max(np.abs(values))) if values.size else 0.0
    return max(value, 1e-12)


def _objective_component_values(
    candidates: Sequence[SearchCandidate],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.array([candidate.score for candidate in candidates], dtype=float)
    memory = np.array(
        [candidate.param_memory + batch_size * candidate.kv_cache_memory for candidate in candidates],
        dtype=float,
    )
    runtime = np.array([candidate.runtimes[batch_size] for candidate in candidates], dtype=float)
    return scores, memory, runtime


def _objective_coefficients(
    candidates: Sequence[SearchCandidate],
    constraints: SearchConstraints,
    batch_size: int,
) -> np.ndarray:
    scores, memory, runtime = _objective_component_values(candidates, batch_size)
    if constraints.objective_mode == "score":
        c = scores.copy()
        if constraints.score_direction == "maximize":
            c = -c
        return c

    score_sign = 1.0 if constraints.score_direction == "minimize" else -1.0
    score_term = score_sign * scores / _scale(scores, constraints.normalize_objectives)
    memory_term = memory / _scale(memory, constraints.normalize_objectives)
    runtime_term = runtime / _scale(runtime, constraints.normalize_objectives)
    return (
        constraints.score_weight * score_term
        + constraints.memory_weight * memory_term
        + constraints.runtime_weight * runtime_term
    )


def _solution_objective(
    selected: Sequence[SearchCandidate],
    constraints: SearchConstraints,
    batch_size: int,
    objective_candidates: Sequence[SearchCandidate] | None = None,
) -> tuple[float, dict[str, float]]:
    scores, memory, runtime = _objective_component_values(selected, batch_size)
    total_score = float(np.sum(scores))
    total_memory = float(np.sum(memory))
    total_runtime = float(np.sum(runtime))
    if constraints.objective_mode == "score":
        objective = total_score if constraints.score_direction == "minimize" else -total_score
        return objective, {
            "score": total_score,
            "memory": total_memory,
            "runtime": total_runtime,
            "weighted_score": objective,
        }

    reference = selected if objective_candidates is None else objective_candidates
    reference_scores, reference_memory, reference_runtime = _objective_component_values(reference, batch_size)
    score_sign = 1.0 if constraints.score_direction == "minimize" else -1.0
    score_term = score_sign * scores / _scale(reference_scores, constraints.normalize_objectives)
    memory_term = memory / _scale(reference_memory, constraints.normalize_objectives)
    runtime_term = runtime / _scale(reference_runtime, constraints.normalize_objectives)
    coeffs = (
        constraints.score_weight * score_term
        + constraints.memory_weight * memory_term
        + constraints.runtime_weight * runtime_term
    )
    weighted_score = float(np.sum(coeffs))
    return weighted_score, {
        "score": total_score,
        "memory": total_memory,
        "runtime": total_runtime,
        "weighted_score": weighted_score,
        "score_weight": float(constraints.score_weight),
        "memory_weight": float(constraints.memory_weight),
        "runtime_weight": float(constraints.runtime_weight),
        "normalized": float(1.0 if constraints.normalize_objectives else 0.0),
    }
