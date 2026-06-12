from __future__ import annotations

import unittest

from distill_nas_core.mip import SearchCandidate, SearchConstraints, solve_nas_mip


def candidate(layer: int, name: str, score: float, memory: float, runtime: float) -> SearchCandidate:
    return SearchCandidate(
        layer_idx=layer,
        name=f"L{layer}:{name}",
        score=score,
        param_memory=memory,
        kv_cache_memory=0.0,
        runtimes={1: runtime, 2: runtime * 1.8},
    )


class MipSearchTest(unittest.TestCase):
    def test_selects_one_candidate_per_layer_under_constraints(self) -> None:
        candidates = [
            [candidate(0, "accurate", 0.1, 10, 10), candidate(0, "fast", 0.4, 4, 3)],
            [candidate(1, "accurate", 0.1, 10, 10), candidate(1, "fast", 0.5, 4, 3)],
        ]
        constraints = SearchConstraints(
            seq_len=16,
            batch_sizes=[1],
            memory_max=14,
            latency_max=13,
            score_direction="minimize",
        )
        solution = solve_nas_mip(candidates, constraints)
        self.assertEqual(len(solution.selected), 2)
        self.assertLessEqual(solution.total_memory, 14)
        self.assertLessEqual(solution.total_runtime, 13)
        self.assertEqual(solution.selected_names, ["L0:fast", "L1:accurate"])

    def test_maximize_score(self) -> None:
        candidates = [
            [candidate(0, "small", 1.0, 1, 1), candidate(0, "large", 5.0, 10, 1)],
            [candidate(1, "small", 1.0, 1, 1), candidate(1, "large", 5.0, 10, 1)],
        ]
        constraints = SearchConstraints(
            seq_len=16,
            batch_sizes=[1],
            memory_max=11,
            score_direction="maximize",
        )
        solution = solve_nas_mip(candidates, constraints)
        self.assertEqual(solution.total_score, 6.0)

    def test_weighted_multi_objective_can_prefer_resource_savings(self) -> None:
        candidates = [
            [candidate(0, "accurate", 0.1, 100, 100), candidate(0, "small", 0.5, 1, 1)],
            [candidate(1, "accurate", 0.1, 100, 100), candidate(1, "small", 0.5, 1, 1)],
        ]
        constraints = SearchConstraints(
            seq_len=16,
            batch_sizes=[1],
            objective_mode="weighted",
            score_weight=0.0,
            memory_weight=1.0,
            runtime_weight=1.0,
        )
        solution = solve_nas_mip(candidates, constraints)

        self.assertEqual(solution.selected_names, ["L0:small", "L1:small"])
        self.assertEqual(solution.objective_components["memory_weight"], 1.0)
        self.assertEqual(solution.objective_components["runtime_weight"], 1.0)

    def test_weighted_multi_objective_keeps_score_tradeoff_when_weighted(self) -> None:
        candidates = [
            [candidate(0, "accurate", 0.1, 100, 100), candidate(0, "small", 0.5, 1, 1)],
            [candidate(1, "accurate", 0.1, 100, 100), candidate(1, "small", 0.5, 1, 1)],
        ]
        constraints = SearchConstraints(
            seq_len=16,
            batch_sizes=[1],
            objective_mode="weighted",
            score_weight=1.0,
            memory_weight=0.0,
            runtime_weight=0.0,
        )
        solution = solve_nas_mip(candidates, constraints)

        self.assertEqual(solution.selected_names, ["L0:accurate", "L1:accurate"])

    def test_batch_specific_caps_are_applied_per_candidate_batch(self) -> None:
        candidates = [
            [
                SearchCandidate(
                    layer_idx=0,
                    name="L0:only",
                    score=0.1,
                    param_memory=0.0,
                    kv_cache_memory=10.0,
                    runtimes={1: 10.0, 2: 20.0},
                )
            ]
        ]
        constraints = SearchConstraints(
            seq_len=16,
            batch_sizes=[1, 2],
            memory_max_by_batch={1: 5.0, 2: 20.0},
            latency_max_by_batch={1: 5.0, 2: 20.0},
        )
        solution = solve_nas_mip(candidates, constraints)

        self.assertEqual(solution.batch_size, 2)
        self.assertEqual(solution.total_memory, 20.0)
        self.assertEqual(solution.total_runtime, 20.0)


if __name__ == "__main__":
    unittest.main()
