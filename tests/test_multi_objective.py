from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from distill_nas_core.mip import SearchCandidate, SearchConstraints
from distill_nas_core.multi_objective import (
    enumerate_feasible_solutions,
    generate_weight_grid,
    pareto_front,
    parse_weight_grid,
    run_weight_sweep,
    solution_to_config,
    write_multi_objective_report,
    write_pareto_configs,
)
from tools.run_multi_objective_search import build_constraints


def candidate(layer: int, name: str, score: float, memory: float, runtime: float) -> SearchCandidate:
    return SearchCandidate(
        layer_idx=layer,
        name=f"L{layer}:{name}",
        score=score,
        param_memory=memory,
        kv_cache_memory=0.0,
        runtimes={1: runtime},
        payload={"variant": name},
    )


class MultiObjectiveSearchTest(unittest.TestCase):
    def test_weight_grid_parsing(self) -> None:
        self.assertIn({"score": 1.0, "memory": 0.0, "runtime": 0.0}, generate_weight_grid(2))
        parsed = parse_weight_grid("score=1,memory=.25,runtime=.5;0,1,0")
        self.assertEqual(parsed[0], {"score": 1.0, "memory": 0.25, "runtime": 0.5})
        self.assertEqual(parsed[1], {"score": 0.0, "memory": 1.0, "runtime": 0.0})

    def test_exact_pareto_and_sweep(self) -> None:
        candidates = [
            [candidate(0, "accurate", 0.1, 10, 10), candidate(0, "small", 0.4, 1, 1)],
            [candidate(1, "accurate", 0.1, 10, 10), candidate(1, "small", 0.4, 1, 1)],
        ]
        constraints = SearchConstraints(seq_len=16, batch_sizes=[1], objective_mode="weighted")
        exact = enumerate_feasible_solutions(candidates, constraints, max_combinations=10)
        front = pareto_front(exact)
        names = [solution.selected_names for solution in front]
        self.assertIn(["L0:accurate", "L1:accurate"], names)
        self.assertIn(["L0:small", "L1:small"], names)

        sweep = run_weight_sweep(
            candidates,
            constraints,
            [{"score": 1.0, "memory": 0.0, "runtime": 0.0}, {"score": 0.0, "memory": 1.0, "runtime": 1.0}],
        )
        self.assertEqual(len(sweep), 2)
        config = solution_to_config(sweep[0]["solution"], rank=0)
        self.assertIn("selected", config)

    def test_report_writer(self) -> None:
        candidates = [[candidate(0, "only", 0.1, 1, 1)]]
        constraints = SearchConstraints(seq_len=16, batch_sizes=[1])
        solution = enumerate_feasible_solutions(candidates, constraints)[0]
        config = solution_to_config(solution, rank=0)
        payload = {
            "scores_json": "scores.json",
            "pareto_source": "exact",
            "sweep_solutions": [config],
            "pareto_front": [config],
            "plot_svg": "<svg></svg>",
        }
        with tempfile.TemporaryDirectory() as tmp:
            report, plot = write_multi_objective_report(
                payload,
                Path(tmp) / "report.md",
                Path(tmp) / "plot.svg",
            )
            self.assertTrue(report.exists())
            self.assertTrue(plot.exists())

    def test_multi_objective_constraints_are_batch_specific(self) -> None:
        candidates = [
            [
                SearchCandidate(
                    layer_idx=0,
                    name="L0:parent",
                    score=0.0,
                    param_memory=10.0,
                    kv_cache_memory=5.0,
                    runtimes={1: 10.0, 2: 30.0},
                    payload={"variant": "parent"},
                )
            ]
        ]
        constraints = build_constraints(
            {"seq_len": 16},
            candidates,
            [1, 2],
            memory_fraction=0.5,
            runtime_fraction=0.5,
            score_direction="minimize",
            normalize_objectives=True,
        )

        self.assertEqual(constraints.memory_max_by_batch, {1: 7.5, 2: 10.0})
        self.assertEqual(constraints.latency_max_by_batch, {1: 5.0, 2: 15.0})

    def test_pareto_config_writer_removes_stale_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "pareto_rank_99.json").write_text("{}", encoding="utf-8")
            write_pareto_configs([{"rank": 0, "selected": []}], target)

            self.assertTrue((target / "pareto_rank_00.json").exists())
            self.assertFalse((target / "pareto_rank_99.json").exists())


if __name__ == "__main__":
    unittest.main()
