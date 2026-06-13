from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from distill_nas_core.benchmarks import load_benchmark_suite, run_benchmark_suite
from distill_nas_core.cli import main as cli_main
from distill_nas_core.data_adapters import load_examples
from distill_nas_core.evaluation import evaluate_artifact
from distill_nas_core.experiment import build_stage_plan, run_experiment
from distill_nas_core.export import export_artifact, load_export_manifest
from distill_nas_core.plugins import list_plugins, register_plugin
from distill_nas_core.profiler import profile_artifact
from distill_nas_core.reporting import write_workflow_report
from distill_nas_core.result_zoo import load_result_manifests, write_result_index
from distill_nas_core.schema import validate_benchmark_suite, validate_experiment_spec, validate_result_manifest
from distill_nas_core.search_space import (
    AttentionSpec,
    attention_spec_from_name,
    make_attention_variant,
    register_attention_variant,
)
from distill_nas_core.toy import TinyCausalLM, TinyConfig


def write_toy_artifact(path: Path) -> None:
    config = TinyConfig(vocab_size=24, hidden_size=16, num_layers=1, num_heads=4, intermediate_size=32, max_seq_len=16)
    model = TinyCausalLM(config)
    torch.save(
        {
            "stage": "gkd_model",
            "backend": "toy",
            "config": config.__dict__,
            "seq_len": 8,
            "batch_size": 2,
            "architecture_config": {"rank": 0, "selected": []},
            "losses": [1.0, 0.5],
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path,
    )


class PlatformToolsTest(unittest.TestCase):
    def test_evaluate_profile_export_and_report_toy_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "gkd_model.pth"
            write_toy_artifact(artifact)

            metrics = evaluate_artifact(artifact, backend="toy", device="cpu", num_batches=1)
            self.assertIn("perplexity", metrics)

            profile = profile_artifact(artifact, backend="toy", device="cpu", batch_sizes=[1], warmup=0, steps=1)
            self.assertIn("profiles", profile)

            manifest = export_artifact(artifact, root / "export")
            self.assertEqual(manifest["stage"], "gkd_model")
            self.assertEqual(load_export_manifest(root / "export")["format"], "omnidistill-nas-export")

            workflow = root / "workflow"
            (workflow / "09_evaluation").mkdir(parents=True)
            (workflow / "09_evaluation" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            report = write_workflow_report(workflow)
            self.assertTrue(report.exists())

    def test_experiment_plan_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = {"backend": "toy", "output_dir": str(Path(tmp) / "workflow"), "stages": ["bld", "evaluate"]}
            plan = build_stage_plan(spec)
            self.assertEqual([stage.name for stage in plan], ["bld", "evaluate"])
            results = run_experiment(spec, dry_run=True)
            self.assertEqual([item["status"] for item in results], ["dry_run", "dry_run"])

    def test_experiment_relative_outputs_use_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            spec = {"backend": "toy", "output_dir": "workflow", "stages": ["bld"]}
            plan = build_stage_plan(spec, workdir=workspace)

            self.assertEqual(plan[0].env["WORKFLOW_WORKDIR"], str(workspace))
            self.assertEqual(plan[0].env["WORKFLOW_OUTPUT_DIR"], str(workspace / "workflow"))
            self.assertEqual(plan[0].outputs[0], workspace / "workflow" / "04_bld_block_library" / "block_library.pth")
            self.assertTrue(Path(plan[0].command[1]).is_absolute())

            results = run_experiment(spec, dry_run=True, workdir=workspace)
            self.assertEqual(results[0]["outputs"][0], str(workspace / "workflow" / "04_bld_block_library" / "block_library.pth"))

    def test_data_adapter_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.json"
            path.write_text(json.dumps([{"prompt": "Hello?", "answer": "World"}]), encoding="utf-8")
            examples = load_examples("json", {"path": str(path)})
        self.assertEqual(examples[0].prompt, "Hello?")
        self.assertEqual(examples[0].target, "World")

    def test_data_adapter_preserves_falsey_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.json"
            path.write_text(
                json.dumps([{"text": "Classify this", "label": 0, "action": False, "state": []}]),
                encoding="utf-8",
            )
            examples = load_examples("json", {"path": str(path)})

        self.assertEqual(examples[0].target, 0)
        self.assertIs(examples[0].action, False)
        self.assertEqual(examples[0].state, [])

    def test_cli_run_dry_run_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config = workspace / "experiment.json"
            config.write_text(
                json.dumps({"backend": "toy", "output_dir": "workflow", "stages": ["bld"]}),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["run", "--config", str(config), "--workdir", str(workspace), "--dry-run"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload[0]["status"], "dry_run")
        self.assertIn(str(workspace / "workflow"), payload[0]["outputs"][0])

    def test_pyproject_declares_console_entrypoints_and_license(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")

        self.assertIn('license = "MIT"', text)
        self.assertIn('omnidistill = "distill_nas_core.cli:main"', text)
        self.assertIn('readme = "README.md"', text)

    def test_public_schema_validators_accept_project_manifests(self) -> None:
        root = Path(__file__).resolve().parents[1]
        experiment = json.loads((root / "configs" / "toy_experiment.json").read_text(encoding="utf-8"))
        benchmark = json.loads((root / "benchmarks" / "suites" / "toy_smoke.json").read_text(encoding="utf-8"))
        result = json.loads((root / "results" / "toy_smoke" / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(validate_experiment_spec(experiment), [])
        self.assertEqual(validate_benchmark_suite(benchmark), [])
        self.assertEqual(validate_result_manifest(result), [])

    def test_benchmark_suite_dry_run_and_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        suite = root / "benchmarks" / "suites" / "toy_smoke.json"
        loaded = load_benchmark_suite(suite)
        self.assertEqual(loaded["name"], "toy-smoke")

        with tempfile.TemporaryDirectory() as tmp:
            payload = run_benchmark_suite(suite, result_dir=Path(tmp) / "runs", dry_run=True, workdir=tmp)
            self.assertTrue((Path(tmp) / "runs" / "benchmark_results.json").exists())

        self.assertEqual(payload["benchmarks"][0]["status"], "dry_run")

        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(stdout):
            exit_code = cli_main(
                [
                    "benchmark",
                    "--suite",
                    str(suite),
                    "--dry-run",
                    "--result-dir",
                    str(Path(tmp) / "runs"),
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("toy_full_workflow", stdout.getvalue())

    def test_result_zoo_report_and_tracking(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = load_result_manifests(root / "results")
        self.assertEqual(manifests[0]["id"], "toy_smoke")

        with tempfile.TemporaryDirectory() as tmp:
            report = write_result_index(root / "results", Path(tmp) / "results.md")
            self.assertIn("Toy full workflow smoke", report.read_text(encoding="utf-8"))

            tracking_path = Path(tmp) / "events.jsonl"
            old_provider = os.environ.get("OMNIDISTILL_TRACKING")
            old_file = os.environ.get("OMNIDISTILL_TRACKING_FILE")
            os.environ["OMNIDISTILL_TRACKING"] = "jsonl"
            os.environ["OMNIDISTILL_TRACKING_FILE"] = str(tracking_path)
            try:
                run_benchmark_suite(root / "benchmarks" / "suites" / "toy_smoke.json", result_dir=Path(tmp) / "runs", dry_run=True)
            finally:
                if old_provider is None:
                    os.environ.pop("OMNIDISTILL_TRACKING", None)
                else:
                    os.environ["OMNIDISTILL_TRACKING"] = old_provider
                if old_file is None:
                    os.environ.pop("OMNIDISTILL_TRACKING_FILE", None)
                else:
                    os.environ["OMNIDISTILL_TRACKING_FILE"] = old_file
            self.assertTrue(tracking_path.exists())

    def test_plugin_registry(self) -> None:
        register_plugin("unit-test-evaluator", "evaluator", "Unit test evaluator.")
        names = {item["name"] for item in list_plugins("evaluator")}
        self.assertIn("unit-test-evaluator", names)

    def test_attention_variant_plugin_registration(self) -> None:
        def spec_factory(name: str, _parent_num_heads: int) -> AttentionSpec:
            return AttentionSpec(name, "plugin_noop", None)

        def module_factory(parent_attention, _spec):
            return torch.nn.Identity()

        register_attention_variant("plugin_noop_attn", spec_factory, module_factory, aliases=["all_plugin_test"])
        spec = attention_spec_from_name("plugin_noop_attn", parent_num_heads=4)
        module = make_attention_variant(torch.nn.Identity(), spec)  # type: ignore[arg-type]
        self.assertIsInstance(module, torch.nn.Identity)

    def test_default_script_generates_multi_objective_before_report(self) -> None:
        run_all = Path(__file__).resolve().parents[1] / "scripts" / "run_all.sh"
        text = run_all.read_text(encoding="utf-8")
        self.assertLess(
            text.index("12_multi_objective_search.sh"),
            text.index("11_export_and_report.sh"),
        )


if __name__ == "__main__":
    unittest.main()
