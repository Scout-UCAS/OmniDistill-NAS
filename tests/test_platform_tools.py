from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from distill_nas_core.data_adapters import load_examples
from distill_nas_core.evaluation import evaluate_artifact
from distill_nas_core.experiment import build_stage_plan, run_experiment
from distill_nas_core.export import export_artifact, load_export_manifest
from distill_nas_core.profiler import profile_artifact
from distill_nas_core.reporting import write_workflow_report
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

    def test_data_adapter_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.json"
            path.write_text(json.dumps([{"prompt": "Hello?", "answer": "World"}]), encoding="utf-8")
            examples = load_examples("json", {"path": str(path)})
        self.assertEqual(examples[0].prompt, "Hello?")
        self.assertEqual(examples[0].target, "World")

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
