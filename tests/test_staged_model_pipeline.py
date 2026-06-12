from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import tools.run_staged_model_pipeline as staged


class FakeCandidateLayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        variant: str,
        config=None,
        layer_idx: int | None = None,
        fla_mode: str = "chunk",
        fla_feature_map: str = "elu",
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.variant = variant
        self.qwen_attn = nn.Linear(1, 1, bias=False)
        self.fla_attn = None

    def forward(self, hidden_states: torch.Tensor, **_: object):
        return hidden_states, None, None


class TrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_state_dict = False

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.loaded_state_dict = True
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class StagedModelPipelineTest(unittest.TestCase):
    def test_gkd_restores_assembled_replacement_weights(self) -> None:
        assembled = {
            "bld_pth": "bld.pth",
            "architecture_config": {
                "rank": 0,
                "selected": [{"layer_idx": 0, "variant": "mha_attn", "name": "L0:mha_attn"}],
            },
            "selected_replacements": [
                {
                    "layer_idx": 0,
                    "variant": "mha_attn",
                    "replacement_attr": "qwen_attn",
                    "replacement_state_dict": {"weight": torch.tensor([[3.0]])},
                }
            ],
            "model_state_dict": {},
        }
        bld = {
            "records": [
                {
                    "layer_idx": 0,
                    "variant": "mha_attn",
                    "replacement_attr": "qwen_attn",
                    "replacement_state_dict": {"weight": torch.tensor([[0.0]])},
                }
            ],
            "fla": {"mode": "chunk", "feature_map": "elu"},
        }
        teacher_loaded = SimpleNamespace(
            model=nn.Module(),
            layers=nn.ModuleList([nn.Identity()]),
            language_config=SimpleNamespace(),
            model_kind="text",
        )
        student_loaded = SimpleNamespace(
            model=TrackingModel(),
            layers=nn.ModuleList([nn.Identity()]),
            language_config=SimpleNamespace(),
            model_kind="text",
        )
        observed: dict[str, float] = {}

        def fake_load_pth(path):
            return assembled if Path(path).name == "assembled.pth" else bld

        def fake_gkd(_teacher, _student, _batches, **_kwargs):
            layer = student_loaded.layers[0]
            observed["weight"] = float(layer.qwen_attn.weight.detach().cpu().item())
            return [0.0]

        args = SimpleNamespace(
            assembled_pth="assembled.pth",
            gkd_steps=1,
            lr=1e-4,
            include_lm_loss=False,
            opd_weight=0.0,
            opd_max_new_tokens=0,
            opd_temperature=None,
            opd_top_k=None,
            teacher_device=None,
            student_device=None,
            strict_action_opd=False,
            allow_partial_checkpoint_load=False,
            save_full_state_dict=False,
            output_pth="unused.pth",
            summary_json="unused.json",
            model_id="fake/model",
        )

        with (
            patch.object(staged, "QwenCandidateLayer", FakeCandidateLayer),
            patch.object(staged, "load_pth", side_effect=fake_load_pth),
            patch.object(
                staged,
                "make_context",
                side_effect=[
                    (teacher_loaded, [], {}, torch.device("cpu"), torch.float32, Path(".")),
                    (student_loaded, [], {}, torch.device("cpu"), torch.float32, Path(".")),
                ],
            ),
            patch.object(staged, "global_knowledge_distillation", side_effect=fake_gkd),
            patch.object(staged, "save_pth", return_value=Path("unused.pth")),
            patch.object(staged, "write_json", return_value=Path("unused.json")),
            patch("builtins.print"),
        ):
            staged.command_gkd(args)

        self.assertEqual(observed["weight"], 3.0)
        self.assertTrue(student_loaded.model.loaded_state_dict)


if __name__ == "__main__":
    unittest.main()
