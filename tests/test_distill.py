from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from distill_nas_core.blocks import make_parent_block
from distill_nas_core.distill import (
    extract_distill_target,
    forward_batch,
    global_knowledge_distillation,
    local_distill_block,
    logits_kl_loss,
    move_batch_to_device,
    sampled_action_reverse_kl_loss,
    sampled_reverse_kl_loss,
    sample_on_policy_sequences,
)
from distill_nas_core.toy import TinyCausalLM, TinyConfig


class DistillationTest(unittest.TestCase):
    def test_kl_loss_is_token_average(self) -> None:
        teacher = torch.tensor([[[2.0, -1.0, 0.5]]])
        student = torch.tensor([[[1.0, 0.0, 0.25]]])

        single_token = logits_kl_loss(teacher, student)
        repeated_tokens = logits_kl_loss(teacher.repeat(1, 4, 1), student.repeat(1, 4, 1))

        self.assertTrue(torch.allclose(single_token, repeated_tokens))

    def test_sampled_reverse_kl_matches_selected_token_log_probs(self) -> None:
        teacher = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]],
                [[-0.25, 0.75, 0.0], [1.5, -0.5, 0.25]],
            ]
        )
        student = torch.tensor(
            [
                [[1.0, 0.25, -0.5], [0.0, 1.5, -0.25]],
                [[0.5, 0.25, -0.5], [1.0, -0.25, 0.5]],
            ]
        )
        target_ids = torch.tensor([[0, 1], [2, 0]])
        token_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

        loss = sampled_reverse_kl_loss(teacher, student, target_ids, token_mask=token_mask)

        teacher_log_probs = torch.log_softmax(teacher, dim=-1)
        student_log_probs = torch.log_softmax(student, dim=-1)
        selected_teacher = teacher_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        selected_student = student_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        expected = ((selected_student - selected_teacher) * token_mask).sum() / token_mask.sum()
        self.assertTrue(torch.allclose(loss, expected))

    def test_sampled_action_reverse_kl_accepts_batched_action_logits(self) -> None:
        torch.manual_seed(29)
        teacher = torch.randn(2, 3, 5)
        student = torch.randn(2, 3, 5)

        loss = sampled_action_reverse_kl_loss(teacher, student)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_distill_target_prefers_actions_over_language_logits(self) -> None:
        output = SimpleNamespace(
            logits=torch.randn(2, 4, 10),
            actions=torch.randn(2, 7),
        )

        target = extract_distill_target(output)

        self.assertEqual(target.name, "actions")
        self.assertEqual(target.metric, "mse")

    def test_sample_on_policy_sequences_appends_tokens(self) -> None:
        torch.manual_seed(11)
        model = TinyCausalLM(TinyConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=4, intermediate_size=32))
        prompts = torch.randint(0, 16, (2, 3))

        sampled = sample_on_policy_sequences(model, prompts, max_new_tokens=2, top_k=4)

        self.assertEqual(sampled.shape, (2, 5))
        self.assertTrue(torch.equal(sampled[:, :3], prompts))

    def test_token_opd_accepts_dict_logits_outputs(self) -> None:
        class DictLogitsCausalLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(20, 8)
                self.head = nn.Linear(8, 20)

            def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **_: object):
                hidden = self.embedding(input_ids)
                return {
                    "logits": self.head(hidden),
                    "hidden_states": (hidden,) if output_hidden_states else (),
                }

        torch.manual_seed(13)
        teacher = DictLogitsCausalLM()
        student = DictLogitsCausalLM()
        batch = {"input_ids": torch.randint(0, 20, (2, 3))}

        losses = global_knowledge_distillation(
            teacher,
            student,
            [batch],
            steps=1,
            lr=1e-4,
            opd_weight=0.25,
            opd_max_new_tokens=2,
            opd_top_k=8,
        )

        self.assertEqual(len(losses), 1)

    def test_token_opd_misconfiguration_raises(self) -> None:
        class FeatureLogitsModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(3, 12)

            def forward(self, features: torch.Tensor, output_hidden_states: bool = False, **_: object):
                hidden = self.proj(features).view(features.shape[0], 2, 6)
                return SimpleNamespace(
                    logits=hidden,
                    hidden_states=(hidden,) if output_hidden_states else (),
                )

        teacher = FeatureLogitsModel()
        student = FeatureLogitsModel()
        batch = {"features": torch.randn(2, 3)}

        with self.assertRaisesRegex(ValueError, "Token OPD requires input_ids"):
            global_knowledge_distillation(
                teacher,
                student,
                [batch],
                steps=1,
                lr=1e-4,
                opd_weight=0.25,
                opd_max_new_tokens=2,
            )

    def test_forward_batch_accepts_dict_batches(self) -> None:
        model = TinyCausalLM(TinyConfig(vocab_size=16, hidden_size=16, num_layers=1, num_heads=4, intermediate_size=32))
        batch = {"input_ids": torch.randint(0, 16, (2, 3)), "unused": "kept"}
        moved = move_batch_to_device(batch, torch.device("cpu"))
        moved.pop("unused")

        output = forward_batch(model, moved)

        self.assertEqual(output.logits.shape[:2], (2, 3))

    def test_local_distill_restores_training_and_grad_state(self) -> None:
        parent = make_parent_block(hidden_size=16, num_heads=4, intermediate_size=32)
        child = make_parent_block(hidden_size=16, num_heads=4, intermediate_size=32)
        parent.train()
        child.eval()
        frozen_parameter = next(child.parameters())
        frozen_parameter.requires_grad_(False)

        parent_training = parent.training
        child_training = child.training
        parent_requires_grad = [parameter.requires_grad for parameter in parent.parameters()]
        child_requires_grad = [parameter.requires_grad for parameter in child.parameters()]

        hidden_batches = [torch.randn(2, 4, 16)]
        losses = local_distill_block(parent, child, hidden_batches, steps=1, lr=1e-3)

        self.assertEqual(len(losses), 1)
        self.assertEqual(parent.training, parent_training)
        self.assertEqual(child.training, child_training)
        self.assertEqual([parameter.requires_grad for parameter in parent.parameters()], parent_requires_grad)
        self.assertEqual([parameter.requires_grad for parameter in child.parameters()], child_requires_grad)

    def test_global_distillation_supports_opd_and_restores_state(self) -> None:
        torch.manual_seed(17)
        config = TinyConfig(vocab_size=24, hidden_size=16, num_layers=1, num_heads=4, intermediate_size=32, max_seq_len=8)
        teacher = TinyCausalLM(config)
        student = TinyCausalLM(config)
        teacher.train()
        student.eval()
        frozen_parameter = next(student.parameters())
        frozen_parameter.requires_grad_(False)

        teacher_training = teacher.training
        student_training = student.training
        teacher_requires_grad = [parameter.requires_grad for parameter in teacher.parameters()]
        student_requires_grad = [parameter.requires_grad for parameter in student.parameters()]

        prompts = [torch.randint(0, config.vocab_size, (2, 3))]
        losses = global_knowledge_distillation(
            teacher,
            student,
            prompts,
            steps=1,
            lr=1e-4,
            opd_weight=0.25,
            opd_max_new_tokens=2,
            opd_top_k=8,
        )

        self.assertEqual(len(losses), 1)
        self.assertEqual(teacher.training, teacher_training)
        self.assertEqual(student.training, student_training)
        self.assertEqual([parameter.requires_grad for parameter in teacher.parameters()], teacher_requires_grad)
        self.assertEqual([parameter.requires_grad for parameter in student.parameters()], student_requires_grad)

    def test_opd_keeps_dict_batch_side_inputs(self) -> None:
        class SideInputCausalLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(20, 8)
                self.head = nn.Linear(8, 20)

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                pixel_values: torch.Tensor | None = None,
                output_hidden_states: bool = False,
                **_: object,
            ):
                if pixel_values is None:
                    raise RuntimeError("pixel_values must be preserved for OPD")
                if attention_mask is None or attention_mask.shape != input_ids.shape:
                    raise RuntimeError("attention_mask must track sampled input_ids")
                side = pixel_values.reshape(input_ids.shape[0], -1).mean(dim=1).view(-1, 1, 1)
                hidden = self.embedding(input_ids) + side
                return SimpleNamespace(
                    logits=self.head(hidden),
                    hidden_states=(hidden,) if output_hidden_states else (),
                )

        torch.manual_seed(23)
        teacher = SideInputCausalLM()
        student = SideInputCausalLM()
        input_ids = torch.randint(0, 20, (2, 3))
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.randn(2, 3, 4, 4),
        }

        losses = global_knowledge_distillation(
            teacher,
            student,
            [batch],
            steps=1,
            lr=1e-4,
            opd_weight=0.25,
            opd_max_new_tokens=2,
            opd_top_k=8,
        )

        self.assertEqual(len(losses), 1)

    def test_global_distillation_supports_continuous_action_opd(self) -> None:
        class ContinuousActionPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(3, 2)

            def forward(self, proprio: torch.Tensor, output_hidden_states: bool = False, **_: object):
                actions = self.proj(proprio)
                hidden_states = (actions.unsqueeze(1),) if output_hidden_states else ()
                return SimpleNamespace(actions=actions, hidden_states=hidden_states)

        torch.manual_seed(31)
        teacher = ContinuousActionPolicy()
        student = ContinuousActionPolicy()
        batch = {"proprio": torch.randn(4, 3)}

        losses = global_knowledge_distillation(
            teacher,
            student,
            [batch],
            steps=1,
            lr=1e-4,
            opd_weight=0.5,
            opd_max_new_tokens=0,
        )

        self.assertEqual(len(losses), 1)

    def test_global_distillation_supports_gaussian_action_opd(self) -> None:
        class GaussianActionPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mean = nn.Linear(3, 2)
                self.log_std = nn.Parameter(torch.zeros(2))

            def forward(self, proprio: torch.Tensor, output_hidden_states: bool = False, **_: object):
                action_mean = self.mean(proprio)
                action_log_std = self.log_std.view(1, -1).expand_as(action_mean)
                hidden_states = (action_mean.unsqueeze(1),) if output_hidden_states else ()
                return SimpleNamespace(
                    action_mean=action_mean,
                    action_log_std=action_log_std,
                    hidden_states=hidden_states,
                )

        torch.manual_seed(41)
        teacher = GaussianActionPolicy()
        student = GaussianActionPolicy()
        batch = {"proprio": torch.randn(4, 3)}

        losses = global_knowledge_distillation(
            teacher,
            student,
            [batch],
            steps=1,
            lr=1e-4,
            opd_weight=0.5,
            strict_action_opd=True,
        )

        self.assertEqual(len(losses), 1)

    def test_forward_batch_filters_vla_auxiliary_keys_when_unsupported(self) -> None:
        class TextOnlyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(20, 8)
                self.head = nn.Linear(8, 20)

            def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(
                    logits=self.head(hidden),
                    hidden_states=(hidden,) if output_hidden_states else (),
                )

        model = TextOnlyModel()
        batch = {
            "input_ids": torch.randint(0, 20, (2, 3)),
            "actions": torch.randn(2, 4),
            "proprio": torch.randn(2, 7),
        }

        output = forward_batch(model, batch, output_hidden_states=True)

        self.assertEqual(output.logits.shape[:2], (2, 3))

    def test_global_distillation_accepts_tensor_hidden_states(self) -> None:
        class TensorHiddenCausalLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(20, 8)
                self.head = nn.Linear(8, 20)

            def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **_: object):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(
                    logits=self.head(hidden),
                    hidden_states=hidden if output_hidden_states else None,
                )

        torch.manual_seed(37)
        teacher = TensorHiddenCausalLM()
        student = TensorHiddenCausalLM()
        batch = {"input_ids": torch.randint(0, 20, (2, 3))}

        losses = global_knowledge_distillation(
            teacher,
            student,
            [batch],
            steps=1,
            lr=1e-4,
        )

        self.assertEqual(len(losses), 1)


if __name__ == "__main__":
    unittest.main()
