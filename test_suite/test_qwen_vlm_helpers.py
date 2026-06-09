from __future__ import annotations

import types
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from PIL import Image

from scripts.run_qwen3_attention_search import (
    COMMON_DATASET_ALIASES,
    DatasetSpec,
    ScoreTarget,
    built_in_examples,
    encode_vlm_prompt,
    extract_score_target,
    find_decoder_layers,
    format_dataset_example,
    load_dataset_examples,
    load_multimodal_image,
    make_multimodal_batches,
    move_batch_to_device,
    normalize_model_kind,
    resolve_language_config,
    score_target_distance,
)


class FakeQwenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.num_heads = 2
        self.self_attn.num_key_value_heads = 2
        self.self_attn.head_dim = 4
        self.mlp = nn.Linear(8, 8)
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)


class FakeVlmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        text_config = types.SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            use_cache=True,
        )
        self.config = types.SimpleNamespace(
            model_type="qwen3_vl",
            vision_config=types.SimpleNamespace(),
            text_config=text_config,
        )
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([FakeQwenLayer()])


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        if kwargs.get("tokenize"):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.ones(1, 3, 4, 4),
            }
        return "rendered prompt"


class FakeActionOutput:
    def __init__(self, actions: torch.Tensor, logits: torch.Tensor | None = None) -> None:
        self.actions = actions
        self.logits = logits


class QwenVlmHelperTest(unittest.TestCase):
    def test_auto_model_kind_detects_vlm_names_and_configs(self) -> None:
        self.assertEqual(normalize_model_kind("auto", "Qwen/Qwen3-VL-2B-Instruct"), "vlm")
        config = types.SimpleNamespace(model_type="custom", vision_config=object(), text_config=object())
        self.assertEqual(normalize_model_kind("auto", "local/model", config=config), "vlm")
        self.assertEqual(normalize_model_kind("auto", "Qwen/Qwen3-0.6B"), "text")
        self.assertEqual(normalize_model_kind("auto", "openvla/openvla-7b"), "vla")
        self.assertEqual(normalize_model_kind("auto", "local/model", config=types.SimpleNamespace(model_type="openvla")), "vla")
        self.assertEqual(normalize_model_kind("vla", "local/model"), "vla")

    def test_find_decoder_layers_inside_vlm_wrapper(self) -> None:
        model = FakeVlmModel()
        layers, path = find_decoder_layers(model)

        self.assertEqual(path, "model.language_model.layers")
        self.assertIs(layers, model.model.language_model.layers)

    def test_resolve_language_config_prefers_text_config(self) -> None:
        model = FakeVlmModel()
        config = resolve_language_config(model, model.model.language_model.layers[0])

        self.assertEqual(config.hidden_size, 8)
        self.assertEqual(config.num_attention_heads, 2)

    def test_encode_vlm_prompt_uses_chat_template_outputs(self) -> None:
        processor = FakeProcessor()
        batch = encode_vlm_prompt(
            processor,
            prompt="What is in the image?",
            image=object(),
            seq_len=16,
            add_generation_prompt=True,
        )

        self.assertEqual(set(batch), {"input_ids", "attention_mask", "pixel_values"})
        self.assertEqual(processor.messages[0]["content"][0]["type"], "image")

    def test_action_outputs_are_scored_with_mse_for_vla(self) -> None:
        teacher = ScoreTarget("actions", torch.tensor([[0.0, 1.0]]), "mse")
        student = FakeActionOutput(actions=torch.tensor([[0.5, 1.5]]), logits=torch.randn(1, 2, 4))

        distance = score_target_distance(teacher, student, model_kind="vla")

        self.assertTrue(torch.allclose(distance, torch.tensor(0.25)))
        self.assertEqual(extract_score_target(student, model_kind="vla").name, "actions")

    def test_move_batch_to_device_keeps_non_tensor_values(self) -> None:
        batch = {"input_ids": torch.tensor([[1]]), "metadata": "keep"}
        moved = move_batch_to_device(batch, torch.device("cpu"))

        self.assertEqual(moved["input_ids"].device.type, "cpu")
        self.assertEqual(moved["metadata"], "keep")

    def test_dataset_formatters_cover_llm_vlm_and_vla_records(self) -> None:
        llm = format_dataset_example(
            {"question": "2 + 2?", "choices": ["3", "4"], "answer": "4"},
            DatasetSpec("unit", "llm", None, None, "test"),
            image_root=None,
            include_target=False,
        )
        vlm = format_dataset_example(
            {"question": "What color is the block?", "image": "frame.png", "answer": "red"},
            DatasetSpec("unit", "vlm", None, None, "test"),
            image_root="/tmp/images",
            include_target=False,
        )
        vla = format_dataset_example(
            {
                "language_instruction": "pick up the red block",
                "observation_image": "obs.png",
                "action": [0.1, 0.0, 0.2],
            },
            DatasetSpec("unit", "vla", None, None, "test"),
            image_root="/tmp/robot",
            include_target=True,
        )

        self.assertIn("2 + 2?", llm.prompt)
        self.assertIn("What color", vlm.prompt)
        self.assertTrue(str(vlm.image).endswith("/tmp/images/frame.png"))
        self.assertIn("vision-language-action", vla.prompt)
        self.assertIn("0.1", vla.prompt)

    def test_common_dataset_aliases_have_supported_formatter_paths(self) -> None:
        self.assertGreater(len(COMMON_DATASET_ALIASES), 0)
        synthetic = {
            "llm": {"question": "2 + 2?", "choices": ["3", "4"], "answer": "4"},
            "vlm": {"question": "What color is the block?", "image": "frame.png", "answer": "red"},
            "vla": {"instruction": "pick up the red block", "image": "frame.png", "action": [0.0, 0.1]},
        }
        for alias, (task, dataset_name, config_name, split) in COMMON_DATASET_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertIn(task, {"llm", "vlm", "vla"})
                self.assertIsInstance(split, str)
                if dataset_name is not None:
                    self.assertIsInstance(dataset_name, str)
                task_example = format_dataset_example(
                    synthetic[task],
                    DatasetSpec(alias, task, dataset_name, config_name, split),
                    image_root="/tmp/images",
                    include_target=False,
                )
                self.assertTrue(task_example.prompt.strip())

    def test_nested_and_bytes_images_are_decoded(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (3, 2), color=(255, 0, 0)).save(buffer, format="PNG")
        example = {
            "question": "What is visible?",
            "observation": {"camera": {"image": {"bytes": buffer.getvalue()}}},
        }
        task = format_dataset_example(
            example,
            DatasetSpec("unit", "vlm", None, None, "test"),
            image_root=None,
            include_target=False,
        )

        image = load_multimodal_image(task.image, size=4)

        self.assertEqual(image.size, (3, 2))

    def test_dataset_multimodal_batches_require_images_by_default(self) -> None:
        processor = FakeProcessor()
        examples = [format_dataset_example(
            {"question": "Missing image?"},
            DatasetSpec("unit", "vlm", None, None, "test"),
            image_root=None,
            include_target=False,
        )]

        with self.assertRaises(ValueError):
            make_multimodal_batches(
                processor,
                examples,
                seq_len=8,
                device=torch.device("cpu"),
                image_path=None,
                image_size=4,
                add_generation_prompt=True,
                model_kind="vlm",
                allow_blank_image=False,
            )

    def test_local_jsonl_dataset_loads_without_hf_datasets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.jsonl"
            path.write_text('{"question":"2 + 2?","choices":["3","4"],"answer":"B"}\n', encoding="utf-8")

            examples = load_dataset_examples(
                DatasetSpec("dataset", "llm", None, None, "train", local_path=str(path)),
                max_prompts=1,
                cache_dir=Path(tmpdir) / "cache",
            )

        self.assertEqual(len(examples), 1)
        self.assertIn("2 + 2?", examples[0].prompt)

    def test_built_in_examples_are_task_specific(self) -> None:
        self.assertIn("large language models", built_in_examples("text", 1)[0].prompt)
        self.assertIn("image", built_in_examples("vlm", 1)[0].prompt.lower())
        self.assertIn("Action:", built_in_examples("vla", 1)[0].prompt)


if __name__ == "__main__":
    unittest.main()
