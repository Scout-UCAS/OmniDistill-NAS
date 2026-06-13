from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from types import SimpleNamespace
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def should_use_vendor_python() -> bool:
    value = os.environ.get("DISTILL_NAS_USE_VENDOR_PYTHON", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return sys.platform.startswith("linux")


FLA_REPO_CANDIDATES = [
    ROOT / "vendor" / "flash-linear-attention-v0.4.2",
    ROOT / "vendor" / "flash-linear-attention",
]
FLA_REPO = next((path for path in FLA_REPO_CANDIDATES if path.exists()), FLA_REPO_CANDIDATES[0])
if FLA_REPO.exists() and str(FLA_REPO) not in sys.path:
    sys.path.insert(0, str(FLA_REPO))
VENDOR = ROOT / "vendor" / "python"
if should_use_vendor_python() and VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HF_CACHE = ROOT / "hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_CACHE / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_CACHE / "transformers"))
os.environ.setdefault("HF_XET_CACHE", str(DEFAULT_HF_CACHE / "xet"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import transformers.modeling_utils as transformers_modeling_utils

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForVision2Seq = None

from distill_nas_core.blocks import QuantizedLinear
from distill_nas_core.distill import forward_batch as safe_forward_batch
from distill_nas_core.mip import SearchCandidate, SearchConstraints, solve_nas_mip


PROMPTS = [
    "Explain neural architecture search for large language models in one paragraph.",
    "Write a short Python function that computes the Fibonacci sequence.",
    "Summarize why KV-cache memory matters during autoregressive decoding.",
    "Translate to Chinese: efficient inference is important for deployment.",
]
MMLU_CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
COMMON_DATASET_ALIASES = {
    "mmlu": ("llm", "cais/mmlu", "abstract_algebra", "test"),
    "mmlu_pro": ("llm", "TIGER-Lab/MMLU-Pro", None, "test"),
    "hellaswag": ("llm", "Rowan/hellaswag", None, "validation"),
    "arc_challenge": ("llm", "allenai/ai2_arc", "ARC-Challenge", "test"),
    "arc_easy": ("llm", "allenai/ai2_arc", "ARC-Easy", "test"),
    "gsm8k": ("llm", "openai/gsm8k", "main", "test"),
    "boolq": ("llm", "google/boolq", None, "validation"),
    "winogrande": ("llm", "winogrande", "winogrande_xl", "validation"),
    "truthfulqa": ("llm", "truthful_qa", "multiple_choice", "validation"),
    "vqav2": ("vlm", "HuggingFaceM4/VQAv2", None, "validation"),
    "okvqa": ("vlm", "lmms-lab/OK-VQA", None, "test"),
    "gqa": ("vlm", "lmms-lab/GQA", None, "testdev"),
    "textvqa": ("vlm", "lmms-lab/TextVQA", None, "validation"),
    "scienceqa": ("vlm", "derek-thomas/ScienceQA", None, "test"),
    "vizwiz": ("vlm", "HuggingFaceM4/VizWiz", None, "validation"),
    "coco_caption": ("vlm", "lmms-lab/COCO-Caption2017", None, "test"),
    "libero": ("vla", "physical-intelligence/libero", None, "train"),
    "lerobot_libero": ("vla", "lerobot/libero", None, "train"),
    "lerobot_pusht": ("vla", "lerobot/pusht", None, "train"),
    "aloha_transfer_cube": ("vla", "lerobot/aloha_sim_transfer_cube_human", None, "train"),
    "aloha_insertion": ("vla", "lerobot/aloha_sim_insertion_human", None, "train"),
    "bridge_v2": ("vla", None, None, "train"),
    "rt1": ("vla", None, None, "train"),
    "open_x_embodiment": ("vla", None, None, "train"),
    "droid": ("vla", None, None, "train"),
}
LLM_PROMPT_SOURCES = {
    name for name, (task, _, _, _) in COMMON_DATASET_ALIASES.items() if task == "llm"
}
VLM_PROMPT_SOURCES = {
    name for name, (task, _, _, _) in COMMON_DATASET_ALIASES.items() if task == "vlm"
}
VLA_PROMPT_SOURCES = {
    name for name, (task, _, _, _) in COMMON_DATASET_ALIASES.items() if task == "vla"
}


@dataclass
class TaskExample:
    prompt: str
    image: Any | None = None
    target: Any | None = None
    action: Any | None = None
    state: Any | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ScoreTarget:
    name: str
    tensor: torch.Tensor
    metric: str


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    task: str
    dataset_name: str | None
    config_name: str | None
    split: str
    local_path: str | None = None


def normalize_prompt_source(source: str) -> str:
    normalized = source.strip().lower().replace("-", "_")
    if normalized in {"built_in", "builtin", "default"}:
        return "built_in"
    if normalized in {"dataset", "hf_dataset", "huggingface", "local", "json", "jsonl", "csv", "parquet"}:
        return "dataset"
    if normalized in COMMON_DATASET_ALIASES:
        return normalized
    raise ValueError(f"unknown prompt source: {source}")


def value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    return True


def normalize_choices(example: dict[str, Any]) -> list[str]:
    choices = example.get("choices")
    if isinstance(choices, dict):
        if "text" in choices:
            choices = choices["text"]
        elif "label" in choices:
            choices = choices["label"]
        else:
            choices = list(choices.values())
    if choices is None:
        for target_key in ("mc1_targets", "mc2_targets"):
            target = example.get(target_key)
            if isinstance(target, dict) and target.get("choices") is not None:
                choices = target["choices"]
                break
    if choices is None and "option1" in example and "option2" in example:
        choices = [example.get("option1"), example.get("option2")]
    if choices is None:
        choices = example.get("options") or example.get("endings") or example.get("candidates") or example.get("answer_choices")
    if choices is None:
        choices = [example.get(label) for label in MMLU_CHOICE_LABELS if example.get(label) is not None]
    if choices is None:
        return []
    if isinstance(choices, str):
        choices = [choices]
    return [str(choice).strip() for choice in choices if str(choice).strip()]


def format_mmlu_prompt(example: dict[str, Any]) -> str:
    question = str(example.get("question", "")).strip()
    choices = normalize_choices(example)
    if not question:
        raise ValueError("MMLU example is missing a non-empty question")
    if len(choices) < 2:
        raise ValueError("MMLU example must contain at least two choices")
    if len(choices) > len(MMLU_CHOICE_LABELS):
        raise ValueError(f"MMLU example has too many choices: {len(choices)}")
    lines = [
        "The following are multiple choice questions. Choose the single best answer.",
        "",
        f"Question: {question}",
    ]
    for index, choice in enumerate(choices):
        lines.append(f"{MMLU_CHOICE_LABELS[index]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def get_first_value(example: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in example and value_is_present(example[name]):
            return example[name]
    return None


def first_present(*values: Any) -> Any:
    for value in values:
        if value_is_present(value):
            return value
    return None


def find_nested_value(value: Any, names: tuple[str, ...], max_depth: int = 4) -> Any:
    if max_depth < 0:
        return None
    if isinstance(value, dict):
        direct = get_first_value(value, names)
        if direct is not None:
            return direct
        for item in value.values():
            found = find_nested_value(item, names, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value[:8]:
            found = find_nested_value(item, names, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


def compact_value(value: Any, max_items: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return "<bytes>"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                parts.append("...")
                break
            parts.append(f"{key}: {compact_value(item, max_items=max_items)}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        items = [compact_value(item, max_items=max_items) for item in value[:max_items]]
        if len(value) > max_items:
            items.append("...")
        return ", ".join(items)
    return str(value)


IMAGE_FIELD_NAMES = (
    "image",
    "images",
    "img",
    "image_path",
    "image_file",
    "file_name",
    "filename",
    "frame",
    "frames",
    "observation_image",
    "rgb",
    "rgb_static",
    "rgb_gripper",
    "front_rgb",
    "wrist_rgb",
    "image_primary",
    "image_wrist",
    "bytes",
)
ACTION_FIELD_NAMES = (
    "action",
    "actions",
    "target_action",
    "action_tokens",
    "control",
    "controls",
    "robot_action",
    "action_delta",
    "delta_action",
)
STATE_FIELD_NAMES = (
    "state",
    "proprio",
    "robot_state",
    "observation_state",
    "joint_state",
    "eef_state",
)


def normalize_image_value(value: Any, image_root: str | None = None, max_depth: int = 5) -> Any | None:
    if max_depth < 0:
        return None
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), (bytes, bytearray)):
            return bytes(value["bytes"])
        for key in IMAGE_FIELD_NAMES + ("path",):
            if key in value and value[key] is not None:
                found = normalize_image_value(value[key], image_root=image_root, max_depth=max_depth - 1)
                if found is not None:
                    return found
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("image", "rgb", "camera", "frame", "observation")):
                found = normalize_image_value(item, image_root=image_root, max_depth=max_depth - 1)
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value[:8]:
            found = normalize_image_value(item, image_root=image_root, max_depth=max_depth - 1)
            if found is not None:
                return found
        return None
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_absolute():
            base = Path(image_root) if image_root else ROOT
            path = base / path
        return str(path)
    if hasattr(value, "convert"):
        return value
    return None


def format_multiple_choice_prompt(
    question: str,
    choices: list[str],
    context: str | None = None,
    prefix: str = "Choose the single best answer.",
) -> str:
    if len(choices) < 2:
        return "\n".join(part for part in [context, question, "Answer:"] if part)
    lines = [prefix, ""]
    if context:
        lines.append(f"Context: {context}")
    lines.append(f"Question: {question}")
    for index, choice in enumerate(choices):
        label = MMLU_CHOICE_LABELS[index] if index < len(MMLU_CHOICE_LABELS) else str(index + 1)
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def format_llm_prompt(example: dict[str, Any], source: str, include_target: bool = False) -> str:
    if source == "mmlu":
        prompt = format_mmlu_prompt(example)
    elif source == "hellaswag":
        context = compact_value(get_first_value(example, ("ctx", "context", "activity_label"))).strip()
        question = context or compact_value(get_first_value(example, ("query", "prompt", "question"))).strip()
        choices = normalize_choices(example)
        prompt = format_multiple_choice_prompt(question, choices)
    elif source == "winogrande":
        sentence = compact_value(get_first_value(example, ("sentence", "question", "prompt"))).strip()
        choices = normalize_choices(example)
        question = sentence.replace("_", "_____") if sentence else "Choose the option that best completes the sentence."
        prompt = format_multiple_choice_prompt(question, choices)
    elif source == "boolq":
        question = compact_value(get_first_value(example, ("question", "query"))).strip()
        context = compact_value(get_first_value(example, ("passage", "context"))).strip()
        prompt = "\n".join(part for part in [f"Passage: {context}" if context else "", f"Question: {question}", "Answer true or false:"] if part)
    elif source == "truthfulqa":
        question = compact_value(get_first_value(example, ("question", "query"))).strip()
        choices = normalize_choices(example)
        prompt = format_multiple_choice_prompt(question, choices)
    else:
        question = compact_value(
            get_first_value(
                example,
                ("question", "query", "prompt", "problem", "instruction", "input", "sentence", "text", "ctx"),
            )
        ).strip()
        context = compact_value(
            get_first_value(example, ("context", "passage", "article", "premise", "paragraph"))
        ).strip()
        choices = normalize_choices(example)
        if not question and context:
            question, context = context, ""
        if not question:
            raise ValueError("example is missing a prompt/question field")
        prompt = format_multiple_choice_prompt(question, choices, context=context or None)

    if include_target:
        target = get_first_value(example, ("answer", "label", "target", "output", "response", "solution"))
        if target is not None:
            prompt = f"{prompt} {compact_value(target)}"
    return prompt


def format_vlm_prompt(example: dict[str, Any], source: str, include_target: bool = False) -> str:
    question = compact_value(
        get_first_value(example, ("question", "query", "prompt", "instruction", "text", "caption", "hint"))
    ).strip()
    if not question:
        question = "Answer the question about the image."
    choices = normalize_choices(example)
    if choices:
        prompt = format_multiple_choice_prompt(question, choices, prefix="Answer the question about the image.")
    elif source == "coco_caption":
        prompt = "Describe the image."
    else:
        prompt = f"{question}\nAnswer:"
    if include_target:
        target = get_first_value(example, ("answer", "answers", "label", "caption", "target", "output"))
        if target is not None:
            prompt = f"{prompt} {compact_value(target)}"
    return prompt


def format_vla_prompt(example: dict[str, Any], include_target: bool = False) -> str:
    instruction_names = (
        "instruction",
        "language_instruction",
        "natural_language_instruction",
        "task",
        "prompt",
        "goal",
        "command",
    )
    instruction = compact_value(
        first_present(
            get_first_value(example, instruction_names),
            find_nested_value(example, instruction_names),
        )
    ).strip()
    if not instruction:
        instruction = "Complete the robot manipulation task shown in the observation."
    state = compact_value(
        first_present(
            get_first_value(example, STATE_FIELD_NAMES),
            find_nested_value(example, STATE_FIELD_NAMES, max_depth=3),
        )
    ).strip()
    lines = [
        "You are a vision-language-action robot policy.",
        f"Instruction: {instruction}",
    ]
    if state:
        lines.append(f"Robot state: {state}")
    lines.append("Predict the next robot action.")
    lines.append("Action:")
    if include_target:
        action = first_present(
            get_first_value(example, ACTION_FIELD_NAMES),
            find_nested_value(example, ACTION_FIELD_NAMES, max_depth=3),
        )
        if action is not None:
            lines[-1] = f"Action: {compact_value(action)}"
    return "\n".join(lines)


def task_from_model_kind(model_kind: str) -> str:
    if model_kind in {"vlm", "vla"}:
        return model_kind
    return "llm"


def infer_task_for_source(source: str, fallback_model_kind: str) -> str:
    if source in COMMON_DATASET_ALIASES:
        return COMMON_DATASET_ALIASES[source][0]
    return task_from_model_kind(fallback_model_kind)


def dataset_spec_from_args(args: argparse.Namespace, source: str, model_kind: str) -> DatasetSpec:
    if source in COMMON_DATASET_ALIASES:
        task, dataset_name, config_name, split = COMMON_DATASET_ALIASES[source]
        if source == "mmlu":
            dataset_name = args.mmlu_dataset
            config_name = args.mmlu_subject
            split = args.mmlu_split
        if args.dataset_name:
            dataset_name = args.dataset_name
        if args.dataset_path:
            dataset_name = None
        if args.dataset_config:
            config_name = args.dataset_config
        if args.dataset_split:
            split = args.dataset_split
        if dataset_name is None and not args.dataset_path:
            raise ValueError(
                f"prompt source {source!r} is a common {task.upper()} dataset family, "
                "but this repo does not hard-code a public HuggingFace id for it. "
                "Pass --dataset-name or --dataset-path."
            )
        return DatasetSpec(
            source=source,
            task=task,
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            local_path=args.dataset_path,
        )

    dataset_name = args.dataset_name
    local_path = args.dataset_path
    if local_path:
        dataset_name = None
    if not dataset_name and not local_path:
        raise ValueError("--prompt-source dataset requires --dataset-name or --dataset-path")
    task = args.dataset_task if args.dataset_task != "auto" else task_from_model_kind(model_kind)
    default_split = "train" if local_path else "validation"
    return DatasetSpec(
        source=source,
        task=task,
        dataset_name=dataset_name,
        config_name=args.dataset_config or None,
        split=args.dataset_split or default_split,
        local_path=local_path,
    )


def load_dataset_stream(spec: DatasetSpec, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    if spec.local_path:
        path = Path(spec.local_path)
        if not path.is_absolute():
            path = ROOT / path
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl", ".csv"}:
            return load_local_dataset_stream(path, split=spec.split)
        if suffix != ".parquet":
            raise ValueError(f"unsupported --dataset-path extension: {path.suffix}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace dataset sources and local Parquet files require the optional 'datasets' package. "
            "Install it into vendor/python or the active Python environment. "
            "Local JSON/JSONL/CSV files can be loaded without this dependency."
        ) from exc

    if spec.local_path:
        return load_dataset(
            "parquet",
            data_files=str(path),
            split=spec.split,
            streaming=True,
            cache_dir=str(cache_dir),
        )

    try:
        if spec.config_name:
            return load_dataset(
                spec.dataset_name,
                spec.config_name,
                split=spec.split,
                streaming=True,
                cache_dir=str(cache_dir),
            )
        return load_dataset(
            spec.dataset_name,
            split=spec.split,
            streaming=True,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load dataset={spec.dataset_name!r}, config={spec.config_name!r}, split={spec.split!r}"
        ) from exc


def load_local_dataset_stream(path: Path, split: str):
    if not path.exists():
        raise FileNotFoundError(f"local dataset path does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                item = json.loads(stripped)
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object")
                yield item
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get(split), list):
            payload = payload[split]
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            payload = payload["data"]
        if not isinstance(payload, list):
            raise ValueError("local JSON dataset must be a list, or a dict containing a split/data list")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"JSON item {index} must be an object")
            yield item
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError(f"unsupported local dataset extension: {path.suffix}")


def format_dataset_example(
    example: dict[str, Any],
    spec: DatasetSpec,
    image_root: str | None,
    include_target: bool,
) -> TaskExample:
    if spec.task == "llm":
        prompt = format_llm_prompt(example, spec.source, include_target=include_target)
    elif spec.task == "vlm":
        prompt = format_vlm_prompt(example, spec.source, include_target=include_target)
    elif spec.task == "vla":
        prompt = format_vla_prompt(example, include_target=include_target)
    else:
        raise ValueError(f"unsupported dataset task: {spec.task}")

    image = None
    if spec.task in {"vlm", "vla"}:
        image = normalize_image_value(
            get_first_value(
                example,
                IMAGE_FIELD_NAMES,
            ),
            image_root=image_root,
        )
        if image is None:
            image = normalize_image_value(example, image_root=image_root)
    target = get_first_value(
        example,
        ("answer", "answers", "label", "target", "output", "response", "caption", "action", "actions"),
    )
    action = None
    state = None
    if spec.task == "vla":
        action = first_present(
            get_first_value(example, ACTION_FIELD_NAMES),
            find_nested_value(example, ACTION_FIELD_NAMES, max_depth=3),
        )
        state = first_present(
            get_first_value(example, STATE_FIELD_NAMES),
            find_nested_value(example, STATE_FIELD_NAMES, max_depth=3),
        )
    metadata = {
        "source": spec.source,
        "task": spec.task,
    }
    if "id" in example:
        metadata["id"] = compact_value(example["id"])
    return TaskExample(prompt=prompt, image=image, target=target, action=action, state=state, metadata=metadata)


def load_dataset_examples(
    spec: DatasetSpec,
    max_prompts: int,
    cache_dir: Path,
    image_root: str | None = None,
    include_target: bool = False,
) -> list[TaskExample]:
    dataset = load_dataset_stream(spec, cache_dir)
    examples: list[TaskExample] = []
    for index, example in enumerate(dataset):
        try:
            examples.append(
                format_dataset_example(
                    example,
                    spec,
                    image_root=image_root,
                    include_target=include_target,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"failed to format dataset example at stream index {index}") from exc
        if len(examples) >= max_prompts:
            break
    if not examples:
        raise RuntimeError(
            f"dataset={spec.dataset_name!r}, config={spec.config_name!r}, split={spec.split!r} yielded no examples"
        )
    return examples


def load_mmlu_prompts(
    dataset_name: str,
    subject: str,
    split: str,
    max_prompts: int,
    cache_dir: Path,
) -> list[str]:
    spec = DatasetSpec(
        source="mmlu",
        task="llm",
        dataset_name=dataset_name,
        config_name=subject,
        split=split,
    )
    return [
        example.prompt
        for example in load_dataset_examples(
            spec,
            max_prompts=max_prompts,
            cache_dir=cache_dir,
        )
    ]


def built_in_examples(model_kind: str, max_prompts: int) -> list[TaskExample]:
    if model_kind == "vla":
        prompts = [
            "You are a vision-language-action robot policy.\nInstruction: pick up the red block and place it in the bowl.\nPredict the next robot action.\nAction:",
            "You are a vision-language-action robot policy.\nInstruction: open the drawer using the handle.\nPredict the next robot action.\nAction:",
            "You are a vision-language-action robot policy.\nInstruction: move the gripper above the blue cube.\nPredict the next robot action.\nAction:",
        ]
    elif model_kind == "vlm":
        prompts = [
            "What objects are visible in the image?\nAnswer:",
            "Describe the scene in one sentence.",
            "What should the assistant pay attention to in this image?\nAnswer:",
        ]
    else:
        prompts = PROMPTS
    return [TaskExample(prompt=prompt, metadata={"source": "built_in"}) for prompt in prompts[:max_prompts]]


def load_prompts_from_args(args: argparse.Namespace, model_kind: str) -> tuple[list[TaskExample], dict[str, Any]]:
    prompt_source = normalize_prompt_source(args.prompt_source)
    if prompt_source == "built_in":
        examples = built_in_examples(model_kind, args.max_prompts)
        metadata = {
            "source": "built_in",
            "task": task_from_model_kind(model_kind),
            "prompt_preview": [example.prompt for example in examples[:2]],
        }
    else:
        dataset_cache_dir = Path(args.dataset_cache_dir)
        if not dataset_cache_dir.is_absolute():
            dataset_cache_dir = ROOT / dataset_cache_dir
        spec = dataset_spec_from_args(args, prompt_source, model_kind)
        examples = load_dataset_examples(
            spec,
            max_prompts=args.max_prompts,
            cache_dir=dataset_cache_dir,
            image_root=args.dataset_image_root,
            include_target=args.include_dataset_target,
        )
        metadata = {
            "source": spec.source,
            "task": spec.task,
            "dataset": spec.dataset_name,
            "config": spec.config_name,
            "split": spec.split,
            "local_path": spec.local_path,
            "cache_dir": str(dataset_cache_dir),
            "image_root": args.dataset_image_root,
            "include_target": args.include_dataset_target,
            "prompt_preview": [example.prompt for example in examples[:2]],
        }
    if not examples:
        raise ValueError("--max-prompts must select at least one prompt")
    return examples, metadata


class _AcceptAllParallelStyles:
    def __contains__(self, item) -> bool:
        return True

    def __repr__(self) -> str:
        return "<tensor parallel style check disabled>"


if getattr(transformers_modeling_utils, "ALL_PARALLEL_STYLES", None) is None:
    transformers_modeling_utils.ALL_PARALLEL_STYLES = _AcceptAllParallelStyles()


@dataclass
class VariantScore:
    layer_idx: int
    variant: str
    kl: float
    metric: str
    target_name: str
    effective_param_memory_bytes: float
    kv_cache_memory_bytes: float
    runtime_proxy: float
    measured_seconds: float


FLA_VARIANT_TO_CLASS = {
    "fla_linear_attn": "LinearAttention",
    "fla_gated_linear_attn": "GatedLinearAttention",
    "fla_based_linear_attn": "BasedLinearAttention",
    "fla_rebased_linear_attn": "ReBasedLinearAttention",
    "fla_deltanet_attn": "DeltaNet",
    "fla_gated_deltanet_attn": "GatedDeltaNet",
    "fla_kimi_delta_attn": "KimiDeltaAttention",
    "fla_multiscale_retention_attn": "MultiScaleRetention",
    "fla_mla_attn": "MultiheadLatentAttention",
    "fla_native_sparse_attn": "NativeSparseAttention",
    "fla_moba_attn": "MoBA",
}
FLA_LINEAR_ATTENTION_VARIANTS = (
    "fla_linear_attn",
    "fla_gated_linear_attn",
    "fla_based_linear_attn",
    "fla_rebased_linear_attn",
    "fla_deltanet_attn",
    "fla_gated_deltanet_attn",
    "fla_kimi_delta_attn",
)
FLA_STRUCTURED_ATTENTION_VARIANTS = (
    "fla_multiscale_retention_attn",
    "fla_mla_attn",
    "fla_native_sparse_attn",
    "fla_moba_attn",
)
QWEN_ATTENTION_VARIANTS = (
    "parent_attn",
    "mha_attn",
    "quant_mha_attn",
    "mqa_attn",
    "gqa_kv2",
    "mfa_kv2",
    "mla_kv2",
    "mka_attn",
    "linear_attn",
    "noop_attn",
)
LINEAR_ATTENTION_VARIANTS = ("linear_attn",) + FLA_LINEAR_ATTENTION_VARIANTS
CORE_ATTENTION_VARIANTS = (
    "parent_attn",
    "mha_attn",
    "quant_mha_attn",
    "mqa_attn",
    "gqa_kv2",
    "mfa_kv2",
    "mla_kv2",
    "mka_attn",
) + LINEAR_ATTENTION_VARIANTS + ("noop_attn",)
ALL_ATTENTION_VARIANTS = CORE_ATTENTION_VARIANTS + FLA_STRUCTURED_ATTENTION_VARIANTS
FLA_VARIANT_ALIASES = {
    "all_fla": tuple(FLA_VARIANT_TO_CLASS),
    "all_qwen_attn": tuple(QWEN_ATTENTION_VARIANTS),
    "all_linear_attn": LINEAR_ATTENTION_VARIANTS,
    "all_core_attn": CORE_ATTENTION_VARIANTS,
    "all_attention": ALL_ATTENTION_VARIANTS,
    "fla_gla_attn": ("fla_gated_linear_attn",),
    "fla_based_attn": ("fla_based_linear_attn",),
    "fla_rebased_attn": ("fla_rebased_linear_attn",),
    "fla_delta_net_attn": ("fla_deltanet_attn",),
    "fla_gated_delta_net_attn": ("fla_gated_deltanet_attn",),
    "fla_kda_attn": ("fla_kimi_delta_attn",),
    "fla_retention_attn": ("fla_multiscale_retention_attn",),
    "fla_multihead_latent_attn": ("fla_mla_attn",),
    "fla_nsa_attn": ("fla_native_sparse_attn",),
}
NON_FLA_VARIANTS = {"parent", "skip_attn", "skip_mlp", "skip_both", *QWEN_ATTENTION_VARIANTS}
DEFAULT_VARIANTS = "parent,skip_attn,skip_mlp,skip_both,all_core_attn,all_fla"


class QwenCandidateLayer(nn.Module):
    """Qwen/Llama-style decoder layer wrapper for NAS attention candidates."""

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
        if variant not in NON_FLA_VARIANTS and variant not in FLA_VARIANT_TO_CLASS:
            raise ValueError(f"unknown variant: {variant}")
        self.base_layer = base_layer
        self.variant = variant
        self.fla_attn = None
        self.qwen_attn = None
        if is_qwen_attention_variant(variant) and variant not in {"parent_attn", "noop_attn"}:
            if config is None:
                raise ValueError(f"config is required for {variant}")
            self.qwen_attn = make_qwen_attention_variant(base_layer.self_attn, config, variant)
        if is_fla_variant(variant):
            if config is None:
                raise ValueError(f"config is required for {variant}")
            self.fla_attn = make_fla_attention_variant(
                base_layer,
                config,
                variant=variant,
                layer_idx=layer_idx,
                mode=fla_mode,
                feature_map=fla_feature_map,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        if self.variant in {"parent", "parent_attn"}:
            return self.base_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if self.variant in {"skip_attn", "skip_both", "noop_attn"}:
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (past_key_value,)
            if self.variant == "skip_both":
                return outputs
        elif is_qwen_attention_variant(self.variant):
            residual = hidden_states
            hidden_states = self.base_layer.input_layernorm(hidden_states)
            assert self.qwen_attn is not None
            attn_outputs = self.qwen_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + attn_outputs[0]
        elif is_fla_variant(self.variant):
            residual = hidden_states
            hidden_states = self.base_layer.input_layernorm(hidden_states)
            assert self.fla_attn is not None
            fla_outputs = self.fla_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=output_attentions,
            )
            attn_hidden, attn_outputs = normalize_attention_outputs(fla_outputs, past_key_value)
            hidden_states = residual + attn_hidden
        else:
            residual = hidden_states
            hidden_states = self.base_layer.input_layernorm(hidden_states)
            attn_outputs = self.base_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + attn_outputs[0]

        if self.variant == "skip_mlp":
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (attn_outputs[1],)
            if use_cache:
                cache_index = 2 if output_attentions else 1
                outputs += (attn_outputs[cache_index],)
            return outputs

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs


class QwenLinearAttentionSubblock(nn.Module):
    """A linear attention replacement initialized as W_o W_v."""

    def __init__(self, parent_attn: nn.Module, config) -> None:
        super().__init__()
        info = qwen_attention_info(parent_attn, config)
        self.linear = nn.Linear(info.hidden_size, info.hidden_size, bias=parent_attn.o_proj.bias is not None)
        self.to_parent_device(parent_attn)
        value_weight, value_bias = expanded_qwen_value_projection(parent_attn, info)
        with torch.no_grad():
            self.linear.weight.copy_(parent_attn.o_proj.weight @ value_weight)
            if self.linear.bias is not None:
                bias = torch.zeros_like(self.linear.bias)
                if value_bias is not None:
                    bias = bias + parent_attn.o_proj.weight @ value_bias
                if parent_attn.o_proj.bias is not None:
                    bias = bias + parent_attn.o_proj.bias
                self.linear.bias.copy_(bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return self.linear(hidden_states), None, kwargs.get("past_key_value")

    def to_parent_device(self, parent_attn: nn.Module) -> None:
        self.to(device=parent_attn.q_proj.weight.device, dtype=parent_attn.q_proj.weight.dtype)


class QwenProjectedAttention(nn.Module):
    """Qwen-compatible causal SDPA with a configurable number of KV heads."""

    def __init__(self, parent_attn: nn.Module, config, target_num_kv_heads: int) -> None:
        super().__init__()
        self.info = qwen_attention_info(parent_attn, config)
        self.num_heads = self.info.num_heads
        self.num_kv_heads = target_num_kv_heads
        self.head_dim = self.info.head_dim
        self.hidden_size = self.info.hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=parent_attn.q_proj.bias is not None)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=parent_attn.k_proj.bias is not None)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=parent_attn.v_proj.bias is not None)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=parent_attn.o_proj.bias is not None)
        self.q_norm = copy.deepcopy(getattr(parent_attn, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(parent_attn, "k_norm", nn.Identity()))
        self.to(device=parent_attn.q_proj.weight.device, dtype=parent_attn.q_proj.weight.dtype)
        with torch.no_grad():
            self.q_proj.load_state_dict(parent_attn.q_proj.state_dict())
            self.o_proj.load_state_dict(parent_attn.o_proj.state_dict())
            for source, dest in [(parent_attn.k_proj, self.k_proj), (parent_attn.v_proj, self.v_proj)]:
                weight, bias = remap_qwen_kv_projection(source, self.info.num_kv_heads, target_num_kv_heads, self.head_dim)
                dest.weight.copy_(weight)
                if dest.bias is not None and bias is not None:
                    dest.bias.copy_(bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return qwen_sdpa_output(self, query, key, value, hidden_states, **kwargs)


class QwenQuantizedProjectedAttention(nn.Module):
    """Qwen-compatible MHA with int8-quantized q/k/v/o projections."""

    def __init__(self, parent_attn: nn.Module, config, target_num_kv_heads: int, num_bits: int = 8) -> None:
        super().__init__()
        self.info = qwen_attention_info(parent_attn, config)
        self.num_heads = self.info.num_heads
        self.num_kv_heads = target_num_kv_heads
        self.head_dim = self.info.head_dim
        self.hidden_size = self.info.hidden_size
        self.q_proj = QuantizedLinear(parent_attn.q_proj.weight, parent_attn.q_proj.bias, num_bits=num_bits)
        k_weight, k_bias = remap_qwen_kv_projection(parent_attn.k_proj, self.info.num_kv_heads, target_num_kv_heads, self.head_dim)
        v_weight, v_bias = remap_qwen_kv_projection(parent_attn.v_proj, self.info.num_kv_heads, target_num_kv_heads, self.head_dim)
        self.k_proj = QuantizedLinear(k_weight, k_bias, num_bits=num_bits)
        self.v_proj = QuantizedLinear(v_weight, v_bias, num_bits=num_bits)
        self.o_proj = QuantizedLinear(parent_attn.o_proj.weight, parent_attn.o_proj.bias, num_bits=num_bits)
        self.q_norm = copy.deepcopy(getattr(parent_attn, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(parent_attn, "k_norm", nn.Identity()))
        self.to(device=parent_attn.q_proj.weight.device)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return qwen_sdpa_output(self, query, key, value, hidden_states, **kwargs)


class QwenFactorizedKVAttention(nn.Module):
    """MFA-style Qwen attention with grouped low-rank K/V projections."""

    def __init__(self, parent_attn: nn.Module, config, target_num_kv_heads: int, latent_dim: int) -> None:
        super().__init__()
        self.info = qwen_attention_info(parent_attn, config)
        self.num_heads = self.info.num_heads
        self.num_kv_heads = target_num_kv_heads
        self.head_dim = self.info.head_dim
        self.hidden_size = self.info.hidden_size
        self.latent_dim = latent_dim
        latent_size = target_num_kv_heads * latent_dim
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=parent_attn.q_proj.bias is not None)
        self.k_down_proj = nn.Linear(self.hidden_size, latent_size, bias=False)
        self.k_up_proj = nn.Linear(latent_size, target_num_kv_heads * self.head_dim, bias=parent_attn.k_proj.bias is not None)
        self.v_down_proj = nn.Linear(self.hidden_size, latent_size, bias=False)
        self.v_up_proj = nn.Linear(latent_size, target_num_kv_heads * self.head_dim, bias=parent_attn.v_proj.bias is not None)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=parent_attn.o_proj.bias is not None)
        self.q_norm = copy.deepcopy(getattr(parent_attn, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(parent_attn, "k_norm", nn.Identity()))
        self.to(device=parent_attn.q_proj.weight.device, dtype=parent_attn.q_proj.weight.dtype)
        k_weight, k_bias = remap_qwen_kv_projection(parent_attn.k_proj, self.info.num_kv_heads, target_num_kv_heads, self.head_dim)
        v_weight, v_bias = remap_qwen_kv_projection(parent_attn.v_proj, self.info.num_kv_heads, target_num_kv_heads, self.head_dim)
        with torch.no_grad():
            self.q_proj.load_state_dict(parent_attn.q_proj.state_dict())
            self.o_proj.load_state_dict(parent_attn.o_proj.state_dict())
        low_rank_init(self.k_down_proj, self.k_up_proj, k_weight, k_bias)
        low_rank_init(self.v_down_proj, self.v_up_proj, v_weight, v_bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_up_proj(self.k_down_proj(hidden_states)).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_up_proj(self.v_down_proj(hidden_states)).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return qwen_sdpa_output(self, query, key, value, hidden_states, **kwargs)


class QwenLatentKVAttention(nn.Module):
    """MLA-style Qwen attention with one shared latent K/V projection."""

    def __init__(self, parent_attn: nn.Module, config, target_num_kv_heads: int, latent_dim: int) -> None:
        super().__init__()
        self.info = qwen_attention_info(parent_attn, config)
        self.num_heads = self.info.num_heads
        self.num_kv_heads = target_num_kv_heads
        self.head_dim = self.info.head_dim
        self.hidden_size = self.info.hidden_size
        self.latent_dim = latent_dim
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=parent_attn.q_proj.bias is not None)
        self.kv_down_proj = nn.Linear(self.hidden_size, latent_dim, bias=False)
        self.k_up_proj = nn.Linear(latent_dim, target_num_kv_heads * self.head_dim, bias=parent_attn.k_proj.bias is not None)
        self.v_up_proj = nn.Linear(latent_dim, target_num_kv_heads * self.head_dim, bias=parent_attn.v_proj.bias is not None)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=parent_attn.o_proj.bias is not None)
        self.q_norm = copy.deepcopy(getattr(parent_attn, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(parent_attn, "k_norm", nn.Identity()))
        self.to(device=parent_attn.q_proj.weight.device, dtype=parent_attn.q_proj.weight.dtype)
        k_weight, k_bias = remap_qwen_kv_projection(parent_attn.k_proj, self.info.num_kv_heads, target_num_kv_heads, self.head_dim)
        v_weight, v_bias = remap_qwen_kv_projection(parent_attn.v_proj, self.info.num_kv_heads, target_num_kv_heads, self.head_dim)
        with torch.no_grad():
            self.q_proj.load_state_dict(parent_attn.q_proj.state_dict())
            self.o_proj.load_state_dict(parent_attn.o_proj.state_dict())
        joint_low_rank_kv_init(self.kv_down_proj, self.k_up_proj, self.v_up_proj, k_weight, k_bias, v_weight, v_bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        latent = self.kv_down_proj(hidden_states)
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_up_proj(latent).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_up_proj(latent).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return qwen_sdpa_output(self, query, key, value, hidden_states, **kwargs)


class QwenMultiKeyAttention(nn.Module):
    """MKA-style Qwen attention with one shared key and low-rank values."""

    def __init__(self, parent_attn: nn.Module, config, value_rank: int) -> None:
        super().__init__()
        self.info = qwen_attention_info(parent_attn, config)
        self.num_heads = self.info.num_heads
        self.num_kv_heads = 1
        self.head_dim = self.info.head_dim
        self.hidden_size = self.info.hidden_size
        self.value_rank = value_rank
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=parent_attn.q_proj.bias is not None)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=parent_attn.k_proj.bias is not None)
        self.v_down_proj = nn.Linear(self.hidden_size, value_rank, bias=False)
        self.v_up_proj = nn.Linear(value_rank, self.num_heads * self.head_dim, bias=parent_attn.v_proj.bias is not None)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=parent_attn.o_proj.bias is not None)
        self.q_norm = copy.deepcopy(getattr(parent_attn, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(parent_attn, "k_norm", nn.Identity()))
        self.to(device=parent_attn.q_proj.weight.device, dtype=parent_attn.q_proj.weight.dtype)
        k_weight = parent_attn.k_proj.weight.detach().view(self.info.num_kv_heads, self.head_dim, self.hidden_size).mean(dim=0)
        k_bias = None
        if parent_attn.k_proj.bias is not None:
            k_bias = parent_attn.k_proj.bias.detach().view(self.info.num_kv_heads, self.head_dim).mean(dim=0)
        value_weight, value_bias = expanded_qwen_value_projection(parent_attn, self.info)
        with torch.no_grad():
            self.q_proj.load_state_dict(parent_attn.q_proj.state_dict())
            self.o_proj.load_state_dict(parent_attn.o_proj.state_dict())
            self.k_proj.weight.copy_(k_weight)
            if self.k_proj.bias is not None and k_bias is not None:
                self.k_proj.bias.copy_(k_bias)
        low_rank_init(self.v_down_proj, self.v_up_proj, value_weight, value_bias)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        value = self.v_up_proj(self.v_down_proj(hidden_states)).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return qwen_sdpa_output(self, query, key, value, hidden_states, **kwargs)


def is_qwen_attention_variant(variant: str) -> bool:
    return variant in QWEN_ATTENTION_VARIANTS


def make_qwen_attention_variant(parent_attn: nn.Module, config, variant: str) -> nn.Module:
    info = qwen_attention_info(parent_attn, config)
    if variant == "mha_attn":
        return QwenProjectedAttention(parent_attn, config, target_num_kv_heads=info.num_heads)
    if variant == "quant_mha_attn":
        return QwenQuantizedProjectedAttention(parent_attn, config, target_num_kv_heads=info.num_heads, num_bits=8)
    if variant == "mqa_attn":
        return QwenProjectedAttention(parent_attn, config, target_num_kv_heads=1)
    if variant.startswith("gqa_kv"):
        return QwenProjectedAttention(parent_attn, config, target_num_kv_heads=parse_kv_suffix(variant))
    if variant.startswith("mfa_kv"):
        return QwenFactorizedKVAttention(
            parent_attn,
            config,
            target_num_kv_heads=parse_kv_suffix(variant),
            latent_dim=max(1, info.head_dim // 2),
        )
    if variant.startswith("mla_kv"):
        return QwenLatentKVAttention(
            parent_attn,
            config,
            target_num_kv_heads=parse_kv_suffix(variant),
            latent_dim=max(1, info.hidden_size // 4),
        )
    if variant == "mka_attn":
        return QwenMultiKeyAttention(parent_attn, config, value_rank=max(1, info.hidden_size // 4))
    if variant == "linear_attn":
        return QwenLinearAttentionSubblock(parent_attn, config)
    raise ValueError(f"unknown Qwen attention variant: {variant}")


def parse_kv_suffix(variant: str) -> int:
    try:
        return int(variant.rsplit("kv", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"expected a variant ending with kv<count>, got {variant}") from exc


def qwen_sdpa_output(module, query, key, value, hidden_states, **kwargs):
    query = module.q_norm(query)
    key = module.k_norm(key)
    query, key = apply_qwen_rotary(query, key, kwargs.get("position_embeddings"))
    if key.shape[1] != query.shape[1]:
        key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
    if value.shape[1] != query.shape[1]:
        value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
    attn_mask = kwargs.get("attention_mask")
    is_causal = attn_mask is None and query.shape[-2] > 1
    attn_output = F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    batch, seq_len, _ = hidden_states.shape
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, module.num_heads * module.head_dim)
    return module.o_proj(attn_output), None, kwargs.get("past_key_value")


def apply_qwen_rotary(query, key, position_embeddings):
    if position_embeddings is None:
        return query, key
    cos, sin = position_embeddings
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(device=query.device, dtype=query.dtype)
    sin = sin.to(device=query.device, dtype=query.dtype)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def qwen_attention_info(parent_attn: nn.Module, config):
    hidden_size = getattr(config, "hidden_size", parent_attn.q_proj.in_features)
    num_heads = getattr(config, "num_attention_heads", None) or getattr(parent_attn, "num_heads", None)
    head_dim = getattr(config, "head_dim", None) or getattr(parent_attn, "head_dim", None)
    if num_heads is None:
        if head_dim is None:
            raise ValueError("cannot infer num_heads for Qwen attention")
        num_heads = parent_attn.q_proj.out_features // head_dim
    if head_dim is None:
        head_dim = hidden_size // num_heads
    num_kv_heads = (
        getattr(config, "num_key_value_heads", None)
        or getattr(parent_attn, "num_key_value_heads", None)
        or getattr(parent_attn, "num_kv_heads", None)
        or parent_attn.k_proj.out_features // head_dim
    )
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def remap_qwen_kv_projection(
    projection: nn.Linear,
    parent_num_kv_heads: int,
    target_num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = projection.weight.detach().view(parent_num_kv_heads, head_dim, projection.in_features)
    bias = projection.bias.detach().view(parent_num_kv_heads, head_dim) if projection.bias is not None else None
    if target_num_kv_heads == parent_num_kv_heads:
        target_weight = weight
        target_bias = bias
    elif parent_num_kv_heads % target_num_kv_heads == 0:
        group = parent_num_kv_heads // target_num_kv_heads
        target_weight = weight.view(target_num_kv_heads, group, head_dim, projection.in_features).mean(dim=1)
        target_bias = bias.view(target_num_kv_heads, group, head_dim).mean(dim=1) if bias is not None else None
    elif target_num_kv_heads % parent_num_kv_heads == 0:
        repeat = target_num_kv_heads // parent_num_kv_heads
        target_weight = weight.repeat_interleave(repeat, dim=0)
        target_bias = bias.repeat_interleave(repeat, dim=0) if bias is not None else None
    else:
        raise ValueError(f"cannot remap {parent_num_kv_heads} KV heads to {target_num_kv_heads}")
    target_weight = target_weight.reshape(target_num_kv_heads * head_dim, projection.in_features)
    if target_bias is not None:
        target_bias = target_bias.reshape(target_num_kv_heads * head_dim)
    return target_weight, target_bias


def expanded_qwen_value_projection(parent_attn: nn.Module, info) -> tuple[torch.Tensor, torch.Tensor | None]:
    return remap_qwen_kv_projection(parent_attn.v_proj, info.num_kv_heads, info.num_heads, info.head_dim)


def low_rank_init(down: nn.Linear, up: nn.Linear, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    rank = down.out_features
    device = down.weight.device
    dtype = down.weight.dtype
    u, s, vh = torch.linalg.svd(weight.detach().float().cpu(), full_matrices=False)
    components = min(rank, int(s.numel()))
    with torch.no_grad():
        down.weight.zero_()
        up.weight.zero_()
        down.weight[:components].copy_(vh[:components].to(device=device, dtype=dtype))
        up.weight[:, :components].copy_((u[:, :components] * s[:components]).to(device=device, dtype=dtype))
        if up.bias is not None:
            if bias is None:
                up.bias.zero_()
            else:
                up.bias.copy_(bias.to(device=device, dtype=dtype))


def joint_low_rank_kv_init(
    down: nn.Linear,
    k_up: nn.Linear,
    v_up: nn.Linear,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor | None,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor | None,
) -> None:
    rank = down.out_features
    device = down.weight.device
    dtype = down.weight.dtype
    joint_weight = torch.cat([k_weight.detach(), v_weight.detach()], dim=0)
    u, s, vh = torch.linalg.svd(joint_weight.float().cpu(), full_matrices=False)
    components = min(rank, int(s.numel()))
    k_rows = k_weight.shape[0]
    joint_up = u[:, :components] * s[:components]
    with torch.no_grad():
        down.weight.zero_()
        k_up.weight.zero_()
        v_up.weight.zero_()
        down.weight[:components].copy_(vh[:components].to(device=device, dtype=dtype))
        k_up.weight[:, :components].copy_(joint_up[:k_rows].to(device=device, dtype=dtype))
        v_up.weight[:, :components].copy_(joint_up[k_rows:].to(device=device, dtype=dtype))
        if k_up.bias is not None:
            k_up.bias.zero_() if k_bias is None else k_up.bias.copy_(k_bias.to(device=device, dtype=dtype))
        if v_up.bias is not None:
            v_up.bias.zero_() if v_bias is None else v_up.bias.copy_(v_bias.to(device=device, dtype=dtype))


def is_fla_variant(variant: str) -> bool:
    return variant in FLA_VARIANT_TO_CLASS


def expand_variants(raw_variants: str) -> list[str]:
    variants: list[str] = []
    for item in [variant.strip() for variant in raw_variants.split(",") if variant.strip()]:
        expanded = FLA_VARIANT_ALIASES.get(item, (item,))
        for variant in expanded:
            if variant not in variants:
                variants.append(variant)
    if "parent" not in variants:
        variants.insert(0, "parent")
    return variants


def normalize_attention_outputs(outputs, fallback_cache):
    if isinstance(outputs, torch.Tensor):
        return outputs, (outputs, None, fallback_cache)
    if not isinstance(outputs, tuple) or not outputs:
        raise TypeError(f"expected FLA attention to return a tensor or non-empty tuple, got {type(outputs)!r}")
    hidden_states = outputs[0]
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"expected first FLA output to be a tensor, got {type(hidden_states)!r}")
    if len(outputs) >= 3:
        return hidden_states, outputs
    if len(outputs) == 2:
        return hidden_states, (hidden_states, outputs[1], fallback_cache)
    return hidden_states, (hidden_states, None, fallback_cache)


def make_fla_attention_variant(
    base_layer: nn.Module,
    config,
    variant: str,
    layer_idx: int | None,
    mode: str,
    feature_map: str,
) -> nn.Module:
    _disable_torch_compile_if_unavailable()
    class_name = FLA_VARIANT_TO_CLASS[variant]
    try:
        fla_layers = importlib.import_module("fla.layers")
        layer_cls = getattr(fla_layers, class_name)
    except Exception as exc:
        raise RuntimeError(
            f"{variant} requires fla.layers.{class_name}. Install a compatible "
            "flash-linear-attention source tree under vendor/flash-linear-attention "
            f"or vendor/flash-linear-attention-v0.4.2. Original error: {exc}"
        ) from exc

    parent_attn = base_layer.self_attn
    dtype = parent_attn.q_proj.weight.dtype
    device = parent_attn.q_proj.weight.device
    kwargs = fla_constructor_kwargs(
        config,
        parent_attn,
        variant=variant,
        mode=mode,
        feature_map=feature_map,
        layer_idx=layer_idx,
    )
    try:
        layer = layer_cls(**kwargs).to(device=device, dtype=dtype)
    except Exception as exc:
        raise RuntimeError(f"failed to construct {variant} ({class_name}) with kwargs={kwargs}") from exc
    initialize_fla_from_qwen(layer, parent_attn, config)
    return layer


def fla_constructor_kwargs(
    config,
    parent_attn: nn.Module,
    variant: str,
    mode: str,
    feature_map: str,
    layer_idx: int | None,
) -> dict:
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    expand_k = parent_attn.q_proj.out_features / hidden_size
    expand_v = parent_attn.o_proj.in_features / hidden_size
    rope_theta = getattr(config, "rope_theta", 10000.0)
    max_position_embeddings = getattr(config, "max_position_embeddings", None)

    if variant == "fla_linear_attn":
        return {
            "mode": mode,
            "hidden_size": hidden_size,
            "expand_k": expand_k,
            "expand_v": expand_v,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "feature_map": feature_map,
            "output_norm": "identity",
            "do_feature_map_norm": False,
            "layer_idx": layer_idx,
        }
    if variant == "fla_gated_linear_attn":
        return {
            "mode": mode,
            "hidden_size": hidden_size,
            "expand_k": 1.0,
            "expand_v": 1.0,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "feature_map": None,
            "use_short_conv": False,
            "use_output_gate": False,
            "fuse_norm": False,
            "layer_idx": layer_idx,
        }
    if variant == "fla_based_linear_attn":
        fla_head_dim = hidden_size // num_heads
        return {
            "hidden_size": hidden_size,
            "feature_dim": fla_head_dim,
            "num_key_value_heads": num_heads,
            "num_heads": num_heads,
            "mode": "parallel",
        }
    if variant == "fla_rebased_linear_attn":
        fla_head_dim = hidden_size // num_heads
        return {
            "hidden_size": hidden_size,
            "l_max": max_position_embeddings or 2048,
            "feature_dim": fla_head_dim,
            "num_key_value_heads": num_heads,
            "num_heads": num_heads,
            "mode": "parallel",
            "layer_idx": layer_idx,
        }
    if variant == "fla_deltanet_attn":
        return {
            "mode": mode if mode in {"chunk", "fused_recurrent"} else "chunk",
            "hidden_size": hidden_size,
            "expand_k": 1.0,
            "expand_v": 1.0,
            "num_heads": num_heads,
            "use_gate": False,
            "use_short_conv": False,
            "layer_idx": layer_idx,
        }
    if variant == "fla_gated_deltanet_attn":
        return {
            "hidden_size": hidden_size,
            "expand_v": 1.0,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "num_v_heads": num_heads,
            "mode": mode if mode in {"chunk", "fused_recurrent"} else "chunk",
            "use_gate": False,
            "use_short_conv": False,
            "layer_idx": layer_idx,
        }
    if variant == "fla_kimi_delta_attn":
        return {
            "hidden_size": hidden_size,
            "expand_v": 1.0,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "num_v_heads": num_heads,
            "mode": mode if mode in {"chunk", "fused_recurrent"} else "chunk",
            "use_short_conv": False,
            "layer_idx": layer_idx,
        }
    if variant == "fla_multiscale_retention_attn":
        return {
            "mode": mode,
            "hidden_size": hidden_size,
            "expand_k": 1.0,
            "expand_v": 1.0,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "feature_map": None,
            "use_short_conv": False,
            "use_output_gate": False,
            "fuse_norm": False,
            "layer_idx": layer_idx,
        }
    if variant == "fla_mla_attn":
        qk_rope_head_dim = max(1, head_dim // 2)
        qk_nope_head_dim = head_dim - qk_rope_head_dim
        return {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "q_lora_rank": None,
            "qk_rope_head_dim": qk_rope_head_dim,
            "kv_lora_rank": max(1, hidden_size // 4),
            "v_head_dim": head_dim,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_head_dim": head_dim,
            "rope_theta": rope_theta,
            "max_position_embeddings": max_position_embeddings,
            "layer_idx": layer_idx,
        }
    if variant == "fla_native_sparse_attn":
        return {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": max(1, num_heads // 16),
            "head_dim": head_dim,
            "qkv_bias": False,
            "block_size": 64,
            "block_counts": 16,
            "window_size": 0,
            "rope_theta": rope_theta,
            "max_position_embeddings": max_position_embeddings,
            "layer_idx": layer_idx,
        }
    if variant == "fla_moba_attn":
        return {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "qkv_bias": False,
            "qk_norm": False,
            "window_size": None,
            "rope_theta": rope_theta,
            "max_position_embeddings": max_position_embeddings,
            "layer_idx": layer_idx,
            "moba_chunk_size": 256,
            "moba_topk": 4,
            "use_output_gate": False,
            "use_flash_moba": False,
        }
    raise ValueError(f"unknown FLA variant: {variant}")


def initialize_fla_from_qwen(layer: nn.Module, parent_attn: nn.Module, config) -> None:
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)
    expanded_k = expanded_qwen_kv_weight(parent_attn.k_proj.weight, num_kv_heads, num_heads, head_dim)
    expanded_v = expanded_qwen_kv_weight(parent_attn.v_proj.weight, num_kv_heads, num_heads, head_dim)
    with torch.no_grad():
        copy_linear_weight_if_compatible(getattr(layer, "q_proj", None), parent_attn.q_proj.weight)
        copy_linear_weight_if_compatible(getattr(layer, "k_proj", None), parent_attn.k_proj.weight, expanded_k)
        copy_linear_weight_if_compatible(getattr(layer, "v_proj", None), parent_attn.v_proj.weight, expanded_v)
        copy_linear_weight_if_compatible(getattr(layer, "o_proj", None), parent_attn.o_proj.weight)


def expanded_qwen_kv_weight(
    weight: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if num_heads == num_kv_heads:
        return weight
    repeat = num_heads // num_kv_heads
    return weight.detach().view(num_kv_heads, head_dim, -1).repeat_interleave(repeat, dim=0).reshape(num_heads * head_dim, -1)


def copy_linear_weight_if_compatible(module, *sources: torch.Tensor) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    for source in sources:
        if module.weight.shape == source.shape:
            module.weight.copy_(source.to(device=module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None:
                module.bias.zero_()
            return True
    return False


def _disable_torch_compile_if_unavailable() -> None:
    try:
        torch.compile(lambda x: x)
    except Exception:
        def identity_compile(fn=None, *args, **kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        torch.compile = identity_compile
    if hasattr(torch, "distributed") and not hasattr(torch.distributed, "DeviceMesh"):
        torch.distributed.DeviceMesh = object
    try:
        import torch.distributed.tensor as distributed_tensor
    except Exception:
        distributed_tensor = None
    if distributed_tensor is not None:
        if not hasattr(distributed_tensor, "Replicate"):
            distributed_tensor.Replicate = object
        if not hasattr(distributed_tensor, "Shard"):
            distributed_tensor.Shard = object
        if not hasattr(distributed_tensor, "Placement"):
            distributed_tensor.Placement = object
        if not hasattr(distributed_tensor, "DTensor"):
            distributed_tensor.DTensor = torch.Tensor
        if not hasattr(distributed_tensor, "distribute_module"):
            def distribute_module(module, *args, **kwargs):
                return module

            distributed_tensor.distribute_module = distribute_module


@dataclass
class LoadedModel:
    model: nn.Module
    processor: Any | None
    tokenizer: Any | None
    model_kind: str
    layers: nn.ModuleList
    layer_path: str
    language_config: Any


def normalize_model_kind(kind: str, model_id: str, config: Any | None = None) -> str:
    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {"text", "llm", "causal_lm"}:
        return "text"
    if normalized in {"vlm", "vision_language", "vision_language_model", "image_text"}:
        return "vlm"
    if normalized in {"vla", "vision_language_action", "vision_language_action_model", "robot_policy"}:
        return "vla"
    if normalized != "auto":
        raise ValueError("--model-kind must be auto, text, vlm, or vla")

    lowered = model_id.lower()
    vla_markers = ("vla", "openvla", "spatialvla", "robot", "robotics", "libero", "rt-1", "rt1", "octo")
    if any(marker in lowered for marker in vla_markers):
        return "vla"
    if config is not None and "vla" in str(getattr(config, "model_type", "")).lower():
        return "vla"
    if config is not None and looks_like_multimodal_config(config):
        return "vlm"
    vlm_markers = ("-vl", "_vl", "vl-", "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "llava", "internvl")
    if any(marker in lowered for marker in vlm_markers):
        return "vlm"
    return "text"


def looks_like_multimodal_config(config: Any) -> bool:
    if config is None:
        return False
    model_type = str(getattr(config, "model_type", "")).lower()
    if any(marker in model_type for marker in ("vl", "vision", "llava", "multimodal", "vla")):
        return True
    has_vision = any(hasattr(config, name) for name in ("vision_config", "vision_config_dict", "visual_config"))
    has_text = any(hasattr(config, name) for name in ("text_config", "language_config", "llm_config"))
    return has_vision and has_text


def load_model_bundle(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, cache_dir: Path) -> LoadedModel:
    config = None
    if args.model_kind == "auto":
        try:
            config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True, cache_dir=str(cache_dir))
        except Exception:
            config = None
    model_kind = normalize_model_kind(args.model_kind, args.model_id, config=config)

    if model_kind in {"vlm", "vla"}:
        try:
            processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, cache_dir=str(cache_dir))
        except Exception as exc:
            raise RuntimeError(
                f"failed to load {model_kind.upper()} processor for {args.model_id!r}. "
                "Install a transformers version that supports this architecture "
                "(Qwen3-VL usually needs transformers>=4.57 or a current source build)."
            ) from exc
        set_processor_pad_token(processor)
        model_classes = multimodal_model_classes(args.model_id, model_kind=model_kind)
        if not model_classes:
            raise RuntimeError(
                f"{model_kind.upper()} support requires a transformers version with AutoModelForImageTextToText "
                "or AutoModelForVision2Seq."
            )
        model = load_multimodal_model_from_candidates(model_classes, args.model_id, dtype=dtype, cache_dir=cache_dir).to(device)
        tokenizer = getattr(processor, "tokenizer", None)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, cache_dir=str(cache_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        processor = None
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
        ).to(device)

    model.eval()
    layers, layer_path = find_decoder_layers(model)
    language_config = resolve_language_config(model, layers[0])
    disable_model_cache(model, language_config)
    return LoadedModel(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        model_kind=model_kind,
        layers=layers,
        layer_path=layer_path,
        language_config=language_config,
    )


def multimodal_model_classes(model_id: str, model_kind: str) -> list[type]:
    classes: list[type] = []
    lowered = model_id.lower()
    if "qwen3-vl" in lowered:
        try:
            qwen3_vl_cls = getattr(importlib.import_module("transformers"), "Qwen3VLForConditionalGeneration", None)
        except Exception:
            qwen3_vl_cls = None
        if qwen3_vl_cls is not None:
            classes.append(qwen3_vl_cls)
    if model_kind == "vla" and "openvla" in lowered:
        try:
            openvla_cls = getattr(importlib.import_module("transformers"), "OpenVLAForActionPrediction", None)
        except Exception:
            openvla_cls = None
        if openvla_cls is not None:
            classes.append(openvla_cls)
    for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
        if cls is not None and cls not in classes:
            classes.append(cls)
    return classes


def load_multimodal_model_from_candidates(
    model_classes: list[type],
    model_id: str,
    dtype: torch.dtype,
    cache_dir: Path,
) -> nn.Module:
    errors = []
    for model_cls in model_classes:
        try:
            return model_cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                cache_dir=str(cache_dir),
            )
        except Exception as exc:
            errors.append(f"{model_cls.__name__}: {type(exc).__name__}: {exc}")
    joined = " | ".join(errors)
    raise RuntimeError(f"failed to load multimodal model {model_id!r}. Tried: {joined}")


def set_processor_pad_token(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def disable_model_cache(model: nn.Module, language_config: Any) -> None:
    for config in (getattr(model, "config", None), language_config):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def get_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def is_decoder_layer_stack(candidate: Any) -> bool:
    if not isinstance(candidate, (nn.ModuleList, list, tuple)) or len(candidate) == 0:
        return False
    layer = candidate[0]
    required = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")
    return all(hasattr(layer, name) for name in required)


def find_decoder_layers(model: nn.Module) -> tuple[nn.ModuleList, str]:
    candidate_paths = (
        "model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "base_model.model.layers",
        "base_model.language_model.layers",
        "base_model.language_model.model.layers",
        "model.text_model.layers",
        "text_model.layers",
    )
    for path in candidate_paths:
        try:
            layers = get_nested_attr(model, path)
        except AttributeError:
            continue
        if is_decoder_layer_stack(layers):
            if not isinstance(layers, nn.ModuleList):
                raise TypeError(f"decoder layers at {path} must be an nn.ModuleList so candidates can be swapped")
            return layers, path

    matches = []
    for module_name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if is_decoder_layer_stack(layers):
            matches.append((f"{module_name}.layers" if module_name else "layers", layers))
    if matches:
        path, layers = matches[0]
        if not isinstance(layers, nn.ModuleList):
            raise TypeError(f"decoder layers at {path} must be an nn.ModuleList so candidates can be swapped")
        return layers, path
    raise RuntimeError("could not locate a Qwen-style language decoder layer stack on the loaded model")


def resolve_language_config(model: nn.Module, sample_layer: nn.Module) -> Any:
    config_candidates = [getattr(model, "config", None)]
    root_config = getattr(model, "config", None)
    if root_config is not None:
        for attr in ("text_config", "language_config", "llm_config", "decoder_config"):
            if hasattr(root_config, attr):
                config_candidates.append(getattr(root_config, attr))

    for path in (
        "language_model.config",
        "language_model.model.config",
        "model.language_model.config",
        "model.language_model.model.config",
        "model.config",
        "text_model.config",
        "model.text_model.config",
    ):
        try:
            config_candidates.append(get_nested_attr(model, path))
        except AttributeError:
            continue

    for config in config_candidates:
        config = config_to_namespace(config)
        if has_qwen_attention_config(config):
            return config

    inferred = infer_config_from_qwen_layer(sample_layer)
    if root_config is not None:
        for name in ("rope_theta", "max_position_embeddings"):
            if hasattr(root_config, name) and not hasattr(inferred, name):
                setattr(inferred, name, getattr(root_config, name))
    return inferred


def config_to_namespace(config: Any) -> Any:
    if isinstance(config, dict):
        return SimpleNamespace(**config)
    return config


def has_qwen_attention_config(config: Any) -> bool:
    if config is None:
        return False
    has_hidden = hasattr(config, "hidden_size")
    has_heads = hasattr(config, "num_attention_heads") or hasattr(config, "num_heads")
    return has_hidden and has_heads


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        elif isinstance(value, (list, tuple)):
            moved[key] = type(value)(item.to(device) if isinstance(item, torch.Tensor) else item for item in value)
        else:
            moved[key] = value
    return moved


def normalize_processor_output(output: Any) -> dict[str, Any]:
    if isinstance(output, torch.Tensor):
        return {"input_ids": output}
    if hasattr(output, "data") and isinstance(output.data, dict):
        output = output.data
    if not isinstance(output, dict):
        raise TypeError(f"processor returned unsupported type: {type(output)!r}")
    return dict(output)


def load_multimodal_image(image: Any | None, size: int):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Multimodal image inputs require Pillow. Install pillow in the active environment.") from exc

    if image is None:
        return Image.new("RGB", (size, size), color=(255, 255, 255))
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(image))).convert("RGB")
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, dict):
        return load_multimodal_image(normalize_image_value(image), size=size)
    if isinstance(image, (list, tuple)):
        return load_multimodal_image(image[0] if image else None, size=size)
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu()
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            array = array.permute(1, 2, 0)
        if array.dtype.is_floating_point:
            array = array.clamp(0, 1).mul(255)
        return Image.fromarray(array.to(torch.uint8).numpy()).convert("RGB")
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        image_path = Path(image)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        return Image.open(image_path).convert("RGB")
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def numeric_tensor_from_value(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            return value.detach().float()
        return value.detach()
    if isinstance(value, (int, float)):
        return torch.tensor([value], dtype=torch.float32)
    if isinstance(value, dict):
        values = []
        for item in value.values():
            tensor = numeric_tensor_from_value(item)
            if tensor is not None and tensor.ndim <= 1:
                values.append(tensor.flatten())
        if values:
            return torch.cat(values).float()
        return None
    if isinstance(value, (list, tuple)):
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError):
            flattened = []
            for item in value:
                tensor = numeric_tensor_from_value(item)
                if tensor is not None:
                    flattened.append(tensor.flatten())
            return torch.cat(flattened).float() if flattened else None
        if tensor.numel() == 0:
            return None
        return tensor.float()
    return None


def split_image_paths(raw: str | None) -> list[str | None]:
    if raw is None or not raw.strip():
        return [None]
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if normalized == "gpu":
            raise RuntimeError("requested GPU device, but no CUDA/MPS GPU is available")
        return torch.device("cpu")
    return torch.device(device)


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name.lower()]


def dtype_nbytes(dtype: torch.dtype) -> int:
    return {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
    }.get(dtype, 2)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def quantized_memory_bytes(module: nn.Module, dtype_bytes: int) -> int:
    total = 0
    for submodule in module.modules():
        if hasattr(submodule, "quantized_memory_bytes") and not list(submodule.children()):
            total += int(submodule.quantized_memory_bytes(dtype_bytes))
    return total


def effective_parameter_elements(module: nn.Module) -> int:
    quantized = 0
    for submodule in module.modules():
        weight_q = getattr(submodule, "weight_q", None)
        if isinstance(weight_q, torch.Tensor):
            quantized += weight_q.numel()
    return count_parameters(module) + quantized


def module_param_count(module: nn.Module, name: str) -> int:
    return count_parameters(getattr(module, name)) if hasattr(module, name) else 0


def effective_param_count(layer: nn.Module, variant: str) -> int:
    total = count_parameters(layer)
    attn = module_param_count(layer, "self_attn")
    mlp = module_param_count(layer, "mlp")
    if variant in {"parent", "parent_attn"}:
        return total
    if is_qwen_attention_variant(variant):
        if variant == "noop_attn":
            return total - attn
        return total - attn + qwen_attention_param_count(layer, variant)
    if is_fla_variant(variant):
        return total - attn + fla_attention_param_count(layer, variant)
    if variant == "skip_attn":
        return total - attn
    if variant == "skip_mlp":
        return total - mlp
    if variant == "skip_both":
        return total - attn - mlp
    raise ValueError(f"unknown variant: {variant}")


def qwen_attention_param_count(layer: nn.Module, variant: str) -> int:
    attention = make_qwen_attention_variant(layer.self_attn, infer_config_from_qwen_layer(layer), variant)
    return effective_parameter_elements(attention)


def effective_param_memory_bytes(layer: nn.Module, variant: str, dtype_bytes: int) -> float:
    total = count_parameters(layer)
    attn = module_param_count(layer, "self_attn")
    if is_qwen_attention_variant(variant) and variant not in {"parent_attn", "noop_attn"}:
        attention = make_qwen_attention_variant(layer.self_attn, infer_config_from_qwen_layer(layer), variant)
        return float((total - attn) * dtype_bytes + count_parameters(attention) * dtype_bytes + quantized_memory_bytes(attention, dtype_bytes))
    return float(effective_param_count(layer, variant) * dtype_bytes)


def fla_attention_param_count(layer: nn.Module, variant: str) -> int:
    config = infer_config_from_qwen_layer(layer)
    attention = make_fla_attention_variant(
        layer,
        config,
        variant=variant,
        layer_idx=getattr(layer.self_attn, "layer_idx", None),
        mode="chunk",
        feature_map="elu",
    )
    return count_parameters(attention)


def infer_config_from_qwen_layer(layer: nn.Module):
    parent_attn = layer.self_attn
    hidden_size = parent_attn.q_proj.in_features
    num_heads = getattr(parent_attn, "num_heads", None) or getattr(parent_attn, "num_attention_heads", None)
    head_dim = getattr(parent_attn, "head_dim", None)
    if num_heads is None:
        if head_dim is None:
            raise ValueError("cannot infer num_heads from Qwen self_attn")
        num_heads = parent_attn.q_proj.out_features // head_dim
    if head_dim is None:
        head_dim = hidden_size // num_heads
    num_kv_heads = getattr(parent_attn, "num_key_value_heads", None) or getattr(parent_attn, "num_kv_heads", None)
    if num_kv_heads is None:
        num_kv_heads = parent_attn.k_proj.out_features // head_dim
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rope_theta=10000.0,
        max_position_embeddings=None,
    )


def kv_cache_bytes(config, variant: str, seq_len: int, dtype_bytes: int) -> int:
    if variant in {"skip_attn", "skip_both", "noop_attn", "linear_attn"} or is_fla_variant(variant):
        return 0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    if variant in {"mha_attn", "quant_mha_attn"}:
        return 2 * config.num_attention_heads * head_dim * seq_len * dtype_bytes
    if variant == "mqa_attn":
        return 2 * head_dim * seq_len * dtype_bytes
    if variant.startswith("gqa_kv"):
        return 2 * parse_kv_suffix(variant) * head_dim * seq_len * dtype_bytes
    if variant.startswith("mfa_kv"):
        return 2 * parse_kv_suffix(variant) * max(1, head_dim // 2) * seq_len * dtype_bytes
    if variant.startswith("mla_kv"):
        return max(1, config.hidden_size // 4) * seq_len * dtype_bytes
    if variant == "mka_attn":
        return (head_dim + max(1, config.hidden_size // 4)) * seq_len * dtype_bytes
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    return 2 * num_kv_heads * head_dim * seq_len * dtype_bytes


def runtime_proxy(layer: nn.Module, config, variant: str, seq_len: int, dtype_bytes: int) -> float:
    params = effective_param_count(layer, variant)
    kv = kv_cache_bytes(config, variant, seq_len, dtype_bytes=1)
    return float(params * seq_len + kv) / 1e12


def make_text_batches(tokenizer, examples: list[TaskExample], seq_len: int, device: torch.device):
    encoded = []
    for example in examples:
        item = tokenizer(
            example.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=False,
        )
        encoded.append(move_batch_to_device(dict(item), device))
    return encoded


def make_multimodal_messages(prompt: str, image: Any, model_kind: str) -> list[dict[str, Any]]:
    if model_kind == "vla" and "Action:" not in prompt:
        prompt = (
            "You are a vision-language-action robot policy.\n"
            f"Instruction: {prompt}\n"
            "Predict the next robot action.\nAction:"
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def encode_multimodal_prompt(
    processor: Any,
    prompt: str,
    image: Any,
    seq_len: int,
    add_generation_prompt: bool,
    model_kind: str,
) -> dict[str, Any]:
    messages = make_multimodal_messages(prompt, image, model_kind=model_kind)
    errors: list[str] = []

    if hasattr(processor, "apply_chat_template"):
        try:
            output = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
            )
            return normalize_processor_output(output)
        except Exception as exc:
            errors.append(f"chat_template_tokenize={type(exc).__name__}: {exc}")

        try:
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:
            errors.append(f"chat_template_render={type(exc).__name__}: {exc}")
            rendered = prompt
    else:
        rendered = prompt

    processor_calls = (
        {"text": [rendered], "images": [image]},
        {"text": rendered, "images": image},
        {"text": [prompt], "images": [image]},
        {"text": prompt, "images": image},
    )
    for kwargs in processor_calls:
        try:
            output = processor(
                **kwargs,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len,
                padding=False,
            )
            return normalize_processor_output(output)
        except Exception as exc:
            errors.append(f"processor={type(exc).__name__}: {exc}")

    joined = " | ".join(errors[-4:])
    raise RuntimeError(f"failed to encode {model_kind.upper()} prompt with processor. Recent errors: {joined}")


def make_multimodal_batches(
    processor: Any,
    examples: list[TaskExample],
    seq_len: int,
    device: torch.device,
    image_path: str | None,
    image_size: int,
    add_generation_prompt: bool,
    model_kind: str,
    allow_blank_image: bool,
) -> list[dict[str, Any]]:
    image_paths = split_image_paths(image_path)
    encoded = []
    for index, example in enumerate(examples):
        image_value = example.image if example.image is not None else image_paths[index % len(image_paths)]
        if image_value is None and not allow_blank_image:
            source = (example.metadata or {}).get("source", "dataset")
            raise ValueError(
                f"{model_kind.upper()} example {index} from {source!r} has no image. "
                "Fix dataset image fields, pass --image-path, or use --allow-blank-image for smoke tests."
            )
        image = load_multimodal_image(image_value, image_size)
        batch = encode_multimodal_prompt(
            processor=processor,
            prompt=example.prompt,
            image=image,
            seq_len=seq_len,
            add_generation_prompt=add_generation_prompt,
            model_kind=model_kind,
        )
        if model_kind == "vla":
            action_tensor = numeric_tensor_from_value(example.action)
            if action_tensor is not None:
                batch["actions"] = action_tensor.unsqueeze(0) if action_tensor.ndim == 1 else action_tensor
            state_tensor = numeric_tensor_from_value(example.state)
            if state_tensor is not None:
                batch["proprio"] = state_tensor.unsqueeze(0) if state_tensor.ndim == 1 else state_tensor
        encoded.append(move_batch_to_device(batch, device))
    return encoded


def make_batches(
    loaded: LoadedModel,
    examples: list[TaskExample],
    seq_len: int,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if loaded.model_kind in {"vlm", "vla"}:
        if loaded.processor is None:
            raise RuntimeError(f"{loaded.model_kind.upper()} mode requires a processor")
        prompt_source = normalize_prompt_source(args.prompt_source)
        allow_blank_image = args.allow_blank_image or prompt_source == "built_in" or bool(args.image_path)
        return make_multimodal_batches(
            loaded.processor,
            examples,
            seq_len,
            device,
            image_path=args.image_path,
            image_size=args.vlm_blank_image_size,
            add_generation_prompt=not args.no_vlm_generation_prompt,
            model_kind=loaded.model_kind,
            allow_blank_image=allow_blank_image,
        )
    if loaded.tokenizer is None:
        raise RuntimeError("text mode requires a tokenizer")
    return make_text_batches(loaded.tokenizer, examples, seq_len, device)


def load_vlm_image(path: str | None, size: int):
    return load_multimodal_image(path, size=size)


def make_vlm_messages(prompt: str, image: Any) -> list[dict[str, Any]]:
    return make_multimodal_messages(prompt, image, model_kind="vlm")


def encode_vlm_prompt(
    processor: Any,
    prompt: str,
    image: Any,
    seq_len: int,
    add_generation_prompt: bool,
) -> dict[str, Any]:
    return encode_multimodal_prompt(
        processor=processor,
        prompt=prompt,
        image=image,
        seq_len=seq_len,
        add_generation_prompt=add_generation_prompt,
        model_kind="vlm",
    )


def logits_kl(parent_logits: torch.Tensor, child_logits: torch.Tensor) -> torch.Tensor:
    parent_probs = F.softmax(parent_logits.float(), dim=-1)
    child_log_probs = F.log_softmax(child_logits.float(), dim=-1)
    return F.kl_div(child_log_probs, parent_probs, reduction="none").sum(dim=-1).mean()


def output_value(output: Any, name: str) -> Any:
    if isinstance(output, dict):
        return output.get(name)
    if hasattr(output, name):
        return getattr(output, name)
    if hasattr(output, "get"):
        try:
            return output.get(name)
        except Exception:
            return None
    return None


def score_target_priority(model_kind: str) -> tuple[tuple[str, str], ...]:
    if model_kind == "vla":
        return (
            ("action_logits", "kl"),
            ("action_log_probs", "kl"),
            ("predicted_action_logits", "kl"),
            ("action_mean", "mse"),
            ("action_means", "mse"),
            ("predicted_action_mean", "mse"),
            ("predicted_action_means", "mse"),
            ("action_mu", "mse"),
            ("action_mus", "mse"),
            ("actions", "mse"),
            ("action", "mse"),
            ("predicted_actions", "mse"),
            ("predicted_action", "mse"),
            ("action_preds", "mse"),
            ("action_pred", "mse"),
            ("logits", "kl"),
        )
    return (("logits", "kl"),)


def extract_score_target(output: Any, model_kind: str, preferred_name: str | None = None) -> ScoreTarget:
    if isinstance(output, torch.Tensor):
        metric = "kl" if output.ndim >= 2 else "mse"
        return ScoreTarget("tensor", output, metric)
    priorities = score_target_priority(model_kind)
    if preferred_name is not None:
        priorities = tuple(item for item in priorities if item[0] == preferred_name) + tuple(
            item for item in priorities if item[0] != preferred_name
        )
    for name, metric in priorities:
        value = output_value(output, name)
        if isinstance(value, torch.Tensor):
            return ScoreTarget(name, value, metric)
    if isinstance(output, (tuple, list)):
        for index, value in enumerate(output):
            if isinstance(value, torch.Tensor):
                metric = "kl" if value.ndim >= 2 else "mse"
                return ScoreTarget(f"tuple_{index}", value, metric)
    raise RuntimeError(
        f"model output does not expose a score target for model_kind={model_kind!r}. "
        "Expected logits/action_logits/action tensors."
    )


def score_target_distance(teacher: ScoreTarget, student_output: Any, model_kind: str) -> torch.Tensor:
    student = extract_score_target(student_output, model_kind=model_kind, preferred_name=teacher.name)
    teacher_tensor = teacher.tensor.to(device=student.tensor.device)
    if teacher_tensor.shape != student.tensor.shape:
        raise RuntimeError(
            f"teacher/student score target shape mismatch for {teacher.name}: "
            f"{tuple(teacher_tensor.shape)} vs {tuple(student.tensor.shape)}"
        )
    if teacher.metric == "kl":
        return logits_kl(teacher_tensor, student.tensor)
    if teacher.metric == "mse":
        return F.mse_loss(student.tensor.float(), teacher_tensor.float())
    raise ValueError(f"unknown score metric: {teacher.metric}")


def precompute_parent_targets(model, batches, model_kind: str) -> list[ScoreTarget]:
    outputs = []
    with torch.inference_mode():
        for batch in batches:
            model_output = safe_forward_batch(model, batch, use_cache=False)
            target = extract_score_target(model_output, model_kind=model_kind)
            outputs.append(ScoreTarget(target.name, target.tensor.detach().cpu(), target.metric))
    return outputs


def precompute_parent_logits(model, batches) -> list[torch.Tensor]:
    return [target.tensor for target in precompute_parent_targets(model, batches, model_kind="text")]


def write_attention_score_outputs(result: dict[str, Any], output_path: Path) -> Path:
    score_dir = output_path.parent / f"{output_path.stem}_attention_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    scores_by_variant: dict[str, list[dict[str, Any]]] = {}
    for score in result.get("scores", []):
        scores_by_variant.setdefault(score["variant"], []).append(score)
    skipped = result.get("fla", {}).get("skipped_variants", {})
    selected = result.get("solution", {}).get("selected", [])
    summary_rows = []

    for variant in result.get("variants", []):
        scores = scores_by_variant.get(variant, [])
        skipped_reason = skipped.get(variant)
        selected_names = [name for name in selected if name.rsplit(":", 1)[-1] == variant]
        status = "scored" if scores else "skipped" if skipped_reason else "not_run"
        variant_result = {
            "model_id": result.get("model_id"),
            "prompt_source": result.get("prompt_source"),
            "seq_len": result.get("seq_len"),
            "searched_layers": result.get("searched_layers"),
            "variant": variant,
            "status": status,
            "score_count": len(scores),
            "selected": selected_names,
            "skipped_reason": skipped_reason,
            "scores": scores,
        }
        (score_dir / f"{variant}.json").write_text(json.dumps(variant_result, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "variant": variant,
                "status": status,
                "score_count": len(scores),
                "selected": ";".join(selected_names),
                "metric": "" if not scores else scores[0].get("metric", "kl"),
                "mean_score": "" if not scores else sum(score["kl"] for score in scores) / len(scores),
                "mean_kl": "" if not scores else sum(score["kl"] for score in scores) / len(scores),
                "skipped_reason": skipped_reason or "",
            }
        )

    (score_dir / "_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    with (score_dir / "_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "status",
                "score_count",
                "selected",
                "metric",
                "mean_score",
                "mean_kl",
                "skipped_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return score_dir


def score_variant(
    model,
    layers,
    layer_idx: int,
    variant: str,
    batches,
    parent_targets,
    model_kind: str,
    language_config,
    fla_mode: str,
    fla_feature_map: str,
) -> tuple[float, float]:
    original_layer = layers[layer_idx]
    layers[layer_idx] = QwenCandidateLayer(
        original_layer,
        variant,
        config=language_config,
        layer_idx=layer_idx,
        fla_mode=fla_mode,
        fla_feature_map=fla_feature_map,
    )
    values = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch, teacher_target in zip(batches, parent_targets, strict=True):
                child_output = safe_forward_batch(model, batch, use_cache=False)
                score = score_target_distance(teacher_target, child_output, model_kind=model_kind)
                values.append(float(score.detach().cpu()))
    finally:
        layers[layer_idx] = original_layer
    return sum(values) / len(values), time.perf_counter() - started


def iter_layer_indices(num_layers: int, max_layers: int, layer_stride: int) -> list[int]:
    indices = list(range(0, num_layers, max(layer_stride, 1)))
    if max_layers > 0:
        indices = indices[:max_layers]
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a NAS candidate search on Qwen-style LLM, VLM, or VLA models.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--model-kind",
        default="auto",
        choices=["auto", "text", "vlm", "vla"],
        help="Model loading path. auto detects common text, VLM, and VLA model ids.",
    )
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE / "models"))
    parser.add_argument(
        "--prompt-source",
        default="built_in",
        help=(
            "Prompt/data source: built_in, dataset, or aliases such as "
            "mmlu/hellaswag/arc_challenge/gsm8k/boolq/vqav2/textvqa/scienceqa/libero/bridge_v2/rt1."
        ),
    )
    parser.add_argument("--list-dataset-aliases", action="store_true", help="Print built-in dataset aliases and exit.")
    parser.add_argument("--mmlu-dataset", default="cais/mmlu")
    parser.add_argument("--mmlu-subject", default="abstract_algebra")
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--dataset-name", default=None, help="HuggingFace dataset id for --prompt-source dataset.")
    parser.add_argument("--dataset-config", default=None, help="Optional HuggingFace dataset config/subset.")
    parser.add_argument("--dataset-split", default=None, help="Dataset split for generic dataset/local sources.")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Local JSON/JSONL/CSV/Parquet file for --prompt-source dataset.",
    )
    parser.add_argument(
        "--dataset-task",
        default="auto",
        choices=["auto", "llm", "vlm", "vla"],
        help="How to format generic dataset examples.",
    )
    parser.add_argument(
        "--dataset-image-root",
        default=None,
        help="Base directory for relative image paths in VLM/VLA datasets.",
    )
    parser.add_argument(
        "--include-dataset-target",
        action="store_true",
        help="Append answer/action target text to formatted prompts when a dataset provides it.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional image path, or comma-separated image paths, for VLM/VLA prompts.",
    )
    parser.add_argument("--vlm-blank-image-size", type=int, default=224)
    parser.add_argument(
        "--allow-blank-image",
        action="store_true",
        help="Allow blank fallback images for VLM/VLA dataset examples with no image field.",
    )
    parser.add_argument(
        "--no-vlm-generation-prompt",
        action="store_true",
        help="Do not append the assistant generation prompt when applying VLM chat templates.",
    )
    parser.add_argument("--dataset-cache-dir", default=str(DEFAULT_HF_CACHE / "datasets"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=8, help="0 means all layers.")
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument(
        "--variants",
        default=DEFAULT_VARIANTS,
        help="Comma-separated variants. Use all_qwen_attn, all_linear_attn, all_core_attn, all_fla, or all_attention aliases.",
    )
    parser.add_argument("--fla-mode", default="chunk", help="Default mode passed to FLA attention layers.")
    parser.add_argument("--fla-feature-map", default="elu", help="Default feature map for compatible FLA layers.")
    parser.add_argument(
        "--no-skip-unavailable-fla",
        action="store_true",
        help="Fail instead of skipping FLA variants missing from the installed FLA version or local CUDA deps.",
    )
    parser.add_argument("--target-param-fraction", type=float, default=0.86)
    parser.add_argument("--target-runtime-fraction", type=float, default=0.86)
    parser.add_argument("--output", default="outputs/qwen3_0_6b_layer_skip_search.json")
    parser.add_argument("--pth-output", default="checkpoints/qwen3_0_6b_layer_skip_search.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_dataset_aliases:
        aliases = {
            name: {
                "task": task,
                "dataset": dataset_name,
                "config": config_name,
                "split": split,
            }
            for name, (task, dataset_name, config_name, split) in sorted(COMMON_DATASET_ALIASES.items())
        }
        print(json.dumps(aliases, indent=2, ensure_ascii=False))
        return
    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype, device)
    torch.set_grad_enabled(False)

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model_bundle(args, device=device, dtype=dtype, cache_dir=cache_dir)
    model = loaded.model
    layers = loaded.layers
    language_config = loaded.language_config
    layer_indices = iter_layer_indices(len(layers), args.max_layers, args.layer_stride)
    examples, prompt_metadata = load_prompts_from_args(args, loaded.model_kind)
    batches = make_batches(loaded, examples, args.seq_len, device, args)
    parent_targets = precompute_parent_targets(model, batches, model_kind=loaded.model_kind)

    dtype_bytes = dtype_nbytes(dtype)
    variants = expand_variants(args.variants)
    candidates_by_layer: list[list[SearchCandidate]] = []
    raw_scores: list[VariantScore] = []
    skipped_variants: dict[str, str] = {}

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        layer_candidates: list[SearchCandidate] = []
        for variant in variants:
            if variant in skipped_variants:
                continue
            try:
                kl, measured_seconds = score_variant(
                    model,
                    layers,
                    layer_idx,
                    variant,
                    batches,
                    parent_targets,
                    model_kind=loaded.model_kind,
                    language_config=language_config,
                    fla_mode=args.fla_mode,
                    fla_feature_map=args.fla_feature_map,
                )
                param_memory = effective_param_memory_bytes(layer, variant, dtype_bytes)
                kv_memory = kv_cache_bytes(language_config, variant, args.seq_len, dtype_bytes)
                proxy = runtime_proxy(layer, language_config, variant, args.seq_len, dtype_bytes)
            except Exception as exc:
                if is_fla_variant(variant) and not args.no_skip_unavailable_fla:
                    skipped_variants[variant] = f"{type(exc).__name__}: {exc}"
                    print(f"skipped_variant={variant} reason={skipped_variants[variant]}", file=sys.stderr)
                    continue
                raise
            name = f"L{layer_idx}:{variant}"
            raw_scores.append(
                VariantScore(
                    layer_idx=layer_idx,
                    variant=variant,
                    kl=kl,
                    metric=parent_targets[0].metric,
                    target_name=parent_targets[0].name,
                    effective_param_memory_bytes=param_memory,
                    kv_cache_memory_bytes=kv_memory,
                    runtime_proxy=proxy,
                    measured_seconds=measured_seconds,
                )
            )
            layer_candidates.append(
                SearchCandidate(
                    layer_idx=layer_idx,
                    name=name,
                    score=kl,
                    param_memory=param_memory,
                    kv_cache_memory=kv_memory,
                    runtimes={1: proxy},
                    payload={"variant": variant},
                )
            )
        candidates_by_layer.append(layer_candidates)

    parent_candidates = []
    for layer in candidates_by_layer:
        for candidate in layer:
            if candidate.payload.get("variant") == "parent":
                parent_candidates.append(candidate)
                break
        else:
            raise RuntimeError("each searched layer must include a parent candidate")
    parent_memory = sum(candidate.param_memory + candidate.kv_cache_memory for candidate in parent_candidates)
    parent_runtime = sum(candidate.runtimes[1] for candidate in parent_candidates)
    constraints = SearchConstraints(
        seq_len=args.seq_len,
        batch_sizes=[1],
        memory_max=parent_memory * args.target_param_fraction,
        latency_max=parent_runtime * args.target_runtime_fraction,
        score_direction="minimize",
    )
    solution = solve_nas_mip(candidates_by_layer, constraints)

    result = {
        "model_id": args.model_id,
        "model_kind": loaded.model_kind,
        "decoder_layer_path": loaded.layer_path,
        "device": str(device),
        "dtype": str(dtype),
        "root": str(ROOT),
        "cache_dir": str(cache_dir),
        "vendor_dir": str(VENDOR),
        "fla_repo_dir": str(FLA_REPO),
        "num_model_layers": len(layers),
        "searched_layers": layer_indices,
        "variants": variants,
        "fla": {
            "mode": args.fla_mode,
            "feature_map": args.fla_feature_map,
            "registered_variants": list(FLA_VARIANT_TO_CLASS),
            "skipped_variants": skipped_variants,
        },
        "num_prompts": len(examples),
        "score_target": {
            "name": parent_targets[0].name if parent_targets else None,
            "metric": parent_targets[0].metric if parent_targets else None,
        },
        "prompt_source": prompt_metadata,
        "multimodal": {
            "image_path": args.image_path,
            "blank_image_size": args.vlm_blank_image_size,
            "add_generation_prompt": not args.no_vlm_generation_prompt,
            "batch_keys": sorted({key for batch in batches for key in batch}),
        },
        "vlm": {
            "image_path": args.image_path,
            "blank_image_size": args.vlm_blank_image_size,
            "add_generation_prompt": not args.no_vlm_generation_prompt,
            "batch_keys": sorted({key for batch in batches for key in batch}),
        },
        "seq_len": args.seq_len,
        "constraints": {
            "memory_max": constraints.memory_max,
            "latency_max": constraints.latency_max,
            "target_param_fraction": args.target_param_fraction,
            "target_runtime_fraction": args.target_runtime_fraction,
        },
        "solution": {
            "selected_batch_size": solution.batch_size,
            "total_kl_score": solution.total_score,
            "total_memory_bytes": solution.total_memory,
            "total_runtime_proxy": solution.total_runtime,
            "throughput_proxy": solution.throughput,
            "selected": solution.selected_names,
        },
        "scores": [asdict(score) for score in raw_scores],
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result["attention_score_dir"] = str(output.parent / f"{output.stem}_attention_scores")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    attention_score_dir = write_attention_score_outputs(result, output)

    pth_output = Path(args.pth_output)
    if not pth_output.is_absolute():
        pth_output = ROOT / pth_output
    pth_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, pth_output)

    print(json.dumps(result["solution"], indent=2))
    print(f"wrote={output}")
    print(f"wrote_attention_scores={attention_score_dir}")
    print(f"wrote_pth={pth_output}")


if __name__ == "__main__":
    main()
