from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TaskExample:
    prompt: str
    target: Any | None = None
    image: Any | None = None
    action: Any | None = None
    state: Any | None = None
    metadata: dict[str, Any] | None = None


DatasetLoader = Callable[[dict[str, Any]], list[TaskExample]]
DATASET_ADAPTERS: dict[str, DatasetLoader] = {}


def register_dataset_adapter(name: str, loader: DatasetLoader) -> None:
    if not name:
        raise ValueError("dataset adapter name must be non-empty")
    DATASET_ADAPTERS[name] = loader


def get_dataset_adapter(name: str) -> DatasetLoader:
    try:
        return DATASET_ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset adapter: {name}") from exc


def load_examples(adapter: str, config: dict[str, Any]) -> list[TaskExample]:
    return get_dataset_adapter(adapter)(config)


def _example_from_mapping(raw: dict[str, Any]) -> TaskExample:
    prompt = raw.get("prompt") or raw.get("question") or raw.get("text") or raw.get("instruction")
    if prompt is None:
        raise ValueError("dataset row is missing a prompt/question/text/instruction field")
    metadata = {key: value for key, value in raw.items() if key not in {"prompt", "question", "text", "instruction"}}
    return TaskExample(
        prompt=str(prompt),
        target=raw.get("target") or raw.get("answer") or raw.get("label"),
        image=raw.get("image") or raw.get("image_path"),
        action=raw.get("action") or raw.get("actions"),
        state=raw.get("state") or raw.get("proprio"),
        metadata=metadata,
    )


def load_json_examples(config: dict[str, Any]) -> list[TaskExample]:
    path = Path(config["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("examples", [])
    if not isinstance(rows, list):
        raise ValueError("JSON dataset must be a list or contain an examples list")
    return [_example_from_mapping(row) for row in rows if isinstance(row, dict)]


def load_jsonl_examples(config: dict[str, Any]) -> list[TaskExample]:
    path = Path(config["path"])
    examples: list[TaskExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("JSONL rows must be objects")
        examples.append(_example_from_mapping(raw))
    return examples


def load_csv_examples(config: dict[str, Any]) -> list[TaskExample]:
    path = Path(config["path"])
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_example_from_mapping(dict(row)) for row in csv.DictReader(handle)]


def load_builtin_examples(_config: dict[str, Any]) -> list[TaskExample]:
    return [
        TaskExample("Explain neural architecture search for language models in one paragraph."),
        TaskExample("Summarize why KV-cache memory matters during autoregressive decoding."),
    ]


register_dataset_adapter("built_in", load_builtin_examples)
register_dataset_adapter("json", load_json_examples)
register_dataset_adapter("jsonl", load_jsonl_examples)
register_dataset_adapter("csv", load_csv_examples)
