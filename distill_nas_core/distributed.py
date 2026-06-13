from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class DevicePlan:
    teacher_device: torch.device
    student_device: torch.device
    gradient_accumulation_steps: int = 1
    use_accelerate: bool = False


DevicePolicy = Callable[[dict[str, Any]], DevicePlan]
DEVICE_POLICIES: dict[str, DevicePolicy] = {}


def resolve_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if normalized == "gpu":
            raise RuntimeError("requested GPU device, but no CUDA/MPS device is available")
        return torch.device("cpu")
    return torch.device(name)


def register_device_policy(name: str, policy: DevicePolicy) -> None:
    if not name:
        raise ValueError("device policy name must be non-empty")
    DEVICE_POLICIES[name] = policy


def get_device_policy(name: str) -> DevicePolicy:
    try:
        return DEVICE_POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown device policy: {name}") from exc


def _default_device_policy(options: dict[str, Any]) -> DevicePlan:
    base = resolve_device(str(options.get("device", "auto")))
    teacher_device = options.get("teacher_device")
    student_device = options.get("student_device")
    return DevicePlan(
        teacher_device=resolve_device(str(teacher_device)) if teacher_device else base,
        student_device=resolve_device(str(student_device)) if student_device else base,
        gradient_accumulation_steps=max(int(options.get("gradient_accumulation_steps", 1)), 1),
        use_accelerate=bool(options.get("use_accelerate", False)),
    )


def make_device_plan(
    device: str = "auto",
    teacher_device: str | None = None,
    student_device: str | None = None,
    gradient_accumulation_steps: int = 1,
    use_accelerate: bool = False,
    policy: str = "default",
) -> DevicePlan:
    return get_device_policy(policy)(
        {
            "device": device,
            "teacher_device": teacher_device,
            "student_device": student_device,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "use_accelerate": use_accelerate,
        }
    )


register_device_policy("default", _default_device_policy)
