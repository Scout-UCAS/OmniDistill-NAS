from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DevicePlan:
    teacher_device: torch.device
    student_device: torch.device
    gradient_accumulation_steps: int = 1
    use_accelerate: bool = False


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


def make_device_plan(
    device: str = "auto",
    teacher_device: str | None = None,
    student_device: str | None = None,
    gradient_accumulation_steps: int = 1,
    use_accelerate: bool = False,
) -> DevicePlan:
    base = resolve_device(device)
    return DevicePlan(
        teacher_device=resolve_device(teacher_device) if teacher_device else base,
        student_device=resolve_device(student_device) if student_device else base,
        gradient_accumulation_steps=max(int(gradient_accumulation_steps), 1),
        use_accelerate=bool(use_accelerate),
    )
