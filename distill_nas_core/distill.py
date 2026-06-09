from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class _ModuleState:
    training: list[tuple[nn.Module, bool]]
    requires_grad: list[tuple[nn.Parameter, bool]]


def _snapshot_module_state(module: nn.Module) -> _ModuleState:
    training = [(submodule, submodule.training) for submodule in module.modules()]
    requires_grad = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    return _ModuleState(training=training, requires_grad=requires_grad)


def _restore_module_state(state: _ModuleState) -> None:
    for module, was_training in state.training:
        module.train(was_training)
    for parameter, required_grad in state.requires_grad:
        parameter.requires_grad_(required_grad)


def normalized_mse(parent_output: torch.Tensor, child_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = F.mse_loss(child_output, parent_output)
    denominator = F.mse_loss(torch.zeros_like(parent_output), parent_output).clamp_min(eps)
    return numerator / denominator


def logits_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    per_token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return per_token_kl.mean() * (temperature**2)


def move_batch_to_device(batch, device: torch.device | str):
    device = torch.device(device)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def batch_input_ids(batch) -> torch.Tensor | None:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        value = batch.get("input_ids")
        return value if isinstance(value, torch.Tensor) else None
    return None


def _extend_sequence_side_input(
    value,
    previous_input_ids: torch.Tensor | None,
    input_ids: torch.Tensor,
    name: str,
):
    if not isinstance(value, torch.Tensor) or previous_input_ids is None:
        return value
    if tuple(value.shape) != tuple(previous_input_ids.shape):
        return value
    extension_len = input_ids.shape[1] - value.shape[1]
    if extension_len <= 0:
        return value
    if name == "attention_mask":
        extension = torch.ones(
            value.shape[0],
            extension_len,
            device=value.device,
            dtype=value.dtype,
        )
    elif name == "position_ids":
        offsets = torch.arange(1, extension_len + 1, device=value.device, dtype=value.dtype).view(1, -1)
        extension = value[:, -1:] + offsets
    else:
        extension = value[:, -1:].expand(-1, extension_len)
    return torch.cat([value, extension], dim=1)


def batch_with_input_ids(batch, input_ids: torch.Tensor, drop_labels: bool = False):
    if isinstance(batch, torch.Tensor):
        return input_ids
    if not isinstance(batch, dict):
        raise TypeError(f"expected tensor or dict batch, got {type(batch)!r}")
    previous_input_ids = batch_input_ids(batch)
    updated = {}
    for key, value in batch.items():
        if drop_labels and key in {"label", "labels"}:
            continue
        if key == "input_ids":
            updated[key] = input_ids
        elif key in {"attention_mask", "position_ids", "token_type_ids"}:
            updated[key] = _extend_sequence_side_input(value, previous_input_ids, input_ids, key)
        else:
            updated[key] = value
    if "input_ids" not in updated:
        updated["input_ids"] = input_ids
    return updated


def forward_batch(model: nn.Module, batch, **kwargs):
    if isinstance(batch, dict):
        merged = dict(batch)
        merged.update(kwargs)
        try:
            return model(**merged)
        except TypeError as exc:
            if not kwargs:
                raise
            try:
                return model(**batch)
            except TypeError:
                raise exc
    try:
        return model(batch, **kwargs)
    except TypeError as exc:
        if not kwargs:
            raise
        try:
            return model(batch)
        except TypeError:
            raise exc


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


def output_tensor(output: Any, name: str, context: str) -> torch.Tensor:
    value = output_value(output, name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{context} requires model output field {name!r} to be a tensor")
    return value


@dataclass(frozen=True)
class DistillTarget:
    name: str
    tensor: torch.Tensor
    metric: str


ACTION_LOGIT_TARGET_NAMES = (
    "action_logits",
    "predicted_action_logits",
    "action_log_probs",
)
ACTION_VALUE_TARGET_NAMES = (
    "actions",
    "action",
    "predicted_actions",
    "predicted_action",
    "action_preds",
    "action_pred",
)
DISTILL_TARGET_PRIORITIES = (
    *((name, "kl") for name in ACTION_LOGIT_TARGET_NAMES),
    *((name, "mse") for name in ACTION_VALUE_TARGET_NAMES),
    ("logits", "kl"),
)


def extract_distill_target(output: Any, preferred_name: str | None = None) -> DistillTarget:
    if isinstance(output, torch.Tensor):
        metric = "kl" if output.ndim >= 2 else "mse"
        return DistillTarget("tensor", output, metric)
    priorities = DISTILL_TARGET_PRIORITIES
    if preferred_name is not None:
        priorities = tuple(item for item in priorities if item[0] == preferred_name) + tuple(
            item for item in priorities if item[0] != preferred_name
        )
    for name, metric in priorities:
        value = output_value(output, name)
        if isinstance(value, torch.Tensor):
            return DistillTarget(name, value, metric)
    if isinstance(output, (tuple, list)):
        for index, value in enumerate(output):
            if isinstance(value, torch.Tensor):
                metric = "kl" if value.ndim >= 2 else "mse"
                return DistillTarget(f"tuple_{index}", value, metric)
    raise RuntimeError("model output does not expose logits or action tensors for distillation")


def has_action_target(output: Any) -> bool:
    return any(isinstance(output_value(output, name), torch.Tensor) for name in ACTION_LOGIT_TARGET_NAMES + ACTION_VALUE_TARGET_NAMES)


def output_distillation_loss(
    teacher_output: Any,
    student_output: Any,
    temperature: float = 1.0,
) -> torch.Tensor:
    teacher = extract_distill_target(teacher_output)
    student = extract_distill_target(student_output, preferred_name=teacher.name)
    teacher_tensor = teacher.tensor.to(device=student.tensor.device)
    if teacher_tensor.shape != student.tensor.shape:
        raise RuntimeError(
            f"teacher/student target shape mismatch for {teacher.name}: "
            f"{tuple(teacher_tensor.shape)} vs {tuple(student.tensor.shape)}"
        )
    if teacher.metric == "kl":
        return logits_kl_loss(teacher_tensor, student.tensor, temperature=temperature)
    if teacher.metric == "mse":
        return F.mse_loss(student.tensor.float(), teacher_tensor.float())
    raise ValueError(f"unknown distillation metric: {teacher.metric}")


def sampled_reverse_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Monte Carlo reverse-KL term for tokens sampled from the student policy."""

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher_logits and student_logits must have matching shapes")
    if teacher_logits.shape[:2] != target_ids.shape:
        raise ValueError("target_ids must match the batch and sequence dimensions of logits")
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    selected_teacher = teacher_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    selected_student = student_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    per_token_reverse_kl = (selected_student - selected_teacher) * (temperature**2)
    if token_mask is None:
        return per_token_reverse_kl.mean()
    token_mask = token_mask.to(device=per_token_reverse_kl.device, dtype=per_token_reverse_kl.dtype)
    if token_mask.shape != per_token_reverse_kl.shape:
        token_mask = token_mask.expand_as(per_token_reverse_kl)
    denominator = token_mask.sum().clamp_min(1.0)
    return (per_token_reverse_kl * token_mask).sum() / denominator


def sampled_action_reverse_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Monte Carlo reverse-KL term for actions sampled from a student policy."""

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher_logits and student_logits must have matching shapes")
    if teacher_logits.ndim < 2:
        raise ValueError("action logits must have at least a batch and class/action dimension")
    flat_student = student_logits.reshape(-1, student_logits.shape[-1]) / temperature
    with torch.no_grad():
        probs = F.softmax(flat_student, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).view(*student_logits.shape[:-1])
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    selected_teacher = teacher_log_probs.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
    selected_student = student_log_probs.gather(dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
    return (selected_student - selected_teacher).mean() * (temperature**2)


def sample_on_policy_sequences(
    student: nn.Module,
    prompt_ids,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Autoregressively sample continuations from the current student policy."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prompt_input_ids = batch_input_ids(prompt_ids)
    if prompt_input_ids is None:
        raise ValueError("OPD sampling requires input_ids in the batch")
    generated = prompt_input_ids
    generated_batch = batch_with_input_ids(prompt_ids, generated, drop_labels=True)
    for _ in range(max_new_tokens):
        logits = output_tensor(forward_batch(student, generated_batch), "logits", "Token OPD sampling")[:, -1, :] / temperature
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be positive when provided")
            keep = min(top_k, logits.shape[-1])
            values, _ = torch.topk(logits, keep, dim=-1)
            threshold = values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        generated_batch = batch_with_input_ids(generated_batch, generated, drop_labels=True)
    return generated


def on_policy_distillation_loss(
    teacher: nn.Module,
    student: nn.Module,
    prompt_ids,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Sample from the student, then minimize reverse KL on generated tokens."""

    prompt_input_ids = batch_input_ids(prompt_ids)
    if prompt_input_ids is None:
        raise ValueError("OPD requires input_ids in the batch")
    prompt_len = prompt_input_ids.shape[1]
    with torch.no_grad():
        sampled_ids = sample_on_policy_sequences(
            student,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        sampled_batch = batch_with_input_ids(prompt_ids, sampled_ids, drop_labels=True)
        teacher_logits = output_tensor(forward_batch(teacher, sampled_batch), "logits", "Token OPD teacher pass")[:, :-1, :]
    student_logits = output_tensor(forward_batch(student, sampled_batch), "logits", "Token OPD student pass")[:, :-1, :]
    target_ids = sampled_ids[:, 1:]
    label_positions = torch.arange(target_ids.shape[1], device=target_ids.device).unsqueeze(0) + 1
    generated_token_mask = label_positions >= prompt_len
    return sampled_reverse_kl_loss(teacher_logits, student_logits, target_ids, generated_token_mask, temperature)


def _extract_named_tensor(output: Any, names: Sequence[str], preferred_name: str | None = None) -> DistillTarget | None:
    ordered_names = tuple(names)
    if preferred_name in ordered_names:
        ordered_names = (preferred_name,) + tuple(name for name in ordered_names if name != preferred_name)
    for name in ordered_names:
        value = output_value(output, name)
        if isinstance(value, torch.Tensor):
            metric = "kl" if name in ACTION_LOGIT_TARGET_NAMES else "mse"
            return DistillTarget(name, value, metric)
    return None


def on_policy_action_distillation_loss(
    teacher: nn.Module,
    student: nn.Module,
    batch,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Action-space OPD for VLA-style policies.

    Discrete action distributions use sampled reverse KL. Continuous action
    predictions use the student action output as the on-policy action sample and
    match the teacher action prediction on the same observation batch.
    """

    with torch.no_grad():
        teacher_output = forward_batch(teacher, batch)
    student_output = forward_batch(student, batch)

    student_logits = _extract_named_tensor(student_output, ACTION_LOGIT_TARGET_NAMES)
    if student_logits is not None:
        teacher_logits = _extract_named_tensor(teacher_output, ACTION_LOGIT_TARGET_NAMES, preferred_name=student_logits.name)
        if teacher_logits is None:
            raise RuntimeError(f"teacher output is missing action logits compatible with {student_logits.name}")
        return sampled_action_reverse_kl_loss(teacher_logits.tensor, student_logits.tensor, temperature=temperature)

    student_action = _extract_named_tensor(student_output, ACTION_VALUE_TARGET_NAMES)
    if student_action is None:
        raise RuntimeError("student output does not expose action tensors for action-space OPD")
    teacher_action = _extract_named_tensor(teacher_output, ACTION_VALUE_TARGET_NAMES, preferred_name=student_action.name)
    if teacher_action is None:
        raise RuntimeError(f"teacher output is missing action tensors compatible with {student_action.name}")
    teacher_tensor = teacher_action.tensor.to(device=student_action.tensor.device)
    if teacher_tensor.shape != student_action.tensor.shape:
        raise RuntimeError(
            f"teacher/student action shape mismatch for {student_action.name}: "
            f"{tuple(teacher_tensor.shape)} vs {tuple(student_action.tensor.shape)}"
        )
    return F.mse_loss(student_action.tensor.float(), teacher_tensor.float())


def hidden_cosine_loss(
    teacher_hidden: list[torch.Tensor] | tuple[torch.Tensor, ...],
    student_hidden: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(teacher_hidden) != len(student_hidden):
        count = min(len(teacher_hidden), len(student_hidden))
        teacher_hidden = teacher_hidden[:count]
        student_hidden = student_hidden[:count]
    losses = []
    for teacher, student in zip(teacher_hidden, student_hidden, strict=False):
        losses.append(1.0 - F.cosine_similarity(student, teacher, dim=-1).mean())
    return torch.stack(losses).mean()


def hidden_state_sequence(value: Any) -> tuple[torch.Tensor, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, torch.Tensor))
    return ()


def _cycle(iterable: Iterable[torch.Tensor]):
    while True:
        seen = False
        for item in iterable:
            seen = True
            yield item
        if not seen:
            raise ValueError("expected a non-empty re-iterable batch source")


def freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def local_distill_block(
    parent_block: nn.Module,
    child_block: nn.Module,
    hidden_batches: Iterable[torch.Tensor],
    steps: int = 100,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
    trainable_modules: Sequence[nn.Module] | None = None,
) -> list[float]:
    """Blockwise local distillation with the normalized MSE loss from Distillation NAS."""

    parent_state = _snapshot_module_state(parent_block)
    child_state = _snapshot_module_state(child_block)
    device = torch.device(device)
    losses: list[float] = []
    try:
        parent_block = parent_block.to(device)
        child_block = child_block.to(device)
        parent_block.eval()
        child_block.train()
        freeze(parent_block)
        if trainable_modules is None:
            unfreeze(child_block)
        else:
            freeze(child_block)
            for module in trainable_modules:
                unfreeze(module)

        trainable_parameters = [p for p in child_block.parameters() if p.requires_grad]
        if not trainable_parameters:
            return losses

        optimizer = torch.optim.AdamW(trainable_parameters, lr=lr)
        batches = _cycle(hidden_batches)
        for _ in range(steps):
            inputs = next(batches).to(device)
            with torch.no_grad():
                parent_output = parent_block(inputs)
            child_output = child_block(inputs)
            loss = normalized_mse(parent_output, child_output)
            if not loss.requires_grad:
                losses.append(float(loss.detach().cpu()))
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return losses
    finally:
        _restore_module_state(parent_state)
        _restore_module_state(child_state)


def global_knowledge_distillation(
    teacher: nn.Module,
    student: nn.Module,
    token_batches: Iterable[torch.Tensor],
    steps: int = 100,
    lr: float = 1e-4,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    include_lm_loss: bool = False,
    opd_weight: float = 0.0,
    opd_max_new_tokens: int = 0,
    opd_temperature: float | None = None,
    opd_top_k: int | None = None,
) -> list[float]:
    """GKD stage with optional on-policy distillation on student samples."""

    if opd_weight < 0:
        raise ValueError("opd_weight must be non-negative")
    if opd_max_new_tokens < 0:
        raise ValueError("opd_max_new_tokens must be non-negative")

    teacher_state = _snapshot_module_state(teacher)
    student_state = _snapshot_module_state(student)
    device = torch.device(device)
    opd_temperature = temperature if opd_temperature is None else opd_temperature
    losses: list[float] = []
    try:
        teacher = teacher.to(device)
        student = student.to(device)
        teacher.eval()
        student.train()
        freeze(teacher)
        unfreeze(student)
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

        batches = _cycle(token_batches)
        for _ in range(steps):
            batch = move_batch_to_device(next(batches), device)
            input_ids = batch_input_ids(batch)
            with torch.no_grad():
                teacher_out = forward_batch(teacher, batch, output_hidden_states=True)
            student_out = forward_batch(student, batch, output_hidden_states=True)

            loss = output_distillation_loss(teacher_out, student_out, temperature)
            teacher_hidden = hidden_state_sequence(output_value(teacher_out, "hidden_states"))
            student_hidden = hidden_state_sequence(output_value(student_out, "hidden_states"))
            if teacher_hidden and student_hidden:
                loss = loss + hidden_cosine_loss(teacher_hidden, student_hidden)
            if include_lm_loss:
                if input_ids is None:
                    raise ValueError("include_lm_loss requires input_ids in each batch")
                student_logits = output_value(student_out, "logits")
                if not isinstance(student_logits, torch.Tensor):
                    raise ValueError("include_lm_loss requires student logits")
                shift_logits = student_logits[:, :-1].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = loss + F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if opd_weight > 0:
                if has_action_target(student_out):
                    opd_loss = on_policy_action_distillation_loss(
                        teacher,
                        student,
                        batch,
                        temperature=opd_temperature,
                    )
                    loss = loss + opd_weight * opd_loss
                elif opd_max_new_tokens > 0:
                    if input_ids is None:
                        raise ValueError("Token OPD requires input_ids in each batch")
                    if not isinstance(output_value(student_out, "logits"), torch.Tensor):
                        raise ValueError("Token OPD requires student logits")
                    opd_loss = on_policy_distillation_loss(
                        teacher,
                        student,
                        batch,
                        max_new_tokens=opd_max_new_tokens,
                        temperature=opd_temperature,
                        top_k=opd_top_k,
                    )
                    loss = loss + opd_weight * opd_loss
                else:
                    raise ValueError(
                        "opd_weight > 0 requires action outputs, or opd_max_new_tokens > 0 for token OPD"
                    )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return losses
    finally:
        _restore_module_state(teacher_state)
        _restore_module_state(student_state)


def take_batches(iterable: Iterable[torch.Tensor], count: int) -> list[torch.Tensor]:
    return list(itertools.islice(iterable, count))
