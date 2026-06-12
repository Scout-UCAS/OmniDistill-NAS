from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .distill import forward_batch, output_value


class VlaEnv(Protocol):
    def reset(self) -> Any: ...

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]: ...


@dataclass(frozen=True)
class RolloutResult:
    total_reward: float
    steps: int
    success: bool
    info: dict[str, Any]


def extract_action(output: Any) -> torch.Tensor:
    for name in ("actions", "action", "predicted_actions", "predicted_action", "action_mean"):
        value = output_value(output, name)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError("policy output does not expose an action tensor")


def rollout_policy(
    env: VlaEnv,
    policy: torch.nn.Module,
    max_steps: int = 100,
    device: torch.device | str = "cpu",
) -> RolloutResult:
    device = torch.device(device)
    observation = env.reset()
    total_reward = 0.0
    info: dict[str, Any] = {}
    policy = policy.to(device).eval()
    for step in range(max_steps):
        batch = observation
        if isinstance(observation, torch.Tensor):
            batch = observation.to(device)
        elif isinstance(observation, dict):
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in observation.items()}
        with torch.no_grad():
            action = extract_action(forward_batch(policy, batch))
        observation, reward, done, info = env.step(action.detach().cpu())
        total_reward += float(reward)
        if done:
            return RolloutResult(total_reward=total_reward, steps=step + 1, success=bool(info.get("success", done)), info=info)
    return RolloutResult(total_reward=total_reward, steps=max_steps, success=bool(info.get("success", False)), info=info)
