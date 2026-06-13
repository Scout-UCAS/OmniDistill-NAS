from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class NoOpSubblock(nn.Module):
    """A subblock replacement that contributes zero to the residual path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class LinearSubblock(nn.Module):
    """A single linear layer with the same input and output hidden size."""

    def __init__(self, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuantizedLinear(nn.Module):
    """Symmetric per-output-channel quantized linear layer with dequantized forward."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None, num_bits: int = 8) -> None:
        super().__init__()
        if num_bits not in {4, 8}:
            raise ValueError("num_bits must be 4 or 8")
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.num_bits = num_bits
        qmax = (2 ** (num_bits - 1)) - 1
        scale = weight.detach().float().abs().amax(dim=1).clamp_min(1e-8) / qmax
        quantized = torch.round(weight.detach().float() / scale[:, None]).clamp(-qmax, qmax).to(torch.int8)
        self.weight_q: torch.Tensor
        self.scale: torch.Tensor
        self.bias: torch.Tensor | None
        self.register_buffer("weight_q", quantized)
        self.register_buffer("scale", scale)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear, num_bits: int = 8) -> "QuantizedLinear":
        return cls(linear.weight, linear.bias, num_bits=num_bits)

    def quantized_memory_bytes(self, dtype_bytes: int = 2) -> int:
        weight_bytes = self.weight_q.numel() if self.num_bits == 8 else (self.weight_q.numel() + 1) // 2
        scale_bytes = self.scale.numel() * 4
        bias_bytes = 0 if self.bias is None else self.bias.numel() * dtype_bytes
        return int(weight_bytes + scale_bytes + bias_bytes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_q.to(dtype=x.dtype) * self.scale.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
    """Tiny GQA-capable causal self-attention used by the NAS demo."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self._shape_q(self.q_proj(x))
        key = self._shape_kv(self.k_proj(x))
        value = self._shape_kv(self.v_proj(x))

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        attn = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(attn)


class FactorizedKVAttention(nn.Module):
    """MFA-style causal attention with low-rank per-group K/V projections."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        latent_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = latent_dim
        self.latent_size = num_kv_heads * latent_dim

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_down_proj: nn.Linear = nn.Linear(hidden_size, self.latent_size, bias=False)
        self.k_up_proj: nn.Linear = nn.Linear(self.latent_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_down_proj: nn.Linear = nn.Linear(hidden_size, self.latent_size, bias=False)
        self.v_up_proj: nn.Linear = nn.Linear(self.latent_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def kv_cache_elements(self, seq_len: int) -> int:
        return 2 * self.num_kv_heads * self.latent_dim * seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self._shape_q(self.q_proj(x))
        key = self._shape_kv(self.k_up_proj(self.k_down_proj(x)))
        value = self._shape_kv(self.v_up_proj(self.v_down_proj(x)))

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        attn = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(attn)


class LatentKVAttention(nn.Module):
    """MLA-style causal attention that reconstructs K/V from one shared latent."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        latent_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = latent_dim

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.kv_down_proj: nn.Linear = nn.Linear(hidden_size, latent_dim, bias=False)
        self.k_up_proj: nn.Linear = nn.Linear(latent_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_up_proj: nn.Linear = nn.Linear(latent_dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def kv_cache_elements(self, seq_len: int) -> int:
        return self.latent_dim * seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self._shape_q(self.q_proj(x))
        latent = self.kv_down_proj(x)
        key = self._shape_kv(self.k_up_proj(latent))
        value = self._shape_kv(self.v_up_proj(latent))

        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        attn = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(attn)


class MultiKeyAttention(nn.Module):
    """MKA-style causal attention with one shared key and low-rank values."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        value_rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if value_rank < 1:
            raise ValueError("value_rank must be positive")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = hidden_size // num_heads
        self.value_rank = value_rank

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.v_down_proj: nn.Linear = nn.Linear(hidden_size, value_rank, bias=False)
        self.v_up_proj: nn.Linear = nn.Linear(value_rank, hidden_size, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def kv_cache_elements(self, seq_len: int) -> int:
        return (self.head_dim + self.value_rank) * seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self._shape_q(self.q_proj(x))
        key = self.k_proj(x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        key = key.repeat_interleave(self.num_heads, dim=1)
        value = self.v_up_proj(self.v_down_proj(x))
        value = value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(attn)


class QuantizedMHAAttention(nn.Module):
    """MHA candidate with quantized q/k/v/o projections."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_proj: QuantizedLinear,
        k_proj: QuantizedLinear,
        v_proj: QuantizedLinear,
        o_proj: QuantizedLinear,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj: QuantizedLinear = q_proj
        self.k_proj: QuantizedLinear = k_proj
        self.v_proj: QuantizedLinear = v_proj
        self.o_proj: QuantizedLinear = o_proj

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def quantized_memory_bytes(self, dtype_bytes: int = 2) -> int:
        return sum(module.quantized_memory_bytes(dtype_bytes) for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self._shape(self.q_proj(x))
        key = self._shape(self.k_proj(x))
        value = self._shape(self.v_proj(x))
        attn = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(attn)


class KernelLinearAttention(nn.Module):
    """Causal linear-attention family used for generic FLA-style candidates."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_map: str = "elu",
        output_gate: bool = False,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.feature_map = feature_map
        self.output_gate = output_gate
        self.eps = eps

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.g_proj: nn.Linear | None = None
        if output_gate:
            self.g_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_map == "elu":
            return F.elu(x) + 1.0
        if self.feature_map == "relu":
            return F.relu(x) + self.eps
        if self.feature_map == "relu_squared":
            return F.relu(x).square() + self.eps
        if self.feature_map == "silu_elu":
            return F.silu(x) + F.elu(x) + 2.0
        raise ValueError(f"unknown feature map: {self.feature_map}")

    def kv_cache_elements(self, seq_len: int) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._feature_map(self._shape(self.q_proj(x)))
        key = self._feature_map(self._shape(self.k_proj(x)))
        value = self._shape(self.v_proj(x))

        kv = torch.einsum("bhtd,bhte->bhtde", key, value).cumsum(dim=2)
        key_sum = key.cumsum(dim=2)
        numerator = torch.einsum("bhtd,bhtde->bhte", query, kv)
        denominator = torch.einsum("bhtd,bhtd->bht", query, key_sum).clamp_min(self.eps)
        output = numerator / denominator.unsqueeze(-1)
        batch, seq_len, _ = x.shape
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        output = self.o_proj(output)
        if self.output_gate:
            assert self.g_proj is not None
            output = output * torch.sigmoid(self.g_proj(x))
        return output


class MultiScaleRetentionAttention(nn.Module):
    """Small retention-style causal attention with one decay per head."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        decays = torch.linspace(0.85, 0.99, num_heads).log().view(1, num_heads, 1, 1)
        self.log_decay: torch.Tensor
        self.register_buffer("log_decay", decays, persistent=False)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def kv_cache_elements(self, seq_len: int) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._shape(self.q_proj(x))
        key = self._shape(self.k_proj(x))
        value = self._shape(self.v_proj(x))
        seq_len = x.shape[1]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        positions = torch.arange(seq_len, device=x.device)
        distance = (positions[:, None] - positions[None, :]).clamp_min(0).to(dtype=scores.dtype)
        causal_mask = positions[:, None] >= positions[None, :]
        scores = scores + distance.view(1, 1, seq_len, seq_len) * self.log_decay.to(dtype=scores.dtype)
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        batch, _, _ = x.shape
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(output)


class LocalSparseAttention(nn.Module):
    """Sliding-window sparse causal attention for generic NSA/MoBA candidates."""

    def __init__(self, hidden_size: int, num_heads: int, window_size: int, bias: bool = True) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj: nn.Linear = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def kv_cache_elements(self, seq_len: int) -> int:
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._shape(self.q_proj(x))
        key = self._shape(self.k_proj(x))
        value = self._shape(self.v_proj(x))
        seq_len = x.shape[1]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        positions = torch.arange(seq_len, device=x.device)
        distance = positions[:, None] - positions[None, :]
        mask = (distance >= 0) & (distance < self.window_size)
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        batch, _, _ = x.shape
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(output)


class SwiGLUFFN(nn.Module):
    """Llama-style gated FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj: nn.Linear = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj: nn.Linear = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj: nn.Linear = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def intermediate(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.intermediate(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with independently replaceable subblocks."""

    def __init__(
        self,
        hidden_size: int,
        attention: nn.Module,
        ffn: nn.Module,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ln_1: nn.LayerNorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attention: nn.Module = attention
        self.ln_2: nn.LayerNorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.ffn: nn.Module = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


@dataclass(frozen=True)
class BlockConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int


def infer_block_config(block: TransformerBlock) -> BlockConfig:
    if not isinstance(block.attention, CausalSelfAttention):
        raise TypeError("block.attention must be CausalSelfAttention")
    if not isinstance(block.ffn, SwiGLUFFN):
        raise TypeError("block.ffn must be SwiGLUFFN")
    return BlockConfig(
        hidden_size=block.hidden_size,
        num_heads=block.attention.num_heads,
        num_kv_heads=block.attention.num_kv_heads,
        intermediate_size=block.ffn.intermediate_size,
    )


def make_parent_block(
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    num_kv_heads: int | None = None,
) -> TransformerBlock:
    attention = CausalSelfAttention(hidden_size, num_heads, num_kv_heads)
    ffn = SwiGLUFFN(hidden_size, intermediate_size)
    return TransformerBlock(hidden_size, attention, ffn)


def copy_layer_norms(source: TransformerBlock, target: TransformerBlock) -> None:
    target.ln_1.load_state_dict(source.ln_1.state_dict())
    target.ln_2.load_state_dict(source.ln_2.state_dict())


def expanded_value_projection(attention: CausalSelfAttention) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return a value projection expanded to one output channel per query head."""

    repeat = attention.num_heads // attention.num_kv_heads
    weight = attention.v_proj.weight.detach()
    bias = attention.v_proj.bias.detach() if attention.v_proj.bias is not None else None
    if repeat == 1:
        return weight, bias

    head_dim = attention.head_dim
    weight = weight.view(attention.num_kv_heads, head_dim, attention.hidden_size)
    weight = weight.repeat_interleave(repeat, dim=0).reshape(attention.hidden_size, attention.hidden_size)
    if bias is not None:
        bias = bias.view(attention.num_kv_heads, head_dim)
        bias = bias.repeat_interleave(repeat, dim=0).reshape(attention.hidden_size)
    return weight, bias


def linear_from_attention(attention: CausalSelfAttention) -> LinearSubblock:
    """Initialize linear attention as W_o W_v, matching self-only attention."""

    hidden_size = attention.hidden_size
    device = attention.o_proj.weight.device
    dtype = attention.o_proj.weight.dtype
    linear = LinearSubblock(hidden_size, bias=attention.o_proj.bias is not None).to(device=device, dtype=dtype)
    value_weight, value_bias = expanded_value_projection(attention)
    with torch.no_grad():
        linear.linear.weight.copy_(attention.o_proj.weight @ value_weight)
        if linear.linear.bias is not None:
            bias = torch.zeros(hidden_size, device=device, dtype=dtype)
            if value_bias is not None:
                bias = bias + attention.o_proj.weight @ value_bias
            if attention.o_proj.bias is not None:
                bias = bias + attention.o_proj.bias
            linear.linear.bias.copy_(bias)
    return linear


def linear_from_ffn(ffn: SwiGLUFFN) -> LinearSubblock:
    """Initialize linear FFN as W_down W_up, ignoring the gate as in the paper."""

    device = ffn.down_proj.weight.device
    dtype = ffn.down_proj.weight.dtype
    linear = LinearSubblock(ffn.hidden_size, bias=ffn.down_proj.bias is not None).to(device=device, dtype=dtype)
    with torch.no_grad():
        linear.linear.weight.copy_(ffn.down_proj.weight @ ffn.up_proj.weight)
        if linear.linear.bias is not None:
            bias = torch.zeros(ffn.hidden_size, device=device, dtype=dtype)
            if ffn.up_proj.bias is not None:
                bias = bias + ffn.down_proj.weight @ ffn.up_proj.bias
            if ffn.down_proj.bias is not None:
                bias = bias + ffn.down_proj.bias
            linear.linear.bias.copy_(bias)
    return linear


def _low_rank_init(
    down: nn.Linear,
    up: nn.Linear,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    """Initialize up(down(x)) as a truncated-SVD approximation of weight @ x."""

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


def _joint_low_rank_kv_init(
    down: nn.Linear,
    k_up: nn.Linear,
    v_up: nn.Linear,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor | None,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor | None,
) -> None:
    """Initialize shared latent KV projections from concatenated K/V weights."""

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
            if k_bias is None:
                k_up.bias.zero_()
            else:
                k_up.bias.copy_(k_bias.to(device=device, dtype=dtype))
        if v_up.bias is not None:
            if v_bias is None:
                v_up.bias.zero_()
            else:
                v_up.bias.copy_(v_bias.to(device=device, dtype=dtype))


def mean_pool_kv_projection(
    projection: nn.Linear,
    parent_num_kv_heads: int,
    target_num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Mean-pool KV heads to initialize a smaller GQA projection."""

    if parent_num_kv_heads % target_num_kv_heads != 0:
        raise ValueError("parent_num_kv_heads must be divisible by target_num_kv_heads")

    group = parent_num_kv_heads // target_num_kv_heads
    weight = projection.weight.detach().view(parent_num_kv_heads, head_dim, projection.in_features)
    weight = weight.view(target_num_kv_heads, group, head_dim, projection.in_features).mean(dim=1)
    weight = weight.reshape(target_num_kv_heads * head_dim, projection.in_features)

    bias = None
    if projection.bias is not None:
        bias = projection.bias.detach().view(parent_num_kv_heads, head_dim)
        bias = bias.view(target_num_kv_heads, group, head_dim).mean(dim=1)
        bias = bias.reshape(target_num_kv_heads * head_dim)
    return weight, bias


def remap_kv_projection(
    projection: nn.Linear,
    parent_num_kv_heads: int,
    target_num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map a KV projection to a different number of KV heads.

    Reducing heads uses mean pooling; expanding heads repeats the available
    projections. This covers MHA, MQA, and intermediate GQA candidates.
    """

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
        indices = torch.linspace(0, parent_num_kv_heads - 1, target_num_kv_heads, device=weight.device)
        indices = indices.round().to(dtype=torch.long)
        target_weight = weight.index_select(0, indices)
        target_bias = bias.index_select(0, indices) if bias is not None else None

    target_weight = target_weight.reshape(target_num_kv_heads * head_dim, projection.in_features)
    if target_bias is not None:
        target_bias = target_bias.reshape(target_num_kv_heads * head_dim)
    return target_weight, target_bias


def copy_full_attention_weights(source: CausalSelfAttention, target: nn.Module) -> None:
    """Copy parent Q/O and expanded K/V weights into full-head attention-like modules."""

    with torch.no_grad():
        q_projection = getattr(target, "q_proj", None)
        if isinstance(q_projection, nn.Linear):
            q_projection.load_state_dict(source.q_proj.state_dict())
        o_projection = getattr(target, "o_proj", None)
        if isinstance(o_projection, nn.Linear):
            o_projection.load_state_dict(source.o_proj.state_dict())
        for source_projection, target_projection in [
            (source.k_proj, getattr(target, "k_proj", None)),
            (source.v_proj, getattr(target, "v_proj", None)),
        ]:
            if not isinstance(target_projection, nn.Linear):
                continue
            weight, bias = remap_kv_projection(
                source_projection,
                source.num_kv_heads,
                source.num_heads,
                source.head_dim,
            )
            target_projection.weight.copy_(weight)
            if target_projection.bias is not None:
                if bias is None:
                    target_projection.bias.zero_()
                else:
                    target_projection.bias.copy_(bias)


def gqa_from_attention(attention: CausalSelfAttention, target_num_kv_heads: int) -> CausalSelfAttention:
    """Initialize a reduced-KV attention module by mean-pooling K/V heads."""

    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype
    target = CausalSelfAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        num_kv_heads=target_num_kv_heads,
        bias=attention.q_proj.bias is not None,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        target.q_proj.load_state_dict(attention.q_proj.state_dict())
        target.o_proj.load_state_dict(attention.o_proj.state_dict())
        for source, dest in [(attention.k_proj, target.k_proj), (attention.v_proj, target.v_proj)]:
            weight, bias = remap_kv_projection(
                source,
                attention.num_kv_heads,
                target_num_kv_heads,
                attention.head_dim,
            )
            dest.weight.copy_(weight)
            if dest.bias is not None and bias is not None:
                dest.bias.copy_(bias)
    return target


def mha_from_attention(attention: CausalSelfAttention) -> CausalSelfAttention:
    """Initialize an MHA candidate with one KV projection per query head."""

    return gqa_from_attention(attention, target_num_kv_heads=attention.num_heads)


def mqa_from_attention(attention: CausalSelfAttention) -> CausalSelfAttention:
    """Initialize an MQA candidate with a single shared K/V projection."""

    return gqa_from_attention(attention, target_num_kv_heads=1)


def quantized_mha_from_attention(attention: CausalSelfAttention, num_bits: int = 8) -> QuantizedMHAAttention:
    """Initialize an MHA candidate whose projections are quantized."""

    device = attention.q_proj.weight.device
    target_num_kv_heads = attention.num_heads
    k_weight, k_bias = remap_kv_projection(
        attention.k_proj,
        attention.num_kv_heads,
        target_num_kv_heads,
        attention.head_dim,
    )
    v_weight, v_bias = remap_kv_projection(
        attention.v_proj,
        attention.num_kv_heads,
        target_num_kv_heads,
        attention.head_dim,
    )
    target = QuantizedMHAAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        q_proj=QuantizedLinear(attention.q_proj.weight, attention.q_proj.bias, num_bits=num_bits),
        k_proj=QuantizedLinear(k_weight, k_bias, num_bits=num_bits),
        v_proj=QuantizedLinear(v_weight, v_bias, num_bits=num_bits),
        o_proj=QuantizedLinear(attention.o_proj.weight, attention.o_proj.bias, num_bits=num_bits),
    ).to(device=device)
    return target


def mfa_from_attention(
    attention: CausalSelfAttention,
    target_num_kv_heads: int | None = None,
    latent_dim: int | None = None,
) -> FactorizedKVAttention:
    """Initialize an MFA-style low-rank grouped K/V attention candidate."""

    target_num_kv_heads = attention.num_kv_heads if target_num_kv_heads is None else target_num_kv_heads
    latent_dim = max(1, attention.head_dim // 2) if latent_dim is None else latent_dim
    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype
    target = FactorizedKVAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        num_kv_heads=target_num_kv_heads,
        latent_dim=latent_dim,
        bias=attention.q_proj.bias is not None,
    ).to(device=device, dtype=dtype)
    k_weight, k_bias = remap_kv_projection(
        attention.k_proj,
        attention.num_kv_heads,
        target_num_kv_heads,
        attention.head_dim,
    )
    v_weight, v_bias = remap_kv_projection(
        attention.v_proj,
        attention.num_kv_heads,
        target_num_kv_heads,
        attention.head_dim,
    )
    with torch.no_grad():
        target.q_proj.load_state_dict(attention.q_proj.state_dict())
        target.o_proj.load_state_dict(attention.o_proj.state_dict())
    _low_rank_init(target.k_down_proj, target.k_up_proj, k_weight, k_bias)
    _low_rank_init(target.v_down_proj, target.v_up_proj, v_weight, v_bias)
    return target


def mla_from_attention(
    attention: CausalSelfAttention,
    target_num_kv_heads: int | None = None,
    latent_dim: int | None = None,
) -> LatentKVAttention:
    """Initialize an MLA-style shared latent K/V attention candidate."""

    target_num_kv_heads = attention.num_kv_heads if target_num_kv_heads is None else target_num_kv_heads
    latent_dim = max(1, attention.hidden_size // 4) if latent_dim is None else latent_dim
    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype
    target = LatentKVAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        num_kv_heads=target_num_kv_heads,
        latent_dim=latent_dim,
        bias=attention.q_proj.bias is not None,
    ).to(device=device, dtype=dtype)
    k_weight, k_bias = remap_kv_projection(
        attention.k_proj,
        attention.num_kv_heads,
        target_num_kv_heads,
        attention.head_dim,
    )
    v_weight, v_bias = remap_kv_projection(
        attention.v_proj,
        attention.num_kv_heads,
        target_num_kv_heads,
        attention.head_dim,
    )
    with torch.no_grad():
        target.q_proj.load_state_dict(attention.q_proj.state_dict())
        target.o_proj.load_state_dict(attention.o_proj.state_dict())
    _joint_low_rank_kv_init(
        target.kv_down_proj,
        target.k_up_proj,
        target.v_up_proj,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
    )
    return target


def mka_from_attention(attention: CausalSelfAttention, value_rank: int | None = None) -> MultiKeyAttention:
    """Initialize an MKA-style shared-key, low-rank-value attention candidate."""

    value_rank = max(1, attention.hidden_size // 4) if value_rank is None else value_rank
    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype
    target = MultiKeyAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        value_rank=value_rank,
        bias=attention.q_proj.bias is not None,
    ).to(device=device, dtype=dtype)

    k_weight = attention.k_proj.weight.detach().view(
        attention.num_kv_heads,
        attention.head_dim,
        attention.hidden_size,
    )
    k_weight = k_weight.mean(dim=0)
    k_bias = None
    if attention.k_proj.bias is not None:
        k_bias = attention.k_proj.bias.detach().view(attention.num_kv_heads, attention.head_dim).mean(dim=0)
    v_weight, v_bias = expanded_value_projection(attention)
    with torch.no_grad():
        target.q_proj.load_state_dict(attention.q_proj.state_dict())
        target.o_proj.load_state_dict(attention.o_proj.state_dict())
        target.k_proj.weight.copy_(k_weight)
        if target.k_proj.bias is not None:
            if k_bias is None:
                target.k_proj.bias.zero_()
            else:
                target.k_proj.bias.copy_(k_bias)
    _low_rank_init(target.v_down_proj, target.v_up_proj, v_weight, v_bias)
    return target


def kernel_linear_from_attention(
    attention: CausalSelfAttention,
    feature_map: str = "elu",
    output_gate: bool = False,
) -> KernelLinearAttention:
    """Initialize a generic FLA-style linear attention candidate from parent attention."""

    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype
    target = KernelLinearAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        feature_map=feature_map,
        output_gate=output_gate,
        bias=attention.q_proj.bias is not None,
    ).to(device=device, dtype=dtype)
    copy_full_attention_weights(attention, target)
    if output_gate:
        assert target.g_proj is not None
        with torch.no_grad():
            target.g_proj.weight.zero_()
            if target.g_proj.bias is not None:
                target.g_proj.bias.fill_(2.0)
    return target


def retention_from_attention(attention: CausalSelfAttention) -> MultiScaleRetentionAttention:
    """Initialize a generic retention-style candidate from parent attention."""

    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype
    target = MultiScaleRetentionAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        bias=attention.q_proj.bias is not None,
    ).to(device=device, dtype=dtype)
    copy_full_attention_weights(attention, target)
    return target


def local_sparse_from_attention(attention: CausalSelfAttention, window_size: int) -> LocalSparseAttention:
    """Initialize a generic sparse-attention candidate from parent attention."""

    device = attention.q_proj.weight.device
    dtype = attention.q_proj.weight.dtype
    target = LocalSparseAttention(
        hidden_size=attention.hidden_size,
        num_heads=attention.num_heads,
        window_size=window_size,
        bias=attention.q_proj.bias is not None,
    ).to(device=device, dtype=dtype)
    copy_full_attention_weights(attention, target)
    return target


def pruned_ffn_from_parent(ffn: SwiGLUFFN, keep_indices: torch.Tensor) -> SwiGLUFFN:
    """Create a smaller FFN by keeping selected intermediate channels."""

    keep_indices = keep_indices.to(dtype=torch.long, device=ffn.up_proj.weight.device)
    target = SwiGLUFFN(ffn.hidden_size, int(keep_indices.numel()), bias=ffn.up_proj.bias is not None).to(
        device=ffn.up_proj.weight.device,
        dtype=ffn.up_proj.weight.dtype,
    )
    with torch.no_grad():
        target.up_proj.weight.copy_(ffn.up_proj.weight.index_select(0, keep_indices))
        target.gate_proj.weight.copy_(ffn.gate_proj.weight.index_select(0, keep_indices))
        target.down_proj.weight.copy_(ffn.down_proj.weight.index_select(1, keep_indices))
        if ffn.up_proj.bias is not None:
            target.up_proj.bias.copy_(ffn.up_proj.bias.index_select(0, keep_indices))
            target.gate_proj.bias.copy_(ffn.gate_proj.bias.index_select(0, keep_indices))
        if ffn.down_proj.bias is not None:
            target.down_proj.bias.copy_(ffn.down_proj.bias)
    return target


def keep_count(intermediate_size: int, ratio: float) -> int:
    return max(1, min(intermediate_size, int(math.ceil(intermediate_size * ratio))))
