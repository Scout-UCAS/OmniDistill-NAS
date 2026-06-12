from __future__ import annotations

import unittest

import torch

from distill_nas_core.blocks import (
    gqa_from_attention,
    linear_from_attention,
    linear_from_ffn,
    make_parent_block,
    mfa_from_attention,
    mha_from_attention,
    mka_from_attention,
    mla_from_attention,
    mqa_from_attention,
    pruned_ffn_from_parent,
    quantized_mha_from_attention,
)
from distill_nas_core.resources import kv_cache_memory_bytes, parameter_memory_bytes
from distill_nas_core.search_space import (
    ALL_ATTENTION_VARIANT_NAMES,
    CORE_ATTENTION_VARIANT_NAMES,
    FLA_ATTENTION_VARIANT_NAMES,
    LINEAR_ATTENTION_VARIANT_NAMES,
    AttentionSpec,
    attention_specs_from_names,
    layer_variant_specs_from_names,
    make_attention_variant,
)


def gpu_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


class BlockInitializationTest(unittest.TestCase):
    @unittest.skipIf(gpu_device() is None, "GPU device is not available")
    def test_alternative_blocks_follow_parent_device(self) -> None:
        device = gpu_device()
        assert device is not None
        parent = make_parent_block(hidden_size=32, num_heads=4, intermediate_size=64).to(device)

        linear_attention = linear_from_attention(parent.attention)
        linear_ffn = linear_from_ffn(parent.ffn)
        mha_attention = mha_from_attention(parent.attention)
        mqa_attention = mqa_from_attention(parent.attention)
        quant_mha_attention = quantized_mha_from_attention(parent.attention)
        gqa_attention = gqa_from_attention(parent.attention, target_num_kv_heads=2)
        mfa_attention = mfa_from_attention(parent.attention, target_num_kv_heads=2, latent_dim=4)
        mla_attention = mla_from_attention(parent.attention, target_num_kv_heads=2, latent_dim=8)
        mka_attention = mka_from_attention(parent.attention, value_rank=8)
        pruned_ffn = pruned_ffn_from_parent(parent.ffn, torch.arange(16, device=device))

        modules = [
            linear_attention,
            linear_ffn,
            mha_attention,
            mqa_attention,
            quant_mha_attention,
            gqa_attention,
            mfa_attention,
            mla_attention,
            mka_attention,
            pruned_ffn,
        ]
        for module in modules:
            tensor = next(module.parameters(), None)
            if tensor is None:
                tensor = next(module.buffers())
            self.assertEqual(tensor.device.type, device.type)

        inputs = torch.randn(2, 8, 32, device=device)
        self.assertEqual(linear_attention(inputs).device.type, device.type)
        self.assertEqual(linear_ffn(inputs).device.type, device.type)
        self.assertEqual(mha_attention(inputs).device.type, device.type)
        self.assertEqual(quant_mha_attention(inputs).device.type, device.type)
        self.assertEqual(mqa_attention(inputs).device.type, device.type)
        self.assertEqual(gqa_attention(inputs).device.type, device.type)
        self.assertEqual(mfa_attention(inputs).device.type, device.type)
        self.assertEqual(mla_attention(inputs).device.type, device.type)
        self.assertEqual(mka_attention(inputs).device.type, device.type)
        self.assertEqual(pruned_ffn(inputs).device.type, device.type)

    def test_attention_family_variants_forward(self) -> None:
        parent = make_parent_block(hidden_size=32, num_heads=4, intermediate_size=64)
        inputs = torch.randn(2, 8, 32)
        specs = [
            AttentionSpec("mha_attn", "mha", 4),
            AttentionSpec("quant_mha_attn", "quant_mha", 4),
            AttentionSpec("mqa_attn", "mqa", 1),
            AttentionSpec("gqa_kv2", "gqa", 2),
            AttentionSpec("mfa_kv2", "mfa", 2, latent_dim=4),
            AttentionSpec("mla_kv2", "mla", 2, latent_dim=8),
            AttentionSpec("mka_attn", "mka", 1, latent_dim=8),
        ]

        for spec in specs:
            with self.subTest(spec=spec.name):
                attention = make_attention_variant(parent.attention, spec)
                self.assertEqual(attention(inputs).shape, inputs.shape)

    def test_all_attention_alias_variants_forward(self) -> None:
        parent = make_parent_block(hidden_size=32, num_heads=4, intermediate_size=64)
        inputs = torch.randn(2, 8, 32)
        specs = attention_specs_from_names("all_attention", parent.attention.num_heads)

        self.assertEqual([spec.name for spec in specs], list(ALL_ATTENTION_VARIANT_NAMES))
        linear_names = [spec.name for spec in attention_specs_from_names("all_linear_attn", parent.attention.num_heads)]
        self.assertEqual(linear_names, list(LINEAR_ATTENTION_VARIANT_NAMES))
        core_names = [spec.name for spec in attention_specs_from_names("all_core_attn", parent.attention.num_heads)]
        self.assertEqual(core_names, list(CORE_ATTENTION_VARIANT_NAMES))
        self.assertTrue(set(LINEAR_ATTENTION_VARIANT_NAMES).issubset(core_names))
        self.assertTrue(set(FLA_ATTENTION_VARIANT_NAMES).issubset([spec.name for spec in specs]))
        for spec in specs:
            with self.subTest(spec=spec.name):
                attention = make_attention_variant(parent.attention, spec)
                self.assertEqual(attention(inputs).shape, inputs.shape)

    def test_attention_family_kv_cache_estimates(self) -> None:
        parent = make_parent_block(hidden_size=32, num_heads=4, intermediate_size=64)
        seq_len = 16
        parent_cache = kv_cache_memory_bytes(parent.attention, seq_len, dtype_bytes=2)
        mha_memory = parameter_memory_bytes(mha_from_attention(parent.attention), dtype_bytes=2)
        quant_mha_memory = parameter_memory_bytes(quantized_mha_from_attention(parent.attention), dtype_bytes=2)
        mqa_cache = kv_cache_memory_bytes(mqa_from_attention(parent.attention), seq_len, dtype_bytes=2)
        mla_cache = kv_cache_memory_bytes(
            mla_from_attention(parent.attention, target_num_kv_heads=2, latent_dim=8),
            seq_len,
            dtype_bytes=2,
        )
        mka_cache = kv_cache_memory_bytes(
            mka_from_attention(parent.attention, value_rank=8),
            seq_len,
            dtype_bytes=2,
        )

        self.assertLess(quant_mha_memory, mha_memory)
        self.assertLess(mqa_cache, parent_cache)
        self.assertLess(mla_cache, parent_cache)
        self.assertLess(mka_cache, parent_cache)

    def test_layer_variant_aliases_map_to_block_specs(self) -> None:
        specs = layer_variant_specs_from_names("parent,skip_attn,skip_mlp,skip_both", parent_num_heads=4)
        self.assertEqual([spec.name for spec in specs], ["parent", "skip_attn", "skip_mlp", "skip_both"])
        self.assertEqual((specs[1].attention.name, specs[1].ffn.name), ("noop_attn", "parent_ffn"))
        self.assertEqual((specs[2].attention.name, specs[2].ffn.name), ("parent_attn", "noop_ffn"))
        self.assertEqual((specs[3].attention.name, specs[3].ffn.name), ("noop_attn", "noop_ffn"))


if __name__ == "__main__":
    unittest.main()
