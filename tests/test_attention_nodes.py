from __future__ import annotations

import functools
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_nodes_attention_test"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

attention_nodes = load_module("attention_nodes", "attention_nodes.py")


class FakeModelPatcher:
    def __init__(self):
        self.model_options = {"transformer_options": {"existing_flag": True}}

    def clone(self) -> "FakeModelPatcher":
        cloned = FakeModelPatcher()
        cloned.model_options = {
            **self.model_options,
            "transformer_options": dict(self.model_options.get("transformer_options", {})),
        }
        return cloned


class AttentionDetectionAndSelectorTests(unittest.TestCase):
    def tearDown(self):
        attention_nodes.detect_available_attentions(force_refresh=True)

    def test_startup_detection_only_lists_available_backends(self):
        # Simulate environment where sageattention is installed, flash_attn is not, xformers is installed
        fake_sage = types.ModuleType("sageattention")
        fake_sage.__version__ = "2.1.0"
        fake_sage.sageattn = lambda q, k, v, **kw: q
        fake_sage.sageattn_qk_int8_pv_fp16_cuda = lambda q, k, v, **kw: q
        fake_sage.sageattn_qk_int8_pv_fp16_triton = lambda q, k, v, **kw: q

        fake_xformers = types.ModuleType("xformers")
        fake_xformers.__version__ = "0.0.29"
        fake_xformers_ops = types.ModuleType("xformers.ops")
        fake_xformers_ops.memory_efficient_attention = lambda q, k, v, attn_bias=None: q
        fake_xformers.ops = fake_xformers_ops

        with mock.patch.dict(
            sys.modules,
            {
                "sageattention": fake_sage,
                "flash_attn": None,
                "flash_attn_interface": None,
                "xformers": fake_xformers,
                "xformers.ops": fake_xformers_ops,
            },
        ):
            det = attention_nodes.detect_available_attentions(force_refresh=True)
            self.assertIn("auto", det["backends"])
            self.assertIn("sage", det["backends"])
            self.assertNotIn("flash", det["backends"])
            self.assertIn("xformers", det["backends"])
            self.assertIn("pytorch", det["backends"])
            self.assertIn("sage_cuda_fp16", det["sub_kernels"])
            self.assertIn("sage_triton", det["sub_kernels"])
            self.assertNotIn("sage_cuda_fp8", det["sub_kernels"])

            input_types = attention_nodes.ZiggartAttentionSelector.INPUT_TYPES()
            combo_backends = input_types["required"]["attention"][0]
            self.assertEqual(combo_backends, det["backends"])

    def test_model_clone_isolation_and_auto_restoration(self):
        node = attention_nodes.ZiggartAttentionSelector()
        base_model = FakeModelPatcher()

        patched_model, info = node.apply_attention(
            model=base_model,
            attention="pytorch",
            sub_kernel="sdpa_math",
            global_override=False,
            fallback_on_error=True,
        )

        self.assertIsNot(patched_model, base_model)
        self.assertNotIn(
            "optimized_attention_override",
            base_model.model_options["transformer_options"],
        )
        self.assertIn(
            "optimized_attention_override",
            patched_model.model_options["transformer_options"],
        )
        self.assertTrue(patched_model.model_options["transformer_options"]["existing_flag"])
        self.assertIn("Active Attention: pytorch", info)
        self.assertIn("Active Sub-Kernel: sdpa_math", info)

        # Switching back to "auto" removes the override on the new clone
        auto_model, auto_info = node.apply_attention(
            model=patched_model,
            attention="auto",
        )
        self.assertNotIn(
            "optimized_attention_override",
            auto_model.model_options["transformer_options"],
        )
        self.assertIn("Active Attention: auto", auto_info)

    def test_pytorch_and_basic_attention_shapes_match(self):
        torch.manual_seed(0)
        b, seq, heads, dim_head = 2, 16, 4, 32
        q = torch.randn(b, seq, heads * dim_head, dtype=torch.float32)
        k = torch.randn(b, seq, heads * dim_head, dtype=torch.float32)
        v = torch.randn(b, seq, heads * dim_head, dtype=torch.float32)

        override_pt = attention_nodes.make_attention_override("pytorch", sub_kernel="sdpa_math")
        override_basic = attention_nodes.make_attention_override("basic")

        # Simulate ComfyUI @wrap_attn passing (orig_func, q, k, v, heads, ...)
        dummy_orig = lambda *a, **kw: None
        out_pt = override_pt(dummy_orig, q, k, v, heads)
        out_basic = override_basic(dummy_orig, q, k, v, heads)

        self.assertEqual(out_pt.shape, (b, seq, heads * dim_head))
        self.assertEqual(out_basic.shape, (b, seq, heads * dim_head))
        self.assertTrue(torch.allclose(out_pt, out_basic, atol=1e-4))

        # Test skip_reshape=True and skip_output_reshape=True (4D HND layout)
        q_4d = q.view(b, seq, heads, dim_head).transpose(1, 2)
        k_4d = k.view(b, seq, heads, dim_head).transpose(1, 2)
        v_4d = v.view(b, seq, heads, dim_head).transpose(1, 2)

        out_pt_4d = override_pt(
            dummy_orig,
            q_4d,
            k_4d,
            v_4d,
            heads,
            skip_reshape=True,
            skip_output_reshape=True,
        )
        self.assertEqual(out_pt_4d.shape, (b, heads, seq, dim_head))
        self.assertTrue(
            torch.allclose(
                out_pt_4d.transpose(1, 2).reshape(b, seq, heads * dim_head),
                out_pt,
                atol=1e-4,
            )
        )

    def test_sage_attention_dispatch_and_mask_fallback(self):
        recorded_calls = []

        def fake_sage_cuda(q, k, v, tensor_layout="HND", is_causal=False):
            recorded_calls.append((q.shape, q.dtype, tensor_layout, is_causal))
            return v

        fake_sage = types.ModuleType("sageattention")
        fake_sage.sageattn = fake_sage_cuda
        fake_sage.sageattn_qk_int8_pv_fp16_cuda = fake_sage_cuda

        with mock.patch.dict(sys.modules, {"sageattention": fake_sage}):
            attention_nodes.detect_available_attentions(force_refresh=True)
            override_sage = attention_nodes.make_attention_override(
                "sage",
                sub_kernel="sage_cuda_fp16",
            )

            b, seq, heads, dim_head = 1, 8, 2, 16
            q = torch.ones(b, seq, heads * dim_head, dtype=torch.float32)
            k = torch.ones(b, seq, heads * dim_head, dtype=torch.float32)
            v = torch.ones(b, seq, heads * dim_head, dtype=torch.float32)

            out = override_sage(q, k, v, heads)
            self.assertEqual(out.shape, (b, seq, heads * dim_head))
            self.assertEqual(out.dtype, torch.float32)
            self.assertEqual(len(recorded_calls), 1)
            self.assertEqual(recorded_calls[0], ((b, heads, seq, dim_head), torch.float16, "HND", False))

            # With a mask, SageAttention should fall back to PyTorch SDPA without calling fake_sage_cuda
            mask = torch.zeros((seq, seq), dtype=torch.float32)
            out_masked = override_sage(q, k, v, heads, mask=mask)
            self.assertEqual(out_masked.shape, (b, seq, heads * dim_head))
            self.assertEqual(len(recorded_calls), 1)

    def test_flash_attention_nhd_layout_conversion(self):
        recorded_shapes = []

        def fake_flash_func(q, k, v, dropout_p=0.0, causal=False):
            recorded_shapes.append((q.shape, q.dtype))
            return v

        fake_flash = types.ModuleType("flash_attn")
        fake_flash.flash_attn_func = fake_flash_func

        with mock.patch.dict(sys.modules, {"flash_attn": fake_flash}):
            attention_nodes.detect_available_attentions(force_refresh=True)
            override_flash = attention_nodes.make_attention_override("flash", sub_kernel="flash_v2")

            b, seq, heads, dim_head = 2, 12, 4, 16
            q = torch.ones(b, seq, heads * dim_head, dtype=torch.float16)
            k = torch.ones(b, seq, heads * dim_head, dtype=torch.float16)
            v = torch.ones(b, seq, heads * dim_head, dtype=torch.float16)

            out = override_flash(q, k, v, heads)
            self.assertEqual(out.shape, (b, seq, heads * dim_head))
            # FlashAttention expects (B, N, H, D)
            self.assertEqual(recorded_shapes[-1], ((b, seq, heads, dim_head), torch.float16))

            # Also verify skip_reshape=True (input is B, H, N, D) and skip_output_reshape=True
            q_hnd = q.view(b, seq, heads, dim_head).transpose(1, 2)
            out_hnd = override_flash(
                q_hnd,
                q_hnd,
                q_hnd,
                heads,
                skip_reshape=True,
                skip_output_reshape=True,
            )
            self.assertEqual(out_hnd.shape, (b, heads, seq, dim_head))
            self.assertEqual(recorded_shapes[-1], ((b, seq, heads, dim_head), torch.float16))

    def test_wrap_attn_recursion_prevention_and_global_override(self):
        # Simulate ComfyUI's comfy.ldm.modules.attention with @wrap_attn decorator
        def wrap_attn(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t_opts = kwargs.get("transformer_options", {})
                if "optimized_attention_override" in t_opts:
                    return t_opts["optimized_attention_override"](func, *args, **kwargs)
                return func(*args, **kwargs)

            return wrapper

        @wrap_attn
        def fake_attention_sub_quad(q, k, v, heads, **kwargs):
            return v * 2.0

        fake_comfy_attn = types.SimpleNamespace(
            attention_sub_quad=fake_attention_sub_quad,
            optimized_attention=fake_attention_sub_quad,
            optimized_attention_masked=fake_attention_sub_quad,
        )

        with mock.patch.object(attention_nodes, "comfy_attention", fake_comfy_attn):
            attention_nodes._ORIGINAL_GLOBAL_ATTN.clear()
            node = attention_nodes.ZiggartAttentionSelector()
            model = FakeModelPatcher()
            patched, _ = node.apply_attention(
                model=model,
                attention="sub_quad",
                global_override=True,
            )

            t_opts = patched.model_options["transformer_options"]
            q = torch.ones((1, 4, 8), dtype=torch.float32)
            # Calling decorated function with transformer_options containing optimized_attention_override
            # must NOT infinitely recurse!
            res = fake_attention_sub_quad(q, q, q, 2, transformer_options=t_opts)
            self.assertTrue(torch.allclose(res, q * 2.0))
            self.assertIsNot(fake_comfy_attn.optimized_attention, fake_attention_sub_quad)

            # Restoring global override
            node.apply_attention(model=model, attention="auto", global_override=False)
            self.assertIs(fake_comfy_attn.optimized_attention, fake_attention_sub_quad)

    def test_fallback_on_error_behavior(self):
        def broken_sage(q, k, v, **kwargs):
            raise RuntimeError("Simulated kernel failure")

        fake_sage = types.ModuleType("sageattention")
        fake_sage.sageattn = broken_sage

        with mock.patch.dict(sys.modules, {"sageattention": fake_sage}):
            attention_nodes.detect_available_attentions(force_refresh=True)

            q = torch.randn(1, 8, 32, dtype=torch.float32)
            override_safe = attention_nodes.make_attention_override("sage", fallback_on_error=True)
            out = override_safe(q, q, q, 2)
            self.assertEqual(out.shape, (1, 8, 32))

            override_strict = attention_nodes.make_attention_override("sage", fallback_on_error=False)
            with self.assertRaisesRegex(RuntimeError, "Simulated kernel failure"):
                override_strict(q, q, q, 2)


if __name__ == "__main__":
    unittest.main()
