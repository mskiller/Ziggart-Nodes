import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file as safetensors_save_file


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_d3dv3_test"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
embedded_model = load_module("embedded_model", "embedded_model.py")
d3d = load_module("direct3d_v3_backend", "direct3d_v3_backend.py")
three_d_nodes = load_module("three_d_nodes", "three_d_nodes.py")


def _make_unit_cube_mesh(half_extent: float = 0.4) -> tuple[torch.Tensor, torch.Tensor]:
    h = float(half_extent)
    verts = torch.tensor(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=torch.int32,
    )
    return verts, faces


def _fake_direct3d_modules(calc_window_fn=None, voxelize_fn=None, sparse_dit_cls=None) -> dict[str, types.ModuleType]:
    pkg = types.ModuleType("direct3d")
    models_pkg = types.ModuleType("direct3d.models")
    utils_pkg = types.ModuleType("direct3d.utils")
    sparse_atten_mod = types.ModuleType("direct3d.models.sparse_atten")
    sparse_dit_mod = types.ModuleType("direct3d.models.sparse_dit")
    voxelization_mod = types.ModuleType("direct3d.utils.voxelization")

    if calc_window_fn is not None:
        sparse_atten_mod.calc_window_partition = calc_window_fn
    if voxelize_fn is not None:
        voxelization_mod.voxelize = voxelize_fn
    if sparse_dit_cls is not None:
        sparse_dit_mod.SparseDiT = sparse_dit_cls

    pkg.models = models_pkg
    pkg.utils = utils_pkg
    models_pkg.sparse_atten = sparse_atten_mod
    models_pkg.sparse_dit = sparse_dit_mod
    utils_pkg.voxelization = voxelization_mod

    return {
        "direct3d": pkg,
        "direct3d.models": models_pkg,
        "direct3d.models.sparse_atten": sparse_atten_mod,
        "direct3d.models.sparse_dit": sparse_dit_mod,
        "direct3d.utils": utils_pkg,
        "direct3d.utils.voxelization": voxelization_mod,
    }


class W8G32LinearCudaTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton W8G32Linear tests")
    def test_triton_w8g32_linear_matches_reference(self):
        if not d3d.TRITON_AVAILABLE:
            self.skipTest("Triton is not installed in this environment")

        torch.manual_seed(1234)
        cases = [
            ((137, 1536), 1536),
            ((64, 5120), 1536),
            ((2, 32, 1536), 1536),
        ]
        for x_shape, out_features in cases:
            in_features = x_shape[-1]
            for dtype in (torch.float16, torch.bfloat16):
                for use_bias in (False, True):
                    with self.subTest(x_shape=x_shape, out_features=out_features, dtype=dtype, use_bias=use_bias):
                        q8 = torch.randint(-127, 128, (out_features, in_features), device="cuda", dtype=torch.int8)
                        scales = (torch.rand((out_features, in_features // 32), device="cuda", dtype=torch.float32) * 0.001) + 0.0002
                        bias = (torch.randn(out_features, device="cuda", dtype=dtype) * 0.05) if use_bias else None
                        x = torch.randn(x_shape, device="cuda", dtype=dtype) * 0.1

                        layer = d3d.W8G32Linear(
                            in_features=in_features,
                            out_features=out_features,
                            q8=q8,
                            scales=scales,
                            bias=bias,
                            group_size=32,
                            use_triton=True,
                        ).to("cuda")

                        with torch.no_grad():
                            out = layer(x)
                            w_ref = d3d.dequantize_w8g32(q8, scales, group_size=32, dtype=dtype)
                            ref = F.linear(x, w_ref, bias)

                        self.assertEqual(out.shape, ref.shape)
                        self.assertEqual(out.dtype, dtype)
                        self.assertTrue(
                            torch.allclose(out, ref, rtol=1e-2, atol=2.5e-2),
                            f"Max abs diff: {(out - ref).abs().max().item()}",
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for W8G32Linear CUDA fallback test")
    def test_pure_pytorch_fallback_matches_f_linear(self):
        torch.manual_seed(5678)
        in_features, out_features = 512, 256
        q8 = torch.randint(-127, 128, (out_features, in_features), device="cuda", dtype=torch.int8)
        scales = (torch.rand((out_features, in_features // 32), device="cuda", dtype=torch.float32) * 0.01) + 0.002
        for dtype in (torch.float16, torch.bfloat16):
            bias = torch.randn(out_features, device="cuda", dtype=dtype) * 0.1
            x = torch.randn((17, in_features), device="cuda", dtype=dtype) * 0.5
            layer = d3d.W8G32Linear(
                in_features=in_features,
                out_features=out_features,
                q8=q8,
                scales=scales,
                bias=bias,
                group_size=32,
                use_triton=False,
            ).to("cuda")
            with torch.no_grad():
                out = layer(x)
                w_ref = d3d.dequantize_w8g32(q8, scales, group_size=32, dtype=dtype)
                ref = F.linear(x, w_ref, bias)
            self.assertTrue(torch.allclose(out, ref, atol=1e-3, rtol=1e-3))


class VoxelizerTests(unittest.TestCase):
    def test_voxelize_mesh_pure_torch_unit_cube_and_dim128(self):
        verts, faces = _make_unit_cube_mesh(0.4)
        for dim in (64, 128):
            voxel = torch.zeros(dim**3, dtype=torch.bool)
            d3d.voxelize_mesh_pure_torch(verts, faces, voxel, len(faces), dim=dim)
            occupied = int(voxel.sum().item())
            self.assertGreater(occupied, 0)
            self.assertLess(occupied, dim**3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton voxelizer test")
    def test_voxelize_mesh_triton_unit_cube_and_subdivision(self):
        if not d3d.TRITON_AVAILABLE:
            self.skipTest("Triton is not installed in this environment")
        verts_cpu, faces_cpu = _make_unit_cube_mesh(0.4)
        verts = verts_cpu.cuda()
        faces = faces_cpu.cuda()

        # Verify large-triangle subdivision directly at dim=128
        sub_verts, sub_faces = d3d._subdivide_large_triangles(verts, faces, dim=128)
        self.assertGreater(len(sub_faces), len(faces))
        self.assertGreater(len(sub_verts), len(verts))

        for dim in (64, 128):
            voxel = torch.zeros(dim**3, device="cuda", dtype=torch.bool)
            d3d.voxelize_mesh_triton(verts, faces, voxel, len(faces), dim=dim)
            occupied = int(voxel.sum().item())
            self.assertGreater(occupied, 0)
            self.assertLess(occupied, dim**3)

    def test_mesh_patches_adaptive_fit_and_error(self):
        verts, faces = _make_unit_cube_mesh(0.5)
        observed_scales: list[float] = []

        def fake_voxelize(centered_verts, _faces, dim=1024, patch_size=8):
            extent = float((centered_verts.amax(dim=0) - centered_verts.amin(dim=0)).max().item())
            scale = extent / 1.9
            observed_scales.append(scale)
            # Token count scales quadratically with mesh extent down to >= 1
            n_tok = max(1, int(round(1000.0 * (scale**2))))
            return torch.zeros((n_tok, 4), dtype=torch.int32)

        with mock.patch.dict(sys.modules, _fake_direct3d_modules(voxelize_fn=fake_voxelize)):
            coords, scale_used = d3d.mesh_patches_adaptive(
                verts,
                faces,
                resolution=1024,
                patch_size=8,
                max_tokens=400,
                fit_large_meshes=True,
            )
            self.assertLessEqual(len(coords), 400)
            self.assertGreater(len(coords), 0)
            self.assertLess(scale_used, 1.0)
            self.assertGreater(len(observed_scales), 1)

            with self.assertRaises(ValueError):
                d3d.mesh_patches_adaptive(
                    verts,
                    faces,
                    resolution=1024,
                    patch_size=8,
                    max_tokens=400,
                    fit_large_meshes=False,
                )


class AlgorithmicTests(unittest.TestCase):
    def test_sparse_dc_infer_box_sdf(self):
        patch_size = 8
        voxel_size = 32
        grid_patches = voxel_size // patch_size

        # Cover all 4x4x4 patches in the 32^3 grid
        pi = torch.arange(grid_patches, dtype=torch.int32)
        gx, gy, gz = torch.meshgrid(pi, pi, pi, indexing="ij")
        latent_index = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

        # Build exact voxel coordinates for each patch sample [N, 8^3, 3]
        local = torch.arange(patch_size, dtype=torch.int32)
        lx, ly, lz = torch.meshgrid(local, local, local, indexing="ij")
        local_offsets = torch.stack([lx.reshape(-1), ly.reshape(-1), lz.reshape(-1)], dim=-1)  # [512, 3]
        vox_coords = (latent_index[:, None, :] * patch_size + local_offsets[None, :, :]).float()

        # Box SDF centered at (15.0, 15.0, 15.0) with half-extent 6.5 (negative inside, positive outside)
        sdf = (vox_coords - 15.0).abs().amax(dim=-1) - 6.5

        verts, faces = d3d.sparse_dc_infer(sdf, latent_index, voxel_size=voxel_size, patch_size=patch_size, mc_threshold=0.0)
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 0)
        self.assertTrue((faces >= 0).all().item())
        self.assertTrue((faces < len(verts)).all().item())
        self.assertTrue((verts >= -0.5).all().item())
        self.assertTrue((verts <= 0.5).all().item())

    def test_apply_guidance_cfg_apg_fdg(self):
        torch.manual_seed(99)
        x_cond = torch.randn(16, 8**3, dtype=torch.float32)
        x_uncond = torch.randn(16, 8**3, dtype=torch.float32)

        out_cfg = d3d.apply_guidance(x_cond, x_uncond, guidance_scale=4.0, guidance_method="cfg", patch_size=8)
        out_apg = d3d.apply_guidance(x_cond, x_uncond, guidance_scale=4.0, guidance_method="apg", patch_size=8)
        out_fdg = d3d.apply_guidance(x_cond, x_uncond, guidance_scale=4.0, guidance_method="fdg", patch_size=8)

        for out in (out_cfg, out_apg, out_fdg):
            self.assertEqual(out.shape, x_cond.shape)
            self.assertTrue(torch.isfinite(out).all().item())

        expected_cfg = x_uncond + 4.0 * (x_cond - x_uncond)
        self.assertTrue(torch.allclose(out_cfg, expected_cfg, atol=1e-6))
        self.assertFalse(torch.allclose(out_cfg, out_apg))
        self.assertFalse(torch.allclose(out_cfg, out_fdg))
        self.assertFalse(torch.allclose(out_apg, out_fdg))

    def test_sample_x0_advanced_speed_and_multiview(self):
        class MockBlock(nn.Module):
            def forward(self, s, c, cond, index, layout, skip_value=None, window_size=4, window_cache=None):
                out = s + 0.01 * cond.mean(dim=1, keepdim=False).mean(dim=-1, keepdim=True)
                if skip_value is not None:
                    out = out + 0.5 * skip_value
                return out

        class MockDecoder(nn.Module):
            def forward(self, x_patches, s, index, window_cache=None, window_size=4):
                return x_patches * 0.5 + 0.1 * s[:, None, :1]

        class MockSparseDiT(nn.Module):
            def __init__(self, patch_size: int = 4, channels: int = 1, dim: int = 16):
                super().__init__()
                self.patch_size = patch_size
                self.in_channels = channels
                self.out_channels = channels
                self.window_size = 4
                self.shift_window = 2
                self.depth = 2
                self.time_factor = 1000.0
                self.t_embedder = lambda t: t.reshape(-1, 1).expand(-1, dim)
                self.x_embedder = nn.Linear(patch_size**3 * channels, dim)
                self.blocks = nn.ModuleList([MockBlock(), MockBlock()])
                self.decoder = MockDecoder()
                self.forward_calls = 0
                self.uncond_calls = 0
                self.cond_calls = 0

        def fake_calc_window_partition(latents_index, window_size, shift):
            return ("win", len(latents_index), int(window_size), int(shift))

        patch_size = 4
        n_tokens = 12
        latents_index = torch.zeros((n_tokens, 4), dtype=torch.int32)
        latents_index[:, 1] = torch.arange(n_tokens, dtype=torch.int32) % 4
        latents_index[:, 2] = torch.arange(n_tokens, dtype=torch.int32) // 4
        cond_single = {"front": torch.ones((1, 8, 16), dtype=torch.float32)}

        def make_counting_forward(mock_model: MockSparseDiT):
            orig_fn = d3d.forward_with_cached_windows

            def wrapped(model, x, index, layout, t, cond, window_cache):
                mock_model.forward_calls += 1
                if float(cond.abs().max().item()) == 0.0:
                    mock_model.uncond_calls += 1
                else:
                    mock_model.cond_calls += 1
                return orig_fn(model, x, index, layout, t, cond, window_cache)

            return wrapped

        with mock.patch.dict(sys.modules, _fake_direct3d_modules(calc_window_fn=fake_calc_window_partition)):
            # 1. Single-view speed_mode="lossless" vs "fast" vs skip_guidance_last_steps=2
            counts = {}
            for tag, sp_mode, skip_last in (
                ("lossless_0", "lossless", 0),
                ("fast_0", "fast", 0),
                ("fast_2", "fast", 2),
            ):
                m = MockSparseDiT(patch_size=patch_size)
                gen = torch.Generator().manual_seed(42)
                with mock.patch.object(d3d, "forward_with_cached_windows", side_effect=make_counting_forward(m)):
                    out = d3d.sample_x0_advanced(
                        m,
                        latents_index,
                        cond_single,
                        gen,
                        patch_size=patch_size,
                        steps=8,
                        guidance_scale=4.0,
                        cfg_interval=(0.0, 1.0),
                        speed_mode=sp_mode,
                        skip_guidance_last_steps=skip_last,
                    )
                self.assertEqual(tuple(out.shape), (n_tokens, patch_size**3))
                counts[tag] = m.forward_calls

            self.assertEqual(counts["lossless_0"], 16)
            self.assertEqual(counts["fast_0"], 15)
            self.assertEqual(counts["fast_2"], 14)
            self.assertLess(counts["fast_0"], counts["lossless_0"])
            self.assertLess(counts["fast_2"], counts["fast_0"])

            # 2. Multiview (4 views): "fast" shares 1 uncond pass & 1 apply_guidance vs "standard" (1 uncond pass & 4 apply_guidance)
            cond_mv = {
                "front": torch.full((1, 8, 16), 0.5, dtype=torch.float32),
                "back": torch.full((1, 8, 16), -0.5, dtype=torch.float32),
                "left": torch.full((1, 8, 16), 0.25, dtype=torch.float32),
                "right": torch.full((1, 8, 16), -0.25, dtype=torch.float32),
            }
            patch_normals = torch.tensor([[0.0, 0.0, 1.0]] * n_tokens, dtype=torch.float32)

            mv_stats = {}
            for mv_mode in ("fast", "standard"):
                m = MockSparseDiT(patch_size=patch_size)
                gen = torch.Generator().manual_seed(42)
                guidance_calls = 0
                orig_guidance = d3d.apply_guidance

                def count_guidance(*args, **kwargs):
                    nonlocal guidance_calls
                    guidance_calls += 1
                    return orig_guidance(*args, **kwargs)

                with (
                    mock.patch.object(d3d, "forward_with_cached_windows", side_effect=make_counting_forward(m)),
                    mock.patch.object(d3d, "apply_guidance", side_effect=count_guidance),
                ):
                    out = d3d.sample_x0_advanced(
                        m,
                        latents_index,
                        cond_mv,
                        gen,
                        patch_size=patch_size,
                        steps=4,
                        guidance_scale=4.0,
                        cfg_interval=(0.0, 1.0),
                        guidance_method="apg",
                        speed_mode="lossless",
                        multiview_speed=mv_mode,
                        patch_normals=patch_normals,
                    )
                mv_stats[mv_mode] = {
                    "uncond": m.uncond_calls,
                    "cond": m.cond_calls,
                    "guidance": guidance_calls,
                    "out": out,
                }

            # 4 steps * 1 shared uncond pass = 4 uncond calls (not 4 * 4 = 16)
            self.assertEqual(mv_stats["fast"]["uncond"], 4)
            self.assertEqual(mv_stats["fast"]["cond"], 16)
            self.assertEqual(mv_stats["fast"]["guidance"], 4)
            self.assertEqual(mv_stats["standard"]["uncond"], 4)
            self.assertEqual(mv_stats["standard"]["cond"], 16)
            self.assertEqual(mv_stats["standard"]["guidance"], 16)
            self.assertFalse(torch.allclose(mv_stats["fast"]["out"], mv_stats["standard"]["out"]))

    def test_flash_attn_sdpa_fallbacks_match_dense_sdpa(self):
        torch.manual_seed(2025)
        heads, dim = 4, 16

        # 1. _sdpa_qkvpacked: [B, N, 3, H, C]
        qkv = torch.randn(2, 7, 3, heads, dim, dtype=torch.float32)
        out_qkv = d3d._sdpa_qkvpacked(qkv)
        ref_qkv = F.scaled_dot_product_attention(
            qkv[:, :, 0].permute(0, 2, 1, 3),
            qkv[:, :, 1].permute(0, 2, 1, 3),
            qkv[:, :, 2].permute(0, 2, 1, 3),
        ).permute(0, 2, 1, 3)
        self.assertTrue(torch.allclose(out_qkv, ref_qkv, atol=1e-5, rtol=1e-5))

        # 2. _sdpa_varlen_kvpacked: q [Mq, H, C], kv [Mkv, 2, H, C]
        cu_q = torch.tensor([0, 5, 12, 12], dtype=torch.int32)
        cu_kv = torch.tensor([0, 7, 11, 14], dtype=torch.int32)
        q = torch.randn(12, heads, dim, dtype=torch.float32)
        kv = torch.randn(14, 2, heads, dim, dtype=torch.float32)
        out_vkv = d3d._sdpa_varlen_kvpacked(q, kv, cu_q, cu_kv, max_seqlen_q=7, max_seqlen_kv=7)
        for i in range(2):
            qs, qe = int(cu_q[i]), int(cu_q[i + 1])
            ks, ke = int(cu_kv[i]), int(cu_kv[i + 1])
            ref_i = F.scaled_dot_product_attention(
                q[qs:qe].transpose(0, 1).unsqueeze(0),
                kv[ks:ke, 0].transpose(0, 1).unsqueeze(0),
                kv[ks:ke, 1].transpose(0, 1).unsqueeze(0),
            ).squeeze(0).transpose(0, 1)
            self.assertTrue(torch.allclose(out_vkv[qs:qe], ref_i, atol=1e-5, rtol=1e-5))

        # 3. _sdpa_varlen_qkvpacked: qkv [M, 3, H, C]
        cu_seq = torch.tensor([0, 4, 11], dtype=torch.int32)
        qkv_var = torch.randn(11, 3, heads, dim, dtype=torch.float32)
        out_vqkv = d3d._sdpa_varlen_qkvpacked(qkv_var, cu_seq, max_seqlen=7)
        for i in range(2):
            s, e = int(cu_seq[i]), int(cu_seq[i + 1])
            ref_i = F.scaled_dot_product_attention(
                qkv_var[s:e, 0].transpose(0, 1).unsqueeze(0),
                qkv_var[s:e, 1].transpose(0, 1).unsqueeze(0),
                qkv_var[s:e, 2].transpose(0, 1).unsqueeze(0),
            ).squeeze(0).transpose(0, 1)
            self.assertTrue(torch.allclose(out_vqkv[s:e], ref_i, atol=1e-5, rtol=1e-5))

    def test_resolve_direct3d_image_encoder_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            model_index = {"image_encoder": "./image_encoder"}

            # No local image_encoder/ dir -> falls back to default DINOv3 HF repo
            res = d3d.resolve_direct3d_image_encoder(model_dir, model_index)
            self.assertEqual(res, {"model_name_or_path": "facebook/dinov3-vith16plus-pretrain-lvd1689m"})

            # Local image_encoder/config.json exists -> uses local path
            enc_dir = model_dir / "image_encoder"
            enc_dir.mkdir(parents=True)
            (enc_dir / "config.json").write_text("{}", encoding="utf-8")
            res_local = d3d.resolve_direct3d_image_encoder(model_dir, model_index)
            self.assertEqual(res_local, {"model_name_or_path": str(enc_dir)})

            # Override takes precedence
            res_override = d3d.resolve_direct3d_image_encoder(model_dir, model_index, image_encoder_override="custom/dino")
            self.assertEqual(res_override, {"model_name_or_path": "custom/dino"})

    def test_load_direct3d_v3_denoiser_w8g32_vs_dequant(self):
        class TinySparseDiT(nn.Module):
            def __init__(self, in_dim: int = 64, out_dim: int = 32):
                super().__init__()
                self.proj = nn.Linear(in_dim, out_dim, bias=True)
                self.norm = nn.LayerNorm(out_dim)

            def forward(self, x):
                return self.norm(self.proj(x))

        torch.manual_seed(77)
        q8 = torch.randint(-127, 128, (32, 64), dtype=torch.int8)
        scales = (torch.rand((32, 2), dtype=torch.float32) * 0.01) + 0.002
        bias = torch.randn(32, dtype=torch.float16) * 0.1
        norm_w = torch.ones(32, dtype=torch.float16)
        norm_b = torch.zeros(32, dtype=torch.float16)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "model.safetensors"
            safetensors_save_file(
                {
                    "proj.weight.q8": q8,
                    "proj.weight.s": scales,
                    "proj.bias": bias,
                    "norm.weight": norm_w,
                    "norm.bias": norm_b,
                },
                str(ckpt_path),
                metadata={"format": "direct3d-v3-w8g32", "group": "32"},
            )

            cfg = {"in_dim": 64, "out_dim": 32}
            x = torch.randn((4, 64), dtype=torch.float16)

            with mock.patch.dict(sys.modules, _fake_direct3d_modules(sparse_dit_cls=TinySparseDiT)):
                m_w8 = d3d.load_direct3d_v3_denoiser(
                    ckpt_path, cfg, dtype=torch.float16, use_triton_int8=False, dequant_to_fp16=False
                )
                self.assertIsInstance(m_w8.proj, d3d.W8G32Linear)
                self.assertEqual(m_w8.proj.q8.dtype, torch.int8)
                self.assertTrue(torch.equal(m_w8.proj.q8, q8))
                self.assertEqual(m_w8.proj.weight.dtype, m_w8.proj.dtype)
                self.assertEqual(m_w8.proj.weight.device, m_w8.proj.q8.device)
                self.assertEqual(m_w8.proj.weight.shape, (m_w8.proj.out_features, m_w8.proj.in_features))
                self.assertNotIn("weight", dict(m_w8.proj.named_parameters()))
                self.assertNotIn("weight", dict(m_w8.proj.named_buffers()))

                m_fp16 = d3d.load_direct3d_v3_denoiser(
                    ckpt_path, cfg, dtype=torch.float16, use_triton_int8=False, dequant_to_fp16=True
                )
                self.assertIsInstance(m_fp16.proj, nn.Linear)
                self.assertEqual(m_fp16.proj.weight.dtype, torch.float16)

                with torch.no_grad():
                    y_w8 = m_w8(x)
                    y_fp16 = m_fp16(x)
                self.assertTrue(torch.allclose(y_w8, y_fp16, atol=1e-3, rtol=1e-3))

    def test_transfer_vertex_colors_from_coarse_hemispheres(self):
        # Coarse vertices in [-0.5, 0.5]^3: red for x > 0, blue for x < 0
        xs = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        ys = np.linspace(-0.5, 0.5, 5, dtype=np.float32)
        zs = np.linspace(-0.5, 0.5, 5, dtype=np.float32)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        coarse_verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
        # Filter out x == 0 so hemisphere assignment is unambiguous
        keep = np.abs(coarse_verts[:, 0]) > 0.05
        coarse_verts = coarse_verts[keep]
        coarse_colors = np.where(
            coarse_verts[:, :1] > 0.0,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        )

        refined_verts = np.array(
            [
                [0.35, 0.10, -0.10],
                [0.20, -0.25, 0.15],
                [-0.35, 0.05, 0.20],
                [-0.22, -0.15, -0.25],
            ],
            dtype=np.float32,
        )
        projected = d3d.transfer_vertex_colors_from_coarse(refined_verts, coarse_verts, coarse_colors)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(projected, expected, atol=1e-5)


class ComfyNodeAndDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.dummy_image = np.zeros((1, 64, 64, 3), dtype=np.float32)
        self.dummy_image[:, 16:48, 16:48, :] = [0.8, 0.4, 0.2]
        self.coarse_verts = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
            dtype=np.float32,
        )
        self.coarse_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        self.coarse_colors = np.full((4, 3), 0.7, dtype=np.float32)
        self.coarse_mesh = {
            "vertices": self.coarse_verts,
            "faces": self.coarse_faces,
            "vertex_colors": self.coarse_colors,
        }

    def test_refine_node_input_types_and_execution(self):
        input_types = three_d_nodes.ZiggartDirect3DV3Refine.INPUT_TYPES()
        self.assertIn("image", input_types["required"])
        self.assertIn("resolution", input_types["required"])
        self.assertIn("sharp_edges", input_types["required"])
        self.assertIn("mesh_data", input_types["optional"])
        self.assertIn("mesh_path", input_types["optional"])
        self.assertEqual(three_d_nodes.ZiggartDirect3DV3Refine.RETURN_TYPES, ("STRING", "ZIGGART_MESH", "IMAGE", "STRING"))

        def fake_run_d3d(**kwargs):
            out_path = kwargs["output_mesh_path"]
            normals = three_d_nodes.compute_vertex_normals(self.coarse_verts, self.coarse_faces)
            three_d_nodes.export_mesh_file(out_path, self.coarse_verts, self.coarse_faces, self.coarse_colors, normals)
            mesh_out = {
                "vertices": self.coarse_verts,
                "faces": self.coarse_faces,
                "vertex_colors": self.coarse_colors,
                "normals": normals,
                "path": str(out_path),
            }
            info = json.dumps({"views": list(kwargs["view_images"].keys()), "resolution": kwargs["resolution"]})
            return out_path, mesh_out, info

        node = three_d_nodes.ZiggartDirect3DV3Refine()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
                mock.patch.object(three_d_nodes.THREED_MANAGER, "run_direct3d_v3", side_effect=fake_run_d3d) as m_run,
            ):
                # 1. Execution with in-memory mesh_data
                mesh_path_1, mesh_data_1, preview_1, info_1 = node.refine_mesh(
                    image=self.dummy_image,
                    mesh_data=self.coarse_mesh,
                    resolution="1024",
                )
                self.assertTrue(Path(mesh_path_1).is_file())
                self.assertEqual(len(mesh_data_1["vertices"]), 4)
                self.assertEqual(tuple(preview_1.shape), (1, 512, 512, 3))
                self.assertEqual(json.loads(info_1)["views"], ["front"])
                self.assertEqual(m_run.call_count, 1)

                # 2. Execution with mesh_path pointing to an .obj file on disk
                input_obj = Path(tmp) / "coarse_input.obj"
                three_d_nodes.export_mesh_file(input_obj, self.coarse_verts, self.coarse_faces, self.coarse_colors)
                mesh_path_2, mesh_data_2, preview_2, info_2 = node.refine_mesh(
                    image=self.dummy_image,
                    mesh_path=str(input_obj),
                    resolution="512",
                )
                self.assertTrue(Path(mesh_path_2).is_file())
                self.assertEqual(len(mesh_data_2["faces"]), 4)
                self.assertEqual(tuple(preview_2.shape), (1, 512, 512, 3))
                self.assertEqual(json.loads(info_2)["resolution"], 512)
                self.assertEqual(m_run.call_count, 2)

    def test_refine_multiview_node_input_types_and_execution(self):
        input_types = three_d_nodes.ZiggartDirect3DV3RefineMultiview.INPUT_TYPES()
        self.assertIn("multiview_speed", input_types["required"])
        for view in ("back", "left", "right", "top", "bottom"):
            self.assertIn(f"{view}_image", input_types["optional"])
            self.assertIn(f"{view}_mask", input_types["optional"])
        self.assertEqual(
            three_d_nodes.ZiggartDirect3DV3RefineMultiview.RETURN_TYPES,
            ("STRING", "ZIGGART_MESH", "IMAGE", "STRING"),
        )

        captured_views = {}

        def fake_run_d3d(**kwargs):
            captured_views["views"] = set(kwargs["view_images"].keys())
            captured_views["multiview_speed"] = kwargs["multiview_speed"]
            out_path = kwargs["output_mesh_path"]
            normals = three_d_nodes.compute_vertex_normals(self.coarse_verts, self.coarse_faces)
            three_d_nodes.export_mesh_file(out_path, self.coarse_verts, self.coarse_faces, self.coarse_colors, normals)
            mesh_out = {
                "vertices": self.coarse_verts,
                "faces": self.coarse_faces,
                "vertex_colors": self.coarse_colors,
                "normals": normals,
                "path": str(out_path),
            }
            return out_path, mesh_out, json.dumps({"views": sorted(captured_views["views"])})

        node = three_d_nodes.ZiggartDirect3DV3RefineMultiview()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
                mock.patch.object(three_d_nodes.THREED_MANAGER, "run_direct3d_v3", side_effect=fake_run_d3d),
            ):
                mesh_path, mesh_data, preview, info_str = node.refine_mesh_multiview(
                    image=self.dummy_image,
                    multiview_speed="fast",
                    mesh_data=self.coarse_mesh,
                    back_image=self.dummy_image,
                    left_image=self.dummy_image,
                )
                self.assertTrue(Path(mesh_path).is_file())
                self.assertEqual(captured_views["views"], {"front", "back", "left"})
                self.assertEqual(captured_views["multiview_speed"], "fast")
                self.assertEqual(tuple(preview.shape), (1, 512, 512, 3))
                self.assertEqual(json.loads(info_str)["views"], ["back", "front", "left"])
                self.assertEqual(len(mesh_data["vertices"]), 4)

    def test_downloader_direct3d_v3_int8_1024_only(self):
        input_types = three_d_nodes.Ziggart3DModelDownloader.INPUT_TYPES()
        self.assertIn("download_direct3d_v3_int8", input_types["optional"])
        self.assertIn("1024_only", input_types["optional"]["download_direct3d_v3_int8"][0])

        fake_result = {
            "variant": "int8",
            "repo_id": "MostAadTech/Direct3D-V3-int8",
            "local_dir": "C:/fake/direct3d_v3/int8",
            "stages": [1024],
            "total_size_mb": 1300.0,
            "files": {"shape_transformers_1024/model.safetensors": {"status": "downloaded", "size_mb": 1300.0}},
            "status": "ready",
        }
        node = three_d_nodes.Ziggart3DModelDownloader()
        with mock.patch.object(d3d, "download_direct3d_v3_files", return_value=fake_result) as m_dl:
            (manifest_str,) = node.download_models(
                download_idol_fp16=False,
                download_idol_fp32=False,
                download_triposplat_aux=False,
                download_birefnet=False,
                download_direct3d_v3_int8="1024_only",
                download_direct3d_v3_fp16="none",
            )
        m_dl.assert_called_once_with(
            variant="int8",
            stages=(1024,),
            force_redownload=False,
            hf_token="",
        )
        manifest = json.loads(manifest_str)
        self.assertTrue(manifest["all_ready"])
        self.assertIn("direct3d_v3_int8", manifest["cached_models"])
        self.assertEqual(manifest["cached_models"]["direct3d_v3_int8"]["stages"], [1024])
        self.assertAlmostEqual(manifest["total_size_mb"], 1300.0, places=2)

    def test_w8g32_linear_preserves_input_dtype_and_forward_with_cached_windows(self):
        q8 = torch.randint(-127, 128, (32, 64), dtype=torch.int8)
        scales = torch.ones((32, 2), dtype=torch.float32) * 0.01
        bias = torch.randn((32,), dtype=torch.float16)
        layer = d3d.W8G32Linear(64, 32, q8, scales, bias=bias, group_size=32, use_triton=False)

        for dt in (torch.float16, torch.bfloat16, torch.float32):
            out = layer(torch.randn((2, 64), dtype=dt))
            self.assertEqual(out.dtype, dt)

        class FakeBlock(nn.Module):
            def forward(self, s, c, cond, index, layout, skip_value=None, window_size=None, window_cache=None):
                return s + cond[:, :1, : s.shape[-1]].squeeze(1)

        class FakeDecoder(nn.Module):
            def forward(self, x_patches, s, index, window_cache=None, window_size=None):
                return torch.ones((x_patches.shape[0], 1), dtype=s.dtype) * 2.5

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.time_factor = 1000.0
                self.depth = 2
                self.window_size = 4
                self.patch_size = 1
                self.in_channels = 4
                self.t_embedder = lambda t: t.unsqueeze(-1)
                self.x_embedder = lambda x: x
                self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])
                self.decoder = FakeDecoder()

        model = FakeModel()
        x = torch.randn((3, 4), dtype=torch.float16)
        t = torch.tensor([0.5], dtype=torch.float32)
        cond = torch.randn((3, 2, 4), dtype=torch.float32)
        out_dec = d3d.forward_with_cached_windows(model, x, None, [], t, cond, (None, None))
        self.assertTrue(torch.all(out_dec <= 1.0))
        self.assertTrue(torch.all(out_dec >= -1.0))


if __name__ == "__main__":
    unittest.main()
