import importlib.util
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_3d_test"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
embedded_model = load_module("embedded_model", "embedded_model.py")
three_d_nodes = load_module("three_d_nodes", "three_d_nodes.py")


def _make_sphere_gaussians(count: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    dirs = rng.normal(size=(count, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8
    radii = rng.uniform(0.35, 0.50, size=(count, 1)).astype(np.float32)
    positions = dirs * radii
    colors = np.clip(0.5 + 0.5 * dirs, 0.0, 1.0).astype(np.float32)
    opacities = np.full(count, 0.9, dtype=np.float32)
    scales = np.full((count, 3), 0.04, dtype=np.float32)
    return positions, colors, opacities, scales


class GaussianPlyAndMeshConversionTests(unittest.TestCase):
    def test_save_and_load_gaussian_ply_roundtrip(self):
        positions, colors, opacities, scales = _make_sphere_gaussians(64)
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = Path(tmp) / "test_splat.ply"
            three_d_nodes.save_gaussian_ply(ply_path, positions, colors, opacities, scales)
            loaded = three_d_nodes.load_gaussian_ply(ply_path)

            self.assertEqual(loaded["num_gaussians"], 64)
            np.testing.assert_allclose(loaded["positions"], positions, atol=1e-4)
            np.testing.assert_allclose(loaded["colors"], colors, atol=1e-3)
            np.testing.assert_allclose(loaded["opacities"], opacities, atol=1e-2)
            np.testing.assert_allclose(loaded["scales"], scales, atol=1e-4)

    def test_gaussian_to_mesh_node_exports_valid_glb_and_obj(self):
        positions, colors, opacities, scales = _make_sphere_gaussians(500)
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = Path(tmp) / "idol_splat.ply"
            three_d_nodes.save_gaussian_ply(ply_path, positions, colors, opacities, scales)

            node = three_d_nodes.ZiggartGaussianToMesh()
            with mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)):
                glb_path_str, mesh_data, preview = node.convert_to_mesh(
                    method="gaussian_density_surface",
                    export_format="glb",
                    opacity_threshold=0.15,
                    grid_resolution=48,
                    target_faces=5000,
                    smoothing_iterations=2,
                    filename_prefix="test_mesh",
                    ply_path=str(ply_path),
                )
                glb_path = Path(glb_path_str)
                self.assertTrue(glb_path.is_file())
                with glb_path.open("rb") as f:
                    magic, version, length = struct.unpack("<4sII", f.read(12))
                self.assertEqual(magic, b"glTF")
                self.assertEqual(version, 2)
                self.assertEqual(length, glb_path.stat().st_size)
                self.assertGreater(len(mesh_data["vertices"]), 10)
                self.assertGreater(len(mesh_data["faces"]), 10)
                self.assertEqual(tuple(preview.shape), (1, 512, 512, 3))

                obj_path_str, _, _ = node.convert_to_mesh(
                    method="gaussian_density_surface",
                    export_format="obj",
                    opacity_threshold=0.15,
                    grid_resolution=48,
                    target_faces=2000,
                    smoothing_iterations=1,
                    filename_prefix="test_obj",
                    gsplat_data=three_d_nodes.load_gaussian_ply(ply_path),
                )
                self.assertTrue(Path(obj_path_str).is_file())
                self.assertIn("v ", Path(obj_path_str).read_text(encoding="utf-8"))


class ThreeDNodesMockedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.dummy_image = np.zeros((1, 64, 64, 3), dtype=np.float32)
        self.dummy_image[:, 16:48, 16:48, :] = [0.9, 0.3, 0.2]

    def test_idol_node_generates_ply_and_preview(self):
        positions, colors, opacities, scales = _make_sphere_gaussians(128)
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run_idol(**kwargs):
                out_ply = kwargs["output_ply_path"]
                three_d_nodes.save_gaussian_ply(out_ply, positions, colors, opacities, scales)
                return out_ply, kwargs["pil_image"]

            node = three_d_nodes.ZiggartIdolImageToGaussian()
            with (
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
                mock.patch.object(three_d_nodes.THREED_MANAGER, "run_idol", side_effect=fake_run_idol),
            ):
                ply_path, gsplat_data, pre_img, preview = node.generate_gaussian(
                    image=self.dummy_image,
                    mode="synthetic",
                    num_gaussians="262144",
                    precision="bf16",
                    seed=7,
                )
                self.assertTrue(Path(ply_path).is_file())
                self.assertEqual(gsplat_data["num_gaussians"], 128)
                self.assertEqual(tuple(pre_img.shape), (1, 64, 64, 3))
                self.assertEqual(tuple(preview.shape), (1, 512, 512, 3))

    def test_idol_pipeline_handles_broken_flash_attn_and_idol_output(self):
        positions, colors, opacities, scales = _make_sphere_gaussians(64)

        class FakeGaussian:
            def save_ply(self, path):
                three_d_nodes.save_gaussian_ply(path, positions, colors, opacities, scales)

        class FakeIdolOutput:
            def __init__(self, pil_img):
                self.gaussian = FakeGaussian()
                self.preprocessed_image = pil_img
                self.camera = {"yaw": 0.0, "pitch": 0.0, "radius": 2.0, "fov": 40.0}

        class FakeIdolPipeline:
            dinov3_dtype_passed = None

            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):
                cls.dinov3_dtype_passed = kwargs.get("dinov3_dtype")
                return cls()

            def __call__(self, pil_image, **kwargs):
                if fake_layers.ATTN_BACKEND == "flash_attn":
                    # Simulates the Windows broken flash_attn_func=None crash
                    return broken_flash_attn.flash_attn_func()
                return FakeIdolOutput(pil_image)

        broken_flash_attn = types.ModuleType("flash_attn")
        broken_flash_attn.flash_attn_func = None
        broken_flash_attn.flash_attn_qkvpacked_func = None

        fake_layers = types.ModuleType("idol.models.layers")
        fake_layers.ATTN_BACKEND = "flash_attn"
        fake_models = types.ModuleType("idol.models")
        fake_models.layers = fake_layers
        fake_idol = types.ModuleType("idol")
        fake_idol.IdolPipeline = FakeIdolPipeline
        fake_idol.models = fake_models

        with tempfile.TemporaryDirectory() as tmp:
            node = three_d_nodes.ZiggartIdolImageToGaussian()
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "flash_attn": broken_flash_attn,
                        "idol": fake_idol,
                        "idol.models": fake_models,
                        "idol.models.layers": fake_layers,
                    },
                ),
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
            ):
                ply_path, gsplat_data, pre_img, preview = node.generate_gaussian(
                    image=self.dummy_image,
                    mode="synthetic",
                    num_gaussians="262144",
                    precision="bf16",
                    seed=42,
                )
                self.assertEqual(fake_layers.ATTN_BACKEND, "sdpa")
                self.assertEqual(FakeIdolPipeline.dinov3_dtype_passed, "bf16")
                self.assertTrue(Path(ply_path).is_file())
                self.assertEqual(gsplat_data["num_gaussians"], 64)

    def test_spar3d_node_runs_with_low_vram_mode(self):
        class FakeMesh:
            vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int32)
            visual = None

            def export(self, path):
                three_d_nodes.export_mesh_file(path, self.vertices, self.faces)

        class FakeSPAR3D:
            low_vram_passed = None

            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):
                cls.low_vram_passed = kwargs.get("low_vram_mode")
                return cls()

            def to(self, device):
                return self

            def eval(self):
                return self

            def run_image(self, image, **kwargs):
                return FakeMesh(), None

        fake_mod = types.ModuleType("spar3d")
        fake_mod.SPAR3D = FakeSPAR3D

        with tempfile.TemporaryDirectory() as tmp:
            node = three_d_nodes.ZiggartSPAR3DImageToMesh()
            with (
                mock.patch.dict(sys.modules, {"spar3d": fake_mod}),
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
            ):
                glb_path, mesh_data, ply_path, preview = node.generate_mesh(
                    image=self.dummy_image,
                    low_vram_mode=True,
                    bake_resolution=1024,
                )
                self.assertTrue(FakeSPAR3D.low_vram_passed)
                self.assertEqual(os.environ.get("SPAR3D_LOW_VRAM"), "1")
                self.assertTrue(Path(glb_path).is_file())
                self.assertTrue(Path(ply_path).is_file())
                self.assertEqual(len(mesh_data["faces"]), 1)
                self.assertEqual(tuple(preview.shape), (1, 512, 512, 3))

    def test_hunyuan3d_node_unloads_shape_before_paint(self):
        events = []

        class FakeMesh:
            vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.int32)
            visual = None

            def export(self, path):
                three_d_nodes.export_mesh_file(path, self.vertices, self.faces)

        class FakeShapePipe:
            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):
                events.append(("load_shape", repo_id, kwargs.get("subfolder")))
                return cls()

            def __call__(self, **kwargs):
                events.append(("run_shape", kwargs.get("octree_resolution")))
                return [FakeMesh()]

        class FakePaintPipe:
            @classmethod
            def from_pretrained(cls, repo_id, **kwargs):
                # Shape pipeline must already be unloaded before Paint loads
                events.append(("load_paint_while_active_is", three_d_nodes.THREED_MANAGER._pipeline))
                return cls()

            def enable_model_cpu_offload(self):
                events.append(("enable_cpu_offload",))

            def __call__(self, mesh, image):
                events.append(("run_paint",))
                return mesh

        hy3d_pkg = types.ModuleType("hy3dgen")
        shapegen_mod = types.ModuleType("hy3dgen.shapegen")
        shapegen_mod.Hunyuan3DDiTFlowMatchingPipeline = FakeShapePipe
        texgen_mod = types.ModuleType("hy3dgen.texgen")
        texgen_mod.Hunyuan3DPaintPipeline = FakePaintPipe

        with tempfile.TemporaryDirectory() as tmp:
            node = three_d_nodes.ZiggartHunyuan3DImageToMesh()
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "hy3dgen": hy3d_pkg,
                        "hy3dgen.shapegen": shapegen_mod,
                        "hy3dgen.texgen": texgen_mod,
                    },
                ),
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
            ):
                glb_path, mesh_data, pre_img, preview = node.generate_mesh(
                    image=self.dummy_image,
                    model_variant="Hunyuan3D-2mini (0.6B - 6GB VRAM)",
                    steps=20,
                    octree_resolution=256,
                    generate_texture=True,
                )
                self.assertTrue(Path(glb_path).is_file())
                self.assertIn(("load_shape", "tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini"), events)
                self.assertIn(("load_paint_while_active_is", None), events)
                self.assertIn(("enable_cpu_offload",), events)
                self.assertIn(("run_paint",), events)

    def test_turntable_renderer_outputs_batch(self):
        positions, colors, opacities, scales = _make_sphere_gaussians(64)
        gsplat = {
            "positions": positions,
            "colors": colors,
            "opacities": opacities,
            "scales": scales,
        }
        renderer = three_d_nodes.Ziggart3DTurntableRenderer()
        images, masks = renderer.render_turntable(
            num_frames=6,
            azimuth_start=0.0,
            azimuth_sweep=360.0,
            elevation=15.0,
            resolution=128,
            gsplat_data=gsplat,
        )
        self.assertEqual(tuple(images.shape), (6, 128, 128, 3))
        self.assertEqual(tuple(masks.shape), (6, 128, 128))


if __name__ == "__main__":
    unittest.main()
