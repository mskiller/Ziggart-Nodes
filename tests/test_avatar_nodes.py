import base64
import importlib.util
import io
import json
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
PACKAGE = "ziggart_avatar_test"


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
avatar_nodes = load_module("avatar_nodes", "avatar_nodes.py")


def _make_humanoid_dummy_mesh() -> dict:
    """Create a simple synthetic humanoid-proportioned triangle mesh."""
    verts = []
    faces = []
    # Stack vertical rings from feet (y=-0.75) to head (y=+0.75)
    n_rings = 16
    n_seg = 12
    for r_i in range(n_rings):
        y = -0.75 + (1.50 * r_i / float(n_rings - 1))
        radius = 0.14 if abs(y) < 0.45 else 0.12
        for s_i in range(n_seg):
            ang = (2.0 * np.pi * s_i) / n_seg
            x = np.cos(ang) * radius
            z = np.sin(ang) * radius
            # Extend arms horizontally around y in [0.30, 0.44]
            if 0.30 <= y <= 0.44 and abs(np.cos(ang)) > 0.7:
                x *= 3.2
            verts.append([x, y, z])

    for r_i in range(n_rings - 1):
        for s_i in range(n_seg):
            i0 = r_i * n_seg + s_i
            i1 = r_i * n_seg + ((s_i + 1) % n_seg)
            i2 = (r_i + 1) * n_seg + ((s_i + 1) % n_seg)
            i3 = (r_i + 1) * n_seg + s_i
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])

    v_arr = np.asarray(verts, dtype=np.float32)
    f_arr = np.asarray(faces, dtype=np.int32)
    c_arr = np.full((len(v_arr), 3), [0.85, 0.72, 0.68], dtype=np.float32)
    return {
        "vertices": v_arr,
        "faces": f_arr,
        "vertex_colors": c_arr,
        "normals": three_d_nodes.compute_vertex_normals(v_arr, f_arr),
    }


class AvatarModelAutoDownloadAndCacheTests(unittest.TestCase):
    def test_auto_download_once_and_reuse_cached_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "models_3d_cache"
            download_calls = []

            def fake_snapshot_download(repo_id, local_dir, **kwargs):
                download_calls.append((repo_id, local_dir))
                p = Path(local_dir)
                p.mkdir(parents=True, exist_ok=True)
                (p / "model.safetensors").write_bytes(b"fake_weights_data" * 100)
                return str(p)

            fake_hf = types.ModuleType("huggingface_hub")
            fake_hf.snapshot_download = fake_snapshot_download

            with (
                mock.patch.object(avatar_nodes, "get_avatar_models_root", return_value=cache_root),
                mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
                mock.patch("subprocess.run"),
            ):
                # 1st call: downloads weights and writes .ziggart_cached_manifest.json
                res1 = avatar_nodes.ensure_avatar_model_cached("hairdeck", auto_download=True)
                self.assertEqual(res1["status"], "ready")
                self.assertFalse(res1["cache_hit"])
                self.assertEqual(len(download_calls), 1)

                # 2nd call: immediately reuses cached local directory without calling snapshot_download!
                res2 = avatar_nodes.ensure_avatar_model_cached("hairdeck", auto_download=True)
                self.assertEqual(res2["status"], "ready")
                self.assertTrue(res2["cache_hit"])
                self.assertEqual(len(download_calls), 1)

                # Test ZiggartAvatarModelDownloader node
                dl_node = avatar_nodes.ZiggartAvatarModelDownloader()
                root_str, status_json = dl_node.ensure_models(
                    download_hairdeck=True,
                    download_make_it_animatable=True,
                    download_unirig=False,
                    download_birefnet_matting=False,
                    auto_download_if_missing=True,
                )
                self.assertEqual(Path(root_str), cache_root)
                parsed = json.loads(status_json)
                self.assertTrue(parsed["models"]["hairdeck"]["cache_hit"])
                self.assertEqual(parsed["models"]["make_it_animatable"]["status"], "ready")


class AvatarPipelineEndToEndTests(unittest.TestCase):
    def setUp(self):
        # Create a 128x128 synthetic character image (dark purple hair at top, peach face/body below)
        img = np.zeros((1, 128, 128, 3), dtype=np.float32)
        img[:, 16:48, 36:92, :] = [0.28, 0.14, 0.42]  # Hair crown + side locks
        img[:, 32:112, 46:82, :] = [0.92, 0.76, 0.68]  # Face & torso
        self.char_image = img

    def test_canonical_prep_and_hairdeck_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            prep_node = avatar_nodes.ZiggartAvatarCanonicalPrep()
            body_img, hair_img, hair_mask, head_crop = prep_node.prepare_avatar_inputs(
                image=self.char_image,
                head_top_ratio=0.38,
                suppress_draping_hair_on_body=True,
                target_resolution=256,
            )
            self.assertEqual(tuple(body_img.shape), (1, 256, 256, 3))
            self.assertEqual(tuple(hair_img.shape), (1, 256, 256, 3))
            self.assertEqual(tuple(hair_mask.shape), (1, 256, 256))
            self.assertEqual(tuple(head_crop.shape), (1, 512, 512, 3))

            hair_node = avatar_nodes.ZiggartHairDeckGenerator()
            with (
                mock.patch.object(avatar_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
            ):
                hair_data, glb_path, preview = hair_node.generate_hair(
                    image=hair_img,
                    style_preset="twin_tails",
                    num_strands=24,
                    segments_per_strand=8,
                    auto_download_weights=False,
                    hair_mask=hair_mask,
                )
                self.assertTrue(Path(glb_path).is_file())
                self.assertEqual(hair_data["num_strands"], 24)
                self.assertEqual(len(hair_data["spring_chains"]), 24)
                self.assertGreater(len(hair_data["vertices"]), 200)
                self.assertEqual(tuple(preview.shape), (1, 512, 512, 3))

    def test_auto_rigger_blendshapes_vrm_export_and_live_studio(self):
        mesh_data = _make_humanoid_dummy_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(avatar_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
                mock.patch.object(three_d_nodes, "_get_output_3d_dir", return_value=Path(tmp)),
            ):
                # 1. Auto-Rig
                rigger = avatar_nodes.ZiggartAutoRigger()
                rigged_mesh, rigged_glb, skel_prev = rigger.rig_character(
                    mesh_data=mesh_data,
                    rig_backend="geometric_capsule_heat",
                    skinning_smoothness=0.08,
                    auto_download_weights=False,
                )
                self.assertTrue(Path(rigged_glb).is_file())
                self.assertEqual(len(rigged_mesh["bone_names"]), 25)
                self.assertEqual(rigged_mesh["joints_0"].shape, (len(mesh_data["vertices"]), 4))
                self.assertEqual(rigged_mesh["weights_0"].shape, (len(mesh_data["vertices"]), 4))
                np.testing.assert_allclose(rigged_mesh["weights_0"].sum(axis=-1), 1.0, atol=1e-4)

                # 2. Build Facial BlendShapes
                bs_node = avatar_nodes.ZiggartFacialBlendshapeBuilder()
                rigged_with_bs, bs_sheet = bs_node.build_blendshapes(
                    rigged_mesh=rigged_mesh,
                    expression_strength=1.2,
                    include_fun_cartoon_morphs=True,
                )
                self.assertIn("aa", rigged_with_bs["morph_targets"])
                self.assertIn("blink", rigged_with_bs["morph_targets"])
                self.assertIn("happy", rigged_with_bs["morph_targets"])
                self.assertIn("chibi_big_head", rigged_with_bs["morph_targets"])
                self.assertEqual(tuple(bs_sheet.shape), (1, 512, 512, 3))

                # 3. Generate HairDeck hair & Assemble VRM 1.0 Avatar with Cat Ears
                hair_node = avatar_nodes.ZiggartHairDeckGenerator()
                hair_data, _, _ = hair_node.generate_hair(
                    image=self.char_image,
                    style_preset="anime_layered",
                    num_strands=16,
                    segments_per_strand=6,
                    auto_download_weights=False,
                )

                assembler = avatar_nodes.ZiggartVRMAssembler()
                avatar_data, vrm_path_str, vrm_preview = assembler.assemble_vrm(
                    rigged_mesh=rigged_with_bs,
                    avatar_title="Unit Test VTuber",
                    avatar_author="Ziggart",
                    export_extension="vrm",
                    hair_data=hair_data,
                    accessory="cat_ears",
                )
                vrm_path = Path(vrm_path_str)
                self.assertTrue(vrm_path.is_file())
                self.assertEqual(vrm_path.suffix, ".vrm")

                # Validate binary glTF 2.0 + VRMC_vrm + VRMC_springBone structure
                with vrm_path.open("rb") as f:
                    magic, version, total_len = struct.unpack("<4sII", f.read(12))
                    json_len, json_type = struct.unpack("<I4s", f.read(8))
                    json_payload = json.loads(f.read(json_len).decode("utf-8"))

                self.assertEqual(magic, b"glTF")
                self.assertEqual(version, 2)
                self.assertEqual(total_len, vrm_path.stat().st_size)
                self.assertEqual(json_type, b"JSON")
                self.assertIn("VRMC_vrm", json_payload["extensions"])
                self.assertIn("VRMC_springBone", json_payload["extensions"])
                self.assertGreater(len(json_payload["extensions"]["VRMC_springBone"]["springs"]), 16)
                self.assertEqual(len(json_payload["skins"][0]["joints"]), 25)

                # 4. Fun FX Node
                fx_node = avatar_nodes.ZiggartAvatarFunFX()
                avatar_fx, fx_json = fx_node.apply_fun_fx(
                    avatar_data=avatar_data,
                    physics_preset="Zero-Gravity Space Hair",
                    add_accessory="keep_existing",
                )
                self.assertIn("Zero-Gravity Space Hair", fx_json)

                # 5. Interactive VRM Pose & Live Webcam Mocap Studio Node
                # Simulate browser Live Webcam frame + pose capture sync
                buf = io.BytesIO()
                Image.new("RGBA", (128, 128), (40, 220, 120, 255)).save(buf, format="PNG")
                uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
                avatar_nodes.store_live_studio_capture(
                    "unit_test_session",
                    {
                        "rotations": {"head": [12.0, -18.0, 5.0]},
                        "expressions": {"aa": 0.85, "happy": 0.9},
                        "frame_data_uri": uri,
                    },
                )

                studio = avatar_nodes.ZiggartVRMPoseAndMocapStudio()
                rgb, mask, depth, normal, openpose, out_vrm = studio.execute_studio(
                    studio_mode="Live Webcam Mocap Capture (Browser Studio)",
                    pose_preset="Peace Sign ✌️ (Kawaii / VTuber)",
                    background_mode="dark_studio",
                    resolution=128,
                    session_id="unit_test_session",
                    avatar_data=avatar_fx,
                )
                self.assertEqual(tuple(rgb.shape), (1, 128, 128, 3))
                self.assertEqual(tuple(mask.shape), (1, 128, 128))
                self.assertEqual(tuple(depth.shape), (1, 128, 128, 3))
                self.assertEqual(tuple(normal.shape), (1, 128, 128, 3))
                self.assertEqual(tuple(openpose.shape), (1, 128, 128, 3))
                self.assertEqual(out_vrm, vrm_path_str)

                # 6. Verify loading .vrm directly from disk when avatar_data is None
                rgb2, mask2, _, _, _, out_vrm2 = studio.execute_studio(
                    studio_mode="Pose Preset / Manual Sliders",
                    pose_preset="Wave Hello 👋 (Friendly Streamer)",
                    resolution=128,
                    avatar_data=None,
                    vrm_or_glb_path=vrm_path_str,
                )
                self.assertEqual(tuple(rgb2.shape), (1, 128, 128, 3))
                self.assertEqual(tuple(mask2.shape), (1, 128, 128))
                self.assertEqual(out_vrm2, vrm_path_str)


if __name__ == "__main__":
    unittest.main()
