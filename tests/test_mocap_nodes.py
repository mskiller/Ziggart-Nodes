from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_mocap_test"


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
mocap_nodes = load_module("mocap_nodes", "mocap_nodes.py")

MOCAP_MODEL_REGISTRY = mocap_nodes.MOCAP_MODEL_REGISTRY
axis_angle_to_rotation_matrix = mocap_nodes.axis_angle_to_rotation_matrix
rotation_matrix_to_euler_angles_zyx = mocap_nodes.rotation_matrix_to_euler_angles_zyx
retarget_smplh_to_vrm = mocap_nodes.retarget_smplh_to_vrm
run_flowhmr_motion_extraction = mocap_nodes.run_flowhmr_motion_extraction
export_mocap_to_bvh = mocap_nodes.export_mocap_to_bvh
is_mocap_model_cached = mocap_nodes.is_mocap_model_cached
ZiggartFlowHMRModelLoader = mocap_nodes.ZiggartFlowHMRModelLoader
ZiggartFlowHMRWebcamCapture = mocap_nodes.ZiggartFlowHMRWebcamCapture
ZiggartFlowHMRVideoExtractor = mocap_nodes.ZiggartFlowHMRVideoExtractor
ZiggartMocapRetargetVRM = mocap_nodes.ZiggartMocapRetargetVRM
ZiggartAvatarMocapDrive = mocap_nodes.ZiggartAvatarMocapDrive
ZiggartMocapExporter = mocap_nodes.ZiggartMocapExporter

generate_stylized_hair_cards_and_chains = avatar_nodes.generate_stylized_hair_cards_and_chains
auto_rig_humanoid_mesh = avatar_nodes.auto_rig_humanoid_mesh
build_avatar_facial_blendshapes = avatar_nodes.build_avatar_facial_blendshapes
ZiggartVRMPoseAndMocapStudio = avatar_nodes.ZiggartVRMPoseAndMocapStudio


def test_mocap_model_registry_and_cache_paths():
    assert "motius_flowhmr_base" in MOCAP_MODEL_REGISTRY
    assert "flowhmr_latest" in MOCAP_MODEL_REGISTRY
    cached, target_dir, _ = is_mocap_model_cached("motius_flowhmr_base")
    assert isinstance(target_dir, Path)
    assert target_dir.name == "flowhmr_base"


def test_axis_angle_and_euler_conversions():
    # Identity rotation
    zero_vec = np.zeros(3, dtype=np.float32)
    R_ident = axis_angle_to_rotation_matrix(zero_vec)
    assert np.allclose(R_ident, np.eye(3), atol=1e-5)
    pitch, yaw, roll = rotation_matrix_to_euler_angles_zyx(R_ident)
    assert abs(pitch) < 1e-4 and abs(yaw) < 1e-4 and abs(roll) < 1e-4

    # 90 degrees around X-axis (Pitch = 90 deg)
    rx = np.array([math.pi / 2.0, 0.0, 0.0], dtype=np.float32)
    Rx = axis_angle_to_rotation_matrix(rx)
    p_x, y_x, r_x = rotation_matrix_to_euler_angles_zyx(Rx)
    assert np.isclose(p_x, 90.0, atol=1e-3)
    assert np.isclose(y_x, 0.0, atol=1e-3)
    assert np.isclose(r_x, 0.0, atol=1e-3)


def test_retarget_smplh_to_vrm_ground_snap_and_bones():
    num_frames = 10
    dummy_smplh = np.zeros((num_frames, 52, 3), dtype=np.float32)
    dummy_trans = np.zeros((num_frames, 3), dtype=np.float32)
    dummy_trans[:, 1] = -0.15  # Negative root position (penetrating ground)

    # Set some rotations on leftUpperArm (index 16) and hips (index 0)
    dummy_smplh[:, 16] = [0.3, 0.0, 0.0]
    dummy_smplh[:, 0] = [0.0, 0.1, 0.0]

    vrm_mocap = retarget_smplh_to_vrm(
        dummy_smplh,
        dummy_trans,
        ground_snap=True,
        smoothing_factor=0.2,
        root_motion_mode="Full 3D Root Motion",
        fps=30.0,
    )

    assert vrm_mocap["frame_count"] == num_frames
    assert len(vrm_mocap["bone_rotations_deg"]) == num_frames
    assert "leftUpperArm" in vrm_mocap["bone_rotations_deg"][0]
    assert "hips" in vrm_mocap["bone_rotations_deg"][0]

    # Ground snap should shift min Y to 0.0
    trans_arr = np.asarray(vrm_mocap["root_translations"])
    assert float(np.min(trans_arr[:, 1])) >= -1e-5


def test_bvh_export(tmp_path: Path):
    num_frames = 5
    mocap_data = {
        "frame_count": num_frames,
        "fps": 30.0,
        "bone_rotations_deg": [
            {"hips": [0.0, 0.0, 0.0], "spine": [5.0, 0.0, 0.0], "chest": [3.0, 0.0, 0.0], "neck": [0.0, 0.0, 0.0], "head": [0.0, 0.0, 0.0]}
            for _ in range(num_frames)
        ],
        "root_translations": [[0.0, 0.0, 0.0] for _ in range(num_frames)],
    }
    bvh_file = tmp_path / "test_motion.bvh"
    res = export_mocap_to_bvh(mocap_data, bvh_file)
    assert res.is_file()
    content = res.read_text(encoding="utf-8")
    assert "HIERARCHY" in content
    assert "ROOT Hips" in content
    assert "MOTION" in content
    assert f"Frames: {num_frames}" in content


def test_mocap_nodes_end_to_end_drive(tmp_path: Path):
    # 1. Model Loader
    loader = ZiggartFlowHMRModelLoader()
    model_handle, info = loader.load_model(
        model_variant="Motius-FlowHMR-Base (Hugging Face)",
        device="cpu",
        use_static_camera=True,
        auto_download_weights=False,
    )
    assert model_handle["ready"] is True
    assert "FlowHMR Model Ready" in info

    # 2. Webcam Capture Simulator
    capture = ZiggartFlowHMRWebcamCapture()
    img_batch, f_count, fps = capture.capture_webcam(
        camera_index=99,  # trigger fallback
        duration_seconds=0.2,
        target_fps=15,
        resolution=128,
    )
    assert f_count == 3
    assert fps == 15.0

    # 3. Video Extractor
    extractor = ZiggartFlowHMRVideoExtractor()
    mocap, preview, summary = extractor.extract_mocap(
        model_handle,
        img_batch,
        camera_mode="static_camera (Webcam / Fixed Shot)",
        flow_steps=10,
        fps=fps,
    )
    assert mocap["frame_count"] == 3
    assert "FlowHMR Mocap Extracted" in summary

    # 4. Retargeter
    retargeter = ZiggartMocapRetargetVRM()
    retargeted, r_log = retargeter.retarget_mocap(mocap, root_motion_mode="Full 3D Root Motion")
    assert retargeted["frame_count"] == 3

    # 5. Drive Procedural Avatar
    dummy_pil = Image.new("RGB", (128, 128), (200, 150, 220))
    hair = generate_stylized_hair_cards_and_chains(dummy_pil, num_strands=10, segments_per_strand=4)
    rigged = auto_rig_humanoid_mesh(hair)
    rigged = build_avatar_facial_blendshapes(rigged)
    avatar_data = {
        **rigged,
        "spring_chains": hair["spring_chains"],
        "hair_vertex_start": 0,
        "hair_vertex_count": len(hair["vertices"]),
    }

    driver = ZiggartAvatarMocapDrive()
    rgb_b, mask_b, depth_b, norm_b, pose_b, glb_path = driver.drive_avatar(
        avatar_data,
        retargeted,
        resolution=128,
        camera_azimuth=15.0,
        camera_elevation=10.0,
        export_animated_glb=True,
        filename_prefix="test_mocap_drive",
    )

    # Check batch shapes (T=3 frames)
    if torch is not None and isinstance(rgb_b, torch.Tensor):
        assert rgb_b.shape[0] == 3
        assert mask_b.shape[0] == 3
        assert depth_b.shape[0] == 3
    else:
        assert len(rgb_b) == 3

    # 6. Exporter
    exporter = ZiggartMocapExporter()
    exported_bvh = exporter.export_mocap(retargeted, format="BVH", filename_prefix="test_exp_mocap")
    assert Path(exported_bvh[0]).is_file()

    # 7. VRM Studio with Mocap Support
    studio = ZiggartVRMPoseAndMocapStudio()
    s_rgb, s_mask, s_depth, s_norm, s_pose, v_path = studio.execute_studio(
        studio_mode="FlowHMR Mocap Animation Drive",
        avatar_data=avatar_data,
        mocap_data=retargeted,
        resolution=128,
    )
    if torch is not None and isinstance(s_rgb, torch.Tensor):
        assert s_rgb.shape[0] == 3
