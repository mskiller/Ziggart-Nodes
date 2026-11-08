from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from server import PromptServer
    from aiohttp import web
except Exception:
    PromptServer = None
    web = None

from .embedded_model import empty_accelerator_cache, release_comfy_vram
from .three_d_nodes import (
    THREED_MANAGER,
    Ziggart3DError,
    _ensure_backend_installed,
    _get_output_3d_dir,
    _next_output_path,
    compute_vertex_normals,
    decimate_mesh_vertex_clustering,
    export_mesh_file,
    laplacian_smooth_mesh,
    numpy_batch_to_tensor,
    pil_to_tensor,
    remove_background_fallback,
    render_3d_view,
    render_4view_contact_sheet,
    tensor_to_pil,
)


# ---------------------------------------------------------------------------
# 1. Persistent Model Auto-Downloader & Local Cache Registry
# ---------------------------------------------------------------------------

AVATAR_MODEL_FOLDER = "ziggart_avatars"

AVATAR_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "hairdeck": {
        "repo_id": "hyz317/HairDeck",
        "git_repo": "https://github.com/visualbruno/ComfyUI-HairDeck.git",
        "subfolder": "hairdeck",
        "description": "HairDeck 3D stylized hair cards + spring-bone chain diffusion weights (~1.5 GB)",
        "gated": False,
    },
    "make_it_animatable": {
        "repo_id": "jasongzy/Make-It-Animatable",
        "git_repo": "https://github.com/jasongzy/Make-It-Animatable.git",
        "subfolder": "make_it_animatable",
        "description": "Make-It-Animatable fast humanoid auto-rigger & skinning weights (~1.2 GB)",
        "gated": False,
    },
    "unirig": {
        "repo_id": "VAST-AI/UniRig",
        "git_repo": "https://github.com/VAST-AI-Research/UniRig.git",
        "subfolder": "unirig",
        "description": "UniRig autoregressive universal skeleton & skinning model (~2.1 GB)",
        "gated": False,
    },
    "birefnet_matting": {
        "repo_id": "ZhengPeng7/BiRefNet-matting",
        "git_repo": "",
        "subfolder": "birefnet_matting",
        "description": "BiRefNet high-precision hair strand alpha matting weights (~0.85 GB)",
        "gated": False,
    },
}


def get_avatar_models_root() -> Path:
    """
    Returns the persistent directory where 3D avatar models (HairDeck, MIA, UniRig)
    are downloaded once and kept permanently across ComfyUI restarts.
    """
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        root = Path(folder_paths.models_dir) / "3d" / AVATAR_MODEL_FOLDER
    else:
        root = Path(__file__).resolve().parent / ".cache" / "models_3d" / AVATAR_MODEL_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    if folder_paths is not None and hasattr(folder_paths, "add_model_folder_path"):
        try:
            folder_paths.add_model_folder_path(AVATAR_MODEL_FOLDER, str(root), is_default=True)
        except TypeError:
            try:
                folder_paths.add_model_folder_path(AVATAR_MODEL_FOLDER, str(root))
            except Exception:
                pass
    return root


def is_model_cached_locally(model_key: str, custom_repo_id: str = "") -> tuple[bool, Path, dict[str, Any]]:
    """
    Checks if a model is already downloaded in the persistent local cache directory.
    Returns (is_cached, target_dir, manifest_dict).
    """
    spec = AVATAR_MODEL_REGISTRY.get(model_key, {})
    repo_id = (custom_repo_id or spec.get("repo_id") or model_key).strip()
    subfolder = spec.get("subfolder") or repo_id.replace("/", "_")
    target_dir = get_avatar_models_root() / subfolder
    manifest_path = target_dir / ".ziggart_cached_manifest.json"

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "ready" and manifest.get("repo_id") == repo_id:
                return True, target_dir, manifest
        except Exception:
            pass

    # Also recognize already-populated directories containing weights (.safetensors, .pth, .ckpt, .bin, .onnx)
    if target_dir.is_dir():
        weight_files = (
            list(target_dir.rglob("*.safetensors"))
            + list(target_dir.rglob("*.pth"))
            + list(target_dir.rglob("*.pt"))
            + list(target_dir.rglob("*.ckpt"))
            + list(target_dir.rglob("*.bin"))
            + list(target_dir.rglob("*.onnx"))
        )
        if weight_files:
            total_bytes = sum(f.stat().st_size for f in weight_files if f.is_file())
            manifest = {
                "status": "ready",
                "model_key": model_key,
                "repo_id": repo_id,
                "local_dir": str(target_dir),
                "num_weight_files": len(weight_files),
                "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            try:
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            except OSError:
                pass
            return True, target_dir, manifest

    return False, target_dir, {}


def ensure_avatar_model_cached(
    model_key: str,
    *,
    custom_repo_id: str = "",
    hf_token: str = "",
    auto_download: bool = True,
    force_redownload: bool = False,
) -> dict[str, Any]:
    """
    Ensures the requested 3D avatar model is downloaded once and cached permanently on disk.
    If already cached and force_redownload=False, returns immediately in <1ms without network traffic.
    """
    spec = AVATAR_MODEL_REGISTRY.get(model_key, {})
    repo_id = (custom_repo_id or spec.get("repo_id") or model_key).strip()

    if not force_redownload:
        cached, target_dir, manifest = is_model_cached_locally(model_key, repo_id)
        if cached:
            manifest["cache_hit"] = True
            return manifest
    else:
        _, target_dir, _ = is_model_cached_locally(model_key, repo_id)

    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / ".ziggart_cached_manifest.json"

    if not auto_download:
        return {
            "status": "not_downloaded",
            "cache_hit": False,
            "model_key": model_key,
            "repo_id": repo_id,
            "local_dir": str(target_dir),
            "num_weight_files": 0,
            "total_size_mb": 0.0,
        }

    if hf_token and hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token.strip()
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()

    # Also ensure the git wrapper/code repo is cloned in .cache/vendor_3d if specified
    git_repo = spec.get("git_repo", "")
    vendor_dir = Path(__file__).resolve().parent / ".cache" / "vendor_3d" / (spec.get("subfolder") or model_key)
    if git_repo and not (vendor_dir / ".git").exists():
        vendor_dir.parent.mkdir(parents=True, exist_ok=True)
        git_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", git_repo, str(vendor_dir)],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
                env=git_env,
            )
        except Exception:
            pass
    if vendor_dir.is_dir() and str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))

    # Download weights via huggingface_hub.snapshot_download into target_dir
    download_error: str | None = None
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        token_arg = hf_token.strip() if (hf_token and hf_token.strip()) else None
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            token=token_arg,
            force_download=bool(force_redownload),
        )
    except Exception as exc:
        download_error = str(exc)

    all_files = [f for f in target_dir.rglob("*") if f.is_file() and f.name != ".ziggart_cached_manifest.json"]
    total_bytes = sum(f.stat().st_size for f in all_files)

    status = "ready" if (all_files and download_error is None) else ("partial_or_offline" if download_error else "ready")
    manifest = {
        "status": status,
        "cache_hit": False,
        "model_key": model_key,
        "repo_id": repo_id,
        "local_dir": str(target_dir),
        "vendor_dir": str(vendor_dir) if vendor_dir.is_dir() else "",
        "num_files": len(all_files),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if download_error:
        manifest["download_note"] = download_error
    if status == "ready":
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except OSError:
            pass
    return manifest


# ---------------------------------------------------------------------------
# 2. Canonical 2D Avatar Preprocessing & Hair / Body Segmentation
# ---------------------------------------------------------------------------


def segment_avatar_hair_and_body(
    pil_img: Image.Image,
    *,
    head_top_ratio: float = 0.36,
    inpaint_scalp: bool = True,
    target_resolution: int = 1024,
) -> dict[str, Any]:
    """
    Prepares a character image for dual 3D Avatar + HairDeck generation:
      1. Removes background to clean RGBA.
      2. Centers & normalizes character height in canonical frame.
      3. Segments the hair region in the upper portrait zone using color/chroma clustering
         around the crown/scalp vs skin/face tone.
      4. Produces a 'body_ready_pil' with protruding lateral hair suppressed/smoothed so
         body 3D generators (Hunyuan3D / SPAR3D / Idol) don't fuse hair blobs onto shoulders.
    """
    rgba = remove_background_fallback(pil_img)
    res = int(np.clip(target_resolution, 256, 2048))

    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    alpha = arr[..., 3]
    fg_y, fg_x = np.where(alpha > 0.15)

    if len(fg_y) > 16:
        y0, y1 = int(fg_y.min()), int(fg_y.max())
        x0, x1 = int(fg_x.min()), int(fg_x.max())
        crop = rgba.crop((x0, y0, x1 + 1, y1 + 1))
        cw, ch = crop.size
        scale = (res * 0.86) / float(max(cw, ch, 1))
        nw = max(16, int(round(cw * scale)))
        nh = max(16, int(round(ch * scale)))
        resized = crop.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
        ox = (res - nw) // 2
        oy = (res - nh) // 2
        canvas.paste(resized, (ox, oy))
        norm_rgba = canvas
        char_bbox = (ox, oy, ox + nw, oy + nh)
    else:
        norm_rgba = rgba.resize((res, res), Image.Resampling.LANCZOS)
        char_bbox = (int(res * 0.15), int(res * 0.08), int(res * 0.85), int(res * 0.92))

    norm_arr = np.asarray(norm_rgba, dtype=np.float32) / 255.0
    rgb = norm_arr[..., :3]
    fg_mask = (norm_arr[..., 3] > 0.15).astype(np.float32)

    bx0, by0, bx1, by1 = char_bbox
    char_h = max(32, by1 - by0)
    char_w = max(32, bx1 - bx0)
    cx = int(0.5 * (bx0 + bx1))

    # Head bounding box in upper portion of character
    head_y0 = max(0, by0 - int(0.02 * char_h))
    head_y1 = min(res, by0 + int(float(head_top_ratio) * char_h))
    head_x0 = max(0, cx - int(0.35 * char_w))
    head_x1 = min(res, cx + int(0.35 * char_w))

    # Estimate hair color from top crown strip vs skin color from central face strip
    crown_y0 = by0
    crown_y1 = min(res, by0 + max(6, int(0.08 * char_h)))
    crown_slice = rgb[crown_y0:crown_y1, head_x0:head_x1]
    crown_fg = fg_mask[crown_y0:crown_y1, head_x0:head_x1] > 0.5
    if np.any(crown_fg):
        hair_ref_color = np.median(crown_slice[crown_fg], axis=0)
    else:
        hair_ref_color = np.array([0.25, 0.18, 0.15], dtype=np.float32)

    face_y0 = min(res - 1, by0 + int(0.10 * char_h))
    face_y1 = min(res, by0 + int(0.22 * char_h))
    face_x0 = max(0, cx - int(0.12 * char_w))
    face_x1 = min(res, cx + int(0.12 * char_w))
    face_slice = rgb[face_y0:face_y1, face_x0:face_x1]
    face_fg = fg_mask[face_y0:face_y1, face_x0:face_x1] > 0.5
    if np.any(face_fg):
        skin_ref_color = np.median(face_slice[face_fg], axis=0)
    else:
        skin_ref_color = np.array([0.88, 0.74, 0.66], dtype=np.float32)

    # Compute hair likelihood in upper 48% of character
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    rel_y = (yy - by0) / float(char_h)
    rel_x = (xx - cx) / float(max(char_w * 0.5, 1.0))

    dist_hair = np.linalg.norm(rgb - hair_ref_color[None, None, :], axis=-1)
    dist_skin = np.linalg.norm(rgb - skin_ref_color[None, None, :], axis=-1)

    in_hair_zone = (rel_y >= -0.02) & (rel_y <= float(head_top_ratio) * 1.25) & (np.abs(rel_x) <= 0.75)
    # Top cap of head is always hair; lower head pixels are hair if closer to crown color than skin color
    is_top_crown = (rel_y >= 0.0) & (rel_y <= 0.09) & (np.abs(rel_x) <= 0.45)
    color_prefers_hair = dist_hair < (dist_skin + 0.06)
    # Exclude central facial oval (eyes/nose/mouth)
    face_oval = ((rel_y - 0.16) / 0.10) ** 2 + (rel_x / 0.22) ** 2 < 1.0

    hair_mask = fg_mask * (
        (is_top_crown | (in_hair_zone & color_prefers_hair & (~face_oval))).astype(np.float32)
    )

    # Smooth hair mask slightly
    hair_mask_pil = Image.fromarray((np.clip(hair_mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    hair_mask_pil = hair_mask_pil.filter(ImageFilter.GaussianBlur(radius=1.5))
    hair_mask = np.asarray(hair_mask_pil, dtype=np.float32) / 255.0

    # Build hair-only RGBA image
    hair_rgba_arr = np.dstack([rgb, norm_arr[..., 3] * hair_mask])
    hair_rgba_pil = Image.fromarray((np.clip(hair_rgba_arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")

    # Build body-ready RGBA image (suppresses wide/draping side hair outside neck/head silhouette)
    body_arr = norm_arr.copy()
    if inpaint_scalp:
        # Side hair strands hanging below head/ears or wider than skull silhouette are made transparent
        skull_oval = ((rel_y - 0.13) / 0.13) ** 2 + (rel_x / 0.25) ** 2 <= 1.0
        draping_side_hair = (hair_mask > 0.35) & (~skull_oval)
        body_arr[draping_side_hair, 3] *= 0.15
        # Softly blend remaining crown cap toward neutral scalp tone so 3D mesh skull is clean
        crown_cap = (hair_mask > 0.35) & skull_oval
        if np.any(crown_cap):
            body_arr[crown_cap, :3] = 0.55 * body_arr[crown_cap, :3] + 0.45 * skin_ref_color[None, :]

    body_ready_pil = Image.fromarray((np.clip(body_arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")

    # Head/Face crop for HairDeck & Facial expression reference
    pad_h = int(0.06 * char_h)
    crop_box = (
        max(0, head_x0 - pad_h),
        max(0, head_y0 - pad_h),
        min(res, head_x1 + pad_h),
        min(res, head_y1 + pad_h),
    )
    head_crop_pil = norm_rgba.crop(crop_box).resize((512, 512), Image.Resampling.LANCZOS)

    return {
        "normalized_rgba": norm_rgba,
        "body_ready_pil": body_ready_pil,
        "hair_rgba_pil": hair_rgba_pil,
        "head_crop_pil": head_crop_pil,
        "hair_mask": hair_mask,
        "fg_mask": fg_mask,
        "hair_ref_color": hair_ref_color.tolist(),
        "skin_ref_color": skin_ref_color.tolist(),
        "char_bbox": char_bbox,
    }


# ---------------------------------------------------------------------------
# 3. HairDeck 3D Stylized Hair Cards & Spring-Bone Chain Generator
# ---------------------------------------------------------------------------


def generate_stylized_hair_cards_and_chains(
    pil_image: Image.Image,
    hair_mask: np.ndarray | None = None,
    *,
    style_preset: str = "anime_layered",
    num_strands: int = 36,
    segments_per_strand: int = 10,
    strand_width: float = 0.038,
    strand_thickness: float = 0.012,
    length_scale: float = 1.0,
    stiffness: float = 0.45,
    gravity_power: float = 0.08,
    drag_force: float = 0.35,
    head_center: tuple[float, float, float] = (0.0, 0.58, 0.0),
    head_radius: float = 0.16,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generates 3D stylized Hair Cards (structured ribbon meshes with thickness, UV coordinates,
    and per-strand colors sampled from the character image) AND hierarchical VRM Spring-Bone
    physics chains (`spring_chains`) ready for live physics simulation and VRM export.
    """
    rng = np.random.default_rng(int(seed))
    rgb_arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    h_img, w_img = rgb_arr.shape[:2]

    # Sample palette colors from the upper hair region of the image
    if hair_mask is not None and hair_mask.shape[:2] == (h_img, w_img) and np.any(hair_mask > 0.25):
        ys, xs = np.where(hair_mask > 0.25)
        sampled_colors = rgb_arr[ys, xs]
    else:
        top_crop = rgb_arr[: max(8, h_img // 3), max(0, w_img // 4) : min(w_img, 3 * w_img // 4)]
        sampled_colors = top_crop.reshape(-1, 3)

    if len(sampled_colors) > 0:
        base_color = np.median(sampled_colors, axis=0).astype(np.float32)
        highlight_color = np.percentile(sampled_colors, 85, axis=0).astype(np.float32)
        shadow_color = np.percentile(sampled_colors, 20, axis=0).astype(np.float32)
    else:
        base_color = np.array([0.28, 0.18, 0.35], dtype=np.float32)
        highlight_color = np.clip(base_color * 1.35, 0.0, 1.0)
        shadow_color = base_color * 0.65

    hx, hy, hz = float(head_center[0]), float(head_center[1]), float(head_center[2])
    r_head = float(head_radius)
    n_strands = int(np.clip(num_strands, 8, 128))
    n_segs = int(np.clip(segments_per_strand, 4, 24))

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_uvs: list[np.ndarray] = []
    spring_chains: list[dict[str, Any]] = []

    vert_offset = 0

    for s_idx in range(n_strands):
        # Distribute strands around scalp dome: front bangs, side locks, and back/ponytail cascade
        frac = s_idx / float(max(n_strands, 1))
        if style_preset == "twin_tails":
            # Cluster side strands into left/right twin-tails + bangs
            if s_idx < n_strands // 4:
                azimuth = rng.uniform(-0.65, 0.65)  # Front bangs (+Z is front)
                strand_len = r_head * 1.15 * float(length_scale)
                flare = 0.08
            elif s_idx % 2 == 0:
                azimuth = rng.uniform(1.15, 2.05)  # Left twin-tail
                strand_len = r_head * 2.8 * float(length_scale)
                flare = 0.32
            else:
                azimuth = rng.uniform(-2.05, -1.15)  # Right twin-tail
                strand_len = r_head * 2.8 * float(length_scale)
                flare = 0.32
        elif style_preset == "short_spiky":
            azimuth = (frac * 2.0 * math.pi) - math.pi
            strand_len = r_head * rng.uniform(0.85, 1.35) * float(length_scale)
            flare = 0.28
        elif style_preset == "long_flowing":
            azimuth = (frac * 2.0 * math.pi) - math.pi
            is_front = abs(azimuth) < 0.75
            strand_len = (r_head * (1.15 if is_front else 3.1)) * float(length_scale)
            flare = 0.16
        else:
            # Default: anime_layered
            azimuth = (frac * 2.0 * math.pi) - math.pi
            is_front = abs(azimuth) < 0.72
            is_side = 0.72 <= abs(azimuth) < 1.55
            if is_front:
                strand_len = r_head * rng.uniform(1.0, 1.3) * float(length_scale)
            elif is_side:
                strand_len = r_head * rng.uniform(1.8, 2.3) * float(length_scale)
            else:
                strand_len = r_head * rng.uniform(2.1, 2.7) * float(length_scale)
            flare = 0.18

        # Crown root position on upper hemisphere
        crown_polar = rng.uniform(0.18, 0.48)
        dir_x = math.sin(azimuth)
        dir_z = math.cos(azimuth)  # +Z = front face, -Z = back of head
        root_x = hx + r_head * math.sin(crown_polar) * dir_x
        root_y = hy + r_head * math.cos(crown_polar)
        root_z = hz + r_head * math.sin(crown_polar) * dir_z

        # Tangent vector along skull perimeter for ribbon width
        tangent = np.array([dir_z, 0.0, -dir_x], dtype=np.float32)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        radial_out = np.array([dir_x, 0.0, dir_z], dtype=np.float32)

        # Build centerline spine points & spring-bone nodes along the strand
        spine_pts = np.zeros((n_segs + 1, 3), dtype=np.float32)
        wave_phase = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(n_segs + 1):
            t = k / float(n_segs)
            # Curve outward past skull radius then drape downward with gentle wave
            arch = math.sin(t * math.pi) * r_head * (0.38 + flare)
            drop = (t ** 1.25) * strand_len
            sway = math.sin(t * math.pi * 1.5 + wave_phase) * 0.015 * t
            px = root_x + dir_x * (r_head * (1.0 - math.sin(crown_polar)) * min(1.0, t * 2.2) + arch) + tangent[0] * sway
            py = root_y - drop + (math.sin(t * math.pi * 0.5) * 0.02 if style_preset == "short_spiky" else 0.0)
            pz = root_z + dir_z * (r_head * (1.0 - math.sin(crown_polar)) * min(1.0, t * 2.2) + arch) + tangent[2] * sway
            spine_pts[k] = [px, py, pz]

        # Build V-shaped / diamond cross-section hair card ribbon (3 vertices per row: Left, Center-Ridge, Right)
        strand_verts = np.zeros(((n_segs + 1) * 3, 3), dtype=np.float32)
        strand_cols = np.zeros(((n_segs + 1) * 3, 3), dtype=np.float32)
        strand_uvs = np.zeros(((n_segs + 1) * 3, 2), dtype=np.float32)

        tint = rng.uniform(-0.04, 0.04, size=(3,)).astype(np.float32)
        for k in range(n_segs + 1):
            t = k / float(n_segs)
            # Taper width at root and tip
            w_profile = math.sin(min(1.0, max(0.08, t)) * math.pi * 0.88) * float(strand_width)
            if t < 0.2:
                w_profile = float(strand_width) * (0.55 + 2.25 * t)
            t_profile = float(strand_thickness) * (1.0 - 0.65 * t)

            center_pt = spine_pts[k]
            left_pt = center_pt - tangent * w_profile
            ridge_pt = center_pt + radial_out * t_profile
            right_pt = center_pt + tangent * w_profile

            row_base = k * 3
            strand_verts[row_base + 0] = left_pt
            strand_verts[row_base + 1] = ridge_pt
            strand_verts[row_base + 2] = right_pt

            # Specular anime hair halo around t in [0.25, 0.45]
            halo = math.exp(-((t - 0.35) / 0.12) ** 2)
            root_tip_shadow = 1.0 - 0.35 * (t ** 2)
            c_mid = np.clip((base_color * root_tip_shadow + highlight_color * 0.35 * halo) + tint, 0.0, 1.0)
            c_edge = np.clip((shadow_color * 0.5 + base_color * 0.5) * root_tip_shadow + tint, 0.0, 1.0)

            strand_cols[row_base + 0] = c_edge
            strand_cols[row_base + 1] = c_mid
            strand_cols[row_base + 2] = c_edge

            strand_uvs[row_base + 0] = [0.0, 1.0 - t]
            strand_uvs[row_base + 1] = [0.5, 1.0 - t]
            strand_uvs[row_base + 2] = [1.0, 1.0 - t]

        strand_faces: list[list[int]] = []
        for k in range(n_segs):
            r0 = vert_offset + k * 3
            r1 = vert_offset + (k + 1) * 3
            # Left ribbon strip
            strand_faces.append([r0 + 0, r1 + 0, r1 + 1])
            strand_faces.append([r0 + 0, r1 + 1, r0 + 1])
            # Right ribbon strip
            strand_faces.append([r0 + 1, r1 + 1, r1 + 2])
            strand_faces.append([r0 + 1, r1 + 2, r0 + 2])

        all_verts.append(strand_verts)
        all_faces.append(np.asarray(strand_faces, dtype=np.int32))
        all_colors.append(strand_cols)
        all_uvs.append(strand_uvs)

        # Create a 4-joint Spring-Bone physics chain per strand (root fixed to head, 3 active physics joints)
        chain_indices = np.linspace(0, n_segs, 4).round().astype(int)
        chain_joints = [spine_pts[idx].tolist() for idx in chain_indices]
        spring_chains.append(
            {
                "name": f"hair_strand_{s_idx:03d}",
                "parent_bone": "head",
                "joints": chain_joints,
                "stiffness": float(stiffness),
                "gravity_power": float(gravity_power),
                "drag_force": float(drag_force),
                "hit_radius": float(strand_width) * 0.5,
                "vertex_start": int(vert_offset),
                "vertex_count": int(len(strand_verts)),
            }
        )
        vert_offset += len(strand_verts)

    vertices = np.vstack(all_verts).astype(np.float32)
    faces = np.vstack(all_faces).astype(np.int32)
    vertex_colors = np.vstack(all_colors).astype(np.float32)
    uvs = np.vstack(all_uvs).astype(np.float32)
    normals = compute_vertex_normals(vertices, faces)

    return {
        "vertices": vertices,
        "faces": faces,
        "vertex_colors": vertex_colors,
        "normals": normals,
        "uvs": uvs,
        "spring_chains": spring_chains,
        "num_strands": n_strands,
        "style_preset": style_preset,
        "head_center": list(head_center),
        "head_radius": r_head,
    }


# ---------------------------------------------------------------------------
# 4. Humanoid Skeleton, Auto-Rigging & Skinning Engine (VRM 1.0 / 0.x Compatible)
# ---------------------------------------------------------------------------

VRM_HUMANOID_BONE_SPECS: list[tuple[str, str | None, tuple[float, float, float]]] = [
    # (bone_name, parent_name, normalized_position_in_canonical_t_pose [x, y, z] where y in [-0.8, +0.8])
    ("hips", None, (0.0, 0.0, 0.0)),
    ("spine", "hips", (0.0, 0.14, 0.0)),
    ("chest", "spine", (0.0, 0.28, 0.0)),
    ("upperChest", "chest", (0.0, 0.38, 0.0)),
    ("neck", "upperChest", (0.0, 0.46, 0.0)),
    ("head", "neck", (0.0, 0.56, 0.0)),
    ("leftEye", "head", (0.045, 0.61, 0.09)),
    ("rightEye", "head", (-0.045, 0.61, 0.09)),
    ("jaw", "head", (0.0, 0.52, 0.08)),
    ("leftShoulder", "upperChest", (0.08, 0.42, 0.0)),
    ("leftUpperArm", "leftShoulder", (0.16, 0.41, 0.0)),
    ("leftLowerArm", "leftUpperArm", (0.34, 0.40, 0.0)),
    ("leftHand", "leftLowerArm", (0.50, 0.39, 0.0)),
    ("rightShoulder", "upperChest", (-0.08, 0.42, 0.0)),
    ("rightUpperArm", "rightShoulder", (-0.16, 0.41, 0.0)),
    ("rightLowerArm", "rightUpperArm", (-0.34, 0.40, 0.0)),
    ("rightHand", "rightLowerArm", (-0.50, 0.39, 0.0)),
    ("leftUpperLeg", "hips", (0.09, -0.04, 0.0)),
    ("leftLowerLeg", "leftUpperLeg", (0.09, -0.38, 0.0)),
    ("leftFoot", "leftLowerLeg", (0.09, -0.70, 0.02)),
    ("leftToes", "leftFoot", (0.09, -0.75, 0.09)),
    ("rightUpperLeg", "hips", (-0.09, -0.04, 0.0)),
    ("rightLowerLeg", "rightUpperLeg", (-0.09, -0.38, 0.0)),
    ("rightFoot", "rightLowerLeg", (-0.09, -0.70, 0.02)),
    ("rightToes", "rightFoot", (-0.09, -0.75, 0.09)),
]


def auto_rig_humanoid_mesh(
    mesh_data: dict[str, Any],
    *,
    rig_backend: str = "auto",
    temperature: float = 0.085,
    normalize_scale: bool = True,
) -> dict[str, Any]:
    """
    Fits a 25-bone VRM Humanoid skeleton to a 3D character mesh (`ZIGGART_MESH`) and
    computes smooth 4-bone skinning weights (`joints_0`: (N, 4) uint16, `weights_0`: (N, 4) float32)
    plus glTF 2.0 `inverse_bind_matrices` ((B, 4, 4) float32).
    """
    verts = np.asarray(mesh_data["vertices"], dtype=np.float32).copy()
    faces = np.asarray(mesh_data["faces"], dtype=np.int32).copy()
    v_cols = np.asarray(
        mesh_data.get("vertex_colors", np.full((len(verts), 3), 0.78, dtype=np.float32)),
        dtype=np.float32,
    ).copy()

    if len(verts) == 0:
        raise ValueError("Cannot auto-rig an empty mesh (0 vertices).")

    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    center = 0.5 * (bmin + bmax)
    span = np.maximum(bmax - bmin, 1e-5)
    height = float(span[1])

    if normalize_scale and height > 1e-5:
        # Normalize character to standard 1.60m height centered around hips (y=0.0, feet at y=-0.76, head top at +0.76)
        scale_factor = 1.52 / height
        verts = (verts - center) * scale_factor
        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)
        span = np.maximum(bmax - bmin, 1e-5)

    # Adapt bone positions to actual mesh bounding box proportions (handles wide/narrow arms or chibi proportions)
    x_scale = float(span[0]) / 1.05
    y_scale = float(span[1]) / 1.52
    z_center = float(np.median(verts[:, 2]))

    bone_names: list[str] = []
    bone_parents: list[int] = []
    bone_positions = np.zeros((len(VRM_HUMANOID_BONE_SPECS), 3), dtype=np.float32)
    name_to_idx: dict[str, int] = {}

    for idx, (b_name, p_name, (nx, ny, nz)) in enumerate(VRM_HUMANOID_BONE_SPECS):
        name_to_idx[b_name] = idx
        bone_names.append(b_name)
        bone_parents.append(name_to_idx[p_name] if p_name is not None else -1)

        bx = nx * max(x_scale, 0.45)
        by = ny * y_scale
        bz = z_center + nz * y_scale
        bone_positions[idx] = [bx, by, bz]

    # Snap arm bones to actual lateral mesh density if character is in A-pose instead of strict T-pose
    upper_torso_mask = (verts[:, 1] > 0.15 * y_scale) & (verts[:, 1] < 0.48 * y_scale)
    left_arm_verts = verts[upper_torso_mask & (verts[:, 0] > 0.18 * max(x_scale, 0.4))]
    right_arm_verts = verts[upper_torso_mask & (verts[:, 0] < -0.18 * max(x_scale, 0.4))]
    if len(left_arm_verts) > 12:
        med_y_left = float(np.median(left_arm_verts[:, 1]))
        bone_positions[name_to_idx["leftLowerArm"], 1] = 0.5 * bone_positions[name_to_idx["leftUpperArm"], 1] + 0.5 * med_y_left
        bone_positions[name_to_idx["leftHand"], 1] = med_y_left
    if len(right_arm_verts) > 12:
        med_y_right = float(np.median(right_arm_verts[:, 1]))
        bone_positions[name_to_idx["rightLowerArm"], 1] = 0.5 * bone_positions[name_to_idx["rightUpperArm"], 1] + 0.5 * med_y_right
        bone_positions[name_to_idx["rightHand"], 1] = med_y_right

    # Compute bone segment distance for each vertex (capsule bone heat skinning)
    num_verts = len(verts)
    num_bones = len(bone_names)
    dist_matrix = np.zeros((num_verts, num_bones), dtype=np.float32)

    # Map each bone to its primary child segment endpoint if one exists
    children_map: dict[int, list[int]] = {i: [] for i in range(num_bones)}
    for idx, p_idx in enumerate(bone_parents):
        if p_idx >= 0:
            children_map[p_idx].append(idx)

    for b_idx in range(num_bones):
        p0 = bone_positions[b_idx]
        c_list = children_map[b_idx]
        if len(c_list) == 1:
            p1 = bone_positions[c_list[0]]
            seg = p1 - p0
            seg_len_sq = float(np.dot(seg, seg))
            if seg_len_sq > 1e-6:
                t = np.clip(np.sum((verts - p0[None, :]) * seg[None, :], axis=-1) / seg_len_sq, 0.0, 1.0)
                proj = p0[None, :] + t[:, None] * seg[None, :]
                d = np.linalg.norm(verts - proj, axis=-1)
            else:
                d = np.linalg.norm(verts - p0[None, :], axis=-1)
        else:
            d = np.linalg.norm(verts - p0[None, :], axis=-1)

        # De-prioritize micro facial sub-bones (leftEye, rightEye, jaw) for global body skinning
        if bone_names[b_idx] in ("leftEye", "rightEye", "jaw"):
            d = d + 0.35
        dist_matrix[:, b_idx] = d

    # Select top-4 closest bones per vertex and compute smooth softmax weights
    temp = max(0.015, float(temperature))
    top4_indices = np.argpartition(dist_matrix, kth=min(3, num_bones - 1), axis=-1)[:, :4]
    row_idx = np.arange(num_verts)[:, None]
    top4_dists = dist_matrix[row_idx, top4_indices]

    # Sort the 4 bones from highest weight (smallest distance) to lowest
    sort_order = np.argsort(top4_dists, axis=-1)
    joints_0 = top4_indices[row_idx, sort_order].astype(np.uint16)
    sorted_dists = top4_dists[row_idx, sort_order]

    logits = -sorted_dists / temp
    logits -= logits.max(axis=-1, keepdims=True)
    exp_w = np.exp(logits).astype(np.float32)
    weights_0 = exp_w / np.maximum(exp_w.sum(axis=-1, keepdims=True), 1e-8)

    # Compute local bone offsets & column-major glTF 2.0 Inverse Bind Matrices
    local_translations = np.zeros_like(bone_positions)
    inverse_bind_matrices = np.zeros((num_bones, 4, 4), dtype=np.float32)
    for b_idx in range(num_bones):
        p_idx = bone_parents[b_idx]
        if p_idx >= 0:
            local_translations[b_idx] = bone_positions[b_idx] - bone_positions[p_idx]
        else:
            local_translations[b_idx] = bone_positions[b_idx]

        ibm = np.eye(4, dtype=np.float32)
        ibm[0, 3] = -bone_positions[b_idx, 0]
        ibm[1, 3] = -bone_positions[b_idx, 1]
        ibm[2, 3] = -bone_positions[b_idx, 2]
        inverse_bind_matrices[b_idx] = ibm

    normals = compute_vertex_normals(verts, faces)

    return {
        "vertices": verts,
        "faces": faces,
        "vertex_colors": v_cols,
        "normals": normals,
        "bone_names": bone_names,
        "bone_parents": bone_parents,
        "bone_positions": bone_positions,
        "local_translations": local_translations,
        "joints_0": joints_0,
        "weights_0": weights_0,
        "inverse_bind_matrices": inverse_bind_matrices,
        "morph_targets": {},
        "rig_backend_used": rig_backend,
    }


# ---------------------------------------------------------------------------
# 5. Facial BlendShapes (Morph Targets) & Fun Cartoon Deformers
# ---------------------------------------------------------------------------

STANDARD_VRM_BLENDSHAPES: tuple[str, ...] = (
    "aa",
    "ih",
    "ou",
    "ee",
    "oh",
    "blink",
    "blinkLeft",
    "blinkRight",
    "happy",
    "angry",
    "sad",
    "relaxed",
    "surprised",
    "lookLeft",
    "lookRight",
    "lookUp",
    "lookDown",
    "anime_shock_eyes",
    "puff_cheeks",
    "chibi_big_head",
)


def build_avatar_facial_blendshapes(
    rigged_mesh: dict[str, Any],
    *,
    expression_strength: float = 1.0,
    include_fun_morphs: bool = True,
) -> dict[str, Any]:
    """
    Generates position delta arrays `(N, 3)` for standard VRM 0.x / 1.0 facial expressions,
    lip-sync visemes (`aa`, `ih`, `ou`, `ee`, `oh`), eye blinks, and fun cartoon morphs.
    """
    verts = np.asarray(rigged_mesh["vertices"], dtype=np.float32)
    bone_names: list[str] = list(rigged_mesh["bone_names"])
    bone_positions = np.asarray(rigged_mesh["bone_positions"], dtype=np.float32)

    head_idx = bone_names.index("head") if "head" in bone_names else 0
    neck_idx = bone_names.index("neck") if "neck" in bone_names else 0
    head_pos = bone_positions[head_idx]
    neck_pos = bone_positions[neck_idx]

    # Head zone vertices
    head_verts_mask = verts[:, 1] >= neck_pos[1]
    if np.any(head_verts_mask):
        h_verts = verts[head_verts_mask]
        h_min = h_verts.min(axis=0)
        h_max = h_verts.max(axis=0)
        head_r = float(max(h_max[0] - h_min[0], h_max[1] - h_min[1], 0.12)) * 0.5
        front_z = float(np.percentile(h_verts[:, 2], 70))
    else:
        head_r = 0.16
        front_z = float(head_pos[2] + 0.08)

    amp = float(expression_strength) * head_r * 0.22
    n_verts = len(verts)

    # Spatial Gaussian weights for Mouth, Left Eye, Right Eye, Cheeks, and Full Head
    mouth_center = np.array([head_pos[0], head_pos[1] - 0.18 * head_r, front_z], dtype=np.float32)
    l_eye_center = np.array([head_pos[0] + 0.28 * head_r, head_pos[1] + 0.18 * head_r, front_z], dtype=np.float32)
    r_eye_center = np.array([head_pos[0] - 0.28 * head_r, head_pos[1] + 0.18 * head_r, front_z], dtype=np.float32)
    l_cheek_center = np.array([head_pos[0] + 0.36 * head_r, head_pos[1] - 0.05 * head_r, front_z], dtype=np.float32)
    r_cheek_center = np.array([head_pos[0] - 0.36 * head_r, head_pos[1] - 0.05 * head_r, front_z], dtype=np.float32)

    def _gauss_weight(center_pt: np.ndarray, radius: float) -> np.ndarray:
        d_sq = np.sum((verts - center_pt[None, :]) ** 2, axis=-1)
        return np.exp(-d_sq / max(2.0 * (radius ** 2), 1e-6)).astype(np.float32)

    w_mouth = _gauss_weight(mouth_center, 0.28 * head_r)
    w_l_eye = _gauss_weight(l_eye_center, 0.22 * head_r)
    w_r_eye = _gauss_weight(r_eye_center, 0.22 * head_r)
    w_eyes = np.clip(w_l_eye + w_r_eye, 0.0, 1.0)
    w_cheeks = np.clip(_gauss_weight(l_cheek_center, 0.26 * head_r) + _gauss_weight(r_cheek_center, 0.26 * head_r), 0.0, 1.0)
    w_head = np.clip((verts[:, 1] - neck_pos[1]) / max(head_r * 0.6, 1e-4), 0.0, 1.0).astype(np.float32)

    lower_lip_sign = np.where(verts[:, 1] < mouth_center[1], -1.0, 0.45).astype(np.float32)
    upper_lid_sign = np.where(verts[:, 1] >= l_eye_center[1], -1.0, 0.35).astype(np.float32)
    x_sign = np.sign(verts[:, 0] - head_pos[0]).astype(np.float32)

    morphs: dict[str, np.ndarray] = {}

    # 1. Lip-sync Visemes (aa, ih, ou, ee, oh)
    morphs["aa"] = np.column_stack(
        [
            np.zeros(n_verts, dtype=np.float32),
            w_mouth * lower_lip_sign * amp * 1.15,
            w_mouth * amp * 0.35,
        ]
    ).astype(np.float32)
    morphs["ih"] = np.column_stack(
        [
            w_mouth * x_sign * amp * 0.65,
            w_mouth * lower_lip_sign * amp * 0.40,
            w_mouth * amp * 0.15,
        ]
    ).astype(np.float32)
    morphs["ou"] = np.column_stack(
        [
            -w_mouth * x_sign * amp * 0.45,
            w_mouth * lower_lip_sign * amp * 0.55,
            w_mouth * amp * 0.70,
        ]
    ).astype(np.float32)
    morphs["ee"] = np.column_stack(
        [
            w_mouth * x_sign * amp * 0.80,
            w_mouth * lower_lip_sign * amp * 0.50,
            w_mouth * amp * 0.20,
        ]
    ).astype(np.float32)
    morphs["oh"] = np.column_stack(
        [
            -w_mouth * x_sign * amp * 0.25,
            w_mouth * lower_lip_sign * amp * 0.95,
            w_mouth * amp * 0.55,
        ]
    ).astype(np.float32)

    # 2. Eye Blinks & Gaze
    morphs["blinkLeft"] = np.column_stack(
        [
            np.zeros(n_verts, dtype=np.float32),
            w_l_eye * upper_lid_sign * amp * 0.85,
            -w_l_eye * amp * 0.15,
        ]
    ).astype(np.float32)
    morphs["blinkRight"] = np.column_stack(
        [
            np.zeros(n_verts, dtype=np.float32),
            w_r_eye * upper_lid_sign * amp * 0.85,
            -w_r_eye * amp * 0.15,
        ]
    ).astype(np.float32)
    morphs["blink"] = (morphs["blinkLeft"] + morphs["blinkRight"]).astype(np.float32)

    morphs["lookLeft"] = np.column_stack([w_eyes * amp * 0.4, np.zeros(n_verts, dtype=np.float32), np.zeros(n_verts, dtype=np.float32)]).astype(np.float32)
    morphs["lookRight"] = -morphs["lookLeft"]
    morphs["lookUp"] = np.column_stack([np.zeros(n_verts, dtype=np.float32), w_eyes * amp * 0.4, np.zeros(n_verts, dtype=np.float32)]).astype(np.float32)
    morphs["lookDown"] = -morphs["lookUp"]

    # 3. Core Emotions
    morphs["happy"] = np.column_stack(
        [
            w_mouth * x_sign * amp * 0.70 + w_cheeks * x_sign * amp * 0.25,
            (w_mouth * np.abs(x_sign) + w_cheeks * 0.6) * amp * 0.65,
            w_cheeks * amp * 0.30,
        ]
    ).astype(np.float32)
    morphs["angry"] = np.column_stack(
        [
            -w_eyes * x_sign * amp * 0.35,
            -w_eyes * (1.0 - np.abs(x_sign)) * amp * 0.55 - w_mouth * amp * 0.25,
            w_eyes * amp * 0.20,
        ]
    ).astype(np.float32)
    morphs["sad"] = np.column_stack(
        [
            np.zeros(n_verts, dtype=np.float32),
            -w_mouth * np.abs(x_sign) * amp * 0.60 - w_eyes * np.abs(x_sign) * amp * 0.35,
            np.zeros(n_verts, dtype=np.float32),
        ]
    ).astype(np.float32)
    morphs["relaxed"] = (0.35 * morphs["happy"] + 0.25 * morphs["blink"]).astype(np.float32)
    morphs["surprised"] = (0.85 * morphs["oh"] + np.column_stack([np.zeros(n_verts, dtype=np.float32), w_eyes * amp * 0.55, w_eyes * amp * 0.35])).astype(np.float32)

    # 4. Fun / Cartoon Modes
    if include_fun_morphs:
        morphs["anime_shock_eyes"] = np.column_stack(
            [
                w_eyes * x_sign * amp * 1.1,
                w_eyes * -upper_lid_sign * amp * 1.25,
                w_eyes * amp * 0.90,
            ]
        ).astype(np.float32)
        morphs["puff_cheeks"] = np.column_stack(
            [
                w_cheeks * x_sign * amp * 1.35,
                w_cheeks * amp * 0.25,
                w_cheeks * amp * 0.95,
            ]
        ).astype(np.float32)
        head_rel = verts - head_pos[None, :]
        morphs["chibi_big_head"] = (head_rel * (w_head[:, None] * 0.48)).astype(np.float32)

    out_mesh = dict(rigged_mesh)
    out_mesh["morph_targets"] = morphs
    return out_mesh


# ---------------------------------------------------------------------------
# 6. Procedural 3D Fun Props & Accessories (Cat Ears, Halo, Visor, Wings)
# ---------------------------------------------------------------------------


def generate_avatar_accessory_mesh(
    accessory_type: str,
    head_pos: tuple[float, float, float] = (0.0, 0.56, 0.0),
    head_radius: float = 0.16,
) -> dict[str, Any] | None:
    """
    Generates a lightweight 3D accessory mesh + Spring-Bone chain anchored to `head`.
    """
    kind = (accessory_type or "none").strip().lower()
    if kind in ("none", "", "false"):
        return None

    hx, hy, hz = float(head_pos[0]), float(head_pos[1]), float(head_pos[2])
    r = float(head_radius)

    verts_list: list[list[float]] = []
    faces_list: list[list[int]] = []
    cols_list: list[list[float]] = []
    spring_chains: list[dict[str, Any]] = []

    if kind == "cat_ears":
        for side, sx in (("left", 1.0), ("right", -1.0)):
            base_idx = len(verts_list)
            bx = hx + sx * r * 0.58
            by = hy + r * 0.78
            bz = hz + r * 0.08
            tip = [bx + sx * r * 0.26, by + r * 0.56, bz + r * 0.04]
            p0 = [bx - sx * r * 0.20, by, bz + r * 0.14]
            p1 = [bx + sx * r * 0.22, by - r * 0.08, bz + r * 0.14]
            p2 = [bx + sx * r * 0.04, by, bz - r * 0.16]
            verts_list.extend([p0, p1, p2, tip])
            cols_list.extend(
                [
                    [0.95, 0.58, 0.72],
                    [0.95, 0.58, 0.72],
                    [0.25, 0.22, 0.28],
                    [0.98, 0.78, 0.86],
                ]
            )
            faces_list.extend(
                [
                    [base_idx + 0, base_idx + 1, base_idx + 3],
                    [base_idx + 1, base_idx + 2, base_idx + 3],
                    [base_idx + 2, base_idx + 0, base_idx + 3],
                    [base_idx + 0, base_idx + 2, base_idx + 1],
                ]
            )
            spring_chains.append(
                {
                    "name": f"prop_cat_ear_{side}",
                    "parent_bone": "head",
                    "joints": [[bx, by, bz], tip],
                    "stiffness": 0.65,
                    "gravity_power": 0.02,
                    "drag_force": 0.25,
                    "hit_radius": 0.02,
                    "vertex_start": base_idx,
                    "vertex_count": 4,
                }
            )
    elif kind == "angel_halo":
        n_ring = 20
        ring_y = hy + r * 1.38
        ring_r = r * 0.68
        tube_r = r * 0.07
        for i in range(n_ring):
            ang = (i / float(n_ring)) * 2.0 * math.pi
            cx = hx + math.cos(ang) * ring_r
            cz = hz + math.sin(ang) * ring_r
            verts_list.append([cx, ring_y + tube_r, cz])
            verts_list.append([cx, ring_y - tube_r, cz])
            verts_list.append([hx + math.cos(ang) * (ring_r + tube_r), ring_y, hz + math.sin(ang) * (ring_r + tube_r)])
            cols_list.extend([[1.0, 0.92, 0.38], [0.96, 0.80, 0.22], [1.0, 0.96, 0.55]])
        for i in range(n_ring):
            b0 = i * 3
            b1 = ((i + 1) % n_ring) * 3
            faces_list.extend(
                [
                    [b0 + 0, b1 + 0, b1 + 2],
                    [b0 + 0, b1 + 2, b0 + 2],
                    [b0 + 2, b1 + 2, b1 + 1],
                    [b0 + 2, b1 + 1, b0 + 1],
                    [b0 + 1, b1 + 1, b1 + 0],
                    [b0 + 1, b1 + 0, b0 + 0],
                ]
            )
        spring_chains.append(
            {
                "name": "prop_angel_halo",
                "parent_bone": "head",
                "joints": [[hx, hy + r, hz], [hx, ring_y, hz]],
                "stiffness": 0.35,
                "gravity_power": 0.0,
                "drag_force": 0.20,
                "hit_radius": 0.04,
                "vertex_start": 0,
                "vertex_count": len(verts_list),
            }
        )
    elif kind == "cyber_visor":
        vy = hy + r * 0.16
        vz = hz + r * 0.72
        vw = r * 0.78
        vh = r * 0.20
        verts_list.extend(
            [
                [hx - vw, vy - vh, vz - r * 0.22],
                [hx - vw * 0.45, vy - vh, vz + r * 0.12],
                [hx + vw * 0.45, vy - vh, vz + r * 0.12],
                [hx + vw, vy - vh, vz - r * 0.22],
                [hx - vw, vy + vh, vz - r * 0.22],
                [hx - vw * 0.45, vy + vh, vz + r * 0.12],
                [hx + vw * 0.45, vy + vh, vz + r * 0.12],
                [hx + vw, vy + vh, vz - r * 0.22],
            ]
        )
        cols_list.extend([[0.05, 0.92, 1.0]] * 8)
        for q in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6)):
            faces_list.append([q[0], q[1], q[2]])
            faces_list.append([q[0], q[2], q[3]])
            faces_list.append([q[0], q[2], q[1]])
            faces_list.append([q[0], q[3], q[2]])
    elif kind == "demon_horns":
        for side, sx in (("left", 1.0), ("right", -1.0)):
            base_idx = len(verts_list)
            bx = hx + sx * r * 0.48
            by = hy + r * 0.75
            bz = hz + r * 0.18
            tip = [bx + sx * r * 0.38, by + r * 0.62, bz - r * 0.12]
            p0 = [bx - sx * r * 0.12, by, bz + r * 0.12]
            p1 = [bx + sx * r * 0.12, by, bz + r * 0.12]
            p2 = [bx, by, bz - r * 0.12]
            verts_list.extend([p0, p1, p2, tip])
            cols_list.extend([[0.22, 0.08, 0.12], [0.22, 0.08, 0.12], [0.18, 0.05, 0.08], [0.85, 0.15, 0.22]])
            faces_list.extend(
                [
                    [base_idx + 0, base_idx + 1, base_idx + 3],
                    [base_idx + 1, base_idx + 2, base_idx + 3],
                    [base_idx + 2, base_idx + 0, base_idx + 3],
                ]
            )

    if not verts_list:
        return None

    v_arr = np.asarray(verts_list, dtype=np.float32)
    f_arr = np.asarray(faces_list, dtype=np.int32)
    c_arr = np.asarray(cols_list, dtype=np.float32)
    return {
        "vertices": v_arr,
        "faces": f_arr,
        "vertex_colors": c_arr,
        "normals": compute_vertex_normals(v_arr, f_arr),
        "spring_chains": spring_chains,
        "accessory_type": kind,
    }


# ---------------------------------------------------------------------------
# 7. Native Binary .VRM (VRM 1.0 + VRM 0.x + glTF 2.0 Skins & SpringBones) Exporter
# ---------------------------------------------------------------------------


def export_vrm_avatar_file(
    output_path: str | Path,
    avatar_data: dict[str, Any],
    *,
    vrm_version: str = "1.0",
    avatar_title: str = "Ziggart 3D Avatar",
    avatar_author: str = "Ziggart-Nodes",
) -> Path:
    """
    Exports a complete, animation-ready `.vrm` or rigged `.glb` binary glTF 2.0 file containing:
      - Mesh geometry (`POSITION`, `NORMAL`, `COLOR_0`, `JOINTS_0`, `WEIGHTS_0`)
      - Morph targets (Facial BlendShapes for lip-sync, eye blink, emotions, fun FX)
      - Humanoid skeleton hierarchy (`nodes`, `skins`, `inverseBindMatrices`)
      - Spring-Bone physics chains (`VRMC_springBone` + `VRM.secondaryAnimation`)
      - VRM Humanoid & Expression metadata (`VRMC_vrm` + `VRM`)
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    verts = np.asarray(avatar_data["vertices"], dtype=np.float32)
    faces = np.asarray(avatar_data["faces"], dtype=np.uint32)
    cols = np.clip(np.asarray(avatar_data["vertex_colors"], dtype=np.float32)[:, :3], 0.0, 1.0)
    normals = avatar_data.get("normals")
    if normals is None or len(normals) != len(verts):
        norms = compute_vertex_normals(verts, faces.astype(np.int32))
    else:
        norms = np.asarray(normals, dtype=np.float32)

    bone_names: list[str] = list(avatar_data["bone_names"])
    bone_parents: list[int] = list(avatar_data["bone_parents"])
    local_translations = np.asarray(avatar_data["local_translations"], dtype=np.float32)
    joints_0 = np.asarray(avatar_data["joints_0"], dtype=np.uint16)
    weights_0 = np.asarray(avatar_data["weights_0"], dtype=np.float32)
    ibms = np.asarray(avatar_data["inverse_bind_matrices"], dtype=np.float32)
    morph_targets: dict[str, np.ndarray] = dict(avatar_data.get("morph_targets") or {})
    spring_chains: list[dict[str, Any]] = list(avatar_data.get("spring_chains") or [])

    num_verts = len(verts)
    num_bones = len(bone_names)

    def _pad4(b: bytes) -> bytes:
        rem = len(b) % 4
        return b if rem == 0 else b + b"\x00" * (4 - rem)

    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    current_offset = 0

    def _add_accessor(
        raw_bytes: bytes,
        component_type: int,
        count: int,
        acc_type: str,
        target: int | None = 34962,
        v_min: list[float] | None = None,
        v_max: list[float] | None = None,
    ) -> int:
        nonlocal current_offset
        padded = _pad4(raw_bytes)
        bv: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": current_offset,
            "byteLength": len(raw_bytes),
        }
        if target is not None:
            bv["target"] = target
        bv_idx = len(buffer_views)
        buffer_views.append(bv)
        chunks.append(padded)
        current_offset += len(padded)

        acc: dict[str, Any] = {
            "bufferView": bv_idx,
            "componentType": component_type,
            "count": int(count),
            "type": acc_type,
        }
        if v_min is not None:
            acc["min"] = v_min
        if v_max is not None:
            acc["max"] = v_max
        acc_idx = len(accessors)
        accessors.append(acc)
        return acc_idx

    v_min = verts.min(axis=0).astype(float).tolist() if num_verts else [0.0, 0.0, 0.0]
    v_max = verts.max(axis=0).astype(float).tolist() if num_verts else [0.0, 0.0, 0.0]

    acc_pos = _add_accessor(verts.astype("<f4").tobytes(), 5126, num_verts, "VEC3", 34962, v_min, v_max)
    acc_norm = _add_accessor(norms.astype("<f4").tobytes(), 5126, num_verts, "VEC3", 34962)
    rgba_cols = np.column_stack([cols, np.ones(num_verts, dtype=np.float32)]).astype("<f4")
    acc_col = _add_accessor(rgba_cols.tobytes(), 5126, num_verts, "VEC4", 34962)
    acc_joints = _add_accessor(joints_0.astype("<u2").tobytes(), 5123, num_verts, "VEC4", 34962)
    acc_weights = _add_accessor(weights_0.astype("<f4").tobytes(), 5126, num_verts, "VEC4", 34962)
    acc_indices = _add_accessor(faces.reshape(-1).astype("<u4").tobytes(), 5125, int(faces.size), "SCALAR", 34963)

    # glTF 2.0 expects column-major 4x4 matrices for inverseBindMatrices
    ibm_col_major = np.transpose(ibms, (0, 2, 1)).astype("<f4")
    acc_ibm = _add_accessor(ibm_col_major.tobytes(), 5126, num_bones, "MAT4", None)

    # Add Morph Targets (BlendShapes)
    gltf_targets: list[dict[str, int]] = []
    target_names: list[str] = []
    expression_preset_map: dict[str, dict[str, Any]] = {}
    vrm0_blendshape_groups: list[dict[str, Any]] = []

    for m_idx, (m_name, delta_arr) in enumerate(morph_targets.items()):
        delta = np.asarray(delta_arr, dtype=np.float32)
        if delta.shape != (num_verts, 3):
            continue
        d_min = delta.min(axis=0).astype(float).tolist()
        d_max = delta.max(axis=0).astype(float).tolist()
        acc_m = _add_accessor(delta.astype("<f4").tobytes(), 5126, num_verts, "VEC3", 34962, d_min, d_max)
        gltf_targets.append({"POSITION": acc_m})
        target_names.append(m_name)
        expression_preset_map[m_name] = {
            "morphTargetBinds": [{"node": 0, "index": m_idx, "weight": 1.0}],
            "isBinary": False,
        }
        vrm0_blendshape_groups.append(
            {
                "name": m_name,
                "presetName": m_name,
                "binds": [{"mesh": 0, "index": m_idx, "weight": 100.0}],
                "materialValues": [],
                "isBinary": False,
            }
        )

    # Build glTF Node hierarchy:
    # Node 0 = Skinned Mesh node
    # Nodes 1..num_bones = Humanoid armature joints
    # Nodes (1+num_bones).. = Spring-Bone joint chains attached to `head`
    gltf_nodes: list[dict[str, Any]] = [
        {"name": "ZiggartAvatarMesh", "mesh": 0, "skin": 0},
    ]

    bone_children: dict[int, list[int]] = {i: [] for i in range(num_bones)}
    root_bone_nodes: list[int] = []
    for b_idx, p_idx in enumerate(bone_parents):
        if p_idx >= 0:
            bone_children[p_idx].append(b_idx + 1)
        else:
            root_bone_nodes.append(b_idx + 1)

    for b_idx, b_name in enumerate(bone_names):
        t_vec = local_translations[b_idx].astype(float).tolist()
        node_entry: dict[str, Any] = {
            "name": b_name,
            "translation": t_vec,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        }
        if bone_children[b_idx]:
            node_entry["children"] = list(bone_children[b_idx])
        gltf_nodes.append(node_entry)

    # Append Spring-Bone joint nodes for HairDeck strands & accessories
    head_bone_idx = bone_names.index("head") if "head" in bone_names else 0
    head_node_idx = head_bone_idx + 1
    head_world_pos = np.asarray(avatar_data["bone_positions"][head_bone_idx], dtype=np.float32)

    vrm1_springs: list[dict[str, Any]] = []
    vrm0_bone_groups: list[dict[str, Any]] = []

    for chain in spring_chains:
        c_joints = np.asarray(chain.get("joints", []), dtype=np.float32)
        if len(c_joints) < 2:
            continue
        chain_node_indices: list[int] = []
        prev_world = head_world_pos
        for j_i, pt in enumerate(c_joints):
            n_idx = len(gltf_nodes)
            rel_t = (pt - prev_world).astype(float).tolist()
            prev_world = pt
            j_node: dict[str, Any] = {
                "name": f"{chain.get('name', 'spring')}_j{j_i}",
                "translation": rel_t,
            }
            if j_i == 0:
                gltf_nodes[head_node_idx].setdefault("children", []).append(n_idx)
            else:
                gltf_nodes[chain_node_indices[-1]].setdefault("children", []).append(n_idx)
            gltf_nodes.append(j_node)
            chain_node_indices.append(n_idx)

        stiff = float(chain.get("stiffness", 0.45))
        grav = float(chain.get("gravity_power", 0.08))
        drag = float(chain.get("drag_force", 0.35))
        rad = float(chain.get("hit_radius", 0.02))

        vrm1_springs.append(
            {
                "name": chain.get("name", "hair_spring"),
                "joints": [
                    {
                        "node": jn,
                        "hitRadius": rad,
                        "stiffness": stiff,
                        "gravityPower": grav,
                        "gravityDir": [0.0, -1.0, 0.0],
                        "dragForce": drag,
                    }
                    for jn in chain_node_indices
                ],
            }
        )
        vrm0_bone_groups.append(
            {
                "comment": chain.get("name", "hair_spring"),
                "stiffiness": stiff,
                "gravityPower": grav,
                "gravityDir": {"x": 0.0, "y": -1.0, "z": 0.0},
                "dragForce": drag,
                "hitRadius": rad,
                "bones": [chain_node_indices[0]],
                "colliderGroups": [],
            }
        )

    human_bones_vrm1 = {b_name: {"node": b_idx + 1} for b_idx, b_name in enumerate(bone_names)}
    human_bones_vrm0 = [{"bone": b_name, "node": b_idx + 1, "useDefaultValues": True} for b_idx, b_name in enumerate(bone_names)]

    primitive_dict: dict[str, Any] = {
        "attributes": {
            "POSITION": acc_pos,
            "NORMAL": acc_norm,
            "COLOR_0": acc_col,
            "JOINTS_0": acc_joints,
            "WEIGHTS_0": acc_weights,
        },
        "indices": acc_indices,
        "material": 0,
        "mode": 4,
    }
    if gltf_targets:
        primitive_dict["targets"] = gltf_targets

    mesh_entry: dict[str, Any] = {
        "name": "ZiggartAvatarMesh",
        "primitives": [primitive_dict],
    }
    if target_names:
        mesh_entry["weights"] = [0.0] * len(target_names)
        mesh_entry["extras"] = {"targetNames": target_names}

    gltf_dict: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "Ziggart-Nodes 3D VRM Avatar Suite"},
        "scene": 0,
        "scenes": [{"nodes": [0] + root_bone_nodes}],
        "nodes": gltf_nodes,
        "skins": [
            {
                "name": "ZiggartHumanoidArmature",
                "joints": [b_idx + 1 for b_idx in range(num_bones)],
                "skeleton": 1,
                "inverseBindMatrices": acc_ibm,
            }
        ],
        "materials": [
            {
                "name": "ZiggartAvatarMToonMaterial",
                "pbrMetallicRoughness": {"metallicFactor": 0.02, "roughnessFactor": 0.78},
                "doubleSided": True,
            }
        ],
        "meshes": [mesh_entry],
        "buffers": [{"byteLength": current_offset}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extensionsUsed": ["VRMC_vrm", "VRMC_springBone", "VRM"],
        "extensions": {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "meta": {
                    "name": avatar_title,
                    "version": "1.0",
                    "authors": [avatar_author],
                    "avatarPermission": "everyone",
                    "allowExcessivelyViolentUsage": False,
                    "allowExcessivelySexualUsage": False,
                    "commercialUsage": "personalNonProfit",
                    "allowPoliticalOrReligiousUsage": False,
                    "allowAntisocialOrHateUsage": False,
                    "creditNotation": "optional",
                    "allowRedistribution": True,
                    "modification": "allowModificationRedistribution",
                },
                "humanoid": {"humanBones": human_bones_vrm1},
                "expressions": {"preset": expression_preset_map},
            },
            "VRMC_springBone": {
                "specVersion": "1.0",
                "springs": vrm1_springs,
            },
            "VRM": {
                "exporterVersion": "Ziggart-VRM-Exporter-1.0",
                "specVersion": "0.0",
                "meta": {
                    "title": avatar_title,
                    "version": "1.0",
                    "author": avatar_author,
                    "allowedUserName": "Everyone",
                    "violentUssageName": "Disallow",
                    "sexualUssageName": "Disallow",
                    "commercialUssageName": "Allow",
                    "licenseName": "Redistribution_Modifications",
                },
                "humanoid": {"humanBones": human_bones_vrm0},
                "blendShapeMaster": {"blendShapeGroups": vrm0_blendshape_groups},
                "secondaryAnimation": {"boneGroups": vrm0_bone_groups, "colliderGroups": []},
            },
        },
    }

    bin_blob = b"".join(chunks)
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    rem_json = len(json_bytes) % 4
    if rem_json != 0:
        json_bytes += b" " * (4 - rem_json)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
    bin_chunk = struct.pack("<I4s", len(bin_blob), b"BIN\x00") + bin_blob

    with path.open("wb") as f:
        f.write(header)
        f.write(json_chunk)
        f.write(bin_chunk)

    return path


def load_vrm_or_glb_avatar(path: str | Path) -> dict[str, Any] | None:
    """
    Loads a binary `.vrm` or `.glb` (glTF 2.0) avatar file from disk into a `ZIGGART_AVATAR` dict.
    Supports both Ziggart-exported VRM/GLB files and general glTF 2.0 binary meshes (auto-rigging
    and building facial BlendShapes if the loaded mesh does not already carry skinning attributes).
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw = p.read_bytes()
        if len(raw) < 28 or raw[:4] != b"glTF":
            return None
        json_len, json_type = struct.unpack_from("<I4s", raw, 12)
        if json_type != b"JSON":
            return None
        json_end = 20 + json_len
        gltf = json.loads(raw[20:json_end].decode("utf-8").strip())
        bin_offset = json_end + 8
        bin_blob = raw[bin_offset:] if len(raw) > bin_offset else b""

        accessors = gltf.get("accessors", [])
        buffer_views = gltf.get("bufferViews", [])
        meshes = gltf.get("meshes", [])
        if not meshes or not meshes[0].get("primitives"):
            return None

        prim = meshes[0]["primitives"][0]
        attrs = prim.get("attributes", {})

        def _read_acc(acc_idx: int | None, dtype: Any, cols: int) -> np.ndarray | None:
            if acc_idx is None or acc_idx < 0 or acc_idx >= len(accessors):
                return None
            acc = accessors[acc_idx]
            bv_idx = acc.get("bufferView")
            if bv_idx is None or bv_idx < 0 or bv_idx >= len(buffer_views):
                return None
            bv = buffer_views[bv_idx]
            offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
            count = int(acc.get("count", 0))
            arr = np.frombuffer(bin_blob, dtype=dtype, count=count * cols, offset=offset).copy()
            return arr.reshape((count, cols)) if cols > 1 else arr

        verts = _read_acc(attrs.get("POSITION"), np.float32, 3)
        if verts is None or len(verts) == 0:
            return None
        norms = _read_acc(attrs.get("NORMAL"), np.float32, 3)
        colors = _read_acc(attrs.get("COLOR_0"), np.float32, 3)
        if colors is None or len(colors) != len(verts):
            colors = np.full((len(verts), 3), 0.78, dtype=np.float32)

        idx_acc_id = prim.get("indices")
        faces: np.ndarray | None = None
        if idx_acc_id is not None and 0 <= idx_acc_id < len(accessors):
            comp_type = accessors[idx_acc_id].get("componentType", 5125)
            idx_dtype = np.uint16 if comp_type == 5123 else np.uint32
            flat_idx = _read_acc(idx_acc_id, idx_dtype, 1)
            if flat_idx is not None and len(flat_idx) >= 3:
                faces = flat_idx[: (len(flat_idx) // 3) * 3].reshape((-1, 3)).astype(np.int32)
        if faces is None or len(faces) == 0:
            n_tri = len(verts) // 3
            faces = np.arange(n_tri * 3, dtype=np.int32).reshape((-1, 3))

        base_mesh = {
            "vertices": verts,
            "faces": faces,
            "vertex_colors": colors[:, :3],
            "normals": norms if norms is not None else compute_vertex_normals(verts, faces),
        }
        rigged = auto_rig_humanoid_mesh(base_mesh, normalize_scale=False)

        joints_0 = _read_acc(attrs.get("JOINTS_0"), np.uint16, 4)
        weights_0 = _read_acc(attrs.get("WEIGHTS_0"), np.float32, 4)
        if joints_0 is not None and weights_0 is not None and len(joints_0) == len(verts):
            rigged["joints_0"] = joints_0
            rigged["weights_0"] = weights_0

        # Restore morph targets if present in primitive
        target_names = (meshes[0].get("extras") or {}).get("targetNames") or []
        prim_targets = prim.get("targets") or []
        morph_targets: dict[str, np.ndarray] = {}
        for t_i, t_dict in enumerate(prim_targets):
            pos_delta = _read_acc(t_dict.get("POSITION"), np.float32, 3)
            if pos_delta is not None and len(pos_delta) == len(verts):
                m_name = target_names[t_i] if t_i < len(target_names) else f"morph_{t_i}"
                morph_targets[str(m_name)] = pos_delta

        if morph_targets:
            rigged["morph_targets"] = morph_targets
        else:
            rigged = build_avatar_facial_blendshapes(rigged)

        rigged["spring_chains"] = []
        rigged["vrm_path"] = str(p)
        rigged["path"] = str(p)
        return rigged
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 8. Forward Kinematics Pose Solver, Spring-Bone Physics & Multi-Pass Renderer
# ---------------------------------------------------------------------------

POSE_PRESETS: dict[str, dict[str, Any]] = {
    "T-Pose (Canonical)": {
        "rotations": {},
        "expressions": {"relaxed": 0.2},
    },
    "A-Pose (Natural Relaxed)": {
        "rotations": {
            "leftUpperArm": [0.0, 0.0, -32.0],
            "rightUpperArm": [0.0, 0.0, 32.0],
            "leftLowerArm": [10.0, 0.0, -8.0],
            "rightLowerArm": [10.0, 0.0, 8.0],
        },
        "expressions": {"happy": 0.25},
    },
    "Peace Sign ✌️ (Kawaii / VTuber)": {
        "rotations": {
            "head": [6.0, -8.0, 10.0],
            "chest": [0.0, 5.0, -4.0],
            "leftUpperArm": [15.0, 20.0, -45.0],
            "leftLowerArm": [35.0, 0.0, -15.0],
            "rightUpperArm": [45.0, -15.0, 25.0],
            "rightLowerArm": [85.0, 10.0, 35.0],
        },
        "expressions": {"happy": 0.9, "blinkRight": 0.85, "aa": 0.35},
    },
    "Wave Hello 👋 (Friendly Streamer)": {
        "rotations": {
            "head": [4.0, 6.0, -8.0],
            "leftUpperArm": [0.0, 0.0, -35.0],
            "rightUpperArm": [15.0, 0.0, 55.0],
            "rightLowerArm": [20.0, 0.0, 65.0],
        },
        "expressions": {"happy": 0.8, "ee": 0.5},
    },
    "Hero Action Stance": {
        "rotations": {
            "hips": [0.0, 18.0, 0.0],
            "chest": [8.0, -12.0, 0.0],
            "head": [-5.0, 12.0, 0.0],
            "leftUpperArm": [30.0, 15.0, -25.0],
            "leftLowerArm": [65.0, 0.0, 0.0],
            "rightUpperArm": [-25.0, -15.0, 30.0],
            "rightLowerArm": [45.0, 0.0, 0.0],
            "leftUpperLeg": [18.0, 0.0, 8.0],
            "rightUpperLeg": [-18.0, 0.0, -8.0],
        },
        "expressions": {"angry": 0.75},
    },
    "Shocked Anime Reaction 😱": {
        "rotations": {
            "chest": [-12.0, 0.0, 0.0],
            "head": [-10.0, 0.0, 6.0],
            "leftUpperArm": [55.0, 20.0, -15.0],
            "leftLowerArm": [95.0, 0.0, 20.0],
            "rightUpperArm": [55.0, -20.0, 15.0],
            "rightLowerArm": [95.0, 0.0, -20.0],
        },
        "expressions": {"surprised": 1.0, "anime_shock_eyes": 0.9, "oh": 0.95},
    },
}


def _euler_deg_to_mat3(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return mz @ my @ mx


def evaluate_avatar_pose_and_physics(
    avatar_data: dict[str, Any],
    *,
    bone_rotations_deg: dict[str, list[float] | tuple[float, float, float]] | None = None,
    expression_weights: dict[str, float] | None = None,
    physics_preset: str = "Normal",
    time_sec: float = 0.0,
) -> dict[str, Any]:
    """
    Applies BlendShape morphs, evaluates forward kinematics on the 25 humanoid bones,
    runs Linear Blend Skinning (LBS) on the mesh, and applies dynamic Spring-Bone physics
    displacement on HairDeck strands and fun props!
    """
    base_verts = np.asarray(avatar_data["vertices"], dtype=np.float32).copy()
    faces = np.asarray(avatar_data["faces"], dtype=np.int32)
    cols = np.asarray(avatar_data["vertex_colors"], dtype=np.float32)
    bone_names: list[str] = list(avatar_data["bone_names"])
    bone_parents: list[int] = list(avatar_data["bone_parents"])
    rest_positions = np.asarray(avatar_data["bone_positions"], dtype=np.float32)
    local_translations = np.asarray(avatar_data["local_translations"], dtype=np.float32)
    joints_0 = np.asarray(avatar_data["joints_0"], dtype=np.int32)
    weights_0 = np.asarray(avatar_data["weights_0"], dtype=np.float32)
    morph_targets: dict[str, np.ndarray] = dict(avatar_data.get("morph_targets") or {})

    # 1. Apply Morph Targets (BlendShapes)
    morphed_verts = base_verts.copy()
    if expression_weights:
        for m_name, w_val in expression_weights.items():
            if abs(float(w_val)) > 1e-4 and m_name in morph_targets:
                morphed_verts = morphed_verts + float(w_val) * np.asarray(morph_targets[m_name], dtype=np.float32)

    # 2. Forward Kinematics across humanoid bone hierarchy
    num_bones = len(bone_names)
    rot_map = bone_rotations_deg or {}
    world_rot = [np.eye(3, dtype=np.float32) for _ in range(num_bones)]
    world_pos = np.zeros((num_bones, 3), dtype=np.float32)

    for b_idx, b_name in enumerate(bone_names):
        euler = rot_map.get(b_name, (0.0, 0.0, 0.0))
        local_r = _euler_deg_to_mat3(euler[0], euler[1], euler[2])
        p_idx = bone_parents[b_idx]
        if p_idx >= 0:
            world_rot[b_idx] = world_rot[p_idx] @ local_r
            world_pos[b_idx] = world_pos[p_idx] + (world_rot[p_idx] @ local_translations[b_idx])
        else:
            world_rot[b_idx] = local_r
            world_pos[b_idx] = local_translations[b_idx]

    # 3. Linear Blend Skinning (LBS)
    # v' = sum_k w_k * (R_k * (v - rest_pos_k) + world_pos_k)
    rot_stack = np.stack(world_rot, axis=0)  # (B, 3, 3)
    skinned_verts = np.zeros_like(morphed_verts)
    for k in range(4):
        b_ids = joints_0[:, k]
        w_k = weights_0[:, k, None]
        rel = morphed_verts - rest_positions[b_ids]
        rotated = np.einsum("nij,nj->ni", rot_stack[b_ids], rel)
        skinned_verts += w_k * (rotated + world_pos[b_ids])

    # 4. Secondary Spring-Bone Physics on HairDeck strands & props
    head_idx = bone_names.index("head") if "head" in bone_names else 0
    head_rot = world_rot[head_idx]
    head_delta = world_pos[head_idx] - rest_positions[head_idx]
    # Angular velocity / tilt proxy from head rotation off identity
    head_tilt_vec = head_rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32) - np.array([0.0, 1.0, 0.0], dtype=np.float32)

    hair_start = int(avatar_data.get("hair_vertex_start", len(skinned_verts)))
    hair_count = int(avatar_data.get("hair_vertex_count", 0))
    if hair_count > 0 and hair_start + hair_count <= len(skinned_verts):
        h_slice = slice(hair_start, hair_start + hair_count)
        h_verts = skinned_verts[h_slice]
        h_y_max = float(h_verts[:, 1].max())
        h_y_min = float(h_verts[:, 1].min())
        h_span = max(0.05, h_y_max - h_y_min)
        strand_t = np.clip((h_y_max - h_verts[:, 1]) / h_span, 0.0, 1.0)[:, None]

        # Physics preset modifiers
        preset_key = (physics_preset or "Normal").lower()
        bounce_mult = 1.0
        anti_gravity = 0.0
        wind_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if "bouncy" in preset_key:
            bounce_mult = 2.2
        elif "zero" in preset_key or "space" in preset_key:
            anti_gravity = 0.09
            bounce_mult = 1.5
        elif "wind" in preset_key:
            wind_vec = np.array([0.06 * math.cos(time_sec * 3.0), 0.02, 0.03 * math.sin(time_sec * 2.5)], dtype=np.float32)

        sway_offset = (
            (-head_tilt_vec[None, :] * 0.14 * bounce_mult - head_delta[None, :] * 0.25 * bounce_mult)
            + np.array([0.0, anti_gravity, 0.0], dtype=np.float32)[None, :]
            + wind_vec[None, :]
        ) * (strand_t ** 1.4)
        skinned_verts[h_slice] = h_verts + sway_offset.astype(np.float32)

    normals = compute_vertex_normals(skinned_verts, faces)
    return {
        "vertices": skinned_verts,
        "faces": faces,
        "vertex_colors": cols,
        "normals": normals,
        "world_bone_positions": world_pos,
        "bone_names": bone_names,
        "bone_parents": bone_parents,
    }


def render_avatar_multipass(
    posed_state: dict[str, Any],
    *,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 10.0,
    resolution: int = 512,
    bg_mode: str = "dark_studio",
) -> dict[str, np.ndarray]:
    """
    Renders 5 synchronized passes of the posed 3D avatar at `(H, W, 3)` / `(H, W)`:
      - `rgb`: Shaded color render
      - `mask`: Alpha silhouette mask
      - `depth`: Normalized depth map `(H, W, 3)`
      - `normal`: World/Camera-space normal map `(H, W, 3)`
      - `openpose`: 2D projected humanoid skeleton stick figure `(H, W, 3)` for ControlNet
    """
    res = int(np.clip(resolution, 128, 1536))
    bg_map = {
        "dark_studio": (0.11, 0.12, 0.16),
        "green_screen": (0.0, 1.0, 0.0),
        "white_studio": (0.96, 0.96, 0.98),
    }
    bg_col = bg_map.get(bg_mode, (0.11, 0.12, 0.16))

    verts = np.asarray(posed_state["vertices"], dtype=np.float32)
    faces = np.asarray(posed_state["faces"], dtype=np.int32)
    cols = np.asarray(posed_state["vertex_colors"], dtype=np.float32)
    norms = np.asarray(posed_state["normals"], dtype=np.float32)

    rgb, mask = render_3d_view(
        positions=verts,
        colors=cols,
        faces=faces,
        normals=norms,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        resolution=res,
        bg_color=bg_col,
    )

    # Normal map pass (encode normals in [0, 1] RGB)
    norm_colors = np.clip(0.5 + 0.5 * norms, 0.0, 1.0).astype(np.float32)
    normal_img, _ = render_3d_view(
        positions=verts,
        colors=norm_colors,
        faces=faces,
        normals=None,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        resolution=res,
        bg_color=(0.5, 0.5, 1.0),
    )
    normal_img = normal_img * mask[..., None]

    # Depth pass
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    rot_y = np.array([[math.cos(az), 0.0, math.sin(az)], [0.0, 1.0, 0.0], [-math.sin(az), 0.0, math.cos(az)]], dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(el), -math.sin(el)], [0.0, math.sin(el), math.cos(el)]], dtype=np.float32)
    rot_mat = rot_x @ rot_y
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    centered = verts - center
    max_ext = max(float(np.max(np.linalg.norm(centered, axis=-1))), 1e-6)
    cam_pts = (centered / max_ext * 0.80) @ rot_mat.T
    z_vals = cam_pts[:, 2]
    z_min, z_max = float(z_vals.min()), float(z_vals.max())
    z_norm = (z_vals - z_min) / max(z_max - z_min, 1e-5)
    depth_cols = np.repeat(np.clip(0.15 + 0.85 * z_norm[:, None], 0.0, 1.0), 3, axis=-1).astype(np.float32)
    depth_img, _ = render_3d_view(
        positions=verts,
        colors=depth_cols,
        faces=None,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        resolution=res,
        bg_color=(0.0, 0.0, 0.0),
        point_radius=2,
    )
    depth_img = depth_img * mask[..., None]

    # OpenPose 2D skeleton pass
    pose_pil = Image.new("RGB", (res, res), (0, 0, 0))
    draw = ImageDraw.Draw(pose_pil)
    world_bones = np.asarray(posed_state["world_bone_positions"], dtype=np.float32)
    bone_parents: list[int] = list(posed_state["bone_parents"])
    b_cam = ((world_bones - center) / max_ext * 0.80) @ rot_mat.T
    cam_dist = 2.4
    bz = np.maximum(cam_dist - b_cam[:, 2], 0.2)
    focal = 1.65
    b_px = ((b_cam[:, 0] * focal / bz) * 0.5 + 0.5) * (res - 1)
    b_py = ((-b_cam[:, 1] * focal / bz) * 0.5 + 0.5) * (res - 1)

    limb_palette = [
        (255, 0, 85),
        (255, 85, 0),
        (255, 170, 0),
        (170, 255, 0),
        (0, 255, 85),
        (0, 255, 255),
        (0, 128, 255),
        (128, 0, 255),
    ]
    for b_idx, p_idx in enumerate(bone_parents):
        if p_idx >= 0:
            col = limb_palette[b_idx % len(limb_palette)]
            draw.line(
                [(float(b_px[p_idx]), float(b_py[p_idx])), (float(b_px[b_idx]), float(b_py[b_idx]))],
                fill=col,
                width=max(2, res // 110),
            )
    r_dot = max(3, res // 90)
    for b_idx in range(len(world_bones)):
        x, y = float(b_px[b_idx]), float(b_py[b_idx])
        draw.ellipse((x - r_dot, y - r_dot, x + r_dot, y + r_dot), fill=(255, 255, 255))

    openpose_img = np.asarray(pose_pil, dtype=np.float32) / 255.0
    return {
        "rgb": rgb,
        "mask": mask,
        "depth": depth_img,
        "normal": normal_img,
        "openpose": openpose_img,
    }


# ---------------------------------------------------------------------------
# 9. In-Memory Live Webcam / Pose Studio Store & ComfyUI HTTP API Routes
# ---------------------------------------------------------------------------

_STUDIO_LIVE_CAPTURE_CACHE: dict[str, dict[str, Any]] = {}
_STUDIO_LOCK = threading.RLock()


def store_live_studio_capture(session_id: str, payload: dict[str, Any]) -> None:
    with _STUDIO_LOCK:
        _STUDIO_LIVE_CAPTURE_CACHE[str(session_id or "default")] = payload


def get_live_studio_capture(session_id: str) -> dict[str, Any] | None:
    with _STUDIO_LOCK:
        return _STUDIO_LIVE_CAPTURE_CACHE.get(str(session_id or "default"))


def _decode_data_uri_to_rgb(data_uri: str, target_res: int = 512) -> tuple[np.ndarray, np.ndarray] | None:
    if not data_uri or "," not in data_uri:
        return None
    try:
        _, b64_part = data_uri.split(",", 1)
        raw = base64.b64decode(b64_part)
        pil = Image.open(io.BytesIO(raw)).convert("RGBA")
        if pil.size != (target_res, target_res):
            pil = pil.resize((target_res, target_res), Image.Resampling.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return arr[..., :3], arr[..., 3]
    except Exception:
        return None


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None) is not None:
    routes = PromptServer.instance.routes

    @routes.post("/ziggart/avatar/sync_capture")
    async def _api_ziggart_avatar_sync_capture(request: Any) -> Any:
        try:
            data = await request.json()
            session_id = str(data.get("session_id") or "default")
            store_live_studio_capture(session_id, data)
            return web.json_response({"ok": True, "session_id": session_id})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    @routes.get("/ziggart/avatar/model_file")
    async def _api_ziggart_avatar_model_file(request: Any) -> Any:
        file_path = request.rel_url.query.get("path", "")
        if not file_path:
            return web.Response(status=404, text="Missing path parameter")
        resolved = Path(file_path).resolve()
        if not resolved.is_file() or resolved.suffix.lower() not in (".vrm", ".glb", ".ply", ".obj"):
            return web.Response(status=404, text="3D model file not found")
        return web.FileResponse(path=str(resolved))


# ---------------------------------------------------------------------------
# 10. ComfyUI Node Definitions (Ziggart/3D/Avatar)
# ---------------------------------------------------------------------------


class ZiggartAvatarModelDownloader:
    """Automatically downloads and permanently caches 3D Avatar & HairDeck models on first run."""

    DESCRIPTION = (
        "Automatically downloads 3D Avatar models (HairDeck, Make-It-Animatable, UniRig, BiRefNet) "
        "into ComfyUI/models/3d/ziggart_avatars on the first run and keeps them cached permanently "
        "so subsequent runs start instantaneously."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "ensure_models"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("models_cache_dir", "cache_status_json")
    OUTPUT_TOOLTIPS = (
        "Root folder on disk where 3D Avatar models are permanently cached (ComfyUI/models/3d/ziggart_avatars).",
        "JSON status summary of cached vs downloaded model weights.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "download_hairdeck": ("BOOLEAN", {"default": True}),
                "download_make_it_animatable": ("BOOLEAN", {"default": True}),
                "download_unirig": ("BOOLEAN", {"default": False}),
                "download_birefnet_matting": ("BOOLEAN", {"default": False}),
                "auto_download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Downloads missing weights once on first run and reuses cached files afterward.",
                    },
                ),
                "force_redownload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "hf_token": (
                    "STRING",
                    {"default": "", "tooltip": "Optional Hugging Face token for gated models."},
                ),
            },
        }

    def ensure_models(
        self,
        download_hairdeck: bool = True,
        download_make_it_animatable: bool = True,
        download_unirig: bool = False,
        download_birefnet_matting: bool = False,
        auto_download_if_missing: bool = True,
        force_redownload: bool = False,
        hf_token: str = "",
    ) -> tuple[str, str]:
        selected = []
        if download_hairdeck:
            selected.append("hairdeck")
        if download_make_it_animatable:
            selected.append("make_it_animatable")
        if download_unirig:
            selected.append("unirig")
        if download_birefnet_matting:
            selected.append("birefnet_matting")

        results: dict[str, Any] = {}
        for key in selected:
            results[key] = ensure_avatar_model_cached(
                key,
                hf_token=hf_token,
                auto_download=bool(auto_download_if_missing),
                force_redownload=bool(force_redownload),
            )

        root_dir = str(get_avatar_models_root())
        return (root_dir, json.dumps({"models_root": root_dir, "models": results}, indent=2))


class ZiggartAvatarCanonicalPrep:
    """Segments character hair vs body and normalizes framing for clean 3D Avatar + HairDeck generation."""

    DESCRIPTION = (
        "Prepares a 2D character image for 3D Avatar creation: removes the background, centers the character "
        "in canonical proportions, segments the hair mask for HairDeck, and produces a bald/scalp-smoothed "
        "body image so 3D body generators (Hunyuan3D / SPAR3D / Idol) don't fuse long hair onto shoulders."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "prepare_avatar_inputs"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("body_ready_image", "hair_source_image", "hair_mask", "head_crop_image")
    OUTPUT_TOOLTIPS = (
        "Normalized character image with side/draping hair suppressed so 3D body generators produce clean neck & shoulder topology.",
        "Normalized character image preserving full hair for HairDeck 3D strand generation.",
        "Segmented binary/feathered hair mask.",
        "Square head & hair crop centered on the upper body.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input 2D character illustration, render, or photo."}),
                "head_top_ratio": (
                    "FLOAT",
                    {
                        "default": 0.36,
                        "min": 0.18,
                        "max": 0.65,
                        "step": 0.02,
                        "tooltip": "Vertical proportion of the character occupied by the head & hair.",
                    },
                ),
                "suppress_draping_hair_on_body": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Removes side/draping hair from the body reference image to prevent fused shoulder geometry.",
                    },
                ),
                "target_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional foreground character mask."}),
            },
        }

    def prepare_avatar_inputs(
        self,
        image: Any,
        head_top_ratio: float = 0.36,
        suppress_draping_hair_on_body: bool = True,
        target_resolution: int = 1024,
        mask: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        pil_img = tensor_to_pil(image, mask=mask)
        prep = segment_avatar_hair_and_body(
            pil_img,
            head_top_ratio=float(head_top_ratio),
            inpaint_scalp=bool(suppress_draping_hair_on_body),
            target_resolution=int(target_resolution),
        )
        body_t = pil_to_tensor(prep["body_ready_pil"])
        hair_t = pil_to_tensor(prep["normalized_rgba"])
        head_t = pil_to_tensor(prep["head_crop_pil"])
        h_mask = prep["hair_mask"][None, ...].astype(np.float32)
        if torch is not None:
            h_mask_t = torch.from_numpy(h_mask)
        else:
            h_mask_t = h_mask
        return (body_t, hair_t, h_mask_t, head_t)


class ZiggartHairDeckGenerator:
    """Generates 3D stylized Hair Cards & physics Spring-Bone chains from a character image (HairDeck)."""

    DESCRIPTION = (
        "Generates animation-ready 3D stylized Hair Cards and hierarchical VRM Spring-Bone physics chains "
        "from a character image. Automatically caches HairDeck weights in ComfyUI/models/3d/ziggart_avatars/hairdeck "
        "on first run and includes a built-in strand-card & spring-bone synthesizer."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "generate_hair"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ZIGGART_HAIR", "STRING", "IMAGE")
    RETURN_NAMES = ("hair_data", "hair_glb_path", "preview_image")
    OUTPUT_TOOLTIPS = (
        "3D HairDeck strand cards and VRM Spring-Bone physics chains ready to connect to ZiggartVRMAssembler.",
        "File path to the exported 3D hair .glb mesh.",
        "4-view orthographic preview contact sheet of the generated 3D hair cards.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Character portrait or full-body image."}),
                "style_preset": (
                    ["anime_layered", "long_flowing", "twin_tails", "short_spiky"],
                    {"default": "anime_layered"},
                ),
                "num_strands": ("INT", {"default": 36, "min": 8, "max": 128, "step": 4}),
                "segments_per_strand": ("INT", {"default": 10, "min": 4, "max": 24, "step": 2}),
                "strand_width": ("FLOAT", {"default": 0.038, "min": 0.01, "max": 0.12, "step": 0.002}),
                "length_scale": ("FLOAT", {"default": 1.0, "min": 0.4, "max": 2.5, "step": 0.05}),
                "spring_stiffness": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 1.0, "step": 0.05}),
                "gravity_power": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.02}),
                "drag_force": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "auto_download_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically download and cache hyz317/HairDeck weights on first run.",
                    },
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
                "filename_prefix": ("STRING", {"default": "ziggart_hairdeck"}),
            },
            "optional": {
                "hair_mask": ("MASK", {"tooltip": "Optional hair mask from ZiggartAvatarCanonicalPrep."}),
                "hf_token": ("STRING", {"default": ""}),
            },
        }

    def generate_hair(
        self,
        image: Any,
        style_preset: str = "anime_layered",
        num_strands: int = 36,
        segments_per_strand: int = 10,
        strand_width: float = 0.038,
        length_scale: float = 1.0,
        spring_stiffness: float = 0.45,
        gravity_power: float = 0.08,
        drag_force: float = 0.35,
        auto_download_weights: bool = True,
        seed: int = 42,
        filename_prefix: str = "ziggart_hairdeck",
        hair_mask: Any = None,
        hf_token: str = "",
    ) -> tuple[dict[str, Any], str, Any]:
        cache_info = ensure_avatar_model_cached(
            "hairdeck",
            hf_token=hf_token,
            auto_download=bool(auto_download_weights),
        )
        pil_img = tensor_to_pil(image)
        mask_np = None
        if hair_mask is not None:
            m_arr = hair_mask.detach().cpu().float().numpy() if hasattr(hair_mask, "detach") else np.asarray(hair_mask, dtype=np.float32)
            while m_arr.ndim > 2:
                m_arr = m_arr[0]
            if m_arr.shape != (pil_img.height, pil_img.width):
                m_pil = Image.fromarray((np.clip(m_arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
                m_pil = m_pil.resize(pil_img.size, Image.Resampling.BILINEAR)
                mask_np = np.asarray(m_pil, dtype=np.float32) / 255.0
            else:
                mask_np = m_arr

        hair_data = generate_stylized_hair_cards_and_chains(
            pil_img,
            hair_mask=mask_np,
            style_preset=style_preset,
            num_strands=int(num_strands),
            segments_per_strand=int(segments_per_strand),
            strand_width=float(strand_width),
            length_scale=float(length_scale),
            stiffness=float(spring_stiffness),
            gravity_power=float(gravity_power),
            drag_force=float(drag_force),
            seed=int(seed),
        )
        hair_data["model_cache"] = cache_info

        out_glb = _next_output_path(filename_prefix or "ziggart_hairdeck", "glb")
        export_mesh_file(
            out_glb,
            hair_data["vertices"],
            hair_data["faces"],
            hair_data["vertex_colors"],
            hair_data["normals"],
        )
        hair_data["path"] = str(out_glb)

        preview = render_4view_contact_sheet(
            positions=hair_data["vertices"],
            colors=hair_data["vertex_colors"],
            faces=hair_data["faces"],
            normals=hair_data["normals"],
        )
        return (hair_data, str(out_glb), preview)


class ZiggartAutoRigger:
    """Automatically rigs a 3D character mesh with a 25-bone VRM Humanoid skeleton and 4-bone skinning weights."""

    DESCRIPTION = (
        "Automatically generates a 25-bone VRM Humanoid armature and smooth 4-bone skinning weights "
        "for any ZIGGART_MESH (from Hunyuan3D, SPAR3D, or Idol GaussianToMesh). Automatically caches "
        "Make-It-Animatable / UniRig weights on first run and includes a built-in capsule-heat humanoid rigger."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "rig_character"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ZIGGART_RIGGED_MESH", "STRING", "IMAGE")
    RETURN_NAMES = ("rigged_mesh", "rigged_glb_path", "skeleton_preview")
    OUTPUT_TOOLTIPS = (
        "Rigged 3D character dictionary containing the 25-bone VRM Humanoid skeleton, skinning weights, and inverse bind matrices.",
        "File path to the exported skinned .glb character file.",
        "Preview image overlaying the 25-bone OpenPose/VRM humanoid armature on the 3D character mesh.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "mesh_data": ("ZIGGART_MESH", {"tooltip": "Input 3D character mesh from Hunyuan3D, SPAR3D, or GaussianToMesh."}),
                "rig_backend": (
                    ["auto (MIA / Geometric)", "make_it_animatable", "unirig", "geometric_capsule_heat"],
                    {"default": "auto (MIA / Geometric)"},
                ),
                "skinning_smoothness": (
                    "FLOAT",
                    {
                        "default": 0.085,
                        "min": 0.02,
                        "max": 0.30,
                        "step": 0.005,
                        "tooltip": "Controls how smoothly skinning weights blend across adjacent joints.",
                    },
                ),
                "auto_download_weights": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically download and cache Make-It-Animatable / UniRig weights on first run.",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "ziggart_rigged_avatar"}),
            },
            "optional": {
                "hf_token": ("STRING", {"default": ""}),
            },
        }

    def rig_character(
        self,
        mesh_data: dict[str, Any],
        rig_backend: str = "auto (MIA / Geometric)",
        skinning_smoothness: float = 0.085,
        auto_download_weights: bool = True,
        filename_prefix: str = "ziggart_rigged_avatar",
        hf_token: str = "",
    ) -> tuple[dict[str, Any], str, Any]:
        model_key = "unirig" if "unirig" in rig_backend.lower() else "make_it_animatable"
        cache_info = {}
        if "geometric" not in rig_backend.lower():
            cache_info = ensure_avatar_model_cached(
                model_key,
                hf_token=hf_token,
                auto_download=bool(auto_download_weights),
            )

        rigged = auto_rig_humanoid_mesh(
            mesh_data,
            rig_backend=rig_backend,
            temperature=float(skinning_smoothness),
            normalize_scale=True,
        )
        rigged["model_cache"] = cache_info

        out_glb = _next_output_path(filename_prefix or "ziggart_rigged_avatar", "glb")
        export_vrm_avatar_file(out_glb, rigged, avatar_title="Ziggart Rigged Avatar")
        rigged["path"] = str(out_glb)

        # Render OpenPose + Mesh composite preview
        posed_state = evaluate_avatar_pose_and_physics(rigged)
        passes = render_avatar_multipass(posed_state, azimuth_deg=20.0, elevation_deg=12.0, resolution=512)
        comp = np.clip(passes["rgb"] * 0.72 + passes["openpose"] * 0.55, 0.0, 1.0)[None, ...]
        return (rigged, str(out_glb), numpy_batch_to_tensor(comp))


class ZiggartFacialBlendshapeBuilder:
    """Adds 20 VRM facial BlendShapes (lip-sync visemes, eye blinks, emotions, and fun cartoon morphs)."""

    DESCRIPTION = (
        "Builds 20 VRM standard & fun morph targets on a rigged 3D avatar mesh: lip-sync vowels "
        "(aa, ih, ou, ee, oh), independent eye blinks (blink, blinkLeft, blinkRight), gaze directions, "
        "emotions (happy, angry, sad, surprised), and fun cartoon morphs (anime_shock_eyes, puff_cheeks, chibi_big_head)."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "build_blendshapes"
    RETURN_TYPES = ("ZIGGART_RIGGED_MESH", "IMAGE")
    RETURN_NAMES = ("rigged_mesh", "expressions_sheet")
    OUTPUT_TOOLTIPS = (
        "Rigged 3D mesh enriched with 20 VRM facial BlendShapes (lip-sync visemes, blinks, emotions, and cartoon morphs).",
        "2x2 contact sheet previewing Neutral, Happy+Wink, Surprised, and Chibi/Lip-Sync expressions.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "rigged_mesh": ("ZIGGART_RIGGED_MESH",),
                "expression_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "include_fun_cartoon_morphs": ("BOOLEAN", {"default": True}),
            }
        }

    def build_blendshapes(
        self,
        rigged_mesh: dict[str, Any],
        expression_strength: float = 1.0,
        include_fun_cartoon_morphs: bool = True,
    ) -> tuple[dict[str, Any], Any]:
        enriched = build_avatar_facial_blendshapes(
            rigged_mesh,
            expression_strength=float(expression_strength),
            include_fun_morphs=bool(include_fun_cartoon_morphs),
        )
        # Render 2x2 expression showcase (Neutral, Happy+Blink, Surprised+Shock, Chibi+AA)
        presets = [
            {},
            {"happy": 1.0, "blinkRight": 0.9},
            {"surprised": 1.0, "oh": 0.9},
            {"chibi_big_head": 0.85, "aa": 0.9} if include_fun_cartoon_morphs else {"aa": 1.0},
        ]
        tiles = []
        for expr in presets:
            st = evaluate_avatar_pose_and_physics(enriched, expression_weights=expr)
            rgb, _ = render_3d_view(
                positions=st["vertices"],
                colors=st["vertex_colors"],
                faces=st["faces"],
                normals=st["normals"],
                azimuth_deg=12.0,
                elevation_deg=10.0,
                resolution=256,
            )
            tiles.append(rgb)
        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bot = np.concatenate([tiles[2], tiles[3]], axis=1)
        sheet = np.concatenate([top, bot], axis=0)[None, ...]
        return (enriched, numpy_batch_to_tensor(sheet))


class ZiggartVRMAssembler:
    """Combines a rigged 3D character mesh, HairDeck 3D hair cards & spring-bones, and exports a .VRM avatar."""

    DESCRIPTION = (
        "Assembles the rigged humanoid character mesh (ZIGGART_RIGGED_MESH), HairDeck 3D hair cards & "
        "spring-bone physics chains (ZIGGART_HAIR), and optional 3D accessories into a complete .vrm "
        "or .glb avatar ready for VRM Pose Editor and Live Webcam Mocap."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "assemble_vrm"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ZIGGART_AVATAR", "STRING", "IMAGE")
    RETURN_NAMES = ("avatar_data", "vrm_path", "preview_image")
    OUTPUT_TOOLTIPS = (
        "Complete ZIGGART_AVATAR structure combining rigged body, 3D HairDeck strands, Spring-Bone chains, and facial BlendShapes.",
        "File path to the exported .vrm or .glb avatar file.",
        "4-view contact sheet preview of the assembled 3D avatar.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "rigged_mesh": ("ZIGGART_RIGGED_MESH",),
                "avatar_title": ("STRING", {"default": "Ziggart Live Avatar"}),
                "avatar_author": ("STRING", {"default": "Ziggart Studio"}),
                "export_extension": (["vrm", "glb"], {"default": "vrm"}),
                "hair_offset_y": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "hair_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "filename_prefix": ("STRING", {"default": "ziggart_avatar"}),
            },
            "optional": {
                "hair_data": ("ZIGGART_HAIR", {"tooltip": "Optional 3D hair cards + spring bones from ZiggartHairDeckGenerator."}),
                "accessory": (
                    ["none", "cat_ears", "angel_halo", "cyber_visor", "demon_horns"],
                    {"default": "none"},
                ),
            },
        }

    def assemble_vrm(
        self,
        rigged_mesh: dict[str, Any],
        avatar_title: str = "Ziggart Live Avatar",
        avatar_author: str = "Ziggart Studio",
        export_extension: str = "vrm",
        hair_offset_y: float = 0.0,
        hair_scale: float = 1.0,
        filename_prefix: str = "ziggart_avatar",
        hair_data: dict[str, Any] | None = None,
        accessory: str = "none",
    ) -> tuple[dict[str, Any], str, Any]:
        verts = np.asarray(rigged_mesh["vertices"], dtype=np.float32).copy()
        faces = np.asarray(rigged_mesh["faces"], dtype=np.int32).copy()
        cols = np.asarray(rigged_mesh["vertex_colors"], dtype=np.float32).copy()
        joints_0 = np.asarray(rigged_mesh["joints_0"], dtype=np.uint16).copy()
        weights_0 = np.asarray(rigged_mesh["weights_0"], dtype=np.float32).copy()
        bone_names: list[str] = list(rigged_mesh["bone_names"])
        bone_positions = np.asarray(rigged_mesh["bone_positions"], dtype=np.float32)
        morph_targets: dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.float32).copy()
            for k, v in (rigged_mesh.get("morph_targets") or {}).items()
        }

        head_idx = bone_names.index("head") if "head" in bone_names else 0
        head_pos = bone_positions[head_idx]
        spring_chains: list[dict[str, Any]] = list(rigged_mesh.get("spring_chains") or [])

        hair_vertex_start = len(verts)
        hair_vertex_count = 0

        def _append_attachment_mesh(att: dict[str, Any], scale_f: float = 1.0, offset_y: float = 0.0) -> None:
            nonlocal verts, faces, cols, joints_0, weights_0
            a_verts = np.asarray(att["vertices"], dtype=np.float32).copy()
            a_faces = np.asarray(att["faces"], dtype=np.int32).copy()
            a_cols = np.asarray(att["vertex_colors"], dtype=np.float32).copy()
            if len(a_verts) == 0:
                return

            # Align attachment relative to actual head bone position
            src_center = np.asarray(att.get("head_center", head_pos), dtype=np.float32)
            a_verts = (a_verts - src_center[None, :]) * float(scale_f) + head_pos[None, :]
            a_verts[:, 1] += float(offset_y)

            v_off = len(verts)
            verts = np.vstack([verts, a_verts])
            faces = np.vstack([faces, a_faces + v_off])
            cols = np.vstack([cols, a_cols])

            # Skin attachment vertices 100% to the `head` bone so they follow head rotation
            a_j = np.zeros((len(a_verts), 4), dtype=np.uint16)
            a_j[:, 0] = head_idx
            a_w = np.zeros((len(a_verts), 4), dtype=np.float32)
            a_w[:, 0] = 1.0
            joints_0 = np.vstack([joints_0, a_j])
            weights_0 = np.vstack([weights_0, a_w])

            # Pad existing morph targets with zeros for the newly appended vertices
            pad_zeros = np.zeros((len(a_verts), 3), dtype=np.float32)
            for m_k in list(morph_targets.keys()):
                if m_k == "chibi_big_head":
                    rel_h = (a_verts - head_pos[None, :]) * 0.48
                    morph_targets[m_k] = np.vstack([morph_targets[m_k], rel_h.astype(np.float32)])
                else:
                    morph_targets[m_k] = np.vstack([morph_targets[m_k], pad_zeros])

            for ch in att.get("spring_chains", []):
                ch_copy = dict(ch)
                ch_joints = np.asarray(ch.get("joints", []), dtype=np.float32)
                if len(ch_joints):
                    ch_joints = (ch_joints - src_center[None, :]) * float(scale_f) + head_pos[None, :]
                    ch_joints[:, 1] += float(offset_y)
                    ch_copy["joints"] = ch_joints.tolist()
                ch_copy["vertex_start"] = int(ch.get("vertex_start", 0)) + v_off
                spring_chains.append(ch_copy)

        if hair_data is not None:
            hair_vertex_start = len(verts)
            _append_attachment_mesh(hair_data, scale_f=float(hair_scale), offset_y=float(hair_offset_y))
            hair_vertex_count = len(verts) - hair_vertex_start

        if accessory and accessory != "none":
            acc_mesh = generate_avatar_accessory_mesh(accessory, head_pos=tuple(head_pos.tolist()))
            if acc_mesh is not None:
                _append_attachment_mesh(acc_mesh, scale_f=1.0, offset_y=0.0)

        normals = compute_vertex_normals(verts, faces)
        avatar_data: dict[str, Any] = {
            **rigged_mesh,
            "vertices": verts,
            "faces": faces,
            "vertex_colors": cols,
            "normals": normals,
            "joints_0": joints_0,
            "weights_0": weights_0,
            "morph_targets": morph_targets,
            "spring_chains": spring_chains,
            "hair_vertex_start": hair_vertex_start,
            "hair_vertex_count": hair_vertex_count,
            "avatar_title": avatar_title,
            "avatar_author": avatar_author,
            "accessory": accessory,
        }

        ext = "vrm" if export_extension == "vrm" else "glb"
        out_path = _next_output_path(filename_prefix or "ziggart_avatar", ext)
        export_vrm_avatar_file(
            out_path,
            avatar_data,
            avatar_title=avatar_title,
            avatar_author=avatar_author,
        )
        avatar_data["vrm_path"] = str(out_path)
        avatar_data["path"] = str(out_path)

        preview = render_4view_contact_sheet(
            positions=verts,
            colors=cols,
            faces=faces,
            normals=normals,
        )
        return (avatar_data, str(out_path), preview)


class ZiggartAvatarFunFX:
    """Configures interactive cartoon physics, props, and webcam gesture-triggered emotes on an avatar."""

    DESCRIPTION = (
        "Adds fun interactive physics presets (Bouncy Anime, Zero-Gravity Space Hair, Bobblehead Chibi, Wind Tunnel), "
        "3D props (Cat Ears, Angel Halo, Cyber Visor, Demon Horns), and webcam gesture emote rules to a ZIGGART_AVATAR."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "apply_fun_fx"
    RETURN_TYPES = ("ZIGGART_AVATAR", "STRING")
    RETURN_NAMES = ("avatar_data", "fx_config_json")
    OUTPUT_TOOLTIPS = (
        "Avatar enriched with interactive physics presets, 3D fun accessories, and webcam gesture emote triggers.",
        "JSON summary of the active physics, prop, and emote configuration.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "avatar_data": ("ZIGGART_AVATAR",),
                "physics_preset": (
                    ["Normal", "Bouncy Anime", "Zero-Gravity Space Hair", "Bobblehead Chibi", "Wind Tunnel"],
                    {"default": "Bouncy Anime"},
                ),
                "add_accessory": (
                    ["keep_existing", "cat_ears", "angel_halo", "cyber_visor", "demon_horns"],
                    {"default": "keep_existing"},
                ),
                "peace_sign_gesture_emote": (
                    ["happy_star_burst", "wink_heart", "none"],
                    {"default": "happy_star_burst"},
                ),
                "open_mouth_gesture_emote": (
                    ["anime_shock_zoom", "fire_aura", "none"],
                    {"default": "anime_shock_zoom"},
                ),
                "voice_lipsync_sensitivity": ("FLOAT", {"default": 1.25, "min": 0.2, "max": 3.0, "step": 0.05}),
            }
        }

    def apply_fun_fx(
        self,
        avatar_data: dict[str, Any],
        physics_preset: str = "Bouncy Anime",
        add_accessory: str = "keep_existing",
        peace_sign_gesture_emote: str = "happy_star_burst",
        open_mouth_gesture_emote: str = "anime_shock_zoom",
        voice_lipsync_sensitivity: float = 1.25,
    ) -> tuple[dict[str, Any], str]:
        updated = dict(avatar_data)
        fx_config = {
            "physics_preset": physics_preset,
            "add_accessory": add_accessory,
            "peace_sign_gesture_emote": peace_sign_gesture_emote,
            "open_mouth_gesture_emote": open_mouth_gesture_emote,
            "voice_lipsync_sensitivity": float(voice_lipsync_sensitivity),
        }
        updated["fun_fx_config"] = fx_config

        if add_accessory != "keep_existing" and updated.get("accessory") != add_accessory:
            assembler = ZiggartVRMAssembler()
            updated, _, _ = assembler.assemble_vrm(
                rigged_mesh=updated,
                avatar_title=updated.get("avatar_title", "Ziggart Fun Avatar"),
                avatar_author=updated.get("avatar_author", "Ziggart Studio"),
                export_extension="vrm",
                accessory=add_accessory,
            )
            updated["fun_fx_config"] = fx_config

        return (updated, json.dumps(fx_config, indent=2))


class ZiggartVRMPoseAndMocapStudio:
    """
    Interactive 3D VRM Pose Editor & Live Webcam Mocap ("Replace Me!") Studio Node.
    Pairs with `web/ziggart_avatar_studio.js` for real-time 60 FPS webcam tracking & pose editing,
    and outputs synchronized RGB, Mask, Depth, Normal, and OpenPose tensors into ComfyUI workflows.
    """

    DESCRIPTION = (
        "Interactive 3D VRM Pose Editor & Live Webcam Mocap Studio! Use the embedded 3D canvas to manually "
        "pose bones, or click '🎥 Live Webcam Mocap (Replace Me)' to drive the avatar with your webcam + mic "
        "in real time (with Pop-Out Broadcast Window for OBS/Discord/Zoom). Outputs IMAGE, MASK, DEPTH, NORMAL, and OPENPOSE."
    )
    CATEGORY = "Ziggart/3D/Avatar"
    FUNCTION = "execute_studio"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "mask", "depth_image", "normal_image", "openpose_image", "vrm_path")
    OUTPUT_TOOLTIPS = (
        "Rendered RGB image (or batch) of the posed/mocap-driven 3D avatar.",
        "Foreground silhouette mask of the posed 3D avatar.",
        "Normalized depth map aligned with the posed 3D avatar for ControlNet Depth.",
        "World/camera-space normal map aligned with the posed 3D avatar for ControlNet Normal.",
        "OpenPose 25-bone skeleton projection aligned with the posed 3D avatar for ControlNet OpenPose.",
        "File path to the active .vrm / .glb avatar file.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "studio_mode": (
                    [
                        "Pose Preset / Manual Sliders",
                        "Live Webcam Mocap Capture (Browser Studio)",
                        "Animated Pose Turntable Batch",
                    ],
                    {"default": "Pose Preset / Manual Sliders"},
                ),
                "pose_preset": (
                    list(POSE_PRESETS.keys()),
                    {"default": "Peace Sign ✌️ (Kawaii / VTuber)"},
                ),
                "background_mode": (
                    ["dark_studio", "green_screen", "white_studio", "webcam_replace_me"],
                    {"default": "dark_studio"},
                ),
                "camera_azimuth": ("FLOAT", {"default": 12.0, "min": -180.0, "max": 180.0, "step": 2.0}),
                "camera_elevation": ("FLOAT", {"default": 10.0, "min": -60.0, "max": 60.0, "step": 2.0}),
                "head_yaw": ("FLOAT", {"default": 0.0, "min": -65.0, "max": 65.0, "step": 1.0}),
                "head_pitch": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 1.0}),
                "head_roll": ("FLOAT", {"default": 0.0, "min": -45.0, "max": 45.0, "step": 1.0}),
                "mouth_open_aa": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "eye_blink": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "emotion_happy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "chibi_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 60, "step": 1}),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 1536, "step": 64}),
                "session_id": ("STRING", {"default": "default"}),
            },
            "optional": {
                "avatar_data": ("ZIGGART_AVATAR",),
                "vrm_or_glb_path": ("STRING", {"default": ""}),
                "webcam_pose_json": ("STRING", {"default": ""}),
            },
        }

    def execute_studio(
        self,
        studio_mode: str = "Pose Preset / Manual Sliders",
        pose_preset: str = "Peace Sign ✌️ (Kawaii / VTuber)",
        background_mode: str = "dark_studio",
        camera_azimuth: float = 12.0,
        camera_elevation: float = 10.0,
        head_yaw: float = 0.0,
        head_pitch: float = 0.0,
        head_roll: float = 0.0,
        mouth_open_aa: float = 0.0,
        eye_blink: float = 0.0,
        emotion_happy: float = 0.5,
        chibi_scale: float = 0.0,
        num_frames: int = 1,
        resolution: int = 512,
        session_id: str = "default",
        avatar_data: dict[str, Any] | None = None,
        vrm_or_glb_path: str = "",
        webcam_pose_json: str = "",
    ) -> tuple[Any, Any, Any, Any, Any, str]:
        # Load .vrm/.glb from disk if provided, or build a default procedural avatar when disconnected
        if avatar_data is None and vrm_or_glb_path and str(vrm_or_glb_path).strip():
            avatar_data = load_vrm_or_glb_avatar(str(vrm_or_glb_path).strip())
        if avatar_data is None:
            dummy_pil = Image.new("RGB", (256, 256), (195, 145, 220))
            hair = generate_stylized_hair_cards_and_chains(dummy_pil, num_strands=20, segments_per_strand=6)
            rigged = auto_rig_humanoid_mesh(hair)
            rigged = build_avatar_facial_blendshapes(rigged)
            avatar_data = {
                **rigged,
                "spring_chains": hair["spring_chains"],
                "hair_vertex_start": 0,
                "hair_vertex_count": len(hair["vertices"]),
                "vrm_path": str(vrm_or_glb_path or ""),
            }

        preset_spec = POSE_PRESETS.get(pose_preset, POSE_PRESETS["T-Pose (Canonical)"])
        rotations: dict[str, list[float]] = {k: list(v) for k, v in preset_spec.get("rotations", {}).items()}
        expressions: dict[str, float] = dict(preset_spec.get("expressions", {}))

        # Apply slider overrides
        base_head = rotations.get("head", [0.0, 0.0, 0.0])
        if abs(head_pitch) > 1e-3 or abs(head_yaw) > 1e-3 or abs(head_roll) > 1e-3:
            rotations["head"] = [
                base_head[0] + float(head_pitch),
                base_head[1] + float(head_yaw),
                base_head[2] + float(head_roll),
            ]
        if mouth_open_aa > 0.0:
            expressions["aa"] = float(mouth_open_aa)
        if eye_blink > 0.0:
            expressions["blink"] = float(eye_blink)
        if emotion_happy >= 0.0:
            expressions["happy"] = float(emotion_happy)
        if chibi_scale > 0.0:
            expressions["chibi_big_head"] = float(chibi_scale)

        # Check if live browser webcam mocap or JSON override is available
        live_capture = get_live_studio_capture(session_id)
        parsed_json: dict[str, Any] = {}
        if webcam_pose_json and webcam_pose_json.strip():
            try:
                parsed_json = json.loads(webcam_pose_json)
            except Exception:
                parsed_json = {}
        elif live_capture is not None and "Live Webcam" in studio_mode:
            parsed_json = live_capture

        if isinstance(parsed_json.get("rotations"), dict):
            for b_k, r_v in parsed_json["rotations"].items():
                if isinstance(r_v, (list, tuple)) and len(r_v) == 3:
                    rotations[str(b_k)] = [float(r_v[0]), float(r_v[1]), float(r_v[2])]
        if isinstance(parsed_json.get("expressions"), dict):
            for e_k, e_v in parsed_json["expressions"].items():
                expressions[str(e_k)] = float(e_v)

        fx_cfg = avatar_data.get("fun_fx_config") or {}
        physics_preset = str(fx_cfg.get("physics_preset", "Bouncy Anime"))

        n_frames = max(1, int(num_frames)) if "Animated" in studio_mode else 1
        res = int(np.clip(resolution, 128, 1536))

        rgb_batch, mask_batch, depth_batch, norm_batch, pose_batch = [], [], [], [], []
        for f_idx in range(n_frames):
            t_sec = f_idx * 0.1
            az = float(camera_azimuth) + (360.0 * f_idx / float(n_frames) if n_frames > 1 else 0.0)
            posed = evaluate_avatar_pose_and_physics(
                avatar_data,
                bone_rotations_deg=rotations,
                expression_weights=expressions,
                physics_preset=physics_preset,
                time_sec=t_sec,
            )
            passes = render_avatar_multipass(
                posed,
                azimuth_deg=az,
                elevation_deg=float(camera_elevation),
                resolution=res,
                bg_mode=background_mode,
            )
            rgb_batch.append(passes["rgb"])
            mask_batch.append(passes["mask"])
            depth_batch.append(passes["depth"])
            norm_batch.append(passes["normal"])
            pose_batch.append(passes["openpose"])

        # If browser sent a direct WebGL canvas capture frame in Live Webcam mode, use it for frame 0 RGB
        if "Live Webcam" in studio_mode and isinstance(parsed_json.get("frame_data_uri"), str):
            decoded = _decode_data_uri_to_rgb(parsed_json["frame_data_uri"], target_res=res)
            if decoded is not None:
                rgb_batch[0], mask_batch[0] = decoded

        vrm_path = str(avatar_data.get("vrm_path") or avatar_data.get("path") or vrm_or_glb_path or "")

        # Notify frontend studio widget of the active VRM model path so it can auto-load in 3D canvas
        if PromptServer is not None and getattr(PromptServer, "instance", None) is not None and vrm_path:
            try:
                PromptServer.instance.send_sync(
                    "ziggart-avatar-loaded",
                    {"session_id": session_id, "vrm_path": vrm_path},
                )
            except Exception:
                pass

        mask_arr = np.stack(mask_batch, axis=0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_arr) if torch is not None else mask_arr

        return (
            numpy_batch_to_tensor(np.stack(rgb_batch, axis=0)),
            mask_tensor,
            numpy_batch_to_tensor(np.stack(depth_batch, axis=0)),
            numpy_batch_to_tensor(np.stack(norm_batch, axis=0)),
            numpy_batch_to_tensor(np.stack(pose_batch, axis=0)),
            vrm_path,
        )


NODE_CLASS_MAPPINGS = {
    "ZiggartAvatarModelDownloader": ZiggartAvatarModelDownloader,
    "ZiggartAvatarCanonicalPrep": ZiggartAvatarCanonicalPrep,
    "ZiggartHairDeckGenerator": ZiggartHairDeckGenerator,
    "ZiggartAutoRigger": ZiggartAutoRigger,
    "ZiggartFacialBlendshapeBuilder": ZiggartFacialBlendshapeBuilder,
    "ZiggartVRMAssembler": ZiggartVRMAssembler,
    "ZiggartAvatarFunFX": ZiggartAvatarFunFX,
    "ZiggartVRMPoseAndMocapStudio": ZiggartVRMPoseAndMocapStudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartAvatarModelDownloader": "Avatar 3D Model Auto-Downloader & Cache Manager",
    "ZiggartAvatarCanonicalPrep": "Avatar 2D Canonical T-Pose & Hair Splitter",
    "ZiggartHairDeckGenerator": "HairDeck 3D Hair Cards & Spring-Bone Generator",
    "ZiggartAutoRigger": "Avatar 3D Humanoid Auto-Rigger (MIA / UniRig / Geometric)",
    "ZiggartFacialBlendshapeBuilder": "Avatar Facial BlendShapes & Viseme Builder",
    "ZiggartVRMAssembler": "Avatar VRM 1.0 / GLB Assembler & SpringBone Linker",
    "ZiggartAvatarFunFX": "Avatar Fun Physics, Props & Emote Configurator",
    "ZiggartVRMPoseAndMocapStudio": "Interactive 3D VRM Pose Editor & Live Webcam Mocap",
}
