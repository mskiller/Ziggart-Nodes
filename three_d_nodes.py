from __future__ import annotations

import atexit
import gc
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
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import folder_paths
except ImportError:  # Allows lightweight unit tests outside ComfyUI.
    folder_paths = None

from .embedded_model import empty_accelerator_cache, release_comfy_vram


THREED_MODEL_FOLDER = "ziggart_3d"
SH_C0 = 0.28209479177387814


def _register_3d_folders() -> None:
    if folder_paths is None:
        return
    model_dir = os.path.join(folder_paths.models_dir, "3d")
    try:
        os.makedirs(model_dir, exist_ok=True)
    except OSError:
        pass
    try:
        folder_paths.add_model_folder_path(THREED_MODEL_FOLDER, model_dir, is_default=True)
    except TypeError:
        folder_paths.add_model_folder_path(THREED_MODEL_FOLDER, model_dir)


_register_3d_folders()


def _get_output_3d_dir(subfolder: str = "ziggart_3d") -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        base = Path(folder_paths.get_output_directory())
    else:
        base = Path(tempfile.gettempdir()) / "comfyui_output"
    target = base / subfolder if subfolder else base
    target.mkdir(parents=True, exist_ok=True)
    return target


def _next_output_path(prefix: str, ext: str, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or _get_output_3d_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in (prefix or "ziggart_3d"))
    ext = ext.lstrip(".").lower()
    timestamp = int(time.time() * 1000) % 100000000
    candidate = out_dir / f"{safe_prefix}_{timestamp}.{ext}"
    counter = 1
    while candidate.exists():
        candidate = out_dir / f"{safe_prefix}_{timestamp}_{counter}.{ext}"
        counter += 1
    return candidate


# ---------------------------------------------------------------------------
# Image & Mask Tensor Utilities
# ---------------------------------------------------------------------------


def tensor_to_pil(image: Any, mask: Any = None) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor/ndarray ([B,H,W,C] or [H,W,C]) and optional MASK to PIL."""
    if hasattr(image, "detach"):
        arr = image.detach().cpu().float().numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HWC or BHWC image with 1, 3, or 4 channels; got shape {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr_u8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr_u8.shape[-1] == 1:
        arr_u8 = np.repeat(arr_u8, 3, axis=-1)

    if mask is not None:
        if hasattr(mask, "detach"):
            m_arr = mask.detach().cpu().float().numpy()
        else:
            m_arr = np.asarray(mask, dtype=np.float32)
        while m_arr.ndim > 2:
            m_arr = m_arr[0]
        if m_arr.shape != arr_u8.shape[:2]:
            m_pil = Image.fromarray((np.clip(m_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
            m_pil = m_pil.resize((arr_u8.shape[1], arr_u8.shape[0]), Image.Resampling.BILINEAR)
            alpha_u8 = np.asarray(m_pil, dtype=np.uint8)
        else:
            alpha_u8 = (np.clip(m_arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        rgba = np.dstack([arr_u8[..., :3], alpha_u8])
        return Image.fromarray(rgba, mode="RGBA")

    if arr_u8.shape[-1] == 4:
        return Image.fromarray(arr_u8, mode="RGBA")
    return Image.fromarray(arr_u8, mode="RGB")


def pil_to_tensor(pil_img: Image.Image) -> Any:
    """Convert PIL Image to ComfyUI [1, H, W, 3] float32 tensor (or numpy array if torch unavailable)."""
    if pil_img.mode == "RGBA":
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        comp = rgb * alpha + (1.0 - alpha) * 1.0  # White background for clean 3D preview
        out = np.clip(comp, 0.0, 1.0)[None, ...]
    else:
        rgb_img = pil_img.convert("RGB")
        out = (np.asarray(rgb_img, dtype=np.float32) / 255.0)[None, ...]
    if torch is not None:
        return torch.from_numpy(out)
    return out


def numpy_batch_to_tensor(batch: np.ndarray) -> Any:
    arr = np.asarray(batch, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if torch is not None:
        return torch.from_numpy(arr)
    return arr


def remove_background_fallback(pil_img: Image.Image) -> Image.Image:
    """Remove background using hy3dgen.rembg, rembg, or corner-color heuristic fallback."""
    if pil_img.mode == "RGBA":
        alpha = np.asarray(pil_img)[..., 3]
        if np.min(alpha) < 250:
            return pil_img

    try:
        from hy3dgen.rembg import BackgroundRemover

        remover = BackgroundRemover()
        res = remover(pil_img.convert("RGB"))
        if isinstance(res, Image.Image):
            return res.convert("RGBA")
    except Exception:
        pass

    try:
        import rembg

        res = rembg.remove(pil_img.convert("RGB"))
        if isinstance(res, Image.Image):
            return res.convert("RGBA")
    except Exception:
        pass

    # Lightweight border-color keying fallback when no rembg library is installed
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    corners = np.vstack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    bg_color = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb - bg_color[None, None, :], axis=-1)
    alpha = np.clip((dist - 12.0) / 28.0, 0.0, 1.0)
    rgba = np.dstack([rgb.astype(np.uint8), (alpha * 255.0 + 0.5).astype(np.uint8)])
    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# 3D Gaussian Splat (.ply) Reader, Writer & Parser
# ---------------------------------------------------------------------------


def load_gaussian_ply(ply_path: str | Path) -> dict[str, Any]:
    """
    Parse a 3D Gaussian Splatting .ply file (binary_little_endian or ascii) into
    activated numpy arrays:
      - positions: (N, 3) float32
      - colors: (N, 3) float32 in [0, 1]
      - opacities: (N,) float32 in [0, 1]
      - scales: (N, 3) float32 (positive world-space scales)
      - rotations: (N, 4) float32 normalized quaternions
      - ply_path: str
    """
    path = Path(ply_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian PLY file not found: {path}")

    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY file (missing end_header): {path}")
            decoded = line.decode("ascii", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        body_offset = f.tell()

    if not header_lines or header_lines[0] != "ply":
        raise ValueError(f"File is not a valid PLY file: {path}")

    fmt = "ascii"
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex_element = False

    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
        "uchar": "<u1",
        "uint8": "<u1",
        "char": "<i1",
        "int8": "<i1",
        "ushort": "<u2",
        "uint16": "<u2",
        "short": "<i2",
        "int16": "<i2",
        "uint": "<u4",
        "uint32": "<u4",
        "int": "<i4",
        "int32": "<i4",
    }

    for line in header_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            if parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex_element = True
            else:
                in_vertex_element = False
        elif parts[0] == "property" and in_vertex_element and len(parts) >= 3:
            if parts[1] == "list":
                continue
            prop_type, prop_name = parts[1], parts[2]
            np_dtype = type_map.get(prop_type, "<f4")
            properties.append((prop_name, np_dtype))

    if vertex_count <= 0 or not properties:
        raise ValueError(f"PLY file contains no vertices: {path}")

    if fmt == "binary_little_endian":
        dtype = np.dtype(properties)
        with path.open("rb") as f:
            f.seek(body_offset)
            raw = np.fromfile(f, dtype=dtype, count=vertex_count)
    elif fmt == "ascii":
        prop_names = [p[0] for p in properties]
        rows = []
        with path.open("r", encoding="ascii", errors="ignore") as f:
            for _ in range(len(header_lines)):
                f.readline()
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                vals = [float(v) for v in line.strip().split()]
                if len(vals) >= len(prop_names):
                    rows.append(vals[: len(prop_names)])
        arr_2d = np.asarray(rows, dtype=np.float32)
        dtype = np.dtype([(name, "<f4") for name in prop_names])
        raw = np.zeros(len(rows), dtype=dtype)
        for idx, name in enumerate(prop_names):
            raw[name] = arr_2d[:, idx]
    else:
        raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")

    names = set(raw.dtype.names or ())
    positions = np.column_stack(
        [
            raw["x"].astype(np.float32) if "x" in names else np.zeros(len(raw), dtype=np.float32),
            raw["y"].astype(np.float32) if "y" in names else np.zeros(len(raw), dtype=np.float32),
            raw["z"].astype(np.float32) if "z" in names else np.zeros(len(raw), dtype=np.float32),
        ]
    )

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.column_stack([raw["f_dc_0"], raw["f_dc_1"], raw["f_dc_2"]]).astype(np.float32)
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    elif {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([raw["red"], raw["green"], raw["blue"]]).astype(np.float32)
        if np.max(rgb) > 1.0:
            rgb = rgb / 255.0
        colors = np.clip(rgb, 0.0, 1.0)
    else:
        colors = np.full((len(raw), 3), 0.75, dtype=np.float32)

    if "opacity" in names:
        raw_op = raw["opacity"].astype(np.float32)
        # Standard 3DGS stores logit opacity; if values lie outside [0, 1], apply sigmoid
        if np.min(raw_op) < -0.01 or np.max(raw_op) > 1.01:
            opacities = 1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))
        else:
            opacities = np.clip(raw_op, 0.0, 1.0)
    else:
        opacities = np.ones(len(raw), dtype=np.float32)

    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        raw_scales = np.column_stack([raw["scale_0"], raw["scale_1"], raw["scale_2"]]).astype(np.float32)
        if np.min(raw_scales) < 0.0:
            scales = np.exp(np.clip(raw_scales, -15.0, 5.0))
        else:
            scales = np.maximum(raw_scales, 1e-5)
    else:
        scales = np.full((len(raw), 3), 0.01, dtype=np.float32)

    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        rot = np.column_stack([raw["rot_0"], raw["rot_1"], raw["rot_2"], raw["rot_3"]]).astype(np.float32)
        norms = np.linalg.norm(rot, axis=-1, keepdims=True)
        rotations = np.where(norms > 1e-8, rot / np.maximum(norms, 1e-8), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    else:
        rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(raw), 1))

    return {
        "positions": positions,
        "colors": colors,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
        "num_gaussians": int(len(positions)),
        "ply_path": str(path),
    }


def save_gaussian_ply(
    output_path: str | Path,
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
) -> Path:
    """Save standard 3D Gaussian Splatting binary_little_endian .ply file."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(positions, dtype=np.float32)
    n = len(pos)
    col = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    f_dc = (col - 0.5) / SH_C0

    if opacities is None:
        logit_op = np.full(n, 2.2, dtype=np.float32)
    else:
        op = np.clip(np.asarray(opacities, dtype=np.float32), 1e-4, 1.0 - 1e-4)
        logit_op = np.log(op / (1.0 - op)).astype(np.float32)

    if scales is None:
        log_scales = np.full((n, 3), -4.6, dtype=np.float32)
    else:
        sc = np.maximum(np.asarray(scales, dtype=np.float32), 1e-6)
        log_scales = np.log(sc).astype(np.float32)

    if rotations is None:
        rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    else:
        rot = np.asarray(rotations, dtype=np.float32)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    structured = np.zeros(n, dtype=dtype)
    structured["x"] = pos[:, 0]
    structured["y"] = pos[:, 1]
    structured["z"] = pos[:, 2]
    structured["f_dc_0"] = f_dc[:, 0]
    structured["f_dc_1"] = f_dc[:, 1]
    structured["f_dc_2"] = f_dc[:, 2]
    structured["opacity"] = logit_op
    structured["scale_0"] = log_scales[:, 0]
    structured["scale_1"] = log_scales[:, 1]
    structured["scale_2"] = log_scales[:, 2]
    structured["rot_0"] = rot[:, 0]
    structured["rot_1"] = rot[:, 1]
    structured["rot_2"] = rot[:, 2]
    structured["rot_3"] = rot[:, 3]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as f:
        f.write(header)
        structured.tofile(f)

    return path


# ---------------------------------------------------------------------------
# Mesh Computation, Export (GLB / OBJ / PLY) & Gaussian -> Mesh Reconstruction
# ---------------------------------------------------------------------------


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smooth area-weighted vertex normals for a triangle mesh."""
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    normals = np.zeros_like(verts, dtype=np.float32)
    if len(tris) == 0 or len(verts) == 0:
        normals[:, 2] = 1.0
        return normals

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    np.add.at(normals, tris[:, 0], face_normals)
    np.add.at(normals, tris[:, 1], face_normals)
    np.add.at(normals, tris[:, 2], face_normals)

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(norms > 1e-8, normals / np.maximum(norms, 1e-8), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return normals


def laplacian_smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 2,
    lam: float = 0.35,
) -> np.ndarray:
    """Apply Taubin / Laplacian smoothing to reduce voxel staircase artifacts."""
    if iterations <= 0 or len(faces) == 0 or len(vertices) == 0:
        return np.asarray(vertices, dtype=np.float32)

    verts = np.asarray(vertices, dtype=np.float32).copy()
    tris = np.asarray(faces, dtype=np.int32)
    n = len(verts)

    # Directed edges from triangle connectivity
    edges_a = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2], tris[:, 1], tris[:, 2], tris[:, 0]])
    edges_b = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0], tris[:, 0], tris[:, 1], tris[:, 2]])
    degrees = np.zeros((n, 1), dtype=np.float32)
    np.add.at(degrees[:, 0], edges_a, 1.0)
    degrees = np.maximum(degrees, 1.0)

    mu = -0.37  # Taubin inflation step to prevent shrinkage
    for _ in range(iterations):
        for factor in (lam, mu):
            neighbor_sum = np.zeros_like(verts)
            np.add.at(neighbor_sum, edges_a, verts[edges_b])
            avg = neighbor_sum / degrees
            verts = verts + factor * (avg - verts)

    return verts


def decimate_mesh_vertex_clustering(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Fast spatial vertex-clustering mesh decimation when face count exceeds target_faces."""
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    cols = np.asarray(vertex_colors, dtype=np.float32) if vertex_colors is not None else None

    if target_faces <= 0 or len(tris) <= target_faces:
        return verts, tris, cols

    ratio = math.sqrt(max(target_faces, 100) / float(len(tris)))
    grid_bins = max(16, int(round(120 * ratio)))

    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    extent = np.maximum(bmax - bmin, 1e-6)
    coords = np.floor(((verts - bmin) / extent) * (grid_bins - 1)).astype(np.int64)
    keys = coords[:, 0] * (grid_bins * grid_bins) + coords[:, 1] * grid_bins + coords[:, 2]

    unique_keys, inverse = np.unique(keys, return_index=False, return_inverse=True)
    num_clusters = len(unique_keys)

    counts = np.bincount(inverse, minlength=num_clusters).astype(np.float32)[:, None]
    new_verts = np.zeros((num_clusters, 3), dtype=np.float32)
    np.add.at(new_verts, inverse, verts)
    new_verts /= np.maximum(counts, 1.0)

    if cols is not None:
        new_cols = np.zeros((num_clusters, 3), dtype=np.float32)
        np.add.at(new_cols, inverse, cols[:, :3])
        new_cols /= np.maximum(counts, 1.0)
    else:
        new_cols = None

    new_tris = inverse[tris]
    non_degenerate = (
        (new_tris[:, 0] != new_tris[:, 1])
        & (new_tris[:, 1] != new_tris[:, 2])
        & (new_tris[:, 2] != new_tris[:, 0])
    )
    new_tris = new_tris[non_degenerate].astype(np.int32)
    if len(new_tris) == 0:
        return verts, tris, cols
    return new_verts, new_tris, new_cols


def export_mesh_file(
    output_path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> Path:
    """
    Export a triangle mesh with vertex colors and normals to .glb (glTF 2.0 binary),
    .obj, or .ply without requiring any external C++ library.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower().lstrip(".")

    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.uint32)
    if normals is None or len(normals) != len(verts):
        norms = compute_vertex_normals(verts, tris.astype(np.int32))
    else:
        norms = np.asarray(normals, dtype=np.float32)

    if vertex_colors is None or len(vertex_colors) != len(verts):
        cols = np.full((len(verts), 3), 0.8, dtype=np.float32)
    else:
        cols = np.clip(np.asarray(vertex_colors, dtype=np.float32)[:, :3], 0.0, 1.0)

    if ext == "obj":
        lines = ["# Exported by Ziggart 3D Nodes\n"]
        for v, c in zip(verts, cols):
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        for n in norms:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for f in tris + 1:
            lines.append(f"f {f[0]}//{f[0]} {f[1]}//{f[1]} {f[2]}//{f[2]}\n")
        path.write_text("".join(lines), encoding="utf-8")
        return path

    if ext == "ply":
        col_u8 = (cols * 255.0 + 0.5).astype(np.uint8)
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(verts)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            f"element face {len(tris)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        body = [header]
        for v, n, c in zip(verts, norms, col_u8):
            body.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {n[0]:.5f} {n[1]:.5f} {n[2]:.5f} {c[0]} {c[1]} {c[2]}\n")
        for f in tris:
            body.append(f"3 {f[0]} {f[1]} {f[2]}\n")
        path.write_text("".join(body), encoding="ascii")
        return path

    # Default: GLB (glTF 2.0 binary container)
    pos_bytes = verts.astype("<f4").tobytes()
    norm_bytes = norms.astype("<f4").tobytes()
    rgba_cols = np.column_stack([cols, np.ones(len(verts), dtype=np.float32)]).astype("<f4")
    col_bytes = rgba_cols.tobytes()
    idx_bytes = tris.reshape(-1).astype("<u4").tobytes()

    def _pad4(b: bytes) -> bytes:
        rem = len(b) % 4
        return b if rem == 0 else b + b"\x00" * (4 - rem)

    pos_bytes = _pad4(pos_bytes)
    norm_bytes = _pad4(norm_bytes)
    col_bytes = _pad4(col_bytes)
    idx_bytes = _pad4(idx_bytes)

    offset_pos = 0
    offset_norm = offset_pos + len(pos_bytes)
    offset_col = offset_norm + len(norm_bytes)
    offset_idx = offset_col + len(col_bytes)
    bin_blob = pos_bytes + norm_bytes + col_bytes + idx_bytes

    v_min = verts.min(axis=0).astype(float).tolist() if len(verts) else [0.0, 0.0, 0.0]
    v_max = verts.max(axis=0).astype(float).tolist() if len(verts) else [0.0, 0.0, 0.0]

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "Ziggart-Nodes 3D Exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "ZiggartMesh"}],
        "materials": [
            {
                "name": "VertexColorMaterial",
                "pbrMetallicRoughness": {"metallicFactor": 0.05, "roughnessFactor": 0.75},
                "doubleSided": True,
            }
        ],
        "meshes": [
            {
                "name": "ZiggartMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offset_pos, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": offset_norm, "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": offset_col, "byteLength": len(col_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": offset_idx, "byteLength": len(idx_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(len(verts)),
                "type": "VEC3",
                "min": v_min,
                "max": v_max,
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(len(verts)),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": int(len(verts)),
                "type": "VEC4",
            },
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": int( tris.size ),
                "type": "SCALAR",
            },
        ],
    }

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


def _extract_surface_from_gaussian_grid(
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    grid_resolution: int = 96,
    iso_level: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure-NumPy Gaussian density field splatting + Dual-Voxel / Marching-Cubes surface
    extractor with weighted color projection.
    """
    res = int(np.clip(grid_resolution, 24, 256))
    bmin = positions.min(axis=0)
    bmax = positions.max(axis=0)
    span = np.maximum(bmax - bmin, 1e-4)
    pad = span * 0.08
    bmin = bmin - pad
    bmax = bmax + pad
    span = bmax - bmin

    # Map points to voxel grid [0, res-1]
    coords_f = ((positions - bmin) / span) * (res - 1)
    coords_i = np.clip(np.round(coords_f).astype(np.int32), 0, res - 1)

    density = np.zeros((res, res, res), dtype=np.float32)
    color_grid = np.zeros((res, res, res, 3), dtype=np.float32)

    weights = np.clip(opacities, 0.05, 1.0)
    # Splat center + 6-neighborhood weighted by Gaussian scale so surface is continuous
    mean_scale = np.mean(scales, axis=-1)
    voxel_size = float(np.max(span)) / float(res)
    radius_voxels = np.clip(np.round(mean_scale / max(voxel_size, 1e-6)), 1, 2).astype(np.int32)

    gx, gy, gz = coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]
    np.add.at(density, (gx, gy, gz), weights)
    for c_idx in range(3):
        np.add.at(color_grid[..., c_idx], (gx, gy, gz), weights * colors[:, c_idx])

    offsets = [
        (1, 0, 0, 0.5),
        (-1, 0, 0, 0.5),
        (0, 1, 0, 0.5),
        (0, -1, 0, 0.5),
        (0, 0, 1, 0.5),
        (0, 0, -1, 0.5),
    ]
    for dx, dy, dz, decay in offsets:
        nx = np.clip(gx + dx * radius_voxels, 0, res - 1)
        ny = np.clip(gy + dy * radius_voxels, 0, res - 1)
        nz = np.clip(gz + dz * radius_voxels, 0, res - 1)
        w_n = weights * decay
        np.add.at(density, (nx, ny, nz), w_n)
        for c_idx in range(3):
            np.add.at(color_grid[..., c_idx], (nx, ny, nz), w_n * colors[:, c_idx])

    # Separable 3-tap box/Gaussian blur along X, Y, Z to form a smooth implicit field
    for axis in (0, 1, 2):
        padded = np.pad(density, 1, mode="edge")
        if axis == 0:
            density = 0.25 * padded[:-2, 1:-1, 1:-1] + 0.5 * padded[1:-1, 1:-1, 1:-1] + 0.25 * padded[2:, 1:-1, 1:-1]
        elif axis == 1:
            density = 0.25 * padded[1:-1, :-2, 1:-1] + 0.5 * padded[1:-1, 1:-1, 1:-1] + 0.25 * padded[1:-1, 2:, 1:-1]
        else:
            density = 0.25 * padded[1:-1, 1:-1, :-2] + 0.5 * padded[1:-1, 1:-1, 1:-1] + 0.25 * padded[1:-1, 1:-1, 2:]

    max_d = float(np.max(density))
    if max_d <= 1e-8:
        raise ValueError("Gaussian density grid is empty after filtering.")
    norm_density = density / max_d
    thresh = float(np.clip(iso_level, 0.02, 0.8))

    # Try scikit-image marching cubes if installed for sub-voxel interpolation
    try:
        from skimage.measure import marching_cubes

        verts_grid, faces, _, _ = marching_cubes(norm_density, level=thresh)
        v_idx = np.clip(np.round(verts_grid).astype(np.int32), 0, res - 1)
        raw_w = density[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2], None]
        raw_c = color_grid[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2]]
        v_colors = np.where(raw_w > 1e-6, raw_c / np.maximum(raw_w, 1e-6), 0.75)
        verts_world = bmin + (verts_grid / float(res - 1)) * span
        return verts_world.astype(np.float32), faces.astype(np.int32), np.clip(v_colors, 0.0, 1.0).astype(np.float32)
    except Exception:
        pass

    # Pure-NumPy Surface Net / Dual Contouring extraction across active boundary faces
    occ = norm_density >= thresh
    if not np.any(occ):
        occ = norm_density >= (float(np.max(norm_density)) * 0.25)

    # Identify occupied boundary cells and build dual vertices at cell centers shifted by gradient
    occ_indices = np.argwhere(occ)
    cell_id_grid = np.full((res, res, res), -1, dtype=np.int32)
    cell_id_grid[occ_indices[:, 0], occ_indices[:, 1], occ_indices[:, 2]] = np.arange(len(occ_indices), dtype=np.int32)

    verts_world = bmin + (occ_indices.astype(np.float32) / float(res - 1)) * span
    cell_w = density[occ_indices[:, 0], occ_indices[:, 1], occ_indices[:, 2], None]
    cell_c = color_grid[occ_indices[:, 0], occ_indices[:, 1], occ_indices[:, 2]]
    v_colors = np.where(cell_w > 1e-6, cell_c / np.maximum(cell_w, 1e-6), 0.75)

    # Connect surface boundary voxels using 2D surface quads triangulated on each exposed face
    # To avoid duplicate corner vertices, allocate corner vertex table for exposed faces
    corner_dict_keys: list[np.ndarray] = []
    quad_corners: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    # +X / -X faces
    p_x = np.argwhere(occ[:-1, :, :] & ~occ[1:, :, :])
    if len(p_x):
        i, j, k = p_x[:, 0], p_x[:, 1], p_x[:, 2]
        c_col = v_colors[cell_id_grid[i, j, k]]
        quad_corners.append(
            (
                np.column_stack([i + 1, j, k]),
                np.column_stack([i + 1, j + 1, k]),
                np.column_stack([i + 1, j + 1, k + 1]),
                np.column_stack([i + 1, j, k + 1]),
                c_col,
            )
        )
    m_x = np.argwhere(~occ[:-1, :, :] & occ[1:, :, :])
    if len(m_x):
        i, j, k = m_x[:, 0] + 1, m_x[:, 1], m_x[:, 2]
        c_col = v_colors[cell_id_grid[i, j, k]]
        quad_corners.append(
            (
                np.column_stack([i, j, k]),
                np.column_stack([i, j, k + 1]),
                np.column_stack([i, j + 1, k + 1]),
                np.column_stack([i, j + 1, k]),
                c_col,
            )
        )

    # +Y / -Y faces
    p_y = np.argwhere(occ[:, :-1, :] & ~occ[:, 1:, :])
    if len(p_y):
        i, j, k = p_y[:, 0], p_y[:, 1], p_y[:, 2]
        c_col = v_colors[cell_id_grid[i, j, k]]
        quad_corners.append(
            (
                np.column_stack([i, j + 1, k]),
                np.column_stack([i, j + 1, k + 1]),
                np.column_stack([i + 1, j + 1, k + 1]),
                np.column_stack([i + 1, j + 1, k]),
                c_col,
            )
        )
    m_y = np.argwhere(~occ[:, :-1, :] & occ[:, 1:, :])
    if len(m_y):
        i, j, k = m_y[:, 0], m_y[:, 1] + 1, m_y[:, 2]
        c_col = v_colors[cell_id_grid[i, j, k]]
        quad_corners.append(
            (
                np.column_stack([i, j, k]),
                np.column_stack([i + 1, j, k]),
                np.column_stack([i + 1, j, k + 1]),
                np.column_stack([i, j, k + 1]),
                c_col,
            )
        )

    # +Z / -Z faces
    p_z = np.argwhere(occ[:, :, :-1] & ~occ[:, :, 1:])
    if len(p_z):
        i, j, k = p_z[:, 0], p_z[:, 1], p_z[:, 2]
        c_col = v_colors[cell_id_grid[i, j, k]]
        quad_corners.append(
            (
                np.column_stack([i, j, k + 1]),
                np.column_stack([i + 1, j, k + 1]),
                np.column_stack([i + 1, j + 1, k + 1]),
                np.column_stack([i, j + 1, k + 1]),
                c_col,
            )
        )
    m_z = np.argwhere(~occ[:, :, :-1] & occ[:, :, 1:])
    if len(m_z):
        i, j, k = m_z[:, 0], m_z[:, 1], m_z[:, 2] + 1
        c_col = v_colors[cell_id_grid[i, j, k]]
        quad_corners.append(
            (
                np.column_stack([i, j, k]),
                np.column_stack([i, j + 1, k]),
                np.column_stack([i + 1, j + 1, k]),
                np.column_stack([i + 1, j, k]),
                c_col,
            )
        )

    if not quad_corners:
        raise ValueError("No surface boundary faces found in Gaussian density grid.")

    all_c0 = np.vstack([q[0] for q in quad_corners])
    all_c1 = np.vstack([q[1] for q in quad_corners])
    all_c2 = np.vstack([q[2] for q in quad_corners])
    all_c3 = np.vstack([q[3] for q in quad_corners])
    all_qcols = np.vstack([q[4] for q in quad_corners])

    num_quads = len(all_c0)
    stacked_corners = np.vstack([all_c0, all_c1, all_c2, all_c3])
    stride = res + 2
    flat_keys = (
        stacked_corners[:, 0].astype(np.int64) * (stride * stride)
        + stacked_corners[:, 1].astype(np.int64) * stride
        + stacked_corners[:, 2].astype(np.int64)
    )
    unique_keys, unique_idx, inv = np.unique(flat_keys, return_index=True, return_inverse=True)
    unique_coords = stacked_corners[unique_idx].astype(np.float32) - 0.5
    final_verts = bmin + (unique_coords / float(res - 1)) * span

    # Accumulate vertex colors from adjacent quads
    final_cols = np.zeros((len(unique_keys), 3), dtype=np.float32)
    col_counts = np.zeros((len(unique_keys), 1), dtype=np.float32)
    repeated_cols = np.vstack([all_qcols, all_qcols, all_qcols, all_qcols])
    np.add.at(final_cols, inv, repeated_cols)
    np.add.at(col_counts[:, 0], inv, 1.0)
    final_cols = np.clip(final_cols / np.maximum(col_counts, 1.0), 0.0, 1.0)

    v0 = inv[0:num_quads]
    v1 = inv[num_quads : 2 * num_quads]
    v2 = inv[2 * num_quads : 3 * num_quads]
    v3 = inv[3 * num_quads : 4 * num_quads]
    tris_a = np.column_stack([v0, v1, v2])
    tris_b = np.column_stack([v0, v2, v3])
    final_faces = np.vstack([tris_a, tris_b]).astype(np.int32)

    return final_verts.astype(np.float32), final_faces, final_cols.astype(np.float32)


def convert_gaussian_to_mesh(
    gsplat_data: dict[str, Any],
    *,
    method: str = "auto",
    opacity_threshold: float = 0.15,
    grid_resolution: int = 128,
    target_faces: int = 50000,
    smoothing_iterations: int = 3,
) -> dict[str, Any]:
    """
    Convert parsed 3D Gaussian Splat data into a watertight/smoothed polygon mesh
    with vertex colors and normals.
    """
    positions = np.asarray(gsplat_data["positions"], dtype=np.float32)
    colors = np.asarray(gsplat_data["colors"], dtype=np.float32)
    opacities = np.asarray(gsplat_data["opacities"], dtype=np.float32)
    scales = np.asarray(gsplat_data["scales"], dtype=np.float32)

    valid = np.isfinite(positions).all(axis=-1) & (opacities >= float(opacity_threshold))
    if np.count_nonzero(valid) < 32:
        # Relax opacity threshold if too few points survive
        valid = np.isfinite(positions).all(axis=-1)

    pos = positions[valid]
    col = colors[valid]
    op = opacities[valid]
    sc = scales[valid]

    if len(pos) < 4:
        raise ValueError("Not enough valid Gaussians to reconstruct a 3D mesh.")

    # Optional Open3D Screened Poisson / Ball Pivoting path
    if method in ("auto", "open3d_poisson", "open3d_ball_pivoting"):
        try:
            import open3d as o3d  # type: ignore[import-not-found]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pos.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(30)

            if method == "open3d_ball_pivoting":
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = float(np.mean(distances)) if len(distances) else 0.02
                radii = [avg_dist * 1.2, avg_dist * 2.4]
                mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
            else:
                depth = max(6, min(10, int(round(math.log2(max(grid_resolution, 64))))))
                mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
                dens_arr = np.asarray(densities)
                if len(dens_arr):
                    mesh_o3d.remove_vertices_by_mask(dens_arr < np.quantile(dens_arr, 0.04))

            if smoothing_iterations > 0:
                mesh_o3d = mesh_o3d.filter_smooth_taubin(number_of_iterations=int(smoothing_iterations))
            if target_faces > 0 and len(mesh_o3d.triangles) > target_faces:
                mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
            mesh_o3d.compute_vertex_normals()

            v_out = np.asarray(mesh_o3d.vertices, dtype=np.float32)
            f_out = np.asarray(mesh_o3d.triangles, dtype=np.int32)
            c_out = (
                np.asarray(mesh_o3d.vertex_colors, dtype=np.float32)
                if mesh_o3d.has_vertex_colors()
                else np.full((len(v_out), 3), 0.75, dtype=np.float32)
            )
            n_out = np.asarray(mesh_o3d.vertex_normals, dtype=np.float32)
            if len(v_out) > 0 and len(f_out) > 0:
                return {
                    "vertices": v_out,
                    "faces": f_out,
                    "vertex_colors": np.clip(c_out, 0.0, 1.0),
                    "normals": n_out,
                    "method_used": "open3d_ball_pivoting" if method == "open3d_ball_pivoting" else "open3d_poisson",
                }
        except Exception:
            if method in ("open3d_poisson", "open3d_ball_pivoting"):
                # Fall through gracefully to built-in density grid extractor so workflows never crash
                pass

    verts, faces, v_cols = _extract_surface_from_gaussian_grid(
        pos,
        col,
        op,
        sc,
        grid_resolution=grid_resolution,
        iso_level=max(0.08, float(opacity_threshold) * 0.8),
    )
    if smoothing_iterations > 0:
        verts = laplacian_smooth_mesh(verts, faces, iterations=int(smoothing_iterations))
    if target_faces > 0 and len(faces) > target_faces:
        verts, faces, v_cols = decimate_mesh_vertex_clustering(verts, faces, v_cols, int(target_faces))
    normals = compute_vertex_normals(verts, faces)

    return {
        "vertices": verts,
        "faces": faces,
        "vertex_colors": v_cols,
        "normals": normals,
        "method_used": "gaussian_density_surface",
    }


# ---------------------------------------------------------------------------
# Fast Vectorized Software Renderer for Gaussian Splats & 3D Meshes
# ---------------------------------------------------------------------------


def render_3d_view(
    *,
    positions: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    azimuth_deg: float = 30.0,
    elevation_deg: float = 20.0,
    resolution: int = 512,
    bg_color: tuple[float, float, float] = (0.12, 0.13, 0.16),
    point_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render a single (H, W, 3) RGB image and (H, W) alpha mask from either a 3D point/Gaussian
    cloud or a triangle mesh at the specified camera azimuth and elevation.
    """
    res = max(64, min(2048, int(resolution)))
    img = np.full((res, res, 3), bg_color, dtype=np.float32)
    mask = np.zeros((res, res), dtype=np.float32)

    pts = np.asarray(positions, dtype=np.float32)
    if len(pts) == 0:
        return img, mask

    cols = np.clip(np.asarray(colors, dtype=np.float32)[:, :3], 0.0, 1.0)

    # Center and normalize coordinates into [-0.82, 0.82]
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    centered = pts - center
    max_extent = float(np.max(np.linalg.norm(centered, axis=-1)))
    if max_extent > 1e-6:
        centered = centered / max_extent * 0.80

    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    cos_a, sin_a = math.cos(az), math.sin(az)
    cos_e, sin_e = math.cos(el), math.sin(el)

    rot_y = np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_e, -sin_e],
            [0.0, sin_e, cos_e],
        ],
        dtype=np.float32,
    )
    rot_mat = rot_x @ rot_y
    cam_pts = centered @ rot_mat.T

    # Simple perspective projection
    cam_dist = 2.4
    z = cam_dist - cam_pts[:, 2]
    z = np.maximum(z, 0.2)
    focal = 1.65
    px = ((cam_pts[:, 0] * focal / z) * 0.5 + 0.5) * (res - 1)
    py = ((-cam_pts[:, 1] * focal / z) * 0.5 + 0.5) * (res - 1)

    if faces is not None and len(faces) > 0:
        tris = np.asarray(faces, dtype=np.int32)
        if normals is None or len(normals) != len(pts):
            normals = compute_vertex_normals(centered, tris)
        rot_normals = normals @ rot_mat.T
        light_dir = np.array([0.45, 0.65, 0.61], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        ndotl = np.clip(rot_normals @ light_dir, 0.0, 1.0)
        shade = 0.32 + 0.68 * ndotl[:, None]
        shaded_cols = np.clip(cols * shade, 0.0, 1.0)

        # Sample face centroids + vertices + edge midpoints for fast dense rasterization
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        sample_px = np.concatenate(
            [
                px,
                (px[v0] + px[v1] + px[v2]) / 3.0,
                0.5 * (px[v0] + px[v1]),
                0.5 * (px[v1] + px[v2]),
                0.5 * (px[v2] + px[v0]),
            ]
        )
        sample_py = np.concatenate(
            [
                py,
                (py[v0] + py[v1] + py[v2]) / 3.0,
                0.5 * (py[v0] + py[v1]),
                0.5 * (py[v1] + py[v2]),
                0.5 * (py[v2] + py[v0]),
            ]
        )
        sample_z = np.concatenate(
            [
                z,
                (z[v0] + z[v1] + z[v2]) / 3.0,
                0.5 * (z[v0] + z[v1]),
                0.5 * (z[v1] + z[v2]),
                0.5 * (z[v2] + z[v0]),
            ]
        )
        tri_col = (shaded_cols[v0] + shaded_cols[v1] + shaded_cols[v2]) / 3.0
        sample_col = np.vstack(
            [
                shaded_cols,
                tri_col,
                0.5 * (shaded_cols[v0] + shaded_cols[v1]),
                0.5 * (shaded_cols[v1] + shaded_cols[v2]),
                0.5 * (shaded_cols[v2] + shaded_cols[v0]),
            ]
        )
        sample_op = np.ones(len(sample_z), dtype=np.float32)
    else:
        sample_px = px
        sample_py = py
        sample_z = z
        sample_col = cols
        sample_op = (
            np.clip(np.asarray(opacities, dtype=np.float32), 0.0, 1.0)
            if opacities is not None
            else np.ones(len(z), dtype=np.float32)
        )

    keep = sample_op >= 0.08
    sample_px = sample_px[keep]
    sample_py = sample_py[keep]
    sample_z = sample_z[keep]
    sample_col = sample_col[keep]
    sample_op = sample_op[keep]

    # Sort back-to-front (painter's order)
    order = np.argsort(-sample_z)
    ix = np.round(sample_px[order]).astype(np.int32)
    iy = np.round(sample_py[order]).astype(np.int32)
    scol = sample_col[order]
    sop = sample_op[order]

    valid = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
    ix, iy, scol, sop = ix[valid], iy[valid], scol[valid], sop[valid]

    rad = max(1, int(point_radius))
    offsets = [(0, 0)]
    if rad >= 2:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)]
    for dx, dy in offsets:
        nx = np.clip(ix + dx, 0, res - 1)
        ny = np.clip(iy + dy, 0, res - 1)
        img[ny, nx] = scol
        mask[ny, nx] = np.maximum(mask[ny, nx], sop)

    return np.clip(img, 0.0, 1.0), np.clip(mask, 0.0, 1.0)


def render_4view_contact_sheet(
    *,
    positions: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    opacities: np.ndarray | None = None,
    tile_resolution: int = 256,
) -> Any:
    """Render a 2x2 multi-angle preview contact sheet (Front, Right, Back, Iso) as a ComfyUI IMAGE tensor."""
    angles = [(0.0, 15.0), (90.0, 15.0), (180.0, 15.0), (35.0, 28.0)]
    tiles = []
    for az, el in angles:
        rgb, _ = render_3d_view(
            positions=positions,
            colors=colors,
            faces=faces,
            normals=normals,
            opacities=opacities,
            azimuth_deg=az,
            elevation_deg=el,
            resolution=tile_resolution,
        )
        tiles.append(rgb)
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
    grid = np.concatenate([top, bottom], axis=0)[None, ...]
    return numpy_batch_to_tensor(grid)


def _extract_trimesh_arrays(mesh_obj: Any) -> dict[str, Any]:
    """Extract vertices, faces, vertex_colors, and normals from a trimesh.Trimesh or Scene object."""
    if hasattr(mesh_obj, "dump"):
        # trimesh.Scene -> concatenate geometries
        geoms = mesh_obj.dump(concatenate=True)
        mesh_obj = geoms

    verts = np.asarray(getattr(mesh_obj, "vertices", np.zeros((0, 3))), dtype=np.float32)
    faces = np.asarray(getattr(mesh_obj, "faces", np.zeros((0, 3))), dtype=np.int32)
    cols = None
    visual = getattr(mesh_obj, "visual", None)
    if visual is not None:
        try:
            if hasattr(visual, "to_color") and getattr(visual, "kind", None) == "texture":
                color_vis = visual.to_color()
                vc = np.asarray(color_vis.vertex_colors, dtype=np.float32)
            else:
                vc = np.asarray(visual.vertex_colors, dtype=np.float32)
            if vc.ndim == 2 and len(vc) == len(verts):
                cols = vc[:, :3] / 255.0 if np.max(vc) > 1.0 else vc[:, :3]
        except Exception:
            cols = None

    if cols is None or len(cols) != len(verts):
        cols = np.full((len(verts), 3), 0.78, dtype=np.float32)
    normals = compute_vertex_normals(verts, faces)
    return {
        "vertices": verts,
        "faces": faces,
        "vertex_colors": np.clip(cols, 0.0, 1.0),
        "normals": normals,
    }


# ---------------------------------------------------------------------------
# Unified 3D Model Manager (RTX 3070 Ti 8GB-16GB VRAM Safe)
# ---------------------------------------------------------------------------


class Ziggart3DError(RuntimeError):
    pass


_BACKEND_REPOS: dict[str, tuple[str, str]] = {
    "idol": ("https://github.com/speridlabs/idol-sdk.git", "idol-sdk"),
    "spar3d": ("https://github.com/Stability-AI/stable-point-aware-3d.git", "stable-point-aware-3d"),
    "hy3dgen": ("https://github.com/Tencent/Hunyuan3D-2.git", "Hunyuan3D-2"),
}


def _ensure_backend_installed(package_name: str, auto_install: bool = True) -> bool:
    """
    Ensure that a 3D backend SDK ('idol', 'spar3d', or 'hy3dgen') is importable.
    If missing and auto_install=True, automatically clones the official SDK into
    Ziggart-Nodes/.cache/vendor_3d/<repo> and installs it into the active ComfyUI Python env.
    """
    if package_name in sys.modules and sys.modules[package_name] is not None:
        return True

    vendor_root = Path(__file__).resolve().parent / ".cache" / "vendor_3d"
    if package_name in _BACKEND_REPOS:
        _, folder_name = _BACKEND_REPOS[package_name]
        local_repo = vendor_root / folder_name
        if local_repo.is_dir() and str(local_repo) not in sys.path:
            sys.path.insert(0, str(local_repo))

    try:
        import importlib.util

        if importlib.util.find_spec(package_name) is not None:
            return True
    except Exception:
        pass

    if not auto_install or package_name not in _BACKEND_REPOS:
        return False

    repo_url, folder_name = _BACKEND_REPOS[package_name]
    vendor_root.mkdir(parents=True, exist_ok=True)
    target_dir = vendor_root / folder_name

    if not (target_dir / ".git").exists():
        clone_cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
        subprocess.run(clone_cmd, check=False, capture_output=True, text=True)

    if target_dir.is_dir():
        if str(target_dir) not in sys.path:
            sys.path.insert(0, str(target_dir))
        pip_cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", str(target_dir)]
        res = subprocess.run(pip_cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(target_dir)], check=False, capture_output=True, text=True)

    try:
        import importlib

        importlib.invalidate_caches()
        return importlib.util.find_spec(package_name) is not None
    except Exception:
        return False



class Ziggart3DModelManager:
    """
    Manages in-process 3D generation pipelines (Idol, SPAR3D, Hunyuan3D-2mini)
    and guarantees that at most one heavy 3D pipeline is resident in GPU VRAM
    at any time on an RTX 3070 Ti.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_kind: str | None = None
        self._active_key: tuple[Any, ...] | None = None
        self._pipeline: Any = None

    def unload(self) -> None:
        with self._lock:
            pipe = self._pipeline
            self._pipeline = None
            self._active_kind = None
            self._active_key = None
            if pipe is not None:
                try:
                    if hasattr(pipe, "to"):
                        pipe.to("cpu")
                    for sub_attr in ("denoiser", "decoder", "decoder_gs", "background_remover", "image_encoder"):
                        sub_mod = getattr(pipe, sub_attr, None)
                        if sub_mod is not None and hasattr(sub_mod, "to"):
                            sub_mod.to("cpu")
                except Exception:
                    pass
                del pipe
            empty_accelerator_cache()
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _prepare_vram(self, kind: str, key: tuple[Any, ...], unload_comfy_models: bool) -> bool:
        """Returns True if the requested pipeline is already loaded."""
        if self._pipeline is not None and self._active_kind == kind and self._active_key == key:
            return True
        self.unload()
        if unload_comfy_models:
            release_comfy_vram()
        return False

    def run_idol(
        self,
        *,
        pil_image: Image.Image,
        mode: str,
        num_gaussians: int,
        precision: str,
        seed: int,
        model_id_or_path: str,
        hf_token: str,
        unload_comfy_models: bool,
        free_vram_after: bool,
        output_ply_path: Path,
    ) -> tuple[Path, Image.Image]:
        with self._lock:
            if hf_token and hf_token.strip():
                os.environ["HF_TOKEN"] = hf_token.strip()
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()

            if unload_comfy_models:
                release_comfy_vram()

            key = ("idol", model_id_or_path, precision)
            if self._active_kind != "idol" or self._active_key != key:
                self.unload()

            _ensure_backend_installed("idol", auto_install=True)

            # Guard against broken/stub flash_attn packages on Windows where
            # importlib.util.find_spec("flash_attn") is not None but flash_attn_func is None
            has_working_flash_attn = False
            try:
                import flash_attn  # type: ignore[import-not-found]

                has_working_flash_attn = callable(getattr(flash_attn, "flash_attn_func", None)) and callable(
                    getattr(flash_attn, "flash_attn_qkvpacked_func", None)
                )
            except Exception:
                has_working_flash_attn = False

            if not has_working_flash_attn:
                os.environ["ATTN_BACKEND"] = "sdpa"

            # Strategy 1: Direct Python SDK (`idol` package from `speridlabs/idol-sdk`)
            try:
                import idol  # type: ignore[import-not-found]

                try:
                    from idol.models import layers as idol_layers  # type: ignore[import-not-found]

                    if not has_working_flash_attn and getattr(idol_layers, "ATTN_BACKEND", None) == "flash_attn":
                        idol_layers.ATTN_BACKEND = "sdpa"
                except Exception:
                    pass

                if self._pipeline is None:
                    loader_fn = (
                        getattr(idol, "IdolPipeline", None)
                        or getattr(idol, "load_pipeline", None)
                        or getattr(idol, "Idol", None)
                    )
                    if loader_fn is not None:
                        dinov3_dtype = "bf16" if precision in ("bf16", "fp16") else "fp32"
                        if hasattr(loader_fn, "from_pretrained"):
                            try:
                                self._pipeline = loader_fn.from_pretrained(
                                    model_id_or_path,
                                    dinov3_dtype=dinov3_dtype,
                                )
                            except TypeError:
                                self._pipeline = loader_fn.from_pretrained(model_id_or_path)
                        else:
                            self._pipeline = loader_fn(model_id_or_path)
                        self._active_kind = "idol"
                        self._active_key = key

                if self._pipeline is not None:
                    result = self._pipeline(
                        pil_image,
                        mode=mode,
                        num_gaussians=int(num_gaussians),
                        seed=int(seed),
                    )
                    if hasattr(result, "gaussian") and hasattr(result.gaussian, "save_ply"):
                        result.gaussian.save_ply(str(output_ply_path))
                    elif hasattr(result, "save_ply"):
                        result.save_ply(str(output_ply_path))
                    elif isinstance(result, dict) and "ply_path" in result:
                        shutil.copyfile(result["ply_path"], output_ply_path)
                    elif isinstance(result, (str, Path)) and Path(result).is_file():
                        shutil.copyfile(result, output_ply_path)

                    if isinstance(result, dict):
                        preprocessed = result.get("preprocessed_image") or result.get("input_image") or pil_image
                    else:
                        preprocessed = (
                            getattr(result, "preprocessed_image", None)
                            or getattr(result, "input_image", None)
                            or pil_image
                        )
                    if free_vram_after:
                        self.unload()
                    return output_ply_path, preprocessed
            except ImportError:
                pass
            except Exception as exc:
                if free_vram_after:
                    self.unload()
                raise Ziggart3DError(f"Idol Python pipeline failed: {exc}") from exc

            # Strategy 2: Official CLI entrypoint `python -m idol.sample`
            self.unload()
            try:
                import importlib.util

                has_idol_sample = importlib.util.find_spec("idol") is not None
            except Exception:
                has_idol_sample = False

            if not has_idol_sample:
                raise Ziggart3DError(
                    "speridlabs/idol-sdk is not installed in this Python environment. "
                    "Install it with:\n"
                    "  git clone https://github.com/speridlabs/idol-sdk.git && pip install -e idol-sdk\n"
                    "And accept the gated Hugging Face terms for facebook/dinov3-vith16plus-pretrain-lvd1689m "
                    "and black-forest-labs/FLUX.2-dev."
                )

            with tempfile.TemporaryDirectory(prefix="ziggart_idol_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                in_img_path = tmp_path / "input.png"
                out_dir = tmp_path / "out"
                out_dir.mkdir(parents=True, exist_ok=True)
                pil_image.save(in_img_path, format="PNG")

                cmd = [
                    sys.executable,
                    "-m",
                    "idol.sample",
                    "--image",
                    str(in_img_path),
                    "--mode",
                    str(mode),
                    "--out",
                    str(out_dir),
                ]
                env = os.environ.copy()
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    err_msg = (proc.stderr or proc.stdout or "Unknown error").strip()
                    raise Ziggart3DError(f"idol.sample failed (exit {proc.returncode}): {err_msg}")

                ply_candidates = sorted(out_dir.rglob("*.ply"))
                if not ply_candidates:
                    ply_candidates = sorted(tmp_path.rglob("*.ply"))
                if not ply_candidates:
                    raise Ziggart3DError("idol.sample finished but did not produce a .ply file.")

                shutil.copyfile(ply_candidates[0], output_ply_path)
                input_pngs = sorted(out_dir.rglob("*_input.png"))
                preprocessed = Image.open(input_pngs[0]).convert("RGBA") if input_pngs else pil_image
                return output_ply_path, preprocessed

    def run_spar3d(
        self,
        *,
        pil_image: Image.Image,
        model_id_or_path: str,
        low_vram_mode: bool,
        bake_resolution: int,
        remesh_option: str,
        target_vertex_count: int,
        hf_token: str,
        unload_comfy_models: bool,
        free_vram_after: bool,
        output_glb_path: Path,
        output_ply_path: Path,
    ) -> tuple[Path, Path, dict[str, Any]]:
        with self._lock:
            if hf_token and hf_token.strip():
                os.environ["HF_TOKEN"] = hf_token.strip()
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()
            if low_vram_mode:
                os.environ["SPAR3D_LOW_VRAM"] = "1"

            key = ("spar3d", model_id_or_path, bool(low_vram_mode))
            already_loaded = self._prepare_vram("spar3d", key, unload_comfy_models)

            try:
                if not already_loaded:
                    _ensure_backend_installed("spar3d", auto_install=True)
                    spar3d_cls = None
                    for mod_name, attr_name in (
                        ("spar3d.models.mesh", "SPAR3D"),
                        ("spar3d.system", "SPAR3D"),
                        ("spar3d.systems.spar3d", "SPAR3DSystem"),
                        ("spar3d", "SPAR3D"),
                    ):
                        try:
                            mod = __import__(mod_name, fromlist=[attr_name])
                            spar3d_cls = getattr(mod, attr_name, None)
                            if spar3d_cls is not None:
                                break
                        except ImportError:
                            continue

                    if spar3d_cls is None:
                        raise Ziggart3DError(
                            "SPAR3D is not installed in this Python environment. "
                            "Install it from https://github.com/Stability-AI/stable-point-aware-3d."
                        )

                    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
                    try:
                        model = spar3d_cls.from_pretrained(
                            model_id_or_path,
                            config_name="config.yaml",
                            weight_name="model.safetensors",
                            low_vram_mode=bool(low_vram_mode),
                        )
                    except TypeError:
                        model = spar3d_cls.from_pretrained(model_id_or_path)

                    if hasattr(model, "to"):
                        model = model.to(device)
                    if hasattr(model, "eval"):
                        model.eval()
                    self._pipeline = model
                    self._active_kind = "spar3d"
                    self._active_key = key

                model = self._pipeline
                rgba_image = remove_background_fallback(pil_image)

                point_cloud = None
                if hasattr(model, "run_image"):
                    res = model.run_image(
                        rgba_image,
                        bake_resolution=int(bake_resolution),
                        remesh=remesh_option if remesh_option != "none" else "none",
                        vertex_count=int(target_vertex_count) if target_vertex_count > 0 else -1,
                        return_points=True,
                    )
                    if isinstance(res, tuple) and len(res) >= 2:
                        mesh_obj, point_cloud = res[0], res[1]
                    else:
                        mesh_obj = res
                elif hasattr(model, "generate_mesh"):
                    mesh_obj = model.generate_mesh(rgba_image)
                else:
                    mesh_obj = model(rgba_image)

                mesh_arrays = _extract_trimesh_arrays(mesh_obj)
                if hasattr(mesh_obj, "export"):
                    mesh_obj.export(str(output_glb_path))
                else:
                    export_mesh_file(
                        output_glb_path,
                        mesh_arrays["vertices"],
                        mesh_arrays["faces"],
                        mesh_arrays["vertex_colors"],
                        mesh_arrays["normals"],
                    )

                if point_cloud is not None and hasattr(point_cloud, "export"):
                    point_cloud.export(str(output_ply_path))
                else:
                    save_gaussian_ply(
                        output_ply_path,
                        mesh_arrays["vertices"],
                        mesh_arrays["vertex_colors"],
                    )

                if free_vram_after:
                    self.unload()

                mesh_arrays["path"] = str(output_glb_path)
                return output_glb_path, output_ply_path, mesh_arrays
            except Ziggart3DError:
                if free_vram_after:
                    self.unload()
                raise
            except Exception as exc:
                if free_vram_after:
                    self.unload()
                raise Ziggart3DError(f"SPAR3D inference failed: {exc}") from exc

    def run_hunyuan3d(
        self,
        *,
        pil_image: Image.Image,
        model_variant: str,
        custom_model_path: str,
        steps: int,
        guidance_scale: float,
        octree_resolution: int,
        num_chunks: int,
        generate_texture: bool,
        max_faces: int,
        seed: int,
        unload_comfy_models: bool,
        free_vram_after: bool,
        output_glb_path: Path,
    ) -> tuple[Path, Image.Image, dict[str, Any]]:
        with self._lock:
            if custom_model_path and custom_model_path.strip():
                repo_id = custom_model_path.strip()
                subfolder = None
            elif "Turbo" in model_variant:
                repo_id = "tencent/Hunyuan3D-2mini"
                subfolder = "hunyuan3d-dit-v2-mini-turbo"
            elif "2mini" in model_variant:
                repo_id = "tencent/Hunyuan3D-2mini"
                subfolder = "hunyuan3d-dit-v2-mini"
            else:
                repo_id = "tencent/Hunyuan3D-2"
                subfolder = "hunyuan3d-dit-v2-0"

            rgba_image = remove_background_fallback(pil_image)

            key = ("hunyuan3d_shape", repo_id, subfolder)
            already_loaded = self._prepare_vram("hunyuan3d_shape", key, unload_comfy_models)

            _ensure_backend_installed("hy3dgen", auto_install=True)
            try:
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import-not-found]
            except ImportError as exc:
                raise Ziggart3DError(
                    "hy3dgen (Hunyuan3D-2) is not installed in this Python environment. "
                    "Install it from https://github.com/Tencent/Hunyuan3D-2 (pip install -e .)."
                ) from exc

            try:
                device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
                dtype = torch.float16 if (torch is not None and device == "cuda") else None

                if not already_loaded:
                    load_kwargs: dict[str, Any] = {"device": device}
                    if dtype is not None:
                        load_kwargs["torch_dtype"] = dtype
                    if subfolder:
                        load_kwargs["subfolder"] = subfolder
                    try:
                        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id, **load_kwargs)
                    except TypeError:
                        load_kwargs.pop("subfolder", None)
                        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id, **load_kwargs)
                    self._active_kind = "hunyuan3d_shape"
                    self._active_key = key

                gen = None
                if torch is not None and seed >= 0:
                    gen = torch.manual_seed(int(seed))

                shape_out = self._pipeline(
                    image=rgba_image,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    octree_resolution=int(octree_resolution),
                    num_chunks=int(num_chunks),
                    generator=gen,
                    output_type="trimesh",
                )
                mesh_obj = shape_out[0] if isinstance(shape_out, (list, tuple)) else shape_out

                # Optional face simplification before texturing (keeps Paint VRAM low on RTX 3070 Ti)
                try:
                    from hy3dgen.shapegen import FaceReducer  # type: ignore[import-not-found]

                    if max_faces > 0:
                        mesh_obj = FaceReducer()(mesh_obj, max_facenum=int(max_faces))
                except Exception:
                    pass

                if generate_texture:
                    # Sequential VRAM strategy: MUST unload Shape pipeline before loading Paint pipeline
                    self.unload()
                    try:
                        from hy3dgen.texgen import Hunyuan3DPaintPipeline  # type: ignore[import-not-found]

                        paint_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
                        if hasattr(paint_pipe, "enable_model_cpu_offload"):
                            paint_pipe.enable_model_cpu_offload()
                        mesh_obj = paint_pipe(mesh=mesh_obj, image=rgba_image)
                        del paint_pipe
                        empty_accelerator_cache()
                    except ImportError as tex_exc:
                        raise Ziggart3DError(
                            f"Hunyuan3DPaintPipeline could not be imported for texture generation: {tex_exc}"
                        ) from tex_exc

                mesh_arrays = _extract_trimesh_arrays(mesh_obj)
                if hasattr(mesh_obj, "export"):
                    mesh_obj.export(str(output_glb_path))
                else:
                    export_mesh_file(
                        output_glb_path,
                        mesh_arrays["vertices"],
                        mesh_arrays["faces"],
                        mesh_arrays["vertex_colors"],
                        mesh_arrays["normals"],
                    )

                if free_vram_after:
                    self.unload()

                mesh_arrays["path"] = str(output_glb_path)
                return output_glb_path, rgba_image, mesh_arrays
            except Ziggart3DError:
                if free_vram_after:
                    self.unload()
                raise
            except Exception as exc:
                if free_vram_after:
                    self.unload()
                raise Ziggart3DError(f"Hunyuan3D inference failed: {exc}") from exc


THREED_MANAGER = Ziggart3DModelManager()
atexit.register(THREED_MANAGER.unload)


# ---------------------------------------------------------------------------
# ComfyUI Node Definitions
# ---------------------------------------------------------------------------


class ZiggartIdolImageToGaussian:
    """Single-image to 3D Gaussian Splat (.ply) using speridlabs/idol (~7.2 GB VRAM)."""

    DESCRIPTION = (
        "Generates a 3D Gaussian Splat (.ply) from a single image using speridlabs/idol. "
        "Optimized for 8GB-16GB GPUs (like RTX 3070 Ti) by automatically evicting ComfyUI "
        "models before inference and releasing VRAM afterward."
    )
    CATEGORY = "Ziggart/3D"
    FUNCTION = "generate_gaussian"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "ZIGGART_GSPLAT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ply_path", "gsplat_data", "preprocessed_image", "preview_image")
    OUTPUT_TOOLTIPS = (
        "Absolute path to the generated 3D Gaussian Splat (.ply) file.",
        "In-memory Gaussian Splat data for ZiggartGaussianToMesh or Ziggart3DTurntableRenderer.",
        "Preprocessed / background-stripped input image.",
        "2x2 multi-angle preview contact sheet of the 3D Gaussian Splat.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input single image (photo, render, or generated art)."}),
                "mode": (
                    ["synthetic", "real"],
                    {
                        "default": "synthetic",
                        "tooltip": "Use 'synthetic' for AI renders/illustrations and 'real' for real photographs.",
                    },
                ),
                "num_gaussians": (
                    ["262144", "524288", "1048576", "131072", "65536"],
                    {
                        "default": "262144",
                        "tooltip": "Number of 3D Gaussians to generate (262144 is the fast 7.2GB VRAM default).",
                    },
                ),
                "precision": (
                    ["bf16", "fp16", "fp32"],
                    {"default": "bf16", "tooltip": "Inference precision (bf16/fp16 recommended for RTX 3070 Ti)."},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, " step": 1}),
                "filename_prefix": ("STRING", {"default": "ziggart_idol_splat"}),
                "unload_comfy_models": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Evict loaded diffusion/VLM models from VRAM before running Idol.",
                    },
                ),
                "free_vram_after": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload the Idol pipeline from GPU VRAM immediately after generation.",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional foreground mask to apply before 3D reconstruction."}),
                "model_id_or_path": (
                    "STRING",
                    {
                        "default": "speridlabs/idol",
                        "tooltip": "Hugging Face repo ID or local directory path for speridlabs/idol.",
                    },
                ),
                "hf_token": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional Hugging Face token for gated DINOv3 / FLUX.2-dev VAE weights.",
                    },
                ),
            },
        }

    def generate_gaussian(
        self,
        image: Any,
        mode: str = "synthetic",
        num_gaussians: str = "262144",
        precision: str = "bf16",
        seed: int = 42,
        filename_prefix: str = "ziggart_idol_splat",
        unload_comfy_models: bool = True,
        free_vram_after: bool = True,
        mask: Any = None,
        model_id_or_path: str = "speridlabs/idol",
        hf_token: str = "",
    ) -> tuple[str, dict[str, Any], Any, Any]:
        pil_img = tensor_to_pil(image, mask=mask)
        out_ply = _next_output_path(filename_prefix, "ply")

        ply_path, preprocessed_pil = THREED_MANAGER.run_idol(
            pil_image=pil_img,
            mode=mode,
            num_gaussians=int(num_gaussians),
            precision=precision,
            seed=int(seed),
            model_id_or_path=model_id_or_path.strip() or "speridlabs/idol",
            hf_token=hf_token,
            unload_comfy_models=bool(unload_comfy_models),
            free_vram_after=bool(free_vram_after),
            output_ply_path=out_ply,
        )

        gsplat_data = load_gaussian_ply(ply_path)
        preview_tensor = render_4view_contact_sheet(
            positions=gsplat_data["positions"],
            colors=gsplat_data["colors"],
            opacities=gsplat_data["opacities"],
        )
        preprocessed_tensor = pil_to_tensor(preprocessed_pil)
        return (str(ply_path), gsplat_data, preprocessed_tensor, preview_tensor)


class ZiggartGaussianToMesh:
    """Convert a 3D Gaussian Splat (.ply or ZIGGART_GSPLAT) into a polygon 3D Mesh (.glb / .obj / .ply)."""

    DESCRIPTION = (
        "Converts an Idol 3D Gaussian Splat (.ply) into a watertight/smoothed polygon 3D Mesh "
        "(.glb, .obj, or .ply) with baked vertex colors and normals. Uses Open3D Poisson/Ball-Pivoting "
        "when available, or a built-in NumPy Gaussian density field surface extractor."
    )
    CATEGORY = "Ziggart/3D"
    FUNCTION = "convert_to_mesh"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "ZIGGART_MESH", "IMAGE")
    RETURN_NAMES = ("mesh_path", "mesh_data", "preview_image")
    OUTPUT_TOOLTIPS = (
        "Absolute path to the exported 3D mesh (.glb, .obj, or .ply).",
        "In-memory mesh dictionary (vertices, faces, vertex_colors, normals).",
        "2x2 shaded multi-angle preview render of the converted mesh.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "method": (
                    ["auto", "gaussian_density_surface", "open3d_poisson", "open3d_ball_pivoting"],
                    {
                        "default": "auto",
                        "tooltip": "Surface reconstruction backend ('auto' uses Open3D Poisson if installed, else Gaussian density surface extraction).",
                    },
                ),
                "export_format": (
                    ["glb", "obj", "ply"],
                    {"default": "glb", "tooltip": "Output 3D mesh file format."},
                ),
                "opacity_threshold": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.01,
                        "max": 0.95,
                        "step": 0.02,
                        "tooltip": "Filter out low-opacity floater Gaussians before surface extraction.",
                    },
                ),
                "grid_resolution": (
                    "INT",
                    {
                        "default": 112,
                        "min": 32,
                        "max": 256,
                        "step": 16,
                        "tooltip": "Voxel/Octree resolution for surface reconstruction.",
                    },
                ),
                "target_faces": (
                    "INT",
                    {
                        "default": 50000,
                        "min": 1000,
                        "max": 500000,
                        "step": 5000,
                        "tooltip": "Target triangle count after mesh decimation.",
                    },
                ),
                "smoothing_iterations": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 15,
                        "step": 1,
                        "tooltip": "Taubin/Laplacian smoothing iterations to remove surface staircase noise.",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "ziggart_idol_mesh"}),
            },
            "optional": {
                "gsplat_data": ("ZIGGART_GSPLAT", {"tooltip": "Gaussian splat data from ZiggartIdolImageToGaussian."}),
                "ply_path": ("STRING", {"default": "", "tooltip": "Or path to an existing .ply Gaussian Splat file."}),
            },
        }

    def convert_to_mesh(
        self,
        method: str = "auto",
        export_format: str = "glb",
        opacity_threshold: float = 0.15,
        grid_resolution: int = 112,
        target_faces: int = 50000,
        smoothing_iterations: int = 3,
        filename_prefix: str = "ziggart_idol_mesh",
        gsplat_data: dict[str, Any] | None = None,
        ply_path: str = "",
    ) -> tuple[str, dict[str, Any], Any]:
        if gsplat_data is None:
            if not ply_path or not str(ply_path).strip():
                raise ValueError("Provide either 'gsplat_data' or a valid 'ply_path' to convert to a mesh.")
            gsplat_data = load_gaussian_ply(ply_path.strip())

        mesh_data = convert_gaussian_to_mesh(
            gsplat_data,
            method=method,
            opacity_threshold=float(opacity_threshold),
            grid_resolution=int(grid_resolution),
            target_faces=int(target_faces),
            smoothing_iterations=int(smoothing_iterations),
        )

        out_path = _next_output_path(filename_prefix, export_format)
        export_mesh_file(
            out_path,
            mesh_data["vertices"],
            mesh_data["faces"],
            mesh_data["vertex_colors"],
            mesh_data["normals"],
        )
        mesh_data["path"] = str(out_path)

        preview_tensor = render_4view_contact_sheet(
            positions=mesh_data["vertices"],
            colors=mesh_data["vertex_colors"],
            faces=mesh_data["faces"],
            normals=mesh_data["normals"],
        )
        return (str(out_path), mesh_data, preview_tensor)


class ZiggartSPAR3DImageToMesh:
    """Single-image to UV-unwrapped textured 3D Mesh (.glb) using Stability AI's SPAR3D (~7 GB VRAM)."""

    DESCRIPTION = (
        "Generates a UV-unwrapped, textured 3D Mesh (.glb) and intermediate point cloud (.ply) "
        "in ~1-2 seconds using Stability AI's SPAR3D. Enables low_vram_mode by default (~7 GB VRAM) "
        "so it runs smoothly on an RTX 3070 Ti."
    )
    CATEGORY = "Ziggart/3D"
    FUNCTION = "generate_mesh"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "ZIGGART_MESH", "STRING", "IMAGE")
    RETURN_NAMES = ("mesh_path", "mesh_data", "point_cloud_ply_path", "preview_image")
    OUTPUT_TOOLTIPS = (
        "Absolute path to the generated textured .glb mesh.",
        "In-memory mesh dictionary for Ziggart3DTurntableRenderer.",
        "Absolute path to SPAR3D's intermediate point cloud (.ply).",
        "2x2 shaded multi-angle preview render of the generated mesh.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to convert into a textured 3D mesh."}),
                "low_vram_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reduces peak VRAM from ~10.5 GB to ~7.0 GB (recommended for RTX 3070 Ti).",
                    },
                ),
                "bake_resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 256,
                        "tooltip": "UV texture bake resolution in pixels.",
                    },
                ),
                "remesh_option": (
                    ["none", "triangle", "quad"],
                    {"default": "none", "tooltip": "Optional remeshing pass on the output topology."},
                ),
                "target_vertex_count": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 200000,
                        "step": 1000,
                        "tooltip": "Target vertex count when remeshing (-1 uses SPAR3D default).",
                    },
                ),
                "filename_prefix": ("STRING", {"default": "ziggart_spar3d_mesh"}),
                "unload_comfy_models": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Evict ComfyUI models from VRAM before running SPAR3D."},
                ),
                "free_vram_after": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Unload SPAR3D from GPU VRAM after mesh generation."},
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional foreground alpha mask."}),
                "model_id_or_path": (
                    "STRING",
                    {
                        "default": "stabilityai/stable-point-aware-3d",
                        "tooltip": "Hugging Face repo ID or local path for SPAR3D weights.",
                    },
                ),
                "hf_token": ("STRING", {"default": "", "tooltip": "Optional Hugging Face access token."}),
            },
        }

    def generate_mesh(
        self,
        image: Any,
        low_vram_mode: bool = True,
        bake_resolution: int = 1024,
        remesh_option: str = "none",
        target_vertex_count: int = -1,
        filename_prefix: str = "ziggart_spar3d_mesh",
        unload_comfy_models: bool = True,
        free_vram_after: bool = True,
        mask: Any = None,
        model_id_or_path: str = "stabilityai/stable-point-aware-3d",
        hf_token: str = "",
    ) -> tuple[str, dict[str, Any], str, Any]:
        pil_img = tensor_to_pil(image, mask=mask)
        out_glb = _next_output_path(filename_prefix, "glb")
        out_ply = _next_output_path(f"{filename_prefix}_points", "ply")

        glb_path, ply_path, mesh_data = THREED_MANAGER.run_spar3d(
            pil_image=pil_img,
            model_id_or_path=model_id_or_path.strip() or "stabilityai/stable-point-aware-3d",
            low_vram_mode=bool(low_vram_mode),
            bake_resolution=int(bake_resolution),
            remesh_option=remesh_option,
            target_vertex_count=int(target_vertex_count),
            hf_token=hf_token,
            unload_comfy_models=bool(unload_comfy_models),
            free_vram_after=bool(free_vram_after),
            output_glb_path=out_glb,
            output_ply_path=out_ply,
        )

        preview_tensor = render_4view_contact_sheet(
            positions=mesh_data["vertices"],
            colors=mesh_data["vertex_colors"],
            faces=mesh_data["faces"],
            normals=mesh_data["normals"],
        )
        return (str(glb_path), mesh_data, str(ply_path), preview_tensor)


class ZiggartHunyuan3DImageToMesh:
    """Single-image to high-detail 3D Mesh (.glb) using Tencent's Hunyuan3D-2mini (~5.5-6 GB VRAM)."""

    DESCRIPTION = (
        "Generates high-sharpness watertight 3D Meshes (.glb) using Hunyuan3D-2mini (0.6B). "
        "Uses a sequential VRAM pipeline on RTX 3070 Ti: Shape generation (~5.5 GB VRAM) runs first, "
        "and if generate_texture is enabled, Shape is unloaded before Hunyuan3DPaint runs."
    )
    CATEGORY = "Ziggart/3D"
    FUNCTION = "generate_mesh"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "ZIGGART_MESH", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mesh_path", "mesh_data", "preprocessed_image", "preview_image")
    OUTPUT_TOOLTIPS = (
        "Absolute path to the generated .glb 3D mesh.",
        "In-memory mesh dictionary for Ziggart3DTurntableRenderer.",
        "Background-removed RGBA input image used for conditioning.",
        "2x2 shaded multi-angle preview render of the generated mesh.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input single image to reconstruct into a 3D mesh."}),
                "model_variant": (
                    [
                        "Hunyuan3D-2mini (0.6B - 6GB VRAM)",
                        "Hunyuan3D-2mini-Turbo (Fast 6GB VRAM)",
                        "Hunyuan3D-2 (Full 10GB+ VRAM)",
                    ],
                    {
                        "default": "Hunyuan3D-2mini (0.6B - 6GB VRAM)",
                        "tooltip": "Hunyuan3D-2mini variants fit comfortably in 6GB-8GB VRAM.",
                    },
                ),
                "steps": ("INT", {"default": 30, "min": 4, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 1.0, "max": 15.0, "step": 0.5}),
                "octree_resolution": (
                    "INT",
                    {
                        "default": 256,
                        "min": 128,
                        "max": 512,
                        "step": 32,
                        "tooltip": "Marching cubes octree resolution (256-380 is ideal for RTX 3070 Ti).",
                    },
                ),
                "num_chunks": (
                    "INT",
                    {
                        "default": 8000,
                        "min": 2000,
                        "max": 50000,
                        "step": 2000,
                        "tooltip": "Volume decoding chunk size (lower values use less peak VRAM).",
                    },
                ),
                "generate_texture": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Sequentially unload Shape model and run Hunyuan3DPaint texture synthesis.",
                    },
                ),
                "max_faces": (
                    "INT",
                    {
                        "default": 40000,
                        "min": 5000,
                        "max": 200000,
                        "step": 5000,
                        "tooltip": "Maximum triangle count after face reduction.",
                    },
                ),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2147483647, "step": 1}),
                "filename_prefix": ("STRING", {"default": "ziggart_hunyuan3d_mesh"}),
                "unload_comfy_models": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Evict ComfyUI models from VRAM before loading Hunyuan3D."},
                ),
                "free_vram_after": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Unload Hunyuan3D from GPU VRAM after inference."},
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional foreground mask."}),
                "custom_model_path": (
                    "STRING",
                    {"default": "", "tooltip": "Optional local path or custom Hugging Face repo override."},
                ),
            },
        }

    def generate_mesh(
        self,
        image: Any,
        model_variant: str = "Hunyuan3D-2mini (0.6B - 6GB VRAM)",
        steps: int = 30,
        guidance_scale: float = 5.5,
        octree_resolution: int = 256,
        num_chunks: int = 8000,
        generate_texture: bool = False,
        max_faces: int = 40000,
        seed: int = 12345,
        filename_prefix: str = "ziggart_hunyuan3d_mesh",
        unload_comfy_models: bool = True,
        free_vram_after: bool = True,
        mask: Any = None,
        custom_model_path: str = "",
    ) -> tuple[str, dict[str, Any], Any, Any]:
        pil_img = tensor_to_pil(image, mask=mask)
        out_glb = _next_output_path(filename_prefix, "glb")

        glb_path, rgba_pil, mesh_data = THREED_MANAGER.run_hunyuan3d(
            pil_image=pil_img,
            model_variant=model_variant,
            custom_model_path=custom_model_path,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            octree_resolution=int(octree_resolution),
            num_chunks=int(num_chunks),
            generate_texture=bool(generate_texture),
            max_faces=int(max_faces),
            seed=int(seed),
            unload_comfy_models=bool(unload_comfy_models),
            free_vram_after=bool(free_vram_after),
            output_glb_path=out_glb,
        )

        preprocessed_tensor = pil_to_tensor(rgba_pil)
        preview_tensor = render_4view_contact_sheet(
            positions=mesh_data["vertices"],
            colors=mesh_data["vertex_colors"],
            faces=mesh_data["faces"],
            normals=mesh_data["normals"],
        )
        return (str(glb_path), mesh_data, preprocessed_tensor, preview_tensor)


class Ziggart3DTurntableRenderer:
    """Render turntable IMAGE and MASK batches from a ZIGGART_GSPLAT or ZIGGART_MESH."""

    DESCRIPTION = (
        "Renders single or multi-frame 360-degree turntable IMAGE and MASK batches from "
        "either a 3D Gaussian Splat (ZIGGART_GSPLAT / .ply) or a 3D Mesh (ZIGGART_MESH) "
        "so you can pipe 3D views directly into KSampler, ControlNet, or video nodes."
    )
    CATEGORY = "Ziggart/3D"
    FUNCTION = "render_turntable"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "Rendered turntable frames as a ComfyUI IMAGE batch [N, H, W, 3].",
        "Foreground alpha masks for each rendered frame [N, H, W].",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "num_frames": ("INT", {"default": 8, "min": 1, "max": 120, "step": 1}),
                "azimuth_start": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 5.0}),
                "azimuth_sweep": ("FLOAT", {"default": 360.0, "min": -360.0, "max": 360.0, "step": 15.0}),
                "elevation": ("FLOAT", {"default": 20.0, "min": -89.0, "max": 89.0, "step": 5.0}),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 1536, "step": 64}),
                "point_radius": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "gsplat_data": ("ZIGGART_GSPLAT",),
                "mesh_data": ("ZIGGART_MESH",),
                "ply_path": ("STRING", {"default": ""}),
            },
        }

    def render_turntable(
        self,
        num_frames: int = 8,
        azimuth_start: float = 0.0,
        azimuth_sweep: float = 360.0,
        elevation: float = 20.0,
        resolution: int = 512,
        point_radius: int = 2,
        gsplat_data: dict[str, Any] | None = None,
        mesh_data: dict[str, Any] | None = None,
        ply_path: str = "",
    ) -> tuple[Any, Any]:
        if mesh_data is not None:
            positions = mesh_data["vertices"]
            colors = mesh_data["vertex_colors"]
            faces = mesh_data.get("faces")
            normals = mesh_data.get("normals")
            opacities = None
        else:
            if gsplat_data is None:
                if not ply_path or not str(ply_path).strip():
                    raise ValueError("Connect 'gsplat_data', 'mesh_data', or provide a 'ply_path' to render.")
                gsplat_data = load_gaussian_ply(ply_path.strip())
            positions = gsplat_data["positions"]
            colors = gsplat_data["colors"]
            faces = None
            normals = None
            opacities = gsplat_data.get("opacities")

        frames_count = max(1, int(num_frames))
        frames_rgb = []
        frames_mask = []
        for idx in range(frames_count):
            step_angle = float(azimuth_start) + (float(azimuth_sweep) * idx / float(frames_count))
            rgb, m = render_3d_view(
                positions=positions,
                colors=colors,
                faces=faces,
                normals=normals,
                opacities=opacities,
                azimuth_deg=step_angle,
                elevation_deg=float(elevation),
                resolution=int(resolution),
                point_radius=int(point_radius),
            )
            frames_rgb.append(rgb)
            frames_mask.append(m)

        batch_rgb = np.stack(frames_rgb, axis=0).astype(np.float32)
        batch_mask = np.stack(frames_mask, axis=0).astype(np.float32)
        if torch is not None:
            return (torch.from_numpy(batch_rgb), torch.from_numpy(batch_mask))
        return (batch_rgb, batch_mask)


NODE_CLASS_MAPPINGS = {
    "ZiggartIdolImageToGaussian": ZiggartIdolImageToGaussian,
    "ZiggartGaussianToMesh": ZiggartGaussianToMesh,
    "ZiggartSPAR3DImageToMesh": ZiggartSPAR3DImageToMesh,
    "ZiggartHunyuan3DImageToMesh": ZiggartHunyuan3DImageToMesh,
    "Ziggart3DTurntableRenderer": Ziggart3DTurntableRenderer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartIdolImageToGaussian": "Idol Image to 3D Gaussian Splat (PLY)",
    "ZiggartGaussianToMesh": "Convert Idol Gaussian Splat to 3D Mesh (GLB/OBJ)",
    "ZiggartSPAR3DImageToMesh": "SPAR3D Image to 3D Mesh (Fast 7GB VRAM)",
    "ZiggartHunyuan3DImageToMesh": "Hunyuan3D-2mini Image to 3D Mesh (GLB)",
    "Ziggart3DTurntableRenderer": "3D Gaussian / Mesh Turntable Renderer",
}
