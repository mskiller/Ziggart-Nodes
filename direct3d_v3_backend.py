"""
Direct3D-V3 Backend for Ziggart-Nodes (v1.11.0).

Credits & Licenses:
- Direct3D-V3 architecture, SparseDiT, and reference inference pipeline:
  DreamTechAI (https://github.com/DreamTechAI/Direct3D-V3), MIT License.
- Direct3D-V3 INT8 (w8g32) quantization format, single-stage 8GB VRAM optimizations,
  multiview refinement, APG/FDG guidance, and sparse dual contouring concepts:
  Mostafa Awad (@MostAadTech, https://github.com/Mstafa-awad/Direct3D-V3_Comfyui_Nodes), MIT License.
- Image encoder: Meta DINOv3 ViT-H+ (DINOv3 License).
"""
from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False

try:
    import folder_paths
except ImportError:
    folder_paths = None


# ---------------------------------------------------------------------------
# Constants & Model Registry
# ---------------------------------------------------------------------------

DIRECT3D_V3_REPOS: dict[str, str] = {
    "int8": "MostAadTech/Direct3D-V3-int8",
    "fp16": "wushuang98/Direct3D-V3",
}

DIRECT3D_V3_STAGES: tuple[int, ...] = (512, 1024, 1536)
DEFAULT_DINOV3_ENCODER_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DEFAULT_MAX_TOKENS = 150_000

VIEW_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "left": (1.0, 0.0, 0.0),
    "right": (-1.0, 0.0, 0.0),
    "top": (0.0, 1.0, 0.0),
    "bottom": (0.0, -1.0, 0.0),
}


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def get_direct3d_v3_root(variant: str = "int8") -> Path:
    """Return persistent directory where Direct3D-V3 weights are cached."""
    clean_var = "fp16" if str(variant).lower().strip() == "fp16" else "int8"
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        base = Path(folder_paths.models_dir) / "3d" / "ziggart_3d"
    else:
        base = Path(__file__).resolve().parent / ".cache" / "models_3d" / "ziggart_3d"
    root = base / "direct3d_v3" / clean_var
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_valid_stage_dir(stage_dir: Path) -> bool:
    return (
        (stage_dir / "config.json").is_file()
        and (stage_dir / "model.safetensors").is_file()
        and (stage_dir / "model.safetensors").stat().st_size > 1024
    )


def find_existing_direct3d_v3_dir(variant: str = "int8", stages: tuple[int, ...] = (1024,)) -> Path | None:
    """
    Locate an existing local Direct3D-V3 directory if already downloaded in Ziggart's cache
    or in a sibling ComfyUI-Direct3D-V3 installation (read-only reuse).
    """
    candidates: list[Path] = [get_direct3d_v3_root(variant)]
    # Check sibling ComfyUI-Direct3D-V3 models folder if present
    custom_nodes_dir = Path(__file__).resolve().parent.parent
    candidates.append(custom_nodes_dir / "ComfyUI-Direct3D-V3" / "models" / "Direct3D-V3")
    if folder_paths is not None and hasattr(folder_paths, "models_dir"):
        candidates.append(Path(folder_paths.models_dir) / "Direct3D-V3")
        candidates.append(Path(folder_paths.models_dir) / "3d" / "Direct3D-V3")

    for cand in candidates:
        if not cand.is_dir():
            continue
        if not (cand / "model_index.json").is_file():
            continue
        if all(_is_valid_stage_dir(cand / f"shape_transformers_{st}") for st in stages):
            return cand
    return None


def download_direct3d_v3_files(
    variant: str = "int8",
    stages: tuple[int, ...] = (1536,),
    *,
    force_redownload: bool = False,
    hf_token: str = "",
) -> dict[str, Any]:
    """Download Direct3D-V3 config and requested stage weights into Ziggart's persistent 3D cache."""
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    clean_var = "fp16" if str(variant).lower().strip() == "fp16" else "int8"
    repo_id = DIRECT3D_V3_REPOS[clean_var]
    dest_root = get_direct3d_v3_root(clean_var)
    token = hf_token.strip() if (hf_token and hf_token.strip()) else os.environ.get("HF_TOKEN")

    files_to_fetch = ["config.json", "model_index.json"]
    for st in stages:
        if st not in DIRECT3D_V3_STAGES:
            raise ValueError(f"Unsupported Direct3D-V3 stage {st}; expected one of {DIRECT3D_V3_STAGES}.")
        files_to_fetch.append(f"shape_transformers_{st}/config.json")
        files_to_fetch.append(f"shape_transformers_{st}/model.safetensors")

    downloaded_files: dict[str, Any] = {}
    total_mb = 0.0
    for rel_path in files_to_fetch:
        local_target = dest_root / rel_path
        if not force_redownload and local_target.is_file() and local_target.stat().st_size > 16:
            sz = round(local_target.stat().st_size / (1024 * 1024), 2)
            total_mb += sz
            downloaded_files[rel_path] = {"status": "cached", "path": str(local_target), "size_mb": sz}
            continue

        local_target.parent.mkdir(parents=True, exist_ok=True)
        fetched = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                local_dir=str(dest_root),
                token=token,
                force_download=bool(force_redownload),
            )
        )
        if fetched.is_file():
            sz = round(fetched.stat().st_size / (1024 * 1024), 2)
            total_mb += sz
            downloaded_files[rel_path] = {"status": "downloaded", "path": str(fetched), "size_mb": sz}
        else:
            downloaded_files[rel_path] = {"status": "missing"}

    return {
        "variant": clean_var,
        "repo_id": repo_id,
        "local_dir": str(dest_root),
        "stages": list(stages),
        "total_size_mb": round(total_mb, 2),
        "files": downloaded_files,
        "status": "ready" if all(v.get("status") in ("cached", "downloaded") for v in downloaded_files.values()) else "incomplete",
    }


def resolve_direct3d_image_encoder(
    model_dir_or_repo: str | Path,
    model_index: dict[str, Any],
    image_encoder_override: str = "",
    hf_token: str = "",
) -> dict[str, Any]:
    """
    Resolve DINOv3 image_encoder kwargs safely.
    Handles MostAadTech/Direct3D-V3-int8's `model_index.json` which specifies `"./image_encoder"`
    even when `image_encoder/` is not bundled in the HF repo — falling back to
    `facebook/dinov3-vith16plus-pretrain-lvd1689m`.
    """
    override = str(image_encoder_override or "").strip()
    if override and override.lower() not in ("true", "false"):
        return {"model_name_or_path": override}

    enc_spec = str(model_index.get("image_encoder", "")).strip()
    if enc_spec.startswith("./"):
        rel = enc_spec[2:].strip("/")
        base_path = Path(model_dir_or_repo) if isinstance(model_dir_or_repo, Path) or Path(str(model_dir_or_repo)).is_dir() else None
        if base_path is not None:
            local_enc = base_path / rel
            if (local_enc / "config.json").is_file():
                return {"model_name_or_path": str(local_enc)}
        return {"model_name_or_path": DEFAULT_DINOV3_ENCODER_ID}

    if enc_spec:
        return {"model_name_or_path": enc_spec}
    return {"model_name_or_path": DEFAULT_DINOV3_ENCODER_ID}


# ---------------------------------------------------------------------------
# W8G32 INT8 Dequantization, Triton Fused Linear Kernel & Checkpoint Loader
# ---------------------------------------------------------------------------


def dequantize_w8g32(
    q: Any,
    s: Any,
    group_size: int = 32,
    dtype: Any = None,
) -> Any:
    """Dequantize `@MostAadTech` direct3d-v3-w8g32 weights: q8 [out, in], s [out, in // group]."""
    assert torch is not None
    if dtype is None:
        dtype = torch.float16
    out_f, in_f = q.shape
    groups = in_f // group_size
    w = q.float().view(out_f, groups, group_size) * s.float().view(out_f, groups, 1)
    return w.view(out_f, in_f).to(dtype)


if TRITON_AVAILABLE and triton is not None and tl is not None:

    @triton.jit
    def _w8g32_matmul_kernel(
        A_ptr,
        Q_ptr,
        S_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_qn,
        stride_qk,
        stride_sn,
        stride_sk,
        stride_cm,
        stride_cn,
        HAS_BIAS: tl.constexpr,
        OUT_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        num_k_blocks = K // BLOCK_K  # K is verified divisible by BLOCK_K == 32

        for k_idx in range(0, num_k_blocks):
            k_base = k_idx * BLOCK_K
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k_base + offs_k[None, :]) * stride_ak
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

            # Load Q as [BLOCK_K, BLOCK_N] so tl.dot(a, b) computes A[M, K] @ Q^T[K, N]
            q_ptrs = Q_ptr + (k_base + offs_k[:, None]) * stride_qk + offs_n[None, :] * stride_qn
            q = tl.load(q_ptrs, mask=mask_n[None, :], other=0)

            # One scale per group of 32 along K (BLOCK_K == 32 => 1 scale per output col)
            s_ptrs = S_ptr + offs_n * stride_sn + k_idx * stride_sk
            s = tl.load(s_ptrs, mask=mask_n, other=0.0).to(tl.float32)

            b = q.to(tl.float32) * s[None, :]
            if OUT_BF16:
                a_cast = a.to(tl.bfloat16)
                b_cast = b.to(tl.bfloat16)
            else:
                a_cast = a.to(tl.float16)
                b_cast = b.to(tl.float16)
            acc = tl.dot(a_cast, b_cast, acc)

        if HAS_BIAS:
            bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        if OUT_BF16:
            tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])
        else:
            tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def _triton_w8g32_linear(
    x: Any,
    q8: Any,
    scales: Any,
    bias: Any = None,
    group_size: int = 32,
) -> Any:
    """Fused Triton W8G32 quantized linear forward pass."""
    assert torch is not None and TRITON_AVAILABLE
    orig_shape = x.shape
    k = orig_shape[-1]
    n = q8.shape[0]
    x_2d = x.reshape(-1, k).contiguous()
    m = x_2d.shape[0]
    if m == 0 or n == 0:
        return torch.empty((*orig_shape[:-1], n), device=x.device, dtype=x.dtype)

    if group_size != 32 or (k % 32) != 0:
        w = dequantize_w8g32(q8, scales, group_size=group_size, dtype=x.dtype)
        return F.linear(x, w, bias.to(x.dtype) if bias is not None else None)

    out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
    if x_2d.dtype != out_dtype:
        x_2d = x_2d.to(out_dtype)

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    block_m = 32 if m <= 64 else 64
    block_n = 64
    block_k = 32

    has_bias = bias is not None
    b_ptr = bias.contiguous() if has_bias else out  # dummy non-null ptr when unused

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _w8g32_matmul_kernel[grid](
        x_2d,
        q8,
        scales,
        b_ptr,
        out,
        m,
        n,
        k,
        x_2d.stride(0),
        x_2d.stride(1),
        q8.stride(0),
        q8.stride(1),
        scales.stride(0),
        scales.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=has_bias,
        OUT_BF16=(out_dtype == torch.bfloat16),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    if x.dtype != out_dtype:
        out = out.to(x.dtype)
    return out.reshape(*orig_shape[:-1], n)


if nn is not None:

    class W8G32Linear(nn.Module):
        """Memory-compact INT8 W8G32 Linear layer with Triton fused matmul & PyTorch fallback."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            q8: Any,
            scales: Any,
            bias: Any = None,
            group_size: int = 32,
            use_triton: bool = True,
        ) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.group_size = int(group_size)
            self.use_triton = bool(use_triton)
            self.register_buffer("q8", q8.to(torch.int8).contiguous())
            self.register_buffer("scales", scales.to(torch.float32).contiguous())
            if bias is not None:
                self.register_buffer("bias", bias.contiguous())
            else:
                self.bias = None

        @property
        def dtype(self) -> Any:
            return self.bias.dtype if self.bias is not None else torch.float16

        @property
        def weight(self) -> Any:
            """Expose .dtype/.device/.shape for callers like sparse_dit.py inspecting layer.weight.dtype."""
            from types import SimpleNamespace

            return SimpleNamespace(
                dtype=self.dtype,
                device=self.q8.device,
                shape=(self.out_features, self.in_features),
            )

        def forward(self, x: Any) -> Any:
            if (
                self.use_triton
                and TRITON_AVAILABLE
                and x.is_cuda
                and self.group_size == 32
                and (self.in_features % 32) == 0
            ):
                return _triton_w8g32_linear(x, self.q8, self.scales, self.bias, self.group_size)
            w = dequantize_w8g32(self.q8, self.scales, group_size=self.group_size, dtype=x.dtype)
            b = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)

else:  # pragma: no cover

    class W8G32Linear:  # type: ignore[no-redef]
        pass


def _set_submodule(root: Any, dotted_name: str, new_module: Any) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def load_direct3d_v3_denoiser(
    path: str | Path,
    denoiser_cfg: dict[str, Any],
    dtype: Any,
    *,
    use_triton_int8: bool = True,
    dequant_to_fp16: bool = False,
) -> Any:
    """
    Load a Direct3D-V3 SparseDiT denoiser from either:
    - `@MostAadTech` `direct3d-v3-w8g32` INT8 safetensors (keeps INT8 weights in VRAM via W8G32Linear,
      or dequantizes to FP16 when `dequant_to_fp16=True`).
    - Official `wushuang98/Direct3D-V3` FP32/FP16 safetensors.
    """
    assert torch is not None
    from safetensors import safe_open  # type: ignore[import-not-found]
    from direct3d.models.sparse_dit import SparseDiT  # type: ignore[import-not-found]

    ckpt_path = Path(path)
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as archive:
        meta = archive.metadata() or {}
        names = set(archive.keys())
        fmt = meta.get("format", "")

        with torch.device("meta"):
            model = SparseDiT(**denoiser_cfg)

        if fmt != "direct3d-v3-w8g32" and not any(k.endswith(".q8") for k in names):
            # Official FP32 / FP16 checkpoint
            keys = list(model.state_dict().keys())
            prefix = ""
            if names != set(keys):
                for cand in ("", "model.", "model.model."):
                    if all(cand + k in names for k in keys):
                        prefix = cand
                        break
                else:
                    raise ValueError(f"Checkpoint {ckpt_path} does not match SparseDiT parameters.")
            tensors = {k: archive.get_tensor(prefix + k).to(dtype=dtype) for k in keys}
            model.load_state_dict(tensors, assign=True, strict=True)
            return model.eval()

        # W8G32 INT8 checkpoint
        group_size = int(meta.get("group", 32) or 32)
        q8_keys = {k[:-3] for k in names if k.endswith(".q8")}
        # Determine optional prefix
        prefix = ""
        for cand in ("", "model.", "model.model."):
            if any((cand + m_name + ".weight") in q8_keys for m_name, _ in model.named_modules()):
                prefix = cand
                break

        quantized_modules: dict[str, str] = {}
        for mod_name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear):
                w_key = f"{prefix}{mod_name}.weight"
                if w_key in q8_keys:
                    quantized_modules[mod_name] = w_key

        if dequant_to_fp16:
            tensors: dict[str, Any] = {}
            for k in model.state_dict().keys():
                full_k = prefix + k
                if full_k in q8_keys:
                    q = archive.get_tensor(full_k + ".q8")
                    s = archive.get_tensor(full_k + ".s")
                    tensors[k] = dequantize_w8g32(q, s, group_size=group_size, dtype=dtype)
                else:
                    tensors[k] = archive.get_tensor(full_k).to(dtype=dtype)
            model.load_state_dict(tensors, assign=True, strict=True)
            return model.eval()

        # Replace quantized nn.Linear layers with compact W8G32Linear
        for mod_name, w_key in quantized_modules.items():
            orig = model.get_submodule(mod_name)
            q = archive.get_tensor(w_key + ".q8")
            s = archive.get_tensor(w_key + ".s")
            b_key = f"{prefix}{mod_name}.bias"
            b = archive.get_tensor(b_key).to(dtype=dtype) if b_key in names else None
            w8_mod = W8G32Linear(
                in_features=orig.in_features,
                out_features=orig.out_features,
                q8=q,
                scales=s,
                bias=b,
                group_size=group_size,
                use_triton=use_triton_int8,
            )
            _set_submodule(model, mod_name, w8_mod)

        # Load remaining non-quantized weights/buffers
        remaining_tensors: dict[str, Any] = {}
        for k in model.state_dict().keys():
            if k.endswith(".q8") or k.endswith(".scales"):
                continue
            # Check if this bias was already loaded into W8G32Linear
            mod_part = k.rsplit(".", 1)[0] if "." in k else ""
            if mod_part in quantized_modules and k.endswith(".bias"):
                continue
            full_k = prefix + k
            if full_k in names:
                remaining_tensors[k] = archive.get_tensor(full_k).to(dtype=dtype)

        model.load_state_dict(remaining_tensors, assign=True, strict=False)
        return model.eval()


# ---------------------------------------------------------------------------
# Zero-Build Windows Shims: Triton/PyTorch Voxelizer, FillHoles, FlashAttn
# ---------------------------------------------------------------------------


if TRITON_AVAILABLE and triton is not None and tl is not None:

    @triton.jit
    def _sat_axis_overlap(
        v0x, v0y, v0z,
        v1x, v1y, v1z,
        v2x, v2y, v2z,
        hx, hy, hz,
        ax, ay, az,
    ):
        p0 = v0x * ax + v0y * ay + v0z * az
        p1 = v1x * ax + v1y * ay + v1z * az
        p2 = v2x * ax + v2y * ay + v2z * az
        pmin = tl.minimum(p0, tl.minimum(p1, p2))
        pmax = tl.maximum(p0, tl.maximum(p1, p2))
        r = hx * tl.abs(ax) + hy * tl.abs(ay) + hz * tl.abs(az)
        is_zero = (ax == 0.0) & (ay == 0.0) & (az == 0.0)
        return is_zero | ((pmin <= r) & (pmax >= -r))

    @triton.jit
    def _voxelize_sat_kernel(
        verts_ptr,
        faces_ptr,
        voxel_ptr,
        num_tris,
        dim: tl.constexpr,
    ):
        t = tl.program_id(0)
        if t >= num_tris:
            return

        f0 = tl.load(faces_ptr + t * 3 + 0)
        f1 = tl.load(faces_ptr + t * 3 + 1)
        f2 = tl.load(faces_ptr + t * 3 + 2)

        x0 = tl.load(verts_ptr + f0 * 3 + 0)
        y0 = tl.load(verts_ptr + f0 * 3 + 1)
        z0 = tl.load(verts_ptr + f0 * 3 + 2)

        x1 = tl.load(verts_ptr + f1 * 3 + 0)
        y1 = tl.load(verts_ptr + f1 * 3 + 1)
        z1 = tl.load(verts_ptr + f1 * 3 + 2)

        x2 = tl.load(verts_ptr + f2 * 3 + 0)
        y2 = tl.load(verts_ptr + f2 * 3 + 1)
        z2 = tl.load(verts_ptr + f2 * 3 + 2)

        min_x = tl.minimum(x0, tl.minimum(x1, x2))
        min_y = tl.minimum(y0, tl.minimum(y1, y2))
        min_z = tl.minimum(z0, tl.minimum(z1, z2))
        max_x = tl.maximum(x0, tl.maximum(x1, x2))
        max_y = tl.maximum(y0, tl.maximum(y1, y2))
        max_z = tl.maximum(z0, tl.maximum(z1, z2))

        inv_step = float(dim - 1)
        step = 1.0 / inv_step
        half_step = 0.5 * step

        i_min = tl.maximum(0, tl.floor((min_x + 0.5) * inv_step).to(tl.int32) - 1)
        j_min = tl.maximum(0, tl.floor((min_y + 0.5) * inv_step).to(tl.int32) - 1)
        k_min = tl.maximum(0, tl.floor((min_z + 0.5) * inv_step).to(tl.int32) - 1)

        i_max = tl.minimum(dim - 1, tl.floor((max_x + 0.5) * inv_step).to(tl.int32) + 1)
        j_max = tl.minimum(dim - 1, tl.floor((max_y + 0.5) * inv_step).to(tl.int32) + 1)
        k_max = tl.minimum(dim - 1, tl.floor((max_z + 0.5) * inv_step).to(tl.int32) + 1)

        # Edges
        e0x, e0y, e0z = x1 - x0, y1 - y0, z1 - z0
        e1x, e1y, e1z = x2 - x1, y2 - y1, z2 - z1
        e2x, e2y, e2z = x0 - x2, y0 - y2, z0 - z2
        nx = e0y * e1z - e0z * e1y
        ny = e0z * e1x - e0x * e1z
        nz = e0x * e1y - e0y * e1x

        for i in range(i_min, i_max + 1):
            cx = i * step - 0.5 + half_step
            for j in range(j_min, j_max + 1):
                cy = j * step - 0.5 + half_step
                for k in range(k_min, k_max + 1):
                    cz = k * step - 0.5 + half_step
                    tv0x, tv0y, tv0z = x0 - cx, y0 - cy, z0 - cz
                    tv1x, tv1y, tv1z = x1 - cx, y1 - cy, z1 - cz
                    tv2x, tv2y, tv2z = x2 - cx, y2 - cy, z2 - cz

                    ok = _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, 1.0, 0.0, 0.0)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, 0.0, 1.0, 0.0)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, 0.0, 0.0, 1.0)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, nx, ny, nz)

                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, 0.0, -e0z, e0y)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, 0.0, -e1z, e1y)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, 0.0, -e2z, e2y)

                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, e0z, 0.0, -e0x)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, e1z, 0.0, -e1x)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, e2z, 0.0, -e2x)

                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, -e0y, e0x, 0.0)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, -e1y, e1x, 0.0)
                    ok = ok & _sat_axis_overlap(tv0x, tv0y, tv0z, tv1x, tv1y, tv1z, tv2x, tv2y, tv2z, half_step, half_step, half_step, -e2y, e2x, 0.0)

                    if ok:
                        idx = i * dim * dim + j * dim + k
                        tl.store(voxel_ptr + idx, 1)


def _subdivide_large_triangles(vertices: Any, faces: Any, dim: int, max_span_vox: int = 16, max_passes: int = 5) -> tuple[Any, Any]:
    """Midpoint-subdivide triangles whose voxel AABB exceeds `max_span_vox` so Triton SAT loops stay tiny."""
    verts = vertices
    tris = faces.long()
    for _ in range(max_passes):
        tv = verts[tris]  # [T, 3, 3]
        span = (tv.amax(dim=1) - tv.amin(dim=1)) * float(dim - 1)
        large_mask = span.amax(dim=-1) > float(max_span_vox)
        if not large_mask.any():
            break
        small_tris = tris[~large_mask]
        big_tris = tris[large_mask]
        v0 = verts[big_tris[:, 0]]
        v1 = verts[big_tris[:, 1]]
        v2 = verts[big_tris[:, 2]]
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)
        base = len(verts)
        nb = len(big_tris)
        verts = torch.cat([verts, m01, m12, m20], dim=0)
        i01 = torch.arange(base, base + nb, device=verts.device, dtype=torch.long)
        i12 = torch.arange(base + nb, base + 2 * nb, device=verts.device, dtype=torch.long)
        i20 = torch.arange(base + 2 * nb, base + 3 * nb, device=verts.device, dtype=torch.long)
        t0 = torch.stack([big_tris[:, 0], i01, i20], dim=1)
        t1 = torch.stack([i01, big_tris[:, 1], i12], dim=1)
        t2 = torch.stack([i20, i12, big_tris[:, 2]], dim=1)
        t3 = torch.stack([i01, i12, i20], dim=1)
        tris = torch.cat([small_tris, t0, t1, t2, t3], dim=0)
    return verts.contiguous(), tris.to(torch.int32).contiguous()


def voxelize_mesh_triton(
    vertices: Any,
    faces: Any,
    voxel: Any,
    num_triangles: int,
    dim: int = 512,
    threshold: float = 8.0,
) -> None:
    """Triton GPU SAT triangle-voxelizer matching `cudavoxelize.voxelize`."""
    assert torch is not None and TRITON_AVAILABLE
    if num_triangles <= 0:
        return
    verts, tris = _subdivide_large_triangles(vertices.to(torch.float32), faces, dim=int(dim))
    n_tris = int(tris.shape[0])
    vox_u8 = voxel.view(torch.uint8)
    _voxelize_sat_kernel[(n_tris,)](
        verts,
        tris,
        vox_u8,
        n_tris,
        dim=int(dim),
    )


def voxelize_mesh_pure_torch(
    vertices: Any,
    faces: Any,
    voxel: Any,
    num_triangles: int,
    dim: int = 512,
    threshold: float = 8.0,
) -> None:
    """Pure-PyTorch adaptive barycentric triangle-surface voxelizer fallback (CPU or GPU)."""
    assert torch is not None
    if num_triangles <= 0:
        return
    dev = vertices.device
    verts = vertices.to(torch.float32)
    tris = faces[:num_triangles].long()
    tv = verts[tris]  # [T, 3, 3]

    # Adaptive sampling density from longest triangle edge in voxel units
    e01 = torch.linalg.norm(tv[:, 1] - tv[:, 0], dim=-1)
    e12 = torch.linalg.norm(tv[:, 2] - tv[:, 1], dim=-1)
    e20 = torch.linalg.norm(tv[:, 0] - tv[:, 2], dim=-1)
    max_edge_vox = float(torch.maximum(torch.maximum(e01, e12), e20).max().item()) * float(dim - 1)
    steps = max(3, min(32, int(math.ceil(max_edge_vox * 1.5)) + 1))

    # Build uniform barycentric grid (u, v, w >= 0, u+v+w == 1)
    lin = torch.linspace(0.0, 1.0, steps, device=dev, dtype=torch.float32)
    uu, vv = torch.meshgrid(lin, lin, indexing="ij")
    mask = (uu + vv) <= 1.0 + 1e-5
    u = uu[mask]
    v = vv[mask]
    w = (1.0 - u - v).clamp(min=0.0)
    bary = torch.stack([u, v, w], dim=-1)  # [S, 3]

    chunk_size = max(512, 131072 // max(1, len(bary)))
    inv_step = float(dim - 1)
    for start in range(0, len(tv), chunk_size):
        chunk_tv = tv[start : start + chunk_size]  # [C, 3, 3]
        pts = torch.einsum("sc,tcd->tsd", bary, chunk_tv).reshape(-1, 3)
        coords = torch.round((pts + 0.5) * inv_step).long()
        valid = (coords >= 0).all(dim=-1) & (coords < dim).all(dim=-1)
        coords = coords[valid]
        if len(coords):
            flat_idx = (coords[:, 0] * dim + coords[:, 1]) * dim + coords[:, 2]
            voxel[flat_idx] = True


def _sdpa_varlen_kvpacked(q: Any, kv: Any, cu_seqlens_q: Any, cu_seqlens_kv: Any, max_seqlen_q: int, max_seqlen_kv: int) -> Any:
    assert torch is not None
    out = torch.empty_like(q)
    b = len(cu_seqlens_q) - 1
    for i in range(b):
        qs, qe = int(cu_seqlens_q[i].item()), int(cu_seqlens_q[i + 1].item())
        ks, ke = int(cu_seqlens_kv[i].item()), int(cu_seqlens_kv[i + 1].item())
        if qe <= qs or ke <= ks:
            out[qs:qe].zero_()
            continue
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)  # [1, H, Sq, D]
        ki = kv[ks:ke, 0].transpose(0, 1).unsqueeze(0)  # [1, H, Sk, D]
        vi = kv[ks:ke, 1].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi)
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)
    return out


def _sdpa_qkvpacked(qkv: Any) -> Any:
    assert torch is not None
    # qkv: [B, N, 3, H, C]
    q = qkv[:, :, 0].permute(0, 2, 1, 3)  # [B, H, N, C]
    k = qkv[:, :, 1].permute(0, 2, 1, 3)
    v = qkv[:, :, 2].permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3).contiguous()


def _sdpa_varlen_qkvpacked(qkv: Any, cu_seqlens: Any, max_seqlen: int) -> Any:
    assert torch is not None
    # qkv: [M, 3, H, C]
    m, _, h, c = qkv.shape
    out = torch.empty((m, h, c), device=qkv.device, dtype=qkv.dtype)
    b = len(cu_seqlens) - 1
    for i in range(b):
        s, e = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        if e <= s:
            continue
        qi = qkv[s:e, 0].transpose(0, 1).unsqueeze(0)
        ki = qkv[s:e, 1].transpose(0, 1).unsqueeze(0)
        vi = qkv[s:e, 2].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi)
        out[s:e] = oi.squeeze(0).transpose(0, 1)
    return out


def install_direct3d_v3_shims() -> dict[str, str]:
    """
    Install zero-build runtime shims for missing C++ extensions (`cudavoxelize`, `fillhole_cpp`,
    and `flash_attn` if missing/incomplete). Leaves native compiled extensions untouched.
    """
    status: dict[str, str] = {}

    # 1. cudavoxelize
    has_native_vox = False
    try:
        if importlib.util.find_spec("cudavoxelize") is not None:
            import cudavoxelize as _cv  # type: ignore[import-not-found]

            has_native_vox = callable(getattr(_cv, "voxelize", None))
    except Exception:
        has_native_vox = False

    if not has_native_vox:
        cv_mod = types.ModuleType("cudavoxelize")

        def _voxelize_dispatch(vertices: Any, faces: Any, voxel: Any, num_triangles: int, dim: int = 512, threshold: float = 8.0) -> None:
            if TRITON_AVAILABLE and vertices.is_cuda:
                try:
                    voxelize_mesh_triton(vertices, faces, voxel, num_triangles, dim, threshold)
                    return
                except Exception:
                    pass
            voxelize_mesh_pure_torch(vertices, faces, voxel, num_triangles, dim, threshold)

        cv_mod.voxelize = _voxelize_dispatch  # type: ignore[attr-defined]
        sys.modules["cudavoxelize"] = cv_mod
        status["cudavoxelize"] = "triton_with_torch_fallback" if TRITON_AVAILABLE else "pure_torch"
    else:
        status["cudavoxelize"] = "native"

    # 2. fillhole_cpp
    has_native_fh = False
    try:
        if importlib.util.find_spec("fillhole_cpp") is not None:
            import fillhole_cpp as _fh  # type: ignore[import-not-found]

            has_native_fh = callable(getattr(_fh, "fill_hole_cuda1", None))
    except Exception:
        has_native_fh = False

    if not has_native_fh and torch is not None:
        fh_mod = types.ModuleType("fillhole_cpp")
        fh_mod.fill_hole_cuda1 = lambda *args, **kwargs: torch.empty((0, 3), dtype=torch.int32)  # type: ignore[attr-defined]
        sys.modules["fillhole_cpp"] = fh_mod
        status["fillhole_cpp"] = "noop_shim"
    else:
        status["fillhole_cpp"] = "native"

    # 3. flash_attn (only shim if missing or broken)
    has_flash = False
    try:
        import flash_attn as _fa  # type: ignore[import-not-found]

        has_flash = (
            callable(getattr(_fa, "flash_attn_varlen_kvpacked_func", None))
            and callable(getattr(_fa, "flash_attn_varlen_qkvpacked_func", None))
            and callable(getattr(_fa, "flash_attn_qkvpacked_func", None))
        )
    except Exception:
        has_flash = False

    if not has_flash:
        fa_mod = types.ModuleType("flash_attn")
        fa_mod.flash_attn_varlen_kvpacked_func = _sdpa_varlen_kvpacked  # type: ignore[attr-defined]
        fa_mod.flash_attn_varlen_qkvpacked_func = _sdpa_varlen_qkvpacked  # type: ignore[attr-defined]
        fa_mod.flash_attn_qkvpacked_func = _sdpa_qkvpacked  # type: ignore[attr-defined]
        sys.modules["flash_attn"] = fa_mod
        status["flash_attn"] = "sdpa_shim"
    else:
        status["flash_attn"] = "native"

    return status


def mesh_patches_adaptive(
    vertices: Any,
    faces: Any,
    resolution: int,
    patch_size: int,
    max_tokens: int,
    fit_large_meshes: bool = True,
) -> tuple[Any, float]:
    """
    Compute sparse occupied patch coordinates for `resolution`, automatically scaling down
    oversized coarse meshes if `fit_large_meshes=True`, or raising a clear error if `False`.
    Returns `(coords, scale_used)`.
    """
    assert torch is not None
    from direct3d.utils.voxelization import voxelize  # type: ignore[import-not-found]

    low, high = vertices.amin(dim=0), vertices.amax(dim=0)
    extent = (high - low).max()
    if float(extent) <= 0.0:
        raise ValueError("The coarse mesh has zero extent.")
    centered = (vertices - (low + high) / 2.0) * (1.9 / extent)
    scale = 1.0
    while True:
        coords = voxelize(centered * scale, faces, dim=int(resolution), patch_size=int(patch_size))
        n_tok = int(len(coords))
        if n_tok == 0:
            raise ValueError(f"No occupied patches at resolution {resolution}.")
        if n_tok <= int(max_tokens):
            return coords, scale
        if not fit_large_meshes:
            raise ValueError(
                f"Coarse mesh needs {n_tok} tokens at resolution {resolution}, exceeding max_tokens={max_tokens}. "
                "Enable fit_large_meshes or increase max_tokens."
            )
        if scale < 1.0 / max(1, int(resolution) - 1):
            raise ValueError(f"At resolution {resolution} the mesh still needs {n_tok} > {max_tokens} tokens.")
        # Step faster when far over budget, then fine-step near threshold
        factor = 0.95 if n_tok > int(max_tokens * 1.35) else 0.99
        scale *= factor


# ---------------------------------------------------------------------------
# Window Caching, Guidance Modes (CFG / APG / FDG) & Multiview Sampling
# ---------------------------------------------------------------------------


def forward_with_cached_windows(
    model: Any,
    x: Any,
    index: Any,
    layout: list[slice],
    t: Any,
    cond: Any,
    window_cache: tuple[Any, Any],
) -> Any:
    """Run SparseDiT forward reusing precomputed shifted-window partitions."""
    x, t = x.to(cond.dtype), t.to(cond.dtype)
    t_emb = model.t_embedder(t * model.time_factor)
    s = model.x_embedder(x)
    c = t_emb
    skip_value_list = []
    for layer, block in enumerate(model.blocks):
        skip_value = None if layer <= model.depth // 2 else skip_value_list.pop()
        s = block(
            s,
            c,
            cond,
            index,
            layout,
            skip_value=skip_value,
            window_size=model.window_size,
            window_cache=window_cache,
        )
        if layer < model.depth // 2:
            skip_value_list.append(s)
    x_patches = x.reshape(-1, model.patch_size**3, model.in_channels)
    x_dec = model.decoder(x_patches, s, index, window_cache=window_cache, window_size=model.window_size).squeeze(-1)
    return x_dec.clamp(-1.0, 1.0)


def apply_guidance(
    x_cond: Any,
    x_uncond: Any,
    guidance_scale: float,
    guidance_method: str = "cfg",
    patch_size: int = 8,
) -> Any:
    """Apply CFG, APG (Adaptive Projected Guidance), or FDG (Frequency-Decoupled Guidance)."""
    assert torch is not None
    s = float(guidance_scale)
    method = str(guidance_method or "cfg").lower().strip()
    diff = x_cond - x_uncond

    if method == "apg":
        # Project diff onto x_cond unit direction to dampen oversaturation
        eps = 1e-8
        norm_c = torch.linalg.norm(x_cond, dim=-1, keepdim=True).clamp(min=eps)
        u = x_cond / norm_c
        d_par = (diff * u).sum(dim=-1, keepdim=True) * u
        d_orth = diff - d_par
        d_guided = d_orth + 0.25 * d_par
        # Gentle norm clamping relative to diff norm
        max_norm = torch.linalg.norm(diff, dim=-1, keepdim=True) * 2.5 + eps
        g_norm = torch.linalg.norm(d_guided, dim=-1, keepdim=True).clamp(min=eps)
        scale_clamp = torch.clamp(max_norm / g_norm, max=1.0)
        return x_cond + (s - 1.0) * d_guided * scale_clamp

    if method == "fdg":
        # Separate low-frequency coarse patch mean vs high-frequency voxel detail
        p = int(patch_size)
        if diff.ndim == 2 and diff.shape[1] == p**3:
            d_vol = diff.view(-1, 1, p, p, p)
            d_low = F.avg_pool3d(d_vol, kernel_size=3, stride=1, padding=1).view_as(diff)
            d_high = diff - d_low
            return x_uncond + s * d_low + (s * 1.35) * d_high
        return x_uncond + s * diff

    return x_uncond + s * diff


def compute_patch_normals(
    vertices: Any,
    faces: Any,
    coords: Any,
    resolution: int,
    patch_size: int,
) -> Any:
    """
    Estimate outward unit normal vector `[N_patches, 3]` for each occupied sparse patch
    from coarse-mesh face normals, with radial fallback for patches without face centroids.
    """
    assert torch is not None
    dev = coords.device
    n_patches = int(len(coords))
    grid_dim = max(1, int(resolution) // int(patch_size))

    verts = vertices.to(device=dev, dtype=torch.float32)
    tris = faces.to(device=dev, dtype=torch.long)
    low, high = verts.amin(dim=0), verts.amax(dim=0)
    center = 0.5 * (low + high)
    extent = (high - low).max().clamp(min=1e-6)
    norm_verts = (verts - center) * (1.9 / extent) * 0.5  # in [-0.475, 0.475]

    v0 = norm_verts[tris[:, 0]]
    v1 = norm_verts[tris[:, 1]]
    v2 = norm_verts[tris[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn_norm = torch.linalg.norm(face_normals, dim=-1, keepdim=True).clamp(min=1e-8)
    face_normals = face_normals / fn_norm
    centroids = (v0 + v1 + v2) / 3.0

    # Map centroids to patch coordinates [0, grid_dim-1]
    p_coords = torch.floor((centroids + 0.5) * float(grid_dim)).long().clamp(0, grid_dim - 1)
    c_keys = (p_coords[:, 0] * grid_dim + p_coords[:, 1]) * grid_dim + p_coords[:, 2]

    patch_ijk = coords[:, 1:4].long().clamp(0, grid_dim - 1)
    p_keys = (patch_ijk[:, 0] * grid_dim + patch_ijk[:, 1]) * grid_dim + patch_ijk[:, 2]
    sorted_p, order = torch.sort(p_keys)
    pos = torch.searchsorted(sorted_p, c_keys).clamp(max=max(0, n_patches - 1))
    matched = (n_patches > 0) & (sorted_p[pos] == c_keys)

    patch_normals = torch.zeros((n_patches, 3), device=dev, dtype=torch.float32)
    if matched.any():
        target_idx = order[pos[matched]]
        patch_normals.index_add_(0, target_idx, face_normals[matched])

    norms = torch.linalg.norm(patch_normals, dim=-1, keepdim=True)
    # Radial fallback for patches with zero accumulated normal
    patch_centers = (patch_ijk.float() + 0.5) / float(grid_dim) - 0.5
    radial_norm = torch.linalg.norm(patch_centers, dim=-1, keepdim=True).clamp(min=1e-6)
    radial_dir = patch_centers / radial_norm
    patch_normals = torch.where(norms > 1e-5, patch_normals / norms.clamp(min=1e-5), radial_dir)
    return patch_normals


@torch.inference_mode()
def sample_x0_advanced(
    model: Any,
    latents_index: Any,
    cond_views: dict[str, Any],
    generator: Any,
    *,
    patch_size: int,
    steps: int,
    guidance_scale: float,
    cfg_interval: tuple[float, float] = (0.1, 0.9),
    guidance_method: str = "cfg",
    speed_mode: str = "lossless",
    skip_guidance_last_steps: int = 0,
    multiview_speed: str = "fast",
    patch_normals: Any = None,
    resolution: int | None = None,
    clamp_range: tuple[float, float] = (-1.0, 1.0),
    on_sampling_step: Any = None,
) -> Any:
    """
    Enhanced Direct3D-V3 x0-prediction sampler with:
    - Cached window partitions (computed once per stage instead of every step)
    - Speed modes (`off`, `lossless`, `fast`, `fastest`)
    - `skip_guidance_last_steps`
    - `cfg` / `apg` / `fdg` guidance
    - Single-view & 6-view directional patch-normal blending
    """
    assert torch is not None
    from direct3d.models.sparse_atten import calc_window_partition  # type: ignore[import-not-found]

    clamp_min, clamp_max = clamp_range
    device = latents_index.device
    num_tokens = len(latents_index)
    latents = torch.randn(
        (num_tokens, patch_size**3 * model.out_channels),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    layout = [slice(0, num_tokens)]

    # Precompute window partitions once for the stage
    win_cache = (
        calc_window_partition(latents_index, model.window_size, 0),
        calc_window_partition(latents_index, model.window_size, model.shift_window),
    )

    sp_mode = str(speed_mode or "lossless").lower().strip()
    eff_skip_last = max(0, int(skip_guidance_last_steps))
    eff_cfg_interval = tuple(cfg_interval)
    if sp_mode == "fast":
        eff_skip_last = max(eff_skip_last, 1 if steps >= 4 else 0)
    elif sp_mode == "fastest":
        eff_skip_last = max(eff_skip_last, 2 if steps >= 5 else (1 if steps >= 3 else 0))
        eff_cfg_interval = (max(0.15, cfg_interval[0]), min(0.80, cfg_interval[1]))

    view_names = list(cond_views.keys())
    if not view_names:
        raise ValueError("cond_views must contain at least one conditioning view.")

    # Compute per-patch view weights when multiple views are supplied
    view_weights: dict[str, Any] | None = None
    if len(view_names) > 1:
        if patch_normals is not None and len(patch_normals) == num_tokens:
            scores = []
            for vn in view_names:
                vdir = torch.tensor(VIEW_DIRECTIONS[vn], device=device, dtype=torch.float32)
                scores.append((patch_normals * vdir[None, :]).sum(dim=-1))
            logits = torch.stack(scores, dim=-1) * 4.0  # temperature=4.0
            probs = torch.softmax(logits, dim=-1)  # [N, V]
            view_weights = {vn: probs[:, idx : idx + 1] for idx, vn in enumerate(view_names)}
        else:
            w_val = 1.0 / float(len(view_names))
            view_weights = {vn: w_val for vn in view_names}

    first_cond = cond_views[view_names[0]]
    uncond = torch.zeros_like(first_cond) if guidance_scale > 1.0 else None
    timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

    for i in range(steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        timestep = t.reshape(1).to(latents.dtype)

        low, high = eff_cfg_interval
        in_interval = (float(t.item()) < high) and (low == 0.0 or float(t.item()) > low)
        do_cfg = (uncond is not None) and (i < steps - eff_skip_last) and in_interval

        if len(view_names) == 1:
            x_cond = forward_with_cached_windows(model, latents, latents_index, layout, timestep, first_cond, win_cache)
            if do_cfg:
                x_uncond = forward_with_cached_windows(model, latents, latents_index, layout, timestep, uncond, win_cache)
                output = apply_guidance(x_cond, x_uncond, guidance_scale, guidance_method, patch_size)
            else:
                output = x_cond
        else:
            assert view_weights is not None
            if do_cfg and str(multiview_speed).lower().strip() == "fast":
                # Fast multiview: blend per-view conditional predictions, share 1 uncond pass
                x_cond_blend = torch.zeros_like(latents)
                for vn in view_names:
                    xv = forward_with_cached_windows(model, latents, latents_index, layout, timestep, cond_views[vn], win_cache)
                    x_cond_blend = x_cond_blend + xv * view_weights[vn]
                x_uncond = forward_with_cached_windows(model, latents, latents_index, layout, timestep, uncond, win_cache)
                output = apply_guidance(x_cond_blend, x_uncond, guidance_scale, guidance_method, patch_size)
            elif do_cfg:
                x_uncond = forward_with_cached_windows(model, latents, latents_index, layout, timestep, uncond, win_cache)
                output = torch.zeros_like(latents)
                for vn in view_names:
                    xv = forward_with_cached_windows(model, latents, latents_index, layout, timestep, cond_views[vn], win_cache)
                    gv = apply_guidance(xv, x_uncond, guidance_scale, guidance_method, patch_size)
                    output = output + gv * view_weights[vn]
            else:
                output = torch.zeros_like(latents)
                for vn in view_names:
                    xv = forward_with_cached_windows(model, latents, latents_index, layout, timestep, cond_views[vn], win_cache)
                    output = output + xv * view_weights[vn]

        output = output.clamp(clamp_min, clamp_max)
        noise = torch.randn(output.shape, device=device, dtype=output.dtype, generator=generator)
        latents = output * t_next + (1.0 - t_next) * noise
        if on_sampling_step is not None:
            on_sampling_step(resolution, i + 1, steps)

    return latents


# ---------------------------------------------------------------------------
# Surface Extraction: Marching Cubes & Sharp-Edges Sparse Dual Contouring
# ---------------------------------------------------------------------------


def _expand_sparse_sdf_keys(latent_sdf: Any, latent_index: Any, voxel_size: int, patch_size: int, mc_threshold: float) -> tuple[Any, Any]:
    assert torch is not None
    dev = latent_sdf.device
    origins = latent_index.long() * int(patch_size)
    i = torch.arange(int(patch_size) ** 3, device=dev)
    offsets = (i // int(patch_size) ** 2, i // int(patch_size) % int(patch_size), i % int(patch_size))
    strides = (voxel_size**2, voxel_size, 1)
    base = sum(origins[:, axis] * stride for axis, stride in enumerate(strides))
    delta = sum(offset * stride for offset, stride in zip(offsets, strides))
    keys = base[:, None] + delta
    valid = torch.ones_like(keys, dtype=torch.bool)
    for axis, offset in enumerate(offsets):
        coord = origins[:, axis, None] + offset
        valid &= (coord >= 0) & (coord < voxel_size - 1)
    keys, order = keys[valid].sort()
    if not len(keys):
        raise ValueError("No SDF samples fall inside the voxel grid.")
    sdf = latent_sdf.reshape_as(valid)[valid][order].float() - float(mc_threshold)
    return keys, sdf


def sparse_dc_infer(
    latent_sdf: Any,
    latent_index: Any,
    voxel_size: int,
    patch_size: int,
    mc_threshold: float = 0.0,
) -> tuple[Any, Any]:
    """
    Sparse Dual Contouring surface extraction for sharp edges (hard-surface / mechanical / armor).
    Solves a regularized QEF per sign-changing cell and emits quads across active grid edges.
    """
    assert torch is not None
    if len(latent_index) == 0:
        raise ValueError("The sparse SDF is empty.")

    dev = latent_sdf.device
    r = int(voxel_size)
    keys, sdf = _expand_sparse_sdf_keys(latent_sdf, latent_index, r, int(patch_size), mc_threshold)

    # Corner offsets in lexicographical (i, j, k) order:
    # c_idx = 4*dx + 2*dy + dz -> offset = dx*r^2 + dy*r + dz
    corner_offsets = torch.tensor(
        [0, 1, r, r + 1, r * r, r * r + 1, r * r + r, r * r + r + 1],
        device=dev,
        dtype=torch.long,
    )
    corner_coords = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        device=dev,
        dtype=torch.float32,
    )
    edge_pairs = torch.tensor(
        [
            [0, 1], [2, 3], [4, 5], [6, 7],  # Z edges
            [0, 2], [1, 3], [4, 6], [5, 7],  # Y edges
            [0, 4], [1, 5], [2, 6], [3, 7],  # X edges
        ],
        device=dev,
        dtype=torch.long,
    )

    active_cells_list = []
    active_corner_idx_list = []
    for origins in keys.split(131072):
        neighbors = origins[:, None] + corner_offsets[None, :]
        idx = torch.searchsorted(keys, neighbors).clamp_max_(len(keys) - 1)
        all_exist = (keys[idx] == neighbors).all(dim=1)
        c_sdf = sdf[idx]
        neg = c_sdf < 0.0
        has_sign_change = neg.any(dim=1) & (~neg.all(dim=1))
        mask = all_exist & has_sign_change
        if mask.any():
            active_cells_list.append(origins[mask])
            active_corner_idx_list.append(idx[mask])

    if not active_cells_list:
        raise ValueError("No surface crosses the requested dual contouring threshold.")

    cell_keys = torch.cat(active_cells_list, dim=0)
    cell_corner_idx = torch.cat(active_corner_idx_list, dim=0)
    # Sort cell_keys for fast neighbor lookup
    cell_keys, c_order = cell_keys.sort()
    cell_corner_idx = cell_corner_idx[c_order]
    cell_sdf = sdf[cell_corner_idx]  # [M, 8]
    m = len(cell_keys)

    # Estimate per-cell gradient from corner SDF differences along X, Y, Z
    gx = ((cell_sdf[:, 4] - cell_sdf[:, 0]) + (cell_sdf[:, 5] - cell_sdf[:, 1]) + (cell_sdf[:, 6] - cell_sdf[:, 2]) + (cell_sdf[:, 7] - cell_sdf[:, 3])) * 0.25
    gy = ((cell_sdf[:, 2] - cell_sdf[:, 0]) + (cell_sdf[:, 3] - cell_sdf[:, 1]) + (cell_sdf[:, 6] - cell_sdf[:, 4]) + (cell_sdf[:, 7] - cell_sdf[:, 5])) * 0.25
    gz = ((cell_sdf[:, 1] - cell_sdf[:, 0]) + (cell_sdf[:, 3] - cell_sdf[:, 2]) + (cell_sdf[:, 5] - cell_sdf[:, 4]) + (cell_sdf[:, 7] - cell_sdf[:, 6])) * 0.25
    cell_grad = torch.stack([gx, gy, gz], dim=-1)  # [M, 3]
    g_norm = torch.linalg.norm(cell_grad, dim=-1, keepdim=True).clamp(min=1e-6)
    cell_normal = cell_grad / g_norm  # [M, 3]

    # Compute sign-changing edge intersections in local [0, 1]^3 cell coordinates
    s0 = cell_sdf[:, edge_pairs[:, 0]]  # [M, 12]
    s1 = cell_sdf[:, edge_pairs[:, 1]]
    e_cross = (s0 < 0.0) != (s1 < 0.0)
    denom = s0 - s1
    safe_denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
    t_edge = (s0 / safe_denom).clamp(0.0, 1.0)  # [M, 12]

    p0 = corner_coords[edge_pairs[:, 0]]  # [12, 3]
    p1 = corner_coords[edge_pairs[:, 1]]
    p_cross = p0[None, :, :] + t_edge[:, :, None] * (p1 - p0)[None, :, :]  # [M, 12, 3]
    w_cross = e_cross.float().unsqueeze(-1)
    p_mean = (p_cross * w_cross).sum(dim=1) / w_cross.sum(dim=1).clamp(min=1.0)  # [M, 3]

    # Regularized QEF: (n n^T + lam * I) x = n (n^T p_mean) + lam * p_mean -> solves smoothly
    lam = 0.15
    n_outer = cell_normal.unsqueeze(-1) * cell_normal.unsqueeze(-2)  # [M, 3, 3]
    eye = torch.eye(3, device=dev, dtype=torch.float32).unsqueeze(0)
    A = n_outer + lam * eye
    rhs_scalar = (cell_normal * p_mean).sum(dim=-1, keepdim=True)
    b = cell_normal * rhs_scalar + lam * p_mean
    local_v = torch.linalg.solve(A, b).clamp(0.0, 1.0)  # [M, 3]

    cell_ijk = torch.stack((cell_keys // (r * r), (cell_keys // r) % r, cell_keys % r), dim=1).float()
    vertices = (cell_ijk + local_v) / float(r - 1) - 0.5

    # Emit quads for every interior grid edge (along X, Y, Z) with a sign change
    # whose 4 surrounding cells all exist in cell_keys
    faces_list = []
    strides_xyz = (r * r, r, 1)
    for axis in range(3):
        a1 = (axis + 1) % 3
        a2 = (axis + 2) % 3
        step_axis = strides_xyz[axis]
        step1 = strides_xyz[a1]
        step2 = strides_xyz[a2]

        # An edge along `axis` starting at `k` has 4 surrounding cells:
        # k, k - step1, k - step2, k - step1 - step2
        c0_k = cell_keys
        c1_k = c0_k - step1
        c2_k = c0_k - step1 - step2
        c3_k = c0_k - step2
        end_k = c0_k + step_axis

        idx1 = torch.searchsorted(cell_keys, c1_k).clamp_max(m - 1)
        idx2 = torch.searchsorted(cell_keys, c2_k).clamp_max(m - 1)
        idx3 = torch.searchsorted(cell_keys, c3_k).clamp_max(m - 1)
        k_end_idx = torch.searchsorted(keys, end_k).clamp_max(len(keys) - 1)
        k_start_idx = torch.searchsorted(keys, c0_k).clamp_max(len(keys) - 1)

        cells_ok = (cell_keys[idx1] == c1_k) & (cell_keys[idx2] == c2_k) & (cell_keys[idx3] == c3_k)
        edge_ok = (keys[k_start_idx] == c0_k) & (keys[k_end_idx] == end_k)
        s_start = sdf[k_start_idx]
        s_end = sdf[k_end_idx]
        flip = (s_start < 0.0) & (s_end >= 0.0)
        unflip = (s_start >= 0.0) & (s_end < 0.0)
        valid_e = cells_ok & edge_ok & (flip | unflip)

        if valid_e.any():
            v0 = torch.arange(m, device=dev, dtype=torch.long)[valid_e]
            v1 = idx1[valid_e]
            v2 = idx2[valid_e]
            v3 = idx3[valid_e]
            is_flip = flip[valid_e]
            q_a = torch.where(is_flip[:, None], torch.stack([v0, v1, v2], dim=1), torch.stack([v0, v2, v1], dim=1))
            q_b = torch.where(is_flip[:, None], torch.stack([v0, v2, v3], dim=1), torch.stack([v0, v3, v2], dim=1))
            faces_list.append(q_a)
            faces_list.append(q_b)

    if not faces_list:
        raise ValueError("Dual contouring produced zero valid quads.")
    faces = torch.cat(faces_list, dim=0).to(torch.int32)
    return vertices, faces


# ---------------------------------------------------------------------------
# Color Transfer, Mesh Decimation & Pipeline Orchestrator
# ---------------------------------------------------------------------------


def transfer_vertex_colors_from_coarse(
    refined_vertices: np.ndarray,
    coarse_vertices: np.ndarray,
    coarse_colors: np.ndarray,
) -> np.ndarray:
    """
    Project vertex colors from an aligned coarse mesh onto a refined Direct3D-V3 mesh
    by normalizing the coarse mesh into the same [-0.475, 0.475] bounding frame.
    """
    r_verts = np.asarray(refined_vertices, dtype=np.float32)
    c_verts = np.asarray(coarse_vertices, dtype=np.float32)
    c_cols = np.clip(np.asarray(coarse_colors, dtype=np.float32)[:, :3], 0.0, 1.0)

    if len(r_verts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(c_verts) == 0 or len(c_cols) != len(c_verts):
        return np.full((len(r_verts), 3), 0.78, dtype=np.float32)

    c_low = c_verts.min(axis=0)
    c_high = c_verts.max(axis=0)
    c_ext = float(np.max(c_high - c_low))
    if c_ext <= 1e-6:
        return np.full((len(r_verts), 3), float(np.mean(c_cols)), dtype=np.float32)
    c_norm = ((c_verts - 0.5 * (c_low + c_high)) * (1.9 / c_ext) * 0.5).astype(np.float32)

    # Match refined bounding box center/scale in case fit_large_meshes scaled it
    r_low = r_verts.min(axis=0)
    r_high = r_verts.max(axis=0)
    r_ext = float(np.max(r_high - r_low))
    if r_ext > 1e-6:
        r_norm = ((r_verts - 0.5 * (r_low + r_high)) * (0.95 / r_ext)).astype(np.float32)
    else:
        r_norm = r_verts

    try:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]

        tree = cKDTree(c_norm)
        _, nn_idx = tree.query(r_norm, k=1)
        return np.clip(c_cols[nn_idx], 0.0, 1.0).astype(np.float32)
    except Exception:
        pass

    # Pure-PyTorch chunked nearest neighbor lookup (zero OOM)
    if torch is not None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if len(c_norm) > 60000:
            step = max(1, len(c_norm) // 60000)
            c_sub = c_norm[::step]
            col_sub = c_cols[::step]
        else:
            c_sub = c_norm
            col_sub = c_cols
        c_t = torch.from_numpy(c_sub).to(dev)
        out_cols = np.empty((len(r_norm), 3), dtype=np.float32)
        chunk = 4096
        for start in range(0, len(r_norm), chunk):
            q_t = torch.from_numpy(r_norm[start : start + chunk]).to(dev)
            dists = torch.cdist(q_t, c_t)
            nn_i = dists.argmin(dim=1).cpu().numpy()
            out_cols[start : start + chunk] = col_sub[nn_i]
        return np.clip(out_cols, 0.0, 1.0).astype(np.float32)

    return np.full((len(r_verts), 3), 0.78, dtype=np.float32)


def decimate_mesh_safe(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Simplify dense extracted meshes using pyfqmr, Open3D, or vertex clustering fallback."""
    from .three_d_nodes import decimate_mesh_vertex_clustering

    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int32)
    cols = np.asarray(vertex_colors, dtype=np.float32) if vertex_colors is not None else None

    if target_faces <= 0 or len(tris) <= int(target_faces):
        return verts, tris, cols

    try:
        import pyfqmr  # type: ignore[import-not-found]

        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(verts.astype(np.float64), tris.astype(np.int32))
        simplifier.simplify_mesh(
            target_count=int(target_faces),
            aggressiveness=5,
            preserve_border=True,
            verbose=False,
        )
        sv, sf, _ = simplifier.getMesh()
        sv_f32 = np.asarray(sv, dtype=np.float32)
        sf_i32 = np.asarray(sf, dtype=np.int32)
        if cols is not None and len(sv_f32) > 0:
            sc = transfer_vertex_colors_from_coarse(sv_f32, verts, cols)
        else:
            sc = None
        return sv_f32, sf_i32, sc
    except Exception:
        pass

    try:
        import open3d as o3d  # type: ignore[import-not-found]

        m_o3d = o3d.geometry.TriangleMesh()
        m_o3d.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        m_o3d.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
        if cols is not None and len(cols) == len(verts):
            m_o3d.vertex_colors = o3d.utility.Vector3dVector(cols[:, :3].astype(np.float64))
        m_o3d = m_o3d.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        sv = np.asarray(m_o3d.vertices, dtype=np.float32)
        sf = np.asarray(m_o3d.triangles, dtype=np.int32)
        sc = np.asarray(m_o3d.vertex_colors, dtype=np.float32) if m_o3d.has_vertex_colors() else None
        if len(sv) > 0 and len(sf) > 0:
            return sv, sf, sc
    except Exception:
        pass

    return decimate_mesh_vertex_clustering(verts, tris, cols, int(target_faces))


def _prepare_view_image(
    pil_img: Image.Image,
    remover: Any,
    device: Any,
    remove_bg_fallback_fn: Any,
) -> tuple[Any, Image.Image]:
    """Ensure RGBA foreground cutout and convert to Direct3D-V3's input tensor."""
    from direct3d.utils.image import prepare_image  # type: ignore[import-not-found]

    rgba = pil_img
    has_alpha = (
        rgba.mode == "RGBA"
        and int(np.min(np.asarray(rgba)[..., 3])) < 250
    )
    if not has_alpha:
        if remover is not None:
            try:
                rgba = remover(rgba, device)
            except Exception:
                rgba = remove_bg_fallback_fn(rgba)
        else:
            rgba = remove_bg_fallback_fn(rgba)
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    return prepare_image(rgba), rgba


def run_direct3d_v3_pipeline(
    *,
    view_images: dict[str, Image.Image],
    coarse_vertices: np.ndarray,
    coarse_faces: np.ndarray,
    coarse_colors: np.ndarray | None,
    model_precision: str = "int8",
    resolution: int = 1024,
    start_from: str = "same_as_resolution",
    steps: int = 8,
    guidance_scale: float = 4.0,
    guidance_method: str = "cfg",
    skip_guidance_last_steps: int = 0,
    speed_mode: str = "lossless",
    multiview_speed: str = "fast",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    fit_large_meshes: bool = True,
    sharp_edges: bool = False,
    target_faces: int = 150000,
    color_transfer: str = "transfer_from_coarse",
    seed: int = 42,
    custom_model_path: str = "",
    image_encoder_override: str = "",
    hf_token: str = "",
    auto_download_weights: bool = True,
    output_mesh_path: Path,
) -> tuple[Path, dict[str, Any], str]:
    """
    Execute Direct3D-V3 single-stage or cascaded mesh refinement with 8 GB VRAM discipline:
    1. Load DINOv3 image encoder once -> encode all view images -> offload to CPU.
    2. For each active resolution stage, load ONLY that stage's denoiser (W8G32 INT8 or FP16),
       sample SDF, extract surface (Marching Cubes or Sparse Dual Contouring), and unload immediately.
    3. Decimate, optionally project coarse vertex colors or run Hunyuan3DPaint, and export mesh.
    """
    assert torch is not None
    from .three_d_nodes import (
        Ziggart3DError,
        _ensure_backend_installed,
        _extract_trimesh_arrays,
        compute_vertex_normals,
        export_mesh_file,
        remove_background_fallback,
    )

    if hf_token and hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token.strip()
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token.strip()

    if not torch.cuda.is_available():
        raise Ziggart3DError("Direct3D-V3 requires a CUDA-enabled GPU.")

    device = torch.device("cuda", torch.cuda.current_device())
    res = int(resolution)
    if res not in DIRECT3D_V3_STAGES:
        raise Ziggart3DError(f"Unsupported resolution {res}; expected one of {DIRECT3D_V3_STAGES}.")

    if str(start_from) == "same_as_resolution":
        active_stages = (res,)
    else:
        st_start = int(start_from)
        active_stages = tuple(s for s in DIRECT3D_V3_STAGES if st_start <= s <= res)
        if not active_stages:
            active_stages = (res,)

    variant = "fp16" if "fp16" in str(model_precision).lower() else "int8"
    use_triton_int8 = (variant == "int8") and (str(speed_mode).lower() != "off")
    dequant_to_fp16 = (variant == "int8") and (str(speed_mode).lower() == "off")

    # Ensure Direct3D-V3 SDK is installed and shims are registered BEFORE importing `direct3d`
    shim_status = install_direct3d_v3_shims()
    if not _ensure_backend_installed("direct3d", auto_install=True):
        raise Ziggart3DError(
            "DreamTechAI/Direct3D-V3 SDK could not be imported or cloned. "
            "Install it with: git clone https://github.com/DreamTechAI/Direct3D-V3.git"
        )

    from direct3d.models.image_encoder import DINOv3FeatureExtractor  # type: ignore[import-not-found]
    from direct3d.utils.image import BackgroundRemover  # type: ignore[import-not-found]
    from direct3d.utils.sparse_mc import sparse_mc_infer  # type: ignore[import-not-found]

    # Locate or download model directory
    clean_custom = str(custom_model_path or "").strip()
    if clean_custom and clean_custom.lower() not in ("true", "false"):
        local_cand = Path(clean_custom).expanduser()
        model_source: str | Path = local_cand if local_cand.is_dir() else clean_custom
    else:
        existing = find_existing_direct3d_v3_dir(variant, active_stages)
        if existing is not None:
            model_source = existing
        elif auto_download_weights:
            dl_info = download_direct3d_v3_files(variant, active_stages, hf_token=hf_token)
            model_source = Path(dl_info["local_dir"])
        else:
            model_source = DIRECT3D_V3_REPOS[variant]

    def _read_model_file(rel: str) -> Path:
        if isinstance(model_source, Path):
            return model_source / rel
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        return Path(hf_hub_download(str(model_source), rel, token=hf_token.strip() or None))

    model_index = json.loads(_read_model_file("model_index.json").read_text(encoding="utf-8"))
    encoder_opts = resolve_direct3d_image_encoder(model_source, model_index, image_encoder_override, hf_token)

    # 1. Foreground matting & DINOv3 feature extraction (offloaded before denoiser loads)
    needs_bg_removal = any(
        img.mode != "RGBA" or int(np.min(np.asarray(img)[..., 3])) >= 250
        for img in view_images.values()
    )
    remover = None
    if needs_bg_removal:
        try:
            remover = BackgroundRemover(local_files_only=False)
        except Exception:
            remover = None

    prepared_pixels: dict[str, Any] = {}
    primary_rgba: Image.Image | None = None
    for vn, pil_img in view_images.items():
        pix, rgba = _prepare_view_image(pil_img, remover, device, remove_background_fallback)
        prepared_pixels[vn] = pix
        if vn == "front" or primary_rgba is None:
            primary_rgba = rgba
    del remover
    _empty_cuda_cache()

    stage_configs = {
        st: json.loads(_read_model_file(f"{model_index[f'shape_transformers_{st}']}/config.json").read_text(encoding="utf-8"))
        for st in active_stages
    }

    dtype = torch.float16
    encoder = DINOv3FeatureExtractor(**encoder_opts, dtype=dtype, local_files_only=False).to(device)
    # Extract features per stage image_size, then immediately free DINOv3 from VRAM
    stage_conds: dict[int, dict[str, Any]] = {}
    try:
        for st in active_stages:
            img_sz = int(stage_configs[st]["image_size"])
            stage_conds[st] = {}
            with torch.autocast("cuda", dtype=dtype):
                for vn, pix in prepared_pixels.items():
                    stage_conds[st][vn] = encoder(pix, img_sz)
    finally:
        encoder.to("cpu")
        del encoder
        _empty_cuda_cache()

    # 2. Sequential stage-by-stage refinement (at most 1 denoiser in VRAM at a time)
    verts_t = torch.tensor(coarse_vertices, dtype=torch.float32, device=device)
    faces_t = torch.tensor(coarse_faces, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    token_counts: dict[int, int] = {}

    for st in active_stages:
        cfg = stage_configs[st]
        patch_size = int(cfg["denoiser"]["patch_size"])
        coords, _ = mesh_patches_adaptive(
            verts_t,
            faces_t,
            resolution=st,
            patch_size=patch_size,
            max_tokens=int(max_tokens),
            fit_large_meshes=bool(fit_large_meshes),
        )
        token_counts[st] = int(len(coords))

        patch_normals = None
        if len(view_images) > 1:
            patch_normals = compute_patch_normals(verts_t, faces_t, coords, st, patch_size)

        ckpt_path = _read_model_file(f"{model_index[f'shape_transformers_{st}']}/model.safetensors")
        denoiser = load_direct3d_v3_denoiser(
            ckpt_path,
            cfg["denoiser"],
            dtype=dtype,
            use_triton_int8=use_triton_int8,
            dequant_to_fp16=dequant_to_fp16,
        ).to(device)

        sampling = cfg["sampling"].copy()
        eff_steps = int(steps) if int(steps) > 0 else int(sampling.get("steps", 8))
        eff_cfg = float(guidance_scale)

        try:
            with torch.autocast("cuda", dtype=dtype):
                sdf = sample_x0_advanced(
                    denoiser,
                    coords.int(),
                    stage_conds[st],
                    generator,
                    patch_size=patch_size,
                    steps=eff_steps,
                    guidance_scale=eff_cfg,
                    cfg_interval=tuple(sampling.get("cfg_interval", (0.1, 0.9))),
                    guidance_method=guidance_method,
                    speed_mode=speed_mode,
                    skip_guidance_last_steps=skip_guidance_last_steps,
                    multiview_speed=multiview_speed,
                    patch_normals=patch_normals,
                    resolution=st,
                    clamp_range=tuple(sampling.get("clamp_range", (-1.0, 1.0))),
                )
        finally:
            denoiser.to("cpu")
            del denoiser
            _empty_cuda_cache()

        mc_thresh = float(cfg.get("mc_threshold", 0.0))
        if sharp_edges:
            verts_t, faces_t = sparse_dc_infer(sdf.float(), coords[:, 1:], st, patch_size, mc_threshold=mc_thresh)
        else:
            verts_t, faces_t = sparse_mc_infer(sdf.float(), coords[:, 1:], st, patch_size, mc_threshold=mc_thresh)
        del sdf, coords
        _empty_cuda_cache()

    ref_verts = verts_t.detach().cpu().numpy().astype(np.float32)
    ref_faces = faces_t.detach().cpu().numpy().astype(np.int32)
    del verts_t, faces_t
    _empty_cuda_cache()

    # 3. Color projection & decimation
    ct_mode = str(color_transfer or "transfer_from_coarse").lower().strip()
    v_cols: np.ndarray | None = None
    if ct_mode == "transfer_from_coarse" and coarse_colors is not None and len(coarse_colors) == len(coarse_vertices):
        v_cols = transfer_vertex_colors_from_coarse(ref_verts, coarse_vertices, coarse_colors)

    if target_faces > 0 and len(ref_faces) > int(target_faces):
        ref_verts, ref_faces, v_cols = decimate_mesh_safe(ref_verts, ref_faces, v_cols, int(target_faces))

    if v_cols is None or len(v_cols) != len(ref_verts):
        v_cols = np.full((len(ref_verts), 3), 0.78, dtype=np.float32)

    normals = compute_vertex_normals(ref_verts, ref_faces)

    # Optional sequential Hunyuan3DPaint texture pass (Direct3D-V3 is already completely unloaded)
    if ct_mode == "hunyuan3d_paint" and primary_rgba is not None:
        import trimesh as _tm  # type: ignore[import-not-found]
        from .three_d_nodes import (
            _install_zero_build_3d_shims,
            _patch_diffusers_trust_remote_code_for_hy3dgen,
        )

        vendor_root = Path(__file__).resolve().parent / ".cache" / "vendor_3d"
        _install_zero_build_3d_shims(vendor_root, auto_install=True)
        _ensure_backend_installed("hy3dgen", auto_install=True)
        paint_pipe = None
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # type: ignore[import-not-found]

            tm_obj = _tm.Trimesh(vertices=ref_verts, faces=ref_faces, process=False)
            _patch_diffusers_trust_remote_code_for_hy3dgen()
            paint_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
            if hasattr(paint_pipe, "enable_model_cpu_offload"):
                paint_pipe.enable_model_cpu_offload()
            painted = paint_pipe(mesh=tm_obj, image=primary_rgba)
            if hasattr(painted, "export") and output_mesh_path.suffix.lower() == ".glb":
                painted.export(str(output_mesh_path))
                painted_arr = _extract_trimesh_arrays(painted)
                mesh_data = {
                    **painted_arr,
                    "path": str(output_mesh_path),
                }
                info = {
                    "variant": variant,
                    "stages": list(active_stages),
                    "tokens_per_stage": token_counts,
                    "vertices": int(len(mesh_data["vertices"])),
                    "faces": int(len(mesh_data["faces"])),
                    "sharp_edges": bool(sharp_edges),
                    "color_transfer": ct_mode,
                    "views": list(view_images.keys()),
                    "shims": shim_status,
                }
                return output_mesh_path, mesh_data, json.dumps(info, indent=2)
        finally:
            if paint_pipe is not None:
                del paint_pipe
            _empty_cuda_cache()

    export_mesh_file(output_mesh_path, ref_verts, ref_faces, v_cols, normals)
    mesh_data = {
        "vertices": ref_verts,
        "faces": ref_faces,
        "vertex_colors": np.clip(v_cols, 0.0, 1.0),
        "normals": normals,
        "path": str(output_mesh_path),
    }
    info = {
        "variant": variant,
        "stages": list(active_stages),
        "tokens_per_stage": token_counts,
        "vertices": int(len(ref_verts)),
        "faces": int(len(ref_faces)),
        "sharp_edges": bool(sharp_edges),
        "color_transfer": ct_mode,
        "views": list(view_images.keys()),
        "shims": shim_status,
    }
    return output_mesh_path, mesh_data, json.dumps(info, indent=2)
