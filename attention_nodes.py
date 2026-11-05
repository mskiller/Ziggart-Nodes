from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    import comfy.ldm.modules.attention as comfy_attention
except Exception:
    comfy_attention = None


# ---------------------------------------------------------------------------
# Startup Detection & Registry of Available Attention Backends
# ---------------------------------------------------------------------------

MAIN_BACKEND_ORDER = [
    "auto",
    "sage",
    "flash",
    "xformers",
    "pytorch",
    "sub_quad",
    "split",
    "basic",
]

_DETECTION_CACHE: dict[str, Any] | None = None
_ORIGINAL_GLOBAL_ATTN: dict[str, Callable[..., Any]] = {}
_WARNED_FALLBACKS: set[str] = set()


def _probe_sage_attention() -> dict[str, Any]:
    """Probe sageattention module and its available sub-kernels."""
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "funcs": {},
        "sub_kernels": [],
    }
    try:
        import sageattention  # type: ignore

        sage_main = getattr(sageattention, "sageattn", None)
        if callable(sage_main):
            info["available"] = True
            info["version"] = getattr(sageattention, "__version__", "installed")
            info["funcs"]["sage_auto"] = sage_main
            info["sub_kernels"].append("sage_auto")

            kernel_map = [
                ("sage_cuda_fp16", "sageattn_qk_int8_pv_fp16_cuda"),
                ("sage_cuda_fp8", "sageattn_qk_int8_pv_fp8_cuda"),
                ("sage_cuda_fp8_sm90", "sageattn_qk_int8_pv_fp8_cuda_sm90"),
                ("sage_triton", "sageattn_qk_int8_pv_fp16_triton"),
            ]
            for sub_name, attr_name in kernel_map:
                fn = getattr(sageattention, attr_name, None)
                if callable(fn):
                    info["funcs"][sub_name] = fn
                    info["sub_kernels"].append(sub_name)
    except Exception:
        pass
    return info


def _probe_flash_attention() -> dict[str, Any]:
    """Probe FlashAttention v2 / v3 packages."""
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "funcs": {},
        "sub_kernels": [],
    }
    # FlashAttention-3 (flash_attn_interface)
    try:
        import flash_attn_interface  # type: ignore

        fn_v3 = getattr(flash_attn_interface, "flash_attn_func", None)
        if callable(fn_v3):
            info["available"] = True
            info["version"] = getattr(flash_attn_interface, "__version__", "3.x")
            info["funcs"]["flash_v3"] = fn_v3
            info["sub_kernels"].append("flash_v3")
    except Exception:
        pass

    # FlashAttention-2 (flash_attn)
    try:
        import flash_attn  # type: ignore

        fn_v2 = getattr(flash_attn, "flash_attn_func", None)
        if callable(fn_v2):
            info["available"] = True
            if info["version"] is None:
                info["version"] = getattr(flash_attn, "__version__", "2.x")
            info["funcs"]["flash_v2"] = fn_v2
            info["sub_kernels"].append("flash_v2")
    except Exception:
        pass

    return info


def _probe_xformers() -> dict[str, Any]:
    """Probe xFormers memory_efficient_attention."""
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "func": None,
    }
    try:
        import xformers  # type: ignore
        import xformers.ops  # type: ignore

        fn = getattr(xformers.ops, "memory_efficient_attention", None)
        if callable(fn):
            info["available"] = True
            info["version"] = getattr(xformers, "__version__", "installed")
            info["func"] = fn
    except Exception:
        pass
    return info


def _probe_pytorch_sdpa() -> dict[str, Any]:
    """Probe PyTorch scaled_dot_product_attention and its SDPA sub-backends."""
    info: dict[str, Any] = {
        "available": hasattr(F, "scaled_dot_product_attention"),
        "version": torch.__version__,
        "sub_kernels": [],
        "sdp_backends": {},
    }
    if not info["available"]:
        return info

    info["sub_kernels"].append("sdpa_auto")

    try:
        from torch.nn.attention import SDPBackend

        backend_candidates = [
            ("sdpa_flash", "FLASH_ATTENTION"),
            ("sdpa_mem_efficient", "EFFICIENT_ATTENTION"),
            ("sdpa_cudnn", "CUDNN_ATTENTION"),
            ("sdpa_math", "MATH"),
        ]
        for sub_name, enum_attr in backend_candidates:
            enum_val = getattr(SDPBackend, enum_attr, None)
            if enum_val is not None:
                info["sdp_backends"][sub_name] = enum_val
                info["sub_kernels"].append(sub_name)
    except Exception:
        # Older PyTorch 2.0-2.2 fallback via torch.backends.cuda.sdp_kernel
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            for sub_name in ("sdpa_flash", "sdpa_mem_efficient", "sdpa_math"):
                info["sub_kernels"].append(sub_name)

    return info


def detect_available_attentions(force_refresh: bool = False) -> dict[str, Any]:
    """
    Detect all attention backends and sub-kernels available on the system.
    Cached at ComfyUI startup unless force_refresh=True.
    """
    global _DETECTION_CACHE
    if _DETECTION_CACHE is not None and not force_refresh:
        return _DETECTION_CACHE

    sage_info = _probe_sage_attention()
    flash_info = _probe_flash_attention()
    xformers_info = _probe_xformers()
    pytorch_info = _probe_pytorch_sdpa()

    backends: list[str] = ["auto"]
    if sage_info["available"]:
        backends.append("sage")
    if flash_info["available"]:
        backends.append("flash")
    if xformers_info["available"]:
        backends.append("xformers")
    if pytorch_info["available"]:
        backends.append("pytorch")

    # Built-in ComfyUI / fallback attentions are always available
    backends.extend(["sub_quad", "split", "basic"])

    sub_kernels: list[str] = ["auto"]
    for sk in sage_info["sub_kernels"]:
        if sk not in sub_kernels:
            sub_kernels.append(sk)
    for sk in flash_info["sub_kernels"]:
        if sk not in sub_kernels:
            sub_kernels.append(sk)
    for sk in pytorch_info["sub_kernels"]:
        if sk not in sub_kernels:
            sub_kernels.append(sk)

    _DETECTION_CACHE = {
        "backends": backends,
        "sub_kernels": sub_kernels,
        "sage": sage_info,
        "flash": flash_info,
        "xformers": xformers_info,
        "pytorch": pytorch_info,
    }
    return _DETECTION_CACHE


def get_available_attention_list() -> list[str]:
    """Return the list of available main attention backends for INPUT_TYPES."""
    return list(detect_available_attentions()["backends"])


def get_available_sub_kernel_list() -> list[str]:
    """Return the list of available attention sub-kernels for INPUT_TYPES."""
    return list(detect_available_attentions()["sub_kernels"])


# ---------------------------------------------------------------------------
# Tensor Layout & Attention Implementations
# ---------------------------------------------------------------------------

def _unwrap_comfy_attn(fn: Callable[..., Any] | None) -> Callable[..., Any] | None:
    """Return the underlying unwrapped function if decorated by ComfyUI's @wrap_attn."""
    if fn is None:
        return None
    return getattr(fn, "__wrapped__", fn)


def _strip_override_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of kwargs without 'optimized_attention_override' in transformer_options."""
    if "transformer_options" not in kwargs or not isinstance(kwargs["transformer_options"], dict):
        return kwargs
    t_opts = kwargs["transformer_options"]
    if "optimized_attention_override" not in t_opts:
        return kwargs
    clean_kwargs = dict(kwargs)
    clean_t_opts = dict(t_opts)
    clean_t_opts.pop("optimized_attention_override", None)
    clean_kwargs["transformer_options"] = clean_t_opts
    return clean_kwargs


def _to_b_h_n_d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    skip_reshape: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Normalize (q, k, v) into 4D (B, H, N, D) layout."""
    if skip_reshape:
        b, _, _, dim_head = q.shape
        return q, k, v, b, dim_head
    b, _, dim_total = q.shape
    dim_head = dim_total // heads
    q_4d = q.view(b, -1, heads, dim_head).transpose(1, 2)
    k_4d = k.view(b, -1, heads, dim_head).transpose(1, 2)
    v_4d = v.view(b, -1, heads, dim_head).transpose(1, 2)
    return q_4d, k_4d, v_4d, b, dim_head


def _from_b_h_n_d(
    out: torch.Tensor,
    b: int,
    heads: int,
    dim_head: int,
    skip_output_reshape: bool,
) -> torch.Tensor:
    """Format 4D (B, H, N, D) output into ComfyUI's expected output shape."""
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def _prepare_mask_for_sdpa(
    mask: torch.Tensor | None,
    b: int,
    heads: int,
    q_len: int,
     dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize ComfyUI attention mask for F.scaled_dot_product_attention."""
    if mask is None:
        return None
    if mask.device != device:
        mask = mask.to(device=device)
    if mask.dtype != torch.bool and mask.dtype != dtype:
        mask = mask.to(dtype=dtype)
    if mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3:
        if mask.shape[0] == b * heads:
            return mask.view(b, heads, mask.shape[-2], mask.shape[-1])
        if mask.shape[0] == b:
            return mask.unsqueeze(1)
        return mask.view(b, -1, mask.shape[-2], mask.shape[-1])
    return mask


@contextlib.contextmanager
def _sdpa_kernel_context(sub_kernel: str):
    """Context manager to restrict PyTorch SDPA to a chosen sub-kernel when requested."""
    if not sub_kernel or sub_kernel in ("auto", "sdpa_auto") or not sub_kernel.startswith("sdpa_"):
        yield
        return

    det = detect_available_attentions()
    sdp_backends = det["pytorch"].get("sdp_backends", {})

    if sub_kernel in sdp_backends:
        try:
            from torch.nn.attention import sdpa_kernel

            with sdpa_kernel(sdp_backends[sub_kernel]):
                yield
            return
        except Exception:
            pass

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        kwargs = {
            "enable_flash": sub_kernel == "sdpa_flash",
            "enable_mem_efficient": sub_kernel == "sdpa_mem_efficient",
            "enable_math": sub_kernel == "sdpa_math",
        }
        try:
            with torch.backends.cuda.sdp_kernel(**kwargs):
                yield
            return
        except Exception:
            pass

    yield


def _attention_basic_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    attn_precision: Any = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Basic matrix-multiplication attention (uses ComfyUI's unwrapped attention_basic or pure PyTorch)."""
    if comfy_attention is not None and hasattr(comfy_attention, "attention_basic"):
        fn = _unwrap_comfy_attn(comfy_attention.attention_basic)
        if callable(fn):
            clean_kwargs = _strip_override_from_kwargs(kwargs)
            return fn(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **clean_kwargs,
            )

    q_4d, k_4d, v_4d, b, dim_head = _to_b_h_n_d(q, k, v, heads, skip_reshape)
    scale = dim_head**-0.5
    scores = torch.matmul(q_4d.float(), k_4d.float().transpose(-2, -1)) * scale
    if mask is not None:
        m = _prepare_mask_for_sdpa(mask, b, heads, q_4d.shape[2], scores.dtype, scores.device)
        if m is not None:
            scores = scores + m
    attn = torch.softmax(scores, dim=-1).to(v_4d.dtype)
    out = torch.matmul(attn, v_4d)
    return _from_b_h_n_d(out, b, heads, dim_head, skip_output_reshape)


def _attention_pytorch_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    attn_precision: Any = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    sub_kernel: str = "auto",
    **kwargs: Any,
) -> torch.Tensor:
    """PyTorch scaled_dot_product_attention with optional SDPA kernel selection."""
    # If default SDPA and ComfyUI's native attention_pytorch is available, we can use SDPA directly
    q_4d, k_4d, v_4d, b, dim_head = _to_b_h_n_d(q, k, v, heads, skip_reshape)
    sdpa_mask = _prepare_mask_for_sdpa(mask, b, heads, q_4d.shape[2], q_4d.dtype, q_4d.device)

    with _sdpa_kernel_context(sub_kernel):
        out = F.scaled_dot_product_attention(
            q_4d,
            k_4d,
            v_4d,
            attn_mask=sdpa_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    return _from_b_h_n_d(out, b, heads, dim_head, skip_output_reshape)


def _attention_sage_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    attn_precision: Any = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    sub_kernel: str = "auto",
    **kwargs: Any,
) -> torch.Tensor:
    """SageAttention implementation supporting auto, cuda_fp16, cuda_fp8, sm90, and triton sub-kernels."""
    if mask is not None:
        # SageAttention does not accept arbitrary attention masks; fall back to PyTorch SDPA
        return _attention_pytorch_impl(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    det = detect_available_attentions()
    sage_funcs = det["sage"]["funcs"]
    if not sage_funcs:
        raise RuntimeError("SageAttention is not installed or available on this system.")

    if sub_kernel in sage_funcs:
        sage_fn = sage_funcs[sub_kernel]
    else:
        sage_fn = sage_funcs.get("sage_auto") or next(iter(sage_funcs.values()))

    in_dtype = v.dtype
    if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

    q_4d, k_4d, v_4d, b, dim_head = _to_b_h_n_d(q, k, v, heads, skip_reshape)

    # Specific CUDA / Triton sub-kernels require contiguous tensors
    if sub_kernel and sub_kernel != "auto" and sub_kernel != "sage_auto":
        q_4d = q_4d.contiguous()
        k_4d = k_4d.contiguous()
        v_4d = v_4d.contiguous()

    out = sage_fn(q_4d, k_4d, v_4d, tensor_layout="HND", is_causal=False)
    out = out.to(in_dtype)
    return _from_b_h_n_d(out, b, heads, dim_head, skip_output_reshape)


def _attention_flash_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    attn_precision: Any = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    sub_kernel: str = "auto",
    **kwargs: Any,
) -> torch.Tensor:
    """FlashAttention v2/v3 implementation expecting (B, N, H, D) layout."""
    if mask is not None:
        return _attention_pytorch_impl(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    det = detect_available_attentions()
    flash_funcs = det["flash"]["funcs"]
    if not flash_funcs:
        raise RuntimeError("FlashAttention is not installed or available on this system.")

    if sub_kernel in flash_funcs:
        flash_fn = flash_funcs[sub_kernel]
    else:
        flash_fn = flash_funcs.get("flash_v3") or flash_funcs.get("flash_v2") or next(iter(flash_funcs.values()))

    in_dtype = v.dtype
    if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

    if skip_reshape:
        # Incoming is (B, H, N, D) -> transpose to (B, N, H, D)
        b, _, _, dim_head = q.shape
        q_nhd = q.transpose(1, 2).contiguous()
        k_nhd = k.transpose(1, 2).contiguous()
        v_nhd = v.transpose(1, 2).contiguous()
    else:
        b, _, dim_total = q.shape
        dim_head = dim_total // heads
        q_nhd = q.view(b, -1, heads, dim_head).contiguous()
        k_nhd = k.view(b, -1, heads, dim_head).contiguous()
        v_nhd = v.view(b, -1, heads, dim_head).contiguous()

    res = flash_fn(q_nhd, k_nhd, v_nhd, dropout_p=0.0, causal=False)
    out_nhd = res[0] if isinstance(res, tuple) else res
    out_nhd = out_nhd.to(in_dtype)

    if skip_output_reshape:
        return out_nhd.transpose(1, 2)
    return out_nhd.reshape(b, -1, heads * dim_head)


def _attention_xformers_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    attn_precision: Any = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """xFormers memory_efficient_attention implementation."""
    det = detect_available_attentions()
    xf_fn = det["xformers"]["func"]
    if not callable(xf_fn):
        raise RuntimeError("xFormers is not installed or available on this system.")

    if skip_reshape:
        b, _, _, dim_head = q.shape
        q_nhd = q.permute(0, 2, 1, 3).contiguous()
        k_nhd = k.permute(0, 2, 1, 3).contiguous()
        v_nhd = v.permute(0, 2, 1, 3).contiguous()
    else:
        b, _, dim_total = q.shape
        dim_head = dim_total // heads
        q_nhd = q.view(b, -1, heads, dim_head).contiguous()
        k_nhd = k.view(b, -1, heads, dim_head).contiguous()
        v_nhd = v.view(b, -1, heads, dim_head).contiguous()

    attn_bias = None
    if mask is not None:
        attn_bias = _prepare_mask_for_sdpa(mask, b, heads, q_nhd.shape[1], q_nhd.dtype, q_nhd.device)
        if attn_bias is not None and attn_bias.shape[1] == 1 and heads > 1:
            attn_bias = attn_bias.expand(b, heads, attn_bias.shape[-2], attn_bias.shape[-1]).contiguous()

    out_nhd = xf_fn(q_nhd, k_nhd, v_nhd, attn_bias=attn_bias)

    if skip_output_reshape:
        return out_nhd.permute(0, 2, 1, 3)
    return out_nhd.reshape(b, -1, heads * dim_head)


def _attention_comfy_named_impl(
    attr_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    attn_precision: Any = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Call an unwrapped built-in ComfyUI attention function (sub_quad, split, basic) with fallback."""
    if comfy_attention is not None and hasattr(comfy_attention, attr_name):
        fn = _unwrap_comfy_attn(getattr(comfy_attention, attr_name))
        if callable(fn):
            clean_kwargs = _strip_override_from_kwargs(kwargs)
            return fn(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **clean_kwargs,
            )
    return _attention_basic_impl(
        q,
        k,
        v,
        heads,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        **kwargs,
    )


def make_attention_override(
    attention: str,
    sub_kernel: str = "auto",
    fallback_on_error: bool = True,
) -> Callable[..., torch.Tensor]:
    """
    Create an attention override callable compatible with both:
    - ComfyUI's `transformer_options["optimized_attention_override"](func, q, k, v, heads, ...)`
    - Direct `optimized_attention(q, k, v, heads, ...)` calls.
    """
    def _override(*args: Any, **kwargs: Any) -> torch.Tensor:
        if args and callable(args[0]) and not isinstance(args[0], torch.Tensor):
            orig_func = args[0]
            call_args = args[1:]
        else:
            orig_func = None
            call_args = args

        if len(call_args) >= 4:
            q, k, v, heads = call_args[:4]
            extra_pos = call_args[4:]
        else:
            q = kwargs.pop("q")
            k = kwargs.pop("k")
            v = kwargs.pop("v")
            heads = kwargs.pop("heads")
            extra_pos = ()

        # Unpack optional positional args (mask, attn_precision, skip_reshape, skip_output_reshape)
        mask = kwargs.pop("mask", extra_pos[0] if len(extra_pos) > 0 else None)
        attn_precision = kwargs.pop("attn_precision", extra_pos[1] if len(extra_pos) > 1 else None)
        skip_reshape = kwargs.pop("skip_reshape", extra_pos[2] if len(extra_pos) > 2 else False)
        skip_output_reshape = kwargs.pop("skip_output_reshape", extra_pos[3] if len(extra_pos) > 3 else False)

        try:
            if attention == "sage":
                return _attention_sage_impl(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    sub_kernel=sub_kernel,
                    **kwargs,
                )
            if attention == "flash":
                return _attention_flash_impl(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    sub_kernel=sub_kernel,
                    **kwargs,
                )
            if attention == "xformers":
                return _attention_xformers_impl(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            if attention == "pytorch":
                return _attention_pytorch_impl(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    sub_kernel=sub_kernel,
                    **kwargs,
                )
            if attention == "sub_quad":
                return _attention_comfy_named_impl(
                    "attention_sub_quad",
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            if attention == "split":
                return _attention_comfy_named_impl(
                    "attention_split",
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            if attention == "basic":
                return _attention_basic_impl(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )

            # "auto" or unknown -> delegate to original unwrapped function if present
            if callable(orig_func):
                unwrapped = _unwrap_comfy_attn(orig_func)
                clean_kwargs = _strip_override_from_kwargs(kwargs)
                return unwrapped(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **clean_kwargs,
                )
            return _attention_pytorch_impl(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        except Exception as exc:
            if not fallback_on_error:
                raise
            warn_key = f"{attention}:{sub_kernel}:{type(exc).__name__}"
            if warn_key not in _WARNED_FALLBACKS:
                _WARNED_FALLBACKS.add(warn_key)
                print(
                    f"[Ziggart Attention] Backend '{attention}' (sub_kernel='{sub_kernel}') "
                    f"raised {type(exc).__name__}: {exc}. Falling back to PyTorch SDPA."
                )
            if hasattr(F, "scaled_dot_product_attention"):
                return _attention_pytorch_impl(
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    sub_kernel="auto",
                    **kwargs,
                )
            return _attention_basic_impl(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

    return _override


def _apply_or_restore_global_override(
    override_fn: Callable[..., torch.Tensor] | None,
) -> bool:
    """Apply or restore global override on comfy.ldm.modules.attention."""
    if comfy_attention is None:
        return False

    for attr in ("optimized_attention", "optimized_attention_masked"):
        if attr not in _ORIGINAL_GLOBAL_ATTN and hasattr(comfy_attention, attr):
            _ORIGINAL_GLOBAL_ATTN[attr] = getattr(comfy_attention, attr)

    if override_fn is None:
        for attr, orig_fn in _ORIGINAL_GLOBAL_ATTN.items():
            setattr(comfy_attention, attr, orig_fn)
        return False

    for attr in ("optimized_attention", "optimized_attention_masked"):
        if hasattr(comfy_attention, attr):
            setattr(comfy_attention, attr, override_fn)
    return True


def _resolve_effective_sub_kernel(attention: str, sub_kernel: str) -> str:
    """Return the effective sub-kernel for the chosen main attention backend."""
    if sub_kernel == "auto":
        return "auto"
    if attention == "sage" and sub_kernel.startswith("sage_"):
        return sub_kernel
    if attention == "flash" and sub_kernel.startswith("flash_"):
        return sub_kernel
    if attention == "pytorch" and sub_kernel.startswith("sdpa_"):
        return sub_kernel
    return "auto"


def format_startup_banner() -> str:
    """Format a concise startup log string showing detected attention backends."""
    det = detect_available_attentions()
    backends_str = ", ".join(det["backends"])
    sub_str = ", ".join(det["sub_kernels"])
    return f"[Ziggart Attention] Detected backends: [{backends_str}] | Sub-kernels: [{sub_str}]"


# ---------------------------------------------------------------------------
# ComfyUI Node Definition
# ---------------------------------------------------------------------------

class ZiggartAttentionSelector:
    """
    Detects available attention backends on the system at ComfyUI startup
    and patches the loaded diffusion MODEL to use the selected attention implementation.
    """

    DESCRIPTION = (
        "Selects the attention implementation (SageAttention, FlashAttention, xFormers, "
        "PyTorch SDPA, Sub-Quad, Split, Basic, or ComfyUI Auto) used by the connected MODEL "
        "during generation. Place this node between your Diffusion Model Loader and KSampler."
    )
    CATEGORY = "Ziggart/Model"
    FUNCTION = "apply_attention"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "attention_info")
    OUTPUT_TOOLTIPS = (
        "The cloned MODEL patched with the selected attention implementation.",
        "Diagnostic summary of the active attention backend, sub-kernel, and system capabilities.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        available_backends = get_available_attention_list()
        available_sub_kernels = get_available_sub_kernel_list()
        default_backend = available_backends[0] if available_backends else "auto"

        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Connect the MODEL output from Load Checkpoint, Load Diffusion Model (UNETLoader), or LoRA Loader.",
                    },
                ),
                "attention": (
                    available_backends,
                    {
                        "default": default_backend,
                        "tooltip": "Main attention backend (only backends detected on this system at ComfyUI startup are listed).",
                    },
                ),
                "sub_kernel": (
                    available_sub_kernels,
                    {
                        "default": "auto",
                        "tooltip": (
                            "Optional sub-kernel for SageAttention (sage_cuda_fp16, sage_cuda_fp8, sage_triton) "
                            "or PyTorch SDPA (sdpa_flash, sdpa_mem_efficient, sdpa_cudnn, sdpa_math). "
                            "Leave on 'auto' for automatic selection."
                        ),
                    },
                ),
                "global_override": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When False (recommended), patches only the cloned MODEL via transformer_options. "
                            "When True, also overrides comfy.ldm.modules.attention.optimized_attention globally."
                        ),
                    },
                ),
                "fallback_on_error": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Automatically falls back to PyTorch SDPA if a layer uses an unsupported "
                            "head dimension, mask, or dtype for the chosen kernel."
                        ),
                    },
                ),
            }
        }

    def apply_attention(
        self,
        model: Any,
        attention: str = "auto",
        sub_kernel: str = "auto",
        global_override: bool = False,
        fallback_on_error: bool = True,
    ) -> tuple[Any, str]:
        det = detect_available_attentions()
        cloned_model = model.clone() if hasattr(model, "clone") else model

        effective_sub_kernel = _resolve_effective_sub_kernel(attention, sub_kernel)

        # Ensure model_options and transformer_options dictionaries exist and are isolated
        if hasattr(cloned_model, "model_options") and isinstance(cloned_model.model_options, dict):
            cloned_model.model_options = dict(cloned_model.model_options)
            t_opts = dict(cloned_model.model_options.get("transformer_options", {}))
            cloned_model.model_options["transformer_options"] = t_opts
        else:
            t_opts = None

        if attention == "auto":
            if t_opts is not None:
                t_opts.pop("optimized_attention_override", None)
            _apply_or_restore_global_override(None)
            global_state = "restored to ComfyUI default"
        else:
            override_fn = make_attention_override(
                attention=attention,
                sub_kernel=effective_sub_kernel,
                fallback_on_error=fallback_on_error,
            )
            if t_opts is not None:
                t_opts["optimized_attention_override"] = override_fn
            if global_override:
                _apply_or_restore_global_override(override_fn)
                global_state = "enabled (global + model)"
            else:
                _apply_or_restore_global_override(None)
                global_state = "disabled (per-model patch only)"

        sub_note = effective_sub_kernel
        if sub_kernel != "auto" and effective_sub_kernel == "auto":
            sub_note = f"auto (ignored '{sub_kernel}' because it does not match '{attention}')"

        info_lines = [
            f"Active Attention: {attention}",
            f"Active Sub-Kernel: {sub_note}",
            f"Scope: {global_state}",
            f"Fallback on Error: {'enabled' if fallback_on_error else 'disabled'}",
            f"System Detected Backends: {', '.join(det['backends'])}",
            f"System Detected Sub-Kernels: {', '.join(det['sub_kernels'])}",
        ]
        return (cloned_model, "\n".join(info_lines))


# Print detection summary once at ComfyUI startup
print(format_startup_banner())


NODE_CLASS_MAPPINGS = {
    "ZiggartAttentionSelector": ZiggartAttentionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartAttentionSelector": "Ziggart Attention Selector",
}
