from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass
import gc
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import zipfile


class EmbeddedLlamaError(RuntimeError):
    pass


DEFAULT_LLAMA_SERVER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bin", "llama.cpp"
)


def default_vlm_threads() -> int:
    """Returns a consistent default CPU thread count for GGUF VLM inference."""
    return max(1, (os.cpu_count() or 4) // 2)


def _norm_path_for_compare(path_str: str) -> str:
    cleaned = (path_str or "").strip()
    if not cleaned:
        return ""
    return os.path.normcase(os.path.abspath(cleaned))


def read_gguf_metadata(path: str) -> dict[str, Any]:
    """
    Lightweight, dependency-free GGUF v2/v3 header metadata reader.
    Extracts architecture, model name, chat_template presence, and CLIP projector fields
    without mapping model tensors into memory. Returns {} on missing/invalid files.
    """
    cleaned = (path or "").strip()
    if not cleaned or not os.path.isfile(cleaned):
        return {}

    scalar_sizes = {
        0: 1,   # uint8
        1: 1,   # int8
        2: 2,   # uint16
        3: 2,   # int16
        4: 4,   # uint32
        5: 4,   # int32
        6: 4,   # float32
        7: 1,   # bool
        10: 8,  # uint64
        11: 8,  # int64
        12: 8,  # float64
    }
    interesting_exact = {
        "general.architecture",
        "general.name",
        "clip.projector_type",
        "clip.has_vision_encoder",
        "clip.has_audio_encoder",
        "tokenizer.chat_template",
    }

    try:
        with open(cleaned, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
            if version not in (2, 3) or n_kv > 1_000_000:
                return {}

            def _read_str() -> str:
                n = struct.unpack("<Q", f.read(8))[0]
                if n > 10_000_000:
                    raise ValueError("GGUF string length exceeds safety bound")
                return f.read(n).decode("utf-8", errors="replace")

            def _skip_or_read(vtype: int, keep: bool) -> Any:
                if vtype in scalar_sizes:
                    raw = f.read(scalar_sizes[vtype])
                    if not keep or len(raw) != scalar_sizes[vtype]:
                        return None
                    if vtype == 7:
                        return bool(raw[0])
                    if vtype == 4:
                        return struct.unpack("<I", raw)[0]
                    if vtype == 5:
                        return struct.unpack("<i", raw)[0]
                    return None
                if vtype == 8:
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > 10_000_000:
                        raise ValueError("GGUF string length exceeds safety bound")
                    if keep:
                        return f.read(n).decode("utf-8", errors="replace")
                    f.seek(n, os.SEEK_CUR)
                    return None
                if vtype == 9:
                    elem_type, count = struct.unpack("<IQ", f.read(12))
                    if count > 10_000_000:
                        raise ValueError("GGUF array count exceeds safety bound")
                    if elem_type in scalar_sizes:
                        f.seek(scalar_sizes[elem_type] * count, os.SEEK_CUR)
                    else:
                        for _ in range(count):
                            _skip_or_read(elem_type, False)
                    return None
                raise ValueError(f"Unsupported GGUF value type: {vtype}")

            out: dict[str, Any] = {"gguf_version": int(version)}
            for _ in range(n_kv):
                key = _read_str()
                vtype = struct.unpack("<I", f.read(4))[0]
                keep = key in interesting_exact or key.endswith(".context_length")
                val = _skip_or_read(vtype, keep)
                if not keep or val is None:
                    continue
                if key == "tokenizer.chat_template":
                    out["has_chat_template"] = bool(str(val).strip())
                else:
                    out[key] = val
            return out
    except Exception:
        return {}


def build_json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build a llama-cpp-python / OpenAI compatible JSON-schema response_format payload."""
    schema_copy = dict(schema)
    if name and "title" not in schema_copy:
        schema_copy["title"] = name
    return {
        "type": "json_object",
        "schema": schema_copy,
    }


def _to_openai_response_format(
    response_format: dict[str, Any] | None,
    fallback_only: bool = False,
) -> dict[str, Any] | None:
    if response_format is None:
        return None
    if fallback_only:
        return {"type": "json_object"}
    rf_type = response_format.get("type")
    if rf_type == "json_object" and isinstance(response_format.get("schema"), dict):
        schema = response_format["schema"]
        name = str(schema.get("title") or "ziggart_response")
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
            },
        }
    return dict(response_format)


@dataclass(frozen=True)
class ModelConfig:
    model_path: str = ""
    mmproj_path: str = ""
    context_size: int = 8192
    gpu_layers: int = -1
    threads: int = 4
    flash_attention: bool = True
    mmproj_on_gpu: bool = True
    enable_prefix_cache: bool = True
    prefix_cache_mb: int = 256
    kv_cache_type: str = "f16"
    n_batch: int = 512
    backend: str = "in_process_gguf"
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = ""
    request_timeout: float = 120.0
    llama_server_path: str = ""
    auto_download_llama_server: bool = False
    image_min_tokens: int = -1


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_UNCLOSED_THINK_CLOSE_RE = re.compile(r"^[\s\S]*?</think>", re.IGNORECASE)
_UNCLOSED_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)

# Gemma 4 reasoning blocks (<|channel>thought ... <channel|> and variants)
_GEMMA_CHANNEL_BLOCK_RE = re.compile(
    r"<\|channel\|?>\s*thought\b[\s\S]*?<\|?channel\|?>",
    re.IGNORECASE,
)
_GEMMA_CHANNEL_UNCLOSED_RE = re.compile(
    r"<\|channel\|?>\s*thought\b[\s\S]*$",
    re.IGNORECASE,
)
_GEMMA_CHANNEL_ORPHAN_CLOSE_RE = re.compile(
    r"^[\s\S]*?<channel\|>",
    re.IGNORECASE,
)

# Alternate <thought> / <|thought|> markers
_THOUGHT_TAG_BLOCK_RE = re.compile(
    r"<\|?thought\|?>[\s\S]*?</\|?thought\|?>",
    re.IGNORECASE,
)
_THOUGHT_TAG_UNCLOSED_RE = re.compile(
    r"<\|?thought\|?>[\s\S]*$",
    re.IGNORECASE,
)

_ALL_THINK_MARKERS_RE = re.compile(
    r"</?think>|<\|?thought\|?>|</\|?thought\|?>|<\|channel\|?>\s*thought\b|<\|?channel\|?>",
    re.IGNORECASE,
)


def strip_thinking_tags(text: str) -> str:
    """
    Removes <think>...</think>, Gemma 4 <|channel>thought...<channel|>, and
    <thought>...</thought> reasoning blocks emitted by thinking VLMs while handling
    truncated unclosed blocks and preserving fallback content if the entire response
    was inside a thinking block.
    """
    if not isinstance(text, str) or not text:
        return ""
    low = text.lower()
    if (
        "<think" not in low
        and "</think>" not in low
        and "<|channel" not in low
        and "<channel|>" not in low
        and "<thought>" not in low
        and "<|thought|>" not in low
    ):
        return text.strip()

    cleaned = text
    cleaned = _THINK_BLOCK_RE.sub("", cleaned)
    cleaned = _GEMMA_CHANNEL_BLOCK_RE.sub("", cleaned)
    cleaned = _THOUGHT_TAG_BLOCK_RE.sub("", cleaned)

    if "</think>" in cleaned.lower():
        cleaned = _UNCLOSED_THINK_CLOSE_RE.sub("", cleaned)
    if "<channel|>" in cleaned.lower():
        cleaned = _GEMMA_CHANNEL_ORPHAN_CLOSE_RE.sub("", cleaned)

    if "<think>" in cleaned.lower():
        before_open = _UNCLOSED_THINK_OPEN_RE.sub("", cleaned).strip()
        cleaned = before_open if before_open else ""
    if "<|channel" in cleaned.lower():
        before_open = _GEMMA_CHANNEL_UNCLOSED_RE.sub("", cleaned).strip()
        cleaned = before_open if before_open else ""
    if "<thought>" in cleaned.lower() or "<|thought|>" in cleaned.lower():
        before_open = _THOUGHT_TAG_UNCLOSED_RE.sub("", cleaned).strip()
        cleaned = before_open if before_open else ""

    cleaned = cleaned.strip()
    if cleaned:
        return cleaned

    # Fallback: if the model only emitted text inside a reasoning tag before hitting max_tokens,
    # unwrap the inner text (returning "" if there were only empty markers).
    inner = _ALL_THINK_MARKERS_RE.sub("", text).strip()
    return inner


def release_comfy_vram() -> None:
    """Ask ComfyUI to evict its managed models before loading the VLM."""
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except (ImportError, AttributeError):
        pass


def empty_accelerator_cache() -> None:
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except (ImportError, AttributeError):
        pass


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _close_handler_safely(handler: Any) -> None:
    if handler is None:
        return
    try:
        if hasattr(handler, "close"):
            handler.close()
        elif hasattr(handler, "_exit_stack") and handler._exit_stack is not None:
            handler._exit_stack.close()
    except Exception:
        pass


def find_llama_server_binary(custom_path: str = "") -> str:
    """
    Locate a local `llama-server` binary (explicit custom_path -> ZIGGART_LLAMA_SERVER_PATH ->
    plugin-local bin/llama.cpp/ -> PATH). Returns empty string if not found.
    """
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    candidates: list[str] = []
    for raw in (custom_path, os.environ.get("ZIGGART_LLAMA_SERVER_PATH", "")):
        cleaned = (raw or "").strip().strip('"')
        if cleaned:
            if os.path.isdir(cleaned):
                candidates.append(os.path.join(cleaned, exe_name))
            else:
                candidates.append(cleaned)

    candidates.append(os.path.join(DEFAULT_LLAMA_SERVER_DIR, exe_name))
    candidates.append(os.path.join(DEFAULT_LLAMA_SERVER_DIR, "build", "bin", exe_name))

    for cand in candidates:
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)

    on_path = shutil.which("llama-server")
    if on_path and os.path.isfile(on_path):
        return os.path.abspath(on_path)
    return ""


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target_path = (dest_resolved / member.filename).resolve()
            if not target_path.is_relative_to(dest_resolved):
                raise EmbeddedLlamaError(f"Blocked unsafe zip entry path: {member.filename}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _select_official_llama_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the best matching Windows CUDA (or CPU fallback) assets from a llama.cpp release."""
    by_name = {str(a.get("name") or ""): a for a in assets if a.get("browser_download_url")}
    names = list(by_name.keys())

    if os.name == "nt":
        for cuda_tag in ("cuda-12.4-x64", "cuda-13.5-x64", "cpu-x64"):
            main_matches = [
                n for n in names if n.startswith("llama-") and cuda_tag in n and n.endswith(".zip")
            ]
            if not main_matches:
                continue
            selected = [by_name[main_matches[0]]]
            if "cuda-" in cuda_tag:
                cudart_matches = [
                    n for n in names if n.startswith("cudart-llama-") and cuda_tag in n and n.endswith(".zip")
                ]
                if cudart_matches:
                    selected.append(by_name[cudart_matches[0]])
            return selected
    else:
        for linux_tag in ("ubuntu-x64", "ubuntu-vulkan-x64"):
            main_matches = [
                n for n in names if n.startswith("llama-") and linux_tag in n and (n.endswith(".zip") or n.endswith(".tar.gz"))
            ]
            if main_matches:
                return [by_name[main_matches[0]]]
    return []


def download_official_llama_server(
    dest_dir: str = DEFAULT_LLAMA_SERVER_DIR,
    release_tag: str = "latest",
) -> str:
    """
    Downloads the official ggml-org/llama.cpp Windows CUDA binary bundle (including `llama-server.exe`
    and `cudart` DLLs) into gitignored `bin/llama.cpp/` and returns the executable path.
    """
    existing = find_llama_server_binary(dest_dir)
    if existing:
        return existing

    api_url = (
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
        if release_tag in ("", "latest")
        else f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{release_tag}"
    )
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "Ziggart-Nodes/1.10", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        raise EmbeddedLlamaError(
            f"Failed to query official ggml-org/llama.cpp release metadata ({api_url}): {exc}"
        ) from exc

    releases = raw_payload if isinstance(raw_payload, list) else [raw_payload]
    release_data: dict[str, Any] = releases[0] if releases else {}
    selected_assets: list[dict[str, Any]] = []
    for candidate_rel in releases:
        if candidate_rel.get("draft"):
            continue
        assets = _select_official_llama_assets(candidate_rel.get("assets") or [])
        if assets:
            release_data = candidate_rel
            selected_assets = assets
            break

    if not selected_assets:
        raise EmbeddedLlamaError(
            "Could not find a compatible Windows/Linux llama.cpp release archive in "
            f"release {release_data.get('tag_name', release_tag)!r}."
        )

    dest_path = Path(dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    print(
        f"[Ziggart LLM] Downloading official llama.cpp {release_data.get('tag_name', '')} "
        f"into {dest_path}...",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for asset in selected_assets:
            asset_name = str(asset["name"])
            asset_url = str(asset["browser_download_url"])
            archive_path = tmp_root / asset_name
            dl_req = urllib.request.Request(asset_url, headers={"User-Agent": "Ziggart-Nodes/1.10"})
            try:
                with urllib.request.urlopen(dl_req, timeout=300) as dl_resp, open(archive_path, "wb") as out_f:
                    shutil.copyfileobj(dl_resp, out_f)
            except Exception as exc:
                raise EmbeddedLlamaError(
                    f"Failed to download llama.cpp asset '{asset_name}': {exc}"
                ) from exc

            if asset_name.endswith(".zip"):
                _safe_extract_zip(archive_path, dest_path)

    # Flatten nested subfolder if release archive packaged binaries inside a subdirectory
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    direct_exe = dest_path / exe_name
    if not direct_exe.is_file():
        for found in dest_path.rglob(exe_name):
            parent = found.parent
            if parent != dest_path:
                for item in parent.iterdir():
                    target = dest_path / item.name
                    if item.is_file() and not target.exists():
                        shutil.move(str(item), str(target))
            break

    final_exe = find_llama_server_binary(str(dest_path))
    if not final_exe:
        raise EmbeddedLlamaError(
            f"Downloaded llama.cpp archive into {dest_path}, but '{exe_name}' was not found."
        )
    return final_exe


_ARCH_HANDLER_PRIORITY: dict[str, tuple[str, ...]] = {
    "gemma4": ("Gemma4ChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "gemma3": ("Gemma3ChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen35": ("Qwen35ChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen3_5": ("Qwen35ChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen35moe": ("Qwen35ChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen3vl": ("Qwen3VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen3vlmoe": ("Qwen3VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen3": ("Qwen3VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen3moe": ("Qwen3VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen25vl": ("Qwen25VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen2_5_vl": ("Qwen25VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "qwen2vl": ("Qwen25VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "glm41v": ("GLM41VChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "glm46v": ("GLM46VChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "lfm2vl": ("LFM2VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "lfm25vl": ("LFM25VLChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "minicpmv": ("MiniCPMV46ChatHandler", "MiniCPMv45ChatHandler", "MiniCPMv26ChatHandler", "GenericMTMDChatHandler", "MTMDChatHandler"),
    "inkling2": ("GenericMTMDChatHandler", "MTMDChatHandler"),
    "inkling2_loop": ("GenericMTMDChatHandler", "MTMDChatHandler"),
}


def _resolve_handler_class(
    arch: str,
    has_chat_template: bool,
    chat_format_mod: Any,
    multimodal_mod: Any = None,
) -> tuple[Any, str]:
    arch_norm = (arch or "").strip().lower()
    candidates = _ARCH_HANDLER_PRIORITY.get(arch_norm)
    if candidates is None:
        candidates = ("GenericMTMDChatHandler", "MTMDChatHandler") if has_chat_template else ("MTMDChatHandler",)

    for cls_name in candidates:
        for mod in (chat_format_mod, multimodal_mod):
            if mod is None:
                continue
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls, cls_name
    raise ImportError("MTMDChatHandler could not be imported from llama_cpp")


def _create_mtmd_chat_handler(
    *,
    mmproj_path: str,
    arch: str,
    has_chat_template: bool,
    use_gpu: bool,
    image_min_tokens: int = -1,
    verbose: bool = False,
) -> tuple[Any, str]:
    """
    Instantiates the best MTMD chat handler in `llama_cpp` for the detected GGUF architecture,
    adapting constructor arguments across different versions of `llama-cpp-python`.
    """
    import llama_cpp.llama_chat_format as chat_format_mod

    try:
        import llama_cpp.llama_multimodal as multimodal_mod
    except ImportError:
        multimodal_mod = None

    cls, cls_name = _resolve_handler_class(arch, has_chat_template, chat_format_mod, multimodal_mod)

    sig_params: dict[str, Any] = {}
    accepts_kwargs = False
    try:
        sig = inspect.signature(cls.__init__)
        sig_params = dict(sig.parameters)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_kwargs = True

    kwargs: dict[str, Any] = {
        "verbose": verbose,
        "use_gpu": use_gpu,
    }
    if "mmproj_path" in sig_params:
        kwargs["mmproj_path"] = mmproj_path
    else:
        kwargs["clip_model_path"] = mmproj_path

    if "chat_format" in sig_params:
        # GenericMTMDChatHandler expects chat_format=None to read tokenizer.chat_template from the GGUF.
        kwargs["chat_format"] = None

    if "enable_thinking" in sig_params:
        kwargs["enable_thinking"] = False

    if int(image_min_tokens) > 0 and ("image_min_tokens" in sig_params or accepts_kwargs):
        kwargs["image_min_tokens"] = int(image_min_tokens)

    try:
        return cls(**kwargs), cls_name
    except TypeError:
        # Defensive retry stripping optional parameters on older wrappers
        for key in ("image_min_tokens", "enable_thinking"):
            kwargs.pop(key, None)
        return cls(**kwargs), cls_name


class EmbeddedModelManager:
    """
    Caches one in-process llama.cpp model (multimodal or text-only), its optional MTMD
    projector, optional RAM prefix KV cache, a managed local llama-server subprocess,
    or an OpenAI-compatible remote VLM/LLM endpoint.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._handler: Any = None
        self._cache: Any = None
        self._config: ModelConfig | None = None
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._server_base_url: str = ""
        self._server_stderr_tail: deque[str] = deque(maxlen=60)
        self._server_stderr_thread: threading.Thread | None = None
        self.load_count: int = 0
        self.reuse_count: int = 0
        self.batch_calls_count: int = 0

    def _active_openai_url(self) -> str:
        if self._server_proc is not None and self._server_base_url:
            return self._server_base_url
        if self._config is not None and getattr(self._config, "backend", "in_process_gguf") == "openai_compatible":
            return self._config.api_base_url.rstrip("/")
        return ""

    def is_loaded(self) -> bool:
        with self._lock:
            if self._server_proc is not None:
                return self._server_proc.poll() is None and bool(self._server_base_url)
            if self._config is not None and self._config.backend == "openai_compatible":
                return bool(self._config.api_base_url)
            return self._model is not None

    def _is_compatible_loaded_config(self, config: ModelConfig) -> bool:
        """
        Checks whether `config` refers to the same loaded GGUF weights, projector, context,
        KV cache settings, and GPU offload configuration, or the same OpenAI-compatible backend.
        """
        if self._config is None:
            return False
        cur = self._config
        cur_backend = getattr(cur, "backend", "in_process_gguf")
        new_backend = getattr(config, "backend", "in_process_gguf")
        if cur_backend != new_backend:
            return False
        if new_backend == "openai_compatible":
            return (
                cur.api_base_url.rstrip("/") == config.api_base_url.rstrip("/")
                and cur.api_key == config.api_key
                and cur.api_model == config.api_model
                and cur.context_size == config.context_size
                and float(cur.request_timeout) == float(config.request_timeout)
            )
        if self._server_proc is not None:
            if self._server_proc.poll() is not None:
                return False
        elif self._model is None:
            return False

        return (
            _norm_path_for_compare(cur.model_path) == _norm_path_for_compare(config.model_path)
            and _norm_path_for_compare(cur.mmproj_path) == _norm_path_for_compare(config.mmproj_path)
            and cur.context_size == config.context_size
            and cur.gpu_layers == config.gpu_layers
            and cur.flash_attention == config.flash_attention
            and cur.mmproj_on_gpu == config.mmproj_on_gpu
            and getattr(cur, "enable_prefix_cache", True) == getattr(config, "enable_prefix_cache", True)
            and int(getattr(cur, "prefix_cache_mb", 256)) == int(getattr(config, "prefix_cache_mb", 256))
            and getattr(cur, "kv_cache_type", "f16") == getattr(config, "kv_cache_type", "f16")
            and int(getattr(cur, "n_batch", 512)) == int(getattr(config, "n_batch", 512))
            and _norm_path_for_compare(getattr(cur, "llama_server_path", ""))
            == _norm_path_for_compare(getattr(config, "llama_server_path", ""))
            and int(getattr(cur, "image_min_tokens", -1)) == int(getattr(config, "image_min_tokens", -1))
        )

    def _start_llama_server(self, config: ModelConfig, arch: str) -> None:
        exe_path = find_llama_server_binary(getattr(config, "llama_server_path", ""))
        if not exe_path and getattr(config, "auto_download_llama_server", False):
            exe_path = download_official_llama_server()
        if not exe_path:
            raise EmbeddedLlamaError(
                "No llama-server binary found. Enable auto_download_llama_server or provide "
                "llama_server_path (or place llama-server.exe inside Ziggart-Nodes/bin/llama.cpp/)."
            )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])

        n_batch = max(1, int(getattr(config, "n_batch", 512)))
        ngl = 999 if int(config.gpu_layers) < 0 else int(config.gpu_layers)
        cmd: list[str] = [
            exe_path,
            "--model",
            config.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(int(config.context_size)),
            "--n-gpu-layers",
            str(ngl),
            "--threads",
            str(max(1, int(config.threads))),
            "--threads-batch",
            str(max(1, int(config.threads))),
            "--batch-size",
            str(n_batch),
            "--flash-attn",
            "on" if config.flash_attention else "off",
            "--jinja",
        ]
        if (config.mmproj_path or "").strip():
            cmd.extend(["--mmproj", config.mmproj_path])
            if not config.mmproj_on_gpu:
                cmd.append("--no-mmproj-offload")
        if arch == "gemma4":
            ubatch = min(max(n_batch, 2048), int(config.context_size))
            cmd.extend(["--ubatch-size", str(ubatch)])

        kv_type = str(getattr(config, "kv_cache_type", "f16") or "f16").lower()
        if kv_type in ("q8_0", "q4_0"):
            cmd.extend(["--cache-type-k", kv_type, "--cache-type-v", kv_type])

        img_min = int(getattr(config, "image_min_tokens", -1))
        if img_min > 0:
            cmd.extend(["--image-min-tokens", str(img_min)])

        popen_kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
            "cwd": os.path.dirname(exe_path),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        self._server_stderr_tail.clear()
        proc = subprocess.Popen(cmd, **popen_kwargs)
        self._server_proc = proc

        def _drain_stderr() -> None:
            pipe = proc.stderr
            if pipe is None:
                return
            try:
                for raw in iter(pipe.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        self._server_stderr_tail.append(line)
            except Exception:
                pass

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()
        self._server_stderr_thread = t

        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + max(60.0, float(getattr(config, "request_timeout", 120.0)))
        last_err = ""
        while time.monotonic() < deadline:
            exit_code = proc.poll()
            if exit_code is not None:
                time.sleep(0.05)
                tail = "\n".join(list(self._server_stderr_tail)[-12:])
                self._stop_server_process()
                legacy_hint = ""
                if arch == "inkling2" and ("check_tensor_dims" in tail or "not found" in tail):
                    legacy_hint = (
                        "\nHint: This file uses the early pre-mainline 'inkling2' schema (logical block_count=48 "
                        "and legacy tensor names) instead of mainline 'inkling2_loop' (physical block_count=24). "
                        "Download an updated GGUF from unsloth/Inkling-2-Mini-GGUF (arch='inkling2_loop') and matching mmproj."
                    )
                raise EmbeddedLlamaError(
                    f"Managed llama-server exited during startup (code {exit_code}) "
                    f"for model '{os.path.basename(config.model_path)}' (arch='{arch or 'unknown'}').\n{tail}{legacy_hint}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                    if resp.status == 200:
                        self._server_base_url = f"http://127.0.0.1:{port}/v1"
                        self._config = config
                        self.load_count += 1
                        return
            except urllib.error.HTTPError as http_exc:
                # 503 = llama-server still loading weights
                last_err = f"HTTP {http_exc.code}"
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.25)

        tail = "\n".join(list(self._server_stderr_tail)[-12:])
        self._stop_server_process()
        raise EmbeddedLlamaError(
            f"Timed out waiting for managed llama-server readiness ({last_err}).\n{tail}"
        )

    def _stop_server_process(self) -> None:
        proc = self._server_proc
        self._server_proc = None
        self._server_base_url = ""
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:
            pass
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except Exception:
                pass

    def load(self, config: ModelConfig, unload_comfy_models: bool, *, force_reload: bool = False) -> None:
        with self._lock:
            if not force_reload and self._is_compatible_loaded_config(config):
                self.reuse_count += 1
                return
            self.unload()

            backend = getattr(config, "backend", "in_process_gguf") or "in_process_gguf"
            if backend == "openai_compatible":
                base_url = (config.api_base_url or "").strip()
                if not base_url.startswith(("http://", "https://")):
                    raise EmbeddedLlamaError(
                        f"Invalid OpenAI-compatible api_base_url: '{base_url}'. Must start with http:// or https://."
                    )
                self._config = config
                self.load_count += 1
                return

            meta = read_gguf_metadata(config.model_path)
            arch = str(meta.get("general.architecture") or "").lower()
            has_chat_template = bool(meta.get("has_chat_template", False))

            if unload_comfy_models:
                release_comfy_vram()

            if backend == "llama_server":
                self._start_llama_server(config, arch)
                return

            llama_cpp_ver = "unknown"
            try:
                import llama_cpp
                from llama_cpp import Llama

                llama_cpp_ver = str(getattr(llama_cpp, "__version__", "unknown"))
            except (ImportError, OSError) as exc:
                if backend == "auto" and (
                    find_llama_server_binary(getattr(config, "llama_server_path", ""))
                    or getattr(config, "auto_download_llama_server", False)
                ):
                    self._start_llama_server(config, arch)
                    return
                raise EmbeddedLlamaError(
                    "llama-cpp-python is required inside ComfyUI. "
                    "Install llama-cpp-python >= 0.3.35 using a wheel that matches your "
                    "CUDA/ROCm/Vulkan setup, then restart ComfyUI."
                ) from exc

            handler: Any = None
            has_mmproj = bool((config.mmproj_path or "").strip())
            try:
                if has_mmproj:
                    handler, _handler_name = _create_mtmd_chat_handler(
                        mmproj_path=config.mmproj_path,
                        arch=arch,
                        has_chat_template=has_chat_template,
                        use_gpu=config.mmproj_on_gpu,
                        image_min_tokens=int(getattr(config, "image_min_tokens", -1)),
                        verbose=False,
                    )

                n_batch = max(1, int(getattr(config, "n_batch", 512)))
                llama_kwargs: dict[str, Any] = {
                    "model_path": config.model_path,
                    "n_ctx": config.context_size,
                    "n_gpu_layers": config.gpu_layers,
                    "n_threads": config.threads,
                    "n_threads_batch": config.threads,
                    "n_batch": n_batch,
                    "flash_attn": config.flash_attention,
                    "verbose": False,
                }
                if arch == "gemma4":
                    llama_kwargs["n_ubatch"] = min(max(n_batch, 2048), int(config.context_size))
                if handler is not None:
                    llama_kwargs["chat_handler"] = handler

                kv_type = str(getattr(config, "kv_cache_type", "f16") or "f16").lower()
                if kv_type in ("q8_0", "q4_0"):
                    ggml_const = getattr(
                        llama_cpp,
                        "GGML_TYPE_Q8_0" if kv_type == "q8_0" else "GGML_TYPE_Q4_0",
                        None,
                    )
                    if ggml_const is not None:
                        llama_kwargs["type_k"] = ggml_const
                        llama_kwargs["type_v"] = ggml_const

                try:
                    model = Llama(**llama_kwargs)
                except TypeError as type_exc:
                    exc_str = str(type_exc)
                    stripped = False
                    for opt_key in ("type_k", "type_v", "n_batch", "n_ubatch"):
                        if opt_key in exc_str and opt_key in llama_kwargs:
                            llama_kwargs.pop(opt_key, None)
                            stripped = True
                    if not stripped:
                        for opt_key in ("type_k", "type_v", "n_batch", "n_ubatch"):
                            llama_kwargs.pop(opt_key, None)
                    model = Llama(**llama_kwargs)

                ram_cache = None
                if getattr(config, "enable_prefix_cache", True):
                    cache_cls = getattr(llama_cpp, "LlamaRAMCache", None)
                    if cache_cls is not None and hasattr(model, "set_cache"):
                        try:
                            cap_bytes = max(16, int(getattr(config, "prefix_cache_mb", 256))) * 1024 * 1024
                            ram_cache = cache_cls(capacity_bytes=cap_bytes)
                            model.set_cache(ram_cache)
                        except Exception:
                            ram_cache = None
            except Exception as exc:
                _close_handler_safely(handler)
                empty_accelerator_cache()

                if backend == "auto" and (
                    find_llama_server_binary(getattr(config, "llama_server_path", ""))
                    or getattr(config, "auto_download_llama_server", False)
                ):
                    try:
                        self._start_llama_server(config, arch)
                        return
                    except Exception as srv_exc:
                        raise EmbeddedLlamaError(
                            f"Could not load GGUF (arch='{arch or 'unknown'}', llama-cpp-python {llama_cpp_ver}): "
                            f"{exc}\nFallback managed llama-server also failed: {srv_exc}"
                        ) from srv_exc

                arch_hint = (
                    f" Detected GGUF architecture '{arch}' may not be supported by llama-cpp-python {llama_cpp_ver} — "
                    "switch backend to 'auto' / 'llama_server' or update llama-cpp-python."
                    if arch
                    else ""
                )
                raise EmbeddedLlamaError(f"Could not load the GGUF model/mmproj: {exc}{arch_hint}") from exc

            self._model = model
            self._handler = handler
            self._cache = ram_cache
            self._config = config
            self.load_count += 1

    def _invoke_local_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        enable_thinking: bool,
        response_format: dict[str, Any] | None,
        stream: bool,
    ) -> Any:
        if self._model is None:
            raise EmbeddedLlamaError("The embedded model is not loaded.")

        if self._handler is not None:
            if hasattr(self._handler, "enable_thinking"):
                try:
                    self._handler.enable_thinking = bool(enable_thinking)
                except Exception:
                    pass
            extra_args = getattr(self._handler, "extra_template_arguments", None)
            if isinstance(extra_args, dict):
                extra_args["enable_thinking"] = bool(enable_thinking)

            call_kwargs: dict[str, Any] = {
                "llama": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": None if seed < 0 else seed,
                "stream": stream,
                "enable_thinking": enable_thinking,
            }
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            target = self._handler
        else:
            if _messages_have_images(messages):
                raise EmbeddedLlamaError(
                    "Image input requires an mmproj GGUF (or an OpenAI-compatible vision server). "
                    "Select a matching mmproj or use text-only prompts."
                )
            call_kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": None if seed < 0 else seed,
                "stream": stream,
            }
            if response_format is not None:
                call_kwargs["response_format"] = response_format
            target = self._model.create_chat_completion

        for _ in range(4):
            try:
                return target(**call_kwargs)
            except TypeError as type_exc:
                err_msg = str(type_exc)
                removed = False
                for opt_key in ("enable_thinking", "response_format", "top_k", "seed"):
                    if opt_key in err_msg and opt_key in call_kwargs:
                        call_kwargs.pop(opt_key, None)
                        removed = True
                if not removed and "response_format" in call_kwargs:
                    call_kwargs.pop("response_format", None)
                    removed = True
                if not removed:
                    raise
        return target(**call_kwargs)

    def _chat_openai_compatible(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        enable_thinking: bool = False,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        cfg = self._config
        if cfg is None:
            raise EmbeddedLlamaError("OpenAI-compatible backend configuration is missing.")
        base_url = self._active_openai_url()
        if not base_url:
            raise EmbeddedLlamaError("OpenAI-compatible base URL is missing.")
        endpoint = base_url.rstrip("/") + "/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key and self._server_proc is None:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        model_id = (
            os.path.basename(cfg.model_path)
            if self._server_proc is not None and cfg.model_path
            else (cfg.api_model or "local-model")
        )
        base_payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": bool(stream),
        }
        if int(seed) >= 0:
            base_payload["seed"] = int(seed)
        if int(top_k) > 0:
            base_payload["top_k"] = int(top_k)

        payload_variants: list[dict[str, Any]] = []
        if self._server_proc is not None:
            with_think = dict(base_payload)
            with_think["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
            payload_variants = [with_think, base_payload]
        else:
            payload_variants = [base_payload]

        attempts: list[dict[str, Any]] = []
        for variant in payload_variants:
            if response_format is not None:
                p1 = dict(variant)
                p1["response_format"] = _to_openai_response_format(response_format, fallback_only=False)
                p2 = dict(variant)
                p2["response_format"] = _to_openai_response_format(response_format, fallback_only=True)
                attempts.extend([p1, p2, dict(variant)])
            else:
                attempts.append(variant)

        last_exc: Exception | None = None
        for payload in attempts:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=float(getattr(cfg, "request_timeout", 120.0))) as resp:
                    if not stream:
                        raw_body = resp.read().decode("utf-8", errors="replace")
                        parsed = json.loads(raw_body)
                        msg_obj = parsed["choices"][0]["message"]
                        content = msg_obj.get("content")
                        if not content and msg_obj.get("reasoning_content") and self._server_proc is not None:
                            content = msg_obj.get("reasoning_content")
                        text = content if isinstance(content, str) else str(content or "")
                        if on_token and text:
                            on_token(text)
                        return text
                    chunks: list[str] = []
                    for raw_line in resp:
                        if stop_event is not None and stop_event.is_set():
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            evt = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        choices = evt.get("choices") or []
                        if not choices:
                            continue
                        delta_obj = choices[0].get("delta") or choices[0].get("message") or {}
                        delta_text = delta_obj.get("content") or ""
                        if delta_text:
                            chunks.append(delta_text)
                            if on_token is not None:
                                on_token(delta_text)
                    return "".join(chunks)
            except urllib.error.HTTPError as http_exc:
                err_body = ""
                try:
                    err_body = http_exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                last_exc = EmbeddedLlamaError(
                    f"OpenAI-compatible endpoint returned HTTP {http_exc.code}: {err_body or http_exc.reason}"
                )
                if http_exc.code in (400, 422) and payload is not attempts[-1]:
                    continue
                raise last_exc from http_exc
            except Exception as exc:
                raise EmbeddedLlamaError(f"OpenAI-compatible request failed: {exc}") from exc

        if last_exc is not None:
            raise last_exc
        raise EmbeddedLlamaError("OpenAI-compatible request failed with no response.")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        enable_thinking: bool,
        response_format: dict[str, Any] | None = None,
        strip_thinking: bool = True,
    ) -> str:
        with self._lock:
            if self._active_openai_url():
                raw_text = self._chat_openai_compatible(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    response_format=response_format,
                    stream=False,
                )
            else:
                if self._model is None:
                    raise EmbeddedLlamaError("The embedded model is not loaded.")
                try:
                    result = self._invoke_local_completion(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                        enable_thinking=enable_thinking,
                        response_format=response_format,
                        stream=False,
                    )
                    content = result["choices"][0]["message"]["content"]
                    raw_text = content if isinstance(content, str) else str(content)
                except EmbeddedLlamaError:
                    raise
                except Exception as exc:
                    raise EmbeddedLlamaError(f"Multimodal inference failed: {exc}") from exc

            text = raw_text.strip()
            if strip_thinking:
                return strip_thinking_tags(text)
            return text

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        enable_thinking: bool,
        response_format: dict[str, Any] | None = None,
        strip_thinking: bool = True,
        on_token: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        """
        Streams completion tokens through `on_token(delta)` and honors `stop_event`
        between chunks. Falls back cleanly to non-streaming completion if unsupported.
        """
        with self._lock:
            if self._active_openai_url():
                raw_text = self._chat_openai_compatible(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    response_format=response_format,
                    stream=True,
                    on_token=on_token,
                    stop_event=stop_event,
                )
                text = raw_text.strip()
                return strip_thinking_tags(text) if strip_thinking else text

            if self._model is None:
                raise EmbeddedLlamaError("The embedded model is not loaded.")

            try:
                result = self._invoke_local_completion(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    response_format=response_format,
                    stream=True,
                )
                if isinstance(result, dict):
                    content = result["choices"][0]["message"]["content"]
                    raw_text = content if isinstance(content, str) else str(content)
                    if on_token is not None and raw_text:
                        on_token(raw_text)
                else:
                    chunks: list[str] = []
                    try:
                        for chunk in result:
                            if stop_event is not None and stop_event.is_set():
                                break
                            if isinstance(chunk, dict):
                                choices = chunk.get("choices") or []
                                if not choices:
                                    continue
                                delta_obj = choices[0].get("delta") or choices[0].get("message") or {}
                                delta_text = delta_obj.get("content") or ""
                                if delta_text:
                                    chunks.append(delta_text)
                                    if on_token is not None:
                                        on_token(delta_text)
                    finally:
                        close_fn = getattr(result, "close", None)
                        if callable(close_fn):
                            try:
                                close_fn()
                            except Exception:
                                pass
                    raw_text = "".join(chunks)
            except TypeError:
                raw_text = self.chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    enable_thinking=enable_thinking,
                    response_format=response_format,
                    strip_thinking=False,
                )
                if on_token is not None and raw_text:
                    on_token(raw_text)
            except EmbeddedLlamaError:
                raise
            except Exception as exc:
                raise EmbeddedLlamaError(f"Multimodal streaming inference failed: {exc}") from exc

            text = raw_text.strip()
            return strip_thinking_tags(text) if strip_thinking else text

    def chat_batch(
        self,
        messages_batch: list[list[dict[str, Any]]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float = 0.9,
        top_k: int = 30,
        seeds: list[int] | int = 0,
        enable_thinking: bool = False,
        response_format: dict[str, Any] | None = None,
        strip_thinking: bool = True,
    ) -> list[str]:
        """
        Executes a batch of multi-modal or text chat requests in a single locked session,
        reusing the loaded model and prefix KV cache across all items in the batch.
        """
        if not messages_batch:
            return []
        with self._lock:
            if not self._active_openai_url() and self._model is None:
                raise EmbeddedLlamaError("The embedded model is not loaded.")
            self.batch_calls_count += 1
            outputs: list[str] = []
            for idx, messages in enumerate(messages_batch):
                if isinstance(seeds, list):
                    if not seeds:
                        cur_seed = 0
                    else:
                        cur_seed = int(seeds[idx]) if idx < len(seeds) else int(seeds[-1])
                else:
                    cur_seed = int(seeds) if int(seeds) < 0 else int((int(seeds) + idx * 13) & 0x7FFFFFFF)
                resp = self.chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=cur_seed,
                    enable_thinking=enable_thinking,
                    response_format=response_format,
                    strip_thinking=strip_thinking,
                )
                outputs.append(resp)
            return outputs

    def unload(self) -> None:
        with self._lock:
            self._stop_server_process()
            model = self._model
            handler = self._handler
            self._model = None
            self._handler = None
            self._cache = None
            self._config = None
            if model is not None:
                try:
                    model.close()
                except Exception:
                    pass
                finally:
                    del model
            if handler is not None:
                _close_handler_safely(handler)
                del handler
            empty_accelerator_cache()


MODEL_MANAGER = EmbeddedModelManager()
atexit.register(MODEL_MANAGER.unload)
