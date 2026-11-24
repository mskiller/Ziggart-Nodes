from __future__ import annotations

import atexit
import gc
import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class EmbeddedLlamaError(RuntimeError):
    pass


def default_vlm_threads() -> int:
    """Returns a consistent default CPU thread count for GGUF VLM inference."""
    return max(1, (os.cpu_count() or 4) // 2)


def _norm_path_for_compare(path_str: str) -> str:
    cleaned = (path_str or "").strip()
    if not cleaned:
        return ""
    return os.path.normcase(os.path.abspath(cleaned))


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


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_UNCLOSED_THINK_CLOSE_RE = re.compile(r"^[\s\S]*?</think>", re.IGNORECASE)
_UNCLOSED_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """
    Removes <think>...</think> reasoning blocks emitted by thinking VLMs
    while handling truncated unclosed <think> blocks and preserving fallback
    content if the entire response was inside <think>.
    """
    if not isinstance(text, str) or not text:
        return ""
    low = text.lower()
    if "<think" not in low and "</think>" not in low:
        return text.strip()
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _UNCLOSED_THINK_CLOSE_RE.sub("", cleaned)
    if "<think>" in cleaned.lower():
        before_open = _UNCLOSED_THINK_OPEN_RE.sub("", cleaned).strip()
        if before_open:
            return before_open
        cleaned = ""
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned
    # Fallback: if model only emitted content inside <think> tags, extract inner text
    inner = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
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


class EmbeddedModelManager:
    """
    Caches one in-process llama.cpp model (multimodal or text-only), its optional MTMD
    projector, optional RAM prefix KV cache, or an OpenAI-compatible remote VLM/LLM endpoint.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._handler: Any = None
        self._cache: Any = None
        self._config: ModelConfig | None = None
        self.load_count: int = 0
        self.reuse_count: int = 0
        self.batch_calls_count: int = 0

    def is_loaded(self) -> bool:
        with self._lock:
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
        if self._model is None:
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
        )

    def load(self, config: ModelConfig, unload_comfy_models: bool, *, force_reload: bool = False) -> None:
        with self._lock:
            if not force_reload and self._is_compatible_loaded_config(config):
                self.reuse_count += 1
                return
            self.unload()

            backend = getattr(config, "backend", "in_process_gguf")
            if backend == "openai_compatible":
                base_url = (config.api_base_url or "").strip()
                if not base_url.startswith(("http://", "https://")):
                    raise EmbeddedLlamaError(
                        f"Invalid OpenAI-compatible api_base_url: '{base_url}'. Must start with http:// or https://."
                    )
                self._config = config
                self.load_count += 1
                return

            if unload_comfy_models:
                release_comfy_vram()
            try:
                import llama_cpp
                from llama_cpp import Llama
            except (ImportError, OSError) as exc:
                raise EmbeddedLlamaError(
                    "llama-cpp-python is required inside ComfyUI. "
                    "Install llama-cpp-python >= 0.3.35 using a wheel that matches your "
                    "CUDA/ROCm/Vulkan setup, then restart ComfyUI."
                ) from exc

            handler: Any = None
            has_mmproj = bool((config.mmproj_path or "").strip())
            if has_mmproj:
                try:
                    from llama_cpp.llama_chat_format import MTMDChatHandler
                except (ImportError, OSError) as exc:
                    raise EmbeddedLlamaError(
                        "llama-cpp-python with MTMD support is required for multimodal GGUF models. "
                        "Install llama-cpp-python >= 0.3.35 with MTMD support."
                    ) from exc

            try:
                if has_mmproj:
                    handler = MTMDChatHandler(
                        clip_model_path=config.mmproj_path,
                        verbose=False,
                        use_gpu=config.mmproj_on_gpu,
                    )

                llama_kwargs: dict[str, Any] = {
                    "model_path": config.model_path,
                    "n_ctx": config.context_size,
                    "n_gpu_layers": config.gpu_layers,
                    "n_threads": config.threads,
                    "n_threads_batch": config.threads,
                    "n_batch": max(1, int(getattr(config, "n_batch", 512))),
                    "flash_attn": config.flash_attention,
                    "verbose": False,
                }
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
                    for opt_key in ("type_k", "type_v", "n_batch"):
                        if opt_key in exc_str and opt_key in llama_kwargs:
                            llama_kwargs.pop(opt_key, None)
                            stripped = True
                    if not stripped:
                        for opt_key in ("type_k", "type_v", "n_batch"):
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
                raise EmbeddedLlamaError(f"Could not load the GGUF model/mmproj: {exc}") from exc

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
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> str:
        cfg = self._config
        if cfg is None:
            raise EmbeddedLlamaError("OpenAI-compatible backend configuration is missing.")
        endpoint = cfg.api_base_url.rstrip("/") + "/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        base_payload: dict[str, Any] = {
            "model": cfg.api_model or "local-model",
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

        attempts: list[dict[str, Any]] = []
        if response_format is not None:
            p1 = dict(base_payload)
            p1["response_format"] = _to_openai_response_format(response_format, fallback_only=False)
            p2 = dict(base_payload)
            p2["response_format"] = _to_openai_response_format(response_format, fallback_only=True)
            attempts.extend([p1, p2, dict(base_payload)])
        else:
            attempts.append(base_payload)

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
                        content = parsed["choices"][0]["message"]["content"]
                        text = content if isinstance(content, str) else str(content)
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
            if self._config is not None and getattr(self._config, "backend", "in_process_gguf") == "openai_compatible":
                raw_text = self._chat_openai_compatible(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
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
            if self._config is not None and getattr(self._config, "backend", "in_process_gguf") == "openai_compatible":
                raw_text = self._chat_openai_compatible(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
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
            is_openai = (
                self._config is not None
                and getattr(self._config, "backend", "in_process_gguf") == "openai_compatible"
            )
            if not is_openai and self._model is None:
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
