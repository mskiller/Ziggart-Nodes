from __future__ import annotations

import atexit
import gc
import re
import threading
from dataclasses import dataclass
from typing import Any


class EmbeddedLlamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    mmproj_path: str
    context_size: int
    gpu_layers: int
    threads: int
    flash_attention: bool
    mmproj_on_gpu: bool


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"^[\s\S]*?</think>", re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """
    Removes <think>...</think> reasoning blocks emitted by thinking VLMs
    while preserving fallback content if the entire response was inside <think>.
    """
    if not text or "<think" not in text.lower() and "</think>" not in text.lower():
        return text.strip()
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _UNCLOSED_THINK_RE.sub("", cleaned)
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


class EmbeddedModelManager:
    """Caches one in-process llama.cpp model and its generic MTMD projector."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Any = None
        self._handler: Any = None
        self._config: ModelConfig | None = None

    def load(self, config: ModelConfig, unload_comfy_models: bool) -> None:
        with self._lock:
            if self._model is not None and self._config == config:
                return
            self.unload()
            if unload_comfy_models:
                release_comfy_vram()
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import MTMDChatHandler
            except (ImportError, OSError) as exc:
                raise EmbeddedLlamaError(
                    "llama-cpp-python with MTMD support is required inside ComfyUI. "
                    "Install llama-cpp-python >= 0.3.35 using a wheel that matches your "
                    "CUDA/ROCm/Vulkan setup, then restart ComfyUI."
                ) from exc

            try:
                handler = MTMDChatHandler(
                    clip_model_path=config.mmproj_path,
                    verbose=False,
                    use_gpu=config.mmproj_on_gpu,
                )
                model = Llama(
                    model_path=config.model_path,
                    chat_handler=handler,
                    n_ctx=config.context_size,
                    n_gpu_layers=config.gpu_layers,
                    n_threads=config.threads,
                    n_threads_batch=config.threads,
                    flash_attn=config.flash_attention,
                    verbose=False,
                )
            except Exception as exc:
                empty_accelerator_cache()
                raise EmbeddedLlamaError(f"Could not load the GGUF model/mmproj: {exc}") from exc

            self._model = model
            self._handler = handler
            self._config = config

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
            if self._model is None or self._handler is None:
                raise EmbeddedLlamaError("The embedded model is not loaded.")
            try:
                call_kwargs: dict[str, Any] = {
                    "llama": self._model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "seed": None if seed < 0 else seed,
                    "stream": False,
                    "enable_thinking": enable_thinking,
                }
                if response_format is not None:
                    call_kwargs["response_format"] = response_format

                try:
                    # Calling the MTMD handler directly lets Qwen's embedded Jinja
                    # template receive enable_thinking, which the public wrapper in
                    # llama-cpp-python does not currently expose.
                    result = self._handler(**call_kwargs)
                except TypeError as type_exc:
                    if "enable_thinking" in str(type_exc):
                        call_kwargs.pop("enable_thinking", None)
                        result = self._handler(**call_kwargs)
                    else:
                        raise
                content = result["choices"][0]["message"]["content"]
            except Exception as exc:
                raise EmbeddedLlamaError(f"Multimodal inference failed: {exc}") from exc
            text = content.strip() if isinstance(content, str) else str(content).strip()
            if strip_thinking:
                return strip_thinking_tags(text)
            return text

    def unload(self) -> None:
        with self._lock:
            model = self._model
            handler = self._handler
            self._model = None
            self._handler = None
            self._config = None
            if model is not None:
                try:
                    model.close()
                except Exception:
                    pass
                finally:
                    del model
            if handler is not None:
                try:
                    if hasattr(handler, "close"):
                        handler.close()
                    elif hasattr(handler, "_exit_stack") and handler._exit_stack is not None:
                        handler._exit_stack.close()
                except Exception:
                    pass
                finally:
                    del handler
            empty_accelerator_cache()


MODEL_MANAGER = EmbeddedModelManager()
atexit.register(MODEL_MANAGER.unload)
