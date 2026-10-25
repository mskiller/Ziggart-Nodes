from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import folder_paths
except ImportError:  # Allows lightweight unit tests outside ComfyUI.
    folder_paths = None

from .embedded_model import MODEL_MANAGER, EmbeddedLlamaError, ModelConfig


MODEL_FOLDER = "llm_gguf"
NO_MODEL = "[No model GGUF found]"
NO_MMPROJ = "[No mmproj GGUF found]"


def _register_model_folder() -> None:
    if folder_paths is None:
        return
    model_dir = os.path.join(folder_paths.models_dir, "LLM", "GGUF")
    try:
        folder_paths.add_model_folder_path(MODEL_FOLDER, model_dir, is_default=True)
    except TypeError:  # Older ComfyUI versions do not have is_default.
        folder_paths.add_model_folder_path(MODEL_FOLDER, model_dir)


_register_model_folder()


def _gguf_files() -> tuple[list[str], list[str]]:
    if folder_paths is None:
        return [NO_MODEL], [NO_MMPROJ]
    names = [
        name
        for name in folder_paths.get_filename_list(MODEL_FOLDER)
        if name.lower().endswith(".gguf")
    ]
    projectors = [
        name
        for name in names
        if any(tag in Path(name).name.lower() for tag in ("mmproj", "projector"))
    ]
    models = [name for name in names if name not in projectors]
    return models or [NO_MODEL], projectors or [NO_MMPROJ]


def _resolve_gguf(selected: str, override: str, label: str) -> str:
    candidate = override.strip()
    if candidate:
        path = Path(os.path.expandvars(os.path.expanduser(candidate))).resolve()
    else:
        if selected in (NO_MODEL, NO_MMPROJ):
            raise ValueError(
                f"No {label} was found. Put it in ComfyUI/models/LLM/GGUF and refresh "
                f"ComfyUI, or provide an absolute {label} path in the advanced field."
            )
        full_path = folder_paths.get_full_path(MODEL_FOLDER, selected)
        if not full_path:
            raise ValueError(f"Could not resolve selected {label}: {selected}")
        path = Path(full_path).resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    if path.suffix.lower() != ".gguf":
        raise ValueError(f"{label} must be a .gguf file: {path}")
    return str(path)


def image_to_data_uri(image: Any, max_dimension: int = 0) -> str:
    """Convert one Comfy IMAGE tensor/array (HWC float 0..1) to a PNG data URI."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().float().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC image with 1, 3, or 4 channels; got {array.shape}")
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        # Multimodal encoders expect RGB; composite transparent pixels onto black.
        rgb = array[..., :3].astype(np.float32)
        alpha = array[..., 3:4].astype(np.float32) / 255.0
        array = (rgb * alpha + 0.5).astype(np.uint8)
    pil_image = Image.fromarray(array)
    if max_dimension > 0 and max(pil_image.size) > max_dimension:
        scale = max_dimension / max(pil_image.size)
        target = (
            max(1, round(pil_image.width * scale)),
            max(1, round(pil_image.height * scale)),
        )
        pil_image = pil_image.resize(target, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class GGUFVisionPrompt:
    """Describe ComfyUI images with a local GGUF vision-language model and mmproj."""

    DESCRIPTION = (
        "Runs a local multimodal GGUF directly inside ComfyUI through llama-cpp-python. "
        "When free_vram_after is enabled, the LLM and mmproj are closed after the answer."
    )
    CATEGORY = "Ziggart/LLM"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "composition", "subject", "style")
    OUTPUT_TOOLTIPS = (
        "The final response. Batch responses are separated by a divider.",
        "Composition specialist notes, when enabled.",
        "Subject specialist notes, when enabled.",
        "Style specialist notes, when enabled.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = _gguf_files()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image or image batch to describe."}),
                "model": (models, {"tooltip": "Language-model GGUF from models/LLM/GGUF."}),
                "mmproj": (projectors, {"tooltip": "Matching mmproj GGUF from models/LLM/GGUF."}),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are an expert image prompt writer. Describe the image accurately as a detailed generation prompt. Output only the prompt.",
                        "multiline": True,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "Turn this image into a detailed image-generation prompt.",
                        "multiline": True,
                    },
                ),
                "free_vram_after": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Close the embedded LLM and mmproj after inference to release their VRAM.",
                    },
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable Qwen3.5's internal thinking mode for every enabled pass.",
                    },
                ),
            },
            "optional": {
                "detail_crop": (
                    "IMAGE",
                    {"tooltip": "Optional high-resolution crop containing details from the main image."},
                ),
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Optional second reference image for additional visual context."},
                ),
                "max_image_dimension": (
                    ["original", "512", "768", "1024", "1536", "2048", "3072", "4096"],
                    {
                        "default": "2048",
                        "tooltip": "Downscale each image so its longest side is at most this size. Images are never enlarged.",
                    },
                ),
                "analyze_composition": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Run a separate composition-analysis inference pass."},
                ),
                "analyze_subject": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Run a separate subject-analysis inference pass."},
                ),
                "analyze_style": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Run a separate style-analysis inference pass."},
                ),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "context_size": ("INT", {"default": 8192, "min": 2048, "max": 262144, "step": 1024}),
                "gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 999,
                        "tooltip": "-1 offloads all possible layers; reduce this if VRAM is insufficient.",
                    },
                ),
                "threads": ("INT", {"default": max(1, (os.cpu_count() or 4) // 2), "min": 1, "max": 256}),
                "flash_attention": ("BOOLEAN", {"default": True}),
                "mmproj_on_gpu": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Disable to keep the vision projector on CPU and save VRAM."},
                ),
                "unload_comfy_models_before": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Evict diffusion models before loading the VLM. They reload automatically when generation starts.",
                    },
                ),
                "model_path_override": (
                    "STRING",
                    {"default": "", "tooltip": "Optional absolute model GGUF path."},
                ),
                "mmproj_path_override": (
                    "STRING",
                    {"default": "", "tooltip": "Optional absolute mmproj GGUF path."},
                ),
                "vlm_config": (
                    "GGUF_VL_MODEL",
                    {
                        "tooltip": "Optional shared VLM configuration from GGUF VLM Loader (overrides local model/mmproj settings)."
                    },
                ),
            },
        }

    def generate(
        self,
        image,
        model,
        mmproj,
        system_prompt,
        prompt,
        free_vram_after,
        enable_thinking,
        detail_crop=None,
        reference_image=None,
        max_image_dimension="2048",
        analyze_composition=False,
        analyze_subject=False,
        analyze_style=False,
        max_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        seed=-1,
        context_size=8192,
        gpu_layers=-1,
        threads=4,
        flash_attention=True,
        mmproj_on_gpu=True,
        unload_comfy_models_before=True,
        model_path_override="",
        mmproj_path_override="",
        vlm_config=None,
    ):
        if isinstance(vlm_config, ModelConfig):
            config = vlm_config
        else:
            model_path = _resolve_gguf(model, model_path_override, "model GGUF")
            projector_path = _resolve_gguf(mmproj, mmproj_path_override, "mmproj GGUF")
            config = ModelConfig(
                model_path=model_path,
                mmproj_path=projector_path,
                context_size=int(context_size),
                gpu_layers=int(gpu_layers),
                threads=int(threads),
                flash_attention=bool(flash_attention),
                mmproj_on_gpu=bool(mmproj_on_gpu),
            )

        try:
            MODEL_MANAGER.load(config, bool(unload_comfy_models_before))
            maximum = 0 if max_image_dimension == "original" else int(max_image_dimension)
            answers: list[str] = []
            pass_results: dict[str, list[str]] = {
                "composition": [],
                "subject": [],
                "style": [],
            }
            pass_prompts = {
                "composition": (
                    "Analyze the visual composition only. Identify framing, camera angle, perspective, "
                    "subject placement, depth, balance, focal hierarchy, and lighting direction. Return "
                    "concise factual notes useful for reconstructing the image."
                ),
                "subject": (
                    "Analyze the subjects only. Describe identity or type, count, appearance, pose, "
                    "expression, clothing, materials, important objects, and spatial relationships. "
                    "Return concise factual notes useful for reconstructing the image."
                ),
                "style": (
                    "Analyze the visual style only. Describe medium, genre, rendering technique, palette, "
                    "contrast, texture, lens or photographic qualities, era, and aesthetic influences. "
                    "Return concise factual notes useful for reconstructing the image."
                ),
            }
            enabled_passes = {
                "composition": bool(analyze_composition),
                "subject": bool(analyze_subject),
                "style": bool(analyze_style),
            }
            fallback_specialist_system = (
                "You are a precise visual-analysis specialist. Do not invent unseen details."
            )

            def specialist_system_message():
                return system_prompt.strip() or fallback_specialist_system

            def specialist_instruction(pass_name):
                custom_prompt = prompt.strip()
                if not custom_prompt:
                    return pass_prompts[pass_name]
                return (
                    custom_prompt
                    + "\n\nThis is the separate "
                    + pass_name
                    + " analysis pass. Concentrate on "
                    + pass_name
                    + " while following the prompt above."
                )

            def frame_for(batch, index):
                if batch is None:
                    return None
                return batch[index] if len(batch) > 1 and index < len(batch) else batch[0]

            def visual_content(index, instruction):
                content = [{"type": "text", "text": instruction + "\n\nMain image:"}]
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_uri(image[index], maximum)},
                    }
                )
                crop = frame_for(detail_crop, index)
                if crop is not None:
                    content.append(
                        {
                            "type": "text",
                            "text": "High-resolution detail crop from the main image; use it to recover fine details:",
                        }
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_uri(crop, maximum)},
                        }
                    )
                reference = frame_for(reference_image, index)
                if reference is not None:
                    content.append(
                        {
                            "type": "text",
                            "text": "Secondary reference image; treat it as supporting context and do not confuse it with the main image:",
                        }
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_uri(reference, maximum)},
                        }
                    )
                return content

            def infer(messages):
                return MODEL_MANAGER.chat(
                    messages,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    seed=int(seed),
                    enable_thinking=bool(enable_thinking),
                )

            for index in range(len(image)):
                notes = []
                for pass_name, enabled in enabled_passes.items():
                    if not enabled:
                        continue
                    specialist_messages = [
                        {
                            "role": "system",
                            "content": specialist_system_message(),
                        },
                        {
                            "role": "user",
                            "content": visual_content(
                                index, specialist_instruction(pass_name)
                            ),
                        },
                    ]
                    result = infer(specialist_messages)
                    pass_results[pass_name].append(result)
                    notes.append(f"{pass_name.title()} analysis:\n{result}")

                final_instruction = prompt
                if notes:
                    final_instruction += (
                        "\n\nSpecialist analysis notes are provided below. Use them as evidence, resolve any "
                        "conflicts against the images, and produce the requested final response.\n\n"
                        + "\n\n".join(notes)
                    )
                messages = []
                if system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                messages.append(
                    {"role": "user", "content": visual_content(index, final_instruction)}
                )
                answers.append(infer(messages))

            separator = "\n\n---\n\n"
            return (
                separator.join(answers),
                separator.join(pass_results["composition"]),
                separator.join(pass_results["subject"]),
                separator.join(pass_results["style"]),
            )
        except EmbeddedLlamaError as exc:
            raise RuntimeError(f"GGUF Vision Prompt failed: {exc}") from exc
        finally:
            if free_vram_after:
                MODEL_MANAGER.unload()


class ZiggartGGUFVLMLoader:
    """Shared GGUF Vision-Language model configuration node."""

    DESCRIPTION = (
        "Creates a reusable GGUF_VL_MODEL configuration that can be shared across "
        "GGUF Vision Prompt, Detailers Krea2, and WeVisOne-VL Detector nodes."
    )
    CATEGORY = "Ziggart/LLM"
    FUNCTION = "create_config"
    RETURN_TYPES = ("GGUF_VL_MODEL",)
    RETURN_NAMES = ("vlm_config",)
    OUTPUT_TOOLTIPS = ("Shared GGUF VLM configuration object.",)

    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = _gguf_files()
        return {
            "required": {
                "model": (models, {"tooltip": "Language-model GGUF from models/LLM/GGUF."}),
                "mmproj": (projectors, {"tooltip": "Matching mmproj GGUF from models/LLM/GGUF."}),
                "context_size": ("INT", {"default": 8192, "min": 1024, "max": 262144, "step": 1024}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999}),
                "threads": ("INT", {"default": max(1, (os.cpu_count() or 4) // 2), "min": 1, "max": 256}),
                "flash_attention": ("BOOLEAN", {"default": True}),
                "mmproj_on_gpu": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model_path_override": ("STRING", {"default": ""}),
                "mmproj_path_override": ("STRING", {"default": ""}),
            },
        }

    def create_config(
        self,
        model: str,
        mmproj: str,
        context_size: int = 8192,
        gpu_layers: int = -1,
        threads: int = 4,
        flash_attention: bool = True,
        mmproj_on_gpu: bool = True,
        model_path_override: str = "",
        mmproj_path_override: str = "",
    ) -> tuple[ModelConfig]:
        model_path = _resolve_gguf(model, model_path_override, "model GGUF")
        projector_path = _resolve_gguf(mmproj, mmproj_path_override, "mmproj GGUF")
        return (
            ModelConfig(
                model_path=model_path,
                mmproj_path=projector_path,
                context_size=int(context_size),
                gpu_layers=int(gpu_layers),
                threads=int(threads),
                flash_attention=bool(flash_attention),
                mmproj_on_gpu=bool(mmproj_on_gpu),
            ),
        )


NODE_CLASS_MAPPINGS = {
    "ZiggartGGUFVisionPrompt": GGUFVisionPrompt,
    "ZiggartGGUFVLMLoader": ZiggartGGUFVLMLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartGGUFVisionPrompt": "GGUF Vision Prompt",
    "ZiggartGGUFVLMLoader": "GGUF VLM Loader",
}
