from __future__ import annotations

import base64
import fnmatch
import io
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import torch
except ImportError:
    torch = None

try:
    import folder_paths
except ImportError:  # Allows lightweight unit tests outside ComfyUI.
    folder_paths = None

try:
    from server import PromptServer
    from aiohttp import web
except Exception:
    PromptServer = None
    web = None

from .embedded_model import MODEL_MANAGER, EmbeddedLlamaError, ModelConfig, strip_thinking_tags


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


def _resolve_vlm_node_config(
    model: str,
    mmproj: str,
    model_path_override: str,
    mmproj_path_override: str,
    context_size: int,
    gpu_layers: int,
    threads: int,
    flash_attention: bool,
    mmproj_on_gpu: bool,
    vlm_config: ModelConfig | None = None,
) -> ModelConfig:
    if isinstance(vlm_config, ModelConfig):
        return vlm_config
    model_path = _resolve_gguf(model, model_path_override, "model GGUF")
    projector_path = _resolve_gguf(mmproj, mmproj_path_override, "mmproj GGUF")
    return ModelConfig(
        model_path=model_path,
        mmproj_path=projector_path,
        context_size=int(context_size),
        gpu_layers=int(gpu_layers),
        threads=int(threads),
        flash_attention=bool(flash_attention),
        mmproj_on_gpu=bool(mmproj_on_gpu),
    )


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
        config = _resolve_vlm_node_config(
            model=model,
            mmproj=mmproj,
            model_path_override=model_path_override,
            mmproj_path_override=mmproj_path_override,
            context_size=context_size,
            gpu_layers=gpu_layers,
            threads=threads,
            flash_attention=flash_attention,
            mmproj_on_gpu=mmproj_on_gpu,
            vlm_config=vlm_config,
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


def _strip_json_fences(text: str) -> str:
    cleaned = strip_thinking_tags(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _format_diff_items(label: str, value: Any) -> str | None:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return f"{label}: " + "; ".join(items)
        return None
    text = str(value).strip() if value is not None else ""
    return f"{label}: {text}" if text else None


def _parse_revision_response(raw_text: str, fallback_prompt: str) -> tuple[str, str, float]:
    """Parse structured JSON revision output from the VLM with a safe text fallback."""
    cleaned = _strip_json_fences(raw_text)
    payload: dict[str, Any] | None = None

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            candidate = json.loads(cleaned[start : end + 1])
            if isinstance(candidate, dict):
                payload = candidate
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        revised = cleaned or fallback_prompt
        return revised, "Raw VLM response (non-JSON fallback).", 5.0

    try:
        score = float(payload.get("fidelity_score", 5.0))
    except (TypeError, ValueError):
        score = 5.0
    score = max(1.0, min(10.0, score))

    revised = str(payload.get("revised_prompt", "")).strip() or fallback_prompt

    report_parts = [
        f"Fidelity score: {score:.1f}/10.0",
        _format_diff_items("Missing or wrong", payload.get("missing_or_wrong")),
        _format_diff_items("Hallucinated elements", payload.get("hallucinated_elements")),
        _format_diff_items("Notes", payload.get("notes") or payload.get("summary")),
    ]
    diff_report = "\n".join(part for part in report_parts if part)
    return revised, diff_report, score


class ZiggartGGUFVisionPromptReviser:
    """Compare original and generated images against the current prompt and output a corrected prompt."""

    DESCRIPTION = (
        "Compares an original target image against a generated image and the prompt used to create it, "
        "diagnoses visual discrepancies, and produces a revised prompt for iterative refinement loops."
    )
    CATEGORY = "Ziggart/LLM"
    FUNCTION = "revise"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("revised_prompt", "diff_report", "fidelity_score", "needs_revision")
    OUTPUT_TOOLTIPS = (
        "Corrected prompt ready for the next generation pass.",
        "Structured diff report listing missing, wrong, and hallucinated elements.",
        "Estimated visual fidelity score (1.0-10.0) of generated_image vs original_image.",
        "True when fidelity_score is below fidelity_threshold.",
    )

    FOCUS_INSTRUCTIONS = {
        "all": "all visual aspects: subject identity, pose, clothing, objects, composition, background, colors, lighting, and artistic style",
        "subject_and_details": "subject identity, facial features, accessories, clothing details, and key objects",
        "composition_and_pose": "framing, camera angle, spatial layout, body pose, gaze direction, and proportions",
        "style_and_lighting": "medium, rendering technique, lighting direction, contrast, mood, and texture",
        "colors_and_background": "color palette, materials, environment, and background elements",
    }

    MODE_INSTRUCTIONS = {
        "surgical_edit": (
            "Keep accurate parts and effective phrasing from current_prompt; surgically fix, replace, "
            "or add only what diverges from the original target image, and remove hallucinated elements."
        ),
        "full_rewrite": (
            "Rewrite the prompt cleanly from scratch so it faithfully reproduces the original target image "
            "while correcting every discrepancy observed in the generated image."
        ),
    }

    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = _gguf_files()
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "Target reference image to match."}),
                "generated_image": ("IMAGE", {"tooltip": "Image generated from current_prompt."}),
                "current_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Prompt used to generate generated_image.",
                    },
                ),
                "model": (models, {"tooltip": "Language-model GGUF from models/LLM/GGUF."}),
                "mmproj": (projectors, {"tooltip": "Matching mmproj GGUF from models/LLM/GGUF."}),
                "free_vram_after": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Close the embedded LLM and mmproj after inference to release VRAM.",
                    },
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable internal thinking mode before producing the final JSON.",
                    },
                ),
            },
            "optional": {
                "revision_mode": (
                    ["surgical_edit", "full_rewrite"],
                    {
                        "default": "surgical_edit",
                        "tooltip": "surgical_edit preserves accurate parts of current_prompt; full_rewrite rewrites cleanly.",
                    },
                ),
                "revision_focus": (
                    list(cls.FOCUS_INSTRUCTIONS.keys()),
                    {
                        "default": "all",
                        "tooltip": "Visual aspect to prioritize when auditing discrepancies.",
                    },
                ),
                "fidelity_threshold": (
                    "FLOAT",
                    {
                        "default": 8.5,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.5,
                        "tooltip": "If fidelity_score is below this threshold, needs_revision is True.",
                    },
                ),
                "previous_diff_notes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional diff notes from prior iterations to avoid repeating or oscillating edits.",
                    },
                ),
                "custom_instruction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional extra instruction for the prompt reviser.",
                    },
                ),
                "max_image_dimension": (
                    ["original", "512", "768", "1024", "1536", "2048", "3072", "4096"],
                    {
                        "default": "2048",
                        "tooltip": "Downscale each image so its longest side is at most this size.",
                    },
                ),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "context_size": ("INT", {"default": 8192, "min": 2048, "max": 262144, "step": 1024}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999}),
                "threads": ("INT", {"default": max(1, (os.cpu_count() or 4) // 2), "min": 1, "max": 256}),
                "flash_attention": ("BOOLEAN", {"default": True}),
                "mmproj_on_gpu": ("BOOLEAN", {"default": True}),
                "unload_comfy_models_before": ("BOOLEAN", {"default": True}),
                "model_path_override": ("STRING", {"default": ""}),
                "mmproj_path_override": ("STRING", {"default": ""}),
                "vlm_config": ("GGUF_VL_MODEL",),
            },
        }

    def revise(
        self,
        original_image,
        generated_image,
        current_prompt,
        model,
        mmproj,
        free_vram_after,
        enable_thinking,
        revision_mode="surgical_edit",
        revision_focus="all",
        fidelity_threshold=8.5,
        previous_diff_notes="",
        custom_instruction="",
        max_image_dimension="2048",
        max_tokens=1024,
        temperature=0.15,
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
        config = _resolve_vlm_node_config(
            model=model,
            mmproj=mmproj,
            model_path_override=model_path_override,
            mmproj_path_override=mmproj_path_override,
            context_size=context_size,
            gpu_layers=gpu_layers,
            threads=threads,
            flash_attention=flash_attention,
            mmproj_on_gpu=mmproj_on_gpu,
            vlm_config=vlm_config,
        )

        system_prompt = (
            "You are an expert image-generation prompt auditor and reviser. "
            "You will receive two images: (1) the TARGET ORIGINAL IMAGE and (2) the GENERATED IMAGE "
            "produced from the current prompt. Compare them carefully, diagnose every mismatch, "
            "and return ONLY a valid JSON object with keys: "
            '"fidelity_score" (float 1.0 to 10.0), '
            '"missing_or_wrong" (array of concise strings), '
            '"hallucinated_elements" (array of concise strings), '
            '"notes" (concise string), and '
            '"revised_prompt" (string containing only the ready-to-use image-generation prompt).'
        )

        focus_desc = self.FOCUS_INSTRUCTIONS.get(revision_focus, self.FOCUS_INSTRUCTIONS["all"])
        mode_desc = self.MODE_INSTRUCTIONS.get(revision_mode, self.MODE_INSTRUCTIONS["surgical_edit"])

        def frame_for(batch, index):
            return batch[index] if len(batch) > 1 and index < len(batch) else batch[0]

        try:
            MODEL_MANAGER.load(config, bool(unload_comfy_models_before))
            maximum = 0 if max_image_dimension == "original" else int(max_image_dimension)
            count = max(len(original_image), len(generated_image))
            revised_prompts: list[str] = []
            diff_reports: list[str] = []
            scores: list[float] = []

            for index in range(count):
                orig_uri = image_to_data_uri(frame_for(original_image, index), maximum)
                gen_uri = image_to_data_uri(frame_for(generated_image, index), maximum)

                instruction_lines = [
                    f"Revision focus: {focus_desc}.",
                    f"Editing strategy: {mode_desc}",
                    f"\nCurrent prompt:\n{current_prompt.strip() or '(empty)'}",
                ]
                if previous_diff_notes.strip():
                    instruction_lines.append(
                        f"\nPrevious iteration notes (avoid oscillating on resolved items):\n{previous_diff_notes.strip()}"
                    )
                if custom_instruction.strip():
                    instruction_lines.append(f"\nAdditional user instruction:\n{custom_instruction.strip()}")

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "\n".join(instruction_lines) + "\n\nImage 1 — TARGET ORIGINAL IMAGE:"},
                            {"type": "image_url", "image_url": {"url": orig_uri}},
                            {"type": "text", "text": "Image 2 — GENERATED IMAGE (from current prompt):"},
                            {"type": "image_url", "image_url": {"url": gen_uri}},
                        ],
                    },
                ]
                raw = MODEL_MANAGER.chat(
                    messages,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    seed=int(seed),
                    enable_thinking=bool(enable_thinking),
                )
                revised, diff_report, score = _parse_revision_response(raw, current_prompt)
                revised_prompts.append(revised)
                diff_reports.append(diff_report)
                scores.append(score)

            separator = "\n\n---\n\n"
            avg_score = round(sum(scores) / len(scores), 2) if scores else 5.0
            needs_revision = any(s < float(fidelity_threshold) for s in scores)
            return (
                separator.join(revised_prompts),
                separator.join(diff_reports),
                float(avg_score),
                bool(needs_revision),
            )
        except EmbeddedLlamaError as exc:
            raise RuntimeError(f"GGUF Vision Prompt Reviser failed: {exc}") from exc
        finally:
            if free_vram_after:
                MODEL_MANAGER.unload()


class ZiggartPromptRevisionGate:
    """Selects between current and revised prompt based on needs_revision and accumulates diff notes."""

    DESCRIPTION = (
        "Utility gate for iterative prompt revision loops: outputs revised_prompt while needs_revision is True "
        "(or keeps current_prompt once converged) and appends latest_diff_report to previous_notes."
    )
    CATEGORY = "Ziggart/LLM"
    FUNCTION = "select"
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("active_prompt", "accumulated_notes", "continue_revision")
    OUTPUT_TOOLTIPS = (
        "Selected prompt: revised_prompt when needs_revision is True (or force_use_revised), else current_prompt.",
        "Cumulative diff notes combining previous_notes and latest_diff_report for the next iteration.",
        "Pass-through boolean indicating whether another revision pass is recommended.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_prompt": ("STRING", {"default": "", "multiline": True}),
                "revised_prompt": ("STRING", {"default": "", "multiline": True}),
                "needs_revision": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "latest_diff_report": ("STRING", {"default": "", "multiline": True}),
                "previous_notes": ("STRING", {"default": "", "multiline": True}),
                "iteration_label": ("STRING", {"default": ""}),
                "force_use_revised": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Always output revised_prompt even when needs_revision is False.",
                    },
                ),
            },
        }

    def select(
        self,
        current_prompt: str,
        revised_prompt: str,
        needs_revision: bool,
        latest_diff_report: str = "",
        previous_notes: str = "",
        iteration_label: str = "",
        force_use_revised: bool = False,
    ) -> tuple[str, str, bool]:
        use_revised = bool(needs_revision) or bool(force_use_revised)
        active_prompt = revised_prompt if (use_revised and revised_prompt.strip()) else current_prompt

        notes_chunks: list[str] = []
        if previous_notes.strip():
            notes_chunks.append(previous_notes.strip())
        if latest_diff_report.strip():
            label = iteration_label.strip()
            entry = f"[{label}]\n{latest_diff_report.strip()}" if label else latest_diff_report.strip()
            notes_chunks.append(entry)

        accumulated_notes = "\n\n".join(notes_chunks)
        return (active_prompt, accumulated_notes, bool(needs_revision))


# ---------------------------------------------------------------------------
# Folder Image Queue Loader & Vision Prompt TXT Saver Helpers
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".gif",
        ".avif",
        ".tga",
        ".ico",
        ".jfif",
    }
)

FOLDER_SORT_METHODS = [
    "natural_alphabetical",
    "date_modified_asc",
    "date_modified_desc",
    "random_seeded",
]

TXT_WRITE_MODES = [
    "append_new_line",
    "overwrite_file",
]

_NATURAL_SPLIT_RE = re.compile(r"(\d+)")
_MD_HEADER_ONLY_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+.*|\*\*[^*\n]+:\*\*|__[^_\n]+:__)\s*$"
)
_MD_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+")
_MD_BULLET_BOLD_LABEL_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])\s*(?:\*\*[^*\n]+:\*\*|__[^_\n]+:__)\s*"
)
_MD_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_MD_BOLD_INLINE_LABEL_RE = re.compile(r"\*\*[^*\n]+:\*\*\s*")
_MD_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{2})([^*_\n]+?)\1")


def _send_node_feedback(node_id: Any, widget_name: str, value: Any) -> None:
    """Sends a synchronous WebSocket event to update a widget value in the ComfyUI frontend."""
    if node_id is None:
        return
    if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
        try:
            PromptServer.instance.send_sync(
                "ziggart-node-feedback",
                {
                    "node_id": str(node_id),
                    "widget_name": widget_name,
                    "type": "STRING",
                    "value": value,
                },
            )
        except Exception:
            pass


def _natural_sort_key(path: Path) -> list[Any]:
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in _NATURAL_SPLIT_RE.split(str(path))
    ]


def _resolve_image_folder(folder_path: str) -> Path:
    cleaned = (folder_path or "").strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError(
            "No folder_path provided. Please enter a folder path containing images."
        )
    expanded = Path(os.path.expandvars(os.path.expanduser(cleaned)))
    if expanded.is_dir():
        return expanded.resolve()
    if not expanded.is_absolute() and folder_paths is not None:
        get_input_dir = getattr(folder_paths, "get_input_directory", None)
        if callable(get_input_dir):
            candidate = Path(get_input_dir()) / expanded
            if candidate.is_dir():
                return candidate.resolve()
    raise ValueError(f"Image folder does not exist or is not a directory: {expanded}")


def _matches_filename_filter(filename: str, pattern: str) -> bool:
    pat = (pattern or "").strip()
    if not pat:
        return True
    low_name = filename.lower()
    low_pat = pat.lower()
    if any(ch in low_pat for ch in ("*", "?", "[")):
        return fnmatch.fnmatch(low_name, low_pat)
    return low_pat in low_name


def _has_nonempty_sidecar_txt(image_path: Path) -> bool:
    txt_path = image_path.with_suffix(".txt")
    try:
        return txt_path.is_file() and txt_path.stat().st_size > 0
    except OSError:
        return False


def scan_image_folder(
    folder_path: str,
    include_subfolders: bool = False,
    skip_existing_txt: bool = False,
    sort_method: str = "natural_alphabetical",
    filename_filter: str = "",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Scans a directory for supported image files, applies optional subfolder recursion,
    filename filtering, existing .txt skipping, and deterministic sorting.
    """
    folder = _resolve_image_folder(folder_path)
    iterator = folder.rglob("*") if include_subfolders else folder.iterdir()

    all_matched: list[Path] = []
    for entry in iterator:
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        if entry.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if not _matches_filename_filter(entry.name, filename_filter):
            continue
        all_matched.append(entry.resolve())

    total_found = len(all_matched)
    skipped_txt_count = 0
    if skip_existing_txt:
        filtered: list[Path] = []
        for img_path in all_matched:
            if _has_nonempty_sidecar_txt(img_path):
                skipped_txt_count += 1
            else:
                filtered.append(img_path)
    else:
        filtered = all_matched

    if sort_method == "date_modified_asc":
        filtered.sort(key=lambda p: (_safe_mtime(p), _natural_sort_key(p)))
    elif sort_method == "date_modified_desc":
        filtered.sort(key=lambda p: (-_safe_mtime(p), _natural_sort_key(p)))
    elif sort_method == "random_seeded":
        filtered.sort(key=_natural_sort_key)
        rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
        rng.shuffle(filtered)
    else:
        filtered.sort(key=_natural_sort_key)

    return {
        "resolved_folder": folder,
        "files": filtered,
        "total_found": total_found,
        "skipped_txt_count": skipped_txt_count,
        "queues_needed": len(filtered),
    }


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _resolve_active_index(index: int, total: int, loop_index: bool) -> int:
    if total <= 0:
        return 0
    idx = max(0, int(index))
    if loop_index:
        return idx % total
    return min(idx, total - 1)


def _build_folder_scan_status(
    scan_result: dict[str, Any],
    image_index: int = 0,
    loop_index: bool = False,
    executing: bool = False,
) -> dict[str, Any]:
    files: list[Path] = scan_result["files"]
    total_found: int = scan_result["total_found"]
    skipped_txt: int = scan_result["skipped_txt_count"]
    queues_needed: int = scan_result["queues_needed"]
    folder: Path = scan_result["resolved_folder"]

    if queues_needed == 0:
        if total_found > 0 and skipped_txt > 0:
            msg = (
                f"✅ All {total_found} images in '{folder.name}' already have a .txt file "
                f"(0 queues needed)."
            )
        else:
            msg = f"⚠️ No matching images found in '{folder}' (0 queues needed)."
        return {
            "active_index": 0,
            "remaining_queues": 0,
            "current_filename": "",
            "current_file_path": "",
            "status_text": msg,
            "queue_summary": msg,
        }

    active_idx = _resolve_active_index(image_index, queues_needed, loop_index)
    remaining_from_index = max(0, queues_needed - active_idx - (1 if executing else 0))
    current_path = files[active_idx]
    skip_note = f" ({skipped_txt} skipped with existing .txt)" if skipped_txt > 0 else ""

    prefix_icon = "🔄 Processing" if executing else "📂 Ready"
    status_lines = [
        f"{prefix_icon} [{active_idx + 1}/{queues_needed}]: {current_path.name}",
        f"📊 Total Queues Needed: {queues_needed} (from {total_found} images{skip_note})",
        f"⏳ Remaining from Index {active_idx}: {max(0, queues_needed - active_idx)} run(s) "
        f"(after current: {max(0, queues_needed - active_idx - 1)})",
    ]
    status_text = "\n".join(status_lines)

    preview_limit = 50
    preview_items = []
    for idx, fpath in enumerate(files[:preview_limit]):
        marker = "👉 " if idx == active_idx else "   "
        preview_items.append(f"{marker}[{idx}] {fpath.name}")
    if len(files) > preview_limit:
        preview_items.append(f"   ... and {len(files) - preview_limit} more image(s)")

    summary_text = (
        status_text
        + f"\n📁 Folder: {folder}\n"
        + "----------------------------------------\n"
        + "\n".join(preview_items)
    )
    return {
        "active_index": active_idx,
        "remaining_queues": max(0, queues_needed - active_idx - 1),
        "remaining_to_queue": max(0, queues_needed - active_idx),
        "current_filename": current_path.name,
        "current_file_path": str(current_path),
        "status_text": status_text,
        "queue_summary": summary_text,
    }


def _load_pil_image_as_tensor_and_mask(image_path: Path) -> tuple[Any, Any]:
    with Image.open(image_path) as raw_img:
        pil_img = ImageOps.exif_transpose(raw_img)
        if pil_img.mode == "I":
            pil_img = pil_img.point(lambda i: i * (1 / 255))
        rgb_img = pil_img.convert("RGB")
        rgb_np = np.asarray(rgb_img, dtype=np.float32) / 255.0
        image_batch = rgb_np[np.newaxis, ...]  # (1, H, W, 3)

        if "A" in pil_img.getbands():
            alpha_np = np.asarray(pil_img.getchannel("A"), dtype=np.float32) / 255.0
            mask_batch = (1.0 - alpha_np)[np.newaxis, ...]
        elif pil_img.mode == "P" and "transparency" in pil_img.info:
            alpha_np = (
                np.asarray(pil_img.convert("RGBA").getchannel("A"), dtype=np.float32)
                / 255.0
            )
            mask_batch = (1.0 - alpha_np)[np.newaxis, ...]
        else:
            mask_batch = np.zeros(
                (1, rgb_np.shape[0], rgb_np.shape[1]), dtype=np.float32
            )

    if torch is not None:
        return torch.from_numpy(image_batch), torch.from_numpy(mask_batch)
    return image_batch, mask_batch


class ZiggartFolderImageQueueLoader:
    """
    Scans a folder of images (jpeg, png, webp, etc.), reports how many queues are needed
    to process all images one-by-one into Ziggart Vision, and loads one image per queue pass.
    """

    DESCRIPTION = (
        "Lists all images in a folder, tells you exactly how many queues are needed to parse "
        "them one-by-one into GGUF Vision Prompt, and loads one image per run as image_index increments."
    )
    CATEGORY = "Ziggart/LLM"
    FUNCTION = "load_image"
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "filename_stem",
        "filename",
        "file_path",
        "directory",
        "current_index",
        "total_queues_needed",
        "remaining_queues",
        "queue_summary",
    )
    OUTPUT_TOOLTIPS = (
        "Loaded RGB image tensor [1, H, W, 3] for GGUF Vision Prompt.",
        "Alpha channel mask [1, H, W] (or zeros if opaque).",
        "Image filename without extension (e.g. 'photo_001').",
        "Image filename with extension (e.g. 'photo_001.webp').",
        "Full resolved path to the active image file.",
        "Resolved directory containing the active image file.",
        "Active 0-based index loaded on this run.",
        "Total number of queues required to process all matching images in the folder.",
        "Number of queues remaining after this image.",
        "Full status report and indexed file list preview.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to the folder containing images (JPEG, PNG, WEBP, BMP, TIFF, GIF, AVIF, etc.).",
                    },
                ),
                "image_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "control_after_generate": True,
                        "tooltip": "0-based image index. Set control_after_generate to 'increment' (or click 'Queue All Remaining Images').",
                    },
                ),
                "sort_method": (
                    FOLDER_SORT_METHODS,
                    {
                        "default": "natural_alphabetical",
                        "tooltip": "How images in the folder are ordered across queue passes.",
                    },
                ),
                "include_subfolders": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Recursively scan subfolders inside folder_path.",
                    },
                ),
                "skip_existing_txt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Skip images that already have a non-empty companion <image_name>.txt file next to them.",
                    },
                ),
                "loop_index": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, wraps image_index modulo total images; otherwise clamps at the last image.",
                    },
                ),
            },
            "optional": {
                "filename_filter": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional substring or glob filter (e.g. '*.webp' or 'V3-Vision').",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                        "tooltip": "Seed used when sort_method is set to 'random_seeded'.",
                    },
                ),
                "status_text": (
                    "STRING",
                    {
                        "default": "Enter a folder_path and click '🔍 Scan Folder & Count Queues'.",
                        "multiline": True,
                        "tooltip": "Live folder scan and queue progress readout.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        folder_path: str = "",
        image_index: int = 0,
        sort_method: str = "natural_alphabetical",
        include_subfolders: bool = False,
        skip_existing_txt: bool = False,
        loop_index: bool = False,
        filename_filter: str = "",
        seed: int = 42,
        status_text: str = "",
        unique_id: Any = None,
    ) -> Any:
        try:
            scan = scan_image_folder(
                folder_path=folder_path,
                include_subfolders=include_subfolders,
                skip_existing_txt=skip_existing_txt,
                sort_method=sort_method,
                filename_filter=filename_filter,
                seed=seed,
            )
            files = scan["files"]
            if not files:
                return f"empty:{folder_path}:{include_subfolders}:{skip_existing_txt}"
            active_idx = _resolve_active_index(image_index, len(files), loop_index)
            target = files[active_idx]
            return f"{target}:{_safe_mtime(target)}:{active_idx}:{len(files)}"
        except Exception:
            return float("nan")

    def load_image(
        self,
        folder_path: str,
        image_index: int = 0,
        sort_method: str = "natural_alphabetical",
        include_subfolders: bool = False,
        skip_existing_txt: bool = False,
        loop_index: bool = False,
        filename_filter: str = "",
        seed: int = 42,
        status_text: str = "",
        unique_id: Any = None,
    ):
        scan = scan_image_folder(
            folder_path=folder_path,
            include_subfolders=bool(include_subfolders),
            skip_existing_txt=bool(skip_existing_txt),
            sort_method=sort_method,
            filename_filter=filename_filter,
            seed=int(seed),
        )
        info = _build_folder_scan_status(
            scan,
            image_index=int(image_index),
            loop_index=bool(loop_index),
            executing=True,
        )
        _send_node_feedback(unique_id, "status_text", info["status_text"])

        files: list[Path] = scan["files"]
        if not files:
            raise ValueError(info["status_text"])

        active_idx: int = info["active_index"]
        active_file: Path = files[active_idx]
        image_tensor, mask_tensor = _load_pil_image_as_tensor_and_mask(active_file)

        return (
            image_tensor,
            mask_tensor,
            active_file.stem,
            active_file.name,
            str(active_file),
            str(active_file.parent),
            int(active_idx),
            int(scan["queues_needed"]),
            int(info["remaining_queues"]),
            info["queue_summary"],
        )


# ---------------------------------------------------------------------------
# Node 2: ZiggartVisionPromptTxtSaver
# ---------------------------------------------------------------------------


def format_vision_prompt_text(
    text: str,
    flatten_to_single_line: bool = True,
    clean_markdown_formatting: bool = False,
    line_prefix: str = "",
    line_suffix: str = "",
) -> str:
    """
    Cleans and optionally flattens a Vision LLM output prompt into a single line with zero
    internal line breaks or empty lines (matching the Krea2 one-prompt-per-line format).
    """
    cleaned = strip_thinking_tags(str(text or "")).strip()
    if not cleaned:
        return ""

    if clean_markdown_formatting:
        processed_lines: list[str] = []
        for raw_line in cleaned.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if _MD_HEADER_ONLY_LINE_RE.match(stripped):
                continue
            line = _MD_HEADING_PREFIX_RE.sub("", stripped)
            line = _MD_BULLET_BOLD_LABEL_RE.sub("", line)
            line = _MD_BULLET_PREFIX_RE.sub("", line)
            line = _MD_BOLD_INLINE_LABEL_RE.sub("", line)
            line = _MD_EMPHASIS_RE.sub(r"\2", line)
            line = line.replace("**", "").strip()
            if line:
                processed_lines.append(line)
        cleaned = "\n".join(processed_lines)

    if flatten_to_single_line:
        cleaned = " ".join(cleaned.split())
    else:
        non_empty_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        cleaned = "\n".join(non_empty_lines)

    if not cleaned:
        return ""

    prefix = str(line_prefix or "")
    suffix = str(line_suffix or "")
    if flatten_to_single_line:
        prefix = " ".join(prefix.split())
        suffix = " ".join(suffix.split())
        parts = [p for p in (prefix, cleaned, suffix) if p]
        return " ".join(parts).strip()

    return f"{prefix}{cleaned}{suffix}".strip()


def _resolve_output_folder(output_folder: str, source_image_path: str = "") -> Path:
    cleaned = (output_folder or "").strip().strip('"').strip("'")
    if cleaned:
        path = Path(os.path.expandvars(os.path.expanduser(cleaned)))
        if not path.is_absolute() and folder_paths is not None:
            get_out_dir = getattr(folder_paths, "get_output_directory", None)
            if callable(get_out_dir):
                path = Path(get_out_dir()) / path
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    src_clean = (source_image_path or "").strip().strip('"').strip("'")
    if src_clean:
        src_path = Path(os.path.expandvars(os.path.expanduser(src_clean)))
        if src_path.is_dir():
            src_path.mkdir(parents=True, exist_ok=True)
            return src_path.resolve()
        if src_path.parent and str(src_path.parent) not in ("", "."):
            src_path.parent.mkdir(parents=True, exist_ok=True)
            return src_path.parent.resolve()

    if folder_paths is not None:
        get_out_dir = getattr(folder_paths, "get_output_directory", None)
        if callable(get_out_dir):
            out = Path(get_out_dir()).resolve()
            out.mkdir(parents=True, exist_ok=True)
            return out

    fallback = Path.cwd() / "output"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


def _read_existing_nonempty_lines(file_path: Path) -> list[str]:
    if not file_path.is_file():
        return []
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        return [line.rstrip("\r\n") for line in fh if line.strip()]


class ZiggartVisionPromptTxtSaver:
    """
    Takes the prompt output from GGUF Vision Prompt, optionally flattens multiline/markdown
    descriptions into a single clean line, and appends a new line to a master .txt file
    on every queue pass (with optional per-image sidecar .txt saving).
    """

    DESCRIPTION = (
        "Receives prompt text from GGUF Vision Prompt, optionally converts multi-line or markdown "
        "descriptions into a single line, and appends each prompt as a new line in a master .txt "
        "file (guaranteed 1 prompt = 1 line, 0 empty lines)."
    )
    CATEGORY = "Ziggart/LLM"
    FUNCTION = "save_prompt"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "formatted_prompt",
        "master_file_path",
        "total_lines_in_file",
        "status_report",
    )
    OUTPUT_TOOLTIPS = (
        "The cleaned/flattened prompt text.",
        "Full path to the master .txt file.",
        "Total non-empty prompt lines currently stored in the master .txt file.",
        "Detailed status report of the save operation.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Connect the 'answer' output from GGUF Vision Prompt.",
                    },
                ),
                "output_folder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Destination folder for the master .txt file. If left blank, uses source_image_path's folder or ComfyUI's output folder.",
                    },
                ),
                "master_filename": (
                    "STRING",
                    {
                        "default": "all_prompts_krea2.txt",
                        "tooltip": "Filename for the consolidated .txt file ('.txt' is added automatically if omitted).",
                    },
                ),
                "write_mode": (
                    TXT_WRITE_MODES,
                    {
                        "default": "append_new_line",
                        "tooltip": "'append_new_line' adds a new line on each queue pass; 'overwrite_file' replaces the file.",
                    },
                ),
                "flatten_to_single_line": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Collapses multiline Vision prompts and line breaks into a single line (1 prompt = 1 line, 0 empty lines).",
                    },
                ),
                "clean_markdown_formatting": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Strips Markdown section headers (**Attire:**, ### Heading), bullet points (- ), and bold/italic markers before flattening.",
                    },
                ),
                "skip_duplicate_lines": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prevents writing a prompt line if the exact same line is already present in the master .txt file.",
                    },
                ),
                "save_individual_sidecar_txt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Also saves an individual <image_name>.txt file per pass (works with skip_existing_txt on the Folder Loader).",
                    },
                ),
            },
            "optional": {
                "line_prefix": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional text prepended to every prompt line before saving.",
                    },
                ),
                "line_suffix": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional text appended to every prompt line before saving.",
                    },
                ),
                "source_image_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: connect 'file_path' (or 'filename_stem') from Ziggart Folder Image Batch & Queue Loader.",
                    },
                ),
                "status_text": (
                    "STRING",
                    {
                        "default": "Ready to save Vision prompts line-by-line.",
                        "multiline": True,
                        "tooltip": "Live readout of saved line count and file destination.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def save_prompt(
        self,
        prompt_text: str,
        output_folder: str = "",
        master_filename: str = "all_prompts_krea2.txt",
        write_mode: str = "append_new_line",
        flatten_to_single_line: bool = True,
        clean_markdown_formatting: bool = False,
        skip_duplicate_lines: bool = True,
        save_individual_sidecar_txt: bool = False,
        line_prefix: str = "",
        line_suffix: str = "",
        source_image_path: str = "",
        status_text: str = "",
        unique_id: Any = None,
    ):
        raw_text = str(prompt_text or "").strip()
        if "\n\n---\n\n" in raw_text:
            raw_segments = [seg for seg in raw_text.split("\n\n---\n\n") if seg.strip()]
        else:
            raw_segments = [raw_text] if raw_text else []

        formatted_entries: list[str] = []
        for seg in raw_segments:
            formatted = format_vision_prompt_text(
                seg,
                flatten_to_single_line=bool(flatten_to_single_line),
                clean_markdown_formatting=bool(clean_markdown_formatting),
                line_prefix=line_prefix,
                line_suffix=line_suffix,
            )
            if formatted:
                formatted_entries.append(formatted)

        dest_dir = _resolve_output_folder(output_folder, source_image_path)
        fname = (master_filename or "all_prompts_krea2.txt").strip() or "all_prompts_krea2.txt"
        if not fname.lower().endswith(".txt"):
            fname += ".txt"
        master_path = (dest_dir / fname).resolve()
        master_path.parent.mkdir(parents=True, exist_ok=True)

        existing_lines = (
            []
            if write_mode == "overwrite_file"
            else _read_existing_nonempty_lines(master_path)
        )
        existing_set = set(existing_lines)

        added_count = 0
        skipped_dup_count = 0
        for entry in formatted_entries:
            if flatten_to_single_line:
                if skip_duplicate_lines and entry in existing_set:
                    skipped_dup_count += 1
                    continue
                existing_lines.append(entry)
                existing_set.add(entry)
                added_count += 1
            else:
                entry_lines = [ln for ln in entry.splitlines() if ln.strip()]
                candidate_block = "\n".join(entry_lines)
                if skip_duplicate_lines and candidate_block in existing_set:
                    skipped_dup_count += 1
                    continue
                existing_lines.extend(entry_lines)
                existing_set.add(candidate_block)
                added_count += len(entry_lines)

        # Write with zero empty lines and trailing newline
        final_content = "\n".join(ln for ln in existing_lines if ln.strip())
        if final_content:
            final_content += "\n"
        with open(master_path, "w", encoding="utf-8") as fh:
            fh.write(final_content)

        total_lines = len([ln for ln in existing_lines if ln.strip()])
        combined_formatted = "\n".join(formatted_entries)

        sidecar_saved_path = ""
        if save_individual_sidecar_txt and combined_formatted:
            sidecar_path = _resolve_sidecar_txt_path(
                dest_dir=dest_dir,
                source_image_path=source_image_path,
                master_stem=master_path.stem,
                line_number=total_lines,
            )
            with open(sidecar_path, "w", encoding="utf-8") as sfh:
                sfh.write(combined_formatted + "\n")
            sidecar_saved_path = str(sidecar_path)

        preview_snippet = (
            combined_formatted[:120] + ("..." if len(combined_formatted) > 120 else "")
            if combined_formatted
            else "(empty input)"
        )
        action_label = (
            f"✅ Saved {added_count} new line(s)"
            if added_count > 0
            else (
                f"⏭️ Skipped duplicate line"
                if skipped_dup_count > 0
                else "⚠️ Empty prompt (nothing written)"
            )
        )
        sidecar_note = f"\n📄 Sidecar: {sidecar_saved_path}" if sidecar_saved_path else ""
        report = (
            f"{action_label} | Total Lines in File: {total_lines}\n"
            f"📁 Master File: {master_path}{sidecar_note}\n"
            f"📝 Last Line: {preview_snippet}"
        )
        _send_node_feedback(unique_id, "status_text", report)

        return {
            "ui": {"text": [report]},
            "result": (
                combined_formatted,
                str(master_path),
                int(total_lines),
                report,
            ),
        }


def _resolve_sidecar_txt_path(
    dest_dir: Path,
    source_image_path: str,
    master_stem: str,
    line_number: int,
) -> Path:
    src_clean = (source_image_path or "").strip().strip('"').strip("'")
    if src_clean:
        src_p = Path(os.path.expandvars(os.path.expanduser(src_clean)))
        if src_p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS or src_p.is_absolute():
            if src_p.parent and src_p.parent.is_dir():
                return (src_p.parent / f"{src_p.stem}.txt").resolve()
        stem = src_p.stem or src_clean
        return (dest_dir / f"{stem}.txt").resolve()

    idx = max(1, int(line_number))
    candidate = dest_dir / f"{master_stem}__{idx:05d}.txt"
    while candidate.exists():
        idx += 1
        candidate = dest_dir / f"{master_stem}__{idx:05d}.txt"
    return candidate.resolve()


_VLM_BOILERPLATE_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:sure[!,.]?\s*)?(?:here\s+is|this\s+is)\s+(?:a\s+|the\s+)?(?:detailed\s+|concise\s+|visual\s+)?(?:description|caption|prompt|analysis)(?:\s+of\s+the\s+(?:image|picture|photo|scene))?\s*[:\-–—]\s*"
    r"|(?:in\s+this\s+(?:image|picture|photo|scene|frame|shot)\s*[,:\-–—]?\s*(?:we\s+see\s+|there\s+is\s+)?)"
    r"|(?:the\s+(?:image|picture|photo|scene|illustration)\s+(?:shows|depicts|features|presents|portrays|captures)\s+)"
    r"|(?:this\s+(?:image|picture|photo|scene|illustration)\s+(?:shows|depicts|features|presents|portrays|captures)\s+)"
    r")+",
    re.IGNORECASE,
)


def _estimate_caption_tokens(text: str) -> int:
    """Fast token estimator for CLIP / Qwen3VL / T5 captions."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    words = re.findall(r"\w+|[^\w\s]", cleaned)
    return max(1, int(round(len(words) * 1.12)))


def curate_single_caption_text(
    raw_text: str,
    *,
    trigger_word_prefix: str = "",
    trigger_word_suffix: str = "",
    strip_vlm_boilerplate: bool = True,
    clean_markdown: bool = True,
    find_and_replace_rules: str = "",
    banned_phrases: str = "",
    format_mode: str = "keep_original",
) -> tuple[str, dict[str, Any]]:
    """
    Cleans, normalizes, and injects trigger words into a single dataset caption string.
    Returns (curated_text, metadata_dict).
    """
    working = format_vision_prompt_text(
        raw_text or "",
        flatten_to_single_line=True,
        clean_markdown_formatting=bool(clean_markdown),
    )

    boilerplate_stripped = False
    if strip_vlm_boilerplate and working:
        new_working = _VLM_BOILERPLATE_PREFIX_RE.sub("", working).strip()
        if new_working and new_working != working:
            boilerplate_stripped = True
            working = new_working

    replacements_applied = 0
    if find_and_replace_rules and find_and_replace_rules.strip():
        for raw_rule in find_and_replace_rules.splitlines():
            rule = raw_rule.strip()
            if not rule or rule.startswith("#"):
                continue
            sep = "->" if "->" in rule else ("=>" if "=>" in rule else ("=" if "=" in rule else None))
            if not sep:
                continue
            src_pat, dst_val = rule.split(sep, 1)
            src_pat = src_pat.strip()
            dst_val = dst_val.strip()
            if not src_pat:
                continue
            pattern = re.compile(re.escape(src_pat), re.IGNORECASE)
            updated, n_subs = pattern.subn(dst_val, working)
            if n_subs > 0:
                replacements_applied += n_subs
                working = updated

    banned_removed = 0
    if banned_phrases and banned_phrases.strip():
        phrases = [
            p.strip()
            for line in banned_phrases.splitlines()
            for p in line.split(",")
            if p.strip()
        ]
        for phrase in phrases:
            pat = re.compile(r"(?:^|(?<=[\s,;]))" + re.escape(phrase) + r"(?=$|[\s,;.])", re.IGNORECASE)
            updated, n_rem = pat.subn("", working)
            if n_rem > 0:
                banned_removed += n_rem
                working = updated

    # Collapse repeated punctuation and whitespace left after removals
    working = re.sub(r"\s*,\s*(?:,\s*)+", ", ", working)
    working = re.sub(r"\s{2,}", " ", working).strip(" ,;-")

    if format_mode == "normalized_tags" and working:
        raw_tags = [t.strip(" .") for t in working.split(",") if t.strip(" .")]
        dedup_tags: list[str] = []
        seen_t: set[str] = set()
        for t in raw_tags:
            low_t = t.lower()
            if low_t not in seen_t:
                seen_t.add(low_t)
                dedup_tags.append(t)
        working = ", ".join(dedup_tags)
    elif format_mode == "natural_prose" and working:
        working = working[0].upper() + working[1:] if len(working) > 1 else working.upper()
        if not working.endswith((".", "!", "?")):
            working += "."

    # Idempotent Trigger Word Prefix & Suffix Injection
    pref = (trigger_word_prefix or "").strip().strip(",")
    suff = (trigger_word_suffix or "").strip().strip(",")
    if pref:
        if not working:
            working = pref
        elif not working.lower().startswith(pref.lower()):
            sep = ", " if not pref.endswith((".", ",", ":")) else " "
            working = f"{pref}{sep}{working}"
    if suff:
        if not working:
            working = suff
        elif not working.rstrip(".!? ").lower().endswith(suff.lower()):
            if format_mode == "natural_prose" and working.endswith("."):
                working = f"{working[:-1]}, {suff}."
            else:
                working = f"{working.rstrip(', ')}, {suff}"

    tok_count = _estimate_caption_tokens(working)
    clip_chunks = max(1, (tok_count + 74) // 75) if tok_count > 0 else 0
    return working, {
        "boilerplate_stripped": boilerplate_stripped,
        "replacements_applied": replacements_applied,
        "banned_removed": banned_removed,
        "estimated_tokens": tok_count,
        "clip_77_chunks": clip_chunks,
    }


def _render_dataset_curator_contact_sheet(
    items: list[dict[str, Any]],
    summary_stats: dict[str, Any],
    max_token_budget: int,
    max_tiles: int = 16,
) -> Any:
    """
    Renders a visual QA contact sheet [1, H, W, 3] displaying dataset image thumbnails,
    token count badges (green=OK, amber=near budget, red=over budget or missing),
    and a top summary banner.
    """
    shown = items[: max(1, min(36, int(max_tiles)))]
    n = len(shown)
    cols = max(1, min(4, int(np.ceil(np.sqrt(max(1, n))))))
    rows = max(1, int(np.ceil(max(1, n) / float(cols))))
    tile_w, tile_h = 220, 220
    header_h = 54
    canvas_w = max(440, cols * tile_w)
    canvas_h = header_h + rows * tile_h

    sheet = Image.new("RGB", (canvas_w, canvas_h), (18, 20, 28))
    draw = ImageDraw.Draw(sheet)

    # Top Summary Banner
    draw.rectangle([0, 0, canvas_w, header_h], fill=(28, 32, 46))
    title_line = (
        f"ZIGGART DATASET QA | Images: {summary_stats.get('total_images', 0)} | "
        f"Captioned: {summary_stats.get('captioned_count', 0)} | "
        f"Missing: {summary_stats.get('missing_txt_count', 0)}"
    )
    sub_line = (
        f"Avg Tokens: {summary_stats.get('avg_tokens', 0):.1f} (Budget: {max_token_budget}) | "
        f"Over Budget: {summary_stats.get('over_budget_count', 0)} | "
        f"Duplicates: {summary_stats.get('duplicate_caption_count', 0)} | "
        f"Mode: {summary_stats.get('write_mode', 'dry_run')}"
    )
    draw.text((12, 8), title_line, fill=(235, 242, 255))
    draw.text((12, 28), sub_line, fill=(148, 163, 184))

    for idx, rec in enumerate(shown):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = header_h + r * tile_h
        pad = 6
        box_w = tile_w - pad * 2
        box_h = tile_h - pad * 2

        try:
            with Image.open(rec["image_path"]) as im:
                im_rgb = ImageOps.exif_transpose(im).convert("RGB")
                im_thumb = ImageOps.fit(im_rgb, (box_w, box_h), method=Image.Resampling.BILINEAR)
        except Exception:
            im_thumb = Image.new("RGB", (box_w, box_h), (40, 44, 58))

        sheet.paste(im_thumb, (x0 + pad, y0 + pad))

        toks = int(rec.get("estimated_tokens", 0))
        missing = bool(rec.get("missing_sidecar", False))
        if missing or toks == 0:
            badge_col = (225, 65, 85)
            status_tag = "MISSING TXT"
        elif toks > max_token_budget:
            badge_col = (235, 75, 75)
            status_tag = f"{toks} tok (OVER)"
        elif toks > int(max_token_budget * 0.85):
            badge_col = (235, 175, 50)
            status_tag = f"{toks} tok (HIGH)"
        else:
            badge_col = (45, 195, 115)
            status_tag = f"{toks} tok (OK)"

        draw.rectangle(
            [x0 + pad, y0 + pad, x0 + pad + box_w, y0 + pad + box_h],
            outline=badge_col,
            width=2,
        )
        draw.rectangle(
            [x0 + pad, y0 + pad, x0 + pad + min(box_w, 130), y0 + pad + 20],
            fill=badge_col,
        )
        draw.text((x0 + pad + 5, y0 + pad + 4), status_tag, fill=(12, 14, 20))

        # Bottom filename & caption snippet strip
        draw.rectangle(
            [x0 + pad, y0 + pad + box_h - 36, x0 + pad + box_w, y0 + pad + box_h],
            fill=(14, 16, 24),
        )
        fname_short = Path(rec["image_path"]).name[:26]
        cap_short = (rec.get("curated_caption") or "[No caption]")[:30]
        draw.text((x0 + pad + 5, y0 + pad + box_h - 33), fname_short, fill=(226, 232, 240))
        draw.text((x0 + pad + 5, y0 + pad + box_h - 17), cap_short, fill=(148, 163, 184))

    arr = np.asarray(sheet, dtype=np.float32) / 255.0
    batch = arr[None, ...]
    if torch is not None:
        return torch.from_numpy(batch)
    return batch


class ZiggartDatasetCaptionCuratorAndTriggerInjector:
    """
    Dataset Caption Curator, Trigger Word Injector & Token QA Inspector for LoRA / Finetuning datasets.
    Scans a folder of images + .txt sidecars, strips VLM boilerplate phrases, applies search/replace
    and banned-phrase rules, injects trigger words idempotently, audits CLIP/Qwen3VL token budgets,
    and outputs a visual QA Contact Sheet + JSON report.
    """

    DESCRIPTION = (
        "Batch Dataset Caption Curator & Trigger Injector for LoRA / Model training folders. "
        "Scans images + .txt sidecars, strips VLM boilerplate ('In this image...'), injects trigger words "
        "without duplication, enforces token budgets, flags missing/duplicate captions, and renders a QA contact sheet."
    )
    CATEGORY = "Ziggart/LLM"
    FUNCTION = "curate_dataset"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "qa_contact_sheet",
        "curated_captions_joined",
        "qa_report_json",
        "processed_count",
    )
    OUTPUT_TOOLTIPS = (
        "Visual QA Contact Sheet showing dataset thumbnails with color-coded token budget & sidecar status badges.",
        "All curated captions joined one-per-line (ready for wildcard files or master prompt lists).",
        "Structured JSON QA report detailing token stats, missing sidecars, duplicates, and over-budget files.",
        "Total number of dataset captions processed.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Path to dataset folder containing images and .txt sidecar captions.",
                    },
                ),
                "write_mode": (
                    [
                        "dry_run_preview_only",
                        "write_in_place_atomic",
                        "export_to_output_folder",
                    ],
                    {
                        "default": "dry_run_preview_only",
                        "tooltip": "dry_run_preview_only: inspects without modifying files. write_in_place_atomic: atomically updates .txt sidecars beside images. export_to_output_folder: writes curated .txt files to output_folder_override.",
                    },
                ),
                "trigger_word_prefix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional LoRA trigger word or style tag prepended to every caption (idempotent: never duplicated if already present).",
                    },
                ),
                "trigger_word_suffix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional trigger word or quality suffix appended to every caption.",
                    },
                ),
                "strip_vlm_boilerplate": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Removes meta VLM openings like 'In this image, we see...', 'The image shows...', 'Here is a description:'.",
                    },
                ),
                "clean_markdown": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Strips Markdown headers, bold markers, and bullet formatting.",
                    },
                ),
                "format_mode": (
                    ["keep_original", "natural_prose", "normalized_tags"],
                    {
                        "default": "keep_original",
                        "tooltip": "keep_original: preserves prose/tags. natural_prose: capitalizes & adds period. normalized_tags: deduplicates comma-separated tags.",
                    },
                ),
                "max_token_budget": (
                    "INT",
                    {
                        "default": 225,
                        "min": 30,
                        "max": 2048,
                        "step": 5,
                        "tooltip": "Token budget threshold used to flag over-length captions in the QA report and contact sheet.",
                    },
                ),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "find_and_replace_rules": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "One rule per line using 'old -> new' or 'old = new' syntax.",
                    },
                ),
                "banned_phrases": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Comma- or newline-separated phrases to strip from all captions.",
                    },
                ),
                "default_caption_if_missing": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional fallback caption used when an image has no .txt sidecar file.",
                    },
                ),
                "output_folder_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Target directory when write_mode is 'export_to_output_folder'.",
                    },
                ),
                "max_preview_tiles": (
                    "INT",
                    {"default": 16, "min": 4, "max": 36, "step": 4},
                ),
            },
        }

    def curate_dataset(
        self,
        folder_path: str,
        write_mode: str = "dry_run_preview_only",
        trigger_word_prefix: str = "",
        trigger_word_suffix: str = "",
        strip_vlm_boilerplate: bool = True,
        clean_markdown: bool = True,
        format_mode: str = "keep_original",
        max_token_budget: int = 225,
        include_subfolders: bool = False,
        find_and_replace_rules: str = "",
        banned_phrases: str = "",
        default_caption_if_missing: str = "",
        output_folder_override: str = "",
        max_preview_tiles: int = 16,
    ) -> tuple[Any, str, str, int]:
        scan = scan_image_folder(
            folder_path=folder_path,
            include_subfolders=include_subfolders,
            skip_existing_txt=False,
            sort_method="natural_alphabetical",
        )
        image_files: list[Path] = scan["files"]

        export_dir: Path | None = None
        if write_mode == "export_to_output_folder":
            if output_folder_override and output_folder_override.strip():
                export_dir = Path(os.path.expandvars(os.path.expanduser(output_folder_override.strip()))).resolve()
            elif folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
                export_dir = (Path(folder_paths.get_output_directory()) / "ziggart_curated_dataset").resolve()
            else:
                export_dir = (scan["resolved_folder"] / "curated_captions").resolve()
            export_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        curated_lines: list[str] = []
        caption_counts: dict[str, int] = {}
        token_lengths: list[int] = []
        missing_files: list[str] = []
        over_budget_files: list[dict[str, Any]] = []
        files_written = 0

        for img_p in image_files:
            sidecar_p = img_p.with_suffix(".txt")
            has_sidecar = sidecar_p.is_file()
            raw_caption = ""
            if has_sidecar:
                try:
                    raw_caption = sidecar_p.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    raw_caption = ""

            missing_sidecar = not has_sidecar or not raw_caption
            if missing_sidecar and default_caption_if_missing.strip():
                raw_caption = default_caption_if_missing.strip()
                missing_sidecar = False

            if missing_sidecar:
                missing_files.append(img_p.name)
                records.append(
                    {
                        "image_path": str(img_p),
                        "sidecar_path": str(sidecar_p),
                        "missing_sidecar": True,
                        "curated_caption": "",
                        "estimated_tokens": 0,
                        "clip_77_chunks": 0,
                    }
                )
                continue

            curated_text, meta = curate_single_caption_text(
                raw_caption,
                trigger_word_prefix=trigger_word_prefix,
                trigger_word_suffix=trigger_word_suffix,
                strip_vlm_boilerplate=strip_vlm_boilerplate,
                clean_markdown=clean_markdown,
                find_and_replace_rules=find_and_replace_rules,
                banned_phrases=banned_phrases,
                format_mode=format_mode,
            )

            toks = int(meta["estimated_tokens"])
            token_lengths.append(toks)
            if curated_text:
                curated_lines.append(curated_text)
                norm_cap = curated_text.lower()
                caption_counts[norm_cap] = caption_counts.get(norm_cap, 0) + 1

            if toks > int(max_token_budget):
                over_budget_files.append(
                    {
                        "file": img_p.name,
                        "estimated_tokens": toks,
                        "clip_77_chunks": meta["clip_77_chunks"],
                    }
                )

            target_write_path: Path | None = None
            if write_mode == "write_in_place_atomic":
                target_write_path = sidecar_p
            elif write_mode == "export_to_output_folder" and export_dir is not None:
                target_write_path = export_dir / f"{img_p.stem}.txt"

            if target_write_path is not None and curated_text:
                tmp_path = target_write_path.with_suffix(".txt.tmp")
                tmp_path.write_text(curated_text + "\n", encoding="utf-8")
                os.replace(tmp_path, target_write_path)
                files_written += 1

            records.append(
                {
                    "image_path": str(img_p),
                    "sidecar_path": str(target_write_path or sidecar_p),
                    "missing_sidecar": False,
                    "curated_caption": curated_text,
                    "estimated_tokens": toks,
                    "clip_77_chunks": meta["clip_77_chunks"],
                    "boilerplate_stripped": meta["boilerplate_stripped"],
                    "replacements_applied": meta["replacements_applied"],
                }
            )

        duplicate_captions = [cap for cap, cnt in caption_counts.items() if cnt > 1]
        avg_tokens = round(float(np.mean(token_lengths)), 2) if token_lengths else 0.0
        min_tokens = int(min(token_lengths)) if token_lengths else 0
        max_tokens = int(max(token_lengths)) if token_lengths else 0

        summary_stats = {
            "folder": str(scan["resolved_folder"]),
            "write_mode": write_mode,
            "total_images": len(image_files),
            "captioned_count": len(token_lengths),
            "missing_txt_count": len(missing_files),
            "files_written": files_written,
            "avg_tokens": avg_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "max_token_budget": int(max_token_budget),
            "over_budget_count": len(over_budget_files),
            "duplicate_caption_count": len(duplicate_captions),
            "missing_files": missing_files,
            "over_budget_files": over_budget_files,
        }

        contact_sheet = _render_dataset_curator_contact_sheet(
            records,
            summary_stats=summary_stats,
            max_token_budget=int(max_token_budget),
            max_tiles=int(max_preview_tiles),
        )
        return (
            contact_sheet,
            "\n".join(curated_lines),
            json.dumps(summary_stats, indent=2, ensure_ascii=False),
            len(token_lengths),
        )


if (
    PromptServer is not None
    and getattr(PromptServer, "instance", None) is not None
    and web is not None
):

    @PromptServer.instance.routes.post("/ziggart/folder_queue/scan")
    async def ziggart_scan_image_folder_api(request):
        try:
            data = await request.json()
            folder_path = str(data.get("folder_path", "") or "")
            include_subfolders = bool(data.get("include_subfolders", False))
            skip_existing_txt = bool(data.get("skip_existing_txt", False))
            sort_method = str(data.get("sort_method", "natural_alphabetical") or "natural_alphabetical")
            filename_filter = str(data.get("filename_filter", "") or "")
            seed = int(data.get("seed", 42) or 42)
            image_index = int(data.get("image_index", 0) or 0)
            loop_index = bool(data.get("loop_index", False))

            scan = scan_image_folder(
                folder_path=folder_path,
                include_subfolders=include_subfolders,
                skip_existing_txt=skip_existing_txt,
                sort_method=sort_method,
                filename_filter=filename_filter,
                seed=seed,
            )
            info = _build_folder_scan_status(
                scan,
                image_index=image_index,
                loop_index=loop_index,
                executing=False,
            )
            return web.json_response(
                {
                    "ok": True,
                    "folder": str(scan["resolved_folder"]),
                    "total_found": scan["total_found"],
                    "skipped_txt_count": scan["skipped_txt_count"],
                    "queues_needed": scan["queues_needed"],
                    "active_index": info["active_index"],
                    "remaining_queues": info["remaining_queues"],
                    "remaining_to_queue": info.get("remaining_to_queue", scan["queues_needed"]),
                    "current_filename": info["current_filename"],
                    "status_text": info["status_text"],
                    "queue_summary": info["queue_summary"],
                }
            )
        except Exception as exc:
            msg = f"⚠️ {exc}"
            return web.json_response(
                {
                    "ok": False,
                    "error": str(exc),
                    "queues_needed": 0,
                    "remaining_to_queue": 0,
                    "status_text": msg,
                },
                status=200,
            )


NODE_CLASS_MAPPINGS = {
    "ZiggartGGUFVisionPrompt": GGUFVisionPrompt,
    "ZiggartGGUFVLMLoader": ZiggartGGUFVLMLoader,
    "ZiggartGGUFVisionPromptReviser": ZiggartGGUFVisionPromptReviser,
    "ZiggartPromptRevisionGate": ZiggartPromptRevisionGate,
    "ZiggartFolderImageQueueLoader": ZiggartFolderImageQueueLoader,
    "ZiggartVisionPromptTxtSaver": ZiggartVisionPromptTxtSaver,
    "ZiggartDatasetCaptionCuratorAndTriggerInjector": ZiggartDatasetCaptionCuratorAndTriggerInjector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartGGUFVisionPrompt": "GGUF Vision Prompt",
    "ZiggartGGUFVLMLoader": "GGUF VLM Loader",
    "ZiggartGGUFVisionPromptReviser": "GGUF Vision Prompt Reviser",
    "ZiggartPromptRevisionGate": "Ziggart Prompt Revision Gate",
    "ZiggartFolderImageQueueLoader": "Ziggart Folder Image Batch & Queue Loader",
    "ZiggartVisionPromptTxtSaver": "Ziggart Vision Prompt TXT & Wildcard Line Saver",
    "ZiggartDatasetCaptionCuratorAndTriggerInjector": "Ziggart Dataset Caption Curator & Trigger Injector (QA)",
}
