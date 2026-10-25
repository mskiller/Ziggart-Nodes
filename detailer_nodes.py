from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
import torch.nn.functional as F

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import comfy.model_management
    SAMPLER_NAMES = comfy.samplers.KSampler.SAMPLERS
    SCHEDULER_NAMES = comfy.samplers.KSampler.SCHEDULERS
except Exception:
    comfy = None
    SAMPLER_NAMES = [
        "euler",
        "euler_ancestral",
        "heun",
        "dpm_2",
        "dpm_2_ancestral",
        "lms",
        "dpm_fast",
        "dpm_adaptive",
        "dpmpp_2s_ancestral",
        "dpmpp_sde",
        "dpmpp_sde_gpu",
        "dpmpp_2m",
        "dpmpp_2m_sde",
        "dpmpp_2m_sde_gpu",
        "dpmpp_3m_sde",
        "dpmpp_3m_sde_gpu",
        "ddpm",
        "lcm",
        "ddim",
        "uni_pc",
        "uni_pc_bh2",
    ]
    SCHEDULER_NAMES = [
        "normal",
        "karras",
        "exponential",
        "sgm_uniform",
        "simple",
        "ddim_uniform",
        "beta",
    ]

try:
    import nodes as comfy_nodes
except ImportError:
    comfy_nodes = None

from .embedded_model import MODEL_MANAGER, EmbeddedLlamaError, ModelConfig, release_comfy_vram
from .nodes import _gguf_files, _resolve_gguf, image_to_data_uri, NO_MODEL, NO_MMPROJ


def parse_bounding_boxes(text: str, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """
    Extracts bounding boxes (x0, y0, x1, y1) in image pixel coordinates from VLM response text.
    Handles multiple coordinate representations:
    1. Qwen-VL style: "box_2d": [ymin, xmin, ymax, xmax]
    2. WeVisOne-VL discrete token syntax: <x0><y0><x1><y1>
    3. Bracketed sequences: [x0, y0, x1, y1] normalized to 1000, 1.0, or raw pixels.
    """
    boxes: list[tuple[int, int, int, int]] = []

    # 1. Look for explicit JSON-like box_2d: [ymin, xmin, ymax, xmax]
    box_2d_matches = re.findall(
        r'box_2d[\"\'\s]*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]',
        text,
        re.IGNORECASE,
    )
    if box_2d_matches:
        for y0_s, x0_s, y1_s, x1_s in box_2d_matches:
            y0, x0, y1, x1 = float(y0_s), float(x0_s), float(y1_s), float(x1_s)
            scale_x = img_w if max(x0, x1) <= 1.0 else (img_w / 1000.0 if max(x0, x1) <= 1000.0 else 1.0)
            scale_y = img_h if max(y0, y1) <= 1.0 else (img_h / 1000.0 if max(y0, y1) <= 1000.0 else 1.0)
            px0 = int(round(min(x0, x1) * scale_x))
            py0 = int(round(min(y0, y1) * scale_y))
            px1 = int(round(max(x0, x1) * scale_x))
            py1 = int(round(max(y0, y1) * scale_y))
            if px1 > px0 and py1 > py0:
                boxes.append((
                    max(0, min(img_w, px0)),
                    max(0, min(img_h, py0)),
                    max(0, min(img_w, px1)),
                    max(0, min(img_h, py1)),
                ))
        if boxes:
            return boxes

    # 2. WeVisOne discrete coordinate tokens: <x0><y0><x1><y1>
    token_matches = re.findall(r'<(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})>', text)
    if token_matches:
        for x0_s, y0_s, x1_s, y1_s in token_matches:
            x0, y0, x1, y1 = float(x0_s), float(y0_s), float(x1_s), float(y1_s)
            px0 = int(round(min(x0, x1) * img_w / 1000.0))
            py0 = int(round(min(y0, y1) * img_h / 1000.0))
            px1 = int(round(max(x0, x1) * img_w / 1000.0))
            py1 = int(round(max(y0, y1) * img_h / 1000.0))
            if px1 > px0 and py1 > py0:
                boxes.append((
                    max(0, min(img_w, px0)),
                    max(0, min(img_h, py0)),
                    max(0, min(img_w, px1)),
                    max(0, min(img_h, py1)),
                ))
        if boxes:
            return boxes

    # 3. Standard bracketed sequence: [x0, y0, x1, y1] or (x0, y0, x1, y1)
    bracket_matches = re.findall(
        r'[\[\(]\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*[\]\)]',
        text,
    )
    for a_s, b_s, c_s, d_s in bracket_matches:
        a, b, c, d = float(a_s), float(b_s), float(c_s), float(d_s)
        max_val = max(a, b, c, d)
        if max_val <= 1.0:
            scale_x, scale_y = img_w, img_h
        elif max_val <= 1000.0:
            scale_x, scale_y = img_w / 1000.0, img_h / 1000.0
        else:
            scale_x, scale_y = 1.0, 1.0

        px0 = int(round(min(a, c) * scale_x))
        py0 = int(round(min(b, d) * scale_y))
        px1 = int(round(max(a, c) * scale_x))
        py1 = int(round(max(b, d) * scale_y))

        if px1 > px0 and py1 > py0:
            boxes.append((
                max(0, min(img_w, px0)),
                max(0, min(img_h, py0)),
                max(0, min(img_w, px1)),
                max(0, min(img_h, py1)),
            ))

    return boxes


def crop_with_padding(
    box: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    crop_factor: float = 1.5,
) -> tuple[int, int, int, int]:
    """
    Expands a bounding box by crop_factor around its center,
    clamps to image boundaries, and aligns dimensions to multiples of 8.
    """
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    nw = bw * max(1.0, crop_factor)
    nh = bh * max(1.0, crop_factor)

    # Compute raw expanded bounds
    px0 = max(0, int(round(cx - nw / 2.0)))
    py0 = max(0, int(round(cy - nh / 2.0)))
    px1 = min(img_w, int(round(cx + nw / 2.0)))
    py1 = min(img_h, int(round(cy + nh / 2.0)))

    # Ensure width and height are divisible by 8 (required for VAE / DiTs)
    w = px1 - px0
    h = py1 - py0
    rem_w = w % 8
    rem_h = h % 8

    if rem_w != 0:
        if px1 + (8 - rem_w) <= img_w:
            px1 += 8 - rem_w
        elif px0 - (8 - rem_w) >= 0:
            px0 -= 8 - rem_w
        else:
            px1 -= rem_w

    if rem_h != 0:
        if py1 + (8 - rem_h) <= img_h:
            py1 += 8 - rem_h
        elif py0 - (8 - rem_h) >= 0:
            py0 -= 8 - rem_h
        else:
            py1 -= rem_h

    px0 = max(0, min(img_w - 8, px0))
    py0 = max(0, min(img_h - 8, py0))
    px1 = max(px0 + 8, min(img_w, px1))
    py1 = max(py0 + 8, min(img_h, py1))

    return px0, py0, px1, py1


def create_feathered_mask(
    crop_w: int,
    crop_h: int,
    feather_radius: int = 16,
) -> torch.Tensor:
    """
    Creates a 2D float tensor [crop_h, crop_w] with smooth alpha falloff near the boundaries
    to ensure seamless inpainting blends.
    """
    if feather_radius <= 0 or crop_w <= feather_radius * 2 or crop_h <= feather_radius * 2:
        return torch.ones((crop_h, crop_w), dtype=torch.float32)

    mask_img = Image.new("L", (crop_w, crop_h), 0)
    draw = ImageDraw.Draw(mask_img)
    pad = max(1, feather_radius)
    draw.rectangle([pad, pad, crop_w - pad - 1, crop_h - pad - 1], fill=255)

    blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    arr = np.array(blurred, dtype=np.float32) / 255.0
    max_val = arr.max()
    if max_val > 0.0:
        arr = arr / max_val
    return torch.from_numpy(arr)


def standardize_and_batch_crops(crop_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Safely batches a list of image tensors [1, H, W, C] or [H, W, C] along dimension 0.
    If crops have differing spatial resolutions (H, W), resizes them to the maximum
    dimensions using bicubic interpolation so they form a valid ComfyUI IMAGE batch.
    """
    if not crop_list:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    normalized = [c if c.ndim == 4 else c.unsqueeze(0) for c in crop_list]

    first_shape = normalized[0].shape[1:3]
    if all(c.shape[1:3] == first_shape for c in normalized):
        return torch.cat(normalized, dim=0)

    max_h = max(c.shape[1] for c in normalized)
    max_w = max(c.shape[2] for c in normalized)

    standardized = []
    for c in normalized:
        if c.shape[1] == max_h and c.shape[2] == max_w:
            standardized.append(c)
        else:
            c_perm = c.permute(0, 3, 1, 2)
            c_res = F.interpolate(
                c_perm,
                size=(max_h, max_w),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
            standardized.append(c_res.permute(0, 2, 3, 1))

    return torch.cat(standardized, dim=0)


def draw_boxes_on_image(
    image_tensor: torch.Tensor,
    boxes: list[tuple[int, int, int, int]],
    label: str = "detected",
) -> torch.Tensor:
    """
    Draws bounding boxes and text tags on an image tensor [H, W, 3] or [1, H, W, 3].
    Returns [1, H, W, 3] float tensor in range [0, 1].
    """
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]

    np_img = (np.clip(image_tensor.detach().cpu().float().numpy(), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil_img = Image.fromarray(np_img, mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    box_colors = [
        (0, 229, 255),   # Bright cyan
        (0, 255, 128),   # Spring green
        (255, 64, 128),  # Coral pink
        (255, 215, 0),   # Gold
        (180, 100, 255), # Neon purple
    ]

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        color = box_colors[i % len(box_colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        tag = f"{label} #{i+1} [{x1-x0}x{y1-y0}]"
        bbox = font.getbbox(tag)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tag_y0 = max(0, y0 - th - 6)
        tag_y1 = tag_y0 + th + 6
        draw.rectangle([x0, tag_y0, x0 + tw + 8, tag_y1], fill=color)
        draw.text((x0 + 4, tag_y0 + 2), tag, fill=(0, 0, 0), font=font)

    res_np = np.array(pil_img, dtype=np.float32) / 255.0
    return torch.from_numpy(res_np).unsqueeze(0)


def detect_objects_with_vlm(
    frame_image: Any,
    model_path: str,
    mmproj_path: str,
    detection_prompt: str,
    max_detections: int,
    gpu_layers: int,
    context_size: int,
    unload_detector_before_sampling: bool,
) -> tuple[list[tuple[int, int, int, int]], str]:
    """
    Runs WeVisOne-VL GGUF in-process to detect objects specified by detection_prompt.
    """
    config = ModelConfig(
        model_path=model_path,
        mmproj_path=mmproj_path,
        context_size=context_size,
        gpu_layers=gpu_layers,
        threads=os.cpu_count() or 4,
        flash_attention=True,
        mmproj_on_gpu=True,
    )
    MODEL_MANAGER.load(config, unload_comfy_models=True)

    system_instruction = (
        "You are an expert visual grounding AI. Locate the requested objects precisely in the image. "
        "Return bounding box coordinates [x0, y0, x1, y1] normalized to 1000."
    )
    user_instruction = (
        f"Detect and locate: '{detection_prompt}'.\n"
        "Return coordinates for all occurrences in the format [x0, y0, x1, y1] normalized from 0 to 1000."
    )

    data_uri = image_to_data_uri(frame_image, max_dimension=1024)
    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instruction},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    try:
        raw_response = MODEL_MANAGER.chat(
            messages,
            max_tokens=256,
            temperature=0.1,
            top_p=0.9,
            top_k=20,
            seed=42,
            enable_thinking=False,
        )
    finally:
        if unload_detector_before_sampling:
            MODEL_MANAGER.unload()
            release_comfy_vram()

    if hasattr(frame_image, "shape"):
        h, w = frame_image.shape[0], frame_image.shape[1]
    else:
        h, w = 1024, 1024

    boxes = parse_bounding_boxes(raw_response, w, h)
    if max_detections > 0:
        boxes = boxes[:max_detections]

    return boxes, raw_response


class ZiggartDetailerKrea2:
    """
    ComfyUI Detailer node using WeVisOne-VL for prompt-driven zone detection
    and Krea 2 Turbo for fast, high-quality resampling and refinement.
    """

    DESCRIPTION = (
        "Detects regions of interest with WeVisOne-VL (eyes, faces, objects, etc.) "
        "and resamples/refines them with Krea 2 Turbo using dedicated prompts and feathered blending."
    )
    CATEGORY = "Ziggart/Detailer"
    FUNCTION = "refine"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "LATENT")
    RETURN_NAMES = (
        "image",
        "cropped_refined",
        "cropped_base",
        "detection_preview",
        "mask",
        "latent",
    )
    OUTPUT_TOOLTIPS = (
        "The final full composited image with refined regions seamlessly blended.",
        "High-resolution refined crops.",
        "Original base crops before refinement.",
        "Visual preview image showing detected bounding boxes and labels.",
        "Full composite mask corresponding to the modified regions.",
        "Latent representation of the final image.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = _gguf_files()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image or batch to refine."}),
                "model": ("MODEL", {"tooltip": "Krea 2 Turbo diffusion model."}),
                "clip": ("CLIP", {"tooltip": "CLIP text encoder."}),
                "vae": ("VAE", {"tooltip": "VAE encoder/decoder."}),
                "detector_model": (models, {"tooltip": "WeVisOne-VL GGUF model from models/LLM/GGUF."}),
                "detector_mmproj": (projectors, {"tooltip": "Matching mmproj GGUF from models/LLM/GGUF."}),
                "detection_prompt": (
                    "STRING",
                    {
                        "default": "eyes",
                        "multiline": False,
                        "tooltip": "Prompt for WeVisOne-VL to locate (e.g., 'eyes', 'face', 'jewelry').",
                    },
                ),
                "detail_positive_prompt": (
                    "STRING",
                    {
                        "default": "ultra-detailed sharp eyes, realistic iris reflections, crystal clear, 8k",
                        "multiline": True,
                        "tooltip": "Positive prompt applied specifically when resampling the detected zone.",
                    },
                ),
                "detail_negative_prompt": (
                    "STRING",
                    {
                        "default": "blurry, noisy, distorted, low quality, artifacts",
                        "multiline": True,
                        "tooltip": "Negative prompt applied when resampling the detected zone.",
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Padding multiplier around the detected box to give context to Krea2.",
                    },
                ),
                "target_detail_resolution": (
                    "INT",
                    {
                        "default": 768,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Resolution to which crops are scaled before sampling (768 recommended for Krea2).",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Sampling steps (8 is optimal for Krea 2 Turbo).",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance scale (1.5 recommended for Krea 2 Turbo).",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 0.40,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Denoise strength (0.3-0.5 for detail refinement, higher for strong redraw).",
                    },
                ),
                "sampler_name": (SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (SCHEDULER_NAMES, {"default": "simple"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_detections": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 20,
                        "tooltip": "Maximum number of detected regions to process per image.",
                    },
                ),
                "feather_mask": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 128,
                        "tooltip": "Feather blur radius in pixels for seamless border blending.",
                    },
                ),
                "free_vram_after": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Release detector VRAM immediately before running Krea 2 Turbo sampling.",
                    },
                ),
            },
            "optional": {
                "detector_model_override": (
                    "STRING",
                    {"default": "", "tooltip": "Optional absolute path to WeVisOne-VL GGUF."},
                ),
                "detector_mmproj_override": (
                    "STRING",
                    {"default": "", "tooltip": "Optional absolute path to mmproj GGUF."},
                ),
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 256, "tooltip": "Detector GPU layers (-1 for all)."},
                ),
                "context_size": (
                    "INT",
                    {"default": 4096, "min": 1024, "max": 32768, "tooltip": "Detector context size."},
                ),
            },
        }

    def _execute_sampling(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        crop_tensor: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        seed: int,
    ) -> torch.Tensor:
        """Encodes crop with VAE, runs KSampler, and decodes back to image tensor."""
        if comfy_nodes is not None:
            # Standard ComfyUI node pipeline
            clip_encoder = comfy_nodes.CLIPTextEncode()
            pos_cond = clip_encoder.encode(clip=clip, text=positive_prompt)[0]
            neg_cond = clip_encoder.encode(clip=clip, text=negative_prompt)[0]

            vae_encoder = comfy_nodes.VAEEncode()
            latent_crop = vae_encoder.encode(vae=vae, pixels=crop_tensor)[0]

            ksampler = comfy_nodes.KSampler()
            refined_latent = ksampler.sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=pos_cond,
                negative=neg_cond,
                latent_image=latent_crop,
                denoise=denoise,
            )[0]

            vae_decoder = comfy_nodes.VAEDecode()
            refined_crop = vae_decoder.decode(vae=vae, samples=refined_latent)[0]
            return refined_crop
        else:
            # Fallback for mock/test environments
            tokens_pos = clip.tokenize(positive_prompt)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
            pos_cond = [[cond_pos, {"pooled_output": pooled_pos}]]

            tokens_neg = clip.tokenize(negative_prompt)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            neg_cond = [[cond_neg, {"pooled_output": pooled_neg}]]

            latent_crop = {"samples": vae.encode(crop_tensor[:, :, :, :3])}
            if comfy is not None and hasattr(comfy, "sample") and hasattr(comfy.sample, "sample"):
                samples = comfy.sample.sample(
                    model,
                    torch.randn_like(latent_crop["samples"]),
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    pos_cond,
                    neg_cond,
                    latent_crop["samples"],
                    denoise=denoise,
                    seed=seed,
                )
                refined_latent = {"samples": samples}
            else:
                refined_latent = latent_crop

            return vae.decode(refined_latent["samples"])

    def refine(
        self,
        image: torch.Tensor,
        model: Any,
        clip: Any,
        vae: Any,
        detector_model: str,
        detector_mmproj: str,
        detection_prompt: str,
        detail_positive_prompt: str,
        detail_negative_prompt: str,
        crop_factor: float,
        target_detail_resolution: int,
        steps: int,
        cfg: float,
        denoise: float,
        sampler_name: str,
        scheduler: str,
        seed: int,
        max_detections: int,
        feather_mask: int,
        free_vram_after: bool,
        detector_model_override: str = "",
        detector_mmproj_override: str = "",
        gpu_layers: int = -1,
        context_size: int = 4096,
    ):
        model_path = _resolve_gguf(detector_model, detector_model_override, "detector_model")
        mmproj_path = _resolve_gguf(detector_mmproj, detector_mmproj_override, "detector_mmproj")

        batch_size, img_h, img_w, channels = image.shape
        final_images: list[torch.Tensor] = []
        detection_previews: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        cropped_base_list: list[torch.Tensor] = []
        cropped_refined_list: list[torch.Tensor] = []

        for b in range(batch_size):
            frame = image[b : b + 1]
            out_frame = frame.clone()
            full_mask = torch.zeros((img_h, img_w), dtype=torch.float32)

            # Step 1: Detect bounding boxes with WeVisOne-VL
            boxes, raw_text = detect_objects_with_vlm(
                frame_image=frame[0],
                model_path=model_path,
                mmproj_path=mmproj_path,
                detection_prompt=detection_prompt,
                max_detections=max_detections,
                gpu_layers=gpu_layers,
                context_size=context_size,
                unload_detector_before_sampling=free_vram_after,
            )

            preview = draw_boxes_on_image(frame[0], boxes, label=detection_prompt)
            detection_previews.append(preview)

            if not boxes:
                final_images.append(out_frame)
                all_masks.append(full_mask.unsqueeze(0))
                continue

            for idx, box in enumerate(boxes):
                # Step 2: Compute padded crop
                cx0, cy0, cx1, cy1 = crop_with_padding(box, img_w, img_h, crop_factor=crop_factor)
                crop_orig = out_frame[:, cy0:cy1, cx0:cx1, :]
                ch, cw = cy1 - cy0, cx1 - cx0

                # Step 3: Resize crop to target detail resolution
                scale = target_detail_resolution / float(max(ch, cw))
                target_w = max(64, int(round(cw * scale / 8.0)) * 8)
                target_h = max(64, int(round(ch * scale / 8.0)) * 8)

                # Interpolate to high-res
                crop_perm = crop_orig.permute(0, 3, 1, 2)
                crop_high = F.interpolate(crop_perm, size=(target_h, target_w), mode="bicubic", align_corners=False)
                crop_high = crop_high.permute(0, 2, 3, 1).clamp(0.0, 1.0)

                cropped_base_list.append(crop_high.clone())

                # Step 4: Resample using Krea 2 Turbo
                cur_seed = (seed + idx * 31) & 0xFFFFFFFFFFFFFFFF
                refined_high = self._execute_sampling(
                    model=model,
                    clip=clip,
                    vae=vae,
                    crop_tensor=crop_high,
                    positive_prompt=detail_positive_prompt,
                    negative_prompt=detail_negative_prompt,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise,
                    seed=cur_seed,
                )
                cropped_refined_list.append(refined_high.clone())

                # Step 5: Resize refined crop back to original crop size
                refined_perm = refined_high.permute(0, 3, 1, 2)
                refined_orig = F.interpolate(refined_perm, size=(ch, cw), mode="bicubic", align_corners=False)
                refined_orig = refined_orig.permute(0, 2, 3, 1).clamp(0.0, 1.0)

                # Step 6: Create feathered blend mask
                feather_m = create_feathered_mask(cw, ch, feather_radius=feather_mask).to(out_frame.device)
                feather_4d = feather_m.unsqueeze(0).unsqueeze(-1)  # [1, ch, cw, 1]

                # Composite back into image
                current_patch = out_frame[:, cy0:cy1, cx0:cx1, :]
                blended_patch = refined_orig * feather_4d + current_patch * (1.0 - feather_4d)
                out_frame[:, cy0:cy1, cx0:cx1, :] = blended_patch

                # Update composite mask
                full_mask[cy0:cy1, cx0:cx1] = torch.maximum(full_mask[cy0:cy1, cx0:cx1], feather_m.cpu())

            final_images.append(out_frame)
            all_masks.append(full_mask.unsqueeze(0))

        final_image_tensor = torch.cat(final_images, dim=0)
        final_preview_tensor = torch.cat(detection_previews, dim=0)
        final_mask_tensor = torch.cat(all_masks, dim=0)

        # Safely batch crops (resizing to matching dimensions if detections had different aspect ratios)
        final_refined_crops = standardize_and_batch_crops(cropped_refined_list)
        final_base_crops = standardize_and_batch_crops(cropped_base_list)

        # Encode final composited image to latent
        try:
            if comfy_nodes is not None:
                vae_encoder = comfy_nodes.VAEEncode()
                final_latent = vae_encoder.encode(vae=vae, pixels=final_image_tensor)[0]
            else:
                final_latent = {"samples": vae.encode(final_image_tensor[:, :, :, :3])}
        except Exception:
            final_latent = {"samples": torch.zeros((batch_size, 4, img_h // 8, img_w // 8))}

        return (
            final_image_tensor,
            final_refined_crops,
            final_base_crops,
            final_preview_tensor,
            final_mask_tensor,
            final_latent,
        )


class ZiggartWeVisOneDetector:
    """
    Dedicated detection node using WeVisOne-VL for prompt-guided object detection
    returning preview overlays, masks, cropped patches, and JSON coordinates.
    """

    DESCRIPTION = (
        "Locates objects using WeVisOne-VL open-vocabulary detection and generates "
        "bounding-box overlays, segment masks, cropped image batches, and JSON coordinates."
    )
    CATEGORY = "Ziggart/Detector"
    FUNCTION = "detect"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("detection_preview", "mask", "cropped_images", "bboxes_json")
    OUTPUT_TOOLTIPS = (
        "Preview image showing detected bounding boxes and tags.",
        "Full-size mask containing the detected bounding boxes.",
        "Batch of cropped detected areas with padding.",
        "JSON array containing detected bounding box coordinates [x0, y0, x1, y1].",
    )

    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = _gguf_files()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to detect objects in."}),
                "detector_model": (models, {"tooltip": "WeVisOne-VL GGUF model."}),
                "detector_mmproj": (projectors, {"tooltip": "Matching mmproj GGUF."}),
                "detection_prompt": (
                    "STRING",
                    {
                        "default": "eyes",
                        "multiline": False,
                        "tooltip": "Prompt describing the objects to locate.",
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Padding multiplier around the detected box.",
                    },
                ),
                "max_detections": (
                    "INT",
                    {"default": 4, "min": 1, "max": 20, "tooltip": "Maximum detections to keep."},
                ),
                "feather_mask": (
                    "INT",
                    {"default": 0, "min": 0, "max": 128, "tooltip": "Feather blur radius for the mask."},
                ),
                "free_vram_after": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Unload WeVisOne-VL after detection."},
                ),
            },
            "optional": {
                "detector_model_override": (
                    "STRING",
                    {"default": "", "tooltip": "Optional absolute path to WeVisOne-VL GGUF."},
                ),
                "detector_mmproj_override": (
                    "STRING",
                    {"default": "", "tooltip": "Optional absolute path to mmproj GGUF."},
                ),
                "gpu_layers": (
                    "INT",
                    {"default": -1, "min": -1, "max": 256, "tooltip": "Detector GPU layers (-1 for all)."},
                ),
                "context_size": (
                    "INT",
                    {"default": 4096, "min": 1024, "max": 32768, "tooltip": "Detector context size."},
                ),
            },
        }

    def detect(
        self,
        image: torch.Tensor,
        detector_model: str,
        detector_mmproj: str,
        detection_prompt: str,
        crop_factor: float,
        max_detections: int,
        feather_mask: int,
        free_vram_after: bool,
        detector_model_override: str = "",
        detector_mmproj_override: str = "",
        gpu_layers: int = -1,
        context_size: int = 4096,
    ):
        model_path = _resolve_gguf(detector_model, detector_model_override, "detector_model")
        mmproj_path = _resolve_gguf(detector_mmproj, detector_mmproj_override, "detector_mmproj")

        batch_size, img_h, img_w, _ = image.shape
        previews: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        all_crops: list[torch.Tensor] = []
        all_bboxes_data: list[dict[str, Any]] = []

        for b in range(batch_size):
            frame = image[b]
            boxes, raw_text = detect_objects_with_vlm(
                frame_image=frame,
                model_path=model_path,
                mmproj_path=mmproj_path,
                detection_prompt=detection_prompt,
                max_detections=max_detections,
                gpu_layers=gpu_layers,
                context_size=context_size,
                unload_detector_before_sampling=free_vram_after,
            )

            preview = draw_boxes_on_image(frame, boxes, label=detection_prompt)
            previews.append(preview)

            full_mask = torch.zeros((img_h, img_w), dtype=torch.float32)
            frame_bboxes = []

            for i, box in enumerate(boxes):
                cx0, cy0, cx1, cy1 = crop_with_padding(box, img_w, img_h, crop_factor=crop_factor)
                crop_patch = frame[cy0:cy1, cx0:cx1, :].unsqueeze(0)
                all_crops.append(crop_patch)

                # Fill mask
                if feather_mask > 0:
                    fm = create_feathered_mask(cx1 - cx0, cy1 - cy0, feather_radius=feather_mask)
                    full_mask[cy0:cy1, cx0:cx1] = torch.maximum(full_mask[cy0:cy1, cx0:cx1], fm)
                else:
                    full_mask[box[1]:box[3], box[0]:box[2]] = 1.0

                frame_bboxes.append({
                    "id": i + 1,
                    "box": list(box),
                    "padded_crop": [cx0, cy0, cx1, cy1],
                    "label": detection_prompt,
                })

            masks.append(full_mask.unsqueeze(0))
            all_bboxes_data.append({"frame_index": b, "detections": frame_bboxes})

        preview_tensor = torch.cat(previews, dim=0)
        mask_tensor = torch.cat(masks, dim=0)
        crop_tensor = standardize_and_batch_crops(all_crops)

        return (
            preview_tensor,
            mask_tensor,
            crop_tensor,
            json.dumps(all_bboxes_data, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "ZiggartDetailerKrea2": ZiggartDetailerKrea2,
    "ZiggartWeVisOneDetector": ZiggartWeVisOneDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartDetailerKrea2": "Detailers Krea2",
    "ZiggartWeVisOneDetector": "WeVisOne-VL Detector",
}
