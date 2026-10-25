# Ziggart Nodes for ComfyUI

A complete collection of local, in-process ComfyUI nodes combining vision-language models (VLMs via GGUF) and diffusion models (**Krea 2 Turbo**, Flux, DiTs, SDXL).

### Included Nodes (10 Nodes)

| Display Name | Internal Class | Category | Purpose |
| :--- | :--- | :--- | :--- |
| **GGUF Vision Prompt** | `ZiggartGGUFVisionPrompt` | `Ziggart/LLM` | Multimodal prompt generation, image analysis, and specialist passes (Composition, Subject, Style). |
| **GGUF VLM Loader** | `ZiggartGGUFVLMLoader` | `Ziggart/LLM` | Reusable `GGUF_VL_MODEL` configuration node that can be shared across all Ziggart VLM & Detailer nodes. |
| **GGUF Quality & Anatomy Critic** | `ZiggartVLMImageCritic` | `Ziggart/LLM` | Scans generated images for defects (hands, eyes, anatomy, text, artifacts), outputting defect boxes, masks, repair prompts, and a pass/fail boolean gate. |
| **GGUF Mask-Aware Inpaint Prompter** | `ZiggartVLMInpaintPrompter` | `Ziggart/LLM` | Highlights a user `MASK` on the scene and uses the VLM to generate a lighting- and style-harmonized inpaint prompt + crop coordinates. |
| **Detailers Krea2** | `ZiggartDetailerKrea2` | `Ziggart/Detailer` | Open-vocabulary object detection (**WeVisOne-VL**) paired with high-resolution **Krea 2 Turbo** resampling, latent noise masking, and Gaussian feathered blending. |
| **Detailers Krea2 (VLM Auto-Prompt)** | `ZiggartAutoPromptDetailer` | `Ziggart/Detailer` | Detects target zones, uses the VLM to write a custom micro-prompt for each individual crop while in VRAM, unloads once, and refines every crop with Krea 2 Turbo. |
| **Krea2 VLM Smart Tile Upscaler** | `ZiggartSmartTileUpscaler` | `Ziggart/Detailer` | Overlapping tiled upscaler (1.0×–4.0×) with optional per-tile VLM captioning (preventing background hallucinations) or fast pure-Krea2 tiled upscaling. |
| **WeVisOne-VL Detector** | `ZiggartWeVisOneDetector` | `Ziggart/Detector` | Standalone open-vocabulary grounding node outputting bounding-box previews, masks, cropped patches, and `bboxes_json`. |
| **BBox Crop** | `ZiggartCropByBBox` | `Ziggart/Detector` | Extracts mod-8 padded, resolution-standardized crops from `bboxes_json` and outputs a `ZIGGART_STITCH_PIPE` for custom ComfyUI workflows. |
| **BBox Feathered Stitch** | `ZiggartStitchByBBox` | `Ziggart/Detector` | Blends externally processed crops back into their original locations using `ZIGGART_STITCH_PIPE` and Gaussian alpha feathering. |

---

## Table of Contents

1. [Key Features](#key-features)
2. [Nodes Reference](#nodes-reference)
   - [1. Detailers Krea2](#1-detailers-krea2)
   - [2. Detailers Krea2 (VLM Auto-Prompt)](#2-detailers-krea2-vlm-auto-prompt)
   - [3. Krea2 VLM Smart Tile Upscaler](#3-krea2-vlm-smart-tile-upscaler)
   - [4. WeVisOne-VL Detector](#4-wevisone-vl-detector)
   - [5. BBox Crop & BBox Feathered Stitch](#5-bbox-crop--bbox-feathered-stitch)
   - [6. GGUF Vision Prompt & GGUF VLM Loader](#6-gguf-vision-prompt--gguf-vlm-loader)
   - [7. GGUF Quality & Anatomy Critic](#7-gguf-quality--anatomy-critic)
   - [8. GGUF Mask-Aware Inpaint Prompter](#8-gguf-mask-aware-inpaint-prompter)
3. [Architecture & Workflows](#architecture--workflows)
4. [Installation](#installation)
5. [Models Download & Setup](#models-download--setup)
6. [Hardware & VRAM Optimization](#hardware--vram-optimization)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## Key Features

- **Zero External Servers**: Runs directly inside the ComfyUI Python process via `llama-cpp-python`. No Ollama, Docker, or standalone `llama-server` needed.
- **Two-Phase Batch Execution**: Across multi-image batches and multi-crop passes, Ziggart runs all VLM grounding/captioning operations first, unloads the VLM from VRAM **once**, caches static CLIP conditioning, and then runs all diffusion sampling passes—eliminating model-swap thrashing.
- **Latent Noise Masking + Pixel Feathering**: Attaches feathered noise masks directly to crop latents during KSampler steps and blends borders with Gaussian alpha feathering, eliminating seams and color shifts even at higher denoise values.
- **Natural Language Grounding**: Detect any feature (`"eyes"`, `"face"`, `"hands"`, `"jewelry"`, `"sneakers"`, `"watch"`) with plain text prompts using Tencent's WeVisOne-VL or Qwen-VL models.
- **Smart `<think>` Tag Handling**: Automatically strips `<think>...</think>` reasoning blocks from thinking VLMs so CLIP prompts and bounding-box parsers stay clean.

---

## Nodes Reference

### 1. Detailers Krea2

*Category: `Ziggart/Detailer`*

Prompt-guided open-vocabulary localization with **WeVisOne-VL** and high-density resampling with **Krea 2 Turbo**.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Input RGB/RGBA image or batch of images. |
| `model` / `clip` / `vae` | `MODEL` / `CLIP` / `VAE` | Diffusion model (Krea 2 Turbo), text encoder, and VAE. |
| `detector_model` / `detector_mmproj` | Dropdown | WeVisOne-VL GGUF model and vision projector from `models/LLM/GGUF/`. |
| `detection_prompt` | `STRING` | Natural language target to detect (e.g. `eyes`, `face`, `hands`). |
| `detail_positive_prompt` | `STRING` | Positive prompt applied when resampling detected crops. |
| `detail_negative_prompt` | `STRING` | Negative prompt for crop resampling. |
| `crop_factor` | `FLOAT` (1.0 - 4.0) | Context padding multiplier around the detected box (default: `1.5`). |
| `target_detail_resolution` | `INT` (256 - 2048) | Upscaling dimension before diffusion sampling (default: `768`). |
| `steps` / `cfg` / `denoise` | `INT` / `FLOAT` / `FLOAT` | Sampling parameters (defaults: `8` steps, `1.5` CFG, `0.40` denoise). |
| `sampler_name` / `scheduler` | Dropdown | Sampler and scheduler (defaults: `euler`, `simple`). |
| `seed` | `INT` | Random seed for sampling. |
| `max_detections` | `INT` (1 - 20) | Maximum number of detected regions to refine per image (default: `2`). |
| `feather_mask` | `INT` (0 - 128) | Gaussian blur radius in pixels for seamless border blending (default: `16`). |
| `free_vram_after` | `BOOLEAN` | Unloads detector VLM from VRAM before Krea 2 Turbo sampling (default: `True`). |
| `use_latent_mask` *(optional)* | `BOOLEAN` | Attaches feathered `noise_mask` to the crop latent during sampling (default: `True`). |
| `skip_upscale_if_larger` *(optional)* | `BOOLEAN` | Preserves native crop resolution if already larger than `target_detail_resolution` (default: `True`). |
| `encode_final_latent` *(optional)* | `BOOLEAN` | Encodes the final composited image into the `latent` output pin (default: `True`). |
| `vlm_config` *(optional)* | `GGUF_VL_MODEL` | Shared model config from **GGUF VLM Loader**. |

#### Outputs

- `image` (`IMAGE`): Full composited image with refined regions seamlessly blended.
- `cropped_refined` (`IMAGE`): Batch of high-resolution refined crops.
- `cropped_base` (`IMAGE`): Batch of original crops before refinement.
- `detection_preview` (`IMAGE`): Preview image with colored bounding boxes and dimensions.
- `mask` (`MASK`): Full-size composite mask of modified areas.
- `latent` (`LATENT`): Latent representation of the final image.

---

### 2. Detailers Krea2 (VLM Auto-Prompt)

*Category: `Ziggart/Detailer`*

Solves multi-instance detailing where a single static prompt does not fit every detected object. While the VLM is loaded for detection, it inspects each padded crop individually and writes a custom descriptive micro-prompt before unloading once and handing off to Krea 2 Turbo.

#### Key Additional Inputs & Outputs

- `prompt_instruction` (`STRING`): Instructions telling the VLM how to describe each cropped region.
- `base_positive_suffix` (`STRING`): Quality tags appended to every VLM-generated crop prompt (default: `"ultra-detailed, sharp focus, high resolution, 8k"`).
- **Output `generated_prompts` (`STRING`)**: JSON log showing the exact prompt generated for each `frame_index` and `detection_id`.

---

### 3. Krea2 VLM Smart Tile Upscaler

*Category: `Ziggart/Detailer`*

Upsamples an image by `upscale_by` (1.0×–4.0×), divides it into overlapping mod-8 tiles, refines each tile with Krea 2 Turbo, and blends them onto a weighted feather canvas.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Source image or batch to upscale. |
| `model` / `clip` / `vae` | `MODEL` / `CLIP` / `VAE` | Krea 2 Turbo diffusion pipeline. |
| `upscale_by` | `FLOAT` (1.0 - 4.0) | Upscale multiplier (default: `2.0`). |
| `tile_size` / `tile_overlap` | `INT` / `INT` | Tile dimensions (default: `768`) and overlap stride in pixels (default: `128`). |
| `enable_vlm_tile_prompts` | `BOOLEAN` | When `True` (default), uses the GGUF VLM to describe each tile individually so background tiles do not hallucinate subjects. Set to `False` for fast pure-Krea2 tiled upscaling without loading a VLM. |
| `base_positive_prompt` / `negative_prompt` | `STRING` | Global quality suffix and negative prompt. |
| `steps` / `cfg` / `denoise` | `INT` / `FLOAT` / `FLOAT` | Defaults: `8` steps, `1.5` CFG, `0.35` denoise. |
| `vlm_model` / `vlm_mmproj` / `vlm_config` *(optional)* | Dropdown / `GGUF_VL_MODEL` | VLM used when `enable_vlm_tile_prompts` is enabled. |

#### Outputs

- `upscaled_image` (`IMAGE`): Seamlessly blended high-resolution image.
- `tile_batch` (`IMAGE`): Batch of individual refined tiles.
- `tile_prompts` (`STRING`): JSON log of tile coordinates and prompts.

---

### 4. WeVisOne-VL Detector

*Category: `Ziggart/Detector`*

Standalone visual grounding node using WeVisOne-VL. Extracts bounding boxes, segmentation masks, cropped patches, and JSON coordinates without running diffusion.

- **Inputs**: `image`, `detector_model`, `detector_mmproj`, `detection_prompt`, `crop_factor`, `max_detections`, `feather_mask`, `free_vram_after`, and optional `vlm_config`.
- **Outputs**: `detection_preview` (`IMAGE`), `mask` (`MASK`), `cropped_images` (`IMAGE`), `bboxes_json` (`STRING`).

---

### 5. BBox Crop & BBox Feathered Stitch

*Category: `Ziggart/Detector`*

Decouples Ziggart's mod-8 padded cropping and Gaussian feathered compositing so you can route detected regions through **any** external ComfyUI nodes (e.g. ControlNet, IPAdapter, Flux Redux, custom KSamplers, face swappers).

1. **BBox Crop (`ZiggartCropByBBox`)**:
   - **Inputs**: `image`, `bboxes_json` (from **WeVisOne-VL Detector**, **GGUF Quality & Anatomy Critic**, or **GGUF Mask-Aware Inpaint Prompter**), `target_resolution` (`768`), `crop_factor_override` (`0.0` = use JSON padding), `skip_upscale_if_larger`, `feather_mask`.
   - **Outputs**: `cropped_images` (`IMAGE`), `crop_masks` (`MASK`), `stitch_pipe` (`ZIGGART_STITCH_PIPE`).
2. **BBox Feathered Stitch (`ZiggartStitchByBBox`)**:
   - **Inputs**: `stitch_pipe` (`ZIGGART_STITCH_PIPE`), `refined_crops` (`IMAGE`), `feather_mask` (`16`), `blend_opacity` (`0.0–1.0`).
   - **Outputs**: `image` (`IMAGE`), `composite_mask` (`MASK`).

---

### 6. GGUF Vision Prompt & GGUF VLM Loader

*Category: `Ziggart/LLM`*

- **GGUF Vision Prompt (`ZiggartGGUFVisionPrompt`)**: Sends images (`image`, optional `detail_crop`, optional `reference_image`) to a local vision-capable GGUF model (e.g. Qwen2.5-VL, Qwen3-VL) with custom system/user instructions and independent **Composition**, **Subject**, and **Style** specialist passes.
- **GGUF VLM Loader (`ZiggartGGUFVLMLoader`)**: Outputs a `GGUF_VL_MODEL` (`vlm_config`) pin that can be wired into any Ziggart VLM or Detailer node so you only configure model paths, `context_size`, `gpu_layers`, and `flash_attention` in one place.

---

### 7. GGUF Quality & Anatomy Critic

*Category: `Ziggart/LLM`*

Automated self-healing inspection node. Scans generated images for defects and outputs localization boxes and repair prompts.

- **Inputs**: `image`, `model`, `mmproj`, `inspection_focus` (`all_defects`, `hands_and_limbs`, `faces_and_eyes`, `text_and_logos`, `custom`), `crop_factor`, `max_detections`, `feather_mask`, `free_vram_after`, and optional `custom_criteria`, `generation_prompt`, `vlm_config`.
- **Outputs**:
  - `preview_image` (`IMAGE`): Annotated defect boxes.
  - `defect_mask` (`MASK`): Mask of defective regions.
  - `bboxes_json` (`STRING`): Ready to wire directly into **BBox Crop**.
  - `fix_positive_prompt` (`STRING`): Tailored repair prompt ready to wire into **Detailers Krea2**.
  - `critique_report` (`STRING`): Structured JSON report with a 1–10 quality score and issue summary.
  - `needs_refinement` (`BOOLEAN`): `True` when at least one defect box was localized.

---

### 8. GGUF Mask-Aware Inpaint Prompter

*Category: `Ziggart/LLM`*

Takes an `IMAGE`, a `MASK` (drawn in ComfyUI's MaskEditor or from a detector), and a short `edit_instruction` (e.g. *"add vintage gold aviator sunglasses"*). It highlights the masked target on the full scene and sends both the global view and close-up crop to the VLM so the generated prompt matches surrounding lighting, perspective, and style.

- **Outputs**: `inpaint_positive_prompt` (`STRING`), `inpaint_negative_prompt` (`STRING`), `annotated_preview` (`IMAGE`), and `bboxes_json` (`STRING`).

---

## Architecture & Workflows

### Workflow A: Self-Healing Generation with Quality Critic + BBox Crop/Stitch

```text
[ Base Generated Image ]
           |
           v
[ GGUF Quality & Anatomy Critic ]  (inspection_focus: "hands_and_limbs")
     |                    |
     | (bboxes_json)      +---> (fix_positive_prompt)
     v                                    |
[ BBox Crop ]                             v
     | (cropped_images) --------> [ Custom KSampler / Detailer ]
     | (stitch_pipe)                      | (refined_crops)
     +--------------------+---------------+
                          |
                          v
              [ BBox Feathered Stitch ]
                          |
                          v
              [ Repaired Final Image ]
```

### Workflow B: Detailers Krea2 (Standard & VLM Auto-Prompt)

```text
               [ Input Image Batch ]
                         |
                         v
       [ Phase 1: WeVisOne-VL Batch Detection ]
       (Optional: VLM Per-Crop Auto-Prompting)
                         |
           (Unload VLM Once & Free VRAM)
                         |
                         v
       [ Phase 2: Krea 2 Turbo Crop Resampling ]
       (Cached CLIP + Latent Noise Mask + 8 Steps)
                         |
                         v
         [ Gaussian Feathered Alpha Blending ]
                         |
                         v
         [ Composited Image + Masks + Latents ]
```

---

## Installation

### 1. Clone into ComfyUI Custom Nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mskiller/Ziggart-Nodes.git
```

### 2. Install Dependencies

Install `llama-cpp-python` with CUDA acceleration into ComfyUI's Python environment:

```powershell
# For Windows Portable ComfyUI with CUDA 12.4:
cd C:\path\to\ComfyUI_windows_portable
.\python_embeded\python.exe -m pip install -U llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

*(Ensure the CUDA version matches your PyTorch build: `cu121`, `cu124`, etc.)*

---

## Models Download & Setup

### Directory Layout

```text
ComfyUI/
├── models/
│   ├── LLM/
│   │   └── GGUF/
│   │       ├── WeVisOne-VL-2B-Q4_K_M.gguf
│   │       ├── WeVisOne-VL-4B-Q4_K_M.gguf
│   │       ├── mmproj-WeVisOne-VL-F16.gguf
│   │       ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
│   │       └── mmproj-Qwen2.5-VL-F16.gguf
│   └── diffusion_models/
│       └── turbo.safetensors (Krea 2 Turbo)
```

### Download Links

- **WeVisOne-VL GGUFs**:
  - [WeVisOne-VL-2B-GGUF (mradermacher)](https://huggingface.co/mradermacher/WeVisOne-VL-2B-GGUF)
  - [WeVisOne-VL-4B-GGUF (mradermacher)](https://huggingface.co/mradermacher/WeVisOne-VL-4B-GGUF)
  - Matching `mmproj` files are in the respective HuggingFace repositories.
- **Krea 2 Turbo Diffusion Model**:
  - [Krea-2-Turbo on HuggingFace](https://huggingface.co/krea/Krea-2-Turbo)
  - Connect with standard ComfyUI `DiffusionModelLoader` / `UNETLoader`, `CLIPLoader`, and `VAELoader`.
- **Vision-Language Models (for GGUF Vision Prompt, Critic, Auto-Prompt & Tile Upscaler)**:
  - [Qwen2.5-VL GGUF Models](https://huggingface.co/collections/Qwen/qwen25-vl-6799bbb72b53f668d227951a)

---

## Hardware & VRAM Optimization

| GPU VRAM | Recommended Settings |
| :--- | :--- |
| **12 GB** | `WeVisOne-VL-2B-Q4_K_M.gguf`, `free_vram_after: True`, `target_detail_resolution: 512`, `encode_final_latent: False`, `gpu_layers: -1`. |
| **16 GB** | `WeVisOne-VL-4B-Q4_K_M.gguf`, `free_vram_after: True`, `target_detail_resolution: 768`, `gpu_layers: -1`. |
| **24 GB+** | `WeVisOne-VL-4B-BF16.gguf`, `free_vram_after: False` (instant repeated inference), `target_detail_resolution: 1024`. |

When `free_vram_after` is enabled, the node closes both the vision model and native `MTMDChatHandler` projector, runs garbage collection, and asks ComfyUI to empty its accelerator cache before the diffusion model samples.

---

## Troubleshooting

- **`No detector_model was found`**: Ensure your `.gguf` files are located in `ComfyUI/models/LLM/GGUF/` and click **Refresh** in ComfyUI.
- **`MTMDChatHandler cannot be imported`**: Update `llama-cpp-python` to `>= 0.3.35`.
- **Object not detected**: Adjust `detection_prompt` (e.g. use specific terms like `"human eyes"`, `"face"`, or `"iris"`). Check `detection_preview` output to inspect model predictions.
- **Out of Memory during sampling**: Ensure `free_vram_after` is set to `True`, disable `encode_final_latent` on 4K images, or reduce `target_detail_resolution` from `768` to `512`.

---

## License

This project is licensed under the Apache 2.0 License. Model weights are subject to their respective creators' licenses (Tencent WeVisOne-VL, Krea 2 Community License, Qwen License).
