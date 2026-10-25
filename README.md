# Ziggart Nodes for ComfyUI

A collection of local, in-process ComfyUI nodes combining cutting-edge vision-language models (VLMs via GGUF) and diffusion models. 

Includes:
- **GGUF Vision Prompt**: In-process multimodal prompt generation, image analysis, and specialist passes (composition, subject, style) using local GGUF models.
- **Detailers Krea2**: High-performance inpainting and resampling detailer using **WeVisOne-VL** for open-vocabulary natural language object detection and **Krea 2 Turbo** for fast, high-density detail refinement.
- **WeVisOne-VL Detector**: Standalone open-vocabulary detection node outputting bounding-box previews, segmentation masks, cropped patches, and JSON coordinates.

---

## Table of Contents

1. [Features](#features)
2. [Nodes Overview](#nodes-overview)
   - [1. Detailers Krea2](#1-detailers-krea2)
   - [2. WeVisOne-VL Detector](#2-wevisone-vl-detector)
   - [3. GGUF Vision Prompt](#3-gguf-vision-prompt)
3. [Architecture & Workflow](#architecture--workflow)
4. [Installation](#installation)
5. [Models Download & Setup](#models-download--setup)
6. [Hardware & VRAM Optimization](#hardware--vram-optimization)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## Features

- **Zero External Servers**: Runs directly in the ComfyUI Python process via `llama-cpp-python`. No Ollama, Docker, or standalone `llama-server` needed.
- **Natural Language Grounding**: Detect any feature (eyes, pupils, face, lips, hands, jewelry, clothing, background elements) with plain text prompts using Tencent's WeVisOne-VL.
- **Krea 2 Turbo Resampling**: Purpose-built for Krea 2 Turbo's 12B DiT architecture (8 steps, CFG 1.5), scaling crops to high pixel density before sampling.
- **Seamless Alpha Feathering**: Gaussian border blending prevents seams, halos, and hard edges between refined regions and original images.
- **Dynamic VRAM Handoff**: Automatically unloads the vision model from GPU memory before Krea 2 Turbo begins sampling, enabling operation on 12 GB and 16 GB GPUs.

---

## Nodes Overview

### 1. Detailers Krea2

*Category: `Ziggart/Detailer`*

Inspired by the Impact-Pack detailer workflow, but designed for prompt-guided open-vocabulary localization and Krea 2 Turbo resampling.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Input image or batch of images. |
| `model` | `MODEL` | Diffusion model (Krea 2 Turbo). |
| `clip` | `CLIP` | CLIP text encoder. |
| `vae` | `VAE` | VAE encoder/decoder. |
| `detector_model` | Dropdown | WeVisOne-VL GGUF model from `models/LLM/GGUF/`. |
| `detector_mmproj` | Dropdown | Matching vision projector (`mmproj-*.gguf`). |
| `detection_prompt` | `STRING` | Natural language description of what to detect (e.g. `eyes`, `face`, `hands`). |
| `detail_positive_prompt` | `STRING` | Prompt applied specifically during crop resampling (e.g. `ultra-detailed sharp eyes, realistic iris reflections, crystal clear`). |
| `detail_negative_prompt` | `STRING` | Negative prompt for crop resampling (e.g. `blurry, deformed, noisy`). |
| `crop_factor` | `FLOAT` (1.0 - 4.0) | Context padding around the detected box (default: `1.5`). Provides contextual orientation for diffusion. |
| `target_detail_resolution` | `INT` (256 - 2048) | Upscaling dimension before diffusion sampling (default: `768`). |
| `steps` | `INT` | Sampling steps (default: `8`, optimal for Krea 2 Turbo). |
| `cfg` | `FLOAT` | Classifier-free guidance scale (default: `1.5`). |
| `denoise` | `FLOAT` (0.0 - 1.0) | Inpainting strength (default: `0.40`). Use `0.30-0.45` for enhancement, `0.60+` for full redraw. |
| `sampler_name` | Dropdown | Sampler (e.g. `euler`, `dpmpp_2m`). |
| `scheduler` | Dropdown | Scheduler (e.g. `simple`, `sgm_uniform`). |
| `seed` | `INT` | Random seed for sampling. |
| `max_detections` | `INT` (1 - 20) | Maximum number of detected regions to process per image (default: `2`). |
| `feather_mask` | `INT` (0 - 128) | Gaussian blur radius in pixels applied to borders for invisible blending (default: `16`). |
| `free_vram_after` | `BOOLEAN` | Unloads WeVisOne-VL from VRAM before sampling begins (default: `True`). |

#### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | The full composited image with refined regions seamlessly blended. |
| `cropped_refined` | `IMAGE` | Batch of high-resolution refined crops. |
| `cropped_base` | `IMAGE` | Batch of original crops before refinement (for comparison). |
| `detection_preview` | `IMAGE` | Preview image with colored bounding boxes and resolution labels. |
| `mask` | `MASK` | Full-size composite mask of the modified areas. |
| `latent` | `LATENT` | Latent representation of the final image. |

---

### 2. WeVisOne-VL Detector

*Category: `Ziggart/Detector`*

A standalone visual grounding node using WeVisOne-VL. Extracts bounding boxes, segmentation masks, cropped patches, and JSON coordinates without running diffusion.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `image` | `IMAGE` | Input image. |
| `detector_model` | Dropdown | WeVisOne-VL GGUF model selector. |
| `detector_mmproj` | Dropdown | Matching mmproj projector. |
| `detection_prompt` | `STRING` | Natural language target (e.g. `"eyes"`, `"jewelry"`, `"sneakers"`). |
| `crop_factor` | `FLOAT` | Bounding box padding multiplier (default: `1.5`). |
| `max_detections` | `INT` | Max items to return (default: `4`). |
| `feather_mask` | `INT` | Feathering radius for the output mask (default: `0`). |
| `free_vram_after` | `BOOLEAN` | Unloads model from memory after detection (default: `True`). |

#### Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `detection_preview` | `IMAGE` | Image showing detected bounding boxes with tags and dimensions. |
| `mask` | `MASK` | Binary or feathered mask of detected objects. |
| `cropped_images` | `IMAGE` | Batch of cropped patches corresponding to the detected regions. |
| `bboxes_json` | `STRING` | Structured JSON with detected coordinates `[x0, y0, x1, y1]`. |

---

### 3. GGUF Vision Prompt

*Category: `Ziggart/LLM`*

Sends images to a local vision-capable GGUF model (e.g. Qwen2.5-VL, Qwen3.5) with custom system prompts, user instructions, and optional specialist passes.

#### Inputs & Features

- `image`: ComfyUI `IMAGE` batch support.
- `detail_crop`: Optional high-resolution crop for fine details.
- `reference_image`: Optional secondary reference image.
- `model` / `mmproj`: GGUF selectors from `models/LLM/GGUF/`.
- `system_prompt` & `prompt`: Multiline instruction strings.
- **Specialist Passes**: Independent switches for Composition, Subject, and Style passes.
- `free_vram_after` & `unload_comfy_models_before`: Intelligent VRAM lifecycle management.

#### Outputs

- `answer`: Synthesized final prompt/description.
- `composition`: Composition analysis notes.
- `subject`: Subject identity and detail notes.
- `style`: Aesthetic and lighting notes.

---

## Architecture & Workflow

```
               [ Input Image ]
                      |
                      v
         [ WeVisOne-VL Detection ]  <-- "eyes", "face", "jewelry"
                      |
                      v
         [ Bounding Box & Padding ]
                      |
         +------------+------------+
         |                         |
         v                         v
  [ Crop & Rescale ]      [ Detection Preview ]
   (e.g. to 768px)
         |
         v
 [ Krea 2 Turbo Resample ] <-- Detail Prompt ("ultra-sharp blue eyes...")
   (8 steps, CFG 1.5, denoise 0.40)
         |
         v
  [ Scale Down & Blend ]   <-- Gaussian Feathering (no seams)
         |
         v
  [ Composited Full Image ] + [ Masks & Latents ]
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
- **Vision-Language Models (for GGUF Vision Prompt)**:
  - [Qwen2.5-VL GGUF Models](https://huggingface.co/collections/Qwen/qwen25-vl-6799bbb72b53f668d227951a)

---

## Hardware & VRAM Optimization

| GPU VRAM | Recommended Settings |
| :--- | :--- |
| **12 GB** | `WeVisOne-VL-2B-Q4_K_M.gguf`, `free_vram_after: True`, `target_detail_resolution: 512`, `gpu_layers: -1`. |
| **16 GB** | `WeVisOne-VL-4B-Q4_K_M.gguf`, `free_vram_after: True`, `target_detail_resolution: 768`, `gpu_layers: -1`. |
| **24 GB+** | `WeVisOne-VL-4B-BF16.gguf`, `free_vram_after: False` (instant repeated inference), `target_detail_resolution: 1024`. |

When `free_vram_after` is enabled, the node releases the vision model's GPU memory, runs garbage collection, and asks ComfyUI to empty its accelerator cache before the diffusion model samples.

---

## Troubleshooting

- **`No detector_model was found`**: Ensure your `.gguf` files are located in `ComfyUI/models/LLM/GGUF/` and click **Refresh** in ComfyUI.
- **`MTMDChatHandler cannot be imported`**: Update `llama-cpp-python` to `>= 0.3.35`.
- **Object not detected**: Adjust `detection_prompt` (e.g. use specific terms like `"human eyes"`, `"face"`, or `"iris"`). Check `detection_preview` output to inspect model predictions.
- **Out of Memory during sampling**: Ensure `free_vram_after` is set to `True`, or reduce `target_detail_resolution` from `768` to `512`.

---

## License

This project is licensed under the Apache 2.0 License. Model weights are subject to their respective creators' licenses (Tencent WeVisOne-VL, Krea 2 Community License, Qwen License).
