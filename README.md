# Ziggart Nodes for ComfyUI

A complete collection of local, in-process ComfyUI nodes combining **700+ MB Wildcard & Canonical Theme Engines**, **GGUF Vision-Language Models (VLMs)**, **WeVisOne-VL Open-Vocabulary Grounding**, and **Krea 2 Turbo** diffusion refinement.

### Included Nodes (18 Nodes)

| Display Name | Internal Class | Category | Purpose |
| :--- | :--- | :--- | :--- |
| **Ziggart Fast Wildcard Processor** | `ZiggartWildcardProcessor` | `Ziggart/Wildcards` | **(Way C1)** Ultra-fast drop-in replacement for `ImpactWildcardProcessor` with byte-offset YAML indexing, $O(1)$ `.txt` lookup, and a working `select_to_add_wildcard` dropdown. |
| **Ziggart Wildcard Processor (All-in-One Pillars & Mixed)** | `ZiggartWildcardProcessorAllInOne` | `Ziggart/Wildcards` | **(Way A)** All-in-one wildcard processor with 6 dedicated `Collection/` Pillar & Mixed dropdowns + `wildcard_text`. |
| **Ziggart Collection Scene Builder** | `ZiggartCollectionSceneBuilder` | `Ziggart/Wildcards` | **(Way C2)** Modular visual scene composer organized around `Collection/` folders (`Macro/YAML/Themes`, `Character/Anatomy`, `Outfit/Pose`, `Environment/Vehicle`, `Style/Camera/Lighting`, and `Mixed`). |
| **Ziggart Fast Wildcard Encode (LoRA)** | `ZiggartWildcardEncode` | `Ziggart/Wildcards` | Expands wildcards, extracts & loads `<lora:name:model_w:clip_w>` tags, and encodes `BREAK`-separated segments into `CONDITIONING`. |
| **Ziggart Canonical Theme & Character-Outfit Resolver** | `ZiggartCanonicalThemeResolver` | `Ziggart/Wildcards` | Reads `Character-Outfits.json` inside `Collection/Themes/*/` to pair characters (`Single`, `Duo`, `Trio`) with their authentic canonical outfits and matching theme environment/style. |
| **Wildcard-to-VLM Prompt Harmonizer** | `ZiggartWildcardVLMEnhancer` | `Ziggart/Wildcards` | Uses your local GGUF LLM/VLM to rewrite stitched wildcard fragments into cohesive natural-language prose for **Krea 2 Turbo / Flux** while keeping character/outfit locks intact. |
| **Ziggart Prompt Recipe & Comparison Saver** | `ZiggartWildcardRecipeSaver` | `Ziggart/Wildcards` | Saves images (or side-by-side `Base vs Refined` comparison grids) with embedded PNG metadata (`wildcard_template`, `populated_text`, `seed`) and a `.json` recipe sidecar. |
| **Detailers Krea2** | `ZiggartDetailerKrea2` | `Ziggart/Detailer` | Open-vocabulary object detection (**WeVisOne-VL**) paired with high-resolution **Krea 2 Turbo** resampling, latent noise masking, and Gaussian feathered blending. |
| **Detailers Krea2 (VLM Auto-Prompt)** | `ZiggartAutoPromptDetailer` | `Ziggart/Detailer` | Detects target zones, uses the VLM to write a custom micro-prompt for each individual crop while in VRAM, unloads once, and refines every crop with Krea 2 Turbo. |
| **Multi-Character Grounded Krea2 Detailer** | `ZiggartMultiSubjectDetailer` | `Ziggart/Detailer` | Locates up to 3 distinct subjects/characters with WeVisOne-VL and refines each subject crop with Krea 2 Turbo using its own dedicated character prompt. |
| **Krea2 VLM Smart Tile Upscaler** | `ZiggartSmartTileUpscaler` | `Ziggart/Detailer` | Overlapping tiled upscaler (1.0×–4.0×) with optional per-tile VLM captioning (preventing background hallucinations) or fast pure-Krea2 tiled upscaling. |
| **GGUF Vision Prompt** | `ZiggartGGUFVisionPrompt` | `Ziggart/LLM` | Multimodal prompt generation, image analysis, and specialist passes (Composition, Subject, Style). |
| **GGUF VLM Loader** | `ZiggartGGUFVLMLoader` | `Ziggart/LLM` | Reusable `GGUF_VL_MODEL` configuration node shared across all Ziggart VLM & Detailer nodes. |
| **GGUF Quality & Anatomy Critic** | `ZiggartVLMImageCritic` | `Ziggart/LLM` | Scans generated images for defects (hands, eyes, anatomy, text, artifacts), outputting defect boxes, masks, repair prompts, and a pass/fail boolean gate. |
| **GGUF Mask-Aware Inpaint Prompter** | `ZiggartVLMInpaintPrompter` | `Ziggart/LLM` | Highlights a user `MASK` on the scene and uses the VLM to generate a lighting- and style-harmonized inpaint prompt + crop coordinates. |
| **WeVisOne-VL Detector** | `ZiggartWeVisOneDetector` | `Ziggart/Detector` | Standalone open-vocabulary grounding node outputting bounding-box previews, masks, cropped patches, and `bboxes_json`. |
| **BBox Crop** | `ZiggartCropByBBox` | `Ziggart/Detector` | Extracts mod-8 padded, resolution-standardized crops from `bboxes_json` and outputs a `ZIGGART_STITCH_PIPE` for custom ComfyUI workflows. |
| **BBox Feathered Stitch** | `ZiggartStitchByBBox` | `Ziggart/Detector` | Blends externally processed crops back into their original locations using `ZIGGART_STITCH_PIPE` and Gaussian alpha feathering. |

---

## Table of Contents

1. [Key Features](#key-features)
2. [Wildcard & Canonical Theme Suite (`Ziggart/Wildcards`)](#wildcard--canonical-theme-suite-ziggartwildcards)
   - [1. Ziggart Fast Wildcard Processor (Way C1)](#1-ziggart-fast-wildcard-processor-way-c1)
   - [2. Ziggart Wildcard Processor — All-in-One Pillars & Mixed (Way A)](#2-ziggart-wildcard-processor--all-in-one-pillars--mixed-way-a)
   - [3. Ziggart Collection Scene Builder (Way C2)](#3-ziggart-collection-scene-builder-way-c2)
   - [4. Ziggart Fast Wildcard Encode (LoRA)](#4-ziggart-fast-wildcard-encode-lora)
   - [5. Ziggart Canonical Theme & Character-Outfit Resolver](#5-ziggart-canonical-theme--character-outfit-resolver)
   - [6. Wildcard-to-VLM Prompt Harmonizer](#6-wildcard-to-vlm-prompt-harmonizer)
   - [7. Ziggart Prompt Recipe & Comparison Saver](#7-ziggart-prompt-recipe--comparison-saver)
3. [Detailer, Grounding & VLM Suite](#detailer-grounding--vlm-suite)
   - [8. Detailers Krea2 & VLM Auto-Prompt Detailer](#8-detailers-krea2--vlm-auto-prompt-detailer)
   - [9. Multi-Character Grounded Krea2 Detailer](#9-multi-character-grounded-krea2-detailer)
   - [10. Krea2 VLM Smart Tile Upscaler](#10-krea2-vlm-smart-tile-upscaler)
   - [11. WeVisOne-VL Detector, BBox Crop & BBox Feathered Stitch](#11-wevisone-vl-detector-bbox-crop--bbox-feathered-stitch)
   - [12. GGUF Vision Prompt, VLM Loader, Quality Critic & Mask Inpaint Prompter](#12-gguf-vision-prompt-vlm-loader-quality-critic--mask-inpaint-prompter)
4. [Architecture & Workflows](#architecture--workflows)
5. [Installation & Models Setup](#installation--models-setup)
6. [Hardware & VRAM Optimization](#hardware--vram-optimization)
7. [License](#license)

---

## Key Features

- **Zero-Freeze 700+ MB Wildcard Engine**:
  - **Streaming Byte-Offset YAML Indexer**: Indexes 50+ MB / 150,000+ line YAML files (like `Ziggart.yaml`) at **any nesting depth** (3, 4, 5+ levels) in `~195 ms` cold and **`36 ms` cached** on ComfyUI startup. Resolves keys by `seek()`ing directly to byte offsets (`~13 ms`) without loading 50 MB into RAM.
  - **$O(1)$ Direct `.txt` Lookup**: Resolves 77,000+ `.txt` files across `Collection/Themes`, `Collection/Combos`, and `Collection/formats-wild` in `0.05 ms` without walking 78,000 files on ComfyUI boot.
  - **Working Categorized Dropdowns**: Organizes wildcards into browser-friendly `Collection/` Pillar dropdowns + a `Mixed` dropdown so ComfyUI never chokes on an 80,000-item combo list.
- **Zero External Servers**: Runs GGUF VLMs directly inside the ComfyUI Python process via `llama-cpp-python`.
- **Two-Phase Batch Execution**: Runs all VLM grounding/captioning passes first, unloads the VLM from VRAM **once**, caches static CLIP conditioning, and then runs all diffusion sampling passes.
- **Latent Noise Masking + Pixel Feathering**: Attaches feathered noise masks directly to crop latents during KSampler steps and blends borders with Gaussian alpha feathering.

---

## Wildcard & Canonical Theme Suite (`Ziggart/Wildcards`)

Automatically discovers wildcards in `ComfyUI/custom_nodes/ComfyUI-Impact-Pack/wildcards`, `ComfyUI/wildcards`, `Ziggart-Nodes/wildcards`, or any custom path—without modifying external folders.

###Full Wildcard Syntax Support
- **Arbitrary-Depth YAML & TXT Tokens**: `__Ziggart/scenes/all__`, `__Ziggart/scenes/01_mysterieusescitesdor/duo/romantic__`, `__Collection/Themes/01-MysterieusesCitesDOr/Character__`, `__female-full__`, or parent rollups like `__Ziggart/scenes__`.
- **Weighted Dynamic Choices**: `{option A|option B|2.5::weighted option C}` (supports nested braces `{a|{b|c}}`).
- **Multi-Select Ranges**: `{2$$, $$__wildcard__}` or `{1-3$$ and $$opt1|opt2|opt3}`.
- **Quantifiers**: `3#__wildcard__`.
- **LoRA Tags**: `<lora:lora_name:1.0:0.8>`.

---

### 1. Ziggart Fast Wildcard Processor (Way C1)

Drop-in replacement for `ImpactWildcardProcessor` with instant startup and a working `select_to_add_wildcard` dropdown.

- **Inputs**: `wildcard_text`, `populated_text`, `mode` (`populate`, `fixed`, `reproduce`), `seed`, `select_to_add_wildcard` (clicking any item appends it directly into `wildcard_text` in the ComfyUI UI, or resolves it directly if `wildcard_text` is empty), `search_wildcard_filter` (type any keyword like `albator` or `bebop` to match across all 78,000+ wildcards), `custom_wildcards_dir`, `refresh_index`.
- **Outputs**: `processed_text` (`STRING`), `populated_text` (`STRING`).

---

### 2. Ziggart Wildcard Processor — All-in-One Pillars & Mixed (Way A)

Combines `wildcard_text` and `populated_text` with **6 dedicated dropdowns** mirroring your `Collection/` structure and root folders:

| Dropdown Widget | Populated From (`Collection/` & `Ziggart.yaml`) |
| :--- | :--- |
| `select_macro_yaml_theme` | `Ziggart.yaml` scenes (`__ziggart/scenes/all__`, `female_all`, `duo_all`, `01_...`), `ziggart/generators/*`, `ziggart/random/*`, `Collection/Themes/*/ALL`, `Collection/Combos/*` |
| `select_character_anatomy` | `Collection/Characters/*`, `Collection/Anatomy/*`, `Collection/DuosTeams/*`, `Collection/Bestiary/*`, `ziggart/character/*` |
| `select_outfit_pose` | `Collection/Outfits/*`, `Collection/Pose/*`, `Collection/Expression/*`, `ziggart/outfit/*`, `ziggart/pose/*`, `ziggart/action_pose/*` |
| `select_env_vehicle` | `Collection/Environment/*`, `Collection/IconicPlaces/*`, `Collection/Vehicle/*`, `ziggart/environment/*` |
| `select_style_camera_light` | `Collection/Style/*`, `Collection/Medium/*`, `Collection/Lighting/*`, `Collection/Camera/*`, `Collection/VFX/*`, `Collection/Sheet/*`, `ziggart/style/*`, `ziggart/camera/*` |
| `select_mixed_other` | Root `.txt` files (`__female-full__`, `__style-full-v2__`, etc.), `ziggart/*`, `Master/*`, `Encours/*`, `jff/*`, `Outfits/*`, `Characters/*`, `Collection/Universes/*`, `Collection/Canonical/*`, `Collection/Index/*` |

- **Dual Mode (`combine_selected_pillars`)**:
  - When `combine_selected_pillars = True` (default): Any non-`[None]` selections across the 6 dropdowns remain visible in their slots and are automatically combined with `wildcard_text` at execution time.
  - When `combine_selected_pillars = False`: Selecting an item from any of the 6 dropdowns immediately inserts `__{wildcard}__` into `wildcard_text` and resets the dropdown to `[None]`.
- **Outputs**: `processed_text` (`STRING`), `populated_text` (`STRING`), `combined_wildcard_recipe` (`STRING`).

---

### 3. Ziggart Collection Scene Builder (Way C2)

Dedicated modular scene composer with `custom_prefix`, the 6 `Collection/` Pillar & Mixed dropdowns, `custom_suffix`, and `seed`.
- **Outputs**:
  - `populated_prompt` (`STRING`): Fully resolved prompt ready for CLIP or VLM harmonization.
  - `wildcard_template` (`STRING`): Unexpanded `__...__` recipe string that can be wired into **Ziggart Fast Wildcard Processor** or **Ziggart Prompt Recipe & Comparison Saver**.

---

### 4. Ziggart Fast Wildcard Encode (LoRA)

Drop-in replacement for `ImpactWildcardEncode`. Expands wildcards, strips and loads `<lora:name:model_w:clip_w>` tags onto `MODEL` and `CLIP`, and encodes `BREAK`-delimited prompt segments into `CONDITIONING`.
- **Outputs**: `model` (`MODEL`), `clip` (`CLIP`), `conditioning` (`CONDITIONING`), `populated_text` (`STRING`).

---

### 5. Ziggart Canonical Theme & Character-Outfit Resolver

Reads `Character-Outfits.json` inside `wildcards/Collection/Themes/<Theme>/` on-demand so characters are **always paired with their authentic `canonical_outfits`** instead of mismatched random outfit rolls.

- **Inputs**:
  - `theme_selection`: Choose any of the 1,500+ themes or `[Random Theme]`.
  - `cast_mode`: `Single (Any Gender)`, `Single (Female)`, `Single (Male)`, `Duo (Any 2 Characters)`, `Duo (Female + Male)`, `Duo (Female + Female)`, or `Trio (3 Characters)`.
  - `include_pose_action`, `include_environment`, `include_vehicle`, `include_creature`, `include_style` (`BOOLEAN` toggles).
  - `theme_search_filter` *(optional)*: Keyword filter when `theme_selection` is `[Random Theme]` (e.g. `"albator"`, `"bebop"`, `"cyberpunk"`).
- **Outputs**:
  - `scene_prompt` (`STRING`): Full canonical scene prompt.
  - `character_1_prompt`, `character_2_prompt`, `character_3_prompt` (`STRING`): Individual canonical character + outfit descriptions—ready to wire directly into **Multi-Character Grounded Krea2 Detailer**!
  - `theme_metadata_json` (`STRING`): JSON breakdown of the resolved theme and cast.

---

### 6. Wildcard-to-VLM Prompt Harmonizer

Bridges your Wildcard Processor and your local **GGUF LLM/VLM**. Stitched wildcard snippets from 6+ `.txt` pools can sometimes contain duplicate tags or conflicting lighting. This node uses your local GGUF model to rewrite the expanded wildcard prompt into fluent, cohesive prose optimized for **Krea 2 Turbo / Flux** while strictly preserving character names, physical traits, and canonical outfits.
- **Styles**: `Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)`, `Cinematic Director Shot (Rich Lighting & Camera)`, `Concise High-Impact Tags & Prose Hybrid`, or `Custom Instruction`.
- **Outputs**: `harmonized_prompt` (`STRING`), `raw_populated_text` (`STRING`).

---

### 7. Ziggart Prompt Recipe & Comparison Saver

Saves generated images (and an optional side-by-side `Base vs Refined` comparison grid when `base_image` is connected and `build_comparison_grid = True`) with embedded PNG metadata (`ziggart_wildcard_template`, `ziggart_populated_text`, `ziggart_seed`, `ziggart_notes`) and an optional `.json` sidecar file so you never lose the exact wildcard recipe behind a great generation.
- **Outputs**: `output_image_or_grid` (`IMAGE`), `recipe_summary` (`STRING`), `saved_paths_json` (`STRING`).

---

## Detailer, Grounding & VLM Suite

### 8. Detailers Krea2 & VLM Auto-Prompt Detailer

*Category: `Ziggart/Detailer`*

- **Detailers Krea2 (`ZiggartDetailerKrea2`)**: Uses **WeVisOne-VL** to locate any natural-language target (`"eyes"`, `"face"`, `"hands"`, `"jewelry"`) and resamples each padded crop with **Krea 2 Turbo** (8 steps, CFG 1.5, latent noise masking, Gaussian border feathering).
- **Detailers Krea2 (VLM Auto-Prompt) (`ZiggartAutoPromptDetailer`)**: Inspects each detected crop with the GGUF VLM while in memory to write a custom per-crop micro-prompt before unloading once and refining with Krea 2 Turbo.

### 9. Multi-Character Grounded Krea2 Detailer

*Category: `Ziggart/Detailer`*

Purpose-built for `duo_all`, `trio_all`, `romantic_all`, and **Ziggart Canonical Theme Resolver** scenes.
- Accepts up to 3 `(subject_N_target, subject_N_detail_prompt)` pairs.
- Grounds each character separately with **WeVisOne-VL** in Phase 1, unloads the VLM once, and refines each character crop with **Krea 2 Turbo** using that character's own canonical prompt in Phase 2.

### 10. Krea2 VLM Smart Tile Upscaler

*Category: `Ziggart/Detailer`*

Tiled 1.0×–4.0× upscaler for Krea 2 Turbo with overlapping mod-8 tiles, optional per-tile VLM captioning (`enable_vlm_tile_prompts = True` to prevent hallucinated subjects in background tiles, or `False` for fast pure-Krea2 tiled upscaling), and weighted Gaussian feather accumulation.

### 11. WeVisOne-VL Detector, BBox Crop & BBox Feathered Stitch

*Category: `Ziggart/Detector`*

- **WeVisOne-VL Detector (`ZiggartWeVisOneDetector`)**: Standalone open-vocabulary detector outputting `detection_preview`, `mask`, `cropped_images`, and `bboxes_json`.
- **BBox Crop (`ZiggartCropByBBox`)** & **BBox Feathered Stitch (`ZiggartStitchByBBox`)**: Modular crop-and-stitch pair that routes `bboxes_json` crops through any external ComfyUI nodes and blends them back with Gaussian feathering.

### 12. GGUF Vision Prompt, VLM Loader, Quality Critic & Mask Inpaint Prompter

*Category: `Ziggart/LLM`*

- **GGUF Vision Prompt (`ZiggartGGUFVisionPrompt`)**: Multimodal image analysis and prompt generation with independent Composition, Subject, and Style specialist passes.
- **GGUF VLM Loader (`ZiggartGGUFVLMLoader`)**: Reusable `GGUF_VL_MODEL` (`vlm_config`) node.
- **GGUF Quality & Anatomy Critic (`ZiggartVLMImageCritic`)**: Scans images for anatomical/rendering defects, outputting `preview_image`, `defect_mask`, `bboxes_json`, `fix_positive_prompt`, `critique_report`, and `needs_refinement`.
- **GGUF Mask-Aware Inpaint Prompter (`ZiggartVLMInpaintPrompter`)**: Highlights a user `MASK` on the full scene so the VLM writes a lighting- and style-consistent inpainting prompt + `bboxes_json`.

---

## Architecture & Workflows

### End-to-End Canonical Duo Workflow

```text
[ Ziggart Canonical Theme & Character-Outfit Resolver ]
   (cast_mode: "Duo (Female + Male)")
     |                       |                         |
     | (scene_prompt)        | (character_1_prompt)    | (character_2_prompt)
     v                       |                         |
[ Wildcard-to-VLM Harmonizer ]                         |
     |                                                 |
     v                                                 |
[ Krea 2 Turbo Base Sampler ]                          |
     | (base_image)                                    |
     v                                                 |
[ Multi-Character Grounded Krea2 Detailer ] <----------+
     | (refined_image)
     v
[ Ziggart Prompt Recipe & Comparison Saver ]
```

---

## Installation & Models Setup

### 1. Clone into ComfyUI Custom Nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mskiller/Ziggart-Nodes.git
```

### 2. Install Dependencies

```powershell
# For Windows Portable ComfyUI with CUDA 12.4:
cd C:\path\to\ComfyUI_windows_portable
.\python_embeded\python.exe -m pip install -U llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### 3. Model Directory Layout

Place `.gguf` vision-language models and `mmproj` projectors in `ComfyUI/models/LLM/GGUF/`:
- **WeVisOne-VL GGUFs**: [WeVisOne-VL-2B-GGUF](https://huggingface.co/mradermacher/WeVisOne-VL-2B-GGUF) / [WeVisOne-VL-4B-GGUF](https://huggingface.co/mradermacher/WeVisOne-VL-4B-GGUF)
- **Qwen2.5-VL GGUFs**: [Qwen2.5-VL Collection](https://huggingface.co/collections/Qwen/qwen25-vl-6799bbb72b53f668d227951a)
- **Krea 2 Turbo**: [Krea-2-Turbo on HuggingFace](https://huggingface.co/krea/Krea-2-Turbo)

---

## Hardware & VRAM Optimization

| GPU VRAM | Recommended Settings |
| :--- | :--- |
| **12 GB** | `WeVisOne-VL-2B-Q4_K_M.gguf`, `free_vram_after: True`, `target_detail_resolution: 512`, `encode_final_latent: False`, `gpu_layers: -1`. |
| **16 GB** | `WeVisOne-VL-4B-Q4_K_M.gguf`, `free_vram_after: True`, `target_detail_resolution: 768`, `gpu_layers: -1`. |
| **24 GB+** | `WeVisOne-VL-4B-BF16.gguf`, `free_vram_after: False` (instant repeated inference), `target_detail_resolution: 1024`. |

---

## License

This project is licensed under the Apache 2.0 License. Model weights are subject to their respective creators' licenses (Tencent WeVisOne-VL, Krea 2 Community License, Qwen License).
