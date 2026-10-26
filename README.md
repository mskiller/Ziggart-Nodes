# Ziggart Nodes for ComfyUI (`v1.2.0`)

A complete collection of **18 local, in-process ComfyUI nodes** combining:
- **Zero-Freeze 700+ MB Wildcard & Canonical Theme Engines** (Way A + Way C architectures)
- **GGUF Vision-Language Models (VLMs)** via `llama-cpp-python`
- **WeVisOne-VL Open-Vocabulary Visual Grounding**
- **Krea 2 Turbo** High-Resolution Detailing & Tiled Upscaling

---

## Included Nodes (18 Nodes)

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

1. [Key Features & 700+ MB Wildcard Benchmarks](#key-features--700-mb-wildcard-benchmarks)
2. [Wildcard & Canonical Theme Suite (`Ziggart/Wildcards`)](#wildcard--canonical-theme-suite-ziggartwildcards)
   - [1. Ziggart Fast Wildcard Processor (Way C1)](#1-ziggart-fast-wildcard-processor-way-c1)
   - [2. Ziggart Wildcard Processor — All-in-One Pillars & Mixed (Way A)](#2-ziggart-wildcard-processor--all-in-one-pillars--mixed-way-a)
   - [3. Ziggart Collection Scene Builder (Way C2)](#3-ziggart-collection-scene-builder-way-c2)
   - [4. Ziggart Fast Wildcard Encode (LoRA)](#4-ziggart-fast-wildcard-encode-lora)
   - [5. Ziggart Canonical Theme & Character-Outfit Resolver](#5-ziggart-canonical-theme--character-outfit-resolver)
   - [6. Wildcard-to-VLM Prompt Harmonizer](#6-wildcard-to-vlm-prompt-harmonizer)
   - [7. Ziggart Prompt Recipe & Comparison Saver](#7-ziggart-prompt-recipe--comparison-saver)
3. [Detailer, Grounding & VLM Suite](#detailer-grounding--vlm-suite)
   - [8. Detailers Krea2](#8-detailers-krea2)
   - [9. Detailers Krea2 (VLM Auto-Prompt)](#9-detailers-krea2-vlm-auto-prompt)
   - [10. Multi-Character Grounded Krea2 Detailer](#10-multi-character-grounded-krea2-detailer)
   - [11. Krea2 VLM Smart Tile Upscaler](#11-krea2-vlm-smart-tile-upscaler)
   - [12. WeVisOne-VL Detector](#12-wevisone-vl-detector)
   - [13. BBox Crop & BBox Feathered Stitch](#13-bbox-crop--bbox-feathered-stitch)
   - [14. GGUF Vision Prompt & GGUF VLM Loader](#14-gguf-vision-prompt--gguf-vlm-loader)
   - [15. GGUF Quality & Anatomy Critic](#15-gguf-quality--anatomy-critic)
   - [16. GGUF Mask-Aware Inpaint Prompter](#16-gguf-mask-aware-inpaint-prompter)
4. [Architecture & Workflows](#architecture--workflows)
5. [Installation](#installation)
6. [Models Download & Setup](#models-download--setup)
7. [Hardware & VRAM Optimization](#hardware--vram-optimization)
8. [Troubleshooting](#troubleshooting)
9. [License](#license)

---

## Key Features & 700+ MB Wildcard Benchmarks

- **Zero-Freeze 700+ MB Wildcard Engine**:
  - **Streaming Byte-Offset YAML Indexer**: Indexes 50+ MB / 150,000+ line YAML files (like `Ziggart.yaml`) at **any nesting depth** (3, 4, 5+ levels) in `~195 ms` cold and **`36 ms` cached** on ComfyUI startup. Resolves keys by `seek()`ing directly to byte offsets (`~13 ms`) without loading 50 MB into RAM.
  - **$O(1)$ Direct `.txt` Lookup**: Resolves 77,000+ `.txt` files across `Collection/Themes`, `Collection/Combos`, and `Collection/formats-wild` in `0.05 ms` without walking 78,000 files on ComfyUI boot.
  - **Working Categorized Dropdowns**: Organizes wildcards into browser-friendly `Collection/` Pillar dropdowns + a `Mixed` dropdown so ComfyUI never chokes on an 80,000-item combo list.
- **Zero External Servers**: Runs GGUF VLMs directly inside the ComfyUI Python process via `llama-cpp-python`. No Ollama, Docker, or standalone `llama-server` needed.
- **Two-Phase Batch Execution**: Across multi-image batches and multi-crop passes, Ziggart runs all VLM grounding/captioning operations first, unloads the VLM from VRAM **once**, caches static CLIP conditioning, and then runs all diffusion sampling passes—eliminating model-swap thrashing.
- **Latent Noise Masking + Pixel Feathering**: Attaches feathered noise masks directly to crop latents during KSampler steps and blends borders with Gaussian alpha feathering, eliminating seams and color shifts even at higher denoise values.
- **Smart `<think>` Tag Handling**: Automatically strips `<think>...</think>` reasoning blocks from thinking VLMs so CLIP prompts and bounding-box parsers stay clean.

### Benchmark vs. Impact Pack (77,146 `.txt` + 1,550 `.json` + 53.5 MB `Ziggart.yaml`)

| Operation | Impact Pack Behavior | Ziggart Wildcard Index (`v1.2.0`) |
| :--- | :--- | :--- |
| **Cold Startup / Refresh** | Minutes (`os.walk` ×2 on 78,697 files + `yaml.FullLoader`) | **195.6 ms** (streaming byte-offset YAML scan + top-level pillar scan) |
| **Warm Startup / Refresh** | Minutes | **36.5 ms** (loads `.cache/wildcard_yaml_offsets.json` verified via `mtime` + `size`) |
| **3+ Level `Ziggart.yaml` Keys** | **0 keys loaded** (silently drops `Ziggart/scenes/all` & all 11,619 nested keys in on-demand mode) | **11,619 YAML keys indexed** across arbitrary `1..N` nesting depths |
| **Dropdown Population** | Empty / broken over 50 MB | **Categorized Pillar & Mixed Dropdowns** (1,502 Themes, 3,286 YAML Macro/Scene keys, + Collection Pillars) |
| **Recursive Resolution (`__Ziggart/scenes/all__`)** | Fails (unresolved token left in prompt) | **13.07 ms** ($O(1)$ byte-offset `seek()` + direct `.txt` path resolution) |

---

## Wildcard & Canonical Theme Suite (`Ziggart/Wildcards`)

Automatically discovers wildcards in `ComfyUI/custom_nodes/ComfyUI-Impact-Pack/wildcards`, `ComfyUI/wildcards`, `Ziggart-Nodes/wildcards`, or any custom path—without modifying external folders.

### Full Wildcard Syntax Support
- **Arbitrary-Depth YAML & TXT Tokens**: `__Ziggart/scenes/all__`, `__Ziggart/scenes/01_mysterieusescitesdor/duo/romantic__`, `__Collection/Themes/01-MysterieusesCitesDOr/Character__`, `__female-full__`, or parent rollups like `__Ziggart/scenes__`.
- **Weighted Dynamic Choices**: `{option A|option B|2.5::weighted option C}` (supports nested braces `{a|{b|c}}`).
- **Multi-Select Ranges**: `{2$$, $$__wildcard__}` or `{1-3$$ and $$opt1|opt2|opt3}`.
- **Quantifiers**: `3#__wildcard__`.
- **LoRA Tags**: `<lora:lora_name:1.0:0.8>`.

---

### 1. Ziggart Fast Wildcard Processor (Way C1)

*Class: `ZiggartWildcardProcessor` | Category: `Ziggart/Wildcards`*

Drop-in replacement for `ImpactWildcardProcessor` with instant startup and a working `select_to_add_wildcard` dropdown.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `wildcard_text` | `STRING` | Prompt with `__wildcard__` or `{a\|b}` syntax. Selecting from `select_to_add_wildcard` appends into this field in the ComfyUI UI (or resolves directly if left empty). |
| `populated_text` | `STRING` | Displays or overrides the expanded prompt when `mode` is `fixed`. |
| `mode` | Dropdown | `populate` (expands `wildcard_text` using `seed`), `fixed` (uses `populated_text` as-is), or `reproduce`. |
| `seed` | `INT` | Deterministic seed for wildcard expansion. |
| `select_to_add_wildcard` | Dropdown | Categorized wildcard list; selecting an item inserts it into `wildcard_text` and resets the dropdown to placeholder. |
| `search_wildcard_filter` *(optional)* | `STRING` | Keyword search (e.g. `albator`, `bebop`, `cyberpunk`): if `wildcard_text` is empty, resolves a matching wildcard across all 78,000+ entries. |
| `custom_wildcards_dir` *(optional)* | `STRING` | Additional wildcards directory path. |
| `refresh_index` *(optional)* | `BOOLEAN` | Enable for one run to rescan YAML/TXT files if you added new files on disk. |

#### Outputs

- `processed_text` (`STRING`): Final fully expanded prompt ready for CLIP or VLM nodes.
- `populated_text` (`STRING`): Populated prompt text.

---

### 2. Ziggart Wildcard Processor — All-in-One Pillars & Mixed (Way A)

*Class: `ZiggartWildcardProcessorAllInOne` | Category: `Ziggart/Wildcards`*

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
- **Outputs**:
  - `processed_text` (`STRING`): Final fully expanded prompt.
  - `populated_text` (`STRING`): Populated prompt text.
  - `combined_wildcard_recipe` (`STRING`): The combined unexpanded wildcard template before resolution.

---

### 3. Ziggart Collection Scene Builder (Way C2)

*Class: `ZiggartCollectionSceneBuilder` | Category: `Ziggart/Wildcards`*

Dedicated modular scene composer with `custom_prefix`, the 6 `Collection/` Pillar & Mixed dropdowns (`select_macro_yaml_theme`, `select_character_anatomy`, `select_outfit_pose`, `select_env_vehicle`, `select_style_camera_light`, `select_mixed_other`), `custom_suffix`, and `seed`.

- **Outputs**:
  - `populated_prompt` (`STRING`): Fully resolved prompt ready for CLIP or VLM harmonization.
  - `wildcard_template` (`STRING`): Unexpanded `__...__` recipe string that can be wired into **Ziggart Fast Wildcard Processor**, **Ziggart Fast Wildcard Encode (LoRA)**, or **Ziggart Prompt Recipe & Comparison Saver**.

---

### 4. Ziggart Fast Wildcard Encode (LoRA)

*Class: `ZiggartWildcardEncode` | Category: `Ziggart/Wildcards`*

Drop-in replacement for `ImpactWildcardEncode`. Expands wildcards, strips and loads `<lora:name:model_w:clip_w>` tags onto `MODEL` and `CLIP`, and encodes `BREAK`-delimited prompt segments into `CONDITIONING`.

- **Inputs**: `model` (`MODEL`), `clip` (`CLIP`), `wildcard_text` (`STRING`), `populated_text` (`STRING`), `mode` (`populate`, `fixed`, `reproduce`), `select_to_add_lora` (appends `<lora:name:1.0:1.0>` into `wildcard_text`), `select_to_add_wildcard`, `seed` (`INT`).
- **Outputs**: `model` (`MODEL`), `clip` (`CLIP`), `conditioning` (`CONDITIONING`), `populated_text` (`STRING`).

---

### 5. Ziggart Canonical Theme & Character-Outfit Resolver

*Class: `ZiggartCanonicalThemeResolver` | Category: `Ziggart/Wildcards`*

Reads `Character-Outfits.json` inside `wildcards/Collection/Themes/<Theme>/` on-demand so characters are **always paired with their authentic `canonical_outfits`** instead of mismatched random outfit rolls, alongside `PoseAction.txt`, `Environment.txt`, `Vehicle.txt`, `Creature.txt`, and `Style.txt` from the same theme.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `theme_selection` | Dropdown | Choose any of the 1,500+ themes or `[Random Theme]`. |
| `cast_mode` | Dropdown | `Single (Any Gender)`, `Single (Female)`, `Single (Male)`, `Duo (Any 2 Characters)`, `Duo (Female + Male)`, `Duo (Female + Female)`, or `Trio (3 Characters)`. |
| `include_pose_action` / `include_environment` / `include_vehicle` / `include_creature` / `include_style` | `BOOLEAN` | Toggle which theme pools to include in `scene_prompt`. |
| `seed` | `INT` | Deterministic seed for theme, character, and outfit selection. |
| `theme_search_filter` *(optional)* | `STRING` | Keyword filter when `theme_selection` is `[Random Theme]` (e.g. `"albator"`, `"bebop"`, `"cyberpunk"`). |

#### Outputs

- `scene_prompt` (`STRING`): Full canonical scene prompt.
- `character_1_prompt`, `character_2_prompt`, `character_3_prompt` (`STRING`): Individual canonical character + outfit descriptions—ready to wire directly into **Multi-Character Grounded Krea2 Detailer**!
- `theme_metadata_json` (`STRING`): JSON breakdown of the resolved theme, cast, and components.

---

### 6. Wildcard-to-VLM Prompt Harmonizer

*Class: `ZiggartWildcardVLMEnhancer` | Category: `Ziggart/Wildcards`*

Bridges your Wildcard Processor and your local **GGUF LLM/VLM**. Stitched wildcard snippets from 6+ `.txt` pools can sometimes contain duplicate tags or conflicting lighting. This node uses your local GGUF model to rewrite the expanded wildcard prompt into fluent, cohesive prose optimized for **Krea 2 Turbo / Flux** while strictly preserving character names, physical traits, and canonical outfits.

- **Inputs**: `wildcard_or_populated_text` (`STRING`), `model` / `mmproj` (GGUF dropdowns), `harmonization_style` (`Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)`, `Cinematic Director Shot (Rich Lighting & Camera)`, `Concise High-Impact Tags & Prose Hybrid`, `Custom Instruction`), `seed`, `free_vram_after`, and optional `custom_instruction`, `reference_image` (`IMAGE`), `vlm_config` (`GGUF_VL_MODEL`).
- **Outputs**: `harmonized_prompt` (`STRING`), `raw_populated_text` (`STRING`).

---

### 7. Ziggart Prompt Recipe & Comparison Saver

*Class: `ZiggartWildcardRecipeSaver` | Category: `Ziggart/Wildcards`*

Saves generated images (and an optional side-by-side `Base vs Refined` comparison grid when `base_image` is connected and `build_comparison_grid = True`) with embedded PNG metadata (`ziggart_wildcard_template`, `ziggart_populated_text`, `ziggart_seed`, `ziggart_notes`) and an optional `.json` sidecar file so you never lose the exact wildcard recipe behind a great generation.

- **Inputs**: `image` (`IMAGE`), `filename_prefix` (`STRING`), `wildcard_template` (`STRING`), `populated_text` (`STRING`), `seed` (`INT`), `save_json_sidecar` (`BOOLEAN`), `build_comparison_grid` (`BOOLEAN`), and optional `base_image` (`IMAGE`), `detailer_or_vlm_notes` (`STRING`), `output_dir_override` (`STRING`).
- **Outputs**: `output_image_or_grid` (`IMAGE`), `recipe_summary` (`STRING`), `saved_paths_json` (`STRING`).

---

## Detailer, Grounding & VLM Suite

### 8. Detailers Krea2

*Class: `ZiggartDetailerKrea2` | Category: `Ziggart/Detailer`*

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

### 9. Detailers Krea2 (VLM Auto-Prompt)

*Class: `ZiggartAutoPromptDetailer` | Category: `Ziggart/Detailer`*

Solves multi-instance detailing where a single static prompt does not fit every detected object. While the VLM is loaded for detection, it inspects each padded crop individually and writes a custom descriptive micro-prompt before unloading once and handing off to Krea 2 Turbo.

- **Key Additional Inputs**:
  - `prompt_instruction` (`STRING`): Instructions telling the VLM how to describe each cropped region.
  - `base_positive_suffix` (`STRING`): Quality tags appended to every VLM-generated crop prompt (default: `"ultra-detailed, sharp focus, high resolution, 8k"`).
- **Additional Output**: `generated_prompts` (`STRING`) — JSON log showing the exact prompt generated for each `frame_index` and `detection_id`.

---

### 10. Multi-Character Grounded Krea2 Detailer

*Class: `ZiggartMultiSubjectDetailer` | Category: `Ziggart/Detailer`*

Purpose-built for `duo_all`, `trio_all`, `romantic_all`, and **Ziggart Canonical Theme Resolver** scenes. Locates up to 3 distinct targets with WeVisOne-VL in Phase 1, unloads the VLM once, and refines each subject crop with Krea 2 Turbo using its own dedicated character prompt in Phase 2.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `image` / `model` / `clip` / `vae` | `IMAGE` / `MODEL` / `CLIP` / `VAE` | Input image batch and Krea 2 Turbo pipeline. |
| `detector_model` / `detector_mmproj` | Dropdown | WeVisOne-VL GGUF model and projector. |
| `subject_1_target` / `subject_1_detail_prompt` | `STRING` | Target label (e.g. `"female character"`) and dedicated positive prompt for Subject 1 (wire from `character_1_prompt`). |
| `subject_2_target` / `subject_2_detail_prompt` | `STRING` | Target label (e.g. `"male character"`) and dedicated positive prompt for Subject 2 (leave prompt blank to skip). |
| `subject_3_target` / `subject_3_detail_prompt` | `STRING` | Target label (e.g. `"creature or companion"`) and dedicated positive prompt for Subject 3 (leave prompt blank to skip). |
| `detail_negative_prompt` | `STRING` | Shared negative prompt for crop refinement. |
| `crop_factor` / `target_detail_resolution` | `FLOAT` / `INT` | Context padding (`1.5`) and crop upscale resolution (`768`). |
| `steps` / `cfg` / `denoise` / `feather_mask` | `INT` / `FLOAT` / `FLOAT` / `INT` | Sampling and blending settings (`8` steps, `1.5` CFG, `0.40` denoise, `16`px feather). |

#### Outputs

- `image` (`IMAGE`): Final composited image with all detected subjects refined using their respective prompts.
- `cropped_refined` (`IMAGE`): Batch of refined subject crops.
- `detection_preview` (`IMAGE`): Preview image showing bounding boxes for all detected subjects.
- `composite_mask` (`MASK`): Full-size composite mask of all refined regions.
- `subjects_log_json` (`STRING`): JSON log of detected subjects, boxes, and applied prompts.

---

### 11. Krea2 VLM Smart Tile Upscaler

*Class: `ZiggartSmartTileUpscaler` | Category: `Ziggart/Detailer`*

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

### 12. WeVisOne-VL Detector

*Class: `ZiggartWeVisOneDetector` | Category: `Ziggart/Detector`*

Standalone visual grounding node using WeVisOne-VL. Extracts bounding boxes, segmentation masks, cropped patches, and JSON coordinates without running diffusion.

- **Inputs**: `image`, `detector_model`, `detector_mmproj`, `detection_prompt`, `crop_factor`, `max_detections`, `feather_mask`, `free_vram_after`, and optional `vlm_config`.
- **Outputs**: `detection_preview` (`IMAGE`), `mask` (`MASK`), `cropped_images` (`IMAGE`), `bboxes_json` (`STRING`).

---

### 13. BBox Crop & BBox Feathered Stitch

*Classes: `ZiggartCropByBBox` & `ZiggartStitchByBBox` | Category: `Ziggart/Detector`*

Decouples Ziggart's mod-8 padded cropping and Gaussian feathered compositing so you can route detected regions through **any** external ComfyUI nodes (e.g. ControlNet, IPAdapter, Flux Redux, custom KSamplers, face swappers).

1. **BBox Crop (`ZiggartCropByBBox`)**:
   - **Inputs**: `image`, `bboxes_json` (from **WeVisOne-VL Detector**, **GGUF Quality & Anatomy Critic**, or **GGUF Mask-Aware Inpaint Prompter**), `target_resolution` (`768`), `crop_factor_override` (`0.0` = use JSON padding), `skip_upscale_if_larger`, `feather_mask`.
   - **Outputs**: `cropped_images` (`IMAGE`), `crop_masks` (`MASK`), `stitch_pipe` (`ZIGGART_STITCH_PIPE`).
2. **BBox Feathered Stitch (`ZiggartStitchByBBox`)**:
   - **Inputs**: `stitch_pipe` (`ZIGGART_STITCH_PIPE`), `refined_crops` (`IMAGE`), `feather_mask` (`16`), `blend_opacity` (`0.0–1.0`).
   - **Outputs**: `image` (`IMAGE`), `composite_mask` (`MASK`).

---

### 14. GGUF Vision Prompt & GGUF VLM Loader

*Classes: `ZiggartGGUFVisionPrompt` & `ZiggartGGUFVLMLoader` | Category: `Ziggart/LLM`*

- **GGUF Vision Prompt (`ZiggartGGUFVisionPrompt`)**: Sends images (`image`, optional `detail_crop`, optional `reference_image`) to a local vision-capable GGUF model (e.g. Qwen2.5-VL, Qwen3-VL) with custom system/user instructions and independent **Composition**, **Subject**, and **Style** specialist passes.
- **GGUF VLM Loader (`ZiggartGGUFVLMLoader`)**: Outputs a `GGUF_VL_MODEL` (`vlm_config`) pin that can be wired into any Ziggart VLM or Detailer node so you only configure model paths, `context_size`, `gpu_layers`, and `flash_attention` in one place.

---

### 15. GGUF Quality & Anatomy Critic

*Class: `ZiggartVLMImageCritic` | Category: `Ziggart/LLM`*

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

### 16. GGUF Mask-Aware Inpaint Prompter

*Class: `ZiggartVLMInpaintPrompter` | Category: `Ziggart/LLM`*

Takes an `IMAGE`, a `MASK` (drawn in ComfyUI's MaskEditor or from a detector), and a short `edit_instruction` (e.g. *"add vintage gold aviator sunglasses"*). It highlights the masked target on the full scene and sends both the global view and close-up crop to the VLM so the generated prompt matches surrounding lighting, perspective, and style.

- **Outputs**: `inpaint_positive_prompt` (`STRING`), `inpaint_negative_prompt` (`STRING`), `annotated_preview` (`IMAGE`), and `bboxes_json` (`STRING`).

---

## Architecture & Workflows

### Workflow A: End-to-End Canonical Duo Scene + Multi-Character Detailing

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

### Workflow B: Self-Healing Generation with Quality Critic + BBox Crop/Stitch

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

### Workflow C: Two-Phase Detailers Krea2 (Standard & VLM Auto-Prompt)

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
- **Vision-Language Models (for GGUF Vision Prompt, Critic, Harmonizer, Auto-Prompt & Tile Upscaler)**:
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

- **Added new `.yaml` or `.txt` wildcard files on disk**: Toggle `refresh_index: True` for one run on **Ziggart Fast Wildcard Processor** (or restart ComfyUI; the byte-offset cache automatically detects modified file size/`mtime`).
- **`No detector_model was found`**: Ensure your `.gguf` files are located in `ComfyUI/models/LLM/GGUF/` and click **Refresh** in ComfyUI.
- **`MTMDChatHandler cannot be imported`**: Update `llama-cpp-python` to `>= 0.3.35`.
- **Object not detected**: Adjust `detection_prompt` (e.g. use specific terms like `"human eyes"`, `"face"`, or `"iris"`). Check `detection_preview` output to inspect model predictions.
- **Out of Memory during sampling**: Ensure `free_vram_after` is set to `True`, disable `encode_final_latent` on 4K images, or reduce `target_detail_resolution` from `768` to `512`.

---

## License

This project is licensed under the Apache 2.0 License. Model weights are subject to their respective creators' licenses (Tencent WeVisOne-VL, Krea 2 Community License, Qwen License).
