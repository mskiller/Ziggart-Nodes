# Ziggart Nodes for ComfyUI (`v1.4.0`)

A complete collection of **29 local, in-process ComfyUI nodes** combining:
- **Zero-Freeze 700+ MB Wildcard & Canonical Theme Engines** (Way A + Way C architectures)
- **Krea2 (`Qwen3VL-4B` CLIP) Prompt Director Suite** (Modular & All-in-One Theme/Universe context, Head-to-Toe 28-Section Anatomy Character Creator with per-section `[None]` & View Presets, 1:1 Character-Outfit linking, and automatic 9-zone Outfit $\rightarrow$ Anatomy occlusion filtering)
- **GGUF Vision-Language Models (VLMs)** via `llama-cpp-python`
- **WeVisOne-VL Open-Vocabulary Visual Grounding**
- **Krea 2 Turbo** High-Resolution Detailing & Tiled Upscaling
- **RTX 3070 Ti (8GB–16GB VRAM) 3D Generation Suite**: **`speridlabs/idol` Image-to-Gaussian Splat (`.ply`)**, **Gaussian-to-Mesh (`.glb`/`.obj`)**, **SPAR3D**, **Hunyuan3D-2mini**, and **360° Turntable Rendering**

---

## Included Nodes (29 Nodes)

| Display Name | Internal Class | Category | Purpose |
| :--- | :--- | :--- | :--- |
| **Ziggart Fast Wildcard Processor** | `ZiggartWildcardProcessor` | `Ziggart/Wildcards` | **(Way C1)** Ultra-fast drop-in replacement for `ImpactWildcardProcessor` with byte-offset YAML indexing, $O(1)$ `.txt` lookup, and a working `select_to_add_wildcard` dropdown. |
| **Ziggart Wildcard Processor (All-in-One Pillars & Mixed)** | `ZiggartWildcardProcessorAllInOne` | `Ziggart/Wildcards` | **(Way A)** All-in-one wildcard processor with 6 dedicated `Collection/` Pillar & Mixed dropdowns + `wildcard_text`. |
| **Ziggart Collection Scene Builder** | `ZiggartCollectionSceneBuilder` | `Ziggart/Wildcards` | **(Way C2)** Modular visual scene composer organized around `Collection/` folders (`Macro/YAML/Themes`, `Character/Anatomy`, `Outfit/Pose`, `Environment/Vehicle`, `Style/Camera/Lighting`, and `Mixed`). |
| **Ziggart Fast Wildcard Encode (LoRA)** | `ZiggartWildcardEncode` | `Ziggart/Wildcards` | Expands wildcards, extracts & loads `<lora:name:model_w:clip_w>` tags, and encodes `BREAK`-separated segments into `CONDITIONING`. |
| **Ziggart Canonical Theme & Character-Outfit Resolver** | `ZiggartCanonicalThemeResolver` | `Ziggart/Wildcards` | Reads `Character-Outfits.json` inside `Collection/Themes/*/` to pair characters (`Single`, `Duo`, `Trio`) with their authentic canonical outfits and matching theme environment/style. |
| **Wildcard-to-VLM Prompt Harmonizer** | `ZiggartWildcardVLMEnhancer` | `Ziggart/Wildcards` | Uses your local GGUF LLM/VLM to rewrite stitched wildcard fragments into cohesive natural-language prose for **Krea 2 Turbo / Flux** while keeping character/outfit locks intact. |
| **Ziggart Prompt Recipe & Comparison Saver** | `ZiggartWildcardRecipeSaver` | `Ziggart/Wildcards` | Saves images (or side-by-side `Base vs Refined` comparison grids) with embedded PNG metadata (`wildcard_template`, `populated_text`, `seed`) and a `.json` recipe sidecar. |
| **Ziggart Krea2 Theme & Universe Context** | `ZiggartThemeUniverseContext` | `Ziggart/Krea2 Prompt Director` | Selects a Theme (`Collection/Themes/*`), Universe (`Collection/Universes/*` with `source.json` branches), or `Free Crossover` context + `mood_filter` and full-prompt exclusion rules. |
| **Ziggart Krea2 Character Unit Builder (Outfit & Anatomy Filter)** | `ZiggartCharacterUnitBuilder` | `Ziggart/Krea2 Prompt Director` | Builds a Primary or Secondary Character slot with 1:1 `Canonical` / `Compatible` / `Free` / `Nude` outfit linking and automatic 9-zone Outfit $\rightarrow$ Anatomy occlusion filtering. |
| **Ziggart Krea2 Head-to-Toe Anatomy Character Creator** | `ZiggartAnatomyCharacterCreator` | `Ziggart/Krea2 Prompt Director` | Builds a character from Head to Toe across **28 anatomical sections** (`Collection/Anatomy/{Female,Male,Creature}`) where any section can be set to `[None]` or controlled via View Presets (`Front`, `Back View`, `Over-the-Shoulder`, `Portrait`, `Face Only`, `Lower Body`). |
| **Ziggart Krea2 Scene Stage (Style, Action, Env, Light, Camera)** | `ZiggartSceneStageAtmosphere` | `Ziggart/Krea2 Prompt Director` | Selects and resolves pure `(style)`, `(action / duo interaction)`, `(environment)` (including `ziggart/Env/*`), `(lighting)`, and `(camera)` components across `Collection/`, `ziggart/`, `jff/`, and `Encours/`. |
| **Ziggart Krea2 Prompt Assembler (Qwen3VL-4B)** | `ZiggartKrea2PromptAssembler` | `Ziggart/Krea2 Prompt Director` | Compiles `char_1_slot`, optional `char_2_slot`, and `stage_data` into **Krea2 (`Qwen3VL-4B`)** prompts (`scoped`, `balanced`, `compact`, `rich`) with token estimation and detailer outputs. |
| **Ziggart Krea2 Master Prompt Director (All-in-One)** | `ZiggartKrea2MasterPromptDirector` | `Ziggart/Krea2 Prompt Director` | Combines Theme/Universe selection, Solo/Duo Character & Outfit Linker, automatic Anatomy Occlusion filtering, Scene Stage, and Krea2 compilation into a single all-in-one node. |
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
| **Idol Image to 3D Gaussian Splat (PLY)** | `ZiggartIdolImageToGaussian` | `Ziggart/3D` | Generates 3D Gaussian Splats (`.ply`, up to 1,048,576 Gaussians) from a single image using **`speridlabs/idol`** (`~7.2 GB` peak VRAM, auto-evicts ComfyUI VRAM). |
| **Convert Idol Gaussian Splat to 3D Mesh (GLB/OBJ)** | `ZiggartGaussianToMesh` | `Ziggart/3D` | Converts `ZIGGART_GSPLAT` or `.ply` Gaussian Splats into watertight/smoothed polygon 3D meshes (`.glb`, `.obj`, `.ply`) with vertex colors and normals. |
| **SPAR3D Image to 3D Mesh (Fast 7GB VRAM)** | `ZiggartSPAR3DImageToMesh` | `Ziggart/3D` | Generates UV-unwrapped textured 3D meshes (`.glb`) + intermediate point clouds (`.ply`) in ~1–2s using Stability AI's **SPAR3D** with `low_vram_mode=True` (`~7.0 GB` VRAM). |
| **Hunyuan3D-2mini Image to 3D Mesh (GLB)** | `ZiggartHunyuan3DImageToMesh` | `Ziggart/3D` | Generates high-sharpness 3D meshes (`.glb`) using **Hunyuan3D-2mini (0.6B, ~5.5 GB VRAM)** and sequentially unloads Shape before running optional PBR Paint. |
| **3D Gaussian / Mesh Turntable Renderer** | `Ziggart3DTurntableRenderer` | `Ziggart/3D` | Renders 360° turntable `IMAGE` and `MASK` batches from any `ZIGGART_GSPLAT` or `ZIGGART_MESH`. |

---

## Table of Contents

1. [Key Features & 700+ MB Wildcard Benchmarks](#key-features--700-mb-wildcard-benchmarks)
2. [Where to Place Your Wildcards (`Ziggart-Nodes/wildcards`)](#where-to-place-your-wildcards-ziggart-nodeswildcards)
3. [Wildcard & Canonical Theme Suite (`Ziggart/Wildcards`)](#wildcard--canonical-theme-suite-ziggartwildcards)
   - [1. Ziggart Fast Wildcard Processor (Way C1)](#1-ziggart-fast-wildcard-processor-way-c1)
   - [2. Ziggart Wildcard Processor — All-in-One Pillars & Mixed (Way A)](#2-ziggart-wildcard-processor--all-in-one-pillars--mixed-way-a)
   - [3. Ziggart Collection Scene Builder (Way C2)](#3-ziggart-collection-scene-builder-way-c2)
   - [4. Ziggart Fast Wildcard Encode (LoRA)](#4-ziggart-fast-wildcard-encode-lora)
   - [5. Ziggart Canonical Theme & Character-Outfit Resolver](#5-ziggart-canonical-theme--character-outfit-resolver)
   - [6. Wildcard-to-VLM Prompt Harmonizer](#6-wildcard-to-vlm-prompt-harmonizer)
   - [7. Ziggart Prompt Recipe & Comparison Saver](#7-ziggart-prompt-recipe--comparison-saver)
4. [Krea2 (`Qwen3VL-4B`) Prompt Director Suite (`Ziggart/Krea2 Prompt Director`)](#krea2-qwen3vl-4b-prompt-director-suite-ziggartkrea2-prompt-director)
   - [8. Ziggart Krea2 Theme & Universe Context](#8-ziggart-krea2-theme--universe-context)
   - [9. Ziggart Krea2 Character Unit Builder](#9-ziggart-krea2-character-unit-builder)
   - [9B. Ziggart Krea2 Head-to-Toe Anatomy Character Creator](#9b-ziggart-krea2-head-to-toe-anatomy-character-creator)
   - [10. Ziggart Krea2 Scene Stage (Style, Action, Env, Light, Camera)](#10-ziggart-krea2-scene-stage-style-action-env-light-camera)
   - [11. Ziggart Krea2 Prompt Assembler (`Qwen3VL-4B`)](#11-ziggart-krea2-prompt-assembler-qwen3vl-4b)
   - [12. Ziggart Krea2 Master Prompt Director (All-in-One)](#12-ziggart-krea2-master-prompt-director-all-in-one)
5. [Detailer, Grounding & VLM Suite](#detailer-grounding--vlm-suite)
   - [13. Detailers Krea2](#13-detailers-krea2)
   - [14. Detailers Krea2 (VLM Auto-Prompt)](#14-detailers-krea2-vlm-auto-prompt)
   - [15. Multi-Character Grounded Krea2 Detailer](#15-multi-character-grounded-krea2-detailer)
   - [16. Krea2 VLM Smart Tile Upscaler](#16-krea2-vlm-smart-tile-upscaler)
   - [17. WeVisOne-VL Detector](#17-wevisone-vl-detector)
   - [18. BBox Crop & BBox Feathered Stitch](#18-bbox-crop--bbox-feathered-stitch)
   - [19. GGUF Vision Prompt & GGUF VLM Loader](#19-gguf-vision-prompt--gguf-vlm-loader)
   - [20. GGUF Quality & Anatomy Critic](#20-gguf-quality--anatomy-critic)
   - [21. GGUF Mask-Aware Inpaint Prompter](#21-gguf-mask-aware-inpaint-prompter)
6. [3D Generation Suite (`Ziggart/3D` — RTX 3070 Ti 8GB–16GB VRAM Optimized)](#3d-generation-suite-ziggart3d--rtx-3070-ti-8gb16gb-vram-optimized)
   - [22. Idol Image to 3D Gaussian Splat (PLY)](#22-idol-image-to-3d-gaussian-splat-ply)
   - [23. Convert Idol Gaussian Splat to 3D Mesh (GLB/OBJ)](#23-convert-idol-gaussian-splat-to-3d-mesh-glbobj)
   - [24. SPAR3D Image to 3D Mesh (Fast 7GB VRAM)](#24-spar3d-image-to-3d-mesh-fast-7gb-vram)
   - [25. Hunyuan3D-2mini Image to 3D Mesh (GLB)](#25-hunyuan3d-2mini-image-to-3d-mesh-glb)
   - [26. 3D Gaussian / Mesh Turntable Renderer](#26-3d-gaussian--mesh-turntable-renderer)
7. [Architecture & Workflows (`example_workflows/`)](#architecture--workflows)
8. [Installation](#installation)
9. [Models Download & Setup](#models-download--setup)
10. [Hardware & VRAM Optimization](#hardware--vram-optimization)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Key Features & 700+ MB Wildcard Benchmarks

- **Zero-Freeze 700+ MB Wildcard Engine**:
  - **Streaming Byte-Offset YAML Indexer**: Indexes 50+ MB / 150,000+ line YAML files (like `Ziggart.yaml`) at **any nesting depth** (3, 4, 5+ levels) in `~195 ms` cold and **`36 ms` cached** on ComfyUI startup. Resolves keys by `seek()`ing directly to byte offsets (`~13 ms`) without loading 50 MB into RAM.
  - **$O(1)$ Direct `.txt` Lookup**: Resolves 77,000+ `.txt` files across `Collection/Themes`, `Collection/Universes`, `Collection/Combos`, and `Collection/formats-wild` in `0.05 ms` without walking 78,000 files on ComfyUI boot.
  - **Working Categorized Dropdowns**: Organizes wildcards into browser-friendly `Collection/` Pillar dropdowns + pure facet selectors so ComfyUI never chokes on an 80,000-item combo list.
- **Krea2 (`Qwen3VL-4B`) Prompt Compiler & Smart Filtering**:
  - **1:1 Character-Outfit Linker**: Supports `Canonical (1:1 Strict Character Outfit)` from `Character-Outfits.json` and `Universes/*/source.json`, `Compatible (Same Theme/Universe + Gender)`, `Free (All Outfit Wildcards across Folders)`, and `Nude / Unclothed`.
  - **Automatic 5-Zone Outfit $\rightarrow$ Anatomy Occlusion Filter**: Inspects the resolved outfit description and automatically removes anatomical wildcard descriptions that are covered by clothing or armor while preserving visible, sheer, torn, or cutout zones.
  - **Strict Full-Prompt Exclusion**: Automatically excludes pre-assembled full-scene prompts (`*-scene`, `*-prompts`, `Themes/*/ALL`, `Combos/*`, `Formats/*`) from component selectors so only pure modular building blocks are selected.
- **Zero External Servers**: Runs GGUF VLMs directly inside the ComfyUI Python process via `llama-cpp-python`. No Ollama, Docker, or standalone `llama-server` needed.
- **Two-Phase Batch Execution**: Across multi-image batches and multi-crop passes, Ziggart runs all VLM grounding/captioning operations first, unloads the VLM from VRAM **once**, caches static CLIP conditioning, and then runs all diffusion sampling passes—eliminating model-swap thrashing.
- **Latent Noise Masking + Pixel Feathering**: Attaches feathered noise masks directly to crop latents during KSampler steps and blends borders with Gaussian alpha feathering, eliminating seams and color shifts even at higher denoise values.
- **Smart `<think>` Tag Handling**: Automatically strips `<think>...</think>` reasoning blocks from thinking VLMs so CLIP prompts and bounding-box parsers stay clean.

### Benchmark vs. Impact Pack (77,146 `.txt` + 1,550 `.json` + 53.5 MB `Ziggart.yaml`)

| Operation | Impact Pack Behavior | Ziggart Wildcard Index (`v1.4.0`) |
| :--- | :--- | :--- |
| **Cold Startup / Refresh** | Minutes (`os.walk` ×2 on 78,697 files + `yaml.FullLoader`) | **195.6 ms** (streaming byte-offset YAML scan + top-level pillar scan) |
| **Warm Startup / Refresh** | Minutes | **36.5 ms** (loads `.cache/wildcard_yaml_offsets.json` verified via `mtime` + `size`) |
| **3+ Level `Ziggart.yaml` Keys** | **0 keys loaded** (silently drops `Ziggart/scenes/all` & all 11,619 nested keys in on-demand mode) | **11,619 YAML keys indexed** across arbitrary `1..N` nesting depths |
| **Dropdown Population** | Empty / broken over 50 MB | **Categorized Pillar, Facet & Mixed Dropdowns** (1,500+ Themes, Universes, 3,286 YAML Macro/Scene keys, + Collection Pillars) |
| **Recursive Resolution (`__Ziggart/scenes/all__`)** | Fails (unresolved token left in prompt) | **13.07 ms** ($O(1)$ byte-offset `seek()` + direct `.txt` path resolution) |

---

## Where to Place Your Wildcards (`Ziggart-Nodes/wildcards`)

To prevent **ComfyUI-Impact-Pack** from scanning 700+ MB / 78,000+ wildcard files on every ComfyUI startup or browser refresh (which causes multi-minute freezes and empty dropdowns in Impact Pack), **move your wildcard files out of `ComfyUI-Impact-Pack/wildcards` and place them directly inside `Ziggart-Nodes/wildcards/`**:

```text
ComfyUI/
└── custom_nodes/
    └── Ziggart-Nodes/
        └── wildcards/                  <-- PUT ALL YOUR WILDCARDS HERE (Primary Location)
            ├── Ziggart.yaml            (50+ MB multi-level YAML wildcards)
            ├── Collection/             (Structured Collection Pillars, Themes & Universes)
            │   ├── Themes/             (01-MysterieusesCitesDOr, 13-Albator, etc. + Character-Outfits.json)
            │   ├── Universes/          (Multi-branch Universes with source.json)
            │   ├── Characters/
            │   ├── Anatomy/
            │   ├── Outfits/
            │   ├── Pose/
            │   ├── Expression/
            │   ├── Environment/
            │   ├── IconicPlaces/
            │   ├── Vehicle/
            │   ├── Style/
            │   ├── Medium/
            │   ├── Lighting/
            │   ├── Camera/
            │   ├── VFX/
            │   ├── Sheet/
            │   └── Combos/
            ├── ziggart/                (Subfolder .txt pools: Style, Outfits, Location, etc.)
            ├── jff/                    (Additional component wildcards: styles, clothes, poses, places)
            ├── Encours/                (In-progress component wildcards)
            └── *.txt                   (Root .txt wildcards: female-all.txt, style-full-v2.txt, etc.)
```

### Why `Ziggart-Nodes/wildcards/` is Best
1. **Eliminates Impact Pack Startup Lag**: Once `ComfyUI-Impact-Pack/wildcards` is empty, Impact Pack no longer walks 78,000 files or runs `yaml.FullLoader` on `Ziggart.yaml` when ComfyUI boots or refreshes.
2. **Highest Discovery Priority (#1)**: `ZiggartWildcardIndex` checks `ComfyUI/custom_nodes/Ziggart-Nodes/wildcards` **first** automatically—no extra path configuration is required.
3. **Git-Safe Out of the Box**: The `Ziggart-Nodes/wildcards/` folder is pre-created in this repository and configured in `.gitignore` (`wildcards/*`, `!wildcards/.gitkeep`) so `git pull` / `git status` inside `Ziggart-Nodes` will never touch, upload, or slow down on your 700+ MB wildcard library.

### Automatic Discovery Order
If you keep wildcards in multiple places, `Ziggart-Nodes` scans directories in this priority order:
1. **`ComfyUI/custom_nodes/Ziggart-Nodes/wildcards`** *(Recommended primary folder)*
2. `custom_wildcards_dir` *(Optional widget override on the node)*
3. `ComfyUI/wildcards`
4. `ComfyUI/custom_nodes/ComfyUI-Impact-Pack/wildcards` *(Fallback if any files remain there)*

---

## Wildcard & Canonical Theme Suite (`Ziggart/Wildcards`)

Automatically discovers wildcards in `Ziggart-Nodes/wildcards` (primary), `ComfyUI/wildcards`, `ComfyUI-Impact-Pack/wildcards`, or any custom path—without modifying external folders.

### Full Wildcard Syntax Support
- **Arbitrary-Depth YAML & TXT Tokens**: `__Ziggart/scenes/all__`, `__Ziggart/scenes/01_mysterieusescitesdor/duo/romantic__`, `__Collection/Themes/01-MysterieusesCitesDOr/Character__`, `__female-full__`, multi-root YAML files (`__krea2-grok/wallpaper__`, `__krea2/all__`), or parent rollups like `__Ziggart/scenes__`.
- **Glob `*` Wildcard Patterns**: `__Collection/formats-wild/Formats/luisroyo/Templates/Female-*__`, `__jff/animals/*__`, or `__Collection/formats-wild/Formats/*/Templates/*__` resolved in $O(1)$ via cached filesystem & YAML glob matching.
- **Weighted Dynamic Choices**: `{option A|option B|2.5::weighted option C}` (supports nested braces `{a|{b|c}}`).
- **Multi-Select Ranges**: `{2$$, $$__wildcard__}` or `{1-3$$ and $$opt1|opt2|opt3}`.
- **Quantifiers**: `3#__wildcard__`.
- **LoRA Tags**: `<lora:lora_name:1.0:0.8>`.
- **Live `populated_text` UI Feedback**: Selecting a wildcard from the dropdown or clicking **Queue** automatically populates the `populated_text` widget in the ComfyUI frontend via WebSocket feedback.

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

## Krea2 (`Qwen3VL-4B`) Prompt Director Suite (`Ziggart/Krea2 Prompt Director`)

The 6 **Krea2 Prompt Director** nodes build structured Solo or Duo prompts directly from your component wildcards (`Collection/Themes`, `Collection/Universes`, `Collection/Anatomy`, `Collection/`, `ziggart/`, `ziggart/Env/`, `jff/`, `Encours/`, `Outfits/`, `Characters/`) while **strictly excluding pre-assembled full-scene prompts** (`*-scene.txt`, `*-prompts.txt`, `Themes/*/ALL.txt`, `Universes/*/ALL.txt`, `Combos/*`, `Formats/*`).

### Target Prompt Structure for Krea2 (`Qwen3VL-4B` CLIP)

- **Solo Mode (1 Character)**:
  > `1 woman, (Render Style: <style>) (<char_1>) [(<char_1_anatomy>)] (wearing <char_1_outfit>) (<char_1_action>) (<environment>) (<lighting>) (<camera>).`
- **Duo / Interaction Mode (2 Characters)**:
  > `2 characters, (Render Style: <style>) (<char_1>) [(<char_1_anatomy>)] (wearing <char_1_outfit>) (<char_2>) [(<char_2_anatomy>)] (wearing <char_2_outfit>) (<shared_action>) (<environment>) (<lighting>) (<camera>).`

### Automatic 9-Zone Outfit $\rightarrow$ Anatomy Occlusion Filter

When `auto_filter_anatomy_by_outfit = True` (default), `OutfitOcclusionAnalyzer` inspects the resolved outfit text and only keeps anatomical descriptions for body zones that are actually visible:

| Anatomical Zone | Associated Anatomy Wildcards | Covered / Blocked By | Re-Allowed By (Transparency / Cutout / Open Modifiers) |
| :--- | :--- | :--- | :--- |
| **`BREASTS_CHEST`** *(Bust / Cleavage / Chest)* | `Breasts-Chest.txt`, `Chest-Torso.txt`, `Torso-Carapace.txt`, `jff/breast-shape.txt` | Heavy/full upper coverage (`full plate armor`, `cuirass`, `breastplate`, `heavy coat`, `cloak`, `parka`, `turtleneck`, `sweater`, `hoodie`, `spacesuit`) | `cleavage`, `plunging`, `low-cut`, `open-chest`, `bikini`, `bra`, `corset`, `bustier`, `sheer`, `see-through`, `translucent`, `lace`, `wet`, `torn`, `skin-tight`, `latex` |
| **`NIPPLES_AREOLAS`** *(Nipples / Areolas)* | `Nipples-Areolas.txt` | Any upper garment (`shirt`, `tunic`, `dress`, `gown`, `armor`, `bikini`, `swimsuit`, `bra`, `bodysuit`, `corset`, `jacket`, `top`, `kimono`) | `sheer`, `see-through`, `transparent`, `translucent`, `mesh`, `fishnet`, `lace unlined`, `wet`, `clinging`, `topless`, `open-cup`, `cupless`, `pasties`, `torn`, `micro-bikini`, `nude` |
| **`WAIST_MIDRIFF`** *(Waist / Midriff / Abs)* | `Waist-Midriff.txt`, `Hips-Waist.txt`, `Abs-Core.txt` | Full-torso garments (`dress`, `gown`, `robe`, `one-piece`, `bodysuit`, `tunic`, `hauberk`, `full armor`, `jumpsuit`, `coat`, `suit`) | `crop top`, `bare midriff`, `exposed midriff`, `bikini`, `two-piece`, `loincloth`, `low-rise`, `belly dancer`, `torn`, `cutout`, `open front`, `sheer` |
| **`BACK_SPINE`** *(Back / Spine / Scapulae)* | `Back-Spine.txt`, `Dorsal-Spines-Crests.txt` | Upper garments covering the back (`cloak`, `cape`, `full armor`, `coat`, `jacket`, `sweater`, `hoodie`, `tunic`, `shirt`, `dress`, `gown`) | `backless`, `open back`, `low back`, `bare back`, `halter`, `strapless`, `tube top`, `bikini`, `topless`, `shirtless`, `sheer`, `torn`, `nude` |
| **`ARMS_SHOULDERS`** *(Shoulders / Arms)* | `Arms-Shoulders.txt`, `Wings-Appendages.txt`, `Extra-Limbs-MultiArms.txt` | Long-sleeved or armored upper wear (`full plate armor`, `pauldrons`, `vambraces`, `long sleeves`, `heavy coat`, `parka`, `hoodie`, `sweater`, `jacket`, `suit`) | `sleeveless`, `strapless`, `off-shoulder`, `bare shoulders`, `tank top`, `halter`, `bikini`, `swimsuit`, `short sleeves`, `topless`, `sheer`, `torn`, `nude` |
| **`BUTTOCKS_HIPS`** *(Buttocks / Hips / Glutes)* | `Buttocks-Glutes.txt`, `Hips-Pelvis.txt`, `Glutes-Legs.txt`, `Tails-Stingers.txt` | Heavy/loose lower garments (`long skirt`, `ballgown`, `flowing robes`, `cloak`, `baggy pants`, `plate greaves`, `trench coat`) | `thong`, `g-string`, `micro-skirt`, `miniskirt`, `high-cut`, `cheeky`, `leotard`, `leggings`, `yoga pants`, `skin-tight`, `sheer`, `torn`, `loincloth`, `bottomless` |
| **`INTIMATE_GENITALS`** *(Intimate Lower Anatomy)* | `Vulva-Intimate.txt`, `Penis-Intimate.txt` | Any lower garment (`pants`, `trousers`, `skirt`, `dress`, `shorts`, `bikini`, `panties`, `underwear`, `briefs`, `swimsuit`, `loincloth`, `armor`, `leggings`) | `nude`, `naked`, `bottomless`, `crotchless`, `open-crotch`, `pulled aside`, `unclothed` |
| **`LEGS_THIGHS`** *(Thighs / Legs / Calves)* | `Legs-Thighs.txt`, `Glutes-Legs.txt`, `Legs-Paws-Hooves.txt` | Full leg coverage (`trousers`, `pants`, `jeans`, `ballgown`, `floor-length`, `long skirt`, `maxi dress`, `greaves`, `full plate armor`, `jumpsuit`) | `shorts`, `miniskirt`, `micro-skirt`, `high slit`, `thigh-high`, `bikini`, `swimsuit`, `leotard`, `loincloth`, `leggings`, `skin-tight`, `sheer`, `torn`, `nude` |
| **`FEET_ANKLES`** *(Feet / Ankles / Toes)* | `Feet-Ankles.txt`, `Paws-Talons-Claws.txt` | Footwear covering feet/ankles (`boots`, `greaves`, `sabatons`, `shoes`, `sneakers`, `heels`, `loafers`, `socks`, `stockings`, `spacesuit`) | `barefoot`, `bare feet`, `bare ankles`, `sandals`, `strappy sandals`, `open-toe`, `anklet`, `foot jewelry`, `toeless`, `nude` |

---

### 8. Ziggart Krea2 Theme & Universe Context

*Class: `ZiggartThemeUniverseContext` | Category: `Ziggart/Krea2 Prompt Director`*

Defines the narrative context (Theme, Universe, or Multi-Folder Free Crossover) and global filtering rules shared with downstream Krea2 Character and Scene Stage nodes.

- **Inputs**:
  - `source_mode`: `Theme (Collection/Themes)`, `Universe (Collection/Universes)`, `Random Theme`, `Random Universe`, or `Free Crossover (All Wildcard Folders)`.
  - `theme_selection`: Choose any of the 1,500+ themes or `[Random Theme]`.
  - `universe_selection`: Choose any multi-branch Universe (`Collection/Universes/*` with `source.json` metadata) or `[Random Universe]`.
  - `mood_filter`: `[Any / No Filter]`, `SFW Only`, `NSFW / Sensual / Glamour`, `Fantasy / Mythology / Medieval`, `Sci-Fi / Cyberpunk / Mecha`, `Anime / Manga / Cel Animation`, `Dark / Gothic / Horror`, or `Urban / Modern / Streetwear`.
  - `exclude_full_prompts` (`BOOLEAN`, default `True`): Excludes pre-assembled full-scene prompts from component rolls.
  - `seed` (`INT`), optional `search_or_keyword_filter` (`STRING`), `custom_wildcards_dir`.
- **Outputs**: `context` (`ZIGGART_CONTEXT`), `active_theme_or_universe` (`STRING`), `default_style` (`STRING`), `default_environment` (`STRING`).

---

### 9. Ziggart Krea2 Character Unit Builder

*Class: `ZiggartCharacterUnitBuilder` | Category: `Ziggart/Krea2 Prompt Director`*

Builds a **Primary** or **Secondary** Character slot with linked Outfit selection and automatic 9-zone Outfit $\rightarrow$ Anatomy occlusion filtering.

- **Inputs**:
  - `slot_role`: `Primary` or `Secondary`.
  - `character_source`: `From Context (Theme / Universe)` or `Wildcard Selector / Category Pool`.
  - `gender_filter`: `Auto / Any`, `Female`, `Male`, `Creature`.
  - `character_wildcard`: Pure character wildcard selector (`Collection/Characters/*`, `ziggart/character/*`, `jff/actress`, `female-all`, etc.).
  - `outfit_link_mode`:
    - `Canonical (1:1 Strict Character Outfit)`: Uses the exact character's `canonical_outfits` from `Character-Outfits.json` or `source.json`.
    - `Compatible (Same Theme/Universe + Gender)`: Restricts outfits to the matching gender pool of the active Theme/Universe or gender-matched `Collection/Outfits/<Gender>/*` and `ziggart/Outfits/*`.
    - `Free (All Outfit Wildcards across Folders)`: Unlocks all pure outfit wildcards across `Collection/Outfits`, `ziggart/Outfits`, `jff/`, and `Encours/`.
    - `Nude / Unclothed`: No clothing; unlocks all 9 anatomical zones.
  - `outfit_wildcard`: Optional specific outfit wildcard when in `Compatible` or `Free` mode.
  - `enable_anatomy` (`BOOLEAN`) & `auto_filter_anatomy_by_outfit` (`BOOLEAN`, default `True`).
  - `anatomy_wildcard`: `[Random From Category]` or a specific anatomy wildcard from `Collection/Anatomy/*` / `jff/breast-shape`.
  - `individual_action_wildcard`: Optional character pose/action wildcard.
  - `seed` (`INT`), optional `context` (`ZIGGART_CONTEXT`), `custom_char_slot_override` (`ZIGGART_CHAR_SLOT`), `custom_character_text`, `custom_outfit_text`, `custom_anatomy_text`, `custom_action_text`.
- **Outputs**: `char_slot` (`ZIGGART_CHAR_SLOT`), `character_prompt_block` (`STRING` — ready for Detailer nodes), `occlusion_debug_info` (`STRING`).

---

### 9B. Ziggart Krea2 Head-to-Toe Anatomy Character Creator

*Class: `ZiggartAnatomyCharacterCreator` | Category: `Ziggart/Krea2 Prompt Director`*

Builds a complete character description **from Head to Toe** directly from the **28 anatomical sections** of `Collection/Anatomy/{Female,Male,Creature}`. Every single section can be set to **`[None]`** (for instance when a character is seen from behind and does not show their eyes, nose, lips, or front chest, or in an upper-body portrait where legs and feet are out of frame), `[Random (Gender Default)]`, a specific wildcard, or controlled automatically via **`view_visibility_preset`**:

- **View Visibility Presets (`view_visibility_preset`)**:
  1. `Full Body — Front / 3-Quarter View` *(Default: enables head-to-toe front/3-quarter sections; sets `back_spine` and `buttocks_glutes` to `[None]`)*
  2. `Back View / Seen From Behind (No Face / No Front Chest)` *(Automatically sets `forehead_brow`, `eyebrows_eyelashes`, `eyes_shape`, `eyes_iris_color`, `nose`, `cheeks_cheekbones`, `mouth_lips`, `teeth_smile`, `chin_jawline`, `chest_breasts`, `nipples_areolas`, `abs_core`, and `intimate_anatomy` to `[None]`, while enabling `hair_*`, `neck_collarbones`, `arms_shoulders`, `back_spine`, `waist_midriff`, `hips_pelvis`, `buttocks_glutes`, `legs_thighs`, and `feet_ankles`)*
  3. `Back Over-the-Shoulder (Back + Side Gaze)` *(Back view + enables `eyes_shape`, `eyes_iris_color`, `eyebrows_eyelashes`, `cheeks_cheekbones`, `mouth_lips`, `chin_jawline` for an over-the-shoulder gaze)*
  4. `Upper Body / Waist-Up Portrait` *(Enables head, hair, full face, neck, shoulders/arms/hands, chest, and waist; sets hips, glutes, legs, and feet to `[None]`)*
  5. `Close-Up Face & Hair Only` *(Enables hair, skin, and all 10 facial/eye sections; sets all torso and limb sections to `[None]`)*
  6. `Lower Body & Legs Focus` *(Enables waist, hips, glutes, legs/thighs, and feet/ankles; sets head, face, and upper body to `[None]`)*
  7. `Custom (Use Individual Section Settings)`
- **28 Head-to-Toe Sections**:
  - **Profile & Skin**: `age_group`, `build_silhouette`, `skin_complexion`, `skin_texture_finish`, `skin_marks`
  - **Hair & Head**: `hair_cut_style`, `hair_color`, `hair_texture`, `beard_facial_hair`
  - **Face & Gaze**: `forehead_brow`, `eyebrows_eyelashes`, `eyes_shape`, `eyes_iris_color`, `ears`, `nose`, `cheeks_cheekbones`, `mouth_lips`, `teeth_smile`, `chin_jawline`
  - **Neck, Arms & Hands**: `neck_collarbones`, `arms_shoulders`, `hands_fingers`
  - **Torso, Chest, Waist & Back**: `chest_breasts`, `nipples_areolas`, `waist_midriff`, `abs_core`, `back_spine`
  - **Pelvis, Glutes, Legs & Feet**: `hips_pelvis`, `buttocks_glutes`, `intimate_anatomy`, `legs_thighs`, `feet_ankles`
- **Optional Outfit Link & 9-Zone Occlusion**: Choose `[None / Pure Anatomy Only]`, `Compatible (Theme/Universe + Gender)`, `Free (Outfit Wildcard)`, or `Nude / Unclothed` with `auto_filter_anatomy_by_outfit = True`.
- **Outputs**: `char_slot` (`ZIGGART_CHAR_SLOT` — wires directly into `ZiggartKrea2PromptAssembler` or `ZiggartCharacterUnitBuilder`), `character_full_prompt` (`STRING`), `head_and_face_prompt` (`STRING` — ideal for Face/Eyes Detailers), `body_and_outfit_prompt` (`STRING`), `anatomy_breakdown_json` (`STRING`).

---

### 10. Ziggart Krea2 Scene Stage (Style, Action, Env, Light, Camera)

*Class: `ZiggartSceneStageAtmosphere` | Category: `Ziggart/Krea2 Prompt Director`*

Selects and resolves the 5 staging facets across `Collection/`, `ziggart/` (including `ziggart/Env/*` and `ziggart/Style/*`), `jff/`, and `Encours/`.

- **Inputs**:
  - `style_wildcard`: `[From Context (Theme / Universe)]`, `[Random From Category]`, or any pure style wildcard (`Collection/Style/*`, `Collection/Medium/*`, `ziggart/Style/*`, `jff/jff-styles`, `style-full-v2`, etc.).
  - `action_or_interaction_wildcard`: `[From Context (Theme / Universe)]`, `[Random From Category]`, `[Duo Bank: Action / Combat (Back-to-Back)]`, `[Duo Bank: Heroic / Iconic Partners (Camaraderie)]`, `[Duo Bank: Romantic / Sensual / Glamour]`, or any pure pose/action wildcard.
  - `environment_wildcard`: `[From Context (Theme / Universe)]`, `[Random From Category]`, or any pure environment wildcard (`Collection/Environment/*`, `Collection/IconicPlaces/*`, `ziggart/Env/*` such as `Dance-env`, `ziggart/Zig-Location`, `jff/jfflieu`, `environnement-full`, etc.).
  - `lighting_wildcard`: `[Random From Category]`, `[None]`, or any lighting wildcard (`Collection/Lighting/*`, `jff/ambiance-light`, etc.).
  - `camera_wildcard`: `[Random From Category]`, `[None]`, or any camera wildcard (`Collection/Camera/*`, `jff/camera-angle`, etc.).
  - `seed` (`INT`), optional `context` (`ZIGGART_CONTEXT`), and optional `custom_style_text`, `custom_action_text`, `custom_environment_text`, `custom_lighting_text`, `custom_camera_text` overrides.
- **Outputs**: `stage_data` (`ZIGGART_STAGE`), `style_str` (`STRING`), `action_str` (`STRING`), `environment_str` (`STRING`), `lighting_str` (`STRING`), `camera_str` (`STRING`).

---

### 11. Ziggart Krea2 Prompt Assembler (`Qwen3VL-4B`)

*Class: `ZiggartKrea2PromptAssembler` | Category: `Ziggart/Krea2 Prompt Director`*

Assembles `char_1_slot`, optional `char_2_slot`, and `stage_data` into the final **Krea2 (`Qwen3VL-4B`)** prompt. Automatically switches between the **Solo** structure (`(char 1 action)`) and **Duo** structure (`(char 2) + (char 2 anatomy) + (char 2 outfit) + (shared action)`) when `char_2_slot` is connected and `enable_secondary_character = True`.

- **Compiler Policies (`compiler_policy`)**:
  - `scoped` *(Default)*: Parenthesized semantic blocks with a leading subject count anchor (`1 woman, (Render Style: ...) (<char_1>) ...`).
  - `balanced`: 9 ordered natural-language IR slots with clause deduplication.
  - `compact`: Condenses background clauses to maximize subject token budget.
  - `rich`: Retains all atmospheric, support, and quality clauses.
- **Outputs**: `krea2_compiled_prompt` (`STRING`), `populated_text` (`STRING`), `char_1_detailer_prompt` (`STRING`), `char_2_detailer_prompt` (`STRING`), `krea2_metadata_json` (`STRING` including `qwen3vl_tokens_estimated`).

---

### 12. Ziggart Krea2 Master Prompt Director (All-in-One)

*Class: `ZiggartKrea2MasterPromptDirector` | Category: `Ziggart/Krea2 Prompt Director`*

Combines Nodes 8–11 (`ZiggartThemeUniverseContext`, `ZiggartCharacterUnitBuilder` ×2, `ZiggartSceneStageAtmosphere`, and `ZiggartKrea2PromptAssembler`) into a **single all-in-one node** with live `populated_text` feedback in ComfyUI.

- **Outputs**: `krea2_compiled_prompt` (`STRING`), `populated_text` (`STRING`), `char_1_detailer_prompt` (`STRING`), `char_2_detailer_prompt` (`STRING`), `wildcard_recipe_summary` (`STRING`), `krea2_metadata_json` (`STRING`).

---

## Detailer, Grounding & VLM Suite

### 13. Detailers Krea2

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

### 14. Detailers Krea2 (VLM Auto-Prompt)

*Class: `ZiggartAutoPromptDetailer` | Category: `Ziggart/Detailer`*

Solves multi-instance detailing where a single static prompt does not fit every detected object. While the VLM is loaded for detection, it inspects each padded crop individually and writes a custom descriptive micro-prompt before unloading once and handing off to Krea 2 Turbo.

- **Key Additional Inputs**:
  - `prompt_instruction` (`STRING`): Instructions telling the VLM how to describe each cropped region.
  - `base_positive_suffix` (`STRING`): Quality tags appended to every VLM-generated crop prompt (default: `"ultra-detailed, sharp focus, high resolution, 8k"`).
- **Additional Output**: `generated_prompts` (`STRING`) — JSON log showing the exact prompt generated for each `frame_index` and `detection_id`.

---

### 15. Multi-Character Grounded Krea2 Detailer

*Class: `ZiggartMultiSubjectDetailer` | Category: `Ziggart/Detailer`*

Purpose-built for `duo_all`, `trio_all`, `romantic_all`, **Ziggart Canonical Theme Resolver**, and **Krea2 Prompt Director** scenes. Locates up to 3 distinct targets with WeVisOne-VL in Phase 1, unloads the VLM once, and refines each subject crop with Krea 2 Turbo using its own dedicated character prompt in Phase 2.

#### Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `image` / `model` / `clip` / `vae` | `IMAGE` / `MODEL` / `CLIP` / `VAE` | Input image batch and Krea 2 Turbo pipeline. |
| `detector_model` / `detector_mmproj` | Dropdown | WeVisOne-VL GGUF model and projector. |
| `subject_1_target` / `subject_1_detail_prompt` | `STRING` | Target label (e.g. `"female character"`) and dedicated positive prompt for Subject 1 (wire from `character_1_prompt` or `char_1_detailer_prompt`). |
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

### 16. Krea2 VLM Smart Tile Upscaler

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

### 17. WeVisOne-VL Detector

*Class: `ZiggartWeVisOneDetector` | Category: `Ziggart/Detector`*

Standalone visual grounding node using WeVisOne-VL. Extracts bounding boxes, segmentation masks, cropped patches, and JSON coordinates without running diffusion.

- **Inputs**: `image`, `detector_model`, `detector_mmproj`, `detection_prompt`, `crop_factor`, `max_detections`, `feather_mask`, `free_vram_after`, and optional `vlm_config`.
- **Outputs**: `detection_preview` (`IMAGE`), `mask` (`MASK`), `cropped_images` (`IMAGE`), `bboxes_json` (`STRING`).

---

### 18. BBox Crop & BBox Feathered Stitch

*Classes: `ZiggartCropByBBox` & `ZiggartStitchByBBox` | Category: `Ziggart/Detector`*

Decouples Ziggart's mod-8 padded cropping and Gaussian feathered compositing so you can route detected regions through **any** external ComfyUI nodes (e.g. ControlNet, IPAdapter, Flux Redux, custom KSamplers, face swappers).

1. **BBox Crop (`ZiggartCropByBBox`)**:
   - **Inputs**: `image`, `bboxes_json` (from **WeVisOne-VL Detector**, **GGUF Quality & Anatomy Critic**, or **GGUF Mask-Aware Inpaint Prompter**), `target_resolution` (`768`), `crop_factor_override` (`0.0` = use JSON padding), `skip_upscale_if_larger`, `feather_mask`.
   - **Outputs**: `cropped_images` (`IMAGE`), `crop_masks` (`MASK`), `stitch_pipe` (`ZIGGART_STITCH_PIPE`).
2. **BBox Feathered Stitch (`ZiggartStitchByBBox`)**:
   - **Inputs**: `stitch_pipe` (`ZIGGART_STITCH_PIPE`), `refined_crops` (`IMAGE`), `feather_mask` (`16`), `blend_opacity` (`0.0–1.0`).
   - **Outputs**: `image` (`IMAGE`), `composite_mask` (`MASK`).

---

### 19. GGUF Vision Prompt & GGUF VLM Loader

*Classes: `ZiggartGGUFVisionPrompt` & `ZiggartGGUFVLMLoader` | Category: `Ziggart/LLM`*

- **GGUF Vision Prompt (`ZiggartGGUFVisionPrompt`)**: Sends images (`image`, optional `detail_crop`, optional `reference_image`) to a local vision-capable GGUF model (e.g. Qwen2.5-VL, Qwen3-VL) with custom system/user instructions and independent **Composition**, **Subject**, and **Style** specialist passes.
- **GGUF VLM Loader (`ZiggartGGUFVLMLoader`)**: Outputs a `GGUF_VL_MODEL` (`vlm_config`) pin that can be wired into any Ziggart VLM or Detailer node so you only configure model paths, `context_size`, `gpu_layers`, and `flash_attention` in one place.

---

### 20. GGUF Quality & Anatomy Critic

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

### 21. GGUF Mask-Aware Inpaint Prompter

*Class: `ZiggartVLMInpaintPrompter` | Category: `Ziggart/LLM`*

Takes an `IMAGE`, a `MASK` (drawn in ComfyUI's MaskEditor or from a detector), and a short `edit_instruction` (e.g. *"add vintage gold aviator sunglasses"*). It highlights the masked target on the full scene and sends both the global view and close-up crop to the VLM so the generated prompt matches surrounding lighting, perspective, and style.

- **Outputs**: `inpaint_positive_prompt` (`STRING`), `inpaint_negative_prompt` (`STRING`), `annotated_preview` (`IMAGE`), and `bboxes_json` (`STRING`).

---

## 3D Generation Suite (`Ziggart/3D` — RTX 3070 Ti 8GB–16GB VRAM Optimized)

All 5 nodes in `Ziggart/3D` share `Ziggart3DModelManager`, which automatically evicts loaded ComfyUI checkpoints/VLMs before running 3D generation (`unload_comfy_models = True`), ensures only **one** 3D model is in VRAM at a time, and **automatically downloads/installs missing 3D SDKs (`idol-sdk`, `stable-point-aware-3d`, `Hunyuan3D-2`) into `Ziggart-Nodes/.cache/vendor_3d/` on first execution**.

### 22. Idol Image to 3D Gaussian Splat (PLY)
*Class: `ZiggartIdolImageToGaussian` | Category: `Ziggart/3D`*
- Uses **`speridlabs/idol`** (`~7.2 GB` peak VRAM, <2s inference) to convert a single image (`synthetic` render or `real` photo) into up to `1,048,576` 3D Gaussians saved as a `.ply` file.
- **Hugging Face Authentication**: Because Idol uses `facebook/dinov3-vith16plus-pretrain-lvd1689m` and `black-forest-labs/FLUX.2-dev` (VAE), paste your Hugging Face token into the `hf_token` input field on the first run (or set `HF_TOKEN` in your environment) after accepting their terms on Hugging Face.
- **Outputs**: `ply_path` (`STRING`), `gsplat_data` (`ZIGGART_GSPLAT`), `preprocessed_image` (`IMAGE`), and `preview_image` (`IMAGE` 2×2 contact sheet).

### 23. Convert Idol Gaussian Splat to 3D Mesh (GLB/OBJ)
*Class: `ZiggartGaussianToMesh` | Category: `Ziggart/3D`*
- Converts `gsplat_data` or any `.ply` Gaussian Splat file into a watertight/smoothed polygon 3D mesh (`.glb`, `.obj`, or `.ply`) with baked vertex colors and normals.
- Supports `auto`, `open3d_poisson`, `open3d_ball_pivoting`, and a built-in zero-dependency `gaussian_density_surface` reconstructor with Taubin smoothing and vertex-clustering decimation.

### 24. SPAR3D Image to 3D Mesh (Fast 7GB VRAM)
*Class: `ZiggartSPAR3DImageToMesh` | Category: `Ziggart/3D`*
- Wraps Stability AI's **`stabilityai/stable-point-aware-3d`** with `low_vram_mode = True` (`SPAR3D_LOW_VRAM=1`, `~7.0 GB` VRAM) enabled by default to produce a UV-unwrapped textured `.glb` mesh and intermediate `.ply` point cloud in ~1–2 seconds.

### 25. Hunyuan3D-2mini Image to 3D Mesh (GLB)
*Class: `ZiggartHunyuan3DImageToMesh` | Category: `Ziggart/3D`*
- Runs Tencent's **`Hunyuan3D-2mini (0.6B - 6GB VRAM)`** (`octree_resolution = 256`, `num_chunks = 8000`) for ultra-sharp watertight geometry.
- **Sequential VRAM Protection**: When `generate_texture = True`, it unloads the Shape DiT model and empties CUDA cache **before** loading `Hunyuan3DPaintPipeline` with CPU offload.

### 26. 3D Gaussian / Mesh Turntable Renderer
*Class: `Ziggart3DTurntableRenderer` | Category: `Ziggart/3D`*
- Renders single or multi-frame 360° turntable `IMAGE` (`[N, H, W, 3]`) and `MASK` (`[N, H, W]`) batches from either a `ZIGGART_GSPLAT` or `ZIGGART_MESH`.

---

## Architecture & Workflows

### Ready-to-Load Example Workflows (`example_workflows/`)

All **29 Ziggart nodes** and their distinct operating modes are provided as **10 ready-to-load ComfyUI workflow JSON files** inside [`example_workflows/`](./example_workflows/README.md) (drag-and-drop directly onto the ComfyUI canvas or open via **Workflow → Browse Templates → Ziggart-Nodes**):

| Workflow File | Featured Ziggart Nodes | Covered Usage Modes |
| :--- | :--- | :--- |
| [`01_Wildcards_3_Ways_and_LoRA_Encode.json`](./example_workflows/01_Wildcards_3_Ways_and_LoRA_Encode.json) | `ZiggartWildcardProcessor`, `ZiggartWildcardProcessorAllInOne`, `ZiggartCollectionSceneBuilder`, `ZiggartWildcardEncode`, `ZiggartWildcardRecipeSaver` | **Way C1** (YAML/Braces, `search_wildcard_filter` keyword search, `fixed` mode), **Way A** (`combine_selected_pillars = True` vs `False`), **Way C2** modular scene builder + `<lora:...>` & `BREAK` conditioning encode. |
| [`02_Canonical_Duo_Trio_Theme_and_MultiCharacter_Detailer.json`](./example_workflows/02_Canonical_Duo_Trio_Theme_and_MultiCharacter_Detailer.json) | `ZiggartCanonicalThemeResolver`, `ZiggartWildcardVLMEnhancer`, `ZiggartMultiSubjectDetailer`, `ZiggartWildcardRecipeSaver`, `ZiggartGGUFVLMLoader` | `Duo (Female + Male)` & `Single` canonical outfit resolution from `Character-Outfits.json`, VLM prompt harmonization, 3-subject grounded detailing, and side-by-side `Base vs Refined` comparison grid saving. |
| [`03_Detailers_Krea2_Standard_Chained_and_VLM_AutoPrompt.json`](./example_workflows/03_Detailers_Krea2_Standard_Chained_and_VLM_AutoPrompt.json) | `ZiggartDetailerKrea2`, `ZiggartAutoPromptDetailer`, `ZiggartGGUFVLMLoader`, `ZiggartWildcardRecipeSaver` | Sequential chained detailing (`face` → `eyes` with latent noise mask) vs. multi-instance autonomous per-crop VLM micro-prompting (`faces and hands`). |
| [`04_Self_Healing_Quality_Critic_and_Surgical_Repair.json`](./example_workflows/04_Self_Healing_Quality_Critic_and_Surgical_Repair.json) | `ZiggartVLMImageCritic`, `ZiggartCropByBBox`, `ZiggartStitchByBBox`, `ZiggartDetailerKrea2`, `ZiggartGGUFVLMLoader`, `ZiggartWildcardRecipeSaver` | Automated defect localization (`hands_and_limbs`) routing either through **BBox Crop → Custom KSampler → BBox Feathered Stitch** or directly into **Detailers Krea2** via `fix_positive_prompt`. |
| [`05_Mask_Aware_Inpainting_and_WeVisOne_Detector.json`](./example_workflows/05_Mask_Aware_Inpainting_and_WeVisOne_Detector.json) | `ZiggartWeVisOneDetector`, `ZiggartVLMInpaintPrompter`, `ZiggartCropByBBox`, `ZiggartStitchByBBox`, `ZiggartGGUFVLMLoader`, `ZiggartWildcardRecipeSaver` | Standalone open-vocabulary mask generation with **WeVisOne-VL Detector**, scene-harmonized prompt generation with **GGUF Mask-Aware Inpaint Prompter**, and 768px crop inpainting + feathered stitch. |
| [`06_Smart_Tile_Upscaler_VLM_vs_Fast_Mode.json`](./example_workflows/06_Smart_Tile_Upscaler_VLM_vs_Fast_Mode.json) | `ZiggartSmartTileUpscaler`, `ZiggartGGUFVLMLoader`, `ZiggartWildcardRecipeSaver` | **Mode A** (`enable_vlm_tile_prompts = True` anti-hallucination per-tile VLM captioning) vs. **Mode B** (`enable_vlm_tile_prompts = False` fast pure-Krea2 tiled upscaling). |
| [`07_Multimodal_VLM_Studio_VisionPrompt_and_Reference_Harmonizer.json`](./example_workflows/07_Multimodal_VLM_Studio_VisionPrompt_and_Reference_Harmonizer.json) | `ZiggartGGUFVisionPrompt`, `ZiggartWeVisOneDetector`, `ZiggartWildcardVLMEnhancer`, `ZiggartGGUFVLMLoader`, `ZiggartWildcardRecipeSaver` | Single-pass Image-to-Prompt, Multi-pass Specialist Analysis (`analyze_composition`, `analyze_subject`, `analyze_style`) with `detail_crop` + `reference_image`, and reference-guided wildcard harmonization. |
| [`08_Master_All_18_Nodes_Complete_Showcase.json`](./example_workflows/08_Master_All_18_Nodes_Complete_Showcase.json) | **All 18 Core 2D/VLM Ziggart Nodes** | Complete 5-column master architecture pipeline connecting the 18 Wildcard, VLM, Detector, and Detailer nodes. |
| [`09_3D_Generation_Idol_Gaussian_To_Mesh_SPAR3D_and_Hunyuan3D.json`](./example_workflows/09_3D_Generation_Idol_Gaussian_To_Mesh_SPAR3D_and_Hunyuan3D.json) | `ZiggartIdolImageToGaussian`, `ZiggartGaussianToMesh`, `ZiggartSPAR3DImageToMesh`, `ZiggartHunyuan3DImageToMesh`, `Ziggart3DTurntableRenderer` | Complete 3D showcase: **Path A** (Idol `.ply` Gaussian Splat $\rightarrow$ `.glb` Mesh + 360° Turntable), **Path B** (SPAR3D 7GB Low-VRAM Textured `.glb`), and **Path C** (Hunyuan3D-2mini 6GB `.glb`). |
| [`10_Krea2_Prompt_Director_Modular_and_Master_Suite.json`](./example_workflows/10_Krea2_Prompt_Director_Modular_and_Master_Suite.json) | `ZiggartThemeUniverseContext`, `ZiggartCharacterUnitBuilder`, `ZiggartAnatomyCharacterCreator`, `ZiggartSceneStageAtmosphere`, `ZiggartKrea2PromptAssembler`, `ZiggartKrea2MasterPromptDirector` | **Pipeline A (Modular 4-Node Chain)**: Theme/Universe Context $\rightarrow$ Primary & Secondary Character Unit Builders (1:1 Canonical Outfit Link + Auto 9-Zone Anatomy Occlusion) $\rightarrow$ Scene Stage (with `ziggart/Env/*`) $\rightarrow$ Krea2 Assembler (`Qwen3VL-4B` `scoped`); **Pipeline B (Head-to-Toe Anatomy Character Creator with per-section `[None]` & View Presets)**; **Pipeline C (All-in-One Master Prompt Director)**. |

### Workflow A: Krea2 (`Qwen3VL-4B`) Modular Prompt Director + Multi-Character Detailing

```text
           [ Ziggart Krea2 Theme & Universe Context ]
         (Theme / Universe / Free Crossover + Mood Filter)
              |                  |                   |
              v                  v                   v
[ Character Unit Builder 1 ] [ Character Unit Builder 2 ] [ Krea2 Scene Stage ]
(Canonical Outfit + Anatomy  (Secondary Character +       (Style, Action/Duo,
 Auto-Occlusion Filter)       Outfit + Anatomy Filter)     Env, Light, Camera)
              |                  |                   |
              +------------------+-------------------+
                                 |
                                 v
            [ Ziggart Krea2 Prompt Assembler (Qwen3VL-4B) ]
          (scoped: 2 characters, (Render Style) (Char 1) ...)
            |                    |                       |
            | (compiled_prompt)  | (char_1_detailer)     | (char_2_detailer)
            v                    |                       |
   [ Krea 2 Base Sampler ]       |                       |
            | (base_image)       |                       |
            v                    v                       v
   [ Multi-Character Grounded Krea2 Detailer ] <---------+
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
├── custom_nodes/
│   └── Ziggart-Nodes/
│       ├── example_workflows/          <-- 10 ready-to-load ComfyUI workflow JSON templates
│       └── wildcards/                  <-- Place Ziggart.yaml, Collection/, and .txt wildcards here
│           ├── Ziggart.yaml
│           ├── Collection/
│           └── *.txt
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

- **Added new `.yaml` or `.txt` wildcard files on disk**: Toggle `refresh_index: True` for one run on **Ziggart Fast Wildcard Processor** or **Ziggart Krea2 Theme & Universe Context** (or restart ComfyUI; the byte-offset cache automatically detects modified file size/`mtime`).
- **`No detector_model was found`**: Ensure your `.gguf` files are located in `ComfyUI/models/LLM/GGUF/` and click **Refresh** in ComfyUI.
- **`MTMDChatHandler cannot be imported`**: Update `llama-cpp-python` to `>= 0.3.35`.
- **Object not detected**: Adjust `detection_prompt` (e.g. use specific terms like `"human eyes"`, `"face"`, or `"iris"`). Check `detection_preview` output to inspect model predictions.
- **Out of Memory during sampling**: Ensure `free_vram_after` is set to `True`, disable `encode_final_latent` on 4K images, or reduce `target_detail_resolution` from `768` to `512`.

---

## License

This project is licensed under the Apache 2.0 License. Model weights are subject to their respective creators' licenses (Tencent WeVisOne-VL, Krea 2 Community License, Qwen License).
