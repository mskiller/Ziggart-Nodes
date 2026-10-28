# Ziggart Nodes — Workflows Exemples (`example_workflows/`)

Ce dossier contient **9 workflows ComfyUI prêts à charger** (`.json`, glisser-déposer directement dans ComfyUI ou accessibles depuis le menu **Workflow -> Browse Templates -> Ziggart-Nodes**) couvrant **l'intégralité des 18 nœuds Ziggart** dans tous leurs modes et cas d'usage.

---

## Liste des 9 Workflows et des Usages Couverts

### 1. [`01_Wildcards_3_Ways_and_LoRA_Encode.json`](./01_Wildcards_3_Ways_and_LoRA_Encode.json)
Tour complet des **5 nœuds de génération par Wildcards** et de leurs différents modes :
- **`ZiggartWildcardProcessor` (Way C1 — Fast Wildcard Processor)** :
  - **Usage 1A (Standard YAML / TXT & Choix dynamiques)** : Résolution récursive `__Ziggart/scenes/all__`, choix pondérés `{optA|optB|2.0::optC}`, multi-sélections `{2$$, $$__wildcard__}`.
  - **Usage 1B (Mode Recherche `search_wildcard_filter`)** : Champ `wildcard_text` laissé vide + filtre mot-clé (`"cyberpunk"`) pour piocher automatiquement parmi les 78 000+ entrées.
  - **Usage 1C (Mode `fixed`)** : Verrouillage de `populated_text` pour figer un prompt positif ou négatif.
- **`ZiggartWildcardProcessorAllInOne` (Way A — All-in-One Pillars & Mixed)** :
  - **Usage 2A (`combine_selected_pillars = True`)** : Combine automatiquement les sélections des 6 menus déroulants `Collection/` (`Macro/YAML/Themes`, `Character/Anatomy`, `Outfit/Pose`, `Environment/Vehicle`, `Style/Camera/Light`, `Mixed/Other`) avec `wildcard_text` et expose la recette brute sur `combined_wildcard_recipe`.
  - **Usage 2B (`combine_selected_pillars = False`)** : Mode insertion interactive où chaque sélection de pilier s'ajoute directement dans `wildcard_text`.
- **`ZiggartCollectionSceneBuilder` (Way C2)** :
  - Assemble visuellement la scène (`custom_prefix` + 6 piliers `Collection/` + `custom_suffix`) et envoie `wildcard_template` vers l'encodeur LoRA et le Recipe Saver.
- **`ZiggartWildcardEncode` (Fast Wildcard Encode LoRA)** :
  - Déploie les wildcards, extrait et charge automatiquement les tags `<lora:nom:poids_model:poids_clip>` sur `MODEL` et `CLIP`, et encode les segments séparés par `BREAK` en `CONDITIONING`.
- **`ZiggartWildcardRecipeSaver`** :
  - Sauvegarde l'image finale avec les métadonnées PNG (`ziggart_wildcard_template`, `ziggart_populated_text`, `ziggart_seed`) et le fichier `.json` associé.

---

### 2. [`02_Canonical_Duo_Trio_Theme_and_MultiCharacter_Detailer.json`](./02_Canonical_Duo_Trio_Theme_and_MultiCharacter_Detailer.json)
Pipeline de bout en bout pour les **scènes Canoniques (Single, Duo, Trio)** avec retouche multi-personnages ciblée :
- **`ZiggartCanonicalThemeResolver`** :
  - **Usage 2A (`cast_mode = "Duo (Female + Male)"`)** : Lit `Character-Outfits.json` à la volée dans `Collection/Themes/*` pour associer chaque personnage à sa vraie tenue canonique, générant `scene_prompt` ainsi que `character_1_prompt`, `character_2_prompt` et `character_3_prompt`.
  - **Usage 2B (`cast_mode = "Single (Any Gender)"` + `theme_search_filter = "albator"`)** : Filtrage thématique par mot-clé.
- **`ZiggartWildcardVLMEnhancer` (Wildcard-to-VLM Prompt Harmonizer)** :
  - Utilise le modèle GGUF local (`Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)`) pour réécrire les fragments de wildcards en prose fluide tout en préservant strictement les noms et tenues canoniques.
- **`ZiggartMultiSubjectDetailer` (Multi-Character Grounded Krea2 Detailer)** :
  - Reçoit `character_1_prompt`, `character_2_prompt` et `character_3_prompt` directement depuis le résolveur canonique, localise jusqu'à 3 sujets distincts avec **WeVisOne-VL**, puis raffine chaque personnage séparément avec **Krea 2 Turbo** et son propre prompt !
- **`ZiggartWildcardRecipeSaver` (Mode Grille Comparative `build_comparison_grid = True`)** :
  - Compare côte-à-côte l'image de base (`base_image`) et l'image raffinée (`image`) tout en enregistrant le journal JSON des sujets détectés (`subjects_log_json`).

---

### 3. [`03_Detailers_Krea2_Standard_Chained_and_VLM_AutoPrompt.json`](./03_Detailers_Krea2_Standard_Chained_and_VLM_AutoPrompt.json)
Tous les usages de **`Detailers Krea2`** et **`Detailers Krea2 (VLM Auto-Prompt)`** :
- **Branche A — `ZiggartDetailerKrea2` enchaînés (Passe 1 `face` -> Passe 2 `eyes`)** :
  - Partage d'un seul **`ZiggartGGUFVLMLoader`** (`vlm_config`), prévisualisation des boîtes englobantes (`detection_preview`) et des crops raffinés (`cropped_refined`), masque de bruit latent (`use_latent_mask = True`) et fusion gaussienne (`feather_mask`).
- **Branche B — `ZiggartAutoPromptDetailer` (Multi-Instance Autonomous Prompting)** :
  - Détecte plusieurs zones (`"faces and hands"`, `max_detections = 4`), inspecte chaque crop individuellement avec le VLM pour rédiger un micro-prompt descriptif sur-mesure (`generated_prompts`), décharge le VLM une seule fois, puis raffine chaque crop avec Krea 2 Turbo.

---

### 4. [`04_Self_Healing_Quality_Critic_and_Surgical_Repair.json`](./04_Self_Healing_Quality_Critic_and_Surgical_Repair.json)
Auto-inspection et réparation chirurgicale des défauts anatomiques avec **`GGUF Quality & Anatomy Critic`** :
- **`ZiggartVLMImageCritic`** :
  - Inspecte l'image générée (`inspection_focus = "hands_and_limbs"` + `generation_prompt`), attribue une note de qualité (1–10), localise les défauts (`bboxes_json`, `defect_mask`, `preview_image`) et rédige un prompt de réparation (`fix_positive_prompt`).
- **Usage A — Pipeline Chirurgical Découplé (`ZiggartCropByBBox` -> `KSampler` -> `ZiggartStitchByBBox`)** :
  - **`BBox Crop`** extrait uniquement les boîtes défectueuses signalées par le Critic (alignées mod-8 à `768px`) + `crop_masks` + `stitch_pipe`.
  - Un `KSampler` personnalisé ré-échantillonne les crops avec `fix_positive_prompt`, puis **`BBox Feathered Stitch`** les réintègre sans couture dans l'image originale.
- **Usage B — Réparation Directe via `ZiggartDetailerKrea2`** :
  - Envoie `fix_positive_prompt` du Critic directement dans `detail_positive_prompt` de **Detailers Krea2**.

---

### 5. [`05_Mask_Aware_Inpainting_and_WeVisOne_Detector.json`](./05_Mask_Aware_Inpainting_and_WeVisOne_Detector.json)
Détection Open-Vocabulary et Inpainting guidé par masque et VLM :
- **`ZiggartWeVisOneDetector` (WeVisOne-VL Detector)** :
  - Détection autonome en vocabulaire ouvert (`"eyes or sunglasses"`), générant `detection_preview`, `mask`, `cropped_images` et `bboxes_json` sans lancer de diffusion.
- **`ZiggartVLMInpaintPrompter` (GGUF Mask-Aware Inpaint Prompter)** :
  - Prend l'image, un `MASK` (issu de **WeVisOne-VL Detector** ou dessiné au **MaskEditor** sur `LoadImage`) et une instruction courte (`edit_instruction`), encadre la cible sur la scène globale et génère `inpaint_positive_prompt`, `inpaint_negative_prompt`, `annotated_preview` et `bboxes_json` harmonisés avec l'éclairage de la scène.
- **`ZiggartCropByBBox` + Inpaint `KSampler` + `ZiggartStitchByBBox`** :
  - Découpe la zone masquée en `768px`, applique l'inpainting avec `SetLatentNoiseMask`, et recolle le résultat avec fondu gaussien.

---

### 6. [`06_Smart_Tile_Upscaler_VLM_vs_Fast_Mode.json`](./06_Smart_Tile_Upscaler_VLM_vs_Fast_Mode.json)
Les 2 modes du **`Krea2 VLM Smart Tile Upscaler` (`ZiggartSmartTileUpscaler`)** :
- **Mode A (`enable_vlm_tile_prompts = True` — Upscaling Intelligent Anti-Hallucination)** :
  - Le VLM GGUF décrit chaque tuile (`768px`, `tile_overlap = 128`) individuellement avant le passage de Krea 2 Turbo, empêchant l'apparition de visages ou personnages fantômes dans les tuiles de décor lors d'un upscale `2.0x`–`4.0x`.
- **Mode B (`enable_vlm_tile_prompts = False` — Mode Rapide Pure-Krea2)** :
  - Upscaling tuilé avec recouvrement et fusion pondérée sans charger de VLM.

---

### 7. [`07_Multimodal_VLM_Studio_VisionPrompt_and_Reference_Harmonizer.json`](./07_Multimodal_VLM_Studio_VisionPrompt_and_Reference_Harmonizer.json)
Tous les usages de **`GGUF Vision Prompt` (`ZiggartGGUFVisionPrompt`)** et de l'harmonisation multimodale :
- **Usage A — Reverse Prompting Rapide (Image-to-Prompt en 1 passe)** :
  - Convertit une image en prompt descriptif pour Krea 2 Turbo / Flux.
- **Usage B — Analyse Spécialiste Multi-Passes (`analyze_composition`, `analyze_subject`, `analyze_style`) + `detail_crop` + `reference_image`** :
  - **`ZiggartWeVisOneDetector`** extrait un crop haute-résolution (`detail_crop`) du sujet principal et l'envoie avec l'image globale et une image de référence (`reference_image`) à **`GGUF Vision Prompt`**, qui produit 4 sorties distinctes (`answer`, `composition`, `subject`, `style`).
- **Usage C — `ZiggartWildcardVLMEnhancer` guidé par `reference_image`** :
  - Harmonise un prompt wildcard (`Cinematic Director Shot`) pour adopter la palette et l'éclairage d'une image de référence connectée.

---

### 8. [`08_Master_All_18_Nodes_Complete_Showcase.json`](./08_Master_All_18_Nodes_Complete_Showcase.json)
**Vitrine Architecturale Complète** réunissant **les 18 nœuds Ziggart** sur un seul canevas organisé en 5 colonnes thématiques :
1. **Wildcard & Canonical Theme Suite** (`ZiggartWildcardProcessor`, `ZiggartWildcardProcessorAllInOne`, `ZiggartCollectionSceneBuilder`, `ZiggartWildcardEncode`, `ZiggartCanonicalThemeResolver`, `ZiggartWildcardVLMEnhancer`)
2. **GGUF VLM Loader, Vision Prompt & Base Diffusion** (`ZiggartGGUFVLMLoader`, `ZiggartGGUFVisionPrompt`)
3. **Detailers Krea2 Suite** (`ZiggartMultiSubjectDetailer`, `ZiggartDetailerKrea2`, `ZiggartAutoPromptDetailer`)
4. **Detector, Quality Critic, Mask Inpaint & BBox Crop/Stitch** (`ZiggartWeVisOneDetector`, `ZiggartVLMImageCritic`, `ZiggartVLMInpaintPrompter`, `ZiggartCropByBBox`, `ZiggartStitchByBBox`)
5. **Smart Tile Upscaler & Recipe Comparison Saver** (`ZiggartSmartTileUpscaler`, `ZiggartWildcardRecipeSaver`)

---

### 9. [`09_Sdxl_Test_Free_Migrated_to_Ziggart.json`](./09_Sdxl_Test_Free_Migrated_to_Ziggart.json)
Version migrée de votre workflow personnel `Sdxl+test-free.json` où les nœuds `ImpactWildcardProcessor` (positif et négatif) ont été remplacés par **`ZiggartWildcardProcessor`** pour un démarrage instantané sans gel sur les 700+ Mo de wildcards.
