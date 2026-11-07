# Ziggart Nodes — Workflows Exemples (`example_workflows/`)

Ce dossier contient **12 workflows ComfyUI prêts à charger** (`.json`, glisser-déposer directement dans ComfyUI ou accessibles depuis le menu **Workflow -> Browse Templates -> Ziggart-Nodes**) couvrant **l'intégralité des 37 nœuds Ziggart** dans tous leurs modes et cas d'usage.

---

## Liste des 12 Workflows et des Usages Couverts

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
**Vitrine Architecturale Complète** réunissant **les 18 nœuds 2D/VLM Ziggart** sur un seul canevas organisé en 5 colonnes thématiques :
1. **Wildcard & Canonical Theme Suite** (`ZiggartWildcardProcessor`, `ZiggartWildcardProcessorAllInOne`, `ZiggartCollectionSceneBuilder`, `ZiggartWildcardEncode`, `ZiggartCanonicalThemeResolver`, `ZiggartWildcardVLMEnhancer`)
2. **GGUF VLM Loader, Vision Prompt & Base Diffusion** (`ZiggartGGUFVLMLoader`, `ZiggartGGUFVisionPrompt`)
3. **Detailers Krea2 Suite** (`ZiggartMultiSubjectDetailer`, `ZiggartDetailerKrea2`, `ZiggartAutoPromptDetailer`)
4. **Detector, Quality Critic, Mask Inpaint & BBox Crop/Stitch** (`ZiggartWeVisOneDetector`, `ZiggartVLMImageCritic`, `ZiggartVLMInpaintPrompter`, `ZiggartCropByBBox`, `ZiggartStitchByBBox`)
5. **Smart Tile Upscaler & Recipe Comparison Saver** (`ZiggartSmartTileUpscaler`, `ZiggartWildcardRecipeSaver`)

---

### 9. [`09_3D_Generation_Idol_Gaussian_To_Mesh_SPAR3D_and_Hunyuan3D.json`](./09_3D_Generation_Idol_Gaussian_To_Mesh_SPAR3D_and_Hunyuan3D.json)
Pipeline complet des **5 nœuds de génération 3D (`Ziggart/3D`)** optimisés pour **RTX 3070 Ti (8 Go – 16 Go VRAM)** :
- **Branche 1 — `ZiggartIdolImageToGaussian` $\rightarrow$ `ZiggartGaussianToMesh` + `Ziggart3DTurntableRenderer`** :
  - Génère un **3D Gaussian Splat (`.ply`)** à partir d'une seule image avec **`speridlabs/idol`** (`~7.2 Go` VRAM), le convertit en maillage polygonal **`.glb` / `.obj`** avec couleurs par sommet et lissage Taubin, et calcule une séquence **Turntable 360°** (`IMAGE` + `MASK`).
- **Branche 2 — `ZiggartSPAR3DImageToMesh` (`low_vram_mode = True`, `~7.0 Go` VRAM)** :
  - Génère un maillage 3D **`.glb` texturé UV** et son nuage de points intermédiaire `.ply` en ~1 à 2 secondes avec **SPAR3D** (Stability AI).
- **Branche 3 — `ZiggartHunyuan3DImageToMesh` (`Hunyuan3D-2mini`, `~5.5 Go` VRAM)** :
  - Génère un maillage 3D `.glb` haute précision avec déchargement séquentiel automatique du modèle Shape avant l'exécution optionnelle de `Hunyuan3DPaintPipeline`.

---

### 10. [`10_Krea2_Prompt_Director_Modular_and_Master_Suite.json`](./10_Krea2_Prompt_Director_Modular_and_Master_Suite.json)
Pipeline complet des **6 nœuds `Ziggart/Krea2 Prompt Director`** optimisés pour **Krea2 (Qwen3VL-4B)** :
- **Pipeline Modulaire (4 nœuds chaînables)** :
  1. **`ZiggartThemeUniverseContext`** : Sélectionne un Thème (`Collection/Themes`), un Univers (`Collection/Universes`), ou le mode `Free Crossover` (tous les dossiers : `Collection`, `ziggart`, `jff`, `Encours`), avec filtre d'ambiance (`mood_filter`) et exclusion stricte des wildcards de prompts complets (`exclude_full_prompts = True`).
  2. **`ZiggartCharacterUnitBuilder` (Primary & Secondary)** : Construit chaque personnage avec liaison Personnage $\leftrightarrow$ Tenue héritée de **Wildcard-GPT** (`Canonical` 1:1, `Compatible`, `Free`, `Nude`) et **filtrage automatique d'occlusion Vêtement $\rightarrow$ Anatomie sur 9 zones** (bloque automatiquement les descriptions de poitrine, tétons, taille/ventre, dos, épaules/bras, fesses, zones intimes, cuisses/jambes ou pieds/chevilles si la tenue tirée les recouvre).
  3. **`ZiggartSceneStageAtmosphere`** : Sélectionne le Style, l'Action / Interaction Duo, l'Environnement (incluant `ziggart/Env/*`), l'Éclairage et la Caméra.
  4. **`ZiggartKrea2PromptAssembler`** : Compile le prompt final selon la structure `scoped` entre parenthèses avec ancre de dénombrement hors parenthèses (ou politiques `balanced`, `compact`, `rich`) et fournit les prompts individuels pour les Detailers ainsi que l'estimation de tokens Qwen3VL-4B.
- **Créateur de Personnage de la Tête aux Pieds (`ZiggartAnatomyCharacterCreator`)** :
  - Construit un personnage (`Female`, `Male`, `Creature`) à travers **28 sections anatomiques** où chaque section peut être mise sur `[None]` ou pilotée par un préréglage de vue (`Full Body Front`, `Back View / Seen From Behind`, `Back Over-the-Shoulder`, `Upper Body Portrait`, `Close-Up Face Only`, `Lower Body & Legs`).
- **Nœud All-in-One (`ZiggartKrea2MasterPromptDirector`)** :
  - Regroupe l'ensemble de la création Solo / Duo, des filtres Tenue/Anatomie, du compilateur Krea2 et du bouton **`🎲 Pre-Generate Prompt (Test Without Queue)`** dans un seul nœud.

---

### 11. [`11_Trio_Ensemble_Spatial_Layout_and_Regional_Conditioning.json`](./11_Trio_Ensemble_Spatial_Layout_and_Regional_Conditioning.json)
Pipeline complet **Multi-Personnages Trio/Ensemble + Conditionnement Régional Spatial + Optimisation d'Attention** :
- **`ZiggartKrea2TrioEnsembleDirector`** :
  - Assemble jusqu'à **3 personnages complets** (`char_1_slot`, `char_2_slot`, `char_3_slot`) + un compagnon/monture/accessoire (`companion_or_prop_text`) et génère à la fois le prompt compilé Krea2 et une disposition spatiale `ZIGGART_SPATIAL_LAYOUT`.
- **`ZiggartSpatialLayoutDesigner` (Interactive Regions)** :
  - Permet de dessiner/redimensionner jusqu'à 4 boîtes englobantes 2D (`[x1, y1, x2, y2]`) directement à la souris sur le canevas interactif du nœud (ou via les curseurs) et de les connecter **optionnellement** à `ZiggartRegionalConditioningComposer`.
- **`ZiggartRegionalConditioningComposer`** :
  - Encode le décor global (`base_scene_prompt`) et chaque personnage (`char_1..3_prompt`) avec des masques spatiaux adoucis (`feather_ratio`) pour éviter toute fuite d'attributs ou de vêtements entre personnages (utilise `spatial_layout` s'il est connecté, ou `layout_preset` sinon).
- **`ZiggartPromptTokenInspectorAndCompressor`** :
  - Analyse le budget de tokens CLIP (blocs de 77 tokens) et Qwen3VL-4B, détecte les doublons/contradictions et compresse les prompts trop longs en préservant les verrous personnages.
- **`ZiggartAttentionSelector`** :
  - Détecte et applique automatiquement le meilleur backend d'attention (`sage`, `flash`, `xformers`, `pytorch` SDPA) sur le `MODEL`.

---

### 12. [`12_Self_Healing_Detailer_Contour_Matting_and_Color_Lock.json`](./12_Self_Healing_Detailer_Contour_Matting_and_Color_Lock.json)
Pipeline **Auto-Réparation en Boucle Fermée, Détourage de Silhouette & Verrouillage Colorimétrique** :
- **`ZiggartContourMaskRefiner`** :
  - Convertit les boîtes rectangulaires de **WeVisOne-VL Detector** en masques de contour ajustés à la silhouette du sujet (contraste bord/fond + énergie Sobel + fermeture morphologique + fondu), évitant les halos de fond autour des doigts ou des cheveux.
- **`ZiggartClosedLoopSelfHealingDetailer`** :
  - Évalue chaque région détectée (`1.0–10.0`), ignore les régions déjà `>= target_score`, ré-échantillonne itérativement (`1..max_iterations`) les zones défectueuses avec `step_denoise_boost`, et conserve systématiquement la meilleure passe (`never_degrade = True`).
- **`ZiggartColorAndFrequencyMatcher`** :
  - Applique un transfert colorimétrique perceptuel de Reinhard (`lab_reinhard_transfer`) combiné à une séparation de fréquences pour préserver les détails haute fréquence tout en verrouillant la teinte de peau et l'éclairage local.
