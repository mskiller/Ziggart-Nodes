from __future__ import annotations

import glob
import json
import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import nodes as comfy_nodes
except ImportError:
    comfy_nodes = None

try:
    from server import PromptServer
    from aiohttp import web
except Exception:
    PromptServer = None
    web = None

from .embedded_model import (
    MODEL_MANAGER,
    EmbeddedLlamaError,
    ModelConfig,
    release_comfy_vram,
)
from .nodes import (
    NO_MMPROJ,
    NO_MODEL,
    _gguf_files,
    _resolve_gguf,
    image_to_data_uri,
)
from .krea2_compiler import (
    ANATOMY_ZONES,
    DUO_INTERACTIONS_BANK,
    ENSEMBLE_SPATIAL_LAYOUT_PRESETS,
    HEAD_TO_TOE_ANATOMY_SECTIONS,
    VIEW_VISIBILITY_PRESETS,
    CharacterSlotData,
    OutfitOcclusionReport,
    SceneStageData,
    analyze_outfit_occlusion,
    assemble_krea2_structured_prompt,
    classify_anatomy_wildcard_zone,
    clean_outfit_prefix,
    estimate_qwen_tokens,
    filter_anatomy_text_by_occlusion,
    get_theme_outfit_choices,
    is_full_prompt_wildcard_key,
    is_full_scene_line,
    is_section_enabled_by_view_preset,
    load_theme_bundle,
    load_universe_bundle,
)


NONE_OPTION = "[None]"
AUTO_VIEW_OPTION = "[Auto (From View Preset)]"
RANDOM_GENDER_DEFAULT_OPTION = "[Random (Gender Default)]"
FROM_CONTEXT_OPTION = "[From Context (Theme / Universe)]"
RANDOM_POOL_OPTION = "[Random From Category]"
SELECT_PLACEHOLDER = "[Select a Wildcard to Insert or Resolve]"
LORA_PLACEHOLDER = "[Select a LoRA to Insert]"
RANDOM_THEME_OPTION = "[Random Theme]"
RANDOM_UNIVERSE_OPTION = "[Random Universe]"

# Fast regex patterns for YAML byte-offset indexing and wildcard syntax
_YAML_KEY_LINE_RE = re.compile(r"^(\s*)([A-Za-z0-9_.\-/ ]+?)\s*:\s*(.*)$")
_YAML_LIST_LINE_RE = re.compile(r"^\s*-\s+(.*)$")
_WILDCARD_TOKEN_RE = re.compile(r"_{2,}([\w.\-+/*\\ ]+?)(?<!_)_{2,}")
_QUANTIFIER_RE = re.compile(r"(?P<quantifier>\d+)#_{2,}(?P<keyword>[\w.\-+/*\\ ]+?)(?<!_)_{2,}", re.IGNORECASE)
_CHOICE_BLOCK_RE = re.compile(r"(?<!\\)\{((?:[^{}]|(?<=\\)[{}])*?)(?<!\\)\}")
_LORA_TAG_RE = re.compile(r"<lora:([^>]+)>", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)$")


def normalize_wildcard_key(key: str) -> str:
    """Normalizes a wildcard key to forward-slash, lowercase, trimmed format."""
    cleaned = key.strip().replace("\\", "/")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    return cleaned.strip("/_ ").lower()


def _is_numeric(val: str) -> bool:
    return bool(_NUMERIC_RE.match(val.strip()))


def discover_default_wildcard_dirs() -> list[str]:
    """
    Automatically finds wildcard directories without modifying any external folders:
    1. Ziggart-Nodes/wildcards (if present)
    2. ComfyUI/wildcards (if present)
    3. ComfyUI/custom_nodes/ComfyUI-Impact-Pack/wildcards (sibling or E:/Comfy fallback)
    """
    candidates: list[Path] = []
    here = Path(__file__).resolve().parent

    candidates.append(here / "wildcards")
    custom_nodes_dir = here.parent
    candidates.append(custom_nodes_dir / "ComfyUI-Impact-Pack" / "wildcards")
    comfy_root = custom_nodes_dir.parent
    candidates.append(comfy_root / "wildcards")

    if folder_paths is not None and hasattr(folder_paths, "base_path"):
        base = Path(folder_paths.base_path)
        candidates.append(base / "wildcards")
        candidates.append(base / "custom_nodes" / "ComfyUI-Impact-Pack" / "wildcards")

    dev_impact = Path(r"E:\Comfy\ComfyUI\custom_nodes\ComfyUI-Impact-Pack\wildcards")
    candidates.append(dev_impact)

    seen: set[str] = set()
    valid_dirs: list[str] = []
    for c in candidates:
        try:
            if c.is_dir():
                norm = os.path.normcase(str(c.resolve()))
                if norm not in seen:
                    seen.add(norm)
                    valid_dirs.append(str(c.resolve()))
        except OSError:
            continue
    return valid_dirs


def index_yaml_byte_offsets(yaml_path: str) -> dict[str, tuple[int, int]]:
    """
    Scans a YAML wildcard file (even 50+ MB / 150,000+ lines) in ~0.2s using a streaming
    indentation state machine. Supports arbitrary dictionary nesting depth (1, 2, 3, 4+ levels)
    and returns {normalized_key: (byte_start, byte_end)} for every list block.
    """
    offsets: dict[str, tuple[int, int]] = {}
    key_stack: list[tuple[int, str]] = []
    active_key: str | None = None
    active_start: int = 0
    active_indent: int = -1

    with open(yaml_path, "rb") as f:
        while True:
            line_start = f.tell()
            raw_line = f.readline()
            if not raw_line:
                if active_key is not None:
                    offsets[active_key] = (active_start, line_start)
                break

            line_str = raw_line.decode("utf-8", errors="replace").lstrip("\ufeff").expandtabs(2)
            stripped = line_str.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line_str) - len(line_str.lstrip(" "))

            if stripped.startswith("- ") or stripped == "-":
                if active_key is None and key_stack:
                    active_key = normalize_wildcard_key("/".join(k for _, k in key_stack))
                    active_start = line_start
                    active_indent = key_stack[-1][0]
                continue

            key_match = _YAML_KEY_LINE_RE.match(line_str)
            if key_match:
                if active_key is not None:
                    offsets[active_key] = (active_start, line_start)
                    active_key = None
                    active_indent = -1

                key_name = key_match.group(2).strip().strip("\"'")
                rest = key_match.group(3).strip()

                while key_stack and key_stack[-1][0] >= indent:
                    key_stack.pop()
                key_stack.append((indent, key_name))

                if rest and not rest.startswith("#"):
                    full_key = normalize_wildcard_key("/".join(k for _, k in key_stack))
                    offsets[full_key] = (line_start, f.tell())

    return offsets


def read_yaml_slice_items(yaml_path: str, byte_start: int, byte_end: int) -> list[str]:
    """Reads only the [byte_start:byte_end] slice of a YAML file and extracts wildcard strings."""
    with open(yaml_path, "rb") as f:
        f.seek(byte_start)
        chunk = f.read(max(0, byte_end - byte_start)).decode("utf-8", errors="replace")

    items: list[str] = []
    for line in chunk.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _YAML_LIST_LINE_RE.match(line)
        if m:
            val = m.group(1).strip()
            if val.startswith('"') and val.endswith('"') and len(val) >= 2:
                val = val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            elif val.startswith("'") and val.endswith("'") and len(val) >= 2:
                val = val[1:-1].replace("''", "'")
            if val:
                items.append(val)
        else:
            km = _YAML_KEY_LINE_RE.match(line)
            if km:
                rest = km.group(3).strip()
                if rest.startswith("[") and rest.endswith("]"):
                    inner = rest[1:-1]
                    for part in inner.split(","):
                        p = part.strip().strip("\"'")
                        if p:
                            items.append(p)
                elif rest and not rest.startswith("#"):
                    val = rest.strip("\"'")
                    if val:
                        items.append(val)
    return items


class ZiggartWildcardIndex:
    """
    High-speed, zero-freeze wildcard index and resolver designed for 700+ MB / 80,000+ file
    wildcard libraries:
    - Indexes arbitrary-depth YAML files (like 53.5 MB Ziggart.yaml and krea2*.yaml) via
      byte-offset slices and persists the index to disk so startup/refresh takes < 40 ms.
    - Organizes wildcards into dedicated Collection Pillar dropdowns + a Mixed dropdown.
    - Resolves .txt wildcards and `*` glob patterns (e.g. `Collection/formats-wild/.../Female-*`,
      `jff/animals/*`) in O(1) via direct path & cached glob lookup without walking 78,000 files on boot.
    """

    PILLAR_FOLDERS = {
        "character_anatomy": ("Characters", "Anatomy", "DuosTeams", "Bestiary", "DisneyPrincesses", "PixieHollow"),
        "outfit_pose": ("Outfits", "Pose", "Expression"),
        "env_vehicle": ("Environment", "IconicPlaces", "Vehicle"),
        "style_camera_light": ("Style", "Medium", "Lighting", "Camera", "VFX", "Sheet", "Format-dev"),
    }

    LEGACY_ALIAS_MAP = {
        "actors1": "jff/actors",
        "actress1": "jff/actress",
        "jffcum": "jff/liquid",
        "mklan-female": "jff/mklan-female-card",
        "mklan-male-card": "male-full",
        "mskiller/mklan-male-card": "male-full",
        "mskiller/mklan-female-card": "jff/mklan-female-card",
        "mskiller/alluring-clothes": "jff/alluring-clothes",
        "mskiller/alluring-pose": "jff/alluring-pose",
        "mskiller/jff-clothes": "jff/jff-clothes",
        "color": "jff/jffcouleur",
        "colors": "jff/jffcouleur",
        "nationality": "jff/jff-female-names",
        "creativelocations": "jff/jfflieu",
        "cc_various_background": "jff/jfflieu",
        "outfits": "outfit-full",
        "outfits_general": "jff/jff-clothes-v3",
        "outfits_sensitive": "jff/lingerie_outfits",
        "aln/clothing/anything": "jff/jff-clothes-v3",
        "cf-punk/bottom-bis": "jff/women-bottom",
        "advanced-post-apocalyptic-outfits": "jff/jff-clothes-v3",
        "characters": "female-full",
        "characters/professions/historical": "jff/class",
        "art_style_prompt": "jff/jff-styles",
        "artist-photographer": "jff/artists-art-nouveau",
        "illustrators": "jff/artists-art-nouveau",
        "painting": "style-full-v2",
        "peaksel_camerashotype": "jff/peaksel_camerashotangle",
        "dinosaur": "creature-full",
        "creatures/species/insects": "creature-full",
        "detail": "jff/hairdetail",
        "drink_prompt_image": "jff/item",
        "food": "jff/item",
        "personmaker/commonethnicities": "jff/jff-female-names",
        "personmaker/commonnationalities": "jff/jff-female-names",
        "ponycharacter": "female-full",
        "promptgeek/photorealistic/camera/angle": "jff/camera-angle",
        "seasons": "jff/jfftemps",
        "tree": "jff/garden-summer",
        "wildcard_character": "female-full",
    }

    LEGACY_GLOB_ALIAS_MAP = {
        "cc_female_clothing_set_*": "jff/jff-clothes*",
        "advanced-female-*": "female-*",
        "bo/chars/*": "characters/*/*",
        "characters_*": "characters/*/*",
        "mskiller/mklan-*male-card": "jff/mklan-*",
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._dirs: list[str] = []
        self._yaml_offsets: dict[str, tuple[str, int, int]] = {}
        self._yaml_children: dict[str, list[str]] = {}
        self._curated_txt_map: dict[str, str] = {}
        self._options_cache: dict[str, list[str]] = {}
        self._glob_files_cache: dict[str, list[str]] = {}
        self._max_cached_keys = 256

        # Categorized dropdown lists matching Collection structure + Mixed
        self.pillars: dict[str, list[str]] = {
            "macro_yaml_themes": [],
            "character_anatomy": [],
            "outfit_pose": [],
            "env_vehicle": [],
            "style_camera_light": [],
            "mixed_other": [],
        }
        # Pure facet component pools (excluding full pre-assembled scene prompts!) across Collection, ziggart, jff, Encours, Outfits, Characters
        self.facets: dict[str, list[str]] = {
            "styles": [],
            "characters": [],
            "anatomy": [],
            "outfits": [],
            "poses_actions": [],
            "environments": [],
            "lighting": [],
            "camera": [],
        }
        self.theme_folders: dict[str, str] = {}  # theme_name -> absolute folder path
        self.universe_folders: dict[str, str] = {}  # universe_name -> absolute folder path
        self._dropdown_items: list[str] = []
        self._initialized = False

    def _cache_file_path(self) -> Path:
        cache_dir = Path(__file__).resolve().parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "wildcard_yaml_offsets.json"

    def ensure_loaded(self, extra_dir: str = "", force_refresh: bool = False) -> None:
        with self._lock:
            dirs = discover_default_wildcard_dirs()
            if extra_dir and extra_dir.strip():
                custom = Path(os.path.expandvars(os.path.expanduser(extra_dir.strip())))
                if custom.is_dir():
                    c_str = str(custom.resolve())
                    if c_str not in dirs:
                        dirs.insert(0, c_str)

            if self._initialized and not force_refresh and dirs == self._dirs:
                return

            self._dirs = dirs
            self._yaml_offsets.clear()
            self._yaml_children.clear()
            self._curated_txt_map.clear()
            self._options_cache.clear()
            self._glob_files_cache.clear()
            self.theme_folders.clear()
            self.universe_folders.clear()
            for k in self.pillars:
                self.pillars[k] = []
            for fk in self.facets:
                self.facets[fk] = []

            cache_file = self._cache_file_path()
            disk_cache: dict[str, Any] = {}
            if cache_file.is_file() and not force_refresh:
                try:
                    with open(cache_file, "r", encoding="utf-8") as cf:
                        disk_cache = json.load(cf)
                except Exception:
                    disk_cache = {}

            cache_dirty = False

            for base_dir in self._dirs:
                base_path = Path(base_dir)
                if not base_path.is_dir():
                    continue

                try:
                    root_entries = sorted(os.scandir(base_dir), key=lambda e: e.name.lower())
                except OSError:
                    continue

                # 1. Root files (.yaml, .yml -> macro_yaml_themes; .txt -> mixed_other)
                for entry in root_entries:
                    if entry.is_file():
                        lower_name = entry.name.lower()
                        if lower_name.endswith((".yaml", ".yml")):
                            if self._index_single_yaml(entry.path, disk_cache):
                                cache_dirty = True
                        elif lower_name.endswith(".txt"):
                            try:
                                if entry.stat().st_size == 0:
                                    continue
                            except OSError:
                                continue
                            stem = os.path.splitext(entry.name)[0]
                            norm_k = normalize_wildcard_key(stem)
                            self._curated_txt_map[norm_k] = entry.path
                            self.pillars["mixed_other"].append(f"__{stem}__")

                # 2. Subdirectories
                for entry in root_entries:
                    if not entry.is_dir():
                        continue
                    if entry.name.lower() == "collection":
                        self._scan_collection_structure(Path(entry.path), base_path, disk_cache)
                    else:
                        self._scan_into_bucket(
                            Path(entry.path),
                            base_path,
                            self.pillars["mixed_other"],
                            disk_cache,
                            max_files=2000,
                        )

            # Route YAML keys into their matching Collection Pillars and build parent rollups
            custom_yaml_top_tokens: list[str] = []
            for y_key in sorted(self._yaml_offsets.keys()):
                parts = y_key.split("/")
                for depth in range(1, len(parts)):
                    parent = "/".join(parts[:depth])
                    self._yaml_children.setdefault(parent, []).append(y_key)

                top_root = parts[0] if parts else ""
                sec = parts[1] if len(parts) > 1 else ""
                token = f"__{y_key}__"

                if top_root != "ziggart":
                    # Friend/custom root YAML files (e.g. krea2.yaml, krea2-grok.yaml, krea2-kimi.yaml, etc.)
                    custom_yaml_top_tokens.append(token)
                    self.pillars["mixed_other"].insert(0, token)
                elif sec in ("character", "anatomy", "bestiary"):
                    self.pillars["character_anatomy"].append(token)
                elif sec in ("outfit", "pose", "action_pose", "expression", "expressions"):
                    self.pillars["outfit_pose"].append(token)
                elif sec in ("environment", "vehicle"):
                    self.pillars["env_vehicle"].append(token)
                elif sec in ("style", "medium", "art_medium", "lighting", "camera", "vfx", "sheet", "sheet_layout", "sheet_elements"):
                    self.pillars["style_camera_light"].append(token)
                elif sec in ("generators", "random", "compatibility"):
                    custom_yaml_top_tokens.append(token)
                elif sec == "scenes":
                    # Include 3-level scene keys (e.g. ziggart/scenes/all, ziggart/scenes/01_...) in dropdown;
                    # deeper 4/5-level subvariants remain in _yaml_offsets for instant resolution & search
                    if len(parts) <= 3:
                        self.pillars["macro_yaml_themes"].append(token)
                elif sec in ("universe", "master"):
                    if len(parts) <= 3:
                        self.pillars["mixed_other"].append(token)
                else:
                    self.pillars["macro_yaml_themes"].append(token)

            # Include 3-level parent rollups (e.g. __krea2-grok__, __ziggart/scenes/01_mysterieusescitesdor__)
            parent_scene_tokens: list[str] = []
            for p_key in sorted(self._yaml_children.keys()):
                if p_key not in self._yaml_offsets:
                    p_parts = p_key.split("/")
                    if len(p_parts) <= 3:
                        parent_scene_tokens.append(f"__{p_key}__")
            self.pillars["macro_yaml_themes"] = (
                custom_yaml_top_tokens + parent_scene_tokens + self.pillars["macro_yaml_themes"]
            )

            # Deduplicate each pillar list while preserving order
            all_combined: list[str] = []
            seen_all: set[str] = set()
            for pk, items in self.pillars.items():
                dedup: list[str] = []
                seen_p: set[str] = set()
                for it in items:
                    low = it.lower()
                    if low not in seen_p:
                        seen_p.add(low)
                        dedup.append(it)
                    if low not in seen_all and len(all_combined) < 3000:
                        seen_all.add(low)
                        all_combined.append(it)
                self.pillars[pk] = dedup

            # Populate pure component facets (excluding full scene prompts) from curated_txt_map and Ziggart.yaml
            self._populate_pure_facets()

            if cache_dirty:
                try:
                    with open(cache_file, "w", encoding="utf-8") as cf:
                        json.dump(disk_cache, cf)
                except Exception:
                    pass

            self._dropdown_items = all_combined
            self._initialized = True

    def _index_single_yaml(self, yaml_path: str, disk_cache: dict[str, Any]) -> bool:
        try:
            stat = os.stat(yaml_path)
        except OSError:
            return False
        fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}"
        cached = disk_cache.get(yaml_path)
        dirty = False
        if isinstance(cached, dict) and cached.get("fp") == fingerprint and "offsets" in cached:
            offsets = {k: (int(v[0]), int(v[1])) for k, v in cached["offsets"].items()}
        else:
            offsets = index_yaml_byte_offsets(yaml_path)
            disk_cache[yaml_path] = {
                "fp": fingerprint,
                "offsets": {k: [v[0], v[1]] for k, v in offsets.items()},
            }
            dirty = True
        for k, (b_start, b_end) in offsets.items():
            self._yaml_offsets[k] = (yaml_path, b_start, b_end)
        return dirty

    def _scan_into_bucket(
        self,
        folder: Path,
        root_dir: Path,
        bucket: list[str],
        disk_cache: dict[str, Any],
        max_files: int = 350,
    ) -> None:
        count = 0
        try:
            for r, _dirs, files in os.walk(folder):
                for fname in sorted(files):
                    lower = fname.lower()
                    full_p = os.path.join(r, fname)
                    if lower.endswith((".yaml", ".yml")):
                        self._index_single_yaml(full_p, disk_cache)
                    elif lower.endswith(".txt"):
                        try:
                            if os.path.getsize(full_p) == 0:
                                continue
                        except OSError:
                            continue
                        rel = os.path.relpath(full_p, root_dir).replace("\\", "/")
                        stem = os.path.splitext(rel)[0]
                        norm_k = normalize_wildcard_key(stem)
                        self._curated_txt_map[norm_k] = full_p
                        bucket.append(f"__{stem}__")
                        count += 1
                        if count >= max_files:
                            return
        except OSError:
            pass

    def _scan_collection_structure(
        self,
        collection_dir: Path,
        root_dir: Path,
        disk_cache: dict[str, Any],
    ) -> None:
        # 1. Scan the 4 structured Collection pillars
        for pillar_key, folder_names in self.PILLAR_FOLDERS.items():
            for fname in folder_names:
                sub = collection_dir / fname
                if sub.is_dir():
                    self._scan_into_bucket(sub, root_dir, self.pillars[pillar_key], disk_cache, max_files=250)

        # 2. Scan Collection/Index into mixed_other
        idx_dir = collection_dir / "Index"
        if idx_dir.is_dir():
            self._scan_into_bucket(idx_dir, root_dir, self.pillars["mixed_other"], disk_cache, max_files=200)

        # 3. Fast single-level scandir of Collection/Combos (avoiding 41k subfile walk)
        combos_dir = collection_dir / "Combos"
        if combos_dir.is_dir():
            try:
                for entry in sorted(os.scandir(combos_dir), key=lambda e: e.name.lower()):
                    if entry.is_file() and entry.name.lower().endswith(".txt"):
                        rel = os.path.relpath(entry.path, root_dir).replace("\\", "/")
                        stem = os.path.splitext(rel)[0]
                        self._curated_txt_map[normalize_wildcard_key(stem)] = entry.path
                        self.pillars["macro_yaml_themes"].append(f"__{stem}__")
                    elif entry.is_dir():
                        all_txt = os.path.join(entry.path, "ALL.txt")
                        if os.path.isfile(all_txt):
                            rel = os.path.relpath(all_txt, root_dir).replace("\\", "/")
                            stem = os.path.splitext(rel)[0]
                            self._curated_txt_map[normalize_wildcard_key(stem)] = all_txt
                            self.pillars["macro_yaml_themes"].append(f"__{stem}__")
            except OSError:
                pass

        # 4. Fast single-level scandir of Collection/Themes (~1,518 theme folders in ~3ms!)
        themes_dir = collection_dir / "Themes"
        if themes_dir.is_dir():
            try:
                for entry in sorted(os.scandir(themes_dir), key=lambda e: e.name.lower()):
                    if entry.is_dir():
                        self.theme_folders[entry.name] = entry.path
                        rel_all = f"Collection/Themes/{entry.name}/ALL"
                        self.pillars["macro_yaml_themes"].append(f"__{rel_all}__")
            except OSError:
                pass

        # 5. Fast single-level scandir of Collection/Universes & Collection/Canonical
        for macro_sub in ("Universes", "Canonical"):
            m_dir = collection_dir / macro_sub
            if m_dir.is_dir():
                try:
                    for entry in sorted(os.scandir(m_dir), key=lambda e: e.name.lower()):
                        if entry.is_dir():
                            if macro_sub == "Universes":
                                self.universe_folders[entry.name] = entry.path
                            self.pillars["mixed_other"].append(f"__Collection/{macro_sub}/{entry.name}/ALL__")
                        elif entry.is_file() and entry.name.lower().endswith(".txt"):
                            rel = os.path.relpath(entry.path, root_dir).replace("\\", "/")
                            stem = os.path.splitext(rel)[0]
                            self.pillars["mixed_other"].append(f"__{stem}__")
                except OSError:
                    pass

        # 6. Fast single-level scandir of Collection/formats-wild/Formats (~130 format folders in <0.5ms)
        formats_dir = collection_dir / "formats-wild" / "Formats"
        if formats_dir.is_dir():
            try:
                for entry in sorted(os.scandir(formats_dir), key=lambda e: e.name.lower()):
                    if entry.is_dir():
                        self.pillars["style_camera_light"].append(
                            f"__Collection/formats-wild/Formats/{entry.name}/Templates/Female-*__"
                        )
                        self.pillars["style_camera_light"].append(
                            f"__Collection/formats-wild/Formats/{entry.name}/Templates/*__"
                        )
            except OSError:
                pass

    def _populate_pure_facets(self) -> None:
        """
        Classifies all non-full-prompt wildcards from Collection, ziggart, jff, Encours, Outfits,
        Characters, and Ziggart.yaml into the 8 pure component facets:
        styles, characters, anatomy, outfits, poses_actions, environments, lighting, camera.
        """
        seen_per_facet: dict[str, set[str]] = {fk: set() for fk in self.facets}

        def _add_facet(facet_name: str, token: str) -> None:
            low = token.lower()
            if low not in seen_per_facet[facet_name]:
                seen_per_facet[facet_name].add(low)
                self.facets[facet_name].append(token)

        # 1. Classify .txt files from _curated_txt_map
        for norm_k, full_path in sorted(self._curated_txt_map.items()):
            if is_full_prompt_wildcard_key(norm_k):
                continue

            # Reconstruct readable relative stem
            token = f"__{norm_k}__"
            for base_d in self._dirs:
                try:
                    if os.path.normcase(full_path).startswith(os.path.normcase(base_d)):
                        rel = os.path.relpath(full_path, base_d).replace("\\", "/")
                        stem = os.path.splitext(rel)[0]
                        token = f"__{stem}__"
                        break
                except ValueError:
                    continue

            k_low = norm_k.lower()
            base_name = k_low.rsplit("/", 1)[-1]

            # Collection structured folders
            if k_low.startswith("collection/style/") or k_low.startswith("collection/medium/") or k_low.startswith("ziggart/style/"):
                _add_facet("styles", token)
            elif k_low.startswith("collection/characters/") or k_low.startswith("characters/") or k_low.startswith("collection/bestiary/") or k_low.startswith("collection/duosteams/") or k_low.startswith("collection/disneyprincesses/") or k_low.startswith("collection/pixiehollow/"):
                _add_facet("characters", token)
            elif k_low.startswith("collection/anatomy/"):
                _add_facet("anatomy", token)
            elif k_low.startswith("collection/outfits/") or k_low.startswith("outfits/") or k_low.startswith("ziggart/outfits/"):
                _add_facet("outfits", token)
            elif k_low.startswith("collection/pose/") or k_low.startswith("collection/expression/"):
                _add_facet("poses_actions", token)
            elif k_low.startswith("collection/environment/") or k_low.startswith("collection/iconicplaces/") or k_low.startswith("ziggart/env/"):
                _add_facet("environments", token)
                if k_low.startswith("ziggart/env/") and token not in self.pillars["env_vehicle"]:
                    self.pillars["env_vehicle"].append(token)
            elif k_low.startswith("collection/lighting/"):
                _add_facet("lighting", token)
            elif k_low.startswith("collection/camera/"):
                _add_facet("camera", token)
            else:
                # Root, jff/, Encours/, ziggart/ root files classification
                if any(w in k_low for w in ("style", "styl", "artists-art-nouveau", "artstyletheme")):
                    _add_facet("styles", token)
                elif any(w in k_low for w in ("camera-angle", "camerafocus", "camerashotangle", "photoshoot-type")):
                    _add_facet("camera", token)
                elif any(w in k_low for w in ("ambiance-light", "art-deco-lighting")) or base_name == "light":
                    _add_facet("lighting", token)
                elif any(w in k_low for w in ("breast-shape", "jffbreast-size", "female-body", "femme-body-description", "pussies", "women-faces")):
                    _add_facet("anatomy", token)
                elif any(w in k_low for w in ("outfit", "clothe", "lingerie", "vetement", "bikini", "swimsuit", "armors", "uniform", "cosplay", "sandralinge", "stucklinge", "man-underwear", "women-bottom", "collar")):
                    _add_facet("outfits", token)
                elif any(w in k_low for w in ("pose", "action", "sex-position", "position_hentai", "sex-act", "villains-act", "emotion-expression")):
                    _add_facet("poses_actions", token)
                elif any(w in k_low for w in ("environnement", "environment", "scenery", "lieu", "location", "maison", "garden-summer", "train-station", "/env/", "-env")):
                    _add_facet("environments", token)
                elif any(
                    w in base_name
                    for w in (
                        "female-all", "female-full", "male-all", "male-full", "creature-all", "creature-full",
                        "actors", "actress", "celeb-female", "1680-elves", "20females", "aln-superheroine",
                        "chinese-myth-women", "cyborg", "dcmheroes", "dcmvillains", "disney-princess",
                        "egypt-myth-women", "eur-myth-women", "females-sexy", "females123", "femme-",
                        "greek-myth-women", "greekgod", "indian-myth-women", "jap-myth-women", "jff-fem-",
                        "jff-female-names", "jff-redheads", "jff-superheroine", "jff-xsuperheroine",
                        "jffprincess", "jffsuperheroine", "mklan-female-card", "monsters", "monstres-list",
                        "myh-heroes", "myth-women", "norse-myth-women", "vilsfw", "vsangel", "french-maiden",
                        "english_maids", "wildcard_teacher",
                    )
                ):
                    _add_facet("characters", token)

        # 2. Classify non-scene Ziggart.yaml keys into pure facets
        for y_key in sorted(self._yaml_offsets.keys()):
            if is_full_prompt_wildcard_key(y_key):
                continue
            parts = y_key.split("/")
            if len(parts) < 2 or parts[0] != "ziggart":
                continue
            sec = parts[1]
            token = f"__{y_key}__"
            if sec in ("style", "medium", "art_medium"):
                _add_facet("styles", token)
            elif sec in ("character", "bestiary"):
                _add_facet("characters", token)
            elif sec == "anatomy":
                _add_facet("anatomy", token)
            elif sec == "outfit":
                _add_facet("outfits", token)
            elif sec in ("pose", "action_pose", "expression", "expressions"):
                _add_facet("poses_actions", token)
            elif sec == "environment":
                _add_facet("environments", token)
            elif sec == "lighting":
                _add_facet("lighting", token)
            elif sec == "camera":
                _add_facet("camera", token)

    def get_dropdown_list(self, extra_dir: str = "") -> list[str]:
        self.ensure_loaded(extra_dir=extra_dir)
        return [SELECT_PLACEHOLDER] + self._dropdown_items

    def get_pillar_dropdown(self, pillar_key: str, placeholder: str = NONE_OPTION, extra_dir: str = "") -> list[str]:
        self.ensure_loaded(extra_dir=extra_dir)
        items = self.pillars.get(pillar_key, [])
        return [placeholder] + (items if items else [SELECT_PLACEHOLDER])

    def get_facet_dropdown(
        self,
        facet_key: str,
        include_context_option: bool = True,
        include_random_option: bool = True,
        extra_dir: str = "",
    ) -> list[str]:
        """Returns dropdown choices for a pure component facet (`styles`, `characters`, `anatomy`, `outfits`, `poses_actions`, `environments`, `lighting`, `camera`)."""
        self.ensure_loaded(extra_dir=extra_dir)
        header: list[str] = [NONE_OPTION]
        if include_context_option:
            header.append(FROM_CONTEXT_OPTION)
        if include_random_option:
            header.append(RANDOM_POOL_OPTION)
        items = self.facets.get(facet_key, [])
        return header + items

    def get_theme_list(self, extra_dir: str = "") -> list[str]:
        self.ensure_loaded(extra_dir=extra_dir)
        names = sorted(self.theme_folders.keys())
        return [RANDOM_THEME_OPTION] + names

    def get_universe_list(self, extra_dir: str = "") -> list[str]:
        self.ensure_loaded(extra_dir=extra_dir)
        names = sorted(self.universe_folders.keys())
        return [RANDOM_UNIVERSE_OPTION] + names

    def search_wildcards(self, query: str, limit: int = 100, extra_dir: str = "") -> list[str]:
        """Searches across YAML keys, Collection pillars, and Theme folders matching `query`."""
        self.ensure_loaded(extra_dir=extra_dir)
        q = normalize_wildcard_key(query.strip("_ "))
        if not q:
            return self._dropdown_items[:limit]

        results: list[str] = []
        seen: set[str] = set()

        for item in self._dropdown_items:
            if q in item.lower():
                results.append(item)
                seen.add(item.lower())
                if len(results) >= limit:
                    return results

        for t_name in self.theme_folders:
            if q in t_name.lower():
                for suffix in ("ALL", "Character", "Outfit", "Environment", "PoseAction", "Style", "Creature", "Vehicle"):
                    candidate = f"__Collection/Themes/{t_name}/{suffix}__"
                    if candidate.lower() not in seen:
                        seen.add(candidate.lower())
                        results.append(candidate)
                        if len(results) >= limit:
                            return results

        return results

    def _remember_cache(self, key: str, items: list[str]) -> list[str]:
        if len(self._options_cache) >= self._max_cached_keys:
            oldest = next(iter(self._options_cache))
            self._options_cache.pop(oldest, None)
        self._options_cache[key] = items
        return items

    def _read_txt_file(self, path: str) -> list[str]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return [
                    line.strip()
                    for line in f.read().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        except OSError:
            return []

    def resolve_key_options(self, raw_key: str, rng: random.Random) -> list[str] | None:
        """
        Resolves a wildcard key (e.g. 'Ziggart/scenes/all', 'Collection/formats-wild/.../Female-*',
        'krea2-grok/wallpaper', or 'jff/animals/*') in O(1) time without walking 78,000 files on boot.
        """
        self.ensure_loaded()
        norm_key = normalize_wildcard_key(raw_key)
        if not norm_key:
            return None

        # 0. Glob / wildcard `*` pattern match (YAML, direct filesystem glob, and curated map)
        if "*" in norm_key:
            exact_pat = "^" + re.escape(norm_key).replace(r"\*", ".*") + "$"
            suffix_pat = r"(?:^|/)" + re.escape(norm_key).replace(r"\*", ".*") + "$"
            cre_exact = re.compile(exact_pat)
            cre_suffix = re.compile(suffix_pat)

            matched_yaml = [k for k in self._yaml_offsets if cre_exact.match(k)]
            if not matched_yaml:
                matched_yaml = [k for k in self._yaml_offsets if cre_suffix.search(k)]
            if matched_yaml:
                chosen = rng.choice(matched_yaml)
                return self.resolve_key_options(chosen, rng)

            matched_files = self._glob_files_cache.get(norm_key)
            if matched_files is None:
                clean_rel = raw_key.strip().replace("\\", "/").strip("/_ ")
                found_paths: list[str] = []
                seen_paths: set[str] = set()

                def _add_glob_hits(pattern: str) -> None:
                    for hit in sorted(glob.glob(pattern)):
                        try:
                            if os.path.isfile(hit) and hit.lower().endswith(".txt") and os.path.getsize(hit) > 0:
                                norm_p = os.path.normcase(os.path.abspath(hit))
                                if norm_p not in seen_paths:
                                    seen_paths.add(norm_p)
                                    found_paths.append(hit)
                            elif os.path.isdir(hit):
                                all_p = os.path.join(hit, "ALL.txt")
                                if os.path.isfile(all_p) and os.path.getsize(all_p) > 0:
                                    norm_p = os.path.normcase(os.path.abspath(all_p))
                                    if norm_p not in seen_paths:
                                        seen_paths.add(norm_p)
                                        found_paths.append(all_p)
                        except OSError:
                            continue

                for base_dir in self._dirs:
                    for rel_variant in (clean_rel, norm_key):
                        _add_glob_hits(os.path.join(base_dir, f"{rel_variant}.txt"))
                        _add_glob_hits(os.path.join(base_dir, rel_variant))
                        _add_glob_hits(os.path.join(base_dir, "jff", f"{rel_variant}.txt"))
                        if "/" in rel_variant:
                            base_part = os.path.basename(rel_variant)
                            if "*" in base_part:
                                _add_glob_hits(os.path.join(base_dir, "jff", f"{base_part}.txt"))

                if not found_paths:
                    for k, path in self._curated_txt_map.items():
                        if cre_exact.match(k) or cre_suffix.search(k):
                            try:
                                if os.path.isfile(path) and os.path.getsize(path) > 0:
                                    found_paths.append(path)
                            except OSError:
                                continue

                if not found_paths and norm_key in self.LEGACY_GLOB_ALIAS_MAP:
                    alias_glob = self.LEGACY_GLOB_ALIAS_MAP[norm_key]
                    return self.resolve_key_options(alias_glob, rng)

                matched_files = found_paths
                self._glob_files_cache[norm_key] = matched_files

            if matched_files:
                chosen_file = rng.choice(matched_files)
                items = self._read_txt_file(chosen_file)
                if items:
                    return items
            return None

        if norm_key in self._options_cache:
            return self._options_cache[norm_key]

        # 1. Exact YAML key match (via byte-offset slice!)
        if norm_key in self._yaml_offsets:
            yaml_path, b_start, b_end = self._yaml_offsets[norm_key]
            items = read_yaml_slice_items(yaml_path, b_start, b_end)
            if items:
                return self._remember_cache(norm_key, items)

        # 2. Parent YAML key match (e.g. __Ziggart/scenes__, __krea2-grok__, or __Ziggart__)
        if norm_key in self._yaml_children:
            child_keys = self._yaml_children[norm_key]
            if child_keys:
                chosen_child = rng.choice(child_keys)
                return self.resolve_key_options(chosen_child, rng)

        # 3. Direct O(1) .txt file lookup in all wildcard directories (and jff/ subfolder)
        clean_rel = raw_key.strip().replace("\\", "/").strip("/_ ")
        rel_variants = (
            clean_rel,
            norm_key,
            clean_rel.replace(" ", "-"),
            clean_rel.replace(" ", "_"),
            norm_key.replace(" ", "_"),
        )
        for base_dir in self._dirs:
            for sub_prefix in ("", "jff"):
                search_root = os.path.join(base_dir, sub_prefix) if sub_prefix else base_dir
                for candidate_rel in rel_variants:
                    candidate_path = os.path.join(search_root, f"{candidate_rel}.txt")
                    if os.path.isfile(candidate_path):
                        items = self._read_txt_file(candidate_path)
                        if items:
                            return self._remember_cache(norm_key, items)
                    dir_all_path = os.path.join(search_root, candidate_rel, "ALL.txt")
                    if os.path.isfile(dir_all_path):
                        items = self._read_txt_file(dir_all_path)
                        if items:
                            return self._remember_cache(norm_key, items)

        # 4. Curated lowercase map lookup
        if norm_key in self._curated_txt_map:
            items = self._read_txt_file(self._curated_txt_map[norm_key])
            if items:
                return self._remember_cache(norm_key, items)

        # 5. Suffix match fallback across YAML and curated TXT
        suffix = "/" + norm_key
        suffix_yaml = [k for k in self._yaml_offsets if k.endswith(suffix)]
        if suffix_yaml:
            return self.resolve_key_options(rng.choice(suffix_yaml), rng)

        suffix_txt = [path for k, path in self._curated_txt_map.items() if k.endswith(suffix)]
        if suffix_txt:
            items = self._read_txt_file(rng.choice(suffix_txt))
            if items:
                return items

        # 6. Basename & legacy alias fallback (for friend's jff/ cross-folder references)
        if norm_key in self.LEGACY_ALIAS_MAP:
            return self.resolve_key_options(self.LEGACY_ALIAS_MAP[norm_key], rng)

        if "/" in norm_key:
            basename = norm_key.rsplit("/", 1)[-1]
            if basename in self.LEGACY_ALIAS_MAP:
                return self.resolve_key_options(self.LEGACY_ALIAS_MAP[basename], rng)
            base_suffix = "/" + basename
            base_matches = [
                path
                for k, path in self._curated_txt_map.items()
                if k == basename or k.endswith(base_suffix)
            ]
            if base_matches:
                items = self._read_txt_file(rng.choice(base_matches))
                if items:
                    return items

        # 7. Trailing-digit fallback (e.g. actors1 -> actors when actors1.txt is a 0-byte file)
        stripped_digits = norm_key.rstrip("0123456789")
        if stripped_digits and stripped_digits != norm_key:
            return self.resolve_key_options(stripped_digits, rng)

        return None


WILDCARD_INDEX = ZiggartWildcardIndex()


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _weighted_choice(options: list[str], count: int, rng: random.Random) -> list[str]:
    """Selects `count` unique items from `options`, respecting optional `weight::text` prefixes."""
    if not options or count <= 0:
        return []
    parsed_items: list[str] = []
    weights: list[float] = []
    for opt in options:
        opt_str = str(opt)
        parts = opt_str.split("::", 1)
        if len(parts) == 2 and _is_numeric(parts[0]):
            w = max(0.0, float(parts[0].strip()))
            val = parts[1]
        else:
            w = 1.0
            val = opt_str
        parsed_items.append(val)
        weights.append(w)

    if count >= len(parsed_items):
        shuffled = list(parsed_items)
        rng.shuffle(shuffled)
        return shuffled

    selected: list[str] = []
    pool_items = list(parsed_items)
    pool_weights = list(weights)
    for _ in range(count):
        total_w = sum(pool_weights)
        if total_w <= 0:
            idx = rng.randrange(len(pool_items))
        else:
            r = rng.uniform(0.0, total_w)
            upto = 0.0
            idx = len(pool_items) - 1
            for i, w in enumerate(pool_weights):
                upto += w
                if r <= upto:
                    idx = i
                    break
        selected.append(pool_items.pop(idx))
        pool_weights.pop(idx)
    return selected


def process_wildcards(
    text: str,
    seed: int = 0,
    max_depth: int = 60,
    extra_dir: str = "",
) -> str:
    """
    Expands all wildcard expressions in `text` deterministically using `seed`:
    - `#` comment stripping
    - `N#__wildcard__` quantifiers
    - `{opt1|opt2|2.0::opt3}` and `{1-3$$, $$opt1|opt2|__wildcard__}` dynamic braces
    - `__path/to/wildcard__` recursive TXT and arbitrary-depth YAML resolution
    """
    if not text:
        return ""

    WILDCARD_INDEX.ensure_loaded(extra_dir=extra_dir)
    rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
    current = _strip_comments(text)

    def _expand_single_choice_block(match: re.Match) -> str:
        inner = match.group(1)
        options = inner.split("|")
        select_min, select_max = 1, 1
        sep = " "

        if "$$" in options[0]:
            parts = options[0].split("$$")
            range_str = parts[0].strip()
            rm = re.match(r"^(\d+)?(?:-(\d+))?$", range_str)
            if rm and (rm.group(1) or rm.group(2)):
                low_s, high_s = rm.group(1), rm.group(2)
                low = int(low_s) if low_s else 1
                high = int(high_s) if high_s else (int(low_s) if low_s else 1)
                select_min, select_max = min(low, high), max(low, high)
                if len(parts) == 2:
                    options[0] = parts[1]
                elif len(parts) >= 3:
                    sep = parts[1]
                    options[0] = "$$".join(parts[2:])

                if len(options) == 1:
                    wm = _WILDCARD_TOKEN_RE.fullmatch(options[0].strip())
                    if wm:
                        pool = WILDCARD_INDEX.resolve_key_options(wm.group(1), rng)
                        if pool:
                            options = list(pool)

        max_possible = min(select_max, len(options))
        min_possible = min(select_min, max_possible)
        count = rng.randint(min_possible, max_possible) if max_possible > min_possible else max_possible
        chosen = _weighted_choice(options, count, rng)
        normalized_chosen: list[str] = []
        for c_item in chosen:
            c_str = c_item.strip()
            if c_str.startswith("__") and not c_str.endswith("__") and "__" not in c_str[2:]:
                normalized_chosen.append(f"{c_str}__")
            else:
                normalized_chosen.append(c_item)
        return sep.join(normalized_chosen)

    for _ in range(max_depth):
        changed = False

        def _quant_repl(m: re.Match) -> str:
            nonlocal changed
            changed = True
            q = max(1, min(20, int(m.group("quantifier"))))
            kw = m.group("keyword")
            return "|".join(f"__{kw}__" for _ in range(q))

        current = _QUANTIFIER_RE.sub(_quant_repl, current)

        while True:
            new_text, num_subs = _CHOICE_BLOCK_RE.subn(_expand_single_choice_block, current)
            if num_subs == 0:
                break
            current = new_text
            changed = True

        tokens = list(_WILDCARD_TOKEN_RE.finditer(current))
        if tokens:
            rebuilt: list[str] = []
            last_idx = 0
            for m in tokens:
                rebuilt.append(current[last_idx : m.start()])
                raw_key = m.group(1)
                pool = WILDCARD_INDEX.resolve_key_options(raw_key, rng)
                if pool:
                    chosen_item = _weighted_choice(pool, 1, rng)[0]
                    rebuilt.append(chosen_item)
                    changed = True
                else:
                    rebuilt.append(m.group(0))
                last_idx = m.end()
            rebuilt.append(current[last_idx:])
            current = "".join(rebuilt)

        if not changed:
            break

    if "{" in current or "}" in current:
        current = current.replace("{", "").replace("}", "")

    return current.strip()


def extract_and_strip_loras(text: str) -> tuple[str, list[tuple[str, float, float]]]:
    """
    Extracts `<lora:name:model_weight:clip_weight>` tags from `text` and returns
    (cleaned_text_without_lora_tags, [(lora_name, model_weight, clip_weight), ...]).
    """
    loras: list[tuple[str, float, float]] = []
    seen: set[str] = set()

    for m in _LORA_TAG_RE.finditer(text):
        parts = [p.strip() for p in m.group(1).split(":") if p.strip()]
        if not parts:
            continue
        name = parts[0]
        model_w = 1.0
        clip_w = None
        if len(parts) >= 2 and _is_numeric(parts[1]):
            model_w = float(parts[1])
        if len(parts) >= 3 and _is_numeric(parts[2]):
            clip_w = float(parts[2])
        if clip_w is None:
            clip_w = model_w
        if name not in seen:
            seen.add(name)
            loras.append((name, model_w, clip_w))

    cleaned = _LORA_TAG_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, loras


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


# ---------------------------------------------------------------------------
# Way C Node 1: ZiggartWildcardProcessor (Compact Drop-In Replacement)
# ---------------------------------------------------------------------------


class ZiggartWildcardProcessor:
    """
    Ultra-fast wildcard processor supporting 700+ MB mixed YAML & TXT wildcard collections
    with instant startup indexing, multi-depth YAML support, and a working wildcard selector.
    """

    DESCRIPTION = (
        "High-speed wildcard processor for massive YAML and TXT wildcard collections. "
        "Supports __wildcard__ (any nesting depth), {a|b|2.0::c} weighted choices, "
        "{2$$, $$__wildcard__} multi-selection, and instant wildcard selection from the dropdown."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "process_text"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("processed_text", "populated_text")
    OUTPUT_TOOLTIPS = (
        "Final fully expanded prompt text ready for CLIP / VLM nodes.",
        "Populated prompt text.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        dropdown = WILDCARD_INDEX.get_dropdown_list()
        return {
            "required": {
                "wildcard_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Write your prompt with __wildcard__ or {a|b} syntax. Selecting from 'select_to_add_wildcard' inserts into this field (or resolves directly if left empty).",
                    },
                ),
                "populated_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Displays or overrides the expanded prompt when mode is 'fixed'.",
                    },
                ),
                "mode": (
                    ["populate", "fixed", "reproduce"],
                    {
                        "default": "populate",
                        "tooltip": "populate: expands wildcard_text using seed. fixed: uses populated_text as-is. reproduce: uses populated_text once.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Deterministic seed for wildcard selection.",
                    },
                ),
                "select_to_add_wildcard": (
                    dropdown,
                    {
                        "default": SELECT_PLACEHOLDER,
                        "tooltip": "Choose a wildcard from the indexed list to insert into wildcard_text (or resolve directly if wildcard_text is empty).",
                    },
                ),
            },
            "optional": {
                "search_wildcard_filter": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional keyword search (e.g. 'albator', 'bebop', 'cyberpunk'): if wildcard_text is empty, resolves a matching wildcard.",
                    },
                ),
                "custom_wildcards_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional additional wildcards directory path.",
                    },
                ),
                "refresh_index": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable for one run to rescan YAML/TXT files if you added new files on disk.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def process_text(
        self,
        wildcard_text: str,
        populated_text: str,
        mode: str = "populate",
        seed: int = 0,
        select_to_add_wildcard: str = SELECT_PLACEHOLDER,
        search_wildcard_filter: str = "",
        custom_wildcards_dir: str = "",
        refresh_index: bool = False,
        unique_id: Any = None,
    ) -> tuple[str, str]:
        if refresh_index:
            WILDCARD_INDEX.ensure_loaded(extra_dir=custom_wildcards_dir, force_refresh=True)
        else:
            WILDCARD_INDEX.ensure_loaded(extra_dir=custom_wildcards_dir)

        if mode in ("fixed", "reproduce") and populated_text.strip():
            final_text = process_wildcards(
                populated_text, seed=seed, extra_dir=custom_wildcards_dir
            )
            if mode == "reproduce":
                _send_node_feedback(unique_id, "populated_text", final_text)
            return (final_text, populated_text)

        source_text = wildcard_text.strip()
        if select_to_add_wildcard and select_to_add_wildcard not in (SELECT_PLACEHOLDER, NONE_OPTION):
            if not source_text:
                source_text = select_to_add_wildcard

        if not source_text and search_wildcard_filter.strip():
            matches = WILDCARD_INDEX.search_wildcards(
                search_wildcard_filter.strip(), limit=50, extra_dir=custom_wildcards_dir
            )
            if matches:
                rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
                source_text = rng.choice(matches)

        if not source_text and populated_text.strip():
            source_text = populated_text.strip()

        expanded = process_wildcards(source_text, seed=seed, extra_dir=custom_wildcards_dir)
        _send_node_feedback(unique_id, "populated_text", expanded)
        return (expanded, expanded)


# ---------------------------------------------------------------------------
# Way A Node: ZiggartWildcardProcessorAllInOne (All-in-One Collection Pillars + Mixed)
# ---------------------------------------------------------------------------


class ZiggartWildcardProcessorAllInOne:
    """
    All-in-One Wildcard Processor (Way A) featuring dedicated dropdowns for each
    `Collection/` folder pillar plus a `Mixed / Root` dropdown alongside `wildcard_text`.
    """

    DESCRIPTION = (
        "All-in-One Wildcard Processor with dedicated Collection Pillar dropdowns "
        "(Macro/YAML/Themes, Character/Anatomy, Outfit/Pose, Environment/Vehicle, "
        "Style/Lighting/Camera, and Mixed/Other) plus custom wildcard_text."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "process_all_in_one"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("processed_text", "populated_text", "combined_wildcard_recipe")
    OUTPUT_TOOLTIPS = (
        "Final fully expanded prompt.",
        "Populated prompt text.",
        "The combined unexpanded wildcard template before resolution.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wildcard_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Write custom prompt/wildcards here, or select from the Collection Pillar dropdowns below.",
                    },
                ),
                "populated_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "mode": (["populate", "fixed", "reproduce"], {"default": "populate"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "combine_selected_pillars": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When True, any non-[None] Pillar dropdown selections below are automatically combined with wildcard_text.",
                    },
                ),
                "select_macro_yaml_theme": (
                    WILDCARD_INDEX.get_pillar_dropdown("macro_yaml_themes"),
                    {"default": NONE_OPTION, "tooltip": "Ziggart.yaml scenes, Collection/Themes, and Collection/Combos."},
                ),
                "select_character_anatomy": (
                    WILDCARD_INDEX.get_pillar_dropdown("character_anatomy"),
                    {"default": NONE_OPTION, "tooltip": "Collection/Characters, Anatomy, DuosTeams, and Bestiary."},
                ),
                "select_outfit_pose": (
                    WILDCARD_INDEX.get_pillar_dropdown("outfit_pose"),
                    {"default": NONE_OPTION, "tooltip": "Collection/Outfits, Pose, and Expression."},
                ),
                "select_env_vehicle": (
                    WILDCARD_INDEX.get_pillar_dropdown("env_vehicle"),
                    {"default": NONE_OPTION, "tooltip": "Collection/Environment, IconicPlaces, and Vehicle."},
                ),
                "select_style_camera_light": (
                    WILDCARD_INDEX.get_pillar_dropdown("style_camera_light"),
                    {"default": NONE_OPTION, "tooltip": "Collection/Style, Medium, Lighting, Camera, VFX, and Sheet."},
                ),
                "select_mixed_other": (
                    WILDCARD_INDEX.get_pillar_dropdown("mixed_other"),
                    {"default": NONE_OPTION, "tooltip": "Root .txt pools, ziggart/, Master/, Encours/, jff/, Universes, Canonical, and Index."},
                ),
            },
            "optional": {
                "custom_wildcards_dir": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def process_all_in_one(
        self,
        wildcard_text: str,
        populated_text: str,
        mode: str = "populate",
        seed: int = 0,
        combine_selected_pillars: bool = True,
        select_macro_yaml_theme: str = NONE_OPTION,
        select_character_anatomy: str = NONE_OPTION,
        select_outfit_pose: str = NONE_OPTION,
        select_env_vehicle: str = NONE_OPTION,
        select_style_camera_light: str = NONE_OPTION,
        select_mixed_other: str = NONE_OPTION,
        custom_wildcards_dir: str = "",
        unique_id: Any = None,
    ) -> tuple[str, str, str]:
        WILDCARD_INDEX.ensure_loaded(extra_dir=custom_wildcards_dir)

        if mode in ("fixed", "reproduce") and populated_text.strip():
            final_text = process_wildcards(
                populated_text, seed=seed, extra_dir=custom_wildcards_dir
            )
            if mode == "reproduce":
                _send_node_feedback(unique_id, "populated_text", final_text)
            return (final_text, populated_text, wildcard_text)

        parts: list[str] = []
        if wildcard_text.strip():
            parts.append(wildcard_text.strip())

        selected_pillars = [
            select_macro_yaml_theme,
            select_character_anatomy,
            select_outfit_pose,
            select_env_vehicle,
            select_style_camera_light,
            select_mixed_other,
        ]
        if combine_selected_pillars or not parts:
            for sel in selected_pillars:
                if sel and sel not in (NONE_OPTION, SELECT_PLACEHOLDER) and sel not in parts:
                    parts.append(sel)

        recipe = ", ".join(parts) if parts else populated_text.strip()
        expanded = process_wildcards(recipe, seed=seed, extra_dir=custom_wildcards_dir)
        _send_node_feedback(unique_id, "populated_text", expanded)
        return (expanded, expanded, recipe)


# ---------------------------------------------------------------------------
# Way C Node 2: ZiggartCollectionSceneBuilder (Modular Scene Composer)
# ---------------------------------------------------------------------------


class ZiggartCollectionSceneBuilder:
    """
    Dedicated visual scene composer built around the `Collection/` folder structure
    plus a `Mixed` selector. Outputs both the unexpanded wildcard recipe and the
    resolved prompt.
    """

    DESCRIPTION = (
        "Visual scene builder organized around your Collection folders (Theme/YAML, "
        "Character/Anatomy, Outfit/Pose, Environment/Vehicle, Style/Camera/Lighting, and Mixed). "
        "Outputs both a wildcard_template for Ziggart Fast Wildcard Processor and a resolved prompt."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "build_scene"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("populated_prompt", "wildcard_template")
    OUTPUT_TOOLTIPS = (
        "Fully resolved scene prompt using the given seed.",
        "Unexpanded wildcard template combining your chosen Collection pillars.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_prefix": ("STRING", {"default": "", "multiline": True}),
                "select_macro_yaml_theme": (
                    WILDCARD_INDEX.get_pillar_dropdown("macro_yaml_themes"),
                    {"default": NONE_OPTION},
                ),
                "select_character_anatomy": (
                    WILDCARD_INDEX.get_pillar_dropdown("character_anatomy"),
                    {"default": NONE_OPTION},
                ),
                "select_outfit_pose": (
                    WILDCARD_INDEX.get_pillar_dropdown("outfit_pose"),
                    {"default": NONE_OPTION},
                ),
                "select_env_vehicle": (
                    WILDCARD_INDEX.get_pillar_dropdown("env_vehicle"),
                    {"default": NONE_OPTION},
                ),
                "select_style_camera_light": (
                    WILDCARD_INDEX.get_pillar_dropdown("style_camera_light"),
                    {"default": NONE_OPTION},
                ),
                "select_mixed_other": (
                    WILDCARD_INDEX.get_pillar_dropdown("mixed_other"),
                    {"default": NONE_OPTION},
                ),
                "custom_suffix": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def build_scene(
        self,
        custom_prefix: str = "",
        select_macro_yaml_theme: str = NONE_OPTION,
        select_character_anatomy: str = NONE_OPTION,
        select_outfit_pose: str = NONE_OPTION,
        select_env_vehicle: str = NONE_OPTION,
        select_style_camera_light: str = NONE_OPTION,
        select_mixed_other: str = NONE_OPTION,
        custom_suffix: str = "",
        seed: int = 0,
    ) -> tuple[str, str]:
        segments: list[str] = []
        if custom_prefix.strip():
            segments.append(custom_prefix.strip())

        for sel in (
            select_macro_yaml_theme,
            select_character_anatomy,
            select_outfit_pose,
            select_env_vehicle,
            select_style_camera_light,
            select_mixed_other,
        ):
            if sel and sel not in (NONE_OPTION, SELECT_PLACEHOLDER):
                segments.append(sel)

        if custom_suffix.strip():
            segments.append(custom_suffix.strip())

        template = ", ".join(segments)
        populated = process_wildcards(template, seed=seed)
        return (populated, template)


# ---------------------------------------------------------------------------
# Companion LoRA Encoder: ZiggartWildcardEncode
# ---------------------------------------------------------------------------


class ZiggartWildcardEncode:
    """
    Expands wildcards, loads any embedded `<lora:name:model_w:clip_w>` tags onto MODEL/CLIP,
    and encodes the final prompt into CONDITIONING (supporting BREAK concatenation).
    """

    DESCRIPTION = (
        "Fast wildcard processor + LoRA loader + CLIP text encoder. Compatible with "
        "ImpactWildcardEncode workflows while supporting 700+ MB YAML/TXT collections without lag."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "encode"
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "clip", "conditioning", "populated_text")

    @classmethod
    def INPUT_TYPES(cls):
        dropdown = WILDCARD_INDEX.get_dropdown_list()
        lora_list = [LORA_PLACEHOLDER]
        if folder_paths is not None:
            try:
                lora_list.extend(folder_paths.get_filename_list("loras"))
            except Exception:
                pass
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "wildcard_text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "populated_text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "mode": (["populate", "fixed", "reproduce"], {"default": "populate"}),
                "select_to_add_lora": (lora_list, {"default": LORA_PLACEHOLDER}),
                "select_to_add_wildcard": (dropdown, {"default": SELECT_PLACEHOLDER}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def encode(
        self,
        model: Any,
        clip: Any,
        wildcard_text: str,
        populated_text: str,
        mode: str = "populate",
        select_to_add_lora: str = LORA_PLACEHOLDER,
        select_to_add_wildcard: str = SELECT_PLACEHOLDER,
        seed: int = 0,
        unique_id: Any = None,
    ):
        processor = ZiggartWildcardProcessor()
        expanded, _ = processor.process_text(
            wildcard_text=wildcard_text,
            populated_text=populated_text,
            mode=mode,
            seed=seed,
            select_to_add_wildcard=select_to_add_wildcard,
            unique_id=unique_id,
        )

        clean_prompt, loras = extract_and_strip_loras(expanded)
        if select_to_add_lora and select_to_add_lora != LORA_PLACEHOLDER:
            loras.append((select_to_add_lora, 1.0, 1.0))

        if comfy_nodes is not None and folder_paths is not None and loras:
            avail_loras = folder_paths.get_filename_list("loras")
            for lora_name, model_w, clip_w in loras:
                matched_lora = None
                for candidate in avail_loras:
                    if candidate == lora_name or candidate.endswith(lora_name) or os.path.splitext(candidate)[0].endswith(lora_name):
                        matched_lora = candidate
                        break
                if matched_lora is not None:
                    try:
                        model, clip = comfy_nodes.LoraLoader().load_lora(
                            model, clip, matched_lora, model_w, clip_w
                        )
                    except Exception as exc:
                        logging.warning(f"[Ziggart Wildcards] Failed to load LoRA {matched_lora}: {exc}")

        segments = [seg.strip() for seg in clean_prompt.split("BREAK") if seg.strip()] or [""]
        conditioning = None
        if comfy_nodes is not None:
            encoder = comfy_nodes.CLIPTextEncode()
            for seg in segments:
                cur = encoder.encode(clip=clip, text=seg)[0]
                if conditioning is None:
                    conditioning = cur
                else:
                    conditioning = comfy_nodes.ConditioningConcat().concat(conditioning, cur)[0]
        else:
            tokens = clip.tokenize(segments[0])
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning = [[cond, {"pooled_output": pooled}]]

        return (model, clip, conditioning, clean_prompt)


# ---------------------------------------------------------------------------
# Complementary Node 2: ZiggartCanonicalThemeResolver
# ---------------------------------------------------------------------------


class ZiggartCanonicalThemeResolver:
    """
    Reads `Character-Outfits.json` inside `Collection/Themes/<Theme>/` on-demand
    to pair characters with their authentic canonical outfits, plus matching
    PoseAction, Environment, Vehicle, Creature, and Style from the same theme!
    """

    DESCRIPTION = (
        "Resolves canonical characters and their authentic outfits from Character-Outfits.json "
        "inside your Collection/Themes folders, preventing mismatched random character/outfit rolls. "
        "Outputs both the full scene prompt and individual character prompts for Multi-Character Detailing."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "resolve_theme"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "scene_prompt",
        "character_1_prompt",
        "character_2_prompt",
        "character_3_prompt",
        "theme_metadata_json",
    )
    OUTPUT_TOOLTIPS = (
        "Complete canonical scene prompt combining character(s), outfits, pose, environment, and style.",
        "Canonical description + outfit for Character 1 (wire into Multi-Character Detailer).",
        "Canonical description + outfit for Character 2 (when Duo or Trio is selected).",
        "Canonical description + outfit for Character 3 (when Trio is selected).",
        "JSON metadata detailing the resolved theme, characters, and components.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "theme_selection": (
                    WILDCARD_INDEX.get_theme_list(),
                    {"default": RANDOM_THEME_OPTION, "tooltip": "Choose a specific theme or [Random Theme]."},
                ),
                "cast_mode": (
                    [
                        "Single (Any Gender)",
                        "Single (Female)",
                        "Single (Male)",
                        "Duo (Any 2 Characters)",
                        "Duo (Female + Male)",
                        "Duo (Female + Female)",
                        "Trio (3 Characters)",
                    ],
                    {"default": "Single (Any Gender)"},
                ),
                "include_pose_action": ("BOOLEAN", {"default": True}),
                "include_environment": ("BOOLEAN", {"default": True}),
                "include_vehicle": ("BOOLEAN", {"default": False}),
                "include_creature": ("BOOLEAN", {"default": False}),
                "include_style": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "theme_search_filter": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional keyword filter when theme_selection is [Random Theme] (e.g. 'cyberpunk', 'albator', 'bebop').",
                    },
                ),
            },
        }

    def resolve_theme(
        self,
        theme_selection: str = RANDOM_THEME_OPTION,
        cast_mode: str = "Single (Any Gender)",
        include_pose_action: bool = True,
        include_environment: bool = True,
        include_vehicle: bool = False,
        include_creature: bool = False,
        include_style: bool = True,
        seed: int = 0,
        theme_search_filter: str = "",
    ) -> tuple[str, str, str, str, str]:
        WILDCARD_INDEX.ensure_loaded()
        rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)

        theme_name = theme_selection
        theme_dir = WILDCARD_INDEX.theme_folders.get(theme_name)

        if theme_name == RANDOM_THEME_OPTION or not theme_dir:
            candidates = sorted(WILDCARD_INDEX.theme_folders.keys())
            if theme_search_filter.strip():
                q = theme_search_filter.strip().lower()
                filtered = [t for t in candidates if q in t.lower()]
                if filtered:
                    candidates = filtered
            if candidates:
                theme_name = rng.choice(candidates)
                theme_dir = WILDCARD_INDEX.theme_folders[theme_name]

        if not theme_dir or not os.path.isdir(theme_dir):
            return ("", "", "", "", json.dumps({"error": "No theme folder found"}))

        def _pick_txt(filename: str) -> str:
            p = os.path.join(theme_dir, filename)
            if os.path.isfile(p):
                lines = WILDCARD_INDEX._read_txt_file(p)
                if lines:
                    return rng.choice(lines)
            return ""

        # Read Character-Outfits.json if available
        json_path = os.path.join(theme_dir, "Character-Outfits.json")
        characters_data: list[dict[str, Any]] = []
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8", errors="replace") as jf:
                    parsed = json.load(jf)
                    if isinstance(parsed, dict) and isinstance(parsed.get("characters"), list):
                        characters_data = parsed["characters"]
            except Exception:
                characters_data = []

        # Determine how many characters and gender filters
        if "Trio" in cast_mode:
            target_count = 3
        elif "Duo" in cast_mode:
            target_count = 2
        else:
            target_count = 1

        char_prompts: list[str] = []
        if characters_data:
            females = [c for c in characters_data if str(c.get("gender", "")).lower() == "female"]
            males = [c for c in characters_data if str(c.get("gender", "")).lower() == "male"]

            selected_chars: list[dict[str, Any]] = []
            if cast_mode == "Single (Female)" and females:
                selected_chars = [rng.choice(females)]
            elif cast_mode == "Single (Male)" and males:
                selected_chars = [rng.choice(males)]
            elif cast_mode == "Duo (Female + Male)" and females and males:
                selected_chars = [rng.choice(females), rng.choice(males)]
            elif cast_mode == "Duo (Female + Female)" and len(females) >= 2:
                selected_chars = rng.sample(females, 2)
            else:
                pool = list(characters_data)
                rng.shuffle(pool)
                selected_chars = pool[: min(target_count, len(pool))]

            for ch in selected_chars:
                desc = str(ch.get("description", "")).strip()
                outfits = ch.get("canonical_outfits") or []
                outfit = rng.choice(outfits).strip() if outfits else _pick_txt("Outfit.txt")
                if desc and outfit:
                    char_prompts.append(f"{desc}, wearing {outfit}")
                elif desc:
                    char_prompts.append(desc)
        else:
            # Fallback to Character.txt + Outfit.txt
            for _ in range(target_count):
                c_txt = _pick_txt("Character.txt")
                o_txt = _pick_txt("Outfit.txt")
                if c_txt and o_txt:
                    char_prompts.append(f"{c_txt}, wearing {o_txt}")
                elif c_txt:
                    char_prompts.append(c_txt)

        while len(char_prompts) < 3:
            char_prompts.append("")

        scene_parts = [cp for cp in char_prompts[:target_count] if cp]
        pose_str = _pick_txt("PoseAction.txt") if include_pose_action else ""
        env_str = _pick_txt("Environment.txt") if include_environment else ""
        veh_str = _pick_txt("Vehicle.txt") if include_vehicle else ""
        cre_str = _pick_txt("Creature.txt") if include_creature else ""
        sty_str = _pick_txt("Style.txt") if include_style else ""

        if pose_str:
            scene_parts.append(pose_str)
        if veh_str:
            scene_parts.append(f"with {veh_str}")
        if cre_str:
            scene_parts.append(f"accompanied by {cre_str}")
        if env_str:
            scene_parts.append(f"in {env_str}")
        if sty_str:
            scene_parts.append(sty_str)

        scene_prompt = process_wildcards(", ".join(scene_parts), seed=seed)
        meta = {
            "theme": theme_name,
            "cast_mode": cast_mode,
            "characters": [cp for cp in char_prompts if cp],
            "pose": pose_str,
            "environment": env_str,
            "vehicle": veh_str,
            "creature": cre_str,
            "style": sty_str,
        }
        return (
            scene_prompt,
            char_prompts[0],
            char_prompts[1],
            char_prompts[2],
            json.dumps(meta, indent=2),
        )


# ---------------------------------------------------------------------------
# Complementary Node 3: ZiggartWildcardVLMEnhancer
# ---------------------------------------------------------------------------


class ZiggartWildcardVLMEnhancer:
    """
    Takes raw `populated_text` from a Wildcard Processor and uses the local GGUF LLM/VLM
    to harmonize stitched wildcard fragments into a cohesive, natural-language cinematic
    prompt optimized for Krea 2 Turbo / Flux while strictly preserving character/outfit locks.
    """

    DESCRIPTION = (
        "Harmonizes multi-pool wildcard expansions into cohesive natural-language prose "
        "using your local GGUF model while preserving canonical character, outfit, and style details."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "harmonize"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("harmonized_prompt", "raw_populated_text")
    OUTPUT_TOOLTIPS = (
        "Harmonized, cohesive generation prompt ready for CLIPTextEncode.",
        "Original raw wildcard text before VLM harmonization.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = _gguf_files()
        return {
            "required": {
                "wildcard_or_populated_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Input prompt or unexpanded wildcard string.",
                    },
                ),
                "model": (models, {"tooltip": "Local GGUF model from models/LLM/GGUF."}),
                "mmproj": (projectors, {"tooltip": "Matching mmproj GGUF."}),
                "harmonization_style": (
                    [
                        "Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)",
                        "Cinematic Director Shot (Rich Lighting & Camera)",
                        "Concise High-Impact Tags & Prose Hybrid",
                        "Custom Instruction",
                    ],
                    {"default": "Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "free_vram_after": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_instruction": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Used when harmonization_style is 'Custom Instruction' (or appended as extra rules).",
                    },
                ),
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Optional visual reference image to guide lighting or mood harmonization."},
                ),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.5, "step": 0.05}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 256}),
                "context_size": ("INT", {"default": 4096, "min": 1024, "max": 32768}),
                "model_path_override": ("STRING", {"default": ""}),
                "mmproj_path_override": ("STRING", {"default": ""}),
                "vlm_config": ("GGUF_VL_MODEL",),
            },
        }

    def harmonize(
        self,
        wildcard_or_populated_text: str,
        model: str,
        mmproj: str,
        harmonization_style: str = "Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)",
        seed: int = 0,
        free_vram_after: bool = True,
        custom_instruction: str = "",
        reference_image: Any = None,
        max_tokens: int = 512,
        temperature: float = 0.25,
        gpu_layers: int = -1,
        context_size: int = 4096,
        model_path_override: str = "",
        mmproj_path_override: str = "",
        vlm_config: ModelConfig | None = None,
    ) -> tuple[str, str]:
        raw_expanded = process_wildcards(wildcard_or_populated_text, seed=seed)
        if not raw_expanded.strip():
            return ("", "")

        if isinstance(vlm_config, ModelConfig):
            config = vlm_config
        else:
            model_path = _resolve_gguf(model, model_path_override, "model GGUF")
            mmproj_path = _resolve_gguf(mmproj, mmproj_path_override, "mmproj GGUF")
            config = ModelConfig(
                model_path=model_path,
                mmproj_path=mmproj_path,
                context_size=int(context_size),
                gpu_layers=int(gpu_layers),
                threads=max(1, (os.cpu_count() or 4) // 2),
                flash_attention=True,
                mmproj_on_gpu=True,
            )

        style_directives = {
            "Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)": (
                "Rewrite the following wildcard-expanded prompt into fluent, vivid natural-language prose "
                "optimized for Krea 2 Turbo and Flux. STRICTLY preserve all character names, physical traits, "
                "canonical outfits, objects, and artistic style medium. Resolve any duplicate words or awkward "
                "comma-spliced fragments. Output ONLY the final prompt."
            ),
            "Cinematic Director Shot (Rich Lighting & Camera)": (
                "Transform the following wildcard-expanded prompt into a cohesive cinematic shot description. "
                "Keep every character identity and outfit detail intact, and unify the camera angle, depth of field, "
                "atmospheric lighting, and color palette. Output ONLY the final prompt."
            ),
            "Concise High-Impact Tags & Prose Hybrid": (
                "Condense and organize the following wildcard-expanded prompt into a punchy, high-density generation "
                "prompt with zero redundancy while keeping all character and outfit specifics. Output ONLY the final prompt."
            ),
            "Custom Instruction": custom_instruction.strip() or "Harmonize this prompt into a cohesive image generation prompt. Output only the prompt.",
        }
        directive = style_directives.get(harmonization_style, style_directives["Krea 2 Turbo / Flux Natural Prose (Strict Character Lock)"])
        if custom_instruction.strip() and harmonization_style != "Custom Instruction":
            directive += f"\nAdditional requirement: {custom_instruction.strip()}"

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"{directive}\n\nRaw Wildcard Prompt:\n{raw_expanded}",
            }
        ]
        if reference_image is not None:
            frame = reference_image[0] if hasattr(reference_image, "ndim") and reference_image.ndim == 4 else reference_image
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_uri(frame, max_dimension=768)},
                }
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert diffusion prompt harmonizer. You never drop character identity or "
                    "canonical outfit details. Output only the final prompt without commentary."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        try:
            MODEL_MANAGER.load(config, unload_comfy_models=True)
            harmonized = MODEL_MANAGER.chat(
                messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=0.9,
                top_k=30,
                seed=int(seed & 0x7FFFFFFF),
                enable_thinking=False,
            )
            return (harmonized or raw_expanded, raw_expanded)
        except EmbeddedLlamaError as exc:
            raise RuntimeError(f"Wildcard-to-VLM Prompt Harmonizer failed: {exc}") from exc
        finally:
            if free_vram_after:
                MODEL_MANAGER.unload()
                release_comfy_vram()


# ---------------------------------------------------------------------------
# Complementary Node 5: ZiggartWildcardRecipeSaver
# ---------------------------------------------------------------------------


class ZiggartWildcardRecipeSaver:
    """
    Saves final images (and an optional side-by-side Base vs Refined comparison grid)
    with full wildcard recipe metadata (raw template, expanded populated_text, VLM/Detailer
    prompts, and seed) embedded in PNG chunks and an optional JSON sidecar file.
    """

    DESCRIPTION = (
        "Saves generated images with embedded PNG metadata and a JSON recipe sidecar "
        "recording the exact wildcard template, expanded prompt, detailer prompts, and seed. "
        "Optionally generates a side-by-side comparison grid when base_image is connected."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "save_recipe"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("output_image_or_grid", "recipe_summary", "saved_paths_json")
    OUTPUT_TOOLTIPS = (
        "The saved image (or side-by-side comparison grid if base_image is provided and build_comparison_grid is True).",
        "Human-readable summary of the wildcard recipe and seed.",
        "JSON array of saved file paths.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Final refined image batch to save."}),
                "filename_prefix": ("STRING", {"default": "Ziggart/Ziggart_Recipe"}),
                "wildcard_template": ("STRING", {"default": "", "multiline": True}),
                "populated_text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "save_json_sidecar": ("BOOLEAN", {"default": True}),
                "build_comparison_grid": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "base_image": ("IMAGE", {"tooltip": "Optional image before detailing for side-by-side comparison."}),
                "detailer_or_vlm_notes": ("STRING", {"default": "", "multiline": True}),
                "output_dir_override": ("STRING", {"default": ""}),
            },
        }

    def save_recipe(
        self,
        image: torch.Tensor,
        filename_prefix: str = "Ziggart/Ziggart_Recipe",
        wildcard_template: str = "",
        populated_text: str = "",
        seed: int = 0,
        save_json_sidecar: bool = True,
        build_comparison_grid: bool = False,
        base_image: torch.Tensor | None = None,
        detailer_or_vlm_notes: str = "",
        output_dir_override: str = "",
    ) -> tuple[torch.Tensor, str, str]:
        if output_dir_override and output_dir_override.strip():
            out_root = Path(os.path.expandvars(os.path.expanduser(output_dir_override.strip())))
        elif folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
            out_root = Path(folder_paths.get_output_directory())
        else:
            out_root = Path(__file__).resolve().parent / "output"

        prefix_clean = filename_prefix.strip().replace("\\", "/").strip("/") or "Ziggart_Recipe"
        target_path_prefix = out_root / prefix_clean
        target_dir = target_path_prefix.parent
        base_stem = target_path_prefix.name
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        batch_size = image.shape[0]
        output_tensors: list[torch.Tensor] = []
        saved_files: list[str] = []

        recipe_data = {
            "seed": int(seed),
            "wildcard_template": wildcard_template,
            "populated_text": populated_text,
            "detailer_or_vlm_notes": detailer_or_vlm_notes,
            "timestamp": timestamp,
        }

        for b in range(batch_size):
            refined_frame = image[b : b + 1]
            if build_comparison_grid and base_image is not None:
                b_idx = min(b, base_image.shape[0] - 1)
                base_frame = base_image[b_idx : b_idx + 1]
                if base_frame.shape[1:3] != refined_frame.shape[1:3]:
                    base_frame = F.interpolate(
                        base_frame.permute(0, 3, 1, 2),
                        size=(refined_frame.shape[1], refined_frame.shape[2]),
                        mode="bicubic",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                c_min = min(base_frame.shape[-1], refined_frame.shape[-1])
                grid_frame = torch.cat([base_frame[..., :c_min], refined_frame[..., :c_min]], dim=2)
                output_tensors.append(grid_frame)
                save_tensor = grid_frame[0]
            else:
                output_tensors.append(refined_frame)
                save_tensor = refined_frame[0]

            np_arr = (np.clip(save_tensor.detach().cpu().float().numpy(), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            pil_img = Image.fromarray(np_arr)

            png_info = PngInfo()
            png_info.add_text("ziggart_wildcard_template", wildcard_template)
            png_info.add_text("ziggart_populated_text", populated_text)
            png_info.add_text("ziggart_seed", str(seed))
            if detailer_or_vlm_notes:
                png_info.add_text("ziggart_notes", detailer_or_vlm_notes)

            file_stem = f"{base_stem}_{timestamp}_{b:02d}"
            png_file = target_dir / f"{file_stem}.png"
            pil_img.save(png_file, format="PNG", pnginfo=png_info)
            saved_files.append(str(png_file))

            if save_json_sidecar:
                json_file = target_dir / f"{file_stem}.json"
                with open(json_file, "w", encoding="utf-8") as jf:
                    json.dump({**recipe_data, "frame_index": b, "image_file": png_file.name}, jf, indent=2)
                saved_files.append(str(json_file))

        summary = (
            f"Seed: {seed}\n"
            f"Template: {wildcard_template}\n"
            f"Populated: {populated_text}\n"
            f"Notes: {detailer_or_vlm_notes}"
        ).strip()

        return (torch.cat(output_tensors, dim=0), summary, json.dumps(saved_files, indent=2))


# ---------------------------------------------------------------------------
# Krea2 (Qwen3VL-4B) Modular & Master Prompt Director Nodes
# ---------------------------------------------------------------------------

MOOD_FILTER_OPTIONS = [
    "[Any / No Filter]",
    "SFW Only (Exclude NSFW / Erotic / Intimate)",
    "NSFW / Sensual / Glamour",
    "Fantasy / Mythology / Medieval",
    "Sci-Fi / Cyberpunk / Mecha",
    "Anime / Manga / Cel Animation",
    "Dark / Gothic / Horror",
    "Urban / Modern / Streetwear",
]

OUTFIT_LINK_MODES = [
    "Canonical (1:1 Strict Character Outfit)",
    "Compatible (Same Theme/Universe + Gender)",
    "Free (All Outfit Wildcards across Folders)",
    "Nude / Unclothed (All Anatomy Visible)",
]

KREA2_COMPILER_POLICIES = [
    "scoped (Parentheses Blocks + Count Anchor)",
    "balanced (Krea2 v2 9-Block Linear Prose)",
    "compact (Solo Focus / Condensed Environment)",
    "rich (Full Detail + Quality Constraints)",
]

DUO_ACTION_PRESET_OPTIONS = [
    NONE_OPTION,
    FROM_CONTEXT_OPTION,
    RANDOM_POOL_OPTION,
    *list(DUO_INTERACTIONS_BANK.keys()),
]

_NSFW_TOKEN_RE = re.compile(
    r"(?:nsfw|erotic|hentai|sex|xxx|lingerie|bikini|microbikini|nude|naked|nudity|boudoir|intimate|vulva|penis|pussies|cum|alluring|sugg-pose|striptease|bdsm|wetlook|seethrough)",
    re.I,
)


def _matches_mood_and_keyword(
    candidate_token_or_line: str,
    mood_filter: str = "[Any / No Filter]",
    custom_keyword: str = "",
) -> bool:
    """Checks if a wildcard token or line matches the active mood_filter and optional custom_keyword."""
    low = candidate_token_or_line.lower()
    if mood_filter == "SFW Only (Exclude NSFW / Erotic / Intimate)":
        if _NSFW_TOKEN_RE.search(low) and "sfw" not in low:
            return False
    elif mood_filter == "NSFW / Sensual / Glamour":
        if not (_NSFW_TOKEN_RE.search(low) or any(w in low for w in ("glamour", "sensual", "swimsuit", "swimwear", "plage", "sexy"))):
            return False
    elif mood_filter == "Fantasy / Mythology / Medieval":
        if not any(w in low for w in ("fantasy", "myth", "medieval", "elf", "elves", "dwarv", "dragon", "magic", "sorcer", "armors", "armor", "ancient", "goddess", "greekgod")):
            return False
    elif mood_filter == "Sci-Fi / Cyberpunk / Mecha":
        if not any(w in low for w in ("scifi", "sci-fi", "cyber", "mecha", "robot", "alien", "space", "cosmic", "synthwave", "neon", "future")):
            return False
    elif mood_filter == "Anime / Manga / Cel Animation":
        if not any(w in low for w in ("anime", "manga", "cel", "toonz", "sentai", "magical-girl", "sexymagical", "ova", "shonen", "seinen", "shojo")):
            return False
    elif mood_filter == "Dark / Gothic / Horror":
        if not any(w in low for w in ("dark", "gothic", "horror", "horreur", "vampir", "asylum", "underworld", "monster", "monstres", "cryptid", "noir")):
            return False
    elif mood_filter == "Urban / Modern / Streetwear":
        if not any(w in low for w in ("urban", "city", "street", "casual", "sport", "gym", "tennis", "modern", "fashion", "academy", "uniform", "maid")):
            return False

    if custom_keyword and custom_keyword.strip():
        kw_parts = [k.strip().lower() for k in custom_keyword.split(",") if k.strip()]
        if kw_parts and not any(k in low for k in kw_parts):
            return False
    return True


def _pick_filtered_wildcard_line(
    facet_key: str,
    selected_option: str,
    rng: random.Random,
    *,
    context_bundle: dict[str, Any] | None = None,
    context_field: str = "",
    mood_filter: str = "[Any / No Filter]",
    custom_keyword: str = "",
    gender_filter: str = "Auto / Any",
    exclude_full_prompts: bool = True,
    extra_dir: str = "",
) -> tuple[str, str]:
    """
    Resolves a single component facet into `(expanded_text, unexpanded_token_or_source)`.
    Supports:
    - `[None]` -> `("", "")`
    - `[From Context (Theme / Universe)]` -> picks from `context_bundle[context_field]`, falling back to random facet pool
    - `[Random From Category]` -> picks a matching wildcard from `WILDCARD_INDEX.facets[facet_key]`
    - Specific `__wildcard__` token -> resolves that wildcard with full-scene line filtering.
    """
    WILDCARD_INDEX.ensure_loaded(extra_dir=extra_dir)

    if not selected_option or selected_option == NONE_OPTION:
        return ("", "")

    # 1. From Context (Theme / Universe)
    if selected_option == FROM_CONTEXT_OPTION:
        if context_bundle and context_field:
            ctx_items = context_bundle.get(context_field) or []
            if isinstance(ctx_items, list) and ctx_items:
                chosen_ctx = str(rng.choice(ctx_items)).strip()
                expanded_ctx = process_wildcards(chosen_ctx, seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
                return (expanded_ctx, chosen_ctx)
        # Fallback if context has no items for this field (e.g. Lighting or Camera which are not in Theme folders)
        selected_option = RANDOM_POOL_OPTION

    # 2. Random Pool from Category
    chosen_token = selected_option
    if selected_option == RANDOM_POOL_OPTION:
        pool = list(WILDCARD_INDEX.facets.get(facet_key, []))
        if exclude_full_prompts:
            pool = [t for t in pool if not is_full_prompt_wildcard_key(t)]
        if gender_filter in ("Female", "Male", "Creature"):
            g_low = gender_filter.lower()
            other_genders = {"female", "male", "creature"} - {g_low}
            g_filtered = [
                t for t in pool
                if g_low in t.lower() or not any(f"/{og}/" in t.lower() or f"{og}-" in t.lower() for og in other_genders)
            ]
            if g_filtered:
                pool = g_filtered
        if mood_filter != "[Any / No Filter]" or custom_keyword.strip():
            m_filtered = [t for t in pool if _matches_mood_and_keyword(t, mood_filter, custom_keyword)]
            if m_filtered:
                pool = m_filtered
        if not pool:
            return ("", "")
        chosen_token = rng.choice(pool)

    # 3. Resolve the chosen wildcard token and filter out full-scene lines if exclude_full_prompts is True
    raw_key = chosen_token.strip().strip("_")
    options = WILDCARD_INDEX.resolve_key_options(raw_key, rng)
    if options:
        valid_lines = list(options)
        if exclude_full_prompts:
            non_scene = [ln for ln in valid_lines if not is_full_scene_line(ln)]
            if non_scene:
                valid_lines = non_scene
        if mood_filter == "SFW Only (Exclude NSFW / Erotic / Intimate)":
            sfw_lines = [ln for ln in valid_lines if not _NSFW_TOKEN_RE.search(ln)]
            if sfw_lines:
                valid_lines = sfw_lines
        if custom_keyword.strip():
            kw_lines = [ln for ln in valid_lines if _matches_mood_and_keyword(ln, "[Any / No Filter]", custom_keyword)]
            if kw_lines:
                valid_lines = kw_lines
        chosen_line = _weighted_choice(valid_lines, 1, rng)[0]
        expanded = process_wildcards(chosen_line, seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        return (expanded, chosen_token)

    # Fallback direct wildcard expansion
    expanded = process_wildcards(chosen_token, seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
    return (expanded, chosen_token)


def _resolve_character_unit(
    *,
    slot_role: str,
    context: dict[str, Any] | None,
    character_source: str,
    gender_filter: str,
    character_wildcard: str,
    custom_character_text: str,
    outfit_link_mode: str,
    outfit_wildcard: str,
    custom_outfit_text: str,
    enable_anatomy: bool,
    auto_filter_anatomy_by_outfit: bool,
    anatomy_wildcard: str,
    custom_anatomy_text: str,
    individual_action_wildcard: str,
    custom_action_text: str,
    seed: int,
    used_char_ids: set[str] | None = None,
    extra_dir: str = "",
) -> tuple[CharacterSlotData, str, str]:
    """
    Core builder logic shared between `ZiggartCharacterUnitBuilder` and `ZiggartKrea2MasterPromptDirector`.
    Resolves character identity, applies 1:1 Canonical / Compatible / Free / Nude outfit linking,
    and runs automatic Outfit -> Anatomy occlusion filtering.
    """
    WILDCARD_INDEX.ensure_loaded(extra_dir=extra_dir)
    rng = random.Random((int(seed) ^ (0xA5A5A5A5 if slot_role == "Secondary" else 0x13579BDF)) & 0xFFFFFFFFFFFFFFFF)

    ctx_bundle: dict[str, Any] | None = context.get("bundle") if isinstance(context, dict) else None
    mood_filter = str(context.get("mood_filter", "[Any / No Filter]")) if isinstance(context, dict) else "[Any / No Filter]"
    kw_filter = str(context.get("keyword_filter", "")) if isinstance(context, dict) else ""
    exclude_full = bool(context.get("exclude_full_prompts", True)) if isinstance(context, dict) else True
    theme_or_univ_name = str(context.get("name", "Free Crossover")) if isinstance(context, dict) else "Free Crossover"

    resolved_gender = gender_filter if gender_filter in ("Female", "Male", "Creature") else "Female"
    char_desc = ""
    selected_char_obj: dict[str, Any] | None = None
    unexpanded_char_token = ""

    # 1. Resolve Character Identity
    if custom_character_text and custom_character_text.strip():
        unexpanded_char_token = custom_character_text.strip()
        char_desc = process_wildcards(unexpanded_char_token, seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        if gender_filter == "Auto / Any":
            c_low = char_desc.lower()
            if any(w in c_low for w in ("boy", "man", "male", "he ", "his ", "gentleman", "warrior man")):
                resolved_gender = "Male"
            elif any(w in c_low for w in ("beast", "creature", "monster", "dragon", "wyvern")):
                resolved_gender = "Creature"
    elif (
        character_source == "From Context (Theme / Universe)"
        or (character_wildcard == FROM_CONTEXT_OPTION and ctx_bundle)
    ) and ctx_bundle and ctx_bundle.get("characters"):
        candidates = list(ctx_bundle["characters"])
        if gender_filter in ("Female", "Male"):
            g_cands = [c for c in candidates if str(c.get("gender", "")).lower() == gender_filter.lower()]
            if g_cands:
                candidates = g_cands
        elif gender_filter == "Creature" and ctx_bundle.get("creatures"):
            cre_line = rng.choice(ctx_bundle["creatures"])
            char_desc = process_wildcards(cre_line, seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
            resolved_gender = "Creature"
            candidates = []

        if candidates:
            if used_char_ids:
                unused = [c for c in candidates if str(c.get("id", c.get("description", ""))) not in used_char_ids]
                if unused:
                    candidates = unused
            selected_char_obj = rng.choice(candidates)
            if used_char_ids is not None:
                used_char_ids.add(str(selected_char_obj.get("id", selected_char_obj.get("description", ""))))
            char_desc = str(selected_char_obj.get("description", "")).strip()
            resolved_gender = str(selected_char_obj.get("gender", resolved_gender)).capitalize()
            unexpanded_char_token = f"[{theme_or_univ_name}:{selected_char_obj.get('id', 'Char')}]"
    else:
        wc_choice = character_wildcard
        if wc_choice in (NONE_OPTION, FROM_CONTEXT_OPTION):
            wc_choice = RANDOM_POOL_OPTION
        char_desc, unexpanded_char_token = _pick_filtered_wildcard_line(
            "characters",
            wc_choice,
            rng,
            context_bundle=ctx_bundle,
            context_field="characters",
            mood_filter=mood_filter,
            custom_keyword=kw_filter,
            gender_filter=gender_filter,
            exclude_full_prompts=exclude_full,
            extra_dir=extra_dir,
        )
        if gender_filter == "Auto / Any":
            tok_low = f"{unexpanded_char_token} {char_desc}".lower()
            if any(w in tok_low for w in ("/male", "male-", "actors", " man,", " boy,")):
                resolved_gender = "Male"
            elif any(w in tok_low for w in ("creature", "bestiary", "monster", "dragon")):
                resolved_gender = "Creature"
            else:
                resolved_gender = "Female"

    # Note: some jff character cards (like femme-krea-list.txt) include a built-in outfit after the character trait.
    # If the user selects an explicit outfit mode or override, we keep `char_desc` clean.

    # 2. Resolve Outfit according to `outfit_link_mode`
    outfit_desc = ""
    unexpanded_outfit_token = ""
    mode_low = outfit_link_mode.lower()

    if "nude" in mode_low:
        outfit_desc = "nude, completely unclothed"
        unexpanded_outfit_token = "nude"
    elif custom_outfit_text and custom_outfit_text.strip():
        unexpanded_outfit_token = custom_outfit_text.strip()
        outfit_desc = process_wildcards(unexpanded_outfit_token, seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
    elif "canonical" in mode_low and selected_char_obj is not None and ctx_bundle is not None:
        choices = get_theme_outfit_choices(ctx_bundle, selected_char_obj, "canonical")
        if choices:
            outfit_desc = process_wildcards(rng.choice(choices), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
            unexpanded_outfit_token = f"[{theme_or_univ_name}:CanonicalOutfit]"
        else:
            # Fallback to compatible outfits in the same theme
            compat_choices = get_theme_outfit_choices(ctx_bundle, selected_char_obj, "compatible")
            if compat_choices:
                outfit_desc = process_wildcards(rng.choice(compat_choices), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
                unexpanded_outfit_token = f"[{theme_or_univ_name}:CompatibleOutfit]"
    elif "compatible" in mode_low and ctx_bundle is not None and outfit_wildcard in (NONE_OPTION, FROM_CONTEXT_OPTION):
        dummy_char = selected_char_obj or {"gender": resolved_gender}
        compat_choices = get_theme_outfit_choices(ctx_bundle, dummy_char, "compatible")
        if compat_choices:
            outfit_desc = process_wildcards(rng.choice(compat_choices), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
            unexpanded_outfit_token = f"[{theme_or_univ_name}:{resolved_gender}-Outfit]"

    # If still empty (e.g. Free mode, or Compatible with a wildcard selected, or no theme context):
    if not outfit_desc and "nude" not in mode_low:
        o_choice = outfit_wildcard
        if o_choice == NONE_OPTION and ("compatible" in mode_low or "free" in mode_low or "canonical" in mode_low):
            o_choice = RANDOM_POOL_OPTION
        if o_choice != NONE_OPTION:
            g_for_outfit = resolved_gender if ("compatible" in mode_low or "canonical" in mode_low) else gender_filter
            outfit_desc, unexpanded_outfit_token = _pick_filtered_wildcard_line(
                "outfits",
                o_choice,
                rng,
                context_bundle=ctx_bundle,
                context_field="outfits",
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                gender_filter=g_for_outfit,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )

    outfit_desc = clean_outfit_prefix(outfit_desc)

    # 3. Analyze Outfit Occlusion -> Filter Anatomy automatically!
    occlusion = analyze_outfit_occlusion(outfit_desc, unexpanded_outfit_token)
    anatomy_desc = ""

    # Block intimate anatomy if character age is minor
    is_minor = bool(selected_char_obj and str(selected_char_obj.get("age", "adult")).lower() == "minor")
    if is_minor:
        occlusion = OutfitOcclusionReport(
            outfit_text=outfit_desc,
            is_nude=False,
            allowed_zones={z: False for z in ANATOMY_ZONES},
            blocked_zones=ANATOMY_ZONES,
            visible_zones=(),
            reason_summary="Minor character age policy: all intimate/body anatomy zones blocked.",
        )

    if enable_anatomy and not is_minor:
        if custom_anatomy_text and custom_anatomy_text.strip():
            raw_anat = process_wildcards(custom_anatomy_text.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
            anatomy_desc = (
                filter_anatomy_text_by_occlusion(raw_anat, occlusion, gender=resolved_gender)
                if auto_filter_anatomy_by_outfit
                else raw_anat
            )
        elif anatomy_wildcard not in (NONE_OPTION, FROM_CONTEXT_OPTION, RANDOM_POOL_OPTION):
            # Specific anatomy wildcard chosen by user: check if its zone is allowed by the outfit
            zone = classify_anatomy_wildcard_zone(anatomy_wildcard)
            if auto_filter_anatomy_by_outfit and zone is not None and not occlusion.allowed_zones.get(zone, True):
                anatomy_desc = ""
            else:
                raw_anat, _ = _pick_filtered_wildcard_line(
                    "anatomy",
                    anatomy_wildcard,
                    rng,
                    mood_filter=mood_filter,
                    custom_keyword=kw_filter,
                    gender_filter=resolved_gender,
                    exclude_full_prompts=exclude_full,
                    extra_dir=extra_dir,
                )
                anatomy_desc = (
                    filter_anatomy_text_by_occlusion(raw_anat, occlusion, gender=resolved_gender)
                    if auto_filter_anatomy_by_outfit
                    else raw_anat
                )
        else:
            # Automatic multi-zone anatomy assembly based on visible zones & gender!
            anat_Pool = list(WILDCARD_INDEX.facets.get("anatomy", []))
            g_low = resolved_gender.lower()
            gender_anat_pool = [
                t for t in anat_Pool
                if f"/{g_low}/" in t.lower()
                or (g_low == "female" and any(w in t.lower() for w in ("breast-shape", "jffbreast", "female-body", "femme-body")))
            ]
            # Filter out ALL.txt so we can pick specific allowed zone files
            zone_files = [t for t in gender_anat_pool if not t.lower().endswith("/all__")]
            allowed_files: list[str] = []
            for tf in zone_files:
                z = classify_anatomy_wildcard_zone(tf)
                if mood_filter == "SFW Only (Exclude NSFW / Erotic / Intimate)" and z in ("NIPPLES_AREOLAS", "INTIMATE_GENITALS"):
                    continue
                if auto_filter_anatomy_by_outfit:
                    if z is None or occlusion.allowed_zones.get(z, False):
                        # Only auto-include INTIMATE_GENITALS if outfit is explicitly nude/bottomless and mood is not SFW
                        if z == "INTIMATE_GENITALS" and not occlusion.is_nude:
                            continue
                        allowed_files.append(tf)
                else:
                    if z != "INTIMATE_GENITALS" or occlusion.is_nude:
                        allowed_files.append(tf)

            if allowed_files:
                # Prioritize visible body contour zones (e.g. Breasts-Chest, Waist-Midriff, Back-Spine, Arms-Shoulders, Buttocks-Glutes, Legs-Thighs, Feet-Ankles, Build-Silhouette/Musculature)
                body_zone_files = [
                    tf for tf in allowed_files
                    if classify_anatomy_wildcard_zone(tf) is not None
                    or any(w in tf.lower() for w in ("build-silhouette", "build-musculature", "size-stature"))
                ]
                chosen_anat_tokens = rng.sample(body_zone_files, min(2, len(body_zone_files))) if body_zone_files else [rng.choice(allowed_files)]
                anat_parts: list[str] = []
                for tok in chosen_anat_tokens:
                    line_exp, _ = _pick_filtered_wildcard_line(
                        "anatomy",
                        tok,
                        rng,
                        mood_filter=mood_filter,
                        gender_filter=resolved_gender,
                        exclude_full_prompts=exclude_full,
                        extra_dir=extra_dir,
                    )
                    if line_exp:
                        anat_parts.append(line_exp)
                raw_anat = ", ".join(anat_parts)
                anatomy_desc = (
                    filter_anatomy_text_by_occlusion(raw_anat, occlusion, gender=resolved_gender)
                    if auto_filter_anatomy_by_outfit
                    else raw_anat
                )

    # 4. Resolve Individual Action (if any)
    indiv_action = ""
    if custom_action_text and custom_action_text.strip():
        indiv_action = process_wildcards(custom_action_text.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
    elif individual_action_wildcard and individual_action_wildcard != NONE_OPTION:
        indiv_action, _ = _pick_filtered_wildcard_line(
            "poses_actions",
            individual_action_wildcard,
            rng,
            context_bundle=ctx_bundle,
            context_field="poses",
            mood_filter=mood_filter,
            custom_keyword=kw_filter,
            gender_filter=resolved_gender,
            exclude_full_prompts=exclude_full,
            extra_dir=extra_dir,
        )

    slot_data = CharacterSlotData(
        slot_role=slot_role,
        gender=resolved_gender,
        character_desc=char_desc,
        anatomy_desc=anatomy_desc,
        outfit_desc=outfit_desc,
        individual_action=indiv_action,
        outfit_link_mode=outfit_link_mode,
        occlusion_report=occlusion,
        theme_or_universe=theme_or_univ_name,
    )
    char_block_str = slot_data.format_character_block(encapsulate_subparts=False)
    debug_info = (
        f"[{slot_role} | {resolved_gender} | Mode: {outfit_link_mode}]\n"
        f"Character: {char_desc[:120]}\n"
        f"Outfit: {outfit_desc[:120]}\n"
        f"Occlusion: {occlusion.reason_summary}\n"
        f"Filtered Anatomy: {anatomy_desc or '[None / Occluded]'}"
    )
    return (slot_data, char_block_str, debug_info)


# ---------------------------------------------------------------------------
# Modular Node 1: ZiggartThemeUniverseContext
# ---------------------------------------------------------------------------


class ZiggartThemeUniverseContext:
    """
    Selects a Theme (`Collection/Themes`), a Universe (`Collection/Universes`), or Free Crossover mode,
    and sets global filtering rules (Mood/Tag filter, Full-Scene Prompt exclusion, Seed) that propagate
    to Character and Scene Stage nodes.
    """

    DESCRIPTION = (
        "Selects a Theme (1,518+ folders with Character-Outfits.json), a Universe (with source.json branches), "
        "or Free Multi-Folder Crossover mode, and configures mood/tag filtering and full-prompt exclusion."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "build_context"
    RETURN_TYPES = ("ZIGGART_CONTEXT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("context", "active_theme_or_universe", "default_style", "default_environment")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mode": (
                    [
                        "Theme (Collection/Themes)",
                        "Universe (Collection/Universes)",
                        "Random Theme",
                        "Random Universe",
                        "Free Crossover (All Folders: Collection, ziggart, jff, Encours)",
                    ],
                    {"default": "Theme (Collection/Themes)"},
                ),
                "theme_selection": (
                    WILDCARD_INDEX.get_theme_list(),
                    {"default": RANDOM_THEME_OPTION},
                ),
                "universe_selection": (
                    WILDCARD_INDEX.get_universe_list(),
                    {"default": RANDOM_UNIVERSE_OPTION},
                ),
                "mood_filter": (
                    MOOD_FILTER_OPTIONS,
                    {"default": "[Any / No Filter]"},
                ),
                "exclude_full_prompts": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Strictly ignores pre-assembled full scene wildcards (*-scene, *-prompts, Themes/ALL, Combos, Formats) when picking components.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "search_or_keyword_filter": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional keyword to filter random Themes/Universes or wildcard files (e.g. 'cyberpunk', 'albator', 'dragonball').",
                    },
                ),
                "custom_wildcards_dir": ("STRING", {"default": ""}),
            },
        }

    def build_context(
        self,
        source_mode: str = "Theme (Collection/Themes)",
        theme_selection: str = RANDOM_THEME_OPTION,
        universe_selection: str = RANDOM_UNIVERSE_OPTION,
        mood_filter: str = "[Any / No Filter]",
        exclude_full_prompts: bool = True,
        seed: int = 0,
        search_or_keyword_filter: str = "",
        custom_wildcards_dir: str = "",
    ) -> tuple[dict[str, Any], str, str, str]:
        WILDCARD_INDEX.ensure_loaded(extra_dir=custom_wildcards_dir)
        rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
        kw = search_or_keyword_filter.strip()

        bundle: dict[str, Any] | None = None
        active_name = "Free Crossover"

        if "Theme" in source_mode:
            t_name = theme_selection
            if source_mode == "Random Theme" or t_name == RANDOM_THEME_OPTION or t_name not in WILDCARD_INDEX.theme_folders:
                candidates = sorted(WILDCARD_INDEX.theme_folders.keys())
                if kw:
                    filtered = [c for c in candidates if kw.lower() in c.lower()]
                    if filtered:
                        candidates = filtered
                if candidates:
                    t_name = rng.choice(candidates)
            t_dir = WILDCARD_INDEX.theme_folders.get(t_name)
            if t_dir:
                bundle = load_theme_bundle(t_dir)
                active_name = t_name

        elif "Universe" in source_mode:
            u_name = universe_selection
            if source_mode == "Random Universe" or u_name == RANDOM_UNIVERSE_OPTION or u_name not in WILDCARD_INDEX.universe_folders:
                candidates = sorted(WILDCARD_INDEX.universe_folders.keys())
                if kw:
                    filtered = [c for c in candidates if kw.lower() in c.lower()]
                    if filtered:
                        candidates = filtered
                if candidates:
                    u_name = rng.choice(candidates)
            u_dir = WILDCARD_INDEX.universe_folders.get(u_name)
            if u_dir:
                bundle = load_universe_bundle(u_dir)
                active_name = u_name

        def_style = ""
        def_env = ""
        if bundle:
            if bundle.get("styles"):
                def_style = process_wildcards(rng.choice(bundle["styles"]), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=custom_wildcards_dir)
            if bundle.get("environments"):
                def_env = process_wildcards(rng.choice(bundle["environments"]), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=custom_wildcards_dir)

        ctx = {
            "source_mode": source_mode,
            "name": active_name,
            "bundle": bundle,
            "mood_filter": mood_filter,
            "keyword_filter": kw,
            "exclude_full_prompts": bool(exclude_full_prompts),
            "seed": int(seed),
            "custom_wildcards_dir": custom_wildcards_dir,
            "default_style": def_style,
            "default_environment": def_env,
        }
        return (ctx, active_name, def_style, def_env)


# ---------------------------------------------------------------------------
# Modular Node 2: ZiggartCharacterUnitBuilder
# ---------------------------------------------------------------------------


class ZiggartCharacterUnitBuilder:
    """
    Builds a complete Character Unit (Primary or Secondary) with:
    - Character selection from Theme/Universe or any pure character wildcard pool
    - Wildcard-GPT Outfit Linker: Canonical (1:1), Compatible (Theme+Gender), Free, or Nude
    - Automatic Outfit -> Anatomy Occlusion Filter (blocks covered zones like chest/nipples/midriff/intimate when covered by armor/clothes)
    """

    DESCRIPTION = (
        "Builds a Primary or Secondary Character slot with 1:1 Canonical/Compatible/Free Outfit linking "
        "and automatic Outfit -> Anatomy occlusion filtering (hides anatomy descriptions covered by the chosen outfit)."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "build_character"
    RETURN_TYPES = ("ZIGGART_CHAR_SLOT", "STRING", "STRING")
    RETURN_NAMES = ("char_slot", "character_prompt_block", "occlusion_debug_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slot_role": (["Primary", "Secondary"], {"default": "Primary"}),
                "character_source": (
                    [
                        "From Context (Theme / Universe)",
                        "Wildcard Selector / Category Pool",
                    ],
                    {"default": "From Context (Theme / Universe)"},
                ),
                "gender_filter": (
                    ["Auto / Any", "Female", "Male", "Creature"],
                    {"default": "Auto / Any"},
                ),
                "character_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("characters"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "outfit_link_mode": (
                    OUTFIT_LINK_MODES,
                    {"default": "Canonical (1:1 Strict Character Outfit)"},
                ),
                "outfit_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("outfits"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "enable_anatomy": ("BOOLEAN", {"default": False}),
                "auto_filter_anatomy_by_outfit": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Automatically blocks anatomical descriptions (e.g. breasts, nipples, midriff, glutes, intimate) if the resolved outfit covers those zones!",
                    },
                ),
                "anatomy_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("anatomy", include_context_option=False),
                    {"default": RANDOM_POOL_OPTION},
                ),
                "individual_action_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("poses_actions"),
                    {"default": NONE_OPTION},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "context": ("ZIGGART_CONTEXT",),
                "custom_char_slot_override": (
                    "ZIGGART_CHAR_SLOT",
                    {
                        "tooltip": "Optional character slot from Ziggart Krea2 Head-to-Toe Anatomy Character Creator to dress or re-filter.",
                    },
                ),
                "custom_character_text": ("STRING", {"default": "", "multiline": False}),
                "custom_outfit_text": ("STRING", {"default": "", "multiline": False}),
                "custom_anatomy_text": ("STRING", {"default": "", "multiline": False}),
                "custom_action_text": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def build_character(
        self,
        slot_role: str = "Primary",
        character_source: str = "From Context (Theme / Universe)",
        gender_filter: str = "Auto / Any",
        character_wildcard: str = FROM_CONTEXT_OPTION,
        outfit_link_mode: str = "Canonical (1:1 Strict Character Outfit)",
        outfit_wildcard: str = FROM_CONTEXT_OPTION,
        enable_anatomy: bool = False,
        auto_filter_anatomy_by_outfit: bool = True,
        anatomy_wildcard: str = RANDOM_POOL_OPTION,
        individual_action_wildcard: str = NONE_OPTION,
        seed: int = 0,
        context: dict[str, Any] | None = None,
        custom_char_slot_override: CharacterSlotData | None = None,
        custom_character_text: str = "",
        custom_outfit_text: str = "",
        custom_anatomy_text: str = "",
        custom_action_text: str = "",
    ) -> tuple[CharacterSlotData, str, str]:
        eff_seed = int(seed) if seed != 0 else (int(context.get("seed", 0)) if isinstance(context, dict) else 0)
        extra_dir = str(context.get("custom_wildcards_dir", "")) if isinstance(context, dict) else ""
        if isinstance(custom_char_slot_override, CharacterSlotData):
            if not custom_character_text.strip() and custom_char_slot_override.character_desc.strip():
                custom_character_text = custom_char_slot_override.character_desc
            if not custom_anatomy_text.strip() and custom_char_slot_override.anatomy_desc.strip():
                custom_anatomy_text = custom_char_slot_override.anatomy_desc
                enable_anatomy = True
            if gender_filter == "Auto / Any" and custom_char_slot_override.gender:
                gender_filter = custom_char_slot_override.gender
        return _resolve_character_unit(
            slot_role=slot_role,
            context=context,
            character_source=character_source,
            gender_filter=gender_filter,
            character_wildcard=character_wildcard,
            custom_character_text=custom_character_text,
            outfit_link_mode=outfit_link_mode,
            outfit_wildcard=outfit_wildcard,
            custom_outfit_text=custom_outfit_text,
            enable_anatomy=enable_anatomy,
            auto_filter_anatomy_by_outfit=auto_filter_anatomy_by_outfit,
            anatomy_wildcard=anatomy_wildcard,
            custom_anatomy_text=custom_anatomy_text,
            individual_action_wildcard=individual_action_wildcard,
            custom_action_text=custom_action_text,
            seed=eff_seed,
            extra_dir=extra_dir,
        )


# ---------------------------------------------------------------------------
# Modular Node 3: ZiggartSceneStageAtmosphere
# ---------------------------------------------------------------------------


class ZiggartSceneStageAtmosphere:
    """
    Selects and resolves the Scene Stage facets:
    (style), (action / duo interaction), (environment), (lighting), (camera)
    across `Collection`, `ziggart`, `jff`, and `Encours` while ignoring pre-built full scene prompts.
    """

    DESCRIPTION = (
        "Builds the Scene Stage (Style, Action/Interaction, Environment, Lighting, Camera) "
        "from Theme/Universe context or any pure component wildcard across Collection, ziggart, jff, and Encours."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "build_stage"
    RETURN_TYPES = ("ZIGGART_STAGE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("stage_data", "style_str", "action_str", "environment_str", "lighting_str", "camera_str")

    @classmethod
    def INPUT_TYPES(cls):
        action_choices = list(DUO_ACTION_PRESET_OPTIONS) + [
            t for t in WILDCARD_INDEX.get_facet_dropdown("poses_actions")
            if t not in DUO_ACTION_PRESET_OPTIONS
        ]
        return {
            "required": {
                "style_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("styles"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "action_or_interaction_wildcard": (
                    action_choices,
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "environment_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("environments"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "lighting_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("lighting", include_context_option=False),
                    {"default": RANDOM_POOL_OPTION},
                ),
                "camera_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("camera", include_context_option=False),
                    {"default": RANDOM_POOL_OPTION},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "context": ("ZIGGART_CONTEXT",),
                "custom_style": ("STRING", {"default": "", "multiline": False}),
                "custom_action": ("STRING", {"default": "", "multiline": False}),
                "custom_environment": ("STRING", {"default": "", "multiline": False}),
                "custom_lighting": ("STRING", {"default": "", "multiline": False}),
                "custom_camera": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def build_stage(
        self,
        style_wildcard: str = FROM_CONTEXT_OPTION,
        action_or_interaction_wildcard: str = FROM_CONTEXT_OPTION,
        environment_wildcard: str = FROM_CONTEXT_OPTION,
        lighting_wildcard: str = RANDOM_POOL_OPTION,
        camera_wildcard: str = RANDOM_POOL_OPTION,
        seed: int = 0,
        context: dict[str, Any] | None = None,
        custom_style: str = "",
        custom_action: str = "",
        custom_environment: str = "",
        custom_lighting: str = "",
        custom_camera: str = "",
    ) -> tuple[SceneStageData, str, str, str, str, str]:
        eff_seed = int(seed) if seed != 0 else (int(context.get("seed", 0)) if isinstance(context, dict) else 0)
        rng = random.Random((eff_seed ^ 0x5F3759DF) & 0xFFFFFFFFFFFFFFFF)
        ctx_bundle: dict[str, Any] | None = context.get("bundle") if isinstance(context, dict) else None
        mood_filter = str(context.get("mood_filter", "[Any / No Filter]")) if isinstance(context, dict) else "[Any / No Filter]"
        kw_filter = str(context.get("keyword_filter", "")) if isinstance(context, dict) else ""
        exclude_full = bool(context.get("exclude_full_prompts", True)) if isinstance(context, dict) else True
        extra_dir = str(context.get("custom_wildcards_dir", "")) if isinstance(context, dict) else ""
        theme_name = str(context.get("name", "Free Crossover")) if isinstance(context, dict) else "Free Crossover"

        # 1. Style
        if custom_style.strip():
            style_str = process_wildcards(custom_style.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        else:
            style_str, _ = _pick_filtered_wildcard_line(
                "styles",
                style_wildcard,
                rng,
                context_bundle=ctx_bundle,
                context_field="styles",
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )

        # 2. Action / Duo Interaction
        if custom_action.strip():
            action_str = process_wildcards(custom_action.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        elif action_or_interaction_wildcard in DUO_INTERACTIONS_BANK:
            action_str = rng.choice(DUO_INTERACTIONS_BANK[action_or_interaction_wildcard])
        else:
            action_str, _ = _pick_filtered_wildcard_line(
                "poses_actions",
                action_or_interaction_wildcard,
                rng,
                context_bundle=ctx_bundle,
                context_field="poses",
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )

        # 3. Environment
        if custom_environment.strip():
            env_str = process_wildcards(custom_environment.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        else:
            env_str, _ = _pick_filtered_wildcard_line(
                "environments",
                environment_wildcard,
                rng,
                context_bundle=ctx_bundle,
                context_field="environments",
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )

        # 4. Lighting
        if custom_lighting.strip():
            light_str = process_wildcards(custom_lighting.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        else:
            light_str, _ = _pick_filtered_wildcard_line(
                "lighting",
                lighting_wildcard,
                rng,
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )

        # 5. Camera
        if custom_camera.strip():
            cam_str = process_wildcards(custom_camera.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        else:
            cam_str, _ = _pick_filtered_wildcard_line(
                "camera",
                camera_wildcard,
                rng,
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )

        stage = SceneStageData(
            style_desc=style_str,
            action_desc=action_str,
            environment_desc=env_str,
            lighting_desc=light_str,
            camera_desc=cam_str,
            theme_or_universe=theme_name,
        )
        return (stage, style_str, action_str, env_str, light_str, cam_str)


# ---------------------------------------------------------------------------
# Modular Node 4: ZiggartKrea2PromptAssembler
# ---------------------------------------------------------------------------


class ZiggartKrea2PromptAssembler:
    """
    Assembles Character 1, optional Character 2, and Scene Stage into a Krea2 (Qwen3VL-4B)
    prompt using your exact parenthesized structure or Krea2CompilerV2 policies (scoped, balanced, compact, rich).
    """

    DESCRIPTION = (
        "Assembles Primary Character, optional Secondary Character, and Scene Stage into an optimal "
        "Krea2 (Qwen3VL-4B) prompt: (style), (char1), (char1_anatomy), (char1_outfit), [(char1_action) OR "
        "(char2), (char2_anatomy), (char2_outfit), (shared_action)], (environment), (lighting), (camera)."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "assemble"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "krea2_compiled_prompt",
        "char_1_detailer_prompt",
        "char_2_detailer_prompt",
        "krea2_stats_and_metadata_json",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "char_1_slot": ("ZIGGART_CHAR_SLOT",),
                "stage_data": ("ZIGGART_STAGE",),
                "krea2_policy": (
                    KREA2_COMPILER_POLICIES,
                    {"default": "scoped (Parentheses Blocks + Count Anchor)"},
                ),
                "separate_parentheses_per_facet": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When True in scoped mode, wraps each facet in its own parentheses: (style) (char1) (char1_anatomy) (wearing char1_outfit)... When False, groups (char1, anatomy, wearing outfit).",
                    },
                ),
                "include_count_anchor": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Prepends the subject count anchor ('1 woman,', '2 characters, a woman and a man,') outside parentheses for Qwen3VL-4B.",
                    },
                ),
            },
            "optional": {
                "char_2_slot": ("ZIGGART_CHAR_SLOT",),
                "custom_prefix": ("STRING", {"default": "", "multiline": False}),
                "custom_suffix": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def assemble(
        self,
        char_1_slot: CharacterSlotData,
        stage_data: SceneStageData,
        krea2_policy: str = "scoped (Parentheses Blocks + Count Anchor)",
        separate_parentheses_per_facet: bool = True,
        include_count_anchor: bool = True,
        char_2_slot: CharacterSlotData | None = None,
        custom_prefix: str = "",
        custom_suffix: str = "",
    ) -> tuple[str, str, str, str]:
        compiled, stats = assemble_krea2_structured_prompt(
            char1=char_1_slot,
            char2=char_2_slot,
            stage=stage_data,
            policy=krea2_policy,
            separate_parentheses_per_facet=separate_parentheses_per_facet,
            include_count_anchor=include_count_anchor,
            custom_prefix=custom_prefix,
            custom_suffix=custom_suffix,
        )
        c1_det = char_1_slot.format_character_block(encapsulate_subparts=False) if char_1_slot else ""
        c2_det = char_2_slot.format_character_block(encapsulate_subparts=False) if char_2_slot else ""
        return (compiled, c1_det, c2_det, json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# Node 5 (All-in-One): ZiggartKrea2MasterPromptDirector
# ---------------------------------------------------------------------------


class ZiggartKrea2MasterPromptDirector:
    """
    All-in-One Krea2 (Qwen3VL-4B) Prompt Director combining Theme/Universe selection,
    Solo or Duo character builder, 1:1 Canonical/Compatible/Free Outfit linking,
    automatic Outfit -> Anatomy occlusion filtering, Scene Stage selectors, and Krea2CompilerV2.
    """

    DESCRIPTION = (
        "All-in-One Krea2 (Qwen3VL-4B) Prompt Director. Builds structured Solo or Duo prompts directly "
        "from your Themes, Universes, Collection, ziggart, jff, and Encours wildcards with automatic "
        "Character-Outfit linking and Outfit -> Anatomy occlusion filtering."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "direct_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "krea2_compiled_prompt",
        "populated_text",
        "char_1_detailer_prompt",
        "char_2_detailer_prompt",
        "krea2_stats_and_metadata_json",
    )

    @classmethod
    def INPUT_TYPES(cls):
        action_choices = list(DUO_ACTION_PRESET_OPTIONS) + [
            t for t in WILDCARD_INDEX.get_facet_dropdown("poses_actions")
            if t not in DUO_ACTION_PRESET_OPTIONS
        ]
        return {
            "required": {
                "source_mode": (
                    [
                        "Theme (Collection/Themes)",
                        "Universe (Collection/Universes)",
                        "Random Theme",
                        "Random Universe",
                        "Free Crossover (All Folders: Collection, ziggart, jff, Encours)",
                    ],
                    {"default": "Theme (Collection/Themes)"},
                ),
                "theme_selection": (
                    WILDCARD_INDEX.get_theme_list(),
                    {"default": RANDOM_THEME_OPTION},
                ),
                "universe_selection": (
                    WILDCARD_INDEX.get_universe_list(),
                    {"default": RANDOM_UNIVERSE_OPTION},
                ),
                "cast_mode": (
                    [
                        "Solo (1 Character)",
                        "Duo (2 Characters + Shared Action)",
                    ],
                    {"default": "Solo (1 Character)"},
                ),
                "krea2_policy": (
                    KREA2_COMPILER_POLICIES,
                    {"default": "scoped (Parentheses Blocks + Count Anchor)"},
                ),
                "separate_parentheses_per_facet": ("BOOLEAN", {"default": True}),
                "include_count_anchor": ("BOOLEAN", {"default": True}),
                "mood_filter": (MOOD_FILTER_OPTIONS, {"default": "[Any / No Filter]"}),
                "exclude_full_prompts": ("BOOLEAN", {"default": True}),
                # --- Style ---
                "style_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("styles"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                # --- Character 1 ---
                "char_1_gender": (["Auto / Any", "Female", "Male", "Creature"], {"default": "Auto / Any"}),
                "char_1_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("characters"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "char_1_outfit_link_mode": (
                    OUTFIT_LINK_MODES,
                    {"default": "Canonical (1:1 Strict Character Outfit)"},
                ),
                "char_1_outfit_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("outfits"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "char_1_enable_anatomy": ("BOOLEAN", {"default": False}),
                "char_1_auto_filter_anatomy": ("BOOLEAN", {"default": True}),
                "char_1_anatomy_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("anatomy", include_context_option=False),
                    {"default": RANDOM_POOL_OPTION},
                ),
                # --- Character 2 (Active when cast_mode is Duo) ---
                "char_2_gender": (["Auto / Any", "Female", "Male", "Creature"], {"default": "Auto / Any"}),
                "char_2_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("characters"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "char_2_outfit_link_mode": (
                    OUTFIT_LINK_MODES,
                    {"default": "Canonical (1:1 Strict Character Outfit)"},
                ),
                "char_2_outfit_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("outfits"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "char_2_enable_anatomy": ("BOOLEAN", {"default": False}),
                "char_2_auto_filter_anatomy": ("BOOLEAN", {"default": True}),
                "char_2_anatomy_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("anatomy", include_context_option=False),
                    {"default": RANDOM_POOL_OPTION},
                ),
                # --- Action / Environment / Lighting / Camera ---
                "action_or_interaction_wildcard": (
                    action_choices,
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "environment_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("environments"),
                    {"default": FROM_CONTEXT_OPTION},
                ),
                "lighting_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("lighting", include_context_option=False),
                    {"default": RANDOM_POOL_OPTION},
                ),
                "camera_wildcard": (
                    WILDCARD_INDEX.get_facet_dropdown("camera", include_context_option=False),
                    {"default": RANDOM_POOL_OPTION},
                ),
                "populated_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Displays the generated Krea2 prompt (or overrides it when mode is 'fixed').",
                    },
                ),
                "mode": (["populate", "fixed", "reproduce"], {"default": "populate"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "search_or_keyword_filter": ("STRING", {"default": ""}),
                "custom_prefix": ("STRING", {"default": ""}),
                "custom_suffix": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def direct_prompt(
        self,
        source_mode: str = "Theme (Collection/Themes)",
        theme_selection: str = RANDOM_THEME_OPTION,
        universe_selection: str = RANDOM_UNIVERSE_OPTION,
        cast_mode: str = "Solo (1 Character)",
        krea2_policy: str = "scoped (Parentheses Blocks + Count Anchor)",
        separate_parentheses_per_facet: bool = True,
        include_count_anchor: bool = True,
        mood_filter: str = "[Any / No Filter]",
        exclude_full_prompts: bool = True,
        style_wildcard: str = FROM_CONTEXT_OPTION,
        char_1_gender: str = "Auto / Any",
        char_1_wildcard: str = FROM_CONTEXT_OPTION,
        char_1_outfit_link_mode: str = "Canonical (1:1 Strict Character Outfit)",
        char_1_outfit_wildcard: str = FROM_CONTEXT_OPTION,
        char_1_enable_anatomy: bool = False,
        char_1_auto_filter_anatomy: bool = True,
        char_1_anatomy_wildcard: str = RANDOM_POOL_OPTION,
        char_2_gender: str = "Auto / Any",
        char_2_wildcard: str = FROM_CONTEXT_OPTION,
        char_2_outfit_link_mode: str = "Canonical (1:1 Strict Character Outfit)",
        char_2_outfit_wildcard: str = FROM_CONTEXT_OPTION,
        char_2_enable_anatomy: bool = False,
        char_2_auto_filter_anatomy: bool = True,
        char_2_anatomy_wildcard: str = RANDOM_POOL_OPTION,
        action_or_interaction_wildcard: str = FROM_CONTEXT_OPTION,
        environment_wildcard: str = FROM_CONTEXT_OPTION,
        lighting_wildcard: str = RANDOM_POOL_OPTION,
        camera_wildcard: str = RANDOM_POOL_OPTION,
        populated_text: str = "",
        mode: str = "populate",
        seed: int = 0,
        search_or_keyword_filter: str = "",
        custom_prefix: str = "",
        custom_suffix: str = "",
        unique_id: Any = None,
    ) -> tuple[str, str, str, str, str]:
        if mode in ("fixed", "reproduce") and populated_text.strip():
            fixed_out = process_wildcards(populated_text.strip(), seed=seed)
            if mode == "reproduce":
                _send_node_feedback(unique_id, "populated_text", fixed_out)
            return (fixed_out, fixed_out, "", "", json.dumps({"mode": mode}))

        ctx_node = ZiggartThemeUniverseContext()
        ctx, active_name, _, _ = ctx_node.build_context(
            source_mode=source_mode,
            theme_selection=theme_selection,
            universe_selection=universe_selection,
            mood_filter=mood_filter,
            exclude_full_prompts=exclude_full_prompts,
            seed=seed,
            search_or_keyword_filter=search_or_keyword_filter,
        )

        used_ids: set[str] = set()
        c1_src = "From Context (Theme / Universe)" if char_1_wildcard == FROM_CONTEXT_OPTION else "Wildcard Selector / Category Pool"
        c1_slot, c1_block, _ = _resolve_character_unit(
            slot_role="Primary",
            context=ctx,
            character_source=c1_src,
            gender_filter=char_1_gender,
            character_wildcard=char_1_wildcard,
            custom_character_text="",
            outfit_link_mode=char_1_outfit_link_mode,
            outfit_wildcard=char_1_outfit_wildcard,
            custom_outfit_text="",
            enable_anatomy=char_1_enable_anatomy,
            auto_filter_anatomy_by_outfit=char_1_auto_filter_anatomy,
            anatomy_wildcard=char_1_anatomy_wildcard,
            custom_anatomy_text="",
            individual_action_wildcard=NONE_OPTION,
            custom_action_text="",
            seed=seed,
            used_char_ids=used_ids,
        )

        c2_slot: CharacterSlotData | None = None
        c2_block = ""
        if "Duo" in cast_mode:
            c2_src = "From Context (Theme / Universe)" if char_2_wildcard == FROM_CONTEXT_OPTION else "Wildcard Selector / Category Pool"
            c2_slot, c2_block, _ = _resolve_character_unit(
                slot_role="Secondary",
                context=ctx,
                character_source=c2_src,
                gender_filter=char_2_gender,
                character_wildcard=char_2_wildcard,
                custom_character_text="",
                outfit_link_mode=char_2_outfit_link_mode,
                outfit_wildcard=char_2_outfit_wildcard,
                custom_outfit_text="",
                enable_anatomy=char_2_enable_anatomy,
                auto_filter_anatomy_by_outfit=char_2_auto_filter_anatomy,
                anatomy_wildcard=char_2_anatomy_wildcard,
                custom_anatomy_text="",
                individual_action_wildcard=NONE_OPTION,
                custom_action_text="",
                seed=seed + 1,
                used_char_ids=used_ids,
            )

        stage_node = ZiggartSceneStageAtmosphere()
        stage_data, _, _, _, _, _ = stage_node.build_stage(
            style_wildcard=style_wildcard,
            action_or_interaction_wildcard=action_or_interaction_wildcard,
            environment_wildcard=environment_wildcard,
            lighting_wildcard=lighting_wildcard,
            camera_wildcard=camera_wildcard,
            seed=seed,
            context=ctx,
        )

        compiled, stats = assemble_krea2_structured_prompt(
            char1=c1_slot,
            char2=c2_slot,
            stage=stage_data,
            policy=krea2_policy,
            separate_parentheses_per_facet=separate_parentheses_per_facet,
            include_count_anchor=include_count_anchor,
            custom_prefix=custom_prefix,
            custom_suffix=custom_suffix,
        )
        stats["active_theme_or_universe"] = active_name
        _send_node_feedback(unique_id, "populated_text", compiled)
        return (compiled, compiled, c1_block, c2_block, json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# Modular Node 6: ZiggartAnatomyCharacterCreator (Head-to-Toe Character Creator)
# ---------------------------------------------------------------------------


def _get_anatomy_section_options(section_def: dict[str, Any]) -> list[str]:
    """Builds the dropdown list for a single Head-to-Toe anatomical section."""
    sec_id = section_def["id"]
    if sec_id == "age_group":
        return [
            AUTO_VIEW_OPTION,
            NONE_OPTION,
            "Age-YoungAdult",
            "Age-Adult",
            "Age-Maturity",
            "Age-Elders",
            RANDOM_GENDER_DEFAULT_OPTION,
        ]
    opts: list[str] = [
        AUTO_VIEW_OPTION,
        NONE_OPTION,
        RANDOM_GENDER_DEFAULT_OPTION,
    ]
    for g_key in ("Female", "Male", "Creature"):
        rel_path = section_def.get(g_key)
        if rel_path:
            tok = f"__{rel_path}__"
            if tok not in opts:
                opts.append(tok)
    if sec_id == "arms_shoulders":
        extra_tok = "__Collection/Anatomy/Creature/Extra-Limbs-MultiArms__"
        if extra_tok not in opts:
            opts.append(extra_tok)
    elif sec_id == "chest_breasts":
        extra_tok = "__jff/breast-shape__"
        if extra_tok not in opts:
            opts.append(extra_tok)
    return opts


class ZiggartAnatomyCharacterCreator:
    """
    Builds a complete Character Description from Head to Toe using the 28 anatomical
    sections of `Collection/Anatomy/{Female,Male,Creature}`.
    Every single section can be set to `[None]` (e.g. when a character is seen from behind
    and does not show their eyes/face/front chest, or in a portrait where legs/feet are out of frame),
    controlled individually or via `view_visibility_preset`, and optionally filtered by Outfit occlusion!
    """

    DESCRIPTION = (
        "Head-to-Toe Character Creator powered by Collection/Anatomy (Female, Male, Creature). "
        "Every anatomical section from Hair, Eyes, Nose, Lips, Neck, Shoulders, Chest, Back, Waist, "
        "Hips, Glutes, Legs to Feet can be toggled to [None], [Random], or controlled via View Presets "
        "(Front, Back View / Seen From Behind, Over-the-Shoulder, Portrait, Face Close-Up, Lower Body)."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "create_anatomy_character"
    RETURN_TYPES = ("ZIGGART_CHAR_SLOT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "char_slot",
        "character_full_prompt",
        "head_and_face_prompt",
        "body_and_outfit_prompt",
        "anatomy_breakdown_json",
    )
    OUTPUT_TOOLTIPS = (
        "Structured CharacterSlotData ready to wire into Ziggart Krea2 Prompt Assembler or Ziggart Character Unit Builder.",
        "Complete head-to-toe character prompt combining all active sections (and optional outfit/action).",
        "Resolved Hair + Head + Face + Eyes description (ideal for Face/Eye Detailers).",
        "Resolved Silhouette + Body Anatomy + Outfit description.",
        "JSON breakdown of active sections, [None] sections, and outfit-occluded sections.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        req: dict[str, Any] = {
            "slot_role": (["Primary", "Secondary"], {"default": "Primary"}),
            "gender": (["Female", "Male", "Creature"], {"default": "Female"}),
            "view_visibility_preset": (
                list(VIEW_VISIBILITY_PRESETS),
                {
                    "default": "Full Body — Front / 3-Quarter View",
                    "tooltip": (
                        "Automatically enables/disables head-to-toe sections that are set to [Auto (From View Preset)]. "
                        "For example, 'Back View / Seen From Behind' disables eyes/face/front chest and enables Back-Spine & Buttocks-Glutes. "
                        "Any section explicitly set to [None] or [Random (Gender Default)] always overrides the preset!"
                    ),
                },
            ),
        }
        for sec_def in HEAD_TO_TOE_ANATOMY_SECTIONS:
            sec_id = sec_def["id"]
            opts = _get_anatomy_section_options(sec_def)
            default_val = NONE_OPTION if sec_id in ("nipples_areolas", "intimate_anatomy") else AUTO_VIEW_OPTION
            req[sec_id] = (opts, {"default": default_val})

        req["outfit_mode"] = (
            [
                "[None / Pure Anatomy Only]",
                "Compatible (Theme/Universe + Gender)",
                "Free (Outfit Wildcard)",
                "Nude / Unclothed",
            ],
            {"default": "[None / Pure Anatomy Only]"},
        )
        req["outfit_wildcard"] = (
            WILDCARD_INDEX.get_facet_dropdown("outfits"),
            {"default": NONE_OPTION},
        )
        req["auto_filter_anatomy_by_outfit"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "When an outfit is active, automatically filters out body sections covered by that outfit (across all 9 body zones).",
            },
        )
        req["individual_action_wildcard"] = (
            WILDCARD_INDEX.get_facet_dropdown("poses_actions"),
            {"default": NONE_OPTION},
        )
        req["seed"] = ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})

        return {
            "required": req,
            "optional": {
                "context": ("ZIGGART_CONTEXT",),
                "custom_identity_prefix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional character name or archetype prefix (e.g. 'elven shieldmaiden', 'cyberpunk netrunner').",
                    },
                ),
                "custom_outfit_text": ("STRING", {"default": "", "multiline": False}),
                "custom_action_text": ("STRING", {"default": "", "multiline": False}),
            },
        }

    def create_anatomy_character(
        self,
        slot_role: str = "Primary",
        gender: str = "Female",
        view_visibility_preset: str = "Full Body — Front / 3-Quarter View",
        outfit_mode: str = "[None / Pure Anatomy Only]",
        outfit_wildcard: str = NONE_OPTION,
        auto_filter_anatomy_by_outfit: bool = True,
        individual_action_wildcard: str = NONE_OPTION,
        seed: int = 0,
        context: dict[str, Any] | None = None,
        custom_identity_prefix: str = "",
        custom_outfit_text: str = "",
        custom_action_text: str = "",
        **section_kwargs: Any,
    ) -> tuple[CharacterSlotData, str, str, str, str]:
        eff_seed = int(seed) if seed != 0 else (int(context.get("seed", 0)) if isinstance(context, dict) else 0)
        extra_dir = str(context.get("custom_wildcards_dir", "")) if isinstance(context, dict) else ""
        WILDCARD_INDEX.ensure_loaded(extra_dir=extra_dir)
        rng = random.Random((eff_seed ^ (0x5A5A5A5A if slot_role == "Secondary" else 0x2468ACE0)) & 0xFFFFFFFFFFFFFFFF)

        ctx_bundle: dict[str, Any] | None = context.get("bundle") if isinstance(context, dict) else None
        mood_filter = str(context.get("mood_filter", "[Any / No Filter]")) if isinstance(context, dict) else "[Any / No Filter]"
        kw_filter = str(context.get("keyword_filter", "")) if isinstance(context, dict) else ""
        exclude_full = bool(context.get("exclude_full_prompts", True)) if isinstance(context, dict) else True
        theme_or_univ_name = str(context.get("name", "Head-to-Toe Anatomy")) if isinstance(context, dict) else "Head-to-Toe Anatomy"

        # 1. Resolve Optional Outfit first so we can apply 9-zone occlusion filtering to body sections if enabled
        outfit_desc = ""
        unexpanded_outfit_token = ""
        om_low = outfit_mode.lower()

        if "nude" in om_low:
            outfit_desc = "nude, completely unclothed"
            unexpanded_outfit_token = "nude"
        elif custom_outfit_text and custom_outfit_text.strip():
            unexpanded_outfit_token = custom_outfit_text.strip()
            outfit_desc = process_wildcards(unexpanded_outfit_token, seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        elif "none" not in om_low:
            if "compatible" in om_low and ctx_bundle is not None and outfit_wildcard in (NONE_OPTION, FROM_CONTEXT_OPTION):
                compat_choices = get_theme_outfit_choices(ctx_bundle, {"gender": gender}, "compatible")
                if compat_choices:
                    outfit_desc = process_wildcards(rng.choice(compat_choices), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
                    unexpanded_outfit_token = f"[{theme_or_univ_name}:{gender}-Outfit]"
            if not outfit_desc:
                o_choice = outfit_wildcard if outfit_wildcard not in (NONE_OPTION, "") else RANDOM_POOL_OPTION
                outfit_desc, unexpanded_outfit_token = _pick_filtered_wildcard_line(
                    "outfits",
                    o_choice,
                    rng,
                    context_bundle=ctx_bundle,
                    context_field="outfits",
                    mood_filter=mood_filter,
                    custom_keyword=kw_filter,
                    gender_filter=gender if "compatible" in om_low else "Auto / Any",
                    exclude_full_prompts=exclude_full,
                    extra_dir=extra_dir,
                )

        outfit_desc = clean_outfit_prefix(outfit_desc)
        has_outfit = bool(outfit_desc) and "none" not in om_low
        occlusion = (
            analyze_outfit_occlusion(outfit_desc, unexpanded_outfit_token)
            if has_outfit
            else OutfitOcclusionReport(
                outfit_text="",
                is_nude=True,
                allowed_zones={z: True for z in ANATOMY_ZONES},
                blocked_zones=(),
                visible_zones=ANATOMY_ZONES,
                reason_summary="Pure Anatomy / No Outfit occlusion.",
            )
        )

        # 2. Walk Head-to-Toe Sections in anatomical order
        profile_parts: list[str] = []
        head_face_parts: list[str] = []
        body_parts: list[str] = []

        active_sections: list[str] = []
        disabled_sections: list[str] = []
        occluded_sections: list[str] = []
        resolved_map: dict[str, str] = {}

        if custom_identity_prefix and custom_identity_prefix.strip():
            pref_exp = process_wildcards(custom_identity_prefix.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
            if pref_exp.strip():
                profile_parts.append(pref_exp.strip().rstrip(".,"))
                resolved_map["custom_identity_prefix"] = pref_exp.strip()

        for sec_def in HEAD_TO_TOE_ANATOMY_SECTIONS:
            sec_id = sec_def["id"]
            sec_group = sec_def["group"]
            sec_zone = sec_def.get("zone")
            user_val = str(section_kwargs.get(sec_id, AUTO_VIEW_OPTION) or AUTO_VIEW_OPTION)

            # Determine if section is enabled or [None]
            if user_val == NONE_OPTION:
                disabled_sections.append(sec_id)
                continue
            if user_val == AUTO_VIEW_OPTION:
                if not is_section_enabled_by_view_preset(sec_id, view_visibility_preset):
                    disabled_sections.append(sec_id)
                    continue

            # Determine target wildcard token for this section & gender
            target_token = ""
            if sec_id == "age_group":
                if gender == "Creature":
                    disabled_sections.append(sec_id)
                    continue
                if user_val in ("Age-YoungAdult", "Age-Adult", "Age-Maturity", "Age-Elders"):
                    target_token = f"__Collection/Anatomy/{gender}/{user_val}__"
                elif user_val == RANDOM_GENDER_DEFAULT_OPTION:
                    age_file = rng.choice(["Age-YoungAdult", "Age-Adult", "Age-Maturity", "Age-Elders"])
                    target_token = f"__Collection/Anatomy/{gender}/{age_file}__"
                else:
                    target_token = f"__Collection/Anatomy/{gender}/Age-Adult__"
            elif user_val.startswith("__") and user_val.endswith("__"):
                target_token = user_val
            else:
                # AUTO_VIEW_OPTION or RANDOM_GENDER_DEFAULT_OPTION -> use gender default path
                rel_p = sec_def.get(gender)
                if not rel_p:
                    disabled_sections.append(sec_id)
                    continue
                target_token = f"__{rel_p}__"

            # Check SFW mood filter & Outfit -> Anatomy occlusion for body zones
            if mood_filter == "SFW Only (Exclude NSFW / Erotic / Intimate)" and sec_zone in ("NIPPLES_AREOLAS", "INTIMATE_GENITALS"):
                occluded_sections.append(sec_id)
                continue

            if has_outfit and auto_filter_anatomy_by_outfit and sec_zone is not None:
                if not occlusion.allowed_zones.get(sec_zone, True):
                    occluded_sections.append(sec_id)
                    continue

            resolved_line, _ = _pick_filtered_wildcard_line(
                "anatomy",
                target_token,
                rng,
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                gender_filter=gender,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )
            cleaned_line = resolved_line.strip().rstrip(".,")
            if not cleaned_line:
                disabled_sections.append(sec_id)
                continue

            active_sections.append(sec_id)
            resolved_map[sec_id] = cleaned_line
            if sec_group == "profile":
                profile_parts.append(cleaned_line)
            elif sec_group in ("head", "face"):
                head_face_parts.append(cleaned_line)
            else:
                body_parts.append(cleaned_line)

        # 3. Resolve Optional Individual Action
        indiv_action = ""
        if custom_action_text and custom_action_text.strip():
            indiv_action = process_wildcards(custom_action_text.strip(), seed=rng.randint(0, 0x7FFFFFFF), extra_dir=extra_dir)
        elif individual_action_wildcard and individual_action_wildcard not in (NONE_OPTION, ""):
            indiv_action, _ = _pick_filtered_wildcard_line(
                "poses_actions",
                individual_action_wildcard,
                rng,
                context_bundle=ctx_bundle,
                context_field="poses",
                mood_filter=mood_filter,
                custom_keyword=kw_filter,
                gender_filter=gender,
                exclude_full_prompts=exclude_full,
                extra_dir=extra_dir,
            )

        # Assemble outputs
        head_and_face_str = ", ".join(profile_parts + head_face_parts)
        body_anatomy_str = ", ".join(body_parts)

        # Ensure `character_desc` is non-empty even in Lower Body Focus mode
        if head_and_face_str:
            char_desc_for_slot = head_and_face_str
            anat_desc_for_slot = body_anatomy_str
        else:
            char_desc_for_slot = body_anatomy_str
            anat_desc_for_slot = ""

        slot_data = CharacterSlotData(
            slot_role=slot_role,
            gender=gender,
            character_desc=char_desc_for_slot,
            anatomy_desc=anat_desc_for_slot,
            outfit_desc=outfit_desc if has_outfit else "",
            individual_action=indiv_action,
            outfit_link_mode=outfit_mode,
            occlusion_report=occlusion,
            theme_or_universe=theme_or_univ_name,
        )

        body_and_outfit_items: list[str] = []
        if body_anatomy_str:
            body_and_outfit_items.append(body_anatomy_str)
        if has_outfit and outfit_desc:
            if "nude" in om_low:
                body_and_outfit_items.append(outfit_desc)
            else:
                body_and_outfit_items.append(f"wearing {outfit_desc}")
        body_and_outfit_str = ", ".join(body_and_outfit_items)

        full_items: list[str] = []
        base_block = slot_data.format_character_block(encapsulate_subparts=False)
        if base_block:
            full_items.append(base_block)
        if indiv_action:
            full_items.append(indiv_action.strip().rstrip(".,"))
        character_full_prompt = ", ".join(full_items)

        breakdown = {
            "slot_role": slot_role,
            "gender": gender,
            "view_visibility_preset": view_visibility_preset,
            "outfit_mode": outfit_mode,
            "resolved_outfit": outfit_desc if has_outfit else "",
            "occlusion_summary": occlusion.reason_summary,
            "active_sections_count": len(active_sections),
            "active_sections": active_sections,
            "none_or_disabled_sections": disabled_sections,
            "occluded_by_outfit_sections": occluded_sections,
            "resolved_sections": resolved_map,
        }
        return (
            slot_data,
            character_full_prompt,
            head_and_face_str,
            body_and_outfit_str,
            json.dumps(breakdown, indent=2, ensure_ascii=False),
        )


# ---------------------------------------------------------------------------
# Upgrade A Node 1: ZiggartKrea2TrioEnsembleDirector
# ---------------------------------------------------------------------------

ENSEMBLE_INTERACTION_MODES = [
    "Coordinated Team Action",
    "Combat Clash / Duel",
    "Conversation / Dramatic Scene",
    "Independent Heroic Poses",
    "Ritual / Tactical Formation",
]


class ZiggartKrea2TrioEnsembleDirector:
    """
    Assembles up to 3 Character Slots (char_1, char_2, char_3) + an optional Companion /
    Mount / Prop descriptor + Scene Stage into a structured Krea2 / Flux prompt, while
    also outputting a ready-to-use ZIGGART_SPATIAL_LAYOUT for Regional Conditioning.
    """

    DESCRIPTION = (
        "Trio & Ensemble Director for Krea2 / Qwen3VL-4B / Flux. Combines up to 3 Character Slots "
        "(char_1_slot, char_2_slot, char_3_slot), an optional Companion/Mount/Prop, and Scene Stage "
        "with spatial composition presets and outputs both the compiled prompt and a ZIGGART_SPATIAL_LAYOUT."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "direct_ensemble"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "ZIGGART_SPATIAL_LAYOUT",
        "STRING",
    )
    RETURN_NAMES = (
        "krea2_compiled_prompt",
        "base_scene_prompt",
        "char_1_detailer_prompt",
        "char_2_detailer_prompt",
        "char_3_detailer_prompt",
        "spatial_layout",
        "krea2_stats_and_metadata_json",
    )

    @classmethod
    def INPUT_TYPES(cls):
        layout_presets = list(ENSEMBLE_SPATIAL_LAYOUT_PRESETS.keys())
        return {
            "required": {
                "ensemble_layout_preset": (
                    layout_presets,
                    {
                        "default": "Left / Center / Right (Trio)",
                        "tooltip": "Spatial composition preset used for prompt hints and regional conditioning masks.",
                    },
                ),
                "interaction_mode": (
                    ENSEMBLE_INTERACTION_MODES,
                    {"default": "Coordinated Team Action"},
                ),
                "krea2_policy": (
                    KREA2_COMPILER_POLICIES,
                    {"default": "scoped (Parentheses Blocks + Count Anchor)"},
                ),
                "separate_parentheses_per_facet": ("BOOLEAN", {"default": True}),
                "include_count_anchor": ("BOOLEAN", {"default": True}),
                "include_spatial_layout_hint": ("BOOLEAN", {"default": True}),
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "char_1_slot": ("ZIGGART_CHAR_SLOT",),
                "char_2_slot": ("ZIGGART_CHAR_SLOT",),
                "char_3_slot": ("ZIGGART_CHAR_SLOT",),
                "stage_data": ("ZIGGART_STAGE_DATA",),
                "theme_context": ("ZIGGART_THEME_CTX",),
                "companion_or_prop_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Optional companion creature, mount, vehicle, or signature prop (supports wildcards).",
                    },
                ),
                "custom_prefix": ("STRING", {"default": ""}),
                "custom_suffix": ("STRING", {"default": ""}),
            },
        }

    def direct_ensemble(
        self,
        ensemble_layout_preset: str = "Left / Center / Right (Trio)",
        interaction_mode: str = "Coordinated Team Action",
        krea2_policy: str = "scoped (Parentheses Blocks + Count Anchor)",
        separate_parentheses_per_facet: bool = True,
        include_count_anchor: bool = True,
        include_spatial_layout_hint: bool = True,
        canvas_width: int = 1024,
        canvas_height: int = 1024,
        seed: int = 0,
        char_1_slot: CharacterSlotData | None = None,
        char_2_slot: CharacterSlotData | None = None,
        char_3_slot: CharacterSlotData | None = None,
        stage_data: SceneStageData | None = None,
        theme_context: dict[str, Any] | None = None,
        companion_or_prop_text: str = "",
        custom_prefix: str = "",
        custom_suffix: str = "",
    ):
        prop_expanded = process_wildcards(companion_or_prop_text.strip(), seed=seed) if companion_or_prop_text.strip() else ""
        prefix_exp = process_wildcards(custom_prefix.strip(), seed=seed + 11) if custom_prefix.strip() else ""
        suffix_exp = process_wildcards(custom_suffix.strip(), seed=seed + 23) if custom_suffix.strip() else ""

        preset_info = ENSEMBLE_SPATIAL_LAYOUT_PRESETS.get(
            ensemble_layout_preset,
            ENSEMBLE_SPATIAL_LAYOUT_PRESETS["Left / Center / Right (Trio)"],
        )
        layout_hint = preset_info.get("hint", "") if include_spatial_layout_hint else ""

        # If stage_data has no action and multiple characters are present, synthesize an interaction phrase
        eff_stage = stage_data
        if eff_stage is None and isinstance(theme_context, dict):
            stage_builder = ZiggartSceneStageAtmosphere()
            eff_stage, _, _, _, _, _ = stage_builder.build_stage(seed=seed, context=theme_context)

        if eff_stage is not None and not eff_stage.action_desc.strip():
            interaction_map = {
                "Coordinated Team Action": "acting in coordinated team formation with dynamic synergy",
                "Combat Clash / Duel": "locked in a high-energy combat confrontation with kinetic tension",
                "Conversation / Dramatic Scene": "engaged in a tense dramatic interaction with expressive eye contact",
                "Independent Heroic Poses": "striking distinct iconic poses within the shared composition",
                "Ritual / Tactical Formation": "arranged in a deliberate tactical formation with atmospheric presence",
            }
            synth_act = interaction_map.get(interaction_mode, "")
            eff_stage = SceneStageData(
                style_desc=eff_stage.style_desc,
                action_desc=synth_act,
                environment_desc=eff_stage.environment_desc,
                lighting_desc=eff_stage.lighting_desc,
                camera_desc=eff_stage.camera_desc,
                theme_or_universe=eff_stage.theme_or_universe,
            )

        compiled, stats = assemble_krea2_structured_prompt(
            char1=char_1_slot,
            char2=char_2_slot,
            char3=char_3_slot,
            companion_or_prop=prop_expanded,
            spatial_layout_hint=layout_hint,
            stage=eff_stage,
            policy=krea2_policy,
            separate_parentheses_per_facet=separate_parentheses_per_facet,
            include_count_anchor=include_count_anchor,
            custom_prefix=prefix_exp,
            custom_suffix=suffix_exp,
        )

        # Build base scene prompt (without character specifics) for Regional Conditioning
        base_parts: list[str] = []
        if eff_stage:
            for s_part in (
                eff_stage.style_desc,
                eff_stage.action_desc,
                eff_stage.environment_desc,
                eff_stage.lighting_desc,
                eff_stage.camera_desc,
            ):
                if s_part and s_part.strip():
                    base_parts.append(s_part.strip().rstrip(".,"))
        if layout_hint:
            base_parts.append(layout_hint)
        base_scene_prompt = ", ".join(base_parts) if base_parts else compiled

        c1_det = char_1_slot.format_character_block(encapsulate_subparts=False) if char_1_slot else ""
        c2_det = char_2_slot.format_character_block(encapsulate_subparts=False) if char_2_slot else ""
        c3_det = char_3_slot.format_character_block(encapsulate_subparts=False) if char_3_slot else ""

        w_px = max(64, int(canvas_width))
        h_px = max(64, int(canvas_height))
        regions_out: list[dict[str, Any]] = []
        for idx, reg in enumerate(preset_info.get("regions", [])):
            rx1 = float(max(0.0, min(1.0, reg["x1"])))
            ry1 = float(max(0.0, min(1.0, reg["y1"])))
            rx2 = float(max(rx1 + 0.02, min(1.0, reg["x2"])))
            ry2 = float(max(ry1 + 0.02, min(1.0, reg["y2"])))
            px1 = int(round(rx1 * w_px))
            py1 = int(round(ry1 * h_px))
            px2 = int(round(rx2 * w_px))
            py2 = int(round(ry2 * h_px))
            regions_out.append(
                {
                    "index": idx + 1,
                    "label": str(reg.get("label", f"Region {idx + 1}")),
                    "x1": rx1,
                    "y1": ry1,
                    "x2": rx2,
                    "y2": ry2,
                    "weight": float(reg.get("weight", 1.0)),
                    "bbox_px": (px1, py1, px2, py2),
                }
            )

        spatial_layout = {
            "preset_name": ensemble_layout_preset,
            "canvas_width": w_px,
            "canvas_height": h_px,
            "hint": layout_hint,
            "regions": regions_out,
        }
        stats["ensemble_layout_preset"] = ensemble_layout_preset
        stats["interaction_mode"] = interaction_mode
        stats["companion_or_prop"] = prop_expanded
        stats["regions_count"] = len(regions_out)

        return (
            compiled,
            base_scene_prompt,
            c1_det,
            c2_det,
            c3_det,
            spatial_layout,
            json.dumps(stats, indent=2, ensure_ascii=False),
        )


# ---------------------------------------------------------------------------
# Upgrade A Node 2: ZiggartSpatialLayoutDesigner
# ---------------------------------------------------------------------------


def _render_spatial_layout_preview(
    canvas_width: int,
    canvas_height: int,
    regions: list[dict[str, Any]],
    background_image: torch.Tensor | None = None,
) -> torch.Tensor:
    """Renders a color-coded visual diagram of the spatial bounding boxes as a ComfyUI IMAGE [1, H, W, 3]."""
    w = max(64, int(canvas_width))
    h = max(64, int(canvas_height))
    if isinstance(background_image, torch.Tensor) and background_image.ndim == 4 and background_image.shape[0] > 0:
        bg_np = (background_image[0, :, :, :3].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        base_img = Image.fromarray(bg_np, mode="RGB").resize((w, h), Image.Resampling.BILINEAR).convert("RGBA")
    else:
        base_img = Image.new("RGBA", (w, h), (22, 24, 32, 255))
        grid_draw = ImageDraw.Draw(base_img)
        step_x = max(32, w // 8)
        step_y = max(32, h // 8)
        for gx in range(0, w, step_x):
            grid_draw.line([(gx, 0), (gx, h)], fill=(45, 50, 66, 255), width=1)
        for gy in range(0, h, step_y):
            grid_draw.line([(0, gy), (w, gy)], fill=(45, 50, 66, 255), width=1)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    palette = [
        ((255, 92, 124, 72), (255, 110, 140, 240)),   # Region 1: Coral Red
        ((72, 185, 255, 72), (96, 205, 255, 240)),    # Region 2: Cyan Blue
        ((110, 235, 145, 72), (130, 250, 165, 240)),  # Region 3: Emerald Green
        ((255, 200, 85, 72), (255, 215, 110, 240)),   # Region 4: Amber Gold
    ]

    for idx, reg in enumerate(regions):
        fill_col, border_col = palette[idx % len(palette)]
        px1, py1, px2, py2 = reg["bbox_px"]
        draw.rectangle([px1, py1, px2, py2], fill=fill_col, outline=border_col, width=max(2, min(w, h) // 180))
        label_str = f"#{idx + 1} {reg.get('label', f'Region {idx + 1}')} (w={reg.get('weight', 1.0):.2f})"
        tag_h = 22
        tag_w = min(w - px1, max(90, len(label_str) * 7 + 12))
        draw.rectangle([px1, py1, px1 + tag_w, min(h, py1 + tag_h)], fill=border_col)
        draw.text((px1 + 6, py1 + 4), label_str, fill=(15, 18, 24, 255))

    comp = Image.alpha_composite(base_img, overlay).convert("RGB")
    arr = np.asarray(comp, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class ZiggartSpatialLayoutDesigner:
    """
    Visual and numeric 2D Bounding Box Designer for multi-character scenes.
    Outputs a ZIGGART_SPATIAL_LAYOUT that connects optionally to ZiggartRegionalConditioningComposer,
    plus ZIGGART_BBOX pixel boxes for Detailers and a color-coded preview IMAGE.
    """

    DESCRIPTION = (
        "Design up to 4 normalized [x1, y1, x2, y2] spatial regions (Char 1, Char 2, Char 3, Prop/BG) "
        "using sliders or the interactive 2D canvas widget. Connects optionally to "
        "ZiggartRegionalConditioningComposer or Ziggart Detailers."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "design_layout"
    RETURN_TYPES = ("ZIGGART_SPATIAL_LAYOUT", "ZIGGART_BBOX", "IMAGE", "STRING")
    RETURN_NAMES = ("spatial_layout", "bboxes", "layout_preview_image", "layout_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "active_regions": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "r1_label": ("STRING", {"default": "Char 1 (Left)"}),
                "r1_x1": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r1_y1": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r1_x2": ("FLOAT", {"default": 0.52, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r1_y2": ("FLOAT", {"default": 0.96, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r1_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "r2_label": ("STRING", {"default": "Char 2 (Right)"}),
                "r2_x1": ("FLOAT", {"default": 0.48, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r2_y1": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r2_x2": ("FLOAT", {"default": 0.98, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r2_y2": ("FLOAT", {"default": 0.96, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r2_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "r3_label": ("STRING", {"default": "Char 3 (Center)"}),
                "r3_x1": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r3_y1": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r3_x2": ("FLOAT", {"default": 0.70, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r3_y2": ("FLOAT", {"default": 0.98, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r3_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "r4_label": ("STRING", {"default": "Prop / Foreground"}),
                "r4_x1": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r4_y1": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 0.98, "step": 0.01}),
                "r4_x2": ("FLOAT", {"default": 0.80, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r4_y2": ("FLOAT", {"default": 0.98, "min": 0.02, "max": 1.0, "step": 0.01}),
                "r4_weight": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "background_reference_image": ("IMAGE",),
                "layout_json_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional JSON array of regions [{x1,y1,x2,y2,weight,label}] to override slider values.",
                    },
                ),
            },
        }

    def design_layout(
        self,
        canvas_width: int = 1024,
        canvas_height: int = 1024,
        active_regions: int = 2,
        r1_label: str = "Char 1 (Left)",
        r1_x1: float = 0.02,
        r1_y1: float = 0.05,
        r1_x2: float = 0.52,
        r1_y2: float = 0.96,
        r1_weight: float = 1.0,
        r2_label: str = "Char 2 (Right)",
        r2_x1: float = 0.48,
        r2_y1: float = 0.05,
        r2_x2: float = 0.98,
        r2_y2: float = 0.96,
        r2_weight: float = 1.0,
        r3_label: str = "Char 3 (Center)",
        r3_x1: float = 0.30,
        r3_y1: float = 0.04,
        r3_x2: float = 0.70,
        r3_y2: float = 0.98,
        r3_weight: float = 1.0,
        r4_label: str = "Prop / Foreground",
        r4_x1: float = 0.20,
        r4_y1: float = 0.55,
        r4_x2: float = 0.80,
        r4_y2: float = 0.98,
        r4_weight: float = 0.9,
        background_reference_image: torch.Tensor | None = None,
        layout_json_override: str = "",
    ):
        w_px = max(64, int(canvas_width))
        h_px = max(64, int(canvas_height))
        raw_specs = [
            (r1_label, r1_x1, r1_y1, r1_x2, r1_y2, r1_weight),
            (r2_label, r2_x1, r2_y1, r2_x2, r2_y2, r2_weight),
            (r3_label, r3_x1, r3_y1, r3_x2, r3_y2, r3_weight),
            (r4_label, r4_x1, r4_y1, r4_x2, r4_y2, r4_weight),
        ][: max(1, min(4, int(active_regions)))]

        if layout_json_override and layout_json_override.strip():
            try:
                parsed = json.loads(layout_json_override.strip())
                if isinstance(parsed, dict) and "regions" in parsed:
                    parsed = parsed["regions"]
                if isinstance(parsed, list) and parsed:
                    raw_specs = [
                        (
                            str(item.get("label", f"Region {i + 1}")),
                            float(item.get("x1", 0.0)),
                            float(item.get("y1", 0.0)),
                            float(item.get("x2", 1.0)),
                            float(item.get("y2", 1.0)),
                            float(item.get("weight", 1.0)),
                        )
                        for i, item in enumerate(parsed[:4])
                        if isinstance(item, dict)
                    ]
            except Exception as exc:
                logging.warning(f"[Ziggart Layout Designer] Invalid layout_json_override ignored: {exc}")

        regions_out: list[dict[str, Any]] = []
        bboxes_out: list[tuple[int, int, int, int]] = []
        for idx, (lbl, x1, y1, x2, y2, weight) in enumerate(raw_specs):
            nx1 = float(max(0.0, min(0.98, min(x1, x2))))
            ny1 = float(max(0.0, min(0.98, min(y1, y2))))
            nx2 = float(max(nx1 + 0.02, min(1.0, max(x1, x2))))
            ny2 = float(max(ny1 + 0.02, min(1.0, max(y1, y2))))
            px1 = int(round(nx1 * w_px))
            py1 = int(round(ny1 * h_px))
            px2 = max(px1 + 8, int(round(nx2 * w_px)))
            py2 = max(py1 + 8, int(round(ny2 * h_px)))
            bboxes_out.append((px1, py1, px2, py2))
            regions_out.append(
                {
                    "index": idx + 1,
                    "label": str(lbl or f"Region {idx + 1}"),
                    "x1": round(nx1, 4),
                    "y1": round(ny1, 4),
                    "x2": round(nx2, 4),
                    "y2": round(ny2, 4),
                    "weight": round(float(weight), 3),
                    "bbox_px": (px1, py1, px2, py2),
                }
            )

        spatial_layout = {
            "preset_name": "Custom Spatial Layout",
            "canvas_width": w_px,
            "canvas_height": h_px,
            "hint": f"custom {len(regions_out)}-region spatial layout",
            "regions": regions_out,
        }
        preview_img = _render_spatial_layout_preview(w_px, h_px, regions_out, background_reference_image)
        return (
            spatial_layout,
            bboxes_out,
            preview_img,
            json.dumps(spatial_layout, indent=2, ensure_ascii=False),
        )


# ---------------------------------------------------------------------------
# Upgrade A Node 3: ZiggartRegionalConditioningComposer
# ---------------------------------------------------------------------------


def _build_feathered_region_mask(
    canvas_w: int,
    canvas_h: int,
    bbox_px: tuple[int, int, int, int],
    feather_ratio: float = 0.10,
) -> torch.Tensor:
    """Creates a [1, H, W] float32 mask for a region with smooth cosine/linear edge feathering."""
    mask = torch.zeros((canvas_h, canvas_w), dtype=torch.float32)
    x1, y1, x2, y2 = bbox_px
    x1 = max(0, min(canvas_w - 1, int(x1)))
    y1 = max(0, min(canvas_h - 1, int(y1)))
    x2 = max(x1 + 1, min(canvas_w, int(x2)))
    y2 = max(y1 + 1, min(canvas_h, int(y2)))

    rw = x2 - x1
    rh = y2 - y1
    region_box = torch.ones((rh, rw), dtype=torch.float32)

    if feather_ratio > 0.0:
        fx = max(1, int(round(rw * min(0.45, float(feather_ratio)))))
        fy = max(1, int(round(rh * min(0.45, float(feather_ratio)))))
        if fx > 1 and rw > 2 * fx:
            ramp_x = torch.linspace(0.0, 1.0, steps=fx, dtype=torch.float32)
            region_box[:, :fx] *= ramp_x.unsqueeze(0)
            region_box[:, -fx:] *= ramp_x.flip(0).unsqueeze(0)
        if fy > 1 and rh > 2 * fy:
            ramp_y = torch.linspace(0.0, 1.0, steps=fy, dtype=torch.float32)
            region_box[:fy, :] *= ramp_y.unsqueeze(1)
            region_box[-fy:, :] *= ramp_y.flip(0).unsqueeze(1)

    mask[y1:y2, x1:x2] = region_box
    return mask.unsqueeze(0)


def _encode_clip_text(clip: Any, text: str) -> list[Any]:
    """Encodes a prompt string into ComfyUI CONDITIONING format."""
    if comfy_nodes is not None and hasattr(comfy_nodes, "CLIPTextEncode"):
        return list(comfy_nodes.CLIPTextEncode().encode(clip=clip, text=text)[0])
    if hasattr(clip, "tokenize") and hasattr(clip, "encode_from_tokens"):
        tokens = clip.tokenize(text)
        res = clip.encode_from_tokens(tokens, return_pooled=True)
        if isinstance(res, tuple) and len(res) == 2:
            cond, pooled = res
        else:
            cond, pooled = res, None
        return [[cond, {"pooled_output": pooled}]]
    raise RuntimeError("Invalid CLIP object passed to ZiggartRegionalConditioningComposer.")


def _apply_conditioning_mask(
    conditioning: list[Any],
    mask: torch.Tensor,
    strength: float = 1.0,
    set_area_to_bounds: bool = False,
) -> list[Any]:
    """Applies a spatial mask to a ComfyUI CONDITIONING list."""
    if comfy_nodes is not None and hasattr(comfy_nodes, "ConditioningSetMask"):
        set_bounds_str = "mask bounds" if set_area_to_bounds else "default"
        return list(
            comfy_nodes.ConditioningSetMask().append(
                conditioning=conditioning,
                mask=mask,
                set_cond_area=set_bounds_str,
                strength=float(strength),
            )[0]
        )
    out: list[Any] = []
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    for t in conditioning:
        n = [t[0], t[1].copy()]
        n[1]["mask"] = mask
        n[1]["set_area_to_bounds"] = bool(set_area_to_bounds)
        n[1]["mask_strength"] = float(strength)
        out.append(n)
    return out


class ZiggartRegionalConditioningComposer:
    """
    Encodes a global base scene prompt plus up to 4 regional prompts (Char 1, Char 2, Char 3, Prop)
    and attaches feathered spatial masks so multi-character outfits/attributes never bleed across subjects.
    Uses `spatial_layout` when connected, or falls back to `layout_preset` when disconnected.
    """

    DESCRIPTION = (
        "Eliminates multi-character attribute bleeding by composing spatially masked ComfyUI CONDITIONING "
        "for Character 1, Character 2, Character 3, and Prop/Region 4. Connects optionally to "
        "ZiggartSpatialLayoutDesigner or uses built-in layout presets."
    )
    CATEGORY = "Ziggart/Krea2 Prompt Director"
    FUNCTION = "compose_regional_conditioning"
    RETURN_TYPES = ("CONDITIONING", "IMAGE", "ZIGGART_BBOX", "STRING")
    RETURN_NAMES = (
        "regional_conditioning",
        "region_masks_preview",
        "region_bboxes",
        "layout_summary_json",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "base_scene_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Global scene context (style, environment, lighting, camera) applied across the full canvas.",
                    },
                ),
                "char_1_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Prompt for Region 1 / Character 1.",
                    },
                ),
                "layout_preset": (
                    list(ENSEMBLE_SPATIAL_LAYOUT_PRESETS.keys()),
                    {
                        "default": "Left / Right Split (Duo)",
                        "tooltip": "Used automatically when the optional `spatial_layout` input is not connected.",
                    },
                ),
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "base_prompt_strength": (
                    "FLOAT",
                    {"default": 0.40, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "region_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.05},
                ),
                "feather_ratio": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 0.45, "step": 0.01},
                ),
                "set_cond_area_to_bounds": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "char_2_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "char_3_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "region_4_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "spatial_layout": (
                    "ZIGGART_SPATIAL_LAYOUT",
                    {
                        "tooltip": "Optional custom layout from ZiggartSpatialLayoutDesigner or ZiggartKrea2TrioEnsembleDirector. Overrides layout_preset when connected.",
                    },
                ),
                "base_conditioning": ("CONDITIONING",),
            },
        }

    def compose_regional_conditioning(
        self,
        clip: Any,
        base_scene_prompt: str,
        char_1_prompt: str,
        layout_preset: str = "Left / Right Split (Duo)",
        canvas_width: int = 1024,
        canvas_height: int = 1024,
        base_prompt_strength: float = 0.40,
        region_strength: float = 1.0,
        feather_ratio: float = 0.10,
        set_cond_area_to_bounds: bool = False,
        char_2_prompt: str = "",
        char_3_prompt: str = "",
        region_4_prompt: str = "",
        spatial_layout: dict[str, Any] | None = None,
        base_conditioning: list[Any] | None = None,
    ):
        if isinstance(spatial_layout, dict) and spatial_layout.get("regions"):
            w_px = max(64, int(spatial_layout.get("canvas_width", canvas_width)))
            h_px = max(64, int(spatial_layout.get("canvas_height", canvas_height)))
            active_preset_name = str(spatial_layout.get("preset_name", "Connected Spatial Layout"))
            raw_regions = list(spatial_layout["regions"])
        else:
            w_px = max(64, int(canvas_width))
            h_px = max(64, int(canvas_height))
            active_preset_name = layout_preset
            preset_info = ENSEMBLE_SPATIAL_LAYOUT_PRESETS.get(
                layout_preset,
                ENSEMBLE_SPATIAL_LAYOUT_PRESETS["Left / Right Split (Duo)"],
            )
            raw_regions = []
            for idx, r in enumerate(preset_info.get("regions", [])):
                rx1, ry1, rx2, ry2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
                raw_regions.append(
                    {
                        "index": idx + 1,
                        "label": str(r.get("label", f"Region {idx + 1}")),
                        "x1": rx1,
                        "y1": ry1,
                        "x2": rx2,
                        "y2": ry2,
                        "weight": float(r.get("weight", 1.0)),
                        "bbox_px": (
                            int(round(rx1 * w_px)),
                            int(round(ry1 * h_px)),
                            int(round(rx2 * w_px)),
                            int(round(ry2 * h_px)),
                        ),
                    }
                )

        combined_cond: list[Any] = []
        full_mask = torch.ones((1, h_px, w_px), dtype=torch.float32)

        if base_conditioning is not None:
            if base_prompt_strength > 0.0:
                combined_cond.extend(
                    _apply_conditioning_mask(
                        list(base_conditioning),
                        full_mask,
                        strength=float(base_prompt_strength),
                        set_area_to_bounds=False,
                    )
                )
        elif base_scene_prompt.strip() and base_prompt_strength > 0.0:
            base_encoded = _encode_clip_text(clip, base_scene_prompt.strip())
            combined_cond.extend(
                _apply_conditioning_mask(
                    base_encoded,
                    full_mask,
                    strength=float(base_prompt_strength),
                    set_area_to_bounds=False,
                )
            )

        region_prompts = [
            char_1_prompt.strip(),
            char_2_prompt.strip(),
            char_3_prompt.strip(),
            region_4_prompt.strip(),
        ]
        used_regions_summary: list[dict[str, Any]] = []
        bboxes_out: list[tuple[int, int, int, int]] = []
        preview_regions: list[dict[str, Any]] = []

        for idx, prompt_str in enumerate(region_prompts):
            if not prompt_str or idx >= len(raw_regions):
                continue
            reg = raw_regions[idx]
            bbox_px = tuple(reg.get("bbox_px") or (
                int(round(float(reg["x1"]) * w_px)),
                int(round(float(reg["y1"]) * h_px)),
                int(round(float(reg["x2"]) * w_px)),
                int(round(float(reg["y2"]) * h_px)),
            ))
            eff_strength = float(region_strength) * float(reg.get("weight", 1.0))
            reg_mask = _build_feathered_region_mask(w_px, h_px, bbox_px, feather_ratio=feather_ratio)
            reg_cond = _encode_clip_text(clip, prompt_str)
            masked_cond = _apply_conditioning_mask(
                reg_cond,
                reg_mask,
                strength=eff_strength,
                set_area_to_bounds=set_cond_area_to_bounds,
            )
            combined_cond.extend(masked_cond)
            bboxes_out.append(bbox_px)
            reg_copy = dict(reg)
            reg_copy["bbox_px"] = bbox_px
            preview_regions.append(reg_copy)
            used_regions_summary.append(
                {
                    "region_index": idx + 1,
                    "label": reg.get("label", f"Region {idx + 1}"),
                    "bbox_px": bbox_px,
                    "effective_strength": round(eff_strength, 3),
                    "prompt_preview": prompt_str[:120],
                }
            )

        if not combined_cond:
            fallback_text = base_scene_prompt.strip() or char_1_prompt.strip() or ""
            combined_cond = _encode_clip_text(clip, fallback_text)

        preview_img = _render_spatial_layout_preview(w_px, h_px, preview_regions or raw_regions)
        summary = {
            "layout_source": "spatial_layout_input" if (isinstance(spatial_layout, dict) and spatial_layout.get("regions")) else "layout_preset",
            "active_preset": active_preset_name,
            "canvas_size": [w_px, h_px],
            "feather_ratio": feather_ratio,
            "active_regions_encoded": len(used_regions_summary),
            "regions": used_regions_summary,
        }
        return (
            combined_cond,
            preview_img,
            bboxes_out,
            json.dumps(summary, indent=2, ensure_ascii=False),
        )


# ---------------------------------------------------------------------------
# Upgrade A Node 4: ZiggartPromptTokenInspectorAndCompressor
# ---------------------------------------------------------------------------

_CONFLICT_TAG_PAIRS: list[tuple[str, set[str], set[str]]] = [
    ("lighting_time", {"bright daylight", "high noon", "midday sun"}, {"pitch black night", "midnight darkness", "moonlit night"}),
    ("camera_framing", {"extreme close-up", "macro portrait"}, {"wide establishing shot", "panoramic full body", "full body shot"}),
    ("color_mode", {"monochrome", "black and white", "1-bit"}, {"vibrant technicolor", "vivid neon palette", "saturated colors"}),
    ("art_medium", {"3d octane render", "unreal engine 5"}, {"flat 2d cel animation", "woodblock print", "1-bit pixel art"}),
]

_FILLER_WORDS_RE = re.compile(
    r"\b(?:very|extremely|super|really|quite|utterly|highly|truly|remarkably|exceptionally)\s+",
    re.IGNORECASE,
)


class ZiggartPromptTokenInspectorAndCompressor:
    """
    Analyzes compiled prompts for CLIP (77-token chunks) and T5/Qwen3VL token budgets,
    flags duplicate or contradictory tags across wildcards, and losslessly compresses
    over-budget prompts while protecting character locks ([CHAR 1], [CHAR 2], [CHAR 3]).
    """

    DESCRIPTION = (
        "Inspects CLIP (77-token chunks) & Qwen3VL/T5 token budgets, detects duplicate or conflicting "
        "wildcard clauses, and compacts over-budget prompts while preserving character locks."
    )
    CATEGORY = "Ziggart/Wildcards"
    FUNCTION = "inspect_and_compress"
    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "optimized_prompt",
        "estimated_token_count",
        "clip_chunk_count",
        "inspector_report_json",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "target_token_budget": (
                    "INT",
                    {"default": 220, "min": 30, "max": 2048, "step": 5},
                ),
                "compression_mode": (
                    ["deduplicate_only", "smart_compact", "strict_clip_77"],
                    {"default": "smart_compact"},
                ),
                "preserve_character_locks": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    def inspect_and_compress(
        self,
        prompt_text: str,
        target_token_budget: int = 220,
        compression_mode: str = "smart_compact",
        preserve_character_locks: bool = True,
        clip: Any = None,
    ) -> tuple[str, int, int, str]:
        raw = (prompt_text or "").strip()
        if not raw:
            empty_rep = {
                "original_tokens": 0,
                "optimized_tokens": 0,
                "clip_77_chunks": 0,
                "duplicates_removed": [],
                "conflicts_detected": [],
                "compression_mode": compression_mode,
            }
            return ("", 0, 0, json.dumps(empty_rep, indent=2))

        budget = 75 if compression_mode == "strict_clip_77" else max(30, int(target_token_budget))
        orig_tokens = estimate_qwen_tokens(raw)

        # Detect conflicts
        low_raw = raw.lower()
        conflicts: list[dict[str, Any]] = []
        for group_name, set_a, set_b in _CONFLICT_TAG_PAIRS:
            found_a = [a for a in set_a if a in low_raw]
            found_b = [b for b in set_b if b in low_raw]
            if found_a and found_b:
                conflicts.append(
                    {
                        "category": group_name,
                        "group_a_matches": found_a,
                        "group_b_matches": found_b,
                    }
                )

        # Split into top-level clauses (preserving parenthesized blocks if present)
        paren_blocks = re.findall(r"\([^()]+\)|[^,]+", raw)
        seen_clauses: set[str] = set()
        duplicates_removed: list[str] = []
        deduped_clauses: list[str] = []

        for block in paren_blocks:
            b_clean = block.strip().strip(",").strip()
            if not b_clean:
                continue
            norm_key = re.sub(r"\s+", " ", b_clean.lower().strip("(). "))
            is_locked = preserve_character_locks and (
                "[char" in norm_key
                or norm_key.startswith("1 ")
                or norm_key.startswith("2 ")
                or norm_key.startswith("3 ")
                or "wearing " in norm_key
            )
            if norm_key in seen_clauses and not is_locked:
                duplicates_removed.append(b_clean)
                continue
            seen_clauses.add(norm_key)
            deduped_clauses.append(b_clean)

        uses_scoped_parens = any(c.startswith("(") and c.endswith(")") for c in deduped_clauses)
        sep = " " if uses_scoped_parens else ", "
        working = sep.join(deduped_clauses)

        if compression_mode in ("smart_compact", "strict_clip_77") and estimate_qwen_tokens(working) > budget:
            working = _FILLER_WORDS_RE.sub("", working)
            working = re.sub(r"\s{2,}", " ", working).strip()

        if compression_mode in ("smart_compact", "strict_clip_77") and estimate_qwen_tokens(working) > budget:
            kept: list[str] = []
            running = 0
            for clause in deduped_clauses:
                c_mod = _FILLER_WORDS_RE.sub("", clause).strip()
                c_tok = estimate_qwen_tokens(c_mod)
                is_locked = preserve_character_locks and (
                    "[char" in c_mod.lower() or "wearing " in c_mod.lower()
                )
                if running + c_tok <= budget or is_locked or not kept:
                    kept.append(c_mod)
                    running += c_tok
            working = sep.join(kept)

        if working and not working.endswith((".", "!", "?", '"', ")")):
            working += "."

        final_tokens = estimate_qwen_tokens(working)
        if clip is not None and hasattr(clip, "tokenize"):
            try:
                tok_dict = clip.tokenize(working)
                if isinstance(tok_dict, dict):
                    first_branch = next(iter(tok_dict.values()))
                    clip_chunks = len(first_branch) if isinstance(first_branch, list) else 1
                elif isinstance(tok_dict, list):
                    clip_chunks = len(tok_dict)
                else:
                    clip_chunks = max(1, (final_tokens + 74) // 75)
            except Exception:
                clip_chunks = max(1, (final_tokens + 74) // 75)
        else:
            clip_chunks = max(1, (final_tokens + 74) // 75)

        report = {
            "original_tokens": orig_tokens,
            "optimized_tokens": final_tokens,
            "tokens_saved": max(0, orig_tokens - final_tokens),
            "target_token_budget": budget,
            "within_budget": final_tokens <= budget,
            "clip_77_chunks": clip_chunks,
            "duplicates_removed_count": len(duplicates_removed),
            "duplicates_removed": duplicates_removed,
            "conflicts_detected": conflicts,
            "compression_mode": compression_mode,
        }
        return (
            working,
            int(final_tokens),
            int(clip_chunks),
            json.dumps(report, indent=2, ensure_ascii=False),
        )


# ---------------------------------------------------------------------------
# Shared Preview Resolution Helper & HTTP Endpoints for ComfyUI Frontend
# ---------------------------------------------------------------------------


def _resolve_node_preview_prompt(cls_type: str, inputs: dict[str, Any], seed_val: int) -> str:
    """
    Resolves the preview prompt for ZiggartWildcardProcessor, ZiggartWildcardProcessorAllInOne,
    ZiggartWildcardEncode, or ZiggartKrea2MasterPromptDirector without running diffusion sampling.
    """
    custom_dir = str(inputs.get("custom_wildcards_dir", "") or "")
    wc_text = str(inputs.get("wildcard_text", "") or "").strip()

    if cls_type == "ZiggartKrea2MasterPromptDirector":
        director = ZiggartKrea2MasterPromptDirector()
        compiled, _, _, _, _ = director.direct_prompt(
            source_mode=str(inputs.get("source_mode", "Theme (Collection/Themes)")),
            theme_selection=str(inputs.get("theme_selection", RANDOM_THEME_OPTION)),
            universe_selection=str(inputs.get("universe_selection", RANDOM_UNIVERSE_OPTION)),
            cast_mode=str(inputs.get("cast_mode", "Solo (1 Character)")),
            krea2_policy=str(inputs.get("krea2_policy", "scoped (Parentheses Blocks + Count Anchor)")),
            separate_parentheses_per_facet=bool(inputs.get("separate_parentheses_per_facet", True)),
            include_count_anchor=bool(inputs.get("include_count_anchor", True)),
            mood_filter=str(inputs.get("mood_filter", "[Any / No Filter]")),
            exclude_full_prompts=bool(inputs.get("exclude_full_prompts", True)),
            style_wildcard=str(inputs.get("style_wildcard", FROM_CONTEXT_OPTION)),
            char_1_gender=str(inputs.get("char_1_gender", "Auto / Any")),
            char_1_wildcard=str(inputs.get("char_1_wildcard", FROM_CONTEXT_OPTION)),
            char_1_outfit_link_mode=str(inputs.get("char_1_outfit_link_mode", "Canonical (1:1 Strict Character Outfit)")),
            char_1_outfit_wildcard=str(inputs.get("char_1_outfit_wildcard", FROM_CONTEXT_OPTION)),
            char_1_enable_anatomy=bool(inputs.get("char_1_enable_anatomy", False)),
            char_1_auto_filter_anatomy=bool(inputs.get("char_1_auto_filter_anatomy", True)),
            char_1_anatomy_wildcard=str(inputs.get("char_1_anatomy_wildcard", RANDOM_POOL_OPTION)),
            char_2_gender=str(inputs.get("char_2_gender", "Auto / Any")),
            char_2_wildcard=str(inputs.get("char_2_wildcard", FROM_CONTEXT_OPTION)),
            char_2_outfit_link_mode=str(inputs.get("char_2_outfit_link_mode", "Canonical (1:1 Strict Character Outfit)")),
            char_2_outfit_wildcard=str(inputs.get("char_2_outfit_wildcard", FROM_CONTEXT_OPTION)),
            char_2_enable_anatomy=bool(inputs.get("char_2_enable_anatomy", False)),
            char_2_auto_filter_anatomy=bool(inputs.get("char_2_auto_filter_anatomy", True)),
            char_2_anatomy_wildcard=str(inputs.get("char_2_anatomy_wildcard", RANDOM_POOL_OPTION)),
            action_or_interaction_wildcard=str(inputs.get("action_or_interaction_wildcard", FROM_CONTEXT_OPTION)),
            environment_wildcard=str(inputs.get("environment_wildcard", FROM_CONTEXT_OPTION)),
            lighting_wildcard=str(inputs.get("lighting_wildcard", RANDOM_POOL_OPTION)),
            camera_wildcard=str(inputs.get("camera_wildcard", RANDOM_POOL_OPTION)),
            populated_text="",
            mode="populate",
            seed=seed_val,
            search_or_keyword_filter=str(inputs.get("search_or_keyword_filter", "") or ""),
            custom_prefix=str(inputs.get("custom_prefix", "") or ""),
            custom_suffix=str(inputs.get("custom_suffix", "") or ""),
        )
        return compiled

    if cls_type == "ZiggartWildcardProcessorAllInOne":
        parts: list[str] = [wc_text] if wc_text else []
        combine_pillars = bool(inputs.get("combine_selected_pillars", True))
        if combine_pillars or not parts:
            for p_key in (
                "select_macro_yaml_theme",
                "select_character_anatomy",
                "select_outfit_pose",
                "select_env_vehicle",
                "select_style_camera_light",
                "select_mixed_other",
            ):
                sel = str(inputs.get(p_key, NONE_OPTION) or "")
                if sel and sel not in (NONE_OPTION, SELECT_PLACEHOLDER) and sel not in parts:
                    parts.append(sel)
        source_text = ", ".join(parts)
    else:
        sel_wc = str(inputs.get("select_to_add_wildcard", SELECT_PLACEHOLDER) or "")
        source_text = wc_text
        if not source_text and sel_wc not in (SELECT_PLACEHOLDER, NONE_OPTION, ""):
            source_text = sel_wc
        if not source_text:
            search_q = str(inputs.get("search_wildcard_filter", "") or "").strip()
            if search_q:
                matches = WILDCARD_INDEX.search_wildcards(search_q, limit=50, extra_dir=custom_dir)
                if matches:
                    rng = random.Random(seed_val & 0xFFFFFFFFFFFFFFFF)
                    source_text = rng.choice(matches)

    if not source_text:
        return ""
    return process_wildcards(source_text, seed=seed_val, extra_dir=custom_dir)


if PromptServer is not None and getattr(PromptServer, "instance", None) is not None and web is not None:
    @PromptServer.instance.routes.get("/ziggart/wildcards/search")
    async def ziggart_search_wildcards(request):
        q = request.rel_url.query.get("q", "")
        limit = int(request.rel_url.query.get("limit", "100"))
        matches = WILDCARD_INDEX.search_wildcards(q, limit=min(500, max(1, limit)))
        return web.json_response({"matches": matches, "count": len(matches)})

    @PromptServer.instance.routes.post("/ziggart/wildcards/process")
    async def ziggart_process_wildcard_api(request):
        data = await request.json()
        cls_type = str(data.get("class_type", "ZiggartWildcardProcessor") or "ZiggartWildcardProcessor")
        seed = int(data.get("seed", 0) or 0)
        inputs = data.get("inputs")
        if isinstance(inputs, dict):
            if "wildcard_text" not in inputs and "text" in data:
                inputs["wildcard_text"] = str(data.get("text", ""))
            expanded = _resolve_node_preview_prompt(cls_type, inputs, seed)
        else:
            text = str(data.get("text", ""))
            expanded = process_wildcards(text, seed=seed)
        return web.json_response({"populated_text": expanded, "used_seed": seed})

    def _ziggart_on_prompt_handler(json_data: dict[str, Any]) -> dict[str, Any]:
        """
        Automatically populates the `populated_text` widget in the ComfyUI UI the moment
        the user clicks 'Queue', matching ImpactWildcardProcessor's live feedback behavior.
        """
        try:
            prompt = json_data.get("prompt", {})
            updated_widgets: dict[str, str] = {}
            for node_id, node_info in prompt.items():
                if not isinstance(node_info, dict):
                    continue
                cls_type = node_info.get("class_type", "")
                if cls_type not in (
                    "ZiggartWildcardProcessor",
                    "ZiggartWildcardProcessorAllInOne",
                    "ZiggartWildcardEncode",
                    "ZiggartKrea2MasterPromptDirector",
                ):
                    continue
                inputs = node_info.get("inputs", {})
                mode = inputs.get("mode", "populate")
                if mode is True:
                    mode = "populate"
                elif mode is False:
                    mode = "fixed"

                existing_pop = str(inputs.get("populated_text", "") or "").strip()
                if mode == "fixed":
                    continue
                if mode == "reproduce" and existing_pop:
                    # Keep the pre-generated or previously populated text locked on Queue!
                    _send_node_feedback(node_id, "populated_text", existing_pop)
                    continue

                raw_seed = inputs.get("seed", 0)
                if isinstance(raw_seed, (int, float)):
                    seed_val = int(raw_seed)
                else:
                    continue

                expanded = _resolve_node_preview_prompt(cls_type, inputs, seed_val)
                if expanded:
                    inputs["populated_text"] = expanded
                    _send_node_feedback(node_id, "populated_text", expanded)
                    updated_widgets[str(node_id)] = expanded

            if updated_widgets:
                nodes_list = (
                    json_data.get("extra_data", {})
                    .get("extra_pnginfo", {})
                    .get("workflow", {})
                    .get("nodes", [])
                )
                if isinstance(nodes_list, list):
                    for n_obj in nodes_list:
                        if isinstance(n_obj, dict):
                            n_id = str(n_obj.get("id", ""))
                            w_vals = n_obj.get("widgets_values")
                            if n_id in updated_widgets and isinstance(w_vals, list) and len(w_vals) > 1:
                                w_vals[1] = updated_widgets[n_id]
        except Exception as exc:
            logging.debug(f"[Ziggart Wildcards] on_prompt handler warning: {exc}")
        return json_data

    if hasattr(PromptServer.instance, "add_on_prompt_handler"):
        PromptServer.instance.add_on_prompt_handler(_ziggart_on_prompt_handler)


NODE_CLASS_MAPPINGS = {
    "ZiggartWildcardProcessor": ZiggartWildcardProcessor,
    "ZiggartWildcardProcessorAllInOne": ZiggartWildcardProcessorAllInOne,
    "ZiggartCollectionSceneBuilder": ZiggartCollectionSceneBuilder,
    "ZiggartWildcardEncode": ZiggartWildcardEncode,
    "ZiggartCanonicalThemeResolver": ZiggartCanonicalThemeResolver,
    "ZiggartWildcardVLMEnhancer": ZiggartWildcardVLMEnhancer,
    "ZiggartWildcardRecipeSaver": ZiggartWildcardRecipeSaver,
    "ZiggartThemeUniverseContext": ZiggartThemeUniverseContext,
    "ZiggartCharacterUnitBuilder": ZiggartCharacterUnitBuilder,
    "ZiggartAnatomyCharacterCreator": ZiggartAnatomyCharacterCreator,
    "ZiggartSceneStageAtmosphere": ZiggartSceneStageAtmosphere,
    "ZiggartKrea2PromptAssembler": ZiggartKrea2PromptAssembler,
    "ZiggartKrea2MasterPromptDirector": ZiggartKrea2MasterPromptDirector,
    "ZiggartKrea2TrioEnsembleDirector": ZiggartKrea2TrioEnsembleDirector,
    "ZiggartSpatialLayoutDesigner": ZiggartSpatialLayoutDesigner,
    "ZiggartRegionalConditioningComposer": ZiggartRegionalConditioningComposer,
    "ZiggartPromptTokenInspectorAndCompressor": ZiggartPromptTokenInspectorAndCompressor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartWildcardProcessor": "Ziggart Fast Wildcard Processor",
    "ZiggartWildcardProcessorAllInOne": "Ziggart Wildcard Processor (All-in-One Pillars & Mixed)",
    "ZiggartCollectionSceneBuilder": "Ziggart Collection Scene Builder",
    "ZiggartWildcardEncode": "Ziggart Fast Wildcard Encode (LoRA)",
    "ZiggartCanonicalThemeResolver": "Ziggart Canonical Theme & Character-Outfit Resolver",
    "ZiggartWildcardVLMEnhancer": "Wildcard-to-VLM Prompt Harmonizer",
    "ZiggartWildcardRecipeSaver": "Ziggart Prompt Recipe & Comparison Saver",
    "ZiggartThemeUniverseContext": "Ziggart Krea2 Theme & Universe Context",
    "ZiggartCharacterUnitBuilder": "Ziggart Krea2 Character Unit Builder (Outfit & Anatomy Filter)",
    "ZiggartAnatomyCharacterCreator": "Ziggart Krea2 Head-to-Toe Anatomy Character Creator",
    "ZiggartSceneStageAtmosphere": "Ziggart Krea2 Scene Stage (Style, Action, Env, Light, Camera)",
    "ZiggartKrea2PromptAssembler": "Ziggart Krea2 Prompt Assembler (Qwen3VL-4B)",
    "ZiggartKrea2MasterPromptDirector": "Ziggart Krea2 Master Prompt Director (All-in-One)",
    "ZiggartKrea2TrioEnsembleDirector": "Ziggart Krea2 Trio & Ensemble Director (3-Char + Spatial Layout)",
    "ZiggartSpatialLayoutDesigner": "Ziggart Spatial Layout & BBox Designer (Interactive Regions)",
    "ZiggartRegionalConditioningComposer": "Ziggart Regional Conditioning Composer (Multi-Char Masking)",
    "ZiggartPromptTokenInspectorAndCompressor": "Ziggart Prompt Token Inspector & Compressor (CLIP / Qwen3VL)",
}
