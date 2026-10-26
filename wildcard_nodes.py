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


NONE_OPTION = "[None]"
SELECT_PLACEHOLDER = "[Select a Wildcard to Insert or Resolve]"
LORA_PLACEHOLDER = "[Select a LoRA to Insert]"
RANDOM_THEME_OPTION = "[Random Theme]"

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

            line_str = raw_line.decode("utf-8", errors="replace")
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
        self.theme_folders: dict[str, str] = {}  # theme_name -> absolute folder path
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
            for k in self.pillars:
                self.pillars[k] = []

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

    def get_dropdown_list(self, extra_dir: str = "") -> list[str]:
        self.ensure_loaded(extra_dir=extra_dir)
        return [SELECT_PLACEHOLDER] + self._dropdown_items

    def get_pillar_dropdown(self, pillar_key: str, placeholder: str = NONE_OPTION, extra_dir: str = "") -> list[str]:
        self.ensure_loaded(extra_dir=extra_dir)
        items = self.pillars.get(pillar_key, [])
        return [placeholder] + (items if items else [SELECT_PLACEHOLDER])

    def get_theme_list(self, extra_dir: str = "") -> list[str]:
        self.ensure_loaded(extra_dir=extra_dir)
        names = sorted(self.theme_folders.keys())
        return [RANDOM_THEME_OPTION] + names

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

        if mode == "fixed" and populated_text.strip():
            final_text = process_wildcards(
                populated_text, seed=seed, extra_dir=custom_wildcards_dir
            )
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

        if mode == "fixed" and populated_text.strip():
            final_text = process_wildcards(
                populated_text, seed=seed, extra_dir=custom_wildcards_dir
            )
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
# Optional Fast HTTP Endpoints for ComfyUI Frontend Extension
# ---------------------------------------------------------------------------

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
        text = str(data.get("text", ""))
        seed = int(data.get("seed", 0))
        expanded = process_wildcards(text, seed=seed)
        return web.json_response({"populated_text": expanded})

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
                ):
                    continue
                inputs = node_info.get("inputs", {})
                mode = inputs.get("mode", "populate")
                if mode is True:
                    mode = "populate"
                elif mode is False:
                    mode = "fixed"

                if mode != "populate":
                    continue

                raw_seed = inputs.get("seed", 0)
                if isinstance(raw_seed, (int, float)):
                    seed_val = int(raw_seed)
                else:
                    continue

                custom_dir = str(inputs.get("custom_wildcards_dir", "") or "")
                wc_text = str(inputs.get("wildcard_text", "") or "").strip()

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

                if source_text:
                    expanded = process_wildcards(source_text, seed=seed_val, extra_dir=custom_dir)
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZiggartWildcardProcessor": "Ziggart Fast Wildcard Processor",
    "ZiggartWildcardProcessorAllInOne": "Ziggart Wildcard Processor (All-in-One Pillars & Mixed)",
    "ZiggartCollectionSceneBuilder": "Ziggart Collection Scene Builder",
    "ZiggartWildcardEncode": "Ziggart Fast Wildcard Encode (LoRA)",
    "ZiggartCanonicalThemeResolver": "Ziggart Canonical Theme & Character-Outfit Resolver",
    "ZiggartWildcardVLMEnhancer": "Wildcard-to-VLM Prompt Harmonizer",
    "ZiggartWildcardRecipeSaver": "Ziggart Prompt Recipe & Comparison Saver",
}
