"""Deterministic Krea2 (Qwen3VL-4B) Prompt Compiler, Character-Outfit Linker, and Outfit-Anatomy Occlusion Engine.

Self-contained module porting the proven Wildcard-GPT Krea2CompilerV2 (scoped, balanced, compact, rich),
the 1:1 Character-Outfits & Universe bindings, Duo/Trio interactions, and the automatic Outfit -> Anatomy
occlusion filter + Full-Prompt exclusion heuristics.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

KREA2_COMPILER_VERSION = "2.0.0"
KREA2_LONG_PROMPT_CHARS = 1600
KREA2_VERY_LONG_PROMPT_CHARS = 2000

_WILDCARD_RE = re.compile(r"__[^_\r\n]+?__")
_SPACE_RE = re.compile(r"\s+")
_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:])")
_REPEATED_COMMA_RE = re.compile(r",{2,}")
_REPEATED_DOT_RE = re.compile(r"\.{2,}")
_TOKEN_RE = re.compile(r"KREA2V2TOKEN(\d+)END")

# ---------------------------------------------------------------------------
# 1. Full-Scene Prompt Exclusion Rules (Files & Multi-Pillar Line Heuristics)
# ---------------------------------------------------------------------------

FULL_PROMPT_FILENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|/)(?:theme|themes)[-_]", re.I),
    re.compile(r"[-_]scene\d*$", re.I),
    re.compile(r"[-_]prompts?\d*$", re.I),
    re.compile(r"(?:^|/)(?:diorama|packaging|tcg|sheet)[-_]", re.I),
    re.compile(r"(?:^|/)test[-_ ]", re.I),
    re.compile(r"(?:^|/)(?:atester|concours|action-scenartio|incredible-scenario|romantic-scenario|prompt conception|new text document|wordflux|myrandom)$", re.I),
    re.compile(r"^collection/themes/[^/]+/all$", re.I),
    re.compile(r"^collection/universes/[^/]+/all$", re.I),
    re.compile(r"^collection/universes/[^/]+/scene-canonical$", re.I),
    re.compile(r"^collection/canonical/", re.I),
    re.compile(r"^collection/combos/(?!pools/)", re.I),
    re.compile(r"^collection/formats-wild/", re.I),
    re.compile(r"^collection/sheet/", re.I),
    re.compile(r"^collection/format-dev/", re.I),
    re.compile(r"^ziggart/scenes/", re.I),
    re.compile(r"^krea2", re.I),
    re.compile(r"/forbidden/", re.I),
    re.compile(r"age-infant|age-teen|mineur|ageplay", re.I),
)


def is_full_prompt_wildcard_key(norm_key: str) -> bool:
    """Returns True if `norm_key` points to a full pre-assembled scene prompt or unsafe/forbidden card."""
    k = norm_key.strip().replace("\\", "/").strip("/_ ").lower()
    if not k:
        return True
    for pat in FULL_PROMPT_FILENAME_PATTERNS:
        if pat.search(k):
            return True
    return False


_FULL_SCENE_LINE_MARKERS = re.compile(
    r"(?:\(\s*render\s+style\s*:|\bstudio\s+(?:medium|wide|close-up)\s+shot\b|\bcollectible\s+trading\s+card\b|\bmultiple\s+views,\s*character\s+model\s+sheet\b|\btilt-shift\s+miniature\s+diorama\b|\bcollectible\s+scale\s+model\s+in\s+a\s+clear\s+blister\b)",
    re.I,
)


def is_full_scene_line(line: str, max_words: int = 85) -> bool:
    """Detects whether a single line inside a wildcard file is actually an already-complete multi-pillar scene prompt."""
    s = line.strip()
    if not s:
        return False
    if _FULL_SCENE_LINE_MARKERS.search(s):
        return True
    words = s.split()
    if len(words) > max_words and s.count(",") >= 6 and s.count(".") >= 2:
        return True
    # Check if line references multiple distinct facet wildcards (e.g. Character + Outfit + Environment + Style)
    tags = _WILDCARD_RE.findall(s.lower())
    if len(tags) >= 3:
        facets = sum(
            1
            for f_kw in ("character", "outfit", "environment", "style", "pose")
            if any(f_kw in t for t in tags)
        )
        if facets >= 3:
            return True
    return False


# ---------------------------------------------------------------------------
# 2. Automatic Outfit -> Anatomy Occlusion Filter
# ---------------------------------------------------------------------------

ANATOMY_ZONES = (
    "BREASTS_CHEST",
    "NIPPLES_AREOLAS",
    "WAIST_MIDRIFF",
    "BUTTOCKS_HIPS",
    "INTIMATE_GENITALS",
)

# Keywords indicating the outfit is completely nude or absent
_NUDE_OUTFIT_RE = re.compile(
    r"\b(?:nude|naked|unclothed|bare\s+body|fully\s+naked|completely\s+nude|nothing\s+at\s+all|sky-clad|au\s+naturel|birthday\s+suit)\b",
    re.I,
)

# Keywords indicating sheer / see-through / wet / torn fabrics that reveal underlying anatomy
_SHEER_OR_TORN_RE = re.compile(
    r"\b(?:sheer|see-through|see\s+through|transparent|translucent|gossamer|mesh|fishnet|unlined\s+lace|wet|soaked|clinging|torn|shredded|ripped|battle-damaged|battle\s+damaged|tattered|strips)\b",
    re.I,
)

# Keywords covering the upper torso / chest completely (blocking breast contour & cleavage unless modified)
_HEAVY_CHEST_COVER_RE = re.compile(
    r"\b(?:full\s+plate|plate\s+armor|breastplate|cuirass|hauberk|chainmail\s+hauberk|heavy\s+coat|winter\s+coat|parka|trench\s+coat|cloak|cape\s+fastened|hoodie|sweater|turtleneck|spacesuit|power\s+armor|mech\s+suit|hazmat|padded\s+armor|gambeson|tunic|doublet|habit|cassock|robe|robes|kimono|combat\s+suit|tactical\s+vest)\b",
    re.I,
)

# Modifiers that explicitly reveal or emphasize breast contour / cleavage even with an upper garment
_CHEST_REVEAL_RE = re.compile(
    r"\b(?:cleavage|plunging|low-cut|low\s+cut|deep\s+v|open-chest|open\s+over\s+the\s+chest|open\s+shirt|open\s+front|bikini|micro-bikini|microbikini|bra|bralette|corset|bustier|lingerie|bodysuit|monokini|swimsuit|leotard|skin-tight|skintight|form-fitting|latex|pvc|spandex|underboob|sideboob|pasties|strapless|tube\s+top|halter\s+top|crop\s+top)\b",
    re.I,
)

# Any upper garment that covers the nipples unless sheer/topless/cupless
_UPPER_GARMENT_RE = re.compile(
    r"\b(?:shirt|blouse|tunic|top|dress|gown|robe|robes|armor|breastplate|cuirass|hauberk|jacket|coat|vest|doublet|sweater|hoodie|bodysuit|leotard|swimsuit|monokini|bikini|bra|bralette|corset|bustier|kimono|uniform|suit|jumpsuit|pasties|tape)\b",
    re.I,
)

# Explicit nipple exposure keywords
_NIPPLE_EXPOSED_RE = re.compile(
    r"\b(?:topless|bare\s+breasts|bare\s+chest|exposed\s+breasts?|exposed\s+chest|exposed\s+nipples?|open-cup|cupless|sheer|see-through|see\s+through|transparent|translucent|wet|soaked|clinging|torn|shredded|ripped|pulling\s+aside|unbuttoned)\b",
    re.I,
)

# Note: pasties cover ONLY the nipples -> BREASTS_CHEST is visible, NIPPLES_AREOLAS is covered!
_PASTIES_ONLY_RE = re.compile(r"\b(?:pasties|nipple\s+tape|x-shaped\s+tape)\b", re.I)

# Midriff / Waist coverage vs exposure
_MIDRIFF_COVER_RE = re.compile(
    r"\b(?:dress|gown|robe|robes|tunic|doublet|vest|jacket|hauberk|armor|breastplate|cuirass|coat|jumpsuit|bodysuit|leotard|one-piece|monokini|overalls|suit|sweater|shirt|blouse|hoodie|corset|kimono|uniform)\b",
    re.I,
)
_MIDRIFF_REVEAL_RE = re.compile(
    r"\b(?:bare\s+midriff|exposed\s+midriff|crop\s+top|cropped|bikini|two-piece|bra\s+and|pasties|loincloth|belly\s+dancer|low-rise|cutout|cut-out|open\s+front|topless|shirtless)\b",
    re.I,
)

# Buttocks / Hips / Legs coverage vs exposure
_GLUTES_HEAVY_COVER_RE = re.compile(
    r"\b(?:ballgown|long\s+skirt|floor-length|flowing\s+robes?|heavy\s+cloak|long\s+coat|trench\s+coat|travelling\s+cape|mantle|baggy\s+pants|cargo\s+pants|trousers|pants|jeans|plate\s+armor|tassets|greaves|riding\s+boots|high\s+leather\s+boots)\b",
    re.I,
)
_GLUTES_REVEAL_RE = re.compile(
    r"\b(?:thong|g-string|micro\s+skirt|micro-skirt|miniskirt|mini\s+skirt|hot\s+pants|booty\s+shorts|micro\s+shorts|shorts|high-cut|high\s+cut|cheeky|leggings|yoga\s+pants|latex|skin-tight|skintight|bodysuit|leotard|swimsuit|bikini|loincloth|bottomless|high-slit)\b",
    re.I,
)

# Intimate genital coverage vs exposure
_LOWER_GARMENT_RE = re.compile(
    r"\b(?:pants|trousers|jeans|shorts|hot\s+pants|skirt|miniskirt|dress|gown|robe|robes|tunic|bikini|swimsuit|monokini|bodysuit|leotard|panties|underwear|briefs|boxers|thong|g-string|loincloth|kilt|leggings|tights|armor|jumpsuit|suit)\b",
    re.I,
)
_INTIMATE_EXPOSED_RE = re.compile(
    r"\b(?:nude|naked|bottomless|crotchless|open-crotch|no\s+panties|no\s+underwear|pulled\s+aside|exposed\s+vulva|exposed\s+groin|spread\s+legs\s+naked|unclothed)\b",
    re.I,
)


@dataclass(frozen=True)
class OutfitOcclusionReport:
    """Result of analyzing an outfit description for anatomical visibility."""
    outfit_text: str
    is_nude: bool
    allowed_zones: dict[str, bool]
    blocked_zones: tuple[str, ...]
    visible_zones: tuple[str, ...]
    reason_summary: str


def analyze_outfit_occlusion(outfit_text: str, outfit_wildcard_key: str = "") -> OutfitOcclusionReport:
    """
    Deterministically analyzes `outfit_text` (and optional `outfit_wildcard_key`) to determine
    which of the 5 anatomical zones are visible vs covered.
    """
    combined = f"{outfit_wildcard_key} {outfit_text}".strip()
    if not combined or _NUDE_OUTFIT_RE.search(combined):
        allowed = {z: True for z in ANATOMY_ZONES}
        return OutfitOcclusionReport(
            outfit_text=outfit_text,
            is_nude=True,
            allowed_zones=allowed,
            blocked_zones=(),
            visible_zones=ANATOMY_ZONES,
            reason_summary="Nude / unclothed: all anatomical zones visible.",
        )

    is_sheer_or_torn = bool(_SHEER_OR_TORN_RE.search(combined))
    has_pasties = bool(_PASTIES_ONLY_RE.search(combined))

    # 1. BREASTS_CHEST (breast shape, cleavage, chest/torso contour)
    heavy_chest = bool(_HEAVY_CHEST_COVER_RE.search(combined))
    chest_reveal = bool(_CHEST_REVEAL_RE.search(combined)) or is_sheer_or_torn or has_pasties
    allow_breasts_chest = (not heavy_chest) or chest_reveal

    # 2. NIPPLES_AREOLAS (explicit nipple/areola detail)
    has_upper = bool(_UPPER_GARMENT_RE.search(combined))
    nipple_exposed = bool(_NIPPLE_EXPOSED_RE.search(combined))
    if has_pasties and not is_sheer_or_torn:
        allow_nipples = False
    elif not has_upper:
        allow_nipples = True
    else:
        allow_nipples = nipple_exposed

    # 3. WAIST_MIDRIFF (waist, midriff, abs)
    midriff_covered = bool(_MIDRIFF_COVER_RE.search(combined))
    midriff_revealed = bool(_MIDRIFF_REVEAL_RE.search(combined)) or is_sheer_or_torn
    allow_waist_midriff = (not midriff_covered) or midriff_revealed

    # 4. BUTTOCKS_HIPS (hips, pelvis, glutes contour)
    glutes_covered = bool(_GLUTES_HEAVY_COVER_RE.search(combined))
    glutes_revealed = bool(_GLUTES_REVEAL_RE.search(combined)) or is_sheer_or_torn
    allow_buttocks_hips = (not glutes_covered) or glutes_revealed

    # 5. INTIMATE_GENITALS (vulva / penis)
    has_lower = bool(_LOWER_GARMENT_RE.search(combined))
    intimate_exposed = bool(_INTIMATE_EXPOSED_RE.search(combined))
    allow_intimate = (not has_lower and not has_upper) or intimate_exposed

    allowed_map = {
        "BREASTS_CHEST": allow_breasts_chest,
        "NIPPLES_AREOLAS": allow_nipples,
        "WAIST_MIDRIFF": allow_waist_midriff,
        "BUTTOCKS_HIPS": allow_buttocks_hips,
        "INTIMATE_GENITALS": allow_intimate,
    }
    visible = tuple(z for z in ANATOMY_ZONES if allowed_map[z])
    blocked = tuple(z for z in ANATOMY_ZONES if not allowed_map[z])
    summary = (
        f"Visible: {', '.join(visible) or 'None (fully armored/covered)'} | "
        f"Blocked by outfit: {', '.join(blocked) or 'None'}"
    )
    return OutfitOcclusionReport(
        outfit_text=outfit_text,
        is_nude=False,
        allowed_zones=allowed_map,
        blocked_zones=blocked,
        visible_zones=visible,
        reason_summary=summary,
    )


def classify_anatomy_wildcard_zone(wildcard_key_or_text: str) -> str | None:
    """
    Maps an anatomy wildcard key or anatomical text fragment to one of the 5 occludable zones,
    or None if it is a non-occludable general trait (e.g. Age-Adult, Beard-FacialHair, Build-Musculature, Creature traits).
    """
    k = wildcard_key_or_text.lower()
    if any(w in k for w in ("nipple", "areola", "nipples-areolas")):
        return "NIPPLES_AREOLAS"
    if any(w in k for w in ("vulva", "pussies", "pussy", "penis", "intimate", "genital", "labia", "clitoris")):
        return "INTIMATE_GENITALS"
    if any(w in k for w in ("breasts-chest", "breast-shape", "jffbreast", "chest-torso", "breast", "cleavage", "bosom", "bustline", "decollet")):
        return "BREASTS_CHEST"
    if any(w in k for w in ("waist-midriff", "abs-core", "waist", "midriff", "navel", "abdomen", "abdominal", "washboard", "rectus", "v-line", "six-pack", "obliques", "hourglass")):
        return "WAIST_MIDRIFF"
    if any(w in k for w in ("buttocks", "glutes", "hips-pelvis", "hips", "pelvis", "buttock", "glutes-legs", "quadricep", "quads", "quad", "calves", "femoral", "thighs")):
        return "BUTTOCKS_HIPS"
    return None


def filter_anatomy_text_by_occlusion(
    anatomy_text: str,
    occlusion: OutfitOcclusionReport,
    gender: str = "Female",
) -> str:
    """
    Filters a resolved or multi-clause anatomy description against `occlusion.allowed_zones`
    and gender consistency so covered body parts are automatically removed.
    """
    if not anatomy_text or not anatomy_text.strip():
        return ""
    if occlusion.is_nude:
        return anatomy_text.strip()

    clauses = [c.strip() for c in re.split(r"[,;]\s*", anatomy_text) if c.strip()]
    kept: list[str] = []
    g_low = gender.lower()

    for clause in clauses:
        c_low = clause.lower()
        # Gender sanity check
        if g_low == "male" and any(w in c_low for w in ("vulva", "labia", "vagina", "breasts", "c-cup", "d-cup", "b-cup", "decolletage")):
            continue
        if g_low == "female" and any(w in c_low for w in ("penis", "testicles", "scrotum", "beard", "mustache")):
            continue

        zone = classify_anatomy_wildcard_zone(clause)
        if zone is not None and not occlusion.allowed_zones.get(zone, True):
            continue
        kept.append(clause)

    return ", ".join(kept)


# ---------------------------------------------------------------------------
# 3. Character-Outfit Linker (Themes `Character-Outfits.json` & Universes `source.json`)
# ---------------------------------------------------------------------------

def clean_outfit_prefix(value: str) -> str:
    """Strips leading 'wearing ' so the assembler can format 'wearing <outfit>' cleanly."""
    cleaned = re.sub(r"^(?:she\s+is\s+wearing|he\s+is\s+wearing|wearing|dressed\s+in)\s+", "", value.strip(), flags=re.I)
    return cleaned.strip().rstrip(".")


def get_theme_outfit_choices(catalog: dict[str, Any], character: dict[str, Any], mode: str) -> list[str]:
    """Ported from Wildcard-GPT `recipe_service.outfit_choices`: returns outfits for canonical, compatible, or free mode."""
    mode_low = mode.lower()
    if "canonical" in mode_low:
        values = character.get("canonical_outfits", [])
    elif "compatible" in mode_low:
        pools = catalog.get("outfit_pools", {})
        gender = str(character.get("gender", "Female")).capitalize()
        values = list(pools.get(f"{gender}-Outfit", [])) + list(pools.get("Mixed-Outfit", []))
        if not values:
            values = list(character.get("canonical_outfits", [])) or list(catalog.get("outfits", []))
    else:
        values = list(catalog.get("outfits", []))
    return list(dict.fromkeys(clean_outfit_prefix(v) for v in values if v and str(v).strip()))


def load_theme_bundle(theme_dir: str | Path) -> dict[str, Any]:
    """Loads a Theme folder (`Character-Outfits.json` + `.txt` facet files) into a structured dictionary."""
    t_path = Path(theme_dir)
    bundle: dict[str, Any] = {
        "kind": "theme",
        "name": t_path.name,
        "path": str(t_path),
        "characters": [],
        "outfit_pools": {},
        "outfits": [],
        "styles": [],
        "environments": [],
        "poses": [],
        "creatures": [],
        "vehicles": [],
    }
    if not t_path.is_dir():
        return bundle

    def _read_lines(fname: str) -> list[str]:
        fp = t_path / fname
        if fp.is_file():
            try:
                return [
                    ln.strip()
                    for ln in fp.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")
                ]
            except OSError:
                return []
        return []

    manifest = t_path / "Character-Outfits.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig", errors="replace"))
            if isinstance(data, dict):
                bundle["characters"] = data.get("characters", [])
                bundle["outfit_pools"] = data.get("outfit_pools", {})
                bundle["outfits"] = data.get("outfits", [])
        except Exception:
            pass

    if not bundle["outfits"]:
        bundle["outfits"] = _read_lines("Outfit.txt")
    if not bundle["outfit_pools"]:
        bundle["outfit_pools"] = {
            "Female-Outfit": _read_lines("Female-Outfit.txt"),
            "Male-Outfit": _read_lines("Male-Outfit.txt"),
            "Mixed-Outfit": _read_lines("Mixed-Outfit.txt"),
        }

    if not bundle["characters"]:
        for g_label, fname in (("Female", "Female.txt"), ("Male", "Male.txt"), ("Any", "Character.txt")):
            for idx, desc in enumerate(_read_lines(fname)):
                bundle["characters"].append({
                    "id": f"txt_{g_label}_{idx}",
                    "description": desc,
                    "gender": "Female" if g_label == "Female" else ("Male" if g_label == "Male" else "Female"),
                    "canonical_outfits": [],
                })

    bundle["styles"] = _read_lines("Style.txt")
    bundle["environments"] = _read_lines("Environment.txt")
    bundle["poses"] = _read_lines("PoseAction.txt")
    bundle["creatures"] = _read_lines("Creature.txt")
    bundle["vehicles"] = _read_lines("Vehicle.txt")
    return bundle


def load_universe_bundle(universe_dir: str | Path) -> dict[str, Any]:
    """Loads a Universe folder (`source.json` and/or `.txt` files across branches) into the same unified bundle format."""
    u_path = Path(universe_dir)
    bundle: dict[str, Any] = {
        "kind": "universe",
        "name": u_path.name,
        "path": str(u_path),
        "characters": [],
        "outfit_pools": {"Female-Outfit": [], "Male-Outfit": [], "Mixed-Outfit": []},
        "outfits": [],
        "styles": [],
        "environments": [],
        "poses": [],
        "creatures": [],
        "vehicles": [],
    }
    if not u_path.is_dir():
        return bundle

    def _read_lines(fp: Path) -> list[str]:
        if fp.is_file():
            try:
                return [
                    ln.strip()
                    for ln in fp.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")
                ]
            except OSError:
                return []
        return []

    source_json = u_path / "source.json"
    if source_json.is_file():
        try:
            data = json.loads(source_json.read_text(encoding="utf-8-sig", errors="replace"))
            branches = data.get("branches", []) if isinstance(data, dict) else []
            for br in branches:
                if not isinstance(br, dict):
                    continue
                st = str(br.get("style", "")).strip()
                if st and st not in bundle["styles"]:
                    bundle["styles"].append(st)
                for env in br.get("environments", []) or []:
                    if env and env not in bundle["environments"]:
                        bundle["environments"].append(env)
                for pos in br.get("poses", []) or []:
                    if pos and pos not in bundle["poses"]:
                        bundle["poses"].append(pos)
                for ch in br.get("characters", []) or []:
                    if not isinstance(ch, dict):
                        continue
                    # Skip minor characters if age is explicitly minor to keep adult anatomy safe
                    age = str(ch.get("age", "adult")).lower()
                    name = str(ch.get("name", "")).strip()
                    traits = str(ch.get("traits", "")).strip()
                    variant = str(ch.get("variant", "")).strip()
                    gender = str(ch.get("gender", "Female")).capitalize()
                    outfit = clean_outfit_prefix(str(ch.get("outfit", "")))
                    desc = f"{name} ({variant}), {traits}" if variant and traits else (f"{name}, {traits}" if traits else name)
                    if desc:
                        bundle["characters"].append({
                            "id": f"{u_path.name}:{name}:{variant}",
                            "name": name,
                            "description": desc,
                            "gender": gender,
                            "age": age,
                            "canonical_outfits": [outfit] if outfit else [],
                        })
                    if outfit:
                        if outfit not in bundle["outfits"]:
                            bundle["outfits"].append(outfit)
                        pool_key = f"{gender}-Outfit" if gender in ("Female", "Male") else "Mixed-Outfit"
                        if outfit not in bundle["outfit_pools"].setdefault(pool_key, []):
                            bundle["outfit_pools"][pool_key].append(outfit)
        except Exception:
            pass

    # Also scan direct or branch .txt files (e.g. Character-Female.txt, Outfit-Female.txt, Style.txt, Environment.txt)
    for root, _dirs, files in os.walk(u_path):
        for fname in sorted(files):
            if not fname.lower().endswith(".txt"):
                continue
            low = fname.lower()
            fp = Path(root) / fname
            if low in ("all.txt", "scene-canonical.txt"):
                continue
            lines = _read_lines(fp)
            if not lines:
                continue
            if "style" in low:
                for ln in lines:
                    if ln not in bundle["styles"]:
                        bundle["styles"].append(ln)
            elif "environment" in low:
                for ln in lines:
                    if ln not in bundle["environments"]:
                        bundle["environments"].append(ln)
            elif "pose" in low:
                for ln in lines:
                    if ln not in bundle["poses"]:
                        bundle["poses"].append(ln)
            elif "outfit-female" in low or "female-outfit" in low:
                for ln in lines:
                    c_ln = clean_outfit_prefix(ln)
                    if c_ln not in bundle["outfit_pools"]["Female-Outfit"]:
                        bundle["outfit_pools"]["Female-Outfit"].append(c_ln)
                    if c_ln not in bundle["outfits"]:
                        bundle["outfits"].append(c_ln)
            elif "outfit-male" in low or "male-outfit" in low:
                for ln in lines:
                    c_ln = clean_outfit_prefix(ln)
                    if c_ln not in bundle["outfit_pools"]["Male-Outfit"]:
                        bundle["outfit_pools"]["Male-Outfit"].append(c_ln)
                    if c_ln not in bundle["outfits"]:
                        bundle["outfits"].append(c_ln)
            elif "character-female" in low or low == "female.txt":
                if not bundle["characters"] or not any(c.get("gender") == "Female" for c in bundle["characters"]):
                    f_outfits = bundle["outfit_pools"]["Female-Outfit"]
                    for idx, ln in enumerate(lines):
                        c_out = [f_outfits[idx]] if idx < len(f_outfits) else []
                        bundle["characters"].append({
                            "id": f"{u_path.name}:F:{idx}",
                            "description": ln,
                            "gender": "Female",
                            "age": "adult",
                            "canonical_outfits": c_out,
                        })
            elif "character-male" in low or low == "male.txt":
                if not any(c.get("gender") == "Male" for c in bundle["characters"]):
                    m_outfits = bundle["outfit_pools"]["Male-Outfit"]
                    for idx, ln in enumerate(lines):
                        c_out = [m_outfits[idx]] if idx < len(m_outfits) else []
                        bundle["characters"].append({
                            "id": f"{u_path.name}:M:{idx}",
                            "description": ln,
                            "gender": "Male",
                            "age": "adult",
                            "canonical_outfits": c_out,
                        })

    return bundle


# ---------------------------------------------------------------------------
# 4. Duo / Multi-Character Interactions Bank (Ported from Wildcard-GPT combo_core.py)
# ---------------------------------------------------------------------------

DUO_INTERACTIONS_BANK: dict[str, list[str]] = {
    "Action / Combat (Back-to-Back)": [
        "fighting back to back against overwhelming odds in perfect sync",
        "unleashing a combined assault together with dynamic motion and energy",
        "leaping into action side by side in a high-stakes confrontation",
        "clashing in a dramatic mid-air duel with sparks and kinetic tension",
    ],
    "Heroic / Iconic Partners (Camaraderie)": [
        "sharing a confident side-by-side hero pose as iconic partners, warm camaraderie",
        "standing shoulder to shoulder with resolute expressions and commanding presence",
        "planning their next move together in a tense atmospheric moment",
    ],
    "Romantic / Sensual / Glamour": [
        "locked in a dramatic passionate embrace, intimate eye contact and soft tension",
        "sharing a playful glamorous boudoir pose with confident alluring smiles",
        "standing close together in an intimate sensual composition, subtle touch",
    ],
    "Human + Creature Companion": [
        "standing in calm mutual trust beside the majestic creature companion, an unbreakable bond",
        "unleashing a combined assault together, human courage and beast fury as one",
    ],
    "Human + Vehicle": [
        "posing proudly beside the iconic vehicle, rider and machine in perfect harmony",
        "racing at full throttle through dramatic scenery, dynamic speed and motion",
    ],
}


# ---------------------------------------------------------------------------
# 5. Deterministic Krea2 Compiler v2 (9-Block IR + Scoped / Balanced / Compact / Rich)
# ---------------------------------------------------------------------------

_FRAMING_KEYWORDS = {
    "shot", "angle", "view", "close-up", "closeup", "portrait", "wide-angle", "wide angle",
    "macro", "telephoto", "cinematic shot", "full body", "medium shot", "establishing shot",
    "dutch angle", "eye-level", "low angle", "high angle", "overhead", "panoramic",
    "depth of field", "bokeh", "composition", "centered", "upper body", "cowboy shot",
    "profile", "dynamic angle", "isometric", "front view", "side view", "back view",
}

_LIGHTING_KEYWORDS = {
    "light", "lighting", "lit", "illumination", "illuminated", "neon", "glow", "glowing",
    "shadow", "shadows", "dark", "bright", "sunlight", "moonlight", "golden hour", "blue hour",
    "palette", "radiant", "diffuse", "rim light", "volumetric", "god rays", "specular",
    "sunbeams", "sun rays", "haze", "reflections", "backlit", "highlights", "soft light", "hard light",
}

_STYLE_KEYWORDS = {
    "animation", "cel animation", "pixel art", "pixel-art", "comic", "stop-motion", "tokusatsu",
    "style", "aesthetic", "aesthetics", "digital art", "illustration", "painting", "photograph", "photo",
    "oil painting", "watercolor", "concept art", "anime", "manga", "render", "rendering", "octane render",
    "unreal engine", "cinematic", "photorealistic", "vintage", "retro", "hyperrealistic",
    "realism", "photorealism", "grain", "texture", "textures", "cel-shaded", "classic ova", "ova look", "gritty",
    "linework", "contours", "contour", "hatching", "cross-hatching", "crosshatching", "brushwork", "brushstrokes",
    "impasto", "gouache", "woodblock", "ukiyo-e", "sumi-e", "ink wash", "airbrush", "airbrushing", "halftone",
    "screentone", "chiaroscuro", "tenebrism", "cinematography", "anamorphic", "film stock", "glaze", "glazes",
    "glazing", "washes", "stippled", "draftsmanship", "draughtsmanship", "shading", "sprites", "low-polygon",
    "1-bit", "dithered", "monochrome", "pastel", "charcoal", "graphite", "acrylic", "fresco", "tempera",
    "lithograph", "chromolithograph", "engraving", "etching", "ligne claire", "bande dessinee", "seinen",
    "shojo", "shonen", "giallo", "technicolor", "kodachrome", "ektachrome", "bleach-bypass", "skip-bleach",
}

_PRIMARY_MEDIUM_ANCHORS = {
    "animation", "cel animation", "pixel art", "pixel-art", "stop-motion", "tokusatsu",
    "style", "aesthetic", "aesthetics", "digital art", "illustration", "painting",
    "oil painting", "watercolor", "concept art", "anime", "manga", "render", "rendering",
    "cinematography", "photography", "linework", "brushwork", "brushstrokes", "impasto",
    "gouache", "woodblock", "ukiyo-e", "sumi-e", "ink wash", "airbrush", "airbrushing",
    "cross-hatching", "crosshatching", "hatching", "draftsmanship", "draughtsmanship",
    "ligne claire", "bande dessinee", "1-bit", "dithered", "lithograph", "chromolithograph",
    "engraving", "etching", "fresco", "tempera", "woodcut", "linocut",
}

_CONSTRAINT_KEYWORDS = {
    "masterpiece", "best quality", "highly detailed", "sharp focus", "hyper-detailed",
    "8k", "4k", "uhd", "trending on artstation", "award winning",
}

_APPAREL_KEYWORDS = {
    "jacket", "armor", "robe", "robes", "dress", "suit", "coat", "cloak", "hood",
    "boots", "gloves", "hat", "helmet", "skirt", "pants", "shirt", "apron",
    "trench coat", "vest", "outfit", "costume", "garment", "kimono", "uniform", "wearing", "bikini", "lingerie",
}

_FEATURE_KEYWORDS = {
    "face", "hair", "eyes", "eye", "skin", "cheek", "cheeks", "cheekbones",
    "tattoos", "tattoo", "runes", "scars", "scar", "beard", "mustache", "wings",
    "horns", "tail", "body", "expression", "gaze", "breasts", "cleavage", "waist", "hips",
}

_ACTION_RE = re.compile(
    r"\b(walking|standing|sitting|flying|holding|looking|running|smiling|posing|fighting|casting|tending|kneeling|riding|leaping|swimming|reading|gazing|floating|smoking|wielding|carrying|repairing|fixing|crafting|building|dancing|working|operating|driving|piloting|embracing|kissing|arching|leaning)\b",
    re.I,
)
_ENV_PREP_RE = re.compile(r"\b(in|on|at|inside|outside|amidst|under|surrounded by)\b", re.I)
_ENV_NOUN_RE = re.compile(
    r"\b(background|city|forest|room|sky|mountain|street|backstreet|terrace|greenhouse|bar|blizzard|snowstorm|cathedral|ruins|landscape|temple|dungeon|desert|ocean|sea|indoor|outdoor|interior|exterior|foliage|trees|garden|hangar|station|ship|starship|castle|bridge|alley|laboratory|lab|geodesic|palace|courtyard|beach|boudoir|bedroom|studio)\b",
    re.I,
)


def estimate_qwen_tokens(text: str) -> int:
    """Estimates Qwen3VL-4B token count for Krea2."""
    if not text or not text.strip():
        return 0
    parts = re.findall(r"\b\w+\b|[^\s\w]", text)
    count = 0
    for part in parts:
        if len(part) <= 8:
            count += 1
        else:
            count += (len(part) + 5) // 6
    return count


def get_headroom_status(char_count: int) -> str:
    if char_count < 500:
        return "short"
    if char_count <= KREA2_LONG_PROMPT_CHARS:
        return "optimal"
    if char_count <= KREA2_VERY_LONG_PROMPT_CHARS:
        return "warning"
    return "critical"


@dataclass(frozen=True)
class CharacterSlotData:
    """Structured representation of a single character unit (Primary or Secondary)."""
    slot_role: str = "Primary"  # "Primary" or "Secondary"
    gender: str = "Female"  # "Female", "Male", "Creature"
    character_desc: str = ""
    anatomy_desc: str = ""
    outfit_desc: str = ""
    individual_action: str = ""
    outfit_link_mode: str = "Canonical (1:1 Strict)"
    occlusion_report: OutfitOcclusionReport | None = None
    theme_or_universe: str = ""

    def format_character_block(self, encapsulate_subparts: bool = False) -> str:
        """Formats the character + filtered anatomy + outfit for Krea2."""
        parts: list[str] = []
        if self.character_desc.strip():
            parts.append(self.character_desc.strip().rstrip(".,"))
        if self.anatomy_desc.strip():
            parts.append(self.anatomy_desc.strip().rstrip(".,"))
        if self.outfit_desc.strip():
            clean_o = clean_outfit_prefix(self.outfit_desc)
            if clean_o:
                if _NUDE_OUTFIT_RE.search(clean_o):
                    parts.append(clean_o)
                else:
                    parts.append(f"wearing {clean_o}")
        if not parts:
            return ""
        if encapsulate_subparts:
            return " ".join(f"({p})" for p in parts)
        return ", ".join(parts)


@dataclass(frozen=True)
class SceneStageData:
    """Structured representation of the scene stage (Style, Action/Interaction, Environment, Lighting, Camera)."""
    style_desc: str = ""
    action_desc: str = ""
    environment_desc: str = ""
    lighting_desc: str = ""
    camera_desc: str = ""
    theme_or_universe: str = ""


def _deduplicate_comma_clauses(text: str) -> str:
    """Removes duplicate comma-separated clauses case-insensitively while preserving order."""
    if not text:
        return ""
    clauses = [c.strip() for c in text.split(",") if c.strip()]
    seen: set[str] = set()
    kept: list[str] = []
    for cl in clauses:
        key = cl.lower().strip(". ")
        if key and key not in seen:
            seen.add(key)
            kept.append(cl)
    return ", ".join(kept)


def _compute_instance_anchor(char1: CharacterSlotData | None, char2: CharacterSlotData | None) -> str:
    """Computes the top-level subject count anchor outside parentheses for Qwen3VL-4B."""
    has_c1 = bool(char1 and char1.character_desc.strip())
    has_c2 = bool(char2 and char2.character_desc.strip())

    def _gender_noun(g: str) -> str:
        gl = g.lower()
        if gl == "female":
            return "1 woman"
        if gl == "male":
            return "1 man"
        if gl == "creature":
            return "1 creature"
        return "1 character"

    if has_c1 and has_c2:
        g1 = (char1.gender if char1 else "Female").lower()
        g2 = (char2.gender if char2 else "Male").lower()
        if g1 == "female" and g2 == "female":
            return "2 women"
        if g1 == "male" and g2 == "male":
            return "2 men"
        if (g1 == "female" and g2 == "male") or (g1 == "male" and g2 == "female"):
            return "2 characters, a woman and a man"
        return "2 characters"
    if has_c1:
        return _gender_noun(char1.gender if char1 else "Female")
    if has_c2:
        return _gender_noun(char2.gender if char2 else "Female")
    return ""


def assemble_krea2_structured_prompt(
    *,
    char1: CharacterSlotData | None,
    char2: CharacterSlotData | None = None,
    stage: SceneStageData | None = None,
    policy: str = "scoped (Parentheses Blocks + Count Anchor)",
    separate_parentheses_per_facet: bool = True,
    include_count_anchor: bool = True,
    custom_prefix: str = "",
    custom_suffix: str = "",
) -> tuple[str, dict[str, Any]]:
    """
    Assembles the final Krea2 (Qwen3VL-4B) prompt according to the user's exact specification
    and the selected Krea2CompilerV2 policy:

    Structure when `separate_parentheses_per_facet=True` (User's exact structure):
      Solo:
        [count_anchor], (style), (character_1), [(character_1_anatomy)], (wearing character_1_outfit),
        (character_1_action), (in environment), (lighting), (camera).
      Duo:
        [count_anchor], (style), (character_1), [(character_1_anatomy)], (wearing character_1_outfit),
        (character_2), [(character_2_anatomy)], (wearing character_2_outfit), (shared_action),
        (in environment), (lighting), (camera).

    Structure when `policy` is `balanced`, `compact`, or `rich`:
      Uses Krea2CompilerV2 linear semantic ordering with deduplication.
    """
    pol_low = policy.lower()
    is_scoped = "scoped" in pol_low
    is_compact = "compact" in pol_low
    is_rich = "rich" in pol_low

    has_c2 = bool(char2 and (char2.character_desc.strip() or char2.outfit_desc.strip()))

    style_str = (stage.style_desc.strip().rstrip(".") if stage and stage.style_desc else "")
    env_str = (stage.environment_desc.strip().rstrip(".") if stage and stage.environment_desc else "")
    light_str = (stage.lighting_desc.strip().rstrip(".") if stage and stage.lighting_desc else "")
    cam_str = (stage.camera_desc.strip().rstrip(".") if stage and stage.camera_desc else "")
    stage_action = (stage.action_desc.strip().rstrip(".") if stage and stage.action_desc else "")

    # Determine action string for Solo vs Duo
    if has_c2:
        # Duo mode: prioritize shared stage action, or combine individual actions
        if stage_action:
            action_str = stage_action
        else:
            acts = [
                a.strip().rstrip(".")
                for a in (
                    char1.individual_action if char1 else "",
                    char2.individual_action if char2 else "",
                )
                if a and a.strip()
            ]
            action_str = ", ".join(acts)
    else:
        # Solo mode: use char1's individual action if provided, else stage action
        c1_act = char1.individual_action.strip().rstrip(".") if char1 and char1.individual_action else ""
        action_str = c1_act or stage_action

    if is_compact and env_str:
        # Compact policy keeps only the primary environment clause to avoid diluting solo subjects
        env_str = env_str.split(",")[0].strip()

    if env_str and not re.match(r"^(?:in|inside|at|on|amidst|within|under|across|beneath|outside|above|along|through|around|beside)\b", env_str, flags=re.I):
        env_formatted = f"in {env_str}"
    else:
        env_formatted = env_str

    anchor = _compute_instance_anchor(char1, char2) if include_count_anchor else ""

    if is_scoped:
        chunks: list[str] = []
        if custom_prefix.strip():
            chunks.append(f"({custom_prefix.strip().rstrip('.')})")

        # 1. (style)
        if style_str:
            if not re.match(r"^render\s+style\s*:", style_str, flags=re.I):
                chunks.append(f"(Render Style: {style_str})")
            else:
                chunks.append(f"({style_str})")

        # 2. Character 1 blocks
        if char1 and (char1.character_desc.strip() or char1.outfit_desc.strip() or char1.anatomy_desc.strip()):
            if separate_parentheses_per_facet:
                if char1.character_desc.strip():
                    chunks.append(f"({char1.character_desc.strip().rstrip('.,')})")
                if char1.anatomy_desc.strip():
                    chunks.append(f"({char1.anatomy_desc.strip().rstrip('.,')})")
                if char1.outfit_desc.strip():
                    c_out1 = clean_outfit_prefix(char1.outfit_desc)
                    if c_out1:
                        prefix_w = "" if _NUDE_OUTFIT_RE.search(c_out1) else "wearing "
                        chunks.append(f"({prefix_w}{c_out1})")
            else:
                c1_block = char1.format_character_block(encapsulate_subparts=False)
                if c1_block:
                    chunks.append(f"({c1_block})")

        # 3. Solo Action OR Character 2 + Shared Action
        if not has_c2:
            if action_str:
                chunks.append(f"({action_str})")
        else:
            if char2:
                if separate_parentheses_per_facet:
                    if char2.character_desc.strip():
                        chunks.append(f"({char2.character_desc.strip().rstrip('.,')})")
                    if char2.anatomy_desc.strip():
                        chunks.append(f"({char2.anatomy_desc.strip().rstrip('.,')})")
                    if char2.outfit_desc.strip():
                        c_out2 = clean_outfit_prefix(char2.outfit_desc)
                        if c_out2:
                            prefix_w2 = "" if _NUDE_OUTFIT_RE.search(c_out2) else "wearing "
                            chunks.append(f"({prefix_w2}{c_out2})")
                else:
                    c2_block = char2.format_character_block(encapsulate_subparts=False)
                    if c2_block:
                        chunks.append(f"({c2_block})")
            if action_str:
                chunks.append(f"({action_str})")

        # 4. (environment)
        if env_formatted:
            chunks.append(f"({env_formatted})")

        # 5. (lighting)
        if light_str:
            chunks.append(f"({light_str})")

        # 6. (camera)
        if cam_str:
            chunks.append(f"({cam_str})")

        if custom_suffix.strip():
            chunks.append(f"({custom_suffix.strip().rstrip('.')})")

        body = " ".join(chunks)
        final_prompt = f"{anchor}, {body}" if (anchor and body) else (body or anchor)
    else:
        # Linear Krea2CompilerV2 ordering (balanced, compact, rich)
        linear_parts: list[str] = []
        if custom_prefix.strip():
            linear_parts.append(custom_prefix.strip().rstrip(".,"))
        if anchor:
            linear_parts.append(anchor)
        if char1:
            c1_b = char1.format_character_block(encapsulate_subparts=False)
            if c1_b:
                linear_parts.append(c1_b)
        if has_c2 and char2:
            c2_b = char2.format_character_block(encapsulate_subparts=False)
            if c2_b:
                linear_parts.append(f"accompanied by {c2_b}")
        if action_str:
            linear_parts.append(action_str)
        if cam_str:
            linear_parts.append(cam_str)
        if env_formatted:
            linear_parts.append(env_formatted)
        if style_str:
            linear_parts.append(style_str)
        if light_str:
            linear_parts.append(light_str)
        if is_rich:
            linear_parts.append("masterpiece, ultra-detailed, sharp focus, high dynamic range")
        if custom_suffix.strip():
            linear_parts.append(custom_suffix.strip().rstrip(".,"))

        final_prompt = _deduplicate_comma_clauses(", ".join(p for p in linear_parts if p))

    # Normalize spacing and terminating punctuation (match Krea2CompilerV2: append period if not ending in . ! ? ")
    final_prompt = _SPACE_RE.sub(" ", unicodedata.normalize("NFC", final_prompt)).strip()
    final_prompt = _BEFORE_PUNCTUATION_RE.sub(r"\1", final_prompt)
    final_prompt = _REPEATED_COMMA_RE.sub(",", final_prompt)
    if final_prompt and not final_prompt.endswith((".", "!", "?", '"')):
        final_prompt += "."

    tokens_est = estimate_qwen_tokens(final_prompt)
    stats = {
        "compiler_version": KREA2_COMPILER_VERSION,
        "policy": policy,
        "mode": "Duo / Interaction" if has_c2 else "Solo",
        "chars": len(final_prompt),
        "words": len(final_prompt.split()),
        "qwen3vl_tokens_estimated": tokens_est,
        "headroom_status": get_headroom_status(len(final_prompt)),
        "char_1_occlusion": char1.occlusion_report.reason_summary if (char1 and char1.occlusion_report) else "N/A",
        "char_2_occlusion": char2.occlusion_report.reason_summary if (char2 and char2.occlusion_report) else "N/A",
    }
    return final_prompt, stats
