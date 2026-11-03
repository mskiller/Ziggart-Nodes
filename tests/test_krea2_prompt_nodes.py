from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_nodes_krea2_test"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

embedded_model = load_module("embedded_model", "embedded_model.py")
nodes = load_module("nodes", "nodes.py")
detailer_nodes = load_module("detailer_nodes", "detailer_nodes.py")
addon_nodes = load_module("addon_nodes", "addon_nodes.py")
krea2_compiler = load_module("krea2_compiler", "krea2_compiler.py")
wildcard_nodes = load_module("wildcard_nodes", "wildcard_nodes.py")


class Krea2PromptDirectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wc_root = Path(self.temp_dir.name)

        # 1. Collection/Themes/01-TestTheme with Character-Outfits.json
        theme_dir = self.wc_root / "Collection" / "Themes" / "01-TestTheme"
        theme_dir.mkdir(parents=True, exist_ok=True)
        (theme_dir / "Style.txt").write_text("1980s Franco-Japanese cel animation masterpiece\n", encoding="utf-8")
        (theme_dir / "Environment.txt").write_text("Golden Temple of Mu surrounded by jungle ruins\n", encoding="utf-8")
        (theme_dir / "PoseAction.txt").write_text("holding the shining sun medallion aloft\n", encoding="utf-8")
        (theme_dir / "ALL.txt").write_text(
            "__Collection/Themes/01-TestTheme/Character__, wearing __Collection/Themes/01-TestTheme/Outfit__\n",
            encoding="utf-8",
        )
        (theme_dir / "Character-Outfits.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "theme": "01-TestTheme",
                    "characters": [
                        {
                            "id": "zia",
                            "description": "ZIA (Inca Heir), slender copper-brown girl with long raven hair",
                            "gender": "Female",
                            "canonical_outfits": [
                                "a white Inca tunic with blue geometric borders and turquoise sash"
                            ],
                        },
                        {
                            "id": "mendoza",
                            "description": "MENDOZA (Spanish Navigator), tall weathered man with wavy dark hair",
                            "gender": "Male",
                            "canonical_outfits": [
                                "a blue Spanish doublet beneath a dark travelling cape and high leather boots"
                            ],
                        },
                    ],
                    "outfits": [
                        "a white Inca tunic with blue geometric borders and turquoise sash",
                        "a blue Spanish doublet beneath a dark travelling cape and high leather boots",
                        "a sheer translucent silk ceremonial robe open over the chest",
                    ],
                    "outfit_pools": {
                        "Female-Outfit": [
                            "a white Inca tunic with blue geometric borders and turquoise sash",
                            "a sheer translucent silk ceremonial robe open over the chest",
                        ],
                        "Male-Outfit": [
                            "a blue Spanish doublet beneath a dark travelling cape and high leather boots"
                        ],
                        "Mixed-Outfit": [],
                    },
                }
            ),
            encoding="utf-8",
        )

        # 2. Collection/Universes/Aladdin with source.json
        univ_dir = self.wc_root / "Collection" / "Universes" / "Aladdin"
        univ_dir.mkdir(parents=True, exist_ok=True)
        (univ_dir / "source.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "franchise": "Aladdin",
                    "branches": [
                        {
                            "key": "Aladdin/LiveAction2019",
                            "style": "live-action fantasy musical, embroidered jewel-colored textiles",
                            "environments": ["Agrabah palace balcony with carved arches"],
                            "poses": ["dynamic adventure scene, expressive gesture"],
                            "characters": [
                                {
                                    "name": "JASMINE",
                                    "gender": "Female",
                                    "age": "adult",
                                    "traits": "long dark hair, warm brown skin and brown eyes",
                                    "outfit": "turquoise embroidered palace dress and gold jewelry",
                                    "variant": "2019 adaptation",
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        # 3. Collection/Anatomy (Female & Male)
        anat_f = self.wc_root / "Collection" / "Anatomy" / "Female"
        anat_f.mkdir(parents=True, exist_ok=True)
        (anat_f / "Breasts-Chest.txt").write_text("voluptuous rounded breasts, deep sculpted cleavage\n", encoding="utf-8")
        (anat_f / "Nipples-Areolas.txt").write_text("perky rose-pink nipples and taut areolas\n", encoding="utf-8")
        (anat_f / "Waist-Midriff.txt").write_text("toned hourglass waist, visible navel and sculpted midriff\n", encoding="utf-8")
        (anat_f / "Vulva-Intimate.txt").write_text("delicate anatomical vulva contour\n", encoding="utf-8")

        anat_m = self.wc_root / "Collection" / "Anatomy" / "Male"
        anat_m.mkdir(parents=True, exist_ok=True)
        (anat_m / "Build-Musculature.txt").write_text("broad muscular shoulders and athletic v-taper frame\n", encoding="utf-8")

        # 4. Collection/Lighting & Camera
        light_dir = self.wc_root / "Collection" / "Lighting"
        light_dir.mkdir(parents=True, exist_ok=True)
        (light_dir / "Cinematic-Dramatic.txt").write_text("warm golden hour volumetric rim lighting\n", encoding="utf-8")

        cam_dir = self.wc_root / "Collection" / "Camera"
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / "Framing.txt").write_text("cinematic medium full shot, 50mm lens, shallow depth of field\n", encoding="utf-8")

        # 5. ziggart, jff, Encours (including full-scene files that MUST be ignored)
        zig_style = self.wc_root / "ziggart" / "Style"
        zig_style.mkdir(parents=True, exist_ok=True)
        (zig_style / "Painting_Style_Kimi.txt").write_text("expressive impasto oil painting brushwork\n", encoding="utf-8")

        zig_outfit = self.wc_root / "ziggart" / "Outfits"
        zig_outfit.mkdir(parents=True, exist_ok=True)
        (zig_outfit / "woman-armors.txt").write_text("gilded celestial paladin full plate armor, heavy engraved bronze breastplate\n", encoding="utf-8")

        encours = self.wc_root / "Encours"
        encours.mkdir(parents=True, exist_ok=True)
        (encours / "Zig-KreaNew25-scene.txt").write_text("(render style: anime) A woman, full scene prompt that should be ignored\n", encoding="utf-8")
        (encours / "action-pose.txt").write_text("leaping dynamically through the air\n", encoding="utf-8")

        jff = self.wc_root / "jff"
        jff.mkdir(parents=True, exist_ok=True)
        (jff / "Anime-DaPrompt-scene.txt").write_text("Full anime scene prompt to ignore\n", encoding="utf-8")
        (jff / "vetements-200-krea.txt").write_text("very small black tape X-shaped nipple pasties covering only the nipples, black vinyl micro skirt\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_scene_prompts_are_excluded_from_pure_facets(self):
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            all_facet_tokens = [
                tok.lower()
                for items in wildcard_nodes.WILDCARD_INDEX.facets.values()
                for tok in items
            ]
            # Full scene files must never appear in component facets
            self.assertFalse(any("zig-kreanew25-scene" in t for t in all_facet_tokens))
            self.assertFalse(any("anime-daprompt-scene" in t for t in all_facet_tokens))
            self.assertFalse(any("01-testtheme/all" in t for t in all_facet_tokens))

            # Pure component files from ziggart/, Encours/, jff/ MUST appear in their respective facets
            self.assertTrue(any("painting_style_kimi" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["styles"])))
            self.assertTrue(any("woman-armors" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["outfits"])))
            self.assertTrue(any("vetements-200-krea" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["outfits"])))
            self.assertTrue(any("encours/action-pose" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["poses_actions"])))

    def test_automatic_outfit_to_anatomy_occlusion_analyzer(self):
        # Case 1: Heavy full plate armor blocks chest, nipples, midriff, glutes, intimate
        rep_armor = krea2_compiler.analyze_outfit_occlusion(
            "gilded celestial paladin full plate armor, heavy engraved bronze breastplate, steel-toed greaves"
        )
        self.assertFalse(rep_armor.allowed_zones["BREASTS_CHEST"])
        self.assertFalse(rep_armor.allowed_zones["NIPPLES_AREOLAS"])
        self.assertFalse(rep_armor.allowed_zones["WAIST_MIDRIFF"])
        self.assertFalse(rep_armor.allowed_zones["INTIMATE_GENITALS"])

        filtered = krea2_compiler.filter_anatomy_text_by_occlusion(
            "voluptuous rounded breasts, perky rose-pink nipples, toned hourglass waist, delicate vulva",
            rep_armor,
            gender="Female",
        )
        self.assertEqual(filtered, "")

        # Case 2: Pasties + micro skirt -> allows BREASTS_CHEST, WAIST_MIDRIFF, BUTTOCKS_HIPS, but blocks NIPPLES_AREOLAS & INTIMATE_GENITALS
        rep_pasties = krea2_compiler.analyze_outfit_occlusion(
            "very small black tape X-shaped nipple pasties covering only the nipples, black vinyl micro skirt"
        )
        self.assertTrue(rep_pasties.allowed_zones["BREASTS_CHEST"])
        self.assertFalse(rep_pasties.allowed_zones["NIPPLES_AREOLAS"])
        self.assertTrue(rep_pasties.allowed_zones["WAIST_MIDRIFF"])
        self.assertTrue(rep_pasties.allowed_zones["BUTTOCKS_HIPS"])
        self.assertFalse(rep_pasties.allowed_zones["INTIMATE_GENITALS"])

        filtered_pasties = krea2_compiler.filter_anatomy_text_by_occlusion(
            "voluptuous rounded breasts, perky rose-pink nipples, toned midriff",
            rep_pasties,
            gender="Female",
        )
        self.assertIn("voluptuous rounded breasts", filtered_pasties)
        self.assertIn("toned midriff", filtered_pasties)
        self.assertNotIn("nipples", filtered_pasties)

        # Case 3: Sheer translucent open-chest robe -> allows BREASTS_CHEST and NIPPLES_AREOLAS
        rep_sheer = krea2_compiler.analyze_outfit_occlusion(
            "a sheer translucent silk ceremonial robe open over the chest"
        )
        self.assertTrue(rep_sheer.allowed_zones["BREASTS_CHEST"])
        self.assertTrue(rep_sheer.allowed_zones["NIPPLES_AREOLAS"])

        # Case 4: Nude -> allows all 5 zones
        rep_nude = krea2_compiler.analyze_outfit_occlusion("nude, completely unclothed")
        self.assertTrue(rep_nude.is_nude)
        self.assertTrue(all(rep_nude.allowed_zones.values()))

    def test_modular_pipeline_solo_and_duo_structure(self):
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)

            ctx_node = wildcard_nodes.ZiggartThemeUniverseContext()
            ctx, active_name, def_style, def_env = ctx_node.build_context(
                source_mode="Theme (Collection/Themes)",
                theme_selection="01-TestTheme",
                seed=7,
            )
            self.assertEqual(active_name, "01-TestTheme")
            self.assertIn("Franco-Japanese", def_style)

            char_node = wildcard_nodes.ZiggartCharacterUnitBuilder()
            # Character 1: Female with Canonical outfit + Anatomy enabled (tunic covers midriff & nipples)
            c1_slot, c1_block, c1_debug = char_node.build_character(
                slot_role="Primary",
                character_source="From Context (Theme / Universe)",
                gender_filter="Female",
                outfit_link_mode="Canonical (1:1 Strict Character Outfit)",
                enable_anatomy=True,
                auto_filter_anatomy_by_outfit=True,
                anatomy_wildcard="__Collection/Anatomy/Female/Nipples-Areolas__",
                seed=7,
                context=ctx,
            )
            self.assertIn("ZIA", c1_slot.character_desc)
            self.assertIn("white Inca tunic", c1_slot.outfit_desc)
            # Nipples-Areolas must be automatically blocked by the opaque white Inca tunic!
            self.assertEqual(c1_slot.anatomy_desc, "")

            # Stage node
            stage_node = wildcard_nodes.ZiggartSceneStageAtmosphere()
            stage_data, st_s, act_s, env_s, light_s, cam_s = stage_node.build_stage(
                seed=7,
                context=ctx,
            )

            # Assembler node (Solo)
            assembler = wildcard_nodes.ZiggartKrea2PromptAssembler()
            solo_prompt, det1, det2, stats_json = assembler.assemble(
                char_1_slot=c1_slot,
                stage_data=stage_data,
                krea2_policy="scoped (Parentheses Blocks + Count Anchor)",
                separate_parentheses_per_facet=True,
                include_count_anchor=True,
            )
            self.assertTrue(solo_prompt.startswith("1 woman, (Render Style:"))
            self.assertIn("(ZIA (Inca Heir), slender copper-brown girl with long raven hair)", solo_prompt)
            self.assertIn("(wearing a white Inca tunic with blue geometric borders and turquoise sash)", solo_prompt)
            self.assertIn("(holding the shining sun medallion aloft)", solo_prompt)
            self.assertIn("(in Golden Temple of Mu surrounded by jungle ruins)", solo_prompt)
            self.assertIn("(warm golden hour volumetric rim lighting)", solo_prompt)
            self.assertIn("(cinematic medium full shot, 50mm lens, shallow depth of field).", solo_prompt)
            self.assertEqual(det2, "")

            # Character 2: Male with Canonical outfit -> Duo mode
            c2_slot, c2_block, _ = char_node.build_character(
                slot_role="Secondary",
                character_source="From Context (Theme / Universe)",
                gender_filter="Male",
                outfit_link_mode="Canonical (1:1 Strict Character Outfit)",
                enable_anatomy=True,
                auto_filter_anatomy_by_outfit=True,
                anatomy_wildcard="__Collection/Anatomy/Male/Build-Musculature__",
                seed=8,
                context=ctx,
            )
            self.assertIn("MENDOZA", c2_slot.character_desc)
            self.assertIn("broad muscular shoulders", c2_slot.anatomy_desc)

            duo_prompt, det1_duo, det2_duo, duo_stats_json = assembler.assemble(
                char_1_slot=c1_slot,
                char_2_slot=c2_slot,
                stage_data=stage_data,
                krea2_policy="scoped (Parentheses Blocks + Count Anchor)",
                separate_parentheses_per_facet=True,
                include_count_anchor=True,
            )
            self.assertTrue(duo_prompt.startswith("2 characters, a woman and a man, (Render Style:"))
            self.assertIn("(MENDOZA (Spanish Navigator), tall weathered man with wavy dark hair)", duo_prompt)
            self.assertIn("(broad muscular shoulders and athletic v-taper frame)", duo_prompt)
            self.assertIn("(wearing a blue Spanish doublet beneath a dark travelling cape and high leather boots)", duo_prompt)
            self.assertTrue(det2_duo != "")

    def test_master_prompt_director_universe_and_policies(self):
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            director = wildcard_nodes.ZiggartKrea2MasterPromptDirector()
            compiled, pop, c1_det, c2_det, stats_str = director.direct_prompt(
                source_mode="Universe (Collection/Universes)",
                universe_selection="Aladdin",
                cast_mode="Solo (1 Character)",
                krea2_policy="balanced (Krea2 v2 9-Block Linear Prose)",
                seed=12,
            )
            self.assertIn("JASMINE", compiled)
            self.assertIn("turquoise embroidered palace dress", compiled)
            self.assertIn("Agrabah palace balcony", compiled)
            stats = json.loads(stats_str)
            self.assertEqual(stats["active_theme_or_universe"], "Aladdin")
            self.assertGreater(stats["qwen3vl_tokens_estimated"], 10)


if __name__ == "__main__":
    unittest.main()
