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

        # 3. Collection/Anatomy (Head-to-Toe Female, Male, Creature)
        anat_f = self.wc_root / "Collection" / "Anatomy" / "Female"
        anat_f.mkdir(parents=True, exist_ok=True)
        (anat_f / "Age-Adult.txt").write_text("adult woman in her prime\n", encoding="utf-8")
        (anat_f / "Build-Silhouette.txt").write_text("graceful athletic feminine silhouette\n", encoding="utf-8")
        (anat_f / "Skin-Complexion.txt").write_text("warm golden-olive skin complexion\n", encoding="utf-8")
        (anat_f / "Skin-Texture-Finish.txt").write_text("smooth satin-like skin finish\n", encoding="utf-8")
        (anat_f / "Hair-Cut-Style.txt").write_text("sleek asymmetrical angled bob haircut\n", encoding="utf-8")
        (anat_f / "Hair-Color.txt").write_text("deep raven-black hair color\n", encoding="utf-8")
        (anat_f / "Hair-Texture.txt").write_text("silky straight hair strands\n", encoding="utf-8")
        (anat_f / "Eyes-Shape.txt").write_text("captivating almond-shaped eyes\n", encoding="utf-8")
        (anat_f / "Eyes-Iris-Color.txt").write_text("luminous emerald-green iris\n", encoding="utf-8")
        (anat_f / "Nose.txt").write_text("delicate straight nose bridge\n", encoding="utf-8")
        (anat_f / "Mouth-Lips.txt").write_text("full plush rose-tinted lips\n", encoding="utf-8")
        (anat_f / "Neck-Collarbones.txt").write_text("slender swan neck and defined collarbones\n", encoding="utf-8")
        (anat_f / "Arms-Shoulders.txt").write_text("toned sculpted shoulders and slender arms\n", encoding="utf-8")
        (anat_f / "Hands-Fingers.txt").write_text("graceful slender fingers\n", encoding="utf-8")
        (anat_f / "Breasts-Chest.txt").write_text("voluptuous rounded breasts, deep sculpted cleavage\n", encoding="utf-8")
        (anat_f / "Nipples-Areolas.txt").write_text("perky rose-pink nipples and taut areolas\n", encoding="utf-8")
        (anat_f / "Waist-Midriff.txt").write_text("toned hourglass waist, visible navel and sculpted midriff\n", encoding="utf-8")
        (anat_f / "Back-Spine.txt").write_text("graceful arched spine, deep natural spinal furrow\n", encoding="utf-8")
        (anat_f / "Hips-Pelvis.txt").write_text("flared feminine hips\n", encoding="utf-8")
        (anat_f / "Buttocks-Glutes.txt").write_text("sculpted rounded glutes\n", encoding="utf-8")
        (anat_f / "Vulva-Intimate.txt").write_text("delicate anatomical vulva contour\n", encoding="utf-8")
        (anat_f / "Legs-Thighs.txt").write_text("long shapely dancer legs and smooth thighs\n", encoding="utf-8")
        (anat_f / "Feet-Ankles.txt").write_text("slender delicate ankles and high foot arches\n", encoding="utf-8")

        anat_m = self.wc_root / "Collection" / "Anatomy" / "Male"
        anat_m.mkdir(parents=True, exist_ok=True)
        (anat_m / "Build-Musculature.txt").write_text("broad muscular shoulders and athletic v-taper frame\n", encoding="utf-8")
        (anat_m / "Beard-FacialHair.txt").write_text("designer heavy stubble along a chiseled jawline\n", encoding="utf-8")

        # 4. Collection/Lighting & Camera
        light_dir = self.wc_root / "Collection" / "Lighting"
        light_dir.mkdir(parents=True, exist_ok=True)
        (light_dir / "Cinematic-Dramatic.txt").write_text("warm golden hour volumetric rim lighting\n", encoding="utf-8")

        cam_dir = self.wc_root / "Collection" / "Camera"
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / "Framing.txt").write_text("cinematic medium full shot, 50mm lens, shallow depth of field\n", encoding="utf-8")

        # 5. ziggart (Style, Outfits, Env), jff, Encours (including full-scene files that MUST be ignored)
        zig_style = self.wc_root / "ziggart" / "Style"
        zig_style.mkdir(parents=True, exist_ok=True)
        (zig_style / "Painting_Style_Kimi.txt").write_text("expressive impasto oil painting brushwork\n", encoding="utf-8")

        zig_env = self.wc_root / "ziggart" / "Env"
        zig_env.mkdir(parents=True, exist_ok=True)
        (zig_env / "Dance-env.txt").write_text("full-length mirrored walls of a professional dance studio, polished wooden floor\n", encoding="utf-8")

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

    def test_full_scene_prompts_are_excluded_and_ziggart_env_is_indexed(self):
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

            # Pure component files from ziggart/ (including ziggart/Env/*), Encours/, jff/ MUST appear in their respective facets
            self.assertTrue(any("painting_style_kimi" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["styles"])))
            self.assertTrue(any("ziggart/env/dance-env" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["environments"])))
            self.assertTrue(any("woman-armors" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["outfits"])))
            self.assertTrue(any("vetements-200-krea" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["outfits"])))
            self.assertTrue(any("encours/action-pose" in t for t in (x.lower() for x in wildcard_nodes.WILDCARD_INDEX.facets["poses_actions"])))

    def test_automatic_outfit_to_anatomy_occlusion_analyzer_9_zones(self):
        # Case 1: Heavy full plate armor + greaves blocks all 9 body zones (chest, nipples, midriff, back, arms, glutes, intimate, legs, feet)
        rep_armor = krea2_compiler.analyze_outfit_occlusion(
            "gilded celestial paladin full plate armor, heavy engraved bronze breastplate, steel-toed greaves, high leather boots"
        )
        for z in (
            "BREASTS_CHEST",
            "NIPPLES_AREOLAS",
            "WAIST_MIDRIFF",
            "BACK_SPINE",
            "ARMS_SHOULDERS",
            "BUTTOCKS_HIPS",
            "INTIMATE_GENITALS",
            "LEGS_THIGHS",
            "FEET_ANKLES",
        ):
            self.assertFalse(rep_armor.allowed_zones[z], f"Expected {z} to be blocked by full plate armor")

        # Case 2: Backless halter evening gown with high slit & barefoot -> allows BACK_SPINE, ARMS_SHOULDERS, LEGS_THIGHS, FEET_ANKLES, blocks WAIST_MIDRIFF & INTIMATE_GENITALS
        rep_backless = krea2_compiler.analyze_outfit_occlusion(
            "backless halter evening gown with a dramatic high slit and plunging cleavage, barefoot"
        )
        self.assertTrue(rep_backless.allowed_zones["BACK_SPINE"])
        self.assertTrue(rep_backless.allowed_zones["ARMS_SHOULDERS"])
        self.assertTrue(rep_backless.allowed_zones["BREASTS_CHEST"])
        self.assertTrue(rep_backless.allowed_zones["LEGS_THIGHS"])
        self.assertTrue(rep_backless.allowed_zones["FEET_ANKLES"])
        self.assertFalse(rep_backless.allowed_zones["NIPPLES_AREOLAS"])
        self.assertFalse(rep_backless.allowed_zones["INTIMATE_GENITALS"])

        # Case 3: Pasties + micro skirt -> allows BREASTS_CHEST, WAIST_MIDRIFF, BUTTOCKS_HIPS, LEGS_THIGHS, but blocks NIPPLES_AREOLAS & INTIMATE_GENITALS
        rep_pasties = krea2_compiler.analyze_outfit_occlusion(
            "very small black tape X-shaped nipple pasties covering only the nipples, black vinyl micro skirt"
        )
        self.assertTrue(rep_pasties.allowed_zones["BREASTS_CHEST"])
        self.assertFalse(rep_pasties.allowed_zones["NIPPLES_AREOLAS"])
        self.assertTrue(rep_pasties.allowed_zones["WAIST_MIDRIFF"])
        self.assertTrue(rep_pasties.allowed_zones["BUTTOCKS_HIPS"])
        self.assertTrue(rep_pasties.allowed_zones["LEGS_THIGHS"])
        self.assertFalse(rep_pasties.allowed_zones["INTIMATE_GENITALS"])

        # Case 4: Nude -> allows all 9 zones
        rep_nude = krea2_compiler.analyze_outfit_occlusion("nude, completely unclothed")
        self.assertTrue(rep_nude.is_nude)
        self.assertEqual(len(rep_nude.allowed_zones), 9)
        self.assertTrue(all(rep_nude.allowed_zones.values()))

    def test_head_to_toe_anatomy_character_creator_views_and_none_sections(self):
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            creator = wildcard_nodes.ZiggartAnatomyCharacterCreator()

            # 1. Back View / Seen From Behind: must automatically omit Eyes, Nose, Mouth, Breasts-Chest and include Back-Spine & Buttocks-Glutes
            slot_back, full_back, head_back, body_back, json_back = creator.create_anatomy_character(
                slot_role="Primary",
                gender="Female",
                view_visibility_preset="Back View / Seen From Behind (No Face / No Front Chest)",
                seed=11,
            )
            breakdown_back = json.loads(json_back)
            self.assertNotIn("eyes_shape", breakdown_back["active_sections"])
            self.assertNotIn("eyes_iris_color", breakdown_back["active_sections"])
            self.assertNotIn("nose", breakdown_back["active_sections"])
            self.assertNotIn("mouth_lips", breakdown_back["active_sections"])
            self.assertNotIn("chest_breasts", breakdown_back["active_sections"])
            self.assertIn("back_spine", breakdown_back["active_sections"])
            self.assertIn("buttocks_glutes", breakdown_back["active_sections"])
            self.assertIn("graceful arched spine", full_back)
            self.assertIn("sculpted rounded glutes", full_back)
            self.assertNotIn("emerald-green iris", full_back)

            # 2. Full Body Front View with explicit [None] on feet_ankles and hands_fingers + Heavy Armor Outfit Occlusion
            slot_front, full_front, head_front, body_front, json_front = creator.create_anatomy_character(
                slot_role="Primary",
                gender="Female",
                view_visibility_preset="Full Body — Front / 3-Quarter View",
                feet_ankles="[None]",
                hands_fingers="[None]",
                outfit_mode="Free (Outfit Wildcard)",
                outfit_wildcard="__ziggart/Outfits/woman-armors__",
                auto_filter_anatomy_by_outfit=True,
                seed=11,
            )
            breakdown_front = json.loads(json_front)
            # Head and face sections remain active even with body armor
            self.assertIn("eyes_shape", breakdown_front["active_sections"])
            self.assertIn("eyes_iris_color", breakdown_front["active_sections"])
            self.assertIn("emerald-green iris", head_front)
            # Explicitly disabled sections are in none_or_disabled_sections
            self.assertIn("feet_ankles", breakdown_front["none_or_disabled_sections"])
            self.assertIn("hands_fingers", breakdown_front["none_or_disabled_sections"])
            # Body zones covered by full plate armor are automatically in occluded_by_outfit_sections
            self.assertIn("chest_breasts", breakdown_front["occluded_by_outfit_sections"])
            self.assertIn("waist_midriff", breakdown_front["occluded_by_outfit_sections"])
            self.assertIn("legs_thighs", breakdown_front["occluded_by_outfit_sections"])

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

            # Stage node using ziggart/Env/Dance-env
            stage_node = wildcard_nodes.ZiggartSceneStageAtmosphere()
            stage_data, st_s, act_s, env_s, light_s, cam_s = stage_node.build_stage(
                environment_wildcard="__ziggart/Env/Dance-env__",
                seed=7,
                context=ctx,
            )
            self.assertIn("mirrored walls of a professional dance studio", env_s)

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
            self.assertIn("(in full-length mirrored walls of a professional dance studio, polished wooden floor)", solo_prompt)
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

    def test_trio_ensemble_director_spatial_layout_and_regional_conditioning(self):
        import torch

        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            c1 = krea2_compiler.CharacterSlotData(
                slot_role="Primary",
                gender="Female",
                character_desc="ZIA, Inca girl with raven hair",
                outfit_desc="white Inca tunic",
            )
            c2 = krea2_compiler.CharacterSlotData(
                slot_role="Secondary",
                gender="Male",
                character_desc="MENDOZA, tall Spanish navigator",
                outfit_desc="blue Spanish doublet and cape",
            )
            c3 = krea2_compiler.CharacterSlotData(
                slot_role="Tertiary",
                gender="Female",
                character_desc="JASMINE, Agrabah princess",
                outfit_desc="turquoise silk dress",
            )
            stage = krea2_compiler.SceneStageData(
                style_desc="1980s cel animation",
                action_desc="",
                environment_desc="Golden Temple of Mu",
                lighting_desc="dramatic rim light",
                camera_desc="wide cinematic shot",
            )

            trio_node = wildcard_nodes.ZiggartKrea2TrioEnsembleDirector()
            # Verify socket type compatibility with ZiggartSceneStageAtmosphere (ZIGGART_STAGE) and ZiggartThemeUniverseContext (ZIGGART_CONTEXT)
            trio_inputs = trio_node.INPUT_TYPES()
            self.assertEqual(trio_inputs["optional"]["stage_data"][0], "ZIGGART_STAGE")
            self.assertEqual(trio_inputs["optional"]["context"][0], "ZIGGART_CONTEXT")

            (
                compiled,
                base_scene,
                c1_det,
                c2_det,
                c3_det,
                spatial_layout,
                stats_json,
            ) = trio_node.direct_ensemble(
                ensemble_layout_preset="Left / Center / Right (Trio)",
                interaction_mode="Coordinated Team Action",
                krea2_policy="scoped (Parentheses Blocks + Count Anchor)",
                canvas_width=512,
                canvas_height=512,
                seed=7,
                char_1_slot=c1,
                char_2_slot=c2,
                char_3_slot=c3,
                stage_data=stage,
                companion_or_prop_text="golden condor automaton",
            )

            self.assertTrue(compiled.startswith("3 characters, 2 women and 1 man"))
            self.assertIn("ZIA", compiled)
            self.assertIn("MENDOZA", compiled)
            self.assertIn("JASMINE", compiled)
            self.assertIn("golden condor automaton", compiled)
            self.assertEqual(len(spatial_layout["regions"]), 3)
            self.assertIn("base_scene_prompt", spatial_layout)
            self.assertIn("char_prompts", spatial_layout)
            stats = json.loads(stats_json)
            self.assertEqual(stats["mode"], "Trio / Ensemble")

            # Test Trio Ensemble Director with ONLY `context` connected (auto-resolving distinct characters & stage)
            ctx_node = wildcard_nodes.ZiggartThemeUniverseContext()
            ctx, _, _, _ = ctx_node.build_context(
                source_mode="Theme (Collection/Themes)",
                theme_selection="01-TestTheme",
                seed=7,
            )
            auto_compiled, auto_base, auto_c1, auto_c2, auto_c3, auto_layout, auto_stats_json = trio_node.direct_ensemble(
                ensemble_layout_preset="Left / Center / Right (Trio)",
                interaction_mode="Combat Clash / Duel",
                seed=7,
                context=ctx,
            )
            self.assertTrue(auto_c1 != "")
            self.assertTrue(auto_c2 != "")
            self.assertTrue(auto_c3 != "")
            self.assertIn("Franco-Japanese", auto_compiled)

            # Test ZiggartSpatialLayoutDesigner custom boxes AND preset selection
            designer = wildcard_nodes.ZiggartSpatialLayoutDesigner()
            custom_layout, bboxes, preview_img, layout_json = designer.design_layout(
                canvas_width=512,
                canvas_height=512,
                active_regions=3,
                r1_label="Left Hero",
                r1_x1=0.05,
                r1_y1=0.10,
                r1_x2=0.45,
                r1_y2=0.90,
                r2_label="Center Hero",
                r2_x1=0.30,
                r2_y1=0.05,
                r2_x2=0.70,
                r2_y2=0.95,
                r3_label="Right Hero",
                r3_x1=0.55,
                r3_y1=0.10,
                r3_x2=0.95,
                r3_y2=0.90,
            )
            self.assertEqual(len(bboxes), 3)
            self.assertEqual(preview_img.shape, (1, 512, 512, 3))

            preset_layout, preset_bboxes, _, _ = designer.design_layout(
                canvas_width=512,
                canvas_height=512,
                layout_preset="Center Hero / Flanking Duo",
            )
            self.assertEqual(preset_layout["preset_name"], "Center Hero / Flanking Duo")
            self.assertEqual(len(preset_bboxes), 3)

            # Test ZiggartRegionalConditioningComposer WITH connected spatial_layout, Partition-of-Unity masks,
            # Scene Context Harmonization, and Timestep Range Unification
            encoded_prompts_log: list[str] = []
            mock_clip = mock.MagicMock()
            mock_clip.tokenize.side_effect = lambda text: (encoded_prompts_log.append(text) or [text])
            mock_clip.encode_from_tokens.side_effect = lambda toks, return_pooled=True: (
                torch.ones((1, 8, 64)),
                torch.ones((1, 64)),
            )

            reg_node = wildcard_nodes.ZiggartRegionalConditioningComposer()
            cond_with_layout, mask_prev1, reg_boxes1, summary1_json = reg_node.compose_regional_conditioning(
                clip=mock_clip,
                base_scene_prompt=base_scene,
                char_1_prompt=c1_det,
                char_2_prompt=c2_det,
                char_3_prompt=c3_det,
                spatial_layout=custom_layout,
                canvas_width=512,
                canvas_height=512,
                mask_mode="gaussian_soft_partition",
                normalize_overlap=True,
                context_harmonization="inject_base_style_and_lighting",
                regional_start_percent=0.0,
                regional_end_percent=1.0,
            )
            s1 = json.loads(summary1_json)
            self.assertEqual(s1["layout_source"], "spatial_layout_input")
            self.assertEqual(s1["active_regions_encoded"], 3)
            # Partition-of-Unity ensures overlapping regional masks never sum > 1.0
            self.assertLessEqual(s1["max_regional_mask_sum"], 1.0001)
            self.assertEqual(len(cond_with_layout), 4)  # 1 complementary bg + 3 regions
            self.assertIn("mask", cond_with_layout[1][1])
            # Scene Context Harmonization injects base scene style/lighting into regional character prompts
            self.assertTrue(any("ZIA" in p and "1980s cel animation" in p for p in encoded_prompts_log))

            # Test Timestep Range Scheduling (regional_end_percent=0.70 adds a 5th global unification pass for 0.70 -> 1.00)
            cond_scheduled, _, _, summary_sched_json = reg_node.compose_regional_conditioning(
                clip=mock_clip,
                spatial_layout=spatial_layout,  # Auto-fills base_scene_prompt, char_1..3_prompt, and full_scene_prompt!
                regional_start_percent=0.0,
                regional_end_percent=0.70,
            )
            s_sched = json.loads(summary_sched_json)
            self.assertTrue(s_sched["late_step_unification_pass"])
            self.assertEqual(len(cond_scheduled), 5)  # 1 bg (0..0.7) + 3 regions (0..0.7) + 1 global unification (0.7..1.0)
            self.assertAlmostEqual(cond_scheduled[-1][1]["start_percent"], 0.70, places=2)
            self.assertAlmostEqual(cond_scheduled[-1][1]["end_percent"], 1.00, places=2)

            # Without spatial_layout connected -> uses layout_preset
            cond_preset, mask_prev2, reg_boxes2, summary2_json = reg_node.compose_regional_conditioning(
                clip=mock_clip,
                base_scene_prompt=base_scene,
                char_1_prompt=c1_det,
                char_2_prompt=c2_det,
                layout_preset="Left / Right Split (Duo)",
                canvas_width=512,
                canvas_height=512,
            )
            s2 = json.loads(summary2_json)
            self.assertEqual(s2["layout_source"], "layout_preset")
            self.assertEqual(s2["active_regions_encoded"], 2)

    def test_token_inspector_pre_generate_and_yaml_bom_tabs(self):
        inspector = wildcard_nodes.ZiggartPromptTokenInspectorAndCompressor()
        noisy_prompt = (
            "[CHAR 1] Zia in white tunic, very extremely detailed, sharp focus, "
            "sharp focus, bright daylight, pitch black night, really super ornate gold border"
        )
        opt_prompt, tok_count, chunks, rep_json = inspector.inspect_and_compress(
            prompt_text=noisy_prompt,
            target_token_budget=25,
            compression_mode="smart_compact",
            preserve_character_locks=True,
        )
        rep = json.loads(rep_json)
        self.assertIn("[CHAR 1]", opt_prompt)
        self.assertGreaterEqual(rep["duplicates_removed_count"], 1)
        self.assertGreaterEqual(len(rep["conflicts_detected"]), 1)

        # Test Pre-Generate helper (_resolve_node_preview_prompt) & reproduce mode
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            preview = wildcard_nodes._resolve_node_preview_prompt(
                "ZiggartWildcardProcessor",
                {"wildcard_text": "{crimson|azure} cape"},
                seed_val=42,
            )
            self.assertIn(preview, ("crimson cape", "azure cape"))

            proc = wildcard_nodes.ZiggartWildcardProcessor()
            out_repro, pop_repro = proc.process_text(
                wildcard_text="{crimson|azure} cape",
                populated_text="locked pre-generated prompt",
                mode="reproduce",
                seed=999,
            )
            self.assertEqual(out_repro, "locked pre-generated prompt")

        # Test YAML byte-offset indexer with UTF-8 BOM and tabs
        bom_yaml = self.wc_root / "bom_tab.yaml"
        bom_yaml.write_bytes(b"\xef\xbb\xbfroot_group:\n\tsub_pool:\n\t\t- alpha item\n\t\t- beta item\n")
        offsets = wildcard_nodes.index_yaml_byte_offsets(str(bom_yaml))
        self.assertIn("root_group/sub_pool", offsets)


if __name__ == "__main__":
    unittest.main()
