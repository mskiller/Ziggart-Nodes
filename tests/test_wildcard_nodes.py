from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_nodes_wildcard_test"


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
wildcard_nodes = load_module("wildcard_nodes", "wildcard_nodes.py")


class WildcardEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.wc_root = Path(self.temp_dir.name)

        # Create 3-level nested Ziggart.yaml
        yaml_content = (
            "Ziggart:\n"
            "  scenes:\n"
            "    all:\n"
            '      - "Scene of __Collection/Themes/01-TestTheme/Character__ wearing __Collection/Themes/01-TestTheme/Outfit__"\n'
            "    female_all:\n"
            '      - "Female portrait of __Collection/Characters/Female/Heroines__"\n'
        )
        (self.wc_root / "Ziggart.yaml").write_text(yaml_content, encoding="utf-8")

        # Create root .txt file
        (self.wc_root / "style-full-v2.txt").write_text(
            "# comment line\ncinematic anime cel shading\nultra detailed oil painting\n",
            encoding="utf-8",
        )

        # Create Collection pillar folders and Theme folder with Character-Outfits.json
        char_dir = self.wc_root / "Collection" / "Characters" / "Female"
        char_dir.mkdir(parents=True, exist_ok=True)
        (char_dir / "Heroines.txt").write_text("Zia of the Incas\n", encoding="utf-8")

        outfit_dir = self.wc_root / "Collection" / "Outfits" / "Female"
        outfit_dir.mkdir(parents=True, exist_ok=True)
        (outfit_dir / "Armor.txt").write_text("golden Valkyrie plate\n", encoding="utf-8")

        theme_dir = self.wc_root / "Collection" / "Themes" / "01-TestTheme"
        theme_dir.mkdir(parents=True, exist_ok=True)
        (theme_dir / "Character.txt").write_text("Esteban the Sun Child\n", encoding="utf-8")
        (theme_dir / "Outfit.txt").write_text("saffron tunic and sun medallion\n", encoding="utf-8")
        (theme_dir / "PoseAction.txt").write_text("holding a golden telescope\n", encoding="utf-8")
        (theme_dir / "Environment.txt").write_text("ancient golden temple of Mu\n", encoding="utf-8")
        (theme_dir / "Style.txt").write_text("1980s Franco-Japanese anime masterpiece\n", encoding="utf-8")
        (theme_dir / "ALL.txt").write_text(
            "__Collection/Themes/01-TestTheme/Character__, __Collection/Themes/01-TestTheme/Outfit__\n",
            encoding="utf-8",
        )
        (theme_dir / "Character-Outfits.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "theme": "01-TestTheme",
                    "characters": [
                        {
                            "id": "esteban",
                            "description": "ESTEBAN (Child of the Sun)",
                            "gender": "Male",
                            "canonical_outfits": ["a saffron-yellow tunic with golden sun medallion"],
                        },
                        {
                            "id": "zia",
                            "description": "ZIA (Inca Heir)",
                            "gender": "Female",
                            "canonical_outfits": ["a white Inca tunic with turquoise sash"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_three_level_yaml_indexing_and_recursive_resolution(self):
        offsets = wildcard_nodes.index_yaml_byte_offsets(str(self.wc_root / "Ziggart.yaml"))
        self.assertIn("ziggart/scenes/all", offsets)
        self.assertIn("ziggart/scenes/female_all", offsets)

        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            resolved = wildcard_nodes.process_wildcards("__Ziggart/scenes/all__", seed=42)
            self.assertEqual(
                resolved,
                "Scene of Esteban the Sun Child wearing saffron tunic and sun medallion",
            )

    def test_choices_quantifiers_and_multi_select_syntax(self):
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            res = wildcard_nodes.process_wildcards(
                "# ignored comment\n{100.0::golden|0.0::silver} armor, {2$$ and $$red|blue}",
                seed=7,
            )
            self.assertIn("golden armor", res)
            self.assertTrue("red and blue" in res or "blue and red" in res)

    def test_way_a_all_in_one_and_way_c_scene_builder(self):
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)

            all_in_one = wildcard_nodes.ZiggartWildcardProcessorAllInOne()
            proc, pop, recipe = all_in_one.process_all_in_one(
                wildcard_text="masterpiece",
                populated_text="",
                mode="populate",
                seed=12,
                combine_selected_pillars=True,
                select_macro_yaml_theme="__ziggart/scenes/female_all__",
                select_outfit_pose="__Collection/Outfits/Female/Armor__",
                select_mixed_other="__style-full-v2__",
            )
            self.assertIn("Zia of the Incas", proc)
            self.assertIn("golden Valkyrie plate", proc)
            self.assertIn("__ziggart/scenes/female_all__", recipe)

            builder = wildcard_nodes.ZiggartCollectionSceneBuilder()
            built_prompt, built_template = builder.build_scene(
                custom_prefix="epic shot",
                select_character_anatomy="__Collection/Characters/Female/Heroines__",
                select_outfit_pose="__Collection/Outfits/Female/Armor__",
                custom_suffix="8k",
                seed=12,
            )
            self.assertEqual(
                built_prompt,
                "epic shot, Zia of the Incas, golden Valkyrie plate, 8k",
            )

    def test_canonical_theme_resolver_duo_locks_authentic_outfits(self):
        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            resolver = wildcard_nodes.ZiggartCanonicalThemeResolver()
            scene, c1, c2, c3, meta_json = resolver.resolve_theme(
                theme_selection="01-TestTheme",
                cast_mode="Duo (Female + Male)",
                include_pose_action=True,
                include_environment=True,
                include_style=True,
                seed=99,
            )
            self.assertIn("ZIA (Inca Heir), wearing a white Inca tunic with turquoise sash", c1)
            self.assertIn("ESTEBAN (Child of the Sun), wearing a saffron-yellow tunic", c2)
            self.assertEqual(c3, "")
            self.assertIn("ancient golden temple of Mu", scene)
            meta = json.loads(meta_json)
            self.assertEqual(meta["theme"], "01-TestTheme")

    @mock.patch.object(wildcard_nodes, "_resolve_gguf", return_value="dummy.gguf")
    def test_wildcard_vlm_enhancer(self, mock_resolve):
        class FakeMgr:
            def load(self, config, unload_comfy_models=True):
                pass

            def chat(self, messages, **kwargs):
                return "Harmonized natural prose scene of Zia of the Incas in golden lighting."

            def unload(self):
                pass

        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ), mock.patch.object(wildcard_nodes, "MODEL_MANAGER", FakeMgr()):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)
            enhancer = wildcard_nodes.ZiggartWildcardVLMEnhancer()
            harmonized, raw = enhancer.harmonize(
                wildcard_or_populated_text="__ziggart/scenes/female_all__",
                model="model.gguf",
                mmproj="mmproj.gguf",
                seed=1,
            )
            self.assertEqual(raw, "Female portrait of Zia of the Incas")
            self.assertIn("Harmonized natural prose", harmonized)

    def test_recipe_saver_writes_png_metadata_and_json_sidecar(self):
        saver = wildcard_nodes.ZiggartWildcardRecipeSaver()
        img = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.8
        base_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        out_dir = self.wc_root / "saved_outputs"
        grid, summary, paths_json = saver.save_recipe(
            image=img,
            filename_prefix="TestRun/Img",
            wildcard_template="__Ziggart/scenes/all__",
            populated_text="Esteban in golden temple",
            seed=777,
            save_json_sidecar=True,
            build_comparison_grid=True,
            base_image=base_img,
            detailer_or_vlm_notes="eyes refined at 768px",
            output_dir_override=str(out_dir),
        )
        # Side-by-side grid doubles width: 64 -> 128
        self.assertEqual(grid.shape, (1, 64, 128, 3))
        saved_paths = json.loads(paths_json)
        self.assertEqual(len(saved_paths), 2)
        png_path = Path(saved_paths[0])
        self.assertTrue(png_path.is_file())
        with Image.open(png_path) as loaded_png:
            self.assertEqual(loaded_png.text.get("ziggart_wildcard_template"), "__Ziggart/scenes/all__")
            self.assertEqual(loaded_png.text.get("ziggart_seed"), "777")

    def test_filesystem_globs_multi_root_yaml_and_jff_folder(self):
        # 1. Create unindexed deep formats-wild glob structure
        luis_dir = self.wc_root / "Collection" / "formats-wild" / "Formats" / "luisroyo" / "Templates"
        luis_dir.mkdir(parents=True, exist_ok=True)
        (luis_dir / "Female-action-environment.txt").write_text(
            "Dark gothic painting by Luis Royo, __Collection/Characters/Female/Heroines__\n",
            encoding="utf-8",
        )
        (luis_dir / "Female-classic-style.txt").write_text(
            "Brooding oil painting by Luis Royo, __Collection/Characters/Female/Heroines__\n",
            encoding="utf-8",
        )

        # 2. Create additional root YAML (like krea2-grok.yaml) and jff/ folder
        krea_yaml = self.wc_root / "krea2-grok.yaml"
        krea_yaml.write_text(
            'krea2-grok:\n  wallpaper:\n    - "Photorealistic shot wearing ___jff/vetements-grok__ and { __Jff/lingerie outfits | silk dress }"\n',
            encoding="utf-8",
        )
        jff_dir = self.wc_root / "jff"
        (jff_dir / "animals").mkdir(parents=True, exist_ok=True)
        (jff_dir / "vetements-grok.txt").write_text("silver latex monokini\n", encoding="utf-8")
        (jff_dir / "lingerie_outfits.txt").write_text("black lace corset\n", encoding="utf-8")
        (jff_dir / "animals" / "cats.txt").write_text("persian cat\n", encoding="utf-8")

        with mock.patch.object(
            wildcard_nodes, "discover_default_wildcard_dirs", return_value=[str(self.wc_root)]
        ):
            wildcard_nodes.WILDCARD_INDEX.ensure_loaded(force_refresh=True)

            # Test `*` filesystem glob on unindexed Collection/formats-wild
            glob_out = wildcard_nodes.process_wildcards(
                "__Collection/formats-wild/Formats/luisroyo/Templates/Female-*__", seed=1
            )
            self.assertIn("Luis Royo", glob_out)
            self.assertIn("Zia of the Incas", glob_out)
            self.assertNotIn("__", glob_out)

            # Test krea2-grok.yaml with 3-underscore token and unclosed choice wildcard
            krea_out = wildcard_nodes.process_wildcards("__krea2-grok/wallpaper__", seed=2)
            self.assertIn("silver latex monokini", krea_out)
            self.assertNotIn("__", krea_out)

            # Test jff/ relative glob `__animals/*__`
            jff_glob_out = wildcard_nodes.process_wildcards("__animals/*__", seed=1)
            self.assertEqual(jff_glob_out, "persian cat")


class MultiSubjectDetailerTests(unittest.TestCase):
    @mock.patch.object(addon_nodes, "_resolve_gguf", return_value="dummy.gguf")
    @mock.patch.object(addon_nodes, "detect_objects_with_vlm")
    def test_multi_subject_detailer_grounds_and_refines_two_characters(self, mock_detect, mock_resolve):
        mock_detect.side_effect = [
            ([(20, 20, 90, 120)], "<20><20><90><120>"),
            ([(140, 30, 220, 130)], "<140><30><220><130>"),
        ]

        mock_model = mock.MagicMock()
        mock_clip = mock.MagicMock()
        tokenized = []
        mock_clip.tokenize.side_effect = lambda t: (tokenized.append(t) or ["tok"])
        mock_clip.encode_from_tokens.return_value = (torch.zeros((1, 8, 768)), torch.zeros((1, 768)))

        mock_vae = mock.MagicMock()
        mock_vae.encode.side_effect = lambda pixels: torch.zeros(
            (pixels.shape[0], 16, pixels.shape[1] // 8, pixels.shape[2] // 8)
        )
        mock_vae.decode.side_effect = lambda samples: torch.ones(
            (samples.shape[0], samples.shape[2] * 8, samples.shape[3] * 8, 3)
        )

        node = addon_nodes.ZiggartMultiSubjectDetailer()
        img = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

        out_img, crops, preview, mask, log_json = node.refine_multi_subject(
            image=img,
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            detector_model="model.gguf",
            detector_mmproj="mmproj.gguf",
            subject_1_target="female character",
            subject_1_detail_prompt="Zia wearing white Inca tunic",
            subject_2_target="male character",
            subject_2_detail_prompt="Esteban wearing saffron tunic",
            subject_3_target="creature",
            subject_3_detail_prompt="",
            detail_negative_prompt="blurry",
            crop_factor=1.5,
            target_detail_resolution=256,
            steps=8,
            cfg=1.5,
            denoise=0.4,
            sampler_name="euler",
            scheduler="simple",
            seed=42,
            feather_mask=8,
            free_vram_after=True,
        )

        self.assertEqual(out_img.shape, (1, 256, 256, 3))
        self.assertEqual(crops.shape[0], 2)
        self.assertIn("Zia wearing white Inca tunic", tokenized)
        self.assertIn("Esteban wearing saffron tunic", tokenized)
        logs = json.loads(log_json)
        self.assertEqual(len(logs), 2)


class ExampleWorkflowsTests(unittest.TestCase):
    def test_example_workflows_cover_all_18_nodes_and_valid_links(self):
        all_mappings = {
            **nodes.NODE_CLASS_MAPPINGS,
            **detailer_nodes.NODE_CLASS_MAPPINGS,
            **addon_nodes.NODE_CLASS_MAPPINGS,
            **wildcard_nodes.NODE_CLASS_MAPPINGS,
        }
        self.assertEqual(len(all_mappings), 18)

        wf_dir = ROOT / "example_workflows"
        self.assertTrue(wf_dir.is_dir(), "example_workflows directory must exist")
        wf_files = sorted(wf_dir.glob("*.json"))
        self.assertGreaterEqual(len(wf_files), 8)

        covered_nodes: set[str] = set()
        for wf_path in wf_files:
            data = json.loads(wf_path.read_text(encoding="utf-8"))
            self.assertIn("nodes", data)
            self.assertIn("links", data)
            node_by_id = {n["id"]: n for n in data["nodes"]}

            for n in data["nodes"]:
                ntype = n.get("type")
                if ntype in all_mappings:
                    covered_nodes.add(ntype)
                    cls = all_mappings[ntype]
                    it = cls.INPUT_TYPES()
                    valid_inputs = set(it.get("required", {}).keys()) | set(it.get("optional", {}).keys())
                    for inp in n.get("inputs", []):
                        self.assertIn(
                            inp["name"],
                            valid_inputs,
                            f"{wf_path.name}: invalid input '{inp['name']}' on {ntype}",
                        )
                    ret_types = getattr(cls, "RETURN_TYPES", ())
                    self.assertEqual(len(n.get("outputs", [])), len(ret_types))

            for link in data["links"]:
                _lid, src_id, src_slot, dst_id, dst_slot, _ltype = link
                self.assertIn(src_id, node_by_id, f"{wf_path.name}: missing src node {src_id}")
                self.assertIn(dst_id, node_by_id, f"{wf_path.name}: missing dst node {dst_id}")
                self.assertLess(src_slot, len(node_by_id[src_id].get("outputs", [])))
                self.assertLess(dst_slot, len(node_by_id[dst_id].get("inputs", [])))

        self.assertEqual(covered_nodes, set(all_mappings.keys()))


if __name__ == "__main__":
    unittest.main()
