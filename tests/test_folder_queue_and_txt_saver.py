import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_nodes_folder_test"


def load_module(name, filename):
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


class FolderImageQueueLoaderTests(unittest.TestCase):
    def _save_image(self, path: Path, color=(255, 0, 0), mode="RGB", size=(16, 12)):
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new(mode, size, color=color)
        img.save(path)

    def test_scans_multiple_formats_and_natural_sort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_image(root / "img_10.png")
            self._save_image(root / "img_2.jpg")
            self._save_image(root / "img_1.webp")
            (root / "notes.txt").write_text("not an image", encoding="utf-8")

            loader = nodes.ZiggartFolderImageQueueLoader()
            out0 = loader.load_image(folder_path=str(root), image_index=0)
            (
                img_tensor,
                mask_tensor,
                stem,
                filename,
                file_path,
                directory,
                curr_idx,
                total_queues,
                remaining_queues,
                summary,
            ) = out0

            self.assertEqual(tuple(img_tensor.shape), (1, 12, 16, 3))
            self.assertEqual(tuple(mask_tensor.shape), (1, 12, 16))
            self.assertEqual(stem, "img_1")
            self.assertEqual(filename, "img_1.webp")
            self.assertEqual(curr_idx, 0)
            self.assertEqual(total_queues, 3)
            self.assertEqual(remaining_queues, 2)
            self.assertIn("Total Queues Needed: 3", summary)

            # Index 1 -> img_2.jpg (natural sort puts 2 before 10)
            out1 = loader.load_image(folder_path=str(root), image_index=1)
            self.assertEqual(out1[3], "img_2.jpg")
            self.assertEqual(out1[6], 1)
            self.assertEqual(out1[8], 1)

            # Index 2 -> img_10.png
            out2 = loader.load_image(folder_path=str(root), image_index=2)
            self.assertEqual(out2[3], "img_10.png")
            self.assertEqual(out2[6], 2)
            self.assertEqual(out2[8], 0)

    def test_subfolders_and_skip_existing_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_image(root / "a_01.png")
            (root / "a_01.txt").write_text("already captioned", encoding="utf-8")
            self._save_image(root / "a_02.png")
            self._save_image(root / "sub" / "a_03.webp")

            loader = nodes.ZiggartFolderImageQueueLoader()

            # Without subfolders, with skip_existing_txt=True -> only a_02.png
            res_no_sub = loader.load_image(
                folder_path=str(root),
                image_index=0,
                include_subfolders=False,
                skip_existing_txt=True,
            )
            self.assertEqual(res_no_sub[3], "a_02.png")
            self.assertEqual(res_no_sub[7], 1)  # 1 queue needed

            # With subfolders and skip_existing_txt=True -> a_02.png and sub/a_03.webp (2 queues)
            res_with_sub = loader.load_image(
                folder_path=str(root),
                image_index=1,
                include_subfolders=True,
                skip_existing_txt=True,
            )
            self.assertEqual(res_with_sub[3], "a_03.webp")
            self.assertEqual(res_with_sub[7], 2)
            self.assertEqual(res_with_sub[8], 0)


class VisionPromptTxtSaverTests(unittest.TestCase):
    def test_flattens_multiline_prompt_to_single_line_matching_krea2_rules(self):
        multiline_prompt = (
            "A highly detailed, anime-style digital painting of a young teenage girl sitting on stone steps in an urban alleyway. "
            "She has shoulder-length brown hair with bangs and wears a white headband.\n\n"
            "The setting features weathered blue-green walls and warm golden sunlight streaming from the right, "
            "casting soft shadows and highlighting falling yellow leaves."
        )
        with tempfile.TemporaryDirectory() as tmp:
            saver = nodes.ZiggartVisionPromptTxtSaver()
            res1 = saver.save_prompt(
                prompt_text=multiline_prompt,
                output_folder=tmp,
                master_filename="all_prompts_krea2.txt",
                write_mode="append_new_line",
                flatten_to_single_line=True,
                clean_markdown_formatting=False,
                skip_duplicate_lines=True,
            )["result"]

            formatted, master_path, total_lines, report = res1
            self.assertNotIn("\n", formatted)
            self.assertNotIn("\r", formatted)
            self.assertEqual(total_lines, 1)

            # Second pass with a new multiline prompt containing Markdown headers & bullets
            md_prompt = (
                "A hyper-detailed illustration of a warrior queen.\n\n"
                "**Attire & Accessories:**\n"
                "- **Armor:** Golden plate armor with emerald gems.\n"
                "- Flowing crimson cape.\n\n"
                "**Lighting & Atmosphere:**\n"
                "- Warm torchlight in a stone hall."
            )
            res2 = saver.save_prompt(
                prompt_text=md_prompt,
                output_folder=tmp,
                master_filename="all_prompts_krea2.txt",
                write_mode="append_new_line",
                flatten_to_single_line=True,
                clean_markdown_formatting=True,
                skip_duplicate_lines=True,
                save_individual_sidecar_txt=True,
                source_image_path=str(Path(tmp) / "queen_002.webp"),
                line_prefix="(Masterpiece),",
                line_suffix="8k",
            )["result"]

            formatted2, _, total_lines2, _ = res2
            self.assertEqual(total_lines2, 2)
            self.assertNotIn("**Attire", formatted2)
            self.assertNotIn("\n", formatted2)
            self.assertTrue(formatted2.startswith("(Masterpiece),"))
            self.assertTrue(formatted2.endswith("8k"))
            self.assertTrue((Path(tmp) / "queen_002.txt").is_file())

            # Verify file on disk has exactly 2 lines and 0 empty lines
            raw_lines = Path(master_path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), 2)
            self.assertTrue(all(line.strip() for line in raw_lines))

            # Duplicate pass should be skipped when skip_duplicate_lines=True
            res_dup = saver.save_prompt(
                prompt_text=multiline_prompt,
                output_folder=tmp,
                master_filename="all_prompts_krea2.txt",
                write_mode="append_new_line",
                flatten_to_single_line=True,
                skip_duplicate_lines=True,
            )["result"]
            self.assertEqual(res_dup[2], 2)

    def test_resolve_sidecar_txt_path_supports_relative_image_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest_dir = Path(tmp)
            rel_img = dest_dir / "subfolder" / "sample_01.webp"
            rel_img.parent.mkdir(parents=True, exist_ok=True)
            rel_img.write_bytes(b"RIFF")
            rel_sidecar = nodes._resolve_sidecar_txt_path(
                dest_dir=dest_dir,
                source_image_path=str(rel_img),
                master_stem="all_prompts",
                line_number=1,
            )
            self.assertEqual(rel_sidecar.name, "sample_01.txt")
            self.assertEqual(rel_sidecar.parent, rel_img.parent.resolve())


class DatasetCaptionCuratorTests(unittest.TestCase):
    def _save_image(self, path: Path, color=(40, 120, 200), size=(24, 24)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=color).save(path)

    def test_curate_single_caption_strips_boilerplate_and_injects_triggers_idempotently(self):
        raw = "In this image, a silver-haired elf warrior stands in a forest, wearing emerald armor."
        curated, meta = nodes.curate_single_caption_text(
            raw,
            trigger_word_prefix="ziggart_style",
            trigger_word_suffix="masterpiece 8k",
            format_mode="natural_prose",
            strip_vlm_boilerplate=True,
            clean_markdown=True,
            find_and_replace_rules="emerald armor -> mithril plate",
            banned_phrases="in a forest",
        )
        self.assertTrue(curated.startswith("ziggart_style, "))
        self.assertTrue(curated.endswith("masterpiece 8k."))
        self.assertIn("mithril plate", curated)
        self.assertNotIn("in this image", curated.lower())
        self.assertNotIn("in a forest", curated.lower())
        self.assertTrue(meta["boilerplate_stripped"])

        # Running a second time must be idempotent (no duplicate trigger prefix/suffix)
        curated_twice, _ = nodes.curate_single_caption_text(
            curated,
            trigger_word_prefix="ziggart_style",
            trigger_word_suffix="masterpiece 8k",
            format_mode="natural_prose",
            strip_vlm_boilerplate=True,
            clean_markdown=True,
        )
        self.assertEqual(curated_twice.lower().count("ziggart_style"), 1)
        self.assertEqual(curated_twice.lower().count("masterpiece 8k"), 1)

    def test_curator_node_audits_dataset_and_writes_in_place(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._save_image(root / "01.png")
            (root / "01.txt").write_text(
                "Here is a detailed prompt: A cinematic portrait of a cyberpunk hacker.",
                encoding="utf-8",
            )
            self._save_image(root / "02.png")
            # 02.txt intentionally missing to test missing_sidecar detection & fallback

            curator = nodes.ZiggartDatasetCaptionCuratorAndTriggerInjector()
            res_dry = curator.curate_dataset(
                folder_path=str(root),
                write_mode="dry_run_preview_only",
                trigger_word_prefix="ohwx_char",
                trigger_word_suffix="lora_v1",
                format_mode="natural_prose",
                strip_vlm_boilerplate=True,
                clean_markdown=True,
                find_and_replace_rules="cyberpunk hacker -> neon netrunner",
                banned_phrases="",
                max_token_budget=128,
                include_subfolders=False,
                default_caption_if_missing="",
                max_preview_tiles=8,
            )

            sheet, joined, report_json, count = res_dry
            self.assertEqual(count, 1)
            self.assertEqual(len(sheet.shape), 4)
            self.assertEqual(sheet.shape[0], 1)
            self.assertEqual(sheet.shape[3], 3)
            self.assertIn("ohwx_char, A cinematic portrait of a neon netrunner", joined)
            # Dry run must not have created 02.txt on disk yet
            self.assertFalse((root / "02.txt").exists())

            report = json.loads(report_json)
            self.assertEqual(report["total_images"], 2)
            self.assertEqual(report["missing_txt_count"], 1)

            # Now run in write_in_place_atomic mode
            curator.curate_dataset(
                folder_path=str(root),
                write_mode="write_in_place_atomic",
                trigger_word_prefix="ohwx_char",
                trigger_word_suffix="lora_v1",
                default_caption_if_missing="standing in a neon alley",
            )
            self.assertTrue((root / "02.txt").is_file())
            self.assertIn("ohwx_char", (root / "01.txt").read_text(encoding="utf-8"))
            self.assertIn("ohwx_char", (root / "02.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
