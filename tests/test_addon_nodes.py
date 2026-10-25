from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_nodes_addon_test"


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


class CropAndStitchNodesTests(unittest.TestCase):
    def test_crop_and_stitch_roundtrip_rgba(self):
        crop_node = addon_nodes.ZiggartCropByBBox()
        stitch_node = addon_nodes.ZiggartStitchByBBox()

        base_img = torch.zeros((1, 256, 256, 4), dtype=torch.float32)
        base_img[:, :, :, 3] = 0.85

        bboxes_json = json.dumps(
            [
                {
                    "frame_index": 0,
                    "detections": [
                        {
                            "id": 1,
                            "box": [40, 40, 100, 100],
                            "padded_crop": [24, 24, 120, 120],
                            "label": "eye",
                        }
                    ],
                }
            ]
        )

        crops, crop_masks, pipe = crop_node.crop(
            image=base_img,
            bboxes_json=bboxes_json,
            target_resolution=256,
            crop_factor_override=0.0,
            skip_upscale_if_larger=False,
            feather_mask=8,
        )
        self.assertEqual(crops.shape, (1, 256, 256, 4))
        self.assertEqual(crop_masks.shape, (1, 256, 256))

        # Simulate external node modifying the RGB crop to white (1.0)
        refined_crops = torch.ones((1, 256, 256, 3), dtype=torch.float32)
        stitched_img, comp_mask = stitch_node.stitch(
            stitch_pipe=pipe,
            refined_crops=refined_crops,
            feather_mask=8,
            blend_opacity=1.0,
        )

        self.assertEqual(stitched_img.shape, (1, 256, 256, 4))
        self.assertEqual(comp_mask.shape, (1, 256, 256))
        # Center of crop (72, 72) should now be close to 1.0 in RGB, outside (5, 5) should remain 0.0
        self.assertGreater(stitched_img[0, 72, 72, 0].item(), 0.9)
        self.assertEqual(stitched_img[0, 5, 5, 0].item(), 0.0)
        # Alpha channel preserved
        self.assertAlmostEqual(stitched_img[0, 72, 72, 3].item(), 0.85, places=5)


class AutoPromptDetailerTests(unittest.TestCase):
    @mock.patch.object(addon_nodes, "_resolve_gguf", return_value="dummy.gguf")
    @mock.patch.object(addon_nodes, "detect_objects_with_vlm")
    def test_auto_prompt_detailer_generates_custom_prompts_per_crop(self, mock_detect, mock_resolve):
        mock_detect.return_value = (
            [(20, 20, 80, 80), (140, 140, 200, 200)],
            "<20><20><80><80><140><140><200><200>",
        )

        class FakeManager:
            def __init__(self):
                self.chat_calls = 0
                self.unloaded = False

            def load(self, config, unload_comfy_models=True):
                pass

            def chat(self, messages, **kwargs):
                self.chat_calls += 1
                return f"custom micro prompt #{self.chat_calls}"

            def unload(self):
                self.unloaded = True

        fake_mgr = FakeManager()
        mock_model = mock.MagicMock()
        mock_clip = mock.MagicMock()
        tokenized_texts = []

        def record_tokenize(text):
            tokenized_texts.append(text)
            return ["tok"]

        mock_clip.tokenize.side_effect = record_tokenize
        mock_clip.encode_from_tokens.return_value = (torch.zeros((1, 8, 768)), torch.zeros((1, 768)))

        mock_vae = mock.MagicMock()
        mock_vae.encode.side_effect = lambda pixels: torch.zeros(
            (pixels.shape[0], 16, pixels.shape[1] // 8, pixels.shape[2] // 8)
        )
        mock_vae.decode.side_effect = lambda samples: torch.ones(
            (samples.shape[0], samples.shape[2] * 8, samples.shape[3] * 8, 3)
        )

        node = addon_nodes.ZiggartAutoPromptDetailer()
        img = torch.zeros((1, 256, 256, 3), dtype=torch.float32)

        with mock.patch.object(addon_nodes, "MODEL_MANAGER", fake_mgr):
            out_img, crops_ref, crops_base, preview, mask, prompts_json = node.auto_refine(
                image=img,
                model=mock_model,
                clip=mock_clip,
                vae=mock_vae,
                detector_model="model.gguf",
                detector_mmproj="mmproj.gguf",
                detection_prompt="hands",
                prompt_instruction="Describe hand.",
                base_positive_suffix="sharp 8k",
                detail_negative_prompt="blurry",
                crop_factor=1.5,
                target_detail_resolution=256,
                steps=8,
                cfg=1.5,
                denoise=0.4,
                sampler_name="euler",
                scheduler="simple",
                seed=42,
                max_detections=2,
                feather_mask=8,
                free_vram_after=True,
            )

        self.assertEqual(out_img.shape, (1, 256, 256, 3))
        self.assertEqual(fake_mgr.chat_calls, 2)
        self.assertTrue(fake_mgr.unloaded)
        parsed_prompts = json.loads(prompts_json)
        self.assertEqual(len(parsed_prompts), 2)
        self.assertEqual(parsed_prompts[0]["prompt"], "custom micro prompt #1, sharp 8k")
        self.assertEqual(parsed_prompts[1]["prompt"], "custom micro prompt #2, sharp 8k")
        self.assertIn("custom micro prompt #1, sharp 8k", tokenized_texts)
        self.assertIn("custom micro prompt #2, sharp 8k", tokenized_texts)


class CriticAndInpaintPrompterTests(unittest.TestCase):
    @mock.patch.object(addon_nodes, "_resolve_gguf", return_value="dummy.gguf")
    def test_vlm_image_critic_parses_defects_and_sets_gate(self, mock_resolve):
        class FakeManager:
            def load(self, config, unload_comfy_models=True):
                pass

            def chat(self, messages, **kwargs):
                return json.dumps(
                    {
                        "score": 4,
                        "issues": "Six fingers on left hand",
                        "fix_prompt": "anatomically correct five-fingered hand, natural skin texture",
                        "boxes": [[200, 300, 400, 500]],
                    }
                )

            def unload(self):
                pass

        node = addon_nodes.ZiggartVLMImageCritic()
        img = torch.ones((1, 200, 200, 3), dtype=torch.float32)

        with mock.patch.object(addon_nodes, "MODEL_MANAGER", FakeManager()):
            preview, mask, bboxes_json, fix_prompt, report_json, needs_refinement = node.critique(
                image=img,
                model="model.gguf",
                mmproj="mmproj.gguf",
                inspection_focus="hands_and_limbs",
                crop_factor=1.5,
                max_detections=4,
                feather_mask=8,
                free_vram_after=True,
            )

        self.assertTrue(needs_refinement)
        self.assertEqual(fix_prompt, "anatomically correct five-fingered hand, natural skin texture")
        self.assertEqual(preview.shape, (1, 200, 200, 3))
        self.assertEqual(mask.shape, (1, 200, 200))
        report = json.loads(report_json)
        self.assertEqual(report[0]["score"], 4)
        self.assertEqual(report[0]["defect_count"], 1)

    @mock.patch.object(addon_nodes, "_resolve_gguf", return_value="dummy.gguf")
    def test_vlm_inpaint_prompter_extracts_mask_bbox(self, mock_resolve):
        class FakeManager:
            def load(self, config, unload_comfy_models=True):
                pass

            def chat(self, messages, **kwargs):
                return "vintage gold aviator sunglasses with warm sunset reflections"

            def unload(self):
                pass

        node = addon_nodes.ZiggartVLMInpaintPrompter()
        img = torch.ones((1, 200, 200, 3), dtype=torch.float32)
        mask = torch.zeros((1, 200, 200), dtype=torch.float32)
        mask[0, 50:100, 60:140] = 1.0

        with mock.patch.object(addon_nodes, "MODEL_MANAGER", FakeManager()):
            pos, neg, preview, bboxes_json = node.generate_inpaint_prompt(
                image=img,
                mask=mask,
                model="model.gguf",
                mmproj="mmproj.gguf",
                edit_instruction="add aviator sunglasses",
                crop_factor=1.5,
                free_vram_after=True,
            )

        self.assertEqual(pos, "vintage gold aviator sunglasses with warm sunset reflections")
        self.assertEqual(preview.shape, (1, 200, 200, 3))
        bbox_data = json.loads(bboxes_json)
        self.assertEqual(bbox_data[0]["detections"][0]["box"], [60, 50, 140, 100])


class SmartTileUpscalerTests(unittest.TestCase):
    def test_smart_tile_upscaler_pure_krea2_mode_without_vlm(self):
        node = addon_nodes.ZiggartSmartTileUpscaler()
        mock_model = mock.MagicMock()
        mock_clip = mock.MagicMock()
        mock_clip.tokenize.return_value = ["tok"]
        mock_clip.encode_from_tokens.return_value = (torch.zeros((1, 8, 768)), torch.zeros((1, 768)))

        mock_vae = mock.MagicMock()
        mock_vae.encode.side_effect = lambda pixels: torch.zeros(
            (pixels.shape[0], 16, pixels.shape[1] // 8, pixels.shape[2] // 8)
        )
        mock_vae.decode.side_effect = lambda samples: torch.ones(
            (samples.shape[0], samples.shape[2] * 8, samples.shape[3] * 8, 3)
        ) * 0.5

        img = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
        # With enable_vlm_tile_prompts=False, no GGUF files are resolved or loaded
        upscaled, tile_batch, tile_prompts_json = node.upscale(
            image=img,
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            upscale_by=2.0,
            tile_size=128,
            tile_overlap=32,
            enable_vlm_tile_prompts=False,
            base_positive_prompt="sharp 8k",
            negative_prompt="blurry",
            steps=8,
            cfg=1.5,
            denoise=0.35,
            sampler_name="euler",
            scheduler="simple",
            seed=42,
            free_vram_after=True,
        )

        # 128x128 upscaled by 2.0 -> 256x256
        self.assertEqual(upscaled.shape, (1, 256, 256, 3))
        self.assertTrue(torch.allclose(upscaled, torch.full((1, 256, 256, 3), 0.5), atol=1e-4))
        self.assertGreater(tile_batch.shape[0], 1)
        # CLIP tokenize called only once for positive and once for negative
        self.assertEqual(mock_clip.tokenize.call_count, 2)


if __name__ == "__main__":
    unittest.main()
