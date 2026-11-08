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


class UpgradeBSelfHealingAndMattingTests(unittest.TestCase):
    def test_contour_mask_refiner_and_stitch_integration(self):
        img = torch.zeros((1, 128, 128, 3), dtype=torch.float32)
        # Bright foreground object in the center of bbox [20, 20, 100, 100]
        img[:, 40:80, 40:80, :] = torch.tensor([0.9, 0.3, 0.2])

        refiner = addon_nodes.ZiggartContourMaskRefiner()
        bboxes = [(20, 20, 100, 100)]
        comb_mask, crop_masks, preview, refined_bboxes = refiner.refine_contours(
            image=img,
            refinement_mode="edge_and_chroma_guided",
            crop_factor=1.2,
            expand_contour_px=4,
            feather_radius_px=6,
            bboxes=bboxes,
        )
        self.assertEqual(comb_mask.shape, (1, 128, 128))
        self.assertEqual(preview.shape, (1, 128, 128, 3))
        self.assertEqual(len(refined_bboxes), 1)
        self.assertIn("contour_mask", refined_bboxes[0])
        # Center of object should have higher mask value than outer corner
        self.assertGreater(comb_mask[0, 60, 60].item(), comb_mask[0, 22, 22].item())

    def test_color_and_frequency_matcher_locks_tone_and_preserves_detail(self):
        orig = torch.full((1, 64, 64, 3), 0.35, dtype=torch.float32)
        # Refined has shifted overall brightness (0.75) plus high-frequency checkerboard detail
        refined = torch.full((1, 64, 64, 3), 0.75, dtype=torch.float32)
        refined[:, ::2, ::2, :] += 0.12

        matcher = addon_nodes.ZiggartColorAndFrequencyMatcher()
        matched, heatmap = matcher.match_color(
            original_image=orig,
            refined_image=refined,
            color_match_mode="lab_reinhard_transfer",
            color_match_strength=1.0,
            high_freq_detail_preserve=1.0,
            low_freq_blur_radius=5,
        )
        self.assertEqual(matched.shape, (1, 64, 64, 3))
        self.assertEqual(heatmap.shape, (1, 64, 64, 3))
        # Mean luminance of matched should be much closer to orig (0.35) than refined (0.78)
        self.assertLess(abs(matched.mean().item() - orig.mean().item()), 0.08)
        # High-frequency detail (std > 0) preserved
        self.assertGreater(matched.std().item(), 0.01)

    def test_closed_loop_self_healing_detailer_iterates_and_improves(self):
        node = addon_nodes.ZiggartClosedLoopSelfHealingDetailer()
        mock_model = mock.MagicMock()
        mock_clip = mock.MagicMock()
        mock_clip.tokenize.return_value = ["tok"]
        mock_clip.encode_from_tokens.return_value = (torch.zeros((1, 8, 64)), torch.zeros((1, 64)))

        mock_vae = mock.MagicMock()
        mock_vae.encode.side_effect = lambda pixels: torch.zeros(
            (pixels.shape[0], 16, pixels.shape[1] // 8, pixels.shape[2] // 8)
        )

        # Decode produces a crisp textured crop so signal quality score increases
        def fake_decode(samples):
            h = samples.shape[2] * 8
            w = samples.shape[3] * 8
            out = torch.full((samples.shape[0], h, w, 3), 0.5, dtype=torch.float32)
            out[:, ::2, ::2, :] = 0.85
            out[:, 1::2, 1::2, :] = 0.15
            return out

        mock_vae.decode.side_effect = fake_decode

        # Flat blurry input image -> low initial score
        img = torch.full((1, 128, 128, 3), 0.5, dtype=torch.float32)
        healed_img, best_score, iters_run, healed_mask, audit_json = node.self_heal(
            image=img,
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            detail_positive_prompt="sharp detailed face",
            detail_negative_prompt="blurry",
            target_score=8.5,
            max_iterations=2,
            crop_factor=1.2,
            target_detail_resolution=128,
            steps=4,
            cfg=1.5,
            base_denoise=0.40,
            step_denoise_boost=0.10,
            sampler_name="euler",
            scheduler="simple",
            seed=42,
            feather_mask=8,
            use_contour_mask=True,
            lock_color_and_lighting=True,
            never_degrade=True,
            bboxes=[(16, 16, 112, 112)],
        )
        self.assertEqual(healed_img.shape, (1, 128, 128, 3))
        self.assertGreaterEqual(iters_run, 1)
        self.assertGreater(best_score, 3.0)
        audit = json.loads(audit_json)
        self.assertEqual(len(audit["regions"]), 1)

        # Test VLM scoring via MODEL_MANAGER.chat + mismatched contour_mask resizing + ZiggartCropByBBox ZIGGART_BBOX support
        fake_vlm_cfg = mock.MagicMock()
        fake_vlm_cfg.model_path = "dummy_qwen3vl.gguf"
        with mock.patch.object(addon_nodes.MODEL_MANAGER, "load") as mock_load, mock.patch.object(
            addon_nodes.MODEL_MANAGER,
            "chat",
            return_value='{"score": 9.2, "fix_hint": "clean hands"}',
        ) as mock_chat:
            v_score, v_hint = addon_nodes._score_crop_with_vlm_or_signal(
                img[:, :64, :64, :], fake_vlm_cfg, "hero portrait"
            )
            mock_load.assert_called_once()
            mock_chat.assert_called_once()
            self.assertGreater(v_score, 6.0)
            self.assertEqual(v_hint, "clean hands")

        # Test mismatched contour_mask tensor inside bboxes dict
        mismatched_bbox_payload = [
            {
                "frame_index": 0,
                "box": (16, 16, 112, 112),
                "padded_crop": (8, 8, 120, 120),
                "contour_mask": torch.ones((32, 32), dtype=torch.float32),
            }
        ]
        healed_img2, _, _, _, _ = node.self_heal(
            image=img,
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            detail_positive_prompt="sharp detailed face",
            detail_negative_prompt="blurry",
            target_score=8.5,
            max_iterations=1,
            bboxes=mismatched_bbox_payload,
        )
        self.assertEqual(healed_img2.shape, (1, 128, 128, 3))

        # Test ZiggartCropByBBox with ZIGGART_BBOX input
        crop_node = addon_nodes.ZiggartCropByBBox()
        crops_t, masks_t, pipe = crop_node.crop(
            image=img,
            bboxes=[(16, 16, 80, 80)],
            target_resolution=64,
        )
        self.assertEqual(crops_t.shape[0], 1)
        self.assertEqual(masks_t.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
