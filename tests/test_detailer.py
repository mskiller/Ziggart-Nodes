from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_nodes_detailer_test"


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


class BoundingBoxParserTests(unittest.TestCase):
    def test_parse_wevisone_coordinate_tokens(self):
        text = "I found the eyes here: <150><200><350><400> and <600><200><800><400>."
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=1000, img_h=1000)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], (150, 200, 350, 400))
        self.assertEqual(boxes[1], (600, 200, 800, 400))

    def test_parse_box_2d_json(self):
        text = '{"box_2d": [200, 150, 400, 350], "label": "eye"}'
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=1000, img_h=1000)
        self.assertEqual(len(boxes), 1)
        # ymin=200, xmin=150, ymax=400, xmax=350 -> (x0, y0, x1, y1) = (150, 200, 350, 400)
        self.assertEqual(boxes[0], (150, 200, 350, 400))

    def test_parse_brackets_normalized_1000(self):
        text = "Coordinates: [100, 250, 300, 450]"
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=500, img_h=1000)
        self.assertEqual(len(boxes), 1)
        # x: 100/1000 * 500 = 50, 300/1000 * 500 = 150
        # y: 250/1000 * 1000 = 250, 450/1000 * 1000 = 450
        self.assertEqual(boxes[0], (50, 250, 150, 450))

    def test_parse_normalized_floats(self):
        text = "Detected at [0.1, 0.2, 0.4, 0.6]"
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=1000, img_h=1000)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (100, 200, 400, 600))

    def test_parse_no_boxes_returns_empty(self):
        text = "None. The requested object is not in the image."
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=1000, img_h=1000)
        self.assertEqual(boxes, [])

    def test_parse_box_2d_left_edge_uniform_scale(self):
        # x1=1 in 0..1000 space must NOT be treated as 1.0 (100% width) while y is 0..1000
        text = '{"box_2d": [200, 0, 400, 1], "label": "edge"}'
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=1000, img_h=1000)
        self.assertEqual(boxes, [(0, 200, 1, 400)])

    def test_parse_ignores_thinking_draft_boxes_and_supports_tuple_pairs(self):
        text = "<think>Maybe at [10, 10, 900, 900]?</think> Final location: <box>(100, 200), (300, 400)</box>"
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=1000, img_h=1000)
        self.assertEqual(boxes, [(100, 200, 300, 400)])

    def test_parse_spaced_wevisone_tokens(self):
        text = "<100> <200> <300> <400>"
        boxes = detailer_nodes.parse_bounding_boxes(text, img_w=1000, img_h=1000)
        self.assertEqual(boxes, [(100, 200, 300, 400)])


class CropAndMaskTests(unittest.TestCase):
    def test_crop_with_padding_is_divisible_by_8(self):
        box = (105, 107, 213, 219)
        px0, py0, px1, py1 = detailer_nodes.crop_with_padding(box, img_w=1000, img_h=1000, crop_factor=1.5)
        self.assertEqual((px1 - px0) % 8, 0)
        self.assertEqual((py1 - py0) % 8, 0)
        self.assertLessEqual(px0, box[0])
        self.assertLessEqual(py0, box[1])
        self.assertGreaterEqual(px1, box[2])
        self.assertGreaterEqual(py1, box[3])

    def test_crop_with_padding_bounds_clamping(self):
        box = (0, 0, 50, 50)
        px0, py0, px1, py1 = detailer_nodes.crop_with_padding(box, img_w=100, img_h=100, crop_factor=2.0)
        self.assertGreaterEqual(px0, 0)
        self.assertGreaterEqual(py0, 0)
        self.assertLessEqual(px1, 100)
        self.assertLessEqual(py1, 100)
        self.assertEqual((px1 - px0) % 8, 0)
        self.assertEqual((py1 - py0) % 8, 0)

    def test_crop_with_padding_near_right_edge_non_mod8_image(self):
        box = (92, 88, 101, 98)
        px0, py0, px1, py1 = detailer_nodes.crop_with_padding(box, img_w=102, img_h=99, crop_factor=1.5)
        self.assertGreaterEqual(px0, 0)
        self.assertGreaterEqual(py0, 0)
        self.assertLessEqual(px1, 102)
        self.assertLessEqual(py1, 99)
        self.assertEqual((px1 - px0) % 8, 0)
        self.assertEqual((py1 - py0) % 8, 0)

    def test_create_feathered_mask(self):
        mask = detailer_nodes.create_feathered_mask(crop_w=64, crop_h=64, feather_radius=8)
        self.assertEqual(mask.shape, (64, 64))
        # Center should be close to 1.0
        self.assertGreater(mask[32, 32].item(), 0.9)
        # Edges should be lower than center due to feathering
        self.assertLess(mask[0, 0].item(), 0.5)


class PreviewDrawTests(unittest.TestCase):
    def test_draw_boxes_on_image(self):
        img = torch.ones((1, 200, 200, 3), dtype=torch.float32)
        boxes = [(20, 30, 80, 90)]
        res = detailer_nodes.draw_boxes_on_image(img, boxes, label="eye")
        self.assertEqual(res.shape, (1, 200, 200, 3))
        self.assertEqual(res.dtype, torch.float32)


class MockSamplerExecutionTests(unittest.TestCase):
    def setUp(self):
        self.detailer = detailer_nodes.ZiggartDetailerKrea2()
        self.detector = detailer_nodes.ZiggartWeVisOneDetector()

    @mock.patch.object(detailer_nodes, "_resolve_gguf", return_value="dummy.gguf")
    @mock.patch.object(detailer_nodes, "detect_objects_with_vlm")
    def test_detailer_full_run_with_detection(self, mock_detect, mock_resolve):
        mock_detect.return_value = ([(50, 50, 150, 150)], "<50><50><150><150>")

        # Create mock model, clip, vae
        mock_model = mock.MagicMock()
        mock_clip = mock.MagicMock()
        mock_clip.tokenize.return_value = ["dummy"]
        mock_clip.encode_from_tokens.return_value = (torch.zeros((1, 10, 768)), torch.zeros((1, 768)))

        mock_vae = mock.MagicMock()
        mock_vae.encode.return_value = torch.zeros((1, 4, 32, 32))
        mock_vae.decode.return_value = torch.ones((1, 256, 256, 3))

        img = torch.ones((1, 256, 256, 3), dtype=torch.float32)

        out_img, crops_refined, crops_base, preview, mask, latent = self.detailer.refine(
            image=img,
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            detector_model="model.gguf",
            detector_mmproj="mmproj.gguf",
            detection_prompt="eyes",
            detail_positive_prompt="sharp eyes",
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
            feather_mask=16,
            free_vram_after=False,
        )

        self.assertEqual(out_img.shape, (1, 256, 256, 3))
        self.assertEqual(mask.shape, (1, 256, 256))
        self.assertEqual(preview.shape, (1, 256, 256, 3))
        self.assertIn("samples", latent)

    @mock.patch.object(detailer_nodes, "_resolve_gguf", return_value="dummy.gguf")
    @mock.patch.object(detailer_nodes, "detect_objects_with_vlm")
    def test_detailer_multiple_differing_aspect_ratio_crops(self, mock_detect, mock_resolve):
        # Two boxes with different aspect ratios
        mock_detect.return_value = (
            [(20, 20, 120, 80), (150, 100, 230, 240)],
            "<20><20><120><80><150><100><230><240>",
        )

        mock_model = mock.MagicMock()
        mock_clip = mock.MagicMock()
        mock_clip.tokenize.return_value = ["dummy"]
        mock_clip.encode_from_tokens.return_value = (torch.zeros((1, 10, 768)), torch.zeros((1, 768)))

        mock_vae = mock.MagicMock()
        mock_vae.encode.side_effect = lambda pixels: torch.zeros((1, 4, pixels.shape[1] // 8, pixels.shape[2] // 8))
        mock_vae.decode.side_effect = lambda samples: torch.ones((1, samples.shape[2] * 8, samples.shape[3] * 8, 3))

        img = torch.ones((1, 512, 512, 3), dtype=torch.float32)

        out_img, crops_refined, crops_base, preview, mask, latent = self.detailer.refine(
            image=img,
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            detector_model="model.gguf",
            detector_mmproj="mmproj.gguf",
            detection_prompt="eyes",
            detail_positive_prompt="sharp eyes",
            detail_negative_prompt="blurry",
            crop_factor=1.5,
            target_detail_resolution=768,
            steps=8,
            cfg=1.5,
            denoise=0.4,
            sampler_name="euler",
            scheduler="simple",
            seed=42,
            max_detections=2,
            feather_mask=16,
            free_vram_after=False,
        )

        self.assertEqual(out_img.shape, (1, 512, 512, 3))
        # Both crops should be safely batched to shape [2, max_h, max_w, 3] without tensor mismatch error!
        self.assertEqual(crops_refined.shape[0], 2)
        self.assertEqual(crops_base.shape[0], 2)
        self.assertEqual(crops_refined.ndim, 4)
        self.assertEqual(crops_base.ndim, 4)

    @mock.patch.object(detailer_nodes, "_resolve_gguf", return_value="dummy.gguf")
    @mock.patch.object(detailer_nodes, "detect_objects_with_vlm")
    def test_detector_bonus_node(self, mock_detect, mock_resolve):
        mock_detect.return_value = ([(30, 40, 120, 140)], "<30><40><120><140>")

        img = torch.ones((1, 200, 200, 3), dtype=torch.float32)
        preview, mask, crops, bboxes_json = self.detector.detect(
            image=img,
            detector_model="model.gguf",
            detector_mmproj="mmproj.gguf",
            detection_prompt="eyes",
            crop_factor=1.5,
            max_detections=2,
            feather_mask=8,
            free_vram_after=False,
        )

        self.assertEqual(preview.shape, (1, 200, 200, 3))
        self.assertEqual(mask.shape, (1, 200, 200))
        # Mask outside the unpadded box [30..120, 40..140] must remain 0 even when feather_mask > 0
        self.assertEqual(mask[0, 10, 10].item(), 0.0)
        self.assertGreater(mask[0, 90, 75].item(), 0.9)
        self.assertGreater(crops.shape[0], 0)
        data = json.loads(bboxes_json)
        self.assertEqual(len(data), 1)
        self.assertEqual(len(data[0]["detections"]), 1)

    @mock.patch.object(detailer_nodes, "_resolve_gguf", return_value="dummy.gguf")
    @mock.patch.object(detailer_nodes, "detect_objects_with_vlm")
    def test_detailer_rgba_batch_two_phase_and_cached_clip(self, mock_detect, mock_resolve):
        mock_detect.side_effect = [
            ([(20, 20, 80, 80), (100, 100, 160, 160)], "<20><20><80><80>"),
            ([(30, 30, 90, 90)], "<30><30><90><90>"),
        ]

        mock_model = mock.MagicMock()
        mock_clip = mock.MagicMock()
        mock_clip.tokenize.return_value = ["dummy"]
        mock_clip.encode_from_tokens.return_value = (torch.zeros((1, 10, 768)), torch.zeros((1, 768)))

        mock_vae = mock.MagicMock()
        mock_vae.encode.side_effect = lambda pixels: torch.zeros((pixels.shape[0], 16, pixels.shape[1] // 8, pixels.shape[2] // 8))
        mock_vae.decode.side_effect = lambda samples: torch.ones((samples.shape[0], samples.shape[2] * 8, samples.shape[3] * 8, 3))

        # 2-frame RGBA (4-channel) batch
        img = torch.ones((2, 256, 256, 4), dtype=torch.float32)
        img[:, :, :, 3] = 0.75

        out_img, crops_refined, crops_base, preview, mask, latent = self.detailer.refine(
            image=img,
            model=mock_model,
            clip=mock_clip,
            vae=mock_vae,
            detector_model="model.gguf",
            detector_mmproj="mmproj.gguf",
            detection_prompt="eyes",
            detail_positive_prompt="sharp eyes",
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
            feather_mask=16,
            free_vram_after=True,
        )

        # RGBA alpha channel preserved
        self.assertEqual(out_img.shape, (2, 256, 256, 4))
        self.assertTrue(torch.allclose(out_img[:, :, :, 3], torch.full((2, 256, 256), 0.75)))
        # Detector unload_detector_before_sampling is False on frame 0 and True on frame 1
        self.assertFalse(mock_detect.call_args_list[0].kwargs["unload_detector_before_sampling"])
        self.assertTrue(mock_detect.call_args_list[1].kwargs["unload_detector_before_sampling"])
        # CLIP tokenize called only twice total (1 pos + 1 neg) across all 3 crops!
        self.assertEqual(mock_clip.tokenize.call_count, 2)


if __name__ == "__main__":
    unittest.main()
