import base64
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_nodes_test"


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


class ImageConversionTests(unittest.TestCase):
    def decode(self, uri):
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))

    def test_rgb_float_image_becomes_png(self):
        array = np.zeros((2, 3, 3), dtype=np.float32)
        array[0, 0] = [1.0, 0.5, 0.0]
        image = self.decode(nodes.image_to_data_uri(array))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (3, 2))
        self.assertEqual(image.getpixel((0, 0)), (255, 128, 0))

    def test_rgba_is_composited_to_rgb(self):
        array = np.array([[[1.0, 0.0, 0.0, 0.5]]], dtype=np.float32)
        image = self.decode(nodes.image_to_data_uri(array))
        self.assertEqual(image.getpixel((0, 0)), (128, 0, 0))

    def test_invalid_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected an HWC image"):
            nodes.image_to_data_uri(np.zeros((3, 4), dtype=np.float32))

    def test_max_dimension_downscales_without_changing_aspect_ratio(self):
        array = np.zeros((100, 200, 3), dtype=np.float32)
        image = self.decode(nodes.image_to_data_uri(array, max_dimension=50))
        self.assertEqual(image.size, (50, 25))

    def test_max_dimension_never_upscales(self):
        array = np.zeros((10, 20, 3), dtype=np.float32)
        image = self.decode(nodes.image_to_data_uri(array, max_dimension=50))
        self.assertEqual(image.size, (20, 10))


class EmbeddedManagerTests(unittest.TestCase):
    def test_chat_passes_thinking_flag_to_mtmd_handler(self):
        calls = {}

        def handler(**kwargs):
            calls.update(kwargs)
            return {"choices": [{"message": {"content": " answer "}}]}

        manager = embedded_model.EmbeddedModelManager()
        manager._model = object()
        manager._handler = handler
        answer = manager.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=32,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            seed=-1,
            enable_thinking=False,
        )
        self.assertEqual(answer, "answer")
        self.assertFalse(calls["enable_thinking"])
        self.assertIsNone(calls["seed"])

    def test_unload_closes_model(self):
        class FakeModel:
            closed = False

            def close(self):
                self.closed = True

        model = FakeModel()
        manager = embedded_model.EmbeddedModelManager()
        manager._model = model
        manager._handler = object()
        manager._config = object()
        manager.unload()
        self.assertTrue(model.closed)
        self.assertIsNone(manager._model)
        self.assertIsNone(manager._handler)


class NodeExecutionTests(unittest.TestCase):
    def test_optional_images_and_specialist_pass_feed_final_synthesis(self):
        class FakeManager:
            def __init__(self):
                self.calls = []
                self.unloaded = False

            def load(self, config, unload_comfy_models):
                self.config = config

            def chat(self, messages, **kwargs):
                self.calls.append(messages)
                return "composition notes" if len(self.calls) == 1 else "final prompt"

            def unload(self):
                self.unloaded = True

        fake = FakeManager()
        original = nodes.MODEL_MANAGER
        nodes.MODEL_MANAGER = fake
        try:
            with mock.patch.object(
                nodes, "_resolve_gguf", side_effect=["model.gguf", "mmproj.gguf"]
            ):
                image = np.zeros((1, 8, 12, 3), dtype=np.float32)
                crop = np.zeros((1, 4, 4, 3), dtype=np.float32)
                reference = np.zeros((1, 6, 6, 3), dtype=np.float32)
                result = nodes.GGUFVisionPrompt().generate(
                    image=image,
                    model=nodes.NO_MODEL,
                    mmproj=nodes.NO_MMPROJ,
                    system_prompt="Write a prompt.",
                    prompt="Describe it.",
                    free_vram_after=True,
                    enable_thinking=False,
                    detail_crop=crop,
                    reference_image=reference,
                    analyze_composition=True,
                )
        finally:
            nodes.MODEL_MANAGER = original

        self.assertEqual(result, ("final prompt", "composition notes", "", ""))
        self.assertEqual(len(fake.calls), 2)
        specialist_messages = fake.calls[0]
        self.assertEqual(specialist_messages[0]["content"], "Write a prompt.")
        specialist_text = specialist_messages[-1]["content"][0]["text"]
        self.assertIn("Describe it.", specialist_text)
        self.assertIn("separate composition analysis pass", specialist_text)
        self.assertNotIn("Analyze the visual composition only", specialist_text)
        final_text = fake.calls[-1][-1]["content"][0]["text"]
        self.assertIn("Composition analysis", final_text)
        content_types = [part["type"] for part in fake.calls[-1][-1]["content"]]
        self.assertEqual(content_types.count("image_url"), 3)
        self.assertTrue(fake.unloaded)


if __name__ == "__main__":
    unittest.main()
