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

        class FakeHandler:
            closed = False

            def close(self):
                self.closed = True

        model = FakeModel()
        handler = FakeHandler()
        manager = embedded_model.EmbeddedModelManager()
        manager._model = model
        manager._handler = handler
        manager._config = object()
        manager.unload()
        self.assertTrue(model.closed)
        self.assertTrue(handler.closed)
        self.assertIsNone(manager._model)
        self.assertIsNone(manager._handler)

    def test_strip_thinking_tags_and_type_error_fallback(self):
        def legacy_handler(llama, messages, max_tokens, temperature, top_p, top_k, seed, stream):
            return {"choices": [{"message": {"content": "<think>draft thoughts</think>\nClean final prompt"}}]}

        manager = embedded_model.EmbeddedModelManager()
        manager._model = object()
        manager._handler = legacy_handler
        answer = manager.chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            seed=42,
            enable_thinking=True,
        )
        self.assertEqual(answer, "Clean final prompt")

    def test_is_compatible_loaded_config_ignores_thread_differences_and_chat_batch(self):
        cfg_a = embedded_model.ModelConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
            context_size=8192,
            gpu_layers=-1,
            threads=14,
            flash_attention=True,
            mmproj_on_gpu=True,
        )
        cfg_b_threads_diff = embedded_model.ModelConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
            context_size=8192,
            gpu_layers=-1,
            threads=8,
            flash_attention=True,
            mmproj_on_gpu=True,
        )
        cfg_c_ctx_diff = embedded_model.ModelConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
            context_size=4096,
            gpu_layers=-1,
            threads=14,
            flash_attention=True,
            mmproj_on_gpu=True,
        )
        manager = embedded_model.EmbeddedModelManager()
        manager._model = object()
        manager._config = cfg_a
        self.assertTrue(manager._is_compatible_loaded_config(cfg_b_threads_diff))
        self.assertFalse(manager._is_compatible_loaded_config(cfg_c_ctx_diff))

        seen_seeds = []

        def batch_handler(**kwargs):
            seen_seeds.append(kwargs.get("seed"))
            return {"choices": [{"message": {"content": f"<think>x</think>item-{len(seen_seeds)}"}}]}

        manager._handler = batch_handler
        outputs = manager.chat_batch(
            [
                [{"role": "user", "content": "crop 1"}],
                [{"role": "user", "content": "crop 2"}],
            ],
            max_tokens=32,
            temperature=0.2,
            top_p=0.9,
            top_k=30,
            seeds=[101, 202],
            enable_thinking=False,
        )
        self.assertEqual(outputs, ["item-1", "item-2"])
        self.assertEqual(seen_seeds, [101, 202])


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

        self.assertEqual(len(result), 5)
        self.assertEqual(result[:4], ("final prompt", "composition notes", "", ""))
        session = result[4]
        self.assertIsInstance(session, nodes.VisionChatSession)
        self.assertEqual(session.initial_answer, "final prompt")
        self.assertTrue(session.thumbnail_uri.startswith("data:image/jpeg;base64,"))
        self.assertEqual(session.messages[-1], {"role": "assistant", "content": "final prompt"})
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

    def test_prompt_reviser_parses_json_and_sends_both_images(self):
        class FakeManager:
            def __init__(self):
                self.calls = []
                self.loaded_config = None
                self.unloaded = False

            def load(self, config, unload_comfy_models):
                self.loaded_config = config

            def chat(self, messages, **kwargs):
                self.calls.append(messages)
                return (
                    "```json\n"
                    '{"fidelity_score": 6.5, '
                    '"missing_or_wrong": ["red scarf missing"], '
                    '"hallucinated_elements": ["extra hat"], '
                    '"notes": "Keep pose, fix outfit.", '
                    '"revised_prompt": "a woman wearing a red scarf, no hat"}\n'
                    "```"
                )

            def unload(self):
                self.unloaded = True

        fake = FakeManager()
        original = nodes.MODEL_MANAGER
        nodes.MODEL_MANAGER = fake
        cfg = embedded_model.ModelConfig(
            model_path="shared.gguf",
            mmproj_path="shared-mm.gguf",
            context_size=8192,
            gpu_layers=-1,
            threads=4,
            flash_attention=True,
            mmproj_on_gpu=True,
        )
        try:
            orig_img = np.zeros((1, 8, 8, 3), dtype=np.float32)
            gen_img = np.ones((1, 8, 8, 3), dtype=np.float32)
            revised, report, score, needs_revision = nodes.ZiggartGGUFVisionPromptReviser().revise(
                original_image=orig_img,
                generated_image=gen_img,
                current_prompt="a woman wearing a hat",
                model=nodes.NO_MODEL,
                mmproj=nodes.NO_MMPROJ,
                free_vram_after=True,
                enable_thinking=False,
                fidelity_threshold=8.5,
                previous_diff_notes="Pass 1: fix background",
                vlm_config=cfg,
            )
        finally:
            nodes.MODEL_MANAGER = original

        self.assertEqual(fake.loaded_config, cfg)
        self.assertEqual(revised, "a woman wearing a red scarf, no hat")
        self.assertAlmostEqual(score, 6.5)
        self.assertTrue(needs_revision)
        self.assertIn("red scarf missing", report)
        self.assertIn("extra hat", report)
        user_content = fake.calls[0][-1]["content"]
        self.assertEqual([p["type"] for p in user_content].count("image_url"), 2)
        self.assertIn("a woman wearing a hat", user_content[0]["text"])
        self.assertIn("Pass 1: fix background", user_content[0]["text"])
        self.assertTrue(fake.unloaded)

    def test_prompt_reviser_plain_text_fallback(self):
        class FakeManager:
            def load(self, config, unload_comfy_models):
                pass

            def chat(self, messages, **kwargs):
                return "clean fallback prompt"

            def unload(self):
                pass

        original = nodes.MODEL_MANAGER
        nodes.MODEL_MANAGER = FakeManager()
        try:
            with mock.patch.object(nodes, "_resolve_gguf", side_effect=["m.gguf", "mm.gguf"]):
                orig_img = np.zeros((1, 4, 4, 3), dtype=np.float32)
                gen_img = np.zeros((1, 4, 4, 3), dtype=np.float32)
                revised, report, score, needs_revision = nodes.ZiggartGGUFVisionPromptReviser().revise(
                    original_image=orig_img,
                    generated_image=gen_img,
                    current_prompt="initial prompt",
                    model=nodes.NO_MODEL,
                    mmproj=nodes.NO_MMPROJ,
                    free_vram_after=False,
                    enable_thinking=False,
                    fidelity_threshold=4.0,
                )
        finally:
            nodes.MODEL_MANAGER = original

        self.assertEqual(revised, "clean fallback prompt")
        self.assertEqual(score, 5.0)
        self.assertFalse(needs_revision)
        self.assertIn("fallback", report.lower())

    def test_prompt_revision_gate_selection_and_notes_accumulation(self):
        gate = nodes.ZiggartPromptRevisionGate()
        active, notes, cont = gate.select(
            current_prompt="old prompt",
            revised_prompt="new prompt",
            needs_revision=True,
            latest_diff_report="Fixed scarf",
            previous_notes="Pass 1: initial",
            iteration_label="Pass 2",
        )
        self.assertEqual(active, "new prompt")
        self.assertTrue(cont)
        self.assertEqual(notes, "Pass 1: initial\n\n[Pass 2]\nFixed scarf")

        active_done, _, cont_done = gate.select(
            current_prompt="converged prompt",
            revised_prompt="tweaked prompt",
            needs_revision=False,
        )
        self.assertEqual(active_done, "converged prompt")
        self.assertFalse(cont_done)

        active_forced, _, _ = gate.select(
            current_prompt="converged prompt",
            revised_prompt="tweaked prompt",
            needs_revision=False,
            force_use_revised=True,
        )
        self.assertEqual(active_forced, "tweaked prompt")

    def test_revision_nodes_registered_in_mappings(self):
        self.assertIn("ZiggartGGUFVisionPromptReviser", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("ZiggartPromptRevisionGate", nodes.NODE_CLASS_MAPPINGS)
        self.assertIn("ZiggartGGUFVisionPromptReviser", nodes.NODE_DISPLAY_NAME_MAPPINGS)
        self.assertIn("ZiggartPromptRevisionGate", nodes.NODE_DISPLAY_NAME_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
