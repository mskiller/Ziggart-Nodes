import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ziggart_vision_chat_test"


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


class FakeModelManager:
    def __init__(self, responses=None, fail_once=False):
        self.responses = list(responses or [])
        self.fail_once = fail_once
        self.calls = []
        self.load_calls = 0
        self.unload_calls = 0

    def load(self, config, unload_comfy_models):
        self.load_calls += 1
        self.last_config = config
        self.last_unload_before = unload_comfy_models

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.fail_once:
            self.fail_once = False
            raise embedded_model.EmbeddedLlamaError("simulated inference error")
        if self.responses:
            return self.responses.pop(0)
        return "updated prompt"

    def unload(self):
        self.unload_calls += 1


def make_test_session(initial_answer="A woman standing in a rainy alley."):
    cfg = embedded_model.ModelConfig(
        model_path="model.gguf",
        mmproj_path="mmproj.gguf",
        context_size=8192,
        gpu_layers=-1,
        threads=4,
        flash_attention=True,
        mmproj_on_gpu=True,
    )
    img_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    return nodes.VisionChatSession(
        model_config=cfg,
        unload_comfy_models_before=True,
        system_prompt="You are an expert prompt writer.",
        initial_prompt="Describe the image.",
        initial_answer=initial_answer,
        visual_parts=[img_part],
        messages=[
            {"role": "system", "content": "You are an expert prompt writer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image.\n\nMain image:"},
                    img_part,
                ],
            },
            {"role": "assistant", "content": initial_answer},
        ],
        thumbnail_uri="data:image/jpeg;base64,BBBB",
        max_tokens=512,
        temperature=0.25,
        top_p=0.9,
        top_k=40,
        seed=42,
        enable_thinking=False,
    )


def wait_for_status(node_id, target_status, timeout=3.0, min_turns=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = nodes.get_vision_chat_state(node_id)
        if st.get("status") == target_status:
            if min_turns is None or st.get("turn_count", 0) >= min_turns:
                return st
        time.sleep(0.02)
    raise AssertionError(
        f"Timed out waiting for node {node_id} status={target_status} (last={nodes.get_vision_chat_state(node_id)})"
    )


class VisionPromptChatGateTests(unittest.TestCase):
    def setUp(self):
        self.original_manager = nodes.MODEL_MANAGER
        with nodes._VISION_CHAT_LOCK:
            nodes._VISION_CHAT_SESSIONS.clear()
            nodes._LAST_VISION_CHAT_STATES.clear()

    def tearDown(self):
        nodes.MODEL_MANAGER = self.original_manager
        with nodes._VISION_CHAT_LOCK:
            nodes._VISION_CHAT_SESSIONS.clear()
            nodes._LAST_VISION_CHAT_STATES.clear()

    def test_bypass_passthrough_returns_immediately_and_unloads_vram(self):
        fake = FakeModelManager()
        nodes.MODEL_MANAGER = fake
        session = make_test_session("Initial prompt")

        gate = nodes.ZiggartGGUFVisionPromptChatGate()
        final_prompt, transcript, turn_count, out_session = gate.review(
            prompt_text="Initial prompt",
            pause_mode="bypass_passthrough",
            free_vram_on_continue=True,
            vision_session=session,
            unique_id="node_bypass",
        )

        self.assertEqual(final_prompt, "Initial prompt")
        self.assertIn("Initial prompt", transcript)
        self.assertEqual(turn_count, 0)
        self.assertIsInstance(out_session, nodes.VisionChatSession)
        self.assertEqual(fake.unload_calls, 1)
        self.assertEqual(nodes.get_vision_chat_state("node_bypass")["status"], "approved")

    def test_interactive_discuss_refine_manual_edit_undo_and_continue(self):
        fake = FakeModelManager(
            responses=[
                "Yes, there is a glowing red neon sign on the left brick wall.",
                "A woman standing in a rainy neon-lit alley with a glowing red neon sign on the left brick wall and silver rings on her fingers.",
            ]
        )
        nodes.MODEL_MANAGER = fake
        session = make_test_session("A woman standing in a rainy alley.")
        gate = nodes.ZiggartGGUFVisionPromptChatGate()

        result_holder = {}
        error_holder = {}

        def worker():
            try:
                result_holder["out"] = gate.review(
                    prompt_text="A woman standing in a rainy alley.",
                    pause_mode="always_pause",
                    free_vram_on_continue=True,
                    keep_full_history=True,
                    vision_session=session,
                    unique_id="node_chat_1",
                )
            except Exception as exc:
                error_holder["err"] = exc

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Wait for initial pause
        st = wait_for_status("node_chat_1", "waiting")
        self.assertTrue(st["can_reply"])
        self.assertEqual(st["active_version"], 1)
        self.assertEqual(st["thumbnail_uri"], "data:image/jpeg;base64,BBBB")

        # 1. Ask / Discuss (should NOT overwrite current_prompt)
        res = nodes.submit_vision_chat_action(
            "node_chat_1",
            "discuss",
            {"message": "Do you see a red neon sign on the left?"},
        )
        self.assertTrue(res["ok"])
        st = wait_for_status("node_chat_1", "waiting", min_turns=1)
        self.assertEqual(st["current_prompt"], "A woman standing in a rainy alley.")
        self.assertEqual(st["version_count"], 1)
        self.assertEqual(st["turn_count"], 1)

        # 2. Refine Prompt (should update current_prompt to v2 and inherit discussion context)
        res = nodes.submit_vision_chat_action(
            "node_chat_1",
            "refine",
            {"message": "Include that red neon sign and mention her silver rings."},
        )
        self.assertTrue(res["ok"])
        st = wait_for_status("node_chat_1", "waiting", min_turns=2)
        self.assertIn("silver rings", st["current_prompt"])
        self.assertEqual(st["version_count"], 2)
        self.assertEqual(st["active_version"], 2)
        self.assertTrue(st["can_undo"])
        # Verify full history carried the discuss turn into the refine call
        self.assertGreaterEqual(len(fake.calls[1][0]), 6)

        # 3. Manual edit -> v3
        res = nodes.submit_vision_chat_action(
            "node_chat_1",
            "update_prompt",
            {"prompt": "Manual experimental rewrite"},
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["state"]["active_version"], 3)
        self.assertEqual(res["state"]["current_prompt"], "Manual experimental rewrite")

        # 4. Undo -> back to v2
        res = nodes.submit_vision_chat_action("node_chat_1", "undo")
        self.assertTrue(res["ok"])
        self.assertEqual(res["state"]["active_version"], 2)
        self.assertIn("silver rings", res["state"]["current_prompt"])

        # 5. Restore v1, then back to v2
        res = nodes.submit_vision_chat_action("node_chat_1", "restore_version", {"version_index": 0})
        self.assertTrue(res["ok"])
        self.assertEqual(res["state"]["current_prompt"], "A woman standing in a rainy alley.")
        res = nodes.submit_vision_chat_action("node_chat_1", "restore_version", {"version_index": 1})
        self.assertTrue(res["ok"])

        # 6. Continue with a tiny final manual tweak
        final_tweaked = res["state"]["current_prompt"] + ", cinematic 35mm photography"
        res = nodes.submit_vision_chat_action(
            "node_chat_1",
            "continue",
            {"prompt": final_tweaked},
        )
        self.assertTrue(res["ok"])

        t.join(timeout=3.0)
        self.assertFalse(t.is_alive())
        self.assertNotIn("err", error_holder)

        final_prompt, transcript, turn_count, out_session = result_holder["out"]
        self.assertEqual(final_prompt, final_tweaked)
        self.assertEqual(turn_count, 2)
        self.assertIn("DISCUSS", transcript.upper())
        self.assertIn("REFINE", transcript.upper())
        self.assertIsInstance(out_session, nodes.VisionChatSession)
        self.assertEqual(fake.unload_calls, 1)
        self.assertNotIn("node_chat_1", nodes._VISION_CHAT_SESSIONS)
        self.assertEqual(nodes.get_vision_chat_state("node_chat_1")["status"], "approved")

    def test_cancel_aborts_cleanly_and_unloads_vram(self):
        fake = FakeModelManager()
        nodes.MODEL_MANAGER = fake
        session = make_test_session("Initial prompt")
        gate = nodes.ZiggartGGUFVisionPromptChatGate()
        error_holder = {}

        def worker():
            try:
                gate.review(
                    prompt_text="Initial prompt",
                    pause_mode="always_pause",
                    free_vram_on_continue=True,
                    vision_session=session,
                    unique_id="node_cancel",
                )
            except Exception as exc:
                error_holder["err"] = exc

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        wait_for_status("node_cancel", "waiting")

        res = nodes.submit_vision_chat_action("node_cancel", "cancel")
        self.assertTrue(res["ok"])
        t.join(timeout=3.0)
        self.assertFalse(t.is_alive())
        self.assertIn("err", error_holder)
        self.assertIn("cancelled", str(error_holder["err"]).lower())
        self.assertEqual(fake.unload_calls, 1)
        self.assertEqual(nodes.get_vision_chat_state("node_cancel")["status"], "cancelled")

    def test_stateless_mode_and_fallback_image_with_vlm_config(self):
        fake = FakeModelManager(responses=["Refined with fallback image"])
        nodes.MODEL_MANAGER = fake
        cfg = embedded_model.ModelConfig(
            model_path="model.gguf",
            mmproj_path="mmproj.gguf",
            context_size=8192,
            gpu_layers=-1,
            threads=4,
            flash_attention=True,
            mmproj_on_gpu=True,
        )
        img = np.zeros((1, 8, 8, 3), dtype=np.float32)
        gate = nodes.ZiggartGGUFVisionPromptChatGate()
        result_holder = {}

        def worker():
            result_holder["out"] = gate.review(
                prompt_text="Draft prompt",
                pause_mode="always_pause",
                free_vram_on_continue=False,
                keep_full_history=False,
                image=img,
                vlm_config=cfg,
                unique_id="node_fallback",
            )

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        wait_for_status("node_fallback", "waiting")

        nodes.submit_vision_chat_action("node_fallback", "refine", {"message": "Make it brighter"})
        wait_for_status("node_fallback", "waiting", min_turns=1)
        nodes.submit_vision_chat_action("node_fallback", "continue")
        t.join(timeout=3.0)

        # In stateless mode, the single user message should include the fresh image_url part
        sent_messages = fake.calls[0][0]
        self.assertEqual(len(sent_messages), 2)  # system + user(text + image)
        self.assertTrue(any(p.get("type") == "image_url" for p in sent_messages[1]["content"]))
        self.assertEqual(result_holder["out"][0], "Refined with fallback image")
        self.assertEqual(fake.unload_calls, 0)

    def test_vlm_error_recovers_to_waiting_and_manual_only_mode_blocks_reply(self):
        fake = FakeModelManager(responses=[], fail_once=True)
        nodes.MODEL_MANAGER = fake
        session = make_test_session("Safe prompt")
        gate = nodes.ZiggartGGUFVisionPromptChatGate()
        result_holder = {}

        def worker():
            result_holder["out"] = gate.review(
                prompt_text="Safe prompt",
                pause_mode="always_pause",
                free_vram_on_continue=True,
                vision_session=session,
                unique_id="node_err",
            )

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        wait_for_status("node_err", "waiting")

        nodes.submit_vision_chat_action("node_err", "refine", {"message": "Add rain"})
        # Wait until error is logged and status returns to waiting
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            st = nodes.get_vision_chat_state("node_err")
            if st["status"] == "waiting" and st["error"]:
                break
            time.sleep(0.02)
        st = nodes.get_vision_chat_state("node_err")
        self.assertIn("simulated inference error", st["error"])
        self.assertEqual(st["current_prompt"], "Safe prompt")

        nodes.submit_vision_chat_action("node_err", "continue")
        t.join(timeout=3.0)
        self.assertEqual(result_holder["out"][0], "Safe prompt")

        # Manual-only mode (no vision_session, no model)
        manual_holder = {}

        def manual_worker():
            manual_holder["out"] = gate.review(
                prompt_text="Manual only",
                pause_mode="always_pause",
                unique_id="node_manual",
            )

        t2 = threading.Thread(target=manual_worker, daemon=True)
        t2.start()
        st2 = wait_for_status("node_manual", "waiting")
        self.assertFalse(st2["can_reply"])
        res_bad = nodes.submit_vision_chat_action("node_manual", "refine", {"message": "Try reply"})
        self.assertFalse(res_bad["ok"])
        nodes.submit_vision_chat_action("node_manual", "continue", {"prompt": "Edited manually"})
        t2.join(timeout=3.0)
        self.assertEqual(manual_holder["out"][0], "Edited manually")


if __name__ == "__main__":
    unittest.main()
