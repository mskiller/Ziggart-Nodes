import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _load_package():
    if "ziggart_nodes_pkg" in sys.modules:
        return sys.modules["ziggart_nodes_pkg"]
    spec = importlib.util.spec_from_file_location(
        "ziggart_nodes_pkg",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ziggart_nodes_pkg"] = module
    spec.loader.exec_module(module)
    return module


pkg = _load_package()
embedded_model = sys.modules["ziggart_nodes_pkg.embedded_model"]
nodes = sys.modules["ziggart_nodes_pkg.nodes"]
wildcard_nodes = sys.modules["ziggart_nodes_pkg.wildcard_nodes"]
avatar_nodes = sys.modules["ziggart_nodes_pkg.avatar_nodes"]
three_d_nodes = sys.modules["ziggart_nodes_pkg.three_d_nodes"]


class _FakeHTTPResponse:
    def __init__(self, body_bytes: bytes, status: int = 200):
        self._buf = io.BytesIO(body_bytes)
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def readline(self) -> bytes:
        return self._buf.readline()

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        line = self._buf.readline()
        if not line:
            raise StopIteration
        return line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class VisionImprovementsAndHardeningTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # Improvement #1: JSON schema formatting, thinking tags, text-only & KV cache
    # ------------------------------------------------------------------

    def test_json_schema_response_format_and_openai_translation(self):
        schema = {
            "type": "object",
            "properties": {"revised_prompt": {"type": "string"}},
            "required": ["revised_prompt"],
        }
        rf = embedded_model.build_json_schema_response_format("PromptRevision", schema)
        self.assertEqual(rf["type"], "json_object")
        self.assertEqual(rf["schema"]["title"], "PromptRevision")

        oai = embedded_model._to_openai_response_format(rf)
        self.assertEqual(oai["type"], "json_schema")
        self.assertEqual(oai["json_schema"]["name"], "PromptRevision")
        self.assertEqual(oai["json_schema"]["schema"]["required"], ["revised_prompt"])

        fallback = embedded_model._to_openai_response_format(rf, fallback_only=True)
        self.assertEqual(fallback, {"type": "json_object"})

    def test_strip_thinking_tags_handles_truncated_unclosed_blocks(self):
        truncated = "<think>long unfinished chain of thought"
        self.assertEqual(embedded_model.strip_thinking_tags(truncated), "long unfinished chain of thought")

        with_answer = "<think>closed</think>Final caption<think>dangling"
        self.assertEqual(embedded_model.strip_thinking_tags(with_answer), "Final caption")

        leading_close = "orphan thought</think>Clean Answer"
        self.assertEqual(embedded_model.strip_thinking_tags(leading_close), "Clean Answer")

    def test_vlm_loader_supports_text_only_kv_cache_and_n_batch(self):
        loader = nodes.ZiggartGGUFVLMLoader()
        with mock.patch.object(nodes, "_resolve_gguf", side_effect=["C:/models/text.gguf", ""]):
            (cfg,) = loader.create_config(
                model="text.gguf",
                mmproj=nodes.NO_MMPROJ,
                context_size=4096,
                gpu_layers=-1,
                threads=4,
                flash_attention=True,
                mmproj_on_gpu=True,
                kv_cache_type="q8_0",
                n_batch=256,
            )
        self.assertEqual(cfg.model_path, "C:/models/text.gguf")
        self.assertEqual(cfg.mmproj_path, "")
        self.assertEqual(cfg.kv_cache_type, "q8_0")
        self.assertEqual(cfg.n_batch, 256)

    def test_caption_presets_and_verbosity_resolution(self):
        self.assertIn("krea2_qwen3vl_prose", nodes.VISION_CAPTION_PRESETS)
        sys_p, usr_p = nodes._resolve_caption_prompts(
            caption_preset="danbooru_pony_tags",
            verbosity="short",
            system_prompt="ignored",
            prompt="ignored",
        )
        self.assertIn("tag", (sys_p + usr_p).lower())
        self.assertIn("concise", usr_p.lower())

    # ------------------------------------------------------------------
    # Improvement #2: Streaming chat, Stop, Reset Context, Region-Pinning, Budget Trim
    # ------------------------------------------------------------------

    def test_focus_crop_part_and_message_budget_trimming(self):
        img = np.zeros((100, 120, 3), dtype=np.uint8)
        img[20:60, 30:90] = [200, 120, 80]
        parts = nodes._build_focus_crop_part(img, [0.25, 0.2, 0.75, 0.6])
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["type"], "text")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))

        # Tiny/invalid box returns empty
        self.assertEqual(nodes._build_focus_crop_part(img, [0.1, 0.1, 0.1, 0.1]), [])

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "initial prompt " * 20},
        ]
        for i in range(12):
            msgs.append({"role": "user", "content": f"follow up question {i} " + ("x" * 120)})
            msgs.append({"role": "assistant", "content": f"reply {i} " + ("y" * 120)})
        trimmed = nodes._trim_messages_to_budget(msgs, context_size=1024, max_tokens=256)
        self.assertLess(len(trimmed), len(msgs))
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertEqual(trimmed[1]["role"], "user")
        self.assertIn("follow up question 11", trimmed[-2]["content"])

    def test_chat_gate_reset_context_action(self):
        gate = nodes.ZiggartGGUFVisionPromptChatGate()
        (api_cfg,) = nodes.ZiggartOpenAIVLMLoader().create_config(
            api_base_url="http://127.0.0.1:8080/v1",
            model_name="test-vlm",
        )

        def driver():
            import time
            for _ in range(40):
                sess = nodes._VISION_CHAT_SESSIONS.get("gate_reset_1")
                if sess is not None:
                    sess.llm_messages.append({"role": "user", "content": "old turn"})
                    sess.llm_messages.append({"role": "assistant", "content": "old reply"})
                    sess.reset_context()
                    sess.action_queue.put({"action": "continue", "prompt": "final manual prompt"})
                    return
                time.sleep(0.02)

        t = threading.Thread(target=driver, daemon=True)
        t.start()
        res = gate.review(
            prompt_text="initial caption",
            pause_mode="always_pause",
            vlm_config=api_cfg,
            free_vram_on_continue=False,
            unique_id="gate_reset_1",
            timeout_seconds=5,
        )
        t.join(timeout=2)
        final_prompt, transcript, turn_count, out_sess = res
        self.assertEqual(final_prompt, "final manual prompt")
        self.assertEqual(turn_count, 0)
        self.assertIn("reset", transcript.lower())
        self.assertIsNotNone(out_sess)
        self.assertEqual(len(out_sess.messages), 3)
        self.assertNotIn("old turn", json.dumps(out_sess.messages))

    # ------------------------------------------------------------------
    # Improvement #3: OpenAI-Compatible Loader + Non-Streaming & SSE Streaming
    # ------------------------------------------------------------------

    def test_openai_vlm_loader_and_mocked_http_and_sse_streaming(self):
        loader = nodes.ZiggartOpenAIVLMLoader()
        with self.assertRaises(ValueError):
            loader.create_config(api_base_url="ftp://bad-host")

        (cfg,) = loader.create_config(
            api_base_url="http://127.0.0.1:8080/v1/",
            model_name="qwen3-vl-test",
            context_size=8192,
            api_key="sk-test",
            request_timeout=45.0,
        )
        self.assertEqual(cfg.backend, "openai_compatible")
        self.assertEqual(cfg.api_base_url, "http://127.0.0.1:8080/v1")

        mgr = embedded_model.EmbeddedModelManager()
        mgr.load(cfg, unload_comfy_models=False)
        self.assertTrue(mgr.is_loaded())

        # Non-streaming mock
        payload = json.dumps({
            "choices": [{"message": {"content": "<think>plan</think>Clean API caption"}}]
        }).encode("utf-8")
        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
            out = mgr.chat(
                [{"role": "user", "content": "describe"}],
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                seed=42,
                enable_thinking=False,
            )
        self.assertEqual(out, "Clean API caption")

        # SSE Streaming + early stop_event mock
        sse_lines = (
            b'data: {"choices":[{"delta":{"content":"Hello "}}]}\n'
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n'
            b'data: [DONE]\n'
        )
        deltas = []
        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(sse_lines)):
            streamed = mgr.chat_stream(
                [{"role": "user", "content": "hi"}],
                on_token=deltas.append,
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                seed=42,
                enable_thinking=False,
            )
        self.assertEqual(streamed, "Hello world")
        self.assertEqual(deltas, ["Hello ", "world"])

        stop_evt = threading.Event()
        deltas2 = []

        def on_delta_stop(chunk):
            deltas2.append(chunk)
            stop_evt.set()

        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(sse_lines)):
            partial = mgr.chat_stream(
                [{"role": "user", "content": "hi"}],
                on_token=on_delta_stop,
                stop_event=stop_evt,
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                seed=42,
                enable_thinking=False,
            )
        self.assertEqual(deltas2, ["Hello "])
        self.assertEqual(partial, "Hello")

    # ------------------------------------------------------------------
    # Improvement #4: Batch Folder Vision Captioner + Resume Manifest
    # ------------------------------------------------------------------

    def test_batch_folder_captioner_single_load_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (32, 32), (200, 100, 50)).save(root / "img_01.png")
            Image.new("RGB", (32, 32), (50, 120, 200)).save(root / "img_02.jpg")

            (api_cfg,) = nodes.ZiggartOpenAIVLMLoader().create_config(
                api_base_url="http://127.0.0.1:8080/v1",
                model_name="qwen3-vl",
            )
            captioner = nodes.ZiggartBatchFolderVisionCaptioner()
            with mock.patch.object(nodes.MODEL_MANAGER, "load") as m_load, \
                 mock.patch.object(nodes.MODEL_MANAGER, "unload"), \
                 mock.patch.object(
                     nodes.MODEL_MANAGER,
                     "chat_batch",
                     return_value=["A sunset portrait of a warrior.", "A serene blue ocean vista."],
                 ) as m_batch:
                master_path, joined, processed, skipped, report_json = captioner.caption_folder(
                    folder_path=str(root),
                    model=nodes.NO_MODEL,
                    mmproj=nodes.NO_MMPROJ,
                    vlm_config=api_cfg,
                    caption_preset="krea2_qwen3vl_prose",
                    verbosity="medium",
                    skip_existing_txt=True,
                    save_sidecar_txt=True,
                    save_master_txt=True,
                    master_filename="captions.txt",
                    free_vram_after=True,
                )

                self.assertEqual(m_load.call_count, 1)
                self.assertEqual(m_batch.call_count, 1)
                self.assertEqual(processed, 2)
                self.assertEqual(skipped, 0)
                self.assertIn("warrior", joined)
                self.assertTrue((root / "img_01.txt").is_file())
                self.assertTrue((root / "img_02.txt").is_file())
                self.assertTrue(Path(master_path).is_file())
                manifest_path = root / ".ziggart_batch_manifest.json"
                self.assertTrue(manifest_path.is_file())
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest.get("processed_count"), 2)

                # Resume run skips existing sidecars without calling chat_batch again
                _, _, proc2, skip2, _ = captioner.caption_folder(
                    folder_path=str(root),
                    model=nodes.NO_MODEL,
                    mmproj=nodes.NO_MMPROJ,
                    vlm_config=api_cfg,
                    skip_existing_txt=True,
                )
                self.assertEqual(proc2, 0)
                self.assertEqual(skip2, 2)
                self.assertEqual(m_batch.call_count, 1)

    # ------------------------------------------------------------------
    # Security & Path Traversal Hardening
    # ------------------------------------------------------------------

    def test_path_traversal_rejections_across_modules(self):
        self.assertEqual(wildcard_nodes.normalize_wildcard_key("../../etc/passwd"), "")
        self.assertEqual(wildcard_nodes.normalize_wildcard_key("C:\\secret\\file"), "")
        self.assertEqual(wildcard_nodes.normalize_wildcard_key("/leading/slash"), "")
        self.assertFalse(wildcard_nodes._is_allowed_wildcard_dir(tempfile.gettempdir()))

        saver = wildcard_nodes.ZiggartWildcardRecipeSaver()
        import torch
        dummy_img = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                saver.save_recipe(
                    image=dummy_img,
                    filename_prefix="../../escape",
                    output_dir_override=tmp,
                )

        self.assertFalse(avatar_nodes._is_allowed_avatar_model_path(Path(tempfile.gettempdir()) / "evil.vrm"))
        with self.assertRaises(ValueError):
            three_d_nodes._next_output_path("../../escape", "glb")

    # ------------------------------------------------------------------
    # Merged __init__.py & Full 64-Node / 18-Workflow Integrity
    # ------------------------------------------------------------------

    def test_merged_init_and_all_workflows_schema_valid(self):
        self.assertEqual(len(pkg.NODE_CLASS_MAPPINGS), 64)
        self.assertEqual(set(pkg.NODE_CLASS_MAPPINGS.keys()), set(pkg.NODE_DISPLAY_NAME_MAPPINGS.keys()))

        wf_dir = ROOT / "example_workflows"
        wf_files = sorted(wf_dir.glob("*.json"))
        self.assertEqual(len(wf_files), 18)

        seen_types = set()
        widget_types = {"INT", "FLOAT", "STRING", "BOOLEAN"}

        for wf_path in wf_files:
            data = json.loads(wf_path.read_text(encoding="utf-8"))
            wf_nodes = data.get("nodes", [])
            wf_links = data.get("links", [])
            max_nid = max((int(n["id"]) for n in wf_nodes if isinstance(n.get("id"), int)), default=0)
            max_lid = max((int(lnk[0]) for lnk in wf_links if isinstance(lnk, list) and lnk), default=0)
            self.assertGreaterEqual(data.get("last_node_id", 0), max_nid, wf_path.name)
            self.assertGreaterEqual(data.get("last_link_id", 0), max_lid, wf_path.name)

            nmap = {n["id"]: n for n in wf_nodes}
            for lnk in wf_links:
                _, oid, oslot, iid, islot, _ = lnk
                self.assertIn(oid, nmap, wf_path.name)
                self.assertIn(iid, nmap, wf_path.name)
                self.assertLess(oslot, len(nmap[oid].get("outputs", [])), wf_path.name)
                self.assertLess(islot, len(nmap[iid].get("inputs", [])), wf_path.name)

            for n in wf_nodes:
                cls_name = n.get("type")
                if cls_name not in pkg.NODE_CLASS_MAPPINGS:
                    continue
                seen_types.add(cls_name)
                cls = pkg.NODE_CLASS_MAPPINGS[cls_name]
                ret = getattr(cls, "RETURN_TYPES", ())
                self.assertEqual(len(n.get("outputs", [])), len(ret), f"{wf_path.name}:{cls_name}")

                expected = []
                it = cls.INPUT_TYPES()
                for sec in ("required", "optional"):
                    for k, spec in (it.get(sec) or {}).items():
                        t0 = spec[0] if isinstance(spec, (tuple, list)) and spec else None
                        cfg = spec[1] if isinstance(spec, (tuple, list)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
                        if isinstance(t0, (list, tuple)) or (isinstance(t0, str) and t0 in widget_types):
                            expected.append((k, t0, cfg))
                            if k in ("seed", "noise_seed"):
                                expected.append(("control_after_generate", ["fixed", "increment", "decrement", "randomize"], {}))

                wv = n.get("widgets_values", [])
                self.assertEqual(len(wv), len(expected), f"{wf_path.name}:{cls_name}")
                for val, (k, t0, cfg) in zip(wv, expected):
                    if isinstance(t0, (list, tuple)) and t0:
                        self.assertIn(val, t0, f"{wf_path.name}:{cls_name}.{k}")
                    elif t0 in ("INT", "FLOAT"):
                        if "min" in cfg:
                            self.assertGreaterEqual(val, cfg["min"], f"{wf_path.name}:{cls_name}.{k}")
                        if "max" in cfg:
                            self.assertLessEqual(val, cfg["max"], f"{wf_path.name}:{cls_name}.{k}")

        self.assertEqual(seen_types, set(pkg.NODE_CLASS_MAPPINGS.keys()))

    # ------------------------------------------------------------------
    # Multi-Architecture GGUF, Gemma4/Qwen35/Inkling2 Handlers & llama-server
    # ------------------------------------------------------------------

    def test_read_gguf_metadata_and_missing_files(self):
        import struct

        def _pack_str(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "not_gguf.bin"
            bad.write_bytes(b"NOPE1234567890")
            self.assertEqual(embedded_model.read_gguf_metadata(str(bad)), {})
            self.assertEqual(embedded_model.read_gguf_metadata(str(Path(tmp) / "missing.gguf")), {})

            gguf_path = Path(tmp) / "test_inkling2.gguf"
            kvs = [
                ("general.architecture", 8, _pack_str("inkling2")),
                ("general.name", 8, _pack_str("Inkling-2-Mini")),
                ("tokenizer.chat_template", 8, _pack_str("{{ messages }}")),
                ("clip.has_vision_encoder", 7, b"\x01"),
            ]
            blob = bytearray(b"GGUF")
            blob.extend(struct.pack("<IQQ", 3, 0, len(kvs)))
            for k, vt, raw_val in kvs:
                blob.extend(_pack_str(k))
                blob.extend(struct.pack("<I", vt))
                blob.extend(raw_val)
            gguf_path.write_bytes(bytes(blob))

            meta = embedded_model.read_gguf_metadata(str(gguf_path))
            self.assertEqual(meta.get("gguf_version"), 3)
            self.assertEqual(meta.get("general.architecture"), "inkling2")
            self.assertEqual(meta.get("general.name"), "Inkling-2-Mini")
            self.assertTrue(meta.get("has_chat_template"))
            self.assertTrue(meta.get("clip.has_vision_encoder"))

    def test_strip_thinking_tags_handles_gemma4_and_thought_variants(self):
        self.assertEqual(
            embedded_model.strip_thinking_tags("<|channel>thought\n<channel|>The square is red."),
            "The square is red.",
        )
        self.assertEqual(
            embedded_model.strip_thinking_tags("<|channel>thought\nLet me look closely.<channel|>A blue vase."),
            "A blue vase.",
        )
        self.assertEqual(
            embedded_model.strip_thinking_tags("<thought>reasoning here</thought>\nClean final prompt."),
            "Clean final prompt.",
        )
        # Empty marker-only output returns empty string cleanly
        self.assertEqual(
            embedded_model.strip_thinking_tags("<|channel>thought\n<channel|>"),
            "",
        )

    def test_architecture_handler_selection_and_enable_thinking_sync(self):
        import types

        class _FakeHandlerBase:
            def __init__(self, mmproj_path=None, verbose=True, use_gpu=True, **kwargs):
                self.mmproj_path = mmproj_path
                self.use_gpu = use_gpu
                self.extra_template_arguments = {}
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append((kwargs.get("enable_thinking"), dict(self.extra_template_arguments)))
                return {"choices": [{"message": {"content": "ok"}}]}

        class Gemma4ChatHandler(_FakeHandlerBase):
            def __init__(self, enable_thinking=True, **kwargs):
                self.enable_thinking = enable_thinking
                super().__init__(**kwargs)

        class Qwen35ChatHandler(_FakeHandlerBase):
            def __init__(self, enable_thinking=True, **kwargs):
                self.enable_thinking = enable_thinking
                super().__init__(**kwargs)

        class Qwen3VLChatHandler(_FakeHandlerBase):
            pass

        class GenericMTMDChatHandler(_FakeHandlerBase):
            def __init__(self, chat_format=None, mmproj_path="", verbose=True, **kwargs):
                self.chat_format = chat_format
                super().__init__(mmproj_path=mmproj_path, verbose=verbose, **kwargs)

        class MTMDChatHandler(_FakeHandlerBase):
            pass

        fake_cf = types.SimpleNamespace(
            Gemma4ChatHandler=Gemma4ChatHandler,
            Qwen35ChatHandler=Qwen35ChatHandler,
            Qwen3VLChatHandler=Qwen3VLChatHandler,
            GenericMTMDChatHandler=GenericMTMDChatHandler,
            MTMDChatHandler=MTMDChatHandler,
        )
        fake_mm = types.SimpleNamespace()
        fake_pkg = types.SimpleNamespace(
            llama_chat_format=fake_cf,
            llama_multimodal=fake_mm,
        )

        with mock.patch.dict(
            sys.modules,
            {
                "llama_cpp": fake_pkg,
                "llama_cpp.llama_chat_format": fake_cf,
                "llama_cpp.llama_multimodal": fake_mm,
            },
        ):
            h_gemma, name_gemma = embedded_model._create_mtmd_chat_handler(
                mmproj_path="g.mmproj.gguf",
                arch="gemma4",
                has_chat_template=True,
                use_gpu=True,
            )
            self.assertEqual(name_gemma, "Gemma4ChatHandler")
            self.assertIsInstance(h_gemma, Gemma4ChatHandler)

            h_q35, name_q35 = embedded_model._create_mtmd_chat_handler(
                mmproj_path="ornith.mmproj.gguf",
                arch="qwen35",
                has_chat_template=True,
                use_gpu=True,
            )
            self.assertEqual(name_q35, "Qwen35ChatHandler")
            self.assertIsInstance(h_q35, Qwen35ChatHandler)

            h_ink, name_ink = embedded_model._create_mtmd_chat_handler(
                mmproj_path="inkling.mmproj.gguf",
                arch="inkling2_loop",
                has_chat_template=True,
                use_gpu=True,
            )
            self.assertEqual(name_ink, "GenericMTMDChatHandler")
            self.assertIsNone(h_ink.chat_format)

            h_leg, name_leg = embedded_model._create_mtmd_chat_handler(
                mmproj_path="legacy.mmproj.gguf",
                arch="",
                has_chat_template=False,
                use_gpu=True,
            )
            self.assertEqual(name_leg, "MTMDChatHandler")

            # Verify live enable_thinking propagation onto handler instance & extra_template_arguments
            mgr = embedded_model.EmbeddedModelManager()
            mgr._model = object()
            mgr._handler = h_gemma
            mgr._invoke_local_completion(
                [{"role": "user", "content": "hi"}],
                max_tokens=32,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                seed=-1,
                enable_thinking=True,
                response_format=None,
                stream=False,
            )
            self.assertTrue(h_gemma.enable_thinking)
            self.assertEqual(h_gemma.calls[-1], (True, {"enable_thinking": True}))

    def test_auto_fallback_to_managed_llama_server_and_clean_unload(self):
        import types

        loader = nodes.ZiggartGGUFVLMLoader()
        with mock.patch.object(nodes, "_resolve_gguf", side_effect=["C:/models/inkling2.gguf", "C:/models/mmproj.gguf"]):
            (cfg,) = loader.create_config(
                model="inkling2.gguf",
                mmproj="mmproj.gguf",
                backend="auto",
                llama_server_path="C:/fake/llama-server.exe",
                auto_download_llama_server=False,
                image_min_tokens=1024,
            )
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.llama_server_path, "C:/fake/llama-server.exe")
        self.assertEqual(cfg.image_min_tokens, 1024)

        fake_llama_mod = types.SimpleNamespace(
            __version__="0.4.7+cu130",
            Llama=mock.Mock(side_effect=ValueError("Failed to load model from file")),
        )
        fake_proc = mock.MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.stderr = io.BytesIO(b"")

        chat_payload = json.dumps({"choices": [{"message": {"content": "Red square from Inkling-2-Mini"}}]}).encode("utf-8")

        mgr = embedded_model.EmbeddedModelManager()
        with mock.patch.dict(sys.modules, {"llama_cpp": fake_llama_mod}), \
             mock.patch.object(embedded_model, "read_gguf_metadata", return_value={"general.architecture": "inkling2", "has_chat_template": True}), \
             mock.patch.object(embedded_model, "_create_mtmd_chat_handler", return_value=(object(), "GenericMTMDChatHandler")), \
             mock.patch.object(embedded_model, "find_llama_server_binary", return_value="C:/fake/llama-server.exe"), \
             mock.patch("subprocess.Popen", return_value=fake_proc) as m_popen, \
             mock.patch("urllib.request.urlopen", side_effect=[_FakeHTTPResponse(b'{"status":"ok"}', status=200), _FakeHTTPResponse(chat_payload, status=200)]):
            mgr.load(cfg, unload_comfy_models=False)
            self.assertTrue(mgr.is_loaded())
            self.assertTrue(mgr._server_base_url.startswith("http://127.0.0.1:"))
            cmd = m_popen.call_args[0][0]
            self.assertIn("--jinja", cmd)
            self.assertIn("--image-min-tokens", cmd)
            self.assertIn("1024", cmd)

            out = mgr.chat(
                [{"role": "user", "content": "describe"}],
                max_tokens=64,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                seed=1,
                enable_thinking=False,
            )
            self.assertEqual(out, "Red square from Inkling-2-Mini")

            mgr.unload()
            self.assertFalse(mgr.is_loaded())
            fake_proc.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
