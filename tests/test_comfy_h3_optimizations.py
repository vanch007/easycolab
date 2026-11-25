import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PY_PATH = os.path.join(REPO_ROOT, "app.py")
IPYNB_PATH = os.path.join(REPO_ROOT, "EasyColab_Server.ipynb")


def _load_app_comfy_helpers() -> Dict[str, Any]:
    with open(APP_PY_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=APP_PY_PATH)

    wanted_assigns = {
        "COMFY_STANDARD_MODEL_SUBDIRS",
        "_COMFY_PROTECTED_PIP_PKGS",
        "COMFY_PRESET_WORKFLOW_PACKS",
        "MODEL_CATALOG",
    }
    wanted_funcs = {
        "_make_zh_guide_note",
        "_prune_workflow_dangling_links",
        "_build_10eros_max_workflow",
        "_build_sulphur3_workflow",
        "_build_ref2va_tara_workflow",
        "_build_seedvr2_upscaler_workflow",
        "_build_recavsr_upscaler_workflow",
        "_build_moss_image_workflow",
        "_build_moss_image_api_prompt_for_request",
        "_build_qwen_image_edit_2511_workflow",
        "_build_qwen_image_edit_2511_api_prompt_for_request",
        "_snap_wan_frames",
        "_build_wan22_remix_workflow",
        "_build_wan22_remix_api_prompt_for_request",
        "_build_h3_drama_director_workflow",
        "_build_h3_drama_director_api_prompt_for_request",
        "_build_qwen3_tts_workflow",
        "_build_drama3_tts_workflow",
        "_build_qwen3_tts_api_prompt_for_request",
        "_build_drama3_tts_api_prompt_for_request",
        "_extract_comfy_prompt_metadata",
        "_build_and_install_comfy_preset_workflows",
        "_get_comfy_preset_packs_status",
        "_safe_symlink_dir",
        "_get_comfy_nvme_models_root",
        "_build_comfyui_launch_cmd",
        "_wire_comfyui_hybrid_models_dir",
        "_stage_comfy_file_to_nvme",
        "_custom_nodes_fingerprint",
        "_build_custom_nodes_bundle_archive",
        "_restore_custom_nodes_from_bundle",
        "_pip_install_cmd",
        "_install_comfy_pack_custom_nodes",
        "build_structured_prompt",
        "_snap_h3_frames",
        "_snap_seedvr2_batch_size",
        "_build_comfy_h3_api_prompt_for_request",
        "_build_seedvr2_api_prompt_for_request",
        "_build_recavsr_api_prompt_for_request",
    }

    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_assigns:
                    selected.append(node)
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in wanted_assigns:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            selected.append(node)

    mod = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod)

    class _DummyRedirect(urllib.request.HTTPRedirectHandler):
        pass

    ns: Dict[str, Any] = {
        "os": os,
        "re": re,
        "sys": sys,
        "json": json,
        "shutil": shutil,
        "subprocess": subprocess,
        "tarfile": tarfile,
        "zipfile": zipfile,
        "urllib": urllib,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "MINIMAX_H3_BF16_FILES": [],
        "GenerateVideoRequest": SimpleNamespace,
        "COMFY_TEMPLATE_RAW_BASE": "https://example.invalid/",
        "_COMFY_STATUS_CACHE": {"ts": 0.0, "counts": (0, 0, 0), "outputs": [], "packs": []},
        "_StripAuthRedirectHandler": _DummyRedirect,
        "_flush_drive_file": lambda p: None,
        "_fetch_official_comfy_template": lambda fn: {},
        "_resolve_comfyui_persistence_paths": lambda: ("/tmp/comfy", "/tmp/persist", "local_only", "local", False),
    }
    exec(compile(mod, APP_PY_PATH, "exec"), ns)
    return ns


class ComfyH3OptimizationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_app_comfy_helpers()

    def test_h3_correctors_subdir_and_uncensored_packs_registered(self):
        subdirs = self.ns["COMFY_STANDARD_MODEL_SUBDIRS"]
        self.assertIn("h3_correctors", subdirs)

        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        self.assertIn("minimax_h3_10eros_max", packs)
        self.assertIn("minimax_h3_sulphur3", packs)

        eros = packs["minimax_h3_10eros_max"]
        eros_models = {(m[0], m[1], m[2]) for m in eros["models"]}
        self.assertIn(
            ("diffusion_models", "10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors", "TenStrip/10Eros-Max"),
            eros_models,
        )
        self.assertIn(
            ("text_encoders", "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors", "Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4"),
            eros_models,
        )

        s3 = packs["minimax_h3_sulphur3"]
        s3_models = {(m[0], m[1], m[2]) for m in s3["models"]}
        self.assertIn(("diffusion_models", "s3.safetensors", "SulphurAI/Sulphur-3"), s3_models)
        self.assertIn(("h3_correctors", "s3_corrector.safetensors", "SulphurAI/Sulphur-3"), s3_models)
        self.assertEqual(len(s3.get("custom_nodes", [])), 1)
        self.assertEqual(s3["custom_nodes"][0]["name"], "ComfyUI-H3-Corrector")
        self.assertIn("S3_ComfyUI.zip", s3["custom_nodes"][0]["url"])

    def test_offline_workflows_for_10eros_max_and_sulphur3(self):
        with tempfile.TemporaryDirectory() as tmp:
            installed = self.ns["_build_and_install_comfy_preset_workflows"](
                tmp, ["minimax_h3_10eros_max", "minimax_h3_sulphur3"]
            )
            self.assertEqual(
                installed,
                [
                    "07_10Eros-Max_无限制多图参考有声视频(8步极速).json",
                    "08_10Eros-Max_无限制首尾帧图生有声视频(8步极速).json",
                    "09_Sulphur-3_无限制文生有声视频(含H3修正器).json",
                    "10_Sulphur-3_无限制图生首尾帧有声视频(含H3修正器).json",
                ],
            )
            wf_dir = os.path.join(tmp, "user", "default", "workflows")
            self.assertTrue(os.path.isfile(os.path.join(wf_dir, ".easycolab_wf_schema_v3")))

            def _assert_no_dangling_links(wf_obj: Dict[str, Any]):
                node_ids = {n["id"] for n in wf_obj["nodes"]}
                link_ids = {lk[0] for lk in wf_obj["links"]}
                for lk in wf_obj["links"]:
                    self.assertIn(lk[1], node_ids)
                    self.assertIn(lk[3], node_ids)
                for n in wf_obj["nodes"]:
                    for inp in n.get("inputs", []):
                        if inp.get("link") is not None:
                            self.assertIn(inp["link"], link_ids)

            with open(os.path.join(wf_dir, installed[0]), "r", encoding="utf-8") as f:
                wf_multi = json.load(f)
            _assert_no_dangling_links(wf_multi)
            multi_nodes = {n["type"]: n for n in wf_multi["nodes"]}
            self.assertIn("MiniMaxH3ReferenceToVideo", multi_nodes)
            ref_inputs = {i["name"]: i for i in multi_nodes["MiniMaxH3ReferenceToVideo"]["inputs"]}
            self.assertIn("ref_images.ref_image_0", ref_inputs)
            self.assertIn("ref_images.ref_image_1", ref_inputs)
            self.assertIsNotNone(ref_inputs["ref_images.ref_image_0"]["link"])
            self.assertIsNotNone(ref_inputs["ref_images.ref_image_1"]["link"])
            self.assertEqual(multi_nodes["KSamplerSelect"]["widgets_values"], ["lcm"])

            with open(os.path.join(wf_dir, installed[1]), "r", encoding="utf-8") as f:
                wf_flf = json.load(f)
            _assert_no_dangling_links(wf_flf)
            flf_nodes = {n["type"]: n for n in wf_flf["nodes"]}
            self.assertIn("MiniMaxH3ImageToVideo", flf_nodes)
            self.assertEqual(flf_nodes["UNETLoader"]["widgets_values"][0], "10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors")

            with open(os.path.join(wf_dir, installed[2]), "r", encoding="utf-8") as f:
                wf_s3_t2v = json.load(f)
            _assert_no_dangling_links(wf_s3_t2v)
            s3_nodes = {n["type"]: n for n in wf_s3_t2v["nodes"]}
            self.assertIn("LoadH3Corrector", s3_nodes)
            self.assertIn("ApplyH3Corrector", s3_nodes)
            self.assertEqual(s3_nodes["LoadH3Corrector"]["widgets_values"], ["s3_corrector.safetensors"])
            self.assertEqual(s3_nodes["ApplyH3Corrector"]["widgets_values"], [1.0, 0.5, 0.001, 0, -1])
            self.assertEqual(s3_nodes["CFGGuider"]["widgets_values"], [1.0])
            self.assertEqual(s3_nodes["MiniMaxH3SigmaShift"]["widgets_values"], [12, 3])
            self.assertEqual(s3_nodes["UNETLoader"]["widgets_values"][0], "s3.safetensors")

            with open(os.path.join(wf_dir, installed[3]), "r", encoding="utf-8") as f:
                wf_s3_i2v = json.load(f)
            _assert_no_dangling_links(wf_s3_i2v)
            s3_i2v_types = {n["type"] for n in wf_s3_i2v["nodes"]}
            self.assertIn("MiniMaxH3ImageToVideo", s3_i2v_types)
            self.assertIn("ApplyH3Corrector", s3_i2v_types)

    def test_build_comfyui_launch_cmd_flags(self):
        fn = self.ns["_build_comfyui_launch_cmd"]
        g4_cmd = fn("/content/ComfyUI", 28188, "drive_symlink", 96.0)
        self.assertIn("--highvram", g4_cmd)
        self.assertIn("--disable-mmap", g4_cmd)
        self.assertIn("--fast", g4_cmd)

        l4_fuse_cmd = fn("/content/ComfyUI", 28188, "drive_symlink", 24.0)
        self.assertNotIn("--highvram", l4_fuse_cmd)
        self.assertIn("--disable-mmap", l4_fuse_cmd)
        self.assertNotIn("--fast", l4_fuse_cmd)

        l4_local_cmd = fn("/content/ComfyUI", 28188, "local_only", 24.0)
        self.assertNotIn("--highvram", l4_local_cmd)
        self.assertNotIn("--disable-mmap", l4_local_cmd)

    def test_hybrid_models_dir_and_nvme_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            comfy_root = os.path.join(tmp, "ComfyUI")
            persist_root = os.path.join(tmp, "drive", "EasyColab_ComfyUI")
            nvme_root = os.path.join(tmp, "fast_comfy_models")
            os.makedirs(comfy_root, exist_ok=True)

            p_diff = os.path.join(persist_root, "models", "diffusion_models")
            os.makedirs(p_diff, exist_ok=True)
            model_a = os.path.join(p_diff, "s3.safetensors")
            model_b = os.path.join(p_diff, "other.safetensors")
            with open(model_a, "wb") as f:
                f.write(b"A" * 4096)
            with open(model_b, "wb") as f:
                f.write(b"B" * 4096)

            staged = self.ns["_stage_comfy_file_to_nvme"](
                model_a, "diffusion_models", "s3.safetensors", nvme_root, min_free_gb=0.001
            )
            self.assertTrue(staged.startswith(nvme_root))
            self.assertTrue(os.path.isfile(staged))

            # Respect low-space guard (huge min_free_gb forces fallback to persist_file)
            fallback = self.ns["_stage_comfy_file_to_nvme"](
                model_b, "diffusion_models", "other.safetensors", nvme_root, min_free_gb=10_000_000.0
            )
            self.assertEqual(fallback, model_b)

            self.ns["_wire_comfyui_hybrid_models_dir"](comfy_root, persist_root, nvme_root)
            link_a = os.path.join(comfy_root, "models", "diffusion_models", "s3.safetensors")
            link_b = os.path.join(comfy_root, "models", "diffusion_models", "other.safetensors")
            self.assertTrue(os.path.islink(link_a))
            self.assertEqual(os.path.realpath(link_a), os.path.realpath(staged))
            self.assertTrue(os.path.islink(link_b))
            self.assertEqual(os.path.realpath(link_b), os.path.realpath(model_b))

    def test_custom_nodes_bundle_pack_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            cn_local = os.path.join(tmp, "ComfyUI", "custom_nodes")
            persist_root = os.path.join(tmp, "persist")
            os.makedirs(os.path.join(cn_local, "MyNode", ".git"), exist_ok=True)
            os.makedirs(os.path.join(cn_local, "MyNode", "__pycache__"), exist_ok=True)
            os.makedirs(os.path.join(cn_local, "easycolab_preset_helper"), exist_ok=True)

            with open(os.path.join(cn_local, "MyNode", "__init__.py"), "w", encoding="utf-8") as f:
                f.write("# node\n")
            with open(os.path.join(cn_local, "MyNode", ".git", "config"), "w", encoding="utf-8") as f:
                f.write("[core]\n")
            with open(os.path.join(cn_local, "MyNode", "__pycache__", "a.pyc"), "wb") as f:
                f.write(b"PYC")
            with open(os.path.join(cn_local, "easycolab_preset_helper", "__init__.py"), "w", encoding="utf-8") as f:
                f.write("# helper\n")

            ok = self.ns["_build_custom_nodes_bundle_archive"](cn_local, persist_root)
            self.assertTrue(ok)
            bundle = os.path.join(persist_root, "custom_nodes_bundle.tar.gz")
            self.assertTrue(os.path.isfile(bundle))

            # Append a malicious traversal entry to test safety on restore
            with tarfile.open(bundle, "r:gz") as rf:
                members = rf.getnames()
            self.assertIn("MyNode/__init__.py", members)
            self.assertFalse(any(".git" in m or "__pycache__" in m or "easycolab_preset_helper" in m for m in members))

            cn_restore = os.path.join(tmp, "ComfyUI_Restored", "custom_nodes")
            cnt = self.ns["_restore_custom_nodes_from_bundle"](cn_restore, persist_root)
            self.assertEqual(cnt, 1)
            self.assertTrue(os.path.isfile(os.path.join(cn_restore, "MyNode", "__init__.py")))

    def test_install_comfy_pack_custom_nodes_zip_safety_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            comfy_root = os.path.join(tmp, "ComfyUI")
            persist_root = os.path.join(tmp, "persist")
            os.makedirs(comfy_root, exist_ok=True)

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("ComfyUI-H3-Corrector/__init__.py", "NODE_CLASS_MAPPINGS = {}\n")
                zf.writestr("ComfyUI-H3-Corrector/corrector.py", "class H3Corrector: pass\n")
                zf.writestr("__MACOSX/ComfyUI-H3-Corrector/._init", "junk")
                zf.writestr("../../evil.txt", "pwned")
            zip_bytes = buf.getvalue()

            class _FakeResp:
                def __init__(self, raw: bytes):
                    self._bio = io.BytesIO(raw)

                def read(self, n: int = -1) -> bytes:
                    return self._bio.read(n)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            fake_opener = mock.MagicMock()
            fake_opener.open.return_value = _FakeResp(zip_bytes)

            spec = [
                {
                    "type": "zip",
                    "name": "ComfyUI-H3-Corrector",
                    "url": "https://huggingface.co/SulphurAI/Sulphur-3/resolve/main/S3_ComfyUI.zip",
                    "subdir": "ComfyUI-H3-Corrector",
                }
            ]
            with mock.patch("urllib.request.build_opener", return_value=fake_opener):
                installed_first = self.ns["_install_comfy_pack_custom_nodes"](comfy_root, persist_root, spec)
                installed_second = self.ns["_install_comfy_pack_custom_nodes"](comfy_root, persist_root, spec)

            self.assertTrue(installed_first)
            self.assertFalse(installed_second)
            self.assertTrue(os.path.isfile(os.path.join(comfy_root, "custom_nodes", "ComfyUI-H3-Corrector", "__init__.py")))
            self.assertTrue(os.path.isfile(os.path.join(persist_root, "custom_nodes", "ComfyUI-H3-Corrector", "corrector.py")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "evil.txt")))

    def test_notebook_cell5_matches_app_py_and_clients_have_new_packs(self):
        with open(APP_PY_PATH, "r", encoding="utf-8") as f:
            app_text = f.read()
        with open(IPYNB_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)
        self.assertEqual("".join(nb["cells"][5]["source"]), app_text)

        client_files = [
            os.path.join(REPO_ROOT, "macos_app", "Sources", "ModelsAndCatalogs.swift"),
            os.path.join(REPO_ROOT, "windows_app", "models.py"),
            os.path.join(REPO_ROOT, "ios_app", "Sources", "LocalizationAndModels.swift"),
            os.path.join(REPO_ROOT, "android_app", "app", "src", "main", "java", "app", "easycolab", "studio", "LocalizationAndModels.kt"),
        ]
        for cf in client_files:
            with open(cf, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("minimax_h3_10eros_max", content, f"missing 10Eros-Max in {cf}")
            self.assertIn("minimax_h3_sulphur3", content, f"missing Sulphur-3 in {cf}")
            self.assertIn("moss_image_edit", content, f"missing MOSS-Image Edit in {cf}")

    def test_native_model_catalog_has_10eros_and_sulphur3(self):
        catalog = self.ns["MODEL_CATALOG"]
        self.assertIn("minimax_h3_10eros_max", catalog)
        self.assertIn("minimax_h3_sulphur3", catalog)
        self.assertEqual(catalog["minimax_h3_10eros_max"]["pipeline_arch"], "comfy_h3_10eros_max")
        self.assertEqual(catalog["minimax_h3_10eros_max"]["comfy_pack_id"], "minimax_h3_10eros_max")
        self.assertEqual(catalog["minimax_h3_sulphur3"]["pipeline_arch"], "comfy_h3_sulphur3")
        self.assertEqual(catalog["minimax_h3_sulphur3"]["comfy_pack_id"], "minimax_h3_sulphur3")

        snap = self.ns["_snap_h3_frames"]
        self.assertEqual(snap(56), 56)
        self.assertEqual(snap(120), 124)
        self.assertEqual(snap(124), 124)
        self.assertEqual(snap(243), 243)

    def test_build_comfy_h3_api_prompt_10eros_multiref_and_fl2va(self):
        builder = self.ns["_build_comfy_h3_api_prompt_for_request"]
        req_multi = SimpleNamespace(
            task_mode="multiref",
            visual_prompt="A woman from <Picture 1> in room <Picture 2> wearing <Picture 3>",
            soundscape_prompt="Speaking in <Audio 1>'s voice",
            music_prompt="Ambient piano",
            negative_prompt="",
            width=1056,
            height=608,
            num_frames=120,
            num_inference_steps=8,
            fps=24,
            ref_image_size="max",
        )
        p_multi = builder(
            "comfy_h3_10eros_max",
            req_multi,
            seed=777,
            ref_image_filenames=["a.png", "b.png", "c.png"],
            ref_audio_filename="voice.wav",
            save_prefix="video/test_multi",
        )
        self.assertEqual(p_multi["1"]["inputs"]["unet_name"], "10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors")
        self.assertEqual(p_multi["8"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(p_multi["8"]["inputs"]["length"], 124)
        self.assertEqual(p_multi["8"]["inputs"]["ref_image_size"], "max")
        self.assertIn("ref_images.ref_image_0", p_multi["8"]["inputs"])
        self.assertIn("ref_images.ref_image_1", p_multi["8"]["inputs"])
        self.assertIn("ref_images.ref_image_2", p_multi["8"]["inputs"])
        self.assertIn("ref_audios.ref_audio_0", p_multi["8"]["inputs"])
        audio_nid = p_multi["8"]["inputs"]["ref_audios.ref_audio_0"][0]
        self.assertEqual(p_multi[audio_nid]["class_type"], "LoadAudio")
        self.assertEqual(p_multi["10"]["inputs"]["sampler_name"], "lcm")
        self.assertEqual(p_multi["9"]["inputs"]["scheduler"], "simple")
        self.assertEqual(p_multi["12"]["class_type"], "BasicGuider")

        req_fl2 = SimpleNamespace(
            task_mode="fl2va",
            visual_prompt="Start to end transition",
            soundscape_prompt="",
            music_prompt="",
            negative_prompt="",
            width=1056,
            height=608,
            num_frames=56,
            num_inference_steps=8,
            fps=24,
        )
        p_fl2 = builder(
            "comfy_h3_10eros_max",
            req_fl2,
            seed=99,
            ref_image_filenames=[],
            first_frame_filename="start.png",
            last_frame_filename="end.png",
        )
        self.assertEqual(p_fl2["8"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertIn("first_frame", p_fl2["8"]["inputs"])
        self.assertIn("last_frame", p_fl2["8"]["inputs"])

    def test_build_comfy_h3_api_prompt_sulphur3_corrector_on_and_off(self):
        builder = self.ns["_build_comfy_h3_api_prompt_for_request"]
        req_on = SimpleNamespace(
            task_mode="t2va",
            visual_prompt="Dynamic action scene",
            soundscape_prompt="",
            music_prompt="",
            negative_prompt="blurry",
            width=1056,
            height=608,
            num_frames=124,
            num_inference_steps=20,
            fps=24,
            corrector_enabled=True,
            corrector_strength=1.2,
            corrector_budget_fraction=0.5,
            corrector_start_step=2,
            corrector_end_step=18,
        )
        p_on = builder("comfy_h3_sulphur3", req_on, seed=11, ref_image_filenames=[])
        self.assertEqual(p_on["1"]["inputs"]["unet_name"], "s3.safetensors")
        self.assertEqual(p_on["8"]["class_type"], "MiniMaxH3TextToVideo")
        self.assertEqual(p_on["10"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(p_on["17"]["class_type"], "LoadH3Corrector")
        self.assertEqual(p_on["18"]["class_type"], "ApplyH3Corrector")
        self.assertEqual(p_on["18"]["inputs"]["strength"], 1.2)
        self.assertEqual(p_on["18"]["inputs"]["start_step"], 2)
        self.assertEqual(p_on["18"]["inputs"]["end_step"], 18)
        self.assertEqual(p_on["12"]["class_type"], "CFGGuider")
        self.assertEqual(p_on["12"]["inputs"]["cfg"], 1.0)
        self.assertEqual(p_on["12"]["inputs"]["model"], ["18", 0])

        req_off = SimpleNamespace(
            task_mode="fl2va",
            visual_prompt="Frame transition without corrector",
            soundscape_prompt="",
            music_prompt="",
            negative_prompt="",
            width=1056,
            height=608,
            num_frames=124,
            num_inference_steps=20,
            fps=24,
            corrector_enabled=False,
            corrector_strength=1.0,
        )
        p_off = builder(
            "comfy_h3_sulphur3",
            req_off,
            seed=12,
            ref_image_filenames=[],
            first_frame_filename="start.png",
        )
        self.assertEqual(p_off["8"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertNotIn("17", p_off)
        self.assertNotIn("18", p_off)
        self.assertEqual(p_off["12"]["inputs"]["model"], ["2", 0])

    def test_macos_native_ui_exposes_multiref_voice_corrector_and_autosleep(self):
        required_tokens = {
            os.path.join(REPO_ROOT, "macos_app", "Sources", "ModelsAndCatalogs.swift"): ["multiref", "motion_transfer", "video_upscale", "drama_multishot", "voice_design", "drama_tts_clone", "minimax_h3_10eros_max", "minimax_h3_sulphur3", "minimax_h3_ref2va_tara", "recavsr_upscaler", "moss_image_edit", "qwen_image_edit_2511", "wan22_remix_nsfw", "minimax_h3_drama_director", "qwen3_tts_voice_design", "drama3_tts", "qwen38_27b_uncensored_gguf"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"): ["ref_images_base64", "ref_audio_base64", "ref_video_base64", "corrector_strength", "tara_strength", "recavsr_scale", "moss_edit_lora_strength", "moss_nsfw_lora_strength", "wan22_shift", "wan22_split_step", "wan22_cfg", "drama_project_seconds", "tts_instruct", "drama3_voice_guidance", "autoSleepAfterGenerate", "disconnectRemoteColab"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MainStudioView+Workspace.swift"): ["multiref", "motion_transfer", "video_upscale", "drama_multishot", "voice_clone", "drama_tts_clone", "h3CorrectorSection", "h3TaraControlSection", "recavsrControlSection", "mossImageControlSection", "wan22RemixControlSection", "h3DramaDirectorControlSection", "qwen3TTSControlSection", "drama3TTSControlSection", "autoSleepAfterGenerate"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel+Services.swift"): ["useCurrentAudioAsVoiceReference"],
        }
        for path, tokens in required_tokens.items():
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            for tok in tokens:
                self.assertIn(tok, content, f"missing {tok} in {path}")

    def test_ref2va_tara_motion_transfer_catalog_pack_graph_and_cli(self):
        catalog = self.ns["MODEL_CATALOG"]
        self.assertIn("minimax_h3_ref2va_tara", catalog)
        self.assertEqual(catalog["minimax_h3_ref2va_tara"]["pipeline_arch"], "comfy_h3_ref2va_tara")
        self.assertEqual(catalog["minimax_h3_ref2va_tara"]["comfy_pack_id"], "minimax_h3_ref2va_tara")

        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        self.assertIn("minimax_h3_ref2va_tara", packs)
        pack = packs["minimax_h3_ref2va_tara"]
        model_files = {fn for _, fn, _, _, _ in pack["models"]}
        self.assertIn("minimax_h3_ref2va_pruned_int8_convrot.safetensors", model_files)
        self.assertIn("minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors", model_files)
        self.assertIn("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", model_files)
        cn_names = {c["name"] for c in pack.get("custom_nodes", [])}
        self.assertIn("ComfyUI-BFSNodes", cn_names)

        builder = self.ns["_build_comfy_h3_api_prompt_for_request"]
        req_tara = SimpleNamespace(
            task_mode="motion_transfer",
            visual_prompt="<Subject 1>: person in <Picture 1>. Replace character in <Video 1>.",
            soundscape_prompt="",
            music_prompt="",
            negative_prompt="",
            width=1056,
            height=608,
            num_frames=124,
            num_inference_steps=8,
            fps=24,
            ref_image_size="match",
            tara_enabled=True,
            tara_lora_strength=0.9,
            tara_strength=2.2,
            tara_window_seconds=0.2,
            tara_falloff_seconds=0.25,
            tara_max_penalty=6.0,
        )
        g_on = builder(
            "comfy_h3_ref2va_tara",
            req_tara,
            seed=42,
            ref_image_filenames=["subject.png"],
            ref_video_filename="guide.mp4",
            save_prefix="video/tara_test",
        )
        self.assertEqual(g_on["1"]["inputs"]["unet_name"], "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
        self.assertEqual(g_on["30"]["inputs"]["lora_name"], "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors")
        self.assertEqual(g_on["31"]["inputs"]["lora_name"], "minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors")
        self.assertEqual(g_on["31"]["inputs"]["strength_model"], 0.9)
        self.assertEqual(g_on["32"]["class_type"], "BFSMiniMaxH3NativeTemporal")
        self.assertEqual(g_on["32"]["inputs"]["strength"], 2.2)
        self.assertEqual(g_on["29"]["class_type"], "VHS_LoadVideo")
        self.assertEqual(g_on["8"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(g_on["8"]["inputs"]["ref_videos.ref_video_0"], ["29", 0])
        self.assertEqual(g_on["9"]["inputs"]["model"], ["32", 0])
        self.assertEqual(g_on["12"]["inputs"]["model"], ["32", 0])

        req_tara.tara_enabled = False
        g_off = builder(
            "comfy_h3_ref2va_tara",
            req_tara,
            seed=42,
            ref_image_filenames=["subject.png"],
            ref_video_filename="guide.mp4",
        )
        self.assertNotIn("32", g_off)
        self.assertEqual(g_off["9"]["inputs"]["model"], ["2", 0])
        self.assertEqual(g_off["12"]["inputs"]["model"], ["2", 0])

        # Verify Skill CLI comfy-run-tara --dry-run executes cleanly
        with tempfile.TemporaryDirectory() as tmp:
            vid = os.path.join(tmp, "guide.mp4")
            img = os.path.join(tmp, "char.png")
            open(vid, "wb").write(b"fake_mp4")
            open(img, "wb").write(b"fake_png")
            cli_path = os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py")
            proc = subprocess.run(
                [sys.executable, cli_path, "comfy-run-tara", "--ref-video", vid, "--ref-image", img, "--prompt", "<Video 1> <Picture 1>", "--dry-run"],
                capture_output=True, text=True, check=True,
            )
            parsed = json.loads(proc.stdout)
            self.assertEqual(parsed["prompt"]["32"]["class_type"], "BFSMiniMaxH3NativeTemporal")
            self.assertIn("ref_videos.ref_video_0", parsed["prompt"]["8"]["inputs"])

    def test_trimmed_catalog_and_packs_match_retained_set(self):
        retained_models = {
            "minimax_h3_10eros_max",
            "minimax_h3_sulphur3",
            "minimax_h3_ref2va_tara",
            "recavsr_upscaler",
            "moss_image_edit",
            "qwen_image_edit_2511",
            "wan22_remix_nsfw",
            "minimax_h3_drama_director",
            "qwen3_tts_voice_design",
            "drama3_tts",
            "qwen38_27b_uncensored_gguf",
        }
        retained_packs = {
            "minimax_h3_10eros_max",
            "minimax_h3_sulphur3",
            "minimax_h3_ref2va_tara",
            "recavsr_upscaler",
            "moss_image_edit",
            "qwen_image_edit_2511",
            "wan22_remix_nsfw",
            "minimax_h3_drama_director",
            "qwen3_tts_voice_design",
            "drama3_tts",
        }
        deleted = {
            "minimax_h3",
            "seedvr2_upscaler",
            "minimax_h3_image",
            "qwen_image_21",
            "z_image_turbo",
            "qwen38_27b",
            "qwen38_27b_gguf",
            "wan22_i2v_flf2v",
            "minimax_h3_i2v_flf2v",
            "ltx23_flf2v",
            "hunyuan15_i2v_1080p",
        }
        catalog_keys = set(self.ns["MODEL_CATALOG"].keys())
        pack_ids = {p["id"] for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        self.assertEqual(catalog_keys, retained_models)
        self.assertEqual(pack_ids, retained_packs)
        self.assertFalse(catalog_keys & deleted)
        self.assertFalse(pack_ids & deleted)

    def test_recavsr_streaming_upscaler_catalog_pack_graph_and_cli(self):
        self.assertIn("recavsr", self.ns["COMFY_STANDARD_MODEL_SUBDIRS"])
        catalog = self.ns["MODEL_CATALOG"]
        self.assertIn("recavsr_upscaler", catalog)
        self.assertEqual(catalog["recavsr_upscaler"]["pipeline_arch"], "comfy_recavsr_upscaler")
        self.assertEqual(catalog["recavsr_upscaler"]["comfy_pack_id"], "recavsr_upscaler")

        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        self.assertIn("recavsr_upscaler", packs)
        pack = packs["recavsr_upscaler"]
        model_files = {(sub, fn) for sub, fn, _, _, _ in pack["models"]}
        self.assertIn(("recavsr", "transformer.safetensors"), model_files)
        self.assertIn(("recavsr", "transformer_int8_convrot.safetensors"), model_files)
        self.assertIn(("recavsr", "flashdecoder.safetensors"), model_files)
        self.assertIn(("recavsr", "prompt.safetensors"), model_files)
        cn_names = {c["name"] for c in pack.get("custom_nodes", [])}
        self.assertIn("ComfyUI-ReCaVSR", cn_names)

        with tempfile.TemporaryDirectory() as tmp:
            installed = self.ns["_build_and_install_comfy_preset_workflows"](tmp, ["recavsr_upscaler"])
            self.assertEqual(installed, ["13_ReCaVSR_Wan2.2_极速流式4K超分.json"])
            wf_path = os.path.join(tmp, "user", "default", "workflows", installed[0])
            wf = json.load(open(wf_path, "r", encoding="utf-8"))
            ntypes = {n["type"] for n in wf["nodes"]}
            self.assertIn("ReCaVSRLoader", ntypes)
            self.assertIn("ReCaVSRUpscale", ntypes)

        builder = self.ns["_build_recavsr_api_prompt_for_request"]
        req_rc = SimpleNamespace(
            recavsr_transformer="transformer_int8_convrot.safetensors",
            recavsr_scale=4.0,
            recavsr_color_fix="wavelet",
            recavsr_kernels="exact",
            recavsr_encoder="hevc nvenc",
        )
        g = builder(req_rc, seed=77, ref_video_filename="dance.mp4", save_prefix="video/recavsr_test")
        self.assertEqual(g["1"]["class_type"], "LoadVideo")
        self.assertEqual(g["2"]["class_type"], "ReCaVSRLoader")
        self.assertEqual(g["2"]["inputs"]["transformer"], "transformer_int8_convrot.safetensors")
        self.assertEqual(g["3"]["class_type"], "ReCaVSRUpscale")
        self.assertEqual(g["3"]["inputs"]["scale"], 4.0)
        self.assertEqual(g["3"]["inputs"]["color_fix"], "wavelet")
        self.assertEqual(g["3"]["inputs"]["decoder"], "flashdecoder")
        self.assertEqual(g["3"]["inputs"]["kernels"], "exact")
        self.assertEqual(g["3"]["inputs"]["encoder"], "hevc nvenc")
        self.assertEqual(g["4"]["class_type"], "SaveVideo")

        with tempfile.TemporaryDirectory() as tmp:
            vid = os.path.join(tmp, "clip.mp4")
            open(vid, "wb").write(b"fake_mp4")
            cli_path = os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py")
            proc = subprocess.run(
                [
                    sys.executable, cli_path, "comfy-upscale-recavsr",
                    "--video", vid,
                    "--transformer", "transformer.safetensors",
                    "--scale", "2.0",
                    "--dry-run",
                ],
                capture_output=True, text=True, check=True,
            )
            parsed = json.loads(proc.stdout)
            self.assertEqual(parsed["prompt"]["2"]["inputs"]["transformer"], "transformer.safetensors")
            self.assertEqual(parsed["prompt"]["3"]["class_type"], "ReCaVSRUpscale")
            self.assertEqual(parsed["prompt"]["3"]["inputs"]["scale"], 2.0)
            self.assertEqual(parsed["prompt"]["3"]["inputs"]["decoder"], "flashdecoder")

    def test_moss_image_uncensored_edit_catalog_pack_graph_and_cli(self):
        catalog = self.ns["MODEL_CATALOG"]
        self.assertIn("moss_image_edit", catalog)
        self.assertEqual(catalog["moss_image_edit"]["pipeline_arch"], "comfy_moss_image_edit")
        self.assertEqual(catalog["moss_image_edit"]["comfy_pack_id"], "moss_image_edit")

        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        self.assertIn("moss_image_edit", packs)
        pack = packs["moss_image_edit"]
        model_files = {(sub, fn) for sub, fn, _, _, _ in pack["models"]}
        self.assertIn(("diffusion_models", "moss_image_sft_bf16.safetensors"), model_files)
        self.assertIn(("text_encoders", "moss_te_heretic_full_int8_convrot.safetensors"), model_files)
        self.assertIn(("vae", "flux2-vae.safetensors"), model_files)
        self.assertIn(("loras", "moss_edit_armV5.safetensors"), model_files)
        self.assertIn(("loras", "MOSS-Image-NSFW-Uncensored-LoRA-ComfyUI.safetensors"), model_files)
        cn_names = {c["name"] for c in pack.get("custom_nodes", [])}
        self.assertIn("ComfyUI-MOSS-Image-Ostris-Edit", cn_names)

        with tempfile.TemporaryDirectory() as tmp:
            installed = self.ns["_build_and_install_comfy_preset_workflows"](tmp, ["moss_image_edit"])
            self.assertEqual(
                installed,
                [
                    "14_MOSS-Image_无限制文生图(Heretic+NSFW).json",
                    "15_MOSS-Image_无限制指令修图(OstrisEdit+armV5).json",
                ],
            )
            wf_edit = json.load(open(os.path.join(tmp, "user", "default", "workflows", installed[1]), "r", encoding="utf-8"))
            ntypes = {n["type"] for n in wf_edit["nodes"]}
            self.assertNotIn("MOSSImageOstrisEditModelPatch", ntypes)
            self.assertIn("TextEncodeMOSSImageEdit", ntypes)
            self.assertIn("SkipLayerGuidanceDiTSimple", ntypes)

        builder = self.ns["_build_moss_image_api_prompt_for_request"]
        req_t2i = SimpleNamespace(
            task_mode="t2i",
            visual_prompt="moss_xvstyle studio portrait",
            negative_prompt="blurry",
            width=1024,
            height=1024,
            num_inference_steps=36,
            moss_cfg=3.0,
            moss_edit_lora_strength=1.0,
            moss_nsfw_lora_enabled=True,
            moss_nsfw_lora_strength=0.65,
            moss_kv_cache=True,
            moss_match_input_size=True,
        )
        g_t2i = builder(req_t2i, seed=101, ref_image_filenames=[], save_prefix="image/moss_t2i")
        self.assertEqual(g_t2i["2"]["inputs"]["type"], "moss_image")
        self.assertEqual(g_t2i["2"]["inputs"]["clip_name"], "moss_te_heretic_full_int8_convrot.safetensors")
        self.assertEqual(g_t2i["6"]["inputs"]["lora_name"], "MOSS-Image-NSFW-Uncensored-LoRA-ComfyUI.safetensors")
        self.assertEqual(g_t2i["8"]["class_type"], "CLIPTextEncode")
        self.assertNotIn("4", g_t2i)

        req_edit = SimpleNamespace(
            task_mode="i2i_edit",
            visual_prompt="change outfit to red dress",
            negative_prompt="",
            width=1024,
            height=1024,
            num_inference_steps=30,
            moss_cfg=3.0,
            moss_edit_lora_strength=1.0,
            moss_nsfw_lora_enabled=True,
            moss_nsfw_lora_strength=0.65,
            moss_kv_cache=True,
            moss_match_input_size=True,
        )
        g_edit = builder(req_edit, seed=202, ref_image_filenames=["a.png", "b.png"], save_prefix="image/moss_edit")
        self.assertNotIn("4", g_edit)
        self.assertEqual(g_edit["5"]["inputs"]["lora_name"], "moss_edit_armV5.safetensors")
        self.assertEqual(g_edit["8"]["class_type"], "TextEncodeMOSSImageEdit")
        self.assertEqual(g_edit["8"]["inputs"]["edit_format"], "ai-toolkit")
        self.assertIn("images.image_0", g_edit["8"]["inputs"])
        self.assertIn("images.image_1", g_edit["8"]["inputs"])
        self.assertEqual(g_edit["26"]["class_type"], "GetImageSize")

        with tempfile.TemporaryDirectory() as tmp:
            img1 = os.path.join(tmp, "ref1.png")
            open(img1, "wb").write(b"fake_png")
            cli_path = os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py")
            proc = subprocess.run(
                [
                    sys.executable, cli_path, "comfy-run-moss",
                    "--mode", "i2i_edit",
                    "--ref-image", img1,
                    "--prompt", "swap background to neon alley",
                    "--dry-run",
                ],
                capture_output=True, text=True, check=True,
            )
            parsed = json.loads(proc.stdout)
            self.assertNotIn("4", parsed["prompt"])
            self.assertEqual(parsed["prompt"]["5"]["inputs"]["lora_name"], "moss_edit_armV5.safetensors")
            self.assertEqual(parsed["prompt"]["8"]["class_type"], "TextEncodeMOSSImageEdit")

    def test_wan22_remix_and_qwen38_uncensored_gguf_catalog_pack_graph_and_cli(self):
        catalog = self.ns["MODEL_CATALOG"]
        self.assertIn("qwen38_27b_uncensored_gguf", catalog)
        qwen_cfg = catalog["qwen38_27b_uncensored_gguf"]
        self.assertEqual(qwen_cfg["pipeline_arch"], "gguf")
        self.assertEqual(qwen_cfg["modality"], "llm_vlm")
        self.assertEqual(qwen_cfg["repo_id"], "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF")
        self.assertEqual(
            [f[0] for f in qwen_cfg["files"]],
            [
                "Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf",
                "mmproj-Qwen3.8-27B-Uncensored-F16.gguf",
                "Qwen3.8-27B-Uncensored-draft-Q4_0.gguf",
            ],
        )

        self.assertIn("wan22_remix_nsfw", catalog)
        wan_cfg = catalog["wan22_remix_nsfw"]
        self.assertEqual(wan_cfg["pipeline_arch"], "comfy_wan22_remix")
        self.assertEqual(wan_cfg["comfy_pack_id"], "wan22_remix_nsfw")
        self.assertEqual(wan_cfg["modality"], "video")

        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        self.assertIn("wan22_remix_nsfw", packs)
        pack = packs["wan22_remix_nsfw"]
        self.assertEqual(pack.get("custom_nodes", []), [])
        filenames = {m[1] for m in pack["models"]}
        self.assertEqual(
            filenames,
            {
                "Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors",
                "Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors",
                "nsfw_wan_umt5-xxl_fp8_scaled.safetensors",
                "wan_2.1_vae.safetensors",
            },
        )
        self.assertEqual(
            set(pack["workflows"]),
            {
                "16_Wan2.2-Remix_无限制单图生视频(v3.0_8步极速).json",
                "17_Wan2.2-Remix_无限制首尾帧过渡(v3.0_8步极速).json",
            },
        )

        wf_i2v = self.ns["_build_wan22_remix_workflow"](is_flf2v=False)
        wf_flf = self.ns["_build_wan22_remix_workflow"](is_flf2v=True)
        self.assertIn("WanImageToVideo", {n["type"] for n in wf_i2v["nodes"]})
        self.assertIn("WanFirstLastFrameToVideo", {n["type"] for n in wf_flf["nodes"]})

        ReqCls = self.ns["GenerateVideoRequest"]
        req_i2v = ReqCls(
            model_id="wan22_remix_nsfw",
            task_mode="i2v",
            visual_prompt="cinematic portrait motion",
            width=1280,
            height=720,
            num_frames=81,
            num_inference_steps=8,
            wan22_shift=8.0,
            wan22_split_step=4,
            wan22_cfg=1.0,
        )
        g_i2v = self.ns["_build_wan22_remix_api_prompt_for_request"](req_i2v, seed=42, first_frame_filename="start.png")
        self.assertEqual(g_i2v["1"]["inputs"]["unet_name"], "Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors")
        self.assertEqual(g_i2v["2"]["inputs"]["unet_name"], "Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors")
        self.assertEqual(g_i2v["5"]["inputs"]["clip_name"], "nsfw_wan_umt5-xxl_fp8_scaled.safetensors")
        self.assertEqual(g_i2v["11"]["class_type"], "WanImageToVideo")
        self.assertEqual(g_i2v["12"]["inputs"]["end_at_step"], 4)
        self.assertEqual(g_i2v["13"]["inputs"]["start_at_step"], 4)

        req_flf = ReqCls(
            model_id="wan22_remix_nsfw",
            task_mode="fl2va",
            visual_prompt="smooth transition",
            width=1280,
            height=720,
            num_frames=80,
            num_inference_steps=8,
            wan22_shift=8.0,
            wan22_split_step=4,
            wan22_cfg=1.0,
        )
        g_flf = self.ns["_build_wan22_remix_api_prompt_for_request"](
            req_flf, seed=42, first_frame_filename="start.png", last_frame_filename="end.png"
        )
        self.assertEqual(g_flf["11"]["class_type"], "WanFirstLastFrameToVideo")
        self.assertEqual(g_flf["11"]["inputs"]["length"], 81)
        self.assertEqual(g_flf["10"]["inputs"]["image"], "end.png")

        with tempfile.TemporaryDirectory() as tmp:
            img1 = os.path.join(tmp, "first.png")
            img2 = os.path.join(tmp, "last.png")
            open(img1, "wb").write(b"fake_png_1")
            open(img2, "wb").write(b"fake_png_2")
            cli_path = os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py")
            proc = subprocess.run(
                [
                    sys.executable, cli_path, "comfy-run-wan22-remix",
                    "--mode", "fl2va",
                    "--first-frame", img1,
                    "--last-frame", img2,
                    "--prompt", "cinematic orbit",
                    "--dry-run",
                ],
                capture_output=True, text=True, check=True,
            )
            parsed = json.loads(proc.stdout)
            self.assertEqual(parsed["prompt"]["11"]["class_type"], "WanFirstLastFrameToVideo")
            self.assertEqual(parsed["prompt"]["5"]["inputs"]["clip_name"], "nsfw_wan_umt5-xxl_fp8_scaled.safetensors")
            self.assertEqual(parsed["prompt"]["10"]["inputs"]["image"], "last.png")

    def test_h3_drama_director_catalog_pack_workflows_api_and_cli(self):
        self.assertIn("vae_approx", self.ns["COMFY_STANDARD_MODEL_SUBDIRS"])
        catalog = self.ns["MODEL_CATALOG"]
        self.assertIn("minimax_h3_drama_director", catalog)
        cfg = catalog["minimax_h3_drama_director"]
        self.assertEqual(cfg["pipeline_arch"], "comfy_h3_drama_director")
        self.assertEqual(cfg["comfy_pack_id"], "minimax_h3_drama_director")
        self.assertEqual(cfg["repo_id"], "seesee75-commits/ComfyUI-MiniMaxH3-Director")

        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        self.assertIn("minimax_h3_drama_director", packs)
        pack = packs["minimax_h3_drama_director"]
        self.assertEqual(
            [cn["subdir"] for cn in pack.get("custom_nodes", [])],
            ["ComfyUI-MiniMaxH3-Director"],
        )
        self.assertIn("taeh3.safetensors", {m[1] for m in pack["models"]})
        self.assertEqual(
            set(pack["workflows"]),
            {
                "18_MiniMax-H3_短剧导演_角色声纹多镜头(10Eros-Max_8步).json",
                "19_MiniMax-H3_短剧导演_双角色分镜表成片(10Eros-Max_8步).json",
            },
        )

        wf_char = self.ns["_build_h3_drama_director_workflow"](with_storyboard=False)
        wf_sb = self.ns["_build_h3_drama_director_workflow"](with_storyboard=True)
        self.assertIn("NGH3Director", {n["type"] for n in wf_char["nodes"]})
        self.assertIn("NGContextAudio", {n["type"] for n in wf_char["nodes"]})
        self.assertIn("NGProjectRender", {n["type"] for n in wf_char["nodes"]})
        self.assertIn("NGStoryboard", {n["type"] for n in wf_sb["nodes"]})

        ReqCls = self.ns["GenerateVideoRequest"]
        req = ReqCls(
            model_id="minimax_h3_drama_director",
            task_mode="drama_multishot",
            visual_prompt="0-7: @Hero walks down neon street.\n7-14: @Hero says: Wait.",
            soundscape_prompt="Rain and footsteps",
            music_prompt="Ambient piano",
            width=1280,
            height=704,
            num_inference_steps=8,
            drama_project_seconds=20.0,
            drama_continuity="video_audio",
            drama_pin_frames="22",
            drama_style="Cinematic",
            drama_character_names=["Hero", "Partner"],
        )
        graph = self.ns["_build_h3_drama_director_api_prompt_for_request"](
            req,
            seed=42,
            ref_image_filenames=["hero.png", "partner.png"],
            ref_audio_filename="hero.wav",
        )
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors")
        self.assertEqual(graph["20"]["class_type"], "NGContextImage")
        self.assertEqual(graph["20"]["inputs"]["name"], "Hero")
        self.assertEqual(graph["21"]["inputs"]["name"], "Partner")
        self.assertEqual(graph["30"]["class_type"], "NGContextAudio")
        self.assertEqual(graph["30"]["inputs"]["speaker"], "ctx_hero")
        self.assertEqual(graph["5"]["class_type"], "NGContextConnector")
        self.assertEqual(graph["6"]["class_type"], "NGH3Director")
        self.assertIs(graph["6"]["inputs"]["turbo"], False)
        self.assertEqual(graph["6"]["inputs"]["steps"], 8)
        self.assertEqual(graph["7"]["class_type"], "NGProjectRender")
        self.assertEqual(graph["8"]["class_type"], "SaveVideo")

        meta = self.ns["_extract_comfy_prompt_metadata"](graph)
        self.assertEqual(meta["inferred_model_id"], "minimax_h3_drama_director")

        with tempfile.TemporaryDirectory() as tmp:
            img1 = os.path.join(tmp, "hero.png")
            aud1 = os.path.join(tmp, "voice.wav")
            open(img1, "wb").write(b"fake_png")
            open(aud1, "wb").write(b"fake_wav")
            cli_path = os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py")
            proc = subprocess.run(
                [
                    sys.executable, cli_path, "comfy-run-h3-drama",
                    "--shots", "0-7: @Hero walks\n@Hero says: Ready",
                    "--ref-image", img1,
                    "--char-name", "Hero",
                    "--ref-audio", aud1,
                    "--duration", "20",
                    "--dry-run",
                ],
                capture_output=True, text=True, check=True,
            )
            parsed = json.loads(proc.stdout)
            self.assertEqual(parsed["prompt"]["6"]["class_type"], "NGH3Director")
            self.assertEqual(parsed["prompt"]["30"]["inputs"]["speaker"], "ctx_hero")
            self.assertIs(parsed["prompt"]["6"]["inputs"]["turbo"], False)

    def test_tts_dual_engine_catalog_packs_workflows_graphs_and_cli(self):
        subdirs = self.ns["COMFY_STANDARD_MODEL_SUBDIRS"]
        self.assertIn("asr", subdirs)
        self.assertIn("qwen-tts", subdirs)

        pip_re = self.ns["_COMFY_PROTECTED_PIP_PKGS"]
        for blocked in ("onnxruntime-openvino", "onnxruntime-gpu", "torch>=2.5", "torchaudio"):
            self.assertIsNotNone(pip_re.match(blocked), f"should filter {blocked}")
        for allowed in ("librosa", "soundfile", "transformers>=4.57.0", "einops", "sox", "onnxruntime", "onnxruntime>=1.16"):
            self.assertIsNone(pip_re.match(allowed), f"should allow {allowed}")

        catalog = self.ns["MODEL_CATALOG"]
        self.assertEqual(catalog["qwen3_tts_voice_design"]["modality"], "audio")
        self.assertEqual(catalog["qwen3_tts_voice_design"]["pipeline_arch"], "comfy_qwen3_tts")
        self.assertEqual(catalog["drama3_tts"]["modality"], "audio")
        self.assertEqual(catalog["drama3_tts"]["pipeline_arch"], "comfy_drama3_tts")

        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        q_pack = packs["qwen3_tts_voice_design"]
        self.assertEqual(len(q_pack.get("hf_snapshots", [])), 3)
        self.assertEqual([cn["subdir"] for cn in q_pack.get("custom_nodes", [])], ["ComfyUI-Qwen-TTS"])
        d_pack = packs["drama3_tts"]
        self.assertEqual(len(d_pack.get("models", [])), 3)
        self.assertEqual([cn["subdir"] for cn in d_pack.get("custom_nodes", [])], ["ComfyUI-Drama3"])

        wf_vd = self.ns["_build_qwen3_tts_workflow"](clone_mode=False)
        wf_vc = self.ns["_build_qwen3_tts_workflow"](clone_mode=True)
        self.assertIn("FB_Qwen3TTSVoiceDesign", {n["type"] for n in wf_vd["nodes"]})
        self.assertIn("FB_Qwen3TTSVoiceClone", {n["type"] for n in wf_vc["nodes"]})

        wf_d1 = self.ns["_build_drama3_tts_workflow"](multi_speaker=False)
        wf_d2 = self.ns["_build_drama3_tts_workflow"](multi_speaker=True)
        self.assertIn("Drama3Generate", {n["type"] for n in wf_d1["nodes"]})
        self.assertIn("SaveAudio", {n["type"] for n in wf_d2["nodes"]})
        # node index 2 is Drama3CodecLoader (id=2); link 13 must be preserved after pruning
        codec_node = next(n for n in wf_d2["nodes"] if n["type"] == "Drama3CodecLoader")
        self.assertIn(13, codec_node["outputs"][0]["links"])

        ReqCls = self.ns["GenerateVideoRequest"]
        req_q_vd = ReqCls(
            model_id="qwen3_tts_voice_design",
            task_mode="voice_design",
            visual_prompt="夜雨微凉。",
            tts_instruct="冷艳清冽女帝声线",
            tts_language="Chinese",
        )
        g_q_vd = self.ns["_build_qwen3_tts_api_prompt_for_request"](req_q_vd, seed=11)
        self.assertEqual(g_q_vd["2"]["class_type"], "FB_Qwen3TTSVoiceDesign")
        self.assertEqual(g_q_vd["3"]["class_type"], "SaveAudio")
        self.assertEqual(self.ns["_extract_comfy_prompt_metadata"](g_q_vd)["inferred_model_id"], "qwen3_tts_voice_design")

        req_q_vc = ReqCls(
            model_id="qwen3_tts_voice_design",
            task_mode="voice_clone",
            visual_prompt="今晚之后，一笔勾销。",
            tts_instruct="冷笑低沉语气",
            tts_language="Chinese",
            tts_ref_text="",
            tts_x_vector_only=False,
        )
        g_q_vc = self.ns["_build_qwen3_tts_api_prompt_for_request"](req_q_vc, seed=12, ref_audio_filename="seed.wav")
        self.assertEqual(g_q_vc["1"]["class_type"], "LoadAudio")
        self.assertEqual(g_q_vc["2"]["class_type"], "FB_Qwen3TTSVoiceClone")

        req_d_clone = ReqCls(
            model_id="drama3_tts",
            task_mode="drama_tts_clone",
            visual_prompt="[sighs] 事情没那么简单。",
            drama3_experts="in RAM",
            drama3_speakers_mode="one take per line",
            drama3_voice_guidance=0.0,
            drama3_shorten_pauses=True,
            tts_ref_text="",
            top_k=20,
        )
        g_d_clone = self.ns["_build_drama3_tts_api_prompt_for_request"](req_d_clone, seed=9, ref_audio_filename="hero.wav")
        self.assertEqual(g_d_clone["41"]["class_type"], "Qwen3ASRTranscribe")
        self.assertEqual(g_d_clone["6"]["inputs"]["voices.voice0"], ["5", 0])
        self.assertEqual(g_d_clone["6"]["inputs"]["top_k"], 0)
        self.assertEqual(self.ns["_extract_comfy_prompt_metadata"](g_d_clone)["inferred_model_id"], "drama3_tts")

        req_d_free = ReqCls(
            model_id="drama3_tts",
            task_mode="drama_tts_free",
            visual_prompt="<|speaker:0|> [whispers] 快走。",
            top_k=0,
        )
        g_d_free = self.ns["_build_drama3_tts_api_prompt_for_request"](req_d_free, seed=10, ref_audio_filename=None)
        self.assertNotIn("voices.voice0", g_d_free["6"]["inputs"])

        with tempfile.TemporaryDirectory() as tmp:
            a0 = os.path.join(tmp, "v0.wav")
            a1 = os.path.join(tmp, "v1.wav")
            open(a0, "wb").write(b"fake_wav0")
            open(a1, "wb").write(b"fake_wav1")
            cli_path = os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py")
            p1 = subprocess.run(
                [sys.executable, cli_path, "comfy-run-qwen3-tts", "--mode", "design", "--text", "测试台词", "--instruct", "成熟男低音", "--dry-run"],
                capture_output=True, text=True, check=True,
            )
            j1 = json.loads(p1.stdout)
            self.assertEqual(j1["prompt"]["2"]["class_type"], "FB_Qwen3TTSVoiceDesign")

            p2 = subprocess.run(
                [sys.executable, cli_path, "comfy-run-qwen3-tts", "--mode", "clone", "--ref-audio", a0, "--text", "测试克隆", "--dry-run"],
                capture_output=True, text=True, check=True,
            )
            j2 = json.loads(p2.stdout)
            self.assertEqual(j2["prompt"]["2"]["class_type"], "FB_Qwen3TTSVoiceClone")

            p3 = subprocess.run(
                [
                    sys.executable, cli_path, "comfy-run-drama3",
                    "--text", "<|speaker:0|> [sighs] A\n<|speaker:1|> [laughs] B",
                    "--ref-audio", a0,
                    "--ref-audio", a1,
                    "--speakers", "one take per line, hearing the line before",
                    "--voice-guidance", "1.0",
                    "--dry-run",
                ],
                capture_output=True, text=True, check=True,
            )
            j3 = json.loads(p3.stdout)
            self.assertEqual(j3["prompt"]["6"]["class_type"], "Drama3Generate")
            self.assertEqual(j3["prompt"]["6"]["inputs"]["top_k"], 0)
            self.assertIn("voices.voice0", j3["prompt"]["6"]["inputs"])
            self.assertIn("voices.voice1", j3["prompt"]["6"]["inputs"])

    def test_qwen_image_edit_2511_and_short_drama_consistency_cli(self):
        catalog = self.ns["MODEL_CATALOG"]
        self.assertIn("qwen_image_edit_2511", catalog)
        self.assertEqual(catalog["qwen_image_edit_2511"]["pipeline_arch"], "comfy_qwen_image_edit_2511")
        packs = {p["id"]: p for p in self.ns["COMFY_PRESET_WORKFLOW_PACKS"]}
        qpack = packs["qwen_image_edit_2511"]
        self.assertEqual(len(qpack["models"]), 5)
        self.assertEqual(qpack.get("custom_nodes", []), [])

        with tempfile.TemporaryDirectory() as tmp:
            installed = self.ns["_build_and_install_comfy_preset_workflows"](tmp, ["qwen_image_edit_2511"])
            self.assertEqual(
                installed,
                [
                    "24_Qwen-Image-Edit-2511_多图角色场景合成(4步极速).json",
                    "25_Qwen-Image-Edit-2511_96视角旋转与分镜改图(MultiAngles).json",
                ],
            )
            wf25 = self.ns["_build_qwen_image_edit_2511_workflow"](multi_angle=True)
            self.assertIn("TextEncodeQwenImageEditPlus", {n["type"] for n in wf25["nodes"]})
            self.assertIn("FluxKontextMultiReferenceLatentMethod", {n["type"] for n in wf25["nodes"]})

        ReqCls = self.ns["GenerateVideoRequest"]
        req = ReqCls(
            model_id="qwen_image_edit_2511",
            task_mode="i2i_edit",
            visual_prompt="<sks> right side view eye-level shot medium shot",
            width=1280,
            height=704,
            num_inference_steps=4,
            qwen_edit_lightning_strength=1.0,
            qwen_edit_angle_strength=0.9,
        )
        g = self.ns["_build_qwen_image_edit_2511_api_prompt_for_request"](req, seed=77, ref_image_filenames=["a.png", "b.png"])
        self.assertEqual(g["4"]["inputs"]["lora_name"], "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        self.assertEqual(g["5"]["inputs"]["lora_name"], "qwen-image-edit-2511-multiple-angles-lora.safetensors")
        self.assertEqual(g["8"]["class_type"], "TextEncodeQwenImageEditPlus")
        self.assertEqual(g["10"]["inputs"]["reference_latents_method"], "index_timestep_zero")
        self.assertEqual(self.ns["_extract_comfy_prompt_metadata"](g)["inferred_model_id"], "qwen_image_edit_2511")

        with tempfile.TemporaryDirectory() as tmp:
            img0 = os.path.join(tmp, "face.png")
            img1 = os.path.join(tmp, "study.png")
            open(img0, "wb").write(b"fake_png0")
            open(img1, "wb").write(b"fake_png1")
            cli_path = os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py")
            pq = subprocess.run(
                [
                    sys.executable, cli_path, "comfy-run-qwen-edit",
                    "--ref-image", img0, "--ref-image", img1,
                    "--angle-lora-strength", "0.9",
                    "--prompt", "<sks> right side view",
                    "--dry-run",
                ],
                capture_output=True, text=True, check=True,
            )
            jq = json.loads(pq.stdout)
            self.assertEqual(jq["prompt"]["8"]["class_type"], "TextEncodeQwenImageEditPlus")
            self.assertEqual(jq["prompt"]["12"]["class_type"], "EmptySD3LatentImage")

            ps = subprocess.run(
                [
                    sys.executable, cli_path, "comfy-sweep-360-plates",
                    "--master-plate", img1,
                    "--scene-desc", "Sunlit wooden study",
                    "--dry-run",
                ],
                capture_output=True, text=True, check=True,
            )
            js = json.loads(ps.stdout)
            self.assertEqual(js["snapped_frames"], 124)
            self.assertEqual(js["prompt"]["8"]["class_type"], "MiniMaxH3ReferenceToVideo")

            for subcmd in ("extract-last-frame", "stitch-drama-jlcut"):
                ph = subprocess.run([sys.executable, cli_path, subcmd, "--help"], capture_output=True, text=True, check=True)
                self.assertEqual(ph.returncode, 0)


if __name__ == "__main__":
    unittest.main()
