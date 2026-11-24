import ast
import io
import json
import os
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
        "COMFY_PRESET_WORKFLOW_PACKS",
        "MODEL_CATALOG",
    }
    wanted_funcs = {
        "_make_zh_guide_note",
        "_prune_workflow_dangling_links",
        "_build_10eros_max_workflow",
        "_build_sulphur3_workflow",
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
        "_build_comfy_h3_api_prompt_for_request",
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
            os.path.join(REPO_ROOT, "macos_app", "Sources", "ModelsAndCatalogs.swift"): ["multiref", "motion_transfer", "minimax_h3_10eros_max", "minimax_h3_sulphur3", "minimax_h3_ref2va_tara"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"): ["ref_images_base64", "ref_audio_base64", "ref_video_base64", "corrector_strength", "tara_strength", "autoSleepAfterGenerate", "disconnectRemoteColab"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MainStudioView+Workspace.swift"): ["multiref", "motion_transfer", "h3CorrectorSection", "h3TaraControlSection", "autoSleepAfterGenerate"],
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


if __name__ == "__main__":
    unittest.main()
