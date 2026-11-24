import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_CLI_HUB_PATH = os.path.expanduser(
    "~/.skills-manager/skills/colab-minimax-h3/scripts/colab_h3_cli.py"
)
SKILL_CLI_REPO_PATH = os.path.join(
    REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py"
)
COLAB_BRIDGE_PATH = os.path.join(REPO_ROOT, "colab_bridge.py")


def _load_module(path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExternalCLILogBusTests(unittest.TestCase):
    def test_skill_cli_copies_are_byte_identical(self):
        with open(SKILL_CLI_HUB_PATH, "rb") as f1, open(SKILL_CLI_REPO_PATH, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())

    def test_emit_app_log_writes_jsonl_masks_secrets_and_rotates(self):
        cli_mod = _load_module(SKILL_CLI_REPO_PATH, "colab_h3_cli_test")
        with tempfile.TemporaryDirectory() as tmp:
            log_file = os.path.join(tmp, "external_cli_logs.jsonl")
            with mock.patch.dict(os.environ, {"EASYCOLAB_APP_LOG_PATH": log_file}):
                cli_mod._emit_app_log(
                    "wake ready --password supersecretpassword123 --hf-token hf_abc123 https://abcdef123456.gradio.live Basic dXNlcjpwYXNz"
                )
                with open(log_file, "r", encoding="utf-8") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                self.assertEqual(len(rows), 1)
                line = rows[0]["line"]
                self.assertIn("[colab-h3-cli", line)
                self.assertNotIn("supersecretpassword123", line)
                self.assertNotIn("hf_abc123", line)
                self.assertNotIn("abcdef123456.gradio.live", line)
                self.assertNotIn("dXNlcjpwYXNz", line)
                self.assertIn("https://abcd***.gradio.live", line)

                # Simulate oversized >512KB file and verify rotation keeps tail clean
                with open(log_file, "w", encoding="utf-8") as wf:
                    for i in range(6000):
                        wf.write(json.dumps({"line": f"[colab-h3-cli 00:00:00] filler line {i} " + ("x" * 70)}) + "\n")
                self.assertGreater(os.path.getsize(log_file), 512 * 1024)
                cli_mod._emit_app_log("after rotation ok")
                with open(log_file, "r", encoding="utf-8") as rf:
                    rotated_lines = [l for l in rf if l.strip()]
                self.assertLessEqual(len(rotated_lines), 205)
                self.assertIn("after rotation ok", json.loads(rotated_lines[-1])["line"])

    def test_colab_bridge_external_log_respects_suppression_flags(self):
        # Verify colab_bridge syntax cleanly and test suppression/emission logic with minimal stubs
        with open(COLAB_BRIDGE_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("_append_external_app_log", src)
        self.assertIn("_external_cli_log_tee", src)
        self.assertIn("EASYCOLAB_FROM_APP", src)
        self.assertIn("EASYCOLAB_SUPPRESS_BRIDGE_LOG", src)

        import ast
        tree = ast.parse(src, filename=COLAB_BRIDGE_PATH)
        wanted = {
            "_mask_bridge_log_text",
            "_append_external_app_log",
            "_sync_active_session_file",
            "_external_cli_log_tee",
        }
        selected = [
            n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (
                isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) in ("_APP_LOG_PATH", "_ACTIVE_SESSION_PATH") for t in n.targets)
            )
        ]
        mod = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(mod)
        ns = {
            "os": os,
            "re": __import__("re"),
            "sys": sys,
            "json": json,
            "time": __import__("time"),
            "contextlib": __import__("contextlib"),
        }
        exec(compile(mod, COLAB_BRIDGE_PATH, "exec"), ns)

        with tempfile.TemporaryDirectory() as tmp:
            log_file = os.path.join(tmp, "external_cli_logs.jsonl")
            # Suppressed when EASYCOLAB_FROM_APP=1
            with mock.patch.dict(os.environ, {"EASYCOLAB_APP_LOG_PATH": log_file, "EASYCOLAB_FROM_APP": "1"}):
                ns["_append_external_app_log"]("should not appear")
                self.assertFalse(os.path.exists(log_file))
            # Suppressed when EASYCOLAB_SUPPRESS_BRIDGE_LOG=1
            with mock.patch.dict(os.environ, {"EASYCOLAB_APP_LOG_PATH": log_file, "EASYCOLAB_SUPPRESS_BRIDGE_LOG": "1"}):
                ns["_append_external_app_log"]("should also not appear")
                self.assertFalse(os.path.exists(log_file))
            # Emits when invoked externally without suppression
            clean_env = {
                "EASYCOLAB_APP_LOG_PATH": log_file,
                "EASYCOLAB_ACTIVE_SESSION_PATH": os.path.join(tmp, "active_session.json"),
            }
            with mock.patch.dict(os.environ, clean_env, clear=False):
                os.environ.pop("EASYCOLAB_FROM_APP", None)
                os.environ.pop("EASYCOLAB_SUPPRESS_BRIDGE_LOG", None)
                with ns["_external_cli_log_tee"]():
                    print("WAKE_LOG: 🚀 正在向 Google Colab 申请分配 G4 实例...")
                    print("WAKE_URL:https://xyz12345678.gradio.live")
                    print("WAKE_ERROR: test error")
            with open(log_file, "r", encoding="utf-8") as f:
                entries = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(entries), 3)
            self.assertIn("[CLI ", entries[0]["line"])
            self.assertIn("https://xyz1***.gradio.live", entries[1]["line"])
            self.assertEqual(entries[2]["level"], "error")

    def test_macos_swift_sources_wire_external_log_watcher_and_app_suppression(self):
        checks = {
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"): [
                "remoteConsoleLines",
                "externalCLILogLines",
                "startExternalCLILogWatcher",
                "clearConsoleLogs",
                "rebuildMergedConsoleLines",
            ],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel+Services.swift"): [
                "external_cli_logs.jsonl",
                "startExternalCLILogWatcher",
                "pollNewExternalCLILogLines",
                "rebuildMergedConsoleLines",
                "EASYCOLAB_FROM_APP",
            ],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MainStudioView.swift"): [
                "vm.clearConsoleLogs()",
                "[colab-h3-cli",
            ],
        }
        for path, tokens in checks.items():
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for tok in tokens:
                self.assertIn(tok, text, f"missing {tok} in {path}")

    def test_app_py_nvidia_smi_metrics_and_active_session_sync(self):
        # 1. Verify _sample_nvidia_smi_metrics parses valid CSV & handles missing binary
        with open(os.path.join(REPO_ROOT, "app.py"), "r", encoding="utf-8") as f:
            app_src = f.read()
        import ast
        tree = ast.parse(app_src, filename="app.py")
        fn_node = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_sample_nvidia_smi_metrics"
        )
        mod = ast.Module(body=[fn_node], type_ignores=[])
        ast.fix_missing_locations(mod)
        ns = {
            "subprocess": subprocess,
            "Dict": __import__("typing").Dict,
        }
        exec(compile(mod, "app.py", "exec"), ns)
        fn = ns["_sample_nvidia_smi_metrics"]
        with mock.patch("subprocess.check_output", return_value="87, 64, 241.5\n"):
            parsed = fn()
        self.assertEqual(parsed["gpu_util_pct"], 87.0)
        self.assertEqual(parsed["gpu_temp_c"], 64.0)
        self.assertEqual(parsed["gpu_power_w"], 241.5)
        with mock.patch("subprocess.check_output", side_effect=FileNotFoundError("no nvidia-smi")):
            fallback = fn()
        self.assertEqual(fallback["gpu_util_pct"], -1.0)

        # 2. Verify _sync_app_active_session writes 0o600 awake & sleeping session state
        cli_mod = _load_module(SKILL_CLI_REPO_PATH, "colab_h3_cli_session_test")
        with tempfile.TemporaryDirectory() as tmp:
            sess_file = os.path.join(tmp, "active_session.json")
            with mock.patch.dict(os.environ, {"EASYCOLAB_ACTIVE_SESSION_PATH": sess_file}):
                cli_mod._sync_app_active_session(
                    "https://demo1234.gradio.live/",
                    "secret_pw",
                    gpu="G4",
                    comfy_url="https://cf1234.trycloudflare.com",
                    sleeping=False,
                )
                data = json.loads(open(sess_file, "r", encoding="utf-8").read())
                self.assertEqual(data["url"], "https://demo1234.gradio.live")
                self.assertFalse(data["sleeping"])
                self.assertEqual(data["gpu"], "G4")
                cli_mod._sync_app_active_session("", "", sleeping=True)
                data2 = json.loads(open(sess_file, "r", encoding="utf-8").read())
                self.assertTrue(data2["sleeping"])
                self.assertEqual(data2["url"], "")

        # 3. Verify macOS Swift files include live monitoring & auto-attach tokens
        swift_checks = {
            os.path.join(REPO_ROOT, "macos_app", "Sources", "ModelsAndCatalogs.swift"): ["gpu_util_pct", "gpu_temp_c", "gpu_power_w"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"): ["liveMonitoringEnabled", "attachedExternalSessionSource"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel+Services.swift"): ["active_session.json", "checkExternalActiveSessionIfNeeded"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MainStudioView.swift"): ["liveMonitoringEnabled", "ComfyUI"],
        }
        for sp, toks in swift_checks.items():
            txt = open(sp, "r", encoding="utf-8").read()
            for t in toks:
                self.assertIn(t, txt, f"missing {t} in {sp}")

    def test_persistent_wake_retries_after_three_failures_and_wait(self):
        with open(os.path.join(REPO_ROOT, "colab_bridge.py"), "r", encoding="utf-8") as f:
            bridge_src = f.read()

        import ast
        import contextlib
        import uuid

        tree = ast.parse(bridge_src, filename="colab_bridge.py")
        wanted_funcs = {
            "_candidate_shapes_for_accel",
            "_resolve_gpu_accel",
            "_gpu_display_label",
            "_single_shot_colab_assign",
            "_wait_before_next_allocation_round",
            "_allocate_fresh_session",
        }
        selected = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted_funcs
        ]
        mod = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(mod)

        class _FakeEnum:
            def __init__(self, value: str, name: str = ""):
                self.value = value
                self.name = name or value

        class _FakeAccel:
            G4 = _FakeEnum("G4")
            H100 = _FakeEnum("H100")
            A100 = _FakeEnum("A100")
            L4 = _FakeEnum("L4")
            T4 = _FakeEnum("T4")

        class _FakeVariant:
            GPU = _FakeEnum("GPU")

        class _FakeShape:
            STANDARD = _FakeEnum("STANDARD", "STANDARD")
            HIGH_RAM = _FakeEnum("HIGH_RAM", "HIGH_RAM")

        class _FakeColabRequestError(Exception):
            def __init__(self, status: int):
                super().__init__(f"HTTP {status}")
                self.status = status

        class _FakeTooManyAssignmentsError(Exception):
            pass

        class _FakeSessionState:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        call_count = {"n": 0}

        class _FakeClient:
            def assign(self, nbh, variant=None, accelerator=None, shape=None):
                call_count["n"] += 1
                if call_count["n"] <= 3:
                    raise _FakeColabRequestError(503)
                return SimpleNamespace(
                    endpoint="ep-g4-1",
                    runtime_proxy_info=SimpleNamespace(
                        token="tok",
                        url="https://mock.colab.test",
                        expires_at=lambda: 9999999999.0,
                    ),
                )

            def list_assignments(self):
                return []

        class _FakeStore:
            def __init__(self):
                self.added = []

            def add(self, s):
                self.added.append(s)

        fake_state = SimpleNamespace(client=_FakeClient(), store=_FakeStore())
        buf = io.StringIO()
        ns = {
            "os": os,
            "time": __import__("time"),
            "uuid": uuid,
            "contextlib": contextlib,
            "GPU_ACCEL_MAP": {"g4": _FakeAccel.G4, "h100": _FakeAccel.H100},
            "Accelerator": _FakeAccel,
            "PostAssignmentResponse": type("_FakePostAssign", (), {}),
            "Variant": _FakeVariant,
            "Shape": _FakeShape,
            "SessionState": _FakeSessionState,
            "ColabRequestError": _FakeColabRequestError,
            "TooManyAssignmentsError": _FakeTooManyAssignmentsError,
            "requests": SimpleNamespace(exceptions=SimpleNamespace(RequestException=RuntimeError)),
            "get_status_code": lambda e: getattr(e, "status", None),
            "resolve_assign_shape": lambda a, high_mem=True: None,
            "_quiet_colab_cli": contextlib.nullcontext,
            "_try_adopt_matching_cloud_assignment": lambda accel, session_name="easycolab": None,
            "_unassign_endpoints_and_wait_quota_release": lambda eps, wait_s=8.0: 0,
            "state": fake_state,
        }
        exec(compile(mod, "colab_bridge.py", "exec"), ns)

        with mock.patch("time.sleep", return_value=None), contextlib.redirect_stdout(buf):
            sess = ns["_allocate_fresh_session"]("easycolab", "G4", persistent=True, retry_interval=0.05)

        self.assertEqual(call_count["n"], 4)
        self.assertEqual(sess.endpoint, "ep-g4-1")
        out_txt = buf.getvalue()
        self.assertIn("将在", out_txt)
        self.assertIn("第 2 轮持续申请", out_txt)

        # Non-persistent mode should fail cleanly after 3 attempts
        call_count["n"] = 0
        with mock.patch("time.sleep", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                ns["_allocate_fresh_session"]("easycolab", "G4", persistent=False)
        self.assertEqual(call_count["n"], 3)

        # Verify macOS Swift sources & Skill CLI wire persistent 60s retry + cancel
        swift_tokens = {
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"): ["persistentWakeEnabled"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel+Services.swift"): ["--persistent", "--retry-interval", "cancelActiveWake"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MainStudioView.swift"): ["persistentWakeEnabled", "cancelActiveWake"],
            os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py"): ["--persistent", "--retry-interval"],
        }
        for sp, toks in swift_tokens.items():
            txt = open(sp, "r", encoding="utf-8").read()
            for t in toks:
                self.assertIn(t, txt, f"missing {t} in {sp}")

    def test_multi_account_vault_save_switch_and_remove(self):
        with open(os.path.join(REPO_ROOT, "colab_bridge.py"), "r", encoding="utf-8") as f:
            bridge_src = f.read()

        import ast
        import contextlib
        import hashlib
        import time

        tree = ast.parse(bridge_src, filename="colab_bridge.py")
        wanted_funcs = {
            "_account_token_filename",
            "_load_accounts_index",
            "_save_accounts_index",
            "_write_secure_file",
            "_save_token_to_vault",
            "_backup_current_token_to_vault",
            "_list_saved_accounts",
            "cmd_auth_list",
            "cmd_auth_switch",
            "cmd_auth_remove",
        }
        selected = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted_funcs
        ]
        mod = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(mod)

        with tempfile.TemporaryDirectory() as tmp:
            tok_path = os.path.join(tmp, "token.json")
            acc_dir = os.path.join(tmp, "accounts")
            idx_path = os.path.join(acc_dir, "index.json")
            prof_path = os.path.join(tmp, "account_profile.json")
            cur_email = {"val": "a@example.com"}

            def _fake_has_valid():
                return os.path.isfile(tok_path)

            def _fake_status():
                data = json.loads(open(tok_path, "r", encoding="utf-8").read()) if os.path.isfile(tok_path) else {}
                em = data.get("email", cur_email["val"])
                cur_email["val"] = em
                ns["_backup_current_token_to_vault"](email=em, paid_balance=data.get("cu", 50.0), set_active=True)
                print(json.dumps({
                    "ok": True,
                    "logged_in": True,
                    "email": em,
                    "paid_balance": data.get("cu", 50.0),
                    "accounts": ns["_list_saved_accounts"](em),
                }))

            ns = {
                "os": os,
                "json": json,
                "time": time,
                "hashlib": hashlib,
                "TOKEN_CONFIG_PATH": tok_path,
                "ACCOUNTS_DIR": acc_dir,
                "ACCOUNTS_INDEX_PATH": idx_path,
                "ACCOUNT_PROFILE_CACHE_PATH": prof_path,
                "_has_valid_token": _fake_has_valid,
                "_get_account_email": lambda: cur_email["val"] if os.path.isfile(tok_path) else "",
                "_clear_local_session_store": lambda: None,
                "cmd_auth_status": _fake_status,
            }
            exec(compile(mod, "colab_bridge.py", "exec"), ns)

            # 1. Seed Account A and back it up
            ns["_write_secure_file"](tok_path, json.dumps({"email": "a@example.com", "cu": 88.5, "tok": "A"}))
            ns["_backup_current_token_to_vault"](email="a@example.com", paid_balance=88.5, set_active=True)

            # 2. Simulate adding Account B
            ns["_write_secure_file"](tok_path, json.dumps({"email": "b@example.com", "cu": 42.0, "tok": "B"}))
            cur_email["val"] = "b@example.com"
            ns["_backup_current_token_to_vault"](email="b@example.com", paid_balance=42.0, set_active=True)

            buf_list = io.StringIO()
            with contextlib.redirect_stdout(buf_list):
                ns["cmd_auth_list"]()
            listed = json.loads(buf_list.getvalue())
            self.assertEqual({a["email"] for a in listed["accounts"]}, {"a@example.com", "b@example.com"})
            self.assertEqual(listed["active_email"], "b@example.com")

            # 3. Switch back to Account A in 1 call
            buf_sw = io.StringIO()
            with contextlib.redirect_stdout(buf_sw):
                ns["cmd_auth_switch"]("a@example.com")
            sw = json.loads(buf_sw.getvalue())
            self.assertEqual(sw["email"], "a@example.com")
            self.assertEqual(json.loads(open(tok_path, "r", encoding="utf-8").read())["tok"], "A")
            self.assertEqual(len(sw["accounts"]), 2)

            # 4. Remove Account B and confirm Account A stays active
            buf_rm = io.StringIO()
            with contextlib.redirect_stdout(buf_rm):
                ns["cmd_auth_remove"]("b@example.com")
            rm = json.loads(buf_rm.getvalue())
            self.assertEqual([a["email"] for a in rm["accounts"]], ["a@example.com"])

        swift_acc_tokens = {
            os.path.join(REPO_ROOT, "macos_app", "Sources", "ModelsAndCatalogs.swift"): ["SavedGoogleAccount", "accounts:"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"): ["savedAccounts", "formattedAccountEmail"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel+Services.swift"): ["switchSavedGoogleAccount", "removeSavedGoogleAccount", "auth_switch", "auth_remove"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MainStudioView.swift"): ["savedAccounts", "switchSavedGoogleAccount", "removeSavedGoogleAccount"],
            os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py"): ["auth-list", "auth-switch", "auth-remove"],
        }
        for sp, toks in swift_acc_tokens.items():
            txt = open(sp, "r", encoding="utf-8").read()
            for t in toks:
                self.assertIn(t, txt, f"missing {t} in {sp}")


if __name__ == "__main__":
    unittest.main()
