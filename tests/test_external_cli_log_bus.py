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

    def test_scoped_account_isolation_and_full_email_display(self):
        with open(os.path.join(REPO_ROOT, "colab_bridge.py"), "r", encoding="utf-8") as f:
            bridge_src = f.read()

        import ast
        import hashlib
        import time

        tree = ast.parse(bridge_src, filename="colab_bridge.py")
        wanted_funcs = {
            "_account_token_filename",
            "_load_accounts_index",
            "_save_accounts_index",
            "_write_secure_file",
            "_save_token_to_vault",
            "_activate_scoped_account_context",
        }
        selected = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted_funcs
        ]
        mod = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(mod)

        with tempfile.TemporaryDirectory() as tmp:
            acc_dir = os.path.join(tmp, "accounts")
            idx_path = os.path.join(acc_dir, "index.json")
            default_tok = os.path.join(tmp, "default_token.json")
            fake_auth_mod = SimpleNamespace(TOKEN_CONFIG_PATH=default_tok)
            fake_state = SimpleNamespace(config_path=None, _store="old", _client="old", _sessions="old")
            ns = {
                "os": os,
                "json": json,
                "time": time,
                "hashlib": hashlib,
                "TOKEN_CONFIG_PATH": default_tok,
                "ACCOUNT_PROFILE_CACHE_PATH": os.path.join(tmp, "default_profile.json"),
                "ACCOUNTS_DIR": acc_dir,
                "ACCOUNTS_INDEX_PATH": idx_path,
                "_colab_auth": fake_auth_mod,
                "state": fake_state,
            }
            exec(compile(mod, "colab_bridge.py", "exec"), ns)

            ns["_save_token_to_vault"]("vanch.alpha@gmail.com", json.dumps({"tok": "A"}))
            ns["_save_token_to_vault"]("vanch.beta@gmail.com", json.dumps({"tok": "B"}))

            canonical = ns["_activate_scoped_account_context"]("VANCH.ALPHA@GMAIL.COM")
            self.assertEqual(canonical, "vanch.alpha@gmail.com")
            self.assertEqual(fake_auth_mod.TOKEN_CONFIG_PATH, ns["TOKEN_CONFIG_PATH"])
            self.assertIn(acc_dir, ns["TOKEN_CONFIG_PATH"])
            self.assertTrue(fake_state.config_path.endswith("_sessions.json"))
            self.assertIsNone(fake_state._client)

            with self.assertRaises(RuntimeError):
                ns["_activate_scoped_account_context"]("missing@example.com")

        # Full email must NOT be masked with "•••"
        vm_txt = open(os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"), "r", encoding="utf-8").read()
        self.assertIn("func formattedAccountEmail", vm_txt)
        fn_block = vm_txt.split("func formattedAccountEmail", 1)[1].split("var displayAccountLabel", 1)[0]
        self.assertNotIn("•••", fn_block)

        multi_checks = {
            os.path.join(REPO_ROOT, "macos_app", "EasyColabStudio.swift"): ["account-workspace", "boundAccountEmail"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "ModelsAndCatalogs.swift"): ["accelerators", "status_ok", "total_active_vms", "total_burn_rate_hourly"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel.swift"): ["boundAccountEmail", "activeBridgeAccountEmail", "perAccountURLDefaultsKey", "showMultiAccountWakeSheet", "totalSavedAccountsActiveVMs", "otherAccountsWithActiveVMs", "lastConnectionError", "isRecoveringTunnel"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "StudioViewModel+Services.swift"): ["EASYCOLAB_ACCOUNT", "scopedAccount", "wakeColabMultiRace", "launchMultiAccountBatchWake", "auth_status_all", "stopAccountRemoteVMs", "recover_tunnel", "recoverRunningColabConnection"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MainStudioView.swift"): ["openWindow", "account-workspace", "boundAccountEmail", "MultiAccountWakeSheet", "MultiAccountLiveMonitorView", "pendingStopAccountEmail", "运行中 · 未连", "已释放", "恢复连接"],
            os.path.join(REPO_ROOT, "macos_app", "Sources", "MultiAccountWakeSheet.swift"): ["MultiAccountWakeCoordinator", "MultiAccountWakeMode", "firstWins", "allParallel", "MultiAccountLiveMonitorView"],
            os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "scripts", "colab_h3_cli.py"): ["--account", "EASYCOLAB_ACCOUNT", "wake-multi", "auth-status-all", "auth_switch"],
            os.path.join(REPO_ROOT, "skills", "colab-minimax-h3", "SKILL.md"): ["auth-status-all", "wake-multi --gpu G4 --mode first --persistent"],
        }
        for sp, toks in multi_checks.items():
            with open(sp, "r", encoding="utf-8") as fh:
                txt = fh.read()
            for t in toks:
                self.assertIn(t, txt, f"missing {t} in {sp}")

    def _load_bridge_wake_multi_ns(self):
        import ast
        import signal
        import threading
        with open(COLAB_BRIDGE_PATH, "r", encoding="utf-8") as f:
            bridge_src = f.read()
        tree = ast.parse(bridge_src, filename="colab_bridge.py")
        wanted_funcs = {
            "_build_bridge_subcommand",
            "_unassign_single_endpoint_for_account",
            "cmd_wake_multi",
        }
        selected = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted_funcs
        ]
        mod = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(mod)
        ns = {
            "os": os,
            "sys": sys,
            "json": json,
            "signal": signal,
            "threading": threading,
            "subprocess": SimpleNamespace(
                Popen=None,
                PIPE=subprocess.PIPE,
                STDOUT=subprocess.STDOUT,
                run=None,
            ),
            "__file__": COLAB_BRIDGE_PATH,
            "_list_saved_accounts": lambda _active="": [],
            "_refresh_all_saved_accounts_live": lambda _active="": [],
        }
        exec(compile(mod, "colab_bridge.py", "exec"), ns)
        return ns

    def test_wake_multi_first_wins_terminates_losers_and_cleans_duplicate_new_vm(self):
        import time
        ns = self._load_bridge_wake_multi_ns()
        terminated_accounts = []
        cleaned_duplicates = []

        class FakeProc:
            def __init__(self, cmd, lines, delay_s=0.0):
                self.cmd = cmd
                self.acct = cmd[cmd.index("--account") + 1]
                self.terminated = False
                self._lines = list(lines)
                self._delay_s = delay_s
                self.stdout = self._iter_lines()

            def _iter_lines(self):
                if self._delay_s > 0:
                    time.sleep(self._delay_s)
                for ln in self._lines:
                    if self.terminated and ln.startswith("WAKE_URL:"):
                        break
                    yield ln + "\n"
                    time.sleep(0.01)

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True
                terminated_accounts.append(self.acct)

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.terminated = True

        def fake_popen(cmd, **kwargs):
            acct = cmd[cmd.index("--account") + 1]
            if acct == "win@example.com":
                return FakeProc(cmd, [
                    "WAKE_NEW_ALLOCATED: ep-winner-01",
                    "WAKE_LOG: 正在上传 app.py...",
                    "WAKE_URL: https://winner.gradio.live",
                ])
            if acct == "race-dup@example.com":
                # Simulates millisecond race: loser emits WAKE_NEW_ALLOCATED right after winner already locked
                return FakeProc(cmd, [
                    "WAKE_NEW_ALLOCATED: ep-loser-dup-99",
                    "WAKE_URL: https://loser.gradio.live",
                ], delay_s=0.05)
            return FakeProc(cmd, ["WAKE_LOG: 排队中..."], delay_s=0.08)

        ns["subprocess"].Popen = fake_popen
        ns["_unassign_single_endpoint_for_account"] = lambda em, ep: cleaned_duplicates.append((em, ep))

        buf = io.StringIO()
        old_stdout = sys.stdout
        old_pwd = os.environ.get("EASYCOLAB_ADMIN_PASSWORD")
        os.environ["EASYCOLAB_ADMIN_PASSWORD"] = "TestPasswordForRace1234567890"
        sys.stdout = buf
        try:
            res = ns["cmd_wake_multi"](
                ["win@example.com", "race-dup@example.com", "slow@example.com"],
                "G4",
                "/tmp/app.py",
                mode="first",
            )
        finally:
            sys.stdout = old_stdout
            if old_pwd is None:
                os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", None)
            else:
                os.environ["EASYCOLAB_ADMIN_PASSWORD"] = old_pwd

        out = buf.getvalue()
        self.assertTrue(res["ok"])
        self.assertEqual(res["winner"], "win@example.com")
        self.assertEqual(res["url"], "https://winner.gradio.live")
        self.assertIn("WAKE_WINNER: win@example.com", out)
        self.assertIn("WAKE_URL: https://winner.gradio.live", out)
        self.assertNotIn("https://loser.gradio.live", out)
        self.assertIn(("race-dup@example.com", "ep-loser-dup-99"), cleaned_duplicates)
        self.assertIn("race-dup@example.com", terminated_accounts)
        self.assertNotIn("win@example.com", cleaned_duplicates)

    def test_wake_multi_all_mode_collects_all_account_urls(self):
        ns = self._load_bridge_wake_multi_ns()

        class FakeProc:
            def __init__(self, cmd):
                acct = cmd[cmd.index("--account") + 1]
                slug = acct.split("@")[0]
                self.stdout = iter([
                    f"WAKE_NEW_ALLOCATED: ep-{slug}\n",
                    f"WAKE_URL: https://{slug}.gradio.live\n",
                ])

            def poll(self):
                return 0

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        ns["subprocess"].Popen = lambda cmd, **kwargs: FakeProc(cmd)
        buf = io.StringIO()
        old_stdout = sys.stdout
        old_pwd = os.environ.get("EASYCOLAB_ADMIN_PASSWORD")
        os.environ["EASYCOLAB_ADMIN_PASSWORD"] = "TestPasswordForBatch123456789"
        sys.stdout = buf
        try:
            res = ns["cmd_wake_multi"](
                ["alpha@example.com", "beta@example.com"],
                "G4",
                "/tmp/app.py",
                mode="all",
            )
        finally:
            sys.stdout = old_stdout
            if old_pwd is None:
                os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", None)
            else:
                os.environ["EASYCOLAB_ADMIN_PASSWORD"] = old_pwd

        out = buf.getvalue()
        self.assertTrue(res["ok"])
        self.assertEqual(res["mode"], "all")
        self.assertEqual(set(res["succeeded"]), {"alpha@example.com", "beta@example.com"})
        self.assertIn("WAKE_ACCOUNT_URL: alpha@example.com https://alpha.gradio.live", out)
        self.assertIn("WAKE_ACCOUNT_URL: beta@example.com https://beta.gradio.live", out)
        self.assertIn("WAKE_MULTI_JSON:", out)

    def test_auth_status_all_probes_all_accounts_concurrently_with_gpu_types(self):
        import ast
        import contextlib
        import hashlib
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with open(COLAB_BRIDGE_PATH, "r", encoding="utf-8") as f:
            bridge_src = f.read()
        tree = ast.parse(bridge_src, filename="colab_bridge.py")
        wanted_funcs = {
            "_account_token_filename",
            "_load_accounts_index",
            "_save_accounts_index",
            "_write_secure_file",
            "_save_token_to_vault",
            "_backup_current_token_to_vault",
            "_list_saved_accounts",
            "_probe_saved_account_live",
            "_refresh_all_saved_accounts_live",
            "cmd_auth_status_all",
        }
        selected = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in wanted_funcs
        ]
        mod = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(mod)

        with tempfile.TemporaryDirectory() as tmp:
            acc_dir = os.path.join(tmp, "accounts")
            idx_path = os.path.join(acc_dir, "index.json")
            default_tok = os.path.join(tmp, "default_token.json")

            class FakeCreds:
                def __init__(self, email_tag):
                    self.email_tag = email_tag
                    self.expired = False
                    self.refresh_token = "r"
                    self.valid = True

                def to_json(self):
                    return json.dumps({"tag": self.email_tag})

            class FakeClient:
                def __init__(self, _env, session):
                    self.tag = session.tag

                def get_consumption_user_info(self):
                    if self.tag == "b@example.com":
                        return SimpleNamespace(paid_compute_units_balance=2490.5, consumption_rate_hourly=15.8, assignments_count=1)
                    if self.tag == "c@example.com":
                        raise RuntimeError("revoked")
                    return SimpleNamespace(paid_compute_units_balance=2500.0, consumption_rate_hourly=0.0, assignments_count=0)

                def list_assignments(self):
                    if self.tag == "b@example.com":
                        return [SimpleNamespace(accelerator=SimpleNamespace(value="G4"))]
                    return []

            ns = {
                "os": os,
                "json": json,
                "time": time,
                "hashlib": hashlib,
                "ThreadPoolExecutor": ThreadPoolExecutor,
                "as_completed": as_completed,
                "PUBLIC_SCOPES": ["openid"],
                "TOKEN_CONFIG_PATH": default_tok,
                "ACCOUNTS_DIR": acc_dir,
                "ACCOUNTS_INDEX_PATH": idx_path,
                "Credentials": SimpleNamespace(
                    from_authorized_user_file=lambda path, _scopes: FakeCreds(json.load(open(path, "r", encoding="utf-8"))["tag"])
                ),
                "AuthorizedSession": lambda creds: SimpleNamespace(tag=creds.email_tag),
                "Request": lambda: object(),
                "Client": FakeClient,
                "Prod": lambda: "prod",
                "_has_valid_token": lambda: True,
                "_get_account_email": lambda: "a@example.com",
            }
            exec(compile(mod, "colab_bridge.py", "exec"), ns)

            for em in ("a@example.com", "b@example.com", "c@example.com"):
                ns["_save_token_to_vault"](em, json.dumps({"tag": em}), set_active=(em == "a@example.com"))
            with open(default_tok, "w", encoding="utf-8") as wf:
                json.dump({"tag": "a@example.com"}, wf)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ns["cmd_auth_status_all"]()
            payload = json.loads(buf.getvalue().strip())

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["email"], "a@example.com")
            self.assertEqual(payload["total_active_vms"], 1)
            self.assertAlmostEqual(payload["total_burn_rate_hourly"], 15.8, places=2)
            by_email = {item["email"]: item for item in payload["accounts"]}
            self.assertEqual(by_email["b@example.com"]["accelerators"], ["G4"])
            self.assertEqual(by_email["b@example.com"]["active_assignments"], 1)
            self.assertTrue(by_email["b@example.com"]["status_ok"])
            self.assertFalse(by_email["c@example.com"]["status_ok"])
            self.assertIn("revoked", by_email["c@example.com"]["error"])

    def test_wake_multi_auto_picks_live_idle_accounts_and_cli_switches_winner(self):
        ns = self._load_bridge_wake_multi_ns()
        spawned_accounts = []

        class FakeProc:
            def __init__(self, cmd):
                acct = cmd[cmd.index("--account") + 1]
                spawned_accounts.append(acct)
                self.stdout = iter([
                    f"WAKE_NEW_ALLOCATED: ep-{acct}\n",
                    "WAKE_URL: https://idlewin.gradio.live\n",
                ])

            def poll(self):
                return 0

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        ns["_refresh_all_saved_accounts_live"] = lambda _active="": [
            {"email": "busy@example.com", "active_assignments": 1, "status_ok": True},
            {"email": "bad@example.com", "active_assignments": 0, "status_ok": False},
            {"email": "idle1@example.com", "active_assignments": 0, "status_ok": True},
            {"email": "idle2@example.com", "active_assignments": 0, "status_ok": True},
        ]
        ns["subprocess"].Popen = lambda cmd, **kwargs: FakeProc(cmd)

        old_pwd = os.environ.get("EASYCOLAB_ADMIN_PASSWORD")
        os.environ["EASYCOLAB_ADMIN_PASSWORD"] = "TestAutoIdlePassword1234567890"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            res = ns["cmd_wake_multi"]([], "G4", "/tmp/app.py", mode="first")
        finally:
            sys.stdout = old_stdout
            if old_pwd is None:
                os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", None)
            else:
                os.environ["EASYCOLAB_ADMIN_PASSWORD"] = old_pwd

        self.assertTrue(res["ok"])
        self.assertEqual(set(spawned_accounts), {"idle1@example.com", "idle2@example.com"})

        # Verify Skill CLI switches default active account & writes default + scoped session on win
        cli_mod = _load_module(SKILL_CLI_REPO_PATH, "colab_h3_cli_wake_multi_test")
        bridge_calls = []

        def fake_run_bridge(bridge_args, env_extra=None, stream=False):
            bridge_calls.append(bridge_args)
            if bridge_args[0] == "wake_multi":
                return 0, "WAKE_WINNER: idle2@example.com\nWAKE_URL: https://idle2win.gradio.live\n"
            if bridge_args[0] == "auth_switch":
                return 0, json.dumps({"ok": True, "logged_in": True, "email": bridge_args[1]})
            return 0, "{}"

        with tempfile.TemporaryDirectory() as tmp:
            fake_home = cli_mod.Path(tmp)
            default_sess = fake_home / ".config" / "colab-minimax-h3" / "session.json"
            active_sess = fake_home / "active_session.json"
            args = SimpleNamespace(
                account="",
                accounts="",
                gpu="G4",
                mode="first",
                password="SecretCliPassword1234567890",
                idle_minutes=10,
                persistent=True,
                retry_interval=60,
            )
            with mock.patch.object(cli_mod.Path, "home", return_value=fake_home), \
                 mock.patch.object(cli_mod, "SESSION_PATH", default_sess), \
                 mock.patch.object(cli_mod, "ACTIVE_SESSION_PATH", active_sess), \
                 mock.patch.object(cli_mod, "_run_bridge", side_effect=fake_run_bridge), \
                 mock.patch.object(cli_mod, "_http_json", return_value={"ok": True}), \
                 mock.patch.object(cli_mod, "_emit_app_log", return_value=None):
                out_buf = io.StringIO()
                old_s = sys.stdout
                sys.stdout = out_buf
                try:
                    cli_mod.cmd_wake_multi(args)
                finally:
                    sys.stdout = old_s
                cli_res = json.loads(out_buf.getvalue())
                self.assertEqual(cli_res["winner"], "idle2@example.com")
                self.assertTrue(cli_res["switched_default_account"])
                self.assertIn(["auth_switch", "idle2@example.com"], bridge_calls)
                self.assertTrue(default_sess.is_file())
                self.assertEqual(json.loads(default_sess.read_text(encoding="utf-8"))["account"], "idle2@example.com")

    def test_recover_tunnel_reports_released_or_recovers_live_url_without_stopping_vm(self):
        import ast
        import contextlib

        with open(COLAB_BRIDGE_PATH, "r", encoding="utf-8") as f:
            bridge_src = f.read()
        tree = ast.parse(bridge_src, filename="colab_bridge.py")
        selected = [
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "cmd_recover_tunnel"
        ]
        mod = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(mod)

        # Case 1: Cloud has 0 assignments -> RECOVER_RELEASED without calling assign/unassign
        client_calls = []

        class FakeReleasedClient:
            def get_consumption_user_info(self):
                return SimpleNamespace(assignments_count=0, consumption_rate_hourly=0.0)

            def assign(self, *a, **kw):
                client_calls.append("assign")

            def unassign(self, *a, **kw):
                client_calls.append("unassign")

        fake_state = SimpleNamespace(
            client=FakeReleasedClient(),
            store=SimpleNamespace(add=lambda s: None),
            history=[],
        )
        ns = {
            "os": os,
            "re": __import__("re"),
            "time": __import__("time"),
            "uuid": __import__("uuid"),
            "tempfile": tempfile,
            "state": fake_state,
            "ContentsClient": lambda s: None,
            "ColabRuntime": lambda *a, **kw: None,
            "_has_valid_token": lambda: True,
            "_safe_sync_sessions": lambda: ({}, []),
            "_quiet_colab_cli": contextlib.nullcontext,
            "_session_from_listed_assignment": lambda a, session_name="easycolab", existing=None: None,
            "_wait_for_runtime_ready": lambda s, timeout_s=18: True,
        }
        exec(compile(mod, "colab_bridge.py", "exec"), ns)

        buf1 = io.StringIO()
        with contextlib.redirect_stdout(buf1):
            ns["cmd_recover_tunnel"]("/tmp/app.py", restart_server=False)
        self.assertIn("RECOVER_RELEASED:", buf1.getvalue())
        self.assertEqual(client_calls, [])

        # Case 2: Cloud G4 VM is running and read-only probe finds live Gradio URL -> RECOVER_URL
        class FakeRuntime:
            def __init__(self, *a, **kw):
                pass

            def execute_code(self, code, allow_stdin=False, timeout=120):
                return [{"text": "RECOVER_URL:https://reconnected42.gradio.live\n"}]

            def stop(self):
                pass

        assign_g4 = SimpleNamespace(endpoint="gpu-g4-live-01", accelerator=SimpleNamespace(value="G4"))
        sess_g4 = SimpleNamespace(name="easycolab", endpoint="gpu-g4-live-01", url="https://proxy.colab", token="tok", kernel_id=None, session_id=None)
        ns["_safe_sync_sessions"] = lambda: ({"easycolab": sess_g4}, [assign_g4])
        ns["_session_from_listed_assignment"] = lambda a, session_name="easycolab", existing=None: sess_g4
        ns["ColabRuntime"] = FakeRuntime

        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            ns["cmd_recover_tunnel"]("/tmp/app.py", restart_server=False)
        out2 = buf2.getvalue()
        self.assertIn("RECOVER_URL:https://reconnected42.gradio.live", out2)
        self.assertNotIn("RECOVER_ERROR:", out2)
        self.assertEqual(client_calls, [])


if __name__ == "__main__":
    unittest.main()
