import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
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
        wanted = {"_mask_bridge_log_text", "_append_external_app_log", "_external_cli_log_tee"}
        selected = [
            n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_APP_LOG_PATH" for t in n.targets))
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
            clean_env = {"EASYCOLAB_APP_LOG_PATH": log_file}
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


if __name__ == "__main__":
    unittest.main()
