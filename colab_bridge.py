#!/usr/bin/env python3
"""
Local Headless Google Colab Bridge for EasyColab macOS App.
Powered by google-colab-cli (inspired by FireTable/comfy-colab-kit).
Supports:
  - auth_status : Check OAuth login state, real CU balance, hourly burn rate, and active VMs
  - auth_url    : Generate Google OAuth2 authorization URL for 1-click browser login
  - auth_code   : Exchange Google authorization code and save ~/.config/colab-cli/token.json
  - wake        : Provision/reuse Colab GPU VM (G4/H100/A100/L4/T4), deploy app.py, and return gradio.live URL
  - keepalive   : Refresh Colab runtime proxy & kernel activity while active
  - stop_all    : Unassign and terminate all active Colab VMs on the account (verify 0 CU/hr burn)
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import uuid
import signal
import hashlib
import argparse
import tempfile
import threading
import subprocess
import contextlib
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from importlib import resources
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import websocket
from websocket import ABNF
from google.auth.transport.requests import Request, AuthorizedSession
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


def _ensure_utf8_stdio():
    """确保 Windows / macOS 子进程管道输出始终采用 UTF-8 编码，避免 emoji 在 GBK 代码页下抛出 UnicodeEncodeError"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_ensure_utf8_stdio()


def _sync_macos_system_proxies_to_env():
    """
    macOS / Linux 原生 GUI 应用启动子进程时默认可能不包含终端的 HTTP_PROXY / HTTPS_PROXY 环境变量，
    而 websocket-client (Jupyter 内核 WebSocket 通道) 仅从 os.environ 读取代理配置。
    此处自动读取 macOS / Windows / Linux 系统级代理 (scutil --proxy / Internet Settings / gsettings / urllib.request.getproxies) 并注入环境变量。
    """
    try:
        proxies = urllib.request.getproxies()
        for k in ("http", "https"):
            if k in proxies and proxies[k]:
                os.environ.setdefault(f"{k}_proxy", proxies[k])
                os.environ.setdefault(f"{k.upper()}_PROXY", proxies[k])
        if sys.platform.startswith("linux") and not (os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")):
            import subprocess
            mode = subprocess.run(
                ["gsettings", "get", "org.gnome.system.proxy", "mode"],
                capture_output=True, text=True, timeout=2
            ).stdout.strip().strip("'")
            if mode == "manual":
                for k in ("http", "https"):
                    host = subprocess.run(
                        ["gsettings", "get", f"org.gnome.system.proxy.{k}", "host"],
                        capture_output=True, text=True, timeout=2
                    ).stdout.strip().strip("'")
                    port_raw = subprocess.run(
                        ["gsettings", "get", f"org.gnome.system.proxy.{k}", "port"],
                        capture_output=True, text=True, timeout=2
                    ).stdout.strip()
                    port = port_raw.split()[-1].strip("'") if port_raw else ""
                    if host and port and port.isdigit() and int(port) > 0:
                        proxy_url = f"http://{host}:{port}"
                        os.environ.setdefault(f"{k}_proxy", proxy_url)
                        os.environ.setdefault(f"{k.upper()}_PROXY", proxy_url)
    except Exception:
        pass


_sync_macos_system_proxies_to_env()
os.environ.setdefault("COLAB_CLI_REQUEST_TIMEOUT", "75")

from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH, REMOTE_REDIRECT_URI
import colab_cli.auth as _colab_auth
from colab_cli.common import state
from colab_cli.client import (
    Accelerator,
    Client,
    Prod,
    Variant,
    Shape,
    resolve_assign_shape,
    PostAssignmentResponse,
    ColabRequestError,
    TooManyAssignmentsError,
)
from colab_cli.utils import get_status_code
from colab_cli.state import SessionState
from colab_cli.runtime import ColabRuntime
from colab_cli.contents import ContentsClient


def _load_client_config() -> dict:
    cfg_path = os.path.expanduser("~/.colab-cli-oauth-config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    res = resources.files("colab_cli").joinpath("oauth_config.json")
    return json.loads(res.read_text(encoding="utf-8"))


def _has_valid_token() -> bool:
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return False
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.valid:
            return True
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
            return True
    except Exception:
        return False
    return False


PKCE_VERIFIER_PATH = os.path.expanduser("~/.config/colab-cli/pkce_verifier.txt")
ACCOUNT_PROFILE_CACHE_PATH = os.path.expanduser("~/.config/colab-cli/account_profile.json")
ACCOUNTS_DIR = os.path.expanduser(
    os.environ.get("EASYCOLAB_ACCOUNTS_DIR", "").strip()
    or "~/.config/colab-cli/accounts"
)
ACCOUNTS_INDEX_PATH = os.path.join(ACCOUNTS_DIR, "index.json")
_APP_LOG_PATH = os.path.expanduser(
    os.environ.get("EASYCOLAB_APP_LOG_PATH", "").strip()
    or "~/Library/Application Support/EasyColab/external_cli_logs.jsonl"
)
_ACTIVE_SESSION_PATH = os.path.expanduser(
    os.environ.get("EASYCOLAB_ACTIVE_SESSION_PATH", "").strip()
    or "~/Library/Application Support/EasyColab/active_session.json"
)


def _sync_active_session_file(
    url: str,
    password: str,
    gpu: str = "",
    sleeping: bool = False,
    source: str = "colab-bridge",
    account: str = "",
) -> None:
    """Sync active/sleeping session to shared App Support JSON for EasyColab GUI real-time attach."""
    if os.environ.get("EASYCOLAB_FROM_APP") == "1" or os.environ.get("EASYCOLAB_SUPPRESS_BRIDGE_LOG") == "1":
        return
    try:
        target = os.path.expanduser(
            os.environ.get("EASYCOLAB_ACTIVE_SESSION_PATH", "").strip() or _ACTIVE_SESSION_PATH
        )
        os.makedirs(os.path.dirname(target), exist_ok=True)
        existing_pw = ""
        existing_comfy = ""
        if not sleeping and os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as rf:
                    prev = json.load(rf) or {}
                existing_pw = str(prev.get("password") or "")
                existing_comfy = str(prev.get("comfy_url") or "")
            except Exception:
                pass
        acct_val = (account or os.environ.get("EASYCOLAB_ACCOUNT", "")).strip()
        payload = {
            "url": (url or "").strip().rstrip("/"),
            "password": "" if sleeping else (password or existing_pw),
            "gpu": gpu or "",
            "comfy_url": "" if sleeping else existing_comfy,
            "account": acct_val,
            "sleeping": bool(sleeping),
            "source": source,
            "updated_at": round(time.time(), 3),
        }
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as wf:
            wf.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            wf.flush()
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
        os.chmod(target, 0o600)
    except Exception:
        pass


def _mask_bridge_log_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"https://([A-Za-z0-9-]{4})[A-Za-z0-9.-]*(\.gradio\.live)",
        r"https://\1***\2",
        text,
    )


def _append_external_app_log(line: str, level: str = "info") -> None:
    """When colab_bridge.py is called directly from external CLI/skills, stream logs into EasyColab app."""
    if os.environ.get("EASYCOLAB_FROM_APP") == "1" or os.environ.get("EASYCOLAB_SUPPRESS_BRIDGE_LOG") == "1":
        return
    try:
        clean = _mask_bridge_log_text(str(line or "")).strip()
        if not clean:
            return
        log_path = os.path.expanduser(
            os.environ.get("EASYCOLAB_APP_LOG_PATH", "").strip() or _APP_LOG_PATH
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if os.path.isfile(log_path) and os.path.getsize(log_path) > 512 * 1024:
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as rf:
                    tail = rf.read().splitlines()[-200:]
                with open(log_path, "w", encoding="utf-8") as wf:
                    wf.write("\n".join(tail) + "\n")
            except Exception:
                pass
        ts_str = time.strftime("%H:%M:%S")
        entry = {
            "ts": round(time.time(), 3),
            "source": "colab-bridge",
            "level": level,
            "line": f"[CLI {ts_str}] {clean}",
        }
        with open(log_path, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            wf.flush()
    except Exception:
        pass


@contextlib.contextmanager
def _external_cli_log_tee():
    """Tee WAKE_* / stop_all output to EasyColab app log bus when invoked outside the macOS GUI app."""
    if os.environ.get("EASYCOLAB_FROM_APP") == "1" or os.environ.get("EASYCOLAB_SUPPRESS_BRIDGE_LOG") == "1":
        yield
        return

    real_stdout = sys.stdout
    buf = []

    class _TeeStream:
        def write(self, s):
            real_stdout.write(s)
            buf.append(s)
            if "\n" in s:
                joined = "".join(buf)
                parts = joined.split("\n")
                buf[:] = [parts[-1]]
                for ln in parts[:-1]:
                    self._mirror(ln)
            return len(s)

        def flush(self):
            real_stdout.flush()

        def isatty(self):
            return getattr(real_stdout, "isatty", lambda: False)()

        def _mirror(self, ln: str):
            s = ln.strip()
            if not s:
                return
            if s.startswith("WAKE_LOG:"):
                _append_external_app_log(s[len("WAKE_LOG:"):].strip(), level="info")
            elif s.startswith("WAKE_URL:"):
                raw_url = s[len("WAKE_URL:"):].strip()
                _append_external_app_log(f"成功获取云端隧道地址: {raw_url}", level="info")
                _sync_active_session_file(
                    url=raw_url,
                    password=os.environ.get("EASYCOLAB_ADMIN_PASSWORD", ""),
                    gpu=os.environ.get("EASYCOLAB_ACTIVE_GPU", ""),
                    sleeping=False,
                    source="colab-bridge",
                )
            elif s.startswith("WAKE_ERROR:"):
                _append_external_app_log(s[len("WAKE_ERROR:"):].strip(), level="error")
            elif s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    msg = obj.get("message") or obj.get("error")
                    if msg:
                        lvl = "error" if obj.get("ok") is False else "info"
                        _append_external_app_log(str(msg), level=lvl)
                except Exception:
                    pass

    tee = _TeeStream()
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = real_stdout
        rem = "".join(buf).strip()
        if rem:
            tee._mirror(rem)


def _clear_local_session_store():
    """切换或退出 Google 账号时，清理本地缓存的旧账号 Colab 会话记录"""
    try:
        existing = state.store.list()
        if isinstance(existing, dict):
            for name in list(existing.keys()):
                state.store.remove(name)
    except Exception:
        pass
    state._client = None
    state._sessions = None


def _get_account_email() -> str:
    """获取当前绑定 Google 账号邮箱（按 refresh_token 指纹本地缓存，避免重复请求）"""
    if not os.path.exists(TOKEN_CONFIG_PATH):
        return ""
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        tok_fp = (creds.refresh_token or creds.token or "")[-20:]
        if tok_fp and os.path.exists(ACCOUNT_PROFILE_CACHE_PATH):
            try:
                with open(ACCOUNT_PROFILE_CACHE_PATH, "r", encoding="utf-8") as cf:
                    cached = json.load(cf)
                if cached.get("fp") == tok_fp and cached.get("email"):
                    return str(cached["email"])
            except Exception:
                pass
        if creds and creds.token:
            r = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=5,
            )
            if r.status_code == 200:
                email = str(r.json().get("email") or "").strip()
                if email and tok_fp:
                    os.makedirs(os.path.dirname(ACCOUNT_PROFILE_CACHE_PATH), exist_ok=True)
                    with open(ACCOUNT_PROFILE_CACHE_PATH, "w", encoding="utf-8") as cf:
                        json.dump({"fp": tok_fp, "email": email}, cf)
                return email
    except Exception:
        pass
    return ""


def _account_token_filename(email: str) -> str:
    norm = (email or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16] + ".json"


def _load_accounts_index() -> dict:
    if not os.path.isfile(ACCOUNTS_INDEX_PATH):
        return {"active_email": "", "accounts": {}}
    try:
        with open(ACCOUNTS_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"active_email": "", "accounts": {}}
        if not isinstance(data.get("accounts"), dict):
            data["accounts"] = {}
        return data
    except Exception:
        return {"active_email": "", "accounts": {}}


def _save_accounts_index(idx: dict) -> None:
    os.makedirs(ACCOUNTS_DIR, mode=0o700, exist_ok=True)
    try:
        os.chmod(ACCOUNTS_DIR, 0o700)
    except Exception:
        pass
    tmp_path = ACCOUNTS_INDEX_PATH + f".tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, ACCOUNTS_INDEX_PATH)


def _write_secure_file(path: str, content: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, mode=0o700, exist_ok=True)
    tmp_path = path + f".tmp_{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)


def _save_token_to_vault(
    email: str,
    token_json: str,
    paid_balance: float | None = None,
    burn_rate_hourly: float | None = None,
    active_assignments: int | None = None,
    accelerators: list[str] | None = None,
    status_ok: bool | None = None,
    error: str | None = None,
    set_active: bool = True,
) -> None:
    norm = (email or "").strip().lower()
    if not norm or not token_json:
        return
    os.makedirs(ACCOUNTS_DIR, mode=0o700, exist_ok=True)
    tok_file = _account_token_filename(norm)
    tok_path = os.path.join(ACCOUNTS_DIR, tok_file)
    _write_secure_file(tok_path, token_json)

    idx = _load_accounts_index()
    existing = idx["accounts"].get(norm) if isinstance(idx["accounts"].get(norm), dict) else {}
    entry = {
        "email": (email or "").strip(),
        "token_file": tok_file,
        "paid_balance": existing.get("paid_balance") if paid_balance is None else paid_balance,
        "burn_rate_hourly": existing.get("burn_rate_hourly") if burn_rate_hourly is None else burn_rate_hourly,
        "active_assignments": existing.get("active_assignments") if active_assignments is None else active_assignments,
        "accelerators": existing.get("accelerators", []) if accelerators is None else list(accelerators),
        "status_ok": existing.get("status_ok", True) if status_ok is None else bool(status_ok),
        "error": (None if status_ok else error) if status_ok is not None else existing.get("error"),
        "updated_at": round(time.time(), 3),
    }
    idx["accounts"][norm] = entry
    if set_active:
        idx["active_email"] = (email or "").strip()
    _save_accounts_index(idx)


def _backup_current_token_to_vault(
    email: str = "",
    paid_balance: float | None = None,
    burn_rate_hourly: float | None = None,
    active_assignments: int | None = None,
    accelerators: list[str] | None = None,
    status_ok: bool | None = None,
    error: str | None = None,
    set_active: bool = True,
) -> str:
    if not os.path.isfile(TOKEN_CONFIG_PATH):
        return ""
    try:
        with open(TOKEN_CONFIG_PATH, "r", encoding="utf-8") as f:
            token_json = f.read().strip()
        if not token_json:
            return ""
        resolved = (email or _get_account_email() or "").strip()
        if not resolved:
            return ""
        _save_token_to_vault(
            resolved,
            token_json,
            paid_balance=paid_balance,
            burn_rate_hourly=burn_rate_hourly,
            active_assignments=active_assignments,
            accelerators=accelerators,
            status_ok=status_ok,
            error=error,
            set_active=set_active,
        )
        return resolved
    except Exception:
        return ""


def _list_saved_accounts(active_email: str = "") -> list[dict]:
    idx = _load_accounts_index()
    active_norm = (active_email or "").strip().lower()
    items = []
    for norm, meta in (idx.get("accounts") or {}).items():
        if not isinstance(meta, dict):
            continue
        tok_file = str(meta.get("token_file") or _account_token_filename(norm))
        tok_path = os.path.join(ACCOUNTS_DIR, tok_file)
        if not os.path.isfile(tok_path):
            continue
        em = str(meta.get("email") or norm).strip()
        accels = meta.get("accelerators")
        items.append({
            "email": em,
            "is_active": bool(active_norm and em.lower() == active_norm),
            "paid_balance": meta.get("paid_balance"),
            "burn_rate_hourly": meta.get("burn_rate_hourly"),
            "active_assignments": meta.get("active_assignments"),
            "accelerators": list(accels) if isinstance(accels, list) else [],
            "status_ok": meta.get("status_ok", True),
            "error": meta.get("error"),
            "updated_at": meta.get("updated_at") or 0.0,
        })
    items.sort(
        key=lambda x: (
            not x["is_active"],
            0 if int(x.get("active_assignments") or 0) > 0 else 1,
            -(x.get("updated_at") or 0.0),
            x["email"].lower(),
        )
    )
    return items


def _probe_saved_account_live(email: str, tok_path: str) -> dict:
    """线程隔离地查询单个已保存账号的实时 CU 余额、消耗速率、活跃 VM 数与 GPU 型号，绝不切换全局状态或改动实例。"""
    try:
        creds = Credentials.from_authorized_user_file(tok_path, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _write_secure_file(tok_path, creds.to_json())
        if not creds or not creds.valid:
            return {
                "email": email,
                "status_ok": False,
                "error": "OAuth credentials invalid or expired",
            }
        client = Client(Prod(), AuthorizedSession(creds))
        info = client.get_consumption_user_info()
        try:
            assignments = client.list_assignments()
        except Exception:
            assignments = []
        accels = [
            a.accelerator.value
            for a in assignments
            if getattr(a, "accelerator", None) is not None
        ]
        vms = max(int(getattr(info, "assignments_count", 0) or 0), len(assignments))
        return {
            "email": email,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": vms,
            "accelerators": accels,
            "status_ok": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "email": email,
            "status_ok": False,
            "error": str(exc),
        }


def _refresh_all_saved_accounts_live(active_email: str = "") -> list[dict]:
    """并发探测保险箱内所有已保存账号的实时云端状态并回写索引（单账号失败不影响其余账号）。"""
    idx = _load_accounts_index()
    accounts_map = idx.get("accounts") or {}
    targets: list[tuple[str, str, str]] = []
    for norm, meta in accounts_map.items():
        if not isinstance(meta, dict):
            continue
        tok_file = str(meta.get("token_file") or _account_token_filename(norm))
        tok_path = os.path.join(ACCOUNTS_DIR, tok_file)
        if not os.path.isfile(tok_path):
            continue
        em = str(meta.get("email") or norm).strip()
        targets.append((norm, em, tok_path))

    if not targets:
        return []

    probed_by_norm: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(targets)))) as pool:
        future_map = {
            pool.submit(_probe_saved_account_live, em, tok_path): norm
            for norm, em, tok_path in targets
        }
        for fut in as_completed(future_map):
            norm = future_map[fut]
            try:
                probed_by_norm[norm] = fut.result()
            except Exception as exc:
                probed_by_norm[norm] = {"status_ok": False, "error": str(exc)}

    now_ts = round(time.time(), 3)
    for norm, res in probed_by_norm.items():
        meta = accounts_map.get(norm)
        if not isinstance(meta, dict):
            continue
        meta["updated_at"] = now_ts
        meta["status_ok"] = bool(res.get("status_ok"))
        meta["error"] = res.get("error")
        if res.get("status_ok"):
            meta["paid_balance"] = res.get("paid_balance")
            meta["burn_rate_hourly"] = res.get("burn_rate_hourly")
            meta["active_assignments"] = res.get("active_assignments")
            meta["accelerators"] = list(res.get("accelerators") or [])
    _save_accounts_index(idx)
    return _list_saved_accounts(active_email)


def _activate_scoped_account_context(account_email: str) -> str:
    """为多账号多开场景隔离指定账号的 Token、Profile 缓存与 sessions.json，避免并发互踩。"""
    target_norm = (account_email or "").strip().lower()
    if not target_norm:
        return ""
    idx = _load_accounts_index()
    meta = (idx.get("accounts") or {}).get(target_norm)
    if isinstance(meta, dict):
        tok_file = str(meta.get("token_file") or _account_token_filename(target_norm))
        canonical = str(meta.get("email") or account_email).strip()
    else:
        tok_file = _account_token_filename(target_norm)
        canonical = account_email.strip()
    tok_path = os.path.join(ACCOUNTS_DIR, tok_file)
    if not os.path.isfile(tok_path):
        raise RuntimeError(f"Saved account token not found in vault: {account_email}")

    stem = os.path.splitext(tok_file)[0]
    globals()["TOKEN_CONFIG_PATH"] = tok_path
    _colab_auth.TOKEN_CONFIG_PATH = tok_path
    globals()["ACCOUNT_PROFILE_CACHE_PATH"] = os.path.join(ACCOUNTS_DIR, f"{stem}_profile.json")
    state.config_path = os.path.join(ACCOUNTS_DIR, f"{stem}_sessions.json")
    state._store = None
    state._client = None
    state._sessions = None
    return canonical


def cmd_auth_url():
    client_config = _load_client_config()
    flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    # 显式传入 select_account consent，强制弹出 Google 多账号选择器以支持自由切换账号
    auth_url, _ = flow.authorization_url(prompt="select_account consent", token_usage="remote")
    os.makedirs(os.path.dirname(PKCE_VERIFIER_PATH), exist_ok=True)
    if getattr(flow, "code_verifier", None):
        with open(PKCE_VERIFIER_PATH, "w", encoding="utf-8") as f:
            f.write(flow.code_verifier)
    print(json.dumps({"ok": True, "auth_url": auth_url}))


def cmd_auth_code(code: str):
    try:
        client_config = _load_client_config()
        flow = InstalledAppFlow.from_client_config(client_config, PUBLIC_SCOPES)
        flow.redirect_uri = REMOTE_REDIRECT_URI
        if os.path.exists(PKCE_VERIFIER_PATH):
            with open(PKCE_VERIFIER_PATH, "r", encoding="utf-8") as f:
                flow.code_verifier = f.read().strip()
        flow.fetch_token(code=code.strip())
        creds = flow.credentials
        # 先把之前正在使用的账号备份进多账号保险箱，切换新账号后仍可随时切回
        _backup_current_token_to_vault(set_active=False)
        _clear_local_session_store()
        for p in (ACCOUNT_PROFILE_CACHE_PATH, PKCE_VERIFIER_PATH):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        _write_secure_file(TOKEN_CONFIG_PATH, creds.to_json())
        return cmd_auth_status()
    except Exception as e:
        print(json.dumps({
            "ok": False,
            "logged_in": _has_valid_token(),
            "accounts": _list_saved_accounts(""),
            "error": str(e),
        }))


def cmd_auth_logout():
    _backup_current_token_to_vault(set_active=False)
    idx = _load_accounts_index()
    idx["active_email"] = ""
    if idx.get("accounts"):
        _save_accounts_index(idx)
    _clear_local_session_store()
    for p in (TOKEN_CONFIG_PATH, PKCE_VERIFIER_PATH, ACCOUNT_PROFILE_CACHE_PATH):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    print(json.dumps({
        "ok": True,
        "logged_in": False,
        "email": "",
        "accounts": _list_saved_accounts(""),
    }))


def cmd_auth_list():
    active_email = ""
    if _has_valid_token():
        active_email = _backup_current_token_to_vault(set_active=True) or _get_account_email()
    print(json.dumps({
        "ok": True,
        "active_email": active_email,
        "accounts": _list_saved_accounts(active_email),
    }))


def cmd_auth_switch(target_email: str):
    target_norm = (target_email or "").strip().lower()
    if not target_norm:
        print(json.dumps({"ok": False, "logged_in": _has_valid_token(), "error": "Missing account email"}))
        return
    _backup_current_token_to_vault(set_active=False)
    idx = _load_accounts_index()
    meta = (idx.get("accounts") or {}).get(target_norm)
    if not isinstance(meta, dict):
        print(json.dumps({
            "ok": False,
            "logged_in": _has_valid_token(),
            "accounts": _list_saved_accounts(""),
            "error": f"Account not found: {target_email}",
        }))
        return
    tok_path = os.path.join(ACCOUNTS_DIR, str(meta.get("token_file") or _account_token_filename(target_norm)))
    if not os.path.isfile(tok_path):
        print(json.dumps({
            "ok": False,
            "logged_in": _has_valid_token(),
            "accounts": _list_saved_accounts(""),
            "error": f"Saved token file missing for {target_email}",
        }))
        return
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_json = f.read()
    _clear_local_session_store()
    if os.path.exists(ACCOUNT_PROFILE_CACHE_PATH):
        try:
            os.remove(ACCOUNT_PROFILE_CACHE_PATH)
        except Exception:
            pass
    _write_secure_file(TOKEN_CONFIG_PATH, tok_json)
    matched_email = str(meta.get("email") or target_email).strip()
    idx["active_email"] = matched_email
    meta["updated_at"] = round(time.time(), 3)
    _save_accounts_index(idx)
    return cmd_auth_status()


def cmd_auth_remove(target_email: str):
    target_norm = (target_email or "").strip().lower()
    cur_email = _get_account_email() if _has_valid_token() else ""
    idx = _load_accounts_index()
    accounts_map = idx.get("accounts") or {}
    removed_meta = accounts_map.pop(target_norm, None)
    if removed_meta and isinstance(removed_meta, dict):
        tok_path = os.path.join(ACCOUNTS_DIR, str(removed_meta.get("token_file") or _account_token_filename(target_norm)))
        if os.path.isfile(tok_path):
            try:
                os.remove(tok_path)
            except Exception:
                pass
    was_active = bool(cur_email and cur_email.lower() == target_norm)
    if was_active:
        idx["active_email"] = ""
    _save_accounts_index(idx)

    if was_active:
        _clear_local_session_store()
        for p in (TOKEN_CONFIG_PATH, ACCOUNT_PROFILE_CACHE_PATH):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        remaining = _list_saved_accounts("")
        if remaining:
            return cmd_auth_switch(remaining[0]["email"])
        print(json.dumps({"ok": True, "logged_in": False, "email": "", "accounts": []}))
        return

    if _has_valid_token():
        return cmd_auth_status()
    print(json.dumps({"ok": True, "logged_in": False, "email": "", "accounts": _list_saved_accounts("")}))


@contextlib.contextmanager
def _quiet_colab_cli():
    """将 colab_cli 内部 typer.echo (如 Pruned stale local session) 重定向至 stderr，保持 stdout 纯净 JSON 输出"""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _safe_sync_sessions():
    for attempt in range(3):
        try:
            with _quiet_colab_cli():
                # 重置 colab_cli 进程内缓存，确保每次都重新读取 sessions.json 并对比云端最新分配
                state._sessions = None
                return state.sync_sessions()
        except Exception:
            if attempt == 2:
                try:
                    with _quiet_colab_cli():
                        listed = state.store.list()
                except Exception:
                    listed = []
                if isinstance(listed, dict):
                    cached = dict(listed)
                else:
                    cached = {s.name: s for s in listed if getattr(s, "name", None)}
                return cached, []
            time.sleep(1.2)


def cmd_auth_status():
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False, "accounts": _list_saved_accounts("")}))
        return

    try:
        info = state.client.get_consumption_user_info()
        local_sessions, assignments = _safe_sync_sessions()
        name_by_ep = {s.endpoint: s.name for s in local_sessions.values()}
        sess_list = []
        for a in assignments:
            sess_list.append({
                "name": name_by_ep.get(a.endpoint, "colab-vm"),
                "endpoint": a.endpoint,
                "accelerator": a.accelerator.value,
                "variant": a.variant.name,
                "shape": a.machine_shape.name,
            })

        email = _get_account_email()
        bal = round(float(info.paid_compute_units_balance), 2)
        rate = round(float(info.consumption_rate_hourly), 2)
        vms = int(info.assignments_count)
        accels = [
            a.accelerator.value
            for a in assignments
            if getattr(a, "accelerator", None) is not None
        ]
        if email:
            _backup_current_token_to_vault(
                email=email,
                paid_balance=bal,
                burn_rate_hourly=rate,
                active_assignments=vms,
                accelerators=accels,
                status_ok=True,
                set_active=True,
            )
        print(json.dumps({
            "ok": True,
            "logged_in": True,
            "email": email,
            "paid_balance": bal,
            "burn_rate_hourly": rate,
            "active_assignments": vms,
            "sessions": sess_list,
            "accounts": _list_saved_accounts(email),
        }))
    except Exception as e:
        email = _get_account_email()
        if email:
            _backup_current_token_to_vault(email=email, set_active=True)
        print(json.dumps({
            "ok": False,
            "logged_in": True,
            "email": email,
            "accounts": _list_saved_accounts(email),
            "error": str(e),
        }))


def cmd_auth_status_all():
    """免切号并发查询所有已保存 Google 账号的实时 CU 余额、消耗速率与在线 GPU 实例状态。"""
    scoped_env = os.environ.get("EASYCOLAB_ACCOUNT", "").strip()
    focus_email = ""
    if _has_valid_token():
        focus_email = _backup_current_token_to_vault(set_active=not bool(scoped_env)) or _get_account_email()
    if scoped_env:
        focus_email = focus_email or scoped_env

    accounts = _refresh_all_saved_accounts_live(active_email=focus_email)
    focus_norm = (focus_email or "").strip().lower()
    focus_item = next(
        (a for a in accounts if (a.get("email") or "").strip().lower() == focus_norm),
        None,
    )

    total_active_vms = sum(int(a.get("active_assignments") or 0) for a in accounts)
    total_burn_rate = round(sum(float(a.get("burn_rate_hourly") or 0.0) for a in accounts), 2)
    total_paid_balance = round(sum(float(a.get("paid_balance") or 0.0) for a in accounts), 2)

    print(json.dumps({
        "ok": True,
        "logged_in": bool(focus_item) or _has_valid_token(),
        "email": (focus_item.get("email") if focus_item else focus_email) or "",
        "paid_balance": focus_item.get("paid_balance") if focus_item else None,
        "burn_rate_hourly": focus_item.get("burn_rate_hourly") if focus_item else None,
        "active_assignments": focus_item.get("active_assignments") if focus_item else 0,
        "accounts": accounts,
        "total_active_vms": total_active_vms,
        "total_burn_rate_hourly": total_burn_rate,
        "total_paid_balance": total_paid_balance,
        "checked_at": round(time.time(), 3),
    }))


ADC_PUBLIC_CLIENT_PREFIX = "764086051850-"


def _build_rclone_drive_conf(token_path: str = TOKEN_CONFIG_PATH) -> str:
    """
    当用户配置了自有 Google Cloud OAuth Client（或显式携带 quota_project_id）时，
    生成云端 rclone Google Drive 备用配置。
    注：colab-cli 内置的公共 ADC Client (764086051850-*) 未绑定 quota_project 时访问
    drive.googleapis.com 会被 Google 直接返回 403 SERVICE_DISABLED，此时跳过 rclone 配置，
    生成的成片由客户端自动拉取保存至本地电脑。
    """
    if not os.path.exists(token_path):
        return ""
    try:
        with open(token_path, "r", encoding="utf-8") as rf:
            pre_raw = json.load(rf)
        pre_cid = str(pre_raw.get("client_id") or "").strip()
        pre_qp = str(pre_raw.get("quota_project_id") or "").strip()
        if pre_cid.startswith(ADC_PUBLIC_CLIENT_PREFIX) and not pre_qp:
            return ""

        creds = Credentials.from_authorized_user_file(token_path, PUBLIC_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        raw = json.loads(creds.to_json())
        client_id = str(raw.get("client_id") or "").strip()
        quota_project = str(raw.get("quota_project_id") or "").strip()
        if client_id.startswith(ADC_PUBLIC_CLIENT_PREFIX) and not quota_project:
            return ""
        scopes = raw.get("scopes") or []
        drive_scope = "drive" if any(s.endswith("/drive") for s in scopes) else (
            "drive.file" if any("drive.file" in s for s in scopes) else ""
        )
        if not drive_scope or not raw.get("token") or not raw.get("refresh_token"):
            return ""
        tok_obj = {
            "access_token": raw["token"],
            "token_type": "Bearer",
            "refresh_token": raw["refresh_token"],
        }
        if raw.get("expiry"):
            tok_obj["expiry"] = raw["expiry"]
        lines = [
            "[easycolab_gdrive]",
            "type = drive",
            f"client_id = {client_id}",
            f"client_secret = {raw.get('client_secret', '')}",
            f"scope = {drive_scope}",
            f"token = {json.dumps(tok_obj, separators=(',', ':'))}",
            "",
        ]
        return "\n".join(lines)
    except Exception:
        return ""


def _unassign_endpoints_and_wait_quota_release(endpoints: list[str], wait_s: float = 8.0) -> int:
    """
    去重释放指定云端 endpoint 列表，并轮询 list_assignments 等待 Google 配额账本真正注销这些实例。
    避免前脚刚调 unassign、后脚立刻调 assign 时因 Google 后端秒级延迟导致 G4/H100 配额仍显示占用而报 503/412。
    """
    unique_eps = []
    seen = set()
    for ep in endpoints:
        if ep and ep not in seen:
            seen.add(ep)
            unique_eps.append(ep)
    if not unique_eps:
        return 0

    stopped_count = 0
    for ep in unique_eps:
        try:
            state.client.unassign(ep)
            stopped_count += 1
        except Exception:
            pass

    if wait_s > 0:
        deadline = time.time() + wait_s
        target_set = set(unique_eps)
        while time.time() < deadline:
            try:
                remaining = {a.endpoint for a in state.client.list_assignments()}
                if not (remaining & target_set):
                    break
            except Exception:
                break
            time.sleep(1.5)
    return stopped_count


def cmd_stop_all():
    if not _has_valid_token():
        print(json.dumps({"ok": False, "error": "尚未登录 Google Colab CLI"}))
        return

    try:
        local_sessions, assignments = _safe_sync_sessions()
        endpoints_to_release: list[str] = []
        for s in list(local_sessions.values()):
            try:
                with _quiet_colab_cli():
                    rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
                    try:
                        rt.execute_code(
                            "import os, urllib.request\n"
                            "try:\n"
                            "    urllib.request.urlopen(urllib.request.Request("
                            "'http://127.0.0.1:27860/minimax_api/comfyui/sync_drive', "
                            "data=b'{}', headers={'Content-Type':'application/json'}, method='POST'), timeout=18)\n"
                            "except Exception:\n"
                            "    pass\n"
                            "try:\n"
                            "    os.sync()\n"
                            "except Exception:\n"
                            "    pass\n"
                            "try:\n"
                            "    if os.path.isdir('/content/drive/MyDrive'):\n"
                            "        from google.colab import drive\n"
                            "        drive.flush_and_unmount()\n"
                            "except Exception:\n"
                            "    pass\n",
                            allow_stdin=False,
                            timeout=30,
                        )
                    except Exception:
                        pass
                    rt.stop(shutdown_kernel=True)
            except Exception:
                pass
            if getattr(s, "endpoint", None):
                endpoints_to_release.append(s.endpoint)
            with _quiet_colab_cli():
                state.store.remove(s.name)

        # 同时清理所有未在本地登记的云端孤儿实例 (Orphan VMs)，去重并等待配额归零
        for a in assignments:
            if getattr(a, "endpoint", None):
                endpoints_to_release.append(a.endpoint)

        stopped_count = _unassign_endpoints_and_wait_quota_release(endpoints_to_release, wait_s=8.0)
        _sync_active_session_file("", "", sleeping=True, source="colab-bridge")

        info = state.client.get_consumption_user_info()
        print(json.dumps({
            "ok": True,
            "stopped_count": stopped_count,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
            "message": f"🛑 已彻底释放全部云端实例！当前计费速率: {info.consumption_rate_hourly:.2f} CU/小时",
        }))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))


def _build_bridge_subcommand(args: list[str]) -> list[str]:
    """Build subprocess argv targeting colab_bridge CLI in both source and frozen builds."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--bridge-cli", *args]
    return [sys.executable, os.path.abspath(__file__), *args]


def cmd_unassign_endpoint(endpoint: str):
    """Unassign only a single specified Colab endpoint without touching other VMs on the account."""
    ep = (endpoint or "").strip()
    if not ep:
        print(json.dumps({"ok": False, "error": "endpoint is empty"}))
        return
    if not _has_valid_token():
        print(json.dumps({"ok": False, "error": "尚未登录 Google Colab CLI"}))
        return
    try:
        try:
            with _quiet_colab_cli():
                local_sessions, _ = _safe_sync_sessions()
                for s in list(local_sessions.values()):
                    if getattr(s, "endpoint", None) == ep:
                        state.store.remove(s.name)
        except Exception:
            pass
        stopped_count = _unassign_endpoints_and_wait_quota_release([ep], wait_s=5.0)
        print(json.dumps({"ok": True, "endpoint": ep, "stopped_count": stopped_count}))
    except Exception as e:
        print(json.dumps({"ok": False, "endpoint": ep, "error": str(e)}))


def _unassign_single_endpoint_for_account(account_email: str, endpoint: str) -> None:
    """Best-effort cleanup for a duplicate VM allocated during a multi-account race."""
    ep = (endpoint or "").strip()
    acct = (account_email or "").strip()
    if not ep or not acct:
        return
    env = os.environ.copy()
    env["EASYCOLAB_ACCOUNT"] = acct
    env["EASYCOLAB_SUPPRESS_BRIDGE_LOG"] = "1"
    env.pop("EASYCOLAB_FROM_APP", None)
    try:
        subprocess.run(
            _build_bridge_subcommand(["--account", acct, "unassign_endpoint", ep]),
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        print(f"WAKE_LOG: ⚠️ [{acct}] 清理并发多余实例 ({ep[:16]}...) 异常: {exc}", flush=True)


def _wait_for_runtime_ready(session: SessionState, timeout_s: int = 45) -> bool:
    """
    等待新分配的 Colab 容器内部 Jupyter Server (/api/kernels) 完成启动。
    规避容器刚启动前几秒出现的 SSLEOFError / ConnectionReset / 502 / 503。
    若检测到 404 (实例如已被销毁)，立即返回 False 以便重新分配新实例。
    """
    base_url = session.url.rstrip("/")
    probe_url = f"{base_url}/api/kernels"
    headers = {
        "Accept": "application/json",
        "X-Colab-Client-Agent": "colab-cli",
        "X-Colab-Runtime-Proxy-Token": session.token,
        "Authorization": f"Bearer {session.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": session.token}
    deadline = time.time() + timeout_s
    not_found_count = 0

    while time.time() < deadline:
        try:
            r = requests.get(probe_url, headers=headers, params=params, timeout=8)
            if r.status_code == 200:
                return True
            if r.status_code in (404, 410):
                not_found_count += 1
                if not_found_count >= 2:
                    return False
        except requests.exceptions.RequestException:
            pass
        time.sleep(2.0)
    return False


GPU_ACCEL_MAP = {
    "h100": Accelerator.H100,
    "g4": Accelerator.G4,
    "a100": Accelerator.A100,
    "l4": Accelerator.L4,
    "t4": Accelerator.T4,
}


def _resolve_gpu_accel(gpu_type: str) -> Accelerator:
    return GPU_ACCEL_MAP.get((gpu_type or "g4").strip().lower(), Accelerator.G4)


def _gpu_display_label(accel: Accelerator) -> str:
    if accel == Accelerator.G4:
        return "G4 (RTX PRO 6000 · 96GB)"
    return accel.value


def _candidate_shapes_for_accel(accel: Accelerator) -> list:
    """
    为各型号 GPU 提供主备机器内存规格 (Shape) 候选序列：
    - G4 / H100：首选不传 shape (默认大内存标准池，避免部分区域对 shape=hm 报 400)，若遇 503 库存不足自动尝试 HIGH_RAM 池；
    - A100 / T4：首选 HIGH_RAM 大内存池，若遇 400/503 自动降级尝试标准内存池；
    - L4：官方固定单规格 (HIGH_MEM_ONLY_ACCELERATORS)，仅使用 None。
    """
    if accel == Accelerator.L4:
        return [None]
    if accel in (Accelerator.G4, Accelerator.H100):
        return [None, Shape.HIGH_RAM]
    primary = resolve_assign_shape(accel, high_mem=True)
    return [primary, None] if primary is not None else [None]


def _session_from_listed_assignment(a, session_name: str = "easycolab", existing: SessionState | None = None) -> SessionState:
    return SessionState(
        name=existing.name if existing and getattr(existing, "name", None) else session_name,
        token=a.runtime_proxy_info.token,
        url=a.runtime_proxy_info.url,
        endpoint=a.endpoint,
        token_expires_at=a.runtime_proxy_info.expires_at(),
        kernel_id=getattr(existing, "kernel_id", None) if existing and existing.endpoint == a.endpoint else None,
        session_id=(
            getattr(existing, "session_id", None)
            if existing and existing.endpoint == a.endpoint and getattr(existing, "session_id", None)
            else uuid.uuid4().hex
        ),
        variant=Variant.GPU.value,
        accelerator=a.accelerator.value,
        machine_shape=Shape.HIGH_RAM.name if a.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
    )


def _try_adopt_matching_cloud_assignment(accel: Accelerator, session_name: str = "easycolab") -> SessionState | None:
    """
    当 POST /tun/m/assign 出现网关 504 / ReadTimeout / 503 重试时，立即核对云端真实 list_assignments()：
    若 Google 后端实际上已在后台完成该型号机器分配，直接纳管返回，杜绝产生孤儿机器占满高配 GPU 名额。
    """
    try:
        assignments = state.client.list_assignments()
    except Exception:
        return None
    for a in assignments:
        if a.accelerator == accel:
            s = _session_from_listed_assignment(a, session_name=session_name)
            with _quiet_colab_cli():
                state.store.add(s)
            return s
    return None


@contextlib.contextmanager
def _single_shot_colab_assign():
    """禁用 colab_cli 内部静默阻塞 70 秒的英文重试循环，交由 EasyColab 透明重试并实时对账云端分配"""
    prev_retries = os.environ.get("COLAB_CLI_ASSIGN_RETRIES")
    os.environ["COLAB_CLI_ASSIGN_RETRIES"] = "0"
    try:
        yield
    finally:
        if prev_retries is None:
            os.environ.pop("COLAB_CLI_ASSIGN_RETRIES", None)
        else:
            os.environ["COLAB_CLI_ASSIGN_RETRIES"] = prev_retries


def _wait_before_next_allocation_round(
    accel: Accelerator,
    gpu_desc: str,
    session_name: str,
    round_num: int,
    max_attempts: int,
    last_err_tag: str,
    retry_interval: float,
) -> SessionState | None:
    """在连续 3 次申请失败后等待指定间隔（默认 60 秒），期间允许外部 SIGTERM 及时中断并在结束时核对云端分配。"""
    wait_s = max(0.01, float(retry_interval or 60.0))
    print(
        f"WAKE_LOG: ⏳ 第 {round_num} 轮连续 {max_attempts} 次申请 {gpu_desc} 均未分到机器 ({last_err_tag})，"
        f"将在 {wait_s:.0f} 秒后自动开始第 {round_num + 1} 轮持续申请（点击「停止」可随时取消）...",
        flush=True,
    )
    deadline = time.monotonic() + wait_s
    next_tick = time.monotonic() + 15.0
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            break
        if wait_s >= 30.0 and now >= next_tick and remaining >= 5.0:
            print(
                f"WAKE_LOG: ⏳ 持续申请等待中：还有 {int(round(remaining))} 秒开始第 {round_num + 1} 轮 {gpu_desc} 申请...",
                flush=True,
            )
            next_tick = now + 15.0
        time.sleep(min(1.0, remaining))
    adopted = _try_adopt_matching_cloud_assignment(accel, session_name=session_name)
    if adopted is not None:
        print(f"WAKE_LOG: ✅ 等待期间检测到云端已完成 {accel.value} 实例分配，直接接管！", flush=True)
    return adopted


def _allocate_fresh_session(
    session_name: str,
    gpu_type: str,
    persistent: bool = False,
    retry_interval: float = 60.0,
) -> SessionState:
    accel = _resolve_gpu_accel(gpu_type)
    gpu_desc = _gpu_display_label(accel)
    shape_candidates = list(_candidate_shapes_for_accel(accel))
    rejected_400_shapes = set()

    mode_hint = f"（已开启持续申请：每轮 3 次失败后隔 {int(retry_interval)}s 自动续申）" if persistent else ""
    print(f"WAKE_LOG: 🚀 正在向 Google Colab 申请分配 {gpu_desc} 实例{mode_hint}...", flush=True)

    max_attempts = 3
    retry_delays = [6.0, 10.0]
    last_err_tag = "HTTP 503"
    res = None
    chosen_shape = shape_candidates[0]
    round_num = 1

    while True:
        nbh_by_shape = {sh: uuid.uuid4() for sh in shape_candidates}
        attempt = 0
        shape_idx = 0
        cleaned_412_orphans = False

        while attempt < max_attempts:
            valid_shapes = [sh for sh in shape_candidates if sh not in rejected_400_shapes] or [None]
            chosen_shape = valid_shapes[shape_idx % len(valid_shapes)]
            nbh = nbh_by_shape.setdefault(chosen_shape, uuid.uuid4())

            try:
                with _quiet_colab_cli(), _single_shot_colab_assign():
                    res = state.client.assign(nbh, variant=Variant.GPU, accelerator=accel, shape=chosen_shape)
                break
            except TooManyAssignmentsError as quota_err:
                adopted = _try_adopt_matching_cloud_assignment(accel, session_name=session_name)
                if adopted is not None:
                    print(f"WAKE_LOG: ✅ 检测到云端已在后台完成 {accel.value} 实例分配，直接接管！", flush=True)
                    return adopted
                if not cleaned_412_orphans:
                    cleaned_412_orphans = True
                    try:
                        lingering = [a.endpoint for a in state.client.list_assignments() if a.endpoint]
                    except Exception:
                        lingering = []
                    if lingering:
                        print("WAKE_LOG: 🧹 检测到账号下残留未释放实例占用并发名额，正在自动清理后重试...", flush=True)
                        _unassign_endpoints_and_wait_quota_release(lingering, wait_s=8.0)
                        continue
                last_err_tag = "HTTP 412 配额占满"
                attempt += 1
                if attempt < max_attempts:
                    delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                    print(
                        f"WAKE_LOG: ⏳ 第 {round_num} 轮 {gpu_desc} 配额受限 (HTTP 412)，{delay:.0f} 秒后进行第 {attempt + 1}/{max_attempts} 次申请...",
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                if not persistent:
                    raise RuntimeError(
                        f"Google Colab 拒绝分配 {gpu_desc} (HTTP 412)：当前账号并发实例数已达上限，或该时段高配机型触发临时配额限制。"
                        f"请确认未在其他浏览器标签页占用实例，稍后重试或临时切换为 A100 / L4。"
                    ) from quota_err
            except (ColabRequestError, requests.exceptions.RequestException) as req_err:
                status = get_status_code(req_err) if isinstance(req_err, ColabRequestError) else None

                # 1) 无论报 504/503/ReadTimeout，第一时间检查云端是否其实已经分配成功
                adopted = _try_adopt_matching_cloud_assignment(accel, session_name=session_name)
                if adopted is not None:
                    print(f"WAKE_LOG: ✅ 检测到云端已在后台完成 {accel.value} 实例分配，直接接管！", flush=True)
                    return adopted

                # 2) 若当前机器规格报 400 Bad Request（如部分池不支持 shape=hm），立即切换候选规格且不消耗重试次数
                if status == 400 and chosen_shape is not None and None not in rejected_400_shapes:
                    rejected_400_shapes.add(chosen_shape)
                    shape_idx = 0
                    continue
                if status == 400:
                    raise RuntimeError(
                        f"Google Colab 后端拒绝分配 {gpu_desc} (HTTP 400)：当前账号订阅等级可能暂不支持该机型或可用计算单元不足，"
                        f"建议切换为 A100 / L4 / T4 实例。"
                    ) from req_err

                retryable = isinstance(req_err, requests.exceptions.RequestException) or status in (429, 500, 502, 503, 504)
                if not retryable:
                    raise

                last_err_tag = f"HTTP {status}" if status else type(req_err).__name__
                attempt += 1
                shape_idx += 1
                if attempt >= max_attempts:
                    break

                delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                print(
                    f"WAKE_LOG: ⏳ 第 {round_num} 轮 Google Colab {gpu_desc} 算力池繁忙 ({last_err_tag})，"
                    f"{delay:.0f} 秒后进行第 {attempt + 1}/{max_attempts} 次申请...",
                    flush=True,
                )
                time.sleep(delay)
                adopted_after_wait = _try_adopt_matching_cloud_assignment(accel, session_name=session_name)
                if adopted_after_wait is not None:
                    print(f"WAKE_LOG: ✅ 检测到云端已完成 {accel.value} 实例分配，直接接管！", flush=True)
                    return adopted_after_wait

        if res is not None:
            break
        if not persistent:
            break
        adopted_next = _wait_before_next_allocation_round(
            accel=accel,
            gpu_desc=gpu_desc,
            session_name=session_name,
            round_num=round_num,
            max_attempts=max_attempts,
            last_err_tag=last_err_tag,
            retry_interval=retry_interval,
        )
        if adopted_next is not None:
            return adopted_next
        round_num += 1

    if res is None:
        raise RuntimeError(
            f"Google Colab 云端当前时段 {gpu_desc} 算力池库存不足 ({last_err_tag})，连续 {max_attempts} 次申请均未分到机器"
            f"（属于 Colab 官方高配机型时段性缺货，非本地客户端故障）。建议稍等几分钟再试，或在左侧临时切换为 A100 / L4 实例。"
        )

    if isinstance(res, PostAssignmentResponse):
        token = res.runtime_proxy_info.token
        url = res.runtime_proxy_info.url
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at()
    else:
        token = res.runtime_proxy_info.token if hasattr(res, "runtime_proxy_info") else getattr(res, "runtime_proxy_token", "")
        url = res.runtime_proxy_info.url if hasattr(res, "runtime_proxy_info") else ""
        endpoint = res.endpoint
        exp = res.runtime_proxy_info.expires_at() if hasattr(res, "runtime_proxy_info") else None

    s = SessionState(
        name=session_name,
        token=token,
        url=url,
        endpoint=endpoint,
        token_expires_at=exp,
        session_id=uuid.uuid4().hex,
        variant=Variant.GPU.value,
        accelerator=accel.value,
        machine_shape=Shape.HIGH_RAM.name if chosen_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
    )
    with _quiet_colab_cli():
        state.store.add(s)
    print(f"WAKE_LOG: ✅ 成功分配 {accel.value} 云端虚拟机！正在等待容器内核就绪...", flush=True)
    return s


def cmd_wake(
    gpu_type: str,
    app_py_path: str,
    persistent: bool = False,
    retry_interval: float = 60.0,
):
    """
    无浏览器一键唤醒 Colab GPU 并自动部署 EasyColab 多模型服务端：
    1. 全量云端分配对账：遍历账号下全部云端 assignments（含未在本地登记的孤儿实例），优先复用同型号健康实例，
       并彻底释放异构/残留孤儿实例且等待配额归零，杜绝孤儿机器占满 G4/H100 单实例配额导致 503/412
    2. 等待容器 Jupyter 端点 (/api/kernels) TLS/HTTP 握手就绪，规避 SSLEOFError
    3. 上传本地最新 app.py 到 /content/app.py
    4. 在远端内核安装依赖并后台拉起 app.py，提取 https://xxxx.gradio.live 隧道链接返回给 Mac 客户端
    """
    if not _has_valid_token():
        print("WAKE_ERROR: 尚未授权登录 Google 账号，请先点击「🔑 绑定账号」", flush=True)
        return

    import getpass
    password = os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "")
    raw_idle_min = os.environ.get("EASYCOLAB_IDLE_TIMEOUT_MINUTES", "").strip()
    try:
        initial_idle_minutes = int(raw_idle_min) if raw_idle_min != "" else None
    except ValueError:
        initial_idle_minutes = None
    if not password and sys.stdin.isatty():
        password = getpass.getpass("Management password (24+ URL-safe characters): ")
    if not password:
        print("WAKE_ERROR: Set a management password before deploying", flush=True)
        return

    try:
        session_name = "easycolab"
        requested_accel = _resolve_gpu_accel(gpu_type)
        print("WAKE_LOG: 🔍 正在检查账号下已有的 Colab 实例...", flush=True)
        local_sessions, assignments = _safe_sync_sessions()
        ep_to_local = {
            s.endpoint: s for s in local_sessions.values() if getattr(s, "endpoint", None)
        }

        preferred_local = (
            local_sessions.get(session_name)
            or local_sessions.get("aevix")
            or local_sessions.get("minimax")
            or (list(local_sessions.values())[0] if local_sessions else None)
        )
        if preferred_local and getattr(preferred_local, "name", None):
            session_name = preferred_local.name

        target_session: SessionState | None = None
        matching_assignments = [a for a in assignments if a.accelerator == requested_accel]
        other_assignments = [a for a in assignments if a.accelerator != requested_accel]
        dead_matching_eps: list[str] = []

        # 1) 优先在账号全部云端活跃分配中查找与目标 GPU 型号一致的健康实例（包含此前超时遗留但实际已创建成功的孤儿实例）
        for a in matching_assignments:
            existing_local = ep_to_local.get(a.endpoint)
            candidate = _session_from_listed_assignment(a, session_name=session_name, existing=existing_local)
            print(
                f"WAKE_LOG: ♻️ 正在检查已有实例 [{candidate.name}] ({a.accelerator.value}) 连通性...",
                flush=True,
            )
            if _wait_for_runtime_ready(candidate, timeout_s=18):
                target_session = candidate
                with _quiet_colab_cli():
                    state.store.add(target_session)
                break
            else:
                print(f"WAKE_LOG: ⚠️ 旧实例 ({a.endpoint[:18]}...) 已失效或正在销毁，准备清理...", flush=True)
                dead_matching_eps.append(a.endpoint)

        newly_allocated = (target_session is None)
        if target_session is not None:
            # 已复用目标型号实例：顺带释放重复同型号孤儿实例及本地记录过的旧异构实例，防止双倍扣费
            extra_eps = list(dead_matching_eps)
            for a in matching_assignments:
                if a.endpoint != target_session.endpoint:
                    extra_eps.append(a.endpoint)
            for s in list(local_sessions.values()):
                if s.endpoint and s.endpoint != target_session.endpoint:
                    extra_eps.append(s.endpoint)
                if s.name != target_session.name:
                    with _quiet_colab_cli():
                        state.prune_session(s.name)
            if extra_eps:
                _unassign_endpoints_and_wait_quota_release(extra_eps, wait_s=5.0)
        else:
            # 2) 未找到可复用的目标型号实例：释放账号下占用的旧型号/失效孤儿实例并等待配额账本归零，再申请新机器
            release_eps = list(dead_matching_eps) + [a.endpoint for a in other_assignments if a.endpoint]
            for s in list(local_sessions.values()):
                if getattr(s, "endpoint", None):
                    release_eps.append(s.endpoint)
                with _quiet_colab_cli():
                    state.prune_session(s.name)

            if release_eps:
                old_labels = sorted({a.accelerator.value for a in other_assignments} | (
                    {requested_accel.value} if dead_matching_eps else set()
                ))
                label_str = " / ".join(old_labels) if old_labels else "旧实例"
                print(
                    f"WAKE_LOG: 🔄 当前已占用实例 [{label_str}]，正在彻底释放并等待配额就绪以切换为 [{requested_accel.value}]...",
                    flush=True,
                )
                _unassign_endpoints_and_wait_quota_release(release_eps, wait_s=8.0)

            use_persistent = bool(persistent) or (
                os.environ.get("EASYCOLAB_PERSISTENT_WAKE", "").strip().lower() in ("1", "true", "yes")
            )
            env_interval = os.environ.get("EASYCOLAB_WAKE_RETRY_INTERVAL", "").strip()
            eff_interval = float(env_interval) if env_interval else float(retry_interval or 60.0)
            target_session = _allocate_fresh_session(
                session_name,
                gpu_type,
                persistent=use_persistent,
                retry_interval=eff_interval,
            )
            if not _wait_for_runtime_ready(target_session, timeout_s=50):
                print("WAKE_LOG: ⏳ 容器仍在初始化，继续尝试连接内核...", flush=True)

        if getattr(target_session, "endpoint", None):
            sig_prefix = "WAKE_NEW_ALLOCATED" if newly_allocated else "WAKE_ALLOCATED"
            print(f"{sig_prefix}: {target_session.endpoint}", flush=True)
        print("WAKE_LOG: 正在上传最新多模型开源服务端代码 (app.py)...", flush=True)
        contents = ContentsClient(target_session)
        for up_attempt in range(4):
            try:
                contents.upload(app_py_path, "content/app.py")
                break
            except Exception as up_err:
                if up_attempt == 3:
                    raise up_err
                time.sleep(2.5)

        contents.upload(os.path.join(os.path.dirname(app_py_path), "runtime_auth.py"), "content/runtime_auth.py")
        rclone_conf_text = _build_rclone_drive_conf()
        if rclone_conf_text:
            tmp_rc_fd, tmp_rc_path = tempfile.mkstemp(prefix="easycolab_rc_", suffix=".conf")
            try:
                with os.fdopen(tmp_rc_fd, "w", encoding="utf-8") as rcf:
                    rcf.write(rclone_conf_text)
                os.chmod(tmp_rc_path, 0o600)
                contents.upload(tmp_rc_path, "content/.easycolab_rclone.conf")
            except Exception:
                pass
            finally:
                if os.path.exists(tmp_rc_path):
                    try:
                        os.remove(tmp_rc_path)
                    except Exception:
                        pass

        print("WAKE_LOG: 正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...", flush=True)
        boot_code = r"""
import os, sys, time, re, shutil, subprocess

with open(ADMIN_STAGE, encoding="ascii") as stream:
    admin_password = stream.read()
os.unlink(ADMIN_STAGE)

if not shutil.which("nginx"):
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "nginx-light"], check=True)
    # apt may start the distribution's default site.
    subprocess.run(["service", "nginx", "stop"], check=False)

# 0. 清理旧实例可能残留的公共 ADC 失效 rclone 配置；若未通过原生 DriveFS 挂载且用户配置了自有 OAuth rclone，则启用备用挂载
rc_existing = os.path.expanduser("~/.config/rclone/rclone.conf")
rc_stage = "/content/.easycolab_rclone.conf"
if os.path.exists(rc_stage):
    rc_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(rc_dir, mode=0o700, exist_ok=True)
    shutil.move(rc_stage, rc_existing)
    os.chmod(rc_existing, 0o600)
elif os.path.exists(rc_existing):
    try:
        rc_txt = open(rc_existing, "r", encoding="utf-8", errors="ignore").read()
        if "client_id = 764086051850-" in rc_txt:
            os.remove(rc_existing)
    except Exception:
        pass

_native_drive_ok = (
    os.path.isdir("/content/drive/MyDrive")
    and os.path.exists("/content")
    and os.stat("/content/drive/MyDrive").st_dev != os.stat("/content").st_dev
)
if not _native_drive_ok and os.path.exists(rc_existing):
    if not shutil.which("rclone"):
        subprocess.run(
            ["bash", "-lc", "curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rclone.zip && unzip -q -o /tmp/rclone.zip -d /tmp && cp /tmp/rclone-*-linux-amd64/rclone /usr/local/bin/rclone && chmod +x /usr/local/bin/rclone && rm -rf /tmp/rclone*"],
            check=False,
        )
    if shutil.which("rclone") and not os.path.ismount("/content/drive/MyDrive"):
        mk_res = subprocess.run(["rclone", "mkdir", "easycolab_gdrive:EasyColab_Drive"], check=False)
        if mk_res.returncode == 0 and os.path.exists("/dev/fuse") and (shutil.which("fusermount3") or shutil.which("fusermount")):
            os.makedirs("/content/drive/MyDrive", exist_ok=True)
            subprocess.Popen(
                [
                    "rclone", "mount", "easycolab_gdrive:EasyColab_Drive", "/content/drive/MyDrive",
                    "--vfs-cache-mode", "writes",
                    "--vfs-write-back", "3s",
                    "--dir-cache-time", "15s",
                    "--drive-chunk-size", "64M",
                    "--buffer-size", "32M",
                    "--allow-non-empty",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(10):
                if os.path.ismount("/content/drive/MyDrive"):
                    print("✅ 已通过自定义 OAuth 自动挂载 Google Drive (/content/drive/MyDrive)！", flush=True)
                    break
                time.sleep(0.5)
            else:
                try:
                    if not os.listdir("/content/drive/MyDrive"):
                        os.rmdir("/content/drive/MyDrive")
                except Exception:
                    pass

# 1. 如果尚未安装 DiffSynth-Studio，自动快速安装 (含最新 transformers 以支持 Qwen3.8-27B)
if not os.path.exists("/content/DiffSynth-Studio"):
    print("正在克隆 DiffSynth-Studio...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/modelscope/DiffSynth-Studio.git", "/content/DiffSynth-Studio"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "bitsandbytes", "huggingface_hub", "hf_transfer", "qwen-vl-utils", "einops", "kernels>=0.17.2,<0.18.0", "av", "gradio", "modelscope", "imageio", "imageio-ffmpeg", "scipy", "soundfile", "librosa"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fla-core", "flash-linear-attention"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--only-binary=:all:", "causal-conv1d"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/DiffSynth-Studio"], check=True)
else:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        import kernels  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "kernels>=0.17.2,<0.18.0"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", "fla-core", "flash-linear-attention"], check=False)

# 2. 检查是否已有运行中的 app.py 或 llama-server，先清理旧进程释放 GPU 显存
subprocess.run(["pkill", "-f", "python.*/content/app.py"], check=False)
subprocess.run(["pkill", "-f", "llama-server"], check=False)
time.sleep(1)

# 3. 后台启动 /content/app.py 并等待 gradio.live 链接生成
log_path = "/content/easycolab_server.log"
if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "w") as out_f:
    child_env = {**os.environ, "EASYCOLAB_ADMIN_PASSWORD": admin_password}
    if INITIAL_IDLE_MINUTES is not None:
        child_env["EASYCOLAB_IDLE_TIMEOUT_MINUTES"] = str(INITIAL_IDLE_MINUTES)
    subprocess.Popen([sys.executable, "-u", "/content/app.py"], cwd="/content", env=child_env, stdout=out_f, stderr=subprocess.STDOUT, start_new_session=True)

url_re = re.compile(r"^EasyColab URL: (https://[a-zA-Z0-9-]+\.gradio\.live)$", re.MULTILINE)
found_url = None
for _ in range(90):
    time.sleep(2)
    if os.path.exists(log_path):
        txt = open(log_path, "r", errors="replace").read()
        m = url_re.search(txt)
        if m:
            found_url = m.group(1)
            break

if found_url:
    print("WAKE_URL:" + found_url, flush=True)
else:
    tail = open(log_path, "r", errors="replace").read()[-600:] if os.path.exists(log_path) else "no log"
    print("WAKE_ERROR: 启动超时，日志尾部: " + tail, flush=True)
"""

        def _on_kid(kid):
            target_session.kernel_id = kid
            state.store.add(target_session)

        def _on_sid(sid):
            target_session.session_id = sid
            state.store.add(target_session)

        def _run_boot_on_runtime(use_saved_kernel: bool):
            rt = ColabRuntime(
                target_session.url,
                target_session.token,
                session_name=target_session.name,
                history=state.history,
                kernel_id=target_session.kernel_id if use_saved_kernel else None,
                session_id=target_session.session_id if use_saved_kernel else None,
                on_kernel_started=_on_kid,
                on_session_started=_on_sid,
            )
            try:
                # Each retry gets its own single-use file. The password never
                # enters the execute_request source or the streamed wake log.
                stage = ".easycolab-management-" + uuid.uuid4().hex
                with tempfile.TemporaryDirectory() as directory:
                    password_file = os.path.join(directory, "management")
                    with open(password_file, "w", encoding="ascii") as stream:
                        stream.write(password)
                    contents.upload(password_file, "content/" + stage)
                code = (
                    "ADMIN_STAGE = " + repr("/content/" + stage) + "\n"
                    + "INITIAL_IDLE_MINUTES = " + repr(initial_idle_minutes) + "\n"
                    + boot_code
                )
                return rt.execute_code(code, allow_stdin=False, timeout=300)
            finally:
                try:
                    rt.stop()
                except Exception:
                    pass

        try:
            outputs = _run_boot_on_runtime(use_saved_kernel=True)
        except Exception:
            target_session.kernel_id = None
            target_session.session_id = None
            outputs = _run_boot_on_runtime(use_saved_kernel=False)

        for out in outputs:
            if "text" in out:
                for line in out["text"].splitlines():
                    print(line, flush=True)
            elif out.get("output_type") == "error":
                print("WAKE_ERROR: " + out.get("evalue", "Execution error"), flush=True)

    except Exception as e:
        print(f"WAKE_ERROR: {str(e)}", flush=True)


def cmd_recover_tunnel(app_py_path: str, restart_server: bool = False):
    """
    无损探测并恢复运行中 Colab 实例的 EasyColab 公网隧道与本地连接状态。
    严禁创建、销毁或释放任何远端 GPU 实例，且绝不误杀 ComfyUI 进程。
    """
    if not _has_valid_token():
        print("RECOVER_ERROR: 尚未授权登录 Google 账号", flush=True)
        return

    try:
        local_sessions, assignments = _safe_sync_sessions()
        if not assignments:
            try:
                info = state.client.get_consumption_user_info()
                vms = int(getattr(info, "assignments_count", 0) or 0)
                rate = float(getattr(info, "consumption_rate_hourly", 0.0) or 0.0)
            except Exception:
                vms = 0
                rate = 0.0
            if vms == 0:
                print("RECOVER_RELEASED: 云端已无运行中实例 (0 VM, 0.00 CU/h)，实例已确认释放。", flush=True)
                return
            print(
                f"RECOVER_ERROR: 账本显示仍有 {vms} 台实例 ({rate:.2f} CU/h)，但未获取到活跃运行时端点，请稍后重试或手动释放。",
                flush=True,
            )
            return

        ep_to_local = {s.endpoint: s for s in local_sessions.values() if getattr(s, "endpoint", None)}
        a = assignments[0]
        existing_local = ep_to_local.get(a.endpoint)
        session_name = existing_local.name if (existing_local and getattr(existing_local, "name", None)) else "easycolab"
        target_session = _session_from_listed_assignment(a, session_name=session_name, existing=existing_local)
        with _quiet_colab_cli():
            state.store.add(target_session)

        print(
            f"WAKE_LOG: 🔍 检测到云端 {a.accelerator.value} 实例仍在运行 ({a.endpoint[:18]}...)，正在探测容器与隧道状态...",
            flush=True,
        )
        if not _wait_for_runtime_ready(target_session, timeout_s=18):
            print(
                "RECOVER_ERROR: 云端账本仍有实例记录，但容器 Jupyter 内核不可达（可能正在重启或刚被冻结），请稍后再试或点击「释放」清理。",
                flush=True,
            )
            return

        def _exec_on_session(code: str, timeout: int = 120):
            def _on_kid(kid):
                target_session.kernel_id = kid
                state.store.add(target_session)

            def _on_sid(sid):
                target_session.session_id = sid
                state.store.add(target_session)

            def _run(use_saved: bool):
                rt = ColabRuntime(
                    target_session.url,
                    target_session.token,
                    session_name=target_session.name,
                    history=state.history,
                    kernel_id=target_session.kernel_id if use_saved else None,
                    session_id=target_session.session_id if use_saved else None,
                    on_kernel_started=_on_kid,
                    on_session_started=_on_sid,
                )
                try:
                    return rt.execute_code(code, allow_stdin=False, timeout=timeout)
                finally:
                    try:
                        rt.stop()
                    except Exception:
                        pass

            try:
                return _run(True)
            except Exception:
                target_session.kernel_id = None
                target_session.session_id = None
                return _run(False)

        if not restart_server:
            probe_code = r"""
import os, re, subprocess, urllib.request, urllib.error

pg = subprocess.run(["pgrep", "-f", "/content/app.py"], capture_output=True, text=True)
app_alive = (pg.returncode == 0 and bool(pg.stdout.strip()))
log_path = "/content/easycolab_server.log"
found_url = ""
if os.path.exists(log_path):
    txt = open(log_path, "r", errors="replace").read()
    matches = re.findall(r"^EasyColab URL:\s*(https://[a-zA-Z0-9-]+\.gradio\.live)\s*$", txt, flags=re.MULTILINE)
    if matches:
        found_url = matches[-1].strip()

tunnel_alive = False
if app_alive and found_url:
    try:
        req = urllib.request.Request(found_url.rstrip("/") + "/minimax_api/status", method="GET")
        with urllib.request.urlopen(req, timeout=6) as resp:
            tunnel_alive = (resp.status in (200, 401))
    except urllib.error.HTTPError as err:
        tunnel_alive = (err.code in (200, 401))
    except Exception:
        tunnel_alive = False

if tunnel_alive and found_url:
    print("RECOVER_URL:" + found_url, flush=True)
else:
    print(f"PROBE_MISS:app_alive={int(app_alive)} has_url={int(bool(found_url))}", flush=True)
"""
            outputs = _exec_on_session(probe_code, timeout=45)
            probe_text = "\n".join(out.get("text", "") for out in outputs if "text" in out)
            m_rec = re.search(r"^RECOVER_URL:(https://[a-zA-Z0-9-]+\.gradio\.live)\s*$", probe_text, flags=re.MULTILINE)
            if m_rec:
                rec_url = m_rec.group(1).strip()
                _sync_active_session_file(
                    rec_url,
                    "",
                    gpu=a.accelerator.value,
                    sleeping=False,
                    source="colab-bridge",
                    account=_get_account_email(),
                )
                print(f"RECOVER_URL:{rec_url}", flush=True)
                print("WAKE_LOG: ✅ 云端服务端与公网隧道均在线，正在重新连接...", flush=True)
                return

        password = os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "")
        if not password:
            print("RECOVER_NEED_RESTART: 云端 GPU 实例存活，但 app.py 或公网隧道已断开，需要重建隧道。", flush=True)
            return

        raw_idle_min = os.environ.get("EASYCOLAB_IDLE_TIMEOUT_MINUTES", "").strip()
        try:
            initial_idle_minutes = int(raw_idle_min) if raw_idle_min != "" else None
        except ValueError:
            initial_idle_minutes = None

        print("WAKE_LOG: 🔄 正在同一台运行中实例上无损重建 app.py 公网隧道（保留 GPU 与 ComfyUI）...", flush=True)
        contents = ContentsClient(target_session)
        for up_attempt in range(4):
            try:
                contents.upload(app_py_path, "content/app.py")
                break
            except Exception as up_err:
                if up_attempt == 3:
                    raise up_err
                time.sleep(2.0)
        contents.upload(os.path.join(os.path.dirname(app_py_path), "runtime_auth.py"), "content/runtime_auth.py")

        stage = ".easycolab-management-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as directory:
            password_file = os.path.join(directory, "management")
            with open(password_file, "w", encoding="ascii") as stream:
                stream.write(password)
            contents.upload(password_file, "content/" + stage)

        restart_code = (
            "ADMIN_STAGE = " + repr("/content/" + stage) + "\n"
            + "INITIAL_IDLE_MINUTES = " + repr(initial_idle_minutes) + "\n"
            + r"""
import os, sys, time, re, subprocess

with open(ADMIN_STAGE, encoding="ascii") as stream:
    admin_password = stream.read()
os.unlink(ADMIN_STAGE)

subprocess.run(["pkill", "-f", "python.*/content/app.py"], check=False)
time.sleep(1)

log_path = "/content/easycolab_server.log"
if os.path.exists(log_path):
    os.remove(log_path)

with open(log_path, "w") as out_f:
    child_env = {**os.environ, "EASYCOLAB_ADMIN_PASSWORD": admin_password}
    if INITIAL_IDLE_MINUTES is not None:
        child_env["EASYCOLAB_IDLE_TIMEOUT_MINUTES"] = str(INITIAL_IDLE_MINUTES)
    subprocess.Popen([sys.executable, "-u", "/content/app.py"], cwd="/content", env=child_env, stdout=out_f, stderr=subprocess.STDOUT, start_new_session=True)

url_re = re.compile(r"^EasyColab URL: (https://[a-zA-Z0-9-]+\.gradio\.live)$", re.MULTILINE)
found_url = None
for _ in range(90):
    time.sleep(2)
    if os.path.exists(log_path):
        txt = open(log_path, "r", errors="replace").read()
        m = url_re.search(txt)
        if m:
            found_url = m.group(1)
            break

if found_url:
    print("WAKE_URL:" + found_url, flush=True)
else:
    tail = open(log_path, "r", errors="replace").read()[-600:] if os.path.exists(log_path) else "no log"
    print("RECOVER_ERROR: 重建隧道超时，日志尾部: " + tail, flush=True)
"""
        )
        outputs = _exec_on_session(restart_code, timeout=240)
        restarted_url = ""
        for out in outputs:
            if "text" in out:
                for line in out["text"].splitlines():
                    print(line, flush=True)
                    if line.startswith("WAKE_URL:"):
                        restarted_url = line[len("WAKE_URL:"):].strip()
            elif out.get("output_type") == "error":
                print("RECOVER_ERROR: " + out.get("evalue", "Execution error"), flush=True)
        if restarted_url:
            _sync_active_session_file(
                restarted_url,
                password,
                gpu=a.accelerator.value,
                sleeping=False,
                source="colab-bridge",
                account=_get_account_email(),
            )
    except Exception as exc:
        print(f"RECOVER_ERROR: {exc}", flush=True)


class _ProxyAuthExpiredError(RuntimeError):
    """Runtime Proxy Token 已过期或失效 (HTTP 401/403)，需立即通过 list_assignments 刷新"""


def _session_token_ttl_seconds(s: SessionState) -> float:
    exp = getattr(s, "token_expires_at", None)
    if exp is None:
        return 3600.0
    try:
        if isinstance(exp, (int, float)):
            return float(exp) - time.time()
        if hasattr(exp, "timestamp"):
            return float(exp.timestamp()) - time.time()
        if isinstance(exp, str) and exp.strip():
            dt = datetime.fromisoformat(exp.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp() - time.time()
    except Exception:
        pass
    return 3600.0


def _sync_and_adopt_sessions():
    local_sessions, assignments = _safe_sync_sessions()
    ep_map = {a.endpoint: a for a in assignments}

    if not local_sessions and assignments:
        a0 = assignments[0]
        s_auto = SessionState(
            name="easycolab",
            token=a0.runtime_proxy_info.token,
            url=a0.runtime_proxy_info.url,
            endpoint=a0.endpoint,
            token_expires_at=a0.runtime_proxy_info.expires_at(),
            session_id=uuid.uuid4().hex,
            variant=Variant.GPU.value,
            accelerator=a0.accelerator.value,
            machine_shape=Shape.HIGH_RAM.name if a0.machine_shape == Shape.HIGH_RAM else Shape.STANDARD.name,
        )
        with _quiet_colab_cli():
            state.store.add(s_auto)
        local_sessions = {"easycolab": s_auto}

    for s in list(local_sessions.values()):
        changed = False
        if s.endpoint in ep_map:
            info_proxy = ep_map[s.endpoint].runtime_proxy_info
            if s.token != info_proxy.token or s.url != info_proxy.url:
                changed = True
            s.token = info_proxy.token
            s.url = info_proxy.url
            s.token_expires_at = info_proxy.expires_at()
        if not getattr(s, "session_id", None):
            s.session_id = uuid.uuid4().hex
            changed = True
        if changed:
            with _quiet_colab_cli():
                state.store.add(s)

    return local_sessions, assignments


def _touch_colab_tunnel_keepalive(endpoint: str) -> bool:
    if not endpoint:
        return False
    try:
        keepalive_url = f"{state.client.colab_domain.rstrip('/')}/tun/m/{endpoint}/keep-alive/"
        state.client._issue_request(
            keepalive_url,
            method="GET",
            headers={"X-Colab-Tunnel": "Google"},
        )
        return True
    except Exception:
        return False


def _resolve_or_start_remote_kernel(s: SessionState, timeout_s: float = 12.0) -> str:
    """
    通过 Colab 官方 Runtime Proxy 查询或启动远端 Jupyter 内核。
    严禁删除其他已有内核，避免误杀并发调试或子任务内核。
    """
    base_url = (s.url or "").rstrip("/")
    if not base_url or not s.token:
        raise RuntimeError("Session missing runtime proxy URL or token")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Colab-Client-Agent": "vscode",
        "X-Colab-Runtime-Proxy-Token": s.token,
        "Authorization": f"Bearer {s.token}",
    }
    params = {"authuser": "0", "colab-runtime-proxy-token": s.token}

    r = requests.get(f"{base_url}/api/kernels", headers=headers, params=params, timeout=timeout_s)
    if r.status_code in (401, 403):
        raise _ProxyAuthExpiredError(f"Runtime proxy HTTP {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"GET /api/kernels returned HTTP {r.status_code}")

    kernels = r.json() if r.text else []
    active_kid = None
    if isinstance(kernels, list) and kernels:
        known_ids = [str(k.get("id")) for k in kernels if isinstance(k, dict) and k.get("id")]
        if known_ids:
            active_kid = s.kernel_id if (s.kernel_id in known_ids) else known_ids[0]

    if not active_kid:
        pr = requests.post(
            f"{base_url}/api/kernels",
            headers=headers,
            params=params,
            json={"name": "python3"},
            timeout=timeout_s,
        )
        if pr.status_code in (401, 403):
            raise _ProxyAuthExpiredError(f"POST /api/kernels HTTP {pr.status_code}")
        if pr.status_code not in (200, 201):
            raise RuntimeError(f"POST /api/kernels returned HTTP {pr.status_code}")
        pdata = pr.json() if pr.text else {}
        active_kid = str(pdata.get("id") or "")

    if not active_kid:
        raise RuntimeError("Unable to obtain remote Jupyter kernel_id")

    changed = False
    if s.kernel_id != active_kid:
        s.kernel_id = active_kid
        changed = True
    if not s.session_id:
        s.session_id = uuid.uuid4().hex
        changed = True
    if changed:
        with _quiet_colab_cli():
            state.store.add(s)
    return active_kid


def _open_colab_proxy_ws(s: SessionState, kernel_id: str, timeout_s: float = 15.0) -> websocket.WebSocket:
    """
    建立直连 Google Colab 官方外部 Runtime Proxy (*.prod.r.colab.dev) 的常驻 WebSocket 通道，
    显式携带 colab-runtime-proxy-token、token、authuser=0 查询参数与认证请求头，
    确保 Google 外部代理层始终观测到活跃 WebSocket 连接 (connections >= 1)。
    """
    base_url = (s.url or "").rstrip("/")
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://"):]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://"):]
    else:
        ws_base = "wss://" + base_url

    sid = s.session_id or uuid.uuid4().hex
    qs = urllib.parse.urlencode({
        "session_id": sid,
        "token": s.token,
        "colab-runtime-proxy-token": s.token,
        "authuser": "0",
    })
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?{qs}"
    headers = [
        f"X-Colab-Runtime-Proxy-Token: {s.token}",
        f"Authorization: Bearer {s.token}",
        "X-Colab-Client-Agent: vscode",
        "User-Agent: EasyColab/2.0",
    ]
    return websocket.create_connection(
        ws_url,
        timeout=timeout_s,
        origin=base_url,
        header=headers,
        enable_multithread=True,
    )


def _drain_proxy_ws_frames(ws: websocket.WebSocket, wait_s: float = 1.0, stop_on_msg_id: str = "") -> bool:
    """排空服务端下发的 Jupyter 消息与控制帧（自动回复 Ping -> Pong，遇到 Close 立即返回 False）"""
    deadline = time.time() + max(0.05, wait_s)
    while True:
        rem = deadline - time.time()
        if rem <= 0:
            return True
        try:
            ws.settimeout(min(rem, 1.0))
            opcode, frame = ws.recv_data(control_frame=True)
            if opcode == ABNF.OPCODE_CLOSE:
                return False
            if stop_on_msg_id and opcode == ABNF.OPCODE_TEXT and frame:
                raw_txt = frame.decode("utf-8", errors="replace") if isinstance(frame, (bytes, bytearray)) else str(frame)
                if stop_on_msg_id in raw_txt and '"execute_reply"' in raw_txt:
                    return True
        except websocket.WebSocketTimeoutException:
            return True
        except Exception:
            return False


def _send_proxy_ws_heartbeat(ws: websocket.WebSocket, session_id: str, touch_client_hb: bool = False) -> bool:
    """在已打开的外部 Runtime Proxy WebSocket 上发送 Ping 与 Jupyter execute_request，刷新代理与内核活跃度"""
    msg_id = uuid.uuid4().hex
    hb_file = "/tmp/.easycolab_client_hb" if touch_client_hb else "/tmp/.easycolab_proxy_hb"
    code = f"import pathlib; pathlib.Path('{hb_file}').touch(); _easycolab_hb = 1"
    msg = {
        "channel": "shell",
        "header": {
            "msg_id": msg_id,
            "username": "easycolab",
            "session": session_id,
            "date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": False,
        },
    }
    try:
        ws.settimeout(10.0)
        ws.ping("easycolab")
        ws.send(json.dumps(msg))
        return _drain_proxy_ws_frames(ws, wait_s=2.0, stop_on_msg_id=msg_id)
    except Exception:
        return False


def cmd_keepalive():
    """
    单次向活跃的 Colab Headless 实例发送官方 Runtime Proxy 探活与轻量 Jupyter 内核脉冲（兼容 CLI / Windows / Linux 调用）。
    macOS 客户端优先使用长连接常驻子进程 keepalive_daemon。
    """
    if not _has_valid_token():
        print(json.dumps({"ok": True, "logged_in": False}))
        return

    try:
        local_sessions, _ = _sync_and_adopt_sessions()
        touched = 0
        warnings = []

        for s in list(local_sessions.values()):
            _touch_colab_tunnel_keepalive(s.endpoint)
            if not s.url or not s.token:
                continue

            ws = None
            try:
                active_kid = _resolve_or_start_remote_kernel(s, timeout_s=12.0)
                ws = _open_colab_proxy_ws(s, active_kid, timeout_s=15.0)
                if _send_proxy_ws_heartbeat(ws, s.session_id or uuid.uuid4().hex, touch_client_hb=True):
                    touched += 1
                else:
                    warnings.append(f"{s.name}: heartbeat closed")
            except Exception as touch_err:
                warnings.append(f"{s.name}: {touch_err}")
            finally:
                if ws is not None:
                    try:
                        ws.close(timeout=1)
                    except Exception:
                        pass

        info = state.client.get_consumption_user_info()
        payload = {
            "ok": True,
            "logged_in": True,
            "touched_kernels": touched,
            "paid_balance": round(float(info.paid_compute_units_balance), 2),
            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
            "active_assignments": int(info.assignments_count),
        }
        if warnings:
            payload["warning"] = "; ".join(warnings)
        print(json.dumps(payload))
    except Exception as e:
        print(json.dumps({"ok": False, "logged_in": True, "error": str(e)}))


def cmd_keepalive_daemon():
    """
    macOS 原生客户端专属常驻保活守护进程：
    - 始终保持一条直连 Google Colab 官方外部 Runtime Proxy (wss://*.prod.r.colab.dev/api/kernels/<id>/channels)
      的真实 WebSocket 长连接，使代理层与 Jupyter 内核的 connections 持续 >= 1（杜绝无浏览器模式下 ~20 分钟孤儿实例强杀）；
    - 每 25 秒在长连接上发送一次 WebSocket Ping + Jupyter execute_request 脉冲；
    - 每 45 秒调用 /tun/m/<endpoint>/keep-alive/ 并拉取实时 CU 余额/速率，通过 KEEPALIVE_JSON: 行流式推送给客户端；
    - 在代理 Token 临近过期 (<10 分钟) 或连接断开时自动刷新 runtimeProxyInfo 并无缝重连；
    - 监听父进程 stdin 管道 EOF 与进程生命周期：一旦 EasyColab.app 断开实例或退出，立即关闭 WebSocket 并零残留退出。
    """
    stop_event = threading.Event()
    initial_ppid = os.getppid()

    def _handle_sig(_signum, _frame):
        stop_event.set()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig_obj = getattr(signal, sig_name, None)
        if sig_obj is not None:
            try:
                signal.signal(sig_obj, _handle_sig)
            except Exception:
                pass

    if not sys.stdin.isatty():
        def _watch_stdin_eof():
            try:
                while not stop_event.is_set():
                    data = sys.stdin.read(1)
                    if data == "":
                        stop_event.set()
                        break
            except Exception:
                stop_event.set()

        threading.Thread(target=_watch_stdin_eof, daemon=True).start()

    def _emit(obj: dict):
        try:
            print("KEEPALIVE_JSON:" + json.dumps(obj, ensure_ascii=False), flush=True)
        except Exception:
            stop_event.set()

    if not _has_valid_token():
        _emit({"ok": True, "logged_in": False, "ws_connected": False})
        return

    ws_map: dict[str, tuple[websocket.WebSocket, str, str]] = {}  # endpoint -> (ws, url, token)

    def _close_ws(ep: str):
        entry = ws_map.pop(ep, None)
        if entry is not None:
            try:
                entry[0].close(timeout=1)
            except Exception:
                pass

    def _close_all_ws():
        for ep in list(ws_map.keys()):
            _close_ws(ep)

    last_sync_ts = 0.0
    last_hb_ts = 0.0
    last_usage_ts = 0.0
    force_resync = True
    local_sessions: dict[str, SessionState] = {}

    try:
        while not stop_event.is_set():
            if os.name != "nt" and os.getppid() != initial_ppid:
                stop_event.set()
                break

            now = time.time()
            min_ttl = min((_session_token_ttl_seconds(s) for s in local_sessions.values()), default=3600.0)
            if force_resync or (now - last_sync_ts) >= 480.0 or min_ttl < 600.0:
                try:
                    local_sessions, _ = _sync_and_adopt_sessions()
                    last_sync_ts = time.time()
                    force_resync = False
                    active_eps = {s.endpoint for s in local_sessions.values() if s.endpoint}
                    for stale_ep in list(ws_map.keys()):
                        if stale_ep not in active_eps:
                            _close_ws(stale_ep)
                except Exception as sync_err:
                    _emit({"ok": False, "logged_in": True, "ws_connected": bool(ws_map), "warning": f"sync: {sync_err}"})

            if not local_sessions:
                if (now - last_usage_ts) >= 30.0:
                    try:
                        info = state.client.get_consumption_user_info()
                        last_usage_ts = time.time()
                        _emit({
                            "ok": True,
                            "logged_in": True,
                            "ws_connected": False,
                            "paid_balance": round(float(info.paid_compute_units_balance), 2),
                            "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
                            "active_assignments": int(info.assignments_count),
                        })
                    except Exception:
                        pass
                stop_event.wait(6.0)
                force_resync = True
                continue

            # 确保每个活跃会话都持有常驻外部 Runtime Proxy WebSocket
            need_hb = (time.time() - last_hb_ts) >= 25.0
            any_ws_ok = False
            last_warn = ""

            for s in list(local_sessions.values()):
                if stop_event.is_set():
                    break
                ep = s.endpoint or s.name
                existing = ws_map.get(ep)
                if existing is not None:
                    old_ws, old_url, old_tok = existing
                    if not getattr(old_ws, "connected", False) or old_url != s.url or old_tok != s.token:
                        _close_ws(ep)
                        existing = None

                if existing is None:
                    try:
                        kid = _resolve_or_start_remote_kernel(s, timeout_s=12.0)
                        new_ws = _open_colab_proxy_ws(s, kid, timeout_s=15.0)
                        if _send_proxy_ws_heartbeat(new_ws, s.session_id or uuid.uuid4().hex, touch_client_hb=False):
                            ws_map[ep] = (new_ws, s.url, s.token)
                            any_ws_ok = True
                            last_hb_ts = time.time()
                        else:
                            try:
                                new_ws.close(timeout=1)
                            except Exception:
                                pass
                            force_resync = True
                    except _ProxyAuthExpiredError as auth_err:
                        last_warn = str(auth_err)
                        force_resync = True
                    except Exception as conn_err:
                        last_warn = str(conn_err)
                        force_resync = True
                else:
                    cur_ws = existing[0]
                    if need_hb:
                        if _send_proxy_ws_heartbeat(cur_ws, s.session_id or uuid.uuid4().hex, touch_client_hb=False):
                            any_ws_ok = True
                        else:
                            _close_ws(ep)
                            force_resync = True
                            last_warn = "ws heartbeat lost, reconnecting"
                    else:
                        if _drain_proxy_ws_frames(cur_ws, wait_s=1.0):
                            any_ws_ok = True
                        else:
                            _close_ws(ep)
                            force_resync = True
                            last_warn = "ws closed by peer, reconnecting"

            if need_hb and any_ws_ok:
                last_hb_ts = time.time()

            now2 = time.time()
            if (now2 - last_usage_ts) >= 45.0 or (last_usage_ts == 0.0):
                for s in list(local_sessions.values()):
                    _touch_colab_tunnel_keepalive(s.endpoint)
                try:
                    info = state.client.get_consumption_user_info()
                    last_usage_ts = now2
                    payload = {
                        "ok": True,
                        "logged_in": True,
                        "ws_connected": any_ws_ok,
                        "paid_balance": round(float(info.paid_compute_units_balance), 2),
                        "burn_rate_hourly": round(float(info.consumption_rate_hourly), 2),
                        "active_assignments": int(info.assignments_count),
                    }
                    if last_warn:
                        payload["warning"] = last_warn
                    _emit(payload)
                except Exception as usage_err:
                    last_usage_ts = now2
                    _emit({"ok": False, "logged_in": True, "ws_connected": any_ws_ok, "warning": str(usage_err)})

            if not any_ws_ok and not stop_event.is_set():
                stop_event.wait(3.0)
            elif need_hb and not stop_event.is_set():
                stop_event.wait(1.0)
    finally:
        _close_all_ws()


def cmd_wake_multi(
    accounts: list[str],
    gpu_type: str,
    app_py_path: str,
    mode: str = "first",
    persistent: bool = False,
    retry_interval: float = 60.0,
) -> dict:
    """
    多账号并发申请 Colab GPU：
    - mode="first": 竞速抢卡（默认）。多个账号同时发起申请，首个分到机器的账号胜出，
      立即终止其余账号并精准释放毫秒级并发多余新实例，仅部署胜出账号并返回 WAKE_WINNER + WAKE_URL。
    - mode="all": 批量多开。并行启动所有指定账号各自的 GPU 并分别部署，输出每个账号的 WAKE_ACCOUNT_URL 与 WAKE_MULTI_JSON。
    """
    eff_mode = mode if mode in ("first", "all") else "first"
    password = os.environ.pop("EASYCOLAB_ADMIN_PASSWORD", "")
    if not password and sys.stdin.isatty():
        import getpass
        password = getpass.getpass("Management password (24+ URL-safe characters): ")
    if not password:
        msg = "Set a management password before deploying"
        print(f"WAKE_ERROR: {msg}", flush=True)
        return {"ok": False, "mode": eff_mode, "error": msg}

    clean_accounts: list[str] = []
    seen_norm: set[str] = set()
    for raw in accounts:
        em = (raw or "").strip()
        if not em:
            continue
        norm = em.lower()
        if norm not in seen_norm:
            seen_norm.add(norm)
            clean_accounts.append(em)

    if not clean_accounts:
        saved = _refresh_all_saved_accounts_live("")
        clean_accounts = [
            str(a.get("email") or "").strip()
            for a in saved
            if str(a.get("email") or "").strip()
            and a.get("status_ok") is not False
            and int(a.get("active_assignments") or 0) == 0
        ]
    if not clean_accounts:
        msg = "未找到可用的空闲已保存 Google 账号，请在弹窗或 --accounts 中指定目标账号。"
        print(f"WAKE_ERROR: {msg}", flush=True)
        return {"ok": False, "mode": eff_mode, "error": msg}

    eff_interval = max(5.0, float(retry_interval or 60.0))
    use_persistent = bool(persistent) or (
        os.environ.get("EASYCOLAB_PERSISTENT_WAKE", "").strip().lower() in ("1", "true", "yes")
    )

    mode_label = "竞速抢卡（抢到首个即停）" if eff_mode == "first" else "批量多开（全部各开一台）"
    print(
        f"WAKE_LOG: 🚀 启动多账号同时申请 [{mode_label}] · {gpu_type} GPU · 共 {len(clean_accounts)} 个账号 ({', '.join(clean_accounts)})",
        flush=True,
    )

    lock = threading.Lock()
    cancel_event = threading.Event()
    procs: dict[str, subprocess.Popen] = {}
    winner_email: str | None = None
    winner_url: str | None = None
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    def _terminate_other_procs(keep_email: str | None):
        with lock:
            targets = [
                p for em, p in procs.items()
                if em != keep_email and p.poll() is None
            ]
        for p in targets:
            try:
                p.terminate()
            except Exception:
                pass

    def _terminate_all_procs():
        cancel_event.set()
        _terminate_other_procs(None)

    prev_handlers: dict[int, object] = {}
    is_main_thread = threading.current_thread() is threading.main_thread()
    if is_main_thread:
        def _on_signal(_signum, _frame):
            _terminate_all_procs()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _on_signal)
            except Exception:
                pass

    def _run_account_worker(email: str):
        nonlocal winner_email, winner_url
        child_env = os.environ.copy()
        child_env["EASYCOLAB_ACCOUNT"] = email
        child_env["EASYCOLAB_ADMIN_PASSWORD"] = password
        child_env["EASYCOLAB_SUPPRESS_BRIDGE_LOG"] = "1"
        child_env.pop("EASYCOLAB_FROM_APP", None)

        child_args = ["--account", email, "wake", "--gpu", gpu_type, "--app-py", app_py_path]
        if use_persistent:
            child_args += ["--persistent", "--retry-interval", str(int(round(eff_interval)))]

        proc = subprocess.Popen(
            _build_bridge_subcommand(child_args),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with lock:
            if cancel_event.is_set() or (eff_mode == "first" and winner_email is not None and winner_email != email):
                try:
                    proc.terminate()
                except Exception:
                    pass
            procs[email] = proc

        dup_new_endpoints: list[str] = []
        try:
            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue

                    if line.startswith("WAKE_NEW_ALLOCATED:") or line.startswith("WAKE_ALLOCATED:"):
                        is_new_alloc = line.startswith("WAKE_NEW_ALLOCATED:")
                        ep = line.split(":", 1)[1].strip()
                        if eff_mode == "first":
                            won_now = False
                            lost_now = False
                            with lock:
                                if not cancel_event.is_set() and winner_email is None:
                                    winner_email = email
                                    won_now = True
                                elif winner_email is not None and winner_email != email:
                                    lost_now = True
                            if won_now:
                                print(f"WAKE_WINNER: {email}", flush=True)
                                if len(clean_accounts) > 1:
                                    print(
                                        f"WAKE_LOG: 🏆 [{email}] 率先分配到 {gpu_type} 实例，正在立即停止其余 {len(clean_accounts) - 1} 个账号的申请并继续部署...",
                                        flush=True,
                                    )
                                _terminate_other_procs(email)
                            elif lost_now and is_new_alloc and ep:
                                dup_new_endpoints.append(ep)
                        continue

                    if line.startswith("WAKE_URL:"):
                        url = line[len("WAKE_URL:"):].strip()
                        if eff_mode == "first":
                            emit_win_url = False
                            announced_winner = False
                            with lock:
                                if not cancel_event.is_set() and (winner_email is None or winner_email == email):
                                    if winner_email is None:
                                        winner_email = email
                                        announced_winner = True
                                    if winner_url is None:
                                        winner_url = url
                                        emit_win_url = True
                                        results[email] = {"ok": True, "email": email, "url": url}
                            if announced_winner:
                                print(f"WAKE_WINNER: {email}", flush=True)
                                _terminate_other_procs(email)
                            if emit_win_url:
                                print(f"WAKE_URL: {url}", flush=True)
                        else:
                            with lock:
                                results[email] = {"ok": True, "email": email, "url": url}
                            print(f"WAKE_ACCOUNT_URL: {email} {url}", flush=True)
                            print(f"WAKE_LOG: ✅ [{email}] 实例与隧道已就绪：{url[:38]}...", flush=True)
                        continue

                    if line.startswith("WAKE_ERROR:"):
                        err_msg = line[len("WAKE_ERROR:"):].strip() or "Unknown wake error"
                        with lock:
                            errors[email] = err_msg
                            cur_winner = winner_email
                        if eff_mode == "all":
                            with lock:
                                results.setdefault(email, {"ok": False, "email": email, "error": err_msg})
                            print(f"WAKE_LOG: ❌ [{email}] 唤醒失败: {err_msg}", flush=True)
                        elif cur_winner == email:
                            print(f"WAKE_ERROR: [{email}] {err_msg}", flush=True)
                        else:
                            if not cancel_event.is_set() and cur_winner is None:
                                print(f"WAKE_LOG: ⚠️ [{email}] {err_msg}", flush=True)
                        continue

                    if line.startswith("WAKE_LOG:"):
                        clean = line[len("WAKE_LOG:"):].strip()
                        if not clean:
                            continue
                        with lock:
                            cur_winner = winner_email
                        if eff_mode == "first" and cur_winner is not None and cur_winner != email:
                            continue
                        print(f"WAKE_LOG: [{email}] {clean}", flush=True)
                        continue

                    if not line.startswith("[colab]") and "Assignment request failed" not in line and "urllib3" not in line:
                        with lock:
                            cur_winner = winner_email
                        if eff_mode == "first" and cur_winner is not None and cur_winner != email:
                            continue
                        print(f"[{email}] {line}", flush=True)
        finally:
            try:
                if proc.poll() is None:
                    proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            with lock:
                cur_winner = winner_email
            if eff_mode == "first" and dup_new_endpoints and cur_winner != email:
                for dup_ep in dup_new_endpoints:
                    print(
                        f"WAKE_LOG: 🧹 [{email}] 并发竞速出现毫秒级重复分配 ({dup_ep[:16]}...)，正在自动精准释放防重复扣费...",
                        flush=True,
                    )
                    _unassign_single_endpoint_for_account(email, dup_ep)

    threads: list[threading.Thread] = []
    try:
        for em in clean_accounts:
            t = threading.Thread(target=_run_account_worker, args=(em,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    finally:
        _terminate_all_procs()
        with lock:
            stray_procs = list(procs.values())
        for p in stray_procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=2.0)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        if is_main_thread:
            for sig, prev in prev_handlers.items():
                try:
                    signal.signal(sig, prev)
                except Exception:
                    pass

    if eff_mode == "first":
        if not winner_url and not cancel_event.is_set():
            print(
                f"WAKE_ERROR: 所有参与账号 ({len(clean_accounts)} 个) 均未成功分配或部署 {gpu_type} 实例，请查看上方日志或稍后重试。",
                flush=True,
            )
        return {
            "ok": bool(winner_url),
            "mode": "first",
            "gpu": gpu_type,
            "winner": winner_email,
            "url": winner_url,
            "errors": errors,
        }

    succeeded = [em for em in clean_accounts if results.get(em, {}).get("ok")]
    failed = [em for em in clean_accounts if em not in succeeded]
    summary = {
        "ok": bool(succeeded),
        "mode": "all",
        "gpu": gpu_type,
        "succeeded": succeeded,
        "failed": failed,
        "results": {
            em: results.get(em, {"ok": False, "email": em, "error": errors.get(em, "Failed")})
            for em in clean_accounts
        },
    }
    print(f"WAKE_MULTI_JSON: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    if not succeeded and not cancel_event.is_set():
        print(
            f"WAKE_ERROR: 批量多开失败：所有勾选账号 ({len(clean_accounts)} 个) 均未能成功唤醒 {gpu_type}。",
            flush=True,
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", type=str, default="", help="Scope bridge execution to a saved Google account email")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth_status")
    sub.add_parser("auth_status_all")
    sub.add_parser("auth_url")
    sub.add_parser("auth_logout")
    sub.add_parser("auth_list")

    p_code = sub.add_parser("auth_code")
    p_code.add_argument("code", type=str)

    p_sw = sub.add_parser("auth_switch")
    p_sw.add_argument("email", type=str)

    p_rm = sub.add_parser("auth_remove")
    p_rm.add_argument("email", type=str)

    p_wake = sub.add_parser("wake")
    p_wake.add_argument("--gpu", type=str, default="G4")
    p_wake.add_argument(
        "--app-py",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    )
    p_wake.add_argument("--persistent", action="store_true")
    p_wake.add_argument("--retry-interval", type=float, default=60.0)

    p_wmulti = sub.add_parser("wake_multi")
    p_wmulti.add_argument("--accounts", type=str, default="", help="Comma-separated saved account emails")
    p_wmulti.add_argument("--gpu", type=str, default="G4")
    p_wmulti.add_argument(
        "--app-py",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    )
    p_wmulti.add_argument("--mode", type=str, default="first", choices=["first", "all"])
    p_wmulti.add_argument("--persistent", action="store_true")
    p_wmulti.add_argument("--retry-interval", type=float, default=60.0)

    p_unep = sub.add_parser("unassign_endpoint")
    p_unep.add_argument("endpoint", type=str)

    p_rec = sub.add_parser("recover_tunnel")
    p_rec.add_argument(
        "--app-py",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    )
    p_rec.add_argument("--restart-server", action="store_true")

    sub.add_parser("keepalive")
    sub.add_parser("keepalive_daemon")
    sub.add_parser("stop_all")

    args = parser.parse_args(argv)
    scoped_account = (getattr(args, "account", "") or os.environ.get("EASYCOLAB_ACCOUNT", "")).strip()
    if scoped_account and args.cmd in ("auth_status", "auth_status_all", "wake", "recover_tunnel", "keepalive", "keepalive_daemon", "stop_all", "unassign_endpoint"):
        try:
            _activate_scoped_account_context(scoped_account)
        except Exception as scope_err:
            if args.cmd == "wake":
                print(f"WAKE_ERROR: 多账号隔离加载失败 ({scoped_account}): {scope_err}", flush=True)
                return
            print(json.dumps({
                "ok": False,
                "logged_in": False,
                "email": scoped_account,
                "accounts": _list_saved_accounts(""),
                "error": str(scope_err),
            }))
            return
    if args.cmd == "auth_status":
        cmd_auth_status()
    elif args.cmd == "auth_status_all":
        cmd_auth_status_all()
    elif args.cmd == "auth_url":
        cmd_auth_url()
    elif args.cmd == "auth_logout":
        cmd_auth_logout()
    elif args.cmd == "auth_list":
        cmd_auth_list()
    elif args.cmd == "auth_switch":
        cmd_auth_switch(args.email)
    elif args.cmd == "auth_remove":
        cmd_auth_remove(args.email)
    elif args.cmd == "auth_code":
        cmd_auth_code(args.code)
    elif args.cmd == "unassign_endpoint":
        cmd_unassign_endpoint(args.endpoint)
    elif args.cmd == "recover_tunnel":
        cmd_recover_tunnel(
            args.app_py,
            restart_server=bool(getattr(args, "restart_server", False)),
        )
    elif args.cmd == "wake":
        with _external_cli_log_tee():
            _append_external_app_log(
                f"正在通过 CLI 直接唤醒 Colab ({args.gpu} GPU{', 持续每60s重申' if args.persistent else ''})..."
            )
            cmd_wake(
                args.gpu,
                args.app_py,
                persistent=bool(getattr(args, "persistent", False)),
                retry_interval=float(getattr(args, "retry_interval", 60.0) or 60.0),
            )
    elif args.cmd == "wake_multi":
        raw_accts = [s.strip() for s in str(getattr(args, "accounts", "") or "").split(",") if s.strip()]
        with _external_cli_log_tee():
            _append_external_app_log(
                f"正在通过 CLI 启动多账号同时申请 ({args.gpu} GPU, 模式={args.mode}, 账号={','.join(raw_accts) or '自动空闲账号'})..."
            )
            cmd_wake_multi(
                raw_accts,
                args.gpu,
                args.app_py,
                mode=args.mode,
                persistent=bool(getattr(args, "persistent", False)),
                retry_interval=float(getattr(args, "retry_interval", 60.0) or 60.0),
            )
    elif args.cmd == "keepalive":
        cmd_keepalive()
    elif args.cmd == "keepalive_daemon":
        cmd_keepalive_daemon()
    elif args.cmd == "stop_all":
        with _external_cli_log_tee():
            _append_external_app_log("正在通过 CLI 直接休眠并释放全部 Colab 实例...")
            cmd_stop_all()


def run_cli(argv=None):
    """供独立 CLI 或打包后的 Windows exe (--bridge-cli) 子进程模式统一调用"""
    try:
        main(argv)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    run_cli()
