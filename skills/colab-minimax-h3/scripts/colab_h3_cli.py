#!/usr/bin/env python3
"""Unified CLI for EasyColab + Google Colab MiniMax-H3 workflows (stdlib only)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import secrets
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_EASYCOLAB_ROOT = "/Users/vanch/easycolab"
SESSION_PATH = Path.home() / ".config" / "colab-minimax-h3" / "session.json"
APP_LOG_PATH = Path(
    os.environ.get("EASYCOLAB_APP_LOG_PATH", "").strip()
    or (Path.home() / "Library" / "Application Support" / "EasyColab" / "external_cli_logs.jsonl")
)
ACTIVE_SESSION_PATH = Path(
    os.environ.get("EASYCOLAB_ACTIVE_SESSION_PATH", "").strip()
    or (Path.home() / "Library" / "Application Support" / "EasyColab" / "active_session.json")
)
ACTIVE_TASK_PATH = Path(
    os.environ.get("EASYCOLAB_ACTIVE_TASK_PATH", "").strip()
    or (Path.home() / "Library" / "Application Support" / "EasyColab" / "active_external_task.json")
)
_CURRENT_EXTERNAL_TASK: Optional[Dict[str, Any]] = None


def _infer_project_info(out_dir: str, prefix: str = "") -> Tuple[str, str]:
    """Infer user-facing project folder name & path from output directory or prefix."""
    generic = {
        "outputs", "output", "work", "renders", "render",
        "videos", "video", "images", "image", "tmp", "temp", "out",
    }
    try:
        p = Path(out_dir or ".").expanduser().resolve()
    except Exception:
        p = Path.cwd().resolve()
    if p.is_file():
        p = p.parent
    cur = p
    for _ in range(6):
        name = cur.name.strip()
        low = name.lower()
        if name and low not in generic and not low.endswith("-outputs") and not low.endswith("_outputs") and cur != cur.parent:
            return name, str(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    cwd = Path.cwd().resolve()
    if cwd.name and cwd.name.lower() not in generic:
        return cwd.name, str(cwd)
    if prefix:
        tail = prefix.strip("/").split("/")[-1]
        if tail:
            return tail, str(p)
    return p.name or "EasyColab Task", str(p)


def _sync_app_external_task(payload: Dict[str, Any]) -> None:
    """Publish structured external task metadata so EasyColab previews project & prompt live."""
    try:
        target = Path(
            os.environ.get("EASYCOLAB_ACTIVE_TASK_PATH", "").strip() or str(ACTIVE_TASK_PATH)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
        os.chmod(target, 0o600)
    except Exception:
        pass


def _model_display_name(model_id: str) -> str:
    mapping = {
        "minimax_h3": "MiniMax-H3 (33B)",
        "minimax_h3_10eros_max": "10Eros-Max (H3 无限制)",
        "minimax_h3_sulphur3": "Sulphur-3 (H3 无限制)",
        "minimax_h3_ref2va_tara": "Ref2VA · TARA 动作复刻",
        "seedvr2_upscaler": "SeedVR2 7B · 视频超分",
        "recavsr_upscaler": "ReCaVSR · 极速流式超分",
        "minimax_h3_image": "MiniMax-H3-Image (33B · 4K)",
        "comfyui_studio": "ComfyUI 节点工作台",
        "qwen38_27b": "Qwen3.8-27B",
        "qwen38_27b_gguf": "Qwen3.8-27B-GGUF",
        "qwen_image_21": "Qwen-Image-2.1",
        "z_image_turbo": "Z-Image-Turbo",
    }
    return mapping.get(model_id, model_id)


def _pack_to_model_id(pack_id: str) -> Tuple[str, str]:
    low = (pack_id or "").lower()
    if "recavsr" in low:
        return "recavsr_upscaler", _model_display_name("recavsr_upscaler")
    if "seedvr2" in low:
        return "seedvr2_upscaler", _model_display_name("seedvr2_upscaler")
    if "tara" in low or "ref2va" in low:
        return "minimax_h3_ref2va_tara", _model_display_name("minimax_h3_ref2va_tara")
    if "10eros" in low:
        return "minimax_h3_10eros_max", _model_display_name("minimax_h3_10eros_max")
    if "sulphur3" in low:
        return "minimax_h3_sulphur3", _model_display_name("minimax_h3_sulphur3")
    return "comfyui_studio", _model_display_name("comfyui_studio")


def _resolve_task_input_paths(items: Optional[List[str]]) -> List[str]:
    resolved: List[str] = []
    for raw in (items or []):
        s = str(raw or "").strip()
        if not s:
            continue
        try:
            resolved.append(str(Path(s).expanduser().resolve()))
        except Exception:
            resolved.append(s)
    return resolved


def _resolve_task_output_paths(items: Optional[List[str]], project_dir: str = "", output_dir: str = "") -> List[str]:
    resolved: List[str] = []
    bases: List[Path] = []
    for b in (str(Path.cwd()), project_dir, output_dir):
        if b and str(b).strip():
            try:
                bases.append(Path(str(b)).expanduser())
            except Exception:
                pass
    for raw in (items or []):
        s = str(raw or "").strip()
        if not s:
            continue
        p = Path(s).expanduser()
        if p.is_absolute():
            try:
                resolved.append(str(p.resolve()))
            except Exception:
                resolved.append(str(p))
            continue
        matched: Optional[str] = None
        for base in bases:
            cand = base / p
            if cand.exists():
                try:
                    matched = str(cand.resolve())
                except Exception:
                    matched = str(cand)
                break
        if not matched:
            try:
                matched = str(p.resolve())
            except Exception:
                matched = s
        resolved.append(matched)
    return resolved


def _begin_external_task(
    *,
    command: str,
    model_id: str,
    model_name: str,
    task_mode: str,
    prompt: str = "",
    soundscape: str = "",
    music: str = "",
    params_summary: str = "",
    params: Optional[Dict[str, Any]] = None,
    input_files: Optional[List[str]] = None,
    out_dir: str = "./outputs",
    prefix: str = "",
    account: str = "",
) -> Dict[str, Any]:
    global _CURRENT_EXTERNAL_TASK
    proj_name, proj_dir = _infer_project_info(out_dir, prefix=prefix)
    try:
        resolved_out = str(Path(out_dir or "./outputs").expanduser().resolve())
    except Exception:
        resolved_out = str(out_dir or "./outputs")
    acct = (account or os.environ.get("EASYCOLAB_ACCOUNT", "")).strip()
    if not acct:
        try:
            acct = str(_load_session().get("account") or "").strip()
        except Exception:
            acct = ""
    now = round(time.time(), 3)
    payload: Dict[str, Any] = {
        "task_id": uuid.uuid4().hex[:12],
        "status": "running",
        "source": "colab-h3-cli",
        "command": command,
        "account": acct,
        "project_name": proj_name,
        "project_dir": proj_dir,
        "output_dir": resolved_out,
        "prefix": prefix or "",
        "model_id": model_id,
        "model_name": model_name,
        "task_mode": task_mode,
        "prompt": (prompt or "").strip(),
        "soundscape": (soundscape or "").strip(),
        "music": (music or "").strip(),
        "params_summary": (params_summary or "").strip(),
        "params": params or {},
        "input_files": [Path(x).name for x in (input_files or []) if str(x).strip()],
        "input_paths": _resolve_task_input_paths(input_files),
        "output_files": [],
        "progress": 0.02,
        "progress_label": f"已提交 {model_name} 任务...",
        "error": "",
        "started_at": now,
        "updated_at": now,
    }
    _CURRENT_EXTERNAL_TASK = payload
    _sync_app_external_task(payload)
    return payload


def _update_external_task_progress(progress: float, label: str = "") -> None:
    if not _CURRENT_EXTERNAL_TASK:
        return
    _CURRENT_EXTERNAL_TASK["progress"] = round(max(0.0, min(1.0, float(progress or 0.0))), 3)
    if label:
        _CURRENT_EXTERNAL_TASK["progress_label"] = label
    _CURRENT_EXTERNAL_TASK["updated_at"] = round(time.time(), 3)
    _sync_app_external_task(_CURRENT_EXTERNAL_TASK)


def _finish_external_task(output_files: Optional[List[str]] = None, error: str = "") -> None:
    global _CURRENT_EXTERNAL_TASK
    if not _CURRENT_EXTERNAL_TASK:
        return
    now = round(time.time(), 3)
    if error:
        _CURRENT_EXTERNAL_TASK["status"] = "failed"
        _CURRENT_EXTERNAL_TASK["error"] = error
        _CURRENT_EXTERNAL_TASK["progress_label"] = f"任务失败: {error[:160]}"
    else:
        _CURRENT_EXTERNAL_TASK["status"] = "completed"
        _CURRENT_EXTERNAL_TASK["progress"] = 1.0
        _CURRENT_EXTERNAL_TASK["output_files"] = _resolve_task_output_paths(
            output_files,
            project_dir=str(_CURRENT_EXTERNAL_TASK.get("project_dir") or ""),
            output_dir=str(_CURRENT_EXTERNAL_TASK.get("output_dir") or ""),
        )
        _CURRENT_EXTERNAL_TASK["progress_label"] = "生成完成"
    _CURRENT_EXTERNAL_TASK["updated_at"] = now
    _sync_app_external_task(_CURRENT_EXTERNAL_TASK)
    _CURRENT_EXTERNAL_TASK = None


def _sync_app_active_session(
    url: str,
    password: str,
    gpu: str = "",
    comfy_url: str = "",
    sleeping: bool = False,
    source: str = "colab-h3-cli",
) -> None:
    """Publish active/sleeping Colab session state so EasyColab macOS app auto-attaches in real time."""
    try:
        target = Path(
            os.environ.get("EASYCOLAB_ACTIVE_SESSION_PATH", "").strip() or str(ACTIVE_SESSION_PATH)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "url": (url or "").strip().rstrip("/"),
            "password": password or "",
            "gpu": gpu or "",
            "comfy_url": (comfy_url or "").strip().rstrip("/"),
            "account": os.environ.get("EASYCOLAB_ACCOUNT", "").strip(),
            "sleeping": bool(sleeping),
            "source": source,
            "updated_at": round(time.time(), 3),
        }
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
        os.chmod(target, 0o600)
    except Exception:
        pass


def _mask_sensitive_text(text: str) -> str:
    """Redact passwords, HF tokens, Basic auth, and full Gradio subdomains from log lines."""
    if not text:
        return ""
    masked = re.sub(
        r"(--(?:password|hf-token)\s+)(\S+)",
        lambda m: f"{m.group(1)}***",
        text,
    )
    masked = re.sub(r"(Basic\s+)[A-Za-z0-9+/=]+", r"\1***", masked)
    masked = re.sub(
        r"https://([A-Za-z0-9-]{4})[A-Za-z0-9.-]*(\.gradio\.live)",
        r"https://\1***\2",
        masked,
    )
    return masked


def _emit_app_log(message: str, source: str = "colab-h3-cli", level: str = "info") -> None:
    """Append a structured log entry so the EasyColab macOS app displays external CLI/skill logs live."""
    try:
        clean = _mask_sensitive_text(str(message or "")).strip()
        if not clean:
            return
        log_path = Path(
            os.environ.get("EASYCOLAB_APP_LOG_PATH", "").strip() or str(APP_LOG_PATH)
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.is_file() and log_path.stat().st_size > 512 * 1024:
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                log_path.write_text("\n".join(tail) + "\n", encoding="utf-8")
            except Exception:
                pass
        ts_str = time.strftime("%H:%M:%S")
        entry = {
            "ts": round(time.time(), 3),
            "source": source,
            "level": level,
            "line": f"[{source} {ts_str}] {clean}",
        }
        with open(log_path, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            wf.flush()
    except Exception:
        pass


def _find_easycolab_root() -> Path:
    candidates = [
        os.environ.get("EASYCOLAB_ROOT", "").strip(),
        str(Path.cwd()),
        DEFAULT_EASYCOLAB_ROOT,
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if (p / "colab_bridge.py").is_file() and (p / "app.py").is_file():
            return p
    return Path(DEFAULT_EASYCOLAB_ROOT)


def _find_bridge_python(root: Path) -> str:
    candidates = [
        os.environ.get("EASYCOLAB_BRIDGE_PYTHON", "").strip(),
        str(Path.home() / "Library" / "Application Support" / "EasyColab" / "venv" / "bin" / "python"),
        str(root / ".venv" / "bin" / "python"),
        sys.executable,
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return sys.executable


def _load_session() -> Dict[str, Any]:
    sess: Dict[str, Any] = {}
    sess_ts: float = 0.0
    if SESSION_PATH.is_file():
        try:
            sess = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
            sess_ts = float(sess.get("updated_at") or 0.0)
            if sess_ts <= 0.0:
                sess_ts = float(SESSION_PATH.stat().st_mtime)
        except Exception:
            sess = {}
            sess_ts = 0.0

    # Adopt a fresher live tunnel session published by EasyColab.app / colab_bridge
    env_act = os.environ.get("EASYCOLAB_ACTIVE_SESSION_PATH", "").strip()
    act_path = Path(env_act) if env_act else ACTIVE_SESSION_PATH
    if act_path.is_file():
        try:
            act = json.loads(act_path.read_text(encoding="utf-8"))
            act_url = str(act.get("url") or "").strip().rstrip("/")
            if not act.get("sleeping") and act_url.startswith("https://"):
                cur_acct = (os.environ.get("EASYCOLAB_ACCOUNT", "").strip() or str(sess.get("account") or "").strip()).lower()
                act_acct = str(act.get("account") or "").strip().lower()
                if not cur_acct or not act_acct or cur_acct == act_acct:
                    act_ts = float(act.get("updated_at") or 0.0)
                    if act_ts <= 0.0:
                        act_ts = float(act_path.stat().st_mtime)
                    if not sess.get("url") or act_ts >= sess_ts:
                        sess["url"] = act_url
                        if act.get("password"):
                            sess["password"] = str(act["password"])
                        if act.get("gpu"):
                            sess["gpu"] = str(act["gpu"])
                        if act.get("comfy_url"):
                            sess["comfy_url"] = str(act["comfy_url"]).rstrip("/")
                        if act.get("account"):
                            sess["account"] = str(act["account"])
                        sess["sleeping"] = False
                        sess["updated_at"] = act_ts
        except Exception:
            pass
    return sess


def _save_session(data: Dict[str, Any]) -> None:
    if isinstance(data, dict):
        data["updated_at"] = round(time.time(), 3)
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SESSION_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, SESSION_PATH)
    os.chmod(SESSION_PATH, 0o600)


def _generate_admin_password() -> str:
    # runtime_auth.py requires [A-Za-z0-9_-]{24,256}
    token = secrets.token_urlsafe(24)
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", token)
    return clean[:32] if len(clean) >= 24 else (clean + "EcH3ColabAuth9999")[:32]


def _resolve_connection(args: argparse.Namespace) -> Tuple[str, str]:
    sess = _load_session()
    url = (
        getattr(args, "url", None)
        or os.environ.get("EASYCOLAB_URL", "").strip()
        or str(sess.get("url") or "").strip()
    ).rstrip("/")
    password = (
        getattr(args, "password", None)
        or os.environ.get("EASYCOLAB_PASSWORD", "").strip()
        or str(sess.get("password") or "").strip()
    )
    if not url:
        raise RuntimeError("Missing EasyColab URL. Run `wake` or `connect --url ... --password ...` first.")
    return url, password


def _auth_headers(password: str, base_url: str = "") -> Dict[str, str]:
    headers: Dict[str, str] = {"User-Agent": "EasyColab-H3-Skill/1.0"}
    if password:
        raw = f"easycolab:{password}".encode("utf-8")
        headers["Authorization"] = "Basic " + base64.b64encode(raw).decode("ascii")
    if base_url:
        headers["Origin"] = base_url.rstrip("/")
    return headers


def _http_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    password: str = "",
    base_url: str = "",
    timeout: int = 30,
) -> Dict[str, Any]:
    headers = _auth_headers(password, base_url)
    data_bytes = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {url}: {err_body[:400]}") from e


def _http_download(url: str, dst_path: Path, password: str = "", base_url: str = "", timeout: int = 180) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(dst_path.suffix + ".part")
    headers = _auth_headers(password, base_url)
    total_size = 0
    supports_range = False
    try:
        probe_req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with urllib.request.urlopen(probe_req, timeout=min(25, timeout)) as probe_resp:
            cr = probe_resp.headers.get("Content-Range") or ""
            m = re.match(r"bytes\s+0-0/(\d+)", cr.strip(), re.IGNORECASE)
            if probe_resp.status == 206 and m:
                total_size = int(m.group(1))
                supports_range = total_size >= 2 * 1024 * 1024
    except Exception:
        supports_range = False

    if supports_range and total_size > 0:
        from concurrent.futures import ThreadPoolExecutor

        chunk_size = max(1024 * 1024, (total_size + 5) // 6)
        ranges = []
        pos = 0
        while pos < total_size:
            end = min(total_size - 1, pos + chunk_size - 1)
            ranges.append((pos, end))
            pos = end + 1
        try:
            with open(tmp, "wb") as wf:
                wf.truncate(total_size)

            def _fetch_part(span: Tuple[int, int]) -> None:
                start, end = span
                part_req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(part_req, timeout=timeout) as resp:
                    if resp.status not in (200, 206):
                        raise RuntimeError(f"Unexpected HTTP {resp.status} for byte range {start}-{end}")
                    data = resp.read()
                with open(tmp, "r+b") as wf:
                    wf.seek(start)
                    wf.write(data)

            with ThreadPoolExecutor(max_workers=min(6, len(ranges))) as pool:
                list(pool.map(_fetch_part, ranges))
            if tmp.stat().st_size == total_size:
                os.replace(tmp, dst_path)
                return dst_path
        except Exception:
            pass

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as wf:
        shutil.copyfileobj(resp, wf, length=1024 * 1024)
    os.replace(tmp, dst_path)
    return dst_path


def _run_bridge(bridge_args: List[str], env_extra: Optional[Dict[str, str]] = None, stream: bool = False) -> Tuple[int, str]:
    root = _find_easycolab_root()
    bridge_py = root / "colab_bridge.py"
    if not bridge_py.is_file():
        raise RuntimeError(f"colab_bridge.py not found under {root}. Set EASYCOLAB_ROOT.")
    py_bin = _find_bridge_python(root)
    env = os.environ.copy()
    env["EASYCOLAB_SUPPRESS_BRIDGE_LOG"] = "1"
    if env_extra:
        env.update(env_extra)
    cmd = [py_bin, str(bridge_py), *bridge_args]
    if not stream:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return proc.returncode, out.strip()

    lines: List[str] = []
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for raw in proc.stdout:
        ln = raw.rstrip("\n")
        lines.append(ln)
        print(ln, flush=True)
        if ln.startswith("WAKE_LOG:"):
            _emit_app_log(ln[len("WAKE_LOG:"):].strip())
        elif ln.startswith("WAKE_URL:"):
            _emit_app_log(f"成功获取云端隧道地址: {ln[len('WAKE_URL:'):].strip()}")
        elif ln.startswith("WAKE_ERROR:"):
            _emit_app_log(ln[len("WAKE_ERROR:"):].strip(), level="error")
        elif ln.strip() and not ln.startswith("[colab]") and "urllib3" not in ln and "Assignment request failed" not in ln:
            _emit_app_log(ln.strip())
    rc = proc.wait()
    return rc, "\n".join(lines)


def snap_frames_to_h3_grid(frames: int) -> Tuple[int, int]:
    """Snap frame count up to MiniMax-H3's 17k+5 grid (min 5)."""
    n = max(5, int(frames))
    snapped = n + (5 - (n % 17)) % 17
    k = (snapped - 5) // 17
    return snapped, k


def _encode_image_b64(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _comfy_upload_file(comfy_url: str, password: str, local_path: str, file_type: str = "input") -> str:
    p = Path(local_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Input media not found: {p}")
    boundary = f"----EasyColabH3{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    raw = p.read_bytes()

    body = bytearray()
    for field_name, field_val in (("type", file_type), ("overwrite", "true")):
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n{field_val}\r\n'.encode("utf-8"))
    body.extend(f"--{boundary}\r\n".encode("ascii"))
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{p.name}"\r\nContent-Type: {mime}\r\n\r\n'.encode("utf-8")
    )
    body.extend(raw)
    body.extend(f"\r\n--{boundary}--\r\n".encode("ascii"))

    headers = _auth_headers(password, comfy_url)
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(f"{comfy_url.rstrip('/')}/upload/image", data=bytes(body), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8", errors="replace"))
    sub = str(res.get("subfolder") or "").strip("/")
    name = str(res.get("name") or p.name)
    return f"{sub}/{name}" if sub else name


def _resolve_comfy_url(args: argparse.Namespace) -> Tuple[str, str]:
    sess = _load_session()
    password = (
        getattr(args, "password", None)
        or os.environ.get("EASYCOLAB_PASSWORD", "").strip()
        or str(sess.get("password") or "").strip()
    )
    explicit = (getattr(args, "comfy_url", None) or os.environ.get("EASYCOLAB_COMFY_URL", "").strip()).rstrip("/")
    if explicit:
        return explicit, password
    try:
        base_url, base_pw = _resolve_connection(args)
        password = password or base_pw
        st = _http_json(f"{base_url}/minimax_api/comfyui/status", password=password, base_url=base_url)
        cf = st.get("comfyui") or {}
        pub = str(cf.get("public_url") or "").strip().rstrip("/")
        if pub:
            sess.update({"url": base_url, "password": password, "comfy_url": pub})
            _save_session(sess)
            return pub, password
    except Exception:
        pass
    cached_cf = str(sess.get("comfy_url") or "").strip().rstrip("/")
    if cached_cf:
        return cached_cf, password
    raise RuntimeError("ComfyUI public_url unavailable. Start ComfyUI with `comfy-start --wait` or pass `--comfy-url`.")


def _build_10eros_api_prompt(
    mode: str,
    prompt_text: str,
    ref_images: List[str],
    ref_audios: List[str],
    first_frame: str,
    last_frame: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    sampler: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    ref_image_size: str,
    seed: int,
    prefix: str,
    unet_name: str,
    clip_name: str,
) -> Dict[str, Any]:
    if len(ref_images) > 9:
        raise ValueError("MiniMaxH3ReferenceToVideo supports at most 9 reference images (<Picture 1>..<Picture 9>).")
    if len(ref_audios) > 3:
        raise ValueError("MiniMaxH3ReferenceToVideo supports at most 3 reference audios (<Audio 1>..<Audio 3>).")

    snapped, _ = snap_frames_to_h3_grid(frames)
    actual_seed = seed if seed >= 0 else secrets.randbelow(2**31 - 1)

    graph: Dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": shift_video, "shift_audio": shift_audio}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "9": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
        "12": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["8", 0]}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["12", 0], "sampler": ["10", 0], "sigmas": ["9", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}},
        "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24.0, "bit_depth": "auto", "color_space": "sRGB", "chroma_subsampling": "none"}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }

    if mode == "multiref":
        cond_inputs: Dict[str, Any] = {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "audio_vae": ["5", 0],
            "prompt": prompt_text,
            "width": int(width),
            "height": int(height),
            "length": snapped,
            "ref_image_size": ref_image_size,
        }
        next_id = 30
        for idx, img_name in enumerate(ref_images):
            nid = str(next_id)
            next_id += 1
            graph[nid] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
            cond_inputs[f"ref_images.ref_image_{idx}"] = [nid, 0]
        for idx, aud_name in enumerate(ref_audios):
            nid = str(next_id)
            next_id += 1
            graph[nid] = {"class_type": "LoadAudio", "inputs": {"audio": aud_name}}
            cond_inputs[f"ref_audios.ref_audio_{idx}"] = [nid, 0]
        graph["8"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    else:
        cond_inputs = {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "prompt": prompt_text,
            "width": int(width),
            "height": int(height),
            "length": snapped,
        }
        if first_frame:
            graph["30"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
            cond_inputs["first_frame"] = ["30", 0]
        if last_frame:
            graph["31"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
            cond_inputs["last_frame"] = ["31", 0]
        graph["8"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}

    return graph


def _build_sulphur3_api_prompt(
    mode: str,
    prompt_text: str,
    first_frame: str,
    last_frame: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    sampler: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    corrector_strength: float,
    budget_fraction: float,
    velocity_floor: float,
    start_step: int,
    end_step: int,
    seed: int,
    prefix: str,
    unet_name: str,
    clip_name: str,
    corrector_name: str,
) -> Dict[str, Any]:
    snapped, _ = snap_frames_to_h3_grid(frames)
    actual_seed = seed if seed >= 0 else secrets.randbelow(2**31 - 1)

    graph: Dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": shift_video, "shift_audio": shift_audio}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "9": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": scheduler, "steps": steps, "denoise": 1.0}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}},
        "17": {"class_type": "LoadH3Corrector", "inputs": {"corrector": corrector_name}},
        "18": {
            "class_type": "ApplyH3Corrector",
            "inputs": {
                "model": ["2", 0],
                "corrector": ["17", 0],
                "strength": float(corrector_strength),
                "budget_fraction": float(budget_fraction),
                "velocity_floor": float(velocity_floor),
                "start_step": int(start_step),
                "end_step": int(end_step),
            },
        },
        "19": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["18", 0],
                "positive": ["6", 0] if mode == "t2va" else ["8", 0],
                "negative": ["7", 0],
                "cfg": 1.0,
            },
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["19", 0],
                "sampler": ["10", 0],
                "sigmas": ["9", 0],
                "latent_image": ["8", 0] if mode == "t2va" else ["8", 1],
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24.0, "bit_depth": "auto", "color_space": "sRGB", "chroma_subsampling": "none"}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }

    if mode == "t2va":
        graph["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt_text}}
        graph["8"] = {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": int(width), "height": int(height), "length": snapped}}
    else:
        cond_inputs: Dict[str, Any] = {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "prompt": prompt_text,
            "width": int(width),
            "height": int(height),
            "length": snapped,
        }
        if first_frame:
            graph["20"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
            cond_inputs["first_frame"] = ["20", 0]
        if last_frame:
            graph["21"] = {"class_type": "LoadImage", "inputs": {"image": last_frame}}
            cond_inputs["last_frame"] = ["21", 0]
        graph["8"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": cond_inputs}

    return graph


def _build_ref2va_tara_api_prompt(
    prompt_text: str,
    ref_video: str,
    ref_images: List[str],
    ref_audios: List[str],
    width: int,
    height: int,
    frames: int,
    steps: int,
    sampler: str,
    scheduler: str,
    shift_video: float,
    shift_audio: float,
    ref_image_size: str,
    lora_strength: float,
    tara_enabled: bool,
    tara_strength: float,
    tara_window: float,
    tara_falloff: float,
    tara_max_penalty: float,
    seed: int,
    prefix: str,
    unet_name: str,
    clip_name: str,
) -> Dict[str, Any]:
    if len(ref_images) > 9:
        raise ValueError("MiniMaxH3ReferenceToVideo supports at most 9 reference images (<Picture 1>..<Picture 9>).")
    if len(ref_audios) > 3:
        raise ValueError("MiniMaxH3ReferenceToVideo supports at most 3 reference audios (<Audio 1>..<Audio 3>).")

    snapped, _ = snap_frames_to_h3_grid(frames)
    actual_seed = seed if seed >= 0 else secrets.randbelow(2**31 - 1)

    graph: Dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "30": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                "strength_model": 1.0,
            },
        },
        "31": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["30", 0],
                "lora_name": "minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors",
                "strength_model": float(lora_strength),
            },
        },
        "2": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["31", 0], "shift_video": shift_video, "shift_audio": shift_audio}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    }

    model_ref: List[Any] = ["2", 0]
    if tara_enabled:
        graph["32"] = {
            "class_type": "BFSMiniMaxH3NativeTemporal",
            "inputs": {
                "model": ["2", 0],
                "motion_video_index": 1,
                "strength": float(tara_strength),
                "window_seconds": float(tara_window),
                "falloff_seconds": float(tara_falloff),
                "max_penalty": float(tara_max_penalty),
                "attention_backend": "flash partitions (no compile)",
                "structure_guidance": 1.0,
                "structure_sigma_min": 0.0,
            },
        }
        model_ref = ["32", 0]

    cond_inputs: Dict[str, Any] = {
        "clip": ["3", 0],
        "vae": ["4", 0],
        "audio_vae": ["5", 0],
        "prompt": prompt_text,
        "width": int(width),
        "height": int(height),
        "length": snapped,
        "ref_image_size": ref_image_size,
    }

    next_id = 40
    for idx, img_name in enumerate(ref_images):
        nid = str(next_id)
        next_id += 1
        graph[nid] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
        cond_inputs[f"ref_images.ref_image_{idx}"] = [nid, 0]

    graph["29"] = {
        "class_type": "VHS_LoadVideo",
        "inputs": {
            "video": ref_video,
            "force_rate": 24.0,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": snapped,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        },
    }
    cond_inputs["ref_videos.ref_video_0"] = ["29", 0]

    for idx, aud_name in enumerate(ref_audios):
        nid = str(next_id)
        next_id += 1
        graph[nid] = {"class_type": "LoadAudio", "inputs": {"audio": aud_name}}
        cond_inputs[f"ref_audios.ref_audio_{idx}"] = [nid, 0]

    graph["8"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond_inputs}
    graph["9"] = {"class_type": "BasicScheduler", "inputs": {"model": model_ref, "scheduler": scheduler, "steps": steps, "denoise": 1.0}}
    graph["10"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}}
    graph["11"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": actual_seed}}
    graph["12"] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["8", 0]}}
    graph["13"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["12", 0], "sampler": ["10", 0], "sigmas": ["9", 0], "latent_image": ["8", 1]}}
    graph["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["4", 0]}}
    graph["15"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}}
    graph["16"] = {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "audio": ["15", 0], "fps": 24.0}}
    graph["17"] = {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}
    return graph


def _snap_seedvr2_batch_size(batch_size: int) -> int:
    n = max(1, int(batch_size or 17))
    k = max(0, round((n - 1) / 4.0))
    return 4 * k + 1


def _build_seedvr2_upscale_api_prompt(
    video_name: str,
    model_name: str,
    resolution: int,
    max_resolution: int,
    batch_size: int,
    temporal_overlap: int,
    color_correction: str,
    keep_audio: bool,
    attention_mode: str,
    seed: int,
    prefix: str,
    decode_tile_size: int = 2048,
) -> Dict[str, Any]:
    actual_seed = seed if seed >= 0 else secrets.randbelow(2**31 - 1)
    snapped_batch = _snap_seedvr2_batch_size(batch_size)
    res = max(256, (int(resolution or 1080) // 2) * 2)
    max_res = max(0, (int(max_resolution or 0) // 2) * 2)
    decode_tiled = res >= 1440
    tile_sz = max(512, (int(decode_tile_size or 2048) // 64) * 64)

    create_inputs: Dict[str, Any] = {
        "images": ["4", 0],
        "fps": ["5", 5],
    }
    if keep_audio:
        create_inputs["audio"] = ["1", 2]

    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "2": {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {
                "model": model_name,
                "device": "cuda:0",
                "blocks_to_swap": 0,
                "swap_io_components": False,
                "offload_device": "none",
                "cache_model": False,
                "attention_mode": attention_mode,
            },
        },
        "3": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": "ema_vae_fp16.safetensors",
                "device": "cuda:0",
                "encode_tiled": False,
                "encode_tile_size": 1024,
                "encode_tile_overlap": 128,
                "decode_tiled": decode_tiled,
                "decode_tile_size": tile_sz,
                "decode_tile_overlap": 128,
                "tile_debug": "false",
                "offload_device": "none",
                "cache_model": False,
            },
        },
        "4": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "image": ["1", 0],
                "dit": ["2", 0],
                "vae": ["3", 0],
                "seed": int(actual_seed) % (2**32 - 1),
                "resolution": res,
                "max_resolution": max_res,
                "batch_size": snapped_batch,
                "uniform_batch_size": True,
                "temporal_overlap": max(0, min(16, int(temporal_overlap))),
                "prepend_frames": 0,
                "color_correction": color_correction,
                "input_noise_scale": 0.0,
                "latent_noise_scale": 0.0,
                "offload_device": "none",
                "enable_debug": False,
            },
        },
        "5": {
            "class_type": "VHS_VideoInfo",
            "inputs": {"video_info": ["1", 3]},
        },
        "6": {
            "class_type": "CreateVideo",
            "inputs": create_inputs,
        },
        "7": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["6", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def _build_recavsr_upscale_api_prompt(
    video_name: str,
    transformer: str,
    scale: float,
    color_fix: str,
    kernels: str,
    encoder: str,
    seed: int,
    prefix: str,
) -> Dict[str, Any]:
    actual_seed = seed if seed >= 0 else secrets.randbelow(2**31 - 1)
    scale_val = max(0.5, min(8.0, round(float(scale or 2.0), 2)))
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "2": {"class_type": "ReCaVSRLoader", "inputs": {"transformer": transformer}},
        "3": {
            "class_type": "ReCaVSRUpscale",
            "inputs": {
                "model": ["2", 0],
                "video": ["1", 0],
                "scale": scale_val,
                "color_fix": color_fix,
                "seed": int(actual_seed) % (2**32 - 1),
                "decoder": "flashdecoder",
                "kernels": kernels,
                "kv_offload": False,
                "encoder": encoder,
                "fps": 30.0,
            },
        },
        "4": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["3", 0],
                "filename_prefix": prefix,
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def _format_progress_bar(fraction: float, width: int = 10) -> str:
    frac = max(0.0, min(1.0, float(fraction or 0.0)))
    filled = max(0, min(width, int(round(width * frac))))
    return ("█" * filled) + ("-" * (width - filled))


def _describe_comfy_node_stage(class_type: str) -> str:
    ct = (class_type or "").strip()
    low = ct.lower()
    if not ct:
        return "处理中"
    if any(k in low for k in ("loader", "checkpoint", "loadimage", "loadvideo", "loadaudio", "loadh3")):
        return "加载模型与输入"
    if "cliptext" in low or "encode" in low:
        return "编码文本与条件特征"
    if "recavsr" in low:
        return "ReCaVSR 流式超分"
    if "seedvr2" in low:
        return "SeedVR2 扩散超分"
    if any(k in low for k in ("sampler", "ksampler", "bfsminimax")):
        return "扩散去噪采样"
    if "vaedecodeaudio" in low:
        return "立体声音频 VAE 解码"
    if "vaedecode" in low:
        return "3D 视频 / 图像 VAE 解码"
    if any(k in low for k in ("createvideo", "savevideo", "videocombine", "saveimage")):
        return "合成与封装成片"
    return ct


class _ComfyWSProgressTracker:
    """Pure-stdlib ComfyUI WebSocket listener providing step/node progress + ETA."""

    def __init__(self, comfy_url: str, password: str, client_id: str, graph: Dict[str, Any]):
        self.comfy_url = comfy_url.rstrip("/")
        self.password = password or ""
        self.client_id = client_id
        self.nodes_map: Dict[str, str] = {
            str(nid): str(node.get("class_type") or "")
            for nid, node in (graph or {}).items()
            if isinstance(node, dict)
        }
        self._lock = threading.Lock()
        self.prompt_id: str = ""
        self.done_nodes: set[str] = set()
        self.node_id: str = ""
        self.node_type: str = ""
        self.stage_label: str = "准备执行"
        self.step_value: int = 0
        self.step_max: int = 0
        self.progress: float = 0.0
        self.eta_s: Optional[float] = None
        self.started_at: float = time.time()
        self.step_started_at: float = self.started_at
        self._version: int = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock = None

    def set_prompt_id(self, prompt_id: str) -> None:
        with self._lock:
            self.prompt_id = str(prompt_id or "").strip()

    def handle_event(self, evt: Dict[str, Any], now_ts: Optional[float] = None) -> bool:
        if not isinstance(evt, dict):
            return False
        mtype = str(evt.get("type") or "").strip()
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        ts = float(now_ts if now_ts is not None else time.time())
        with self._lock:
            ev_pid = str(data.get("prompt_id") or "").strip()
            if self.prompt_id and ev_pid and ev_pid != self.prompt_id:
                return False

            total_nodes = max(len(self.nodes_map), len(self.done_nodes) + 1)
            if mtype == "execution_start":
                if ev_pid and not self.prompt_id:
                    self.prompt_id = ev_pid
                self.started_at = ts
                self.step_started_at = ts
                self.progress = max(self.progress, 0.03)
                self.stage_label = "初始化节点图"
                self._version += 1
                return True

            if mtype == "execution_cached":
                for n in data.get("nodes") or []:
                    if n is not None:
                        self.done_nodes.add(str(n))
                self._version += 1
                return True

            if mtype == "executing":
                node = data.get("node")
                if node is None:
                    self.progress = 1.0
                    self.eta_s = 0.0
                    self.stage_label = "节点图执行完成"
                    self._version += 1
                    return True
                nid = str(node)
                if self.node_id and self.node_id != nid:
                    self.done_nodes.add(self.node_id)
                self.node_id = nid
                self.node_type = self.nodes_map.get(nid, nid)
                self.stage_label = _describe_comfy_node_stage(self.node_type)
                self.step_value = 0
                self.step_max = 0
                self.step_started_at = ts
                self.eta_s = None
                total_nodes = max(len(self.nodes_map), len(self.done_nodes) + 1)
                self.progress = max(0.04, min(0.95, 0.04 + 0.91 * (len(self.done_nodes) / float(max(1, total_nodes)))))
                self._version += 1
                return True

            if mtype == "executed":
                nid = str(data.get("node") or "")
                if nid:
                    self.done_nodes.add(nid)
                    self._version += 1
                    return True
                return False

            if mtype in ("progress", "progress_state"):
                if mtype == "progress":
                    val = int(data.get("value") or 0)
                    mx = max(1, int(data.get("max") or 1))
                    nid = str(data.get("node") or self.node_id or "")
                else:
                    nodes_st = data.get("nodes") if isinstance(data.get("nodes"), dict) else {}
                    running = None
                    for k_nid, info_n in nodes_st.items():
                        if not isinstance(info_n, dict):
                            continue
                        if info_n.get("state") == "finished":
                            self.done_nodes.add(str(k_nid))
                        elif info_n.get("state") == "running":
                            running = (str(k_nid), info_n)
                    if running is None:
                        return False
                    nid, info_n = running
                    val = int(info_n.get("value") or 0)
                    mx = max(1, int(info_n.get("max") or 1))

                if nid:
                    self.node_id = nid
                    self.node_type = self.nodes_map.get(nid, self.node_type or nid)
                    self.stage_label = _describe_comfy_node_stage(self.node_type)
                if val <= 0:
                    self.step_started_at = ts
                    self.eta_s = None
                elif mx > val:
                    step_el = max(0.05, ts - self.step_started_at)
                    self.eta_s = round(step_el * (mx - val) / float(val), 1)
                else:
                    self.eta_s = 0.0
                self.step_value = val
                self.step_max = mx
                step_frac = max(0.0, min(1.0, val / float(mx)))
                total_nodes = max(len(self.nodes_map), len(self.done_nodes) + 1)
                combined = (len(self.done_nodes) + step_frac) / float(max(1, total_nodes))
                self.progress = max(0.05, min(0.98, 0.04 + 0.94 * combined))
                self._version += 1
                return True

            if mtype in ("execution_success", "execution_error", "execution_interrupted"):
                self.progress = 1.0 if mtype == "execution_success" else self.progress
                self.eta_s = 0.0
                self._version += 1
                return True
        return False

    def snapshot(self) -> Tuple[int, float]:
        with self._lock:
            return self._version, self.progress

    def format_status_line(self, elapsed_s: float) -> str:
        with self._lock:
            pct = int(round(max(0.0, min(1.0, self.progress)) * 100))
            bar = _format_progress_bar(self.progress)
            total_nodes = max(len(self.nodes_map), len(self.done_nodes) + 1)
            done_disp = min(total_nodes, len(self.done_nodes) + (1 if self.node_id else 0))
            step_part = f"{self.step_value}/{self.step_max} 步 · " if self.step_max > 0 else ""
            node_name = self.node_type or self.node_id or "ComfyUI"
            eta_part = f" · 剩余约 {int(round(self.eta_s))}s" if (self.eta_s is not None and self.eta_s >= 1.0) else ""
            return (
                f"ComfyUI 进度 {pct}% [{bar}] {step_part}{self.stage_label} "
                f"({node_name} · 节点 {done_disp}/{total_nodes} · 耗时 {int(round(elapsed_s))}s{eta_part})"
            )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass

    def _run(self) -> None:
        parsed = urllib.parse.urlparse(self.comfy_url)
        host = parsed.hostname or ""
        if not host:
            return
        use_ssl = parsed.scheme.lower() == "https"
        port = parsed.port or (443 if use_ssl else 80)
        base_path = parsed.path.rstrip("/")
        ws_path = f"{base_path}/ws?clientId={urllib.parse.quote(self.client_id)}"
        host_hdr = f"{host}:{port}" if parsed.port else host

        while not self._stop_event.is_set():
            sock_obj = None
            try:
                raw_sock = socket.create_connection((host, port), timeout=6.0)
                if use_ssl:
                    ctx = ssl.create_default_context()
                    sock_obj = ctx.wrap_socket(raw_sock, server_hostname=host)
                else:
                    sock_obj = raw_sock
                self._sock = sock_obj

                key = base64.b64encode(os.urandom(16)).decode("ascii")
                hdr_lines = [
                    f"GET {ws_path} HTTP/1.1",
                    f"Host: {host_hdr}",
                    "Upgrade: websocket",
                    "Connection: Upgrade",
                    f"Sec-WebSocket-Key: {key}",
                    "Sec-WebSocket-Version: 13",
                    "User-Agent: EasyColab-H3-Skill/1.0",
                ]
                if self.password:
                    raw_auth = f"easycolab:{self.password}".encode("utf-8")
                    hdr_lines.append("Authorization: Basic " + base64.b64encode(raw_auth).decode("ascii"))
                sock_obj.sendall(("\r\n".join(hdr_lines) + "\r\n\r\n").encode("ascii"))

                buf = bytearray()
                while b"\r\n\r\n" not in buf:
                    chunk = sock_obj.recv(4096)
                    if not chunk:
                        raise EOFError("WS handshake closed")
                    buf.extend(chunk)
                    if len(buf) > 16384:
                        raise ValueError("WS handshake header too large")
                hdr_end = buf.index(b"\r\n\r\n") + 4
                first_line = bytes(buf[:hdr_end]).decode("latin-1", errors="replace").splitlines()[0]
                del buf[:hdr_end]
                if " 101 " not in first_line:
                    raise RuntimeError(f"WS upgrade rejected: {first_line}")

                sock_obj.settimeout(25.0)
                frag_opcode = 0
                frag_buf = bytearray()

                def _read_n(n: int) -> bytes:
                    while len(buf) < n:
                        ch = sock_obj.recv(max(4096, n - len(buf)))
                        if not ch:
                            raise EOFError("WS closed")
                        buf.extend(ch)
                    out = bytes(buf[:n])
                    del buf[:n]
                    return out

                while not self._stop_event.is_set():
                    try:
                        b1, b2 = _read_n(2)
                    except socket.timeout:
                        continue
                    fin = bool(b1 & 0x80)
                    opcode = b1 & 0x0F
                    masked = bool(b2 & 0x80)
                    length = b2 & 0x7F
                    if length == 126:
                        length = struct.unpack(">H", _read_n(2))[0]
                    elif length == 127:
                        length = struct.unpack(">Q", _read_n(8))[0]
                    mask_key = _read_n(4) if masked else b""
                    payload = _read_n(length) if length else b""
                    if masked and payload:
                        payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
                    if opcode == 0x8:
                        break
                    if opcode == 0x9:
                        mk = os.urandom(4)
                        pl = payload[:125]
                        masked_pl = bytes(b ^ mk[i % 4] for i, b in enumerate(pl))
                        sock_obj.sendall(bytes([0x8A, 0x80 | len(pl)]) + mk + masked_pl)
                        continue
                    if opcode == 0xA:
                        continue
                    if opcode in (0x1, 0x2):
                        frag_opcode = opcode
                        frag_buf = bytearray(payload)
                    elif opcode == 0x0:
                        frag_buf.extend(payload)
                    if not fin:
                        continue
                    if frag_opcode == 0x1:
                        try:
                            evt = json.loads(frag_buf.decode("utf-8", errors="replace"))
                        except Exception:
                            evt = None
                        if isinstance(evt, dict):
                            self.handle_event(evt)
            except Exception:
                if self._stop_event.is_set():
                    break
                time.sleep(2.0)
            finally:
                if sock_obj is not None:
                    try:
                        sock_obj.close()
                    except Exception:
                        pass
                self._sock = None


def _submit_and_wait_comfy(comfy_url: str, password: str, graph: Dict[str, Any], out_dir: Path, timeout_s: int = 1800) -> List[str]:
    client_id = uuid.uuid4().hex
    tracker = _ComfyWSProgressTracker(comfy_url, password, client_id, graph)
    tracker.start()
    try:
        res = _http_json(
            f"{comfy_url}/prompt",
            method="POST",
            payload={"prompt": graph, "client_id": client_id},
            password=password,
            base_url=comfy_url,
            timeout=30,
        )
        prompt_id = str(res.get("prompt_id") or "").strip()
        if not prompt_id:
            raise RuntimeError(f"ComfyUI rejected prompt: {json.dumps(res, ensure_ascii=False)}")
        tracker.set_prompt_id(prompt_id)
        msg_sub = f"ComfyUI 进度 2% [{_format_progress_bar(0.02)}] 任务已入队 (prompt_id={prompt_id}, 共 {len(graph)} 节点)..."
        print(msg_sub, flush=True)
        _emit_app_log(msg_sub)

        t_start = time.time()
        deadline = t_start + timeout_s
        history_item: Optional[Dict[str, Any]] = None
        last_emitted_ver = -1
        last_emit_ts = t_start
        while time.time() < deadline:
            hist = _http_json(f"{comfy_url}/history/{prompt_id}", password=password, base_url=comfy_url, timeout=30)
            if prompt_id in hist:
                history_item = hist[prompt_id]
                break

            now = time.time()
            cur_ver, cur_prog = tracker.snapshot()
            if cur_ver > last_emitted_ver and (now - last_emit_ts) >= 2.5:
                prog_line = tracker.format_status_line(now - t_start)
                print(prog_line, flush=True)
                _emit_app_log(prog_line)
                _update_external_task_progress(cur_prog, prog_line)
                last_emitted_ver = cur_ver
                last_emit_ts = now
            elif (now - last_emit_ts) >= 12.0:
                fallback_line = None
                try:
                    q_st = _http_json(f"{comfy_url}/queue", password=password, base_url=comfy_url, timeout=8)
                    q_pend = q_st.get("queue_pending") or []
                    q_pos = next((i + 1 for i, it in enumerate(q_pend) if isinstance(it, list) and len(it) > 1 and it[1] == prompt_id), None)
                    if q_pos is not None:
                        fallback_line = f"ComfyUI 进度 3% [{_format_progress_bar(0.03)}] 队列第 {q_pos} 位等待中 (耗时 {int(now - t_start)}s)..."
                except Exception:
                    pass
                if not fallback_line:
                    fallback_line = tracker.format_status_line(now - t_start)
                print(fallback_line, flush=True)
                _emit_app_log(fallback_line)
                _update_external_task_progress(cur_prog, fallback_line)
                last_emit_ts = now

            time.sleep(2.0)

        if not history_item:
            raise TimeoutError(f"Timed out waiting for ComfyUI prompt_id={prompt_id}")
    finally:
        tracker.stop()

    status_info = history_item.get("status") or {}
    if status_info.get("status_str") == "error":
        err_txt = f"ComfyUI execution error: {json.dumps(status_info, ensure_ascii=False)}"
        _emit_app_log(err_txt, level="error")
        raise RuntimeError(err_txt)

    saved_files: List[str] = []
    outputs = history_item.get("outputs") or {}
    for _, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        for key in ("videos", "gifs", "images"):
            items = node_out.get(key) or []
            for it in items:
                fname = it.get("filename")
                if not fname:
                    continue
                sub = it.get("subfolder", "")
                ftype = it.get("type", "output")
                qs = urllib.parse.urlencode({"filename": fname, "subfolder": sub, "type": ftype})
                dl_url = f"{comfy_url}/view?{qs}"
                local_p = out_dir / fname
                downloaded = False
                if ftype == "output":
                    sess = _load_session()
                    gradio_base = str(sess.get("url") or "").strip().rstrip("/")
                    gradio_pw = str(sess.get("password") or password or "").strip()
                    if gradio_base:
                        rel = f"{sub.strip('/')}/{fname}" if sub.strip("/") else fname
                        encoded_rel = urllib.parse.quote(rel.lstrip("/"), safe="/")
                        gradio_url = f"{gradio_base}/minimax_api/comfyui/output_media/{encoded_rel}"
                        try:
                            _http_download(gradio_url, local_p, password=gradio_pw, base_url=gradio_base, timeout=300)
                            downloaded = True
                        except Exception:
                            downloaded = False
                if not downloaded:
                    _http_download(dl_url, local_p, password=password, base_url=comfy_url, timeout=300)
                saved_files.append(str(local_p))
                _emit_app_log(f"ComfyUI 成片下载完成: {local_p}")
    return saved_files


def _wait_for_native_flag(base_url: str, password: str, flag_key: str, model_id: Optional[str] = None, timeout_s: int = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    qs = f"?model_id={urllib.parse.quote(model_id)}" if model_id else ""
    while time.time() < deadline:
        st = _http_json(f"{base_url}/minimax_api/status{qs}", password=password, base_url=base_url)
        msg = st.get("status_message") or st.get("message") or ""
        prog_val = float(st.get("task_progress") if st.get("task_progress") is not None else (st.get("progress") or 0.0))
        pct = int(round(max(0.0, min(1.0, prog_val)) * 100))
        bar = _format_progress_bar(prog_val)
        line = f"[{flag_key}] 进度 {pct}% [{bar}] {msg}"
        print(line, flush=True)
        _emit_app_log(line)
        _update_external_task_progress(prog_val, line)
        if not st.get(flag_key):
            return st
        time.sleep(4.0)
    raise TimeoutError(f"Timed out waiting for {flag_key}")


def _wait_for_comfy_flag(base_url: str, password: str, flag_key: str, timeout_s: int = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = _http_json(f"{base_url}/minimax_api/comfyui/status", password=password, base_url=base_url)
        cf = st.get("comfyui") or {}
        prog_val = float(cf.get("progress") or 0.0)
        pct = int(round(max(0.0, min(1.0, prog_val)) * 100))
        bar = _format_progress_bar(prog_val)
        line = f"[comfyui:{flag_key}] 进度 {pct}% [{bar}] {cf.get('status_message')}"
        print(line, flush=True)
        _emit_app_log(line)
        _update_external_task_progress(prog_val, line)
        if not cf.get(flag_key):
            return cf
        time.sleep(4.0)
    raise TimeoutError(f"Timed out waiting for comfyui.{flag_key}")


def _poll_native_job(base_url: str, password: str, job_id: str, out_dir: Path, timeout_s: int = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = _http_json(f"{base_url}/minimax_api/job/{job_id}", password=password, base_url=base_url)
        status = job.get("status")
        prog_val = float(job.get("progress") or 0.0)
        pct = int(round(max(0.0, min(1.0, prog_val)) * 100))
        bar = _format_progress_bar(prog_val)
        line = f"[job {job_id}] {status} 进度 {pct}% [{bar}] | {job.get('message')}"
        print(line, flush=True)
        _emit_app_log(line, level=("error" if status == "failed" else "info"))
        if status not in ("completed", "failed"):
            _update_external_task_progress(prog_val, line)
        if status == "completed":
            downloaded: List[str] = []
            items = job.get("completed_videos") or []
            if not items and job.get("filename"):
                items = [{"filename": job["filename"], "video_url": job.get("video_url") or f"/minimax_api/video/{job['filename']}"}]
            for it in items:
                fn = it.get("filename")
                rel_url = it.get("video_url") or (f"/minimax_api/media/{fn}" if fn else "")
                if fn and rel_url:
                    full_url = rel_url if rel_url.startswith("http") else f"{base_url}{rel_url}"
                    dst = out_dir / fn
                    _http_download(full_url, dst, password=password, base_url=base_url, timeout=300)
                    downloaded.append(str(dst))
                    _emit_app_log(f"任务成片已下载至本地: {dst}")
            job["downloaded_files"] = downloaded
            return job
        if status == "failed":
            raise RuntimeError(f"Job {job_id} failed: {job.get('message')}")
        time.sleep(4.0)
    raise TimeoutError(f"Timed out waiting for job {job_id}")


def cmd_snap_frames(args: argparse.Namespace) -> None:
    raw_frames = args.frames if args.frames is not None else round(float(args.seconds) * int(args.fps))
    snapped, k = snap_frames_to_h3_grid(raw_frames)
    print(json.dumps({"requested_frames": raw_frames, "snapped_frames": snapped, "k": k, "duration_sec": round(snapped / float(args.fps), 2)}, indent=2))


def cmd_extract_voice_seed(args: argparse.Namespace) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH.")
    cmd = ["ffmpeg", "-y"]
    if args.start is not None and args.start > 0:
        cmd.extend(["-ss", str(args.start)])
    cmd.extend(["-i", args.input])
    if args.duration is not None and args.duration > 0:
        cmd.extend(["-t", str(args.duration)])
    cmd.extend(["-vn", "-af", "aresample=32000,pan=stereo|c0=c0|c1=c0", args.output])
    subprocess.run(cmd, check=True)
    print(json.dumps({"ok": True, "voice_seed": str(Path(args.output).resolve()), "sample_rate": 32000, "channels": 2}))


def _auto_sleep_runtime(base_url: str, password: str, sync_comfy: bool = False) -> Dict[str, Any]:
    """Sync outputs/ComfyUI (if requested), disconnect runtime, and terminate all Colab VMs via stop_all."""
    _emit_app_log("触发自动休眠：正在同步云端产出并释放 Colab 实例...")
    report: Dict[str, Any] = {
        "triggered": True,
        "synced_comfy": False,
        "disconnect_ok": False,
        "stop_all_rc": -1,
        "stop_all_output": "",
    }
    if sync_comfy and base_url:
        try:
            _http_json(
                f"{base_url}/minimax_api/comfyui/sync_drive",
                method="POST",
                payload={},
                password=password,
                base_url=base_url,
                timeout=90,
            )
            report["synced_comfy"] = True
            _emit_app_log("ComfyUI 资产与产物已同步至 Google Drive。")
        except Exception as sync_err:
            report["sync_error"] = str(sync_err)

    if base_url:
        try:
            _http_json(
                f"{base_url}/minimax_api/disconnect",
                method="POST",
                payload={},
                password=password,
                base_url=base_url,
                timeout=30,
            )
            report["disconnect_ok"] = True
        except Exception as disc_err:
            report["disconnect_error"] = str(disc_err)

    try:
        rc, out = _run_bridge(["stop_all"])
        report["stop_all_rc"] = rc
        report["stop_all_output"] = out
        _emit_app_log(
            f"自动休眠完成：Colab 实例已释放 (rc={rc})。",
            level=("info" if rc == 0 else "error"),
        )
    except Exception as stop_err:
        report["stop_all_error"] = str(stop_err)
        _emit_app_log(f"自动休眠释放异常: {stop_err}", level="error")

    sess = _load_session()
    sess.pop("url", None)
    sess.pop("comfy_url", None)
    sess["sleeping"] = True
    sess["last_sleep_ts"] = int(time.time())
    _save_session(sess)
    _sync_app_active_session("", "", sleeping=True)
    return report


def cmd_wake(args: argparse.Namespace) -> None:
    root = _find_easycolab_root()
    pwd = args.password or _generate_admin_password()
    if not re.fullmatch(r"[A-Za-z0-9_-]{24,256}", pwd):
        raise ValueError("Password must be 24-256 URL-safe ASCII characters ([A-Za-z0-9_-]).")
    _emit_app_log(
        f"正在通过 Skill CLI 唤醒 {args.gpu} GPU 实例并部署 app.py"
        f"{'（已开启持续每 ' + str(int(args.retry_interval)) + 's 自动重申）' if getattr(args, 'persistent', False) else ''}..."
    )
    bridge_args = ["wake", "--gpu", args.gpu, "--app-py", str(root / "app.py")]
    if getattr(args, "persistent", False):
        bridge_args += ["--persistent", "--retry-interval", str(max(5, int(getattr(args, "retry_interval", 60) or 60)))]
    rc, out = _run_bridge(bridge_args, env_extra={"EASYCOLAB_ADMIN_PASSWORD": pwd}, stream=True)
    if rc != 0:
        raise RuntimeError(f"colab_bridge wake exited with code {rc}")
    m = re.search(r"https://[A-Za-z0-9.-]+\.gradio\.live", out)
    if not m:
        raise RuntimeError("Wake completed but no gradio.live URL found in output.")
    url = m.group(0)
    idle_mins = max(1, int(getattr(args, "idle_minutes", 10) or 10))
    idle_configured = False
    try:
        _http_json(
            f"{url}/minimax_api/config_idle",
            method="POST",
            payload={"idle_timeout_minutes": idle_mins},
            password=pwd,
            base_url=url,
            timeout=15,
        )
        idle_configured = True
    except Exception:
        idle_configured = False
    sess = _load_session()
    sess.update({"url": url, "password": pwd, "gpu": args.gpu, "sleeping": False, "idle_timeout_minutes": idle_mins})
    _save_session(sess)
    _sync_app_active_session(url, pwd, gpu=args.gpu, sleeping=False)
    _emit_app_log(f"{args.gpu} 实例已就绪 ({url})，闲置自动释放={idle_mins}m。")
    print(json.dumps({
        "ok": True,
        "url": url,
        "gpu": args.gpu,
        "idle_timeout_minutes": idle_mins,
        "idle_watchdog_set": idle_configured,
        "session_file": str(SESSION_PATH),
    }, indent=2))


def cmd_recover(args: argparse.Namespace) -> None:
    root = _find_easycolab_root()
    sess = _load_session()
    fresh_pwd = args.password or _generate_admin_password()
    if not re.fullmatch(r"[A-Za-z0-9_-]{24,256}", fresh_pwd):
        raise ValueError("Password must be 24-256 URL-safe ASCII characters ([A-Za-z0-9_-]).")
    bridge_args = ["recover_tunnel", "--app-py", str(root / "app.py")]
    if getattr(args, "restart", False):
        bridge_args.append("--restart-server")
    idle_mins = max(1, int(getattr(args, "idle_minutes", 10) or sess.get("idle_timeout_minutes") or 10))
    rc, out = _run_bridge(
        bridge_args,
        env_extra={"EASYCOLAB_ADMIN_PASSWORD": fresh_pwd, "EASYCOLAB_IDLE_TIMEOUT_MINUTES": str(idle_mins)},
        stream=True,
    )
    if rc != 0:
        raise RuntimeError(f"colab_bridge recover_tunnel exited with code {rc}")
    m_wake = re.search(r"^WAKE_URL:(https://[A-Za-z0-9.-]+\.gradio\.live)\s*$", out, flags=re.MULTILINE)
    m_rec = re.search(r"^RECOVER_URL:(https://[A-Za-z0-9.-]+\.gradio\.live)\s*$", out, flags=re.MULTILINE)
    if m_wake:
        url = m_wake.group(1).strip()
        pwd = fresh_pwd
    elif m_rec:
        url = m_rec.group(1).strip()
        pwd = str(sess.get("password") or "")
    else:
        raise RuntimeError(f"Recover did not return a live gradio.live URL:\n{out.strip()}")
    sess.update({"url": url, "sleeping": False})
    if pwd:
        sess["password"] = pwd
    _save_session(sess)
    _sync_app_active_session(url, pwd, gpu=str(sess.get("gpu") or ""), sleeping=False)
    print(json.dumps({"ok": True, "url": url, "restarted": bool(m_wake), "session_file": str(SESSION_PATH)}, indent=2))


def _scoped_session_path(account_email: str) -> Path:
    em = (account_email or "").strip().lower()
    if not em:
        return SESSION_PATH
    acct_tag = hashlib.sha256(em.encode("utf-8")).hexdigest()[:12]
    return Path.home() / ".config" / "colab-minimax-h3" / f"session_{acct_tag}.json"


def _save_scoped_session_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def cmd_wake_multi(args: argparse.Namespace) -> None:
    root = _find_easycolab_root()
    pwd = args.password or _generate_admin_password()
    if not re.fullmatch(r"[A-Za-z0-9_-]{24,256}", pwd):
        raise ValueError("Password must be 24-256 URL-safe ASCII characters ([A-Za-z0-9_-]).")
    idle_mins = max(1, int(getattr(args, "idle_minutes", 10) or 10))
    accounts_csv = (getattr(args, "accounts", "") or "").strip()
    mode = getattr(args, "mode", "first") or "first"

    _emit_app_log(
        f"正在通过 Skill CLI 启动多账号同时申请 ({args.gpu} GPU, mode={mode}, accounts={accounts_csv or 'auto-idle'})..."
    )
    bridge_args = [
        "wake_multi",
        "--gpu",
        args.gpu,
        "--app-py",
        str(root / "app.py"),
        "--mode",
        mode,
    ]
    if accounts_csv:
        bridge_args += ["--accounts", accounts_csv]
    if getattr(args, "persistent", False):
        bridge_args += ["--persistent", "--retry-interval", str(max(5, int(getattr(args, "retry_interval", 60) or 60)))]

    rc, out = _run_bridge(
        bridge_args,
        env_extra={
            "EASYCOLAB_ADMIN_PASSWORD": pwd,
            "EASYCOLAB_IDLE_TIMEOUT_MINUTES": str(idle_mins),
        },
        stream=True,
    )
    if rc != 0:
        raise RuntimeError(f"colab_bridge wake_multi exited with code {rc}")

    if mode == "all":
        m_json = re.findall(r"^WAKE_MULTI_JSON:\s*(\{.*\})\s*$", out, flags=re.MULTILINE)
        if not m_json:
            raise RuntimeError("wake_multi (all) finished without WAKE_MULTI_JSON summary.")
        summary = json.loads(m_json[-1])
        results = summary.get("results") or {}
        prev_acct = os.environ.get("EASYCOLAB_ACCOUNT", "")
        try:
            for em in summary.get("succeeded") or []:
                info = results.get(em) or {}
                url = str(info.get("url") or "").strip().rstrip("/")
                if not url:
                    continue
                try:
                    _http_json(
                        f"{url}/minimax_api/config_idle",
                        method="POST",
                        payload={"idle_timeout_minutes": idle_mins},
                        password=pwd,
                        base_url=url,
                        timeout=15,
                    )
                except Exception:
                    pass
                sp = _scoped_session_path(em)
                sess_obj: Dict[str, Any] = {}
                if sp.is_file():
                    try:
                        sess_obj = json.loads(sp.read_text(encoding="utf-8"))
                    except Exception:
                        sess_obj = {}
                sess_obj.update({"url": url, "password": pwd, "gpu": args.gpu, "account": em, "sleeping": False, "idle_timeout_minutes": idle_mins})
                _save_scoped_session_file(sp, sess_obj)
                os.environ["EASYCOLAB_ACCOUNT"] = em
                _sync_app_active_session(url, pwd, gpu=args.gpu, sleeping=False)
                info["session_file"] = str(sp)
        finally:
            if prev_acct:
                os.environ["EASYCOLAB_ACCOUNT"] = prev_acct
            else:
                os.environ.pop("EASYCOLAB_ACCOUNT", None)
        if not summary.get("ok"):
            raise RuntimeError("No account succeeded in wake_multi (all)")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    m_url = re.search(r"https://[A-Za-z0-9.-]+\.gradio\.live", out)
    if not m_url:
        raise RuntimeError("wake_multi completed but no gradio.live URL found in output.")
    url = m_url.group(0)
    m_win = re.search(r"^WAKE_WINNER:\s*(\S+)\s*$", out, flags=re.MULTILINE)
    winner = m_win.group(1).strip() if m_win else ""
    idle_configured = False
    try:
        _http_json(
            f"{url}/minimax_api/config_idle",
            method="POST",
            payload={"idle_timeout_minutes": idle_mins},
            password=pwd,
            base_url=url,
            timeout=15,
        )
        idle_configured = True
    except Exception:
        idle_configured = False

    sp = _scoped_session_path(winner)
    sess_obj = {}
    if sp.is_file():
        try:
            sess_obj = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            sess_obj = {}
    sess_obj.update({"url": url, "password": pwd, "gpu": args.gpu, "account": winner, "sleeping": False, "idle_timeout_minutes": idle_mins})
    _save_scoped_session_file(sp, sess_obj)
    default_session_path = Path.home() / ".config" / "colab-minimax-h3" / "session.json"
    if sp != default_session_path:
        _save_scoped_session_file(default_session_path, sess_obj)

    switched_default = False
    explicit_account = (getattr(args, "account", "") or "").strip()
    if winner and not explicit_account:
        try:
            sw_rc, _ = _run_bridge(["auth_switch", winner])
            switched_default = (sw_rc == 0)
        except Exception:
            switched_default = False

    prev_acct = os.environ.get("EASYCOLAB_ACCOUNT", "")
    try:
        if winner:
            os.environ["EASYCOLAB_ACCOUNT"] = winner
        _sync_app_active_session(url, pwd, gpu=args.gpu, sleeping=False)
    finally:
        if prev_acct:
            os.environ["EASYCOLAB_ACCOUNT"] = prev_acct
        else:
            os.environ.pop("EASYCOLAB_ACCOUNT", None)
    _emit_app_log(f"多账号竞速成功：[{winner or 'active'}] {args.gpu} 就绪 ({url})。")
    print(json.dumps({
        "ok": True,
        "mode": "first",
        "winner": winner,
        "switched_default_account": switched_default,
        "url": url,
        "gpu": args.gpu,
        "idle_timeout_minutes": idle_mins,
        "idle_watchdog_set": idle_configured,
        "session_file": str(sp),
    }, indent=2))


def cmd_connect(args: argparse.Namespace) -> None:
    url = args.url.rstrip("/")
    pwd = args.password or ""
    st = _http_json(f"{url}/minimax_api/status", password=pwd, base_url=url)
    sess = _load_session()
    sess.update({"url": url, "password": pwd})
    if args.comfy_url:
        sess["comfy_url"] = args.comfy_url.rstrip("/")
    elif isinstance(st.get("comfyui"), dict) and st["comfyui"].get("public_url"):
        sess["comfy_url"] = str(st["comfyui"]["public_url"]).rstrip("/")
    _save_session(sess)
    _sync_app_active_session(
        url,
        pwd,
        gpu=str(st.get("gpu_name") or ""),
        comfy_url=str(sess.get("comfy_url") or ""),
        sleeping=False,
    )
    _emit_app_log(f"已绑定云端实例 {url} (GPU: {st.get('gpu_name')})")
    print(json.dumps({"ok": True, "url": url, "comfy_url": sess.get("comfy_url", ""), "gpu_name": st.get("gpu_name")}, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Colab MiniMax-H3 & EasyColab CLI")
    parser.add_argument("--account", default="", help="Pin execution & session state to a specific saved Google account email (for multi-GPU concurrency)")
    parser.add_argument("--url", help="Override EasyColab Gradio URL")
    parser.add_argument("--password", help="Override EasyColab admin password")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Local 0-CCU helpers
    p_snap = sub.add_parser("snap-frames", help="Snap duration/frames to MiniMax-H3 17k+5 grid")
    p_snap.add_argument("--seconds", type=float, default=5.0)
    p_snap.add_argument("--frames", type=int, default=None)
    p_snap.add_argument("--fps", type=int, default=24)

    p_voice = sub.add_parser("extract-voice-seed", help="Extract 32kHz stereo Self-Anchor Voice seed via ffmpeg")
    p_voice.add_argument("--input", required=True)
    p_voice.add_argument("--output", required=True)
    p_voice.add_argument("--start", type=float, default=0.0)
    p_voice.add_argument("--duration", type=float, default=5.0)

    # Colab lifecycle
    sub.add_parser("auth-status", help="Check active Google Colab OAuth & CU balance")
    sub.add_parser("auth-status-all", help="Concurrently query OAuth health, CU balance, active VMs & GPU models across ALL saved accounts")
    sub.add_parser("auth-url", help="Get Google Colab OAuth login URL")
    sub.add_parser("auth-list", help="List all saved Google Colab accounts")
    p_asw = sub.add_parser("auth-switch", help="Switch active Google Colab account in 1 command")
    p_asw.add_argument("email", help="Saved Google account email to activate")
    p_arm = sub.add_parser("auth-remove", help="Remove a saved Google Colab account from local vault")
    p_arm.add_argument("email", help="Saved Google account email to remove")
    p_ac = sub.add_parser("auth-code", help="Exchange OAuth authorization code")
    p_ac.add_argument("code")
    p_wake = sub.add_parser("wake", help="Wake Colab GPU VM and deploy EasyColab")
    p_wake.add_argument("--gpu", default="G4", choices=["G4", "H100", "A100", "L4", "T4"])
    p_wake.add_argument("--password", default="")
    p_wake.add_argument("--idle-minutes", type=int, default=10, help="Cloud idle auto-sleep timeout in minutes (default: 10)")
    p_wake.add_argument("--persistent", action="store_true", help="Keep retrying every --retry-interval seconds if 3 allocation attempts fail")
    p_wake.add_argument("--retry-interval", type=int, default=60, help="Seconds to wait before next 3-attempt round when --persistent is set (default: 60)")
    p_wmulti = sub.add_parser("wake-multi", help="Concurrently wake Colab GPUs across multiple saved accounts (first-wins race or all-parallel)")
    p_wmulti.add_argument("--accounts", default="", help="Comma-separated Google emails (empty = auto-pick idle saved accounts)")
    p_wmulti.add_argument("--gpu", default="G4", choices=["G4", "H100", "A100", "L4", "T4"])
    p_wmulti.add_argument("--mode", default="first", choices=["first", "all"], help="first = race and stop others on first win; all = provision one GPU per account")
    p_wmulti.add_argument("--password", default="")
    p_wmulti.add_argument("--idle-minutes", type=int, default=10, help="Cloud idle auto-sleep timeout in minutes (default: 10)")
    p_wmulti.add_argument("--persistent", action="store_true", help="Keep retrying every --retry-interval seconds when allocation fails")
    p_wmulti.add_argument("--retry-interval", type=int, default=60, help="Seconds between 3-attempt rounds when --persistent is set (default: 60)")
    p_conn = sub.add_parser("connect", help="Save and test connection to an already running EasyColab instance")
    p_conn.add_argument("--url", required=True)
    p_conn.add_argument("--password", default="")
    p_conn.add_argument("--comfy-url", default="")
    p_rec = sub.add_parser("recover", help="Losslessly probe or hot-restart app.py tunnel on an already running Colab VM")
    p_rec.add_argument("--restart", action="store_true", help="Hot-redeploy app.py and rebuild gradio.live tunnel without releasing the GPU VM")
    p_rec.add_argument("--password", default="")
    p_rec.add_argument("--idle-minutes", type=int, default=10)
    p_idle = sub.add_parser("config-idle", help="Configure cloud idle auto-sleep watchdog (minutes)")
    p_idle.add_argument("--minutes", type=int, default=10)
    sub.add_parser("keepalive", help="Send keepalive ping to active Colab runtime")
    sub.add_parser("disconnect", help="Ask running EasyColab instance to unassign itself")
    sub.add_parser("stop-all", help="Terminate all active Colab VMs and verify 0 CU/hr burn")

    # Native EasyColab API
    p_st = sub.add_parser("status", help="Query EasyColab GPU, storage, and model status")
    p_st.add_argument("--model", default=None)

    for cname in ("download-model", "load-model"):
        pm = sub.add_parser(cname, help=f"{cname.replace('-', ' ').title()} in EasyColab")
        pm.add_argument("--model", default="minimax_h3")
        pm.add_argument("--hf-token", default="")
        pm.add_argument("--wait", action="store_true")

    p_ul = sub.add_parser("unload-model", help="Unload native model from VRAM")
    p_ul.add_argument("--model", default="all")

    p_gv = sub.add_parser("generate-video", help="Generate T2VA/FLF2V via native MiniMax-H3 BF16 API")
    p_gv.add_argument("--prompt", required=True)
    p_gv.add_argument("--soundscape", default="")
    p_gv.add_argument("--music", default="")
    p_gv.add_argument("--first-frame", default="")
    p_gv.add_argument("--last-frame", default="")
    p_gv.add_argument("--width", type=int, default=832)
    p_gv.add_argument("--height", type=int, default=480)
    p_gv.add_argument("--frames", type=int, default=124)
    p_gv.add_argument("--steps", type=int, default=35)
    p_gv.add_argument("--fps", type=int, default=24)
    p_gv.add_argument("--seed", type=int, default=-1)
    p_gv.add_argument("--batch", type=int, default=1)
    p_gv.add_argument("--out", default="./outputs")
    p_gv.add_argument("--keep-alive", action="store_true", help="Keep Colab VM awake after this run (default: auto-sleep & release VM to save CU)")

    p_gi = sub.add_parser("generate-image", help="Generate or edit 4K image via native MiniMax-H3-Image API")
    p_gi.add_argument("--prompt", required=True)
    p_gi.add_argument("--first-frame", default="")
    p_gi.add_argument("--last-frame", default="")
    p_gi.add_argument("--width", type=int, default=2048)
    p_gi.add_argument("--height", type=int, default=2048)
    p_gi.add_argument("--steps", type=int, default=6)
    p_gi.add_argument("--seed", type=int, default=-1)
    p_gi.add_argument("--batch", type=int, default=1)
    p_gi.add_argument("--out", default="./outputs")
    p_gi.add_argument("--keep-alive", action="store_true", help="Keep Colab VM awake after this run (default: auto-sleep & release VM to save CU)")

    # ComfyUI management
    sub.add_parser("comfy-status", help="Query ComfyUI status and preset packs")
    p_cs = sub.add_parser("comfy-start", help="Start ComfyUI and Cloudflare tunnel")
    p_cs.add_argument("--unload-existing", action="store_true")
    p_cs.add_argument("--force-restart", action="store_true")
    p_cs.add_argument("--wait", action="store_true")
    sub.add_parser("comfy-stop", help="Sync and stop ComfyUI")
    sub.add_parser("comfy-free-vram", help="Unload ComfyUI models from VRAM")
    sub.add_parser("comfy-sync-drive", help="Sync ComfyUI nodes, models, and workflows to Google Drive")

    p_cip = sub.add_parser("comfy-install-pack", help="1-click install ComfyUI preset pack + custom nodes")
    p_cip.add_argument("--pack", required=True, choices=[
        "recavsr_upscaler", "seedvr2_upscaler", "minimax_h3_ref2va_tara", "minimax_h3_10eros_max", "minimax_h3_sulphur3",
        "minimax_h3_i2v_flf2v", "wan22_i2v_flf2v", "ltx23_flf2v", "hunyuan15_i2v_1080p",
    ])
    p_cip.add_argument("--no-auto-start", action="store_true")
    p_cip.add_argument("--hf-token", default="")
    p_cip.add_argument("--wait", action="store_true")

    p_cdw = sub.add_parser("comfy-download-weight", help="Download a custom ComfyUI weight")
    p_cdw.add_argument("--src", required=True)
    p_cdw.add_argument("--filename", default="")
    p_cdw.add_argument("--subfolder", default="diffusion_models")
    p_cdw.add_argument("--hf-token", default="")
    p_cdw.add_argument("--wait", action="store_true")

    p_cdo = sub.add_parser("comfy-download-output", help="Download a file from ComfyUI output/")
    p_cdo.add_argument("--rel-path", required=True)
    p_cdo.add_argument("--out", required=True)

    # Direct ComfyUI API runners
    p_eros = sub.add_parser("comfy-run-10eros", help="Run TenStrip/10Eros-Max (multiref or flf2v) via ComfyUI API")
    p_eros.add_argument("--mode", default="multiref", choices=["multiref", "flf2v"])
    p_eros.add_argument("--prompt", required=True)
    p_eros.add_argument("--ref-image", action="append", default=[], help="Reference image path (up to 9)")
    p_eros.add_argument("--ref-audio", action="append", default=[], help="Reference audio path (up to 3)")
    p_eros.add_argument("--first-frame", default="")
    p_eros.add_argument("--last-frame", default="")
    p_eros.add_argument("--width", type=int, default=1056)
    p_eros.add_argument("--height", type=int, default=608)
    p_eros.add_argument("--frames", type=int, default=124)
    p_eros.add_argument("--steps", type=int, default=8)
    p_eros.add_argument("--sampler", default="lcm")
    p_eros.add_argument("--scheduler", default="simple")
    p_eros.add_argument("--shift-video", type=float, default=12.0)
    p_eros.add_argument("--shift-audio", type=float, default=3.0)
    p_eros.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    p_eros.add_argument("--seed", type=int, default=-1)
    p_eros.add_argument("--prefix", default="video/10Eros_Max_CLI")
    p_eros.add_argument("--unet", default="10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors")
    p_eros.add_argument("--clip", default="qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors")
    p_eros.add_argument("--comfy-url", default="")
    p_eros.add_argument("--out", default="./outputs")
    p_eros.add_argument("--keep-alive", action="store_true", help="Keep Colab VM awake after this run (default: auto-sleep & release VM to save CU)")
    p_eros.add_argument("--dry-run", action="store_true")

    p_s3 = sub.add_parser("comfy-run-sulphur3", help="Run SulphurAI/Sulphur-3 + H3 Corrector via ComfyUI API")
    p_s3.add_argument("--mode", default="t2va", choices=["t2va", "i2va"])
    p_s3.add_argument("--prompt", default="")
    p_s3.add_argument("--prompt-file", default="")
    p_s3.add_argument("--first-frame", default="")
    p_s3.add_argument("--last-frame", default="")
    p_s3.add_argument("--width", type=int, default=1056)
    p_s3.add_argument("--height", type=int, default=608)
    p_s3.add_argument("--frames", type=int, default=124)
    p_s3.add_argument("--steps", type=int, default=20)
    p_s3.add_argument("--sampler", default="euler")
    p_s3.add_argument("--scheduler", default="simple")
    p_s3.add_argument("--shift-video", type=float, default=12.0)
    p_s3.add_argument("--shift-audio", type=float, default=3.0)
    p_s3.add_argument("--corrector-strength", type=float, default=1.0)
    p_s3.add_argument("--budget-fraction", type=float, default=0.5)
    p_s3.add_argument("--velocity-floor", type=float, default=0.001)
    p_s3.add_argument("--start-step", type=int, default=0)
    p_s3.add_argument("--end-step", type=int, default=-1)
    p_s3.add_argument("--seed", type=int, default=-1)
    p_s3.add_argument("--prefix", default="video/Sulphur3_CLI")
    p_s3.add_argument("--unet", default="s3.safetensors")
    p_s3.add_argument("--clip", default="qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors")
    p_s3.add_argument("--corrector", default="s3_corrector.safetensors")
    p_s3.add_argument("--comfy-url", default="")
    p_s3.add_argument("--out", default="./outputs")
    p_s3.add_argument("--keep-alive", action="store_true", help="Keep Colab VM awake after this run (default: auto-sleep & release VM to save CU)")
    p_s3.add_argument("--dry-run", action="store_true")

    p_tara = sub.add_parser("comfy-run-tara", help="Run Ref2VA + TARA Motion Transfer via ComfyUI API (<Video 1> + <Picture 1..9>)")
    p_tara.add_argument("--ref-video", required=True, help="Motion guide video path (<Video 1>)")
    p_tara.add_argument("--ref-image", action="append", default=[], help="Target character/scene image path (up to 9)")
    p_tara.add_argument("--ref-audio", action="append", default=[], help="Optional voice reference audio (up to 3)")
    p_tara.add_argument("--prompt", required=True)
    p_tara.add_argument("--width", type=int, default=1056)
    p_tara.add_argument("--height", type=int, default=608)
    p_tara.add_argument("--frames", type=int, default=124)
    p_tara.add_argument("--steps", type=int, default=8)
    p_tara.add_argument("--sampler", default="euler")
    p_tara.add_argument("--scheduler", default="simple")
    p_tara.add_argument("--shift-video", type=float, default=12.0)
    p_tara.add_argument("--shift-audio", type=float, default=3.0)
    p_tara.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    p_tara.add_argument("--lora-strength", type=float, default=1.0)
    p_tara.add_argument("--tara-strength", type=float, default=2.0)
    p_tara.add_argument("--tara-window", type=float, default=0.20)
    p_tara.add_argument("--tara-falloff", type=float, default=0.25)
    p_tara.add_argument("--tara-max-penalty", type=float, default=6.0)
    p_tara.add_argument("--disable-tara", action="store_true", help="Bypass BFSMiniMaxH3NativeTemporal node")
    p_tara.add_argument("--seed", type=int, default=-1)
    p_tara.add_argument("--prefix", default="video/Ref2VA_TARA_CLI")
    p_tara.add_argument("--unet", default="minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    p_tara.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    p_tara.add_argument("--comfy-url", default="")
    p_tara.add_argument("--out", default="./outputs")
    p_tara.add_argument("--keep-alive", action="store_true", help="Keep Colab VM awake after this run (default: auto-sleep & release VM to save CU)")
    p_tara.add_argument("--dry-run", action="store_true")

    p_up = sub.add_parser("comfy-upscale", help="Upscale a video using SeedVR2 7B/3B FP16 via ComfyUI API (PRO 6000 untiled)")
    p_up.add_argument("--video", required=True, help="Source video path (.mp4/.mov/.webm)")
    p_up.add_argument(
        "--model",
        default="seedvr2_ema_7b_fp16.safetensors",
        choices=[
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp16.safetensors",
            "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        ],
    )
    p_up.add_argument("--resolution", type=int, default=1080, help="Target short-edge resolution (e.g. 1080, 1440, 2160)")
    p_up.add_argument("--max-resolution", type=int, default=0, help="Optional long-edge cap (0 = disabled)")
    p_up.add_argument("--batch-size", type=int, default=17, help="Frames per batch (auto-snapped to 4n+1 grid)")
    p_up.add_argument("--temporal-overlap", type=int, default=2, help="Temporal overlap frames between batches (0-16)")
    p_up.add_argument("--color-correction", default="lab", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
    p_up.add_argument("--no-audio", action="store_true", help="Drop original audio instead of muxing into upscaled output")
    p_up.add_argument("--attention-mode", default="sdpa", choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"])
    p_up.add_argument("--decode-tile-size", type=int, default=2048, help="VAE decode tile size when resolution >= 1440 (default: 2048 for PRO 6000)")
    p_up.add_argument("--seed", type=int, default=-1)
    p_up.add_argument("--prefix", default="video/SeedVR2_CLI")
    p_up.add_argument("--comfy-url", default="")
    p_up.add_argument("--out", default="./outputs")
    p_up.add_argument("--keep-alive", action="store_true", help="Keep Colab VM awake after this run (default: auto-sleep & release VM to save CU)")
    p_up.add_argument("--dry-run", action="store_true")

    p_rc = sub.add_parser("comfy-upscale-recavsr", help="Fast 1-step streaming video upscale via ReCaVSR + FlashDecoder (~15x faster 4K)")
    p_rc.add_argument("--video", required=True, help="Source video path (.mp4/.mov/.webm)")
    p_rc.add_argument(
        "--transformer",
        default="transformer.safetensors",
        choices=["transformer.safetensors", "transformer_int8_convrot.safetensors"],
    )
    p_rc.add_argument("--scale", type=float, default=2.0, help="Upscale multiplier (recommended: 2.0 for 720p->2K, 4.0 for 480p->4K)")
    p_rc.add_argument("--color-fix", default="none", choices=["none", "adain", "wavelet"])
    p_rc.add_argument("--kernels", default="fast", choices=["fast", "exact"])
    p_rc.add_argument("--encoder", default="auto", choices=["auto", "hevc nvenc", "h264 nvenc", "h264"])
    p_rc.add_argument("--seed", type=int, default=-1)
    p_rc.add_argument("--prefix", default="video/ReCaVSR_CLI")
    p_rc.add_argument("--comfy-url", default="")
    p_rc.add_argument("--out", default="./outputs")
    p_rc.add_argument("--keep-alive", action="store_true", help="Keep Colab VM awake after this run (default: auto-sleep & release VM to save CU)")
    p_rc.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    scoped_acct = (getattr(args, "account", "") or os.environ.get("EASYCOLAB_ACCOUNT", "")).strip()
    if scoped_acct:
        os.environ["EASYCOLAB_ACCOUNT"] = scoped_acct
        acct_tag = hashlib.sha256(scoped_acct.lower().encode("utf-8")).hexdigest()[:12]
        globals()["SESSION_PATH"] = Path.home() / ".config" / "colab-minimax-h3" / f"session_{acct_tag}.json"
    if args.cmd not in ("snap-frames",):
        _emit_app_log(f"[Skill/CLI] >>> 开始执行命令: {args.cmd}")

    try:
        return _execute_cli_command(args)
    except Exception as exc:
        _emit_app_log(f"[Skill/CLI] 命令 {args.cmd} 失败: {exc}", level="error")
        raise


def _execute_cli_command(args: argparse.Namespace) -> int:
    if args.cmd == "snap-frames":
        cmd_snap_frames(args)
        return 0
    if args.cmd == "extract-voice-seed":
        cmd_extract_voice_seed(args)
        _emit_app_log(f"声纹音频切片完成 -> {args.output}")
        return 0
    if args.cmd in ("auth-status", "auth-status-all", "auth-url", "auth-list", "keepalive", "stop-all"):
        rc, out = _run_bridge([args.cmd.replace("-", "_")])
        if args.cmd == "stop-all":
            _emit_app_log(f"已执行 stop-all 释放全部 Colab 实例 (rc={rc})", level=("info" if rc == 0 else "error"))
        print(out)
        return rc
    if args.cmd in ("auth-switch", "auth-remove"):
        rc, out = _run_bridge([args.cmd.replace("-", "_"), args.email])
        print(out)
        return rc
    if args.cmd == "auth-code":
        rc, out = _run_bridge(["auth_code", args.code])
        print(out)
        return rc
    if args.cmd == "wake":
        cmd_wake(args)
        return 0
    if args.cmd == "wake-multi":
        cmd_wake_multi(args)
        return 0
    if args.cmd == "connect":
        cmd_connect(args)
        return 0
    if args.cmd == "recover":
        cmd_recover(args)
        return 0

    if args.cmd in ("comfy-run-10eros", "comfy-run-sulphur3", "comfy-run-tara", "comfy-upscale", "comfy-upscale-recavsr"):
        if args.dry_run:
            base_url, password = "", ""
        else:
            try:
                base_url, password = _resolve_connection(args)
            except Exception:
                base_url, password = "", ""
    else:
        base_url, password = _resolve_connection(args)

    if args.cmd == "config-idle":
        res = _http_json(
            f"{base_url}/minimax_api/config_idle",
            method="POST",
            payload={"idle_timeout_minutes": max(1, int(args.minutes))},
            password=password,
            base_url=base_url,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "disconnect":
        print(json.dumps(_auto_sleep_runtime(base_url, password, sync_comfy=False), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "status":
        qs = f"?model_id={urllib.parse.quote(args.model)}" if args.model else ""
        print(json.dumps(_http_json(f"{base_url}/minimax_api/status{qs}", password=password, base_url=base_url), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "download-model":
        _begin_external_task(
            command="download-model",
            model_id=args.model,
            model_name=_model_display_name(args.model),
            task_mode="",
            prompt=f"下载模型权重：{args.model}",
            params_summary=args.model,
        )
        try:
            res = _http_json(f"{base_url}/minimax_api/download_model", method="POST", payload={"model_id": args.model, "hf_token": args.hf_token}, password=password, base_url=base_url)
            if args.wait:
                res = _wait_for_native_flag(base_url, password, "is_downloading", args.model)
            _finish_external_task(output_files=[])
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "load-model":
        _begin_external_task(
            command="load-model",
            model_id=args.model,
            model_name=_model_display_name(args.model),
            task_mode="",
            prompt=f"加载模型至 GPU 显存：{args.model}",
            params_summary=args.model,
        )
        try:
            res = _http_json(f"{base_url}/minimax_api/load_model", method="POST", payload={"model_id": args.model, "hf_token": args.hf_token}, password=password, base_url=base_url)
            if args.wait:
                res = _wait_for_native_flag(base_url, password, "is_loading_vram", args.model)
            _finish_external_task(output_files=[])
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "unload-model":
        res = _http_json(f"{base_url}/minimax_api/unload_model", method="POST", payload={"model_id": args.model}, password=password, base_url=base_url)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "generate-video":
        snapped, _ = snap_frames_to_h3_grid(args.frames)
        ff = _encode_image_b64(args.first_frame)
        lf = _encode_image_b64(args.last_frame)
        task_mode = "fl2va" if (ff or lf) else "t2va"
        _emit_app_log(f"提交 MiniMax-H3 视频生成 ({task_mode}, {args.width}x{args.height}, {snapped}帧, {args.steps}步)...")
        payload = {
            "model_id": "minimax_h3",
            "task_mode": task_mode,
            "visual_prompt": args.prompt,
            "soundscape_prompt": args.soundscape,
            "music_prompt": args.music,
            "first_frame_base64": ff,
            "last_frame_base64": lf,
            "width": args.width,
            "height": args.height,
            "num_frames": snapped,
            "num_inference_steps": args.steps,
            "fps": args.fps,
            "seed": args.seed,
            "randomize_seed": args.seed < 0,
            "batch_count": args.batch,
        }
        sleep_report: Dict[str, Any] = {"triggered": False, "reason": "keep_alive"}
        done: Dict[str, Any] = {}
        _begin_external_task(
            command="generate-video",
            model_id="minimax_h3",
            model_name="MiniMax-H3 (33B)",
            task_mode=("fl2va" if (ff or lf) else "t2va"),
            prompt=args.prompt,
            soundscape=args.soundscape,
            music=args.music,
            params_summary=f"{args.width}x{args.height} · {snapped}帧 · {args.steps}步 · {args.fps}fps",
            params={
                "width": int(args.width),
                "height": int(args.height),
                "frames": int(snapped),
                "steps": int(args.steps),
                "fps": int(args.fps),
            },
            input_files=[x for x in (args.first_frame, args.last_frame) if x],
            out_dir=args.out,
        )
        try:
            job = _http_json(f"{base_url}/minimax_api/generate", method="POST", payload=payload, password=password, base_url=base_url, timeout=60)
            if job.get("status") == "failed":
                raise RuntimeError(job.get("message", "Generation rejected"))
            done = _poll_native_job(base_url, password, job["job_id"], Path(args.out))
            _finish_external_task(output_files=done.get("downloaded_files") or [])
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password, sync_comfy=False)
        done["auto_sleep"] = sleep_report
        print(json.dumps(done, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "generate-image":
        ff = _encode_image_b64(args.first_frame)
        lf = _encode_image_b64(args.last_frame)
        payload = {
            "model_id": "minimax_h3_image",
            "task_mode": "edit" if (ff or lf) else "t2i",
            "visual_prompt": args.prompt,
            "first_frame_base64": ff,
            "last_frame_base64": lf,
            "width": args.width,
            "height": args.height,
            "num_inference_steps": args.steps,
            "seed": args.seed,
            "randomize_seed": args.seed < 0,
            "batch_count": args.batch,
        }
        sleep_report = {"triggered": False, "reason": "keep_alive"}
        done = {}
        _begin_external_task(
            command="generate-image",
            model_id="minimax_h3_image",
            model_name="MiniMax-H3-Image (33B · 4K)",
            task_mode=("i2i_edit" if (ff or lf) else "t2i"),
            prompt=args.prompt,
            params_summary=f"{args.width}x{args.height} · {args.steps}步 · batch={args.batch}",
            params={
                "width": int(args.width),
                "height": int(args.height),
                "steps": int(args.steps),
            },
            input_files=[x for x in (args.first_frame, args.last_frame) if x],
            out_dir=args.out,
        )
        try:
            job = _http_json(f"{base_url}/minimax_api/generate", method="POST", payload=payload, password=password, base_url=base_url, timeout=60)
            if job.get("status") == "failed":
                raise RuntimeError(job.get("message", "Generation rejected"))
            done = _poll_native_job(base_url, password, job["job_id"], Path(args.out))
            _finish_external_task(output_files=done.get("downloaded_files") or [])
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password, sync_comfy=False)
        done["auto_sleep"] = sleep_report
        print(json.dumps(done, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-status":
        print(json.dumps(_http_json(f"{base_url}/minimax_api/comfyui/status", password=password, base_url=base_url), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-start":
        _begin_external_task(
            command="comfy-start",
            model_id="comfyui_studio",
            model_name=_model_display_name("comfyui_studio"),
            task_mode="comfyui",
            prompt="启动云端 ComfyUI 节点工作台并挂载持久化目录",
            params_summary="ComfyUI Canvas",
        )
        try:
            res = _http_json(f"{base_url}/minimax_api/comfyui/start", method="POST", payload={"unload_existing": args.unload_existing, "force_restart": args.force_restart}, password=password, base_url=base_url)
            if args.wait:
                cf = _wait_for_comfy_flag(base_url, password, "is_starting")
                if cf.get("public_url"):
                    sess = _load_session()
                    sess["comfy_url"] = str(cf["public_url"]).rstrip("/")
                    _save_session(sess)
                res = {"comfyui": cf}
            _finish_external_task(output_files=[])
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-stop":
        print(json.dumps(_http_json(f"{base_url}/minimax_api/comfyui/stop", method="POST", payload={}, password=password, base_url=base_url, timeout=60), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-free-vram":
        print(json.dumps(_http_json(f"{base_url}/minimax_api/comfyui/free_vram", method="POST", payload={}, password=password, base_url=base_url), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-sync-drive":
        print(json.dumps(_http_json(f"{base_url}/minimax_api/comfyui/sync_drive", method="POST", payload={}, password=password, base_url=base_url, timeout=180), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-install-pack":
        pack_model_id, pack_model_name = _pack_to_model_id(args.pack)
        _begin_external_task(
            command="comfy-install-pack",
            model_id=pack_model_id,
            model_name=pack_model_name,
            task_mode="",
            prompt=f"一键配齐 ComfyUI 工作流包：{args.pack}",
            params_summary=args.pack,
        )
        try:
            res = _http_json(
                f"{base_url}/minimax_api/comfyui/install_pack",
                method="POST",
                payload={"pack_id": args.pack, "auto_start": not args.no_auto_start, "hf_token": args.hf_token},
                password=password,
                base_url=base_url,
            )
            if args.wait:
                cf = _wait_for_comfy_flag(base_url, password, "installing_pack_id")
                if cf.get("is_starting"):
                    cf = _wait_for_comfy_flag(base_url, password, "is_starting")
                if cf.get("public_url"):
                    sess = _load_session()
                    sess["comfy_url"] = str(cf["public_url"]).rstrip("/")
                    _save_session(sess)
                res = {"comfyui": cf}
            _finish_external_task(output_files=[])
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-download-weight":
        _begin_external_task(
            command="comfy-download-weight",
            model_id="comfyui_studio",
            model_name=_model_display_name("comfyui_studio"),
            task_mode="comfyui",
            prompt=f"下载权重至 ComfyUI/models/{args.subfolder}: {args.src}",
            params_summary=f"{args.subfolder}",
        )
        try:
            res = _http_json(
                f"{base_url}/minimax_api/comfyui/download_weight",
                method="POST",
                payload={"url_or_repo": args.src, "filename": args.filename, "subfolder": args.subfolder, "hf_token": args.hf_token},
                password=password,
                base_url=base_url,
            )
            if args.wait:
                res = {"comfyui": _wait_for_comfy_flag(base_url, password, "is_downloading_weight")}
            _finish_external_task(output_files=[])
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-download-output":
        encoded = urllib.parse.quote(args.rel_path.lstrip("/"), safe="/")
        dst = _http_download(f"{base_url}/minimax_api/comfyui/output_media/{encoded}", Path(args.out), password=password, base_url=base_url)
        print(json.dumps({"ok": True, "downloaded": str(dst)}))
        return 0

    if args.cmd == "comfy-run-10eros":
        if args.dry_run:
            ref_imgs = [Path(p).name for p in args.ref_image]
            ref_auds = [Path(p).name for p in args.ref_audio]
            ff_name = Path(args.first_frame).name if args.first_frame else ""
            lf_name = Path(args.last_frame).name if args.last_frame else ""
            graph = _build_10eros_api_prompt(
                args.mode, args.prompt, ref_imgs, ref_auds, ff_name, lf_name,
                args.width, args.height, args.frames, args.steps, args.sampler, args.scheduler,
                args.shift_video, args.shift_audio, args.ref_image_size, args.seed, args.prefix, args.unet, args.clip,
            )
            print(json.dumps({"auto_sleep_after_run": not args.keep_alive, "prompt": graph}, ensure_ascii=False, indent=2))
            return 0
        sleep_report = {"triggered": False, "reason": "keep_alive"}
        files: List[str] = []
        cf_pw = password
        _begin_external_task(
            command="comfy-run-10eros",
            model_id="minimax_h3_10eros_max",
            model_name="10Eros-Max (H3 无限制)",
            task_mode=("fl2va" if args.mode == "flf2v" else "multiref"),
            prompt=args.prompt,
            params_summary=f"{args.width}x{args.height} · {snap_frames_to_h3_grid(args.frames)[0]}帧 · {args.steps}步 · {args.sampler}",
            params={
                "width": int(args.width),
                "height": int(args.height),
                "frames": int(snap_frames_to_h3_grid(args.frames)[0]),
                "steps": int(args.steps),
                "fps": 24,
            },
            input_files=[x for x in [*args.ref_image, *args.ref_audio, args.first_frame, args.last_frame] if x],
            out_dir=args.out,
            prefix=args.prefix,
        )
        try:
            comfy_url, cf_pw = _resolve_comfy_url(args)
            _emit_app_log(
                f"提交 10Eros-Max 推理 ({args.mode}, 参考图={len(args.ref_image)}张, 声纹={len(args.ref_audio)}段, {args.steps}步)..."
            )
            ref_imgs = [_comfy_upload_file(comfy_url, cf_pw, p) for p in args.ref_image]
            ref_auds = [_comfy_upload_file(comfy_url, cf_pw, p) for p in args.ref_audio]
            ff_name = _comfy_upload_file(comfy_url, cf_pw, args.first_frame) if args.first_frame else ""
            lf_name = _comfy_upload_file(comfy_url, cf_pw, args.last_frame) if args.last_frame else ""
            graph = _build_10eros_api_prompt(
                args.mode, args.prompt, ref_imgs, ref_auds, ff_name, lf_name,
                args.width, args.height, args.frames, args.steps, args.sampler, args.scheduler,
                args.shift_video, args.shift_audio, args.ref_image_size, args.seed, args.prefix, args.unet, args.clip,
            )
            files = _submit_and_wait_comfy(comfy_url, cf_pw, graph, Path(args.out))
            _finish_external_task(output_files=files)
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password or cf_pw, sync_comfy=bool(base_url))
        _emit_app_log(f"10Eros-Max 生成完成，输出文件: {', '.join(files)}")
        print(json.dumps({"ok": True, "files": files, "auto_sleep": sleep_report}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "comfy-run-sulphur3":
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
        if not prompt_text.strip():
            raise ValueError("Provide --prompt or --prompt-file for Sulphur-3.")
        if args.dry_run:
            ff_name = Path(args.first_frame).name if args.first_frame else ""
            lf_name = Path(args.last_frame).name if args.last_frame else ""
            graph = _build_sulphur3_api_prompt(
                args.mode, prompt_text, ff_name, lf_name,
                args.width, args.height, args.frames, args.steps, args.sampler, args.scheduler,
                args.shift_video, args.shift_audio, args.corrector_strength, args.budget_fraction,
                args.velocity_floor, args.start_step, args.end_step, args.seed, args.prefix,
                args.unet, args.clip, args.corrector,
            )
            print(json.dumps({"auto_sleep_after_run": not args.keep_alive, "prompt": graph}, ensure_ascii=False, indent=2))
            return 0
        sleep_report = {"triggered": False, "reason": "keep_alive"}
        files = []
        cf_pw = password
        _begin_external_task(
            command="comfy-run-sulphur3",
            model_id="minimax_h3_sulphur3",
            model_name="Sulphur-3 (H3 无限制)",
            task_mode=("fl2va" if args.mode == "i2va" else "t2va"),
            prompt=prompt_text,
            params_summary=f"{args.width}x{args.height} · {snap_frames_to_h3_grid(args.frames)[0]}帧 · {args.steps}步 · 修正器={args.corrector_strength}",
            params={
                "width": int(args.width),
                "height": int(args.height),
                "frames": int(snap_frames_to_h3_grid(args.frames)[0]),
                "steps": int(args.steps),
                "fps": 24,
                "corrector_enabled": bool(float(args.corrector_strength) > 0),
                "corrector_strength": float(args.corrector_strength),
            },
            input_files=[x for x in (args.first_frame, args.last_frame) if x],
            out_dir=args.out,
            prefix=args.prefix,
        )
        try:
            comfy_url, cf_pw = _resolve_comfy_url(args)
            _emit_app_log(
                f"提交 Sulphur-3 推理 ({args.mode}, {args.steps}步, 修正器强度={args.corrector_strength})..."
            )
            ff_name = _comfy_upload_file(comfy_url, cf_pw, args.first_frame) if args.first_frame else ""
            lf_name = _comfy_upload_file(comfy_url, cf_pw, args.last_frame) if args.last_frame else ""
            graph = _build_sulphur3_api_prompt(
                args.mode, prompt_text, ff_name, lf_name,
                args.width, args.height, args.frames, args.steps, args.sampler, args.scheduler,
                args.shift_video, args.shift_audio, args.corrector_strength, args.budget_fraction,
                args.velocity_floor, args.start_step, args.end_step, args.seed, args.prefix,
                args.unet, args.clip, args.corrector,
            )
            files = _submit_and_wait_comfy(comfy_url, cf_pw, graph, Path(args.out))
            _finish_external_task(output_files=files)
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password or cf_pw, sync_comfy=bool(base_url))
        _emit_app_log(f"Sulphur-3 生成完成，输出文件: {', '.join(files)}")
        print(json.dumps({"ok": True, "files": files, "auto_sleep": sleep_report}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "comfy-run-tara":
        if not Path(args.ref_video).expanduser().is_file():
            raise FileNotFoundError(f"Guide video not found: {args.ref_video}")
        if args.dry_run:
            ref_imgs = [Path(p).name for p in args.ref_image]
            ref_auds = [Path(p).name for p in args.ref_audio]
            vid_name = Path(args.ref_video).name
            graph = _build_ref2va_tara_api_prompt(
                args.prompt, vid_name, ref_imgs, ref_auds,
                args.width, args.height, args.frames, args.steps, args.sampler, args.scheduler,
                args.shift_video, args.shift_audio, args.ref_image_size, args.lora_strength,
                not args.disable_tara, args.tara_strength, args.tara_window, args.tara_falloff,
                args.tara_max_penalty, args.seed, args.prefix, args.unet, args.clip,
            )
            print(json.dumps({"auto_sleep_after_run": not args.keep_alive, "prompt": graph}, ensure_ascii=False, indent=2))
            return 0
        sleep_report = {"triggered": False, "reason": "keep_alive"}
        files = []
        cf_pw = password
        _begin_external_task(
            command="comfy-run-tara",
            model_id="minimax_h3_ref2va_tara",
            model_name="Ref2VA · TARA 动作复刻",
            task_mode="motion_transfer",
            prompt=args.prompt,
            params_summary=f"{args.width}x{args.height} · {snap_frames_to_h3_grid(args.frames)[0]}帧 · {args.steps}步 · TARA={'ON' if not args.disable_tara else 'OFF'}",
            params={
                "width": int(args.width),
                "height": int(args.height),
                "frames": int(snap_frames_to_h3_grid(args.frames)[0]),
                "steps": int(args.steps),
                "fps": 24,
                "tara_enabled": bool(not args.disable_tara),
                "tara_strength": float(args.tara_strength),
            },
            input_files=[x for x in [args.ref_video, *args.ref_image, *args.ref_audio] if x],
            out_dir=args.out,
            prefix=args.prefix,
        )
        try:
            comfy_url, cf_pw = _resolve_comfy_url(args)
            _emit_app_log(
                f"提交 Ref2VA TARA 动作复刻 ({args.steps}步, 参考图={len(args.ref_image)}张, TARA={'ON' if not args.disable_tara else 'OFF'})..."
            )
            vid_name = _comfy_upload_file(comfy_url, cf_pw, args.ref_video)
            ref_imgs = [_comfy_upload_file(comfy_url, cf_pw, p) for p in args.ref_image]
            ref_auds = [_comfy_upload_file(comfy_url, cf_pw, p) for p in args.ref_audio]
            graph = _build_ref2va_tara_api_prompt(
                args.prompt, vid_name, ref_imgs, ref_auds,
                args.width, args.height, args.frames, args.steps, args.sampler, args.scheduler,
                args.shift_video, args.shift_audio, args.ref_image_size, args.lora_strength,
                not args.disable_tara, args.tara_strength, args.tara_window, args.tara_falloff,
                args.tara_max_penalty, args.seed, args.prefix, args.unet, args.clip,
            )
            files = _submit_and_wait_comfy(comfy_url, cf_pw, graph, Path(args.out))
            _finish_external_task(output_files=files)
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password or cf_pw, sync_comfy=bool(base_url))
        _emit_app_log(f"Ref2VA TARA 动作复刻完成，输出文件: {', '.join(files)}")
        print(json.dumps({"ok": True, "files": files, "auto_sleep": sleep_report}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "comfy-upscale":
        if not Path(args.video).expanduser().is_file():
            raise FileNotFoundError(f"Source video not found: {args.video}")
        if args.dry_run:
            vid_name = Path(args.video).name
            graph = _build_seedvr2_upscale_api_prompt(
                vid_name, args.model, args.resolution, args.max_resolution,
                args.batch_size, args.temporal_overlap, args.color_correction,
                not args.no_audio, args.attention_mode, args.seed, args.prefix,
                decode_tile_size=args.decode_tile_size,
            )
            print(json.dumps({"auto_sleep_after_run": not args.keep_alive, "prompt": graph}, ensure_ascii=False, indent=2))
            return 0
        sleep_report = {"triggered": False, "reason": "keep_alive"}
        files = []
        cf_pw = password
        _begin_external_task(
            command="comfy-upscale",
            model_id="seedvr2_upscaler",
            model_name="SeedVR2 7B · 视频超分",
            task_mode="video_upscale",
            prompt=f"SeedVR2 视频超分修复：{Path(args.video).name} -> 短边 {args.resolution}p ({args.model})",
            params_summary=f"{args.resolution}p · batch={_snap_seedvr2_batch_size(args.batch_size)} · color={args.color_correction}",
            params={
                "seedvr2_model": args.model,
                "seedvr2_resolution": int(args.resolution),
                "seedvr2_batch_size": int(_snap_seedvr2_batch_size(args.batch_size)),
                "seedvr2_temporal_overlap": int(args.temporal_overlap),
                "seedvr2_color_correction": args.color_correction,
                "seedvr2_keep_audio": bool(not args.no_audio),
            },
            input_files=[args.video],
            out_dir=args.out,
            prefix=args.prefix,
        )
        try:
            comfy_url, cf_pw = _resolve_comfy_url(args)
            _emit_app_log(
                f"提交 SeedVR2 视频超分 ({args.model}, 目标短边={args.resolution}p, batch={_snap_seedvr2_batch_size(args.batch_size)}, 保留音轨={'ON' if not args.no_audio else 'OFF'})..."
            )
            vid_name = _comfy_upload_file(comfy_url, cf_pw, args.video)
            graph = _build_seedvr2_upscale_api_prompt(
                vid_name, args.model, args.resolution, args.max_resolution,
                args.batch_size, args.temporal_overlap, args.color_correction,
                not args.no_audio, args.attention_mode, args.seed, args.prefix,
                decode_tile_size=args.decode_tile_size,
            )
            files = _submit_and_wait_comfy(comfy_url, cf_pw, graph, Path(args.out))
            _finish_external_task(output_files=files)
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password or cf_pw, sync_comfy=bool(base_url))
        _emit_app_log(f"SeedVR2 视频超分完成，输出文件: {', '.join(files)}")
        print(json.dumps({"ok": True, "files": files, "auto_sleep": sleep_report}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "comfy-upscale-recavsr":
        if not Path(args.video).expanduser().is_file():
            raise FileNotFoundError(f"Source video not found: {args.video}")
        if args.dry_run:
            vid_name = Path(args.video).name
            graph = _build_recavsr_upscale_api_prompt(
                vid_name, args.transformer, args.scale, args.color_fix,
                args.kernels, args.encoder, args.seed, args.prefix,
            )
            print(json.dumps({"auto_sleep_after_run": not args.keep_alive, "prompt": graph}, ensure_ascii=False, indent=2))
            return 0
        sleep_report = {"triggered": False, "reason": "keep_alive"}
        files = []
        cf_pw = password
        _begin_external_task(
            command="comfy-upscale-recavsr",
            model_id="recavsr_upscaler",
            model_name="ReCaVSR · 极速流式超分",
            task_mode="video_upscale",
            prompt=f"ReCaVSR 流式超分放大：{Path(args.video).name} -> {args.scale}x ({args.transformer})",
            params_summary=f"{args.scale}x · {args.transformer} · {args.kernels}",
            params={
                "recavsr_transformer": args.transformer,
                "recavsr_scale": float(args.scale),
                "recavsr_color_fix": args.color_fix,
                "recavsr_kernels": args.kernels,
                "recavsr_encoder": args.encoder,
            },
            input_files=[args.video],
            out_dir=args.out,
            prefix=args.prefix,
        )
        try:
            comfy_url, cf_pw = _resolve_comfy_url(args)
            _emit_app_log(
                f"提交 ReCaVSR 流式超分 ({args.transformer}, scale={args.scale}x, kernels={args.kernels}, FlashDecoder)..."
            )
            vid_name = _comfy_upload_file(comfy_url, cf_pw, args.video)
            graph = _build_recavsr_upscale_api_prompt(
                vid_name, args.transformer, args.scale, args.color_fix,
                args.kernels, args.encoder, args.seed, args.prefix,
            )
            files = _submit_and_wait_comfy(comfy_url, cf_pw, graph, Path(args.out))
            _finish_external_task(output_files=files)
        except Exception as task_err:
            _finish_external_task(error=str(task_err))
            raise
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password or cf_pw, sync_comfy=bool(base_url))
        _emit_app_log(f"ReCaVSR 流式超分完成，输出文件: {', '.join(files)}")
        print(json.dumps({"ok": True, "files": files, "auto_sleep": sleep_report}, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
