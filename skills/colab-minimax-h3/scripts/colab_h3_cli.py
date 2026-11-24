#!/usr/bin/env python3
"""Unified CLI for EasyColab + Google Colab MiniMax-H3 workflows (stdlib only)."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
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
    if not SESSION_PATH.is_file():
        return {}
    try:
        return json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_session(data: Dict[str, Any]) -> None:
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
    req = urllib.request.Request(url, headers=_auth_headers(password, base_url))
    tmp = dst_path.with_suffix(dst_path.suffix + ".part")
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


def _submit_and_wait_comfy(comfy_url: str, password: str, graph: Dict[str, Any], out_dir: Path, timeout_s: int = 1800) -> List[str]:
    client_id = uuid.uuid4().hex
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
    msg_sub = f"Submitted ComfyUI prompt_id={prompt_id}, waiting for completion..."
    print(msg_sub, flush=True)
    _emit_app_log(f"ComfyUI 任务已入队 (prompt_id={prompt_id})，等待云端渲染完成...")

    deadline = time.time() + timeout_s
    history_item: Optional[Dict[str, Any]] = None
    tick_count = 0
    while time.time() < deadline:
        hist = _http_json(f"{comfy_url}/history/{prompt_id}", password=password, base_url=comfy_url, timeout=30)
        if prompt_id in hist:
            history_item = hist[prompt_id]
            break
        tick_count += 1
        if tick_count % 5 == 0:
            _emit_app_log(f"ComfyUI 推理运行中 (prompt_id={prompt_id}, 已等待 {tick_count * 3}s)...")
        time.sleep(3.0)

    if not history_item:
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt_id={prompt_id}")

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
        prog = st.get("progress")
        line = f"[{flag_key}] progress={prog} | {msg}"
        print(line, flush=True)
        _emit_app_log(line)
        if not st.get(flag_key):
            return st
        time.sleep(4.0)
    raise TimeoutError(f"Timed out waiting for {flag_key}")


def _wait_for_comfy_flag(base_url: str, password: str, flag_key: str, timeout_s: int = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = _http_json(f"{base_url}/minimax_api/comfyui/status", password=password, base_url=base_url)
        cf = st.get("comfyui") or {}
        line = f"[comfyui:{flag_key}] progress={cf.get('progress')} | {cf.get('status_message')}"
        print(line, flush=True)
        _emit_app_log(line)
        if not cf.get(flag_key):
            return cf
        time.sleep(4.0)
    raise TimeoutError(f"Timed out waiting for comfyui.{flag_key}")


def _poll_native_job(base_url: str, password: str, job_id: str, out_dir: Path, timeout_s: int = 3600) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = _http_json(f"{base_url}/minimax_api/job/{job_id}", password=password, base_url=base_url)
        status = job.get("status")
        line = f"[job {job_id}] status={status} progress={job.get('progress')} | {job.get('message')}"
        print(line, flush=True)
        _emit_app_log(line, level=("error" if status == "failed" else "info"))
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
    _emit_app_log(f"正在通过 Skill CLI 唤醒 {args.gpu} GPU 实例并部署 app.py...")
    rc, out = _run_bridge(["wake", "--gpu", args.gpu, "--app-py", str(root / "app.py")], env_extra={"EASYCOLAB_ADMIN_PASSWORD": pwd}, stream=True)
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
    sub.add_parser("auth-status", help="Check Google Colab OAuth & CU balance")
    sub.add_parser("auth-url", help="Get Google Colab OAuth login URL")
    p_ac = sub.add_parser("auth-code", help="Exchange OAuth authorization code")
    p_ac.add_argument("code")
    p_wake = sub.add_parser("wake", help="Wake Colab GPU VM and deploy EasyColab")
    p_wake.add_argument("--gpu", default="G4", choices=["G4", "H100", "A100", "L4", "T4"])
    p_wake.add_argument("--password", default="")
    p_wake.add_argument("--idle-minutes", type=int, default=10, help="Cloud idle auto-sleep timeout in minutes (default: 10)")
    p_conn = sub.add_parser("connect", help="Save and test connection to an already running EasyColab instance")
    p_conn.add_argument("--url", required=True)
    p_conn.add_argument("--password", default="")
    p_conn.add_argument("--comfy-url", default="")
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
        "minimax_h3_10eros_max", "minimax_h3_sulphur3", "minimax_h3_i2v_flf2v",
        "wan22_i2v_flf2v", "ltx23_flf2v", "hunyuan15_i2v_1080p",
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

    args = parser.parse_args(argv)
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
    if args.cmd in ("auth-status", "auth-url", "keepalive", "stop-all"):
        rc, out = _run_bridge([args.cmd.replace("-", "_")])
        if args.cmd == "stop-all":
            _emit_app_log(f"已执行 stop-all 释放全部 Colab 实例 (rc={rc})", level=("info" if rc == 0 else "error"))
        print(out)
        return rc
    if args.cmd == "auth-code":
        rc, out = _run_bridge(["auth_code", args.code])
        print(out)
        return rc
    if args.cmd == "wake":
        cmd_wake(args)
        return 0
    if args.cmd == "connect":
        cmd_connect(args)
        return 0

    if args.cmd in ("comfy-run-10eros", "comfy-run-sulphur3"):
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
        res = _http_json(f"{base_url}/minimax_api/download_model", method="POST", payload={"model_id": args.model, "hf_token": args.hf_token}, password=password, base_url=base_url)
        if args.wait:
            res = _wait_for_native_flag(base_url, password, "is_downloading", args.model)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "load-model":
        res = _http_json(f"{base_url}/minimax_api/load_model", method="POST", payload={"model_id": args.model, "hf_token": args.hf_token}, password=password, base_url=base_url)
        if args.wait:
            res = _wait_for_native_flag(base_url, password, "is_loading_vram", args.model)
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
        try:
            job = _http_json(f"{base_url}/minimax_api/generate", method="POST", payload=payload, password=password, base_url=base_url, timeout=60)
            if job.get("status") == "failed":
                raise RuntimeError(job.get("message", "Generation rejected"))
            done = _poll_native_job(base_url, password, job["job_id"], Path(args.out))
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
        try:
            job = _http_json(f"{base_url}/minimax_api/generate", method="POST", payload=payload, password=password, base_url=base_url, timeout=60)
            if job.get("status") == "failed":
                raise RuntimeError(job.get("message", "Generation rejected"))
            done = _poll_native_job(base_url, password, job["job_id"], Path(args.out))
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
        res = _http_json(f"{base_url}/minimax_api/comfyui/start", method="POST", payload={"unload_existing": args.unload_existing, "force_restart": args.force_restart}, password=password, base_url=base_url)
        if args.wait:
            cf = _wait_for_comfy_flag(base_url, password, "is_starting")
            if cf.get("public_url"):
                sess = _load_session()
                sess["comfy_url"] = str(cf["public_url"]).rstrip("/")
                _save_session(sess)
            res = {"comfyui": cf}
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
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "comfy-download-weight":
        res = _http_json(
            f"{base_url}/minimax_api/comfyui/download_weight",
            method="POST",
            payload={"url_or_repo": args.src, "filename": args.filename, "subfolder": args.subfolder, "hf_token": args.hf_token},
            password=password,
            base_url=base_url,
        )
        if args.wait:
            res = {"comfyui": _wait_for_comfy_flag(base_url, password, "is_downloading_weight")}
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
        finally:
            if not args.keep_alive:
                sleep_report = _auto_sleep_runtime(base_url, password or cf_pw, sync_comfy=bool(base_url))
        _emit_app_log(f"Sulphur-3 生成完成，输出文件: {', '.join(files)}")
        print(json.dumps({"ok": True, "files": files, "auto_sleep": sleep_report}, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
