"""Unified Cloud Runtime & Multimodal Model Client for EasyColab G4.

Wraps `colab_h3_cli.py` and EasyColab `/minimax_api/*` + `/v1/chat/completions` + ComfyUI `/prompt`
with dual-layer authentication (`Basic easycolab:<pwd>` + `x-api-key`), automatic G4 multi-account
wake, VRAM choreography, and structured JSON extraction.
"""

from __future__ import annotations

import base64
import io
import json
import re
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

CLI_DIR = Path("/Users/vanch/easycolab/skills/colab-minimax-h3/scripts")
CLI_SCRIPT = CLI_DIR / "colab_h3_cli.py"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))
import colab_h3_cli as cli  # type: ignore


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from LLM output (stripping <think> blocks and markdown fences)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {raw_text[:400]!r}")
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        # Remove trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", snippet)
        return json.loads(repaired)


def image_to_data_uri(image_path: Path, max_edge: int = 1280) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class EasyColabCloudClient:
    """Manages G4 VM lifecycle, Qwen3.8-27B-Uncensored-VL calls, and ComfyUI pipelines."""

    def __init__(self, idle_minutes: int = 30) -> None:
        self.idle_minutes = idle_minutes
        self._api_key = "ec_drama_studio_" + secrets.token_hex(12)
        self._api_key_configured = False

    def _run_cli(self, args: Sequence[str], timeout: int = 2400) -> str:
        cmd = [sys.executable, str(CLI_SCRIPT), *args]
        proc = subprocess.run(cmd, check=True, text=True, capture_output=False, timeout=timeout)
        return "" if proc.stdout is None else proc.stdout

    def session_connection(self) -> tuple[str, str]:
        sess = cli._load_session()
        url = str(sess.get("url") or "").strip().rstrip("/")
        pwd = str(sess.get("password") or "").strip()
        return url, pwd

    def ensure_awake(self) -> tuple[str, str]:
        """Verify live EasyColab G4 connection; auto-recover or wake-multi if offline."""
        url, pwd = self.session_connection()
        if url and not cli._load_session().get("sleeping", False):
            try:
                st = cli._http_json(f"{url}/minimax_api/status", password=pwd, base_url=url, timeout=12)
                if st.get("ok") or "gpu_name" in st:
                    return url, pwd
            except Exception:
                pass

        # Check if any account already has an active VM we can losslessly recover
        try:
            probe = subprocess.check_output(
                [sys.executable, str(CLI_SCRIPT), "auth-status-all"], text=True, timeout=30
            )
            info = json.loads(probe.strip().splitlines()[-1])
            for acct in info.get("accounts") or []:
                if int(acct.get("active_assignments") or 0) > 0:
                    em = acct["email"]
                    print(f"[DramaStudio] 检测到在线实例 ({em})，正在恢复隧道连接...", flush=True)
                    subprocess.run(
                        [sys.executable, str(CLI_SCRIPT), "--account", em, "recover", "--restart", "--idle-minutes", str(self.idle_minutes)],
                        check=True,
                    )
                    return self.session_connection()
        except Exception:
            pass

        print("[DramaStudio] 当前无在线 G4 实例，启动多账号并发唤醒 (wake-multi --gpu G4)...", flush=True)
        subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "wake-multi", "--gpu", "G4", "--mode", "first",
                "--idle-minutes", str(self.idle_minutes), "--persistent",
            ],
            check=True,
        )
        url, pwd = self.session_connection()
        if not url:
            raise RuntimeError("Failed to obtain EasyColab URL after wake-multi.")
        return url, pwd

    def ensure_comfy_packs(self, pack_ids: Sequence[str]) -> None:
        url, pwd = self.ensure_awake()
        cst = cli._http_json(f"{url}/minimax_api/comfy/status", password=pwd, base_url=url, timeout=25)
        comfy_info = cst.get("comfyui") or {}
        if not comfy_info.get("is_running"):
            print("[DramaStudio] 正在启动云端 ComfyUI 服务...", flush=True)
            subprocess.run([sys.executable, str(CLI_SCRIPT), "comfy-start", "--wait"], check=True)
            cst = cli._http_json(f"{url}/minimax_api/comfy/status", password=pwd, base_url=url, timeout=25)
            comfy_info = cst.get("comfyui") or {}

        ready_map = {p["id"]: bool(p.get("ready")) for p in (comfy_info.get("preset_packs") or [])}
        for pid in pack_ids:
            if not ready_map.get(pid, False):
                print(f"[DramaStudio] 正在安装/校验预设模型包: {pid} ...", flush=True)
                subprocess.run(
                    [sys.executable, str(CLI_SCRIPT), "comfy-install-pack", "--pack", pid, "--wait"],
                    check=True,
                )

    def free_comfy_vram(self) -> None:
        subprocess.run([sys.executable, str(CLI_SCRIPT), "comfy-free-vram"], check=True)

    def ensure_llm_loaded(self, model_id: str = "qwen38_27b_uncensored_gguf") -> None:
        url, pwd = self.ensure_awake()
        st = cli._http_json(f"{url}/minimax_api/status", password=pwd, base_url=url, timeout=20)
        loaded_ids = st.get("loaded_model_ids") or []
        if model_id not in loaded_ids:
            print(f"[DramaStudio] 正在准备编剧/审片多模态大模型 {model_id} ...", flush=True)
            subprocess.run(
                [sys.executable, str(CLI_SCRIPT), "download-model", "--model", model_id, "--wait"],
                check=True,
            )
            subprocess.run(
                [sys.executable, str(CLI_SCRIPT), "load-model", "--model", model_id, "--wait"],
                check=True,
            )
        if not self._api_key_configured:
            cli._http_json(
                f"{url}/minimax_api/config_api_key",
                method="POST",
                payload={"api_key": self._api_key, "initialize_only": False},
                password=pwd,
                base_url=url,
                timeout=15,
            )
            self._api_key_configured = True

    def unload_llm(self) -> None:
        url, pwd = self.session_connection()
        if not url:
            return
        try:
            subprocess.run([sys.executable, str(CLI_SCRIPT), "unload-model", "--model", "all"], check=True)
        except Exception:
            pass

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[Sequence[Path]] = None,
        temperature: float = 0.35,
        max_tokens: int = 8192,
        enable_thinking: bool = False,
    ) -> str:
        """Call Qwen3.8-27B-Uncensored-GGUF (with F16 mmproj vision support) via /v1/chat/completions."""
        self.ensure_llm_loaded()
        url, pwd = self.session_connection()
        if images:
            content: Any = [{"type": "image_url", "image_url": {"url": image_to_data_uri(Path(p))}} for p in images]
            content.append({"type": "text", "text": user_prompt})
        else:
            content = user_prompt

        payload = {
            "model": "Qwen3.8-27B-Uncensored-Q4_K_M",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "top_p": 0.85,
            "presence_penalty": 1.1,
            "max_tokens": max_tokens,
            "stream": False,
            "enable_thinking": enable_thinking,
        }
        headers = cli._auth_headers(pwd, url)
        headers["Content-Type"] = "application/json"
        headers["x-api-key"] = self._api_key

        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty /v1/chat/completions response: {data}")
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "").strip()

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[Sequence[Path]] = None,
        temperature: float = 0.35,
        max_tokens: int = 8192,
        retries: int = 2,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        cur_prompt = user_prompt
        for attempt in range(retries + 1):
            raw = self.chat_completion(
                system_prompt=system_prompt,
                user_prompt=cur_prompt,
                images=images,
                temperature=temperature if attempt == 0 else 0.2,
                max_tokens=max_tokens,
                enable_thinking=False,
            )
            try:
                return extract_json_object(raw)
            except Exception as exc:
                last_err = exc
                cur_prompt = (
                    f"{user_prompt}\n\n[FORMAT ERROR ON PREVIOUS ATTEMPT]: Your previous reply could not be parsed as strict JSON ({exc}). "
                    "Output ONLY ONE valid RFC8259 JSON object starting with '{' and ending with '}', with double-quoted keys and properly escaped strings."
                )
        raise RuntimeError(f"Failed to obtain valid JSON from Qwen3.8-27B after {retries + 1} attempts: {last_err}")

    def run_moss_image(
        self,
        prompt: str,
        out_file: Path,
        width: int = 1280,
        height: int = 704,
        steps: int = 28,
        seed: int = 42,
        ref_image: Optional[Path] = None,
    ) -> Path:
        out_dir = out_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"_tmp_moss_{out_file.stem}_{int(time.time()*1000)%100000}"
        cmd = [
            "comfy-run-moss",
            "--mode", "i2i_edit" if ref_image else "t2i",
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--steps", str(steps),
            "--disable-nsfw-lora",
            "--no-match-input-size",
            "--seed", str(seed),
            "--prefix", prefix,
            "--out", str(out_dir),
            "--keep-alive",
        ]
        if ref_image:
            cmd.extend(["--ref-image", str(ref_image)])
        self._run_cli(cmd)
        matches = sorted(out_dir.glob(f"{prefix}_*.png"))
        if not matches:
            raise RuntimeError(f"MOSS-Image produced no PNG for {prefix}")
        matches[-1].replace(out_file)
        for extra in matches[:-1]:
            extra.unlink(missing_ok=True)
        return out_file

    def run_qwen_edit(
        self,
        prompt: str,
        ref_images: Sequence[Path],
        out_file: Path,
        width: int = 1280,
        height: int = 704,
        steps: int = 4,
        angle_lora_strength: float = 0.0,
        seed: int = 42,
    ) -> Path:
        out_dir = out_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"_tmp_qedit_{out_file.stem}_{int(time.time()*1000)%100000}"
        cmd = [
            "comfy-run-qwen-edit",
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--steps", str(steps),
            "--cfg", "1.0",
            "--lightning-lora-strength", "1.0",
            "--angle-lora-strength", str(angle_lora_strength),
            "--seed", str(seed),
            "--prefix", prefix,
            "--out", str(out_dir),
            "--keep-alive",
        ]
        for r in ref_images[:3]:
            cmd.extend(["--ref-image", str(r)])
        self._run_cli(cmd)
        matches = sorted(out_dir.glob(f"{prefix}_*.png"))
        if not matches:
            raise RuntimeError(f"Qwen-Image-Edit-2511 produced no PNG for {prefix}")
        matches[-1].replace(out_file)
        for extra in matches[:-1]:
            extra.unlink(missing_ok=True)
        return out_file

    def run_qwen3_tts_design(self, instruct_zh: str, text_zh: str, out_file: Path, seed: int = 42) -> Path:
        out_dir = out_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"_tmp_qtts_{out_file.stem}_{int(time.time()*1000)%100000}"
        self._run_cli([
            "comfy-run-qwen3-tts",
            "--mode", "design",
            "--language", "Chinese",
            "--instruct", instruct_zh,
            "--text", text_zh,
            "--seed", str(seed),
            "--prefix", prefix,
            "--out", str(out_dir),
            "--keep-alive",
        ])
        matches = sorted(list(out_dir.glob(f"{prefix}*.*")))
        if not matches:
            raise RuntimeError(f"Qwen3-TTS VoiceDesign produced no audio for {prefix}")
        matches[-1].replace(out_file)
        for extra in matches[:-1]:
            extra.unlink(missing_ok=True)
        return out_file

    def run_drama3_line(self, ref_audio: Path, drama3_text: str, out_file: Path, seed: int = 42) -> Path:
        out_dir = out_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"_tmp_d3_{out_file.stem}_{int(time.time()*1000)%100000}"
        self._run_cli([
            "comfy-run-drama3",
            "--ref-audio", str(ref_audio),
            "--speakers", "one take per line",
            "--voice-guidance", "1.5",
            "--temperature", "0.9",
            "--top-p", "0.9",
            "--top-k", "0",
            "--seed", str(seed),
            "--text", drama3_text,
            "--prefix", prefix,
            "--out", str(out_dir),
            "--keep-alive",
        ])
        matches = sorted(list(out_dir.glob(f"{prefix}*.*")))
        if not matches:
            raise RuntimeError(f"Drama 3 produced no audio for {prefix}")
        matches[-1].replace(out_file)
        for extra in matches[:-1]:
            extra.unlink(missing_ok=True)
        return out_file

    def run_10eros_video(
        self,
        prompt: str,
        ref_images: Sequence[Path],
        ref_audio: Optional[Path],
        out_file: Path,
        width: int = 1280,
        height: int = 704,
        frames: int = 73,
        ref_image_size: str = "max",
        seed: int = 42,
    ) -> Path:
        out_dir = out_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"_tmp_h3_{out_file.stem}_{int(time.time()*1000)%100000}"
        cmd = [
            "comfy-run-10eros",
            "--mode", "multiref",
            "--prompt", prompt,
            "--width", str(width),
            "--height", str(height),
            "--frames", str(frames),
            "--steps", "8",
            "--ref-image-size", ref_image_size,
            "--seed", str(seed),
            "--prefix", prefix,
            "--out", str(out_dir),
            "--keep-alive",
        ]
        for img in ref_images[:6]:
            cmd.extend(["--ref-image", str(img)])
        if ref_audio:
            cmd.extend(["--ref-audio", str(ref_audio)])
        self._run_cli(cmd, timeout=2400)
        matches = sorted(out_dir.glob(f"{prefix}_*.mp4"))
        if not matches:
            raise RuntimeError(f"10Eros-Max produced no MP4 for {prefix}")
        matches[-1].replace(out_file)
        for extra in matches[:-1]:
            extra.unlink(missing_ok=True)
        return out_file

    def run_qwen3_asr_blind(self, audio_path: Path, language: str = "Chinese") -> str:
        """Strictly blind Qwen3-ASR 1.7B transcription (`context=''`) on ComfyUI."""
        s = cli._load_session()
        cu, pw = cli._resolve_comfy_url(type("A", (), {"comfy_url": "", "password": s.get("password", "")})())
        uploaded = cli._comfy_upload_file(cu, pw, str(audio_path), file_type="input")
        graph = {
            "1": {"class_type": "LoadAudio", "inputs": {"audio": uploaded}},
            "2": {"class_type": "Qwen3ASRLoader", "inputs": {"model_name": "qwen3_asr_1.7b_bf16.safetensors"}},
            "3": {
                "class_type": "Qwen3ASRTranscribe",
                "inputs": {
                    "model": ["2", 0],
                    "audio": ["1", 0],
                    "language": language,
                    "context": "",
                    "max_new_tokens": 2048,
                },
            },
            "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
        }
        res = cli._http_json(f"{cu}/prompt", method="POST", payload={"prompt": graph}, password=pw, base_url=cu, timeout=30)
        pid = res["prompt_id"]
        deadline = time.time() + 300
        while time.time() < deadline:
            hist = cli._http_json(f"{cu}/history/{pid}", password=pw, base_url=cu, timeout=20)
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Qwen3-ASR error: {json.dumps(status, ensure_ascii=False)}")
                outs = entry.get("outputs", {})
                if "4" in outs:
                    txt_val = outs["4"].get("text") or outs["4"].get("string") or ""
                    if isinstance(txt_val, list):
                        return "\n".join(str(v) for v in txt_val)
                    return str(txt_val)
                return ""
            time.sleep(1.2)
        raise TimeoutError("Timed out waiting for Qwen3ASRTranscribe")
