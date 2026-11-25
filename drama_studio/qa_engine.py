"""Automated Multimodal Quality Assurance Engine (Gates 1, 2, 3, 4, 5).

Includes:
- Blind `Qwen3-ASR 1.7B` + F0 pitch prosody & speaking-rate verification (Gates 2, 3, 5A).
- FFmpeg single-pass video signal detector (`BLACK` / `FREEZE` / `CUT`) + auto-trim calculator.
- Timestamped dark-mode `ContactSheet` grid composer for multi-frame video review.
- `Qwen3.8-27B-Uncensored-VL` vision inspector for assets (Gate 1), keyframes (Gate 4),
  and shot footage contact sheets (Gate 5B).
"""

from __future__ import annotations

import difflib
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cloud_client import EasyColabCloudClient
from .rules import normalize_zh_chars

BLACK_RE = re.compile(r"black_start:(-?[\d.]+)\s+black_end:(-?[\d.]+)")
FREEZE_START_RE = re.compile(r"lavfi\.freezedetect\.freeze_start:\s*(-?[\d.]+)")
FREEZE_END_RE = re.compile(r"lavfi\.freezedetect\.freeze_end:\s*(-?[\d.]+)")
SCENE_CUT_RE = re.compile(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:(-?[\d.]+)")


@dataclass
class VideoMachineSignals:
    duration_s: float
    black_spans: List[Tuple[float, float]] = field(default_factory=list)
    freeze_spans: List[Tuple[float, float]] = field(default_factory=list)
    internal_cuts: List[float] = field(default_factory=list)
    recommended_in_s: float = 0.0
    recommended_out_s: float = 0.0

    def tags_near(self, t: float, tol: float = 0.12) -> List[str]:
        tags: List[str] = []
        if any(abs(t - c) <= tol for c in self.internal_cuts):
            tags.append("CUT")
        if any(a - 0.02 <= t <= b + 0.02 for a, b in self.black_spans):
            tags.append("BLACK")
        if any(a - 0.02 <= t <= b + 0.02 for a, b in self.freeze_spans):
            tags.append("FREEZE")
        return tags


def compute_audio_prosody_stats(audio_path: Path) -> Dict[str, float]:
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(audio_path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    )
    x = np.frombuffer(raw, dtype=np.float32)
    sr = 16000.0
    dur = len(x) / sr if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0

    f0s: List[float] = []
    frame_len = int(0.04 * sr)
    hop = int(0.02 * sr)
    min_lag = int(sr / 380.0)
    max_lag = int(sr / 70.0)
    for st in range(0, max(0, len(x) - frame_len), hop):
        fr = x[st : st + frame_len]
        fr_rms = float(np.sqrt(np.mean(fr**2)))
        if fr_rms < max(0.012, rms * 0.35):
            continue
        fr = fr - np.mean(fr)
        ac = np.correlate(fr, fr, mode="full")[frame_len - 1 :]
        if ac[0] <= 1e-8:
            continue
        seg = ac[min_lag:max_lag]
        if len(seg) == 0:
            continue
        best = int(np.argmax(seg))
        if seg[best] / ac[0] > 0.35:
            f0s.append(sr / float(min_lag + best))

    return {
        "duration_s": round(dur, 2),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "median_f0_hz": round(float(np.median(f0s)) if f0s else 0.0, 1),
        "f0_std_hz": round(float(np.std(f0s)) if len(f0s) >= 3 else 0.0, 1),
    }


def verify_spoken_audio(
    cloud: EasyColabCloudClient,
    audio_path: Path,
    expected_text: str,
    min_similarity: float = 0.80,
    min_f0_std: float = 12.0,
    max_duration_s: float = 13.5,
) -> Dict[str, Any]:
    stats = compute_audio_prosody_stats(audio_path)
    transcript = cloud.run_qwen3_asr_blind(audio_path)
    norm_exp = normalize_zh_chars(expected_text)
    norm_got = normalize_zh_chars(transcript)

    if not norm_exp:
        sim = 1.0
        passed = stats["rms"] >= 0.010 and stats["duration_s"] <= max_duration_s
    else:
        sim = round(difflib.SequenceMatcher(None, norm_exp, norm_got).ratio(), 3) if norm_got else 0.0
        short_ok = len(norm_exp) <= 3 and (norm_exp in norm_got or sim >= 0.5)
        leaked = bool(re.search(r"Direction|speaker|录音棚|微哑|女帝|男宠|配音", transcript, re.I))
        f0_ok = (stats["f0_std_hz"] >= min_f0_std) or (len(norm_exp) <= 5)
        passed = (
            stats["rms"] >= 0.018
            and stats["duration_s"] <= max_duration_s
            and (sim >= min_similarity or short_ok)
            and f0_ok
            and not leaked
        )
    return {
        **stats,
        "expected_norm": norm_exp,
        "transcript_raw": transcript,
        "transcript_norm": norm_got,
        "similarity": sim,
        "passed": passed,
    }


def probe_video_duration(video_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        text=True,
    ).strip()
    return round(float(out or 0.0), 3)


def detect_video_signals(video_path: Path, allow_speech_hold: bool = False) -> VideoMachineSignals:
    """Single-pass FFmpeg detection for black frames, freeze spans, and internal jump cuts."""
    dur = probe_video_duration(video_path)
    vf = "setpts=PTS-STARTPTS,blackdetect=d=0.10,freezedetect=n=-58dB:d=0.55,select='gt(scene,0.42)',showinfo"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-i", str(video_path), "-map", "0:v:0", "-vf", vf, "-an", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    log = proc.stderr or ""
    blacks: List[Tuple[float, float]] = []
    freezes: List[Tuple[float, float]] = []
    cuts: List[float] = []
    freeze_st: Optional[float] = None

    for line in log.splitlines():
        if m := BLACK_RE.search(line):
            blacks.append((round(float(m.group(1)), 3), round(float(m.group(2)), 3)))
        elif m := FREEZE_START_RE.search(line):
            freeze_st = float(m.group(1))
        elif m := FREEZE_END_RE.search(line):
            if freeze_st is not None:
                freezes.append((round(freeze_st, 3), round(float(m.group(1)), 3)))
                freeze_st = None
        elif "showinfo" in line and (m := SCENE_CUT_RE.search(line)):
            t_c = round(float(m.group(1)), 3)
            if 0.12 < t_c < dur - 0.12:
                cuts.append(t_c)
    if freeze_st is not None and dur - freeze_st >= 0.55:
        freezes.append((round(freeze_st, 3), dur))

    rec_in = 0.0
    rec_out = dur
    for b_st, b_ed in blacks:
        if b_st <= 0.08 and b_ed < dur * 0.4:
            rec_in = max(rec_in, b_ed)
        elif b_ed >= dur - 0.08 and b_st > dur * 0.6:
            rec_out = min(rec_out, b_st)

    if not allow_speech_hold:
        for f_st, f_ed in freezes:
            if f_ed >= dur - 0.08 and f_st >= max(1.5, dur * 0.68):
                # Shave excess trailing freeze while leaving a natural 0.20s settle
                rec_out = min(rec_out, round(min(dur, f_st + 0.20), 3))

    if rec_out - rec_in < 1.2:
        rec_in, rec_out = 0.0, dur

    return VideoMachineSignals(
        duration_s=dur,
        black_spans=blacks,
        freeze_spans=freezes,
        internal_cuts=sorted(set(cuts)),
        recommended_in_s=round(rec_in, 3),
        recommended_out_s=round(rec_out, 3),
    )


def build_video_contact_sheet(
    video_path: Path,
    out_jpg: Path,
    shot_id: str,
    signals: VideoMachineSignals,
    num_frames: int = 6,
) -> Path:
    """Compose a timestamped 3x2 dark-mode contact sheet with BLACK/FREEZE/CUT badges."""
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.5, signals.duration_s)
    sample_times = [
        round(min(dur - 0.06, max(0.08, dur * (i + 0.5) / float(num_frames))), 2)
        for i in range(num_frames)
    ]
    sample_times[0] = min(0.20, dur * 0.1)
    sample_times[-1] = max(0.20, round(dur - 0.14, 2))

    tiles: List[Image.Image] = []
    tmp_dir = out_jpg.parent / "_tmp_tiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, t in enumerate(sample_times):
        f_jpg = tmp_dir / f"{shot_id}_t{idx}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(f_jpg)],
            check=True,
        )
        tiles.append(Image.open(f_jpg).convert("RGB"))
        f_jpg.unlink(missing_ok=True)

    cols = 3
    rows = math.ceil(len(tiles) / cols)
    tw = 480
    th = max(200, int(round(tw * tiles[0].height / float(tiles[0].width))))
    pad = 10
    hdr_h = 40
    lbl_h = 30
    sheet_w = cols * tw + (cols + 1) * pad
    sheet_h = hdr_h + rows * (th + lbl_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (pad, 12),
        f"{shot_id} | dur={signals.duration_s:.2f}s | trim=[{signals.recommended_in_s:.2f}s..{signals.recommended_out_s:.2f}s] | cuts={len(signals.internal_cuts)}",
        fill=(235, 240, 245),
        font=font,
    )

    badge_colors = {"CUT": (80, 195, 255), "BLACK": (255, 145, 55), "FREEZE": (255, 95, 190)}
    for idx, (tile, t) in enumerate(zip(tiles, sample_times)):
        r, c = divmod(idx, cols)
        x = pad + c * (tw + pad)
        y = hdr_h + pad + r * (th + lbl_h + pad)
        sheet.paste(tile.resize((tw, th), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 4, y + th + 6), f"t={t:.2f}s", fill=(220, 225, 230), font=font)
        bx = x + 90
        for tag in signals.tags_near(t):
            draw.text((bx, y + th + 6), f"[{tag}]", fill=badge_colors.get(tag, (255, 220, 80)), font=font)
            bx += 65

    sheet.save(out_jpg, format="JPEG", quality=88)
    return out_jpg


def vlm_inspect_image_asset(
    cloud: EasyColabCloudClient,
    asset_type: str,
    candidate_img: Path,
    requirement_summary: str,
    ref_face: Optional[Path] = None,
) -> Dict[str, Any]:
    """Gate 1 & Gate 4 VLM Visual Inspector via Qwen3.8-27B-Uncensored-VL."""
    imgs = [candidate_img]
    ref_note = ""
    if ref_face and ref_face.is_file():
        imgs.append(ref_face)
        ref_note = "Image 2 is the canonical character face anchor; verify Image 1 preserves the same person's facial identity and gender."

    sys_p = (
        "You are a strict Cinema Quality Control VLM Supervisor. Inspect Image 1 carefully against the production spec. "
        "Respond ONLY with a single JSON object: "
        '{"passed": true/false, "score": 1-10, "issues": ["..."], "fix_hint": "..."}. '
        "Pass (`passed: true`, score >= 7) ONLY if there is no severe anatomical deformity, no garbled watermark/subtitles, "
        "and the core subject/layout matches the requirement."
    )
    usr_p = f"Asset Type: {asset_type}\nRequirement: {requirement_summary}\n{ref_note}\nReturn JSON verdict now."
    res = cloud.chat_json(sys_p, usr_p, images=imgs, temperature=0.15, max_tokens=600)
    score = float(res.get("score") or 7.0)
    passed = bool(res.get("passed", score >= 6.5)) and score >= 6.5
    return {"passed": passed, "score": score, "issues": res.get("issues") or [], "fix_hint": str(res.get("fix_hint") or "")}


def vlm_inspect_shot_contact_sheet(
    cloud: EasyColabCloudClient,
    contact_sheet: Path,
    shot_summary: str,
    signals: VideoMachineSignals,
    ref_face: Optional[Path] = None,
) -> Dict[str, Any]:
    """Gate 5B VLM Footage Inspector reading ContactSheet + FFmpeg machine signals."""
    imgs = [contact_sheet]
    ref_note = ""
    if ref_face and ref_face.is_file():
        imgs.append(ref_face)
        ref_note = "Image 2 is the primary character's face reference."

    sys_p = (
        "You are a Cinema Editor & Footage QC Supervisor (modeled after ArcReel review-footage). "
        "Image 1 is a 6-frame timestamped ContactSheet of one generated video shot. "
        "Evaluate visual continuity, character/scene stability, and whether any black/garbled meltdown occurs. "
        "Output ONLY a JSON object: "
        '{"passed": true/false, "score": 1-10, "trim_in_s": number, "trim_out_s": number, "issues": ["..."]}.'
    )
    usr_p = (
        f"Shot Spec: {shot_summary}\n"
        f"FFmpeg Signals: duration={signals.duration_s:.2f}s, black={signals.black_spans}, "
        f"freeze={signals.freeze_spans}, internal_cuts={signals.internal_cuts}, "
        f"suggested_trim=[{signals.recommended_in_s:.2f}, {signals.recommended_out_s:.2f}].\n"
        f"{ref_note}\nReturn JSON verdict."
    )
    res = cloud.chat_json(sys_p, usr_p, images=imgs, temperature=0.15, max_tokens=600)
    score = float(res.get("score") or 7.5)
    passed = bool(res.get("passed", score >= 6.0)) and score >= 6.0
    t_in = max(0.0, min(signals.duration_s - 1.0, float(res.get("trim_in_s", signals.recommended_in_s) or 0.0)))
    t_out = max(t_in + 1.0, min(signals.duration_s, float(res.get("trim_out_s", signals.recommended_out_s) or signals.duration_s)))
    return {
        "passed": passed,
        "score": score,
        "trim_in_s": round(t_in, 3),
        "trim_out_s": round(t_out, 3),
        "issues": res.get("issues") or [],
    }
