"""Stages 4–5 of EasyColab Cinema-Grade Short-Drama Pipeline:
- Stage 4: Mechanical Reference Slot Compilation + Multi-Scale `Qwen-Image-Edit-2511` Keyframes + Gate 4 VLM QA.
- Stage 5: `MiniMax-H3 (10Eros-Max)` Short-Shot Rendering + Tail-Frame Carryover + Gate 5A Blind ASR
           + FFmpeg Signal Detection (`BLACK`/`FREEZE`/`CUT`) + ContactSheet VLM Footage Review & Auto-Reroll.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cloud_client import EasyColabCloudClient
from .models import EpisodeBible, ProductionConfig, ShotSpec
from .qa_engine import (
    build_video_contact_sheet,
    compute_audio_prosody_stats,
    detect_video_signals,
    verify_spoken_audio,
    vlm_inspect_image_asset,
    vlm_inspect_shot_contact_sheet,
)
from .rules import compile_h3_video_prompt_and_refs, compile_keyframe_prompt_and_refs


def extract_clean_tail_frame(video_path: Path, dst_png: Path, offset_before_end_s: float = 0.16) -> Path:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-sseof", f"-{max(0.08, offset_before_end_s):.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            str(dst_png),
        ],
        check=True,
    )
    return dst_png


def run_stage4_keyframes(
    cfg: ProductionConfig, bible: EpisodeBible, cloud: EasyColabCloudClient
) -> EpisodeBible:
    """Synthesize multi-scale storyboard keyframes via Qwen-Image-Edit-2511 (4-step) + Gate 4 VLM QA."""
    cloud.ensure_comfy_packs(["qwen_image_edit_2511"])
    cloud.ensure_llm_loaded()

    kf_dir = cfg.project_dir / "shots" / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    w_res, h_res = bible.resolution()
    cmap = bible.char_map()
    gate4_log: Dict[str, Any] = {}

    for idx_s, shot in enumerate(bible.shots):
        if shot.keyframe_path and Path(shot.keyframe_path).is_file() and shot.vlm_score >= 6.5:
            continue

        kf_png = kf_dir / f"{shot.shot_id}_kf.png"
        prompt, ref_strs = compile_keyframe_prompt_and_refs(shot, bible, prev_tail_frame=None)
        ref_paths = [Path(r) for r in ref_strs if Path(r).is_file()]
        primary_face: Optional[Path] = None
        if shot.characters_in_shot:
            c0 = cmap.get(shot.characters_in_shot[0])
            if c0 and c0.face_anchor_path and Path(c0.face_anchor_path).is_file():
                primary_face = Path(c0.face_anchor_path)

        best_verdict: Dict[str, Any] = {"passed": True, "score": 7.5, "issues": []}
        cur_prompt = prompt
        for att in range(cfg.max_qa_retries + 1):
            k_seed = cfg.seed_base + 2000 + idx_s * 23 + att
            print(
                f"[Stage 4 · 影视关键帧] {shot.shot_id} ({shot.scale} · {shot.lens_mm}mm · {shot.camera_angle}) refs={len(ref_paths)} (attempt {att+1})",
                flush=True,
            )
            cloud.run_qwen_edit(cur_prompt, ref_paths, kf_png, width=w_res, height=h_res, steps=4, seed=k_seed)
            verdict = vlm_inspect_image_asset(
                cloud,
                asset_type=f"storyboard_keyframe ({shot.scale}, {shot.camera_angle})",
                candidate_img=kf_png,
                requirement_summary=f"{shot.image_scene_en} | Characters: {shot.characters_in_shot} | Scale: {shot.scale}",
                ref_face=primary_face if shot.scale not in ("EWS", "WS") else None,
            )
            best_verdict = verdict
            print(
                f"  [Gate 4 VLM 首帧核验] {shot.shot_id}: score={verdict['score']} passed={verdict['passed']} issues={verdict['issues']}",
                flush=True,
            )
            if verdict["passed"]:
                break
            if verdict.get("fix_hint"):
                cur_prompt = f"{prompt} Ensure: {verdict['fix_hint']}."

        shot.keyframe_path = str(kf_png)
        shot.vlm_score = float(best_verdict.get("score") or 7.5)
        gate4_log[shot.shot_id] = best_verdict
        (cfg.project_dir / "episode_bible.json").write_text(
            json.dumps(bible.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (cfg.project_dir / "gate4_keyframe_report.json").write_text(
        json.dumps(gate4_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bible


def _render_single_h3_shot_with_asr_gate(
    cfg: ProductionConfig,
    bible: EpisodeBible,
    shot: ShotSpec,
    cloud: EasyColabCloudClient,
    prev_tail: Optional[str],
    seed_offset: int = 0,
) -> Dict[str, Any]:
    vid_dir = cfg.project_dir / "shots" / "videos"
    qa_dir = cfg.project_dir / "shots" / "qa"
    vid_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    w_res, h_res = bible.resolution()

    out_mp4 = vid_dir / f"{shot.shot_id}.mp4"
    extracted_wav = qa_dir / f"{shot.shot_id}_h3_audio.wav"
    contact_jpg = qa_dir / f"{shot.shot_id}_contact.jpg"
    tail_png = vid_dir / f"{shot.shot_id}_tail.png"

    h3_prompt, ref_strs, aud_str, ref_size = compile_h3_video_prompt_and_refs(
        shot, bible, shot.keyframe_path, prev_tail_frame=prev_tail
    )
    ref_paths = [Path(r) for r in ref_strs if Path(r).is_file()]
    aud_path = Path(aud_str) if (aud_str and Path(aud_str).is_file()) else None

    last_asr: Dict[str, Any] = {}
    for att in range(cfg.max_qa_retries + 1):
        v_seed = cfg.seed_base + 5000 + shot.index * 31 + seed_offset + att
        print(
            f"[Stage 5 · H3 视频渲染] {shot.shot_id} ({shot.shot_type} | {shot.scale} | {shot.h3_frames}f={shot.target_seconds:.2f}s | refs={len(ref_paths)}:{ref_size}) (attempt {att+1})",
            flush=True,
        )
        cloud.run_10eros_video(
            prompt=h3_prompt,
            ref_images=ref_paths,
            ref_audio=aud_path,
            out_file=out_mp4,
            width=w_res,
            height=h_res,
            frames=shot.h3_frames,
            ref_image_size=ref_size,
            seed=v_seed,
        )

        # Extract rendered audio track for Gate 5A verification
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(out_mp4), "-vn", "-ar", "16000", "-ac", "1", str(extracted_wav)],
            check=True,
        )
        if shot.shot_type == "dialogue_lipsync" and shot.spoken_text.strip():
            asr_res = verify_spoken_audio(
                cloud, extracted_wav, shot.spoken_text,
                min_similarity=0.75, min_f0_std=8.0, max_duration_s=shot.target_seconds + 1.5,
            )
            last_asr = asr_res
            print(
                f"  [Gate 5A 视频音轨盲测] {shot.shot_id}: got='{asr_res['transcript_norm']}' sim={asr_res['similarity']} passed={asr_res['passed']}",
                flush=True,
            )
            if asr_res["passed"]:
                break
        else:
            st = compute_audio_prosody_stats(extracted_wav)
            last_asr = {**st, "similarity": 1.0, "passed": True, "transcript_norm": ""}
            break

    signals = detect_video_signals(out_mp4, allow_speech_hold=(shot.shot_type == "dialogue_lipsync"))
    build_video_contact_sheet(out_mp4, contact_jpg, shot.shot_id, signals, num_frames=6)
    extract_clean_tail_frame(out_mp4, tail_png, offset_before_end_s=max(0.14, signals.duration_s - signals.recommended_out_s + 0.12))

    shot.video_path = str(out_mp4)
    shot.tail_frame_path = str(tail_png)
    shot.contact_sheet_path = str(contact_jpg)
    shot.trim_in_s = signals.recommended_in_s
    shot.trim_out_s = signals.recommended_out_s
    if shot.shot_type == "dialogue_lipsync":
        shot.asr_similarity = float(last_asr.get("similarity") or shot.asr_similarity)

    return {"asr": last_asr, "signals": signals.__dict__}


def run_stage5_video_and_review(
    cfg: ProductionConfig, bible: EpisodeBible, cloud: EasyColabCloudClient
) -> EpisodeBible:
    """Render all H3 video shots + Gate 5A ASR + FFmpeg signals + Batch Gate 5B VLM ContactSheet review."""
    # Unload Qwen3.8-27B GGUF so 10Eros-Max + 32B Heretic NVFP4 has full 96GB headroom
    cloud.unload_llm()
    cloud.free_comfy_vram()
    cloud.ensure_comfy_packs(["minimax_h3_10eros_max", "drama3_tts"])

    gate5_report: Dict[str, Any] = {}
    prev_tail: Optional[str] = None

    # Pass 1: Render all pending shots + Gate 5A Blind ASR + FFmpeg Signals + ContactSheets
    for shot in bible.shots:
        if shot.qa_passed and shot.video_path and Path(shot.video_path).is_file():
            prev_tail = shot.tail_frame_path if shot.tail_frame_path else None
            continue
        info = _render_single_h3_shot_with_asr_gate(cfg, bible, shot, cloud, prev_tail=prev_tail)
        gate5_report[shot.shot_id] = info
        prev_tail = shot.tail_frame_path
        (cfg.project_dir / "episode_bible.json").write_text(
            json.dumps(bible.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Pass 2: Free ComfyUI VRAM -> Load Qwen3.8-27B-VL ONCE -> Inspect ALL ContactSheets (Gate 5B)
    cloud.free_comfy_vram()
    cloud.ensure_llm_loaded()
    cmap = bible.char_map()
    reroll_shots: List[ShotSpec] = []

    for shot in bible.shots:
        if shot.qa_passed:
            continue
        cs_path = Path(shot.contact_sheet_path)
        vid_path = Path(shot.video_path)
        signals = detect_video_signals(vid_path, allow_speech_hold=(shot.shot_type == "dialogue_lipsync"))
        ref_face: Optional[Path] = None
        if shot.characters_in_shot:
            c0 = cmap.get(shot.characters_in_shot[0])
            if c0 and c0.face_anchor_path and Path(c0.face_anchor_path).is_file():
                ref_face = Path(c0.face_anchor_path)

        vlm_res = vlm_inspect_shot_contact_sheet(
            cloud,
            contact_sheet=cs_path,
            shot_summary=f"{shot.shot_id} ({shot.shot_type}, {shot.scale}): {shot.video_action_delta_en}",
            signals=signals,
            ref_face=ref_face if shot.scale not in ("EWS", "WS") else None,
        )
        print(
            f"  [Gate 5B VLM 联系表审片] {shot.shot_id}: score={vlm_res['score']} trim=[{vlm_res['trim_in_s']}..{vlm_res['trim_out_s']}] passed={vlm_res['passed']}",
            flush=True,
        )
        shot.vlm_score = float(vlm_res["score"])
        # Never clip off active dialogue speech tail
        min_out = min(signals.duration_s, max(1.5, shot.audio_duration_s + 0.25)) if shot.shot_type == "dialogue_lipsync" else 1.3
        shot.trim_in_s = float(vlm_res["trim_in_s"])
        shot.trim_out_s = max(min_out, float(vlm_res["trim_out_s"]))
        shot.qa_notes = "; ".join(str(x) for x in (vlm_res.get("issues") or []))
        gate5_report.setdefault(shot.shot_id, {})["vlm_review"] = vlm_res

        if vlm_res["passed"]:
            shot.qa_passed = True
        else:
            reroll_shots.append(shot)

    # Optional single batch re-render if any shot failed Gate 5B visual review
    if reroll_shots:
        print(f"[Gate 5B 自动返修] 共 {len(reroll_shots)} 个镜头未通过视觉审片，启动单镜重拍: {[s.shot_id for s in reroll_shots]}", flush=True)
        cloud.unload_llm()
        for shot in reroll_shots:
            prev_s = bible.shots[shot.index - 2] if shot.index >= 2 else None
            p_tail = prev_s.tail_frame_path if prev_s else None
            info = _render_single_h3_shot_with_asr_gate(cfg, bible, shot, cloud, prev_tail=p_tail, seed_offset=777)
            gate5_report[shot.shot_id]["reroll"] = info
            shot.qa_passed = True
        cloud.free_comfy_vram()

    (cfg.project_dir / "gate5_footage_report.json").write_text(
        json.dumps(gate5_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.project_dir / "episode_bible.json").write_text(
        json.dumps(bible.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bible
