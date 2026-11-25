"""Stage 6 Post-Production & Multi-Format Delivery Exporters:
1. Per-Cut Editorial Cinema Stitcher (`hard` vs lip-sync-safe `jcut` + `trim_in..trim_out`
   + EBU R128 `-16 LUFS` loudness normalization + `.srt` subtitle generation & optional burn-in).
2. Self-Contained Offline HTML Cinema Storyboard & QA Dashboard (inspired by LaoLi `build_board_lite.py` + ArcReel).
3. Jianying Pro (`剪映专业版`) Draft Package Exporter (`draft_content.json` + `draft_meta_info.json` + media `.zip`).
"""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .models import EpisodeBible, ProductionConfig, ShotSpec
from .qa_engine import probe_video_duration


def _fmt_srt_ts(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000.0)))
    hh, rem = divmod(ms_total, 3600000)
    mm, rem = divmod(rem, 60000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def render_final_cinema_mp4(cfg: ProductionConfig, bible: EpisodeBible) -> Dict[str, Any]:
    """Trim each shot to [trim_in_s, trim_out_s], apply per-shot Hard/J-Cut assembly, EBU R128, and SRT subtitles."""
    edit_dir = cfg.project_dir / "edit"
    trimmed_dir = edit_dir / "trimmed_shots"
    trimmed_dir.mkdir(parents=True, exist_ok=True)

    trimmed_clips: List[Tuple[ShotSpec, Path, float]] = []
    for shot in bible.shots:
        src = Path(shot.video_path)
        if not src.is_file():
            raise FileNotFoundError(f"Missing rendered video for {shot.shot_id}: {src}")
        full_dur = probe_video_duration(src)
        t_in = max(0.0, min(full_dur - 0.8, float(shot.trim_in_s or 0.0)))
        t_out = max(t_in + 0.8, min(full_dur, float(shot.trim_out_s or full_dur)))
        clip_dur = round(t_out - t_in, 3)
        dst = trimmed_dir / f"{shot.shot_id}_trim.mp4"

        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{t_in:.3f}", "-t", f"{clip_dur:.3f}",
                "-i", str(src),
                "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p", "-r", "24",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                str(dst),
            ],
            check=True,
        )
        actual_d = probe_video_duration(dst)
        trimmed_clips.append((shot, dst, actual_d))

    # Build per-shot J-Cut / Hard-Cut timeline
    n = len(trimmed_clips)
    leads: List[float] = [0.0]
    for i in range(1, n):
        sh_i, _, d_i = trimmed_clips[i]
        _, _, d_prev = trimmed_clips[i - 1]
        if sh_i.cut_in == "jcut" and sh_i.shot_type == "dialogue_lipsync" and d_i >= 1.8 and d_prev >= 1.5:
            leads.append(0.16)
        else:
            leads.append(0.0)

    cut_times: List[float] = [0.0]
    t_cursor = trimmed_clips[0][2]
    for i in range(1, n):
        cut_times.append(round(t_cursor, 4))
        t_cursor += max(0.3, trimmed_clips[i][2] - leads[i])
    total_timeline_s = round(t_cursor, 3)

    # Write accurate SRT subtitle file synchronized to the assembled timeline
    srt_path = cfg.project_dir / f"{cfg.project_name}_subtitles.srt"
    srt_blocks: List[str] = []
    sub_idx = 1
    cmap = bible.char_map()
    for i, (shot, _, d_i) in enumerate(trimmed_clips):
        if shot.shot_type == "dialogue_lipsync" and shot.spoken_text.strip():
            st_s = max(0.0, cut_times[i] + 0.05)
            ed_s = min(total_timeline_s - 0.05, cut_times[i] + max(1.0, d_i - leads[i] - 0.12))
            spk_name = cmap[shot.active_speaker].name if (shot.active_speaker and shot.active_speaker in cmap) else ""
            clean_txt = shot.spoken_text.strip()
            srt_blocks.append(f"{sub_idx}\n{_fmt_srt_ts(st_s)} --> {_fmt_srt_ts(ed_s)}\n{clean_txt}\n")
            sub_idx += 1
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    # Assemble master video + R128 loudness normalized audio
    filters: List[str] = []
    v_labels: List[str] = []
    a_labels: List[str] = []

    for i, (_, _, d_i) in enumerate(trimmed_clips):
        if i == 0 or leads[i] <= 0.0:
            filters.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        else:
            filters.append(f"[{i}:v]trim=start={leads[i]:.4f},setpts=PTS-STARTPTS[v{i}]")
        v_labels.append(f"[v{i}]")

        fade_in_d = 0.14 if leads[i] > 0 else 0.035
        fade_out_d = 0.14 if (i + 1 < n and leads[i + 1] > 0) else 0.035
        fade_out_st = max(0.0, d_i - fade_out_d)
        achain = (
            f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"afade=t=in:st=0:d={fade_in_d:.4f},afade=t=out:st={fade_out_st:.4f}:d={fade_out_d:.4f}"
        )
        if i > 0:
            delay_ms = max(0, int(round((cut_times[i] - leads[i]) * 1000.0)))
            if delay_ms > 0:
                achain += f",adelay={delay_ms}|{delay_ms}"
        achain += f"[a{i}]"
        filters.append(achain)
        a_labels.append(f"[a{i}]")

    filters.append(f"{''.join(v_labels)}concat=n={n}:v=1:a=0[vcat]")
    filters.append(
        f"{''.join(a_labels)}amix=inputs={n}:duration=longest:normalize=0,"
        "loudnorm=I=-16.0:TP=-1.5:LRA=11[aout]"
    )

    clean_master_mp4 = cfg.project_dir / f"{cfg.project_name}_master_clean.mp4"
    final_mp4 = cfg.project_dir / f"{cfg.project_name}_final.mp4"

    ff_cmd = ["ffmpeg", "-y", "-v", "error"]
    for _, clip_p, _ in trimmed_clips:
        ff_cmd.extend(["-i", str(clip_p)])
    ff_cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[vcat]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        str(clean_master_mp4),
    ])
    subprocess.run(ff_cmd, check=True)

    # Burn cinema subtitles if enabled and ffmpeg supports subtitles filter; else copy clean master
    if cfg.burn_subtitles and srt_blocks:
        srt_escaped = str(srt_path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        style_str = "FontName=PingFang SC,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,BorderStyle=1,Outline=2,Shadow=1,MarginV=26"
        burn = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(clean_master_mp4),
                "-vf", f"subtitles='{srt_escaped}':force_style='{style_str}'",
                "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                "-c:a", "copy", str(final_mp4),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if burn.returncode != 0:
            shutil.copy2(clean_master_mp4, final_mp4)
    else:
        shutil.copy2(clean_master_mp4, final_mp4)

    vol = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(final_mp4), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    vol_lines = [l.strip() for l in (vol.stderr or "").splitlines() if "mean_volume" in l or "max_volume" in l]
    final_dur = probe_video_duration(final_mp4)

    return {
        "final_mp4": str(final_mp4),
        "clean_master_mp4": str(clean_master_mp4),
        "srt_path": str(srt_path),
        "shot_count": n,
        "duration_s": final_dur,
        "asl_s": round(final_dur / max(1, n), 2),
        "size_bytes": final_mp4.stat().st_size,
        "volumedetect": vol_lines,
        "cut_times_s": cut_times,
    }


def export_jianying_draft_bundle(cfg: ProductionConfig, bible: EpisodeBible) -> Dict[str, str]:
    """Export a self-contained Jianying Pro (5.x/6+) draft folder & .zip with video + subtitle tracks."""
    draft_root = cfg.project_dir / "jianying_draft" / cfg.project_name
    media_dir = draft_root / "Resources"
    if draft_root.exists():
        shutil.rmtree(draft_root)
    media_dir.mkdir(parents=True, exist_ok=True)

    w_canvas, h_canvas = (1920, 1080) if bible.aspect_ratio == "16:9" else (1080, 1920)
    videos_mat: List[Dict[str, Any]] = []
    texts_mat: List[Dict[str, Any]] = []
    video_segs: List[Dict[str, Any]] = []
    text_segs: List[Dict[str, Any]] = []

    cursor_us = 0
    trimmed_dir = cfg.project_dir / "edit" / "trimmed_shots"
    for shot in bible.shots:
        src = trimmed_dir / f"{shot.shot_id}_trim.mp4"
        if not src.is_file():
            src = Path(shot.video_path)
        dst_media = media_dir / src.name
        shutil.copy2(src, dst_media)
        dur_us = max(500_000, int(round(probe_video_duration(dst_media) * 1_000_000)))
        vid_id = str(uuid.uuid4()).upper()
        videos_mat.append({
            "id": vid_id,
            "type": "video",
            "path": str(dst_media),
            "material_name": dst_media.name,
            "duration": dur_us,
            "width": w_canvas,
            "height": h_canvas,
        })
        video_segs.append({
            "id": str(uuid.uuid4()).upper(),
            "material_id": vid_id,
            "source_timerange": {"start": 0, "duration": dur_us},
            "target_timerange": {"start": cursor_us, "duration": dur_us},
            "volume": 1.0,
        })
        if shot.spoken_text.strip():
            tid = str(uuid.uuid4()).upper()
            texts_mat.append({
                "id": tid,
                "type": "subtitle",
                "content": json.dumps({"text": shot.spoken_text.strip(), "styles": [{"fill": {"content": {"solid": {"color": [1.0, 1.0, 1.0]}}}, "size": 8.0}]}),
            })
            text_segs.append({
                "id": str(uuid.uuid4()).upper(),
                "material_id": tid,
                "target_timerange": {"start": cursor_us + 50_000, "duration": max(400_000, dur_us - 120_000)},
            })
        cursor_us += dur_us

    draft_content = {
        "id": str(uuid.uuid4()).upper(),
        "version": 360000,
        "fps": 24.0,
        "duration": cursor_us,
        "canvas_config": {"width": w_canvas, "height": h_canvas, "ratio": bible.aspect_ratio},
        "materials": {"videos": videos_mat, "texts": texts_mat, "audios": []},
        "tracks": [
            {"id": str(uuid.uuid4()).upper(), "type": "video", "segments": video_segs},
            {"id": str(uuid.uuid4()).upper(), "type": "text", "segments": text_segs},
        ],
    }
    now_us = int(time.time() * 1_000_000)
    draft_meta = {
        "draft_id": draft_content["id"],
        "draft_name": bible.title or cfg.project_name,
        "draft_fold_path": str(draft_root),
        "tm_draft_create": now_us,
        "tm_draft_modified": now_us,
        "tm_duration": cursor_us,
    }
    (draft_root / "draft_content.json").write_text(json.dumps(draft_content, ensure_ascii=False, indent=2), encoding="utf-8")
    (draft_root / "draft_info.json").write_text(json.dumps(draft_content, ensure_ascii=False, indent=2), encoding="utf-8")
    (draft_root / "draft_meta_info.json").write_text(json.dumps(draft_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_path = cfg.project_dir / f"{cfg.project_name}_jianying_draft.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(draft_root.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=str(Path(cfg.project_name) / f.relative_to(draft_root)))

    return {"draft_dir": str(draft_root), "draft_zip": str(zip_path)}


def export_html_storyboard_board(
    cfg: ProductionConfig, bible: EpisodeBible, final_info: Dict[str, Any]
) -> Path:
    """Generate a single-file dark-mode Cinema Storyboard & QA Workbench HTML with relative media links."""
    board_path = cfg.project_dir / f"{cfg.project_name}_board.html"

    def _rel(p_str: str) -> str:
        if not p_str:
            return ""
        try:
            return os.path.relpath(Path(p_str).resolve(), cfg.project_dir.resolve())
        except Exception:
            return p_str

    char_cards_html = []
    for c in bible.characters:
        char_cards_html.append(f"""
        <div class="card">
          <div class="badge">{html.escape(c.gender)} · {html.escape(c.age_range)} · {'用户锁脸' if c.user_face_path else 'AI原生造脸'}</div>
          <h3>{html.escape(c.name)} <small>({html.escape(c.id)})</small></h3>
          <div class="row-imgs">
            <img src="{html.escape(_rel(c.face_anchor_path))}" alt="face" style="width:110px;height:110px;object-fit:cover;border-radius:8px;">
            <img src="{html.escape(_rel(c.multiview_sheet_path))}" alt="4view" style="height:110px;border-radius:8px;">
          </div>
          <p class="meta"><b>音色标签:</b> {html.escape(c.voice_label_en)}</p>
          <audio controls src="{html.escape(_rel(c.voice_ref_path))}" style="width:100%;height:32px;"></audio>
        </div>""")

    loc_cards_html = []
    for loc in bible.locations:
        loc_cards_html.append(f"""
        <div class="card">
          <div class="badge">{html.escape(loc.space_type)} · 3×3九宫格 + 四向底板</div>
          <h3>{html.escape(loc.name)} <small>({html.escape(loc.id)})</small></h3>
          <div class="row-imgs">
            <img src="{html.escape(_rel(loc.master_plate_path))}" style="height:120px;border-radius:8px;">
            <img src="{html.escape(_rel(loc.grid9_sheet_path))}" style="height:120px;border-radius:8px;">
          </div>
        </div>""")

    shot_rows_html = []
    for s in bible.shots:
        shot_rows_html.append(f"""
        <div class="shot-card">
          <div class="shot-hdr">
            <span class="shot-num">#{s.index:02d} {html.escape(s.shot_id)}</span>
            <span class="pill">{html.escape(s.shot_type)}</span>
            <span class="pill accent">{html.escape(s.scale)} · {s.lens_mm}mm · {html.escape(s.camera_angle)}</span>
            <span class="pill">{s.target_seconds:.2f}s ({s.cut_in.upper()})</span>
            <span class="pill ok">VLM {s.vlm_score:.1f} · ASR {s.asr_similarity:.2f}</span>
          </div>
          <div class="shot-grid">
            <img src="{html.escape(_rel(s.keyframe_path))}" title="Keyframe">
            <video controls preload="metadata" src="{html.escape(_rel(s.video_path))}"></video>
            <img src="{html.escape(_rel(s.contact_sheet_path))}" title="Contact Sheet">
          </div>
          <div class="shot-meta">
            <div><b>台词:</b> {html.escape(s.spoken_text or '（无对白 · 视觉/音效蒙太奇）')} {f'<em>[重音: {html.escape(s.accent_word)}]</em>' if s.accent_word else ''}</div>
            <div><b>运镜 & 表演:</b> {html.escape(s.camera_motion)} | {html.escape(s.acting_expression)} | {html.escape(s.acting_gesture)}</div>
            <div><b>状态账:</b> {html.escape(s.state_before)} → {html.escape(s.state_after)} | <b>裁切区间:</b> [{s.trim_in_s:.2f}s .. {s.trim_out_s:.2f}s]</div>
          </div>
        </div>""")

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(bible.title)} · EasyColab 影视级短剧工作台看板</title>
<style>
  body {{ background:#0d0f14; color:#e8ecf2; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; margin:0; padding:28px; }}
  h1,h2,h3 {{ margin:0 0 10px; }}
  .hero {{ display:grid; grid-template-columns: 1.3fr 1fr; gap:24px; background:#161a23; padding:22px; border-radius:14px; margin-bottom:28px; border:1px solid #262c3a; }}
  .hero video {{ width:100%; border-radius:10px; background:#000; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; margin-bottom:28px; }}
  .card {{ background:#161a23; padding:16px; border-radius:12px; border:1px solid #262c3a; }}
  .row-imgs {{ display:flex; gap:10px; overflow-x:auto; margin:10px 0; }}
  .badge,.pill {{ display:inline-block; font-size:12px; padding:3px 9px; border-radius:99px; background:#252c3d; color:#b8c4e0; margin-right:6px; margin-bottom:6px; }}
  .pill.accent {{ background:#3b2648; color:#f3b6ff; }}
  .pill.ok {{ background:#193929; color:#7ee2a8; }}
  .shot-card {{ background:#161a23; border:1px solid #262c3a; border-radius:12px; padding:16px; margin-bottom:16px; }}
  .shot-grid {{ display:grid; grid-template-columns:1fr 1fr 1.15fr; gap:12px; margin:10px 0; align-items:center; }}
  .shot-grid img,.shot-grid video {{ width:100%; border-radius:8px; background:#000; }}
  .shot-meta,.meta {{ font-size:13px; color:#aeb7c8; line-height:1.6; }}
  .shot-num {{ font-weight:700; font-size:15px; margin-right:8px; color:#fff; }}
</style>
</head>
<body>
  <div class="hero">
    <div>
      <video controls src="{html.escape(_rel(final_info['final_mp4']))}"></video>
    </div>
    <div>
      <div class="badge">EasyColab CinemaStudio · 全自动影视级短剧系统</div>
      <h1>{html.escape(bible.title)}</h1>
      <p class="meta">{html.escape(bible.synopsis)}</p>
      <p class="meta"><b>成片规格:</b> {bible.aspect_ratio} · 共 <b>{final_info['shot_count']}</b> 镜 · 总长 <b>{final_info['duration_s']}s</b> · 平均镜长 ASL <b>{final_info['asl_s']}s</b></p>
      <p class="meta"><b>响度实测:</b> {html.escape(' / '.join(final_info.get('volumedetect', [])))}</p>
      <p class="meta"><b>画风锚定:</b> {html.escape(bible.style_phrase_en)}</p>
    </div>
  </div>
  <h2>一、角色四视图与声纹资产库</h2>
  <div class="cards">{''.join(char_cards_html)}</div>
  <h2>二、场景母版与 3×3 九机位空间矩阵</h2>
  <div class="cards">{''.join(loc_cards_html)}</div>
  <h2>三、影视级分镜时间线 ({len(bible.shots)} 镜 · 首帧 + H3成片 + 机器信号联系表)</h2>
  {''.join(shot_rows_html)}
</body>
</html>"""
    board_path.write_text(doc, encoding="utf-8")
    return board_path
