"""Cinema Cinematography Grammar, Gate 0 Linter, and Pure-Function Reference Compiler.

Combines:
1. `manju-laoli-skill` camera rules (<=30% flat-frontal cap, 25-40% close-up quota,
   scale-to-focal-length binding, <=5.8s anti-talking-head cap, expression+gesture dual lock).
2. `ArcReel` pure-function reference slot numbering (`@[id]` -> `<Picture N>` / `<Audio 1>`)
   so LLMs never hand-number image slots or leak dangling references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    CharacterSpec,
    EpisodeBible,
    LocationSpec,
    PropSpec,
    ShotSpec,
    VALID_CUT_MODES,
    VALID_SCALES,
    VALID_SHOT_TYPES,
)

# MiniMax-H3 native temporal grid: 17k + 5 frames @ 24fps
H3_GRID_FRAMES = [39, 56, 73, 90, 107, 124, 141, 158, 175, 192]

# LaoLi `short-drama-camera-system.md` §二 Scale -> Focal Length (mm) table
SCALE_LENS_BOUNDS: Dict[str, Tuple[int, int, int]] = {
    # scale: (preferred_mm, min_mm, max_mm)
    "EWS": (20, 14, 28),
    "WS": (24, 16, 35),
    "MWS": (28, 24, 50),
    "MS": (40, 28, 85),
    "OTS": (50, 35, 85),
    "MCU": (65, 50, 100),
    "CU": (85, 75, 135),
    "ECU": (100, 85, 200),
}

# LaoLi `scene-grid-template.md` §二 3x3 Camera Matrices by Space Type
SCENE_GRID_9_MATRICES: Dict[str, List[Tuple[str, str]]] = {
    "indoor_semi": [
        ("g1_entrance", "Eye-level entrance doorway view looking inward across the main room"),
        ("g2_main_zone", "Main activity area wide shot showing primary furniture layout"),
        ("g3_window_side", "90-degree side view facing sunlit lattice windows and light beams"),
        ("g4_reverse_door", "180-degree reverse angle from inside looking back at the entrance doors"),
        ("g5_high_plan", "35-degree elevated high-angle overview of room layout and floor"),
        ("g6_diag_tele", "Diagonal 85mm telephoto compressed depth view across the room"),
        ("g7_prop_zone", "Close-medium view of the main desk/table and signature set dressings"),
        ("g8_wall_side", "270-degree opposite side wall view showing architectural screens and decor"),
        ("g9_window_amb", "Atmospheric light-beam and window lattice cutaway empty view"),
    ],
    "outdoor_building": [
        ("g1_front_wide", "Eye-level front establishing wide shot of the full building"),
        ("g2_front_med", "Front medium view of main entrance steps and central axis"),
        ("g3_side_wide", "90-degree side profile wide shot showing architectural depth"),
        ("g4_reverse_wide", "180-degree reverse courtyard view looking back toward the gate"),
        ("g5_high_overview", "High-angle overview of roofs, courtyard, and pathways"),
        ("g6_diag_corner", "45-degree three-quarter corner view with foreground eaves"),
        ("g7_threshold_detail", "Close architectural view of doorway threshold and carved pillars"),
        ("g8_opposite_side", "270-degree opposite lateral walkway and corridor view"),
        ("g9_sky_eaves", "Atmospheric upward cutaway of sunlit eaves and sky"),
    ],
    "natural_street": [
        ("g1_entry_wide", "Entry wide view looking down the main street or terrain path"),
        ("g2_center_stage", "Central action ground wide shot showing ground texture and landmarks"),
        ("g3_side_alley", "90-degree lateral perspective along side stalls or foliage"),
        ("g4_reverse_street", "180-degree reverse view looking back toward the entry path"),
        ("g5_high_slope", "Elevated overview from balcony or slope showing full path topology"),
        ("g6_low_ground", "Low-angle ground-level view emphasizing cobblestones/floorboards"),
        ("g7_landmark_detail", "Close view of signature environmental landmark"),
        ("g8_narrow_choke", "270-degree angled view of doorway/archway choke point"),
        ("g9_skyline_amb", "Atmospheric skyline and directional light cutaway"),
    ],
}

WALL_TO_GRID_INDEX = {
    "000_front": 1,     # g2_main_zone (0-indexed -> 1)
    "090_right": 2,     # g3_window_side
    "180_reverse": 3,   # g4_reverse_door
    "270_left": 7,      # g8_wall_side
    "high_overview": 4, # g5_high_plan
    "low_ground": 5,    # g6
    "detail": 6,        # g7
    "ambient": 8,       # g9
}


def snap_seconds_to_h3_grid(seconds: float, min_frames: int = 56, max_frames: int = 158) -> Tuple[int, float]:
    raw_frames = max(min_frames, min(max_frames, int(round(seconds * 24.0))))
    snapped = min((f for f in H3_GRID_FRAMES if f >= raw_frames), default=max_frames)
    snapped = max(min_frames, min(max_frames, snapped))
    return snapped, round(snapped / 24.0, 2)


def normalize_zh_chars(text: str) -> str:
    t = re.sub(r"\[[^\]]*\]|<[^>]+>", "", text or "")
    t = re.sub(r"[\s，。！？、；：“”‘’（）()《》【】—…,.!?;:\"'~·\-]+", "", t)
    return t.strip()


def extract_quoted_dialogues(story_text: str) -> List[str]:
    """Extract explicit quoted lines (“...” or 「...」 or "...") from user input story."""
    quotes: List[str] = []
    for pat in (r"“([^”]{2,120})”", r"「([^」]{2,120})」", r'"([^"\n]{2,120})"'):
        for m in re.finditer(pat, story_text or ""):
            q = normalize_zh_chars(m.group(1))
            if len(q) >= 2 and q not in quotes:
                quotes.append(q)
    return quotes


def is_conventional_flat_shot(shot: ShotSpec) -> bool:
    ang = (shot.camera_angle or "").lower()
    has_non_flat_dir = any(
        k in ang for k in ("45", "侧", "profile", "90", "过肩", "ots", "shoulder", "pov", "主观", "反打", "reverse")
    )
    has_non_flat_elev = any(
        k in ang for k in ("仰", "俯", "low", "high", "bird", "鸟瞰", "dutch", "荷兰", "贴地", "floor")
    )
    return (not has_non_flat_dir) and (not has_non_flat_elev)


def normalize_and_enrich_bible(bible: EpisodeBible, target_duration_s: float = 48.0) -> EpisodeBible:
    """Auto-fix deterministic mechanical fields (frame snapping, lens bounds, cut modes, IDs)."""
    for idx, shot in enumerate(bible.shots, start=1):
        shot.index = idx
        if not shot.shot_id:
            shot.shot_id = f"s{idx:02d}"
        shot.scale = shot.scale.upper() if shot.scale.upper() in VALID_SCALES else "MS"
        if shot.shot_type not in VALID_SHOT_TYPES:
            shot.shot_type = "dialogue_lipsync" if shot.spoken_text.strip() else "reaction_cutaway"

        pref_mm, min_mm, max_mm = SCALE_LENS_BOUNDS.get(shot.scale, (50, 28, 100))
        if not (min_mm <= int(shot.lens_mm or 0) <= max_mm):
            shot.lens_mm = pref_mm

        # Estimate realistic clause duration if spoken_text exists
        zh_norm = normalize_zh_chars(shot.spoken_text)
        if zh_norm and shot.shot_type == "dialogue_lipsync":
            est_sec = max(2.33, len(zh_norm) / 3.4 + 0.65)
            dur = max(float(shot.target_seconds or 0.0), est_sec)
            dur = min(dur, 5.88)
            frames, snapped_s = snap_seconds_to_h3_grid(dur, min_frames=56, max_frames=158)
            shot.h3_frames = frames
            shot.target_seconds = snapped_s
            if not shot.active_speaker and shot.characters_in_shot:
                shot.active_speaker = shot.characters_in_shot[0]
            if shot.active_speaker and shot.active_speaker not in shot.characters_in_shot:
                shot.characters_in_shot.insert(0, shot.active_speaker)
        else:
            shot.active_speaker = None
            shot.spoken_text = ""
            dur = max(2.33, min(float(shot.target_seconds or 2.33), 3.75))
            frames, snapped_s = snap_seconds_to_h3_grid(dur, min_frames=56, max_frames=90)
            shot.h3_frames = frames
            shot.target_seconds = snapped_s

        # 3D/CG base rule from LaoLi: forbid flat frontal CU
        if bible.visual_base == "3d_cg" and shot.scale in ("CU", "ECU") and is_conventional_flat_shot(shot):
            shot.camera_angle = "侧45度+微仰"

        # Editorial cut-in rule: hard cuts on action/insert/reaction/establishing; J-cut on dialogue after non-impact
        if shot.cut_in not in VALID_CUT_MODES:
            shot.cut_in = "jcut" if (idx > 1 and shot.shot_type == "dialogue_lipsync") else "hard"
        if idx == 1 or shot.shot_type in ("establishing", "insert_prop", "action_impact", "reaction_cutaway"):
            shot.cut_in = "hard"

        if not shot.location_id and bible.locations:
            shot.location_id = bible.locations[0].id

    return bible


@dataclass
class Gate0LintResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def lint_episode_bible(bible: EpisodeBible, target_duration_s: float = 48.0) -> Gate0LintResult:
    """Gate 0 Cinema Grammar Linter enforcing LaoLi + ArcReel production rules."""
    errors: List[str] = []
    warnings: List[str] = []

    n_shots = len(bible.shots)
    min_required_shots = max(12, int(round(target_duration_s / 4.0)))
    if n_shots < min_required_shots:
        errors.append(
            f"[C_SHOT_COUNT] 镜头数量不足：{n_shots} 镜 < 最低影视级要求 {min_required_shots} 镜（目标总长 ~{target_duration_s:.0f}s，严禁几镜到底站桩对话）。"
        )

    if not bible.characters:
        errors.append("[C_NO_CHAR] 未提取出任何角色资产。")
    if not bible.locations:
        errors.append("[C_NO_LOC] 未提取出任何场景资产。")

    total_dur = sum(s.target_seconds for s in bible.shots)
    asl = total_dur / n_shots if n_shots else 99.0
    if asl > 4.3:
        errors.append(f"[C10_ASL_TOO_SLOW] 平均镜长 ASL={asl:.2f}s > 4.30s，节奏拖沓像舞台站桩，必须拆镜并插入特写/反应镜。")

    # Check single-shot duration & long clauses
    for s in bible.shots:
        zh_len = len(normalize_zh_chars(s.spoken_text))
        if s.shot_type == "dialogue_lipsync":
            if zh_len == 0:
                errors.append(f"[C_EMPTY_LINE] {s.shot_id} 标为 dialogue_lipsync 但台词为空。")
            elif zh_len > 24:
                errors.append(
                    f"[C_LONG_LINE] {s.shot_id} 单镜台词长达 {zh_len} 字（'{s.spoken_text}'），超过 24 字上限，必须按标点气口拆成两个分镜或插入反应切镜！"
                )
            if not s.acting_expression.strip() or not s.acting_gesture.strip():
                errors.append(f"[C_ACTING_DUAL] {s.shot_id} 台词镜缺失具体神态(acting_expression)或具体动作(acting_gesture)双挂描写。")
            if not s.accent_word.strip():
                warnings.append(f"[C_ACCENT] {s.shot_id} 未标注台词重音字。")
        if s.target_seconds > 6.2:
            errors.append(f"[C10_MAX_DUR] {s.shot_id} 单镜时长 {s.target_seconds:.2f}s > 6.2s 硬上限。")

    # Consecutive talking-head check (LaoLi C27d)
    for i in range(1, n_shots):
        prev_s, cur_s = bible.shots[i - 1], bible.shots[i]
        if (
            prev_s.shot_type == "dialogue_lipsync"
            and cur_s.shot_type == "dialogue_lipsync"
            and prev_s.active_speaker == cur_s.active_speaker
            and prev_s.scale == cur_s.scale
            and prev_s.camera_angle == cur_s.camera_angle
        ):
            errors.append(
                f"[C27d_STATIC_MONO] {prev_s.shot_id}->{cur_s.shot_id} 同一角色({cur_s.active_speaker})连续两镜同景别同机位站桩独白，必须切换景别/机位或插入反应/道具特写！"
            )

    # Scale diversity & CU budget (LaoLi C18n)
    unique_scales = {s.scale for s in bible.shots}
    if len(unique_scales) < 4:
        errors.append(f"[C18_SCALE_VAR] 全片景别种类仅 {len(unique_scales)} 种 ({sorted(unique_scales)})，至少需覆盖 4 种（远/全/中/近/特写）。")

    char_shots = [s for s in bible.shots if s.characters_in_shot]
    cu_char_shots = [s for s in char_shots if s.scale in ("CU", "ECU", "MCU")]
    strict_cu_shots = [s for s in bible.shots if s.scale in ("CU", "ECU")]
    cu_ratio = len(strict_cu_shots) / float(n_shots) if n_shots else 0.0
    if n_shots >= 8 and cu_ratio < 0.20:
        errors.append(f"[C18n_LOW_CU] 特写镜头(CU/ECU)占比仅 {cu_ratio*100:.1f}% < 20%，缺乏眼神、道具与情绪重锤特写。")

    non_human_inserts = [s for s in bible.shots if s.shot_type in ("establishing", "insert_prop", "action_impact") or not s.characters_in_shot]
    if len(non_human_inserts) < 2:
        errors.append(f"[C_NO_MONTAGE] 空镜/物件微距特写/动作冲击镜仅 {len(non_human_inserts)} 个，至少需要 2 个以构成影视蒙太奇。")

    reactions = [s for s in bible.shots if s.shot_type == "reaction_cutaway" or "OTS" in s.scale or "过肩" in s.camera_angle]
    if len(bible.characters) >= 2 and len(reactions) < 2:
        errors.append("[C17b_NO_REACTION_OTS] 多人对手戏缺少听者反应镜(reaction_cutaway)与过肩镜头(OTS)，至少需合计 >= 2 镜。")

    flat_count = sum(1 for s in bible.shots if is_conventional_flat_shot(s))
    flat_ratio = flat_count / float(n_shots) if n_shots else 0.0
    if n_shots >= 8 and flat_ratio > 0.38:
        errors.append(
            f"[C_FLAT_CAM] 平视正面常规镜头占比 {flat_ratio*100:.1f}% > 38% 红线，请将更多对话与反应镜改为侧45度、过肩、仰拍或俯拍！"
        )

    # Asset reference integrity
    cmap = bible.char_map()
    lmap = bible.loc_map()
    pmap = bible.prop_map()
    for s in bible.shots:
        if s.location_id not in lmap:
            errors.append(f"[C_BAD_LOC] {s.shot_id} 引用了未登记场景 '{s.location_id}'。")
        for cid in s.characters_in_shot:
            if cid not in cmap:
                errors.append(f"[C_BAD_CHAR] {s.shot_id} 引用了未登记角色 '{cid}'。")
        if s.active_speaker and s.active_speaker not in cmap:
            errors.append(f"[C_BAD_SPEAKER] {s.shot_id} 说话人 '{s.active_speaker}' 未在角色表登记。")
        for pid in s.props_in_shot:
            if pid not in pmap:
                warnings.append(f"[C_UNREG_PROP] {s.shot_id} 引用道具 '{pid}' 未独立建卡，将退化为文本描述。")

    # Source quoted dialogue preservation check (LaoLi `script_feed_diff.py` spirit)
    src_quotes = extract_quoted_dialogues(bible.source_story)
    if src_quotes:
        all_spoken = "".join(normalize_zh_chars(s.spoken_text) for s in bible.shots)
        missing_q = []
        for q in src_quotes:
            # Check 4-gram or substring overlap
            grams = [q[j : j + 4] for j in range(max(1, len(q) - 3))] if len(q) >= 4 else [q]
            hit = sum(1 for g in grams if g in all_spoken)
            if hit / float(len(grams)) < 0.6:
                missing_q.append(q)
        if missing_q:
            errors.append(f"[C_LOST_DIALOGUE] 原文引号台词丢失或被擅删：{missing_q}，必须逐句拆入分镜口播！")

    return Gate0LintResult(
        passed=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        metrics={
            "shot_count": float(n_shots),
            "total_duration_s": round(total_dur, 2),
            "asl_s": round(asl, 2),
            "cu_ratio": round(cu_ratio, 3),
            "flat_camera_ratio": round(flat_ratio, 3),
            "montage_inserts": float(len(non_human_inserts)),
            "reaction_ots_shots": float(len(reactions)),
        },
    )


def _replace_semantic_mentions(text: str, id_to_label: Dict[str, str], fallback_names: Dict[str, str]) -> str:
    """Pure-function @[id] replacement inspired by ArcReel `render_reference_mentions`."""
    def _repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key in id_to_label:
            return id_to_label[key]
        return fallback_names.get(key, key)

    return re.sub(r"@\[([^\]]+)\]", _repl, text or "")


def select_location_plate(loc: LocationSpec, plate_slot: str) -> str:
    if plate_slot in loc.angle_plates and Path(loc.angle_plates[plate_slot]).is_file():
        return loc.angle_plates[plate_slot]
    for fallback_key in ("000_front", "g2_main_zone", "g1_entrance"):
        if fallback_key in loc.angle_plates and Path(loc.angle_plates[fallback_key]).is_file():
            return loc.angle_plates[fallback_key]
    return loc.master_plate_path


def compile_keyframe_prompt_and_refs(
    shot: ShotSpec,
    bible: EpisodeBible,
    prev_tail_frame: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Deterministically assemble <=3 reference images + Qwen-Image-Edit-2511 prompt."""
    cmap = bible.char_map()
    lmap = bible.loc_map()
    pmap = bible.prop_map()

    refs: List[str] = []
    id_to_pic: Dict[str, str] = {}
    fallback_names: Dict[str, str] = {}
    for c in bible.characters:
        fallback_names[c.id] = c.name
        fallback_names[c.name] = c.name
    for l in bible.locations:
        fallback_names[l.id] = l.name
        fallback_names[l.name] = l.name
    for p in bible.props:
        fallback_names[p.id] = p.name
        fallback_names[p.name] = p.name

    loc = lmap.get(shot.location_id) or bible.locations[0]
    loc_plate = select_location_plate(loc, shot.plate_slot)

    # Priority 1: Up to 2 characters in shot (using locked face anchor, or 3q view if wide)
    chars_added: List[CharacterSpec] = []
    for cid in shot.characters_in_shot[:2]:
        c = cmap.get(cid)
        if not c:
            continue
        ref_p = c.face_anchor_path or c.view_paths.get("front", "") or c.portrait_path
        if ref_p and Path(ref_p).is_file() and len(refs) < 2:
            refs.append(ref_p)
            pic_tag = f"Picture {len(refs)}"
            id_to_pic[c.id] = f"the exact person from {pic_tag} ({c.name})"
            id_to_pic[c.name] = id_to_pic[c.id]
            chars_added.append(c)

    # Priority 2: Hero prop card if space remains (< 2 refs used before scene plate)
    props_added: List[PropSpec] = []
    for pid in shot.props_in_shot:
        p = pmap.get(pid)
        if p and p.card_path and Path(p.card_path).is_file() and len(refs) < 2:
            refs.append(p.card_path)
            pic_tag = f"Picture {len(refs)}"
            id_to_pic[p.id] = f"the exact object from {pic_tag} ({p.name})"
            id_to_pic[p.name] = id_to_pic[p.id]
            props_added.append(p)

    # Priority 3: Previous shot tail frame (if carryover and slot free) OR directional location plate
    if prev_tail_frame and Path(prev_tail_frame).is_file() and shot.carryover_prev and len(refs) < 3:
        refs.append(prev_tail_frame)
        loc_pic_tag = f"Picture {len(refs)}"
    elif loc_plate and Path(loc_plate).is_file() and len(refs) < 3:
        refs.append(loc_plate)
        loc_pic_tag = f"Picture {len(refs)}"
    else:
        loc_pic_tag = "the scene"

    id_to_pic[loc.id] = f"{loc_pic_tag}'s {loc.name}"
    id_to_pic[loc.name] = id_to_pic[loc.id]

    scene_body = _replace_semantic_mentions(shot.image_scene_en, id_to_pic, fallback_names)
    lock_clauses: List[str] = []
    for idx_c, c in enumerate(chars_added, start=1):
        lock_clauses.append(
            f"Strictly preserve Picture {idx_c}'s exact facial bone structure, eyes, hairstyle, and signature outfit ({c.costume_spec_en})."
        )
    for p in props_added:
        p_idx = refs.index(p.card_path) + 1
        lock_clauses.append(f"Match Picture {p_idx}'s exact prop design ({p.isolated_desc_en}).")
    if not chars_added and shot.shot_type in ("establishing", "insert_prop"):
        if shot.shot_type == "establishing":
            lock_clauses.append("Empty architectural establishing frame with zero people in frame.")

    prompt = (
        f"{bible.style_phrase_en}. "
        f"Shot scale: {shot.scale}, {shot.lens_mm}mm lens, camera angle: {shot.camera_angle} inside {loc_pic_tag}. "
        f"{scene_body} "
        f"Lighting: {shot.lighting_en or loc.time_weather_en}. "
        f"{' '.join(lock_clauses)} "
        "Zero text, zero subtitles, zero watermarks, zero split-screen borders."
    )
    return re.sub(r"\s+", " ", prompt).strip(), refs


def compile_h3_video_prompt_and_refs(
    shot: ShotSpec,
    bible: EpisodeBible,
    keyframe_path: str,
    prev_tail_frame: Optional[str] = None,
) -> Tuple[str, List[str], Optional[str], str]:
    """Deterministically compile MiniMax-H3 (10Eros-Max) multimodal prompt, image refs, audio ref, and ref_image_size."""
    cmap = bible.char_map()
    lmap = bible.loc_map()
    pmap = bible.prop_map()

    refs: List[str] = [keyframe_path]
    notes: List[str] = [
        f"- <Picture 1>: Approved opening {shot.scale} ({shot.lens_mm}mm, {shot.camera_angle}) keyframe establishing exact framing, lighting, and initial pose."
    ]
    id_to_label: Dict[str, str] = {}
    fallback_names: Dict[str, str] = {c.id: c.name for c in bible.characters}
    fallback_names.update({l.id: l.name for l in bible.locations})
    fallback_names.update({p.id: p.name for p in bible.props})

    speaker_spec: Optional[CharacterSpec] = cmap.get(shot.active_speaker) if shot.active_speaker else None
    if speaker_spec:
        # Active speaker always gets (S1) binding + high-res face anchor + empty 3q costume card
        if speaker_spec.face_anchor_path and Path(speaker_spec.face_anchor_path).is_file():
            refs.append(speaker_spec.face_anchor_path)
            notes.append(f"- <Picture {len(refs)}>: High-resolution facial identity lock for speaking character (S1) {speaker_spec.name}.")
        v3q = speaker_spec.view_paths.get("3q") or speaker_spec.view_paths.get("front")
        if v3q and Path(v3q).is_file() and shot.scale not in ("ECU", "CU"):
            refs.append(v3q)
            notes.append(f"- <Picture {len(refs)}>: Empty-handed costume and proportion reference for (S1) {speaker_spec.name}.")
        id_to_label[speaker_spec.id] = f"character (S1) {speaker_spec.name}"
        id_to_label[speaker_spec.name] = id_to_label[speaker_spec.id]

    for cid in shot.characters_in_shot:
        c = cmap.get(cid)
        if not c or (speaker_spec and c.id == speaker_spec.id):
            continue
        if c.face_anchor_path and Path(c.face_anchor_path).is_file() and len(refs) < 5:
            refs.append(c.face_anchor_path)
            p_idx = len(refs)
            notes.append(f"- <Picture {p_idx}>: Facial identity anchor for co-present character {c.name}.")
            id_to_label[c.id] = f"{c.name} (matching <Picture {p_idx}>)"
            id_to_label[c.name] = id_to_label[c.id]

    for pid in shot.props_in_shot:
        p = pmap.get(pid)
        if p and p.card_path and Path(p.card_path).is_file() and len(refs) < 6:
            refs.append(p.card_path)
            p_idx = len(refs)
            notes.append(f"- <Picture {p_idx}>: Isolated prop reference card for {p.name}.")
            id_to_label[p.id] = f"<Picture {p_idx}>'s {p.name}"
            id_to_label[p.name] = id_to_label[p.id]

    loc = lmap.get(shot.location_id) or bible.locations[0]
    loc_plate = select_location_plate(loc, shot.plate_slot)
    if loc_plate and Path(loc_plate).is_file() and len(refs) < 6 and shot.scale not in ("ECU",):
        refs.append(loc_plate)
        p_idx = len(refs)
        notes.append(f"- <Picture {p_idx}>: Locked directional architectural set plate for {loc.name}.")
        id_to_label[loc.id] = f"<Picture {p_idx}>"
        id_to_label[loc.name] = f"<Picture {p_idx}>"

    if prev_tail_frame and Path(prev_tail_frame).is_file() and shot.carryover_prev and len(refs) < 6:
        refs.append(prev_tail_frame)
        notes.append(f"- <Picture {len(refs)}>: Immediate preceding shot tail frame locking color temperature and continuity.")

    audio_ref: Optional[str] = None
    if shot.shot_type == "dialogue_lipsync" and speaker_spec and shot.audio_32k_path and Path(shot.audio_32k_path).is_file():
        audio_ref = shot.audio_32k_path
        notes.append(
            f"- <Audio 1>: Cloned Mandarin speech track driving exact voice timbre, cadence, and syllable-accurate lip sync for (S1) {speaker_spec.name} ONLY."
        )

    motion_delta = _replace_semantic_mentions(shot.video_action_delta_en, id_to_label, fallback_names)
    dur = shot.target_seconds
    if audio_ref and speaker_spec:
        speech_clause = (
            f"SOUND LOCK: Use <Audio 1> identically from 0.00s to {dur:.2f}s without re-timing; "
            f"(S1) speaks in Mandarin Chinese with crisp lip and jaw synchronization matching <Audio 1>: \"(S1) {shot.spoken_text}\"."
        )
    else:
        speech_clause = "No spoken dialogue in this shot; keep characters' mouths naturally closed or reacting silently without speech."

    anti_clone = (
        "No extra background people; zero people in frame."
        if not shot.characters_in_shot
        else f"Exactly {len(shot.characters_in_shot)} person(s) in frame; no duplicate or hallucinated extra figures."
    )
    integrated = (
        f"At 0.00 seconds, <Picture 1> is fully integrated as the opening frame ({shot.scale}, {shot.lens_mm}mm, {shot.camera_angle}). "
        f"From 0.00s to {dur:.2f}s, the camera executes {shot.camera_motion}. "
        f"{motion_delta} {speech_clause} {anti_clone} "
        "No subtitles, no on-screen text, no logos, no watermarks."
    )

    soundscape = shot.soundscape_foley_en or (
        "Close-mic crisp Mandarin vocal delivery by (S1) with subtle clothing rustle and natural room tone."
        if audio_ref else "Crisp physical foley sound effects synchronized to motion with subtle room ambience, zero speech."
    )
    music = shot.music_cue_en or "Subtle cinematic dramatic-comedy underscore without overpowering foley or speech."

    full_prompt = (
        "[Reference Notes]\n"
        + "\n".join(notes)
        + "\n\n[integrated_multimodal_description]\n"
        + re.sub(r"\s+", " ", integrated).strip()
        + "\n\n[overall_soundscape]\n"
        + soundscape.strip()
        + "\n\n[non_diegetic_music]\n"
        + music.strip()
    )

    # Use `max` (2048px) when multiple characters or medium/wide facial preservation is needed; `match` for pure inserts/empty shots
    ref_size = "match" if (not shot.characters_in_shot or shot.scale == "ECU" and shot.shot_type == "insert_prop") else "max"
    return full_prompt, refs, audio_ref, ref_size
