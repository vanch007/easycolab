"""Stages 1–3 of EasyColab Cinema-Grade Short-Drama Pipeline:
- Stage 1: VLM Face Pre-Analysis + Two-Phase Qwen3.8-27B Showrunner & Director Breakdown + Gate 0 Auto-Repair.
- Stage 2: Dual-Track Character Factory (User Face -> Lock & 4-View vs Zero-Face -> T2I Face + 4-View),
           Scene Master + 3x3 9-Grid Multi-Angle Slicer, Isolated Prop Cards + Gate 1 VLM QA.
- Stage 3: Qwen3-TTS 1.7B Human Voice Seed Casting (Gate 2) + Fish Audio Drama 3 35B Clause Dubbing (Gate 3).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .cloud_client import EasyColabCloudClient
from .models import (
    CharacterSpec,
    EmotionBeat,
    EpisodeBible,
    LocationSpec,
    ProductionConfig,
    PropSpec,
    ShotSpec,
)
from .qa_engine import compute_audio_prosody_stats, verify_spoken_audio, vlm_inspect_image_asset
from .rules import (
    SCENE_GRID_9_MATRICES,
    lint_episode_bible,
    normalize_and_enrich_bible,
    snap_seconds_to_h3_grid,
)


def _crop_face_anchor_from_image(src_img: Path, dst_face: Path) -> Path:
    """Create a clean upper-center square portrait crop focusing on head, hair, glasses & collar."""
    img = Image.open(src_img).convert("RGB")
    w, h = img.size
    side = int(round(min(w, h * 0.72)))
    side = max(320, min(w, h, side))
    cx = w // 2
    left = max(0, min(w - side, cx - side // 2))
    top = max(0, min(h - side, int(round(h * 0.03))))
    crop = img.crop((left, top, left + side, top + side))
    crop = crop.resize((768, 768), Image.Resampling.LANCZOS)
    dst_face.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst_face, format="PNG")
    return dst_face


def _slice_grid_image(sheet_path: Path, rows: int, cols: int) -> List[Image.Image]:
    img = Image.open(sheet_path).convert("RGB")
    w, h = img.size
    cw = w // cols
    ch = h // rows
    cells: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * cw
            y0 = r * ch
            x1 = (c + 1) * cw if c < cols - 1 else w
            y1 = (r + 1) * ch if r < rows - 1 else h
            # Trim 1.5% inner border to remove thin white grid lines cleanly
            mx = int((x1 - x0) * 0.015)
            my = int((y1 - y0) * 0.015)
            cells.append(img.crop((x0 + mx, y0 + my, x1 - mx, y1 - my)))
    return cells


def _analyze_user_supplied_faces(
    cloud: EasyColabCloudClient, char_faces: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    """Inspect user-uploaded character reference photos with Qwen3.8-27B-VL so Stage 1 locks exact real traits."""
    notes: Dict[str, Dict[str, str]] = {}
    for role_key, raw_path in char_faces.items():
        p = Path(raw_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"User-supplied --char-face for '{role_key}' not found: {p}")
        print(f"[Stage 1.0] VLM 解析用户输入角色参考图: {role_key} -> {p.name}", flush=True)
        res = cloud.chat_json(
            system_prompt=(
                "You are a Film Casting & Wardrobe Continuity Supervisor. Analyze Image 1 and output ONLY a JSON object: "
                '{"gender": "female/male", "age_range": "e.g. 26-30", "face_10_locks_en": "detailed English description of face shape, eyebrows, eyes, glasses/earrings/hairpin, nose, lips, hairstyle & hair color", '
                '"visible_costume_en": "detailed English description of visible clothing/collar/color/pattern, ignoring any hand-held props"}.'
            ),
            user_prompt=f"Character Name/Role: {role_key}. Extract exact facial & wardrobe visual locks from Image 1.",
            images=[p],
            temperature=0.15,
            max_tokens=800,
        )
        notes[role_key] = {
            "path": str(p),
            "gender": str(res.get("gender") or ""),
            "age_range": str(res.get("age_range") or ""),
            "face_10_locks_en": str(res.get("face_10_locks_en") or ""),
            "visible_costume_en": str(res.get("visible_costume_en") or ""),
        }
    return notes


STAGE1_SYSTEM_PROMPT = """You are an elite Cinema Short-Drama Showrunner & Director combining:
1) `manju-laoli-skill` Camera & Acting Grammar:
   - Target 14 to 18 shots for a ~45-55s short drama (Average Shot Length 2.5s–3.6s). NEVER output 5-8 long static talking-head shots!
   - Conventional eye-level frontal shots MUST be <= 30%. Default dialogue camera is `侧45度` (45-deg three-quarter), `过肩` (OTS with explicit foreground shoulder side), `仰拍` (low-angle power), or `俯拍` (high-angle vulnerability).
   - Close-ups (`CU` / `ECU`) MUST account for 25%–40% of shots (eye-glint memory beats, key prop inserts, spoken accent peaks, listener reactions).
   - Include >= 2 non-dialogue visual montage shots (`establishing` empty sunlit room/courtyard, `insert_prop` macro prop interaction, `action_impact` doorway burst / object crash) AND >= 2 `reaction_cutaway` or `OTS` shots!
   - 3-Second Anti-Talking-Head Law: Any long spoken sentence (> 18 Chinese characters or >= 2 clauses) MUST be split at natural clause boundaries across 2+ shots (each <= 20 Chinese characters, 2.3s–5.2s), or interleaved with a reaction/prop cutaway!
   - Every `dialogue_lipsync` shot MUST preserve ALL original story dialogue meaning/lines without omission, and specify `accent_word` (1 stressed Chinese word), `acting_expression` (concrete facial muscle/gaze change in English), and `acting_gesture` (concrete hand/posture motion in English).
2) `ArcReel` Two-Layer Visual Authoring & Asset Mention Syntax:
   - Register clean `characters` (id: lowercase ascii slug like `vivi`), `locations` (id like `loc_study`), and hero interactive `props` (id like `prop_pointer` — decoupled from character bodies!).
   - Inside `image_scene_en` and `video_action_delta_en`, ALWAYS reference registered assets via `@[id]` (e.g., `@[vivi]`, `@[loc_study]`, `@[prop_pointer]`). NEVER write `Picture 1` or `图1` manually!
   - `image_scene_en` (Photographer view): static single opening frame composition, character placement, gaze, and visible prop state.
   - `video_action_delta_en` (Action Designer view): ONLY physical observable motion, gesture trajectory, micro-expression shift, and prop interaction during this shot (NO static appearance repetition, NO edit trigger verbs like 'replace/modify').
   - Voice casting (`voice_instruct_zh`): hyper-specific human vocal anatomy (age, vocal fold texture, breathiness/chest resonance, articulation habit, studio close-mic realism — ZERO robotic broadcaster tone!).
   - Drama 3 tag (`drama3_acting_tag`): concise English bracket cue + optional inline `[sighs]`/`[chuckles]`/`[gasps]` matching the character's timbre and line emotion.

Output ONLY ONE valid JSON object matching the exact schema requested."""


def _build_stage1_user_prompt(
    cfg: ProductionConfig,
    vlm_face_notes: Dict[str, Dict[str, str]],
    lint_feedback: Optional[List[str]] = None,
) -> str:
    min_shots = max(14, int(round(cfg.target_duration_s / 3.3)))
    face_info_txt = (
        json.dumps(vlm_face_notes, ensure_ascii=False, indent=2)
        if vlm_face_notes
        else "None supplied (design photorealistic distinct faces for all characters from story context)."
    )
    feedback_block = ""
    if lint_feedback:
        feedback_block = (
            "\n\n[CRITICAL GATE 0 LINT VIOLATIONS TO FIX IMMEDIATELY]:\n"
            + "\n".join(f"- {e}" for e in lint_feedback)
            + "\nFix EVERY violation above while keeping all story plot beats and original quoted lines intact!\n"
        )

    return f"""Convert the following story into a complete Cinema-Grade Short-Drama Production Bible JSON.

[INPUT STORY]:
{cfg.story_text}

[TARGET SPECS]:
- Aspect Ratio: {cfg.aspect_ratio}
- Visual Base: {cfg.visual_base}
- Target Total Duration: ~{cfg.target_duration_s:.0f} seconds
- Target Shot Count: {min_shots} to {min_shots + 4} shots (ASL ~2.6s–3.5s)
- User-Supplied Character Reference Face Analysis (MUST preserve their exact facial & wardrobe traits if listed):
{face_info_txt}
{feedback_block}
[REQUIRED JSON SCHEMA]:
{{
  "title": "Chinese Episode Title",
  "synopsis": "1-2 sentence Chinese summary",
  "style_phrase_en": "30-word English film cinematography & color palette lock",
  "characters": [
    {{
      "id": "lowercase_slug",
      "name": "Chinese Name matching story/user key",
      "aliases": ["other names in story"],
      "gender": "female or male",
      "age_range": "26-30",
      "body_ratio": "7.0-head realistic East Asian proportion, height & build",
      "role_badge_accessory": "worn signature accessory (e.g. gold-rimmed glasses, gold hairpin — NOT hand-held props)",
      "costume_spec_en": "Detailed English empty-handed outfit, fabric, collar, color, shoes",
      "posture_habit_en": "Characteristic posture",
      "anti_drift_banlist_en": "What NOT to draw on this character (e.g. no hand-held pointer, no clipboard, no modern hoodie)",
      "face_10_locks_en": "Detailed English 10-feature East Asian facial bone & skin lock",
      "voice_instruct_zh": "具象化真人声线描述（年龄、声带颗粒感/气声/胸腔共鸣、说话习惯、近讲棚录感，40-70字中文）",
      "voice_desc_en": "2-sentence English voice timbre & delivery profile",
      "voice_label_en": "<=7 words English timbre tag",
      "audition_line_zh": "18-26字贴合人设但情绪平稳自然的中文试音台词（用于铸造干净声纹种子）"
    }}
  ],
  "locations": [
    {{
      "id": "loc_slug",
      "name": "Chinese Scene Name",
      "space_type": "indoor_semi | outdoor_building | natural_street",
      "era_style_en": "Detailed English architectural style & materials",
      "time_weather_en": "Specific light direction, color temperature (K), sunbeams/weather",
      "master_desc_en": "60-word English empty-set establishing architectural description (zero people)",
      "four_walls_en": {{
        "000_front": "Main front focal wall/furniture",
        "090_right": "Right 90-deg side windows/wall",
        "180_reverse": "Reverse 180-deg doorway/entrance looking toward exterior",
        "270_left": "Left 270-deg opposite wall/screens"
      }}
    }}
  ],
  "props": [
    {{
      "id": "prop_slug",
      "name": "Chinese Prop Name",
      "isolated_desc_en": "40-word English isolated studio product photo description (material, shape, color, zero hands/people)"
    }}
  ],
  "beats": [
    {{
      "beat_id": "B01",
      "emotion_keyword": "情绪词",
      "story_action": "角色在做什么",
      "context_eid": "E01..E15",
      "core_camera_lock": "侧45度 · 中近景 · 50mm"
    }}
  ],
  "shots": [
    {{
      "shot_id": "s01",
      "index": 1,
      "beat_id": "B01",
      "shot_type": "establishing | insert_prop | dialogue_lipsync | reaction_cutaway | action_impact",
      "scale": "EWS | WS | MWS | MS | OTS | MCU | CU | ECU",
      "lens_mm": 50,
      "camera_angle": "e.g. 侧45度+微仰 / 后肩过右肩 / 大俯角 / 微距贴地",
      "camera_motion": "single smooth physical camera move in English",
      "location_id": "loc_slug",
      "plate_slot": "000_front | 090_right | 180_reverse | 270_left",
      "characters_in_shot": ["char_id"],
      "active_speaker": "char_id or null",
      "props_in_shot": ["prop_id"],
      "spoken_text": "Exact Chinese clause (<=20 chars) for dialogue_lipsync, or empty string",
      "drama3_acting_tag": "[ husky alto Chinese actress late 20s, dry sarcasm ] [sighs]",
      "accent_word": "单个重音词或空字符串",
      "acting_expression": "Concrete visible facial/eye/breathing muscle action in English",
      "acting_gesture": "Concrete visible hand/weight/object interaction in English",
      "state_before": "进镜姿态与持物",
      "state_after": "出镜姿态与持物",
      "image_scene_en": "35-60 words English static opening frame using @[id] mentions ONLY",
      "lighting_en": "Specific light source, angle, Kelvin warmth, rim light",
      "video_action_delta_en": "35-60 words English observable motion & micro-expression arc using @[id] mentions ONLY",
      "soundscape_foley_en": "Specific physical foley SFX + room tone in English",
      "music_cue_en": "Specific instrument/mood underscore cue in English",
      "target_seconds": 3.0,
      "cut_in": "hard | jcut",
      "carryover_prev": false
    }}
  ]
}}"""


def run_stage1_showrunner(cfg: ProductionConfig, cloud: EasyColabCloudClient) -> EpisodeBible:
    """Execute VLM face pre-analysis + Qwen3.8-27B Showrunner with Gate 0 auto-repair."""
    bible_path = cfg.project_dir / "episode_bible.json"
    if bible_path.is_file():
        existing = EpisodeBible.from_dict(json.loads(bible_path.read_text(encoding="utf-8")))
        chk = lint_episode_bible(existing, cfg.target_duration_s)
        if chk.passed:
            print(f"[Stage 1] 复用已锁定通过 Gate 0 的剧本分镜圣经 ({len(existing.shots)} 镜, ASL={chk.metrics['asl_s']}s)。", flush=True)
            return existing

    vlm_notes = _analyze_user_supplied_faces(cloud, cfg.char_faces)
    lint_errors: Optional[List[str]] = None
    best_bible: Optional[EpisodeBible] = None

    for attempt in range(cfg.max_qa_retries + 2):
        print(f"[Stage 1] 正在调用 Qwen3.8-27B 编剧与导演决策引擎 (Attempt {attempt + 1})...", flush=True)
        raw_obj = cloud.chat_json(
            system_prompt=STAGE1_SYSTEM_PROMPT,
            user_prompt=_build_stage1_user_prompt(cfg, vlm_notes, lint_feedback=lint_errors),
            temperature=0.4 if attempt == 0 else 0.25,
            max_tokens=12288,
        )
        raw_obj["project_name"] = cfg.project_name
        raw_obj["aspect_ratio"] = cfg.aspect_ratio
        raw_obj["visual_base"] = cfg.visual_base
        raw_obj["source_story"] = cfg.story_text
        if cfg.style_phrase_en:
            raw_obj["style_phrase_en"] = cfg.style_phrase_en

        bible = EpisodeBible.from_dict(raw_obj)
        # Attach user-supplied face & voice paths by matching id, name, alias, or substring
        for c in bible.characters:
            keys_to_try = [c.id, c.name, *c.aliases]
            for uk, uinfo in vlm_notes.items():
                if any(k and (k == uk or k in uk or uk in k) for k in keys_to_try):
                    c.user_face_path = uinfo["path"]
                    break
            for uk, upath in cfg.char_voices.items():
                if any(k and (k == uk or k in uk or uk in k) for k in keys_to_try):
                    c.user_voice_path = str(Path(upath).expanduser().resolve())
                    break

        bible = normalize_and_enrich_bible(bible, cfg.target_duration_s)
        res = lint_episode_bible(bible, cfg.target_duration_s)
        best_bible = bible
        print(
            f"[Gate 0 导演语法机检] passed={res.passed} | shots={int(res.metrics.get('shot_count',0))} "
            f"| ASL={res.metrics.get('asl_s')}s | CU={res.metrics.get('cu_ratio',0)*100:.1f}% "
            f"| flat_cam={res.metrics.get('flat_camera_ratio',0)*100:.1f}% | errors={len(res.errors)}",
            flush=True,
        )
        if res.passed:
            break
        for err in res.errors:
            print(f"  -> Gate 0 Error: {err}", flush=True)
        lint_errors = res.errors

    assert best_bible is not None
    bible_path.write_text(json.dumps(best_bible.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (cfg.project_dir / "gate0_lint_report.json").write_text(
        json.dumps(lint_episode_bible(best_bible, cfg.target_duration_s).__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return best_bible


def run_stage2_asset_factory(
    cfg: ProductionConfig, bible: EpisodeBible, cloud: EasyColabCloudClient
) -> EpisodeBible:
    """Generate/lock all Character Cards (with or without user face), 3x3 Scene Grids, and Prop Cards + Gate 1 VLM QA."""
    cloud.ensure_comfy_packs(["moss_image_edit", "qwen_image_edit_2511"])
    char_dir = cfg.project_dir / "assets" / "characters"
    loc_dir = cfg.project_dir / "assets" / "locations"
    prop_dir = cfg.project_dir / "assets" / "props"
    for d in (char_dir, loc_dir, prop_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Characters (Dual-Track: User Face Supplied vs Zero-Face Auto-Generated)
    for idx_c, c in enumerate(bible.characters):
        if c.qa_passed and c.face_anchor_path and Path(c.face_anchor_path).is_file() and c.multiview_sheet_path and Path(c.multiview_sheet_path).is_file():
            continue

        face_anchor = char_dir / f"{c.id}_face.png"
        portrait_png = char_dir / f"{c.id}_portrait.png"
        sheet_png = char_dir / f"{c.id}_multiview.png"

        if c.user_face_path and Path(c.user_face_path).is_file():
            src_ref = Path(c.user_face_path)
            shutil.copy2(src_ref, portrait_png)
            _crop_face_anchor_from_image(src_ref, face_anchor)
            print(f"[Stage 2A · 角色锁脸] {c.name} ({c.id}): 使用输入人脸图 -> {face_anchor.name}", flush=True)
        else:
            # Zero-Face Mode: Generate photorealistic/stylized hero portrait first via MOSS T2I + Gate 1 VLM check
            for att in range(cfg.max_qa_retries + 1):
                p_seed = cfg.seed_base + 100 + idx_c * 37 + att
                portrait_prompt = (
                    f"{bible.style_phrase_en}. Bust-up studio portrait photograph of an East Asian {c.gender} aged {c.age_range} ({c.name}). "
                    f"Facial features: {c.face_10_locks_en}. Worn accessories: {c.role_badge_accessory or 'none'}. "
                    f"Wardrobe collar & shoulders: {c.costume_spec_en}. "
                    "Natural matte skin texture with fine pores, crisp catchlights in eyes, neutral light-gray studio backdrop, "
                    "centered head-and-shoulders framing, zero hands, zero props, zero text."
                )
                print(f"[Stage 2A · 零参考生脸] 正在为 {c.name} ({c.id}) 生成高清人脸母图 (attempt {att+1})...", flush=True)
                cloud.run_moss_image(portrait_prompt, portrait_png, width=1024, height=1024, steps=28, seed=p_seed)
                v_chk = vlm_inspect_image_asset(
                    cloud, "character_portrait", portrait_png,
                    f"Clear single {c.gender} portrait ({c.age_range}) matching {c.face_10_locks_en}, zero extra people, zero text."
                )
                if v_chk["passed"]:
                    break
            _crop_face_anchor_from_image(portrait_png, face_anchor)
            src_ref = portrait_png

        c.portrait_path = str(portrait_png)
        c.face_anchor_path = str(face_anchor)

        # Generate Empty-Handed 4-View Character Turnaround Sheet (`front`, `3q`, `side`, `back`)
        for att in range(cfg.max_qa_retries + 1):
            s_seed = cfg.seed_base + 200 + idx_c * 41 + att
            mv_prompt = (
                f"Live-action costume turnaround reference sheet of the exact same East Asian {c.gender} from Picture 1 ({c.body_ratio}), "
                "re-posed COMPLETELY EMPTY-HANDED with both hands empty and hanging relaxed by the sides, holding nothing at all. "
                "Exactly 4 evenly spaced full-body standing views side-by-side from left to right on a clean seamless light-gray studio backdrop, head-to-shoes fully visible: "
                "1) Front full-body view; 2) 45-degree three-quarter view; 3) 90-degree side profile view; 4) Full back view. "
                f"Consistent appearance across all 4 views: {c.face_10_locks_en}, {c.role_badge_accessory}, {c.costume_spec_en}. "
                f"Forbidden: {c.anti_drift_banlist_en}, zero hand-held props, zero text, zero labels, zero 5th panel."
            )
            print(f"[Stage 2A · 角色四视图] 生成 {c.name} 空手四视图设定卡 (attempt {att+1})...", flush=True)
            cloud.run_moss_image(mv_prompt, sheet_png, width=1536, height=864, steps=28, seed=s_seed, ref_image=src_ref)
            v_chk = vlm_inspect_image_asset(
                cloud, "character_4view_sheet", sheet_png,
                f"Full-body turnaround sheet of {c.name} wearing {c.costume_spec_en}, empty-handed, clean studio background.",
                ref_face=face_anchor,
            )
            print(f"  [Gate 1 VLM 角色卡核验] {c.id}: score={v_chk['score']} passed={v_chk['passed']}", flush=True)
            if v_chk["passed"]:
                break

        c.multiview_sheet_path = str(sheet_png)
        views = _slice_grid_image(sheet_png, rows=1, cols=4)
        for v_name, v_img in zip(("front", "3q", "side", "back"), views):
            vp = char_dir / f"{c.id}_{v_name}.png"
            v_img.save(vp, format="PNG")
            c.view_paths[v_name] = str(vp)
        c.qa_passed = True

    # 2. Isolated Prop Cards (Decoupled from characters)
    for idx_p, prop in enumerate(bible.props):
        if prop.qa_passed and prop.card_path and Path(prop.card_path).is_file():
            continue
        card_png = prop_dir / f"{prop.id}.png"
        for att in range(cfg.max_qa_retries + 1):
            p_seed = cfg.seed_base + 400 + idx_p * 29 + att
            p_prompt = (
                f"Isolated studio prop reference photograph of {prop.isolated_desc_en}, "
                "centered on a clean seamless neutral light-gray studio surface, crisp studio lighting, "
                "realistic material textures, zero people, zero hands, zero text, zero watermarks."
            )
            print(f"[Stage 2B · 独立道具卡] 生成 {prop.name} ({prop.id}) (attempt {att+1})...", flush=True)
            cloud.run_moss_image(p_prompt, card_png, width=1024, height=768, steps=28, seed=p_seed)
            v_chk = vlm_inspect_image_asset(
                cloud, "prop_card", card_png,
                f"Isolated prop object ({prop.isolated_desc_en}) with zero people and zero hands."
            )
            print(f"  [Gate 1 VLM 道具卡核验] {prop.id}: score={v_chk['score']} passed={v_chk['passed']}", flush=True)
            if v_chk["passed"]:
                break
        prop.card_path = str(card_png)
        prop.qa_passed = True

    # 3. Locations: Master Empty Plate + LaoLi 3x3 9-Grid Multi-Angle Sheet + 4 Directional Hero Plates
    w_res, h_res = bible.resolution()
    for idx_l, loc in enumerate(bible.locations):
        if loc.qa_passed and loc.master_plate_path and Path(loc.master_plate_path).is_file() and len(loc.angle_plates) >= 4:
            continue

        master_png = loc_dir / f"{loc.id}_000_front.png"
        grid9_png = loc_dir / f"{loc.id}_grid9.png"
        for att in range(cfg.max_qa_retries + 1):
            l_seed = cfg.seed_base + 600 + idx_l * 53 + att
            m_prompt = (
                f"{bible.style_phrase_en}. Empty architectural establishing set plate of {loc.name} ({loc.era_style_en}). "
                f"{loc.master_desc_en}. Front focal wall: {loc.four_walls_en.get('000_front', '')}. "
                f"Lighting: {loc.time_weather_en}. Completely empty set with ZERO people, zero human silhouettes, zero text."
            )
            print(f"[Stage 2C · 场景母版] 生成 {loc.name} 空镜母版 (attempt {att+1})...", flush=True)
            cloud.run_moss_image(m_prompt, master_png, width=w_res, height=h_res, steps=28, seed=l_seed)
            v_chk = vlm_inspect_image_asset(
                cloud, "empty_scene_plate", master_png,
                f"Empty architectural set of {loc.name} ({loc.master_desc_en}) with ZERO people in frame."
            )
            print(f"  [Gate 1 VLM 场景母版核验] {loc.id}: score={v_chk['score']} passed={v_chk['passed']}", flush=True)
            if v_chk["passed"]:
                break

        loc.master_plate_path = str(master_png)
        loc.angle_plates["000_front"] = str(master_png)

        # Free MOSS-Image from ComfyUI VRAM before loading Qwen-Image-Edit-2511 alongside Qwen3.8-27B
        cloud.free_comfy_vram()

        # Generate 3x3 9-Angle Scene Grid Sheet (LaoLi `scene-grid-template.md`) + Slice into 9 sub-plates
        matrix = SCENE_GRID_9_MATRICES.get(loc.space_type, SCENE_GRID_9_MATRICES["indoor_semi"])
        cell_lines = "; ".join(f"Panel {i+1}: {desc}" for i, (_, desc) in enumerate(matrix))
        grid_prompt = (
            f"16:9 multi-camera architectural contact sheet containing EXACTLY 9 equal 3x3 panels of Picture 1's exact {loc.name} ({loc.era_style_en}), "
            "preserving identical architecture, wood/stone materials, furniture, and sunlight direction across all 9 panels, camera movement ONLY, ZERO people: "
            f"{cell_lines}. Zero text, zero labels, zero people."
        )
        print(f"[Stage 2C · 场景九宫格] 生成 {loc.name} 3x3 九机位场景矩阵并自动切分...", flush=True)
        cloud.run_qwen_edit(
            grid_prompt, [master_png], grid9_png,
            width=1920, height=1088, steps=4, seed=cfg.seed_base + 700 + idx_l,
        )
        loc.grid9_sheet_path = str(grid9_png)
        cells = _slice_grid_image(grid9_png, rows=3, cols=3)
        for (slot_key, _), cell_img in zip(matrix, cells):
            sp = loc_dir / f"{loc.id}_{slot_key}.png"
            cell_img.resize((w_res, h_res), Image.Resampling.LANCZOS).save(sp, format="PNG")
            loc.angle_plates[slot_key] = str(sp)

        # Also synthesize crisp full-res directional hero plates (090_right, 180_reverse, 270_left) via Qwen-Edit-2511
        dir_specs = [
            ("090_right", loc.four_walls_en.get("090_right", "90-degree side wall and sunlit windows")),
            ("180_reverse", loc.four_walls_en.get("180_reverse", "180-degree reverse view facing the main entrance doorway looking toward the exterior")),
            ("270_left", loc.four_walls_en.get("270_left", "270-degree opposite side wall and architectural decor")),
        ]
        for d_idx, (d_slot, d_desc) in enumerate(dir_specs):
            dp = loc_dir / f"{loc.id}_{d_slot}.png"
            d_prompt = (
                f"{bible.style_phrase_en}. Empty architectural camera rotation inside Picture 1's exact {loc.name}: "
                f"show the {d_slot} perspective ({d_desc}), keeping identical materials, flooring, and {loc.time_weather_en}. "
                "Completely empty room with ZERO people, zero text."
            )
            cloud.run_qwen_edit(
                d_prompt, [master_png], dp,
                width=w_res, height=h_res, steps=4, seed=cfg.seed_base + 750 + idx_l * 10 + d_idx,
            )
            loc.angle_plates[d_slot] = str(dp)

        loc.qa_passed = True

    (cfg.project_dir / "episode_bible.json").write_text(
        json.dumps(bible.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bible


def _prepare_h3_32k_clause_wav(src_audio: Path, dst_wav: Path, pre_pad_s: float = 0.14) -> tuple[int, float, float]:
    """Convert Drama 3 clause audio to 32kHz stereo WAV with pre-breath pad & trailing pad snapped to 17k+5 H3 grid."""
    stats = compute_audio_prosody_stats(src_audio)
    raw_dur = stats["duration_s"]
    needed_s = max(2.33, raw_dur + pre_pad_s + 0.28)
    frames, h3_sec = snap_seconds_to_h3_grid(needed_s, min_frames=56, max_frames=175)
    pad_tail = max(0.05, round(h3_sec - (raw_dur + pre_pad_s), 3))
    ms = int(round(pre_pad_s * 1000))
    af = f"adelay={ms}|{ms},apad=pad_dur={pad_tail:.3f},atrim=0:{h3_sec:.3f}"
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src_audio), "-af", af, "-ar", "32000", "-ac", "2", "-c:a", "pcm_s16le", str(dst_wav)],
        check=True,
    )
    return frames, h3_sec, round(raw_dur, 2)


def run_stage3_voice_and_dubbing(
    cfg: ProductionConfig, bible: EpisodeBible, cloud: EasyColabCloudClient
) -> EpisodeBible:
    """Forge 1 human-realistic voice seed per character (Gate 2) + per-shot Drama 3 clause dubbing (Gate 3)."""
    # Free LLM VRAM before loading 44GB Drama 3 MoE + Qwen3-TTS + Qwen3-ASR
    cloud.unload_llm()
    cloud.free_comfy_vram()
    cloud.ensure_comfy_packs(["qwen3_tts_voice_design", "drama3_tts"])

    voice_dir = cfg.project_dir / "assets" / "voices"
    audio_dir = cfg.project_dir / "shots" / "audio"
    voice_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Gate 2: Character Voice Seed Casting (1x per character, reused across all shots)
    for idx_c, c in enumerate(bible.characters):
        if c.voice_ref_path and Path(c.voice_ref_path).is_file():
            continue
        ref_flac = voice_dir / f"{c.id}_ref.flac"
        if c.user_voice_path and Path(c.user_voice_path).is_file():
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", c.user_voice_path, "-ar", "44100", "-ac", "1", str(ref_flac)],
                check=True,
            )
            c.voice_ref_path = str(ref_flac)
            print(f"[Stage 3A · 声纹锁定] {c.name}: 使用外部传入声纹种子 -> {ref_flac.name}", flush=True)
            continue

        audition_txt = c.audition_line_zh.strip() or f"今天这桩差事按原计划推进，大家都把精神打起来。"
        instruct_txt = c.voice_instruct_zh.strip() or (
            "28岁青年演员真实棚录对白音色，声带质感清晰自然，有呼吸气口与情绪层次，拒绝播音腔"
        )
        for att in range(cfg.max_qa_retries + 1):
            v_seed = cfg.seed_base + 900 + idx_c * 31 + att
            print(f"[Stage 3A · 声音设计] Qwen3-TTS 铸造 {c.name} ({c.id}) 专属声纹种子 (attempt {att+1})...", flush=True)
            cloud.run_qwen3_tts_design(instruct_txt, audition_txt, ref_flac, seed=v_seed)
            qa = verify_spoken_audio(cloud, ref_flac, audition_txt, min_similarity=0.82, min_f0_std=14.0, max_duration_s=14.0)
            print(
                f"  [Gate 2 ASR+韵律核验] {c.id}: sim={qa['similarity']} f0={qa['median_f0_hz']}Hz(±{qa['f0_std_hz']}) dur={qa['duration_s']}s passed={qa['passed']}",
                flush=True,
            )
            if qa["passed"]:
                break
        c.voice_ref_path = str(ref_flac)

    # Gate 3: Per-Shot Clause Emotional Dubbing via Fish Audio Drama 3 35B
    cmap = bible.char_map()
    gate3_log: Dict[str, Any] = {}
    for idx_s, shot in enumerate(bible.shots):
        if shot.shot_type != "dialogue_lipsync" or not shot.spoken_text.strip() or not shot.active_speaker:
            continue
        if shot.audio_32k_path and Path(shot.audio_32k_path).is_file() and shot.asr_similarity >= 0.78:
            continue

        speaker = cmap[shot.active_speaker]
        ref_flac = Path(speaker.voice_ref_path)
        raw_flac = audio_dir / f"{shot.shot_id}_raw.flac"
        wav_32k = audio_dir / f"{shot.shot_id}_32k.wav"

        # Inject <emphasis> around accent_word if present and not already tagged
        line_body = shot.spoken_text.strip()
        if shot.accent_word and shot.accent_word in line_body and "<" not in line_body:
            line_body = line_body.replace(shot.accent_word, f"<emphasis>{shot.accent_word}</emphasis>", 1)
        tag_prefix = shot.drama3_acting_tag.strip()
        if tag_prefix and not tag_prefix.startswith("["):
            tag_prefix = f"[{tag_prefix}]"
        drama3_prompt = f"<|speaker:0|>{tag_prefix} {line_body}".strip()

        best_qa: Optional[Dict[str, Any]] = None
        for att in range(cfg.max_qa_retries + 1):
            d_seed = cfg.seed_base + 1200 + idx_s * 19 + att
            print(f"[Stage 3B · 分句戏剧配音] {shot.shot_id} ({speaker.name}): {line_body} (attempt {att+1})", flush=True)
            cloud.run_drama3_line(ref_flac, drama3_prompt, raw_flac, seed=d_seed)
            qa = verify_spoken_audio(cloud, raw_flac, shot.spoken_text, min_similarity=0.78, min_f0_std=10.0, max_duration_s=9.5)
            best_qa = qa
            print(
                f"  [Gate 3 分句ASR盲测] {shot.shot_id}: got='{qa['transcript_norm']}' sim={qa['similarity']} dur={qa['duration_s']}s passed={qa['passed']}",
                flush=True,
            )
            if qa["passed"]:
                break

        assert best_qa is not None
        pre_pad = 0.16 if shot.cut_in == "jcut" else 0.08
        frames, h3_sec, speech_dur = _prepare_h3_32k_clause_wav(raw_flac, wav_32k, pre_pad_s=pre_pad)
        shot.audio_flac_path = str(raw_flac)
        shot.audio_32k_path = str(wav_32k)
        shot.audio_duration_s = speech_dur
        shot.h3_frames = frames
        shot.target_seconds = h3_sec
        shot.asr_similarity = float(best_qa["similarity"])
        gate3_log[shot.shot_id] = best_qa

    cloud.free_comfy_vram()
    (cfg.project_dir / "gate3_dubbing_report.json").write_text(
        json.dumps(gate3_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.project_dir / "episode_bible.json").write_text(
        json.dumps(bible.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bible
