"""Typed domain contracts for the EasyColab Cinema-Grade Short-Drama System."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_VISUAL_BASES = ("live_action", "3d_cg", "anime_2d")
VALID_ASPECT_RATIOS = ("16:9", "9:16")
VALID_SPACE_TYPES = ("indoor_semi", "outdoor_building", "natural_street")
VALID_SHOT_TYPES = (
    "establishing",
    "insert_prop",
    "dialogue_lipsync",
    "reaction_cutaway",
    "action_impact",
)
VALID_SCALES = ("EWS", "WS", "MWS", "MS", "OTS", "MCU", "CU", "ECU")
VALID_CUT_MODES = ("hard", "jcut")


@dataclass
class CharacterSpec:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    gender: str = "female"
    age_range: str = "26-30"
    body_ratio: str = "7.0-head realistic East Asian proportion, 168cm slender poise"
    # LaoLi Identity Triad + 10-point facial anchor + 8-field costume spec
    role_badge_accessory: str = ""
    costume_spec_en: str = ""
    posture_habit_en: str = ""
    anti_drift_banlist_en: str = ""
    face_10_locks_en: str = ""
    # Voice casting specs (Qwen3-TTS VoiceDesign + Fish Audio Drama 3)
    voice_instruct_zh: str = ""
    voice_desc_en: str = ""
    voice_label_en: str = ""
    audition_line_zh: str = ""
    # Populated asset paths (relative to project_dir or absolute)
    user_face_path: str = ""
    user_voice_path: str = ""
    portrait_path: str = ""
    face_anchor_path: str = ""
    multiview_sheet_path: str = ""
    view_paths: Dict[str, str] = field(default_factory=dict)
    voice_ref_path: str = ""
    qa_passed: bool = False


@dataclass
class LocationSpec:
    id: str
    name: str
    space_type: str = "indoor_semi"
    era_style_en: str = ""
    time_weather_en: str = "Bright warm afternoon sunlight with crisp Tyndall sunbeams"
    master_desc_en: str = ""
    four_walls_en: Dict[str, str] = field(default_factory=dict)
    master_plate_path: str = ""
    grid9_sheet_path: str = ""
    angle_plates: Dict[str, str] = field(default_factory=dict)
    qa_passed: bool = False


@dataclass
class PropSpec:
    id: str
    name: str
    isolated_desc_en: str = ""
    card_path: str = ""
    qa_passed: bool = False


@dataclass
class EmotionBeat:
    beat_id: str
    emotion_keyword: str
    story_action: str
    context_eid: str = "E01"
    core_camera_lock: str = "侧45度 · 中近景 · 50mm"


@dataclass
class ShotSpec:
    shot_id: str
    index: int
    beat_id: str = "B01"
    shot_type: str = "dialogue_lipsync"
    scale: str = "MCU"
    lens_mm: int = 50
    camera_angle: str = "侧45度+平视"
    camera_motion: str = "slow subtle dolly-in"
    location_id: str = ""
    plate_slot: str = "000_front"
    characters_in_shot: List[str] = field(default_factory=list)
    active_speaker: Optional[str] = None
    props_in_shot: List[str] = field(default_factory=list)
    # Dialogue & LaoLi acting locks
    spoken_text: str = ""
    drama3_acting_tag: str = ""
    accent_word: str = ""
    acting_expression: str = ""
    acting_gesture: str = ""
    state_before: str = "—"
    state_after: str = "—"
    # ArcReel-style 2nd-stage visual authoring fields (uses @[id] semantic mentions)
    image_scene_en: str = ""
    lighting_en: str = ""
    video_action_delta_en: str = ""
    soundscape_foley_en: str = ""
    music_cue_en: str = ""
    # Timing & Editorial
    target_seconds: float = 3.04
    h3_frames: int = 73
    cut_in: str = "hard"
    carryover_prev: bool = False
    # Generated artifacts & QA metrics
    audio_flac_path: str = ""
    audio_32k_path: str = ""
    audio_duration_s: float = 0.0
    keyframe_path: str = ""
    video_path: str = ""
    tail_frame_path: str = ""
    contact_sheet_path: str = ""
    trim_in_s: float = 0.0
    trim_out_s: float = 0.0
    asr_similarity: float = 0.0
    vlm_score: float = 0.0
    qa_notes: str = ""
    qa_passed: bool = False


@dataclass
class EpisodeBible:
    project_name: str
    title: str
    synopsis: str
    aspect_ratio: str = "16:9"
    visual_base: str = "live_action"
    style_phrase_en: str = (
        "Cinematic 35mm live-action dramatic comedy, natural matte skin pores, "
        "warm directional key light with crisp contrast, shallow anamorphic depth of field"
    )
    source_story: str = ""
    characters: List[CharacterSpec] = field(default_factory=list)
    locations: List[LocationSpec] = field(default_factory=list)
    props: List[PropSpec] = field(default_factory=list)
    beats: List[EmotionBeat] = field(default_factory=list)
    shots: List[ShotSpec] = field(default_factory=list)

    def char_map(self) -> Dict[str, CharacterSpec]:
        out: Dict[str, CharacterSpec] = {}
        for c in self.characters:
            out[c.id] = c
            out[c.name] = c
            for al in c.aliases:
                if al:
                    out[al] = c
        return out

    def loc_map(self) -> Dict[str, LocationSpec]:
        out: Dict[str, LocationSpec] = {}
        for loc in self.locations:
            out[loc.id] = loc
            out[loc.name] = loc
        return out

    def prop_map(self) -> Dict[str, PropSpec]:
        out: Dict[str, PropSpec] = {}
        for p in self.props:
            out[p.id] = p
            out[p.name] = p
        return out

    def resolution(self) -> tuple[int, int]:
        if self.aspect_ratio == "9:16":
            return (704, 1280)
        return (1280, 704)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeBible":
        return cls(
            project_name=str(data.get("project_name") or "drama_project"),
            title=str(data.get("title") or "Untitled Short Drama"),
            synopsis=str(data.get("synopsis") or ""),
            aspect_ratio=str(data.get("aspect_ratio") or "16:9"),
            visual_base=str(data.get("visual_base") or "live_action"),
            style_phrase_en=str(data.get("style_phrase_en") or ""),
            source_story=str(data.get("source_story") or ""),
            characters=[CharacterSpec(**c) for c in (data.get("characters") or [])],
            locations=[LocationSpec(**l) for l in (data.get("locations") or [])],
            props=[PropSpec(**p) for p in (data.get("props") or [])],
            beats=[EmotionBeat(**b) for b in (data.get("beats") or [])],
            shots=[ShotSpec(**s) for s in (data.get("shots") or [])],
        )


@dataclass
class ProductionConfig:
    project_dir: Path
    story_text: str
    project_name: str = "cinema_drama"
    char_faces: Dict[str, str] = field(default_factory=dict)
    char_voices: Dict[str, str] = field(default_factory=dict)
    aspect_ratio: str = "16:9"
    visual_base: str = "live_action"
    style_phrase_en: str = ""
    target_duration_s: float = 48.0
    max_qa_retries: int = 2
    burn_subtitles: bool = True
    export_jianying: bool = True
    export_html_board: bool = True
    keep_alive: bool = True
    seed_base: int = 20261125
