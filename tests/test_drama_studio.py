import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from drama_studio.cloud_client import extract_json_object
from drama_studio.exporters import (
    export_html_storyboard_board,
    export_jianying_draft_bundle,
    render_final_cinema_mp4,
)
from drama_studio.models import (
    CharacterSpec,
    EmotionBeat,
    EpisodeBible,
    LocationSpec,
    ProductionConfig,
    PropSpec,
    ShotSpec,
)
from drama_studio.qa_engine import build_video_contact_sheet, detect_video_signals
from drama_studio.rules import (
    compile_h3_video_prompt_and_refs,
    compile_keyframe_prompt_and_refs,
    lint_episode_bible,
    normalize_and_enrich_bible,
)


def _make_sample_bible(tmp_path: Path) -> EpisodeBible:
    face_vivi = tmp_path / "vivi_face.png"
    face_hou = tmp_path / "hou_face.png"
    prop_ptr = tmp_path / "pointer.png"
    plate_000 = tmp_path / "study_000.png"
    aud_wav = tmp_path / "s02.wav"

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=256x256:d=0.1", "-frames:v", "1", str(face_vivi)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=green:s=256x256:d=0.1", "-frames:v", "1", str(face_hou)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=pink:s=256x256:d=0.1", "-frames:v", "1", str(prop_ptr)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gold:s=640x352:d=0.1", "-frames:v", "1", str(plate_000)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=240:duration=2.5", "-ar", "32000", "-ac", "2", str(aud_wav)], check=True)

    scales_angles = [
        ("establishing", "EWS", 20, "侧45度+微俯", [], None, ""),
        ("insert_prop", "ECU", 100, "微距俯拍", [], None, ""),
        ("dialogue_lipsync", "MCU", 65, "侧45度+微仰", ["vivi"], "vivi", "这个月报表又是负增长！"),
        ("reaction_cutaway", "CU", 85, "侧45度+平视", ["houhou"], None, ""),
        ("dialogue_lipsync", "OTS", 50, "后肩过右肩", ["vivi", "houhou"], "vivi", "都不懂降本增效吗？"),
        ("insert_prop", "ECU", 100, "微距平视", [], None, ""),
        ("dialogue_lipsync", "MS", 50, "侧45度+平视", ["houhou", "vivi"], "houhou", "老板消消气嘛。"),
        ("dialogue_lipsync", "CU", 85, "侧面+微仰", ["houhou"], "houhou", "今晚安排解压服务吗？"),
        ("action_impact", "MWS", 28, "大仰角", [], None, ""),
        ("reaction_cutaway", "CU", 85, "侧45度+微俯", ["vivi", "houhou"], None, ""),
        ("dialogue_lipsync", "CU", 85, "大仰角", ["vivi"], "vivi", "进门不敲门？"),
        ("dialogue_lipsync", "MWS", 35, "大俯角", ["houhou", "vivi"], "vivi", "把规范抄一百遍。"),
    ]
    shots = []
    for idx, (stype, sc, mm, ang, chars, spk, txt) in enumerate(scales_angles, start=1):
        shots.append(
            ShotSpec(
                shot_id=f"s{idx:02d}",
                index=idx,
                shot_type=stype,
                scale=sc,
                lens_mm=mm,
                camera_angle=ang,
                camera_motion="slow dolly-in",
                location_id="loc_study",
                plate_slot="000_front",
                characters_in_shot=list(chars),
                active_speaker=spk,
                props_in_shot=["prop_ptr"] if idx in (2, 3, 12) else [],
                spoken_text=txt,
                accent_word=txt[:2] if txt else "",
                acting_expression="sharp eyebrow arch and narrowed gaze" if txt else "frozen wide-eyed stare",
                acting_gesture="taps pointer crisply against ledger" if txt else "hand pauses mid-air",
                image_scene_en="Inside @[loc_study], @[vivi] holds @[prop_ptr] facing @[houhou].",
                video_action_delta_en="@[vivi] taps @[prop_ptr] sharply while @[houhou] smirks.",
                audio_32k_path=str(aud_wav) if spk else "",
                target_seconds=2.8,
            )
        )

    bible = EpisodeBible(
        project_name="unit_test_drama",
        title="女帝职场整顿",
        synopsis="测试短剧",
        source_story="“这个月报表又是负增长！”“都不懂降本增效吗？”“老板消消气嘛。”",
        characters=[
            CharacterSpec(
                id="vivi", name="女帝", costume_spec_en="crimson dragon blazer",
                face_anchor_path=str(face_vivi), view_paths={"front": str(face_vivi), "3q": str(face_vivi)},
                voice_ref_path=str(aud_wav),
            ),
            CharacterSpec(
                id="houhou", name="秘书", costume_spec_en="emerald satin qipao",
                face_anchor_path=str(face_hou), view_paths={"front": str(face_hou), "3q": str(face_hou)},
                voice_ref_path=str(aud_wav),
            ),
        ],
        locations=[
            LocationSpec(
                id="loc_study", name="御书房", master_plate_path=str(plate_000),
                grid9_sheet_path=str(plate_000), angle_plates={"000_front": str(plate_000)},
            )
        ],
        props=[PropSpec(id="prop_ptr", name="教鞭", isolated_desc_en="pink metallic pointer", card_path=str(prop_ptr))],
        beats=[EmotionBeat(beat_id="B01", emotion_keyword="威压", story_action="训话")],
        shots=shots,
    )
    return normalize_and_enrich_bible(bible, target_duration_s=38.0)


class TestDramaStudioEngine(unittest.TestCase):
    def test_gate0_linter_passes_cinema_storyboard_and_catches_talking_heads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bible = _make_sample_bible(Path(td))
            res = lint_episode_bible(bible, target_duration_s=38.0)
            self.assertTrue(res.passed, f"Expected pass, got errors: {res.errors}")

            # Trigger talking-head & low-shot-count errors
            bad_bible = EpisodeBible.from_dict(bible.to_dict())
            bad_bible.shots = bad_bible.shots[:4]
            bad_bible.shots[2].spoken_text = "这是一段超级无敌漫长根本不切镜头的站桩式念稿台词足足超过二十五个汉字肯定要被拦截！"
            bad_res = lint_episode_bible(bad_bible, target_duration_s=45.0)
            self.assertFalse(bad_res.passed)
            joined = "\n".join(bad_res.errors)
            self.assertIn("C_SHOT_COUNT", joined)
            self.assertIn("C_LONG_LINE", joined)

    def test_mechanical_reference_compiler_replaces_mentions_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bible = _make_sample_bible(tmp)
            shot = bible.shots[4]  # OTS shot with vivi, houhou, spoken_text
            kf_prompt, kf_refs = compile_keyframe_prompt_and_refs(shot, bible)
            self.assertEqual(len(kf_refs), 3)
            self.assertNotIn("@[", kf_prompt)
            self.assertIn("Picture 1", kf_prompt)
            self.assertIn("Picture 2", kf_prompt)

            kf_dummy = tmp / "kf.png"
            shutil_src = Path(kf_refs[-1])
            kf_dummy.write_bytes(shutil_src.read_bytes())
            h3_prompt, h3_refs, h3_aud, ref_sz = compile_h3_video_prompt_and_refs(shot, bible, str(kf_dummy))
            self.assertNotIn("@[", h3_prompt)
            self.assertIn("<Picture 1>", h3_prompt)
            self.assertIn("<Audio 1>", h3_prompt)
            self.assertIsNotNone(h3_aud)
            self.assertEqual(ref_sz, "max")
            self.assertLessEqual(len(h3_refs), 6)

    def test_extract_json_object_handles_think_and_fences(self) -> None:
        raw = "<think>planning...</think>\n```json\n{\"passed\": true, \"score\": 9,}\n```"
        parsed = extract_json_object(raw)
        self.assertTrue(parsed["passed"])
        self.assertEqual(parsed["score"], 9)

    def test_signals_contact_sheet_stitch_html_and_jianying_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bible = _make_sample_bible(tmp)
            bible.shots = bible.shots[:3]
            for s in bible.shots:
                mp4 = tmp / f"{s.shot_id}.mp4"
                kf = tmp / f"{s.shot_id}_kf.png"
                cs = tmp / f"{s.shot_id}_cs.jpg"
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-v", "error",
                        "-f", "lavfi", "-i", f"testsrc=size=640x352:rate=24:duration=2.4",
                        "-f", "lavfi", "-i", "sine=frequency=330:duration=2.4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4),
                    ],
                    check=True,
                )
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-frames:v", "1", str(kf)], check=True)
                sig = detect_video_signals(mp4)
                build_video_contact_sheet(mp4, cs, s.shot_id, sig, num_frames=6)
                self.assertTrue(cs.is_file())
                s.video_path = str(mp4)
                s.keyframe_path = str(kf)
                s.contact_sheet_path = str(cs)
                s.trim_in_s = sig.recommended_in_s
                s.trim_out_s = sig.recommended_out_s

            cfg = ProductionConfig(
                project_dir=tmp / "proj",
                story_text="测试故事",
                project_name="demo_ep1",
                burn_subtitles=False,
            )
            cfg.project_dir.mkdir(parents=True, exist_ok=True)
            final_info = render_final_cinema_mp4(cfg, bible)
            self.assertTrue(Path(final_info["final_mp4"]).is_file())
            self.assertGreater(final_info["duration_s"], 5.0)

            board = export_html_storyboard_board(cfg, bible, final_info)
            self.assertTrue(board.is_file())
            self.assertIn("demo_ep1", board.read_text(encoding="utf-8"))

            jy = export_jianying_draft_bundle(cfg, bible)
            self.assertTrue(Path(jy["draft_zip"]).is_file())
            self.assertTrue((Path(jy["draft_dir"]) / "draft_content.json").is_file())


if __name__ == "__main__":
    unittest.main()
