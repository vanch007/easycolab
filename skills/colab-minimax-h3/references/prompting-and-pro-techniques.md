# MiniMax-H3 Prompting & Pro Production Techniques

## 1. `17k+5` Frame Grid & Resolution Reference (24 fps)

MiniMax-H3 snaps temporal length up to `17k + 5` frames:

| `k` | Frames | Duration @ 24fps | Recommended Use |
|---|---|---|---|
| `3` | `56` | `2.3s` | Quick motion/lighting test |
| `4` | `73` | `3.0s` | Short reaction shot |
| `7` | `124` | `5.2s` | Standard single shot (sweet spot) |
| `11` | `192` | `8.0s` | Extended dialogue / action beat |
| `14` | `243` | `10.1s` | Medium continuous take |
| `21` | `362` | `15.1s` | Max trained single-pass duration |

Common 16:9 / 9:16 resolutions (multiples of 32):
- **Fast Draft / Low VRAM**: `832×480` (16:9) or `480×832` (9:16)
- **10Eros / S3 Default (`0.64 MP`)**: `1056×608` (16:9)
- **Cinema 768P (`~0.98 MP`)**: `1344×768` (16:9) or `768×1344` (9:16)

---

## 2. Multi-Image Reference Mastery (`<Picture 1>` .. `<Picture 9>`)

Available in `10Eros-Max` (`MiniMaxH3ReferenceToVideo`):
- **Slot Mapping**: `ref_image_0` -> `<Picture 1>`, `ref_image_1` -> `<Picture 2>`, ..., `ref_image_8` -> `<Picture 9>`.
- **Recommended Split (1–4 images)**:
  - `<Picture 1>`: Primary character portrait / turnaround (keep natural skin pores, avoid plastic airbrushed references).
  - `<Picture 2>`: Environment / room layout.
  - `<Picture 3>`: Second character or outfit/prop detail.
- **`ref_image_size`**:
  - `"match"` (default): scales down preserving aspect to match target pixel area; fast and low VRAM.
  - `"max"`: keeps up to 2048px short edge; reference tokens ride through every denoising step (much slower, best only for strict facial identity).
- **Avoid Pose Gravity in Dialogue**:
  - If a character must stay in a **90° side profile**, feed a **90° profile** `<Picture 1>` (e.g., extracted from the previous shot), avoid plural facial nouns (`wide eyes`, `knitted brows`), and avoid negative clauses (`without turning toward camera`) — Qwen3-VL activates `turning` + `camera` tokens and turns the head anyway!

---

## 3. Voice Reference & Dialogue Mastery (`<Audio 1>` .. `<Audio 3>`)

### A. Self-Anchor Voice (Zero AI Read-Aloud Tone)
External TTS dry voices often sound stiff/dubbed when cloned by H3. Instead:
1. Run one unreferenced H3 pass (`t2va` or `multiref` without `ref_audio_0`) until H3 invents a natural voice timbre you like.
2. Extract 3–5 seconds into 32kHz stereo WAV:
   ```bash
   ./scripts/colab_h3_cli.py extract-voice-seed --input clip.mp4 --output char_voice_seed.wav --start 0 --duration 5
   ```
3. Feed `char_voice_seed.wav` into `ref_audio_0` (`<Audio 1>`) for all subsequent shots of that character.

### B. Speech-Active Single-Speaker Gating
- **Never connect two speakers' audios simultaneously** in a 5s shot unless they genuinely overlap; cross-attention bleeds timbres between characters.
- Connect **only the active speaker** to `ref_audio_0` and bind explicitly in the prompt:
  `<Picture 1> (<Audio 1>) speaks in Mandarin Chinese: "..."`

### C. Unlocking High Emotion from a Calm Voice Seed (`T8mars` Protocol)
`<Audio 1>` locks vocal tract timbre, while text paralinguistics + transient soundscape unlock pitch & loudness:
1. **Paralinguistic Action Prefix**: `<Picture 1> (<Audio 1>) slams his palm onto the table and shouts furiously in Mandarin Chinese with veins visible on his neck: "..."`
2. **Timed Transient Soundscape**: `Audio: A sudden explosive palm slam on the conference table at 00:00.400, immediately followed by <Audio 1>'s booming, furious roared dialogue echoing in the room.`

---

## 4. Sulphur-3 & `s3_corrector` Tuning

- **Structured JSON Captions** (recommended for `09_Sulphur-3` T2VA): specify `subjects` (`label`, `appearance`, `position`, `motion`, `interaction`), `objects`, `environment`, `camera`, `temporal` (`overall_action`, `motion_sequence`), `audio_context`, and `salient_details` (see `09_Sulphur-3` default prompt).
- **Corrector Controls (`ApplyH3Corrector`)**:
  - `strength`: `1.0` = trained residual strength; `0.0` = bypass corrector (for A/B checks); `>1.0` amplifies motion/detail correction.
  - `budget_fraction`: `0.5` (matches S3 corrector training cap on residual RMS vs velocity RMS).
  - `start_step` / `end_step` (0-based, inclusive): e.g., `0` to `4` on a 20-step run applies correction only on early structure-forming steps; `-1` runs through the final step.
- **LoRA & Realism Note**:
  - On `Sulphur-3` / `10Eros-Max`, high-rank attention LoRAs degrade quality. Prefer low-strength MLP or realism LoRAs (`fal/MiniMax-H3-Realism-People-LoRA` with trigger `r34l1sm`, or `TenStrip/MMH3_Sulphur3Eros_Loras-MLP-energy-patches`).

---

## 5. Shot Discipline & Optional Multi-Segment Continuity

For single 5s shots (`124` frames) and multi-shot sequences:
- **1 Camera Move Per 5s Clip**: Multiple timed camera sentences inside one 5s prompt trigger unintended internal jump cuts in Qwen3-VL.
- **180° Axis Lock**: State screen positions explicitly (`<Picture 1> on screen-left facing right, <Picture 2> on screen-right facing left`).
- **Silent Tail Buffer**: End the action slightly before 5s (`both hold a tense silent gaze with mouths closed`) so trailing speech doesn't clip at frame 124.
- **No Subtitles**: Always end prompts with `No subtitles` to prevent garbled burnt-in text.
- **Optional Long-Shot Custom Nodes** (installable via ComfyUI-Manager, persisted automatically in `custom_nodes_bundle.tar.gz`):
  - `ComfyUI-H3-Motion-Context` (22-frame latent continuation): set `MiniMaxH3JoinMatch` to `tuning="manual"`, `strength=0.35`, `min_excess=2.5`, `match_spread=False`, `match_detail=False` to prevent multi-segment contrast fogging.
  - `BUNNY_H3_Conditioning_Bridge` (multi-character physical interaction anti-bleed).
  - `Comfyui_Minimax_h3_latent_Upscaler` (3D spatial latent upscaler to remove background shimmer).

---

## 6. Cost & Auto-Sleep Discipline

- G4 (`RTX PRO 6000 Blackwell`, 96GB VRAM) burns ~`8.71 CCU/hour`.
- Keep all prompt drafting, image prep, and `extract-voice-seed` offline at `0 CCU`.
- **Default Auto-Sleep**: Every generation command (`comfy-run-10eros`, `comfy-run-sulphur3`, `generate-video`, `generate-image`) automatically downloads outputs first, then runs `sync_drive` (for ComfyUI) + `disconnect` + `stop-all` inside `finally` — even if generation fails.
- **Idle Watchdog**: `wake` sets `--idle-minutes 10` on the VM by default (override via `wake --idle-minutes N` or `config-idle --minutes N`).
- **Multi-Shot Batch Pattern**: Pass `--keep-alive` on shots `1..N-1` to preserve 0s model residency, then omit `--keep-alive` on shot `N` so it auto-sleeps immediately upon completion.
