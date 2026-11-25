# Per-Stage Official Prompt Engineering Guide (EasyColab Short-Drama Pipeline)

Use dedicated, model-native prompt structures at every stage of the pipeline instead of generic one-size-fits-all text. Never advance from one stage to the next without passing its Model-in-the-Loop Quality Gate.

---

## Stage 1 — Script, Four-Wall Set Bible, Prop Ledger & Shot Breakdown (`Qwen3.8-27B-Uncensored`)

- **Goal**: Lock spatial geometry, props, and character beats *before* generating any pixels or audio.
- **Structure**: Strict JSON Showrunner Contract (`location_bible` with **all 4 walls**: `front_wall`, `right_wall`, `back_reverse_door_wall`, `left_wall`, `prop_ledger`, `character_design_lock`, `voice_design_specs`, `shots[]`).
- **Rules**:
  1. Every recurring set (`@Study`) gets an immutable **Four-Wall Set Bible** (time of day, sunbeam angle, key furniture, rear door hinges/material, courtyard outside doors) so reverse angles never hallucinate a new room.
  2. Every held or interactive prop (glasses, pointer, memorials, bowls, trays) gets a permanent decoupled `prop_ledger` entry specifying shape, material, color, and resting state.
  3. One shot = one beat = one primary speaker line (`2–13s`) so per-shot Drama 3 audio and opening keyframes stay synchronized.

---

## Stage 2 — Empty-Handed Multi-View Character Kit, Prop Cards & 360° Scene Plate Library

1. **Empty-Handed 4-View Turnaround + 3-Part Character Kit (Prop Decoupling — Reddit/X Standard)**:
   - Generate character turnarounds **empty-handed** (hands empty/relaxed at sides unless an accessory is permanently worn like glasses/hairpin) on seamless light-gray studio backdrop so characters never "weld" props (pointers, folders, trays) into every unrelated shot.
   - Generate separate isolated **Prop Reference Cards** (`props/*.png`) on clean light-gray background when interactive hero props reappear across cuts.
   - Split every character asset pack into 3 high-priority anchors:
     1. **`{char}_face.png`**: High-res close-up square face+collar crop from the canonical reference image (locks East Asian facial identity in H3 / Qwen-Edit).
     2. **`{char}_3q.png` / `{char}_side.png` / `{char}_back.png`**: Clean body/costume turnaround views sliced from the `1536×864` MOSS turnaround sheet.
     3. **`{char}_canonical.png`**: Clean full-body empty-handed neutral pose.
2. **360° Empty-Set Camera Sweep & 4-Angle Plate Library (`comfy-sweep-360-plates`)**:
   - After locking the establishing empty-room plate with `MOSS-Image T2I` (`1280×704`), run `comfy-sweep-360-plates` (`10Eros-Max --frames 124`) to pan 360° smoothly through the empty room and automatically extract **4 directional plates** (`000_front`, `090_right`, `180_reverse`, `270_left`).
   - Always feed the shot's **matching directional plate** (e.g., `180_reverse` for doorway entrances) instead of reusing `000_front` on reverse angles.

- **MOSS-Image Turnaround Prompt Rules**:
  - Plain natural-language paragraphs (`60–150 words`), `steps >= 28` (`28–40`), `CFG = 3.0`, `Euler / simple`, `--disable-nsfw-lora`, zero collage labels.
- **Gate 1 (Multimodal Vision)**: Inspect every turnaround sheet and all 4 extracted 360° scene plates (`read_file`) before keyframe composition.

---

## Stage 3 — Multi-Image Storyboard Keyframes (`Qwen-Image-Edit-2511` & `10Eros-Max`)

**Do NOT use `MOSS-Image + armV5` for multi-image character+scene compositing** (`armV5` with VAE pixel-locks `image_0`, while disconnecting VAE drops facial identity). Use **either**:

1. **Pure Static Multi-Ref & 96-Pose Camera Rotation (`Qwen-Image-Edit-2511` — `comfy-run-qwen-edit`)**:
   - **0 custom nodes**, native `TextEncodeQwenImageEditPlus` (`image1..image3`) + `FluxKontextMultiReferenceLatentMethod("index_timestep_zero")` + `ModelSamplingAuraFlow(shift=3.1)` + `CFGNorm(1.0)`.
   - **4-Step Lightning Mode**: `--steps 4 --cfg 1.0 --lightning-lora-strength 1.0` (fast 4-step iteration).
   - **96-Pose Angle Rotation (`--angle-lora-strength 0.9`)**: Prepend exact `<sks>` camera pose tokens from `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`:
     - Format: `<sks> <azimuth> <elevation> <distance>`
     - Azimuths: `front view`, `front-right quarter view`, `right side view`, `back-right quarter view`, `back view`, `back-left quarter view`, `left side view`, `front-left quarter view`
     - Elevations: `low-angle shot`, `eye-level shot`, `elevated shot`, `high-angle shot`
     - Distances: `close-up`, `medium shot`, `wide shot`
     - Prompt structure: `<sks> right side view eye-level shot medium shot, preserve exact face identity, glasses, hairstyle, and costume details`.
2. **Dynamic/Multi-Character Composition Fallback (`10Eros-Max --mode multiref --ref-image-size max`)**:
   - Feed `{char}_face.png` + full-body/3q cards + matching directional scene plate (`000_front` / `180_reverse`) into 22 frames and extract frame 10 with `ffmpeg`.
- **Gate 4 (Strict Face-Lock & Set-Lock Vision QA)**: Side-by-side `read_file` check of every `shot_0N_kf.png` against `{char}_face.png` and directional `scene_plates/*.png` before rendering full video.

---

## Stage 4 — Character Voice Casting & Emotional Acting (`Fish Audio Drama 3 35B`)

Based on Fish Audio Drama 3 Technical Report **Appendix C (Table 10)** & `ComfyUI-Drama3` source (`prompt.py` / `chunking.py`):

- **One-Time Reference Seed Rule**: Forge **1 expressive, human-like reference seed per character ONCE** (persisted under `voices/{char}_ref.flac`) and reuse it via Drama 3 voice cloning for every dialogue line across all episodes.
- **Line-Level Acting Brief (`[...]` at line start)**:
  - Keep it **one concise natural-language bracket** under the 300-byte chunk budget describing **vocal anatomy + age + timbre + delivery intent** (e.g., `[warm smoky alto Chinese actress early 30s, dry commanding wit, close-mic live-action dialogue]`).
  - **NEVER** use robotic `"calm crisp studio narration, steady pacing"` (Drama 3 explicitly mirrors the energy of its prompt/reference: *"a flat clip gives a flat take, a lively one a lively take"*).
  - **NEVER** stack long English `Direction: ...\nLine: ...` blocks that exceed `CHUNK_BYTES=300` or leak spoken tags.
- **Inline Point Cues (`[...]`) & Scoped Phrase Cues (`<cue>...</cue>`)**:
  - Point cues: `[sighs]`, `[chuckles]`, `[laughs]`, `[pause]`, `[gasps]`, `[sniffs]`.
  - Scoped cues around exact words: `<whisper>...</whisper>`, `<soft>...</soft>`, `<slow>...</slow>`, `<faster>...</faster>`, `<emphasis>...</emphasis>`, `<forceful>...</forceful>`.
  - Single clear intent per sentence; never nest brackets or overload short lines with conflicting tags.
- **Sampling Defaults**: `--temperature 1.0 --top-p 0.9 --top-k 0 --speakers "one take per line"` (zero cross-speaker bleed).
- **Gate 2 & Gate 3 (Blind `Qwen3-ASR 1.7B` + Prosody Check)**:
  - Run strictly blind ASR (`context=""`) so ASR never hallucinates from expected text.
  - Check normalized character similarity (`>= 0.85` ref seed, `>= 0.78` emotional line), natural speaking rate (`2.6–6.5 chars/s` ref seed, `duration <= 14s`), expressive pitch variation (`f0_std_hz >= 18.0 Hz` on ref seeds so flat AI monotones fail), distinct gender/character `median_f0_hz`, and zero leaked direction tags.

---

## Stage 5 — Audio-Visual Multi-Ref Video Generation (`MiniMax-H3 / 10Eros-Max`)

Based on official MiniMax-H3 Multimodal Reference (`<Picture 1..9>`, `<Audio 1..3>`) template:

```text
[Reference Notes]
- <Picture 1>: Opening keyframe establishing framing, lighting, and character pose inside the set.
- <Picture 2>: Primary speaker front character reference card locking facial features and wardrobe.
- <Picture 3>: Secondary angle (3/4 view) or scene partner reference card maintaining 3D consistency.
- <Picture 4>: Locked architectural set plate ensuring background continuity.
- <Audio 1>: Cloned emotional Chinese dialogue audio driving voice timbre, cadence, and lip sync for (S1).

[integrated_multimodal_description]
At 0.00 seconds, <Picture 1> is fully integrated as the opening frame. From 0.00s to Xs, the camera performs [Motion Type + Amplitude + Speed, e.g., a subtle push-in with small amplitude at slow speed]. Character (S1) strictly locking facial identity from <Picture 2> and costume from <Picture 3> (motion-only delta: do not restate hair/eye details that conflict with <Picture 2>; write physical action + emotion arc only) speaks with synchronized lip and jaw motion matching <Audio 1>: "(S1) <exact Chinese line>" inside <Picture 4>.

[overall_soundscape]
Close-mic clear Mandarin vocal delivery synchronized to (S1)'s lip movement, realistic physical foley, subtle room ambience.

[non_diegetic_music]
Light playful Chinese comedy underscore matched to beat pacing without overpowering dialogue.
```

- **Shot-to-Shot Continuity & Editorial Assembly (`extract-last-frame` + `stitch-drama-jlcut`)**:
  - For consecutive same-camera beats, extract the previous shot's crisp tail frame (`extract-last-frame`) and supply it as `<Picture 1>` for the next shot.
  - Stitch final shot MP4s using `stitch-drama-jlcut --audio-lead 0.20 --normalize-lufs -16` so each incoming line leads the visual cut by 200ms with 0ms lip-sync offset and unified EBU R128 loudness.
- **Gate 5 (Dual Model QA per Shot MP4)**:
  1. Extract shot audio -> verify via blind `Qwen3-ASR 1.7B` (`context=""`) + `volumedetect`.
  2. Extract start (`0.5s`), mid (`50%`), and end (`dur - 0.5s`) frames -> inspect via multimodal vision (`read_file`) to confirm zero prop morphing, zero set-jumping, and 3D character likeness across motion before stitching.
