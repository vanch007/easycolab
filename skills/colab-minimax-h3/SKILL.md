---
name: colab-minimax-h3
description: Use when generating audio-video or 4K images with MiniMax-H3 on Google Colab (G4/H100/A100/L4) via EasyColab or ComfyUI, including uncensored 10Eros-Max multi-image/voice reference, Sulphur-3 with s3_corrector, native BF16 T2VA/FLF2V, H3-Image 4K, voice anchoring, or Colab GPU wake/stop management.
---

# Colab MiniMax-H3 Full-Stack Skill

## Overview

End-to-end orchestration and production playbook for running **MiniMax-H3** (33B native joint audio-video and 4K image models) on **Google Colab** (especially 96GB G4 RTX PRO 6000 Blackwell) via **EasyColab** (`colab_bridge.py` + `app.py` `/minimax_api/*` + headless ComfyUI API).

Use `./scripts/colab_h3_cli.py` for the listed runtime, model, and generation commands. For custom Turbo reference-motion graphs, use the connected ComfyUI instance. Run `--help` on any CLI subcommand for full flags.

## Model & Engine Selection Matrix

| Engine / Pack ID | Where It Runs | Uncensored | Modalities & Key Strengths | Speed / Steps | Recommended GPU |
|---|---|---|---|---|---|
| `minimax_h3_10eros_max` (`TenStrip/10Eros-Max`) | ComfyUI (`07` / `08`) / Native App | Yes (`Heretic NVFP4` + `10Eros-Max v1.0`) | Up to **9 ref images** (`<Picture 1..9>`), up to **3 ref audios** (`<Audio 1..3>`), first/last frame (`FLF2V`) | **8 steps** (`LCM / simple`) | G4 96GB / A100 80GB (48GB+ VRAM) |
| `minimax_h3_sulphur3` (`SulphurAI/Sulphur-3`) | ComfyUI (`09` / `10`) / Native App | Yes (`Heretic NVFP4` + `s3.safetensors`) | Full S3 backbone + **`s3_corrector.safetensors`** (fp32 2B velocity residual corrector) + auto-installed `ComfyUI-H3-Corrector` | **20 steps** (`Euler / simple`, `CFG=1`) | G4 96GB / A100 80GB |
| `minimax_h3_ref2va_tara` (`Ref2VA + TARA`) | ComfyUI (`11`) / Native App | Standard (Uncensored prompt) | **Motion Transfer**: guide video (`<Video 1>`) + 1–9 refs + `ComfyUI-BFSNodes` (`BFSMiniMaxH3NativeTemporal`) + `minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors` | **4–8 steps** (`Euler / simple`) | G4 96GB / A100 80GB |
| `recavsr_upscaler` (`ReCaVSR Wan2.2-5B`) | ComfyUI (`13`) / Native App | N/A (Super-Resolution) | **Fast Streaming 4K Upscaler** (~15× faster than SeedVR2): `bf16`/`int8_convrot` + `FlashDecoder`, `1.5×/2×/4×`, zero OOM on long clips | **1 step / block** (`21+8` streaming) | G4 96GB / 24GB+ (17.3 GB weights) |
| `moss_image_edit` (`MOSS-Image Heretic + armV5`) | ComfyUI (`14` / `15`) / Native App | Yes (`Heretic 9B` + `NSFW LoRA`) | **Uncensored T2I Set Plates & Single-Ref Turnarounds**: `MOSS-Image BF16` + `OstrisEdit armV5` + `flux2-vae` (use `qwen_image_edit_2511` or `10Eros-Max` for multi-image compositing) | **30–40 steps** (`Euler / simple`, `CFG=3.0`) | G4 96GB / A100 / L4 (19.4 GB weights) |
| `qwen_image_edit_2511` (`Qwen-Image-Edit-2511 + Multi-Angles`) | ComfyUI (`24` / `25`) / Native App | Standard | **Multi-Ref Storyboard Keyframes & 96-Pose Camera Rotation**: 0 custom nodes, `image1..3` + `LightX2V 4-Step` + `fal 96-Pose Multi-Angles LoRA` | **4 steps** (Lightning `CFG=1.0`, `shift=3.1`) | G4 96GB / A100 / 24GB+ (29.2 GB weights) |
| `wan22_remix_nsfw` (`FX-FeiHou/wan2.2-Remix`) | ComfyUI (`16` / `17`) / Native App | Yes (`NSFW Remix v3.0` + `NSFW UMT5`) | **Fast Uncensored Video I2V / FLF2V**: High+Low 14B Lighting v3.0 fp8 + `NSFW-Wan-UMT5-XXL`, 0 custom nodes | **8 steps** (`Euler / simple`, `shift=8.0`, `CFG=1.0`, `split=4`) | G4 96GB / A100 / L4 (33.1 GB weights) |
| `minimax_h3_drama_director` (`ComfyUI-MiniMaxH3-Director`) | ComfyUI (`18` / `19`) / Native App | Yes (reuses `10Eros-Max` + `Heretic`) | **Short-Drama Timeline Studio**: multi-shot beats (`0-7: @Hero ...`), character cards, locked voice cards (`NGContextAudio`), `NGStoryboard`, seamless AV chunk chaining | **8 steps** (`LCM / simple`, `Shift 12/3`) | G4 96GB / A100 80GB (+20MB `taeh3`, 0 extra giant weights) |
| `qwen3_tts_voice_design` (`Qwen3-TTS 1.7B`) | ComfyUI (`20` / `21`) / Native App | Yes (`Apache-2.0`) | **Zero-Audio Voice Design + Emotional Clone**: forge brand-new character voice seeds from natural language (`VoiceDesign`) or clone 3–15s speech (`Base`) for H3 Director / Drama 3 | Fast (~25s) | G4 96GB / A100 / L4 (9.74 GB weights) |
| `drama3_tts` (`Fish Audio Drama 3 35B`) | ComfyUI (`22` / `23`) / Native App | Community License | **Dramatic Multi-Speaker Acting**: 35B MoE FP8 + `Qwen3-ASR 1.7B`, inline stage cues (`[sighs]`, `[whispers]`, `[laughs]`), up to 16 speakers (`<|speaker:0..15|>`) | Fast (2.0–2.7× RT) | G4 96GB / A100 80GB (44.23 GB weights) |
| `qwen38_27b_uncensored_gguf` (`Qwen3.8-27B-Uncensored`) | Native OpenAI API (`/v1/chat/completions`) | Yes (Heretic BF16 abliterated) | **Uncensored Script / Storyboard / VLM Reverse-Prompt**: `Q4_K_M` + `F16 mmproj` + explicit `MTP draft` | ~17.8 GB total | G4 96GB / A100 / L4 (24GB+) |

## End-to-End Runbook

1. **Stage 0 — Zero-CCU Local Prep (Do BEFORE booting G4)**:
   - Prepare character/scene reference images (`1344x768` or `1056x608`), prompts, and 32kHz stereo voice seeds:
     - `./scripts/colab_h3_cli.py snap-frames --seconds 5` (snaps to `17k+5`: `56`, `73`, `124`, `192`, `243`, `362`)
     - `./scripts/colab_h3_cli.py extract-voice-seed --input raw_clip.mp4 --output voice_seed.wav --start 0 --duration 5`
2. **Stage 1 — Wake or Connect to Colab**:
   - Check live status across **all** saved accounts at once (CU balance, active VMs & GPU models without switching accounts):
     `./scripts/colab_h3_cli.py auth-status-all`
   - **Default when booting a new GPU VM (multi-account concurrent race, first-wins)**:
     `./scripts/colab_h3_cli.py wake-multi --gpu G4 --mode first --persistent`
     - Automatically queries live status across all saved accounts, skips accounts that already have an active VM, races all idle accounts in parallel (`--retry-interval 60` after 3 failed attempts), immediately cancels losing accounts + cleans any duplicate millisecond allocation on first win, switches the default active account & session to whichever account succeeded, and syncs `active_session.json`. (Works seamlessly even if only 1 account is saved; pass `--accounts a@g.com,b@g.com` or `--mode all` for parallel GPUs).
   - For an existing EasyColab G4 session, attach (`./scripts/colab_h3_cli.py connect --url https://xxxx.gradio.live --password <pwd>`) or wake a specific single account (`./scripts/colab_h3_cli.py --account <email> wake --gpu G4 --persistent`). See [reusing G4 and running reference-motion shots](./references/models-and-workflows.md#reusing-g4-for-turbo-reference-motion-shots).
3. **Stage 2 — Install Pack or Check Status**:
   - ComfyUI packs:
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack moss_image_edit --wait` (MOSS-Image Uncensored + Edit armV5)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack qwen_image_edit_2511 --wait` (Qwen-Image-Edit-2511 Multi-Ref + 96-Pose Multi-Angles + 4-Step Lightning)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack wan22_remix_nsfw --wait` (Wan2.2-Remix NSFW 8-Step I2V / FLF2V)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_drama_director --wait` (MiniMax-H3 Short-Drama Timeline Studio reusing 10Eros-Max)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack qwen3_tts_voice_design --wait` (Qwen3-TTS 1.7B VoiceDesign + Base)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack drama3_tts --wait` (Fish Audio Drama 3 35B MoE + Qwen3-ASR)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack recavsr_upscaler --wait` (ReCaVSR Fast Streaming 4K Video Upscaler)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_ref2va_tara --wait` (Motion Transfer with TARA)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_10eros_max --wait`
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_sulphur3 --wait`
4. **Stage 3 — Execute Generation & Auto-Sleep (Default)**:
   - The generation commands (`comfy-run-h3-drama`, `comfy-run-qwen3-tts`, `comfy-run-drama3`, `comfy-run-wan22-remix`, `comfy-run-moss`, `comfy-run-qwen-edit`, `comfy-sweep-360-plates`, `comfy-upscale-recavsr`, `comfy-run-tara`, `comfy-run-10eros`, `comfy-run-sulphur3`) download successful outputs. With default cleanup enabled, they sync ComfyUI to Drive (if applicable), disconnect, and run `stop-all` in `finally`, including on errors.
   - **Full-Automatic Cinema-Grade Short-Drama System (`drama-produce` — Recommended for End-to-End Short Dramas)**:
     - Input ONLY `--story` (or `--story-file`) + optional `--char-face Role=/path/face.png` (auto-generates photorealistic faces when omitted) -> automatically runs 6 gated stages (`Qwen3.8-27B` 14-20 shot breakdown & Gate 0 camera-grammar lint -> Dual-track empty-handed 4-view character sheets + `3x3` 9-angle scene grids + prop cards + Gate 1 VLM QA -> `Qwen3-TTS` voice casting + `Drama 3` clause dubbing + Gate 2/3 blind `Qwen3-ASR` -> `Qwen-Image-Edit-2511` keyframes + Gate 4 VLM -> `MiniMax-H3 10Eros-Max` multi-shot rendering + FFmpeg `BLACK/FREEZE/CUT` signals + `ContactSheet` Gate 5B VLM review -> J/L-Cut EBU R128 MP4 + `.srt` + HTML Storyboard Workbench + Jianying Pro `.zip` draft):
     `./scripts/colab_h3_cli.py drama-produce --story-file story.txt --char-face "女帝=./vivi.png" --aspect 16:9 --duration 48 --out ./outputs/my_drama`
   - Zero-Audio Character Voice Seed (`Qwen3-TTS 1.7B` — **generate ONE clean neutral reference seed per character ONCE & persist it**):
     `./scripts/colab_h3_cli.py comfy-run-qwen3-tts --mode design --language Chinese --instruct "28岁冷艳古风女帝，清冽磁性中低音，咬字清晰平稳，棚录质感" --text "今天下午两点的例会按原计划在御书房举行，请提前整理本月报表。" --out ./outputs/voices/hero_ref.flac`
   - Dramatic Multi-Speaker Emotional TTS (`Fish Audio Drama 3 35B` — **reuse the locked character voice seeds via cloning + inline stage cues `[sighs]`/`[whispers]`/`[chuckles]` for ALL dialogue & acting**):
     `./scripts/colab_h3_cli.py comfy-run-drama3 --ref-audio ./outputs/voices/hero_ref.flac --ref-audio ./outputs/voices/partner_ref.flac --speakers "one take per line, hearing the line before" --voice-guidance 1.0 --text "<|speaker:0|>[sighs] ...\n<|speaker:1|>[whispers] ..." --out ./outputs`
   - Multi-Shot Short Drama Timeline (`ComfyUI-MiniMaxH3-Director` + `10Eros-Max`):
     `./scripts/colab_h3_cli.py comfy-run-h3-drama --duration 20 --ref-image hero.png --char-name Hero --ref-audio hero_voice.wav --shots "0-7: Wide tracking @Hero...\n7-15: Close-up...\n@Hero says: ..." --out ./outputs`
   - Uncensored Fast I2V / FLF2V Video (`Wan2.2-Remix NSFW v3.0`, 8 steps):
     `./scripts/colab_h3_cli.py comfy-run-wan22-remix --mode i2v --first-frame keyframe.png --prompt "cinematic camera push in..." --out ./outputs`
   - Uncensored Image Gen & Turnarounds (`MOSS-Image + Heretic + OstrisEdit armV5`):
     `./scripts/colab_h3_cli.py comfy-run-moss --mode i2i_edit --ref-image photo.png --prompt "Live-action 4-view empty-handed costume turnaround sheet..." --out ./outputs`
   - 360° Empty-Set Plate Library (`comfy-sweep-360-plates` -> `000_front` / `090_right` / `180_reverse` / `270_left`):
     `./scripts/colab_h3_cli.py comfy-sweep-360-plates --master-plate loc_study.png --scene-desc "Carved wooden palace study..." --out ./outputs/scene_plates`
   - Face-Locked Multi-Ref Storyboard Keyframes & 96-Pose Camera Rotation (`Qwen-Image-Edit-2511`):
     `./scripts/colab_h3_cli.py comfy-run-qwen-edit --ref-image face.png --ref-image body.png --ref-image scene_180_reverse.png --steps 4 --cfg 1.0 --lightning-lora-strength 1.0 --prompt "..." --out ./outputs`
   - Shot-to-Shot Last-Frame Carryover & Lip-Sync-Safe J-Cut Assembly (0-CCU):
     `./scripts/colab_h3_cli.py extract-last-frame --video shot_01.mp4 --output shot_01_last.png`
     `./scripts/colab_h3_cli.py stitch-drama-jlcut --inputs shot_01.mp4 shot_02.mp4 shot_03.mp4 --audio-lead 0.20 --normalize-lufs -16 --output final_drama.mp4`
   - Fast Streaming Video Super-Resolution (`ReCaVSR + FlashDecoder`, 2× for 720p→2K or 4× for 480p→4K):
     `./scripts/colab_h3_cli.py comfy-upscale-recavsr --video ./outputs/clip.mp4 --scale 2.0 --transformer transformer.safetensors --out ./outputs`
   - Motion Transfer (`Ref2VA + TARA`):
     `./scripts/colab_h3_cli.py comfy-run-tara --ref-video dance.mp4 --ref-image target_char.png --prompt "<Subject 1>: person in <Picture 1>. Replace character in <Video 1> with <Subject 1>..." --out ./outputs`
   - Multi-image + voice reference (`10Eros-Max`):
     `./scripts/colab_h3_cli.py comfy-run-10eros --mode multiref --prompt "The woman from <Picture 1> (<Audio 1>) in <Picture 2> speaks Mandarin Chinese: ..." --ref-image char.png --ref-image scene.png --ref-audio voice_seed.wav --out ./outputs`
   - Sulphur-3 + H3 Corrector:
     `./scripts/colab_h3_cli.py comfy-run-sulphur3 --mode t2va --prompt-file shot.json --corrector-strength 1.0 --out ./outputs`
   - **Mandatory Model Quality Gate at Every Step**: Never advance to the next pipeline stage without verifying the current stage's output using a model: verify every generated audio clip (`voices/*`, dialogue) with `Qwen3-ASR 1.7B` (`>= 85%` ref seed / `>= 80%` dialogue transcript accuracy + duration/RMS), inspect every generated image/keyframe with multimodal vision against the prop & location ledger, and verify the final video with BOTH extracted-audio ASR and multi-frame vision inspection.
   - **Per-Stage Official Prompt Engineering**: Follow [references/prompt-engineering.md](./references/prompt-engineering.md) for dedicated model-native prompts at every stage (`MOSS-Image` turnaround paragraphs, `MOSS-Image Edit` Two-Sentence Rule, `Drama 3` Appendix C vocal/cue syntax, `MiniMax-H3` 4-section `<Picture>/<Audio>` multimodal template).
   - **Multi-shot batch runs**: use `--keep-alive` on intermediate shots. Omit it on the final shot only when account-wide release is authorized. Keep it on every shot when attaching to a user-owned or shared session without that authorization.
5. **Stage 4 — Manual Fallback Shutdown (If `--keep-alive` or Setup-Only Run)**:
   - After downloading outputs and confirming the queue is clear, use the authorized shutdown scope. `disconnect` can unassign the connected runtime. `stop-all` terminates all active VMs on the account and requires authorization for that scope. After an authorized account-wide stop, verify `active_assignments == 0` and `burn_rate_hourly == 0.0`.

## CLI Quick Reference (`./scripts/colab_h3_cli.py`)

| Group | Subcommands | Notes |
|---|---|---|
| Auth & VM | `auth-status`, `auth-status-all`, `auth-url`, `auth-code`, `auth-list`, `auth-switch`, `auth-remove`, `wake`, `wake-multi`, `connect`, `config-idle`, `keepalive`, `disconnect`, `stop-all` | Multi-account vault (`accounts/`); `auth-status-all` probes all accounts concurrently; default `wake-multi --gpu G4 --mode first --persistent` races all idle accounts and auto-switches to whichever wins; pass `--account <email>` for parallel GPUs |
| Native H3 | `status`, `download-model`, `load-model`, `unload-model`, `generate-video`, `generate-image` | Talks to `/minimax_api/*`; generation commands **auto-sleep on finish/error** unless `--keep-alive` is passed |
| ComfyUI | `comfy-status`, `comfy-start`, `comfy-stop`, `comfy-free-vram`, `comfy-sync-drive`, `comfy-install-pack`, `comfy-download-weight`, `comfy-download-output` | Uses EasyColab NVMe hybrid symlinks, `custom_nodes_bundle.tar.gz`, and G4 `--highvram --disable-mmap --fast` |
| Direct H3 Pipelines | `comfy-run-h3-drama`, `comfy-run-qwen3-tts`, `comfy-run-drama3`, `comfy-run-wan22-remix`, `comfy-run-moss`, `comfy-run-qwen-edit`, `comfy-sweep-360-plates`, `comfy-upscale-recavsr`, `comfy-run-tara`, `comfy-run-10eros`, `comfy-run-sulphur3` | Uploads inputs, runs API graph, downloads FLAC/PNG/MP4, **auto-syncs & auto-sleeps on finish/error** unless `--keep-alive` is passed; supports `--dry-run` |
| 0-CCU Local | `snap-frames`, `extract-voice-seed`, `extract-last-frame`, `stitch-drama-jlcut` | Offline `17k+5` calculator, 32kHz voice seed slicer, tail-frame extractor, and J-Cut EBU R128 loudness-normalized stitcher |

## Critical Production Rules & Gotchas

1. **`17k+5` Frame Grid (24fps)**: Always snap frame counts to `17k+5` (`56≈2.3s`, `124≈5.2s`, `243≈10.1s`, `362≈15.1s`). Non-grid counts auto-snap upwards.
2. **Multi-Image Reference (`10Eros-Max` / `MiniMaxH3ReferenceToVideo`)**:
   - Hard limits: max **9** images (`<Picture 1>`..`<Picture 9>`), max **3** audios (`<Audio 1>`..`<Audio 3>`), max **3** videos (`<Video 1>`..`<Video 3>`).
   - Keep `ref_image_size="match"` for fast iteration; switch to `"max"` (2048px short edge) only for tight facial identity close-ups (significantly higher memory/time).
   - Unplug unused reference slots so extra images do not bleed into framing.
3. **Voice Reference (`Self-Anchor Voice` & Single-Speaker Gating)**:
   - Connect **only the speaking character's** `<Audio 1>` (`ref_audio_0`) per clip; passing two voices simultaneously causes cross-attention bleed and speaker swapping.
   - Prefer H3's own extracted 32kHz stereo voice (`extract-voice-seed`) over external TTS to avoid robotic read-aloud cadence.
   - To unlock shouting/intense emotion from a calm voice seed, prepend explicit paralinguistic action (`slams palm onto table and shouts furiously...`) AND add a transient physical sound cue in `Audio:` / `overall_soundscape`.
4. **Sulphur-3 + `s3_corrector` Rules**:
   - Use `s3.safetensors` (not `s3_w4a8.safetensors`, which requires PyTorch `cu130` kernels).
   - `LoadH3Corrector` stays in **fp32** (~8GB VRAM); `CFG` is fixed at `1.0` (negative prompt ignored). Set `strength=0` to A/B test raw S3 without corrector.
5. **Preventing Camera-Turn & Axis-Crossing Artifacts**:
   - In 90° profile dialogue, use a matched 90° profile reference image; avoid plural words (`eyes`, `brows`) and negative phrasing (`without turning toward camera`), which activate those tokens in Qwen3-VL.
   - Lock screen positions explicitly (`<Picture 1> on screen-left facing right`), use **1 camera move per 5s clip**, and end with a silent closed-mouth hold + `No subtitles`.
7. **Prompt Writing Rules (from `h3-prompt-journal` & `ancient-rap-battle`)**:
   - **Grant/Deny Split**: State what `<Picture 1>` preserves (identity/garment) and explicitly deny static pose inheritance (`NOT a pose/opening-frame reference; begin directly with beat 1 below`). Use waist-up crops for close-ups and lip-sync.
   - **Executable Physics & Time-Window Budget**: Write `force source -> weight shift -> peak -> recovery` instead of adjective stacking; match complexity to window length (`0.3–0.6s` single accent, `0.6–1.2s` short chain, `1.2–2.5s` compound phrase) to avoid jitter.
   - **1:1 Action-Camera Sync & Cut Separation**: Share exact timestamps between body peaks and camera moves; explicitly separate hard cuts from single-take continuous camera shifts; keep **1 lyric line per lip-sync clip** and **1 outfit state per clip**.
   - **Self-Contained Segments**: Repeat full identity locks, framing, lighting, and audio/BGM grid in every segment, ending with `No subtitles, no on-screen text, no watermarks`.
   - Full templates and 10-point pre-submission checklist: [prompting-and-pro-techniques.md](./references/prompting-and-pro-techniques.md#7-prompt-authoring-blueprint-from-h3-prompt-journal--ancient-rap-battle).
8. **Security & Auto-Sleep Cost Hygiene**:
   - Never share `*.gradio.live`, `*.trycloudflare.com`, or admin passwords in screenshots or logs.
   - The generation CLI's default `finally` cleanup calls `disconnect` and `stop-all`, including on errors. Direct ComfyUI graph submission does not invoke that CLI wrapper. Check scope and ownership before using the default cleanup.

## Deep References

- Models, exact HF paths, VRAM budgets, ComfyUI node wiring & `/minimax_api/*` endpoints: [models-and-workflows.md](./references/models-and-workflows.md)
- Prompt engineering, `17k+5` grids, Self-Anchor Voice, Sulphur-3 corrector tuning & multi-segment long-shot continuity: [prompting-and-pro-techniques.md](./references/prompting-and-pro-techniques.md)
