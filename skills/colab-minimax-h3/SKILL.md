---
name: colab-minimax-h3
description: Use when generating audio-video or 4K images with MiniMax-H3 on Google Colab (G4/H100/A100/L4) via EasyColab or ComfyUI, including uncensored 10Eros-Max multi-image/voice reference, Sulphur-3 with s3_corrector, native BF16 T2VA/FLF2V, H3-Image 4K, voice anchoring, or Colab GPU wake/stop management.
---

# Colab MiniMax-H3 Full-Stack Skill

## Overview

End-to-end orchestration and production playbook for running **MiniMax-H3** (33B native joint audio-video and 4K image models) on **Google Colab** (especially 96GB G4 RTX PRO 6000 Blackwell) via **EasyColab** (`colab_bridge.py` + `app.py` `/minimax_api/*` + headless ComfyUI API).

All remote operations are encapsulated in `./scripts/colab_h3_cli.py`. Run `--help` on any subcommand for full flags.

## Model & Engine Selection Matrix

| Engine / Pack ID | Where It Runs | Uncensored | Modalities & Key Strengths | Speed / Steps | Recommended GPU |
|---|---|---|---|---|---|
| `minimax_h3_10eros_max` (`TenStrip/10Eros-Max`) | ComfyUI (`07` / `08`) | Yes (`Heretic NVFP4` + `10Eros-Max v1.0`) | Up to **9 ref images** (`<Picture 1..9>`), up to **3 ref audios** (`<Audio 1..3>`), first/last frame (`FLF2V`) | **8 steps** (`LCM / simple`) | G4 96GB / A100 80GB (48GB+ VRAM) |
| `minimax_h3_sulphur3` (`SulphurAI/Sulphur-3`) | ComfyUI (`09` / `10`) | Yes (`Heretic NVFP4` + `s3.safetensors`) | Full S3 backbone + **`s3_corrector.safetensors`** (fp32 2B velocity residual corrector) + auto-installed `ComfyUI-H3-Corrector` | **20 steps** (`Euler / simple`, `CFG=1`) | G4 96GB / A100 80GB |
| `minimax_h3_i2v_flf2v` (Official Turbo) | ComfyUI (`03` / `04`) | Standard | Official pruned int8 + 8-step Turbo LoRA, single-image & FLF2V with stereo audio | **4–8 steps** (`res_multistep / simple`) | G4 / A100 / L4 (with offload) |
| `minimax_h3` (33B BF16 Native) | EasyColab API (`/minimax_api/generate`) | Standard | Native DiffSynth BF16 T2VA & FLF2V (4–15s), G4 176GB DDR5 dual-backbone hot standby | **35 steps** (0ms cached prompt repeat) | G4 96GB |
| `minimax_h3_image` (33B 4K) | EasyColab API (`/minimax_api/generate`) | Standard | Native 6-step 4K (16MP) T2I & reference image editing; auto-shares 32B encoder with `minimax_h3` | **6 steps** | G4 96GB / A100 80GB |

## End-to-End Runbook

1. **Stage 0 — Zero-CCU Local Prep (Do BEFORE booting G4)**:
   - Prepare character/scene reference images (`1344x768` or `1056x608`), prompts, and 32kHz stereo voice seeds:
     - `./scripts/colab_h3_cli.py snap-frames --seconds 5` (snaps to `17k+5`: `56`, `73`, `124`, `192`, `243`, `362`)
     - `./scripts/colab_h3_cli.py extract-voice-seed --input raw_clip.mp4 --output voice_seed.wav --start 0 --duration 5`
2. **Stage 1 — Wake or Connect to Colab**:
   - Check OAuth & CU balance: `./scripts/colab_h3_cli.py auth-status`
   - Boot GPU VM & deploy EasyColab (sets a 10-minute cloud idle auto-sleep watchdog by default): `./scripts/colab_h3_cli.py wake --gpu G4`
   - Or attach to an existing server: `./scripts/colab_h3_cli.py connect --url https://xxxx.gradio.live --password <pwd>`
3. **Stage 2 — Install Pack or Load Native Model**:
   - ComfyUI uncensored packs:
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_10eros_max --wait`
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_sulphur3 --wait`
   - Native DiffSynth models:
     - `./scripts/colab_h3_cli.py download-model --model minimax_h3 --wait`
     - `./scripts/colab_h3_cli.py load-model --model minimax_h3 --wait`
4. **Stage 3 — Execute Generation & Auto-Sleep (Default)**:
   - Every generation command (`comfy-run-10eros`, `comfy-run-sulphur3`, `generate-video`, `generate-image`) **automatically pulls outputs, syncs ComfyUI to Drive (if ComfyUI), disconnects, and runs `stop-all` in `finally` — even if an error occurs**.
   - Multi-image + voice reference (`10Eros-Max`):
     `./scripts/colab_h3_cli.py comfy-run-10eros --mode multiref --prompt "The woman from <Picture 1> (<Audio 1>) in <Picture 2> speaks Mandarin Chinese: ..." --ref-image char.png --ref-image scene.png --ref-audio voice_seed.wav --out ./outputs`
   - Sulphur-3 + H3 Corrector:
     `./scripts/colab_h3_cli.py comfy-run-sulphur3 --mode t2va --prompt-file shot.json --corrector-strength 1.0 --out ./outputs`
   - Native BF16 / 4K Image:
     `./scripts/colab_h3_cli.py generate-video ...` or `./scripts/colab_h3_cli.py generate-image ...`
   - **Multi-shot batch runs**: pass `--keep-alive` on intermediate shots, and omit `--keep-alive` on the final shot so the last run automatically sleeps and releases the VM.
5. **Stage 4 — Manual Fallback Shutdown (If `--keep-alive` or Setup-Only Run)**:
   - If you exited early before running a generation command or passed `--keep-alive`, run `./scripts/colab_h3_cli.py stop-all` and verify `active_assignments == 0` & `burn_rate_hourly == 0.0`.

## CLI Quick Reference (`./scripts/colab_h3_cli.py`)

| Group | Subcommands | Notes |
|---|---|---|
| Auth & VM | `auth-status`, `auth-url`, `auth-code`, `wake`, `connect`, `config-idle`, `keepalive`, `disconnect`, `stop-all` | `wake` sets `--idle-minutes 10` watchdog; saves active URL + Basic Auth password (`0600`) |
| Native H3 | `status`, `download-model`, `load-model`, `unload-model`, `generate-video`, `generate-image` | Talks to `/minimax_api/*`; generation commands **auto-sleep on finish/error** unless `--keep-alive` is passed |
| ComfyUI | `comfy-status`, `comfy-start`, `comfy-stop`, `comfy-free-vram`, `comfy-sync-drive`, `comfy-install-pack`, `comfy-download-weight`, `comfy-download-output` | Uses EasyColab NVMe hybrid symlinks, `custom_nodes_bundle.tar.gz`, and G4 `--highvram --disable-mmap --fast` |
| Direct H3 Pipelines | `comfy-run-10eros`, `comfy-run-sulphur3` | Uploads inputs, runs API graph, downloads MP4, **auto-syncs & auto-sleeps on finish/error** unless `--keep-alive` is passed; supports `--dry-run` |
| 0-CCU Local | `snap-frames`, `extract-voice-seed` | Offline `17k+5` calculator & `ffmpeg` 32kHz stereo voice seed extractor |

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
6. **Security & Auto-Sleep Cost Hygiene**:
   - Never share `*.gradio.live`, `*.trycloudflare.com`, or admin passwords in screenshots or logs.
   - Auto-sleep is **ON by default** for every generation command (`finally` block runs `comfyui/sync_drive` + `disconnect` + `stop-all` after pulling outputs). Only pass `--keep-alive` during consecutive multi-shot runs, and always omit it on the last shot (or run `stop-all`).

## Deep References

- Models, exact HF paths, VRAM budgets, ComfyUI node wiring & `/minimax_api/*` endpoints: [models-and-workflows.md](./references/models-and-workflows.md)
- Prompt engineering, `17k+5` grids, Self-Anchor Voice, Sulphur-3 corrector tuning & multi-segment long-shot continuity: [prompting-and-pro-techniques.md](./references/prompting-and-pro-techniques.md)
