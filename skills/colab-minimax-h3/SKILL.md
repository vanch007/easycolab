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
| `minimax_h3_ref2va_tara` (`Ref2VA + TARA`) | ComfyUI (`11`) / Native App | Standard (Uncensored prompt) | **Motion Transfer**: guide video (`<Video 1>`) + 1–9 refs + `ComfyUI-BFSNodes` (`BFSMiniMaxH3NativeTemporal`) + `minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors` | **4–8 steps** (`Euler / simple`) | G4 96GB / A100 80GB |
| `recavsr_upscaler` (`ReCaVSR Wan2.2-5B`) | ComfyUI (`13`) / Native App | N/A (Super-Resolution) | **Fast Streaming 4K Upscaler** (~15× faster than SeedVR2): `bf16`/`int8_convrot` + `FlashDecoder`, `1.5×/2×/4×`, zero OOM on long clips | **1 step / block** (`21+8` streaming) | G4 96GB / 24GB+ (17.3 GB weights) |
| `seedvr2_upscaler` (`SeedVR2 7B FP16`) | ComfyUI (`12`) / Native App | N/A (Super-Resolution) | **Heavy Restoration Upscaler**: `7B FP16` / `7B-sharp FP16` / `3B FP16`, arbitrary short-edge `1080p/2K/4K`, best for badly degraded footage | **1 step** (full 3D VAE) | G4 96GB PRO 6000 (17 GB weights) |
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
   - Check live status across **all** saved accounts at once (CU balance, active VMs & GPU models without switching accounts):
     `./scripts/colab_h3_cli.py auth-status-all`
   - **Default when booting a new GPU VM (multi-account concurrent race, first-wins)**:
     `./scripts/colab_h3_cli.py wake-multi --gpu G4 --mode first --persistent`
     - Automatically queries live status across all saved accounts, skips accounts that already have an active VM, races all idle accounts in parallel (`--retry-interval 60` after 3 failed attempts), immediately cancels losing accounts + cleans any duplicate millisecond allocation on first win, switches the default active account & session to whichever account succeeded, and syncs `active_session.json`. (Works seamlessly even if only 1 account is saved; pass `--accounts a@g.com,b@g.com` or `--mode all` for parallel GPUs).
   - For an existing EasyColab G4 session, attach (`./scripts/colab_h3_cli.py connect --url https://xxxx.gradio.live --password <pwd>`) or wake a specific single account (`./scripts/colab_h3_cli.py --account <email> wake --gpu G4 --persistent`). See [reusing G4 and running reference-motion shots](./references/models-and-workflows.md#reusing-g4-for-turbo-reference-motion-shots).
3. **Stage 2 — Install Pack or Load Native Model**:
   - ComfyUI packs:
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack recavsr_upscaler --wait` (ReCaVSR Fast Streaming 4K Video Upscaler)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack seedvr2_upscaler --wait` (SeedVR2 7B FP16 Heavy Restoration Upscaler)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_ref2va_tara --wait` (Motion Transfer with TARA)
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_10eros_max --wait`
     - `./scripts/colab_h3_cli.py comfy-install-pack --pack minimax_h3_sulphur3 --wait`
   - Native DiffSynth models:
     - `./scripts/colab_h3_cli.py download-model --model minimax_h3 --wait`
     - `./scripts/colab_h3_cli.py load-model --model minimax_h3 --wait`
4. **Stage 3 — Execute Generation & Auto-Sleep (Default)**:
   - The generation commands (`comfy-upscale-recavsr`, `comfy-upscale`, `comfy-run-tara`, `comfy-run-10eros`, `comfy-run-sulphur3`, `generate-video`, `generate-image`) download successful outputs. With default cleanup enabled, they sync ComfyUI to Drive (if applicable), disconnect, and run `stop-all` in `finally`, including on errors.
   - Fast Streaming Video Super-Resolution (`ReCaVSR + FlashDecoder`, 2× for 720p→2K or 4× for 480p→4K):
     `./scripts/colab_h3_cli.py comfy-upscale-recavsr --video ./outputs/clip.mp4 --scale 2.0 --transformer transformer.safetensors --out ./outputs`
   - Deep Restoration Video Super-Resolution (`SeedVR2 7B FP16` to 1080p/2K/4K, adaptive VAE decode tile `2048` on 2K/4K):
     `./scripts/colab_h3_cli.py comfy-upscale --video ./outputs/clip.mp4 --resolution 2160 --decode-tile-size 2048 --model seedvr2_ema_7b_fp16.safetensors --out ./outputs`
   - Motion Transfer (`Ref2VA + TARA`):
     `./scripts/colab_h3_cli.py comfy-run-tara --ref-video dance.mp4 --ref-image target_char.png --prompt "<Subject 1>: person in <Picture 1>. Replace character in <Video 1> with <Subject 1>..." --out ./outputs`
   - Multi-image + voice reference (`10Eros-Max`):
     `./scripts/colab_h3_cli.py comfy-run-10eros --mode multiref --prompt "The woman from <Picture 1> (<Audio 1>) in <Picture 2> speaks Mandarin Chinese: ..." --ref-image char.png --ref-image scene.png --ref-audio voice_seed.wav --out ./outputs`
   - Sulphur-3 + H3 Corrector:
     `./scripts/colab_h3_cli.py comfy-run-sulphur3 --mode t2va --prompt-file shot.json --corrector-strength 1.0 --out ./outputs`
   - Native BF16 / 4K Image:
     `./scripts/colab_h3_cli.py generate-video ...` or `./scripts/colab_h3_cli.py generate-image ...`
   - **Multi-shot batch runs**: use `--keep-alive` on intermediate shots. Omit it on the final shot only when account-wide release is authorized. Keep it on every shot when attaching to a user-owned or shared session without that authorization.
5. **Stage 4 — Manual Fallback Shutdown (If `--keep-alive` or Setup-Only Run)**:
   - After downloading outputs and confirming the queue is clear, use the authorized shutdown scope. `disconnect` can unassign the connected runtime. `stop-all` terminates all active VMs on the account and requires authorization for that scope. After an authorized account-wide stop, verify `active_assignments == 0` and `burn_rate_hourly == 0.0`.

## CLI Quick Reference (`./scripts/colab_h3_cli.py`)

| Group | Subcommands | Notes |
|---|---|---|
| Auth & VM | `auth-status`, `auth-status-all`, `auth-url`, `auth-code`, `auth-list`, `auth-switch`, `auth-remove`, `wake`, `wake-multi`, `connect`, `config-idle`, `keepalive`, `disconnect`, `stop-all` | Multi-account vault (`accounts/`); `auth-status-all` probes all accounts concurrently; default `wake-multi --gpu G4 --mode first --persistent` races all idle accounts and auto-switches to whichever wins; pass `--account <email>` for parallel GPUs |
| Native H3 | `status`, `download-model`, `load-model`, `unload-model`, `generate-video`, `generate-image` | Talks to `/minimax_api/*`; generation commands **auto-sleep on finish/error** unless `--keep-alive` is passed |
| ComfyUI | `comfy-status`, `comfy-start`, `comfy-stop`, `comfy-free-vram`, `comfy-sync-drive`, `comfy-install-pack`, `comfy-download-weight`, `comfy-download-output` | Uses EasyColab NVMe hybrid symlinks, `custom_nodes_bundle.tar.gz`, and G4 `--highvram --disable-mmap --fast` |
| Direct H3 Pipelines | `comfy-upscale-recavsr`, `comfy-upscale`, `comfy-run-tara`, `comfy-run-10eros`, `comfy-run-sulphur3` | Uploads inputs, runs API graph, downloads MP4, **auto-syncs & auto-sleeps on finish/error** unless `--keep-alive` is passed; supports `--dry-run` |
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
5. **SeedVR2 7B Video Upscaler Rules (PRO 6000 96GB)**:
   - Always keep `cache_model=False` when `offload_device="none"` (the node raises an error if `cache_model=True` with `offload_device="none"`).
   - Keep `decode_tiled=False` at `<=1080p` for maximum speed; for `>=1440p` (`2K`/`4K`), enable `decode_tiled=True` with `--decode-tile-size 2048` (`overlap=128`) so 96GB VRAM avoids Phase 3 OOM without 1024px slowdown.
   - `batch_size` auto-snaps to `4n+1` (`13`, `17`, `21`).
6. **Preventing Camera-Turn & Axis-Crossing Artifacts**:
   - In 90° profile dialogue, use a matched 90° profile reference image; avoid plural words (`eyes`, `brows`) and negative phrasing (`without turning toward camera`), which activate those tokens in Qwen3-VL.
   - Lock screen positions explicitly (`<Picture 1> on screen-left facing right`), use **1 camera move per 5s clip**, and end with a silent closed-mouth hold + `No subtitles`.
7. **Security & Auto-Sleep Cost Hygiene**:
   - Never share `*.gradio.live`, `*.trycloudflare.com`, or admin passwords in screenshots or logs.
   - The generation CLI's default `finally` cleanup calls `disconnect` and `stop-all`, including on errors. Direct ComfyUI graph submission does not invoke that CLI wrapper. Check scope and ownership before using the default cleanup.

## Deep References

- Models, exact HF paths, VRAM budgets, ComfyUI node wiring & `/minimax_api/*` endpoints: [models-and-workflows.md](./references/models-and-workflows.md)
- Prompt engineering, `17k+5` grids, Self-Anchor Voice, Sulphur-3 corrector tuning & multi-segment long-shot continuity: [prompting-and-pro-techniques.md](./references/prompting-and-pro-techniques.md)
