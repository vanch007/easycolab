# MiniMax-H3 Models, Workflows & EasyColab API Reference

## 1. Full Model Catalog (ComfyUI & Native DiffSynth)

### A. Uncensored ComfyUI Pack 1: `minimax_h3_10eros_max` (~39.1 GB)

- **Main DiT**: `TenStrip/10Eros-Max` -> `10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors` (`models/diffusion_models/`, 21.32 GiB)
- **Uncensored Encoder**: `Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4` -> `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` (`models/text_encoders/`, 14.61 GiB)
- **Video VAE (shared)**: `Comfy-Org/MiniMax-H3` -> `minimax_h3_video_vae_int8_convrot.safetensors` (`models/vae/`, 2.62 GiB)
- **Audio VAE (shared)**: `Comfy-Org/MiniMax-H3` -> `minimax_h3_audio_vae_fp32.safetensors` (`models/vae/`, 0.56 GiB)
- **Preset Workflows**:
  - `07_10Eros-Max_无限制多图参考有声视频(8步极速).json` (`MiniMaxH3ReferenceToVideo`)
  - `08_10Eros-Max_无限制首尾帧图生有声视频(8步极速).json` (`MiniMaxH3ImageToVideo`)
- **Optimal Sampling**: `LCM / simple`, **8–12 steps**, `MiniMaxH3SigmaShift(12.0, 3.0)`, `CFG = 1.0`.

### B. Uncensored ComfyUI Pack 2: `minimax_h3_sulphur3` (~62.7 GB)

- **Main DiT**: `SulphurAI/Sulphur-3` -> `s3.safetensors` (`models/diffusion_models/`, 37.46 GiB)
  - *Note*: Avoid `s3_w4a8.safetensors` unless PyTorch `cu130` W4A8 kernels are installed on the VM.
- **Velocity Residual Corrector**: `SulphurAI/Sulphur-3` -> `s3_corrector.safetensors` (`models/h3_correctors/`, 7.44 GiB, stays **fp32** = ~8 GB VRAM)
- **Custom Nodes (auto-installed from `S3_ComfyUI.zip`)**: `ComfyUI-H3-Corrector` (`LoadH3Corrector`, `ApplyH3Corrector`)
- **Uncensored Encoder & Shared VAEs**: same `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` + shared Video/Audio VAEs.
- **Preset Workflows**:
  - `09_Sulphur-3_无限制文生有声视频(含H3修正器).json`
  - `10_Sulphur-3_无限制图生首尾帧有声视频(含H3修正器).json`
- **Optimal Sampling**: `Euler / simple`, **20 steps**, `MiniMaxH3SigmaShift(12.0, 3.0)`, `CFGGuider(cfg=1.0)`.

### C. Official ComfyUI Pack: `minimax_h3_i2v_flf2v` (~39.1 GB)

- **Main DiT**: `Comfy-Org/MiniMax-H3` -> `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (19.53 GiB)
- **Encoder**: `Comfy-Org/MiniMax-H3` -> `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (14.61 GiB)
- **Turbo LoRA**: `lightx2v/Minimax-h3-Turbo` -> `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (1.82 GiB)
- **Shared VAEs**: same Video/Audio VAEs.

### D. EasyColab Native DiffSynth Engines (G4 96GB + 176GB DDR5)

| Model ID | HuggingFace Repo | Size | Capabilities |
|---|---|---|---|
| `minimax_h3` | `MiniMaxAI/MiniMax-H3` | ~134.1 GB | BF16 T2VA & FLF2V (4–15s, 24fps + 32kHz stereo), dual-backbone RAM hot-standby + Qwen3-VL prompt cache |
| `minimax_h3_image` | `MiniMaxAI/MiniMax-H3-Image` | ~135.3 GB | 6-step native 4K (16MP) T2I & reference editing; auto-shares 32B `text_encoder` with `minimax_h3` |

---

## 2. ComfyUI Node Wiring Summary

### `MiniMaxH3ReferenceToVideo` (Multi-Image + Audio + Video Reference)
- **Inputs**: `clip`, `vae` (optional, anchors images in DiT latent space), `audio_vae` (optional, encodes reference audio for DiT), `prompt`, `width`, `height`, `length`, `ref_image_size` (`match` | `max`), `ref_images.ref_image_0..8` (max 9), `ref_videos.ref_video_0..2` (max 3), `ref_video_audios.ref_video_audio_0..2` (max 3), `ref_audios.ref_audio_0..2` (max 3).
- **Outputs**: `(0: CONDITIONING, 1: LATENT)`.

### `MiniMaxH3ImageToVideo` (First/Last Frame)
- **Inputs**: `clip`, `vae`, `first_frame` (optional), `last_frame` (optional), `prompt`, `width`, `height`, `length`.
- **Outputs**: `(0: CONDITIONING, 1: LATENT)`.

### `LoadH3Corrector` & `ApplyH3Corrector`
- `LoadH3Corrector(corrector="s3_corrector.safetensors")` -> `H3_CORRECTOR`
- `ApplyH3Corrector(model, corrector, strength=1.0, budget_fraction=0.5, velocity_floor=0.001, start_step=0, end_step=-1)` -> `MODEL` (hooked via `post_cfg` after `MiniMaxH3SigmaShift` and before `CFGGuider`).

---

## 3. EasyColab G4 Cold-Start & Drive Architecture (`app.py`)

When running on Google Colab with Google Drive mounted (`drive_symlink` mode):
- **NVMe Hybrid Model Symlinks**: `ComfyUI/models/*` uses file-level symlinks preferring `/content/fast_comfy_models` on local NVMe SSD, falling back to Drive FUSE only for non-staged files (avoids 25 MB/s FUSE `mmap` stalls).
- **22GB Disk Guard**: Local NVMe staging & direct-to-NVMe downloads check `free > size + 22 GB` and back up asynchronously to Drive via `_async_backup_to_drive`.
- **Custom Node Single-Archive Bundle**: `custom_nodes_bundle.tar.gz` packs plugins (excluding `.git` & `__pycache__`) for fast sequential restore on new VMs.
- **G4 Launch Flags**: `--highvram --disable-mmap --fast` on VRAM >= 64 GB (`--disable-mmap` on lower VRAM under `drive_symlink`).

---

## 4. EasyColab REST API Reference

All admin routes require HTTP Basic Auth (`Authorization: Basic base64("easycolab:<password>")`) and matching `Origin` header when browser-like origin is sent.

- **Status & Native Models**:
  - `GET /minimax_api/status?model_id=...`
  - `POST /minimax_api/download_model` (`{"model_id": "...", "hf_token": ""}`)
  - `POST /minimax_api/load_model` (`{"model_id": "...", "precision_mode": "bf16", "offload_mode": "disk"}`)
  - `POST /minimax_api/unload_model` (`{"model_id": "all"}`)
  - `POST /minimax_api/generate` -> returns `job_id`
  - `GET /minimax_api/job/{job_id}` -> poll until `completed` / `failed`
  - `GET /minimax_api/video/{filename}` | `GET /minimax_api/media/{filename}`
  - `POST /minimax_api/disconnect`
- **ComfyUI Studio**:
  - `GET /minimax_api/comfyui/status`
  - `POST /minimax_api/comfyui/start` (`{"unload_existing": false, "force_restart": false}`)
  - `POST /minimax_api/comfyui/stop`
  - `POST /minimax_api/comfyui/free_vram`
  - `POST /minimax_api/comfyui/sync_drive`
  - `POST /minimax_api/comfyui/install_pack` (`{"pack_id": "minimax_h3_10eros_max", "auto_start": true}`)
  - `POST /minimax_api/comfyui/download_weight` (`{"url_or_repo": "...", "filename": "...", "subfolder": "loras"}`)
  - `GET /minimax_api/comfyui/output_media/{rel_path}`
