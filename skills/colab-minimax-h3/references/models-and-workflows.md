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

### B2. Motion Transfer ComfyUI Pack: `minimax_h3_ref2va_tara` (~40.9 GB)

- **Main DiT**: `Comfy-Org/MiniMax-H3` -> `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (`models/diffusion_models/`, 20.97 GiB)
- **Encoder**: `Comfy-Org/MiniMax-H3` -> `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (`models/text_encoders/`, 14.61 GiB)
- **Turbo LoRA**: `Comfy-Org/MiniMax-H3` -> `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (`models/loras/`, 1.96 GiB)
- **Motion Transfer TARA LoRA**: `Alissonerdx/Minimax-H3-ComfyUI` -> `minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors` (`models/loras/`, 0.13 GiB)
- **Custom Nodes (auto-cloned)**: `ComfyUI-BFSNodes` (`https://github.com/alisson-anjos/ComfyUI-BFSNodes.git`) -> provides `BFSMiniMaxH3NativeTemporal`
- **Shared VAEs**: same Video/Audio VAEs.
- **TARA Defaults**: `w=0.20s`, `f=0.25s`, `strength=2.0`, `max_penalty=6.0`, `attention_backend="flash partitions (no compile)"`, **8 steps** (`Euler / simple`, `CFG=1.0`), 24fps guide video grid.

### B4. Fast Streaming Video Super-Resolution Pack: `recavsr_upscaler` (~17.3 GB)

- **Main Transformer (BF16 default)**: `kopper/ReCaVSR` -> `transformer.safetensors` (`models/recavsr/`, 10.6 GB)
- **Fast INT8 ConvRot Variant**: `PedroMarinhoDev/ReCaVSR-ComfyUI` -> `recavsr/transformer_int8_convrot.safetensors` (`models/recavsr/`, 6.3 GB, ~1.8× faster)
- **FlashDecoder & Prompt Embed**: `kopper/ReCaVSR` -> `flashdecoder.safetensors` (0.23 GB) & `prompt.safetensors` (0.17 GB) in `models/recavsr/`
- **Custom Nodes (auto-cloned)**: `ComfyUI-ReCaVSR` (`https://github.com/PedroMarinhoDev/ComfyUI-ReCaVSR.git`) -> `ReCaVSRLoader`, `ReCaVSRUpscale`
- **Preset Workflow**: `13_ReCaVSR_Wan2.2_极速流式4K超分.json` (`LoadVideo -> ReCaVSRLoader -> ReCaVSRUpscale -> SaveVideo`)
- **Recommended Settings**: `scale=2.0` for 1056×608 / 720p AI clips (~2K), `scale=4.0` for 480p/540p clips (native 4× 4K), `decoder="flashdecoder"`, `kernels="fast"`, `color_fix="none"`, `encoder="auto"` (NVENC hardware encode + audio/FPS passthrough).

### B5. Uncensored Image Gen & Edit Pack: `moss_image_edit` (~19.4 GB)

- **Main Diffusion**: `Comfy-Org/MOSS-Image` -> `moss_image_sft_bf16.safetensors` (`models/diffusion_models/`, 9.47 GB)
- **Uncensored Encoder**: `pottokao/MOSS-Image-Text-Encoder-Heretic` -> `moss_te_heretic_full_int8_convrot.safetensors` (`models/text_encoders/`, 9.08 GB, decensored Qwen3.5-9B keeping full vision tower)
- **VAE**: `Comfy-Org/MOSS-Image` -> `flux2-vae.safetensors` (`models/vae/`, 0.31 GB)
- **Edit LoRA**: `akhaliq/MOSS-Image-Edit-HQ` -> `moss_edit_armV5.safetensors` (`models/loras/`, 0.46 GB, ai-toolkit format + `kv_cache=True`)
- **Uncensored NSFW LoRA**: `axiomofmind/Moss-image-nsfw-uncensored-lora` -> `MOSS-Image-NSFW-Uncensored-LoRA-ComfyUI.safetensors` (`models/loras/`, 0.07 GB, optional trigger `moss_xvstyle`)
- **Custom Nodes (auto-cloned)**: `ComfyUI-MOSS-Image-Ostris-Edit` (`https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit.git`) -> `TextEncodeMOSSImageOstrisEdit`, `MOSSImageOstrisEditModelPatch`
- **Preset Workflows**:
  - `14_MOSS-Image_无限制文生图(Heretic+NSFW).json`
  - `15_MOSS-Image_无限制指令修图(OstrisEdit+armV5).json`
- **Recommended Settings**: `Euler / simple`, **30–40 steps**, `CFG = 3.0`, `SkipLayerGuidanceDiTSimple(single_layers="22")`, `armV5` edit LoRA strength `1.0`, NSFW LoRA strength `0.65` (max 3 ref images via `image1..image3`).

### B6. Uncensored Fast I2V / FLF2V Pack: `wan22_remix_nsfw` (~33.1 GB)

- **High/Low Diffusion (v3.0 Lighting 8-step)**: `FX-FeiHou/wan2.2-Remix` -> `NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors` & `low` (`models/diffusion_models/`, 13.31 GB each)
- **Uncensored Text Encoder**: `NSFW-API/NSFW-Wan-UMT5-XXL` -> `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` (`models/text_encoders/`, 6.27 GB, `CLIPLoader(type="wan")`)
- **VAE**: `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` -> `split_files/vae/wan_2.1_vae.safetensors` (`models/vae/`, 0.24 GB)
- **Custom Nodes**: None (100% pure built-in ComfyUI nodes: `ModelSamplingSD3`, `WanImageToVideo`, `WanFirstLastFrameToVideo`, `KSamplerAdvanced`, `CreateVideo`, `SaveVideo`)
- **Preset Workflows**:
  - `16_Wan2.2-Remix_无限制单图生视频(v3.0_8步极速).json`
  - `17_Wan2.2-Remix_无限制首尾帧过渡(v3.0_8步极速).json`
- **Recommended Settings**: `Euler / simple`, **8 steps**, `split_step = 4` (`0->4` high, `4->10000` low), `ModelSamplingSD3 shift = 8.0`, `CFG = 1.0`, frames `4k+1` (`81 ≈ 5s` @ `16fps`).

### B7. MiniMax-H3 Short-Drama Timeline Studio: `minimax_h3_drama_director` (~39.13 GB shared + 20 MB TAE)

- **Shared Weights**: Reuses `10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors` (both `model_fl2va` and `model_ref2va`), `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors`, `minimax_h3_video_vae_int8_convrot.safetensors`, `minimax_h3_audio_vae_fp32.safetensors` (+ `vae_approx/taeh3.safetensors` from `OzzyGT/taeh3`).
- **Custom Nodes (auto-cloned)**: `ComfyUI-MiniMaxH3-Director` (`https://github.com/seesee75-commits/ComfyUI-MiniMaxH3-Director.git`)
- **Preset Workflows**:
  - `18_MiniMax-H3_短剧导演_角色声纹多镜头(10Eros-Max_8步).json` (`NGContextImage` + `NGContextAudio` + `NGH3Director` + `NGProjectRender`)
  - `19_MiniMax-H3_短剧导演_双角色分镜表成片(10Eros-Max_8步).json` (dual characters + `NGStoryboard`)
- **Recommended Settings**: `turbo=False` (8-step already baked into `10Eros-Max`), `LCM / simple`, **8 steps**, `Shift 12/3`, `continuity="video_audio"`, `pin_frames="22"`, beat script `0-7: @Hero ...` and `@Hero says: ...`.

### B8. Natural Language Voice Design + Emotional Clone: `qwen3_tts_voice_design` (~9.74 GB)

- **HF Snapshots (`models/qwen-tts/`)**: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (4.52 GB), `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (4.54 GB), `Qwen/Qwen3-TTS-Tokenizer-12Hz` (0.68 GB). Apache-2.0.
- **Custom Nodes (auto-cloned)**: `ComfyUI-Qwen-TTS` (`https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git`, filtered to skip `onnxruntime-openvino`/`sox`).
- **Preset Workflows**:
  - `20_Qwen3-TTS_自然语言声音设计(VoiceDesign_1.7B).json` (`FB_Qwen3TTSVoiceDesign` -> `SaveAudio`)
  - `21_Qwen3-TTS_声纹克隆与情感配音(Base_1.7B).json` (`LoadAudio` -> `FB_Qwen3TTSVoiceClone` -> `SaveAudio`)
- **Drama Pipeline Synergy**: Generate clean 8–15s character voice seeds in `VoiceDesign` from pure text descriptions, then attach to `MiniMax-H3 Director` (`NGContextAudio`) or `Drama 3`.

### B9. Dramatic Multi-Speaker Emotional TTS: `drama3_tts` (~44.23 GB)

- **Weights**: `PedroMarinhoDev/Drama-3-ComfyUI/drama3_fp8experts_bf16.safetensors` (`models/diffusion_models/`, 38.58 GB), `drama3_codec.safetensors` (`models/vae/`, 1.57 GB), `PedroMarinhoDev/Qwen3-ASR-ComfyUI/qwen3_asr_1.7b_bf16.safetensors` (`models/asr/`, 4.08 GB).
- **Custom Nodes (auto-cloned)**: `ComfyUI-Drama3` (`https://github.com/PedroMarinhoDev/ComfyUI-Drama3.git`, zero pip deps).
- **Preset Workflows**:
  - `22_Drama3_35B情感配音与音色克隆(自动ASR转录).json` (`Drama3Loader` + `Drama3CodecLoader` + `Drama3LoadReferenceAudio` + `Qwen3ASRTranscribe` + `Drama3EncodeVoice` + `Drama3Generate` + `Drama3Decode` + `SaveAudio`)
  - `23_Drama3_35B多角色情感飙戏与舞台指令.json` (dual-speaker `voices.voice0` + `voices.voice1`, `speakers="one take per line, hearing the line before"`, `voice_guidance=1.0`).
- **Critical Drama 3 Settings**: Keep `top_k=0`, `temperature=1.0`, `top_p=0.9` to avoid mid-pause silence traps; use `[sighs]`, `[laughs]`, `[whispers]`, `[gasp]` and `<|speaker:0..15|>` tags.

### C1. Uncensored LLM/VLM Gateway: `qwen38_27b_uncensored_gguf` (~17.8 GB)

- **Repo**: `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` (Apache-2.0, Heretic BF16-abliterated)
- **Files**: `Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf` (15.41 GB), `mmproj-Qwen3.8-27B-Uncensored-F16.gguf` (0.86 GB), `Qwen3.8-27B-Uncensored-draft-Q4_0.gguf` (1.56 GB, MTP draft head)
- **Usage**: Load via native workspace or `/minimax_api/model/load` (`qwen38_27b_uncensored_gguf`), then call `/v1/chat/completions` for uncensored scriptwriting, shot breakdown, or VLM reverse-prompting.

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
