"""
Data models, static catalogs, resolution presets, ComfyUI workflow packs,
i18n translation helpers, and path/string utilities for the EasyColab Windows Client
(1:1 parity with macOS, defaulting to Simplified Chinese "zh").
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Core i18n State & Translation Helpers (Default: Simplified Chinese "zh") ─

_CURRENT_LANG: str = "zh"


def get_lang() -> str:
    return _CURRENT_LANG


def is_en() -> bool:
    return _CURRENT_LANG == "en"


def set_lang(lang: str) -> None:
    global _CURRENT_LANG
    _CURRENT_LANG = "en" if str(lang).strip().lower().startswith("en") else "zh"


def tr(zh: str, en: str) -> str:
    return en if _CURRENT_LANG == "en" else zh


_SHARD_DESC_EXACT: Dict[str, str] = {
    "3D 视频 VAE BF16 (9.7 GB)": "3D Video VAE BF16 (9.7 GB)",
    "32kHz 立体声 VAE BF16 (0.6 GB)": "32kHz Stereo Audio VAE BF16 (0.6 GB)",
    "Unsloth UD-Q4_K_M 主模型 (16.5 GB)": "Unsloth UD-Q4_K_M Main Weights (16.5 GB)",
    "多模态视觉投影器 (0.93 GB)": "Multimodal Vision Projector (0.93 GB)",
    "Unsloth MTP 多词元投机解码头 (1.31 GB)": "Unsloth MTP Speculative Head (1.31 GB)",
    "图像 VAE": "Image VAE",
}

_SHARD_DESC_PREFIXES: List[Tuple[str, str]] = [
    ("DiT 音视频主干 BF16 ", "DiT A/V Backbone BF16 "),
    ("多模态编码器 BF16 ", "Multimodal Encoder BF16 "),
    ("权重分片 ", "Weight Shard "),
    ("编码器 ", "Text Encoder "),
]


def localize_shard_desc(desc: Optional[str]) -> Optional[str]:
    if desc is None or not is_en() or not desc:
        return desc
    if desc in _SHARD_DESC_EXACT:
        return _SHARD_DESC_EXACT[desc]
    out = desc
    for zh_p, en_p in _SHARD_DESC_PREFIXES:
        if out.startswith(zh_p):
            return en_p + out[len(zh_p) :]
    return out


_SERVER_EXACT_MAP: Dict[str, str] = {
    "尚未连接云端 GPU": "Cloud GPU Not Connected",
    "云端 ComfyUI 尚未启动": "Cloud ComfyUI not started",
    "Google Drive 自动软链持久化": "Google Drive Auto-Symlink Persistence",
    "Google Drive 实时软链挂载": "Google Drive Live Symlink Mounted",
    "Google Drive 云盘自动持久化": "Google Drive Cloud Auto-Persistence",
    "实例本地高速存储": "Instance Local High-Speed Storage",
    "尚未启动（支持一键拉起 ComfyUI + Google Drive 自动软链持久化）": (
        "Not started (1-click ComfyUI + Google Drive symlink persistence)"
    ),
    "ComfyUI 已停止（节点、模型与工作流已持久化保存至 Google Drive）": (
        "ComfyUI stopped (nodes, models & workflows persisted to Google Drive)"
    ),
    "正在初始化 ComfyUI 环境与 Google Drive 持久化目录...": (
        "Initializing ComfyUI environment & Google Drive persistence directories..."
    ),
    "正在释放现有常驻模型显存，为 ComfyUI 腾出完整 96GB 空间...": (
        "Releasing co-resident VRAM models to free full 96GB for ComfyUI..."
    ),
    "正在从 Google Drive 恢复历史持久化的 ComfyUI 节点、模型与工作流...": (
        "Restoring persisted ComfyUI custom nodes, models & workflows from Google Drive..."
    ),
    "正在克隆官方最新 ComfyUI 核心引擎...": "Cloning official ComfyUI core engine...",
    "正在安装 ComfyUI 核心依赖（保留当前 G4 Blackwell CUDA PyTorch）...": (
        "Installing ComfyUI core dependencies (preserving CUDA PyTorch)..."
    ),
    "正在恢复已持久化的自定义节点并配置 ComfyUI-Manager...": (
        "Restoring persisted custom nodes & configuring ComfyUI-Manager..."
    ),
    "正在检查并安装已保存自定义节点的 Python 依赖包...": (
        "Checking & installing Python dependencies for custom nodes..."
    ),
    "正在拉起 ComfyUI 节点推理引擎 (端口 8188)...": "Starting ComfyUI node engine (port 8188)...",
    "ComfyUI 内核已就绪，正在建立 Cloudflare 高速全屏画布隧道...": (
        "ComfyUI engine ready, opening Cloudflare fullscreen canvas tunnel..."
    ),
    "正在先从 Google Drive 检查已持久化的历史模型权重...": (
        "Checking persisted model weights in Google Drive..."
    ),
    "正在将配齐的模型与中文工作流持久化同步至 Google Drive...": (
        "Syncing installed models & workflows to Google Drive..."
    ),
    "正在预装并校验 llama.cpp CUDA 高速推理引擎...": "Verifying llama.cpp CUDA inference engine...",
    "正在映射 MiniMax-H3 BF16 多模态编码器权重 (14 分片)...": (
        "Mapping MiniMax-H3 BF16 multimodal encoder weights (14 shards)..."
    ),
    "正在映射 MiniMax-H3 BF16 33B DiT 主干权重 (13 分片)...": (
        "Mapping MiniMax-H3 BF16 33B DiT backbone weights (13 shards)..."
    ),
    "正在映射 3D 视频 VAE 与 32kHz 立体声 VAE 权重...": (
        "Mapping 3D Video VAE & 32kHz Stereo Audio VAE weights..."
    ),
    "正在构建 MiniMax-H3 BF16 原生 GPU 计算图与 Processor...": (
        "Building MiniMax-H3 native BF16 GPU graph & Processor..."
    ),
    "正在映射 Qwen-Image-2.1 权重并构建 GPU 管线...": "Mapping Qwen-Image-2.1 weights & building GPU pipeline...",
    "正在映射 Z-Image-Turbo 权重并构建 8 步极速生图管线...": (
        "Mapping Z-Image-Turbo weights & building 8-step turbo pipeline..."
    ),
    "正在预热 Qwen3.8-27B-UD-Q4_K_M.gguf 与 mmproj 视觉权重...": (
        "Warming up Qwen3.8-27B-UD-Q4_K_M.gguf & mmproj vision weights..."
    ),
    "正在自动补齐 Unsloth MTP 投机解码加速头 (1.31 GB)...": (
        "Fetching Unsloth MTP speculative decoding head (1.31 GB)..."
    ),
    "正在检查账号下已有的 Colab 实例...": "Checking existing Colab runtimes on your account...",
    "旧实例已失效或正在销毁，正在自动清理并重新分配新实例...": (
        "Previous runtime expired; cleaning up and allocating a fresh runtime..."
    ),
    "容器仍在初始化，继续尝试连接内核...": "Container still initializing, retrying kernel connection...",
    "正在上传最新多模型开源服务端代码 (app.py)...": "Uploading multi-model server engine (app.py)...",
    "正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...": (
        "Configuring Google Drive persistence, engines & tunnel (~40-90s)..."
    ),
    "尚未授权登录 Google 账号，请先点击「绑定账号」": "Google account not authorized. Please click Bind first.",
    "已释放全部云端实例": "All cloud runtimes released",
    "已释放全部云端实例。": "All cloud runtimes released.",
}

_SERVER_REGEX_RULES: List[Tuple[re.Pattern[str], object]] = [
    (
        re.compile(r"^已无缝接管在线 ComfyUI 画布（(.+?)）$"),
        lambda m: f"Reconnected to active ComfyUI canvas ({localize_server_text(m.group(1))})",
    ),
    (
        re.compile(r"^ComfyUI 进程已退出 \(code=([^)]+)\)，可点击重新启动$"),
        r"ComfyUI process exited (code=\1). Click to restart.",
    ),
    (
        re.compile(r"^正在建立 (.+?) 目录软链 \(models / workflows / output\)\.\.\.$"),
        lambda m: f"Linking {localize_server_text(m.group(1))} directories (models / workflows / output)...",
    ),
    (
        re.compile(r"^正在 8 线程极速下载 (.+?) 至 (.+?)\.\.\.$"),
        r"8-thread downloading \1 to \2...",
    ),
    (
        re.compile(r"^正在下载 (.+?) 至 (.+?)\.\.\.$"),
        r"Downloading \1 to \2...",
    ),
    (
        re.compile(r"^正在准备配齐「(.+?)」（共 (\d+) 个官方权重）\.\.\.$"),
        r'Preparing workflow pack "\1" (\2 official weights)...',
    ),
    (
        re.compile(r"^正在生成「(.+?)」全中文三步傻瓜注释工作流\.\.\.$"),
        r'Generating guided workflow templates for "\1"...',
    ),
    (
        re.compile(r"^正在为 (.+?) 启动 8 路并发 HTTP Range 极速下载通道\.\.\.$"),
        r"Starting 8-thread HTTP Range downloader for \1...",
    ),
    (
        re.compile(r"^\[(\d+/\d+)\] (.+?) 已就绪 \((.+?)\)，跳过下载$"),
        r"[\1] \2 ready (\3), skipping",
    ),
    (
        re.compile(r"^\[(\d+/\d+)\] 正在 8 线程并发下载 (.+?) \((.+?)\)\.\.\.$"),
        lambda m: f"[{m.group(1)}] 8-thread downloading {m.group(2)} ({localize_shard_desc(m.group(3))})...",
    ),
    (
        re.compile(r"^正在获取 (.+?) 的分词器与 Processor 配置文件\.\.\.$"),
        r"Fetching tokenizer & processor configs for \1...",
    ),
    (
        re.compile(r"^正在评估显存余量并装载 (.+?) 至 GPU\.\.\.$"),
        r"Checking free VRAM and loading \1 onto GPU...",
    ),
    (
        re.compile(r"^\[显存常驻加速\] 正在将子模块 (.+?) 预热锁入 GPU 显存\.\.\.$"),
        r"[VRAM Pinning] Locking submodule \1 into GPU VRAM...",
    ),
    (
        re.compile(r"^正在向 Google Colab 申请分配 (.+?) 实例\.\.\.$"),
        r"Requesting Google Colab \1 runtime...",
    ),
    (
        re.compile(r"^成功分配 (.+?) 云端虚拟机！正在等待容器内核就绪\.\.\.$"),
        r"Allocated \1 cloud VM! Waiting for kernel startup...",
    ),
    (
        re.compile(r"^当前已分配实例型号为 \[(.+?)\]，正在释放并切换为您选择的 \[(.+?)\]\.\.\.$"),
        r"Current runtime is [\1], releasing and switching to [\2]...",
    ),
    (
        re.compile(r"^正在检查已有实例 \[(.+?)\] \((.+?)\) 连通性\.\.\.$"),
        r"Checking connectivity of existing runtime [\1] (\2)...",
    ),
    (
        re.compile(r"^启动超时，日志尾部:\s*(.*)$"),
        r"Startup timed out. Recent logs: \1",
    ),
]


def localize_server_text(text: str) -> str:
    """Translate common backend / bridge status strings when UI is in English mode."""
    if not text or not is_en():
        return text
    stripped = text.strip()
    if stripped in _SERVER_EXACT_MAP:
        return _SERVER_EXACT_MAP[stripped]
    for pattern, repl in _SERVER_REGEX_RULES:
        if pattern.search(stripped):
            return pattern.sub(repl, stripped)  # type: ignore[arg-type]
    out = stripped
    for zh_p, en_p in _SHARD_DESC_PREFIXES:
        if zh_p in out:
            out = out.replace(zh_p, en_p)
    for zh_exact, en_exact in _SHARD_DESC_EXACT.items():
        if zh_exact in out:
            out = out.replace(zh_exact, en_exact)
    return out


# ─── Data Models & Dynamic Bilingual Catalogs ────────────────────────────────


class ModelModality(str, Enum):
    VIDEO_AUDIO = "video_audio"
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    LLM_VLM = "llm_vlm"
    COMFY_UI = "comfyui"


@dataclass(frozen=True)
class TaskModeItem:
    id: str
    title_zh: str
    title_en: str
    subtitle_zh: str
    subtitle_en: str
    icon: str

    @property
    def title(self) -> str:
        return tr(self.title_zh, self.title_en)

    @property
    def subtitle(self) -> str:
        return tr(self.subtitle_zh, self.subtitle_en)


class ModelFileItem:
    def __init__(
        self,
        name: str,
        desc: Optional[str] = None,
        exists: bool = False,
        size_gb: float = 0.0,
        desc_en: Optional[str] = None,
    ) -> None:
        self.name: str = name
        self._desc_zh: Optional[str] = desc
        self._desc_en: Optional[str] = desc_en
        self.exists: bool = exists
        self.size_gb: float = size_gb

    @property
    def desc(self) -> Optional[str]:
        if is_en():
            return self._desc_en if self._desc_en else localize_shard_desc(self._desc_zh)
        return self._desc_zh

    def clone(self) -> "ModelFileItem":
        return ModelFileItem(
            name=self.name,
            desc=self._desc_zh,
            exists=self.exists,
            size_gb=self.size_gb,
            desc_en=self._desc_en,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ModelFileItem":
        return cls(
            name=str(d.get("name", "")),
            desc=d.get("desc"),
            exists=bool(d.get("exists", False)),
            size_gb=float(d.get("size_gb", 0.0) or 0.0),
        )


@dataclass
class OpenModelSpec:
    id: str
    name_zh: str
    name_en: str
    repo_id: str
    badge_zh: str
    badge_en: str
    summary_zh: str
    summary_en: str
    approx_size_gb: float
    icon: str
    modality: ModelModality
    default_steps: int
    modes: List[TaskModeItem]
    default_files: List[ModelFileItem]
    include_pattern: str = ""
    is_custom: bool = False

    @property
    def name(self) -> str:
        return tr(self.name_zh, self.name_en)

    @property
    def badge(self) -> str:
        return tr(self.badge_zh, self.badge_en)

    @property
    def summary(self) -> str:
        return tr(self.summary_zh, self.summary_en)

    @classmethod
    def from_custom_dict(cls, d: dict) -> "OpenModelSpec":
        mod_raw = str(d.get("modality") or "llm_vlm")
        if mod_raw == "video_audio":
            mod = ModelModality.VIDEO_AUDIO
            icon = "film.stack"
            steps = 35
            modes = [
                TaskModeItem(
                    id="t2va",
                    title_zh="文生音视频",
                    title_en="Text to Video+Audio",
                    subtitle_zh="输入镜头与声音描述，同步生成视频与立体声音轨",
                    subtitle_en="Describe visuals and sound to generate video with synced stereo audio",
                    icon="film",
                ),
                TaskModeItem(
                    id="fl2va",
                    title_zh="首尾帧生视频",
                    title_en="First/Last Frame to Video",
                    subtitle_zh="上传首帧或尾帧参考图，精准控制起止画面与运镜过渡",
                    subtitle_en="Upload start/end reference frames for precise composition & camera transitions",
                    icon="photo.on.rectangle.angled",
                ),
            ]
        elif mod_raw == "image":
            mod = ModelModality.IMAGE
            icon = "sparkles.rectangle.stack"
            steps = 20
            modes = [
                TaskModeItem(
                    id="t2i",
                    title_zh="高清文生图",
                    title_en="HD Text-to-Image",
                    subtitle_zh="自定义 HuggingFace 图像生成管线",
                    subtitle_en="Custom HuggingFace image generation pipeline",
                    icon="photo.artframe",
                ),
                TaskModeItem(
                    id="i2i_edit",
                    title_zh="图生图风格重绘",
                    title_en="Image-to-Image Restyle",
                    subtitle_zh="上传参考底图并调节重绘幅度，保留构图生成全新细节",
                    subtitle_en="Upload a reference image and adjust denoising strength to restyle",
                    icon="wand.and.rays",
                ),
            ]
        elif mod_raw == "audio":
            mod = ModelModality.AUDIO
            icon = "waveform.badge.mic"
            steps = 1
            modes = [
                TaskModeItem(
                    id="voice_design",
                    title_zh="自然语言声音设计",
                    title_en="Natural Voice Design",
                    subtitle_zh="用自然语言描述捏造全新角色声线",
                    subtitle_en="Design a character voice from plain text description",
                    icon="waveform",
                ),
                TaskModeItem(
                    id="voice_clone",
                    title_zh="声纹克隆与情感配音",
                    title_en="Voice Clone + Emotion",
                    subtitle_zh="上传参考音频并附加情绪指令合成语音",
                    subtitle_en="Clone a reference clip with emotion instructions",
                    icon="mic",
                ),
            ]
        else:
            mod = ModelModality.LLM_VLM
            icon = "shippingbox.fill"
            steps = 20
            modes = [
                TaskModeItem(
                    id="chat",
                    title_zh="深度推理对话",
                    title_en="Deep Reasoning Chat",
                    subtitle_zh="自定义 HuggingFace 模型对话、长思维链推理与代码创作",
                    subtitle_en="Custom HuggingFace model chat, CoT reasoning & coding",
                    icon="bubble.left.and.bubble.right",
                ),
                TaskModeItem(
                    id="vlm",
                    title_zh="图像视觉理解",
                    title_en="Vision Understanding",
                    subtitle_zh="拖入图片进行画面解析、OCR 识别或反推提示词",
                    subtitle_en="Drop images for scene analysis, OCR, or prompt reverse-engineering",
                    icon="eye",
                ),
                TaskModeItem(
                    id="api",
                    title_zh="API 协议网关",
                    title_en="API Protocol Gateway",
                    subtitle_zh="提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                    subtitle_en="Public dual-protocol gateway compatible with OpenAI (/v1) & Anthropic (/v1/messages)",
                    icon="network",
                ),
            ]
        repo_id = str(d.get("repo_id") or "")
        name = str(d.get("name") or repo_id.split("/")[-1] or "Custom Model")
        size_gb = float(d.get("approx_size_gb") or 0.0)
        arch = str(d.get("pipeline_arch") or "gguf").upper()
        inc = str(d.get("include_pattern") or "")
        badge_zh = f"自定义 {arch} · {size_gb:.1f} GB" if size_gb > 0 else f"自定义 HF · {arch}"
        badge_en = f"Custom {arch} · {size_gb:.1f} GB" if size_gb > 0 else f"Custom HF · {arch}"
        pat_desc = f" ({inc})" if inc else ""
        summary_zh = str(d.get("category") or f"自定义 HuggingFace 仓库 {repo_id}{pat_desc}")
        summary_en = f"Custom HuggingFace repo {repo_id}{pat_desc}"
        return cls(
            id=str(d.get("id") or f"custom_{uuid.uuid4().hex[:8]}"),
            name_zh=name,
            name_en=name,
            repo_id=repo_id,
            badge_zh=badge_zh,
            badge_en=badge_en,
            summary_zh=summary_zh,
            summary_en=summary_en,
            approx_size_gb=size_gb,
            icon=icon,
            modality=mod,
            default_steps=steps,
            modes=modes,
            default_files=[],
            include_pattern=inc,
            is_custom=True,
        )


@dataclass(frozen=True)
class ResolutionOption:
    id: str
    label_zh: str
    label_en: str
    width: int
    height: int

    @property
    def label(self) -> str:
        return tr(self.label_zh, self.label_en)


@dataclass(frozen=True)
class PresetTemplate:
    modality: ModelModality
    title_zh: str
    title_en: str
    visual_zh: str
    visual_en: str = ""
    soundscape: str = ""
    music: str = ""

    @property
    def title(self) -> str:
        return tr(self.title_zh, self.title_en)

    @property
    def visual(self) -> str:
        return tr(self.visual_zh, self.visual_en or self.visual_zh)


@dataclass
class LoadedModelSummary:
    id: str
    name: str
    mode: Optional[str] = None
    vram_gb: Optional[float] = None
    context_length: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "LoadedModelSummary":
        raw_id = str(d.get("id", ""))
        raw_name = strip_emoji(str(d.get("name", "")))
        if raw_id in MODEL_SPEC_MAP:
            raw_name = MODEL_SPEC_MAP[raw_id].name
        return cls(
            id=raw_id,
            name=raw_name,
            mode=strip_emoji(str(d["mode"])) if d.get("mode") else None,
            vram_gb=float(d["vram_gb"]) if d.get("vram_gb") is not None else None,
            context_length=int(d["context_length"]) if d.get("context_length") is not None else None,
        )


@dataclass
class ComfyOutputMediaItem:
    rel_path: str
    filename: str
    media_type: str = "video"
    size_mb: float = 0.0
    mtime: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "ComfyOutputMediaItem":
        return cls(
            rel_path=str(d.get("rel_path", "")),
            filename=str(d.get("filename", "")),
            media_type=str(d.get("media_type") or "video"),
            size_mb=float(d.get("size_mb") or 0.0),
            mtime=int(d.get("mtime") or 0),
        )


_COMFY_PACK_I18N: Dict[str, Tuple[str, str, str, str, str, str]] = {
    "minimax_h3_10eros_max": (
        "10Eros-Max (MiniMax-H3 无限制) · 多图参考 + 首尾帧 8步极速",
        "10Eros-Max (Uncensored H3) · Multi-Image Reference + FLF2V 8-Step",
        "无内容限制 · 多图参考 8步",
        "Uncensored · Multi-Ref 8-Step",
        "基于 TenStrip/10Eros-Max v1.0 与无审核 Heretic 编码器，强化 <Picture 1>/<Picture 2> 多图参考与首尾帧一致性，内置 8 步 Turbo 加速",
        "Built on TenStrip/10Eros-Max v1.0 & uncensored Heretic encoder for <Picture 1>/<Picture 2> multi-image reference & FLF2V in 8 turbo steps",
    ),
    "minimax_h3_sulphur3": (
        "Sulphur-3 (MiniMax-H3 无限制) · 满血底座 + H3 效果修正器",
        "Sulphur-3 (Uncensored H3) · Full Backbone + H3 Corrector",
        "无内容限制 · 含修正器",
        "Uncensored · With Corrector",
        "内置 SulphurAI/Sulphur-3 满血主干 + 无审核 Heretic 编码器 + s3_corrector 速度残差修正器与专用节点，支持文生/图生有声视频并大幅增强动态与画质",
        "Full SulphurAI/Sulphur-3 backbone + uncensored Heretic encoder + s3_corrector residual network & auto-installed nodes for superior fidelity",
    ),
    "wan22_remix_nsfw": (
        "Wan2.2-Remix 无限制 · 14B NSFW 图生/首尾帧 (v3.0 8步)",
        "Wan2.2-Remix (Uncensored 14B) · I2V + FLF2V (v3.0 8-Step)",
        "无内容限制 · 极致写实动态",
        "Uncensored · 8-Step Realism",
        "基于 FX-FeiHou/wan2.2-Remix NSFW I2V v3.0 (High+Low FP8 Lightning) + NSFW-Wan-UMT5-XXL 编码器，100% 内置节点 8 步高写实生成",
        "FX-FeiHou/wan2.2-Remix NSFW I2V v3.0 (High+Low FP8 Lightning) + NSFW-Wan-UMT5-XXL encoder, 100% built-in nodes in 8 steps",
    ),
    "minimax_h3_drama_director": (
        "MiniMax-H3 短剧导演 · 角色声纹多镜头 + 分镜表成片 (10Eros-Max 8步)",
        "MiniMax-H3 Drama Director · Multi-Shot + Storyboard (10Eros-Max 8-Step)",
        "短剧时间线 · 音画无缝续拍",
        "Drama Timeline · Seamless Multi-Shot",
        "集成 ComfyUI-MiniMaxH3-Director (NGH3Director + NGStoryboard) 并复用 10Eros-Max 8步无限制底座；支持角色图卡、声纹卡与音画无缝长视频续拍",
        "ComfyUI-MiniMaxH3-Director (NGH3Director + NGStoryboard) on uncensored 10Eros-Max 8-step with character/voice cards & seamless multi-shot chaining",
    ),
    "qwen3_tts_voice_design": (
        "Qwen3-TTS 1.7B · 自然语言声音设计 + 情感声纹克隆 (Apache-2.0)",
        "Qwen3-TTS 1.7B · Natural Voice Design + Emotional Voice Clone",
        "声音设计 · 情感配音",
        "Voice Design · Emotional Clone",
        "支持零音频用中文/英文自然语言捏造角色声线 (VoiceDesign)，并支持 3~15s 声纹克隆与情绪指令 (Base)",
        "Design brand-new character voices from text (VoiceDesign 1.7B) or clone 3–15s speech with emotion cues (Base 1.7B)",
    ),
    "drama3_tts": (
        "Fish Audio Drama 3 · 35B 戏剧级情感配音与 16 角色飙戏 (含 Qwen3-ASR)",
        "Fish Audio Drama 3 · 35B MoE Dramatic TTS + Qwen3-ASR",
        "35B MoE · 舞台指令",
        "35B MoE · Stage Cues",
        "集成 PedroMarinhoDev/ComfyUI-Drama3（零 pip 依赖），支持 Qwen3-ASR 自动转录、[sighs]/[whispers]/[laughs] 舞台指示与多角色互听飙戏",
        "Zero-pip ComfyUI-Drama3 (2026-10-29 FP8) + Qwen3-ASR 1.7B auto-transcription, [sighs]/[whispers]/[laughs] cues & 16-speaker dialogue",
    ),
}


@dataclass
class ComfyPresetPackItem:
    id: str
    _title: str
    _badge: str
    _scenario: str
    approx_size_gb: float
    workflows: List[str]
    installed_models: int = 0
    total_models: int = 0
    workflows_installed: bool = False
    ready: bool = False

    @property
    def title(self) -> str:
        entry = _COMFY_PACK_I18N.get(self.id)
        if entry:
            return tr(entry[0], entry[1])
        return localize_server_text(self._title)

    @property
    def badge(self) -> str:
        entry = _COMFY_PACK_I18N.get(self.id)
        if entry:
            return tr(entry[2], entry[3])
        return localize_server_text(self._badge)

    @property
    def scenario(self) -> str:
        entry = _COMFY_PACK_I18N.get(self.id)
        if entry:
            return tr(entry[4], entry[5])
        return localize_server_text(self._scenario)

    @property
    def is_primary_recommended(self) -> bool:
        return self.id == "minimax_h3_10eros_max"

    @classmethod
    def from_dict(cls, d: dict) -> "ComfyPresetPackItem":
        return cls(
            id=str(d.get("id", "")),
            _title=str(d.get("title", "")),
            _badge=str(d.get("badge", "")),
            _scenario=str(d.get("scenario", "")),
            approx_size_gb=float(d.get("approx_size_gb") or 0.0),
            workflows=list(d.get("workflows") or []),
            installed_models=int(d.get("installed_models") or 0),
            total_models=int(d.get("total_models") or 0),
            workflows_installed=bool(d.get("workflows_installed", False)),
            ready=bool(d.get("ready", False)),
        )


@dataclass
class LocalMediaItem:
    path: Path
    filename: str
    is_image: bool
    created_ts: float
    elapsed_seconds: Optional[float] = None


@dataclass
class ChatTurnItem:
    role: str  # "user" | "assistant"
    content: str
    thinking: str = ""
    has_image: bool = False
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ─── Static Catalogs (1:1 Parity with macOS EasyColabStudio.swift) ────────────

OPEN_MODEL_CATALOG: List[OpenModelSpec] = [
    OpenModelSpec(
        id="minimax_h3_drama_director",
        name_zh="MiniMax-H3 短剧导演 · 多镜头连拍",
        name_en="MiniMax-H3 Drama Director · Multi-Shot",
        repo_id="seesee75-commits/ComfyUI-MiniMaxH3-Director",
        badge_zh="短剧时间线 · 复用 10Eros-Max · 8步",
        badge_en="Drama Timeline · 10Eros-Max · 8-Step",
        summary_zh="集成 ComfyUI-MiniMaxH3-Director 与 10Eros-Max 无限制 8 步底座，支持角色定妆卡、声纹锁定与音画无缝续拍",
        summary_en="ComfyUI-MiniMaxH3-Director on uncensored 10Eros-Max 8-step with character/voice cards & seamless AV multi-shot continuity",
        approx_size_gb=39.13,
        modality=ModelModality.VIDEO_AUDIO,
        modes=[
            TaskModeItem("drama_multishot", "多镜头短剧连拍", "Multi-Shot Drama Timeline", "按时间线分镜表渲染任意时长短剧，自动音画无缝衔接", "Render multi-shot short dramas on a timeline with seamless AV continuity"),
            TaskModeItem("drama_storyboard", "角色声纹分镜表", "Character + Storyboard", "上传角色图卡与声纹参考，搭配分镜面板生成连续有声短剧", "Pair character refs & locked voice with storyboard panels for cohesive drama"),
        ],
    ),
    OpenModelSpec(
        id="qwen3_tts_voice_design",
        name_zh="Qwen3-TTS 1.7B · 声音设计",
        name_en="Qwen3-TTS 1.7B · Voice Design",
        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        badge_zh="文字捏声线 + 情感克隆 · 9.7 GB",
        badge_en="Voice Design + Clone · 9.7 GB",
        summary_zh="Apache-2.0 开源自然语言声音设计 (VoiceDesign 1.7B) + 零样本情感克隆 (Base 1.7B)，可生成声纹种子喂给 H3 导演",
        summary_en="Apache-2.0 natural Voice Design (1.7B) + zero-shot emotional cloning (Base 1.7B) to forge voice seeds for H3 Director",
        approx_size_gb=9.74,
        modality=ModelModality.AUDIO,
        default_steps=1,
        modes=[
            TaskModeItem("voice_design", "自然语言声音设计", "Natural Voice Design", "零参考音频，用自然语言描述音色、年龄与剧情感现捏声纹", "Zero-audio voice synthesis: describe timbre, age & emotion in plain text", "waveform"),
            TaskModeItem("voice_clone", "声纹克隆与情感配音", "Voice Clone + Emotion", "上传 3~15 秒人声参考 (<Audio 1>) 并附加情绪指令生成配音", "Clone 3–15s speech (<Audio 1>) with emotion directives", "mic"),
        ],
    ),
    OpenModelSpec(
        id="drama3_tts",
        name_zh="Drama 3 · 35B 戏剧情感配音",
        name_en="Fish Audio Drama 3 · 35B TTS",
        repo_id="PedroMarinhoDev/Drama-3-ComfyUI",
        badge_zh="35B MoE · 舞台指令 + 多角色 · 44.2 GB",
        badge_en="35B MoE · Stage Cues + Dialogue · 44.2 GB",
        summary_zh="Fish Audio Drama 3 35B MoE (FP8) + Codec + Qwen3-ASR 1.7B 自动转录，支持 [sighs]/[whispers]/[laughs] 舞台指示与多角色飙戏",
        summary_en="Fish Audio Drama 3 35B MoE (FP8) + Codec + Qwen3-ASR 1.7B: inline stage cues & multi-speaker acting",
        approx_size_gb=44.23,
        modality=ModelModality.AUDIO,
        default_steps=1,
        modes=[
            TaskModeItem("drama_tts_clone", "角色声纹戏剧配音 (含 ASR)", "Character Acting (ASR Clone)", "上传 3~20 秒参考声纹，Qwen3-ASR 自动转录并支持舞台标签", "Upload 3–20s speech, auto-transcribed by Qwen3-ASR with stage cues", "theatermasks"),
            TaskModeItem("drama_tts_free", "无参考自由配音 (多角色脚本)", "Voice-Free Script", "使用 <|speaker:0|>/<|speaker:1|> 与情绪标签直接生成多角色对白", "Multi-speaker dialogue with <|speaker:0|>/<|speaker:1|> & cues", "waveform"),
        ],
    ),
    OpenModelSpec(
        id="qwen38_27b_uncensored_gguf",
        name_zh="Qwen3.8-27B 无限制 · GGUF",
        name_en="Qwen3.8-27B Uncensored · GGUF",
        repo_id="JonathanColetti/Qwen3.8-27B-Uncensored-GGUF",
        badge_zh="Heretic 去审查 + MTP · 17.8 GB",
        badge_en="Heretic Uncensored + MTP · 17.8 GB",
        summary_zh="Heretic BF16 去审查 Qwen3.8-27B Q4_K_M + F16 视觉投影器 + MTP 投机解码头，256K 上下文无限制对话/看图反推/API 网关",
        summary_en="Heretic BF16-abliterated Qwen3.8-27B Q4_K_M + F16 vision mmproj + MTP draft head for uncensored Chat, Vision & API gateway",
        approx_size_gb=17.83,
        icon="cpu",
        modality=ModelModality.LLM_VLM,
        default_steps=20,
        modes=[
            TaskModeItem(
                id="chat",
                title_zh="无限制深度推理对话",
                title_en="Uncensored Reasoning Chat",
                subtitle_zh="Heretic 去审查 + MTP 投机解码，用于短剧剧本、分镜策划与无限制对话",
                subtitle_en="Heretic uncensored + MTP speculative decoding for scripts, storyboards & chat",
                icon="bubble.left.and.bubble.right",
            ),
            TaskModeItem(
                id="vlm",
                title_zh="无限制看图反推 (VLM)",
                title_en="Uncensored Vision Prompting",
                subtitle_zh="传图无拒答解析构图、姿态服饰并反推生图/生视频 Prompt",
                subtitle_en="Uncensored photo analysis & image/video prompt reverse-engineering",
                icon="eye",
            ),
            TaskModeItem(
                id="api",
                title_zh="API 协议网关",
                title_en="API Protocol Gateway",
                subtitle_zh="提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                subtitle_en="Dual-protocol public gateway supporting OpenAI (/v1) & Anthropic (/v1/messages)",
                icon="network",
            ),
        ],
        default_files=[
            ModelFileItem(
                name="Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf",
                desc="Heretic 去审查 Q4_K_M 主模型 (15.41 GB)",
                desc_en="Heretic Uncensored Q4_K_M Main Weights (15.41 GB)",
            ),
            ModelFileItem(
                name="mmproj-Qwen3.8-27B-Uncensored-F16.gguf",
                desc="F16 视觉投影器 mmproj (0.86 GB)",
                desc_en="F16 Vision Projector mmproj (0.86 GB)",
            ),
            ModelFileItem(
                name="Qwen3.8-27B-Uncensored-draft-Q4_0.gguf",
                desc="MTP 投机解码草稿头 Q4_0 (1.56 GB)",
                desc_en="MTP Speculative Draft Head Q4_0 (1.56 GB)",
            ),
        ],
    ),
    OpenModelSpec(
        id="comfyui_studio",
        name_zh="ComfyUI 节点工作台",
        name_en="ComfyUI Node Studio",
        repo_id="comfyanonymous/ComfyUI",
        badge_zh="节点视频创作 · 云盘软链持久化",
        badge_en="Node Workflows · Drive Symlink",
        summary_zh="一键拉起云端 ComfyUI + Manager 全屏节点画布，自定义节点、视频权重与工作流自动软链持久化至 Google Drive",
        summary_en="One-click cloud ComfyUI + Manager fullscreen node canvas with auto Google Drive symlink persistence for nodes, models & workflows",
        approx_size_gb=0.0,
        icon="point.3.connected.trianglepath.dotted",
        modality=ModelModality.COMFY_UI,
        default_steps=25,
        modes=[
            TaskModeItem(
                id="comfy_canvas",
                title_zh="全屏节点画布",
                title_en="Fullscreen Node Canvas",
                subtitle_zh="一键部署并跳转浏览器全屏节点画布，复用 G4 保活、折线监控与自动休眠",
                subtitle_en="Deploy & open fullscreen node canvas in browser with G4 keepalive, telemetry & auto-sleep",
                icon="macwindow.badge.plus",
            ),
            TaskModeItem(
                id="comfy_assets",
                title_zh="极速下权重与回传成片",
                title_en="Fast Weight DL & Output Pull",
                subtitle_zh="8 线程极速下载视频模型至云盘 models 目录，并将 output 成片一键回传本机",
                subtitle_en="8-thread high-speed model download to Drive and 1-click output media sync to local PC",
                icon="externaldrive.badge.icloud",
            ),
        ],
        default_files=[],
    ),
]

MODEL_SPEC_MAP: Dict[str, OpenModelSpec] = {m.id: m for m in OPEN_MODEL_CATALOG}

VIDEO_RESOLUTIONS: List[ResolutionOption] = [
    ResolutionOption(
        "v_832_480",
        "832 × 480 (16:9 横屏标准 · 推荐)",
        "832 × 480 (16:9 Landscape · Recommended)",
        832,
        480,
    ),
    ResolutionOption(
        "v_480_832",
        "480 × 832 (9:16 竖屏短视频)",
        "480 × 832 (9:16 Vertical Short)",
        480,
        832,
    ),
    ResolutionOption(
        "v_1280_720",
        "1280 × 720 (16:9 720P 高清)",
        "1280 × 720 (16:9 720P HD)",
        1280,
        720,
    ),
    ResolutionOption(
        "v_720_1280",
        "720 × 1280 (9:16 720P 竖屏)",
        "720 × 1280 (9:16 720P Vertical)",
        720,
        1280,
    ),
    ResolutionOption(
        "v_640_640",
        "640 × 640 (1:1 方形)",
        "640 × 640 (1:1 Square)",
        640,
        640,
    ),
]

IMAGE_RESOLUTIONS: List[ResolutionOption] = [
    ResolutionOption(
        "i_1024_1024",
        "1024 × 1024 (1:1 超清方图 · 推荐)",
        "1024 × 1024 (1:1 Square HD · Recommended)",
        1024,
        1024,
    ),
    ResolutionOption(
        "i_1344_768",
        "1344 × 768 (16:9 电影横幅)",
        "1344 × 768 (16:9 Cinematic Wide)",
        1344,
        768,
    ),
    ResolutionOption(
        "i_768_1344",
        "768 × 1344 (9:16 竖版海报)",
        "768 × 1344 (9:16 Vertical Poster)",
        768,
        1344,
    ),
    ResolutionOption(
        "i_1536_1024",
        "1536 × 1024 (3:2 单反摄影)",
        "1536 × 1024 (3:2 DSLR Photo)",
        1536,
        1024,
    ),
]

PRESET_TEMPLATES: List[PresetTemplate] = [
    PresetTemplate(
        modality=ModelModality.VIDEO_AUDIO,
        title_zh="[动漫] 新海诚风 · 盛夏樱花铁道口",
        title_en="[Anime] Makoto Shinkai · Cherry Blossom Railway Crossing",
        visual_zh=(
            "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor "
            "school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past "
            "in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink "
            "cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, "
            "vibrant cinematic lighting, smooth 24fps keyframe animation."
        ),
        soundscape=(
            "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning "
            "bell, warm spring wind rustling through leaves, a soft gasp from the girl."
        ),
        music="Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings.",
    ),
    PresetTemplate(
        modality=ModelModality.VIDEO_AUDIO,
        title_zh="[动漫] 飞碟社风 · 苍蓝雷光剑术对决",
        title_en="[Anime] ufotable Sakuga · Blue Lightning Sword Duel",
        visual_zh=(
            "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime "
            "swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, "
            "swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and "
            "glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur."
        ),
        soundscape=(
            "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, "
            "heavy stone debris crumbling, fierce battle cry."
        ),
        music="Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs.",
    ),
    PresetTemplate(
        modality=ModelModality.VIDEO_AUDIO,
        title_zh="[写实] 冰岛巨型瀑布 FPV 穿越机航拍",
        title_en="[Realistic] Icelandic Waterfall FPV Drone Dive",
        visual_zh=(
            "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt "
            "columns under golden hour sunlight, mist rising into a rainbow."
        ),
        soundscape="Roaring waterfall rushing water, wind whooshing past the camera, distant seabirds calling.",
        music="Epic cinematic acoustic guitar and soaring cello.",
    ),
    PresetTemplate(
        modality=ModelModality.IMAGE,
        title_zh="[生图] 东方赛博霓虹商业海报（含中文排版）",
        title_en="[Image] Cyberpunk Shanghai Neon Poster (Typography)",
        visual_zh=(
            "超清商业电影海报摄影，雨夜霓虹闪烁的未来上海外滩街道，一位身着机能风银色夹克的少女回眸凝视镜头，"
            "景深虚化光斑，顶部醒目发光现代中文字体排版写着“未来纪元 2026”，极致光影细节，8K画质。"
        ),
        visual_en=(
            "Ultra-HD commercial movie poster photography, futuristic rain-slicked Shanghai Bund street glowing with "
            "neon reflections at night, a young woman in a sleek silver techwear jacket looking back over her shoulder "
            'at the camera, creamy bokeh light orbs, bold glowing modern title typography at the top reading "FUTURE ERA 2026", '
            "cinematic lighting, 8K detail."
        ),
    ),
    PresetTemplate(
        modality=ModelModality.IMAGE,
        title_zh="[生图] 胶片质感午后咖啡馆肖像",
        title_en="[Image] Medium Format Film Afternoon Tokyo Cafe",
        visual_zh=(
            "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden "
            "blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam "
            "beside an open sketchbook, natural film grain, ultra-detailed textures."
        ),
    ),
    PresetTemplate(
        modality=ModelModality.LLM_VLM,
        title_zh="[推理] 为 AI 视频导演生成 3 组新海诚风分镜提示词",
        title_en="[Reasoning] Write 3 Makoto Shinkai Anime Storyboard Prompts",
        visual_zh=(
            "请帮我构思 3 组极具画面感的新海诚风格动画短片分镜，每组包含："
            "1) 英文 Visual & Camera 镜头提示词；2) Soundscape 环境音描述；3) Background Music 配乐描述。"
        ),
        visual_en=(
            "Please design 3 vivid Makoto Shinkai-style anime short film storyboards. For each shot include: "
            "1) Visual & Camera prompt; 2) Soundscape description; 3) Background Music description."
        ),
    ),
    PresetTemplate(
        modality=ModelModality.LLM_VLM,
        title_zh="[视觉] 深度解析图片构图并反推生图/生视频 Prompt",
        title_en="[Vision] Analyze Composition & Reverse-Engineer T2I/T2V Prompt",
        visual_zh="请仔细观察这张图片的主体、光影色调、镜头景别与艺术风格，先做简要中文分析，然后反推出一段高质量的英文生图/生视频 Prompt。",
        visual_en=(
            "Carefully examine the subject, lighting, color grading, camera framing, and artistic style of this image. "
            "Provide a brief breakdown first, then reverse-engineer a detailed English T2I / T2V generation prompt."
        ),
    ),
]


def build_default_comfy_preset_packs() -> List[ComfyPresetPackItem]:
    return [
        ComfyPresetPackItem(
            id="minimax_h3_10eros_max",
            _title="10Eros-Max (MiniMax-H3 无限制) · 多图参考 + 首尾帧 8步极速",
            _badge="无内容限制 · 多图参考 8步",
            _scenario="基于 TenStrip/10Eros-Max v1.0 与无审核 Heretic 编码器，强化 <Picture 1>/<Picture 2> 多图参考与首尾帧一致性，内置 8 步 Turbo 加速",
            approx_size_gb=39.1,
            workflows=[
                "07_10Eros-Max_无限制多图参考有声视频(8步极速).json",
                "08_10Eros-Max_无限制首尾帧图生有声视频(8步极速).json",
            ],
            installed_models=0,
            total_models=4,
        ),
        ComfyPresetPackItem(
            id="minimax_h3_sulphur3",
            _title="Sulphur-3 (MiniMax-H3 无限制) · 满血底座 + H3 效果修正器",
            _badge="无内容限制 · 含修正器",
            _scenario="内置 SulphurAI/Sulphur-3 满血主干 + 无审核 Heretic 编码器 + s3_corrector 速度残差修正器与专用节点，支持文生/图生有声视频并大幅增强动态与画质",
            approx_size_gb=62.7,
            workflows=[
                "09_Sulphur-3_无限制文生有声视频(含H3修正器).json",
                "10_Sulphur-3_无限制图生首尾帧有声视频(含H3修正器).json",
            ],
            installed_models=0,
            total_models=5,
        ),
        ComfyPresetPackItem(
            id="moss_image_edit",
            _title="MOSS-Image 无限制 · 文生图 & 指令修图 (Heretic + armV5 + NSFW LoRA)",
            _badge="无限制 · 3图指令修图",
            _scenario="内置 MOSS-Image SFT BF16 + Heretic 无审核 Qwen3.5-9B + Ostris Edit (armV5) + NSFW Uncensored LoRA，一次配齐无限制文生图与修图",
            approx_size_gb=19.4,
            workflows=[
                "14_MOSS-Image_无限制文生图(Heretic+NSFW).json",
                "15_MOSS-Image_无限制指令修图(OstrisEdit+armV5).json",
            ],
            installed_models=0,
            total_models=5,
        ),
        ComfyPresetPackItem(
            id="wan22_remix_nsfw",
            _title="Wan2.2-Remix 无限制 · 14B NSFW 图生/首尾帧 (v3.0 8步)",
            _badge="无内容限制 · 极致写实动态",
            _scenario="基于 FX-FeiHou/wan2.2-Remix NSFW I2V v3.0 (High+Low FP8 Lightning) + NSFW-Wan-UMT5-XXL 编码器，纯内置节点 8 步高写实生成",
            approx_size_gb=33.13,
            workflows=[
                "16_Wan2.2-Remix_无限制单图生视频(v3.0_8步极速).json",
                "17_Wan2.2-Remix_无限制首尾帧过渡(v3.0_8步极速).json",
            ],
            installed_models=0,
            total_models=4,
        ),
        ComfyPresetPackItem(
            id="minimax_h3_drama_director",
            _title="MiniMax-H3 短剧导演 · 角色声纹多镜头 + 分镜表成片 (10Eros-Max 8步)",
            _badge="短剧时间线 · 音画无缝续拍",
            _scenario="集成 ComfyUI-MiniMaxH3-Director (NGH3Director + NGStoryboard) 并复用 10Eros-Max 8步无限制底座；支持角色图卡、声纹卡与音画无缝长视频续拍",
            approx_size_gb=39.13,
            workflows=[
                "18_MiniMax-H3_短剧导演_角色声纹多镜头(10Eros-Max_8步).json",
                "19_MiniMax-H3_短剧导演_双角色分镜表成片(10Eros-Max_8步).json",
            ],
            installed_models=0,
            total_models=5,
        ),
        ComfyPresetPackItem(
            id="qwen3_tts_voice_design",
            _title="Qwen3-TTS 1.7B · 自然语言声音设计 + 情感声纹克隆 (Apache-2.0)",
            _badge="声音设计 · 情感配音",
            _scenario="支持零音频用中文/英文自然语言捏造角色声线 (VoiceDesign)，并支持 3~15s 声纹克隆与情绪指令 (Base)",
            approx_size_gb=9.74,
            workflows=[
                "20_Qwen3-TTS_自然语言声音设计(VoiceDesign_1.7B).json",
                "21_Qwen3-TTS_声纹克隆与情感配音(Base_1.7B).json",
            ],
            installed_models=0,
            total_models=3,
        ),
        ComfyPresetPackItem(
            id="drama3_tts",
            _title="Fish Audio Drama 3 · 35B 戏剧级情感配音与 16 角色飙戏 (含 Qwen3-ASR)",
            _badge="35B MoE · 舞台指令",
            _scenario="集成 PedroMarinhoDev/ComfyUI-Drama3（零 pip 依赖），支持 Qwen3-ASR 自动转录、[sighs]/[whispers]/[laughs] 舞台指示与多角色互听飙戏",
            approx_size_gb=44.23,
            workflows=[
                "22_Drama3_35B情感配音与音色克隆(自动ASR转录).json",
                "23_Drama3_35B多角色情感飙戏与舞台指令.json",
            ],
            installed_models=0,
            total_models=3,
        ),
    ]


COMFY_WEIGHT_SUBFOLDERS: List[str] = [
    "diffusion_models",
    "checkpoints",
    "loras",
    "vae",
    "text_encoders",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "h3_correctors",
    "audio_encoders",
    "asr",
    "qwen-tts",
    "gguf",
]

GPU_CHOICES = [
    ("G4", "G4 · PRO 6000"),
    ("H100", "H100 · 80GB"),
    ("A100", "A100 · 80GB"),
    ("L4", "L4 · 24GB"),
    ("T4", "T4 · 16GB"),
]


def get_idle_timeout_choices() -> List[Tuple[int, str]]:
    return [
        (15, tr("15 分钟", "15 min")),
        (30, tr("30 分钟", "30 min")),
        (60, tr("60 分钟", "60 min")),
        (0, tr("关闭", "Off")),
    ]


IDLE_TIMEOUT_CHOICES = [
    (15, "15 分钟"),
    (30, "30 分钟"),
    (60, "60 分钟"),
    (0, "关闭"),
]

VIDEO_DURATION_PRESETS = [
    (90, "4s"),
    (124, "5s"),
    (192, "8s"),
    (243, "10s"),
    (294, "12s"),
    (362, "15s"),
]


# ─── String & Path Helpers ───────────────────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE0F"
    "\U0000200D"
    "\U000020E3"
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    if not text:
        return ""
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def snap_video_frames(val: int) -> int:
    snapped = int(round((val - 5) / 17.0)) * 17 + 5
    return max(90, min(362, snapped))


def format_duration(sec: float, compact: bool = False) -> str:
    s = max(0.0, float(sec))
    if s < 60.0:
        return f"{s:.1f}s" if (compact or is_en()) else f"{s:.1f} 秒"
    mins = int(s) // 60
    rem = s - mins * 60
    if compact:
        return f"{mins}m{rem:.0f}s"
    if is_en():
        return f"{mins}m {rem:.1f}s"
    return f"{mins}分{rem:.1f}秒"


def format_uptime_hms(seconds: int) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    rem = s % 60
    return f"{h:02d}:{m:02d}:{rem:02d}"


def mask_url_display(text: str, mask_enabled: bool) -> str:
    if not mask_enabled or not text:
        return text
    out = re.sub(r"https?://[a-zA-Z0-9.-]+\.gradio\.live", "https://••••••••••••.gradio.live", text)
    out = re.sub(r"https?://[a-zA-Z0-9.-]+\.trycloudflare\.com", "https://••••••••••••.trycloudflare.com", out)
    return out


def mask_email_display(email: str, mask_enabled: bool) -> str:
    if not email:
        return ""
    if not mask_enabled or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    prefix = local[:2] if len(local) >= 2 else local[:1]
    return f"{prefix}•••@{domain}"


def get_project_or_bundle_root() -> Path:
    """Locate repository root (in source mode) or PyInstaller bundle directory (in .exe mode)."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_resource_path(filename: str) -> Path:
    """Find bundled resource (app.py, colab_bridge.py, EasyColab_Server.ipynb, AppIcon.ico)."""
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / filename)
        candidates.append(exe_dir / "_internal" / filename)
        if hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS) / filename)
    win_dir = Path(__file__).resolve().parent
    repo_root = win_dir.parent
    candidates.extend([
        repo_root / filename,
        win_dir / filename,
        win_dir / "assets" / filename,
    ])
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def get_app_config_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) / "EasyColab" if base else (Path.home() / "AppData" / "Local" / "EasyColab")
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "EasyColab"
    else:
        root = Path.home() / ".config" / "EasyColab"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_local_save_dir() -> Path:
    """Default local media output directory: ~/Videos/EasyColab on Windows/Linux, ~/Movies/EasyColab on Mac."""
    if sys.platform == "darwin":
        save_dir = Path.home() / "Movies" / "EasyColab"
    else:
        xdg_videos = os.environ.get("XDG_VIDEOS_DIR", "").strip()
        if xdg_videos and Path(xdg_videos).is_dir():
            videos_root = Path(xdg_videos)
        else:
            videos_root = Path.home() / "Videos"
            if not videos_root.exists():
                videos_root = Path(os.path.expanduser("~")) / "Videos"
        save_dir = videos_root / "EasyColab"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_legacy_media_dirs() -> List[Path]:
    parent = get_local_save_dir().parent
    dirs = [get_local_save_dir()]
    for name in ("Aevix", "Colab_AI_Studio", "MiniMax_H3_Videos"):
        d = parent / name
        if d.is_dir():
            dirs.append(d)
    return dirs


def open_path_in_os(path: Path, reveal: bool = False) -> None:
    """Open file/folder in Explorer (Windows), Finder (macOS), or File Manager (Linux)."""
    import subprocess

    target = Path(path).resolve()
    try:
        if sys.platform == "win32":
            if reveal and target.is_file():
                subprocess.Popen(["explorer", f"/select,{str(target)}"])
            else:
                os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            if reveal and target.is_file():
                subprocess.Popen(["open", "-R", str(target)])
            else:
                subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target.parent if reveal and target.is_file() else target)])
    except Exception:
        pass
