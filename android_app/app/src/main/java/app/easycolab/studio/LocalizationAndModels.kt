package app.easycolab.studio

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.UUID

enum class AppLanguage(val code: String) {
    ZH("zh"),
    EN("en")
}

object LanguageManager {
    var lang by mutableStateOf(AppLanguage.ZH)
        private set

    val isEN: Boolean get() = lang == AppLanguage.EN

    fun init(context: Context) {
        val saved = context.getSharedPreferences("easycolab_prefs", Context.MODE_PRIVATE)
            .getString("appLanguage", "zh")
        lang = if (saved == "en") AppLanguage.EN else AppLanguage.ZH
    }

    fun setLanguage(newLang: AppLanguage, context: Context? = null) {
        lang = newLang
        context?.getSharedPreferences("easycolab_prefs", Context.MODE_PRIVATE)
            ?.edit()?.putString("appLanguage", newLang.code)?.apply()
    }

    fun toggle(context: Context? = null) {
        setLanguage(if (lang == AppLanguage.ZH) AppLanguage.EN else AppLanguage.ZH, context)
    }
}

fun L(zh: String, en: String): String = if (LanguageManager.lang == AppLanguage.EN) en else zh

fun stripEmoji(text: String): String {
    if (text.isEmpty()) return ""
    val sb = StringBuilder(text.length)
    var i = 0
    while (i < text.length) {
        val cp = text.codePointAt(i)
        val count = Character.charCount(cp)
        val isEmoji = (cp in 0x1F300..0x1FAFF)
            || (cp in 0x2600..0x27BF)
            || cp == 0xFE0F || cp == 0x200D || cp == 0x20E3
            || (cp in 0x1F1E6..0x1F1FF)
        if (!isEmoji) {
            sb.appendCodePoint(cp)
        }
        i += count
    }
    return sb.toString().replace(Regex("\\s{2,}"), " ").trim()
}

fun localizeModelFileDesc(raw: String): String {
    if (!LanguageManager.isEN || raw.isEmpty()) return raw
    return raw
        .replace(" [RAM内存盘]", " [RAM Disk]")
        .replace("DiT 音视频主干 BF16", "DiT Audio-Video Backbone BF16")
        .replace("多模态编码器 BF16", "Multimodal Encoder BF16")
        .replace("3D 视频 VAE BF16", "3D Video VAE BF16")
        .replace("32kHz 立体声 VAE BF16", "32kHz Stereo Audio VAE BF16")
        .replace("权重分片", "Weight Shard")
        .replace("Unsloth UD-Q4_K_M 主模型", "Unsloth UD-Q4_K_M Main Model")
        .replace("多模态视觉投影器", "Multimodal Vision Projector")
        .replace("Unsloth MTP 多词元投机解码头", "Unsloth MTP Speculative Decoding Head")
        .replace("编码器", "Encoder")
        .replace("图像 VAE", "Image VAE")
}

fun localizeServerText(text: String): String {
    if (text.isEmpty()) return text
    val adaptedZh = text
        .replace("Mac 客户端", "App 客户端")
        .replace("左侧", "算力页/上方")
    if (!LanguageManager.isEN) return adaptedZh

    val exactMap = mapOf(
        "未加载" to "Not Loaded",
        "未连接 Colab 服务端" to "Colab Server Disconnected",
        "尚未启动云端 ComfyUI 工作台" to "Cloud ComfyUI Studio Not Started",
        "Google Drive 自动软链持久化" to "Google Drive Auto-Symlink Persistence",
        "Google Drive 实时软链挂载" to "Google Drive Live Symlink Mount",
        "Google Drive 云盘自动持久化" to "Google Drive Auto-Persistence",
        "实例本地高速存储" to "Instance Local NVMe Storage",
        "已彻底释放全部云端实例，计费速率归零。" to "All cloud Colab instances terminated. Billing rate is now 0 CU/h.",
        "已手动停止当前推理任务" to "Inference cancelled by user",
        "正在上传最新多模型开源服务端代码 (app.py)..." to "Uploading latest multi-model server engine (app.py)...",
        "正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)..." to "Configuring Google Drive persistence, multi-model engine & public tunnel (~40–90s)...",
        "正在克隆 DiffSynth-Studio..." to "Cloning DiffSynth-Studio..."
    )
    exactMap[text]?.let { return it }

    var s = localizeModelFileDesc(text)
    val replacements = listOf(
        "已无缝接管在线 ComfyUI 画布" to "Connected to active ComfyUI canvas",
        "ComfyUI 进程已退出" to "ComfyUI process exited",
        "，可点击重新启动" to "; tap to restart",
        "正在 8 线程极速下载 " to "8-thread fast downloading ",
        "正在 8 线程并发下载 " to "8-thread downloading ",
        "正在下载 " to "Downloading ",
        " 已就绪 (" to " ready (",
        ")，跳过下载" to "), skipping",
        "正在评估显存余量并装载 " to "Checking VRAM & loading ",
        " 至 GPU..." to " into GPU...",
        " 正在编码多模态上下文并启动推理..." to " encoding multimodal context & starting inference...",
        " 推理完成！耗时 " to " inference complete! Elapsed ",
        " 正在使用多模态编码器解析提示词" to " encoding prompts with multimodal encoder",
        "DiT 采样已完成，正在执行" to "DiT sampling finished, running ",
        "3D 视频 VAE 解码" to "3D Video VAE decoding",
        "视频帧解码完成，正在执行 32kHz 立体声音轨解码" to "Video frames decoded, decoding 32kHz stereo audio",
        " 正在执行去噪采样第 " to " denoising step ",
        " 步 (" to " (",
        "VAE 解码完成，正在合成 32kHz 立体声音轨并封装 MP4" to "VAE decoded, muxing 32kHz stereo audio into MP4",
        "正在写入无损 PNG 图像文件" to "Writing lossless PNG image",
        "生成完成！耗时 " to "Generation complete! Elapsed ",
        "生成出错: " to "Generation error: ",
        "尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。" to "is not loaded in Colab GPU VRAM. Tap 'Load to VRAM' first.",
        "尚未加载到显存" to "is not loaded in VRAM",
        "当前已常驻：" to "currently resident: ",
        "已通过 Google OAuth 自动挂载 Google Drive" to "Mounted Google Drive automatically via Google OAuth"
    )
    for ((zhSub, enSub) in replacements) {
        if (s.contains(zhSub)) s = s.replace(zhSub, enSub)
    }
    return s
}

fun formatDuration(sec: Double, compact: Boolean = false): String {
    val s = sec.coerceAtLeast(0.0)
    if (s < 60.0) {
        if (compact) return String.format(Locale.US, "%.1fs", s)
        return if (LanguageManager.isEN) String.format(Locale.US, "%.1f s", s)
        else String.format(Locale.US, "%.1f 秒", s)
    }
    val mins = (s / 60.0).toInt()
    val rem = s - mins * 60.0
    if (compact) return String.format(Locale.US, "%dm%.0fs", mins, rem)
    return if (LanguageManager.isEN) String.format(Locale.US, "%dm %.1fs", mins, rem)
    else String.format(Locale.US, "%d分%.1f秒", mins, rem)
}

enum class ModelModality(val raw: String) {
    VIDEO_AUDIO("video_audio"),
    IMAGE("image"),
    AUDIO("audio"),
    LLM_VLM("llm_vlm"),
    COMFY_UI("comfyui")
}

data class TaskModeItem(
    val id: String,
    val titleZh: String,
    val titleEn: String,
    val subtitleZh: String,
    val subtitleEn: String
) {
    val title: String get() = L(titleZh, titleEn)
    val subtitle: String get() = L(subtitleZh, subtitleEn)
}

data class ModelFileItem(
    val name: String,
    val desc: String? = null,
    val exists: Boolean = false,
    val sizeGb: Double = 0.0
) {
    val localizedDesc: String? get() = desc?.takeIf { it.isNotEmpty() }?.let { localizeModelFileDesc(it) }
}

data class OpenModelSpec(
    val id: String,
    val nameZh: String,
    val nameEn: String,
    val shortTitleZh: String,
    val shortTitleEn: String,
    val repoId: String,
    val badgeZh: String,
    val badgeEn: String,
    val summaryZh: String,
    val summaryEn: String,
    val approxSizeGb: Double,
    val modality: ModelModality,
    val defaultSteps: Float,
    val modes: List<TaskModeItem>,
    val defaultFiles: List<ModelFileItem>,
    val includePattern: String = "",
    val isCustom: Boolean = false
) {
    val name: String get() = L(nameZh, nameEn)
    val shortTitle: String get() = L(shortTitleZh, shortTitleEn)
    val badge: String get() = L(badgeZh, badgeEn)
    val summary: String get() = L(summaryZh, summaryEn)

    companion object {
        fun fromCustomJson(obj: JSONObject): OpenModelSpec {
            val modRaw = obj.optString("modality", "llm_vlm")
            val mod = when (modRaw) {
                "video_audio" -> ModelModality.VIDEO_AUDIO
                "image" -> ModelModality.IMAGE
                "audio" -> ModelModality.AUDIO
                else -> ModelModality.LLM_VLM
            }
            val repoId = obj.optString("repo_id", "")
            val rawName = obj.optString("name", repoId.substringAfterLast('/').ifEmpty { "Custom HF" })
            val sizeGb = obj.optDouble("approx_size_gb", 0.0).let { if (it.isNaN()) 0.0 else it }
            val arch = obj.optString("pipeline_arch", "gguf").ifEmpty { "gguf" }.uppercase(Locale.US)
            val inc = obj.optString("include_pattern", "")
            val badgeZh = if (sizeGb > 0.0) String.format(Locale.US, "自定义 %s · %.1f GB", arch, sizeGb) else "自定义 HF · $arch"
            val badgeEn = if (sizeGb > 0.0) String.format(Locale.US, "Custom %s · %.1f GB", arch, sizeGb) else "Custom HF · $arch"
            val patDesc = if (inc.isNotEmpty()) " ($inc)" else ""
            val summaryZh = obj.optString("category", "").ifEmpty { "自定义 HuggingFace 仓库 $repoId$patDesc" }
            val summaryEn = "Custom HuggingFace repo $repoId$patDesc"
            val (steps, modes) = when (mod) {
                ModelModality.VIDEO_AUDIO -> 35f to listOf(
                    TaskModeItem("t2va", "文生音视频", "Text to Video+Audio", "输入镜头与声音描述生成视频", "Generate HD video & stereo audio from prompts"),
                    TaskModeItem("fl2va", "首尾帧生视频", "Keyframes to Video", "从相册选择首帧/尾帧参考图", "Pick start/end keyframes from Gallery")
                )
                ModelModality.IMAGE -> 20f to listOf(
                    TaskModeItem("t2i", "高清文生图", "HD Text-to-Image", "自定义 HuggingFace 图像生成管线", "Custom HuggingFace image generation"),
                    TaskModeItem("i2i_edit", "图生图重绘", "Image Restyle", "从手机相册选参考图重绘", "Pick a photo and adjust denoising strength")
                )
                ModelModality.AUDIO -> 1f to listOf(
                    TaskModeItem("voice_design", "自然语言声音设计", "Voice Design", "零音频描述音色与剧情感捏造声纹", "Create a character voice from text"),
                    TaskModeItem("voice_clone", "声纹克隆情感配音", "Voice Clone", "上传参考音频并附加情感指令合成", "Clone speech with emotion instructions")
                )
                else -> 20f to listOf(
                    TaskModeItem("chat", "深度推理对话", "Reasoning Chat", "自定义 HuggingFace 模型对话与推理", "Custom HuggingFace model chat & reasoning"),
                    TaskModeItem("vlm", "看图反推/视觉", "Vision & Prompt", "多模态看图分析与提示词生成", "Multimodal photo analysis & prompt generation"),
                    TaskModeItem("api", "API 协议网关", "API Gateway", "OpenAI (/v1) 与 Anthropic (/v1/messages) 接口", "OpenAI (/v1) & Anthropic (/v1/messages) endpoints")
                )
            }
            return OpenModelSpec(
                id = obj.optString("id", "custom_" + UUID.randomUUID().toString().take(8)),
                nameZh = rawName,
                nameEn = rawName,
                shortTitleZh = rawName,
                shortTitleEn = rawName,
                repoId = repoId,
                badgeZh = badgeZh,
                badgeEn = badgeEn,
                summaryZh = summaryZh,
                summaryEn = summaryEn,
                approxSizeGb = sizeGb,
                modality = mod,
                defaultSteps = steps,
                modes = modes,
                defaultFiles = emptyList(),
                includePattern = inc,
                isCustom = true
            )
        }
    }
}

val OPEN_MODEL_CATALOG: List<OpenModelSpec> = listOf(
    OpenModelSpec(
        id = "minimax_h3_drama_director",
        nameZh = "MiniMax-H3 短剧导演 · 多镜头连拍",
        nameEn = "MiniMax-H3 Drama Director · Multi-Shot",
        shortTitleZh = "H3 短剧导演",
        shortTitleEn = "H3 Drama",
        repoId = "seesee75-commits/ComfyUI-MiniMaxH3-Director",
        badgeZh = "短剧时间线 · 复用 10Eros-Max",
        badgeEn = "Drama Timeline · 10Eros-Max",
        summaryZh = "集成 ComfyUI-MiniMaxH3-Director + 10Eros-Max 8步底座，支持角色定妆卡、声纹锁定与音画无缝长视频续拍",
        summaryEn = "ComfyUI-MiniMaxH3-Director on uncensored 10Eros-Max 8-step with character/voice cards & seamless AV continuity",
        approxSizeGb = 39.13,
        modality = ModelModality.VIDEO,
        defaultSteps = 8f,
        modes = listOf(
            TaskModeItem("drama_multishot", "多镜头短剧连拍", "Multi-Shot Drama", "按时间线分镜表渲染多镜头短剧，自动音画无缝续拍", "Render multi-shot short dramas with seamless AV chunk chaining"),
            TaskModeItem("drama_storyboard", "角色声纹分镜表", "Storyboard + Voice", "搭配角色图卡、声纹参考与分镜画板成片", "Pair character refs, voice card & storyboard panels into a cohesive short")
        ),
        defaultFiles = listOf(
            ModelFileItem("diffusion_models/10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors", "10Eros-Max 8步主干 (21.32 GB · 共享)"),
            ModelFileItem("vae_approx/taeh3.safetensors", "MiniMax-H3 预览 TAE (0.02 GB)")
        )
    ),
    OpenModelSpec(
        id = "qwen3_tts_voice_design",
        nameZh = "Qwen3-TTS 1.7B · 声音设计",
        nameEn = "Qwen3-TTS 1.7B · Voice Design",
        shortTitleZh = "Qwen3-TTS 音色",
        shortTitleEn = "Qwen3-TTS",
        repoId = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        badgeZh = "文字捏声线 + 情感克隆 · 9.7 GB",
        badgeEn = "Voice Design + Clone · 9.7 GB",
        summaryZh = "Apache-2.0 开源自然语言声音设计 (VoiceDesign 1.7B) + 零样本情感克隆 (Base 1.7B)",
        summaryEn = "Apache-2.0 natural Voice Design (1.7B) + zero-shot emotional cloning (Base 1.7B)",
        approxSizeGb = 9.74,
        modality = ModelModality.AUDIO,
        defaultSteps = 1f,
        modes = listOf(
            TaskModeItem("voice_design", "自然语言声音设计", "Voice Design", "零音频描述音色与情绪捏造角色声纹", "Describe voice & emotion from text without ref audio"),
            TaskModeItem("voice_clone", "声纹克隆情感配音", "Voice Clone", "上传参考音频并附加情感指令合成", "Clone speech with emotion instructions")
        ),
        defaultFiles = listOf(
            ModelFileItem("qwen-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign/model.safetensors", "Qwen3-TTS 1.7B 声音设计 (4.52 GB)"),
            ModelFileItem("qwen-tts/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors", "Qwen3-TTS 1.7B 声纹克隆 (4.54 GB)"),
            ModelFileItem("qwen-tts/Qwen3-TTS-Tokenizer-12Hz/model.safetensors", "Qwen3-TTS 12Hz Tokenizer (0.68 GB)")
        )
    ),
    OpenModelSpec(
        id = "drama3_tts",
        nameZh = "Drama 3 · 35B 戏剧情感配音",
        nameEn = "Fish Audio Drama 3 · 35B TTS",
        shortTitleZh = "Drama 3 配音",
        shortTitleEn = "Drama 3 TTS",
        repoId = "PedroMarinhoDev/Drama-3-ComfyUI",
        badgeZh = "35B MoE · 舞台指令 + 多角色",
        badgeEn = "35B MoE · Stage Cues + Dialogue",
        summaryZh = "Fish Audio Drama 3 35B MoE (FP8) + Codec + Qwen3-ASR 1.7B，支持 [sighs]/[whispers]/[laughs] 舞台指示与多角色飙戏",
        summaryEn = "Fish Audio Drama 3 35B MoE (FP8) + Codec + Qwen3-ASR 1.7B with [sighs]/[whispers]/[laughs] cues & dialogue",
        approxSizeGb = 44.23,
        modality = ModelModality.AUDIO,
        defaultSteps = 1f,
        modes = listOf(
            TaskModeItem("drama_tts_clone", "声纹戏剧配音 (ASR)", "Acting Clone (ASR)", "上传参考声纹，ASR 自动转录并支持舞台指示", "Upload voice clip with auto Qwen3-ASR & stage cues"),
            TaskModeItem("drama_tts_free", "无参考多角色配音", "Multi-Speaker Script", "使用 <|speaker:0|>/<|speaker:1|> 生成戏剧对白", "Synthesize multi-speaker dialogue with inline cues")
        ),
        defaultFiles = listOf(
            ModelFileItem("diffusion_models/drama3_fp8experts_bf16.safetensors", "Drama 3 35B FP8 权重 (38.58 GB)"),
            ModelFileItem("vae/drama3_codec.safetensors", "Drama 3 Codec (1.57 GB)"),
            ModelFileItem("asr/qwen3_asr_1.7b_bf16.safetensors", "Qwen3-ASR 1.7B (4.08 GB)")
        )
    ),
    OpenModelSpec(
        id = "qwen38_27b_uncensored_gguf",
        nameZh = "Qwen3.8-27B 无限制 · GGUF",
        nameEn = "Qwen3.8-27B Uncensored · GGUF",
        shortTitleZh = "Qwen3.8 无限制",
        shortTitleEn = "Qwen3.8 Uncensored",
        repoId = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF",
        badgeZh = "Heretic 去审查 + MTP · 17.8 GB",
        badgeEn = "Heretic Uncensored + MTP · 17.8 GB",
        summaryZh = "Heretic 去审查 Qwen3.8-27B Q4_K_M + F16 视觉投影器 + MTP 投机解码，支持 256K 上下文、看图反推与 API 网关",
        summaryEn = "Heretic abliterated Qwen3.8-27B Q4_K_M + F16 vision mmproj + MTP draft head for uncensored chat, vision & API gateway",
        approxSizeGb = 17.83,
        modality = ModelModality.LLM_VLM,
        defaultSteps = 20f,
        modes = listOf(
            TaskModeItem("chat", "无限制推理对话", "Uncensored Chat", "Heretic 去审查 + MTP 投机解码，用于短剧剧本与深度对话", "Heretic uncensored + MTP speculative decoding for scripts & reasoning"),
            TaskModeItem("vlm", "无限制看图反推", "Uncensored Vision", "传照片无拒答解析构图与反推生图/生视频提示词", "Analyze photos & reverse-engineer prompts without refusals"),
            TaskModeItem("api", "API 协议网关", "API Gateway", "OpenAI (/v1) 与 Anthropic (/v1/messages) 接口", "OpenAI (/v1) & Anthropic (/v1/messages) endpoints")
        ),
        defaultFiles = listOf(
            ModelFileItem("Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf", "Heretic 去审查 Q4_K_M 主模型 (15.41 GB)"),
            ModelFileItem("mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "F16 视觉投影器 mmproj (0.86 GB)"),
            ModelFileItem("Qwen3.8-27B-Uncensored-draft-Q4_0.gguf", "MTP 投机解码草稿头 Q4_0 (1.56 GB)")
        )
    ),
    OpenModelSpec(
        id = "comfyui_studio",
        nameZh = "ComfyUI 节点工作台",
        nameEn = "ComfyUI Node Studio",
        shortTitleZh = "ComfyUI 工作台",
        shortTitleEn = "ComfyUI Studio",
        repoId = "comfyanonymous/ComfyUI",
        badgeZh = "4套预置包 · 云盘持久化",
        badgeEn = "4 Preset Packs · Drive Sync",
        summaryZh = "一键拉起云端 ComfyUI + 4 套视频工作流包，支持内置画布预览与成片一键存手机相册",
        summaryEn = "1-click cloud ComfyUI + 4 preset video packs, Chrome Custom Tab canvas & 1-tap pull to Android Gallery",
        approxSizeGb = 0.0,
        modality = ModelModality.COMFY_UI,
        defaultSteps = 25f,
        modes = listOf(
            TaskModeItem("comfy_canvas", "云端工作流与画布", "Workflows & Canvas", "一键配齐 Wan2.2 / MiniMax / LTX / 混元预置包并打开画布", "Install Wan2.2 / MiniMax / LTX / Hunyuan packs & open canvas"),
            TaskModeItem("comfy_assets", "下权重与拉取成片", "Weights & Outputs", "8 线程极速下模型至云盘，并将 ComfyUI 成片一键拉回手机", "8-thread cloud weight download & 1-tap pull outputs to phone")
        ),
        defaultFiles = emptyList()
    )
)

data class ResolutionOption(
    val id: String,
    val labelZh: String,
    val labelEn: String,
    val width: Int,
    val height: Int
) {
    val label: String get() = L(labelZh, labelEn)
}

val VIDEO_RESOLUTIONS: List<ResolutionOption> = listOf(
    ResolutionOption("v_480_832", "480 × 832 (9:16 竖屏短视频 · 推荐)", "480 × 832 (9:16 Vertical Short · Pick)", 480, 832),
    ResolutionOption("v_832_480", "832 × 480 (16:9 横屏标准)", "832 × 480 (16:9 Landscape Standard)", 832, 480),
    ResolutionOption("v_720_1280", "720 × 1280 (9:16 720P 高清竖屏)", "720 × 1280 (9:16 720P HD Vertical)", 720, 1280),
    ResolutionOption("v_1280_720", "1280 × 720 (16:9 720P 高清横屏)", "1280 × 720 (16:9 720P HD Landscape)", 1280, 720),
    ResolutionOption("v_640_640", "640 × 640 (1:1 方形)", "640 × 640 (1:1 Square)", 640, 640)
)

val IMAGE_RESOLUTIONS: List<ResolutionOption> = listOf(
    ResolutionOption("i_768_1344", "768 × 1344 (9:16 手机全屏海报 · 推荐)", "768 × 1344 (9:16 Mobile Poster · Pick)", 768, 1344),
    ResolutionOption("i_1024_1024", "1024 × 1024 (1:1 超清方图)", "1024 × 1024 (1:1 Square HD)", 1024, 1024),
    ResolutionOption("i_1344_768", "1344 × 768 (16:9 电影横幅)", "1344 × 768 (16:9 Cinematic Widescreen)", 1344, 768),
    ResolutionOption("i_1536_1024", "1536 × 1024 (3:2 单反摄影)", "1536 × 1024 (3:2 DSLR Photo)", 1536, 1024)
)

data class PresetTemplate(
    val id: String,
    val category: ModelModality,
    val nameZh: String,
    val nameEn: String,
    val visualZh: String,
    val visualEn: String,
    val soundscape: String,
    val music: String
) {
    val name: String get() = L(nameZh, nameEn)
    val visual: String get() = L(visualZh, visualEn)
}

val PRESET_TEMPLATES: List<PresetTemplate> = listOf(
    PresetTemplate(
        id = "tpl_shinkai_train",
        category = ModelModality.VIDEO_AUDIO,
        nameZh = "新海诚风 · 盛夏樱花铁道口",
        nameEn = "Makoto Shinkai · Cherry Blossom Railway",
        visualZh = "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        visualEn = "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        soundscape = "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        music = "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings."
    ),
    PresetTemplate(
        id = "tpl_ufotable_duel",
        category = ModelModality.VIDEO_AUDIO,
        nameZh = "飞碟社风 · 苍蓝雷光剑术对决",
        nameEn = "Ufotable Sakuga · Azure Lightning Duel",
        visualZh = "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        visualEn = "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        soundscape = "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        music = "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs."
    ),
    PresetTemplate(
        id = "tpl_iceland_fpv",
        category = ModelModality.VIDEO_AUDIO,
        nameZh = "冰岛巨型瀑布 FPV 穿越机航拍",
        nameEn = "Iceland Waterfall FPV Drone Dive",
        visualZh = "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        visualEn = "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        soundscape = "Roaring waterfall rushing water, wind whooshing past the camera, distant seabirds calling.",
        music = "Epic cinematic acoustic guitar and soaring cello."
    ),
    PresetTemplate(
        id = "tpl_cyber_poster",
        category = ModelModality.IMAGE,
        nameZh = "东方赛博霓虹海报（含中文排版）",
        nameEn = "Cyberpunk Neon Bund Poster",
        visualZh = "超清商业电影海报摄影，雨夜霓虹闪烁的未来上海外滩街道，一位身着机能风银色夹克的少女回眸凝视镜头，景深虚化光斑，顶部醒目发光现代中文字体排版写着“未来纪元 2026”，极致光影细节，8K画质。",
        visualEn = "Ultra-HD commercial sci-fi movie poster photography, rain-slicked cyberpunk Shanghai Bund street glowing with neon reflections at night, a young woman in a futuristic silver techwear jacket looking back at the camera, shallow depth of field bokeh flares, bold glowing modern title typography at the top reading \"FUTURE ERA 2026\" (未来纪元 2026), intricate lighting details, 8K.",
        soundscape = "",
        music = ""
    ),
    PresetTemplate(
        id = "tpl_llm_storyboard",
        category = ModelModality.LLM_VLM,
        nameZh = "分镜构思",
        nameEn = "Storyboards",
        visualZh = "请帮我构思 3 组极具画面感的新海诚风格动画短片分镜，每组包含：1) 英文 Visual & Camera 镜头提示词；2) Soundscape 环境音描述；3) Background Music 配乐描述。",
        visualEn = "Please design 3 vivid Makoto Shinkai-style anime short film storyboard shots. For each shot, provide: 1) Visual & Camera prompt; 2) Soundscape & SFX prompt; 3) Background Music prompt.",
        soundscape = "",
        music = ""
    ),
    PresetTemplate(
        id = "tpl_vlm_reverse",
        category = ModelModality.LLM_VLM,
        nameZh = "看图反推",
        nameEn = "Vision Prompt",
        visualZh = "请仔细观察这张图片的主体、光影色调、镜头景别与艺术风格，先做简要中文分析，然后反推出一段高质量的英文生图/生视频 Prompt。",
        visualEn = "Carefully analyze the subject, lighting, color palette, camera framing, and artistic style of this image. First give a concise visual breakdown, then write a high-quality English image/video generation prompt.",
        soundscape = "",
        music = ""
    )
)

data class LoadedModelSummary(
    val id: String,
    val name: String,
    val mode: String? = null,
    val vramGb: Double? = null,
    val contextLength: Int? = null
) {
    val localizedName: String
        get() = OPEN_MODEL_CATALOG.firstOrNull { it.id == id }?.name ?: localizeServerText(name)
}

data class ComfyOutputMediaItem(
    val relPath: String,
    val filename: String,
    val mediaType: String? = null,
    val sizeMb: Double? = null
)

data class ComfyPresetPackItem(
    val id: String,
    val title: String,
    val badge: String,
    val scenario: String,
    val approxSizeGb: Double,
    val installedModels: Int,
    val totalModels: Int,
    val ready: Boolean
) {
    val localizedTitle: String
        get() {
            if (!LanguageManager.isEN) return title
            return when (id) {
                "minimax_h3_10eros_max" -> "10Eros-Max (Uncensored H3) · Multi-Ref + FLF2V"
                "minimax_h3_sulphur3" -> "Sulphur-3 (Uncensored H3) · Backbone + Corrector"
                "moss_image_edit" -> "MOSS-Image (Uncensored) · T2I + Instruction Edit"
                "wan22_remix_nsfw" -> "Wan2.2-Remix (Uncensored 14B) · I2V + FLF2V"
                "minimax_h3_drama_director" -> "MiniMax-H3 Drama Director · Multi-Shot + Storyboard"
                "qwen3_tts_voice_design" -> "Qwen3-TTS 1.7B · Natural Voice Design + Clone"
                "drama3_tts" -> "Fish Audio Drama 3 · 35B MoE Dramatic TTS"
                else -> localizeServerText(title)
            }
        }

    val localizedBadge: String
        get() {
            if (!LanguageManager.isEN) return badge
            return when (id) {
                "minimax_h3_10eros_max" -> "Uncensored · Multi-Ref 8-Step"
                "minimax_h3_sulphur3" -> "Uncensored · With Corrector"
                "moss_image_edit" -> "Uncensored · 3-Ref Edit"
                "wan22_remix_nsfw" -> "Uncensored · 8-Step Realism"
                "minimax_h3_drama_director" -> "Drama Timeline · Seamless AV"
                "qwen3_tts_voice_design" -> "Voice Design · Emotional Clone"
                "drama3_tts" -> "35B MoE · Stage Cues"
                else -> localizeServerText(badge)
            }
        }

    val localizedScenario: String
        get() {
            if (!LanguageManager.isEN) return scenario
            return when (id) {
                "minimax_h3_10eros_max" -> "TenStrip/10Eros-Max v1.0 + uncensored Heretic encoder for <Picture 1>/<Picture 2> multi-ref & FLF2V"
                "minimax_h3_sulphur3" -> "Full Sulphur-3 backbone + uncensored Heretic encoder + s3_corrector residual network"
                "moss_image_edit" -> "MOSS-Image SFT BF16 + Heretic decensored Qwen3.5-9B + Ostris Edit (armV5) + NSFW LoRA"
                "wan22_remix_nsfw" -> "FX-FeiHou/wan2.2-Remix NSFW I2V v3.0 (High+Low FP8) + NSFW-Wan-UMT5-XXL encoder in 8 steps"
                "minimax_h3_drama_director" -> "ComfyUI-MiniMaxH3-Director (NGH3Director + NGStoryboard) on uncensored 10Eros-Max 8-step"
                "qwen3_tts_voice_design" -> "Design new character voices from text (VoiceDesign 1.7B) or clone 3–15s speech with emotion cues (Base 1.7B)"
                "drama3_tts" -> "Zero-pip ComfyUI-Drama3 (2026-10-29 FP8) + Qwen3-ASR 1.7B, [sighs]/[whispers]/[laughs] stage cues & multi-speaker acting"
                else -> localizeServerText(scenario)
            }
        }
}

val DEFAULT_COMFY_PRESET_PACKS: List<ComfyPresetPackItem> = listOf(
    ComfyPresetPackItem("minimax_h3_10eros_max", "10Eros-Max (MiniMax-H3 无限制) · 多图参考 + 首尾帧 8步极速", "无内容限制 · 多图参考 8步", "基于 TenStrip/10Eros-Max v1.0 与无审核 Heretic 编码器，强化 <Picture 1>/<Picture 2> 多图参考", 39.1, 0, 4, false),
    ComfyPresetPackItem("minimax_h3_sulphur3", "Sulphur-3 (MiniMax-H3 无限制) · 满血底座 + H3 效果修正器", "无内容限制 · 含修正器", "内置 SulphurAI/Sulphur-3 满血主干 + 无审核 Heretic 编码器 + s3_corrector 效果修正器", 62.7, 0, 5, false),
    ComfyPresetPackItem("moss_image_edit", "MOSS-Image 无限制 · 文生图 & 指令修图 (Heretic + armV5 + NSFW LoRA)", "无限制 · 3图指令修图", "内置 MOSS-Image SFT BF16 + Heretic 无审核 Qwen3.5-9B + Ostris Edit (armV5) + NSFW LoRA", 19.4, 0, 5, false),
    ComfyPresetPackItem("wan22_remix_nsfw", "Wan2.2-Remix 无限制 · 14B NSFW 图生/首尾帧 (v3.0 8步)", "无内容限制 · 极致写实动态", "基于 FX-FeiHou/wan2.2-Remix NSFW I2V v3.0 + NSFW-Wan-UMT5-XXL，纯内置节点 8 步高写实生成", 33.13, 0, 4, false),
    ComfyPresetPackItem("minimax_h3_drama_director", "MiniMax-H3 短剧导演 · 角色声纹多镜头 + 分镜表成片 (10Eros-Max 8步)", "短剧时间线 · 音画无缝续拍", "集成 ComfyUI-MiniMaxH3-Director 并复用 10Eros-Max 8步无限制底座，支持角色图卡、声纹卡与长视频续拍", 39.13, 0, 5, false),
    ComfyPresetPackItem("qwen3_tts_voice_design", "Qwen3-TTS 1.7B · 自然语言声音设计 + 情感声纹克隆 (Apache-2.0)", "声音设计 · 情感配音", "支持零音频用中文/英文自然语言捏造角色声线 (VoiceDesign)，并支持 3~15s 声纹克隆与情绪指令 (Base)", 9.74, 0, 3, false),
    ComfyPresetPackItem("drama3_tts", "Fish Audio Drama 3 · 35B 戏剧级情感配音与 16 角色飙戏 (含 Qwen3-ASR)", "35B MoE · 舞台指令", "集成 PedroMarinhoDev/ComfyUI-Drama3（零 pip 依赖），支持 Qwen3-ASR 自动转录、[sighs]/[whispers]/[laughs] 舞台指示与多角色飙戏", 44.23, 0, 3, false)
)

data class LocalMediaItem(
    val file: File,
    val filename: String,
    val isImage: Boolean,
    val createdAtMillis: Long,
    val elapsedSeconds: Double?,
    val fileSizeMb: Double
)

data class HFTrendingModelItem(
    val repoId: String,
    val name: String,
    val pipelineTag: String,
    val modality: String,
    val isGguf: Boolean,
    val isGated: Boolean,
    val suggestedPattern: String,
    val likes: Int,
    val downloads: Int,
    val paramLabel: String = "",
    val estVramGb: Double = 0.0,
    val deployStatus: String = "unknown",
    val deployLabelZh: String = "显存待探测",
    val deployLabelEn: String = "VRAM TBD"
) {
    val localizedDeployLabel: String
        get() = stripEmoji(L(deployLabelZh.ifEmpty { "显存待探测" }, deployLabelEn.ifEmpty { "VRAM TBD" }))
}

val UNSUPPORTED_HF_PIPELINE_TAGS = setOf(
    "image-to-3d", "text-to-3d", "image-segmentation", "object-detection",
    "depth-estimation", "mask-generation", "keypoint-detection", "robotics",
    "reinforcement-learning", "tabular-classification", "tabular-regression",
    "time-series-forecasting", "audio-classification", "automatic-speech-recognition",
    "text-to-speech", "audio-to-audio", "feature-extraction", "sentence-similarity",
    "token-classification", "text-classification", "zero-shot-image-classification",
    "zero-shot-classification", "fill-mask", "question-answering",
    "table-question-answering", "video-classification"
)

val SUPPORTED_TRENDING_PIPELINE_TAGS = setOf(
    "text-generation", "image-text-to-text", "visual-question-answering",
    "any-to-any", "text-to-image", "image-to-image", "text-to-video", "image-to-video"
)

fun analyzeHFModelSizingJson(item: JSONObject, repoId: String, isGguf: Boolean): HFSizingResult {
    val ptag = item.optString("pipeline_tag", "").trim().lowercase(java.util.Locale.US)
    val st = item.optJSONObject("safetensors")
    val gg = item.optJSONObject("gguf")
    var totalParams = st?.optLong("total", 0L)?.takeIf { it > 0L }
        ?: gg?.optLong("total", 0L)?.takeIf { it > 0L }
        ?: 0L
    if (totalParams <= 0L) {
        val m = Regex("(?:^|[-_/])(\\d+(?:\\.\\d+)?)\\s*([bBmMtT])(?:[-_/\\b]|$)").find(repoId)
        if (m != null) {
            val v = m.groupValues[1].toDoubleOrNull() ?: 0.0
            when (m.groupValues[2].uppercase()) {
                "T" -> totalParams = (v * 1_000_000_000_000.0).toLong()
                "B" -> totalParams = (v * 1_000_000_000.0).toLong()
                "M" -> totalParams = (v * 1_000_000.0).toLong()
            }
        }
    }
    val paramLabel = when {
        totalParams >= 1_000_000_000_000L -> String.format(java.util.Locale.US, "%.1fT", totalParams / 1e12)
        totalParams >= 1_000_000_000L -> {
            val b = totalParams / 1e9
            if (b >= 20.0) String.format(java.util.Locale.US, "%.0fB", b)
            else String.format(java.util.Locale.US, "%.1fB", b).replace(".0B", "B")
        }
        totalParams >= 1_000_000L -> String.format(java.util.Locale.US, "%.0fM", totalParams / 1e6)
        else -> ""
    }
    val paramsB = if (totalParams > 0L) totalParams / 1e9 else 0.0
    val pmap = st?.optJSONObject("parameters")
    val estVramGb = when {
        isGguf -> if (paramsB > 0.0) kotlin.math.round((paramsB * 0.62 + 1.2) * 10.0) / 10.0 else 0.0
        pmap != null && pmap.length() > 0 -> {
            var bytesTotal = 0.0
            val keys = pmap.keys()
            while (keys.hasNext()) {
                val dt = keys.next()
                val cnt = pmap.optDouble(dt, 0.0)
                val dtUp = dt.uppercase()
                bytesTotal += when {
                    dtUp.contains("32") -> cnt * 4.0
                    dtUp.contains("16") -> cnt * 2.0
                    dtUp.contains("8") -> cnt * 1.0
                    dtUp.contains("4") -> cnt * 0.55
                    else -> cnt * 2.0
                }
            }
            kotlin.math.round(((bytesTotal / 1e9) * 1.12 + 1.2) * 10.0) / 10.0
        }
        paramsB > 0.0 -> kotlin.math.round((paramsB * 2.1 + 1.2) * 10.0) / 10.0
        else -> 0.0
    }
    val vInt = kotlin.math.max(1, kotlin.math.round(estVramGb).toInt())
    return when {
        !isGguf && ptag in UNSUPPORTED_HF_PIPELINE_TAGS -> HFSizingResult(paramLabel, estVramGb, "too_large", "非内置管线 ($ptag · 需专用SDK)", "Custom Pipeline ($ptag)")
        estVramGb <= 0.0 -> HFSizingResult(paramLabel, 0.0, "unknown", "显存待探测", "VRAM TBD")
        estVramGb <= 14.2 -> HFSizingResult(paramLabel, estVramGb, "great", "适合 · T4(15G)可跑 (~${vInt}G)", "Fit: T4 15G (~${vInt}G)")
        estVramGb <= 22.5 -> HFSizingResult(paramLabel, estVramGb, "good", "适合 · 需L4(24G) (~${vInt}G)", "Fit: L4 24G (~${vInt}G)")
        estVramGb <= 39.0 -> HFSizingResult(paramLabel, estVramGb, "a100", "可部署 · 需A100(40G) (~${vInt}G)", "Fit: A100 40G (~${vInt}G)")
        estVramGb <= 78.0 -> HFSizingResult(paramLabel, estVramGb, "heavy", "需A100(80G) (~${vInt}G)", "Needs A100 80G (~${vInt}G)")
        isGguf && paramsB <= 240.0 -> HFSizingResult(paramLabel, estVramGb, "heavy", "需A100(80G)+CPU卸载 (~${vInt}G)", "A100 80G + Offload (~${vInt}G)")
        else -> HFSizingResult(paramLabel, estVramGb, "too_large", "超单卡上限 (~${vInt}G · 建议搜GGUF)", "Too Large (~${vInt}G · Try GGUF)")
    }
}

data class HFSizingResult(
    val paramLabel: String,
    val estVramGb: Double,
    val deployStatus: String,
    val deployZh: String,
    val deployEn: String
)

data class HFModelWeightFileItem(
    val filename: String,
    val sizeGb: Double,
    val fitStatus: String,
    val fitZh: String,
    val fitEn: String
) {
    val localizedFit: String get() = stripEmoji(L(fitZh, fitEn))
}

data class HFModelDetailInfo(
    val repoId: String,
    val name: String,
    val pipelineTag: String,
    val modality: String,
    val isGguf: Boolean,
    val isGated: Boolean,
    val license: String,
    val baseModel: String,
    val architecture: String,
    val contextLength: Int,
    val likes: Int,
    val downloads: Int,
    val usedStorageGb: Double,
    val paramLabel: String,
    val deployLabelZh: String,
    val deployLabelEn: String,
    val weightFiles: List<HFModelWeightFileItem>,
    val readmeExcerpt: String
) {
    val localizedDeployLabel: String get() = stripEmoji(L(deployLabelZh, deployLabelEn))
}

data class ChatTurnItem(
    val id: String = UUID.randomUUID().toString(),
    val role: String,
    val content: String,
    val thinking: String = "",
    val hasImage: Boolean = false
)

fun JSONObject.optDoubleOrNull(key: String): Double? =
    if (has(key) && !isNull(key)) optDouble(key).takeIf { !it.isNaN() } else null

fun JSONObject.optIntOrNull(key: String): Int? =
    if (has(key) && !isNull(key)) optInt(key) else null

fun JSONObject.optBooleanOrNull(key: String): Boolean? =
    if (has(key) && !isNull(key)) optBoolean(key) else null

fun JSONObject.optStringOrNull(key: String): String? =
    if (has(key) && !isNull(key)) optString(key) else null

fun JSONArray.toStringList(): List<String> = (0 until length()).mapNotNull { optString(it, null) }
