import SwiftUI
import UIKit
import Foundation

// MARK: - Bilingual Localization Manager (Default: Simplified Chinese "zh")

enum AppLanguage: String, CaseIterable {
    case zh = "zh"
    case en = "en"
}

extension Notification.Name {
    static let easyColabLanguageDidChange = Notification.Name("EasyColabIOS.LanguageDidChange")
}

final class LanguageManager: ObservableObject {
    static let shared = LanguageManager()

    @Published var lang: AppLanguage {
        didSet {
            UserDefaults.standard.set(lang.rawValue, forKey: "appLanguage")
            if oldValue != lang {
                NotificationCenter.default.post(name: .easyColabLanguageDidChange, object: nil)
            }
        }
    }

    var isEN: Bool { lang == .en }

    private init() {
        if UserDefaults.standard.string(forKey: "appLanguage") == "en" {
            self.lang = .en
        } else {
            self.lang = .zh
        }
    }

    func toggle() {
        lang = (lang == .zh) ? .en : .zh
    }
}

@inline(__always)
func L(_ zh: String, _ en: String) -> String {
    LanguageManager.shared.lang == .en ? en : zh
}

// MARK: - Global iOS Keyboard Dismissal Helper, Accessory Bar & Tap-Outside Installer

@MainActor
func dismissIOSKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
}

@MainActor
final class IOSKeyboardObserver: ObservableObject {
    static let shared = IOSKeyboardObserver()

    @Published private(set) var isVisible: Bool = false

    private init() {
        let nc = NotificationCenter.default
        nc.addObserver(forName: UIResponder.keyboardWillShowNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.isVisible = true }
        }
        nc.addObserver(forName: UIResponder.keyboardWillHideNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.isVisible = false }
        }
        nc.addObserver(forName: UITextField.textDidBeginEditingNotification, object: nil, queue: .main) { [weak self] note in
            guard let tf = note.object as? UITextField else { return }
            Task { @MainActor in self?.ensureAccessoryToolbar(on: tf) }
        }
        nc.addObserver(forName: UITextView.textDidBeginEditingNotification, object: nil, queue: .main) { [weak self] note in
            guard let tv = note.object as? UITextView, tv.isEditable else { return }
            Task { @MainActor in self?.ensureAccessoryToolbar(on: tv) }
        }
    }

    private func makeAccessoryToolbar(tag: Int) -> UIToolbar {
        let bar = UIToolbar(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: 42))
        bar.tag = tag
        bar.sizeToFit()

        let flex = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)

        var btnConfig = UIButton.Configuration.gray()
        btnConfig.cornerStyle = .capsule
        btnConfig.baseForegroundColor = .label
        btnConfig.image = UIImage(
            systemName: "keyboard.chevron.compact.down",
            withConfiguration: UIImage.SymbolConfiguration(pointSize: 12, weight: .bold)
        )
        btnConfig.imagePadding = 5
        btnConfig.contentInsets = NSDirectionalEdgeInsets(top: 5, leading: 12, bottom: 5, trailing: 12)
        var attrTitle = AttributedString(L("收起键盘", "Done"))
        attrTitle.font = .system(size: 13, weight: .bold)
        btnConfig.attributedTitle = attrTitle

        let btn = UIButton(configuration: btnConfig, primaryAction: UIAction { _ in
            Task { @MainActor in dismissIOSKeyboard() }
        })
        let doneItem = UIBarButtonItem(customView: btn)
        bar.items = [flex, doneItem]
        return bar
    }

    private func ensureAccessoryToolbar(on textField: UITextField) {
        let langTag = LanguageManager.shared.isEN ? 202 : 101
        if textField.inputAccessoryView?.tag == langTag { return }
        textField.inputAccessoryView = makeAccessoryToolbar(tag: langTag)
        textField.reloadInputViews()
    }

    private func ensureAccessoryToolbar(on textView: UITextView) {
        let langTag = LanguageManager.shared.isEN ? 202 : 101
        if textView.inputAccessoryView?.tag == langTag { return }
        textView.inputAccessoryView = makeAccessoryToolbar(tag: langTag)
        textView.reloadInputViews()
    }
}

struct IOSKeyboardDismissTapInstaller: UIViewRepresentable {
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> UIView {
        let view = InstallerHostView()
        view.backgroundColor = .clear
        view.isUserInteractionEnabled = false
        view.onMoveToWindowCallback = { [weak coordinator = context.coordinator] window in
            coordinator?.installIfNeeded(on: window)
        }
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        if let window = uiView.window {
            context.coordinator.installIfNeeded(on: window)
        }
    }

    final class InstallerHostView: UIView {
        var onMoveToWindowCallback: ((UIWindow?) -> Void)?
        override func didMoveToWindow() {
            super.didMoveToWindow()
            onMoveToWindowCallback?(window)
        }
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        private static let gestureName = "EasyColabIOSKeyboardDismissTap"

        func installIfNeeded(on window: UIWindow?) {
            guard let window else { return }
            if window.gestureRecognizers?.contains(where: { $0.name == Self.gestureName }) == true {
                return
            }
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleBackgroundTap(_:)))
            tap.name = Self.gestureName
            tap.cancelsTouchesInView = false
            tap.delaysTouchesBegan = false
            tap.delaysTouchesEnded = false
            tap.delegate = self
            window.addGestureRecognizer(tap)
        }

        @objc private func handleBackgroundTap(_ recognizer: UITapGestureRecognizer) {
            guard recognizer.state == .ended else { return }
            Task { @MainActor in
                dismissIOSKeyboard()
            }
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
            guard IOSKeyboardObserver.shared.isVisible else { return false }
            var current: UIView? = touch.view
            while let view = current {
                if view is UITextField || view is UITextView || view is UIControl {
                    return false
                }
                let typeName = String(describing: type(of: view))
                if typeName.contains("TextField")
                    || typeName.contains("TextView")
                    || typeName.contains("TextEditor")
                    || typeName.contains("Keyboard")
                    || typeName.contains("ContextMenu")
                    || typeName.contains("Button") {
                    return false
                }
                current = view.superview
            }
            return true
        }

        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
        ) -> Bool {
            true
        }
    }
}

func localizeModelFileDesc(_ raw: String) -> String {
    guard LanguageManager.shared.isEN, !raw.isEmpty else { return raw }
    var s = raw
    s = s.replacingOccurrences(of: " [RAM内存盘]", with: " [RAM Disk]")
    s = s.replacingOccurrences(of: "DiT 音视频主干 BF16", with: "DiT Audio-Video Backbone BF16")
    s = s.replacingOccurrences(of: "多模态编码器 BF16", with: "Multimodal Encoder BF16")
    s = s.replacingOccurrences(of: "3D 视频 VAE BF16", with: "3D Video VAE BF16")
    s = s.replacingOccurrences(of: "32kHz 立体声 VAE BF16", with: "32kHz Stereo Audio VAE BF16")
    s = s.replacingOccurrences(of: "权重分片", with: "Weight Shard")
    s = s.replacingOccurrences(of: "Unsloth UD-Q4_K_M 主模型", with: "Unsloth UD-Q4_K_M Main Model")
    s = s.replacingOccurrences(of: "多模态视觉投影器", with: "Multimodal Vision Projector")
    s = s.replacingOccurrences(of: "Unsloth MTP 多词元投机解码头", with: "Unsloth MTP Speculative Decoding Head")
    s = s.replacingOccurrences(of: "编码器", with: "Encoder")
    s = s.replacingOccurrences(of: "图像 VAE", with: "Image VAE")
    return s
}

func localizeServerText(_ text: String) -> String {
    guard !text.isEmpty else { return text }
    let adaptedZh = text
        .replacingOccurrences(of: "Mac 客户端", with: "App 客户端")
        .replacingOccurrences(of: "左侧", with: "算力页/上方")
    guard LanguageManager.shared.isEN else { return adaptedZh }

    let exactMap: [String: String] = [
        "未加载": "Not Loaded",
        "未连接 Colab 服务端": "Colab Server Disconnected",
        "尚未启动云端 ComfyUI 工作台": "Cloud ComfyUI Studio Not Started",
        "Google Drive 自动软链持久化": "Google Drive Auto-Symlink Persistence",
        "Google Drive 实时软链挂载": "Google Drive Live Symlink Mount",
        "Google Drive 云盘自动持久化": "Google Drive Auto-Persistence",
        "实例本地高速存储": "Instance Local NVMe Storage",
        "已彻底释放全部云端实例，计费速率归零。": "All cloud Colab instances terminated. Billing rate is now 0 CU/h.",
        "已手动停止当前推理任务": "Inference cancelled by user",
        "正在上传最新多模型开源服务端代码 (app.py)...": "Uploading latest multi-model server engine (app.py)...",
        "正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...": "Configuring Google Drive persistence, multi-model engine & public tunnel (~40–90s)...",
        "正在克隆 DiffSynth-Studio...": "Cloning DiffSynth-Studio..."
    ]
    if let exact = exactMap[text] { return exact }

    var s = localizeModelFileDesc(text)
    let replacements: [(String, String)] = [
        ("已无缝接管在线 ComfyUI 画布", "Connected to active ComfyUI canvas"),
        ("ComfyUI 进程已退出", "ComfyUI process exited"),
        ("，可点击重新启动", "; tap to restart"),
        ("正在 8 线程极速下载 ", "8-thread fast downloading "),
        ("正在 8 线程并发下载 ", "8-thread downloading "),
        ("正在下载 ", "Downloading "),
        (" 已就绪 (", " ready ("),
        (")，跳过下载", "), skipping"),
        ("正在评估显存余量并装载 ", "Checking VRAM & loading "),
        (" 至 GPU...", " into GPU..."),
        (" 正在编码多模态上下文并启动推理...", " encoding multimodal context & starting inference..."),
        (" 推理完成！耗时 ", " inference complete! Elapsed "),
        (" 正在使用多模态编码器解析提示词", " encoding prompts with multimodal encoder"),
        ("DiT 采样已完成，正在执行", "DiT sampling finished, running "),
        ("3D 视频 VAE 解码", "3D Video VAE decoding"),
        ("视频帧解码完成，正在执行 32kHz 立体声音轨解码", "Video frames decoded, decoding 32kHz stereo audio"),
        (" 正在执行去噪采样第 ", " denoising step "),
        (" 步 (", " ("),
        ("VAE 解码完成，正在合成 32kHz 立体声音轨并封装 MP4", "VAE decoded, muxing 32kHz stereo audio into MP4"),
        ("正在写入无损 PNG 图像文件", "Writing lossless PNG image"),
        ("生成完成！耗时 ", "Generation complete! Elapsed "),
        ("生成出错: ", "Generation error: "),
        ("尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。", "is not loaded in Colab GPU VRAM. Tap 'Load to VRAM' first."),
        ("尚未加载到显存", "is not loaded in VRAM"),
        ("当前已常驻：", "currently resident: "),
        ("已通过 Google OAuth 自动挂载 Google Drive", "Mounted Google Drive automatically via Google OAuth")
    ]
    for (zhSub, enSub) in replacements where s.contains(zhSub) {
        s = s.replacingOccurrences(of: zhSub, with: enSub)
    }
    return s
}

func stripEmoji(_ text: String) -> String {
    var scalars = String.UnicodeScalarView()
    for s in text.unicodeScalars {
        let p = s.properties
        let v = s.value
        let isEmoji = p.isEmojiPresentation
            || (p.isEmoji && v > 0x238C)
            || v == 0xFE0F || v == 0x200D || v == 0x20E3
            || (0x1F1E6...0x1F1FF).contains(v)
        if !isEmoji { scalars.append(s) }
    }
    var result = String(scalars)
    while result.contains("  ") { result = result.replacingOccurrences(of: "  ", with: " ") }
    return result.trimmingCharacters(in: .whitespaces)
}

func formatDuration(_ sec: Double, compact: Bool = false) -> String {
    let s = max(0.0, sec)
    if s < 60.0 {
        if compact { return String(format: "%.1fs", s) }
        return LanguageManager.shared.isEN
            ? String(format: "%.1f s", s)
            : String(format: "%.1f 秒", s)
    }
    let mins = Int(s) / 60
    let rem = s - Double(mins * 60)
    if compact { return String(format: "%dm%.0fs", mins, rem) }
    return LanguageManager.shared.isEN
        ? String(format: "%dm %.1fs", mins, rem)
        : String(format: "%d分%.1f秒", mins, rem)
}

extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}

// MARK: - Multimodal Models & Presets Catalog

enum ModelModality: String, Codable {
    case videoAudio = "video_audio"
    case video = "video"
    case image = "image"
    case audio = "audio"
    case llmVLM = "llm_vlm"
    case comfyUI = "comfyui"
}

struct TaskModeItem: Identifiable, Hashable {
    let id: String
    let titleZh: String
    let titleEn: String
    let subtitleZh: String
    let subtitleEn: String
    let icon: String

    var title: String { L(titleZh, titleEn) }
    var subtitle: String { L(subtitleZh, subtitleEn) }
}

struct ModelFileItem: Codable, Identifiable, Hashable {
    var id: String { name }
    var name: String
    var desc: String?
    var exists: Bool
    var size_gb: Double

    var localizedDesc: String? {
        guard let raw = desc, !raw.isEmpty else { return nil }
        return localizeModelFileDesc(raw)
    }
}

struct CustomHFModelDTO: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var repo_id: String
    var modality: String
    var category: String?
    var approx_size_gb: Double?
    var include_pattern: String?
    var pipeline_arch: String?
    var files_count: Int?
    var is_custom: Bool?
}

struct OpenModelSpec: Identifiable, Hashable {
    let id: String
    let nameZh: String
    let nameEn: String
    let shortTitleZh: String
    let shortTitleEn: String
    let repoId: String
    let badgeZh: String
    let badgeEn: String
    let summaryZh: String
    let summaryEn: String
    let approxSizeGB: Double
    let icon: String
    let modality: ModelModality
    let defaultSteps: Double
    let modes: [TaskModeItem]
    let defaultFiles: [ModelFileItem]
    var includePattern: String = ""
    var isCustom: Bool = false

    var name: String { L(nameZh, nameEn) }
    var shortTitle: String { L(shortTitleZh, shortTitleEn) }
    var badge: String { L(badgeZh, badgeEn) }
    var summary: String { L(summaryZh, summaryEn) }

    static func == (lhs: OpenModelSpec, rhs: OpenModelSpec) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    static func fromCustomDTO(_ dto: CustomHFModelDTO) -> OpenModelSpec {
        let mod: ModelModality
        switch dto.modality {
        case "video_audio": mod = .videoAudio
        case "image": mod = .image
        case "audio": mod = .audio
        default: mod = .llmVLM
        }
        let sizeGB = dto.approx_size_gb ?? 0.0
        let arch = (dto.pipeline_arch ?? "gguf").uppercased()
        let badgeZh = sizeGB > 0 ? String(format: "自定义 %@ · %.1f GB", arch, sizeGB) : "自定义 HF · \(arch)"
        let badgeEn = sizeGB > 0 ? String(format: "Custom %@ · %.1f GB", arch, sizeGB) : "Custom HF · \(arch)"
        let patDesc = (dto.include_pattern?.isEmpty == false) ? " (\(dto.include_pattern!))" : ""
        let summaryZh = dto.category ?? "自定义 HuggingFace 仓库 \(dto.repo_id)\(patDesc)"
        let summaryEn = "Custom HuggingFace repo \(dto.repo_id)\(patDesc)"
        let modes: [TaskModeItem]
        let iconName: String
        let defSteps: Double
        switch mod {
        case .videoAudio, .video:
            iconName = "film.stack"
            defSteps = 35
            modes = [
                TaskModeItem(id: "t2va", titleZh: "文生音视频", titleEn: "Text to Video+Audio", subtitleZh: "输入镜头与声音描述生成视频", subtitleEn: "Generate video & audio from prompts", icon: "film"),
                TaskModeItem(id: "fl2va", titleZh: "首尾帧生视频", titleEn: "Keyframes to Video", subtitleZh: "从相册选首帧/尾帧参考图", subtitleEn: "Pick start/end keyframes from Photos", icon: "photo.on.rectangle.angled")
            ]
        case .image:
            iconName = "sparkles.rectangle.stack"
            defSteps = 20
            modes = [
                TaskModeItem(id: "t2i", titleZh: "高清文生图", titleEn: "Text-to-Image", subtitleZh: "自定义 HuggingFace 图像生成管线", subtitleEn: "Custom HuggingFace image generation", icon: "photo.artframe"),
                TaskModeItem(id: "i2i_edit", titleZh: "图生图重绘", titleEn: "Image Restyle", subtitleZh: "从相册选参考图重绘", subtitleEn: "Pick reference photo to restyle", icon: "wand.and.rays")
            ]
        case .audio:
            iconName = "waveform.badge.mic"
            defSteps = 1
            modes = [
                TaskModeItem(id: "voice_design", titleZh: "自然语言声音设计", titleEn: "Voice Design", subtitleZh: "无需参考音频，用自然语言描述捏造声线", subtitleEn: "Create a character voice from natural language", icon: "waveform"),
                TaskModeItem(id: "voice_clone", titleZh: "声纹克隆与情感配音", titleEn: "Voice Clone + Emotion", subtitleZh: "上传参考人声并附加情绪指令合成", subtitleEn: "Clone a voice clip with emotion directives", icon: "mic")
            ]
        case .llmVLM, .comfyUI:
            iconName = "shippingbox.fill"
            defSteps = 20
            modes = [
                TaskModeItem(id: "chat", titleZh: "深度推理对话", titleEn: "Reasoning Chat", subtitleZh: "自定义 HuggingFace 模型对话与推理", subtitleEn: "Custom HuggingFace model chat & reasoning", icon: "bubble.left.and.bubble.right"),
                TaskModeItem(id: "vlm", titleZh: "看图反推/视觉", titleEn: "Vision & Prompt", subtitleZh: "多模态看图分析与提示词生成", subtitleEn: "Multimodal vision & prompt analysis", icon: "eye"),
                TaskModeItem(id: "api", titleZh: "API 协议网关", titleEn: "API Gateway", subtitleZh: "OpenAI (/v1) 与 Anthropic (/v1/messages) 接口", subtitleEn: "OpenAI (/v1) & Anthropic (/v1/messages) endpoints", icon: "network")
            ]
        }
        return OpenModelSpec(
            id: dto.id,
            nameZh: dto.name,
            nameEn: dto.name,
            shortTitleZh: dto.name,
            shortTitleEn: dto.name,
            repoId: dto.repo_id,
            badgeZh: badgeZh,
            badgeEn: badgeEn,
            summaryZh: summaryZh,
            summaryEn: summaryEn,
            approxSizeGB: sizeGB,
            icon: iconName,
            modality: mod,
            defaultSteps: defSteps,
            modes: modes,
            defaultFiles: [],
            includePattern: dto.include_pattern ?? "",
            isCustom: true
        )
    }
}

let OPEN_MODEL_CATALOG: [OpenModelSpec] = [
    OpenModelSpec(
        id: "minimax_h3_drama_director",
        nameZh: "MiniMax-H3 短剧导演 · 多镜头连拍",
        nameEn: "MiniMax-H3 Drama Director · Multi-Shot",
        shortTitleZh: "H3 短剧导演",
        shortTitleEn: "H3 Drama",
        repoId: "seesee75-commits/ComfyUI-MiniMaxH3-Director",
        badgeZh: "短剧时间线 · 复用 10Eros-Max",
        badgeEn: "Drama Timeline · 10Eros-Max",
        summaryZh: "集成 ComfyUI-MiniMaxH3-Director + 10Eros-Max 8步底座，支持角色定妆卡、声纹锁定与音画无缝长视频续拍",
        summaryEn: "ComfyUI-MiniMaxH3-Director on uncensored 10Eros-Max 8-step with character/voice cards & seamless AV continuity",
        approxSizeGB: 39.13,
        icon: "film.stack",
        modality: .video,
        defaultSteps: 8,
        modes: [
            TaskModeItem(id: "drama_multishot", titleZh: "多镜头短剧连拍", titleEn: "Multi-Shot Drama", subtitleZh: "按时间线渲染多镜头短剧，自动音画无缝续拍", subtitleEn: "Render multi-shot short dramas with seamless AV chunk chaining", icon: "film"),
            TaskModeItem(id: "drama_storyboard", titleZh: "角色声纹分镜表", titleEn: "Storyboard + Voice", subtitleZh: "搭配角色图卡、声纹参考与分镜画板成片", subtitleEn: "Pair character refs, voice card & storyboard panels into a cohesive short", icon: "rectangle.grid.2x2")
        ],
        defaultFiles: [
            ModelFileItem(name: "diffusion_models/10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors", desc: "10Eros-Max 8步主干 (21.32 GB · 共享)", exists: false, size_gb: 0),
            ModelFileItem(name: "vae_approx/taeh3.safetensors", desc: "MiniMax-H3 预览 TAE (0.02 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "qwen3_tts_voice_design",
        nameZh: "Qwen3-TTS 1.7B · 声音设计",
        nameEn: "Qwen3-TTS 1.7B · Voice Design",
        shortTitleZh: "Qwen3-TTS 音色",
        shortTitleEn: "Qwen3-TTS",
        repoId: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        badgeZh: "文字捏声线 + 情感克隆 · 9.7 GB",
        badgeEn: "Voice Design + Clone · 9.7 GB",
        summaryZh: "Apache-2.0 开源自然语言声音设计 (VoiceDesign 1.7B) + 零样本情感克隆 (Base 1.7B)",
        summaryEn: "Apache-2.0 natural Voice Design (1.7B) + zero-shot emotional cloning (Base 1.7B)",
        approxSizeGB: 9.74,
        icon: "waveform.badge.mic",
        modality: .audio,
        defaultSteps: 1,
        modes: [
            TaskModeItem(id: "voice_design", titleZh: "自然语言声音设计", titleEn: "Voice Design", subtitleZh: "零音频描述音色与情绪捏造角色声纹", subtitleEn: "Describe voice & emotion from text without ref audio", icon: "waveform"),
            TaskModeItem(id: "voice_clone", titleZh: "声纹克隆情感配音", titleEn: "Voice Clone", subtitleZh: "上传参考音频并附加情感指令合成", subtitleEn: "Clone speech with emotion instructions", icon: "mic")
        ],
        defaultFiles: [
            ModelFileItem(name: "qwen-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign/model.safetensors", desc: "Qwen3-TTS 1.7B 声音设计 (4.52 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "qwen-tts/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors", desc: "Qwen3-TTS 1.7B 声纹克隆 (4.54 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "qwen-tts/Qwen3-TTS-Tokenizer-12Hz/model.safetensors", desc: "Qwen3-TTS 12Hz Tokenizer (0.68 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "drama3_tts",
        nameZh: "Drama 3 · 35B 戏剧情感配音",
        nameEn: "Fish Audio Drama 3 · 35B TTS",
        shortTitleZh: "Drama 3 配音",
        shortTitleEn: "Drama 3 TTS",
        repoId: "PedroMarinhoDev/Drama-3-ComfyUI",
        badgeZh: "35B MoE · 舞台指令 + 多角色",
        badgeEn: "35B MoE · Stage Cues + Dialogue",
        summaryZh: "Fish Audio Drama 3 35B MoE (FP8) + Codec + Qwen3-ASR 1.7B，支持 [sighs]/[whispers]/[laughs] 舞台指示与多角色飙戏",
        summaryEn: "Fish Audio Drama 3 35B MoE (FP8) + Codec + Qwen3-ASR 1.7B with [sighs]/[whispers]/[laughs] cues & dialogue",
        approxSizeGB: 44.23,
        icon: "theatermasks.fill",
        modality: .audio,
        defaultSteps: 1,
        modes: [
            TaskModeItem(id: "drama_tts_clone", titleZh: "声纹戏剧配音 (ASR)", titleEn: "Acting Clone (ASR)", subtitleZh: "上传参考声纹，ASR 自动转录并支持舞台指示", subtitleEn: "Upload voice clip with auto Qwen3-ASR & stage cues", icon: "theatermasks"),
            TaskModeItem(id: "drama_tts_free", titleZh: "无参考多角色配音", titleEn: "Multi-Speaker Script", subtitleZh: "使用 <|speaker:0|>/<|speaker:1|> 生成戏剧对白", subtitleEn: "Synthesize multi-speaker dialogue with inline cues", icon: "waveform")
        ],
        defaultFiles: [
            ModelFileItem(name: "diffusion_models/drama3_fp8experts_bf16.safetensors", desc: "Drama 3 35B FP8 权重 (38.58 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "vae/drama3_codec.safetensors", desc: "Drama 3 Codec (1.57 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "asr/qwen3_asr_1.7b_bf16.safetensors", desc: "Qwen3-ASR 1.7B (4.08 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "qwen38_27b_uncensored_gguf",
        nameZh: "Qwen3.8-27B 无限制 · GGUF",
        nameEn: "Qwen3.8-27B Uncensored · GGUF",
        shortTitleZh: "Qwen3.8 无限制",
        shortTitleEn: "Qwen3.8 Uncensored",
        repoId: "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF",
        badgeZh: "Heretic 去审查 + MTP · 17.8 GB",
        badgeEn: "Heretic Uncensored + MTP · 17.8 GB",
        summaryZh: "Heretic 去审查 Qwen3.8-27B Q4_K_M + F16 视觉投影器 + MTP 投机解码，支持 256K 上下文、看图反推与 API 网关",
        summaryEn: "Heretic abliterated Qwen3.8-27B Q4_K_M + F16 vision mmproj + MTP draft head for uncensored chat, vision & API gateway",
        approxSizeGB: 17.83,
        icon: "cpu.fill",
        modality: .llmVLM,
        defaultSteps: 20,
        modes: [
            TaskModeItem(id: "chat", titleZh: "无限制推理对话", titleEn: "Uncensored Chat", subtitleZh: "Heretic 去审查 + MTP 投机解码，支持长思维链与短剧剧本创作", subtitleEn: "Heretic uncensored + MTP speculative decoding for scripts & reasoning", icon: "bubble.left.and.bubble.right"),
            TaskModeItem(id: "vlm", titleZh: "无限制看图反推", titleEn: "Uncensored Vision", subtitleZh: "传照片无拒答解析构图、姿态与反推生图/生视频 Prompt", subtitleEn: "Analyze photos & reverse-engineer prompts without refusals", icon: "eye"),
            TaskModeItem(id: "api", titleZh: "API 协议网关", titleEn: "API Gateway", subtitleZh: "OpenAI (/v1) 与 Anthropic (/v1/messages) 接口", subtitleEn: "OpenAI (/v1) & Anthropic (/v1/messages) endpoints", icon: "network")
        ],
        defaultFiles: [
            ModelFileItem(name: "Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf", desc: "Heretic 去审查 Q4_K_M 主模型 (15.41 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", desc: "F16 视觉投影器 mmproj (0.86 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "Qwen3.8-27B-Uncensored-draft-Q4_0.gguf", desc: "MTP 投机解码草稿头 Q4_0 (1.56 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "comfyui_studio",
        nameZh: "ComfyUI 节点工作台",
        nameEn: "ComfyUI Node Studio",
        shortTitleZh: "ComfyUI 工作台",
        shortTitleEn: "ComfyUI Studio",
        repoId: "comfyanonymous/ComfyUI",
        badgeZh: "4套预置包 · 云盘持久化",
        badgeEn: "4 Preset Packs · Drive Sync",
        summaryZh: "一键拉起云端 ComfyUI + 4 套开源视频工作流包，支持内置画布预览与成片一键存手机",
        summaryEn: "1-click cloud ComfyUI + 4 open video workflow packs, in-app canvas & 1-tap output pull to iPhone",
        approxSizeGB: 0.0,
        icon: "point.3.connected.trianglepath.dotted",
        modality: .comfyUI,
        defaultSteps: 25,
        modes: [
            TaskModeItem(id: "comfy_canvas", titleZh: "云端工作流与画布", titleEn: "Workflows & Canvas", subtitleZh: "一键配齐 Wan2.2 / MiniMax-H3 / LTX-2.3 / 混元预置包并打开画布", subtitleEn: "Install Wan2.2 / MiniMax-H3 / LTX-2.3 / Hunyuan packs & open canvas", icon: "rectangle.connected.to.line.below"),
            TaskModeItem(id: "comfy_assets", titleZh: "下权重与拉取成片", titleEn: "Weights & Outputs", subtitleZh: "8 线程极速下模型至云盘，并将 ComfyUI 成片一键拉回手机", subtitleEn: "8-thread cloud weight download & 1-tap pull outputs to phone", icon: "arrow.down.doc")
        ],
        defaultFiles: []
    )
]

struct ResolutionOption: Identifiable, Hashable {
    let id: String
    let labelZh: String
    let labelEn: String
    let width: Int
    let height: Int

    var label: String { L(labelZh, labelEn) }
}

let VIDEO_RESOLUTIONS: [ResolutionOption] = [
    ResolutionOption(id: "v_480_832", labelZh: "480 × 832 (9:16 竖屏短视频 · 手机推荐)", labelEn: "480 × 832 (9:16 Vertical Short · Mobile Pick)", width: 480, height: 832),
    ResolutionOption(id: "v_832_480", labelZh: "832 × 480 (16:9 横屏标准)", labelEn: "832 × 480 (16:9 Landscape Standard)", width: 832, height: 480),
    ResolutionOption(id: "v_720_1280", labelZh: "720 × 1280 (9:16 720P 高清竖屏)", labelEn: "720 × 1280 (9:16 720P HD Vertical)", width: 720, height: 1280),
    ResolutionOption(id: "v_1280_720", labelZh: "1280 × 720 (16:9 720P 高清横屏)", labelEn: "1280 × 720 (16:9 720P HD Landscape)", width: 1280, height: 720),
    ResolutionOption(id: "v_640_640", labelZh: "640 × 640 (1:1 方形)", labelEn: "640 × 640 (1:1 Square)", width: 640, height: 640)
]

let IMAGE_RESOLUTIONS: [ResolutionOption] = [
    ResolutionOption(id: "i_768_1344", labelZh: "768 × 1344 (9:16 手机全屏海报 · 推荐)", labelEn: "768 × 1344 (9:16 Mobile Poster · Recommended)", width: 768, height: 1344),
    ResolutionOption(id: "i_1024_1024", labelZh: "1024 × 1024 (1:1 超清方图)", labelEn: "1024 × 1024 (1:1 Square HD)", width: 1024, height: 1024),
    ResolutionOption(id: "i_1344_768", labelZh: "1344 × 768 (16:9 电影横幅)", labelEn: "1344 × 768 (16:9 Cinematic Widescreen)", width: 1344, height: 768),
    ResolutionOption(id: "i_1536_1024", labelZh: "1536 × 1024 (3:2 单反摄影)", labelEn: "1536 × 1024 (3:2 DSLR Photo)", width: 1536, height: 1024)
]

struct PresetTemplate: Identifiable {
    let id: String
    let category: ModelModality
    let nameZh: String
    let nameEn: String
    let visualZh: String
    let visualEn: String
    let soundscape: String
    let music: String

    var name: String { L(nameZh, nameEn) }
    var visual: String { L(visualZh, visualEn) }
}

let PRESET_TEMPLATES: [PresetTemplate] = [
    PresetTemplate(
        id: "tpl_shinkai_train",
        category: .videoAudio,
        nameZh: "新海诚风 · 盛夏樱花铁道口",
        nameEn: "Makoto Shinkai · Cherry Blossom Railway",
        visualZh: "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        visualEn: "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        soundscape: "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        music: "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings."
    ),
    PresetTemplate(
        id: "tpl_ufotable_duel",
        category: .videoAudio,
        nameZh: "飞碟社风 · 苍蓝雷光剑术对决",
        nameEn: "Ufotable Sakuga · Azure Lightning Duel",
        visualZh: "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        visualEn: "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        soundscape: "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        music: "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs."
    ),
    PresetTemplate(
        id: "tpl_iceland_fpv",
        category: .videoAudio,
        nameZh: "冰岛巨型瀑布 FPV 穿越机航拍",
        nameEn: "Iceland Waterfall FPV Drone Dive",
        visualZh: "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        visualEn: "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        soundscape: "Roaring waterfall rushing water, wind whooshing past the camera, distant seabirds calling.",
        music: "Epic cinematic acoustic guitar and soaring cello."
    ),
    PresetTemplate(
        id: "tpl_cyber_poster",
        category: .image,
        nameZh: "东方赛博霓虹海报（含中文排版）",
        nameEn: "Cyberpunk Neon Bund Poster",
        visualZh: "超清商业电影海报摄影，雨夜霓虹闪烁的未来上海外滩街道，一位身着机能风银色夹克的少女回眸凝视镜头，景深虚化光斑，顶部醒目发光现代中文字体排版写着“未来纪元 2026”，极致光影细节，8K画质。",
        visualEn: "Ultra-HD commercial sci-fi movie poster photography, rain-slicked cyberpunk Shanghai Bund street glowing with neon reflections at night, a young woman in a futuristic silver techwear jacket looking back at the camera, shallow depth of field bokeh flares, bold glowing modern title typography at the top reading \"FUTURE ERA 2026\" (未来纪元 2026), intricate lighting details, 8K.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_film_cafe",
        category: .image,
        nameZh: "哈苏胶片感 · 午后东京咖啡馆",
        nameEn: "Hasselblad Film · Afternoon Cafe",
        visualZh: "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam beside an open sketchbook, natural film grain, ultra-detailed textures.",
        visualEn: "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam beside an open sketchbook, natural film grain, ultra-detailed textures.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_llm_storyboard",
        category: .llmVLM,
        nameZh: "生成 3 组新海诚风 AI 视频分镜",
        nameEn: "Design 3 Shinkai AI Video Storyboards",
        visualZh: "请帮我构思 3 组极具画面感的新海诚风格动画短片分镜，每组包含：1) 英文 Visual & Camera 镜头提示词；2) Soundscape 环境音描述；3) Background Music 配乐描述。",
        visualEn: "Please design 3 vivid Makoto Shinkai-style anime short film storyboard shots. For each shot, provide: 1) Visual & Camera prompt; 2) Soundscape & SFX prompt; 3) Background Music prompt.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_vlm_reverse",
        category: .llmVLM,
        nameZh: "看图深度解析并反推生图/生视频 Prompt",
        nameEn: "Analyze Photo & Reverse-Engineer Prompt",
        visualZh: "请仔细观察这张图片的主体、光影色调、镜头景别与艺术风格，先做简要中文分析，然后反推出一段高质量的英文生图/生视频 Prompt。",
        visualEn: "Carefully analyze the subject, lighting, color palette, camera framing, and artistic style of this image. First give a concise visual breakdown, then write a high-quality English image/video generation prompt.",
        soundscape: "",
        music: ""
    )
]

// MARK: - Server API Wire DTOs

struct LoadedModelSummary: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var mode: String?
    var vram_gb: Double?
    var context_length: Int?

    var localizedName: String {
        if let spec = OPEN_MODEL_CATALOG.first(where: { $0.id == id }) {
            return spec.name
        }
        return localizeServerText(name)
    }
}

struct ComfyOutputMediaItem: Codable, Identifiable, Hashable {
    var id: String { rel_path }
    var rel_path: String
    var filename: String
    var media_type: String?
    var size_mb: Double?
    var mtime: Int?
}

struct ComfyPresetPackItem: Codable, Identifiable, Hashable {
    var id: String
    var title: String
    var badge: String
    var scenario: String
    var approx_size_gb: Double
    var workflows: [String]
    var installed_models: Int
    var total_models: Int
    var workflows_installed: Bool
    var ready: Bool

    var localizedTitle: String {
        guard LanguageManager.shared.isEN else { return title }
        switch id {
        case "minimax_h3_10eros_max": return "10Eros-Max (Uncensored H3) · Multi-Ref + FLF2V"
        case "minimax_h3_sulphur3": return "Sulphur-3 (Uncensored H3) · Backbone + Corrector"
        case "moss_image_edit": return "MOSS-Image (Uncensored) · T2I + Instruction Edit"
        case "wan22_remix_nsfw": return "Wan2.2-Remix (Uncensored 14B) · I2V + FLF2V"
        case "minimax_h3_drama_director": return "MiniMax-H3 Drama Director · Multi-Shot + Storyboard"
        case "qwen3_tts_voice_design": return "Qwen3-TTS 1.7B · Natural Voice Design + Clone"
        case "drama3_tts": return "Fish Audio Drama 3 · 35B MoE Dramatic TTS"
        default: return localizeServerText(title)
        }
    }

    var localizedBadge: String {
        guard LanguageManager.shared.isEN else { return badge }
        switch id {
        case "minimax_h3_10eros_max": return "Uncensored · Multi-Ref 8-Step"
        case "minimax_h3_sulphur3": return "Uncensored · With Corrector"
        case "moss_image_edit": return "Uncensored · 3-Ref Edit"
        case "wan22_remix_nsfw": return "Uncensored · 8-Step Realism"
        case "minimax_h3_drama_director": return "Drama Timeline · Seamless AV"
        case "qwen3_tts_voice_design": return "Voice Design · Emotional Clone"
        case "drama3_tts": return "35B MoE · Stage Cues"
        default: return localizeServerText(badge)
        }
    }

    var localizedScenario: String {
        guard LanguageManager.shared.isEN else { return scenario }
        switch id {
        case "minimax_h3_10eros_max": return "TenStrip/10Eros-Max v1.0 + uncensored Heretic encoder for <Picture 1>/<Picture 2> multi-ref & FLF2V in 8 steps"
        case "minimax_h3_sulphur3": return "Full Sulphur-3 backbone + uncensored Heretic encoder + s3_corrector residual network for higher fidelity"
        case "moss_image_edit": return "MOSS-Image SFT BF16 + Heretic decensored Qwen3.5-9B + Ostris Edit (armV5) + NSFW LoRA for unrestricted T2I & editing"
        case "wan22_remix_nsfw": return "FX-FeiHou/wan2.2-Remix NSFW I2V v3.0 (High+Low FP8) + NSFW-Wan-UMT5-XXL encoder in 8 steps"
        case "minimax_h3_drama_director": return "ComfyUI-MiniMaxH3-Director (NGH3Director + NGStoryboard) on uncensored 10Eros-Max 8-step with character/voice cards"
        case "qwen3_tts_voice_design": return "Design new character voices from text (VoiceDesign 1.7B) or clone 3–15s speech with emotion cues (Base 1.7B)"
        case "drama3_tts": return "Zero-pip ComfyUI-Drama3 (2026-10-29 FP8) + Qwen3-ASR 1.7B, [sighs]/[whispers]/[laughs] stage cues & multi-speaker acting"
        default: return localizeServerText(scenario)
        }
    }
}

let DEFAULT_COMFY_PRESET_PACKS: [ComfyPresetPackItem] = [
    ComfyPresetPackItem(
        id: "minimax_h3_10eros_max",
        title: "10Eros-Max (MiniMax-H3 无限制) · 多图参考 + 首尾帧 8步极速",
        badge: "无内容限制 · 多图参考 8步",
        scenario: "基于 TenStrip/10Eros-Max v1.0 与无审核 Heretic 编码器，强化 <Picture 1>/<Picture 2> 多图参考与首尾帧一致性",
        approx_size_gb: 39.1,
        workflows: ["07_10Eros-Max_无限制多图参考有声视频(8步极速).json", "08_10Eros-Max_无限制首尾帧图生有声视频(8步极速).json"],
        installed_models: 0, total_models: 4, workflows_installed: false, ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_sulphur3",
        title: "Sulphur-3 (MiniMax-H3 无限制) · 满血底座 + H3 效果修正器",
        badge: "无内容限制 · 含修正器",
        scenario: "内置 SulphurAI/Sulphur-3 满血主干 + 无审核 Heretic 编码器 + s3_corrector 速度残差修正器",
        approx_size_gb: 62.7,
        workflows: ["09_Sulphur-3_无限制文生有声视频(含H3修正器).json", "10_Sulphur-3_无限制图生首尾帧有声视频(含H3修正器).json"],
        installed_models: 0, total_models: 5, workflows_installed: false, ready: false
    ),
    ComfyPresetPackItem(
        id: "moss_image_edit",
        title: "MOSS-Image 无限制 · 文生图 & 指令修图 (Heretic + armV5 + NSFW LoRA)",
        badge: "无限制 · 3图指令修图",
        scenario: "内置 MOSS-Image SFT BF16 + Heretic 无审核 Qwen3.5-9B + Ostris Edit (armV5) + NSFW LoRA",
        approx_size_gb: 19.4,
        workflows: ["14_MOSS-Image_无限制文生图(Heretic+NSFW).json", "15_MOSS-Image_无限制指令修图(OstrisEdit+armV5).json"],
        installed_models: 0, total_models: 5, workflows_installed: false, ready: false
    ),
    ComfyPresetPackItem(
        id: "wan22_remix_nsfw",
        title: "Wan2.2-Remix 无限制 · 14B NSFW 图生/首尾帧 (v3.0 8步)",
        badge: "无内容限制 · 极致写实动态",
        scenario: "基于 FX-FeiHou/wan2.2-Remix NSFW I2V v3.0 (High+Low FP8) + NSFW-Wan-UMT5-XXL，纯内置节点 8 步高写实生成",
        approx_size_gb: 33.13,
        workflows: ["16_Wan2.2-Remix_无限制单图生视频(v3.0_8步极速).json", "17_Wan2.2-Remix_无限制首尾帧过渡(v3.0_8步极速).json"],
        installed_models: 0, total_models: 4, workflows_installed: false, ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_drama_director",
        title: "MiniMax-H3 短剧导演 · 角色声纹多镜头 + 分镜表成片 (10Eros-Max 8步)",
        badge: "短剧时间线 · 音画无缝续拍",
        scenario: "集成 ComfyUI-MiniMaxH3-Director 并复用 10Eros-Max 8步无限制底座，支持角色图卡、声纹卡与长视频续拍",
        approx_size_gb: 39.13,
        workflows: ["18_MiniMax-H3_短剧导演_角色声纹多镜头(10Eros-Max_8步).json", "19_MiniMax-H3_短剧导演_双角色分镜表成片(10Eros-Max_8步).json"],
        installed_models: 0, total_models: 5, workflows_installed: false, ready: false
    ),
    ComfyPresetPackItem(
        id: "qwen3_tts_voice_design",
        title: "Qwen3-TTS 1.7B · 自然语言声音设计 + 情感声纹克隆 (Apache-2.0)",
        badge: "声音设计 · 情感配音",
        scenario: "支持零音频用中文/英文自然语言捏造角色声线 (VoiceDesign)，并支持 3~15s 声纹克隆与情绪指令 (Base)",
        approx_size_gb: 9.74,
        workflows: ["20_Qwen3-TTS_自然语言声音设计(VoiceDesign_1.7B).json", "21_Qwen3-TTS_声纹克隆与情感配音(Base_1.7B).json"],
        installed_models: 0, total_models: 3, workflows_installed: false, ready: false
    ),
    ComfyPresetPackItem(
        id: "drama3_tts",
        title: "Fish Audio Drama 3 · 35B 戏剧级情感配音与 16 角色飙戏 (含 Qwen3-ASR)",
        badge: "35B MoE · 舞台指令",
        scenario: "集成 PedroMarinhoDev/ComfyUI-Drama3（零 pip 依赖），支持 Qwen3-ASR 自动转录、[sighs]/[whispers]/[laughs] 舞台指示与多角色飙戏",
        approx_size_gb: 44.23,
        workflows: ["22_Drama3_35B情感配音与音色克隆(自动ASR转录).json", "23_Drama3_35B多角色情感飙戏与舞台指令.json"],
        installed_models: 0, total_models: 3, workflows_installed: false, ready: false
    )
]

struct ComfyRemoteStatus: Codable {
    var is_running: Bool?
    var is_starting: Bool?
    var is_syncing_drive: Bool?
    var is_downloading_weight: Bool?
    var installing_pack_id: String?
    var progress: Double?
    var status_message: String?
    var public_url: String?
    var local_port: Int?
    var queue_running: Int?
    var queue_pending: Int?
    var persistence_mode: String?
    var persistence_label: String?
    var drive_persisted: Bool?
    var persist_root: String?
    var custom_nodes_count: Int?
    var models_count: Int?
    var workflows_count: Int?
    var last_sync_ts: Int?
    var recent_outputs: [ComfyOutputMediaItem]?
    var preset_packs: [ComfyPresetPackItem]?
}

struct ColabStatusResponse: Codable {
    var ok: Bool?
    var cuda_available: Bool?
    var gpu_name: String?
    var vram_free_gb: Double?
    var vram_total_gb: Double?
    var vram_used_gb: Double?
    var model_loaded: Bool?
    var any_model_loaded: Bool?
    var selected_model_id: String?
    var loaded_model_id: String?
    var loaded_model_ids: [String]?
    var loaded_models_detail: [LoadedModelSummary]?
    var current_model: String?
    var message: String?
    var context_length: Int?
    var loaded_context_length: Int?
    var files: [ModelFileItem]?
    var downloaded_count: Int?
    var total_count: Int?
    var all_downloaded: Bool?
    var total_size_gb: Double?
    var drive_persisted: Bool?
    var bf16_shards_count: Int?
    var is_downloading: Bool?
    var is_loading_vram: Bool?
    var task_progress: Double?
    var task_message: String?
    var uptime_seconds: Int?
    var cu_burn_rate_per_hour: Double?
    var session_cu_used: Double?
    var idle_timeout_minutes: Int?
    var idle_remaining_seconds: Int?
    var api_key_enabled: Bool?
    var api_request_count: Int?
    var console_lines: [String]?
    var ram_total_gb: Double?
    var ram_used_gb: Double?
    var ram_free_gb: Double?
    var disk_total_gb: Double?
    var disk_used_gb: Double?
    var disk_free_gb: Double?
    var shm_total_gb: Double?
    var shm_used_gb: Double?
    var shm_free_gb: Double?
    var drive_total_gb: Double?
    var drive_used_gb: Double?
    var drive_free_gb: Double?
    var comfyui: ComfyRemoteStatus?
    var custom_models: [CustomHFModelDTO]?
    var registered_model: CustomHFModelDTO?
    var action_ok: Bool?
    var action_message: String?
}

struct CompletedMediaItem: Codable, Hashable {
    var index: Int?
    var seed: Int?
    var filename: String
    var video_url: String
    var media_type: String?
    var elapsed: Double?
}

struct LiveTelemetrySnapshot: Codable {
    var cuda_available: Bool?
    var gpu_name: String?
    var vram_free_gb: Double?
    var vram_total_gb: Double?
    var vram_used_gb: Double?
    var cu_burn_rate_per_hour: Double?
    var ram_total_gb: Double?
    var ram_used_gb: Double?
    var ram_free_gb: Double?
    var disk_total_gb: Double?
    var disk_used_gb: Double?
    var disk_free_gb: Double?
    var shm_total_gb: Double?
    var shm_used_gb: Double?
    var shm_free_gb: Double?
    var drive_total_gb: Double?
    var drive_used_gb: Double?
    var drive_free_gb: Double?
}

struct ColabJobResponse: Codable {
    var job_id: String
    var model_id: String?
    var status: String
    var progress: Double
    var message: String
    var seed: Int?
    var elapsed: Double?
    var filename: String?
    var video_url: String?
    var media_type: String?
    var text_output: String?
    var thinking_output: String?
    var batch_count: Int?
    var completed_count: Int?
    var completed_videos: [CompletedMediaItem]?
    var console_lines: [String]?
    var telemetry: LiveTelemetrySnapshot?
}

struct LocalMediaItem: Identifiable, Hashable {
    var id: String { filename }
    let url: URL
    let filename: String
    let isImage: Bool
    let createdAt: Date
    let elapsedSeconds: Double?
    let fileSizeMB: Double
}

struct HFTrendingModelItem: Identifiable, Codable, Hashable {
    var id: String { repo_id }
    let repo_id: String
    let name: String
    let pipeline_tag: String
    let modality: String
    let is_gguf: Bool
    let is_gated: Bool
    let suggested_pattern: String
    let likes: Int
    let downloads: Int
    var param_count: Int? = nil
    var param_label: String? = nil
    var est_vram_gb: Double? = nil
    var deploy_status: String? = nil
    var deploy_label_zh: String? = nil
    var deploy_label_en: String? = nil

    var localizedDeployLabel: String {
        let zh = (deploy_label_zh?.isEmpty == false) ? deploy_label_zh! : "显存待探测"
        let en = (deploy_label_en?.isEmpty == false) ? deploy_label_en! : "VRAM TBD"
        return stripEmojiCharacters(L(zh, en))
    }

    var deployBadgeColor: Color {
        switch deploy_status ?? "unknown" {
        case "great", "good":
            return .green
        case "a100":
            return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy":
            return .orange
        case "too_large":
            return .red
        default:
            return .secondary
        }
    }
}

func stripEmojiCharacters(_ text: String) -> String {
    let emojis = ["🟢", "🟡", "🟠", "🔴", "⚪", "📦", "📄", "🔥", "🔒", "🔑", "🌐", "♥", "⬇", "ⓘ"]
    var result = text
    for e in emojis {
        result = result.replacingOccurrences(of: e, with: "")
    }
    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}

func cleanHFReadmeExcerpt(_ rawMd: String) -> String {
    var body = rawMd
    if body.hasPrefix("---"),
       let rEnd = body.range(of: "\n---", range: body.index(body.startIndex, offsetBy: 3)..<body.endIndex) {
        body = String(body[rEnd.upperBound...])
    }
    body = body.replacingOccurrences(of: "<!--[\\s\\S]*?-->", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "<a[^>]*>\\s*<img[^>]*>\\s*</a>", with: "", options: [.regularExpression, .caseInsensitive])
    body = body.replacingOccurrences(of: "<img[^>]*>", with: "", options: [.regularExpression, .caseInsensitive])
    body = body.replacingOccurrences(of: "<h[1-4][^>]*>(.*?)</h[1-4]>", with: "\n### $1\n", options: [.regularExpression, .caseInsensitive])
    body = body.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "\\[!\\[[^\\]]*\\]\\([^\\)]*\\)\\]\\([^\\)]*\\)", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "!\\[[^\\]]*\\]\\([^\\)]*\\)", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "&nbsp;", with: " ")
        .replacingOccurrences(of: "&amp;", with: "&")
        .replacingOccurrences(of: "&lt;", with: "<")
        .replacingOccurrences(of: "&gt;", with: ">")
        .replacingOccurrences(of: "&quot;", with: "\"")

    var cleanLines: [String] = []
    var totalChars = 0
    for ln in body.components(separatedBy: .newlines) {
        let s = ln.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.isEmpty || s.hasPrefix("```") || s.hasPrefix("|---") || s.hasPrefix("| ---") { continue }
        cleanLines.append(s)
        totalChars += s.count + 1
        if totalChars >= 1400 { break }
    }
    var joined = cleanLines.joined(separator: "\n")
    joined = joined.replacingOccurrences(of: "\\[([^\\]]+)\\]\\([^\\)\\n]*$", with: "$1", options: .regularExpression)
    return joined
}

func renderHFInlineMarkdown(_ raw: String) -> AttributedString {
    var cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    cleaned = cleaned.replacingOccurrences(of: "\\[([^\\]]+)\\]\\([^\\)\\n]*$", with: "$1", options: .regularExpression)
    let oddBold = cleaned.components(separatedBy: "**").count - 1
    if oddBold % 2 != 0, let lastRange = cleaned.range(of: "**", options: .backwards) {
        cleaned.removeSubrange(lastRange)
    }
    if let attr = try? AttributedString(
        markdown: cleaned,
        options: AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: true,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
    ) {
        return attr
    }
    var plain = cleaned
    plain = plain.replacingOccurrences(of: "\\[([^\\]]+)\\]\\(([^\\)]+)\\)", with: "$1", options: .regularExpression)
    plain = plain.replacingOccurrences(of: "**", with: "")
    plain = plain.replacingOccurrences(of: "__", with: "")
    plain = plain.replacingOccurrences(of: "`", with: "")
    return AttributedString(plain)
}

let UNSUPPORTED_HF_PIPELINE_TAGS: Set<String> = [
    "image-to-3d", "text-to-3d", "image-segmentation", "object-detection",
    "depth-estimation", "mask-generation", "keypoint-detection", "robotics",
    "reinforcement-learning", "tabular-classification", "tabular-regression",
    "time-series-forecasting", "audio-classification", "automatic-speech-recognition",
    "text-to-speech", "audio-to-audio", "feature-extraction", "sentence-similarity",
    "token-classification", "text-classification", "zero-shot-image-classification",
    "zero-shot-classification", "fill-mask", "question-answering",
    "table-question-answering", "video-classification"
]

let SUPPORTED_TRENDING_PIPELINE_TAGS: Set<String> = [
    "text-generation", "image-text-to-text", "visual-question-answering",
    "any-to-any", "text-to-image", "image-to-image", "text-to-video", "image-to-video"
]

func analyzeHFModelSizing(_ obj: [String: Any], repoId: String, isGGUF: Bool, fallbackStorageGb: Double = 0.0) -> (
    paramCount: Int,
    paramLabel: String,
    estVramGb: Double,
    deployStatus: String,
    deployZh: String,
    deployEn: String
) {
    let ptag = ((obj["pipeline_tag"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    let st = obj["safetensors"] as? [String: Any]
    let gg = obj["gguf"] as? [String: Any]
    var totalParams: Int64 = 0
    if let n = st?["total"] as? NSNumber {
        totalParams = n.int64Value
    } else if let n = gg?["total"] as? NSNumber {
        totalParams = n.int64Value
    }
    if totalParams <= 0 {
        if let regex = try? NSRegularExpression(pattern: "(?:^|[-_/])(\\d+(?:\\.\\d+)?)\\s*([bBmMtT])(?:[-_/\\b]|$)"),
           let match = regex.firstMatch(in: repoId, range: NSRange(repoId.startIndex..., in: repoId)),
           let r1 = Range(match.range(at: 1), in: repoId),
           let r2 = Range(match.range(at: 2), in: repoId),
           let val = Double(repoId[r1]) {
            let unit = repoId[r2].uppercased()
            if unit == "T" { totalParams = Int64(val * 1_000_000_000_000) }
            else if unit == "B" { totalParams = Int64(val * 1_000_000_000) }
            else if unit == "M" { totalParams = Int64(val * 1_000_000) }
        }
    }

    let paramLabel: String
    if totalParams >= 1_000_000_000_000 {
        paramLabel = String(format: "%.1fT", Double(totalParams) / 1_000_000_000_000.0)
    } else if totalParams >= 1_000_000_000 {
        let b = Double(totalParams) / 1_000_000_000.0
        if b >= 20 {
            paramLabel = String(format: "%.0fB", b)
        } else {
            let s = String(format: "%.1fB", b)
            paramLabel = s.replacingOccurrences(of: ".0B", with: "B")
        }
    } else if totalParams >= 1_000_000 {
        paramLabel = String(format: "%.0fM", Double(totalParams) / 1_000_000.0)
    } else {
        paramLabel = ""
    }

    let paramsB = totalParams > 0 ? (Double(totalParams) / 1_000_000_000.0) : 0.0
    var estVramGb: Double = 0.0
    if isGGUF {
        estVramGb = paramsB > 0 ? ((paramsB * 0.62 + 1.2) * 10).rounded() / 10 : 0.0
    } else if let pmap = st?["parameters"] as? [String: Any], !pmap.isEmpty {
        var bytesTotal: Double = 0.0
        for (dtype, rawCnt) in pmap {
            let cnt = (rawCnt as? NSNumber)?.doubleValue ?? 0.0
            let dt = dtype.uppercased()
            if dt.contains("32") { bytesTotal += cnt * 4.0 }
            else if dt.contains("16") { bytesTotal += cnt * 2.0 }
            else if dt.contains("8") { bytesTotal += cnt * 1.0 }
            else if dt.contains("4") { bytesTotal += cnt * 0.55 }
            else { bytesTotal += cnt * 2.0 }
        }
        estVramGb = (((bytesTotal / 1_000_000_000.0) * 1.12 + 1.2) * 10).rounded() / 10
    } else if paramsB > 0 {
        estVramGb = ((paramsB * 2.1 + 1.2) * 10).rounded() / 10
    } else if fallbackStorageGb > 0 {
        estVramGb = ((fallbackStorageGb * 1.12 + 1.0) * 10).rounded() / 10
    }

    let status: String
    let zh: String
    let en: String
    let vramInt = max(1, Int(estVramGb.rounded()))
    if !isGGUF && UNSUPPORTED_HF_PIPELINE_TAGS.contains(ptag) {
        status = "too_large"
        zh = "非内置管线 (\(ptag) · 需专用SDK)"
        en = "Custom Pipeline (\(ptag))"
    } else if estVramGb <= 0 {
        status = "unknown"
        zh = "显存待探测"
        en = "VRAM TBD"
    } else if estVramGb <= 14.2 {
        status = "great"
        zh = "适合 · T4(15G)可跑 (~\(vramInt)G)"
        en = "Fit: T4 15G (~\(vramInt)G)"
    } else if estVramGb <= 22.5 {
        status = "good"
        zh = "适合 · 需L4(24G) (~\(vramInt)G)"
        en = "Fit: L4 24G (~\(vramInt)G)"
    } else if estVramGb <= 39.0 {
        status = "a100"
        zh = "可部署 · 需A100(40G) (~\(vramInt)G)"
        en = "Fit: A100 40G (~\(vramInt)G)"
    } else if estVramGb <= 78.0 {
        status = "heavy"
        zh = "需A100(80G) (~\(vramInt)G)"
        en = "Needs A100 80G (~\(vramInt)G)"
    } else if isGGUF && paramsB <= 240.0 {
        status = "heavy"
        zh = "需A100(80G)+CPU卸载 (~\(vramInt)G)"
        en = "A100 80G + Offload (~\(vramInt)G)"
    } else {
        status = "too_large"
        zh = "超单卡上限 (~\(vramInt)G · 建议搜GGUF)"
        en = "Too Large (~\(vramInt)G · Try GGUF)"
    }
    return (Int(clamping: totalParams), paramLabel, estVramGb, status, zh, en)
}

struct HFTrendingAPIResponse: Codable {
    var ok: Bool?
    var category: String?
    var models: [HFTrendingModelItem]?
}

struct HFModelWeightFileItem: Identifiable, Codable, Hashable {
    var id: String { filename }
    let filename: String
    let size_gb: Double
    let fit_status: String
    let fit_zh: String
    let fit_en: String

    var localizedFit: String { stripEmojiCharacters(L(fit_zh, fit_en)) }

    var fitColor: Color {
        switch fit_status {
        case "great", "good": return .green
        case "a100": return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy": return .orange
        case "too_large": return .red
        default: return .secondary
        }
    }
}

struct HFModelDetailInfo: Identifiable, Codable, Hashable {
    var id: String { repo_id }
    var ok: Bool?
    var repo_id: String
    var name: String
    var author: String
    var pipeline_tag: String
    var modality: String
    var is_gguf: Bool
    var is_gated: Bool
    var license: String
    var base_model: String
    var architecture: String
    var context_length: Int
    var languages: [String]
    var likes: Int
    var downloads: Int
    var last_modified: String
    var used_storage_gb: Double
    var param_count: Int
    var param_label: String
    var est_vram_gb: Double
    var deploy_status: String
    var deploy_label_zh: String
    var deploy_label_en: String
    var weight_files: [HFModelWeightFileItem]
    var readme_excerpt: String

    var localizedDeployLabel: String {
        stripEmojiCharacters(L(deploy_label_zh.isEmpty ? "显存待探测" : deploy_label_zh, deploy_label_en.isEmpty ? "VRAM TBD" : deploy_label_en))
    }

    var deployBadgeColor: Color {
        switch deploy_status {
        case "great", "good": return .green
        case "a100": return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy": return .orange
        case "too_large": return .red
        default: return .secondary
        }
    }
}

struct ChatTurnItem: Identifiable, Hashable {
    let id = UUID()
    var role: String
    var content: String
    var thinking: String = ""
    var hasImage: Bool = false
}
