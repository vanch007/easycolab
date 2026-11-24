import SwiftUI
import AppKit

// MARK: - 中间工作区：外部 CLI / Skill / 云端 ComfyUI 运行任务与提示词实时预览卡片

struct ExternalTaskPreviewCard: View {
    @ObservedObject var vm: StudioViewModel
    let task: ExternalTaskSnapshot
    let palette: StudioPalette
    @ObservedObject private var i18n = LanguageManager.shared

    private var statusColor: Color {
        switch task.status {
        case "completed": return .blue
        case "failed": return .red
        default: return .green
        }
    }

    private var statusBadgeText: String {
        if task.source == "comfyui-remote" {
            return L("云端 ComfyUI 队列任务", "Cloud ComfyUI Active Task")
        }
        switch task.status {
        case "completed": return L("CLI / 技能任务已完成", "Skill / CLI Completed")
        case "failed": return L("CLI / 技能任务失败", "Skill / CLI Failed")
        default: return L("CLI / 技能正在运行", "Skill / CLI Running")
        }
    }

    private var resolvedFolderPath: String? {
        if let p = task.project_dir, !p.isEmpty, FileManager.default.fileExists(atPath: p) {
            return p
        }
        if let o = task.output_dir, !o.isEmpty, FileManager.default.fileExists(atPath: o) {
            return o
        }
        return nil
    }

    private var firstExistingOutputURL: URL? {
        task.resolvedOutputURLs.first
    }

    private var needsModelPageJump: Bool {
        guard let mId = task.model_id?.trimmingCharacters(in: .whitespacesAndNewlines), !mId.isEmpty else {
            return false
        }
        return vm.selectedModelId != mId
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // 1. 顶部栏：运行状态 + 项目名称 + 模型与模式 + 自动跟随 + 实时百分比 + 折叠/隐藏按钮
            HStack(spacing: 8) {
                StatusDot(color: statusColor)

                Text(statusBadgeText)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(statusColor)

                if let proj = task.project_name, !proj.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "folder.fill")
                            .font(.system(size: 10))
                        Text(proj)
                            .lineLimit(1)
                    }
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Color.primary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                    .help(task.project_dir ?? task.output_dir ?? proj)
                }

                if let modelTitle = task.model_name, !modelTitle.isEmpty {
                    let modeSuffix = (task.task_mode ?? "").isEmpty ? "" : " · \(task.task_mode!)"
                    Text("\(modelTitle)\(modeSuffix)")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2.5)
                        .background(Capsule().fill(Color.primary.opacity(0.06)))
                }

                Spacer(minLength: 6)

                Button {
                    vm.autoFollowExternalTasks.toggle()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: vm.autoFollowExternalTasks ? "arrow.triangle.2.circlepath.circle.fill" : "pause.circle")
                            .font(.system(size: 10.5))
                        Text(L("自动跟随", "Auto-Follow"))
                            .font(.system(size: 10.5, weight: .medium))
                    }
                    .foregroundStyle(vm.autoFollowExternalTasks ? Color.accentColor : Color.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 2.5)
                    .background(Capsule().fill(vm.autoFollowExternalTasks ? Color.accentColor.opacity(0.14) : Color.primary.opacity(0.06)))
                }
                .buttonStyle(.plain)
                .help(L("开启后，收到 CLI / 技能或云端 ComfyUI 任务时自动跳转中间工作区到对应模型功能页并同步参数", "Automatically switch the center workspace to the active CLI/cloud task's model page and sync parameters"))

                if task.status == "running", let prog = task.progress {
                    Text("\(Int(round(max(0.0, min(1.0, prog)) * 100)))%")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.accentColor))
                }

                Button {
                    withAnimation(.easeInOut(duration: 0.16)) {
                        vm.externalTaskCardExpanded.toggle()
                    }
                } label: {
                    Image(systemName: vm.externalTaskCardExpanded ? "chevron.up.circle.fill" : "chevron.down.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help(vm.externalTaskCardExpanded ? L("收起任务详情", "Collapse task details") : L("展开任务详情", "Expand task details"))

                Button {
                    vm.externalTaskCardDismissedId = task.task_id
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help(L("暂时隐藏此任务预览卡片", "Hide external task preview"))
            }

            if task.status == "running", let prog = task.progress {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: max(0.0, min(1.0, prog)))
                        .progressViewStyle(.linear)
                        .controlSize(.small)
                    if vm.externalTaskCardExpanded, let lbl = task.progress_label, !lbl.isEmpty {
                        Text(lbl)
                            .font(.system(size: 10.5, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
            } else if vm.externalTaskCardExpanded, task.status == "failed", let err = task.error, !err.isEmpty {
                Text(err)
                    .font(.system(size: 11))
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }

            if vm.externalTaskCardExpanded {
                // 2. 提示词与音景实时预览
                if let prompt = task.prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    VStack(alignment: .leading, spacing: 5) {
                        HStack(spacing: 6) {
                            Text(L("当前任务提示词", "Active Task Prompt"))
                                .font(.system(size: 10.5, weight: .semibold))
                                .foregroundStyle(.secondary)
                            if let params = task.params_summary, !params.isEmpty {
                                Text("· \(params)")
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                        }

                        ScrollView {
                            Text(prompt)
                                .font(.system(size: 12))
                                .foregroundStyle(.primary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        }
                        .frame(maxHeight: 92)
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 7).fill(Color.primary.opacity(0.04)))

                        if let snd = task.soundscape, !snd.isEmpty {
                            Text(L("环境音效/对白：\(snd)", "Soundscape: \(snd)"))
                                .font(.system(size: 10.5))
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                                .textSelection(.enabled)
                        }
                        if let mus = task.music, !mus.isEmpty {
                            Text(L("背景音乐：\(mus)", "Music: \(mus)"))
                                .font(.system(size: 10.5))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .textSelection(.enabled)
                        }
                    }
                }

                // 3. 输入素材标签 + 快捷操作按钮
                HStack(spacing: 8) {
                    if let inputs = task.input_files, !inputs.isEmpty {
                        HStack(spacing: 4) {
                            Image(systemName: "paperclip")
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                            Text(inputs.prefix(4).joined(separator: ", ") + (inputs.count > 4 ? " +\(inputs.count - 4)" : ""))
                                .font(.system(size: 10.5, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }

                    Spacer(minLength: 4)

                    if needsModelPageJump {
                        Button {
                            vm.syncWorkspaceToExternalTask(task, force: true)
                        } label: {
                            Label(L("跳转到任务页", "Jump to Task Page"), systemImage: "arrow.right.circle.fill")
                                .font(.system(size: 10.5, weight: .semibold))
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    } else {
                        Button {
                            vm.syncWorkspaceToExternalTask(task, force: true)
                        } label: {
                            Label(L("同步参数到编辑区", "Sync Params to Editor"), systemImage: "arrow.down.doc")
                                .font(.system(size: 10.5, weight: .medium))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    if let folderPath = resolvedFolderPath {
                        Button {
                            NSWorkspace.shared.open(URL(fileURLWithPath: folderPath))
                        } label: {
                            Label(L("打开项目目录", "Open Project"), systemImage: "folder")
                                .font(.system(size: 10.5, weight: .medium))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    if let outURL = firstExistingOutputURL {
                        Button {
                            vm.refreshLocalMedia()
                            vm.selectMedia(outURL)
                        } label: {
                            Label(L("预览成片", "Preview Output"), systemImage: "play.rectangle.fill")
                                .font(.system(size: 10.5, weight: .medium))
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                }
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(palette.sidebarCardFill)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(statusColor.opacity(0.32), lineWidth: 1.2)
                )
        )
    }
}
