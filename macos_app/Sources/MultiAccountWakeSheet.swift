import SwiftUI
import AppKit

// MARK: - 多账号同时申请 GPU 调度器与配置弹窗

enum MultiAccountWakeMode: String, CaseIterable, Identifiable {
    case firstWins = "first"
    case allParallel = "all"

    var id: String { rawValue }
}

struct PendingAutoWakeRequest {
    let email: String
    let gpu: String
    let persistent: Bool
    let createdAt: Date
}

@MainActor
final class MultiAccountWakeCoordinator {
    static let shared = MultiAccountWakeCoordinator()
    private var pendingByEmail: [String: PendingAutoWakeRequest] = [:]

    func enqueue(email: String, gpu: String, persistent: Bool) {
        let trimmed = email.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = trimmed.lowercased()
        guard !key.isEmpty else { return }
        pendingByEmail[key] = PendingAutoWakeRequest(
            email: trimmed,
            gpu: gpu,
            persistent: persistent,
            createdAt: Date()
        )
    }

    func consume(email: String) -> PendingAutoWakeRequest? {
        let key = email.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty, let req = pendingByEmail.removeValue(forKey: key) else {
            return nil
        }
        if Date().timeIntervalSince(req.createdAt) > 30.0 {
            return nil
        }
        return req
    }
}

extension Notification.Name {
    static let easyColabTriggerAutoWake = Notification.Name("EasyColab.TriggerAutoWake")
}

struct MultiAccountWakeSheet: View {
    @ObservedObject var vm: StudioViewModel
    @ObservedObject private var i18n = LanguageManager.shared
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openWindow) private var openWindow

    @State private var selectedEmails: Set<String> = []
    @State private var mode: MultiAccountWakeMode = .firstWins
    @State private var hasUserEditedSelection: Bool = false
    @State private var isRefreshingAccounts: Bool = false

    private var orderedSelectedEmails: [String] {
        vm.savedAccounts
            .filter { selectedEmails.contains($0.email.lowercased()) }
            .map { $0.email }
    }

    private var idleAccountCount: Int {
        vm.savedAccounts.filter { ($0.active_assignments ?? 0) == 0 }.count
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "bolt.horizontal.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 3) {
                    Text(L("多账号同时申请 GPU", "Multi-Account Concurrent GPU Wake"))
                        .font(.headline)
                    Text(L(
                        "同时使用多个已绑定 Google 账号向 Colab 提交算力申请，提高抢卡成功率或同时跑多台 GPU。",
                        "Submit GPU allocation requests across multiple saved Google accounts concurrently."
                    ))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                if isRefreshingAccounts {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text(L("申请策略", "Allocation Strategy"))
                    .font(.subheadline.weight(.semibold))

                HStack(spacing: 10) {
                    strategyCard(
                        itemMode: .firstWins,
                        title: L("竞速抢卡（抢到首个即停）", "Race (First Wins)"),
                        subtitle: L(
                            "多个账号并发抢 1 台 GPU，谁先抢到立刻停止其余账号并自动切入连接，防多卡扣费。",
                            "All selected accounts race for 1 GPU; stops losers immediately on first success."
                        ),
                        icon: "flag.checkered"
                    )
                    strategyCard(
                        itemMode: .allParallel,
                        title: L("批量多开（全部各开一台）", "Batch (All Parallel)"),
                        subtitle: L(
                            "同时为所有勾选账号各申请 1 台 GPU，并自动弹出各自独立窗口运行不同任务。",
                            "Provision 1 GPU per selected account in separate isolated windows simultaneously."
                        ),
                        icon: "rectangle.stack.badge.plus"
                    )
                }
            }

            HStack(spacing: 12) {
                HStack(spacing: 6) {
                    Text(L("目标机型", "Target GPU"))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Picker(L("机型", "GPU"), selection: $vm.wakeGPUChoice) {
                        Text("G4 · PRO 6000").tag("G4")
                        Text("H100 · 80GB").tag("H100")
                        Text("A100 · 80GB").tag("A100")
                        Text("L4 · 24GB").tag("L4")
                        Text("T4 · 16GB").tag("T4")
                    }
                    .labelsHidden()
                    .frame(width: 148)
                }

                Toggle(isOn: $vm.persistentWakeEnabled) {
                    Text(L("无卡时每 60s 持续重试", "Auto-retry every 60s"))
                        .font(.callout)
                }
                .toggleStyle(.checkbox)

                Spacer()
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text(L(
                        "参与账号（默认自动跳过已有运行中 VM · 已选 \(orderedSelectedEmails.count)/\(vm.savedAccounts.count)）",
                        "Accounts (Active VMs skipped by default · \(orderedSelectedEmails.count)/\(vm.savedAccounts.count) selected)"
                    ))
                    .font(.subheadline.weight(.semibold))

                    Spacer()

                    Button(L("仅选空闲", "Idle Only")) {
                        hasUserEditedSelection = true
                        selectedIdleAccountsOnly()
                    }
                    .controlSize(.small)
                    .disabled(idleAccountCount == 0)

                    Button(L("全选", "All")) {
                        hasUserEditedSelection = true
                        selectedEmails = Set(vm.savedAccounts.map { $0.email.lowercased() })
                    }
                    .controlSize(.small)

                    Button(L("清空", "Clear")) {
                        hasUserEditedSelection = true
                        selectedEmails.removeAll()
                    }
                    .controlSize(.small)
                }

                ScrollView {
                    VStack(spacing: 6) {
                        ForEach(vm.savedAccounts) { acc in
                            accountSelectionRow(acc)
                        }
                    }
                    .padding(6)
                }
                .frame(minHeight: 140, maxHeight: 230)
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.035)))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))

                if idleAccountCount == 0 && !vm.savedAccounts.isEmpty {
                    Text(L(
                        "提示：当前所有已保存账号均已有运行中的 VM，已默认全部跳过；如需复用或强制重申，请手动勾选。",
                        "Note: All saved accounts currently have active VMs and were skipped by default. Check any account manually to reuse or re-deploy."
                    ))
                    .font(.caption)
                    .foregroundStyle(.orange)
                }
            }

            if mode == .allParallel && orderedSelectedEmails.count > 1 {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                    Text(L(
                        "批量多开将为选中的 \(orderedSelectedEmails.count) 个账号各申请 1 台 GPU 并分别扣除对应账号 CU 额度。",
                        "Batch mode will provision 1 GPU for each of the \(orderedSelectedEmails.count) selected accounts and bill CU per account."
                    ))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 7).fill(Color.orange.opacity(0.10)))
            }

            Divider()

            HStack {
                Button(L("刷新账号状态", "Refresh Status")) {
                    Task { await refreshAccountsSnapshot() }
                }
                .controlSize(.regular)
                .disabled(isRefreshingAccounts)

                Spacer()

                Button(L("取消", "Cancel")) {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Button {
                    startMultiWakeAndDismiss()
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: mode == .firstWins ? "bolt.fill" : "macwindow.badge.plus")
                        Text(primaryConfirmTitle)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(orderedSelectedEmails.isEmpty || vm.isWakingColab)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(18)
        .frame(width: 540)
        .onAppear {
            if !hasUserEditedSelection {
                selectedIdleAccountsOnly()
            }
            Task { await refreshAccountsSnapshot() }
        }
    }

    private var primaryConfirmTitle: String {
        let count = orderedSelectedEmails.count
        if mode == .firstWins {
            return L("开始 \(count) 账号竞速申请", "Race \(count) Accounts")
        }
        return L("同时启动 \(count) 台独立 GPU", "Start \(count) Parallel GPUs")
    }

    private func selectedIdleAccountsOnly() {
        selectedEmails = Set(
            vm.savedAccounts
                .filter { ($0.active_assignments ?? 0) == 0 }
                .map { $0.email.lowercased() }
        )
    }

    private func refreshAccountsSnapshot() async {
        isRefreshingAccounts = true
        await vm.refreshOfficialColabAccountUsage()
        isRefreshingAccounts = false
        if !hasUserEditedSelection {
            selectedIdleAccountsOnly()
        }
    }

    private func strategyCard(itemMode: MultiAccountWakeMode, title: String, subtitle: String, icon: String) -> some View {
        let isSelected = (mode == itemMode)
        return Button {
            mode = itemMode
        } label: {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    Image(systemName: icon)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                    Text(title)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Color.primary)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                    Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                        .font(.system(size: 13))
                        .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                }
                Text(subtitle)
                    .font(.system(size: 10.5))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(10)
            .frame(maxWidth: .infinity, minHeight: 74, alignment: .topLeading)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isSelected ? Color.accentColor.opacity(0.10) : Color.primary.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(isSelected ? Color.accentColor : Color.primary.opacity(0.10), lineWidth: isSelected ? 1.5 : 1)
            )
        }
        .buttonStyle(.plain)
    }

    private func accountSelectionRow(_ acc: SavedGoogleAccount) -> some View {
        let key = acc.email.lowercased()
        let isSelected = selectedEmails.contains(key)
        let activeVMs = acc.active_assignments ?? 0
        let isCurrentWindow = acc.email.caseInsensitiveCompare(vm.accountEmail) == .orderedSame
        return Button {
            hasUserEditedSelection = true
            if isSelected {
                selectedEmails.remove(key)
            } else {
                selectedEmails.insert(key)
            }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                    .font(.system(size: 14))
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(vm.formattedAccountEmail(acc.email))
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(Color.primary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        if isCurrentWindow {
                            Text(L("当前", "Current"))
                                .font(.system(size: 9.5, weight: .semibold))
                                .foregroundStyle(Color.accentColor)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                        }
                    }
                    if let bal = acc.paid_balance {
                        Text(String(format: "%.1f CU", bal))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer(minLength: 6)

                if activeVMs > 0 {
                    let gpuTag = (acc.accelerators ?? []).isEmpty ? "" : "\((acc.accelerators ?? []).joined(separator: "/")) · "
                    Text(L("\(gpuTag)\(activeVMs) VM · 默认跳过", "\(gpuTag)\(activeVMs) VM · Skipped"))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.orange)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2.5)
                        .background(Capsule().fill(Color.orange.opacity(0.14)))
                } else if acc.status_ok == false {
                    Text(L("凭证异常", "Auth Error"))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.red)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2.5)
                        .background(Capsule().fill(Color.red.opacity(0.14)))
                } else {
                    Text(L("空闲可用", "Idle"))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 2.5)
                        .background(Capsule().fill(Color.green.opacity(0.14)))
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.accentColor.opacity(0.08) : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func startMultiWakeAndDismiss() {
        let targets = orderedSelectedEmails
        guard !targets.isEmpty else { return }
        let chosenMode = mode
        dismiss()
        switch chosenMode {
        case .firstWins:
            Task {
                await vm.wakeColabMultiRace(accounts: targets) { winnerEmail in
                    openWindow(id: "account-workspace", value: winnerEmail)
                }
            }
        case .allParallel:
            vm.launchMultiAccountBatchWake(accounts: targets) { accountEmail in
                openWindow(id: "account-workspace", value: accountEmail)
            }
        }
    }
}

// MARK: - 侧边栏免切号多账号实时监控面板

struct MultiAccountLiveMonitorView: View {
    @ObservedObject var vm: StudioViewModel
    @ObservedObject private var i18n = LanguageManager.shared
    @Environment(\.openWindow) private var openWindow
    @AppStorage("easycolab.multiAccountMonitorExpanded") private var isExpanded: Bool = true

    private var activeCount: Int { vm.totalSavedAccountsActiveVMs }
    private var burnRate: Double { vm.totalSavedAccountsBurnRate }
    private var otherActiveCount: Int { vm.otherAccountsWithActiveVMs.count }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 5) {
                Button {
                    withAnimation(.easeInOut(duration: 0.16)) {
                        isExpanded.toggle()
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(.secondary)
                        Text(L("多账号实时监控", "All Accounts Monitor"))
                            .font(.system(size: 10.5, weight: .semibold))
                            .foregroundStyle(.primary)
                    }
                }
                .buttonStyle(.plain)

                Spacer(minLength: 4)

                if activeCount > 0 {
                    Text(L(
                        "\(activeCount) 运行 · \(String(format: "%.1f", burnRate)) CU/h",
                        "\(activeCount) Active · \(String(format: "%.1f", burnRate)) CU/h"
                    ))
                    .font(.system(size: 9.5, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.green)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1.5)
                    .background(Capsule().fill(Color.green.opacity(0.14)))
                } else {
                    Text(L(
                        "全员空闲 · \(String(format: "%.0f", vm.totalSavedAccountsCUBalance)) CU",
                        "All Idle · \(String(format: "%.0f", vm.totalSavedAccountsCUBalance)) CU"
                    ))
                    .font(.system(size: 9.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1.5)
                    .background(Capsule().fill(Color.primary.opacity(0.06)))
                }

                Button {
                    Task { await vm.refreshOfficialColabAccountUsage(includeAllAccounts: true) }
                } label: {
                    if vm.isRefreshingAllAccounts {
                        ProgressView().controlSize(.mini)
                    } else {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
                .disabled(vm.isRefreshingAllAccounts)
                .help(L("立即并发刷新全部已保存账号的实时 GPU 与 CU 状态（不切换当前账号）", "Refresh live GPU & CU status across all saved accounts concurrently without switching"))
            }

            if !isExpanded && otherActiveCount > 0 {
                Button {
                    withAnimation(.easeInOut(duration: 0.16)) { isExpanded = true }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "bolt.circle.fill")
                            .font(.system(size: 10))
                            .foregroundStyle(.orange)
                        Text(L(
                            "另有 \(otherActiveCount) 个非当前账号有实例运行（点击展开）",
                            "\(otherActiveCount) other account(s) running VMs (expand)"
                        ))
                        .font(.system(size: 9.5, weight: .medium))
                        .foregroundStyle(.orange)
                    }
                }
                .buttonStyle(.plain)
            }

            if isExpanded {
                VStack(spacing: 4) {
                    ForEach(vm.savedAccounts) { acc in
                        liveMonitorRow(acc)
                    }
                }
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.primary.opacity(0.035)))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.primary.opacity(0.07), lineWidth: 1))
    }

    private func liveMonitorRow(_ acc: SavedGoogleAccount) -> some View {
        let isCurrent = vm.cliLoggedIn && acc.email.caseInsensitiveCompare(vm.accountEmail) == .orderedSame
        let activeVMs = acc.active_assignments ?? 0
        let hasFailure = (acc.status_ok == false)
        let accels = (acc.accelerators ?? []).joined(separator: "/")
        let dotColor: Color = hasFailure ? .red : (activeVMs > 0 ? .green : .secondary.opacity(0.45))

        return VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 5) {
                Circle()
                    .fill(dotColor)
                    .frame(width: 6, height: 6)

                Text(vm.formattedAccountEmail(acc.email))
                    .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                    .help(acc.email)

                if isCurrent {
                    Text(L("当前", "Current"))
                        .font(.system(size: 8.5, weight: .bold))
                        .foregroundStyle(Color.accentColor)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.accentColor.opacity(0.14)))
                }

                Spacer(minLength: 2)

                if !isCurrent {
                    if !vm.isPinnedMultiAccountWindow {
                        Button {
                            Task { await vm.switchSavedGoogleAccount(acc.email) }
                        } label: {
                            Image(systemName: "arrow.left.arrow.right")
                                .font(.system(size: 9.5))
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        .help(L("在当前窗口切换到 \(acc.email)", "Switch this window to \(acc.email)"))
                    }

                    Button {
                        openWindow(id: "account-workspace", value: acc.email)
                    } label: {
                        Image(systemName: "macwindow.badge.plus")
                            .font(.system(size: 9.5))
                            .foregroundStyle(Color.accentColor)
                    }
                    .buttonStyle(.plain)
                    .help(L("新窗口运行 \(acc.email)", "Open \(acc.email) in new window"))
                }

                if activeVMs > 0 {
                    Button {
                        vm.pendingStopAccountEmail = acc.email
                    } label: {
                        Text(L("释放", "Stop"))
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(.red)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(Capsule().fill(Color.red.opacity(0.12)))
                    }
                    .buttonStyle(.plain)
                    .help(L("释放账号 \(acc.email) 下的全部运行中实例", "Terminate active VMs under \(acc.email)"))
                }
            }

            HStack(spacing: 6) {
                if let bal = acc.paid_balance {
                    Text(String(format: "%.1f CU", bal))
                        .font(.system(size: 9.5, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                if hasFailure {
                    Text(L("凭证异常", "Auth Error"))
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(.red)
                        .help(acc.error ?? "OAuth failed")
                } else if activeVMs > 0 {
                    let gpuPrefix = accels.isEmpty ? "" : "\(accels) · "
                    let rateVal = acc.burn_rate_hourly ?? 0.0
                    let rateSuffix = rateVal > 0 ? String(format: " · %.1f/h", rateVal) : ""
                    Text("\(gpuPrefix)\(activeVMs) VM\(rateSuffix)")
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.green)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.green.opacity(0.12)))
                } else {
                    Text(L("空闲", "Idle"))
                        .font(.system(size: 9))
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)
            }
            .padding(.leading, 11)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(activeVMs > 0 ? Color.green.opacity(0.06) : Color.primary.opacity(0.025))
        )
    }
}
