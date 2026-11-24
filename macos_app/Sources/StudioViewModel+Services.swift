import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import ObjectiveC.runtime

// MARK: - StudioViewModel 扩展：Google OAuth 桥接、自定义 HF 模型热榜与详情、ComfyUI 与网关同步

final class KeepalivePipeLineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()

    func appendAndExtractLines(_ chunk: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        pending.append(chunk)
        var lines: [String] = []
        while let nlIdx = pending.firstIndex(of: 0x0A) {
            let lineData = pending.subdata(in: 0..<nlIdx)
            pending.removeSubrange(0...nlIdx)
            if let lineStr = String(data: lineData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !lineStr.isEmpty {
                lines.append(lineStr)
            }
        }
        return lines
    }

    func flushRemainingLine() -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard !pending.isEmpty else { return nil }
        let rem = pending
        pending.removeAll()
        if let lineStr = String(data: rem, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !lineStr.isEmpty {
            return lineStr
        }
        return nil
    }

    func appendAndExtractPayloads(_ chunk: Data) -> [String] {
        appendAndExtractLines(chunk).compactMap { lineStr in
            guard lineStr.hasPrefix("KEEPALIVE_JSON:") else { return nil }
            return String(lineStr.dropFirst("KEEPALIVE_JSON:".count))
        }
    }
}

extension StudioViewModel {
    // MARK: - 本地 colab_bridge.py 交互与零配置环境自举
    func ensureBridgePythonBin(installIfMissing: Bool) async -> String? {
        if let cached = cachedBridgePythonBin, FileManager.default.isExecutableFile(atPath: cached) {
            return cached
        }
        let existing = await Task.detached(priority: .userInitiated) {
            LocalRuntimePaths.findReadyBridgePython()
        }.value
        if let existing {
            cachedBridgePythonBin = existing
            return existing
        }
        guard installIfMissing else { return nil }

        guard let sysPy = LocalRuntimePaths.findSystemPython3() else {
            statusMessage = L(
                "未检测到 macOS Python 3 环境，请先安装 Xcode Command Line Tools (xcode-select --install)。",
                "macOS Python 3 not found. Please install Xcode Command Line Tools (xcode-select --install)."
            )
            return nil
        }

        let venvDir = LocalRuntimePaths.managedVenvDir
        let venvPy = venvDir.appendingPathComponent("bin/python").path
        let reqFile = LocalRuntimePaths.resolveBundledOrRepoFile("requirements.txt")
        let hasReqFile = FileManager.default.fileExists(atPath: reqFile)

        statusMessage = L(
            "首次使用正在自动初始化本地 Colab 连接依赖环境 (约 15~30 秒)...",
            "Initializing local Colab bridge Python dependencies for first-time setup (~15–30s)..."
        )
        liveConsoleLines.append(L(
            "[本地初始化] 正在于 ~/Library/Application Support/EasyColab/venv 安装 google-colab-cli 依赖...",
            "[Local Init] Installing google-colab-cli dependencies in ~/Library/Application Support/EasyColab/venv..."
        ))

        let ok = await Task.detached(priority: .userInitiated) { () -> Bool in
            let mkProc = Process()
            mkProc.executableURL = URL(fileURLWithPath: sysPy)
            mkProc.arguments = ["-m", "venv", venvDir.path]
            mkProc.standardOutput = Pipe()
            mkProc.standardError = Pipe()
            do {
                try mkProc.run()
                mkProc.waitUntilExit()
                guard mkProc.terminationStatus == 0 else { return false }
            } catch {
                return false
            }

            let pipProc = Process()
            pipProc.executableURL = URL(fileURLWithPath: venvPy)
            if hasReqFile {
                pipProc.arguments = ["-m", "pip", "install", "-q", "--upgrade", "pip", "-r", reqFile]
            } else {
                pipProc.arguments = [
                    "-m", "pip", "install", "-q", "--upgrade", "pip",
                    "google-colab-cli>=0.7.5",
                    "google-auth>=2.28.0",
                    "google-auth-oauthlib>=1.2.0",
                    "requests>=2.31.0",
                    "websocket-client>=1.7.0"
                ]
            }
            pipProc.standardOutput = Pipe()
            pipProc.standardError = Pipe()
            do {
                try pipProc.run()
                pipProc.waitUntilExit()
                return pipProc.terminationStatus == 0
            } catch {
                return false
            }
        }.value

        if ok, FileManager.default.isExecutableFile(atPath: venvPy) {
            cachedBridgePythonBin = venvPy
            liveConsoleLines.append(L(
                "[本地初始化] Colab 本地连接组件已就绪。",
                "[Local Init] Local Colab bridge environment is ready."
            ))
            return venvPy
        } else {
            statusMessage = L(
                "自动安装本地连接依赖失败，请检查网络或执行 pip install -r requirements.txt。",
                "Failed to install local bridge dependencies. Check your network or run pip install -r requirements.txt."
            )
            return nil
        }
    }

    nonisolated static func extractJSONPayload(from rawOutput: String) -> String {
        let trimmed = rawOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let lines = trimmed.components(separatedBy: .newlines)
        for line in lines.reversed() {
            let candidate = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if candidate.hasPrefix("{") && candidate.hasSuffix("}") {
                return candidate
            }
        }
        return trimmed
    }

    func runBridgeCommand(_ args: [String], installIfMissing: Bool = false, scopedAccount: String? = nil) async -> String {
        guard let pyBin = await ensureBridgePythonBin(installIfMissing: installIfMissing) else {
            return ""
        }
        let script = self.bridgeScript
        let cmdName = args.first ?? ""
        let scopedCommands = ["auth_status", "auth_status_all", "wake", "recover_tunnel", "keepalive", "keepalive_daemon", "stop_all"]
        var envOverrides: [String: String] = ["EASYCOLAB_FROM_APP": "1"]
        let resolvedAcct = (scopedAccount ?? activeBridgeAccountEmail ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !resolvedAcct.isEmpty && scopedCommands.contains(cmdName) {
            envOverrides["EASYCOLAB_ACCOUNT"] = resolvedAcct
        }
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: pyBin)
                proc.environment = ProcessInfo.processInfo.environment.merging(envOverrides) { _, new in new }
                proc.arguments = [script] + args
                let outPipe = Pipe()
                proc.standardOutput = outPipe
                proc.standardError = FileHandle.nullDevice
                do {
                    try proc.run()
                    let data = outPipe.fileHandleForReading.readDataToEndOfFile()
                    proc.waitUntilExit()
                    let out = String(data: data, encoding: .utf8) ?? ""
                    continuation.resume(returning: Self.extractJSONPayload(from: out))
                } catch {
                    continuation.resume(returning: "")
                }
            }
        }
    }

    func refreshOfficialColabAccountUsage(includeAllAccounts: Bool = true) async {
        if includeAllAccounts && isRefreshingAllAccounts {
            return
        }
        if includeAllAccounts {
            isRefreshingAllAccounts = true
        }
        defer {
            if includeAllAccounts {
                isRefreshingAllAccounts = false
            }
        }

        var decoded: BridgeAuthStatusResponse? = nil
        if includeAllAccounts {
            let outAll = await runBridgeCommand(["auth_status_all"], installIfMissing: false)
            if let data = outAll.data(using: .utf8),
               let parsed = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) {
                decoded = parsed
            }
        }
        if decoded == nil {
            let out = await runBridgeCommand(["auth_status"], installIfMissing: false)
            if let data = out.data(using: .utf8),
               let parsed = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) {
                decoded = parsed
            }
        }
        guard let decoded else { return }

        if isPinnedMultiAccountWindow {
            if let logged = decoded.logged_in {
                cliLoggedIn = logged
            }
        } else {
            cliLoggedIn = decoded.logged_in ?? false
        }
        if let em = decoded.email, !em.isEmpty {
            accountEmail = em
        } else if !cliLoggedIn {
            accountEmail = ""
        }
        if let bal = decoded.paid_balance {
            officialCUBalance = bal
            totalBudgetCU = bal
        }
        if let rate = decoded.burn_rate_hourly { officialBurnRate = rate }
        if let accs = decoded.accounts { savedAccounts = accs }
        lastAllAccountsSyncAt = Date()
        if let count = decoded.active_assignments {
            activeVMCount = count
            if count == 0 && !isConnected && (!loadedModelsDetail.isEmpty || vramTotalGB > 0) {
                resetStaleRemoteRuntimeState(reason: L(
                    "云端 Colab 实例当前未运行，已自动清理历史显存状态；点击左侧「唤醒实例并部署」即可启动。",
                    "No active cloud Colab runtime detected; cleared stale VRAM state. Click 'Wake & Deploy' on the left to launch."
                ))
            }
        }
    }

    func stopAccountRemoteVMs(_ email: String) async {
        let target = email.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !target.isEmpty else { return }
        if target.caseInsensitiveCompare(accountEmail.trimmingCharacters(in: .whitespacesAndNewlines)) == .orderedSame {
            await disconnectRemoteColab()
            return
        }
        statusMessage = L(
            "正在远程释放账号 \(target) 下的 Colab 实例（不切换当前窗口账号）...",
            "Terminating cloud Colab VMs under \(target) without switching accounts..."
        )
        let out = await runBridgeCommand(["stop_all"], installIfMissing: false, scopedAccount: target)
        if let data = out.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data),
           decoded.ok == true {
            let msg = decoded.message ?? L(
                "已释放账号 \(target) 下全部实例。",
                "Released all Colab instances under \(target)."
            )
            statusMessage = msg
            liveConsoleLines.append("[Colab-CLI] [\(target)] \(msg)")
        } else {
            statusMessage = L(
                "释放账号 \(target) 实例失败，请检查凭证或稍后重试。",
                "Failed to release instances under \(target)."
            )
        }
        await refreshOfficialColabAccountUsage(includeAllAccounts: true)
    }

    func startGoogleOAuthLogin(switchingAccount: Bool = false) async {
        let out = await runBridgeCommand(["auth_url"], installIfMissing: true)
        guard let data = out.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let authURLStr = json["auth_url"] as? String,
              let url = URL(string: authURLStr) else {
            if !statusMessage.contains("初始化") && !statusMessage.contains("Initializ") && !statusMessage.contains("Python") {
                statusMessage = L("获取 Google 授权链接失败", "Failed to obtain Google OAuth authorization URL")
            }
            return
        }
        oauthCodeText = ""
        showOAuthCodeInput = true
        NSWorkspace.shared.open(url)
        statusMessage = switchingAccount
            ? L("已在浏览器打开 Google 账号选择页，请选择新账号授权并粘贴授权码（点取消可保留原账号）。", "Opened Google account chooser in browser. Authorize a new account and paste the code (Cancel keeps current account).")
            : L("已在浏览器打开 Google 授权页，请允许访问后粘贴授权码。", "Opened Google authorization page in browser. Grant access and paste the auth code.")
    }

    func logoutGoogleAccount() async {
        stopActiveWakeProcess()
        stopKeepaliveDaemon()
        let out = await runBridgeCommand(["auth_logout"], installIfMissing: false)
        if let data = out.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data),
           let accs = decoded.accounts {
            savedAccounts = accs
        }
        cliLoggedIn = false
        accountEmail = ""
        showOAuthCodeInput = false
        oauthCodeText = ""
        officialCUBalance = nil
        officialBurnRate = nil
        activeVMCount = 0
        colabServerURL = ""
        resetStaleRemoteRuntimeState(reason: L(
            "已退出当前 Google 账号（历史账号仍保留在换号菜单中，点击即可切回）。",
            "Signed out of current Google account (saved accounts remain available in the Account menu)."
        ))
    }

    func switchSavedGoogleAccount(_ email: String) async {
        stopActiveWakeProcess()
        stopKeepaliveDaemon()
        let label = formattedAccountEmail(email)
        statusMessage = L("正在切换至 Google 账号 \(label)...", "Switching to Google account \(label)...")
        let out = await runBridgeCommand(["auth_switch", email], installIfMissing: false, scopedAccount: "")
        guard let data = out.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            statusMessage = L("切换 Google 账号失败。", "Failed to switch Google account.")
            return
        }
        if let accs = decoded.accounts { savedAccounts = accs }
        guard decoded.logged_in == true else {
            statusMessage = L(
                "切换失败：\(decoded.error ?? "凭证无效，请重新授权")",
                "Switch failed: \(decoded.error ?? "invalid credentials, please re-authorize")"
            )
            return
        }
        let targetEmail = decoded.email ?? email
        let savedURL = UserDefaults.standard.string(forKey: "easycolab.serverURL.\(targetEmail.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())") ?? ""
        colabServerURL = savedURL
        resetStaleRemoteRuntimeState(reason: "")
        cliLoggedIn = true
        accountEmail = targetEmail
        officialCUBalance = decoded.paid_balance
        officialBurnRate = decoded.burn_rate_hourly
        activeVMCount = decoded.active_assignments ?? 0
        if let bal = decoded.paid_balance { totalBudgetCU = bal }
        let balText = decoded.paid_balance != nil ? String(format: "%.2f CU", decoded.paid_balance!) : ""
        statusMessage = L(
            "已切换至账号 \(displayAccountLabel)\(balText.isEmpty ? "" : "（余额 \(balText)）")",
            "Switched to \(displayAccountLabel)\(balText.isEmpty ? "" : " (balance \(balText))")"
        )
        if !savedURL.isEmpty {
            await checkColabStatus(silent: true)
        }
    }

    func removeSavedGoogleAccount(_ email: String) async {
        let wasActive = (accountEmail.caseInsensitiveCompare(email) == .orderedSame)
        let out = await runBridgeCommand(["auth_remove", email], installIfMissing: false, scopedAccount: "")
        guard let data = out.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            return
        }
        if let accs = decoded.accounts { savedAccounts = accs }
        if wasActive {
            stopActiveWakeProcess()
            stopKeepaliveDaemon()
            colabServerURL = ""
            resetStaleRemoteRuntimeState(reason: "")
            cliLoggedIn = decoded.logged_in ?? false
            accountEmail = decoded.email ?? ""
            officialCUBalance = decoded.paid_balance
            officialBurnRate = decoded.burn_rate_hourly
            activeVMCount = decoded.active_assignments ?? 0
            if let bal = decoded.paid_balance { totalBudgetCU = bal }
        }
        statusMessage = L(
            "已移除账号 \(formattedAccountEmail(email))。",
            "Removed saved account \(formattedAccountEmail(email))."
        )
    }

    func submitGoogleOAuthCode() async {
        let code = oauthCodeText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !code.isEmpty else { return }
        let prevEmail = accountEmail
        let wasLoggedIn = cliLoggedIn
        statusMessage = L("正在验证授权码并同步账户余额...", "Verifying authorization code & syncing CU balance...")
        let out = await runBridgeCommand(["auth_code", code], installIfMissing: true)
        guard let data = out.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            statusMessage = L("授权验证失败，请重试。", "Authorization verification failed. Please try again.")
            return
        }
        if let accs = decoded.accounts { savedAccounts = accs }
        if decoded.logged_in == true {
            let newEmail = decoded.email ?? ""
            if wasLoggedIn && (prevEmail != newEmail || !colabServerURL.isEmpty) {
                colabServerURL = ""
                resetStaleRemoteRuntimeState(reason: "")
            }
            cliLoggedIn = true
            accountEmail = newEmail
            showOAuthCodeInput = false
            oauthCodeText = ""
            officialCUBalance = decoded.paid_balance
            officialBurnRate = decoded.burn_rate_hourly
            activeVMCount = decoded.active_assignments ?? 0
            if let bal = decoded.paid_balance {
                totalBudgetCU = bal
            }
            let emailTag = newEmail.isEmpty ? "" : " (\(displayAccountLabel))"
            statusMessage = L(
                "Google 账号绑定成功\(emailTag)！可用额度: \(String(format: "%.2f", decoded.paid_balance ?? 0)) CU",
                "Google account bound\(emailTag)! Available balance: \(String(format: "%.2f", decoded.paid_balance ?? 0)) CU"
            )
        } else {
            statusMessage = L(
                "授权码无效: \(decoded.error ?? "请重新获取")",
                "Invalid authorization code: \(decoded.error ?? "please generate a new code")"
            )
        }
    }

    func stopActiveWakeProcess() {
        if let proc = activeWakeProcess, proc.isRunning {
            proc.terminate()
        }
        activeWakeProcess = nil
    }

    func cancelActiveWake() {
        stopActiveWakeProcess()
        isWakingColab = false
        activeMultiWakeAccountCount = 0
        let msg = L("已手动停止持续申请部署。", "Cancelled continuous wake & deploy.")
        liveConsoleLines.append("[Colab-CLI] \(msg)")
        statusMessage = msg
    }

    func handleWakeOutputLine(_ line: String, gpu: String, password: String) {
        if line.hasPrefix("WAKE_URL:") {
            let url = String(line.dropFirst("WAKE_URL:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            self.colabServerURL = url
            self.adminPassword = password
            self.wakeObtainedURLThisRun = true
            self.liveConsoleLines.append(L(
                "成功获取云端隧道地址: \(self.maskedURLDisplay(url))",
                "Obtained public tunnel URL: \(self.maskedURLDisplay(url))"
            ))
            self.statusMessage = L(
                "云端 \(gpu) 实例已唤醒就绪，正在同步模型状态...",
                "Cloud \(gpu) instance is ready. Syncing model status..."
            )
        } else if line.hasPrefix("WAKE_ERROR:") {
            let errBody = String(line.dropFirst("WAKE_ERROR:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            let displayErr = errBody.isEmpty ? line : errBody
            self.liveConsoleLines.append(displayErr)
            self.statusMessage = L("唤醒失败: \(displayErr)", "Wake failed: \(displayErr)")
        } else if line.hasPrefix("WAKE_LOG:") {
            let clean = String(line.dropFirst("WAKE_LOG:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            if !clean.isEmpty {
                self.liveConsoleLines.append(clean)
                self.statusMessage = clean
            }
        } else if !line.hasPrefix("[colab]") && !line.contains("Assignment request failed") && !line.contains("urllib3") {
            self.liveConsoleLines.append(line)
        }
    }

    func wakeColabHeadless() async {
        guard cliLoggedIn else {
            statusMessage = L("请先绑定 Google 账号后再使用免网页一键唤醒。", "Please bind your Google account before using headless 1-click wake.")
            return
        }
        stopActiveWakeProcess()
        stopKeepaliveDaemon()
        wakeObtainedURLThisRun = false
        isWakingColab = true
        statusMessage = L(
            "正在通过 Google Colab API 后台分配 \(wakeGPUChoice) 实例并部署多模型引擎...",
            "Provisioning \(wakeGPUChoice) runtime via Google Colab API & deploying multi-model engine..."
        )
        liveConsoleLines = [L(
            "[Colab-CLI] 正在启动免网页一键唤醒流程 (\(wakeGPUChoice) GPU)...",
            "[Colab-CLI] Starting headless 1-click wake (\(wakeGPUChoice) GPU)..."
        )]

        guard let pyBin = await ensureBridgePythonBin(installIfMissing: true) else {
            isWakingColab = false
            return
        }
        let script = self.bridgeScript
        let gpu = self.wakeGPUChoice
        let password = UUID().uuidString + UUID().uuidString
        let idleMin = self.idleTimeoutMinutes
        let appPy = self.appPyPath
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pyBin)
        var wakeEnv: [String: String] = [
            "EASYCOLAB_FROM_APP": "1",
            "EASYCOLAB_ADMIN_PASSWORD": password,
            "EASYCOLAB_IDLE_TIMEOUT_MINUTES": "\(idleMin)"
        ]
        if let scopedAcct = activeBridgeAccountEmail {
            wakeEnv["EASYCOLAB_ACCOUNT"] = scopedAcct
        }
        proc.environment = ProcessInfo.processInfo.environment.merging(wakeEnv) { _, new in new }
        var wakeArgs = [script, "wake", "--gpu", gpu, "--app-py", appPy]
        if persistentWakeEnabled {
            wakeArgs += ["--persistent", "--retry-interval", "60"]
        }
        proc.arguments = wakeArgs
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        self.activeWakeProcess = proc

        let lineBuffer = KeepalivePipeLineBuffer()

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                    let data = handle.availableData
                    guard !data.isEmpty else { return }
                    let lines = lineBuffer.appendAndExtractLines(data)
                    guard !lines.isEmpty else { return }
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        for line in lines {
                            self.handleWakeOutputLine(line, gpu: gpu, password: password)
                        }
                    }
                }

                do {
                    try proc.run()
                    proc.waitUntilExit()
                } catch {
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.statusMessage = L(
                            "启动唤醒进程失败: \(error.localizedDescription)",
                            "Failed to launch wake process: \(error.localizedDescription)"
                        )
                    }
                }
                pipe.fileHandleForReading.readabilityHandler = nil
                let tailData = pipe.fileHandleForReading.readDataToEndOfFile()
                var finalLines = lineBuffer.appendAndExtractLines(tailData)
                if let rem = lineBuffer.flushRemainingLine() {
                    finalLines.append(rem)
                }
                let capturedFinal = finalLines
                Task { @MainActor [weak self] in
                    if let self {
                        for line in capturedFinal {
                            self.handleWakeOutputLine(line, gpu: gpu, password: password)
                        }
                    }
                    continuation.resume()
                }
            }
        }

        if self.activeWakeProcess === proc {
            self.activeWakeProcess = nil
        }
        isWakingColab = false
        await refreshOfficialColabAccountUsage()
        if wakeObtainedURLThisRun && !colabServerURL.isEmpty {
            await checkColabStatus()
            await ensureKeepaliveDaemonRunning()
        }
    }

    func wakeColabMultiRace(
        accounts: [String],
        openWinnerWindow: @escaping @MainActor (String) -> Void
    ) async {
        var targets: [String] = []
        var seenNorm: Set<String> = []
        for raw in accounts {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let norm = trimmed.lowercased()
            if !trimmed.isEmpty, !seenNorm.contains(norm) {
                seenNorm.insert(norm)
                targets.append(trimmed)
            }
        }
        guard !targets.isEmpty else {
            statusMessage = L("请至少选择一个 Google 账号。", "Please select at least one Google account.")
            return
        }

        if targets.count == 1 {
            let single = targets[0]
            if single.caseInsensitiveCompare(accountEmail) == .orderedSame {
                await wakeColabHeadless()
                return
            }
            if !isPinnedMultiAccountWindow {
                await switchSavedGoogleAccount(single)
                if cliLoggedIn && accountEmail.caseInsensitiveCompare(single) == .orderedSame {
                    await wakeColabHeadless()
                }
                return
            }
            MultiAccountWakeCoordinator.shared.enqueue(
                email: single,
                gpu: wakeGPUChoice,
                persistent: persistentWakeEnabled
            )
            openWinnerWindow(single)
            try? await Task.sleep(nanoseconds: 250_000_000)
            NotificationCenter.default.post(
                name: .easyColabTriggerAutoWake,
                object: nil,
                userInfo: ["email": single]
            )
            return
        }

        stopActiveWakeProcess()
        stopKeepaliveDaemon()
        wakeObtainedURLThisRun = false
        isWakingColab = true
        activeMultiWakeAccountCount = targets.count
        statusMessage = L(
            "正在使用 \(targets.count) 个账号同时竞速申请 \(wakeGPUChoice) GPU（抢到首个即停）...",
            "Racing \(targets.count) accounts concurrently for \(wakeGPUChoice) GPU (first wins)..."
        )
        liveConsoleLines = [L(
            "[Colab-CLI] 启动多账号并发竞速申请 (\(wakeGPUChoice) GPU · \(targets.count) 个账号)...",
            "[Colab-CLI] Starting concurrent multi-account GPU race (\(wakeGPUChoice) GPU · \(targets.count) accounts)..."
        )]

        guard let pyBin = await ensureBridgePythonBin(installIfMissing: true) else {
            isWakingColab = false
            activeMultiWakeAccountCount = 0
            return
        }

        let script = self.bridgeScript
        let gpu = self.wakeGPUChoice
        let password = UUID().uuidString + UUID().uuidString
        let idleMin = self.idleTimeoutMinutes
        let appPy = self.appPyPath
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pyBin)
        var wakeEnv: [String: String] = [
            "EASYCOLAB_FROM_APP": "1",
            "EASYCOLAB_ADMIN_PASSWORD": password,
            "EASYCOLAB_IDLE_TIMEOUT_MINUTES": "\(idleMin)"
        ]
        wakeEnv.removeValue(forKey: "EASYCOLAB_ACCOUNT")
        var mergedEnv = ProcessInfo.processInfo.environment.merging(wakeEnv) { _, new in new }
        mergedEnv.removeValue(forKey: "EASYCOLAB_ACCOUNT")
        proc.environment = mergedEnv

        var wakeArgs = [
            script, "wake_multi",
            "--gpu", gpu,
            "--app-py", appPy,
            "--accounts", targets.joined(separator: ","),
            "--mode", "first"
        ]
        if persistentWakeEnabled {
            wakeArgs += ["--persistent", "--retry-interval", "60"]
        }
        proc.arguments = wakeArgs
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        self.activeWakeProcess = proc

        var raceWinnerEmail: String? = nil
        var raceWinnerURL: String? = nil
        let lineBuffer = KeepalivePipeLineBuffer()

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                    let data = handle.availableData
                    guard !data.isEmpty else { return }
                    let lines = lineBuffer.appendAndExtractLines(data)
                    guard !lines.isEmpty else { return }
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        for line in lines {
                            if line.hasPrefix("WAKE_WINNER:") {
                                let w = String(line.dropFirst("WAKE_WINNER:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                                if !w.isEmpty {
                                    raceWinnerEmail = w
                                }
                            } else if line.hasPrefix("WAKE_URL:") {
                                let url = String(line.dropFirst("WAKE_URL:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                                if !url.isEmpty {
                                    raceWinnerURL = url
                                    self.wakeObtainedURLThisRun = true
                                    self.liveConsoleLines.append(L(
                                        "成功获取云端隧道地址: \(self.maskedURLDisplay(url))",
                                        "Obtained public tunnel URL: \(self.maskedURLDisplay(url))"
                                    ))
                                    self.statusMessage = L(
                                        "竞速成功！云端 \(gpu) 实例已就绪，正在切换连接...",
                                        "Race won! Cloud \(gpu) instance ready, connecting..."
                                    )
                                }
                            } else {
                                self.handleWakeOutputLine(line, gpu: gpu, password: password)
                            }
                        }
                    }
                }

                do {
                    try proc.run()
                    proc.waitUntilExit()
                } catch {
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.statusMessage = L(
                            "启动多账号竞速失败: \(error.localizedDescription)",
                            "Failed to launch multi-account race: \(error.localizedDescription)"
                        )
                    }
                }
                pipe.fileHandleForReading.readabilityHandler = nil
                let tailData = pipe.fileHandleForReading.readDataToEndOfFile()
                var finalLines = lineBuffer.appendAndExtractLines(tailData)
                if let rem = lineBuffer.flushRemainingLine() {
                    finalLines.append(rem)
                }
                let capturedFinal = finalLines
                Task { @MainActor [weak self] in
                    if let self {
                        for line in capturedFinal {
                            if line.hasPrefix("WAKE_WINNER:") {
                                let w = String(line.dropFirst("WAKE_WINNER:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                                if !w.isEmpty {
                                    raceWinnerEmail = w
                                }
                            } else if line.hasPrefix("WAKE_URL:") {
                                let url = String(line.dropFirst("WAKE_URL:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                                if !url.isEmpty {
                                    raceWinnerURL = url
                                    self.wakeObtainedURLThisRun = true
                                }
                            } else {
                                self.handleWakeOutputLine(line, gpu: gpu, password: password)
                            }
                        }
                    }
                    continuation.resume()
                }
            }
        }

        if self.activeWakeProcess === proc {
            self.activeWakeProcess = nil
        }
        isWakingColab = false
        activeMultiWakeAccountCount = 0

        if let wonURL = raceWinnerURL, !wonURL.isEmpty {
            let winner = (raceWinnerEmail?.isEmpty == false ? raceWinnerEmail! : accountEmail)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            Self.saveAdminPassword(password, for: wonURL)
            if !winner.isEmpty {
                UserDefaults.standard.set(wonURL, forKey: Self.perAccountURLDefaultsKey(for: winner))
            }

            if !isPinnedMultiAccountWindow {
                if !winner.isEmpty && winner.caseInsensitiveCompare(accountEmail) != .orderedSame {
                    let swOut = await runBridgeCommand(["auth_switch", winner], installIfMissing: false, scopedAccount: "")
                    if let data = swOut.data(using: .utf8),
                       let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) {
                        if let accs = decoded.accounts { savedAccounts = accs }
                        if decoded.logged_in == true {
                            cliLoggedIn = true
                            accountEmail = decoded.email ?? winner
                            officialCUBalance = decoded.paid_balance
                            officialBurnRate = decoded.burn_rate_hourly
                            activeVMCount = decoded.active_assignments ?? 0
                            if let bal = decoded.paid_balance { totalBudgetCU = bal }
                        }
                    }
                    resetStaleRemoteRuntimeState(reason: "")
                }
                self.colabServerURL = wonURL
                self.adminPassword = password
                await checkColabStatus()
                await ensureKeepaliveDaemonRunning()
            } else if winner.caseInsensitiveCompare(accountEmail) == .orderedSame {
                self.colabServerURL = wonURL
                self.adminPassword = password
                await checkColabStatus()
                await ensureKeepaliveDaemonRunning()
            } else {
                openWinnerWindow(winner)
                statusMessage = L(
                    "竞速成功！胜出账号 \(winner) 已在新独立窗口连接 \(gpu) 实例。",
                    "Race won! Opening \(winner) in its own window for \(gpu) runtime."
                )
            }
        }
        await refreshOfficialColabAccountUsage()
    }

    func launchMultiAccountBatchWake(
        accounts: [String],
        openAccountWindow: @escaping @MainActor (String) -> Void
    ) {
        var targets: [String] = []
        var seenNorm: Set<String> = []
        for raw in accounts {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let norm = trimmed.lowercased()
            if !trimmed.isEmpty, !seenNorm.contains(norm) {
                seenNorm.insert(norm)
                targets.append(trimmed)
            }
        }
        guard !targets.isEmpty else { return }

        let currentAcct = (boundAccountEmail ?? accountEmail).trimmingCharacters(in: .whitespacesAndNewlines)
        var otherTargets: [String] = []
        var runInCurrentWindow = false
        for acc in targets {
            if !runInCurrentWindow && !currentAcct.isEmpty && acc.caseInsensitiveCompare(currentAcct) == .orderedSame && cliLoggedIn && !isWakingColab {
                runInCurrentWindow = true
            } else {
                otherTargets.append(acc)
            }
        }

        let gpu = self.wakeGPUChoice
        let persistent = self.persistentWakeEnabled
        statusMessage = L(
            "正在为 \(targets.count) 个账号同步并发唤醒独立 \(gpu) GPU 窗口...",
            "Launching parallel \(gpu) GPU wake across \(targets.count) account windows..."
        )
        liveConsoleLines.append(L(
            "[Colab-CLI] 批量多开启动：\(targets.joined(separator: ", ")) (\(gpu) GPU)",
            "[Colab-CLI] Batch multi-GPU wake queued: \(targets.joined(separator: ", ")) (\(gpu) GPU)"
        ))

        if runInCurrentWindow {
            Task { await self.wakeColabHeadless() }
        }

        guard !otherTargets.isEmpty else { return }
        Task { @MainActor in
            for acc in otherTargets {
                MultiAccountWakeCoordinator.shared.enqueue(
                    email: acc,
                    gpu: gpu,
                    persistent: persistent
                )
                openAccountWindow(acc)
                try? await Task.sleep(nanoseconds: 250_000_000)
                NotificationCenter.default.post(
                    name: .easyColabTriggerAutoWake,
                    object: nil,
                    userInfo: ["email": acc]
                )
            }
        }
    }

    func consumePendingAutoWakeIfNeeded() async {
        let targetAcct = (boundAccountEmail ?? accountEmail).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !targetAcct.isEmpty, !isWakingColab else { return }
        guard let req = MultiAccountWakeCoordinator.shared.consume(email: targetAcct) else { return }
        wakeGPUChoice = req.gpu
        persistentWakeEnabled = req.persistent
        await wakeColabHeadless()
    }

    func recoverRunningColabConnection(forceRestartServer: Bool = false) async {
        guard cliLoggedIn, !isRecoveringTunnel, !isWakingColab else { return }
        isRecoveringTunnel = true
        defer { isRecoveringTunnel = false }

        statusMessage = L(
            "正在核实云端实例状态并尝试无损恢复连接...",
            "Verifying cloud runtime & attempting lossless tunnel recovery..."
        )
        liveConsoleLines.append(L(
            "[Colab-CLI] 开始无损核实云端实例并恢复隧道连接...",
            "[Colab-CLI] Verifying cloud runtime & recovering tunnel..."
        ))

        await refreshOfficialColabAccountUsage(includeAllAccounts: true)
        if activeVMCount == 0 {
            lastConnectionError = ""
            let msg = L(
                "已核实云端无运行中实例（0 VM · 0.00 CU/h），实例已被释放、未计费。",
                "Confirmed: 0 active VMs (0.00 CU/h). Instance is released and not billing."
            )
            statusMessage = msg
            liveConsoleLines.append("[Colab-CLI] \(msg)")
            return
        }

        if !forceRestartServer && !normalizedBaseURL.isEmpty && !adminPassword.isEmpty {
            await checkColabStatus(silent: false)
            if isConnected {
                lastConnectionError = ""
                await ensureKeepaliveDaemonRunning()
                return
            }
        }

        let needsRestartImmediately = forceRestartServer || lastConnectionError.contains("401")
        guard let pyBin = await ensureBridgePythonBin(installIfMissing: false) else { return }
        let script = self.bridgeScript
        let appPy = self.appPyPath
        let freshPassword = UUID().uuidString + UUID().uuidString
        let idleMin = self.idleTimeoutMinutes
        let scopedAcct = self.activeBridgeAccountEmail

        func runRecoverSubprocess(restart: Bool) async -> (recoveredURL: String?, restartedURL: String?, released: Bool, needRestart: Bool, errorMsg: String?) {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: pyBin)
            var envOverrides: [String: String] = [
                "EASYCOLAB_FROM_APP": "1",
                "EASYCOLAB_ADMIN_PASSWORD": freshPassword,
                "EASYCOLAB_IDLE_TIMEOUT_MINUTES": "\(idleMin)"
            ]
            if let scopedAcct, !scopedAcct.isEmpty {
                envOverrides["EASYCOLAB_ACCOUNT"] = scopedAcct
            }
            proc.environment = ProcessInfo.processInfo.environment.merging(envOverrides) { _, new in new }
            var recArgs = [script, "recover_tunnel", "--app-py", appPy]
            if restart {
                recArgs.append("--restart-server")
            }
            proc.arguments = recArgs
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe

            let lineBuffer = KeepalivePipeLineBuffer()
            var foundRecURL: String? = nil
            var foundWakeURL: String? = nil
            var isReleased = false
            var isNeedRestart = false
            var foundError: String? = nil

            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                    let handleLines: ([String]) -> Void = { lines in
                        for line in lines {
                            if line.hasPrefix("RECOVER_URL:") {
                                foundRecURL = String(line.dropFirst("RECOVER_URL:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                            } else if line.hasPrefix("WAKE_URL:") {
                                foundWakeURL = String(line.dropFirst("WAKE_URL:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                            } else if line.hasPrefix("RECOVER_RELEASED:") {
                                isReleased = true
                                foundError = String(line.dropFirst("RECOVER_RELEASED:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                            } else if line.hasPrefix("RECOVER_NEED_RESTART:") {
                                isNeedRestart = true
                            } else if line.hasPrefix("RECOVER_ERROR:") {
                                foundError = String(line.dropFirst("RECOVER_ERROR:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                            } else if line.hasPrefix("WAKE_LOG:") {
                                let clean = String(line.dropFirst("WAKE_LOG:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                                if !clean.isEmpty {
                                    Task { @MainActor [weak self] in
                                        self?.liveConsoleLines.append(clean)
                                        self?.statusMessage = clean
                                    }
                                }
                            }
                        }
                    }

                    pipe.fileHandleForReading.readabilityHandler = { handle in
                        let data = handle.availableData
                        guard !data.isEmpty else { return }
                        let lines = lineBuffer.appendAndExtractLines(data)
                        if !lines.isEmpty { handleLines(lines) }
                    }
                    do {
                        try proc.run()
                        proc.waitUntilExit()
                    } catch {
                        foundError = error.localizedDescription
                    }
                    pipe.fileHandleForReading.readabilityHandler = nil
                    let tail = pipe.fileHandleForReading.readDataToEndOfFile()
                    var tailLines = lineBuffer.appendAndExtractLines(tail)
                    if let rem = lineBuffer.flushRemainingLine() { tailLines.append(rem) }
                    if !tailLines.isEmpty { handleLines(tailLines) }
                    continuation.resume()
                }
            }
            return (foundRecURL, foundWakeURL, isReleased, isNeedRestart, foundError)
        }

        var res = await runRecoverSubprocess(restart: needsRestartImmediately)
        if res.released {
            activeVMCount = 0
            officialBurnRate = 0.0
            lastConnectionError = ""
            statusMessage = res.errorMsg ?? L("已确认云端实例释放。", "Cloud runtime confirmed released.")
            await refreshOfficialColabAccountUsage(includeAllAccounts: true)
            return
        }

        if let recURL = res.recoveredURL, !recURL.isEmpty {
            self.colabServerURL = recURL
            if !self.adminPassword.isEmpty {
                await checkColabStatus(silent: false)
                if isConnected {
                    await ensureKeepaliveDaemonRunning()
                    await refreshOfficialColabAccountUsage(includeAllAccounts: true)
                    return
                }
            }
            // 本地缺失密码、密码不匹配(401)或隧道探测波动时，自动在同一台运行中实例上无损重建 app.py 隧道
            res = await runRecoverSubprocess(restart: true)
        } else if res.needRestart && !needsRestartImmediately {
            res = await runRecoverSubprocess(restart: true)
        }

        if let newURL = res.restartedURL, !newURL.isEmpty {
            Self.saveAdminPassword(freshPassword, for: newURL)
            self.colabServerURL = newURL
            self.adminPassword = freshPassword
            await checkColabStatus(silent: false)
            if isConnected {
                await ensureKeepaliveDaemonRunning()
            }
        } else if let err = res.errorMsg, !err.isEmpty {
            lastConnectionError = err
            statusMessage = err
            liveConsoleLines.append("[Colab-CLI] \(err)")
        }
        await refreshOfficialColabAccountUsage(includeAllAccounts: true)
    }

    func stopKeepaliveDaemon() {
        keepaliveDaemonStdoutPipe?.fileHandleForReading.readabilityHandler = nil
        try? keepaliveDaemonStdinPipe?.fileHandleForWriting.close()
        if let proc = keepaliveDaemonProcess, proc.isRunning {
            proc.terminate()
        }
        keepaliveDaemonProcess = nil
        keepaliveDaemonStdinPipe = nil
        keepaliveDaemonStdoutPipe = nil
        keepaliveProxyWSConnected = false
    }

    func handleKeepaliveDaemonPayloadLine(_ jsonLine: String) {
        guard let data = jsonLine.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            return
        }
        if let logged = decoded.logged_in {
            cliLoggedIn = logged
        }
        if let wsConn = decoded.ws_connected {
            keepaliveProxyWSConnected = wsConn
        }
        if let bal = decoded.paid_balance {
            officialCUBalance = bal
            totalBudgetCU = bal
        }
        if let rate = decoded.burn_rate_hourly {
            officialBurnRate = rate
        }
        if let count = decoded.active_assignments {
            activeVMCount = count
        }
    }

    func ensureKeepaliveDaemonRunning() async {
        guard cliLoggedIn, isConnected, !isWakingColab else { return }
        if let existing = keepaliveDaemonProcess, existing.isRunning {
            return
        }
        guard !isStartingKeepaliveDaemon else { return }
        isStartingKeepaliveDaemon = true
        defer { isStartingKeepaliveDaemon = false }

        guard let pyBin = await ensureBridgePythonBin(installIfMissing: false) else { return }
        guard cliLoggedIn, isConnected, !isWakingColab else { return }
        if let existing = keepaliveDaemonProcess, existing.isRunning {
            return
        }

        stopKeepaliveDaemon()

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pyBin)
        var kaEnv: [String: String] = ["EASYCOLAB_FROM_APP": "1"]
        if let scopedAcct = activeBridgeAccountEmail {
            kaEnv["EASYCOLAB_ACCOUNT"] = scopedAcct
        }
        proc.environment = ProcessInfo.processInfo.environment.merging(kaEnv) { _, new in new }
        proc.arguments = [self.bridgeScript, "keepalive_daemon"]
        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        proc.standardInput = stdinPipe
        proc.standardOutput = stdoutPipe
        proc.standardError = FileHandle.nullDevice

        let lineBuffer = KeepalivePipeLineBuffer()

        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else { return }
            let completeLines = lineBuffer.appendAndExtractPayloads(chunk)

            if !completeLines.isEmpty {
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    for payload in completeLines {
                        self.handleKeepaliveDaemonPayloadLine(payload)
                    }
                }
            }
        }

        proc.terminationHandler = { [weak self] exitedProc in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.keepaliveDaemonProcess === exitedProc {
                    self.keepaliveDaemonStdoutPipe?.fileHandleForReading.readabilityHandler = nil
                    try? self.keepaliveDaemonStdinPipe?.fileHandleForWriting.close()
                    self.keepaliveDaemonProcess = nil
                    self.keepaliveDaemonStdinPipe = nil
                    self.keepaliveDaemonStdoutPipe = nil
                    self.keepaliveProxyWSConnected = false
                }
            }
        }

        do {
            try proc.run()
            self.keepaliveDaemonProcess = proc
            self.keepaliveDaemonStdinPipe = stdinPipe
            self.keepaliveDaemonStdoutPipe = stdoutPipe
        } catch {
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            try? stdinPipe.fileHandleForWriting.close()
        }
    }

    func keepaliveColabRuntime() async {
        guard cliLoggedIn, !isWakingColab else { return }
        if isConnected {
            await ensureKeepaliveDaemonRunning()
            if keepaliveDaemonProcess?.isRunning == true {
                return
            }
        }
        let out = await runBridgeCommand(["keepalive"])
        guard let data = out.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(BridgeAuthStatusResponse.self, from: data) else {
            return
        }
        if let bal = decoded.paid_balance {
            officialCUBalance = bal
            totalBudgetCU = bal
        }
        if let rate = decoded.burn_rate_hourly { officialBurnRate = rate }
        if let accs = decoded.accounts { savedAccounts = accs }
        if let count = decoded.active_assignments { activeVMCount = count }
    }

    var activeSessionFileURL: URL {
        LocalRuntimePaths.appSupportDir.appendingPathComponent("active_session.json")
    }

    func checkExternalActiveSessionIfNeeded() async {
        let candidates = [
            activeSessionFileURL,
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".config/colab-minimax-h3/session.json")
        ]
        var newestObj: [String: Any]? = nil
        var newestUpdatedAt: Double = 0
        for fileURL in candidates {
            guard FileManager.default.fileExists(atPath: fileURL.path),
                  let data = try? Data(contentsOf: fileURL),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            var ts = (obj["updated_at"] as? Double) ?? 0
            if ts <= 0,
               let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
               let mdate = attrs[.modificationDate] as? Date {
                ts = mdate.timeIntervalSince1970
            }
            if ts > newestUpdatedAt {
                newestUpdatedAt = ts
                newestObj = obj
            }
        }
        guard let obj = newestObj, newestUpdatedAt > lastObservedExternalSessionUpdatedAt else {
            return
        }
        lastObservedExternalSessionUpdatedAt = newestUpdatedAt
        let isSleeping = (obj["sleeping"] as? Bool) ?? false
        let rawURL = ((obj["url"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanURL = rawURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let pw = (obj["password"] as? String) ?? ""
        let src = (obj["source"] as? String) ?? "external-cli"
        let sessAcct = ((obj["account"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if let pinnedAcct = boundAccountEmail, !pinnedAcct.isEmpty {
            if sessAcct.isEmpty || sessAcct.caseInsensitiveCompare(pinnedAcct) != .orderedSame {
                return
            }
        }

        if isSleeping {
            if isConnected && attachedExternalSessionSource != nil {
                attachedExternalSessionSource = nil
                resetStaleRemoteRuntimeState(reason: L(
                    "检测到外部 Skill / CLI 已释放并休眠 Colab 实例，监控状态已同步归零。",
                    "External Skill/CLI put Colab to sleep; live monitor synced to offline."
                ))
                await refreshOfficialColabAccountUsage()
            }
            return
        }

        if !cleanURL.isEmpty && cleanURL.hasPrefix("https://") {
            let currentClean = normalizedBaseURL
            if !isConnected || currentClean != cleanURL {
                colabServerURL = cleanURL
                if !pw.isEmpty {
                    adminPassword = pw
                }
                attachedExternalSessionSource = src
                await checkColabStatus(silent: true)
            }
        }
    }

    func startBackgroundTelemetryLoop() {
        heartbeatTask?.cancel()
        heartbeatTask = Task {
            await checkExternalActiveSessionIfNeeded()
            if !normalizedBaseURL.isEmpty {
                await checkColabStatus(silent: true)
            }
            var tick = 0
            while !Task.isCancelled {
                let sleepNs: UInt64 = liveMonitoringEnabled ? 1_500_000_000 : 4_000_000_000
                try? await Task.sleep(nanoseconds: sleepNs)
                tick += 1
                guard liveMonitoringEnabled else { continue }

                await checkExternalActiveSessionIfNeeded()
                if isConnected {
                    // 生成任务进行中由 pollColabJob 附带 telemetry 实时更新图表，每 15s 低频同步全局状态
                    if !isGenerating || tick % 10 == 0 {
                        await checkColabStatus(silent: true)
                    }
                    if cliLoggedIn && !isWakingColab {
                        await ensureKeepaliveDaemonRunning()
                    }
                } else {
                    if keepaliveDaemonProcess != nil {
                        stopKeepaliveDaemon()
                    }
                    if !normalizedBaseURL.isEmpty && tick % 4 == 0 {
                        await checkColabStatus(silent: true)
                    }
                }

                // 实时监控全部已保存 Google 账号：离线每 ~15s、多账号在线每 ~30s 并发刷新（无需切换账号）
                if !isWakingColab {
                    if !isConnected && (cliLoggedIn || !savedAccounts.isEmpty) && tick % 10 == 0 {
                        await refreshOfficialColabAccountUsage(includeAllAccounts: true)
                    } else if isConnected && cliLoggedIn {
                        if keepaliveDaemonProcess?.isRunning != true && tick % 14 == 0 {
                            await keepaliveColabRuntime()
                        }
                        if savedAccounts.count >= 2 && tick % 20 == 0 {
                            await refreshOfficialColabAccountUsage(includeAllAccounts: true)
                        }
                    }
                }
            }
        }
    }

    var externalCLILogFileURL: URL {
        LocalRuntimePaths.appSupportDir.appendingPathComponent("external_cli_logs.jsonl")
    }

    func startExternalCLILogWatcher() {
        externalLogWatcherTask?.cancel()
        let logURL = externalCLILogFileURL

        if let attrs = try? FileManager.default.attributesOfItem(atPath: logURL.path),
           let modDate = attrs[.modificationDate] as? Date,
           let sizeNum = attrs[.size] as? NSNumber {
            let size = sizeNum.uint64Value
            if Date().timeIntervalSince(modDate) < 12 * 3600, size > 0 {
                let tailOffset: UInt64 = size > 32 * 1024 ? (size - 32 * 1024) : 0
                if let handle = try? FileHandle(forReadingFrom: logURL) {
                    try? handle.seek(toOffset: tailOffset)
                    let chunk = handle.readDataToEndOfFile()
                    try? handle.close()
                    let parsed = Self.parseExternalLogChunk(chunk, skipPartialFirstLine: tailOffset > 0)
                    if !parsed.isEmpty {
                        externalCLILogLines = Array(parsed.suffix(40))
                        rebuildMergedConsoleLines()
                    }
                }
            }
            externalLogFileOffset = size
        }

        externalLogWatcherTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 600_000_000)
                guard let self else { break }
                self.pollNewExternalCLILogLines()
            }
        }
    }

    private static func parseExternalLogChunk(_ data: Data, skipPartialFirstLine: Bool = false) -> [String] {
        guard !data.isEmpty, let raw = String(data: data, encoding: .utf8) else { return [] }
        var rawLines = raw.components(separatedBy: .newlines)
        if skipPartialFirstLine && !rawLines.isEmpty {
            rawLines.removeFirst()
        }
        var result: [String] = []
        for entryStr in rawLines {
            let trimmed = entryStr.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if let entryData = trimmed.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: entryData) as? [String: Any],
               let lineText = obj["line"] as? String {
                let clean = lineText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !clean.isEmpty {
                    result.append(clean)
                }
            } else {
                result.append(trimmed)
            }
        }
        return result
    }

    func pollNewExternalCLILogLines() {
        let logURL = externalCLILogFileURL
        guard FileManager.default.fileExists(atPath: logURL.path) else { return }
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: logURL.path),
              let sizeNum = attrs[.size] as? NSNumber else { return }
        let fileSize = sizeNum.uint64Value
        if fileSize < externalLogFileOffset {
            externalLogFileOffset = 0
        }
        guard fileSize > externalLogFileOffset else { return }
        guard let handle = try? FileHandle(forReadingFrom: logURL) else { return }
        defer { try? handle.close() }
        do {
            try handle.seek(toOffset: externalLogFileOffset)
            let data = handle.readDataToEndOfFile()
            externalLogFileOffset = fileSize
            let added = Self.parseExternalLogChunk(data)
            guard !added.isEmpty else { return }
            externalCLILogLines.append(contentsOf: added)
            if externalCLILogLines.count > 250 {
                externalCLILogLines.removeFirst(externalCLILogLines.count - 250)
            }
            rebuildMergedConsoleLines()
            if !isWakingColab && !isGenerating && !isDownloadingWeights && !isLoadingModel,
               let latest = added.last {
                statusMessage = latest
            }
        } catch {
            return
        }
    }

    func rebuildMergedConsoleLines() {
        if !remoteConsoleLines.isEmpty && !externalCLILogLines.isEmpty {
            liveConsoleLines = remoteConsoleLines + ["─── 外部 Skill / CLI 实时日志 ───"] + externalCLILogLines
        } else if !externalCLILogLines.isEmpty {
            liveConsoleLines = externalCLILogLines
        } else if !remoteConsoleLines.isEmpty {
            liveConsoleLines = remoteConsoleLines
        }
    }

    // MARK: - 本地历史媒体库（支持 MP4 视频与 PNG/JPG 图片，兼容旧目录）
    func refreshLocalMedia() {
        var allURLs: [URL] = []
        for dir in [localSaveDir] + legacySaveDirs {
            if let urls = try? FileManager.default.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.creationDateKey],
                options: [.skipsHiddenFiles]
            ) {
                allURLs.append(contentsOf: urls)
            }
        }

        let validExts: Set<String> = ["mp4", "png", "jpg", "jpeg", "webp"]
        var seenNames = Set<String>()
        let filtered = allURLs.filter { url in
            guard validExts.contains(url.pathExtension.lowercased()) else { return false }
            return seenNames.insert(url.lastPathComponent).inserted
        }
        let items: [LocalMediaItem] = filtered.compactMap { url in
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let date = (attrs?[.creationDate] as? Date) ?? Date.distantPast
            let ext = url.pathExtension.lowercased()
            let fname = url.lastPathComponent
            return LocalMediaItem(
                url: url,
                filename: fname,
                isImage: ext != "mp4",
                createdAt: date,
                elapsedSeconds: mediaElapsedMap[fname]
            )
        }.sorted(by: { $0.createdAt > $1.createdAt })

        self.localMediaItems = items
        if currentMediaURL == nil, let first = items.first {
            selectMedia(first.url)
        }
    }

    func selectMedia(_ url: URL) {
        currentMediaURL = url
        let ext = url.pathExtension.lowercased()
        if ext == "mp4" {
            currentPreviewImage = nil
            player = AVPlayer(url: url)
            player?.play()
        } else {
            player?.pause()
            player = nil
            currentPreviewImage = NSImage(contentsOf: url)
        }
    }

    func applyStatusResponse(_ decoded: ColabStatusResponse) {
        isConnected = true
        gpuName = decoded.gpu_name ?? "Colab GPU"
        vramFreeGB = decoded.vram_free_gb ?? 0.0
        vramTotalGB = decoded.vram_total_gb ?? 0.0
        vramUsedGB = decoded.vram_used_gb ?? 0.0
        gpuUtilPct = decoded.gpu_util_pct ?? gpuUtilPct
        gpuTempC = decoded.gpu_temp_c ?? gpuTempC
        gpuPowerW = decoded.gpu_power_w ?? gpuPowerW
        lastMonitorSyncAt = Date()
        ramTotalGB = decoded.ram_total_gb ?? ramTotalGB
        ramUsedGB = decoded.ram_used_gb ?? ramUsedGB
        ramFreeGB = decoded.ram_free_gb ?? ramFreeGB
        diskTotalGB = decoded.disk_total_gb ?? diskTotalGB
        diskUsedGB = decoded.disk_used_gb ?? diskUsedGB
        diskFreeGB = decoded.disk_free_gb ?? diskFreeGB
        shmTotalGB = decoded.shm_total_gb ?? shmTotalGB
        shmUsedGB = decoded.shm_used_gb ?? shmUsedGB
        shmFreeGB = decoded.shm_free_gb ?? shmFreeGB
        driveTotalGB = decoded.drive_total_gb ?? driveTotalGB
        driveUsedGB = decoded.drive_used_gb ?? driveUsedGB
        driveFreeGB = decoded.drive_free_gb ?? driveFreeGB
        recordTelemetrySample()
        if let remoteCustom = decoded.custom_models {
            mergeRemoteCustomModels(remoteCustom)
        }
        loadedModelId = decoded.loaded_model_id
        if let ids = decoded.loaded_model_ids {
            loadedModelIds = ids
        } else if let single = decoded.loaded_model_id, decoded.model_loaded == true {
            loadedModelIds = [single]
        } else if decoded.any_model_loaded == false {
            loadedModelIds = []
        }
        if let details = decoded.loaded_models_detail {
            loadedModelsDetail = details.map { item in
                var clean = item
                clean.name = stripEmoji(item.name)
                if let m = item.mode { clean.mode = stripEmoji(m) }
                return clean
            }
        } else if decoded.any_model_loaded == false {
            loadedModelsDetail = []
        }
        modelLoaded = loadedModelIds.contains(selectedModelId) || (decoded.model_loaded ?? false)
        currentModelName = decoded.current_model ?? L(Self.defaultModelNameZh, Self.defaultModelNameEn)
        loadedContextLengthTokens = decoded.loaded_context_length ?? 0

        if let remoteCtx = decoded.context_length {
            if remoteCtx == contextLengthTokens {
                remoteContextSynced = true
            } else if !remoteContextSynced && Date().timeIntervalSince(lastContextSyncAttemptAt) >= 8.0 {
                lastContextSyncAttemptAt = Date()
                Task { await syncContextLengthToColab() }
            }
        }

        if let files = decoded.files, !files.isEmpty {
            modelFiles = files
        }
        downloadedCount = decoded.downloaded_count ?? downloadedCount
        totalFileCount = decoded.total_count ?? activeModelSpec.defaultFiles.count
        allDownloaded = decoded.all_downloaded ?? (downloadedCount == totalFileCount && totalFileCount > 0)
        totalDiskSizeGB = decoded.total_size_gb ?? totalDiskSizeGB
        drivePersisted = decoded.drive_persisted ?? false
        bf16ShardsCount = decoded.bf16_shards_count ?? 0
        isDownloadingWeights = decoded.is_downloading ?? false
        isLoadingModel = decoded.is_loading_vram ?? false
        syncMacOSRuntimeActivityAssertion()
        if cliLoggedIn && !isWakingColab && keepaliveDaemonProcess?.isRunning != true {
            Task { await ensureKeepaliveDaemonRunning() }
        }
        if let p = decoded.task_progress { modelTaskProgress = p }

        uptimeSeconds = decoded.uptime_seconds ?? uptimeSeconds
        cuBurnRatePerHour = decoded.cu_burn_rate_per_hour ?? cuBurnRatePerHour
        sessionCUUsed = decoded.session_cu_used ?? sessionCUUsed
        idleRemainingSeconds = decoded.idle_remaining_seconds ?? idleRemainingSeconds
        if let remoteIdle = decoded.idle_timeout_minutes {
            if remoteIdle == idleTimeoutMinutes {
                remoteIdleSynced = true
            } else if Date().timeIntervalSince(lastIdleSyncAttemptAt) >= 8.0 {
                remoteIdleSynced = false
                lastIdleSyncAttemptAt = Date()
                let targetIdle = idleTimeoutMinutes
                Task { await updateIdleTimeoutConfig(targetIdle) }
            }
        }
        apiRequestCount = decoded.api_request_count ?? apiRequestCount
        if decoded.api_key_configured == false && !remoteApiKeySynced {
            remoteApiKeySynced = true
            Task { await syncApiKeyToColab(initializeOnly: true) }
        }
        if let lines = decoded.console_lines, !lines.isEmpty {
            remoteConsoleLines = lines
            rebuildMergedConsoleLines()
        }
        if let cf = decoded.comfyui {
            comfyRunning = cf.is_running ?? false
            comfyStarting = cf.is_starting ?? false
            comfySyncingDrive = cf.is_syncing_drive ?? false
            comfyDownloadingWeight = cf.is_downloading_weight ?? false
            comfyInstallingPackId = cf.installing_pack_id ?? ""
            comfyProgress = cf.progress ?? 0.0
            if let sm = cf.status_message, !sm.isEmpty {
                comfyStatusMessage = sm
            }
            comfyPublicURL = cf.public_url ?? ""
            comfyQueueRunning = cf.queue_running ?? 0
            comfyQueuePending = cf.queue_pending ?? 0
            if let pl = cf.persistence_label, !pl.isEmpty {
                comfyPersistenceLabel = pl
            }
            comfyDrivePersisted = cf.drive_persisted ?? drivePersisted
            if let pr = cf.persist_root, !pr.isEmpty {
                comfyPersistRoot = pr
            }
            comfyCustomNodesCount = cf.custom_nodes_count ?? 0
            comfyModelsCount = cf.models_count ?? 0
            comfyWorkflowsCount = cf.workflows_count ?? 0
            if let outs = cf.recent_outputs {
                comfyRecentOutputs = outs
            }
            if let packs = cf.preset_packs, !packs.isEmpty {
                comfyPresetPacks = packs
            }
        }
    }

    func openComfyUIInBrowser() {
        guard let url = URL(string: comfyPublicURL), !comfyPublicURL.isEmpty else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: - 自定义 HuggingFace 模型仓库注册与管理

    func loadPersistedCustomModels() {
        guard let data = UserDefaults.standard.data(forKey: customModelsDefaultsKey),
              let dtos = try? JSONDecoder().decode([CustomHFModelDTO].self, from: data) else { return }
        let filtered = dtos.filter { $0.repo_id.lowercased() != "minimaxai/minimax-h3-image" }
        customModels = filtered.map { OpenModelSpec.fromCustomDTO($0) }
        if filtered.count != dtos.count {
            savePersistedCustomModels()
        }
    }

    func savePersistedCustomModels() {
        let dtos: [CustomHFModelDTO] = customModels.map { spec in
            let modStr: String
            switch spec.modality {
            case .videoAudio, .video: modStr = "video_audio"
            case .image: modStr = "image"
            case .llmVLM, .comfyUI: modStr = "llm_vlm"
            }
            return CustomHFModelDTO(
                id: spec.id,
                name: spec.nameZh,
                repo_id: spec.repoId,
                include_pattern: spec.includePattern,
                modality: modStr,
                pipeline_arch: nil,
                category: spec.summaryZh,
                approx_size_gb: spec.approxSizeGB,
                is_custom: true
            )
        }
        if let data = try? JSONEncoder().encode(dtos) {
            UserDefaults.standard.set(data, forKey: customModelsDefaultsKey)
        }
    }

    func mergeRemoteCustomModels(_ remoteDTOs: [CustomHFModelDTO]) {
        let validRemote = remoteDTOs.filter { $0.repo_id.lowercased() != "minimaxai/minimax-h3-image" }
        var existingById: [String: OpenModelSpec] = [:]
        for item in customModels where item.repoId.lowercased() != "minimaxai/minimax-h3-image" {
            existingById[item.id] = item
        }
        for dto in validRemote {
            existingById[dto.id] = OpenModelSpec.fromCustomDTO(dto)
        }
        let orderedIds = validRemote.map { $0.id } + customModels.map { $0.id }.filter { id in !validRemote.contains(where: { $0.id == id }) }
        customModels = orderedIds.compactMap { existingById[$0] }
        savePersistedCustomModels()
    }

    func normalizeHFRepoId(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://huggingface.co/", "http://huggingface.co/", "https://hf.co/", "http://hf.co/", "huggingface.co/", "hf.co/"] {
            if s.lowercased().hasPrefix(prefix) {
                s = String(s.dropFirst(prefix.count))
                break
            }
        }
        let parts = s.split(separator: "/").map(String.init)
        if parts.count >= 2 {
            return "\(parts[0])/\(parts[1])"
        }
        return s
    }

    func fetchTrendingHFModels(category: String? = nil, searchQuery: String = "", forceRefresh: Bool = false) async {
        let cat = (category ?? trendingHFCategory).trimmingCharacters(in: .whitespacesAndNewlines)
        let sq = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        if trendingHFCategory != cat {
            trendingHFCategory = cat
        }
        isLoadingTrendingHF = true
        defer { isLoadingTrendingHF = false }

        let base = normalizedBaseURL
        if isConnected, !base.isEmpty {
            var comps = URLComponents(string: "\(base)/minimax_api/hf_trending")
            comps?.queryItems = [
                URLQueryItem(name: "category", value: cat),
                URLQueryItem(name: "limit", value: "12"),
                URLQueryItem(name: "hf_token", value: hfToken),
                URLQueryItem(name: "force_refresh", value: forceRefresh ? "true" : "false"),
                URLQueryItem(name: "search_query", value: sq)
            ]
            if let url = comps?.url, var req = try? adminRequest(url) {
                req.timeoutInterval = 12
                if let (data, resp) = try? await adminSession.data(for: req),
                   (resp as? HTTPURLResponse)?.statusCode == 200,
                   let dec = try? JSONDecoder().decode(HFTrendingAPIResponse.self, from: data),
                   let models = dec.models, !models.isEmpty {
                    trendingHFModels = models
                    return
                }
            }
        }

        let expandQS = "&expand[]=safetensors&expand[]=gguf&expand[]=likes&expand[]=downloads&expand[]=pipeline_tag&expand[]=tags&expand[]=gated"
        let encodedSQ = sq.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        let searchQS = encodedSQ.isEmpty ? "" : "&search=\(encodedSQ)"

        let queryPaths: [String]
        switch cat {
        case "gguf":
            queryPaths = ["/api/models?sort=trendingScore&direction=-1&filter=gguf&limit=20\(searchQS)\(expandQS)"]
        case "llm":
            if sq.isEmpty {
                queryPaths = [
                    "/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-generation&limit=18\(expandQS)",
                    "/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-text-to-text&limit=10\(expandQS)"
                ]
            } else {
                queryPaths = ["/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-generation&limit=20\(searchQS)\(expandQS)"]
            }
        case "image_video":
            queryPaths = [
                "/api/models?sort=trendingScore&direction=-1&pipeline_tag=text-to-image&limit=12\(searchQS)\(expandQS)",
                "/api/models?sort=trendingScore&direction=-1&pipeline_tag=image-to-video&limit=10\(searchQS)\(expandQS)"
            ]
        default:
            queryPaths = ["/api/models?sort=trendingScore&direction=-1&limit=22\(searchQS)\(expandQS)"]
        }

        let tok = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        var parsed: [HFTrendingModelItem] = []
        var seen = Set<String>()

        for qpath in queryPaths {
            var rawArr: [[String: Any]]? = nil
            for host in ["https://huggingface.co", "https://hf-mirror.com"] {
                guard let url = URL(string: "\(host)\(qpath)") else { continue }
                var req = URLRequest(url: url)
                req.timeoutInterval = 10
                req.setValue("EasyColab-macOS/2.0", forHTTPHeaderField: "User-Agent")
                if !tok.isEmpty {
                    req.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization")
                }
                if let (data, resp) = try? await URLSession.shared.data(for: req),
                   (resp as? HTTPURLResponse)?.statusCode == 200,
                   let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] {
                    rawArr = arr
                    break
                }
            }
            guard let arr = rawArr else { continue }
            for obj in arr {
                let rid = ((obj["id"] as? String) ?? (obj["modelId"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                guard rid.contains("/"), !seen.contains(rid) else { continue }
                seen.insert(rid)
                let ridLow = rid.lowercased()
                let ptag = ((obj["pipeline_tag"] as? String) ?? "").lowercased()
                let tags = ((obj["tags"] as? [String]) ?? []).map { $0.lowercased() }
                let isGGUF = ridLow.contains("gguf") || tags.contains("gguf")
                let gatedVal = obj["gated"]
                let isGated: Bool = {
                    if let b = gatedVal as? Bool { return b }
                    if let s = gatedVal as? String {
                        let sl = s.lowercased()
                        return !sl.isEmpty && sl != "false" && sl != "0" && sl != "none"
                    }
                    return false
                }()
                if sq.isEmpty && cat == "all" && !isGGUF {
                    if UNSUPPORTED_HF_PIPELINE_TAGS.contains(ptag) { continue }
                    if !ptag.isEmpty && !SUPPORTED_TRENDING_PIPELINE_TAGS.contains(ptag) { continue }
                    if ptag.isEmpty && !tags.contains("transformers") && !tags.contains("diffusers") && !tags.contains("text-generation") {
                        continue
                    }
                }
                let modality: String
                if ptag.contains("video") || tags.contains("text-to-video") || tags.contains("image-to-video") {
                    modality = "video_audio"
                } else if ptag.contains("text-to-image") || ptag.contains("image-to-image") || (tags.contains("diffusers") && !isGGUF) {
                    modality = "image"
                } else {
                    modality = "llm_vlm"
                }
                let shortName = rid.split(separator: "/").last.map(String.init) ?? rid
                let sizing = analyzeHFModelSizing(obj, repoId: rid, isGGUF: isGGUF)
                parsed.append(HFTrendingModelItem(
                    repo_id: rid,
                    name: shortName,
                    pipeline_tag: ptag.isEmpty ? (isGGUF ? "gguf" : "text-generation") : ptag,
                    modality: modality,
                    is_gguf: isGGUF,
                    is_gated: isGated,
                    suggested_pattern: isGGUF ? "*Q4_K_M*.gguf" : "",
                    likes: (obj["likes"] as? Int) ?? 0,
                    downloads: (obj["downloads"] as? Int) ?? 0,
                    param_count: sizing.paramCount,
                    param_label: sizing.paramLabel,
                    est_vram_gb: sizing.estVramGb,
                    deploy_status: sizing.deployStatus,
                    deploy_label_zh: sizing.deployZh,
                    deploy_label_en: sizing.deployEn
                ))
                if parsed.count >= 12 { break }
            }
            if parsed.count >= 12 { break }
        }
        if !parsed.isEmpty || !sq.isEmpty {
            trendingHFModels = parsed
        }
    }

    func fetchHFModelDetail(repoId: String) async {
        let cleanId = normalizeHFRepoId(repoId)
        guard cleanId.contains("/") else { return }
        showHFModelDetailOverlay = true
        isLoadingHFModelDetail = true
        defer { isLoadingHFModelDetail = false }

        let base = normalizedBaseURL
        let tok = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        if isConnected, !base.isEmpty {
            var comps = URLComponents(string: "\(base)/minimax_api/hf_model_detail")
            comps?.queryItems = [
                URLQueryItem(name: "repo_id", value: cleanId),
                URLQueryItem(name: "hf_token", value: tok)
            ]
            if let url = comps?.url, var req = try? adminRequest(url) {
                req.timeoutInterval = 14
                if let (data, resp) = try? await adminSession.data(for: req),
                   (resp as? HTTPURLResponse)?.statusCode == 200,
                   var dec = try? JSONDecoder().decode(HFModelDetailInfo.self, from: data),
                   dec.ok == true {
                    dec.readme_excerpt = cleanHFReadmeExcerpt(dec.readme_excerpt)
                    activeHFModelDetail = dec
                    return
                }
            }
        }

        // Offline / direct HuggingFace API fallback
        var metaObj: [String: Any]? = nil
        for host in ["https://huggingface.co", "https://hf-mirror.com"] {
            guard let url = URL(string: "\(host)/api/models/\(cleanId)?blobs=true") else { continue }
            var req = URLRequest(url: url)
            req.timeoutInterval = 12
            req.setValue("EasyColab-macOS/2.0", forHTTPHeaderField: "User-Agent")
            if !tok.isEmpty {
                req.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization")
            }
            if let (data, resp) = try? await URLSession.shared.data(for: req),
               (resp as? HTTPURLResponse)?.statusCode == 200,
               let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                metaObj = dict
                break
            }
        }
        guard let meta = metaObj else { return }

        let rid = ((meta["id"] as? String) ?? (meta["modelId"] as? String) ?? cleanId).trimmingCharacters(in: .whitespacesAndNewlines)
        let ridLow = rid.lowercased()
        let ptag = ((meta["pipeline_tag"] as? String) ?? "").lowercased()
        let tags = ((meta["tags"] as? [String]) ?? []).map { $0.lowercased() }
        let ggufDict = meta["gguf"] as? [String: Any]
        let isGGUF = ridLow.contains("gguf") || tags.contains("gguf") || (ggufDict != nil && !(ggufDict?.isEmpty ?? true))
        let gatedVal = meta["gated"]
        let isGated: Bool = {
            if let b = gatedVal as? Bool { return b }
            if let s = gatedVal as? String {
                let sl = s.lowercased()
                return !sl.isEmpty && sl != "false" && sl != "0" && sl != "none"
            }
            return false
        }()
        let modality: String
        if ptag.contains("video") || tags.contains("text-to-video") || tags.contains("image-to-video") {
            modality = "video_audio"
        } else if ptag.contains("text-to-image") || ptag.contains("image-to-image") || (tags.contains("diffusers") && !isGGUF) {
            modality = "image"
        } else {
            modality = "llm_vlm"
        }
        let cardData = meta["cardData"] as? [String: Any]
        let configData = meta["config"] as? [String: Any]

        var lic = (cardData?["license"] as? String) ?? ""
        if lic.isEmpty, let licArr = cardData?["license"] as? [String] {
            lic = licArr.joined(separator: ", ")
        }
        if lic.isEmpty { lic = L("未标注 (见模型页)", "Unspecified") }

        var baseModel = (cardData?["base_model"] as? String) ?? ""
        if baseModel.isEmpty, let bmArr = cardData?["base_model"] as? [String] {
            baseModel = bmArr.prefix(2).joined(separator: ", ")
        }

        var archStr = ""
        if let archs = configData?["architectures"] as? [String], let first = archs.first {
            archStr = first
        } else if let gArch = ggufDict?["architecture"] as? String {
            archStr = gArch
        } else if let mType = configData?["model_type"] as? String {
            archStr = mType
        }
        let ctxLen = (ggufDict?["context_length"] as? NSNumber)?.intValue ?? 0
        let langs: [String] = {
            if let arr = cardData?["language"] as? [String] { return Array(arr.prefix(8)) }
            if let s = cardData?["language"] as? String { return [s] }
            return []
        }()
        let usedStorageBytes = (meta["usedStorage"] as? NSNumber)?.doubleValue ?? 0.0
        let usedStorageGb = usedStorageBytes > 0 ? ((usedStorageBytes / 1_073_741_824.0) * 100).rounded() / 100 : 0.0

        var weightFiles: [HFModelWeightFileItem] = []
        var totalWeightFilesGb: Double = 0.0
        if let sibs = meta["siblings"] as? [[String: Any]] {
            for sib in sibs {
                let fname = ((sib["rfilename"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                let fl = fname.lowercased()
                guard fl.hasSuffix(".gguf") || fl.hasSuffix(".safetensors") || fl.hasSuffix(".bin") || fl.hasSuffix(".pth") else { continue }
                let lfs = sib["lfs"] as? [String: Any]
                let fBytes = (sib["size"] as? NSNumber)?.doubleValue ?? (lfs?["size"] as? NSNumber)?.doubleValue ?? 0.0
                let sizeGb = fBytes > 0 ? ((fBytes / 1_073_741_824.0) * 100).rounded() / 100 : 0.0
                totalWeightFilesGb += sizeGb
                let vramNeed = sizeGb > 0 ? ((sizeGb * 1.12 + 1.0) * 10).rounded() / 10 : 0.0
                let fStatus: String
                let fZh: String
                let fEn: String
                if vramNeed <= 0 {
                    fStatus = "unknown"; fZh = "体积待测"; fEn = "Size TBD"
                } else if vramNeed <= 14.5 {
                    fStatus = "great"; fZh = "T4(15G)可跑"; fEn = "T4 15G Ready"
                } else if vramNeed <= 22.5 {
                    fStatus = "good"; fZh = "L4(24G)可跑"; fEn = "L4 24G Ready"
                } else if vramNeed <= 39.0 {
                    fStatus = "a100"; fZh = "需A100(40G)"; fEn = "A100 40G"
                } else if vramNeed <= 78.0 {
                    fStatus = "heavy"; fZh = "需A100(80G)"; fEn = "A100 80G"
                } else {
                    fStatus = "too_large"; fZh = "超单卡显存"; fEn = ">80G VRAM"
                }
                weightFiles.append(HFModelWeightFileItem(
                    filename: fname,
                    size_gb: sizeGb,
                    fit_status: fStatus,
                    fit_zh: fZh,
                    fit_en: fEn
                ))
                if weightFiles.count >= 36 { break }
            }
        }

        let fallbackGb = (!isGGUF && totalWeightFilesGb > 0) ? totalWeightFilesGb : usedStorageGb
        let sizing = analyzeHFModelSizing(meta, repoId: rid, isGGUF: isGGUF, fallbackStorageGb: fallbackGb)

        var readmeExcerpt = ""
        for host in ["https://huggingface.co", "https://hf-mirror.com"] {
            guard let rURL = URL(string: "\(host)/\(rid)/raw/main/README.md") else { continue }
            var rReq = URLRequest(url: rURL)
            rReq.timeoutInterval = 6
            rReq.setValue("EasyColab-macOS/2.0", forHTTPHeaderField: "User-Agent")
            if !tok.isEmpty {
                rReq.setValue("Bearer \(tok)", forHTTPHeaderField: "Authorization")
            }
            if let (rData, rResp) = try? await URLSession.shared.data(for: rReq),
               (rResp as? HTTPURLResponse)?.statusCode == 200,
               let rawMd = String(data: rData.prefix(32768), encoding: .utf8) {
                readmeExcerpt = cleanHFReadmeExcerpt(rawMd)
                if !readmeExcerpt.isEmpty { break }
            }
        }

        let shortName = rid.split(separator: "/").last.map(String.init) ?? rid
        let ownerName = (meta["author"] as? String) ?? (rid.split(separator: "/").first.map(String.init) ?? "")
        let lastMod = String(((meta["lastModified"] as? String) ?? "").prefix(10))

        activeHFModelDetail = HFModelDetailInfo(
            ok: true,
            repo_id: rid,
            name: shortName,
            author: ownerName,
            pipeline_tag: ptag.isEmpty ? (isGGUF ? "gguf" : "text-generation") : ptag,
            modality: modality,
            is_gguf: isGGUF,
            is_gated: isGated,
            license: lic,
            base_model: baseModel,
            architecture: archStr,
            context_length: ctxLen,
            languages: langs,
            likes: (meta["likes"] as? Int) ?? 0,
            downloads: (meta["downloads"] as? Int) ?? 0,
            last_modified: lastMod,
            used_storage_gb: usedStorageGb,
            param_count: sizing.paramCount,
            param_label: sizing.paramLabel,
            est_vram_gb: sizing.estVramGb,
            deploy_status: sizing.deployStatus,
            deploy_label_zh: sizing.deployZh,
            deploy_label_en: sizing.deployEn,
            weight_files: weightFiles,
            readme_excerpt: readmeExcerpt
        )
    }

    func registerCustomHFModel(autoDownload: Bool = false) async {
        let repo = normalizeHFRepoId(customRepoInput)
        guard repo.contains("/"), repo.split(separator: "/").count == 2 else {
            customModelErrorMessage = L(
                "请输入合法的 HuggingFace 仓库地址（格式：作者/仓库名，如 unsloth/Qwen3.8-27B-GGUF）",
                "Please enter a valid HuggingFace Repo ID (format: owner/repo, e.g. unsloth/Qwen3.8-27B-GGUF)"
            )
            return
        }
        customModelErrorMessage = ""
        isRegisteringCustomModel = true
        defer { isRegisteringCustomModel = false }

        let inc = customIncludePatternInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let customName = customDisplayNameInput.trimmingCharacters(in: .whitespacesAndNewlines)
        let modPref = customModalityInput

        if repo.lowercased() == "minimaxai/minimax-h3-image" && inc.isEmpty,
           let builtinSpec = OPEN_MODEL_CATALOG.first(where: { $0.id == "minimax_h3_image" }) {
            customModels.removeAll { $0.repoId.lowercased() == "minimaxai/minimax-h3-image" }
            savePersistedCustomModels()
            switchSelectedModel(builtinSpec)
            showAddCustomModelSheet = false
            customRepoInput = ""
            customIncludePatternInput = ""
            customDisplayNameInput = ""
            statusMessage = L(
                "已切换至内置专属引擎「MiniMax-H3-Image (33B · 4K)」（支持 6 步原生 4K ModularPipeline 与编码器零拷贝共享）！",
                "Switched to built-in dedicated engine 'MiniMax-H3-Image (33B · 4K)' (6-step native 4K ModularPipeline + shared encoder)!"
            )
            if autoDownload && isConnected {
                await startRemoteDownloadWeights()
            }
            return
        }

        let base = normalizedBaseURL
        if isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/custom_model/register") {
            do {
                var req = try adminRequest(url)
                req.httpMethod = "POST"
                req.timeoutInterval = 35
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.httpBody = try JSONSerialization.data(withJSONObject: [
                    "repo_id": repo,
                    "include_pattern": inc,
                    "custom_name": customName,
                    "display_name": customName,
                    "modality": modPref,
                    "modality_hint": modPref,
                    "hf_token": hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
                ])
                let (data, _) = try await adminSession.data(for: req)
                let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
                if decoded.action_ok == false {
                    let srvMsg = decoded.action_message ?? ""
                    if !srvMsg.contains("all_rfilenames") {
                        customModelErrorMessage = srvMsg.isEmpty ? L("无法解析该 HuggingFace 仓库", "Failed to resolve HuggingFace repository") : srvMsg
                        return
                    }
                } else {
                    applyStatusResponse(decoded)
                    if let reg = decoded.registered_model {
                        let spec = OpenModelSpec.fromCustomDTO(reg)
                        if let idx = customModels.firstIndex(where: { $0.id == spec.id }) {
                            customModels[idx] = spec
                        } else {
                            customModels.append(spec)
                        }
                        savePersistedCustomModels()
                        switchSelectedModel(spec)
                    }
                    if let am = decoded.action_message, !am.isEmpty {
                        statusMessage = am
                    }
                    showAddCustomModelSheet = false
                    customRepoInput = ""
                    customIncludePatternInput = ""
                    customDisplayNameInput = ""
                    if autoDownload {
                        await startRemoteDownloadWeights()
                    }
                    return
                }
            } catch {
                // 若远端旧版或网络波动，回退为本地先注册
            }
        }

        let slugBase = (repo + (inc.isEmpty ? "" : "_\(inc)"))
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9]+", with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        let modelId = "custom_" + String(slugBase.prefix(40))
        let resolvedMod: String
        if modPref == "auto" {
            let lower = (repo + " " + inc).lowercased()
            if lower.contains("flux") || lower.contains("image") || lower.contains("sdxl") || lower.contains("z-image") {
                resolvedMod = "image"
            } else if lower.contains("video") || lower.contains("wan") || lower.contains("ltx") || lower.contains("minimax") || lower.contains("hunyuan") {
                resolvedMod = "video_audio"
            } else {
                resolvedMod = "llm_vlm"
            }
        } else {
            resolvedMod = modPref
        }
        let repoShort = String(repo.split(separator: "/").last ?? Substring(repo))
        let finalName = customName.isEmpty ? (inc.isEmpty ? repoShort : "\(repoShort) (\(inc))") : customName
        let archGuess = (repo.lowercased().contains("gguf") || inc.lowercased().contains(".gguf")) ? "gguf" : (resolvedMod == "image" ? "diffusers_image" : "transformers")
        let dto = CustomHFModelDTO(
            id: modelId,
            name: finalName,
            repo_id: repo,
            include_pattern: inc,
            modality: resolvedMod,
            pipeline_arch: archGuess,
            category: "自定义 HuggingFace 仓库 (\(repo))",
            approx_size_gb: 0.0,
            is_custom: true
        )
        let spec = OpenModelSpec.fromCustomDTO(dto)
        if let idx = customModels.firstIndex(where: { $0.id == spec.id }) {
            customModels[idx] = spec
        } else {
            customModels.append(spec)
        }
        savePersistedCustomModels()
        switchSelectedModel(spec)
        showAddCustomModelSheet = false
        customRepoInput = ""
        customIncludePatternInput = ""
        customDisplayNameInput = ""
        statusMessage = L(
            "已添加自定义模型「\(finalName)」(\(repo))，连接 Colab 后即可并发拉取至 Google Drive。",
            "Added custom model '\(finalName)' (\(repo)). Connect to Colab to download weights to Google Drive."
        )
    }

    func removeCustomHFModel(_ spec: OpenModelSpec) async {
        guard spec.isCustom else { return }
        customModels.removeAll { $0.id == spec.id }
        savePersistedCustomModels()
        if selectedModelId == spec.id {
            switchSelectedModel(OPEN_MODEL_CATALOG[0])
        }
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/custom_model/remove") else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 20
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "model_id": spec.id,
                "delete_weights": false
            ])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
        } catch {}
    }

    func startRemoteComfyUI(forceRestart: Bool = false, autoOpenBrowser: Bool = true) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/start") else {
            statusMessage = L(
                "请先唤醒或连接云端 G4 实例后再启动 ComfyUI。",
                "Please wake or connect to a cloud G4 instance before starting ComfyUI."
            )
            return
        }
        comfyStarting = true
        comfyProgress = 0.04
        let tip = forceRestart
            ? L("正在重启云端 ComfyUI 并重新加载自定义节点...", "Restarting cloud ComfyUI & reloading custom nodes...")
            : L("正在部署云端 ComfyUI 并挂载 Google Drive 持久化软链...", "Deploying cloud ComfyUI & mounting Google Drive persistence symlinks...")
        comfyStatusMessage = tip
        statusMessage = tip

        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 30
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "unload_existing": comfyUnloadOthersOnStart,
                "force_restart": forceRestart
            ])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            // 持续轮询直到启动完成并拿到 trycloudflare 公网画布地址
            for _ in 0..<180 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = comfyStatusMessage
                if comfyRunning && !comfyPublicURL.isEmpty && !comfyStarting {
                    statusMessage = L(
                        "ComfyUI 全屏节点工作台已就绪！正在为您在浏览器中打开画布...",
                        "ComfyUI fullscreen node canvas is ready! Opening in your browser..."
                    )
                    if autoOpenBrowser {
                        openComfyUIInBrowser()
                    }
                    return
                }
                if !comfyStarting && !comfyRunning {
                    break
                }
            }
        } catch {
            comfyStarting = false
            statusMessage = L(
                "启动 ComfyUI 请求失败: \(error.localizedDescription)",
                "Failed to start ComfyUI: \(error.localizedDescription)"
            )
        }
    }

    func stopRemoteComfyUI() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/stop") else { return }
        statusMessage = L(
            "正在同步自定义节点与工作流到 Google Drive 并停止 ComfyUI...",
            "Syncing custom nodes & workflows to Google Drive and stopping ComfyUI..."
        )
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 35
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            statusMessage = comfyStatusMessage
        } catch {
            statusMessage = L(
                "停止 ComfyUI 失败: \(error.localizedDescription)",
                "Failed to stop ComfyUI: \(error.localizedDescription)"
            )
        }
    }

    func freeRemoteComfyUIVRAM() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/free_vram") else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 20
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            if let am = decoded.action_message, !am.isEmpty {
                statusMessage = am
            }
        } catch {}
    }

    func syncRemoteComfyUIToDrive() async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/sync_drive") else { return }
        comfySyncingDrive = true
        statusMessage = L(
            "正在将 ComfyUI 自定义节点、模型与工作流同步至 Google Drive...",
            "Syncing ComfyUI custom nodes, models & workflows to Google Drive..."
        )
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 90
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)
            if let am = decoded.action_message, !am.isEmpty {
                statusMessage = am
            }
        } catch {
            comfySyncingDrive = false
            statusMessage = L(
                "同步至 Google Drive 失败: \(error.localizedDescription)",
                "Failed to sync to Google Drive: \(error.localizedDescription)"
            )
        }
    }

    func startComfyWeightDownload() async {
        let src = comfyWeightSourceInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !src.isEmpty else {
            statusMessage = L(
                "请先粘贴 HuggingFace 模型直链、下载 URL 或 仓库 ID",
                "Please paste a Hugging Face direct link, download URL, or repo ID first"
            )
            return
        }
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/download_weight") else {
            statusMessage = L("请先连接 Colab 实例后再下载权重", "Please connect to a Colab instance before downloading weights")
            return
        }
        comfyDownloadingWeight = true
        comfyProgress = 0.05
        statusMessage = L(
            "正在通过云端 8 线程通道下载权重至 ComfyUI/models/\(comfyWeightSubfolder)...",
            "Downloading weights via 8-thread cloud channel to ComfyUI/models/\(comfyWeightSubfolder)..."
        )
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 25
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "url_or_repo": src,
                "filename": comfyWeightFilenameInput.trimmingCharacters(in: .whitespacesAndNewlines),
                "subfolder": comfyWeightSubfolder,
                "hf_token": hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
            ])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            while comfyDownloadingWeight {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = comfyStatusMessage
            }
        } catch {
            comfyDownloadingWeight = false
            statusMessage = L(
                "提交权重下载失败: \(error.localizedDescription)",
                "Failed to start weight download: \(error.localizedDescription)"
            )
        }
    }

    func installComfyPresetPack(_ pack: ComfyPresetPackItem, openWhenReady: Bool = true) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/install_pack") else {
            statusMessage = L(
                "请先在左侧唤醒或连接云端 G4 实例后再一键配齐工作流包。",
                "Please wake or connect to a cloud G4 instance on the left before installing a workflow pack."
            )
            return
        }
        comfyDownloadingWeight = true
        comfyInstallingPackId = pack.id
        comfyProgress = 0.04
        let tip = L(
            "正在为您一键配齐「\(pack.localizedTitle)」模型权重与全中文工作流...",
            "Installing models & workflow templates for \"\(pack.localizedTitle)\"..."
        )
        comfyStatusMessage = tip
        statusMessage = tip

        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.timeoutInterval = 30
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: [
                "pack_id": pack.id,
                "auto_start": true,
                "hf_token": hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
            ])
            let (data, resp) = try await adminSession.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            if code == 404 {
                comfyDownloadingWeight = false
                comfyInstallingPackId = ""
                let upgradeTip = L(
                    "当前云端实例仍在运行旧版服务，请点左侧「唤醒实例并部署」热更新服务端（不会释放现有 G4 实例与画布），随后即可一键配齐工作流！",
                    "The cloud instance is running an older server version. Click 'Wake & Deploy' on the left to hot-update app.py without releasing your G4 VM, then install the pack!"
                )
                comfyStatusMessage = upgradeTip
                statusMessage = upgradeTip
                return
            }
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            applyStatusResponse(decoded)

            for _ in 0..<600 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await checkColabStatus(silent: true)
                statusMessage = comfyStatusMessage
                if !comfyDownloadingWeight && comfyInstallingPackId.isEmpty && !comfyStarting {
                    break
                }
            }
            if openWhenReady && comfyRunning && !comfyPublicURL.isEmpty {
                openComfyUIInBrowser()
            }
        } catch {
            comfyDownloadingWeight = false
            comfyInstallingPackId = ""
            statusMessage = L(
                "一键配齐预置工作流包失败: \(error.localizedDescription)",
                "Failed to install preset workflow pack: \(error.localizedDescription)"
            )
        }
    }

    func downloadComfyOutputToMac(_ item: ComfyOutputMediaItem) async {
        let base = normalizedBaseURL
        let encodedRel = item.rel_path
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        guard !base.isEmpty, let url = URL(string: "\(base)/minimax_api/comfyui/output_media/\(encodedRel)") else { return }
        comfyPullingOutputRelPath = item.rel_path
        statusMessage = L(
            "正在从 ComfyUI 云端输出目录拉取 \(item.filename) 到本地 Mac...",
            "Pulling \(item.filename) from ComfyUI cloud output folder to Mac..."
        )
        defer { comfyPullingOutputRelPath = nil }
        do {
            let (tmpURL, resp) = try await adminSession.download(for: adminRequest(url))
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                statusMessage = L(
                    "拉取成片失败: 云端文件不存在或尚未写入完成",
                    "Failed to pull output: remote file does not exist or is still being written"
                )
                return
            }
            let safeName = "comfy_" + (item.filename as NSString).lastPathComponent
            let dstURL = localSaveDir.appendingPathComponent(safeName)
            if FileManager.default.fileExists(atPath: dstURL.path) {
                try? FileManager.default.removeItem(at: dstURL)
            }
            try FileManager.default.moveItem(at: tmpURL, to: dstURL)
            refreshLocalMedia()
            let ext = dstURL.pathExtension.lowercased()
            currentMediaURL = dstURL
            if ["png", "jpg", "jpeg", "webp", "gif"].contains(ext), let img = NSImage(contentsOf: dstURL) {
                player?.pause()
                player = nil
                currentPreviewImage = img
            } else {
                currentPreviewImage = nil
                player = AVPlayer(url: dstURL)
                player?.play()
            }
            statusMessage = L(
                "已将 ComfyUI 成片保存至本地目录: \(dstURL.lastPathComponent)",
                "Saved ComfyUI output to local folder: \(dstURL.lastPathComponent)"
            )
        } catch {
            statusMessage = L(
                "下载 ComfyUI 成片失败: \(error.localizedDescription)",
                "Failed to download ComfyUI output: \(error.localizedDescription)"
            )
        }
    }

    func regenerateApiKey() {
        apiAccessKey = "sk-colab-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(20).lowercased()
        Task { await syncApiKeyToColab() }
    }

    func syncApiKeyToColab(initializeOnly: Bool = false) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_api_key") else { return }
        let payloadKey = apiAccessKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !payloadKey.isEmpty else { return }
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["api_key": payloadKey, "initialize_only": initializeOnly])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            remoteApiKeySynced = true
            applyStatusResponse(decoded)
        } catch {}
    }

    func syncContextLengthToColab(announce: Bool = false) async {
        let base = normalizedBaseURL
        guard isConnected, !base.isEmpty, let url = URL(string: "\(base)/minimax_api/config_context") else { return }
        let targetTokens = contextLengthTokens
        do {
            var req = try adminRequest(url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: ["context_length": targetTokens])
            let (data, _) = try await adminSession.data(for: req)
            let decoded = try JSONDecoder().decode(ColabStatusResponse.self, from: data)
            remoteContextSynced = true
            applyStatusResponse(decoded)
            if announce, let msg = decoded.task_message, !msg.isEmpty {
                statusMessage = msg
            }
        } catch {}
    }

}
