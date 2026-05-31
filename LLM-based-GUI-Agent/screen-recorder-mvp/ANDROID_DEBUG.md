# Android 端启动与调试指南

按下面步骤在电脑上运行 Android 工程，并在真机上完成调试与「手机 → PC」上传联调。

---

## 0. 五分钟跑通（先看这个）

首次在新机器运行前（只需做一次）：

- Windows:
  ```powershell
  cd screen-recorder-mvp/pc
  python -m venv venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```
- macOS/Linux:
  ```bash
  cd screen-recorder-mvp/pc
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

1. **PC 端先启动（建议在 venv 中）**：`cd screen-recorder-mvp/pc` → 激活 venv → `python main_web.py`，在 Recording 页点击「启动上传服务」。
2. **Android 安装运行**：Android Studio 打开 `screen-recorder-mvp/android`，连接真机后 Run。
3. **手机录屏**：点击「开始录制」，系统弹窗点允许，录制几秒后停止。
4. **手机上传**：Data 页填 PC IP + `8765`，点击「上传到电脑」。
5. **PC 验证**：Data 页确认收到视频，Settings 页启动分析，Data 页做记忆检索。

常用命令（可选）：

```bash
adb devices
adb logcat -c
adb logcat | rg "ScreenRecordSvc|MainActivity"
adb shell pm grant com.screenrecorder.mvp android.permission.POST_NOTIFICATIONS
```

> 如果终端没有 `rg`，可改用：`adb logcat | findstr "ScreenRecordSvc MainActivity"`（Windows CMD/PowerShell）。

---

## 一、电脑上需要做的操作

### 1. 安装 Android Studio（若尚未安装）

- 下载：<https://developer.android.com/studio>
- 安装时勾选 **Android SDK**、**Android SDK Platform**、**Android Virtual Device**（模拟器可选）。
- 安装完成后打开 Android Studio，完成首次向导（下载 SDK 等）。

### 2. 打开项目并同步

1. 打开 Android Studio，选择 **File → Open**。
2. 选中本仓库里的 **`screen-recorder-mvp/android`** 文件夹（不要选成 `screen-recorder-mvp` 或 `XOOCITY`）。
3. 等待右下角 **“Sync Project with Gradle Files”** 完成（首次会下载 Gradle 和依赖，可能几分钟）。
4. 若提示 “Gradle sync failed”：
   - 检查网络（需能访问 Google/ Maven）。
   - 或 **File → Invalidate Caches → Invalidate and Restart** 后再同步一次。

### 3. 连接手机并运行 App

任选一种方式即可：

- **方式 A：USB 数据线**（见下文「手机上 → 开发者选项与 USB 调试」）
- **方式 B：无线调试（不用数据线）**（见下方 **「无线调试（不用数据线）」**）

连接成功后，在 Android Studio 顶部：
- **运行设备**：下拉选你的真机（例如 `XXX (Android 12)`）。
- 点击绿色 **Run** 或按 `Shift+F10`，等待安装完成，手机会自动打开 App。

---

### 无线调试（不用数据线）

**要求**：手机 **Android 11 及以上**，且手机与电脑在同一 WiFi 下。

**手机端：**

1. 打开 **设置 → 系统 → 开发者选项**（没有则先到「关于手机」里连续点 7 次版本号）。
2. 找到并打开 **无线调试**（Wireless debugging）。
3. 点进「无线调试」：
   - 若提示「使用配对码配对设备」，点 **使用配对码配对设备**，记下当前页显示的 **IP:端口**（如 `192.168.1.105:37123`）和 **6 位配对码**（如 `123456`）。
   - 若没有配对码入口，则只记下 **IP 和端口**（如 `192.168.1.105:5555`），用于下面「电脑端」的 `adb connect`。

**电脑端（首次配对时）：**

1. 打开命令行，进入 Android SDK 的 `platform-tools` 目录：
   - **PowerShell**（推荐）：
     ```powershell
     cd $env:LOCALAPPDATA\Android\Sdk\platform-tools
     ```
   - **CMD**：
     ```cmd
     cd %LOCALAPPDATA%\Android\Sdk\platform-tools
     ```
   若 SDK 装在 D 盘，改为：`cd D:\Android\Sdk\platform-tools`（按实际路径修改）。
2. **若手机上有 6 位配对码**（Android 11+ 常见）：
   ```bash
   adb pair <手机IP>:<配对端口>
   ```
   按提示输入配对码（配对端口和 IP 在手机「无线调试」页有写，有时是「使用配对码配对设备」里另一组端口）。
3. 配对成功后，再连接调试端口（手机「无线调试」页会显示「已配对设备」或一组 **IP:端口**）：
   ```bash
   adb connect <手机IP>:<连接端口>
   ```
   例如：`adb connect 192.168.1.105:5555`。
4. 在 Android Studio 里刷新设备列表（或 **View → Tool Windows → Device Manager**），应能看到已连接的设备，然后像 USB 一样点 **Run** 安装并运行 App。

**之后**：只要手机和电脑在同一 WiFi、且手机未关闭「无线调试」，下次只需在电脑执行一次 `adb connect <手机IP>:<端口>` 即可，无需再配对。

---

## 二、手机上需要做的操作

### 1. 开启开发者选项与 USB 调试（用于安装/调试）

- **开发者选项**（不同品牌位置略有差异）：
  - 进入 **设置 → 关于手机**，连续点击 **版本号** 约 7 次，直到提示“您已处于开发者模式”。
- **USB 调试**（用数据线时需要）：
  - **设置 → 系统 → 开发者选项**，打开 **USB 调试**。
- **无线调试**（不用数据线时）：同上进入开发者选项，打开 **无线调试**，按上文「无线调试」步骤在电脑用 `adb pair` / `adb connect` 连接。
- 用数据线连接时，若弹出 **“是否允许 USB 调试？”**，选 **允许** 即可。

### 2. 运行 App 时的权限（勾选清单）

- [ ] **屏幕录制权限**（必选）  
  点击 App 内「开始录制」后，系统会弹出 **“允许录制屏幕”/“捕获屏幕”** 的授权框，必须点 **立即开始** 或 **允许**，否则无法录制。
- [ ] **通知权限**（Android 13+ 建议开启）  
  若通知栏没有“正在录制”的提示，可到 **设置 → 应用 → 屏幕录制 MVP → 通知** 中打开通知权限，便于确认录制状态。
- [ ] **网络权限**（Manifest 中 `INTERNET`）  
  若上传报 `EPERM` 或无法连接，请确认已正确打包最新 Manifest。

### 3. 录制视频在哪里看

- **手机本机**：录制文件保存在应用私有目录 `.../Android/data/com.screenrecorder.mvp/files/Movies/ScreenRecords/`（App 内「本机屏幕录制」下方会显示完整路径）。Android 11+ 下用系统文件管理器可能无法直接进入该目录，建议用「上传到电脑」传到 PC 后查看。
- **在 PC 上看**：PC 端运行 `main_web.py` 并点击「启动上传接收服务」后，在手机 App 填写 PC IP 和端口 8765，点「上传到电脑」；上传成功后到 PC 端点击「打开录制文件夹」，即可在 `recordings/` 目录中看到视频。若 Windows 自带「媒体播放器」无法播放，请用 **VLC** 或 **PotPlayer** 打开（录制的为 MP4/H.264，VLC 兼容性更好）。

### 4. 与 PC 同网（用于「上传到电脑」）

- 手机和电脑需在 **同一 WiFi** 下。
- 在 PC 端先 **启动上传接收服务**，记下显示的 **本机 IP**（如 `192.168.1.100`）。
- 在手机 App 的 **PC IP 地址** 里填写该 IP，**端口** 填 **8765**，再点 **上传到电脑**。

---

## 三、推荐调试流程（手机 + PC 联调）

1. **PC 端**：运行 `python main_web.py`（或 `run.bat`），点击 **「启动上传接收服务」**，记下显示的 IP（如 `http://192.168.1.100:8765/upload` 里的 `192.168.1.100`）。
2. **手机**：与电脑同一 WiFi，用 **USB 连接** 或 **无线调试** 连接电脑，在 Android Studio 中 **Run** 安装并打开「屏幕录制 MVP」。
3. **手机 App**：点 **「开始录制」** → 在系统弹窗中允许 **录制屏幕** → 随意操作手机几秒 → 点 **「停止录制」**。
4. **手机 App**：在 **PC IP** 填 `192.168.1.100`，**端口** 填 `8765`，点 **「上传到电脑」**。
5. **PC 端**：点 **「打开录制文件夹」**，应能看到刚上传的 MP4 文件。

---

## 四、常见问题

| 现象 | 处理 |
|------|------|
| Android Studio 列表里没有设备 | USB：确认 USB 调试已开、数据线可传数据，换口/换线，或重新点「允许 USB 调试」。无线：确认同一 WiFi、无线调试已开，在电脑执行 `adb connect <IP>:<端口>` 后再刷新设备列表。 |
| 点击「开始录制」没反应 | 点「开始录制」后会出现系统授权弹窗，请到**多任务/最近任务**或**通知栏**里找“允许录制屏幕/立即开始”并点允许；若仍无反应，看 Logcat 是否有崩溃。 |
| 录制/上传后文件 0 字节 | 说明未真正录到内容。请**重新录制**：点「开始录制」后，务必在**系统弹出的窗口**里点「立即开始」或「允许」（不要点取消或返回）；录几秒后再点「停止录制」。若系统弹窗被其他 App 挡住，可先切到多任务里找到“屏幕录制”或“捕获屏幕”的窗口再点允许。 |
| 上传失败 / 超时 | 手机和 PC 是否同一 WiFi；PC 防火墙是否放行 8765 端口；PC 端是否已点「启动上传接收服务」。若提示「CLEARTEXT communication not permitted」，已通过 network_security_config 允许 HTTP；若提示「EPERM (Operation not permitted)」，已添加 INTERNET 权限，请重新编译安装 App。 |
| Gradle 同步失败 | 检查网络；在 Android Studio 中 **File → Invalidate Caches → Invalidate and Restart** 后重试。 |

### 构建失败速查（补充）

- **Gradle sync failed**：先检查网络与代理；再 Invalidate Caches；最后重启 IDE 再同步。
- **SDK 版本不匹配**：本项目 `compileSdk/targetSdk=34`，请在 SDK Manager 安装 API 34。
- **JDK 版本问题**：优先使用 Android Studio Embedded JDK 17。
- **设备看不到**：先执行 `adb devices`，状态应为 `device`（不是 `unauthorized` / `offline`）。

### 上传失败: failed to connect to /192.168.x.x (port 8765) from /172.16.x.x ... after 30000ms

该报错表示**手机连不上 PC 的 8765 端口**，常见原因有二：

**1. 手机和 PC 不在同一网段（最常见）**

- 报错里会同时出现两个 IP：`from /172.16.40.169`（手机）和 `to ... 192.168.2.236`（PC）。若手机是 **172.16.x.x** 而 PC 是 **192.168.x.x**，说明两者不在同一局域网（例如手机连的是热点或另一台路由，PC 连的是家里 WiFi）。
- **处理**：手机和 PC 都连到**同一个 WiFi**，确保手机获取的 IP 与 PC 同网段（如 PC 是 `192.168.2.236`，手机应为 `192.168.2.xxx`）。在手机 WiFi 详情里可查看本机 IP；PC 端程序里会显示「本机 IP」，两边网段一致即可。

**2. Windows 防火墙拦截了 8765 端口**

- 即使同网，若防火墙未放行，也会一直超时。
- **处理**：在 PC 上放行 8765 端口（或放行 Python）：
  - **控制面板 → Windows Defender 防火墙 → 高级设置 → 入站规则 → 新建规则**；
  - 选「端口」→ TCP，特定本地端口填 **8765**；
  - 选「允许连接」，勾选「专用」「域」（按需勾选「公用」），命名如「ScreenRecorder 8765」。
  - 或临时关闭防火墙测试（仅用于排查，用完后建议重新开启）。

**自检**：PC 端已显示 `Running on http://192.168.2.236:8765` 时，在同一 WiFi 下用手机浏览器访问 `http://192.168.2.236:8765/health`，若打不开，多半是网络不同或防火墙未放行。

### 真机差异（国产 ROM）补充

- 后台省电策略可能导致录屏服务被杀，建议给应用关闭电池优化。
- 系统录屏授权弹窗可能被悬浮窗遮挡（聊天气泡、会议工具），可到最近任务手动切回授权窗口。
- 若经常出现录屏后 0 字节，先排除“未点系统授权”再看编码兼容问题。

---

## 五、项目路径小结

- **在 Android Studio 中要打开的目录**：`screen-recorder-mvp/android`（即包含 `build.gradle.kts`、`settings.gradle.kts`、`app` 的那一层）。
- **PC 端程序所在目录**：`screen-recorder-mvp/pc`，在此目录用 `venv\Scripts\python main_web.py`（推荐）或 `run.bat` 启动。
