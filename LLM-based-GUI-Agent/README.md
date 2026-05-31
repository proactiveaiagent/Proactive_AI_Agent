# LLM-based Multi-modal GUI Agent for All day Screen Recording and Recognition — 使用说明

**Android 应用显示名称：XOOGUIAGT**（`applicationId` 仍为 `com.screenrecorder.mvp`。）

本仓库包含 **PC 端（Windows）** 和 **Android 端** 的最小可运行演示：后台屏幕录制，并将手机录制视频传输到电脑。  
PC 端现提供两套 UI 入口：
- **Web 桌面壳（推荐）**：`screen-recorder-mvp/pc/main_web.py`（FastAPI + Web UI）
- **Tkinter 旧入口（兼容）**：`screen-recorder-mvp/pc/main.py`

PC 端支持两种交付方式：
- **开发/源码运行**（本机 Python 环境）
- **离线交付包运行**（双击 exe，无需本机 Python）

---

## 一、PC 端（Windows / macOS）

### 环境要求

- Windows 10/11 或 macOS
- Python 3.8+

### 安装与运行

#### Windows（PowerShell / CMD）

1. 进入目录：
   ```bash
   cd screen-recorder-mvp/pc
   ```
2. 创建虚拟环境（可选）并安装依赖：
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 启动程序（推荐 Web 壳，**务必在激活 venv 后**用同一环境下的 python）：
   ```bash
   venv\Scripts\activate
   python main_web.py
   ```
   如需兼容旧版 Tk UI：
   ```bash
   venv\Scripts\python.exe main.py
   ```
   或直接使用 venv 的 python（Web 壳）：
   ```bash
   venv\Scripts\python.exe main_web.py
   ```

#### macOS（zsh / bash）

> macOS/Linux 下路径分隔符是 `/`，不能使用 Windows 的 `venv\Scripts\activate`。

1. 进入目录：
   ```bash
   cd screen-recorder-mvp/pc
   ```
2. 创建虚拟环境并安装依赖：
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. 启动程序（推荐 Web 壳）：
   ```bash
   source venv/bin/activate
   python main_web.py
   ```
   如需兼容旧版 Tk UI：
   ```bash
   ./venv/bin/python main.py
   ```
   或直接使用 venv 的 python（Web 壳）：
   ```bash
   ./venv/bin/python main_web.py
   ```

### 双端新 UI（设计稿落地）

- 设计规范与 Tokens：`screen-recorder-mvp/design/tokens.json`
- 组件与信息架构说明：`screen-recorder-mvp/design/DESIGN_SYSTEM.md`
- Android 已重构为四导航：`Recording / Earnings / Data / Settings`
- PC 已迁移 Web 壳，桥接 API 在 `pc/app_server.py`，状态控制在 `pc/desktop_controller.py`

### PC Web 壳前端构建

```bash
cd screen-recorder-mvp/pc/webui
npm install
npm run build
```

构建产物输出到 `pc/webui/dist/`，由 `main_web.py` 自动挂载；开发说明见 `pc/webui/README.md`。

### 打包为桌面程序（可选）

可将 PC 端打包为独立 exe，无需安装 Python，便于分发给他人使用。

1. 在 `screen-recorder-mvp/pc` 目录下执行：
   ```bash
   .\build.bat
   ```
   （PowerShell 下必须写 `.\build.bat`；CMD 下可直接写 `build.bat`。）
2. 打包完成后，程序位于 `dist\ScreenRecorderPC\` 文件夹内。双击 `ScreenRecorderPC.exe` 即可运行。
3. **打包后图标没变**：Windows 会缓存 exe 图标。若重新打包后任务栏/窗口仍显示旧图标，可：先关闭程序，删除 `dist\ScreenRecorderPC` 再重新执行 `.\build.bat`；或清除图标缓存（如删除 `%LOCALAPPDATA%\IconCache.db` 后重启资源管理器），或重启电脑后再打开新 exe。
4. **数据目录**：打包后，录制视频和分析结果会保存在 **exe 所在目录** 下的 `recordings`、`analysis_output` 文件夹。可将整个 `ScreenRecorderPC` 文件夹复制到任意位置使用。
5. **本地模型部署目录**：若在应用内选择 `local` 后端，首次点击「一键部署本地模型」会自动下载模型到 **exe 所在目录** 下的 `models/`。建议首次部署时保持联网；后续可离线复用本地模型。

### 交付包说明（推荐给最终用户）

当前提供两类离线交付目录：

- `screen-recorder-mvp/pc/dist_4b_offline/ScreenRecorderPC`
  - 离线压缩包：`ScreenRecorderPC_offline_4B.zip`
  - 预置模型：`mPLUG/GUI-Owl-1.5-4B-Instruct`
  - 推荐：8GB 显存设备（速度一般，较稳定）
- `screen-recorder-mvp/pc/dist_8b_offline/ScreenRecorderPC`
  - 离线压缩包：`ScreenRecorderPC_offline_8B.zip`
  - 预置模型：`mPLUG/GUI-Owl-1.5-8B-Instruct`
  - 推荐：24GB 显存设备（效果更好）

**最终用户使用步骤（离线包）**：
1. 解压离线压缩包；
2. 双击 `ScreenRecorderPC.exe`；
3. 在「GUI Agent 视频分析」中选择 `local`，直接分析视频即可（无需 Python、无需联网下载模型）。

**离线包分发策略（重要）**：

- `dist_4b_offline/`、`dist_8b_offline/` 等大目录仅用于离线交付，不提交 Git。
- 推荐通过网盘/NAS/制品库分发离线包，Git 仓库仅保留构建脚本与文档。

**依赖**：打包需已安装 Python 和项目依赖；构建时会自动安装 PyInstaller。若需自定义（如单文件 exe），可编辑 `screen_recorder_pc.spec` 后执行 `pyinstaller screen_recorder_pc.spec`。  
**图标**：PC 端 exe 与 Android 端应用使用同一套图标（屏幕+录制圆点主题），由 `pc/make_icon.py` 生成 `icon.ico` 与 `icon_512.png`；Android 会使用同图标的 `res/drawable/ic_launcher.png`。修改图标后重新运行 `python make_icon.py` 即可。

### 版本与更新

- 当前版本号在窗口标题中显示（如 `v1.0.0`），定义在 `pc/version.py` 的 `__version__`。
- 发布新版本时：修改 `version.py` 中的 `__version__`，重新执行 `.\build.bat` 生成新的 `dist\ScreenRecorderPC`，将整个文件夹提供给用户覆盖或替换即可完成更新。

**常见启动错误：**

| 报错 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'mss'` | 未安装依赖或未在 venv 中运行 | 在 `pc` 目录下执行 `venv\Scripts\pip install -r requirements.txt`，再用 `venv\Scripts\python main_web.py` 启动 |
| `zsh: command not found: venvScriptsactivate` | 在 macOS 的 zsh 中误用了 Windows 命令 `venv\Scripts\activate`（反斜杠会被 shell 吃掉） | 改用 `source venv/bin/activate`，并使用 `python3 -m venv venv` 创建虚拟环境 |
| `No module named 'distutils'`（安装 numpy 时） | Python 3.12+ 已移除 distutils，旧版 numpy 从源码编译失败 | 已用宽松版本依赖，直接重新执行 `pip install -r requirements.txt` |
| `UnicodeDecodeError` 读 requirements.txt | 文件含中文注释在 Windows 默认编码下解码失败 | 已去掉 requirements 中的中文注释，若仍报错请用英文路径/无中文环境重试 |

### 功能说明

- **本机屏幕录制**：点击「开始录制」即可录制当前屏幕，再次点击「停止录制」保存为 `recordings/` 下的 MP4 文件。
- **接收手机视频**：点击「启动上传接收服务」后，程序会显示本机 IP 和端口（如 `http://192.168.1.100:8765/upload`）。确保手机与电脑在同一 WiFi 下，在手机 App（**XOOGUIAGT**）中填写该 IP 和端口（默认 8765），即可将手机录制的视频上传到电脑，文件同样保存在 `recordings/` 目录。

**在电脑上播放录制/上传的 MP4：**  
若 Windows 自带的「媒体播放器」提示无法打开（格式不支持或文件损坏），请改用 **VLC**、**PotPlayer** 等播放器打开。录制的文件为标准 MP4/H.264，用 VLC 一般可正常播放。

---

## 二、Android 端

**详细启动与调试步骤（含电脑操作、手机设置、联调流程）：见 [ANDROID_DEBUG.md](ANDROID_DEBUG.md)。**

### 5 分钟跑通（推荐顺序）

1. **PC 端先启动上传服务**：在 `screen-recorder-mvp/pc` 运行 `python main_web.py`，进入 Recording 页点击「启动上传服务」。
2. **Android 安装运行**：Android Studio 打开 `screen-recorder-mvp/android`（工程名与顶部标题为上述完整项目名称），Run 到真机，安装 **XOOGUIAGT**。
3. **手机录制**：在 Recording 页点击「开始录制」，系统弹窗点允许，录几秒后停止。
4. **手机上传**：在 Data 页填 PC 的局域网 IP 与端口 `8765`，点击「上传到电脑」。
5. **PC 验证分析/记忆**：PC 端 Data 页确认文件到达，Settings 页启动分析，Data 页执行记忆检索。

### Android 快速联调命令（可选）

```bash
# 1) 设备是否在线
adb devices

# 2) 关键日志（建议先清一次）
adb logcat -c
adb logcat | rg "ScreenRecordSvc|MainActivity"

# 3) Android 13+ 通知权限（如需要）
adb shell pm grant com.screenrecorder.mvp android.permission.POST_NOTIFICATIONS
```

### 权限勾选清单（先过一遍）

- [ ] `INTERNET`（Manifest，上传必须）
- [ ] 录屏系统授权（每次开始录制时系统弹窗点允许）
- [ ] Android 13+ 通知权限（后台录制状态更稳定可见）
- [ ] 手机与 PC 同一 WiFi（上传链路前置条件）

### 环境要求

- Android Studio（推荐 Ladybug 或更新版本）
- 真机或模拟器：Android 8.0（API 26）及以上

### 构建与安装

1. 用 Android Studio 打开项目根目录下的 `screen-recorder-mvp/android` 文件夹。
2. 等待 Gradle 同步完成。
3. 连接 Android 设备或启动模拟器，点击 Run 运行并安装 App。

### 功能说明

- **新 UI 信息架构**：底部四导航 `Recording / Earnings / Data / Settings`。
- **屏幕录制**：点击「开始录制」，在系统弹窗中授权「允许录制屏幕」后开始录制；再次点击「停止录制」结束。视频保存在应用私有目录（可通过「上传到电脑」后到 PC 的 `recordings/` 中查看）。
- **上传到电脑**：在 `Data` 页中填写电脑局域网 IP（与 PC 端显示地址一致），端口默认 8765。先确保 PC 端已「启动上传接收服务」，再点击「上传到电脑」，即可将录制视频上传到电脑 `recordings/` 文件夹。

### 录制视频在哪里看

- **手机本机**：录制文件保存在应用私有目录 `/storage/emulated/0/Android/data/com.screenrecorder.mvp/files/Movies/ScreenRecords/`（App 内「本机屏幕录制」下方会显示完整路径）。Android 11+ 系统文件管理器一般无法直接进入该目录，建议通过「上传到电脑」传到 PC 后查看。
- **在 PC 上看**：PC 端点「打开录制文件夹」，或直接打开 `screen-recorder-mvp/pc/recordings/` 目录。

### 注意事项

- 首次录制需在系统弹窗中同意「捕获屏幕」权限。
- 若 Android 13+ 通知栏无录制提示，请在系统设置中为应用开启「通知」权限。
- 手机与电脑须在同一 WiFi 下才能上传成功；若电脑有防火墙，需放行 8765 端口。

### 构建失败速查（Android Studio）

- **Gradle sync failed**：先看网络，再执行 `File -> Invalidate Caches -> Invalidate and Restart` 后重试。
- **SDK 版本不匹配**：本项目 `compileSdk/targetSdk=34`，请在 SDK Manager 安装 Android 14 (API 34)。
- **JDK 版本问题**：请使用 JDK 17（Android Studio 默认可选 Embedded JDK 17）。
- **设备不显示**：先 `adb devices` 确认连接状态，再检查 USB 调试或无线调试配对。

### 真机差异提醒（国产 ROM 常见）

- 后台限制可能导致录屏服务被系统回收，建议给应用关闭电池优化。
- 录屏授权弹窗可能被悬浮窗或其他应用遮挡，可到最近任务中手动切回授权窗口。
- 若录制后文件为 0 字节，优先检查授权是否真的点了“允许/立即开始”。

### 常见问题与排查

以下问题均已在开发过程中实际遇到并解决，供复现与排查参考。

#### 1. 上传失败：CLEARTEXT communication not permitted

- **现象**：手机端点「上传到电脑」后提示 `CLEARTEXT communication to x.x.x.x not permitted by network security policy`。
- **原因**：Android 9 (API 28) 起默认禁止明文 HTTP 请求，而 PC 端 Flask 接收服务使用的是 HTTP。
- **解决**：已在 `res/xml/network_security_config.xml` 中允许明文流量，并在 `AndroidManifest.xml` 的 `<application>` 标签中添加 `android:networkSecurityConfig="@xml/network_security_config"`。若你从旧版代码升级，请确认这两处都已添加，然后重新编译安装。

#### 2. 上传失败：socket failed: EPERM (Operation not permitted)

- **现象**：手机端点「上传到电脑」后提示 `socket failed: EPERM (Operation not permitted)`。
- **原因**：`AndroidManifest.xml` 中缺少 `<uses-permission android:name="android.permission.INTERNET" />`。
- **解决**：已在 Manifest 中添加该权限。请确认 Manifest 顶部包含 `INTERNET` 权限声明，然后重新编译安装。

#### 3. 录制文件 0 字节 / 点击「开始录制」后无反应

- **现象**：点击「开始录制」后，状态仍显示「未录制」，或录制停止后文件为 0 字节。上传到 PC 后打不开视频或文件大小为 0。
- **可能原因**：
  1. **系统授权弹窗未点击「允许」**：点击「开始录制」后，Android 系统会弹出一个**单独的授权窗口**（"是否允许录制屏幕"/ "立即开始"）。如果没在这个窗口里点「立即开始」或「允许」，录制不会真正开始。
  2. **弹窗被遮挡**：部分国产 ROM（华为/荣耀/小米等）可能将弹窗压到后台，或被飞书、微信等有悬浮窗功能的 App 遮挡。需要到**多任务/最近任务**中找到该弹窗再点允许。
  3. **权限被系统自动拒绝**：荣耀 Magic5 等机型在后台可能自动拦截录屏权限。需要到 **设置 → 应用 → XOOGUIAGT → 权限** 中检查是否有「屏幕录制」或「投屏」相关权限开关，手动打开。
  4. **录制时间太短**：刚点「开始录制」就立刻「停止录制」，可能写入 0 字节。请至少录几秒再停止。
- **排查方法**：
  1. 在 Android Studio 底部打开 **Logcat**，过滤 `ScreenRecordSvc` 或 `MainActivity`。
  2. 点击「开始录制」后观察日志：
     - 若看到 `MediaProjection obtained successfully` → `MediaRecorder started`：录制已正常开始，需等几秒再停止。
     - 若看到 `getMediaProjection returned null`：系统授权被拒绝，需重新点「开始录制」并在系统弹窗中点「允许」。
     - 若看到 `MediaRecorder.prepare() failed`：设备兼容问题，请将完整日志反馈。
  3. 停止录制后日志会输出文件路径和大小（`File: xxx, size: xxx bytes`），据此判断是否录入成功。
- **已做的代码改进**：
  - Service 和 Activity 中加了详细日志（TAG 分别为 `ScreenRecordSvc` 和 `MainActivity`），所有关键步骤和异常均输出到 Logcat。
  - Service 内部出错时通过 LocalBroadcast 将错误信息传回 Activity，界面上会弹出 Toast 显示具体原因并恢复按钮状态。
  - 上传前自动跳过 0 字节文件，并提示用户重新录制。

#### 4. PC 端上传的 MP4 在 Windows 媒体播放器中打不开

- **现象**：双击打开视频时，Windows 自带「媒体播放器」提示"无法打开"、"格式不支持"或"文件已损坏"（错误码 0xC00036C4）。
- **原因**：Windows 媒体播放器对部分 H.264/MP4 编码兼容性不佳，不代表文件真的损坏。
- **解决**：使用 **VLC**（https://www.videolan.org/）或 **PotPlayer** 打开同一文件，一般可正常播放。

#### 5. PC 端 `python main_web.py` 启动报 ModuleNotFoundError

- **现象**：运行 `python main_web.py` 时提示 `ModuleNotFoundError: No module named 'mss'`（或 `cv2`、`flask` 等）。
- **原因**：依赖未安装到当前使用的 Python 环境，或直接用了系统 Python 而非 venv 中的 Python。
- **解决**：务必在 `pc` 目录下用同一个 venv 安装依赖并运行：
  ```bash
  cd screen-recorder-mvp/pc
  venv\Scripts\pip install -r requirements.txt
  venv\Scripts\python main_web.py
  ```

#### 6. 安装依赖时报 `No module named 'distutils'`

- **现象**：执行 `pip install -r requirements.txt` 时，numpy 安装失败，报 `ModuleNotFoundError: No module named 'distutils'`。
- **原因**：Python 3.12+ 已移除 `distutils`，旧版 numpy 需从源码编译时依赖 distutils。
- **解决**：`requirements.txt` 已改为宽松版本（`numpy>=1.24.0,<2`），pip 会自动选择有预编译 wheel 的版本。重新执行 `pip install -r requirements.txt` 即可。

#### 7. 安装依赖时报 `UnicodeDecodeError`

- **现象**：执行 `pip install -r requirements.txt` 时报 `UnicodeDecodeError: 'gbk' codec can't decode byte...`。
- **原因**：requirements.txt 中包含中文注释，Windows 默认 GBK 编码无法解析 UTF-8 中文。
- **解决**：已去掉 requirements.txt 中的中文注释。若仍报错，请确认文件编码为 UTF-8 或在纯英文路径下重试。

---

## 三、演示流程建议

联调与回归步骤见：`screen-recorder-mvp/INTEGRATION_TEST.md`

1. **仅 PC 演示**：运行 `python main_web.py` → 点击「开始录制」→ 操作桌面 → 点击「停止录制」→ 点击「打开录制文件夹」查看 MP4。
2. **手机 → PC 演示**：PC 端启动并「启动上传接收服务」→ 记下显示的 `http://<IP>:8765/upload` 中的 IP → 手机端填写该 IP 和 8765 → 手机端「开始录制」→ 操作手机 → 「停止录制」→ 手机端「上传到电脑」→ 在 PC 的「打开录制文件夹」中查看收到的视频。

---

## 四、项目结构

```
screen-recorder-mvp/
├── pc/
│   ├── main.py            # PC 主界面与入口
│   ├── main_web.py        # PC Web 桌面壳入口（推荐）
│   ├── app_server.py      # FastAPI 桥接 API + 静态资源挂载
│   ├── desktop_controller.py # 录屏/上传/分析/记忆状态控制
│   ├── recorder.py        # 屏幕录制（mss + OpenCV）
│   ├── receiver.py        # 接收手机上传（Flask）
│   ├── config.py          # 配置管理（API Key、采样参数等）
│   ├── gui_agent_api.py   # 百炼 API 封装（GUI-Owl 调用）
│   ├── analyzer.py        # 视频分析模块（帧提取、去重、API 调用、报告导出）
│   ├── memory.py          # 记忆模块（存储、清洗、压缩、用户画像、检索）
│   ├── docs/              # 调研与设计文档（如 M3_AGENT_MEMORY_RESEARCH.md）
│   ├── version.py         # 应用版本号（用于标题与后续更新）
│   ├── make_icon.py       # 生成 PC/Android 共用图标（icon.ico、icon_512.png）
│   ├── icon.ico / icon_512.png  # 应用图标（由 make_icon.py 生成）
│   ├── screen_recorder_pc.spec  # PyInstaller 打包配置
│   ├── build.bat          # 一键打包为桌面 exe 的脚本
│   ├── webui/             # Vite + React 前端工程
│   ├── requirements.txt
│   ├── recordings/        # 录制与接收的视频存放目录（自动创建）
│   ├── analysis_output/   # 视频分析结果输出目录（自动创建）
│   └── memory/            # 记忆数据库（SQLite，自动创建）
├── android/
│   └── app/
│       └── src/main/
│           ├── java/.../MainActivity.kt
│           ├── java/.../ScreenRecordService.kt
│           └── res/...
├── design/
│   ├── tokens.json        # 双端共享 Design Tokens
│   └── DESIGN_SYSTEM.md   # 双端组件与信息架构规范
├── INTEGRATION_TEST.md    # 双端联调与回归步骤
└── README.md
```

---

## 五、GUI Agent 视频分析

PC 端内置了视频分析功能，能自动识别录屏中的应用名称、页面、UI 元素、用户操作等信息。

支持两种分析后端（UI 下拉可切换）：

| 模式 | 模型 | 优点 | 适用场景 |
|------|------|------|---------|
| **模式一：本地 Transformers（默认）** | `mPLUG/GUI-Owl-1.5-8B-Instruct`（可改 4B/2B） | 无需云 API；可在 exe 首次运行自动部署模型 | 有 NVIDIA GPU，希望本地推理 |
| **模式二：百炼云端 API** | `qwen-vl-max` | 零本地显存占用，开箱即用 | 无 GPU 或需要云端推理 |

> 参考：[MobileAgent v3.5](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v3.5) 中的 GUI-Owl 1.5 是专为 GUI 自动化训练的模型，在 ScreenSpot、AndroidWorld 等基准上表现显著优于通用 VL 模型。

### 前置条件

#### 模式一：本地 Transformers（推荐 Windows）

1. 在 `pc` 目录准备运行环境（开发态）：
   ```bash
   cd screen-recorder-mvp/pc
   python -m venv venv
   venv\Scripts\activate
   pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```
2. 运行 `python main_web.py`，在「GUI Agent 视频分析」选择 `local` 后端，填写模型 ID（可选）后点击「一键部署本地模型」。
3. 首次部署会自动下载对应模型到应用目录下 `models/`，后续复用本地权重。

**按设备选择模型建议**：
- **8GB 显存（如 RTX 4060 Laptop）**：优先 `mPLUG/GUI-Owl-1.5-4B-Instruct`，若仍慢可换 2B
- **24GB 显存（如 RTX 4090）**：可用 `mPLUG/GUI-Owl-1.5-8B-Instruct`
- **无可用 NVIDIA GPU**：建议切换 `api` 后端走云端

#### 模式二：百炼云端 API

1. **申请阿里云百炼 API Key**：前往 [百炼控制台](https://bailian.console.aliyun.com/) 注册并创建 API Key。
2. **设置环境变量**（可选，也可在界面中直接填写）：
   ```bash
   # Windows CMD
   set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

   # PowerShell
   $env:DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxx"
   ```
3. **安装依赖**：
   ```bash
   cd screen-recorder-mvp/pc
   venv\Scripts\pip install -r requirements.txt
   ```

> 参考：GUI-Owl 1.5 模型与推理方式可见 [MobileAgent v3.5](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-v3.5)、[4B](https://huggingface.co/mPLUG/GUI-Owl-1.5-4B-Instruct)、[8B](https://huggingface.co/mPLUG/GUI-Owl-1.5-8B-Instruct)。

### 使用方法

1. 启动 PC 端程序 `python main_web.py`（`main.py` 为兼容旧 UI 入口）。
2. 在「GUI Agent 视频分析」区域选择后端：
   - `local`：本地模型（首次可点「一键部署本地模型」自动下载并加载）  
   - `api`：云端 API（需填写 API Key）
3. 点击「选择视频」选择录制好的 MP4 文件。
4. 点击「开始分析」，进度条会实时显示分析进度。
5. 分析完成后，点击「打开分析结果」查看输出。

### 输出说明

分析结果保存在 `analysis_output/<视频文件名>/` 目录下：

```
analysis_output/
└── record_20260310_164433/
    ├── frames/              # 关键帧截图 PNG
    │   ├── frame_0000_00s.png
    │   ├── frame_0001_02s.png
    │   └── ...
    ├── report.json          # 结构化 JSON 报告
    └── report.txt           # 人类可读文本报告
```

每一帧的分析结果包含：
- `app_name` — 当前应用名称
- `page_name` — 页面/界面名称
- `elements` — UI 元素列表（按钮、文本、输入框等）
- `visible_text` — 屏幕上可见的文本内容
- `user_action` — 推断的用户操作
- `description` — 屏幕内容的整体描述

### 分析流水线

1. 按固定间隔（默认 2 秒）从视频中提取帧
2. 通过 SSIM 算法去除高度相似的冗余帧，保留关键帧
3. 将每个关键帧发送到所选后端（本地模型或云 API）进行 GUI 内容识别
4. 解析返回结果为结构化 JSON，同时生成文本报告

### 配置参数

可在 `config.py` 中调整：
- `FRAME_SAMPLE_INTERVAL_SEC` — 帧采样间隔（默认 2 秒）
- `SSIM_THRESHOLD` — SSIM 去重阈值（默认 0.95，越高越严格）
- `LOCAL_GUI_OWL_MODEL_ID` — 本地模型 ID（默认 8B）
- `LOCAL_MODEL_GPU_MEMORY` / `LOCAL_MODEL_CPU_MEMORY` — 本地推理内存预算
- `LOCAL_MODEL_MAX_NEW_TOKENS` — 每帧生成 token 上限
- `LOCAL_IMAGE_MAX_EDGE` — 输入图像缩放上限
- `DASHSCOPE_MODEL` — 云端 API 模型（默认 `qwen-vl-max`）
- `ANALYSIS_OUTPUT_DIR` — 输出目录（默认 `analysis_output`）

---

## 六、记忆存储与检索

分析完成后的识别结果会**自动写入记忆**，支持数据清洗、压缩、用户画像和检索。

### 功能说明

- **自动存储**：每次视频分析完成后，识别结果自动存入 `memory/` 目录下的 SQLite 数据库。
- **数据清洗与压缩**：
  - 合并相似数据（同应用+同页面且在 10 秒时间窗口内）
  - 去除歧义数据（空应用名或过短描述）
  - 去除不合理数据（连续重复条目）
- **用户画像**：基于压缩记录统计最常用应用、各应用常用页面，支持「查看用户画像」。
- **检索**：支持**关键词**和**自然语言查询**（自动分词后匹配应用名、页面、操作、描述等字段）。

### 使用方法

1. 完成视频分析后，记忆会自动更新（日志会显示「已写入记忆」）。
2. 在「记忆」区域可查看原始/压缩条数，点击「刷新统计」更新。
3. 在搜索框输入关键词（如「微信」「设置」）或自然语言（如「用户打开微信聊天」），点击「检索」进行查询。
4. 点击「查看用户画像」查看应用使用统计和常用页面。

### 数据存储位置

- 开发/脚本运行：`screen-recorder-mvp/pc/memory/memory.db`
- 打包后：`ScreenRecorderPC/memory/memory.db`（与 exe 同级）

### 参考

记忆设计参考了 [M3-Agent](https://github.com/ByteDance-Seed/m3-agent) 的实体中心、分层记忆思路，详见 `pc/docs/M3_AGENT_MEMORY_RESEARCH.md`。

---

此为 **LLM-based Multi-modal GUI Agent for All day Screen Recording and Recognition** 的最小可运行原型，仅作演示与功能验证；后续可在此基础上增加多段录制列表、自动上传、设置帧率/码率等。
