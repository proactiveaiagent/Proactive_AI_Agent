# Proactive AI Agent

[English](README.md)

**Proactive AI Agent** 是一个感知多路输入、执行多通道输出的上下文感知助手——不再局限于第一人称 VR 视频分析。

**输入** — Agent 接收并融合以下数据：
- **用户拍摄视频** — 智能眼镜、手机摄像头或第一人称 POV 录像中的画面与语音
- **手机/电脑录屏** — 屏幕录制视频，用于理解应用界面、UI 操作和工作流程
- **可穿戴设备数据** — 心率、步数、位置、姿态等传感器信号（可单独使用，也可与视频组合）

**输出** — 识别出的需求会转化为具体行动，通过以下通道交付：
- **数字信息与应用服务** — AR 叠加、通知、提示信息、应用内引导
- **手机/电脑操作** — 打开应用、点击界面、输入文字、发送消息、浏览网页
- **智能硬件与机器人指令** — 智能家居控制、机器人动作指令

7 层记忆系统保存场景历史、用户偏好和过往交互，使 Agent 能够识别上下文、主动推断需求，并跟踪用户是否遵循了先前的建议。

## 项目结构

```
Proactive_Agent/
├── code/
│   ├── agent.py           # 主 Agent 流水线（多模态输入 / 多通道输出）
│   ├── api_server.py      # Qwen3-VL + Whisper API 服务
│   ├── memory.py          # 7 层记忆系统 + HintMemory
│   ├── memory/            # 记忆持久化数据（memory.json, hints.json）
│   └── output/            # 运行输出（分析结果、帧、音频）
├── models/
│   ├── Qwen3-VL-4B-Instruct/   # 视觉语言模型
│   └── whisper-tiny/           # 语音识别模型
├── test_data/             # 按场景分类的测试视频
└── results/               # 运行结果输出目录
```

## 输入模态

通过 `input_source` 和可选的 `wearable_data` 参数配置：

| 来源 | `input_source` | 说明 |
|------|----------------|------|
| **用户拍摄视频** | `camera` | 智能眼镜、手机摄像头或第一人称 POV 录像 |
| **手机/电脑录屏** | `screen_record` | 屏幕录制视频（应用界面、UI 操作、工作流程） |
| **可穿戴设备数据** | `wearable` / `multimodal` | 传感器读数（心率、步数、位置、姿态等），支持 JSON 字典或文件 |

视频与可穿戴数据组合使用：

```python
assistant = VRAssistant(
    video_path="../test_data/test_data/2.travel_abroad/2.1.mp4",
    input_source="multimodal",
    wearable_data={"heart_rate": 110, "stress": "elevated"},
)
```

仅可穿戴数据（无视频）：

```python
assistant = VRAssistant(
    input_source="wearable",
    wearable_data="memory/wearable_sample.json",
)
```

## 输出通道

Part 3 生成的每项方案带有 `output_type` 标签，在 Part 4 中路由到对应通道执行：

| 通道 | `output_type` | 示例 |
|------|---------------|------|
| **数字信息与应用服务** | `digital_info` | AR 叠加、通知、提示、应用内引导 |
| **手机/电脑操作** | `device_control` | 打开应用、点击 UI、输入文字、发送消息、浏览网页 |
| **智能硬件与机器人指令** | `hardware_robot` | 开灯、调温、机器人手臂动作 |

## 工作流程

Agent 处理分为四个阶段：

| 阶段 | 内容 | 说明 |
|------|------|------|
| **Phase 0** | 预检查 | 行为合规检查 — 用户是否遵循了上次会话的建议？ |
| **Phase A** | Part 1–3 | 场景分析 → 需求识别 → 方案生成 → 写入记忆 |
| **Phase B** | Part 4 | 多通道输出交付 + 用户反馈 |
| **Phase C** | 记忆整合 | 在 Part 4 反馈完成后触发，默认后台运行 |

帧提取、音频提取和转写在 `process()` 中**只执行一次**（可穿戴纯数据模式会跳过视频提取）。

### Part 1–4 详解

1. **场景识别** — 地点、屏幕内容、人物、用户行为
2. **需求分析** — 识别当前最需要的 3 项需求（结合视频、录屏和可穿戴信号）
3. **方案生成** — 针对每项需求给出带输出通道标签的具体行动
4. **输出交付与反馈** — 通过数字/设备/硬件通道执行，并收集用户确认

## 记忆系统

`memory.py` 包含两个互补的存储模块：

**PersonMemory** — 7 层分层记忆：

- **Layer 1–3**：当前时刻 → 同环境历史 → 今日全部时刻
- **Layer 4–6**：压缩的近期任务、长期事件、用户画像
- **Layer 7**：按时间、活动、人物、地点分类的归档

支持 add、update、query、retrieve、compress、sort、combine、delete、highlight 等 9 种操作。

**HintMemory** — 用户自定义的触发条件→需求规则，持久化在 `memory/hints.json`。

## 环境要求

- Python 3.10+
- CUDA GPU（推荐，用于 Qwen3-VL 推理）
- 依赖：`torch`、`transformers`、`fastapi`、`uvicorn`、`opencv-python`、`moviepy`、`faster-whisper`、`Pillow`、`requests`

## 快速开始

### 1. 下载模型

将以下模型放入 `models/` 目录：

- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) → `models/Qwen3-VL-4B-Instruct/`
- [whisper-tiny](https://huggingface.co/openai/whisper-tiny) → `models/whisper-tiny/`

### 2. 启动 API 服务

```bash
cd code
python api_server.py
```

| 端点 | 功能 |
|------|------|
| `POST /analyze` | 单帧图像 + 文本分析 |
| `POST /analyze_video` | 多帧视频序列分析 |
| `POST /analyze_raw_video` | 原始视频文件分析 |
| `POST /transcribe` | 音频转写 |
| `POST /consolidate` | 纯文本 LLM 调用（可穿戴数据分析） |
| `GET /health` | 服务健康检查 |

### 3. 运行 Agent

```bash
cd code
python agent.py
```

```python
# 用户拍摄视频（默认）
assistant = VRAssistant("../test_data/test_data/2.travel_abroad/2.1.mp4", input_source="camera")

# 手机/电脑录屏
assistant = VRAssistant("../screen_record.mp4", input_source="screen_record")

# 视频 + 可穿戴传感器
assistant = VRAssistant(video_path, input_source="multimodal", wearable_data={"heart_rate": 95})

results = assistant.process(consolidation_blocking=False)
```

## 配置说明

### Agent 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_source` | `"camera"` | 输入类型：`camera`、`screen_record`、`wearable`、`multimodal` |
| `wearable_data` | `None` | 可穿戴传感器字典或 JSON 文件路径 |
| `qwen_api_url` | `http://localhost:8000` | API 服务地址 |
| `consolidation_blocking` | `False` | Phase C 是否在后台运行 |
| `INPUT_MODE` | `"frame"` | 视频解码模式：`frame` / `video` / `raw_video` |
| `VIDEO_NUM_FRAMES` | `4` | `INPUT_MODE == "video"` 时的采样帧数 |
| `RAW_VIDEO_FPS` | `1.0` | `raw_video` 模式的 FPS 提示 |
| `RAW_VIDEO_MAX_FRAMES` | `16` | `raw_video` 模式的最大帧数 |

### 输入模式（视频解码）

| 模式 | 说明 |
|------|------|
| `frame` | 仅提取首帧 — 最快，GPU 开销最低 |
| `video` | 均匀采样多帧，通过 `/analyze_video` 发送 |
| `raw_video` | 上传原始视频到 `/analyze_raw_video` |

### 服务环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `WHISPER_MODEL_SIZE` | `medium` | Whisper 模型大小 |

## 测试数据

`test_data/test_data/` 包含 12 个场景目录：

| 目录 | 场景 |
|------|------|
| `1.social_hint` | 社交提示 |
| `2.travel_abroad` | 出国旅行 |
| `3.forgetting_item` | 遗忘物品 |
| `4.first_aid` | 急救 |
| `5.health_safeguard` | 健康防护 |
| `6.hotel_reception` | 酒店前台 |
| `7.photo_taking` | 拍照 |
| `8.fraud_detection` | 诈骗识别 |
| `9.security_alert` | 安全警报 |
| `10.conversation_assistance` | 对话辅助 |
| `11.emotional support` | 情感支持 |
| `12.shopping_selection` | 购物选择 |

## License

待定
