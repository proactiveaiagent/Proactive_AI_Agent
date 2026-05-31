# Proactive AI Agent

[English](README.md)

基于第一人称 VR 视角视频的**主动式个人助手**。系统从用户佩戴眼镜录制的视频中提取画面与语音，结合多层记忆，自动识别场景、分析需求并生成解决方案。

## 项目结构

```
Proactive_Agent/
├── code/
│   ├── agent.py           # 主 Agent 流水线
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

## 工作流程

Agent 处理分为四个阶段：

| 阶段 | 内容 | 说明 |
|------|------|------|
| **Phase 0** | 预检查 | 行为合规检查 — 用户是否遵循了上次会话的建议？ |
| **Phase A** | Part 1–3 | 场景分析 → 需求识别 → 方案生成 → 写入记忆 |
| **Phase B** | Part 4 | AR 展示与用户反馈 |
| **Phase C** | 记忆整合 | 在 Part 4 反馈完成后触发，默认后台运行 |

帧提取、音频提取和转写在 `process()` 中**只执行一次**，Phase 0 与 Phase A 共享结果，避免重复计算。

### Part 1–4 详解

1. **场景识别** — 地点、时间、人物、用户行为
2. **需求分析** — 识别用户当前最需要的 3 项需求
3. **方案生成** — 针对每项需求给出具体建议
4. **用户反馈** — 确认或修正分析结果，用于更新记忆

## 记忆系统

`memory.py` 包含两个互补的存储模块：

**PersonMemory** — 7 层分层记忆：

- **Layer 1–3**：当前时刻 → 同环境历史 → 今日全部时刻
- **Layer 4–6**：压缩的近期任务、长期事件、用户画像
- **Layer 7**：按时间、活动、人物、地点分类的归档

支持 add、update、query、retrieve、compress、sort、combine、delete、highlight 等 9 种操作。

**HintMemory** — 用户自定义的触发条件→需求规则，持久化在 `memory/hints.json`，运行时注入需求分析。

## 环境要求

- Python 3.10+
- CUDA GPU（推荐，用于 Qwen3-VL 推理）
- 依赖：`torch`、`transformers`、`fastapi`、`uvicorn`、`opencv-python`、`moviepy`、`faster-whisper`、`Pillow`、`requests`

## 快速开始

### 1. 下载模型

将以下模型放入 `models/` 目录：

- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) → `models/Qwen3-VL-4B-Instruct/`
- [whisper-tiny](https://huggingface.co/openai/whisper-tiny) → `models/whisper-tiny/`（或通过 `faster-whisper` 自动下载）

### 2. 启动 API 服务

```bash
cd code
python api_server.py
```

服务默认监听 `http://0.0.0.0:8000`，提供以下接口：

| 端点 | 功能 |
|------|------|
| `POST /analyze` | 单帧图像 + 文本分析 |
| `POST /analyze_video` | 多帧视频序列分析 |
| `POST /analyze_raw_video` | 原始视频文件分析（Qwen-VL 原生解码） |
| `POST /transcribe` | 音频转写 |
| `POST /consolidate` | 通过 LLM 进行记忆整合 |
| `GET /health` | 服务健康检查 |

### 3. 运行 Agent

```bash
cd code
python agent.py
```

在 `agent.py` 底部修改 `video_path` 指向你的测试视频，例如：

```python
video_path = "../test_data/test_data/2.travel_abroad/2.1.mp4"
assistant = VRAssistant(video_path)
results = assistant.process(consolidation_blocking=False)
```

## 配置说明

### Agent 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `qwen_api_url` | `http://localhost:8000` | API 服务地址 |
| `num_threads` | CPU 核心数 × 0.75 | PyTorch 线程数 |
| `consolidation_blocking` | `False` | Phase C 是否在后台运行 |
| `INPUT_MODE` | `"frame"` | 视频输入模式：`"frame"`、`"video"` 或 `"raw_video"` |
| `VIDEO_NUM_FRAMES` | `4` | `INPUT_MODE == "video"` 时的采样帧数 |
| `RAW_VIDEO_FPS` | `1.0` | `raw_video` 模式的 FPS 提示 |
| `RAW_VIDEO_MAX_FRAMES` | `16` | `raw_video` 模式的最大帧数 |

### 输入模式

| 模式 | 说明 |
|------|------|
| `frame` | 仅提取首帧 — 最快，GPU 开销最低 |
| `video` | 均匀采样 `VIDEO_NUM_FRAMES` 帧，通过 `/analyze_video` 发送 |
| `raw_video` | 上传原始视频到 `/analyze_raw_video`，由模型原生解码 |

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
