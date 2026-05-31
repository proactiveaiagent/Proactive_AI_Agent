# Proactive AI Agent

[English](README.md)

基于第一人称 VR 视角视频的**主动式个人助手**。系统从用户佩戴眼镜录制的视频中提取画面与语音，结合多层记忆，自动识别场景、分析需求并生成解决方案。

## 项目结构

```
Proactive_Agent/
├── code/
│   ├── agent.py           # 主 Agent 流水线（推荐）
│   ├── agent_v0.py        # 早期版本，支持多种输入模式
│   ├── api_server_v0.py   # Qwen3-VL + Whisper API 服务
│   ├── memory.py          # 7 层分层记忆系统
│   ├── memory2.py         # 记忆模块备用实现
│   └── memory/            # 记忆持久化数据
├── models/
│   ├── Qwen3-VL-4B-Instruct/   # 视觉语言模型
│   └── whisper-tiny/           # 语音识别模型
├── test_data/             # 测试视频
└── results/               # 运行结果输出目录
```

## 工作流程

Agent 处理分为三个阶段：

| 阶段 | 内容 | 说明 |
|------|------|------|
| **Phase A** | Part 1–3 | 并行提取帧/音频 → 转写 → 场景分析 → 需求识别 → 方案生成 → 写入记忆 |
| **Phase B** | Part 4 | AR 展示与用户反馈 |
| **Phase C** | 记忆整合 | 在 Part 4 反馈完成后触发，不阻塞响应 |

### Part 1–4 详解

1. **场景识别** — 地点、时间、人物、用户行为
2. **需求分析** — 识别用户当前最需要的 3 项需求
3. **方案生成** — 针对每项需求给出具体建议
4. **用户反馈** — 确认或修正分析结果，用于更新记忆

## 记忆系统

`memory.py` 实现了 7 层分层记忆：

- **Layer 1–3**：当前时刻 → 同环境历史 → 今日全部时刻
- **Layer 4–6**：压缩的近期任务、长期事件、用户画像
- **Layer 7**：按时间、活动、人物、地点分类的归档

支持 add、update、query、retrieve、compress、sort、combine、delete、highlight 等 9 种操作。

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
python api_server_v0.py
```

服务默认监听 `http://0.0.0.0:8000`，提供以下接口：

| 端点 | 功能 |
|------|------|
| `POST /analyze` | 单帧图像 + 文本分析 |
| `POST /analyze_video` | 多帧视频序列分析 |
| `POST /transcribe` | 音频转写 |
| `GET /health` | 服务健康检查 |

### 3. 运行 Agent

```bash
cd code
python agent.py
```

在 `agent.py` 底部修改 `video_path` 指向你的测试视频，例如：

```python
video_path = "../test_data/test_data/test4.mp4"
assistant = VRAssistant(video_path)
results = assistant.process(consolidation_blocking=False)
```

## 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `qwen_api_url` | `http://localhost:8000` | API 服务地址 |
| `num_threads` | CPU 核心数 × 0.75 | PyTorch 线程数 |
| `WHISPER_MODEL_SIZE` | `medium` | Whisper 模型大小（环境变量） |
| `consolidation_blocking` | `False` | Phase C 是否在后台运行 |

## 测试数据

`test_data/test_data/` 包含多段第一人称场景视频，涵盖机场、酒店、饮食等日常场景，可用于评估 Agent 的场景识别与需求分析能力。

## License

待定
