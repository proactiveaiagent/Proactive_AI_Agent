# Proactive AI Agent

[中文文档](README.zh-CN.md)

A **proactive personal assistant** powered by first-person VR video. The system extracts visual and audio signals from footage captured by smart glasses, combines them with a multi-layer memory system, and automatically recognizes scenes, infers user needs, and generates actionable solutions.

## Project Structure

```
Proactive_Agent/
├── code/
│   ├── agent.py           # Main agent pipeline (recommended)
│   ├── agent_v0.py        # Earlier version with multiple input modes
│   ├── api_server_v0.py   # Qwen3-VL + Whisper API server
│   ├── memory.py          # 7-layer hierarchical memory system
│   ├── memory2.py         # Alternate memory implementation
│   └── memory/            # Persistent memory data
├── models/
│   ├── Qwen3-VL-4B-Instruct/   # Vision-language model
│   └── whisper-tiny/           # Speech recognition model
├── test_data/             # Test videos
└── results/               # Output directory for run results
```

## Pipeline Overview

The agent runs in three phases:

| Phase | Scope | Description |
|-------|-------|-------------|
| **Phase A** | Part 1–3 | Parallel frame/audio extraction → transcription → scene analysis → need inference → solution generation → memory write |
| **Phase B** | Part 4 | AR display and user feedback |
| **Phase C** | Consolidation | Triggered after Part 4 feedback; does not block the response path |

### Part 1–4 Breakdown

1. **Scene Recognition** — location, time, people, and user actions
2. **Need Analysis** — identify the user's top 3 current needs
3. **Solution Generation** — provide concrete suggestions for each need
4. **User Feedback** — confirm or correct results to update memory

## Memory System

`memory.py` implements a 7-layer hierarchical memory:

- **Layers 1–3**: current moment → same-environment history → all moments today
- **Layers 4–6**: compressed recent tasks, long-term events, and user profile
- **Layer 7**: archive classified by time, activity, person, and location

Supports 9 operations: add, update, query, retrieve, compress, sort, combine, delete, and highlight.

## Requirements

- Python 3.10+
- CUDA GPU (recommended for Qwen3-VL inference)
- Dependencies: `torch`, `transformers`, `fastapi`, `uvicorn`, `opencv-python`, `moviepy`, `faster-whisper`, `Pillow`, `requests`

## Quick Start

### 1. Download Models

Place the following models under `models/`:

- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) → `models/Qwen3-VL-4B-Instruct/`
- [whisper-tiny](https://huggingface.co/openai/whisper-tiny) → `models/whisper-tiny/` (or let `faster-whisper` download automatically)

### 2. Start the API Server

```bash
cd code
python api_server_v0.py
```

The server listens on `http://0.0.0.0:8000` by default and exposes:

| Endpoint | Description |
|----------|-------------|
| `POST /analyze` | Single-frame image + text analysis |
| `POST /analyze_video` | Multi-frame video sequence analysis |
| `POST /transcribe` | Audio transcription |
| `GET /health` | Health check |

### 3. Run the Agent

```bash
cd code
python agent.py
```

Update `video_path` at the bottom of `agent.py` to point to your test video:

```python
video_path = "../test_data/test_data/test4.mp4"
assistant = VRAssistant(video_path)
results = assistant.process(consolidation_blocking=False)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `qwen_api_url` | `http://localhost:8000` | API server URL |
| `num_threads` | CPU cores × 0.75 | PyTorch thread count |
| `WHISPER_MODEL_SIZE` | `medium` | Whisper model size (env var) |
| `consolidation_blocking` | `False` | Whether Phase C runs in the foreground |

## Test Data

`test_data/test_data/` contains first-person scene videos covering everyday settings such as airports, hotels, and dining. Use them to evaluate scene recognition and need analysis.

## License

TBD
