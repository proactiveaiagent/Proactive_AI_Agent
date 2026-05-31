# Proactive AI Agent

[中文文档](README.zh-CN.md)

A **proactive personal assistant** powered by first-person VR video. The system extracts visual and audio signals from footage captured by smart glasses, combines them with a multi-layer memory system, and automatically recognizes scenes, infers user needs, and generates actionable solutions.

## Project Structure

```
Proactive_Agent/
├── code/
│   ├── agent.py           # Main agent pipeline
│   ├── api_server.py      # Qwen3-VL + Whisper API server
│   ├── memory.py          # 7-layer memory + HintMemory
│   ├── memory/            # Persistent memory data (memory.json, hints.json)
│   └── output/            # Run outputs (analysis, frames, audio)
├── models/
│   ├── Qwen3-VL-4B-Instruct/   # Vision-language model
│   └── whisper-tiny/           # Speech recognition model
├── test_data/             # Scenario-based test videos
└── results/               # Output directory for run results
```

## Pipeline Overview

The agent runs in four phases:

| Phase | Scope | Description |
|-------|-------|-------------|
| **Phase 0** | Pre-check | Compliance check — did the user follow solutions from the previous session? |
| **Phase A** | Part 1–3 | Scene analysis → need inference → solution generation → memory write |
| **Phase B** | Part 4 | AR display and user feedback |
| **Phase C** | Consolidation | Triggered after Part 4 feedback; runs in the background by default |

Frame extraction, audio extraction, and transcription are performed **once** in `process()` and shared between Phase 0 and Phase A.

### Part 1–4 Breakdown

1. **Scene Recognition** — location, time, people, and user actions
2. **Need Analysis** — identify the user's top 3 current needs
3. **Solution Generation** — provide concrete suggestions for each need
4. **User Feedback** — confirm or correct results to update memory

## Memory System

`memory.py` implements two complementary stores:

**PersonMemory** — 7-layer hierarchical memory:

- **Layers 1–3**: current moment → same-environment history → all moments today
- **Layers 4–6**: compressed recent tasks, long-term events, and user profile
- **Layer 7**: archive classified by time, activity, person, and location

Supports 9 operations: add, update, query, retrieve, compress, sort, combine, delete, and highlight.

**HintMemory** — user-curated trigger→need rules persisted in `memory/hints.json`, injected into need analysis at runtime.

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
python api_server.py
```

The server listens on `http://0.0.0.0:8000` by default and exposes:

| Endpoint | Description |
|----------|-------------|
| `POST /analyze` | Single-frame image + text analysis |
| `POST /analyze_video` | Multi-frame video sequence analysis |
| `POST /analyze_raw_video` | Raw video file analysis (native Qwen-VL decoding) |
| `POST /transcribe` | Audio transcription |
| `POST /consolidate` | Memory consolidation via LLM |
| `GET /health` | Health check |

### 3. Run the Agent

```bash
cd code
python agent.py
```

Update `video_path` at the bottom of `agent.py` to point to your test video:

```python
video_path = "../test_data/test_data/2.travel_abroad/2.1.mp4"
assistant = VRAssistant(video_path)
results = assistant.process(consolidation_blocking=False)
```

## Configuration

### Agent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `qwen_api_url` | `http://localhost:8000` | API server URL |
| `num_threads` | CPU cores × 0.75 | PyTorch thread count |
| `consolidation_blocking` | `False` | Whether Phase C runs in the foreground |
| `INPUT_MODE` | `"frame"` | Video input mode: `"frame"`, `"video"`, or `"raw_video"` |
| `VIDEO_NUM_FRAMES` | `4` | Frames to sample when `INPUT_MODE == "video"` |
| `RAW_VIDEO_FPS` | `1.0` | FPS hint for `raw_video` mode |
| `RAW_VIDEO_MAX_FRAMES` | `16` | Max frames for `raw_video` mode |

### Input Modes

| Mode | Description |
|------|-------------|
| `frame` | Extract the first frame only — fastest, lowest GPU cost |
| `video` | Uniformly sample `VIDEO_NUM_FRAMES` frames and send via `/analyze_video` |
| `raw_video` | Upload the raw clip to `/analyze_raw_video` for native decoding |

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `medium` | Whisper model size |

## Test Data

`test_data/test_data/` contains 12 scenario folders with first-person videos:

| Folder | Scenario |
|--------|----------|
| `1.social_hint` | Social hints |
| `2.travel_abroad` | Travel abroad |
| `3.forgetting_item` | Forgetting items |
| `4.first_aid` | First aid |
| `5.health_safeguard` | Health safeguard |
| `6.hotel_reception` | Hotel reception |
| `7.photo_taking` | Photo taking |
| `8.fraud_detection` | Fraud detection |
| `9.security_alert` | Security alert |
| `10.conversation_assistance` | Conversation assistance |
| `11.emotional support` | Emotional support |
| `12.shopping_selection` | Shopping selection |

## License

TBD
