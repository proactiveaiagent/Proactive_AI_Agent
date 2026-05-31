# Proactive AI Agent

[中文文档](README.zh-CN.md)

**Proactive AI Agent** is a context-aware assistant that perceives the world through multiple input streams and acts through multiple output channels — not just a VR video analyzer.

**Inputs** — the agent ingests and fuses:
- **User-shot video** — first-person footage from smart glasses, phone cameras, or POV recordings
- **Screen recordings** — phone or computer screen captures for app/UI/workflow context
- **Wearable sensor data** — heart rate, steps, location, posture, and other telemetry (standalone or combined with video)

**Outputs** — inferred needs are turned into actions delivered via:
- **Digital info & app services** — AR overlays, notifications, tips, and in-app guidance
- **Phone / PC operations** — open apps, tap UI, type text, send messages, navigate browsers
- **Hardware & robot commands** — smart-home controls and robot motion instructions

A 7-layer memory system retains scene history, user preferences, and past interactions so the agent can recognize context, infer needs proactively, and follow up on whether previous advice was followed.

## Project Structure

```
Proactive_Agent/
├── code/
│   ├── agent.py           # Main agent pipeline (multi-modal I/O)
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

## Input Modalities

Configure via `input_source` and optional `wearable_data` when creating a `VRAssistant`:

| Source | `input_source` | Description |
|--------|----------------|-------------|
| **User-shot video** | `camera` | First-person footage from smart glasses, phone camera, or POV recording |
| **Screen recording** | `screen_record` | Phone or computer screen-capture video (apps, UI, workflows) |
| **Wearable data** | `wearable` / `multimodal` | Sensor readings (heart rate, steps, location, posture) as JSON dict or file |

Combine video with wearable telemetry using `input_source="multimodal"`:

```python
assistant = VRAssistant(
    video_path="../test_data/test_data/2.travel_abroad/2.1.mp4",
    input_source="multimodal",
    wearable_data={"heart_rate": 110, "stress": "elevated"},
)
```

Wearable-only mode (no video):

```python
assistant = VRAssistant(
    input_source="wearable",
    wearable_data="memory/wearable_sample.json",
)
```

## Output Channels

Each Part 3 solution is tagged with an `output_type` and routed to the matching channel in Part 4:

| Channel | `output_type` | Examples |
|---------|---------------|----------|
| **Digital info & app services** | `digital_info` | AR overlays, notifications, tips, in-app guidance |
| **Phone / PC operations** | `device_control` | Open app, tap UI, type text, send message, navigate browser |
| **Hardware & robot commands** | `hardware_robot` | Turn on lights, adjust thermostat, robot arm motion |

## Pipeline Overview

The agent runs in four phases:

| Phase | Scope | Description |
|-------|-------|-------------|
| **Phase 0** | Pre-check | Compliance check — did the user follow solutions from the previous session? |
| **Phase A** | Part 1–3 | Scene analysis → need inference → solution generation → memory write |
| **Phase B** | Part 4 | Multi-channel output delivery + user feedback |
| **Phase C** | Consolidation | Triggered after Part 4 feedback; runs in the background by default |

Frame extraction, audio extraction, and transcription are performed **once** in `process()` and shared between Phase 0 and Phase A (skipped in wearable-only mode).

### Part 1–4 Breakdown

1. **Scene Recognition** — location, on-screen context, people, user actions
2. **Need Analysis** — top 3 current needs (informed by video, screen, and wearable signals)
3. **Solution Generation** — concrete actions tagged with an output channel
4. **Output Delivery & Feedback** — execute via digital / device / hardware channels, then collect user confirmation

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
| `POST /consolidate` | Text-only LLM call (wearable-only analysis) |
| `GET /health` | Health check |

### 3. Run the Agent

```bash
cd code
python agent.py
```

Examples at the bottom of `agent.py`:

```python
# User-shot video (default)
assistant = VRAssistant("../test_data/test_data/2.travel_abroad/2.1.mp4", input_source="camera")

# Screen recording
assistant = VRAssistant("../screen_record.mp4", input_source="screen_record")

# Video + wearable sensors
assistant = VRAssistant(video_path, input_source="multimodal", wearable_data={"heart_rate": 95})

results = assistant.process(consolidation_blocking=False)
```

## Configuration

### Agent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_source` | `"camera"` | Input type: `camera`, `screen_record`, `wearable`, `multimodal` |
| `wearable_data` | `None` | Wearable sensor dict or path to JSON file |
| `qwen_api_url` | `http://localhost:8000` | API server URL |
| `num_threads` | CPU cores × 0.75 | PyTorch thread count |
| `consolidation_blocking` | `False` | Whether Phase C runs in the foreground |
| `INPUT_MODE` | `"frame"` | Video decode mode: `"frame"`, `"video"`, or `"raw_video"` |
| `VIDEO_NUM_FRAMES` | `4` | Frames to sample when `INPUT_MODE == "video"` |
| `RAW_VIDEO_FPS` | `1.0` | FPS hint for `raw_video` mode |
| `RAW_VIDEO_MAX_FRAMES` | `16` | Max frames for `raw_video` mode |

### Input Modes (video decoding)

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
