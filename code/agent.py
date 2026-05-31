"""
Proactive Agent — Multi-modal input / multi-channel output pipeline
====================================================================
Inputs (video + optional wearable telemetry):
  • camera        — first-person video captured by the user (smart glasses / phone camera)
  • screen_record — phone or computer screen-recording video
  • wearable      — sensor data from watches, bands, glasses, etc. (JSON dict or file)
  Combine camera/screen_record video with wearable data via input_source="multimodal".

Outputs (Part 4 delivery channels):
  • digital_info    — AR overlays, notifications, digital content, app services
  • device_control  — phone / PC operations (open app, tap, type, navigate UI)
  • hardware_robot  — smart-home devices and robot motion / action commands

Timing split:
  • PHASE 0  (Pre-check)  : compliance check — did user follow last session's solutions?
  • PHASE A  (Part 1-3)   : parallel extraction → transcribe → analyse → memory add
  • PHASE B  (Part 4)     : multi-channel output + user feedback  ← separate timer
  • PHASE C  (Post-4)     : memory consolidation  ← triggered AFTER Part4 feedback,
                            never during Part1-3, so it never blocks response time

Note: frame extraction, audio extraction, and transcription are performed ONCE
in process() and shared between Phase 0 and Phase A to avoid duplicate work.

Memory: uses memory3.PersonMemory (7-layer, 9 operations)
  - Moments are stored in layer1 (up to MAX_LAYER1=5).
  - They graduate to layer2 only when layer1 overflows.
  - Each moment uses "id" as its key (not "moment_id").
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
from memory import PersonMemory, HintMemory
import time
from functools import wraps
import requests
import json
import re
import textwrap
import threading


# ════════════════════════════════════════════════════════════════════════════
# 🎛️  CONFIG SWITCHES — change these to control how the agent ingests video
# ════════════════════════════════════════════════════════════════════════════
#
#   INPUT_MODE
#     "frame"     → (default, fastest) extract only the first frame and
#                   analyse it with /analyze. Cheapest path on the GPU.
#     "video"     → uniformly sample VIDEO_NUM_FRAMES frames across the clip
#                   and send them all in one /analyze_video request, so
#                   Qwen-VL sees motion / action progression instead of a
#                   single still. Latency ≈ linear in frame count.
#     "raw_video" → upload the raw video file to /analyze_raw_video and let
#                   Qwen-VL's processor decode + sample frames natively.
#                   The model sees the full clip (no client-side picking),
#                   but it's the slowest option because the processor will
#                   typically decode more frames than the sampled mode.
#
#   VIDEO_NUM_FRAMES
#     Number of frames to sample when INPUT_MODE == "video".
#     Keep this small (4–8) — cost grows roughly linearly with frame count.
#     Ignored in "frame" and "raw_video" modes.
#
#   RAW_VIDEO_FPS / RAW_VIDEO_MAX_FRAMES
#     Hints passed to /analyze_raw_video so the processor caps its internal
#     frame decoding (otherwise long clips can blow up GPU memory). Set
#     either to None to let the processor use its own defaults.
#
#   All four can be overridden per-instance via VRAssistant(input_mode=...,
#   video_num_frames=..., raw_video_fps=..., raw_video_max_frames=...).
# ════════════════════════════════════════════════════════════════════════════

INPUT_MODE = "frame"            # "frame", "video", or "raw_video"
VIDEO_NUM_FRAMES = 4            # only used when INPUT_MODE == "video"
RAW_VIDEO_FPS = 1.0             # only used when INPUT_MODE == "raw_video"
RAW_VIDEO_MAX_FRAMES = 16       # only used when INPUT_MODE == "raw_video"

# INPUT_SOURCE — what kind of video / sensor data is being ingested
#   "camera"        → user-shot first-person or POV video (default)
#   "screen_record" → phone / computer screen-recording video
#   "wearable"      → wearable sensor data only (no video required)
#   "multimodal"    → video (camera or screen_record) + wearable_data combined
INPUT_SOURCE = "camera"

# Valid Part-3 / Part-4 output channels
OUTPUT_CHANNELS = ("digital_info", "device_control", "hardware_robot")


# ---------------------------------------------------------------------------
# Timer decorator
# ---------------------------------------------------------------------------

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# VRAssistant
# ---------------------------------------------------------------------------

class VRAssistant:
    def __init__(self, video_path=None, output_dir="output",
                 num_threads=None, qwen_api_url="http://localhost:8000",
                 input_mode: str = None, video_num_frames: int = None,
                 raw_video_fps: float = None, raw_video_max_frames: int = None,
                 input_source: str = None, wearable_data=None):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.qwen_api_url = qwen_api_url

        self.input_source = (input_source or INPUT_SOURCE).lower()
        valid_sources = ("camera", "screen_record", "wearable", "multimodal")
        if self.input_source not in valid_sources:
            raise ValueError(
                f"input_source must be one of {valid_sources}, got {self.input_source!r}"
            )
        self.wearable_data = self._load_wearable_data(wearable_data)

        if self.input_source == "wearable" and not self.wearable_data:
            raise ValueError("input_source='wearable' requires wearable_data")
        if self.input_source != "wearable" and not self.video_path:
            raise ValueError(f"input_source={self.input_source!r} requires video_path")

        # Resolve input-mode config: per-instance arg > module-level constant
        self.input_mode = (input_mode or INPUT_MODE).lower()
        if self.input_mode not in ("frame", "video", "raw_video"):
            raise ValueError(
                f"input_mode must be 'frame', 'video', or 'raw_video', got {self.input_mode!r}"
            )
        self.video_num_frames = video_num_frames or VIDEO_NUM_FRAMES
        self.raw_video_fps = raw_video_fps if raw_video_fps is not None else RAW_VIDEO_FPS
        self.raw_video_max_frames = (
            raw_video_max_frames if raw_video_max_frames is not None else RAW_VIDEO_MAX_FRAMES
        )

        if self.input_mode == "video":
            print(f"🎛️  Input mode: video  (sampling {self.video_num_frames} frames)")
        elif self.input_mode == "raw_video":
            print(
                f"🎛️  Input mode: raw_video  "
                f"(fps={self.raw_video_fps}, max_frames={self.raw_video_max_frames})"
            )
        else:
            print(f"🎛️  Input mode: frame")

        source_labels = {
            "camera": "user-shot video",
            "screen_record": "phone/PC screen recording",
            "wearable": "wearable sensor data",
            "multimodal": "video + wearable sensors",
        }
        print(f"📡 Input source: {self.input_source} ({source_labels[self.input_source]})")
        if self.wearable_data:
            print(f"⌚ Wearable sensors loaded: {', '.join(self.wearable_data.keys())}")

        # CPU threading
        if num_threads is None:
            num_threads = int(os.cpu_count() * 0.75)
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Device: {self.device}")
        if self.device == "cuda":
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"🔧 CPU cores: {os.cpu_count()}  threads: {num_threads}")

        # Separate timing buckets
        #   phase_0 = Pre-check  (compliance check against last session's solutions)
        #   phase_a = Part1-3    (extraction → analysis → memory add)
        #   phase_b = Part4      (multi-channel output + user feedback)
        #   phase_c = Post-4     (consolidation)
        self.timings_phase_0: dict = {}
        self.timings_phase_a: dict = {}
        self.timings_phase_b: dict = {}
        self.timings_phase_c: dict = {}

        # Memory
        print("Initialising memory (7-layer)...")
        self.memory = PersonMemory()

        # Hint memory — extra user-curated trigger→need rules, separate from
        # the episodic 7-layer memory. Hints are spliced into the analysis
        # prompt so the agent can recognise needs the base model misses.
        # Add a hint at runtime with: assistant.add_hint("when X is seen", "output Y")
        # Or edit memory/hints.json by hand.
        self.hint_memory = HintMemory()
        n_hints = len(self.hint_memory.list())
        if n_hints:
            print(f"💡 Loaded {n_hints} user hint rule(s) from {self.hint_memory.hints_file}")

        # API health check
        print("Checking API server (Qwen + Whisper)...")
        try:
            resp = requests.get(f"{self.qwen_api_url}/health", timeout=2)
            h = resp.json()
            ok_q = h.get("model_loaded", False)
            ok_w = h.get("whisper_loaded", False)
            if ok_q and ok_w:
                print("✅ API server ready (Qwen ✓  Whisper ✓)")
            else:
                missing = [m for m, ok in [("Qwen", ok_q), ("Whisper", ok_w)] if not ok]
                print(f"⚠️  API server running but missing: {', '.join(missing)}")
        except Exception:
            print("❌ API server not available. Run: python api_server.py")

    # -----------------------------------------------------------------------
    # ─── Input-source helpers ─────────────────────────────────────────────
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_wearable_data(wearable_data) -> dict:
        """Accept a dict or a path to a JSON file with wearable sensor readings."""
        if wearable_data is None:
            return {}
        if isinstance(wearable_data, dict):
            return wearable_data
        path = Path(wearable_data)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {"readings": data}
        raise FileNotFoundError(f"Wearable data file not found: {path}")

    def _format_wearable_section(self) -> str:
        if not self.wearable_data:
            return ""
        lines = ["Wearable sensor data:"]
        for key, value in self.wearable_data.items():
            if isinstance(value, dict):
                detail = ", ".join(f"{k}={v}" for k, v in value.items())
                lines.append(f"- {key}: {detail}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_input_context(self) -> str:
        """Describe the active input modalities for LLM prompts."""
        parts = []
        if self.input_source in ("camera", "screen_record", "multimodal") and self.video_path:
            if self.input_source == "screen_record":
                parts.append(
                    "Input type: phone/computer SCREEN RECORDING. Focus on on-screen UI, "
                    "apps, text, notifications, cursor/touch interactions, and workflow context."
                )
            else:
                parts.append(
                    "Input type: user-shot FIRST-PERSON video (camera / smart glasses). "
                    "Focus on the physical scene, people, objects, and user actions."
                )
        if self.input_source in ("wearable", "multimodal") and self.wearable_data:
            parts.append(
                "Supplemental wearable telemetry is provided — use heart rate, steps, "
                "location, posture, or other sensor signals to refine need analysis."
            )
        if self.input_source == "wearable" and not self.video_path:
            parts.append(
                "Input type: WEARABLE SENSOR DATA ONLY (no video). Infer context and needs "
                "from physiological / motion / location signals."
            )
        wearable_block = self._format_wearable_section()
        if wearable_block:
            parts.append(wearable_block)
        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # ─── Shared extraction helpers ────────────────────────────────────────
    # -----------------------------------------------------------------------

    @timer
    def extract_first_frame(self) -> str:
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Failed to extract first frame")
        path = self.output_dir / "first_frame.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    @timer
    def extract_video_frames(self, num_frames: int = None) -> list:
        """
        Uniformly sample N frames from the video and write them as JPEGs.

        Used only when self.input_mode == "video". Sampling N small frames
        keeps the GPU cost roughly linear in N (rather than reading every
        single frame in the clip), so a value of 4–8 keeps the video path
        close in latency to the original first-frame path.
        """
        n = num_frames or self.video_num_frames
        if n < 1:
            n = 1

        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if total <= 0:
            # Some containers don't report frame count — fall back to reading
            # frames sequentially until exhaustion (rare path).
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                raise ValueError("Failed to read any frames from video")
            indices = np.linspace(0, len(frames) - 1, num=min(n, len(frames)), dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            # Uniformly spaced frame indices across the clip
            indices = np.linspace(0, max(total - 1, 0), num=min(n, total), dtype=int)
            sampled = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, fr = cap.read()
                if ok:
                    sampled.append(fr)
            cap.release()
            if not sampled:
                raise ValueError("Failed to sample any frames from video")

        out_paths = []
        for i, fr in enumerate(sampled):
            p = self.output_dir / f"video_frame_{i:02d}.jpg"
            cv2.imwrite(str(p), fr)
            out_paths.append(str(p))
        print(f"🎞️  Sampled {len(out_paths)} frame(s) for video-mode analysis")
        return out_paths

    @timer
    def extract_audio(self) -> str:
        path = self.output_dir / "audio.wav"
        video = VideoFileClip(self.video_path)
        video.audio.write_audiofile(str(path), verbose=False, logger=None)
        video.close()
        return str(path)

    @timer
    def transcribe_audio(self, audio_path: str) -> dict:
        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    f"{self.qwen_api_url}/transcribe",
                    files={"audio": (Path(audio_path).name, f, "audio/wav")},
                    timeout=120
                )
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("segments") or not data.get("full_text"):
                    data["no_speech"] = True
                    print("⚠️  No speech detected — visual-only analysis.")
                return data
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"⚠️  Transcription failed ({e}). Proceeding with visual-only analysis.")
            return {"language": "unknown", "full_text": "", "segments": [], "no_speech": True}

    def _extract_visual(self) -> list:
        """
        Return the list of frame paths to feed into the vision model.
          - 'frame'     → [first_frame_jpg]
          - 'video'     → [frame_0_jpg, ..., frame_{n-1}_jpg]   (uniform sample)
          - 'raw_video' → []  (no client-side extraction; the raw file is
                          uploaded by _call_vision_api → /analyze_raw_video)
        Centralised so Phase 0 and Phase A see exactly the same visual input.
        """
        if self.input_mode == "video":
            return self.extract_video_frames(self.video_num_frames)
        if self.input_mode == "raw_video":
            print("🎬 Raw-video mode: skipping client-side frame extraction "
                  "— the video file will be uploaded to /analyze_raw_video.")
            return []
        return [self.extract_first_frame()]

    def _call_vision_api(self, frame_paths: list, prompt: str, timeout: int = 120) -> requests.Response:
        """
        Dispatch a vision request to the right endpoint:
          - raw_video mode → POST /analyze_raw_video  (uploads the original clip)
          - 1 frame        → POST /analyze            (single image, original path)
          - N frames       → POST /analyze_video      (multi-image, sampled mode)
        Returns the raw `requests.Response` so callers can keep their existing
        error-handling logic.
        """
        # ── raw video: send the original clip and let Qwen decode it ─────
        if self.input_mode == "raw_video":
            data = {"prompt": prompt}
            if self.raw_video_fps is not None:
                data["fps"] = str(self.raw_video_fps)
            if self.raw_video_max_frames is not None:
                data["max_frames"] = str(self.raw_video_max_frames)
            with open(self.video_path, "rb") as vf:
                return requests.post(
                    f"{self.qwen_api_url}/analyze_raw_video",
                    files={"video": (Path(self.video_path).name, vf, "video/mp4")},
                    data=data,
                    timeout=max(timeout, 300),  # raw decoding can be slow on long clips
                )

        # ── single still ─────────────────────────────────────────────────
        if len(frame_paths) == 1:
            with open(frame_paths[0], "rb") as f:
                return requests.post(
                    f"{self.qwen_api_url}/analyze",
                    files={"image": f},
                    data={"prompt": prompt},
                    timeout=timeout,
                )

        # ── multi-frame (sampled video mode) ─────────────────────────────
        files = []
        opened = []
        try:
            for p in frame_paths:
                fh = open(p, "rb")
                opened.append(fh)
                files.append(("images", (Path(p).name, fh, "image/jpeg")))
            return requests.post(
                f"{self.qwen_api_url}/analyze_video",
                files=files,
                data={"prompt": prompt},
                timeout=timeout,
            )
        finally:
            for fh in opened:
                try:
                    fh.close()
                except Exception:
                    pass

    def _call_text_api(self, prompt: str, timeout: int = 120) -> requests.Response:
        """Text-only analysis via /consolidate (wearable-only or sensor-heavy input)."""
        return requests.post(
            f"{self.qwen_api_url}/consolidate",
            json={"prompt": prompt},
            timeout=timeout,
        )

    def format_transcript(self, td: dict) -> str:
        if td.get("no_speech") or not td.get("segments"):
            return "Language: unknown\n\n[No speech detected — visual-only analysis]\n"
        out = f"Language: {td['language']}\n\nTranscript with timestamps:\n"
        for seg in td["segments"]:
            out += f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n"
        return out

    # -----------------------------------------------------------------------
    # ─── PHASE 0 — Behaviour compliance check ────────────────────────────
    # -----------------------------------------------------------------------

    def _get_last_moment(self) -> dict | None:
        """
        Return the most recent moment, or None if memory is empty.

        Moments are stored in layer1 (PersonMemory.MAX_LAYER1 = 5 slots).
        They graduate to layer2 only when layer1 overflows, so layer2 is
        empty for the first 5 moments. We always check layer1 first, then
        fall back to layer2 for older sessions.

        Each moment uses the key "id" (set by _empty_moment() in memory3.py),
        NOT "moment_id".
        """
        layer1 = self.memory.memory.get("layer1", [])
        if layer1:
            return layer1[-1]
        # Fallback: layer2 holds moments that graduated out of layer1
        layer2 = self.memory.memory.get("layer2", [])
        if layer2:
            return layer2[-1]
        return None

    def run_phase_0(self, frame_path, formatted_transcript: str) -> dict:
        """
        Phase 0 — Behaviour compliance check.
        Checks whether the user followed the last session's recommended solutions.

        Uses the already-extracted frame and transcript (passed in from process())
        so no duplicate extraction cost is incurred.

        Timing is tracked in self.timings_phase_0, completely separate from
        Phase A / B / C buckets.

        Returns a dict:
          {
            "skipped": bool,
            "compliance": dict | None,   # None when skipped
            "reason": str | None         # set when skipped
          }
        """
        phase_start = time.time()

        print("\n" + "=" * 60)
        print("🔍 PHASE 0 — Behaviour compliance check")
        print("=" * 60)

        # ── retrieve last moment's solutions from memory ─────────────────
        t0 = time.time()
        last_moment = self._get_last_moment()
        self.timings_phase_0["memory_retrieval"] = time.time() - t0

        if not last_moment:
            print("⏭️  No previous moment found — skipping compliance check.")
            self.timings_phase_0["total_phase_0"] = time.time() - phase_start
            return {"skipped": True, "compliance": None, "reason": "no_previous_moment"}

        last_solutions = last_moment.get("solutions", [])
        if not last_solutions:
            print("⏭️  No previous solutions found — skipping compliance check.")
            self.timings_phase_0["total_phase_0"] = time.time() - phase_start
            return {"skipped": True, "compliance": None, "reason": "no_previous_solutions"}

        # moment["id"] is the correct key (defined in _empty_moment() in memory3.py)
        last_moment_id = last_moment.get("id")

        solutions_text = "\n".join(
            f"  - Solution {i + 1}: {s.get('solution', '')}"
            for i, s in enumerate(last_solutions)
        )
        print(f"📋 Last moment id : {last_moment_id}")
        print(f"📋 Last solutions to check against:\n{solutions_text}")

        # ── LLM compliance check ─────────────────────────────────────────
        t0 = time.time()
        prompt = f"""You are reviewing whether a user followed their assistant's previous recommendations.

{self._format_input_context()}

PREVIOUS RECOMMENDED SOLUTIONS:
{solutions_text}

CURRENT OBSERVATION:
{formatted_transcript}

Based on the current input, did the user act according to any of the previous solutions?

Respond ONLY with this JSON format, no markdown, no extra text:
{{
  "followed": [1, 2],
  "not_followed": [3],
  "uncertain": [],
  "summary": "One sentence summary of compliance."
}}

Rules:
- "followed"     : solution numbers the user clearly acted on
- "not_followed" : solution numbers the user clearly ignored
- "uncertain"    : solution numbers where it is unclear
- Keep "summary" to one sentence."""

        try:
            frame_paths = [frame_path] if isinstance(frame_path, str) else (list(frame_path) if frame_path else [])
            if frame_paths:
                resp = self._call_vision_api(frame_paths, prompt, timeout=60)
            else:
                resp = self._call_text_api(prompt, timeout=60)
            self.timings_phase_0["llm_check"] = time.time() - t0

            if resp.status_code != 200:
                print(f"⚠️  Compliance check API error: {resp.status_code}")
                self.timings_phase_0["total_phase_0"] = time.time() - phase_start
                return {"skipped": True, "compliance": None, "reason": "api_error"}

            raw = resp.json()["analysis"]
            compliance = self._parse_json_safe(raw)

            if not compliance:
                print(f"⚠️  Could not parse compliance JSON. Raw: {raw[:200]}")
                self.timings_phase_0["total_phase_0"] = time.time() - phase_start
                return {"skipped": True, "compliance": None, "reason": "parse_error"}

        except Exception as e:
            print(f"⚠️  Compliance check failed: {e}")
            self.timings_phase_0["llm_check"] = time.time() - t0
            self.timings_phase_0["total_phase_0"] = time.time() - phase_start
            return {"skipped": True, "compliance": None, "reason": f"exception: {e}"}

        # ── store compliance result back into the previous moment ────────
        # Use moment["id"] — NOT "moment_id" — to match memory3.py's schema
        t0 = time.time()
        if last_moment_id:
            self.memory.update_feedback(
                moment_id=last_moment_id,
                corrections={"compliance_check": compliance},
                confirmed=len(compliance.get("followed", [])) > 0
            )
        self.timings_phase_0["memory_update"] = time.time() - t0

        self.timings_phase_0["total_phase_0"] = time.time() - phase_start

        print(
            f"✅ Compliance result — "
            f"followed={compliance.get('followed')}  "
            f"not_followed={compliance.get('not_followed')}  "
            f"uncertain={compliance.get('uncertain')}"
        )
        print(f"📝 {compliance.get('summary', '')}")

        return {"skipped": False, "compliance": compliance}

    # -----------------------------------------------------------------------
    # ─── PHASE A helpers ─────────────────────────────────────────────────
    # -----------------------------------------------------------------------

    @timer
    def analyze_with_qwen(self, frame_path,
                          transcript_text: str, memory_context: str) -> str:
        """
        Part 1 — scene / people / action recognition
        Part 2 — need analysis
        Part 3 — solution generation
        All three are combined in a single LLM call to avoid latency.

        `frame_path` may be a single path, a list of paths, or None (wearable-only).
        """
        input_context = self._format_input_context()
        no_speech = "[No speech detected" in transcript_text
        ts_section = (
            "Transcript: [No audio transcript — infer from visual / sensor data only]"
            if no_speech else f"Transcript:\n{transcript_text}"
        )

        hints_block = self.hint_memory.format_for_prompt()
        hints_section = f"\n{hints_block}\n" if hints_block else ""

        has_video = frame_path is not None and (
            isinstance(frame_path, str) or len(frame_path) > 0
        )
        if has_video and self.input_mode == "raw_video":
            visual_note = (
                "You are given the raw video clip directly. Reason over the entire clip — "
                "motion, action progression, audio cues if present, and what changes over time."
            )
        elif has_video and isinstance(frame_path, (list, tuple)) and len(frame_path) > 1:
            visual_note = (
                f"You are given {len(frame_path)} ordered frames sampled uniformly across a short "
                "video clip. Reason over the whole sequence (motion, action progression, what "
                "changes between frames)."
            )
        elif has_video:
            visual_note = "You are given a single still frame. Base your answer on it."
        else:
            visual_note = "No video is available — base analysis on wearable sensor data only."

        prompt = f"""You are a proactive personal assistant with multi-modal perception.

{input_context}

{visual_note}

{memory_context}
{hints_section}
{ts_section}

Based on the input above, provide:

## PART 1 — Scene Recognition
- Location: [specific location or on-screen context]
- Time/Occasion: [time, festival, event …]
- People: [name1 (relationship), name2 (relationship), …]
- User Action: [what the user is doing — physical action or on-screen activity]

## PART 2 — Need Analysis
Identify the user's top 3 needs in priority order. Consider physical context, screen content,
and wearable signals (heart rate, fatigue, location) when available.
For each need:
- Need [N]: [description]  (Confidence: [0-1])

## PART 3 — Solutions
For each need, propose ONE concrete solution routed to the best output channel:
- Solution [N]: [brief summary]
  Output Type: digital_info | device_control | hardware_robot
  Action: [specific deliverable]

Output channel guide:
- digital_info    → AR overlay, notification, digital content, or in-app service
- device_control  → phone/PC operation (open app, tap, type, navigate UI, send message)
- hardware_robot  → smart-home device or robot command (turn on light, move robot arm, etc.)

Keep the response structured and concise."""

        frame_paths = [frame_path] if isinstance(frame_path, str) else (list(frame_path) if frame_path else [])
        if frame_paths or self.input_mode == "raw_video":
            resp = self._call_vision_api(frame_paths, prompt, timeout=180)
        else:
            resp = self._call_text_api(prompt, timeout=180)
        if resp.status_code == 200:
            return resp.json()["analysis"]
        raise Exception(f"API error: {resp.json().get('error', 'Unknown')}")

    @staticmethod
    def _normalize_output_type(raw: str) -> str:
        val = raw.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "digital": "digital_info", "digital_info": "digital_info",
            "info": "digital_info", "ar": "digital_info", "app_service": "digital_info",
            "device": "device_control", "device_control": "device_control",
            "phone": "device_control", "pc": "device_control", "computer": "device_control",
            "hardware": "hardware_robot", "hardware_robot": "hardware_robot",
            "robot": "hardware_robot", "smart_home": "hardware_robot",
        }
        return aliases.get(val, "digital_info")

    def _infer_output_type(self, need_text: str, solution_text: str, action_text: str = "") -> str:
        combined = f"{need_text} {solution_text} {action_text}".lower()
        device_kw = ("open app", "tap", "click", "type", "send message", "navigate",
                     "screenshot", "copy", "paste", "browser", "phone", "computer", "pc")
        hardware_kw = ("turn on", "turn off", "light", "thermostat", "robot",
                       "vacuum", "speaker", "lock door", "smart home", "arm", "move")
        if any(k in combined for k in hardware_kw):
            return "hardware_robot"
        if any(k in combined for k in device_kw):
            return "device_control"
        return "digital_info"

    def parse_analysis(self, text: str) -> dict:
        """Extract structured fields from the Part1-3 response."""
        result = {
            "people": [],
            "location": None,
            "user_action": "",
            "scene": "",
            "needs": [],
            "solutions": []
        }
        lines = text.split("\n")
        current_sol_idx = -1
        for line in lines:
            ll = line.lower().strip()
            if ll.startswith("- location:") or ll.startswith("location:"):
                result["location"] = line.split(":", 1)[-1].strip()
            elif ll.startswith("- people:") or ll.startswith("people:"):
                raw = line.split(":", 1)[-1].strip()
                raw_people = re.split(r"[,，、]", raw)
                result["people"] = [
                    re.sub(r"（[^）]*）|\([^)]*\)", "", p).strip()
                    for p in raw_people
                    if re.sub(r"（[^）]*）|\([^)]*\)", "", p).strip()
                    and re.sub(r"（[^）]*）|\([^)]*\)", "", p).strip().lower()
                    not in ("none", "n/a")
                ]
            elif ll.startswith("- user action:") or ll.startswith("user action:"):
                result["user_action"] = line.split(":", 1)[-1].strip()
            elif re.match(r"^-?\s*need\s*\[?\d", ll):
                body = line.split(":", 1)[-1].strip()
                conf_match = re.search(r"\(confidence:\s*([0-9.]+)\)", body, re.I)
                conf = float(conf_match.group(1)) if conf_match else 0.8
                need_text = re.sub(r"\(confidence:[^)]+\)", "", body, flags=re.I).strip()
                result["needs"].append({"need": need_text, "confidence": conf})
            elif re.match(r"^-?\s*solution\s*\[?\d", ll):
                body = line.split(":", 1)[-1].strip()
                result["solutions"].append({
                    "solution": body,
                    "output_type": "digital_info",
                    "action": body,
                    "type_explicit": False,
                })
                current_sol_idx = len(result["solutions"]) - 1
            elif re.match(r"^-?\s*output type:", ll) and current_sol_idx >= 0:
                raw_type = line.split(":", 1)[-1].strip()
                result["solutions"][current_sol_idx]["output_type"] = self._normalize_output_type(raw_type)
                result["solutions"][current_sol_idx]["type_explicit"] = True
            elif re.match(r"^-?\s*action:", ll) and current_sol_idx >= 0:
                result["solutions"][current_sol_idx]["action"] = line.split(":", 1)[-1].strip()

        result["scene"] = result.get("location", "")
        for i, sol in enumerate(result["solutions"]):
            if i < len(result["needs"]):
                sol["need"] = result["needs"][i]["need"]
            if not sol.get("type_explicit"):
                sol["output_type"] = self._infer_output_type(
                    sol.get("need", ""), sol.get("solution", ""), sol.get("action", "")
                )

        return result

    # -----------------------------------------------------------------------
    # ─── PHASE A — main Part1-3 pipeline ─────────────────────────────────
    # -----------------------------------------------------------------------

    def run_phase_a(self,
                    frame_path: str = None,
                    audio_path: str = None,
                    transcript_data: dict = None,
                    formatted_transcript: str = None) -> dict:
        """
        Runs the full Part1-3 pipeline.

        If frame_path / audio_path / transcript_data / formatted_transcript are
        provided (pre-extracted by process()), they are reused directly and the
        corresponding extraction/transcription steps are skipped so no work is
        duplicated between Phase 0 and Phase A.

        Returns everything needed for Phase B (Part4) and Phase C (consolidation).
        Populates self.timings_phase_a.
        """
        phase_start = time.time()

        print("\n" + "=" * 60)
        print("🚀 PHASE A — Part1-3 (extraction → analysis → memory add)")
        print("=" * 60)

        # ── parallel extraction (only if not already done) ───────────────
        if self.input_source == "wearable":
            frame_path = frame_path or []
            audio_path = audio_path or None
            print("⏭️  Wearable-only mode — skipping video/audio extraction.")
        elif frame_path is None or audio_path is None:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=2) as ex:
                ff = ex.submit(self._extract_visual)
                af = ex.submit(self.extract_audio)
                frame_path = ff.result()       # list of frame paths
                audio_path = af.result()
            self.timings_phase_a["parallel_extraction"] = time.time() - t0
            print(f"✅ Extraction: {self.timings_phase_a['parallel_extraction']:.2f}s")
        else:
            print("⏭️  Reusing pre-extracted frame(s) & audio from process().")

        # ── transcription (only if not already done) ─────────────────────
        if self.input_source == "wearable" and transcript_data is None:
            transcript_data = {"language": "unknown", "full_text": "", "segments": [], "no_speech": True}
            formatted_transcript = self._format_input_context()
            print("⏭️  Using wearable sensor context (no audio).")
        elif transcript_data is None:
            t0 = time.time()
            transcript_data = self.transcribe_audio(audio_path)
            self.timings_phase_a["transcribe"] = time.time() - t0
            formatted_transcript = self.format_transcript(transcript_data)
        else:
            print("⏭️  Reusing pre-transcribed data from process().")

        if not transcript_data.get("no_speech"):
            print(f"📝 Language: {transcript_data['language']}")
            print(formatted_transcript)

        # ── memory context retrieval ─────────────────────────────────────
        t0 = time.time()
        memory_context = self.memory.get_all_memory()
        self.timings_phase_a["memory_retrieval"] = time.time() - t0
        print(f"\n{memory_context}\n")

        # ── Part1-3 LLM analysis ─────────────────────────────────────────
        t0 = time.time()
        analysis_text = self.analyze_with_qwen(frame_path, formatted_transcript, memory_context)
        self.timings_phase_a["analyze_qwen"] = time.time() - t0
        print("\n" + "=" * 60)
        print("📊 PART1-3 ANALYSIS")
        print("=" * 60)
        print(analysis_text)

        # ── memory ADD (add, not consolidate) ────────────────────────────
        t0 = time.time()
        parsed = self.parse_analysis(analysis_text)
        moment_id = self.memory.add(
            scene=parsed["scene"],
            user_action=parsed["user_action"],
            needs=parsed["needs"],
            solutions=parsed["solutions"],
            people=parsed["people"],
            location=parsed["location"],
            extra_notes=transcript_data.get("language", "")
        )
        self.timings_phase_a["memory_add"] = time.time() - t0
        print(f"💾 Memory stored: moment_id={moment_id}  "
              f"people={parsed['people']}  location={parsed['location']}")

        self.timings_phase_a["total_phase_a"] = time.time() - phase_start

        return {
            "frame_path": frame_path,
            "audio_path": audio_path,
            "transcript": transcript_data,
            "formatted_transcript": formatted_transcript,
            "analysis_text": analysis_text,
            "parsed": parsed,
            "moment_id": moment_id,
        }

    # -----------------------------------------------------------------------
    # ─── PHASE B — Part4 (multi-channel output + feedback) ───────────────
    # -----------------------------------------------------------------------

    def run_phase_b(self, phase_a_result: dict) -> dict:
        """
        Part4: route Part3 solutions to output channels, execute delivery,
        then collect user feedback.

        Output channels:
          digital_info    — AR overlay, notifications, app services
          device_control  — phone / PC operations
          hardware_robot  — smart-home devices and robot commands
        """
        phase_start = time.time()

        print("\n" + "=" * 60)
        print("📤 PHASE B — Part4 (multi-channel output + user feedback)")
        print("=" * 60)

        parsed = phase_a_result["parsed"]
        moment_id = phase_a_result["moment_id"]

        t0 = time.time()
        output_plan = self._decide_output_plan(parsed)
        self.timings_phase_b["output_decision"] = time.time() - t0

        print("\n📺 Output Plan:")
        print(json.dumps(output_plan, indent=2, ensure_ascii=False))

        t0 = time.time()
        self._execute_outputs(output_plan, parsed)
        self.timings_phase_b["output_execution"] = time.time() - t0

        t0 = time.time()
        feedback = self._collect_feedback_simulated(parsed, moment_id)
        self.timings_phase_b["feedback_collection"] = time.time() - t0

        if feedback:
            self.memory.update_feedback(
                moment_id=moment_id,
                corrections=feedback.get("corrections"),
                confirmed=feedback.get("confirmed", False),
                user_rating=feedback.get("rating")
            )
            if feedback.get("confirmed"):
                self.memory.highlight(moment_id)
                print(f"⭐ Moment {moment_id} highlighted as confirmed-correct.")

        self.timings_phase_b["total_phase_b"] = time.time() - phase_start

        return {
            "output_plan": output_plan,
            "ar_plan": output_plan,  # backward compatibility
            "feedback": feedback,
            "moment_id": moment_id,
        }

    def _decide_output_plan(self, parsed: dict) -> dict:
        """
        Decide output channel, modality, interaction mode, and timing for each solution.
        """
        plan = {"solutions": []}
        for i, (need, sol) in enumerate(zip(parsed["needs"], parsed["solutions"])):
            conf = need.get("confidence", 0.8)
            need_text = need.get("need", "").lower()
            output_type = sol.get("output_type", "digital_info")
            action = sol.get("action", sol.get("solution", ""))

            if output_type == "device_control":
                modality = "device_ui"
                channel_label = "Phone/PC Operation"
            elif output_type == "hardware_robot":
                modality = "device_command"
                channel_label = "Hardware/Robot Command"
            elif any(k in need_text for k in ("navigate", "direction", "map", "route")):
                modality = "3D"
                channel_label = "Digital Info (AR 3D)"
            elif conf < 0.6:
                modality = "text"
                channel_label = "Digital Info (Text)"
            else:
                modality = "voice+text"
                channel_label = "Digital Info (Voice+Text)"

            if conf < 0.7:
                interaction = "need_confirmation"
            elif any(k in need_text for k in ("pay", "purchase", "buy", "order")):
                interaction = "solution_confirmation"
            elif output_type in ("device_control", "hardware_robot"):
                interaction = "solution_confirmation"
            else:
                interaction = "direct_display"

            if any(k in need_text for k in ("emergency", "urgent", "danger", "help")):
                timing = "immediate"
            elif interaction in ("need_confirmation", "solution_confirmation"):
                timing = "wait_interaction"
            else:
                timing = "immediate" if conf >= 0.8 else "delayed"

            plan["solutions"].append({
                "index": i + 1,
                "need": need.get("need", ""),
                "solution": sol.get("solution", ""),
                "action": action,
                "output_type": output_type,
                "channel_label": channel_label,
                "modality": modality,
                "interaction": interaction,
                "timing": timing,
                "confidence": conf,
            })
        return plan

    def _execute_outputs(self, output_plan: dict, parsed: dict):
        """
        Simulate delivery across all output channels.
        In production, each channel would call its respective device API.
        """
        channel_icons = {
            "digital_info": "📱💬",
            "device_control": "🖥️👆",
            "hardware_robot": "🤖🏠",
        }
        print("\n--- Multi-Channel Output (simulated) ---")
        for item in output_plan["solutions"]:
            icon = channel_icons.get(item["output_type"], "📱")
            timing_icon = "⚡" if item["timing"] == "immediate" else "⏳"
            interact_icon = "✅" if item["interaction"] == "direct_display" else "❓"
            print(
                f"  {icon}{timing_icon}{interact_icon} "
                f"[{item['channel_label']}] Need {item['index']}: {item['action'][:100]}"
            )
        print("--- End Output ---\n")

    # Backward-compatible aliases
    _decide_ar_presentation = _decide_output_plan
    _present_on_ar = _execute_outputs

    def _collect_feedback_simulated(self, parsed: dict, moment_id: str) -> dict:
        """
        Simulated feedback collection.
        In production this would listen for voice/gesture/tap input.
        Returns a feedback dict or None if no feedback.
        """
        # Auto-confirm if all needs have confidence >= 0.8
        all_high_conf = all(n.get("confidence", 0) >= 0.8 for n in parsed["needs"])
        if all_high_conf:
            print("✅ Auto-confirmed (all needs confidence ≥ 0.8)")
            return {"confirmed": True, "corrections": {}, "rating": 5}
        else:
            print("⚠️  Low-confidence needs — skipping auto-confirm (awaiting user input).")
            return {"confirmed": False, "corrections": {}, "rating": None}

    # -----------------------------------------------------------------------
    # ─── PHASE C — Post-Part4 consolidation ──────────────────────────────
    # -----------------------------------------------------------------------

    def run_phase_c(self, phase_b_result: dict, blocking: bool = False):
        """
        Memory consolidation: compress → sort → combine.
        MUST be called after Phase B (Part4 feedback) so corrections are stored.
        Can run in a background thread (blocking=False) to avoid delaying UI.
        """
        if blocking:
            self._consolidation_worker(phase_b_result)
        else:
            t = threading.Thread(
                target=self._consolidation_worker,
                args=(phase_b_result,),
                daemon=True
            )
            t.start()
            print("🔄 Phase C (consolidation) started in background thread.")
            return t

    def _should_consolidate(self) -> bool:
        total = self.memory.memory["metadata"]["total_moments"]
        last = self.memory.memory["metadata"].get("last_consolidation")
        return (total % 3 == 0) or (last is None and total > 1)

    def _consolidation_worker(self, phase_b_result: dict):
        """Background consolidation: compress + sort + combine."""
        if not self._should_consolidate():
            enc = self.memory.memory["metadata"]["total_moments"]
            last = self.memory.memory["metadata"].get("last_consolidation")
            print(f"⏭️  Skipping consolidation (moments={enc}, last={last})")
            return

        phase_start = time.time()
        print(f"\n🧹 PHASE C — Memory consolidation (background)")

        memory_json = json.dumps(self.memory.memory, indent=2, ensure_ascii=False)

        # Limit payload size to avoid overwhelming the LLM context
        if len(memory_json) > 8000:
            trimmed = {
                "layer1": self.memory.memory["layer1"],
                "layer2": self.memory.memory["layer2"][-5:],
                "layer4": self.memory.memory["layer4"],
                "layer5": self.memory.memory["layer5"],
                "layer6": self.memory.memory["layer6"],
                "layer7_indices": {
                    "people": self.memory.memory["layer7"]["people"],
                    "locations": self.memory.memory["layer7"]["locations"],
                    "activity_events": self.memory.memory["layer7"]["activity_events"],
                },
                "metadata": self.memory.memory["metadata"]
            }
            memory_json = json.dumps(trimmed, indent=2, ensure_ascii=False)

        consolidation_prompt = f"""You are a memory consolidation system. Analyse the current memory and:

CURRENT MEMORY:
{memory_json}

TASKS:
1. Compress layers 1-3 into updated summaries for layers 4, 5, 6.
2. Sort layer-7 indices: assign canonical tags to people, locations, activity_events.
3. Combine near-duplicate location / event names into canonical forms.
4. Extract and update user profile (name, preferences, habits) for layer 6.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{{
  "compress": {{
    "layer4": {{
      "summary": "...",
      "current_tasks": ["..."],
      "life_trajectory": "..."
    }},
    "layer5": {{
      "summary": "...",
      "key_events": ["..."],
      "long_term_patterns": "..."
    }},
    "layer6": {{
      "summary": "...",
      "profile": {{
        "name": "...",
        "basic_info": {{}},
        "preferences": {{}},
        "habits": {{}}
      }}
    }}
  }},
  "sort": {{
    "people": {{"canonical_name": ["moment_id_1", "moment_id_2"]}},
    "locations": {{"canonical_location": ["moment_id_1"]}},
    "activity_events": {{"event_tag": ["moment_id_1"]}}
  }},
  "combine": {{
    "locations": {{"old fuzzy name": "canonical name"}},
    "activity_events": {{"old tag": "canonical tag"}}
  }}
}}

IMPORTANT:
- Respond ONLY with valid JSON
- Be aggressive in merging near-duplicate location descriptions
- Layer 4 = recent/current (days); Layer 5 = long-term (months/years); Layer 6 = stable profile"""

        try:
            t0 = time.time()
            resp = requests.post(
                f"{self.qwen_api_url}/consolidate",
                json={"prompt": consolidation_prompt},
                timeout=90
            )
            self.timings_phase_c["llm_consolidation"] = time.time() - t0

            if resp.status_code != 200:
                print(f"❌ Consolidation API error: {resp.status_code}")
                return

            raw = resp.json()["analysis"]
            data = self._parse_json_safe(raw)

            if not data:
                print(f"⚠️  Could not parse consolidation JSON. Preview: {raw[:300]}")
                return

            # Apply operations
            t0 = time.time()
            if "compress" in data:
                self.memory.compress(data["compress"])
            if "sort" in data:
                self.memory.sort(data["sort"])
            if "combine" in data:
                self.memory.combine(data["combine"])
            self.timings_phase_c["apply_operations"] = time.time() - t0

            self.timings_phase_c["total_phase_c"] = time.time() - phase_start
            print(f"✅ Consolidation complete in {self.timings_phase_c['total_phase_c']:.2f}s")
            print(f"\n📋 Updated Memory:\n{self.memory.get_all_memory()}")

        except Exception as e:
            print(f"❌ Consolidation failed: {e}")
            import traceback
            traceback.print_exc()

    # -----------------------------------------------------------------------
    # ─── Hint memory convenience API ─────────────────────────────────────
    # -----------------------------------------------------------------------
    #
    # Hints are user-curated rules of the form
    #     "WHEN <trigger seen in scene>  THEN output need: <specific need>"
    # They are spliced into the Phase-A analysis prompt so the LLM applies
    # them automatically when the trigger matches. Edit memory/hints.json by
    # hand, or use these helpers from your driver script.

    def add_hint(self, when: str, then: str) -> str:
        """Add a new trigger→need rule. Returns the hint id."""
        hint_id = self.hint_memory.add(when, then)
        print(f"💡 Hint added [{hint_id}]: WHEN '{when}' THEN '{then}'")
        return hint_id

    def remove_hint(self, hint_id: str) -> bool:
        return self.hint_memory.remove(hint_id)

    def list_hints(self) -> list:
        return self.hint_memory.list()

    @staticmethod
    def _parse_json_safe(text: str) -> dict | None:
        """Try several strategies to extract JSON from LLM output."""
        try:
            return json.loads(text)
        except Exception:
            pass
        for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
        return None

    # -----------------------------------------------------------------------
    # ─── Main entry point ────────────────────────────────────────────────
    # -----------------------------------------------------------------------

    def process(self, consolidation_blocking: bool = False) -> dict:
        """
        Full pipeline:
          [shared] frame + audio extraction + transcription  (done ONCE, shared by Phase 0 & A)
          Phase 0 (compliance check)  → Phase A (Part1-3) → Phase B (Part4) → Phase C (background)

        consolidation_blocking=True will wait for Phase C before returning
        (useful for testing; in production keep it False).
        """
        wall_start = time.time()

        if self.input_source == "wearable":
            print("\n" + "=" * 60)
            print("📦 Wearable-only input (no video extraction)")
            print("=" * 60)
            frame_path = []
            audio_path = None
            transcript_data = {"language": "unknown", "full_text": "", "segments": [], "no_speech": True}
            formatted_transcript = self._format_input_context()
        else:
            # ── Shared extraction (done once, reused by Phase 0 and Phase A) ─
            print("\n" + "=" * 60)
            print("📦 Shared extraction (frame + audio + transcription)")
            print("=" * 60)
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=2) as ex:
                ff = ex.submit(self._extract_visual)
                af = ex.submit(self.extract_audio)
                frame_path = ff.result()
                audio_path = af.result()
            if self.input_mode == "raw_video":
                print(f"✅ Audio extracted in {time.time() - t0:.2f}s "
                      f"(visual = raw video, no client-side frames)")
            else:
                print(f"✅ Frame(s) & audio extracted in {time.time() - t0:.2f}s "
                      f"({len(frame_path)} frame{'s' if len(frame_path) != 1 else ''})")

            t0 = time.time()
            transcript_data = self.transcribe_audio(audio_path)
            formatted_transcript = self.format_transcript(transcript_data)
            if self.wearable_data:
                formatted_transcript += "\n\n" + self._format_wearable_section()
            print(f"✅ Transcription done in {time.time() - t0:.2f}s")

        # ── Phase 0: compliance check against last session's solutions ───
        phase_0 = self.run_phase_0(frame_path, formatted_transcript)

        # ── Phase A: Part1-3 analysis (reuses pre-extracted data) ────────
        phase_a = self.run_phase_a(
            frame_path=frame_path,
            audio_path=audio_path,
            transcript_data=transcript_data,
            formatted_transcript=formatted_transcript
        )

        # ── Phase B: multi-channel output + user feedback ─────────────────
        phase_b = self.run_phase_b(phase_a)

        # ── Phase C: consolidation (after feedback, never blocking Part1-4) ─
        consolidation_thread = self.run_phase_c(phase_b, blocking=consolidation_blocking)

        wall_elapsed = time.time() - wall_start
        self._print_timing_summary(wall_elapsed, consolidation_blocking)
        self._save_results(phase_a, phase_b, phase_0, wall_elapsed)

        return {
            "phase_0": phase_0,
            "phase_a": phase_a,
            "phase_b": phase_b,
            "consolidation_thread": consolidation_thread,
            "timings": {
                "phase_0": self.timings_phase_0,
                "phase_a": self.timings_phase_a,
                "phase_b": self.timings_phase_b,
                "phase_c": self.timings_phase_c,  # may be empty if background
            }
        }

    # -----------------------------------------------------------------------
    # ─── Helpers ─────────────────────────────────────────────────────────
    # -----------------------------------------------------------------------

    def _print_timing_summary(self, wall_elapsed: float, phase_c_included: bool):
        print("\n" + "=" * 60)
        print("⏱️  TIMING SUMMARY")
        print("=" * 60)

        print("\n── Phase 0 (Compliance Check) ──────────────────────────────")
        if self.timings_phase_0:
            for k, v in sorted(self.timings_phase_0.items(), key=lambda x: x[1], reverse=True):
                print(f"  {k:.<38} {v:>6.2f}s")
        else:
            print("  (skipped)")

        print("\n── Phase A (Part1-3) ──────────────────────────────────────")
        for k, v in sorted(self.timings_phase_a.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:.<38} {v:>6.2f}s")

        print("\n── Phase B (Part4 Output + Feedback) ────────────────────────")
        for k, v in sorted(self.timings_phase_b.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:.<38} {v:>6.2f}s")

        if self.timings_phase_c and phase_c_included:
            print("\n── Phase C (Consolidation) ─────────────────────────────────")
            for k, v in sorted(self.timings_phase_c.items(), key=lambda x: x[1], reverse=True):
                print(f"  {k:.<38} {v:>6.2f}s")
        elif not phase_c_included:
            print("\n── Phase C (Consolidation) ─── running in background thread ──")

        print(f"\n  {'wall_clock_total':.<38} {wall_elapsed:>6.2f}s")
        print("=" * 60 + "\n")

    def _save_results(self, phase_a: dict, phase_b: dict,
                      phase_0: dict, wall_elapsed: float):
        out = self.output_dir / "analysis_result.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"Device: {self.device}\n")
            if self.device == "cuda":
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")

            f.write(f"\n{'='*50}\nPHASE 0 — COMPLIANCE CHECK\n{'='*50}\n")
            if phase_0.get("skipped"):
                f.write(f"Skipped: {phase_0.get('reason')}\n")
            else:
                f.write(json.dumps(phase_0["compliance"], indent=2, ensure_ascii=False))

            f.write(f"\n\n{'='*50}\nPHASE A — PART1-3 ANALYSIS\n{'='*50}\n")
            f.write(phase_a["analysis_text"])

            f.write(f"\n\n{'='*50}\nPHASE B — OUTPUT PLAN\n{'='*50}\n")
            plan = phase_b.get("output_plan") or phase_b.get("ar_plan", {})
            f.write(json.dumps(plan, indent=2, ensure_ascii=False))

            f.write(f"\n\n{'='*50}\nFINAL MEMORY STATE\n{'='*50}\n")
            f.write(self.memory.get_all_memory())

            f.write(f"\n\n{'='*50}\nTIMINGS\n{'='*50}\n")
            f.write("Phase 0:\n")
            for k, v in self.timings_phase_0.items():
                f.write(f"  {k}: {v:.2f}s\n")
            f.write("Phase A:\n")
            for k, v in self.timings_phase_a.items():
                f.write(f"  {k}: {v:.2f}s\n")
            f.write("Phase B:\n")
            for k, v in self.timings_phase_b.items():
                f.write(f"  {k}: {v:.2f}s\n")
            f.write(f"wall_clock_total: {wall_elapsed:.2f}s\n")
        print(f"✅ Results saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Input examples ────────────────────────────────────────────────────
    # Camera (user-shot video):
    video_path = "../test_data/test_data/2.travel_abroad/2.1.mp4"
    assistant = VRAssistant(video_path, input_source="camera")

    # Screen recording:
    # assistant = VRAssistant("../screen_record.mp4", input_source="screen_record")

    # Wearable sensors only:
    # assistant = VRAssistant(
    #     input_source="wearable",
    #     wearable_data={"heart_rate": 110, "steps": 8200, "stress": "elevated"},
    # )

    # Multimodal (video + wearable):
    # assistant = VRAssistant(
    #     video_path,
    #     input_source="multimodal",
    #     wearable_data="memory/wearable_sample.json",
    # )

    # ── Input mode (frame / video / raw_video) ────────────────────────────
    # assistant = VRAssistant(video_path, input_mode="video", video_num_frames=4)

    # ── (Optional) seed the hint memory with extra trigger→need rules ────
    # Uncomment / add as many as you need. They are persisted to
    # memory/hints.json so they survive across runs.
    #
    # assistant.add_hint(
    #     when="a man wearing a suit is seated in the hotel lobby",
    #     then="remind the user that this is Richard, today's birthday guest",
    # )
    # assistant.add_hint(
    #     when="the user is about to eat greasy / fried food",
    #     then="remind the user to keep to their diet and pick a lighter option",
    # )

    # blocking=False  → Phase C runs in background (default, best for production)
    # blocking=True   → wait for consolidation before printing final timing
    results = assistant.process(consolidation_blocking=False)