"""Memory-owned speech decoding; does not change the shared perception API."""
from __future__ import annotations

import os
import threading
from pathlib import Path

_MODEL = None
_LOCK = threading.Lock()


def transcribe_memory_audio(path: Path, language_hint: str | None = None) -> dict:
    """Decode stored evidence using an explicit language and VAD.

    Models are loaded only from a provisioned local directory; no downloads or
    credentials are accessed during an interactive memory operation.
    """
    global _MODEL
    configured = os.environ.get("MEMORY_WHISPER_MODEL_PATH")
    model_path = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data/models/memory-whisper-small"
    if not (model_path / "model.bin").is_file():
        raise FileNotFoundError("Memory speech model has not been provisioned")
    with _LOCK:
        if _MODEL is None:
            from faster_whisper import WhisperModel
            _MODEL = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=4)
        segments, info = _MODEL.transcribe(
            str(path), language=language_hint if language_hint in {"zh", "en"} else None,
            beam_size=5, vad_filter=True, condition_on_previous_text=False,
            vad_parameters={"min_silence_duration_ms": 350, "speech_pad_ms": 200},
        )
        rows = [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                 "avg_logprob": round(s.avg_logprob, 4), "no_speech_prob": round(s.no_speech_prob, 4)}
                for s in segments]
    supported = [row for row in rows if row["avg_logprob"] >= -1.0 and row["no_speech_prob"] < 0.6]
    return {"full_text": " ".join(row["text"] for row in supported), "segments": rows,
            "raw_full_text": " ".join(row["text"] for row in rows),
            "language": info.language, "engine": "memory-whisper-small-cpu-int8",
            "language_hint": language_hint, "low_confidence_segments": len(rows) - len(supported)}
