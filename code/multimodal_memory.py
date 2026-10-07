"""Compact, content-addressed visual and audio evidence for episodic memory.

The seven-layer JSON keeps descriptions and references. Selected WebP frames
and short Opus snippets live once in an asset directory shared by moments.
Source video is never copied into memory storage.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_LIMIT = 12_000
AUDIO_LIMIT = 6_000
TEMPORAL_IMAGE_LIMIT = 6_000
TEMPORAL_AUDIO_BUDGET = 12_000
CAPTURE_VERSION = "temporal-v3-speech"


def media_refs(media: dict, kind: str) -> list[dict]:
    """Read temporal evidence and legacy single samples without duplicates."""
    plural = "visual_frames" if kind == "visual" else "audio_clips"
    refs = media.get(plural)
    refs = [*refs, media.get(kind)] if isinstance(refs, list) and refs else [media.get(kind)]
    result, seen = [], set()
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get("asset"):
            continue
        key = (ref["asset"], ref.get("at_s"))
        if key not in seen:
            result.append(ref)
            seen.add(key)
    return result


def _run_ffmpeg(args: list[str], input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *args],
        input=input_bytes, capture_output=True, timeout=30, check=False,
    )
    return proc.stdout if proc.returncode == 0 else b""


def _visual_hash(data: bytes) -> str:
    from PIL import Image, ImageOps

    image = ImageOps.grayscale(Image.open(io.BytesIO(data))).resize((9, 8))
    pixels = list(image.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def _audio_signature(data: bytes) -> list[int]:
    """Sixteen broad spectral bands; a compact comparison cue, not ASR."""
    import numpy as np

    pcm = _run_ffmpeg(["-i", "pipe:0", "-ac", "1", "-ar", "8000",
                       "-f", "f32le", "pipe:1"], input_bytes=data)
    samples = np.frombuffer(pcm, dtype="<f4")
    if len(samples) < 2048:
        return []
    bands = np.zeros(16, dtype=np.float64)
    count = 0
    edges = np.linspace(1, 513, 17, dtype=int)
    for offset in range(0, len(samples) - 2048, 1024):
        power = np.abs(np.fft.rfft(samples[offset:offset + 2048] * np.hanning(2048)))
        bands += np.array([power[a:b].mean() for a, b in zip(
            edges[:-1], edges[1:])])
        count += 1
    if not count or not bands.max():
        return []
    bands = np.log1p(bands / count)
    bands /= max(float(np.linalg.norm(bands)), 1e-9)
    return [int(round(float(x) * 127)) for x in bands]


def _speaker_names(transcript: str) -> list[str]:
    names = []
    for line in transcript.splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z -]{0,29}):", line)
        if match:
            name = match.group(1).strip()
            if name not in names:
                names.append(name)
    return names[:12]


def _portable_video_id(value: Any) -> str:
    """Keep a dataset-relative source identifier, never a host absolute path."""
    normalized = str(value or "").replace("\\", "/")
    absolute = normalized.startswith("/")
    parts = [part for part in normalized.strip("/").split("/")
             if part and part not in {".", ".."}]
    absolute = absolute or bool(parts and parts[0].endswith(":"))
    marker = next((i for i, part in enumerate(parts)
                   if part.casefold() == "egolife"), None)
    if marker is not None:
        parts = parts[marker + 1:]
    elif absolute:
        parts = parts[-1:]
    return "/".join(parts)[:512]


class MultimodalAssetStore:
    """Store selected evidence beside a PersonMemory database."""

    def __init__(self, memory_dir: str | Path):
        self.root = Path(memory_dir).resolve()
        self.asset_root = self.root / "assets"
        self.asset_budget = int(float(os.environ.get("MEMORY_ASSET_BUDGET_GB", "80"))
                                * (1024 ** 3))
        self.total_budget = int(os.environ.get("MEMORY_TOTAL_BUDGET_BYTES", "100000000000"))
        self.non_asset_reserve = int(os.environ.get("MEMORY_NON_ASSET_RESERVE_BYTES", "2000000000"))
        self._bytes_cached: int | None = None

    def _put(self, kind: str, data: bytes, suffix: str, limit: int) -> dict | None:
        if not data or len(data) > limit:
            return None
        digest = hashlib.sha256(data).hexdigest()[:24]
        relative = Path("assets") / kind / f"{digest}.{suffix}"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            memory_file = self.root / "memory.json"
            text_bytes = memory_file.stat().st_size if memory_file.is_file() else 0
            effective_budget = min(self.asset_budget, max(0, self.total_budget - self.non_asset_reserve - text_bytes))
            if self.size_bytes() + len(data) > effective_budget:
                return None
            tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            self._bytes_cached = (self._bytes_cached or 0) + len(data)
        return {"asset": relative.as_posix(), "bytes": len(data)}

    def capture(self, video: str | Path, start_s: float, end_s: float | None,
                transcript: str = "", source_id: str = "",
                fields: dict | None = None,
                recent_media: list[dict] | None = None,
                source_provenance: dict | None = None) -> dict[str, Any]:
        """Retain three ordered instants and bounded speech excerpts.

        The structured Qwen scene/action/entities remain in the parent moment.
        Missing media only removes that modality; it never erases text memory.
        """
        fields = fields or {}
        start = max(0.0, float(start_s or 0))
        end = float(end_s) if end_s is not None else start + 30
        result: dict[str, Any] = {
            "schema": 1,
            "capture_version": CAPTURE_VERSION,
            "source_id": str(source_id),
            "start_s": round(start, 2),
            "end_s": round(end, 2),
            "entities": {
                "people": fields.get("people") or [],
                "objects": fields.get("objects") or [],
                "environments": fields.get("environments") or [],
                "activity": fields.get("activity") or "",
                "location": fields.get("location") or "",
                "speakers": _speaker_names(transcript),
            },
        }
        result["audio_language_hint"] = "zh" if re.search(r"[\u3400-\u9fff]", transcript) else None
        if isinstance(source_provenance, dict):
            saved_provenance = {
                key: source_provenance[key]
                for key in ("source_video_id", "source_segment_id", "clip_start_s",
                            "clip_end_s", "extraction_batch_id")
                if source_provenance.get(key) not in (None, "")
            }
            if saved_provenance.get("source_video_id"):
                saved_provenance["source_video_id"] = _portable_video_id(
                    saved_provenance["source_video_id"])
            result["source_provenance"] = saved_provenance
        path = Path(video)
        if not path.is_file():
            result["capture_status"] = "source_missing"
            return result
        try:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(path)], capture_output=True, timeout=10, check=True)
            duration = float(json.loads(probe.stdout)["format"]["duration"])
            end = min(end, duration)
            result["source_duration_s"] = round(duration, 3)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            result["capture_status"] = "source_duration_unverified"
            return result
        if end <= start:
            result["capture_status"] = "invalid_interval"
            return result
        result["end_s"] = round(end, 3)
        result.setdefault("source_provenance", {}).setdefault(
            "source_video_id", _portable_video_id(path))
        result["source_provenance"].update({"clip_start_s": round(start, 3),
                                            "clip_end_s": round(end, 3)})
        # Near-identical pictures remain separate observations. Content-addressed
        # _put shares only identical encoded bytes; dHash is a retrieval cue.
        last_time = max(start, end - 0.1)
        times = sorted(set(round(t, 3) for t in (
            start, (start + last_time) / 2, last_time)))
        frames = []
        for sample_time in times:
            for width, quality in ((288, 42), (224, 32), (176, 25)):
                image = _run_ffmpeg([
                    "-ss", f"{sample_time:.3f}", "-i", str(path), "-frames:v", "1",
                    "-vf", f"scale={width}:-2", "-c:v", "libwebp",
                    "-quality", str(quality), "-f", "webp", "pipe:1",
                ])
                try:
                    image_hash = _visual_hash(image) if image else ""
                except Exception:
                    continue
                if not image_hash:
                    continue
                saved = self._put("visual", image, "webp", TEMPORAL_IMAGE_LIMIT)
                if saved:
                    saved.update({"mime": "image/webp", "dhash": image_hash,
                                  "at_s": sample_time, "source_id": str(source_id)})
                    frames.append(saved)
                    break
        if frames:
            result["visual_frames"] = frames
            result["visual"] = dict(frames[len(frames) // 2])

        # Up to ten seconds of speech. For long clips, sample start/middle/end
        # so a short instruction in the middle is not systematically omitted.
        # The three windows are deliberately bounded and their coverage is saved.
        if end - start <= 10:
            spans = [(start, min(end - start, 10.0))]
        else:
            window = min(10.0 / 3.0, (end - start) / 3.0)
            midpoint = (start + end) / 2.0
            spans = [
                (start, window),
                (max(start, midpoint - window / 2.0), window),
                (max(start, end - window), window),
            ]
        clips, audio_bytes = [], 0
        for audio_start, audio_duration in (spans if transcript.strip() else []):
            audio = _run_ffmpeg([
                "-ss", f"{audio_start:.3f}", "-i", str(path), "-t", f"{audio_duration:.3f}",
                "-vn", "-ac", "1", "-ar", "16000", "-af", "loudnorm=I=-23:TP=-2:LRA=7", "-c:a", "libopus",
                "-b:a", "6k", "-vbr", "off", "-f", "ogg", "pipe:1",
            ])
            saved = self._put("audio", audio, "ogg", TEMPORAL_AUDIO_BUDGET - audio_bytes)
            if saved:
                try:
                    signature = _audio_signature(audio)
                except Exception:
                    signature = []
                saved.update({"mime": "audio/ogg", "signature": signature,
                              "at_s": round(audio_start, 3),
                              "duration_s": round(audio_duration, 3),
                              "source_id": str(source_id)})
                clips.append(saved)
                audio_bytes += len(audio)
        if clips:
            result["audio_clips"] = clips
            result["audio"] = dict(clips[0])
        result["audio_coverage_s"] = round(sum(x["duration_s"] for x in clips), 3)
        result["audio_coverage_ratio"] = round(result["audio_coverage_s"] / (end - start), 4)
        result["audio_sampling"] = "whole_event" if end - start <= 10 else "three_point_start_middle_end"
        result["capture_status"] = "captured" if frames or clips else "no_decodable_media"
        return result

    def size_bytes(self) -> int:
        if self._bytes_cached is None:
            self._bytes_cached = (sum(p.stat().st_size for p in self.asset_root.rglob("*")
                                      if p.is_file()) if self.asset_root.exists() else 0)
        return self._bytes_cached

    def release_unreferenced(self, moments: dict) -> list[str]:
        """Remove only owned assets no longer referenced by any stored moment."""
        referenced = set()
        for moment in moments.values():
            for kind in ("visual", "audio"):
                for ref in media_refs(moment.get("multimodal") or {}, kind):
                    referenced.add(ref["asset"])
        removed = []
        for kind in ("visual", "audio"):
            folder = self.asset_root / kind
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if path.is_file() and path.relative_to(self.root).as_posix() not in referenced:
                    removed.append(path.relative_to(self.root).as_posix())
                    self._bytes_cached = max(0, self.size_bytes() - path.stat().st_size)
                    path.unlink()
        return removed

    def compact_to_budget(self, moments: dict, budget_bytes: int | None = None,
                          protected_recent: int = 20) -> dict:
        """Evict old low-priority media when over budget; retain text facts.

        Highlighted and the newest window remain protected. Shared assets are
        only removed once their last moment reference has been retired.
        """
        memory_file = self.root / "memory.json"
        text_bytes = memory_file.stat().st_size if memory_file.is_file() else 0
        effective_budget = min(self.asset_budget, max(0, self.total_budget - self.non_asset_reserve - text_bytes))
        budget = effective_budget if budget_bytes is None else min(effective_budget, max(0, budget_bytes))
        before = self.size_bytes()
        if before <= budget:
            return {"before_bytes": before, "after_bytes": before,
                    "evicted_moments": 0, "removed_assets": 0}
        ordered = sorted(moments.values(), key=lambda m: str(m.get("timestamp") or ""))
        protected_count = max(0, int(protected_recent))
        protected_ids = {m.get("id") for m in ordered[-protected_count:]} if protected_count else set()
        candidates = [m for m in ordered if m.get("id") not in protected_ids
                      and not m.get("highlighted")]
        candidates.sort(key=lambda m: (
            int(m.get("access_count") or 0), str(m.get("last_used") or ""),
            str(m.get("timestamp") or "")))
        evicted = 0
        removed_assets = 0
        for moment in candidates:
            mm = moment.get("multimodal") or {}
            if not (mm.get("visual") or mm.get("audio")):
                continue
            mm.pop("audio", None)
            mm.pop("visual", None)
            mm.pop("audio_clips", None)
            mm.pop("visual_frames", None)
            mm["evicted_at"] = datetime.now().isoformat()
            evicted += 1
            removed_assets += len(self.release_unreferenced(moments))
            if self.size_bytes() <= budget:
                break
        return {"before_bytes": before, "after_bytes": self.size_bytes(),
                "evicted_moments": evicted, "removed_assets": removed_assets,
                "target_met": self.size_bytes() <= budget}


def compare_multimodal(left: dict, right: dict) -> dict[str, Any]:
    """Transparent pairwise similarity; missing modalities are not scored."""
    a, b = (left.get("multimodal") or left), (right.get("multimodal") or right)
    scores = {}
    try:
        h1, h2 = a["visual"]["dhash"], b["visual"]["dhash"]
        scores["visual"] = round(1 - (int(h1, 16) ^ int(h2, 16)).bit_count() / 64, 4)
    except (KeyError, TypeError, ValueError):
        pass
    try:
        x, y = a["audio"]["signature"], b["audio"]["signature"]
        if len(x) == len(y) == 16:
            dot = sum(i * j for i, j in zip(x, y))
            norm = math.sqrt(sum(i * i for i in x) * sum(j * j for j in y))
            if norm:
                scores["audio"] = round(max(0.0, dot / norm), 4)
    except (KeyError, TypeError, ValueError):
        pass
    e1, e2 = a.get("entities") or {}, b.get("entities") or {}
    tags1 = {str(v).lower() for key in ("people", "objects", "environments")
             for v in (e1.get(key) or [])}
    tags2 = {str(v).lower() for key in ("people", "objects", "environments")
             for v in (e2.get(key) or [])}
    for e in (e1, e2):
        if e.get("activity"):
            (tags1 if e is e1 else tags2).add(str(e["activity"]).lower())
        if e.get("location"):
            (tags1 if e is e1 else tags2).add(str(e["location"]).lower())
    if tags1 and tags2:
        scores["entities"] = round(len(tags1 & tags2) / len(tags1 | tags2), 4)
    weights = {"visual": 0.5, "audio": 0.25, "entities": 0.25}
    total = sum(weights[key] for key in scores)
    return {"score": round(sum(scores[key] * weights[key] for key in scores) / total, 4)
            if total else None, "modalities": scores}
