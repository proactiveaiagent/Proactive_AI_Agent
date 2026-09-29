"""Compact, content-addressed visual and audio evidence for episodic memory.

The seven-layer JSON keeps descriptions and references. Selected WebP frames
and short Opus snippets live once in an asset directory shared by moments.
Source video is never copied into memory storage.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_LIMIT = 12_000
AUDIO_LIMIT = 6_000


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


class MultimodalAssetStore:
    """Store selected evidence beside a PersonMemory database."""

    def __init__(self, memory_dir: str | Path):
        self.root = Path(memory_dir).resolve()
        self.asset_root = self.root / "assets"
        self.asset_budget = int(float(os.environ.get("MEMORY_ASSET_BUDGET_GB", "80"))
                                * (1024 ** 3))
        self._bytes_cached: int | None = None

    def _put(self, kind: str, data: bytes, suffix: str, limit: int) -> dict | None:
        if not data or len(data) > limit:
            return None
        digest = hashlib.sha256(data).hexdigest()[:24]
        relative = Path("assets") / kind / f"{digest}.{suffix}"
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if self.size_bytes() + len(data) > self.asset_budget:
                return None
            tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            self._bytes_cached = (self._bytes_cached or 0) + len(data)
        return {"asset": relative.as_posix(), "bytes": len(data)}

    def capture(self, video: str | Path, start_s: float, end_s: float | None,
                transcript: str = "", source_id: str = "",
                fields: dict | None = None,
                recent_media: list[dict] | None = None) -> dict[str, Any]:
        """Attach one bounded visual frame and, when speech exists, short audio.

        The structured Qwen scene/action/entities remain in the parent moment.
        Missing media only removes that modality; it never erases text memory.
        """
        fields = fields or {}
        start = max(0.0, float(start_s or 0))
        end = max(start + 0.1, float(end_s or start + 30))
        sample_time = start + min((end - start) / 2, 20.0)
        result: dict[str, Any] = {
            "schema": 1,
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
        path = Path(video)
        if not path.is_file():
            return result

        for width, quality in ((288, 48), (224, 35), (176, 28)):
            image = _run_ffmpeg([
                "-ss", f"{sample_time:.3f}", "-i", str(path), "-frames:v", "1",
                "-vf", f"scale={width}:-2", "-c:v", "libwebp",
                "-quality", str(quality), "-f", "webp", "pipe:1",
            ])
            try:
                image_hash = _visual_hash(image) if image else ""
            except Exception:
                # Some truncated videos make ffmpeg return non-decodable bytes.
                continue
            if not image_hash:
                continue
            reused = None
            for previous in (recent_media or [])[-100:]:
                ref = (previous or {}).get("visual") or {}
                try:
                    if image_hash and (int(image_hash, 16) ^ int(ref["dhash"], 16)).bit_count() <= 4:
                        reused = dict(ref)
                        break
                except (KeyError, TypeError, ValueError):
                    continue
            if reused:
                reused["at_s"] = round(sample_time, 2)
                result["visual"] = reused
                break
            saved = self._put("visual", image, "webp", IMAGE_LIMIT)
            if saved:
                saved.update({"mime": "image/webp", "dhash": image_hash,
                              "at_s": round(sample_time, 2)})
                result["visual"] = saved
                break

        if transcript.strip():
            audio_start = max(start, sample_time - 1.0)
            audio = _run_ffmpeg([
                "-ss", f"{audio_start:.3f}", "-i", str(path), "-t", "2",
                "-vn", "-ac", "1", "-ar", "12000", "-c:a", "libopus",
                "-b:a", "12k", "-vbr", "off", "-f", "ogg", "pipe:1",
            ])
            saved = self._put("audio", audio, "ogg", AUDIO_LIMIT)
            if saved:
                try:
                    signature = _audio_signature(audio)
                except Exception:
                    signature = []
                saved.update({"mime": "audio/ogg", "signature": signature,
                              "at_s": round(audio_start, 2), "duration_s": 2})
                result["audio"] = saved
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
                ref = (moment.get("multimodal") or {}).get(kind) or {}
                if isinstance(ref, dict) and ref.get("asset"):
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
        budget = self.asset_budget if budget_bytes is None else max(0, budget_bytes)
        before = self.size_bytes()
        if before <= budget:
            return {"before_bytes": before, "after_bytes": before,
                    "evicted_moments": 0, "removed_assets": 0}
        ordered = sorted(moments.values(), key=lambda m: str(m.get("timestamp") or ""))
        protected_ids = {m.get("id") for m in ordered[-protected_recent:]}
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
