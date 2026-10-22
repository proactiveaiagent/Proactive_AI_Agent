"""Compact, content-addressed visual evidence for episodic memory.

The seven-layer JSON keeps descriptions and references. Selected WebP frames
live once in an asset directory shared by moments. Source video is never
copied into memory storage; spoken content is retained only as text.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


IMAGE_LIMIT = 12_000
TEMPORAL_IMAGE_LIMIT = 6_000
CAPTURE_VERSION = "temporal-v10-qwen-six-input-frames-2026-10-20"


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


def strip_audio_assets(node: Any) -> bool:
    """Remove legacy raw-audio references while preserving all text memory."""
    changed = False
    if isinstance(node, dict):
        media = node.get("multimodal")
        if isinstance(media, dict):
            for key in ("audio", "audio_clips", "audio_segments", "audio_coverage_s",
                        "audio_coverage_ratio", "audio_sampling"):
                if key in media:
                    media.pop(key, None)
                    changed = True
        for value in node.values():
            changed = strip_audio_assets(value) or changed
    elif isinstance(node, list):
        for value in node:
            changed = strip_audio_assets(value) or changed
    return changed


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

    def _put(self, kind: str, data: bytes, suffix: str, limit: int,
             allow_over_budget: bool = False) -> dict | None:
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
            if self.size_bytes() + len(data) > effective_budget and not allow_over_budget:
                return None
            tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            self._bytes_cached = (self._bytes_cached or 0) + len(data)
        return {"asset": relative.as_posix(), "bytes": len(data)}

    def sample_visual_candidates(self, video: str | Path, start_s: float,
                                 end_s: float | None, source_id: str = "",
                                 count: int = 7,
                                 full_resolution: bool = False,
                                 sample_times: list[float] | None = None) -> list[dict[str, Any]]:
        """Sample transient candidates; callers persist only the selected frames."""
        path = Path(video)
        if not path.is_file():
            return []
        start = max(0.0, float(start_s or 0))
        end = float(end_s) if end_s is not None else start + 30
        try:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", str(path)], capture_output=True, timeout=10, check=True)
            duration = float(json.loads(probe.stdout)["format"]["duration"])
            end = min(end, duration)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return []
        if end <= start:
            return []
        last_time = max(start, end - 0.1)
        if sample_times:
            times = sorted(set(round(float(value), 3) for value in sample_times
                               if start <= float(value) <= last_time))
        else:
            n = max(3, min(7, int(count or 7)))
            times = sorted(set(round(start + (last_time - start) * i / (n - 1), 3)
                               for i in range(n)))
        frames = []
        for sample_time in times:
            if full_resolution:
                image = _run_ffmpeg([
                    "-ss", f"{sample_time:.3f}", "-i", str(path), "-frames:v", "1",
                    "-q:v", "2", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
                ])
                mime = "image/jpeg"
                max_bytes = 16_000_000
            else:
                image = _run_ffmpeg([
                    "-ss", f"{sample_time:.3f}", "-i", str(path), "-frames:v", "1",
                    "-vf", "scale=512:-2", "-c:v", "libwebp",
                    "-quality", "55", "-f", "webp", "pipe:1",
                ])
                mime = "image/webp"
                max_bytes = 32_000
            if not image or len(image) > max_bytes:
                continue
            try:
                image_hash = _visual_hash(image)
            except Exception:
                continue
            if not image_hash:
                continue
            frames.append({"bytes": image, "mime": mime,
                           "dhash": image_hash, "at_s": sample_time,
                           "source_id": str(source_id),
                           "source_resolution": bool(full_resolution)})
        return frames

    @staticmethod
    def fit_visual_candidate(data: bytes, limit: int = TEMPORAL_IMAGE_LIMIT) -> bytes | None:
        """Keep high-resolution candidates transient; persist the clearest <=6KB version."""
        try:
            from PIL import Image
            source = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            return None
        for width in (512, 448, 384, 320, 288):
            height = max(2, round(source.height * width / source.width))
            image = source.resize((width, height), Image.Resampling.LANCZOS)
            for quality in (48, 42, 36, 30, 24, 18):
                output = io.BytesIO()
                image.save(output, format="WEBP", quality=quality, method=6)
                encoded = output.getvalue()
                if len(encoded) <= limit:
                    return encoded
        return None

    @staticmethod
    def crop_visual_candidate(data: bytes, bbox: list | tuple,
                              limit: int = 12_000) -> bytes | None:
        """Create a padded, normalized-bbox crop for one grounded object."""
        try:
            from PIL import Image, ImageOps
            values = [float(value) for value in bbox]
            if len(values) != 4 or not all(math.isfinite(value) for value in values):
                return None
            x1, y1, x2, y2 = [min(1.0, max(0.0, value)) for value in values]
            if x2 - x1 < 0.015 or y2 - y1 < 0.015:
                return None
            source = ImageOps.exif_transpose(
                Image.open(io.BytesIO(data))).convert("RGB")
            width, height = source.size
            left, top = x1 * width, y1 * height
            right, bottom = x2 * width, y2 * height
            box_w, box_h = right - left, bottom - top
            pad_x = max(box_w * 0.22, width * 0.025)
            pad_y = max(box_h * 0.22, height * 0.025)
            left, right = max(0, left - pad_x), min(width, right + pad_x)
            top, bottom = max(0, top - pad_y), min(height, bottom + pad_y)
            # Preserve enough surrounding context for small objects such as glasses.
            min_w, min_h = min(width, max(64, round(width * 0.14))), min(height, max(64, round(height * 0.14)))
            if right - left < min_w:
                center = (left + right) / 2
                left, right = max(0, center - min_w / 2), min(width, center + min_w / 2)
                if right - left < min_w:
                    left = max(0, right - min_w)
            if bottom - top < min_h:
                center = (top + bottom) / 2
                top, bottom = max(0, center - min_h / 2), min(height, center + min_h / 2)
                if bottom - top < min_h:
                    top = max(0, bottom - min_h)
            # Grounders occasionally return an image-sized box for broad objects
            # (walls, doors, floors). Keep the resulting attachment a real local
            # view instead of silently persisting the whole frame as a "crop".
            max_w = max(2, round(width * 0.78))
            max_h = max(2, round(height * 0.78))
            if right - left > max_w:
                center = (left + right) / 2
                left = max(0, center - max_w / 2)
                right = min(width, left + max_w)
                left = max(0, right - max_w)
            if bottom - top > max_h:
                center = (top + bottom) / 2
                top = max(0, center - max_h / 2)
                bottom = min(height, top + max_h)
                top = max(0, bottom - max_h)
            crop = source.crop((round(left), round(top), round(right), round(bottom)))
            for target_width in (512, 448, 384, 320, 256):
                if crop.width > target_width:
                    target_height = max(2, round(crop.height * target_width / crop.width))
                    candidate = crop.resize((target_width, target_height), Image.Resampling.LANCZOS)
                else:
                    candidate = crop
                for quality in (82, 74, 66, 58, 50, 42, 34):
                    output = io.BytesIO()
                    candidate.save(output, format="WEBP", quality=quality, method=6)
                    encoded = output.getvalue()
                    if len(encoded) <= limit:
                        return encoded
        except Exception:
            return None
        return None

    @staticmethod
    def crop_visual_original(data: bytes, bbox: list | tuple,
                             face: bool = False) -> bytes | None:
        """Crop a source-resolution person/face region and save high-quality JPEG."""
        try:
            from PIL import Image, ImageOps
            values = [float(value) for value in bbox]
            if len(values) != 4 or not all(math.isfinite(value) for value in values):
                return None
            x1, y1, x2, y2 = [min(1.0, max(0.0, value)) for value in values]
            if x2 <= x1 or y2 <= y1:
                return None
            source = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            width, height = source.size
            left, top, right, bottom = x1 * width, y1 * height, x2 * width, y2 * height
            pad_x = (right - left) * (0.38 if face else 0.16)
            pad_y = (bottom - top) * (0.48 if face else 0.12)
            box = (max(0, round(left-pad_x)), max(0, round(top-pad_y)),
                   min(width, round(right+pad_x)), min(height, round(bottom+pad_y)))
            output = io.BytesIO()
            source.crop(box).save(output, format="JPEG", quality=96,
                                  subsampling=0, optimize=True)
            return output.getvalue()
        except Exception:
            return None

    def capture(self, video: str | Path, start_s: float, end_s: float | None,
                source_id: str = "",
                fields: dict | None = None,
                recent_media: list[dict] | None = None,
                source_provenance: dict | None = None,
                qwen_input_frames: list[dict] | None = None) -> dict[str, Any]:
        """Retain the actual Qwen input frames as scene-checkback evidence.

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
                "speakers": [],
            },
        }
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
            probe_data = json.loads(probe.stdout)
            duration = float(probe_data["format"]["duration"])
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
        # These are the JPEG payloads submitted to Qwen, not another set of
        # independently sampled/downsized scene thumbnails. Store each instant's
        # timestamp and ordinal even when content-addressing shares identical bytes.
        frames = []
        for index, item in enumerate(qwen_input_frames or [], start=1):
            if not isinstance(item, dict):
                continue
            image = item.get("bytes")
            if not isinstance(image, bytes) or not image:
                continue
            try:
                sample_time = round(float(item.get("at_s")), 3)
            except (TypeError, ValueError):
                continue
            saved = self._put("visual", image, "jpg", 8 * 1024 * 1024)
            if not saved:
                continue
            saved.update({
                "mime": "image/jpeg", "at_s": sample_time,
                "source_id": str(source_id), "frame_index": index,
                "visual_role": "qwen_input_frame", "quality_tier": "recent",
            })
            frames.append(saved)
        if frames:
            result["visual_quality_tier"] = "recent"
            result["visual_frame_source"] = "qwen_input"
            result["qwen_input_frame_count"] = len(qwen_input_frames or [])
            result["visual_frames"] = frames
            result["visual"] = dict(frames[len(frames) // 2])
        else:
            # Compatibility fallback for callers without a Qwen result. These
            # samples remain explicitly distinct from Qwen inputs in the UI.
            last_time = max(start, end - 0.1)
            times = sorted(set(round(t, 3) for t in (
                start, (start + last_time) / 2, last_time)))
            for sample_time in times:
                for width, quality in ((384, 58), (320, 48), (288, 42), (224, 32), (176, 25)):
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
                                      "at_s": sample_time, "source_id": str(source_id),
                                      "visual_role": "source_sample"})
                        frames.append(saved)
                        break
            if frames:
                for frame in frames:
                    frame["quality_tier"] = "recent"
                result["visual_quality_tier"] = "recent"
                result["visual_frame_source"] = "source_sample_fallback"
                result["visual_frames"] = frames
                result["visual"] = dict(frames[len(frames) // 2])

        result["capture_status"] = "captured" if frames else "no_decodable_media"
        return result

    def capture_context_entities(self, frames: list[dict], fields: dict | None = None,
                                 source_id: str = "", clip_start_s: float = 0.0) -> dict[str, Any]:
        """Deprecated compatibility shim: context facts remain text, not image entities.

        Dedicated visual evidence is stored by the person/object grounding pipelines.
        Scene, location, environment, activity, and event facts retain their textual
        content and source timestamps on the memory record itself.
        """
        return {"schema": 2, "status": "not_applicable",
                "method": "textual context with source provenance",
                "assets": [], "items": []}

    def size_bytes(self) -> int:
        if self._bytes_cached is None:
            self._bytes_cached = (sum(p.stat().st_size for p in self.asset_root.rglob("*")
                                      if p.is_file()) if self.asset_root.exists() else 0)
        return self._bytes_cached

    def release_unreferenced(self, moments: dict) -> list[str]:
        """Remove only owned assets no longer referenced by any stored moment."""
        referenced = set()
        for moment in moments.values():
            media = moment.get("multimodal") or {}
            for kind in ("visual", "audio"):
                for ref in media_refs(media, kind):
                    referenced.add(ref["asset"])
            for item in (media.get("object_visuals") or {}).get("items") or []:
                for ref in item.get("frame_refs") or []:
                    if ref.get("asset"):
                        referenced.add(ref["asset"])
                    if ref.get("source_asset"):
                        referenced.add(ref["source_asset"])
            for item in (media.get("person_visuals") or {}).get("items") or []:
                for ref in item.get("frame_refs") or []:
                    if ref.get("asset"):
                        referenced.add(ref["asset"])
                    if ref.get("source_asset"):
                        referenced.add(ref["source_asset"])
            entity_visuals = media.get("entity_visuals") or {}
            for ref in entity_visuals.get("assets") or []:
                if ref.get("asset"):
                    referenced.add(ref["asset"])
                if ref.get("source_asset"):
                    referenced.add(ref["source_asset"])
            for item in entity_visuals.get("items") or []:
                for ref in item.get("frame_refs") or []:
                    if ref.get("asset"):
                        referenced.add(ref["asset"])
                    if ref.get("source_asset"):
                        referenced.add(ref["source_asset"])
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

    @staticmethod
    def _compress_archive_visual(data: bytes, max_bytes: int = 3000) -> bytes | None:
        """Downsample an aged visual while keeping a small source image available."""
        try:
            from PIL import Image, ImageOps
            source = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception:
            return None
        for width in (224, 176, 144):
            target_width = min(width, source.width)
            target_height = max(2, round(source.height * target_width / source.width))
            image = source.resize((target_width, target_height), Image.Resampling.LANCZOS)
            for quality in (36, 30, 24, 18):
                output = io.BytesIO()
                image.save(output, format="WEBP", quality=quality, method=6)
                encoded = output.getvalue()
                if len(encoded) <= max_bytes and len(encoded) < len(data):
                    return encoded
        return None

    def _demote_aged_visuals(self, moments: dict, recent_days: int = 30,
                             protected_recent: int = 20) -> dict:
        """Keep recent images clearer and retain smaller source images for older memories."""
        now = datetime.now()
        cutoff = now.date() - timedelta(days=max(1, int(recent_days)))
        ordered = sorted(moments.values(), key=lambda m: str(m.get("timestamp") or ""))
        newest_ids = {str(m.get("id") or "") for m in ordered[-max(0, int(protected_recent)):]} if protected_recent else set()
        old_ids = set()
        protected_ids = set(newest_ids)
        for moment in ordered:
            mid = str(moment.get("id") or "")
            if moment.get("highlighted") or moment.get("nightly_importance"):
                protected_ids.add(mid)
            value = str(moment.get("timestamp") or "").strip()
            try:
                timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if timestamp.tzinfo:
                    timestamp = timestamp.astimezone().replace(tzinfo=None)
            except (TypeError, ValueError):
                protected_ids.add(mid)
                continue
            if timestamp.date() <= cutoff and mid not in protected_ids:
                old_ids.add(mid)
            else:
                protected_ids.add(mid)

        def visual_references(moment):
            mm = moment.get("multimodal") or {}
            refs = []
            refs.extend((ref, "asset") for ref in media_refs(mm, "visual"))
            objects = mm.get("object_visuals") or {}
            for item in objects.get("items") or []:
                if not isinstance(item, dict):
                    continue
                for ref in item.get("frame_refs") or []:
                    if isinstance(ref, dict):
                        refs.append((ref, "asset"))
                        if ref.get("source_asset"):
                            refs.append((ref, "source_asset"))
            people = mm.get("person_visuals") or {}
            for item in people.get("items") or []:
                if not isinstance(item, dict):
                    continue
                for ref in item.get("frame_refs") or []:
                    if isinstance(ref, dict):
                        refs.append((ref, "asset"))
                        if ref.get("source_asset"):
                            refs.append((ref, "source_asset"))
            entity_visuals = mm.get("entity_visuals") or {}
            for ref in entity_visuals.get("assets") or []:
                if isinstance(ref, dict):
                    refs.append((ref, "asset"))
                    if ref.get("source_asset"):
                        refs.append((ref, "source_asset"))
            for item in entity_visuals.get("items") or []:
                if not isinstance(item, dict):
                    continue
                for ref in item.get("frame_refs") or []:
                    if isinstance(ref, dict):
                        refs.append((ref, "asset"))
                        if ref.get("source_asset"):
                            refs.append((ref, "source_asset"))
            return refs

        protected_assets = set()
        for moment in ordered:
            mid = str(moment.get("id") or "")
            for ref, field in visual_references(moment):
                asset = ref.get(field)
                if asset and (mid not in old_ids or ref.get("nightly_importance")):
                    protected_assets.add(str(asset))

        refs_by_asset = {}
        for moment in ordered:
            mid = str(moment.get("id") or "")
            if mid not in old_ids:
                continue
            mm = moment.get("multimodal") or {}
            if mm.get("visual_quality_tier") == "archive":
                continue
            for ref, field in visual_references(moment):
                asset = str(ref.get(field) or "")
                if (not asset or asset in protected_assets
                        or ref.get("quality_tier") == "archive"
                        or ref.get("nightly_importance")
                        or ref.get("visual_role") in {"person_crop", "face_crop", "qwen_input_frame"}):
                    continue
                refs_by_asset.setdefault(asset, []).append((mid, ref, field))

        changed_moments = set()
        demoted_assets = 0
        removed_assets = 0
        for asset, ref_rows in refs_by_asset.items():
            absolute = (self.root / asset).resolve()
            if self.root not in absolute.parents or not absolute.is_file():
                continue
            try:
                original = absolute.read_bytes()
            except OSError:
                continue
            encoded = self._compress_archive_visual(original)
            if not encoded:
                continue
            saved = self._put("visual", encoded, "webp", len(encoded), allow_over_budget=True)
            if not saved or saved.get("asset") == asset:
                continue
            for mid, ref, field in ref_rows:
                ref[field] = saved["asset"]
                if field == "asset":
                    ref["bytes"] = saved["bytes"]
                ref["quality_tier"] = "archive"
                changed_moments.add(mid)
            demoted_assets += 1
            # Bound transient duplicate files if compaction starts near its budget.
            if demoted_assets % 100 == 0:
                removed_assets += len(self.release_unreferenced(moments))

        demoted_at = now.isoformat(timespec="seconds")
        for mid in changed_moments:
            moment = moments.get(mid) or {}
            mm = moment.get("multimodal") or {}
            tiers = [ref.get("quality_tier") for ref, _field in visual_references(moment)]
            mm["visual_quality_tier"] = "archive" if tiers and all(tier == "archive" for tier in tiers) else "mixed"
            mm["visual_quality_updated_at"] = demoted_at
            moment["multimodal"] = mm
            moments[mid] = moment
        if changed_moments:
            removed_assets += len(self.release_unreferenced(moments))
        return {
            "demoted_moments": len(changed_moments),
            "demoted_assets": demoted_assets,
            "removed_assets": removed_assets,
            "updated_at": demoted_at if changed_moments else None,
            "_changed_moment_ids": sorted(changed_moments),
        }

    def compact_to_budget(self, moments: dict, budget_bytes: int | None = None,
                          protected_recent: int = 20) -> dict:
        """Tier old images before evicting; keep structured facts and source links."""
        memory_file = self.root / "memory.json"
        text_bytes = memory_file.stat().st_size if memory_file.is_file() else 0
        effective_budget = min(self.asset_budget, max(0, self.total_budget - self.non_asset_reserve - text_bytes))
        budget = effective_budget if budget_bytes is None else min(effective_budget, max(0, budget_bytes))
        original_before = self.size_bytes()
        recent_days = int(os.environ.get("MEMORY_RECENT_MEDIA_DAYS", "30"))
        demotion = self._demote_aged_visuals(
            moments, recent_days=recent_days, protected_recent=protected_recent)
        before_eviction = self.size_bytes()
        changed_ids = list(demotion.pop("_changed_moment_ids", []))
        if before_eviction <= budget:
            return {
                "before_bytes": original_before, "after_bytes": before_eviction,
                "evicted_moments": 0, "removed_assets": demotion["removed_assets"],
                "demoted_moments": demotion["demoted_moments"],
                "demoted_assets": demotion["demoted_assets"],
                "target_met": True, "_changed_moment_ids": changed_ids,
            }
        ordered = sorted(moments.values(), key=lambda m: str(m.get("timestamp") or ""))
        protected_count = max(0, int(protected_recent))
        protected_ids = {m.get("id") for m in ordered[-protected_count:]} if protected_count else set()
        candidates = [m for m in ordered if m.get("id") not in protected_ids
                      and not m.get("highlighted") and not m.get("nightly_importance")]
        candidates.sort(key=lambda m: (
            int(m.get("access_count") or 0), str(m.get("last_used") or ""),
            str(m.get("timestamp") or "")))
        evicted = 0
        removed_assets = demotion["removed_assets"]
        for moment in candidates:
            mm = moment.get("multimodal") or {}
            if not (mm.get("visual") or mm.get("audio")):
                continue
            mm.pop("audio", None)
            mm.pop("visual", None)
            mm.pop("audio_clips", None)
            mm.pop("visual_frames", None)
            mm.pop("object_visuals", None)
            mm.pop("person_visuals", None)
            mm["evicted_at"] = datetime.now().isoformat()
            evicted += 1
            changed_ids.append(str(moment.get("id") or ""))
            removed_assets += len(self.release_unreferenced(moments))
            if self.size_bytes() <= budget:
                break
        after = self.size_bytes()
        return {
            "before_bytes": original_before, "after_bytes": after,
            "evicted_moments": evicted, "removed_assets": removed_assets,
            "demoted_moments": demotion["demoted_moments"],
            "demoted_assets": demotion["demoted_assets"],
            "target_met": after <= budget,
            "_changed_moment_ids": sorted(set(changed_ids)),
        }


def compare_multimodal(left: dict, right: dict) -> dict[str, Any]:
    """Compare every retained temporal sample, with inspectable best-pair times."""
    a, b = (left.get("multimodal") or left), (right.get("multimodal") or right)
    scores: dict[str, float] = {}
    evidence: dict[str, Any] = {}

    def best_pair(left_refs, right_refs, value_of, similarity_of):
        left_samples = [(ref, value_of(ref)) for ref in left_refs]
        right_samples = [(ref, value_of(ref)) for ref in right_refs]
        left_samples = [(ref, value) for ref, value in left_samples if value is not None]
        right_samples = [(ref, value) for ref, value in right_samples if value is not None]
        pairs = []
        for left_ref, left_value in left_samples:
            for right_ref, right_value in right_samples:
                score = similarity_of(left_value, right_value)
                if score is not None:
                    pairs.append((score, left_ref, right_ref))
        if not pairs:
            return None, None
        score, left_ref, right_ref = max(
            pairs, key=lambda row: (row[0], str(row[1].get("asset") or ""),
                                    str(row[2].get("asset") or "")))
        return score, {
            "left_at_s": round(float(left_ref.get("at_s") or 0), 2),
            "right_at_s": round(float(right_ref.get("at_s") or 0), 2),
            "pairs_compared": len(pairs),
        }

    def visual_value(ref):
        try:
            value = str(ref.get("dhash") or "")
            return int(value, 16) if len(value) == 16 else None
        except (TypeError, ValueError):
            return None

    def visual_similarity(x, y):
        return 1 - (x ^ y).bit_count() / 64

    visual, visual_evidence = best_pair(
        media_refs(a, "visual"), media_refs(b, "visual"),
        visual_value, visual_similarity)
    if visual is not None:
        scores["visual"] = round(visual, 4)
        evidence["visual"] = visual_evidence

    e1, e2 = a.get("entities") or {}, b.get("entities") or {}
    tags1 = {str(v).casefold() for key in ("people", "objects", "environments", "speakers")
             for v in (e1.get(key) or [])}
    tags2 = {str(v).casefold() for key in ("people", "objects", "environments", "speakers")
             for v in (e2.get(key) or [])}
    for e, tags in ((e1, tags1), (e2, tags2)):
        if e.get("activity"):
            tags.add(str(e["activity"]).casefold())
        if e.get("location"):
            tags.add(str(e["location"]).casefold())
    if tags1 and tags2:
        scores["entities"] = round(len(tags1 & tags2) / len(tags1 | tags2), 4)
    weights = {"visual": 0.67, "entities": 0.33}
    total = sum(weights[key] for key in scores)
    return {
        "score": round(sum(scores[key] * weights[key] for key in scores) / total, 4)
        if total else None,
        "modalities": scores,
        "evidence": evidence,
    }
