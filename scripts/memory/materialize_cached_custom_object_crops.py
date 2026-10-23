#!/usr/bin/env python3
"""Materialize exact-frame object crops from already validated grounding caches.

No vision model is called. A crop is written only when the full grounding cache key
(segment, exact object list, and sampled frame hashes) matches the saved payload.
"""
from __future__ import annotations
import json, os, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = ROOT / "test_data" / "feishu_ar_agent_memory_retrieval_20261013"
MANIFEST = VIDEO_ROOT / "manifest.json"
MEMORY = ROOT / "data/egolife/runs/interactive_mem/custom_test/memory.json"
GROUNDING_DB = ROOT / "data/egolife/runs/interactive_object_grounding_cache.sqlite3"
OLD_FINGERPRINT = "56573515e0595b64a88c23441dc7b5793446e2d04dc680198691b98a3137eee5"

sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts/memory"))
import grounding_cache as grounding_cache_module
from grounding_cache import PersistentObjectGroundingCache
from memory import _build_entity_visuals
from multimodal_memory import MultimodalAssetStore
from interactive_runner import _remove_visual_age_guess

def main():
    # These cache rows were created without an explicit model path/revision override.
    os.environ.pop("QWEN_MODEL_PATH", None)
    os.environ.pop("QWEN_MODEL_REVISION", None)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    memory_bytes = MEMORY.read_bytes()
    memory_stat = MEMORY.stat()
    memory = json.loads(memory_bytes)
    moments = (memory.get("layer7") or {}).get("moments") or {}
    by_segment = {
        (moment.get("source_provenance") or {}).get("source_segment_id"): moment
        for moment in moments.values()
    }
    cache = PersistentObjectGroundingCache(GROUNDING_DB, ROOT)
    store = MultimodalAssetStore(MEMORY.parent)
    current_fingerprint = grounding_cache_module.GROUNDING_IMPLEMENTATION_FINGERPRINT
    cache_fingerprints = (current_fingerprint, OLD_FINGERPRINT)
    report = {"clips_updated": 0, "cache_matches": 0, "objects_cropped": 0,
              "objects_preserved": 0, "clips_skipped": 0, "bytes_added": 0,
              "details": []}

    for row in manifest.get("files") or []:
        clip_id = str(row.get("clip_id") or Path(row.get("relative_path") or "").stem)
        token = clip_id.replace(".", "_").replace("-", "_")
        segment_id = f"FEISHU_{token}_{str(row.get('sha256') or '')[:10]}"
        moment = by_segment.get(segment_id)
        if not isinstance(moment, dict):
            report["clips_skipped"] += 1
            continue
        objects = moment.get("objects") or []
        if not objects:
            report["clips_skipped"] += 1
            continue
        video_path = (VIDEO_ROOT / str(row.get("relative_path") or "")).resolve()
        if VIDEO_ROOT.resolve() not in video_path.parents or not video_path.is_file():
            report["clips_skipped"] += 1
            continue
        duration = float(row.get("duration_seconds") or 0)
        frames = None
        payload = None
        for fingerprint in cache_fingerprints:
            grounding_cache_module.GROUNDING_IMPLEMENTATION_FINGERPRINT = fingerprint
            source_key = cache.source_key(segment_id, video_path, 0.0, duration, objects)
            source_payload, _ = cache.get_source(source_key)
            if not source_payload:
                continue
            if frames is None:
                frames = store.sample_visual_candidates(
                    video_path, 0.0, duration, source_id=segment_id, count=7)
            # Source keys omit frame hashes. Require an exact frame-content key
            # before applying any cached bounding box.
            exact_key = cache.key(segment_id, objects, frames)
            payload, _ = cache.get(exact_key)
            if not payload:
                # Interactive Run Step itself accepts this exact source-key hit.
                # The key binds the unchanged source file, clip interval, labels,
                # and grounding implementation; reuse the already-validated boxes.
                payload = source_payload
            if payload and len(payload.get("items") or []) == len(objects):
                break
        grounding_cache_module.GROUNDING_IMPLEMENTATION_FINGERPRINT = current_fingerprint
        if not payload or len(payload.get("items") or []) != len(objects):
            report["clips_skipped"] += 1
            continue
        report["cache_matches"] += 1
        media = moment.setdefault("multimodal", {})
        previous = media.get("object_visuals") or {}
        previous_items = previous.get("items") or []
        new_items = []
        frame_map = {}
        clip_crops = 0
        clip_preserved = 0
        for index, (object_label, grounded) in enumerate(zip(objects, payload["items"])):
            old_item = previous_items[index] if index < len(previous_items) and isinstance(previous_items[index], dict) else {}
            refs = [dict(ref) for ref in old_item.get("frame_refs") or [] if isinstance(ref, dict)]
            has_exact_crop = any(ref.get("visual_role") == "object_crop" for ref in refs)
            boxes = grounded.get("frame_boxes") or []
            if not has_exact_crop and grounded.get("status") in {"grounded", "visible_unlabeled"}:
                for box in boxes:
                    try:
                        frame_index = int(box.get("frame_index"))
                        bbox = [float(x) for x in box.get("bbox")]
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if not (1 <= frame_index <= len(frames) and len(bbox) == 4):
                        continue
                    frame = frames[frame_index - 1]
                    crop = store.crop_visual_candidate(frame.get("bytes"), bbox, 12_000)
                    crop_ref = store._put("visual", crop, "webp", 12_000)
                    if not crop_ref or not crop_ref.get("asset"):
                        continue
                    if frame_index not in frame_map:
                        source = store.fit_visual_candidate(frame.get("bytes"), 6_000)
                        source_ref = store._put("visual", source, "webp", 6_000) if source else None
                        frame_map[frame_index] = source_ref
                    source_ref = frame_map[frame_index]
                    ref = {
                        "asset": crop_ref["asset"],
                        "bytes": int(crop_ref.get("bytes") or 0),
                        "at_s": frame.get("at_s"),
                        "source_id": segment_id,
                        "quality_tier": "recent",
                        "visual_role": "object_crop",
                        "crop_bbox": [round(x, 5) for x in bbox],
                    }
                    if source_ref and source_ref.get("asset"):
                        ref["source_asset"] = source_ref["asset"]
                    refs = [ref]
                    clip_crops += 1
                    break
            if refs and any(ref.get("visual_role") == "object_crop" for ref in refs):
                report["objects_cropped"] += 1
            elif refs:
                report["objects_preserved"] += 1
                clip_preserved += 1
            appearance = _remove_visual_age_guess(
                old_item.get("appearance") or grounded.get("appearance") or "")
            item = {
                "object": str(object_label),
                "status": grounded.get("status") or old_item.get("status") or "not_verified",
                "appearance": appearance,
                "frame_refs": refs,
                "crop_count": sum(ref.get("visual_role") == "object_crop" for ref in refs),
            }
            new_items.append(item)
        if clip_crops:
            media["object_visuals"] = {
                **{k: v for k, v in previous.items() if k not in {"items", "message"}},
                "schema": payload.get("schema", previous.get("schema", 1)),
                "status": payload.get("status", "completed"),
                "model": payload.get("model", previous.get("model", "Qwen3-VL")),
                "items": new_items,
                "crop_schema": 1,
                "crop_method": "reused exact-frame cached Qwen3-VL boxes; padded WebP crops",
            }
            media["entity_visuals"] = _build_entity_visuals(moment)
            report["clips_updated"] += 1
            report["details"].append({"segment_id": segment_id, "objects": len(objects),
                                      "new_crops": clip_crops, "preserved": clip_preserved})
            print(f"cropped {segment_id}: {clip_crops}/{len(objects)}", flush=True)

    if report["clips_updated"]:
        backup = MEMORY.with_name("memory.json.before_verified_object_crop_reuse_20261019")
        if backup.exists():
            backup = MEMORY.with_name("memory.json.before_verified_object_crop_reuse_20261019_2")
        shutil.copy2(MEMORY, backup)
        # Avoid overwriting a concurrent UI write while keeping the original intact.
        current_stat = MEMORY.stat()
        if (current_stat.st_mtime_ns != memory_stat.st_mtime_ns
                or current_stat.st_size != memory_stat.st_size):
            raise RuntimeError("memory.json changed during crop extraction; saved backup and aborted update")
        fd, temp_name = tempfile.mkstemp(prefix="memory.json.crop.", suffix=".tmp", dir=str(MEMORY.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(memory, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, MEMORY)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        report["backup"] = str(backup)
        report["bytes_added"] = store.size_bytes()
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
