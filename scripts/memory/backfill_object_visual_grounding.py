#!/usr/bin/env python3
"""Backfill per-object visual evidence into a new, resumable evaluation snapshot.

This never mutates the source run. Qwen's object/appearance judgments are model
outputs, not human ground truth; uncertain objects remain not_verified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multimodal_memory import MultimodalAssetStore, media_refs  # noqa: E402
from qwen_perceive import QwenPerceiver  # noqa: E402

DEFAULT_SOURCE = ROOT / "data/egolife/runs/multimodal_full_day_v8_20261007"
DEFAULT_OUTPUT = ROOT / "data/egolife/runs/multimodal_full_day_v10_object_crops_20261012"


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def read_progress(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            mid = str(row.get("id") or "")
            if mid and isinstance(row.get("object_visuals"), dict):
                rows[mid] = row["object_visuals"]
    return rows


def unique_labels(values) -> list[str]:
    labels, seen = [], set()
    for value in values if isinstance(values, list) else []:
        label = str(value or "").strip()
        key = label.casefold()
        if label and key not in seen:
            labels.append(label[:100])
            seen.add(key)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many not-yet-checkpointed moments")
    args = parser.parse_args()

    source_file = args.input_dir / "memory.json"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    health = QwenPerceiver(api_url=args.api_url,
                           cache_file=args.output_dir / "unused_grounding_cache.json").health()
    if not health.get("model_loaded"):
        raise RuntimeError(f"Qwen service is not ready: {health}")
    model_label = health.get("model") or "Qwen3-VL"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_assets = args.input_dir / "assets"
    target_assets = args.output_dir / "assets"
    if source_assets.is_dir() and not target_assets.exists():
        shutil.copytree(source_assets, target_assets)
    progress_file = args.output_dir / "object_grounding_progress.jsonl"
    progress = read_progress(progress_file)
    asset_store = MultimodalAssetStore(args.output_dir)

    db = json.loads(source_file.read_text(encoding="utf-8"))
    archive = db.get("layer7", {}).get("moments", {})
    pending = [(str(mid), moment) for mid, moment in archive.items()
               if str(mid) not in progress]
    if args.limit > 0:
        pending = pending[:args.limit]
    print(json.dumps({
        "source": str(source_file), "output": str(args.output_dir),
        "moments_total": len(archive), "already_checkpointed": len(progress),
        "moments_this_run": len(pending), "model": model_label,
    }, ensure_ascii=False), flush=True)

    started = time.perf_counter()
    with progress_file.open("a", encoding="utf-8") as checkpoint:
        for run_index, (mid, moment) in enumerate(pending, 1):
            mm = moment.get("multimodal") or {}
            labels = unique_labels(moment.get("objects") or [])
            refs = media_refs(mm, "visual")[:7]
            frames = []
            for ref in refs:
                asset = str(ref.get("asset") or "")
                file = args.output_dir / asset
                if asset and file.is_file():
                    frames.append({
                        **ref,
                        "mime": ref.get("mime") or "image/webp",
                        "bytes": file.read_bytes(),
                    })

            parsed: dict[str, dict] = {}
            chunk_states = []
            if not labels:
                state = "not_needed"
            elif not frames:
                state = "no_frames"
            else:
                for offset in range(0, len(labels), 12):
                    chunk = labels[offset:offset + 12]
                    result = QwenPerceiver(
                        api_url=args.api_url,
                        cache_file=args.output_dir / "unused_grounding_cache.json",
                    ).ground_objects(chunk, frames)
                    chunk_states.append(str(result.get("status") or "unavailable"))
                    for item in result.get("items") or []:
                        if isinstance(item, dict) and item.get("object"):
                            parsed[str(item["object"]).casefold()] = item
                state = ("completed" if chunk_states and
                         all(item == "completed" for item in chunk_states)
                         else "partial")

            items = []
            for label in labels:
                raw = parsed.get(label.casefold()) or {
                    "status": "not_verified", "appearance": "", "frame_indices": [],
                }
                frame_indices = raw.get("frame_indices") or []
                boxes = {}
                for box in raw.get("frame_boxes") or []:
                    try:
                        box_index = int(box.get("frame_index"))
                        bbox = box.get("bbox")
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if isinstance(bbox, list) and len(bbox) == 4:
                        boxes[box_index] = bbox
                frame_indices = list(dict.fromkeys([
                    *[index for index in frame_indices if isinstance(index, int)],
                    *boxes.keys(),
                ]))
                frame_links = []
                for index in frame_indices[:1]:
                    if isinstance(index, int) and 1 <= index <= len(frames):
                        ref = frames[index - 1]
                        if ref.get("asset"):
                            crop = None
                            bbox = boxes.get(index)
                            if bbox:
                                crop_bytes = asset_store.crop_visual_candidate(
                                    frames[index - 1].get("bytes"), bbox, 12_000)
                                crop = asset_store._put("visual", crop_bytes, "webp", 12_000)
                            frame_link = {
                                "asset": crop.get("asset") if crop else ref["asset"],
                                "at_s": ref.get("at_s"),
                                "source_id": ref.get("source_id") or mm.get("source_id") or "",
                                "visual_role": "object_crop" if crop else "source_frame",
                            }
                            if crop:
                                frame_link["source_asset"] = ref["asset"]
                                frame_link["crop_bbox"] = [round(float(value), 5) for value in bbox]
                            frame_links.append(frame_link)
                status = str(raw.get("status") or "not_verified")
                if status in {"grounded", "visible_unlabeled"} and not frame_links:
                    status = "not_verified"
                appearance = str(raw.get("appearance") or "")[:240]
                if status == "not_verified":
                    appearance = ""
                items.append({
                    "object": label,
                    "status": status,
                    "appearance": appearance,
                    "frame_refs": frame_links,
                })
            grounding = {
                "schema": 1,
                "status": state,
                "model": "Qwen3-VL",
                "prompt_version": "object-visual-grounding-v3-bbox",
                "frame_count": len(frames),
                "items": items,
            }
            row = {"id": mid, "object_visuals": grounding}
            checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
            checkpoint.flush()
            progress[mid] = grounding

            if run_index % 10 == 0 or run_index == len(pending):
                counts = Counter(
                    item.get("status", "not_verified")
                    for item in items
                )
                print(json.dumps({
                    "progress": f"{len(progress)}/{len(archive)}",
                    "current_id": mid,
                    "current_objects": len(items),
                    "current_grounded": counts.get("grounded", 0),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }, ensure_ascii=False), flush=True)

    for mid, grounding in progress.items():
        moment = archive.get(mid)
        if moment is not None:
            moment.setdefault("multimodal", {})["object_visuals"] = grounding

    output_memory = args.output_dir / "memory.json"
    atomic_write_json(output_memory, db)
    source_asset_bytes = sum(
        file.stat().st_size for file in (args.input_dir / "assets").rglob("*")
        if file.is_file()
    )
    asset_bytes = sum(
        file.stat().st_size for file in (args.output_dir / "assets").rglob("*")
        if file.is_file()
    )
    source_capacity = {}
    capacity_path = args.input_dir / "capacity_report.json"
    if capacity_path.is_file():
        source_capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    memory_bytes = output_memory.stat().st_size
    daily_count = max(len(archive), 1)
    total_bytes = memory_bytes + asset_bytes
    ten_year_bytes = total_bytes / daily_count * 725 * 3650
    capacity_report = {
        **source_capacity,
        "dataset": str(source_file),
        "stored_moments": len(archive),
        "memory_json_bytes": memory_bytes,
        "asset_bytes": asset_bytes,
        "total_bytes": total_bytes,
        "mean_bytes_per_moment": round(total_bytes / daily_count, 1),
        "projected_10yr_gb_at_725_moments_per_day": round(ten_year_bytes / 1e9, 2),
        "projected_10yr_gib_at_725_moments_per_day": round(ten_year_bytes / (1024 ** 3), 2),
        "previous_v8_total_bytes": source_capacity.get("total_bytes"),
        "incremental_annotation_bytes": memory_bytes - source_file.stat().st_size,
        "raw_video_included": False,
        "projection_note": "Measured v9 memory plus retained evidence assets, linearly projected at 725 moments/day; excludes original video, software, and unmeasured external index overhead.",
    }
    atomic_write_json(args.output_dir / "capacity_report.json", capacity_report)
    item_counts = Counter()
    moments_with_items = 0
    grounded_with_frame = 0
    objects_with_crops = 0
    crop_refs = 0
    for moment in archive.values():
        items = (moment.get("multimodal") or {}).get("object_visuals", {}).get("items") or []
        moments_with_items += bool(items)
        for item in items:
            item_counts[str(item.get("status") or "not_verified")] += 1
            refs = item.get("frame_refs") or []
            if item.get("status") in {"grounded", "visible_unlabeled"} and refs:
                grounded_with_frame += 1
            if any(ref.get("visual_role") == "object_crop" for ref in refs):
                objects_with_crops += 1
                crop_refs += sum(ref.get("visual_role") == "object_crop" for ref in refs)
    report = {
        "source_run": str(args.input_dir),
        "output_run": str(args.output_dir),
        "model": model_label,
        "moments_total": len(archive),
        "moments_with_object_annotations": moments_with_items,
        "object_mentions": sum(len(moment.get("objects") or [])
                               for moment in archive.values()),
        "unique_object_labels": sum(len(unique_labels(moment.get("objects") or []))
                                    for moment in archive.values()),
        "object_status_counts": dict(item_counts),
        "objects_with_grounded_frame_refs": grounded_with_frame,
        "objects_with_local_crops": objects_with_crops,
        "object_crop_refs": crop_refs,
        "elapsed_s_this_run": round(time.perf_counter() - started, 1),
        "label": "Qwen self-reported visual grounding; not human-verified ground truth",
    }
    atomic_write_json(args.output_dir / "object_grounding_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
