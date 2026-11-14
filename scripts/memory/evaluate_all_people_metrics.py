#!/usr/bin/env python3
"""Measure six-person multimodal storage and saved-index retrieval for the viz page."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "data/egolife/runs/multimodal_aligned_v8_20261007"
LONG_RUN_DIR = ROOT / "data/egolife/runs/multimodal_full_day_v9_object_grounded_20261011"
MEMORY_PATH = RUN_DIR / "memory.json"
SAMPLE_PATH = ROOT / "data/egolife/testset/short_term/moments_scattered_v2_120.jsonl"
CAPACITY_REPORT = RUN_DIR / "all_people_capacity_report.json"
RETRIEVAL_REPORT = RUN_DIR / "all_people_index_retrieval_report.json"
PARTICIPANTS = {
    "A1": "Jake", "A2": "Alice", "A3": "Tasha",
    "A4": "Lucia", "A5": "Katrina", "A6": "Shure",
}
INDEX_DIMENSIONS = ("people", "locations", "time_nodes", "activity_events", "environments", "objects")
PROJECTION_MOMENTS_PER_DAY = 725


def source_id(record: dict) -> str:
    multimodal = record.get("multimodal") or {}
    provenance = record.get("source_provenance") or {}
    return str(multimodal.get("source_id") or provenance.get("source_segment_id") or "")


def participant_for(source: str) -> str | None:
    match = re.search(r"(?:^|_)(A[1-6])_", source)
    return match.group(1) if match else None


def asset_refs(record: dict) -> dict[str, set[str]]:
    multimodal = record.get("multimodal") or {}
    visual: set[str] = set()
    audio: set[str] = set()
    for item in multimodal.get("visual_frames") or []:
        if isinstance(item, dict) and item.get("asset"):
            visual.add(str(item["asset"]))
    for item in multimodal.get("audio_clips") or []:
        if isinstance(item, dict) and item.get("asset"):
            audio.add(str(item["asset"]))
    for kind, target in (("visual", visual), ("audio", audio)):
        item = multimodal.get(kind)
        if isinstance(item, dict) and item.get("asset"):
            target.add(str(item["asset"]))
    for item in (multimodal.get("object_visuals") or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        for frame_ref in item.get("frame_refs") or []:
            if not isinstance(frame_ref, dict):
                continue
            for field in ("asset", "source_asset"):
                if frame_ref.get(field):
                    visual.add(str(frame_ref[field]))
    return {"visual": visual, "audio": audio}


def asset_path(relative: str, run_dir: Path = RUN_DIR) -> Path:
    asset_root = (run_dir / "assets").resolve()
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(asset_root):
        raise ValueError(f"asset path escapes run asset directory: {relative}")
    return path


def json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct * len(ordered)) - 1)]


def benchmark_saved_index(run_dir: Path, query_count: int = 100) -> dict:
    memory = json.loads((run_dir / "memory.json").read_text(encoding="utf-8"))
    layer7 = memory["layer7"]
    moments = layer7["moments"]
    indexes = {name: layer7.get(name) or {} for name in INDEX_DIMENSIONS}
    keys_by_dimension = {
        name: sorted(str(key) for key, ids in index.items()
                     if any(mid in moments for mid in (ids or [])))
        for name, index in indexes.items()
    }
    if any(not keys for keys in keys_by_dimension.values()):
        raise RuntimeError(
            f"{run_dir.name} has an entity index with no queryable keys: "
            f"{ {key: len(value) for key, value in keys_by_dimension.items()} }"
        )

    record_payloads = {
        moment_id: json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for moment_id, record in moments.items()
    }
    all_assets: set[str] = set()
    for record in moments.values():
        refs = asset_refs(record)
        all_assets.update(refs["visual"])
        all_assets.update(refs["audio"])
    missing_assets = [ref for ref in sorted(all_assets)
                      if not asset_path(ref, run_dir).is_file()]
    if missing_assets:
        raise RuntimeError(
            f"{run_dir.name} has {len(missing_assets)} missing linked media assets: "
            f"{missing_assets[:8]}"
        )

    def query_at(ordinal: int) -> tuple[str, int, int, int]:
        dimension = INDEX_DIMENSIONS[ordinal % len(INDEX_DIMENSIONS)]
        slot = (ordinal // len(INDEX_DIMENSIONS)) % len(keys_by_dimension[dimension])
        term = keys_by_dimension[dimension][slot]
        moment_ids = indexes[dimension].get(term) or []
        records = [json.loads(record_payloads[mid])
                   for mid in moment_ids if mid in record_payloads]
        linked_paths: set[str] = set()
        for record in records:
            refs = asset_refs(record)
            linked_paths.update(refs["visual"])
            linked_paths.update(refs["audio"])
        bytes_read = 0
        assets_read = 0
        for ref in linked_paths:
            payload = asset_path(ref, run_dir).read_bytes()
            bytes_read += len(payload)
            assets_read += 1
        return dimension, len(records), bytes_read, assets_read

    warmup_queries = 12
    for ordinal in range(warmup_queries):
        query_at(ordinal)
    durations_ms: list[float] = []
    dimension_counts: Counter[str] = Counter()
    retrieved_record_counts: list[int] = []
    linked_bytes_read: list[int] = []
    linked_assets_read: list[int] = []
    for ordinal in range(query_count):
        started = time.perf_counter_ns()
        dimension, record_count, bytes_read, assets_read = query_at(ordinal + warmup_queries)
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        dimension_counts[dimension] += 1
        retrieved_record_counts.append(record_count)
        linked_bytes_read.append(bytes_read)
        linked_assets_read.append(assets_read)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "saved_memory": str((run_dir / "memory.json").relative_to(ROOT)),
        "indexed_moments": len(moments),
        "index_dimensions": list(INDEX_DIMENSIONS),
        "index_keys_per_dimension": {
            name: len(keys) for name, keys in keys_by_dimension.items()
        },
        "query_mix": dict(dimension_counts),
        "warmup_queries": warmup_queries,
        "measured_queries": len(durations_ms),
        "p50_ms": percentile(durations_ms, 0.50),
        "p95_ms": percentile(durations_ms, 0.95),
        "max_ms": max(durations_ms),
        "mean_ms": statistics.mean(durations_ms),
        "mean_records_materialized_per_query": statistics.mean(retrieved_record_counts),
        "mean_linked_media_bytes_read_per_query": statistics.mean(linked_bytes_read),
        "mean_linked_media_assets_read_per_query": statistics.mean(linked_assets_read),
        "linked_media_assets_verified": len(all_assets),
        "missing_linked_media_assets": 0,
        "media_assets_included": [
            "visual_frames", "audio_clips", "visual", "audio",
            "object_visuals.items.frame_refs.asset",
            "object_visuals.items.frame_refs.source_asset",
        ],
        "scope_note": (
            "Lookup uses the saved people/location/time/activity/environment/object indexes, "
            "materializes complete matched records including metadata, and reads linked image/audio "
            "files, including per-object crops and their original source frames. The memory index "
            "and process are already loaded; filesystem cache may be warm. "
            "Excludes source-video analysis, model inference, and LLM generation."
        ),
    }


def project_path(value: Path) -> Path:
    path = value.expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def report_path(value: Path) -> str:
    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(value.resolve())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure per-person multimodal capacity and saved-index retrieval.")
    parser.add_argument("--short-run-dir", type=Path, default=RUN_DIR,
                        help="run directory with six-person memory.json and assets")
    parser.add_argument("--long-run-dir", type=Path, default=LONG_RUN_DIR,
                        help="full-day run directory for the long-term index benchmark")
    parser.add_argument("--sample-path", type=Path, default=SAMPLE_PATH,
                        help="120-row six-person sample JSONL used to select saved memories")
    parser.add_argument("--output-dir", type=Path,
                        help="where to write reports (default: short run directory)")
    args = parser.parse_args(argv)
    run_dir = project_path(args.short_run_dir)
    long_run_dir = project_path(args.long_run_dir)
    sample_path = project_path(args.sample_path)
    report_dir = project_path(args.output_dir) if args.output_dir else run_dir
    memory_path = run_dir / "memory.json"
    capacity_report = report_dir / "all_people_capacity_report.json"
    retrieval_report = report_dir / "all_people_index_retrieval_report.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    for required in (memory_path, sample_path, long_run_dir / "memory.json"):
        if not required.is_file():
            raise FileNotFoundError(f"required evaluation input not found: {required}")
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    layer7 = memory["layer7"]
    moments = layer7["moments"]
    sample_rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(sample_rows) != 120:
        raise RuntimeError(f"expected 120 short-term sample rows, found {len(sample_rows)}")

    sample_ids: dict[str, str] = {}
    for row in sample_rows:
        sid = str(row.get("id") or "")
        pid = participant_for(sid)
        if not sid or pid not in PARTICIPANTS:
            raise RuntimeError(f"cannot determine participant for sample row: {sid}")
        sample_ids[sid] = pid

    saved_by_source: dict[str, tuple[str, dict]] = {}
    for moment_id, record in moments.items():
        sid = source_id(record)
        if sid:
            saved_by_source[sid] = (moment_id, record)
    missing_records = sorted(set(sample_ids) - set(saved_by_source))
    if missing_records:
        raise RuntimeError(f"{len(missing_records)} short-term sample records are not saved in memory")

    per_person: dict[str, dict] = {}
    missing_assets: list[str] = []
    sample_moment_ids: set[str] = set()
    total_visual_refs = 0
    total_audio_refs = 0
    for pid, person_name in PARTICIPANTS.items():
        selected = [(sid, saved_by_source[sid][0], saved_by_source[sid][1])
                    for sid, owner in sample_ids.items() if owner == pid]
        if len(selected) != 20:
            raise RuntimeError(f"{pid} expected 20 samples, found {len(selected)}")
        record_size = sum(json_bytes(record) for _, _, record in selected)
        visual_refs: set[str] = set()
        audio_refs: set[str] = set()
        visual_ref_count = 0
        audio_ref_count = 0
        for _, moment_id, record in selected:
            sample_moment_ids.add(moment_id)
            refs = asset_refs(record)
            visual_refs.update(refs["visual"])
            audio_refs.update(refs["audio"])
            visual_ref_count += len(refs["visual"])
            audio_ref_count += len(refs["audio"])
        all_refs = visual_refs | audio_refs
        media_size = 0
        for ref in all_refs:
            path = asset_path(ref, run_dir)
            if not path.is_file():
                missing_assets.append(ref)
                continue
            media_size += path.stat().st_size
        total_visual_refs += visual_ref_count
        total_audio_refs += audio_ref_count
        total_bytes = record_size + media_size
        per_person[pid] = {
            "participant": pid,
            "name": person_name,
            "sampled_moments": len(selected),
            "record_json_bytes": record_size,
            "unique_visual_assets": len(visual_refs),
            "unique_audio_assets": len(audio_refs),
            "visual_asset_bytes": sum(asset_path(ref, run_dir).stat().st_size for ref in visual_refs if asset_path(ref, run_dir).is_file()),
            "audio_asset_bytes": sum(asset_path(ref, run_dir).stat().st_size for ref in audio_refs if asset_path(ref, run_dir).is_file()),
            "multimodal_asset_bytes": media_size,
            "total_bytes": total_bytes,
            "bytes_per_moment": total_bytes / len(selected),
            "visual_asset_references": visual_ref_count,
            "audio_asset_references": audio_ref_count,
        }
    if missing_assets:
        raise RuntimeError(f"{len(set(missing_assets))} referenced assets are missing: {sorted(set(missing_assets))[:8]}")

    average_bytes_per_moment = statistics.mean(p["bytes_per_moment"] for p in per_person.values())
    average_record_bytes = statistics.mean(p["record_json_bytes"] / p["sampled_moments"] for p in per_person.values())
    average_media_bytes = statistics.mean(p["multimodal_asset_bytes"] / p["sampled_moments"] for p in per_person.values())
    capacity = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_dataset": report_path(sample_path),
        "saved_memory": report_path(memory_path),
        "short_run": run_dir.name,
        "long_term_run": long_run_dir.name,
        "participant_count": len(per_person),
        "sampled_moments": len(sample_ids),
        "sampled_moments_per_participant": 20,
        "matched_saved_moments": len(sample_moment_ids),
        "participants": list(per_person.values()),
        "average_bytes_per_moment": average_bytes_per_moment,
        "average_record_bytes_per_moment": average_record_bytes,
        "average_multimodal_bytes_per_moment": average_media_bytes,
        "average_per_person_sample_bytes": statistics.mean(p["total_bytes"] for p in per_person.values()),
        "average_unique_visual_assets_per_participant": statistics.mean(p["unique_visual_assets"] for p in per_person.values()),
        "average_unique_audio_assets_per_participant": statistics.mean(p["unique_audio_assets"] for p in per_person.values()),
        "visual_asset_references": total_visual_refs,
        "audio_asset_references": total_audio_refs,
        "projection_moments_per_day": PROJECTION_MOMENTS_PER_DAY,
        "average_projected_daily_bytes_per_person": average_bytes_per_moment * PROJECTION_MOMENTS_PER_DAY,
        "raw_video_included": False,
        "scope_note": "Each participant's 20 saved memory records and their unique linked image/audio assets were measured separately. The six per-record byte rates are averaged equally. Raw video, runtime index/cache overhead, and database/container overhead are excluded.",
    }

    short_retrieval = benchmark_saved_index(run_dir)
    short_retrieval["profile_label"] = "短期样本（六人）"
    long_retrieval = benchmark_saved_index(long_run_dir)
    long_retrieval["profile_label"] = "长期整日样本（Jake）"
    # Keep top-level fields compatible with the existing UI while adding a
    # separately measured full-day long-term index.
    retrieval = {
        **short_retrieval,
        "profiles": {
            "short_term": short_retrieval,
            "long_term": long_retrieval,
        },
        "scope_note": (
            "Short-term and full-day long-term saved indexes are benchmarked separately. "
            "Each query looks up IDs across people/location/time/activity/environment/object, "
            "materializes complete records with metadata, and reads linked image/audio files, "
            "including object crops and source frames. "
            "Both indexes are already loaded and filesystem cache may be warm; source-video "
            "analysis, inference, and LLM generation are excluded."
        ),
    }
    capacity_report.write_text(json.dumps(capacity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    retrieval_report.write_text(json.dumps(retrieval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"capacity": capacity, "retrieval": retrieval}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()