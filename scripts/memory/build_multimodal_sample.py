"""Materialize a representative EgoLife multimodal memory and size report.

Uses existing English Qwen cache. It never invokes the model or copies raw
hour-long videos into the memory directory. Re-running resumes by source ID.
"""

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import PersonMemory
from multimodal_memory import MultimodalAssetStore
from qwen_perceive import QwenPerceiver, clip_range, video_path_of


RUNS = ROOT / "data" / "egolife" / "runs"
DATA = ROOT / "data" / "egolife" / "testset" / "short_term" / "moments_scattered_v2_120.jsonl"
OUTPUT = RUNS / "multimodal_baseline_v5_20260929"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-day", action="store_true", help="Use 725 unique English DAY1 moments")
    args = parser.parse_args()
    data_path = (ROOT / "data" / "egolife" / "testset" / "long_term" / "full_day.jsonl"
                 if args.long_day else DATA)
    output = RUNS / "multimodal_full_day_v5_20260929" if args.long_day else OUTPUT
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.long_day:
        rows = [row for row in rows if str(row.get("id", "")).endswith("_en")]
    cache = QwenPerceiver(cache_file=RUNS / "qwen_cache_expanded_eval_v5_visible_actions_full.json").cache
    memory = PersonMemory(memory_dir=str(output))
    assets = MultimodalAssetStore(output)
    existing = {m.get("multimodal", {}).get("source_id") for m in
                memory.memory["layer7"]["moments"].values()}
    missing_cache = missing_video = captured = 0
    for index, row in enumerate(rows, 1):
        source_id = row.get("id", "")
        if source_id in existing:
            continue
        fields = (cache.get(source_id) or {}).get("fields") or {}
        if not fields or not fields.get("scene"):
            missing_cache += 1
            continue
        video = video_path_of(row)
        if video is None:
            missing_video += 1
            continue
        start, end = clip_range(row)
        recent = [m.get("multimodal") for m in
                  list(memory.memory["layer7"]["moments"].values())[-100:]]
        evidence = assets.capture(video, start, end,
                                  transcript=row.get("transcript") or "",
                                  source_id=source_id, fields=fields,
                                  recent_media=recent)
        memory.add(
            scene=fields.get("scene") or "", user_action=fields.get("user_action") or "",
            needs=[], solutions=[], people=fields.get("people") or [],
            location=fields.get("location") or "", activity=fields.get("activity") or "",
            environments=fields.get("environments") or [],
            objects=fields.get("objects") or [],
            source_evidence=fields.get("source_evidence") or "",
            multimodal=evidence,
        )
        existing.add(source_id)
        captured += 1
        if captured % 10 == 0:
            print(f"captured {captured} / {len(rows)}", flush=True)

    moments = list(memory.memory["layer7"]["moments"].values())
    n = len(moments)
    asset_bytes = assets.size_bytes()
    memory_bytes = memory.memory_file.stat().st_size
    projected_10yr_gib = ((asset_bytes + memory_bytes) / n * 725 * 365.25 * 10
                          / (1024 ** 3)) if n else None
    report = {
        "dataset": str(data_path.relative_to(ROOT)),
        "source_rows": len(rows), "stored_moments": n,
        "captured_this_run": captured,
        "missing_cache": missing_cache, "missing_video": missing_video,
        "visual_evidence_count": sum(bool(m.get("multimodal", {}).get("visual")) for m in moments),
        "audio_evidence_count": sum(bool(m.get("multimodal", {}).get("audio")) for m in moments),
        "unique_visual_assets": len(list((output / "assets" / "visual").glob("*.webp"))),
        "unique_audio_assets": len(list((output / "assets" / "audio").glob("*.ogg"))),
        "memory_json_bytes": memory_bytes, "asset_bytes": asset_bytes,
        "total_bytes": asset_bytes + memory_bytes,
        "mean_bytes_per_moment": round((asset_bytes + memory_bytes) / n, 1) if n else None,
        "projected_10yr_gib_at_725_moments_per_day": round(projected_10yr_gib, 2)
        if projected_10yr_gib is not None else None,
        "raw_video_included": False,
        "projection_note": "Linear estimate from this sampled mix; not a ten-year measurement.",
    }
    (output / "capacity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
