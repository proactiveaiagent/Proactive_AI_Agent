#!/usr/bin/env python3
"""Build a separate summary for the isolated 20-clip Qwen prompt-v2 pilot."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/egolife/runs"
REVIEW = RUNS / "ai_blind_review"
sys.path.insert(0, str(ROOT / "scripts/memory"))
from qwen_perceive import parse_analysis

SOURCE = RUNS / "qwen_prompt_v2_review_batch3_20260928.json"
ANNOTATIONS = REVIEW / "independent_sample_A1_batch3.csv"
JUDGEMENT = REVIEW / "qwen_prompt_v2_review_batch3_judgement.json"
V1_JUDGEMENT = REVIEW / "independent_sample_A1_batch3_judgement.json"
OUTPUT = RUNS / "qwen_prompt_v2_review_batch3_summary_20260928.json"
DIMS = ("people", "locations", "activity_events", "environments", "objects")


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    judgement = json.loads(JUDGEMENT.read_text(encoding="utf-8"))
    v1 = json.loads(V1_JUDGEMENT.read_text(encoding="utf-8"))
    with ANNOTATIONS.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    per_dim = {}
    for dim in DIMS:
        states = judgement["status_by_sample"][dim]
        v1states = v1["status_by_sample"][dim]
        current = {state: len(states[state]) for state in ("correct", "partial", "incorrect")}
        old = {state: len(v1states[state]) for state in ("correct", "partial", "incorrect")}
        per_dim[dim] = {
            "pilot": current,
            "pilot_fully_supported_percent": round(current["correct"] / 20 * 100, 1),
            "v1_same_clip_sample": old,
            "full_support_change_percentage_points": round((current["correct"] - old["correct"]) / 20 * 100, 1),
        }
    details = []
    for item in source["details"]:
        sid = item["id"]
        parsed = parse_analysis(item.get("raw", ""))
        fields = {
            "people": parsed.get("people", []),
            "locations": [parsed["location"]] if parsed.get("location") else [],
            "activity_events": [parsed["activity"]] if parsed.get("activity") else [],
            "environments": parsed.get("environments", []),
            "objects": parsed.get("objects", []),
        }
        row = rows[sid]
        dims = {}
        for dim in DIMS:
            status = next(state for state in ("correct", "partial", "incorrect")
                          if sid in judgement["status_by_sample"][dim][state])
            dims[dim] = {
                "reference": json.loads(row.get(dim + "_json") or "[]"),
                "prediction": fields[dim],
                "judgement": status,
            }
        details.append({
            "sample_id": sid, "num_frames": item.get("num_frames"),
            "elapsed_ms": item.get("elapsed_ms"),
            "evidence_note": row.get("reviewer_notes", ""), "dimensions": dims,
        })
    counts = Counter()
    for item in per_dim.values():
        for state, value in item["pilot"].items():
            counts[state] += value
    report = {
        "created_at": "2026-09-28",
        "benchmark": "Isolated Qwen memory-perception prompt-v2 pilot",
        "prompt_version": source.get("prompt_version"),
        "sample": {
            "clips": len(details), "participant": "A1_JAKE", "days": ["DAY1"],
            "frame_count_distribution": dict(sorted(Counter(str(x["num_frames"]) for x in details).items())),
            "cache_isolated": source.get("cache_isolated"),
            "same_clips_as_v1": True, "production_cache_modified": False,
        },
        "overall": {
            "fully_supported_fields": counts["correct"],
            "partially_supported_fields": counts["partial"],
            "incorrect_or_unsupported_fields": counts["incorrect"],
            "judged_fields": len(details) * len(DIMS),
            "full_support_percent": round(counts["correct"] / (len(details) * len(DIMS)) * 100, 1),
        },
        "by_dimension": per_dim,
        "limitations": judgement.get("limitations", []),
        "details": details,
        "source_report": str(SOURCE.relative_to(ROOT)),
        "judgement_file": str(JUDGEMENT.relative_to(ROOT)),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "overall": report["overall"],
                      "by_dimension": per_dim}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
