#!/usr/bin/env python3
"""Build a blinded, human-editable annotation sheet from the long-term EgoLife set."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/egolife/testset/long_term/full_day.jsonl"
DEFAULT_OUTPUT = ROOT / "data/egolife/testset/long_term/category_review_template.csv"

def local_video_path(value):
    return str(value or "").replace("\\", "/").replace("EgoLife", "egolife")

def sample_id(row):
    value = str(row.get("id", ""))
    for suffix in ("_en", "_cn"):
        if value.endswith(suffix):
            return value[:-len(suffix)]
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    groups = defaultdict(list)
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            video = local_video_path(row.get("video"))
            start = float(row.get("start_time", 0))
            end = float(row.get("end_time", 0))
            groups[(video, start, end)].append(row)

    fields = [
        "sample_id", "source_row_ids_json", "video_path", "video_available",
        "start_seconds", "end_seconds", "transcripts_json",
        "annotation_status", "reviewer_id", "confidence", "reviewer_notes",
        "people_json", "locations_json", "activity_events_json",
        "environments_json", "objects_json",
    ]
    output_rows = []
    video_count = 0
    for (video, start, end), rows in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        rows.sort(key=lambda row: (0 if str(row.get("id", "")).endswith("_en") else 1, str(row.get("id", ""))))
        path = ROOT / video
        video_available = path.is_file()
        video_count += int(video_available)
        transcripts = []
        for row in rows:
            value = str(row.get("transcript") or "").strip()
            if value and value not in transcripts:
                transcripts.append(value)
        output_rows.append({
            "sample_id": sample_id(rows[0]),
            "source_row_ids_json": json.dumps(sorted(str(row.get("id", "")) for row in rows), ensure_ascii=False),
            "video_path": video,
            "video_available": str(video_available).lower(),
            "start_seconds": f"{start:.2f}",
            "end_seconds": f"{end:.2f}",
            "transcripts_json": json.dumps(transcripts, ensure_ascii=False),
            "annotation_status": "pending",
            "reviewer_id": "",
            "confidence": "",
            "reviewer_notes": "",
            "people_json": "",
            "locations_json": "",
            "activity_events_json": "",
            "environments_json": "",
            "objects_json": "",
        })

    if not output_rows:
        raise SystemExit("No rows found in input dataset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({
        "ok": True,
        "input_rows": sum(map(len, groups.values())),
        "unique_clip_units": len(output_rows),
        "videos_available": video_count,
        "output": str(args.output),
        "blinded": True,
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
