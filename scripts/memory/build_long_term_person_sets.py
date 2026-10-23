"""Build person-specific DAY1 long-term runner sets from local EgoLife clips.

The structured source has bilingual rows for six participants. This builder keeps
English rows only, annotates them with the project's six-dimension ground truth,
and excludes rows whose source video is not present locally. The metadata records
whether each participant set is a complete day or a partial video subset.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_testset as bt

SOURCE = ROOT / "data/egolife/structured/moments.jsonl"
OUT = ROOT / "data/egolife/testset/long_term/person_sets"
VIDEO_ROOT = ROOT / "data/egolife"

def hour_of(row):
    parts = str(row.get("id", "")).split("_")
    stamp = parts[3] if len(parts) > 3 else "00000000"
    return stamp[:2] if len(stamp) >= 2 else "00"

def participant_of(row):
    parts = str(row.get("id", "")).split("_")
    return "_".join(parts[1:3]) if len(parts) >= 3 else ""

def local_video(row):
    raw = str(row.get("video") or "")
    rel = raw.replace("data/EgoLife/", "data/egolife/")
    return ROOT / rel

def main():
    grouped = defaultdict(list)
    with SOURCE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("id", "")).endswith("_en"):
                continue
            grouped[participant_of(row)].append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for participant, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (hour_of(r), float(r.get("start_time", 0))))
        available = [r for r in rows if local_video(r).is_file()]
        if not available:
            continue

        prepared = []
        for source in available:
            row = dict(source)
            row["ground_truth"] = bt.annotate_classification(row)
            row["_hour"] = hour_of(row)
            prepared.append(row)

        out_file = OUT / f"{participant}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for row in prepared:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        coverage = len(available) / len(rows) if rows else 0
        meta = {
            "participant": participant,
            "day": "DAY1",
            "total_candidates": len(rows),
            "available_count": len(available),
            "coverage": round(coverage, 4),
            "complete": len(available) == len(rows),
            "hours_covered": sorted({r["_hour"] for r in prepared}),
            "first_id": prepared[0]["id"],
            "last_id": prepared[-1]["id"],
            "file": out_file.name,
            "source": "structured/moments.jsonl; English rows with local source video",
        }
        meta_file = OUT / f"{participant}.meta.json"
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        summary.append(meta)

    (OUT / "index.json").write_text(
        json.dumps({"datasets": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for meta in summary:
        pct = round(meta["coverage"] * 100, 1)
        status = "complete" if meta["complete"] else "partial"
        print(f"{meta['participant']}: {meta['available_count']}/{meta['total_candidates']} "
              f"({pct}%, {status}), hours={','.join(meta['hours_covered'])}")

if __name__ == "__main__":
    main()
