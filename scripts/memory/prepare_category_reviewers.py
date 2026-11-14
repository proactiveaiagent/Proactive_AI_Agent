#!/usr/bin/env python3
"""Create two independently shuffled, prediction-blind category review CSVs."""
import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "data/egolife/testset/long_term/category_review_template.csv"
LABEL_COLUMNS = (
    "people_json", "locations_json", "activity_events_json",
    "environments_json", "objects_json",
)
RESET_COLUMNS = ("reviewer_id", "confidence", "reviewer_notes", *LABEL_COLUMNS)


def read_template(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError("Template must contain a sample_id column")
        rows = list(reader)
        columns = reader.fieldnames
    if not rows:
        raise ValueError("Template is empty")
    ids = [row.get("sample_id", "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Template sample_id values must be non-empty and unique")
    missing = (set(RESET_COLUMNS) | {"annotation_status"}) - set(columns)
    if missing:
        raise ValueError(f"Template is missing required columns: {sorted(missing)}")
    return rows, columns


def write_packet(rows, columns, output, seed):
    packet = [dict(row) for row in rows]
    for row in packet:
        row["annotation_status"] = "pending"
        for column in RESET_COLUMNS:
            row[column] = ""
    random.Random(seed).shuffle(packet)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/egolife/testset/long_term/review_packets")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    rows, columns = read_template(args.template)
    ids = {row["sample_id"] for row in rows}
    first = args.output_dir / "reviewer_1.csv"
    second = args.output_dir / "reviewer_2.csv"
    write_packet(rows, columns, first, args.seed)
    write_packet(rows, columns, second, args.seed + 1)
    print(f"created two blind packets: {len(ids)} samples each")
    print(f"reviewer 1: {first}")
    print(f"reviewer 2: {second}")


if __name__ == "__main__":
    main()
