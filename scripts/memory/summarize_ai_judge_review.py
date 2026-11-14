#!/usr/bin/env python3
"""Summarize independent AI semantic reviews against cached Qwen labels."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/egolife/runs/ai_blind_review"
TEMPLATE = ROOT / "data/egolife/testset/long_term/category_review_template_expanded.csv"
DEFAULT_ANNOTATIONS = [
    RUNS / "independent_sample_30.csv",
    RUNS / "independent_sample_A1_batch2.csv",
    RUNS / "independent_sample_A1_batch3.csv",
]
DEFAULT_JUDGEMENTS = [
    RUNS / "independent_sample_30_judgement.json",
    RUNS / "independent_sample_A1_batch2_judgement.json",
    RUNS / "independent_sample_A1_batch3_judgement.json",
]
DEFAULT_QWEN = ROOT / "data/egolife/runs/qwen_predictions_category_review_expanded_825.json"
DEFAULT_OUTPUT = RUNS / "ai_judge_review_report_20260927.json"
DIMS = ("people", "locations", "activity_events", "environments", "objects")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="+", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--judgements", nargs="+", type=Path, default=DEFAULT_JUDGEMENTS)
    parser.add_argument("--qwen-report", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if len(args.annotations) != len(args.judgements):
        raise ValueError("Each annotation CSV needs its matching judgement JSON")

    rows = []
    status_by_dimension = {dim: {state: set() for state in ("correct", "partial", "incorrect")}
                           for dim in DIMS}
    reviewer = None
    protocols = []
    corrections = []
    rubric = None
    batch_counts = []
    for csv_path, judgement_path in zip(args.annotations, args.judgements):
        batch_rows = read_csv(csv_path)
        judgement = json.loads(judgement_path.read_text(encoding="utf-8"))
        reviewer = reviewer or judgement.get("reviewer")
        rubric = rubric or judgement.get("rubric")
        protocols.append(judgement.get("blind_review_protocol", ""))
        corrections.extend(judgement.get("post_comparison_corrections", []))
        ids = [row.get("sample_id", "") for row in batch_rows]
        if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid sample IDs in {csv_path}")
        if set(ids) != set(judgement.get("sample_ids", [])):
            raise ValueError(f"Judgement IDs do not match {csv_path}")
        for dim in DIMS:
            mapping = judgement.get("status_by_sample", {}).get(dim, {})
            assigned = set()
            for state in ("correct", "partial", "incorrect"):
                values = set(mapping.get(state, []))
                if assigned & values:
                    raise ValueError(f"{dim}: sample has multiple judgements in {judgement_path}")
                assigned |= values
                status_by_dimension[dim][state].update(values)
            if assigned != set(ids):
                raise ValueError(f"{dim}: incomplete judgement coverage in {judgement_path}")
        batch_counts.append({"file": csv_path.name, "clips": len(batch_rows)})
        rows.extend(batch_rows)

    ids = [row.get("sample_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Review batches contain overlapping sample IDs")

    with args.template.open(encoding="utf-8-sig", newline="") as stream:
        population_rows = list(csv.DictReader(stream))
    population = Counter()
    population_by_hour = defaultdict(Counter)
    def hour_of(sample_id):
        parts = str(sample_id or "").split("_")
        return parts[3][:2] if len(parts) > 3 and len(parts[3]) >= 2 else "unknown"
    for row in population_rows:
        participant = row["video_path"].split("/")[2]
        sample_id = Path(row["video_path"]).stem
        population[participant] += 1
        population_by_hour[participant][hour_of(sample_id)] += 1
    total_population = sum(population.values())

    qwen = json.loads(args.qwen_report.read_text(encoding="utf-8"))
    details = {item["id"]: item for item in qwen["scenarios"]["long_term"]["details"]}
    sample_by_participant = Counter()
    confidence = Counter()
    records = []
    for row in rows:
        sample_id = row["sample_id"]
        parts = (row.get("video_path") or "").split("/")
        participant = parts[2] if len(parts) > 2 else "unknown"
        sample_by_participant[participant] += 1
        confidence[row.get("confidence", "")] += 1
        source_ids = json.loads(row.get("source_row_ids_json") or "[]")
        qwen_item = next((details[value] for value in source_ids if value in details), None)
        if qwen_item is None:
            raise ValueError(f"Qwen report has no output for {sample_id}")
        categories = {}
        for dim in DIMS:
            reference = json.loads(row.get(dim + "_json") or "[]")
            prediction = qwen_item.get(dim, {}).get("predicted", [])
            state = next(name for name in ("correct", "partial", "incorrect")
                         if sample_id in status_by_dimension[dim][name])
            categories[dim] = {
                "ai_reference": reference,
                "qwen_prediction": prediction,
                "judgement": state,
            }
        records.append({
            "sample_id": sample_id,
            "participant": participant,
            "confidence": row.get("confidence", ""),
            "evidence_note": row.get("reviewer_notes", ""),
            "dimensions": categories,
        })

    per_dimension = {}
    for dim in DIMS:
        states = status_by_dimension[dim]
        fully = len(states["correct"])
        partial = len(states["partial"])
        incorrect = len(states["incorrect"])
        weighted_full = 0.0
        strata = {}
        for participant, population_n in sorted(population.items()):
            sample_n = sample_by_participant[participant]
            correct_n = sum(
                sample_id in states["correct"]
                for sample_id, row in (
                    (row["sample_id"], row)
                    for row in rows
                    if row["video_path"].split("/")[2] == participant
                )
            )
            rate = correct_n / sample_n if sample_n else None
            if participant == "A1_JAKE":
                for hour, hour_population in population_by_hour[participant].items():
                    hour_rows = [row for row in rows
                                 if row["video_path"].split("/")[2] == participant
                                 and hour_of(row["sample_id"]) == hour]
                    if hour_rows:
                        hour_correct = sum(row["sample_id"] in states["correct"] for row in hour_rows)
                        weighted_full += (hour_correct / len(hour_rows)) * hour_population
                    elif rate is not None:
                        # No sampled clips in this hour: use the participant estimate and disclose this fallback.
                        weighted_full += rate * hour_population
            elif rate is not None:
                weighted_full += rate * population_n
            strata[participant] = {
                "population_clips": population_n,
                "reviewed_clips": sample_n,
                "fully_supported": correct_n,
                "sample_rate": round(rate, 4) if rate is not None else None,
            }
        per_dimension[dim] = {
            "fully_supported": fully,
            "partially_supported": partial,
            "incorrect_or_unsupported": incorrect,
            "participant_balanced_full_rate": round(fully / len(rows), 4),
            "participant_balanced_full_rate_percent": round(fully / len(rows) * 100, 1),
            "population_weighted_full_rate_estimate": round(weighted_full / total_population, 4),
            "population_weighted_full_rate_percent_estimate": round(weighted_full / total_population * 100, 1),
            "strata": strata,
        }

    total_fields = len(rows) * len(DIMS)
    full = sum(item["fully_supported"] for item in per_dimension.values())
    partial = sum(item["partially_supported"] for item in per_dimension.values())
    incorrect = sum(item["incorrect_or_unsupported"] for item in per_dimension.values())
    weighted_rate = sum(item["population_weighted_full_rate_estimate"] for item in per_dimension.values()) / len(DIMS)
    report = {
        "created_at": __import__("datetime").datetime.now().date().isoformat(),
        "reviewer": reviewer,
        "review_type": "AI semantic review; not human gold-standard adjudication",
        "blind_protocol": " ".join(value for value in protocols if value),
        "rubric": rubric,
        "post_comparison_corrections": corrections,
        "sample": {
            "unique_clips_reviewed": len(rows),
            "candidate_unique_clips": total_population,
            "participant_distribution": dict(sorted(sample_by_participant.items())),
            "population_distribution": dict(sorted(population.items())),
            "confidence_distribution": dict(sorted(confidence.items())),
            "source_days": ["DAY1"],
            "batches": batch_counts,
            "evidence": "five frames at +2, +8, +15, +22, +28 seconds plus bilingual transcript",
        },
        "overall": {
            "fully_supported_fields": full,
            "partially_supported_fields": partial,
            "incorrect_or_unsupported_fields": incorrect,
            "judged_fields": total_fields,
            "participant_balanced_full_rate": round(full / total_fields, 4),
            "participant_balanced_full_rate_percent": round(full / total_fields * 100, 1),
            "population_weighted_full_rate_estimate": round(weighted_rate, 4),
            "population_weighted_full_rate_percent_estimate": round(weighted_rate * 100, 1),
            "target_percent": 95,
            "sample_status": "below_target_sample",
            "interpretation": "Full support is the conservative score; partial fields do not count as correct. The population estimate weights A1_JAKE by hour bins and participant strata, but remains a small AI-reviewed point estimate, not a population-level certification.",
        },
        "by_dimension": per_dimension,
        "records": records,
        "limitations": [
            "All reviewed clips are DAY1; this does not assess multi-day content accuracy.",
            "Five sampled frames and transcript do not show every moment in a clip.",
            "The five small participant strata have only five reviewed clips each, so their weighted contributions are uncertain.",
            "AI assessment is not an independent human gold standard.",
            "The follow-up A1_JAKE batch is allocated two clips per available hour bin; hour-bin weighting is applied, but small per-hour samples make the estimate uncertain.",
            "Qwen predictions were accessed before final judgment for the follow-up batch; reviewer anchoring is possible.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample": report["sample"],
                      "overall": report["overall"], "by_dimension": {
                          key: {k: v for k, v in value.items() if k != "strata"}
                          for key, value in per_dimension.items()
                      }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
