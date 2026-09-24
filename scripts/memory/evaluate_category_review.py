#!/usr/bin/env python3
"""Score adjudicated category labels against a cached Qwen evaluation report."""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "data/egolife/testset/long_term/category_review_template.csv"
DIMS = {
    "people": ("people_json", "people"),
    "locations": ("locations_json", "locations"),
    "activity_events": ("activity_events_json", "activity_events"),
    "environments": ("environments_json", "environments"),
    "objects": ("objects_json", "objects"),
}
LIST_DIMS = ("people", "locations", "environments", "objects")


def norm(value):
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def load_aliases(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("aliases file must contain a JSON object")
    aliases = {}
    errors = []
    for dim in DIMS:
        mapping = raw.get(dim, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"aliases[{dim}] must be a JSON object")
        normalized = {}
        for alias, canonical in mapping.items():
            key = norm(alias)
            target = norm(canonical)
            if not key or not target:
                errors.append(f"{dim}: alias and canonical label must be non-empty")
            elif key in normalized and normalized[key] != target:
                errors.append(f"{dim}: normalized alias {alias!r} has conflicting mappings")
            else:
                normalized[key] = target
        aliases[dim] = normalized
    for dim, mapping in aliases.items():
        for alias, canonical in mapping.items():
            if canonical not in mapping or mapping[canonical] != canonical:
                errors.append(
                    f"{dim}: canonical target {canonical!r} must have an explicit self-mapping"
                )
    if errors:
        raise ValueError("Invalid aliases file:\n- " + "\n- ".join(errors[:30]))
    return aliases


def canonical_set(values, dim, aliases, sample_id, unmapped):
    result = set()
    for value in values:
        if not isinstance(value, str) or not norm(value):
            continue
        label = norm(value)
        canonical = aliases[dim].get(label)
        if canonical is None:
            unmapped.add((dim, label, sample_id))
        else:
            result.add(canonical)
    return result


def parse_json_list(row, column, sample_id):
    raw = (row.get(column) or "").strip()
    if not raw:
        raise ValueError(f"{sample_id}: {column} is blank; use [] for a reviewed empty field")
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{sample_id}: {column} must be a JSON array of strings")
    return value


def load_qwen_details(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    scenario = report.get("scenarios", {}).get("long_term")
    if not isinstance(scenario, dict) or not isinstance(scenario.get("details"), list):
        raise ValueError("Qwen report must contain scenarios.long_term.details")
    details = {}
    duplicates = []
    for item in scenario["details"]:
        if not isinstance(item, dict):
            raise ValueError("Qwen report details must contain JSON objects")
        item_id = item.get("id")
        if not item_id:
            continue
        if item_id in details:
            duplicates.append(item_id)
        details[item_id] = item
    if duplicates:
        raise ValueError(f"Qwen report contains duplicate detail ids: {duplicates[:5]}")
    return details


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(score, 4)


def read_rows(path, description):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{description} CSV is empty")
    sample_ids = [row.get("sample_id", "") for row in rows]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{description} sample_id values must be non-empty and unique")
    return rows, set(sample_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True,
                    help="Adjudicated CSV copied from category_review_template.csv")
    ap.add_argument("--qwen-report", type=Path, required=True,
                    help="evaluate_qwen.py long-term JSON report with per-row details")
    ap.add_argument("--aliases", type=Path, required=True,
                    help="JSON mapping per dimension: observed alias -> canonical label")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="Authoritative full review template; defaults to the 725-sample set")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Allow scoring a subset; its report is always marked complete=false")
    ap.add_argument("--output", type=Path,
                    help="Output JSON; defaults to data/egolife/runs/category_review_*.json")
    args = ap.parse_args()

    aliases = load_aliases(args.aliases)
    qwen = load_qwen_details(args.qwen_report)
    rows, annotation_ids = read_rows(args.annotations, "Annotation")
    _, expected_ids = read_rows(args.template, "Authoritative template")
    extra_ids = sorted(annotation_ids - expected_ids)
    missing_ids = sorted(expected_ids - annotation_ids)
    if extra_ids:
        raise ValueError(f"Annotations contain {len(extra_ids)} ids absent from template: {extra_ids[:5]}")
    if missing_ids and not args.allow_partial:
        raise ValueError(
            f"Annotations cover {len(annotation_ids)}/{len(expected_ids)} template samples; "
            f"pass --allow-partial only for an explicitly partial/pilot run "
            f"(missing examples: {missing_ids[:5]})"
        )

    reviewed, pending, excluded = [], [], []
    missing_qwen = []
    errors = []
    unmapped = set()
    for row in rows:
        sample_id = row["sample_id"]
        status = norm(row.get("annotation_status"))
        if status == "pending":
            pending.append(sample_id)
            continue
        if status == "excluded":
            excluded.append({"sample_id": sample_id, "reason": row.get("reviewer_notes", "")})
            continue
        if status != "reviewed":
            errors.append(f"{sample_id}: unsupported annotation_status={status!r}")
            continue
        if not (row.get("reviewer_id") or "").strip():
            errors.append(f"{sample_id}: reviewer_id is blank")
            continue
        if norm(row.get("confidence")) not in {"high", "medium", "low"}:
            errors.append(f"{sample_id}: confidence must be high, medium, or low")
            continue
        try:
            source_ids = json.loads(row.get("source_row_ids_json") or "[]")
        except json.JSONDecodeError as exc:
            errors.append(f"{sample_id}: source_row_ids_json is invalid JSON: {exc}")
            continue
        if not isinstance(source_ids, list) or any(not isinstance(value, str) for value in source_ids):
            errors.append(f"{sample_id}: source_row_ids_json must be an array of strings")
            continue
        source_ids = sorted(source_ids, key=lambda value: (0 if value.endswith("_en") else 1, value))
        qwen_item = next((qwen[source_id] for source_id in source_ids if source_id in qwen), None)
        if qwen_item is None:
            missing_qwen.append(sample_id)
            continue

        scored = {"sample_id": sample_id}
        for dim, (human_col, _) in DIMS.items():
            try:
                human = parse_json_list(row, human_col, sample_id)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
                continue
            predicted_record = qwen_item.get(dim)
            if not isinstance(predicted_record, dict) or not isinstance(predicted_record.get("predicted"), list):
                errors.append(f"{sample_id}: Qwen report has no predicted list for {dim}")
                continue
            predicted = predicted_record["predicted"]
            if any(not isinstance(item, str) for item in predicted):
                errors.append(f"{sample_id}: Qwen predicted {dim} contains non-string labels")
                continue
            if any(not norm(item) for item in human):
                errors.append(f"{sample_id}: {human_col} contains a blank label; use [] for an empty field")
                continue
            if any(not norm(item) for item in predicted):
                errors.append(f"{sample_id}: Qwen predicted {dim} contains a blank label")
                continue
            gold_set = canonical_set(human, dim, aliases, sample_id, unmapped)
            predicted_set = canonical_set(predicted, dim, aliases, sample_id, unmapped)
            if dim == "activity_events" and len(human) != 1:
                errors.append(f"{sample_id}: activity_events must contain exactly one primary gold label")
            scored[dim] = {
                "gold": sorted(gold_set),
                "predicted": sorted(predicted_set),
                "exact_match": gold_set == predicted_set,
            }
        if all(dim in scored for dim in DIMS):
            reviewed.append(scored)

    if errors:
        raise ValueError("Annotation validation failed:\n- " + "\n- ".join(errors[:30]))
    if unmapped:
        examples = [
            {"dimension": dim, "label": label, "sample_id": sample_id}
            for dim, label, sample_id in sorted(unmapped)[:30]
        ]
        raise ValueError(
            f"{len(unmapped)} observed labels have no explicit alias mapping; "
            f"add exhaustive mappings (including canonical self-mappings): "
            f"{json.dumps(examples, ensure_ascii=False)}"
        )
    if missing_qwen:
        raise ValueError(f"Missing Qwen rows for {len(missing_qwen)} reviewed samples; examples: {missing_qwen[:5]}")
    if not reviewed:
        raise ValueError("No complete reviewed samples found")

    metrics = {}
    total_exact = 0
    for dim in DIMS:
        exact = sum(item[dim]["exact_match"] for item in reviewed)
        total_exact += exact
        if dim in LIST_DIMS:
            tp = fp = fn = 0
            for item in reviewed:
                gold, predicted = set(item[dim]["gold"]), set(item[dim]["predicted"])
                tp += len(gold & predicted)
                fp += len(predicted - gold)
                fn += len(gold - predicted)
            precision, recall, score = f1(tp, fp, fn)
            metrics[dim] = {
                "samples": len(reviewed), "exact_set_accuracy": round(exact / len(reviewed), 4),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": score,
            }
        else:
            confusion = Counter()
            for item in reviewed:
                gold = item[dim]["gold"]
                predicted = item[dim]["predicted"]
                g = gold[0]
                p = predicted[0] if len(predicted) == 1 else "__none__" if not predicted else "__multiple__"
                confusion[(g, p)] += 1
            metrics[dim] = {
                "samples": len(reviewed), "top1_accuracy": round(exact / len(reviewed), 4),
                "confusion_matrix": {
                    gold: dict(sorted((pred, count) for (g, pred), count in confusion.items() if g == gold))
                    for gold in sorted({g for g, _ in confusion})
                },
            }

    complete = (
        annotation_ids == expected_ids
        and len(reviewed) == len(expected_ids)
        and not pending
        and not excluded
    )
    result = {
        "ok": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "annotations": str(args.annotations),
        "template": str(args.template),
        "qwen_report": str(args.qwen_report),
        "aliases": str(args.aliases),
        "expected_samples": len(expected_ids),
        "annotated_samples": len(annotation_ids),
        "missing_samples": len(missing_ids),
        "missing_sample_ids": missing_ids,
        "extra_samples": 0,
        "reviewed_samples": len(reviewed),
        "pending_samples": len(pending),
        "excluded_samples": excluded,
        "complete": complete,
        "per_dimension_metrics": metrics,
        "macro_exact_accuracy": round(sum(value.get("exact_set_accuracy", value.get("top1_accuracy", 0.0)) for value in metrics.values()) / len(metrics), 4),
        "overall_field_exact_accuracy": round(total_exact / (len(reviewed) * len(DIMS)), 4),
        "details": reviewed,
    }
    output = args.output or ROOT / "data/egolife/runs" / f"category_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "ok", "expected_samples", "annotated_samples", "missing_samples", "reviewed_samples",
        "pending_samples", "excluded_samples", "complete", "per_dimension_metrics",
        "macro_exact_accuracy", "overall_field_exact_accuracy"
    )}, ensure_ascii=False, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()

