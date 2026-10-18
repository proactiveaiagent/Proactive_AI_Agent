#!/usr/bin/env python3
"""Validate frozen Feishu retrieval annotations without scoring incomplete labels."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from prepare_feishu_retrieval_annotations import (
    DATASET,
    SPLIT_VERSION,
    load_manifest,
    split_by_scenario,
)

SCHEMA_VERSION = 2
DEFAULT_INPUT = DATASET / "retrieval_annotations.template.jsonl"
FACT_TYPES = {"person", "object", "location", "activity", "environment", "event"}
REVIEW_STATES = {"unreviewed", "annotated", "adjudicated"}


def read_jsonl(path):
    rows = []
    errors = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line {line_number}: row must be a JSON object")
                continue
            row["_line_number"] = line_number
            rows.append(row)
    return rows, errors


def string_list(row, key, label, errors, blockers, required_when_reviewed=False):
    value = row.get(key)
    line = row.get("_line_number", "?")
    if value is None:
        if required_when_reviewed and row.get("review_status") in {"annotated", "adjudicated"}:
            blockers[f"{label} missing"] += 1
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(f"line {line}: {key} must be null or an array of non-empty strings")
        return None
    if len(set(value)) != len(value):
        errors.append(f"line {line}: {key} contains duplicate IDs")
    return value


def validate(path, split):
    manifest, entries = load_manifest()
    expected = {str(entry["clip_id"]): entry for entry in entries}
    frozen = split_by_scenario(entries)
    rows, errors = read_jsonl(path)
    row_by_id = {}

    for row in rows:
        line = row["_line_number"]
        row_blockers = Counter()
        clip_id = row.get("query_clip_id")
        if not isinstance(clip_id, str) or clip_id not in expected:
            errors.append(f"line {line}: query_clip_id is missing or absent from manifest")
            continue
        if clip_id in row_by_id:
            errors.append(f"line {line}: duplicate query_clip_id {clip_id}")
            continue
        row_by_id[clip_id] = row
        source = expected[clip_id]
        if row.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"line {line}: expected schema_version {SCHEMA_VERSION}")
        if row.get("query_id") != f"feishu:{clip_id}":
            errors.append(f"line {line}: query_id does not match query_clip_id")
        if row.get("category") != source.get("scenario"):
            errors.append(f"line {line}: category differs from manifest")
        if row.get("input_video") != source.get("relative_path"):
            errors.append(f"line {line}: input_video differs from manifest")
        if row.get("split_scheme") != SPLIT_VERSION or row.get("split") != frozen[clip_id]:
            errors.append(f"line {line}: frozen split mismatch for {clip_id}")

        state = row.get("review_status")
        if state not in REVIEW_STATES:
            errors.append(f"line {line}: invalid review_status")
        if state in {"annotated", "adjudicated"} and not str(row.get("annotator") or "").strip():
            row_blockers["annotator missing"] += 1
        if state == "adjudicated" and not str(row.get("adjudicator") or "").strip():
            row_blockers["adjudicator missing"] += 1
        annotator = str(row.get("annotator") or "").strip()
        adjudicator = str(row.get("adjudicator") or "").strip()
        if state == "adjudicated" and annotator and adjudicator and annotator.casefold() == adjudicator.casefold():
            row_blockers["adjudicator must be a second person"] += 1
        if state == "unreviewed":
            row_blockers["not reviewed"] += 1
        elif state == "annotated":
            row_blockers["not adjudicated"] += 1

        history = string_list(row, "history_clip_ids", "history", errors, row_blockers, True)
        relevant = string_list(row, "relevant_memory_clip_ids", "relevant memories", errors, row_blockers, True)
        negatives = string_list(row, "hard_negative_clip_ids", "hard negatives", errors, row_blockers, True)
        for field, values in (("history_clip_ids", history), ("relevant_memory_clip_ids", relevant), ("hard_negative_clip_ids", negatives)):
            if values is not None:
                unknown = sorted(set(values) - set(expected))
                if unknown:
                    errors.append(f"line {line}: {field} references unknown clips: {unknown}")
        if history is not None and clip_id in history:
            errors.append(f"line {line}: query clip cannot be part of its own history")
        if history is not None and relevant is not None and not set(relevant).issubset(history):
            errors.append(f"line {line}: relevant memories must be available in history")
        if history is not None and negatives is not None and not set(negatives).issubset(history):
            errors.append(f"line {line}: hard negatives must be available in history")
        if relevant is not None and negatives is not None and set(relevant) & set(negatives):
            errors.append(f"line {line}: relevant and hard-negative clip sets overlap")

        facts = row.get("facts_to_retrieve")
        if not isinstance(facts, list):
            errors.append(f"line {line}: facts_to_retrieve must be an array")
            facts = []
        evidence = row.get("source_evidence")
        if not isinstance(evidence, list):
            errors.append(f"line {line}: source_evidence must be an array")
            evidence = []
        if state == "adjudicated" and relevant and not facts:
            row_blockers["relevant memories lack fact labels"] += 1
        if facts and not evidence:
            row_blockers["facts lack source evidence"] += 1

        fact_sources = set()
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                errors.append(f"line {line}: fact {index} must be an object")
                continue
            if not isinstance(fact.get("fact_type"), str) or fact.get("fact_type") not in FACT_TYPES:
                errors.append(f"line {line}: fact {index} has invalid fact_type")
            for key in ("entity", "fact", "source_clip_id"):
                if not isinstance(fact.get(key), str) or not fact[key].strip():
                    errors.append(f"line {line}: fact {index} requires non-empty {key}")
            source_id = fact.get("source_clip_id")
            if isinstance(source_id, str):
                fact_sources.add(source_id)
                if source_id not in expected:
                    errors.append(f"line {line}: fact {index} references unknown source clip")
                if relevant is not None and source_id not in relevant:
                    errors.append(f"line {line}: fact {index} source must be a relevant memory")
        evidence_sources = set()
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"line {line}: evidence {index} must be an object")
                continue
            source_id = item.get("clip_id")
            if not isinstance(source_id, str) or source_id not in expected:
                errors.append(f"line {line}: evidence {index} references unknown clip")
                continue
            evidence_sources.add(source_id)
            start, end = item.get("start_s"), item.get("end_s")
            if (
                not isinstance(start, (int, float))
                or isinstance(start, bool)
                or not isinstance(end, (int, float))
                or isinstance(end, bool)
                or not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end < start
            ):
                errors.append(f"line {line}: evidence {index} requires a valid non-negative time range")
            if not isinstance(item.get("modality"), str) or item.get("modality") not in {"video", "audio"}:
                errors.append(f"line {line}: evidence {index} modality must be video or audio")
            if relevant is not None and source_id not in relevant:
                errors.append(f"line {line}: evidence must refer to a relevant memory")
        if fact_sources and not fact_sources.issubset(evidence_sources):
            errors.append(f"line {line}: every fact source must have source_evidence")
        row["_scoring_blockers"] = row_blockers

    missing = sorted(set(expected) - set(row_by_id))
    extra_count = len(rows) - len(row_by_id)
    if missing:
        errors.append(f"annotation file is missing {len(missing)} manifest clips")
    if extra_count:
        errors.append(f"annotation file contains {extra_count} duplicate or invalid rows")

    targets = [row_by_id[clip_id] for clip_id in row_by_id if frozen[clip_id] == split]
    adjudicated = sum(row.get("review_status") == "adjudicated" for row in targets)
    target_blockers = Counter()
    for row in targets:
        target_blockers.update(row.get("_scoring_blockers", {}))
    ready = bool(targets) and not errors and adjudicated == len(targets) and not target_blockers
    return {
        "input": str(path),
        "schema_version": SCHEMA_VERSION,
        "split_scheme": SPLIT_VERSION,
        "target_split": split,
        "manifest_rows": len(expected),
        "annotation_rows": len(rows),
        "target_rows": len(targets),
        "adjudicated_target_rows": adjudicated,
        "structural_error_count": len(errors),
        "structural_errors": errors,
        "scoring_blocker_counts": dict(sorted(target_blockers.items())),
        "ready_for_scoring": ready,
        "accuracy_computed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = validate(args.input, args.split)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["structural_error_count"]:
        return 1
    if args.require_ready and not report["ready_for_scoring"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
