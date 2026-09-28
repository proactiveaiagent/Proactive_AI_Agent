#!/usr/bin/env python3
"""Re-perceive official QA target clips into a separate, resumable cache.

The baseline cache remains immutable. Only source video and source transcript
are passed into Qwen; official labels and answers are used to select benchmark
clips but are never added to model inputs or memory entries.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/egolife/runs"
TEST = ROOT / "data/egolife/testset"
SOURCE_CACHE = RUNS / "qwen_cache_expanded_eval_source_evidence.json"
OUTPUT_CACHE = RUNS / "qwen_cache_expanded_eval_v3_official_targets.json"
LONG_QA = TEST / "long_term/egolifeqa_official_a1_jake.json"
LONG_CLIPS = TEST / "long_term/multiday_qa_clipset.jsonl"
SHORT_CLIPS = TEST / "short_term/official_video_qa_v1_clips_84.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def make_target_ids():
    sys.path.insert(0, str(ROOT / "scripts/memory"))
    from evaluate_egolifeqa_multiday_memory import parse_target_points, parse_stamp

    intervals = {}
    for row in read_jsonl(LONG_CLIPS):
        match = re.fullmatch(r"DAY([1-7])_A1_JAKE_(\d{8})_en", str(row.get("id", "")))
        if not match:
            continue
        day, stamp = int(match.group(1)), match.group(2)
        start = parse_stamp(stamp)
        if start is not None:
            intervals.setdefault(day, []).append((start, start + 30.0, row["id"]))

    long_targets = set()
    questions = json.loads(LONG_QA.read_text(encoding="utf-8"))
    for question in questions:
        points, _ = parse_target_points(question.get("target_time"))
        for point in points:
            for start, end, sample_id in intervals.get(point["day"], []):
                if start <= point["seconds"] < end:
                    long_targets.add(sample_id)

    short_targets = {str(row.get("id", "")) for row in read_jsonl(SHORT_CLIPS)}
    short_targets.discard("")
    return long_targets, short_targets, len(questions)


def make_rows():
    paths = [
        TEST / "long_term/multiday_qa_clipset.jsonl",
        TEST / "short_term/moments_scattered_v2_120.jsonl",
        TEST / "long_term/multiday_qa_clipset_v1_520.jsonl",
        TEST / "short_term/moments_scattered.jsonl",
        ROOT / "data/egolife/structured/moments.jsonl",
    ]
    rows = {}
    for path in paths:
        if path.exists():
            for row in read_jsonl(path):
                rows.setdefault(str(row.get("id") or ""), row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--output-cache", type=Path, default=OUTPUT_CACHE)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many missing target clips; zero means all")
    parser.add_argument("--plan-only", action="store_true",
                        help="Print counts and validate source coverage without calling Qwen")
    args = parser.parse_args()
    if args.source_cache.resolve() == args.output_cache.resolve():
        raise SystemExit("Refusing to overwrite the source cache")

    source = json.loads(args.source_cache.read_text(encoding="utf-8"))
    long_targets, short_targets, question_count = make_target_ids()
    target_ids = sorted(long_targets | short_targets)
    rows = make_rows()
    missing_cache = [sample_id for sample_id in target_ids if sample_id not in source]
    missing_rows = [sample_id for sample_id in target_ids if sample_id not in rows]
    sys.path.insert(0, str(ROOT / "scripts/memory"))
    from qwen_perceive import PROMPT_VERSION, QwenPerceiver, video_path_of

    missing_videos = [sample_id for sample_id in target_ids
                      if sample_id in rows and video_path_of(rows[sample_id]) is None]
    plan = {
        "baseline_cache": str(args.source_cache),
        "output_cache": str(args.output_cache),
        "prompt_version": PROMPT_VERSION,
        "long_official_questions": question_count,
        "long_unique_target_clips": len(long_targets),
        "short_official_target_clips": len(short_targets),
        "unique_target_clips": len(target_ids),
        "target_ids_missing_from_baseline": len(missing_cache),
        "target_ids_missing_source_rows": len(missing_rows),
        "target_ids_missing_local_videos": len(missing_videos),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if missing_cache or missing_rows or missing_videos:
        raise SystemExit("Target source coverage is incomplete; refusing partial benchmark")
    if args.plan_only:
        return

    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_cache.with_suffix(".manifest.json")
    if args.output_cache.exists():
        output = json.loads(args.output_cache.read_text(encoding="utf-8"))
        if manifest_path.exists():
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old_manifest.get("baseline_cache") != str(args.source_cache.resolve()):
                raise SystemExit("Existing output cache was initialized from a different baseline")
            if old_manifest.get("target_ids_sha256") != __import__("hashlib").sha256(
                    "\n".join(target_ids).encode()).hexdigest():
                raise SystemExit("Existing output cache uses a different target set")
    else:
        output = source
        args.output_cache.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        import hashlib
        manifest = {
            **plan,
            "baseline_cache": str(args.source_cache.resolve()),
            "output_cache": str(args.output_cache.resolve()),
            "target_ids_sha256": hashlib.sha256("\n".join(target_ids).encode()).hexdigest(),
            "refresh_scope": "Officially labelled short-term and multi-day QA target clips are refreshed; all distractor memories are held at baseline fields.",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    perceiver = QwenPerceiver(cache_file=args.output_cache)
    health = perceiver.health()
    if not health.get("model_loaded"):
        raise SystemExit(f"Qwen service is unavailable: {health}")
    todo = [sample_id for sample_id in target_ids
            if (perceiver.cache.get(sample_id) or {}).get("prompt_version") != PROMPT_VERSION]
    if args.limit:
        todo = todo[:args.limit]
    real_save = perceiver._save_cache
    writes = 0

    def save_batched():
        nonlocal writes
        writes += 1
        if writes == 1 or writes % 5 == 0:
            real_save()

    perceiver._save_cache = save_batched
    failures = []
    started = time.time()
    for index, sample_id in enumerate(todo, 1):
        result = perceiver.perceive(rows[sample_id], force=True)
        if not result.get("fields"):
            failures.append({"id": sample_id, "error": result.get("error", "unknown")})
        if index == 1 or index % 10 == 0 or index == len(todo):
            print(json.dumps({
                "done": index,
                "total_this_run": len(todo),
                "refreshed_total": sum(
                    (perceiver.cache.get(item) or {}).get("prompt_version") == PROMPT_VERSION
                    for item in target_ids),
                "failed_this_run": len(failures),
                "latest_id": sample_id,
                "latest_ms": result.get("elapsed_ms"),
                "elapsed_min": round((time.time() - started) / 60, 1),
            }, ensure_ascii=False), flush=True)
    perceiver._save_cache = real_save
    real_save()
    failures_path = args.output_cache.with_name(args.output_cache.stem + "_failures.json")
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "finished_this_run": True,
        "cache": str(args.output_cache),
        "manifest": str(manifest_path),
        "refreshed_target_clips": sum(
            (perceiver.cache.get(item) or {}).get("prompt_version") == PROMPT_VERSION
            for item in target_ids),
        "target_clip_total": len(target_ids),
        "failed_this_run": len(failures),
        "failure_file": str(failures_path),
        "elapsed_min": round((time.time() - started) / 60, 1),
    }, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
