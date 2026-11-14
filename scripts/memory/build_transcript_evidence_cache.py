#!/usr/bin/env python3
"""Build a derived Qwen cache with speaker-labelled English transcript evidence.

The source cache is read-only. Official answers, captions, and labels are never
copied into memory; only the dataset source transcript is attached to each
matching cached clip.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/egolife/runs"
DEFAULT_CACHE = RUNS / "qwen_cache_expanded_eval.json"
DEFAULT_MOMENTS = [
    ROOT / "data/egolife/structured/moments.jsonl",
    ROOT / "data/egolife/testset/long_term/multiday_qa_clipset.jsonl",
    ROOT / "data/egolife/testset/long_term/multiday_qa_clipset_v1_520.jsonl",
    ROOT / "data/egolife/testset/short_term/moments_scattered_v2_120.jsonl",
    ROOT / "data/egolife/testset/short_term/moments_scattered.jsonl",
]
DEFAULT_OUTPUT = RUNS / "qwen_cache_expanded_eval_source_evidence.json"


def read_moments(paths):
    rows = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    rows.setdefault(str(row.get("id") or ""), row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--moments", type=Path, action="append",
        help="JSONL source file; may be repeated (defaults to structured + official clip sets)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-chars", type=int, default=4000)
    args = parser.parse_args()
    if args.cache.resolve() == args.output.resolve():
        raise SystemExit("Refusing to overwrite the source cache; choose a separate output path")

    import sys
    sys.path.insert(0, str(ROOT / "scripts/memory"))
    from qwen_perceive import extract_english_transcript

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    source_files = args.moments or DEFAULT_MOMENTS
    moments = read_moments(source_files)
    attached = missing_moment = missing_fields = empty_evidence = 0
    char_count = 0
    for sample_id, entry in cache.items():
        row = moments.get(str(sample_id))
        fields = entry.get("fields") if isinstance(entry, dict) else None
        if not row:
            missing_moment += 1
            continue
        if not isinstance(fields, dict):
            missing_fields += 1
            continue
        evidence = extract_english_transcript(
            row.get("transcript") or "", max_chars=args.max_chars)
        fields["source_evidence"] = evidence
        entry["source_evidence_version"] = "speaker-labelled-english-v1"
        if evidence:
            attached += 1
        else:
            empty_evidence += 1
        char_count += len(evidence)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_cache": str(args.cache),
        "source_moments": [str(path) for path in source_files],
        "output": str(args.output),
        "cached_clips": len(cache),
        "clips_with_source_evidence": attached,
        "clips_without_exact_source_moment": missing_moment,
        "clips_without_parsed_fields": missing_fields,
        "clips_with_empty_or_non_English_source_transcript": empty_evidence,
        "total_evidence_characters": char_count,
        "average_characters_per_cached_clip": round(char_count / max(len(cache), 1), 1),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
