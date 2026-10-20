"""Replay the Feishu custom test set into a scratch run directory.

Uses the existing perception/transcript caches so it needs no GPU and cannot
touch the live session output under data/egolife/runs/interactive_mem.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(HERE))

REAL_RUNS = ROOT / "data" / "egolife" / "runs"
CACHE_FILES = (
    "qwen_cache_memory_prompt_v3.json",
    "interactive_audio_transcript_cache.json",
    "interactive_scene_retrieval_cache.sqlite3",
    "interactive_scene_plan_cache.sqlite3",
    "interactive_object_grounding_cache.sqlite3",
    "interactive_person_profile_cache.json",
)


def make_scratch_runs(scratch: Path) -> Path:
    runs = scratch / "runs"
    (runs / "interactive_mem").mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES:
        source, link = REAL_RUNS / name, runs / name
        if source.exists() and not link.exists():
            link.symlink_to(source)
    return runs


def summarize(memory: dict) -> dict:
    from entity_records import EntityRecordIndex

    moments = [m for m in (memory.get("layer7") or {}).get("moments", {}).values()
               if isinstance(m, dict) and not m.get("merged_into")]
    rows = []
    for moment in moments:
        facts = [f for f in moment.get("temporal_facts") or [] if isinstance(f, dict)]
        observations = [p for p in moment.get("person_observations") or [] if isinstance(p, dict)]
        segment = ((moment.get("source_provenance") or {}).get("source_segment_id")
                   or moment.get("id") or "")
        rows.append({
            "segment": segment,
            "people": moment.get("people") or [],
            "person_observations": len(observations),
            "spoken_aliases": sum(1 for p in observations if p.get("spoken_aliases")),
            "relationships": sum(1 for p in observations if p.get("relationships")),
            "facts_visual": sum(1 for f in facts if f.get("modality") == "visual"),
            "facts_speech": sum(1 for f in facts if f.get("modality") == "speech"),
            "facts_with_affected": sum(1 for f in facts if f.get("affected_people")),
            "facts_with_move": sum(1 for f in facts
                                   if f.get("from_location") or f.get("to_location")),
            "source_evidence_chars": len(str(moment.get("source_evidence") or "")),
            "interaction_chars": len(str(moment.get("interaction_details") or "")),
            "objects": len(moment.get("objects") or []),
            "environments": len(moment.get("environments") or []),
            "location": bool(moment.get("location")),
        })
    records = EntityRecordIndex(memory.get("layer7") or {}).records
    people = [r for r in records if r.get("kind") == "person"]

    def observations(record):
        return [row for event in record.get("events") or []
                for row in event.get("person_observations") or [] if isinstance(row, dict)]

    by_kind = {}
    for record in records:
        by_kind[record.get("kind")] = by_kind.get(record.get("kind"), 0) + 1
    return {
        "moments": len(moments),
        "per_moment": sorted(rows, key=lambda row: row["segment"]),
        "totals": {key: sum(row[key] for row in rows) for key in (
            "person_observations", "spoken_aliases", "relationships", "facts_visual",
            "facts_speech", "facts_with_affected", "facts_with_move", "objects",
            "environments")},
        "entity_records": {
            "total": len(records),
            "by_kind": by_kind,
            "person": len(people),
            "person_with_spoken_alias": sum(
                1 for r in people if any(x.get("spoken_aliases") for x in observations(r))),
            "person_with_relationship": sum(
                1 for r in people if any(x.get("relationships") for x in observations(r))),
            "person_with_readable_name": sum(
                1 for r in people
                if any(x.get("readable_name") and x.get("name_evidence")
                       for x in observations(r))),
            "person_names": sorted({str(r.get("name")) for r in people})[:60],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="scratch run name")
    parser.add_argument("--limit", type=int, default=0, help="max clips (0 = all)")
    parser.add_argument("--only", default="", help="comma-separated segment ids")
    parser.add_argument("--capture-media", action="store_true",
                        help="also write multimodal assets (default off, faster)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="re-run model inference instead of reusing source-matched caches")
    parser.add_argument("--scratch-root",
                        default=str(ROOT / "tmp" / "offline_replay"))
    args = parser.parse_args()

    scratch = Path(args.scratch_root) / args.label
    if not args.capture_media:
        os.environ["MEMORY_CAPTURE_MULTIMODAL"] = "0"
    runs = make_scratch_runs(scratch)

    import interactive_runner as ir
    ir.RUNS = runs
    wanted = {item.strip() for item in args.only.split(",") if item.strip()}

    session = ir.InteractiveSession("custom_test", dataset="feishu:all")
    if wanted:
        keep = [index for index, step in enumerate(session.steps)
                if any(input_id in wanted
                       for input_id in [str(item.get("id") or "")
                                        for item in (step.get("inputs") or [])])]
        session.steps = [session.steps[index] for index in keep]
        session.pos = 0
    done = 0
    while session.pos < len(session.steps):
        if args.limit and done >= args.limit:
            break
        message = session.run_step(force=args.force_refresh)
        done += 1
        print(f"[{done}] {message}", flush=True)

    memory = json.loads((runs / "interactive_mem" / "current" / "memory.json")
                        .read_text(encoding="utf-8"))
    report = summarize(memory)
    out = scratch / "summary.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"totals": report["totals"],
                      "entity_records": report["entity_records"]},
                     ensure_ascii=False, indent=2))
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
