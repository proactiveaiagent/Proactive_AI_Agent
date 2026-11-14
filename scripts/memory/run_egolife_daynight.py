"""Run one EgoLife day through real Qwen perception, daytime writes, and nightly consolidation."""
import argparse
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import PersonMemory, _is_valid_index_tag, _is_valid_person_tag
from daytime_processor import DaytimeProcessor
from nighttime_processor import NightlyProcessor
from qwen_perceive import QwenPerceiver

RUNS = ROOT / "data/egolife/runs"
DATA = ROOT / "data/egolife/testset/long_term/full_day.jsonl"


def load_rows():
    return [json.loads(line) for line in DATA.open(encoding="utf-8") if line.strip()]


def unique_events(rows):
    # The long-term set has en/cn labels for identical video/time/transcript inputs.
    # Keep one event per exact perception input so bilingual annotations do not
    # become duplicate memories.
    seen, unique = set(), []
    for row in rows:
        signature = (row.get("video"), row.get("start_time"), row.get("end_time"),
                     row.get("transcript", ""))
        if signature not in seen:
            seen.add(signature)
            unique.append(row)
    return unique


def event_timestamp(moment, day):
    parts = str(moment.get("id", "")).split("_")
    hhmmss = parts[3][:6] if len(parts) > 3 and parts[3][:6].isdigit() else "120000"
    return f"{day}T{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="Perception sample limit; 0 runs full day")
    ap.add_argument("--force", action="store_true", help="Re-run Qwen and ignore cache")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--day", default=date.today().isoformat())
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows()
    rows = unique_events(all_rows)
    if args.limit > 0:
        rows = rows[:args.limit]
    perceiver = QwenPerceiver(api_url=args.api_url)
    health = perceiver.health()
    if not health.get("model_loaded"):
        raise RuntimeError(f"Qwen API not ready: {health}")

    report = {"day": args.day, "model": health, "dataset_rows": len(all_rows),
              "unique_events_selected": len(rows), "deduplicated_input_rows": len(all_rows) - len(unique_events(all_rows)),
              "perception_source_counts": {}, "failed_samples": [], "index_integrity_errors": [], "rejected_tags": {}}
    start_all = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="egolife_daynight_", dir=RUNS) as temp:
        memory = PersonMemory(memory_dir=temp)
        daytime = DaytimeProcessor(memory)
        daytime.start_scene(f"egolife_{args.day}")
        stage_start = time.perf_counter()
        for i, row in enumerate(rows, 1):
            perceived = perceiver.perceive(row, force=args.force)
            source = perceived.get("source", "unknown")
            report["perception_source_counts"][source] = report["perception_source_counts"].get(source, 0) + 1
            fields = perceived.get("fields") or {}
            if not fields or not (fields.get("scene") or fields.get("user_action")):
                report["failed_samples"].append({"id": row.get("id"), "error": perceived.get("error")})
                continue
            mid = daytime.ingest(
                scene=fields.get("scene") or "",
                user_action=fields.get("user_action") or "",
                needs=fields.get("needs") or [], solutions=fields.get("solutions") or [],
                people=fields.get("people") or None,
                location=fields.get("location"), activity=fields.get("activity"),
                environments=fields.get("environments") or None,
                objects=fields.get("objects") or None,
                extra_notes=(fields.get("time_occasion") or "")[:200],
                timestamp=event_timestamp(row, args.day),
            )
            moment = memory.memory["layer7"]["moments"][mid]
            if mid not in memory.memory["layer7"]["time_nodes"].get(args.day, []):
                report["index_integrity_errors"].append({"id": row.get("id"), "dimension": "time_nodes", "tag": args.day})
            for dim, key in (("people", "people"), ("locations", "location"),
                             ("activity_events", "activity"), ("environments", "environments"),
                             ("objects", "objects")):
                tags = moment.get(key) or []
                tags = tags if isinstance(tags, list) else [tags]
                for tag in tags:
                    valid = (_is_valid_person_tag(tag) if dim == "people"
                             else _is_valid_index_tag(dim, tag))
                    if not valid:
                        report["rejected_tags"][dim] = report["rejected_tags"].get(dim, 0) + 1
                    elif mid not in memory.memory["layer7"][dim].get(tag, []):
                        report["index_integrity_errors"].append({"id": row.get("id"), "dimension": dim, "tag": tag})
            if i % 100 == 0 or i == len(rows):
                print(f"[daytime] {i}/{len(rows)} clips; {len(report['failed_samples'])} failures", flush=True)
        report["daytime_seconds"] = round(time.perf_counter() - stage_start, 2)
        report["short_term_cleared"] = daytime.end_scene()
        before_nightly = (memory.memory_dir / "memory.json").stat().st_size
        stage_start = time.perf_counter()
        nightly = NightlyProcessor(memory, qwen_api_url=args.api_url).run(day=args.day)
        report["nightly_seconds"] = round(time.perf_counter() - stage_start, 2)
        report["nightly"] = nightly
        after_nightly = (memory.memory_dir / "memory.json").stat().st_size
        report["memory_json_bytes"] = {"before_nightly": before_nightly, "after_nightly": after_nightly}
        report["layer_counts"] = {
            "layer1": len(memory.memory["layer1"]), "layer2": len(memory.memory["layer2"]),
            "layer3": len(memory.memory["layer3"]),
            "layer7_moments": len(memory.memory["layer7"]["moments"]),
            "layer7_index_keys": {k: len(memory.memory["layer7"].get(k, {})) for k in
                ("people", "locations", "time_nodes", "activity_events", "environments", "objects")},
            "layer4_summary_chars": len(memory.memory["layer4"].get("summary", "")),
            "layer5_summary_chars": len(memory.memory["layer5"].get("summary", "")),
            "profile_fields_with_values": [k for k, v in memory.get_profile().items() if v],
        }
        replay = NightlyProcessor(memory, qwen_api_url=args.api_url).run(day=args.day)
        report["idempotency_replay"] = replay.get("skipped")
        report["ok"] = (not report["failed_samples"] and not report["index_integrity_errors"]
                         and bool(nightly.get("ok")) and report["idempotency_replay"] == "unchanged")
    report["elapsed_seconds"] = round(time.perf_counter() - start_all, 2)
    out = RUNS / f"daynight_e2e_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report_path": str(out),
                      "dataset_rows": report["dataset_rows"], "unique_events_selected": report["unique_events_selected"],
                      "deduplicated_input_rows": report["deduplicated_input_rows"], "failed_samples": len(report["failed_samples"]),
                      "nightly_ok": report["nightly"].get("ok"), "profile_updated": report["nightly"].get("profile_updated"),
                      "nightly_skipped_on_replay": report["idempotency_replay"],
                      "layer_counts": report["layer_counts"], "memory_json_bytes": report["memory_json_bytes"],
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()