"""Build the visualization replay from actual Qwen memory writes, never GT fields."""
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts/memory"))
from memory import PersonMemory, LAYER7_INDICES
from run_scenarios import snapshot, build_canonical_map
from qwen_perceive import extract_english_transcript

TEST = ROOT / "data/egolife/testset"
RUNS = ROOT / "data/egolife/runs"
CACHE_PATH = Path(os.environ.get("EGOLIFE_QWEN_CACHE", str(RUNS / "qwen_cache_memory_prompt_v3.json")))
BACKUP = RUNS / "prompt_v1_baseline"


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def timestamp_from_id(sample_id):
    parts = str(sample_id).split("_")
    stamp = parts[3] if len(parts) > 3 and len(parts[3]) >= 6 else "12000000"
    return f"{date.today().isoformat()}T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}"


def add_qwen_moment(mem, row, cache):
    sample_id = row.get("id", "")
    fields = (cache.get(sample_id) or {}).get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"Qwen cache has no parsed memory for {sample_id}")
    mid = mem.add(
        scene=str(fields.get("scene") or ""),
        user_action=str(fields.get("user_action") or ""),
        needs=fields.get("needs") if isinstance(fields.get("needs"), list) else [],
        solutions=fields.get("solutions") if isinstance(fields.get("solutions"), list) else [],
        people=fields.get("people") if isinstance(fields.get("people"), list) else None,
        location=str(fields.get("location") or "") or None,
        activity=str(fields.get("activity") or "") or None,
        environments=fields.get("environments") if isinstance(fields.get("environments"), list) else None,
        objects=fields.get("objects") if isinstance(fields.get("objects"), list) else None,
        extra_notes=str(fields.get("time_occasion") or ""),
        timestamp=timestamp_from_id(sample_id),
        source_evidence=str(fields.get("source_evidence") or
                            extract_english_transcript(row.get("transcript") or "")),
    )
    return mid


def backup_v1_outputs():
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in ("short_term_snapshots.json", "short_term_memory.json",
                 "long_term_snapshots.json", "long_term_memory.json"):
        src = RUNS / name
        dst = BACKUP / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def run_short_term(cache, stage_dir):
    rows = read_jsonl(TEST / "short_term/moments_scattered.jsonl")
    plan = json.loads((TEST / "short_term/crud_plan.json").read_text(encoding="utf-8"))
    missing = [r["id"] for r in rows if r["id"] not in cache]
    if missing:
        raise ValueError(f"Short-term cache missing {len(missing)} rows: {missing[:5]}")

    mem = PersonMemory(memory_dir=str(stage_dir / "short_term_mem_qwen_v3"))
    snaps = [snapshot(mem, "初始状态 · Qwen V3")]
    id_map, source_map = {}, {}
    for i, row in enumerate(rows, 1):
        mid = add_qwen_moment(mem, row, cache)
        id_map[row["id"]] = mid
        source_map[mid] = row["id"]
        if i % 10 == 0 or i == len(rows):
            snaps.append(snapshot(mem, f"Qwen 写入 {i} 条", action=f"add #{i}"))

    query_results = []
    for q in plan.get("query", []):
        hits = mem.query(q["query"], top_k=5)
        query_results.append({
            "query": q["query"], "returned_count": len(hits),
            "source_ids": [source_map.get(h.get("id"), "") for h in hits],
            "memory_ids": [h.get("id", "") for h in hits],
        })
        snaps.append(snapshot(mem, f"检索「{q['query']}」", action="query"))

    for case in plan.get("update", []):
        mid = id_map.get(case["id"])
        if mid:
            mem.update(mid, {case["field"]: case["new_value"]})
            snaps.append(snapshot(mem, f"用户纠正 {case['field']}", action="update"))

    for sample_id in plan.get("highlight", []):
        if id_map.get(sample_id):
            mem.highlight(id_map[sample_id])
    snaps.append(snapshot(mem, "用户标记重要记忆", action="highlight"))

    deleted, protected = [], []
    for case in plan.get("delete", []):
        mid = id_map.get(case["id"])
        if mid and mem.delete(mid, reason=case.get("reason", "user request")):
            deleted.append(case["id"])
        elif mid:
            protected.append(case["id"])
    snaps.append(snapshot(mem, "按用户请求清理记忆", action="delete"))

    highlight_ids = [id_map[s] for s in plan.get("highlight", []) if id_map.get(s)]
    protection_check = None
    if highlight_ids:
        blocked = not mem.delete(highlight_ids[0], reason="保护规则验证")
        forced = mem.delete(highlight_ids[0], reason="用户强制删除验证", force=True)
        protection_check = {"protected_by_default": blocked, "force_delete_succeeded": forced}
    snaps.append(snapshot(mem, "CRUD 最终状态", action="final"))
    mem._save()

    out = {
        "scenario": "short_term",
        "desc": "60 个不连续视频片段 · 实际 Qwen V3 记忆 + 用户 CRUD 操作",
        "memory_source": str(CACHE_PATH),
        "ground_truth_content_used": False,
        "moments_total": len(rows),
        "crud_summary": {
            "add": len(id_map), "query": len(query_results),
            "update": len(plan.get("update", [])),
            "highlight": len(plan.get("highlight", [])),
            "delete": len(deleted),
            "deleted_ids": deleted, "protected_delete_ids": protected,
            "highlight_protection_check": protection_check,
        },
        "query_results": query_results,
        "source_id_map": source_map,
        "snapshots": snaps,
        "final": snapshot(mem, "最终状态"),
    }
    (stage_dir / "short_term_snapshots.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "short_term_memory.json").write_text(
        json.dumps(mem.memory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"short-term: {len(rows)} Qwen moments; CRUD={out['crud_summary']}")


def build_observed_summary(mem, participant, day, source_count):
    archive = mem.memory["layer7"]
    people = Counter({k: len(v) for k, v in archive.get("people", {}).items()})
    locations = Counter({k: len(v) for k, v in archive.get("locations", {}).items()})
    activities = Counter({k: len(v) for k, v in archive.get("activity_events", {}).items()})
    top_people = ", ".join(f"{k} ({n})" for k, n in people.most_common(4)) or "none identified"
    top_locations = ", ".join(f"{k} ({n})" for k, n in locations.most_common(4)) or "none identified"
    top_activities = ", ".join(f"{k} ({n})" for k, n in activities.most_common(4)) or "none identified"

    moments = list(archive.get("moments", {}).values())
    moments.sort(key=lambda m: (m.get("timestamp", ""), m.get("id", "")))
    key_events = []
    for m in moments:
        action = str(m.get("user_action") or m.get("scene") or "").strip()
        if action and action not in key_events:
            clock = str(m.get("timestamp") or "")[11:16]
            key_events.append(f"{clock} — {action[:180]}")
        if len(key_events) >= 8:
            break

    return {
        "layer4": {
            "summary": (f"{participant}, {day}: {source_count} Qwen-perceived event clips. "
                        f"Frequently observed people: {top_people}. "
                        f"Observed settings: {top_locations}. "
                        f"Activities: {top_activities}."),
            "current_tasks": [],
            "source_moment_count": source_count,
        },
        "layer5": {
            "summary": (f"Archive contains {day} only. Cross-day habits and stable long-term "
                        "patterns were not inferred from this single-day sample."),
            "key_events": key_events,
            "long_term_patterns": [],
            "source_moment_count": source_count,
        },
    }


def build_qwen_timeline(rows, cache):
    hours = defaultdict(lambda: {
        "count": 0, "people": Counter(), "locations": Counter(),
        "activities": Counter(), "first_id": None, "last_id": None,
    })
    for row in rows:
        sample_id = row["id"]
        hour = sample_id.split("_")[3][:2] if len(sample_id.split("_")) > 3 else "00"
        key = f"{hour}:00"
        segment = hours[key]
        segment["count"] += 1
        segment["last_id"] = sample_id
        if segment["first_id"] is None:
            segment["first_id"] = sample_id
        fields = cache[sample_id]["fields"]
        for person in fields.get("people") or []:
            segment["people"][str(person)] += 1
        location = fields.get("location")
        if location:
            segment["locations"][str(location)] += 1
        activity = fields.get("activity")
        if activity:
            segment["activities"][str(activity)] += 1
    return [{
        "hour": hour, "count": v["count"],
        "people": v["people"].most_common(5),
        "locations": v["locations"].most_common(5),
        "activities": v["activities"].most_common(5),
        "first_id": v["first_id"], "last_id": v["last_id"],
    } for hour, v in sorted(hours.items())]


def run_long_term(cache, stage_dir):
    rows = [r for r in read_jsonl(TEST / "long_term/full_day.jsonl")
            if str(r.get("id", "")).endswith("_en")]
    missing = [r["id"] for r in rows if r["id"] not in cache]
    if missing:
        raise ValueError(f"Long-term cache missing {len(missing)} rows: {missing[:5]}")
    meta = json.loads((TEST / "long_term/timeline_meta.json").read_text(encoding="utf-8"))
    participant = meta.get("participant", "unknown")
    day = meta.get("day", "DAY1")
    mem = PersonMemory(memory_dir=str(stage_dir / "long_term_mem_qwen_v3"))
    snaps = [snapshot(mem, "初始状态 · Qwen V3")]
    id_map, source_map = {}, {}
    current_hour = None
    for i, row in enumerate(rows, 1):
        mid = add_qwen_moment(mem, row, cache)
        id_map[row["id"]] = mid
        source_map[mid] = row["id"]
        hour = row["id"].split("_")[3][:2]
        if current_hour is not None and hour != current_hour:
            snaps.append(snapshot(mem, f"{current_hour}:00 结束", action="hour_end"))
        current_hour = hour
    if current_hour is not None:
        snaps.append(snapshot(mem, f"{current_hour}:00 结束（白天结束）", action="day_end"))
    before = snapshot(mem, "夜间整理前")

    # L7 tags already come from each Qwen memory write; rebuild/sort from those tags only.
    analysis = {key: {tag: list(ids) for tag, ids in mem.memory["layer7"].get(key, {}).items()}
                for key in LAYER7_INDICES}
    mem.sort(analysis)
    snaps.append(snapshot(mem, "夜间整理 · Qwen 六维索引 sort", action="sort"))

    canonical_map = build_canonical_map(mem)
    if canonical_map:
        mem.combine(canonical_map)
    snaps.append(snapshot(mem, "夜间整理 · 合并 combine", action="combine"))

    summary = build_observed_summary(mem, participant, day, len(rows))
    mem.compress(summary)
    snaps.append(snapshot(mem, "夜间整理 · 证据约束摘要 compress", action="compress"))

    # One day cannot support stable profile claims; keep L6 empty and explicit.
    mem.memory["layer6"]["summary"] = (
        "No stable user-profile attributes inferred: this replay covers one day only.")
    mem._save()
    snaps.append(snapshot(mem, "画像检查 · 跨日证据不足，未生成属性", action="profile_hold"))

    after = snapshot(mem, "夜间整理后")
    integrity_errors = []
    l7 = mem.memory["layer7"]
    valid_ids = set(l7.get("moments", {}))
    for key in LAYER7_INDICES:
        for tag, ids in l7.get(key, {}).items():
            for mid in ids:
                if mid not in valid_ids:
                    integrity_errors.append({"index": key, "tag": tag, "moment_id": mid})

    out = {
        "scenario": "long_term",
        "desc": f"{participant} {day} · 725 个 Qwen V3 事件记忆片段",
        "memory_source": str(CACHE_PATH),
        "ground_truth_content_used": False,
        "participant": participant, "day": day,
        "moments_total": len(rows),
        "hours_covered": sorted({x["hour"] for x in build_qwen_timeline(rows, cache)}),
        "timeline_segments": build_qwen_timeline(rows, cache),
        "before_consolidation": before,
        "after_consolidation": after,
        "index_integrity_errors": integrity_errors,
        "source_id_map": source_map,
        "snapshots": snaps,
        "final": after,
    }
    mem._save()
    (stage_dir / "long_term_snapshots.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "long_term_memory.json").write_text(
        json.dumps(mem.memory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"long-term: {len(rows)} Qwen moments; L7 index integrity errors={len(integrity_errors)}")


def main():
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    short_rows = read_jsonl(TEST / "short_term/moments_scattered.jsonl")
    long_rows = [r for r in read_jsonl(TEST / "long_term/full_day.jsonl")
                 if str(r.get("id", "")).endswith("_en")]
    required = {r["id"] for r in short_rows + long_rows}
    missing = required - set(cache)
    if missing:
        raise SystemExit(f"Qwen V3 cache incomplete: {len(missing)} required rows missing; examples: {sorted(missing)[:8]}")
    backup_v1_outputs()
    RUNS.mkdir(parents=True, exist_ok=True)
    output_names = ("short_term_snapshots.json", "short_term_memory.json",
                    "long_term_snapshots.json", "long_term_memory.json")
    with tempfile.TemporaryDirectory(prefix="qwen_v3_snapshots_", dir=RUNS) as stage:
        stage_dir = Path(stage)
        run_short_term(cache, stage_dir)
        run_long_term(cache, stage_dir)
        for name in output_names:
            os.replace(stage_dir / name, RUNS / name)
    print(json.dumps({"memory_source": str(CACHE_PATH), "short_rows": len(short_rows),
                      "long_unique_rows": len(long_rows), "snapshots_committed_together": True},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
