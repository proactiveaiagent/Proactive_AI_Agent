"""Causal multi-day retrieval evaluation for official EgoLifeQA A1_JAKE.

This benchmark currently uses only cached Qwen clips. It supports query dates
DAY1-DAY7 and only adds a memory clip when its full 30-second interval ended
before the question timestamp, preventing within-clip future leakage.
"""
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory

RUNS = ROOT / "data/egolife/runs"
QA_PATH = ROOT / "data/egolife/testset/long_term/egolifeqa_official_a1_jake.json"
CACHE_PATH = Path(os.environ.get("EGOLIFE_QWEN_CACHE", str(RUNS / "qwen_cache.json")))
FPS = 30.0
CLIP_SECONDS = 30.0
TOP_K = 10
DAY_PATTERN = re.compile(r"DAY\s*([1-7])", re.I)
STAMP_PATTERN = re.compile(r"(?:(DAY\s*[1-7])[_-])?(\d{8})", re.I)


def normalize_day(value):
    match = DAY_PATTERN.search(str(value or ""))
    return int(match.group(1)) if match else None


def parse_stamp(value):
    value = str(value or "").strip()
    if not re.fullmatch(r"\d{8}", value):
        return None
    return (int(value[:2]) * 3600 + int(value[2:4]) * 60 +
            int(value[4:6]) + int(value[6:8]) / FPS)


def parse_target_points(target):
    if not isinstance(target, dict):
        return [], 0
    default_day = normalize_day(target.get("date"))
    raw_values = target.get("time_list")
    if not isinstance(raw_values, list):
        raw_values = [target.get("time", "")]
    points = []
    invalid = 0
    for raw in raw_values:
        text = str(raw or "")
        matches = list(STAMP_PATTERN.finditer(text))
        if not matches and re.fullmatch(r"\d{8}", text.strip()):
            matches = [None]
        if not matches:
            invalid += 1
            continue
        for match in matches:
            if match is None:
                explicit_day, stamp = None, text.strip()
            else:
                explicit_day, stamp = match.group(1), match.group(2)
            day = normalize_day(explicit_day) if explicit_day else default_day
            seconds = parse_stamp(stamp)
            if day is None or seconds is None:
                invalid += 1
                continue
            points.append({"day": day, "stamp": stamp, "seconds": seconds})
    unique = {(p["day"], p["stamp"]): p for p in points}
    return list(unique.values()), invalid


def iso_timestamp(day, stamp):
    synthetic_date = date(2026, 9, 1) + timedelta(days=day - 1)
    return f"{synthetic_date.isoformat()}T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}"


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))], 3)


def summarize(rows):
    if not rows:
        return {}
    count = len(rows)
    metrics = {}
    for k in (1, 5, 10):
        hit = 0
        recall = 0.0
        for row in rows:
            expected = set(row["expected_ids"])
            got = set(row["retrieved_ids"][:k])
            hit += bool(expected & got)
            recall += len(expected & got) / len(expected) if expected else 0
        metrics[f"evidence_hit_at_{k}"] = round(hit / count, 4)
        metrics[f"evidence_recall_at_{k}"] = round(recall / count, 4)
    rr = []
    for row in rows:
        rank = next((i + 1 for i, mid in enumerate(row["retrieved_ids"][:TOP_K])
                     if mid in set(row["expected_ids"])), None)
        rr.append(1 / rank if rank else 0.0)
    metrics.update({
        "evidence_mrr_at_10": round(statistics.mean(rr), 4),
        "query_count": count,
        "annotated_evidence_count": sum(len(r["expected_ids"]) for r in rows),
    })
    return metrics


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    clipset_path = ROOT / "data/egolife/testset/long_term/multiday_qa_clipset.jsonl"
    selected_later_ids = {
        json.loads(line)["id"] for line in clipset_path.open(encoding="utf-8")
        if line.strip() and int(json.loads(line)["_day"]) > 1
    }

    clips = []
    cache_counts = Counter()
    for sample_id, value in cache.items():
        match = re.fullmatch(r"DAY([1-7])_A1_JAKE_(\d{8})_en", str(sample_id))
        fields = value.get("fields") if isinstance(value, dict) else None
        if not match or not isinstance(fields, dict):
            continue
        day, stamp = int(match.group(1)), match.group(2)
        if day > 1 and sample_id not in selected_later_ids:
            continue
        seconds = parse_stamp(stamp)
        if seconds is None:
            continue
        cache_counts[day] += 1
        clips.append({"sample_id": sample_id, "day": day, "stamp": stamp,
                      "start": seconds, "end": seconds + CLIP_SECONDS,
                      "fields": fields})
    clips.sort(key=lambda c: (c["day"], c["start"], c["sample_id"]))
    clips_by_day = defaultdict(list)
    for clip in clips:
        clips_by_day[clip["day"]].append(clip)

    eligible, excluded = [], Counter()
    refs = Counter()
    for item in qa:
        qtime = item.get("query_time") or {}
        qday = normalize_day(qtime.get("date"))
        qstamp = str(qtime.get("time") or "")
        qseconds = parse_stamp(qstamp)
        if qday is None or qseconds is None:
            excluded["invalid_query_time"] += 1
            continue
        points, invalid_points = parse_target_points(item.get("target_time"))
        refs["target_points_total"] += len(points)
        refs["invalid_target_points"] += invalid_points
        expected = set()
        mapped_points = 0
        noncausal_points = 0
        uncached_points = 0
        for point in points:
            if point["day"] > qday or (point["day"] == qday and point["seconds"] >= qseconds):
                noncausal_points += 1
                continue
            # A target is evidence only if the matching clip exists in Qwen cache
            # and its complete interval is observable before the query cutoff.
            candidates = [c for c in clips_by_day.get(point["day"], [])
                          if c["start"] <= point["seconds"] < c["end"]]
            if not candidates:
                uncached_points += 1
                continue
            safe = [c for c in candidates
                    if c["day"] < qday or c["end"] <= qseconds]
            if not safe:
                noncausal_points += 1
                continue
            mapped_points += 1
            expected.update(c["sample_id"] for c in safe)
        refs["mapped_target_points"] += mapped_points
        refs["noncausal_target_points"] += noncausal_points
        refs["target_points_without_cached_clip"] += uncached_points
        if not expected:
            if noncausal_points and not uncached_points:
                excluded["target_not_observable_before_query"] += 1
            else:
                excluded["no_cached_target_evidence"] += 1
            continue
        relation = "cross_day" if any(p["day"] < qday for p in points) else "same_day"
        eligible.append({
            "id": str(item.get("ID", "")),
            "type": item.get("type", "unknown"),
            "query": item.get("question", ""),
            "question_day": qday,
            "question_time": qstamp,
            "target_points": points,
            "target_day": (item.get("target_time") or {}).get("date"),
            "expected_ids": sorted(expected),
            "answer_choice": item.get("answer"),
            "answer_text": item.get("choice_" + str(item.get("answer", "")).lower(), ""),
            "temporal_relation": relation,
        })
    eligible.sort(key=lambda x: (x["question_day"], parse_stamp(x["question_time"]), x["id"]))

    results, latencies = [], []
    with tempfile.TemporaryDirectory(prefix="egolifeqa_multiday_", dir=RUNS) as td:
        mem = PersonMemory(memory_dir=td)
        mem._save = lambda: None
        id_map = {}
        add_index = 0
        for query in eligible:
            qday, qseconds = query["question_day"], parse_stamp(query["question_time"])
            while add_index < len(clips):
                clip = clips[add_index]
                if not (clip["day"] < qday or (clip["day"] == qday and clip["end"] <= qseconds)):
                    break
                f = clip["fields"]
                scene = str(f.get("scene") or "")
                action = str(f.get("user_action") or "")
                if not action:
                    add_index += 1
                    continue
                needs = f.get("needs") if isinstance(f.get("needs"), list) else []
                solutions = f.get("solutions") if isinstance(f.get("solutions"), list) else []
                people = f.get("people") if isinstance(f.get("people"), list) else None
                location = str(f.get("location") or "") or None
                activity = str(f.get("activity") or "") or None
                environments = f.get("environments") if isinstance(f.get("environments"), list) else None
                objects = f.get("objects") if isinstance(f.get("objects"), list) else None
                notes = str(f.get("time_occasion") or "")[:200]
                mid = mem.add(
                    scene=scene, user_action=action, needs=needs, solutions=solutions,
                    people=people, location=location, activity=activity,
                    environments=environments, objects=objects, extra_notes=notes,
                    timestamp=iso_timestamp(clip["day"], clip["stamp"]),
                    source_evidence=str(f.get("source_evidence") or ""),
                )
                id_map[mid] = clip["sample_id"]
                add_index += 1
            started = time.perf_counter()
            hits = mem.query(query["query"], top_k=TOP_K)
            latencies.append((time.perf_counter() - started) * 1000)
            query["retrieved_ids"] = [id_map.get(m.get("id"), "") for m in hits]
            results.append(query)

    by_type = {}
    by_day = {}
    by_relation = {}
    for kind in sorted({r["type"] for r in results}):
        by_type[kind] = summarize([r for r in results if r["type"] == kind])
    for day in sorted({r["question_day"] for r in results}):
        subset = [r for r in results if r["question_day"] == day]
        by_day[f"DAY{day}"] = {"metrics": summarize(subset), "query_count": len(subset)}
    for relation in ("same_day", "cross_day"):
        subset = [r for r in results if r["temporal_relation"] == relation]
        if subset:
            by_relation[relation] = summarize(subset)

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "official EgoLifeQA A1_JAKE · causal multi-day slice",
        "source_file": str(QA_PATH),
        "source_question_count": len(qa),
        "query_count_by_day_all": dict(sorted(Counter(f"DAY{normalize_day((q.get('query_time') or {}).get('date'))}"
            for q in qa if normalize_day((q.get("query_time") or {}).get("date"))).items())),
        "eligible_query_count": len(results),
        "eligible_query_count_by_day": {k: v["query_count"] for k, v in by_day.items()},
        "excluded": dict(excluded),
        "target_reference_coverage": dict(refs),
        "memory_cache_file": str(CACHE_PATH),
        "transcript_evidence_clip_count": sum(
            bool(c["fields"].get("source_evidence")) for c in clips),
        "memory_cached_clips_by_day": {f"DAY{k}": v for k, v in sorted(cache_counts.items())},
        "memory_cached_clips_total": len(clips),
        "method": {
            "memory_source": "Cached Qwen3-VL perception fields; exact-ID speaker-labelled source transcript evidence is included when present. Official answers, choices, and dense captions are never inserted as memories.",
            "causal_cutoff": "A 30-second clip is available only after its full interval ends at or before the question timestamp. All earlier days are eligible.",
            "target_mapping": "Official target_time points are mapped to the cached 30-second source clip containing each timestamp; malformed/missing timestamps are counted separately.",
            "metrics_scope": "Evidence retrieval Hit/Recall/MRR only; this run does not score answer generation or a fully populated seven-day Qwen memory.",
            "coverage_limit": "The expanded cache contains the full DAY1 Qwen long-term baseline plus 460 official-QA target clips and 300 evenly sampled DAY2-DAY7 distractors (50 per day). Later-day coverage is sparse and must not be described as a full-week ingest.",
            "future_leakage": 0,
            "same_clip_future_frames": 0,
        },
        "overall": summarize(results),
        "by_question_type": by_type,
        "by_question_day": by_day,
        "by_temporal_relation": by_relation,
        "query_latency_ms": {"p50": percentile(latencies, 0.50),
                             "p95": percentile(latencies, 0.95), "count": len(latencies)},
        "details": results,
    }
    out = RUNS / f"egolifeqa_memory_retrieval_qwen_multiday_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in (
        "source_question_count", "query_count_by_day_all", "eligible_query_count",
        "eligible_query_count_by_day", "excluded", "target_reference_coverage",
        "memory_cached_clips_by_day", "overall", "by_temporal_relation",
        "query_latency_ms", "method")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("report:", out)


if __name__ == "__main__":
    main()
