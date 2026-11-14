"""Evaluate temporal evidence retrieval on official EgoLifeQA A1_JAKE annotations.

Only cached Qwen perception outputs are written to an isolated PersonMemory.
For each query, only clips beginning before the question time are available.
This evaluates evidence retrieval, not answer generation or overall accuracy.
"""
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory

RUNS = ROOT / "data/egolife/runs"
QA_PATH = ROOT / "data/egolife/testset/long_term/egolifeqa_official_a1_jake.json"
FULL_DAY = ROOT / "data/egolife/testset/long_term/full_day.jsonl"
CACHE_PATH = Path(os.environ.get("EGOLIFE_QWEN_CACHE", str(RUNS / "qwen_cache.json")))
SOURCE_URL = "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/EgoLifeQA/EgoLifeQA_A1_JAKE.json"
FPS = 30.0
CLIP_SECONDS = 30.0
TOP_K = 10


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clock_seconds(value):
    value = str(value)
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"Expected HHMMSSFF timestamp, got {value!r}")
    return (int(value[0:2]) * 3600 + int(value[2:4]) * 60 +
            int(value[4:6]) + int(value[6:8]) / FPS)


def clock_iso(value):
    value = str(value)
    return (f"{date.today().isoformat()}T{value[0:2]}:{value[2:4]}:"
            f"{value[4:6]}")


def target_values(target):
    if not isinstance(target, dict):
        return []
    if isinstance(target.get("time_list"), list):
        return target["time_list"]
    return [target["time"]] if target.get("time") else []


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered)-1, int((len(ordered)-1)*p))], 3)


def summarize(rows):
    if not rows:
        return {}
    count = len(rows)
    evidence_n = sum(len(r["expected_ids"]) for r in rows)
    hits = {}
    recalls = {}
    for k in (1, 5, 10):
        hits[f"evidence_hit_at_{k}"] = round(sum(bool(set(r["retrieved_ids"][:k]) & set(r["expected_ids"])) for r in rows) / count, 4)
        recalls[f"evidence_recall_at_{k}"] = round(sum(len(set(r["retrieved_ids"][:k]) & set(r["expected_ids"])) / len(r["expected_ids"]) for r in rows) / count, 4)
    rr = []
    for r in rows:
        rank = next((i+1 for i, mid in enumerate(r["retrieved_ids"][:10]) if mid in r["expected_ids"]), None)
        rr.append(1 / rank if rank else 0.0)
    return {**hits, **recalls, "evidence_mrr_at_10": round(statistics.mean(rr), 4), "query_count": count, "annotated_evidence_count": evidence_n}


def main():
    source_mode = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    if source_mode not in {"qwen", "annotated"}:
        raise SystemExit("usage: evaluate_egolifeqa_memory.py [qwen|annotated]")
    RUNS.mkdir(parents=True, exist_ok=True)
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    rows = [r for r in load_jsonl(FULL_DAY) if str(r.get("id", "")).endswith("_en")]

    # Map annotation timestamps to source clip IDs (30-second video units).
    clips = []
    missing_cache = []
    for row in rows:
        sample_id = row["id"]
        fields = (cache.get(sample_id) or {}).get("fields")
        if not isinstance(fields, dict):
            missing_cache.append(sample_id)
            continue
        parts = sample_id.split("_")
        stamp = parts[3]
        clips.append({"sample_id": sample_id, "stamp": stamp,
                      "start": clock_seconds(stamp), "fields": fields,
                      "row": row})
    clips.sort(key=lambda x: (x["start"], x["sample_id"]))

    eligible, excluded = [], defaultdict(int)
    evidence_total = evidence_available = 0
    for item in qa:
        query_time = item.get("query_time") or {}
        if str(query_time.get("date", "")).upper() != "DAY1":
            excluded["query_not_DAY1"] += 1
            continue
        try:
            qtime = clock_seconds(query_time.get("time", ""))
        except ValueError:
            excluded["invalid_query_time"] += 1
            continue
        target = item.get("target_time") or {}
        if str(target.get("date", "")).upper() != "DAY1":
            excluded["target_not_DAY1"] += 1
            continue
        values = target_values(target)
        evidence_total += len(values)
        expected = set()
        for value in values:
            try:
                tsec = clock_seconds(value)
            except ValueError:
                continue
            # EgoLifeQA evidence points to the 30s clip containing the timestamp.
            matches = [c["sample_id"] for c in clips if c["start"] <= tsec < c["start"] + CLIP_SECONDS]
            # Only count evidence which is both available in the cache and causal.
            matches = [mid for mid in matches if next(c["start"] for c in clips if c["sample_id"] == mid) < qtime]
            if matches:
                expected.update(matches)
        evidence_available += len(expected)
        if not expected:
            excluded["no_cached_target_evidence"] += 1
            continue
        eligible.append({"id": str(item.get("ID", "")), "type": item.get("type", "unknown"),
                         "query": item.get("question", ""), "question_time": query_time.get("time"),
                         "target_time": values, "expected_ids": sorted(expected),
                         "answer_choice": item.get("answer"),
                         "answer_text": item.get("choice_" + str(item.get("answer", "")).lower(), ""),
                         "reason": item.get("reason", "")})

    eligible.sort(key=lambda x: (clock_seconds(x["question_time"]), x["id"]))
    results = []
    add_index = 0
    query_latencies = []
    with tempfile.TemporaryDirectory(prefix="egolifeqa_memory_", dir=RUNS) as td:
        mem = PersonMemory(memory_dir=td)
        save = mem._save
        mem._save = lambda: None
        id_map = {}
        for q in eligible:
            qtime = clock_seconds(q["question_time"])
            while add_index < len(clips) and clips[add_index]["start"] < qtime:
                clip = clips[add_index]
                if source_mode == "qwen":
                    f = clip["fields"]
                    scene = str(f.get("scene") or "")
                    user_action = str(f.get("user_action") or "")
                    needs = f.get("needs") if isinstance(f.get("needs"), list) else []
                    solutions = f.get("solutions") if isinstance(f.get("solutions"), list) else []
                    people = f.get("people") if isinstance(f.get("people"), list) else None
                    location = str(f.get("location") or "") or None
                    activity = str(f.get("activity") or "") or None
                    environments = f.get("environments") if isinstance(f.get("environments"), list) else None
                    objects = f.get("objects") if isinstance(f.get("objects"), list) else None
                    notes = str(f.get("time_occasion") or "")[:200]
                else:
                    row = clip["row"]
                    scene = str(row.get("scene") or "")
                    user_action = str(row.get("user_action") or "")
                    needs, solutions, people, location, activity = [], [], None, None, None
                    environments, objects = None, None
                    notes = "\n".join(x for x in (
                        "Transcript: " + str(row.get("transcript") or ""),
                        "Dense caption: " + str(row.get("dense_caption") or ""),
                    ) if x.strip())
                mid = mem.add(
                    scene=scene, user_action=user_action, needs=needs, solutions=solutions,
                    people=people, location=location, activity=activity,
                    environments=environments, objects=objects, extra_notes=notes,
                    timestamp=clock_iso(clip["stamp"]),
                )
                id_map[mid] = clip["sample_id"]
                add_index += 1
            started = time.perf_counter()
            hits = mem.query(q["query"], top_k=TOP_K)
            query_latencies.append((time.perf_counter() - started) * 1000)
            q["retrieved_ids"] = [id_map.get(m.get("id"), "") for m in hits]
            results.append(q)
        mem._save = save

    by_type = {}
    for kind in sorted({r["type"] for r in results}):
        by_type[kind] = summarize([r for r in results if r["type"] == kind])
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "official EgoLifeQA A1_JAKE",
        "source_url": SOURCE_URL,
        "source_file_bytes": QA_PATH.stat().st_size,
        "source_question_count": len(qa),
        "query_day1_count": sum(str(q.get("query_time", {}).get("date", "")).upper() == "DAY1" for q in qa),
        "eligible_query_count": len(results),
        "excluded": dict(excluded),
        "source_clips": len(rows),
        "qwen_cached_clips": len(clips),
        "qwen_cache_file": str(CACHE_PATH),
        "qwen_cache_missing_clips": len(missing_cache),
        "evidence_timestamps_total_day1": evidence_total,
        "evidence_events_mapped_to_qwen_cache": evidence_available,
        "method": {
            "memory_source": ("cached Qwen3-VL perception outputs from the actual ingest cache" if source_mode == "qwen" else "EgoLife scene/action/transcript/dense-caption annotation text; oracle-assisted retrieval upper bound, not deployed-model accuracy"),
            "causal_cutoff": "For each QA question, only events whose clip start time precedes query_time are added before retrieval",
            "relevance": "Official EgoLifeQA target_time mapped to its 30-second source video ID; multi-event target_time maps to multiple IDs",
            "metrics_scope": "Evidence retrieval only (Hit/Recall/MRR). Does not measure answer generation, full semantic accuracy, or cross-day L4-L6 summarization.",
            "source_mode": source_mode,
            "p_at_5_comparable_to_design_doc": False,
            "p_at_5_note": "Official QA provides sparse timestamped evidence (usually one clip/query), so evidence Hit/Recall is reported instead of fixed-denominator P@5; this result must not be compared directly to the manual P@5 acceptance threshold.",
            "future_memory_leakage": 0,
        },
        "overall": summarize(results),
        "by_question_type": by_type,
        "query_latency_ms": {"p50": percentile(query_latencies, 0.50), "p95": percentile(query_latencies, 0.95), "count": len(query_latencies)},
        "details": results,
    }
    out = RUNS / f"egolifeqa_memory_retrieval_{source_mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("created_at", "source_question_count", "query_day1_count", "eligible_query_count", "excluded", "qwen_cached_clips", "evidence_timestamps_total_day1", "evidence_events_mapped_to_qwen_cache", "overall", "by_question_type", "query_latency_ms")}, ensure_ascii=False, indent=2))
    print("report:", out)


if __name__ == "__main__":
    main()
