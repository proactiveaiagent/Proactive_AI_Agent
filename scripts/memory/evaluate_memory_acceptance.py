"""Measure memory retrieval accuracy, latency, and JSON storage on EgoLife test sets.

Retrieval labels seed a temporary memory so this measures the memory/query path
independently of Qwen perception. Stored text is the original EgoLife scene and
action text; ground-truth labels are used only as test tags and never persisted
as a separate ground-truth field.
"""
import json
import os
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

TEST = ROOT / "data/egolife/testset"
RUNS = ROOT / "data/egolife/runs"
QWEN_CACHE_PATH = Path(os.environ.get("EGOLIFE_QWEN_CACHE", str(RUNS / "qwen_cache.json")))
K_VALUES = (1, 3, 5, 10)
REPEATS = 20


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def event_timestamp(moment):
    parts = str(moment.get("id", "")).split("_")
    hhmmss = parts[3][:6] if len(parts) > 3 and parts[3][:6].isdigit() else "120000"
    return f"{date.today().isoformat()}T{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"


def seed_memory(rows, directory, checkpoints=()):
    mem = PersonMemory(memory_dir=str(directory))
    save = mem._save
    mem._save = lambda: None
    id_map = {}
    sizes = {}
    checkpoint_set = set(checkpoints)
    for i, row in enumerate(rows, 1):
        gt = row.get("ground_truth") or {}
        locs = gt.get("location_gt") or []
        mid = mem.add(
            scene=row.get("scene", ""),
            user_action=row.get("user_action", ""),
            needs=[], solutions=[],
            people=gt.get("people_gt") or None,
            location=locs[0] if locs else None,
            activity=gt.get("activity_gt") or None,
            environments=gt.get("environments_gt") or None,
            objects=gt.get("objects_gt") or None,
            timestamp=event_timestamp(row),
        )
        id_map[mid] = row.get("id", "")
        if i in checkpoint_set:
            save()
            sizes[i] = (directory / "memory.json").stat().st_size
    mem._save = save
    save()
    sizes[len(rows)] = (directory / "memory.json").stat().st_size
    return mem, id_map, sizes


def percentile(samples, p):
    if not samples:
        return None
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * p)))
    return round(ordered[idx], 3)


def score_queries(mem, id_map, queries, repeats=1):
    totals = {k: [] for k in K_VALUES}
    recalls = {k: [] for k in K_VALUES}
    by_type = defaultdict(lambda: {k: [] for k in K_VALUES})
    by_type_recall = defaultdict(lambda: {k: [] for k in K_VALUES})
    reciprocal, hit5, details, latencies = [], [], [], []
    for q in queries:
        expected = set(q.get("positives") or [])
        started = time.perf_counter()
        raw = mem.query(q["query"], top_k=max(K_VALUES))
        latencies.append((time.perf_counter() - started) * 1000)
        found = [id_map.get(m.get("id"), "") for m in raw]
        found = [x for x in found if x]
        hits = [item in expected for item in found]
        rank = next((i + 1 for i, yes in enumerate(hits) if yes), None)
        reciprocal.append(1 / rank if rank else 0.0)
        hit5.append(float(any(hits[:5])))
        result = {"query": q["query"], "type": q.get("type"),
                  "expected_count": len(expected), "returned": found[:10],
                  "relevant_at_10": sum(hits[:10])}
        details.append(result)
        for k in K_VALUES:
            p_at_k = sum(hits[:k]) / k
            recall_at_k = sum(hits[:k]) / len(expected) if expected else 0.0
            totals[k].append(p_at_k)
            recalls[k].append(recall_at_k)
            by_type[q.get("type", "unknown")][k].append(p_at_k)
            by_type_recall[q.get("type", "unknown")][k].append(recall_at_k)
    metrics = {f"precision_at_{k}": round(statistics.mean(vals), 4) if vals else None
               for k, vals in totals.items()}
    metrics.update({f"recall_at_{k}": round(statistics.mean(vals), 4) if vals else None
                    for k, vals in recalls.items()})
    metrics.update({"mrr_at_10": round(statistics.mean(reciprocal), 4) if reciprocal else None,
                    "hit_rate_at_5": round(statistics.mean(hit5), 4) if hit5 else None,
                    "query_count": len(queries)})
    per_type = {
        t: {
            **{f"precision_at_{k}": round(statistics.mean(by_type[t][k]), 4) if by_type[t][k] else None
               for k in K_VALUES},
            **{f"recall_at_{k}": round(statistics.mean(by_type_recall[t][k]), 4) if by_type_recall[t][k] else None
               for k in K_VALUES},
        } for t in by_type
    }
    if repeats > 1:
        for q in queries:
            for _ in range(repeats - 1):
                t0 = time.perf_counter()
                mem.query(q["query"], top_k=10)
                latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    return {"metrics": metrics, "by_query_type": per_type, "latency_ms": {
                "samples": len(latencies), "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95), "mean": round(statistics.mean(latencies), 3) if latencies else None},
            "details": details}




def seed_qwen_memory(rows, directory, cache):
    """Seed an isolated memory with cached model outputs, as the ingest path does."""
    mem = PersonMemory(memory_dir=str(directory))
    save = mem._save
    mem._save = lambda: None
    id_map = {}
    missing = []
    for row in rows:
        sample_id = row.get("id", "")
        fields = (cache.get(sample_id) or {}).get("fields")
        if not isinstance(fields, dict):
            missing.append(sample_id)
            continue
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
            extra_notes=str(fields.get("time_occasion") or "")[:200],
            timestamp=event_timestamp(row),
        )
        id_map[mid] = sample_id
    mem._save = save
    save()
    if missing:
        raise ValueError(f"Qwen cache missing {len(missing)}/{len(rows)} rows; examples: {missing[:5]}")
    return mem, id_map


def dataset_queries(rows, min_support=2, per_dimension=8):
    """Build diagnostic queries from this split's rule labels; these are weak-label checks."""
    dims = {
        "people": "people_gt",
        "locations": "location_gt",
        "environments": "environments_gt",
        "objects": "objects_gt",
        "activity_events": "activity_gt",
    }
    query_hits = {dim: defaultdict(set) for dim in dims}
    for row in rows:
        gt = row.get("ground_truth") or {}
        mid = row.get("id", "")
        if not mid:
            continue
        for dim, field in dims.items():
            values = gt.get(field) or []
            if not isinstance(values, list):
                values = [values]
            for value in values:
                label = str(value).strip()
                if label:
                    query_hits[dim][label].add(mid)
    queries = []
    for dim, labels in query_hits.items():
        ranked = sorted(labels.items(), key=lambda item: (-len(item[1]), item[0]))
        kept = 0
        for label, ids in ranked:
            if len(ids) < min_support:
                continue
            queries.append({"query": label, "type": dim, "support": len(ids),
                            "positives": sorted(ids)})
            kept += 1
            if kept >= per_dimension:
                break
    return queries


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    sample_rows = load_jsonl(TEST / "moments_sampled.jsonl")
    queries = json.loads((TEST / "queries.json").read_text(encoding="utf-8"))
    short_rows = load_jsonl(TEST / "short_term/moments_scattered.jsonl")
    long_rows = load_jsonl(TEST / "long_term/full_day.jsonl")
    long_unique_rows = [row for row in long_rows if str(row.get("id", "")).endswith("_en")]
    qwen_cache = json.loads(QWEN_CACHE_PATH.read_text(encoding="utf-8"))
    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "method": {"accuracy": "PersonMemory.query over ground-truth-seeded temporary moments; isolates retrieval plumbing from perception.",
                         "storage": "Original EgoLife scene and user_action strings plus six-dimensional ground-truth tags; no videos or separate ground_truth payload stored.",
                         "capacity_limit_bytes": 100 * 1024 ** 3,
                         "qwen_cache_file": str(QWEN_CACHE_PATH)},
              "datasets": {}}
    with tempfile.TemporaryDirectory(prefix="memory_acceptance_", dir=RUNS) as td:
        root = Path(td)
        mem, id_map, _ = seed_memory(sample_rows, root / "retrieval_900")
        report["datasets"]["sampled_900_retrieval"] = score_queries(mem, id_map, queries, repeats=REPEATS)
        report["datasets"]["sampled_900_retrieval"]["moments"] = len(sample_rows)

        short_mem, short_id_map, short_sizes = seed_memory(short_rows, root / "short_60", checkpoints=(30,))
        short_file_bytes = (root / "short_60/memory.json").stat().st_size
        short_lat = score_queries(short_mem, {}, [{"query": "phone", "positives": []}], repeats=REPEATS)["latency_ms"]
        report["datasets"]["short_term_60"] = {
            "moments": len(short_rows), "memory_json_bytes": short_file_bytes,
            "bytes_per_moment": round(short_file_bytes / max(1, len(short_rows)), 2),
            "capacity_at_100_gib_moments_estimate": int((100 * 1024 ** 3) / max(1, short_file_bytes / max(1, len(short_rows)))),
            "query_latency_ms": short_lat,
        }
        short_queries = dataset_queries(short_rows)
        report["datasets"]["short_term_60"]["retrieval"] = score_queries(
            short_mem, short_id_map, short_queries, repeats=REPEATS)
        report["datasets"]["short_term_60"]["retrieval"]["label_basis"] = "automatic rule labels; diagnostic only"
        short_qwen_mem, short_qwen_map = seed_qwen_memory(
            short_rows, root / "short_60_qwen", qwen_cache)
        report["datasets"]["short_term_60"]["qwen_backed_retrieval"] = score_queries(
            short_qwen_mem, short_qwen_map, short_queries, repeats=REPEATS)
        report["datasets"]["short_term_60"]["qwen_backed_retrieval"]["label_basis"] = "cached Qwen perception indexed; compared with automatic rule labels"

        long_mem, _, long_sizes = seed_memory(long_rows, root / "long_1450", checkpoints=(60, 725))
        long_file_bytes = (root / "long_1450/memory.json").stat().st_size
        long_lat = score_queries(long_mem, {}, [{"query": "phone", "positives": []}], repeats=REPEATS)["latency_ms"]
        report["datasets"]["long_term_1450"] = {
            "moments": len(long_rows), "memory_json_bytes": long_file_bytes,
            "bytes_per_moment": round(long_file_bytes / max(1, len(long_rows)), 2),
            "capacity_at_100_gib_moments_estimate": int((100 * 1024 ** 3) / max(1, long_file_bytes / max(1, len(long_rows)))),
            "growth_checkpoints_bytes": {str(k): v for k, v in long_sizes.items()},
            "query_latency_ms": long_lat,
        }
        long_unique_mem, long_id_map, _ = seed_memory(
            long_unique_rows, root / "long_725_unique")
        long_queries = dataset_queries(long_unique_rows, min_support=8)
        report["datasets"]["long_term_725_unique_retrieval"] = {
            "moments": len(long_unique_rows),
            "unique_model_inputs": len(long_unique_rows),
            "retrieval": score_queries(
                long_unique_mem, long_id_map, long_queries, repeats=REPEATS),
            "label_basis": "automatic rule labels on one English row per video; diagnostic only",
        }
        long_qwen_mem, long_qwen_map = seed_qwen_memory(
            long_unique_rows, root / "long_725_qwen", qwen_cache)
        report["datasets"]["long_term_725_unique_retrieval"]["qwen_backed_retrieval"] = score_queries(
            long_qwen_mem, long_qwen_map, long_queries, repeats=REPEATS)
        report["datasets"]["long_term_725_unique_retrieval"]["qwen_backed_retrieval"]["label_basis"] = "cached Qwen perception indexed; compared with automatic rule labels"
    acceptance = report["datasets"]["sampled_900_retrieval"]["metrics"]
    report["acceptance"] = {
        "scope": "sampled_900 retrieval-only with ground-truth tags seeded into a temporary memory; not end-to-end short/long accuracy",
        "valid_as_overall_memory_accuracy_claim": False,
        "precision_at_5_target": 0.95,
        "precision_at_5_met": (acceptance["precision_at_5"] or 0) >= 0.95,
        "hit_rate_at_5_target": 0.98,
        "hit_rate_at_5_met": (acceptance["hit_rate_at_5"] or 0) >= 0.98,
        "retrieval_p95_ms_target": 1000,
        "retrieval_p95_ms_met": report["datasets"]["sampled_900_retrieval"]["latency_ms"]["p95"] <= 1000,
        "storage_100_gib_met": max(report["datasets"][x]["memory_json_bytes"] for x in ("short_term_60", "long_term_1450")) <= 100 * 1024 ** 3,
        "split_qwen_indexed_diagnostics": {
            "short_term_qwen_p5": report["datasets"]["short_term_60"]["qwen_backed_retrieval"]["metrics"]["precision_at_5"],
            "short_term_qwen_hit5": report["datasets"]["short_term_60"]["qwen_backed_retrieval"]["metrics"]["hit_rate_at_5"],
            "long_term_qwen_p5": report["datasets"]["long_term_725_unique_retrieval"]["qwen_backed_retrieval"]["metrics"]["precision_at_5"],
            "long_term_qwen_hit5": report["datasets"]["long_term_725_unique_retrieval"]["qwen_backed_retrieval"]["metrics"]["hit_rate_at_5"],
            "ground_truth_basis": "automatic rule labels; diagnostic only, human adjudication required",
        },
    }
    out = RUNS / f"memory_acceptance_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report_path": str(out),
        "acceptance": report["acceptance"],
        "sampled_900": {
            "metrics": report["datasets"]["sampled_900_retrieval"]["metrics"],
            "latency_ms": report["datasets"]["sampled_900_retrieval"]["latency_ms"],
        },
        "short_term_60": {
            "moments": report["datasets"]["short_term_60"]["moments"],
            "memory_json_bytes": report["datasets"]["short_term_60"]["memory_json_bytes"],
            "query_latency_ms": report["datasets"]["short_term_60"]["query_latency_ms"],
        },
        "long_term_1450": {
            "moments": report["datasets"]["long_term_1450"]["moments"],
            "memory_json_bytes": report["datasets"]["long_term_1450"]["memory_json_bytes"],
            "query_latency_ms": report["datasets"]["long_term_1450"]["query_latency_ms"],
        },
        "short_term_gold_seeded_retrieval": report["datasets"]["short_term_60"]["retrieval"]["metrics"],
        "short_term_qwen_indexed_retrieval": report["datasets"]["short_term_60"]["qwen_backed_retrieval"]["metrics"],
        "long_term_gold_seeded_retrieval": report["datasets"]["long_term_725_unique_retrieval"]["retrieval"]["metrics"],
        "long_term_qwen_indexed_retrieval": report["datasets"]["long_term_725_unique_retrieval"]["qwen_backed_retrieval"]["metrics"],
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()