"""Measure memory retrieval accuracy, latency, and JSON storage on EgoLife test sets.

Retrieval labels seed a temporary memory so this measures the memory/query path
independently of Qwen perception. Stored text is the original EgoLife scene and
action text; ground-truth labels are used only as test tags and never persisted
as a separate ground-truth field.
"""
import json
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
    by_type = defaultdict(lambda: {k: [] for k in K_VALUES})
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
            totals[k].append(p_at_k)
            by_type[q.get("type", "unknown")][k].append(p_at_k)
    metrics = {f"precision_at_{k}": round(statistics.mean(vals), 4) if vals else None
               for k, vals in totals.items()}
    metrics.update({"mrr_at_10": round(statistics.mean(reciprocal), 4) if reciprocal else None,
                    "hit_rate_at_5": round(statistics.mean(hit5), 4) if hit5 else None,
                    "query_count": len(queries)})
    per_type = {t: {f"precision_at_{k}": round(statistics.mean(v[k]), 4) if v[k] else None
                    for k in K_VALUES} for t, v in by_type.items()}
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


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    sample_rows = load_jsonl(TEST / "moments_sampled.jsonl")
    queries = json.loads((TEST / "queries.json").read_text(encoding="utf-8"))
    short_rows = load_jsonl(TEST / "short_term/moments_scattered.jsonl")
    long_rows = load_jsonl(TEST / "long_term/full_day.jsonl")
    report = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "method": {"accuracy": "PersonMemory.query over ground-truth-seeded temporary moments; isolates retrieval plumbing from perception.",
                         "storage": "Original EgoLife scene and user_action strings plus six-dimensional ground-truth tags; no videos or separate ground_truth payload stored.",
                         "capacity_limit_bytes": 100 * 1024 ** 3},
              "datasets": {}}
    with tempfile.TemporaryDirectory(prefix="memory_acceptance_", dir=RUNS) as td:
        root = Path(td)
        mem, id_map, _ = seed_memory(sample_rows, root / "retrieval_900")
        report["datasets"]["sampled_900_retrieval"] = score_queries(mem, id_map, queries, repeats=REPEATS)
        report["datasets"]["sampled_900_retrieval"]["moments"] = len(sample_rows)

        short_mem, _, short_sizes = seed_memory(short_rows, root / "short_60", checkpoints=(30,))
        short_file_bytes = (root / "short_60/memory.json").stat().st_size
        short_lat = score_queries(short_mem, {}, [{"query": "phone", "positives": []}], repeats=REPEATS)["latency_ms"]
        report["datasets"]["short_term_60"] = {
            "moments": len(short_rows), "memory_json_bytes": short_file_bytes,
            "bytes_per_moment": round(short_file_bytes / max(1, len(short_rows)), 2),
            "capacity_at_100_gib_moments_estimate": int((100 * 1024 ** 3) / max(1, short_file_bytes / max(1, len(short_rows)))),
            "query_latency_ms": short_lat,
        }

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
    acceptance = report["datasets"]["sampled_900_retrieval"]["metrics"]
    report["acceptance"] = {
        "precision_at_5_target": 0.95,
        "precision_at_5_met": (acceptance["precision_at_5"] or 0) >= 0.95,
        "hit_rate_at_5_target": 0.98,
        "hit_rate_at_5_met": (acceptance["hit_rate_at_5"] or 0) >= 0.98,
        "retrieval_p95_ms_target": 1000,
        "retrieval_p95_ms_met": report["datasets"]["sampled_900_retrieval"]["latency_ms"]["p95"] <= 1000,
        "storage_100_gib_met": max(report["datasets"][x]["memory_json_bytes"] for x in ("short_term_60", "long_term_1450")) <= 100 * 1024 ** 3,
    }
    out = RUNS / f"memory_acceptance_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(out), "acceptance": report["acceptance"],
                      "sampled_900": {"metrics": report["datasets"]["sampled_900_retrieval"]["metrics"], "latency_ms": report["datasets"]["sampled_900_retrieval"]["latency_ms"]},
                      "short_term_60": report["datasets"]["short_term_60"],
                      "long_term_1450": report["datasets"]["long_term_1450"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()