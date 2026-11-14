"""Prewarm Qwen V3 cache for the expanded short-term and multi-day testsets."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/memory"))
from qwen_perceive import QwenPerceiver

TEST = ROOT / "data/egolife/testset"
RUNS = ROOT / "data/egolife/runs"
BASELINE = RUNS / "qwen_cache_memory_prompt_v3.json"
CACHE = RUNS / "qwen_cache_expanded_eval.json"
REPORT = RUNS / "qwen_cache_expanded_eval_report.json"


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    rows_by_id = {}
    # Prefer the main sampled row if the same video appears in both sets, since
    # it carries the short-term set's exact transcript and ground truth.
    for row in read_jsonl(TEST / "long_term/multiday_qa_clipset.jsonl"):
        rows_by_id.setdefault(row["id"], row)
    for row in read_jsonl(TEST / "short_term/moments_scattered_v2_120.jsonl"):
        rows_by_id[row["id"]] = row
    rows = sorted(rows_by_id.values(),
                  key=lambda r: (int(r.get("_day", 1)),
                                 str(r.get("id", "").split("_")[3]),
                                 r.get("id", "")))

    RUNS.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    if CACHE.exists():
        expanded = json.loads(CACHE.read_text(encoding="utf-8"))
        for key, value in baseline.items():
            expanded.setdefault(key, value)
    else:
        expanded = dict(baseline)
    CACHE.write_text(json.dumps(expanded, ensure_ascii=False, indent=2), encoding="utf-8")

    perceiver = QwenPerceiver(cache_file=CACHE)
    health = perceiver.health()
    if not health.get("model_loaded"):
        raise SystemExit(f"Qwen service unavailable: {health}")
    missing = [r for r in rows if r["id"] not in perceiver.cache]
    print(json.dumps({
        "short_and_multiday_unique_rows": len(rows),
        "baseline_cache_entries": len(baseline),
        "expanded_cache_entries_before": len(perceiver.cache),
        "rows_missing_before": len(missing),
        "health": health,
        "cache_file": str(CACHE),
    }, ensure_ascii=False), flush=True)

    save_real = perceiver._save_cache
    writes = 0
    def save_batched():
        nonlocal writes
        writes += 1
        if writes == 1 or writes % 5 == 0:
            save_real()
    perceiver._save_cache = save_batched

    started = time.time()
    failures = []
    for index, row in enumerate(missing, 1):
        result = perceiver.perceive(row)
        if not result.get("fields"):
            failures.append({"id": row["id"], "error": result.get("error", "unknown")})
        if index == 1 or index % 10 == 0 or index == len(missing):
            elapsed = time.time() - started
            print(json.dumps({
                "done": index, "total": len(missing),
                "success": index-len(failures), "failed": len(failures),
                "elapsed_min": round(elapsed/60, 1),
                "latest_ms": result.get("elapsed_ms"),
                "cache_entries": len(perceiver.cache),
            }, ensure_ascii=False), flush=True)
    perceiver._save_cache = save_real
    save_real()

    report = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_cache": str(BASELINE),
        "expanded_cache": str(CACHE),
        "baseline_entries": len(baseline),
        "expanded_entries": len(perceiver.cache),
        "testset_unique_rows": len(rows),
        "rows_missing_before": len(missing),
        "success": len(missing)-len(failures),
        "failures": failures,
        "elapsed_minutes": round((time.time()-started)/60, 2),
        "model_health": health,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
