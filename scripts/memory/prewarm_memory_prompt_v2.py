"""Re-perceive the short/long memory acceptance clips into a separate cache."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/memory"))
from qwen_perceive import QwenPerceiver, video_path_of

RUNS = ROOT / "data/egolife/runs"
OUT = RUNS / "qwen_cache_memory_prompt_v2.json"
TEST = ROOT / "data/egolife/testset"


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    rows = read_jsonl(TEST / "short_term/moments_scattered.jsonl")
    rows += [r for r in read_jsonl(TEST / "long_term/full_day.jsonl")
             if str(r.get("id", "")).endswith("_en")]
    unique = {}
    for row in rows:
        unique.setdefault(row.get("id", ""), row)
    rows = [r for r in unique.values() if r.get("id")]
    missing = [r.get("id", "") for r in rows if video_path_of(r) is None]
    if missing:
        raise SystemExit(f"Missing source videos: {len(missing)}; examples: {missing[:5]}")

    perceiver = QwenPerceiver(cache_file=OUT)
    health = perceiver.health()
    if not health.get("model_loaded"):
        raise SystemExit(f"Qwen service unavailable: {health}")
    RUNS.mkdir(parents=True, exist_ok=True)
    save_real = perceiver._save_cache
    writes = 0

    def save_batched():
        nonlocal writes
        writes += 1
        if writes == 1 or writes % 10 == 0:
            save_real()

    perceiver._save_cache = save_batched
    started = time.time()
    failures = []
    for i, row in enumerate(rows, 1):
        result = perceiver.perceive(row, force=False)
        if not result.get("fields"):
            failures.append({"id": row["id"], "error": result.get("error", "unknown")})
        if i == 1 or i % 10 == 0 or i == len(rows):
            elapsed = time.time() - started
            print(json.dumps({"done": i, "total": len(rows), "success": i-len(failures),
                              "failed": len(failures), "elapsed_min": round(elapsed/60, 1),
                              "latest_ms": result.get("elapsed_ms")}, ensure_ascii=False), flush=True)
    perceiver._save_cache = save_real
    save_real()
    failure_path = RUNS / "qwen_cache_memory_prompt_v2_failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"finished": True, "cache_file": str(OUT),
                      "success": len(rows)-len(failures), "failed": len(failures),
                      "failure_file": str(failure_path),
                      "elapsed_min": round((time.time()-started)/60, 1)}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
