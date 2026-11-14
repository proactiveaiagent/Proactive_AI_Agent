from collections import OrderedDict
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/memory"))
from qwen_perceive import QwenPerceiver, _contains_cjk, video_path_of

TEST = ROOT / "data/egolife/testset"
DEFAULT_CACHE = ROOT / "data/egolife/runs/qwen_cache_memory_prompt_v3.json"
CACHE_PATH = Path(os.environ.get("EGOLIFE_QWEN_CACHE", str(DEFAULT_CACHE)))


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    rows = read_jsonl(TEST / "short_term/moments_scattered.jsonl")
    rows += [row for row in read_jsonl(TEST / "long_term/full_day.jsonl")
             if str(row.get("id", "")).endswith("_en")]
    unique = OrderedDict((row.get("id", ""), row) for row in rows if row.get("id"))
    perceiver = QwenPerceiver(cache_file=CACHE_PATH)
    health = perceiver.health()
    if not health.get("model_loaded"):
        raise SystemExit(f"Qwen service unavailable: {health}")

    invalid = []
    for sample_id, row in unique.items():
        record = perceiver.cache.get(sample_id) or {}
        fields_text = json.dumps(record.get("fields") or {}, ensure_ascii=False)
        if (not record.get("fields") or _contains_cjk(fields_text)
                or _contains_cjk(record.get("raw", ""))):
            invalid.append((sample_id, row))

    failures = []
    started = time.time()
    for index, (sample_id, row) in enumerate(invalid, 1):
        perceiver.cache.pop(sample_id, None)
        perceiver._save_cache()
        retry_row = dict(row)
        retry_row["transcript"] = re.sub(
            r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+",
            " ", str(row.get("transcript") or ""))
        result = perceiver.perceive(retry_row, force=True)
        if not result.get("fields"):
            failures.append({"id": sample_id, "error": result.get("error", "unknown")})
        if index == 1 or index % 5 == 0 or index == len(invalid):
            print(json.dumps({"repaired": index, "total_invalid": len(invalid),
                              "failed": len(failures), "latest_id": sample_id,
                              "elapsed_seconds": round(time.time()-started, 1)}, ensure_ascii=False),
                  flush=True)

    failure_path = CACHE_PATH.parent / "qwen_cache_memory_prompt_v3_english_repair_failures.json"
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total_records": len(unique), "repaired_records": len(invalid),
                      "repair_failures": len(failures), "cache_file": str(CACHE_PATH),
                      "failure_file": str(failure_path)}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
