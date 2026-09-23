"""Batch Qwen3-VL golden-set evaluation with one-to-one precision/recall/F1."""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_perceive import QwenPerceiver

DATA = {
    "short_term": ROOT / "data/egolife/testset/short_term/moments_scattered.jsonl",
    "long_term": ROOT / "data/egolife/testset/long_term/full_day.jsonl",
}
DIMS = {
    "people": ("people", "people_gt"),
    "locations": ("location", "location_gt"),
    "activity_events": ("activity", "activity_gt"),
    "environments": ("environments", "environments_gt"),
    "objects": ("objects", "objects_gt"),
}


def norm(value):
    return re.sub(r"[^\w]+", "", str(value or "").casefold(), flags=re.UNICODE)


def match_score(prediction, golden):
    p, g = norm(prediction), norm(golden)
    if not p or not g:
        return 0
    if p == g:
        return 2
    return 1 if p in g or g in p else 0


def matched_count(predictions, goldens):
    """Maximum one-to-one bipartite matching; exact matches are tried first."""
    edges = []
    for p in predictions:
        edges.append(sorted((g for g in range(len(goldens)) if match_score(p, goldens[g])),
                            key=lambda g: -match_score(p, goldens[g])))
    order = sorted(range(len(predictions)), key=lambda i: (len(edges[i]), i))
    owner = {}

    def assign(pi, seen):
        for gi in edges[pi]:
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in owner or assign(owner[gi], seen):
                owner[gi] = pi
                return True
        return False

    return sum(assign(pi, set()) for pi in order)


def values(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def load_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def perception_signature(row):
    # Paired bilingual rows can share perception only when every model input is identical.
    return (row.get("video"), row.get("start_time"), row.get("end_time"), row.get("transcript", ""))


def percentile(samples, p):
    if not samples:
        return None
    ordered = sorted(samples)
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=["short_term", "long_term", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0, help="Per scenario limit; 0 means all")
    ap.add_argument("--force", action="store_true", help="Bypass perception cache and call Qwen")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--output", help="JSON report path; defaults under data/egolife/runs")
    args = ap.parse_args()
    selected = list(DATA) if args.scenario == "both" else [args.scenario]
    perceiver = QwenPerceiver(api_url=args.api_url)
    health = perceiver.health()
    if not health.get("model_loaded"):
        print(json.dumps({"ok": False, "error": "Qwen service is not ready", "health": health}, ensure_ascii=False))
        raise SystemExit(2)

    report = {"ok": True, "created_at": datetime.now().isoformat(timespec="seconds"),
              "model": health, "cache_used": not args.force, "scenarios": {},
              "time_index_check": {"description": "Dataset hour labels are checked against source IDs; this is input metadata, not a visual-model score."}}
    t_all = time.time()
    for scenario in selected:
        rows = load_rows(DATA[scenario])
        if args.limit > 0:
            rows = rows[:args.limit]
        sums = {d: {"tp": 0, "fp": 0, "fn": 0, "predictions": 0, "goldens": 0} for d in DIMS}
        failures, latencies, details = [], [], []
        sources = Counter()
        time_checked = time_correct = 0
        shared_inputs = {}
        for i, row in enumerate(rows, 1):
            signature = perception_signature(row)
            mid = row.get("id", "")
            if not args.force and signature in shared_inputs:
                out = dict(shared_inputs[signature])
                out["source"] = "shared"
            elif not args.force and mid in perceiver.cache:
                out = perceiver.perceive(row)
                if out.get("fields"):
                    shared_inputs[signature] = dict(perceiver.cache[mid])
            else:
                out = perceiver.perceive(row, force=args.force)
                if not args.force and out.get("fields"):
                    shared_inputs[signature] = dict(perceiver.cache.get(mid, out))
            sources[out.get("source", "unknown")] += 1
            fields = out.get("fields") or {}
            if out.get("elapsed_ms") is not None:
                latencies.append(out["elapsed_ms"])
            if not fields:
                failures.append({"id": row.get("id"), "error": out.get("error", "empty perception")})
            item = {"id": row.get("id"), "source": out.get("source"), "elapsed_ms": out.get("elapsed_ms")}
            for dim, (pred_key, gold_key) in DIMS.items():
                pred, gold = values(fields.get(pred_key)), values((row.get("ground_truth") or {}).get(gold_key))
                tp = matched_count(pred, gold)
                sums[dim]["tp"] += tp
                sums[dim]["fp"] += len(pred) - tp
                sums[dim]["fn"] += len(gold) - tp
                sums[dim]["predictions"] += len(pred)
                sums[dim]["goldens"] += len(gold)
                item[dim] = {"predicted": pred, "golden": gold, "tp": tp}
            gt_time = (row.get("ground_truth") or {}).get("time_gt")
            hour = row.get("_hour")
            if gt_time and hour:
                time_checked += 1
                time_correct += int(str(gt_time).endswith("_" + str(hour)))
            details.append(item)
            if i % 10 == 0 or i == len(rows):
                print(f"[{scenario}] {i}/{len(rows)} clips processed", flush=True)

        metrics = {}
        for dim, c in sums.items():
            precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
            recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics[dim] = {**c, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        report["scenarios"][scenario] = {
            "samples": len(rows), "unique_model_inputs": len({perception_signature(r) for r in rows}),
            "shared_result_samples": sources.get("shared", 0),
            "failed_samples": len(failures), "failures": failures,
            "perception_sources": dict(sources),
            "latency_ms": {"mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
                           "p50": percentile(latencies, 0.50),
                           "p95": percentile(latencies, 0.95)},
            "metrics": metrics,
            "time_index_check": {"checked": time_checked, "correct": time_correct,
                                 "accuracy": round(time_correct / time_checked, 4) if time_checked else None},
            "details": details,
        }
    report["elapsed_seconds"] = round(time.time() - t_all, 2)
    out_path = Path(args.output) if args.output else ROOT / "data/egolife/runs" / f"qwen_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    print(json.dumps({"ok": report["ok"], "report_path": str(out_path), "elapsed_seconds": report["elapsed_seconds"], "scenarios": {k: {"samples": v["samples"], "failed_samples": v["failed_samples"], "metrics": v["metrics"], "time_index_check": v["time_index_check"], "latency_ms": v["latency_ms"]} for k, v in report["scenarios"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()