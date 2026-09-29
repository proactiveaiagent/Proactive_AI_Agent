"""Run three evidence-grounded semantic memory scenarios on a copy of EgoLife.

This is a qualitative scenario review, not an accuracy benchmark. The source
memory file is copied first so retrieval access counters cannot modify it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from daytime_processor import DaytimeProcessor
from memory import PersonMemory


SCENARIOS = [
    {
        "id": "alice_support",
        "title": "Alice：基于过往事件提供下一步协助",
        "situation": (
            "Alice says she wants to take a shower. What helpful next action should I take? "
            "Use relevant remembered context if available; do not invent a preference."
        ),
        "review_target": "检查是否引用 Alice 的相关事件，并把建议限制在记忆支持的范围内。",
    },
    {
        "id": "fainting_safety",
        "title": "有人晕倒：区分病史证据与通用安全建议",
        "situation": (
            "Someone nearby has just fainted and fallen. What immediate action should I take? "
            "Check memory for relevant medical history, state clearly if none is available, "
            "and give general safety guidance without diagnosing."
        ),
        "review_target": "记忆没有相关病史时，必须标明证据不足，同时给出通用急救建议。",
    },
    {
        "id": "country_tipping",
        "title": "切换国家：核实小费提醒的个人依据",
        "situation": (
            "The user has moved from China to the United States and is finishing a restaurant meal. "
            "Should I remind them about tipping? Check memory for a stored location or tipping preference; "
            "separate remembered facts from general local custom."
        ),
        "review_target": "不得把通用习俗伪装成用户记忆；没有国家或偏好记录时应说明前提。",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/egolife/runs/multimodal_full_day_v5_20260929/memory.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/egolife/runs/semantic_scenario_review_20260929",
    )
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_file = output_dir / "memory.json"
    if not memory_file.exists():
        shutil.copy2(source, memory_file)

    memory = PersonMemory(str(output_dir))
    daytime = DaytimeProcessor(memory)
    result_path = output_dir / "scenario_review.json"
    results = {}
    if result_path.exists():
        try:
            results = json.loads(result_path.read_text(encoding="utf-8")).get("scenarios", {})
        except (OSError, ValueError, TypeError):
            results = {}

    report = {
        "title": "EgoLife 语义决策场景审阅",
        "source_memory": str(source.relative_to(ROOT)),
        "source_moments": len(memory.memory.get("layer7", {}).get("moments", {})),
        "review_type": "qualitative_examples_not_accuracy_benchmark",
        "scenarios": results,
    }

    for scenario in SCENARIOS:
        if scenario["id"] in results:
            continue
        started = time.perf_counter()
        answer = daytime.advise(scenario["situation"])
        cited = set((answer.get("recommendation") or {}).get("evidence_ids") or [])
        candidates = answer.get("candidates") or []
        cited_candidates = [item for item in candidates if item.get("id") in cited]
        report["scenarios"][scenario["id"]] = {
            "title": scenario["title"],
            "situation": scenario["situation"],
            "review_target": scenario["review_target"],
            "recommendation": answer.get("recommendation") or {},
            "model_recommendation": answer.get("model_recommendation") or {},
            "policy_adjustments": answer.get("policy_adjustments") or [],
            "retrieval_rounds": answer.get("retrieval_rounds") or [],
            "candidates": candidates,
            "cited_candidate_count": len(cited_candidates),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
        tmp = result_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(result_path)
        print(json.dumps({
            "completed": scenario["id"],
            "elapsed_ms": report["scenarios"][scenario["id"]]["elapsed_ms"],
            "evidence_ids": sorted(cited),
            "action": (answer.get("recommendation") or {}).get("action", ""),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
