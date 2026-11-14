"""End-to-end regression for daytime ingestion followed by nightly consolidation."""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from daytime_processor import DaytimeProcessor
from memory import PersonMemory
from nighttime_processor import NightlyProcessor


def model(prompt):
    if "TASK: NIGHTLY_CHUNK_SUMMARY" in prompt:
        return json.dumps({
            "summary": "Repeatedly worked and read at home.",
            "key_events": ["Read at home"],
            "patterns": ["Often works at home"],
            "profile_evidence": "Repeated home activity",
        })
    if "TASK: NIGHTLY_ROLLUP" in prompt:
        return json.dumps({
            "layer4": {"summary": "Today at home", "current_tasks": ["reading"],
                       "life_trajectory": "Home activities"},
            "layer5": {"summary": "Long-term home pattern", "key_events": [],
                       "long_term_patterns": "Reads at home"},
            "layer6_summary": "Enjoys reading at home",
            "profile_evidence": "Repeatedly reads at home",
            "combine": {},
        })
    if "TASK: NIGHTLY_PROFILE_EXTRACTION" in prompt:
        return json.dumps({
            "preferences": {"hobbies": [{
                "value": "reading", "confidence": 0.9,
                "evidence": "Repeated daily observations",
            }]}
        })
    raise AssertionError("Unexpected model prompt")


def main():
    with tempfile.TemporaryDirectory(prefix="day-night-") as folder:
        memory = PersonMemory(memory_dir=folder)
        daytime = DaytimeProcessor(memory, translate_fn=lambda value: "translated " + value)
        daytime.start_scene("home-evening")
        for i in range(8):
            daytime.ingest(
                scene=f"我在家阅读第{i}次",
                user_action="阅读书籍",
                needs=[],
                solutions=[],
                people=["Alex"],
                location="home",
                activity="reading",
                environments=["home"],
                objects=["book"],
            )
        assert daytime.flush_translations(timeout=5)
        assert memory.memory["layer7"]["moments"]
        assert any("normalized" in m for m in memory.memory["layer7"]["moments"].values())

        processor = NightlyProcessor(memory, llm_call=model, chunk_chars=1000)
        result = processor.run(day=date.today().isoformat(), daytime_processor=daytime)
        assert result["ok"] and result["moment_count"] == 8
        assert result["profile_updated"]
        assert memory.memory["layer4"]["summary"] == "Today at home"
        assert memory.memory["layer5"]["summary"] == "Long-term home pattern"
        assert memory.get_profile()["preferences"]["hobbies"][0]["value"] == "reading"

        cleared = daytime.end_scene()
        assert cleared == 8
        assert not memory.memory["layer1"] and not memory.memory["layer2"]
        assert len(memory.memory["layer3"]) == 8
        assert len(memory.memory["layer7"]["moments"]) == 8
        assert memory.query("reading", top_k=10)
        print("✅ Daytime asynchronous translation completed before nightly summary")
        print("✅ Nightly generated summaries, indices, and profile from archived moments")
        print("✅ Scene close cleared Layer 1/2 while Layer 3/7 stayed queryable")


if __name__ == "__main__":
    main()
