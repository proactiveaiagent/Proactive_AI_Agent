"""Regression coverage for the nightly memory consolidation pipeline."""
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from memory import PersonMemory
from nighttime_processor import NightlyProcessor


def respond(prompt):
    if "TASK: NIGHTLY_CHUNK_SUMMARY" in prompt:
        return json.dumps({
            "summary": "Observed daily activity, with repeated visits to home.",
            "key_events": ["Worked at home"],
            "patterns": ["Frequently at home"],
            "profile_evidence": "Repeated home activity",
        })
    if "TASK: NIGHTLY_ROLLUP" in prompt:
        return json.dumps({
            "layer4": {
                "summary": "Daily summary",
                "current_tasks": ["work"],
                "life_trajectory": "Worked from home",
            },
            "layer5": {
                "summary": "Long-term summary",
                "key_events": ["remote work"],
                "long_term_patterns": "Works remotely",
            },
            "layer6_summary": "Prefers a home working environment",
            "profile_evidence": "Repeatedly works from home",
            "combine": {
                "people": {},
                "locations": {"Home": "home"},
                "activity_events": {},
                "environments": {},
                "objects": {},
            },
        })
    if "TASK: NIGHTLY_PROFILE_EXTRACTION" in prompt:
        return json.dumps({
            "demographics": {},
            "preferences": {"hobbies": [{
                "value": "reading",
                "confidence": 0.9,
                "evidence": "Repeated observations",
            }]},
            "frequent_locations": [{
                "value": "home",
                "confidence": 0.9,
                "evidence": "Repeated observations",
            }],
            "behavior_patterns": {},
            "personality": {},
            "goals": [],
            "decisions": {},
            "motivations": [],
        })
    raise AssertionError("Unexpected prompt type")


def main():
    plan_facts = NightlyProcessor._stated_plans([{
        "id": "clip-plan-1",
        "source_time": {"timestamp": "2026-10-17T10:00:00"},
        "temporal_facts": [{
            "subject_type": "person", "subject": "unidentified speaker",
            "predicate": "stated_plan", "value": "I plan to visit tomorrow.",
            "status": "stated_plan", "modality": "speech",
            "evidence": "I plan to visit tomorrow.",
            "evidence_ref": {"source_segment_id": "clip-plan-1", "modality": "speech"},
        }],
    }])
    assert len(plan_facts) == 1, plan_facts
    assert plan_facts[0]["value"] == "I plan to visit tomorrow."
    assert plan_facts[0]["predicate"] == "stated_plan"
    assert plan_facts[0]["status"] == "stated_plan"
    assert plan_facts[0]["source_moment_id"] == "clip-plan-1"
    assert plan_facts[0]["evidence_ref"]["modality"] == "speech"

    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    with tempfile.TemporaryDirectory(prefix="nightly-memory-") as temp:
        memory = PersonMemory(memory_dir=temp)
        ids = []
        for i in range(6):
            place = "Home" if i % 2 == 0 else "home"
            ids.append(memory.add(
                scene="Home work scene " + ("x" * 350),
                user_action="Reading and working " + ("y" * 250),
                needs=[{"need": "focus", "confidence": 0.8}],
                solutions=[],
                people=["Alex"],
                location=place,
                activity="working",
                environments=["home office"],
                objects=["laptop"],
            ))

        # A previous-day archive item must not be included in today's run.
        memory.memory["layer7"]["moments"][ids[-1]]["timestamp"] = yesterday + "T23:59:00"
        calls = []
        def counted(prompt):
            calls.append(prompt)
            return respond(prompt)

        nightly = NightlyProcessor(memory, llm_call=counted, chunk_chars=1000)
        result = nightly.run(day=today)
        assert result["ok"]
        assert result["moment_count"] == 5, result
        assert result["chunk_count"] > 1, result
        assert result["indices_written"]["people"] == 1
        assert "Alex" in memory.memory["layer7"]["people"]
        assert "Home" not in memory.memory["layer7"]["locations"]
        assert "home" in memory.memory["layer7"]["locations"]
        assert memory.memory["layer4"]["summary"] == "Daily summary"
        assert memory.memory["layer5"]["summary"] == "Long-term summary"
        assert memory.get_profile()["preferences"]["hobbies"][0]["value"] == "reading"
        assert result["profile_updated"]
        assert len(calls) == result["chunk_count"] + 2
        assert memory.memory["metadata"]["nightly_processed_days"][today]["moment_count"] == 5
        print("✅ Nightly gathers the full day's archive and excludes yesterday")
        print("✅ Chunk summaries, six-dimension index repair, alias merge, Layer 4/5 rollup")
        print("✅ Validated profile update and nightly idempotency metadata")

        previous_call_count = len(calls)
        skipped = nightly.run(day=today)
        assert skipped["skipped"] == "unchanged"
        assert len(calls) == previous_call_count
        print("✅ Identical source data skips duplicate model calls")

        # A malformed profile response fails before any persistent consolidation.
        newest = memory.add(
            scene="New event",
            user_action="new event",
            needs=[],
            solutions=[],
            location="office",
            activity="meeting",
            environments=["office"],
        )
        before_summary = memory.memory["layer4"]["summary"]
        before_layer7 = json.dumps(memory.memory["layer7"], sort_keys=True)
        def bad_profile(prompt):
            if "TASK: NIGHTLY_CHUNK_SUMMARY" in prompt:
                return respond(prompt)
            if "TASK: NIGHTLY_ROLLUP" in prompt:
                return respond(prompt)
            return "not json"
        broken = NightlyProcessor(memory, llm_call=bad_profile)
        try:
            broken.run(day=today)
            raise AssertionError("malformed profile should fail")
        except ValueError:
            pass
        assert memory.memory["layer4"]["summary"] == before_summary
        assert json.dumps(memory.memory["layer7"], sort_keys=True) == before_layer7
        assert newest in memory.memory["layer7"]["moments"]
        print("✅ Invalid model output is rejected before consolidation mutates memory")


if __name__ == "__main__":
    main()
