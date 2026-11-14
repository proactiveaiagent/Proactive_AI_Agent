"""Optional real-model smoke for NightlyProcessor; requires local Qwen API."""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from memory import PersonMemory
from nighttime_processor import NightlyProcessor


def main():
    with tempfile.TemporaryDirectory(prefix="nightly-qwen-smoke-") as folder:
        memory = PersonMemory(memory_dir=folder)
        samples = [
            ("Home", "Reading a novel after dinner", "reading"),
            ("Home", "Reading another chapter before bed", "reading"),
            ("Office", "Preparing slides for a morning meeting", "working"),
            ("Office", "Reviewing project notes and planning tomorrow", "planning"),
        ]
        for place, action, activity in samples:
            memory.add(
                scene=f"User is at {place}",
                user_action=action,
                needs=[],
                solutions=[],
                people=[],
                location=place,
                activity=activity,
                environments=[place.lower()],
                objects=["book"] if activity == "reading" else ["laptop"],
            )
        report = NightlyProcessor(memory).run(day=date.today().isoformat())
        assert report["ok"], report
        assert report["moment_count"] == len(samples), report
        assert memory.memory["layer4"]["summary"], "Layer 4 summary was empty"
        assert memory.memory["layer5"]["summary"], "Layer 5 summary was empty"
        assert memory.memory["metadata"]["nightly_processed_days"], "No idempotency record"
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
