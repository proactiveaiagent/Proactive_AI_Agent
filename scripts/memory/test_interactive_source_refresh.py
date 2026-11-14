"""Interactive reruns refresh stale source memories from current cache data."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from memory import PersonMemory
from interactive_runner import InteractiveSession
from qwen_perceive import PROMPT_VERSION, LIST_PARSER_VERSION


class FakePerceiver:
    def __init__(self):
        self.result = {
            "source": "parser_repair",
            "fields": {
                "scene": "A smiling elder greets a visitor at a snowy gate.",
                "user_action": "",
                "interaction_details": "A visitor approaches and places a hand on the elder's arm.",
                "person_observations": [{
                    "entity": "unidentified person 1",
                    "appearance": "elderly woman with a gray headscarf and blue coat",
                    "apparent_age_range": "75+",
                    "apparent_age_basis": "visual_estimate",
                    "frame_index": 1,
                    "evidence": "face and clothing visible",
                }],
                "temporal_facts": [{
                    "subject_type": "person",
                    "subject": "unidentified person 2",
                    "predicate": "approached",
                    "value": "unidentified person 1",
                    "frame_index": 2,
                    "evidence": "visitor moves toward elder",
                    "status": "completed",
                    "confidence": "high",
                }],
                "people": [],
                "location": "snowy rural courtyard",
                "activity": "greeting",
                "environments": ["snow-covered courtyard with a wooden gate"],
                "objects": ["wooden gate", "red lantern"],
                "time_occasion": "",
            },
            "source_input_fingerprint": {
                "video_sha256": "a" * 64,
                "clip_sha256": "b" * 64,
            },
            "prompt_version": PROMPT_VERSION,
            "list_parser_version": LIST_PARSER_VERSION,
            "frame_times_s": [0.0, 2.99],
            "raw": "current source output",
            "elapsed_ms": 1,
            "num_frames": 6,
        }

    def perceive(self, *args, **kwargs):
        return self.result


class InteractiveSourceRefreshTests(unittest.TestCase):
    def test_existing_clip_uses_current_cache_and_updates_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersonMemory(directory)
            mid = memory.add(
                scene="Old scene summary.",
                user_action="",
                needs=[],
                solutions=[],
                location="courtyard",
                objects=["wooden gate"],
                person_observations=[{
                    "entity": "unidentified person 1",
                    "appearance": "elderly woman in blue coat",
                    "frame_index": 1,
                    "evidence": "person visible",
                }],
                temporal_facts=[{
                    "subject_type": "person", "subject": "unidentified speaker",
                    "predicate": "stated_plan", "value": "visit tomorrow",
                    "modality": "speech", "audio_start_s": 1.0, "audio_end_s": 2.0,
                }],
                source_provenance={
                    "source_video_id": "video/clip.mp4",
                    "source_segment_id": "clip-1",
                    "clip_start_s": 0.0, "clip_end_s": 5.0,
                    "source_video_sha256": "a" * 64,
                    "source_clip_sha256": "b" * 64,
                    "perception_prompt_version": "old-prompt",
                    "list_parser_version": "old-parser",
                },
            )
            session = InteractiveSession.__new__(InteractiveSession)
            session.mem = memory
            session.perceiver = FakePerceiver()
            session._active_step_cancel = None
            session._qwen_fields = {}
            session.extraction_batch_id = "test-batch"
            progress = []
            with patch("interactive_runner.video_path_of", return_value=Path("/fake/clip.mp4")), \
                 patch("interactive_runner._source_video_id", return_value="video/clip.mp4"), \
                 patch("interactive_runner.clip_range", return_value=(0.0, 5.0)):
                refreshed = session._refresh_existing_source_record(
                    {"id": "clip-1", "video": "video/clip.mp4",
                     "start_time": 0.0, "end_time": 5.0},
                    mid, False,
                    lambda stage, message, output=None: progress.append((stage, message)),
                )
            stored = memory.memory["layer7"]["moments"][mid]

        self.assertIsNotNone(refreshed)
        self.assertEqual(len(memory.memory["layer7"]["moments"]), 1)
        self.assertEqual(stored["scene"], "A smiling elder greets a visitor at a snowy gate.")
        self.assertEqual(stored["person_observations"][0]["apparent_age_range"], "75+")
        self.assertEqual(stored["person_observations"][0]["evidence_ref"]["frame_time_s"], 0.0)
        self.assertEqual(
            {row.get("modality") for row in stored["temporal_facts"]},
            {"speech", "visual"},
        )
        self.assertEqual(stored["source_provenance"]["perception_prompt_version"], PROMPT_VERSION)
        self.assertEqual(stored["source_provenance"]["list_parser_version"], LIST_PARSER_VERSION)
        self.assertEqual(stored["source_provenance"]["source_clip_sha256"], "b" * 64)
        self.assertTrue(stored["source_refresh_history"])
        self.assertTrue(progress)


if __name__ == "__main__":
    unittest.main(verbosity=2)
