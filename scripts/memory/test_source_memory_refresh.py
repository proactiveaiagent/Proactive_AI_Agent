"""Cache-version refresh updates the same source record without losing corrections."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory


class SourceMemoryRefreshTests(unittest.TestCase):
    def make_memory(self, directory):
        memory = PersonMemory(directory)
        segment = "clip-1"
        moment_id = memory.add(
            scene="An older extraction.",
            user_action="",
            needs=[],
            solutions=[],
            location="courtyard",
            activity="greeting",
            environments=["snowy courtyard"],
            objects=["wooden gate"],
            person_observations=[{
                "entity": "unidentified person 1",
                "appearance": "wearing a blue coat",
                "frame_index": 1,
                "evidence": "blue coat visible",
            }],
            temporal_facts=[{
                "subject_type": "person",
                "subject": "unidentified person 1",
                "predicate": "smiled",
                "value": "at the gate",
                "modality": "visual",
                "frame_index": 1,
                "evidence": "person smiling",
            }, {
                "subject_type": "person",
                "subject": "unidentified speaker",
                "predicate": "stated_plan",
                "value": "visit tomorrow",
                "modality": "speech",
                "audio_start_s": 1.0,
                "audio_end_s": 2.0,
                "evidence_ref": {"source_segment_id": segment, "modality": "speech"},
            }],
            source_provenance={
                "source_video_id": "social/clip.mp4",
                "source_segment_id": segment,
                "clip_start_s": 0.0,
                "clip_end_s": 8.0,
                "source_video_sha256": "a" * 64,
                "source_clip_sha256": "b" * 64,
                "perception_prompt_version": "old-prompt",
                "list_parser_version": "old-parser",
            },
        )
        return memory, moment_id

    def test_refresh_replaces_same_source_fields_and_preserves_speech_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            memory, moment_id = self.make_memory(directory)
            old = memory.memory["layer7"]["moments"][moment_id]
            ok = memory.refresh_source_moment(
                moment_id,
                {
                    "scene": "A smiling elder greets a visitor at a snowy gate.",
                    "person_observations": [{
                        "entity": "unidentified person 1",
                        "appearance": "elderly woman with gray headscarf and blue coat",
                        "apparent_age_range": "75+",
                        "apparent_age_basis": "visual_estimate",
                        "frame_index": 1,
                        "evidence": "face and clothing visible",
                    }],
                    "temporal_facts": [{
                        "subject_type": "person",
                        "subject": "unidentified person 1",
                        "predicate": "smiled",
                        "value": "while greeting visitor",
                        "modality": "visual",
                        "frame_index": 1,
                        "evidence": "smiling while standing at the gate",
                    }, old["temporal_facts"][1]],
                    "objects": ["wooden gate", "red lantern"],
                    "environments": ["snow-covered rural courtyard"],
                },
                {
                    "source_video_id": "social/clip.mp4",
                    "source_segment_id": "clip-1",
                    "clip_start_s": 0.0,
                    "clip_end_s": 8.0,
                    "source_video_sha256": "a" * 64,
                    "source_clip_sha256": "b" * 64,
                    "perception_prompt_version": "new-prompt",
                    "list_parser_version": "new-parser",
                },
                expected_segment_id="clip-1",
            )
            stored = memory.memory["layer7"]["moments"][moment_id]
            reloaded = PersonMemory(directory).memory["layer7"]["moments"][moment_id]

        self.assertTrue(ok)
        self.assertEqual(stored["scene"], "A smiling elder greets a visitor at a snowy gate.")
        self.assertEqual(stored["person_observations"][0]["apparent_age_range"], "75+")
        self.assertEqual(stored["objects"], ["wooden gate", "red lantern"])
        self.assertEqual(len(stored["temporal_facts"]), 2)
        self.assertEqual(stored["temporal_facts"][1]["modality"], "speech")
        self.assertEqual(stored["source_provenance"]["perception_prompt_version"], "new-prompt")
        self.assertEqual(stored["source_refresh_history"][0]["previous_fields"]["scene"], "An older extraction.")
        self.assertEqual(reloaded["source_provenance"]["list_parser_version"], "new-parser")
        self.assertIn(moment_id, memory.memory["layer7"]["objects"]["red lantern"])

    def test_user_corrected_field_is_protected_during_source_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            memory, moment_id = self.make_memory(directory)
            moment = memory.memory["layer7"]["moments"][moment_id]
            moment["feedback"]["corrections"]["scene"] = "User-confirmed scene."
            ok = memory.refresh_source_moment(
                moment_id,
                {"scene": "Model's new scene.", "location": "village gate"},
                {"source_segment_id": "clip-1",
                 "perception_prompt_version": "new-prompt",
                 "list_parser_version": "new-parser"},
                expected_segment_id="clip-1",
            )
            stored = memory.memory["layer7"]["moments"][moment_id]

        self.assertTrue(ok)
        self.assertEqual(stored["scene"], "An older extraction.")
        self.assertEqual(stored["location"], "village gate")
        self.assertEqual(stored["source_provenance"]["perception_prompt_version"], "new-prompt")

    def test_refresh_rejects_different_source_or_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            memory, moment_id = self.make_memory(directory)
            before = memory.memory["layer7"]["moments"][moment_id]["scene"]
            wrong_segment = memory.refresh_source_moment(
                moment_id, {"scene": "Wrong clip."},
                {"source_segment_id": "clip-2"}, expected_segment_id="clip-2")
            wrong_hash = memory.refresh_source_moment(
                moment_id, {"scene": "Changed file."},
                {"source_segment_id": "clip-1", "source_video_sha256": "c" * 64},
                expected_segment_id="clip-1")
            after = memory.memory["layer7"]["moments"][moment_id]["scene"]

        self.assertFalse(wrong_segment)
        self.assertFalse(wrong_hash)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
