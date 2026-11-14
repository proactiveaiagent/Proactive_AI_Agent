"""Source video fingerprints survive canonical memory writes and reloads."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory
from entity_records import EntityRecordIndex, memory_facts


class SourceProvenanceMemoryTests(unittest.TestCase):
    def test_source_hashes_survive_memory_write_reload_and_entity_projection(self):
        video_hash = "a" * 64
        clip_hash = "b" * 64
        with tempfile.TemporaryDirectory(prefix="source-provenance-memory-") as directory:
            memory = PersonMemory(directory)
            moment_id = memory.add(
                scene="A person places a phone on a table.",
                user_action="",
                needs=[],
                solutions=[],
                source_provenance={
                    "source_video_id": "DAY1_A1_JAKE_12000000",
                    "source_segment_id": "clip-1",
                    "clip_start_s": 0.0,
                    "clip_end_s": 4.0,
                    "source_video_sha256": video_hash,
                    "source_clip_sha256": clip_hash,
                    "perception_prompt_version": "memory-v7.4-test",
                    "list_parser_version": "json-arrays-test",
                },
            )
            reloaded = PersonMemory(directory)
            stored = reloaded.memory["layer7"]["moments"][moment_id]
            projected = memory_facts(stored)

        self.assertEqual(stored["source_provenance"]["source_video_sha256"], video_hash)
        self.assertEqual(stored["source_provenance"]["source_clip_sha256"], clip_hash)
        self.assertEqual(stored["source_provenance"]["perception_prompt_version"], "memory-v7.4-test")
        self.assertEqual(stored["source_provenance"]["list_parser_version"], "json-arrays-test")
        self.assertEqual(projected["source_provenance"]["source_video_sha256"], video_hash)
        self.assertEqual(projected["source_provenance"]["source_clip_sha256"], clip_hash)



    def test_rich_visual_facts_and_transition_sources_survive_retrieval_projection(self):
        segment = "clip-rich-1"
        ref_before = {
            "source_segment_id": segment, "modality": "visual",
            "frame_index": 2, "frame_time_s": 2.99,
            "clip_start_s": 0.0, "clip_end_s": 15.1,
        }
        ref_after = {
            "source_segment_id": segment, "modality": "visual",
            "frame_index": 4, "frame_time_s": 8.98,
            "clip_start_s": 0.0, "clip_end_s": 15.1,
        }
        with tempfile.TemporaryDirectory(prefix="rich-source-memory-") as directory:
            memory = PersonMemory(directory)
            moment_id = memory.add(
                scene="An elderly woman greets a visitor at a snowy courtyard gate.",
                user_action="", needs=[], solutions=[],
                people=["unidentified person 1"],
                location="rural courtyard", activity="greeting a visitor",
                environments=["snow-covered courtyard"], objects=["phone"],
                interaction_details="The visitor approaches and places a hand on her shoulder.",
                person_observations=[{
                    "entity": "unidentified person 1",
                    "appearance": "elderly woman wearing a grey headscarf and blue padded jacket",
                    "apparent_age_range": "60-74", "apparent_age_basis": "visual_estimate",
                    "frame_index": 1, "frame_time_s": 0.0,
                    "evidence": "grey headscarf and blue jacket visible in frame 1",
                    "confidence": "high",
                    "evidence_ref": {**ref_before, "frame_index": 1, "frame_time_s": 0.0},
                }],
                temporal_facts=[{
                    "subject_type": "object", "subject": "phone", "predicate": "moved",
                    "value": "moved from the table to a shelf", "from_location": "table",
                    "to_location": "shelf", "actor": "unidentified person 1",
                    "frame_index": 2, "frame_time_s": 2.99, "time_scope": "frames 2 to 4",
                    "evidence": "phone is on the table in frame 2 and on the shelf in frame 4",
                    "evidence_ref": ref_before, "evidence_refs": [ref_before, ref_after],
                    "status": "completed", "confidence": "high",
                }],
                source_time={"day_label": "DAY1", "time_of_day": "11:09:30",
                             "clip_start_s": 0.0, "clip_end_s": 15.1,
                             "basis": "dataset_relative", "source_segment_id": segment},
                source_provenance={"source_video_id": "A4_LUCIA/DAY1/clip.mp4",
                                   "source_segment_id": segment, "clip_start_s": 0.0,
                                   "clip_end_s": 15.1, "source_video_sha256": "c" * 64},
                multimodal={"schema": 1, "source_id": segment,
                            "visual_frames": [{"asset": "assets/visual/frame.webp",
                                                 "at_s": 2.99, "source_id": segment}],
                            "object_visuals": {"status": "completed", "items": [{
                                "object": "phone", "appearance": "black rectangular device",
                                "status": "grounded", "frame_refs": [{
                                    "asset": "assets/visual/phone.webp", "at_s": 2.99,
                                    "source_id": segment,
                                }],
                            }] }},
            )
            reloaded = PersonMemory(directory)
            stored = reloaded.memory["layer7"]["moments"][moment_id]
            facts = memory_facts(stored)
            entities = EntityRecordIndex(reloaded.memory["layer7"])
            person = entities.query(kind="person", query="unidentified person 1")["items"][0]
            obj = entities.query(kind="object", query="phone")["items"][0]

        self.assertEqual(facts["person_observations"][0]["apparent_age_range"], "60-74")
        self.assertEqual(facts["person_observations"][0]["evidence_ref"]["frame_time_s"], 0.0)
        transition = facts["temporal_facts"][0]
        self.assertEqual((transition["from_location"], transition["to_location"]), ("table", "shelf"))
        self.assertEqual([ref["frame_time_s"] for ref in transition["evidence_refs"]], [2.99, 8.98])
        self.assertEqual(facts["source_time"]["time_of_day"], "11:09:30")
        self.assertEqual(facts["object_observations"][0]["appearance"], "black rectangular device")
        self.assertEqual(person["events"][0]["temporal_facts"][0]["evidence_refs"][1]["frame_time_s"], 8.98)
        self.assertEqual(obj["events"][0]["object_observations"][0]["evidence_refs"][0]["frame_time_s"], 2.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
