"""Regression tests for source-fingerprint-aware perception prewarming."""
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "code"))
import prewarm_multimodal_evidence as prewarm


class PrewarmSourceFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.moment = {"id": "clip-1"}
        self.patches = [
            patch.object(prewarm, "video_path_of", return_value=Path("clip.mp4")),
            patch.object(prewarm, "clip_range", return_value=(1.0, 4.0)),
            patch.object(prewarm, "source_video_input_fingerprint", return_value="source-v1"),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_matching_fingerprint_is_current(self):
        self.assertTrue(prewarm._source_fingerprint_matches(
            {"source_input_fingerprint": "source-v1"}, self.moment
        ))

    def test_missing_fingerprint_is_not_a_cache_hit(self):
        self.assertFalse(prewarm._source_fingerprint_matches({}, self.moment))

    def test_changed_fingerprint_is_not_a_cache_hit(self):
        self.assertFalse(prewarm._source_fingerprint_matches(
            {"source_input_fingerprint": "source-old"}, self.moment
        ))


    def test_parser_upgrade_immediately_repairs_remaining_quality_gap(self):
        entry = {
            "prompt_version": prewarm.PROMPT_VERSION,
            "source_input_fingerprint": "source-v1",
            "list_parser_version": "old-parser",
            "raw": "source text",
            "fields": {"person_observations": [], "temporal_facts": []},
        }
        parsed_fields = {"person_observations": [], "temporal_facts": []}
        repaired_fields = {"person_observations": [{"entity": "unidentified person 1"}]}
        perceiver = Mock()
        perceiver.perceive.return_value = {
            "source": "parser_repair", "fields": parsed_fields
        }
        with patch.object(prewarm, "_structured_fact_quality_error",
                          return_value="person gap"):
            with patch.object(prewarm, "repair_cached_quality_gap",
                              return_value={"source": "qwen_repair",
                                            "fields": repaired_fields}) as repair:
                result = prewarm.resolve_cached_perception(perceiver, self.moment, entry)
        self.assertEqual(result["source"], "qwen_repair")
        perceiver.perceive.assert_called_once()
        repair.assert_called_once_with(perceiver, self.moment)

    def test_parser_upgrade_skips_model_repair_when_quality_is_complete(self):
        entry = {
            "prompt_version": prewarm.PROMPT_VERSION,
            "source_input_fingerprint": "source-v1",
            "list_parser_version": "old-parser",
            "raw": "source text",
            "fields": {"person_observations": [], "temporal_facts": []},
        }
        perceiver = Mock()
        perceiver.perceive.return_value = {
            "source": "parser_repair", "fields": {"person_observations": []}
        }
        with patch.object(prewarm, "_structured_fact_quality_error", return_value=""):
            with patch.object(prewarm, "repair_cached_quality_gap") as repair:
                result = prewarm.resolve_cached_perception(perceiver, self.moment, entry)
        self.assertEqual(result["source"], "parser_repair")
        perceiver.perceive.assert_called_once()
        repair.assert_not_called()


    def test_scene_person_review_ignores_first_person_and_explicit_negation(self):
        self.assertTrue(prewarm.needs_person_observation_review({
            "scene": "A woman in a blue coat stands at the doorway.",
            "person_observations": [],
        }))
        self.assertFalse(prewarm.needs_person_observation_review({
            "scene": "First-person view as the camera wearer opens a drawer.",
            "person_observations": [],
        }))
        self.assertFalse(prewarm.needs_person_observation_review({
            "scene": "No people are visible in the empty room.",
            "person_observations": [],
        }))

    def test_scene_review_repairs_people_using_source_frames_and_persists_status(self):
        fields = {
            "scene": "A man in a suit stands at a desk.",
            "people": [],
            "person_observations": [],
            "temporal_facts": [],
            "objects": [],
        }
        entry = {
            "prompt_version": prewarm.PROMPT_VERSION,
            "source_input_fingerprint": "source-v1",
            "fields": fields,
            "raw": "Scene: A man in a suit stands at a desk.\nPerson Observations: []",
        }
        perceiver = Mock()
        perceiver.cache = {"clip-1": entry}
        perceiver._cache_lock = threading.RLock()
        perceiver.num_frames = 6
        perceiver.next_api_url.return_value = "http://127.0.0.1:8000"
        response = Mock()
        response.json.return_value = {"analysis": json.dumps({
            "person_observations": [{
                "entity": "unidentified person 1",
                "appearance": "adult man with short dark hair wearing a suit",
                "frame_index": 1,
                "evidence": "face and dark suit visible in frame 1",
                "confidence": "high",
            }],
            "entity_map": [],
        })}
        with patch.object(prewarm, "extract_frames", return_value=([b"sampled-frame"], [0.0])):
            with patch.object(prewarm.requests, "post", return_value=response) as post:
                result = prewarm.repair_cached_quality_gap(perceiver, self.moment)
        self.assertEqual(result["source"], "qwen_repair")
        self.assertEqual(len(result["fields"]["person_observations"]), 1)
        updated = perceiver.cache["clip-1"]
        self.assertEqual(updated["person_observation_review"]["status"], "observations_found")
        self.assertIn("unidentified person 1", updated["raw"])
        self.assertEqual(post.call_args.kwargs["files"][0][1][1], b"sampled-frame")
        self.assertEqual(set(post.call_args.kwargs["data"]), {"prompt", "request_id"})
        self.assertNotIn("FEISHU", post.call_args.kwargs["data"]["prompt"])

    def test_scene_review_marks_sampled_frames_with_no_visible_person(self):
        fields = {
            "scene": "A person is described near a doorway.",
            "people": [],
            "person_observations": [],
            "temporal_facts": [],
        }
        entry = {
            "prompt_version": prewarm.PROMPT_VERSION,
            "source_input_fingerprint": "source-v1",
            "fields": fields,
            "raw": "Scene: A person is described near a doorway.\nPerson Observations: []",
        }
        perceiver = Mock()
        perceiver.cache = {"clip-1": entry}
        perceiver._cache_lock = threading.RLock()
        perceiver.num_frames = 6
        perceiver.next_api_url.return_value = "http://127.0.0.1:8000"
        response = Mock()
        response.json.return_value = {
            "analysis": json.dumps({"person_observations": [], "entity_map": []})
        }
        with patch.object(prewarm, "extract_frames", return_value=([b"sampled-frame"], [0.0])):
            with patch.object(prewarm.requests, "post", return_value=response):
                result = prewarm.repair_cached_quality_gap(perceiver, self.moment)
        updated = perceiver.cache["clip-1"]
        self.assertEqual(result["source"], "qwen_repair")
        self.assertEqual(updated["person_observation_review"]["status"],
                         "no_person_visible_in_sampled_frames")
        self.assertFalse(prewarm.needs_person_observation_review(
            updated["fields"], updated
        ))



    def test_scene_review_salvages_unmatched_person_references_after_retries(self):
        fields = {
            "scene": "A woman in a blue coat places a cup on a table.",
            "people": ["unidentified person 1", "unidentified person 2"],
            "person_observations": [{
                "entity": "unidentified person 1",
                "appearance": "woman in a blue coat",
                "frame_index": 1,
                "evidence": "blue coat visible in frame 1",
            }],
            "temporal_facts": [
                {"subject_type": "person", "subject": "unidentified person 1",
                 "actor": "unidentified person 1", "predicate": "placed",
                 "value": "places a cup on a table", "affected_people": [],
                 "frame_index": 1, "evidence": "woman reaches toward the table"},
                {"subject_type": "person", "subject": "unidentified person 2",
                 "actor": "unidentified person 2", "predicate": "approached",
                 "value": "approached the doorway", "affected_people": [],
                 "frame_index": 1, "evidence": "second figure near the doorway"},
                {"subject_type": "object", "subject": "cup", "predicate": "rested",
                 "value": "on the table", "affected_people": [],
                 "frame_index": 1, "evidence": "cup visible on the table"},
            ],
        }
        raw = (
            "Scene: A woman in a blue coat places a cup on a table.\n"
            "People: unidentified person 1, unidentified person 2\n"
            "Person Observations: original model output"
        )
        entry = {
            "prompt_version": prewarm.PROMPT_VERSION,
            "source_input_fingerprint": "source-v1",
            "fields": fields,
            "raw": raw,
        }
        perceiver = Mock()
        perceiver.cache = {"clip-1": entry}
        perceiver._cache_lock = threading.RLock()
        perceiver.num_frames = 6
        perceiver.next_api_url.return_value = "http://127.0.0.1:8000"
        response = Mock()
        response.json.return_value = {"analysis": json.dumps({
            "person_observations": [{
                "entity": "unidentified person 1",
                "appearance": "woman in a blue coat",
                "frame_index": 1,
                "evidence": "blue coat visible in frame 1",
                "confidence": "high",
            }],
            "entity_map": [],
        })}
        with patch.object(prewarm, "extract_frames", return_value=([b"sampled-frame"], [0.0])):
            with patch.object(prewarm.requests, "post", return_value=response) as post:
                result = prewarm.repair_cached_quality_gap(perceiver, self.moment)
        self.assertEqual(post.call_count, 3)
        self.assertEqual(result["source"], "quality_repair")
        self.assertEqual(result["fields"]["people"], ["unidentified person 1"])
        self.assertEqual(
            [row["subject"] for row in result["fields"]["temporal_facts"]],
            ["unidentified person 1", "cup"],
        )
        updated = perceiver.cache["clip-1"]
        self.assertEqual(
            updated["person_observation_review"]["status"],
            "unresolved_references_removed_after_source_review",
        )
        self.assertEqual(updated["raw_model_output"], raw)
        self.assertIn("unidentified person 2", updated["quality_repair"]["unverified_person_claims_removed"][0]["entity"])
        self.assertFalse(prewarm.needs_person_observation_review(
            updated["fields"], updated
        ))


    def test_dry_run_does_not_write_grounding_source_alias(self):
        moment = {"id": "clip-1"}
        entry = {
            "prompt_version": prewarm.PROMPT_VERSION,
            "list_parser_version": prewarm.LIST_PARSER_VERSION,
            "source_input_fingerprint": "source-v1",
            "fields": {
                "scene": "A phone rests on a table.",
                "people": [], "person_observations": [],
                "temporal_facts": [], "objects": ["phone"],
            },
        }
        perceiver = Mock(cache={"clip-1": entry})
        grounding = Mock()
        grounding.source_key.return_value = "source-key"
        grounding.get_source.return_value = (None, None)
        grounding.key.return_value = "frame-key"
        grounding.get.return_value = ({"items": []}, None)
        with patch.object(prewarm, "load_moments", return_value=[moment]), \
             patch.object(prewarm, "QwenPerceiver", return_value=perceiver), \
             patch.object(prewarm, "PersistentObjectGroundingCache", return_value=grounding), \
             patch.object(prewarm, "_source_fingerprint_matches", return_value=True), \
             patch.object(prewarm, "video_path_of", return_value=Path("clip.mp4")), \
             patch.object(prewarm, "clip_range", return_value=(0.0, 2.0)), \
             patch.object(prewarm.MultimodalAssetStore, "sample_visual_candidates",
                          return_value=[{"bytes": b"frame"}]), \
             patch.object(prewarm, "video_audio_chunks", return_value=iter(())), \
             patch.object(prewarm, "PersistentTranscriptCache", return_value=Mock()), \
             patch.object(prewarm, "log"):
            result = prewarm.run(dry_run=True)
        self.assertEqual(result["grounding_hit"], 1)
        grounding.put_source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
