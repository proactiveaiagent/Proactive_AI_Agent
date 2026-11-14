"""Cached scene facts require a verifiable source and keep quality fallbacks."""
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import qwen_perceive as qp


class CachedSceneQualityTests(unittest.TestCase):
    def make_perceiver(self):
        directory = tempfile.TemporaryDirectory(prefix="qwen-cache-quality-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        db_patcher = patch.object(qp, "SOURCE_FINGERPRINT_DB", root / "fingerprints.sqlite3")
        db_patcher.start()
        self.addCleanup(db_patcher.stop)
        video = root / "source.mp4"
        video.write_bytes(b"verifiable source video")
        fingerprint = qp.source_video_input_fingerprint(video, 0.0, 5.0)
        perceiver = qp.QwenPerceiver(cache_file=root / "cache.json")
        return perceiver, video, fingerprint

    def cached_entry(self, fields, fingerprint, prompt_version=None, raw=""):
        return {
            "prompt_version": prompt_version or qp.PROMPT_VERSION,
            "list_parser_version": qp.LIST_PARSER_VERSION,
            "source_input_fingerprint": fingerprint,
            "fields": fields,
            "raw": raw,
        }

    def test_person_appearance_gap_keeps_scene_cache_hit(self):
        perceiver, video, fingerprint = self.make_perceiver()
        moment_id = "clip-with-person-fact"
        fields = {
            "scene": "A person stands near shopping bags.",
            "people": [],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person",
                "subject": "unidentified person 1",
                "actor": "unidentified person 1",
                "predicate": "appeared",
                "value": "standing near shopping bags",
                "affected_people": [],
                "frame_index": 1,
                "evidence": "person visible near shopping bags",
            }],
        }
        perceiver.cache[moment_id] = self.cached_entry(
            fields, fingerprint, raw="Scene: A person stands near shopping bags."
        )
        with patch.object(qp, "video_path_of", return_value=video), \
             patch.object(qp, "clip_range", return_value=(0.0, 5.0)), \
             patch.object(qp.requests, "post", side_effect=AssertionError("cache should avoid inference")):
            result = perceiver.perceive({"id": moment_id, "start_time": 0.0, "end_time": 5.0})
        self.assertEqual(result["source"], "cache")
        self.assertIn("quality_warning", result)
        self.assertIn(moment_id, perceiver.cache)
        self.assertEqual(result["fields"]["temporal_facts"][0]["value"],
                         "standing near shopping bags")

    def test_old_prompt_cache_is_fast_fallback_but_remains_available(self):
        perceiver, video, fingerprint = self.make_perceiver()
        perceiver.cache["clip"] = self.cached_entry(
            {"scene": "A person moves a chair.", "person_observations": [],
             "temporal_facts": []}, fingerprint, prompt_version="old-prompt",
            raw="Scene: A person moves a chair.",
        )
        with patch.object(qp, "video_path_of", return_value=video), \
             patch.object(qp, "clip_range", return_value=(0.0, 5.0)):
            hit = perceiver.perceive(
                {"id": "clip", "start_time": 0.0, "end_time": 5.0},
                allow_legacy_cache=True,
            )
        self.assertEqual(hit["source"], "legacy_cache")
        self.assertEqual(hit["legacy_prompt_version"], "old-prompt")
        self.assertIn("clip", perceiver.cache)

    def test_old_cache_survives_failed_prompt_refresh(self):
        perceiver, _video, _fingerprint = self.make_perceiver()
        moment_id = "clip-with-old-prompt-cache"
        perceiver.cache[moment_id] = {
            "prompt_version": "old-prompt",
            "fields": {"scene": "A person moves a chair.", "person_observations": [],
                       "temporal_facts": []},
            "raw": "Scene: A person moves a chair.",
        }
        with patch.object(qp, "video_path_of", return_value=None):
            result = perceiver.perceive({"id": moment_id})
        self.assertEqual(result["source"], "error")
        self.assertIn(moment_id, perceiver.cache)
        self.assertEqual(perceiver.cache[moment_id]["prompt_version"], "old-prompt")

    def test_cache_instance_reloads_atomic_external_updates(self):
        directory = tempfile.TemporaryDirectory(prefix="qwen-cache-refresh-")
        self.addCleanup(directory.cleanup)
        cache_path = Path(directory.name) / "cache.json"
        writer = qp.QwenPerceiver(cache_file=cache_path)
        reader = qp.QwenPerceiver(cache_file=cache_path)

        writer.cache["first"] = {"prompt_version": qp.PROMPT_VERSION, "ts": "2026-10-18 01:00:00"}
        writer._save_cache()
        self.assertIn("first", reader.cache)

        writer.cache["second"] = {"prompt_version": qp.PROMPT_VERSION, "ts": "2026-10-18 01:00:01"}
        writer._save_cache()
        self.assertIn("second", reader.cache)

    def test_stale_cache_snapshot_cannot_overwrite_newer_version(self):
        directory = tempfile.TemporaryDirectory(prefix="qwen-cache-merge-")
        self.addCleanup(directory.cleanup)
        cache_path = Path(directory.name) / "cache.json"
        stale = qp.QwenPerceiver(cache_file=cache_path)
        writer = qp.QwenPerceiver(cache_file=cache_path)
        old_entry = {"prompt_version": "memory-v7.1-old", "ts": "2026-10-18 01:00:00"}
        current_entry = {
            "prompt_version": qp.PROMPT_VERSION,
            "list_parser_version": qp.LIST_PARSER_VERSION,
            "source_input_fingerprint": {"video_sha256": "v", "clip_sha256": "c"},
            "ts": "2026-10-18 01:00:01",
        }

        stale.cache["clip"] = old_entry
        stale._save_cache()
        self.assertEqual(stale.cache["clip"]["prompt_version"], "memory-v7.1-old")

        writer.cache["clip"] = current_entry
        writer._save_cache()

        # Simulate a dirty old in-memory snapshot that had not noticed the update.
        stale._cache["clip"] = old_entry
        stale._dirty_cache_keys.add("clip")
        stale._save_cache()
        reader = qp.QwenPerceiver(cache_file=cache_path)
        self.assertEqual(reader.cache["clip"]["prompt_version"], qp.PROMPT_VERSION)
        self.assertEqual(reader.cache["clip"]["list_parser_version"], qp.LIST_PARSER_VERSION)

    def test_unsupported_object_move_still_requires_source_recheck(self):
        perceiver, _video, _fingerprint = self.make_perceiver()
        moment_id = "clip-with-unverified-move"
        perceiver.cache[moment_id] = {
            "prompt_version": qp.PROMPT_VERSION,
            "fields": {
                "scene": "A phone is on a table.",
                "person_observations": [],
                "temporal_facts": [{
                    "subject_type": "object",
                    "subject": "phone",
                    "predicate": "moved",
                    "value": "phone moved",
                    "frame_index": 1,
                    "evidence": "phone is visible",
                }],
            },
            "raw": "Scene: A phone is on a table.",
        }
        with patch.object(qp, "video_path_of", return_value=None):
            result = perceiver.perceive({"id": moment_id})
        self.assertEqual(result["source"], "error")
        self.assertIn(moment_id, perceiver.cache)
        self.assertEqual(perceiver.cache[moment_id]["fields"]["temporal_facts"][0]["predicate"],
                         "moved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
