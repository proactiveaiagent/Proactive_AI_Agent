"""Semantic-index document and synchronization regression tests."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from semantic_retrieval import (
    DOCUMENT_VERSION, PersistentSemanticIndex, memory_text,
)


class FakeEncoder:
    model_id = "test-encoder-v1"

    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size=32):
        import numpy as np
        self.calls.extend(texts)
        vectors = []
        for text in texts:
            raw = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) + 1
            vectors.append(vector / np.linalg.norm(vector))
        return vectors


class SemanticMemoryTextTests(unittest.TestCase):
    def test_document_indexes_observed_details_and_excludes_advice_fields(self):
        moment = {
            "scene": "Two people meet at a snowy gate.",
            "interaction_details": "The younger woman approaches and embraces the older woman.",
            "location": "rural courtyard",
            "source_time": {"day_label": "DAY2", "time_of_day": "morning"},
            "person_observations": [
                {
                    "entity": "unidentified person 1",
                    "appearance": "grey headscarf and blue padded jacket",
                    "apparent_age_range": "60-74",
                    "apparent_age_basis": "visual_estimate",
                    "age": "71",
                    "age_basis": "unknown",
                    "evidence": "visible near the gate",
                }
            ],
            "temporal_facts": [
                {
                    "subject_type": "object",
                    "subject": "blue folder",
                    "predicate": "moved",
                    "value": "placed on the desk",
                    "actor": "unidentified person 2",
                    "from_location": "lap",
                    "to_location": "desk",
                    "time_scope": "00:10-00:15",
                    "frame_time_s": 12.5,
                    "time_order": 3,
                    "evidence": "folder visible on desk in later frame",
                }
            ],
            "object_observations": [
                {"name": "blue folder", "state": "on the desk", "description": "rectangular"}
            ],
            "needs": ["diagnose why a person fainted"],
            "solutions": ["call emergency services"],
        }

        text = memory_text(moment)

        for expected in (
            "grey headscarf", "60-74", "blue folder", "from location: lap",
            "to location: desk", "DAY2", "morning", "rural courtyard",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("71", text)
        self.assertNotIn("diagnose why", text)
        self.assertNotIn("call emergency services", text)

    def test_exact_age_is_indexed_only_when_visible_text_supports_it(self):
        text = memory_text({
            "person_observations": [
                {"entity": "person A", "age": "42", "age_basis": "visible_text"}
            ]
        })
        self.assertIn("age stated in visible text: 42", text)


class SemanticIndexSyncTests(unittest.TestCase):
    def test_versioned_incremental_sync_and_inactive_record_removal(self):
        encoder = FakeEncoder()
        main = {
            "id": "m1",
            "scene": "A person checks a blue folder.",
            "merged_sources": ["m2"],
        }
        source = {
            "id": "m2",
            "merged_into": "m1",
            "interaction_details": "The folder was moved from lap to desk.",
        }
        stale = {"id": "m3", "scene": "stale record", "stale": True}
        already_merged = {"id": "m4", "scene": "merged record", "merged_into": "m1"}
        moments = {"m1": main, "m2": source, "m3": stale, "m4": already_merged}

        with tempfile.TemporaryDirectory(prefix="semantic-index-test-") as directory:
            index = PersistentSemanticIndex(directory)
            self.assertFalse(index.ready(moments, encoder, revision=1))
            first = index.sync(moments, encoder, revision=1)
            self.assertEqual(first["active"], 1)
            self.assertEqual(first["added_or_updated"], 1)
            self.assertTrue(index.ready(moments, encoder, revision=1))
            self.assertIn("moved from lap to desk", encoder.calls[0])

            main["scene"] = "A person puts the folder on the desk."
            changed = index.sync(moments, encoder, revision=2)
            self.assertEqual(changed["added_or_updated"], 1)
            self.assertTrue(index.ready(moments, encoder, revision=2))

            with index._connect() as db:
                db.execute("DELETE FROM vectors WHERE moment_id='m1'")
            self.assertFalse(index.ready(moments, encoder, revision=2))
            repaired = index.sync(moments, encoder, revision=2)
            self.assertEqual(repaired["added_or_updated"], 1)
            self.assertTrue(index.ready(moments, encoder, revision=2))

            moments["m5"] = {"id": "m5", "scene": "Temporary note."}
            added = index.sync(moments, encoder, revision=3)
            self.assertEqual(added["added_or_updated"], 1)
            del moments["m5"]
            deleted = index.sync(moments, encoder, revision=4)
            self.assertEqual(deleted["removed"], 1)
            self.assertTrue(index.ready(moments, encoder, revision=4))

            main["stale"] = True
            removed = index.sync(moments, encoder, revision=5)
            self.assertEqual(removed["removed"], 1)
            self.assertEqual(index.stats()["vectors"], 0)
            self.assertTrue(index.ready(moments, encoder, revision=5))

    def test_old_document_version_is_not_ready_until_rebuilt(self):
        encoder = FakeEncoder()
        with tempfile.TemporaryDirectory(prefix="semantic-index-version-") as directory:
            index = PersistentSemanticIndex(directory)
            with index._connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO index_meta(key,value) VALUES('model_id',?)",
                    (encoder.model_id,),
                )
                db.execute(
                    "INSERT OR REPLACE INTO index_meta(key,value) VALUES('document_version',?)",
                    ("1",),
                )
            self.assertNotEqual(DOCUMENT_VERSION, "1")
            self.assertFalse(index.ready({}, encoder, revision=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
