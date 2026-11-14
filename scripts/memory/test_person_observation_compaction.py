"""Regression tests for compact, source-linked person observations."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import _clean_source_fact_rows


class PersonObservationCompactionTests(unittest.TestCase):
    def test_repeated_appearance_compacts_but_keeps_all_source_times(self):
        rows = []
        for i, t in enumerate((0.0, 2.99, 5.99, 8.98, 11.97, 14.97), 1):
            rows.append({
                "entity": "unidentified person 1",
                "appearance": "wearing a gray knit hat and a dark blue jacket",
                "frame_index": i,
                "frame_time_s": t,
                "evidence_ref": {
                    "source_segment_id": "clip-1.3",
                    "modality": "visual",
                    "frame_index": i,
                    "frame_time_s": t,
                },
            })
        rows.extend([
            {"entity": "unidentified person 2", "appearance": "pink hat"},
            {"entity": "unidentified person 3", "appearance": "red coat"},
        ])
        compact = _clean_source_fact_rows(rows, "person", 2)
        self.assertEqual(len(compact), 2)
        self.assertEqual(compact[0]["entity"], "unidentified person 1")
        self.assertEqual(len(compact[0]["evidence_refs"]), 6)
        self.assertEqual(compact[0]["evidence_ref"]["frame_time_s"], 0.0)
        self.assertEqual(compact[1]["entity"], "unidentified person 2")

    def test_distinct_appearance_or_explicit_age_stays_separate(self):
        rows = [
            {"entity": "Alice", "appearance": "blue coat", "age_basis": "stated",
             "age": "32", "evidence_ref": {"frame_time_s": 0.0}},
            {"entity": "Alice", "appearance": "red coat", "age_basis": "stated",
             "age": "32", "evidence_ref": {"frame_time_s": 1.0}},
            {"entity": "Alice", "appearance": "blue coat", "age_basis": "stated",
             "age": "33", "evidence_ref": {"frame_time_s": 2.0}},
        ]
        compact = _clean_source_fact_rows(rows, "person", 8)
        self.assertEqual(len(compact), 3)
        self.assertEqual([row["age"] for row in compact], ["32", "32", "33"])

    def test_face_features_survive_and_distinguish_person_observations(self):
        rows = [
            {"entity": "unidentified person 1", "appearance": "blue jacket",
             "face_features": "round face, glasses", "frame_index": 1,
             "evidence_ref": {"frame_time_s": 1.0}},
            {"entity": "unidentified person 1", "appearance": "blue jacket",
             "face_features": "long face, no visible glasses", "frame_index": 2,
             "evidence_ref": {"frame_time_s": 2.0}},
        ]
        compact = _clean_source_fact_rows(rows, "person", 8)
        self.assertEqual(len(compact), 2)
        self.assertEqual([row["face_features"] for row in compact], [
            "round face, glasses", "long face, no visible glasses"
        ])

    def test_age_without_explicit_basis_is_removed_before_compaction(self):
        rows = [
            {"entity": "person", "appearance": "coat", "age": "elderly",
             "age_basis": "unknown", "frame_index": i}
            for i in (1, 2, 3)
        ]
        compact = _clean_source_fact_rows(rows, "person", 8)
        self.assertEqual(len(compact), 1)
        self.assertNotIn("age", compact[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
