"""Offline tests for video-only person-observation repair."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import qwen_perceive as qp


class PersonRepairTests(unittest.TestCase):
    def make_person(self, entity, frame_index=1, appearance="navy jacket"):
        return {
            "entity": entity,
            "appearance": appearance,
            "frame_index": frame_index,
            "evidence": f"{appearance} visible in frame {frame_index}",
            "confidence": "high",
        }

    def test_person_repair_preserves_separate_face_features(self):
        fields = {"people": [], "person_observations": [], "temporal_facts": []}
        payload = {"person_observations": [{
            **self.make_person("unidentified person 1", appearance="blue coat"),
            "face_features": "round face and glasses",
        }]}
        repaired = qp._apply_person_repair(fields, payload, [], frame_count=6)
        self.assertEqual(repaired["person_observations"][0]["face_features"],
                         "round face and glasses")

    def test_parse_fenced_json(self):
        fence = chr(96) * 3
        payload = qp._parse_person_repair(
            f"{fence}json\n{{\"person_observations\": []}}\n{fence}"
        )
        self.assertEqual(payload, {"person_observations": []})

    def test_parse_accepts_fenced_array_output(self):
        fence = chr(96) * 3
        row = self.make_person("unidentified person 1", 1, "blue jacket")
        payload = qp._parse_person_repair(
            f"{fence}json\n{json.dumps([row])}\n{fence}"
        )
        self.assertEqual(len(payload["person_observations"]), 1)
        self.assertEqual(payload["person_observations"][0]["entity"], "unidentified person 1")

    def test_parse_ignores_trailing_model_text_after_json(self):
        payload = qp._parse_person_repair(
            '{"person_observations": []}\nAdditional output: {"entity_map": []}'
        )
        self.assertEqual(payload, {"person_observations": []})

    def test_deduplicates_frames_and_preserves_neutral_id(self):
        fields = {
            "scene": "A person stands by a desk.",
            "people": [],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "unidentified person 1",
                "actor": "unidentified person 1", "affected_people": [],
            }],
        }
        payload = {"person_observations": [
            self.make_person("unidentified person 1", 1),
            self.make_person("unidentified person 1", 2),
        ]}
        repaired = qp._apply_person_repair(
            fields, payload, qp._person_reference_labels(fields), frame_count=6
        )
        self.assertEqual(len(repaired["person_observations"]), 1)
        self.assertEqual(repaired["person_observations"][0]["frame_index"], 1)
        self.assertEqual(qp._structured_fact_quality_error(repaired), "")

    def test_single_visible_reference_is_mapped_to_neutral_id(self):
        fields = {
            "scene": "A staff member puts a folder on a desk.",
            "people": ["staff member"],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "staff member",
                "actor": "staff member", "affected_people": [],
            }],
        }
        payload = {"person_observations": [
            self.make_person("unidentified person 1")
        ]}
        repaired = qp._apply_person_repair(
            fields, payload, qp._person_reference_labels(fields), frame_count=6
        )
        self.assertEqual(repaired["people"], ["unidentified person 1"])
        self.assertEqual(
            repaired["temporal_facts"][0]["actor"], "unidentified person 1"
        )
        self.assertEqual(qp._structured_fact_quality_error(repaired), "")

    def test_rejects_out_of_range_or_unproven_rows(self):
        fields = {
            "scene": "A person stands by a desk.",
            "people": ["unidentified person 1"],
            "person_observations": [],
            "temporal_facts": [],
        }
        payload = {"person_observations": [
            self.make_person("unidentified person 1", frame_index=7)
        ]}
        repaired = qp._apply_person_repair(
            fields, payload, qp._person_reference_labels(fields), frame_count=6
        )
        self.assertEqual(repaired["person_observations"], [])
        self.assertTrue(qp._structured_fact_quality_error(repaired))


    def test_maps_role_reference_only_to_model_grounded_neutral_id(self):
        fields = {
            "scene": "A staff member hands a receipt across the counter.",
            "people": ["staff member"],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "staff member",
                "actor": "staff member", "affected_people": [],
            }],
        }
        references = qp._person_reference_labels(fields)
        visual_pass = {
            "person_observations": [
                self.make_person("unidentified person 1", 1, "bald person behind counter"),
                self.make_person("unidentified person 2", 2, "blue shirt and name tag"),
            ]
        }
        repaired = qp._apply_person_repair(fields, visual_pass, references, frame_count=6)
        alias_pass = {
            "entity_map": [{
                "reference": "staff member",
                "entity": "unidentified person 2",
            }]
        }
        mapped = qp._apply_person_repair(repaired, alias_pass, references, frame_count=6)
        self.assertEqual(mapped["people"], ["unidentified person 2"])
        self.assertEqual(mapped["temporal_facts"][0]["actor"], "unidentified person 2")
        self.assertEqual(qp._structured_fact_quality_error(mapped), "")

    def test_repair_prompt_preserves_existing_neutral_ids_exactly(self):
        prompt = qp._build_person_repair_prompt(
            {"existing_person_labels": ["unidentified person 2", "the woman in a red coat"]},
            ["unidentified person 2", "the woman in a red coat"],
        )
        self.assertIn('"required_neutral_entity_ids": ["unidentified person 2"]', prompt)
        self.assertIn('"descriptive_person_references": ["the woman in a red coat"]', prompt)
        self.assertIn("Do not renumber, swap, or replace these IDs", prompt)

    def test_repair_keeps_supported_broad_age_range_but_not_guessed_exact_age(self):
        fields = {
            "people": ["unidentified person 1"],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "unidentified person 1",
                "actor": "unidentified person 1", "affected_people": [],
            }],
        }
        person = self.make_person("unidentified person 1")
        person.update({
            "apparent_age_range": "60-74",
            "apparent_age_basis": "visual_estimate",
            "age": "68",
            "age_basis": "visual_estimate",
        })
        repaired = qp._apply_person_repair(
            fields, {"person_observations": [person]},
            qp._person_reference_labels(fields), frame_count=6,
        )
        row = repaired["person_observations"][0]
        self.assertEqual(row["apparent_age_range"], "60-74")
        self.assertEqual(row["apparent_age_basis"], "visual_estimate")
        self.assertEqual(row["age"], "")
        self.assertEqual(row["age_basis"], "unknown")

    def test_repair_rejects_unlisted_apparent_age_range(self):
        fields = {
            "people": ["unidentified person 1"],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "unidentified person 1",
                "actor": "unidentified person 1", "affected_people": [],
            }],
        }
        person = self.make_person("unidentified person 1")
        person["apparent_age_range"] = "elderly"
        repaired = qp._apply_person_repair(
            fields, {"person_observations": [person]},
            qp._person_reference_labels(fields), frame_count=6,
        )
        row = repaired["person_observations"][0]
        self.assertEqual(row["apparent_age_range"], "")
        self.assertEqual(row["apparent_age_basis"], "unknown")

    def test_neutral_entity_map_cannot_swap_numbered_clip_ids(self):
        fields = {
            "scene": "Two people stand in a room.",
            "people": ["unidentified person 1"],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "unidentified person 1",
                "actor": "unidentified person 1", "affected_people": [],
            }],
        }
        payload = {
            "person_observations": [self.make_person("unidentified person 2")],
            "entity_map": [{
                "reference": "unidentified person 1",
                "entity": "unidentified person 2",
            }],
        }
        repaired = qp._apply_person_repair(
            fields, payload, qp._person_reference_labels(fields), frame_count=6
        )
        self.assertEqual(repaired["temporal_facts"][0]["subject"], "unidentified person 1")
        self.assertTrue(qp._structured_fact_quality_error(repaired))

    def test_scene_text_alone_does_not_invent_person_appearance(self):
        fields = {
            "scene": "A woman walks through a quiet terminal.",
            "people": [],
            "person_observations": [],
            "temporal_facts": [],
        }
        self.assertEqual(qp._structured_fact_quality_error(fields), "")


    def test_replace_multiline_person_array_preserves_adjacent_fields(self):
        raw = (
            "Scene: A person opens a gate.\n"
            "Person Observations: [\n{\"entity\":\"broken\"}\n"
            "Temporal Facts: []\nActivity: Opening a gate"
        )
        repaired = qp._replace_field_value(
            raw, "Person Observations",
            '[{\"entity\":\"unidentified person 1\",\"appearance\":\"blue coat\",'
            '\"frame_index\":1,\"evidence\":\"blue coat visible\"}]'
        )
        fields = qp.parse_analysis(repaired, strict=True)
        self.assertEqual(fields["scene"], "A person opens a gate.")
        self.assertEqual(fields["person_observations"][0]["entity"], "unidentified person 1")
        self.assertEqual(fields["activity"], "Opening a gate")

    def test_replace_appends_missing_field(self):
        repaired = qp._replace_field_value(
            "Scene: A person waves.\nTemporal Facts: []",
            "Person Observations", "[]"
        )
        self.assertIn("Person Observations: []", repaired)
        self.assertEqual(qp.parse_analysis(repaired, strict=True)["person_observations"], [])

    def test_redaction_requires_substantive_english_content(self):
        self.assertTrue(qp._has_usable_english_memory({
            "scene": "A woman opens a snow-covered wooden gate."
        }))
        self.assertFalse(qp._has_usable_english_memory({
            "scene": "unreadable text",
            "temporal_facts": [{"value": "unreadable text"}],
        }))


    def test_untyped_nonperson_actor_on_object_fact_is_not_person_reference(self):
        fields = {
            "people": [], "person_observations": [],
            "temporal_facts": [{
                "subject_type": "object", "subject": "display table",
                "actor": "store", "predicate": "contained", "value": "beverage cans",
            }],
        }
        self.assertEqual(qp._person_reference_labels(fields), [])
        self.assertEqual(qp._person_observation_coverage_error(fields), "")

    def test_neutral_person_actor_on_object_fact_remains_required(self):
        fields = {
            "people": [], "person_observations": [],
            "temporal_facts": [{
                "subject_type": "object", "subject": "phone",
                "actor": "unidentified person 2", "predicate": "handled",
            }],
        }
        self.assertEqual(qp._person_reference_labels(fields), ["unidentified person 2"])
        self.assertIn("unidentified person 2", qp._person_observation_coverage_error(fields))

    def test_camera_wearer_is_not_required_to_have_visible_appearance(self):
        fields = {
            "people": [], "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "camera wearer",
                "actor": "camera-wearer", "predicate": "held", "value": "phone",
            }],
        }
        self.assertEqual(qp._person_reference_labels(fields), [])
        self.assertEqual(qp._person_observation_coverage_error(fields), "")


    def test_orphan_person_references_are_dropped_but_grounded_memory_remains(self):
        fields = {
            "scene": "A person in a blue coat stands beside a phone on a table.",
            "people": ["unidentified person 1", "unidentified person 2", "camera wearer"],
            "person_observations": [self.make_person("unidentified person 1", 1, "blue coat")],
            "temporal_facts": [
                {"subject_type": "person", "subject": "unidentified person 1",
                 "actor": "unidentified person 1", "predicate": "appeared",
                 "value": "standing by the table", "affected_people": [],
                 "frame_index": 1, "evidence": "blue coat visible"},
                {"subject_type": "person", "subject": "unidentified person 2",
                 "actor": "unidentified person 2", "predicate": "appeared",
                 "value": "sitting by the table", "affected_people": [],
                 "frame_index": 2, "evidence": "person visible"},
                {"subject_type": "object", "subject": "phone", "predicate": "rested",
                 "value": "on the table", "affected_people": [],
                 "frame_index": 1, "evidence": "phone visible on table"},
            ],
        }
        sanitized, removed = qp._drop_unverified_person_references(fields)
        self.assertEqual(sanitized["people"], ["unidentified person 1", "camera wearer"])
        self.assertEqual(
            [row["subject"] for row in sanitized["temporal_facts"]],
            ["unidentified person 1", "phone"],
        )
        self.assertEqual(qp._structured_fact_quality_error(sanitized), "")
        self.assertTrue(any(row.get("entity") == "unidentified person 2"
                            for row in removed))
        self.assertEqual(fields["temporal_facts"][1]["subject"], "unidentified person 2")

    def test_blank_person_observation_is_removed_without_losing_other_people(self):
        fields = {
            "scene": "Two people stand in a room.",
            "people": ["unidentified person 1", "unidentified person 2"],
            "person_observations": [
                self.make_person("unidentified person 1", 1, "red jacket"),
                self.make_person("unidentified person 2", 2, ""),
            ],
            "temporal_facts": [
                {"subject_type": "person", "subject": "unidentified person 1",
                 "actor": "unidentified person 1", "affected_people": [],
                 "predicate": "stood", "value": "near a doorway",
                 "frame_index": 1, "evidence": "red jacket near doorway"},
                {"subject_type": "person", "subject": "unidentified person 2",
                 "actor": "unidentified person 2", "affected_people": [],
                 "predicate": "stood", "value": "by a wall",
                 "frame_index": 2, "evidence": "person visible by wall"},
            ],
        }
        sanitized, removed = qp._drop_unverified_person_references(fields)
        self.assertEqual(
            [row["entity"] for row in sanitized["person_observations"]],
            ["unidentified person 1"],
        )
        self.assertEqual(
            [row["subject"] for row in sanitized["temporal_facts"]],
            ["unidentified person 1"],
        )
        self.assertEqual(qp._structured_fact_quality_error(sanitized), "")
        self.assertTrue(any(row.get("kind") == "person_observation" for row in removed))


    def test_salvage_gate_accepts_only_after_remaining_fields_pass_quality(self):
        fields = {
            "scene": "A person in a blue coat stands beside a phone on a table.",
            "people": ["unidentified person 1", "unidentified person 2"],
            "person_observations": [self.make_person("unidentified person 1", 1, "blue coat")],
            "temporal_facts": [
                {"subject_type": "person", "subject": "unidentified person 1",
                 "actor": "unidentified person 1", "predicate": "appeared",
                 "value": "standing by table", "affected_people": [],
                 "frame_index": 1, "evidence": "blue coat visible"},
                {"subject_type": "person", "subject": "unidentified person 2",
                 "actor": "unidentified person 2", "predicate": "appeared",
                 "value": "sitting by table", "affected_people": [],
                 "frame_index": 2, "evidence": "person visible"},
                {"subject_type": "object", "subject": "phone", "predicate": "rested",
                 "value": "on the table", "affected_people": [],
                 "frame_index": 1, "evidence": "phone visible on table"},
            ],
        }
        salvaged, removed = qp._salvage_person_reference_gaps(fields)
        self.assertIsNotNone(salvaged)
        self.assertEqual(qp._structured_fact_quality_error(salvaged), "")
        self.assertEqual(
            [row["subject"] for row in salvaged["temporal_facts"]],
            ["unidentified person 1", "phone"],
        )
        self.assertTrue(removed)

    def test_salvage_gate_does_not_bypass_unverified_object_move(self):
        fields = {
            "scene": "A person stands near a chair.",
            "people": ["unidentified person 1"],
            "person_observations": [],
            "temporal_facts": [
                {"subject_type": "person", "subject": "unidentified person 1",
                 "actor": "unidentified person 1", "affected_people": [],
                 "predicate": "stood", "value": "near chair"},
                {"subject_type": "object", "subject": "chair", "predicate": "moved",
                 "value": "chair moved", "frame_index": 1,
                 "evidence": "chair visible"},
            ],
        }
        salvaged, removed = qp._salvage_person_reference_gaps(fields)
        self.assertIsNone(salvaged)
        self.assertEqual(removed, [])

    def test_salvage_gate_rejects_when_no_substantive_scene_remains(self):
        fields = {
            "scene": "",
            "people": ["unidentified person 1"],
            "person_observations": [],
            "temporal_facts": [{
                "subject_type": "person", "subject": "unidentified person 1",
                "actor": "unidentified person 1", "affected_people": [],
                "predicate": "appeared", "value": "standing",
            }],
        }
        salvaged, _removed = qp._salvage_person_reference_gaps(fields)
        self.assertIsNone(salvaged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
