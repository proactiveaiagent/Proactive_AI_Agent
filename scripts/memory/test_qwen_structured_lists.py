import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import qwen_perceive as qp


class StructuredListParsingTests(unittest.TestCase):
    def test_json_object_array_is_parsed_without_brackets_or_quotes(self):
        raw = """Scene: Snowy courtyard.
Location: courtyard.
Time/Occasion:
People: []
Person Observations: []
User Action:
Activity: greeting.
Objects: ["red lantern", "wooden gate", "stack of firewood"]
Environment: An outdoor courtyard with snow, brick walls, and a wooden gate.
Interaction Details: two people greet.
Temporal Facts: []"""
        fields = qp.parse_analysis(raw, strict=True)
        self.assertEqual(fields["objects"], ["red lantern", "wooden gate", "stack of firewood"])
        self.assertEqual(fields["environments"], [
            "An outdoor courtyard with snow, brick walls, and a wooden gate"
        ])

    def test_person_face_features_parse_as_a_separate_field(self):
        raw = """Scene: A woman stands near a snowy gate.
Location: courtyard.
Time/Occasion:
People: []
Person Observations: [{"entity":"unidentified person 1","appearance":"blue coat and grey headscarf","face_features":"round face and visible smile","frame_index":1,"evidence":"face visible in frame 1","confidence":"high"}]
User Action:
Activity: greeting.
Objects: []
Environment: Snowy courtyard.
Interaction Details: two people greet.
Temporal Facts: []"""
        fields = qp.parse_analysis(raw, strict=True)
        self.assertEqual(fields["person_observations"][0]["face_features"],
                         "round face and visible smile")

    def test_plain_object_list_keeps_legacy_comma_format(self):
        self.assertEqual(qp._split_list("phone, boxes, tablecloth"),
                         ["phone", "boxes", "tablecloth"])

    def test_environment_json_array_remains_multiple_descriptions(self):
        self.assertEqual(qp._split_list('["snow-covered ground", "brick walls"]',
                                        preserve_commas=True),
                         ["snow-covered ground", "brick walls"])

    def test_same_source_cache_reparses_lists_without_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"same source")
            with patch.object(qp, "SOURCE_FINGERPRINT_DB", root / "fingerprints.sqlite3"):
                fingerprint = qp.source_video_input_fingerprint(video, 0.0, 5.0)
                perceiver = qp.QwenPerceiver(cache_file=root / "cache.json")
                raw = """Scene: A woman stands near a snowy gate.
Location: rural courtyard.
Time/Occasion:
People: []
Person Observations: []
User Action:
Activity: greeting.
Objects: ["red lantern", "wooden gate"]
Environment: Snow covers the ground, walls, and gate.
Interaction Details: two women greet.
Temporal Facts: []"""
                perceiver.cache["clip"] = {
                    "prompt_version": qp.PROMPT_VERSION,
                    "list_parser_version": "old-parser",
                    "source_input_fingerprint": fingerprint,
                    "fields": {
                        "scene": "cached",
                        "objects": ['["red lantern"', '"wooden gate"]'],
                        "environments": ["Snow covers the ground", "walls", "and gate"],
                        "person_observations": [],
                        "temporal_facts": [],
                    },
                    "raw": raw,
                }
                with patch.object(qp, "video_path_of", return_value=video):
                    with patch.object(qp, "clip_range", return_value=(0.0, 5.0)):
                        with patch.object(qp.requests, "post", side_effect=AssertionError("must reparse cache only")):
                            result = perceiver.perceive({"id": "clip", "start_time": 0.0, "end_time": 5.0})
            self.assertEqual(result["source"], "parser_repair")
            self.assertEqual(result["fields"]["objects"], ["red lantern", "wooden gate"])
            self.assertEqual(result["fields"]["environments"], [
                "Snow covers the ground, walls, and gate"
            ])
            self.assertEqual(result["list_parser_version"], qp.LIST_PARSER_VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
