"""Focused recovery for truncated structured temporal-fact output."""
import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import qwen_perceive as qp


class TemporalFactRepairTests(unittest.TestCase):
    def test_parse_fenced_json_array(self):
        fence = chr(96) * 3
        self.assertEqual(
            qp._parse_json_array(fence + "json\n[{\"x\": 1}]\n" + fence),
            [{"x": 1}],
        )

    def test_rejects_text_after_array(self):
        with self.assertRaisesRegex(ValueError, "text after"):
            qp._parse_json_array("[] trailing")

    def test_filters_unverified_moves_and_invalid_frame_refs(self):
        rows = [
            {"subject_type": "object", "subject": "phone", "predicate": "used",
             "value": "A hand taps the phone.", "frame_index": 1,
             "evidence": "A hand touches the phone in frame 1."},
            {"subject_type": "object", "subject": "bag", "predicate": "moved",
             "value": "The bag moved.", "frame_index": 1,
             "evidence": "The bag is visible."},
            {"subject_type": "person", "subject": "person 8", "predicate": "appeared",
             "value": "A person is visible.", "frame_index": 8,
             "evidence": "A person is visible in frame 8."},
        ]
        result = qp._apply_temporal_fact_repair(rows, 2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["subject"], "phone")

    def test_focused_repair_prompt_limits_rows_and_requires_transition_evidence(self):
        prompt = qp._build_temporal_fact_repair_prompt(6)
        self.assertIn("up to 8 distinct", prompt)
        self.assertIn("from_location", prompt)
        self.assertIn("frame_index from 1 to 6", prompt)

    def test_temporal_repair_rejects_more_than_eight_rows(self):
        rows = [{
            "subject_type": "event", "subject": f"event {i}", "predicate": "occurred",
            "value": "A visible event.", "frame_index": 1, "evidence": "Visible in frame 1."
        } for i in range(9)]
        with self.assertRaisesRegex(ValueError, "more than 8"):
            qp._apply_temporal_fact_repair(rows, 6)

    def test_perceive_drops_unsupported_move_and_keeps_supported_facts(self):
        fields = (
            "Scene: A person taps a phone on a table beside a shopping cart.\n"
            "Location: Indoor room\nTime/Occasion:\nPeople:\n"
            "Person Observations: []\nUser Action:\nActivity: Handling a phone\n"
            "Objects: phone, shopping cart\nEnvironment: Indoor room\n"
            "Interaction Details: A person taps the phone.\n"
            "Temporal Facts: " + json.dumps([
                {"subject_type": "object", "subject": "phone", "predicate": "used",
                 "value": "A hand taps the phone on the table.", "actor": "camera wearer",
                 "affected_people": [], "frame_index": 1,
                 "evidence": "A hand touches the phone in frame 1."},
                {"subject_type": "object", "subject": "shopping cart", "predicate": "moved",
                 "value": "The shopping cart moved.", "actor": "camera wearer",
                 "affected_people": [], "frame_index": 2,
                 "evidence": "The shopping cart is visible in frame 2."},
            ])
        )

        class Response:
            status_code = 200
            ok = True

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / "cache.json")
            perceiver.api_urls = ["http://qwen"]
            perceiver.api_url = "http://qwen"
            post_calls = []

            def post(url, **kwargs):
                post_calls.append(kwargs)
                return Response({"analysis": fields})

            health = Response({"model_loaded": True, "model": "test-model"})
            with patch.object(qp, "video_path_of", return_value="/video/source.mp4"):
                with patch.object(qp, "clip_range", return_value=(0.0, 3.0)):
                    with patch.object(qp, "extract_frames", return_value=([b"frame"], [0.0])):
                        with patch.object(qp.requests, "get", return_value=health):
                            with patch.object(qp.requests, "post", side_effect=post):
                                result = perceiver.perceive(
                                    {"id": "clip-drop-unsupported-move",
                                     "start_time": 0.0, "end_time": 3.0},
                                    persist_cache=False,
                                )

        self.assertEqual(result["source"], "qwen")
        self.assertEqual(len(post_calls), 1)
        facts = result["fields"]["temporal_facts"]
        self.assertEqual([fact["subject"] for fact in facts], ["phone"])
        self.assertIn("shopping cart", result["raw"])
        self.assertEqual(result["quality_repair"]["removed_count"], 1)
        self.assertEqual(
            result["quality_repair"]["removed_facts"][0]["subject"], "shopping cart"
        )

    def test_person_array_repair_still_filters_unverified_object_moves(self):
        malformed = (
            "Scene: A person stands beside a chair in a room.\n"
            "Location: room\nTime/Occasion:\nPeople:\n"
            "Person Observations: [{incomplete\n"
            "User Action:\nActivity: Standing\nObjects: chair\n"
            "Environment: room\nInteraction Details:\n"
            "Temporal Facts: " + json.dumps([{
                "subject_type": "object", "subject": "chair", "predicate": "moved",
                "value": "chair moved from corner to table",
                "from_location": "corner", "to_location": "table",
                "frame_index": 0, "evidence": "chair visible",
            }])
        )

        class Response:
            status_code = 200
            ok = True

            def __init__(self, analysis):
                self.analysis = analysis

            def json(self):
                return {"analysis": self.analysis}

        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / "cache.json")
            perceiver.api_urls = ["http://qwen"]
            perceiver.api_url = "http://qwen"
            post_calls = []

            def post(url, **kwargs):
                post_calls.append(kwargs)
                if len(post_calls) <= 4:
                    return Response(malformed)
                return Response('{"person_observations":[]}')

            health = Response("")
            with patch.object(qp, "video_path_of", return_value="/video/source.mp4"):
                with patch.object(qp, "clip_range", return_value=(0.0, 3.0)):
                    with patch.object(qp, "extract_frames", return_value=([b"frame"], [0.0])):
                        with patch.object(qp.requests, "get", return_value=health):
                            with patch.object(qp.requests, "post", side_effect=post):
                                result = perceiver.perceive(
                                    {"id": "clip-person-array-move",
                                     "start_time": 0.0, "end_time": 3.0},
                                    persist_cache=False,
                                )

        self.assertEqual(result["source"], "qwen")
        self.assertEqual(len(post_calls), 5)
        self.assertEqual(result["fields"]["temporal_facts"], [])
        self.assertIn("chair", result["raw"])
        self.assertEqual(
            result["quality_repair"]["removed_facts"][0]["subject"], "chair"
        )

    def test_perceive_salvages_complete_facts_before_truncated_tail(self):
        valid = (
            "{\"subject_type\":\"object\",\"subject\":\"phone\","
            "\"predicate\":\"used\",\"value\":\"A hand taps the phone.\","
            "\"actor\":\"camera wearer\",\"frame_index\":1,"
            "\"evidence\":\"A hand touches the phone in frame 1.\"}"
        )
        malformed = (
            "Scene: A person taps a phone on the table.\n"
            "Location: Indoor room\nTime/Occasion:\nPeople:\nPerson Observations: []\n"
            "User Action:\nActivity: Using a phone\nObjects: phone\nEnvironment: room\n"
            "Interaction Details: A person taps the phone.\n"
            "Temporal Facts: [" + valid + ", {\"subject_type\":\"object\",\"subject\":"
        )

        class Response:
            status_code = 200
            ok = True

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / "cache.json")
            perceiver.api_urls = ["http://qwen"]
            perceiver.api_url = "http://qwen"
            post_calls = []

            def post(url, **kwargs):
                post_calls.append(kwargs)
                return Response({"analysis": malformed})

            health = Response({"model_loaded": True, "model": "test-model"})
            with patch.object(qp, "video_path_of", return_value="/video/source.mp4"):
                with patch.object(qp, "clip_range", return_value=(0.0, 3.0)):
                    with patch.object(qp, "extract_frames", return_value=([b"frame"], [0.0])):
                        with patch.object(qp.requests, "get", return_value=health):
                            with patch.object(qp.requests, "post", side_effect=post):
                                result = perceiver.perceive(
                                    {"id": "clip-prefix", "start_time": 0.0, "end_time": 3.0},
                                    persist_cache=False,
                                )

        self.assertEqual(result["source"], "qwen")
        self.assertEqual(len(post_calls), 1)
        self.assertEqual(result["fields"]["temporal_facts"][0]["predicate"], "used")
        self.assertEqual(
            result["temporal_fact_repair"]["source"],
            "same_video_frames_truncated_prefix",
        )
        self.assertEqual(result["temporal_fact_repair"]["facts_added"], 1)

    def test_prefix_recovery_salvages_verified_people_and_preserves_original_output(self):
        observation = {
            "entity": "unidentified person 1",
            "appearance": "wearing a dark coat and grey headscarf",
            "frame_index": 1,
            "evidence": "The coat and headscarf are visible in frame 1.",
        }
        facts = [
            {"subject_type": "person", "subject": "unidentified person 1",
             "predicate": "smiled", "value": "Smiled broadly",
             "actor": "unidentified person 1", "affected_people": [],
             "frame_index": 1, "evidence": "The person smiles in frame 1."},
            {"subject_type": "person", "subject": "unidentified person 2",
             "predicate": "approached", "value": "Approached person 1",
             "actor": "unidentified person 2",
             "affected_people": ["unidentified person 1"],
             "frame_index": 1, "evidence": "A second person approaches in frame 1."},
        ]
        raw = (
            "Scene: Two people interact at a snowy gate.\n"
            "Location: Rural courtyard\nTime/Occasion:\n"
            "People: unidentified person 1, unidentified person 2\n"
            "Person Observations: " + json.dumps([observation]) +
            "\nUser Action: A younger person approaches.\n"
            "Activity: Greeting at a gate\nObjects: wooden gate\n"
            "Environment: Snow covers the courtyard.\n"
            "Interaction Details: One person approaches another.\n"
            "Temporal Facts: " + json.dumps(facts)[:-1] +
            ', {"subject_type":"person","subject":'
        )
        recovered = qp._recover_truncated_temporal_fact_prefix(raw, 2)
        self.assertIsNotNone(recovered)
        returned_raw, fields, accepted_rows, repair = recovered
        self.assertEqual(returned_raw, raw)
        self.assertEqual(fields["people"], ["unidentified person 1"])
        self.assertEqual(len(accepted_rows), 1)
        self.assertEqual(accepted_rows[0]["subject"], "unidentified person 1")
        person_repair = repair["person_reference_repair"]
        self.assertEqual(person_repair["method"], "drop_unverified_person_references")
        self.assertTrue(person_repair["raw_model_output_preserved"])
        self.assertEqual(person_repair["removed_count"], 2)

    def test_perceive_keeps_verified_prefix_and_audits_orphan_person(self):
        observation = {
            "entity": "unidentified person 1",
            "appearance": "wearing a dark coat and grey headscarf",
            "frame_index": 1,
            "evidence": "The coat and headscarf are visible in frame 1.",
        }
        facts = [
            {"subject_type": "person", "subject": "unidentified person 1",
             "predicate": "smiled", "value": "Smiled broadly",
             "actor": "unidentified person 1", "affected_people": [],
             "frame_index": 1, "evidence": "The person smiles in frame 1."},
            {"subject_type": "person", "subject": "unidentified person 2",
             "predicate": "approached", "value": "Approached person 1",
             "actor": "unidentified person 2",
             "affected_people": ["unidentified person 1"],
             "frame_index": 1, "evidence": "A second person approaches in frame 1."},
        ]
        raw = (
            "Scene: Two people interact at a snowy gate.\n"
            "Location: Rural courtyard\nTime/Occasion:\n"
            "People: unidentified person 1, unidentified person 2\n"
            "Person Observations: " + json.dumps([observation]) +
            "\nUser Action: A younger person approaches.\n"
            "Activity: Greeting at a gate\nObjects: wooden gate\n"
            "Environment: Snow covers the courtyard.\n"
            "Interaction Details: One person approaches another.\n"
            "Temporal Facts: " + json.dumps(facts)[:-1] +
            ', {"subject_type":"person","subject":'
        )

        class Response:
            status_code = 200
            ok = True

            def json(self):
                return {"analysis": raw}

        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / "cache.json")
            perceiver.api_urls = ["http://qwen"]
            perceiver.api_url = "http://qwen"
            post_calls = []
            def post(url, **kwargs):
                post_calls.append(kwargs)
                return Response()

            health = Response()
            with patch.object(qp, "video_path_of", return_value="/video/source.mp4"):
                with patch.object(qp, "clip_range", return_value=(0.0, 2.0)):
                    with patch.object(qp, "extract_frames", return_value=([b"frame"], [0.0])):
                        with patch.object(qp.requests, "get", return_value=health):
                            with patch.object(qp.requests, "post", side_effect=post):
                                result = perceiver.perceive(
                                    {"id": "clip-orphan-prefix",
                                     "start_time": 0.0, "end_time": 2.0},
                                    persist_cache=False,
                                )

        self.assertEqual(result["source"], "qwen")
        self.assertEqual(len(post_calls), 1)
        self.assertEqual(result["raw"], raw)
        self.assertEqual(result["fields"]["people"], ["unidentified person 1"])
        self.assertEqual(len(result["fields"]["temporal_facts"]), 1)
        self.assertEqual(
            result["quality_repair"]["person_reference_repair"]["removed_count"], 2
        )
        self.assertTrue(
            result["quality_repair"]["person_reference_repair"]["raw_model_output_preserved"]
        )

    def test_perceive_keeps_valid_scene_when_temporal_array_has_no_complete_row(self):
        malformed = (
            "Scene: An elderly woman stands smiling in a snowy courtyard gate.\n"
            "Location: Rural courtyard\nTime/Occasion:\nPeople: []\n"
            "Person Observations: " + json.dumps([{
                "entity": "unidentified person 1",
                "appearance": "wearing a blue padded jacket, dark pants, and a grey headscarf",
                "frame_index": 1,
                "evidence": "The jacket, pants, and headscarf are visible in frame 1.",
            }]) + "\nUser Action: A younger woman approaches the gate.\n"
            "Activity: Greeting at the gate\nObjects: wooden gate\n"
            "Environment: Snow covers the courtyard.\n"
            "Interaction Details: A younger woman approaches the elderly woman.\n"
            "Temporal Facts: [{\"subject_type\":\"person\",\"subject\":"
        )

        class Response:
            status_code = 200
            ok = True

            def __init__(self, analysis):
                self.analysis = analysis

            def json(self):
                return {"analysis": self.analysis}

        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / "cache.json")
            perceiver.api_urls = ["http://qwen"]
            perceiver.api_url = "http://qwen"
            post_calls = []

            def post(url, **kwargs):
                post_calls.append(kwargs)
                if len(post_calls) <= 4:
                    return Response(malformed)
                return Response("[{\"subject_type\":\"person\",\"subject\":")

            health = Response("")
            with patch.object(qp, "video_path_of", return_value="/video/source.mp4"):
                with patch.object(qp, "clip_range", return_value=(0.0, 3.0)):
                    with patch.object(qp, "extract_frames", return_value=([b"frame"], [0.0])):
                        with patch.object(qp.requests, "get", return_value=health):
                            with patch.object(qp.requests, "post", side_effect=post):
                                result = perceiver.perceive(
                                    {"id": "clip-empty-truncated-prefix",
                                     "start_time": 0.0, "end_time": 3.0},
                                    persist_cache=False,
                                )

        self.assertEqual(result["source"], "qwen")
        self.assertEqual(len(post_calls), 5)
        self.assertEqual(result["fields"]["location"], "Rural courtyard")
        self.assertEqual(result["fields"]["activity"], "Greeting at the gate")
        self.assertEqual(result["fields"]["temporal_facts"], [])
        self.assertEqual(
            result["temporal_fact_repair"]["source"],
            "same_video_frames_incomplete_array_empty_fallback",
        )
        self.assertTrue(result["temporal_fact_repair"]["raw_model_output_invalid"])
        self.assertIn("Temporal Facts: []", result["raw"])

    def test_perceive_keeps_scene_when_closed_temporal_array_is_malformed(self):
        malformed = (
            "Scene: An elderly woman stands smiling in a snowy courtyard gate.\n"
            "Location: Rural courtyard\nTime/Occasion:\nPeople: []\n"
            "Person Observations: " + json.dumps([{
                "entity": "unidentified person 1",
                "appearance": "wearing a blue padded jacket and grey headscarf",
                "frame_index": 1,
                "evidence": "The jacket and headscarf are visible in frame 1.",
            }]) + "\nUser Action: A younger woman approaches the gate.\n"
            "Activity: Greeting at the gate\nObjects: wooden gate\n"
            "Environment: Snow covers the courtyard.\n"
            "Interaction Details: A younger woman approaches the elderly woman.\n"
            "Temporal Facts: [{invalid}]"
        )

        class Response:
            status_code = 200
            ok = True

            def __init__(self, analysis):
                self.analysis = analysis

            def json(self):
                return {"analysis": self.analysis}

        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / "cache.json")
            perceiver.api_urls = ["http://qwen"]
            perceiver.api_url = "http://qwen"
            post_calls = []

            def post(url, **kwargs):
                post_calls.append(kwargs)
                return Response(malformed)

            health = Response("")
            with patch.object(qp, "video_path_of", return_value="/video/source.mp4"):
                with patch.object(qp, "clip_range", return_value=(0.0, 3.0)):
                    with patch.object(qp, "source_video_input_fingerprint",
                                      return_value={"video_sha256": "video", "clip_sha256": "clip"}):
                        with patch.object(qp, "extract_frames", return_value=([b"frame"], [0.0])):
                            with patch.object(qp.requests, "get", return_value=health):
                                with patch.object(qp.requests, "post", side_effect=post):
                                    moment = {"id": "clip-closed-malformed-array",
                                              "start_time": 0.0, "end_time": 3.0}
                                    result = perceiver.perceive(moment, persist_cache=True)
                                    cached_result = perceiver.perceive(moment)

        self.assertEqual(result["source"], "qwen")
        self.assertEqual(len(post_calls), 5)
        self.assertEqual(result["fields"]["location"], "Rural courtyard")
        self.assertEqual(result["fields"]["objects"], ["wooden gate"])
        self.assertEqual(result["fields"]["temporal_facts"], [])
        self.assertEqual(
            result["temporal_fact_repair"]["source"],
            "same_video_frames_incomplete_array_empty_fallback",
        )
        self.assertTrue(result["temporal_fact_repair"]["raw_model_output_invalid"])
        self.assertIn("Temporal Facts: []", result["raw"])
        self.assertIn("Temporal Facts: [{invalid}]", result["raw_model_output"])
        self.assertEqual(cached_result["source"], "cache")
        self.assertEqual(cached_result["raw"], malformed)
        self.assertEqual(cached_result["raw_model_output"], malformed)

    def test_perceive_recovers_truncated_temporal_array_from_same_frames(self):
        malformed = (
            "Scene: A person taps a phone on the table.\n"
            "Location: Indoor room\nTime/Occasion:\nPeople:\nPerson Observations: []\n"
            "User Action:\nActivity: Using a phone\nObjects: phone\nEnvironment: room\n"
            "Interaction Details: A person taps the phone.\n"
            "Temporal Facts: [{\"subject_type\":\"object\",\"subject\":\"phone\","
        )
        repair = (
            "[{\"subject_type\":\"object\",\"subject\":\"phone\",\"predicate\":\"used\","
            "\"value\":\"A hand taps the phone on the table.\",\"actor\":\"camera wearer\","
            "\"affected_people\":[],\"frame_index\":1,\"time_order\":1,"
            "\"evidence\":\"A hand touches the phone in frame 1.\",\"confidence\":\"high\"}]"
        )

        class Response:
            status_code = 200
            ok = True

            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / "cache.json")
            perceiver.api_urls = ["http://qwen"]
            perceiver.api_url = "http://qwen"
            post_calls = []

            def post(url, **kwargs):
                post_calls.append(kwargs)
                prompt = kwargs.get("data", {}).get("prompt", "")
                is_repair = "Return exactly one complete JSON array" in prompt
                return Response({"analysis": repair if is_repair else malformed})

            health = Response({"model_loaded": True, "model": "test-model"})
            with patch.object(qp, "video_path_of", return_value="/video/source.mp4") as video_mock:
                with patch.object(qp, "clip_range", return_value=(0.0, 3.0)):
                    with patch.object(qp, "extract_frames", return_value=([b"frame"], [0.0])):
                        with patch.object(qp.requests, "get", return_value=health):
                            with patch.object(qp.requests, "post", side_effect=post):
                                result = perceiver.perceive(
                                    {"id": "clip-1", "start_time": 0.0, "end_time": 3.0},
                                    persist_cache=False,
                                )

        self.assertEqual(result["source"], "qwen")
        self.assertEqual(len(post_calls), 5)
        self.assertEqual(result["fields"]["temporal_facts"][0]["predicate"], "used")
        self.assertEqual(result["temporal_fact_repair"]["facts_added"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
