"""Regression: benchmark text must never enter Qwen video-perception requests."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import qwen_perceive as qp


class VideoOnlyPerceptionInputTests(unittest.TestCase):
    def test_perception_request_contains_frames_but_no_manifest_or_transcript_text(self):
        sentinels = (
            "MANIFEST_SCENE_SENTINEL",
            "REFERENCE_ANSWER_SENTINEL",
            "TRANSCRIPT_SENTINEL",
            "PRIVATE_CATEGORY_SENTINEL",
            "FEISHU_CLIP_ID_SENTINEL",
        )
        moment = {
            "id": sentinels[-1],
            "video": "test_data/example.mp4",
            "start_time": 0.0,
            "end_time": 5.0,
            "scene": sentinels[0],
            "reference_answer": sentinels[1],
            "transcript": sentinels[2],
            "test_category": sentinels[3],
        }
        analysis = """Scene: A person stands in a room.
Location: room
Time/Occasion:
People:
Person Observations: []
User Action:
Activity: standing
Objects: chair
Environment: room
Interaction Details: A person stands in a room.
Temporal Facts: []"""
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"analysis": analysis}

            @staticmethod
            def raise_for_status():
                return None

        with tempfile.TemporaryDirectory(prefix="video-only-perception-") as tmp:
            perceiver = qp.QwenPerceiver(api_url="http://model.invalid",
                                         cache_file=Path(tmp) / "cache.json")
            with patch.object(qp, "video_path_of", return_value=Path(tmp) / "clip.mp4"), \
                 patch.object(qp, "extract_frames",
                              return_value=([b"frame-bytes"], [0.5])), \
                 patch.object(qp.requests, "post", return_value=Response()) as post, \
                 patch.object(perceiver, "health", return_value={"model": "Qwen3-VL"}):
                result = perceiver.perceive(moment, persist_cache=False)

        self.assertEqual(result.get("source"), "qwen")
        self.assertTrue(result.get("fields"))
        self.assertEqual(post.call_count, 1)
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["files"], [
            ("images", ("f0.jpg", b"frame-bytes", "image/jpeg"))
        ])
        self.assertEqual(set(kwargs["data"]), {"prompt", "request_id"})
        request_text = repr((kwargs["data"], kwargs["files"]))
        for sentinel in sentinels:
            self.assertNotIn(sentinel, request_text)


    def test_schema_retry_uses_only_video_frames_and_generated_prompt(self):
        sentinels = (
            "MANIFEST_SCENE_SENTINEL",
            "REFERENCE_ANSWER_SENTINEL",
            "TRANSCRIPT_SENTINEL",
            "PRIVATE_CATEGORY_SENTINEL",
            "FEISHU_CLIP_ID_SENTINEL",
        )
        moment = {
            "id": sentinels[-1],
            "video": "test_data/example.mp4",
            "start_time": 0.0,
            "end_time": 5.0,
            "scene": sentinels[0],
            "reference_answer": sentinels[1],
            "transcript": sentinels[2],
            "test_category": sentinels[3],
        }
        malformed = "Scene: A person stands in a room."
        valid = """Scene: A person stands in a room.
Location: room
Time/Occasion:
People:
Person Observations: []
User Action:
Activity: standing
Objects: chair
Environment: room
Interaction Details: A person stands in a room.
Temporal Facts: []"""

        class Response:
            status_code = 200

            def __init__(self, analysis):
                self.analysis = analysis

            def json(self):
                return {"analysis": self.analysis}

            @staticmethod
            def raise_for_status():
                return None

        with tempfile.TemporaryDirectory(prefix="video-only-retry-") as tmp:
            perceiver = qp.QwenPerceiver(
                api_url="http://model.invalid",
                cache_file=Path(tmp) / "cache.json",
            )
            with patch.object(qp, "video_path_of", return_value=Path(tmp) / "clip.mp4"), \
                 patch.object(qp, "extract_frames", return_value=([b"frame-bytes"], [0.5])), \
                 patch.object(qp.requests, "post",
                              side_effect=[Response(malformed), Response(valid)]) as post, \
                 patch.object(perceiver, "health", return_value={"model": "Qwen3-VL"}):
                result = perceiver.perceive(moment, persist_cache=False)

        self.assertEqual(result.get("source"), "qwen")
        self.assertEqual(post.call_count, 2)
        for call in post.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(set(kwargs["data"]), {"prompt", "request_id"})
            self.assertEqual(kwargs["files"], [
                ("images", ("f0.jpg", b"frame-bytes", "image/jpeg"))
            ])
            request_text = repr((kwargs["data"], kwargs["files"]))
            for sentinel in sentinels:
                self.assertNotIn(sentinel, request_text)


    def test_incomplete_person_array_repair_uses_only_video_frames(self):
        sentinels = (
            "MANIFEST_SCENE_SENTINEL",
            "REFERENCE_ANSWER_SENTINEL",
            "TRANSCRIPT_SENTINEL",
            "PRIVATE_CATEGORY_SENTINEL",
            "FEISHU_CLIP_ID_SENTINEL",
        )
        moment = {
            "id": sentinels[-1],
            "video": "test_data/example.mp4",
            "start_time": 0.0,
            "end_time": 5.0,
            "scene": sentinels[0],
            "reference_answer": sentinels[1],
            "transcript": sentinels[2],
            "test_category": sentinels[3],
        }
        malformed = """Scene: An empty room with a chair and no people visible.
Location: room
Time/Occasion:
People:
Person Observations: [{"entity":"person 1"
User Action:
Activity:
Objects: chair
Environment: empty room
Interaction Details:
Temporal Facts: []"""

        class Response:
            status_code = 200

            def __init__(self, analysis):
                self.analysis = analysis

            def json(self):
                return {"analysis": self.analysis}

            @staticmethod
            def raise_for_status():
                return None

        with tempfile.TemporaryDirectory(prefix="video-only-person-repair-") as tmp:
            perceiver = qp.QwenPerceiver(
                api_url="http://model.invalid",
                cache_file=Path(tmp) / "cache.json",
            )
            with patch.object(qp, "video_path_of", return_value=Path(tmp) / "clip.mp4"), \
                 patch.object(qp, "extract_frames", return_value=([b"frame-bytes"], [0.5])), \
                 patch.object(qp.requests, "post", side_effect=[
                     *[Response(malformed) for _ in range(4)],
                     Response('{"person_observations": [], "entity_map": []}'),
                 ]) as post, \
                 patch.object(perceiver, "health", return_value={"model": "Qwen3-VL"}):
                result = perceiver.perceive(moment, persist_cache=False)

        self.assertEqual(result.get("source"), "qwen")
        self.assertEqual(result.get("fields", {}).get("person_observations"), [])
        self.assertEqual(post.call_count, 5)
        self._assert_requests_are_video_only(post.call_args_list, sentinels)

    def test_truncated_temporal_fact_repair_uses_only_video_frames(self):
        sentinels = (
            "MANIFEST_SCENE_SENTINEL",
            "REFERENCE_ANSWER_SENTINEL",
            "TRANSCRIPT_SENTINEL",
            "PRIVATE_CATEGORY_SENTINEL",
            "FEISHU_CLIP_ID_SENTINEL",
        )
        moment = {
            "id": sentinels[-1],
            "video": "test_data/example.mp4",
            "start_time": 0.0,
            "end_time": 5.0,
            "scene": sentinels[0],
            "reference_answer": sentinels[1],
            "transcript": sentinels[2],
            "test_category": sentinels[3],
        }
        malformed = """Scene: An empty room with a chair and no people visible.
Location: room
Time/Occasion:
People:
Person Observations: []
User Action:
Activity:
Objects: chair
Environment: empty room
Interaction Details:
Temporal Facts: [not-json"""

        class Response:
            status_code = 200

            def __init__(self, analysis):
                self.analysis = analysis

            def json(self):
                return {"analysis": self.analysis}

            @staticmethod
            def raise_for_status():
                return None

        with tempfile.TemporaryDirectory(prefix="video-only-temporal-repair-") as tmp:
            perceiver = qp.QwenPerceiver(
                api_url="http://model.invalid",
                cache_file=Path(tmp) / "cache.json",
            )
            with patch.object(qp, "video_path_of", return_value=Path(tmp) / "clip.mp4"), \
                 patch.object(qp, "extract_frames", return_value=([b"frame-bytes"], [0.5])), \
                 patch.object(qp.requests, "post", side_effect=[
                     *[Response(malformed) for _ in range(4)],
                     Response("[]"),
                 ]) as post, \
                 patch.object(perceiver, "health", return_value={"model": "Qwen3-VL"}):
                result = perceiver.perceive(moment, persist_cache=False)

        self.assertEqual(result.get("source"), "qwen")
        self.assertEqual(result.get("fields", {}).get("temporal_facts"), [])
        self.assertEqual(post.call_count, 5)
        self._assert_requests_are_video_only(post.call_args_list, sentinels)

    def _assert_requests_are_video_only(self, calls, sentinels):
        expected_frames = [
            ("images", ("f0.jpg", b"frame-bytes", "image/jpeg"))
        ]
        for call in calls:
            kwargs = call.kwargs
            self.assertEqual(set(kwargs["data"]), {"prompt", "request_id"})
            self.assertEqual(kwargs["files"], expected_frames)
            request_text = repr((kwargs["data"], kwargs["files"]))
            for sentinel in sentinels:
                self.assertNotIn(sentinel, request_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
