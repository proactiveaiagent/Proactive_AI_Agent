"""Regression for source-video cache invalidation and clip-bound fingerprints."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import qwen_perceive as qp


ANALYSIS = """Scene: A person stands in a room.
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


class SourceFingerprintTests(unittest.TestCase):
    def test_fingerprint_covers_content_and_clip_interval_even_if_mtime_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            database = root / "fingerprints.sqlite3"
            with patch.object(qp, "SOURCE_FINGERPRINT_DB", database):
                video.write_bytes(b"original")
                original_stat = video.stat()
                first = qp.source_video_input_fingerprint(video, 0.0, 3.0)
                same = qp.source_video_input_fingerprint(video, 0.0, 3.0)
                different_clip = qp.source_video_input_fingerprint(video, 1.0, 4.0)
                self.assertEqual(first, same)
                self.assertEqual(first["video_sha256"], different_clip["video_sha256"])
                self.assertNotEqual(first["clip_sha256"], different_clip["clip_sha256"])

                time.sleep(0.01)
                video.write_bytes(b"modified")  # same byte length as the original
                os.utime(video, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                changed = qp.source_video_input_fingerprint(video, 0.0, 3.0)
                self.assertEqual(video.stat().st_mtime_ns, original_stat.st_mtime_ns)

        self.assertNotEqual(first["video_sha256"], changed["video_sha256"])
        self.assertNotEqual(first["clip_sha256"], changed["clip_sha256"])

    def test_missing_source_video_cannot_reuse_an_unverifiable_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(
                api_url="http://qwen.invalid", cache_file=Path(directory) / "cache.json"
            )
            perceiver.cache["clip-1"] = {
                "prompt_version": qp.PROMPT_VERSION,
                "source_input_fingerprint": {"video_sha256": "a" * 64,
                                              "clip_sha256": "b" * 64},
                "fields": {"scene": "cached scene", "temporal_facts": [],
                           "person_observations": []},
                "raw": "",
            }
            with patch.object(qp, "video_path_of", return_value=None), \
                 patch.object(qp, "clip_range", return_value=(0.0, 5.0)):
                result = perceiver.perceive(
                    {"id": "clip-1", "start_time": 0.0, "end_time": 5.0},
                    persist_cache=False,
                )

        self.assertEqual(result.get("source"), "error")
        self.assertIn("视频文件不存在", result.get("error", ""))

    def test_changed_source_video_invalidates_current_version_perception_cache(self):
        class Response:
            status_code = 200
            ok = True

            @staticmethod
            def json():
                return {"analysis": ANALYSIS}

            @staticmethod
            def raise_for_status():
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            database = root / "fingerprints.sqlite3"
            cache_file = root / "cache.json"
            video.write_bytes(b"original")
            original_stat = video.stat()
            perceiver = qp.QwenPerceiver(api_url="http://qwen.invalid", cache_file=cache_file)
            perceiver.api_urls = ["http://qwen.invalid"]
            perceiver.api_url = "http://qwen.invalid"
            with patch.object(qp, "SOURCE_FINGERPRINT_DB", database):
                cached_source = qp.source_video_input_fingerprint(video, 0.0, 5.0)
                perceiver.cache["clip-1"] = {
                    "prompt_version": qp.PROMPT_VERSION,
                    "source_input_fingerprint": cached_source,
                    "fields": {"scene": "stale scene", "temporal_facts": [],
                               "person_observations": []},
                    "raw": "",
                }
                time.sleep(0.01)
                video.write_bytes(b"modified")
                os.utime(video, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                with patch.object(qp, "video_path_of", return_value=video),                      patch.object(qp, "clip_range", return_value=(0.0, 5.0)),                      patch.object(qp, "extract_frames", return_value=([b"frame-bytes"], [0.5])),                      patch.object(perceiver, "health", return_value={"model": "Qwen3-VL"}),                      patch.object(qp.requests, "post", return_value=Response()) as post:
                    result = perceiver.perceive(
                        {"id": "clip-1", "start_time": 0.0, "end_time": 5.0},
                        persist_cache=False,
                    )

        self.assertEqual(result.get("source"), "qwen")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["fields"]["scene"], "A person stands in a room.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
