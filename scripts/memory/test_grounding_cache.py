"""Offline tests for persistent object-grounding cache keys and aliases."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from grounding_cache import PersistentObjectGroundingCache


class GroundingSourceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="grounding-source-cache-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.cache = PersistentObjectGroundingCache(root / "cache.sqlite3", root)
        self.video = root / "clip.mp4"
        self.video.write_bytes(b"source-clip-v1")

    def test_source_alias_round_trip(self):
        key = self.cache.source_key("clip-1", self.video, 0, 8.0, ["phone", "table"])
        payload = {"status": "completed", "items": [{"object": "phone"}]}
        self.assertTrue(key)
        self.assertIsNone(self.cache.get_source(key)[0])
        self.assertTrue(self.cache.put_source(key, payload))
        self.assertEqual(self.cache.get_source(key)[0], payload)

    def test_source_alias_invalidates_on_file_change(self):
        key_before = self.cache.source_key("clip-1", self.video, 0, 8.0, ["phone"])
        self.assertTrue(self.cache.put_source(key_before, {"status": "completed", "items": []}))
        self.video.write_bytes(b"source-clip-v2-changed")
        key_after = self.cache.source_key("clip-1", self.video, 0, 8.0, ["phone"])
        self.assertNotEqual(key_before, key_after)
        self.assertIsNone(self.cache.get_source(key_after)[0])

    def test_source_alias_invalidates_on_clip_or_label_change(self):
        base = self.cache.source_key("clip-1", self.video, 0, 8.0, ["phone"])
        other_clip = self.cache.source_key("clip-2", self.video, 0, 8.0, ["phone"])
        other_interval = self.cache.source_key("clip-1", self.video, 1, 8.0, ["phone"])
        other_label = self.cache.source_key("clip-1", self.video, 0, 8.0, ["phone", "table"])
        self.assertEqual(len({base, other_clip, other_interval, other_label}), 4)

    def test_only_completed_payloads_can_be_stored(self):
        key = self.cache.source_key("clip-1", self.video, 0, 8.0, ["phone"])
        self.assertFalse(self.cache.put_source(key, {"status": "failed"}))
        self.assertIsNone(self.cache.get_source(key)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
