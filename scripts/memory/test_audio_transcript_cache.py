#!/usr/bin/env python3
"""Regression tests for transcript cache refresh and lossless writes."""
import json
import tempfile
import unittest
from pathlib import Path
from audio_transcript_cache import PersistentTranscriptCache

class PersistentTranscriptCacheTests(unittest.TestCase):
    def test_existing_reader_refreshes_after_another_instance_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcripts.json"
            session_cache = PersistentTranscriptCache(path)
            self.assertIsNone(session_cache.get("clip-a"))
            prewarmer_cache = PersistentTranscriptCache(path)
            prewarmer_cache.put_if_absent("clip-a", {"text": "already transcribed"})
            self.assertEqual(session_cache.get("clip-a"), {"text": "already transcribed"})

    def test_separate_writers_merge_without_lost_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcripts.json"
            web_cache = PersistentTranscriptCache(path)
            prewarmer_cache = PersistentTranscriptCache(path)
            web_cache.put_if_absent("clip-web", {"text": "web"})
            prewarmer_cache.put_if_absent("clip-prewarm", {"text": "prewarm"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {
                "clip-web": {"text": "web"},
                "clip-prewarm": {"text": "prewarm"},
            })

if __name__ == "__main__":
    unittest.main()
