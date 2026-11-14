"""Regression checks for recency-based visual asset tiering."""
import io
import os
import random
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from multimodal_memory import MultimodalAssetStore


class MultimodalRecencyTierTest(unittest.TestCase):
    def source_image(self):
        randomizer = random.Random(7)
        image = Image.new("RGB", (640, 480))
        image.putdata([(randomizer.randrange(256), randomizer.randrange(256),
                        randomizer.randrange(256)) for _ in range(640 * 480)])
        output = io.BytesIO()
        image.save(output, "WEBP", quality=90)
        return output.getvalue()

    def make_fixture(self, shared_recent=False, unknown_time=False):
        directory = tempfile.TemporaryDirectory(prefix="multimodal-recency-")
        root = Path(directory.name)
        (root / "memory.json").write_text("{}", encoding="utf-8")
        store = MultimodalAssetStore(root)
        source = self.source_image()
        ref = store._put("visual", source, "webp", len(source) + 1)
        old_time = "" if unknown_time else (datetime.now() - timedelta(days=90)).isoformat()
        moments = {
            "old": {
                "id": "old",
                "timestamp": old_time,
                "multimodal": {"visual_frames": [
                    dict(ref, at_s=1.0, quality_tier="recent")
                ]},
            }
        }
        if shared_recent:
            moments["new"] = {
                "id": "new",
                "timestamp": datetime.now().isoformat(),
                "multimodal": {"visual_frames": [
                    dict(ref, at_s=1.0, quality_tier="recent")
                ]},
            }
        return directory, root, store, moments, ref

    @patch.dict(os.environ, {
        "MEMORY_ASSET_BUDGET_GB": "1",
        "MEMORY_TOTAL_BUDGET_BYTES": "10000000",
        "MEMORY_NON_ASSET_RESERVE_BYTES": "0",
    })
    def test_old_visual_is_downsampled_but_still_viewable(self):
        from PIL import Image
        directory, root, store, moments, before = self.make_fixture()
        with directory:
            result = store._demote_aged_visuals(
                moments, recent_days=30, protected_recent=0)
            ref = moments["old"]["multimodal"]["visual_frames"][0]
            archived = root / ref["asset"]
            self.assertEqual(result["demoted_assets"], 1)
            self.assertEqual(ref["quality_tier"], "archive")
            self.assertTrue(archived.is_file())
            self.assertLess(archived.stat().st_size, before["bytes"])
            with Image.open(archived) as image:
                image.verify()
                self.assertLessEqual(max(image.size), 224)
            self.assertFalse((root / before["asset"]).exists())

    @patch.dict(os.environ, {
        "MEMORY_ASSET_BUDGET_GB": "1",
        "MEMORY_TOTAL_BUDGET_BYTES": "10000000",
        "MEMORY_NON_ASSET_RESERVE_BYTES": "0",
    })
    def test_shared_recent_visual_is_never_downsampled(self):
        directory, root, store, moments, before = self.make_fixture(shared_recent=True)
        with directory:
            result = store._demote_aged_visuals(
                moments, recent_days=30, protected_recent=0)
            self.assertEqual(result["demoted_assets"], 0)
            self.assertEqual(
                moments["old"]["multimodal"]["visual_frames"][0]["asset"],
                before["asset"],
            )
            self.assertTrue((root / before["asset"]).is_file())

    @patch.dict(os.environ, {
        "MEMORY_ASSET_BUDGET_GB": "1",
        "MEMORY_TOTAL_BUDGET_BYTES": "10000000",
        "MEMORY_NON_ASSET_RESERVE_BYTES": "0",
    })
    def test_unknown_timestamp_is_preserved_at_original_quality(self):
        directory, root, store, moments, before = self.make_fixture(unknown_time=True)
        with directory:
            result = store._demote_aged_visuals(
                moments, recent_days=30, protected_recent=0)
            self.assertEqual(result["demoted_assets"], 0)
            self.assertEqual(
                moments["old"]["multimodal"]["visual_frames"][0]["asset"],
                before["asset"],
            )
            self.assertTrue((root / before["asset"]).is_file())


if __name__ == "__main__":
    from PIL import Image
    unittest.main(verbosity=2)
