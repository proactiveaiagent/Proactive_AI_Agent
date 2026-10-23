#!/usr/bin/env python3
"""Regression coverage for linked media included by storage and retrieval metrics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_all_people_metrics import asset_refs


class AssetReferenceTests(unittest.TestCase):
    def test_includes_object_crops_and_source_frames(self) -> None:
        record = {
            "multimodal": {
                "visual_frames": [{"asset": "assets/visual/frame.webp"}],
                "audio_clips": [{"asset": "assets/audio/clip.ogg"}],
                "object_visuals": {
                    "items": [
                        {
                            "frame_refs": [
                                {
                                    "asset": "assets/visual/crop.webp",
                                    "source_asset": "assets/visual/frame.webp",
                                }
                            ]
                        },
                        {"frame_refs": [{"asset": "assets/visual/second-crop.webp"}]},
                    ]
                },
            }
        }
        self.assertEqual(
            asset_refs(record),
            {
                "visual": {
                    "assets/visual/frame.webp",
                    "assets/visual/crop.webp",
                    "assets/visual/second-crop.webp",
                },
                "audio": {"assets/audio/clip.ogg"},
            },
        )


if __name__ == "__main__":
    unittest.main()
