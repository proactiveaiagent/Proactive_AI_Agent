"""Bind content-addressed grounding rows to stable source-clip identities.

This CPU-only pass decodes the seven frames once, reads existing completed
grounding entries, and stores source aliases. It never calls a model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "code"))

import prewarm_multimodal_evidence as prewarm
from grounding_cache import PersistentObjectGroundingCache
from multimodal_memory import MultimodalAssetStore
from qwen_perceive import QwenPerceiver, clip_range, video_path_of

RUNS = ROOT / "data" / "egolife" / "runs"
QWEN_CACHE = RUNS / "qwen_cache_memory_prompt_v3.json"
GROUNDING_CACHE = RUNS / "interactive_object_grounding_cache.sqlite3"


def run(limit: int | None = None, all_profiles: bool = False) -> dict:
    moments = prewarm.load_moments(all_profiles)
    if not all_profiles:
        moments = [m for m in moments if str(m.get("id") or "").startswith("FEISHU_")]
    if limit:
        moments = moments[:max(1, int(limit))]
    perceiver = QwenPerceiver(cache_file=QWEN_CACHE)
    cache = PersistentObjectGroundingCache(GROUNDING_CACHE, ROOT)
    media = object.__new__(MultimodalAssetStore)
    totals = {"clips": len(moments), "aliases_added": 0, "already_indexed": 0,
              "grounding_missing": 0, "frame_decode_failed": 0, "no_objects": 0}
    for moment in moments:
        mid = str(moment.get("id") or "")
        entry = perceiver.cache.get(mid) or {}
        fields = entry.get("fields") or {}
        labels = fields.get("objects") or []
        video = video_path_of(moment)
        if entry.get("prompt_version") != prewarm.PROMPT_VERSION or not labels or not video:
            totals["no_objects"] += 1
            continue
        start, end = clip_range(moment)
        source_key = cache.source_key(mid, video, start, end, labels)
        existing, _ = cache.get_source(source_key)
        if existing is not None:
            totals["already_indexed"] += 1
            continue
        frames = media.sample_visual_candidates(video, start, end, source_id=mid, count=7)
        if not frames:
            totals["frame_decode_failed"] += 1
            continue
        content_key = cache.key(mid, labels, frames)
        payload, _ = cache.get(content_key)
        if payload is None:
            totals["grounding_missing"] += 1
            continue
        if cache.put_source(source_key, payload):
            totals["aliases_added"] += 1
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-profiles", action="store_true",
                        help="build aliases for every selectable short/long/custom profile")
    args = parser.parse_args()
    print(run(args.limit, args.all_profiles))
