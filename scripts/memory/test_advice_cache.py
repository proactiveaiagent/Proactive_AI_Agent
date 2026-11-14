"""Regression checks for persistent scene and retrieval caches."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
from advice_cache import PersistentAdviceCache


class TrackingConnection:
    def __init__(self, connection, closed):
        self.connection = connection
        self.closed = closed

    def execute(self, *args, **kwargs):
        return self.connection.execute(*args, **kwargs)

    def commit(self):
        return self.connection.commit()

    def rollback(self):
        return self.connection.rollback()

    def close(self):
        self.closed.append(True)
        return self.connection.close()


class PersistentSceneCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scene-cache-")
        self.root = Path(self.temp.name)
        (self.root / "code").mkdir()
        (self.root / "code" / "memory_reasoner.py").write_text("v1", encoding="utf-8")
        self.environment = patch.dict(os.environ, {
            "QWEN_MODEL_PATH": str(self.root / "missing-model"),
            "WHISPER_MODEL_PATH": str(self.root / "missing-whisper"),
        })
        self.environment.start()
        self.cache = PersistentAdviceCache(self.root / "cache.sqlite3", self.root)
        self.closed = []
        connect = self.cache._connect

        def tracked_connect():
            return TrackingConnection(connect(), self.closed)

        self.cache._connect = tracked_connect

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def memory(self, moment_id="session-a", scene="A person places a phone on a table."):
        moment = {
            "id": moment_id,
            "timestamp": "2026-10-16T11:09:30",
            "scene": scene,
            "source_provenance": {"source_segment_id": "clip-01"},
        }
        return {"layer7": {"moments": {moment_id: moment}}}

    def test_retrieval_cache_reuses_stable_moment_ids_and_closes_connections(self):
        first_key, first_ids = self.cache.key(
            self.memory("session-a"), "dataset", "scene-retrieval-v1",
            "A person places a phone on a table.", {"source_facts": ["phone placed"]},
        )
        result = {
            "mode": "memory_retrieval",
            "candidates": [{"moment_id": "session-a", "text": "Phone placed on table."}],
        }
        self.assertTrue(self.cache.put_retrieval(first_key, result, first_ids, 42))
        second_key, second_ids = self.cache.key(
            self.memory("session-b"), "dataset", "scene-retrieval-v1",
            "A person places a phone on a table.", {"source_facts": ["phone placed"]},
        )
        self.assertEqual(first_key, second_key)
        cached, _elapsed = self.cache.get(second_key, second_ids)
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["candidates"][0]["moment_id"], "session-b")
        self.assertEqual(cached[1]["inference_ms"], 42)
        self.assertEqual(cached[1]["duration_ms"], 42)
        self.assertEqual(cached[1]["duration_kind"], "retrieval_total")
        self.assertEqual(len(self.closed), 2)

    def test_memory_or_scene_change_invalidates_retrieval_cache(self):
        original, _ = self.cache.key(
            self.memory(), "dataset", "scene-retrieval-v1",
            "A person places a phone on a table.", {"source_facts": ["phone placed"]},
        )
        changed_memory, _ = self.cache.key(
            self.memory(scene="A person moves a phone into a drawer."),
            "dataset", "scene-retrieval-v1",
            "A person places a phone on a table.", {"source_facts": ["phone placed"]},
        )
        changed_scene, _ = self.cache.key(
            self.memory(), "dataset", "scene-retrieval-v1",
            "A person moves the phone into a drawer.", {"source_facts": ["phone placed"]},
        )
        self.assertNotEqual(original, changed_memory)
        self.assertNotEqual(original, changed_scene)

    def test_scene_plan_key_reuses_same_scene_across_session_ids(self):
        context = {
            "source_moment_id": "random-session-moment-a",
            "source_facts": [{"scene": "Two people stand by a gate."}],
            "visual_object_observations": ["wooden gate"],
        }
        first, _ = self.cache.key_scene_plan("dataset", "Two people by a gate.", context)
        context["source_moment_id"] = "random-session-moment-b"
        second, _ = self.cache.key_scene_plan("dataset", "Two people by a gate.", context)
        self.assertEqual(first, second)
        changed, _ = self.cache.key_scene_plan(
            "dataset", "One person opens the gate.", context
        )
        self.assertNotEqual(first, changed)

    def test_video_source_hash_invalidates_retrieval_and_scene_plan_keys(self):
        before = self.memory()
        before_moment = before["layer7"]["moments"]["session-a"]
        before_moment["source_provenance"].update({
            "source_video_sha256": "a" * 64,
            "source_clip_sha256": "b" * 64,
        })
        before_context = {
            "source_facts": before_moment,
            "source_moment_id": "session-a",
            "retrieval_policy_version": "scene-memory-retrieval-v6",
        }
        retrieval_before, _ = self.cache.key(
            before, "dataset", "scene-retrieval-v1", "scene", before_context
        )
        plan_before, _ = self.cache.key_scene_plan("dataset", "scene", before_context)

        after = self.memory()
        after_moment = after["layer7"]["moments"]["session-a"]
        after_moment["source_provenance"].update({
            "source_video_sha256": "c" * 64,
            "source_clip_sha256": "d" * 64,
        })
        after_context = {
            "source_facts": after_moment,
            "source_moment_id": "session-a",
            "retrieval_policy_version": "scene-memory-retrieval-v6",
        }
        retrieval_after, _ = self.cache.key(
            after, "dataset", "scene-retrieval-v1", "scene", after_context
        )
        plan_after, _ = self.cache.key_scene_plan("dataset", "scene", after_context)

        self.assertNotEqual(retrieval_before, retrieval_after)
        self.assertNotEqual(plan_before, plan_after)

    def test_source_version_change_invalidates_cache_key(self):
        before, _ = self.cache.key(
            self.memory(), "dataset", "scene-retrieval-v1", "scene", {"facts": []}
        )
        (self.root / "code" / "memory_reasoner.py").write_text("v2", encoding="utf-8")
        after, _ = self.cache.key(
            self.memory(), "dataset", "scene-retrieval-v1", "scene", {"facts": []}
        )
        self.assertNotEqual(before, after)

    def test_retrieval_cache_rejects_partial_or_wrong_mode_payloads(self):
        key, ids = self.cache.key(
            self.memory(), "dataset", "scene-retrieval-v1", "scene", {"facts": []}
        )
        self.assertFalse(self.cache.put_retrieval(
            key, {"mode": "memory_retrieval", "error": "cancelled"}, ids, 1
        ))
        self.assertFalse(self.cache.put_retrieval(
            key, {"recommendation": {"text": "do something"}}, ids, 1
        ))

    def test_scene_plan_cache_round_trip(self):
        key, ids = self.cache.key(
            {}, "dataset", "scene-plan-v1", "two people at home", {"entities": ["person"]}
        )
        result = {"mode": "scene_plan", "scene_plan": {
            "entities": ["person"], "searches": ["person at home"],
        }}
        self.assertTrue(self.cache.put_scene_plan(key, result, ids, 18))
        cached, _elapsed = self.cache.get(key, ids)
        self.assertEqual(cached[0], result)
        self.assertEqual(cached[1]["inference_ms"], 18)
        self.assertEqual(cached[1]["duration_ms"], 18)
        self.assertEqual(cached[1]["duration_kind"], "scene_plan")

    def test_legacy_cache_schema_reads_duration_without_claiming_its_kind(self):
        key, stable_ids = self.cache.key(
            self.memory(), "dataset", "scene-retrieval-v1", "scene", {"facts": []}
        )
        payload = json.dumps({"mode": "memory_retrieval", "candidates": []})
        with sqlite3.connect(self.cache.path) as connection:
            connection.execute(
                "CREATE TABLE advice_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "created_at REAL NOT NULL, last_access REAL NOT NULL, inference_ms INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT INTO advice_cache VALUES (?, ?, ?, ?, ?)",
                (key, payload, 1.0, 1.0, 7651),
            )
        cached, _lookup_ms = self.cache.get(key, stable_ids)
        self.assertEqual(cached[1]["duration_ms"], 7651)
        self.assertEqual(cached[1]["duration_kind"], "legacy_unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
