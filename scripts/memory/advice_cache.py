"""Persistent, context-validated cache for proactive memory recommendations."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
import time
from pathlib import Path


CACHE_SCHEMA = 1
MAX_ENTRIES = 2000
VOLATILE_KEYS = {
    "access_count", "last_used", "extraction_batch_id", "session_start",
    "last_consolidation",
}
ID_KEYS = {
    "id", "moment_id", "memory_id", "source_moment_id", "from_id", "to_id",
    "source_id", "target_id",
}
ID_LIST_KEYS = {
    "moment_ids", "memory_ids", "evidence_ids", "candidate_ids", "source_ids",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize(value, stable_ids, parent_key=""):
    if isinstance(value, dict):
        if parent_key == "moments":
            rows = [
                {"cache_moment_id": stable_ids.get(str(moment_id), "unmapped:" + str(moment_id)),
                 "moment": _normalize(moment, stable_ids)}
                for moment_id, moment in value.items()
            ]
            return sorted(rows, key=lambda row: row["cache_moment_id"])
        return {
            key: _normalize(item, stable_ids, str(key))
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, stable_ids, parent_key) for item in value]
    if isinstance(value, str):
        return stable_ids.get(value, value)
    return value


def _moment_ids(memory_state):
    archive = (memory_state.get("layer7") or {}).get("moments") or {}
    groups = {}
    for moment_id, moment in archive.items():
        provenance = moment.get("source_provenance") or {}
        source_id = str(provenance.get("source_segment_id") or "").strip()
        base = "segment:" + source_id if source_id else "legacy"
        semantic = dict(_normalize(moment, {}))
        semantic.pop("id", None)
        signature = hashlib.sha256(_json(semantic).encode("utf-8")).hexdigest()[:20]
        groups.setdefault(base, []).append((str(moment_id), signature))
    stable_ids = {}
    for base, rows in groups.items():
        for moment_id, signature in rows:
            stable_ids[moment_id] = f"{base}:{signature}"
    return stable_ids


def _source_versions(project_root):
    files = (
        project_root / "scripts/memory/advice_cache.py",
        project_root / "scripts/memory/interactive_runner.py",
        project_root / "scripts/memory/qwen_perceive.py",
        project_root / "code/api_server.py",
        project_root / "code/entity_records.py",
        project_root / "code/memory_reasoner.py",
        project_root / "code/memory.py",
        project_root / "code/multimodal_memory.py",
        project_root / "code/daytime_processor.py",
    )
    versions = []
    for path in files:
        try:
            versions.append((path.relative_to(project_root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
        except OSError:
            continue
    return versions


def _model_version():
    model_path = Path(os.environ.get(
        "QWEN_MODEL_PATH", "/data/cxr25/zhx/HF_HOME/qwen/Qwen3-VL-8B-Instruct"
    ))
    weights = []
    if model_path.is_dir():
        candidates = [model_path / name for name in (
            "config.json", "generation_config.json", "processor_config.json",
            "preprocessor_config.json",
        )]
        for pattern in ("*.safetensors", "pytorch_model*.bin", "*.index.json"):
            candidates.extend(model_path.glob(pattern))
        for path in sorted(set(candidates), key=lambda item: item.name):
            try:
                stat = path.stat()
                weights.append((path.name, stat.st_size, stat.st_mtime_ns))
            except OSError:
                pass
    return {
        "qwen_path": str(model_path),
        "qwen_revision": os.environ.get("QWEN_MODEL_REVISION", ""),
        "whisper_path": os.environ.get("WHISPER_MODEL_PATH", ""),
        "whisper_size": os.environ.get("WHISPER_MODEL_SIZE", "tiny"),
        "weights": weights,
    }


def _runtime_options():
    prefixes = ("MEMORY_", "QWEN_MODEL_", "WHISPER_MODEL_")
    excluded = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    return {
        key: value for key, value in os.environ.items()
        if key.startswith(prefixes) and not any(secret in key.upper() for secret in excluded)
    }


class PersistentAdviceCache:
    """SQLite cache that only hits for identical memory, model and code versions."""

    def __init__(self, path, project_root, max_entries=MAX_ENTRIES):
        self.path = Path(path)
        self.project_root = Path(project_root)
        self.max_entries = int(max_entries)

    def key(self, memory_state, dataset_id, mode, situation, context):
        stable_ids = _moment_ids(memory_state)
        value = {
            "schema": CACHE_SCHEMA,
            "dataset": str(dataset_id or ""),
            "mode": str(mode or ""),
            "situation": str(situation or "").strip()[:1200],
            "context": _normalize(context or {}, stable_ids),
            "memory": _normalize(memory_state, stable_ids),
            "model": _model_version(),
            "source": _source_versions(self.project_root),
            "runtime": _runtime_options(),
        }
        digest = hashlib.sha256(_json(value).encode("utf-8")).hexdigest()
        return digest, stable_ids

    def key_scene_plan(self, dataset_id, situation, context):
        """Key a reusable scene plan without its ephemeral session moment ID."""
        stable_context = {
            key: value for key, value in (context or {}).items()
            if key != "source_moment_id"
        }
        return self.key(
            {}, dataset_id, "scene_retrieval_plan_v3", situation, stable_context
        )

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=8)
        connection.execute("PRAGMA busy_timeout=8000")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS advice_cache ("
            "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL, "
            "last_access REAL NOT NULL, inference_ms INTEGER NOT NULL, "
            "duration_ms INTEGER, duration_kind TEXT)"
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(advice_cache)")}
        if "duration_ms" not in columns:
            connection.execute("ALTER TABLE advice_cache ADD COLUMN duration_ms INTEGER")
        if "duration_kind" not in columns:
            connection.execute("ALTER TABLE advice_cache ADD COLUMN duration_kind TEXT")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, key, stable_ids):
        started = time.perf_counter()
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT payload, created_at, inference_ms, duration_ms, duration_kind "
                    "FROM advice_cache WHERE cache_key=?",
                    (key,),
                ).fetchone()
                if not row:
                    return None, round((time.perf_counter() - started) * 1000)
                connection.execute(
                    "UPDATE advice_cache SET last_access=? WHERE cache_key=?",
                    (time.time(), key),
                )
            reverse_ids = {stable: moment_id for moment_id, stable in stable_ids.items()}
            result = _normalize(json.loads(row[0]), reverse_ids)
            duration_ms = row[3] if row[3] is not None else row[2]
            metadata = {
                "created_at": row[1],
                "inference_ms": int(row[2] or 0),  # Backward-compatible alias.
                "duration_ms": int(duration_ms or 0),
                "duration_kind": str(row[4] or "legacy_unknown"),
            }
            return (result, metadata), round((time.perf_counter() - started) * 1000)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return None, round((time.perf_counter() - started) * 1000)

    def put(self, key, result, stable_ids, inference_ms):
        if not isinstance(result, dict) or not isinstance(result.get("recommendation"), dict):
            return False
        return self._put_payload(key, result, stable_ids, inference_ms, "reasoning_total")

    def put_retrieval(self, key, result, stable_ids, elapsed_ms):
        """Persist only successful retrieval-only results, never partial/cancelled runs."""
        if (not isinstance(result, dict)
                or result.get("mode") != "memory_retrieval"
                or result.get("cancelled") or result.get("error")):
            return False
        return self._put_payload(key, result, stable_ids, elapsed_ms, "retrieval_total")

    def put_scene_plan(self, key, result, stable_ids, elapsed_ms, duration_kind="scene_plan"):
        """Persist only a scene plan that contains no memory-dependent candidates."""
        if (not isinstance(result, dict)
                or result.get("mode") != "scene_plan"
                or not isinstance(result.get("scene_plan"), dict)
                or result.get("cancelled") or result.get("error")):
            return False
        return self._put_payload(key, result, stable_ids, elapsed_ms, duration_kind)

    def _put_payload(self, key, result, stable_ids, duration_ms, duration_kind):
        try:
            canonical = _normalize(result, stable_ids)
            payload = json.dumps(canonical, ensure_ascii=False)
            now = time.time()
            with self._connection() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO advice_cache "
                    "(cache_key, payload, created_at, last_access, inference_ms, duration_ms, duration_kind) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key, payload, now, now, int(duration_ms or 0),
                     int(duration_ms or 0), str(duration_kind or "unknown")),
                )
                connection.execute(
                    "DELETE FROM advice_cache WHERE cache_key IN ("
                    "SELECT cache_key FROM advice_cache ORDER BY last_access DESC LIMIT -1 OFFSET ?)",
                    (self.max_entries,),
                )
            return True
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return False
