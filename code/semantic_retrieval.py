"""Persistent, incrementally synchronized text-semantic vector index.

The index is optionally wired to PersonMemory CRUD and query via
MEMORY_SEMANTIC_RRF=1. Raw pixels and audio are not embedded; online fusion
remains opt-in until consistency, latency, and held-out retrieval are validated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

MODEL_ID = "thenlper/gte-base"
SCHEMA_VERSION = "1"
DOCUMENT_VERSION = "3"
FIELD_LABELS = {
    "scene": "Scene",
    "user_action": "Action",
    "interaction_details": "People and event sequence",
    "people": "People",
    "location": "Location",
    "activity": "Activity",
    "environments": "Environment",
    "objects": "Objects",
    "object_states": "Object states and locations",
    "notes": "Additional observed context",
    "time_occasion": "Time",
}


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(part for item in value if (part := _text(item)))
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {part}" for key, item in sorted(value.items())
            if (part := _text(item))
        )
    return str(value).strip()


def _observation_rows(value):
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _row_text(label: str, row: dict, fields: tuple[tuple[str, str], ...]) -> str:
    parts = [f"{name}: {_text(row.get(key))}" for key, name in fields
             if _text(row.get(key))]
    return f"{label}: " + "; ".join(parts) if parts else ""


def memory_text(moment: dict) -> str:
    """Build retrieval text from observed memory fields, never inferred advice.

    The content version is bumped whenever this representation changes so a
    stale persistent vector index cannot be treated as current.
    """
    parts = []
    for key, label in FIELD_LABELS.items():
        value = moment.get(key)
        text = _text(value)
        if text:
            parts.append(f"{label}: {text}")

    source_time = moment.get("source_time")
    if isinstance(source_time, dict):
        time_fields = (
            ("day_label", "day"), ("time_of_day", "time of day"),
            ("time_occasion", "occasion"), ("occurred_at", "occurred at"),
            ("time_scope", "observed time range"),
            ("clip_start_s", "clip start seconds"),
            ("clip_end_s", "clip end seconds"),
        )
        time_text = _row_text("Source time", source_time, time_fields)
        if time_text:
            parts.append(time_text)

    for row in _observation_rows(moment.get("person_observations")):
        fields = [
            ("entity", "person"),
            ("appearance", "visible appearance"),
            ("visual_label", "visible description"),
            ("evidence", "visual evidence"),
            ("frame_time_s", "observed at seconds"),
        ]
        apparent_age = _text(row.get("apparent_age_range"))
        if apparent_age and row.get("apparent_age_basis") == "visual_estimate":
            fields.append(("apparent_age_range", "broad visual age estimate"))
        if row.get("age_basis") == "visible_text":
            fields.append(("age", "age stated in visible text"))
        line = _row_text("Person observation", row, tuple(fields))
        if line:
            parts.append(line)

    fact_fields = (
        ("subject_type", "type"), ("subject", "subject"),
        ("predicate", "action or state"), ("value", "detail"),
        ("actor", "actor"), ("affected_people", "affected people"),
        ("from_location", "from location"), ("to_location", "to location"),
        ("from_state", "previous state"), ("to_state", "current state"),
        ("time_scope", "time range"), ("time_order", "event order"),
        ("frame_time_s", "observed at seconds"), ("status", "status"),
        ("evidence", "visual or spoken evidence"),
    )
    for row in _observation_rows(moment.get("temporal_facts")):
        line = _row_text("Timed fact", row, fact_fields)
        if line:
            parts.append(line)

    object_fields = (
        ("entity", "object"), ("object", "object"), ("name", "object"),
        ("label", "label"), ("description", "description"),
        ("appearance", "appearance"), ("state", "state"),
        ("location", "location"), ("from_location", "from location"),
        ("to_location", "to location"), ("condition", "condition"),
        ("time_scope", "time range"), ("frame_time_s", "observed at seconds"),
        ("evidence", "evidence"),
    )
    for row in _observation_rows(moment.get("object_observations")):
        line = _row_text("Object observation", row, object_fields)
        if line:
            parts.append(line)

    for row in _observation_rows(moment.get("object_states")):
        fields = [
            ("object", "object"), ("predicate", "transition"),
            ("state", "state"), ("location", "location"),
            ("status", "status"), ("scope", "scope"),
            ("confidence", "confidence"), ("observed_at_s", "observed at seconds"),
            ("inferred_at_s", "inferred at seconds"), ("evidence", "evidence"),
        ]
        line = _row_text("Object state/location", row, tuple(fields))
        if line:
            parts.append(line)

    # Keep raw dialogue available for text search without letting a long
    # transcript crowd out the structured visual and temporal facts above.
    dialogue = _text(moment.get("source_evidence"))
    if dialogue:
        parts.append("Dialogue evidence: " + dialogue[:1200])
    return "\n".join(parts)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GTEEncoder:
    """Load the pinned local GTE snapshot; never downloads during a memory operation."""
    def __init__(self, model_root: str | Path | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        root = Path(model_root or os.environ.get(
            "MEMORY_GTE_MODEL_ROOT",
            "/data/cxr25/zhx/HF_HOME/sentence_transformers/models--thenlper--gte-base/snapshots",
        ))
        if root.name == "snapshots":
            snapshots = sorted(path for path in root.iterdir() if path.is_dir())
            if not snapshots:
                raise FileNotFoundError(f"No local GTE snapshot in {root}")
            root = snapshots[-1]
        torch.set_num_threads(max(1, int(os.environ.get("MEMORY_GTE_CPU_THREADS", "4"))))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(root), local_files_only=True).eval()
        self.model_id = MODEL_ID + "@" + root.name

    def encode(self, texts: list[str], batch_size: int = 32):
        import numpy as np
        result = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            result.extend(pooled.cpu().numpy().astype(np.float16))
        return result


class PersistentSemanticIndex:
    """SQLite sidecar keyed by source moment ID with hash-based incremental sync.

    sync() serves explicit maintenance and the opt-in background worker. Do not
    place model loading/vectorization on a request path.
    Each PersonMemory should receive its own memory_dir (and therefore index).
    """
    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.path = self.memory_dir / "semantic_index.sqlite3"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors ("
                       "moment_id TEXT PRIMARY KEY, text_hash TEXT NOT NULL, "
                       "model_id TEXT NOT NULL, dimension INTEGER NOT NULL, "
                       "vector BLOB NOT NULL, updated_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO index_meta(key,value) VALUES('schema_version',?)",
                       (SCHEMA_VERSION,))

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def _active_documents(moments: dict) -> dict[str, str]:
        documents = {}
        for mid, moment in moments.items():
            if not isinstance(moment, dict) or moment.get("stale") or moment.get("merged_into"):
                continue
            text = memory_text(moment)
            source_texts = []
            for source_id in moment.get("merged_sources", []):
                source = moments.get(source_id)
                if (not isinstance(source, dict) or source.get("merged_into") != moment.get("id")
                        or source.get("stale")):
                    continue
                source_text = memory_text(source)
                if source_text:
                    source_texts.append(source_text)
            if source_texts:
                text += "\nMerged source records:\n" + "\n\n".join(source_texts)
            if text:
                documents[str(mid)] = text
        return documents

    def sync(self, moments: dict, encoder: GTEEncoder, batch_size: int = 32,
             revision: int | None = None) -> dict:
        """Upsert changed/new active moments and remove deleted/stale/merged rows."""
        import numpy as np
        documents = self._active_documents(moments)
        digests = {mid: text_digest(text) for mid, text in documents.items()}
        with self._connect() as db:
            old = {row[0]: row[1] for row in db.execute(
                "SELECT moment_id,text_hash FROM vectors WHERE model_id=?", (encoder.model_id,))}
            obsolete = set(old) - set(documents)
            changed = [mid for mid, digest in digests.items() if old.get(mid) != digest]
            if obsolete:
                db.executemany("DELETE FROM vectors WHERE moment_id=?", [(mid,) for mid in obsolete])
            encoded = encoder.encode([documents[mid] for mid in changed], batch_size=batch_size) if changed else []
            for mid, vector in zip(changed, encoded):
                packed = np.asarray(vector, dtype="<f2").tobytes()
                db.execute("INSERT OR REPLACE INTO vectors VALUES(?,?,?,?,?,?)",
                           (mid, digests[mid], encoder.model_id, len(vector), packed, time.time()))
            # A model change invalidates rows from all earlier encoders.
            db.execute("DELETE FROM vectors WHERE model_id<>?", (encoder.model_id,))
            corpus_digest = hashlib.sha256(json.dumps(
                sorted(digests.items()), ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('model_id',?)",
                       (encoder.model_id,))
            db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('document_version',?)",
                       (DOCUMENT_VERSION,))
            db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('corpus_digest',?)",
                       (corpus_digest,))
            db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('active_count',?)",
                       (str(len(documents)),))
            if revision is not None:
                db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('memory_revision',?)",
                           (str(revision),))
        return {"active": len(documents), "added_or_updated": len(changed),
                "removed": len(obsolete), "unchanged": len(documents) - len(changed),
                "model_id": encoder.model_id, "bytes": self.path.stat().st_size}

    def search(self, query: str, encoder: GTEEncoder, top_k: int = 5,
               allowed_ids: set[str] | None = None) -> list[dict]:
        """Cosine-rank the existing index; query encoding may still be expensive."""
        import numpy as np
        if top_k <= 0 or not query.strip():
            return []
        query_vector = np.asarray(encoder.encode([query])[0], dtype=np.float32)
        with self._connect() as db:
            rows = db.execute(
                "SELECT moment_id,dimension,vector FROM vectors WHERE model_id=?",
                (encoder.model_id,)).fetchall()
        scored = []
        for mid, dimension, blob in rows:
            if allowed_ids is not None and mid not in allowed_ids:
                continue
            vector = np.frombuffer(blob, dtype="<f2", count=dimension).astype(np.float32)
            score = float(query_vector @ vector)
            scored.append({"moment_id": mid, "score": score})
        scored.sort(key=lambda row: (-row["score"], row["moment_id"]))
        return scored[:top_k]

    def ready(self, moments: dict, encoder: GTEEncoder,
              revision: int | None = None) -> bool:
        """Check freshness by revision in queries; retain digest validation for maintenance callers."""
        with self._connect() as db:
            meta = dict(db.execute("SELECT key,value FROM index_meta"))
        if meta.get("model_id") != encoder.model_id:
            return False
        if meta.get("document_version") != DOCUMENT_VERSION:
            return False
        try:
            expected_count = int(meta.get("active_count", "-1"))
        except (TypeError, ValueError):
            return False
        with self._connect() as db:
            actual_count = db.execute(
                "SELECT count(*) FROM vectors WHERE model_id=?", (encoder.model_id,)
            ).fetchone()[0]
        if actual_count != expected_count:
            return False
        if revision is not None:
            return meta.get("memory_revision") == str(revision)
        documents = self._active_documents(moments)
        digest = hashlib.sha256(json.dumps(
            sorted((mid, text_digest(text)) for mid, text in documents.items()),
            ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return meta.get("corpus_digest") == digest

    def stats(self) -> dict:
        with self._connect() as db:
            count = db.execute("SELECT count(*) FROM vectors").fetchone()[0]
            models = [row[0] for row in db.execute("SELECT DISTINCT model_id FROM vectors")]
        return {"path": str(self.path), "vectors": count, "models": models,
                "bytes": self.path.stat().st_size if self.path.exists() else 0}


# One lazy, process-wide encoder and one coalescing background worker. The worker
# keeps only the newest pending snapshot per memory directory so rapid CRUD writes
# do not create a queue of obsolete full-corpus rebuilds.
_ENCODER = None
_ENCODER_LOADING = False
_WORKER_STARTED = False
_PENDING: dict[str, tuple[dict | None, int | None]] = {}
_WORKER_LOCK = threading.Lock()


def loaded_encoder():
    """Return the warm encoder without waiting for model initialization."""
    return _ENCODER


def _background_sync_loop():
    global _ENCODER, _ENCODER_LOADING, _WORKER_STARTED
    while True:
        with _WORKER_LOCK:
            if not _PENDING:
                _WORKER_STARTED = False
                return
            memory_dir = sorted(_PENDING)[0]
            moments, revision = _PENDING.pop(memory_dir)
            if _ENCODER is None:
                _ENCODER_LOADING = True
        try:
            if moments is None:
                # PersonMemory has already atomically persisted this generation.
                # Read the latest complete snapshot off-thread, avoiding a full
                # deepcopy of hundreds or thousands of moments on the CRUD path.
                memory_path = Path(memory_dir) / "memory.json"
                if memory_path.exists():
                    with memory_path.open("r", encoding="utf-8") as handle:
                        persisted = json.load(handle)
                    moments = persisted.get("layer7", {}).get("moments", {})
                    revision = persisted.get("metadata", {}).get("semantic_index_revision", revision)
                else:
                    moments = {}
            if _ENCODER is None:
                _ENCODER = GTEEncoder()
            PersistentSemanticIndex(memory_dir).sync(moments, _ENCODER, revision=revision)
        except Exception as exc:
            # Keep the failure visible to service logs; query remains on BM25.
            print(f"[semantic-index] sync failed for {memory_dir}: {exc}", flush=True)
        finally:
            _ENCODER_LOADING = False


def schedule_sync(memory_dir: str | Path, moments: dict | None = None,
                  revision: int | None = None) -> None:
    """Queue a snapshot, or read PersonMemory's atomic file in the background."""
    global _WORKER_STARTED
    if os.environ.get("MEMORY_SEMANTIC_RRF", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    key = str(Path(memory_dir).resolve())
    # Explicit maintenance callers can still hand in a mutable dictionary, so
    # freeze those snapshots. PersonMemory passes None after os.replace and lets
    # the worker read the durable snapshot without request-thread deepcopy cost.
    snapshot = copy.deepcopy(moments) if moments is not None else None
    with _WORKER_LOCK:
        _PENDING[key] = (snapshot, revision)
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
    worker = threading.Thread(target=_background_sync_loop,
                              name="memory-semantic-index", daemon=True)
    worker.start()
