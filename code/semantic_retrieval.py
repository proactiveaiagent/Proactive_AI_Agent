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
FIELD_LABELS = {
    "scene": "Scene", "user_action": "Action", "people": "People",
    "location": "Location", "activity": "Activity", "environments": "Environment",
    "objects": "Objects", "needs": "Needs", "solutions": "Solutions",
    "time_occasion": "Time", "source_evidence": "Dialogue evidence",
}


def memory_text(moment: dict) -> str:
    """Create a stable, field-aware embedding input without media file contents."""
    parts = []
    for key, label in FIELD_LABELS.items():
        value = moment.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if item not in (None, ""))
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        parts.append(f"{label}: {value}")
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
            db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES( schema_version,?)",
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

    def sync(self, moments: dict, encoder: GTEEncoder, batch_size: int = 32) -> dict:
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
            db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('corpus_digest',?)",
                       (corpus_digest,))
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

    def ready(self, moments: dict, encoder: GTEEncoder) -> bool:
        """True only when this model's index exactly matches the active memory text."""
        documents = self._active_documents(moments)
        digest = hashlib.sha256(json.dumps(
            sorted((mid, text_digest(text)) for mid, text in documents.items()),
            ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        with self._connect() as db:
            meta = dict(db.execute("SELECT key,value FROM index_meta"))
        return meta.get("model_id") == encoder.model_id and meta.get("corpus_digest") == digest

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
_PENDING: dict[str, dict] = {}
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
            moments = _PENDING.pop(memory_dir)
            if _ENCODER is None:
                _ENCODER_LOADING = True
        try:
            if _ENCODER is None:
                _ENCODER = GTEEncoder()
            PersistentSemanticIndex(memory_dir).sync(moments, _ENCODER)
        except Exception as exc:
            # Keep the failure visible to service logs; query remains on BM25.
            print(f"[semantic-index] sync failed for {memory_dir}: {exc}", flush=True)
        finally:
            _ENCODER_LOADING = False


def schedule_sync(memory_dir: str | Path, moments: dict) -> None:
    """Queue an immutable memory snapshot; model loading runs off request thread."""
    global _WORKER_STARTED
    if os.environ.get("MEMORY_SEMANTIC_RRF", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    key = str(Path(memory_dir).resolve())
    snapshot = copy.deepcopy(moments)
    with _WORKER_LOCK:
        _PENDING[key] = snapshot
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
    worker = threading.Thread(target=_background_sync_loop,
                              name="memory-semantic-index", daemon=True)
    worker.start()
