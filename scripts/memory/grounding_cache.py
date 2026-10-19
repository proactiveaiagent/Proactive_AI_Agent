"""Persistent cache for identical object-to-frame grounding inputs."""
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

# Stable fingerprint of the object-grounding implementation used by the
# existing cache rows. Perception prompt edits must not invalidate grounding.
# Bump this when ground_objects' prompt or result schema changes.
GROUNDING_IMPLEMENTATION_FINGERPRINT = "opencv-hog-clip-vit-l14-grid-v1-20261019"

def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _usable_completed_payload(payload):
    """Accept reliable matches and cached low-confidence outcomes for exact inputs."""
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        return False
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        status = str(item.get("status") or "")
        if status == "not_verified":
            return False
        if status not in {"grounded", "visible_unlabeled", "not_visible", "uncertain"}:
            return False
        if status in {"grounded", "visible_unlabeled"} and not (
                item.get("frame_indices") or item.get("frame_boxes") or item.get("frame_refs")):
            return False
        if status == "grounded" and not str(item.get("appearance") or "").strip():
            return False
    return True

class PersistentObjectGroundingCache:
    def __init__(self, path, project_root):
        self.path=Path(path)
        self.project_root=Path(project_root)

    def key(self, segment_id, objects, frames):
        frame_rows=[]
        for frame in frames or []:
            raw=frame.get("bytes") or b""
            if isinstance(raw, str):
                raw=raw.encode("utf-8")
            frame_rows.append({"sha256":hashlib.sha256(raw).hexdigest(),"at_s":frame.get("at_s")})
        value={"schema":1,"segment":str(segment_id or ""),
               "objects":[str(x) for x in (objects or [])],"frames":frame_rows,
               "perceiver_source":GROUNDING_IMPLEMENTATION_FINGERPRINT,
               "model_path":os.environ.get("QWEN_MODEL_PATH",""),
               "model_revision":os.environ.get("QWEN_MODEL_REVISION","")}
        return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()

    def source_key(self, segment_id, video_path, start_s, end_s, objects):
        """Fast stable lookup key; invalidates when the clip, objects or grounder change."""
        try:
            stat = Path(video_path).stat()
        except (OSError, TypeError, ValueError):
            return ""
        value = {
            "schema": 1,
            "segment": str(segment_id or ""),
            "video_size": int(stat.st_size),
            "video_mtime_ns": int(stat.st_mtime_ns),
            "clip": [round(float(start_s or 0), 3),
                     round(float(end_s), 3) if end_s is not None else None],
            "objects": [str(x) for x in (objects or [])],
            "perceiver_source": GROUNDING_IMPLEMENTATION_FINGERPRINT,
            "model_path": os.environ.get("QWEN_MODEL_PATH", ""),
            "model_revision": os.environ.get("QWEN_MODEL_REVISION", ""),
            "sample_count": 7,
        }
        return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()

    def _connect(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(str(self.path),timeout=8)
        db.execute("PRAGMA busy_timeout=8000")
        db.execute("CREATE TABLE IF NOT EXISTS grounding_cache (cache_key TEXT PRIMARY KEY,payload TEXT NOT NULL,created_at REAL NOT NULL,last_access REAL NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS grounding_source_cache (source_key TEXT PRIMARY KEY,payload TEXT NOT NULL,created_at REAL NOT NULL,last_access REAL NOT NULL)")
        try: os.chmod(self.path,0o600)
        except OSError: pass
        return db

    def get(self,key):
        if not key: return None,0
        start=time.perf_counter()
        try:
            with self._connect() as db:
                row=db.execute("SELECT payload FROM grounding_cache WHERE cache_key=?",(key,)).fetchone()
                if not row: return None,round((time.perf_counter()-start)*1000)
                db.execute("UPDATE grounding_cache SET last_access=? WHERE cache_key=?",(time.time(),key))
            payload=json.loads(row[0])
            return (payload if _usable_completed_payload(payload) else None),round((time.perf_counter()-start)*1000)
        except (OSError,sqlite3.Error,ValueError,TypeError):
            return None,round((time.perf_counter()-start)*1000)

    def put(self,key,payload):
        if not key or not _usable_completed_payload(payload): return False
        try:
            now=time.time()
            with self._connect() as db:
                db.execute("INSERT OR REPLACE INTO grounding_cache VALUES (?,?,?,?)",(key,_json(payload),now,now))
                db.execute("DELETE FROM grounding_cache WHERE cache_key IN (SELECT cache_key FROM grounding_cache ORDER BY last_access DESC LIMIT -1 OFFSET 5000)")
            return True
        except (OSError,sqlite3.Error,TypeError,ValueError):
            return False

    def get_source(self, source_key):
        if not source_key:
            return None, 0
        start = time.perf_counter()
        try:
            with self._connect() as db:
                row = db.execute(
                    "SELECT payload FROM grounding_source_cache WHERE source_key=?",
                    (source_key,),
                ).fetchone()
                if not row:
                    return None, round((time.perf_counter() - start) * 1000)
                db.execute(
                    "UPDATE grounding_source_cache SET last_access=? WHERE source_key=?",
                    (time.time(), source_key),
                )
            payload = json.loads(row[0])
            valid = _usable_completed_payload(payload)
            return (payload if valid else None), round((time.perf_counter() - start) * 1000)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return None, round((time.perf_counter() - start) * 1000)

    def put_source(self, source_key, payload):
        if not source_key or not _usable_completed_payload(payload):
            return False
        try:
            now = time.time()
            with self._connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO grounding_source_cache VALUES (?,?,?,?)",
                    (source_key, _json(payload), now, now),
                )
                db.execute(
                    "DELETE FROM grounding_source_cache WHERE source_key IN "
                    "(SELECT source_key FROM grounding_source_cache "
                    "ORDER BY last_access DESC LIMIT -1 OFFSET 5000)"
                )
            return True
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return False
