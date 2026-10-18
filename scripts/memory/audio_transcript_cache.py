"""Cross-process cache for source-audio transcripts."""
from __future__ import annotations
import copy
import fcntl
import json
import os
import tempfile
import threading
from pathlib import Path

class PersistentTranscriptCache:
    """Refresh atomic replacements and merge writes under a process lock."""
    def __init__(self, path):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._lock = threading.RLock()
        self._cache = None
        self._stamp = None

    @staticmethod
    def _signature(stat_result):
        return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size,
                stat_result.st_mtime_ns, stat_result.st_ctime_ns)

    def _file_signature(self):
        try:
            return self._signature(self.path.stat())
        except OSError:
            return None

    def _read_snapshot(self):
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                signature = self._signature(os.fstat(handle.fileno()))
                value = json.load(handle)
        except (OSError, ValueError, TypeError):
            return {}, None
        return (value if isinstance(value, dict) else {}), signature

    def _refresh_locked(self):
        current = self._file_signature()
        if self._cache is None or current != self._stamp:
            self._cache, self._stamp = self._read_snapshot()

    def get(self, key):
        with self._lock:
            self._refresh_locked()
            value = (self._cache or {}).get(str(key))
            return copy.deepcopy(value) if isinstance(value, dict) else None

    def contains(self, key):
        return self.get(key) is not None

    def put_if_absent(self, key, value):
        key = str(key or "")
        if not key or not isinstance(value, dict):
            return None, False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            with os.fdopen(lock_fd, "a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    snapshot, _ = self._read_snapshot()
                    current = snapshot.get(key)
                    inserted = not isinstance(current, dict)
                    if inserted:
                        current = copy.deepcopy(value)
                        snapshot[key] = current
                        fd, temporary_name = tempfile.mkstemp(
                            prefix=self.path.name + ".", suffix=".tmp",
                            dir=str(self.path.parent))
                        try:
                            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                                json.dump(snapshot, handle, ensure_ascii=False)
                                handle.flush()
                                os.fsync(handle.fileno())
                            os.replace(temporary_name, self.path)
                            try:
                                os.chmod(self.path, 0o600)
                            except OSError:
                                pass
                        finally:
                            try:
                                os.unlink(temporary_name)
                            except FileNotFoundError:
                                pass
                    self._cache = snapshot
                    self._stamp = self._file_signature()
                    return copy.deepcopy(current), inserted
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
