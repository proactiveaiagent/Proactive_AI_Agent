# -*- coding: utf-8 -*-
"""Memory module: store GUI recognition results, clean/compress, build user profile, support search.

Week 3: Store recognition results, data cleaning & compression, personalized user profile,
        keyword and natural language retrieval.

Reference: M3-Agent (https://github.com/ByteDance-Seed/m3-agent) - entity-centric memory.
"""
import json
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Optional


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


class MemoryStore:
    """Store, clean, compress GUI recognition results; build user profile; support retrieval."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(os.path.dirname(__file__), "memory", "memory.db")
        _ensure_dir(db_path)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_name TEXT,
                    frame_index INTEGER,
                    timestamp_sec REAL,
                    app_name TEXT,
                    page_name TEXT,
                    user_action TEXT,
                    description TEXT,
                    visible_text TEXT,
                    elements TEXT,
                    screenshot TEXT,
                    source_path TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compressed_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name TEXT,
                    page_name TEXT,
                    summary TEXT,
                    time_range TEXT,
                    frame_count INTEGER,
                    raw_ids TEXT,
                    created_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            for idx in (
                "CREATE INDEX IF NOT EXISTS idx_raw_app ON raw_records(app_name)",
                "CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_records(timestamp_sec)",
                "CREATE INDEX IF NOT EXISTS idx_comp_app ON compressed_records(app_name)",
            ):
                conn.execute(idx)
            conn.commit()

    def add_recognition_results(self, report: dict, source_path: str = "") -> int:
        """Store recognition results from analyzer report. Returns count of inserted records."""
        if "frames" not in report:
            return 0
        now = datetime.now().isoformat()
        count = 0
        with sqlite3.connect(self.db_path) as conn:
            for frame in report["frames"]:
                if frame.get("error"):
                    continue
                conn.execute(
                    """INSERT INTO raw_records (
                        video_name, frame_index, timestamp_sec, app_name, page_name,
                        user_action, description, visible_text, elements, screenshot,
                        source_path, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        report.get("video", ""),
                        frame.get("frame_index", 0),
                        frame.get("timestamp_sec", 0),
                        (frame.get("app_name") or "").strip(),
                        (frame.get("page_name") or "").strip(),
                        (frame.get("user_action") or "").strip(),
                        (frame.get("description") or "").strip(),
                        json.dumps(frame.get("visible_text") or [], ensure_ascii=False),
                        json.dumps(frame.get("elements") or [], ensure_ascii=False),
                        frame.get("screenshot", ""),
                        source_path,
                        now,
                    ),
                )
                count += 1
            conn.commit()
        return count

    def clean_and_compress(self) -> dict:
        """Merge similar data, remove ambiguous/illogical, produce compressed records.

        - Merge: same app+page within 10s time window
        - Remove ambiguous: empty app_name or description < 3 chars
        - Remove illogical: consecutive identical entries (dedup)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, video_name, frame_index, timestamp_sec, app_name, page_name,
                          user_action, description, visible_text, elements
                   FROM raw_records
                   WHERE app_name != '' AND LENGTH(TRIM(description)) >= 3
                   ORDER BY video_name, timestamp_sec"""
            ).fetchall()

        if not rows:
            return {"merged": 0, "removed_ambiguous": 0, "compressed": 0}

        # Deduplicate consecutive identical (app, page, description prefix)
        seen_key = None
        kept = []
        for r in rows:
            key = (r["app_name"], r["page_name"], (r["description"] or "")[:50])
            if key == seen_key:
                continue
            seen_key = key
            kept.append(dict(r))

        # Merge similar: group by (app, page) within 10s windows
        merged = []
        i = 0
        while i < len(kept):
            r = kept[i]
            group = [r]
            j = i + 1
            while j < len(kept) and kept[j]["app_name"] == r["app_name"] and kept[j]["page_name"] == r["page_name"]:
                if kept[j]["timestamp_sec"] - r["timestamp_sec"] <= 10:
                    group.append(kept[j])
                    j += 1
                else:
                    break
            summary = group[0]["description"]
            if len(group) > 1:
                summary = f"[{len(group)}次] {summary}"
            merged.append({
                "app_name": r["app_name"],
                "page_name": r["page_name"],
                "summary": summary,
                "time_range": f"{group[0]['timestamp_sec']:.0f}s-{group[-1]['timestamp_sec']:.0f}s",
                "frame_count": len(group),
                "raw_ids": ",".join(str(x["id"]) for x in group),
            })
            i = j

        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM compressed_records")
            for m in merged:
                conn.execute(
                    """INSERT INTO compressed_records (app_name, page_name, summary, time_range, frame_count, raw_ids, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (m["app_name"], m["page_name"], m["summary"], m["time_range"], m["frame_count"], m["raw_ids"], now),
                )
            conn.commit()

        return {
            "merged": len(merged),
            "removed_ambiguous": len(rows) - len(kept),
            "compressed": len(merged),
        }

    def build_user_profile(self) -> dict:
        """Build personalized user profile from compressed records (app usage, common pages)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT app_name, page_name, frame_count FROM compressed_records"
            ).fetchall()

        app_counts = defaultdict(int)
        app_pages = defaultdict(lambda: defaultdict(int))
        for r in rows:
            app_counts[r["app_name"]] += r["frame_count"]
            app_pages[r["app_name"]][r["page_name"]] += r["frame_count"]

        top_apps = sorted(app_counts.items(), key=lambda x: -x[1])[:15]
        profile = {
            "top_apps": [{"app": a, "count": c} for a, c in top_apps],
            "app_pages": {app: dict(sorted(pages.items(), key=lambda x: -x[1])[:5])
                          for app, pages in app_pages.items()},
            "total_events": sum(app_counts.values()),
            "updated_at": datetime.now().isoformat(),
        }

        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_profile (key, value, updated_at) VALUES (?,?,?)",
                ("profile", json.dumps(profile, ensure_ascii=False), now),
            )
            conn.commit()
        return profile

    def get_user_profile(self) -> Optional[dict]:
        """Get cached user profile."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT value FROM user_profile WHERE key = 'profile'").fetchone()
        if row:
            return json.loads(row[0])
        return None

    def search_keywords(self, keywords: list[str], limit: int = 50) -> list[dict]:
        """Search by keywords (SQL LIKE, OR logic)."""
        if not keywords:
            return []
        placeholders = " OR ".join(
            "(app_name LIKE ? OR page_name LIKE ? OR user_action LIKE ? OR description LIKE ? OR visible_text LIKE ?)"
            for _ in keywords
        )
        params = []
        for kw in keywords:
            term = f"%{kw}%"
            params.extend([term] * 5)
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT id, video_name, timestamp_sec, app_name, page_name, user_action, description, visible_text
                    FROM raw_records WHERE {placeholders} ORDER BY timestamp_sec DESC LIMIT ?""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def search_natural_language(self, query: str, limit: int = 50) -> list[dict]:
        """Search by natural language: extract keywords and search.

        Extracts CJK sequences and alphanumeric tokens, then runs keyword search.
        """
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", query.strip())
        tokens = [t for t in tokens if len(t) >= 1]
        if not tokens:
            return []
        return self.search_keywords(tokens, limit=limit)

    def get_stats(self) -> dict:
        """Get basic memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            raw_count = conn.execute("SELECT COUNT(*) FROM raw_records").fetchone()[0]
            comp_count = conn.execute("SELECT COUNT(*) FROM compressed_records").fetchone()[0]
        return {"raw_records": raw_count, "compressed_records": comp_count}


def store_report_into_memory(report: dict, source_path: str = "", memory_dir: str = "") -> dict:
    """Convenience: store report, clean/compress, build profile. Returns stats."""
    db_path = ""
    if memory_dir:
        db_path = os.path.join(memory_dir, "memory.db")
    store = MemoryStore(db_path)
    added = store.add_recognition_results(report, source_path)
    if added == 0:
        return {"added": 0}
    stats = store.clean_and_compress()
    profile = store.build_user_profile()
    return {"added": added, **stats, "profile_updated": True}
