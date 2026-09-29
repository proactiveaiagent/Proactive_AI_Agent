"""End-of-day consolidation for the two-stage memory pipeline.

Reads one day's Layer 7 archive, repairs six indices, summarizes bounded batches,
updates Layers 4/5, extracts a validated Layer 6 profile, and records an
idempotency digest. Dataset ground truth is never read here.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

_CODE_DIR = str(Path(__file__).resolve().parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from memory import _is_valid_index_tag, _is_valid_person_tag, _today_str
from profile_extractor import (
    build_profile_extraction_prompt,
    has_any_value,
    parse_profile_output,
    validate_profile,
)

DEFAULT_QWEN_API_URL = "http://localhost:8000"
DEFAULT_CHUNK_CHARS = 18000
_INDEX_FIELDS = {
    "people": ("people", "list"),
    "locations": ("location", "scalar"),
    "time_nodes": (None, "date"),
    "activity_events": ("activity", "scalar"),
    "environments": ("environments", "list"),
    "objects": ("objects", "list"),
}
_COMBINABLE = ("people", "locations", "activity_events", "environments", "objects")


class NightlyProcessor:
    """Consolidate one day after daytime ingestion has closed.

    llm_call is injectable for offline tests; otherwise local Qwen /consolidate
    is used.
    """

    def __init__(self, memory, qwen_api_url=DEFAULT_QWEN_API_URL,
                 llm_call: Optional[Callable[[str], str]] = None,
                 chunk_chars=DEFAULT_CHUNK_CHARS, timeout=180):
        if chunk_chars < 1000:
            raise ValueError("chunk_chars must be at least 1000")
        self.memory = memory
        self.qwen_api_url = qwen_api_url.rstrip("/")
        self.llm_call = llm_call
        self.chunk_chars = chunk_chars
        self.timeout = timeout
        self._run_lock = threading.Lock()

    @staticmethod
    def _moment_day(moment):
        ts = str(moment.get("timestamp") or "")
        try:
            return date.fromisoformat(ts[:10]).isoformat()
        except ValueError:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
            except (TypeError, ValueError):
                return None

    def _moments(self, day):
        """Layer 7 is the source of truth; Layer 1-3 are compatibility fallback."""
        result = {}
        archive = self.memory.memory.get("layer7", {}).get("moments", {})
        for mid, item in archive.items() if isinstance(archive, dict) else []:
            if isinstance(item, dict) and self._moment_day(item) == day:
                result[str(mid)] = item
        for layer in ("layer1", "layer2", "layer3"):
            for item in self.memory.memory.get(layer, []):
                if isinstance(item, dict) and self._moment_day(item) == day:
                    mid = str(item.get("id") or "")
                    if mid:
                        result.setdefault(mid, item)
        return sorted(result.values(), key=lambda m: (
            str(m.get("timestamp") or ""), str(m.get("id") or "")))

    @staticmethod
    def _brief(item):
        out = {"id": item.get("id", ""), "timestamp": item.get("timestamp", "")}
        for key, limit in (("scene", 500), ("user_action", 500),
                           ("location", 200), ("activity", 200),
                           ("notes", 500), ("extra_notes", 500)):
            if item.get(key):
                out[key] = str(item[key])[:limit]
        for key in ("people", "environments", "objects", "needs", "solutions"):
            if item.get(key):
                val = item[key]
                out[key] = val[:12] if isinstance(val, list) else val
        out["confirmed"] = bool((item.get("feedback") or {}).get("confirmed"))
        out["highlighted"] = bool(item.get("highlighted"))
        return out

    def _chunks(self, moments):
        chunks, current, size = [], [], 2
        for item in moments:
            brief = self._brief(item)
            n = len(json.dumps(brief, ensure_ascii=False))
            if current and size + n > self.chunk_chars:
                chunks.append(current)
                current, size = [], 2
            current.append(brief)
            size += n
        if current:
            chunks.append(current)
        return chunks

    def _call(self, prompt):
        if self.llm_call:
            value = self.llm_call(prompt)
            return json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
        r = requests.post(f"{self.qwen_api_url}/consolidate",
                          json={"prompt": prompt}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Qwen consolidate API {r.status_code}: {r.text[:300]}")
        value = r.json().get("analysis")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("Qwen consolidate API returned empty analysis")
        return value

    @staticmethod
    def _json(raw):
        from profile_extractor import parse_json_safe
        value = parse_json_safe(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Model response is not a JSON object: {raw[:240]}")
        return value

    def _summarize(self, day, i, total, chunk):
        prompt = f"""TASK: NIGHTLY_CHUNK_SUMMARY
Summarize this chronological slice of memories for {day}. Use only observations
provided; distinguish repeated patterns from one-off events; preserve uncertainty.
Return JSON only:
{{"summary":"short factual summary","key_events":["..."],"patterns":["..."],"profile_evidence":"evidence for stable profile extraction"}}
Slice {i}/{total}:
{json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}
"""
        value = self._json(self._call(prompt))
        if not isinstance(value.get("summary"), str):
            raise ValueError(f"Chunk {i} has no summary")
        for key in ("key_events", "patterns"):
            if not isinstance(value.get(key, []), list):
                value[key] = []
        value["profile_evidence"] = str(value.get("profile_evidence") or "")
        return value

    def _sort_analysis(self, day, moments):
        result = {key: {} for key in _INDEX_FIELDS}
        for item in moments:
            mid = str(item.get("id") or "")
            if not mid:
                continue
            result["time_nodes"].setdefault(day, []).append(mid)
            for key, (field, kind) in _INDEX_FIELDS.items():
                if key == "time_nodes":
                    continue
                vals = item.get(field) or []
                if kind == "scalar":
                    vals = [vals]
                for tag in vals:
                    if not isinstance(tag, str) or not tag.strip():
                        continue
                    tag = tag.strip()
                    valid = (_is_valid_person_tag(tag) if key == "people"
                             else _is_valid_index_tag(key, tag))
                    if valid:
                        ids = result[key].setdefault(tag, [])
                        if mid not in ids:
                            ids.append(mid)
        return result

    def _rollup(self, day, summaries, sort_data):
        old = self.memory.memory
        prior = {"layer4": old.get("layer4", {}),
                 "layer5": old.get("layer5", {}),
                 "layer6_summary": old.get("layer6", {}).get("summary", ""),
                 "profile": old.get("layer6", {}).get("profile", {})}
        tags = {k: sorted(v)[:80] for k, v in sort_data.items() if k in _COMBINABLE}
        prompt = f"""TASK: NIGHTLY_ROLLUP
Update recent and long-term memory using {day}'s summaries. Keep supported
prior long-term patterns. Layer 4 is the recent daily view; Layer 5 retains
durable events and patterns. Profile is extracted separately.
Return one valid JSON object only; do not add markdown or repeat source data.
Keep the response concise to avoid truncation:
- layer4.summary: at most 100 words; current_tasks: at most 5 short items;
  life_trajectory: at most 80 words.
- layer5.summary: at most 120 words; key_events: at most 8 short items;
  long_term_patterns: at most 5 short items.
- layer6_summary: at most 80 words; profile_evidence: at most 250 words.
- combine must contain the five maps shown below, with no more than 10 alias
  pairs total. Leave maps empty unless an alias is clearly supported.
Do not echo the tag lists or enumerate source memories.
{{
 "layer4":{{"summary":"...","current_tasks":["..."],"life_trajectory":"..."}},
 "layer5":{{"summary":"...","key_events":["..."],"long_term_patterns":"..."}},
 "layer6_summary":"brief stable profile overview",
 "profile_evidence":"brief evidence-grounded text for profile extraction",
 "combine":{{"people":{{}},"locations":{{}},"activity_events":{{}},"environments":{{}},"objects":{{}}}}
}}
Never merge distinct people or places.
Prior memory: {json.dumps(prior, ensure_ascii=False, separators=(',', ':'))}
Today's summaries: {json.dumps(summaries, ensure_ascii=False, separators=(',', ':'))}
Today's tags: {json.dumps(tags, ensure_ascii=False, separators=(',', ':'))}
"""
        data = self._json(self._call(prompt))
        if not isinstance(data.get("layer4"), dict) or not isinstance(data.get("layer5"), dict):
            raise ValueError("Rollup must contain layer4 and layer5 objects")
        return data

    @staticmethod
    def _compress_payload(data):
        l4, l5 = data.get("layer4") or {}, data.get("layer5") or {}
        return {
            "layer4": {
                "summary": str(l4.get("summary") or ""),
                "current_tasks": l4.get("current_tasks")
                    if isinstance(l4.get("current_tasks"), list) else [],
                "life_trajectory": str(l4.get("life_trajectory") or ""),
            },
            "layer5": {
                "summary": str(l5.get("summary") or ""),
                "key_events": l5.get("key_events")
                    if isinstance(l5.get("key_events"), list) else [],
                "long_term_patterns": str(l5.get("long_term_patterns") or ""),
            },
            "layer6": {"summary": str(data.get("layer6_summary") or "")},
        }

    def _combine_map(self, raw, sort_data):
        if not isinstance(raw, dict):
            return {}
        l7 = self.memory.memory.get("layer7", {})
        available = {k: set(l7.get(k, {})) | set(sort_data.get(k, {})) for k in _COMBINABLE}
        accepted = {}
        for key in _COMBINABLE:
            values = raw.get(key)
            if not isinstance(values, dict):
                continue
            pairs = {}
            for old, new in values.items():
                if not isinstance(old, str) or not isinstance(new, str):
                    continue
                if old == new or old not in available[key]:
                    continue
                valid = (_is_valid_person_tag(new) if key == "people"
                         else _is_valid_index_tag(key, new))
                if valid:
                    pairs[old] = new
            if pairs:
                accepted[key] = pairs
        return accepted

    def run(self, day=None, daytime_processor=None, force=False):
        """Run once per day. Identical source data is skipped unless forced."""
        with self._run_lock:
            day = day or self.memory.memory.get("metadata", {}).get("today") or _today_str()
            try:
                day = date.fromisoformat(day).isoformat()
            except ValueError as exc:
                raise ValueError("day must be YYYY-MM-DD") from exc
            translation_flushed = None
            if daytime_processor is not None:
                translation_flushed = daytime_processor.flush_translations(timeout=30)

            moments = self._moments(day)
            if not moments:
                return {"ok": True, "day": day, "skipped": "no_moments", "moment_count": 0}
            source = [self._brief(item) for item in moments]
            digest = hashlib.sha256(json.dumps(
                source, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            metadata = self.memory.memory.setdefault("metadata", {})
            previous = metadata.setdefault("nightly_processed_days", {}).get(day)
            if not force and previous and previous.get("source_digest") == digest:
                return {"ok": True, "day": day, "skipped": "unchanged",
                        "moment_count": len(moments), "source_digest": digest}

            file = getattr(self.memory, "memory_file", None)
            before = file.stat().st_size if file and file.exists() else 0
            start = time.perf_counter()
            chunks = self._chunks(moments)
            summaries = [self._summarize(day, i, len(chunks), chunk)
                         for i, chunk in enumerate(chunks, 1)]
            sort_data = self._sort_analysis(day, moments)
            rollup = self._rollup(day, summaries, sort_data)
            evidence = str(rollup.get("profile_evidence") or "").strip()
            if not evidence:
                evidence = "\n".join(x["profile_evidence"] for x in summaries
                                     if x.get("profile_evidence"))
            profile_prompt = "TASK: NIGHTLY_PROFILE_EXTRACTION\n" + \
                build_profile_extraction_prompt(evidence or json.dumps(summaries, ensure_ascii=False))
            profile_raw = self._call(profile_prompt)
            parsed = parse_profile_output(profile_raw)
            if parsed is None:
                raise ValueError(f"Profile response is invalid: {profile_raw[:240]}")
            profile, rejected = validate_profile(parsed)
            combine = self._combine_map(rollup.get("combine"), sort_data)

            # Finish all model calls and validation before persistent mutations.
            self.memory.sort(sort_data)
            # Call both APIs even when there are no aliases/new profile fields:
            # the operation audit should record the attempted maintenance action,
            # and update_profile also recomputes confidence decay / stale flags.
            self.memory.combine(combine)
            self.memory.compress(self._compress_payload(rollup))
            self.memory.update_profile(
                profile, moment_id=f"nightly:{day}",
                timestamp=datetime.now().astimezone().isoformat())

            completed = self.memory.memory["metadata"].setdefault(
                "nightly_processed_days", {})
            # combine() may canonicalize source tags and mark duplicates as
            # highlighted. Store the post-apply digest so the next invocation
            # recognizes those deterministic outputs instead of repeating a day.
            processed_source = [self._brief(item) for item in self._moments(day)]
            digest = hashlib.sha256(json.dumps(
                processed_source, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")).hexdigest()
            completed[day] = {
                "source_digest": digest,
                "moment_count": len(moments),
                "completed_at": datetime.now().astimezone().isoformat(),
                "profile_attributes": int(has_any_value(profile)),
            }
            cutoff = (date.fromisoformat(day) - timedelta(days=90)).isoformat()
            for old_day in list(completed):
                if not isinstance(old_day, str) or old_day < cutoff:
                    del completed[old_day]
            self.memory._save()
            elapsed = time.perf_counter() - start
            after = file.stat().st_size if file and file.exists() else 0
            return {
                "ok": True, "day": day, "moment_count": len(moments),
                "chunk_count": len(chunks),
                "indices_written": {k: len(v) for k, v in sort_data.items()},
                "aliases_combined": sum(len(v) for v in combine.values()),
                "profile_updated": has_any_value(profile),
                "profile_rejected": rejected,
                "translation_flushed": translation_flushed,
                "elapsed_seconds": round(elapsed, 3),
                "memory_file_bytes_before": before,
                "memory_file_bytes_after": after,
                "source_digest": digest,
            }


__all__ = ["NightlyProcessor"]
