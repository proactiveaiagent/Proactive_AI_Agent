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

    @staticmethod
    def _write_cache(path, cache):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _call(self, prompt, cache, cache_path, use_cache):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if use_cache and key in cache["responses"]:
            cache["hits"] = cache.get("hits", 0) + 1
            return cache["responses"][key], True
        if self.llm_call:
            value = self.llm_call(prompt)
            raw = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
        else:
            r = requests.post(f"{self.qwen_api_url}/consolidate",
                              json={"prompt": prompt}, timeout=self.timeout)
            if r.status_code != 200:
                raise RuntimeError(f"Qwen consolidate API {r.status_code}: {r.text[:300]}")
            raw = r.json().get("analysis")
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("Qwen consolidate API returned empty analysis")
        self._json(raw)
        cache["misses"] = cache.get("misses", 0) + 1
        cache["responses"][key] = raw
        self._write_cache(cache_path, cache)
        return raw, False

    @staticmethod
    def _json(raw):
        from profile_extractor import parse_json_safe
        value = parse_json_safe(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Model response is not a JSON object: {raw[:240]}")
        return value

    def _summarize(self, day, i, total, chunk, cache, cache_path, use_cache):
        prompt = f"""TASK: NIGHTLY_CHUNK_SUMMARY
Summarize this chronological slice of memories for {day}. Use only observations
provided; distinguish repeated patterns from one-off events; preserve uncertainty.
Return JSON only:
{{"summary":"short factual summary","key_events":["..."],"patterns":["..."],"profile_evidence":"evidence for stable profile extraction"}}
Slice {i}/{total}:
{json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}
"""
        raw, cache_hit = self._call(prompt, cache, cache_path, use_cache)
        value = self._json(raw)
        if not isinstance(value.get("summary"), str):
            raise ValueError(f"Chunk {i} has no summary")
        for key in ("key_events", "patterns"):
            if not isinstance(value.get(key, []), list):
                value[key] = []
        value["profile_evidence"] = str(value.get("profile_evidence") or "")
        return value, cache_hit

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

    def _rollup(self, day, summaries, sort_data, cache, cache_path, use_cache):
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
        raw, cache_hit = self._call(prompt, cache, cache_path, use_cache)
        data = self._json(raw)
        if not isinstance(data.get("layer4"), dict) or not isinstance(data.get("layer5"), dict):
            raise ValueError("Rollup must contain layer4 and layer5 objects")
        return data, cache_hit

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

    def run(self, day=None, daytime_processor=None, force=False, use_cache=True,
            progress_callback=None):
        """Run once per day, resuming completed model calls from a prompt cache."""
        cache = {"responses": {}, "hits": 0, "misses": 0}
        def report(stage, label, current=0, total=0, message=""):
            if progress_callback:
                progress_callback({"stage": stage, "label": label, "current": current,
                                   "total": total, "message": message,
                                   "cache_hits": cache["hits"], "model_calls": cache["misses"]})
        with self._run_lock:
            day = day or self.memory.memory.get("metadata", {}).get("today") or _today_str()
            try:
                day = date.fromisoformat(day).isoformat()
            except ValueError as exc:
                raise ValueError("day must be YYYY-MM-DD") from exc
            report("prepare", "准备夜间整理", message=f"日期 {day}")
            translation_flushed = None
            if daytime_processor is not None:
                report("translations", "等待白天翻译任务完成", message="最多等待 30 秒")
                translation_flushed = daytime_processor.flush_translations(timeout=30)

            moments = self._moments(day)
            if not moments:
                report("done", "夜间整理已跳过", 1, 1, "当天没有可整理的记忆")
                return {"ok": True, "day": day, "skipped": "no_moments",
                        "moment_count": 0, "chunk_count": 0, "profile_updated": False,
                        "cache_hits": 0, "model_calls": 0}
            source = [self._brief(item) for item in moments]
            digest = hashlib.sha256(json.dumps(
                source, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            cache_path = Path(self.memory.memory_dir) / "nightly_cache" / f"{day}-{digest}.json"
            if use_cache and cache_path.exists():
                try:
                    loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                    if loaded.get("source_digest") == digest and isinstance(loaded.get("responses"), dict):
                        cache["responses"] = loaded["responses"]
                except (OSError, ValueError, TypeError):
                    pass
            cache["source_digest"] = digest
            report("cache", "检查整理缓存", message=(
                f"已缓存 {len(cache['responses'])} 项；" + ("启用缓存" if use_cache else "本次不读缓存")))
            metadata = self.memory.memory.setdefault("metadata", {})
            previous = metadata.setdefault("nightly_processed_days", {}).get(day)
            if not force and previous and previous.get("source_digest") == digest:
                media_compaction = self.memory.compact_multimodal_assets()
                report("done", "夜间整理已跳过", 1, 1, "记忆内容未变化")
                return {"ok": True, "day": day, "skipped": "unchanged",
                        "moment_count": len(moments), "chunk_count": 0,
                        "profile_updated": False, "source_digest": digest,
                        "cache_hits": 0, "model_calls": 0,
                        "multimodal_compaction": media_compaction}

            file = getattr(self.memory, "memory_file", None)
            before = file.stat().st_size if file and file.exists() else 0
            start = time.perf_counter()
            chunks = self._chunks(moments)
            summaries = []
            for i, chunk in enumerate(chunks, 1):
                report("summarize", "分批归纳当天记忆", i - 1, len(chunks), f"正在处理第 {i}/{len(chunks)} 批")
                summary, hit = self._summarize(day, i, len(chunks), chunk, cache, cache_path, use_cache)
                summaries.append(summary)
                report("summarize", "分批归纳当天记忆", i, len(chunks),
                       f"第 {i}/{len(chunks)} 批完成（{'缓存命中' if hit else '模型生成'}）")
            report("sort", "建立六维分类索引", 0, 1, f"整理 {len(moments)} 条记忆")
            sort_data = self._sort_analysis(day, moments)
            report("sort", "建立六维分类索引", 1, 1, "六个索引维度已整理")
            report("rollup", "合并近期与长期摘要", 0, 1, "调用整理模型")
            rollup, rollup_hit = self._rollup(day, summaries, sort_data, cache, cache_path, use_cache)
            report("rollup", "合并近期与长期摘要", 1, 1,
                   "完成（缓存命中）" if rollup_hit else "模型生成完成")
            evidence = str(rollup.get("profile_evidence") or "").strip()
            if not evidence:
                evidence = "\n".join(x["profile_evidence"] for x in summaries
                                     if x.get("profile_evidence"))
            profile_prompt = "TASK: NIGHTLY_PROFILE_EXTRACTION\n" + \
                build_profile_extraction_prompt(evidence or json.dumps(summaries, ensure_ascii=False))
            report("profile", "抽取并校验长期画像", 0, 1, "分析稳定偏好和行为模式")
            profile_raw, profile_hit = self._call(profile_prompt, cache, cache_path, use_cache)
            parsed = parse_profile_output(profile_raw)
            if parsed is None:
                raise ValueError(f"Profile response is invalid: {profile_raw[:240]}")
            profile, rejected = validate_profile(parsed)
            combine = self._combine_map(rollup.get("combine"), sort_data)
            report("profile", "抽取并校验长期画像", 1, 1,
                   "校验通过（缓存命中）" if profile_hit else "模型生成并校验通过")

            # Finish all model calls and validation before persistent mutations.
            report("apply", "写入整理结果", 0, 4, "更新六维索引")
            self.memory.sort(sort_data)
            report("apply", "写入整理结果", 1, 4, "更新实体合并关系")
            # Call both APIs even when there are no aliases/new profile fields:
            # the operation audit should record the attempted maintenance action,
            # and update_profile also recomputes confidence decay / stale flags.
            self.memory.combine(combine)
            report("apply", "写入整理结果", 2, 4, "压缩近期与长期摘要")
            self.memory.compress(self._compress_payload(rollup))
            report("apply", "写入整理结果", 3, 4, "更新长期画像")
            self.memory.update_profile(
                profile, moment_id=f"nightly:{day}",
                timestamp=datetime.now().astimezone().isoformat())

            completed = self.memory.memory["metadata"].setdefault(
                "nightly_processed_days", {})
            # combine() may canonicalize source tags. Store the post-apply
            # digest so the next invocation recognizes those deterministic
            # outputs instead of repeating a day.
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
            report("apply", "写入整理结果", 4, 4, "保存整理结果")
            media_compaction = self.memory.compact_multimodal_assets()
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
                "cache_hits": cache["hits"],
                "model_calls": cache["misses"],
                "cache_enabled": bool(use_cache),
                "multimodal_compaction": media_compaction,
            }


__all__ = ["NightlyProcessor"]
