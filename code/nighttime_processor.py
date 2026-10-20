"""End-of-day consolidation for the two-stage memory pipeline.

Reads one day's Layer 7 archive, repairs six indices, summarizes bounded batches,
updates Layers 4/5, extracts a validated Layer 6 profile, and records an
idempotency digest. Dataset ground truth is never read here.
"""
from __future__ import annotations

import hashlib
import json
import os
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

DEFAULT_QWEN_API_URL = (
    os.environ.get("QWEN_API_URL", "").strip().rstrip("/")
    or next((item.strip().rstrip("/") for item in
             os.environ.get("QWEN_API_URLS", "").split(",") if item.strip()), "")
    or "http://127.0.0.1:8001"
)
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

    def _moments(self, day, all_accumulated=False):
        """Layer 7 is the source of truth; optionally consolidate the full accumulation pool."""
        result = {}
        archive = self.memory.memory.get("layer7", {}).get("moments", {})
        for mid, item in archive.items() if isinstance(archive, dict) else []:
            if isinstance(item, dict) and (all_accumulated or self._moment_day(item) == day):
                result[str(mid)] = item
        for layer in ("layer1", "layer2", "layer3"):
            for item in self.memory.memory.get(layer, []):
                if isinstance(item, dict) and (all_accumulated or self._moment_day(item) == day):
                    mid = str(item.get("id") or "")
                    if mid:
                        result.setdefault(mid, item)
        return sorted(result.values(), key=lambda m: (
            str(m.get("timestamp") or ""), str(m.get("id") or "")))

    @staticmethod
    def _brief(item):
        out = {"id": item.get("id", ""), "timestamp": item.get("timestamp", "")}
        provenance = item.get("source_provenance") or {}
        if provenance.get("source_video_id"):
            out["source_video_id"] = str(provenance["source_video_id"])[:300]
        if provenance.get("source_segment_id"):
            out["source_segment_id"] = str(provenance["source_segment_id"])[:240]
        for key, limit in (("scene", 500), ("user_action", 500),
                           ("interaction_details", 1200),
                           ("location", 200), ("activity", 200),
                           ("notes", 500), ("extra_notes", 500)):
            if item.get(key):
                out[key] = str(item[key])[:limit]
        if item.get("source_time"):
            out["source_time"] = item["source_time"]
        for key, limit in (("person_observations", 8), ("temporal_facts", 16)):
            if item.get(key):
                out[key] = item[key][:limit] if isinstance(item[key], list) else item[key]
        # Keep timestamped ASR in the consolidation input so explicit spoken
        # plans remain available to Layer 4/5. It is unattributed: never infer
        # which visible person spoke from transcript text alone.
        if item.get("source_evidence"):
            out["source_evidence"] = str(item["source_evidence"])[:4000]
        # Roll up observed entities, not downstream inference outputs.
        for key in ("people", "environments", "objects"):
            if item.get(key):
                val = item[key]
                out[key] = val[:12] if isinstance(val, list) else val
        out["confirmed"] = bool((item.get("feedback") or {}).get("confirmed"))
        out["highlighted"] = bool(item.get("highlighted"))
        return out

    @staticmethod
    def _stated_plans(moments):
        """Carry explicitly stated plans into L4 without converting them to advice."""
        plans, seen = [], set()
        for moment in moments:
            source_time = moment.get("source_time") or {}
            for fact in moment.get("temporal_facts") or []:
                if (not isinstance(fact, dict)
                        or str(fact.get("predicate") or "").casefold() not in {"plan", "stated_plan"}
                        or fact.get("status") != "stated_plan"):
                    continue
                plan = {
                    key: fact[key] for key in (
                        "subject_type", "subject", "predicate", "status", "value", "actor",
                        "time_scope", "time_order", "modality", "evidence", "confidence",
                        "evidence_ref") if fact.get(key) not in (None, "")
                }
                plan["source_moment_id"] = str(moment.get("id") or "")
                plan["source_time"] = source_time
                signature = json.dumps(plan, ensure_ascii=False, sort_keys=True)
                if signature not in seen:
                    seen.add(signature)
                    plans.append(plan)
        return plans[:80]

    def _chunks(self, moments):
        chunks, current, size = [], [], 2
        current_source = None
        for item in moments:
            brief = self._brief(item)
            source = str(brief.get("source_video_id") or "")
            n = len(json.dumps(brief, ensure_ascii=False))
            if current and (source != current_source or size + n > self.chunk_chars):
                chunks.append(current)
                current, size = [], 2
            current_source = source
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
Keep explicit person appearance/age evidence separate from identity, preserve
plans as plans, and retain object state/location changes with their before/after
details and event order. source_evidence is timestamped automatic speech
transcription from the source video; speakers are not identified, and ASR may be
wrong. Do not attribute speech to a visible person. Preserve only explicit plans
and never turn a plan into a completed action or guess missing ages, dates,
identities, or locations.
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

    def _sort_analysis(self, day, moments, all_accumulated=False):
        result = {key: {} for key in _INDEX_FIELDS}
        for item in moments:
            mid = str(item.get("id") or "")
            if not mid:
                continue
            time_key = (self._moment_day(item) or day) if all_accumulated else day
            result["time_nodes"].setdefault(time_key, []).append(mid)
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
        old_l4 = old.get("layer4", {}) or {}
        old_l5 = old.get("layer5", {}) or {}
        old_l6 = old.get("layer6", {}) or {}
        prior = {
            "layer4_summary": str(old_l4.get("summary") or "")[:1200],
            "life_trajectory": str(old_l4.get("life_trajectory") or "")[:800],
            "layer5_summary": str(old_l5.get("summary") or "")[:1400],
            "layer5_key_events": (old_l5.get("key_events") or [])[:8],
            "layer5_patterns": str(old_l5.get("long_term_patterns") or "")[:800],
            "layer6_summary": str(old_l6.get("summary") or "")[:800],
        }
        # Keep the roll-up context bounded. Source moments and the six indices
        # are already stored separately; sending all tags and the full profile
        # caused the model to expand alias maps until its JSON was truncated.
        compact_summaries = []
        for item in summaries:
            compact_summaries.append({
                "summary": str(item.get("summary") or "")[:1000],
                "key_events": [str(x)[:220] for x in (item.get("key_events") or [])[:5]],
                "patterns": [str(x)[:220] for x in (item.get("patterns") or [])[:4]],
                "profile_evidence": str(item.get("profile_evidence") or "")[:800],
                "source_video_ids": item.get("source_video_ids") or [],
                "source_moment_ids": item.get("source_moment_ids") or [],
            })
        prompt = f"""TASK: NIGHTLY_ROLLUP
Update factual recent and long-term memory for {day} from the compact summaries.
Preserve supported prior long-term information. Do not infer identity or merge
people/places; entity aliases are handled by the evidence-backed entity layer.
Each summary carries source_video_ids and source_moment_ids. Keep event facts attached
to those sources; do not combine unrelated videos into one event or shared life history.
Label source-specific key events with their source video when more than one source exists.
Exclude diagnoses, recommendations, inferred needs, reminders, and actions.
Return exactly one concise JSON object, no markdown or source-memory dump:
{{
 "layer4_summary":"at most 90 words",
 "layer5_summary":"at most 100 words",
 "key_events":["at most 6 short evidence-grounded events"],
 "long_term_patterns":["at most 4 repeated patterns; [] if unsupported"],
 "layer6_summary":"at most 60 words; preserve prior if no new stable evidence",
 "profile_evidence":"at most 120 words for separate profile validation"
}}
Prior: {json.dumps(prior, ensure_ascii=False, separators=(',', ':'))}
Today's chunk summaries: {json.dumps(compact_summaries, ensure_ascii=False, separators=(',', ':'))}
"""
        try:
            raw, cache_hit = self._call(prompt, cache, cache_path, use_cache)
            output = self._json(raw)
            if not isinstance(output, dict):
                raise ValueError("roll-up output is not an object")
            data = {
                "layer4": {
                    "summary": str(output.get("layer4_summary") or "").strip(),
                    "current_tasks": [],
                    "life_trajectory": prior["life_trajectory"],
                },
                "layer5": {
                    "summary": str(output.get("layer5_summary") or "").strip(),
                    "key_events": output.get("key_events") if isinstance(output.get("key_events"), list) else [],
                    "long_term_patterns": "; ".join(str(x) for x in
                        (output.get("long_term_patterns") or [])[:4] if str(x).strip()),
                },
                "layer6_summary": str(output.get("layer6_summary") or prior["layer6_summary"]).strip(),
                "profile_evidence": str(output.get("profile_evidence") or "").strip(),
                "combine": {key: {} for key in _COMBINABLE},
            }
            if not data["layer4"]["summary"] or not data["layer5"]["summary"]:
                raise ValueError("roll-up omitted required recent/long-term summary")
            return data, cache_hit
        except ValueError as exc:
            # The chunk summaries have already passed JSON validation and remain
            # cached. Preserve their evidence and complete the night run without
            # trusting malformed/truncated roll-up output.
            def words(text, limit):
                return " ".join(str(text or "").split()[:limit])
            def unique(values, limit):
                result, seen = [], set()
                for value in values:
                    value = " ".join(str(value or "").split())
                    key = value.casefold()
                    if value and key not in seen:
                        seen.add(key)
                        result.append(value)
                    if len(result) >= limit:
                        break
                return result
            current_summary = words(" ".join(x.get("summary", "") for x in compact_summaries), 90)
            key_events = unique([event for x in compact_summaries
                                 for event in x.get("key_events", [])], 6)
            patterns = unique([item for x in compact_summaries
                               for item in x.get("patterns", [])], 4)
            profile_evidence = words(" ".join(x.get("profile_evidence", "")
                                                for x in compact_summaries), 120)
            prior_summary = words(prior["layer5_summary"], 50)
            new_events = words(" ".join(key_events), 50)
            data = {
                "layer4": {"summary": current_summary or words(prior["layer4_summary"], 90),
                           "current_tasks": [], "life_trajectory": prior["life_trajectory"]},
                "layer5": {"summary": words((prior_summary + " " + new_events).strip(), 100)
                           or current_summary,
                           "key_events": key_events,
                           "long_term_patterns": "; ".join(patterns)},
                "layer6_summary": prior["layer6_summary"],
                "profile_evidence": profile_evidence,
                "combine": {key: {} for key in _COMBINABLE},
                "_rollup_fallback": str(exc)[:240],
            }
            if not data["layer4"]["summary"] or not data["layer5"]["summary"]:
                raise ValueError("roll-up invalid and no validated summary fallback exists") from exc
            return data, False

    @staticmethod
    def _compress_payload(data, source_moment_ids, coverage, moments=None):
        l4, l5 = data.get("layer4") or {}, data.get("layer5") or {}
        source_moment_ids = list(dict.fromkeys(str(x) for x in source_moment_ids if x))
        return {
            "layer4": {
                "summary": str(l4.get("summary") or ""),
                "current_tasks": l4.get("current_tasks")
                    if isinstance(l4.get("current_tasks"), list) else [],
                "stated_plans": NightlyProcessor._stated_plans(moments or []),
                "life_trajectory": str(l4.get("life_trajectory") or ""),
                "source_moment_ids": source_moment_ids,
                "source_link_coverage": "complete",
            },
            "layer5": {
                "summary": str(l5.get("summary") or ""),
                "key_events": l5.get("key_events")
                    if isinstance(l5.get("key_events"), list) else [],
                "long_term_patterns": str(l5.get("long_term_patterns") or ""),
                "source_moment_ids": source_moment_ids,
                "source_link_coverage": coverage["layer5"],
            },
            "layer6": {
                "summary": str(data.get("layer6_summary") or ""),
                "source_moment_ids": source_moment_ids,
                "source_link_coverage": coverage["layer6"],
            },
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
            progress_callback=None, all_accumulated=False):
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

            moments = self._moments(day, all_accumulated=all_accumulated)
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
                summary["source_video_ids"] = sorted({
                    str(item.get("source_video_id") or "")
                    for item in chunk if item.get("source_video_id")
                })
                summary["source_moment_ids"] = [
                    str(item.get("id") or "") for item in chunk if item.get("id")
                ]
                summaries.append(summary)
                report("summarize", "分批归纳当天记忆", i, len(chunks),
                       f"第 {i}/{len(chunks)} 批完成（{'缓存命中' if hit else '模型生成'}）")
            report("sort", "建立六维分类索引", 0, 1, f"整理 {len(moments)} 条记忆")
            sort_data = self._sort_analysis(
                day, moments, all_accumulated=all_accumulated
            )
            report("sort", "建立六维分类索引", 1, 1, "六个索引维度已整理")
            report("rollup", "合并近期与长期摘要", 0, 1, "调用整理模型")
            rollup, rollup_hit = self._rollup(day, summaries, sort_data, cache, cache_path, use_cache)
            report("rollup", "合并近期与长期摘要", 1, 1,
                   ("模型格式无效，已基于通过校验的分块摘要安全回退"
                    if rollup.get("_rollup_fallback") else
                    ("完成（缓存命中）" if rollup_hit else "模型生成完成")))
            source_videos = {
                str(((item.get("source_provenance") or {}).get("source_video_id") or ""))
                for item in moments
                if ((item.get("source_provenance") or {}).get("source_video_id"))
            }
            mixed_video_scope = len(source_videos) > 1
            if mixed_video_scope:
                profile, rejected, profile_hit = {}, ["本次整理包含多个源视频，未将其合并为单一用户画像"], False
                report("profile", "跳过跨视频用户画像", 1, 1,
                       "保留事件归纳；不同源视频不直接合并成同一用户画像")
            else:
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
                report("profile", "抽取并校验长期画像", 1, 1,
                       "校验通过（缓存命中）" if profile_hit else "模型生成并校验通过")
            combine = self._combine_map(rollup.get("combine"), sort_data)

            # Finish all model calls and validation before persistent mutations.
            report("apply", "写入整理结果", 0, 4, "更新六维索引")
            self.memory.sort(sort_data)
            report("apply", "写入整理结果", 1, 4, "更新实体合并关系")
            # Call both APIs even when there are no aliases/new profile fields:
            # the operation audit should record the attempted maintenance action,
            # and update_profile also recomputes confidence decay / stale flags.
            self.memory.combine(combine)
            report("apply", "写入整理结果", 2, 4, "压缩近期与长期摘要")
            source_moment_ids = [str(item.get("id")) for item in moments if item.get("id")]
            def has_content(value):
                if isinstance(value, dict):
                    if "value" in value:
                        return bool(value.get("value"))
                    return any(has_content(item) for item in value.values())
                if isinstance(value, (list, tuple)):
                    return any(has_content(item) for item in value)
                return bool(value)

            previous_l5 = self.memory.memory.get("layer5", {})
            previous_l6 = self.memory.memory.get("layer6", {})
            has_previous_l5 = any(has_content(previous_l5.get(key))
                                   for key in ("summary", "key_events", "long_term_patterns"))
            has_previous_l6 = (has_content(previous_l6.get("summary"))
                               or has_content(previous_l6.get("profile")))
            coverage = {
                "layer5": ("complete" if not has_previous_l5
                           or previous_l5.get("source_link_coverage") == "complete"
                           else "partial"),
                "layer6": ("complete" if not has_previous_l6
                           or previous_l6.get("source_link_coverage") == "complete"
                           else "partial"),
            }
            self.memory.compress(self._compress_payload(
                rollup, source_moment_ids, coverage, moments=moments))
            report("apply", "写入整理结果", 3, 4, "更新长期画像")
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
