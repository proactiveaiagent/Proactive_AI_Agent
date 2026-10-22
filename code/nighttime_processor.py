"""End-of-day consolidation for the two-stage memory pipeline.

Reads one day's Layer 7 archive, repairs six indices, summarizes bounded batches,
updates Layers 4/5, extracts a validated Layer 6 profile, and records an
idempotency digest. Dataset ground truth is never read here.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
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
        # Include compact asset identities in the nightly digest. A changed crop
        # or newly available Qwen input frame must invalidate stale curation.
        media = item.get("multimodal") or {}
        refs = list(media.get("visual_frames") or [])
        if isinstance(media.get("visual"), dict):
            refs.append(media["visual"])
        for group_key in ("person_visuals", "object_visuals", "entity_visuals"):
            group = media.get(group_key) or {}
            for visual in group.get("items") or []:
                if isinstance(visual, dict):
                    refs.extend(visual.get("frame_refs") or [])
            if group_key == "entity_visuals":
                refs.extend(group.get("assets") or [])
        visual_evidence = []
        seen_refs = set()
        for ref in refs:
            if not isinstance(ref, dict) or not ref.get("asset"):
                continue
            stamp = (str(ref.get("asset")), str(ref.get("visual_role") or ""),
                     str(ref.get("at_s", ref.get("frame_time_s", ""))))
            if stamp in seen_refs:
                continue
            seen_refs.add(stamp)
            visual_evidence.append({
                "asset": stamp[0], "role": stamp[1], "at_s": stamp[2],
                "identity_verified": ref.get("identity_verified") is True,
                "quality_tier": str(ref.get("quality_tier") or ""),
            })
        if visual_evidence:
            out["visual_evidence_refs"] = visual_evidence[:24]
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

    def _rollup(self, day, summaries, sort_data, cache, cache_path, use_cache, moments=None, progress=None):
        old = self.memory.memory
        old_l4, old_l5, old_l6 = (old.get(key, {}) or {} for key in ("layer4", "layer5", "layer6"))
        prior = {
            "layer4_summary": str(old_l4.get("summary") or "")[:1200],
            "life_trajectory": str(old_l4.get("life_trajectory") or "")[:800],
            "layer5_summary": str(old_l5.get("summary") or "")[:1600],
            "layer5_key_events": [],
            "layer5_patterns": str(old_l5.get("long_term_patterns") or "")[:1000],
            "layer6_summary": str(old_l6.get("summary") or "")[:800],
        }
        # Keep prior event citations but remove verbose generated prose before
        # passing them back to the consolidator.
        prior_events = []
        for raw_event in (old_l5.get("key_events") or [])[:40]:
            text = str(raw_event or "").strip()
            if not text:
                continue
            cited = list(dict.fromkeys(re.findall(
                r"\[(?:moment|moments):([^\]]+)\]", text)))
            cited = [mid for value in cited for mid in value.split(",") if mid]
            summary = re.sub(r"^(?:\[(?:source|moment|moments):[^\]]+\]\s*)+", "", text)
            prior_events.append({"summary": summary[:520],
                                 "source_moment_ids": cited[:12]})
        prior["layer5_key_events"] = prior_events
        compact_summaries = [{
            "summary": str(item.get("summary") or "")[:1000],
            "key_events": [str(x)[:220] for x in (item.get("key_events") or [])[:6]],
            "patterns": [str(x)[:220] for x in (item.get("patterns") or [])[:5]],
            "profile_evidence": str(item.get("profile_evidence") or "")[:800],
            "source_video_ids": item.get("source_video_ids") or [],
            "source_moment_ids": item.get("source_moment_ids") or [],
        } for item in summaries]
        from nightly_curation import NightlyCurator
        # First consolidate each source family independently. This keeps the
        # visual/person evidence together where clip continuity exists, while
        # bounding the Qwen context even when a full custom dataset is large.
        families = {}
        for moment in moments or []:
            provenance = moment.get("source_provenance") or {}
            video = str(provenance.get("source_video_id") or "")
            family = video.rsplit("/", 1)[0] if "/" in video else video
            families.setdefault(family or "unscoped", []).append(moment)
        family_rows = []
        group_identity = []
        group_moment_highlights = []
        group_image_highlights = []
        summaries_by_moment = {}
        for summary in summaries:
            for mid in summary.get("source_moment_ids") or []:
                summaries_by_moment[str(mid)] = summary

        for group_index, (family, family_moments) in enumerate(sorted(families.items()), 1):
            family_ids = {str(item.get("id") or "") for item in family_moments}
            evidence = NightlyCurator.evidence_bundle(family_moments)
            compact_rows = []
            for row in evidence["moments"]:
                compact = dict(row)
                compact["scene"] = str(compact.get("scene") or "")[:360]
                compact["interaction_details"] = str(compact.get("interaction_details") or "")[:240]
                compact["source_evidence"] = str(compact.get("source_evidence") or "")[:650]
                compact["speech_and_person_facts"] = [
                    {key: (str(value)[:200] if isinstance(value, str) else value)
                     for key, value in fact.items()}
                    for fact in (compact.get("speech_and_person_facts") or [])[:8]
                ]
                compact["person_observations"] = [
                    {key: (str(value)[:180] if isinstance(value, str) else value)
                     for key, value in person.items()}
                    for person in (compact.get("person_observations") or [])[:6]
                ]
                compact_rows.append(compact)
            # The archive retains every frame. For curation, expose a bounded
            # representative set: verified entity crops first, then all six
            # Qwen inputs (when present).
            image_rows = []
            for mid in sorted(family_ids):
                candidates = [item for item in evidence["image_candidates"]
                              if item["ref_id"].startswith(mid + "#image:")]
                candidates.sort(key=lambda item: (
                    0 if item.get("identity_verified") else 1,
                    0 if item.get("visual_role") in {"face_crop", "person_crop", "object_crop"} else 1,
                    float(item.get("at_s") or 0), item["ref_id"]))
                image_rows.extend(candidates[:12])
            chunk_rows = []
            for mid in sorted(family_ids):
                summary = summaries_by_moment.get(mid) or {}
                if summary:
                    chunk_rows.append({
                        "moment_id": mid,
                        "summary": str(summary.get("summary") or "")[:260],
                        "key_events": [str(value)[:180] for value in
                                       (summary.get("key_events") or [])[:3]],
                        "profile_evidence": str(summary.get("profile_evidence") or "")[:220],
                    })
            group_prompt = f"""TASK: NIGHTLY_SOURCE_FAMILY_CURATION
Consolidate this related source-video family, not each clip independently.
Use the evidence below to connect recurring people across its clips, preserve
distinct events with citations, compress repeated descriptions, and select only
the most useful source memories and representative images.
Resolve a visible person to a spoken family name only when introduction/kinship
speech is tied to that visible person or matching person evidence supports it.
Never merge clip-local labels by gender, clothing, or label alone. Preserve
uncertainty. Do not attribute undiarized speech to a face. Keep Qwen input frames
and source moments as audit evidence; this stage only marks curation importance.

Return JSON only:
{{"summary":"one concise family-level synthesis, <=80 words",
 "key_events":[{{"summary":"concise event","source_moment_ids":["valid IDs"]}}],
 "profile_evidence":"stable repeated facts with source IDs, <=100 words",
 "identity_resolutions":[{{"moment_id":"target ID","observed_entity":"current label",
   "canonical_entity":"supported name","confidence":0.0,"evidence_moment_ids":["IDs"],
   "reason":"evidence"}}],
 "highlight_moments":[{{"moment_id":"...","score":0.0,"reason":"...","evidence_moment_ids":["..."]}}],
 "highlight_images":[{{"ref_id":"exact image candidate ID","score":0.0,"reason":"..."}}]}}
Use only IDs in this family. Every key event needs a source ID. Keep at most 8
events, 4 memory highlights, and 6 image highlights. Return representative
person/relationship/object evidence, not every frame.
Family: {family}
Chunk summaries: {json.dumps(chunk_rows, ensure_ascii=False, separators=(',', ':'))}
Source evidence: {json.dumps(compact_rows, ensure_ascii=False, separators=(',', ':'))}
Image candidates: {json.dumps(image_rows, ensure_ascii=False, separators=(',', ':'))}
"""
            if progress:
                progress(group_index - 1, len(families) + 1,
                         f"正在整理来源组 {group_index}/{len(families)}：{family}")
            raw_group, _group_hit = self._call(group_prompt, cache, cache_path, use_cache)
            group = self._json(raw_group)
            valid_images = {item["ref_id"] for item in image_rows}
            local_events = []
            for event in group.get("key_events") or []:
                if not isinstance(event, dict):
                    continue
                refs = list(dict.fromkeys(str(mid) for mid in event.get("source_moment_ids") or []
                                          if str(mid) in family_ids))
                text = " ".join(str(event.get("summary") or "").split())[:260]
                if refs and text:
                    local_events.append({"summary": text, "source_moment_ids": refs})
            local_identity = [item for item in (group.get("identity_resolutions") or [])
                              if isinstance(item, dict)
                              and str(item.get("moment_id") or "") in family_ids]
            local_moments = [item for item in (group.get("highlight_moments") or [])
                             if isinstance(item, dict)
                             and str(item.get("moment_id") or "") in family_ids]
            local_images = [item for item in (group.get("highlight_images") or [])
                            if isinstance(item, dict)
                            and str(item.get("ref_id") or "") in valid_images]
            group_identity.extend(local_identity[:12])
            group_moment_highlights.extend(local_moments[:4])
            group_image_highlights.extend(local_images[:6])
            family_rows.append({
                "family": family,
                "source_moment_ids": sorted(family_ids),
                "summary": " ".join(str(group.get("summary") or "").split())[:420],
                "key_events": local_events[:8],
                "profile_evidence": str(group.get("profile_evidence") or "")[:500],
                "identity_resolutions": local_identity[:12],
                "highlight_moments": local_moments[:4],
                "highlight_images": local_images[:6],
            })
            if progress:
                progress(group_index, len(families) + 1,
                         f"已完成来源组 {group_index}/{len(families)}：{family}")

        prompt = f"""TASK: NIGHTLY_ROLLUP
You are performing actual cross-memory consolidation for {day}, not writing one
summary per clip. Integrate the new evidence with retained prior memory:
compress repeated descriptions, merge recurring people/relations and stable
patterns, remove low-value duplicate items from the compact Layer 5 event list,
and identify a small number of durable, high-utility memories and source images.
Layer 7 source moments remain the audit trail and must not be deleted by this
summary step.

IDENTITY RESOLUTION:
- Resolve clip-local person labels only when the evidence supports the same
  visible person. A spoken introduction / kinship alias linked to a visual person,
  a validated identity match, or a consistent sequence with matching appearance
  and scene can support a link. Cite the source moment IDs.
- Do not join people just because both are called woman 1/person 1, have the same
  gender/clothing, or appear in the same broad category. Keep uncertain links
  unresolved. Never mix two visible people from a group image.
- Keep distinct events and source citations even when the same person appears
  across several videos. Never attribute undiarized speech to a face without
  explicit visual/person evidence.
- Layer 6 profile evidence must come from repeated or explicit stable facts.
  Cross-clip event continuity is allowed; unsupported ownership, location,
  identity, or intent is not.
- Global index alias maps are unsafe for clip-local labels; leave combine.people
  empty. Per-moment identity_resolutions are the safe way to link those labels.
- A highlight is a curation/importance mark, not a claim that a user confirmed
  the fact. Choose only standout, reusable memories/images (score >= 0.82);
  do not mark every item.

Return JSON only:
{{
 "layer4_summary":"current/recent compressed summary, at most 100 words",
 "layer5_summary":"compact durable summary across new and retained evidence, at most 120 words",
 "key_events":[{{"summary":"concise event","source_moment_ids":["valid source IDs"]}}],
 "replace_key_events":true,
 "retained_source_moment_ids":["IDs supporting the retained Layer 5 summary/events/patterns"],
 "long_term_patterns":["only repeated patterns with source support; at most 6"],
 "layer6_summary":"at most 60 words; preserve prior if no new stable evidence",
 "profile_evidence":"at most 180 words for separate validated profile extraction",
 "combine":{{"people":{{}},"locations":{{}},"activity_events":{{}},"environments":{{}},"objects":{{}}}},
 "identity_resolutions":[{{"moment_id":"target moment","observed_entity":"clip-local label",
   "canonical_entity":"stable identity","confidence":0.0,
   "evidence_moment_ids":["source moments"],"reason":"short evidence explanation"}}],
 "highlight_moments":[{{"moment_id":"...","score":0.0,"reason":"why this is unusually reusable",
   "evidence_moment_ids":["..."]}}],
 "highlight_images":[{{"ref_id":"exact image ref_id from candidates","score":0.0,
   "reason":"why this is the representative frame"}}]
}}
For key_events, retain at most 24 after removing semantic duplicates and low-value
one-off details. Every retained event must include its source_moment_ids. Preserve
important relationship changes, explicit plans, object-location transitions,
safety/health events, and recurring meaningful interactions. Do not combine
different events into a single event. Patterns and Layer 5 summary may consolidate
across source videos, but remain source-grounded. Retain at most 8 highlighted
memories and 12 highlighted images; usually fewer. Select image refs only from the
provided list. Output valid source IDs and image IDs exactly as given.

Prior compact memory: {json.dumps(prior, ensure_ascii=False, separators=(',', ':'))}
Source-family consolidations: {json.dumps(family_rows, ensure_ascii=False, separators=(',', ':'))}
Only select global highlight candidates from these source-family candidates. Identity
resolutions from source-family evidence will be preserved with their citations.
"""
        try:
            raw, cache_hit = self._call(prompt, cache, cache_path, use_cache)
            output = self._json(raw)
            if not isinstance(output, dict):
                raise ValueError("roll-up output is not an object")
            # The family pass carries the detailed evidence-backed identity
            # decisions and candidate marks; the global pass only chooses a
            # compact durable subset and may add equally cited links.
            output["identity_resolutions"] = (
                group_identity + (output.get("identity_resolutions")
                                  if isinstance(output.get("identity_resolutions"), list) else []))
            if not output.get("highlight_moments"):
                output["highlight_moments"] = group_moment_highlights
            if not output.get("highlight_images"):
                output["highlight_images"] = group_image_highlights
            if not output.get("profile_evidence"):
                output["profile_evidence"] = " ".join(
                    row["profile_evidence"] for row in family_rows if row.get("profile_evidence"))[:1800]

            l4 = output.get("layer4") if isinstance(output.get("layer4"), dict) else {}
            l5 = output.get("layer5") if isinstance(output.get("layer5"), dict) else {}
            archive = self.memory.memory.get("layer7", {}).get("moments", {})
            valid_ids = {str(mid) for mid in archive}
            valid_ids.update(str(item.get("id")) for item in moments or [] if item.get("id"))

            def event_text(event):
                if isinstance(event, dict):
                    summary = " ".join(str(event.get("summary") or "").split())[:360]
                    refs = list(dict.fromkeys(str(mid) for mid in
                        (event.get("source_moment_ids") or []) if str(mid) in valid_ids))
                else:
                    summary = " ".join(str(event or "").split())[:520]
                    raw_refs = re.findall(r"\[(?:moment|moments):([^\]]+)\]", summary)
                    refs = list(dict.fromkeys(mid for value in raw_refs
                                              for mid in value.split(",") if mid in valid_ids))
                    summary = re.sub(r"^(?:\[(?:source|moment|moments):[^\]]+\]\s*)+", "", summary)
                # Keep only compact events traceable to source moments.
                if not summary or not refs:
                    return ""
                return "[moments:" + ",".join(refs) + "] " + summary

            raw_events = output.get("key_events", l5.get("key_events", []))
            if not isinstance(raw_events, list):
                raw_events = []
            events = [value for value in (event_text(item) for item in raw_events) if value][:24]
            raw_patterns = output.get("long_term_patterns", l5.get("long_term_patterns", []))
            if isinstance(raw_patterns, str):
                patterns = [x.strip() for x in raw_patterns.split(";") if x.strip()][:6]
            elif isinstance(raw_patterns, list):
                patterns = [str(x).strip()[:240] for x in raw_patterns[:6] if str(x).strip()]
            else:
                patterns = []
            summary4 = str(output.get("layer4_summary") or l4.get("summary") or "").strip()
            summary5 = str(output.get("layer5_summary") or l5.get("summary") or "").strip()
            data = {
                "layer4": {
                    "summary": summary4,
                    "current_tasks": output.get("current_tasks", l4.get("current_tasks", []))
                        if isinstance(output.get("current_tasks", l4.get("current_tasks", [])), list) else [],
                    "life_trajectory": str(output.get("life_trajectory") or l4.get("life_trajectory")
                                            or prior["life_trajectory"]),
                },
                "layer5": {
                    "summary": summary5, "key_events": events,
                    "long_term_patterns": "; ".join(patterns),
                },
                "layer6_summary": str(output.get("layer6_summary") or prior["layer6_summary"]).strip(),
                "profile_evidence": str(output.get("profile_evidence") or "").strip(),
                "combine": output.get("combine") if isinstance(output.get("combine"), dict) else {},
                "identity_resolutions": output.get("identity_resolutions")
                    if isinstance(output.get("identity_resolutions"), list) else [],
                "highlight_moments": output.get("highlight_moments")
                    if isinstance(output.get("highlight_moments"), list) else [],
                "highlight_images": output.get("highlight_images")
                    if isinstance(output.get("highlight_images"), list) else [],
                "_replace_key_events": (output.get("replace_key_events") is True and bool(events)),
                "_retained_source_moment_ids": list(dict.fromkeys(
                    [str(mid) for mid in (output.get("retained_source_moment_ids") or [])
                     if str(mid) in valid_ids]
                    + [mid for event in events
                       for value in re.findall(r"\[moments:([^\]]+)\]", event)
                       for mid in value.split(",") if mid in valid_ids])),
            }
            if not data["layer4"]["summary"] or not data["layer5"]["summary"]:
                raise ValueError("roll-up omitted required recent/long-term summary")
            return data, cache_hit
        except ValueError as exc:
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
            current_summary = words(" ".join(x.get("summary", "") for x in compact_summaries), 100)
            key_events = unique([event for x in compact_summaries
                                 for event in x.get("key_events", [])], 24)
            patterns = unique([item for x in compact_summaries
                               for item in x.get("patterns", [])], 6)
            profile_evidence = words(" ".join(x.get("profile_evidence", "")
                                                for x in compact_summaries), 180)
            prior_summary = words(prior["layer5_summary"], 60)
            data = {
                "layer4": {"summary": current_summary or words(prior["layer4_summary"], 100),
                           "current_tasks": [], "life_trajectory": prior["life_trajectory"]},
                "layer5": {"summary": words((prior_summary + " " + " ".join(key_events)).strip(), 120)
                           or current_summary, "key_events": key_events,
                           "long_term_patterns": "; ".join(patterns)},
                "layer6_summary": prior["layer6_summary"], "profile_evidence": profile_evidence,
                "combine": {}, "identity_resolutions": [], "highlight_moments": [],
                "highlight_images": [], "_replace_key_events": False,
                "_retained_source_moment_ids": [],
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
            # Person aliases are source-scoped; only per-moment, cited identity
            # resolutions may connect them across clips.
            if key == "people":
                continue
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
            report("rollup", "跨片段压缩与合并", 0, 1,
                   "同时核对来源片段、人物关系、重复事件和可保留的重点")
            rollup, rollup_hit = self._rollup(
                day, summaries, sort_data, cache, cache_path, use_cache, moments=moments,
                progress=lambda current, total, message: report(
                    "rollup", "跨片段压缩与合并", current, total, message)
            )
            report("rollup", "跨片段压缩与合并", 1, 1,
                   ("模型格式无效，已基于有效分块摘要回退"
                    if rollup.get("_rollup_fallback") else
                    ("完成（缓存命中）" if rollup_hit else "完成")))

            evidence = str(rollup.get("profile_evidence") or "").strip()
            if not evidence:
                evidence = "\n".join(x["profile_evidence"] for x in summaries
                                     if x.get("profile_evidence"))
            profile_prompt = "TASK: NIGHTLY_PROFILE_EXTRACTION\n" + build_profile_extraction_prompt(
                evidence or json.dumps(summaries, ensure_ascii=False))
            report("profile", "抽取并校验长期画像", 0, 1, "合并当前批次的稳定事实证据")
            profile_raw, profile_hit = self._call(profile_prompt, cache, cache_path, use_cache)
            parsed = parse_profile_output(profile_raw)
            if parsed is None:
                raise ValueError(f"Profile response is invalid: {profile_raw[:240]}")
            profile, rejected = validate_profile(parsed)
            report("profile", "抽取并校验长期画像", 1, 1,
                   "校验通过（缓存命中）" if profile_hit else "模型生成并校验通过")

            from nightly_curation import NightlyCurator
            curator = NightlyCurator(self.memory)
            curation = curator.validate(rollup, moments)
            report("curation", "核验跨片段人物关联", 0,
                   len(curation["identity_resolutions"]), "只应用带来源证据的身份关联")
            identity_rewrites = curator.apply_identity_resolutions(
                curation["identity_resolutions"])
            curator.remove_old_index_memberships(identity_rewrites)
            # Identity labels have changed on the source moments; rebuild the
            # current six-way index from those canonicalized records.
            sort_data = self._sort_analysis(
                day, moments, all_accumulated=all_accumulated
            )
            report("curation", "核验跨片段人物关联", len(identity_rewrites),
                   len(identity_rewrites), "来源片段与人物图像保持原样可回查")
            combine = self._combine_map(rollup.get("combine"), sort_data)

            # Finish all model calls and validation before persistent mutations.
            report("apply", "写入整理结果", 0, 6, "更新六维索引")
            self.memory.sort(sort_data)
            report("apply", "写入整理结果", 1, 6, "更新可安全合并的地点、活动和物品标签")
            # Call both APIs even when there are no aliases/new profile fields:
            # the operation audit should record the attempted maintenance action,
            # and update_profile also recomputes confidence decay / stale flags.
            self.memory.combine(combine)
            duplicate_pairs = curator.exact_duplicate_pairs(moments)
            merged_duplicate_count = 0
            for source_id, target_id in duplicate_pairs:
                result = self.memory.merge_moments(
                    source_id, target_id, confirmed_duplicate=True,
                    reason="nightly exact duplicate from the same source segment",
                )
                if result.get("status") in {"merged", "already_merged"}:
                    merged_duplicate_count += 1
            report("apply", "写入整理结果", 2, 6,
                   f"完全重复事件已归并 {merged_duplicate_count} 条，源记录和附件保留")
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
            if rollup.get("_replace_key_events"):
                layer5 = self.memory.memory.get("layer5", {})
                layer5["key_events"] = (rollup.get("layer5") or {}).get("key_events", [])[:24]
                retained_ids = rollup.get("_retained_source_moment_ids") or []
                if retained_ids:
                    layer5["source_moment_ids"] = list(dict.fromkeys(retained_ids))
                    layer5["source_moment_count"] = len(layer5["source_moment_ids"])
                    layer5["source_link_coverage"] = "complete"
                self.memory._save()
            report("apply", "写入整理结果", 3, 6, "更新长期画像")
            self.memory.update_profile(
                profile, moment_id=f"nightly:{day}",
                timestamp=datetime.now().astimezone().isoformat())
            report("apply", "写入整理结果", 4, 6, "标注重要记忆和代表画面")
            importance_marked = curator.apply_importance(
                curation["highlight_moments"], curation["highlight_images"],
                curation["image_ref_map"])

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
            report("apply", "写入整理结果", 5, 6, "整理重复附件并保护重点图像")
            media_compaction = self.memory.compact_multimodal_assets()
            report("apply", "写入整理结果", 6, 6, "保存整理结果")
            elapsed = time.perf_counter() - start
            after = file.stat().st_size if file and file.exists() else 0
            return {
                "ok": True, "day": day, "moment_count": len(moments),
                "chunk_count": len(chunks),
                "indices_written": {k: len(v) for k, v in sort_data.items()},
                "aliases_combined": sum(len(v) for v in combine.values()),
                "identity_resolutions": len(identity_rewrites),
                "duplicate_events_merged": merged_duplicate_count,
                "importance_marks_added": importance_marked,
                "key_events_replaced": bool(rollup.get("_replace_key_events")),
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
