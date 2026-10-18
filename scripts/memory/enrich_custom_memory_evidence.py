#!/usr/bin/env python3
"""Review cached custom-video memories against source frames and enrich missing evidence."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "code"))

from interactive_runner import feishu_test_moments
from qwen_perceive import (
    _post_with_inference_cancel,
    PROMPT_VERSION, QwenPerceiver, _apply_temporal_fact_repair,
    _build_person_repair_prompt, _build_temporal_fact_repair_prompt,
    _person_observation_coverage_error, _person_reference_labels,
    _repair_person_references_from_video, _structured_fact_quality_error,
    _parse_json_array, clip_range, extract_frames, video_path_of,
)
from grounding_cache import PersistentObjectGroundingCache
from multimodal_memory import MultimodalAssetStore

RUNS = ROOT / "data" / "egolife" / "runs"
CACHE_FILE = RUNS / "qwen_cache_memory_prompt_v3.json"
GROUNDING_FILE = RUNS / "interactive_object_grounding_cache.sqlite3"
API_URLS = [
    value.strip().rstrip("/")
    for value in os.environ.get("QWEN_API_URLS", os.environ.get("QWEN_API_URL", "")).split(",")
    if value.strip()
]
if not API_URLS:
    API_URLS = ["http://127.0.0.1:8001"]
os.environ.pop("QWEN_API_URLS", None)

PERSON_CUE = re.compile(
    r"\b(?:man|men|woman|women|person|people|individual|individuals|"
    r"child|children|boy|girl|baby|infant|toddler|newborn|staff member|worker|"
    r"customer|patient|doctor|driver|passenger|patron|human figure)\b",
    re.I,
)
PERSON_NEGATION = re.compile(
    r"\b(?:no|not|without|none of)\s+(?:any\s+)?(?:other\s+)?"
    r"(?:people|persons?|humans?|individuals?)\b|"
    r"\b(?:people|persons?|humans?|individuals?)\s+"
    r"(?:are|were|is)\s+(?:not|never)\s+(?:visible|present|seen)\b",
    re.I,
)


def request_analysis(api_url: str, prompt: str, images: list[bytes], label: str) -> str:
    last_error = None
    for attempt in range(3):
        request_id = uuid.uuid4().hex
        try:
            response = _post_with_inference_cancel(
                api_url + "/analyze_video",
                files=[("images", (f"{label}_{i + 1}.jpg", data, "image/jpeg"))
                       for i, data in enumerate(images)],
                data={"prompt": prompt, "request_id": request_id},
                timeout=900,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:180]}")
            payload = response.json()
            return str(payload.get("analysis") or "")
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last_error or "model request failed"))


def person_cue(fields: dict) -> bool:
    text = " ".join(
        str(fields.get(key) or "")
        for key in ("scene", "activity", "interaction_details", "user_action", "objects")
    )
    text = re.sub(r"\bcamera wearer\b|\bfirst[- ]person\b", " ", text, flags=re.I)
    return bool(PERSON_CUE.search(PERSON_NEGATION.sub(" ", text)))


def _snapshot_entry(cache_file: Path, mid: str) -> dict:
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    row = data.get(mid) if isinstance(data, dict) else None
    return copy.deepcopy(row) if isinstance(row, dict) else {}


def main() -> int:
    moments = feishu_test_moments("feishu:all")
    if len(moments) != 35:
        raise RuntimeError(f"Expected 35 custom clips, found {len(moments)}")
    for api_url in API_URLS[:4]:
        health = requests.get(api_url + "/health", timeout=8).json()
        if not health.get("model_loaded"):
            raise RuntimeError(f"Qwen model is not ready at {api_url}")
    api_urls = API_URLS[:4]
    print(f"custom clips={len(moments)}; Qwen endpoints={len(api_urls)}; prompt={PROMPT_VERSION}")

    backup_dir = ROOT / "archive" / "20261018_memory_enrichment"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / "qwen_cache_memory_prompt_v3.before_custom_frame_review.json"
    if CACHE_FILE.exists() and not backup.exists():
        shutil.copy2(CACHE_FILE, backup)
        print(f"cache backup={backup}")

    grounding_cache = PersistentObjectGroundingCache(GROUNDING_FILE, ROOT)
    media = object.__new__(MultimodalAssetStore)
    endpoint_locks = [threading.Lock() for _ in api_urls]
    full_reanalysis_lock = threading.Lock()

    def process(index: int, moment: dict) -> dict:
        mid = str(moment.get("id") or "")
        entry = _snapshot_entry(CACHE_FILE, mid)
        fields = copy.deepcopy(entry.get("fields") or {})
        if entry.get("prompt_version") != PROMPT_VERSION or not fields:
            return {"id": mid, "status": "stale_or_missing_cache", "fields": fields}
        original_fields = copy.deepcopy(fields)
        api_index = index % len(api_urls)
        api_url = api_urls[api_index]
        person_log = {"status": "not_needed"}
        fact_log = {"status": "not_needed"}
        raw_repair = ""
        person_raw_holder = []
        buffers = []
        video = video_path_of(moment)
        full_review_log = {"status": "not_needed"}
        source_basename = Path(video).name.casefold() if video else ""
        if source_basename in {"5.3.mp4", "7.1.mp4"}:
            # These two clips were independently flagged during frame audit:
            # 5.3 had a cross-clip clothing carry-over; 7.1 omitted a visible baby.
            # Re-run the complete source-only scene prompt for these clips before
            # applying focused repairs; serialize cache writes to the shared JSON.
            with full_reanalysis_lock:
                try:
                    fresh_perceiver = QwenPerceiver(api_url=api_url, cache_file=CACHE_FILE)
                    with endpoint_locks[api_index]:
                        fresh_result = fresh_perceiver.perceive(
                            moment, force=True, allow_legacy_cache=False
                        )
                    fresh_fields = fresh_result.get("fields")
                    fresh_people = [
                        row for row in (fresh_fields or {}).get("person_observations") or []
                        if isinstance(row, dict) and str(row.get("appearance") or "").strip()
                    ] if isinstance(fresh_fields, dict) else []
                    quality_error = (
                        _structured_fact_quality_error(fresh_fields)
                        if isinstance(fresh_fields, dict) else "full reanalysis returned no fields"
                    )
                    if not quality_error and fresh_people:
                        fields = copy.deepcopy(fresh_fields)
                        entry = copy.deepcopy(fresh_perceiver.cache.get(mid) or entry)
                        full_review_log = {
                            "status": "reperceived",
                            "source": fresh_result.get("source"),
                            "prompt_version": fresh_result.get("prompt_version"),
                            "people": len(fresh_people),
                            "facts": len(fresh_fields.get("temporal_facts") or []),
                        }
                    else:
                        full_review_log = {
                            "status": "reanalysis_rejected",
                            "error": (quality_error or "no source-grounded person observation")[:240],
                        }
                except Exception as exc:
                    full_review_log = {
                        "status": "reanalysis_failed", "error": str(exc)[:240]
                    }
        start, end = clip_range(moment)
        # Review every source clip so people omitted from both the summary and list can be found.
        needs_person_pass = bool(video)
        existing_facts = fields.get("temporal_facts") or []
        needs_fact_pass = len(existing_facts) < 4 and bool(
            fields.get("scene") or fields.get("activity") or fields.get("interaction_details")
        )
        if (needs_person_pass or needs_fact_pass) and video:
            buffers, _times = extract_frames(
                video, start, end, 6, return_times=True
            )
        if needs_person_pass and buffers:
            references = _person_reference_labels(fields)
            context = {
                key: fields.get(key)
                for key in (
                    "scene", "location", "activity", "interaction_details",
                    "user_action", "people", "person_observations", "temporal_facts",
                )
                if fields.get(key)
            }
            prompt = _build_person_repair_prompt(context, references)
            with endpoint_locks[api_index]:
                try:
                    def request_person_repair(repair_prompt, frames):
                        raw = request_analysis(api_url, repair_prompt, frames, "person_review")
                        person_raw_holder.append(raw)
                        return raw

                    repaired, person_log = _repair_person_references_from_video(
                        fields, len(buffers), buffers, request_person_repair
                    )
                    quality_error = _structured_fact_quality_error(repaired)
                    if quality_error:
                        person_log = {
                            **(person_log if isinstance(person_log, dict) else {}),
                            "status": "review_rejected",
                            "repair_error": quality_error[:240],
                        }
                    else:
                        fields = repaired
                except Exception as exc:
                    person_log = {"status": "review_failed", "error": str(exc)[:240]}
        elif needs_person_pass:
            person_log = {"status": "review_failed", "error": "source frames unavailable"}

        if needs_fact_pass and buffers:
            aliases = [
                str(row.get("entity") or "").strip()
                for row in fields.get("person_observations") or []
                if isinstance(row, dict) and str(row.get("entity") or "").strip()
            ]
            alias_rule = (
                "\nFor every person reference, use only these exact visual aliases: "
                + json.dumps(aliases, ensure_ascii=True)
                + ". Do not introduce another person label. If no listed person is clearly involved, "
                  "use an event subject and describe only what the frames show."
                + "\nExisting candidate facts from these same frames (not ground truth; verify them and "
                  "return only additional distinct facts): "
                + json.dumps(fields.get("temporal_facts") or [], ensure_ascii=True)
            )
            prompt = _build_temporal_fact_repair_prompt(len(buffers)) + alias_rule
            with endpoint_locks[api_index]:
                try:
                    raw_repair = request_analysis(
                        api_url, prompt, buffers, "event_fact_review"
                    )
                    rows = _apply_temporal_fact_repair(
                        _parse_json_array(raw_repair), len(buffers)
                    )
                    candidate = copy.deepcopy(fields)
                    merged = list(candidate.get("temporal_facts") or [])
                    seen = {
                        (str(row.get("subject_type") or "").casefold(),
                         str(row.get("subject") or "").strip().casefold(),
                         str(row.get("predicate") or "").strip().casefold(),
                         re.sub(r"\s+", " ", str(row.get("value") or "")).strip().casefold())
                        for row in merged if isinstance(row, dict)
                    }
                    added = 0
                    for row in rows:
                        key = (
                            str(row.get("subject_type") or "").casefold(),
                            str(row.get("subject") or "").strip().casefold(),
                            str(row.get("predicate") or "").strip().casefold(),
                            re.sub(r"\s+", " ", str(row.get("value") or "")).strip().casefold(),
                        )
                        if key not in seen and len(merged) < 16:
                            merged.append(row)
                            seen.add(key)
                            added += 1
                    candidate["temporal_facts"] = merged
                    quality_error = _structured_fact_quality_error(candidate)
                    if quality_error:
                        raise ValueError(quality_error)
                    fields = candidate
                    fact_log = {
                        "status": "added" if added else "no_additional_facts",
                        "method": "same_video_frames_event_fact_review",
                        "facts_added": added,
                        "total_facts": len(merged),
                    }
                except Exception as exc:
                    fact_log = {"status": "review_failed", "error": str(exc)[:240]}
        elif needs_fact_pass:
            fact_log = {"status": "review_failed", "error": "source frames unavailable"}

        object_log = {"status": "not_needed", "objects": 0, "grounded": 0}
        object_labels = [str(value).strip() for value in fields.get("objects") or []
                         if str(value).strip()][:12]
        if full_review_log.get("status") == "reperceived" and video and object_labels:
            try:
                object_frames = media.sample_visual_candidates(
                    video, start, end, source_id=mid, count=7
                )
                if not object_frames:
                    raise RuntimeError("no sampled frames for changed object labels")
                source_key = grounding_cache.source_key(
                    mid, video, start, end, object_labels
                )
                cached, lookup_ms = grounding_cache.get_source(source_key)
                lookup = "source_cache_hit" if cached is not None else ""
                if cached is None:
                    grounding_key = grounding_cache.key(mid, object_labels, object_frames)
                    cached, lookup_ms = grounding_cache.get(grounding_key)
                    lookup = "cache_hit" if cached is not None else ""
                    if cached is None:
                        with endpoint_locks[api_index]:
                            cached = QwenPerceiver(
                                api_url=api_url, cache_file=CACHE_FILE
                            ).ground_objects(object_labels, object_frames)
                        if not grounding_cache.put(grounding_key, cached):
                            raise RuntimeError("object grounding result was not cached")
                        lookup = "created"
                    if source_key:
                        grounding_cache.put_source(source_key, cached)
                object_log = {
                    "status": lookup,
                    "objects": len(object_labels),
                    "grounded": sum(
                        item.get("status") == "grounded"
                        for item in (cached.get("items") or [])
                        if isinstance(item, dict)
                    ),
                    "lookup_ms": lookup_ms,
                }
            except Exception as exc:
                object_log = {
                    "status": "grounding_failed", "objects": len(object_labels),
                    "grounded": 0, "error": str(exc)[:180],
                }

        person_rows = [
            row for row in fields.get("person_observations") or []
            if isinstance(row, dict) and str(row.get("entity") or "").strip()
        ]
        crop_log = {"status": "not_needed", "people": 0, "grounded": 0, "crop_candidates": 0}
        if person_rows and video:
            person_frames = media.sample_visual_candidates(
                video, start, end, source_id=mid, count=7
            )
            if person_frames:
                labels = [
                    "person-crop-v1: " + str(row.get("entity") or "").strip()
                    + " | " + str(row.get("appearance") or "").strip()
                    for row in person_rows[:8]
                ]
                try:
                    key = grounding_cache.key(mid, labels, person_frames)
                    with endpoint_locks[api_index]:
                        cached, lookup_ms = grounding_cache.get(key)
                        if cached is None:
                            cached = QwenPerceiver(
                                api_url=api_url, cache_file=CACHE_FILE
                            ).ground_people(person_rows, person_frames)
                            grounding_cache.put(key, cached)
                            lookup = "created"
                        else:
                            lookup = f"hit:{lookup_ms}ms"
                    items = cached.get("items") or []
                    grounded = sum(item.get("status") == "grounded" for item in items)
                    crop_log = {
                        "status": lookup,
                        "people": len(person_rows),
                        "grounded": grounded,
                        "crop_candidates": sum(bool(item.get("frame_boxes")) for item in items),
                    }
                except Exception as exc:
                    crop_log = {
                        "status": "grounding_failed",
                        "people": len(person_rows),
                        "grounded": 0,
                        "crop_candidates": 0,
                        "error": str(exc)[:180],
                    }
            else:
                crop_log = {
                    "status": "no_sampled_frames", "people": len(person_rows),
                    "grounded": 0, "crop_candidates": 0,
                }

        fields_changed = fields != original_fields
        save_entry = bool(
            fields_changed or needs_person_pass or needs_fact_pass or person_rows
            or full_review_log.get("status") != "not_needed"
            or object_log.get("status") != "not_needed"
        )
        if fields_changed:
            entry["pre_custom_frame_review_fields"] = original_fields
            entry["fields"] = fields
            entry["source"] = "cache_frame_quality_repair"
        if full_review_log.get("status") != "not_needed":
            entry["full_source_frame_reanalysis"] = full_review_log
        if needs_person_pass:
            entry["person_frame_review"] = person_log
        if person_raw_holder:
            entry["person_frame_review_raw"] = person_raw_holder[-1]
        if needs_fact_pass:
            entry["temporal_fact_repair"] = fact_log
            if raw_repair:
                entry["temporal_fact_repair_raw"] = raw_repair
        if person_rows:
            entry["person_crop_preheat"] = crop_log
        if object_log.get("status") != "not_needed":
            entry["object_crop_preheat"] = object_log
        entry["custom_frame_review_version"] = "2026-10-18-v1"
        entry["custom_frame_review_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return {
            "id": mid,
            "status": "updated" if fields_changed else "reviewed",
            "fields": fields,
            "entry": entry,
            "changed": save_entry,
            "fields_changed": fields_changed,
            "person": person_log,
            "facts": fact_log,
            "crops": crop_log,
            "objects": object_log,
            "full_review": full_review_log,
        }

    results = [None] * len(moments)
    with ThreadPoolExecutor(max_workers=len(api_urls), thread_name_prefix="custom-frame-review") as pool:
        futures = {pool.submit(process, i, moment): i for i, moment in enumerate(moments)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as exc:
                results[i] = {"id": moments[i].get("id"), "status": "failed", "error": str(exc)[:240]}
            row = results[i]
            print(
                f"{row.get('id')} | {row.get('status')} | "
                f"people={len((row.get('fields') or {}).get('person_observations') or [])} "
                f"facts={len((row.get('fields') or {}).get('temporal_facts') or [])} "
                f"crop={row.get('crops',{}).get('grounded',0)}/{row.get('crops',{}).get('people',0)} "
                f"object_crop={row.get('objects',{}).get('grounded',0)}/{row.get('objects',{}).get('objects',0)} "
                f"person_review={row.get('person',{}).get('status')} "
                f"fact_review={row.get('facts',{}).get('status')} "
                f"full_review={row.get('full_review',{}).get('status')}"
            )

    writer = QwenPerceiver(api_url=api_urls[0], cache_file=CACHE_FILE)
    saved_entries = 0
    fields_changed = 0
    for row in results:
        if not row or not row.get("changed"):
            continue
        mid = row["id"]
        writer.cache[mid] = row["entry"]
        writer._dirty_cache_keys.add(mid)
        saved_entries += 1
        fields_changed += int(bool(row.get("fields_changed")))
    if saved_entries:
        writer._save_cache()
    print(f"saved review metadata for {saved_entries} custom cache entries")

    print(
        "SUMMARY clips={} fields_changed={} person_reviewed={} person_review_failed={} "
        "fact_added={} fact_review_failed={} crop_grounded={}/{}".format(
            len(results),
            fields_changed,
            sum(bool(row and row.get("person", {}).get("status") not in {"not_needed", "review_failed"})
                for row in results),
            sum(bool(row and row.get("person", {}).get("status") == "review_failed")
                for row in results),
            sum(bool(row and row.get("facts", {}).get("status") == "added")
                for row in results),
            sum(bool(row and row.get("facts", {}).get("status") == "review_failed")
                for row in results),
            sum((row or {}).get("crops", {}).get("grounded", 0) for row in results),
            sum((row or {}).get("crops", {}).get("people", 0) for row in results),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
