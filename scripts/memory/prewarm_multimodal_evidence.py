"""Warm independent caches for the Feishu custom video test set.

Inputs are the copied video assets only. No benchmark caption, transcript,
reference answer, or test metadata is sent to either model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "code"))

import interactive_runner as ir  # noqa: E402
from interactive_runner import feishu_test_moments  # noqa: E402
from qwen_perceive import (
    _post_with_inference_cancel,  # noqa: E402
    LIST_PARSER_VERSION, PROMPT_VERSION, QwenPerceiver, _apply_person_repair, _contains_cjk,
    _object_move_evidence_error, _parse_person_repair, _build_person_repair_prompt,
    _person_observation_coverage_error, _person_reference_labels,
    _salvage_person_reference_gaps, _repair_person_references_from_video,
    _replace_field_value, _structured_fact_quality_error,
    _normalize_camera_wearer_actor_aliases, _temporal_fact_coverage_error, clip_range,
    extract_frames, parse_analysis, source_video_input_fingerprint, video_path_of,
)
from grounding_cache import PersistentObjectGroundingCache  # noqa: E402
from multimodal_memory import MultimodalAssetStore, _run_ffmpeg  # noqa: E402
from audio_transcript_cache import PersistentTranscriptCache  # noqa: E402
from face_detection import detect_faces, expanded_face_sample_times  # noqa: E402

RUNS = ROOT / "data" / "egolife" / "runs"
QWEN_CACHE = RUNS / "qwen_cache_memory_prompt_v3.json"
GROUNDING_CACHE = RUNS / "interactive_object_grounding_cache.sqlite3"
SPEECH_CACHE = RUNS / "interactive_audio_transcript_cache.json"
API_URLS = [
    value.strip().rstrip("/")
    for value in os.environ.get("QWEN_API_URLS", os.environ.get(
        "QWEN_API_URL", "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004"
    )).split(",") if value.strip()
]
API = API_URLS[0]
PROGRESS_EVERY = 5  # Publish live status often enough for long-running cache jobs.
CACHE_SAVE_EVERY = 25  # Keep full-file atomic rewrites bounded.


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _source_fingerprint_matches(entry: dict, moment: dict) -> bool:
    """Only treat a perception cache row as current for its exact source clip."""
    video = video_path_of(moment)
    if not video:
        return False
    try:
        start, end = clip_range(moment)
        expected = source_video_input_fingerprint(video, start, end)
    except (OSError, TypeError, ValueError):
        return False
    return bool(expected and entry.get("source_input_fingerprint") == expected)


def video_audio_chunks(moment: dict):
    video = video_path_of(moment)
    if not video:
        return
    start, end = clip_range(moment)
    end = float(end) if end is not None else float(start) + 30.0
    cursor = max(0.0, float(start or 0.0))
    while cursor < end:
        duration = min(20.0, end - cursor)
        data = _run_ffmpeg([
            "-ss", f"{cursor:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-f", "wav", "pipe:1",
        ])
        if data:
            yield cursor, duration, data
        cursor += duration


def load_moments(all_profiles: bool) -> list[dict]:
    profiles = []
    if all_profiles:
        # Warm the interactive short-term datasets first so the current
        # UI scenario becomes reusable before the larger long-term corpus.
        profiles.extend([
            ("short-base", ir._load_jsonl(ir.SHORT_FILE)),
            ("short-expanded", ir._load_jsonl(ir.SHORT_EXPANDED_FILE)),
        ])
    profiles.append(("custom", feishu_test_moments("feishu:all")))
    if all_profiles:
        for participant in ir.PARTICIPANT_NAMES:
            path = ir.LONG_PERSON_DIR / f"{participant}.jsonl"
            if path.is_file():
                profiles.append((f"long-{participant}", ir._load_jsonl(path)))
    unique = {}
    for _label, rows in profiles:
        for moment in rows:
            mid = str(moment.get("id") or "")
            if mid:
                unique.setdefault(mid, moment)
    return list(unique.values())


def load_scoped_moments(short_half: bool, custom_all: bool,
                        long_participants: list[str] | None,
                        short_first: int | None = None,
                        short_all: bool = False) -> tuple[list[dict], str]:
    """Load only the explicitly requested profiles for a focused warm run."""
    profiles = []
    labels = []
    if short_all:
        profiles.extend([
            ("short-base-all", ir._load_jsonl(ir.SHORT_FILE)),
            ("short-expanded-all", ir._load_jsonl(ir.SHORT_EXPANDED_FILE)),
        ])
        labels.append("all short-term clips")
    elif short_first is not None:
        count = max(0, int(short_first))
        profiles.extend([
            ("short-base-first", ir._load_jsonl(ir.SHORT_FILE)[:count]),
            ("short-expanded-first", ir._load_jsonl(ir.SHORT_EXPANDED_FILE)[:count]),
        ])
        labels.append(f"first {count} clips in each short-term set")
    elif short_half:
        # Evenly sample every other row in each short-term set, covering its full timeline.
        profiles.extend([
            ("short-base-half", ir._load_jsonl(ir.SHORT_FILE)[::2]),
            ("short-expanded-half", ir._load_jsonl(ir.SHORT_EXPANDED_FILE)[::2]),
        ])
        labels.append("short-term half (evenly sampled)")
    if custom_all:
        profiles.append(("custom-all", feishu_test_moments("feishu:all")))
        labels.append("custom test set (all)")
    selected = []
    aliases = {}
    for key, display in ir.PARTICIPANT_NAMES.items():
        aliases[key.casefold()] = key
        aliases[display.casefold()] = key
    for raw in long_participants or []:
        participant = aliases.get(str(raw).strip().casefold())
        if not participant:
            valid = ", ".join(f"{key} ({name})" for key, name in ir.PARTICIPANT_NAMES.items())
            raise ValueError(f"unknown long-term participant {raw!r}; choose from {valid}")
        if participant in selected:
            continue
        selected.append(participant)
        path = ir.LONG_PERSON_DIR / f"{participant}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"long-term profile is missing: {path}")
        profiles.append((f"long-{participant}", ir._load_jsonl(path)))
    if selected:
        labels.append("long-term " + ", ".join(ir.PARTICIPANT_NAMES[p] for p in selected))
    if not profiles:
        raise ValueError("focused prewarm needs at least one profile selection")
    unique = {}
    for _label, rows in profiles:
        for moment in rows:
            mid = str(moment.get("id") or "")
            if mid:
                unique.setdefault(mid, moment)
    return list(unique.values()), " + ".join(labels)



def _normalize_or_drop_unverified_moves(fields: dict) -> tuple[dict, int]:
    """Structure source-backed 'from X to Y' facts; discard unsupported moves."""
    import re

    normalized = dict(fields)
    facts = []
    removed = 0
    for original in fields.get("temporal_facts") or []:
        if not isinstance(original, dict):
            continue
        row = dict(original)
        is_move = (str(row.get("subject_type") or "").casefold() == "object"
                   and str(row.get("predicate") or "").casefold() == "moved")
        if is_move:
            has_pair = bool((row.get("from_location") and row.get("to_location"))
                            or (row.get("from_state") and row.get("to_state")))
            has_frame = isinstance(row.get("frame_index"), int) and row["frame_index"] > 0
            has_evidence = bool(str(row.get("evidence") or "").strip())
            if not has_pair:
                value = str(row.get("value") or "")
                match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[.;]|$)",
                                  value, flags=re.I)
                if match:
                    row["from_location"] = match.group(1).strip()
                    row["to_location"] = match.group(2).strip()
                    has_pair = True
            if not (has_pair and has_frame and has_evidence):
                removed += 1
                continue
        facts.append(row)
    normalized["temporal_facts"] = facts
    return normalized, removed



def _normalize_person_frame_indices(payload: dict, frame_count: int) -> dict:
    """Normalize a model's zero-based frame indices to the stored 1-based schema."""
    rows = payload.get("person_observations")
    if not isinstance(rows, list):
        return payload
    indices = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            indices.append(int(row.get("frame_index")))
        except (TypeError, ValueError):
            continue
    zero_based = 0 in indices
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("frame_index"))
        except (TypeError, ValueError):
            continue
        if zero_based:
            index += 1
            evidence = str(row.get("evidence") or "")
            row["evidence"] = re.sub(
                r"\\bframe\\s+(\\d+)\\b",
                lambda match: f"frame {int(match.group(1)) + 1}",
                evidence, flags=re.I)
        row["frame_index"] = index
    return payload


_PERSON_REVIEW_CUE = re.compile(
    r"\b(?:man|men|woman|women|person|people|individual|individuals|"
    r"child|children|baby|infant|toddler|newborn|boy|girl|staff member|worker|customer|patient|doctor|"
    r"driver|passenger|patron|elderly person|young person)\b",
    flags=re.I,
)
_PERSON_REVIEW_NEGATION = re.compile(
    r"\b(?:no|not|without|none of)\s+(?:any\s+)?(?:other\s+)?"
    r"(?:people|persons?|humans?|individuals?)\b|"
    r"\b(?:people|persons?|humans?|individuals?)\s+"
    r"(?:are|were|is)\s+(?:not|never)\s+(?:visible|present|seen)\b",
    flags=re.I,
)


_PERSON_GROUP_CUE = re.compile(
    r"\b(?:several|multiple|many|both|two|three|four|five|six)\s+"
    r"(?:people|persons?|men|women|individuals?|children|patrons|passengers|customers|workers)\b|"
    r"\bgroup\s+of\s+(?:people|persons?|individuals?|children)\b|"
    r"\b(?:another|other)\s+(?:person|man|woman|individual|child)\b|"
    r"\b(?:others|crowd|people|men|women|individuals|children|patrons|passengers|customers|workers)\b",
    flags=re.I,
)


def needs_person_observation_review(fields: dict, entry: dict | None = None) -> bool:
    """Review missing people and plural scenes even when one person was found."""
    review = (entry or {}).get("person_observation_review") or {}
    if review.get("frame_inventory_reviewed") is True:
        return False
    observations = fields.get("person_observations") or []
    has_observation = any(
        isinstance(row, dict) and str(row.get("appearance") or "").strip()
        for row in observations
    )
    text = " ".join(str(fields.get(key) or "")
                    for key in ("scene", "interaction_details", "activity"))
    text = re.sub(r"\bcamera wearer\b|\bfirst[- ]person\b", " ", text, flags=re.I)
    text = _PERSON_REVIEW_NEGATION.sub(" ", text)
    if not _PERSON_REVIEW_CUE.search(text):
        return False
    if not has_observation:
        return True
    return bool(_PERSON_GROUP_CUE.search(text))

def repair_cached_quality_gap(perceiver: QwenPerceiver, moment: dict) -> dict:
    """Repair cached person coverage using one source-video-only focused pass."""
    mid = str(moment.get("id") or "")
    with perceiver._cache_lock:
        original = perceiver.cache.get(mid)
        original = json.loads(json.dumps(original)) if original else None
    if not original or original.get("prompt_version") != PROMPT_VERSION:
        return {"source": "error", "error": "no current cache entry", "fields": None}
    fields = json.loads(json.dumps(original.get("fields") or {}))
    fields = _normalize_camera_wearer_actor_aliases(fields)
    fields, removed_moves = _normalize_or_drop_unverified_moves(fields)

    if _contains_cjk(json.dumps(fields, ensure_ascii=False)) or _contains_cjk(original.get("raw", "")):
        return {"source": "needs_full_reperception", "error": "cached text contains CJK", "fields": None}

    person_error = _person_observation_coverage_error(fields)
    person_review_needed = needs_person_observation_review(fields, original)
    person_review_performed = False
    person_review_frame_count = 0
    best_candidate_fields = None
    removed_person_claims = []
    if person_error or person_review_needed:
        video = video_path_of(moment)
        if not video:
            return {"source": "error", "error": "source video unavailable", "fields": None}
        start, end = clip_range(moment)
        try:
            buffers, _frame_times = extract_frames(video, start, end, perceiver.num_frames,
                                                    return_times=True)
            if not buffers:
                return {"source": "error", "error": "no frames available for person review",
                        "fields": None}
            person_review_performed = True
            person_review_frame_count = len(buffers)
            person_repair_source = "same_video_frames_scene_review"
            references = _person_reference_labels(fields)
            facts = []
            for fact in fields.get("temporal_facts") or []:
                if not isinstance(fact, dict):
                    continue
                facts.append({key: fact.get(key) for key in (
                    "subject_type", "subject", "actor", "predicate", "value",
                    "affected_people", "frame_index", "evidence",
                ) if fact.get(key) is not None})
            context = {
                "scene_from_same_frames": fields.get("scene"),
                "interaction_details_from_same_frames": fields.get("interaction_details"),
                "existing_person_labels": references,
                "source_facts_from_same_frames": facts,
            }
            prompt = _build_person_repair_prompt(context, references)
            files = [("images", (f"f{i}.jpg", data, "image/jpeg"))
                     for i, data in enumerate(buffers)]

            def request_person_repair(repair_prompt, image_files):
                request_id = __import__("uuid").uuid4().hex
                api_url = perceiver.next_api_url()
                perceiver._activate_request(request_id, api_url)
                try:
                    response = _post_with_inference_cancel(
                        f"{api_url}/analyze_video",
                        files=image_files,
                        data={"prompt": repair_prompt, "request_id": request_id},
                        timeout=900,
                    )
                    if response.status_code != 200:
                        raise ValueError(
                            f"focused person fallback returned HTTP {response.status_code}"
                        )
                    return response.json().get("analysis", "")
                finally:
                    perceiver._finish_request(request_id)

            def retry_same_frame_repair(candidate):
                return _repair_person_references_from_video(
                    candidate, len(buffers), files, request_person_repair
                )

            last_error = ""
            repair_feedback = ""
            repaired_fields = None
            patched_raw = original.get("raw", "")

            # Preserve a previous frame-grounded review for this fingerprinted clip
            # when a later incomplete review returns an empty observation list.
            prior_payload = None
            prior_fields = original.get("pre_custom_frame_review_fields")
            prior_rows = ([row for row in (prior_fields or {}).get("person_observations") or []
                           if isinstance(row, dict)]
                          if isinstance(prior_fields, dict) else [])
            if prior_rows:
                prior_payload = {"person_observations": prior_rows, "entity_map": []}
            if prior_payload is None:
                prior_raw = original.get("person_frame_review_raw") or ""
                if prior_raw:
                    try:
                        candidate_payload = _parse_person_repair(prior_raw)
                        if candidate_payload.get("person_observations"):
                            prior_payload = candidate_payload
                    except (TypeError, ValueError):
                        prior_payload = None
            if prior_payload is not None:
                try:
                    restored = _apply_person_repair(fields, prior_payload, references, len(buffers))
                    if not _structured_fact_quality_error(restored):
                        repaired_fields = restored
                        best_candidate_fields = restored
                        rows = restored.get("person_observations") or []
                        replacement = _replace_field_value(
                            patched_raw, "Person Observations",
                            json.dumps(rows, ensure_ascii=False))
                        if replacement is not None:
                            patched_raw = replacement
                        person_repair_source = "same_video_prior_review_restore"
                except (TypeError, ValueError):
                    repaired_fields = None
                    best_candidate_fields = None

            for attempt in (range(3) if repaired_fields is None else ()):
                request_id = __import__("uuid").uuid4().hex
                api_url = perceiver.next_api_url()
                perceiver._activate_request(request_id, api_url)
                try:
                    attempt_prompt = prompt
                    if repair_feedback:
                        attempt_prompt += (
                            "\nThe previous answer left references unresolved. Recheck the supplied pixels. "
                            "Any missing existing neutral ID must appear directly as the exact same observation "
                            "entity, with no renumbering or swapping; entity_map is only for descriptive labels. "
                            "Do not map a group or role by guess, and do not invent appearance for someone who is "
                            "not visible. Unmatched references and available visual IDs are data only: "
                            + repair_feedback + "\nReturn the full corrected JSON object."
                        )
                    response = _post_with_inference_cancel(
                        f"{api_url}/analyze_video",
                        files=files,
                        data={"prompt": attempt_prompt, "request_id": request_id},
                        timeout=900,
                    )
                    response.raise_for_status()
                    payload = _parse_person_repair(response.json().get("analysis", ""))
                    payload = _normalize_person_frame_indices(payload, len(buffers))
                    if _contains_cjk(json.dumps(payload, ensure_ascii=False)):
                        raise ValueError("person repair returned CJK text")
                    repaired_fields = _apply_person_repair(fields, payload, references, len(buffers))
                    best_candidate_fields = repaired_fields
                    quality_error = _structured_fact_quality_error(repaired_fields)
                    if quality_error:
                        observed = {" ".join(str(row.get("entity") or "").casefold().split())
                                    for row in repaired_fields.get("person_observations") or []
                                    if isinstance(row, dict)}
                        remaining = [label for label in _person_reference_labels(repaired_fields)
                                     if " ".join(label.casefold().split()) not in observed]
                        direct_ids = [
                            label for label in references
                            if re.fullmatch(r"unidentified person \d+", label, re.I)
                        ]
                        repair_feedback = json.dumps({
                            "coverage_error": quality_error,
                            "unmatched_references": remaining[:16],
                            "exact_neutral_ids_still_missing": [
                                label for label in direct_ids
                                if " ".join(label.casefold().split()) not in observed
                            ],
                            "available_observation_ids": sorted(observed),
                        }, ensure_ascii=True)
                        raise ValueError(quality_error)
                    rows = repaired_fields.get("person_observations") or []
                    replacement = _replace_field_value(
                        patched_raw, "Person Observations",
                        json.dumps(rows, ensure_ascii=False))
                    if replacement is not None:
                        patched_raw = replacement
                    break
                except Exception as exc:
                    last_error = str(exc)[:220]
                    repaired_fields = None
                    if attempt < 2:
                        time.sleep(1 + attempt)
                finally:
                    perceiver._finish_request(request_id)
            if repaired_fields is None:
                if best_candidate_fields is None:
                    return {"source": "error", "error": last_error or "person repair failed", "fields": None}
                salvaged, removed_person_claims, salvage_log = _salvage_person_reference_gaps(
                    best_candidate_fields, retry_same_frame_repair
                )
                if salvaged is None:
                    fallback_error = (salvage_log.get("repair_error")
                                      if isinstance(salvage_log, dict) else "")
                    return {"source": "error",
                            "error": last_error or fallback_error or "person repair failed",
                            "fields": None}
                fields = salvaged
            else:
                fields = repaired_fields
        except Exception as exc:
            return {"source": "error", "error": str(exc)[:220], "fields": None}
    else:
        quality_error = _structured_fact_quality_error(fields)
        if quality_error:
            return {"source": "error", "error": quality_error, "fields": None}

    with perceiver._cache_lock:
        updated = dict(original)
        updated["fields"] = fields
        updated["raw"] = patched_raw if person_review_performed else original.get("raw", "")
        if person_review_performed:
            observation_count = len(fields.get("person_observations") or [])
            coverage_gaps = [item for item in removed_person_claims
                             if isinstance(item, dict) and item.get("kind") == "person_coverage_gap"]
            removed_claims = [item for item in removed_person_claims
                              if not (isinstance(item, dict) and item.get("kind") == "person_coverage_gap")]
            updated["person_observation_review"] = {
                "status": ("partial_person_coverage_after_source_review" if coverage_gaps else
                           "unresolved_references_removed_after_source_review"
                           if removed_claims else
                           "observations_found" if observation_count else
                           "no_person_visible_in_sampled_frames"),
                "frame_inventory_reviewed": True,
                "frame_count": person_review_frame_count,
                "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                **({"removed_unverified_references": removed_person_claims}
                   if removed_person_claims else {}),
            }
            updated["person_observation_repair"] = {
                "source": person_repair_source,
                "observations_added": observation_count,
                "unverified_references_removed": len(removed_claims),
                "coverage_gaps": coverage_gaps,
                "raw_model_output_preserved": True,
            }
        quality_repair = {}
        if removed_moves:
            quality_repair["unsupported_moved_facts_removed"] = removed_moves
        if removed_person_claims:
            coverage_gaps = [item for item in removed_person_claims
                             if isinstance(item, dict) and item.get("kind") == "person_coverage_gap"]
            removed_claims = [item for item in removed_person_claims
                              if not (isinstance(item, dict) and item.get("kind") == "person_coverage_gap")]
            if removed_claims:
                quality_repair["unverified_person_claims_removed"] = removed_claims
            if coverage_gaps:
                quality_repair["person_coverage_gaps"] = coverage_gaps
        if quality_repair:
            updated["quality_repair"] = {
                "method": "source_evidence_validation", **quality_repair
            }
        if any(not (isinstance(item, dict) and item.get("kind") == "person_coverage_gap")
               for item in removed_person_claims):
            updated["raw_model_output"] = (original.get("raw_model_output")
                                             or original.get("raw", ""))
        perceiver.cache[mid] = updated
    return {"source": ("cache_person_repair"
                        if person_review_performed
                        and person_repair_source == "same_video_prior_review_restore"
                        else "qwen_repair"
                        if person_review_performed and not removed_person_claims
                        else "quality_repair"),
            "fields": fields, "removed_moves": removed_moves,
            "removed_person_claims": removed_person_claims}


def _cache_text_needs_full_reperception(entry: dict) -> bool:
    fields = entry.get("fields") or {}
    return (_contains_cjk(json.dumps(fields, ensure_ascii=False))
            or _contains_cjk(entry.get("raw", "")))


def resolve_cached_perception(perceiver: QwenPerceiver, moment: dict, entry: dict,
                              refresh_quality_gap: bool = False) -> dict:
    """Refresh scenes as a whole; use focused repairs only for person coverage."""
    result = perceiver.perceive(moment, force=False, persist_cache=False, allow_quality_warning=True)
    if result.get("source") == "error" or not result.get("fields"):
        return result

    fields = result["fields"]
    quality_error = (_structured_fact_quality_error(fields)
                     or _temporal_fact_coverage_error(fields))
    if quality_error and (refresh_quality_gap or not result.get("quality_warning")
                          or result.get("source") == "cache_quality_warning"):
        # Person-only repair cannot restore missing event or state facts.
        result = perceiver.perceive(moment, force=True, persist_cache=False, allow_quality_warning=True)
        if result.get("source") == "error" or not result.get("fields"):
            return result
        fields = result["fields"]
        quality_error = (_structured_fact_quality_error(fields)
                         or _temporal_fact_coverage_error(fields))
        if quality_error:
            result["quality_warning"] = quality_error

    mid = str(moment.get("id") or "")
    with perceiver._cache_lock:
        refreshed = perceiver.cache.get(mid) or entry or {}
    if needs_person_observation_review(fields, refreshed):
        repaired = repair_cached_quality_gap(perceiver, moment)
        if repaired.get("source") == "needs_full_reperception":
            repaired = perceiver.perceive(moment, force=True, persist_cache=False, allow_quality_warning=True)
        if repaired.get("source") == "error" or not repaired.get("fields"):
            if result.get("quality_warning"):
                result["quality_warning"] += "; "
            result["quality_warning"] = (
                str(result.get("quality_warning") or "")
                + str(repaired.get("error") or "person observation review failed")
            )[:500]
        else:
            result = repaired

    if result.get("quality_warning"):
        # Cache availability and content quality are separate outcomes. Keep the
        # source-matched current-prompt result reusable, while exposing the exact
        # quality gap to the UI and evaluation report instead of dropping the row.
        if result.get("fields"):
            return result
        return {"source": "error", "error": result["quality_warning"], "fields": None}
    return result


def run(limit: int | None = None, dry_run: bool = False,
        all_profiles: bool = False, short_half: bool = False,
        custom_all: bool = False, long_participants: list[str] | None = None,
        short_first: int | None = None, short_all: bool = False,
        crop_only: bool = False) -> dict:
    scoped = short_half or short_first is not None or short_all or custom_all or bool(long_participants)
    if all_profiles and scoped:
        raise ValueError("--all-profiles cannot be combined with focused profile selections")
    if scoped:
        moments, scope = load_scoped_moments(
            short_half, custom_all, long_participants, short_first, short_all)
    else:
        moments = load_moments(all_profiles)
        scope = "all selectable profiles" if all_profiles else "custom test set"
    if limit:
        moments = moments[:max(1, int(limit))]
    perceiver = QwenPerceiver(api_urls=API_URLS, cache_file=QWEN_CACHE)
    grounding_cache = PersistentObjectGroundingCache(GROUNDING_CACHE, ROOT)
    # sample_visual_candidates is a read-only method; no media assets are written.
    media = object.__new__(MultimodalAssetStore)
    speech_cache = PersistentTranscriptCache(SPEECH_CACHE)

    def warm_face_cache(source_id, video, start, end, frame_times):
        face_times = expanded_face_sample_times(
            start, end, frame_times, max_samples=12)
        highres = media.sample_visual_candidates(
            video, start, end, source_id=source_id,
            count=len(face_times) or 7, full_resolution=True,
            sample_times=face_times or None)
        if not highres:
            return {"face_cache_failed": 1}
        key = grounding_cache.key(
            source_id, ["yolov8n-face-detections-v2-dense"], highres)
        cached, _ = grounding_cache.get_faces(key)
        if isinstance(cached, dict) and cached.get("model") == "YOLOv8n-Face":
            return {"face_cache_hit": 1}
        faces = detect_faces(highres)
        if grounding_cache.put_faces(key, {"schema": 1, "status": "completed",
                                           "model": "YOLOv8n-Face", "frame_count": len(highres),
                                           "faces": faces}):
            return {"face_cache_new": 1}
        return {"face_cache_failed": 1}

    totals = {
        "clips": len(moments), "perception_cache_hits": 0, "perception_refreshed": 0,
        "perception_new": 0, "perception_repaired": 0, "perception_failed": 0,
        "perception_quality_warning_hits": 0,
        "grounding_hit": 0, "grounding_source_hit": 0, "grounding_new": 0,
        "grounding_failed": 0, "grounding_not_needed": 0,
        "audio_hit": 0, "audio_new": 0, "audio_failed": 0, "audio_silent": 0,
        "person_grounding_hit": 0, "person_grounding_new": 0,
        "person_grounding_failed": 0, "person_grounding_not_needed": 0,
        "face_cache_hit": 0, "face_cache_new": 0, "face_cache_failed": 0,
    }

    if crop_only:
        crop_totals = {"clips": len(moments), "object_hit": 0, "object_new": 0,
                       "object_failed": 0, "object_not_needed": 0,
                       "person_hit": 0, "person_new": 0,
                       "person_failed": 0, "person_not_needed": 0,
                       "face_cache_hit": 0, "face_cache_new": 0, "face_cache_failed": 0,
                       "source_mismatch_or_no_fields": 0}

        def crop_one(moment):
            mid = str(moment.get("id") or "")
            entry = perceiver.cache.get(mid) or {}
            fields = entry.get("fields") or {}
            if not fields or not _source_fingerprint_matches(entry, moment):
                return {"source_mismatch_or_no_fields": 1}
            video = video_path_of(moment)
            if not video:
                return {"object_failed": 1, "person_failed": 1, "id": mid,
                        "error": "source video unavailable"}
            start, end = clip_range(moment)
            frames = media.sample_visual_candidates(video, start, end,
                                                    source_id=mid, count=7)
            if not frames:
                return {"object_failed": 1, "person_failed": 1, "id": mid,
                        "error": "no sampled visual frames"}
            result = {}
            # Match InteractiveSession exactly; normalizing labels creates a different cache key.
            labels = fields.get("objects") or []
            if labels:
                source_key = grounding_cache.source_key(mid, video, start, end, labels)
                cached = grounding_cache.get_source(source_key)[0]
                key = grounding_cache.key(mid, labels, frames)
                if cached is None:
                    cached = grounding_cache.get(key)[0]
                if cached is None:
                    cached = perceiver.ground_objects(labels, frames)
                    if grounding_cache.put(key, cached):
                        grounding_cache.put_source(source_key, cached)
                        result["object_new"] = 1
                    else:
                        result["object_failed"] = 1
                        result["error"] = str(cached.get("message") or cached.get("status"))
                else:
                    grounding_cache.put_source(source_key, cached)
                    result["object_hit"] = 1
            else:
                result["object_not_needed"] = 1
            people = [row for row in fields.get("person_observations") or []
                      if isinstance(row, dict) and str(row.get("entity") or "").strip()][:8]
            if people:
                frame_times = entry.get("frame_times_s") or []
                person_frames = media.sample_visual_candidates(
                    video, start, end, source_id=mid, count=len(frame_times) or 7,
                    sample_times=frame_times or None)
                person_labels = ["person-crop-v8-qwen-anchor-cache-key-mutual-face-identity: " + str(row.get("entity") or "").strip()
                                 + " | " + str(row.get("appearance") or "").strip() + " | anchors=" + ir._person_anchor_cache_suffix(row)
                                 for row in people]
                face_result = warm_face_cache(mid, video, start, end, frame_times)
                for stat in ("face_cache_hit", "face_cache_new", "face_cache_failed"):
                    crop_totals[stat] += int(face_result.get(stat, 0) or 0)
                key = grounding_cache.key(mid, person_labels, person_frames)
                cached = grounding_cache.get(key)[0]
                if cached is None:
                    cached = perceiver.ground_people(people, person_frames)
                    if grounding_cache.put(key, cached):
                        result["person_new"] = 1
                    else:
                        result["person_failed"] = 1
                        result["error"] = str(cached.get("message") or cached.get("status"))
                else:
                    result["person_hit"] = 1
            else:
                result["person_not_needed"] = 1
            result["id"] = mid
            return result

        worker_count = 4
        with ThreadPoolExecutor(max_workers=worker_count,
                                thread_name_prefix="crop-only-warm") as pool:
            futures = [pool.submit(crop_one, moment) for moment in moments]
            for completed, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"object_failed": 1, "person_failed": 1,
                              "error": str(exc)[:180]}
                for key in crop_totals:
                    if key != "clips":
                        crop_totals[key] += int(result.get(key, 0) or 0)
                if result.get("error"):
                    log(f"crop-only cache issue {result.get('id','')}: {result['error']}")
                if completed % PROGRESS_EVERY == 0 or completed == len(moments):
                    log(f"crop-only cache {completed}/{len(moments)}: "
                        f"object hit/new/failed={crop_totals['object_hit']}/"
                        f"{crop_totals['object_new']}/{crop_totals['object_failed']}, "
                        f"person hit/new/failed={crop_totals['person_hit']}/"
                        f"{crop_totals['person_new']}/{crop_totals['person_failed']}, "
                        f"source mismatch={crop_totals['source_mismatch_or_no_fields']}")
        log(f"{scope} crop-only cache pass complete: {crop_totals}")
        return crop_totals

    if dry_run:
        for moment in moments:
            mid = str(moment.get("id") or "")
            entry = perceiver.cache.get(mid) or {}
            fields = entry.get("fields") or {}
            source_matches = _source_fingerprint_matches(entry, moment) if entry else False
            if not entry:
                totals["perception_new"] += 1
            elif entry.get("prompt_version") != PROMPT_VERSION or not source_matches:
                totals["perception_refreshed"] += 1
            elif entry.get("list_parser_version") != LIST_PARSER_VERSION:
                totals["perception_repaired"] += 1
            else:
                totals["perception_cache_hits"] += 1
                if (_structured_fact_quality_error(fields)
                        or needs_person_observation_review(fields, entry)):
                    totals["perception_quality_warning_hits"] += 1
            labels = fields.get("objects") or []
            video = video_path_of(moment)
            if not labels:
                totals["grounding_not_needed"] += 1
            elif video:
                start, end = clip_range(moment)
                source_key = grounding_cache.source_key(mid, video, start, end, labels)
                source_cached, _ = grounding_cache.get_source(source_key)
                if source_cached is not None:
                    totals["grounding_hit"] += 1
                    totals["grounding_source_hit"] += 1
                else:
                    frames = media.sample_visual_candidates(video, start, end,
                                                            source_id=mid, count=7)
                    if frames:
                        key = grounding_cache.key(mid, labels, frames)
                        cached, _ = grounding_cache.get(key)
                        if cached is not None:
                            totals["grounding_hit"] += 1
                            # Inventory mode must remain read-only; source aliases
                            # are written only by the actual prewarm pass.
                        else:
                            totals["grounding_new"] += 1
                    else:
                        totals["grounding_failed"] += 1
                        log(f"object-frame sample unavailable for {mid}")
            else:
                totals["grounding_failed"] += 1
                log(f"video asset unavailable for object grounding: {mid}")
            person_rows = fields.get("person_observations") or []
            if entry.get("prompt_version") != PROMPT_VERSION or not person_rows:
                totals["person_grounding_not_needed"] += 1
            elif video:
                start, end = clip_range(moment)
                frame_times = entry.get("frame_times_s") or []
                person_frames = media.sample_visual_candidates(
                    video, start, end, source_id=mid, count=len(frame_times) or 7,
                    sample_times=frame_times or None)
                person_labels = [
                    "person-crop-v8-qwen-anchor-cache-key-mutual-face-identity: " + str(row.get("entity") or "").strip()
                    + " | " + str(row.get("appearance") or "").strip() + " | anchors=" + ir._person_anchor_cache_suffix(row)
                    for row in person_rows[:8] if isinstance(row, dict)
                ]
                if person_frames and person_labels:
                    person_key = grounding_cache.key(mid, person_labels, person_frames)
                    person_cached, _ = grounding_cache.get(person_key)
                    totals["person_grounding_hit" if person_cached is not None
                           else "person_grounding_new"] += 1
                else:
                    totals["person_grounding_failed"] += 1
            else:
                totals["person_grounding_failed"] += 1
            found_audio = False
            for _, _, audio_data in video_audio_chunks(moment):
                found_audio = True
                key = "whisper-source-audio-v2:" + hashlib.sha256(audio_data).hexdigest()
                totals["audio_hit" if speech_cache.contains(key) else "audio_new"] += 1
            if not found_audio:
                totals["audio_silent"] += 1
        log(f"dry run only; cache inventory: {totals}")
        return totals

    health_rows = []
    for api_url in API_URLS:
        health = requests.get(api_url + "/health", timeout=5).json()
        if not health.get("model_loaded") or not health.get("whisper_loaded"):
            raise RuntimeError(f"Qwen/Whisper API is not fully loaded: {api_url}")
        if not isinstance(health.get("gpu"), dict) or not health["gpu"].get("name"):
            raise RuntimeError(
                f"CUDA is unavailable at {api_url}; refusing to run cache prewarm on CPU"
            )
        health_rows.append((api_url, health))
    log(f"healthy model endpoints: {len(health_rows)}")

    # Repair only source-backed memory gaps. Existing scene/event understanding
    # stays intact; focused video-only person repair avoids re-running the whole scene.
    worker_count = max(1, min(8, len(perceiver.api_urls)))

    def perceive_one(index, moment):
        mid = str(moment.get("id") or "")
        with perceiver._cache_lock:
            entry = perceiver.cache.get(mid)
            entry = json.loads(json.dumps(entry)) if entry else None
        fields = (entry or {}).get("fields") or {}
        had_quality_gap = bool(
            _structured_fact_quality_error(fields)
            or needs_person_observation_review(fields, entry)
        )
        had_current_prompt = bool(entry and entry.get("prompt_version") == PROMPT_VERSION)
        if (entry and fields and had_current_prompt
                and _source_fingerprint_matches(entry, moment) and not had_quality_gap):
            result = dict(entry)
            result["fields"] = dict(fields)
            result["source"] = "cache"
        else:
            # A source-matched row is fast to reuse only when it passes the
            # current content checks. During explicit prewarm, regenerate rows
            # with known gaps instead of reporting a hit with a permanent warning.
            result = resolve_cached_perception(
                perceiver, moment, entry, refresh_quality_gap=had_quality_gap)
        return index, mid, had_quality_gap, bool(entry), had_current_prompt, result

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="perception-warm") as pool:
        futures = {pool.submit(perceive_one, index, moment): index
                   for index, moment in enumerate(moments, 1)}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                index, mid, had_quality_gap, had_entry, had_current_prompt, result = future.result()
            except Exception as exc:
                totals["perception_failed"] += 1
                log(f"perception cache repair failed: {str(exc)[:180]}")
                continue
            if result.get("source") == "error" or not result.get("fields"):
                totals["perception_failed"] += 1
                log(f"perception cache repair failed {mid}: {str(result.get('error') or '')[:500]}")
            else:
                fields = result.get("fields") or {}
                repair_metadata = result.get("quality_repair") or {}
                quality_gap = bool(
                    result.get("quality_warning")
                    or repair_metadata.get("non_blocking_temporal_coverage_warning")
                    or _structured_fact_quality_error(fields)
                    or needs_person_observation_review(fields, result)
                )
                if quality_gap:
                    totals["perception_quality_warning_hits"] += 1
                source = result.get("source")
                if source in {"cache", "cache_quality_warning"}:
                    totals["perception_cache_hits"] += 1
                elif source in {"qwen_repair", "quality_repair", "parser_repair", "cache_person_repair"} or (
                        had_quality_gap and had_current_prompt and source == "qwen"):
                    totals["perception_repaired"] += 1
                elif source == "qwen" and had_entry:
                    totals["perception_refreshed"] += 1
                elif source == "qwen":
                    totals["perception_new"] += 1
                else:
                    totals["perception_failed"] += 1
                    log(f"perception cache repair returned unknown source {mid}: {source}")
            if completed % CACHE_SAVE_EVERY == 0 or completed == len(moments):
                perceiver._save_cache()
            if completed % PROGRESS_EVERY == 0 or completed == len(moments):
                log(
                    f"perception cache pass {completed}/{len(moments)}: "
                    f"hit {totals['perception_cache_hits']}, "
                    f"refreshed {totals['perception_refreshed']}, "
                    f"new {totals['perception_new']}, "
                    f"repaired {totals['perception_repaired']}, "
                    f"quality-warning {totals['perception_quality_warning_hits']}, "
                    f"failed {totals['perception_failed']}"
                )
    perceiver._save_cache()

    # Ground independent clips concurrently across the healthy model endpoint pool.

    def ground_one(index, moment):
        mid = str(moment.get("id") or "")
        entry = perceiver.cache.get(mid) or {}
        fields = entry.get("fields") or {}
        labels = fields.get("objects") or []
        video = video_path_of(moment)
        if not labels or not _source_fingerprint_matches(entry, moment):
            return {"grounding_not_needed": 1}
        if not video:
            return {"grounding_failed": 1, "id": mid, "error": "video asset unavailable"}
        start, end = clip_range(moment)
        source_key = grounding_cache.source_key(mid, video, start, end, labels)
        source_cached, _ = grounding_cache.get_source(source_key)
        if source_cached is not None:
            return {"grounding_hit": 1, "grounding_source_hit": 1}
        frames = media.sample_visual_candidates(video, start, end, source_id=mid, count=7)
        if not frames:
            return {"grounding_failed": 1, "id": mid, "error": "object-frame sample unavailable"}
        key = grounding_cache.key(mid, labels, frames)
        cached, _ = grounding_cache.get(key)
        if cached is not None:
            if source_key:
                grounding_cache.put_source(source_key, cached)
            return {"grounding_hit": 1}
        last_result = {}
        for attempt in range(3):
            try:
                last_result = perceiver.ground_objects(labels, frames)
                if grounding_cache.put(key, last_result):
                    if source_key:
                        grounding_cache.put_source(source_key, last_result)
                    return {"grounding_new": 1}
            except Exception as exc:
                last_result = {"message": str(exc)}
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        return {
            "grounding_failed": 1, "id": mid,
            "error": str(last_result.get("message") or last_result.get("status") or "not cached")[:160],
        }

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ground-warm") as pool:
        futures = {
            pool.submit(ground_one, index, moment): index
            for index, moment in enumerate(moments, 1)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                result = {"grounding_failed": 1, "error": str(exc)[:160]}
            for key in ("grounding_hit", "grounding_source_hit", "grounding_new",
                        "grounding_failed", "grounding_not_needed"):
                totals[key] += int(result.get(key, 0) or 0)
            if result.get("error"):
                log(f"object-frame grounding issue {result.get('id','')}: {result['error']}")
            if completed % PROGRESS_EVERY == 0 or completed == len(moments):
                log(f"object-frame cache {completed}/{len(moments)}: hit {totals['grounding_hit']} "
                    f"(source fast-path {totals['grounding_source_hit']}), new {totals['grounding_new']}, "
                    f"failed {totals['grounding_failed']}")

    # Person grounding shares the persistent grounding cache used by interactive steps.
    def ground_person_one(index, moment):
        mid = str(moment.get("id") or "")
        entry = perceiver.cache.get(mid) or {}
        fields = entry.get("fields") or {}
        rows = [row for row in fields.get("person_observations") or []
                if isinstance(row, dict) and str(row.get("entity") or "").strip()]
        if not rows or not _source_fingerprint_matches(entry, moment):
            return {"person_grounding_not_needed": 1}
        video = video_path_of(moment)
        if not video:
            return {"person_grounding_failed": 1, "id": mid, "error": "video asset unavailable"}
        start, end = clip_range(moment)
        frame_times = entry.get("frame_times_s") or []
        frames = media.sample_visual_candidates(
            video, start, end, source_id=mid, count=len(frame_times) or 7,
            sample_times=frame_times or None)
        labels = ["person-crop-v8-qwen-anchor-cache-key-mutual-face-identity: " + str(row.get("entity") or "").strip()
                  + " | " + str(row.get("appearance") or "").strip() + " | anchors=" + ir._person_anchor_cache_suffix(row) for row in rows[:8]]
        if not frames or not labels:
            return {"person_grounding_failed": 1, "id": mid, "error": "person-frame sample unavailable"}
        face_result = warm_face_cache(mid, video, start, end, frame_times)
        key = grounding_cache.key(mid, labels, frames)
        cached, _ = grounding_cache.get(key)
        if cached is not None:
            return {"person_grounding_hit": 1, **face_result}
        try:
            result = perceiver.ground_people(rows, frames)
            if grounding_cache.put(key, result) and result.get("status") == "completed":
                return {"person_grounding_new": 1, **face_result}
            return {"person_grounding_failed": 1, "id": mid, **face_result,
                    "error": str(result.get("message") or result.get("status") or "not cached")[:160]}
        except Exception as exc:
            return {"person_grounding_failed": 1, "id": mid, **face_result,
                    "error": str(exc)[:160]}

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="person-ground-warm") as pool:
        futures = {pool.submit(ground_person_one, index, moment): index
                   for index, moment in enumerate(moments, 1)}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                result = {"person_grounding_failed": 1, "error": str(exc)[:160]}
            for key in ("person_grounding_hit", "person_grounding_new",
                        "person_grounding_failed", "person_grounding_not_needed",
                        "face_cache_hit", "face_cache_new", "face_cache_failed"):
                totals[key] += int(result.get(key, 0) or 0)
            if result.get("error"):
                log(f"person-frame grounding issue {result.get('id','')}: {result['error']}")
            if completed % PROGRESS_EVERY == 0 or completed == len(moments):
                log(f"person-frame cache {completed}/{len(moments)}: hit {totals['person_grounding_hit']}, "
                    f"new {totals['person_grounding_new']}, failed {totals['person_grounding_failed']}; "
                    f"YOLO face cache hit/new/failed {totals['face_cache_hit']}/"
                    f"{totals['face_cache_new']}/{totals['face_cache_failed']}")

    # Decode and transcribe audio in parallel; each miss is sent round-robin to
    # an API server bound to one GPU. Cache writes remain atomic and serialized.
    def transcribe_one(index, moment):
        found_audio = False
        local = {"audio_hit": 0, "audio_new": 0, "audio_failed": 0, "audio_silent": 0}
        mid = str(moment.get("id") or "")
        for offset, duration, audio_data in video_audio_chunks(moment):
            found_audio = True
            key = "whisper-source-audio-v2:" + hashlib.sha256(audio_data).hexdigest()
            cached = speech_cache.get(key) is not None
            if cached:
                local["audio_hit"] += 1
                continue
            decoded = None
            last_error = None
            for attempt in range(3):
                try:
                    response = _post_with_inference_cancel(
                        perceiver.next_api_url() + "/transcribe",
                        files={"audio": ("source_audio.wav", audio_data, "audio/wav")},
                        timeout=(5, 180),
                    )
                    response.raise_for_status()
                    decoded = response.json()
                    if not isinstance(decoded, dict):
                        raise ValueError("transcription response is not an object")
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
            if decoded is not None:
                _stored, inserted = speech_cache.put_if_absent(key, decoded)
                local["audio_new" if inserted else "audio_hit"] += 1
            else:
                local["audio_failed"] += 1
                local.setdefault("errors", []).append(
                    f"{mid} at {offset:.1f}s: {str(last_error)[:160]}"
                )
        if not found_audio:
            local["audio_silent"] += 1
        return local

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="speech-warm") as pool:
        futures = {
            pool.submit(transcribe_one, index, moment): index
            for index, moment in enumerate(moments, 1)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                result = {"audio_failed": 1, "errors": [str(exc)[:160]]}
            for key in ("audio_hit", "audio_new", "audio_failed", "audio_silent"):
                totals[key] += int(result.get(key, 0) or 0)
            for error in result.get("errors") or []:
                log(f"audio cache issue: {error}")
            if completed % PROGRESS_EVERY == 0 or completed == len(moments):
                log(f"audio transcript cache {completed}/{len(moments)}: hit {totals['audio_hit']}, new {totals['audio_new']}, failed {totals['audio_failed']}, silent {totals['audio_silent']}")
    # Profile enrichment is a separate step in the interactive run path. Warm its
    # cache from the same transcript and sampled frames so Run Step does not issue
    # another model request after perception and transcription have already hit.
    profile_cache = PersistentTranscriptCache(
        RUNS / "interactive_person_profile_cache.json")
    profile_runner = object.__new__(ir.InteractiveSession)
    profile_runner.perceiver = perceiver
    profile_runner.person_profile_cache = profile_cache
    profile_totals = {"hit": 0, "new": 0, "failed": 0, "skipped": 0}

    def transcript_from_cache(moment):
        rows = []
        mid = str(moment.get("id") or "")
        for offset, duration, audio_data in video_audio_chunks(moment):
            key = "whisper-source-audio-v2:" + hashlib.sha256(audio_data).hexdigest()
            decoded = speech_cache.get(key)
            if not isinstance(decoded, dict):
                continue
            language = str(decoded.get("language") or "unknown")[:24]
            segments = decoded.get("segments") or []
            if segments:
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    text = str(segment.get("text") or "").strip()
                    if not text:
                        continue
                    start_s = offset + max(0.0, float(segment.get("start") or 0))
                    end_s = offset + max(float(segment.get("start") or 0),
                                         float(segment.get("end") or 0))
                    rows.append((round(start_s, 2), round(end_s, 2), language, text[:1000]))
            else:
                text = str(decoded.get("full_text") or "").strip()
                if text:
                    rows.append((round(offset, 2), round(offset + duration, 2),
                                 language, text[:1000]))
        return "\n".join(
            f"[audio {start_s:.2f}–{end_s:.2f}s; {language}] {text}"
            for start_s, end_s, language, text in rows
        )[:4000]

    def profile_one(moment):
        mid = str(moment.get("id") or "")
        entry = perceiver.cache.get(mid) or {}
        fields = entry.get("fields") or {}
        people = [row for row in fields.get("person_observations") or []
                  if isinstance(row, dict)]
        if not any(str(row.get("entity") or "").strip() for row in people):
            return {"skipped": 1}
        if not _source_fingerprint_matches(entry, moment):
            return {"skipped": 1}
        speech = transcript_from_cache(moment)
        if not speech:
            return {"skipped": 1}
        video = video_path_of(moment)
        if not video:
            return {"failed": 1, "id": mid, "error": "video asset unavailable"}
        start, end = clip_range(moment)
        frame_times = entry.get("frame_times_s") or []
        frames = media.sample_visual_candidates(
            video, start, end, source_id=mid, count=len(frame_times) or 7,
            sample_times=frame_times or None)
        if not frames:
            return {"failed": 1, "id": mid, "error": "profile frame sampling failed"}
        cache_key = ir.person_profile_cache_key(
            mid, speech, people, frames, fields.get("temporal_facts") or [])
        if profile_cache.get(cache_key) is not None:
            return {"hit": 1}
        # Retry against the round-robin pool if a busy API rejects one request.
        for attempt in range(3):
            profile_runner._enrich_person_profiles_from_speech(
                people, speech, frames, mid,
                visual_facts=fields.get("temporal_facts") or [])
            if profile_cache.get(cache_key) is not None:
                return {"new": 1}
            if attempt < 2:
                time.sleep(1)
        return {"failed": 1, "id": mid, "error": "profile cache was not written"}

    with ThreadPoolExecutor(max_workers=worker_count,
                            thread_name_prefix="profile-warm") as pool:
        futures = [pool.submit(profile_one, moment) for moment in moments]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:
                result = {"failed": 1, "error": str(exc)[:160]}
            for key in profile_totals:
                profile_totals[key] += int(result.get(key, 0) or 0)
            if result.get("error"):
                log(f"person-profile cache issue {result.get('id','')}: {result['error']}")
            if completed % PROGRESS_EVERY == 0 or completed == len(moments):
                log(f"person-profile cache {completed}/{len(moments)}: "
                    f"hit {profile_totals['hit']}, new {profile_totals['new']}, "
                    f"failed {profile_totals['failed']}, skipped {profile_totals['skipped']}")
    totals.update({f"person_profile_{key}": value
                   for key, value in profile_totals.items()})
    log(f"{scope} multimodal cache pass complete: {totals}")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--crop-only", action="store_true",
                        help="warm local HOG/CLIP person/object grounding caches only; never call Qwen or Whisper")
    parser.add_argument("--all-profiles", action="store_true")
    short_group = parser.add_mutually_exclusive_group()
    short_group.add_argument("--short-all", action="store_true",
                             help="include all clips in both short-term sets")
    short_group.add_argument("--short-half", action="store_true",
                             help="evenly sample half of both short-term sets")
    short_group.add_argument("--short-first", type=int, metavar="N",
                             help="warm the first N clips in each short-term set")
    parser.add_argument("--custom-all", action="store_true",
                        help="include every custom test-set clip")
    parser.add_argument("--long-participants", default="",
                        help="comma-separated long-term participant IDs or names")
    args = parser.parse_args()
    long_participants = [x.strip() for x in args.long_participants.split(",") if x.strip()]
    if args.short_first is not None and args.short_first < 1:
        parser.error("--short-first must be at least 1")
    run(limit=args.limit, dry_run=args.dry_run, all_profiles=args.all_profiles,
        short_half=args.short_half, custom_all=args.custom_all,
        long_participants=long_participants, short_first=args.short_first,
        short_all=args.short_all, crop_only=args.crop_only)
