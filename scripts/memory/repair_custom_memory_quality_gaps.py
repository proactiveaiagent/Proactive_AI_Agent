#!/usr/bin/env python3
"""Source-grounded, generic repair for temporal-fact gaps in custom video memories."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "scripts" / "memory"))

from interactive_runner import feishu_test_moments  # noqa: E402
from qwen_perceive import (  # noqa: E402
    PROMPT_VERSION,
    _action_cues_in_text,
    _apply_temporal_fact_repair,
    _build_temporal_fact_coverage_repair_prompt,
    _camera_wearer_reference,
    _object_move_evidence_error,
    _parse_json_array,
    _parse_json_array_prefix,
    _temporal_fact_coverage_error,
    _unresolved_person_reference_error,
    clip_range,
    extract_frames,
    video_path_of,
)

CACHE = ROOT / "data" / "egolife" / "runs" / "qwen_cache_memory_prompt_v3.json"
_GENERIC_PERSON_FACT_SUBJECTS = {
    "person", "the person", "visible person", "people", "travelers",
    "travellers", "person in frame", "anonymous person",
}
API_URLS = [
    value.strip().rstrip("/")
    for value in os.environ.get(
        "QWEN_API_URLS",
        "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004",
    ).split(",")
    if value.strip()
]


def request_images(api_url: str, prompt: str, images: list[bytes], label: str) -> str:
    response = requests.post(
        api_url + "/analyze_video",
        files=[("images", (f"{label}_{index + 1}.jpg", image, "image/jpeg"))
               for index, image in enumerate(images)],
        data={"prompt": prompt, "request_id": uuid.uuid4().hex},
        timeout=900,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:240]}")
    return str(response.json().get("analysis") or "")


def _narrative_claim_audit_prompt(fields: dict, coverage_gap: str) -> str:
    allowed = ("scene", "user_action", "interaction_details", "activity")
    narratives = {key: str(fields.get(key) or "")[:1600] for key in allowed}
    facts = [row for row in fields.get("temporal_facts") or []
             if isinstance(row, dict)][:16]
    return "\n".join((
        "Audit the narrative against ONLY the supplied ordered frames and source-linked temporal facts.",
        "Do not add facts, infer intent, infer an action from its aftermath, or remove any action that is visibly supported.",
        "If a narrated action is not directly supported by these frames, remove only that unsupported action from the affected narrative field. Keep all other supported details and preserve the original wording where possible.",
        "Return JSON only with a 'fields' object containing scene, user_action, interaction_details, and activity. Leave supported fields unchanged. Do not include temporal facts.",
        "Current coverage warning: " + coverage_gap[:500],
        "Narrative fields: " + json.dumps(narratives, ensure_ascii=False),
        "Existing source-linked facts: " + json.dumps(facts, ensure_ascii=False),
    ))


def _validated_narrative_updates(original: dict, candidate: dict, coverage_gap: str):
    if not isinstance(candidate, dict):
        return {}, "model response has no fields object"
    allowed = ("scene", "user_action", "interaction_details", "activity")
    missing_match = re.search(r"missing cues \[(.*?)\]", coverage_gap)
    missing = {value.strip() for value in missing_match.group(1).split(",") if value.strip()} if missing_match else set()
    updates = {}
    for key in allowed:
        if key not in candidate:
            continue
        before = str(original.get(key) or "")
        after = str(candidate.get(key) or "").strip()
        if after == before:
            continue
        if len(after) > max(len(before), 1600) or (before and not after):
            return {}, f"unsafe narrative length change in {key}"
        before_words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", before.casefold())
        after_words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", after.casefold())
        if after_words and sum(word in set(before_words) for word in after_words) / len(after_words) < 0.82:
            return {}, f"rewrite in {key} changed too much source wording"
        before_cues = _action_cues_in_text(before, narrative=True)
        after_cues = _action_cues_in_text(after, narrative=True)
        if after_cues - before_cues:
            return {}, f"rewrite in {key} introduced action cues"
        if (before_cues - after_cues) - missing:
            return {}, f"rewrite in {key} removed cues not listed as unsupported"
        updates[key] = after
    return updates, ""


def _known_person_aliases(fields: dict) -> set[str]:
    aliases = {
        " ".join(str(row.get("entity") or "").casefold().split())
        for row in fields.get("person_observations") or []
        if isinstance(row, dict) and row.get("entity")
    }
    aliases.update(" ".join(str(value or "").casefold().split())
                   for value in fields.get("people") or [] if str(value or "").strip())
    return aliases


def _normalize_missing_subject_types(rows: list, fields: dict) -> list:
    """Fill a missing type only when the subject matches a known same-clip entity."""
    people = _known_person_aliases(fields)
    people.update(" ".join(str(value or "").casefold().split())
                  for value in fields.get("people") or [] if str(value or "").strip())
    objects = {" ".join(str(value or "").casefold().split())
               for value in fields.get("objects") or [] if str(value or "").strip()}
    typed = {}
    for fact in fields.get("temporal_facts") or []:
        if isinstance(fact, dict) and fact.get("subject") and fact.get("subject_type"):
            typed[" ".join(str(fact["subject"]).casefold().split())] = str(
                fact["subject_type"]).casefold()
    location = " ".join(str(fields.get("location") or "").casefold().split())
    output = []
    for original in rows:
        if not isinstance(original, dict):
            output.append(original)
            continue
        row = dict(original)
        if not str(row.get("subject_type") or "").strip():
            subject = " ".join(str(row.get("subject") or "").casefold().split())
            if subject in people or subject in _GENERIC_PERSON_FACT_SUBJECTS or _camera_wearer_reference(subject):
                row["subject_type"] = "person"
            elif subject in objects or typed.get(subject) == "object":
                row["subject_type"] = "object"
            elif location and subject == location:
                row["subject_type"] = "place"
        output.append(row)
    return output


def _keep_source_linked_people(row: dict, aliases: set[str]) -> dict | None:
    """Do not add a new person identity from an event-repair pass."""
    fact = dict(row)
    subject_type = str(fact.get("subject_type") or "").casefold()
    subject = " ".join(str(fact.get("subject") or "").casefold().split())
    if subject_type == "person" and subject in _GENERIC_PERSON_FACT_SUBJECTS:
        # Keep anonymous, visibly supported actions as events, not durable person
        # identities; the identity/reference validator must not invent a person card.
        fact["subject_type"] = "event"
        fact["subject"] = "visible person activity"
        subject_type = "event"
    elif (subject_type == "person" and subject not in aliases
          and not _camera_wearer_reference(subject)):
        return None
    actor = str(fact.get("actor") or "").strip()
    actor_key = " ".join(actor.casefold().split())
    if (actor and actor_key not in aliases
            and not _camera_wearer_reference(actor_key)):
        fact.pop("actor", None)
    affected = fact.get("affected_people")
    if isinstance(affected, list):
        fact["affected_people"] = [
            person for person in affected
            if " ".join(str(person or "").casefold().split()) in aliases
            or _camera_wearer_reference(person)
        ]
    return fact


def _merge_facts(existing: list, additions: list, aliases: set[str]) -> list:
    merged = [dict(row) for row in existing if isinstance(row, dict)]
    seen = {
        tuple(" ".join(str(row.get(key) or "").casefold().split())
              for key in ("subject_type", "subject", "predicate", "value"))
        for row in merged
    }
    for original in additions:
        row = _keep_source_linked_people(original, aliases)
        if row is None:
            continue
        key = tuple(" ".join(str(row.get(name) or "").casefold().split())
                    for name in ("subject_type", "subject", "predicate", "value"))
        if key not in seen and len(merged) < 16:
            merged.append(row)
            seen.add(key)
    return merged


def write_entry(moment_id: str, entry: dict) -> None:
    lock_path = CACHE.with_name(CACHE.name + ".lock")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        disk = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        current = disk.get(moment_id) or {}
        if current.get("prompt_version") != PROMPT_VERSION:
            raise RuntimeError(f"{moment_id}: cache version changed during repair; refusing stale write")
        merged = dict(current)
        merged.update(entry)
        merged["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        disk[moment_id] = merged
        fd, temp_path = tempfile.mkstemp(prefix=CACHE.name + ".", suffix=".tmp",
                                         dir=str(CACHE.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(disk, output, ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, CACHE)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def repair_one(moment: dict, snapshot_entry: dict, api_url: str, passes: int = 2,
               replay_raw_only: bool = False) -> dict:
    moment_id = str(moment.get("id") or "")
    entry = dict(snapshot_entry)
    fields = dict(entry.get("fields") or {})
    before_error = _temporal_fact_coverage_error(fields)
    if not before_error:
        return {"id": moment_id, "status": "already_complete", "facts_added": 0}
    if entry.get("prompt_version") != PROMPT_VERSION:
        return {"id": moment_id, "status": "version_mismatch", "facts_added": 0}

    repair_meta = entry.get("temporal_fact_repair") or {}
    frames = []
    if replay_raw_only:
        frame_count = max(1, int(repair_meta.get("source_frame_count") or 6))
    else:
        video = video_path_of(moment)
        if video is None:
            return {"id": moment_id, "status": "missing_video", "facts_added": 0}
        start, end = clip_range(moment)
        frames = extract_frames(video, start, end, 6)
        if not frames:
            return {"id": moment_id, "status": "no_frames", "facts_added": 0}
        frame_count = len(frames)

    existing = list(fields.get("temporal_facts") or [])
    cached_raw = (entry.get("temporal_fact_repair_raw") or []
                  if repair_meta.get("method") in {
                      "generic_same_video_temporal_coverage_repair_v1",
                      "generic_same_video_temporal_coverage_repair_v2",
                      "generic_same_video_temporal_coverage_repair_v3",
                  }
                  else [])
    raw_responses = []
    raw_reused = 0
    errors = []
    total_added = 0

    def apply_raw(raw: str, pass_label: str) -> bool:
        nonlocal existing, total_added
        gap = _temporal_fact_coverage_error({**fields, "temporal_facts": existing})
        if not gap:
            return True
        room = min(8, max(0, 16 - len(existing)))
        if room <= 0:
            errors.append(f"{pass_label}: 16-fact limit reached before coverage was complete")
            return False
        try:
            try:
                raw_rows = _parse_json_array(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                raw_rows, complete = _parse_json_array_prefix(raw)
                if not raw_rows:
                    raise ValueError("model did not return any complete JSON fact rows")
                # Keep only complete JSON objects before a truncated tail;
                # source/evidence and merge validators still apply to each row.
            raw_rows = _normalize_missing_subject_types(raw_rows[:room], fields)
            additions = _apply_temporal_fact_repair(raw_rows, frame_count)
            if not additions:
                errors.append(f"{pass_label}: no valid source-backed facts returned")
                return False
            previous = existing
            candidate = _merge_facts(existing, additions, _known_person_aliases(fields))
            candidate_fields = {**fields, "temporal_facts": candidate}
            baseline_move_error = _object_move_evidence_error(fields)
            baseline_reference_error = _unresolved_person_reference_error(fields)
            move_error = _object_move_evidence_error(candidate_fields)
            reference_error = _unresolved_person_reference_error(candidate_fields)
            new_move_error = move_error if move_error != baseline_move_error else ""
            new_reference_error = reference_error if reference_error != baseline_reference_error else ""
            if new_move_error or new_reference_error:
                errors.append(f"{pass_label}: rejected unsafe additions: " + "; ".join(
                    value for value in (new_move_error, new_reference_error) if value))
                return False
            existing = candidate
            total_added += max(0, len(existing) - len(previous))
            return True
        except Exception as exc:
            errors.append(f"{pass_label}: {str(exc)[:260]}")
            return False

    for cached_index, raw in enumerate(cached_raw[:max(1, min(2, passes))]):
        if not _temporal_fact_coverage_error({**fields, "temporal_facts": existing}):
            break
        raw_responses.append(str(raw))
        raw_reused += 1
        apply_raw(str(raw), f"cached pass {cached_index + 1}")

    for pass_index in range(max(1, min(2, passes))):
        if replay_raw_only or not _temporal_fact_coverage_error(
                {**fields, "temporal_facts": existing}):
            break
        gap = _temporal_fact_coverage_error({**fields, "temporal_facts": existing})
        if not gap:
            break
        room = min(8, max(0, 16 - len(existing)))
        if room <= 0:
            errors.append("16-fact limit reached before coverage was complete")
            break
        prompt = _build_temporal_fact_coverage_repair_prompt(
            {**fields, "temporal_facts": existing}, len(frames), room
        )
        prompt += (
            "\nReturn grounded facts for the remaining action cues when visible. Use exact person aliases "
            "already present in Person Observations; if a visible person has no alias, use a generic label "
            "such as person or visible person. Do not create a name or infer identity. Preserve "
            "event order; omit an action only when the supplied frames do not support it. A person's "
            "movement may be recorded from visible direction/transition; object movement requires both "
            "before and after location or state. Every JSON row MUST include subject_type, subject, "
            "predicate, value, frame_index, and evidence. subject_type must be exactly person, object, "
            "place, environment, or event; use person only for an existing visual alias, and object only "
            "for an item in the existing object list or typed facts. Output no duplicate frame-by-frame "
            "rows for the same stable state or action."
        )
        try:
            raw = request_images(api_url, prompt, frames, f"facts_{moment_id}_{pass_index + 1}")
            raw_responses.append(raw)
            apply_raw(raw, f"pass {pass_index + 1}")
        except Exception as exc:
            errors.append(f"pass {pass_index + 1}: {str(exc)[:260]}")

    fields["temporal_facts"] = existing
    after_error = _temporal_fact_coverage_error(fields)
    narrative_audit = {"status": "not_needed", "fields_changed": [],
                       "coverage_before": after_error[:500], "errors": []}
    if after_error and not replay_raw_only:
        try:
            audit_raw = request_images(
                api_url, _narrative_claim_audit_prompt(fields, after_error), frames,
                f"narrative_{moment_id}")
            start, end = audit_raw.find("{"), audit_raw.rfind("}")
            if start < 0 or end < start:
                raise ValueError("model did not return a JSON object")
            audit_payload = json.loads(audit_raw[start:end + 1])
            updates, validation_error = _validated_narrative_updates(
                fields, audit_payload.get("fields"), after_error)
            if validation_error:
                narrative_audit["errors"].append(validation_error)
                narrative_audit["status"] = "rejected"
            elif updates:
                before_fields = dict(fields)
                fields.update(updates)
                revised_error = _temporal_fact_coverage_error(fields)
                if len(revised_error) > len(after_error):
                    fields = before_fields
                    narrative_audit["errors"].append("updated narratives increased the coverage gap")
                    narrative_audit["status"] = "rejected"
                else:
                    narrative_audit["status"] = "repaired" if not revised_error else "partial"
                    narrative_audit["fields_changed"] = sorted(updates)
                    after_error = revised_error
            else:
                narrative_audit["status"] = "no_change"
            entry["narrative_claim_repair_raw"] = audit_raw[:12000]
        except Exception as exc:
            narrative_audit["status"] = "failed"
            narrative_audit["errors"].append(str(exc)[:260])
    narrative_audit["coverage_after"] = after_error[:500]
    entry["narrative_claim_repair"] = narrative_audit
    entry["fields"] = fields
    entry["source"] = "cache_temporal_fact_repair"
    entry["temporal_fact_repair"] = {
        "status": "repaired" if not after_error else "partial" if total_added else "unresolved",
        "method": "generic_same_video_temporal_coverage_repair_v3",
        "passes": len(raw_responses),
        "raw_reused": raw_reused,
        "facts_added": total_added,
        "total_facts": len(existing),
        "coverage_before": before_error[:500],
        "coverage_after": after_error[:500],
        "errors": errors[:4],
        "source_frame_count": frame_count,
        "repaired_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if raw_responses:
        entry["temporal_fact_repair_raw"] = raw_responses
    quality = dict(entry.get("quality_repair") or {})
    if after_error:
        quality["non_blocking_temporal_coverage_warning"] = after_error[:500]
        entry["quality_warning"] = after_error[:500]
    else:
        quality.pop("non_blocking_temporal_coverage_warning", None)
        old_warning = str(entry.get("quality_warning") or "")
        if old_warning.startswith("The scene contains "):
            entry.pop("quality_warning", None)
    entry["quality_repair"] = quality or None
    write_entry(moment_id, entry)
    return {
        "id": moment_id,
        "status": entry["temporal_fact_repair"]["status"],
        "facts_added": total_added,
        "facts_total": len(existing),
        "remaining_gap": after_error[:220],
        "errors": errors[:2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list temporal gaps without calling models")
    parser.add_argument("--workers", type=int, default=4, help="concurrent existing Qwen endpoints; max 4")
    parser.add_argument("--passes", type=int, default=2, help="focused repair passes per clip; max 2")
    parser.add_argument("--clip-ids", default="", help="optional comma-separated IDs for diagnosis")
    parser.add_argument("--replay-raw-only", action="store_true",
                        help="reprocess stored repair outputs without calling any model endpoint")
    args = parser.parse_args()

    moments = feishu_test_moments("feishu:all")
    if len(moments) != 35:
        raise RuntimeError(f"expected 35 custom clips, got {len(moments)}")
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    requested = {value.strip() for value in args.clip_ids.split(",") if value.strip()}
    tasks = []
    for moment in moments:
        moment_id = str(moment.get("id") or "")
        entry = cache.get(moment_id) or {}
        fields = entry.get("fields") or {}
        if entry.get("prompt_version") != PROMPT_VERSION:
            continue
        if _temporal_fact_coverage_error(fields) and (not requested or moment_id in requested):
            tasks.append((moment, entry))
    print(f"custom clips={len(moments)} temporal gaps={len(tasks)} prompt={PROMPT_VERSION}", flush=True)
    if args.dry_run:
        for moment, entry in tasks:
            fields = entry.get("fields") or {}
            print(moment.get("id"), "facts=", len(fields.get("temporal_facts") or []),
                  "gap=", _temporal_fact_coverage_error(fields)[:280], flush=True)
        return
    if requested and not tasks:
        raise RuntimeError("none of the requested clips has a current-version temporal coverage gap")
    healthy = []
    if not args.replay_raw_only:
        if not API_URLS:
            raise RuntimeError("no Qwen API endpoints configured")
        for api_url in API_URLS[:4]:
            response = requests.get(api_url + "/health", timeout=8)
            health = response.json()
            if health.get("model_loaded") and health.get("gpu", {}).get("name"):
                healthy.append(api_url)
        if not healthy:
            raise RuntimeError("no healthy GPU-backed Qwen endpoints; refusing repair")
    workers = max(1, min(4, max(1, len(healthy)), max(1, args.workers)))
    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="temporal-cache-repair") as pool:
        futures = {
            pool.submit(repair_one, moment, entry,
                        healthy[index % len(healthy)] if healthy else "",
                        args.passes, args.replay_raw_only): moment
            for index, (moment, entry) in enumerate(tasks)
        }
        for future in as_completed(futures):
            moment = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"id": moment.get("id"), "status": "failed", "error": str(exc)[:400]}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    completed = sum(row.get("status") == "repaired" for row in results)
    partial = sum(row.get("status") == "partial" for row in results)
    failed = sum(row.get("status") == "failed" for row in results)
    print(f"completed={completed} partial={partial} failed={failed} attempted={len(results)}", flush=True)


if __name__ == "__main__":
    main()
