"""
interactive_runner.py — 交互式步进执行器
======================================================================
把 run_scenarios.py 的一次性流程拆成一步一步，供可视化网站交互执行：
人可以看到当前输入的是哪个视频、时间点，手工点击「运行」再走下一步。

短期场景：每条视频片段一个 add 步骤 → query / update / highlight / delete。
长期场景：白天按小时批量写入 → NightlyProcessor 汇总 / 分类 / 画像更新。

被 viz_server.py 调用（单线程 HTTPServer，无需加锁）：
    InteractiveSession("short_term")   创建会话
    session.state()                    当前状态（下一步信息 + 已执行日志 + 最新快照）
    session.run_step()                 执行当前步，返回结果消息
"""

import copy
import hashlib
import json
import re
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import (PersonMemory, _build_entity_visuals, _portable_video_id,
                    entity_graph_summary, normalize_object_transition_facts,
                    normalize_user_facing_memory_copy)  # noqa: E402
from entity_records import memory_facts
from daytime_processor import DaytimeProcessor  # noqa: E402
from nighttime_processor import NightlyProcessor  # noqa: E402
from multimodal_memory import (CAPTURE_VERSION, MultimodalAssetStore, _run_ffmpeg,
                                media_refs, strip_audio_assets)  # noqa: E402
import run_scenarios as rs               # noqa: E402  复用 snapshot / 夜间整理辅助函数
from qwen_perceive import (QwenPerceiver, golden_eval, clip_range, video_path_of,
                           PROMPT_VERSION, LIST_PARSER_VERSION,
                           _organize_person_entity_aliases,
                           _retain_episode_objects, extract_frames)   # noqa: E402
from advice_cache import PersistentAdviceCache  # noqa: E402
from grounding_cache import (PersistentObjectGroundingCache,
                             PERSON_GROUNDING_IMPLEMENTATION_FINGERPRINT)  # noqa: E402
from speech_identity import (attach_spoken_identity,
                             person_labels_from_observations)  # noqa: E402
from speech_facts import (_AUDIO_EVIDENCE_LINE_RE,
                          _extract_explicit_speech_facts,
                          _summarize_explicit_speech_facts)  # noqa: E402
from audio_transcript_cache import PersistentTranscriptCache  # noqa: E402
from transcript_quality import is_meaningful_transcript_text  # noqa: E402
from object_crop_policy import (evaluate_object_image_policy, filter_person_entities_from_objects, select_object_crop_candidates)  # noqa: E402
from face_detection import (detect_faces, assign_faces_to_people,
                            expanded_face_sample_times)  # noqa: E402
from clip_crop_grounding import resolve_face_gallery_candidates  # noqa: E402
from face_identity import link_records as _link_face_identity_records  # noqa: E402
from scene_retrieval import PostWriteRetrievalMixin  # noqa: E402
from clinical_grounding import ground_clinical_visit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "data" / "egolife" / "testset"
RUNS = ROOT / "data" / "egolife" / "runs"
VIDEO_ROOT = (ROOT / "data" / "egolife").resolve()

_AGE_CLAIM_RE = re.compile(
    r"\b(?:(?P<article>a|an|the)\s+)?"
    r"(?:(?:elderly|older|younger|young|old|middle[- ]aged|senior)\s+"
    r"(?P<qualified>woman|man|person|adult|individual|child|children|boy|girl)|"
    r"(?P<noun>baby|infant|toddler|newborn|child|children|boy|girl|adult))\b",
    re.I,
)


PERSON_PROFILE_CACHE_VERSION = "source-grounded-role-inference-prompt-v44-english-profile-final-validation-2026-11-06"
CONTENT_SUMMARY_VERSION = "source-grounded-audio-integrated-gist-v38-2026-10-25"


def _summary_sentences(value, limit=700):
    """Normalize a short event summary into readable, de-duplicated sentences."""
    text = " ".join(str(value or "").split()).strip()
    text = re.sub(r"\b(?:content\s+)?summary\s*:\s*", "", text, flags=re.I)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\s*[;；]\s*", text)
    rows = []
    seen = set()
    for part in parts:
        sentence = part.strip(" .;；\t")
        if not sentence:
            continue
        key = re.sub(r"[^\w\u4e00-\u9fff]+", " ", sentence.casefold()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(sentence)
    return rows


def _compose_event_summary(visual_summary, speech_summary, max_chars=700):
    """Keep the visible event and add distinct, source-grounded speech context."""
    visual = _summary_sentences(visual_summary)
    spoken = _summary_sentences(speech_summary)
    if not visual:
        visual, spoken = spoken, []
    selected = list(visual[:3])
    selected_keys = [set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", row.casefold()))
                     for row in selected]
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "with",
            "is", "was", "are", "were", "then", "they", "he", "she", "it", "this",
            "that", "who", "which", "their", "his", "her", "as", "by", "from", "for"}
    for sentence in spoken:
        words = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", sentence.casefold())) - stop
        if not words:
            continue
        duplicate = False
        for existing, existing_words in zip(selected, selected_keys):
            if sentence.casefold() in existing.casefold() or existing.casefold() in sentence.casefold():
                duplicate = True
                break
            shared = len(words & existing_words)
            if min(len(words), len(existing_words)) >= 3 and shared / min(len(words), len(existing_words)) >= 0.78:
                duplicate = True
                break
        if not duplicate:
            selected.append(sentence)
            selected_keys.append(words)
        if len(selected) >= 5:
            break
    result = " ".join(row.rstrip(" .") + "." for row in selected).strip()
    return result[:max_chars].rstrip()




def _content_summary_for_fields(fields, speech_context="", prior=None):
    """Always provide one concise, source-grounded situation gist for a memory."""
    fields = fields if isinstance(fields, dict) else {}
    prior = prior if isinstance(prior, dict) else {}
    for candidate in (speech_context, fields.get("content_summary"), fields.get("activity"), fields.get("scene"),
                      prior.get("content_summary"), prior.get("activity"), prior.get("scene")):
        sentences = _summary_sentences(candidate, limit=500)
        if sentences:
            return sentences[0][:500]
    location = str(fields.get("location") or prior.get("location") or "").strip().rstrip(" .")
    return (f"A video scene at {location}." if location else "")[:500]


def _person_anchor_cache_suffix(row):
    payload = {}
    # Source-reference wrappers are attached after perception and differ
    # between runtime and cache warming. Cache identity only on the semantic
    # anchor frame index; sampled frame bytes already fingerprint the evidence.
    for key in ("frame_index",):
        value = row.get(key) if isinstance(row, dict) else None
        if value is not None:
            payload[key] = value
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _person_evidence_anchor_index(row, frame_count):
    """Prefer the last explicitly observed frame over a conflicting parser index.

    Later frame mentions often describe the clearest visible moment, such as a
    silhouette becoming fully visible after first appearing in the background.
    """
    evidence = str((row or {}).get("evidence") or "")
    indices = [int(value) for value in re.findall(r"\bframes?\s*#?\s*(\d+)\b", evidence, re.I)]
    if not indices:
        return None
    index = indices[-1]
    return index if 1 <= index <= int(frame_count or 0) else None


def _sample_person_gallery_inputs(media_store, video_path, clip_start, clip_end,
                                 source_id, qwen_frames, max_samples=36):
    """Keep Qwen anchors first, then add spaced source frames for gallery matching."""
    frames = list(qwen_frames or [])
    anchor_times = [float(row.get("at_s")) for row in frames
                    if isinstance(row, dict) and row.get("at_s") is not None]
    if not video_path or not Path(video_path).is_file():
        return frames
    dense_times = expanded_face_sample_times(
        clip_start, clip_end, anchor_times, max_samples=max_samples)
    extra_times = [at for at in dense_times
                   if not any(abs(at - anchor) <= 0.20 for anchor in anchor_times)]
    if extra_times:
        extra_frames = media_store.sample_visual_candidates(
            video_path, clip_start, clip_end, source_id=source_id,
            count=len(extra_times), sample_times=extra_times)
        frames.extend(extra_frames or [])
    return frames


def _grounded_face_assignments(items, lowres_frames, highres_frames, detections):
    """Map Qwen-anchored people to faces without guessing in group frames.

    Prefer a mutual, unique match between a person proposal and a detected face.
    If proposal geometry misses, allow a direct one-to-one association only when
    exactly one Qwen-anchored person and exactly one confident face occur at that
    anchor frame. This handles close-ups while keeping multi-person frames out
    of identity galleries.
    """
    grouped = {}
    anchored_by_frame = {}
    for item_index, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        anchors = set()
        for raw_index in item.get("anchor_frame_indices") or []:
            try:
                frame_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 1 <= frame_index <= len(lowres_frames):
                anchors.add(frame_index)
                anchored_by_frame.setdefault(frame_index - 1, set()).add(item_index)
        candidate_frames = set()
        for raw_index in item.get("identity_candidate_frame_indices") or []:
            try:
                frame_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 1 <= frame_index <= len(lowres_frames):
                candidate_frames.add(frame_index)
        eligible_frames = anchors | candidate_frames
        for row in item.get("frame_boxes") or []:
            if not isinstance(row, dict):
                continue
            try:
                frame_index = int(row.get("frame_index"))
                bbox = row.get("bbox")
                if (frame_index not in eligible_frames or not isinstance(bbox, list)
                        or len(bbox) != 4):
                    continue
                low_time = float(lowres_frames[frame_index - 1].get("at_s") or 0)
                high_index = min(range(len(highres_frames)), key=lambda j: abs(
                    float(highres_frames[j].get("at_s") or 0) - low_time))
            except (TypeError, ValueError):
                continue
            score = float(row.get("score") or item.get("score") or 0)
            previous = grouped.setdefault(high_index, {}).get(item_index)
            if previous is None or score > previous[1]:
                grouped[high_index][item_index] = (bbox, score)

    assignments = {}
    for high_index, people in grouped.items():
        faces = detections[high_index] if high_index < len(detections) else []
        unique = assign_faces_to_people(
            [{"key": index, "bbox": value[0]} for index, value in people.items()],
            faces, min_confidence=0.05)
        for item_index, face in unique.items():
            assignments[(item_index, high_index)] = {
                **face, "_identity_assignment": "mutual_unique_face_geometry"}

    # A sole face on a sole Qwen-anchored person frame is enough to crop that
    # face directly. Appearance-propagated frames never use this shortcut;
    # they need the person-box/face mutual geometry match above.
    for low_index, candidate_indices in anchored_by_frame.items():
        if len(candidate_indices) != 1 or not highres_frames:
            continue
        low_time = float(lowres_frames[low_index].get("at_s") or 0)
        high_index = min(range(len(highres_frames)), key=lambda j: abs(
            float(highres_frames[j].get("at_s") or 0) - low_time))
        if abs(float(highres_frames[high_index].get("at_s") or 0) - low_time) > 0.35:
            continue
        item_index = next(iter(candidate_indices))
        if (item_index, high_index) in assignments:
            continue
        faces = detections[high_index] if high_index < len(detections) else []
        confident_faces = [face for face in faces if isinstance(face, dict)
                           and float(face.get("confidence") or 0) >= 0.25
                           and isinstance(face.get("bbox"), list)
                           and len(face["bbox"]) == 4]
        # A prominent Qwen-anchored subject can be accompanied by a tiny
        # background face detection. Ignore that detection only when the
        # anchored main face is high-confidence and overwhelmingly larger;
        # similarly sized faces remain ambiguous and are not assigned.
        if len(confident_faces) > 1:
            areas = []
            for face in confident_faces:
                box = face["bbox"]
                areas.append(max(0.0, float(box[2])-float(box[0])) *
                             max(0.0, float(box[3])-float(box[1])))
            largest_index = max(range(len(areas)), key=areas.__getitem__)
            largest_area = areas[largest_index]
            second_area = max((area for index, area in enumerate(areas)
                               if index != largest_index), default=0.0)
            if (largest_area >= 0.004 and second_area > 0
                    and largest_area >= 8.0 * second_area
                    and float(confident_faces[largest_index].get("confidence") or 0) >= 0.70):
                confident_faces = [confident_faces[largest_index]]
        if len(confident_faces) == 1:
            assignments[(item_index, high_index)] = {
                **confident_faces[0],
                "_identity_assignment": "single_person_single_face_qwen_anchor"}
    return assignments

def person_profile_cache_key(source_segment_id, transcript, observations, frames,
                              visual_facts=None, document_evidence=None,
                              historical_people=None, visual_context=None,
                              related_memories=None):
    """Build one shared cache key for runtime enrichment and cache prewarming."""
    speech = str(transcript or "").strip()[:4000]
    image_frames = [frame for frame in (frames or [])
                    if isinstance(frame, dict) and frame.get("bytes")][:7]
    payload = {
        # Invalidate pre-versioned profile cache entries that may have cached
        # generic/empty labels and otherwise bypass current speech-role logic.
        "profile_cache_version": "person-profile-v6-evidence-calibrated-role-and-context-label-2026-11-06",
        "source": str(source_segment_id or ""),
        "transcript": hashlib.sha256(speech.encode("utf-8")).hexdigest(),
        "people": [(str(row.get("entity") or ""), str(row.get("appearance") or ""),
                    str(row.get("face_features") or ""), str(row.get("apparent_age_range") or ""),
                    str(row.get("apparent_age_basis") or ""))
                   for row in (observations or []) if isinstance(row, dict)],
        "frame_times": [round(float(frame.get("at_s") or 0), 3)
                        for frame in image_frames],
        "frame_sha256": [
            hashlib.sha256(bytes(frame.get("bytes") or b"")).hexdigest()
            for frame in image_frames
        ],
        "visual_context": {
            str(key): str(value or "")[:1200]
            for key, value in (visual_context or {}).items()
            if key in {"scene", "content_summary", "activity", "interaction_details", "location", "environments"}
        },
        "visual_facts": [
            {key: row.get(key) for key in ("actor", "subject", "predicate", "value",
                                            "affected_people", "frame_index", "evidence")}
            for row in (visual_facts or []) if isinstance(row, dict)
        ][:32],
        "documents": [
            {key: row.get(key) for key in ("document", "text", "frame_index", "evidence")}
            for row in (document_evidence or []) if isinstance(row, dict)
        ][:8],
        "historical_people": [
            {key: row.get(key) for key in
             ("name", "roles", "profile_context", "source_segment_id", "reference_image_sha256")}
            for row in (historical_people or []) if isinstance(row, dict)
        ][:4],
        "related_memories": [
            {key: row.get(key) for key in
             ("id", "scene", "summary", "people", "location", "activity", "objects", "source_provenance")}
            for row in (related_memories or []) if isinstance(row, dict)
        ][:5],
        "version": PERSON_PROFILE_CACHE_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True)
                          .encode("utf-8")).hexdigest()


def _normalized_evidence_quote(value):
    return "".join(str(value or "").casefold().split()).strip(
        "，。！？、,.!?;；:：\"'“”‘’")


def _collect_document_evidence(document_evidence, visual_facts):
    """Collect exact, frame-linked document text from the visual pass."""
    rows = []
    seen = set()
    sources = [(row, True) for row in (document_evidence or [])]
    sources.extend((row, False) for row in (visual_facts or [])
                   if isinstance(row, dict)
                   and str(row.get("predicate") or "").casefold() == "documented")
    by_key = {}
    for row, is_document_row in sources:
        if not isinstance(row, dict):
            continue
        document = " ".join(str(row.get("document") or row.get("subject") or "").split())[:100]
        text = " ".join(str(row.get("text") or row.get("value") or "").split())[:600]
        if text.startswith(f"Visible {document} text:"):
            text = text[len(f"Visible {document} text:"):].strip()
        evidence = " ".join(str(row.get("evidence") or "").split())[:240]
        key = (document.casefold(), text.casefold(), row.get("frame_index"))
        if not document or not text:
            continue
        if key in seen:
            existing = rows[by_key[key]]
            if not existing.get("actor") and row.get("actor"):
                existing["actor"] = str(row.get("actor"))[:100]
            if not existing.get("evidence") and evidence:
                existing["evidence"] = evidence
            continue
        seen.add(key)
        clean = {"document": document, "text": text, "evidence": evidence}
        if row.get("actor"):
            clean["actor"] = str(row.get("actor"))[:100]
        if isinstance(row.get("frame_index"), int) and not isinstance(row.get("frame_index"), bool):
            clean["frame_index"] = row["frame_index"]
        if not is_document_row and not clean.get("frame_index"):
            continue
        by_key[key] = len(rows)
        rows.append(clean)
    return rows[:8]


def _merge_document_evidence_rows(primary, fallback):
    """Retain source-grounded document reads when a same-source refresh omits OCR."""
    merged = []
    by_key = {}
    for row in list(primary or []) + list(fallback or []):
        if not isinstance(row, dict):
            continue
        document = " ".join(str(row.get("document") or row.get("subject") or "").split())
        text = " ".join(str(row.get("text") or row.get("value") or "").split())
        # Old cached multimodal records may still carry routine shelf price tags.
        # Do not let fallback merging resurrect them as document/object cards.
        if re.search(r"(?i)\b(?:price|shelf)\s*(?:tag|label)\b|价签|价格标签", document):
            continue
        if not document or not text:
            continue
        frame_index = row.get("frame_index")
        key = (document.casefold(), text.casefold(), frame_index)
        if key in by_key:
            current = merged[by_key[key]]
            for name in ("evidence", "actor", "evidence_ref", "modality", "frame_time_s"):
                if not current.get(name) and row.get(name):
                    current[name] = row[name]
            continue
        clean = dict(row)
        clean["document"] = document[:100]
        clean["text"] = text[:600]
        by_key[key] = len(merged)
        merged.append(clean)
    return merged[:8]


def _link_document_fields_to_people(observations, documents, visual_facts,
                                    source_segment_id, frame_times):
    """Attach document fields to a person only with a frame-grounded physical link."""
    rows = [dict(row) for row in observations or [] if isinstance(row, dict)]
    by_label = {}
    for index, row in enumerate(rows):
        label = " ".join(str(row.get("entity") or "").split()).casefold()
        if label:
            by_label.setdefault(label, []).append(index)
        by_label[f"person_{index + 1}"] = [index]
    for document in documents or []:
        if not isinstance(document, dict):
            continue
        try:
            frame_index = int(document.get("frame_index"))
        except (TypeError, ValueError):
            continue
        if frame_index < 1:
            continue
        text = " ".join(str(document.get("text") or "").split())[:600]
        name_match = re.search(
            r"(?i)(?:full\s+name|name)\s*[:#=\-]?\s*"
            r"([^,;|]+?)(?=\s+(?:date\s+of\s+birth|birth\s+date|dob|age)\b|[,;|]|$)", text)
        dob_match = re.search(
            r"(?i)(?:date\s+of\s+birth|birth\s+date|dob)\s*[:#=\-]?\s*"
            r"([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|"
            r"[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4}|"
            r"[0-9]{4}[-/]\s*[0-9]{1,2}[-/]\s*[0-9]{1,2})", text)
        name = " ".join(name_match.group(1).split()).strip() if name_match else ""
        dob = " ".join(dob_match.group(1).split()).strip() if dob_match else ""
        document_context = " ".join((str(document.get("document") or ""), text)).casefold()
        # A medical record describes its patient; the doctor's act of holding it
        # identifies the presenter only. Resolve patient identity after speech
        # and addressee evidence are available instead of naming the clinician.
        if re.search(r"\b(?:medical|patient|diagnosis|clinical|hospital)\b|(?:病历|患者|诊断|医院)",
                     document_context, re.I):
            continue
        linked_indices = set()
        actor = " ".join(str(document.get("actor") or "").split()).casefold()
        if actor:
            linked_indices.update(by_label.get(actor, []))
        for fact in visual_facts or []:
            if not isinstance(fact, dict) or fact.get("frame_index") != frame_index:
                continue
            if str(fact.get("predicate") or "").casefold() != "documented":
                continue
            fact_actor = " ".join(str(fact.get("actor") or "").split()).casefold()
            if fact_actor:
                linked_indices.update(by_label.get(fact_actor, []))
        evidence = " ".join(str(document.get("evidence") or "").split()).casefold()
        physical_link = bool(re.search(
            r"\b(?:held|holding|presented|presenting|handed|receiving|received)\b|"
            r"(?:手持|拿着|递给|递出|展示|交给)", evidence))
        # A profile anchored on the same frame and explicitly describing a
        # document interaction is a direct link even when the visual parser
        # and profile parser use different labels for the same person.
        if not linked_indices and physical_link:
            direct_rows = []
            for index, row in enumerate(rows):
                try:
                    same_frame = int(row.get("frame_index")) == frame_index
                except (TypeError, ValueError):
                    same_frame = False
                row_context = " ".join((
                    str(row.get("evidence") or ""),
                    str(row.get("appearance") or ""),
                )).casefold()
                mentions_document = bool(re.search(
                    r"\b(?:card|document|identity|id|passport|receipt|bill|ticket|screen)\b|"
                    r"(?:卡片|证件|身份证|护照|收据|账单|票据|屏幕)", row_context))
                if same_frame and mentions_document:
                    direct_rows.append(index)
            if len(direct_rows) == 1:
                linked_indices.add(direct_rows[0])
        # A generic physical-link description can identify the sole visible
        # candidate, but never chooses among multiple people by proximity alone.
        if not linked_indices and physical_link and len(rows) == 1:
            linked_indices.add(0)
        if len(linked_indices) != 1:
            continue
        target = rows[next(iter(linked_indices))]
        if not name and not dob:
            continue
        # Prefer an exact printed name for an anonymous/generic visual label.
        current = " ".join(str(target.get("entity") or "").split()).strip()
        generic = bool(re.fullmatch(r"(?i)(?:person|man|woman|individual)(?:\s+\d+)?", current))
        if name and (not current or generic):
            target["entity"] = name[:100]
        if dob:
            target["date_of_birth"] = dob[:40]
        fields = []
        if name:
            fields.append(f"Name: {name}")
        if dob:
            fields.append(f"Date of birth: {dob}")
        note = ("Presented document shows " + "; ".join(fields)
                + ". The printed details are source claims, not independently verified.")
        current_context = " ".join(str(target.get("profile_context") or "").split()).strip()
        if note.casefold() not in current_context.casefold():
            target["profile_context"] = (current_context + " " + note).strip()[:700]
        target["profile_confidence"] = "medium"
        target["evidence"] = (str(target.get("evidence") or "").strip()
                              + f"; linked to {document.get('document') or 'document'} shown in frame {frame_index}")[:500]
        ref = {"source_segment_id": str(source_segment_id or "")[:128],
               "modality": "document", "document": str(document.get("document") or "")[:100],
               "quote": "; ".join(fields)[:300], "frame_index": frame_index}
        if frame_index <= len(frame_times):
            ref["frame_time_s"] = round(float(frame_times[frame_index - 1]), 3)
        refs = list(target.get("evidence_refs") or [])
        if ref not in refs:
            refs.append(ref)
        target["evidence_refs"] = refs[:16]
        target.setdefault("evidence_ref", ref)
    return rows


def _validate_reconciled_facts(proposed, observations, transcript, documents,
                               frame_times, source_segment_id, dialogue_turns=None,
                               max_facts=16):
    """Keep only model-proposed facts with verifiable audio, document, or frame refs."""
    lines = []
    for line in str(transcript or "").splitlines():
        match = _AUDIO_EVIDENCE_LINE_RE.match(line.strip())
        if not match:
            continue
        try:
            start_s = float(match.group("start"))
            end_s = float(match.group("end"))
        except (TypeError, ValueError):
            continue
        lines.append((str(match.group("text") or "").strip(), start_s, end_s))
    candidate_observations = {f"person_{index + 1}": row
                              for index, row in enumerate(observations or [])
                              if isinstance(row, dict) and index < 16}
    candidates = {key: str(row.get("entity") or "").strip()
                  for key, row in candidate_observations.items()}

    def map_person(value):
        raw = " ".join(str(value or "").split()).strip()
        return candidates.get(raw, raw)

    def map_person_text(value):
        text = " ".join(str(value or "").split()).strip()
        for candidate_id, label in candidates.items():
            if label and label != candidate_id:
                text = re.sub(r"(?<![\w])" + re.escape(candidate_id) + r"(?![\w])",
                              lambda _match, replacement=label: replacement,
                              text, flags=re.I)
        return text

    valid = []
    seen = set()
    for row in proposed or []:
        if not isinstance(row, dict):
            continue
        audio_matches = []
        for quote in row.get("evidence_quotes") or []:
            norm = _normalized_evidence_quote(quote)
            if not norm:
                continue
            for text, start_s, end_s in lines:
                line_norm = _normalized_evidence_quote(text)
                if norm in line_norm or line_norm in norm:
                    audio_matches.append((text, start_s, end_s))
                    break
        document_matches = []
        quotes = row.get("document_quotes") or []
        if isinstance(row.get("document_quote"), str):
            quotes = list(quotes) + [row.get("document_quote")]
        for quote in quotes:
            norm = _normalized_evidence_quote(quote)
            if not norm:
                continue
            for doc in documents or []:
                text_norm = _normalized_evidence_quote(doc.get("text"))
                if norm in text_norm or text_norm in norm:
                    document_matches.append((str(quote).strip(), doc))
                    break
        frame_indices = row.get("frame_indices") or []
        if isinstance(row.get("frame_index"), int):
            frame_indices = list(frame_indices) + [row.get("frame_index")]
        frame_indices = list(dict.fromkeys(
            value for value in frame_indices
            if isinstance(value, int) and not isinstance(value, bool)
            and 1 <= value <= len(frame_times)))
        # A frame citation is only accepted when it corresponds to a document or
        # visible fact actually supplied to this reconciliation call.
        cited_frames = {doc.get("frame_index") for _quote, doc in document_matches
                        if isinstance(doc.get("frame_index"), int)}
        cited_frames.update(frame_indices)
        if not audio_matches and not document_matches and not cited_frames:
            continue
        subject_type = str(row.get("subject_type") or "event").strip().casefold()
        if subject_type not in {"person", "object", "place", "environment", "event"}:
            subject_type = "event"
        raw_subject = " ".join(str(row.get("subject") or "event").split()).strip()
        linked_document_matches = document_matches
        source_person = candidate_observations.get(raw_subject)
        if subject_type == "person" and source_person is not None:
            source_frames = {value for value in row.get("frame_indices") or []
                             if isinstance(value, int) and not isinstance(value, bool)
                             and 1 <= value <= len(frame_times)}
            source_frame = source_person.get("frame_index")
            if isinstance(source_frame, int) and 1 <= source_frame <= len(frame_times):
                source_frames.add(source_frame)
            linked_document_matches = [
                (quote, doc) for quote, doc in document_matches
                if isinstance(doc.get("frame_index"), int)
                and doc.get("frame_index") in source_frames
            ]
        if (subject_type == "person"
                and re.search(r"\b(?:date[_ -]?of[_ -]?birth|birth[_ -]?date|dob)\b",
                              str(row.get("predicate") or ""), re.I)
                and not linked_document_matches):
            continue
        # A document attribute is not assigned to a person unless the source frame
        # directly anchors both that person and the document.
        if subject_type == "person" and document_matches and not linked_document_matches:
            document_matches = []
        else:
            document_matches = linked_document_matches
        subject = map_person(raw_subject)
        actor = map_person(row.get("actor") or "unknown")
        affected = list(dict.fromkeys(map_person(value) for value in row.get("affected_people") or []
                                         if str(value or "").strip()))[:8]
        turn = next((item for item in dialogue_turns or []
                     if any(_normalized_evidence_quote(text) in _normalized_evidence_quote(item.get("quote"))
                            or _normalized_evidence_quote(item.get("quote")) in _normalized_evidence_quote(text)
                            for text, _start, _end in audio_matches)), None)
        turn_speaker = "unknown"
        if turn:
            turn_speaker = map_person(turn.get("speaker") or "unknown")
            if turn_speaker != "unknown":
                actor = turn_speaker
        if audio_matches:
            # A dialogue turn is not enough to identify a visible speaker. Keep a
            # proposed candidate attribution only when the validated turn has
            # time-aligned multi-frame support; otherwise preserve the utterance
            # with an unknown speaker. Do not fan a turn's single addressee out to
            # every fact extracted from a sentence containing multiple people.
            self_reference = re.compile(
                r"(?i)(?<![\w])(?:i|i'm|i am|me|my|mine|we|we're|we are|us|our|ours)(?![\w])"
                r"|我(?:们)?(?:是|要|需要|的|在|刚|想)?"
            )
            has_self_reference = any(self_reference.search(text)
                                     for text, _start, _end in audio_matches)
            actor_is_visible_candidate = actor.casefold() in {
                value.casefold() for value in candidates.values() if value
            }
            if actor_is_visible_candidate and actor.casefold() != turn_speaker.casefold():
                actor = "unknown"
            elif actor.casefold() == "user" and not has_self_reference and turn_speaker != "user":
                actor = "unknown"
        affected = list(dict.fromkeys(affected))[:8]
        predicate = map_person_text(row.get("predicate") or "other")[:80]
        value = map_person_text(row.get("value") or "")[:500]
        if not value:
            continue
        confidence = str(row.get("confidence") or "medium").casefold()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        if actor == "user" and not audio_matches:
            confidence = "low"
        refs = []
        for text, start_s, end_s in audio_matches[:4]:
            refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                         "modality": "speech", "audio_start_s": round(start_s, 3),
                         "audio_end_s": round(end_s, 3), "quote": text[:300]})
        for quote, doc in document_matches[:4]:
            ref = {"source_segment_id": str(source_segment_id or "")[:128],
                   "modality": "document", "document": doc.get("document", "")[:100],
                   "quote": quote[:300]}
            frame_index = doc.get("frame_index")
            if isinstance(frame_index, int) and 1 <= frame_index <= len(frame_times):
                ref["frame_index"] = frame_index
                ref["frame_time_s"] = round(float(frame_times[frame_index - 1]), 3)
            refs.append(ref)
        for frame_index in frame_indices[:4]:
            refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                         "modality": "visual", "frame_index": frame_index,
                         "frame_time_s": round(float(frame_times[frame_index - 1]), 3)})
        key = (subject_type, subject.casefold(), predicate.casefold(), value.casefold())
        if key in seen:
            continue
        seen.add(key)
        fact = {"subject_type": subject_type, "subject": subject,
                "predicate": predicate, "value": value, "actor": actor,
                "affected_people": affected, "status": str(row.get("status") or "inferred")[:40],
                "confidence": confidence,
                "modality": ("multimodal" if len({ref.get("modality") for ref in refs}) > 1
                             else (refs[0].get("modality") if refs else "visual")),
                "evidence": "; ".join(dict.fromkeys(
                    [text for text, _start, _end in audio_matches[:2]]
                    + [quote for quote, _doc in document_matches[:2]]) )[:600],
                "evidence_refs": refs[:12]}
        if refs:
            fact["evidence_ref"] = refs[0]
        if isinstance(row.get("time_order"), int) and not isinstance(row.get("time_order"), bool):
            fact["time_order"] = row["time_order"]
        valid.append(fact)
    return valid[:max(0, int(max_facts))]


def _stable_person_observation_key(row):
    """Return evidence-backed anchor identity independent of list order and display name."""
    if not isinstance(row, dict):
        return ""
    explicit_id = str(row.get("observation_id") or "").strip()
    if explicit_id:
        return "id:" + explicit_id
    anchors = {}
    for key in ("frame_index", "frame_time_s", "evidence_ref", "evidence_refs"):
        value = row.get(key)
        if value is not None:
            anchors[key] = value
    if anchors:
        return "anchor:" + json.dumps(
            anchors, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    evidence = " ".join(str(row.get("evidence") or "").casefold().split())
    face = " ".join(str(row.get("face_features") or "").casefold().split())
    appearance = " ".join(str(row.get("appearance") or "").casefold().split())
    if not evidence and not face and not appearance:
        return ""
    fallback = {"evidence": evidence[:240], "face_features": face[:240],
                "appearance": appearance[:240]}
    return "fallback:" + json.dumps(
        fallback, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _person_label_mapping_by_anchor(before, after):
    """Map labels only for unique stable observations; never rely on model row ordering."""
    old_groups, new_groups = {}, {}
    for side, rows in ((old_groups, before or []), (new_groups, after or [])):
        for row in rows:
            key = _stable_person_observation_key(row)
            if key:
                side.setdefault(key, []).append(row)
    candidates_by_label = {}
    for key in set(old_groups) & set(new_groups):
        old_rows, new_rows = old_groups[key], new_groups[key]
        pairs = []
        if len(old_rows) == len(new_rows) == 1:
            pairs = [(old_rows[0], new_rows[0])]
        else:
            # Duplicate frame anchors can contain multiple people. Resolve only
            # unique exact face-description matches; otherwise leave names untouched.
            by_face = {}
            for row in new_rows:
                face = " ".join(str(row.get("face_features") or "").casefold().split())
                if face:
                    by_face.setdefault(face, []).append(row)
            for row in old_rows:
                face = " ".join(str(row.get("face_features") or "").casefold().split())
                matches = by_face.get(face, [])
                reverse_count = sum(
                    1 for other in old_rows
                    if " ".join(str(other.get("face_features") or "").casefold().split()) == face)
                if face and len(matches) == 1 and reverse_count == 1:
                    pairs.append((row, matches[0]))
        for old_row, new_row in pairs:
            old = " ".join(str(old_row.get("entity") or "").split()).strip()
            new = " ".join(str(new_row.get("entity") or "").split()).strip()
            if old and new and old.casefold() != new.casefold():
                candidates_by_label.setdefault(old.casefold(), set()).add(new)
    # An old display label can refer to multiple people. In that case it is
    # unsafe as a global substitution key (e.g. two observations both called
    # "woman" can become Grandma and Mom). Keep only one-to-one label changes;
    # anchor-specific references are handled separately below.
    return {old: next(iter(targets)) for old, targets in candidates_by_label.items()
            if len(targets) == 1}


def _person_observation_frame_indices(row):
    """Collect source-frame anchors from one person observation."""
    if not isinstance(row, dict):
        return set()
    indices = set()
    raw_values = [row.get("frame_index")]
    raw_values.extend(row.get("anchor_frame_indices") or [])
    refs = [row.get("evidence_ref")]
    refs.extend(row.get("evidence_refs") or [])
    for ref in refs:
        if isinstance(ref, dict) and ref.get("modality") == "visual":
            raw_values.append(ref.get("frame_index"))
    for raw in raw_values:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if index > 0:
            indices.add(index)
    return indices


def _remap_person_visual_labels_by_anchor(media, observations):
    """Bind crop labels to the unique enriched observation at their source frames."""
    if not isinstance(media, dict):
        return media
    group = media.get("person_visuals")
    if not isinstance(group, dict):
        return media
    labels_by_frame = {}
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        label = " ".join(str(row.get("entity") or "").split()).strip()
        if not label:
            continue
        # Primary anchors identify candidates; reaction frames may overlap.
        primary = set()
        for raw in [row.get("frame_index"), *(row.get("anchor_frame_indices") or [])]:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            if index > 0:
                primary.add(index)
        for index in primary or _person_observation_frame_indices(row):
            labels_by_frame.setdefault(index, set()).add(label)
    for item in group.get("items") or []:
        if not isinstance(item, dict):
            continue
        anchors = set()
        for raw in item.get("anchor_frame_indices") or []:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            if index > 0:
                anchors.add(index)
        targets = set().union(*(labels_by_frame.get(index, set()) for index in anchors)) if anchors else set()
        if len(targets) == 1:
            item["person"] = next(iter(targets))
    return media



def _reconcile_people_with_retrieved_faces(observations, media, retrieved, asset_root, source_id):
    """Apply only unique, thresholded face matches from retrieved memories."""
    rows=[dict(r) for r in observations or [] if isinstance(r,dict)]
    if not rows or not isinstance(media,dict) or not asset_root: return rows,[]
    def refs(items,label):
        key=str(label or "").strip().casefold()
        for item in items or []:
            if not isinstance(item,dict) or str(item.get("person") or "").strip().casefold()!=key: continue
            found=[dict(r) for r in item.get("frame_refs") or [] if isinstance(r,dict)
                   and r.get("visual_role")=="face_crop" and r.get("identity_verified") is True
                   and str(r.get("identity_assignment") or "").startswith("qwen_visual_anchor+") and r.get("asset")]
            if found: return found[:12]
        return []
    records=[]; current={}; history={}; names={}; sources={}; source_id=str(source_id or "")
    visuals=((media.get("person_visuals") or {}).get("items") or [])
    for i,row in enumerate(rows):
        label=str(row.get("entity") or "").strip(); images=refs(visuals,label)
        if not label or not images: continue
        rid=f"current:{i}:{label}"; current[rid]=i; names[rid]=label; sources[rid]=source_id
        records.append({"id":rid,"kind":"person","name":label,"events":[
            {"moment_id":source_id,"source_provenance":{"source_segment_id":source_id},"images":images}]})
    for mem in retrieved or []:
        if not isinstance(mem,dict): continue
        mid=str(mem.get("id") or "").strip()
        mm=mem.get("multimodal") or {}; items=((mm.get("person_visuals") or {}).get("items") or [])
        for j,person in enumerate(mem.get("person_observations") or []):
            if not isinstance(person,dict): continue
            entity=str(person.get("entity") or "").strip()
            name=str(person.get("readable_name") or entity).strip()
            images=refs(items,entity or name)
            if not name or not images: continue
            rid=f"history:{mid}:{j}:{name}"; history[rid]=person; names[rid]=name
            sources[rid]=str((mem.get("source_provenance") or {}).get("source_segment_id") or mid)
            records.append({"id":rid,"kind":"person","name":name,"events":[
                {"moment_id":mid,"source_provenance":{"source_segment_id":sources[rid]},"images":images}]})
    if not current or not history: return rows,[]
    try: _link_face_identity_records(records,asset_root=asset_root)
    except Exception: return rows,[]
    edges=[]
    for record in records:
        for edge in record.get("identity_matches") or []:
            if not isinstance(edge,dict): continue
            try: score=float(edge.get("score"))
            except (TypeError,ValueError): continue
            a,b=str(edge.get("left") or ""),str(edge.get("right") or "")
            if a in current and b in history: edges.append((a,b,score))
            elif b in current and a in history: edges.append((b,a,score))
    threshold=float(os.environ.get("MEMORY_FACE_SIMILARITY_THRESHOLD","0.64"))
    generic=re.compile(r"(?i)^(?:man|woman|person|visible person|person in .+|speaker|guest|employee)(?:\s*#?\d+)?$")
    applied=[]
    for rid,i in current.items():
        row=rows[i]; old=str(row.get("entity") or "").strip()
        if not old or not generic.fullmatch(old): continue
        by_name={}
        for current_id,hid,score in edges:
            name=" ".join(str(names.get(hid) or "").split()).strip()
            if current_id==rid and score>=threshold and name and not generic.fullmatch(name):
                if score>by_name.get(name,(-1.0,""))[0]: by_name[name]=(score,hid)
        ranked=sorted(by_name.items(),key=lambda x:x[1][0],reverse=True)
        if not ranked or (len(ranked)>1 and ranked[0][1][0]-ranked[1][1][0]<0.08): continue
        name,(score,hid)=ranked[0]; prior=history[hid]
        row["entity"]=name; row["readable_name"]=name
        for field in ("roles","relationships","profile_context"):
            if not row.get(field) and prior.get(field): row[field]=copy.deepcopy(prior[field])
        row["cross_clip_identity"]={"status":"probable_visual_match","name":name,
            "confidence":"high" if score>=0.75 else "medium","score":round(score,5),
            "model":"InsightFace buffalo_l / ArcFace",
            "basis":"unique match to a retrieved, identity-verified historical face crop",
            "evidence_ref":{"modality":"visual","source_segment_id":source_id,
                "matched_source_segment_id":sources.get(hid,""),"matched_memory_id":hid.split(":")[1]}}
        applied.append({"from":old,"to":name,"score":round(score,5),
            "matched_source_segment_id":sources.get(hid,"")})
    return rows,applied


def _remap_person_fact_references_by_anchor(fields, before, after):
    """Remap visual fact entity references only when their frame anchor is unique."""
    if not isinstance(fields, dict):
        return fields
    old_labels = {str(row.get("entity") or "").strip().casefold()
                  for row in before or [] if isinstance(row, dict)
                  and str(row.get("entity") or "").strip()}
    if not old_labels:
        return fields
    labels_by_frame = {}
    for row in after or []:
        if not isinstance(row, dict):
            continue
        label = " ".join(str(row.get("entity") or "").split()).strip()
        if label:
            for index in _person_observation_frame_indices(row):
                labels_by_frame.setdefault(index, set()).add(label)
    for fact in fields.get("temporal_facts") or []:
        if not isinstance(fact, dict):
            continue
        refs = [fact.get("evidence_ref")]
        refs.extend(fact.get("evidence_refs") or [])
        if fact.get("modality") != "visual" and not any(
                isinstance(ref, dict) and ref.get("modality") == "visual" for ref in refs):
            continue
        indices = _person_observation_frame_indices(fact)
        for ref in refs:
            if isinstance(ref, dict) and ref.get("modality") == "visual":
                indices.update(_person_observation_frame_indices(ref))
        targets = set().union(*(labels_by_frame.get(index, set()) for index in indices)) if indices else set()
        if len(targets) != 1:
            continue
        target = next(iter(targets))
        for key in ("subject", "actor"):
            value = fact.get(key)
            if isinstance(value, str) and value.strip().casefold() in old_labels:
                fact[key] = target
        affected = fact.get("affected_people")
        if isinstance(affected, list):
            fact["affected_people"] = [target if isinstance(value, str)
                                        and value.strip().casefold() in old_labels else value
                                        for value in affected]
        elif isinstance(affected, str) and affected.strip().casefold() in old_labels:
            fact["affected_people"] = target
    return fields


def _remap_person_labels_in_fields(fields, mapping):
    """Update display references while leaving types, IDs, quotes, and source evidence immutable."""
    if not isinstance(fields, dict) or not mapping:
        return fields
    def replace_text(value):
        if not isinstance(value, str):
            return value
        for old, new in sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True):
            if old:
                value = re.sub(r"(?<![\w])" + re.escape(old) + r"(?![\w])",
                               lambda _match, replacement=new: replacement,
                               value, flags=re.I)
        return value
    for key in ("scene", "content_summary", "user_action", "interaction_details", "activity", "location",
                "environments"):
        value = fields.get(key)
        if isinstance(value, str):
            fields[key] = replace_text(value)
        elif isinstance(value, list):
            fields[key] = [replace_text(item) for item in value]
    if isinstance(fields.get("people"), list):
        fields["people"] = [replace_text(item) for item in fields["people"]]
    for row in fields.get("person_observations") or []:
        if isinstance(row, dict) and isinstance(row.get("entity"), str):
            row["entity"] = replace_text(row["entity"])
    for row in fields.get("temporal_facts") or []:
        if not isinstance(row, dict):
            continue
        for key in ("subject", "actor", "affected_people"):
            value = row.get(key)
            if isinstance(value, str):
                row[key] = replace_text(value)
            elif isinstance(value, list):
                row[key] = [replace_text(item) for item in value]
    return fields


def _person_name_ordinal_aliases(observations):
    """Create conservative aliases for names accidentally suffixed by a detector index."""
    aliases = {}
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        label = " ".join(str(row.get("entity") or "").split()).strip()
        if not label:
            continue
        base = re.sub(r"\s+\d+$", "", label).strip()
        generic = re.fullmatch(
            r"(?i)(?:person|man|woman|individual|unknown|person at doorway)(?:\s+\d+)?",
            base)
        tokens = base.split()
        if len(tokens) < 2 or not all(token[:1].isupper() for token in tokens) or generic:
            continue
        for index in range(1, 17):
            aliases[f"{base} {index}"] = base
    return aliases


def _remap_person_labels_in_media(value, mapping, key=""):
    """Remap visible entity labels, never source evidence, identifiers, or type fields."""
    if not mapping:
        return value
    normalized = {str(old).strip().casefold(): str(new)
                  for old, new in mapping.items() if str(old).strip()}
    protected = (key in {"id", "entity_id", "observation_id", "subject_type", "entity_type",
                         "type", "modality", "status", "confidence", "quote", "quotes",
                         "evidence", "evidence_ref", "evidence_refs", "source_evidence",
                         "source_id", "source_segment_id", "asset", "path", "bbox",
                         "frame_index", "time_order"}
                 or key.endswith("_id") or key.startswith("source_")
                 or key.startswith("evidence"))
    if protected:
        return value
    if isinstance(value, str):
        for old, new in sorted(normalized.items(), key=lambda pair: len(pair[0]), reverse=True):
            if old:
                value = re.sub(r"(?<![\w])" + re.escape(old) + r"(?![\w])",
                               lambda _match, replacement=new: replacement,
                               value, flags=re.I)
        return value
    if isinstance(value, list):
        return [_remap_person_labels_in_media(item, mapping, key) for item in value]
    if isinstance(value, dict):
        return {child_key: _remap_person_labels_in_media(item, mapping, str(child_key))
                for child_key, item in value.items()}
    return value


def _remove_visual_age_guess(value):
    """Remove age labels inferred from appearance while preserving the person reference."""
    text = str(value or "")
    def neutral_person(match):
        article = str(match.group("article") or "").casefold()
        noun = str(match.group("qualified") or match.group("noun") or "").casefold()
        if noun == "children":
            return "the people" if article == "the" else "people"
        if article == "the":
            return "the person"
        if article in {"a", "an"}:
            return "a person"
        return "person"
    return _AGE_CLAIM_RE.sub(neutral_person, text)


def _source_video_id(moment, video_path=None):
    """Return a portable EgoLife-relative video ID, never a host absolute path."""
    raw = ""
    if video_path:
        try:
            raw = Path(video_path).resolve().relative_to(VIDEO_ROOT).as_posix()
        except (OSError, ValueError):
            try:
                raw = Path(video_path).resolve().relative_to(FEISHU_TEST_ROOT).as_posix()
            except (OSError, ValueError):
                try:
                    raw = Path(video_path).resolve().relative_to(ROOT).as_posix()
                except (OSError, ValueError):
                    raw = ""
    raw = raw or str(moment.get("video_id") or moment.get("video") or "")
    return _portable_video_id(raw)

SHORT_FILE = TESTSET / "short_term" / "moments_scattered.jsonl"
SHORT_PLAN = TESTSET / "short_term" / "crud_plan.json"
SHORT_EXPANDED_FILE = TESTSET / "short_term" / "moments_scattered_v2_120.jsonl"
SHORT_EXPANDED_PLAN = TESTSET / "short_term" / "crud_plan_v2.json"
LONG_FILE = TESTSET / "long_term" / "full_day.jsonl"
LONG_META = TESTSET / "long_term" / "timeline_meta.json"
LONG_PERSON_DIR = TESTSET / "long_term" / "person_sets"
FEISHU_TEST_ROOT = (ROOT / "test_data" / "feishu_ar_agent_memory_retrieval_20261013").resolve()
FEISHU_TEST_MANIFEST = FEISHU_TEST_ROOT / "manifest.json"
FEISHU_TESTSET_LABELS = {
    "1.social_hint": "社交提示", "2.travel_abroad": "出国旅行",
    "3.forgetting_item": "遗忘物品", "4.first_aid": "急救场景",
    "5.health_safeguard": "健康保障", "6.hotel_reception": "酒店接待",
    "7.photo_taking": "拍照协助", "8.fraud_detection": "诈骗识别",
    "9.security_alert": "安全警报", "10.conversation_assistance": "对话辅助",
    "11.emotional support": "情绪支持", "12.shopping_selection": "购物选择",
}


def site_perceiver():
    """Use the memory-eval V3 cache for this visualization, with an env override."""
    configured = os.environ.get("EGOLIFE_QWEN_CACHE")
    if configured:
        cache_path = Path(configured)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
    else:
        cache_path = RUNS / "qwen_cache_memory_prompt_v3.json"
        if not cache_path.exists():
            cache_path = RUNS / "qwen_cache.json"
    configured_pool = os.environ.get("QWEN_API_URLS", "").strip()
    single_override = os.environ.get("QWEN_API_URL", "").strip()
    api_urls = ([value.strip() for value in configured_pool.split(",") if value.strip()]
                if configured_pool else
                ([single_override] if single_override else
                 [f"http://127.0.0.1:{port}" for port in (8001, 8002, 8003, 8004)]))
    return QwenPerceiver(api_urls=api_urls, cache_file=cache_path)


SCENARIO_META = {
    "short_term": {"name": "短期记忆场景", "desc": "不连续视频片段 · 验证 Layer1~7 完整 CRUD"},
    "long_term": {"name": "长期记忆场景", "desc": "完整一天 · 验证记忆流转与夜间整理"},
    "custom_test": {"name": "自定义测试集", "desc": "按场景分类逐段写入与检索；有累积记忆时可随时执行夜间归纳"},
}


def _clock_from_id(mid):
    """DAY1_A2_ALICE_11093000_en → 11:09:30"""
    parts = str(mid).split("_")
    if len(parts) >= 4 and re.fullmatch(r'DAY\d+', parts[0], re.I):
        t = parts[3]
        if (len(t) >= 6 and t[:6].isdigit() and int(t[:2]) < 24
                and int(t[2:4]) < 60 and int(t[4:6]) < 60):
            return f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
    return ""


def _intra_hour_offset(mid):
    """文件名时间戳相对整点小时的秒数：DAY1_A2_ALICE_11093000 → 9*60+30 = 570s"""
    parts = str(mid).split("_")
    if _clock_from_id(mid):
        t = parts[3]
        return int(t[2:4]) * 60 + int(t[4:6])
    return 0


def moment_brief(m):
    """单条片段的非标注元数据；语义内容只允许来自模型的视频感知。"""
    video = m.get("video") or ""                 # data/EgoLife/A2_ALICE/DAY1/xxx.mp4

    # 本地视频路径（目录是小写 egolife），存在则给出可播放 URL
    local = ROOT / video.replace("EgoLife", "egolife") if video else None
    has_video = local is not None and local.exists()
    video_url = None
    clip_start = clip_end = None
    if has_video:
        resolved = local.resolve()
        try:
            rel = resolved.relative_to(VIDEO_ROOT)
        except ValueError:
            try:
                rel = resolved.relative_to(FEISHU_TEST_ROOT)
            except ValueError:
                rel = resolved.relative_to(ROOT)
        video_url = "/api/video?p=" + quote(str(rel))
        # start_time/end_time 是相对整点小时的偏移；换算成 30s 片段内秒数
        if m.get("start_time") is not None:
            off = _intra_hour_offset(m.get("id", ""))
            clip_start = round(max(0.0, m["start_time"] - off), 2)
            clip_end = round(m.get("end_time", m["start_time"] + 30) - off, 2)

    return {
        "id": m.get("id", ""),
        "video": video.rsplit("/", 1)[-1] if video else "",
        "clock": _clock_from_id(m.get("id", "")),
        "start_time": m.get("start_time"),
        "end_time": m.get("end_time"),
        "hour": m.get("_hour", ""),
        "people": [],
        "location": "",
        "scene": "",
        "action": "",
        "video_url": video_url,
        "clip_start": clip_start,
        "clip_end": clip_end,
    }


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def feishu_test_moments(dataset):
    """Build deterministic video-only moments from the copied Feishu inventory."""
    if not FEISHU_TEST_MANIFEST.is_file():
        raise ValueError("Feishu test video manifest is missing")
    manifest = json.load(open(FEISHU_TEST_MANIFEST, encoding="utf-8"))
    files = manifest.get("files") or []
    selector = str(dataset or "")
    category = selector.split(":", 1)[1] if selector.startswith("feishu:") else "all"
    if category != "all":
        files = [row for row in files if row.get("scenario") == category]

    def chapter_order(row):
        scenario_numbers = tuple(int(value) for value in re.findall(
            r"\d+", str(row.get("scenario") or "")))
        clip_numbers = tuple(int(value) for value in re.findall(
            r"\d+", str(row.get("clip_id") or Path(
                str(row.get("relative_path") or "")).stem)))
        return scenario_numbers, clip_numbers, str(row.get("relative_path") or "").casefold()

    # The UI runs the custom set in the Feishu chapter/video sequence even if
    # the manifest was regenerated from an unordered filesystem traversal.
    files = sorted(files, key=chapter_order)
    moments = []
    for row in files:
        relative = str(row.get("relative_path") or "")
        video_path = (FEISHU_TEST_ROOT / relative).resolve()
        if FEISHU_TEST_ROOT not in video_path.parents or not video_path.is_file():
            continue
        clip_id = str(row.get("clip_id") or Path(relative).stem)
        token = clip_id.replace(".", "_").replace("-", "_")
        digest = str(row.get("sha256") or "")[:10]
        moments.append({
            "id": f"FEISHU_{token}_{digest}",
            "video": (Path("test_data") / FEISHU_TEST_ROOT.name / relative).as_posix(),
            "start_time": 0.0,
            "end_time": float(row.get("duration_seconds") or 0),
            "duration_seconds": float(row.get("duration_seconds") or 0),
            "_hour": "12",
            "test_category": str(row.get("scenario") or ""),
            "test_clip_id": clip_id,
            "transcript": "",
        })
    if not moments:
        raise ValueError(f"No usable Feishu videos in dataset: {dataset}")
    return moments


def _attach_location_visual_evidence(media, fields):
    """Link a place to one existing Qwen input frame without storing a duplicate image."""
    if not isinstance(media, dict):
        return
    location = str((fields or {}).get("location") or "").strip()
    frames = [row for row in media.get("visual_frames") or []
              if isinstance(row, dict) and row.get("visual_role") == "qwen_input_frame"
              and row.get("asset")]
    if not location or not frames:
        media["location_visuals"] = {"schema": 1, "status": "not_available", "items": []}
        return
    facts = [row for row in (fields or {}).get("temporal_facts") or []
             if isinstance(row, dict) and str(row.get("subject_type") or "").casefold() == "place"]
    fact = next((row for row in facts if row.get("frame_index")), None)
    frame_index = None
    if fact:
        try:
            frame_index = int(fact.get("frame_index"))
        except (TypeError, ValueError):
            frame_index = None
    selected = next((row for row in frames
                     if frame_index and int(row.get("frame_index") or 0) == frame_index), None)
    if selected is None:
        selected = frames[len(frames) // 2]
    ref = dict(selected)
    ref["visual_role"] = "location_source_frame"
    ref["place_evidence_basis"] = (
        str(fact.get("evidence") or "")[:300] if fact
        else "地点没有独立裁剪框；保留一张代表性 Qwen 输入原帧供回查")
    media["location_visuals"] = {
        "schema": 1, "status": "source_frame",
        "items": [{"location": location, "status": "source_frame",
                   "evidence": ref["place_evidence_basis"], "frame_refs": [ref]}],
    }


PARTICIPANT_NAMES = {
    "A1_JAKE": "Jake",
    "A2_ALICE": "Alice",
    "A3_TASHA": "Tasha",
    "A4_LUCIA": "Lucia",
    "A5_KATRINA": "Katrina",
    "A6_SHURE": "Shure",
}


def interactive_dataset_options():
    """Datasets available in the step-by-step runner."""
    options = []
    for dataset, path, plan_path, title in (
        ("base", SHORT_FILE, SHORT_PLAN, "短期 CRUD 演示集"),
        ("expanded", SHORT_EXPANDED_FILE, SHORT_EXPANDED_PLAN, "短期扩展集"),
    ):
        rows = _load_jsonl(path)
        counts = {}
        for row in rows:
            parts = str(row.get("id", "")).split("_")
            participant = "_".join(parts[1:3]) if len(parts) >= 3 else ""
            if participant:
                counts[participant] = counts.get(participant, 0) + 1
        per_person = next(iter(counts.values()), 0)
        purpose = (
            f"{len(rows)} 段，6 人各 {per_person} 段；原 CRUD 步进演示集。"
            if dataset == "base" else
            f"{len(rows)} 段，6 人各 {per_person} 段；扩展短期测评集，可逐段写入并执行 CRUD。"
        )
        options.append({
            "value": f"short_term|{dataset}|all",
            "scenario": "short_term",
            "dataset": dataset,
            "participant": None,
            "label": f"{title} · 全部人物（{len(rows)} 段）",
            "note": purpose,
        })
        for participant in PARTICIPANT_NAMES:
            if participant not in counts:
                continue
            name = PARTICIPANT_NAMES[participant]
            options.append({
                "value": f"short_term|{dataset}|{participant}",
                "scenario": "short_term",
                "dataset": dataset,
                "participant": participant,
                "label": f"{title} · {name}（{counts[participant]} 段）",
                "note": f"{purpose} 当前只加载 {name} 的片段。",
            })

    long_index = json.load(open(LONG_PERSON_DIR / "index.json", encoding="utf-8"))
    for meta in long_index.get("datasets", []):
        participant = meta["participant"]
        name = PARTICIPANT_NAMES.get(participant, participant)
        complete = meta.get("complete", False)
        coverage = round(float(meta.get("coverage", 0)) * 100, 1)
        if complete:
            label = f"长期完整日集 · {name}（{meta['available_count']} 段）"
            note = (
                f"DAY1 完整视频集：{meta['available_count']} 段，覆盖 "
                f"{len(meta.get('hours_covered', []))} 个有数据时段。"
            )
        else:
            label = (
                f"长期部分集 · {name}（{meta['available_count']}/"
                f"{meta['total_candidates']} 段 · {coverage}%）"
            )
            note = (
                f"DAY1 视频不完整：仅 {meta['available_count']}/"
                f"{meta['total_candidates']} 段（{coverage}%）可运行；"
                "结果仅代表本地已下载片段，不是完整日测评。"
            )
        options.append({
            "value": f"long_term|long_day|{participant}",
            "scenario": "long_term",
            "dataset": "long_day",
            "participant": participant,
            "label": label,
            "note": note,
        })

    if FEISHU_TEST_MANIFEST.is_file():
        manifest = json.load(open(FEISHU_TEST_MANIFEST, encoding="utf-8"))
        files = manifest.get("files") or []
        categories = {}
        for row in files:
            category = str(row.get("scenario") or "")
            if category:
                categories[category] = categories.get(category, 0) + 1
        source_note = (
            "飞书文档中的视频测试集。每段视频按原片输入，依次执行场景感知、记忆写入和历史记忆检索；"
            "本模块只提供相关记忆及来源，不生成行动建议。"
        )
        if files:
            options.append({
                "value": "custom_test|feishu:all|all",
                "scenario": "custom_test",
                "dataset": "feishu:all",
                "participant": None,
                "source": "feishu",
                "label": f"飞书自定义测试集 · 全部类别（{len(files)} 段）",
                "note": f"{len(categories)} 类、{len(files)} 段视频。{source_note}",
            })
        for category in sorted(categories, key=lambda value: int(value.split(".", 1)[0])
                                if value.split(".", 1)[0].isdigit() else 999):
            label = FEISHU_TESTSET_LABELS.get(category, category)
            count = categories[category]
            options.append({
                "value": f"custom_test|feishu:{category}|all",
                "scenario": "custom_test",
                "dataset": f"feishu:{category}",
                "participant": None,
                "source": "feishu",
                "label": f"飞书自定义测试集 · {label}（{count} 段）",
                "note": f"{category} · {count} 段视频。{source_note}",
            })
    return options


def _person_profile_is_relevant(row, fields):
    """Store a person profile only when the person matters to this user's episode."""
    if not isinstance(row, dict):
        return False
    roles = [str(value or "").casefold() for value in row.get("roles") or []]
    entity = str(row.get("entity") or "").strip()
    generic_entity = bool(re.fullmatch(
        r"(?i)(?:man|woman|person|male|female|stranger|passerby|bystander)(?:\s+\d+)?",
        entity))
    relation_values = row.get("relationships") or []
    if isinstance(relation_values, str):
        relation_values = [relation_values]
    relations = " ".join(str(value or "") for value in relation_values).strip()
    aliases = [item for item in row.get("spoken_aliases") or []
               if isinstance(item, dict)
               and item.get("status") == "probable_visual_match"
               and isinstance(item.get("evidence_ref"), dict)
               and item["evidence_ref"].get("modality") == "speech"]
    if aliases or relations:
        return True

    incidental_roles = {"passerby", "bystander", "background person",
                        "background extra", "stranger",
                        "stranger (relationship not established in this clip)",
                        "unidentified person", "unidentified"}
    specific_roles = [role for role in roles
                      if role not in incidental_roles
                      and not role.startswith("stranger ")
                      and role not in {"person", "man", "woman"}]
    if specific_roles:
        return True
    if not generic_entity and entity:
        return True

    # Undiarized speech alone is not enough. Require visible, direct engagement
    # supported by the episode interaction or by this person's own evidence.
    interaction = " ".join(str(fields.get(key) or "") for key in
                           ("interaction_details", "activity", "user_action"))
    row_evidence = " ".join(str(row.get(key) or "") for key in
                            ("evidence", "profile_context", "interaction_details", "dialogue"))
    direct_cue = re.compile(
        r"(?i)\b(?:talks? to|speaks? to|greets?|asks?|answers?|responds?|"
        r"serves?|assists?|helps?|hands? (?:over|to)|gives? (?:to)?|"
        r"receives? (?:from)?|shakes? hands?|waves? at|interacts? with|"
        r"has a conversation with|exchanges? (?:words|greetings))\b|"
        r"交谈|对话|问候|询问|回答|回应|服务|帮助|递给|交给|握手|挥手|互动|"
        r"交流|打招呼|攀谈")
    if direct_cue.search(row_evidence):
        return True
    if direct_cue.search(interaction):
        # In a multi-person scene, a generic cue may refer to someone else.
        people = [item for item in fields.get("person_observations") or []
                  if isinstance(item, dict)]
        if len(people) <= 1 or re.search(re.escape(entity), interaction, re.I):
            return True

    # Keep a lone, clearly episode-central subject (for example the man shown
    # throughout an elevator clip), even when their relationship is unknown.
    all_roles = " ".join(roles)
    people = [item for item in fields.get("person_observations") or []
              if isinstance(item, dict)]
    focal_text = " ".join(str(fields.get(key) or "") for key in
                          ("content_summary", "activity", "user_action",
                           "interaction_details", "scene"))
    incidental_scene = bool(re.search(
        r"(?i)\b(?:background (?:person|figure|extra)|walks? past|passes? by|bystander|passerby)\b|背景人物|路过|经过",
        focal_text))
    active_scene = bool(re.search(
        r"(?i)\b(?:opens?|steps? out|walks?|approaches?|greets?|touches?|helps?|assists?|serves?|presents?|stands?|sits?|speaks?|holds?|uses?|wipes?|covers?)\b",
        focal_text))
    entity_in_episode = bool(entity and re.search(re.escape(entity), interaction, re.I))
    if ((len(people) == 1 or entity_in_episode) and "passerby" not in all_roles
            and not incidental_scene and active_scene):
        return True
    return False


def _repair_anonymized_person_references(fields):
    """Keep incidental people in scene prose while leaving them out of profiles."""
    if not isinstance(fields, dict):
        return fields
    text_fields = ("scene", "content_summary", "interaction_details", "activity",
                   "user_action", "notes")
    repairs = (
        (re.compile(r"\ban\s+in(?=\s+)", re.I), "a person in"),
        (re.compile(r"\ban\s+who\b", re.I), "a person who"),
        (re.compile(r"\banother\s+who\b", re.I), "another person who"),
        (re.compile(r"\banother\s+is\s+(?=(?:seated|sitting|standing|walking)\b)", re.I), "another person is "),
        (re.compile(r"\bpasses\s+an\s+who\b", re.I), "passes a person who"),
    )
    for key in text_fields:
        value = fields.get(key)
        if not isinstance(value, str):
            continue
        for pattern, replacement in repairs:
            value = pattern.sub(replacement, value)
        fields[key] = value
    return fields

def _filter_incidental_person_profiles(rows, fields):
    return [row for row in rows if _person_profile_is_relevant(row, fields)]


def _ensure_episode_role(row, fields, transcript=""):
    """Attach an evidence-calibrated episode role without guessing biography."""
    if not isinstance(row, dict):
        return row
    roles = _clean_episode_roles(row.get("roles") or [], row.get("entity"))
    context = " ".join(str(fields.get(key) or "") for key in (
        "scene", "content_summary", "user_action", "interaction_details",
        "location", "activity", "environments", "objects"))
    visual = " ".join(str(row.get(key) or "") for key in (
        "appearance", "evidence", "profile_context", "face_features"))
    relationships = " ".join(str(value or "") for value in row.get("relationships") or [])
    entity = str(row.get("entity") or "")
    aliases = [str(item.get("alias") or "") for item in row.get("spoken_aliases") or []
               if isinstance(item, dict) and item.get("status") == "probable_visual_match"
               and isinstance(item.get("evidence_ref"), dict)
               and item["evidence_ref"].get("modality") == "speech"]
    text = (context + " " + visual + " " + relationships + " "
            + entity + " " + " ".join(aliases) + " " + str(transcript or "")).casefold()
    # Prefer an explicit entity label or a visual-linked spoken alias/relationship
    # over generic model labels such as "speaker".
    role_aliases = {
        "grandmother": ("姥姥", "外婆", "奶奶", "祖母", "grandma", "grandmother"),
        "grandfather": ("外公", "爷爷", "祖父", "grandpa", "grandfather"),
        "mother": ("妈妈", "母亲", "妈", "mom", "mother", "mum"),
        "father": ("爸爸", "父亲", "爸", "dad", "father"),
        "son": ("son", "儿子"),
        "daughter": ("daughter", "女儿"),
        "baby / infant": ("baby", "infant", "婴儿", "宝宝"),
        "service staff": ("waiter", "waitress", "server", "cashier", "服务员"),
        "neighbor": ("邻居", "neighbor", "neighbour"),
        "friend": ("朋友", "friend"),
    }
    direct_identity_text = " ".join([entity, *aliases]).casefold()
    relationship_text = relationships.casefold()
    mapped_role = next((role for role, terms in role_aliases.items()
                        if any(term.casefold() in direct_identity_text for term in terms)), "")
    if not mapped_role:
        mapped_role = next((role for role, terms in role_aliases.items()
                            if any(term.casefold() in relationship_text for term in terms)), "")

    # Restaurant staff must be identified from this person's own service duties,
    # never from a waiter mentioned elsewhere in the scene. A seated, camera-facing
    # patron is a guest even when background staff are serving other tables.
    service_roles = {"service staff", "server", "waiter", "waitress"}
    service_duty = re.compile(
        r"\b(?:serves?|serving|takes? (?:an? )?orders?|delivers? (?:food|dishes|drinks)|"
        r"brings? (?:food|dishes|drinks)|clears? (?:the )?(?:table|dishes)|"
        r"processes? (?:a )?payment|hands? (?:the )?(?:menu|bill|check))\b", re.I)
    row_service_evidence = " ".join(str(row.get(key) or "") for key in
                                    ("appearance", "evidence", "profile_context"))
    linked_interaction = False
    if entity:
        interaction = str(fields.get("interaction_details") or "")
        entity_pattern = re.escape(entity)
        duty_pattern = r"(?:serves?|serving|takes? (?:an? )?orders?|delivers? (?:food|dishes|drinks)|brings? (?:food|dishes|drinks)|clears? (?:the )?(?:table|dishes)|processes? (?:a )?payment)"
        # Restrict actor linkage to one clause; nearby actions by another person
        # must not leak across sentence boundaries or long multi-person clauses.
        linked_interaction = bool(re.search(
            rf"{entity_pattern}[^.!?;]{{0,70}}{duty_pattern}", interaction, re.I)
            or re.search(rf"\b(?:by|performed by)\s+{entity_pattern}\b.{0,50}{duty_pattern}", interaction, re.I))
    has_direct_service_duty = bool(service_duty.search(row_service_evidence) or linked_interaction)
    seated_restaurant_guest = bool(
        re.search(r"\b(?:seated|sitting)\b", visual, re.I)
        and re.search(r"\b(?:restaurant|dining|diner|table|chair)\b", visual + " " + context, re.I))
    if not has_direct_service_duty:
        roles = [role for role in roles if str(role).strip().casefold() not in service_roles]
        row["roles"] = roles
    if seated_restaurant_guest and not has_direct_service_duty:
        roles = [role for role in roles if str(role).strip().casefold() not in service_roles]
        row["roles"] = roles
    has_medical_uniform = bool(re.search(
        r"\b(?:medical|nurse|doctor|paramedic)\b.{0,50}\b(?:uniform|scrubs|coat)\b|\b(?:uniform|scrubs|coat)\b.{0,50}\b(?:medical|nurse|doctor|paramedic)\b|\bred cross armband\b|医疗.{0,30}制服|红十字臂章|护士服|医生服",
        visual, re.I))
    if (any("medical responder" in str(role).casefold() for role in roles)
            and not has_medical_uniform
            and re.search(r"\b(?:bent forward|receives? (?:medical )?aid|being attended to|patient)\b|弯腰|接受急救|被照料|患者", visual + " " + context, re.I)):
        row["roles"] = ["person receiving medical assistance"]
        row["profile_confidence"] = row.get("profile_confidence") or "medium"
        return row
    specific = [role for role in roles
                if not re.search(r"(?i)\b(?:unidentified|unknown)\b", role)
                and role.casefold() not in {"speaker", "person", "man", "woman", "participant"}
                and not re.search(r"(?i)^(?:likely|possibly|probable|inferred|apparent)\b", role.strip())]
    if mapped_role:
        row["roles"] = [mapped_role]
        if re.search(r"[\u3400-\u9fff]", entity):
            row.setdefault("visual_entity", entity)
            row["entity"] = mapped_role
        row["profile_confidence"] = row.get("profile_confidence") or "medium"
        return row
    if specific:
        row["roles"] = specific
        return row
    row_context = (visual + " " + context).casefold()
    if (re.search(r"\bmain subject\b|主要人物", entity, re.I)
            and re.search(r"\buser\b.{0,40}(?:performs?|does|walks?|holds?)|用户.{0,30}(?:进行|做|走|拿)", context, re.I)):
        row["roles"] = ["user / camera wearer"]
        row["profile_confidence"] = row.get("profile_confidence") or "medium"
        return row
    birthday_recipient = bool(
        re.search(r"\bhappy birthday\b|生日快乐", text, re.I)
        and re.search(r"\b(?:thank you|means a lot|didn't expect anyone to remember)\b|谢谢|没想到有人记得", text, re.I)
        and len([item for item in fields.get("person_observations") or []
                 if isinstance(item, dict)]) == 1)
    if birthday_recipient:
        row["roles"] = ["birthday recipient"]
        row["profile_confidence"] = row.get("profile_confidence") or "medium"
        return row
    medical_cue = bool(re.search(r"\b(?:medical|first aid|red cross|nurse|doctor|paramedic)\b|医疗|急救|红十字|护士|医生", context + " " + visual, re.I))
    medical_appearance = bool(re.search(
        r"\b(?:medical|nurse|doctor|paramedic)\b.{0,50}\b(?:uniform|scrubs|coat)\b|\b(?:uniform|scrubs|coat)\b.{0,50}\b(?:medical|nurse|doctor|paramedic)\b|\bred cross armband\b|医疗.{0,30}制服|红十字臂章|护士服|医生服",
        visual, re.I))
    if medical_cue and medical_appearance and re.search(r"\b(?:kneels? to attend|attends? to|first aid|treats?)\b|跪下照料|照料|急救|治疗", context, re.I):
        row["roles"] = ["medical responder"]
        row["profile_confidence"] = row.get("profile_confidence") or "medium"
        return row
    if medical_cue and re.search(r"\b(?:bent forward|receives? (?:medical )?aid|being attended to|patient)\b|弯腰|接受急救|被照料|患者", row_context, re.I):
        row["roles"] = ["person receiving medical assistance"]
        row["profile_confidence"] = row.get("profile_confidence") or "medium"
        return row
    role = ""
    # Use person-linked duties only; scene-level service words can describe a
    # different, incidental worker and must not relabel the main participant.
    if not role and has_direct_service_duty:
        role = "service staff"
    elif not role and (re.search(r"\b(?:product|supplement|healthcare|brand|package|box)\b", text, re.I)
          and re.search(r"\b(?:works well|recommend|promot|buy|order|for sale|product)\b", text, re.I)
          and re.search(r"\b(?:holding|holds|presenting|shows|displaying)\b|展示|拿着|举着", text, re.I)):
        role = "product presenter"
    elif not role and re.search(r"\b(?:restaurant|diner|cafe)\b|餐厅|饭店", text, re.I):
        # The person is a restaurant guest even when their clip only contains
        # introductions or conversation rather than an eating action.
        role = "diner / restaurant guest"
    elif not role and (re.search(r"\b(?:sidewalk|street|path|track|public square)\b|人行道|街道|步道", text, re.I)
          and re.search(r"\b(?:walks? past|walking past|passes? by|walks? ahead|walking ahead)\b|路过|经过", text, re.I)):
        role = "passerby"
    elif not role and re.search(r"(?i)repeated mouth movement.*speaking|mouth movements.*speaking|speaker.*to the camera", visual):
        role = "speaker"
    else:
        role = role or "stranger (relationship not established in this clip)"
    row["roles"] = [role]
    row["profile_confidence"] = row.get("profile_confidence") or "low"
    return row


def _clean_episode_roles(values, entity=""):
    """Keep grounded role labels and drop unresolved alternative-role compounds.

    A slash often records competing model guesses (for example, "doctor/boss"),
    not a coherent role. Retain a specific grounded singleton already present in
    the list; otherwise omit the ambiguous compound and preserve the visual label.
    """
    if not isinstance(values, (list, tuple)):
        return []
    def canonical(value):
        text = re.sub(r"(?i)^\s*(?:likely|possibly|inferred)\s+", "", str(value or ""))
        text = re.sub(r"\([^)]*\)", "", text).strip().casefold()
        words = re.findall(r"[a-z]+", text)
        text = " ".join(words)
        groups = (
            {"doctor", "physician", "clinician"},
            {"boss", "manager", "employer representative"},
            {"server", "waiter", "waitress"},
            {"customer", "client", "shopper"},
            {"teacher", "educator", "instructor"},
        )
        for group in groups:
            if text in group:
                return min(group)
        return text
    clean = []
    seen_canonical = set()
    labels = [" ".join(str(value or "").split()).strip()[:80]
              for value in values if isinstance(value, str) and str(value).strip()]
    grounded_roles = {canonical(label) for label in labels
                      if not re.match(r"(?i)^\s*(?:likely|possibly|inferred)\s+", label)}
    for value in labels:
        if not isinstance(value, str) or not value.strip():
            continue
        label = " ".join(value.split()).strip()[:80]
        is_uncertain = bool(re.match(r"(?i)^\s*(?:likely|possibly|inferred)\s+", label))
        if is_uncertain and grounded_roles and canonical(label) not in grounded_roles:
            continue
        if "/" in label:
            parts = [part.strip() for part in label.split("/") if part.strip()]
            identities = {canonical(part) for part in parts}
            if len(identities) > 1:
                continue
            if not parts:
                continue
            label = parts[0]
        canonical_label = canonical(label)
        if (label.casefold() not in {item.casefold() for item in clean}
                and canonical_label not in seen_canonical):
            clean.append(label)
            seen_canonical.add(canonical_label)
    return clean[:6]


class InteractiveSession(PostWriteRetrievalMixin):
    """一个场景的步进执行会话。"""

    def __init__(self, key, participant=None, dataset="base", perceiver=None,
                 resume_state=None):
        assert key in SCENARIO_META, f"unknown scenario: {key}"
        self.key = key
        self.participant = participant
        self.dataset_key = dataset
        self.pos = 0            # 已执行步数 = 下一步索引
        self.auto_paused = False
        self.step_cancelled = False
        self.nightly_progress = None
        self.step_progress = None
        self.log = []           # [{kind, title, msg, snapshot, perception}]
        self._cancel_lock = threading.Lock()
        self._active_step_cancel = None
        self._active_step_force = False
        self._active_reasoning_request_id = None
        self._pending_cancel = False
        self._resume_state = resume_state if isinstance(resume_state, dict) else None
        self.extraction_batch_id = f"{key}:{dataset}:{uuid.uuid4().hex[:12]}"
        self.perceiver = perceiver or site_perceiver()
        self.qwen_health = self.perceiver.health()
        self.speech_cache_path = RUNS / "interactive_audio_transcript_cache.json"
        self._speech_cache = PersistentTranscriptCache(self.speech_cache_path)
        self.person_profile_cache = PersistentTranscriptCache(
            RUNS / "interactive_person_profile_cache.json"
        )
        self.retrieval_cache = PersistentAdviceCache(
            RUNS / "interactive_scene_retrieval_cache.sqlite3", ROOT
        )
        self.scene_plan_cache = PersistentAdviceCache(
            RUNS / "interactive_scene_plan_cache.sqlite3", ROOT
        )
        self.grounding_cache = PersistentObjectGroundingCache(
            RUNS / "interactive_object_grounding_cache.sqlite3", ROOT
        )
        self._build()

    def _install_operation_audit(self):
        """Record actual memory API calls made by this interactive session."""
        self.operation_log = []
        self._active_step_no = None
        self._active_step_title = None
        tracked = (
            "add", "query", "query_hierarchical", "update", "update_feedback",
            "highlight", "delete", "sort", "combine", "compress", "update_profile",
            "find_merge_candidates", "merge_moments", "refresh_source_moment",
        )
        for name in tracked:
            original = getattr(self.mem, name, None)
            if not callable(original):
                continue

            def audited(*args, __name=name, __original=original, **kwargs):
                started = time.perf_counter()
                before_moment = None
                before_profile = None
                if __name in ("update", "refresh_source_moment", "update_feedback",
                              "highlight", "delete"):
                    mid = str(args[0] if args else kwargs.get("moment_id", ""))
                    item = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid)
                    before_moment = copy.deepcopy(item) if isinstance(item, dict) else None
                if __name == "merge_moments":
                    archive = self.mem.memory.get("layer7", {}).get("moments", {})
                    source_id = str(args[0] if args else kwargs.get("source_id", ""))
                    target_id = str(args[1] if len(args) > 1 else kwargs.get("target_id", ""))
                    before_moment = {
                        "source": copy.deepcopy(archive.get(source_id, {})),
                        "target": copy.deepcopy(archive.get(target_id, {})),
                    }
                if __name == "update_profile":
                    before_profile = copy.deepcopy(
                        self.mem.memory.get("layer6", {}).get("profile", {})
                    )
                try:
                    result = __original(*args, **kwargs)
                except Exception as exc:
                    self._append_operation(__name, args, kwargs, None,
                                           started, error=exc,
                                           before_moment=before_moment,
                                           before_profile=before_profile)
                    raise
                self._append_operation(
                    __name, args, kwargs, result, started,
                    before_moment=before_moment,
                    before_profile=before_profile,
                )
                return result

            setattr(self.mem, name, audited)

    def _append_operation(self, name, args, kwargs, result, started,
                          error=None, before_moment=None, before_profile=None):
        """Append a factual audit event with the affected memory content."""
        status = "failed" if error else (
            "blocked" if result is False or
            (isinstance(result, dict) and result.get("ok") is False)
            else "success"
        )
        if name in ("update", "refresh_source_moment") and result is False:
            mid = str(args[0] if args else kwargs.get("moment_id", ""))
            exists = mid in self.mem.memory.get("layer7", {}).get("moments", {})
            status = "no_change" if exists else "blocked"
        summary = self._operation_summary(
            name, args, kwargs, result, error, before_moment, before_profile
        )
        self.operation_log.append({
            "seq": len(self.operation_log) + 1,
            "time": time.strftime("%H:%M:%S"),
            "step": self._active_step_no,
            "step_title": self._active_step_title or "",
            "operation": name,
            "status": status,
            "summary": summary,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        })

    @staticmethod
    def _brief_value(value, limit=110):
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value if value is not None else "—")
        return " ".join(text.split())[:limit]

    def _moment_detail(self, moment):
        """Return the visible, factual fields of a memory without mutating it."""
        if not isinstance(moment, dict):
            return {}
        return {
            "id": str(moment.get("id") or ""),
            "timestamp": moment.get("timestamp") or "",
            "scene": moment.get("scene") or "",
            "action": moment.get("user_action") or "",
            "interaction_details": moment.get("interaction_details") or "",
            "content_summary": moment.get("content_summary") or "",
            "people": moment.get("people") or [],
            "person_observations": copy.deepcopy(moment.get("person_observations") or []),
            "temporal_facts": copy.deepcopy(moment.get("temporal_facts") or []),
            "source_time": copy.deepcopy(moment.get("source_time") or {}),
            "location": moment.get("location") or "",
            "activity": moment.get("activity") or "",
            "objects": moment.get("objects") or [],
            "notes": moment.get("notes") or "",
            "highlighted": bool(moment.get("highlighted")),
            "highlight_source": moment.get("highlight_source") or "",
            "highlighted_at": moment.get("highlighted_at") or "",
            "merged_into": moment.get("merged_into") or "",
            "merged_sources": moment.get("merged_sources") or [],
            "merge_history": moment.get("merge_history") or [],
        }

    def _operation_summary(self, name, args, kwargs, result, error,
                           before_moment=None, before_profile=None):
        if error:
            return f"{name} 失败：{str(error)[:140]}"
        arg0 = args[0] if args else kwargs.get("moment_id")
        if name == "add":
            mid = str(result or "")
            item = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            text = str(item.get("scene") or item.get("user_action") or "").strip()
            text = text.replace("\n", " ")[:100]
            return f"写入记忆 {mid}" + (f" · {text}" if text else "")
        if name in ("query", "query_hierarchical"):
            query = str(arg0 if arg0 is not None else kwargs.get("query_text", ""))[:80]
            hits = result if isinstance(result, list) else []
            examples = []
            for hit in hits[:3]:
                if not isinstance(hit, dict):
                    continue
                moment = hit.get("moment") if isinstance(hit.get("moment"), dict) else hit
                mid = str(moment.get("id") or "无 ID")
                content = str(moment.get("scene") or moment.get("user_action") or "").replace("\n", " ")
                layer = f"L{hit['layer']} · " if hit.get("layer") else ""
                examples.append(f"{layer}{mid}: {content[:100]}")
            detail = "；命中：" + "；".join(examples) if examples else ""
            return f"检索「{query}」· 命中 {len(hits)} 条{detail}"
        if name == "find_merge_candidates":
            candidates = result if isinstance(result, list) else []
            examples = [
                f"{row.get('source_id')}→{row.get('target_id')} "
                f"({row.get('similarity', 0):.0%})"
                for row in candidates[:3]
            ]
            suffix = " · " + "；".join(examples) if examples else ""
            return f"相似事件候选 {len(candidates)} 组（需人工确认）{suffix}"
        if name in ("update", "refresh_source_moment"):
            mid = str(arg0 or "")
            updates = args[1] if len(args) > 1 else kwargs.get("updates", {})
            exists = mid in self.mem.memory.get("layer7", {}).get("moments", {})
            if result is False:
                return f"更新记忆 {mid} · 字段未变化" if exists else f"更新记忆 {mid} · 未找到"
            after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            changes = []
            if isinstance(updates, dict):
                for field, new_value in updates.items():
                    old_value = (before_moment or {}).get(field)
                    actual_value = after.get(field)
                    changes.append(
                        f"{field}：{self._brief_value(old_value)} → {self._brief_value(actual_value)}"
                    )
            verb = "刷新源记忆" if name == "refresh_source_moment" else "更新记忆"
            return f"{verb} {mid}" + (" · " + "；".join(changes) if changes else "")
        if name == "update_feedback":
            mid = str(arg0 or "")
            corrections = kwargs.get("corrections", {})
            if len(args) > 1 and isinstance(args[1], dict):
                corrections = args[1]
            detail = "；".join(
                f"{key}：{self._brief_value((before_moment or {}).get(key))} → {self._brief_value(value)}"
                for key, value in corrections.items()
            ) if isinstance(corrections, dict) else ""
            confirmed = kwargs.get("confirmed", False)
            rating = kwargs.get("user_rating")
            return f"反馈修正 {mid}" + (f" · {detail}" if detail else "") + (
                f" · 确认={'是' if confirmed else '否'} · 评分={rating if rating is not None else '未评分'}"
            )
        if name == "highlight":
            mid = str(arg0 or "")
            item = before_moment or self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            content = str(item.get("scene") or item.get("user_action") or "").replace("\n", " ")[:100]
            result_text = "已标记重点" if result else "标记失败"
            return f"{result_text} {mid}" + (f" · {content}" if content else "")
        if name == "delete":
            mid = str(arg0 or "")
            reason = kwargs.get("reason", "manual")
            item = before_moment or {}
            content = str(item.get("scene") or item.get("user_action") or "").replace("\n", " ")[:100]
            result_text = "已删除" if result else "被保护/未找到"
            return f"{result_text} {mid} · 原因：{reason}" + (f" · 内容：{content}" if content else "")
        if name == "sort":
            analysis = arg0 if isinstance(arg0, dict) else {}
            tags = sum(len(v) for v in analysis.values() if isinstance(v, dict))
            refs = sum(len(ids) for dim in analysis.values() if isinstance(dim, dict)
                       for ids in dim.values() if isinstance(ids, list))
            return f"整理索引 · {len(analysis)} 个维度、{tags} 个标签、{refs} 条引用"
        if name == "combine":
            details = result if isinstance(result, dict) else {}
            aliases = details.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            examples = []
            for alias in aliases[:4]:
                if not isinstance(alias, dict):
                    continue
                dimension = str(alias.get("dimension") or "索引")
                old = self._brief_value(alias.get("from"), 48)
                new = self._brief_value(alias.get("to"), 48)
                affected = int(alias.get("affected_moments") or 0)
                item = f"{dimension}「{old}」→「{new}」·影响 {affected} 条"
                sample_ids = alias.get("sample_moment_ids")
                if isinstance(sample_ids, list) and sample_ids:
                    item += "·例 " + ", ".join(str(mid) for mid in sample_ids[:3])
                    if affected > len(sample_ids[:3]):
                        item += f" 等 {affected} 条"
                if alias.get("overlap_moments"):
                    item += f"·两索引共有 {alias['overlap_moments']} 条"
                if alias.get("field_updates"):
                    item += f"·字段更新 {alias['field_updates']} 条"
                if alias.get("list_duplicates_removed"):
                    item += f"·列表去重 {alias['list_duplicates_removed']} 项"
                examples.append(item)
            summary = f"规范化标签 {len(aliases)} 组"
            if examples:
                summary += " · " + "；".join(examples)
                if len(aliases) > len(examples):
                    summary += f"；另有 {len(aliases) - len(examples)} 组"
            elif not aliases:
                summary += "（无有效别名需要合并）"
            if details.get("updated_moments"):
                summary += f" · 记忆记录更新 {details['updated_moments']} 条"
            return summary
        if name == "merge_moments":
            details = result if isinstance(result, dict) else {}
            if not details.get("ok"):
                return str(details.get("message") or "合并未执行")
            if details.get("status") == "already_merged":
                return f"来源 {details.get('source_id')} 已归并到 {details.get('target_id')}（无重复改动）"
            source_id = details.get("source_id") or "?"
            target_id = details.get("target_id") or "?"
            fields = "、".join(details.get("merged_fields") or []) or "没有新增列表字段"
            conflicts = len(details.get("scalar_conflicts") or [])
            media = details.get("media_preserved") or []
            retained = sum(bool(item.get("visual")) for item in media)
            return (f"合并事件 {source_id} → {target_id} · 列表字段：{fields}"
                    f" · 标量冲突留在来源并记录 {conflicts} 项"
                    f" · {retained}/{len(media)} 条来源保留多模态引用")
        if name == "compress":
            payload = arg0 if isinstance(arg0, dict) else {}
            layers = "、".join(k.upper() for k in payload if k in ("layer4", "layer5", "layer6"))
            return f"压缩整理摘要 · 更新 {layers or '长期摘要'}"
        if name == "update_profile":
            fields = list(arg0) if isinstance(arg0, dict) else []
            profile = result if isinstance(result, (dict, list)) else self.mem.memory.get("layer6", {}).get("profile", {})
            def count_attrs(node):
                if isinstance(node, dict):
                    if "value" in node and "stale" in node:
                        return (1, int(bool(node.get("stale"))))
                    totals = [count_attrs(value) for value in node.values()]
                elif isinstance(node, list):
                    totals = [count_attrs(value) for value in node]
                else:
                    return (0, 0)
                return (sum(x[0] for x in totals), sum(x[1] for x in totals))
            total, stale = count_attrs(profile)
            detail = f"画像 {total} 项 · 过时 {stale} 项" if total else "暂无画像字段 · 衰减状态已检查"

            def conflicts(node, path=""):
                found = []
                if isinstance(node, dict) and "value" in node:
                    for old_value in node.get("history", []) or []:
                        if not isinstance(old_value, dict):
                            continue
                        found.append({
                            "path": path,
                            "old": old_value.get("value"),
                            "new": node.get("value"),
                            "resolution": old_value.get("resolution") or {},
                        })
                elif isinstance(node, dict):
                    for key, value in node.items():
                        found.extend(conflicts(value, f"{path}.{key}" if path else str(key)))
                elif isinstance(node, list):
                    for index, value in enumerate(node):
                        found.extend(conflicts(value, f"{path}[{index + 1}]"))
                return found

            before_conflicts = conflicts(before_profile or {})
            before_signatures = {
                (item["path"], json.dumps(item["old"], ensure_ascii=False, sort_keys=True, default=str),
                 item["resolution"].get("resolved_at", ""))
                for item in before_conflicts
            }
            new_conflicts = [
                item for item in conflicts(profile)
                if (item["path"], json.dumps(item["old"], ensure_ascii=False, sort_keys=True, default=str),
                    item["resolution"].get("resolved_at", "")) not in before_signatures
            ]
            basis_labels = {
                "observations": "观测次数更多",
                "recency": "最近观察",
                "confidence": "置信度更高",
                "existing_value_tiebreak": "条件相同时保留旧值",
            }
            conflict_note = ""
            if new_conflicts:
                examples = []
                for item in new_conflicts[:3]:
                    resolution = item["resolution"]
                    examples.append(
                        f"{item['path']}「{self._brief_value(item['old'], 32)}」→"
                        f"「{self._brief_value(item['new'], 32)}」"
                        f"（{basis_labels.get(resolution.get('basis'), '裁决依据未记录')}）"
                    )
                conflict_note = f" · 新增冲突裁决 {len(new_conflicts)} 项：{'；'.join(examples)}"
                if len(new_conflicts) > len(examples):
                    conflict_note += f"；另有 {len(new_conflicts) - len(examples)} 项"
                conflict_note += " · 旧值保留在画像历史"
            names = f" · 字段 {chr(12289).join(map(str, fields[:8]))}" if fields else ""
            return f"更新用户画像 · {detail}{conflict_note}{names}"
        return name

    # ------------------------------------------------------------------
    # 感知 → 记忆写入
    # Qwen 看视频产出记忆条目；数据集 caption / ground_truth 只用于评估。
    # ------------------------------------------------------------------

    def cancel_current(self):
        """Interrupt the active step; completed items remain available for retry."""
        with self._cancel_lock:
            event = self._active_step_cancel
            if event is None:
                # Queue the stop if it arrives between STEP_LOCK acquisition and
                # run_step installing its cancellation event.
                self._pending_cancel = True
            else:
                event.set()
        cancel_active = getattr(self.perceiver, "cancel_active", None)
        if callable(cancel_active):
            threading.Thread(target=cancel_active, daemon=True,
                             name="cancel-active-perception").start()
        return {"requested": True, "model_cancel_sent": False,
                "model_cancel_pending": callable(cancel_active)}

    def _set_step_progress(self, **updates):
        current = dict(self.step_progress or {})
        completed = list(current.get("completed") or [])
        if "completed_item" in updates:
            item = updates.pop("completed_item")
            if item:
                completed.append(item)
                completed = completed[-4:]
        current.update(updates)
        current["completed"] = completed
        current["updated_at"] = time.time()
        self.step_progress = current

    @staticmethod
    def _progress_text(m, info):
        fields = info.get("qwen") or {}
        scene = fields.get("scene") or ""
        action = fields.get("user_action") or ""
        people = fields.get("people") or []
        if isinstance(people, list):
            people = ", ".join(str(x) for x in people[:5])
        text = " · ".join(x for x in (scene, action) if x)
        if people:
            text += f" · People: {people}"
        related = info.get("retrieved_memories") or []
        before_rows = info.get("retrieved_memories_before_write") or []
        before_meta = info.get("retrieval_before_write_meta") or {}
        if before_rows:
            text += f" · 写入前召回历史记忆 {len(before_rows)} 条"
        elif before_meta:
            text += " · 写入前未召回相关历史记忆"
        return {"id": m.get("id", ""), "text": text or info.get("error") or "处理完成",
                "retrieved_memories": related,
                "retrieval_meta": info.get("retrieval_meta") or {},
                "retrieved_memories_before_write": before_rows,
                "retrieval_before_write_meta": before_meta}

    def _transcribe_audio_evidence(self, multimodal, source_segment_id, video_path=None,
                                   clip_start_s=0.0, clip_end_s=None, progress=None,
                                   timing_sink=None):
        """Transcribe source-video audio only; never read benchmark text or captions."""
        media = multimodal if isinstance(multimodal, dict) else {}
        jobs = []
        decode_started = time.perf_counter()

        def add_audio_timing(key, started):
            if timing_sink is not None:
                timing_sink[key] = timing_sink.get(key, 0) + round(
                    (time.perf_counter() - started) * 1000)
        # Decode source-video audio transiently for transcription. Only the text
        # transcript and explicit spoken facts enter memory; no audio clip is saved.
        if video_path:
            try:
                start = max(0.0, float(clip_start_s or 0.0))
                end = float(clip_end_s) if clip_end_s is not None else start + 30.0
                if end > start:
                    cursor = start
                    while cursor < end:
                        duration = min(20.0, end - cursor)
                        audio_bytes = _run_ffmpeg([
                            "-ss", f"{cursor:.3f}", "-i", str(video_path),
                            "-t", f"{duration:.3f}", "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
                        ])
                        if audio_bytes:
                            jobs.append({"bytes": audio_bytes, "name": "source_audio.wav",
                                         "mime": "audio/wav", "offset": cursor,
                                         "duration": duration})
                        cursor += duration
            except Exception:
                # A decode failure only omits transcript text; memory writing
                # remains available and no legacy audio asset is read as fallback.
                jobs = []
        add_audio_timing("audio_decode_ms", decode_started)
        if timing_sink is not None:
            timing_sink["audio_chunks"] = len(jobs)
            timing_sink.setdefault("speech_cache_hits", 0)
            timing_sink.setdefault("speech_cache_misses", 0)
        rows = []
        for job in jobs:
            audio_data = job.get("bytes")
            if audio_data is None:
                try:
                    audio_data = job["path"].read_bytes()
                except OSError:
                    continue
            cache_key = "whisper-source-audio-v2:" + hashlib.sha256(audio_data).hexdigest()
            lookup_started = time.perf_counter()
            decoded = self._speech_cache.get(cache_key)
            add_audio_timing("speech_cache_lookup_ms", lookup_started)
            if timing_sink is not None:
                timing_sink["speech_cache_hits" if decoded is not None else "speech_cache_misses"] += 1
            if decoded is None:
                if progress:
                    progress("speech_transcription", "正在把视频语音转成文字")
                asr_started = time.perf_counter()
                try:
                    response = requests.post(
                        self.perceiver.next_api_url() + "/transcribe",
                        files={"audio": (job["name"], audio_data, job["mime"])},
                        timeout=(5, 180),
                    )
                    response.raise_for_status()
                    decoded = response.json()
                    if not isinstance(decoded, dict):
                        continue
                    decoded, _inserted = self._speech_cache.put_if_absent(
                        cache_key, decoded
                    )
                except Exception:
                    # Audio is optional evidence; keep the visual memory write available.
                    continue
                finally:
                    add_audio_timing("whisper_transcription_request_ms", asr_started)
            language = str(decoded.get("language") or "unknown")[:24]
            offset = float(job.get("offset") or 0)
            segments = decoded.get("segments") or []
            if segments:
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    text = str(segment.get("text") or "").strip()
                    if not is_meaningful_transcript_text(text, language):
                        continue
                    start = offset + max(0.0, float(segment.get("start") or 0))
                    end = offset + max(float(segment.get("start") or 0),
                                       float(segment.get("end") or 0))
                    rows.append({"start_s": round(start, 2), "end_s": round(end, 2),
                                 "language": language, "text": text[:1000],
                                 "source_segment_id": str(source_segment_id or "")[:128]})
            else:
                text = str(decoded.get("full_text") or "").strip()
                if is_meaningful_transcript_text(text, language):
                    rows.append({"start_s": round(offset, 2),
                                 "end_s": round(offset + float(job.get("duration") or 0), 2),
                                 "language": language, "text": text[:1000],
                                 "source_segment_id": str(source_segment_id or "")[:128]})
        if timing_sink is not None:
            timing_sink["transcript_segments"] = len(rows)
        return "\n".join(
            f"[audio {row['start_s']:.2f}–{row['end_s']:.2f}s; {row['language']}] {row['text']}"
            for row in rows
        )[:4000]

    def _historical_people_mentioned_in_speech(self, transcript, current_source_id):
        """Find exact named-person matches in earlier clips and load one face reference.

        The ordering comes from the selected dataset, so a stale record for this
        same source clip or a later clip cannot leak into the current analysis.
        """
        text = str(transcript or "")
        ordered_ids = [str(row.get("id") or "") for row in self.moments
                       if isinstance(row, dict)]
        current_index = next((index for index, value in enumerate(ordered_ids)
                              if value == str(current_source_id or "")), None)
        if current_index is None:
            return []
        earlier_ids = set(ordered_ids[:current_index])
        if not earlier_ids:
            return []

        def exact_mention(name):
            name = " ".join(str(name or "").split()).strip()
            if not name or len(name) < 2:
                return False
            if re.fullmatch(r"(?i)(?:man|woman|person|speaker|guest|employee|user)\s*\d*", name):
                return False
            if re.search(r"[A-Za-z]", name):
                return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(name) +
                                      r"(?![A-Za-z0-9])", text, re.I))
            return name in text

        archive = (self.mem.memory.get("layer7") or {}).get("moments") or {}
        candidates = {}
        for moment_id, moment in archive.items():
            if not isinstance(moment, dict) or moment.get("stale") or moment.get("merged_into"):
                continue
            provenance = moment.get("source_provenance") or {}
            source_id = str(provenance.get("source_segment_id") or
                            (moment.get("source_time") or {}).get("source_segment_id") or
                            (moment.get("multimodal") or {}).get("source_id") or "")
            if source_id not in earlier_ids:
                continue
            source_index = ordered_ids.index(source_id)
            for person in moment.get("person_observations") or []:
                if not isinstance(person, dict):
                    continue
                name = " ".join(str(person.get("readable_name") or person.get("entity") or "").split()).strip()
                if not exact_mention(name):
                    continue
                key = name.casefold()
                existing = candidates.get(key)
                if existing and existing.get("source_index", -1) >= source_index:
                    continue
                roles = [" ".join(str(value).split()).strip()[:80]
                         for value in person.get("roles") or []
                         if isinstance(value, str) and value.strip()]
                profile = " ".join(str(person.get("profile_context") or "").split()).strip()[:420]
                if not profile:
                    profile = " ".join(str(person.get("appearance") or "").split()).strip()[:300]
                entry = {"name": name, "roles": roles[:4], "profile_context": profile,
                         "source_segment_id": source_id,
                         "source_index": source_index,
                         "evidence_ref": copy.deepcopy(person.get("evidence_ref") or {})}
                mm = moment.get("multimodal") or {}
                visual_items = (mm.get("entity_visuals") or {}).get("items") or []
                visual_item = next((item for item in visual_items
                                    if isinstance(item, dict)
                                    and str(item.get("entity") or "").casefold() == key
                                    and item.get("entity_type") == "person"), None)
                refs = (visual_item or {}).get("frame_refs") or []
                face_ref = next((ref for ref in refs if isinstance(ref, dict)
                                 and ref.get("visual_role") == "face_crop"
                                 and ref.get("identity_verified")), None)
                if face_ref:
                    asset = str(face_ref.get("asset") or "")
                    try:
                        path = (self.media_store.root / asset).resolve()
                        path.relative_to(self.media_store.root.resolve())
                        image_bytes = path.read_bytes()
                        if image_bytes and len(image_bytes) <= 120_000:
                            entry["reference_image_bytes"] = image_bytes
                            entry["reference_image_mime"] = str(face_ref.get("mime") or "image/jpeg")
                            entry["reference_image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
                    except (OSError, ValueError):
                        pass
                candidates[key] = entry
        ordered = sorted(candidates.values(), key=lambda row: row.get("source_index", -1), reverse=True)
        return [{key: value for key, value in row.items() if key != "source_index"}
                for row in ordered[:3]]

    def _enrich_person_profiles_from_speech(self, observations, transcript,
                                            frames, source_segment_id,
                                            visual_facts=None,
                                            document_evidence=None,
                                            cancel_event=None, progress=None,
                                            visual_context=None, related_memories=None):
        """Reconcile people, documents, and event claims against source evidence."""
        rows = [dict(row) for row in observations or [] if isinstance(row, dict)]
        for row in rows:
            if isinstance(row.get("roles"), list):
                row["roles"] = _clean_episode_roles(row["roles"], row.get("entity"))
        speech = str(transcript or "").strip()[:4000]
        image_frames = [frame for frame in (frames or [])
                        if isinstance(frame, dict) and frame.get("bytes")][:7]
        documents = _collect_document_evidence(document_evidence, visual_facts)
        historical_people = self._historical_people_mentioned_in_speech(
            speech, source_segment_id)
        frame_times = [round(float(frame.get("at_s") or 0), 3) for frame in image_frames]
        rows = _link_document_fields_to_people(
            rows, documents, visual_facts, source_segment_id, frame_times)
        if not speech or not image_frames:
            if os.environ.get("MEMORY_PROFILE_TRACE"):
                print("PROFILE_TRACE " + json.dumps({"source": str(source_segment_id or ""),
                      "early_return": "speech_or_frames_missing", "has_speech": bool(speech),
                      "frames": len(image_frames), "people": [row.get("entity") for row in rows]},
                      ensure_ascii=False), flush=True)
            return rows, "", "", "", []
        # Keep evidence-backed local address matching available when the optional
        # joint vision-language enrichment service is offline or rejects a response.
        fallback_lines = []
        for line in speech.splitlines():
            match = _AUDIO_EVIDENCE_LINE_RE.match(line.strip())
            if not match:
                continue
            try:
                start_s, end_s = float(match.group("start")), float(match.group("end"))
            except (TypeError, ValueError):
                continue
            text = str(match.group("text") or "").strip()
            if text:
                fallback_lines.append((text, start_s, end_s))
        fallback_facts = [{
            "subject_type": "person", "subject": "speech interaction",
            "predicate": "utterance", "value": text,
            "modality": "speech", "status": "stated", "confidence": "medium",
            "evidence": text,
            "evidence_ref": {"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "speech", "audio_start_s": round(start_s, 3),
                             "audio_end_s": round(end_s, 3)},
        } for text, start_s, end_s in fallback_lines]
        fallback_rows = [dict(row) for row in rows]
        attach_spoken_identity(fallback_rows, fallback_facts, visual_facts=visual_facts)
        cache_key = person_profile_cache_key(
            source_segment_id, speech, rows, image_frames, visual_facts, documents,
            historical_people=historical_people,
            visual_context=visual_context, related_memories=related_memories)
        cached = self.person_profile_cache.get(cache_key)
        cached_people = cached.get("people") if isinstance(cached, dict) else None
        cached_profile_valid = (isinstance(cached_people, list)
                                and (bool(cached_people)
                                     or cached.get("_empty_result_validated") is True))
        if cached_profile_valid:
            payload = cached
            profile_cache_hit = True
        else:
            profile_cache_hit = False
            if cancel_event is not None and cancel_event.is_set():
                return fallback_rows, "", "", "", []
            frame_map = [{"frame": index + 1,
                          "time_s": round(float(frame.get("at_s") or 0), 3)}
                         for index, frame in enumerate(image_frames)]
            candidates = [{"candidate_id": f"person_{index + 1}",
                           "entity": str(row.get("entity") or "")[:100],
                           "appearance": str(row.get("appearance") or "")[:220],
                           "face_features": str(row.get("face_features") or "")[:160],
                           "apparent_age_range": str(row.get("apparent_age_range") or ""),
                           "apparent_age_basis": str(row.get("apparent_age_basis") or ""),
                           "anchor_frame_index": row.get("frame_index"),
                           "visual_evidence": str(row.get("evidence") or "")[:180]}
                          for index, row in enumerate(rows[:16])]
            prompt = (
                "Reconcile the ordered frames, existing visual facts, legible document text, and time-coded speech "
                "from this same clip into a concise episodic memory. LANGUAGE RULE: Write every generated human-readable semantic field in natural English, including scene_context, location_context, user_action, people labels, identities, roles, relationships, profile_context, fact subjects, predicates, values, actors, and affected_people. Translate supported Chinese role/kinship terms (for example 姥姥 → grandmother, 妈妈 → mother, 医生 → doctor, 路人 → passerby); never leave Chinese in those semantic fields. Keep proper names in established Latin spelling, or use Hanyu Pinyin when only a Chinese name is available; do not translate a name’s meaning. Exact transcript/document quotations and evidence fields may remain in their original language for auditability. Put original-language words only in exact quote/evidence fields; never copy them into a summary or semantic value. This is a language requirement, not permission to guess an identity. Use only those inputs; do not use filenames, "
                "dataset labels, prior assumptions, or outside knowledge. Candidate IDs identify the supplied visible "
                "people and are unique even if labels repeat. Track a candidate only when the frames establish visual "
                "continuity; otherwise keep the observations separate. Do not merge background people based only on "
                "similar clothing or posture. "
                "Treat transcript segments as undiarized unless a speaker is explicitly identified. Separate speaker, "
                "addressee, visible person, and first-person user; do not assign a voice to a visible person merely "
                "because a frame occurs at the same time. Use synchronized speech cues when available. If they are absent, "
                "a visible central participant may still receive a calibrated episode-role inference when the full visual "
                "context, communicative actions, and speech act converge and no plausible competing speaker/candidate exists; "
                "do not claim certain speaker diarization from that inference. A single open-mouth frame by itself is "
                "insufficient. If multiple people are visible, require cues that distinguish the speaker from listeners. If one utterance addresses multiple roles, split its clauses "
                "and keep each addressee separate; an instruction to tell someone something names the instruction's "
                "target, not necessarily the current listener. Never copy one turn's addressee to every fact in that "
                "utterance. Compare ordered frame times with time-coded speech and use "
                "medium confidence when synchronized mouth movement supports the match. A kinship vocative identifies "
                "the addressee, not the speaker. When the first-person camera wearer is addressed as Dad, that identifies "
                "the addressee as the speaker's father; if a unique visible candidate's repeated, time-aligned mouth "
                "movement identifies that candidate as the speaker, label them as the user's son at medium confidence. "
                "Do not label an addressed visible listener as the son, or infer the speaker from the word Dad alone. "
                "This supports only a relative "
                "older-generation inference for the user when the son is visibly an adult; never infer an exact age for "
                "the unseen user. Store a coarse apparent-age band only for the visible person when multiple compatible "
                "facial-age cues and gray/white hair support it; hair color alone is insufficient. Cite the visual cues. "
                "Parse the full exchange into ordered turns: a name used in "
                "direct address identifies the listener, not the speaker; a later answer, thanks, surprise, or other "
                "response may identify the visible listener when the timing and visible reaction agree. Keep the initial "
                "greeting and the reply with different speakers when dialogue and reaction support that distinction. "
                "Use medium confidence for turn-linked speaker inference without diarization, and unknown when the "
                "sequence does not distinguish speakers. Adjacent ASR chunks are not automatically separate turns; "
                "keep consecutive statements with the same address/topic under the same speaker unless diarization, "
                "a clear lexical response, or other direct turn-change evidence distinguishes speakers. A time boundary "
                "or facial-expression change alone is not a reply. A visible person may continue speaking across several "
                "transcript chunks. Infer participant roles from the full "
                "request-response sequence, visible interaction, setting, observed duties, and speech, distinguishing a service recipient from a provider. "
                "Roles and relationships are specific to this episode unless repeated independent evidence supports a durable profile. "
                "For every visually central or interacting person, provide a source-grounded episode identity when possible: "
                "speaker, addressee, family relation, customer, service worker, clinician, presenter, bystander, or passerby. "
                "When one central visible participant is shown communicating and the time-coded speech describes a consequential "
                "role performed in the same setting, infer the narrow episode role at medium confidence even if the addressee is "
                "off-screen; cite the role-bearing utterance and the visual frame(s). Keep the off-screen addressee unidentified. "
                "Do not leave people empty when this evidence supports a candidate role. "
                "Use the person's observed action and the complete dialogue; do not infer seniority or a permanent occupation "
                "from clothing, politeness, or a single transaction. A service worker who welcomes a guest and completes "
                "check-in/payment is service or front-desk staff unless authority as a manager is directly evidenced. "
                "If no relationship or function is supported, state 'unidentified person; relationship unknown in this clip' "
                "and preserve the visual label as an alias; use 'stranger' or 'passerby' only when the scene supports that status. "
                "Infer episode roles from what a person demonstrably does in the full interaction, not from generic appearance labels. "
                "A role-bearing speech act or action (such as directing a consequential workplace decision, providing care, serving a guest, "
                "or accepting a payment) can support a narrow functional role when the surrounding scene and interaction agree; describe it "
                "as likely/inferred rather than a permanent occupation. Choose a concise, familiar episode label that preserves the role's "
                "meaning (for example, 'boss / manager' for a supported employer representative), rather than reverting to 'man' or 'woman'. "
                "Link a role-bearing speaker to a visible candidate using the complete "
                "clip: direct synchronized speaking cues are strongest; when those are unavailable, a uniquely visible central participant "
                "whose communicative actions and scene context consistently match the speech act may support a medium-confidence episode-level "
                "role, but state the inference and cite both speech and visual evidence. Do not make this link when multiple candidates or "
                "competing off-screen speakers remain plausible. Always distinguish speaker from addressee: second-person wording such as 'you' "
                "identifies the target, not the speaker. For authority or instruction speech, assign the role to the supported speaker, never "
                "to the addressed target merely because the target is described in the utterance. Do not infer a role from clothing, venue alone, "
                "ambiguous wording, or quoted speech. If evidence does not resolve the speaker or role, preserve the quote as attributed speech "
                "and leave the person role unknown. Do not attribute a request or reply to the user from transcript order alone. "
                "When inferring the user's episode role, combine the full request-response sequence, first-person viewpoint, "
                "and visible interaction; cite the most supportive response or interaction, and keep confidence medium "
                "if a speaker is inferred from turn-taking. Earlier named-person context, if supplied, may resolve a "
                "current addressee only when the exact name is spoken and a current visible candidate is uniquely linked "
                "to that exchange; use a supplied face reference to compare identity, never name someone from context alone. "
                "A role from an earlier episode can support identity continuity but is not by itself a permanent user job. "
                "For documents, keep exact printed claims separate from verified facts and distinguish the holder/presenter "
                "from the document subject: physical handling proves who holds or presents a document, not who is named in it. "
                "A medical record's name, age, and diagnosis belong to its patient; do not transfer them to a visible clinician. "
                "If a care provider directly addresses the camera wearer by a matching name and says 'your' about the same "
                "condition, link the patient facts to user and label the visible care provider as doctor only when the clinical "
                "setting, attire, record presentation, and advice support that role. Otherwise leave the subject unresolved. "
                "Never calculate an age from a birth date. Do not expose "
                "unneeded account or government-ID numbers. "
                "When visible candidates and speech or document evidence support a relationship, name, or episode role, "
                "return a people row with the exact candidate_id, evidence quote, and current frame index. Do not return "
                "an empty people list merely because the role is not a proper name; keep the label empty if unresolved, "
                "but preserve the supported relationship/context. Never guess a candidate when the mapping is ambiguous. "
                "Extract atomic people attributes and event facts. A fact may cite exact transcript text, exact document "
                "text, visible frame indexes, or multiple sources. Do not emit a fact unless its citations support its "
                "subject, actor, and value. Treat meaningful nondialogue speech such as product claims, warnings, "
                "instructions, offers, or narration as episode content even when speaker identity and turn-taking are "
                "unknown. Record it with an exact quote, unknown speaker when needed, and attributed wording such as "
                "'the speaker claims'; do not convert a claim into verified truth. Omit only unintelligible/filler or "
                "clearly unrelated ambient speech. Preserve uncertainty rather than filling gaps. For inferred links, use medium "
                "or low confidence and say what the evidence supports; do not infer mental state, intent, relationship, "
                "payment, diagnosis, or completion from silence or aftermath alone. For money exchanges, distinguish a quoted price, intent to pay, cash offered, recipient acceptance, and confirmed completion; preserve an offer as an offer unless acceptance/completion is directly supported. A directly quoted third-party opinion "
                "about the user is an attributed claim, not a verified personality trait. Preserve the exact quote as a "
                "source-linked fact with subject=user, actor=the supported speaker, status=quoted_unverified_claim, and "
                "confidence no higher than medium; keep it retrievable as what that person said, but do not promote it "
                "into the user's stable profile unless the user confirms it or independent evidence repeats it. A warning "
                "is an interaction/event, not proof that the user was previously deceived. Whenever speech explicitly "
                "describes the addressee's tendency, create a separate attributed-claim fact about that addressee, "
                "cite the exact quote, and keep the speaker as actor; do not convert it into a verified trait. Treat a first-person user's visible "
                "hands/body as the user's action, never as another person. When a visible person presents an item toward "
                "the lens, attribute the offer/presentation to that person; do not reverse it into a user-given action "
                "unless the frame directly shows the user's hand carrying it outward. "
                "Return scene_context as one concise, high-level situation summary: state what the clip is about overall (for example a family visit, hotel check-in, meal, medical consultation, or emergency), integrating the visible setting and the most meaningful supported speech. Prefer occasion, purpose, or outcome over a chronology of gestures or dialogue. When spoken claims or instructions define why an item is being presented or how people relate, integrate that meaning; for example, if a person displays a product while claiming benefits, describe a product presentation or promotion and attribute the benefits as the speaker's claims. Do not reduce a meaningful spoken situation to someone merely speaking to camera. Infer a social occasion such as a likely date or friendly meeting only when multiple independent cues from the setting, reciprocal interaction, and dialogue converge; use calibrated wording such as 'appears to be' for an inference, and never infer a relationship from gender, co-presence, or a restaurant alone. For a hazard, describe the user's response only when ordered frames support the relative movement; do not infer movement from a camera pan alone. Do not list actions, dialogue turns, speaker labels, transcript facts, or audit language; those belong in Scene and Temporal Facts. If a broader purpose is unsupported, use a neutral setting-and-activity gist. When identity is uncertain, say a speaker discusses, offers, or reports something without assigning the voice to a visible person or user from timing alone. Attribute unverifiable claims as claims. Cite an exact transcript quote in context_evidence_quotes for any speech-derived summary. Also return user_action when supported, with exact transcript quotes. Do not claim an unobserved agent suggestion unless supplied prior context supports it. "
                "Return JSON only with this schema: {\"people\":[{\"candidate_id\":\"person_1\","
                "\"display_label\":\"supported name or role, otherwise empty\",\"roles\":[\"episode role\"],"
                "\"relationships\":[\"supported relationship\"],\"profile_context\":\"concise person context and cited visual age cues when applicable\","
                "\"apparent_age_range\":\"baby|toddler|child|teenager|18-29|30-44|45-59|60-74|75+ or empty\","
                "\"apparent_age_basis\":\"visual_estimate|unknown\","
                "\"evidence_quotes\":[\"exact transcript quote\"],\"document_quotes\":[\"exact printed quote\"],"
                "\"frame_indices\":[1],\"confidence\":\"high|medium|low\"}],"
                "\"user_role\":{\"role\":\"role only in this episode or empty\","
                "\"evidence_quotes\":[\"exact transcript quote\"],\"document_quotes\":[],"
                "\"frame_indices\":[1],\"confidence\":\"high|medium|low\"},"
                "\"dialogue_turns\":[{\"turn_order\":1,\"quote\":\"exact complete transcript utterance\","
                "\"speaker\":\"candidate_id|user|unknown\",\"addressee\":\"candidate_id|user|unknown\","
                "\"act\":\"greeting|question|wish|answer|response|thanks|other\","
                "\"frame_indices\":[1],\"confidence\":\"high|medium|low\"}],"
                "\"user_action\":\"one concise sentence or empty\","
                "\"user_action_evidence_quotes\":[\"exact transcript quote\"],"
                "\"facts\":[{\"subject_type\":\"person|object|place|environment|event\","
                "\"subject\":\"candidate_id, user, or concise source label\","
                "\"predicate\":\"short semantic relation/action/attribute\","
                "\"value\":\"one atomic source-grounded claim\",\"actor\":\"candidate_id|user|unknown\","
                "\"affected_people\":[\"candidate_id or user\"],\"status\":\"observed|stated|inferred|quoted_unverified_claim\","
                "\"evidence_quotes\":[\"exact transcript quote\"],"
                "\"document_quotes\":[\"exact printed quote\"],\"frame_indices\":[1],"
                "\"confidence\":\"high|medium|low\",\"time_order\":1}],"
                "\"scene_context\":\"one concise sentence or empty\","
                "\"context_evidence_quotes\":[\"exact transcript quote\"],"
                "\"location_context\":\"explicit spoken place name or empty\","
                "\"location_evidence_quotes\":[\"exact quote\"]}. "
                "Use candidate IDs exactly as supplied. For each dialogue turn, cite the whole exact utterance, "
                "and cite current frame(s) only when they show a reaction/interaction relevant to that turn. If a "
                "visible candidate is the uniquely addressed listener and later responds while visibly reacting, use "
                "the matching historical name/role only when the supplied prior record and face reference agree. "
                "Do not attribute both sides of an exchange to one person. Do not create a person observation for the unseen user. "
                "For every name/role/relationship update, provide matching quote(s) and frame index(es); never leave "
                "evidence_quotes empty for a speech-supported update. If the exact quote is not available, omit that "
                "update. A direct kinship vocative can support the speaker/listener relationship only with the speaker "
                "and visible-person mapping justified separately. For a document "
                "attribute, provide the exact printed quote and a frame linking that document to the candidate. "
                "Leave unsupported labels empty and do not invent identity. Visual scene context: "
                + json.dumps({key: str(value or "")[:500]
                              for key, value in (visual_context or {}).items()},
                             ensure_ascii=False)
                + ". Candidate people: "
                + json.dumps(candidates, ensure_ascii=False)
                + ". Ordered frame map: " + json.dumps(frame_map, ensure_ascii=False)
                + ". Existing visual facts: " + json.dumps([
                    {key: row.get(key) for key in ("actor", "subject", "predicate", "value",
                                                   "affected_people", "frame_index", "evidence")}
                    for row in (visual_facts or []) if isinstance(row, dict)
                ][:32], ensure_ascii=False)
                + ". Document evidence: " + json.dumps(documents, ensure_ascii=False)
                + ". Earlier memory profiles whose exact names occur in this transcript: "
                + json.dumps([{key: value for key, value in row.items()
                               if key not in {"reference_image_bytes", "reference_image_mime"}}
                              for row in historical_people], ensure_ascii=False)
                + ". Historical face-reference images, appended after all current clip frames, in order: "
                + json.dumps([{"reference_image": index + 1, "name": row.get("name"),
                               "source_segment_id": row.get("source_segment_id")}
                              for index, row in enumerate(historical_people)
                              if row.get("reference_image_bytes")], ensure_ascii=False)
                + ". Related memories retrieved before writing this clip (historical evidence only; do not assume identity unless the current visual/text evidence matches): "
                + json.dumps([{key: row.get(key) for key in
                               ("id", "scene", "summary", "people", "person_observations",
                                "location", "activity", "objects", "source_provenance")}
                              for row in (related_memories or []) if isinstance(row, dict)][:5],
                             ensure_ascii=False)
                + ". Time-coded transcript:\n" + speech
            )
            files = [("images", (f"profile_{index + 1}.jpg", frame["bytes"],
                                  frame.get("mime") or "image/jpeg"))
                     for index, frame in enumerate(image_frames)]
            files.extend(("images", (f"history_person_{index + 1}.jpg",
                                      row["reference_image_bytes"],
                                      row.get("reference_image_mime") or "image/jpeg"))
                         for index, row in enumerate(historical_people)
                         if row.get("reference_image_bytes"))
            request_id = uuid.uuid4().hex
            api_url = self.perceiver.next_api_url()
            self.perceiver._activate_request(request_id, api_url)
            try:
                if progress:
                    progress("person_profile_enrichment",
                             "正在把语音身份与关系线索归入人物画像")
                response = requests.post(
                    f"{api_url}/analyze_video", files=files,
                    data={"prompt": prompt, "request_id": request_id},
                    timeout=(5, 240),
                )
                if cancel_event is not None and cancel_event.is_set():
                    return fallback_rows, "", "", "", []
                if response.status_code != 200:
                    return fallback_rows, "", "", "", []
                raw = str(response.json().get("analysis") or "").strip()
                start = raw.find("{")
                end = raw.rfind("}")
                if start < 0 or end < start:
                    return fallback_rows, "", "", "", []
                payload = json.loads(raw[start:end + 1])
                if not isinstance(payload, dict) or not isinstance(payload.get("people"), list):
                    return fallback_rows, "", "", "", []
                payload["scene_context"] = _remove_visual_age_guess(
                    payload.get("scene_context"))
                payload["location_context"] = " ".join(str(payload.get("location_context") or "").split())[:180]
                payload["people"] = [
                    {**row, "profile_context": _remove_visual_age_guess(
                        row.get("profile_context"))}
                    for row in payload["people"] if isinstance(row, dict)
                ]
                payload["_profile_cache_schema"] = "candidate-aligned-v1"
                if not payload["people"]:
                    payload["_empty_result_validated"] = True
                self.person_profile_cache.put_if_absent(cache_key, payload)
            except Exception:
                return fallback_rows, "", "", "", []
            finally:
                self.perceiver._finish_request(request_id)

        if isinstance(payload, dict):
            payload = dict(payload)
            payload["scene_context"] = _remove_visual_age_guess(
                payload.get("scene_context"))
            payload["location_context"] = " ".join(str(payload.get("location_context") or "").split())[:180]
            clean_people = []
            for update_row in payload.get("people") or []:
                if not isinstance(update_row, dict):
                    continue
                clean_row = dict(update_row)
                clean_row["profile_context"] = _remove_visual_age_guess(
                    clean_row.get("profile_context"))
                clean_people.append(clean_row)
            payload["people"] = clean_people

        def normalize_quote(value):
            return "".join(str(value or "").casefold().split()).strip("，。！？、,.!?;；:：\"'“”‘’")

        transcript_lines = []
        for line in speech.splitlines():
            match = _AUDIO_EVIDENCE_LINE_RE.match(line.strip())
            if not match:
                continue
            try:
                start_s, end_s = float(match.group("start")), float(match.group("end"))
            except (TypeError, ValueError):
                continue
            transcript_lines.append((str(match.group("text") or "").strip(), start_s, end_s))
        def match_documents(quotes):
            matched = []
            for quote in quotes or []:
                normalized = normalize_quote(quote)
                if not normalized:
                    continue
                for document in documents:
                    document_text = normalize_quote(document.get("text"))
                    if normalized in document_text or document_text in normalized:
                        matched.append((str(quote).strip(), document))
                        break
            return matched

        candidate_rows = {f"person_{index + 1}": row
                          for index, row in enumerate(rows[:16])}
        for candidate_id, row in candidate_rows.items():
            row.setdefault("candidate_id", candidate_id)
            row.setdefault("visual_entity", str(row.get("entity") or "").strip())
        def descriptor_tokens(value):
            generic = {"person", "people", "woman", "man", "wearer", "visible",
                       "the", "a", "an", "in", "with", "wearing", "standing",
                       "sitting", "smiling", "candidate", "unknown"}
            tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}",
                                str(value or "").casefold())
            return {token.replace("grey", "gray") for token in tokens
                    if token not in generic and len(token) > 1}

        def match_profile_candidate(update_row):
            """Resolve legacy profile payloads to a unique visual candidate.

            Older profile cache entries contain a descriptive visual label rather
            than candidate_id. Accept only a unique frame anchor or a unique
            appearance-token match; ambiguous candidates remain unlinked.
            """
            candidate_id = str(update_row.get("candidate_id") or "").strip()
            target = candidate_rows.get(candidate_id)
            if target is not None:
                return candidate_id, target
            label = str(update_row.get("entity") or "").strip().casefold()
            exact = rows_by_label.get(label, [])
            if len(exact) == 1:
                target = exact[0]
                candidate_id = next((key for key, row in candidate_rows.items()
                                     if row is target), "")
                return candidate_id, target
            frame_indices = [value for value in update_row.get("frame_indices") or []
                             if isinstance(value, int) and not isinstance(value, bool)
                             and 1 <= value <= len(image_frames)]
            try:
                frame_value = int(update_row.get("frame") or 0)
            except (TypeError, ValueError):
                frame_value = 0
            if 1 <= frame_value <= len(image_frames):
                frame_indices.append(frame_value)
            frame_matches = [(key, row) for key, row in candidate_rows.items()
                             if isinstance(row.get("frame_index"), int)
                             and row.get("frame_index") in frame_indices]
            if len(frame_matches) == 1:
                return frame_matches[0]
            update_description = " ".join(str(update_row.get(field) or "")
                                           for field in ("entity", "profile_context"))
            label_tokens = descriptor_tokens(update_description)
            if label_tokens:
                scored = []
                for key, row in candidate_rows.items():
                    description = " ".join(str(row.get(field) or "") for field in
                                            ("entity", "appearance", "face_features", "evidence"))
                    overlap = len(label_tokens & descriptor_tokens(description))
                    if overlap:
                        scored.append((overlap, key, row))
                scored.sort(key=lambda item: (-item[0], item[1]))
                if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                    return scored[0][1], scored[0][2]
            return "", None

        valid_dialogue_turns = []
        for proposed_turn in payload.get("dialogue_turns") or []:
            if not isinstance(proposed_turn, dict):
                continue
            quote = " ".join(str(proposed_turn.get("quote") or "").split()).strip()[:400]
            normalized_quote = normalize_quote(quote)
            matched_line = next((line for line in transcript_lines
                                 if normalized_quote and
                                 (normalized_quote in normalize_quote(line[0])
                                  or normalize_quote(line[0]) in normalized_quote)), None)
            speaker = str(proposed_turn.get("speaker") or "unknown").strip()
            addressee = str(proposed_turn.get("addressee") or "unknown").strip()
            if (not matched_line
                    or speaker not in set(candidate_rows) | {"user", "unknown"}
                    or addressee not in set(candidate_rows) | {"user", "unknown"}):
                continue
            frame_indices = [value for value in proposed_turn.get("frame_indices") or []
                             if isinstance(value, int) and not isinstance(value, bool)
                             and 1 <= value <= len(image_frames)]
            if speaker in candidate_rows:
                aligned_frames = {
                    index for index, time_s in enumerate(frame_times, 1)
                    if matched_line[1] - 0.25 <= time_s <= matched_line[2] + 0.25
                }
                candidate_text = " ".join(
                    str(candidate_rows[speaker].get(key) or "")
                    for key in ("appearance", "face_features", "evidence", "profile_context")
                )
                visible_speech_cue = bool(re.search(
                    r"\b(?:speaking|talking|mouth\s+(?:(?:is|appears?\s+)?(?:moving|open(?:\s+as\s+if\s+(?:speaking|talking))?))|lip\s+movement|appears?\s+to\s+speak)\b",
                    candidate_text, re.I))
                # Confidence emitted by the model is not diarization. Require
                # source frames inside this utterance and a visual speaking cue;
                # otherwise retain the turn with speaker=unknown.
                if len(set(frame_indices) & aligned_frames) < 2 or not visible_speech_cue:
                    speaker = "unknown"
            elif speaker == "user":
                quote_text = matched_line[0]
                has_first_person = bool(re.search(
                    r"(?i)(?<![\w])(?:i|i'm|i am|me|my|mine|we|we're|we are|us|our|ours)(?![\w])"
                    r"|我(?:们)?(?:是|要|需要|的|在|刚|想)?", quote_text))
                if not has_first_person:
                    speaker = "unknown"
            confidence = str(proposed_turn.get("confidence") or "low").casefold()
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            valid_dialogue_turns.append({
                "turn_order": (proposed_turn.get("turn_order")
                               if isinstance(proposed_turn.get("turn_order"), int) else 0),
                "quote": matched_line[0], "start_s": matched_line[1], "end_s": matched_line[2],
                "speaker": speaker, "addressee": addressee,
                "act": " ".join(str(proposed_turn.get("act") or "other").split()).casefold()[:40],
                "frame_indices": sorted(set(frame_indices)), "confidence": confidence,
            })
        valid_dialogue_turns.sort(key=lambda row: (row["start_s"], row["end_s"]))

        valid_dialogue_turns.sort(key=lambda row: (row["start_s"], row["end_s"]))
        rows_by_label = {}
        for row in rows[:16]:
            label = str(row.get("entity") or "").strip().casefold()
            if label:
                rows_by_label.setdefault(label, []).append(row)
        updated_candidates = set()
        trace_updates = []
        for update_row in payload.get("people") or []:
            if not isinstance(update_row, dict):
                continue
            candidate_id, target = match_profile_candidate(update_row)
            if target is None or (candidate_id and candidate_id in updated_candidates):
                trace_updates.append({"update_entity": str(update_row.get("entity") or "")[:80],
                                      "candidate_id": candidate_id, "matched_target": None})
                continue
            if candidate_id:
                updated_candidates.add(candidate_id)
            update_row["roles"] = _clean_episode_roles(
                update_row.get("roles") or [], target.get("entity"))
            quotes = [str(value).strip()[:300] for value in update_row.get("evidence_quotes") or []
                      if isinstance(value, str) and str(value).strip()]
            matched = []
            for quote in quotes[:6]:
                normalized = normalize_quote(quote)
                if not normalized:
                    continue
                for line_text, start_s, end_s in transcript_lines:
                    line_norm = normalize_quote(line_text)
                    if normalized in line_norm or line_norm in normalized:
                        matched.append((line_text, start_s, end_s))
                        break
            document_matches = match_documents(update_row.get("document_quotes"))
            cited_frames = [value for value in update_row.get("frame_indices") or []
                            if isinstance(value, int) and not isinstance(value, bool)
                            and 1 <= value <= len(image_frames)]
            try:
                legacy_frame = int(update_row.get("frame") or 0)
            except (TypeError, ValueError):
                legacy_frame = 0
            if 1 <= legacy_frame <= len(image_frames):
                cited_frames.append(legacy_frame)
            speaker_turns = [
                turn for turn in valid_dialogue_turns
                if turn.get("speaker") == candidate_id
                and any(normalize_quote(turn.get("quote")) == normalize_quote(line[0])
                        or normalize_quote(turn.get("quote")) in normalize_quote(line[0])
                        or normalize_quote(line[0]) in normalize_quote(turn.get("quote"))
                        for line in matched)
            ]
            speaker_frames = sorted({
                frame_index for turn in speaker_turns
                for frame_index in turn.get("frame_indices") or []
                if isinstance(frame_index, int) and 1 <= frame_index <= len(image_frames)
            })
            cited_frames.extend(speaker_frames)
            target_frame = target.get("frame_index")
            document_linked_frames = {doc.get("frame_index") for _, doc in document_matches
                                      if isinstance(doc.get("frame_index"), int)}
            candidate_evidence_frames = set(cited_frames)
            if isinstance(target_frame, int) and 1 <= target_frame <= len(image_frames):
                candidate_evidence_frames.add(target_frame)
            document_matches = [
                (quote, document) for quote, document in document_matches
                if isinstance(document.get("frame_index"), int)
                and document.get("frame_index") in candidate_evidence_frames
            ]
            document_linked_frames = {doc.get("frame_index") for _, doc in document_matches}
            if not matched and not document_matches:
                continue
            confidence = str(update_row.get("confidence") or "low").casefold()
            target["profile_confidence"] = confidence if confidence in {"high", "medium", "low"} else "low"
            if confidence not in {"high", "medium"}:
                continue
            display = " ".join(str(update_row.get("display_label") or "").split()).strip(" .，。;；:")
            # Keep semantic/profile output in English; source-language text stays
            # in exact quotes and evidence references only.
            if re.search(r"[\u3400-\u9fff]", display):
                display = ""
            contextual_labels = []
            for label_field in ("roles", "relationships"):
                values = update_row.get(label_field)
                if isinstance(values, list):
                    contextual_labels.extend(str(value) for value in values if isinstance(value, str))
            if (not display and confidence in {"high", "medium"} and matched
                    and isinstance(target_frame, int) and target_frame in cited_frames):
                role_labels = list(dict.fromkeys(
                    " ".join(str(value).split()).strip()
                    for value in (update_row.get("roles") or [])
                    if isinstance(value, str) and str(value).strip()
                    and str(value).strip().casefold() not in {
                        "person", "man", "woman", "unknown", "visible person", "camera wearer", "user"}
                ))
                if len(role_labels) == 1:
                    display = role_labels[0]
            contextual_label_supported = bool(display) and any(
                normalize_quote(display) in normalize_quote(value)
                for value in contextual_labels if normalize_quote(value)
            )
            visual_linked_label = bool(
                (set(cited_frames) & document_linked_frames)
                or (isinstance(target_frame, int) and target_frame in document_linked_frames)
                or (isinstance(target_frame, int) and target_frame in cited_frames)
            )
            transcript_named_label = any(
                normalize_quote(display) in normalize_quote(item[0]) for item in matched
            ) if display else False
            document_named_label = any(
                normalize_quote(display) in normalize_quote(document.get("text"))
                for _quote, document in document_matches
            ) if display else False
            if (display and len(display) <= 80 and confidence in {"high", "medium"}
                    and (transcript_named_label or document_named_label
                         or (contextual_label_supported and visual_linked_label))):
                target.setdefault("visual_entity", str(target.get("entity") or "")[:120])
                target["entity"] = display
            age_band = " ".join(str(update_row.get("apparent_age_range") or "").casefold().split())
            age_basis = str(update_row.get("apparent_age_basis") or "").casefold()
            allowed_age_bands = {"baby", "toddler", "child", "teenager", "18-29", "30-44",
                                 "45-59", "60-74", "75+"}
            if (age_band in allowed_age_bands and age_basis == "visual_estimate"
                    and cited_frames and target_frame in candidate_evidence_frames
                    and confidence in {"medium", "high"}):
                target["apparent_age_range"] = age_band
                target["apparent_age_basis"] = "visual_estimate"
            relationships = update_row.get("relationships")
            if isinstance(relationships, list):
                clean_relations = list(dict.fromkeys(
                    " ".join(str(value).split()).strip()[:180]
                    for value in relationships if isinstance(value, str) and value.strip()
                    and not re.search(r"[\u3400-\u9fff]", value)
                ))[:8]
                if clean_relations:
                    target["relationships"] = clean_relations
            profile_context = " ".join(str(update_row.get("profile_context") or "").split()).strip()
            if re.search(r"[\u3400-\u9fff]", profile_context):
                profile_context = ""
            if profile_context and cited_frames:
                existing_context = " ".join(str(target.get("profile_context") or "").split()).strip()
                if existing_context and existing_context.casefold() not in profile_context.casefold():
                    profile_context = (profile_context + " " + existing_context).strip()
                target["profile_context"] = profile_context[:700]
            roles = update_row.get("roles")
            if isinstance(roles, list) and cited_frames:
                clean_roles = list(dict.fromkeys(
                    " ".join(str(value).split()).strip()[:80]
                    for value in roles if isinstance(value, str) and value.strip()
                    and not re.search(r"[\u3400-\u9fff]", value)
                ))[:6]
                if clean_roles:
                    target["roles"] = clean_roles
            # Keep exact document attributes on the person card only when the
            # field is present in a cited document directly linked to this face.
            for _quote, document in document_matches:
                document_frame = document.get("frame_index")
                if (not isinstance(document_frame, int)
                        or document_frame not in candidate_evidence_frames):
                    continue
                doc_text = str(document.get("text") or "")
                dob_match = re.search(
                    r"(?i)(?:date\s+of\s+birth|birth\s+date|dob)\s*[:\-]?\s*"
                    r"([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4}|"
                    r"[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4}|"
                    r"[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})", doc_text)
                if dob_match:
                    dob_text = " ".join(dob_match.group(1).split())
                    target["date_of_birth"] = dob_text[:40]
                    current_context = str(target.get("profile_context") or "").strip()
                    dob_note = f"Date of birth shown on the presented document: {dob_text}."
                    if dob_note.casefold() not in current_context.casefold():
                        target["profile_context"] = (current_context + " " + dob_note).strip()[:700]
            refs = list(target.get("evidence_refs") or [])
            for _line_text, start_s, end_s in matched:
                refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "speech", "audio_start_s": round(start_s, 3),
                             "audio_end_s": round(end_s, 3)})
            for quote, document in document_matches:
                document_frame = document.get("frame_index")
                ref = {"source_segment_id": str(source_segment_id or "")[:128],
                       "modality": "document", "document": document.get("document", "")[:100],
                       "quote": quote[:300]}
                if isinstance(document_frame, int) and 1 <= document_frame <= len(image_frames):
                    ref["frame_index"] = document_frame
                    ref["frame_time_s"] = round(float(image_frames[document_frame - 1].get("at_s") or 0), 3)
                refs.append(ref)
            for frame_index in set(cited_frames):
                refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "visual", "frame_index": frame_index,
                             "frame_time_s": round(float(image_frames[frame_index - 1].get("at_s") or 0), 3)})
            unique_refs = []
            seen_refs = set()
            for ref in refs:
                key = (ref.get("source_segment_id"), ref.get("modality"),
                       ref.get("frame_index"), ref.get("frame_time_s"),
                       ref.get("audio_start_s"), ref.get("audio_end_s"))
                if key not in seen_refs:
                    seen_refs.add(key)
                    unique_refs.append(ref)
            if unique_refs:
                target["evidence_refs"] = unique_refs[:16]
                target["evidence_ref"] = unique_refs[0]
            trace_updates.append({"update_entity": str(update_row.get("entity") or "")[:80],
                                  "candidate_id": candidate_id,
                                  "matched_target": str(target.get("entity") or "")[:80],
                                  "matched_quotes": len(matched), "cited_frames": cited_frames[:4],
                                  "display_label": display,
                                  "transcript_named": transcript_named_label,
                                  "roles": list(target.get("roles") or [])[:3]})
            # Preserve an exact spoken name/role as an auditable identity anchor
            # only after the profile update has been uniquely tied to this visible
            # candidate and the quote exists in the current audio transcript.
            if display and transcript_named_label and cited_frames:
                visual_index = (target_frame if isinstance(target_frame, int)
                                and target_frame in cited_frames else cited_frames[0])
                if 1 <= visual_index <= len(image_frames):
                    line_text, start_s, end_s = matched[0]
                    aliases = [dict(alias) for alias in target.get("spoken_aliases") or []
                               if isinstance(alias, dict)]
                    alias_row = {
                        "alias": display, "status": "probable_visual_match",
                        "confidence": confidence,
                        "basis": "exact spoken label plus unique visual-candidate alignment",
                        "evidence_ref": {
                            "source_segment_id": str(source_segment_id or "")[:128],
                            "modality": "speech", "quote": line_text[:300],
                            "audio_start_s": round(start_s, 3),
                            "audio_end_s": round(end_s, 3),
                        },
                        "visual_evidence_ref": {
                            "source_segment_id": str(source_segment_id or "")[:128],
                            "modality": "visual", "frame_index": visual_index,
                            "frame_time_s": round(float(image_frames[visual_index - 1].get("at_s") or 0), 3),
                        },
                    }
                    aliases = [alias for alias in aliases
                               if str(alias.get("alias") or "").casefold() != display.casefold()]
                    aliases.append(alias_row)
                    target["spoken_aliases"] = aliases[-8:]
        if os.environ.get("MEMORY_PROFILE_TRACE"):
            print("PROFILE_TRACE " + json.dumps({"source": str(source_segment_id or ""),
                  "cache_hit": profile_cache_hit, "frames": len(image_frames),
                  "input_people": [row.get("entity") for row in rows],
                  "updates": trace_updates}, ensure_ascii=False), flush=True)
        for role_row in payload.get("roles") or []:
            if not isinstance(role_row, dict):
                continue
            candidate_id = str(role_row.get("candidate_id") or "").strip()
            target = candidate_rows.get(candidate_id)
            role = " ".join(str(role_row.get("role") or "").split()).strip()[:80]
            quote_values = [str(value).strip() for value in role_row.get("evidence_quotes") or []
                            if isinstance(value, str) and str(value).strip()]
            quote_valid = any(
                normalize_quote(quote) in normalize_quote(line_text)
                or normalize_quote(line_text) in normalize_quote(quote)
                for quote in quote_values[:4] for line_text, _start, _end in transcript_lines
            )
            if target is None or not role or not quote_valid:
                continue
            current_roles = [str(value).strip() for value in target.get("roles") or []
                             if str(value).strip()]
            if role.casefold() not in {value.casefold() for value in current_roles}:
                current_roles.append(role)
            target["roles"] = _clean_episode_roles(current_roles, target.get("entity"))
        # Link a directly addressed prior identity only when the same visible
        # candidate is independently identified as the responding listener.
        for history in historical_people:
            name = str(history.get("name") or "").strip()
            if not name:
                continue
            addressed = [turn for turn in valid_dialogue_turns
                         if turn.get("speaker") == "user"
                         and turn.get("addressee") in candidate_rows
                         and normalize_quote(name) in normalize_quote(turn.get("quote"))]
            linked_candidates = []
            for greeting in addressed:
                candidate_id = greeting["addressee"]
                replies = [turn for turn in valid_dialogue_turns
                           if turn.get("speaker") == candidate_id
                           and turn.get("start_s", 0) >= greeting.get("start_s", 0)
                           and turn.get("act") in {"answer", "response", "thanks"}
                           and turn.get("confidence") in {"medium", "high"}
                           and turn.get("frame_indices")]
                if replies:
                    linked_candidates.append((candidate_id, greeting, replies[0]))
            unique_links = {candidate_id: (greeting, reply)
                            for candidate_id, greeting, reply in linked_candidates}
            if len(unique_links) != 1:
                continue
            candidate_id, (greeting, reply) = next(iter(unique_links.items()))
            target = candidate_rows[candidate_id]
            current_label = str(target.get("entity") or "").strip()
            if current_label and not re.fullmatch(
                    r"(?i)(?:man|woman|person|speaker|guest|employee|user)\s*\d*", current_label):
                if current_label.casefold() != name.casefold():
                    continue
            target["entity"] = name
            prior_roles = [" ".join(str(value).split()).strip()[:80]
                           for value in history.get("roles") or []
                           if isinstance(value, str) and value.strip()]
            if prior_roles:
                target["roles"] = _clean_episode_roles(prior_roles, name)
            target["profile_confidence"] = "medium"
            target["profile_context"] = (
                f"Linked to the earlier profile for {name} from "
                f"{history.get('source_segment_id')}; in this clip the user addresses this "
                "visible candidate by name and the candidate responds in the same exchange. "
                + str(history.get("profile_context") or "")
            )[:700]
            relations = list(target.get("relationships") or [])
            if "addressed by the user" not in {str(value).casefold() for value in relations}:
                relations.append("addressed by the user")
            target["relationships"] = relations[:8]
            refs = list(target.get("evidence_refs") or [])
            for turn in (greeting, reply):
                refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "speech", "quote": turn["quote"][:300],
                             "audio_start_s": round(turn["start_s"], 3),
                             "audio_end_s": round(turn["end_s"], 3)})
                for frame_index in turn.get("frame_indices") or []:
                    refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                                 "modality": "visual", "frame_index": frame_index,
                                 "frame_time_s": round(frame_times[frame_index - 1], 3)})
            prior_ref = history.get("evidence_ref")
            if isinstance(prior_ref, dict) and prior_ref.get("source_segment_id"):
                refs.append(copy.deepcopy(prior_ref))
            deduped = []
            seen_ref_keys = set()
            for ref in refs:
                key = (ref.get("source_segment_id"), ref.get("modality"),
                       ref.get("frame_index"), ref.get("audio_start_s"),
                       ref.get("audio_end_s"), ref.get("quote"))
                if key not in seen_ref_keys:
                    seen_ref_keys.add(key)
                    deduped.append(ref)
            target["evidence_refs"] = deduped[:16]
            if deduped:
                target["evidence_ref"] = deduped[0]
        context_quotes = [str(value).strip()[:300]
                          for value in payload.get("context_evidence_quotes") or []
                          if isinstance(value, str) and str(value).strip()]
        context_matches = []
        for quote in context_quotes[:6]:
            normalized = normalize_quote(quote)
            if not normalized:
                continue
            for line_text, start_s, end_s in transcript_lines:
                line_norm = normalize_quote(line_text)
                if normalized in line_norm or line_norm in normalized:
                    context_matches.append((line_text, start_s, end_s))
                    break
        scene_context = " ".join(str(payload.get("scene_context") or "").split()).strip()[:700]
        if not context_matches:
            scene_context = ""
        location_quotes = [str(value).strip()[:300]
                           for value in payload.get("location_evidence_quotes") or []
                           if isinstance(value, str) and str(value).strip()]
        location_matches = []
        for quote in location_quotes[:4]:
            normalized = normalize_quote(quote)
            if not normalized:
                continue
            for line_text, start_s, end_s in transcript_lines:
                line_norm = normalize_quote(line_text)
                if normalized in line_norm or line_norm in normalized:
                    location_matches.append((line_text, start_s, end_s))
                    break
        location_context = " ".join(str(payload.get("location_context") or "").split()).strip()[:180]
        if not location_matches:
            location_context = ""
        user_action = " ".join(str(payload.get("user_action") or "").split()).strip()[:500]
        user_action_quotes = [str(value).strip() for value in
                              payload.get("user_action_evidence_quotes") or []
                              if isinstance(value, str) and value.strip()]
        user_action_matches = []
        for quote in user_action_quotes[:8]:
            normalized = normalize_quote(quote)
            if not normalized:
                continue
            for line_text, start_s, end_s in transcript_lines:
                line_norm = normalize_quote(line_text)
                if normalized in line_norm or line_norm in normalized:
                    user_action_matches.append((line_text, start_s, end_s))
                    break
        if not user_action_matches:
            user_action = ""
            # If the model produced source-validated user dialogue turns but no
            # separate action sentence, preserve the user's contribution as a
            # compact, quote-grounded action. This works for any clear exchange
            # and avoids leaving the user side of an event blank.
            user_turns = [turn for turn in valid_dialogue_turns
                          if turn.get("speaker") == "user"
                          and turn.get("confidence") in {"medium", "high"}
                          and turn.get("act") in {"greeting", "question", "wish", "answer", "thanks"}
                          and turn.get("quote")]
            action_clauses = []
            action_quotes = []
            for turn in user_turns[:4]:
                quote = str(turn.get("quote") or "").strip()
                addressee_id = str(turn.get("addressee") or "unknown")
                addressee_row = candidate_rows.get(addressee_id)
                addressee = (str(addressee_row.get("entity") or addressee_id).strip()
                             if isinstance(addressee_row, dict) else addressee_id)
                act = turn.get("act")
                target = addressee if addressee != "unknown" else "someone"
                if act == "greeting":
                    clause = f"greeted {target}"
                elif act == "question":
                    clause = f"asked {target} a question"
                elif act == "wish":
                    clause = f"wished {target} well"
                elif act == "thanks":
                    clause = f"thanked {target}"
                else:
                    clause = f"responded to {target}"
                action_clauses.append(clause)
                action_quotes.append((quote, turn.get("start_s", 0), turn.get("end_s", 0)))
            if action_clauses:
                user_action = ("The user " + "; ".join(action_clauses) + ".")[:500]
                user_action_matches = action_quotes
        reconciled_facts = _validate_reconciled_facts(
            payload.get("facts"), rows, speech, documents, frame_times, source_segment_id,
            dialogue_turns=valid_dialogue_turns)
        # The joint model may omit meaningful transcript content even when ASR
        # succeeded. Reconcile the existing deterministic, time-coded speech
        # extractor through the same evidence validator and merge only facts
        # whose source quote is present in this clip.
        explicit_speech_rows = []
        explicit_validated = []
        try:
            explicit_speech_rows = _extract_explicit_speech_facts(
                speech, source_segment_id)
            explicit_proposed = []
            for source_fact in explicit_speech_rows:
                if not isinstance(source_fact, dict):
                    continue
                semantic_values = [
                    source_fact.get(key)
                    for key in ("subject", "predicate", "value", "actor")
                ] + list(source_fact.get("affected_people") or [])
                if any(re.search(r"[\u3400-\u9fff]", str(value or ""))
                       for value in semantic_values):
                    # The model-authored English fact may still represent this
                    # source quote. Keep the quote itself only as evidence.
                    continue
                quote_candidates = []
                source_refs = source_fact.get("evidence_refs")
                if isinstance(source_refs, list) and source_refs:
                    for evidence_ref in source_refs:
                        if not isinstance(evidence_ref, dict):
                            continue
                        try:
                            ref_start = float(evidence_ref.get("audio_start_s"))
                            ref_end = float(evidence_ref.get("audio_end_s"))
                        except (TypeError, ValueError):
                            continue
                        quote_candidates.extend(
                            text for text, start_s, end_s in transcript_lines
                            if start_s <= ref_end and end_s >= ref_start)
                if not quote_candidates:
                    evidence_ref = source_fact.get("evidence_ref")
                    if isinstance(evidence_ref, dict):
                        try:
                            ref_start = float(evidence_ref.get("audio_start_s"))
                            ref_end = float(evidence_ref.get("audio_end_s"))
                        except (TypeError, ValueError):
                            ref_start = ref_end = None
                        if ref_start is not None:
                            quote_candidates = [
                                text for text, start_s, end_s in transcript_lines
                                if start_s <= ref_end and end_s >= ref_start]
                if not quote_candidates:
                    evidence = str(source_fact.get("evidence") or "").strip()
                    quote_candidates = [text for text, _start_s, _end_s in transcript_lines
                                        if normalize_quote(evidence) == normalize_quote(text)]
                quote_candidates = list(dict.fromkeys(
                    quote for quote in quote_candidates if quote))
                if not quote_candidates:
                    continue
                explicit_proposed.append({
                    "subject_type": source_fact.get("subject_type", "event"),
                    "subject": source_fact.get("subject", "event"),
                    "predicate": source_fact.get("predicate", "spoken_event"),
                    "value": source_fact.get("value", ""),
                    "actor": source_fact.get("actor", "unknown"),
                    "affected_people": source_fact.get("affected_people") or [],
                    "status": source_fact.get("status", "stated"),
                    "confidence": source_fact.get("confidence", "medium"),
                    "evidence_quotes": quote_candidates[:4],
                    "time_order": source_fact.get("time_order"),
                })
            explicit_validated = _validate_reconciled_facts(
                explicit_proposed, rows, speech, documents, frame_times,
                source_segment_id, dialogue_turns=valid_dialogue_turns,
                max_facts=48)
        except Exception as exc:
            if progress:
                progress("speech_fact_reconciliation_error",
                         f"确定性语音事实核对失败，保留模型已验证结果（{type(exc).__name__}）")
        speech_fact_priority = {
            "self_reported_current_health_event": 0,
            "payment_requested": 0,
            "stated_birthday_event": 0,
            "invited_to_try_food": 0,
            "spoken_total": 1,
            "spoken_travel_instruction": 1,
            "quoted_product_claim": 1,
            "stated_health_claim": 1,
            "spoken_health_claim": 1,
            "self_reported_health_history": 1,
            "gave_spoken_instruction": 1,
            "instruction_then_targeted_greeting": 1,
            "addressed_person": 2,
            "requested_check_in": 2,
            "wished_good_stay": 2,
            "stated_relationship": 2,
            "stated_name": 2,
            "stated_relationship": 2,
            "stated_age": 2,
            "reported_employment_event": 2,
            "stated_experience": 2,
            "greeted": 3,
            "asked_about_wellbeing": 3,
            "thanked": 3,
            "spoken_transaction_detail": 3,
            "stated_preference": 3,
            "spoken_family_reference": 5,
        }
        def explicit_fact_order(row):
            predicate = str(row.get("predicate") or "").casefold()
            ref = row.get("evidence_ref") if isinstance(row.get("evidence_ref"), dict) else {}
            try:
                time_s = float(ref.get("audio_start_s") or 0)
            except (TypeError, ValueError):
                time_s = 0.0
            return (speech_fact_priority.get(predicate, 4), time_s)
        explicit_validated = sorted(explicit_validated, key=explicit_fact_order)[:16]
        seen_fact_keys = {
            (str(row.get("predicate") or "").casefold(),
             normalize_quote(row.get("evidence") or row.get("value") or ""))
            for row in reconciled_facts if isinstance(row, dict)
        }
        for fact in explicit_validated:
            key = (str(fact.get("predicate") or "").casefold(),
                   normalize_quote(fact.get("evidence") or fact.get("value") or ""))
            if key not in seen_fact_keys:
                reconciled_facts.append(fact)
                seen_fact_keys.add(key)

        # Keep precise speech events in Temporal Facts. The user-facing situation
        # summary stays a single high-level sentence from the quote-validated joint model.
        if speech and (not scene_context or not reconciled_facts) and image_frames:
            if cancel_event is None or not cancel_event.is_set():
                try:
                    if progress:
                        progress("speech_content_fallback",
                                 "正在核对有转写但尚未进入记忆摘要的语音内容")
                    schema = {
                        "summary": "one high-level sentence about the overall situation or purpose, using visual context and only the most meaningful supported speech; no play-by-play or transcript list",
                        "summary_quotes": ["exact quote from one transcript segment"],
                        "facts": [{"subject_type": "event|object|person|place", "subject": "source-grounded subject",
                                   "predicate": "short relation", "value": "one atomic spoken claim/event",
                                   "actor": "unknown unless directly supported", "affected_people": [],
                                   "status": "stated|quoted_unverified_claim",
                                   "evidence_quotes": ["exact quote from one transcript segment"],
                                   "confidence": "high|medium|low"}]
                    }
                    visual_context = json.dumps({
                        key: str((visual_context or {}).get(key) or "")[:500]
                        for key in ("scene", "content_summary", "activity", "interaction_details", "location", "environments")
                    }, ensure_ascii=False)
                    fallback_prompt = (
                        "Write one high-level gist of what this clip is about, using the visible setting and only speech that changes the overall situation or purpose. Do not narrate gestures or list speech facts. Keep the speaker unknown unless directly identified; "
                        "never attribute speech to a visible person from timing alone. Phrase unverified claims as "
                        "claims, not truths. Ignore only filler, unintelligible words, or unrelated background speech. "
                        "Write every generated summary, subject, predicate, and value in natural English, regardless of transcript language. "
                        "Keep original wording only in exact evidence quote fields. Copy exact evidence quotes from a single transcript segment; do not join segments. Return JSON "
                        "matching this schema: " + json.dumps(schema, ensure_ascii=False) + ". Visual context: "
                        + visual_context + ". Time-coded transcript:\\n" + speech
                    )
                    sample_indexes = sorted(set([0, len(image_frames) // 2, len(image_frames) - 1]))
                    files = [("images", (f"speech_context_{index + 1}.jpg",
                                          image_frames[index]["bytes"],
                                          image_frames[index].get("mime") or "image/jpeg"))
                             for index in sample_indexes]
                    request_id = uuid.uuid4().hex
                    api_url = self.perceiver.next_api_url()
                    self.perceiver._activate_request(request_id, api_url)
                    try:
                        response = requests.post(
                            f"{api_url}/analyze_video", files=files,
                            data={"prompt": fallback_prompt, "request_id": request_id},
                            timeout=(5, 180))
                    finally:
                        self.perceiver._finish_request(request_id)
                    if response.status_code == 200:
                        raw = str(response.json().get("analysis") or "").strip()
                        start, end = raw.find("{"), raw.rfind("}")
                        if start >= 0 and end >= start:
                            fallback = json.loads(raw[start:end + 1])
                            def quote_cited(quote):
                                norm = normalize_quote(quote)
                                return bool(norm and any(
                                    norm in normalize_quote(line_text)
                                    or normalize_quote(line_text) in norm
                                    for line_text, _start, _end in transcript_lines))
                            summary = " ".join(str(fallback.get("summary") or "").split()).strip()[:700]
                            if summary and any(quote_cited(q) for q in fallback.get("summary_quotes") or []) and not scene_context:
                                scene_context = summary
                            facts = _validate_reconciled_facts(
                                fallback.get("facts"), rows, speech, documents,
                                frame_times, source_segment_id, dialogue_turns=[])
                            keys = {(str(x.get("subject") or "").casefold(),
                                     str(x.get("predicate") or "").casefold(),
                                     str(x.get("value") or "").casefold())
                                    for x in reconciled_facts if isinstance(x, dict)}
                            for fact in facts:
                                key = (str(fact.get("subject") or "").casefold(),
                                       str(fact.get("predicate") or "").casefold(),
                                       str(fact.get("value") or "").casefold())
                                if key not in keys:
                                    reconciled_facts.append(fact)
                                    keys.add(key)
                except Exception as exc:
                    if progress:
                        progress("speech_content_fallback_error",
                                 f"语音补充失败，保留原始视觉记忆与已校验事实（{type(exc).__name__}）")
        user_role = payload.get("user_role") if isinstance(payload.get("user_role"), dict) else {}
        role = " ".join(str(user_role.get("role") or "").split()).strip()[:100]
        role_quotes = [str(value).strip() for value in user_role.get("evidence_quotes") or []
                       if isinstance(value, str) and value.strip()]
        role_matches = []
        for quote in role_quotes[:6]:
            normalized = normalize_quote(quote)
            for line_text, start_s, end_s in transcript_lines:
                if normalized and (normalized in normalize_quote(line_text)
                                   or normalize_quote(line_text) in normalized):
                    role_matches.append((line_text, start_s, end_s))
                    break
        role_frames = [value for value in user_role.get("frame_indices") or []
                       if isinstance(value, int) and not isinstance(value, bool)
                       and 1 <= value <= len(image_frames)]
        role_documents = match_documents(user_role.get("document_quotes"))
        # An inferred user role must not turn an undiarized utterance into a
        # user quote. Prefer the validated exchange/context quotes as evidence;
        # only retain the role-specific quote alone when it explicitly speaks
        # from the first person. This keeps scene-level role inference while
        # avoiding false speech attribution across clips and speakers.
        explicit_first_person = any(re.search(
            r"\b(?:i|i'm|i am|we|we're|we are|my|our|me)\b|我|我们|我的|咱们",
            text, re.IGNORECASE) for text, _start, _end in role_matches)
        if len(context_matches) >= 2:
            role_evidence_matches = context_matches
        elif explicit_first_person:
            role_evidence_matches = role_matches
        else:
            role_evidence_matches = []
        if role and role_evidence_matches and (role_frames or role_documents):
            refs = [{"source_segment_id": str(source_segment_id or "")[:128],
                     "modality": "speech", "audio_start_s": round(start_s, 3),
                     "audio_end_s": round(end_s, 3), "quote": text[:300]}
                    for text, start_s, end_s in role_evidence_matches[:4]]
            for quote, document in role_documents[:4]:
                refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "document", "document": document.get("document", "")[:100],
                             "quote": quote[:300], "frame_index": document.get("frame_index")})
            for frame_index in role_frames[:4]:
                refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "visual", "frame_index": frame_index,
                             "frame_time_s": round(frame_times[frame_index - 1], 3)})
            role_fact = {"subject_type": "event", "subject": "user",
                         "predicate": "performed_role_in_event",
                         "value": f"The user appears to be acting as {role} in this episode.",
                         "actor": "user", "affected_people": [], "status": "inferred",
                         "confidence": "medium", "modality": "multimodal",
                         "evidence": "; ".join(text for text, _start, _end in role_evidence_matches[:4]),
                         "evidence_refs": refs[:12], "evidence_ref": refs[0]}
            reconciled_facts.append(role_fact)
        # The joint model can identify several spoken roles but still collapse
        # multiple vocatives onto one person. Resolve explicit forms of address
        # against source-grounded visual participants in the same clip; an
        # ambiguous role remains an auditable low-confidence alias, not a face
        # match or permanent identity merge.
        speech_identity_facts = [{
            "subject_type": "person", "subject": "speech interaction",
            "predicate": "utterance", "value": text,
            "modality": "speech", "status": "stated", "confidence": "medium",
            "evidence": text,
            "evidence_ref": {"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "speech",
                             "audio_start_s": round(start_s, 3),
                             "audio_end_s": round(end_s, 3)},
        } for text, start_s, end_s in transcript_lines if text]
        identity_visual_facts = []
        visual_to_profile_label = {}
        for person in rows:
            visual_label = " ".join(str(person.get("visual_entity") or "").split()).strip()
            profile_label = " ".join(str(person.get("entity") or "").split()).strip()
            if (visual_label and profile_label
                    and visual_label.casefold() != profile_label.casefold()):
                visual_to_profile_label[visual_label] = profile_label
        for fact in visual_facts or []:
            if not isinstance(fact, dict):
                continue
            linked_fact = dict(fact)
            for field in ("actor", "subject", "value"):
                text = str(linked_fact.get(field) or "")
                for visual_label, profile_label in sorted(
                        visual_to_profile_label.items(), key=lambda item: -len(item[0])):
                    text = re.sub(
                        r"(?<![\w])" + re.escape(visual_label) + r"(?![\w])",
                        lambda _match, replacement=profile_label: replacement,
                        text, flags=re.I)
                linked_fact[field] = text
            identity_visual_facts.append(linked_fact)
        # A single visible participant can be linked to the speech only when the
        # source scene itself reports repeated mouth movement during the ordered
        # utterance. This enables calibrated speaker-role inference from an
        # address such as Dad/Mom without treating the vocative as the speaker.
        visual_context_fields = (visual_context if isinstance(visual_context, dict)
                                 else {"scene": str(visual_context or "")})
        scene_text = " ".join(str(visual_context_fields.get(key) or "") for key in
                               ("scene", "content_summary", "activity", "interaction_details"))
        repeated_speech_cue = bool(re.search(
            r"(?i)(?:mouth movements?.{0,100}(?:speaking|talking)|"
            r"(?:speaking|talking).{0,80}mouth movements?)", scene_text))
        if len(rows) == 1 and transcript_lines and repeated_speech_cue:
            cue = "Ordered source frames show repeated mouth movement during the utterance, suggesting this visible person is speaking."
            prior_evidence = " ".join(str(rows[0].get("evidence") or "").split())
            if cue.casefold() not in prior_evidence.casefold():
                rows[0]["evidence"] = (prior_evidence + " " + cue).strip()[:500]
        attach_spoken_identity(rows, speech_identity_facts,
                               visual_facts=identity_visual_facts or visual_facts)
        return rows, scene_context, location_context, user_action, reconciled_facts[:32]

    @staticmethod
    def _entity_key(value):
        return " ".join(str(value or "").casefold().split()).strip(" .,:;-_")

    @staticmethod
    def _entity_crop_gap(media, fields):
        """Detect missing expected entity crops so a later run can retry them."""
        media = media if isinstance(media, dict) else {}
        fields = fields if isinstance(fields, dict) else {}
        object_decisions = evaluate_object_image_policy(
            fields.get("objects") or [], fields.get("temporal_facts") or [],
            document_evidence=fields.get("document_evidence") or [],
            source_evidence=fields.get("source_evidence") or "",
            scene_context="\n".join(str(fields.get(key) or "") for key in
                                    ("scene", "content_summary", "activity", "interaction_details")),
            person_observations=fields.get("person_observations") or [])
        objects = [row["object"] for row in object_decisions if row.get("keep_image")]
        objects = objects[:12]
        people = [str(row.get("entity") or "").strip()
                  for row in (fields.get("person_observations") or [])
                  if isinstance(row, dict) and str(row.get("entity") or "").strip()][:8]
        person_visual = media.get("person_visuals") or {}
        if people and int(person_visual.get("crop_schema") or 0) < 18:
            return True
        for key, labels, label_key, role in (
            ("object_visuals", objects, "object", "object_crop"),
            ("person_visuals", people, "person", "face_crop"),
        ):
            labels = [label for label in labels if label]
            if not labels:
                continue
            visual = media.get(key)
            if not isinstance(visual, dict) or visual.get("status") in {"unavailable", "error"}:
                return True
            if key == "person_visuals" and int(visual.get("crop_schema") or 0) < 18:
                return True
            if key == "object_visuals" and int(visual.get("crop_schema") or 0) < 4:
                return True
            normalize = lambda value: " ".join(str(value or "").casefold().split())
            items = {normalize(item.get(label_key)): item
                     for item in visual.get("items") or [] if isinstance(item, dict)}
            expected_labels = {normalize(label) for label in labels}
            if any(normalize(item.get(label_key)) not in expected_labels
                   for item in visual.get("items") or [] if isinstance(item, dict)):
                # A stale crop can survive under a former clothing/background
                # alias after a person is relabeled or pruned. Treat extra cards
                # as a gallery gap so the next refresh rebuilds the exact set.
                return True
            for label in labels:
                item = items.get(normalize(label))
                if not item:
                    return True
                if item.get("status") == "not_visible":
                    continue
                if item.get("status") == "uncertain":
                    if key == "person_visuals" and int(visual.get("crop_schema") or 0) < 18:
                        return True
                    continue
                if item.get("status") in {"unavailable", "not_verified", "error"}:
                    return True
                if not any(
                    isinstance(ref, dict) and ref.get("asset")
                    and ((key == "person_visuals" and ref.get("visual_role") == "face_crop" and ref.get("identity_verified") is True and str(ref.get("identity_assignment") or "").startswith("qwen_visual_anchor+")) or (key != "person_visuals" and (ref.get("visual_role") == role or ref.get("crop_bbox") or (ref.get("visual_role") == "source_frame" and ref.get("localization_status") in {"context_only", "low_confidence_context"}))))
                    for ref in item.get("frame_refs") or []
                ):
                    return True
        return False

    def _repair_entity_crop_evidence(self, dataset_id, fields, video_path,
                                     clip_start, clip_end, media, frames,
                                     progress=None):
        """Retry failed person/object grounding for a previously written moment."""
        frames = list(frames or [])
        visual_refs = media.get("visual_frames") or (
            [media.get("visual")] if media.get("visual") else [])
        qwen_times = [float(ref.get("at_s")) for ref in visual_refs
                      if isinstance(ref, dict) and ref.get("visual_role") == "qwen_input_frame"
                      and ref.get("at_s") is not None]
        if qwen_times:
            qwen_frames = self.media_store.sample_visual_candidates(
                video_path, clip_start, clip_end, source_id=dataset_id,
                count=len(qwen_times), sample_times=qwen_times)
            frames = _sample_person_gallery_inputs(
                self.media_store, video_path, clip_start, clip_end, dataset_id,
                 qwen_frames, max_samples=36)
        elif not frames:
            frames = self.media_store.sample_visual_candidates(
                video_path, clip_start, clip_end, source_id=dataset_id, count=7)
        if not frames:
            for ref in visual_refs[:3]:
                path = Path(self.mem.memory_dir) / str(ref.get("asset") or "")
                if ref.get("asset") and path.is_file():
                    frames.append({**ref, "mime": ref.get("mime") or "image/webp",
                                   "bytes": path.read_bytes()})

        def unavailable(labels, label_key, exc):
            message = str(exc)[:500]
            return {"schema": 1, "status": "unavailable", "model": "local-crop-grounder",
                    "message": message,
                    "items": [{label_key: str(label), "status": "unavailable",
                               "appearance": "", "error": message,
                               "frame_refs": [], "crop_count": 0} for label in labels]}

        def run_grounding(kind, labels, label_key, source_key="", temporal_facts=(),
                          object_observations=()):
            highres_frames, face_detections = [], []
            face_detection_status, face_detection_error = "not_applicable", ""
            if kind == "person" and video_path and Path(video_path).is_file():
                frame_times = [float(frame.get("at_s")) for frame in frames
                               if frame.get("at_s") is not None]
                highres_frames = self.media_store.sample_visual_candidates(
                    video_path, clip_start, clip_end, source_id=dataset_id,
                    count=len(frame_times) or 7, full_resolution=True,
                    sample_times=frame_times or None)
                try:
                    face_detections = detect_faces(highres_frames) if highres_frames else []
                    face_detection_status = "completed"
                except Exception as face_exc:
                    face_detection_status = "unavailable"
                    face_detection_error = str(face_exc)[:400]
            if not labels or not frames:
                return {"schema": 1, "status": "no_visual_evidence", "items": [],
                        "message": "没有可用于定位的画面"}
            result = self.grounding_cache.get_source(source_key)[0] if source_key else None
            cache_labels = labels
            if kind == "person":
                cache_labels = [
                     "person-crop-v18-dense-face-count-residual-gallery: " + str(row.get("entity") or "").strip()
                    + " | " + str(row.get("appearance") or "").strip()
                    + " | face=" + str(row.get("face_features") or "").strip() + " | anchors=" + _person_anchor_cache_suffix(row) for row in labels]
            cache_evidence = temporal_facts
            if kind != "person":
                cache_evidence = {"temporal_facts": list(temporal_facts or []),
                                  "document_evidence": fields.get("document_evidence") or [],
                                  "object_observations": list(object_observations or [])}
            cache_key = self.grounding_cache.key(dataset_id, cache_labels, frames, cache_evidence, implementation=PERSON_GROUNDING_IMPLEMENTATION_FINGERPRINT if cache_labels is not labels else None)
            if result is None:
                result = self.grounding_cache.get(cache_key)[0]
            if result is None:
                if progress:
                    progress("person_visual_grounding" if kind == "person"
                             else "object_visual_grounding",
                             "正在重试人物局部画面" if kind == "person"
                             else "正在重试物品局部画面")
                try:
                    cancel_event = self._active_step_cancel
                    if cancel_event is not None and cancel_event.is_set():
                        return {"schema": 1, "status": "cancelled", "items": []}
                    result = (self.perceiver.ground_people(labels, frames,
                               cancel_event=cancel_event) if kind == "person" else
                              self.perceiver.ground_objects(labels, frames,
                               cancel_event=cancel_event, temporal_facts=temporal_facts,
                               source_frame_times=frame_times,
                               document_evidence=fields.get("document_evidence") or []))
                except Exception as exc:
                    names = [row.get("entity") for row in labels] if kind == "person" else labels
                    return unavailable(names, label_key, exc)
            if not isinstance(result, dict) or result.get("status") != "completed":
                message = str((result.get("message") if isinstance(result, dict) else "")
                              or "视觉模型未返回可用定位")
                names = [row.get("entity") for row in labels] if kind == "person" else labels
                failed = unavailable(names, label_key, message)
                if isinstance(result, dict):
                    failed.update({key: value for key, value in result.items()
                                   if key not in {"items", "status", "message"}})
                return failed
            result = copy.deepcopy(result)
            self.grounding_cache.put(cache_key, result)
            if source_key:
                self.grounding_cache.put_source(source_key, result)
            result_items = result.get("items") or []
            face_assignments = (_grounded_face_assignments(
                result_items, frames, highres_frames, face_detections)
                if kind == "person" and highres_frames else {})
            for item_index, item in enumerate(result_items):
                if not isinstance(item, dict):
                    continue
                boxes = {}
                box_scores = {}
                for box in item.get("frame_boxes") or []:
                    try:
                        index, bbox = int(box.get("frame_index")), box.get("bbox")
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if 1 <= index <= len(frames) and isinstance(bbox, list) and len(bbox) == 4:
                        boxes[index] = bbox
                        try:
                            box_scores[index] = float(box.get("score"))
                        except (AttributeError, TypeError, ValueError):
                            pass
                indices = list(dict.fromkeys(
                    [index for index in item.get("frame_indices") or []
                     if isinstance(index, int)] + list(boxes)))
                anchor_indices = set()
                appearance_indices = set()
                if kind == "person":
                    for raw_index in item.get("anchor_frame_indices") or []:
                        try:
                            anchor_indices.add(int(raw_index))
                        except (TypeError, ValueError):
                            continue
                    for raw_index in item.get("identity_candidate_frame_indices") or []:
                        try:
                            appearance_indices.add(int(raw_index))
                        except (TypeError, ValueError):
                            continue
                    indices = [index for index in indices
                               if index in anchor_indices or index in appearance_indices]
                refs = []
                for index in indices:
                    if index not in boxes:
                        continue
                    frame, bbox = frames[index - 1], boxes[index]
                    bbox_is_context = (
                        float(bbox[0]) <= 0.005 and float(bbox[1]) <= 0.005
                        and float(bbox[2]) >= 0.995 and float(bbox[3]) >= 0.995)
                    if kind == "object" and (
                            bbox_is_context or box_scores.get(index, float(item.get("score") or 0)) < 0.28):
                        context = self.media_store.fit_visual_candidate(frame.get("bytes"), 6_000)
                        source_ref = self.media_store._put("visual", context, "webp", 6_000) if context else None
                        if source_ref and source_ref.get("asset"):
                            refs.append({
                                "asset": source_ref["asset"],
                                "bytes": int(source_ref.get("bytes") or 0),
                                "at_s": frame.get("at_s"),
                                "source_id": frame.get("source_id") or dataset_id,
                                "visual_role": "source_frame",
                                "localization_status": "context_only" if bbox_is_context else "low_confidence_context",
                            })
                            item["localization_status"] = "context_only" if bbox_is_context else "low_confidence_context"
                        continue
                    if kind == "person":
                        sample_time = float(frame.get("at_s") or 0)
                        hi_index = min(range(len(highres_frames)),
                                       key=lambda j: abs(float(highres_frames[j].get("at_s") or 0)-sample_time)) if highres_frames else -1
                        if hi_index < 0 or abs(float(highres_frames[hi_index].get("at_s") or 0) - sample_time) > 0.35:
                            continue
                        face_row = face_assignments.get((item_index, hi_index))
                        if not face_row:
                            continue
                        source_frame = highres_frames[hi_index]
                        face_bbox = face_row.get("bbox") or []
                        face_data = self.media_store.crop_visual_original(
                            source_frame.get("bytes"), face_bbox, face=True)
                        face_ref = self.media_store._put(
                            "visual", face_data, "jpg", len(face_data or b"") + 1)
                        if face_ref and face_ref.get("asset"):
                            refs.append({
                                "asset": face_ref["asset"],
                                "bytes": int(face_ref.get("bytes") or 0),
                                "at_s": source_frame.get("at_s"),
                                "source_id": source_frame.get("source_id") or dataset_id,
                                "source_asset": source_frame.get("asset"),
                                "visual_role": "face_crop",
                                "identity_verified": True,
                                "identity_assignment": "qwen_visual_anchor+" +
                                    ("unique_appearance_clip+" if index in appearance_indices
                                     else "direct_qwen_frame+") +
                                    str(face_row.get("_identity_assignment") or "mutual_unique_face_geometry"),
                                "face_bbox": [round(float(value), 5) for value in face_bbox],
                                "detector": "YOLOv8n-Face",
                                "detector_confidence": round(float(face_row.get("confidence") or 0), 5),
                                "image_quality": "source_resolution_jpeg_q96",
                            })
                        continue
                    crop = self.media_store.crop_visual_candidate(frame.get("bytes"), bbox, 12_000)
                    crop_ref = self.media_store._put("visual", crop, "webp", 12_000)
                    if not crop_ref or not crop_ref.get("asset"):
                        continue
                    refs.append({"asset": crop_ref["asset"], "bytes": int(crop_ref.get("bytes") or 0),
                                 "at_s": frame.get("at_s"), "source_id": frame.get("source_id") or dataset_id,
                                 "visual_role": "object_crop",
                                 "crop_bbox": [round(float(value), 5) for value in bbox]})
                    if len(refs) >= 3:
                        break
                item["frame_refs"], item["crop_count"] = refs, len(refs)
                item.pop("frame_indices", None)
                item.pop("frame_boxes", None)
                if any(ref.get("visual_role") in {"object_crop", "face_crop"} for ref in refs):
                    item["status"] = "grounded"
                elif refs and kind == "object":
                    item["status"] = "visible_unlabeled"
                elif not refs and item.get("status") not in {"not_visible", "uncertain"}:
                    item["status"] = "uncertain"
                    item["error"] = "本地裁剪器未能生成局部画面；已保留来源画面"
            if kind == "person":
                gallery_candidates = []
                for item_index, item in enumerate(result_items):
                    for ref in item.get("frame_refs") or []:
                        if not isinstance(ref, dict) or not ref.get("asset"):
                            continue
                        asset_path = (self.media_store.root / str(ref.get("asset"))).resolve()
                        try:
                            asset_path.relative_to(self.media_store.root.resolve())
                            face_bytes = asset_path.read_bytes()
                        except (OSError, ValueError):
                            continue
                        at_s = float(ref.get("at_s") or 0)
                        high_index = min(range(len(highres_frames)), key=lambda j: abs(
                            float(highres_frames[j].get("at_s") or 0) - at_s)) if highres_frames else -1
                        gallery_candidates.append({
                            "item_index": item_index,
                            "frame_index": min(range(len(frames)), key=lambda j: abs(
                                float(frames[j].get("at_s") or 0) - at_s)) + 1 if frames else 0,
                            "highres_index": high_index,
                            "is_anchor": "direct_qwen_frame" in str(ref.get("identity_assignment") or ""),
                            "frame_face_count": len(face_detections[high_index])
                                if 0 <= high_index < len(face_detections) else 0,
                            "face_bytes": face_bytes,
                            "face_bbox": ref.get("face_bbox") or [],
                            "ref": ref,
                        })
                resolved = resolve_face_gallery_candidates(result_items, gallery_candidates)
                filtered_refs = [[] for _ in result_items]
                for candidate in resolved:
                    target_index = int(candidate.get("target_item_index", -1))
                    if not (0 <= target_index < len(result_items)):
                        continue
                    ref = dict(candidate["ref"])
                    if not candidate.get("is_anchor"):
                        method = str(candidate.get("identity_method") or "")
                        ref["identity_assignment"] = "qwen_visual_anchor+" + (
                            "face_similarity_clip+" if method == "face_similarity_clip"
                            else "unique_appearance_clip+") + str(
                                ref.get("identity_assignment") or "mutual_unique_face_geometry")
                        ref["identity_match_score"] = round(float(candidate.get("identity_score") or 0), 5)
                        ref["identity_match_margin"] = round(float(candidate.get("identity_margin") or 0), 5)
                        ref["identity_match_method"] = method
                    filtered_refs[target_index].append(ref)
                for item_index, item in enumerate(result_items):
                    def portrait_rank(ref):
                        bbox = ref.get("face_bbox") if isinstance(ref, dict) else None
                        area = 0.0
                        if isinstance(bbox, list) and len(bbox) == 4:
                            try:
                                area = max(0.0, float(bbox[2])-float(bbox[0])) * max(
                                    0.0, float(bbox[3])-float(bbox[1]))
                            except (TypeError, ValueError):
                                area = 0.0
                        return (float(ref.get("detector_confidence") or 0),
                                area, float(ref.get("at_s") or 0))
                    filtered_refs[item_index].sort(key=portrait_rank, reverse=True)
                    item["frame_refs"] = filtered_refs[item_index]
                    item["crop_count"] = len(filtered_refs[item_index])
                    if not filtered_refs[item_index] and item.get("status") not in {"not_visible", "uncertain"}:
                        item["status"] = "uncertain"
            result["crop_schema"] = 18 if kind == "person" else 4
            if kind == "person":
                result["face_detection_model"] = "YOLOv8n-Face"
                result["face_detection_device"] = os.environ.get("MEMORY_FACE_GPU", "cuda:0")
                result["face_detection_status"] = face_detection_status
                if face_detection_error:
                    result["face_detection_error"] = face_detection_error
            result["crop_method"] = ("Qwen identity anchors + dense GPU face detection + CLIP face-identity validation and relabeling"
                                     if kind == "person" else
                                     "event-anchored CLIP regions with high-resolution Qwen source frames and padded WebP crops")
            return result

        object_labels = [
            str(row.get("name") or row.get("object") or row.get("entity") or "").strip()
            if isinstance(row, dict) else str(row or "").strip()
            for row in fields.get("objects") or []]
        object_labels = [label for label in object_labels if label][:12]
        person_rows = [row for row in fields.get("person_observations") or []
                       if isinstance(row, dict) and str(row.get("entity") or "").strip()][:8]
        if object_labels:
            repair_frame_times = [float(ref.get("at_s")) for ref in
                                  (media.get("visual_frames") or [])
                                  if isinstance(ref, dict)
                                  and ref.get("visual_role") == "qwen_input_frame"
                                  and ref.get("at_s") is not None]
            source_key = self.grounding_cache.source_key(
                dataset_id, video_path, clip_start, clip_end, object_labels,
                evidence={"temporal_facts": fields.get("temporal_facts") or [],
                          "document_evidence": fields.get("document_evidence") or [],
                          "object_observations": fields.get("object_observations") or []},
                source_frame_times=repair_frame_times)
            media["object_visuals"] = run_grounding(
                "object", object_labels, "object", source_key,
                temporal_facts=fields.get("temporal_facts") or [],
                object_observations=fields.get("object_observations") or [],
                qwen_fallback=True)
        if person_rows:
            media["person_visuals"] = run_grounding("person", person_rows, "person")
        return media

    def _materialize_qwen_input_frames(self, result, video_path, clip_start, clip_end):
        """Return the exact inference payload, or re-decode the cached input times."""
        frame_times = result.get("frame_times_s") or []
        frame_bytes = result.get("qwen_input_frame_bytes") or []
        if len(frame_bytes) != len(frame_times) or not frame_bytes:
            if not video_path or not Path(video_path).is_file():
                return []
            frame_bytes, recovered_times = extract_frames(
                Path(video_path), clip_start, clip_end, self.perceiver.num_frames,
                return_times=True, sample_times=frame_times or None)
            frame_times = recovered_times
        return [
            {"bytes": data, "at_s": frame_times[index], "frame_index": index + 1}
            for index, data in enumerate(frame_bytes)
            if index < len(frame_times) and isinstance(data, bytes) and data
        ]

    def _has_complete_qwen_input_gallery(self, media):
        expected = max(1, int(getattr(self.perceiver, "num_frames", 6) or 6))
        refs = (media.get("visual_frames") or []) if isinstance(media, dict) else []
        actual = sum(1 for ref in refs if isinstance(ref, dict)
                     and ref.get("visual_role") == "qwen_input_frame"
                     and ref.get("asset"))
        return actual == expected

    def _refresh_existing_multimodal(self, m, moment_id, fields, source_provenance,
                                     qwen_input_frames=None, progress=None):
        """Refresh source-linked context evidence without requiring model inference.

        Existing person/object crops are retained only when their exact entity labels
        still occur in the refreshed perception. New detector boxes are never guessed.
        """
        stored = self.mem.memory.get("layer7", {}).get("moments", {}).get(moment_id)
        if not isinstance(stored, dict):
            return False
        old_media = stored.get("multimodal") or {}
        old_media_provenance = old_media.get("source_provenance") or {}
        if (old_media.get("capture_version") == CAPTURE_VERSION
                and old_media_provenance.get("perception_prompt_version") == PROMPT_VERSION
                and old_media_provenance.get("list_parser_version") == LIST_PARSER_VERSION
                and self._has_complete_qwen_input_gallery(old_media)
                and not self._entity_crop_gap(old_media, fields)):
            return False
        video_path = video_path_of(m)
        if not video_path or not Path(video_path).is_file():
            return False
        clip_start, clip_end = clip_range(m)
        try:
            media = self.media_store.capture(
                video_path, clip_start, clip_end,
                source_id=str(m.get("id") or ""), fields=fields,
                recent_media=[x.get("multimodal") for x in
                              list(self.mem.memory.get("layer7", {}).get("moments", {}).values())[-100:]],
                source_provenance=source_provenance,
                qwen_input_frames=qwen_input_frames,
            )
            if not isinstance(media, dict) or media.get("capture_status") != "captured":
                return False
            _attach_location_visual_evidence(media, fields)
            context_frames = self.media_store.sample_visual_candidates(
                video_path, clip_start, clip_end,
                source_id=str(m.get("id") or ""), count=7)
            # Scene/location/activity/event facts stay textual with provenance.
            # entity_visuals is rebuilt below from person/object crops only.
            media["entity_visuals"] = {"schema": 2, "assets": [], "items": []}
            current_objects = {
                self._entity_key(x.get("name") if isinstance(x, dict) else x)
                for x in fields.get("objects") or []
            }
            current_people = {
                self._entity_key(x.get("entity"))
                for x in fields.get("person_observations") or [] if isinstance(x, dict)
            }
            for key, allowed, label_key in (
                ("object_visuals", current_objects, "object"),
                ("person_visuals", current_people, "person"),
            ):
                old = old_media.get(key)
                if not isinstance(old, dict):
                    continue
                if key == "object_visuals" and int(old.get("crop_schema") or 0) < 4:
                    continue
                role = "object_crop" if key == "object_visuals" else "face_crop"
                kept = [dict(item) for item in old.get("items") or []
                        if isinstance(item, dict)
                        and self._entity_key(item.get(label_key)) in allowed
                        and (item.get("status") == "not_visible" or any(
                            isinstance(ref, dict) and ref.get("asset")
                            and ref.get("visual_role") == role
                            and (key != "person_visuals" or (role == "face_crop" and ref.get("identity_verified") is True and str(ref.get("identity_assignment") or "").startswith("qwen_visual_anchor+")))
                            for ref in item.get("frame_refs") or []))]
                if kept:
                    media[key] = {**old, "items": kept}
            media = self._repair_entity_crop_evidence(
                str(m.get("id") or ""), fields, video_path,
                clip_start, clip_end, media, context_frames, progress=progress)
            # Keep earlier successful crops if this retry still cannot reach the
            # vision endpoint; never replace good evidence with an empty failure.
            for key, label_key in (("object_visuals", "object"),
                                   ("person_visuals", "person")):
                previous = old_media.get(key) or {}
                current = media.get(key) or {}
                role = "object_crop" if key == "object_visuals" else "face_crop"
                legacy_person_gallery = key == "person_visuals" and int(previous.get("crop_schema") or 0) < 18
                merged = {} if legacy_person_gallery else {
                    self._entity_key(item.get(label_key)): dict(item)
                    for item in previous.get("items") or [] if isinstance(item, dict)
                    and self._entity_key(item.get(label_key)) in allowed
                    and (item.get("status") == "not_visible" or any(
                        isinstance(ref, dict) and ref.get("asset")
                        and ref.get("visual_role") == role
                        and (key != "person_visuals" or (role == "face_crop" and ref.get("identity_verified") is True and str(ref.get("identity_assignment") or "").startswith("qwen_visual_anchor+")))
                        for ref in item.get("frame_refs") or []))
                }
                for item in current.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    label = self._entity_key(item.get(label_key))
                    has_crop = any(
                        isinstance(ref, dict) and ref.get("asset")
                        and ref.get("visual_role") == role
                        and (key != "person_visuals" or (role == "face_crop" and ref.get("identity_verified") is True and str(ref.get("identity_assignment") or "").startswith("qwen_visual_anchor+")))
                        for ref in item.get("frame_refs") or [])
                    if label and (has_crop or item.get("status") == "not_visible"
                                  or label not in merged):
                        merged[label] = dict(item)
                if merged or current:
                    media[key] = {**current, "items": list(merged.values())}
            media["capture_version"] = CAPTURE_VERSION
            media["source_provenance"] = dict(source_provenance)
            stored["multimodal"] = media
            provenance = stored.setdefault("source_provenance", {})
            if isinstance(provenance, dict):
                provenance["multimodal_capture_version"] = CAPTURE_VERSION
            media["entity_visuals"] = _build_entity_visuals(stored)
            self.mem._save()
            return True
        except Exception:
            # Text-memory refresh remains usable if local video decoding or asset
            # storage fails; leave the previous multimodal payload untouched.
            return False

    def _refresh_existing_source_record(self, m, moment_id, force, update):
        """Refresh an existing record only from a current, fingerprinted cache row."""
        stored = self.mem.memory.get("layer7", {}).get("moments", {}).get(moment_id)
        if not isinstance(stored, dict):
            return None
        old = stored.get("source_provenance") or {}
        dataset_id = str(m.get("id") or "")
        video_path = video_path_of(m)
        clip_start, clip_end = clip_range(m)
        expected_video = _source_video_id(m, video_path)
        try:
            same_clip = (
                str(old.get("source_segment_id") or "") == dataset_id
                and str(old.get("source_video_id") or "") == expected_video
                and abs(float(old.get("clip_start_s")) - float(clip_start or 0)) <= 0.05
                and (clip_end is None or abs(float(old.get("clip_end_s")) - float(clip_end)) <= 0.05)
            )
        except (TypeError, ValueError):
            same_clip = False
        if not same_clip:
            return None
        profile_prompt_stale = (
            old.get("person_profile_prompt_version") != PERSON_PROFILE_CACHE_VERSION
        )
        stale_version = (
            old.get("perception_prompt_version") != PROMPT_VERSION
            or old.get("list_parser_version") != LIST_PARSER_VERSION
            or profile_prompt_stale
        )
        stored_multimodal = stored.get("multimodal") or {}
        media_provenance = stored_multimodal.get("source_provenance") or {}
        stale_multimodal = (
            stored_multimodal.get("capture_version") != CAPTURE_VERSION
            or media_provenance.get("perception_prompt_version") != PROMPT_VERSION
            or media_provenance.get("list_parser_version") != LIST_PARSER_VERSION
            or not self._has_complete_qwen_input_gallery(stored_multimodal)
            or self._entity_crop_gap(stored_multimodal, stored)
        )
        stale_summary = (
            old.get("content_summary_version") != CONTENT_SUMMARY_VERSION
            or not str(stored.get("content_summary") or "").strip()
        )
        if not force and not stale_version and not stale_multimodal and not stale_summary:
            return None
        update("cache_refresh", "正在核验缓存版本，并刷新同一视频片段的已有记忆")
        def perception_progress(event):
            update(event.get("stage", "model_inference"),
                   event.get("message", "正在读取当前片段的有效缓存"))
        result = self.perceiver.perceive(
            {"id": dataset_id, "video": m.get("video", ""),
             "start_time": m.get("start_time"), "end_time": m.get("end_time")},
            force=force, cancel_event=self._active_step_cancel,
            progress_callback=perception_progress, allow_legacy_cache=False,
            allow_stale_cache_fallback=False, allow_quality_warning=True,
        )
        fields = result.get("fields")
        fingerprint = result.get("source_input_fingerprint") or {}
        stale_cache_fallback = bool(result.get("stale_cache_fallback"))
        if (not isinstance(fields, dict) or not fingerprint.get("video_sha256")
                or not fingerprint.get("clip_sha256")
                or (not stale_cache_fallback
                    and (result.get("prompt_version") != PROMPT_VERSION
                         or result.get("list_parser_version") != LIST_PARSER_VERSION))):
            return None
        # Same-source stale visual extraction can still support local transcript
        # reconciliation and crop repair. Its stale status is preserved below.
        fields = copy.deepcopy(fields)
        stale_multimodal = (
            stale_multimodal
            or self._entity_crop_gap(stored_multimodal, fields)
        )
        frame_times = result.get("frame_times_s") or []
        def attach_refs(rows, modality):
            linked = []
            for original in rows if isinstance(rows, list) else []:
                if not isinstance(original, dict):
                    continue
                row = dict(original)
                ref = {
                    "source_segment_id": dataset_id,
                    "clip_start_s": round(float(clip_start or 0), 2),
                    "clip_end_s": round(float(clip_end), 2) if clip_end is not None else None,
                    "modality": modality,
                }
                index = row.get("frame_index")
                if isinstance(index, int) and not isinstance(index, bool):
                    ref["frame_index"] = index
                    if 1 <= index <= len(frame_times):
                        ref["frame_time_s"] = frame_times[index - 1]
                        row["frame_time_s"] = frame_times[index - 1]
                row.setdefault("modality", modality)
                row["evidence_ref"] = ref
                linked.append(row)
            return linked
        person_rows = [dict(row) for row in fields.get("person_observations") or []
                       if isinstance(row, dict)]
        for row in person_rows:
            evidence = str(row.get("evidence") or "")
            anchor_index = _person_evidence_anchor_index(row, len(frame_times))
            if anchor_index is not None:
                row["frame_index"] = anchor_index
        fields["person_observations"] = attach_refs(person_rows, "visual")
        # A newer perception cache can omit a previously grounded person even
        # though the exact same source clip still has an identity-verified face crop.
        # Preserve only observations anchored by that source's verified person crop.
        if not fields["person_observations"]:
            verified_labels = set()
            person_visuals = stored_multimodal.get("person_visuals") or {}
            for item in person_visuals.get("items") or []:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("person") or "").strip().casefold()
                refs = [ref for ref in item.get("frame_refs") or [] if isinstance(ref, dict)]
                if label and any(
                        ref.get("identity_verified") is True
                        and ref.get("visual_role") == "face_crop"
                        and str(ref.get("source_id") or dataset_id) == dataset_id
                        for ref in refs):
                    verified_labels.add(label)
            prior_rows = [
                dict(row) for row in stored.get("person_observations") or []
                if isinstance(row, dict)
                and str(row.get("entity") or "").strip().casefold() in verified_labels
            ]
            if prior_rows:
                fields["person_observations"] = attach_refs(prior_rows, "visual")
        fields["document_evidence"] = attach_refs(
            _merge_document_evidence_rows(
                fields.get("document_evidence") or [],
                (stored_multimodal.get("document_evidence") or stored.get("document_evidence") or []),
            ), "visual")
        # A refresh replaces the visual rows from the new cache read and carries over
        # the speech rows already attached to the stored moment.
        visual_facts = attach_refs(fields.get("temporal_facts") or [], "visual")
        previous_facts = stored.get("temporal_facts") or []
        previous_speech_facts = [copy.deepcopy(row) for row in previous_facts
                                 if isinstance(row, dict) and row.get("modality") == "speech"]
        speech_facts = previous_speech_facts
        fields["temporal_facts"] = visual_facts + speech_facts
        if stale_summary or stale_cache_fallback or profile_prompt_stale:
            # Recover source audio and local identity reconciliation even when the
            # optional Qwen profile call is offline. Keep any existing summary.
            summary_audio = self._transcribe_audio_evidence(
                stored_multimodal, dataset_id, video_path=video_path,
                clip_start_s=clip_start, clip_end_s=clip_end,
                progress=lambda stage, message: update(stage, message),
            )
            shared_context = ""
            if summary_audio:
                fields["source_evidence"] = summary_audio
                speech_facts = []
                for fact in previous_speech_facts:
                    ref = fact.get("evidence_ref") if isinstance(fact.get("evidence_ref"), dict) else {}
                    source_id = str(ref.get("source_segment_id") or "")
                    if source_id and source_id != dataset_id:
                        speech_facts.append(fact)
                fields["temporal_facts"] = visual_facts + speech_facts
                observations = fields.get("person_observations") or []
                observations_before_profile = [dict(row) for row in observations if isinstance(row, dict)]
                profile_frames = self._materialize_qwen_input_frames(
                    result, video_path, clip_start, clip_end)
                observations, shared_context, spoken_location, spoken_user_action, reconciled_facts = self._enrich_person_profiles_from_speech(
                    observations, summary_audio, profile_frames, dataset_id,
                    visual_facts=visual_facts,
                    document_evidence=fields.get("document_evidence") or [],
                    cancel_event=self._active_step_cancel,
                    progress=lambda stage, message: update(stage, message),
                )
                fields["temporal_facts"].extend(reconciled_facts)
                if spoken_user_action:
                    fields["user_action"] = spoken_user_action
                fields["person_observations"] = observations
                alias_map = _person_label_mapping_by_anchor(
                    observations_before_profile, observations)
                label_map = {**alias_map, **_person_name_ordinal_aliases(observations)}
                _remap_person_fact_references_by_anchor(
                    fields, observations_before_profile, observations)
                _remap_person_labels_in_fields(fields, label_map)
                _remap_person_visual_labels_by_anchor(stored_multimodal, observations)
                stored_multimodal = _remap_person_labels_in_media(stored_multimodal, label_map)
                stored["multimodal"] = stored_multimodal
                if spoken_location:
                    fields["location"] = spoken_location
            prior_summary = str(stored.get("content_summary") or "").strip()
            if not prior_summary:
                fields["content_summary"] = _content_summary_for_fields(
                    fields, speech_context=shared_context, prior=stored)
            elif stale_cache_fallback:
                fields["content_summary"] = prior_summary
            else:
                fields["content_summary"] = _content_summary_for_fields(
                    fields, speech_context=shared_context, prior=stored)
            if not fields.get("interaction_details"):
                fields["interaction_details"] = str(stored.get("interaction_details") or "")[:700]
            if fields.get("document_evidence"):
                stored_multimodal["document_evidence"] = copy.deepcopy(fields["document_evidence"])
            if fields.get("document_evidence"):
                stored["multimodal"] = stored_multimodal
        normalize_object_transition_facts(fields)
        normalize_user_facing_memory_copy(fields)
        source_day = next((part.upper() for part in dataset_id.split("_")
                           if part.upper().startswith("DAY") and part[3:].isdigit()), "")
        source_time = {
            "day_label": source_day,
            "time_of_day": _clock_from_id(dataset_id),
            "time_occasion": str(fields.get("time_occasion") or "")[:200],
            "clip_start_s": round(float(clip_start or 0), 2),
            "basis": "dataset_relative" if source_day else "video_relative",
            "source_segment_id": dataset_id,
        }
        if clip_end is not None:
            source_time["clip_end_s"] = round(float(clip_end), 2)
        source_provenance = {
            "source_video_id": expected_video,
            "source_segment_id": dataset_id,
            "clip_start_s": round(float(clip_start or 0), 2),
            "clip_end_s": round(float(clip_end), 2) if clip_end is not None else None,
            "source_video_sha256": fingerprint["video_sha256"],
            "source_clip_sha256": fingerprint["clip_sha256"],
            "perception_prompt_version": result.get("prompt_version") or old.get("perception_prompt_version"),
            "list_parser_version": result.get("list_parser_version") or old.get("list_parser_version"),
            "multimodal_capture_version": CAPTURE_VERSION,
            "extraction_batch_id": old.get("extraction_batch_id") or self.extraction_batch_id,
            "content_summary_version": (old.get("content_summary_version")
                                         if stale_cache_fallback and stored.get("content_summary")
                                         else CONTENT_SUMMARY_VERSION),
            "person_profile_prompt_version": PERSON_PROFILE_CACHE_VERSION,
        }
        if stale_cache_fallback:
            source_provenance["perception_status"] = "stale_cache_fallback"
            source_provenance["perception_refresh_error"] = str(
                result.get("stale_cache_reason") or old.get("perception_refresh_error") or "")[:500]
            source_provenance["local_evidence_refresh"] = "speech_identity_and_multimodal"
        updates = {
            "scene": fields.get("scene") or "",
            "user_action": fields.get("user_action") or "",
            "interaction_details": fields.get("interaction_details") or stored.get("interaction_details") or "",
            "content_summary": fields.get("content_summary") or stored.get("content_summary") or "",
            "person_observations": fields.get("person_observations") or [],
            "temporal_facts": fields.get("temporal_facts") or [],
            "source_evidence": fields.get("source_evidence") or stored.get("source_evidence") or "",
            "source_time": source_time,
            "location": fields.get("location") or "",
            "activity": fields.get("activity") or "",
            "environments": fields.get("environments") or [],
            "objects": fields.get("objects") or [],
        }
        observed_labels = [
            str(row.get("entity") or "").strip()
            for row in fields.get("person_observations") or []
            if isinstance(row, dict) and str(row.get("entity") or "").strip()
        ]
        source_people = [str(value or "").strip()
                         for value in fields.get("people") or []
                         if str(value or "").strip()]
        # Keep the moment index synchronized with the refreshed participant set.
        # Do not carry old anonymous clothing/background aliases forward when the
        # source-grounded pass has replaced them with stable semantic labels.
        if observed_labels or source_people:
            updates["people"] = list(dict.fromkeys(source_people + observed_labels))
        if fields.get("time_occasion"):
            updates["notes"] = str(fields["time_occasion"])[:200]
        old_identity_rows = stored.get("person_observations") or []
        old_identity_labels = sorted(str(row.get("entity") or "").strip().casefold()
                                     for row in old_identity_rows if isinstance(row, dict))
        new_identity_labels = sorted(str(row.get("entity") or "").strip().casefold()
                                     for row in updates.get("person_observations") or []
                                     if isinstance(row, dict))
        identity_refreshed = old_identity_labels != new_identity_labels
        refreshed = self.mem.refresh_source_moment(
            moment_id, updates, source_provenance, expected_segment_id=dataset_id)
        multimodal_refreshed = False
        if stale_multimodal:
            qwen_input_frames = self._materialize_qwen_input_frames(
                result, video_path, clip_start, clip_end)
            multimodal_refreshed = self._refresh_existing_multimodal(
                m, moment_id, fields, source_provenance,
                qwen_input_frames=qwen_input_frames,
                progress=lambda stage, message: update(stage, message))
        if fields.get("document_evidence"):
            current = self.mem.memory.get("layer7", {}).get("moments", {}).get(moment_id)
            current_media = current.get("multimodal") if isinstance(current, dict) else None
            if isinstance(current_media, dict):
                current_media["document_evidence"] = copy.deepcopy(fields["document_evidence"])
                self.mem._save()
        result["multimodal_refreshed"] = multimodal_refreshed
        result["identity_refreshed"] = identity_refreshed
        result["local_evidence_refreshed"] = bool(
            stale_cache_fallback and (refreshed or multimodal_refreshed))
        if not refreshed and not multimodal_refreshed:
            return None
        if result.get("source") == "cache_quality_warning":
            target = self.mem.memory.get("layer7", {}).get("moments", {}).get(moment_id)
            if isinstance(target, dict):
                provenance = target.setdefault("source_provenance", {})
                provenance["perception_status"] = "cache_quality_warning"
                provenance["perception_quality_warning"] = str(
                    result.get("quality_warning") or "")[:700]
                self.mem._save()
        fields["source_provenance"] = source_provenance
        self._qwen_fields[dataset_id] = fields
        return result, fields

    def _add_one(self, m, force=False, progress=None):
        """用 Qwen 的感知产出写入一条记忆，返回 (moment_id, info)。"""
        dataset_id = m.get("id", "")
        item_started = time.perf_counter()
        operation_ms = {}
        progress_stage_ms = {}
        active_stage = "prepare"
        active_stage_started = item_started

        def record_duration(name, started):
            operation_ms[name] = round((time.perf_counter() - started) * 1000)

        def close_progress_stage(now=None):
            nonlocal active_stage_started
            if active_stage and active_stage_started is not None:
                end = time.perf_counter() if now is None else now
                elapsed = max(0, round((end - active_stage_started) * 1000))
                progress_stage_ms[active_stage] = progress_stage_ms.get(active_stage, 0) + elapsed
            active_stage_started = now

        def update(stage, message, output=None):
            nonlocal active_stage, active_stage_started
            now = time.perf_counter()
            if str(stage) != active_stage:
                close_progress_stage(now)
                active_stage = str(stage)
                active_stage_started = now
            if progress:
                self._set_step_progress(**progress, stage=stage, message=message,
                                        output=output)

        def finish_timing(info):
            close_progress_stage()
            operation_ms["total_wall_ms"] = round((time.perf_counter() - item_started) * 1000)
            info["stage_timings_ms"] = {
                **operation_ms,
                "progress_stage_ms": dict(progress_stage_ms),
            }
            try:
                cache_sources = {"cache", "cache_quality_warning", "stale_cache_fallback",
                                 "legacy_cache", "label_normalization_only"}
                record = {
                    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_segment_id": str(dataset_id),
                    "perception_source": info.get("source"),
                    "perception_model": "Qwen3-VL-8B-Instruct",
                    "perception_model_called": info.get("source") not in cache_sources,
                    "perception_elapsed_ms": info.get("elapsed_ms"),
                    "perception_elapsed_kind": (
                        "historical_cached_inference" if info.get("source") in cache_sources
                        else "current_model_inference"
                    ),
                    "stage_timings_ms": info["stage_timings_ms"],
                    "retrieval_cache": ((info.get("retrieval_meta") or {}).get("cache") or {}).get("status"),
                    "retrieval_before_write": {
                        "status": ((info.get("retrieval_before_write_meta") or {}).get("status")),
                        "method": ((info.get("retrieval_before_write_meta") or {}).get("retrieval_method")),
                        "used_for_current_memory": bool(info.get("retrieved_memories_before_write")),
                        "candidate_count": len(info.get("retrieved_memories_before_write") or []),
                        "candidates": [{
                            "moment_id": str(row.get("id") or ""),
                            "source_segment_id": str(((row.get("source_provenance") or {}).get("source_segment_id")) or ""),
                            "score": row.get("score"),
                            "scene": str(row.get("scene") or "")[:300],
                            "people": row.get("people") or [],
                            "objects": row.get("objects") or [],
                        } for row in (info.get("retrieved_memories_before_write") or [])[:5]
                          if isinstance(row, dict)],
                    },
                    "retrieval_stage_timings_ms": ((info.get("retrieval_meta") or {}).get("stage_timings_ms") or {}),
                    "retrieval_wrapper_stage_timings_ms": ((info.get("retrieval_meta") or {}).get("wrapper_stage_timings_ms") or {}),
                    "retrieval_wrapper_elapsed_ms": ((info.get("retrieval_meta") or {}).get("wrapper_elapsed_ms")),
                }
                path = Path(self.mem.memory_dir) / "processing_latency_log.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except (OSError, TypeError, ValueError):
                pass
            return info
        cancel_event = self._active_step_cancel
        if cancel_event is not None and cancel_event.is_set():
            info = {"dataset_id": dataset_id, "source": "cancelled",
                    "cancelled": True, "error": "当前片段处理已终止"}
            update("cancelled", info["error"])
            finish_timing(info)
            return None, info
        existing_mid = self.id_map.get(dataset_id)
        if existing_mid:
            cached = self.perceiver.cache.get(dataset_id) or {}
            fields = cached.get("fields") or self._qwen_fields.get(dataset_id) or {}
            refresh = self._refresh_existing_source_record(
                m, existing_mid, force, update)
            refreshed_existing = refresh is not None
            identity_refreshed = False
            if refreshed_existing:
                refresh_result, fields = refresh
                cached = refresh_result
                identity_refreshed = bool(refresh_result.get("identity_refreshed"))
                if (refresh_result.get("stale_cache_fallback")
                        and not refresh_result.get("local_evidence_refreshed")):
                    refreshed_existing = False
            info = {
                "dataset_id": dataset_id,
                "moment_id": existing_mid,
                "qwen": fields,
                "raw": cached.get("raw", ""),
                "elapsed_ms": cached.get("elapsed_ms"),
                "num_frames": cached.get("num_frames"),
                "source": cached.get("source", "cache"),
                "memory_refreshed": refreshed_existing,
                "stale_cache_fallback": bool(cached.get("stale_cache_fallback")),
                "quality_warning": cached.get("quality_warning", ""),
                "multimodal_refreshed": bool(cached.get("multimodal_refreshed")),
                "identity_refreshed": identity_refreshed,
                "eval": golden_eval(fields, m.get("ground_truth", {})),
            }
            if refreshed_existing and cached.get("multimodal_refreshed"):
                message = "已更新同一片段的记忆与多模态证据，没有重复新增"
            elif refreshed_existing:
                message = "已用当前缓存更新同一片段的记忆，没有重复新增"
            elif identity_refreshed:
                message = "已根据已存语音与画面证据补齐人物身份，没有重复新增"
            else:
                message = "这条记忆已在当前会话中，复用现有记录"
            retrieval_fields = dict(fields)
            retrieval_fields["source_evidence"] = str(
                retrieval_fields.get("source_evidence") or "")
            retrieval_started = time.perf_counter()
            retrieved, retrieval_meta = self._retrieve_before_write(
                dataset_id, retrieval_fields, dataset_id=dataset_id, progress=progress)
            record_duration("history_retrieval_before_write_wall_ms", retrieval_started)
            info["retrieved_memories_before_write"] = retrieved
            info["retrieval_before_write_meta"] = retrieval_meta
            info["retrieved_memories"] = []
            info["retrieval_meta"] = {"status": "reused_prewrite", "duration_ms":
                retrieval_meta.get("wrapper_elapsed_ms", 0), "used_for_current_memory": bool(retrieved)}
            retrieval_message = ("片段处理完成；写入前索引检索到 %d 条相关历史记忆" % len(retrieved)
                                 if retrieved else "片段处理完成；写入前未检索到相关历史记忆")
            update("done", retrieval_message, self._progress_text(m, info))
            self._set_step_progress(completed_item=self._progress_text(m, info))
            finish_timing(info)
            return existing_mid, info
        cancel_event = self._active_step_cancel
        update("prepare", "正在准备视频片段与记忆写入")
        def on_perception(event):
            update(event.get("stage", "model_inference"),
                   event.get("message", "正在处理当前片段"))
        # Strict allowlist: benchmark captions, transcripts and answer labels
        # never enter the perception/cache/model call.
        perception_input = {
            "id": m.get("id", ""),
            "video": m.get("video", ""),
            "start_time": m.get("start_time"),
            "end_time": m.get("end_time"),
        }
        perception_started = time.perf_counter()
        r = self.perceiver.perceive(perception_input, force=force, cancel_event=cancel_event,
                                    progress_callback=on_perception,
                                    allow_legacy_cache=False,
                                    allow_stale_cache_fallback=False,
                                    allow_quality_warning=True)
        record_duration("perception_call_wall_ms", perception_started)
        operation_ms["perception_reported_elapsed_ms"] = int(r.get("elapsed_ms") or 0)
        operation_ms["perception_reported_elapsed_kind"] = (
            "historical_cached_inference"
            if r.get("source") in {"cache", "cache_quality_warning", "stale_cache_fallback",
                                   "legacy_cache", "label_normalization_only"}
            else "current_model_inference"
        )
        if cancel_event is not None and cancel_event.is_set():
            info = {"dataset_id": dataset_id, "source": "cancelled",
                    "cancelled": True, "error": "当前片段处理已终止"}
            update("cancelled", info["error"])
            finish_timing(info)
            return None, info
        if not r.get("fields"):
            info = {"error": r.get("error", "感知失败"),
                          "source": r.get("source", "error"),
                          "cancelled": bool(r.get("cancelled"))}
            update("cancelled" if info.get("cancelled") else "error",
                   info.get("error") or "感知失败", self._progress_text(m, info))
            self._set_step_progress(completed_item=self._progress_text(m, info))
            finish_timing(info)
            return None, info
        f = _retain_episode_objects(dict(r["fields"]))
        update("write_memory", "感知结果已生成，正在写入记忆", self._progress_text(m, {"qwen": f}))
        multimodal = None
        media_error = None
        video_path = video_path_of(m)
        clip_start, clip_end = clip_range(m)
        source_provenance = {
            "source_video_id": _source_video_id(m, video_path),
            "source_segment_id": dataset_id,
            "clip_start_s": round(float(clip_start or 0), 2),
            "clip_end_s": round(float(clip_end), 2) if clip_end is not None else None,
            "extraction_batch_id": self.extraction_batch_id,
            "perception_prompt_version": r.get("prompt_version") or PROMPT_VERSION,
            "list_parser_version": r.get("list_parser_version") or LIST_PARSER_VERSION,
            "multimodal_capture_version": CAPTURE_VERSION,
            "content_summary_version": CONTENT_SUMMARY_VERSION,
            "person_profile_prompt_version": PERSON_PROFILE_CACHE_VERSION,
        }
        if r.get("stale_cache_fallback"):
            source_provenance["perception_status"] = "stale_cache_fallback"
            source_provenance["perception_refresh_error"] = str(
                r.get("stale_cache_reason") or "")[:500]
        elif r.get("source") == "cache_quality_warning":
            source_provenance["perception_status"] = "cache_quality_warning"
            source_provenance["perception_quality_warning"] = str(
                r.get("quality_warning") or "")[:700]
        input_fingerprint = r.get("source_input_fingerprint") or {}
        if input_fingerprint.get("video_sha256"):
            source_provenance["source_video_sha256"] = input_fingerprint["video_sha256"]
            source_provenance["source_clip_sha256"] = input_fingerprint.get("clip_sha256", "")
        source_day = next((part.upper() for part in str(dataset_id).split("_")
                           if part.upper().startswith("DAY")
                           and part[3:].isdigit()), "")
        source_time = {
            "day_label": source_day,
            "time_of_day": _clock_from_id(dataset_id),
            "time_occasion": str(f.get("time_occasion") or "")[:200],
            "clip_start_s": round(float(clip_start or 0), 2),
            "basis": "dataset_relative" if source_day else "video_relative",
            "source_segment_id": dataset_id,
        }
        if clip_end is not None:
            source_time["clip_end_s"] = round(float(clip_end), 2)
        frame_times = r.get("frame_times_s") or []

        def attach_evidence_refs(rows, default_modality):
            linked = []
            for original in rows if isinstance(rows, list) else []:
                if not isinstance(original, dict):
                    continue
                row = dict(original)
                ref = {
                    "source_segment_id": source_provenance["source_segment_id"],
                    "clip_start_s": source_provenance["clip_start_s"],
                    "clip_end_s": source_provenance.get("clip_end_s"),
                    "modality": str(row.get("modality") or default_modality)[:16],
                }
                frame_index = row.get("frame_index")
                if isinstance(frame_index, int) and not isinstance(frame_index, bool):
                    ref["frame_index"] = frame_index
                    if 1 <= frame_index <= len(frame_times):
                        ref["frame_time_s"] = frame_times[frame_index - 1]
                        row["frame_time_s"] = frame_times[frame_index - 1]
                row["evidence_ref"] = ref
                scope = str(row.get("time_scope") or "")
                if (default_modality == "visual"
                        and str(row.get("predicate") or "").casefold() in {
                            "move", "moved", "relocate", "relocated", "transfer", "transferred"
                        }):
                    span = re.search(r"frames?\s+(\d+)\s*(?:to|[-–])\s*(\d+)", scope, re.I)
                    if span:
                        start_index, end_index = int(span.group(1)), int(span.group(2))
                        transition_refs = []
                        for index in dict.fromkeys((start_index, end_index)):
                            if not (1 <= index <= len(frame_times)):
                                continue
                            transition_refs.append({
                                "source_segment_id": source_provenance["source_segment_id"],
                                "clip_start_s": source_provenance["clip_start_s"],
                                "clip_end_s": source_provenance.get("clip_end_s"),
                                "modality": "visual",
                                "frame_index": index,
                                "frame_time_s": frame_times[index - 1],
                            })
                        if len(transition_refs) > 1:
                            row["evidence_refs"] = transition_refs
                linked.append(row)
            return linked

        raw_person_observations = []
        for original in f.get("person_observations") or []:
            if not isinstance(original, dict):
                continue
            row = dict(original)
            evidence = str(row.get("evidence") or "")
            anchor_index = _person_evidence_anchor_index(row, len(frame_times))
            if anchor_index is not None:
                row["frame_index"] = anchor_index
            raw_person_observations.append(row)
        person_observations = attach_evidence_refs(
            raw_person_observations, "visual")
        temporal_facts = attach_evidence_refs(
            f.get("temporal_facts") or [], "visual")
        object_observations = attach_evidence_refs(
            f.get("object_observations") or [], "visual")
        f["document_evidence"] = attach_evidence_refs(
            f.get("document_evidence") or [], "visual")
        f["person_observations"] = person_observations
        f["temporal_facts"] = temporal_facts
        f["object_observations"] = object_observations
        shared_candidate_frames = None
        early_source_audio_text = ""
        audio_transcription_attempted = False
        if video_path and os.environ.get("MEMORY_CAPTURE_MULTIMODAL", "1") != "0":
            try:
                if self.media_store.size_bytes() + 48 * 1024 * 1024 > self.media_store.asset_budget:
                    self.mem.compact_multimodal_assets(
                        max(0, self.media_store.asset_budget - 48 * 1024 * 1024))
                    self.media_store._bytes_cached = None
                media_capture_started = time.perf_counter()
                qwen_input_frames = self._materialize_qwen_input_frames(
                    r, video_path, clip_start, clip_end)
                multimodal = self.media_store.capture(
                    video_path, clip_start, clip_end,
                    source_id=dataset_id, fields=f,
                    recent_media=[x.get("multimodal") for x in
                                  list(self.mem.memory["layer7"]["moments"].values())[-100:]],
                    source_provenance=source_provenance,
                    qwen_input_frames=qwen_input_frames,
                )
                record_duration("video_decode_and_multimodal_capture_ms", media_capture_started)
                _attach_location_visual_evidence(multimodal, f)
                # Transcribe before deciding which objects merit retained images:
                # ownership, purchase, gift, and presentation language can change
                # image value even when the visual object list is unchanged.
                audio_started = time.perf_counter()
                early_source_audio_text = self._transcribe_audio_evidence(
                    multimodal, dataset_id, video_path=video_path,
                    clip_start_s=clip_start, clip_end_s=clip_end,
                    progress=lambda stage, message: update(stage, message),
                    timing_sink=operation_ms,
                )
                audio_transcription_attempted = True
                record_duration("audio_decode_cache_and_transcription_wall_ms", audio_started)
                if early_source_audio_text:
                    f["source_evidence"] = early_source_audio_text
                # Audio can establish an explicit keep/track request for an item.
                # Reapply the future-value filter before indexing or cropping it.
                f = _retain_episode_objects(f)
                kept_objects, person_object_labels = filter_person_entities_from_objects(
                    f.get("objects") or [], f.get("person_observations") or [])
                if person_object_labels:
                    f["objects"] = kept_objects
                    excluded_person_objects = {
                        str(value.get("object") if isinstance(value, dict) else value).strip().casefold()
                        for value in person_object_labels
                    }
                    f["object_observations"] = [
                        row for row in f.get("object_observations") or []
                        if not isinstance(row, dict)
                        or str(row.get("object") or row.get("entity") or "").strip().casefold()
                        not in excluded_person_objects
                    ]
                object_decisions = evaluate_object_image_policy(
                    f.get("objects") or [], f.get("temporal_facts") or [],
                    document_evidence=f.get("document_evidence") or [],
                    source_evidence=early_source_audio_text or f.get("source_evidence") or "",
                    scene_context="\n".join(str(f.get(key) or "") for key in
                                            ("scene", "content_summary", "activity", "interaction_details")),
                    person_observations=f.get("person_observations") or [])
                object_labels = [row["object"] for row in object_decisions
                                 if row.get("keep_image")]
                skipped_object_crops = [
                    {"object": row["object"], "reason_codes": row.get("reason_codes") or [],
                     "keep_fact": bool(row.get("keep_fact")), "keep_image": False,
                     "confidence": row.get("confidence", "low"),
                     "source_refs": row.get("source_refs") or []}
                    for row in object_decisions if not row.get("keep_image")]
                multimodal["object_crop_policy"] = {
                    "version": "evidence-value-object-image-v9",
                    "decisions": object_decisions,
                    "eligible": list(object_labels),
                    "skipped": skipped_object_crops,
                }
                visual_refs = multimodal.get("visual_frames") or (
                    [multimodal.get("visual")] if multimodal.get("visual") else [])
                if object_labels and visual_refs:
                    object_pipeline_started = time.perf_counter()
                    update("object_visual_grounding_lookup", "正在检查物品与画面的持久缓存")
                    object_source_times = [float(ref.get("at_s")) for ref in visual_refs
                                           if isinstance(ref, dict)
                                           and ref.get("visual_role") == "qwen_input_frame"
                                           and ref.get("at_s") is not None]
                    source_grounding_key = self.grounding_cache.source_key(
                        dataset_id, video_path, clip_start, clip_end, object_labels,
                        evidence={"temporal_facts": f.get("temporal_facts") or [],
                                  "document_evidence": f.get("document_evidence") or [],
                                  "object_observations": f.get("object_observations") or []},
                        source_frame_times=object_source_times
                    )
                    object_source_lookup_started = time.perf_counter()
                    try:
                        grounding, grounding_lookup_ms = self.grounding_cache.get_source(
                            source_grounding_key
                        )
                    except Exception:
                        grounding, grounding_lookup_ms = None, 0
                    record_duration("object_source_cache_lookup_wall_ms", object_source_lookup_started)
                    # A source-cache hit skips model inference, not frame decoding: cached
                    # bounding boxes still need the matching pixels to materialize object crops.
                    object_frame_sample_started = time.perf_counter()
                    candidate_frames = self.media_store.sample_visual_candidates(
                        video_path, clip_start, clip_end, source_id=dataset_id, count=7)
                    qwen_times = [float(ref.get("at_s")) for ref in visual_refs
                                  if isinstance(ref, dict)
                                  and ref.get("visual_role") == "qwen_input_frame"
                                  and ref.get("at_s") is not None]
                    if qwen_times:
                        precise_frames = self.media_store.sample_visual_candidates(
                            video_path, clip_start, clip_end, source_id=dataset_id,
                            count=len(qwen_times), sample_times=qwen_times,
                            full_resolution=True)
                        by_time = {
                            round(float(frame.get("at_s") or 0), 3): frame
                            for frame in (candidate_frames or []) if isinstance(frame, dict)
                        }
                        for frame in precise_frames or []:
                            if isinstance(frame, dict):
                                by_time[round(float(frame.get("at_s") or 0), 3)] = frame
                        candidate_frames = sorted(by_time.values(),
                                                  key=lambda frame: float(frame.get("at_s") or 0))
                    record_duration("object_candidate_frame_sampling_ms", object_frame_sample_started)
                    shared_candidate_frames = candidate_frames
                    grounding_key = None
                    if grounding is None:
                        frame_inputs = candidate_frames
                        if not frame_inputs:
                            for ref in visual_refs[:3]:
                                asset_path = Path(self.mem.memory_dir) / str(ref.get("asset") or "")
                                if ref.get("asset") and asset_path.is_file():
                                    frame_inputs.append({
                                        **ref, "mime": ref.get("mime") or "image/webp",
                                        "bytes": asset_path.read_bytes(),
                                    })
                        object_detail_lookup_started = time.perf_counter()
                        try:
                            grounding_key = self.grounding_cache.key(
                                dataset_id, object_labels, frame_inputs,
                                {"temporal_facts": f.get("temporal_facts") or [],
                                 "document_evidence": f.get("document_evidence") or [],
                                 "object_observations": f.get("object_observations") or []}
                            )
                            grounding, grounding_lookup_ms = self.grounding_cache.get(grounding_key)
                        except Exception:
                            grounding_key, grounding, grounding_lookup_ms = None, None, 0
                        record_duration("object_grounding_cache_lookup_wall_ms", object_detail_lookup_started)
                        if grounding is not None and source_grounding_key:
                            self.grounding_cache.put_source(source_grounding_key, grounding)
                    operation_ms["object_grounding_cache_hit"] = grounding is not None
                    if grounding is not None:
                        update("object_grounding_cache_hit",
                               f"物品画面对照命中缓存（{grounding_lookup_ms} ms），跳过本地 CLIP 匹配")
                    else:
                        update("object_visual_grounding",
                               "缓存未命中，正在逐件核对物品与关键帧")
                        try:
                            object_model_started = time.perf_counter()
                            grounding = self.perceiver.ground_objects(
                                object_labels, frame_inputs, cancel_event=cancel_event,
                                temporal_facts=f.get("temporal_facts") or [],
                                source_frame_times=frame_times,
                                document_evidence=f.get("document_evidence") or [],
                                object_observations=f.get("object_observations") or [],
                                qwen_fallback=True,
                                progress_callback=lambda event: update(
                                    event.get("stage", "object_visual_grounding"),
                                    event.get("message", "正在核对物品画面"),
                                    self._progress_text(m, {"qwen": f}),
                                ),
                            )
                            record_duration("object_grounding_model_call_ms", object_model_started)
                        except Exception as exc:
                            grounding_error = str(exc)[:500]
                            grounding = {
                                "schema": 1, "status": "unavailable", "model": "opencv-hog+clip-grid-v1",
                                "message": grounding_error,
                                "items": [{"object": str(label), "status": "unavailable",
                                           "error": grounding_error, "frame_refs": [],
                                           "crop_count": 0, "frame_indices": []}
                                          for label in object_labels[:12]],
                            }
                        self.grounding_cache.put(grounding_key, grounding)
                        if source_grounding_key:
                            self.grounding_cache.put_source(source_grounding_key, grounding)

                    object_crop_started = time.perf_counter()
                    update("object_crop_materialization",
                           "正在抽取关键帧并生成物品局部画面附件")
                    saved_by_index = {}
                    if candidate_frames and grounding.get("status") == "completed":
                        coverage = {}
                        for item in grounding.get("items") or []:
                            if item.get("status") in {"grounded", "visible_unlabeled"}:
                                for index in item.get("frame_indices") or []:
                                    if 1 <= index <= len(candidate_frames):
                                        coverage.setdefault(index, set()).add(item.get("object"))
                        selected = []
                        uncovered = {
                            item.get("object") for item in (grounding.get("items") or [])
                            if item.get("status") in {"grounded", "visible_unlabeled"}
                        }
                        while uncovered and len(selected) < 3:
                            best = max(
                                (index for index in coverage if index not in selected),
                                key=lambda index: len(coverage[index] & uncovered),
                                default=None,
                            )
                            if best is None or not (coverage[best] & uncovered):
                                break
                            selected.append(best)
                            uncovered -= coverage[best]
                        for anchor in (1, (len(candidate_frames) + 1) // 2, len(candidate_frames)):
                            if len(selected) >= 3:
                                break
                            if anchor not in selected and 1 <= anchor <= len(candidate_frames):
                                selected.append(anchor)
                        for index in selected[:3]:
                            frame = candidate_frames[index - 1]
                            retained_image = self.media_store.fit_visual_candidate(
                                frame.get("bytes"), 6_000)
                            saved = self.media_store._put(
                                "visual", retained_image, "webp", 6_000)
                            if saved:
                                saved.update({
                                    "mime": "image/webp",
                                    "dhash": frame.get("dhash") or "",
                                    "at_s": frame.get("at_s"),
                                    "source_id": frame.get("source_id") or dataset_id,
                                    "quality_tier": "recent",
                                })
                                saved_by_index[index] = saved
                        if saved_by_index:
                            selected_refs = list(saved_by_index.values())
                            multimodal["visual_sampling"] = {
                                "candidate_frames": len(candidate_frames),
                                "retained_frames": len(selected_refs),
                                "strategy": "object_coverage_with_temporal_anchors",
                            }
                    else:
                        saved_by_index = {
                            index: ref for index, ref in enumerate(visual_refs[:3], start=1)
                        }
                    for item in grounding.get("items") or []:
                        boxes = {}
                        box_scores = {}
                        for box in item.get("frame_boxes") or []:
                            try:
                                box_index = int(box.get("frame_index"))
                                bbox = box.get("bbox")
                            except (AttributeError, TypeError, ValueError):
                                continue
                            max_frame_index = len(candidate_frames) if candidate_frames else 7
                            if 1 <= box_index <= max_frame_index and isinstance(bbox, list) and len(bbox) == 4:
                                boxes[box_index] = bbox
                                try:
                                    box_scores[box_index] = float(box.get("score"))
                                except (AttributeError, TypeError, ValueError):
                                    pass
                        requested = []
                        for index in item.pop("frame_indices", []):
                            if isinstance(index, int) and index not in requested:
                                requested.append(index)
                        for index in boxes:
                            if index not in requested:
                                requested.append(index)
                        frame_refs = []
                        for index in requested[:1]:
                            if candidate_frames:
                                if not (1 <= index <= len(candidate_frames)):
                                    continue
                                frame = candidate_frames[index - 1]
                                source_ref = saved_by_index.get(index)
                                if not source_ref:
                                    source_bytes = self.media_store.fit_visual_candidate(
                                        frame.get("bytes"), 6_000)
                                    source_ref = self.media_store._put(
                                        "visual", source_bytes, "webp", 6_000)
                                    if source_ref:
                                        source_ref.update({
                                            "mime": "image/webp",
                                            "dhash": frame.get("dhash") or "",
                                            "at_s": frame.get("at_s"),
                                            "source_id": frame.get("source_id") or dataset_id,
                                            "quality_tier": "recent",
                                        })
                                        saved_by_index[index] = source_ref
                            else:
                                if not visual_refs:
                                    continue
                                ratio = (index - 1) / 6 if index > 1 else 0
                                ref_index = round(ratio * (len(visual_refs) - 1))
                                source_ref = visual_refs[max(0, min(ref_index, len(visual_refs) - 1))]
                                frame = {"at_s": source_ref.get("at_s"),
                                         "source_id": source_ref.get("source_id") or dataset_id}
                            if not source_ref or not source_ref.get("asset"):
                                continue
                            crop_ref = None
                            bbox = boxes.get(index) if candidate_frames else None
                            is_context_bbox = bool(
                                bbox and len(bbox) == 4
                                and float(bbox[0]) <= 0.005 and float(bbox[1]) <= 0.005
                                and float(bbox[2]) >= 0.995 and float(bbox[3]) >= 0.995
                            )
                            low_confidence_bbox = bool(
                                bbox and box_scores.get(index, float(item.get("score") or 0)) < 0.28)
                            # A full-frame fallback is contextual evidence, not a
                            # localized object crop. Do not present it as if the
                            # named object had been accurately isolated.
                            if bbox and not is_context_bbox and not low_confidence_bbox:
                                crop_bytes = self.media_store.crop_visual_candidate(
                                    frame.get("bytes"), bbox, 12_000)
                                crop_ref = self.media_store._put(
                                    "visual", crop_bytes, "webp", 12_000)
                            primary_ref = crop_ref or source_ref
                            frame_ref = {
                                "asset": primary_ref.get("asset"),
                                "bytes": int(primary_ref.get("bytes") or 0),
                                "at_s": frame.get("at_s"),
                                "source_id": frame.get("source_id") or dataset_id,
                                "quality_tier": "recent",
                            }
                            if crop_ref:
                                frame_ref.update({
                                    "source_asset": source_ref["asset"],
                                    "crop_bbox": [round(float(value), 5) for value in bbox],
                                    "visual_role": "object_crop",
                                })
                            else:
                                frame_ref["visual_role"] = "source_frame"
                                frame_ref["localization_status"] = (
                                    "context_only" if is_context_bbox else
                                    "low_confidence_context" if low_confidence_bbox else
                                    "local_crop_unavailable"
                                )
                                if low_confidence_bbox:
                                    item["localization_status"] = "low_confidence_context"
                            frame_refs.append(frame_ref)
                        item["frame_refs"] = frame_refs
                        item["crop_count"] = sum(
                            ref.get("visual_role") == "object_crop" for ref in frame_refs)
                        item.pop("frame_boxes", None)
                        if item.get("status") in {"grounded", "visible_unlabeled"} and not item["frame_refs"]:
                            item["status"] = "uncertain"
                            item["error"] = "本地裁剪器未能生成局部画面；已保留来源画面"
                    for item in grounding.get("items") or []:
                        item["appearance"] = _remove_visual_age_guess(item.get("appearance"))
                    grounding["crop_schema"] = 4
                    grounding["crop_method"] = (
                        "confidence-gated local region; weak localizations and full-frame fallbacks remain "
                        "source-frame context, never labeled as object crops"
                    )
                    multimodal["object_visuals"] = grounding
                    # Keep the semantic record aligned with grounded cards when
                    # numbered aliases resolve to the exact same source region.
                    collapsed_aliases = grounding.get("collapsed_object_aliases") or []
                    alias_map = {str(row.get("alias") or "").strip().casefold():
                                 str(row.get("canonical") or "").strip()
                                 for row in collapsed_aliases if isinstance(row, dict)}
                    if alias_map:
                        f["objects"] = [label for label in (f.get("objects") or [])
                                        if str(label or "").strip().casefold() not in alias_map]
                        object_observations = [row for row in object_observations
                                               if str(row.get("object") or "").strip().casefold() not in alias_map]
                        for fact in temporal_facts:
                            if (str(fact.get("subject_type") or "").casefold() == "object"
                                    and str(fact.get("subject") or "").strip().casefold() in alias_map):
                                fact["subject"] = alias_map[str(fact.get("subject") or "").strip().casefold()]
                        f["object_observations"] = object_observations
                        f["temporal_facts"] = temporal_facts
                    record_duration("object_crop_generation_and_asset_write_ms", object_crop_started)
                    record_duration("object_grounding_pipeline_wall_ms", object_pipeline_started)
                    if cancel_event is not None and cancel_event.is_set():
                        info = {
                            "error": "物品画面核对已停止，当前片段未写入记忆",
                            "source": "cancelled", "cancelled": True,
                        }
                        finish_timing(info)
                        return None, info
                person_rows = [row for row in person_observations
                               if isinstance(row, dict) and str(row.get("entity") or "").strip()]
                if person_rows and visual_refs:
                    person_pipeline_started = time.perf_counter()
                    update("person_visual_grounding_lookup", "正在检查人物局部画面的持久缓存")
                    person_frame_sample_started = time.perf_counter()
                    qwen_frame_times = [
                        float(ref.get("at_s")) for ref in visual_refs
                        if isinstance(ref, dict)
                        and ref.get("visual_role") == "qwen_input_frame"
                        and ref.get("at_s") is not None
                    ]
                    # Preserve the six exact Qwen frames as identity anchors, then
                    # add spaced in-between/end frames for appearance-grounded gallery crops.
                    qwen_person_frames = self.media_store.sample_visual_candidates(
                        video_path, clip_start, clip_end, source_id=dataset_id,
                        count=len(qwen_frame_times) or 7,
                        sample_times=qwen_frame_times or None)
                    person_frames = _sample_person_gallery_inputs(
                        self.media_store, video_path, clip_start, clip_end,
                         dataset_id, qwen_person_frames, max_samples=36)
                    reused_object_frames = False
                    if not person_frames:
                        person_frames = shared_candidate_frames or self.media_store.sample_visual_candidates(
                            video_path, clip_start, clip_end, source_id=dataset_id, count=7)
                    person_frame_times = [float(frame.get("at_s")) for frame in person_frames
                                          if frame.get("at_s") is not None]
                    person_highres_frames = self.media_store.sample_visual_candidates(
                        video_path, clip_start, clip_end, source_id=dataset_id,
                        count=len(person_frame_times) or 7, full_resolution=True,
                        sample_times=person_frame_times or None)
                    face_cache_key = None
                    try:
                        face_cache_key = self.grounding_cache.key(
                            dataset_id, ["yolov8n-face-detections-v3-hires-dense"], person_highres_frames)
                        cached_faces, face_lookup_ms = self.grounding_cache.get_faces(face_cache_key)
                        if isinstance(cached_faces, dict):
                            person_face_detections = cached_faces["faces"]
                            face_detection_status = "cache_hit"
                            operation_ms["face_detection_cache_hit"] = True
                            operation_ms["face_detection_cache_lookup_ms"] = face_lookup_ms
                        else:
                            person_face_detections = detect_faces(person_highres_frames) if person_highres_frames else []
                            face_detection_status = "completed"
                            operation_ms["face_detection_cache_hit"] = False
                            if face_cache_key and person_highres_frames:
                                self.grounding_cache.put_faces(face_cache_key, {
                                    "schema": 1, "status": "completed",
                                    "model": "YOLOv8n-Face",
                                    "frame_count": len(person_highres_frames),
                                    "faces": person_face_detections,
                                })
                        face_detection_error = ""
                    except Exception as face_exc:
                        person_face_detections = []
                        face_detection_status = "unavailable"
                        face_detection_error = str(face_exc)[:400]
                    record_duration("person_candidate_frame_sampling_ms", person_frame_sample_started)
                    operation_ms["person_candidate_frames_reused_from_objects"] = reused_object_frames
                    if not person_frames:
                        for ref in visual_refs[:3]:
                            asset_path = Path(self.mem.memory_dir) / str(ref.get("asset") or "")
                            if ref.get("asset") and asset_path.is_file():
                                person_frames.append({
                                    **ref, "mime": ref.get("mime") or "image/webp",
                                    "bytes": asset_path.read_bytes(),
                                })
                    person_key = None
                    person_cache_lookup_started = time.perf_counter()
                    try:
                        # Include the appearance hint so repeated clip-local labels cannot
                        # accidentally reuse a crop for a different visually described person.
                        person_cache_labels = [
                             "person-crop-v18-dense-face-count-residual-gallery: " + str(row.get("entity") or "").strip()
                            + " | " + str(row.get("appearance") or "").strip()
                    + " | face=" + str(row.get("face_features") or "").strip() + " | anchors=" + _person_anchor_cache_suffix(row)
                            for row in person_rows[:8]
                        ]
                        person_key = self.grounding_cache.key(
                            dataset_id, person_cache_labels, person_frames, person_face_detections,
                            implementation=PERSON_GROUNDING_IMPLEMENTATION_FINGERPRINT)
                        person_grounding, person_lookup_ms = self.grounding_cache.get(person_key)
                    except Exception:
                        person_key, person_grounding, person_lookup_ms = None, None, 0
                    record_duration("person_grounding_cache_lookup_wall_ms", person_cache_lookup_started)
                    operation_ms["person_grounding_cache_hit"] = person_grounding is not None
                    if person_grounding is not None:
                        update("person_grounding_cache_hit",
                               f"人物画面缓存已命中（{person_lookup_ms} ms），跳过视觉推理")
                    else:
                        update("person_visual_grounding", "缓存未命中，正在定位人物并生成局部裁剪")
                        try:
                            person_model_started = time.perf_counter()
                            person_grounding = self.perceiver.ground_people(
                                person_rows, person_frames, cancel_event=cancel_event,
                                face_detections=person_face_detections,
                                progress_callback=lambda event: update(
                                    event.get("stage", "person_visual_grounding"),
                                    event.get("message", "正在定位人物画面"),
                                    self._progress_text(m, {"qwen": f}),
                                ),
                            )
                            record_duration("person_grounding_model_call_ms", person_model_started)
                        except Exception as exc:
                            grounding_error = str(exc)[:500]
                            person_grounding = {
                                "schema": 1, "status": "unavailable", "model": "opencv-hog+clip-grid-v1",
                                "message": grounding_error,
                                "items": [{"person": str(row.get("entity") or ""),
                                           "status": "unavailable",
                                           "appearance": str(row.get("appearance") or ""),
                                           "error": grounding_error, "frame_refs": [],
                                           "crop_count": 0, "frame_indices": [],
                                           "frame_boxes": []}
                                          for row in person_rows[:8]],
                            }
                        self.grounding_cache.put(person_key, person_grounding)

                    person_crop_started = time.perf_counter()
                    person_items = person_grounding.get("items") or []
                    face_assignments = _grounded_face_assignments(
                        person_items, person_frames, person_highres_frames, person_face_detections)
                    gallery_candidates = []
                    for item_index, item in enumerate(person_items):
                        box_rows = {int(box.get("frame_index")): box
                                    for box in item.get("frame_boxes") or []
                                    if isinstance(box, dict)
                                    and isinstance(box.get("frame_index"), int)
                                    and isinstance(box.get("bbox"), list)}
                        boxes = {index: row.get("bbox") for index, row in box_rows.items()}
                        anchors = set()
                        for raw_index in item.get("anchor_frame_indices") or []:
                            try:
                                anchors.add(int(raw_index))
                            except (TypeError, ValueError):
                                continue
                        appearance_indices = set()
                        for raw_index in item.get("identity_candidate_frame_indices") or []:
                            try:
                                appearance_indices.add(int(raw_index))
                            except (TypeError, ValueError):
                                continue
                        for index in sorted((anchors | appearance_indices).intersection(boxes)):
                            if not (1 <= index <= len(person_frames)):
                                continue
                            ground_time = float(person_frames[index - 1].get("at_s") or 0)
                            highres_index = min(
                                range(len(person_highres_frames)),
                                key=lambda j: abs(float(person_highres_frames[j].get("at_s") or 0)-ground_time),
                            ) if person_highres_frames else -1
                            if highres_index < 0:
                                continue
                            source_frame = person_highres_frames[highres_index]
                            if abs(float(source_frame.get("at_s") or 0) - ground_time) > 0.35:
                                continue
                            face_row = face_assignments.get((item_index, highres_index))
                            if not face_row:
                                continue
                            face_bbox = face_row.get("bbox") or []
                            face_bytes = self.media_store.crop_visual_original(
                                source_frame.get("bytes"), face_bbox, face=True)
                            if not face_bytes:
                                continue
                            gallery_candidates.append({
                                "item_index": item_index,
                                "frame_index": index,
                                "highres_index": highres_index,
                                "is_anchor": index in anchors,
                                "identity_evidence": str(box_rows[index].get("identity_evidence") or ""),
                                "identity_margin": float(box_rows[index].get("identity_margin") or 0.0),
                                "person_box_score": float(box_rows[index].get("score") or 0.0),
                                "frame_face_count": len(person_face_detections[highres_index])
                                    if highres_index < len(person_face_detections) else 0,
                                "face_bytes": face_bytes,
                                "face_bbox": face_bbox,
                                "face_row": face_row,
                                "source_frame": source_frame,
                            })
                    resolved_gallery_candidates = resolve_face_gallery_candidates(
                        person_items, gallery_candidates)
                    refs_by_person = [[] for _ in person_items]
                    seen_assets_by_person = [set() for _ in person_items]
                    for candidate in resolved_gallery_candidates:
                        target_index = int(candidate.get("target_item_index", -1))
                        if not (0 <= target_index < len(person_items)):
                            continue
                        source_frame = candidate["source_frame"]
                        face_row = candidate["face_row"]
                        face_bbox = face_row.get("bbox") or []
                        face_bytes = candidate["face_bytes"]
                        face_ref = self.media_store._put(
                            "visual", face_bytes, "jpg", len(face_bytes) + 1)
                        if not face_ref or not face_ref.get("asset"):
                            continue
                        if face_ref["asset"] in seen_assets_by_person[target_index]:
                            continue
                        seen_assets_by_person[target_index].add(face_ref["asset"])
                        identity_method = str(candidate.get("identity_method") or "")
                        if candidate.get("is_anchor"):
                            evidence_method = "direct_qwen_frame+"
                        elif identity_method == "face_similarity_clip":
                            evidence_method = "face_similarity_clip+"
                        else:
                            evidence_method = "unique_appearance_clip+"
                        ref = {
                            "asset": face_ref["asset"],
                            "bytes": int(face_ref.get("bytes") or 0),
                            "at_s": source_frame.get("at_s"),
                            "source_id": source_frame.get("source_id") or dataset_id,
                            "visual_role": "face_crop",
                            "identity_verified": True,
                            "identity_assignment": "qwen_visual_anchor+" + evidence_method +
                                str(face_row.get("_identity_assignment") or "mutual_unique_face_geometry"),
                            "source_asset": source_frame.get("asset"),
                            "face_bbox": [round(float(value), 5) for value in face_bbox],
                            "detector": "YOLOv8n-Face",
                            "detector_confidence": round(float(face_row.get("confidence") or 0), 5),
                            "image_quality": "source_resolution_jpeg_q96",
                        }
                        if not candidate.get("is_anchor"):
                            ref["identity_match_score"] = round(float(candidate.get("identity_score") or 0), 5)
                            ref["identity_match_margin"] = round(float(candidate.get("identity_margin") or 0), 5)
                            ref["identity_match_method"] = identity_method
                        refs_by_person[target_index].append(ref)
                    for item_index, item in enumerate(person_items):
                        frame_refs = refs_by_person[item_index]
                        item["frame_refs"] = frame_refs
                        item["crop_count"] = len(frame_refs)
                        item.pop("frame_boxes", None)
                        item.pop("frame_indices", None)
                        if not frame_refs and item.get("status") not in {"uncertain", "not_visible"}:
                            item["status"] = "uncertain"
                            item["error"] = "No identity-verified face crop; source frames remain available"
                    person_grounding["crop_schema"] = 18
                    person_grounding["face_detection_model"] = "YOLOv8n-Face"
                    person_grounding["face_detection_device"] = os.environ.get("MEMORY_FACE_GPU", "cuda:0")
                    person_grounding["face_detection_status"] = face_detection_status
                    if face_detection_error:
                        person_grounding["face_detection_error"] = face_detection_error
                    person_grounding["crop_method"] = (
                        "Qwen identity anchors + dense GPU face detection + CLIP face-identity validation and relabeling")
                    multimodal["person_visuals"] = person_grounding
                    record_duration("person_crop_generation_and_asset_write_ms", person_crop_started)
                    record_duration("person_grounding_pipeline_wall_ms", person_pipeline_started)
                    if cancel_event is not None and cancel_event.is_set():
                        info = {
                            "error": "人物画面核对已停止，当前片段未写入记忆",
                            "source": "cancelled", "cancelled": True,
                        }
                        finish_timing(info)
                        return None, info
                # Context facts remain textual with source provenance. Dedicated
                # image evidence is limited to grounded people and objects.
                multimodal["entity_visuals"] = {"schema": 2, "assets": [], "items": []}
            except Exception as exc:
                # A media extraction failure must not prevent text memory CRUD.
                media_error = str(exc)
        source_audio_text = early_source_audio_text
        if not audio_transcription_attempted:
            audio_started = time.perf_counter()
            source_audio_text = self._transcribe_audio_evidence(
                multimodal, dataset_id, video_path=video_path,
                clip_start_s=clip_start, clip_end_s=clip_end,
                progress=lambda stage, message: update(stage, message),
                timing_sink=operation_ms,
            )
            record_duration("audio_decode_cache_and_transcription_wall_ms", audio_started)
        speech_scene_context = ""
        speech_user_action = ""
        reconciled_facts = []
        identity_links = []
        retrieval_before_started = time.perf_counter()
        retrieval_fields = dict(f)
        retrieval_fields["source_evidence"] = str(
            retrieval_fields.get("source_evidence") or source_audio_text or "")
        retrieved_memories_before_write, retrieval_before_meta = self._retrieve_before_write(
            dataset_id, retrieval_fields, dataset_id=dataset_id, progress=progress)
        record_duration("history_retrieval_before_write_wall_ms", retrieval_before_started)
        related_memories_for_profile = retrieved_memories_before_write[:5]
        if source_audio_text:
            original_observations = [
                dict(row) for row in person_observations if isinstance(row, dict)
            ]
            profile_frames = (locals().get("person_frames")
                              or shared_candidate_frames
                              or locals().get("qwen_input_frames") or [])
            person_observations, speech_scene_context, spoken_location, speech_user_action, reconciled_facts = self._enrich_person_profiles_from_speech(
                person_observations, source_audio_text, profile_frames, dataset_id,
                visual_facts=temporal_facts,
                document_evidence=f.get("document_evidence") or [],
                related_memories=related_memories_for_profile,
                cancel_event=cancel_event,
                visual_context={key: f.get(key) for key in
                                ("scene", "content_summary", "activity", "interaction_details", "location", "environments")},
                progress=lambda stage, message: update(stage, message),
            )
            identity_links = [
                {"term": str(alias.get("alias") or ""),
                 "entity": str(row.get("entity") or ""),
                 "usage": str(alias.get("usage") or ""),
                 "confidence": str(alias.get("confidence") or ""),
                 "basis": str(alias.get("basis") or "")}
                for row in person_observations if isinstance(row, dict)
                for alias in row.get("spoken_aliases") or [] if isinstance(alias, dict)
            ]
            # Include speech facts before role resolution so a document's named
            # subject can be linked to the addressee without relabeling its holder.
            temporal_facts = list(f.get("temporal_facts") or temporal_facts) + list(reconciled_facts or [])
            f["temporal_facts"] = temporal_facts
            reconciled_facts = []
            f["person_observations"] = person_observations
            f["document_evidence"] = f.get("document_evidence") or []
            ground_clinical_visit(f, person_observations, source_audio_text, dataset_id)
            person_observations = f.get("person_observations") or person_observations
            temporal_facts = f.get("temporal_facts") or temporal_facts
            if spoken_location:
                f["location"] = spoken_location
            if speech_user_action:
                f["user_action"] = speech_user_action
            f["person_observations"] = person_observations
            f["people"] = list(dict.fromkeys(
                str(row.get("entity") or "").strip()
                for row in person_observations
                if isinstance(row, dict) and str(row.get("entity") or "").strip()
            ))
            alias_map = _person_label_mapping_by_anchor(
                original_observations, person_observations)
            label_map = {**alias_map, **_person_name_ordinal_aliases(person_observations)}
            _remap_person_fact_references_by_anchor(
                f, original_observations, person_observations)
            _remap_person_labels_in_fields(f, label_map)
            _remap_person_visual_labels_by_anchor(multimodal, person_observations)
            multimodal = _remap_person_labels_in_media(multimodal, label_map)
        before_face_reid = [dict(row) for row in person_observations if isinstance(row, dict)]
        person_observations, face_identity_matches = _reconcile_people_with_retrieved_faces(
            person_observations, multimodal, related_memories_for_profile,
            self.media_store.root, dataset_id)
        if face_identity_matches:
            f["person_observations"] = person_observations
            _remap_person_fact_references_by_anchor(
                f, before_face_reid, person_observations)
            _remap_person_visual_labels_by_anchor(multimodal, person_observations)
            f["people"] = list(dict.fromkeys(
                str(row.get("entity") or "").strip()
                for row in person_observations
                if isinstance(row, dict) and str(row.get("entity") or "").strip()
            ))
            identity_links.extend({
                "term": row["to"], "entity": row["to"],
                "confidence": "high" if row["score"] >= 0.75 else "medium",
                "basis": "unique verified cross-clip face embedding",
                "source_segment_id": row.get("matched_source_segment_id", ""),
                "match_score": row["score"],
            } for row in face_identity_matches)
        # Speech facts now come from the shared source-reconciliation step above;
        # no topic-specific regex correction layer rewrites them afterward.
        if source_audio_text:
            f["source_evidence"] = source_audio_text
        # Keep a distinct situation-level gist. The chronological scene and
        # source-linked interaction facts remain in their own fields.
        social_context = str(speech_scene_context or "").strip()
        f["content_summary"] = _content_summary_for_fields(
            f, speech_context=social_context)
        f["people"] = person_labels_from_observations(
            f.get("people") or [], person_observations)
        if reconciled_facts:
            temporal_facts = list(f.get("temporal_facts") or temporal_facts) + reconciled_facts
            f["temporal_facts"] = temporal_facts
        if multimodal and f.get("document_evidence"):
            multimodal["document_evidence"] = copy.deepcopy(f["document_evidence"])
        f["person_observations"] = person_observations
        f["temporal_facts"] = temporal_facts
        normalize_object_transition_facts(f)
        temporal_facts = f.get("temporal_facts") or temporal_facts
        # Descriptive fields are generated in English by the source-analysis
        # prompts. Exact transcript/document quotations remain in their source
        # language as auditable evidence; do not post-translate generated fields.
        f["person_observations"] = person_observations
        f["temporal_facts"] = temporal_facts
        normalize_user_facing_memory_copy(f)
        temporal_facts = f.get("temporal_facts") or temporal_facts
        # Apply role conflict cleanup after all document and clinical reconciliation,
        # which may add a specific role after the initial speech pass.
        for person_row in person_observations:
            if isinstance(person_row, dict):
                _ensure_episode_role(person_row, f, source_audio_text)
                # Validate all user-facing profile semantics after role inference.
                # Original-language speech remains in source evidence references.
                role_values = [str(value).strip() for value in person_row.get("roles") or []
                               if isinstance(value, str) and value.strip()
                               and not re.search(r"[\u3400-\u9fff]", value)]
                person_row["roles"] = role_values
                relation_values = person_row.get("relationships") or []
                if isinstance(relation_values, str):
                    relation_values = [relation_values]
                person_row["relationships"] = [
                    str(value).strip() for value in relation_values
                    if isinstance(value, str) and value.strip()
                    and not re.search(r"[\u3400-\u9fff]", value)
                ]
                if re.search(r"[\u3400-\u9fff]", str(person_row.get("profile_context") or "")):
                    person_row["profile_context"] = ""
                person_row["spoken_aliases"] = [
                    alias for alias in person_row.get("spoken_aliases") or []
                    if isinstance(alias, dict)
                    and not re.search(r"[\u3400-\u9fff]", str(alias.get("alias") or ""))
                ]
                entity = str(person_row.get("entity") or "").strip()
                if re.search(r"[\u3400-\u9fff]", entity):
                    fallback = next((role for role in role_values
                                     if role.casefold() not in {"man", "woman", "person"}), "")
                    visual_label = str(person_row.get("visual_entity") or "").strip()
                    person_row["entity"] = fallback or (
                        visual_label if visual_label and not re.search(r"[\u3400-\u9fff]", visual_label)
                        else "unidentified person")
        # Rebind crop groups after names and episode roles have been canonicalized.
        _remap_person_visual_labels_by_anchor(multimodal, person_observations)
        retained_people = _filter_incidental_person_profiles(person_observations, f)
        dropped_entities = {
            str(row.get("entity") or "").strip().casefold()
            for row in person_observations
            if isinstance(row, dict)
            and row not in retained_people
            and str(row.get("entity") or "").strip()
        }
        person_observations = retained_people
        f["person_observations"] = person_observations
        f["people"] = list(dict.fromkeys(
            str(row.get("entity") or "").strip()
            for row in person_observations
            if isinstance(row, dict) and str(row.get("entity") or "").strip()
        ))
        if dropped_entities and isinstance(multimodal, dict):
            entities = multimodal.get("entities")
            if isinstance(entities, dict):
                entities["people"] = [
                    value for value in entities.get("people") or []
                    if str(value or "").strip().casefold() not in dropped_entities
                ]
            person_visuals = multimodal.get("person_visuals")
            dropped_assets = set()
            if isinstance(person_visuals, dict):
                kept_items = []
                for item in person_visuals.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("person") or item.get("entity") or "").strip().casefold() in dropped_entities:
                        for ref in item.get("frame_refs") or []:
                            if isinstance(ref, dict) and ref.get("asset"):
                                dropped_assets.add(str(ref["asset"]))
                    else:
                        kept_items.append(item)
                person_visuals["items"] = kept_items
            entity_visuals = multimodal.get("entity_visuals")
            if isinstance(entity_visuals, dict):
                old_assets = entity_visuals.get("assets") or []
                keep_indices = [
                    index for index, asset in enumerate(old_assets)
                    if not (isinstance(asset, dict)
                            and str(asset.get("asset") or "") in dropped_assets)
                ]
                remap = {old_index: new_index for new_index, old_index in enumerate(keep_indices)}
                entity_visuals["assets"] = [old_assets[index] for index in keep_indices]
                kept_items = []
                for item in entity_visuals.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    if (str(item.get("entity") or "").strip().casefold() in dropped_entities
                            and item.get("entity_type") == "person"):
                        continue
                    if isinstance(item.get("asset_indices"), list):
                        item["asset_indices"] = [
                            remap[index] for index in item["asset_indices"]
                            if index in remap
                        ]
                    kept_items.append(item)
                entity_visuals["items"] = kept_items
        _repair_anonymized_person_references(f)
        if cancel_event is not None and cancel_event.is_set():
            info = {"error": "当前片段处理已终止", "source": "cancelled", "cancelled": True}
            finish_timing(info)
            update("cancelled", info["error"])
            return None, info
        memory_write_started = time.perf_counter()
        mid = self.daytime.ingest(
            scene=f["scene"],
            user_action=f["user_action"],
            needs=[],
            solutions=[],
            people=f["people"] or None,
            location=f["location"],
            activity=f["activity"],
            environments=f["environments"],
            objects=f["objects"],
            extra_notes=(f.get("time_occasion") or "")[:200],
            timestamp=(time.strftime("%Y-%m-%d") + "T" + (_clock_from_id(m.get("id", "")) or time.strftime("%H:%M:%S"))),
            source_evidence=f.get("source_evidence") or "",
            interaction_details=f.get("interaction_details") or "",
            content_summary=f.get("content_summary") or "",
            person_observations=person_observations,
            temporal_facts=temporal_facts,
            source_time=source_time,
            multimodal=multimodal,
            source_provenance=source_provenance,
        )
        if f.get("object_observations"):
            self.mem.update(mid, {"object_observations": f["object_observations"]})
        record_duration("memory_write_and_index_ms", memory_write_started)
        if multimodal and multimodal.get("visual_sampling"):
            self.media_store.release_unreferenced(
                self.mem.memory["layer7"]["moments"])
        self.id_map[m["id"]] = mid
        self._qwen_fields[m["id"]] = f
        info = {
            "dataset_id": m.get("id", ""),
            "moment_id": mid,
            "qwen": f,
            "raw": r.get("raw", ""),
            "raw_model_output": r.get("raw_model_output", ""),
            "elapsed_ms": r.get("elapsed_ms"),
            "num_frames": r.get("num_frames"),
            "source": r.get("source", "qwen"),
            "quality_warning": r.get("quality_warning", ""),
            "stale_cache_fallback": bool(r.get("stale_cache_fallback")),
            "spoken_identity_links": identity_links,
            "retrieved_memories_before_write": retrieved_memories_before_write,
            "cross_clip_face_identity_matches": face_identity_matches,
            "retrieval_before_write_meta": retrieval_before_meta,
            "multimodal_refreshed": False,
            "eval": golden_eval(f, m.get("ground_truth", {})),
            "multimodal": multimodal,
            "media_error": media_error,
        }
        info["retrieved_memories"] = []
        info["retrieval_meta"] = {"status": "reused_prewrite",
            "duration_ms": retrieval_before_meta.get("wrapper_elapsed_ms", 0),
            "used_for_current_memory": bool(retrieved_memories_before_write)}
        retrieval_message = ("片段已写入；写入前索引检索到 %d 条相关历史记忆" %
                             len(retrieved_memories_before_write)
                             if retrieved_memories_before_write else
                             "片段已写入；写入前未检索到相关历史记忆")
        update("done", retrieval_message, self._progress_text(m, info))
        finish_timing(info)
        self._set_step_progress(completed_item=self._progress_text(m, info))
        return mid, info

    # ------------------------------------------------------------------
    # 会话构建
    # ------------------------------------------------------------------

    def _build(self):
        # Interactive sessions share one persistent accumulation pool. Dataset
        # selectors control the next steps, not the lifetime of stored memories.
        mem_dir = RUNS / "interactive_mem" / "current"
        mem_dir.mkdir(parents=True, exist_ok=True)
        resuming = self._resume_state is not None
        memory_path = mem_dir / "memory.json"
        if not memory_path.exists():
            # Seed the new shared pool with existing interactive L7 events while
            # leaving the scenario-specific source directories untouched.
            sources = []
            for name in ("short_term", "long_term", "custom_test"):
                source_dir = RUNS / "interactive_mem" / name
                source_file = source_dir / "memory.json"
                if source_file.is_file():
                    try:
                        loaded = json.loads(source_file.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        continue
                    if isinstance(loaded, dict):
                        sources.append((name, source_dir, loaded))
            if sources:
                seeded = copy.deepcopy(sources[0][2])
                moments = {}
                for name, source_dir, data in sources:
                    for moment_id, moment in ((data.get("layer7") or {}).get("moments") or {}).items():
                        if moment_id not in moments:
                            moments[moment_id] = copy.deepcopy(moment)
                    asset_dir = source_dir / "assets"
                    if asset_dir.is_dir():
                        shutil.copytree(asset_dir, mem_dir / "assets", dirs_exist_ok=True)
                    corrections = source_dir / "person_identity_corrections.json"
                    if corrections.is_file() and not (mem_dir / corrections.name).exists():
                        shutil.copy2(corrections, mem_dir / corrections.name)
                seeded["layer1"], seeded["layer2"], seeded["layer3"] = [], [], []
                seeded["layer4"] = {
                    "summary": "", "current_tasks": [], "stated_plans": [],
                    "life_trajectory": "", "last_updated": None,
                    "source_moment_count": 0, "source_moment_ids": [],
                    "source_link_coverage": "unknown",
                }
                seeded["layer5"] = {
                    "summary": "", "key_events": [], "long_term_patterns": [],
                    "last_updated": None, "source_moment_count": 0,
                    "source_moment_ids": [], "source_link_coverage": "unknown",
                }
                seeded["layer6"] = {
                    "summary": "", "profile": {}, "last_updated": None,
                    "source_moment_count": 0, "source_moment_ids": [],
                    "source_link_coverage": "unknown",
                }
                seeded["layer7"] = {
                    "people": {}, "locations": {}, "time_nodes": {},
                    "activity_events": {}, "environments": {}, "objects": {},
                    "moments": moments,
                }
                metadata = seeded.setdefault("metadata", {})
                metadata["total_moments"] = len(moments)
                metadata["nightly_processed_days"] = {}
                metadata["accumulation_seed_sources"] = [name for name, _, _ in sources]
                temporary = memory_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(seeded, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, memory_path)
        self.mem = PersonMemory(memory_dir=str(mem_dir))
        self.media_store = MultimodalAssetStore(mem_dir)
        if strip_audio_assets(self.mem.memory):
            self.mem._save()
            self.media_store._bytes_cached = None
            self.media_store.release_unreferenced(
                self.mem.memory.get("layer7", {}).get("moments", {}))
        self._install_operation_audit()
        self.daytime = DaytimeProcessor(self.mem)
        self.daytime.start_scene(f"{self.key}_{time.strftime('%Y-%m-%d')}")
        self.id_map = {}        # 数据集 id → 记忆模型 moment_id
        self._qwen_fields = {}  # 数据集 id → Qwen 产出（sort / 展示用）

        if self.key == "short_term":
            self.steps = self._plan_short()
        elif self.key == "custom_test":
            self.steps = self._plan_custom()
        else:
            self.steps = self._plan_long()

        # The shared memory pool survives dataset switches and process restarts.
        # Restore the source-to-memory map for every session so rerunning a clip
        # refreshes its existing episode instead of appending a duplicate.
        archive = self.mem.memory.get("layer7", {}).get("moments", {})
        for moment_id, moment in archive.items():
            if not isinstance(moment, dict) or moment.get("stale") or moment.get("merged_into"):
                continue
            provenance = moment.get("source_provenance") or {}
            source_time = moment.get("source_time") or {}
            multimodal = moment.get("multimodal") or {}
            source_id = str(provenance.get("source_segment_id")
                            or source_time.get("source_segment_id")
                            or multimodal.get("source_id") or "")
            if not source_id:
                continue
            previous_id = self.id_map.get(source_id)
            previous = archive.get(previous_id) or {}
            rank = (str(moment.get("timestamp") or ""), str(moment_id))
            previous_rank = (str(previous.get("timestamp") or ""), str(previous_id or ""))
            if not previous_id or rank > previous_rank:
                self.id_map[source_id] = moment_id

        if resuming:
            checkpoint = self._resume_state
            expected_dataset = str(checkpoint.get("dataset_id") or "")
            if expected_dataset and expected_dataset != self.dataset_id:
                raise ValueError("会话恢复数据集与当前记忆库不一致")
            self.pos = max(0, min(int(checkpoint.get("pos") or 0), len(self.steps)))
            self.log = copy.deepcopy(checkpoint.get("log") or [])
            strip_audio_assets(self.log)
            self.operation_log = copy.deepcopy(checkpoint.get("operations") or [])
            self.auto_paused = bool(checkpoint.get("auto_paused"))
            self.step_cancelled = bool(checkpoint.get("step_cancelled"))
            for moment in self.moments:
                cached = self.perceiver.cache.get(moment.get("id", "")) or {}
                if cached.get("fields"):
                    self._qwen_fields[moment["id"]] = cached["fields"]
            self.step_progress = None
            self.nightly_progress = None
            # Refresh the latest restored visual snapshot from disk so a
            # completed media backfill is visible without re-running the step.
            if self.log:
                latest = self.log[-1]
                snap = rs.snapshot(
                    self.mem, str(latest.get("title") or "恢复后的记忆快照"),
                    action=str(latest.get("kind") or "resume"))
                snap["full"] = self._full_snapshot()
                snap["layer7_active_moments"] = snap["full"]["layer7_active_total"]
                snap["layer7_merged_moments"] = snap["full"]["layer7_merged_total"]
                latest["snapshot"] = snap
            self._resume_state = None
        else:
            self._log_snapshot("初始状态", "init", "会话创建，等待第一步")

    def _display_brief(self, m):
        """Current-step preview uses the same cached English fields that feed memory."""
        brief = moment_brief(m)
        cache_entry = self.perceiver.cache.get(m.get("id", "")) or {}
        cache_is_current = cache_entry.get("prompt_version") == PROMPT_VERSION
        brief['preview_source'] = 'previous_perception' if cache_is_current else 'video_only'
        brief['preview_prompt_version'] = cache_entry.get('prompt_version') if cache_is_current else ''
        cached = (cache_entry.get("fields") or {}) if cache_is_current else {}
        if cached:
            if cached.get("scene"):
                brief["scene"] = cached["scene"]
            if cached.get("user_action"):
                brief["action"] = cached["user_action"]
            for key in ("people", "location", "activity"):
                if cached.get(key):
                    brief[key] = cached[key]
        return brief

    def _plan_short(self):
        """短期场景：逐条写入 + CRUD 计划，可按人物筛选当前演示集。"""
        if self.dataset_key not in ("base", "expanded"):
            raise ValueError(f"未知短期数据集：{self.dataset_key}")
        source_file, plan_file = (
            (SHORT_EXPANDED_FILE, SHORT_EXPANDED_PLAN)
            if self.dataset_key == "expanded" else (SHORT_FILE, SHORT_PLAN)
        )
        all_moments = _load_jsonl(source_file)
        all_plan = json.load(open(plan_file, encoding="utf-8"))
        if self.participant:
            moments = [
                m for m in all_moments
                if "_".join(str(m.get("id", "")).split("_")[1:3]) == self.participant
            ]
            if not moments:
                raise ValueError(f"短期交互集没有参与者 {self.participant}")
            def for_participant(item_id):
                return "_".join(str(item_id).split("_")[1:3]) == self.participant

            plan = {
                **all_plan,
                "add": [m.get("id") for m in moments],
                "update": [x for x in all_plan.get("update", [])
                           if for_participant(x.get("id"))],
                "highlight": [x for x in all_plan.get("highlight", [])
                              if for_participant(x)],
                "delete": [x for x in all_plan.get("delete", [])
                           if for_participant(x.get("id"))],
            }
            # 每个人物都保留完整的 CRUD 演示路径；没有预设操作时，
            # 从该人物自己的片段生成一条可追溯的演示操作。
            # Do not synthesize memory operations from benchmark labels.
            # A missing CRUD demo is preferable to contaminating memory.
            if not plan["highlight"]:
                plan["highlight"] = [moments[0]["id"]]
            if not plan["delete"]:
                delete_target = next(
                    (m["id"] for m in reversed(moments)
                     if m["id"] not in plan["highlight"]),
                    moments[-1]["id"],
                )
                plan["delete"] = [{
                    "id": delete_target,
                    "reason": "短期场景结束后清理该人物的演示记忆",
                }]
        else:
            moments = all_moments
            plan = all_plan

        # CRUD 演示只写入明确标注的演示备注，不把官方参考标签写回记忆。
        plan = dict(plan)
        plan["update"] = [
            {**case, "field": "notes", "new_value": "CRUD field update demo"}
            for case in plan.get("update", [])
        ]
        self.moments = moments
        self.plan = plan
        self.dataset_options = interactive_dataset_options()
        self.dataset_id = f"short_term|{self.dataset_key}|{self.participant or 'all'}"
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")
        self.participant_label = PARTICIPANT_NAMES.get(
            self.participant, "全部人物" if not self.participant else self.participant)
        steps = []

        # 1) add：每条视频片段一步（Qwen 感知 → 记忆）
        for i, m in enumerate(moments, 1):
            def run(m=m, i=i, force=False):
                context = {"item_index": 1, "item_total": 1,
                           "label": f"写入记忆 · {i}/{len(moments)}"}
                mid, info = self._add_one(m, force=force, progress=context)
                if mid is None:
                    msg = ("实时推理已停止" if info.get("cancelled") else "感知失败")
                    return f"{msg}：{info.get('error')}", {"kind": "single", **info}
                msg = f"Qwen 看过 {m['id']} → 写入记忆 {mid}（{info['source']}，{info['elapsed_ms']}ms）"
                return msg, {"kind": "single", **info}
            steps.append({
                "kind": "add",
                "title": f"感知并写入视频片段 {i}/{len(moments)}",
                "inputs": [self._display_brief(m)],
                "run": run,
            })

        # 相似事件只进入待确认队列；用户逐条核对后才会写入归并关系。
        def run_merge_candidates(force=False):
            candidates = self.mem.find_merge_candidates(limit=100)
            result = {
                "kind": "operation_result", "operation": "merge_candidates",
                "title": "相似事件候选 · 逐条确认后才会合并",
                "candidates": candidates,
            }
            return f"发现 {len(candidates)} 组相似事件候选；没有自动合并", result
        steps.append({
            "kind": "merge_candidates",
            "title": "检查相似事件候选（需要逐条确认）",
            "detail": {"note": "候选依据：同源片段重叠≥80%，或120秒内Scene与Action合并文本相似度≥92%且共享人物/物品。画面与语音转写文本只供核对；扫描不改数据，逐条确认后才归并。"},
            "inputs": [],
            "run": run_merge_candidates,
        })

        # 2) query
        for q in plan.get("query", []):
            def run(q=q, force=False):
                hits = self.daytime.query(q["query"], top_k=3)
                items = [self._moment_detail(hit) for hit in hits]
                result = {
                    "kind": "operation_result", "operation": "query",
                    "title": "检索命中内容", "query": q["query"],
                    "count": len(hits), "items": items,
                }
                return f"检索「{q['query']}」命中 {len(hits)} 条", result
            steps.append({
                "kind": "query",
                "title": f"检索「{q['query']}」",
                "detail": {"query": q["query"], "top_k": 3},
                "inputs": [],
                "run": run,
            })

        # 3) update
        for case in plan.get("update", []):
            def run(case=case, force=False):
                mid = self.id_map.get(case["id"])
                if not mid:
                    return f"未找到 {case['id']}，跳过", {
                        "kind": "operation_result", "operation": "update",
                        "title": "更新对象未找到", "items": [],
                    }
                before = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
                field = case["field"]
                old_value = copy.deepcopy(before.get(field))
                ok = self.mem.update(mid, {field: case["new_value"]})
                after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
                item = self._moment_detail(after)
                item["changes"] = [{
                    "field": field, "before": old_value,
                    "after": copy.deepcopy(after.get(field)),
                }]
                result = {
                    "kind": "operation_result", "operation": "update",
                    "title": "字段更新前后", "items": [item],
                }
                return f"更新 {case['id']}.{field}（{'成功' if ok else '字段未变化'}）", result
            steps.append({
                "kind": "update",
                "title": f"更新字段 {case['field']}",
                "detail": {"id": case["id"], "field": case["field"],
                           "new_value": case["new_value"]},
                "inputs": [],
                "run": run,
            })

        # 4) update_feedback：在独立交互演示库模拟纠错、确认与评分，不使用评测标签。
        update_ids = {case.get("id") for case in plan.get("update", [])}
        protected_ids = set(plan.get("highlight", [])) | {
            case.get("id") for case in plan.get("delete", [])
        }
        feedback_moment = next(
            (m for m in moments if m.get("id") not in update_ids | protected_ids),
            moments[0] if moments else None,
        )
        def run_feedback(force=False, m=feedback_moment):
            if not m:
                return "没有可反馈的记忆", None
            mid = self.id_map.get(m.get("id"))
            if not mid:
                return f"未找到 {m.get('id')}，反馈未保存", None
            corrections = {"notes": "CRUD feedback correction demo"}
            before = copy.deepcopy(
                self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            )
            ok = self.mem.update_feedback(
                mid, corrections=corrections, confirmed=True, user_rating=5,
            )
            after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            item = self._moment_detail(after)
            item["corrections"] = [
                {"field": field, "before": copy.deepcopy(before.get(field)), "after": value}
                for field, value in corrections.items()
            ]
            item["confirmed"] = after.get("feedback", {}).get("confirmed")
            item["user_rating"] = after.get("feedback", {}).get("user_rating")
            result_detail = {
                "kind": "operation_result", "operation": "update_feedback",
                "title": "模拟反馈修正内容", "simulated": True,
                "items": [item],
            }
            result = "已保存模拟纠错、确认和 5 星评分" if ok else "模拟反馈保存失败"
            return result, result_detail
        steps.append({
            "kind": "update_feedback",
            "title": "模拟反馈 · 纠错、确认与评分",
            "detail": {
                "id": feedback_moment.get("id", "") if feedback_moment else "",
                "note": "仅写入交互演示库的演示备注；不读取官方参考标签。确认和评分为模拟输入，不代表真实用户评价。",
            },
            "inputs": [],
            "run": run_feedback,
        })

        # 5) highlight
        hl_ids = plan.get("highlight", [])
        def run_highlight(force=False):
            items, n = [], 0
            moments = self.mem.memory.get("layer7", {}).get("moments", {})
            for sid in hl_ids:
                mid = self.id_map.get(sid)
                before = copy.deepcopy(moments.get(mid, {})) if mid else {}
                ok = bool(mid and self.mem.highlight(mid))
                after = moments.get(mid, before) if mid else before
                if ok:
                    n += 1
                item = self._moment_detail(after)
                item.update({"dataset_id": sid, "marked": ok,
                             "was_highlighted": bool(before.get("highlighted"))})
                items.append(item)
            result = {
                "kind": "operation_result", "operation": "highlight",
                "title": "已标记为重点的记忆", "items": items,
            }
            return f"标记重要记忆 {n}/{len(hl_ids)} 条", result
        steps.append({
            "kind": "highlight",
            "title": "标记重要记忆",
            "detail": {"ids": hl_ids},
            "inputs": [],
            "run": run_highlight,
        })

        # 6) delete（含 highlight 保护验证）
        del_cases = plan.get("delete", [])
        def run_delete(force=False):
            targets = []
            for case in del_cases:
                mid = self.id_map.get(case["id"])
                moment = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid) if mid else None
                targets.append((case, mid, copy.deepcopy(moment) if isinstance(moment, dict) else None))
            cleared = self.daytime.end_scene()
            msgs = [f"场景结束：清理 Layer1/2 {cleared} 条"]
            items, ok_n = [], 0
            for case, mid, before in targets:
                if not mid:
                    items.append({"dataset_id": case.get("id"), "deleted": False,
                                  "reason": case.get("reason", "manual"), "memory": {}})
                    continue
                ok = self.mem.delete(mid, reason=case.get("reason", "manual"))
                if ok:
                    ok_n += 1
                items.append({
                    **self._moment_detail(before or {}),
                    "dataset_id": case.get("id"),
                    "deleted": bool(ok),
                    "reason": case.get("reason", "manual"),
                })
            msgs.append(f"清理短期记忆：删除 {ok_n} 条")
            protection = []
            # 验证 highlight 保护，并把被验证的目标也明确展示出来。
            prot = [self.id_map[s] for s in hl_ids if self.id_map.get(s)]
            if prot:
                protected_moment = self.mem.memory.get("layer7", {}).get("moments", {}).get(prot[0], {})
                protected_detail = self._moment_detail(copy.deepcopy(protected_moment))
                blocked = not self.mem.delete(prot[0], reason="验证保护")
                forced = self.mem.delete(prot[0], reason="验证强制删除", force=True)
                protection.append({
                    **protected_detail, "default_delete_blocked": bool(blocked),
                    "force_deleted": bool(forced),
                })
                msgs.append(f"highlight 保护：默认删除被拒绝 {'✅' if blocked else '❌'}；"
                            f"force 强删 {'成功 ✅' if forced else '失败 ❌'}")
            result = {
                "kind": "operation_result", "operation": "delete",
                "title": "删除对象及保护验证", "items": items,
                "protection_checks": protection,
            }
            return "；".join(msgs), result
        steps.append({
            "kind": "delete",
            "title": "清理短期记忆（含保护验证）",
            "detail": {"ids": [c["id"] for c in del_cases]},
            "inputs": [],
            "run": run_delete,
        })

        return steps

    def _nightly_consolidation_step(self, title=None, note=None):
        """Build the shared scene-end/nightly consolidation step."""
        def run_nightly(force=False, use_cache=True, close_scene=True):
            self.nightly_progress = {"running": True, "stage": "start",
                                     "label": "启动夜间整理", "current": 0,
                                     "total": 0, "message": "", "updated_at": time.time()}
            def on_progress(event):
                self.nightly_progress = {"running": True, **event, "updated_at": time.time()}
            try:
                result = NightlyProcessor(self.mem).run(
                    day=self.mem.memory["metadata"]["today"],
                    daytime_processor=self.daytime,
                    force=force,
                    use_cache=use_cache,
                    progress_callback=on_progress,
                    all_accumulated=True,
                )
            except Exception as exc:
                self.nightly_progress = {
                    "running": False, "stage": "error", "label": "夜间整理失败",
                    "current": 0, "total": 0, "message": str(exc)[:300],
                    "updated_at": time.time(),
                }
                raise
            self.nightly_progress = {
                "running": False, "stage": "done", "label": "夜间整理完成",
                "current": 1, "total": 1,
                "message": f"缓存命中 {result.get('cache_hits', 0)} 项，模型调用 {result.get('model_calls', 0)} 次",
                "cache_hits": result.get("cache_hits", 0),
                "model_calls": result.get("model_calls", 0),
                "updated_at": time.time(),
            }
            if not result.get("ok"):
                return "夜间整理失败", {"kind": "nightly", "result": result}
            cleared = self.daytime.end_scene() if close_scene else 0
            result["short_term_cleared"] = cleared
            result["scene_kept_open"] = not close_scene
            profile_result = (
                "画像更新成功" if result.get("profile_updated")
                else "未发现足够稳定的画像属性"
            )
            msg = (
                f"场景结束，夜间归纳完成：处理 {result['moment_count']} 条，"
                f"分 {result['chunk_count']} 批，{profile_result}；"
                f"已清理 Layer1/2 {cleared} 条，Layer7 归档保留"
            )
            return msg, {"kind": "nightly", "result": result}
        return {
            "kind": "nightly",
            "title": title or "场景结束，夜间归纳",
            "detail": {"note": note or (
                "复用记忆流动中的夜间流程：从跨视频、跨测试集累积的记忆建立分类索引、"
                "归纳 L4/L5，并尝试更新 L6 用户画像。只按已有记忆和明确证据归纳；"
                "若测试片段没有指向同一用户的稳定信息，画像保持为空是正常结果。"
            )},
            "inputs": [],
            "run": run_nightly,
        }

    def _plan_custom(self):
        """Run a selected test collection as an isolated, one-off retrieval evaluation."""
        self.dataset_options = interactive_dataset_options()
        if not self.dataset_key.startswith("feishu:"):
            self.dataset_key = "feishu:all"
        selector = self.dataset_key
        moments = feishu_test_moments(selector)
        if not moments:
            raise ValueError("所选自定义测试集没有可运行的视频")

        category = selector.split(":", 1)[1]
        category_name = ("全部类别" if category == "all"
                         else FEISHU_TESTSET_LABELS.get(category, category))
        self.participant = None
        self.moments = moments
        self.meta = {
            "participant": "Feishu custom test set",
            "available_count": len(moments),
            "total_candidates": len(moments),
            "coverage": 1.0,
        }
        self.dataset_id = f"custom_test|{selector}|all"
        self.participant_label = category_name
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")

        steps = []
        total = len(moments)
        for index, moment in enumerate(moments, 1):
            def run(moment=moment, index=index, force=False):
                context = {
                    "item_index": index,
                    "item_total": total,
                    "label": f"自定义测评 · 片段 {index}/{total}",
                }
                self._set_step_progress(
                    **context,
                    stage="item_start",
                    message=f"正在感知并检索第 {index}/{total} 段：{moment.get('id', '')}",
                )
                moment_id, info = self._add_one(moment, force=force, progress=context)
                if moment_id is None:
                    message = "实时推理已停止" if info.get("cancelled") else "感知失败"
                    return f"{message}：{info.get('error')}", {"kind": "single", **info}
                return (
                    None,
                    {"kind": "single", **info},
                )

            steps.append({
                "kind": "add",
                "title": f"场景测评 · 写入并检索片段 {index}/{total}",
                "inputs": [self._display_brief(moment)],
                "run": run,
            })
        steps.append(self._nightly_consolidation_step())
        return steps

    def _plan_long(self):
        """长期场景：白天按小时批量写入 + 夜间整理 4 步。"""
        self.dataset_options = interactive_dataset_options()
        if self.dataset_key.startswith("feishu:"):
            selector = self.dataset_key
            moments = feishu_test_moments(selector)
            category = selector.split(":", 1)[1]
            category_name = ("全部类别" if category == "all"
                             else FEISHU_TESTSET_LABELS.get(category, category))
            self.participant = None
            self.moments = moments
            self.meta = {
                "participant": "Feishu custom test set",
                "hours_covered": ["12"],
                "available_count": len(moments),
                "total_candidates": len(moments),
                "coverage": 1.0,
                "complete": False,
            }
            self.dataset_id = f"long_term|{selector}|all"
            self.participant_label = f"自定义测试集 · {category_name}"
        else:
            self.dataset_key = "long_day"
            default_meta = json.load(open(LONG_META, encoding="utf-8"))
            participant = self.participant or default_meta.get("participant", "A1_JAKE")
            meta_path = LONG_PERSON_DIR / f"{participant}.meta.json"
            data_path = LONG_PERSON_DIR / f"{participant}.jsonl"
            if not meta_path.exists() or not data_path.exists():
                raise ValueError(f"没有 {participant} 的长期数据集")
            meta = json.load(open(meta_path, encoding="utf-8"))
            moments = _load_jsonl(data_path)
            self.participant = participant
            self.moments = moments
            self.meta = meta
            self.dataset_id = f"long_term|long_day|{participant}"
            self.participant_label = PARTICIPANT_NAMES.get(participant, participant)
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")
        steps = []

        # 白天：按小时分批
        by_hour = {}
        for m in moments:
            h = m.get("_hour") or (m["id"].split("_")[3][:2] if len(m["id"].split("_")) > 3 else "??")
            by_hour.setdefault(h, []).append(m)

        for h in sorted(by_hour):
            group = by_hour[h]
            def run(group=group, h=h, force=False):
                items, ok = [], 0
                cancelled = False
                for item_index, m in enumerate(group, 1):
                    context = {"item_index": item_index, "item_total": len(group),
                               "label": f"{h}:00 时段 · 逐条写入"}
                    self._set_step_progress(**context, stage="item_start",
                                            message=f"正在处理第 {item_index}/{len(group)} 条：{m.get('id', '')}")
                    mid, info = self._add_one(m, force=force, progress=context)
                    if mid:
                        ok += 1
                    items.append(info)
                    if info.get("cancelled") or (self._active_step_cancel and self._active_step_cancel.is_set()):
                        cancelled = True
                        break
                msg = f"{h}:00 时段：Qwen 感知 {ok}/{len(group)} 条并写入记忆"
                if cancelled:
                    msg = f"实时推理已停止；本时段已完成 {ok}/{len(group)} 条"
                return msg, {
                    "kind": "batch", "items": items, "n": len(group),
                    "ok": ok, "cancelled": cancelled,
                }
            steps.append({
                "kind": "add_batch",
                "title": f"白天感知写入 · {h}:00 时段（{len(group)} 条）",
                "inputs": [self._display_brief(m) for m in group],
                "run": run,
            })

        def run_merge_candidates(force=False):
            candidates = self.mem.find_merge_candidates(limit=100)
            result = {
                "kind": "operation_result", "operation": "merge_candidates",
                "title": "相似事件候选 · 逐条确认后才会合并",
                "candidates": candidates,
            }
            return f"发现 {len(candidates)} 组相似事件候选；没有自动合并", result
        steps.append({
            "kind": "merge_candidates",
            "title": "检查相似事件候选（需要逐条确认）",
            "detail": {"note": "候选依据：同源片段重叠≥80%，或120秒内Scene与Action合并文本相似度≥92%且共享人物/物品。画面与语音转写文本只供核对；扫描不改数据，逐条确认后才归并。"},
            "inputs": [],
            "run": run_merge_candidates,
        })

        steps.append(self._nightly_consolidation_step(
            title="夜间整理 · sort 归类、combine 合并、compress 压缩、update_profile 画像更新",
            note="按真实 Qwen 感知结果执行归类、同义项合并、L4/L5 摘要压缩和画像更新；"
                 "画像只写入稳定且有来源的属性，不读取 ground-truth。",
        ))

        return steps

    # ------------------------------------------------------------------
    # 执行与状态
    # ------------------------------------------------------------------

    def _full_snapshot(self):
        """提取各层实际内容（供前端展示完整记忆，不只是数值）。"""
        m = self.mem.memory
        l7 = m["layer7"]

        # 反向映射：记忆 moment_id → 原始数据集 moment，用于补充视频地址 / 时间点
        orig_by_id = {x.get("id"): x for x in getattr(self, "moments", [])}
        rev = {mid: did for did, mid in self.id_map.items()}

        def vid_of(mid):
            """该条记忆对应的视频（可播放地址 + 片段区间 + 拍摄时刻）。"""
            orig = orig_by_id.get(rev.get(mid))
            if not orig:
                return None
            b = moment_brief(orig)
            if not b.get("video_url"):
                return None
            return {"video": b["video"], "video_url": b["video_url"],
                    "clip_start": b["clip_start"], "clip_end": b["clip_end"],
                    "clock": b["clock"]}

        def fmt(x, with_stale=False):
            d = {"id": x.get("id", ""), "scene": x.get("scene", ""),
                 "user_action": x.get("user_action", ""),
                 "interaction_details": x.get("interaction_details", ""),
                 "content_summary": x.get("content_summary", ""),
                 "location": x.get("location", ""),
                 "activity": x.get("activity", ""),
                 "people": x.get("people", []),
                 "objects": x.get("objects", []),
                 "environments": x.get("environments", []),
                 "notes": x.get("notes", ""),
                 "person_observations": copy.deepcopy(x.get("person_observations", [])),
                 "temporal_facts": copy.deepcopy(x.get("temporal_facts", [])),
                 "source_time": copy.deepcopy(x.get("source_time", {})),
                 "highlight": x.get("highlighted", False),
                 "timestamp": x.get("timestamp", "")}
            if with_stale:
                d["stale"] = x.get("stale", False)
                d["merged_into"] = x.get("merged_into", "")
                d["merged_sources"] = copy.deepcopy(x.get("merged_sources", []))
                d["merge_history"] = copy.deepcopy(x.get("merge_history", [])[-5:])
            media = x.get("multimodal")
            if isinstance(media, dict):
                d["multimodal"] = copy.deepcopy(media)
                strip_audio_assets(d)
                for temporal_ref in media_refs(d["multimodal"], "visual"):
                    temporal_ref["url"] = "/api/memory_asset?p=" + quote(temporal_ref["asset"])
                ref = d["multimodal"].get("visual")
                if isinstance(ref, dict) and ref.get("asset"):
                    ref["url"] = "/api/memory_asset?p=" + quote(ref["asset"])
                object_visuals = d["multimodal"].get("object_visuals") or {}
                for visuals in (object_visuals, d["multimodal"].get("person_visuals") or {}):
                    for item in visuals.get("items") or []:
                        for frame_ref in item.get("frame_refs") or []:
                            if frame_ref.get("asset"):
                                frame_ref["url"] = "/api/memory_asset?p=" + quote(frame_ref["asset"])
                            if frame_ref.get("source_asset"):
                                frame_ref["source_url"] = "/api/memory_asset?p=" + quote(frame_ref["source_asset"])
                entity_visuals = d["multimodal"].get("entity_visuals") or {}
                entity_assets = entity_visuals.get("assets") or []
                for asset_ref in entity_assets:
                    if asset_ref.get("asset"):
                        asset_ref["url"] = "/api/memory_asset?p=" + quote(asset_ref["asset"])
                    if asset_ref.get("source_asset"):
                        asset_ref["source_url"] = "/api/memory_asset?p=" + quote(asset_ref["source_asset"])
                for entity_item in entity_visuals.get("items") or []:
                    entity_item["frame_refs"] = [
                        copy.deepcopy(entity_assets[index])
                        for index in entity_item.get("asset_indices") or []
                        if isinstance(index, int) and 0 <= index < len(entity_assets)
                    ]
            source = x.get("source_provenance")
            if isinstance(source, dict):
                d["source_provenance"] = copy.deepcopy(source)
                d["source_provenance"]["source_video_id"] = _portable_video_id(
                    d["source_provenance"].get("source_video_id"))
            v = vid_of(x.get("id", ""))
            if v:
                d["video"] = v
            return d

        # 索引：列出每个维度下的 tag 及其命中的 moment id（限每个维度 15 个 tag，每个 tag 5 个 id）
        indices = {}
        for dim in ("people", "locations", "time_nodes", "activity_events", "environments", "objects"):
            idx = l7.get(dim, {})
            indices[dim] = {tag: ids[:5] for tag, ids in list(idx.items())[:15]}

        layer7_moments = list(l7.get("moments", {}).values())
        merged_count = sum(bool(item.get("merged_into")) for item in layer7_moments)
        object_index_samples = {}
        object_ranked = sorted(l7.get("objects", {}).items(),
                               key=lambda item: (-len(item[1]), str(item[0])))[:8]
        for tag, mids in object_ranked:
            # One representative record per visible top key keeps snapshots compact.
            mid = next((candidate for candidate in reversed(mids)
                        if candidate in l7.get("moments", {})), None)
            if mid:
                object_index_samples[tag] = [fmt(l7["moments"][mid], True)]
        def asset_bytes_for(node):
            """Count unique stored visual assets referenced by one memory layer."""
            asset_paths = set()

            def collect(value):
                if isinstance(value, dict):
                    for field in ("asset", "source_asset"):
                        asset = value.get(field)
                        if isinstance(asset, str):
                            normalized = asset.replace(chr(92), "/").lstrip("/")
                            if normalized.startswith("assets/"):
                                asset_paths.add(normalized)
                    for child in value.values():
                        collect(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        collect(child)

            collect(node)
            total = 0
            asset_root = self.media_store.asset_root.resolve()
            for asset in asset_paths:
                path = (self.media_store.root / asset).resolve()
                try:
                    path.relative_to(asset_root)
                except ValueError:
                    continue
                if path.is_file():
                    total += path.stat().st_size
            return total

        layer_asset_bytes = {
            "layer1": asset_bytes_for(m.get("layer1", [])),
            "layer2": asset_bytes_for(m.get("layer2", [])),
            "layer3": asset_bytes_for(m.get("layer3", [])),
            "layer4": asset_bytes_for(m.get("layer4", {})),
            "layer5": asset_bytes_for(m.get("layer5", {})),
            "layer6": asset_bytes_for(m.get("layer6", {})),
            "layer7": asset_bytes_for(l7.get("moments", {})),
        }
        return {
            "layer1": [fmt(x) for x in m.get("layer1", [])[:30]],
            "layer2": [fmt(x) for x in m.get("layer2", [])[:30]],
            "layer3": [fmt(x) for x in m.get("layer3", [])[:30]],
            "layer4": m.get("layer4", {}),
            "layer5": m.get("layer5", {}),
            "layer6": m.get("layer6", {}),
            "layer7_moments": [fmt(x, True) for x in list(l7.get("moments", {}).values())[:80]],
            "object_index_samples": object_index_samples,
            "layer7_total": len(l7.get("moments", {})),
            "layer7_active_total": len(layer7_moments) - merged_count,
            "layer7_merged_total": merged_count,
            "indices": indices,
            "metadata": m.get("metadata", {}),
            "entity_relationships": entity_graph_summary(l7, limit=12),
            "multimodal_asset_bytes": self.media_store.size_bytes(),
            "layer_asset_bytes": layer_asset_bytes,
            "daytime": self.daytime.snapshot(),
        }

    def _log_snapshot(self, title, kind, msg, perception=None):
        snap = rs.snapshot(self.mem, title, action=kind)
        snap["full"] = self._full_snapshot()
        snap["layer7_active_moments"] = snap["full"]["layer7_active_total"]
        snap["layer7_merged_moments"] = snap["full"]["layer7_merged_total"]
        self.log.append({"kind": kind, "title": title, "msg": msg,
                         "snapshot": snap, "perception": perception})

    def run_step(self, force=False, automatic=False, continuous_kind=None, use_cache=True,
                 progress_id=None):
        """执行当前步；连续模式仅跨越相邻的同类步骤。"""
        self.auto_paused = False
        self.step_cancelled = False
        if self.pos >= len(self.steps):
            return "所有步骤已执行完毕"
        step = self.steps[self.pos]
        if automatic:
            expected_kind = continuous_kind or step["kind"]
            if step["kind"] != expected_kind:
                self.auto_paused = True
                return (f"连续运行已暂停：当前操作类型已从“{expected_kind}”切换为“{step['kind']}”。"
                        f"下一步“{step['title']}”未执行；可继续运行该类操作。")
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._active_step_cancel = cancel_event
            self._active_step_force = bool(force)
            if self._pending_cancel:
                cancel_event.set()
                self._pending_cancel = False
        self._active_step_no = self.pos + 1
        self._active_step_title = step["title"]
        step_item_total = len(step.get("inputs") or []) if step["kind"] == "add_batch" else 1
        self._set_step_progress(
            running=True, stage="starting", label=step["title"],
            step_index=self.pos + 1, step_total=len(self.steps),
            item_index=0, item_total=step_item_total, message="正在启动当前步骤",
            output=None, completed=[], run_id=progress_id or uuid.uuid4().hex,
        )
        try:
            if cancel_event.is_set():
                msg, perception = "当前步骤已终止；没有开始后续处理", {"cancelled": True}
            elif step["kind"] == "nightly":
                msg, perception = step["run"](force=force, use_cache=use_cache)
            else:
                msg, perception = step["run"](force=force)
        except Exception as exc:
            self._set_step_progress(running=False, stage="error", message=str(exc)[:300])
            raise
        finally:
            self._active_step_no = None
            self._active_step_title = None
            with self._cancel_lock:
                if self._active_step_cancel is cancel_event:
                    self._active_step_cancel = None
                    self._active_step_force = False
        self._set_step_progress(running=False,
                                stage="cancelled" if cancel_event.is_set() else "done",
                                message=msg, output=None)
        if cancel_event.is_set():
            self.step_cancelled = True
            if isinstance(perception, dict):
                perception["cancelled"] = True
            msg = "实时推理已停止；已完成的记忆会保留，当前步骤未标为完成，可再次运行。"
            self._log_snapshot(step["title"], step["kind"], msg, perception)
            return msg
        self._log_snapshot(step["title"], step["kind"], msg, perception)
        self.pos += 1
        return msg

    def has_accumulated_memory(self):
        """Return whether this session has any persisted scene memories."""
        memory = getattr(self.mem, "memory", {}) or {}
        if any(memory.get(f"layer{number}") for number in (1, 2, 3)):
            return True
        layer7 = memory.get("layer7") or {}
        moments = layer7.get("moments") if isinstance(layer7, dict) else None
        return bool(moments)

    def run_nightly_now(self, force=False, use_cache=True, progress_id=None):
        """Consolidate accumulated memories without advancing the video-step cursor."""
        if not self.has_accumulated_memory():
            raise ValueError("当前还没有可整理的记忆")
        self.step_cancelled = False
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._active_step_cancel = cancel_event
            self._active_step_force = bool(force)
            if self._pending_cancel:
                cancel_event.set()
                self._pending_cancel = False
        self._active_step_no = self.pos + 1
        self._active_step_title = "手动夜间整理归纳"
        self._set_step_progress(
            running=True, stage="starting", label="手动夜间整理归纳",
            step_index=min(self.pos + 1, len(self.steps)), step_total=len(self.steps),
            item_index=0, item_total=1, message="正在整理当前累积记忆",
            output=None, completed=[], run_id=progress_id or uuid.uuid4().hex,
        )
        try:
            if cancel_event.is_set():
                msg, result = "夜间整理已取消", {"kind": "nightly", "cancelled": True}
            else:
                step = self._nightly_consolidation_step(title="手动夜间整理归纳")
                msg, result = step["run"](
                    force=force, use_cache=use_cache, close_scene=False
                )
        except Exception as exc:
            self._set_step_progress(running=False, stage="error", message=str(exc)[:300])
            raise
        finally:
            self._active_step_no = None
            self._active_step_title = None
            with self._cancel_lock:
                if self._active_step_cancel is cancel_event:
                    self._active_step_cancel = None
                    self._active_step_force = False
        self._set_step_progress(
            running=False, stage="cancelled" if cancel_event.is_set() else "done",
            message=msg, output=None,
        )
        self._log_snapshot("手动夜间整理归纳", "nightly_manual", msg, result)
        self.step_cancelled = bool(cancel_event.is_set())
        return msg

    def prewarm(self, limit=None):
        """批量预热 Qwen 缓存（后台任务调用）。返回 (成功, 失败)。"""
        ms = self.moments[:limit] if limit else self.moments
        return self.perceiver.prewarm(ms)

    def state(self):
        """返回给前端的完整状态。"""
        total = len(self.steps)
        cur = None
        if self.pos < total:
            s = self.steps[self.pos]
            cur = {
                "kind": s["kind"],
                "title": s["title"],
                "detail": s.get("detail", {}),
                "inputs": s.get("inputs", []),
            }
        # 原始输出在前端折叠显示；这里保留完整文本，避免用户看到人为截断的字段。
        slim = []
        for e in self.log:
            e2 = dict(e)
            p = e.get("perception")
            if p:
                e2["perception"] = dict(p)
            slim.append(e2)

        meta = SCENARIO_META[self.key]
        cache_entries = getattr(self.perceiver, "cache", {}) or {}
        current_cache_count = sum(
            entry.get("prompt_version") == PROMPT_VERSION
            for entry in cache_entries.values() if isinstance(entry, dict)
        )
        extra = {
            "participant": self.participant,
            "participant_label": self.participant_label,
            "dataset_id": self.dataset_id,
            "dataset_key": self.dataset_key,
            "dataset_note": self.dataset_note,
            "dataset_options": self.dataset_options,
        }
        if self.key == "long_term":
            extra["hours_covered"] = self.meta.get("hours_covered", [])
        display_name = meta["name"]
        description = meta["desc"]
        if self.key == "short_term" and self.participant:
            display_name += f" · {self.participant_label}"
        elif self.key == "long_term":
            extra["available_count"] = self.meta.get("available_count", len(self.moments))
            extra["total_candidates"] = self.meta.get("total_candidates", len(self.moments))
            extra["coverage"] = self.meta.get("coverage", 1.0)
            if self.dataset_key.startswith("feishu:"):
                display_name += " · 飞书自定义测试集"
                extra["complete_day"] = False
                extra["custom_test_dataset"] = True
            else:
                display_name += f" · {self.participant_label}"
                if not self.meta.get("complete", False):
                    display_name += "（部分视频）"
                extra["complete_day"] = self.meta.get("complete", False)
            description = self.dataset_note
        elif self.key == "custom_test":
            display_name += f" · {self.participant_label}"
            extra["available_count"] = self.meta.get("available_count", len(self.moments))
            extra["total_candidates"] = self.meta.get("total_candidates", len(self.moments))
            extra["coverage"] = self.meta.get("coverage", 1.0)
            extra["custom_test_dataset"] = True
            extra["complete_day"] = False
            description = self.dataset_note
        return {
            "scenario": self.key,
            "name": display_name,
            "desc": description,
            "total": total,
            "pos": self.pos,
            "finished": self.pos >= total,
            "has_memory": self.has_accumulated_memory(),
            "current": cur,
            "auto_paused": self.auto_paused,
            "step_cancelled": self.step_cancelled,
            "nightly_progress": dict(self.nightly_progress) if self.nightly_progress else None,
            "step_progress": dict(self.step_progress) if self.step_progress else None,
            "log": slim,
            "operations": [dict(event) for event in self.operation_log[-2000:]],
            "snapshot": self.log[-1]["snapshot"] if self.log else None,
            "qwen": {
                "ok": bool(self.qwen_health.get("model_loaded")),
                "gpu": (self.qwen_health.get("gpu") or {}).get("name"),
                "cached": current_cache_count,
                "legacy_cached": max(0, len(cache_entries) - current_cache_count),
                "cache_prompt_version": PROMPT_VERSION,
            },
            **extra,
        }


if __name__ == "__main__":
    # 命令行自测：逐步执行并打印
    s = InteractiveSession("short_term")
    while not s.state()["finished"]:
        st = s.state()
        print(f"[{st['pos']+1}/{st['total']}] {st['current']['title']} ... ", end="")
        print(s.run_step())
    print("done")
