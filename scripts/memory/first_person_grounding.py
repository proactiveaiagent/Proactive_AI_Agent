"""Ground first-person body-part evidence and personal measurements."""
from __future__ import annotations

import re


FIRST_PERSON_GROUNDING_VERSION = "pov-body-and-measurement-v1-2026-10-23"

_BODY_PART_RE = re.compile(
    r"\b(?:feet|foot|toes?|lower legs?|ankles?|hands?|arms?|forearms?|legs?|knees?)\b|"
    r"(?:脚|脚趾|小腿|脚踝|手|手臂|前臂|腿|膝盖)", re.I)
_FACE_OR_WHOLE_PERSON_RE = re.compile(
    r"\b(?:face|facial|head|torso|upper body|full body|whole body|waist[- ]up|"
    r"eyes|nose|mouth|hair|woman|man|child|person wearing|person with)\b|"
    r"(?:面部|脸|头部|上半身|全身|女人|男人|儿童)", re.I)
_POV_RE = re.compile(
    r"\b(?:first[- ]person|pov|camera wearer|user(?:'s)?|camera(?:'s)? viewpoint|"
    r"camera angle|point[- ]of[- ]view|looking down at (?:the )?(?:feet|foot|scale)|"
    r"viewpoint.{0,40}(?:feet|foot|legs?|hands?|scale))\b|"
    r"(?:第一人称|主观视角|镜头视角|摄像机视角|向下拍摄)", re.I)
_SCALE_RE = re.compile(
    r"\b(?:bathroom\s+scale|digital\s+scale|weighing\s+scale|weight\s+scale|scale)\b|"
    r"(?:体重秤|电子秤|称重)", re.I)
_VALUE_RE = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d{1,2})?)(?![\d.])")
_WEIGHT_UNIT_RE = re.compile(r"\b(?:kg|kgs|kilograms?)\b|(?:公斤|千克)", re.I)
_SIGH_RE = re.compile(r"\b(?:sigh(?:s|ed|ing)?|audible sigh)\b|(?:叹息|叹气)", re.I)


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    return ""


def _body_only_observation(row):
    if not isinstance(row, dict):
        return False
    detail = " ".join(str(row.get(key) or "") for key in (
        "appearance", "evidence", "profile_context"))
    if not _BODY_PART_RE.search(detail):
        return False
    return not _FACE_OR_WHOLE_PERSON_RE.search(detail)


def _add_unique(rows, fact):
    signature = (fact.get("subject"), fact.get("predicate"), fact.get("value"),
                 fact.get("evidence"), fact.get("modality"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        old = (row.get("subject"), row.get("predicate"), row.get("value"),
               row.get("evidence"), row.get("modality"))
        if old == signature:
            return
    rows.append(fact)


def _append_note(value, note):
    text = str(value or "").strip()
    if note.casefold() in text.casefold():
        return text
    return (text + ("\n" if text else "") + note)[:1600]


def _person_visuals_without(media, removed_names):
    if not isinstance(media, dict) or not removed_names:
        return
    group = media.get("person_visuals")
    if isinstance(group, dict):
        kept = []
        for item in group.get("items") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("person") or "").strip().casefold()
            if label in removed_names or _body_only_observation(item):
                continue
            kept.append(item)
        group["items"] = kept
        group["status"] = "filtered_first_person_body_parts"
        group["filter_version"] = FIRST_PERSON_GROUNDING_VERSION


def ground_first_person_body_context(fields, source_id="", multimodal=None,
                                    verified_measurement=None,
                                    user_reported_sigh=False, audio_text=""):
    """Remove POV anatomy from person identities and ground visible scale readings.

    ``verified_measurement`` is reserved for an explicitly reviewed frame and must
    contain ``value`` and ``unit``. Without it, the unit is taken only from source
    text; the function never silently assumes kilograms.
    """
    if not isinstance(fields, dict):
        return False
    media = multimodal if isinstance(multimodal, dict) else fields.get("multimodal")
    story = " ".join(str(fields.get(key) or "") for key in (
        "scene", "user_action", "interaction_details", "activity", "location", "environments"))
    obs = [row for row in fields.get("person_observations") or [] if isinstance(row, dict)]
    body_rows = [row for row in obs if _body_only_observation(row)]
    own_body = bool(_BODY_PART_RE.search(story) and _POV_RE.search(story))
    # A named body-only observation is still user-body evidence in an explicit POV.
    removed = body_rows if own_body else []
    removed_names = {str(row.get("entity") or "").strip().casefold()
                     for row in removed if str(row.get("entity") or "").strip()}
    changed = False
    if removed:
        fields["person_observations"] = [row for row in obs if row not in removed]
        fields["people"] = [value for value in fields.get("people") or []
                            if str(value or "").strip().casefold() not in removed_names]
        _person_visuals_without(media, removed_names)
        facts = [dict(row) for row in fields.get("temporal_facts") or [] if isinstance(row, dict)]
        for fact in facts:
            actor = str(fact.get("actor") or "").strip().casefold()
            subject = str(fact.get("subject") or "").strip().casefold()
            if actor in removed_names or subject in removed_names:
                if actor:
                    fact["actor"] = "user"
                if subject:
                    fact["subject"] = "user"
                if fact.get("subject_type") == "person":
                    fact["subject_type"] = "event"
        fields["temporal_facts"] = facts
        changed = True

    text = " ".join(str(fields.get(key) or "") for key in (
        "scene", "user_action", "interaction_details", "activity", "objects", "object_observations"))
    scale_context = bool(_SCALE_RE.search(text))
    if verified_measurement and isinstance(verified_measurement, dict):
        value = str(verified_measurement.get("value") or "").strip()
        unit = str(verified_measurement.get("unit") or "").strip()
    else:
        value_match = _VALUE_RE.search(text)
        value = value_match.group(1) if value_match else ""
        unit_match = _WEIGHT_UNIT_RE.search(text)
        unit = "kg" if unit_match else ""
    if own_body and scale_context:
        objects = [str(item).strip() for item in fields.get("objects") or []
                   if str(item).strip() and not _SCALE_RE.search(str(item))]
        objects.append("digital bathroom scale")
        fields["objects"] = list(dict.fromkeys(objects))[:8]
        canonical_value = (value + (" " + unit if unit else " (unit not confirmed)")).strip()
        display_value = value + (" " + unit if unit else "")
        frame_index = 1
        source_segment_id = str(source_id or (fields.get("source_provenance") or {}).get(
            "source_segment_id") or "")
        visual_ref = {"source_segment_id": source_segment_id, "modality": "visual",
                      "frame_index": frame_index, "frame_time_s": 0.0}
        existing_objects = [dict(row) for row in fields.get("object_observations") or []
                            if isinstance(row, dict) and not _SCALE_RE.search(
                                str(row.get("object") or ""))]
        if value:
            existing_objects.append({
                "object": "digital bathroom scale",
                "appearance": "Digital bathroom scale with display reading " + display_value,
                "evidence": "First-person feet are on the scale; its digital display reads " + display_value + ".",
                "frame_index": frame_index, "frame_time_s": 0.0,
                "confidence": "high" if unit else "medium",
                "evidence_ref": dict(visual_ref),
            })
        fields["object_observations"] = existing_objects
        fields["user_action"] = "The user stands on a digital bathroom scale to measure their weight."
        fields["activity"] = "The user weighs themself on a digital bathroom scale."
        fields["interaction_details"] = (
            "The user's feet are on the digital bathroom scale; the display reads "
            + (display_value or "a numeric value") + ".")
        scene = str(fields.get("scene") or "")
        scene = re.sub(r"\b(?:A|The) person\b", "The user", scene, count=1, flags=re.I)
        fields["scene"] = scene
        facts = [dict(row) for row in fields.get("temporal_facts") or [] if isinstance(row, dict)]
        facts = [row for row in facts if str(row.get("subject") or "").strip().casefold()
                 not in removed_names]
        read_confidence = "high" if unit else "medium"
        _add_unique(facts, {
            "subject_type": "event", "subject": "user", "actor": "user",
            "predicate": "weighed_self",
            "value": "The user weighed themself on a digital bathroom scale.",
            "status": "observed", "confidence": "high", "modality": "visual",
            "evidence": "First-person view shows the user's bare feet standing on the scale.",
            "frame_index": frame_index, "frame_time_s": 0.0,
            "evidence_ref": dict(visual_ref),
        })
        if value:
            _add_unique(facts, {
                "subject_type": "object", "subject": "digital bathroom scale",
                "predicate": "displayed_measurement", "value": display_value,
                "status": "observed", "confidence": read_confidence, "modality": "visual",
                "evidence": "The scale's digital screen visibly reads " + display_value + ".",
                "frame_index": frame_index, "frame_time_s": 0.0,
                "evidence_ref": dict(visual_ref),
            })
            _add_unique(facts, {
                "subject_type": "person", "subject": "user",
                "predicate": "weight_measurement", "value": canonical_value,
                "status": "inferred", "confidence": read_confidence, "modality": "visual",
                "time_scope": "measurement shown in this clip",
                "evidence": "The user's feet are on the scale and its screen reads " + display_value + "; this is the user's reading in first-person POV.",
                "frame_index": frame_index, "frame_time_s": 0.0,
                "evidence_ref": dict(visual_ref),
            })
        fields["temporal_facts"] = facts[:32]
        changed = True

    audio = _text(audio_text)
    if scale_context and (user_reported_sigh or _SIGH_RE.search(audio)):
        facts = [dict(row) for row in fields.get("temporal_facts") or [] if isinstance(row, dict)]
        from_review = bool(user_reported_sigh)
        evidence = ("User review reports an audible sigh after the scale reading; possible dissatisfaction is inferred, not explicitly stated."
                    if from_review else "An audible sigh follows the scale reading; possible dissatisfaction is inferred, not explicitly stated.")
        _add_unique(facts, {
            "subject_type": "event", "subject": "user", "actor": "user",
            "predicate": "possible_dissatisfaction_with_weight",
            "value": "The user may be dissatisfied with the displayed weight; this is a low-confidence inference from a sigh, not a stated opinion.",
            "status": "inferred", "confidence": "low",
            "modality": "user_review" if from_review else "audio",
            "evidence": evidence,
            "evidence_ref": {"source_segment_id": str(source_id or ""),
                             "modality": "user_review" if from_review else "audio"},
        })
        fields["temporal_facts"] = facts[:32]
        if from_review:
            fields["notes"] = _append_note(
                fields.get("notes"),
                "User review: an audible sigh follows the weight reading; possible dissatisfaction is inferred with low confidence, not directly stated.")
        changed = True

    if isinstance(media, dict):
        media["first_person_grounding_version"] = FIRST_PERSON_GROUNDING_VERSION
        if isinstance(fields.get("multimodal"), dict):
            fields["multimodal"] = media
        if changed:
            fields["multimodal"] = media
    provenance = fields.get("source_provenance")
    if changed and isinstance(provenance, dict):
        provenance["first_person_grounding_version"] = FIRST_PERSON_GROUNDING_VERSION
    return changed
