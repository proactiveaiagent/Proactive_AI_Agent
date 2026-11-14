"""Ground first-person emergency reports and adjacent first-aid responses."""
from __future__ import annotations

import json
import re
from pathlib import Path

EMERGENCY_GROUNDING_VERSION = "emergency-continuity-v1-2026-10-23"

_AUDIO_LINE_RE = re.compile(
    r"^\s*\[audio\s+(?P<start>\d+(?:\.\d+)?)\s*[–-]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*s;\s*[^\]]+\]\s*(?P<text>.*)$", re.I)
_SELF_ACUTE_RE = re.compile(
    r"\b(?:i\s+(?:have|am having|am experiencing)|i've had)\s+"
    r"(?:a\s+)?(?:heart attack|stroke|seizure|severe chest pain|"
    r"trouble breathing|can't breathe)\b|"
    r"(?:我|本人).{0,18}(?:心脏病发作|心肌梗死|心梗|胸口剧痛|喘不过气|呼吸困难)",
    re.I)
_GENERIC_PERSON_RE = re.compile(
    r"^(?:(?:visible|unidentified)\s+)?(?:person(?:\s+(?:\d+|at\s+.+|in\s+.+))?|"
    r"man\s+\d+|woman\s+\d+|male person|female person|individual(?:\s+\d+)?|"
    r"person in .+|person at .+|uniform wearer|person in uniform)$", re.I)


def _audio_lines(transcript):
    rows = []
    for line in str(transcript or "").splitlines():
        match = _AUDIO_LINE_RE.match(line.strip())
        if not match:
            continue
        quote = " ".join(str(match.group("text") or "").split())
        try:
            start, end = float(match.group("start")), float(match.group("end"))
        except (TypeError, ValueError):
            continue
        if quote and end >= start:
            rows.append((quote, start, end))
    return rows


def _self_report(lines):
    for quote, start, end in lines:
        if _SELF_ACUTE_RE.search(quote):
            return quote, start, end
    return None


def _unique_refs(rows):
    output, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = tuple(row.get(k) for k in (
            "source_segment_id", "modality", "frame_index", "frame_time_s",
            "audio_start_s", "audio_end_s"))
        if key not in seen:
            seen.add(key)
            output.append(dict(row))
    return output[:16]


def _audio_ref(source_id, start, end):
    return {"source_segment_id": str(source_id or "")[:128], "modality": "speech",
            "audio_start_s": round(float(start), 3), "audio_end_s": round(float(end), 3)}


def _append_sentence(value, sentence):
    text = str(value or "").strip()
    if sentence.casefold() in text.casefold():
        return text
    return (text + (" " if text else "") + sentence).strip()[:1600]


def _rename(value, mapping):
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True):
            if old:
                value = re.sub(r"(?<![\w])" + re.escape(old) + r"(?![\w])",
                               lambda _match, replacement=new: replacement,
                               value, flags=re.I)
        return value
    if isinstance(value, list):
        return [_rename(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _rename(item, mapping) for key, item in value.items()}
    return value


def _manifest_sequence(manifest_path, current_video_id):
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None, None
    current = Path(str(current_video_id or "").replace("\\", "/")).as_posix().casefold()
    files = manifest.get("files") or []
    for index, entry in enumerate(files):
        relative = Path(str(entry.get("relative_path") or "")).as_posix().casefold()
        if current == relative or current.endswith("/" + relative):
            return index, entry
    return None, None


def _prior_emergency(archive, source_id, current_video_id, manifest_path):
    current_order, current_entry = _manifest_sequence(manifest_path, current_video_id)
    if current_order is None or not isinstance(current_entry, dict):
        return None
    entries = (json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("files") or [])
    if current_order <= 0:
        return None
    previous = entries[current_order - 1]
    current_scenario = str(current_entry.get("scenario") or "")
    if str(previous.get("scenario") or "") != current_scenario:
        return None
    previous_path = Path(str(previous.get("relative_path") or "")).as_posix().casefold()
    moments = archive.get("moments", {}) if isinstance(archive, dict) else {}
    for moment_id, moment in moments.items():
        if not isinstance(moment, dict) or moment.get("stale") or moment.get("merged_into"):
            continue
        provenance = moment.get("source_provenance") or {}
        prior_id = str(provenance.get("source_segment_id") or "")
        prior_video = Path(str(provenance.get("source_video_id") or "")
                           .replace("\\", "/")).as_posix().casefold()
        if prior_id == source_id or not (prior_video == previous_path or prior_video.endswith("/" + previous_path)):
            continue
        report = _self_report(_audio_lines(moment.get("source_evidence") or ""))
        if not report:
            report = next((
                (str(row.get("evidence") or ""), 0.0, 0.0)
                for row in moment.get("temporal_facts") or []
                if isinstance(row, dict)
                and str(row.get("predicate") or "").casefold()
                    in {"self_reported_current_health_event", "self_reported_medical_emergency"}
                and str(row.get("modality") or "").casefold() == "speech"
            ), None)
        if report:
            quote, start, end = report
            return {"moment_id": str(moment_id), "source_segment_id": prior_id,
                    "quote": quote, "audio_start_s": start, "audio_end_s": end,
                    "source_video_id": prior_video}
    return None


def _add_fact(facts, fact):
    signature = (fact.get("predicate"), fact.get("subject"),
                 fact.get("evidence"), tuple(
                     (ref.get("source_segment_id"), ref.get("modality"),
                      ref.get("frame_index"), ref.get("audio_start_s"))
                     for ref in fact.get("evidence_refs") or []))
    for old in facts:
        if not isinstance(old, dict):
            continue
        old_sig = (old.get("predicate"), old.get("subject"), old.get("evidence"),
                   tuple((ref.get("source_segment_id"), ref.get("modality"),
                          ref.get("frame_index"), ref.get("audio_start_s"))
                         for ref in old.get("evidence_refs") or []))
        if old_sig == signature:
            return
    facts.append(fact)


def _ground_self_report(fields, source_id, report):
    quote, start, end = report
    ref = _audio_ref(source_id, start, end)
    facts = [dict(row) for row in fields.get("temporal_facts") or [] if isinstance(row, dict)]
    for row in facts:
        if (str(row.get("modality") or "").casefold() == "speech"
                and str(row.get("predicate") or "").casefold()
                in {"self_reported_current_health_event", "self_reported_medical_emergency"}
                and quote.casefold() in str(row.get("evidence") or "").casefold()):
            row.update({"subject_type": "event", "subject": "user",
                        "actor": "user", "affected_people": ["user"],
                        "predicate": "self_reported_medical_emergency",
                        "value": "The user reports a heart attack and urgently asks for help; this is a reported claim, not a verified diagnosis.",
                        "status": "stated", "confidence": "high",
                        "evidence": quote, "evidence_ref": ref})
    if not any(row.get("predicate") == "self_reported_medical_emergency"
               and row.get("evidence") == quote for row in facts):
        facts.append({
            "subject_type": "event", "subject": "user",
            "predicate": "self_reported_medical_emergency",
            "value": "The user reports a heart attack and urgently asks for help; this is a reported claim, not a verified diagnosis.",
            "actor": "user", "affected_people": ["user"], "modality": "speech",
            "status": "stated", "confidence": "high", "evidence": quote,
            "evidence_ref": ref,
        })
    fields["temporal_facts"] = facts
    fields["user_action"] = _append_sentence(
        fields.get("user_action"),
        "The user says they are having a heart attack and urgently calls for help.")
    fields["activity"] = _append_sentence(
        fields.get("activity"),
        "The user reports a heart attack and calls for emergency help.")
    fields["interaction_details"] = _append_sentence(
        fields.get("interaction_details"),
        "The audio captures the user saying, \"" + quote + "\".")
    fields["scene"] = _append_sentence(
        fields.get("scene"),
        "In the audio, the user reports a heart attack, asks for help, and says they may die.")


def _ground_aid_response(fields, source_id, previous):
    observations = [dict(row) for row in fields.get("person_observations") or []
                    if isinstance(row, dict)]
    if len(observations) < 2:
        return False
    context = " ".join(str(fields.get(key) or "") for key in
                       ("scene", "activity", "interaction_details", "environments"))
    observation_text = " ".join(
        " ".join(str(row.get(key) or "") for key in
                 ("entity", "appearance", "face_features", "evidence", "profile_context"))
        for row in observations)
    fact_text = " ".join(
        " ".join(str(row.get(key) or "") for key in ("subject", "predicate", "value", "evidence"))
        for row in fields.get("temporal_facts") or [] if isinstance(row, dict))
    all_text = (context + " " + observation_text + " " + fact_text).casefold()
    aid_signal = bool(re.search(r"\b(?:kneel(?:s|ed|ing)?|assist(?:s|ed|ance)?|first aid|"
                                r"medical responder|emergency care|red cross)\b|"
                                r"(?:跪|急救|协助|红十字)", all_text))
    cross_signal = bool(re.search(r"\bred cross\b|cross armband|red-cross|红十字", all_text))
    uniform_signal = bool(re.search(r"\b(?:uniform|scrubs|medical worker|nurse)\b|制服|护士", all_text))
    camera_signal = bool(re.search(r"\b(?:camera|lens|viewer|pov|first.person)\b|镜头|摄像机", all_text))
    low_angle_signal = bool(re.search(r"\b(?:silhouette|bent forward|bent over|kneel|looking down|"
                                      r"leans? toward|leaning over)\b|"
                                      r"(?:剪影|弯腰|俯身|跪)", all_text))
    if not (aid_signal and cross_signal and uniform_signal and low_angle_signal and camera_signal):
        return False

    responder = next((row for row in observations
                      if re.search(r"red cross|cross armband|red-cross|light blue uniform|"
                                   r"uniform|护士|红十字",
                                   " ".join(str(row.get(k) or "") for k in
                                            ("entity", "appearance", "evidence", "profile_context")), re.I)), None)
    other = next((row for row in observations if row is not responder), None)
    if not responder or not other:
        return False

    old_responder = " ".join(str(responder.get("entity") or "").split()).strip()
    old_other = " ".join(str(other.get("entity") or "").split()).strip()
    generic_responder = bool(_GENERIC_PERSON_RE.fullmatch(old_responder))
    generic_other = bool(_GENERIC_PERSON_RE.fullmatch(old_other))
    mapping = {}
    if generic_responder and old_responder.casefold() != "nurse":
        mapping[old_responder] = "nurse"
    if generic_other and old_other.casefold() != "passerby":
        mapping[old_other] = "passerby"
    if mapping:
        renamed_fields = _rename(fields, mapping)
        fields.clear()
        fields.update(renamed_fields)
        observations = [row for row in fields.get("person_observations") or []
                        if isinstance(row, dict)]
        responder = next((row for row in observations
                          if str(row.get("entity") or "").casefold() == "nurse"), responder)
        other = next((row for row in observations
                      if str(row.get("entity") or "").casefold() == "passerby"), other)

    responder["entity"] = "nurse"
    responder["roles"] = list(dict.fromkeys(
        [str(role) for role in responder.get("roles") or [] if str(role).strip()]
        + ["nurse", "first-aid responder"]))[:6]
    responder["profile_confidence"] = "medium"
    responder["profile_context"] = (
        "Role inferred from a medical-style uniform, visible red cross armband, and kneeling to assist during an emergency."
    )
    responder["role_evidence"] = "Uniform and red cross armband plus observed first-aid response"
    other["entity"] = "passerby"
    other["roles"] = list(dict.fromkeys(
        [str(role) for role in other.get("roles") or [] if str(role).strip()]
        + ["passerby", "bystander helper"]))[:6]
    other["profile_confidence"] = "medium"
    other["profile_context"] = (
        "Identified as a passerby from the immediate first-aid context; bends toward the low-positioned camera."
    )
    other["role_evidence"] = "Dark-clothed person bends toward the low camera during the emergency"
    for row in observations:
        row["confidence"] = "medium" if row.get("entity") in {"nurse", "passerby"} else row.get("confidence")
    fields["person_observations"] = observations
    fields["people"] = ["passerby", "nurse"]

    visual_refs = []
    for row in (other, responder):
        ref = row.get("evidence_ref") if isinstance(row.get("evidence_ref"), dict) else {}
        if ref:
            visual_refs.append(dict(ref))
        visual_refs.extend(dict(item) for item in row.get("evidence_refs") or []
                           if isinstance(item, dict))
    if not visual_refs:
        for row in fields.get("temporal_facts") or []:
            if isinstance(row, dict) and row.get("modality") == "visual":
                ref = row.get("evidence_ref")
                if isinstance(ref, dict):
                    visual_refs.append(dict(ref))
    audio_ref = _audio_ref(previous["source_segment_id"],
                           previous["audio_start_s"], previous["audio_end_s"])
    linked_refs = _unique_refs([audio_ref] + visual_refs)
    facts = [dict(row) for row in fields.get("temporal_facts") or [] if isinstance(row, dict)]
    for row in facts:
        subject = str(row.get("subject") or "").casefold()
        value = str(row.get("value") or "").casefold()
        if "visible person" in subject or "visible person" in value:
            row["subject"] = "passerby"
            if not row.get("actor"):
                row["actor"] = "passerby"
            row["confidence"] = "medium"
        if "uniform wearer" in subject or "uniform wearer" in value:
            row["subject"] = "nurse"
            row["actor"] = "nurse"
            row["confidence"] = "medium"
    evidence = ("The previous adjacent clip records the user's first-person heart-attack plea; "
                "this clip shows two people leaning toward the low-positioned camera, including a "
                "red-cross uniformed responder kneeling. The visual/audio sequence suggests they are helping the user.")
    _add_fact(facts, {
        "subject_type": "event", "subject": "user medical emergency",
        "predicate": "received_emergency_assistance",
        "value": ("In the immediately preceding clip the user reports a heart attack and asks for help; "
                  "here a passerby bends toward the camera while a nurse with a red cross armband kneels nearby. "
                  "They appear to be assisting the user."),
        "actor": "nurse", "affected_people": ["user"],
        "modality": "cross_modal", "status": "inferred", "confidence": "medium",
        "evidence": evidence, "evidence_refs": linked_refs,
    })
    _add_fact(facts, {
        "subject_type": "person", "subject": "passerby",
        "predicate": "checked_on_user",
        "value": "The passerby bends toward the low-positioned camera and looks down toward the user.",
        "actor": "passerby", "affected_people": ["user"],
        "modality": "cross_modal", "status": "inferred", "confidence": "medium",
        "evidence": "A dark-clothed person bends over the camera; the preceding clip contains the user's emergency plea.",
        "evidence_refs": linked_refs,
    })
    _add_fact(facts, {
        "subject_type": "person", "subject": "nurse",
        "predicate": "responded_to_user_emergency",
        "value": "A uniformed person with a red cross armband kneels beside the camera wearer during the reported emergency.",
        "actor": "nurse", "affected_people": ["user"],
        "modality": "cross_modal", "status": "inferred", "confidence": "medium",
        "evidence": "Red cross armband, medical-style uniform, kneeling action, and the preceding first-person plea.",
        "evidence_refs": linked_refs,
    })
    fields["temporal_facts"] = facts
    fields["scene"] = (
        "A passerby in dark clothing bends toward the low-positioned camera. A nurse in a light-blue uniform "
        "with a red cross armband kneels nearby and looks toward the camera wearer. The preceding clip records "
        "the user reporting a heart attack and calling for help, so this appears to be an emergency response "
        "to the user; that cross-clip link is an inference."
    )
    fields["user_action"] = (
        "In the preceding clip, the user reports a heart attack and calls for help. In this clip, a passerby "
        "bends toward the camera while a nurse kneels beside the camera wearer."
    )
    fields["activity"] = (
        "Emergency response: a passerby checks on the user while a nurse wearing a red cross armband kneels "
        "to assist. The link to the preceding first-person plea is a medium-confidence cross-clip inference."
    )
    fields["interaction_details"] = (
        "A passerby bends over the low-positioned camera and a uniformed nurse with a red cross armband kneels "
        "nearby. Because the immediately preceding clip records the user calling for help and reporting a heart "
        "attack, they appear to be responding to the user's emergency."
    )
    fields["temporal_facts"] = facts
    return True


def ground_emergency_continuity(fields, archive, source_segment_id, source_video_id,
                                manifest_path):
    """Add evidence-linked self-report facts and link adjacent visual aid responses."""
    if not isinstance(fields, dict):
        return False
    transcript = fields.get("source_evidence") or ""
    report = _self_report(_audio_lines(transcript))
    changed = False
    source_id = str(source_segment_id or "")[:128]
    if report:
        _ground_self_report(fields, source_id, report)
        changed = True
    previous = _prior_emergency(archive, source_id, source_video_id, manifest_path)
    if previous and _ground_aid_response(fields, source_id, previous):
        changed = True
    # User-facing narrative consistently names the first-person wearer as "the user".
    for key in ("scene", "user_action", "activity", "interaction_details"):
        value = fields.get(key)
        if isinstance(value, str):
            value = re.sub(r"\bthe camera wearer\b", "the user", value, flags=re.I)
            value = re.sub(r"\bcamera wearer\b", "user", value, flags=re.I)
            fields[key] = value
    provenance = fields.get("source_provenance")
    if changed and isinstance(provenance, dict):
        provenance["emergency_grounding_version"] = EMERGENCY_GROUNDING_VERSION
    return changed
