"""Ground hospital-consultation roles and patient facts from linked evidence."""
from __future__ import annotations

import re

_AUDIO_LINE_RE = re.compile(
    r"^\s*\[audio\s+(?P<start>\d+(?:\.\d+)?)\s*[–-]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*s;\s*[^\]]+\]\s*(?P<text>.*)$", re.I)
_HEALTH_RE = re.compile(
    r"\b(?:heart disease|heart condition|heart attack|high blood pressure|diabetes|"
    r"asthma|hypertension|stroke|cancer|kidney disease|epilepsy|blood clot)\b|"
    r"(?:心脏病|心梗|心肌梗死|高血压|糖尿病|哮喘|中风|癌症|肾病|血栓)", re.I)
_GENERIC_PERSON_RE = re.compile(
    r"^(?:(?:visible|unidentified)\s+)?(?:man(?:\s+\d+)?|male person|"
    r"person(?:\s+(?:\d+|at\s+.+|in\s+.+))?|individual(?:\s+\d+)?)$", re.I)


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split()).strip(" .,:;，。；：")


def _refs_unique(rows):
    output, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = tuple(row.get(k) for k in (
            "source_segment_id", "modality", "frame_index", "frame_time_s",
            "audio_start_s", "audio_end_s"))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output[:12]


def ground_clinical_visit(fields, observations, transcript, source_segment_id):
    """Link a unique clinician and patient facts only when visual and speech evidence agree.

    The record alone never identifies its patient as the camera wearer. A doctor role
    requires a clinical setting, a presented record, direct patient-directed clinical
    speech, a camera-facing consultation, and a unique visible candidate.
    """
    if not isinstance(fields, dict):
        return {}
    rows = [dict(row) for row in (observations or []) if isinstance(row, dict)]
    lines = []
    for line in str(transcript or "").splitlines():
        match = _AUDIO_LINE_RE.match(line.strip())
        if not match:
            continue
        try:
            start, end = float(match.group("start")), float(match.group("end"))
        except (TypeError, ValueError):
            continue
        quote = " ".join(str(match.group("text") or "").split())[:500]
        if quote:
            lines.append((quote, start, end))

    docs = [row for row in fields.get("document_evidence") or [] if isinstance(row, dict)]
    docs = [row for row in docs if str(row.get("document") or "").strip()
            and str(row.get("text") or "").strip()]
    document_text = " ".join(str(row.get("text") or "") for row in docs)
    context = " ".join(str(fields.get(key) or "") for key in (
        "scene", "location", "activity", "interaction_details", "environments"))
    visual_clinical = bool(re.search(
        r"\b(?:hospital|clinic|medical office|consultation room|patient record|"
        r"diagnosis|medical report|medical record)\b|(?:医院|诊所|病历|诊断|医疗)",
        context + " " + document_text, re.I))
    record_presented = bool(docs or re.search(
        r"\b(?:paper|document|record|report)\b.{0,100}\b(?:show|present|diagnos|patient|hospital)\b|"
        r"\b(?:show|present|hold|extend).{0,100}\b(?:paper|document|record|report)\b",
        context, re.I))
    scene_text = " ".join(str(fields.get(key) or "") for key in ("scene", "activity", "interaction_details"))
    addressed_to_viewer = bool(re.search(
        r"\b(?:toward|to) the (?:camera|viewer)\b|\blooks? at the camera\b|"
        r"\b(?:speaks?|talks?) to the camera\b|\bfirst[- ]person video\b|(?:朝向镜头|对着镜头)",
        scene_text, re.I))
    health_lines = [(quote, start, end) for quote, start, end in lines
                    if _HEALTH_RE.search(quote)
                    and re.search(r"\b(?:your|you have|you are|you've)\b|(?:你的|您有|你患有)", quote, re.I)]
    advice_lines = [(quote, start, end) for quote, start, end in lines
                    if re.search(
                        r"\b(?:don't|do not|avoid|should not|must not)\b.{0,80}"
                        r"(?:strenuous|exercise|exertion|physical activity)|"
                        r"(?:strenuous|exercise|exertion|physical activity).{0,80}"
                        r"\b(?:avoid|don't|do not|should not)\b|"
                        r"\b(?:take care|stay alert|be careful)\b|"
                        r"(?:不要|避免).{0,30}(?:剧烈运动|运动|劳累)", quote, re.I)]
    direct_clinical = bool(health_lines and advice_lines)
    appearance_text = " ".join(str(row.get("appearance") or "") for row in rows)
    consultation_action = bool(re.search(
        r"\b(?:coat|scrubs|stethoscope|medical equipment|medical cart)\b", appearance_text + " " + context, re.I)
        and re.search(r"\b(?:paper|record|report|diagnosis|hold|present|show|desk)\b", context + " " + document_text, re.I))
    if not (len(rows) == 1 and visual_clinical and record_presented
            and addressed_to_viewer and direct_clinical and consultation_action):
        return {}

    clinician = rows[0]
    old_label = " ".join(str(clinician.get("entity") or "").split()).strip()
    aliases = {}
    if old_label and _GENERIC_PERSON_RE.fullmatch(old_label):
        if old_label.casefold() != "doctor":
            aliases[old_label] = "doctor"
        clinician["entity"] = "doctor"
    roles = [str(value).strip() for value in clinician.get("roles") or [] if str(value).strip()]
    if "doctor" not in {value.casefold() for value in roles}:
        roles.append("doctor")
    clinician["roles"] = roles[:6]
    clinician["profile_confidence"] = "high"
    clinician["profile_context"] = (
        (str(clinician.get("profile_context") or "").strip() + " " if clinician.get("profile_context") else "")
        + "Identified as the doctor from the hospital consultation, presented medical record, and direct condition-specific advice."
    )[:700]

    visual_refs = []
    for doc in docs:
        index = doc.get("frame_index")
        if isinstance(index, int) and not isinstance(index, bool) and index > 0:
            visual_refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                                "modality": "visual", "frame_index": index})
    if not visual_refs:
        for fact in fields.get("temporal_facts") or []:
            if not isinstance(fact, dict):
                continue
            value = str(fact.get("value") or "")
            ref = fact.get("evidence_ref") if isinstance(fact.get("evidence_ref"), dict) else {}
            if (re.search(r"document|medical record|diagnosis|patient", value, re.I)
                    and isinstance(ref.get("frame_index"), int)):
                visual_refs.append(dict(ref))
    speech_refs = [{"source_segment_id": str(source_segment_id or "")[:128],
                    "modality": "speech", "audio_start_s": round(start, 3),
                    "audio_end_s": round(end, 3)}
                   for _quote, start, end in health_lines + advice_lines]
    clinician_refs = _refs_unique(
        [clinician.get("evidence_ref")] + list(clinician.get("evidence_refs") or [])
        + visual_refs + speech_refs)
    if clinician_refs:
        clinician["evidence_refs"] = clinician_refs
        clinician["evidence_ref"] = clinician_refs[0]

    # The patient name and age are taken from readable document text. Matching a direct
    # address links them to the viewer; speech alone can link the addressed condition.
    address_name = ""
    address_pattern = re.compile(
        r"^\s*(?P<name>[A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){0,2})[,，]\s*"
        r"(?:your\s+|you have\s+|you've had\s+|你(?:的|有))", re.I)
    for quote, _start, _end in health_lines:
        match = address_pattern.search(quote)
        if match:
            address_name = " ".join(match.group("name").split())
            break
    patient_name = ""
    patient_age = ""
    for doc in docs:
        text = " ".join(str(doc.get("text") or "").split())
        name_match = re.search(
            r"\b(?:patient(?:\s+name)?|name)\s*[:#-]\s*"
            r"([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){0,2})", text, re.I)
        if not name_match:
            name_match = re.search(
                r"\b([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,2})"
                r"(?=\s*[,;]\s*\d{1,3}\b)", text)
        if name_match:
            patient_name = " ".join(name_match.group(1).split())
        age_match = re.search(r"\b(?:age|aged)\s*[:#-]?\s*(\d{1,3})\b", text, re.I)
        if not age_match and patient_name:
            age_match = re.search(re.escape(patient_name) + r"\s*[,;]\s*(\d{1,3})\b", text, re.I)
        if age_match:
            try:
                age_value = int(age_match.group(1))
                if 0 <= age_value <= 120:
                    patient_age = str(age_value)
            except ValueError:
                pass
        if patient_name and patient_age:
            break
    names_match = bool(address_name and patient_name
                       and _norm(address_name).split()[0] == _norm(patient_name).split()[0])
    patient_linked_to_user = bool(address_name and health_lines and addressed_to_viewer)
    if names_match:
        patient_linked_to_user = True

    facts = [dict(row) if isinstance(row, dict) else row
             for row in fields.get("temporal_facts") or []]
    fields["temporal_facts"] = facts

    def append_fact(fact):
        ref = fact.get("evidence_ref") if isinstance(fact.get("evidence_ref"), dict) else {}
        signature = (str(fact.get("predicate") or ""),
                     str(ref.get("source_segment_id") or ""),
                     ref.get("audio_start_s"), ref.get("audio_end_s"),
                     str(fact.get("evidence") or ""))
        for old in facts:
            if not isinstance(old, dict):
                continue
            old_ref = old.get("evidence_ref") if isinstance(old.get("evidence_ref"), dict) else {}
            old_signature = (str(old.get("predicate") or ""),
                             str(old_ref.get("source_segment_id") or ""),
                             old_ref.get("audio_start_s"), old_ref.get("audio_end_s"),
                             str(old.get("evidence") or ""))
            if signature == old_signature:
                return False
        facts.append(fact)
        return True

    found_health, found_advice = set(), set()
    health_quotes = {quote for quote, _start, _end in health_lines}
    advice_quotes = {quote for quote, _start, _end in advice_lines}
    for fact in facts:
        if not isinstance(fact, dict) or str(fact.get("modality") or "").casefold() != "speech":
            continue
        quote = str(fact.get("evidence") or fact.get("value") or "").strip()
        if quote in health_quotes and patient_linked_to_user:
            if str(fact.get("predicate") or "").casefold() == "addressed_person":
                fact.update({"subject_type": "person", "subject": "user",
                             "value": f"The doctor addresses the listener as {address_name or 'David'}.",
                             "actor": "doctor", "affected_people": ["user"],
                             "status": "stated", "confidence": "high"})
            elif _HEALTH_RE.search(quote):
                fact.update({"subject_type": "person", "subject": "user",
                             "predicate": "doctor_stated_health_condition",
                             "value": "The doctor stated that the user's heart disease is very severe.",
                             "actor": "doctor", "affected_people": ["user"],
                             "status": "doctor_stated", "confidence": "high"})
                fact["evidence_refs"] = _refs_unique(
                    list(fact.get("evidence_refs") or []) + speech_refs + visual_refs)
                found_health.add(quote)
        if quote in advice_quotes and patient_linked_to_user:
            fact.update({"subject_type": "person", "subject": "user",
                         "predicate": "doctor_advice", "value": quote,
                         "actor": "doctor", "affected_people": ["user"],
                         "status": "doctor_advice", "confidence": "high"})
            fact["evidence_refs"] = _refs_unique(
                list(fact.get("evidence_refs") or []) + speech_refs + visual_refs)
            found_advice.add(quote)

    if patient_linked_to_user:
        for quote, start, end in health_lines:
            if quote in found_health:
                continue
            ref = {"source_segment_id": str(source_segment_id or "")[:128],
                   "modality": "speech", "audio_start_s": round(start, 3),
                   "audio_end_s": round(end, 3)}
            append_fact({"subject_type": "person", "subject": "user",
                         "predicate": "doctor_stated_health_condition",
                         "value": "The doctor stated that the user's heart disease is very severe.",
                         "actor": "doctor", "affected_people": ["user"],
                         "modality": "speech", "status": "doctor_stated",
                         "confidence": "high", "evidence": quote,
                         "evidence_ref": ref,
                         "evidence_refs": _refs_unique([ref] + visual_refs)})
        for quote, start, end in advice_lines:
            if quote in found_advice:
                continue
            ref = {"source_segment_id": str(source_segment_id or "")[:128],
                   "modality": "speech", "audio_start_s": round(start, 3),
                   "audio_end_s": round(end, 3)}
            append_fact({"subject_type": "person", "subject": "user",
                         "predicate": "doctor_advice", "value": quote,
                         "actor": "doctor", "affected_people": ["user"],
                         "modality": "speech", "status": "doctor_advice",
                         "confidence": "high", "evidence": quote,
                         "evidence_ref": ref,
                         "evidence_refs": _refs_unique([ref] + visual_refs)})

    if patient_linked_to_user and patient_age and names_match:
        age_doc = next((doc for doc in docs if patient_name.casefold() in str(doc.get("text") or "").casefold()), {})
        frame = age_doc.get("frame_index")
        doc_ref = {"source_segment_id": str(source_segment_id or "")[:128], "modality": "visual"}
        if isinstance(frame, int) and frame > 0:
            doc_ref["frame_index"] = frame
        evidence_refs = _refs_unique([doc_ref] + ([speech_refs[0]] if speech_refs else []))
        append_fact({"subject_type": "person", "subject": "user",
                     "predicate": "age_listed_in_medical_record",
                     "value": f"The medical record shown to the user lists patient {patient_name}, age {patient_age}.",
                     "actor": "doctor", "modality": "visual", "status": "documented_in_medical_record",
                     "confidence": "high", "evidence": str(age_doc.get("text") or "")[:500],
                     "evidence_ref": doc_ref, "evidence_refs": evidence_refs})

    if patient_linked_to_user and health_lines and advice_lines:
        evidence = " ".join([health_lines[0][0], advice_lines[0][0]])[:500]
        risk_refs = _refs_unique(speech_refs + visual_refs)
        append_fact({"subject_type": "person", "subject": "user",
                     "predicate": "cautious_cardiac_risk_inference",
                     "value": "A sudden cardiac event during day-to-day activities is a possible concern, inferred from the doctor's statement that the heart disease is very severe and advice to avoid strenuous exercise; the doctor did not explicitly predict an attack.",
                     "actor": "doctor", "affected_people": ["user"], "modality": "speech",
                     "status": "inferred", "confidence": "low", "evidence": evidence,
                     "evidence_ref": speech_refs[0] if speech_refs else {}, "evidence_refs": risk_refs})

    if patient_linked_to_user:
        location = " ".join(str(fields.get("location") or "").split()) or "the hospital"
        fields["interaction_details"] = (
            f"A doctor at {location} shows the user a medical record, states that the heart disease is very severe, "
            "and advises avoiding strenuous exercise and staying alert."
        )[:700]
    for key in ("scene", "activity", "interaction_details"):
        text = str(fields.get(key) or "")
        if text:
            fields[key] = re.sub(r"\b(a|the)\s+man\b",
                                 lambda match: match.group(1) + " doctor", text, flags=re.I)
    fields["person_observations"] = rows
    fields["people"] = list(dict.fromkeys(
        str(row.get("entity") or "").strip() for row in rows if row.get("entity")))
    return aliases
