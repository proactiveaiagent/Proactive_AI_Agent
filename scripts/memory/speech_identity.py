"""Attach spoken forms of address to visible people without inventing identities.

The perception cache describes people by appearance ("headscarf wearer"), while the
audio often says who they are ("这是姥姥"). Whisper gives no speaker diarization, so a
spoken term may only rename a visible person when the clip itself singles that person
out; otherwise the term stays a speech fact and the person keeps the visual label.
"""
from __future__ import annotations

import re

MAX_ALIASES_PER_PERSON = 4
MAX_RELATIONSHIPS_PER_PERSON = 6
_USAGE_RANK = {"introduction": 0, "self": 1, "address": 2}

_CANONICAL_TERMS = (
    "姥姥", "外婆", "奶奶", "爷爷", "外公", "婆婆", "公公",
    "妈妈", "母亲", "爸爸", "父亲", "叔叔", "伯伯", "阿姨", "婶婶", "舅舅", "姑姑",
    "哥哥", "姐姐", "弟弟", "妹妹", "孙子", "孙女", "外孙", "儿子", "女儿",
    "丈夫", "妻子", "老公", "老婆", "邻居", "朋友", "同事", "老师", "医生",
    "护士", "服务员", "收银员", "顾客",
)
_TERM_ALIASES = {
    "grandmother": "grandma", "grandma": "grandma", "grandfather": "grandpa",
    "grandpa": "grandpa", "mother": "mom", "mom": "mom", "mum": "mom",
    "father": "dad", "dad": "dad", "aunt": "aunt", "uncle": "uncle",
    "sister": "sister", "brother": "brother", "wife": "wife", "husband": "husband",
    "friend": "friend", "neighbour": "neighbor", "neighbor": "neighbor",
    "colleague": "coworker", "coworker": "coworker", "teacher": "teacher",
    "doctor": "doctor", "nurse": "nurse", "waiter": "server", "server": "server",
    "cashier": "cashier", "customer": "customer", "grandson": "grandson",
    "granddaughter": "granddaughter", "son": "son", "daughter": "daughter",
}
# Longer Chinese terms first so 姥姥 wins over 姥 and 外婆 over 外.
_CHINESE_TERM_RE = re.compile("|".join(sorted(
    (re.escape(term) for term in _CANONICAL_TERMS), key=len, reverse=True)))
_ENGLISH_TERMS = tuple(_TERM_ALIASES)
_ENGLISH_TERM_RE = re.compile(
    r"\b(" + "|".join(sorted(_ENGLISH_TERMS, key=len, reverse=True)) + r")\b", re.I)
_CAMERA_FACING_RE = re.compile(
    r"\b(?:toward(?:s)? (?:the )?camera|at the camera|facing (?:the )?camera|"
    r"approach(?:es|ing)? (?:the )?camera|walk(?:s|ing)? toward(?:s)? (?:the )?camera|"
    r"looks? at the camera|smil\w+ at the camera)\b", re.I)
_APPROACH_RE = re.compile(r"\b(?:approach(?:es|ed|ing)?|reach(?:es|ed|ing)?|walk(?:s|ed|ing)? toward|come up to|came up to|go(?:es|ing)? up to|went up to|step(?:s|ped|ping)? toward|move(?:s|d|ing)? toward)\b", re.I)
_SPEAKER_IS_CAMERA_WEARER = re.compile(r"^\s*(?:camera wearer|the camera wearer)\b", re.I)
_INTRODUCE_RE = re.compile(r"(?:这是|这就是|这位是|那个是|这个是|她是|他是|介绍一?下)")
_SELF_CN_RE = re.compile(r"我(?:是|叫)\s*")
_SELF_EN_RE = re.compile(r"\bI(?:'m| am)\s+(?:your\s+|the\s+|my\s+)?", re.I)
_ADDRESS_PREPOSITION_CN_RE = re.compile(r"[跟给对向叫]\s*")
_ADDRESS_PARTICLE_CN_RE = re.compile(r"\s*(?:好|呀|啊|呢|，|,|！|。|、)")
_ADDRESS_EN_RE = re.compile(r"\b(?:hi|hello|to|for)\s+", re.I)
_QUOTE_LIMIT = 160


def _clean(value, limit):
    return " ".join(str(value or "").split()).strip()[:limit]


def terms_in_text(text):
    """Return canonical relationship terms appearing in one utterance, in order."""
    found = []
    for match in _CHINESE_TERM_RE.finditer(str(text or "")):
        found.append(match.group(0))
    for match in _ENGLISH_TERM_RE.finditer(str(text or "")):
        found.append(_TERM_ALIASES[match.group(1).casefold()])
    # ASR may render 姥姥 as the homophone 老老 in an explicit New Year greeting.
    if re.search(r"老老.{0,4}(?:过年好|新年好|拜年|好孩子)", str(text or "")):
        found.append("姥姥")
    return list(dict.fromkeys(found))


def _usage_of(term, quote):
    """Classify how an utterance uses the term; only these classes can identify a person."""
    text = str(quote or "")
    escaped = re.escape(term)
    if re.search(_SELF_CN_RE.pattern + escaped, text):
        return "self"
    if re.search(_SELF_EN_RE.pattern + escaped, text, re.I):
        return "self"
    for pattern in (_INTRODUCE_RE.pattern, r"\b(?:this|that)\s+is\s+(?:my\s+|the\s+)?"):
        if re.search(pattern + r"[“\"']?\s*" + escaped, text, re.I):
            return "introduction"
    if re.search(_ADDRESS_PREPOSITION_CN_RE.pattern + escaped, text):
        return "address"
    if re.search(r"(?:谢谢|感谢)(?:您|你)?\s*" + escaped, text):
        return "address"
    if re.search(r"\b(?:thanks(?:\s+to)?|thank\s+you\s+to)\s+" + escaped + r"\b", text, re.I):
        return "address"
    # New-Year greetings address the following family role directly (for example
    # “姥姥过年好”); requiring a separate “对/给” particle misses this common form.
    if re.search(escaped + r"\s*(?:过年好|新年好|拜年|你好|您好)", text):
        return "address"
    if term == "姥姥" and re.search(r"老老(?:过年|新年)?好", text):
        return "address"
    if re.search(escaped + _ADDRESS_PARTICLE_CN_RE.pattern, text, re.I):
        return "address"
    if re.search(_ADDRESS_EN_RE.pattern + escaped, text, re.I):
        return "address"
    # English vocatives commonly start a sentence ("Dad, you ...") or are
    # set off by commas ("..., Dad, ..."). They name the listener.
    if re.search(r"^\s*" + escaped +
                 r"(?:\s*[,!:]|\s+you\b)", text, re.I):
        return "address"
    if re.search(r"[,;]\s*" + escaped + r"\s*[,!?:]", text, re.I):
        return "address"
    return "mention"


def _person_text(row):
    return " ".join(_clean(row.get(key), 300) for key in
                    ("appearance", "face_features", "evidence"))


def _approach_recipients(visual_facts):
    """Map a visible entity to the person they walked up to, from observed approach facts."""
    recipients = {}
    for fact in visual_facts or []:
        if not isinstance(fact, dict):
            continue
        predicate = _clean(fact.get("predicate"), 60).casefold()
        actor = _clean(fact.get("actor") or fact.get("subject"), 120)
        target = _clean(fact.get("value"), 240)
        if not actor or not target:
            continue
        # Action verbs are often in the fact value ("moved: approached ..."),
        # not the controlled predicate ("moved"). Inspect both while retaining
        # the full target phrase for later unique-entity matching.
        action_text = f"{predicate} {target}"
        if _APPROACH_RE.search(action_text) and not _SPEAKER_IS_CAMERA_WEARER.match(actor):
            recipients.setdefault(actor.lower(), []).append(target)
    return recipients


def _candidate_for_term(term, usage, observations, visual_facts, quote=""):
    """Resolve spoken roles only when the clip supplies a unique visual participant."""
    rows = [row for row in observations or []
            if isinstance(row, dict) and _clean(row.get("entity"), 120)]
    if not rows:
        return None, ""
    if usage in {"address", "introduction"}:
        # If an existing visual observation already carries the exact spoken role
        # label, attach the quote as support instead of leaving a valid label
        # unaudited merely because other people are also visible.
        exact = [row for row in rows
                 if _clean(row.get("entity"), 120).casefold() == _clean(term, 120).casefold()]
        if len(exact) == 1:
            return exact[0], "当前可见人物已有同名身份标签，语音在同片中明确称呼/介绍该身份"
    facing = [row for row in rows if _CAMERA_FACING_RE.search(_person_text(row))]
    speakers = [row for row in rows if _visibly_speaking(row)]
    if usage == "self":
        # Self-identification belongs to its speaker. Link only a unique visible
        # speaker, or a sole camera-facing participant in an explicit interaction.
        if len(speakers) == 1:
            return speakers[0], "同片画面唯一可见说话者与明确自我介绍对应"
        if len(facing) == 1:
            return facing[0], "画面中唯一朝镜头互动的人物与明确自我介绍对应"
        return None, ""
    if usage == "address":
        # A vocative identifies the listener, never the speaker.
        recipients = _approach_recipients(visual_facts)
        approached = []
        for row in rows:
            entity = _clean(row.get("entity"), 120).casefold()
            if any(entity and (entity in target.casefold()
                               or target.casefold() in entity)
                   for targets in recipients.values() for target in targets):
                approached.append(row)
        approached = list({id(row): row for row in approached}.values())
        if len(approached) == 1:
            return approached[0], "画面中的靠近/问候动作指向这位称呼对象"
        direct_greeting = bool(re.search(
            r"(?:过年好|新年好|拜年|你好|您好|\b(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))\b)",
            str(quote or ""), re.I))
        if len(rows) == 1 and direct_greeting:
            return rows[0], "唯一可见人物与明确问候/拜年称呼相对应"
        if len(speakers) == 1 and len(rows) > 1:
            listeners = [row for row in rows if row is not speakers[0]]
            if len(listeners) == 1:
                return listeners[0], "画面中唯一说话者与唯一可见听者构成互动"
        return None, ""
    if len(facing) == 1:
        return facing[0], "画面里只有这一位人物面朝镜头互动"
    if len(rows) == 1:
        return rows[0], "clip 只拍到一位可见人物"
    recipients = _approach_recipients(visual_facts)
    approached = []
    for row in rows:
        entity = _clean(row.get("entity"), 120).casefold()
        for targets in recipients.values():
            for target in targets:
                if entity and (entity in target.casefold() or target.casefold() in entity):
                    approached.append(row)
    approached = list({id(row): row for row in approached}.values())
    if len(approached) == 1:
        return approached[0], "clip 里另一位人物走向/问候的正是这位人物"
    return None, ""


def person_labels_from_observations(people, observations):
    """Fill the event's person list from visual observations and supported names.

    Appearance labels remain on each observation as the visual anchor. The event
    person list uses a spoken or readable name only when the observation carries
    its source evidence; otherwise it keeps the visual description.
    """
    rows = [row for row in observations or [] if isinstance(row, dict)]
    labels_by_entity = {}
    for row in rows:
        entity = _clean(row.get("entity"), 120)
        if not entity:
            continue
        readable = _clean(row.get("readable_name"), 120)
        if readable and _clean(row.get("name_evidence"), 320):
            labels_by_entity[entity.casefold()] = readable
            continue
        visual_entity = _clean(row.get("visual_entity"), 120)
        if visual_entity:
            labels_by_entity[visual_entity.casefold()] = entity
        spoken = next((item for item in row.get("spoken_aliases") or []
                       if isinstance(item, dict)
                       and _clean(item.get("alias"), 120)
                       and item.get("status") == "probable_visual_match"
                       and _clean(item.get("basis"), 320)
                       and isinstance(item.get("evidence_ref"), dict)
                       and item["evidence_ref"].get("modality") == "speech"
                       and item["evidence_ref"].get("source_segment_id")), None)
        if spoken:
            labels_by_entity[entity.casefold()] = _clean(spoken.get("alias"), 120)

    output = []
    seen = set()

    def add(value):
        original = _clean(value, 180)
        if not original:
            return
        label = labels_by_entity.get(original.casefold(), original)
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            output.append(label)

    for value in people or []:
        add(value)
    for row in rows:
        add(row.get("entity"))
    return output


def _basis(term, usage, quote, why):
    usage_label = {"introduction": "被介绍为", "address": "被称呼为", "self": "自称为"}.get(usage, "")
    return (f"语音{_clean(quote, _QUOTE_LIMIT)}{usage_label}{term}；{why}。"
            "语音未做说话人分离，此关联按画面唯一线索给出。")[:320]


def _visibly_speaking(row):
    text = _person_text(row)
    return bool(re.search(
        r"\b(?:speaking|appears? to speak|mouth (?:is )?moving as if speaking)\b",
        text, re.I))


def _child_role_for_address(term, row):
    """Infer a speaker's family role from a parent vocative, never from age."""
    term = str(term or "").casefold()
    parents = {"dad", "father", "mom", "mother", "爸爸", "父亲", "爸",
               "妈妈", "母亲", "妈"}
    if term not in parents:
        return ""
    text = " ".join(str(row.get(key) or "") for key in
                    ("entity", "appearance", "face_features", "evidence"))
    if re.search(r"\b(?:man|male)\b", text, re.I):
        return "son"
    if re.search(r"\b(?:woman|female)\b", text, re.I):
        return "daughter"
    return "child"


def _attach_speaker_role(row, role, info):
    quote = _clean(info.get("quote"), _QUOTE_LIMIT)
    addressed = _clean(info.get("term"), 60) or quote
    basis = (f"语音原句“{quote}”中说话者称对方为“{addressed}”；"
             f"同片画面将当前人物唯一定位为说话者，据此记录为对方的{role}，置信度中。")
    aliases = [item for item in (row.get("spoken_aliases") or [])
               if isinstance(item, dict)
               and not (str(item.get("alias") or "").casefold() in
                        {"dad", "father", "mom", "mother", "爸爸", "父亲", "妈妈", "母亲"}
                        and item.get("status") == "probable_visual_match")]
    aliases.append({"alias": role, "status": "probable_visual_match",
                    "confidence": "medium", "basis": basis,
                    "evidence_ref": dict(info.get("evidence_ref") or {})})
    row["spoken_aliases"] = aliases[-MAX_ALIASES_PER_PERSON:]
    row["visual_entity"] = str(row.get("entity") or "")[:120]
    row["entity"] = role
    row["profile_context"] = f"画面中的说话者称对方为“{addressed}”；对方未与当前人物合并。"
    relations = [value for value in (row.get("relationships") or [])
                 if isinstance(value, str)]
    relation = f"说话者被画面定位为{role}；语音称呼对方为“{addressed}”"
    if relation not in relations:
        relations.append(relation)
    row["relationships"] = relations[-MAX_RELATIONSHIPS_PER_PERSON:]
    row["profile_confidence"] = "medium"


def attach_spoken_identity(observations, speech_facts, visual_facts=None):
    """Annotate visible people with evidence-backed spoken terms.

    Returns {"linked": [...], "unlinked": [...]} so callers can report what stayed unknown.
    """
    rows = [row for row in observations or [] if isinstance(row, dict)]
    rows_by_id = {id(row): row for row in rows}
    linked, unlinked = [], []
    if not rows:
        return {"linked": linked, "unlinked": unlinked}

    participants = {}
    co_mentions = {}
    for fact in speech_facts or []:
        if not isinstance(fact, dict):
            continue
        quote = _clean(fact.get("evidence") or fact.get("value"), 400)
        if not quote:
            continue
        ref = fact.get("evidence_ref") if isinstance(fact.get("evidence_ref"), dict) else {}
        try:
            spoke_at = float(ref.get("audio_start_s") or 0.0)
        except (TypeError, ValueError):
            spoke_at = 0.0
        quote_terms = terms_in_text(quote)
        for term in quote_terms:
            usage = _usage_of(term, quote)
            if usage == "mention":
                continue
            co_mentions.setdefault(term, set()).update(
                other for other in quote_terms
                if other != term and _usage_of(other, quote) != "mention")
            entry = participants.get(term)
            if entry is None or (_USAGE_RANK[usage], spoke_at) < (
                    _USAGE_RANK[entry["usage"]], entry["spoke_at"]):
                participants[term] = {"term": term, "usage": usage, "quote": quote,
                                      "spoke_at": spoke_at, "evidence_ref": ref}

    # One visible person normally maps to one spoken identity. When one utterance
    # explicitly distinguishes two addressed roles, a sole remaining participant
    # can receive a low-confidence link after the first role is grounded.
    claims = []
    for term, info in participants.items():
        visible_speakers = [row for row in rows_by_id.values() if _visibly_speaking(row)]
        if info["usage"] == "address" and len(visible_speakers) == 1:
            speaker = visible_speakers[0]
            speaker_role = _child_role_for_address(term, speaker)
            if speaker_role:
                _attach_speaker_role(speaker, speaker_role, info)
                linked.append({"term": speaker_role,
                               "entity": _clean(speaker.get("entity"), 120),
                               "usage": "speaker_role", "confidence": "medium",
                               "basis": "唯一可见说话者称对方为父/母辈，性别标签只用于区分儿子/女儿，不推测年龄"})
                unlinked.append({"term": term, "usage": info["usage"],
                                 "quote": info["quote"],
                                 "reason": "称呼指向听者；听者未与说话者合并"})
                continue
        candidate, why = _candidate_for_term(term, info["usage"],
                                             list(rows_by_id.values()), visual_facts,
                                             quote=info["quote"])
        if candidate is None:
            unlinked.append({"term": term, "usage": info["usage"], "quote": info["quote"],
                             "reason": ("自称说话人未与可见人物同步，身份保持未知"
                                        if info["usage"] == "self" else
                                        "画面里没有唯一可对应的人物，称呼保持未关联")})
            continue
        claims.append((_USAGE_RANK[info["usage"]], info["spoke_at"], term, info, candidate, why))

    taken_by = {}
    for rank, spoke_at, term, info, candidate, why in sorted(
            claims, key=lambda claim: (claim[0], claim[1], claim[2])):
        holder = taken_by.get(id(candidate))
        match_confidence = "medium"
        if holder and holder != term:
            remaining = [row for row in rows_by_id.values() if id(row) not in taken_by]
            if holder in co_mentions.get(term, set()) and len(remaining) == 1:
                # One utterance distinguishes two addressees; after the directly
                # grounded person is assigned, the sole remaining visible person
                # is a low-confidence visual candidate for the other role.
                candidate = remaining[0]
                match_confidence = "low"
                why = (f"同一句语音分别称呼{holder}和{term}；{holder}已由直接称呼/互动关联到另一人物，"
                       "唯一剩余可见人物作为低置信度候选；语音未做说话人分离")
            else:
                unlinked.append({"term": term, "usage": info["usage"], "quote": info["quote"],
                                 "reason": f"同一位可见人物已按更早的语音线索关联为{holder}"})
                continue
        taken_by[id(candidate)] = term
        aliases = [item for item in (candidate.get("spoken_aliases") or [])
                   if isinstance(item, dict)]
        if not any(item.get("alias") == term for item in aliases):
            aliases.append({
                "alias": term,
                "status": "probable_visual_match",
                "confidence": match_confidence,
                "basis": _basis(term, info["usage"], info["quote"], why),
                "evidence_ref": dict(info["evidence_ref"]),
                "visual_evidence_ref": dict(candidate.get("evidence_ref") or {}),
                "evidence_refs": [dict(ref) for ref in
                                  (info.get("evidence_ref"), candidate.get("evidence_ref"))
                                  if isinstance(ref, dict)],
            })
            candidate["spoken_aliases"] = aliases[-MAX_ALIASES_PER_PERSON:]
        relationships = [item for item in (candidate.get("relationships") or [])
                         if isinstance(item, str)]
        relationship = _clean(
            f"语音中{('被介绍为' if info['usage'] == 'introduction' else '被称呼为')}"
            f"{term}（依据：“{info['quote']}”）", 180)
        if relationship not in relationships:
            relationships.append(relationship)
            candidate["relationships"] = relationships[-MAX_RELATIONSHIPS_PER_PERSON:]
        old_entity = _clean(candidate.get("entity"), 120)
        if old_entity and old_entity.casefold() != term.casefold():
            if not candidate.get("visual_entity"):
                candidate["visual_entity"] = old_entity
            candidate["entity"] = term
        linked.append({"term": term, "entity": _clean(candidate.get("entity"), 120),
                       "usage": info["usage"], "confidence": match_confidence, "basis": why})
    return {"linked": linked, "unlinked": unlinked}
