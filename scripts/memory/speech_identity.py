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
    r"\b(?:toward the camera|at the camera|facing the camera|approach(?:es|ing)? the camera|"
    r"walk(?:s|ing)? toward the camera|looks? at the camera|smil\w+ at the camera)\b", re.I)
_APPROACH_RE = re.compile(r"\b(?:approached|reached|walked toward|came up to|went up to)\b", re.I)
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
        target = _clean(fact.get("value"), 120)
        if not actor or not target or not predicate:
            continue
        if _APPROACH_RE.search(predicate) and not _SPEAKER_IS_CAMERA_WEARER.match(actor):
            recipients.setdefault(actor.lower(), []).append(target)
    return recipients


def _candidate_for_term(term, usage, observations, visual_facts):
    """Pick the one visible person the clip itself singles out, or return None."""
    rows = [row for row in observations or []
            if isinstance(row, dict) and _clean(row.get("entity"), 120)]
    if not rows:
        return None, ""
    if usage == "self":
        # Without diarization or active-speaker evidence, a self-reference can
        # belong to the camera wearer rather than any person shown in the clip.
        return None, ""
    if usage == "address":
        # A vocative names the listener, never the speaking person. Link it only
        # when an observed approach identifies who received the address.
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
            return approached[0], "clip 里另一位人物走向/问候的正是这位人物"
        return None, ""
    facing = [row for row in rows if _CAMERA_FACING_RE.search(_person_text(row))]
    if len(facing) == 1:
        return facing[0], "clip 里只有这一位人物面朝镜头移动或微笑"
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
    term = str(term or "").casefold()
    parents = {"dad", "father", "爸爸", "父亲", "爸", "mom", "mother",
               "妈妈", "母亲", "妈"}
    if term not in parents:
        return ""
    text = " ".join(str(row.get(key) or "") for key in
                    ("entity", "appearance", "face_features", "evidence"))
    if re.search(r"\b(?:man|male|boy)\b", text, re.I):
        return "son"
    if re.search(r"\b(?:woman|female|girl)\b", text, re.I):
        return "daughter"
    return "child"


def _attach_speaker_role(row, role, info):
    quote = _clean(info.get("quote"), _QUOTE_LIMIT)
    addressed = _clean(info.get("term"), 60) or quote
    basis = (f"语音原句“{quote}”中说话者称对方为“{addressed}”；"
             f"画面证据显示该人物正在说话，据此记录其为对方的{role}，置信度中。")
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
                if isinstance(value, str)
                and not re.search(r"(?:father|mother) of the speaker|被称呼为(?:dad|father|mom|mother)",
                                  value, re.I)]
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
        for term in terms_in_text(quote):
            usage = _usage_of(term, quote)
            if usage == "mention":
                continue
            entry = participants.get(term)
            if entry is None or (_USAGE_RANK[usage], spoke_at) < (
                    _USAGE_RANK[entry["usage"]], entry["spoke_at"]):
                participants[term] = {"term": term, "usage": usage, "quote": quote,
                                      "spoke_at": spoke_at, "evidence_ref": ref}

    # One visible person can only answer to one spoken term. Claims compete in order
    # of evidence strength (introduction, then self-report, then address) and the
    # speaker's turn time; the losers stay unlinked instead of piling aliases onto
    # the same card.
    claims = []
    for term, info in participants.items():
        visible_speakers = [row for row in rows_by_id.values() if _visibly_speaking(row)]
        if info["usage"] == "address" and len(visible_speakers) == 1:
            speaker = visible_speakers[0]
            reverse_role = _child_role_for_address(term, speaker)
            if reverse_role:
                _attach_speaker_role(speaker, reverse_role, info)
                why = "同片画面显示该人物正在说话，语音称呼对方为父/母辈称谓"
                linked.append({"term": reverse_role,
                               "entity": _clean(speaker.get("entity"), 120),
                               "usage": "speaker_role", "confidence": "medium",
                               "basis": why})
                unlinked.append({"term": term, "usage": info["usage"],
                                 "quote": info["quote"],
                                 "reason": "这是说话者对听者的称呼，听者身份不与说话者合并"})
                continue
        candidate, why = _candidate_for_term(term, info["usage"],
                                             list(rows_by_id.values()), visual_facts)
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
        if holder and holder != term:
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
                "confidence": "medium",
                "basis": _basis(term, info["usage"], info["quote"], why),
                "evidence_ref": dict(info["evidence_ref"]),
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
        linked.append({"term": term, "entity": _clean(candidate.get("entity"), 120),
                       "usage": info["usage"], "confidence": "medium", "basis": why})
    return {"linked": linked, "unlinked": unlinked}
