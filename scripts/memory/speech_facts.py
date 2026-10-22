"""Extract explicit, timestamped facts from transcript evidence."""

import re

from speech_identity import terms_in_text, _usage_of as _speech_term_usage


_AUDIO_EVIDENCE_LINE_RE = re.compile(
    r"^\[audio\s+(?P<start>\d+(?:\.\d+)?)\s*[–-]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*s;\s*(?P<language>[^\]]+)\]\s*(?P<text>.+)$",
    re.I,
)
_EXPLICIT_ENGLISH_AGE_RE = re.compile(
    r"\b(?:i am|i'm)\s+(\d{1,3})\s+(?:years? old|yo)\b", re.I)
_EXPLICIT_CHINESE_AGE_RE = re.compile(r"(?:我|本人)(?:今年)?\s*(\d{1,3})\s*岁")

_ENGLISH_PERSON_SUBJECT = (
    r"(?:[A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){0,2}"
    r"|(?i:(?:my|his|her|their)\s+(?:mother|mom|mum|father|dad|daughter|son|"
    r"wife|husband|sister|brother|friend|colleague|doctor))"
    r"|(?i:he|she|they))"
)
_CHINESE_PERSON_SUBJECT = (
    r"(?:我(?:们)?(?:的)?(?:妈妈|母亲|妈|爸爸|父亲|爸|奶奶|爷爷|女儿|儿子|"
    r"妻子|丈夫|姐妹|兄弟|朋友|同事)?|他(?:们)?|她(?:们)?|[\u4e00-\u9fff]{2,4})"
)
_EXPLICIT_ENGLISH_OTHER_AGE_RE = re.compile(
    rf"\b(?P<subject>{_ENGLISH_PERSON_SUBJECT})\s+(?:(?:is|are)\s+)?"
    r"(?P<age>\d{1,3})\s+(?:years? old|yo)\b"
)
_EXPLICIT_CHINESE_OTHER_AGE_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*"
    r"(?:(?:今年|现在|目前)\s*)?(?P<age>\d{1,3})\s*岁"
)
_EXPLICIT_ENGLISH_OTHER_PLAN_RE = re.compile(
    rf"\b(?P<subject>{_ENGLISH_PERSON_SUBJECT})\s+"
    r"(?:(?:is|are)\s+)?(?:planning to|plans to|going to|will|shall|gonna)\b(?!\s+(?:not|never)\b)"
)
_EXPLICIT_CHINESE_OTHER_PLAN_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*"
    r"(?P<time_scope>未来几天|这几天|今天|今晚|明天|后天|本周|这周|下周(?:[一二三四五六日天])?|"
    r"周[一二三四五六日天]|月底|最近)?\s*"
    r"(?<!没有)(?<!不)(?<!没)(?<!未)(?:打算|计划|准备|将要|将|会|要)"
)
_ENGLISH_HEALTH_CONDITION = (
    r"(?:type\s+\d+\s+diabetes|high blood pressure|heart disease|heart condition|heart attack|"
    r"kidney disease|blood clot|diabetes|asthma|hypertension|epilepsy|cancer|stroke|"
    r"arthritis|dementia|allerg(?:y|ies))"
)
_EXPLICIT_ENGLISH_OTHER_HEALTH_RE = re.compile(
    rf"\b(?P<subject>{_ENGLISH_PERSON_SUBJECT})\s+(?:"
    rf"(?:(?:has|had|suffers? from|was diagnosed with|has been diagnosed with|"
    rf"is diagnosed with)\s+(?:a\s+|an\s+)?(?P<condition>{_ENGLISH_HEALTH_CONDITION})\b)"
    rf"|(?:(?:takes?|is taking)\s+(?:medication|medicine|insulin|pills?)\s+for\s+"
    rf"(?P<med_condition>{_ENGLISH_HEALTH_CONDITION})\b)"
    rf"|(?:(?:is|are)\s+allergic to\s+(?P<allergen>[^.!?,;]{{1,40}}))"
    rf"|(?:(?:had|has had|underwent)\s+(?P<surgery>surgery|a surgery)\b)"
    rf")",
    re.I,
)
_CHINESE_HEALTH_CONDITION = (
    r"(?:糖尿病|高血压|哮喘|心脏病|心肌梗死|心梗|心脏病发作|癫痫|癌症|中风|关节炎|痴呆|老年痴呆|"
    r"血栓|肾病|过敏|手术|降压药|胰岛素|药物)"
)
_EXPLICIT_CHINESE_OTHER_HEALTH_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*(?:曾经|以前|目前|现在|一直)?"
    rf"(?<!没)(?<!不)(?<!未)(?:患有|患过|得过|有|被诊断为|被诊断出|"
    rf"确诊(?:患有)?|正在服用|服用|做过)"
    rf"[^。！？.!?]{{0,16}}?(?P<condition>{_CHINESE_HEALTH_CONDITION})"
)
_EXPLICIT_CHINESE_OTHER_ALLERGY_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*对"
    rf"(?P<allergen>[^。！？.!?]{{1,20}}?)(?<!不)过敏"
)
_EXPLICIT_PLAN_TIME_RE = re.compile(
    r"\b(?:the day after tomorrow|tomorrow|today|tonight|this week|this weekend|"
    r"next week|next month|in a few days|in \d{1,2} days|on "
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"
    r"|(?:未来几天|这几天|今天|今晚|明天|后天|本周|这周|下周(?:[一二三四五六日天])?|"
    r"周[一二三四五六日天]|月底|最近)",
    re.I,
)

_EXPLICIT_ENGLISH_PLAN_RE = re.compile(
    r"\b(?:i(?:'m| am) planning to|i plan to|i(?:'m| am) going to|"
    r"i(?:'m| am) gonna|i(?:'ll| will)|we(?:'re| are) planning to|"
    r"we plan to|we(?:'re| are) going to|we(?:'re| are) gonna|we(?:'ll| will))\b(?!\s+(?:not|never)\b)",
    re.I,
)
_EXPLICIT_CHINESE_PLAN_RE = re.compile(
    r"(?:我|我们)(?:打算|计划|准备)(?:要)?[^。！？.!?]{1,180}")
_EXPLICIT_ENGLISH_HEALTH_RE = re.compile(
    r"\b(?:i have(?!\s+(?:no|not|never)\b)|i've had(?!\s+(?:no|not|never)\b)|i had(?!\s+(?:no|not|never)\b)|i was diagnosed with(?!\s+(?:no|not|never)\b)|i have been diagnosed with(?!\s+(?:no|not|never)\b)|"
    r"i suffer from|i take (?:medication|medicine|insulin|pills?) for|"
    r"i am taking (?:medication|medicine|insulin|pills?) for)\b.{0,100}"
    r"\b(?:diabetes|asthma|hypertension|high blood pressure|heart disease|"
    r"heart condition|heart attack|epilepsy|cancer|stroke|arthritis|dementia|allerg(?:y|ies)|"
    r"blood clot|kidney disease|medication|insulin|surgery)\b"
    r"|\b(?:i'm|i am) allergic to [\w-]+(?:\s+[\w-]+){0,4}\b", re.I)
_EXPLICIT_CHINESE_HEALTH_RE = re.compile(
    r"(?:我|本人)(?:(?:曾经|以前|目前|现在|一直)?(?:患有|患过|得过|有|被诊断为|"
    r"被诊断出|确诊(?:患有)?|正在服用|服用|做过))[^。！？.!?]{0,80}"
    r"(?:糖尿病|高血压|哮喘|心脏病|心肌梗死|心梗|心脏病发作|癫痫|癌症|中风|关节炎|痴呆|老年痴呆|"
    r"血栓|肾病|过敏|手术|降压药|胰岛素|药物)|"
    r"(?:我|本人)对[^。！？.!?]{1,20}(?<!没有)(?<!不)(?<!没)过敏")


_EXPLICIT_ENGLISH_SELF_NAME_RE = re.compile(
    r"(?i:\b(?:my name is|you can call me|please call me)\s+)"
    r"(?P<name>[A-Za-z][A-Za-z'’.-]{1,40}(?:\s+[A-Z][A-Za-z'’.-]{1,40}){0,2})\b")
_EXPLICIT_CHINESE_SELF_NAME_RE = re.compile(
    r"(?:我的名字是|我叫)(?P<name>[\u4e00-\u9fff]{2,4})(?=\s|[,，。！？.!?]|$)")
_EXPLICIT_ENGLISH_OTHER_NAME_RE = re.compile(
    r"\b(?P<subject>(?i:he|she|they)|"
    r"(?i:my|our|his|her|their)\s+(?:mother|mom|mum|father|dad|daughter|son|wife|husband|"
    r"sister|brother|friend|colleague|partner|boyfriend|girlfriend|roommate))"
    r"(?i:'s name is|\s+is called|\s+goes by)\s+"
    r"(?P<name>[A-Za-z][A-Za-z'’.-]{1,40}(?:\s+[A-Z][A-Za-z'’.-]{1,40}){0,2})\b")
_EXPLICIT_CHINESE_OTHER_NAME_RE = re.compile(
    r"(?P<subject>他|她|他们|她们|我(?:的)?(?:妈妈|母亲|爸爸|父亲|奶奶|爷爷|女儿|儿子|妻子|丈夫|"
    r"姐妹|兄弟|朋友|同事|伴侣|男朋友|女朋友|室友))"
    r"(?:的名字是|的名字叫|叫)(?P<name>[\u4e00-\u9fff]{2,4})(?=\s|[,，。！？.!?]|$)")
_EXPLICIT_ENGLISH_PREFERENCE_RE = re.compile(
    r"\b(?:I|we)(?:\s+really)?\s+(?:(?:do not|don't|never)\s+)?"
    r"(?:like|love|enjoy|prefer|hate|dislike)\s+[^.!?;]{2,140}", re.I)
_EXPLICIT_CHINESE_PREFERENCE_RE = re.compile(
    r"(?:我|我们)(?:最|更|比较|非常)?(?:喜欢|爱|享受|偏好|讨厌|不喜欢|不爱)"
    r"[^。！？.!?;]{2,100}")
_EXPLICIT_ENGLISH_OTHER_PREFERENCE_RE = re.compile(
    r"\b(?P<subject>(?i:he|she|they)|"
    r"[A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){0,2}|"
    r"(?i:(?:my|our|his|her|their)\s+(?:mother|mom|mum|father|dad|daughter|son|"
    r"wife|husband|sister|brother|friend|colleague|partner|boyfriend|girlfriend|roommate)))"
    r"\s+(?:(?i:really)\s+)?(?i:likes|loves|enjoys|prefers|hates|dislikes)\s+"
    r"[^.!?;]{2,140}")
_EXPLICIT_CHINESE_OTHER_PREFERENCE_RE = re.compile(
    r"(?P<subject>他|她|他们|她们|我(?:们)?(?:的)?(?:妈妈|母亲|爸爸|父亲|奶奶|爷爷|"
    r"女儿|儿子|妻子|丈夫|朋友|同事|伴侣|男朋友|女朋友|室友)|[\u4e00-\u9fff]{2,4})"
    r"(?:最|更|比较)?(?:喜欢|爱|偏好|讨厌|不喜欢)[^。！？.!?;]{2,100}")
_EXPLICIT_ENGLISH_RELATION_RE = re.compile(
    r"\b(?:my|our)\s+(?:mother|mom|mum|father|dad|daughter|son|wife|husband|"
    r"sister|brother|friend|colleague|partner|boyfriend|girlfriend|roommate|doctor)\b|"
    r"\b(?:I(?:'m| am)|we(?:'re| are))\s+your\s+neighbou?r\b", re.I)
_EXPLICIT_CHINESE_RELATION_RE = re.compile(
    r"(?:我|我们)(?:的)?(?:妈妈|母亲|妈|爸爸|父亲|爸|奶奶|爷爷|女儿|儿子|妻子|丈夫|"
    r"姐妹|兄弟|朋友|同事|伴侣|男朋友|女朋友|室友|医生)")
_SPOKEN_VOCATIVE_NAME_RE = re.compile(
    r"^\s*(?i:hi|hello|hey|welcome|good\s+(?:morning|afternoon|evening))"
    r"\b[,!\s]+(?P<name>[A-Z][a-zA-Z'’-]*(?:\s+[A-Z][a-zA-Z'’-]*){0,2})\b")
_EXPLICIT_SELF_NEIGHBOR_RE = re.compile(
    r"\b(?:I(?:'m| am)|we(?:'re| are))\s+your\s+neighbou?r\b", re.I)
_EXPLICIT_SELF_ROLE_RE = re.compile(
    r"\b(?:I(?:'m| am)|we(?:'re| are))\s+your\s+"
    r"(?P<role>grandma|grandmother|grandpa|grandfather|mother|mom|mum|"
    r"father|dad|aunt|uncle|sister|brother|friend|neighbou?r|coworker|colleague)\b",
    re.I)
_EXPLICIT_CHINESE_SELF_ROLE_RE = re.compile(
    r"(?:我是|我就是|我们是|我们就是)(?:你|你们)的?"
    r"(?P<role>奶奶|外婆|爷爷|外公|妈妈|母亲|爸爸|父亲|朋友|邻居|同事|姐妹|兄弟)")
_EXPLICIT_SPOKEN_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))\b|"
    r"(?:姥姥|老老|外婆|奶奶|爷爷|外公|妈妈|母亲|爸爸|父亲|您好|你好)"
    r"(?:过年|新年|春节)?好|给您拜年(?:了)?|拜年好",
    re.I)
_EXPLICIT_SPOKEN_THANKS_RE = re.compile(
    r"\b(?:thank\s+you|thanks(?:\s+to)?|many\s+thanks)\b|"
    r"(?:谢谢|感谢)(?:您|你|[\u4e00-\u9fff]{1,4})?",
    re.I)
_HESITATION_ONLY_RE = re.compile(
    r"^\s*(?:(?:uh|um|er|erm|hmm|ah|oh|呃|嗯|啊)[,，、\s]*){2,}[.!?！？。]*\s*$",
    re.I,
)
_EXPLICIT_WELLBEING_QUESTION_RE = re.compile(
    r"\bhow\s+(?:are|have)\s+you(?:\s+today)?\b|"
    r"\bhow(?:'ve| have)\s+you\s+been\b", re.I)
_DIRECT_ADDRESSED_HEALTH_RE = re.compile(
    rf"^\s*(?P<name>[A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){{0,2}})"
    rf"[,，]\s*(?:your|你|您的)\s+(?:heart disease|heart condition|heart attack|"
    rf"high blood pressure|diabetes|asthma|癌症|心脏病|心肌梗死|心梗|高血压)", re.I)
_EXPLICIT_HEALTH_PRODUCT_CLAIM_RE = re.compile(
    r"\b(?:health|healthcare)\s+product\b.{0,120}\b(?:works?|address(?:es)?|"
    r"treats?|cures?|prevents?|improves?|keeps?)\b", re.I)
_EXPLICIT_FIRST_VISIT_RE = re.compile(
    r"\b(?:I|we)(?:'ve| have)?\s+never\s+(?:been here before|been to|visited)\b", re.I)
_EXPLICIT_FIRED_RE = re.compile(
    r"\b(?:you(?:'re| are)\s+fired|leave our company from today)\b", re.I)
_EXPLICIT_BIRTHDAY_RE = re.compile(r"\bhappy birthday\b", re.I)
_EXPLICIT_AIRPORT_INSTRUCTION_RE = re.compile(
    r"\b(?:please\s+(?:proceed|follow|continue|go|head)|passengers?\s+(?:arriving|should|please))\b", re.I)
_EXPLICIT_TRANSACTION_RE = re.compile(
    r"\b(?:bill|dollars?|tips?|pay|paid|payment|receipt)\b|\$\s?\d+", re.I)
_EXPLICIT_PAYMENT_REQUEST_RE = re.compile(
    r"\b(?:please\s+)?(?:pay|make\s+(?:a\s+)?payment|payment\s+is\s+due|"
    r"you\s+(?:need|have)\s+to\s+pay)\b|"
    r"(?:请|麻烦).{0,8}(?:付款|支付|付钱)|(?:需要|要).{0,5}(?:付款|支付|付钱)", re.I)

_EXPLICIT_ENGLISH_BACKGROUND_RE = re.compile(
    r"\b(?:I(?:'m| am) from|I live in|I lived in|I grew up in|I moved to|I moved from|"
    r"I recently arrived in|I just arrived in|I came from|I traveled from|I travelled from)\b", re.I)
_EXPLICIT_CHINESE_BACKGROUND_RE = re.compile(
    r"(?:我来自|我住在|我住过|我刚从|我最近从|我搬到|我长大在|我出生于)")
_EXPLICIT_ENGLISH_EXPERIENCE_RE = re.compile(
    r"\b(?:I|we)(?:'ve| have)?\s+(?:been to|visited|went to|traveled to|travelled to|"
    r"tried|used|watched|played|ate|worked at|lived in)\b|"
    r"\b(?:I|we)\s+(?:visited|went to|traveled to|travelled to|tried|used|watched|"
    r"played|ate|worked at|lived in)\b", re.I)
_EXPLICIT_CHINESE_EXPERIENCE_RE = re.compile(
    r"(?:我|我们)(?:以前|曾经)?(?:去过|到过|住过|吃过|看过|玩过|做过|用过)"
    r"[^。！？.!?;]{0,100}")


def _speech_fact_subject(raw_subject):
    """Keep quoted referents literal; unresolved pronouns never bind to visual people."""
    subject = " ".join(str(raw_subject or "").strip().split())
    folded = subject.casefold()
    if folded in {"i", "me", "we", "myself", "ourselves", "我", "我们", "本人"}:
        return "unidentified speaker"
    if folded in {"he", "she", "they", "他", "她", "他们", "她们"}:
        return f"unidentified person referred to as {folded}"
    return subject[:120]


def _sanitize_social_context(value, transcript):
    """Remove appearance-inferred age and unsupported place ownership claims."""
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    # Relationship words such as grandma/grandchild remain intact; only visual
    # age adjectives are removed because age may be read from on-screen text only.
    text = re.sub(
        r"\b(an?|the)\s+(?:elderly|older|old|young|younger|middle-aged|teenage|teen)\s+"
        r"(woman|man|person|adult|visitor|speaker|individual)\b",
        lambda m: (m.group(1) + " " + m.group(2)) if m.group(1).casefold() != "an"
        else "a " + m.group(2), text, flags=re.I)
    text = re.sub(r"\b(?:elderly|older|younger|middle-aged|teenage|teen)\s+"
                  r"(?=(?:grandmother|grandfather|grandparent)\b)", "", text, flags=re.I)
    text = re.sub(r"(?:年长的|年迈的|年老的|年轻的|老年)(?=(?:女性|男人|男性|女人|人物|人))",
                  "", text)
    spoken = str(transcript or "")
    explicit_owned_place = bool(re.search(
        r"\b(?:my|your|our|their|her|his|grandma'?s|grandmother'?s|grandpa'?s)\s+"
        r"(?:[\w-]+\s+){0,2}(?:home|house)\b|\b(?:at home|at my house|at your house)\b",
        spoken, re.I) or re.search(
        r"(?:姥姥|奶奶|外婆|爷爷|外公)(?:的)?家|(?:在|回|到|来)家(?:里|中|门口)?",
        spoken))
    if not explicit_owned_place:
        # Keep the visible spatial scene while removing ownership inferred from
        # residential architecture, decorations, or a doorway.
        text = re.sub(
            r"\b(steps?|walks?|comes?|emerges?)\s+(?:out of|from)\s+"
            r"(?:(?:a|the|their|her|his|family's|family|traditional|rural|snow-covered|"
            r"festive|warm)\s+){0,3}(?:home|house)\b",
            r"\1 through the visible doorway", text, flags=re.I)
        text = re.sub(
            r"\b(?:at|in|inside|outside|near)\s+"
            r"(?:(?:a|the|their|her|his|family's|family|traditional|rural|snow-covered|"
            r"festive|warm)\s+){0,3}(?:home|house)\b",
            "in the visible courtyard", text, flags=re.I)
        text = re.sub(r"\b(?:family's|family)\s+(?=(?:visible\s+)?courtyard\b)",
                      "visible ", text, flags=re.I)
        text = re.sub(r"\b(?:home|house)\b", "visible setting", text, flags=re.I)
        for source, target in (("姥姥家", "雪地院落"), ("奶奶家", "雪地院落"),
                               ("外婆家", "雪地院落"), ("爷爷家", "雪地院落"),
                               ("外公家", "雪地院落"), ("家门口", "门口场景"),
                               ("家里", "室内场景"), ("家中", "室内场景"),
                               ("自家", "现场")):
            text = text.replace(source, target)
    return text[:1200]


def _explicit_plan_time_scope(quote):
    match = _EXPLICIT_PLAN_TIME_RE.search(str(quote or ""))
    return match.group(0).strip() if match else ""


def _extract_explicit_speech_facts(transcript, source_segment_id):
    """Keep explicit identity, preference, relationship, background, experience, age,
    plan, and health claims as text facts with audio provenance.

    ASR has no speaker diarization. First-person claims stay attached to an
    unidentified speaker; third-person referents are stored literally and are
    never joined to a visually observed person by pronoun or guesswork.
    """
    facts = []
    audio_rows = []
    instruction_rows = []
    for line in str(transcript or "").splitlines():
        match = _AUDIO_EVIDENCE_LINE_RE.match(line.strip())
        if not match:
            continue
        quote = match.group("text").strip()
        try:
            audio_start, audio_end = round(float(match.group("start")), 3), round(float(match.group("end")), 3)
        except (TypeError, ValueError):
            continue
        if not quote or audio_end < audio_start:
            continue
        ref = {"source_segment_id": str(source_segment_id or "")[:128],
               "modality": "speech", "audio_start_s": audio_start,
               "audio_end_s": audio_end}
        audio_rows.append({"quote": quote, "start": audio_start, "end": audio_end,
                           "evidence_ref": ref})
        for term in terms_in_text(quote):
            escaped = re.escape(term)
            chinese_instruction = re.search(
                r"(?:跟|对|向|给)\s*" + escaped + r"\s*(?:说|讲|叫|喊|拜年|打招呼)", quote)
            english_instruction = re.search(
                r"\b(?:tell|say(?:\s+to)?|greet|address|call)\s+(?:your\s+|the\s+)?"
                + escaped + r"\b", quote, re.I)
            if chinese_instruction or english_instruction:
                instruction_rows.append({"term": term, "quote": quote,
                                         "start": audio_start, "end": audio_end,
                                         "evidence_ref": ref})
    transcript_text = " ".join(row["quote"] for row in audio_rows)
    airport_announcement = bool(
        re.search(r"\b(?:welcome to|arriving at|now arriving at)\b.{0,100}\bairport\b", transcript_text, re.I)
        and re.search(r"\b(?:please proceed|immigration|customs|passengers|flight)\b", transcript_text, re.I)
    )
    last_addressed_name = ""
    last_addressed_end = None
    for line in str(transcript or "").splitlines():
        match = _AUDIO_EVIDENCE_LINE_RE.match(line.strip())
        if not match:
            continue
        quote = match.group("text").strip()
        try:
            audio_start = round(float(match.group("start")), 3)
            audio_end = round(float(match.group("end")), 3)
        except (TypeError, ValueError):
            continue
        if not quote or audio_end < audio_start:
            continue
        if _HESITATION_ONLY_RE.fullmatch(quote):
            facts.append({
                "subject_type": "event", "subject": "speech interaction",
                "predicate": "spoken_disfluency",
                "value": "The transcript contains repeated hesitation fillers; the speaker is unidentified.",
                "modality": "speech", "status": "observed", "confidence": "low",
                "evidence": quote[:500],
                "evidence_ref": {
                    "source_segment_id": str(source_segment_id or "")[:128],
                    "modality": "speech", "audio_start_s": audio_start,
                    "audio_end_s": audio_end,
                },
            })
        evidence_ref = {
            "source_segment_id": str(source_segment_id or "")[:128],
            "modality": "speech",
            "audio_start_s": audio_start,
            "audio_end_s": audio_end,
        }

        self_role_match = (_EXPLICIT_SELF_ROLE_RE.search(quote)
                           or _EXPLICIT_CHINESE_SELF_ROLE_RE.search(quote))
        neighbor_self_identified = bool(_EXPLICIT_SELF_NEIGHBOR_RE.search(quote))
        role_aliases = {
            "grandmother": "grandma", "grandma": "grandma",
            "grandfather": "grandpa", "grandpa": "grandpa",
            "mother": "mom", "mom": "mom", "mum": "mom",
            "father": "dad", "dad": "dad", "neighbour": "neighbor",
            "neighbor": "neighbor", "colleague": "coworker", "coworker": "coworker",
            "奶奶": "grandma", "外婆": "grandma", "爷爷": "grandpa", "外公": "grandpa",
            "妈妈": "mom", "母亲": "mom", "爸爸": "dad", "父亲": "dad",
            "朋友": "friend", "邻居": "neighbor", "同事": "coworker",
            "姐妹": "sister", "兄弟": "brother",
        }
        self_role = (role_aliases.get(self_role_match.group("role").casefold())
                     if self_role_match else None)
        speaker_alias = f"{self_role} 1" if self_role else "unidentified speaker"

        def add_spoken_claim(predicate, value, subject=None,
                             status="stated", actor=None,
                             affected_people=None):
            fact = {
                "subject_type": "person", "subject": subject or speaker_alias,
                "predicate": predicate, "value": str(value or "")[:500],
                "modality": "speech", "status": status,
                "confidence": "medium", "evidence": quote[:500],
                "evidence_ref": dict(evidence_ref),
            }
            # Whisper does not identify voices. Store the utterance source honestly
            # and link only explicitly named or addressed people as text references.
            if actor is None:
                actor = speaker_alias
            if actor:
                fact["actor"] = str(actor)[:120]
            if affected_people:
                fact["affected_people"] = list(dict.fromkeys(
                    str(person).strip()[:120] for person in affected_people if str(person).strip()
                ))
            facts.append(fact)

        vocative = _SPOKEN_VOCATIVE_NAME_RE.search(quote)
        directed_health = _DIRECT_ADDRESSED_HEALTH_RE.search(quote)
        addressed_name = (directed_health.group("name").strip()
                          if directed_health else
                          vocative.group("name").strip() if vocative else "")
        if last_addressed_end is not None and audio_start - last_addressed_end > 3.0:
            last_addressed_name = ""
        if addressed_name:
            last_addressed_name = addressed_name
        elif last_addressed_name:
            addressed_name = last_addressed_name
        last_addressed_end = audio_end
        if addressed_name and (directed_health or vocative):
            add_spoken_claim(
                "addressed_person", f"A speaker addresses the listener as {addressed_name}.",
                subject=_speech_fact_subject(addressed_name),
                actor=speaker_alias,
                affected_people=[_speech_fact_subject(addressed_name)])

        greeting_name = addressed_name
        if _EXPLICIT_SPOKEN_GREETING_RE.search(quote):
            for term in terms_in_text(quote):
                direct_term_greeting = bool(re.search(
                    re.escape(term) + r"(?:过年|新年|春节)?好", quote))
                if term == "姥姥" and re.search(r"老老(?:过年|新年)?好", quote):
                    direct_term_greeting = True
                if direct_term_greeting:
                    greeting_name = term
                    break
        if greeting_name and _EXPLICIT_SPOKEN_GREETING_RE.search(quote):
            last_addressed_name = greeting_name
            last_addressed_end = audio_end
        greeting_target = (_speech_fact_subject(greeting_name) if greeting_name
                           else "unidentified addressee")
        greeting_target_label = greeting_name or "the listener"
        if _EXPLICIT_SPOKEN_GREETING_RE.search(quote):
            add_spoken_claim(
                "greeted", f"{speaker_alias} greeted {greeting_target_label}.",
                subject=speaker_alias, status="spoken_event", actor=speaker_alias,
                affected_people=[greeting_target])
        if _EXPLICIT_WELLBEING_QUESTION_RE.search(quote):
            add_spoken_claim(
                "asked_about_wellbeing",
                f"{speaker_alias} asked {greeting_target_label} how they were.",
                subject=speaker_alias, status="spoken_event", actor=speaker_alias,
                affected_people=[greeting_target])
        if _EXPLICIT_SPOKEN_THANKS_RE.search(quote) and not airport_announcement:
            thanks_name = ""
            for term in terms_in_text(quote):
                position = quote.find(term)
                prefix = quote[max(0, position - 12):position] if position >= 0 else ""
                if position >= 0 and re.search(r"(?:谢谢|感谢)(?:您|你)?\s*$", prefix):
                    thanks_name = term
                    break
            thanks_name = thanks_name or addressed_name
            thanks_target = (_speech_fact_subject(thanks_name)
                             if thanks_name else "unidentified addressee")
            add_spoken_claim(
                "thanked", f"{speaker_alias} thanked {thanks_name or 'the listener'}.",
                subject=speaker_alias, status="spoken_event", actor=speaker_alias,
                affected_people=[thanks_target])
        for spoken_term in terms_in_text(quote):
            # One row per spoken participant: the term itself becomes retrievable and
            # keeps the verbatim quote as evidence, instead of hiding behind
            # "unidentified speaker" as its only subject.
            add_spoken_claim(
                "spoken_family_reference", quote,
                subject=spoken_term, status="quoted_relationship_term",
                actor=speaker_alias, affected_people=[spoken_term])
        if _EXPLICIT_HEALTH_PRODUCT_CLAIM_RE.search(quote):
            add_spoken_claim(
                "quoted_product_claim", quote, status="quoted_unverified_claim")
        if _EXPLICIT_FIRED_RE.search(quote):
            add_spoken_claim(
                "reported_employment_event", quote,
                subject="unidentified addressee",
                affected_people=["unidentified addressee"])
        if _EXPLICIT_BIRTHDAY_RE.search(quote):
            birthday_person = (_speech_fact_subject(addressed_name) if addressed_name
                               else "unidentified addressee")
            add_spoken_claim(
                "stated_birthday_event", quote,
                subject=birthday_person, affected_people=[birthday_person])
        if airport_announcement and (
                _EXPLICIT_AIRPORT_INSTRUCTION_RE.search(quote)
                or re.search(r"\bairport\b", quote, re.I)):
            facts.append({
                "subject_type": "event", "subject": "airport public-address announcement",
                "predicate": "spoken_travel_instruction", "value": quote[:500],
                "actor": "public-address announcement", "affected_people": ["passengers"],
                "modality": "speech", "status": "observed", "confidence": "high",
                "evidence": quote[:500], "evidence_ref": dict(evidence_ref),
            })
        elif _EXPLICIT_AIRPORT_INSTRUCTION_RE.search(quote):
            facts.append({
                "subject_type": "event", "subject": "public-address announcement",
                "predicate": "spoken_travel_instruction", "value": quote[:500],
                "actor": "public-address announcement", "affected_people": ["passengers"],
                "modality": "speech", "status": "observed", "confidence": "medium",
                "evidence": quote[:500], "evidence_ref": dict(evidence_ref),
            })
        if _EXPLICIT_TRANSACTION_RE.search(quote):
            bill_total = re.search(
                r"\b(?:your |the )?(?:bill|total|check)\s+(?:is|was|comes to|came to)\s+"
                r"(?:\$\s*)?(?:\d+(?:\.\d{1,2})?|[a-z]+(?:[ -][a-z]+){0,3})\s*"
                r"(?:u\.?s\.?\s*)?(?:dollars?|bucks?)\b|"
                r"(?:账单|合计|总计|一共)[^。！？.!?]{0,12}(?:是|为)?\s*"
                r"(?:\$\s*)?[零一二两三四五六七八九十百千\d]+(?:\.\d+)?\s*(?:美元|美金|元|块钱)",
                quote, re.I)
            if bill_total:
                facts.append({
                    "subject_type": "object", "subject": "restaurant bill", "predicate": "spoken_total",
                    "value": quote[:500], "actor": "unidentified speaker",
                    "modality": "speech", "status": "stated", "confidence": "high",
                    "evidence": quote[:500], "evidence_ref": dict(evidence_ref),
                })
            elif _EXPLICIT_PAYMENT_REQUEST_RE.search(quote):
                delivery_order = bool(re.search(
                    r"\b(?:delivery\s+order|delivery|order\s+delivery|your\s+order)\b|"
                    r"(?:外卖|送餐|配送订单|送来的订单)", quote, re.I))
                amount = ""
                amount_match = re.search(
                    r"\$\s*(\d+(?:\.\d{1,2})?)|"
                    r"\b(\d+(?:\.\d{1,2})?)\s*(?:u\.?s\.?\s*)?"
                    r"(?:dollars?|bucks?)\b|"
                    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
                    r"\s+dollars?\b", quote, re.I)
                if amount_match:
                    numeric = amount_match.group(1) or amount_match.group(2)
                    if numeric:
                        amount = "$" + format(float(numeric), "g")
                    else:
                        amount_word = amount_match.group(3).casefold()
                        amount_value = {"one":1,"two":2,"three":3,"four":4,"five":5,
                                        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                                        "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
                                        "fifteen":15,"sixteen":16,"seventeen":17,
                                        "eighteen":18,"nineteen":19,"twenty":20}[amount_word]
                        amount = "$" + str(amount_value)
                subject = "delivery order" if delivery_order else "purchase transaction"
                requested = (f"The speaker asked the user to pay {amount} for the delivery order."
                             if delivery_order and amount else
                             f"The speaker asked the user to pay {amount}." if amount else
                             "The speaker asked the user to pay for the delivery order."
                             if delivery_order else "The speaker explicitly requested payment.")
                facts.append({
                    "subject_type": "event", "subject": subject,
                    "predicate": "payment_requested", "value": requested,
                    "actor": "unidentified speaker", "affected_people": ["unidentified addressee"],
                    "modality": "speech", "status": "stated", "confidence": "high",
                    "evidence": quote[:500], "evidence_ref": dict(evidence_ref),
                })
            else:
                facts.append({
                    "subject_type": "event", "subject": "purchase transaction",
                    "predicate": "spoken_transaction_detail", "value": quote[:500],
                    "actor": "unidentified speaker", "modality": "speech",
                    "status": "stated", "confidence": "medium",
                    "evidence": quote[:500], "evidence_ref": dict(evidence_ref),
                })

        name_match = (_EXPLICIT_ENGLISH_SELF_NAME_RE.search(quote)
                      or _EXPLICIT_CHINESE_SELF_NAME_RE.search(quote))
        if name_match:
            add_spoken_claim("stated_name", name_match.group("name").strip())
        other_name = (_EXPLICIT_ENGLISH_OTHER_NAME_RE.search(quote)
                      or _EXPLICIT_CHINESE_OTHER_NAME_RE.search(quote))
        if other_name:
            add_spoken_claim(
                "stated_name", other_name.group("name").strip(),
                subject=_speech_fact_subject(other_name.group("subject")))
        if (_EXPLICIT_ENGLISH_PREFERENCE_RE.search(quote)
                or _EXPLICIT_CHINESE_PREFERENCE_RE.search(quote)):
            add_spoken_claim("stated_preference", quote)
        other_preference = (_EXPLICIT_ENGLISH_OTHER_PREFERENCE_RE.search(quote)
                            or _EXPLICIT_CHINESE_OTHER_PREFERENCE_RE.search(quote))
        if other_preference:
            add_spoken_claim(
                "stated_preference", quote,
                subject=_speech_fact_subject(other_preference.group("subject")))
        if (_EXPLICIT_ENGLISH_RELATION_RE.search(quote)
                or _EXPLICIT_CHINESE_RELATION_RE.search(quote)
                or self_role_match):
            relationship_value = quote
            relationship_status = "stated"
            if self_role:
                relationship_status = "self_identified"
            if neighbor_self_identified:
                target = addressed_name or "the listener"
                relationship_value = (
                    f"{speaker_alias} self-identifies as {target}'s neighbor.")
                relationship_status = "self_identified"
            add_spoken_claim(
                "stated_relationship", relationship_value,
                subject=speaker_alias, status=relationship_status,
                actor=speaker_alias,
                affected_people=[_speech_fact_subject(addressed_name)]
                if addressed_name else None)
        if (_EXPLICIT_ENGLISH_BACKGROUND_RE.search(quote)
                or _EXPLICIT_CHINESE_BACKGROUND_RE.search(quote)):
            add_spoken_claim("stated_background", quote)
        if (_EXPLICIT_ENGLISH_EXPERIENCE_RE.search(quote)
                or _EXPLICIT_FIRST_VISIT_RE.search(quote)
                or _EXPLICIT_CHINESE_EXPERIENCE_RE.search(quote)):
            add_spoken_claim("stated_experience", quote)

        age_match = _EXPLICIT_ENGLISH_AGE_RE.search(quote)
        age_subject = speaker_alias if age_match else ""
        age_is_english = bool(age_match)
        if not age_match:
            age_match = _EXPLICIT_CHINESE_AGE_RE.search(quote)
            age_subject = speaker_alias if age_match else ""
            age_is_english = False
        if not age_match:
            age_match = _EXPLICIT_ENGLISH_OTHER_AGE_RE.search(quote)
            if age_match:
                age_subject = _speech_fact_subject(age_match.group("subject"))
                age_is_english = True
        if not age_match:
            age_match = _EXPLICIT_CHINESE_OTHER_AGE_RE.search(quote)
            if age_match:
                age_subject = _speech_fact_subject(age_match.group("subject"))
                age_is_english = False

        if age_match:
            age = int(age_match.group("age") if age_match.groupdict().get("age")
                      else age_match.group(1))
            if 1 <= age <= 120:
                age_value = f"{age} years old" if age_is_english else f"{age}岁"
                facts.append({
                    "subject_type": "person", "subject": age_subject,
                    "predicate": "stated_age", "value": age_value,
                    "modality": "speech", "status": "stated",
                    "confidence": "medium", "evidence": quote[:500],
                    "evidence_ref": dict(evidence_ref),
                })

        first_person_plan = (
            _EXPLICIT_ENGLISH_PLAN_RE.search(quote)
            or _EXPLICIT_CHINESE_PLAN_RE.search(quote)
        )
        non_plan_future = re.search(
            r"\b(?:I|we)\s+(?:will|shall|'ll)\s+(?:die|be dead|not survive)\b",
            quote, re.I)
        if non_plan_future:
            first_person_plan = None
        other_plan = None if first_person_plan or non_plan_future else (
            _EXPLICIT_ENGLISH_OTHER_PLAN_RE.search(quote)
            or _EXPLICIT_CHINESE_OTHER_PLAN_RE.search(quote)
        )
        plan_subject = speaker_alias if first_person_plan else ""
        if other_plan:
            plan_subject = _speech_fact_subject(other_plan.group("subject"))
            if plan_subject in {"今天", "今晚", "明天", "后天", "这几天", "本周", "下周"}:
                other_plan = None
                plan_subject = ""
        if first_person_plan or other_plan:
            fact = {
                "subject_type": "person", "subject": plan_subject,
                "predicate": "stated_plan", "value": quote[:500],
                "modality": "speech", "status": "stated_plan",
                "confidence": "medium", "evidence": quote[:500],
                "evidence_ref": dict(evidence_ref),
            }
            time_scope = _explicit_plan_time_scope(quote)
            if time_scope:
                fact["time_scope"] = time_scope[:80]
            facts.append(fact)

        health_match = _EXPLICIT_ENGLISH_HEALTH_RE.search(quote)
        health_subject = speaker_alias if health_match else ""
        if not health_match:
            health_match = _EXPLICIT_CHINESE_HEALTH_RE.search(quote)
            health_subject = speaker_alias if health_match else ""
        if not health_match:
            health_match = _EXPLICIT_ENGLISH_OTHER_HEALTH_RE.search(quote)
            if health_match:
                health_subject = _speech_fact_subject(health_match.group("subject"))
        if not health_match:
            health_match = (_EXPLICIT_CHINESE_OTHER_HEALTH_RE.search(quote)
                            or _EXPLICIT_CHINESE_OTHER_ALLERGY_RE.search(quote))
            if health_match:
                health_subject = _speech_fact_subject(health_match.group("subject"))
        if not health_match and directed_health:
            health_match = directed_health
            health_subject = _speech_fact_subject(directed_health.group("name"))

        if health_match:
            urgent_current_report = bool(
                re.search(r"\b(?:i have|i(?:'m| am) having)\s+(?:a\s+)?(?:heart attack|stroke)\b",
                          quote, re.I)
                or re.search(r"\b(?:help me|i will die|i(?:'m| am) dying)\b", quote, re.I)
                or re.search(r"(?:救命|我快不行了|我要死了)", quote)
            )
            explicit_history = bool(
                re.search(r"\b(?:i had|i have had|i've had|i was diagnosed with|"
                          r"i have been diagnosed with|i have a history of)\b", quote, re.I)
                or re.search(r"(?:我以前|我曾经|患过|得过|被诊断为|确诊)", quote)
            )
            if directed_health:
                health_predicate = "spoken_health_claim"
                health_status = "unverified_claim"
            elif urgent_current_report:
                health_predicate = "self_reported_current_health_event"
                health_status = "self_reported_unverified"
            elif explicit_history:
                health_predicate = "self_reported_health_history"
                health_status = "self_reported"
            else:
                health_predicate = "spoken_health_claim"
                health_status = "self_reported_unverified" if health_subject == "unidentified speaker" else "unverified_claim"
            health_fact = {
                "subject_type": "person", "subject": health_subject,
                "predicate": health_predicate, "value": quote[:500],
                "modality": "speech", "status": health_status,
                "confidence": "medium", "evidence": quote[:500],
                "evidence_ref": dict(evidence_ref),
                "actor": speaker_alias,
            }
            if directed_health and health_subject:
                health_fact["affected_people"] = [health_subject]
            facts.append(health_fact)

        if len(facts) >= 16:
            break
    for instruction in instruction_rows:
        instruction_fact = {
            "subject_type": "event", "subject": "speech interaction",
            "predicate": "gave_spoken_instruction", "actor": "unidentified speaker",
            "affected_people": ["unidentified listener", instruction["term"]],
            "value": instruction["quote"][:500], "modality": "speech",
            "status": "explicit_instruction", "confidence": "medium",
            "evidence": instruction["quote"][:500],
            "evidence_ref": dict(instruction["evidence_ref"]),
            "time_order": len(facts) + 1,
        }
        facts.append(instruction_fact)
        for response in audio_rows:
            if response["start"] < instruction["end"] or response["start"] - instruction["end"] > 8.0:
                continue
            if not _EXPLICIT_SPOKEN_GREETING_RE.search(response["quote"]):
                continue
            term_re = re.escape(instruction["term"])
            direct_greeting = bool(re.search(
                term_re + r"(?:\s*(?:过年|新年|春节)?好|\s*拜年)", response["quote"], re.I))
            if instruction["term"] == "姥姥" and re.search(r"老老(?:过年|新年)?好", response["quote"]):
                direct_greeting = True
            if not direct_greeting:
                continue
            facts.append({
                "subject_type": "event", "subject": "speech interaction",
                "predicate": "instruction_then_targeted_greeting",
                "actor": "unidentified listener", "affected_people": [instruction["term"]],
                "value": (f"A later utterance greets {instruction['term']} after an explicit "
                          "instruction; voices are not diarized, so the speaker identity is unverified."),
                "modality": "speech", "status": "observed_sequence",
                "confidence": "medium",
                "evidence": f"Instruction: {instruction['quote']} | Following utterance: {response['quote']}"[:500],
                "evidence_ref": dict(response["evidence_ref"]),
                "evidence_refs": [dict(instruction["evidence_ref"]), dict(response["evidence_ref"])],
                "time_order": len(facts) + 1,
            })
            break
    return facts


def _summarize_explicit_speech_facts(facts, identity_links=None):
    """Summarize observable speech acts without assigning undiarized turns."""
    rows = [row for row in facts or [] if isinstance(row, dict)]
    links = (identity_links or {}).get("linked") or []
    visible_speaker = next((
        str(row.get("entity") or row.get("term") or "").strip()
        for row in links if row.get("usage") in {"self", "introduction"}
        and str(row.get("entity") or row.get("term") or "").strip()
    ), "")
    if visible_speaker:
        speaker = "The neighbor" if visible_speaker.casefold() in {"neighbor", "neighbour"} else (
            f"The person identified as {visible_speaker}")
    else:
        speaker = "The audio"
    events = []
    for row in rows:
        predicate = str(row.get("predicate") or "")
        ref = row.get("evidence_ref") if isinstance(row.get("evidence_ref"), dict) else {}
        try:
            start = float(ref.get("audio_start_s") or 0)
        except (TypeError, ValueError):
            start = 0.0
        targets = [str(value).strip() for value in row.get("affected_people") or []
                   if str(value).strip() and not str(value).startswith("unidentified")]
        target = targets[0] if targets else ""
        if predicate == "greeted":
            if target and visible_speaker:
                events.append((start, f"{speaker} greets {target}."))
            elif target:
                events.append((start, f"The audio records a greeting directed to {target}."))
            elif visible_speaker:
                events.append((start, f"{speaker} greets the listener."))
            else:
                events.append((start, "The audio records a greeting to the listener."))
        elif predicate == "asked_about_wellbeing":
            if target and visible_speaker:
                events.append((start, f"{speaker} asks {target} how they are."))
            elif target:
                events.append((start, f"The speaker asks {target} how they are."))
            elif visible_speaker:
                events.append((start, f"{speaker} asks how the listener is."))
            else:
                events.append((start, "The speaker asks the listener how they are."))
        elif predicate == "thanked":
            if target:
                events.append((start, f"The audio records a speaker thanking {target}."))
            else:
                events.append((start, "The audio records a speaker giving thanks."))
        elif predicate == "spoken_disfluency":
            events.append((start, "A following utterance contains repeated hesitation fillers; its speaker is not diarized."))
        elif predicate == "gave_spoken_instruction":
            events.append((start, "The audio contains an explicit instruction; the speaker and listener are not diarized."))
        elif predicate == "instruction_then_targeted_greeting":
            events.append((start, str(row.get("value") or "A later utterance follows an explicit greeting instruction.")))
    if not events:
        return ""
    # Keep one concise clause per event and avoid leaking an isolated transcript.
    seen = set()
    ordered = []
    for _time, sentence in sorted(events, key=lambda item: item[0]):
        key = sentence.casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(sentence)
    return " ".join(ordered)[:1000]
