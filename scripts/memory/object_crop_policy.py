from __future__ import annotations

import re


PERSONAL_ITEM_TERMS = (
    "wallet", "purse", "key", "keys", "phone", "mobile phone", "smartphone", "medication", "medicine", "pill bottle",
    "cane", "walker", "glasses", "spectacles", "watch", "passport", "id card",
    "identity card", "badge", "hearing aid", "prosthesis", "handbag", "backpack",
    "钱包", "钥匙", "手机", "药", "拐杖", "助行器", "眼镜", "手表",
    "护照", "身份证", "工牌", "助听器", "手提包", "背包",
)
ACTION_TERMS = (
    "hold", "held", "picked up", "pick up", "take", "took", "carry", "carried",
    "use", "used", "wear", "wore", "put", "placed", "move", "moved", "give",
    "gave", "receive", "received", "select", "selected", "choose", "chose",
    "purchase", "purchased", "buy", "bought", "drop", "dropped", "open", "opened",
    "close", "closed", "wearing", "拿起", "拿着", "使用", "穿戴", "放置", "移动",
    "交给", "收到", "选择", "购买", "掉落", "打开", "关闭",
)
LOW_CONFIDENCE = {"low", "uncertain", "unknown", "very_low"}


def _label(value):
    if isinstance(value, dict):
        value = value.get("name") or value.get("object") or value.get("entity") or ""
    return " ".join(str(value or "").split()).strip()


def _norm(value):
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", str(value or "").casefold()).strip()


def _same_entity(left, right):
    a, b = _norm(left), _norm(right)
    return bool(a and b and (a == b or (len(a) >= 4 and (a in b or b in a))))


def select_object_crop_candidates(objects, temporal_facts=()):
    """Keep crops for identified personal items with direct source-backed interaction or ownership."""
    facts = [row for row in temporal_facts or () if isinstance(row, dict)]
    eligible, skipped = [], []
    seen = set()
    for value in objects or ():
        label = _label(value)
        key = _norm(label)
        if not label or key in seen:
            continue
        seen.add(key)
        normalized_label = _norm(label)
        personalized = False
        for term in PERSONAL_ITEM_TERMS:
            normalized_term = _norm(term)
            if not normalized_term:
                continue
            if re.fullmatch(r"[a-z0-9 ]+", normalized_term):
                escaped = re.escape(normalized_term).replace("\\ ", r"\s+")
                if re.search(r"(?<![a-z0-9])" + escaped + r"s?(?![a-z0-9])",
                             normalized_label):
                    personalized = True
                    break
            elif normalized_term in normalized_label:
                personalized = True
                break
        explicitly_owned = isinstance(value, dict) and any(
            str(value.get(field) or "").strip()
            for field in ("owner", "owner_id", "owned_by", "personalized_for", "belongs_to")
        )
        linked_actions = []
        ownership_fact = False
        for fact in facts:
            if str(fact.get("subject_type") or "").casefold() != "object":
                continue
            if not _same_entity(fact.get("subject"), label):
                continue
            predicate = _norm(fact.get("predicate"))
            value_text = _norm(fact.get("value"))
            if any(term in predicate or term in value_text
                   for term in ("owned by", "belongs to", "personal property",
                                "belongs", "owned", "归属", "属于")):
                ownership_fact = True
            if str(fact.get("confidence") or "").casefold() in LOW_CONFIDENCE:
                continue
            action = _norm(fact.get("predicate"))
            action_text = _norm(fact.get("value"))
            is_action = any(_norm(term) in action or _norm(term) in action_text
                            for term in ACTION_TERMS)
            has_person_link = bool(str(fact.get("actor") or "").strip()
                                   or fact.get("affected_people"))
            if is_action and has_person_link:
                linked_actions.append(fact)
        if explicitly_owned or ownership_fact or (personalized and linked_actions):
            eligible.append(label)
        else:
            skipped.append({"object": label, "reason": "no_personal_or_event_link_evidence"})
    return eligible, skipped
