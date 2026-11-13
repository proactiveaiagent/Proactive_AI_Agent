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
COMPONENT_TERMS = ("drawer", "handle", "button", "lid", "cap", "zipper", "page", "screen", "display", "wheel", "knob", "抽屉", "把手", "按钮", "盖子", "拉链", "页面", "屏幕", "旋钮")
PARENT_OBJECT_TERMS = ("table", "desk", "cabinet", "dresser", "柜子", "桌子", "书桌")
TISSUE_TERMS = ("tissue", "napkin", "paper towel", "纸巾", "餐巾")
TRANSIENT_OR_CONSUMABLE_TERMS = (
    "chip", "chips", "snack", "food", "drink", "beverage", "candy", "gum",
    "tissue", "napkin", "paper towel", "price tag", "shelf tag", "sale tag",
    "sign", "warning sign", "poster", "advertisement", "ad", "packaging",
    "包装", "薯片", "零食", "食品", "饮料", "纸巾", "餐巾", "价签", "标价签",
    "警示牌", "告示牌", "海报", "广告",
)
DURABLE_PERSONAL_ITEM_TERMS = (
    "wallet", "purse", "key", "keys", "passport", "id card", "identity card",
    "badge", "hearing aid", "prosthesis", "cane", "walker", "personalized bag",
    "custom bag", "keepsake", "sentimental item", "钱包", "钥匙", "护照", "身份证",
    "工牌", "助听器", "义肢", "拐杖", "助行器", "纪念品",
)


def _label(value):
    if isinstance(value, dict):
        value = value.get("name") or value.get("object") or value.get("entity") or ""
    return " ".join(str(value or "").split()).strip()


def _norm(value):
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", str(value or "").casefold()).strip()


def _contains_term(text, term):
    text, term = _norm(text), _norm(term)
    if not text or not term:
        return False
    if re.search(r"[a-z0-9]", term):
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text))
    return term in text


def _same_entity(left, right):
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # Match complete English token sequences; keep Chinese phrase matching for
    # compound names, but avoid loose English substrings such as "door" in
    # unrelated text or "bill" in a currency label.
    if re.search(r"[a-z0-9]", a) or re.search(r"[a-z0-9]", b):
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(short) + r"(?![a-z0-9])", long))
    return a in b or b in a


def _person_names(person_observations):
    names = set()
    for row in person_observations or ():
        if not isinstance(row, dict):
            continue
        for value in (row.get("entity"), row.get("source_entity")):
            key = _norm(value)
            if key:
                names.add(re.sub(r"\s+\d+$", "", key))
        for alias in row.get("spoken_aliases") or ():
            value = alias.get("alias") if isinstance(alias, dict) else alias
            key = _norm(value)
            if key:
                names.add(re.sub(r"\s+\d+$", "", key))
    return names


def filter_person_entities_from_objects(objects, person_observations=()):
    """Drop object candidates that are aliases of visible people.

    Use whole normalized labels (and numbered person aliases) only, so a person
    called “baby” does not cause a real “baby stroller” object to disappear.
    """
    person_names = _person_names(person_observations)
    kept, removed = [], []
    for raw in objects or ():
        label = _label(raw)
        key = _norm(label)
        canonical = re.sub(r"\s+\d+$", "", key)
        (removed if key in person_names or canonical in person_names else kept).append(raw)
    return kept, removed


def evaluate_object_image_policy(objects, temporal_facts=(), document_evidence=(),
                                 source_evidence="", scene_context="",
                                 person_observations=()):
    """Decide separately whether to remember an object fact and retain an image.

    Text facts and OCR are retained independently from object images. Persistent
    portraits require a durable, user-linked identity need, explicit personalized
    uniqueness, or supported sentimental significance. Documents, consumables,
    price labels, advertisements, and temporary signs stay as textual/source
    evidence rather than long-term object portraits.
    """
    facts = [row for row in temporal_facts or () if isinstance(row, dict)]
    docs = [row for row in document_evidence or () if isinstance(row, dict)]
    speech = _norm(source_evidence)
    context = _norm(scene_context)
    decisions, seen = [], set()
    person_names = _person_names(person_observations)
    for raw in objects or ():
        label = _label(raw)
        key = _norm(label)
        if not label or key in seen:
            continue
        seen.add(key)
        if key in person_names or re.sub(r"\s+\d+$", "", key) in person_names:
            decisions.append({
                "object": label, "keep_fact": False, "keep_image": False,
                "reason_codes": ["person_entity_not_object"],
                "confidence": "high", "source_refs": [],
            })
            continue
        matching = []
        for fact in facts:
            if str(fact.get("subject_type") or "").casefold() != "object":
                continue
            target = fact.get("subject") or fact.get("object")
            if not _same_entity(target, label):
                continue
            action_detail = " ".join(str(fact.get(field) or "")
                                     for field in ("predicate", "value", "evidence"))
            # A parent item must not inherit an action performed on one of its
            # parts (e.g. “opened drawer” must not become “opened table”).
            component_action = (any(_contains_term(action_detail, term) for term in COMPONENT_TERMS)
                                and any(_same_entity(label, term) for term in PARENT_OBJECT_TERMS)
                                and not _same_entity(label, action_detail))
            if component_action:
                continue
            matching.append(fact)
        matching_docs = [doc for doc in docs if
                         _same_entity(doc.get("document") or doc.get("object") or
                                      doc.get("entity") or doc.get("label"), label)
                         or _same_entity(label, doc.get("document_type"))]
        document_terms = ("receipt", "invoice", "prescription", "document", "账单", "收据", "处方", "单据", "病历")
        currency_bill = bool(re.search(r"(?:[$€£]|\b\d+\s*(?:dollar|usd))", label.casefold()))
        explicit_document = any(_contains_term(label, term) for term in document_terms)
        if not matching_docs and explicit_document:
            matching_docs = [doc for doc in docs if str(doc.get("text") or "").strip()]
        if not matching_docs and _contains_term(label, "bill") and not currency_bill:
            matching_docs = [doc for doc in docs if str(doc.get("text") or "").strip()]
        evidence_text = " ".join(str(value or "") for fact in matching
                                 for value in (fact.get("predicate"), fact.get("value"),
                                               fact.get("evidence"), fact.get("actor")))
        evidence_norm = _norm(evidence_text)
        label_norm = _norm(label)
        reasons = []
        personal = any(_same_entity(label, term) for term in PERSONAL_ITEM_TERMS)
        if personal:
            reasons.append("unique_identity")
        explicitly_owned = isinstance(raw, dict) and any(
            str(raw.get(field) or "").strip()
            for field in ("owner", "owner_id", "owned_by", "personalized_for", "belongs_to"))
        ownership_fact = any(any(term in _norm(fact.get("predicate")) or
                                     term in _norm(fact.get("value"))
                                     for term in ("owned by", "belongs to", "personal property",
                                                  "belongs", "owned", "归属", "属于", "用户的"))
                              for fact in matching)
        if explicitly_owned or ownership_fact:
            reasons.append("personal_ownership")
        user_linked = any(
            str(fact.get("actor") or "").strip().casefold() in {"user", "camera wearer", "wearer", "用户"}
            or any(str(person).strip().casefold() in {"user", "camera wearer", "wearer", "用户"}
                   for person in (fact.get("affected_people") or []))
            for fact in matching)
        linked_action = any(
            any(_contains_term(" ".join((str(fact.get("predicate") or ""),
                                         str(fact.get("value") or ""))), term)
                for term in ACTION_TERMS)
            and str(fact.get("confidence") or "").casefold() not in LOW_CONFIDENCE
            for fact in matching)
        meaningful_interaction_terms = (
            "hold", "held", "pick up", "picked up", "take", "took", "carry", "carried",
            "use", "used", "select", "selected", "choose", "chose", "purchase",
            "purchased", "buy", "bought", "receive", "received", "give", "gave",
            "handed over", "offer", "offered", "show", "present", "wipe", "wipes",
            "fold", "folded", "clean", "cleaned", "consume", "consumed", "eat", "ate",
            "drink", "drank", "展示", "拿起", "拿着", "使用", "选择", "购买", "收到", "交给", "递给",
        )
        direct_scene_interaction = any(
            any(_contains_term(" ".join((str(fact.get("predicate") or ""),
                                         str(fact.get("value") or ""))), term)
                for term in meaningful_interaction_terms)
            and str(fact.get("confidence") or "").casefold() not in LOW_CONFIDENCE
            and bool(str(fact.get("actor") or "").strip() or fact.get("affected_people"))
            for fact in matching)
        # If the structured fact extractor attached an object's action to its
        # actor instead of the object, accept a narrative link only when that
        # object and action occur in the same sentence. A whole-clip match lets
        # background furniture borrow an unrelated payment, greeting, or handoff.
        scene_linked_action = False
        tissue_use = any(_same_entity(label, term) for term in TISSUE_TERMS)
        # Narrative fallback verbs must describe the candidate item itself.
        # Exclude broad “held/used” terms here because phrases like “camera
        # held by the user” otherwise make every nearby object look handled.
        scene_action_terms = (
            "pick up", "picked up", "take", "took", "select", "selected", "choose", "chose",
            "purchase", "purchased", "buy", "bought", "receive", "received", "handed over",
            "offer", "offered", "wipe", "wipes", "fold", "folded", "clean", "cleaned",
            "consume", "consumed", "eat", "ate", "drink", "drank", "打开", "拿起", "选择", "购买", "收到",
        )
        for text in (scene_context, source_evidence):
            for sentence in re.split(r"(?<=[.!?;])\s+|[\n\r]+", str(text or "")):
                if not _same_entity(label, sentence):
                    continue
                tissue_action_terms = ("hold", "held", "holding", "wipe", "wipes", "dab", "dabs",
                                       "blow nose", "擦", "捂", "擤") if tissue_use else ()
                if not any(_contains_term(sentence, term) for term in scene_action_terms + tissue_action_terms):
                    continue
                component_action = (any(_contains_term(sentence, term) for term in COMPONENT_TERMS)
                                    and any(_same_entity(label, term) for term in PARENT_OBJECT_TERMS))
                # A parent object does not inherit an action performed on a
                # drawer, handle, screen, or other separately named component.
                if component_action:
                    continue
                user_linked_sentence = any(_contains_term(sentence, term) for term in
                                           ("user", "camera wearer", "wearer", "用户", "recipient", "recipient's"))
                if user_linked_sentence or tissue_use:
                    scene_linked_action = True
                    break
            if scene_linked_action:
                break
        direct_scene_interaction = direct_scene_interaction or scene_linked_action
        def meaningful_location_change(fact):
            before = _norm(fact.get("from_location"))
            after = _norm(fact.get("to_location"))
            transient_states = {"open", "opened", "closed", "close", "shut", "unlocked", "locked"}
            return bool(before and after and before != after
                        and before not in transient_states and after not in transient_states)
        location_change = any(meaningful_location_change(fact) for fact in matching)
        state_change = any(bool(fact.get("from_state") and fact.get("to_state")
                                and _norm(fact.get("from_state")) != _norm(fact.get("to_state")))
                           for fact in matching)
        # A changed drawer/door state is not by itself a lasting personal memory.
        # Retain an image for a location/state transition when the object is
        # personally identifiable/owned; ordinary scene interaction stays fact-only.
        stateful = (location_change and (personal or explicitly_owned or ownership_fact)) or (
            state_change and (personal or explicitly_owned or ownership_fact))
        if personal and linked_action and user_linked:
            reasons.append("personal_interaction")
        if direct_scene_interaction:
            reasons.append("direct_scene_interaction")
        if stateful and (user_linked or explicitly_owned or ownership_fact):
            reasons.append("persistent_location_or_state")
        readable_document_fact = any(
            _contains_term(fact.get("predicate"), "documented")
            and len(str(fact.get("value") or fact.get("evidence") or "").strip()) >= 3
            for fact in matching)
        if matching_docs:
            readable = any(len(str(doc.get("text") or doc.get("evidence") or "").strip()) >= 3
                           for doc in matching_docs)
            if readable:
                reasons.append("readable_document")
        if readable_document_fact:
            reasons.append("readable_document")
        # Keep images for explicitly personal, lasting significance. Generic praise,
        # purchase language, ads, and food preferences remain searchable facts only.
        emotional_cues = ("gift", "given to me", "received as a gift", "favorite of mine",
                          "favourite of mine", "sentimental", "keepsake", "family heirloom",
                          "personally meaningful", "special to me", "my favorite", "my favourite",
                          "礼物", "送给我的", "我的珍藏", "纪念品", "有纪念意义", "对我很重要")
        emotional_linked = False
        for text in (evidence_text, source_evidence, scene_context):
            for sentence in re.split(r"(?<=[.!?;])\s+|[\n\r]+", str(text or "")):
                if _same_entity(label, sentence) and any(_contains_term(sentence, term)
                                                         for term in emotional_cues):
                    emotional_linked = True
                    break
            if emotional_linked:
                break
        if emotional_linked:
            reasons.append("emotional_salience")
        presentation_terms = ("present", "presents", "presenting", "presented", "promotion", "promotes", "promoting", "promote", "promotion", "demonstrate",
                              "show to camera", "sell", "selling", "推销", "展示给", "介绍产品")
        presentation_linked = False
        for text in (evidence_text, scene_context):
            for sentence in re.split(r"(?<=[.!?;])\s+|[\n\r]+", str(text or "")):
                if _same_entity(label, sentence) and any(_contains_term(sentence, term)
                                                        for term in presentation_terms):
                    presentation_linked = True
                    break
            if presentation_linked:
                break
        presentation_fact = any(
            _same_entity(fact.get("subject") or fact.get("object"), label)
            and any(_contains_term(" ".join(str(fact.get(field) or "")
                                            for field in ("predicate", "value", "evidence")), term)
                    for term in presentation_terms)
            for fact in matching)
        presentation_linked = presentation_linked or presentation_fact
        if presentation_linked:
            reasons.append("meaningful_presentation")
        measurement_only = (any(term in key for term in ("scale", "weighing scale", "体重计", "体重秤"))
                            and any(_norm(fact.get("predicate")) in {"displayed", "reads", "measured", "shows"}
                                    for fact in matching)
                            and not any(reason in reasons for reason in
                                        ("personal_ownership", "readable_document", "emotional_salience")))
        if measurement_only:
            reasons = [reason for reason in reasons if reason != "unique_identity"]
            reasons.append("measurement_only")
        # A category such as wallet/phone is only a candidate signal. Keep a
        # crop when the clip provides an actual user link, a consequential state,
        # readable document evidence, emotional meaning, or product presentation.
        # This avoids saving every incidental phone/scale while still preserving
        # a wallet whose placement/handling is grounded in the scene.
        durable_personal = any(_same_entity(label, term) for term in DURABLE_PERSONAL_ITEM_TERMS)
        personalized_item = isinstance(raw, dict) and any(
            str(raw.get(field) or "").strip()
            for field in ("personalized_for", "distinctive_identifier", "serial_number", "owner_id", "owned_by")
        )
        lasting_emotional_value = "emotional_salience" in reasons
        durable_user_link = durable_personal and (explicitly_owned or ownership_fact or
                                                   (user_linked and linked_action))
        unique_owned_item = personalized_item and (explicitly_owned or ownership_fact or user_linked)
        # Documents, consumables, price labels, advertisements, and temporary signs
        # still contribute OCR/event facts, but are evidence rather than durable
        # object portraits. Their source frames and document_evidence stay intact.
        document_like = any(_contains_term(label, term) for term in (
            "receipt", "invoice", "prescription", "document", "bill", "report", "病历", "收据", "处方", "单据", "文件"))
        transient_or_consumable = any(_contains_term(label, term)
                                      for term in TRANSIENT_OR_CONSUMABLE_TERMS)
        if document_like:
            reasons.append("document_evidence_only")
        elif transient_or_consumable:
            reasons.append("ordinary_consumable_or_temporary_object")
        if durable_user_link:
            reasons.append("durable_personal_identity")
        if unique_owned_item:
            reasons.append("personalized_unique_identity")
        keep_image = bool(durable_user_link or unique_owned_item or lasting_emotional_value) and not measurement_only and not document_like and not transient_or_consumable
        if personal and not keep_image and not any(code in reasons for code in (
                "document_evidence_only", "ordinary_consumable_or_temporary_object")):
            reasons.append("generic_category_without_durable_identity_evidence")
        if not reasons:
            if matching or any(_same_entity(doc.get("document") or doc.get("object") or "", label)
                               for doc in docs):
                reasons.append("background_or_low_future_value")
            else:
                reasons.append("no_personal_or_event_link_evidence")
        decisions.append({
            "object": label, "keep_fact": True, "keep_image": keep_image,
            "reason_codes": list(dict.fromkeys(reasons)),
            "confidence": "high" if any(code in reasons for code in
                                         ("readable_document", "personal_ownership")) else
                          "medium" if keep_image else "low",
            "source_refs": [fact.get("evidence_ref") for fact in matching
                            if isinstance(fact.get("evidence_ref"), dict)][:4],
        })
    return decisions


def select_object_crop_candidates(objects, temporal_facts=(), document_evidence=(),
                                  source_evidence="", scene_context=""):
    """Compatibility view returning image-worthy labels and explicit skip reasons."""
    decisions = evaluate_object_image_policy(
        objects, temporal_facts, document_evidence, source_evidence, scene_context)
    eligible = [row["object"] for row in decisions if row.get("keep_image")]
    skipped = [{"object": row["object"], "reason_codes": row.get("reason_codes") or [],
                "keep_fact": bool(row.get("keep_fact")), "keep_image": False,
                "confidence": row.get("confidence", "low"), "source_refs": row.get("source_refs") or []}
               for row in decisions if not row.get("keep_image")]
    return eligible, skipped
