"""
qwen_perceive.py — Qwen3-VL 感知客户端（记忆的上游）
======================================================================
职责边界：本模块只做「感知产出」——
    视频片段 → 抽帧 → Qwen3-VL → 结构化记忆条目
7 层记忆系统只负责存储 / 组织 / 检索 / 生命周期。

感知输入严格限定为视频抽帧。测试集的 transcript、caption、ground_truth 和其他描述
都不发送给 Qwen，也不作为记忆或来源证据；golden_eval() 只在推理完成后单独计算。

用法：
    p = QwenPerceiver()
    out = p.perceive(moment)              # 走缓存；无缓存则调真模型
    out = p.perceive(moment, force=True)  # 强制实时调用真模型
    p.prewarm(moments)                    # 批量预热缓存
"""

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import re
import threading
import time
import uuid
from pathlib import Path

import requests

_REQUESTS_POST_ORIGINAL = requests.post

def _post_with_inference_cancel(url, *args, **kwargs):
    """Cancel a server-side generation when its client request aborts or times out."""
    try:
        return _REQUESTS_POST_ORIGINAL(url, *args, **kwargs)
    except Exception:
        request_data = kwargs.get("data") or {}
        request_id = str(request_data.get("request_id") or "").strip()
        if request_id and str(url).rstrip("/").endswith("/analyze_video"):
            cancel_url = str(url).rsplit("/", 1)[0] + "/cancel_inference"
            try:
                _REQUESTS_POST_ORIGINAL(
                    cancel_url, json={"request_id": request_id}, timeout=(3, 8)
                )
            except Exception:
                pass
        raise

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "egolife" / "runs"
CACHE_FILE = RUNS / "qwen_cache.json"
SOURCE_FINGERPRINT_DB = RUNS / "source_video_fingerprints.sqlite3"
_SOURCE_FINGERPRINT_LOCK = threading.RLock()

DEFAULT_API = os.environ.get("QWEN_API_URL", "http://127.0.0.1:8001")
DEFAULT_FRAMES = 6
PROMPT_VERSION = "memory-v8.0-visible-text-atomic-evidence-2026-10-19"
LIST_PARSER_VERSION = "json-arrays-person-alias-visible-name-text-only-age-v6-2026-10-19"


# ---------------------------------------------------------------------------
# Prompt: turn ordered frames sampled from a video into searchable visual facts.
# ---------------------------------------------------------------------------

def build_prompt(n_frames: int) -> str:
    return f"""Create a detailed, source-grounded episodic memory using only these {n_frames} ordered frames sampled from one first-person video clip.

Use visible pixels only. Do not use transcript, subtitle files, captions, test annotations, reference answers, filenames, or dataset metadata as evidence. Do not invent names, age or age categories, dialogue, intent, plans, relationships, or facts not supported by the frames. Age words such as elderly, young, child, or baby are not visual traits and require explicit readable text. Read on-screen text only when clearly legible and directly relevant.

Understand the full clip as one coherent scene. Before answering, scan all frames in order and make an internal inventory of the visible people, their actions and interactions, important objects and their locations/states, the activity, setting, and any visible time/place clues. Record the scene as a short chronological account, not a bag of keywords. Preserve every distinct, clearly supported action or state change and who/what it affects. A person or unchanged object seen in several frames is one entity; use one stable, descriptive visual alias and summarize its visible interval. Do not omit an un-named visible person merely because the scene summary can be written without them. If identity continuity is unclear, keep observations separate. Never use numbered labels such as person 1/person 2 or generic labels as the stored identity.

Return exactly these English fields, each on its own line:
Scene: two to four concise chronological sentences describing the whole scene, people, actions, interactions, object changes, and outcome; quote short, important legible text only when it materially identifies a person, place, object, or event
Location: the broadest clearly supported place
Time/Occasion: only a date, clock, or occasion clearly readable in a supplied frame; otherwise blank
People: only literal names clearly readable in supplied-frame text and visibly linked to a specific person; otherwise leave blank. Never infer names from appearance, roles, filenames, or context
Person Observations: one JSON array with one row for each distinct, visually distinguishable person in the clip (up to 16). Include people in the background when their appearance is actually visible; do not collapse several distinguishable people into one row. Schema: {{"entity":"person wearing a blue padded jacket and gray headscarf","appearance":"hair, clothing, accessories, posture, and distinguishing visual traits","face_features":"clear visible face details only; otherwise empty","readable_name":"literal name from legible in-frame text directly linked to this person, or empty","name_evidence":"exact visible name text and where it appears, or empty","age":"exact age only when readable in-frame text links it to this person; otherwise empty","age_basis":"visible_text|unknown","frame_index":1,"evidence":"specific visible detail","confidence":"high|medium|low"}}. For unnamed people, have the model organize each person under a short, stable descriptive visual alias based on distinctive visible traits (clothing, accessories, clear facial details); start with person and describe what distinguishes them, never use numbered or generic placeholders. Prefer traits that remain visible across frames; use position only when it is clearly stable and needed to distinguish otherwise similar people. If frames do not support a distinction, do not invent one or claim separate identities. Describe each person once, combining stable appearance with useful visible posture/location. Put facial details only in face_features and do not duplicate them in appearance. Leave face_features empty when the face is too small, obscured, or unclear. Use the exact same descriptive alias in Scene, Interaction Details, Temporal Facts, affected_people, and actor wherever that person is referenced. Never estimate age or age category from appearance; record exact age only when legible on-screen text links it to this person. Use readable_name/name_evidence only for exact visible text directly linked to the person; otherwise leave both empty. Do not infer real-world identity, gender, role, or relationship from appearance.
User Action: visible actions performed by the camera wearer only; otherwise blank
Activity: the specific observable activity, its participants, roles only when directly supported, what each participant is doing, and how the activity develops; avoid vague labels
Objects: up to 12 important visible items and their locations/states, especially handled or moved items; transcribe short, clearly legible names, ages, dates, diagnoses, labels, or plans printed on a relevant item, explicitly as text displayed on that item
Environment: concise details of indoor/outdoor setting, layout, legible signs or landmarks (quote exact short wording and where it appears), lighting, weather, and background; omit unsupported guesses
Interaction Details: a chronological, semicolon-separated account of who did what to which person or object, including responses, transfers, and affected people; use the exact descriptive aliases from Person Observations
Temporal Facts: one JSON array of up to 16 separately retrievable facts. If any visible action, interaction, object handling/change, or meaningful change in posture/expression occurs, this array must not be empty. Record every distinct action/state change as its own atomic row with actor, affected entity, before/after state or location when shown, order, and specific frame evidence; do not collapse all actions into one row. A static clip with no separable action may use []. Schema: {{"subject_type":"person|object|place|environment|event","subject":"entity label","predicate":"appearance|state|located_at|moved|held|used|placed|picked_up|handed_over|participated_in|occurred_at|other","value":"one specific supported fact","actor":"person who visibly acted, if known","affected_people":["only people visibly affected by this fact"],"from_location":"only when visually supported","to_location":"only when visually supported","from_state":"only when visually supported","to_state":"only when visually supported","time_scope":"observed clip-time range when supported","time_order":1,"frame_index":1,"modality":"visual","evidence":"short description of visible pixels or legible text","status":"observed|completed|unclear","confidence":"high|medium|low"}}. Include distinct actions/interactions, important object states and transitions, activity, place, environmental conditions, and time context. Preserve actor, affected people, before/after state/location, and order. When a legible document, sign, label, menu, or screen carries important information, add a separate evidence-linked fact quoting the exact visible text and identify the source item; record displayed claims as text shown, not as independently verified truth. Use each person observation’s exact descriptive alias for person subject, actor, and affected_people; never use numbered placeholders. Do not repeat unchanged facts across frames. Do not claim an item moved unless the frames show the transition. Keep each fact atomic and evidence-linked. Return up to 16 high-value facts: prioritize actions and interactions, object changes, person details, and place/environment context; cover more than one category when the clip supports it instead of filling the list with repeated appearance facts. Keep each value and evidence phrase short, omit unsupported or empty optional fields, and fit all arrays completely within the response. Use the correct subject type: a moved item is an object, while the person is its actor.

For sequence questions, preserve only order visible across these frames. Do not infer speech, plans, diagnosis, needs, advice, reminders, names, age, or relationships. Do not infer a person’s identity, intent, or role from appearance. Do not guess obscured text or transform displayed medical claims into a diagnosis. Keep concise wording, but retain useful supported detail in structured observations and facts while keeping both arrays complete: person appearance at most 18 words, face_features at most 8 words, person evidence at most 12 words; each Temporal Facts value at most 20 words and evidence at most 12 words. Omit empty optional keys and do not repeat the scene in each fact. Before returning, check that every key action in Scene, User Action, and Interaction Details has a corresponding Temporal Facts row; actors and affected people use exact visual aliases; visible object locations/states and environment are represented; and every distinguishable person has a matching observation. People names must match name_evidence. Use [] for Temporal Facts only when no separable action or state change is visible. Use [] for Person Observations only when no human figure or identifiable body part is visible. Leave unsupported details blank and add no other text."""


_FIELD_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(Scene|Location|Time/Occasion|People|Person Observations|User Action|"
    r"Activity|Objects|Environment|Interaction Details|Temporal Facts)\s*:\s*(.*)$",
    re.I,
)


def _field(text: str, label: str) -> str:
    """Extract a field value, including continuation and bullet lines."""
    target = label.casefold()
    active = False
    parts = []
    for line in str(text or "").splitlines():
        match = _FIELD_LINE_RE.match(line)
        if match:
            if active:
                break
            if match.group(1).casefold() == target:
                active = True
                value = match.group(2).strip()
                if value:
                    parts.append(value)
            continue
        if active:
            value = line.strip()
            if value:
                # Preserve list-item boundaries; _split_list uses bullets as strong delimiters.
                parts.append("\n" + value if value.startswith(("- ", "* ")) else value)
    return " ".join(parts).strip()


def _replace_field_value(text: str, label: str, value: str) -> str | None:
    """Replace one labeled field and its continuation lines without touching other fields."""
    lines = str(text or "").splitlines()
    target = label.casefold()
    start = None
    end = len(lines)
    canonical = label
    for index, line in enumerate(lines):
        match = _FIELD_LINE_RE.match(line)
        if not match:
            continue
        if start is None:
            if match.group(1).casefold() == target:
                start = index
                canonical = match.group(1)
        else:
            end = index
            break
    if start is None:
        prefix = str(text or "").rstrip()
        return f"{prefix}\n{label}: {value}" if prefix else f"{label}: {value}"
    lines[start:end] = [f"{canonical}: {value}"]
    return "\n".join(lines)


def _empty_scalar(value: str) -> str:
    """Treat JSON empty-container/null markers as absent in prose fields."""
    text = str(value or "").strip()
    if text.casefold() in {"[]", "[ ]", "{}", "{ }", "null", "none", "undefined"}:
        return ""
    return text


def _split_top_level_list(text: str) -> list[str]:
    """Split comma/semicolon lists only outside quotes and bracketed descriptions."""
    parts, start = [], 0
    stack = []
    quote = None
    closing_quotes = {"“": "”", "‘": "’"}
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "“", "‘"}:
            quote = closing_quotes.get(char, char)
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if stack and char == stack[-1]:
            stack.pop()
            continue
        if not stack and char in ",，、;；":
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_list(raw: str, person: bool = False, preserve_commas: bool = False):
    """Parse JSON arrays when supplied; preserve narrative environment text."""
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        parts = [item for item in parsed if isinstance(item, str)]
    elif text.casefold() in {"[]", "[ ]", "none", "null", "n/a", "unknown", "-"}:
        return []
    elif re.search(r"(?:^|\n)\s*[-*]\s+", text):
        parts = [re.sub(r"^\s*[-*]\s+", "", line).strip()
                 for line in text.splitlines() if line.strip()]
    elif preserve_commas:
        parts = [text]
    else:
        parts = _split_top_level_list(text)
    out = []
    person_placeholders = {"other", "others", "other people", "people", "group",
                           "crowd", "someone", "somebody", "unknown person",
                           "unidentified person", "camera wearer",
                           "none identified", "none clearly identifiable",
                           "no one identifiable", "no one clearly identifiable",
                           "no identifiable people", "no people identified"}
    for item in parts:
        if not isinstance(item, str):
            continue
        if re.search(r"\b(implied|possibly|maybe|unclear|uncertain|inferred|guess(?:ed)?)\b", item, re.I):
            continue
        item = item.strip("[]{}\"' " + chr(96)).strip(".，。, ;；: ")
        if not item or item.casefold() in ("none", "n/a", "unknown", "null", "-"):
            continue
        if person and (item.casefold() in person_placeholders or
                       re.fullmatch(r"(?:one|two|three|several|many|a few) others?", item, re.I)):
            continue
        out.append(item)
    return out

_PERSON_INDEXED_LABEL_RE = re.compile(r"^(?:(?:unidentified|unknown)\s+)?person\s+\d+$", re.I)
_GENERIC_PERSON_LABELS = {"person", "unidentified person", "unknown person", "someone", "somebody"}
_PERSON_LABEL_SCHEMA_VERSION = 2


def _is_generic_person_label(value: object) -> bool:
    label = " ".join(str(value or "").casefold().split())
    return label in _GENERIC_PERSON_LABELS or bool(_PERSON_INDEXED_LABEL_RE.fullmatch(label))


def _visual_person_alias(row: dict) -> str:
    """Use the model's own visible description as a stable, searchable label."""
    appearance = " ".join(str(row.get("appearance") or "").split()).strip(" .;,:-")
    face = " ".join(str(row.get("face_features") or "").split()).strip(" .;,:-")
    if not appearance and not face:
        return ""
    descriptor = appearance or face
    if appearance and face and len(descriptor) < 80:
        descriptor = f"{appearance}; {face}"
    descriptor = descriptor[:180].rstrip(" .;,:-")
    if re.match(r"^(?:person\b|someone\b|somebody\b)", descriptor, re.I):
        alias = descriptor
    elif re.match(r"^(?:wearing|with|in|has|holding|standing|sitting|seated|walking)\b", descriptor, re.I):
        alias = "person " + descriptor
    else:
        alias = "person with " + descriptor
    return alias[:200].rstrip(" .;,:-")


def _organize_person_entity_aliases(fields: dict) -> dict:
    """Replace old numbered labels with aliases from Qwen's visual descriptions."""
    if not isinstance(fields, dict):
        return fields
    rows = [dict(row) for row in fields.get("person_observations") or []
             if isinstance(row, dict)]
    mapping = {}
    for row in rows:
        original = " ".join(str(row.get("entity") or "").split())
        if not _is_generic_person_label(original):
            continue
        alias = _visual_person_alias(row)
        if alias:
            mapping[original] = alias
            row["entity"] = alias
    if not mapping:
        return fields

    def remap_tree(value):
        if isinstance(value, str):
            result = value
            for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
                result = re.sub(r"(?<![\w])" + re.escape(old) + r"(?![\w])",
                                lambda _match, replacement=new: replacement,
                                result, flags=re.I)
            return result
        if isinstance(value, list):
            return [remap_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: remap_tree(item) for key, item in value.items()}
        return value

    organized = remap_tree(dict(fields))
    organized["person_observations"] = rows
    organized["person_label_schema_version"] = _PERSON_LABEL_SCHEMA_VERSION
    return organized


_INFERRED_AGE_CATEGORY_RE = re.compile(
    r"\b(?:elderly|elder|young|younger|older|middle[- ]aged|teenage|teenager|"
    r"adolescent|adult|senior|mature)\b|\bold(?=\s+(?:person|man|woman|"
    r"people|men|women|child|children|individual)\b)", re.I
)
_PERSON_AGE_NOUN_RE = re.compile(
    r"\b(?:baby|babies|infant|infants|toddler|toddlers|newborn|newborns|"
    r"child|children|girl|girls|boy|boys|teen|teens|adolescent|adolescents|"
    r"adult|adults|senior|seniors|man|men|woman|women)\b", re.I
)
_AGE_RANGE_RE = re.compile(
    r"\b(?:in\s+)?(?:his|her|their)\s+(?:(?:early|mid|late)\s+)?\d{2}s\b|"
    r"\b\d{1,3}[- ]year[- ]old\b|"
    r"\b\d{1,3}\s+years?\s+old\b|\baged\s+\d{1,3}\b", re.I
)

def _strip_unverified_age_categories(fields: dict) -> dict:
    """Keep age only in the text-backed age field, never as a visual guess."""
    def clean(value, key=""):
        if key in {"age", "age_basis", "readable_name", "name_evidence"}:
            return value
        if isinstance(value, str):
            text = _AGE_RANGE_RE.sub(" ", value)
            text = _INFERRED_AGE_CATEGORY_RE.sub("", text)
            plural_person_terms = {
                "babies", "infants", "toddlers", "newborns", "children", "girls",
                "boys", "teens", "adolescents", "adults", "seniors", "men", "women",
            }
            text = _PERSON_AGE_NOUN_RE.sub(
                lambda match: "people" if match.group(0).casefold() in plural_person_terms else "person",
                text,
            )
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\s+([,.;:!?])", r"\1", text)
            text = re.sub(r"([,;:])\s*([,;:])", r"\1 \2", text)
            text = re.sub(r"\bAn\s+(?=(?:woman|person)\b)", "A ", text, flags=re.I)
            text = re.sub(r"\ba\s+people\b", "people", text, flags=re.I)
            return text.strip()
        if isinstance(value, list):
            return [clean(item, key) for item in value]
        if isinstance(value, dict):
            return {name: clean(item, name) for name, item in value.items()}
        return value
    return clean(fields)


def parse_analysis(text: str, strict: bool = False) -> dict:
    """Qwen 输出 → 结构化记忆条目（对齐 memory.add 的参数）。"""
    location = _empty_scalar(_field(text, "Location"))
    people = _split_list(_field(text, "People"), person=True)
    action = _empty_scalar(_field(text, "User Action"))
    for marker in ("none specified", "none", "not specified", "n/a", "unknown"):
        if action.casefold().startswith(marker):
            action = action[len(marker):].lstrip(" .,:;")
            break
    prefix = "interaction details:"
    if action.casefold().startswith(prefix):
        action = action[len(prefix):].lstrip()
    interaction = _empty_scalar(_field(text, "Interaction Details"))
    activity = _empty_scalar(_field(text, "Activity"))
    objects = _split_list(_field(text, "Objects"))
    envs = _split_list(_field(text, "Environment"), preserve_commas=True)

    def json_rows(label, limit):
        try:
            parsed = json.loads(_field(text, label))
        except (TypeError, ValueError) as exc:
            if not strict:
                return []
            raise ValueError(f"{label} is not a complete JSON array") from exc
        if not isinstance(parsed, list):
            if not strict:
                return []
            raise ValueError(f"{label} must be a JSON array")
        allowed = {
            "person": {"entity", "appearance", "face_features", "readable_name", "name_evidence", "age", "age_basis", "frame_index", "evidence", "confidence"},
            "fact": {"subject_type", "subject", "predicate", "value", "actor", "affected_people", "from_location", "to_location", "from_state", "to_state", "time_scope", "time_order", "frame_index", "modality", "evidence", "status", "confidence"},
        }["person" if label == "Person Observations" else "fact"]
        out = []
        for row in parsed[:limit]:
            if not isinstance(row, dict):
                continue
            clean = {}
            for key in allowed:
                value = row.get(key)
                if key in {"frame_index", "time_order"}:
                    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                        clean[key] = min(value, 999)
                elif key == "affected_people" and isinstance(value, list):
                    placeholders = {"none", "null", "unknown", "n/a", "na", "not specified", "no one"}
                    clean[key] = list(dict.fromkeys(
                        item.strip()[:120] for item in value
                        if isinstance(item, str) and item.strip()
                        and item.strip().casefold() not in placeholders))[:8]
                elif isinstance(value, str) and value.strip():
                    if value.strip().casefold() in {
                            "none", "null", "unknown", "n/a", "na", "not specified"}:
                        continue
                    clean[key] = value.strip()[:500]
            if label == "Person Observations":
                age_text = str(clean.get("age") or "").strip()
                age_basis = str(clean.get("age_basis") or "").casefold()
                try:
                    age_value = int(age_text)
                except ValueError:
                    age_value = -1
                if age_basis != "visible_text" or not 0 <= age_value <= 120:
                    clean.pop("age", None)
                    clean.pop("age_basis", None)
                else:
                    clean["age"] = str(age_value)
                    clean["age_basis"] = "visible_text"
            if clean.get("entity") or clean.get("subject"):
                out.append(clean)
        if label == "Person Observations":
            for row in out:
                if not str(row.get("readable_name") or "").strip() or not str(row.get("name_evidence") or "").strip():
                    row.pop("readable_name", None)
                    row.pop("name_evidence", None)
        return out

    person_observations = json_rows("Person Observations", 16)
    verified_names = {
        " ".join(str(row.get("readable_name") or "").casefold().split())
        for row in person_observations if row.get("readable_name") and row.get("name_evidence")
    }
    people = [value for value in people
              if " ".join(str(value or "").casefold().split()) in verified_names]

    result = {
        # Store the event narrative separately from the location so search can find facts.
        "scene": _empty_scalar(_field(text, "Scene")) or location,
        "user_action": action,
        "interaction_details": interaction,
        "person_observations": person_observations,
        "temporal_facts": json_rows("Temporal Facts", 16),
        # Provenance is attached by the runner after perception. Never copy
        # benchmark-side transcript text into model output or stored memory.
        "source_evidence": "",
        "people": people,
        "location": location or None,
        "activity": activity or None,
        "environments": envs or None,
        "objects": objects or None,
        "time_occasion": _empty_scalar(_field(text, "Time/Occasion")),
        "needs": [],
        "solutions": [],
    }
    return _strip_unverified_age_categories(
        _normalize_camera_wearer_actor_aliases(_organize_person_entity_aliases(result))
    )


def _camera_wearer_reference(value: object) -> bool:
    key = " ".join(str(value or "").casefold().replace("-", " ").split())
    key = re.sub(r"^(?:the|a|an)\s+", "", key)
    return key == "camera wearer"


def _normalize_camera_wearer_actor_aliases(fields: dict) -> dict:
    """Keep first-person actions under the camera-wearer role without inventing a portrait."""
    if not isinstance(fields, dict):
        return fields
    explicit_wearer = any(
        re.search(r"\bcamera[- ]wearer\b", str(fields.get(key) or ""), re.I)
        for key in ("user_action", "interaction_details")
    )
    if not explicit_wearer:
        return fields
    generic = {"person", "a person", "the person", "unidentified person", "someone"}
    facts = []
    mapped = set()
    wearer_seen = False
    for original in fields.get("temporal_facts") or []:
        if not isinstance(original, dict):
            facts.append(original)
            continue
        row = dict(original)
        actor = str(row.get("actor") or "").strip()
        key = " ".join(actor.casefold().split())
        subject_type = str(row.get("subject_type") or "").casefold()
        if _camera_wearer_reference(actor):
            if actor != "camera wearer":
                row["actor"] = "camera wearer"
            wearer_seen = True
        elif key in generic and subject_type == "object":
            # The user_action explicitly attributes this object interaction to the
            # wearer; use that role instead of an anonymous person alias.
            row["actor"] = "camera wearer"
            mapped.add(key)
            wearer_seen = True
        facts.append(row)
    if not wearer_seen:
        return fields
    updated = dict(fields)
    updated["temporal_facts"] = facts
    people = []
    seen_people = set()
    for value in fields.get("people") or []:
        label = str(value or "").strip()
        key = " ".join(label.casefold().split())
        if _camera_wearer_reference(label) or key in mapped:
            label = "camera wearer"
        norm = " ".join(label.casefold().split())
        if label and norm not in seen_people:
            people.append(label)
            seen_people.add(norm)
    if "camera wearer" not in seen_people:
        people.append("camera wearer")
    updated["people"] = people

    # Rewrite repeated singular references to the same POV actor only when no
    # other person or group is described. "First-person perspective" is camera
    # terminology, not a visible-human observation.
    if not (fields.get("person_observations") or []):
        narrative_keys = ("scene", "activity", "interaction_details", "user_action")
        narrative = " ".join(str(fields.get(key) or "") for key in narrative_keys)
        other_person = re.search(
            r"\b(?:another|other|background|group|several|multiple|people|crowd|"
            r"two|three|four|man|woman|child|children|passenger|customer|worker)\b",
            narrative, re.I,
        )
        without_pov_phrase = re.sub(
            r"\bfirst[- ]person(?:\s+perspective)?\b", " ", narrative, flags=re.I
        )
        person_mentions = re.findall(
            r"\b(?:person|persons|people|man|woman|child|children)\b",
            without_pov_phrase, re.I,
        )
        if not other_person and person_mentions:
            for key in ("scene", "activity", "interaction_details"):
                value = str(updated.get(key) or "")
                value = re.sub(
                    r"\b(?:a|the) person\b", "the camera wearer", value,
                    flags=re.I,
                )
                updated[key] = re.sub(
                    r"\bfirst[- ]person(?:\s+perspective)?\b",
                    "camera POV", value, flags=re.I,
                )
    return updated

def _person_observation_references(fields: dict) -> list[str]:
    """Return person names and model-generated visual aliases referenced by facts.

    Actor values are expected to be people under the extraction schema; skip
    values that clearly name an object or place so malformed model output does
    not create a false person.
    """
    people = [str(value).strip() for value in fields.get("people") or []
              if str(value or "").strip()]
    explicit = {" ".join(value.casefold().split()) for value in people}
    labels = []
    seen = set()

    def add(value):
        text = " ".join(str(value or "").split())
        key = text.casefold()
        if not text or _camera_wearer_reference(text) or key in seen:
            return
        seen.add(key)
        labels.append(text)

    for value in people:
        add(value)
    for row in fields.get("person_observations") or []:
        if isinstance(row, dict):
            add(row.get("entity"))
    for row in fields.get("temporal_facts") or []:
        if not isinstance(row, dict):
            continue
        is_person_fact = str(row.get("subject_type") or "").casefold() == "person"
        if is_person_fact:
            add(row.get("subject"))
            add(row.get("actor"))
        actor = str(row.get("actor") or "").strip()
        actor_key = " ".join(actor.casefold().split())
        non_person_labels = {
            " ".join(str(value or "").casefold().split())
            for value in (fields.get("objects") or []) + (fields.get("environments") or [])
            if str(value or "").strip()
        }
        non_person_labels.update({
            " ".join(str(value or "").casefold().split())
            for value in (fields.get("location"), fields.get("activity"))
            if str(value or "").strip()
        })
        if actor and actor_key not in non_person_labels:
            add(actor)
        for value in row.get("affected_people") or []:
            add(value)
    return labels


_VISIBLE_HUMAN_CONTEXT_RE = re.compile(
    r"\b(?:people|person|persons|humans?|human figures?|men|women|children|boys|girls|"
    r"man|woman|child|boy|girl|bab(?:y|ies)|infants?|toddlers?|newborns?|"
    r"individuals?|someone|somebody|couple|crowd|family|friends|passengers?|attendees?|"
    r"customers?|workers?|staff|guests?|travelers?|tourists?|commuters?|pedestrians?|elderly|teenagers?)\b", re.I,
)
_VISIBLE_PLURAL_GROUP_RE = re.compile(
    r"\b(?:people|persons|humans|men|women|children|boys|girls|crowd)\b|"
    r"\b(?:several|multiple|many|various)\s+(?:other\s+)?"
    r"(?:people|persons|individuals|participants|travelers|tourists|commuters|pedestrians|men|women|children|boys|girls)\b|"
    r"\bgroup of (?:visible )?(?:people|persons|individuals)\b|"
    r"\b(?:both|two|three|four|five|six|\d+)\s+(?:visible\s+)?"
    r"(?:people|persons|men|women|children|boys|girls|participants|travelers|tourists|commuters|pedestrians)\b|"
    r"\b(?:a couple|the couple|others?)\s+(?:sit|sits|stand|stands|walk|walks|"
    r"talk|talks|gather|gathers|interact|interacts|are|were)\b",
    re.I,
)
_VISIBLE_PERSON_COUNT_RE = re.compile(
    r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|a pair of|a couple of)"
    r"\s+(?:visible\s+)?(?:people|persons?|humans?|travelers?|tourists?|commuters?|pedestrians?|men|women|children|boys|girls|"
    r"babies|infants|toddlers|newborns)\b", re.I,
)
_PERSON_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a pair of": 2, "a couple of": 2,
}


def _visible_person_narrative_error(fields: dict, observations: list[dict]) -> str:
    """Reject narrative claims of visible people when visual records are missing."""
    narrative = " ".join(
        str(fields.get(key) or "")
        for key in ("scene", "activity", "interaction_details", "user_action")
    )
    if not narrative.strip():
        return ""
    positive_narrative = re.sub(
        r"\b(?:no|not any|none of the)\s+(?:visible\s+)?(?:people|persons|humans)\b",
        " ", narrative, flags=re.I,
    )
    if not observations and _VISIBLE_HUMAN_CONTEXT_RE.search(positive_narrative):
        return ("Scene, Activity, or Interaction Details describe a visible person/group, "
                "but Person Observations is empty; inspect the supplied frames and add "
                "source-grounded visual records.")
    count_match = _VISIBLE_PERSON_COUNT_RE.search(positive_narrative)
    if count_match:
        count_text = " ".join(count_match.group("count").casefold().split())
        expected = int(count_text) if count_text.isdigit() else _PERSON_COUNT_WORDS[count_text]
        if expected <= 16 and len(observations) < expected:
            return (f"The narrative identifies {expected} visible people but only "
                    f"{len(observations)} Person Observations rows are present.")
    if len(observations) < 2 and _VISIBLE_PLURAL_GROUP_RE.search(positive_narrative):
        return ("The narrative describes multiple visible people, but fewer than two "
                "Person Observations rows are present; inspect all supplied frames.")
    if (len(observations) < 2 and re.search(
            r"\b(?:another|second|other)\s+(?:person|man|woman|child|boy|girl)\b|"
            r"\b(?:man|woman|person|child|boy|girl)\b.{0,80}\band\s+"
            r"(?:a|the|another)\s+(?:man|woman|person|child|boy|girl)\b",
            positive_narrative, re.I)):
        return "The narrative distinguishes multiple people but Person Observations does not."
    return ""


def _person_observation_coverage_error(fields: dict) -> str | None:
    """Check that explicitly person-typed references have visible observations.

    This checks video-only model output. It does not classify ambiguous actors
    on object/place events as people or invent appearance descriptions.
    """
    observations = [row for row in fields.get("person_observations") or []
                    if isinstance(row, dict) and str(row.get("entity") or "").strip()]
    visible_body_part = re.compile(r"\b(?:hand|hands|palm|finger|fingers|forearm|arm|wrist|foot|feet)\b", re.I)
    narrative = " ".join(str(fields.get(key) or "") for key in
                         ("scene", "user_action", "interaction_details", "activity"))
    camera_context_visible = bool(re.search(
        r"\b(?:camera[- ]wearer|first[- ]person\s+(?:view|perspective|camera|footage|POV)|POV)\b",
        narrative, re.I,
    ))
    camera_wearer_body_visible = (
        camera_context_visible and bool(visible_body_part.search(narrative))
    )
    for fact in fields.get("temporal_facts") or []:
        if not isinstance(fact, dict) or not _camera_wearer_reference(fact.get("actor")):
            continue
        evidence = " ".join(str(fact.get(key) or "") for key in
                            ("evidence", "value", "from_location", "to_location",
                             "from_state", "to_state"))
        if visible_body_part.search(evidence):
            camera_wearer_body_visible = True
            break
    camera_wearer_observed = any(
        re.match(r"^(?:the\s+)?camera[- ]wearer\b", str(row.get("entity") or ""), re.I)
        and visible_body_part.search(" ".join(str(row.get(key) or "") for key in
                                               ("entity", "appearance", "evidence")))
        for row in observations
    )
    if camera_wearer_body_visible and not camera_wearer_observed:
        return ("The camera wearer's visible hand or other body part performs an action, "
                "but no source-grounded partial-body observation is stored; inspect the supplied frames.")
    observed = {" ".join(str(row.get("entity") or "").casefold().split()): row
                for row in observations}
    missing = []
    for entity in _person_observation_references(fields):
        # The camera wearer is a first-person role and may be outside the frame;
        # retain their event references without requiring an impossible visual crop.
        if _camera_wearer_reference(entity):
            continue
        key = " ".join(str(entity or "").casefold().split())
        if key and key not in observed and key not in missing:
            missing.append(key)
    if missing:
        return "Temporal Facts or People names refer to people without matching Person Observations: " + ", ".join(missing[:8])

    narrative_error = _visible_person_narrative_error(fields, observations)
    if narrative_error:
        return narrative_error

    numbered = [str(row.get("entity") or "") for row in observations
                if _is_generic_person_label(row.get("entity"))]
    if numbered:
        return "Person Observations still uses generic or numbered labels; replace them with visual descriptions: " + ", ".join(numbered[:8])

    blank_appearance = [str(row.get("entity")) for row in observations
                        if not str(row.get("appearance") or "").strip()]
    if blank_appearance:
        return "Person Observations has an entity without a visible appearance description: " + ", ".join(blank_appearance[:8])
    return ""


def _person_reference_labels(fields: dict) -> list:
    """Collect explicit person references for a focused video-only repair pass."""
    return _person_observation_references(fields)[:16]


def _build_person_repair_prompt(context: dict, reference_labels: list) -> str:
    """Ask the video-only repair pass for stable visual aliases and exact mappings."""
    payload = dict(context)
    payload["person_references_to_resolve"] = reference_labels
    return (
        "Use only the attached images extracted from this source video. The JSON context below was "
        "generated earlier from these same images; it is not ground truth. Independently inspect every supplied "
        "frame and enumerate distinct, visually distinguishable people, including background people when "
        "their appearance is clear. An empty person_references_to_resolve list does NOT mean no person is visible. "
        "Create one Person Observations row per distinguishable person across the clip, not one per frame. "
        "Also record a camera wearer when the first-person view visibly shows their hands, forearms, feet, or "
        "other body parts doing an action. Use a descriptive alias such as 'camera wearer with a gray-gloved hand' based on the visible body part, not the role alone, and describe "
        "only the visible body parts; explicitly say the face/full body is outside the sampled view when applicable. "
        "Do not turn an off-screen actor with no visible body part into a visual observation. "
        "Verify every match against visible pixels. Use a stable descriptive visual alias based on distinctive "
        "visible clothing, accessories, or clear facial details; never number people or invent real names. Add readable_name and name_evidence only for exact legible in-frame text visibly linked to the person; otherwise leave both empty. Do not infer age categories from appearance. "
        "Reuse the same alias for the same visible person. Add an entity_map row from an existing reference "
        "to the new alias only when frame evidence clearly supports the match; omit uncertain mappings. Do not "
        "infer real identity, age, gender, job, relationship, or plans. Put clearly visible facial details in "
        "face_features; describe hair, body, clothing, accessories, and posture in appearance. Record age only "
        "when legible on-screen text explicitly links the age to that person, marked age_basis=visible_text; "
        "otherwise leave age blank. Include a 1-based sampled-frame index and visible evidence. Use an empty "
        "array only if no human figure is visible. Return one JSON object with person_observations and entity_map, "
        'JSON only. Schema: {"person_observations":[{"entity":"person wearing a blue padded jacket and gray headscarf",'
        '"appearance":"wearing a blue padded jacket and gray headscarf","face_features":"clear visible facial details, or empty",'
        '"readable_name":"","name_evidence":"","age":"","age_basis":"unknown","frame_index":1,'
        '"evidence":"visible detail in frame 1","confidence":"high|medium|low"}], '
        '"entity_map":[{"reference":"person wearing a blue jacket","entity":"person wearing a blue padded jacket and gray headscarf"}]}. Context: '
        + json.dumps(payload, ensure_ascii=True)
    )


def _parse_person_repair(raw: str) -> dict:
    """Parse a focused person-observation response without accepting prose."""
    text = str(raw or "").strip()
    fence = chr(96) * 3
    text = re.sub(r"^\s*" + fence + r"(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*" + fence + r"\s*$", "", text)
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("人物补全结果不是 JSON 对象或数组")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError:
        # A truncated outer object can still contain complete, source-grounded
        # observation rows. Recover only complete JSON values, then run the same
        # strict evidence and coverage checks before any cache write.
        key_pos = text.find('"person_observations"')
        array_start = text.find("[", key_pos) if key_pos >= 0 else text.find("[")
        rows, _complete = _parse_json_array_prefix(text[array_start:]) if array_start >= 0 else ([], False)
        if not rows:
            raise
        return {"person_observations": rows}
    if isinstance(payload, list):
        return {"person_observations": payload}
    if not isinstance(payload, dict):
        raise ValueError("人物补全结果不是 JSON 对象或数组")
    return payload


def _parse_json_array(raw: str) -> list:
    text = str(raw or "").strip()
    fence = chr(96) * 3
    text = re.sub(r"^\s*" + fence + r"(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*" + fence + r"\s*$", "", text)
    start = text.find("[")
    if start < 0:
        raise ValueError("temporal fact repair did not return a JSON array")
    payload, end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, list):
        raise ValueError("temporal fact repair must return a JSON array")
    if text[start + end:].strip():
        raise ValueError("temporal fact repair returned text after the JSON array")
    return payload


def _parse_json_array_prefix(raw: str) -> tuple[list, bool]:
    """Read complete JSON values from the start of an array, tolerating a truncated tail."""
    text = str(raw or "").strip()
    fence = chr(96) * 3
    text = re.sub(r"^\s*" + fence + r"(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*" + fence + r"\s*$", "", text)
    start = text.find("[")
    if start < 0:
        return [], False
    decoder = json.JSONDecoder()
    cursor = start + 1
    rows = []
    while True:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            return rows, False
        if text[cursor] == "]":
            return rows, not text[cursor + 1:].strip()
        try:
            value, end = decoder.raw_decode(text, cursor)
        except (ValueError, json.JSONDecodeError):
            return rows, False
        rows.append(value)
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            return rows, False
        if text[cursor] == ",":
            cursor += 1
            continue
        if text[cursor] == "]":
            return rows, not text[cursor + 1:].strip()
        return rows, False


def _recover_truncated_temporal_fact_prefix(
    raw: str, frame_count: int, validate_source_fact_quality: bool = True
):
    """Recover complete video-grounded facts before a malformed array tail.

    Keep the exact model response for audit. When a complete fact prefix refers
    to a person without a visible observation, remove only those unsupported
    person references and retain the independently supported scene/facts.
    """
    source_rows, complete = _parse_json_array_prefix(_field(raw, "Temporal Facts"))
    if complete or not source_rows:
        return None
    rows = _apply_temporal_fact_repair(source_rows, frame_count)
    if not rows:
        return None
    base_raw = _replace_field_value(raw, "Temporal Facts", "[]")
    if base_raw is None:
        return None
    parse_analysis(base_raw, strict=True)
    patched_raw = _replace_field_value(
        base_raw, "Temporal Facts", json.dumps(rows, ensure_ascii=False)
    )
    if patched_raw is None:
        return None
    fields = parse_analysis(patched_raw, strict=True)
    repair = {}
    if validate_source_fact_quality:
        fields, removed_moves = _drop_unverified_object_moves(fields)
        if removed_moves:
            repair.update({
                "method": "drop_unverified_object_moves",
                "removed_count": len(removed_moves),
                "removed_facts": removed_moves,
            })
        person_error = _person_observation_coverage_error(fields)
        move_error = _object_move_evidence_error(fields)
        if person_error and not move_error:
            sanitized, removed_people = _salvage_person_reference_gaps(fields)
            if sanitized is None:
                raise ValueError(person_error)
            fields = sanitized
            if removed_people:
                repair["person_reference_repair"] = {
                    "method": "drop_unverified_person_references",
                    "removed_count": len(removed_people),
                    "removed_items": removed_people,
                    "raw_model_output_preserved": True,
                }
        quality_error = _structured_fact_quality_error(fields)
        if quality_error:
            raise ValueError(quality_error)
    rows = list(fields.get("temporal_facts") or [])
    return raw, fields, rows, (repair or None)

def _build_temporal_fact_repair_prompt(frame_count: int) -> str:
    """Keep the focused fallback compact enough to return a complete grounded array."""
    return (
        "Inspect only the supplied ordered video frames. Return exactly one complete JSON array "
        "of up to 8 distinct, independently useful visual facts. Prioritize visible person "
        "appearance and interaction, important object states or locations, environment, and event "
        "order. Keep each value and evidence phrase to one short sentence; do not repeat the same "
        "fact. Use only frame-supported facts. Do not infer identity, exact age, relationships, "
        "plans, dialogue, diagnosis, advice, or anything outside the frames. Every row must include "
        "subject_type (person|object|place|environment|event), subject, predicate, value, a 1-based "
        "sampled frame_index from 1 to " + str(max(1, int(frame_count))) + ", and concise visible-pixel "
        "evidence. For an object predicate moved, include visibly supported from_location and "
        "to_location or from_state and to_state; otherwise do not describe it as moved. Omit empty "
        "optional fields. Use English only, return [] if no useful fact is visible, and do not add "
        "prose. Schema: [{\"subject_type\":\"object\",\"subject\":\"phone\","
        "\"predicate\":\"used\",\"value\":\"A person taps the phone on the table.\","
        "\"actor\":\"person wearing a blue jacket\",\"affected_people\":[],"
        "\"frame_index\":1,\"time_order\":1,"
        "\"evidence\":\"A hand touches the phone in frame 1.\","
        "\"confidence\":\"medium\"}]"
    )


def _apply_temporal_fact_repair(rows: list, frame_count: int) -> list:
    allowed_types = {"person", "object", "place", "environment", "event"}
    if len(rows) > 8:
        raise ValueError("temporal fact repair returned more than 8 facts")
    output = []
    rejected = 0
    for original in rows:
        if not isinstance(original, dict):
            rejected += 1
            continue
        subject_type = str(original.get("subject_type") or "").strip().casefold()
        subject = str(original.get("subject") or "").strip()
        predicate = str(original.get("predicate") or "").strip()
        value = str(original.get("value") or "").strip()
        evidence = str(original.get("evidence") or "").strip()
        try:
            frame_index = int(original.get("frame_index"))
        except (TypeError, ValueError):
            frame_index = 0
        if (subject_type not in allowed_types or not subject or not predicate or not value
                or not evidence or isinstance(original.get("frame_index"), bool)
                or frame_index < 1 or frame_index > max(1, frame_count)):
            rejected += 1
            continue
        fact = {
            "subject_type": subject_type,
            "subject": subject[:120],
            "predicate": predicate[:80],
            "value": value[:500],
            "frame_index": frame_index,
            "evidence": evidence[:500],
            "modality": "visual",
        }
        for key in ("actor", "from_location", "to_location", "from_state",
                    "to_state", "time_scope"):
            extra = str(original.get(key) or "").strip()
            if extra:
                fact[key] = extra[:300]
        affected = original.get("affected_people")
        if isinstance(affected, list):
            fact["affected_people"] = list(dict.fromkeys(
                str(item).strip()[:120] for item in affected
                if isinstance(item, str) and item.strip()))[:8]
        try:
            number = int(original.get("time_order"))
        except (TypeError, ValueError):
            number = 0
        if number > 0 and not isinstance(original.get("time_order"), bool):
            fact["time_order"] = min(number, 999)
        status = str(original.get("status") or "").casefold()
        if status in {"observed", "completed", "unclear"}:
            fact["status"] = status
        confidence = str(original.get("confidence") or "").casefold()
        if confidence in {"high", "medium", "low"}:
            fact["confidence"] = confidence
        if predicate.casefold() == "moved":
            has_location_pair = bool(fact.get("from_location") and fact.get("to_location"))
            has_state_pair = bool(fact.get("from_state") and fact.get("to_state"))
            if not (has_location_pair or has_state_pair):
                rejected += 1
                continue
        output.append(fact)
    if rows and not output:
        raise ValueError(
            f"temporal fact repair returned no valid source-backed facts ({rejected} rejected)"
        )
    return output


def _apply_person_repair(fields: dict, payload: dict, reference_labels: list,
                         frame_count: int) -> dict:
    """Attach video-grounded appearances and reconcile references to visual aliases."""
    observations = []
    for row in payload.get("person_observations") or []:
        if not isinstance(row, dict):
            continue
        entity = " ".join(str(row.get("entity") or "").split())
        appearance = " ".join(str(row.get("appearance") or "").split())
        face_features = " ".join(str(row.get("face_features") or "").split())
        evidence = " ".join(str(row.get("evidence") or "").split())
        frame_index = row.get("frame_index")
        if (_is_generic_person_label(entity) or not entity or len(entity) > 200
                or not appearance or not evidence
                or not isinstance(frame_index, int) or isinstance(frame_index, bool)
                or frame_index < 1 or frame_index > max(1, frame_count)):
            continue
        age_text = str(row.get("age") or "").strip()
        age_basis = str(row.get("age_basis") or "").casefold()
        try:
            age_value = int(age_text)
        except ValueError:
            age_value = -1
        if age_basis != "visible_text" or not 0 <= age_value <= 120:
            age_text = ""
            age_basis = "unknown"
        observation = {
            "entity": entity,
            "appearance": appearance[:500],
            **({"face_features": face_features[:500]} if face_features else {}),
            "age": age_text,
            "age_basis": age_basis,
            "readable_name": (" ".join(str(row.get("readable_name") or "").split())[:160] if str(row.get("readable_name") or "").strip() and str(row.get("name_evidence") or "").strip() else ""),
            "name_evidence": (" ".join(str(row.get("name_evidence") or "").split())[:300] if str(row.get("readable_name") or "").strip() and str(row.get("name_evidence") or "").strip() else ""),
            "frame_index": frame_index,
            "evidence": evidence[:500],
            "confidence": (str(row.get("confidence") or "medium").lower()
                           if str(row.get("confidence") or "").lower()
                           in {"high", "medium", "low"} else "medium"),
        }
        existing = next((item for item in observations
                         if item["entity"].casefold() == entity.casefold()), None)
        if existing is None:
            observations.append(observation)
        elif len(appearance) + len(face_features) > len(existing["appearance"]) + len(existing.get("face_features") or ""):
            observations[observations.index(existing)] = observation

    if not observations:
        observations = [dict(row) for row in fields.get("person_observations") or []
                        if isinstance(row, dict)
                        and str(row.get("entity") or "").strip()
                        and str(row.get("appearance") or "").strip()]
    if not observations:
        return fields

    observed = {" ".join(str(row["entity"]).casefold().split()) for row in observations}
    valid_references = {" ".join(str(label).casefold().split()) for label in reference_labels}
    entity_map = {}
    raw_map = payload.get("entity_map") or []
    if isinstance(raw_map, dict):
        raw_map = [{"reference": key, "entity": value} for key, value in raw_map.items()]
    if isinstance(raw_map, list):
        for item in raw_map:
            if not isinstance(item, dict):
                continue
            reference = " ".join(str(item.get("reference") or "").casefold().split())
            entity = " ".join(str(item.get("entity") or "").casefold().split())
            if reference in valid_references and entity in observed:
                entity_map[reference] = entity
    if not entity_map and len(reference_labels) == 1 and len(observations) == 1:
        reference = " ".join(str(reference_labels[0]).casefold().split())
        entity = " ".join(observations[0]["entity"].casefold().split())
        if reference and reference not in observed:
            entity_map[reference] = entity

    # If the model omitted entity_map, reconcile a descriptive reference only
    # when exactly one frame-grounded observation shares several distinctive
    # visible attributes. This never maps neutral IDs or conflicting clothing.
    alias_stop = {"person", "people", "wearing", "with", "who", "has", "have",
                  "the", "a", "an", "and", "or", "of", "in", "on", "at",
                  "near", "standing", "sitting", "holding", "visible"}
    color_terms = {"black", "white", "gray", "grey", "blue", "red", "green",
                   "yellow", "orange", "pink", "purple", "brown", "beige",
                   "navy", "dark", "light"}
    apparel_terms = {"jacket", "coat", "shirt", "tshirt", "top", "pants", "trousers",
                     "dress", "skirt", "hat", "cap", "scarf", "headscarf",
                     "headband", "hood", "hoodie", "sweatshirt", "sweater", "cardigan",
                     "vest", "blouse", "suit", "blazer", "tie", "shoes", "boots",
                     "sneakers", "glasses", "necklace", "uniform"}
    def visual_tokens(value):
        return {token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
                if len(token) > 2 and token not in alias_stop}
    mapped_entities = set(entity_map.values())
    for reference_label in reference_labels:
        reference_key = " ".join(str(reference_label).casefold().split())
        if (not reference_key or reference_key in observed or reference_key in entity_map
                or _is_generic_person_label(reference_label)):
            continue
        ref_tokens = visual_tokens(reference_label)
        ref_colors = ref_tokens & color_terms
        if len(ref_tokens) < 2:
            continue
        ranked = []
        for observation in observations:
            entity_key = " ".join(observation["entity"].casefold().split())
            if entity_key in mapped_entities:
                continue
            candidate_text = " ".join((observation["entity"], observation["appearance"],
                                       observation.get("face_features") or ""))
            candidate_tokens = visual_tokens(candidate_text)
            candidate_colors = candidate_tokens & color_terms
            shared = ref_tokens & candidate_tokens
            if (len(shared) < 2 or len(shared & apparel_terms) < 1
                    or len(shared) / len(ref_tokens) < 0.65
                    or (ref_colors and candidate_colors and not (ref_colors & candidate_colors))):
                continue
            ranked.append((len(shared) / len(ref_tokens), entity_key))
        ranked.sort(reverse=True)
        if ranked and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.15):
            entity_map[reference_key] = ranked[0][1]
            mapped_entities.add(ranked[0][1])

    repaired = dict(fields)
    def remap(value):
        text = str(value or "").strip()
        exact = entity_map.get(" ".join(text.casefold().split()))
        if exact:
            return exact
        result = text
        for old_label, new_label in sorted(entity_map.items(), key=lambda item: len(item[0]), reverse=True):
            result = re.sub(r"(?<![\w])" + re.escape(old_label) + r"(?![\w])",
                            lambda _match, replacement=new_label: replacement,
                            result, flags=re.I)
        return result

    repaired["people"] = [remap(value) for value in (fields.get("people") or [])]
    for key in ("scene", "interaction_details", "user_action", "activity"):
        if repaired.get(key):
            repaired[key] = remap(repaired[key])
    repaired_facts = []
    for original in fields.get("temporal_facts") or []:
        if not isinstance(original, dict):
            continue
        fact = dict(original)
        for key in ("subject", "actor", "value", "evidence"):
            if fact.get(key):
                fact[key] = remap(fact[key])
        if isinstance(fact.get("affected_people"), list):
            fact["affected_people"] = [remap(value) for value in fact["affected_people"]]
        repaired_facts.append(fact)
    repaired["temporal_facts"] = repaired_facts
    repaired["person_observations"] = observations
    return _strip_unverified_age_categories(_organize_person_entity_aliases(repaired))


def _object_move_evidence_error(fields: dict) -> str:
    """Require an observable before/after pair before storing an object move."""
    incomplete = []
    for row in fields.get("temporal_facts") or []:
        if (not isinstance(row, dict)
                or str(row.get("subject_type") or "").casefold() != "object"
                or str(row.get("predicate") or "").casefold() != "moved"):
            continue
        has_location_pair = bool(row.get("from_location") and row.get("to_location"))
        has_state_pair = bool(row.get("from_state") and row.get("to_state"))
        has_frame = isinstance(row.get("frame_index"), int) and row["frame_index"] > 0
        if not (has_location_pair or has_state_pair) or not has_frame or not row.get("evidence"):
            incomplete.append(str(row.get("subject") or "object"))
    if incomplete:
        labels = ", ".join(dict.fromkeys(incomplete[:8]))
        return ("Object movement facts lack a source-backed before/after location or state "
                f"and frame evidence: {labels}")
    return ""


def _drop_unverified_object_moves(fields: dict) -> tuple[dict, list[dict]]:
    """Remove only object-movement claims that lack a source-backed transition.

    Other scene fields and facts are preserved. The original model text remains in
    raw so the rejected claim can be audited without entering searchable facts.
    """
    updated = dict(fields)
    kept = []
    dropped = []
    for original in fields.get("temporal_facts") or []:
        if not isinstance(original, dict):
            kept.append(original)
            continue
        row = dict(original)
        is_move = (str(row.get("subject_type") or "").casefold() == "object"
                   and str(row.get("predicate") or "").casefold() == "moved")
        if not is_move:
            kept.append(row)
            continue
        has_location_pair = bool(row.get("from_location") and row.get("to_location"))
        has_state_pair = bool(row.get("from_state") and row.get("to_state"))
        if not (has_location_pair or has_state_pair):
            value = str(row.get("value") or "")
            match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[.;]|$)",
                              value, flags=re.I)
            if match:
                row["from_location"] = match.group(1).strip()
                row["to_location"] = match.group(2).strip()
                has_location_pair = bool(row["from_location"] and row["to_location"])
        frame_index = row.get("frame_index")
        has_frame = (isinstance(frame_index, int) and not isinstance(frame_index, bool)
                     and frame_index > 0)
        has_evidence = bool(str(row.get("evidence") or "").strip())
        if not ((has_location_pair or has_state_pair) and has_frame and has_evidence):
            dropped.append({
                "subject": str(row.get("subject") or "object")[:120],
                "value": str(row.get("value") or "")[:240],
                "frame_index": frame_index if isinstance(frame_index, int) else None,
            })
            continue
        kept.append(row)
    updated["temporal_facts"] = kept
    return updated, dropped


def _drop_unverified_person_references(fields: dict) -> tuple[dict, list[dict]]:
    """Remove only person claims that cannot be linked to a visible observation.

    The original model response is kept separately as raw for audit. This
    function never creates or renumbers people; unrelated scene and object facts
    remain available for retrieval.
    """
    observations = [row for row in fields.get("person_observations") or []
                    if isinstance(row, dict)]
    usable = {
        " ".join(str(row.get("entity") or "").casefold().split())
        for row in observations if str(row.get("appearance") or "").strip()
    }
    unresolved = {
        " ".join(str(label or "").casefold().split())
        for label in _person_observation_references(fields)
        if (" ".join(str(label or "").casefold().split()) not in usable
            or _is_generic_person_label(label))
    }
    blank_observations = [
        row for row in observations
        if (not str(row.get("appearance") or "").strip()
            or _is_generic_person_label(row.get("entity")))
    ]
    if not unresolved and not blank_observations:
        return fields, []

    removed = []
    for row in blank_observations:
        removed.append({
            "kind": "person_observation",
            "entity": str(row.get("entity") or "")[:120],
            "frame_index": row.get("frame_index"),
        })

    updated = dict(fields)
    updated["person_observations"] = [
        row for row in observations
        if str(row.get("appearance") or "").strip()
        and not _is_generic_person_label(row.get("entity"))
    ]

    kept_people = []
    for value in fields.get("people") or []:
        label = str(value or "").strip()
        key = " ".join(label.casefold().split())
        if not label:
            continue
        if key in usable or _camera_wearer_reference(label):
            kept_people.append(value)
        else:
            removed.append({"kind": "person_reference", "entity": label[:120]})
    updated["people"] = kept_people

    kept_facts = []
    for original in fields.get("temporal_facts") or []:
        if not isinstance(original, dict):
            kept_facts.append(original)
            continue
        references = []
        for value in [original.get("subject"), original.get("actor"),
                      *(original.get("affected_people") or [])]:
            label = str(value or "").strip()
            key = " ".join(label.casefold().split())
            if key in unresolved and label not in references:
                references.append(label)
        if references:
            removed.append({
                "kind": "temporal_fact",
                "subject_type": str(original.get("subject_type") or "")[:40],
                "subject": str(original.get("subject") or "")[:120],
                "predicate": str(original.get("predicate") or "")[:80],
                "frame_index": original.get("frame_index"),
                "unverified_people": references[:8],
            })
            continue
        kept_facts.append(original)
    updated["temporal_facts"] = kept_facts
    return updated, removed


_VISIBLE_ACTION_CUE_RE = re.compile(
    r"\b(?:approach(?:es|ed|ing)?|enter(?:s|ed|ing)?|exit(?:s|ed|ing)?|emerg(?:e|es|ed|ing)|walk(?:s|ed|ing)?|run(?:s|ning)?|sit(?:s|ting)?|stand(?:s|ing)?|turn(?:s|ed|ing)?|reach(?:es|ed|ing)?|pick(?:s|ed)? up|put(?:s|ting)?|place(?:s|d|ing)?|hand(?:s|ed|ing)? over|give(?:s|gave|n)?|receive(?:s|d|ing)?|hold(?:s|ing)?|use(?:s|d|ing)?|open(?:s|ed|ing)?|clos(?:e|es|ed|ing)|mov(?:e|es|ed|ing)|carry(?:ies|ied|ing)?|write(?:s|wrote|ing)?|eras(?:e|es|ed|ing)|smil(?:e|es|ed|ing)|laugh(?:s|ed|ing)?|rais(?:e|es|ed|ing)|lower(?:s|ed|ing)?|lift(?:s|ed|ing)?|bend(?:s|ing)?|kneel(?:s|ed|ing)?|lie(?:s|d)?|eats?|drinks?|wash(?:es|ed|ing)?|clean(?:s|ed|ing)?|rinse(?:s|d|ing)?|interact(?:s|ed|ing)?|pass(?:es|ed|ing)?|transfer(?:s|red|ring)?|tap(?:s|ped|ping)?|point(?:s|ed|ing)?|touch(?:es|ed|ing)?|hug(?:s|ged|ging)?|nod(?:s|ded|ding)?|clap(?:s|ped|ping)?|adjust(?:s|ed|ing)?|grab(?:s|bed|bing)?|step(?:s|ped|ping)?|wait(?:s|ed|ing)?|wave(?:s|d|ing)?|read(?:s|ing)?|show(?:s|ed|ing)?|inspect(?:s|ed|ing)?|examine(?:s|d|ing)?|check(?:s|ed|ing)?|pull(?:s|ed|ing)?|push(?:es|ed|ing)?|drop(?:s|ped|ping)?|set(?:s|ting)? down|put(?:s|ting)? away|remove(?:s|d|ing)?|pack(?:s|ed|ing)?|unpack(?:s|ed|ing)?|cook(?:s|ed|ing)?|cut(?:s|ting)?|serve(?:s|d|ing)?|drive(?:s|d|ing)?|ride(?:s|d|ing)?|board(?:s|ed|ing)?|arrive(?:s|d|ing)?|depart(?:s|ed|ing)?|changes? expression|changed posture|bends? forward|looks? (?:up|down|around))\b", re.I
)


def _canonical_action_cue(value: str) -> str:
    token = str(value or "").casefold().strip()
    if token in {"picks up", "pick up", "picked up"}:
        return "pick up"
    if token in {"hands over", "handed over", "handing over"}:
        return "hand over"
    known_e = {"approach", "move", "place", "smile", "raise", "close", "change", "erase", "use", "write"}
    if token.endswith("ing"):
        base = token[:-3]
        if len(base) > 2 and base[-1] == base[-2] and not base.endswith("ss"):
            base = base[:-1]
        if base + "e" in known_e:
            base += "e"
        return base
    if token.endswith("ied"):
        return token[:-3] + "y"
    if token.endswith("ed"):
        base = token[:-2]
        return base + "e" if base + "e" in known_e else base
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es"):
        base = token[:-2]
        return base + "e" if base + "e" in known_e else base
    if token.endswith("ss"):
        return token
    if token.endswith("s") and len(token) > 2:
        return token[:-1]
    return token

def _temporal_fact_coverage_error(fields: dict) -> str:
    """Require most narrated actions to survive as distinct retrievable facts."""
    if not isinstance(fields, dict):
        return ""
    narratives = [
        str(fields.get(key) or "")
        for key in ("scene", "user_action", "interaction_details", "activity")
    ]
    # Validate against the richest single narrative field. This catches a detailed
    # Interaction Details field even when Scene is present but overly terse, without
    # double-counting the same action repeated across several summary fields.
    cue_sets = [
        [_canonical_action_cue(match.group(0))
         for match in _VISIBLE_ACTION_CUE_RE.finditer(narrative)]
        for narrative in narratives
    ]
    cues = max(cue_sets, key=len, default=[])
    if not cues:
        return ""
    facts = [row for row in fields.get("temporal_facts") or [] if isinstance(row, dict)]
    action_facts = [row for row in facts if str(row.get("predicate") or "").casefold()
                    not in {"appearance", "appeared", "located_at", "occurred_at",
                            "present", "visible", "environment"}]
    # Count distinct facts, so repeating the same subject/predicate/value cannot
    # stand in for missing actions. Require 75% of narrated action cues (up to eight
    # facts); an unsupported cue still must be removed from the scene summary.
    distinct = set()
    for row in action_facts:
        key = tuple(" ".join(str(row.get(field) or "").casefold().split())
                    for field in ("subject", "predicate", "value"))
        if all(key):
            distinct.add(key)
    required = max(1, min(8, (3 * len(cues) + 3) // 4))
    if len(distinct) < required:
        return (
            f"The scene contains {len(cues)} action/state cues but only "
            f"{len(distinct)} distinct atomic event facts; add separately grounded facts "
            "for missing actions or remove unsupported narrative claims."
        )
    return ""

def _build_temporal_fact_coverage_repair_prompt(fields: dict, frame_count: int,
                                                  max_additional: int) -> str:
    """Ask Qwen to fill only source-grounded action/state facts missing from a valid result."""
    fields = fields if isinstance(fields, dict) else {}
    scene = str(fields.get("scene") or "")[:1400]
    interaction = str(fields.get("interaction_details") or "")[:900]
    activity = str(fields.get("activity") or "")[:700]
    cues = sorted({
        _canonical_action_cue(match.group(0))
        for match in _VISIBLE_ACTION_CUE_RE.finditer(" ".join((scene, interaction, activity)))
    })
    existing = []
    for row in fields.get("temporal_facts") or []:
        if not isinstance(row, dict):
            continue
        existing.append({key: str(row.get(key) or "")[:160]
                         for key in ("subject", "predicate", "value")})
    aliases = [str(row.get("entity") or "")[:100]
               for row in fields.get("person_observations") or []
               if isinstance(row, dict) and row.get("entity")]
    parts = [
        "Inspect ONLY the supplied ordered video frames. Scene and existing facts are hints, not proof; verify every claim against pixels.",
        "Return one JSON array of ONLY additional visible facts not already represented. Separate each distinct action or meaningful state change into its own atomic row.",
        "Include actor, affected person/object, visible before/after state or location, and order when supported. For a stable posture/expression use predicate=state, but do not treat a continuing pose as a change.",
        "Use exact existing aliases; never number people or infer identity/relationship. Quote important readable text as displayed, not as verified truth. Do not infer speech, plans, diagnosis, advice, or unseen movement.",
        "Omit any cue not supported by pixels. Each value <=20 words; evidence <=12 words; 1-based frame_index 1.." + str(max(1, int(frame_count))) + ". Return <=" + str(max(0, min(8, int(max_additional)))) + " rows, English only, JSON only.",
        "Scene hint: " + scene,
        "Interaction hint: " + interaction,
        "Activity hint: " + activity,
        "Action cues to check: " + json.dumps(cues, ensure_ascii=True),
        "Existing facts (do not repeat): " + json.dumps(existing, ensure_ascii=True),
        "Existing visual person aliases: " + json.dumps(aliases, ensure_ascii=True),
        'Schema: [{"subject_type":"person|object|place|environment|event","subject":"entity label","predicate":"state|used|moved|placed|other","value":"one supported fact","actor":"visible actor or empty","affected_people":[],"from_location":"supported or empty","to_location":"supported or empty","time_order":1,"frame_index":1,"evidence":"visible detail","confidence":"high|medium|low"}]',
    ]
    return "\n".join(parts)


def _structured_fact_quality_error(fields: dict) -> str:
    errors = [error for error in (
        _person_observation_coverage_error(fields),
        _object_move_evidence_error(fields),
        _temporal_fact_coverage_error(fields),
    ) if error]
    return " ".join(errors)


class PersonRepairCancelled(RuntimeError):
    """A cancelled same-video repair must never fall through to person deletion."""


def _repair_person_references_from_video(
    fields: dict, frame_count: int, frame_files: list, request_video_analysis,
) -> tuple[dict, dict]:
    """Repair missing appearance using only the current clip's sampled frames."""
    references = _person_reference_labels(fields)
    context = {
        key: fields.get(key)
        for key in ("scene", "location", "activity", "interaction_details", "user_action", "people", "person_observations", "temporal_facts")
        if fields.get(key)
    }
    prompt = _build_person_repair_prompt(context, references)
    raw_payload = request_video_analysis(prompt, frame_files)
    payload = raw_payload if isinstance(raw_payload, dict) else _parse_person_repair(raw_payload)
    if _contains_cjk(json.dumps(payload, ensure_ascii=False)):
        raise ValueError("focused person pass returned non-English text")
    repaired = _apply_person_repair(fields, payload, references, frame_count)
    move_error = _object_move_evidence_error(repaired)
    if move_error:
        raise ValueError(move_error)
    if not _has_usable_english_memory(repaired):
        raise ValueError("no substantive memory remained after video person repair")
    person_error = _person_observation_coverage_error(repaired)
    return repaired, {
        "status": "repaired" if not person_error else "partial_repair",
        "method": "same_video_frames_person_repair",
        "repaired_people": len(repaired.get("person_observations") or []),
        "repair_error": person_error,
        "removed_count": 0,
        "removed_items": [],
        "raw_model_output_preserved": True,
    }


def _salvage_person_reference_gaps(
    fields: dict, repair_person_references=None,
) -> tuple[dict | None, list[dict], dict]:
    """Try same-video repair first; remove unresolved references only after failure."""
    if (not _person_observation_coverage_error(fields)
            or _object_move_evidence_error(fields)):
        return None, [], {"status": "not_applicable"}
    if repair_person_references is None:
        return None, [], {
            "status": "repair_unavailable",
            "method": "same_video_frames_person_repair",
            "removed_count": 0,
            "removed_items": [],
            "raw_model_output_preserved": True,
        }
    repair_error = ""
    repaired = None
    repair_log = {}
    try:
        repaired, repair_log = repair_person_references(fields)
        if (repaired is not None
                and not _structured_fact_quality_error(repaired)
                and _has_usable_english_memory(repaired)):
            return repaired, [], repair_log
        repair_error = (
            str(repair_log.get("repair_error") or "")
            if isinstance(repair_log, dict) else ""
        ) or "video repair did not satisfy person or fact quality checks"
    except PersonRepairCancelled:
        raise
    except Exception as exc:
        # An API/CUDA/timeout error means the source review did not happen.
        # Never turn an unavailable repair into deletion of source references.
        repair_error = str(exc)[:300] or exc.__class__.__name__
        return None, [], {
            "status": "repair_failed_preserved",
            "method": "same_video_frames_person_repair",
            "repair_error": repair_error,
            "repaired_people": 0,
            "removed_count": 0,
            "removed_items": [],
            "raw_model_output_preserved": True,
        }
    candidate_after_repair = repaired if repaired is not None else fields
    sanitized, removed = _drop_unverified_person_references(candidate_after_repair)
    coverage_error = _person_observation_coverage_error(sanitized)
    move_error = _object_move_evidence_error(sanitized)
    if move_error or not _has_usable_english_memory(sanitized):
        return None, [], {
            "status": "repair_failed_no_valid_salvage",
            "method": "same_video_repair_then_drop_unverified_references",
            "repair_error": repair_error or move_error,
            "repaired_people": (repair_log.get("repaired_people", 0)
                                if isinstance(repair_log, dict) else 0),
            "removed_count": len(removed),
            "removed_items": removed,
            "raw_model_output_preserved": True,
        }

    narrative_gap_prefixes = (
        "Scene, Activity, or Interaction Details describe a visible person/group",
        "The narrative identifies ",
        "The narrative describes multiple visible people",
        "The narrative distinguishes multiple people",
    )
    if coverage_error and not coverage_error.startswith(narrative_gap_prefixes):
        return None, [], {
            "status": "repair_failed_no_valid_salvage",
            "method": "same_video_repair_then_drop_unverified_references",
            "repair_error": coverage_error,
            "repaired_people": (repair_log.get("repaired_people", 0)
                                if isinstance(repair_log, dict) else 0),
            "removed_count": len(removed),
            "removed_items": removed,
            "raw_model_output_preserved": True,
        }
    if coverage_error:
        coverage_gap = {
            "kind": "person_coverage_gap",
            "reason": coverage_error[:300],
            "visible_observation_count": len(sanitized.get("person_observations") or []),
        }
        removed.append(coverage_gap)
        return sanitized, removed, {
            "status": "partial_person_coverage_after_source_review",
            "method": "same_video_repair_then_drop_unverified_references",
            "repair_error": repair_error,
            "person_coverage_gap": coverage_error,
            "repaired_people": (repair_log.get("repaired_people", 0)
                                if isinstance(repair_log, dict) else 0),
            "removed_count": len(removed),
            "removed_items": removed,
            "raw_model_output_preserved": True,
        }
    return sanitized, removed, {
        "status": "removed_after_repair_failure" if removed else "repaired_after_source_review",
        "method": "same_video_repair_then_drop_unverified_references",
        "repair_error": repair_error,
        "repaired_people": (repair_log.get("repaired_people", 0)
                            if isinstance(repair_log, dict) else 0),
        "removed_count": len(removed),
        "removed_items": removed,
        "raw_model_output_preserved": True,
    }


def _contains_cjk(text: str) -> bool:
    """Return True when model output still contains untranslated CJK text."""
    return bool(re.search(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]", str(text or "")))


def _sanitize_cjk_spans(text: str) -> str:
    """Redact unsupported non-English spans without discarding the whole scene."""
    return re.sub(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+",
                  "unreadable text", str(text or ""))


def _has_usable_english_memory(fields: dict) -> bool:
    """Require at least one substantive English fact after untranslated spans are redacted."""
    ignored = {"subject_type", "predicate", "confidence", "status", "modality",
               "frame_index", "time_order", "age_basis"}

    def walk(value):
        if isinstance(value, str):
            cleaned = re.sub(r"\bunreadable text\b", " ", value, flags=re.I)
            return len(re.sub(r"[^A-Za-z0-9]", "", cleaned)) >= 8
        if isinstance(value, dict):
            return any(walk(item) for key, item in value.items() if key not in ignored)
        if isinstance(value, (list, tuple)):
            return any(walk(item) for item in value)
        return False

    return any(walk(value) for key, value in fields.items()
               if key not in {"source_evidence", "needs", "solutions"})


# ---------------------------------------------------------------------------
# 视频 → 抽帧
# ---------------------------------------------------------------------------

def _intra_hour_offset(mid: str) -> int:
    """文件名时间戳相对整点小时的秒数：..._11093000 → 9*60+30 = 570"""
    parts = str(mid).split("_")
    if (len(parts) >= 4 and re.fullmatch(r'DAY\d+', parts[0], re.I)
            and len(parts[3]) >= 6 and parts[3][:6].isdigit()
            and int(parts[3][:2]) < 24 and int(parts[3][2:4]) < 60 and int(parts[3][4:6]) < 60):
        t = parts[3]
        return int(t[2:4]) * 60 + int(t[4:6])
    return 0


def _video_file_sha256(path: Path):
    """Return a persisted content hash, invalidated by source-file identity changes."""
    try:
        resolved = Path(path).resolve()
        before = resolved.stat()
    except OSError:
        return None
    cache_key = str(resolved)
    signature = (
        int(before.st_size), int(before.st_mtime_ns), int(before.st_ctime_ns),
        int(getattr(before, "st_ino", 0)), int(getattr(before, "st_dev", 0)),
    )

    def ensure_schema(connection):
        connection.execute(
            "CREATE TABLE IF NOT EXISTS source_file_hashes ("
            "path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, "
            "sha256 TEXT NOT NULL)"
        )
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(source_file_hashes)"
        ).fetchall()}
        for name in ("ctime_ns", "inode", "device"):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE source_file_hashes ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                )

    with _SOURCE_FINGERPRINT_LOCK:
        connection = None
        try:
            connection = sqlite3.connect(str(SOURCE_FINGERPRINT_DB), timeout=30)
            connection.execute("PRAGMA busy_timeout=30000")
            ensure_schema(connection)
            row = connection.execute(
                "SELECT size, mtime_ns, ctime_ns, inode, device, sha256 "
                "FROM source_file_hashes WHERE path=?", (cache_key,),
            ).fetchone()
            connection.commit()
        except (OSError, sqlite3.Error):
            row = None
        finally:
            if connection is not None:
                connection.close()
    if row and tuple(int(value) for value in row[:5]) == signature:
        return row[5]

    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
        after = resolved.stat()
    except OSError:
        return None
    after_signature = (
        int(after.st_size), int(after.st_mtime_ns), int(after.st_ctime_ns),
        int(getattr(after, "st_ino", 0)), int(getattr(after, "st_dev", 0)),
    )
    if after_signature != signature:
        return None
    value = digest.hexdigest()
    with _SOURCE_FINGERPRINT_LOCK:
        connection = None
        try:
            connection = sqlite3.connect(str(SOURCE_FINGERPRINT_DB), timeout=30)
            connection.execute("PRAGMA busy_timeout=30000")
            ensure_schema(connection)
            connection.execute(
                "INSERT OR REPLACE INTO source_file_hashes "
                "(path,size,mtime_ns,sha256,ctime_ns,inode,device) VALUES (?,?,?,?,?,?,?)",
                (cache_key, *signature[:2], value, *signature[2:]),
            )
            connection.commit()
        except (OSError, sqlite3.Error):
            pass
        finally:
            if connection is not None:
                connection.close()
    return value

def source_video_input_fingerprint(path: Path, start: float, end):
    """Hash source video bytes plus the exact clip interval used for perception."""
    video_hash = _video_file_sha256(path)
    if not video_hash:
        return None
    try:
        size_bytes = Path(path).stat().st_size
    except OSError:
        return None
    start_value = round(float(start or 0), 3)
    end_value = round(float(end), 3) if end is not None else None
    clip_payload = json.dumps(
        {"video_sha256": video_hash, "start_s": start_value, "end_s": end_value},
        sort_keys=True, separators=(",", ":"),
    )
    return {
        "video_sha256": video_hash,
        "clip_sha256": hashlib.sha256(clip_payload.encode("utf-8")).hexdigest(),
        "clip_start_s": start_value,
        "clip_end_s": end_value,
        "video_size_bytes": size_bytes,
    }


def clip_range(m: dict):
    """片段在视频文件内的 [start, end] 秒。"""
    st = m.get("start_time")
    if st is None:
        return 0.0, None
    off = _intra_hour_offset(m.get("id", ""))
    return max(0.0, st - off), m.get("end_time", st + 30) - off


def video_path_of(m: dict):
    """数据集 video 字段 → 本地视频绝对路径；不存在返回 None。"""
    v = m.get("video") or ""
    if not v:
        return None
    p = ROOT / v.replace("EgoLife", "egolife")
    return p if p.exists() else None


def extract_frames(vp: Path, start: float, end, n: int = DEFAULT_FRAMES,
                   return_times: bool = False):
    """在片段区间内均匀抽 n 帧；可同时返回实际读取帧的源视频秒数。"""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {vp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = (total / fps) if fps > 0 else 0.0

    if end is None or end <= start:
        end = (dur or (start + 30))
    end = min(end, dur) if dur else end

    # Video intervals are end-exclusive. Sampling exactly at end often asks
    # OpenCV for a frame past EOF, silently dropping the final (often decisive) frame.
    last_valid_time = max(start, end - 1.0 / fps)
    times = [start + (last_valid_time - start) * i / max(n - 1, 1)
             for i in range(n)] if n > 1 else [start]
    bufs = []
    frame_times = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
            ok, fr = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok2:
                bufs.append(buf.tobytes())
                frame_times.append(round(float(t), 3))
    finally:
        cap.release()
    if not bufs:
        raise RuntimeError(f"抽帧失败: {vp}")
    return (bufs, frame_times) if return_times else bufs


# ---------------------------------------------------------------------------
# 感知客户端
# ---------------------------------------------------------------------------

class QwenPerceiver:
    def __init__(self, api_url: str = DEFAULT_API, num_frames: int = DEFAULT_FRAMES,
                 cache_file: Path = CACHE_FILE):
        configured_urls = [
            value.strip().rstrip("/")
            for value in os.environ.get("QWEN_API_URLS", "").split(",")
            if value.strip()
        ]
        self.api_urls = configured_urls or [api_url.rstrip("/")]
        self.api_url = self.api_urls[0]
        self._api_url_lock = threading.Lock()
        self._api_url_cursor = 0
        self.num_frames = num_frames
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache = None
        self._cache_stamp = None
        self._dirty_cache_keys = set()
        self._cache_lock = threading.RLock()
        self._active_request_lock = threading.Lock()
        self._active_request_id = None
        self._active_api_url = self.api_url
        self._active_requests = {}

    def next_api_url(self) -> str:
        """Choose a healthy-pool candidate in round-robin order."""
        with self._api_url_lock:
            api_url = self.api_urls[self._api_url_cursor % len(self.api_urls)]
            self._api_url_cursor += 1
            return api_url

    def _activate_request(self, request_id: str, api_url: str) -> None:
        with self._active_request_lock:
            self._active_requests[request_id] = api_url
            self._active_request_id = request_id
            self._active_api_url = api_url

    def _finish_request(self, request_id: str) -> None:
        with self._active_request_lock:
            self._active_requests.pop(request_id, None)
            if self._active_request_id == request_id:
                if self._active_requests:
                    latest_id = next(reversed(self._active_requests))
                    self._active_request_id = latest_id
                    self._active_api_url = self._active_requests[latest_id]
                else:
                    self._active_request_id = None
                    self._active_api_url = self.api_url

    def cancel_active(self):
        """Cancel all currently active requests across the configured model pool."""
        with self._active_request_lock:
            active = list(self._active_requests.items())
            if not active and self._active_request_id:
                active = [(self._active_request_id,
                           self._active_api_url or self.api_url)]
        cancelled = False
        for request_id, api_url in active:
            try:
                response = _post_with_inference_cancel(
                    f"{api_url}/cancel_inference",
                    json={"request_id": request_id},
                    timeout=(2, 5),
                )
                cancelled = cancelled or (
                    response.ok and bool(response.json().get("cancelled"))
                )
            except Exception:
                continue
        return cancelled

    @staticmethod
    def _cancelled_result():
        return {
            "source": "cancelled",
            "cancelled": True,
            "error": "实时推理已停止",
            "fields": None,
        }

    # -- 缓存 ---------------------------------------------------------------
    @staticmethod
    def _signature(stat_result):
        return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size,
                stat_result.st_mtime_ns, stat_result.st_ctime_ns)

    def _cache_file_signature(self):
        try:
            return self._signature(self.cache_file.stat())
        except OSError:
            return None

    def _read_cache_snapshot(self):
        try:
            with self.cache_file.open("r", encoding="utf-8") as handle:
                signature = self._signature(os.fstat(handle.fileno()))
                value = json.load(handle)
        except (OSError, ValueError, TypeError):
            return {}, None
        return (value if isinstance(value, dict) else {}), signature

    @staticmethod
    def _entry_rank(entry):
        if not isinstance(entry, dict):
            return (-1, -1, -1, "")
        fingerprint = entry.get("source_input_fingerprint")
        source_backed = int(
            isinstance(fingerprint, dict)
            and bool(fingerprint.get("video_sha256"))
            and bool(fingerprint.get("clip_sha256"))
        )
        return (
            int(entry.get("prompt_version") == PROMPT_VERSION),
            source_backed,
            int(entry.get("list_parser_version") == LIST_PARSER_VERSION),
            str(entry.get("ts") or ""),
        )

    @classmethod
    def _prefer_cache_entry(cls, current, candidate):
        if candidate is None:
            return current
        if current is None or cls._entry_rank(candidate) >= cls._entry_rank(current):
            return candidate
        return current

    @property
    def cache(self):
        with self._cache_lock:
            current_signature = self._cache_file_signature()
            if self._cache is None or current_signature != self._cache_stamp:
                disk_cache, disk_signature = self._read_cache_snapshot()
                if self._cache is not None:
                    for key in self._dirty_cache_keys:
                        if key in self._cache:
                            disk_cache[key] = self._prefer_cache_entry(
                                disk_cache.get(key), self._cache[key])
                self._cache = disk_cache
                self._cache_stamp = disk_signature
            return self._cache

    def _save_cache(self):
        with self._cache_lock:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            _ = self.cache  # Refresh other-process writes before merging.
            lock_path = self.cache_file.with_name(self.cache_file.name + ".lock")
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                with os.fdopen(lock_fd, "a+b") as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    try:
                        disk_cache, _disk_signature = self._read_cache_snapshot()
                        merged = dict(disk_cache)
                        for key, candidate in (self._cache or {}).items():
                            merged[key] = self._prefer_cache_entry(
                                merged.get(key), candidate)
                        fd, temporary_name = tempfile.mkstemp(
                            prefix=self.cache_file.name + ".",
                            suffix=".tmp", dir=str(self.cache_file.parent))
                        try:
                            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                                json.dump(merged, handle, ensure_ascii=False, indent=2)
                                handle.flush()
                                os.fsync(handle.fileno())
                            os.replace(temporary_name, self.cache_file)
                        finally:
                            try:
                                os.unlink(temporary_name)
                            except FileNotFoundError:
                                pass
                        self._cache = merged
                        self._cache_stamp = self._cache_file_signature()
                        self._dirty_cache_keys.clear()
                    finally:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except BaseException:
                # fdopen owns lock_fd once entered; close only if opening failed.
                raise

    # -- 健康检查 -----------------------------------------------------------
    def health(self):
        errors = []
        for api_url in self.api_urls:
            try:
                r = requests.get(f"{api_url}/health", timeout=3)
                payload = r.json() if r.status_code == 200 else {"model_loaded": False}
                if payload.get("model_loaded"):
                    payload["api_url"] = api_url
                    return payload
                errors.append(f"{api_url}: model not loaded")
            except Exception as exc:
                errors.append(f"{api_url}: {exc}")
        return {"model_loaded": False, "error": "; ".join(errors)[:500]}

    # -- 核心 ---------------------------------------------------------------
    def perceive(self, m: dict, force: bool = False, cancel_event=None,
                 progress_callback=None, persist_cache: bool = True,
                 allow_legacy_cache: bool = False,
                 allow_stale_cache_fallback: bool = False):
        """
        产出一条记忆条目。force=True 强制实时调用真模型（忽略缓存）。

        返回 {source, fields, raw, elapsed_ms, num_frames, model, error}
        source: 'cache' | 'qwen'
        """
        mid = m.get("id", "")
        # Keep valid scene/fact caches usable even when optional appearance detail is
        # incomplete; unsupported object-movement transitions still require repair.
        validate_source_fact_quality = True
        def emit(stage, message):
            if progress_callback:
                try:
                    progress_callback({"stage": stage, "message": message})
                except Exception:
                    pass
        if cancel_event is not None and cancel_event.is_set():
            return self._cancelled_result()
        vp = video_path_of(m)
        start, end = clip_range(m)
        input_fingerprint = source_video_input_fingerprint(vp, start, end) if vp else None
        with self._cache_lock:
            cached_entry = dict(self.cache[mid]) if mid in self.cache else None
        source_matches = (
            vp is not None
            and input_fingerprint is not None
            and (cached_entry or {}).get("source_input_fingerprint") == input_fingerprint
        )

        def stale_cache_fallback(reason):
            if (not allow_stale_cache_fallback or force or not cached_entry
                    or not source_matches):
                return None
            old_fields = cached_entry.get("fields")
            if not isinstance(old_fields, dict):
                return None
            old_raw = str(cached_entry.get("raw_model_output")
                          or cached_entry.get("raw") or "")
            if (_contains_cjk(json.dumps(old_fields, ensure_ascii=False))
                    or _contains_cjk(old_raw)
                    or _object_move_evidence_error(old_fields)
                    or not _has_usable_english_memory(old_fields)):
                return None
            hit = dict(cached_entry)
            hit["fields"] = dict(old_fields)
            hit["source"] = "stale_cache_fallback"
            hit["stale_cache_fallback"] = True
            hit["stale_prompt_version"] = cached_entry.get("prompt_version")
            hit["stale_cache_reason"] = str(reason or "refresh unavailable")[:500]
            warning = (
                "临时显示与当前视频指纹一致的旧缓存；"
                "该内容未按当前提示词复核，也未覆盖原缓存。"
            )
            person_error = _person_observation_coverage_error(old_fields)
            if person_error:
                warning += " " + person_error[:240]
            hit["quality_warning"] = warning
            if old_raw:
                hit["raw"] = old_raw
            emit(
                "cache_stale_fallback",
                "已临时显示同一视频的旧缓存，并标记为待复核",
            )
            return hit

        if (cached_entry and cached_entry.get("prompt_version") != PROMPT_VERSION
                and source_matches and allow_stale_cache_fallback and not force):
            # Interactive step execution prioritizes immediate, source-identical output;
            # prewarm/force runs still perform the full current-prompt refresh.
            fast_old = stale_cache_fallback(
                "同一视频已有缓存，但感知提示词版本较旧；等待 GPU 预热复核。")
            if fast_old:
                emit("cache_stale_fallback", "同源旧版缓存已即时读取，当前结果会标记为待复核")
                return fast_old

        if cached_entry and (cached_entry.get("prompt_version") != PROMPT_VERSION or not source_matches):
            legacy_fields = cached_entry.get("fields") or {}
            legacy_valid = (
                not _contains_cjk(json.dumps(legacy_fields, ensure_ascii=False))
                and not _contains_cjk(cached_entry.get("raw", ""))
                and not _object_move_evidence_error(legacy_fields)
            )
            if allow_legacy_cache and not force and legacy_valid and source_matches:
                hit = dict(cached_entry)
                hit["source"] = "legacy_cache"
                hit["legacy_prompt_version"] = cached_entry.get("prompt_version")
                person_error = _person_observation_coverage_error(legacy_fields)
                if person_error:
                    hit["quality_warning"] = person_error
                emit("cache_legacy", "使用旧版场景缓存快速返回；新版细节缓存完成后自动切换")
                if hit.get("raw_model_output"):
                    hit["raw"] = hit["raw_model_output"]
                return hit
            emit("cache_outdated", "感知版本或源视频内容已变化，正在按当前视频重新提取")
            # Retain the old entry until a new, validated result replaces it.
            # It is not returned as a current-version hit, but a failed refresh
            # must not erase the last usable source-grounded scene.
        if (not force and cached_entry and cached_entry.get("prompt_version") == PROMPT_VERSION
                and source_matches):
            hit = dict(cached_entry)
            cached_fields = dict(hit.get("fields") or {})
            if hit.get("list_parser_version") != LIST_PARSER_VERSION:
                try:
                    reparsed = parse_analysis(hit.get("raw") or hit.get("raw_model_output", ""), strict=True)
                    for field in ("objects", "environments", "people", "person_observations",
                                  "temporal_facts", "scene", "interaction_details", "user_action",
                                  "activity", "location", "time_occasion"):
                        cached_fields[field] = reparsed.get(field)
                    cached_fields["person_label_schema_version"] = _PERSON_LABEL_SCHEMA_VERSION
                    hit["fields"] = cached_fields
                    hit["list_parser_version"] = LIST_PARSER_VERSION
                    hit["list_parser_repaired_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with self._cache_lock:
                        self.cache[mid] = dict(hit)
                        self._dirty_cache_keys.add(mid)
                        if persist_cache:
                            self._save_cache()
                    emit("cache_reparse", "整理已缓存的人物描述、事件引用、物品与环境字段，无需重新调用视觉模型")
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    emit("cache_reparse_failed", f"缓存文本无法安全重解析，将重新核验：{str(exc)[:120]}")
                    cached_fields = dict(hit.get("fields") or {})
            cached_fields["source_evidence"] = ""
            hit["fields"] = cached_fields
            fields_text = json.dumps(hit.get("fields") or {}, ensure_ascii=False)
            person_error = _person_observation_coverage_error(cached_fields)
            move_error = _object_move_evidence_error(cached_fields)
            fact_error = _temporal_fact_coverage_error(cached_fields)
            if (allow_stale_cache_fallback and not force
                    and hit.get("list_parser_version") == LIST_PARSER_VERSION
                    and not _contains_cjk(fields_text) and not _contains_cjk(hit.get("raw", ""))
                    and not move_error and (person_error or fact_error)
                    and _has_usable_english_memory(cached_fields)):
                warning_parts = [error for error in (person_error, fact_error) if error]
                hit["source"] = "cache_quality_warning"
                hit["quality_warning"] = (
                    "已即时复用与当前视频指纹一致的缓存；其中部分人物观察或事件事实仍待补强。"
                    + " " + " ".join(warning_parts)[:500]
                )
                if hit.get("raw_model_output"):
                    hit["raw"] = hit["raw_model_output"]
                emit("cache_quality_warning", "缓存已即时命中；该片段有质量缺口，等待预热任务按完整门槛复核")
                return hit

            if (hit.get("list_parser_version") == LIST_PARSER_VERSION
                    and not _contains_cjk(fields_text) and not _contains_cjk(hit.get("raw", ""))
                    and not move_error and not fact_error):
                hit["source"] = "parser_repair" if hit.get("list_parser_version") == LIST_PARSER_VERSION and cached_entry.get("list_parser_version") != LIST_PARSER_VERSION else "cache"
                if person_error:
                    # A cache hit is not complete while visible people are missing.
                    # Reinspect only the source frames and persist the repaired aliases;
                    # never silently accept an empty/incomplete person list as a hit.
                    emit("cache_person_repair", "缓存命中但人物观察不完整，正在用原视频帧补齐")
                    try:
                        repair_frames, repair_times = extract_frames(
                            vp, start, end, self.num_frames, return_times=True)
                        if not repair_frames:
                            raise RuntimeError("source frames unavailable for cached person repair")
                        raw_repair_outputs = []
                        repair_log = {}
                        repaired_fields = None
                        last_repair_error = person_error
                        for repair_attempt in range(2):
                            if cancel_event is not None and cancel_event.is_set():
                                raise PersonRepairCancelled("cached person repair cancelled")
                            repair_request_id = uuid.uuid4().hex
                            repair_api_url = self.next_api_url()
                            self._activate_request(repair_request_id, repair_api_url)
                            def request_cached_person_repair(repair_prompt, frame_bytes):
                                if repair_attempt > 0:
                                    repair_prompt += (
                                        "\nSecond source review: the existing cache is missing a visibly described "
                                        "human. Enumerate each distinct human visible in these exact frames and "
                                        "describe only visible appearance. Do not use numbered labels or age categories."
                                    )
                                if cancel_event is not None and cancel_event.is_set():
                                    raise PersonRepairCancelled("cached person repair cancelled")
                                upload = [("images", (f"cache_person_{i + 1}.jpg", data, "image/jpeg"))
                                          for i, data in enumerate(frame_bytes)]
                                response = _post_with_inference_cancel(
                                    f"{repair_api_url}/analyze_video", files=upload,
                                    data={"prompt": repair_prompt, "request_id": repair_request_id},
                                    timeout=900,
                                )
                                if cancel_event is not None and cancel_event.is_set() or response.status_code == 409:
                                    raise PersonRepairCancelled("cached person repair cancelled")
                                if response.status_code != 200:
                                    raise ValueError(
                                        f"cached person repair returned HTTP {response.status_code}")
                                raw_output = str(response.json().get("analysis") or "")
                                raw_repair_outputs.append(raw_output)
                                return raw_output
                            try:
                                candidate_fields, repair_log = _repair_person_references_from_video(
                                    cached_fields, len(repair_frames), repair_frames,
                                    request_cached_person_repair,
                                )
                            finally:
                                self._finish_request(repair_request_id)
                            last_repair_error = _person_observation_coverage_error(candidate_fields) or ""
                            if not last_repair_error:
                                repaired_fields = candidate_fields
                                break
                        hit["person_observation_repair"] = {
                            "status": "repaired_from_cache" if repaired_fields else "incomplete",
                            "method": "same_video_frames_cache_hit_repair",
                            "repaired_people": len((repaired_fields or candidate_fields or {}).get("person_observations") or []),
                            "attempts": len(raw_repair_outputs),
                            "frame_times_s": repair_times,
                            "repair_error": last_repair_error[:300],
                            "raw_model_output_preserved": True,
                        }
                        if raw_repair_outputs:
                            hit["person_observation_repair_raw"] = raw_repair_outputs
                        if repaired_fields:
                            hit["fields"] = repaired_fields
                            hit.pop("quality_warning", None)
                            hit["source"] = "cache_person_repair"
                            hit["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            with self._cache_lock:
                                self.cache[mid] = dict(hit)
                                self._dirty_cache_keys.add(mid)
                                self._save_cache()
                            cached_fields = repaired_fields
                            emit("cache_person_repair", "已从原视频帧补齐人物观察并更新缓存")
                        else:
                            hit["quality_warning"] = last_repair_error or person_error
                            emit("cache_person_repair_failed", "人物观察仍不完整；保留原始记忆并标记待核验")
                    except PersonRepairCancelled:
                        return self._cancelled_result()
                    except Exception as repair_exc:
                        hit["person_observation_repair"] = {
                            "status": "failed", "method": "same_video_frames_cache_hit_repair",
                            "repair_error": str(repair_exc)[:300],
                            "raw_model_output_preserved": True,
                        }
                        hit["quality_warning"] = person_error
                        emit("cache_person_repair_failed", f"人物补全失败，保留原始缓存：{str(repair_exc)[:120]}")
                else:
                    emit("cache", "命中 Qwen 缓存，直接读取已有场景事实")
                if hit.get("raw_model_output"):
                    hit["raw"] = hit["raw_model_output"]
                return hit
            if move_error:
                emit("cache_quality", "缓存中的物品移动缺少前后状态证据，将重新核验")
            if fact_error:
                emit("cache_quality", "缓存的事件顺序或动作事实覆盖不足，正在按原视频补齐")
            # Keep invalid current-version output available for audit until a
            # validated replacement is written; this path still skips the hit.

        if vp is None:
            return {"source": "error", "error": "视频文件不存在", "fields": None}

        t0 = time.time()
        try:
            emit("extract_frames", "正在从视频片段抽取关键帧")
            bufs, frame_times_s = extract_frames(
                vp, start, end, self.num_frames, return_times=True)
            base_prompt = build_prompt(len(bufs))
            files = [("images", (f"f{i}.jpg", b, "image/jpeg"))
                     for i, b in enumerate(bufs)]
            raw = ""
            raw_model_output = None
            fields = None
            candidate_fields = None
            person_repair_metadata = None
            temporal_fact_repair_metadata = None
            quality_repair_metadata = None

            def sanitize_unverified_moves(candidate):
                nonlocal quality_repair_metadata
                if not validate_source_fact_quality:
                    return candidate
                sanitized, dropped = _drop_unverified_object_moves(candidate)
                if dropped:
                    if quality_repair_metadata is None:
                        quality_repair_metadata = {
                            "method": "drop_unverified_object_moves",
                            "removed_count": 0,
                            "removed_facts": [],
                        }
                    quality_repair_metadata["removed_count"] += len(dropped)
                    quality_repair_metadata["removed_facts"].extend(dropped)
                    emit("repair_temporal_facts",
                         f"已排除 {len(dropped)} 条缺少前后状态证据的物品移动事实")
                return sanitized

            def request_person_repair(prompt, image_files):
                if cancel_event is not None and cancel_event.is_set():
                    raise PersonRepairCancelled("person repair cancelled")
                repair_request_id = uuid.uuid4().hex
                repair_api_url = self.next_api_url()
                self._activate_request(repair_request_id, repair_api_url)
                try:
                    emit("repair_people", "正在使用当前视频帧补全人物外观")
                    response = _post_with_inference_cancel(
                        f"{repair_api_url}/analyze_video",
                        files=image_files,
                        data={"prompt": prompt, "request_id": repair_request_id},
                        timeout=900,
                    )
                finally:
                    self._finish_request(repair_request_id)
                if ((cancel_event is not None and cancel_event.is_set())
                        or response.status_code == 409):
                    raise PersonRepairCancelled("person repair cancelled")
                if response.status_code != 200:
                    raise ValueError(
                        f"focused person pass returned HTTP {response.status_code}"
                    )
                return response.json().get("analysis", "")

            def repair_person_references(candidate):
                return _repair_person_references_from_video(
                    candidate, len(bufs), files, request_person_repair
                )

            def record_person_repair(repair_log):
                nonlocal quality_repair_metadata, person_repair_metadata
                if quality_repair_metadata is None:
                    quality_repair_metadata = {}
                quality_repair_metadata["person_reference_repair"] = repair_log
                if repair_log.get("status") == "repaired" or repair_log.get("repaired_people"):
                    person_repair_metadata = repair_log
                if repair_log.get("status") == "repaired":
                    emit("repair_people", "人物外观已由当前视频帧补全，原始输出保留为来源记录")
                elif repair_log.get("status") == "removed_after_repair_failure":
                    emit("repair_people", "视频帧补全失败，已记录原因并仅移除未核实人物")
                else:
                    emit("repair_people", "人物补全未通过校验，处理结果与原因已记录")

            validation_error = None
            validation_error_kind = None
            for attempt in range(4):
                if cancel_event is not None and cancel_event.is_set():
                    return self._cancelled_result()
                prompt = base_prompt
                if validation_error_kind == "scene_fact_quality":
                    prompt += ("\n\nThe previous answer was rejected because its own scene facts were incomplete: "
                               + str(validation_error)[:300] + ". Inspect the supplied video frames again and "
                               "correct the structured fields. Every visible person referenced in Temporal Facts "
                               "must have one Person Observations row using the exact same descriptive visual alias; "
                               "describe body/clothing separately from face_features. Age may be recorded only when legible on-screen text explicitly "
                               "links it to this person; never estimate age from appearance. "
                               "Do not infer identity, role, or relationship. Use [] only if no person is visible in "
                               "any supplied frame. An object may be called moved only when the frames support both "
                               "its before and after location or state; include frame evidence, otherwise keep only "
                               "the state that is directly visible. Return the complete valid JSON arrays.")
                elif validation_error_kind == "schema":
                    prompt += ("\n\nThe previous response was rejected because its structured JSON was incomplete or invalid: "
                               + str(validation_error)[:180] + ". Regenerate the entire answer from the source. "
                               "Person Observations and Temporal Facts must each be one complete valid JSON array on one line. "
                               "Use [] when empty. Do not truncate either array, use comments, or add text after the final field.")
                elif validation_error_kind == "language":
                    prompt += ("\n\nRewrite the previous answer. It violated the English-only rule. "
                               "Translate or paraphrase every phrase into English; never include CJK characters. "
                               "For an idiom whose meaning is uncertain, say it is an unclear colloquial phrase.")
                elif validation_error_kind == "fallback":
                    prompt += ("\n\nFinal rewrite using only visible frame pixels. Do not transcribe or summarize "
                               "speech, and do not infer dialogue. Every field value must contain English text only "
                               "and zero CJK characters. "
                               "Return complete valid JSON arrays, using [] when empty.")
                elif validation_error_kind == "final_english_only":
                    prompt = (
                        "Use only these video frames. Ignore all speech and all writing/signs, especially "
                        "non-English writing; do not quote or transcribe any text. Return concise English only. "
                        "Output exactly these fields, each on its own line: Scene (one sentence describing the "
                        "visible event and action sequence); Location (broad visible place or blank); Time/Occasion "
                        "(only clearly readable time/date or blank); People (visually readable names only or blank); "
                        "Person Observations (one complete JSON array of descriptive visual aliases and visible "
                        "appearance, face_features, frame_index, evidence, and text-supported age only; [] if none); User Action (camera wearer only or blank); "
                        "Activity (specific visible activity); Objects (important visible items); Environment "
                        "(short visible setting); Interaction Details (visible actions only); Temporal Facts "
                        "(one complete JSON array of source-backed facts, [] if none). Do not infer identity, age, "
                        "plans, dialogue, diagnosis, advice, or unseen movement. Return JSON arrays complete and "
                        "do not add any text after the fields."
                    )
                request_id = uuid.uuid4().hex
                api_url = self.next_api_url()
                self._activate_request(request_id, api_url)
                try:
                    emit("model_inference", f"正在请求 Qwen3-VL 分析 {len(bufs)} 帧（尝试 {attempt + 1}/4）")
                    resp = _post_with_inference_cancel(
                        f"{api_url}/analyze_video",
                        files=files,
                        data={"prompt": prompt, "request_id": request_id},
                        timeout=900,
                    )
                except Exception:
                    if cancel_event is not None and cancel_event.is_set():
                        return self._cancelled_result()
                    raise
                finally:
                    self._finish_request(request_id)
                if (cancel_event is not None and cancel_event.is_set()) or resp.status_code == 409:
                    return self._cancelled_result()
                if resp.status_code != 200:
                    error = f"API {resp.status_code}: {resp.text[:200]}"
                    fallback = stale_cache_fallback(error)
                    if fallback is not None:
                        return fallback
                    return {"source": "error", "error": error, "fields": None}
                raw = resp.json().get("analysis", "")
                raw_model_output = raw
                emit("parse_output", "模型已返回，正在校验语言并整理 Scene / Action 等字段")
                if _contains_cjk(raw):
                    validation_error = "output contains non-English characters"
                    if attempt < 2:
                        # First ask for an English rewrite, then explicitly omit signage and speech.
                        validation_error_kind = "fallback" if attempt >= 1 else "language"
                        continue
                    raw = _sanitize_cjk_spans(raw)
                    try:
                        fields = parse_analysis(raw, strict=True)
                        fields = sanitize_unverified_moves(fields)
                        candidate_fields = fields
                        coverage_error = (_structured_fact_quality_error(fields)
                                          if validate_source_fact_quality else "")
                        if coverage_error:
                            validation_error = coverage_error
                            validation_error_kind = "scene_fact_quality"
                            fields = None
                            emit("validate_people", "人物观察与场景事实不一致，正在补齐来源明确的外观记录")
                            if (validate_source_fact_quality
                                    and _person_observation_coverage_error(candidate_fields)
                                    and not _object_move_evidence_error(candidate_fields)):
                                break
                            continue
                        if _has_usable_english_memory(fields):
                            validation_error = None
                            validation_error_kind = None
                            break
                        validation_error = "no substantive English memory remained after redaction"
                    except ValueError as exc:
                        validation_error = str(exc)
                    validation_error_kind = (
                        "final_english_only" if attempt == 2 else "language"
                    )
                    continue
                try:
                    fields = parse_analysis(raw, strict=True)
                    fields = sanitize_unverified_moves(fields)
                    candidate_fields = fields
                    coverage_error = (_structured_fact_quality_error(fields)
                                      if validate_source_fact_quality else "")
                    if coverage_error:
                        validation_error = coverage_error
                        validation_error_kind = "scene_fact_quality"
                        fields = None
                        emit("validate_people", "人物观察与场景事实不一致，正在补齐来源明确的外观记录")
                        if (validate_source_fact_quality
                                and _person_observation_coverage_error(candidate_fields)
                                and not _object_move_evidence_error(candidate_fields)):
                            break
                        continue
                    validation_error = None
                    break
                except ValueError as exc:
                    validation_error = str(exc)
                    validation_error_kind = "schema"
                    emit("validate_output", "结构化字段不完整，正在要求模型重写后重新校验")
                    if "Temporal Facts is not a complete JSON array" in validation_error:
                        try:
                            recovered = _recover_truncated_temporal_fact_prefix(
                                raw, len(bufs), validate_source_fact_quality
                            )
                            if recovered:
                                raw, fields, recovered_rows, recovered_quality_repair = recovered
                                candidate_fields = fields
                                temporal_fact_repair_metadata = {
                                    "source": "same_video_frames_truncated_prefix",
                                    "facts_added": len(recovered_rows),
                                }
                                if recovered_quality_repair:
                                    if quality_repair_metadata is None:
                                        quality_repair_metadata = {}
                                    quality_repair_metadata.update(recovered_quality_repair)
                                validation_error = None
                                validation_error_kind = None
                                emit("repair_temporal_facts",
                                     "Recovered complete video-grounded facts before truncated output")
                                break
                        except (ValueError, TypeError, json.JSONDecodeError) as prefix_exc:
                            validation_error = (
                                f"{validation_error}; safe prefix recovery failed: "
                                f"{str(prefix_exc)[:120]}"
                            )
            if (fields is None and validation_error_kind == "schema"
                    and "Person Observations is not a complete JSON array"
                    in str(validation_error or "")):
                try:
                    base_raw = _replace_field_value(raw, "Person Observations", "[]")
                    if base_raw is None:
                        raise ValueError("Person Observations field was not found")
                    base_fields = parse_analysis(base_raw, strict=True)
                    references = _person_reference_labels(base_fields)
                    repair_prompt = (
                        "Inspect ONLY the supplied video frames. Return one complete valid JSON object with "
                        "person_observations and entity_map. For every clearly visible person, assign or reuse "
                        "a short stable descriptive visual alias from visible traits; never number people. "
                        "Put clearly visible facial details in face_features; describe body/clothing/posture in appearance. Leave face_features empty if unclear. "
                        "Record age only when legible on-screen text explicitly links it to this person; never estimate age from appearance. "
                        "Do not guess real identity, job, or relationship, "
                        "or use information from outside the frames. Include frame_index and visual evidence. If no person "
                        "is visible, use an empty array. Earlier person references generated from these same frames: "
                        + json.dumps(references, ensure_ascii=True)
                        + ". Add an entity_map row only when pixels clearly map a reference to one descriptive alias; "
                        "omit uncertain mappings. Schema: {\"person_observations\":[{\"entity\":\"person wearing a blue jacket\",\"appearance\":\"visible body/clothing details\",\"face_features\":\"clear facial details, or empty\",\"age\":\"\",\"age_basis\":\"unknown\",\"frame_index\":1,\"evidence\":\"visible detail\",\"confidence\":\"high|medium|low\"}],\"entity_map\":[{\"reference\":\"source label\",\"entity\":\"person wearing a blue jacket\"}]}. Return JSON only."
                    )
                    repair_request_id = uuid.uuid4().hex
                    repair_api_url = self.next_api_url()
                    self._activate_request(repair_request_id, repair_api_url)
                    try:
                        emit("repair_people", "正在用同一组视频帧修复不完整的人物数组")
                        repair_response = _post_with_inference_cancel(
                            f"{repair_api_url}/analyze_video",
                            files=files,
                            data={"prompt": repair_prompt, "request_id": repair_request_id},
                            timeout=900,
                        )
                    finally:
                        self._finish_request(repair_request_id)
                    if (cancel_event is not None and cancel_event.is_set()) or repair_response.status_code == 409:
                        return self._cancelled_result()
                    if repair_response.status_code != 200:
                        raise ValueError(f"focused person pass returned HTTP {repair_response.status_code}")
                    repair_payload = _parse_person_repair(
                        repair_response.json().get("analysis", ""))
                    if _contains_cjk(json.dumps(repair_payload, ensure_ascii=False)):
                        raise ValueError("focused person pass returned non-English text")
                    rows = repair_payload.get("person_observations")
                    if not isinstance(rows, list):
                        raise ValueError("focused person pass omitted person_observations array")
                    patched_raw = _replace_field_value(
                        raw, "Person Observations", json.dumps(rows, ensure_ascii=False))
                    if patched_raw is None:
                        raise ValueError("Person Observations field was not found")
                    repaired_fields = parse_analysis(patched_raw, strict=True)
                    repaired_fields = _apply_person_repair(
                        repaired_fields, repair_payload, references, len(bufs))
                    repaired_fields = sanitize_unverified_moves(repaired_fields)
                    repair_quality_error = (
                        _structured_fact_quality_error(repaired_fields)
                        if validate_source_fact_quality else "")
                    if repair_quality_error:
                        raise ValueError(repair_quality_error)
                    fields = repaired_fields
                    raw = patched_raw
                    candidate_fields = fields
                    person_repair_metadata = {
                        "source": "same_video_frames_schema_repair",
                        "observations_added": len(fields.get("person_observations") or []),
                        "references_reconciled": len(repair_payload.get("entity_map") or []),
                    }
                    validation_error = None
                    validation_error_kind = None
                    emit("repair_people", "人物数组已修复并通过结构校验")
                except (ValueError, TypeError, json.JSONDecodeError, requests.RequestException) as repair_exc:
                    validation_error = f"{validation_error}; focused person array repair failed: {repair_exc}"
                    # The model may truncate the observations array even though the
                    # narrative and event facts are usable. Reuse the standard
                    # same-video repair-then-salvage path before rejecting the clip.
                    try:
                        salvaged_fields, removed_people, salvage_log = _salvage_person_reference_gaps(
                            base_fields, repair_person_references
                        )
                        if salvaged_fields is not None:
                            salvage_quality_error = (
                                _structured_fact_quality_error(salvaged_fields)
                                if validate_source_fact_quality else ""
                            )
                            if salvage_quality_error:
                                raise ValueError(salvage_quality_error)
                            fields = salvaged_fields
                            candidate_fields = fields
                            raw_model_output = raw
                            person_repair_metadata = salvage_log
                            quality_repair_metadata = {
                                "person_reference_repair": salvage_log,
                                "person_array_schema_failure": str(repair_exc)[:300],
                            }
                            validation_error = None
                            validation_error_kind = None
                            emit(
                                "repair_people",
                                "人物数组异常：已先检查同段视频补全；仅对无法核实的引用记录剔除与原因"
                            )
                    except PersonRepairCancelled:
                        return self._cancelled_result()
                    except Exception as salvage_exc:
                        validation_error = (
                            f"{validation_error}; repair-then-salvage failed: "
                            f"{str(salvage_exc)[:180]}"
                        )

            if (fields is None and validate_source_fact_quality
                    and candidate_fields is not None
                    and validation_error_kind == "scene_fact_quality"
                    and _person_observation_coverage_error(candidate_fields)):
                references = _person_reference_labels(candidate_fields)
                repair_schema = (
                    "{\"person_observations\":[{\"entity\":\"person wearing a blue jacket\","
                    "\"appearance\":\"visible body/clothing/posture details\",\"face_features\":\"clear visible facial details, or empty\","
                    "\"frame_index\":1,\"evidence\":\"visible detail in frame 1\","
                    "\"confidence\":\"high|medium|low\"}],"
                    "\"entity_map\":[{\"reference\":\"existing label\",\"entity\":\"person wearing a blue jacket\"}]}"
                )
                id_instruction = (
                    "Create descriptive aliases from visible traits, never numbered labels. Map these existing "
                    "person references to a new alias only when the frames clearly support it: "
                    + json.dumps(references, ensure_ascii=True)
                )
                repair_prompt = (
                    "Inspect ONLY the supplied video frames. Return one valid JSON object with key "
                    "person_observations. For every visibly present person, "
                    + id_instruction
                    + ". Reuse exactly the same descriptive alias when the same individual appears in multiple frames. "
                    "Return one row per distinct person, not a separate row for each frame. Describe only "
                    "visible face features when clear, plus clothing and distinguishing visual features. "
                    "Record age only when legible on-screen text explicitly links it to this person; never estimate age from appearance. "
                    "Do not guess real identity, job, or relationship, "
                    "or use information from outside these frames. Include evidence and a 1-based frame_index. If no person is visible, return an empty "

                    "array. Schema: " + repair_schema
                )
                request_id = uuid.uuid4().hex
                api_url = self.next_api_url()
                self._activate_request(request_id, api_url)
                try:
                    emit("repair_people", "正在用同一组视频帧补齐人物外观和实体对应关系")
                    repair_response = _post_with_inference_cancel(
                        f"{api_url}/analyze_video",
                        files=files,
                        data={"prompt": repair_prompt, "request_id": request_id},
                        timeout=900,
                    )
                except Exception:
                    repair_response = None
                finally:
                    self._finish_request(request_id)
                if (cancel_event is not None and cancel_event.is_set()) or (
                        repair_response is not None and repair_response.status_code == 409):
                    return self._cancelled_result()
                try:
                    if repair_response is not None and repair_response.status_code == 200:
                        repair_raw = repair_response.json().get("analysis", "")
                        repair_payload = _parse_person_repair(repair_raw)
                        repaired_fields = _apply_person_repair(
                            candidate_fields, repair_payload, references, len(bufs))
                        repair_error = _structured_fact_quality_error(repaired_fields)
                        references_reconciled = len(repair_payload.get("entity_map") or [])
                        if repair_error and repaired_fields.get("person_observations"):
                            observed_ids = {
                                " ".join(str(row.get("entity") or "").casefold().split())
                                for row in repaired_fields.get("person_observations") or []
                                if isinstance(row, dict)
                            }
                            unresolved = [
                                label for label in references
                                if " ".join(label.casefold().split()) not in observed_ids
                            ]
                            if unresolved:
                                alias_context = []
                                for label in unresolved:
                                    facts = []
                                    for fact in candidate_fields.get("temporal_facts") or []:
                                        if not isinstance(fact, dict):
                                            continue
                                        related = [fact.get("subject"), fact.get("actor")]
                                        related.extend(fact.get("affected_people") or [])
                                        if any(" ".join(str(value or "").casefold().split())
                                               == " ".join(label.casefold().split())
                                               for value in related):
                                            evidence = str(fact.get("evidence") or fact.get("value") or "").strip()
                                            if evidence and evidence not in facts:
                                                facts.append(evidence[:160])
                                    alias_context.append({"reference": label, "clip_facts": facts[:3]})
                                visible_people = [
                                    {
                                        "entity": row.get("entity"),
                                        "appearance": row.get("appearance"),
                                        "frame_index": row.get("frame_index"),
                                    }
                                    for row in repaired_fields.get("person_observations") or []
                                    if isinstance(row, dict)
                                ]
                                alias_prompt = (
                                    "Inspect ONLY these same video frames. The reference labels and clip facts "
                                    "below were generated by a prior pass on these frames; they are not external "
                                    "captions, transcripts, or annotations. The visible_people list contains "
                                    "descriptive visual aliases and appearances already extracted from these frames. Map a "
                                    "reference to one descriptive alias only when frame evidence and clip facts "
                                    "clearly identify the same person. A role word is not a verified identity; "
                                    "do not infer a job, name, relationship, or intent. Return one JSON object "
                                    "only using this schema: "
                                    + json.dumps({"entity_map": [
                                        {"reference": "source label",
                                         "entity": "person wearing a blue jacket"}
                                    ]}, ensure_ascii=False)
                                    + ". Omit any uncertain mapping. References: "
                                    + json.dumps(alias_context, ensure_ascii=False)
                                    + ". Visible people: "
                                    + json.dumps(visible_people, ensure_ascii=False)
                                )
                                alias_request_id = uuid.uuid4().hex
                                alias_api_url = self.next_api_url()
                                self._activate_request(alias_request_id, alias_api_url)
                                try:
                                    emit("repair_people", "正在核对人物记录与本片中已识别的人物对应关系")
                                    alias_response = _post_with_inference_cancel(
                                        f"{alias_api_url}/analyze_video",
                                        files=files,
                                        data={"prompt": alias_prompt,
                                              "request_id": alias_request_id},
                                        timeout=900,
                                    )
                                except Exception:
                                    alias_response = None
                                finally:
                                    self._finish_request(alias_request_id)
                                if (cancel_event is not None and cancel_event.is_set()) or (
                                        alias_response is not None
                                        and alias_response.status_code == 409):
                                    return self._cancelled_result()
                                if alias_response is not None and alias_response.status_code == 200:
                                    alias_payload = _parse_person_repair(
                                        alias_response.json().get("analysis", ""))
                                    mapped_fields = _apply_person_repair(
                                        repaired_fields, alias_payload, references, len(bufs))
                                    map_rows = alias_payload.get("entity_map") or []
                                    references_reconciled += (
                                        len(map_rows) if isinstance(map_rows, list)
                                        else len(map_rows) if isinstance(map_rows, dict) else 0
                                    )
                                    mapped_error = _structured_fact_quality_error(mapped_fields)
                                    if not mapped_error:
                                        repaired_fields = mapped_fields
                                        repair_error = ""
                                    else:
                                        repair_error = mapped_error
                        if not repair_error:
                            fields = repaired_fields
                            person_repair_metadata = {
                                "source": "same_video_frames",
                                "observations_added": len(repaired_fields.get("person_observations") or []),
                                "references_reconciled": references_reconciled,
                            }
                            validation_error = None
                            validation_error_kind = None
                            emit("repair_people", "人物外观已由原视频帧补齐并通过结构校验")
                        else:
                            validation_error = repair_error
                    elif repair_response is not None:
                        validation_error = (
                            f"{validation_error}; focused person pass returned "
                            f"HTTP {repair_response.status_code}"
                        )
                except (ValueError, TypeError, json.JSONDecodeError) as repair_exc:
                    validation_error = f"{validation_error}; focused person pass failed: {repair_exc}"

            if (fields is None and validation_error_kind == "schema"
                    and "Temporal Facts is not a complete JSON array"
                    in str(validation_error or "")):
                try:
                    recovered = _recover_truncated_temporal_fact_prefix(
                        raw, len(bufs), validate_source_fact_quality
                    )
                    if recovered:
                        raw, fields, recovered_rows, recovered_quality_repair = recovered
                        candidate_fields = fields
                        temporal_fact_repair_metadata = {
                            "source": "same_video_frames_truncated_prefix",
                            "facts_added": len(recovered_rows),
                        }
                        if recovered_quality_repair:
                            if quality_repair_metadata is None:
                                quality_repair_metadata = {}
                            quality_repair_metadata.update(recovered_quality_repair)
                        validation_error = None
                        validation_error_kind = None
                        emit("repair_temporal_facts",
                             "Recovered complete video-grounded facts before truncated output")
                except (ValueError, TypeError, json.JSONDecodeError) as prefix_exc:
                    validation_error = (
                        f"{validation_error}; safe prefix recovery failed: "
                        f"{str(prefix_exc)[:120]}"
                    )

            if (fields is None and validation_error_kind == "schema"
                    and "Temporal Facts is not a complete JSON array"
                    in str(validation_error or "")
                    and "Temporal Facts" in str(raw or "")
                    and "Person Observations" in str(raw or "")):
                try:
                    base_raw = _replace_field_value(raw, "Temporal Facts", "[]")
                    if base_raw is None:
                        raise ValueError("Temporal Facts field was not found")
                    parse_analysis(base_raw, strict=True)
                    repair_prompt = _build_temporal_fact_repair_prompt(len(bufs))
                    repair_request_id = uuid.uuid4().hex
                    repair_api_url = self.next_api_url()
                    self._activate_request(repair_request_id, repair_api_url)
                    try:
                        emit("repair_temporal_facts", "Recovering structured event facts from the same video frames")
                        repair_response = _post_with_inference_cancel(
                            f"{repair_api_url}/analyze_video",
                            files=files,
                            data={"prompt": repair_prompt, "request_id": repair_request_id},
                            timeout=900,
                        )
                    finally:
                        self._finish_request(repair_request_id)
                    if ((cancel_event is not None and cancel_event.is_set())
                            or repair_response.status_code == 409):
                        return self._cancelled_result()
                    if repair_response.status_code != 200:
                        raise ValueError(
                            f"focused temporal-fact pass returned HTTP {repair_response.status_code}"
                        )
                    repair_raw = repair_response.json().get("analysis", "")
                    if _contains_cjk(repair_raw):
                        raise ValueError("focused temporal-fact pass returned non-English text")
                    rows = _apply_temporal_fact_repair(
                        _parse_json_array(repair_raw), len(bufs)
                    )
                    patched_raw = _replace_field_value(
                        base_raw, "Temporal Facts", json.dumps(rows, ensure_ascii=False)
                    )
                    if patched_raw is None:
                        raise ValueError("Temporal Facts field was not found")
                    repaired_fields = parse_analysis(patched_raw, strict=True)
                    quality_error = _structured_fact_quality_error(repaired_fields)
                    if quality_error:
                        raise ValueError(quality_error)
                    fields = repaired_fields
                    raw = patched_raw
                    candidate_fields = fields
                    temporal_fact_repair_metadata = {
                        "source": "same_video_frames_schema_repair",
                        "facts_added": len(rows),
                    }
                    validation_error = None
                    validation_error_kind = None
                    emit("repair_temporal_facts", "Event facts repaired and source-validated")
                except (ValueError, TypeError, json.JSONDecodeError, requests.RequestException) as repair_exc:
                    validation_error = (
                        f"{validation_error}; focused temporal-fact repair failed: {repair_exc}"
                    )

            if (fields is None and validate_source_fact_quality
                    and candidate_fields is not None
                    and validation_error_kind == "scene_fact_quality"
                    and not _person_observation_coverage_error(candidate_fields)
                    and not _object_move_evidence_error(candidate_fields)
                    and _temporal_fact_coverage_error(candidate_fields)):
                try:
                    existing_facts = [row for row in candidate_fields.get("temporal_facts") or []
                                      if isinstance(row, dict)]
                    repair_limit = min(8, max(0, 16 - len(existing_facts)))
                    if repair_limit <= 0:
                        raise ValueError("no room remains for missing atomic facts")
                    repair_prompt = _build_temporal_fact_coverage_repair_prompt(
                        candidate_fields, len(bufs), repair_limit
                    )
                    repair_request_id = uuid.uuid4().hex
                    repair_api_url = self.next_api_url()
                    self._activate_request(repair_request_id, repair_api_url)
                    try:
                        emit("repair_temporal_facts", "正在用同一组视频帧补齐缺失的原子事件事实")
                        repair_response = _post_with_inference_cancel(
                            f"{repair_api_url}/analyze_video",
                            files=files,
                            data={"prompt": repair_prompt, "request_id": repair_request_id},
                            timeout=900,
                        )
                    finally:
                        self._finish_request(repair_request_id)
                    if ((cancel_event is not None and cancel_event.is_set())
                            or repair_response.status_code == 409):
                        return self._cancelled_result()
                    if repair_response.status_code != 200:
                        raise ValueError(
                            f"focused temporal coverage pass returned HTTP {repair_response.status_code}"
                        )
                    repair_raw = str(repair_response.json().get("analysis") or "")
                    if _contains_cjk(repair_raw):
                        raise ValueError("focused temporal coverage pass returned non-English text")
                    additions = _apply_temporal_fact_repair(
                        _parse_json_array(repair_raw), len(bufs)
                    )
                    if not additions:
                        raise ValueError("focused temporal coverage pass returned no grounded facts")
                    combined = list(existing_facts)
                    seen = {
                        (str(row.get("subject") or "").casefold(),
                         str(row.get("predicate") or "").casefold(),
                         str(row.get("value") or "").casefold())
                        for row in combined
                    }
                    for row in additions:
                        key = (str(row.get("subject") or "").casefold(),
                               str(row.get("predicate") or "").casefold(),
                               str(row.get("value") or "").casefold())
                        if key not in seen:
                            combined.append(row)
                            seen.add(key)
                    if len(combined) > 16:
                        raise ValueError("combined temporal fact array exceeds the 16-row limit")
                    patched_raw = _replace_field_value(
                        raw, "Temporal Facts", json.dumps(combined, ensure_ascii=False)
                    )
                    if patched_raw is None:
                        raise ValueError("Temporal Facts field was not found")
                    repaired_fields = parse_analysis(patched_raw, strict=True)
                    repaired_fields = sanitize_unverified_moves(repaired_fields)
                    coverage_error = _structured_fact_quality_error(repaired_fields)
                    if coverage_error:
                        raise ValueError(coverage_error)
                    raw_model_output = raw
                    raw = patched_raw
                    fields = repaired_fields
                    candidate_fields = fields
                    temporal_fact_repair_metadata = {
                        "source": "same_video_frames_atomic_coverage_repair",
                        "facts_added": len(combined) - len(existing_facts),
                    }
                    validation_error = None
                    validation_error_kind = None
                    emit("repair_temporal_facts", "原子事件事实已补齐并通过来源校验")
                except (ValueError, TypeError, json.JSONDecodeError, requests.RequestException) as repair_exc:
                    validation_error = (
                        f"{validation_error}; focused temporal coverage repair failed: "
                        f"{str(repair_exc)[:240]}"
                    )

            if (fields is None and validate_source_fact_quality
                    and candidate_fields is not None
                    and validation_error_kind == "scene_fact_quality"):
                try:
                    sanitized_fields, removed_person_claims, repair_log = (
                        _salvage_person_reference_gaps(
                            candidate_fields, repair_person_references
                        )
                    )
                except PersonRepairCancelled:
                    return self._cancelled_result()
                record_person_repair(repair_log)
                if sanitized_fields is not None:
                    fields = sanitized_fields
                    validation_error = None
                    validation_error_kind = None

            # If the model exhausted the same-frame repair and emitted no complete
            # Temporal Facts row at all, keep the independently parsed scene,
            # people, location, activity, objects, and environment. This fallback
            # is limited to an actually open array and still enforces the same
            # person/object evidence gates; it never fabricates or partially parses
            # a fact row.
            if (fields is None and validation_error_kind == "schema"
                    and "Temporal Facts is not a complete JSON array"
                    in str(validation_error or "")):
                temporal_text = _field(raw, "Temporal Facts").strip()
                partial_rows, complete_array = _parse_json_array_prefix(temporal_text)
                if (temporal_text.startswith("[") and not complete_array
                        and not partial_rows):
                    try:
                        scene_only_raw = _replace_field_value(raw, "Temporal Facts", "[]")
                        if scene_only_raw is None:
                            raise ValueError("Temporal Facts field was not found")
                        scene_only_fields = parse_analysis(scene_only_raw, strict=True)
                        quality_error = (_structured_fact_quality_error(scene_only_fields)
                                         if validate_source_fact_quality else "")
                        if quality_error:
                            raise ValueError(quality_error)
                        if not _has_usable_english_memory(scene_only_fields):
                            raise ValueError("no substantive memory remained after removing the incomplete array")
                        fields = scene_only_fields
                        raw_model_output = raw
                        raw = scene_only_raw
                        candidate_fields = fields
                        temporal_fact_repair_metadata = {
                            "source": "same_video_frames_incomplete_array_empty_fallback",
                            "facts_added": 0,
                            "complete_facts_retained": 0,
                            "raw_model_output_invalid": True,
                        }
                        validation_error = None
                        validation_error_kind = None
                        emit("repair_temporal_facts",
                             "Kept validated scene details; the incomplete event-fact array was omitted")
                    except (ValueError, TypeError, json.JSONDecodeError) as fallback_exc:
                        validation_error = (
                            f"{validation_error}; safe scene-only fallback failed: "
                            f"{str(fallback_exc)[:120]}"
                        )

            if fields is None:
                fallback = stale_cache_fallback(
                    validation_error or "current analysis did not pass validation"
                )
                if fallback is not None:
                    return fallback
                if validation_error_kind == "language":
                    reason = "Qwen output remained non-English after retry; rejected from memory"
                elif validation_error_kind == "scene_fact_quality":
                    reason = f"关键人物观察或物品变化缺少视频证据，未写入: {validation_error or 'quality check failed'}"
                else:
                    reason = f"Structured facts rejected after retry: {validation_error or 'invalid output'}"
                return {
                    "source": "error",
                    "error": reason,
                    "fields": None,
                    "person_observation_repair": person_repair_metadata,
                    "quality_repair": quality_repair_metadata,
                }
        except Exception as e:
            fallback = stale_cache_fallback(str(e))
            if fallback is not None:
                return fallback
            return {"source": "error", "error": str(e), "fields": None}
        elapsed = int((time.time() - t0) * 1000)
        out = {
            "fields": fields,
            "raw": raw,
            "elapsed_ms": elapsed,
            "num_frames": len(bufs) if bufs else 0,
            "frame_times_s": frame_times_s,
            "model": self.health().get("model", "Qwen3-VL"),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_version": PROMPT_VERSION,
            "list_parser_version": LIST_PARSER_VERSION,
            "source_input_fingerprint": input_fingerprint,
            "person_observation_repair": person_repair_metadata,
            "temporal_fact_repair": temporal_fact_repair_metadata,
            "quality_repair": quality_repair_metadata,
        }
        if raw_model_output is not None and raw_model_output != raw:
            out["raw_model_output"] = raw_model_output
        with self._cache_lock:
            self.cache[mid] = out
            self._dirty_cache_keys.add(mid)
            if persist_cache:
                self._save_cache()
        out["source"] = "qwen"
        return out

    def ground_objects(self, objects, frames, cancel_event=None, progress_callback=None):
        """Ground stored object labels in the retained keyframes without guessing."""
        candidates = []
        for value in objects or []:
            label = str(value or "").strip()
            if label and label.casefold() not in {x.casefold() for x in candidates}:
                candidates.append(label[:100])
            if len(candidates) >= 12:
                break
        if not candidates:
            return {"schema": 1, "status": "not_needed", "model": "Qwen3-VL", "items": []}
        if not frames:
            return {
                "schema": 1, "status": "no_frames", "model": "Qwen3-VL",
                "items": [{"object": label, "status": "not_verified",
                           "appearance": "", "frame_refs": []} for label in candidates],
            }

        def emit(stage, message):
            if progress_callback:
                try:
                    progress_callback({"stage": stage, "message": message})
                except Exception:
                    pass

        if cancel_event is not None and cancel_event.is_set():
            return {"schema": 1, "status": "cancelled", "model": "Qwen3-VL", "items": []}
        max_frames = min(7, len(frames))
        source_map = [
            {"image": i + 1, "time_s": round(float(frame.get("at_s") or 0), 2)}
            for i, frame in enumerate(frames[:max_frames])
        ]
        prompt = (
            "Inspect only the attached image pixels. Do not use event context, filenames, transcripts, "
            "or assumptions. For every candidate object below, return exactly one row in ONLY JSON: "
            '{"objects":[{"object":"exact candidate label","visible":true,'
            '"appearance":"brief visible color, shape, material, or position","frames":[1],'
            '"frame_boxes":[{"frame":1,"bbox":[0.20,0.30,0.48,0.62]}]}]}. '
            "Use the exact candidate label. Set visible=false, appearance empty, frames=[], frame_boxes=[] "
            "if no sampled frame clearly shows that object. Include only frames where that specific object "
            "is actually visible; never attach every frame to every object. For each visible frame, give one "
            "approximate tight bounding box around the whole object, with a small margin. bbox is normalized "
            "to the attached image dimensions as [x_min,y_min,x_max,y_max], each coordinate from 0 to 1, "
            "origin at the image top-left; do not return pixel coordinates. The crop should isolate the item "
            "while retaining a little surrounding context. If the item is too small or ambiguous to locate, "
            "omit its box but do not invent coordinates. Describe only properties supported by pixels; do not "
            "guess brand, identity, purpose, or hidden details. If visible in several frames, choose only "
            "the single clearest frame for the object. Return at most one frame number and one bounding box "
            "per object; do not reuse coordinates from one image for another. "
            "Frame-to-time map: " + json.dumps(source_map, ensure_ascii=False) +
            ". Candidate labels: " + json.dumps(candidates, ensure_ascii=False)
        )
        request_id = uuid.uuid4().hex
        api_url = self.next_api_url()
        emit("object_visual_grounding", "正在逐件核对物品与关键帧")
        self._activate_request(request_id, api_url)
        try:
            response = _post_with_inference_cancel(
                api_url + "/analyze_video",
                files=[("images", (f"object_frame_{i + 1}.webp",
                                   frame["bytes"], frame.get("mime") or "image/webp"))
                       for i, frame in enumerate(frames[:max_frames])],
                data={"prompt": prompt, "request_id": request_id},
                timeout=900,
            )
            if cancel_event is not None and cancel_event.is_set():
                return {"schema": 1, "status": "cancelled", "model": "Qwen3-VL", "items": []}
            if response.status_code == 409:
                return {"schema": 1, "status": "cancelled", "model": "Qwen3-VL", "items": []}
            if response.status_code != 200:
                raise ValueError(
                    f"visual grounding HTTP {response.status_code}: {response.text[:600]}"
                )
            raw = str((response.json() or {}).get("analysis") or "").strip()
            # Redact unsupported writing while keeping usable visual boxes; a
            # language-only issue must not force the same crop to be re-inferred forever.
            if _contains_cjk(raw):
                raw = _sanitize_cjk_spans(raw)
            payload = None
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                decoder = json.JSONDecoder()
                for position, char in enumerate(raw):
                    if char not in "{[":
                        continue
                    try:
                        candidate, _ = decoder.raw_decode(raw[position:])
                    except (TypeError, ValueError):
                        continue
                    if isinstance(candidate, list):
                        payload = {"objects": candidate}
                        break
                    if isinstance(candidate, dict) and any(
                            isinstance(candidate.get(key), list)
                            for key in ("objects", "items", "detections", "results")):
                        payload = candidate
                        break
            if isinstance(payload, list):
                payload = {"objects": payload}
            if isinstance(payload, dict) and not isinstance(payload.get("objects"), list):
                for key in ("items", "detections", "results"):
                    if isinstance(payload.get(key), list):
                        payload = {"objects": payload[key]}
                        break
            if (isinstance(payload, dict) and not isinstance(payload.get("objects"), list)
                    and isinstance(payload.get("object"), str)):
                payload = {"objects": [payload]}
            if not isinstance(payload, dict):
                raise ValueError("visual grounding response was not JSON")
            rows = payload.get("objects")
            rows = rows if isinstance(rows, list) else []
            by_label = {label.casefold(): label for label in candidates}
            def match_candidate_label(reported):
                raw_label = str(reported or "").strip()
                exact = by_label.get(raw_label.casefold())
                if exact:
                    return exact
                normalized = re.sub(r"[^a-z0-9]+", " ", raw_label.casefold()).strip()
                tokens = set(normalized.split())
                if not normalized:
                    return None
                matches = []
                for candidate in candidates:
                    candidate_norm = re.sub(r"[^a-z0-9]+", " ", candidate.casefold()).strip()
                    candidate_tokens = set(candidate_norm.split())
                    if (normalized in candidate_norm or candidate_norm in normalized
                            or (len(tokens) >= 2 and tokens.issubset(candidate_tokens))
                            or (len(tokens) == 1 and tokens == candidate_tokens)):
                        matches.append(candidate)
                return matches[0] if len(matches) == 1 else None
            parsed = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = match_candidate_label(row.get("object"))
                if not label:
                    continue
                indices = []
                for index in row.get("frames") or []:
                    if isinstance(index, int) and not isinstance(index, bool) and 1 <= index <= max_frames:
                        if index not in indices:
                            indices.append(index)
                frame_boxes = []
                raw_boxes = row.get("frame_boxes") or row.get("boxes") or []
                if isinstance(raw_boxes, list):
                    for box_row in raw_boxes:
                        if not isinstance(box_row, dict):
                            continue
                        index = box_row.get("frame", box_row.get("frame_index"))
                        bbox = box_row.get("bbox") or box_row.get("box")
                        if isinstance(bbox, dict):
                            bbox = [bbox.get(key) for key in ("x1", "y1", "x2", "y2")]
                        try:
                            coords = [float(value) for value in bbox]
                        except (TypeError, ValueError):
                            continue
                        if (not isinstance(index, int) or isinstance(index, bool)
                                or not 1 <= index <= max_frames or len(coords) != 4
                                or not all(math.isfinite(value) and 0 <= value <= 1 for value in coords)
                                or coords[2] - coords[0] < 0.015
                                or coords[3] - coords[1] < 0.015):
                            continue
                        if index not in indices:
                            indices.append(index)
                        frame_boxes.append({
                            "frame_index": index,
                            "bbox": [round(value, 5) for value in coords],
                        })
                if not frame_boxes and len(indices) == 1:
                    bbox = row.get("bbox")
                    if isinstance(bbox, dict):
                        bbox = [bbox.get(key) for key in ("x1", "y1", "x2", "y2")]
                    try:
                        coords = [float(value) for value in bbox]
                    except (TypeError, ValueError):
                        coords = []
                    if (len(coords) == 4 and all(math.isfinite(value) and 0 <= value <= 1 for value in coords)
                            and coords[2] - coords[0] >= 0.015 and coords[3] - coords[1] >= 0.015):
                        frame_boxes.append({
                            "frame_index": indices[0],
                            "bbox": [round(value, 5) for value in coords],
                        })
                indices = sorted(set(indices))[:1]
                frame_boxes = [box for box in frame_boxes if box["frame_index"] in indices][:1]
                visible = row.get("visible") is True and bool(indices)
                appearance = str(row.get("appearance") or "").strip()
                if re.fullmatch(r"unreadable text", appearance, re.I):
                    appearance = ""
                explicit_not_visible = row.get("visible") is False
                if not visible:
                    appearance = ""
                    indices = []
                    frame_boxes = []
                parsed[label.casefold()] = {
                    "object": label,
                    "status": ("grounded" if appearance else "visible_unlabeled" if visible
                               else "not_visible" if explicit_not_visible else "not_verified"),
                    "appearance": appearance[:240],
                    "frame_indices": indices[:2],
                    "frame_boxes": frame_boxes,
                }
            items = [parsed.get(label.casefold(), {
                "object": label, "status": "not_verified", "appearance": "",
                "frame_indices": [], "frame_boxes": [],
            }) for label in candidates]
            return {"schema": 2, "status": "completed", "model": "Qwen3-VL",
                    "prompt_version": "object-visual-grounding-v4-bbox-verified-labels", "items": items}
        except Exception as exc:
            return {
                "schema": 1, "status": "unavailable", "model": "Qwen3-VL",
                "message": str(exc)[:240],
                "items": [{"object": label, "status": "not_verified",
                           "appearance": "", "frame_refs": []} for label in candidates],
            }
        finally:
            self._finish_request(request_id)

    def ground_people(self, people, frames, cancel_event=None, progress_callback=None):
        """Locate clip-local people in sampled frames and return crop boxes only."""
        candidates, seen = [], set()
        for value in people or []:
            row = value if isinstance(value, dict) else {"entity": value}
            label = str(row.get("entity") or row.get("person") or "").strip()
            if not label or label.casefold() in seen:
                continue
            seen.add(label.casefold())
            candidates.append({"person": label[:100],
                               "appearance": str(row.get("appearance") or "")[:220]})
            if len(candidates) >= 8:
                break
        if not candidates:
            return {"schema": 1, "status": "not_needed", "model": "Qwen3-VL", "items": []}
        if not frames:
            return {"schema": 1, "status": "no_frames", "model": "Qwen3-VL",
                    "items": [{"person": row["person"], "status": "not_verified",
                               "appearance": "", "frame_refs": []} for row in candidates]}

        def emit(stage, message):
            if progress_callback:
                try:
                    progress_callback({"stage": stage, "message": message})
                except Exception:
                    pass

        if cancel_event is not None and cancel_event.is_set():
            return {"schema": 1, "status": "cancelled", "model": "Qwen3-VL", "items": []}
        max_frames = min(7, len(frames))
        source_map = [{"image": i + 1, "time_s": round(float(frame.get("at_s") or 0), 2)}
                      for i, frame in enumerate(frames[:max_frames])]
        prompt = (
            "Inspect only the attached image pixels. Match each candidate to a visibly distinct "
            "person using the optional clothing/appearance hint, not an assumed identity. Return "
            "ONLY JSON in this schema: {\"people\":[{\"person\":\"exact candidate label\","
            "\"visible\":true,\"frames\":[1],\"frame_boxes\":[{\"frame\":1,"
            "\"bbox\":[0.20,0.10,0.48,0.90]}]}]}. Use each exact candidate label once. "
            "A person is visible only when the pixels clearly support the match. Return only the "
            "single clearest frame per person. For a full-body person, give a normalized tight box "
            "around the visible body with a small margin. For a candidate whose label starts with 'camera wearer', "
            "a clearly visible hand, forearm, or foot is sufficient: box only that visible body part "
            "and set coverage='partial'. For other candidates set coverage='full' when the visible "
            "person is grounded. Coordinates are [x_min,y_min,x_max,y_max] from 0 to 1, origin top-left. "
            "Do not invent a box or reuse coordinates between frames. If the candidate is occluded, too small, "
            "or ambiguous, set visible=false and use empty frames and frame_boxes. If the person is "
            "visible=false and use empty frames and frame_boxes. Do not infer age, identity, "
            "relationships, or any attribute beyond the pixels. Hints: "
            + json.dumps(candidates, ensure_ascii=False)
            + ". Frame-to-time map: " + json.dumps(source_map, ensure_ascii=False)
        )
        request_id = uuid.uuid4().hex
        emit("person_visual_grounding", "正在定位人物画面并生成局部裁剪")
        with self._active_request_lock:
            self._active_request_id = request_id
        try:
            response = _post_with_inference_cancel(
                self.api_url + "/analyze_video",
                files=[("images", (f"person_frame_{i + 1}.webp", frame["bytes"],
                                    frame.get("mime") or "image/webp"))
                       for i, frame in enumerate(frames[:max_frames])],
                data={"prompt": prompt, "request_id": request_id}, timeout=900,
            )
            if cancel_event is not None and cancel_event.is_set() or response.status_code == 409:
                return {"schema": 1, "status": "cancelled", "model": "Qwen3-VL", "items": []}
            response.raise_for_status()
            raw = str((response.json() or {}).get("analysis") or "").strip()
            if _contains_cjk(raw):
                raise ValueError("person visual grounding response was not English")
            payload = None
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                decoder = json.JSONDecoder()
                for position, char in enumerate(raw):
                    if char != "{":
                        continue
                    try:
                        candidate, _ = decoder.raw_decode(raw[position:])
                    except (TypeError, ValueError):
                        continue
                    if isinstance(candidate, dict) and isinstance(candidate.get("people"), list):
                        payload = candidate
                        break
            if not isinstance(payload, dict):
                raise ValueError("person visual grounding response was not JSON")
            by_label = {row["person"].casefold(): row["person"] for row in candidates}
            parsed = {}
            for row in payload.get("people") or []:
                if not isinstance(row, dict):
                    continue
                label = by_label.get(str(row.get("person") or "").strip().casefold())
                if not label:
                    continue
                boxes = []
                raw_boxes = row.get("frame_boxes") or []
                if isinstance(raw_boxes, list):
                    for box_row in raw_boxes:
                        if not isinstance(box_row, dict):
                            continue
                        index = box_row.get("frame", box_row.get("frame_index"))
                        bbox = box_row.get("bbox") or box_row.get("box")
                        try:
                            coords = [float(part) for part in bbox]
                        except (TypeError, ValueError):
                            continue
                        if (not isinstance(index, int) or isinstance(index, bool)
                                or not 1 <= index <= max_frames or len(coords) != 4
                                or not all(math.isfinite(part) and 0 <= part <= 1 for part in coords)
                                or coords[2] - coords[0] < 0.025
                                or coords[3] - coords[1] < 0.06):
                            continue
                        boxes.append({"frame_index": index,
                                      "bbox": [round(part, 5) for part in coords]})
                boxes.sort(key=lambda item: item["frame_index"])
                box = boxes[0] if boxes else None
                visible = row.get("visible") is True and box is not None
                coverage = "partial" if label.casefold().startswith("camera wearer") and row.get("coverage") == "partial" else "full"
                parsed[label.casefold()] = {
                    "person": label,
                    "status": "grounded" if visible else "not_verified",
                    "coverage": coverage if visible else "unknown",
                    "appearance": next((item["appearance"] for item in candidates
                                         if item["person"] == label), "") if visible else "",
                    "frame_indices": [box["frame_index"]] if visible else [],
                    "frame_boxes": [box] if visible else [],
                }
            items = [parsed.get(row["person"].casefold(), {
                "person": row["person"], "status": "not_verified", "appearance": "",
                "frame_indices": [], "frame_boxes": [],
            }) for row in candidates]
            return {"schema": 1, "status": "completed", "model": "Qwen3-VL",
                    "prompt_version": "person-visual-grounding-v2-partial-body-box", "items": items}
        except Exception as exc:
            return {"schema": 1, "status": "unavailable", "model": "Qwen3-VL",
                    "message": str(exc)[:240],
                    "items": [{"person": row["person"], "status": "not_verified",
                               "appearance": "", "frame_refs": []} for row in candidates]}
        finally:
            with self._active_request_lock:
                if self._active_request_id == request_id:
                    self._active_request_id = None

    def prewarm(self, moments, force: bool = False):
        """批量预热缓存，返回 (成功数, 失败数)。"""
        ok = fail = 0
        for m in moments:
            r = self.perceive(m, force=force)
            if r.get("fields"):
                ok += 1
            else:
                fail += 1
        return ok, fail


# ---------------------------------------------------------------------------
# Golden label 评估（数据集标注只在这里用，绝不进记忆）
# ---------------------------------------------------------------------------

def _norm(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


def _match(q, g):
    """单项匹配：相等或互为子串。"""
    qn, gn = _norm(q), _norm(g)
    if not qn or not gn:
        return False
    return qn == gn or qn in gn or gn in qn


def golden_eval(fields: dict, gt: dict):
    """
    对比 Qwen 产出 vs 数据集 golden 标注。

    返回 {dim: {golden:[], qwen:[], hit:[], miss:[], extra:[]}, ...}
    """
    if not fields:
        return {}

    dims = [
        ("people", fields.get("people") or [], gt.get("people_gt") or []),
        ("locations", [fields.get("location")] if fields.get("location") else [],
         gt.get("location_gt") or []),
        ("activity_events", [fields.get("activity")] if fields.get("activity") else [],
         [gt.get("activity_gt")] if gt.get("activity_gt") else []),
        ("objects", fields.get("objects") or [], gt.get("objects_gt") or []),
        ("environments", fields.get("environments") or [], gt.get("environments_gt") or []),
    ]

    out = {}
    for dim, qwen_items, golden_items in dims:
        golden_items = [g for g in golden_items if g]
        hit, miss = [], []
        for g in golden_items:
            (hit if any(_match(q, g) for q in qwen_items) else miss).append(g)
        extra = [q for q in qwen_items if not any(_match(q, g) for g in golden_items)]
        out[dim] = {
            "golden": golden_items,
            "qwen": qwen_items,
            "hit": hit,
            "miss": miss,
            "extra": extra,
            "recall": (len(hit) / len(golden_items)) if golden_items else None,
        }
    return out


if __name__ == "__main__":
    import sys
    from interactive_runner import _load_jsonl, SHORT_FILE

    moments = _load_jsonl(SHORT_FILE)
    p = QwenPerceiver()
    print("Qwen 服务:", p.health())
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for m in moments[:n]:
        r = p.perceive(m, force=True)
        print("\n" + "=" * 60)
        print(m["id"], "| source:", r["source"], "| 耗时:", r.get("elapsed_ms"), "ms")
        if r.get("fields"):
            print("Qwen 产出:", json.dumps(r["fields"], ensure_ascii=False))
            print("Golden   :", json.dumps(m.get("ground_truth", {}), ensure_ascii=False))
            ev = golden_eval(r["fields"], m.get("ground_truth", {}))
            for dim, e in ev.items():
                if e["golden"]:
                    print(f"  {dim}: 命中 {len(e['hit'])}/{len(e['golden'])} "
                          f"miss={e['miss']} extra={e['extra']}")
