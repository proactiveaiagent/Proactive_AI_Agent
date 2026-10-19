"""
memory3.py — Hierarchical 7-Layer Memory System
================================================
Layer 1 : Current moment  — Part1 scene/action + Part2 needs + Part3 solutions + Part4 feedback
Layer 2 : Same-env moments — same external scene from earlier in the session
Layer 3 : All moments today
Layer 4 : Compressed recent — user's current tasks / periodic life trajectory (text summary)
Layer 5 : Compressed older  — key moments & events that shaped the user / long-term stable patterns (text)
Layer 6 : Compressed older  — user profile & personal preferences / basic background info (text)
Layer 7 : Classified archive — tagged by: time-node, activity-event, person, location

DB Operations
  add       — store Part1-4 data when new data arrives
  update    — Part4 corrects Part1-3 data; habits changed → update layers 4-5-6
  query     — Part2 queries relevant data for need analysis
  retrieve  — pull related data for Part2 analysis; pull Layer-7 data for layers 4-5-6
  compress  — summarise raw layers 4-5-6-7 into text + few images  (triggered after need analysis)
  sort      — classify same-type data under shared tags in layer 7  (triggered at compress time)
  combine   — canonicalize layer 7 index aliases while preserving source moments
  delete    — one-off / accidental / erroneous entries; post-compress duplicates
  highlight — data explicitly confirmed or marked important by the user
"""

import json
import math
import os
import re
import copy
from difflib import SequenceMatcher
from collections import Counter, defaultdict
import time
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat()


def _today_str() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# User profile helpers（规格书 Part 2 骨架 + 04 文档 2.4 合并语义）
# ---------------------------------------------------------------------------

# 画像字段白名单（与 profile_extractor.STABLE_FIELDS 对齐）
PROFILE_FIELDS = ("demographics", "preferences", "frequent_locations", "behavior_patterns",
                  "personality", "goals", "decisions", "motivations")
# dict 型字段（值是 {子键: ...}，区别于 list 型 frequent_locations）
# dict 型新字段：personality（性格特点，子键分类）/ decisions（选择决策，子键分类）
# list 型：frequent_locations / goals（规划目标）/ motivations（动机）
PROFILE_DICT_FIELDS = {"demographics", "preferences", "behavior_patterns",
                       "personality", "decisions"}

# layer7 索引维度（规范「存储的外在内容」6 类）
#   从近到远排序：人物 / 空间位置 / 时间
INDEX_NEAR_FAR = ("people", "locations", "time_nodes")
#   从高频到低频排序：动作事件 / 环境场景 / 物品
INDEX_FREQUENCY = ("activity_events", "environments", "objects")
LAYER7_INDICES = INDEX_NEAR_FAR + INDEX_FREQUENCY

# 低置信度阈值（对齐 04 文档 2.5：confidence < 0.3 不入库）
MIN_CONFIDENCE = 0.3
# 列表去重的相似度阈值（对齐 04 文档 2.4.4：初始 0.85）
MERGE_SIM_THRESHOLD = 0.85


def _bigram_jaccard(a, b) -> float:
    """bigram Jaccard 相似度，用于列表去重与单值冲突判断。"""
    a, b = str(a).lower(), str(b).lower()
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    def _bigrams(s):
        return {s[i:i + 2] for i in range(len(s) - 1)}
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _wrap_attr(value, source: str = "legacy", timestamp: Optional[str] = None) -> Dict:
    """把裸值包装成 AttrValue（用于旧数据迁移）。"""
    return {
        "value": value,
        "confidence": 1.0,
        "rank_score": 1.0,
        "source": source,
        "timestamp": timestamp or _now_iso(),
        "last_seen": timestamp or _now_iso(),
        "observations": 1,
    }


PLACEHOLDER_TAGS = {
    "canonical_name", "canonical_location", "canonical_tag",
    "event_tag", "moment_id_1", "moment_id_2",
    "other", "others", "other people", "unknown person", "unidentified person",
    "someone", "somebody", "group", "crowd", "camera wearer",
    "two others", "three others", "several others", "a few others"
}

NON_PERSON_PATTERNS = (
    "no identifiable", "not identifiable", "none visible", "no person",
    "no individual", "no one", "nobody", "server in a", "travelers",
    "customers are", "patrons and staff", "eating near", "hands over"
)


def _is_valid_person_tag(tag: str) -> bool:
    """判断是否为有效的人名/角色标识，过滤非人名描述、否定句与占位符（修复 G7 索引污染）。"""
    if not tag or not isinstance(tag, str):
        return False
    cleaned = tag.strip().strip(".，。, ")
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered in {"none", "n/a", "na", "null", "nil", "无", "暂无", "无人", "没有", "无法识别", "不详", "独自一人", "独自", "not applicable"}:
        return False
    if lowered in PLACEHOLDER_TAGS:
        return False
    if re.fullmatch(r"(?:(?:unidentified|unknown)\s+)?person\s+\d+", lowered):
        return False
    # Descriptive visual aliases are longer than conventional names, but remain
    # useful searchable entity keys when they are concise and non-sentential.
    if len(cleaned) > 180:
        return False
    for pat in NON_PERSON_PATTERNS:
        if pat in lowered:
            return False
    if not lowered.startswith("person "):
        for verb_clause in (" is ", " are ", " was ", " were ", " visible", " seated", " standing", " walking"):
            if verb_clause in lowered:
                return False
    return True


def _is_valid_index_tag(index_key: str, tag: str) -> bool:
    """验证 layer7 索引键是否合法。"""
    if not tag or not isinstance(tag, str):
        return False
    cleaned = tag.strip().strip(".，。, ")
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered in PLACEHOLDER_TAGS:
        return False
    if index_key == "people":
        return _is_valid_person_tag(cleaned)
    if index_key == "locations":
        if lowered in {"none", "n/a", "unknown", "无", "暂无", "canonical_location"}:
            return False
        if len(cleaned) > 180:
            return False
    if index_key == "activity_events":
        if lowered in {"none", "n/a", "unknown", "无", "暂无", "event_tag"}:
            return False
        if len(cleaned) > 220:
            return False
    if index_key == "environments":
        if lowered in {"none", "n/a", "unknown", "无", "暂无"} or len(cleaned) > 260:
            return False
    if index_key == "objects":
        if lowered in {"none", "n/a", "unknown", "无", "暂无"}:
            return False
        if len(cleaned) > 120 or re.match(
                r"^(?:from|to|toward|towards|into|out of|near|at|inside|outside)\s+", lowered):
            return False
    if index_key == "time_nodes" and len(cleaned) > 220:
        return False
    return True


def _clean_layer7_indices(layer7: Dict) -> Dict[str, List[str]]:
    """清洗 layer7 索引：移除占位符、描述性长句、无意义负向词（G7 缺陷修复）。"""
    removed = {k: [] for k in LAYER7_INDICES}
    if not isinstance(layer7, dict):
        return removed
    for k in LAYER7_INDICES:
        idx = layer7.get(k, {})
        if not isinstance(idx, dict):
            continue
        bad_keys = [tag for tag in list(idx.keys()) if not _is_valid_index_tag(k, tag)]
        for b in bad_keys:
            del idx[b]
            removed[k].append(b)
    return removed


def _upgrade_legacy_person_labels(moment: Dict) -> bool:
    """Replace legacy numbered person placeholders with visible-description aliases."""
    if not isinstance(moment, dict):
        return False
    observations = moment.get("person_observations") or []
    mapping = {}
    for row in observations:
        if not isinstance(row, dict):
            continue
        original = " ".join(str(row.get("entity") or "").split())
        if not re.fullmatch(r"(?:(?:unidentified|unknown)\s+)?person\s+\d+", original, re.I):
            continue
        appearance = " ".join(str(row.get("appearance") or "").split()).strip(" .;,:-")
        face = " ".join(str(row.get("face_features") or "").split()).strip(" .;,:-")
        alias_appearance = re.split(
            r",\s*(?:standing|sitting|seated|walking|moving|turning|looking)\b",
            appearance, maxsplit=1, flags=re.I)[0].strip(" .;,:-")
        descriptor = alias_appearance or appearance or face
        if not descriptor:
            continue
        alias = descriptor if re.match(r"^person\b", descriptor, re.I) else (
            "person " + descriptor if re.match(
                r"^(?:wearing|with|in|has|holding|standing|sitting|seated)\b", descriptor, re.I)
            else "person with " + descriptor)
        alias = alias[:180].rstrip(" .;,:-")
        if alias and alias.casefold() != original.casefold():
            mapping[original] = alias
            row["entity"] = alias
    if not mapping:
        return False

    def replace(value):
        if isinstance(value, str):
            result = value
            for original, alias in sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True):
                result = re.sub(r"(?<![\w])" + re.escape(original) + r"(?![\w])",
                                lambda _match, replacement=alias: replacement,
                                result, flags=re.I)
            return result
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    for field in ("scene", "user_action", "interaction_details", "location", "activity",
                  "environments", "objects", "people", "person_observations", "temporal_facts",
                  "multimodal"):
        if field in moment:
            moment[field] = replace(moment[field])
    return True


def _moment_time_tags(moment: Dict) -> list:
    """Return only source-grounded time keys; never index ingest time as event time."""
    source = moment.get("source_time") or {}
    tags = []
    for key in ("absolute_date", "day_label", "time_of_day"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            value = value.strip()[:128]
            if value not in tags:
                tags.append(value)
    day = str(source.get("day_label") or source.get("absolute_date") or "").strip()
    clock = str(source.get("time_of_day") or "").strip()
    if day and clock:
        tags.append(f"{day} {clock}"[:220])
    basis = str(source.get("basis") or "").casefold()
    segment = str(source.get("source_segment_id") or
                  (moment.get("source_provenance") or {}).get("source_segment_id") or "").strip()
    if not tags and segment and basis in {"video_relative", "dataset_relative"}:
        try:
            start = float(source.get("clip_start_s", 0) or 0)
            end = float(source.get("clip_end_s", start) or start)
            if math.isfinite(start) and math.isfinite(end) and start >= 0 and end >= start:
                tags.append(f"{segment} · 视频相对 {start:.2f}–{end:.2f}s"[:220])
        except (TypeError, ValueError):
            pass
    if not tags and not source:
        timestamp = str(moment.get("timestamp") or "")
        try:
            tags.append(datetime.fromisoformat(timestamp).date().isoformat())
        except (TypeError, ValueError):
            pass
    return list(dict.fromkeys(tag for tag in tags if _is_valid_index_tag("time_nodes", tag)))


def _moment_index_tags(moment: Dict) -> Dict[str, list]:
    """Derive retrieval keys from the stored source-linked moment, not only caller labels."""
    if not isinstance(moment, dict):
        return {key: [] for key in LAYER7_INDICES}
    tags = {key: [] for key in LAYER7_INDICES}

    def add(index_key, value):
        if isinstance(value, str):
            value = value.strip()
        if value and _is_valid_index_tag(index_key, value) and value not in tags[index_key]:
            tags[index_key].append(value)

    for person in moment.get("people") or []:
        add("people", person)
    for row in moment.get("person_observations") or []:
        if isinstance(row, dict):
            add("people", row.get("entity"))
    add("locations", moment.get("location"))
    add("activity_events", moment.get("activity"))
    for value in moment.get("environments") or []:
        add("environments", value)
    for value in moment.get("objects") or []:
        add("objects", value)
    for row in moment.get("object_observations") or []:
        if isinstance(row, dict):
            add("objects", row.get("object"))

    for fact in moment.get("temporal_facts") or []:
        if not isinstance(fact, dict):
            continue
        subject = fact.get("subject")
        kind = str(fact.get("subject_type") or "").casefold()
        if kind == "person":
            add("people", subject)
        elif kind == "object":
            add("objects", subject)
        elif kind == "place":
            add("locations", subject)
        elif kind == "environment":
            add("environments", subject)
        elif kind == "event":
            add("activity_events", subject)
        if kind == "place":
            add("locations", fact.get("from_location"))
            add("locations", fact.get("to_location"))
        if kind == "person":
            add("people", fact.get("actor"))
            for affected in fact.get("affected_people") or []:
                add("people", affected)
    tags["time_nodes"] = _moment_time_tags(moment)
    return tags


def _build_entity_visuals(moment: Dict) -> dict:
    """Link each memory entity to existing image assets without duplicating image bytes."""
    media = moment.get("multimodal") or {}
    if not isinstance(media, dict):
        return {"schema": 1, "assets": [], "items": []}
    assets, asset_index, items = [], {}, []
    provided_entity_visuals = media.get("entity_visuals") or {}
    provided_assets = (provided_entity_visuals.get("assets") or []
                       if isinstance(provided_entity_visuals, dict) else [])
    provided_context_items = [
        item for item in (provided_entity_visuals.get("items") or [])
        if isinstance(item, dict)
        and item.get("entity_type") in {"scene", "location", "environment", "activity", "event"}
    ] if isinstance(provided_entity_visuals, dict) else []
    source_id = str((moment.get("source_provenance") or {}).get("source_segment_id") or
                    media.get("source_id") or "")

    def asset_id(ref, role=None):
        if not isinstance(ref, dict) or not isinstance(ref.get("asset"), str) or not ref["asset"].startswith("assets/"):
            return None
        asset = ref["asset"].replace(chr(92), "/")
        key = (asset, role or ref.get("visual_role") or "")
        if key not in asset_index:
            row = {"asset": asset, "visual_role": key[1] or "source_frame"}
            for name in ("bytes", "mime", "dhash", "at_s", "source_id", "quality_tier", "crop_bbox"):
                value = ref.get(name)
                if value not in (None, ""):
                    row[name] = copy.deepcopy(value)
            if not row.get("source_id") and source_id:
                row["source_id"] = source_id
            if ref.get("source_asset"):
                row["source_asset"] = str(ref["source_asset"]).replace(chr(92), "/")
            asset_index[key] = len(assets)
            assets.append(row)
        return asset_index[key]

    def add_item(kind, entity, description, refs, fallback_refs=None, status="linked"):
        entity = " ".join(str(entity or "").split()).strip()
        if not entity:
            return
        indices = []
        for ref in refs or []:
            index = asset_id(ref)
            if index is not None and index not in indices:
                indices.append(index)
        if not indices:
            for ref in fallback_refs or []:
                index = asset_id(ref, "scene_context_fallback")
                if index is not None and index not in indices:
                    indices.append(index)
        items.append({"entity_type": kind, "entity": entity[:500],
                      "description": " ".join(str(description or "").split())[:600],
                      "status": status if indices else "no_visual_evidence",
                      "asset_indices": indices[:8]})

    scene_frames = []
    for ref in media.get("visual_frames") or []:
        if isinstance(ref, dict):
            scene_frames.append({**ref, "visual_role": "scene_context"})
    if not scene_frames and isinstance(media.get("visual"), dict):
        scene_frames.append({**media["visual"], "visual_role": "scene_context"})
    scene_frames = scene_frames[:4]

    # Person/object crops are the preferred evidence. Keep a full scene frame as
    # a fallback so an entity is still inspectable when a crop could not be made.
    person_visuals = media.get("person_visuals") or {}
    person_items = person_visuals.get("items") or [] if isinstance(person_visuals, dict) else []
    people_by_name = {str(row.get("person") or "").casefold(): row
                      for row in person_items if isinstance(row, dict)}
    for row in moment.get("person_observations") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("entity") or "").strip()
        visual = people_by_name.get(name.casefold()) or {}
        refs = visual.get("frame_refs") or []
        add_item("person", name, row.get("appearance") or visual.get("appearance"),
                 refs, scene_frames, "grounded" if refs else "context_fallback")

    object_visuals = media.get("object_visuals") or {}
    object_items = object_visuals.get("items") or [] if isinstance(object_visuals, dict) else []
    objects_by_name = {str(row.get("object") or "").casefold(): row
                        for row in object_items if isinstance(row, dict)}
    observations = moment.get("object_observations") or []
    if not observations:
        observations = [{"object": row.get("object"), "appearance": row.get("appearance")}
                        for row in object_items if isinstance(row, dict)]
    for row in observations:
        if not isinstance(row, dict):
            continue
        name = str(row.get("object") or "").strip()
        visual = objects_by_name.get(name.casefold()) or {}
        refs = visual.get("frame_refs") or []
        add_item("object", name, row.get("appearance") or visual.get("appearance"),
                 refs, scene_frames, "grounded" if refs else "context_fallback")

    if not provided_context_items:
        add_item("scene", moment.get("scene"), "", scene_frames, status="context")
    if moment.get("location") and not provided_context_items:
        add_item("location", moment.get("location"), "", scene_frames[1:2] or scene_frames,
                 status="context")
    if not provided_context_items:
        for environment in moment.get("environments") or []:
            add_item("environment", environment, "", scene_frames, status="context")

    # Activity-level images are selected from the observed event times where possible.
    facts = [row for row in moment.get("temporal_facts") or [] if isinstance(row, dict)]
    activity_refs = []
    for fact in facts:
        if fact.get("modality") not in (None, "visual"):
            continue
        ref = fact.get("evidence_ref") or {}
        try:
            at = float(ref.get("frame_time_s", fact.get("frame_time_s")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(at):
            continue
        if scene_frames:
            best = min(scene_frames, key=lambda frame: abs(float(frame.get("at_s") or 0) - at))
            if best not in activity_refs:
                activity_refs.append(best)
    if not activity_refs:
        activity_refs = scene_frames
    if moment.get("activity") and not provided_context_items:
        add_item("activity", moment.get("activity"), "", activity_refs[:4], status="context")
    for fact in ([] if provided_context_items else facts[:16]):
        kind = str(fact.get("subject_type") or "").casefold()
        if kind not in {"event", "person", "object", "place", "environment"}:
            continue
        label = " · ".join(str(fact.get(key) or "").strip()
                            for key in ("subject", "predicate", "value") if fact.get(key))
        ref = fact.get("evidence_ref") or {}
        refs = []
        try:
            at = float(ref.get("frame_time_s", fact.get("frame_time_s")))
        except (TypeError, ValueError):
            at = None
        if at is not None and scene_frames:
            refs = [min(scene_frames, key=lambda frame: abs(float(frame.get("at_s") or 0) - at))]
        add_item("event", label, str(fact.get("time_scope") or ""), refs,
                 scene_frames, "event_frame" if refs else "context_fallback")
    for item in provided_context_items:
        entity_type = str(item.get("entity_type") or "")
        entity = " ".join(str(item.get("entity") or "").split()).strip()
        if not entity:
            continue
        indices = []
        for index in item.get("asset_indices") or []:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(provided_assets):
                continue
            asset_index_value = asset_id(provided_assets[index])
            if asset_index_value is not None and asset_index_value not in indices:
                indices.append(asset_index_value)
        items.append({
            "entity_type": entity_type, "entity": entity[:500],
            "description": " ".join(str(item.get("description") or "").split())[:600],
            "status": str(item.get("status") or ("grounded" if indices else "no_visual_evidence")),
            "asset_indices": indices[:8],
        })
    return {"schema": 1, "assets": assets, "items": items[:64]}


def _migrate_profile(profile: Dict) -> Dict:
    """把旧 layer6.profile（name/basic_info/preferences/habits）迁移到规格书骨架字段。

    09-03 已验证：缺字段不报错；此处做尽力迁移，空值丢弃。
    """
    new = {
        "demographics": {},
        "preferences": {},
        "frequent_locations": [],
        "behavior_patterns": {},
        "personality": {},
        "goals": [],
        "decisions": {},
        "motivations": [],
    }
    if not isinstance(profile, dict):
        return new

    # 已是新结构：补齐缺失字段后直接返回
    if any(k in profile for k in ("demographics", "frequent_locations", "behavior_patterns")):
        for k, v in new.items():
            if k in profile:
                new[k] = profile[k]
        return new

    # 旧结构：尽力迁移
    if profile.get("name"):
        new["demographics"]["name"] = _wrap_attr(profile["name"])
    if isinstance(profile.get("basic_info"), dict):
        for k, v in profile["basic_info"].items():
            if v:
                new["demographics"][k] = _wrap_attr(v)
    if isinstance(profile.get("preferences"), dict):
        for k, v in profile["preferences"].items():
            if v:
                new["preferences"][k] = [_wrap_attr(x) for x in (v if isinstance(v, list) else [v]) if x]
    if isinstance(profile.get("habits"), dict):
        new["behavior_patterns"]["with_agents"] = [
            _wrap_attr(v) for v in profile["habits"].values() if v
        ]
    return new


def _finalize_attr(attr: Dict, source: str, timestamp: str) -> Optional[Dict]:
    """规范化单个 AttrValue：补 source/timestamp/last_seen，observations 归 1。空值返回 None。"""
    if not isinstance(attr, dict):
        return None
    value = attr.get("value")
    if value is None or value == "":
        return None
    try:
        conf = float(attr.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    try:
        rank_score = float(attr.get("rank_score", attr.get("importance", attr.get("strength", conf))))
    except (TypeError, ValueError):
        rank_score = conf
    ts = timestamp or _now_iso()
    out = {
        "value": value,
        "confidence": max(0.0, min(1.0, conf)),
        "rank_score": max(0.0, min(1.0, rank_score)),
        "source": attr.get("source") or source or "llm_inference",
        "evidence": str(attr.get("evidence", ""))[:200],
        "timestamp": ts,
        "last_seen": ts,
        "observations": 1,
    }
    # 三类衰减曲线（设计说明书 §5）：LLM 标注优先；带 expires_at 强制 deadline（语义校正）
    if attr.get("expires_at"):
        out["decay_type"] = DECAY_DEADLINE
        out["expires_at"] = str(attr["expires_at"])
    elif attr.get("decay_type") in (DECAY_STABLE, DECAY_DECAYING, DECAY_DEADLINE):
        out["decay_type"] = attr["decay_type"]
        if "decay_rate" in attr:
            try:
                out["decay_rate"] = float(attr["decay_rate"])
            except (TypeError, ValueError):
                pass
    # 未标注 decay_type：不设，由 _assign_default_decay 按字段路径兜底
    return out


def _corroborate_attr(old: Dict, new: Dict) -> Dict:
    """同值再次出现时累加证据，并保留独立的排序强度。"""
    old_count = max(1, int(old.get("observations", 1) or 1))
    new_count = max(1, int(new.get("observations", 1) or 1))
    old_conf = float(old.get("confidence", 0))
    new_conf = float(new.get("confidence", 0))
    old_rank = float(old.get("rank_score", old_conf))
    new_rank = float(new.get("rank_score", new_conf))
    old["observations"] = old_count + new_count
    old["confidence"] = 1 - (1 - old_conf) * (1 - new_conf)
    old["rank_score"] = (old_rank * old_count + new_rank * new_count) / old["observations"]
    old["last_seen"] = max(old.get("last_seen", ""), new.get("last_seen", ""))
    if new_conf > old_conf:
        old["source"] = new["source"]
    return old


def _resolve_conflict(old: Dict, new: Dict) -> Dict:
    """单值冲突裁决：观测次数 > 最近观察 > 置信度；平局保留已有值。

    败者追加到 winner.history，并写明裁决依据、双方来源和值与裁决时间，
    让画像当前值可追溯到被替代的观察。
    """
    old_count = max(1, int(old.get("observations", 1) or 1))
    new_count = max(1, int(new.get("observations", 1) or 1))
    old_seen = str(old.get("last_seen", "") or "")
    new_seen = str(new.get("last_seen", "") or "")
    old_conf = float(old.get("confidence", 0) or 0)
    new_conf = float(new.get("confidence", 0) or 0)

    if new_count != old_count:
        winner, loser = (new, old) if new_count > old_count else (old, new)
        basis = "observations"
    elif new_seen != old_seen:
        winner, loser = (new, old) if new_seen > old_seen else (old, new)
        basis = "recency"
    elif new_conf != old_conf:
        winner, loser = (new, old) if new_conf > old_conf else (old, new)
        basis = "confidence"
    else:
        winner, loser = old, new
        basis = "existing_value_tiebreak"

    # Carry forward prior history whichever value wins. Avoid nesting a whole
    # history tree inside one loser entry as conflicts accumulate.
    history = []
    seen = set()
    for candidate in (old, new):
        entries = candidate.get("history")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            signature = (
                str(entry.get("value", "")),
                str((entry.get("resolution") or {}).get("resolved_at", "")),
                str(entry.get("source", "")),
            )
            if signature not in seen:
                seen.add(signature)
                history.append(entry)

    loser_entry = {key: value for key, value in loser.items() if key != "history"}
    loser_entry["resolution"] = {
        "winner_value": winner.get("value"),
        "winner_source": winner.get("source", ""),
        "basis": basis,
        "resolved_at": _now_iso(),
    }
    history.append(loser_entry)
    winner["history"] = history
    return winner

def _replace_stale_attr(old: Dict, new: Dict, prior_effective_confidence: Optional[float] = None) -> Dict:
    """Promote a current observation over a stale scalar while keeping an audit trail."""
    winner = copy.deepcopy(new)
    history = []
    seen = set()
    for entry in old.get("history", []) if isinstance(old.get("history"), list) else []:
        if not isinstance(entry, dict):
            continue
        signature = (str(entry.get("value", "")),
                     str((entry.get("resolution") or {}).get("resolved_at", "")),
                     str(entry.get("source", "")))
        if signature not in seen:
            seen.add(signature)
            history.append(copy.deepcopy(entry))
    loser = {key: copy.deepcopy(value) for key, value in old.items() if key != "history"}
    loser["resolution"] = {
        "winner_value": winner.get("value"),
        "winner_source": winner.get("source", ""),
        "basis": "stale_replaced_by_current_observation",
        "prior_effective_confidence": prior_effective_confidence if prior_effective_confidence is not None else old.get("effective_confidence"),
        "resolved_at": _now_iso(),
    }
    history.append(loser)
    winner["history"] = history
    return winner


def _merge_attr_list(old_list: List, new_list: List, threshold: float = MERGE_SIM_THRESHOLD, now: Optional[str] = None) -> List:
    """追加去重；过期条目不再吸收新观察，但仍留存以供追溯。"""
    for na in new_list:
        hit = next(
            (o for o in old_list
             if isinstance(o, dict) and "value" in o
             and effective_confidence(o, now=now) >= STALE_THRESHOLD
             and _bigram_jaccard(o["value"], na["value"]) >= threshold),
            None,
        )
        if hit:
            _corroborate_attr(hit, na)
        else:
            old_list.append(na)
    return old_list


def _filter_low_confidence(node, min_conf: float = MIN_CONFIDENCE):
    """递归剔除 confidence < min_conf 的 AttrValue；返回过滤后节点。"""
    if isinstance(node, list):
        return [x for x in node
                if isinstance(x, dict) and x.get("confidence", 0) >= min_conf]
    if isinstance(node, dict):
        if "value" in node:
            return node if node.get("confidence", 0) >= min_conf else None
        out = {}
        for k, v in node.items():
            fv = _filter_low_confidence(v, min_conf)
            if fv not in (None, [], {}):
                out[k] = fv
        return out
    return None


# ---------------------------------------------------------------------------
# 检索分词（09-12 定稿：统一英文分词——非英文语种先翻译为英文再分词）
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """中英混合分词：英文/数字按单词边界切分，中文按单字切分。零 LLM 依赖。

    设计依据（设计说明书 §6.2，09-20 修订）：统一分词不再依赖「先翻译为英文」，
    而是让 tokenize 同时处理中英文——中文输出单字 token（命中中文原文），
    英文输出单词 token（命中写入时预翻译的 normalized）。检索同步链路零 LLM。
    """
    if not text:
        return []
    tokens, buf = [], ""
    for ch in text.lower():
        if ('a' <= ch <= 'z') or ('0' <= ch <= '9') or ch == "'":
            buf += ch
        elif '\u4e00' <= ch <= '\u9fff':   # 中文字符 → 单字 token
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    return tokens



_SEARCH_EN_STOPWORDS = frozenset({
    "a", "about", "after", "all", "also", "am", "an", "and", "are", "as", "at",
    "be", "because", "been", "being", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "more",
    "most", "my", "no", "nor", "of", "off", "on", "once", "only", "or", "our",
    "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "now", "currently"
})
_SEARCH_ZH_STOPWORDS = frozenset("的了在是有我你他她它和与及就也都很过把被给让而或又吗呢吧对从到这那一个些谁什么如何哪怎么现在刚才")

def _retrieval_tokens(text: str) -> list:
    """Return content-bearing terms for lexical retrieval; preserve Chinese character tokens."""
    return [token for token in tokenize(text)
            if token not in _SEARCH_EN_STOPWORDS
            and token not in _SEARCH_ZH_STOPWORDS
            and (len(token) > 1 or '\u4e00' <= token <= '\u9fff')]

def _memory_search_field_weights():
    return {
        "scene": 1.0,
        "user_action": 1.0,
        "interaction_details": 1.2,
        "person_observations": 1.1,
        "temporal_facts": 1.3,
        "object_observations": 1.2,
        "source_time": 0.8,
        "activity": 1.0,
        "people": 1.0,
        "location": 1.0,
        "objects": 1.0,
        "environments": 1.0,
        "activity_events": 1.0,
        "time_occasion": 0.7,
        "extra_notes": 0.6,
        "normalized": 0.5,
        "source_evidence": float(os.environ.get("MEMORY_SOURCE_EVIDENCE_WEIGHT", "0.25")),
        "multimodal_entities": 0.8,
        "merged_source_text": 0.6,
    }


_MEMORY_SEARCH_METADATA = {
    "id", "timestamp", "created_at", "updated_at", "last_used",
    "access_count", "stale", "decay_type", "decay_rate", "expires_at",
    "confidence", "highlighted", "feedback", "source_ids", "layer",
    "source_provenance", "canonicalization_history", "entity_graph",
    "needs", "solutions", "multimodal", "merged_into", "merged_sources",
    "merge_history", "merge_event", "source_refresh_history",
    "_merged_sources_records",
    "_merged_search_text", "_group_highlighted",
}


def _as_memory_search_text(value):
    if isinstance(value, dict):
        return " ".join(_as_memory_search_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_as_memory_search_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _build_memory_search_index(documents: list) -> dict:
    """Build field-aware BM25 statistics once; callers own cache invalidation."""
    active_documents = [doc for doc in documents
                        if isinstance(doc, dict) and not doc.get("stale")
                        and not doc.get("merged_into")]
    field_names = set(_memory_search_field_weights()) | {"other"}
    field_counts = []
    field_lengths = []
    document_frequency = Counter()

    for document in active_documents:
        values = {}
        for key, value in document.items():
            if key == "_merged_search_text":
                if value:
                    values["merged_source_text"] = _as_memory_search_text(value)
                continue
            if key in _MEMORY_SEARCH_METADATA:
                continue
            if key == "multimodal":
                entities = value.get("entities") if isinstance(value, dict) else None
                if entities:
                    values["multimodal_entities"] = _as_memory_search_text(entities)
                continue
            field = key if key in field_names else "other"
            values[field] = (values.get(field, "") + " " +
                             _as_memory_search_text(value)).strip()

        counts = {
            field: Counter(_retrieval_tokens(values.get(field, "").lower()))
            for field in field_names
        }
        lengths = {field: sum(counter.values()) for field, counter in counts.items()}
        field_counts.append(counts)
        field_lengths.append(lengths)
        for field, counter in counts.items():
            for term in counter:
                document_frequency[(field, term)] += 1

    averages = {}
    for field in field_names:
        nonempty = [lengths[field] for lengths in field_lengths if lengths[field] > 0]
        averages[field] = sum(nonempty) / len(nonempty) if nonempty else 1.0

    postings = {}
    for doc_index, counts in enumerate(field_counts):
        for field, counter in counts.items():
            for term, tf in counter.items():
                postings.setdefault(term, {}).setdefault(field, []).append((doc_index, tf))

    return {
        "documents": active_documents,
        "field_lengths": field_lengths,
        "document_frequency": document_frequency,
        "average_lengths": averages,
        "postings": postings,
        "live_count": len(active_documents),
    }


def _rank_memory_search_index(query_text: str, index_data: dict) -> list:
    """Score a prebuilt BM25 index while preserving the established tie order."""
    documents = index_data["documents"]
    if not documents:
        return []
    query_terms = list(dict.fromkeys(_retrieval_tokens(str(query_text or "")[:1200])))
    if not query_terms:
        return []

    field_weights = _memory_search_field_weights()
    df = index_data["document_frequency"]
    averages = index_data["average_lengths"]
    lengths = index_data["field_lengths"]
    scores = {}
    k1, b = 1.2, 0.75

    for term in query_terms:
        for field, rows in index_data["postings"].get(term, {}).items():
            idf = math.log(1.0 + (index_data["live_count"] - df[(field, term)] + 0.5)
                           / (df[(field, term)] + 0.5))
            average_length = averages[field]
            weight = field_weights.get(field, 0.75)
            for doc_index, tf in rows:
                length = lengths[doc_index][field]
                norm_tf = (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * length / average_length)
                )
                scores[doc_index] = scores.get(doc_index, 0.0) + weight * idf * norm_tf

    ranked = [(score, doc_index, documents[doc_index])
              for doc_index, score in scores.items() if score > 0]
    ranked.sort(key=lambda item: (
        -item[0], not bool(item[2].get("highlighted") or
                           item[2].get("_group_highlighted")), item[1]
    ))
    return [(score, document) for score, _, document in ranked]


def _rank_memory_docs(query_text: str, documents: list) -> list:
    """Compatibility entry point for one-off BM25 queries."""
    return _rank_memory_search_index(query_text, _build_memory_search_index(documents))

def _contains_cjk(text: str) -> bool:
    """检测是否含中日韩字符（需先翻译为英文才能分词）。"""
    if not text:
        return False
    return any(
        '\u4e00' <= ch <= '\u9fff'      # CJK 统一汉字
        or '\u3040' <= ch <= '\u30ff'   # 日文假名
        or '\uac00' <= ch <= '\ud7af'   # 韩文
        for ch in text
    )


# ---------------------------------------------------------------------------
# 时间衰减与陈旧淘汰（设计说明书 §5：三类衰减曲线，09-16 落地）
# ---------------------------------------------------------------------------

# 兼容旧数据：无 decay_type 时的默认渐变衰减率（0.5%/天，半衰期约 139 天）
STABLE_DAILY_DECAY = 0.005
# 陈旧淘汰阈值：有效置信度 < 0.3 标记 stale（不参与检索，不物理删除）
STALE_THRESHOLD = 0.3

# 三类衰减曲线类型（设计说明书 §5.2）
DECAY_STABLE = "stable"       # 身份偏好类：较长有效期或永久有效
DECAY_DECAYING = "decaying"   # 经历状态类：随时间流逝逐渐衰减
DECAY_DEADLINE = "deadline"   # 优惠券类：过期前完全有效，过期瞬间失效

# 各曲线衰减率（每天）
DECAY_RATE_PERMANENT = 0.0      # 硬身份（性别/姓名/受教育）：永久，不衰减
DECAY_RATE_STABLE = 0.001       # 偏好/行为/习惯/地点：长有效（半衰期约 1.9 年）
DECAY_RATE_DECAYING = 0.005     # 经历状态：渐变（半衰期约 139 天）

# 硬身份字段（永久有效，decay=0）——设计说明书 §3.3 字段字典
PERMANENT_FIELDS = {"name", "gender", "identity", "education"}
# 渐变字段（经历状态类，随时间的推移自然变化）——设计说明书 §3.3
DECAYING_FIELDS = {"occupation"}


def _now_before(expires_at: str, now: Optional[str] = None) -> bool:
    """判断 now 是否早于 expires_at（过期前）。解析失败保守视为未过期。"""
    try:
        exp = datetime.fromisoformat(expires_at)
        now_dt = datetime.fromisoformat(now) if now else datetime.now()
        return now_dt < exp
    except (TypeError, ValueError):
        return True


def effective_confidence(attr: Dict, daily_decay: float = STABLE_DAILY_DECAY,
                         now: Optional[str] = None) -> float:
    """按 decay_type 计算有效置信度（设计说明书 §5 三类衰减曲线）。

    - stable：不衰减 / 极低衰减（decay_rate 由字段决定，默认 0.001）
    - decaying：随时间渐变衰减（decay_rate 默认 0.005）
    - deadline：过期前 eff=confidence，过期后 eff=0（阶跃）
    - 旧数据（无 decay_type）：按 daily_decay 渐变衰减（兼容）
    """
    conf = float(attr.get("confidence", 0))

    dt = attr.get("decay_type")
    if dt == DECAY_DEADLINE:
        if attr.get("expires_at"):
            return conf if _now_before(attr["expires_at"], now) else 0.0
        # deadline 缺失 expires_at → 按渐变兜底
        daily = float(attr.get("decay_rate", DECAY_RATE_DECAYING))
    elif dt == DECAY_STABLE:
        daily = float(attr.get("decay_rate", DECAY_RATE_STABLE))
    elif dt == DECAY_DECAYING:
        daily = float(attr.get("decay_rate", DECAY_RATE_DECAYING))
    else:
        daily = daily_decay  # 旧数据兼容

    last_seen = attr.get("last_seen")
    if not last_seen:
        return conf
    try:
        last = datetime.fromisoformat(last_seen)
        now_dt = datetime.fromisoformat(now) if now else datetime.now()
        days = max(0, (now_dt - last).days)
    except (TypeError, ValueError):
        days = 0
    return max(conf * (1 - daily) ** days, 0.0)


def _default_decay_for_path(path: tuple) -> tuple:
    """字段路径 → 默认 (decay_type, decay_rate)（设计说明书 §5.3 兜底）。

    - demographics 硬身份字段 → stable 永久（0.0）
    - demographics.occupation（经历状态）→ decaying（0.005）
    - 其余（preferences / frequent_locations / behavior_patterns 等）→ stable 长有效（0.001）
    """
    if not path:
        return (DECAY_STABLE, DECAY_RATE_STABLE)
    field = path[0]
    sub = path[1] if len(path) > 1 else ""
    if field == "demographics":
        if sub in PERMANENT_FIELDS:
            return (DECAY_STABLE, DECAY_RATE_PERMANENT)
        if sub in DECAYING_FIELDS:
            return (DECAY_DECAYING, DECAY_RATE_DECAYING)
    return (DECAY_STABLE, DECAY_RATE_STABLE)


def _assign_default_decay(node, path: tuple = ()):
    """递归给缺失 decay_type 的 AttrValue 按字段路径补默认策略（就地修改）。"""
    if isinstance(node, list):
        for x in node:
            _assign_default_decay(x, path)
        return node
    if isinstance(node, dict):
        if "value" in node:
            if "decay_type" not in node:
                dt, rate = _default_decay_for_path(path)
                node["decay_type"] = dt
                node["decay_rate"] = rate
            return node
        for k, v in node.items():
            _assign_default_decay(v, path + (k,))
        return node
    return node


def _apply_decay_and_stale(node, daily_decay: float = STABLE_DAILY_DECAY,
                           now: Optional[str] = None):
    """递归重算 effective_confidence 并标记 stale（就地修改）。"""
    if isinstance(node, list):
        for a in node:
            if isinstance(a, dict) and "value" in a:
                eff = effective_confidence(a, daily_decay, now)
                a["effective_confidence"] = round(eff, 4)
                a["stale"] = eff < STALE_THRESHOLD
        return node
    if isinstance(node, dict):
        if "value" in node:
            eff = effective_confidence(node, daily_decay, now)
            node["effective_confidence"] = round(eff, 4)
            node["stale"] = eff < STALE_THRESHOLD
            return node
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                node[k] = _apply_decay_and_stale(v, daily_decay, now)
        return node
    return node


def _active_profile_snapshot(profile: Dict, now: Optional[str] = None) -> Dict:
    """Return a read-only retrieval view with expired/stale facts omitted.

    Stored profile data remains intact for the UI and audit history. This copy
    recomputes decay at read time, so a profile that has aged since the last
    nightly run cannot leak stale facts into a decision prompt.
    """
    candidate = copy.deepcopy(profile) if isinstance(profile, dict) else {}
    candidate = _assign_default_decay(candidate)
    candidate = _apply_decay_and_stale(candidate, now=now)

    def prune(node):
        if isinstance(node, list):
            values = [prune(item) for item in node]
            return [item for item in values if item not in (None, [], {})]
        if isinstance(node, dict):
            if "value" in node:
                if node.get("stale"):
                    return None
                # Conflict history is for audit/display, not current inference.
                return {key: value for key, value in node.items() if key != "history"}
            values = {key: prune(value) for key, value in node.items()}
            return {key: value for key, value in values.items()
                    if value not in (None, [], {})}
        return node

    return prune(candidate) or {}


def _has_profile_content(profile: Dict) -> bool:
    """判断规格书骨架画像是否至少有一个非空属性（用于冷启动判断）。"""
    def count(node):
        if isinstance(node, list):
            return sum(1 for x in node if isinstance(x, dict) and "value" in x)
        if isinstance(node, dict):
            if "value" in node:
                return 1
            return sum(count(v) for v in node.values())
        return 0
    return any(count(profile.get(f)) > 0 for f in PROFILE_FIELDS)


def _collect_attrs(node, out: list, sort_key: str = "confidence"):
    """收集非 stale 画像，并按该维度的文档排序规则排列。"""
    if isinstance(node, dict) and "value" in node:
        if not node.get("stale"):
            out.append(f"{node['value']}({node.get('confidence', 0):.2f})")
    elif isinstance(node, list):
        if sort_key == "frequency":
            key_fn = lambda x: x.get("observations", 1) if isinstance(x, dict) else 0
        elif sort_key == "rank_score":
            key_fn = lambda x: x.get("rank_score", x.get("confidence", 0)) if isinstance(x, dict) else 0
        else:
            key_fn = lambda x: x.get("confidence", 0) if isinstance(x, dict) else 0
        for item in sorted(node, key=key_fn, reverse=True):
            _collect_attrs(item, out, sort_key)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_attrs(value, out, sort_key)


def _empty_moment() -> Dict:
    """A single captured moment (goes into Layer 1 → 2 → 3)."""
    return {
        "timestamp": _now_iso(),
        # Part 1
        "scene": "",          # external environment description
        "user_action": "",    # user's own behaviour / action
        "interaction_details": "",  # source-grounded who-did-what and event order
        "person_observations": [],   # source-linked appearance and explicitly stated age
        "temporal_facts": [],        # source-linked plans and object/person state changes
        "object_observations": [],   # grounded appearance and location cues with frame provenance
        "source_time": {},           # source-relative date/time, distinct from ingest time
        "source_evidence": "",  # timestamped source-video speech; speaker is not inferred
        # Part 2
        "needs": [],          # [{"need": str, "confidence": float}]
        # Part 3
        "solutions": [],      # [{"need": str, "solution": str}]
        # Part 4 feedback (filled in later, does NOT block Part 1-3)
        "feedback": {
            "confirmed": False,
            "corrections": {},   # {field: corrected_value}
            "user_rating": None
        },
        "highlighted": False,
        "layer": 1
    }


def _clean_source_fact_rows(rows, kind: str, limit: int) -> list:
    """Validate compact, source-linked facts; collapse repeated person-frame descriptions."""
    allowed = ({"entity", "appearance", "face_features", "apparent_age_range", "apparent_age_basis", "age", "age_basis", "frame_index",
                "frame_time_s", "evidence", "confidence", "evidence_ref", "evidence_refs",
                "spoken_aliases", "relationships", "profile_context",
                "profile_confidence"}
               if kind == "person" else
               {"subject_type", "subject", "predicate", "value", "actor",
                "affected_people", "from_location", "to_location", "from_state",
                "to_state", "time_scope", "time_order", "frame_index",
                "frame_time_s", "modality", "evidence", "status", "confidence",
                "evidence_ref"})
    output = []
    seen_people = {}

    def clean_ref(source_ref):
        if not isinstance(source_ref, dict):
            return None
        ref = {}
        segment = source_ref.get("source_segment_id")
        if isinstance(segment, str) and segment.strip():
            ref["source_segment_id"] = segment.strip()[:128]
        modality = source_ref.get("modality")
        if modality in {"visual", "speech"}:
            ref["modality"] = modality
        frame_index = source_ref.get("frame_index")
        if isinstance(frame_index, int) and not isinstance(frame_index, bool) and frame_index > 0:
            ref["frame_index"] = min(frame_index, 999)
        for key in ("frame_time_s", "clip_start_s", "clip_end_s",
                    "audio_start_s", "audio_end_s"):
            try:
                number = float(source_ref.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number >= 0:
                ref[key] = round(number, 3)
        return ref or None

    def add_refs(row, refs):
        combined = list(row.get("evidence_refs") or [])
        if not combined and isinstance(row.get("evidence_ref"), dict):
            combined.append(row["evidence_ref"])
        combined.extend(ref for ref in refs if ref)
        unique = []
        seen = set()
        for ref in combined:
            cleaned = clean_ref(ref)
            if not cleaned:
                continue
            signature = (cleaned.get("source_segment_id"), cleaned.get("modality"),
                         cleaned.get("frame_index"), cleaned.get("frame_time_s"),
                         cleaned.get("audio_start_s"), cleaned.get("audio_end_s"))
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(cleaned)
            if len(unique) >= 16:
                break
        if unique:
            row["evidence_ref"] = unique[0]
            if len(unique) > 1:
                row["evidence_refs"] = unique
            else:
                row.pop("evidence_refs", None)

    # Apply the limit after deduplication so repeated frames cannot crowd out
    # observations of other people in the same video.
    for source in rows[:64] if isinstance(rows, list) else []:
        if not isinstance(source, dict):
            continue
        row = {}
        for key in allowed - {"evidence_ref", "evidence_refs"}:
            value = source.get(key)
            if key in {"frame_index", "time_order"}:
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    row[key] = min(value, 999)
            elif key == "frame_time_s":
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number) and number >= 0:
                    row[key] = round(number, 3)
            elif key == "affected_people" and isinstance(value, list):
                row[key] = list(dict.fromkeys(
                    item.strip()[:120] for item in value
                    if isinstance(item, str) and item.strip()))[:8]
            elif key == "relationships" and isinstance(value, list):
                relationships = list(dict.fromkeys(
                    " ".join(str(item).split()).strip()[:180]
                    for item in value if isinstance(item, str) and item.strip()
                ))[:8]
                if relationships:
                    row[key] = relationships
            elif key == "profile_confidence":
                if value in {"high", "medium", "low"}:
                    row[key] = value
            elif key == "spoken_aliases" and isinstance(value, list):
                aliases = []
                for item in value[:4]:
                    if not isinstance(item, dict):
                        continue
                    alias = str(item.get("alias") or "").strip()[:80]
                    if not alias:
                        continue
                    cleaned_alias = {"alias": alias}
                    if item.get("status") == "probable_visual_match":
                        cleaned_alias["status"] = item["status"]
                    if item.get("confidence") in {"high", "medium", "low"}:
                        cleaned_alias["confidence"] = item["confidence"]
                    basis = str(item.get("basis") or "").strip()
                    if basis:
                        cleaned_alias["basis"] = basis[:320]
                    ref = clean_ref(item.get("evidence_ref"))
                    if ref:
                        cleaned_alias["evidence_ref"] = ref
                    aliases.append(cleaned_alias)
                if aliases:
                    row[key] = aliases
            elif isinstance(value, str) and value.strip():
                row[key] = value.strip()[:500]
        refs = []
        supplied_refs = source.get("evidence_refs")
        if isinstance(supplied_refs, list):
            refs.extend(ref for ref in supplied_refs if isinstance(ref, dict))
        singular_ref = source.get("evidence_ref")
        if isinstance(singular_ref, dict):
            refs.append(singular_ref)
        if refs:
            add_refs(row, refs)
        elif row.get("frame_time_s") is not None or row.get("frame_index") is not None:
            ref = {}
            if row.get("frame_time_s") is not None:
                ref["frame_time_s"] = row["frame_time_s"]
            if row.get("frame_index") is not None:
                ref["frame_index"] = row["frame_index"]
            add_refs(row, [ref])
        if kind == "person":
            # Exact age is source-stated only; a separate broad visual band is explicitly marked as an estimate.
            if row.get("age_basis") not in {"stated", "visible_text"}:
                row.pop("age", None)
            if row.get("age_basis") not in {"stated", "visible_text", "unknown"}:
                row.pop("age_basis", None)
            if (row.get("apparent_age_basis") != "visual_estimate"
                    or row.get("apparent_age_range") not in {
                        "child", "teenager", "18-29", "30-44", "45-59", "60-74", "75+"
                    }):
                row.pop("apparent_age_range", None)
                row.pop("apparent_age_basis", None)
        if row.get("confidence") not in {"high", "medium", "low"}:
            row.pop("confidence", None)
        if row.get("modality") not in {"visual", "speech"}:
            row.pop("modality", None)
        if row.get("status") not in {
                "observed", "stated", "self_identified", "spoken_event", "stated_plan",
                "self_reported", "self_reported_unverified", "unverified_claim",
                "quoted_unverified_claim", "quoted_relationship_term", "completed", "unclear"}:
            row.pop("status", None)
        if not (row.get("entity") or row.get("subject")):
            continue
        if kind == "person":
            key = (
                " ".join(str(row.get("entity") or "").split()).casefold(),
                " ".join(str(row.get("appearance") or "").split()).casefold(),
                " ".join(str(row.get("face_features") or "").split()).casefold(),
                row.get("age_basis", ""),
                row.get("age", "") if row.get("age_basis") in {"stated", "visible_text"} else "",
            )
            if key in seen_people:
                add_refs(output[seen_people[key]], refs or ([row.get("evidence_ref")] if row.get("evidence_ref") else []))
                rank = {"low": 0, "medium": 1, "high": 2}
                old_conf, new_conf = output[seen_people[key]].get("confidence"), row.get("confidence")
                if rank.get(new_conf, -1) > rank.get(old_conf, -1):
                    output[seen_people[key]]["confidence"] = new_conf
                continue
            if len(output) >= limit:
                continue
            seen_people[key] = len(output)
        elif len(output) >= limit:
            continue
        output.append(row)
    return output


def normalize_object_transition_facts(moment: Dict) -> Dict:
    """Make an explicitly moved item the fact subject while preserving its actor."""
    if not isinstance(moment, dict):
        return moment
    facts = moment.get("temporal_facts") or []
    if not isinstance(facts, list):
        return moment
    objects = moment.get("objects")
    if not isinstance(objects, list):
        objects = []
    human_terms = re.compile(
        r"\b(?:person|people|man|woman|boy|girl|child|he|she|they)\b", re.I)
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        predicate = str(fact.get("predicate") or "").casefold().strip()
        if predicate not in {"move", "moved", "relocate", "relocated", "transfer", "transferred"}:
            continue
        value = str(fact.get("value") or "").strip()
        if not value:
            continue
        current_subject = str(fact.get("subject") or "").strip()
        original_type = str(fact.get("subject_type") or "").casefold()
        actor = str(fact.get("actor") or "").strip()
        if (original_type == "object" and actor
                and re.match(r"^(?:from|to|toward|towards|into|out of)\s+", current_subject, re.I)):
            # A place phrase produced by an older extractor is not an object. If
            # the fact names a person actor, preserve the movement as that person's event.
            fact["subject_type"] = "person"
            fact["subject"] = actor
            continue
        if original_type == "object" and current_subject:
            candidate = current_subject
        else:
            match = re.match(
                r"^(?:moved|relocated|transferred|placed)\s+(?:(?:the|a|an)\s+)?(.+?)\s+(?:from|out of|off|onto|to)\b",
                value, re.I)
            if not match:
                match = re.match(
                    r"^(?:(?:the|a|an)\s+)?(.+?)\s+(?:from|out of|off|onto|to)\b",
                    value, re.I)
            candidate = match.group(1) if match else ""
            candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate.strip(), flags=re.I)
        candidate = " ".join(candidate.split()).strip(" .,:;-")[:120]
        if not candidate or human_terms.search(candidate):
            continue
        if not actor and original_type == "person":
            actor = current_subject
        if actor:
            fact["actor"] = actor
        fact["subject_type"] = "object"
        fact["subject"] = candidate
        if not any(str(item).casefold() == candidate.casefold() for item in objects):
            objects.append(candidate)
    if objects:
        moment["objects"] = list(dict.fromkeys(
            str(item).strip() for item in objects if str(item).strip()
            and not re.match(r"^(?:from|to|toward|towards|into|out of)\s+", str(item).strip(), re.I)))[:40]
    return moment


def object_visual_observations(moment: Dict, limit: int = 16) -> list:
    """Expose compact, source-linked object appearance facts from stored grounding."""
    if not isinstance(moment, dict):
        return []
    saved = moment.get("object_observations")
    if isinstance(saved, list) and saved:
        return copy.deepcopy(saved[:max(0, int(limit))])
    media = moment.get("multimodal") or {}
    grounding = media.get("object_visuals") or {} if isinstance(media, dict) else {}
    rows = []
    fallback_segment = str((moment.get("source_provenance") or {}).get("source_segment_id") or "")
    for item in grounding.get("items") or []:
        if not isinstance(item, dict) or item.get("status") not in {"grounded", "visible_unlabeled"}:
            continue
        label = " ".join(str(item.get("object") or "").split())[:180]
        appearance = " ".join(str(item.get("appearance") or "").split())[:500]
        if not label and not appearance:
            continue
        refs = []
        for ref in (item.get("frame_refs") or [])[:3]:
            if not isinstance(ref, dict):
                continue
            entry = {"source_segment_id": str(ref.get("source_id") or fallback_segment)[:128],
                     "modality": "visual"}
            try:
                at = float(ref.get("at_s"))
                if math.isfinite(at) and at >= 0:
                    entry["frame_time_s"] = round(at, 3)
            except (TypeError, ValueError):
                pass
            if entry.get("source_segment_id") or entry.get("frame_time_s") is not None:
                refs.append(entry)
        rows.append({"object": label, "appearance": appearance,
                     "status": item.get("status"), "evidence_refs": refs})
        if len(rows) >= max(0, int(limit)):
            break
    return rows


def _portable_video_id(value: Any) -> str:
    """Store a dataset-relative source ID, never a host-specific absolute path."""
    normalized = str(value or "").replace("\\", "/")
    absolute = normalized.startswith("/")
    parts = [part for part in normalized.strip("/").split("/")
             if part and part not in {".", ".."}]
    absolute = absolute or bool(parts and parts[0].endswith(":"))
    marker = next((i for i, part in enumerate(parts)
                   if part.casefold() == "egolife"), None)
    if marker is not None:
        parts = parts[marker + 1:]
    elif absolute:
        parts = parts[-1:]
    return "/".join(parts)[:512]


def _moment_entities(moment: Dict) -> List[Dict[str, str]]:
    """Read typed entities from one event; these links are derived, not duplicated in storage."""
    rows = []
    media = moment.get("multimodal")
    mm_entities = (media.get("entities") or {}) if isinstance(media, dict) else {}
    if not isinstance(mm_entities, dict):
        mm_entities = {}
    fields = [
        ("person", moment.get("people") or []),
        ("location", [moment.get("location")] if moment.get("location") else []),
        ("activity", [moment.get("activity")] if moment.get("activity") else []),
        ("object", moment.get("objects") or []),
        ("environment", moment.get("environments") or []),
        ("person", mm_entities.get("people") or []),
        ("location", [mm_entities.get("location")] if mm_entities.get("location") else []),
        ("activity", [mm_entities.get("activity")] if mm_entities.get("activity") else []),
        ("object", mm_entities.get("objects") or []),
        ("environment", mm_entities.get("environments") or []),
    ]
    seen_by_type = {}
    fields.append(("person", [row.get("entity") for row in moment.get("person_observations") or []
                              if isinstance(row, dict)]))
    kind_map = {"person": "person", "object": "object", "place": "location", "environment": "environment"}
    for fact in moment.get("temporal_facts") or []:
        if not isinstance(fact, dict):
            continue
        if fact.get("subject_type") in kind_map:
            fields.append((kind_map[fact["subject_type"]], [fact.get("subject")]))
        fields.append(("person", [fact.get("actor")] + (fact.get("affected_people") or [])))
    for entity_type, values in fields:
        if not isinstance(values, (list, tuple)):
            continue
        seen = seen_by_type.setdefault(entity_type, set())
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            label = " ".join(value.strip().split())[:180]
            key = label.casefold()
            if key not in seen:
                rows.append({"type": entity_type, "name": label})
                seen.add(key)
    speakers = mm_entities.get("speakers") or []
    if isinstance(speakers, (list, tuple)):
        known_people = {row["name"].casefold() for row in rows if row["type"] == "person"}
        for speaker in speakers:
            if isinstance(speaker, str) and speaker.strip() and speaker.casefold() not in known_people:
                rows.append({"type": "person", "name": " ".join(speaker.strip().split())[:180]})
                known_people.add(speaker.casefold())
    return rows


def entity_graph_summary(layer7: Dict, limit: int = 8) -> Dict:
    """Build a compact, event-grounded entity graph view without storing duplicate indexes."""
    moments = {key: value for key, value in (layer7.get("moments") or {}).items()
               if isinstance(value, dict) and not value.get("stale") and not value.get("merged_into")}
    entity_keys = set()
    relation_count = 0
    for moment in moments.values():
        entities = _moment_entities(moment)
        entity_keys.update((row["type"], row["name"].casefold()) for row in entities)
        relation_count += len(entities)  # each typed entity is linked to its source event
        people_count = sum(row["type"] == "person" for row in entities)
        relation_count += min(people_count * (people_count - 1) // 2, 66)

    samples = []
    ordered = sorted(moments.values(), key=lambda item: str(item.get("timestamp") or ""),
                     reverse=True)
    for moment in ordered[:max(0, int(limit))]:
        entities = _moment_entities(moment)
        people = [row["name"] for row in entities if row["type"] == "person"]
        provenance = dict(moment.get("source_provenance") or {})
        if provenance.get("source_video_id"):
            provenance["source_video_id"] = _portable_video_id(
                provenance["source_video_id"])
        media = moment.get("multimodal") or {}
        visual_refs = media.get("visual_frames") or ([media["visual"]] if media.get("visual") else [])
        audio_refs = media.get("audio_clips") or ([media["audio"]] if media.get("audio") else [])
        def compact_refs(refs, keys):
            return [{key: ref.get(key) for key in keys if ref.get(key) is not None}
                    for ref in refs[:3] if isinstance(ref, dict)]
        compact_media = {
            "source_id": str(media.get("source_id") or provenance.get("source_segment_id") or ""),
            "visual_frames": compact_refs(
                visual_refs, ("asset", "bytes", "mime", "at_s", "source_id")),
            "audio_clips": compact_refs(
                audio_refs, ("asset", "bytes", "mime", "at_s", "duration_s", "source_id")),
            "person_crops": compact_refs(
                [ref for item in (media.get("person_visuals") or {}).get("items") or []
                 for ref in item.get("frame_refs") or []],
                ("asset", "bytes", "mime", "at_s", "source_id", "visual_role", "crop_bbox")),
            "object_crops": compact_refs(
                [ref for item in (media.get("object_visuals") or {}).get("items") or []
                 for ref in item.get("frame_refs") or []],
                ("asset", "bytes", "mime", "at_s", "source_id", "visual_role", "crop_bbox")),
        }
        relations = []
        for entity in entities:
            predicate = {
                "person": "participated_in",
                "location": "occurred_at",
                "activity": "classified_as",
                "object": "involves_object",
                "environment": "has_environment",
            }[entity["type"]]
            relations.append({"entity": entity["name"], "type": entity["type"],
                              "relation": predicate, "evidence_id": moment.get("id")})
        for i, left in enumerate(people[:12]):
            for right in people[i + 1:12]:
                relations.append({"entity": left, "related_entity": right,
                                  "relation": "co_present_in_event",
                                  "evidence_id": moment.get("id")})
        samples.append({
            "moment_id": moment.get("id"),
            "timestamp": moment.get("timestamp"),
            "source_time": moment.get("source_time") or {},
            "event": str(moment.get("scene") or moment.get("activity") or "")[:200],
            "action": str(moment.get("user_action") or "")[:240],
            "interaction_details": str(moment.get("interaction_details") or "")[:360],
            "people": [row["name"] for row in entities if row["type"] == "person"],
            "person_observations": list(moment.get("person_observations") or [])[:16],
            "temporal_facts": list(moment.get("temporal_facts") or [])[:32],
            "activity": str(moment.get("activity") or ""),
            "location": str(moment.get("location") or ""),
            "objects": [row["name"] for row in entities if row["type"] == "object"],
            "environments": [row["name"] for row in entities if row["type"] == "environment"],
            "entities": entities[:24],
            "relations": relations[:48],
            "source_provenance": provenance,
            "multimodal": compact_media,
        })
    return {"entity_count": len(entity_keys), "relation_count": relation_count,
            "storage_mode": "derived_from_source_moments", "samples": samples}


def entity_relation_evidence(layer7: Dict, names: List[str], limit: int = 4) -> List[Dict]:
    """Return only source moments that directly support a requested entity association."""
    moments = layer7.get("moments") or {}
    indexes = [layer7.get(key) or {} for key in
               ("people", "locations", "activity_events", "objects", "environments")]
    matches = []
    for requested in names:
        wanted = str(requested or "").strip().casefold()
        if not wanted:
            continue
        candidate_ids = []
        for index in indexes:
            for tag, ids in index.items():
                if str(tag).casefold() == wanted:
                    candidate_ids.extend(ids[-max(1, limit * 2):])
        seen_ids = set()
        for moment_id in reversed(candidate_ids):
            if moment_id in seen_ids:
                continue
            seen_ids.add(moment_id)
            moment = moments.get(moment_id)
            if not moment or moment.get("stale") or moment.get("merged_into"):
                continue
            entities = _moment_entities(moment)
            associated = [row for row in entities if row["name"].casefold() != wanted]
            if not associated:
                continue
            matches.append({
                "entity": str(requested),
                "associated_entities": associated[:12],
                "relation": "observed_in_same_source_event",
                "evidence_id": moment_id,
                "timestamp": str(moment.get("timestamp") or ""),
            })
            if len(matches) >= max(1, limit):
                return matches
    return matches


def _empty_db() -> Dict:
    return {
        # --- raw episodic layers ---
        "layer1": [],     # current moment (last N moments, usually just 1)
        "layer2": [],     # same-env moments from this session
        "layer3": [],     # all moments from today

        # --- compressed summary layers (text + optional image paths) ---
        "layer4": {
            "summary": "",
            "current_tasks": [],
            "stated_plans": [],
            "life_trajectory": "",
            "last_updated": None,
            "source_moment_count": 0,
            "source_moment_ids": [],
            "source_link_coverage": "unknown"
        },
        "layer5": {
            "summary": "",
            "key_events": [],
            "long_term_patterns": "",
            "last_updated": None,
            "source_moment_count": 0,
            "source_moment_ids": [],
            "source_link_coverage": "unknown"
        },
        "layer6": {
            "summary": "",
            "profile": {
                "demographics": {},
                "preferences": {},
                "frequent_locations": [],
                "behavior_patterns": {},
                "personality": {},
                "goals": [],
                "decisions": {},
                "motivations": []
            },
            "last_updated": None,
            "source_moment_count": 0,
            "source_moment_ids": [],
            "source_link_coverage": "unknown"
        },

        # --- classified archive ---
        "layer7": {
            "people": {},          # 人物（从近到远）
            "locations": {},       # 空间位置（从近到远）
            "time_nodes": {},      # 时间节点 日/周/月/年（从近到远）
            "activity_events": {}, # 活动事件（从高频到低频）
            "environments": {},    # 环境场景（从高频到低频）
            "objects": {},         # 物品（从高频到低频）
            "moments": {}          # {moment_id: moment_dict}  — master store
        },

        # --- housekeeping ---
        "metadata": {
            "total_moments": 0,
            "total_encounters": 0,   # kept for backward compat
            "last_consolidation": None,
            "session_start": _now_iso(),
            "today": _today_str()
        }
    }


# ---------------------------------------------------------------------------
# PersonMemory (drop-in replacement for memory2.PersonMemory)
# ---------------------------------------------------------------------------

class PersonMemory:
    """
    7-layer hierarchical memory with 9 DB operations.

    The key design change vs memory2:
      * Consolidation (compress / sort / combine) is NEVER called inside
        add() or update() — it must be triggered explicitly by the pipeline
        AFTER Part2+3 need-analysis finishes, so it never blocks Part1-3.
      * Part4 feedback is stored via update_feedback() and runs
        asynchronously / after the main pipeline returns.
    """

    MAX_LAYER1 = 5      # keep last N moments in the "current" layer
    MAX_LAYER2 = 20     # same-env history within a session
    MAX_LAYER3 = 1000   # today's window（设计说明书 §2.4：100→1000，EgoLife 实测校准）

    def __init__(self, memory_dir: str = "memory", translate_fn=None):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.memory_file = self.memory_dir / "memory.json"
        self.memory: Dict = self._load()
        self._lexical_search_cache = None
        # A monotonic generation lets query check index freshness without
        # re-hashing the full archive on every retrieval.
        self.memory.setdefault("metadata", {}).setdefault("semantic_index_revision", 0)
        # 非英文 → 英文翻译函数（统一英文分词策略，见设计说明书 §6.2）。
        # 可注入；None 时跳过翻译（非英文文本无法分词，检索走原文兜底降级）。
        self.translate_fn = translate_fn
        self._semantic_index_enabled = os.environ.get("MEMORY_SEMANTIC_RRF", "0").lower() in {"1", "true", "yes", "on"}
        if self.memory_file.exists() and self._ensure_rich_entity_memory():
            self._save()
        if self._semantic_index_enabled:
            from semantic_retrieval import schedule_sync
            revision = self.memory.get("metadata", {}).get("semantic_index_revision", 0)
            schedule_sync(self.memory_dir, revision=revision)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict:
        if self.memory_file.exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Migrate old format if needed
            if "layer1" not in data:
                data = self._migrate_from_v2(data)
            # Migrate old layer6.profile (name/basic_info/preferences/habits) to
            # spec-skeleton fields (demographics/preferences/frequent_locations/behavior_patterns).
            data["layer6"]["profile"] = _migrate_profile(data["layer6"].get("profile", {}))
            # Additive migration: legacy summaries have no auditable source mapping.
            for layer_no in (4, 5, 6):
                layer = data.setdefault(f"layer{layer_no}", {})
                layer.setdefault("source_moment_ids", [])
                layer.setdefault("source_link_coverage", "unknown")
                layer.setdefault("source_moment_count", 0)
            # 清洗 layer7 索引污染（修复 G7 缺陷）
            _clean_layer7_indices(data.get("layer7", {}))
            return data
        return _empty_db()

    def _ensure_rich_entity_memory(self) -> bool:
        """One-time additive migration for entity indexes, event types and visual links."""
        metadata = self.memory.setdefault("metadata", {})
        if int(metadata.get("rich_entity_memory_version", 0) or 0) >= 1:
            return False
        changed = False
        layer7 = self.memory.setdefault("layer7", {})
        moments = layer7.setdefault("moments", {})
        for moment_id, moment in moments.items():
            if not isinstance(moment, dict):
                continue
            before = copy.deepcopy(moment)
            _upgrade_legacy_person_labels(moment)
            normalize_object_transition_facts(moment)
            media = moment.get("multimodal")
            if isinstance(media, dict):
                entity_visuals = _build_entity_visuals(moment)
                if entity_visuals.get("items") and media.get("entity_visuals") != entity_visuals:
                    media["entity_visuals"] = entity_visuals
            if moment != before:
                changed = True
            for layer_key in ("layer1", "layer2", "layer3"):
                for index, window_moment in enumerate(self.memory.get(layer_key, [])):
                    if window_moment.get("id") == moment_id and window_moment != moment:
                        self.memory[layer_key][index] = moment
                        changed = True
        _clean_layer7_indices(layer7)
        for index_key in LAYER7_INDICES:
            index = layer7.setdefault(index_key, {})
            for tag, ids in list(index.items()):
                valid_ids = list(dict.fromkeys(mid for mid in ids if mid in moments))
                if valid_ids != ids:
                    changed = True
                if valid_ids:
                    index[tag] = valid_ids
                else:
                    index.pop(tag, None)
        for moment_id, moment in moments.items():
            if not isinstance(moment, dict) or moment.get("merged_into"):
                continue
            for index_key, tags in _moment_index_tags(moment).items():
                index = layer7.setdefault(index_key, {})
                for tag in tags:
                    ids = index.setdefault(tag, [])
                    if moment_id not in ids:
                        ids.append(moment_id)
                        changed = True
        metadata["rich_entity_memory_version"] = 1
        return True

    def clean_layer7_indices(self) -> Dict[str, List[str]]:
        """清洗 layer7 索引并落盘保存。"""
        removed = _clean_layer7_indices(self.memory.get("layer7", {}))
        if any(removed.values()):
            self._save()
        return removed

    def _save(self):
        metadata = self.memory.setdefault("metadata", {})
        try:
            current_revision = int(metadata.get("semantic_index_revision", 0))
        except (TypeError, ValueError):
            current_revision = 0
        metadata["semantic_index_revision"] = current_revision + 1
        # 原子落盘：先写临时文件再 rename，防止并发写坏 / 写一半崩溃（04 文档 2.5 步骤⑥）。
        tmp = self.memory_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            # Windows/archive snapshots retain all fields; whitespace is not data.
            json.dump(self.memory, f, separators=(",", ":"), ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.memory_file)
        if getattr(self, "_semantic_index_enabled", False):
            from semantic_retrieval import schedule_sync
            schedule_sync(self.memory_dir, revision=metadata["semantic_index_revision"])

    def _migrate_from_v2(self, old: Dict) -> Dict:
        """Best-effort migration from memory2 format."""
        db = _empty_db()
        # port people → layer7 people index
        for name, pdata in old.get("people", {}).items():
            db["layer7"]["people"][name] = []
        # port locations → layer7 locations index
        for loc in old.get("locations", {}).keys():
            db["layer7"]["locations"][loc] = []
        db["metadata"]["total_encounters"] = old.get("metadata", {}).get("total_encounters", 0)
        return db

    # ------------------------------------------------------------------
    # Internal ID generation
    # ------------------------------------------------------------------

    def _new_moment_id(self) -> str:
        return f"m_{int(time.time() * 1000)}_{self.memory['metadata']['total_moments']}"

    # ------------------------------------------------------------------
    # Operation 1 : ADD
    # ------------------------------------------------------------------

    def add(self,
            scene: str,
            user_action: str,
            needs: List[Dict],
            solutions: List[Dict],
            people: List[str] = None,
            location: str = None,
            activity: str = None,
            environments: List[str] = None,
            objects: List[str] = None,
            extra_notes: str = "",
            timestamp: str = None,
            source_evidence: str = "",
            interaction_details: str = "",
            person_observations: List[Dict] = None,
            temporal_facts: List[Dict] = None,
            source_time: Dict = None,
            multimodal: Dict = None,
            source_provenance: Dict = None) -> str:
        """
        Store a complete Part1+2+3 result.
        Returns the moment_id so Part4 can reference it later.

        Layers updated: 1, 2, 3, 7 (index only)
        Layers 4-6 are updated only during consolidation (not here).
        """
        moment = _empty_moment()
        event_time = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
        moment["timestamp"] = event_time.isoformat()
        moment["scene"] = scene
        moment["user_action"] = user_action
        if interaction_details:
            moment["interaction_details"] = str(interaction_details)[:4000]
        if isinstance(person_observations, list):
            moment["person_observations"] = _clean_source_fact_rows(
                person_observations, "person", 16)
        if isinstance(temporal_facts, list):
            moment["temporal_facts"] = _clean_source_fact_rows(
                temporal_facts, "fact", 32)
        if isinstance(source_time, dict):
            clean_time = {}
            for key in ("absolute_date", "day_label", "time_of_day", "time_occasion", "basis", "source_segment_id"):
                value = source_time.get(key)
                if isinstance(value, str) and value.strip():
                    clean_time[key] = value.strip()[:128]
            for key in ("clip_start_s", "clip_end_s"):
                try:
                    value = float(source_time.get(key))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0:
                    clean_time[key] = round(value, 3)
            if clean_time:
                moment["source_time"] = clean_time
        if source_evidence:
            moment["source_evidence"] = source_evidence
        # Legacy arguments remain accepted for caller compatibility. Needs and
        # proposed solutions belong to downstream reasoning, not stored facts.
        if location:
            moment["location"] = location
        if people:
            moment["people"] = people
        if activity:
            moment["activity"] = activity
        if environments:
            moment["environments"] = environments      # 环境场景（规范新增维度）
        if objects:
            moment["objects"] = objects                # 物品（规范新增维度）
        if extra_notes:
            moment["notes"] = extra_notes
        if isinstance(multimodal, dict) and multimodal.get("schema") == 1:
            moment["multimodal"] = multimodal
        normalize_object_transition_facts(moment)
        if isinstance(moment.get("multimodal"), dict):
            entity_visuals = _build_entity_visuals(moment)
            if entity_visuals.get("items"):
                moment["multimodal"]["entity_visuals"] = entity_visuals
        moment["object_observations"] = object_visual_observations(moment)
        objects = moment.get("objects") or []
        if isinstance(source_provenance, dict):
            provenance = {}
            for key in ("source_video_id", "source_segment_id", "clip_start_s",
                        "clip_end_s", "source_video_sha256", "source_clip_sha256",
                        "perception_prompt_version", "list_parser_version",
                        "extraction_batch_id"):
                value = source_provenance.get(key)
                if value not in (None, ""):
                    provenance[key] = value
            if provenance.get("source_video_id"):
                provenance["source_video_id"] = _portable_video_id(
                    provenance["source_video_id"])
            if provenance:
                moment["source_provenance"] = provenance

        moment_id = self._new_moment_id()
        moment["id"] = moment_id

        # 预翻译（统一英文分词：写入时翻译非英文文本存 normalized，检索查英文）
        if self.translate_fn:
            normalized = {}
            for field, raw in (("scene", scene), ("user_action", user_action),
                               ("interaction_details", interaction_details),
                               ("location", location), ("activity", activity),
                               ("notes", extra_notes), ("source_evidence", source_evidence)):
                if raw and _contains_cjk(str(raw)):
                    try:
                        normalized[field] = self.translate_fn(str(raw))
                    except Exception:
                        pass  # 翻译失败降级：不存 normalized，原文兜底
            if normalized:
                moment["normalized"] = normalized

        # ---- 跨天清理（设计说明书 §2.4：窗口视图，新一天清空窗口）----
        today = _today_str()
        if self.memory["metadata"]["today"] != today:
            self.memory["layer1"].clear()
            self.memory["layer2"].clear()
            self.memory["layer3"].clear()
            self.memory["metadata"]["today"] = today
            self.memory["metadata"]["session_start"] = _now_iso()

        # ---- Layer 1 (current window) ----
        self.memory["layer1"].append(moment)
        if len(self.memory["layer1"]) > self.MAX_LAYER1:
            # oldest moment slides out: same-env → layer2, otherwise → layer3
            old = self.memory["layer1"].pop(0)
            self._graduate_to_layer2(old, moment)

        # ---- Layer 3 (today's window) ----
        self.memory["layer3"].append(moment)
        if len(self.memory["layer3"]) > self.MAX_LAYER3:
            # 窗口超限：最旧滑出（真身保留在 layer7.moments，不丢数据）
            self.memory["layer3"].pop(0)

        # ---- Layer 7 master store ----
        self.memory["layer7"]["moments"][moment_id] = moment

        # ---- Layer 7 indices ----
        for index_key, tags in _moment_index_tags(moment).items():
            for tag in tags:
                self.memory["layer7"][index_key].setdefault(tag, []).append(moment_id)

        # ---- metadata ----
        self.memory["metadata"]["total_moments"] += 1
        self.memory["metadata"]["total_encounters"] += 1

        self._save()
        return moment_id

    def _graduate_to_layer2(self, moment: Dict, current: Dict):
        """layer1 滑出（设计说明书 §2.4 / 09-22 层级规范）：与当前 moment 同环境场景
        （environments 有交集）→ layer2；否则直接滑入 layer3。
        无 environments 时降级为 location+activity 判定。
        layer2 超限时最旧滑入 layer3（不丢弃，真身保留在 layer7.moments）。
        窗口内 moment 唯一（滑入前按 id 去重，防止 add 时已入 layer3 的重复）。
        """
        def _push(target: list, m: Dict, layer_no: int):
            m["layer"] = layer_no
            if not any(x.get("id") == m.get("id") for x in target):
                target.append(m)

        # 规范（09-22 层级定义）：layer2 归类依据是「环境场景」
        # —— Part1 识别的环境场景相同即同场景。环境场景有交集 → layer2。
        cur_envs = set(current.get("environments") or [])
        moment_envs = set(moment.get("environments") or [])
        if cur_envs and moment_envs:
            same_env = bool(cur_envs & moment_envs)
        else:
            # 降级：无环境场景标注（旧数据/未传）时用 location+activity 判定
            same_env = (
                bool(moment.get("location"))
                and moment.get("location") == current.get("location")
                and moment.get("activity") == current.get("activity")
            )
        if same_env:
            _push(self.memory["layer2"], moment, 2)
            if len(self.memory["layer2"]) > self.MAX_LAYER2:
                old = self.memory["layer2"].pop(0)
                _push(self.memory["layer3"], old, 3)
        else:
            _push(self.memory["layer3"], moment, 3)
            if len(self.memory["layer3"]) > self.MAX_LAYER3:
                self.memory["layer3"].pop(0)

    # ------------------------------------------------------------------
    # Operation 2 : UPDATE
    # ------------------------------------------------------------------

    def update_feedback(self, moment_id: str, corrections: Dict = None,
                        confirmed: bool = False, user_rating: int = None) -> bool:
        """保存 Part4 反馈；修正沿用 UPDATE 校验并同步 Layer7 索引。"""
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment:
            return False

        feedback_corrections = corrections if isinstance(corrections, dict) else {}
        valid_corrections = self._validated_updates(feedback_corrections)
        if valid_corrections:
            self.update(moment_id, valid_corrections)
            moment = self.memory["layer7"]["moments"].get(moment_id)
            if not moment:
                return False

        fb = moment.setdefault("feedback", {
            "confirmed": False, "corrections": {}, "user_rating": None
        })
        if not isinstance(fb, dict):
            fb = {"confirmed": False, "corrections": {}, "user_rating": None}
            moment["feedback"] = fb
        compliance_only = (
            "compliance_check" in feedback_corrections and not valid_corrections
        )
        if not compliance_only and isinstance(confirmed, bool):
            fb["confirmed"] = confirmed
        if feedback_corrections:
            stored = fb.get("corrections")
            if not isinstance(stored, dict):
                stored = {}
                fb["corrections"] = stored
            stored.update(feedback_corrections)
        if user_rating is not None and isinstance(user_rating, (int, float)) \
                and not isinstance(user_rating, bool) and 1 <= user_rating <= 5:
            fb["user_rating"] = user_rating

        if fb.get("confirmed"):
            moment["highlighted"] = True
            moment["highlight_source"] = "feedback_confirmed"
            moment["highlighted_at"] = _now_iso()
        self._sync_window_moment(moment_id)
        self._save()
        return True

    def update_habits(self, profile_updates: Dict):
        """[已弃用] 旧画像更新接口（浅覆盖，会丢历史）。请改用 update_profile()。"""
        print("⚠️  update_habits() 已弃用 —— 画像请用 update_profile()（合并语义 + 原子落盘）")
        self.update_profile(profile_updates, moment_id="update_habits")

    # ------------------------------------------------------------------
    # User profile API（规格书骨架，唯一画像写入入口）
    # ------------------------------------------------------------------

    def _merge_profile_node(self, old, new, source: str, timestamp: str):
        """递归合并画像节点：list → 追加去重；AttrValue → 佐证/裁决；dict → 递归。"""
        if isinstance(new, list):
            new_attrs = []
            for x in new:
                a = _finalize_attr(x, source, timestamp)
                if a and a["confidence"] >= MIN_CONFIDENCE:
                    new_attrs.append(a)
            return _merge_attr_list(old if isinstance(old, list) else [], new_attrs, now=timestamp)

        if isinstance(new, dict):
            if "value" in new:
                na = _finalize_attr(new, source, timestamp)
                if na is None or na["confidence"] < MIN_CONFIDENCE:
                    return old
                if isinstance(old, dict) and "value" in old:
                    old_effective = effective_confidence(old, now=timestamp)
                    if old_effective < STALE_THRESHOLD:
                        return _replace_stale_attr(old, na, old_effective)
                    if _bigram_jaccard(old["value"], na["value"]) >= MERGE_SIM_THRESHOLD:
                        return _corroborate_attr(old, na)
                    return _resolve_conflict(old, na)
                return na
            # 嵌套 dict：递归合并子键
            if not isinstance(old, dict):
                old = {}
            for k, v in new.items():
                if v in (None, [], {}):
                    continue
                old[k] = self._merge_profile_node(old.get(k), v, source, timestamp)
            return old

        return old

    def update_profile(self, extracted: Dict, moment_id: Optional[str] = None,
                       timestamp: Optional[str] = None) -> Dict:
        """画像更新的唯一入口（对齐 04 文档 2.5）。与 compress() 完全解耦。

        流程：字段白名单过滤 → 低置信丢弃 → 递归合并进 layer6.profile → 原子落盘。
        禁止用 compress() 写 layer6.profile。
        """
        if not isinstance(extracted, dict):
            return self.get_profile()

        source = moment_id or "llm_inference"
        ts = timestamp or _now_iso()

        profile = _migrate_profile(self.memory["layer6"]["profile"])
        for field in PROFILE_FIELDS:
            node = extracted.get(field)
            if node is None or node in ([], {}):
                continue
            profile[field] = self._merge_profile_node(profile.get(field), node, source, ts)

        # 字段路径默认衰减策略兜底（LLM 未标注 decay_type 时，按设计说明书 §5.3 补默认）
        profile = _assign_default_decay(profile)
        # 按 decay_type 重算有效置信度 + 陈旧标记（三类衰减曲线，见 §5）
        profile = _apply_decay_and_stale(profile)

        self.memory["layer6"]["profile"] = profile
        self.memory["layer6"]["last_updated"] = _now_iso()
        self._save()
        return profile

    def get_profile(self) -> Dict:
        """返回完整 layer6 画像（含 stale 项与冲突历史，供页面审计）。"""
        return self.memory["layer6"]["profile"]

    def get_profile_for_retrieval(self, now: Optional[str] = None) -> Dict:
        """Return only current, non-stale profile facts for reasoning prompts."""
        return _active_profile_snapshot(
            self.memory.get("layer6", {}).get("profile") or {}, now=now
        )

    # ------------------------------------------------------------------
    # Operation 3 : QUERY  (called by Part2 before need analysis)
    # ------------------------------------------------------------------

    def _query_tokens(self, *texts: str) -> list:
        """查询词 → tokens：中英混合分词，不实时调用 LLM（对齐 §9.3 硬约束）。

        统一分词策略：tokenize 同时处理中文（单字）与英文（单词）——
        中文查询词按单字命中 moment 中文原文，英文查询词按单词命中写入时
        预翻译的 normalized。检索同步链路零 LLM 调用。
        """
        tokens = []
        for text in texts:
            if text:
                tokens += tokenize(str(text))
        return tokens

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search the Layer 7 archive with stopword filtering and BM25 ranking."""
        return [row["moment"] for row in self.query_ranked(query_text, top_k)]

    def query_ranked(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Return active Layer 7 memories with their field-aware BM25 scores."""
        if top_k <= 0:
            return []
        revision = self.memory.get("metadata", {}).get("semantic_index_revision", 0)
        cache = self._lexical_search_cache
        if cache is None or cache.get("revision") != revision:
            documents = [
                self._retrieval_group_moment(moment)
                for moment in self.memory["layer7"]["moments"].values()
                if not moment.get("merged_into")
            ]
            cache = _build_memory_search_index(documents)
            cache["revision"] = revision
            cache["all_documents"] = documents
            self._lexical_search_cache = cache
        documents = cache["all_documents"]
        lexical = _rank_memory_search_index(query_text, cache)
        if self._semantic_index_enabled:
            try:
                from semantic_retrieval import PersistentSemanticIndex, loaded_encoder
                encoder = loaded_encoder()
                if encoder is not None:
                    archive = self.memory.get("layer7", {}).get("moments", {})
                    index = PersistentSemanticIndex(self.memory_dir)
                    revision = self.memory.get("metadata", {}).get("semantic_index_revision", 0)
                    if index.ready(archive, encoder, revision=revision):
                        depth = min(len(documents), max(50, top_k * 10))
                        semantic = index.search(query_text, encoder, top_k=depth)
                        lexical_ids = [str(moment.get("id") or "") for _, moment in lexical[:depth]]
                        fused = {}
                        for ranked_ids in (lexical_ids, [row["moment_id"] for row in semantic]):
                            for rank, moment_id in enumerate(ranked_ids, 1):
                                if moment_id:
                                    fused[moment_id] = fused.get(moment_id, 0.0) + 1.0 / (60 + rank)
                        by_id = {str(moment.get("id") or ""): moment for moment in documents}
                        ordered = sorted(fused, key=lambda mid: (-fused[mid], mid))
                        return [
                            {"score": round(fused[mid], 6), "moment": by_id[mid],
                             "retrieval_method": "bm25_gte_rrf"}
                            for mid in ordered[:top_k] if mid in by_id
                        ]
            except Exception as exc:
                # A stale/missing/corrupt sidecar must not block ordinary retrieval.
                print(f"[semantic-index] query fallback to BM25: {exc}", flush=True)
        return [
            {"score": round(score, 6), "moment": moment,
             "retrieval_method": "bm25"}
            for score, moment in lexical[:top_k]
        ]

    def _retrieval_group_moment(self, moment: Dict) -> Dict:
        """Build a temporary search view containing the source evidence."""
        result = copy.deepcopy(moment)
        archive = self.memory.get("layer7", {}).get("moments", {})
        source_records = []
        for source_id in moment.get("merged_sources", []):
            source = archive.get(source_id)
            if not source or source.get("merged_into") != moment.get("id"):
                continue
            source_records.append({
                key: copy.deepcopy(source.get(key))
                for key in ("id", "timestamp", "scene", "user_action",
                            "interaction_details", "person_observations",
                            "temporal_facts", "object_observations", "source_time", "people",
                            "location", "activity", "objects", "environments",
                            "source_evidence", "multimodal", "highlighted")
                if source.get(key) is not None
            })
        normalize_object_transition_facts(result)
        result["object_observations"] = object_visual_observations(result)
        result["_merged_sources_records"] = source_records
        result["_merged_search_text"] = " ".join(
            json.dumps(row, ensure_ascii=False) for row in source_records
        )
        result["_group_highlighted"] = bool(
            moment.get("highlighted")
            or any(row.get("highlighted") for row in source_records)
        )
        return result

    def query_summary_layers(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Search L4-L6 summaries and return explicit source lineage."""
        if top_k <= 0 or not _retrieval_tokens(str(query_text or "")[:1200]):
            return []
        query_tokens = set(_retrieval_tokens(str(query_text or "")[:1200]))
        documents, originals = [], {}

        def active_profile_facts(node, path=""):
            if isinstance(node, dict) and "value" in node:
                if node.get("stale") or not node.get("value"):
                    return []
                return [f"{path}: {node['value']}" if path else str(node["value"])]
            if isinstance(node, dict):
                result = []
                for key, value in node.items():
                    result.extend(active_profile_facts(value, f"{path}.{key}" if path else key))
                return result
            if isinstance(node, list):
                return [fact for value in node for fact in active_profile_facts(value, path)]
            return []

        for layer_no in (4, 5, 6):
            layer = self.memory.get(f"layer{layer_no}", {})
            if not isinstance(layer, dict):
                continue
            document = {key: value for key, value in layer.items() if key not in {
                "source_moment_ids", "source_link_coverage", "source_moment_count",
                "last_updated", "profile",
            }}
            profile_facts = active_profile_facts(layer.get("profile") or {}) if layer_no == 6 else []
            matched_facts = [fact for fact in profile_facts
                             if query_tokens & set(_retrieval_tokens(fact))]
            if matched_facts:
                document["profile_facts"] = matched_facts
            document["id"] = f"layer{layer_no}_summary"
            documents.append(document)
            originals[document["id"]] = (layer_no, layer, matched_facts)

        rows = []
        for score, document in _rank_memory_docs(query_text, documents)[:top_k]:
            layer_no, layer, matched_facts = originals[document["id"]]
            summary = str(layer.get("summary") or "").strip()
            if layer_no == 6 and matched_facts:
                summary = (summary + "\n" if summary else "") + "相关稳定事实：" + "；".join(matched_facts[:12])
            if not summary:
                keys = ("current_tasks", "life_trajectory") if layer_no == 4 else ("key_events", "long_term_patterns")
                summary = "；".join(str(value) for key in keys
                                    for value in (layer.get(key) if isinstance(layer.get(key), list)
                                                  else [layer.get(key)]) if value)
            if not summary:
                continue
            source_ids = list(layer.get("source_moment_ids") or [])
            rows.append({
                "layer": layer_no, "score": round(score, 4), "summary": summary[:1600],
                "source_moment_ids": source_ids[:40],
                "source_moment_count": max(int(layer.get("source_moment_count") or 0), len(source_ids)),
                "source_link_coverage": layer.get("source_link_coverage", "unknown"),
            })
        return rows

    def query_hierarchical(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Rank unique recent/archive episodes together; summaries supplement source records."""
        query_text = str(query_text or "")[:1200]
        if top_k <= 0 or not _retrieval_tokens(query_text):
            return []

        memberships = {}
        window_only = {}
        for layer_no in (1, 2, 3):
            for moment in self._sorted_layer_moments(f"layer{layer_no}"):
                moment_id = str(moment.get("id") or "")
                if not moment_id or moment.get("merged_into"):
                    continue
                memberships.setdefault(moment_id, []).append(layer_no)
                window_only.setdefault(moment_id, moment)

        archive = self.memory.get("layer7", {}).get("moments", {})
        for moment_id, moment in archive.items() if isinstance(archive, dict) else []:
            if not isinstance(moment, dict) or moment.get("merged_into"):
                continue
            memberships.setdefault(str(moment_id), []).append(7)
            window_only.setdefault(str(moment_id), self._retrieval_group_moment(moment))

        # query_ranked preserves the optional BM25 + semantic-index fusion.
        archive_rows = self.query_ranked(query_text, top_k=max(50, top_k * 10))
        ranked_moments = []
        ranked_ids = set()
        for row in archive_rows:
            moment = row.get("moment") or {}
            moment_id = str(moment.get("id") or "")
            if not moment_id or moment_id in ranked_ids:
                continue
            ranked_ids.add(moment_id)
            ranked_moments.append((float(row.get("score") or 0), moment))

        # Preserve any valid rolling-window row missing from the canonical archive.
        orphan_docs = [moment for mid, moment in window_only.items()
                       if mid not in archive and mid not in ranked_ids]
        ranked_moments.extend(_rank_memory_docs(query_text, orphan_docs))

        results = []
        for score, moment in ranked_moments:
            moment_id = str(moment.get("id") or "")
            layers = sorted(set(memberships.get(moment_id, [])))
            display_layer = next((number for number in (1, 2, 3) if number in layers), 7)
            results.append({"layer": display_layer, "matched_layers": layers,
                            "score": round(float(score), 6), "moment": moment})
            if len(results) >= top_k:
                break

        if len(results) < top_k:
            for row in self.query_summary_layers(query_text, top_k=top_k - len(results)):
                moment = {"id": f"layer{row['layer']}_summary", "summary": row["summary"],
                          "source_moment_ids": row["source_moment_ids"],
                          "source_moment_count": row["source_moment_count"],
                          "source_link_coverage": row["source_link_coverage"]}
                results.append({"layer": row["layer"], "matched_layers": [row["layer"]],
                                "score": row["score"], "moment": moment})
        return results[:top_k]

    def _sorted_layer_moments(self, layer_key: str) -> list:
        """规范（09-22 层级）：层内记忆按时间远近顺序排列，返回最近优先（时间倒序）的列表。"""
        items = self.memory.get(layer_key, [])
        if not isinstance(items, list):
            return []
        return sorted(items, key=lambda m: m.get("timestamp", ""), reverse=True)

    def _sorted_index_tags(self, index_key: str) -> list:
        """按规范（09-22 分类排序规则）返回 layer7 索引的 tag 列表。

        - 从近到远（people / locations / time_nodes）：最近出现过的 tag 在前
        - 从高频到低频（activity_events / environments / objects）：moment 数最多的 tag 在前
        """
        index = self.memory["layer7"].get(index_key, {})
        if not index:
            return []
        moments = self.memory["layer7"]["moments"]

        if index_key in INDEX_NEAR_FAR:
            def latest(tag: str) -> str:
                ids = index.get(tag, [])
                ts = [moments[i].get("timestamp", "") for i in ids if i in moments]
                return max(ts) if ts else ""
            return sorted(index.keys(), key=latest, reverse=True)
        return sorted(index.keys(), key=lambda t: len(index.get(t, [])), reverse=True)

    # ------------------------------------------------------------------
    # Operation 4 : RETRIEVE  (called by Part2 / consolidation)
    # ------------------------------------------------------------------

    def retrieve(self, people: List[str] = None, location: str = None,
                 activity: str = None, layer: int = None, top_k: int = 5,
                 query_text: str = None, include_summaries: bool = True,
                 exclude_ids: set = None,
                 include_source_evidence: bool = False) -> str:
        """按标签和当前查询检索 L7，并返回完整的 Part1–4 事实。"""
        if top_k <= 0:
            return "No relevant memory found."
        parts = []
        if include_summaries:
            for lnum, lkey in [(4, "layer4"), (5, "layer5"), (6, "layer6")]:
                if layer and layer != lnum:
                    continue
                summary = self.memory.get(lkey, {}).get("summary")
                if summary:
                    parts.append(f"[Layer {lnum}] {summary}")
                if lnum == 4:
                    plans = self.memory.get(lkey, {}).get("stated_plans") or []
                    for plan in plans[:8] if isinstance(plans, list) else []:
                        if not isinstance(plan, dict):
                            continue
                        source = str(plan.get("source_moment_id") or "")
                        when = plan.get("source_time") or {}
                        scope = plan.get("time_scope") or ""
                        description = "; ".join(str(value) for value in (
                            plan.get("actor"), plan.get("value"), scope,
                            when.get("day_label"), when.get("time_of_day"), source)
                            if value)
                        if description:
                            parts.append(f"[Layer 4 明确说出的计划（尚不代表已完成）] {description}")

        query_text = str(query_text or "")[:1200]
        query_tokens = set(self._query_tokens(
            query_text, *(str(x) for x in (people or [])),
            str(location or ""), str(activity or "")))
        excluded = set(exclude_ids or ())
        scored = []
        for moment_id, stored in self.memory["layer7"]["moments"].items():
            if (moment_id in excluded or stored.get("stale")
                    or stored.get("merged_into")):
                continue
            moment = self._retrieval_group_moment(stored)
            searchable = json.dumps({key: value for key, value in moment.items()
                                     if key not in {"source_provenance", "multimodal", "needs", "solutions"}},
                                    ensure_ascii=False).lower()
            score = sum(1 for token in query_tokens if token in searchable)
            if score:
                scored.append((score, bool(moment.get("highlighted")
                                           or moment.get("_group_highlighted")),
                               moment.get("timestamp", ""), moment))

        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        for _, _, _, moment in scored[:top_k]:
            def values(items):
                if not isinstance(items, list):
                    return str(items or "")
                rendered = []
                for item in items:
                    if isinstance(item, dict):
                        value = (item.get("need") or item.get("solution")
                                 or item.get("value") or item.get("action"))
                    else:
                        value = item
                    if value:
                        rendered.append(str(value))
                return "；".join(rendered)

            feedback = moment.get("feedback") or {}
            corrections = feedback.get("corrections") or {}
            feedback_text = "；".join(
                f"{key}={value}" for key, value in corrections.items()
            ) if isinstance(corrections, dict) else ""
            if feedback.get("confirmed"):
                feedback_text = (feedback_text + "；" if feedback_text else "") + "用户已确认"
            if moment.get("highlighted") and not feedback_text:
                feedback_text = "已标记为重点/重复记忆"

            line = f"[Layer7/{moment.get('id','')}] {moment.get('timestamp','')[:19]}"
            source_time = moment.get("source_time") or {}
            if source_time:
                source_clock = " · ".join(str(value) for value in (
                    source_time.get("day_label"), source_time.get("time_of_day")) if value)
                if source_clock:
                    line += f" | 来源时间: {source_clock}"
            source_id = (moment.get("source_provenance") or {}).get("source_segment_id")
            if source_id:
                line += f" | 来源片段: {str(source_id)[:128]}"
            fields = [
                ("场景", moment.get("scene")),
                ("动作", moment.get("user_action")),
                ("人物交互与事件顺序", moment.get("interaction_details")),
                ("人物", values(moment.get("people"))),
                ("地点", moment.get("location")),
                ("活动", moment.get("activity")),
                ("物品", json.dumps(moment.get("objects") or [], ensure_ascii=False)),
                ("物品观察", json.dumps(moment.get("object_observations") or [], ensure_ascii=False)),
                ("环境", json.dumps(moment.get("environments") or [], ensure_ascii=False)),
                ("人物观察", json.dumps(moment.get("person_observations") or [], ensure_ascii=False)),
                ("时间事实", json.dumps(moment.get("temporal_facts") or [], ensure_ascii=False)),
                ("来源时间", json.dumps(moment.get("source_time") or {}, ensure_ascii=False)),
                ("反馈修正", feedback_text),
            ]
            if include_source_evidence and moment.get("source_evidence"):
                evidence = str(moment["source_evidence"]).replace(chr(10), " ").strip()
                line += f" | Transcript evidence: {evidence[:1200]}"
            for label, value in fields:
                if value:
                    limit = 3200 if label == "时间事实" else 1800 if label == "物品观察" else 1600 if label == "人物观察" else 500 if label in {"人物交互与事件顺序", "来源时间"} else 300 if label in {"环境", "物品"} else 180
                    line += f" | {label}: {str(value).replace(chr(10), ' ').strip()[:limit]}"
            parts.append(line)
            for source in moment.get("_merged_sources_records", []):
                source_line = f"  └ 合并来源 [{source.get('id','')}] {str(source.get('timestamp',''))[:19]}"
                for label, key in (("Scene", "scene"), ("Action", "user_action"),
                                   ("People", "people"), ("Location", "location"),
                                   ("Activity", "activity")):
                    value = source.get(key)
                    if value:
                        source_line += f" | {label}: {str(value).replace(chr(10), ' ').strip()[:180]}"
                media = source.get("multimodal") or {}
                if media:
                    source_line += " | retained media: " + ("visual " if (media.get("visual") or {}).get("asset") else "") + ("audio" if (media.get("audio") or {}).get("asset") else "")
                parts.append(source_line)

        return chr(10).join(parts) if parts else "No relevant memory found."

    # Operation 5 : COMPRESS  (called AFTER Part2+3, never during)
    # ------------------------------------------------------------------

    def compress(self, llm_summary: Dict):
        """
        Update layers 4, 5, 6 with LLM-generated compressed summaries.

        llm_summary format:
        {
          "layer4": {"summary": "...", "current_tasks": [...], "life_trajectory": "..."},
          "layer5": {"summary": "...", "key_events": [...], "long_term_patterns": "..."},
          "layer6": {"summary": "...", "profile": {...}}
        }
        """
        for lkey in ["layer4", "layer5", "layer6"]:
            if lkey in llm_summary:
                updates = llm_summary[lkey]
                layer = self.memory[lkey]
                for k, v in updates.items():
                    # layer6.profile 是画像，必须走 update_profile()（04 文档 2.5 职责边界）。
                    # compress() 禁止直接写，否则 profile 内部列表会被 dict.update 浅覆盖丢历史
                    # （09-03 已实证）。
                    if lkey == "layer6" and k == "profile":
                        print("⚠️  compress() 忽略 layer6.profile —— 画像请用 update_profile() 写入")
                        continue
                    if k == "source_moment_ids":
                        available = self.memory.get("layer7", {}).get("moments", {})
                        valid_ids = list(dict.fromkeys(
                            str(mid) for mid in (v if isinstance(v, list) else [])
                            if str(mid) in available
                        ))
                        if lkey == "layer4":
                            layer[k] = valid_ids
                        else:
                            layer[k] = list(dict.fromkeys(list(layer.get(k) or []) + valid_ids))
                        continue
                    if k == "source_link_coverage":
                        if v in {"complete", "partial", "unknown"}:
                            layer[k] = v
                        continue
                    if isinstance(v, dict) and isinstance(layer.get(k), dict):
                        layer[k].update(v)
                    elif isinstance(v, list) and isinstance(layer.get(k), list):
                        # 列表合并去重（设计说明书 §4.4）：json.dumps + sort_keys 精确比较，
                        # 支持 str 与嵌套 dict 元素（str(item) 对 dict 的顺序不稳定）。
                        existing = {json.dumps(x, ensure_ascii=False, sort_keys=True)
                                    for x in layer[k]}
                        for item in v:
                            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                            if key not in existing:
                                layer[k].append(item)
                                existing.add(key)
                    else:
                        # 空值保护：空字符串/None 不覆盖已有值（防 LLM 返回空摘要丢历史）
                        if v is not None and v != "":
                            layer[k] = v
                layer["last_updated"] = _now_iso()
                layer["source_moment_count"] = len(layer.get("source_moment_ids") or [])

        self.memory["metadata"]["last_consolidation"] = _now_iso()
        self._save()

    # ------------------------------------------------------------------
    # Operation 6 : SORT  (called together with compress)
    # ------------------------------------------------------------------

    def sort(self, sort_analysis: Dict):
        """layer7 索引归类（设计说明书 §4.4 定稿）。

        sort_analysis format:
        {
          "people": {"姥姥": ["m_xxx", ...]},
          "locations": {"family_courtyard": ["m_xxx", ...]},
          "activity_events": {"chinese_new_year": ["m_xxx", ...]},
          "time_nodes": {"2024-02-10": ["m_xxx", ...]}
        }
        规则：索引键过校验（防 G7 污染）；过滤悬挂 moment_id（不存在的 id 不写索引）；
        同键 moment_id 集合去重合并。
        """
        moments = self.memory["layer7"]["moments"]
        for index_key in LAYER7_INDICES:
            if index_key not in sort_analysis:
                continue
            for tag, ids in sort_analysis[index_key].items():
                if not _is_valid_index_tag(index_key, tag):
                    continue
                valid_ids = [i for i in ids if i in moments]  # 过滤悬挂引用
                # Keep first-seen order stable across runs; a set here made
                # index snapshots change order nondeterministically.
                merged = []
                seen = set()
                for moment_id in list(self.memory["layer7"][index_key].get(tag, [])) + valid_ids:
                    if moment_id not in seen:
                        seen.add(moment_id)
                        merged.append(moment_id)
                self.memory["layer7"][index_key][tag] = merged
        self._save()

    # ------------------------------------------------------------------
    # Operation 7 : COMBINE  (called together with compress)
    # ------------------------------------------------------------------

    def combine(self, canonical_map: Dict):
        """合并 layer7 索引近义旧键到规范键（设计说明书 §4.4 定稿）。

        canonical_map format:
        {
          "people": {"old fuzzy name": "canonical_name"},
          "locations": {"old tag": "canonical tag"},
          "activity_events": {"old tag": "canonical tag"}
        }

        - canonical 过校验（防 G7 污染）；old_key 不存在或等于 canonical 则跳过（幂等）。
        - 按稳定顺序合并 moment_id 并同步更新 moments 主存储里的引用字段；列表字段替换后去重。
        - 在受影响的 moment 上保留 canonicalization_history，记录来源标签和规范标签；不删除原始 moment。
        - 返回本次实际归并明细，供操作日志展示。
        """
        # 索引 → (moments 引用字段, 字段类型[str 单值 / list 列表])
        REF_FIELD = {
            "locations": ("location", "str"),
            "people": ("people", "list"),
            "activity_events": ("activity", "str"),
            "environments": ("environments", "list"),
            "objects": ("objects", "list"),
        }
        report = {"aliases": [], "updated_moments": 0}
        updated_moment_ids = set()
        moments = self.memory["layer7"]["moments"]
        for index_key, mapping in canonical_map.items():
            if index_key not in self.memory["layer7"] or not isinstance(mapping, dict):
                continue
            ref = REF_FIELD.get(index_key)
            index = self.memory["layer7"][index_key]
            for old_key, canonical in mapping.items():
                if not _is_valid_index_tag(index_key, canonical):
                    continue
                if old_key not in index or old_key == canonical:
                    continue
                ids = list(dict.fromkeys(index.pop(old_key) or []))
                canonical_ids_before = list(index.get(canonical, []))
                canonical_id_set = set(canonical_ids_before)
                alias_id_set = set(ids)
                overlap_count = len(canonical_id_set & alias_id_set)
                merged_ids = []
                seen_ids = set()
                for mid in canonical_ids_before + list(ids):
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        merged_ids.append(mid)
                index[canonical] = merged_ids
                alias_report = {
                    "dimension": index_key,
                    "from": old_key,
                    "to": canonical,
                    "affected_moments": 0,
                    "sample_moment_ids": [],
                    "overlap_moments": overlap_count,
                    "field_updates": 0,
                    "list_duplicates_removed": 0,
                }
                # 同步更新 moments 主存储引用字段
                for mid in ids:
                    m = moments.get(mid)
                    if not m:
                        continue
                    alias_report["affected_moments"] += 1
                    if len(alias_report["sample_moment_ids"]) < 5:
                        alias_report["sample_moment_ids"].append(mid)
                    changed = False
                    history = m.get("canonicalization_history")
                    if not isinstance(history, list):
                        history = []
                        m["canonicalization_history"] = history
                    history.append({
                        "dimension": index_key,
                        "from": old_key,
                        "to": canonical,
                        "at": _now_iso(),
                    })
                    changed = True
                    if ref:
                        field, kind = ref
                        if kind == "str":
                            if m.get(field) == old_key:
                                m[field] = canonical
                                alias_report["field_updates"] += 1
                                changed = True
                        elif kind == "list":
                            if isinstance(m.get(field), list) and old_key in m[field]:
                                replaced = [canonical if x == old_key else x for x in m[field]]
                                after_values = list(dict.fromkeys(replaced))
                                alias_report["field_updates"] += 1
                                alias_report["list_duplicates_removed"] += len(replaced) - len(after_values)
                                m[field] = after_values
                                changed = True
                    if changed:
                        updated_moment_ids.add(mid)
                        self._sync_window_moment(mid)
                report["aliases"].append(alias_report)
        report["updated_moments"] = len(updated_moment_ids)
        self._save()
        return report

    # ------------------------------------------------------------------
    # Operation 8 : EVENT MERGE (explicit, audited source retention)
    # ------------------------------------------------------------------

    def find_merge_candidates(self, limit: int = 30,
                              max_time_delta_s: int = 120) -> List[Dict]:
        """Create a review queue; candidates are never merged automatically."""
        archive = self.memory.get("layer7", {}).get("moments", {})
        active = [(mid, item) for mid, item in archive.items()
                  if isinstance(item, dict) and not item.get("stale")
                  and not item.get("merged_into")]
        by_entity_time, by_source = defaultdict(list), defaultdict(list)
        pairs, parsed_times = set(), {}
        bucket_seconds = 300

        def entities(item):
            values = set()
            for field in ("people", "objects", "environments"):
                raw = item.get(field) or []
                if isinstance(raw, str):
                    raw = [raw]
                values.update(field[:1] + ":" + x.casefold().strip()
                              for x in raw if isinstance(x, str) and x.strip())
            for field in ("location", "activity"):
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    values.add(field[:1] + ":" + value.casefold().strip())
            return values

        for mid, item in active:
            try:
                event_time = datetime.fromisoformat(str(item.get("timestamp") or ""))
                parsed_times[mid] = event_time
            except (TypeError, ValueError):
                continue
            bucket = int(event_time.timestamp() // bucket_seconds)
            for tag in entities(item):
                for nearby in (bucket - 1, bucket, bucket + 1):
                    pairs.update(tuple(sorted((mid, other)))
                                 for other in by_entity_time[(nearby, tag)])
                by_entity_time[(bucket, tag)].append(mid)
            media = item.get("multimodal") or {}
            source_key = media.get("source_id") if isinstance(media, dict) else None
            if source_key:
                pairs.update(tuple(sorted((mid, other))) for other in by_source[source_key])
                by_source[source_key].append(mid)

        def norm(value):
            return re.sub(r"\W+", " ", str(value or "").casefold()).strip()

        normalized_text = {
            mid: norm(str(item.get("scene") or "") + " " +
                      str(item.get("user_action") or ""))[:1200]
            for mid, item in active
        }

        def text_similarity(left_id, right_id, threshold_only=False):
            a, b = normalized_text[left_id], normalized_text[right_id]
            if not a or not b:
                return 0.0
            if a == b:
                return 1.0
            matcher = SequenceMatcher(None, a, b)
            # quick_ratio is an upper bound. For text-only candidates, skip
            # exact ratio if the combined Scene + Action cannot reach threshold.
            if threshold_only and (
                matcher.real_quick_ratio() < 0.92 or matcher.quick_ratio() < 0.92
            ):
                return 0.0
            return matcher.ratio()

        def media_overlap(left, right):
            a, b = left.get("multimodal") or {}, right.get("multimodal") or {}
            if not isinstance(a, dict) or not isinstance(b, dict):
                return 0.0
            if not a.get("source_id") or a.get("source_id") != b.get("source_id"):
                return 0.0
            try:
                a0, a1 = float(a.get("start_s")), float(a.get("end_s"))
                b0, b1 = float(b.get("start_s")), float(b.get("end_s"))
            except (TypeError, ValueError):
                return 0.0
            shorter = min(max(0.0, a1 - a0), max(0.0, b1 - b0))
            overlap = max(0.0, min(a1, b1) - max(a0, b0))
            return overlap / shorter if shorter else 0.0

        def brief(mid, item):
            media = item.get("multimodal") or {}
            return {
                "id": mid, "timestamp": item.get("timestamp", ""),
                "scene": str(item.get("scene") or "")[:240],
                "user_action": str(item.get("user_action") or "")[:300],
                "people": item.get("people") or [],
                "location": item.get("location") or "",
                "activity": item.get("activity") or "",
                "objects": item.get("objects") or [],
                "has_visual": bool((media.get("visual") or {}).get("asset")),
                "has_audio": bool((media.get("audio") or {}).get("asset")),
            }

        try:
            from multimodal_memory import compare_multimodal
        except ImportError:
            compare_multimodal = lambda left, right: {"score": None, "modalities": {}}

        candidates = []
        for left_id, right_id in pairs:
            left, right = archive[left_id], archive[right_id]
            delta = abs((parsed_times[left_id] - parsed_times[right_id]).total_seconds())
            shared = sorted(entities(left) & entities(right))
            shared_people_objects = [x for x in shared if x.startswith(("p:", "o:"))]
            overlap = media_overlap(left, right)
            media_candidate = overlap >= 0.80
            if not media_candidate and delta > max_time_delta_s:
                continue
            media_similarity = compare_multimodal(left, right)
            modalities = media_similarity.get("modalities") or {}
            similarity = text_similarity(left_id, right_id,
                                         threshold_only=not media_candidate)
            text_candidate = (
                delta <= max_time_delta_s
                and similarity >= 0.92
                and bool(shared_people_objects)
            )
            if not (media_candidate or text_candidate):
                continue
            target_id, source_id = sorted(
                (left_id, right_id), key=lambda mid: (parsed_times[mid], mid))
            candidates.append({
                "target_id": target_id, "source_id": source_id,
                "similarity": round(max(similarity, overlap), 4),
                "text_similarity": round(similarity, 4),
                "media_overlap": round(overlap, 4),
                "multimodal_similarity": modalities,
                "time_delta_s": round(delta, 2),
                "shared_entities": shared,
                "reason": (
                    "同一视频来源且片段高度重叠" if media_candidate else
                    "时间接近、Scene 与 Action 合并文本高度相似且共享人物或物品"
                ),
                "target": brief(target_id, archive[target_id]),
                "source": brief(source_id, archive[source_id]),
            })
        candidates.sort(key=lambda row: (
            -row["similarity"], row["time_delta_s"], row["target_id"], row["source_id"]))
        return candidates[:max(0, int(limit))]

    def merge_moments(self, source_id: str, target_id: str,
                      confirmed_duplicate: bool = False,
                      reason: str = "manual duplicate review") -> Dict:
        """Explicit merge: retain source L7/media, union lists, audit conflicts."""
        if not confirmed_duplicate:
            return {"ok": False, "status": "confirmation_required",
                    "message": "请先人工确认这两条记录是同一事件。"}
        if not source_id or not target_id or source_id == target_id:
            return {"ok": False, "status": "invalid_ids",
                    "message": "source_id 与 target_id 必须是不同的有效记忆 ID。"}
        archive = self.memory.get("layer7", {}).get("moments", {})
        source, target = archive.get(source_id), archive.get(target_id)
        if not source or not target:
            return {"ok": False, "status": "not_found",
                    "message": "来源或规范记忆不存在。"}
        if target.get("merged_into"):
            return {"ok": False, "status": "target_is_alias",
                    "message": "规范目标本身已归并；请改选当前规范记录。"}
        if source.get("merged_into"):
            if source.get("merged_into") == target_id:
                return {"ok": True, "status": "already_merged",
                        "source_id": source_id, "target_id": target_id,
                        "merged_sources": target.get("merged_sources", [])}
            return {"ok": False, "status": "source_is_alias",
                    "message": "来源已归并到其他记忆，不能建立新的归并关系。"}

        child_ids = [
            mid for mid in source.get("merged_sources", [])
            if mid in archive and archive[mid].get("merged_into") == source_id]
        alias_ids, now = [source_id] + child_ids, _now_iso()
        list_fields = ("people", "objects", "environments", "needs", "solutions")
        target_fields_before = {
            field: copy.deepcopy(target.get(field)) for field in list_fields
        }
        source_index_tags = {
            alias_id: {
                index_key: [tag for tag, ids in
                            self.memory["layer7"].get(index_key, {}).items()
                            if alias_id in ids]
                for index_key in LAYER7_INDICES
            }
            for alias_id in alias_ids
        }
        target_index_tags = {
            index_key: [tag for tag, ids in
                        self.memory["layer7"].get(index_key, {}).items()
                        if target_id in ids]
            for index_key in LAYER7_INDICES
        }
        source_window_membership = {
            alias_id: {
                layer_key: [i for i, row in enumerate(self.memory.get(layer_key, []))
                            if row.get("id") == alias_id]
                for layer_key in ("layer1", "layer2", "layer3")
            }
            for alias_id in alias_ids
        }
        target_window_membership = {
            layer_key: [i for i, row in enumerate(self.memory.get(layer_key, []))
                        if row.get("id") == target_id]
            for layer_key in ("layer1", "layer2", "layer3")
        }
        source_merge_state_before = {
            alias_id: {
                "merged_into": copy.deepcopy(archive[alias_id].get("merged_into")),
                "merge_event": copy.deepcopy(archive[alias_id].get("merge_event")),
            }
            for alias_id in alias_ids
        }
        target_merged_sources_before = copy.deepcopy(
            target.get("merged_sources") if isinstance(target.get("merged_sources"), list) else [])
        merged_fields = []
        for field in list_fields:
            values = []
            for alias_id in alias_ids:
                current = archive.get(alias_id, {}).get(field)
                if isinstance(current, list):
                    values.extend(current)
            old_values = target.get(field) if isinstance(target.get(field), list) else []
            combined, seen = [], set()
            for value in list(old_values) + values:
                key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    combined.append(value)
            if combined != old_values:
                target[field] = combined
                merged_fields.append(field)

        scalar_conflicts = []
        for alias_id in alias_ids:
            row = archive.get(alias_id, {})
            for field in ("scene", "user_action", "location", "activity", "timestamp"):
                a, b = target.get(field), row.get(field)
                if a and b and a != b:
                    scalar_conflicts.append({
                        "source_id": alias_id, "field": field,
                        "target_value": a, "source_value": b,
                    })

        prior = target.get("merged_sources")
        if not isinstance(prior, list):
            prior = []
        target["merged_sources"] = list(dict.fromkeys(prior + alias_ids))
        history = target.setdefault("merge_history", [])
        if not isinstance(history, list):
            history = []
            target["merge_history"] = history
        media_preserved = []
        for alias_id in alias_ids:
            media = archive.get(alias_id, {}).get("multimodal") or {}
            media_preserved.append({
                "id": alias_id,
                "visual": bool((media.get("visual") or {}).get("asset")),
                "audio": bool((media.get("audio") or {}).get("asset")),
            })
        history.append({
            "source_id": source_id, "source_ids": alias_ids,
            "target_id": target_id, "at": now, "reason": str(reason)[:240],
            "merged_fields": merged_fields,
            "target_fields_before": target_fields_before,
            "target_fields_after": {
                field: copy.deepcopy(target.get(field)) for field in list_fields
            },
            "source_index_tags_before": source_index_tags,
            "target_index_tags_before": target_index_tags,
            "source_window_membership_before": source_window_membership,
            "target_window_membership_before": target_window_membership,
            "source_merge_state_before": source_merge_state_before,
            "target_merged_sources_before": target_merged_sources_before,
            "target_merged_sources_after": copy.deepcopy(target["merged_sources"]),
            "scalar_conflicts": scalar_conflicts,
            "source_highlighted": any(
                bool(archive.get(mid, {}).get("highlighted")) for mid in alias_ids),
            "media_preserved": media_preserved,
        })
        for alias_id in alias_ids:
            alias = archive[alias_id]
            alias["merged_into"] = target_id
            alias["merge_event"] = {
                "target_id": target_id, "at": now, "reason": str(reason)[:240]}

        alias_set = set(alias_ids)
        for index_key in LAYER7_INDICES:
            index = self.memory["layer7"].get(index_key, {})
            for tag, ids in list(index.items()):
                if any(mid in alias_set for mid in ids):
                    rewritten = list(dict.fromkeys(
                        target_id if mid in alias_set else mid for mid in ids))
                    index[tag] = rewritten
        for layer_key in ("layer1", "layer2", "layer3"):
            rows = self.memory.get(layer_key, [])
            positions = [i for i,row in enumerate(rows) if row.get("id") in alias_set]
            if not positions:
                continue
            target_present = any(row.get("id") == target_id for row in rows)
            kept = [row for row in rows if row.get("id") not in alias_set]
            if not target_present:
                kept.insert(min(positions[0], len(kept)), target)
            self.memory[layer_key] = kept
        self._sync_window_moment(target_id)
        self._save()
        return {
            "ok": True, "status": "merged", "source_id": source_id,
            "source_ids": alias_ids, "target_id": target_id,
            "merged_fields": merged_fields, "scalar_conflicts": scalar_conflicts,
            "media_preserved": media_preserved,
            "merged_source_count": len(target["merged_sources"]), "at": now,
        }

    def unmerge_moments(self, source_id: str, target_id: str,
                        reason: str = "manual merge undo") -> Dict:
        """Undo the latest explicit event merge without deleting source records/assets.

        The operation refuses legacy merge entries without full window/index snapshots,
        edits made after a merge, and attempts to undo an older merge in a chain.
        """
        if not source_id or not target_id or source_id == target_id:
            return {"ok": False, "status": "invalid_ids",
                    "message": "source_id 与 target_id 必须是不同的有效记忆 ID。"}
        archive = self.memory.get("layer7", {}).get("moments", {})
        source, target = archive.get(source_id), archive.get(target_id)
        if not source or not target:
            return {"ok": False, "status": "not_found",
                    "message": "来源或规范记忆不存在。"}
        history = target.get("merge_history")
        if not isinstance(history, list) or not history:
            return {"ok": False, "status": "no_merge_history",
                    "message": "没有可撤销的归并记录。"}
        event = history[-1]
        if event.get("undone_at"):
            return {"ok": False, "status": "already_undone",
                    "message": "最近一次归并已经撤销。"}
        if event.get("source_id") != source_id or event.get("target_id") != target_id:
            return {"ok": False, "status": "not_latest_merge",
                    "message": "只能撤销该规范记录最近一次归并；请先撤销后续归并。"}

        required = ("source_ids", "target_fields_before", "target_fields_after",
                    "source_index_tags_before", "target_index_tags_before",
                    "source_window_membership_before", "target_window_membership_before",
                    "source_merge_state_before", "target_merged_sources_before",
                    "target_merged_sources_after")
        if any(key not in event for key in required):
            return {"ok": False, "status": "undo_snapshot_missing",
                    "message": "该归并缺少完整的撤销快照；为避免误改数据，未执行撤销。"}

        source_ids = list(dict.fromkeys(event.get("source_ids") or []))
        if source_id not in source_ids or any(mid not in archive for mid in source_ids):
            return {"ok": False, "status": "source_missing",
                    "message": "归并来源记录不完整；未执行撤销。"}
        list_fields = ("people", "objects", "environments", "needs", "solutions")
        current_target_fields = {field: copy.deepcopy(target.get(field)) for field in list_fields}
        if current_target_fields != event.get("target_fields_after"):
            return {"ok": False, "status": "target_changed",
                    "message": "归并后目标记忆字段已有变化；请先处理后续修改，未执行撤销。"}
        if target.get("merged_sources", []) != event.get("target_merged_sources_after"):
            return {"ok": False, "status": "merge_chain_changed",
                    "message": "归并链已有后续变化；请按最近归并顺序撤销。"}
        relations = event.get("source_merge_state_before") or {}
        for mid in source_ids:
            if mid not in relations or archive[mid].get("merged_into") != target_id:
                return {"ok": False, "status": "source_relation_changed",
                        "message": "来源归并关系已变化；未执行撤销。"}

        for field in list_fields:
            target[field] = copy.deepcopy(event["target_fields_before"].get(field))
        target["merged_sources"] = copy.deepcopy(event["target_merged_sources_before"])

        # Restore the source aliases' original relationships. A source may itself
        # have contained older aliases before it was merged into the target.
        for mid in source_ids:
            prior = relations[mid]
            alias = archive[mid]
            if prior.get("merged_into"):
                alias["merged_into"] = copy.deepcopy(prior["merged_into"])
            else:
                alias.pop("merged_into", None)
            if prior.get("merge_event") is not None:
                alias["merge_event"] = copy.deepcopy(prior["merge_event"])
            else:
                alias.pop("merge_event", None)

        # Rebuild only affected index memberships from the saved pre-merge tags.
        source_tags = event["source_index_tags_before"]
        target_tags = event["target_index_tags_before"]
        for index_key in LAYER7_INDICES:
            index = self.memory["layer7"].setdefault(index_key, {})
            source_tag_map = {
                mid: set((source_tags.get(mid) or {}).get(index_key) or [])
                for mid in source_ids
            }
            target_tag_set = set(target_tags.get(index_key) or [])
            affected = set(target_tag_set)
            affected.update(tag for tags in source_tag_map.values() for tag in tags)
            affected.update(tag for tag, ids in index.items()
                            if target_id in ids or any(mid in ids for mid in source_ids))
            for tag in affected:
                ids = [mid for mid in index.get(tag, [])
                       if mid not in source_ids and mid != target_id]
                if tag in target_tag_set:
                    ids.append(target_id)
                for mid in source_ids:
                    if tag in source_tag_map[mid]:
                        ids.append(mid)
                ids = list(dict.fromkeys(ids))
                if ids:
                    index[tag] = ids
                else:
                    index.pop(tag, None)

        # Restore exact L1-L3 membership and relative order using original positions.
        source_windows = event["source_window_membership_before"]
        target_windows = event["target_window_membership_before"]
        for layer_key in ("layer1", "layer2", "layer3"):
            rows = [row for row in self.memory.get(layer_key, [])
                    if row.get("id") not in source_ids and row.get("id") != target_id]
            restored = []
            for mid in source_ids:
                for position in (source_windows.get(mid) or {}).get(layer_key, []):
                    restored.append((int(position), mid, archive[mid]))
            for position in target_windows.get(layer_key, []):
                restored.append((int(position), target_id, target))
            for position, _mid, record in sorted(restored, key=lambda item: item[0]):
                rows.insert(max(0, min(position, len(rows))), record)
            self.memory[layer_key] = rows

        event["undone_at"] = _now_iso()
        event["undo_reason"] = str(reason)[:240]
        self._save()
        return {
            "ok": True, "status": "unmerged", "source_id": source_id,
            "source_ids": source_ids, "target_id": target_id,
            "restored_source_count": len(source_ids), "undone_at": event["undone_at"],
            "media_preserved": copy.deepcopy(event.get("media_preserved", [])),
        }

    # ------------------------------------------------------------------
    # Operation 9 : DELETE
    # ------------------------------------------------------------------

    def delete(self, moment_id: str = None, reason: str = "manual",
               force: bool = False) -> bool:
        """删除 moment（设计说明书 §4.6）。

        - 清理范围：moments 主存储 + 四个索引引用 + layer1/2/3 条目。
        - highlight 保护：highlighted=True 且非 force → 拒绝删除（返回 False）。
        - metadata.total_moments 不减（moment_id 依赖累计序号，防 ID 复用冲突）。
        - 返回：删除成功 True；moment 不存在或被保护拒绝 False。
        """
        if not moment_id:
            return False
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment:
            return False
        if moment.get("highlighted") and not force:
            print(f"⛔ 拒绝删除 highlighted moment {moment_id}（force=True 可强制）")
            return False

        del self.memory["layer7"]["moments"][moment_id]
        # Remove from indices
        for index_key in LAYER7_INDICES:
            index = self.memory["layer7"].get(index_key, {})
            for tag, ids in list(index.items()):
                if moment_id in ids:
                    ids.remove(moment_id)
                if not ids:
                    index.pop(tag, None)
        # Remove from raw layers
        for lkey in ["layer1", "layer2", "layer3"]:
            self.memory[lkey] = [m for m in self.memory[lkey]
                                 if m.get("id") != moment_id]
        self._save()
        if moment.get("multimodal"):
            from multimodal_memory import MultimodalAssetStore
            MultimodalAssetStore(self.memory_dir).release_unreferenced(
                self.memory["layer7"]["moments"])
        print(f"🗑️  Deleted moment {moment_id} ({reason})")
        return True

    def update_multimodal(self, moment_id: str, evidence: Dict) -> bool:
        """Replace media references while retaining the event and six indices."""
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment or not isinstance(evidence, dict) or evidence.get("schema") != 1:
            return False
        moment["multimodal"] = evidence
        self._sync_window_moment(moment_id)
        self._save()
        from multimodal_memory import MultimodalAssetStore
        MultimodalAssetStore(self.memory_dir).release_unreferenced(
            self.memory["layer7"]["moments"])
        return True

    def compact_multimodal_assets(self, budget_bytes: int = None) -> Dict:
        """Tier aged images and reclaim assets without removing text facts."""
        from multimodal_memory import MultimodalAssetStore
        result = MultimodalAssetStore(self.memory_dir).compact_to_budget(
            self.memory["layer7"]["moments"], budget_bytes=budget_bytes)
        changed_ids = result.pop("_changed_moment_ids", [])
        for mid in changed_ids:
            if mid in self.memory["layer7"]["moments"]:
                self._sync_window_moment(mid)
        if changed_ids:
            self._save()
        return result

    # ------------------------------------------------------------------
    # Operation 10 : HIGHLIGHT
    # ------------------------------------------------------------------

    def highlight(self, moment_id: str) -> bool:
        """标记 moment 为 confirmed-correct（设计说明书 §4.6）。

        效果：① 删除清扫中受保护（delete 默认拒删）；② v2：检索排序提权、
        画像更新高权重佐证。moment 不存在返回 False。
        """
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment:
            return False
        changed = not moment.get("highlighted", False)
        provenance_changed = (
            moment.get("highlight_source") != "highlight_api"
            or not moment.get("highlighted_at")
        )
        moment["highlighted"] = True
        if provenance_changed:
            moment["highlight_source"] = "highlight_api"
            moment["highlighted_at"] = _now_iso()
        changed = self._sync_window_moment(moment_id) or changed or provenance_changed
        if changed:
            self._save()
        return True

    # ------------------------------------------------------------------
    # Context helpers (called by pipeline)
    # ------------------------------------------------------------------

    def get_context_for_analysis(self, people: List[str] = None,
                                  location: str = None,
                                  activity: str = None,
                                  query_text: str = None,
                                  top_k: int = 5) -> str:
        """Return source-linked observed memories relevant to the current scene."""
        query_text = str(query_text or "")[:1200]
        top_k = max(0, int(top_k))
        lines = ["=== MEMORY CONTEXT ==="]
        hits = self.query_hierarchical(query_text, top_k=top_k) if query_text.strip() else []

        def add_moment(layer_no, moment, matched_layers=None):
            moment_id = str(moment.get("id") or "")
            if moment_id in self.memory.get("layer7", {}).get("moments", {}):
                moment = self._retrieval_group_moment(moment)
            layers = matched_layers or [layer_no]
            layer_label = ",".join(str(value) for value in layers)
            fields = [
                ("场景", moment.get("scene")), ("动作", moment.get("user_action")),
                ("人物", moment.get("people")), ("地点", moment.get("location")),
                ("活动", moment.get("activity")), ("物品", moment.get("objects")),
                ("环境", moment.get("environments")),
            ]
            line = f"[Layers {layer_label} · {str(moment.get('timestamp') or '')[:19]} {moment_id}]"
            for label, value in fields:
                if value:
                    line += f" | {label}: {str(value).replace(chr(10), ' ')[:240]}"
            for label, value, limit in (
                ("人物外观", moment.get("person_observations"), 1600),
                ("事件与状态事实", moment.get("temporal_facts"), 3200),
                ("物品观察", moment.get("object_observations") or object_visual_observations(moment), 1800),
                ("来源时间", moment.get("source_time"), 500),
            ):
                if value:
                    line += f" | {label}: {json.dumps(value, ensure_ascii=False)[:limit]}"
            if moment.get("source_evidence"):
                line += " | 语音来源: " + str(moment["source_evidence"]).replace(chr(10), " ")[:360]
            lines.append(line)
            for source in moment.get("_merged_sources_records", []):
                source_line = f"[合并来源 · {source.get('timestamp','')[:19]} {source.get('id','')}]"
                for label, key in (("场景", "scene"), ("动作", "user_action"), ("人物", "people"),
                                   ("地点", "location"), ("活动", "activity"),
                                   ("物品", "objects"), ("环境", "environments")):
                    value = source.get(key)
                    if value:
                        source_line += f" | {label}: {str(value).replace(chr(10), ' ')[:180]}"
                if source.get("multimodal"):
                    source_line += " | 多模态来源仍保留"
                lines.append(source_line)

        if hits:
            for hit in hits:
                layer_no = hit["layer"]
                moment = hit["moment"]
                if moment.get("summary"):
                    ids = ", ".join(moment.get("source_moment_ids") or [])
                    coverage = moment.get("source_link_coverage", "unknown")
                    lines.append(f"[Layer {layer_no} summary · 来源覆盖 {coverage} · {ids}] {moment['summary'][:1200]}")
                else:
                    add_moment(layer_no, moment, hit.get("matched_layers"))
        elif not query_text and top_k:
            seen = set()
            for layer_no in (1, 2, 3):
                for moment in self._sorted_layer_moments(f"layer{layer_no}"):
                    moment_id = str(moment.get("id") or "")
                    if not moment_id or moment_id in seen:
                        continue
                    seen.add(moment_id)
                    add_moment(layer_no, moment)
                    if len(seen) >= top_k:
                        break
                if len(seen) >= top_k:
                    break
            for layer_no in (4, 5, 6):
                layer = self.memory.get(f"layer{layer_no}", {})
                if layer.get("summary"):
                    ids = ", ".join(layer.get("source_moment_ids") or [])
                    coverage = layer.get("source_link_coverage", "unknown")
                    lines.append(f"[Layer {layer_no} summary · 来源覆盖 {coverage} · {ids}] {layer['summary'][:1000]}")

        return chr(10).join(lines) if len(lines) > 1 else "No previous memory found."

    def get_all_memory(self) -> str:
        """Human-readable full memory dump for debugging."""
        lines = ["=== FULL MEMORY DUMP ==="]
        lines.append(f"Total moments: {self.memory['metadata']['total_moments']}")
        lines.append(f"Layer1 (current): {len(self.memory['layer1'])} moments")
        lines.append(f"Layer2 (same-env): {len(self.memory['layer2'])} moments")
        lines.append(f"Layer3 (today): {len(self.memory['layer3'])} moments")

        for lnum, lkey in [(4, "layer4"), (5, "layer5"), (6, "layer6")]:
            s = self.memory[lkey].get("summary", "")
            lines.append(f"Layer{lnum}: {s[:120] if s else '(empty)'}")

        l7 = self.memory["layer7"]
        lines.append(f"Layer7 people: {list(l7['people'].keys())[:10]}")
        lines.append(f"Layer7 locations: {list(l7['locations'].keys())[:10]}")
        lines.append(f"Layer7 events: {list(l7['activity_events'].keys())[:10]}")
        lines.append(f"Layer7 moments stored: {len(l7['moments'])}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Operation 2 : UPDATE（改正 moment 的 Part1-3 数据）
    # ------------------------------------------------------------------

    UPDATE_FIELDS = ("scene", "user_action", "needs", "solutions",
                     "people", "location", "activity", "notes", "source_evidence",
                     "interaction_details", "person_observations", "temporal_facts",
                     "source_time", "environments", "objects")

    def _validated_updates(self, updates: Dict) -> Dict:
        """按记忆 schema 筛选可更新字段和值类型。"""
        if not isinstance(updates, dict):
            return {}
        string_fields = {"scene", "user_action", "location", "activity", "notes",
                         "source_evidence", "interaction_details"}
        list_fields = {"people", "environments", "objects", "needs", "solutions"}
        clean = {}
        for field, value in updates.items():
            if field not in self.UPDATE_FIELDS or value is None:
                continue
            if field == "person_observations" and isinstance(value, list):
                clean[field] = _clean_source_fact_rows(value, "person", 16)
                continue
            if field == "temporal_facts" and isinstance(value, list):
                clean[field] = _clean_source_fact_rows(value, "fact", 32)
                continue
            if field == "source_time" and isinstance(value, dict):
                time_value = {}
                for key in ("absolute_date", "day_label", "time_of_day", "time_occasion", "basis", "source_segment_id"):
                    item = value.get(key)
                    if isinstance(item, str) and item.strip():
                        time_value[key] = item.strip()[:128]
                for key in ("clip_start_s", "clip_end_s"):
                    try:
                        item = float(value.get(key))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(item) and item >= 0:
                        time_value[key] = round(item, 3)
                clean[field] = time_value
                continue
            if field in string_fields:
                if isinstance(value, str):
                    clean[field] = value
                continue
            if field not in list_fields or not isinstance(value, list):
                continue
            if field == "people":
                clean[field] = [x for x in value
                                if isinstance(x, str) and _is_valid_person_tag(x)]
            elif field in ("environments", "objects"):
                clean[field] = [x for x in value if isinstance(x, str) and
                                _is_valid_index_tag(field, x)]
            else:
                clean[field] = [x for x in value if isinstance(x, (dict, str))]
        return clean

    def _sync_window_moment(self, moment_id: str) -> bool:
        """同步主存储到仍存活的 L1/L2/L3 窗口副本。"""
        moment = self.memory.get("layer7", {}).get("moments", {}).get(moment_id)
        if not moment:
            return False
        changed = False
        for layer_key in ("layer1", "layer2", "layer3"):
            for i, current in enumerate(self.memory.get(layer_key, [])):
                if current.get("id") == moment_id:
                    if current != moment:
                        changed = True
                    self.memory[layer_key][i] = moment
        return changed

    def _sync_indices_for_moment(self, moment_id: str, moment: Dict,
                                 old_people=None, old_location=None, old_activity=None,
                                 old_environments=None, old_objects=None):
        """Reconcile every index tag for one moment after an edit or cache refresh."""
        layer7 = self.memory.setdefault("layer7", {})
        for index_key in LAYER7_INDICES:
            index = layer7.setdefault(index_key, {})
            for tag, ids in list(index.items()):
                if moment_id in ids:
                    ids[:] = [mid for mid in ids if mid != moment_id]
                if not ids:
                    index.pop(tag, None)
        for index_key, tags in _moment_index_tags(moment).items():
            index = layer7.setdefault(index_key, {})
            for tag in tags:
                ids = index.setdefault(tag, [])
                if moment_id not in ids:
                    ids.append(moment_id)

    def update(self, moment_id: str, updates: Dict) -> bool:
        """改正指定 moment 的 Part1-3 数据，并同步 Layer7 标签索引。"""
        clean = self._validated_updates(updates)
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment or not clean:
            return False

        old_people = moment.get("people") or []
        old_location = moment.get("location")
        old_activity = moment.get("activity")
        old_environments = moment.get("environments") or []
        old_objects = moment.get("objects") or []
        changed_fields = []
        for field, value in clean.items():
            if moment.get(field) != value:
                moment[field] = value
                changed_fields.append(field)
        if not changed_fields:
            return False
        normalize_object_transition_facts(moment)
        if isinstance(moment.get("multimodal"), dict):
            entity_visuals = _build_entity_visuals(moment)
            if entity_visuals.get("items"):
                moment["multimodal"]["entity_visuals"] = entity_visuals

        # 清除可能过期的英文缓存；改写中文时再生成对应 normalized 值。
        searchable_fields = {"scene", "user_action", "interaction_details",
                             "person_observations", "temporal_facts", "source_time",
                             "location", "activity", "notes", "source_evidence"}
        if searchable_fields.intersection(changed_fields):
            normalized = dict(moment.get("normalized") or {})
            for field in searchable_fields.intersection(changed_fields):
                normalized.pop(field, None)
                value = moment.get(field)
                if self.translate_fn and value and _contains_cjk(str(value)):
                    try:
                        translated = self.translate_fn(str(value))
                        if translated:
                            normalized[field] = translated
                    except Exception:
                        pass
            if normalized:
                moment["normalized"] = normalized
            else:
                moment.pop("normalized", None)

        self._sync_indices_for_moment(
            moment_id, moment, old_people, old_location, old_activity,
            old_environments, old_objects)
        self._sync_window_moment(moment_id)
        self._save()
        return True

    def refresh_source_moment(self, moment_id: str, updates: Dict,
                              source_provenance: Dict,
                              expected_segment_id: str) -> bool:
        """Refresh model-extracted fields in place for the exact same source clip.

        The caller must already have validated the current source video fingerprint.
        User feedback corrections are protected, and every replaced value is retained
        in a bounded audit history so cache upgrades never erase the prior extraction.
        """
        moment = self.memory.get("layer7", {}).get("moments", {}).get(moment_id)
        if not isinstance(moment, dict) or not isinstance(source_provenance, dict):
            return False
        old_provenance = moment.get("source_provenance") or {}
        expected = str(expected_segment_id or "")
        if (not expected
                or str(old_provenance.get("source_segment_id") or "") != expected
                or str(source_provenance.get("source_segment_id") or "") != expected):
            return False
        for key in ("source_video_sha256", "source_clip_sha256"):
            old_hash, new_hash = old_provenance.get(key), source_provenance.get(key)
            if old_hash and new_hash and str(old_hash) != str(new_hash):
                return False

        clean = self._validated_updates(updates)
        feedback = moment.get("feedback") or {}
        corrections = feedback.get("corrections") if isinstance(feedback, dict) else {}
        protected = {str(key).split(".", 1)[0].split("[", 1)[0]
                     for key in corrections} if isinstance(corrections, dict) else set()
        clean = {key: value for key, value in clean.items() if key not in protected}

        old_people = moment.get("people") or []
        old_location = moment.get("location")
        old_activity = moment.get("activity")
        old_environments = moment.get("environments") or []
        old_objects = moment.get("objects") or []
        previous_fields = {}
        changed_fields = []
        for field, value in clean.items():
            if moment.get(field) != value:
                previous_fields[field] = copy.deepcopy(moment.get(field))
                moment[field] = value
                changed_fields.append(field)

        provenance_keys = (
            "source_video_id", "source_segment_id", "clip_start_s", "clip_end_s",
            "source_video_sha256", "source_clip_sha256", "perception_prompt_version",
            "list_parser_version", "extraction_batch_id",
        )
        new_provenance = {
            key: source_provenance[key] for key in provenance_keys
            if source_provenance.get(key) not in (None, "")
        }
        if new_provenance.get("source_video_id"):
            new_provenance["source_video_id"] = _portable_video_id(
                new_provenance["source_video_id"])
        provenance_changed = any(
            old_provenance.get(key) != value for key, value in new_provenance.items())
        if not changed_fields and not provenance_changed:
            return False
        normalize_object_transition_facts(moment)
        if isinstance(moment.get("multimodal"), dict):
            entity_visuals = _build_entity_visuals(moment)
            if entity_visuals.get("items"):
                moment["multimodal"]["entity_visuals"] = entity_visuals

        searchable_fields = {"scene", "user_action", "interaction_details",
                             "person_observations", "temporal_facts", "source_time",
                             "location", "activity", "notes", "source_evidence"}
        normalized = dict(moment.get("normalized") or {})
        for field in searchable_fields.intersection(changed_fields):
            normalized.pop(field, None)
            value = moment.get(field)
            if self.translate_fn and value and _contains_cjk(str(value)):
                try:
                    translated = self.translate_fn(str(value))
                    if translated:
                        normalized[field] = translated
                except Exception:
                    pass
        if normalized:
            moment["normalized"] = normalized
        else:
            moment.pop("normalized", None)

        if provenance_changed:
            moment["source_provenance"] = {**old_provenance, **new_provenance}
            previous_fields["source_provenance"] = copy.deepcopy(old_provenance)
            changed_fields.append("source_provenance")
        history = moment.setdefault("source_refresh_history", [])
        history.append({
            "refreshed_at": _now_iso(),
            "from_prompt_version": old_provenance.get("perception_prompt_version"),
            "to_prompt_version": new_provenance.get("perception_prompt_version"),
            "from_parser_version": old_provenance.get("list_parser_version"),
            "to_parser_version": new_provenance.get("list_parser_version"),
            "changed_fields": list(changed_fields),
            "previous_fields": previous_fields,
        })
        moment["source_refresh_history"] = history[-8:]
        self._sync_indices_for_moment(
            moment_id, moment, old_people, old_location, old_activity,
            old_environments, old_objects)
        self._sync_window_moment(moment_id)
        self._save()
        return True

    def get_context(self, people: List[str] = None, location: str = None) -> str:
        """Backward-compatible shim."""
        return self.get_context_for_analysis(people=people, location=location)


# ---------------------------------------------------------------------------
# HintMemory — extra memory bank for user-supplied "when X seen → output need Y"
# ---------------------------------------------------------------------------
#
# This is a SEPARATE memory store from PersonMemory. It is *not* episodic;
# it holds user-curated rules that get injected into the analysis prompt so
# the agent can recognise needs the base model would otherwise miss.
#
# Each hint is a small dict:
#   {
#     "id":       "h_<timestamp>",
#     "when":     "trigger description (objects/scene/conditions to look for)",
#     "then":     "the specific need / action the agent should output when the trigger matches",
#     "added_at": "ISO timestamp"
#   }
#
# Public ops:
#   add(when, then)              -> hint_id
#   remove(hint_id)              -> bool
#   list()                       -> List[Dict]
#   clear()                      -> None
#   format_for_prompt()          -> str  (block to splice into the LLM prompt)
# ---------------------------------------------------------------------------


class HintMemory:
    """User-curated trigger→need rules, stored alongside PersonMemory."""

    def __init__(self, memory_dir: str = "memory", filename: str = "hints.json"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.hints_file = self.memory_dir / filename
        self.hints: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self.hints_file.exists():
            try:
                with open(self.hints_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def _save(self):
        with open(self.hints_file, "w", encoding="utf-8") as f:
            json.dump(self.hints, f, indent=2, ensure_ascii=False)

    def add(self, when: str, then: str) -> str:
        hint_id = f"h_{int(time.time() * 1000)}_{len(self.hints)}"
        self.hints.append({
            "id": hint_id,
            "when": when.strip(),
            "then": then.strip(),
            "added_at": _now_iso(),
        })
        self._save()
        return hint_id

    def remove(self, hint_id: str) -> bool:
        before = len(self.hints)
        self.hints = [h for h in self.hints if h.get("id") != hint_id]
        changed = len(self.hints) != before
        if changed:
            self._save()
        return changed

    def list(self) -> List[Dict]:
        return list(self.hints)

    def clear(self):
        self.hints = []
        self._save()

    def format_for_prompt(self) -> str:
        """Return a prompt-ready block, or an empty string if no hints."""
        if not self.hints:
            return ""
        lines = ["=== USER HINT RULES (apply these when the trigger is observed) ==="]
        for i, h in enumerate(self.hints, 1):
            lines.append(f"  {i}. WHEN: {h.get('when','')}")
            lines.append(f"     THEN output need: {h.get('then','')}")
        lines.append("=== END USER HINT RULES ===")
        return "\n".join(lines)
