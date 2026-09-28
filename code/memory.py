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
  combine   — merge new data with similar existing data in layers 4-5-6-7
  delete    — one-off / accidental / erroneous entries; post-compress duplicates
  highlight — user-confirmed correct data; data that recurs and is merged/similar
"""

import json
import math
import os
from collections import Counter
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
    if len(cleaned) > 30:
        return False
    for pat in NON_PERSON_PATTERNS:
        if pat in lowered:
            return False
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
        if len(cleaned) > 80:
            return False
    if index_key == "activity_events":
        if lowered in {"none", "n/a", "unknown", "无", "暂无", "event_tag"}:
            return False
        if len(cleaned) > 80:
            return False
    if index_key in ("environments", "objects"):
        if lowered in {"none", "n/a", "unknown", "无", "暂无"}:
            return False
        if len(cleaned) > 80:
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
    """单值型冲突裁决：频次 > 最近 > 置信度，败者入 history 供审计。"""
    if new.get("observations", 0) > old.get("observations", 0):
        winner, loser = new, old
    elif new.get("observations", 0) < old.get("observations", 0):
        winner, loser = old, new
    elif new.get("last_seen", "") > old.get("last_seen", ""):
        winner, loser = new, old
    elif new.get("confidence", 0) > old.get("confidence", 0):
        winner, loser = new, old
    else:
        winner, loser = old, new
    winner.setdefault("history", []).append(loser)
    return winner


def _merge_attr_list(old_list: List, new_list: List, threshold: float = MERGE_SIM_THRESHOLD) -> List:
    """列表型维度：追加去重。相似 → 佐证合并，否则追加新条目。绝不整体覆盖。"""
    for na in new_list:
        hit = next(
            (o for o in old_list
             if isinstance(o, dict) and "value" in o
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

def _rank_memory_docs(query_text: str, documents: list) -> list:
    """Rank active memories with field-aware BM25.

    Text is scored within semantic fields (scene/action/entities/tags) instead
    of flattening a whole record into one long document. This keeps concise
    people, object, place, and activity fields searchable when action text is
    much longer. Normalized translated text remains searchable at a lower
    weight so multilingual queries still work without live translation.
    """
    documents = [doc for doc in documents
                 if isinstance(doc, dict) and not doc.get("stale")]
    if not documents:
        return []
    query_terms = list(dict.fromkeys(_retrieval_tokens(str(query_text or "")[:1200])))
    if not query_terms:
        return []

    field_weights = {
        "scene": 1.0,
        "user_action": 1.0,
        "activity": 1.0,
        "people": 1.0,
        "location": 1.0,
        "objects": 1.0,
        "environments": 1.0,
        "activity_events": 1.0,
        "needs": 1.0,
        "solutions": 1.0,
        "time_occasion": 0.7,
        "extra_notes": 0.6,
        "normalized": 0.5,
        "source_evidence": float(os.environ.get("MEMORY_SOURCE_EVIDENCE_WEIGHT", "1.4")),
    }
    metadata = {
        "id", "timestamp", "created_at", "updated_at", "last_used",
        "access_count", "stale", "decay_type", "decay_rate", "expires_at",
        "confidence", "highlighted", "feedback", "source_ids", "layer",
    }

    def as_search_text(value):
        if isinstance(value, dict):
            return " ".join(as_search_text(item) for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return " ".join(as_search_text(item) for item in value)
        if value is None:
            return ""
        return str(value)

    # Keep common semantic fields independent. Unknown content fields are
    # searchable at a conservative weight for schema compatibility.
    all_fields = set(field_weights)
    field_counts = []
    field_lengths = []
    document_frequency = Counter()
    for document in documents:
        values = {}
        for key, value in document.items():
            if key in metadata:
                continue
            field = key if key in field_weights else "other"
            values[field] = (values.get(field, "") + " " + as_search_text(value)).strip()
        counts = {
            field: Counter(_retrieval_tokens(values.get(field, "").lower()))
            for field in all_fields | {"other"}
        }
        lengths = {field: sum(counter.values()) for field, counter in counts.items()}
        field_counts.append(counts)
        field_lengths.append(lengths)
        for field, counter in counts.items():
            for term in counter:
                document_frequency[(field, term)] += 1

    live_count = len(documents)
    average_lengths = {}
    for field in all_fields | {"other"}:
        nonempty = [lengths[field] for lengths in field_lengths if lengths[field] > 0]
        average_lengths[field] = sum(nonempty) / len(nonempty) if nonempty else 1.0

    k1, b = 1.2, 0.75
    ranked = []
    for index, counts_by_field in enumerate(field_counts):
        score = 0.0
        for term in query_terms:
            for field, counts in counts_by_field.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = document_frequency[(field, term)]
                idf = math.log(1.0 + (live_count - df + 0.5) / (df + 0.5))
                average_length = average_lengths[field]
                length = field_lengths[index][field]
                norm_tf = (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * length / average_length)
                )
                score += field_weights.get(field, 0.75) * idf * norm_tf
        if score > 0:
            ranked.append((score, index, documents[index]))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(score, document) for score, _, document in ranked]

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
        "source_evidence": "",  # speaker-labelled transcript facts supporting the event
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
            "life_trajectory": "",
            "last_updated": None,
            "source_moment_count": 0
        },
        "layer5": {
            "summary": "",
            "key_events": [],
            "long_term_patterns": "",
            "last_updated": None,
            "source_moment_count": 0
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
            "source_moment_count": 0
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
        # 非英文 → 英文翻译函数（统一英文分词策略，见设计说明书 §6.2）。
        # 可注入；None 时跳过翻译（非英文文本无法分词，检索走原文兜底降级）。
        self.translate_fn = translate_fn

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
            # 清洗 layer7 索引污染（修复 G7 缺陷）
            _clean_layer7_indices(data.get("layer7", {}))
            return data
        return _empty_db()

    def clean_layer7_indices(self) -> Dict[str, List[str]]:
        """清洗 layer7 索引并落盘保存。"""
        removed = _clean_layer7_indices(self.memory.get("layer7", {}))
        if any(removed.values()):
            self._save()
        return removed

    def _save(self):
        # 原子落盘：先写临时文件再 rename，防止并发写坏 / 写一半崩溃（04 文档 2.5 步骤⑥）。
        tmp = self.memory_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.memory_file)

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
            source_evidence: str = "") -> str:
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
        if source_evidence:
            moment["source_evidence"] = source_evidence
        moment["needs"] = needs
        moment["solutions"] = solutions
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

        moment_id = self._new_moment_id()
        moment["id"] = moment_id

        # 预翻译（统一英文分词：写入时翻译非英文文本存 normalized，检索查英文）
        if self.translate_fn:
            normalized = {}
            for field, raw in (("scene", scene), ("user_action", user_action),
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
        event_day = event_time.date().isoformat()
        self.memory["layer7"]["time_nodes"].setdefault(event_day, []).append(moment_id)

        if people:
            for p in people:
                if _is_valid_person_tag(p):
                    self.memory["layer7"]["people"].setdefault(p, []).append(moment_id)

        if location and _is_valid_index_tag("locations", location):
            self.memory["layer7"]["locations"].setdefault(location, []).append(moment_id)

        if activity and _is_valid_index_tag("activity_events", activity):
            self.memory["layer7"]["activity_events"].setdefault(activity, []).append(moment_id)

        if environments:
            for e in environments:
                if _is_valid_index_tag("environments", e):
                    self.memory["layer7"]["environments"].setdefault(e, []).append(moment_id)

        if objects:
            for o in objects:
                if _is_valid_index_tag("objects", o):
                    self.memory["layer7"]["objects"].setdefault(o, []).append(moment_id)

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
            return _merge_attr_list(old if isinstance(old, list) else [], new_attrs)

        if isinstance(new, dict):
            if "value" in new:
                na = _finalize_attr(new, source, timestamp)
                if na is None or na["confidence"] < MIN_CONFIDENCE:
                    return old
                if isinstance(old, dict) and "value" in old:
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
        """返回当前 layer6 画像（规格书骨架结构）。"""
        return self.memory["layer6"]["profile"]

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
        if top_k <= 0:
            return []
        documents = list(self.memory["layer7"]["moments"].values())
        return [moment for _, moment in _rank_memory_docs(query_text, documents)[:top_k]]

    def query_hierarchical(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search L1→L5 in order, using BM25 within each layer and early stopping."""
        if top_k <= 0 or not _retrieval_tokens(str(query_text or "")[:1200]):
            return []

        results = []
        seen_ids = set()
        for layer_no in range(1, 6):
            if layer_no <= 3:
                layer_key = f"layer{layer_no}"
                documents = self._sorted_layer_moments(layer_key)
                for score, moment in _rank_memory_docs(query_text, documents):
                    moment_id = moment.get("id")
                    if moment_id and moment_id in seen_ids:
                        continue
                    if moment_id:
                        seen_ids.add(moment_id)
                    results.append({"layer": layer_no, "score": round(score, 4), "moment": moment})
                    if len(results) >= top_k:
                        break
            else:
                layer = self.memory.get(f"layer{layer_no}", {})
                ranked = _rank_memory_docs(query_text, [layer])
                if ranked:
                    score, _ = ranked[0]
                    moment = {"id": f"layer{layer_no}_summary", "layer": layer_no, **layer}
                    results.append({"layer": layer_no, "score": round(score, 4), "moment": moment})
            if len(results) >= top_k:
                return results[:top_k]
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

        query_text = str(query_text or "")[:1200]
        query_tokens = set(self._query_tokens(
            query_text, *(str(x) for x in (people or [])),
            str(location or ""), str(activity or "")))
        excluded = set(exclude_ids or ())
        scored = []
        for moment_id, moment in self.memory["layer7"]["moments"].items():
            if moment_id in excluded or moment.get("stale"):
                continue
            searchable = json.dumps(moment, ensure_ascii=False).lower()
            score = sum(1 for token in query_tokens if token in searchable)
            if score:
                scored.append((score, bool(moment.get("highlighted")),
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
            fields = [
                ("场景", moment.get("scene")),
                ("动作", moment.get("user_action")),
                ("需求", values(moment.get("needs"))),
                ("方案", values(moment.get("solutions"))),
                ("人物", values(moment.get("people"))),
                ("地点", moment.get("location")),
                ("活动", moment.get("activity")),
                ("反馈修正", feedback_text),
            ]
            if include_source_evidence and moment.get("source_evidence"):
                evidence = str(moment["source_evidence"]).replace(chr(10), " ").strip()
                line += f" | Transcript evidence: {evidence[:1200]}"
            for label, value in fields:
                if value:
                    line += f" | {label}: {str(value).replace(chr(10), ' ').strip()[:180]}"
            parts.append(line)

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
                layer["source_moment_count"] = self.memory["metadata"]["total_moments"]

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
                existing = set(self.memory["layer7"][index_key].get(tag, []))
                existing.update(valid_ids)
                self.memory["layer7"][index_key][tag] = list(existing)
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
        - 合并 moment_id 集合并同步更新 moments 主存储里的引用字段：
          locations → location 字段；people → people 列表元素；activity_events → activity 字段。
        """
        # 索引 → (moments 引用字段, 字段类型[str 单值 / list 列表])
        REF_FIELD = {
            "locations": ("location", "str"),
            "people": ("people", "list"),
            "activity_events": ("activity", "str"),
            "environments": ("environments", "list"),
            "objects": ("objects", "list"),
        }
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
                ids = index.pop(old_key)
                existing = set(index.get(canonical, []))
                existing.update(ids)
                index[canonical] = list(existing)
                # 规范（09-22 highlight）：重复出现 / 归类合并的相似相同数据自动标记
                if len(existing) > len(ids):
                    for mid in ids:
                        m = self.memory["layer7"]["moments"].get(mid)
                        if m:
                            m["highlighted"] = True
                # 同步更新 moments 主存储引用字段
                if ref:
                    field, kind = ref
                    moments = self.memory["layer7"]["moments"]
                    for mid in existing:
                        m = moments.get(mid)
                        if not m:
                            continue
                        if kind == "str":
                            if m.get(field) == old_key:
                                m[field] = canonical
                        elif kind == "list":
                            if isinstance(m.get(field), list) and old_key in m[field]:
                                m[field] = [canonical if x == old_key else x
                                            for x in m[field]]
                        self._sync_window_moment(mid)
        self._save()

    # ------------------------------------------------------------------
    # Operation 8 : DELETE
    # ------------------------------------------------------------------

    def delete(self, moment_id: str = None, reason: str = "manual",
               force: bool = False) -> bool:
        """删除 moment（设计说明书 §4.5）。

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
        print(f"🗑️  Deleted moment {moment_id} ({reason})")
        return True

    # ------------------------------------------------------------------
    # Operation 9 : HIGHLIGHT
    # ------------------------------------------------------------------

    def highlight(self, moment_id: str) -> bool:
        """标记 moment 为 confirmed-correct（设计说明书 §4.5）。

        效果：① 删除清扫中受保护（delete 默认拒删）；② v2：检索排序提权、
        画像更新高权重佐证。moment 不存在返回 False。
        """
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment:
            return False
        changed = not moment.get("highlighted", False)
        moment["highlighted"] = True
        changed = self._sync_window_moment(moment_id) or changed
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
        """按文档优先级查询 L1→L5，再注入 L6 画像和命中的 L7 原始事实。"""
        lines = ["=== MEMORY CONTEXT ==="]
        query_text = str(query_text or "")[:1200]
        top_k = max(0, int(top_k))
        hits = self.query_hierarchical(query_text, top_k=top_k) if query_text.strip() else []
        seen_ids = set()

        def render_items(items):
            if not isinstance(items, list):
                return str(items or "")
            values = []
            for item in items:
                if isinstance(item, dict):
                    value = (item.get("need") or item.get("solution")
                             or item.get("value") or item.get("action"))
                else:
                    value = item
                if value:
                    values.append(str(value))
            return "；".join(values)

        def add_moment(layer_no, moment):
            moment_id = moment.get("id")
            if moment_id:
                seen_ids.add(moment_id)
            fields = [
                ("场景", moment.get("scene")),
                ("动作", moment.get("user_action")),
                ("需求", render_items(moment.get("needs"))),
                ("方案", render_items(moment.get("solutions"))),
                ("人物", render_items(moment.get("people"))),
                ("地点", moment.get("location")),
                ("活动", moment.get("activity")),
            ]
            feedback = moment.get("feedback") or {}
            corrections = feedback.get("corrections") or {}
            feedback_text = "；".join(
                f"{key}={value}" for key, value in corrections.items()
            ) if isinstance(corrections, dict) else ""
            if feedback.get("confirmed"):
                feedback_text = (feedback_text + "；" if feedback_text else "") + "用户已确认"
            if feedback_text:
                fields.append(("Part4反馈", feedback_text))
            line = f"[Layer {layer_no} · {moment.get('timestamp','')[:19]} {moment_id or ''}]"
            for label, value in fields:
                if value:
                    line += f" | {label}: {str(value).replace(chr(10), ' ')[:180]}"
            lines.append(line)

        if hits:
            for hit in hits:
                layer_no, moment = hit["layer"], hit["moment"]
                if layer_no <= 3:
                    add_moment(layer_no, moment)
                elif moment.get("summary"):
                    lines.append(f"[Layer {layer_no}] {moment['summary'][:1200]}")
        elif not query_text and top_k:
            # 没有查询语句时以最近窗口作短上下文；同一 moment 只注入一次。
            for layer_no in (1, 2, 3):
                for moment in self._sorted_layer_moments(f"layer{layer_no}"):
                    if moment.get("id") and moment["id"] in seen_ids:
                        continue
                    add_moment(layer_no, moment)
                    if len(seen_ids) >= top_k:
                        break
                if len(seen_ids) >= top_k:
                    break
            for layer_no in (4, 5):
                summary = self.memory.get(f"layer{layer_no}", {}).get("summary")
                if summary:
                    lines.append(f"[Layer {layer_no}] {summary[:1200]}")

        # L6 profile 的顺序采用文档所列频率、确定性、重要性和强度维度。
        profile = self.memory["layer6"]["profile"]
        profile_parts = []
        for field in PROFILE_FIELDS:
            field_attrs = []
            if field in ("behavior_patterns", "frequent_locations"):
                sort_key = "frequency"
            elif field == "personality":
                sort_key = "confidence"
            elif field in ("preferences", "goals", "decisions", "motivations"):
                sort_key = "rank_score"
            else:
                sort_key = "confidence"
            _collect_attrs(profile.get(field), field_attrs, sort_key)
            if field_attrs:
                profile_parts.append(f"{field}:[{'; '.join(field_attrs)}]")
        profile_summary = self.memory["layer6"].get("summary")
        if profile_summary:
            lines.append(f"[Layer 6 summary] {profile_summary[:1000]}")
        if profile_parts:
            lines.append(f"[Layer 6 profile] {' | '.join(profile_parts)}")

        # L7 检索补全场景、动作、需求、方案及 Part4 反馈；同步链路不调用 LLM。
        extra = self.retrieve(
            people=people, location=location, activity=activity,
            query_text=query_text, top_k=top_k, include_summaries=False,
            exclude_ids=seen_ids)
        if extra != "No relevant memory found.":
            lines.append(extra)

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
                     "environments", "objects")

    def _validated_updates(self, updates: Dict) -> Dict:
        """按记忆 schema 筛选可更新字段和值类型。"""
        if not isinstance(updates, dict):
            return {}
        string_fields = {"scene", "user_action", "location", "activity", "notes", "source_evidence"}
        list_fields = {"people", "environments", "objects", "needs", "solutions"}
        clean = {}
        for field, value in updates.items():
            if field not in self.UPDATE_FIELDS or value is None:
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
                                 old_people, old_location, old_activity,
                                 old_environments=None, old_objects=None):
        """moment 的 people/location/activity/environments/objects 变更时联动 layer7 索引
        （设计说明书 §4.2 update 语义：新键过校验，旧键清理空引用）。
        """
        def sync(key: str, old_val, new_val):
            if old_val == new_val:
                return
            idx = self.memory["layer7"][key]
            olds = [old_val] if isinstance(old_val, str) else (old_val or [])
            for tag in olds:
                if tag in idx and moment_id in idx[tag]:
                    idx[tag].remove(moment_id)
                    if not idx[tag]:
                        del idx[tag]
            news = [new_val] if isinstance(new_val, str) else (new_val or [])
            for tag in news:
                if not tag or not _is_valid_index_tag(key, tag):
                    continue
                idx.setdefault(tag, [])
                if moment_id not in idx[tag]:
                    idx[tag].append(moment_id)

        sync("people", old_people, moment.get("people"))
        sync("locations", old_location, moment.get("location"))
        sync("activity_events", old_activity, moment.get("activity"))
        sync("environments", old_environments, moment.get("environments"))
        sync("objects", old_objects, moment.get("objects"))

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

        # 清除可能过期的英文缓存；改写中文时再生成对应 normalized 值。
        searchable_fields = {"scene", "user_action", "location", "activity", "notes", "source_evidence"}
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
