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
import os
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
PROFILE_FIELDS = ("demographics", "preferences", "frequent_locations", "behavior_patterns")
# dict 型字段（值是 {子键: ...}，区别于 list 型 frequent_locations）
PROFILE_DICT_FIELDS = {"demographics", "preferences", "behavior_patterns"}

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
        "source": source,
        "timestamp": timestamp or _now_iso(),
        "last_seen": timestamp or _now_iso(),
        "observations": 1,
    }


PLACEHOLDER_TAGS = {
    "canonical_name", "canonical_location", "canonical_tag",
    "event_tag", "moment_id_1", "moment_id_2"
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
    return True


def _clean_layer7_indices(layer7: Dict) -> Dict[str, List[str]]:
    """清洗 layer7 索引：移除占位符、描述性长句、无意义负向词（G7 缺陷修复）。"""
    removed = {"people": [], "locations": [], "activity_events": [], "time_nodes": []}
    if not isinstance(layer7, dict):
        return removed
    for k in ["people", "locations", "activity_events", "time_nodes"]:
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
    ts = timestamp or _now_iso()
    return {
        "value": value,
        "confidence": max(0.0, min(1.0, conf)),
        "source": attr.get("source") or source or "llm_inference",
        "evidence": str(attr.get("evidence", ""))[:200],
        "timestamp": ts,
        "last_seen": ts,
        "observations": 1,
    }


def _corroborate_attr(old: Dict, new: Dict) -> Dict:
    """佐证合并：同值重复出现 → observations 累加、confidence 增强、last_seen 更新。"""
    old["observations"] += new.get("observations", 1)
    old["confidence"] = 1 - (1 - old["confidence"]) * (1 - new["confidence"])
    old["last_seen"] = max(old.get("last_seen", ""), new.get("last_seen", ""))
    if new["confidence"] > old["confidence"]:
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
# 检索分词（04 文档 2.3.4 定稿：字符 bigram，兼容中英，零依赖）
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """中文按字符 bigram 切分，英文/数字按空格与边界切分。零第三方依赖。

    解决 09-03 实测的 `query("在图书馆学习") → 0 条` 问题：
    中文整句按 split() 只出 1 个 token，bigram 后能命中「图书馆」等多字词。
    """
    if not text:
        return []
    tokens, buf = [], ""
    for ch in text.lower():
        if '\u4e00' <= ch <= '\u9fff':          # 中文字符
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        elif ch.isalnum():                       # 英数，累积成词
            buf += ch
        else:                                    # 分隔符
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)

    # 中文相邻单字再生成 bigram，提升多字词可匹配性
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if len(t) == 1 and '\u4e00' <= t <= '\u9fff':
            if i + 1 < len(tokens) and len(tokens[i + 1]) == 1 \
               and '\u4e00' <= tokens[i + 1] <= '\u9fff':
                out.append(t + tokens[i + 1])
    return out


# ---------------------------------------------------------------------------
# 时间衰减与陈旧淘汰（04 文档 2.4.4）
# ---------------------------------------------------------------------------

# 衰减率（每天）：稳定量（layer6 画像）0.5%；中期量 2%；瞬时量 15%（画像全为稳定量）
STABLE_DAILY_DECAY = 0.005
# 陈旧淘汰阈值：有效置信度 < 0.3 标记 stale（不参与检索，不物理删除）
STALE_THRESHOLD = 0.3


def effective_confidence(attr: Dict, daily_decay: float = STABLE_DAILY_DECAY,
                         now: Optional[str] = None) -> float:
    """有效置信度 = 原始置信度 × (1-daily_decay)^days，days 按 last_seen 起算。"""
    conf = float(attr.get("confidence", 0))
    last_seen = attr.get("last_seen")
    if not last_seen:
        return conf
    try:
        last = datetime.fromisoformat(last_seen)
        now_dt = datetime.fromisoformat(now) if now else datetime.now()
        days = max(0, (now_dt - last).days)
    except (TypeError, ValueError):
        days = 0
    return max(conf * (1 - daily_decay) ** days, 0.0)


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


def _collect_attrs(node, out: list):
    """递归收集非 stale 的 AttrValue 为 'value(conf)' 字符串（供画像注入 prompt）。"""
    if isinstance(node, dict) and "value" in node:
        if not node.get("stale"):
            out.append(f"{node['value']}({node.get('confidence', 0):.2f})")
    elif isinstance(node, list):
        for x in node:
            _collect_attrs(x, out)
    elif isinstance(node, dict):
        for v in node.values():
            _collect_attrs(v, out)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

def _empty_moment() -> Dict:
    """A single captured moment (goes into Layer 1 → 2 → 3)."""
    return {
        "timestamp": _now_iso(),
        # Part 1
        "scene": "",          # external environment description
        "user_action": "",    # user's own behaviour / action
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
                "behavior_patterns": {}
            },
            "last_updated": None,
            "source_moment_count": 0
        },

        # --- classified archive ---
        "layer7": {
            "time_nodes": {},      # {"2024-02-10": [moment_ids]}
            "activity_events": {}, # {"chinese_new_year": [moment_ids]}
            "people": {},          # {"姥姥": [moment_ids]}
            "locations": {},       # {"family_courtyard": [moment_ids]}
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

    MAX_LAYER1 = 5    # keep last N moments in the "current" layer
    MAX_LAYER2 = 20   # same-env history within a session
    MAX_LAYER3 = 100  # all moments today (before daily compress)

    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.memory_file = self.memory_dir / "memory.json"
        self.memory: Dict = self._load()

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
            extra_notes: str = "") -> str:
        """
        Store a complete Part1+2+3 result.
        Returns the moment_id so Part4 can reference it later.

        Layers updated: 1, 2, 3, 7 (index only)
        Layers 4-6 are updated only during consolidation (not here).
        """
        moment = _empty_moment()
        moment["scene"] = scene
        moment["user_action"] = user_action
        moment["needs"] = needs
        moment["solutions"] = solutions
        if location:
            moment["location"] = location
        if people:
            moment["people"] = people
        if activity:
            moment["activity"] = activity
        if extra_notes:
            moment["notes"] = extra_notes

        moment_id = self._new_moment_id()
        moment["id"] = moment_id

        # ---- Layer 1 (current window) ----
        self.memory["layer1"].append(moment)
        if len(self.memory["layer1"]) > self.MAX_LAYER1:
            # oldest moment graduates to layer2
            old = self.memory["layer1"].pop(0)
            self._graduate_to_layer2(old)

        # ---- Layer 3 (today) ----
        self.memory["layer3"].append(moment)
        if len(self.memory["layer3"]) > self.MAX_LAYER3:
            self.memory["layer3"].pop(0)

        # ---- Layer 7 master store ----
        self.memory["layer7"]["moments"][moment_id] = moment

        # ---- Layer 7 indices ----
        today = _today_str()
        self.memory["layer7"]["time_nodes"].setdefault(today, []).append(moment_id)

        if people:
            for p in people:
                if _is_valid_person_tag(p):
                    self.memory["layer7"]["people"].setdefault(p, []).append(moment_id)

        if location and _is_valid_index_tag("locations", location):
            self.memory["layer7"]["locations"].setdefault(location, []).append(moment_id)

        if activity and _is_valid_index_tag("activity_events", activity):
            self.memory["layer7"]["activity_events"].setdefault(activity, []).append(moment_id)

        # ---- metadata ----
        self.memory["metadata"]["total_moments"] += 1
        self.memory["metadata"]["total_encounters"] += 1

        self._save()
        return moment_id

    def _graduate_to_layer2(self, moment: Dict):
        """Move a moment from layer1 to layer2 if env matches current scene."""
        moment["layer"] = 2
        self.memory["layer2"].append(moment)
        if len(self.memory["layer2"]) > self.MAX_LAYER2:
            self.memory["layer2"].pop(0)

    # ------------------------------------------------------------------
    # Operation 2 : UPDATE
    # ------------------------------------------------------------------

    def update_feedback(self, moment_id: str, corrections: Dict = None,
                        confirmed: bool = False, user_rating: int = None):
        """
        Part4 feedback: correct Part1-3 data.
        This is called AFTER the main pipeline and does not block it.
        """
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment:
            return

        fb = moment["feedback"]
        fb["confirmed"] = confirmed
        if corrections:
            fb["corrections"].update(corrections)
            # Apply corrections directly to moment fields
            for field, val in corrections.items():
                if field in moment:
                    moment[field] = val
        if user_rating is not None:
            fb["user_rating"] = user_rating

        # Sync back into layer1/2/3 if still present
        for layer_key in ["layer1", "layer2", "layer3"]:
            for i, m in enumerate(self.memory[layer_key]):
                if m.get("id") == moment_id:
                    self.memory[layer_key][i] = moment

        self._save()

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

        # 时间衰减重算 + 陈旧标记（04 文档 2.5 步骤④⑤，用真实当前时间相对 last_seen 计算）
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

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """关键词检索（bigram 分词，兼容中文），按 token 命中数排序取 Top-K。"""
        tokens = tokenize(query_text)
        if not tokens:
            return []

        scored = []
        for m in self.memory["layer7"]["moments"].values():
            if m.get("stale"):
                continue
            text = json.dumps(m, ensure_ascii=False).lower()
            score = sum(1 for t in tokens if t in text)
            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    # ------------------------------------------------------------------
    # Operation 4 : RETRIEVE  (called by Part2 / consolidation)
    # ------------------------------------------------------------------

    def retrieve(self, people: List[str] = None, location: str = None,
                 activity: str = None, layer: int = None, top_k: int = 5) -> str:
        """检索结构化上下文（相关性排序取 Top-K，跳过 stale）。"""
        parts = []

        # Layer 4-6 compressed summaries
        for lnum, lkey in [(4, "layer4"), (5, "layer5"), (6, "layer6")]:
            if layer and layer != lnum:
                continue
            ldata = self.memory[lkey]
            if ldata.get("summary"):
                parts.append(f"[Layer {lnum}] {ldata['summary']}")

        # Layer 7：按 person/location/activity 关键词打分排序
        query_tokens = []
        for p in (people or []):
            query_tokens += tokenize(p)
        if location:
            query_tokens += tokenize(location)
        if activity:
            query_tokens += tokenize(activity)

        scored = []
        for m in self.memory["layer7"]["moments"].values():
            if m.get("stale"):
                continue
            text = json.dumps(m, ensure_ascii=False).lower()
            score = sum(1 for t in query_tokens if t in text)
            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, m in scored[:top_k]:
            needs_str = "; ".join(n.get("need", "") for n in m.get("needs", []))
            parts.append(
                f"[Layer7/{m.get('id','')}] {m.get('timestamp','')[:10]} | "
                f"Scene: {m.get('scene','')[:80]} | "
                f"Needs: {needs_str[:80]}"
            )

        return "\n".join(parts) if parts else "No relevant memory found."

    # ------------------------------------------------------------------
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
                        # Merge lists (avoid duplicates)
                        existing = set(str(x) for x in layer[k])
                        for item in v:
                            if str(item) not in existing:
                                layer[k].append(item)
                                existing.add(str(item))
                    else:
                        layer[k] = v
                layer["last_updated"] = _now_iso()
                layer["source_moment_count"] = self.memory["metadata"]["total_moments"]

        self.memory["metadata"]["last_consolidation"] = _now_iso()
        self._save()

    # ------------------------------------------------------------------
    # Operation 6 : SORT  (called together with compress)
    # ------------------------------------------------------------------

    def sort(self, sort_analysis: Dict):
        """
        Re-classify layer7 indices based on LLM analysis.

        sort_analysis format:
        {
          "people": {"姥姥": ["m_xxx", ...]},
          "locations": {"family_courtyard": ["m_xxx", ...]},
          "activity_events": {"chinese_new_year": ["m_xxx", ...]},
          "time_nodes": {"2024-02-10": ["m_xxx", ...]}
        }
        """
        for index_key in ["people", "locations", "activity_events", "time_nodes"]:
            if index_key in sort_analysis:
                for tag, ids in sort_analysis[index_key].items():
                    if not _is_valid_index_tag(index_key, tag):
                        continue
                    existing = set(self.memory["layer7"][index_key].get(tag, []))
                    existing.update(ids)
                    self.memory["layer7"][index_key][tag] = list(existing)
        self._save()

    # ------------------------------------------------------------------
    # Operation 7 : COMBINE  (called together with compress)
    # ------------------------------------------------------------------

    def combine(self, canonical_map: Dict):
        """
        Merge near-duplicate layer7 index keys into a canonical form.

        canonical_map format:
        {
          "locations": {"Old fuzzy name": "canonical_name"},
          "activity_events": {"old tag": "canonical tag"}
        }
        """
        for index_key, mapping in canonical_map.items():
            if index_key not in self.memory["layer7"]:
                continue
            index = self.memory["layer7"][index_key]
            for old_key, canonical in mapping.items():
                if not _is_valid_index_tag(index_key, canonical):
                    continue
                if old_key in index and old_key != canonical:
                    ids = index.pop(old_key)
                    existing = set(index.get(canonical, []))
                    existing.update(ids)
                    index[canonical] = list(existing)
                    # Update moment references too
                    for mid in existing:
                        m = self.memory["layer7"]["moments"].get(mid)
                        if m:
                            if index_key == "locations" and m.get("location") == old_key:
                                m["location"] = canonical
        self._save()

    # ------------------------------------------------------------------
    # Operation 8 : DELETE
    # ------------------------------------------------------------------

    def delete(self, moment_id: str = None, reason: str = "manual"):
        """
        Remove a one-off / accidental / post-compress duplicate moment.
        """
        if moment_id and moment_id in self.memory["layer7"]["moments"]:
            del self.memory["layer7"]["moments"][moment_id]
            # Remove from indices
            for index_key in ["people", "locations", "activity_events", "time_nodes"]:
                for tag, ids in self.memory["layer7"][index_key].items():
                    if moment_id in ids:
                        ids.remove(moment_id)
            # Remove from raw layers
            for lkey in ["layer1", "layer2", "layer3"]:
                self.memory[lkey] = [m for m in self.memory[lkey]
                                     if m.get("id") != moment_id]
            self._save()
            print(f"🗑️  Deleted moment {moment_id} ({reason})")

    # ------------------------------------------------------------------
    # Operation 9 : HIGHLIGHT
    # ------------------------------------------------------------------

    def highlight(self, moment_id: str):
        """
        Mark a moment as confirmed-correct / high-value.
        Highlighted moments survive delete sweeps and get higher retrieval priority.
        """
        if moment_id in self.memory["layer7"]["moments"]:
            self.memory["layer7"]["moments"][moment_id]["highlighted"] = True
            self._save()

    # ------------------------------------------------------------------
    # Context helpers (called by pipeline)
    # ------------------------------------------------------------------

    def get_context_for_analysis(self, people: List[str] = None,
                                  location: str = None,
                                  activity: str = None) -> str:
        """按 04 文档 2.3.1 early-stop 路径组装上下文：
        layer6 稳定画像 → layer5 长期模式 → layer4 近期任务 → layer7 索引细节。
        只渲染非 stale 画像；同步链路不含任何 LLM 调用。
        """
        lines = ["=== MEMORY CONTEXT ==="]

        # ① layer6 稳定画像（直读，排除 stale，value 带置信度）
        profile = self.memory["layer6"]["profile"]
        profile_parts = []
        for field in PROFILE_FIELDS:
            field_attrs = []
            _collect_attrs(profile.get(field), field_attrs)
            if field_attrs:
                profile_parts.append(f"{field}:[{'; '.join(field_attrs)}]")
        if profile_parts:
            lines.append(f"[Profile] {' | '.join(profile_parts)}")

        # ② layer5 长期模式
        if self.memory["layer5"].get("summary"):
            lines.append(f"[Long-term] {self.memory['layer5']['summary']}")

        # ③ layer4 近期任务
        if self.memory["layer4"].get("summary"):
            lines.append(f"[Recent] {self.memory['layer4']['summary']}")

        # ④ layer7 索引细节（相关性排序，取够即停）
        extra = self.retrieve(people=people, location=location, activity=activity)
        if extra and extra != "No relevant memory found.":
            lines.append(extra)

        return "\n".join(lines) if len(lines) > 1 else "No previous memory found."

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
    # Backward-compatible shim (so old code calling memory.update() works)
    # ------------------------------------------------------------------

    def update(self, people: List[str], location: str, notes: Optional[str] = None):
        """Backward-compatible shim — stores a minimal moment."""
        self.add(
            scene=location,
            user_action="",
            needs=[],
            solutions=[],
            people=people,
            location=location,
            extra_notes=notes or ""
        )

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
