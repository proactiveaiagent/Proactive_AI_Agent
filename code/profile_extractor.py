"""
profile_extractor.py — 用户画像抽取模块（Step 05）
=====================================================
画像字段以规格书 `proactive_ar_agent.md` Part 2 为唯一骨架，输出对齐
`04-0904-记忆模型与分层检索定稿.md` 2.4.3 节的 schema。

字段来源（规格书 Part 2）：

| 抽取字段            | 规格书出处 | 落层 |
|---|---|---|
| demographics        | L102 个人账号信息（姓名/年龄/性别/身份/居住区域/职业/受教育/社会关系） | layer6 |
| preferences         | L102（food / hobbies） | layer6 |
| frequent_locations  | L102 | layer6 |
| behavior_patterns   | L97 用户交互习惯（with_surroundings / with_ar_system / with_agents） | layer6 |

用户状态（情绪/专注度/视线等，规格书 L96 `status_inference`/`gaze_target`）是**动态量**，
不进 layer6 画像，由 moment 字段在 layer1~4 承载，故不在本模块输出字段内。

每个属性值的元字段对齐 04 文档 2.4.2 `AttrValue`，抽取阶段额外携带：
  value / confidence / evidence（供人工核对抽取质量）
写入阶段由 `update_profile()` 补充 source / timestamp / last_seen / observations。

本模块为纯逻辑（prompt 模板 + 解析 + 校验），不依赖 GPU，可离线单测。
LLM 调用由调用方注入（与 `/consolidate` 接口解耦）。
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 字段定义（规格书 Part 2 骨架）
# ---------------------------------------------------------------------------

STABLE_FIELDS = ["demographics", "preferences", "frequent_locations", "behavior_patterns"]

# 嵌套 dict 型字段（值是 {子键: AttrValue 或 [AttrValue] 或更深嵌套}）
DICT_FIELDS = {"demographics", "preferences", "behavior_patterns"}

# 低置信度阈值（对齐 04 文档 2.5：confidence < 0.3 不入库）
DEFAULT_MIN_CONFIDENCE = 0.3


def empty_profile() -> Dict:
    """返回空画像：dict 字段为 {}，list 字段为 []。"""
    profile: Dict = {}
    for f in STABLE_FIELDS:
        profile[f] = {} if f in DICT_FIELDS else []
    return profile


# ---------------------------------------------------------------------------
# Prompt 模板（规格书字段）
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are a user profiling system. Given the agent's scene/need analysis of a short user interaction, extract the user's LONG-TERM PROFILE. The profile fields follow a fixed schema (do NOT add fields outside it):

- demographics: identity attributes (name, age, gender, identity, living_region, occupation, education, social_relations)
- preferences: {{ "food": [...], "hobbies": [...] }}
- frequent_locations: [ ... ]
- behavior_patterns:
    - with_surroundings: [ ... ]  (typical habits interacting with the physical environment)
    - with_ar_system: {{ "common_apps": [...], "typical_behaviors": [...] }}
    - with_agents: [ ... ]  (historical patterns of interacting with AI agents)

DO NOT extract transient state (current emotion, attention focus, gaze, current goal/intention) — those are NOT long-term profile and are handled elsewhere.

RULES:
1. Only output a field if you have EVIDENCE from the analysis. Do NOT hallucinate.
2. Each attribute value carries "confidence" (0.0-1.0) and "evidence" (short quote/paraphrase from the analysis).
3. A first-time / single observation should have LOW confidence (< 0.5); repeated evidence raises confidence.
4. OPTIONAL: each attribute may carry "decay_type" to indicate its temporal nature:
   - "stable"   : identity / preference (gender, name, taste) — long-lasting or permanent
   - "decaying" : life-stage / state (e.g. occupation, "currently a student") — fades over time
   - "deadline" : time-limited (coupon, membership expiry) — MUST also give "expires_at" in ISO format
   If unsure, OMIT "decay_type" (the system assigns a default by field).
5. Output ONLY valid JSON (no markdown, no code fences).

OUTPUT FORMAT:
{{
  "demographics": {{ "name": {{"value":"...","confidence":0.9,"evidence":"..."}}, "age": {{...}}, "social_relations": [{{"value":"...","confidence":0.6,"evidence":"..."}}] }},
  "preferences": {{ "food": [{{"value":"spicy","confidence":0.85,"evidence":"..."}}], "hobbies": [{{"value":"...","confidence":0.6,"evidence":"..."}}] }},
  "frequent_locations": [{{"value":"...","confidence":0.6,"evidence":"..."}}],
  "behavior_patterns": {{
    "with_surroundings": [{{"value":"...","confidence":0.6,"evidence":"..."}}],
    "with_ar_system": {{ "common_apps": [{{"value":"...","confidence":0.6,"evidence":"..."}}], "typical_behaviors": [{{"value":"...","confidence":0.6,"evidence":"..."}}] }},
    "with_agents": [{{"value":"...","confidence":0.6,"evidence":"..."}}]
  }}
}}

USER INTERACTION ANALYSIS:
{analysis_text}"""


def build_profile_extraction_prompt(analysis_text: str) -> str:
    """基于 Phase A 的 Part1-3 分析文本，构造画像抽取 prompt。"""
    return _PROMPT_TEMPLATE.format(analysis_text=analysis_text.strip())


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------

def parse_json_safe(text: str) -> Optional[dict]:
    """多重策略从 LLM 输出中提取 JSON。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# 解析 + 校验
# ---------------------------------------------------------------------------

def _normalise_attr(item: Any) -> Optional[Dict]:
    """把单个 AttrValue 规范化；value 为空则返回 None。"""
    if not isinstance(item, dict):
        return None
    value = item.get("value")
    if value is None or value == "":
        return None
    try:
        conf = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {
        "value": value,
        "confidence": conf,
        "evidence": str(item.get("evidence", ""))[:200],
        "source": item.get("source", "llm_inference"),
    }


def _parse_node(node: Any) -> Any:
    """递归解析任意嵌套结构：AttrValue / [AttrValue] / {子键: 递归值}。"""
    if isinstance(node, list):
        out = []
        for item in node:
            a = _normalise_attr(item)
            if a is not None:
                out.append(a)
        return out
    if isinstance(node, dict):
        # 是 AttrValue（含 "value" 键）还是嵌套 dict？
        if "value" in node:
            return _normalise_attr(node)
        out = {}
        for k, v in node.items():
            parsed = _parse_node(v)
            if parsed not in (None, [], {}):
                out[k] = parsed
        return out
    return None


def parse_profile_output(raw_text: str) -> Optional[Dict]:
    """把 LLM 抽取输出解析为对齐 04 文档 2.4.3 的结构化画像。None 表示解析失败。"""
    data = parse_json_safe(raw_text)
    if not isinstance(data, dict):
        return None

    profile = empty_profile()
    for f in STABLE_FIELDS:
        src = data.get(f)
        if src is None:
            continue
        if f in DICT_FIELDS:
            parsed = _parse_node(src)
            if isinstance(parsed, dict):
                profile[f] = parsed
        else:
            parsed = _parse_node(src)
            if isinstance(parsed, list):
                profile[f] = parsed
    return profile


def _filter_node(node: Any, min_confidence: float) -> Tuple[Any, int]:
    """递归过滤低置信度项，返回 (过滤后节点, 被剔除项数)。"""
    if isinstance(node, list):
        kept = [a for a in node if isinstance(a, dict) and a["confidence"] >= min_confidence]
        return kept, len(node) - len(kept)
    if isinstance(node, dict):
        if "value" in node:  # AttrValue 叶子
            keep = node["confidence"] >= min_confidence
            return (node if keep else None), (0 if keep else 1)
        out = {}
        dropped = 0
        for k, v in node.items():
            fv, d = _filter_node(v, min_confidence)
            dropped += d
            if fv not in (None, [], {}):
                out[k] = fv
        return out, dropped
    return None, 0


def validate_profile(profile: Dict,
                     min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                     ) -> Tuple[Dict, List[str]]:
    """
    校验并清洗画像（对齐 04 文档 2.5）：
    - 递归剔除 confidence < min_confidence 的项
    - 返回 (清洗后画像, 被剔除字段名列表)
    """
    cleaned = empty_profile()
    rejected: List[str] = []

    for f in STABLE_FIELDS:
        node = profile.get(f)
        if node is None:
            continue
        filtered, dropped = _filter_node(node, min_confidence)
        if dropped > 0:
            rejected.append(f"{f}({dropped})")
        if filtered not in (None, [], {}):
            cleaned[f] = filtered

    return cleaned, rejected


def _count_node(node: Any) -> int:
    """递归统计非空 AttrValue 数量。"""
    if isinstance(node, list):
        return sum(1 for a in node if isinstance(a, dict) and "value" in a)
    if isinstance(node, dict):
        if "value" in node:
            return 1
        return sum(_count_node(v) for v in node.values())
    return 0


def has_any_value(profile: Dict) -> bool:
    """画像里是否至少有一个非空属性。"""
    return sum(_count_node(profile.get(f)) for f in STABLE_FIELDS) > 0


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

def extract_profile_from_raw(raw_llm_text: str) -> Optional[Dict]:
    """一步完成：解析 → 校验。返回清洗后的画像，或 None（解析失败）。"""
    parsed = parse_profile_output(raw_llm_text)
    if parsed is None:
        return None
    cleaned, _rejected = validate_profile(parsed)
    return cleaned
