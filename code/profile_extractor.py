"""
profile_extractor.py — 用户画像抽取模块（Step 05）
=====================================================
基于 Mental World Modeling (MWM) 论文（arXiv:2607.27201v1）的 9 个心理变量
指导 LLM 抽取，但**输出结构对齐 `04-0904-记忆模型与分层检索定稿.md` §5.3 定稿的
语义化 schema**，以便 Step 06 的 `update_profile()` 直接消费。

字段与 MWM 变量的对应（04 §2.4 / §5.3）：

| 抽取字段            | MWM 变量 | 稳定性 | 落层 |
|---|---|---|---|
| demographics        | id       | 稳定   | layer6 |
| preferences         | d        | 稳定   | layer6 |
| hobbies             | d        | 稳定   | layer6 |
| personality         | d        | 稳定   | layer6 |
| routines            | d + n    | 稳定   | layer6 |
| characteristics     | d + c    | 稳定   | layer6 |
| constraints         | c        | 稳定   | layer6 |
| frequent_locations  | n / α    | 稳定   | layer6 |
| beliefs             | b        | 中期   | layer4/5（衰减更快） |

每个属性值的元字段对齐 04 §5.2 `AttrValue`，抽取阶段额外携带：
  value / confidence / evidence（供人工核对抽取质量）
写入阶段由 `update_profile()` 补充 source / timestamp / last_seen / observations。

q（注意焦点）/ g（目标）/ iota（意图）/ e（情绪）为瞬时量，**不进 layer6.profile**
（落 layer1~5，由 compress 处理），故不在本模块输出字段内 —— 仅在 prompt 中
说明它们属于"瞬时、不写入长期画像"，以约束 LLM 不要误抽。

本模块为纯逻辑（prompt 模板 + 解析 + 校验），不依赖 GPU，可离线单测。
LLM 调用由调用方注入（与 `/consolidate` 接口解耦）。
"""

import json
import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 字段定义（对齐 04 §5.3）
# ---------------------------------------------------------------------------

# 稳定字段（落 layer6，累积，低衰减）
STABLE_FIELDS = [
    "demographics",
    "preferences",
    "hobbies",
    "personality",
    "routines",
    "characteristics",
    "constraints",
    "frequent_locations",
]
# 中期字段（b 信念，落 layer4/5，衰减更快）
DYNAMIC_FIELDS = ["beliefs"]

ALL_FIELDS = STABLE_FIELDS + DYNAMIC_FIELDS

# 嵌套 dict 型字段（值不是列表，而是 {子键: AttrValue 或 AttrValue[]}）
DICT_FIELDS = {"demographics", "preferences"}

# 低置信度阈值（对齐 04 §6.3：confidence < 0.3 不入库）
DEFAULT_MIN_CONFIDENCE = 0.3


def empty_profile() -> Dict:
    """返回空画像：每个字段初始化为空（dict 字段为 {}，list 字段为 []）。"""
    profile: Dict = {}
    for f in ALL_FIELDS:
        profile[f] = {} if f in DICT_FIELDS else []
    return profile


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are a user profiling system. Given the agent's scene/need analysis of a short user interaction, extract the user's PROFILE following the Mental World Modeling (MWM) framework.

MWM says a user's mental state has 9 variables. For this profiling task, ONLY extract the ones that belong in a LONG-TERM profile:
- id  : identity attributes (age, gender, occupation, living region) -> demographics
- d   : dispositions (stable personality traits and preferences) -> preferences / hobbies / personality
- d+n : routines (recurring daily/weekly patterns) -> routines
- d+c : characteristics (stable traits + behavioral limits) -> characteristics
- c   : constraints (allergy, dietary, budget, mobility) -> constraints
- n/a : frequently visited places -> frequent_locations
- b   : beliefs (what the user currently believes, MAY BE WRONG) -> beliefs

DO NOT extract these transient variables (they are NOT long-term profile, handled elsewhere):
- q (attention focus), g (goals), iota (intentions), e (emotions)

RULES:
1. Only output a field if you have EVIDENCE. Do NOT hallucinate.
2. STABLE fields (all except beliefs) require REPEATED evidence — if this is a single/first observation, set confidence LOW (< 0.5).
3. beliefs (b) are INFERENCES that may be wrong; do NOT merge them into personality/preferences (d).
4. Every attribute value carries "confidence" (0.0-1.0) and "evidence" (short quote/paraphrase from the analysis).
5. Output ONLY valid JSON (no markdown, no code fences).

OUTPUT FORMAT:
{{
  "demographics": {{ "name": {{"value":"...","confidence":0.9,"evidence":"..."}}, "age": {{...}}, "occupation": {{...}}, "living_region": {{...}} }},
  "preferences": {{ "food": [{{"value":"spicy","confidence":0.85,"evidence":"..."}}], "music": [{{...}}] }},
  "hobbies":             [{{"value":"...","confidence":0.6,"evidence":"..."}}],
  "personality":         [{{"value":"...","confidence":0.6,"evidence":"..."}}],
  "routines":            [{{"value":"...","confidence":0.6,"evidence":"..."}}],
  "characteristics":     [{{"value":"...","confidence":0.6,"evidence":"..."}}],
  "constraints":         [{{"value":"...","confidence":0.6,"evidence":"..."}}],
  "frequent_locations":  [{{"value":"...","confidence":0.6,"evidence":"..."}}],
  "beliefs":             [{{"value":"...","confidence":0.7,"evidence":"..."}}]
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

def _normalise_attr(item: Dict, field: str) -> Optional[Dict]:
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
    attr = {
        "value": value,
        "confidence": conf,
        "evidence": str(item.get("evidence", ""))[:200],
    }
    # 抽取阶段补充 source 占位（Step 06 的 update_profile 会用真实 moment_id 覆盖）
    attr["source"] = item.get("source", "llm_inference")
    return attr


def _parse_attr_list(src, field: str) -> List[Dict]:
    """解析一个 list 型字段（hobbies/personality/...），返回 AttrValue 列表。"""
    if not isinstance(src, list):
        return []
    out = []
    for item in src:
        a = _normalise_attr(item, field)
        if a is not None:
            out.append(a)
    return out


def _parse_dict_field(src, field: str) -> Dict:
    """解析嵌套 dict 型字段（demographics/preferences）。
    demographics: {子键: AttrValue}
    preferences:  {子键: [AttrValue]}
    """
    if not isinstance(src, dict):
        return {}
    out = {}
    for k, v in src.items():
        if isinstance(v, list):
            lst = _parse_attr_list(v, field)
            if lst:
                out[k] = lst
        elif isinstance(v, dict):
            a = _normalise_attr(v, field)
            if a is not None:
                out[k] = a
    return out


def parse_profile_output(raw_text: str) -> Optional[Dict]:
    """把 LLM 抽取输出解析为对齐 04 §5.3 的结构化画像。None 表示解析失败。"""
    data = parse_json_safe(raw_text)
    if not isinstance(data, dict):
        return None

    profile = empty_profile()
    for f in STABLE_FIELDS + DYNAMIC_FIELDS:
        src = data.get(f)
        if f in DICT_FIELDS:
            profile[f] = _parse_dict_field(src, f)
        else:
            profile[f] = _parse_attr_list(src, f)
    return profile


def _count_attrs(profile: Dict) -> int:
    """统计画像里非空的 AttrValue 数量（用于判空）。"""
    n = 0
    for f in ALL_FIELDS:
        v = profile.get(f)
        if f in DICT_FIELDS:
            for sub in v.values():
                if isinstance(sub, list):
                    n += len(sub)
                else:
                    n += 1
        else:
            n += len(v)
    return n


def validate_profile(profile: Dict,
                     min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                     ) -> Tuple[Dict, List[str]]:
    """
    校验并清洗画像（对齐 04 §6.3）：
    - 剔除 confidence < min_confidence 的项
    - 返回 (清洗后画像, 被剔除字段名列表)
    """
    cleaned = empty_profile()
    rejected: List[str] = []

    for f in ALL_FIELDS:
        src = profile.get(f, {} if f in DICT_FIELDS else [])
        if f in DICT_FIELDS:
            for k, v in src.items():
                if isinstance(v, list):
                    keep = [a for a in v if a["confidence"] >= min_confidence]
                    if keep:
                        cleaned[f][k] = keep
                    else:
                        rejected.append(f"{f}.{k}")
                else:
                    if v["confidence"] >= min_confidence:
                        cleaned[f][k] = v
                    else:
                        rejected.append(f"{f}.{k}")
        else:
            keep = [a for a in src if a["confidence"] >= min_confidence]
            dropped = len(src) - len(keep)
            if dropped > 0:
                rejected.append(f"{f}({dropped})")
            cleaned[f] = keep

    return cleaned, rejected


def has_any_value(profile: Dict) -> bool:
    """画像里是否至少有一个非空属性。"""
    return _count_attrs(profile) > 0


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
