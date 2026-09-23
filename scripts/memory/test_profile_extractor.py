"""
test_profile_extractor.py — 09-07 画像抽取模块离线单测（纯 CPU）
==============================================================
验证 profile_extractor.py（画像字段 = 规格书 Part 2 骨架）的：
  1. parse_profile_output —— 解析 demographics/preferences/frequent_locations/behavior_patterns
  2. validate_profile —— 低置信度（<0.3）剔除
  3. 静态 vs 动态分存 —— 动态状态（情绪/专注度）不进画像字段
  4. parse_json_safe —— 非 JSON / markdown 代码块兜底
  5. agent._should_consolidate —— Phase C 触发条件
  6. prompt 模板 —— 规格书骨架字段 + 动态状态排除

用法：python test_profile_extractor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import profile_extractor as pe  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def section(t):
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. 完整字段解析（规格书骨架）
# ---------------------------------------------------------------------------

def test_full_parse():
    section("1. 完整字段解析（规格书 Part 2 骨架）")
    raw = '''
```json
{
  "demographics": {
    "name": {"value": "Richard", "confidence": 0.9, "evidence": "birthday guest"},
    "occupation": {"value": "frequent traveler", "confidence": 0.7, "evidence": "arriving at LAX"},
    "social_relations": [{"value": "traveling with family", "confidence": 0.6, "evidence": "family at airport"}]
  },
  "preferences": {
    "food": [{"value": "spicy", "confidence": 0.85, "evidence": "likes spicy food"}],
    "hobbies": [{"value": "photography", "confidence": 0.6, "evidence": "taking photos"}]
  },
  "frequent_locations": [
    {"value": "LAX", "confidence": 0.6, "evidence": "at LAX"},
    {"value": "airport lounges", "confidence": 0.2, "evidence": "unclear"}
  ],
  "behavior_patterns": {
    "with_surroundings": [{"value": "checks flight status on arrival", "confidence": 0.5, "evidence": "checks board"}],
    "with_ar_system": {
      "common_apps": [{"value": "navigation", "confidence": 0.6, "evidence": "uses AR nav"}],
      "typical_behaviors": [{"value": "follows step-by-step directions", "confidence": 0.6, "evidence": "wants clear guidance"}]
    },
    "with_agents": [{"value": "asks for directions", "confidence": 0.6, "evidence": "asks agent"}]
  },
  "status_inference": {"emotion": "anxious", "confidence": 0.8, "evidence": "irrelevant"}
}
```
'''
    p = pe.parse_profile_output(raw)
    check("解析成功", p is not None)

    # demographics（dict 型，含单值 AttrValue + list 型 social_relations）
    check("demographics.name 已解析", p["demographics"].get("name") is not None)
    check("demographics.occupation 已解析", p["demographics"].get("occupation") is not None)
    check("demographics.social_relations 是列表且含 1 项",
          isinstance(p["demographics"].get("social_relations"), list)
          and len(p["demographics"]["social_relations"]) == 1)

    # preferences（dict 型，含 list 型 food/hobbies）
    check("preferences.food 含 1 项",
          isinstance(p["preferences"].get("food"), list) and len(p["preferences"]["food"]) == 1)
    check("preferences.hobbies 含 1 项",
          isinstance(p["preferences"].get("hobbies"), list) and len(p["preferences"]["hobbies"]) == 1)

    # frequent_locations（list 型）
    check("frequent_locations 含 2 项", len(p["frequent_locations"]) == 2)

    # behavior_patterns（嵌套 dict）
    bp = p.get("behavior_patterns", {})
    check("behavior_patterns.with_surroundings 是列表",
          isinstance(bp.get("with_surroundings"), list) and len(bp["with_surroundings"]) == 1)
    check("behavior_patterns.with_ar_system.common_apps 是列表",
          isinstance(bp.get("with_ar_system", {}).get("common_apps"), list)
          and len(bp["with_ar_system"]["common_apps"]) == 1)
    check("behavior_patterns.with_agents 是列表",
          isinstance(bp.get("with_agents"), list) and len(bp["with_agents"]) == 1)

    # 动态状态不进画像
    check("status_inference 未被写入画像（动态状态排除）", "status_inference" not in p)

    # 元字段
    check("confidence 在 [0,1]", 0.0 <= p["preferences"]["food"][0]["confidence"] <= 1.0)
    check("evidence 已保留", bool(p["preferences"]["food"][0]["evidence"]))
    return p


# ---------------------------------------------------------------------------
# 2. 校验：低置信度剔除
# ---------------------------------------------------------------------------

def test_validate(p):
    section("2. 校验：低置信度（<0.3）剔除")
    cleaned, rejected = pe.validate_profile(p, min_confidence=0.3)
    # frequent_locations 第二项 confidence=0.2，应被剔除
    check("低置信度 frequent_locations(0.2) 被剔除", any("frequent_locations" in r for r in rejected),
          f"rejected={rejected}")
    check("剔除后 frequent_locations 仅 1 项", len(cleaned["frequent_locations"]) == 1)
    check("demographics(0.9) 保留", cleaned["demographics"].get("name") is not None)
    check("preferences.food(0.85) 保留", len(cleaned["preferences"]["food"]) == 1)
    return cleaned


# ---------------------------------------------------------------------------
# 3. 静态 vs 动态分存
# ---------------------------------------------------------------------------

def test_static_vs_dynamic():
    section("3. 静态 vs 动态分存（动态状态不进画像）")
    raw = '''
{
  "preferences": {"food": [{"value": "spicy", "confidence": 0.7, "evidence": "likes spicy"}]},
  "status_inference": {"emotion": "anxious", "confidence": 0.8, "evidence": "..."},
  "gaze_target": "boarding screen",
  "current_goal": "reach gate"
}
'''
    p = pe.parse_profile_output(raw)
    check("静态字段 preferences 已解析", p["preferences"]["food"] is not None)
    check("动态字段 status_inference 未进画像", "status_inference" not in p)
    check("动态字段 gaze_target 未进画像", "gaze_target" not in p)
    check("动态字段 current_goal 未进画像", "current_goal" not in p)


# ---------------------------------------------------------------------------
# 4. 解析兜底
# ---------------------------------------------------------------------------

def test_parse_fallback():
    section("4. 解析兜底")
    check("无围栏 JSON 可解析", pe.parse_json_safe('{"a": 1}') is not None)
    check("```json 围栏可解析", pe.parse_json_safe('```json\n{"a": 1}\n```') is not None)
    check("纯文本返回 None", pe.parse_json_safe("I am not json at all") is None)
    check("空串返回 None", pe.parse_json_safe("") is None)
    check("parse_profile_output 纯文本返回 None", pe.parse_profile_output("not json") is None)
    p = pe.parse_profile_output("{}")
    check("空对象解析成功但为空画像", p is not None and not pe.has_any_value(p))


# ---------------------------------------------------------------------------
# 5. Phase C 触发条件
# ---------------------------------------------------------------------------

def test_should_consolidate():
    section("5. Phase C 触发条件（直接测 agent.py 真实的 _should_consolidate）")
    from agent import VRAssistant

    class _FakeMemory:
        def __init__(self, total, last):
            self.memory = {"metadata": {"total_moments": total, "last_consolidation": last}}

    class _FakeSelf:
        def __init__(self, total, last):
            self.memory = _FakeMemory(total, last)

    should = VRAssistant._should_consolidate

    check("首次 total=1 不触发", should(_FakeSelf(1, None)) is False)
    check("首次 total=2 不触发", should(_FakeSelf(2, None)) is False)
    check("首次 total=3 触发", should(_FakeSelf(3, None)) is True)
    check("total=6 触发", should(_FakeSelf(6, "x")) is True)
    check("total=4 不触发", should(_FakeSelf(4, "x")) is False)
    check("total=5 不触发", should(_FakeSelf(5, "x")) is False)


# ---------------------------------------------------------------------------
# 6. prompt 模板
# ---------------------------------------------------------------------------

def test_prompt():
    section("6. prompt 模板（规格书骨架字段）")
    p = pe.build_profile_extraction_prompt("## PART 1\n- Location: LAX\n- Need 1: navigate")
    for f in pe.STABLE_FIELDS:
        check(f"prompt 含字段 {f}", f in p)
    check("prompt 含 preferences.food/hobbies 说明", "hobbies" in p)
    check("prompt 含 behavior_patterns 子字段 with_agents", "with_agents" in p)
    check("prompt 排除动态状态（emotion/attention）", "emotion" in p.lower() or "attention" in p)
    check("prompt 含分析文本", "LAX" in p)
    check("prompt 要求只输出 JSON", "only valid JSON" in p.lower() or "ONLY valid JSON" in p)


if __name__ == "__main__":
    p = test_full_parse()
    test_validate(p)
    test_static_vs_dynamic()
    test_parse_fallback()
    test_should_consolidate()
    test_prompt()

    print(f"\n{'=' * 60}")
    print(f"  通过 {PASS} / {PASS + FAIL}，失败 {FAIL}")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL else 0)
