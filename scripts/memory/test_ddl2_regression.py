"""
test_ddl2_regression.py — DDL 2 全量回归测试套件（覆盖 09-03 实证缺陷）
======================================================================
验证对象：
  1. G7 缺陷回归：Layer 7 索引防污染与清洗验证（人名/地点/占位符过滤）
  2. D1 缺陷回归：Phase C daemon 线程与触发条件（保证画像能真正生成）
  3. D2 缺陷回归：Phase A 检索接入（get_context_for_analysis 替换全量 dump）
  4. D3 缺陷回归：画像增量更新不丢历史（列表追加去重 + 冲突裁决 + 职责边界）
  5. G1 缺陷回归：中文检索 0 命中（bigram 分词与命中）
  6. G5 缺陷回归：retrieve 相关性排序与 Top-K（替代硬截断）
  7. G6 缺陷回归：新画像 schema 注入 get_context_for_analysis（修复旧字段检查 bug）
  8. G7 缺陷回归：agent.py parse_analysis 非实体描述过滤

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python test_ddl2_regression.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import memory as M  # noqa: E402
from memory import (  # noqa: E402
    PersonMemory,
    _is_valid_person_tag,
    _is_valid_index_tag,
    _clean_layer7_indices,
    tokenize,
    effective_confidence,
)
from agent import VRAssistant  # noqa: E402
from profile_extractor import validate_profile, parse_profile_output  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def section(title: str):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


def run_all_tests():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    # -----------------------------------------------------------------------
    section("1. G7 缺陷回归：Layer 7 索引防污染与清洗")
    # -----------------------------------------------------------------------
    # 1.1 _is_valid_person_tag 判定
    check("合法中文人名识别 (张三)", _is_valid_person_tag("张三"))
    check("合法英文人名识别 (Alice)", _is_valid_person_tag("Alice"))
    check("合法社会角色识别 (服务员)", _is_valid_person_tag("服务员"))
    check("占位符 canonical_name 被拒绝", not _is_valid_person_tag("canonical_name"))
    check("否定词 'None' 被拒绝", not _is_valid_person_tag("None"))
    check("否定词 'No identifiable individuals' 被拒绝", not _is_valid_person_tag("No identifiable individuals"))
    check("场景描述长句被拒绝", not _is_valid_person_tag("A server in a blue uniform hands over the bill. Customers are seated at tables"))
    check("动词分句被拒绝 ('Multiple travelers visible')", not _is_valid_person_tag("Multiple travelers and staff visible"))

    # 1.2 索引清洗函数 _clean_layer7_indices
    dirty_l7 = {
        "people": {
            "Alice": ["m1"],
            "canonical_name": ["m1", "m2"],
            "No identifiable individuals visible.": ["m2"],
            "Multiple travelers and airport staff visible": ["m3"],
        },
        "locations": {
            "LAX Airport": ["m1"],
            "canonical_location": ["m2"],
        },
        "activity_events": {
            "boarding": ["m1"],
            "event_tag": ["m2"],
        },
        "time_nodes": {"2026-09-10": ["m1"]},
    }
    removed = _clean_layer7_indices(dirty_l7)
    check("people 脏键被全部清除", set(dirty_l7["people"].keys()) == {"Alice"})
    check("locations 占位符被清除", set(dirty_l7["locations"].keys()) == {"LAX Airport"})
    check("activity_events 占位符被清除", set(dirty_l7["activity_events"].keys()) == {"boarding"})
    check("清洗统计正确记录", "canonical_name" in removed["people"] and "canonical_location" in removed["locations"])

    # 1.3 PersonMemory 增量写入防污染
    tmpdir = tempfile.mkdtemp(prefix="test_ddl2_g7_")
    try:
        mem = PersonMemory(memory_dir=tmpdir)
        m_id = mem.add(
            scene="Airport gate",
            user_action="Waiting for flight",
            needs=[{"need": "Check flight", "confidence": 0.9}],
            solutions=[{"solution": "Look at screen", "output_type": "digital_info", "action": "Look"}],
            people=["Bob", "No identifiable individuals visible", "canonical_name"],
            location="LAX Terminal 4",
            activity="Flight waiting",
        )
        check("合法实体 Bob 写入 people 索引", "Bob" in mem.memory["layer7"]["people"])
        check("污染描述不进入 people 索引", "No identifiable individuals visible" not in mem.memory["layer7"]["people"])
        check("占位符不进入 people 索引", "canonical_name" not in mem.memory["layer7"]["people"])

        # sort 防污染
        mem.sort({
            "people": {
                "Charlie": [m_id],
                "canonical_name": [m_id],
                "Unknown passenger walking past": [m_id],
            },
            "locations": {
                "Gate 42B": [m_id],
                "canonical_location": [m_id],
            }
        })
        check("sort 中合法姓名 Charlie 写入", "Charlie" in mem.memory["layer7"]["people"])
        check("sort 中占位符 canonical_name 拒绝写入", "canonical_name" not in mem.memory["layer7"]["people"])
        check("sort 中长句描述拒绝写入", "Unknown passenger walking past" not in mem.memory["layer7"]["people"])
        check("sort 中合法地点 Gate 42B 写入", "Gate 42B" in mem.memory["layer7"]["locations"])
        check("sort 中占位符 canonical_location 拒绝写入", "canonical_location" not in mem.memory["layer7"]["locations"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("2. D1 缺陷回归：Phase C 线程模式与触发条件")
    # -----------------------------------------------------------------------
    agent_dummy = object.__new__(VRAssistant)
    agent_dummy.memory = PersonMemory(memory_dir=tempfile.mkdtemp(prefix="test_d1_"))
    try:
        agent_dummy.memory.memory["metadata"]["total_moments"] = 2
        agent_dummy.memory.memory["metadata"]["last_consolidation"] = None
        check("total=2 且未整理时不触发", not agent_dummy._should_consolidate())

        agent_dummy.memory.memory["metadata"]["total_moments"] = 3
        check("total=3 首次积累达标触发", agent_dummy._should_consolidate())

        agent_dummy.memory.memory["metadata"]["last_consolidation"] = "2026-09-10T10:00:00"
        agent_dummy.memory.memory["metadata"]["total_moments"] = 4
        check("有历史且 total=4 不触发", not agent_dummy._should_consolidate())

        agent_dummy.memory.memory["metadata"]["total_moments"] = 6
        check("有历史且 total=6 周期达标触发", agent_dummy._should_consolidate())
    finally:
        shutil.rmtree(agent_dummy.memory.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("3. D2 缺陷回归：Phase A 检索接口（get_context_for_analysis）")
    # -----------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="test_d2_")
    try:
        mem = PersonMemory(memory_dir=tmpdir)
        # 空库冷启动
        ctx_empty = mem.get_context_for_analysis("user", "library")
        check("冷启动不注入画像段落", "[Profile]" not in ctx_empty)
        check("冷启动给出未发现记忆提示", "No previous memory found" in ctx_empty)

        # 写入画像与记忆
        mem.update_profile({
            "demographics": {"name": {"value": "李华", "confidence": 0.95}},
            "preferences": {"food": [{"value": "清淡", "confidence": 0.8}]},
            "frequent_locations": [{"value": "市图书馆", "confidence": 0.85}],
            "behavior_patterns": {
                "with_ar_system": {
                    "common_apps": [{"value": "电子书阅读", "confidence": 0.8}]
                }
            }
        })
        mem.memory["layer4"]["summary"] = "正在备考计算机二级"
        mem.memory["layer5"]["summary"] = "长期在周末前往图书馆自主学习"
        mem.add(
            scene="市图书馆三楼自习室",
            user_action="阅读教材",
            needs=[{"need": "寻找参考书", "confidence": 0.9}],
            solutions=[{"solution": "检索馆藏", "output_type": "digital_info", "action": "查书"}],
            location="市图书馆",
            activity="学习",
        )

        ctx = mem.get_context_for_analysis("李华", "市图书馆")
        check("上下文包含 [Profile] 段", "[Profile]" in ctx)
        check("画像姓名注入成功", "李华" in ctx)
        check("画像偏好注入成功", "清淡" in ctx)
        check("画像常去地点注入成功", "市图书馆" in ctx)
        check("层级4摘要注入成功", "正在备考计算机二级" in ctx)
        check("层级5长期模式注入成功", "长期在周末前往图书馆" in ctx)
        check("层级7索引细节检索命中", "市图书馆三楼自习室" in ctx)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("4. D3 缺陷回归：画像增量更新不丢历史与职责边界")
    # -----------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="test_d3_")
    try:
        mem = PersonMemory(memory_dir=tmpdir)
        # 第一次写入 hobbies: 摄影
        mem.update_profile({
            "preferences": {"hobbies": [{"value": "摄影", "confidence": 0.8, "evidence": "携带单反"}]}
        })
        # 第二次写入 hobbies: 徒步
        mem.update_profile({
            "preferences": {"hobbies": [{"value": "徒步", "confidence": 0.75, "evidence": "穿登山鞋"}]}
        })
        profile = mem.get_profile()
        hobby_vals = [h["value"] for h in profile["preferences"]["hobbies"]]
        check("列表追加去重保留历史 摄影", "摄影" in hobby_vals)
        check("列表追加去重包含新值 徒步", "徒步" in hobby_vals)
        check("hobbies 总数累积为 2（未被浅覆盖）", len(hobby_vals) == 2)

        # 单值冲突裁决：频次打平 -> 最近优先
        mem.update_profile({
            "demographics": {"age": {"value": 25, "confidence": 0.7, "timestamp": "2026-09-01T10:00:00"}}
        })
        mem.update_profile({
            "demographics": {"age": {"value": 26, "confidence": 0.7, "timestamp": "2026-09-10T10:00:00"}}
        })
        prof = mem.get_profile()
        check("冲突裁决最新值胜出 (26)", prof["demographics"]["age"]["value"] == 26)
        check("败者记录于 history", len(prof["demographics"]["age"].get("history", [])) == 1)

        # compress 职责边界：compress 不修改 profile
        mem.update_profile({
            "demographics": {"name": {"value": "王五", "confidence": 0.95}}
        })
        mem.compress({
            "layer6": {
                "summary": "长期用户总结",
                "profile": {"demographics": {"name": "被错误覆盖的名字"}}
            }
        })
        check("compress 不覆盖 layer6.profile", mem.get_profile()["demographics"]["name"]["value"] == "王五")
        check("compress 正常更新 summary", mem.memory["layer6"]["summary"] == "长期用户总结")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("5. G1 缺陷回归：中文分词与检索命中（09-03 实证 0 命中回归）")
    # -----------------------------------------------------------------------
    tokens_cn = tokenize("在图书馆学习")
    check("中文 bigram 分词包含 '图书'", "图书" in tokens_cn)
    check("中文 bigram 分词包含 '学习'", "学习" in tokens_cn)

    tmpdir = tempfile.mkdtemp(prefix="test_g1_")
    try:
        mem = PersonMemory(memory_dir=tmpdir)
        mem.add(
            scene="在图书馆学习人工智能",
            user_action="看书笔记",
            needs=[{"need": "查阅文献", "confidence": 0.9}],
            solutions=[{"solution": "打开论文库", "output_type": "digital_info", "action": "查阅"}],
            location="大学图书馆",
            activity="学习研究",
        )
        res_cn = mem.query("在图书馆学习")
        check("中文 query 成功命中 (>=1 条)", len(res_cn) >= 1)
        check("命中内容包含目标场景", "在图书馆学习人工智能" in res_cn[0]["scene"])

        res_en = mem.query("study in library")
        check("英文 query 保持命中兼容", len(res_en) >= 1)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("6. G5 缺陷回归：retrieve 相关性打分排序 + Top-K 截断")
    # -----------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="test_g5_")
    try:
        mem = PersonMemory(memory_dir=tmpdir)
        # 存入不同匹配度的 moment
        mem.add(
            scene="机场航站楼办理登机牌",
            user_action="排队",
            needs=[{"need": "托运行李", "confidence": 0.8}],
            solutions=[{"solution": "提示柜台", "output_type": "digital_info", "action": "排队"}],
            location="洛杉矶国际机场",
            activity="机场登机",
        )
        mem.add(
            scene="机场贵宾厅候机喝咖啡",
            user_action="休息",
            needs=[{"need": "登机广播", "confidence": 0.8}],
            solutions=[{"solution": "闹钟提醒", "output_type": "digital_info", "action": "提醒"}],
            location="洛杉矶国际机场",
            activity="休息候机",
        )
        mem.add(
            scene="酒店大堂办理入住",
            user_action="等待前台",
            needs=[{"need": "房卡", "confidence": 0.8}],
            solutions=[{"solution": "出示预订码", "output_type": "digital_info", "action": "出示"}],
            location="希尔顿酒店",
            activity="入住",
        )

        query_results = mem.query("洛杉矶国际机场", top_k=2)
        check("query Top-K=2 返回 2 条记录", len(query_results) == 2)
        locations_retrieved = [m.get("location") for m in query_results]
        check("返回记录均为相关机场", all(l == "洛杉矶国际机场" for l in locations_retrieved))
        check("无关酒店记录被排出在 Top-K 外", "希尔顿酒店" not in locations_retrieved)

        retrieve_str = mem.retrieve(location="洛杉矶国际机场", top_k=2)
        check("retrieve 返回格式化文本", "Scene: 机场" in retrieve_str)
        check("retrieve 不含酒店", "希尔顿酒店" not in retrieve_str)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("7. G6 缺陷回归：新 Schema 字段注入 get_context_for_analysis")
    # -----------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="test_g6_")
    try:
        mem = PersonMemory(memory_dir=tmpdir)
        mem.update_profile({
            "demographics": {
                "name": {"value": "张同学", "confidence": 0.9},
                "occupation": {"value": "算法工程师", "confidence": 0.85},
            },
            "behavior_patterns": {
                "with_ar_system": {
                    "common_apps": [{"value": "代码审查助手", "confidence": 0.8}],
                    "typical_behaviors": [{"value": "注视代码后语音呼叫", "confidence": 0.75}],
                },
                "with_agents": [{"value": "偏好简短主动提醒", "confidence": 0.85}],
            }
        })
        ctx = mem.get_context_for_analysis("张同学", "实验室")
        check("demographics 字段成功展平注入", "算法工程师" in ctx)
        check("behavior_patterns common_apps 注入", "代码审查助手" in ctx)
        check("behavior_patterns with_agents 注入", "偏好简短主动提醒" in ctx)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("8. G7 缺陷回归：agent.py parse_analysis 过滤非实体语句")
    # -----------------------------------------------------------------------
    sample_analysis_text = """
- Location: Los Angeles International Airport (LAX)
- People: Multiple travelers and airport staff visible, but no identifiable individuals, Alice, 服务员
- User Action: walking toward gate 42
Need 1: find nearest restroom (confidence: 0.85)
Solution 1: turn left at duty-free shop
"""
    agent_inst = object.__new__(VRAssistant)
    parsed = agent_inst.parse_analysis(sample_analysis_text)
    check("parse_analysis 提取地点正确", parsed["location"] == "Los Angeles International Airport (LAX)")
    check("过滤掉描述性长句和否定短语", "Multiple travelers and airport staff visible" not in parsed["people"])
    check("过滤掉否定子句", "but no identifiable individuals" not in parsed["people"])
    check("保留真实人名 Alice", "Alice" in parsed["people"])
    check("保留真实角色 服务员", "服务员" in parsed["people"])
    check("最终 people 列表仅含有效实体", set(parsed["people"]) == {"Alice", "服务员"})

    # -----------------------------------------------------------------------
    section("测试结果汇总")
    # -----------------------------------------------------------------------
    print(f"总测试项: {PASS + FAIL}")
    print(f"通过: {PASS}")
    print(f"失败: {FAIL}")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
