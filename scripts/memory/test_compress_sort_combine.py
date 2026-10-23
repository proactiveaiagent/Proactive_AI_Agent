"""
test_compress_sort_combine.py — DDL 3 压缩/归类/合并操作回归测试（9.17）
====================================================================
验证对象（设计说明书 §4.4 定稿）：
  1. compress：写 layer4/5/6 文本摘要；layer6.profile 跳过；list 精确去重；幂等
  2. sort：索引归类；非法 tag 防 G7 污染；悬挂 moment_id 过滤；同键去重
  3. combine：近义索引键合并 + moments 主存储引用同步（people/locations/activity）；幂等

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python test_compress_sort_combine.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import memory as M  # noqa: E402
from memory import PersonMemory  # noqa: E402

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


def make_mem():
    return PersonMemory(memory_dir=tempfile.mkdtemp(prefix="test_compress_sort_combine_"))


def add_moment(mem, people=None, location=None, activity=None):
    return mem.add(
        scene="测试场景",
        user_action="测试动作",
        needs=[{"need": "测试需求", "confidence": 0.9}],
        solutions=[{"solution": "测试方案", "output_type": "digital_info", "action": "测试"}],
        people=people or [],
        location=location or "",
        activity=activity or "",
    )


def run_all_tests():
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    # -----------------------------------------------------------------------
    section("1. compress：写 layer4/5/6 文本摘要")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        mem.compress({
            "layer4": {"summary": "最近在学 Rust", "current_tasks": ["学 Rust"], "life_trajectory": "转码中"},
            "layer5": {"summary": "关键事件", "key_events": ["换工作"], "long_term_patterns": "爱学习"},
            "layer6": {"summary": "用户画像摘要", "profile": {"demographics": {"name": "张三"}}},
        })
        check("layer4.summary 写入", mem.memory["layer4"]["summary"] == "最近在学 Rust",
              f"got {mem.memory['layer4']['summary']!r}")
        check("layer4.current_tasks 写入", mem.memory["layer4"]["current_tasks"] == ["学 Rust"])
        check("layer5.summary 写入", mem.memory["layer5"]["summary"] == "关键事件")
        check("layer5.key_events 写入", mem.memory["layer5"]["key_events"] == ["换工作"])
        check("layer6.summary 写入", mem.memory["layer6"]["summary"] == "用户画像摘要")

        # layer6.profile 必须跳过（职责边界：画像走 update_profile）
        # compress 传入的 name 不应出现在 profile 里
        check("layer6.profile 被跳过（不直接写）",
              "name" not in mem.memory["layer6"]["profile"].get("demographics", {}),
              f"profile={mem.memory['layer6'].get('profile')!r}")
        check("last_consolidation 更新", mem.memory["metadata"]["last_consolidation"] is not None)
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("2. compress：list 精确去重（str + dict 元素）")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        # 先写入初始 list
        mem.compress({"layer4": {"current_tasks": ["学 Rust", {"task": "读论文", "due": "周五"}]}})
        # 再次 compress，包含重复 + 新元素
        mem.compress({"layer4": {"current_tasks": ["学 Rust", {"task": "读论文", "due": "周五"}, "学英语"]}})
        tasks = mem.memory["layer4"]["current_tasks"]
        check("list 去重后长度正确（2 旧 + 1 新）", len(tasks) == 3, f"got {tasks!r}")
        check("重复 str 元素只保留一份", tasks.count("学 Rust") == 1, f"got {tasks!r}")
        check("重复 dict 元素只保留一份",
              sum(1 for x in tasks if isinstance(x, dict) and x.get("task") == "读论文") == 1,
              f"got {tasks!r}")
        check("新元素追加", "学英语" in tasks)
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("3. compress：空 summary 跳过 + 幂等")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        mem.compress({"layer4": {"summary": "第一次"}})
        mem.compress({"layer4": {"summary": ""}})  # 空 summary 不应覆盖
        check("空 summary 不覆盖已有值", mem.memory["layer4"]["summary"] == "第一次",
              f"got {mem.memory['layer4']['summary']!r}")

        # 幂等：重复 compress 相同内容，list 不重复
        mem.compress({"layer4": {"current_tasks": ["任务A"]}})
        mem.compress({"layer4": {"current_tasks": ["任务A"]}})
        check("重复 compress 幂等（list 不重复）",
              mem.memory["layer4"]["current_tasks"].count("任务A") == 1,
              f"got {mem.memory['layer4']['current_tasks']!r}")
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("4. sort：索引归类 + 防污染 + 悬挂过滤 + 去重")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        m1 = add_moment(mem, people=["张三"], location="图书馆", activity="学习")
        m2 = add_moment(mem, people=["张三", "李四"], location="图书馆", activity="学习")

        mem.sort({
            "people": {
                "张三": [m1, m2],
                "canonical_name": [m1],           # 占位符 → 拒绝
                "Unknown person walking past": [m1],  # 长句描述 → 拒绝
            },
            "locations": {
                "图书馆": [m1, m2],
                "canonical_location": [m1],       # 占位符 → 拒绝
            },
            "activity_events": {
                "学习": [m1, m2],
                "event_tag": [m1],                # 占位符 → 拒绝
            },
            "time_nodes": {
                "2026-09-17": [m1, m2],
            },
        })
        people_idx = mem.memory["layer7"]["people"]
        loc_idx = mem.memory["layer7"]["locations"]
        act_idx = mem.memory["layer7"]["activity_events"]
        time_idx = mem.memory["layer7"]["time_nodes"]

        check("合法人名 张三 写入", "张三" in people_idx)
        check("合法人名 李四 写入", "李四" in people_idx)
        check("占位符 canonical_name 拒绝", "canonical_name" not in people_idx)
        check("长句描述拒绝", "Unknown person walking past" not in people_idx)
        check("合法地点 图书馆 写入", "图书馆" in loc_idx)
        check("占位符 canonical_location 拒绝", "canonical_location" not in loc_idx)
        check("合法活动 学习 写入", "学习" in act_idx)
        check("占位符 event_tag 拒绝", "event_tag" not in act_idx)
        check("时间节点 2026-09-17 写入", "2026-09-17" in time_idx)

        # 同键去重
        check("张三的 moment_id 去重",
              len(people_idx["张三"]) == 2 and set(people_idx["张三"]) == {m1, m2},
              f"got {people_idx['张三']!r}")

        # 悬挂引用过滤
        mem.sort({"people": {"王五": ["nonexistent_id"]}})
        check("悬挂 moment_id 不写入索引", "王五" in people_idx and
              "nonexistent_id" not in people_idx.get("王五", []),
              f"got {people_idx.get('王五')!r}")
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("5. combine：locations 合并 + moment 引用同步")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        m1 = add_moment(mem, people=["张三"], location="老图书馆", activity="学习")
        m2 = add_moment(mem, people=["李四"], location="老图书馆", activity="学习")
        m3 = add_moment(mem, people=["王五"], location="家", activity="学习")

        # 预置旧索引键
        mem.memory["layer7"]["locations"]["老图书馆"] = [m1, m2]
        mem.memory["layer7"]["locations"]["家"] = [m3]

        mem.combine({"locations": {"老图书馆": "图书馆"}})

        loc_idx = mem.memory["layer7"]["locations"]
        check("旧键 老图书馆 被移除", "老图书馆" not in loc_idx, f"got {list(loc_idx.keys())}")
        check("规范键 图书馆 合并 moment_id",
              set(loc_idx["图书馆"]) == {m1, m2}, f"got {loc_idx.get('图书馆')!r}")
        check("未涉及的 家 不受影响", loc_idx["家"] == [m3])
        check("moment m1.location 同步为 图书馆", mem.memory["layer7"]["moments"][m1]["location"] == "图书馆",
              f"got {mem.memory['layer7']['moments'][m1]['location']!r}")
        check("moment m2.location 同步为 图书馆", mem.memory["layer7"]["moments"][m2]["location"] == "图书馆")
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("6. combine：people 合并 + 列表元素同步")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        m1 = add_moment(mem, people=["张", "李"], location="图书馆", activity="学习")
        m2 = add_moment(mem, people=["张"], location="家", activity="读书")
        m3 = add_moment(mem, people=["王"], location="家", activity="读书")

        mem.memory["layer7"]["people"]["张"] = [m1, m2]
        mem.memory["layer7"]["people"]["王"] = [m3]

        mem.combine({"people": {"张": "张三"}})

        people_idx = mem.memory["layer7"]["people"]
        check("旧键 张 被移除", "张" not in people_idx, f"got {list(people_idx.keys())}")
        check("规范键 张三 合并 moment_id",
              set(people_idx["张三"]) == {m1, m2}, f"got {people_idx.get('张三')!r}")
        check("未涉及的 王 不受影响", people_idx["王"] == [m3])
        check("moment m1.people 列表元素 张→张三",
              mem.memory["layer7"]["moments"][m1]["people"] == ["张三", "李"],
              f"got {mem.memory['layer7']['moments'][m1]['people']!r}")
        check("moment m2.people 列表元素 张→张三",
              mem.memory["layer7"]["moments"][m2]["people"] == ["张三"])
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("7. combine：activity_events 合并 + 列表元素同步")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        m1 = add_moment(mem, people=["张三"], location="图书馆", activity="阅读")
        m2 = add_moment(mem, people=["李四"], location="家", activity="阅读")

        mem.memory["layer7"]["activity_events"]["阅读"] = [m1, m2]

        mem.combine({"activity_events": {"阅读": "读书"}})

        act_idx = mem.memory["layer7"]["activity_events"]
        check("旧键 阅读 被移除", "阅读" not in act_idx, f"got {list(act_idx.keys())}")
        check("规范键 读书 合并 moment_id",
              set(act_idx["读书"]) == {m1, m2}, f"got {act_idx.get('读书')!r}")
        check("moment m1.activity 同步为 读书",
              mem.memory["layer7"]["moments"][m1]["activity"] == "读书",
              f"got {mem.memory['layer7']['moments'][m1]['activity']!r}")
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    section("8. combine：幂等 + 非法 canonical 跳过")
    # -----------------------------------------------------------------------
    mem = make_mem()
    try:
        m1 = add_moment(mem, people=["张三"], location="图书馆", activity="学习")
        mem.memory["layer7"]["locations"]["图书馆"] = [m1]

        # 幂等 1：old_key 不存在 → 跳过
        mem.combine({"locations": {"不存在的键": "新键"}})
        check("old_key 不存在时跳过", "不存在的键" not in mem.memory["layer7"]["locations"])

        # 幂等 2：old_key == canonical → 跳过
        before = dict(mem.memory["layer7"]["locations"])
        mem.combine({"locations": {"图书馆": "图书馆"}})
        check("old_key == canonical 时跳过（索引不变）",
              mem.memory["layer7"]["locations"] == before)

        # 幂等 3：重复 combine 不重复 moment_id
        mem.combine({"locations": {"图书馆": "市立图书馆"}})
        mem.combine({"locations": {"图书馆": "市立图书馆"}})
        check("重复 combine 幂等（moment_id 不重复）",
              len(mem.memory["layer7"]["locations"]["市立图书馆"]) == 1,
              f"got {mem.memory['layer7']['locations']['市立图书馆']!r}")

        # 非法 canonical → 跳过（防 G7 污染）
        mem.combine({"locations": {"市立图书馆": "canonical_name"}})
        check("非法 canonical（占位符）被拒绝", "canonical_name" not in mem.memory["layer7"]["locations"])
        check("非法 canonical 不删除旧键", "市立图书馆" in mem.memory["layer7"]["locations"])
    finally:
        shutil.rmtree(mem.memory_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    print(f"\n{'=' * 65}\n测试完成：{PASS} 通过，{FAIL} 失败\n{'=' * 65}")
    return FAIL == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
