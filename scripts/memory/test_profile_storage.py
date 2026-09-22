"""
test_profile_storage.py — 分层存储落地（Step 06）单元测试
============================================================
验证对象：
  - `memory.py` 的画像 API：update_profile() / get_profile()（规格书骨架字段）
  - 列表追加去重（修复 09-03 的 hobbies 丢历史缺陷）
  - 单值冲突裁决（频次 > 最近 > 置信度，败者入 history）
  - 佐证合并（同值 observations 累加 + confidence 增强）
  - 低置信度过滤（confidence < 0.3 不入库）
  - 旧数据兼容（旧 profile 结构 name/basic_info/preferences/habits 迁移不报错）
  - compress() 职责边界（不再写 layer6.profile）
  - 原子落盘（保存后重载数据仍在）
  - `agent.py` 语法检查（consolidation prompt 的 f-string 转义）

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python test_profile_storage.py
"""

import ast
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from memory import PersonMemory  # noqa: E402

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


_counter = {"n": 0}


def make_mem(tmpdir):
    """每个测试用独立子目录，避免共享 memory.json 导致数据串扰。"""
    _counter["n"] += 1
    d = tmpdir / f"case_{_counter['n']}"
    d.mkdir(parents=True, exist_ok=True)
    return PersonMemory(memory_dir=str(d))


def T1_write_and_read(tmpdir):
    print("\n[T1] update_profile / get_profile 基本读写")
    m = make_mem(tmpdir)
    m.update_profile(
        {"demographics": {"name": {"value": "张三", "confidence": 0.9, "evidence": "用户自述"}}},
        moment_id="m_test_1", timestamp="2026-09-08T10:00:00",
    )
    p = m.get_profile()
    check("demographics.name 已写入", p["demographics"]["name"]["value"] == "张三")
    check("source 记录来源 moment",
          p["demographics"]["name"]["source"] == "m_test_1",
          f"got {p['demographics']['name'].get('source')}")
    check("首次观察 observations=1", p["demographics"]["name"]["observations"] == 1)
    check("timestamp 已记录", p["demographics"]["name"]["timestamp"] == "2026-09-08T10:00:00")
    check("last_updated 已更新", m.memory["layer6"]["last_updated"] is not None)


def T2_list_append_no_loss(tmpdir):
    print("\n[T2] 列表追加去重（09-03 回归：hobbies 不丢历史）")
    m = make_mem(tmpdir)
    m.update_profile(
        {"preferences": {"hobbies": [
            {"value": "photography", "confidence": 0.8},
            {"value": "travel", "confidence": 0.7},
        ]}},
        moment_id="m1", timestamp="2026-09-08T10:00:00",
    )
    m.update_profile(
        {"preferences": {"hobbies": [{"value": "hiking", "confidence": 0.6}]}},
        moment_id="m2", timestamp="2026-09-08T11:00:00",
    )
    hobbies = [a["value"] for a in m.get_profile()["preferences"]["hobbies"]]
    check("旧值 photography 保留", "photography" in hobbies, f"got {hobbies}")
    check("旧值 travel 保留", "travel" in hobbies, f"got {hobbies}")
    check("新值 hiking 追加", "hiking" in hobbies, f"got {hobbies}")
    check("总数=3（非覆盖）", len(hobbies) == 3, f"got {len(hobbies)}")


def T3_corroborate_same_value(tmpdir):
    print("\n[T3] 同值佐证合并：observations 累加 + confidence 增强")
    m = make_mem(tmpdir)
    m.update_profile(
        {"preferences": {"food": [{"value": "spicy", "confidence": 0.5}]}},
        moment_id="m1", timestamp="2026-09-08T10:00:00",
    )
    m.update_profile(
        {"preferences": {"food": [{"value": "spicy", "confidence": 0.6}]}},
        moment_id="m2", timestamp="2026-09-08T11:00:00",
    )
    food = m.get_profile()["preferences"]["food"]
    check("仍只有 1 条（不重复追加）", len(food) == 1, f"got {len(food)}")
    check("observations 累加为 2", food[0]["observations"] == 2, f"got {food[0]['observations']}")
    check("confidence 增强(>0.6)",
          food[0]["confidence"] > 0.6, f"got {food[0]['confidence']:.4f}")
    check("last_seen 更新为最新", food[0]["last_seen"] == "2026-09-08T11:00:00")


def T4_conflict_resolution(tmpdir):
    print("\n[T4] 单值冲突裁决：频次打平 → 最近优先，败者入 history")
    m = make_mem(tmpdir)
    m.update_profile(
        {"demographics": {"age": {"value": "30", "confidence": 0.7}}},
        moment_id="m1", timestamp="2026-09-08T10:00:00",
    )
    m.update_profile(
        {"demographics": {"age": {"value": "25", "confidence": 0.8}}},
        moment_id="m2", timestamp="2026-09-08T11:00:00",
    )
    age = m.get_profile()["demographics"]["age"]
    check("最近值 25 胜出", age["value"] == "25", f"got {age['value']}")
    check("败者 30 入 history",
          "history" in age and any(h["value"] == "30" for h in age["history"]),
          f"history={age.get('history')}")


def T5_low_confidence_filter(tmpdir):
    print("\n[T5] 低置信度过滤：confidence < 0.3 不入库")
    m = make_mem(tmpdir)
    m.update_profile(
        {"preferences": {"hobbies": [{"value": "rare_hobby", "confidence": 0.1}]}},
        moment_id="m1", timestamp="2026-09-08T10:00:00",
    )
    hobbies = m.get_profile()["preferences"]["hobbies"]
    check("低置信项被丢弃", hobbies == [], f"got {hobbies}")


def T6_old_data_compat(tmpdir):
    print("\n[T6] 旧数据兼容：旧 profile 结构迁移不报错")
    memdir = tmpdir / "legacy"
    memdir.mkdir()
    old_db = {
        "layer1": [], "layer2": [], "layer3": [],
        "layer4": {"summary": "", "current_tasks": [], "life_trajectory": "",
                   "last_updated": None, "source_moment_count": 0},
        "layer5": {"summary": "", "key_events": [], "long_term_patterns": "",
                   "last_updated": None, "source_moment_count": 0},
        "layer6": {
            "summary": "",
            "profile": {
                "name": "张三",
                "basic_info": {"age": "30"},
                "preferences": {"food": "spicy"},
                "habits": {"morning": "drink coffee"},
            },
            "last_updated": None, "source_moment_count": 0,
        },
        "layer7": {"time_nodes": {}, "activity_events": {}, "people": {},
                   "locations": {}, "moments": {}},
        "metadata": {"total_moments": 0, "total_encounters": 0,
                     "last_consolidation": None, "session_start": "", "today": ""},
    }
    (memdir / "memory.json").write_text(json.dumps(old_db, ensure_ascii=False), encoding="utf-8")

    m = PersonMemory(memory_dir=str(memdir))  # 不报错即通过
    p = m.get_profile()
    check("name 迁移到 demographics.name",
          p["demographics"].get("name", {}).get("value") == "张三", f"got {p['demographics']}")
    check("basic_info.age 迁移到 demographics.age",
          p["demographics"].get("age", {}).get("value") == "30")
    check("旧 preferences.food 迁移到 preferences.food",
          p["preferences"].get("food", [{}])[0].get("value") == "spicy")
    check("旧 habits 迁移到 behavior_patterns.with_agents",
          p["behavior_patterns"].get("with_agents", [{}])[0].get("value") == "drink coffee")
    check("新结构字段齐全",
          set(p.keys()) == {"demographics", "preferences", "frequent_locations", "behavior_patterns",
                            "personality", "goals", "decisions", "motivations"})


def T7_compress_no_profile(tmpdir):
    print("\n[T7] compress() 职责边界：不再写 layer6.profile")
    m = make_mem(tmpdir)
    m.update_profile(
        {"demographics": {"name": {"value": "张三", "confidence": 0.9}}},
        moment_id="m1", timestamp="2026-09-08T10:00:00",
    )
    m.compress({"layer6": {
        "summary": "new summary",
        "profile": {"demographics": {"name": {"value": "李四", "confidence": 0.9}}},
    }})
    name = m.get_profile()["demographics"]["name"]["value"]
    check("profile 未被 compress 覆盖（仍是 张三）", name == "张三", f"got {name}")
    check("layer6.summary 正常写入", m.memory["layer6"]["summary"] == "new summary")


def T8_atomic_persist_reload(tmpdir):
    print("\n[T8] 原子落盘 + 重载")
    memdir = tmpdir / "persist"
    memdir.mkdir()
    m = PersonMemory(memory_dir=str(memdir))
    m.update_profile(
        {"frequent_locations": [{"value": "library", "confidence": 0.8}]},
        moment_id="m1", timestamp="2026-09-08T10:00:00",
    )
    check("无残留 .tmp 文件", not list(memdir.glob("*.tmp")), f"found {list(memdir.glob('*.tmp'))}")

    m2 = PersonMemory(memory_dir=str(memdir))
    locs = [a["value"] for a in m2.get_profile()["frequent_locations"]]
    check("重载后数据仍在", locs == ["library"], f"got {locs}")


def T9_agent_syntax():
    print("\n[T9] agent.py 语法检查（consolidation prompt 的 f-string 转义）")
    try:
        src = (PROJECT_ROOT / "code" / "agent.py").read_text(encoding="utf-8")
        ast.parse(src)
        check("agent.py 语法正确", True)
    except SyntaxError as e:
        check("agent.py 语法正确", False, f"SyntaxError: {e}")


def T10_static_vs_dynamic(tmpdir):
    print("\n[T10] 静态画像落 layer6（动态状态不进 profile）")
    m = make_mem(tmpdir)
    # 动态状态（status_inference/gaze_target）不在 profile 白名单，应被忽略
    m.update_profile(
        {
            "demographics": {"name": {"value": "张三", "confidence": 0.9}},
            "status_inference": {"emotion": {"value": "happy", "confidence": 0.9}},  # 非白名单
        },
        moment_id="m1", timestamp="2026-09-08T10:00:00",
    )
    p = m.get_profile()
    check("静态画像 demographics 写入", p["demographics"]["name"]["value"] == "张三")
    check("动态状态 status_inference 被拒绝（不在白名单）",
          "status_inference" not in p, f"got keys={list(p.keys())}")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="profile_store_"))
    try:
        T1_write_and_read(tmp)
        T2_list_append_no_loss(tmp)
        T3_corroborate_same_value(tmp)
        T4_conflict_resolution(tmp)
        T5_low_confidence_filter(tmp)
        T6_old_data_compat(tmp)
        T7_compress_no_profile(tmp)
        T8_atomic_persist_reload(tmp)
        T9_agent_syntax()
        T10_static_vs_dynamic(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 50}\n结果：{PASS} 通过 / {FAIL} 失败")
    sys.exit(0 if FAIL == 0 else 1)
