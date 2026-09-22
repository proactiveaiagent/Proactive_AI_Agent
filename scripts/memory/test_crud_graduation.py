"""
test_crud_graduation.py — 9.15 CRUD 补齐与层间晋升测试（设计说明书 §2.4 / §4.2 / §4.5）
================================================================================
验证对象：
  - update()：moment 字段白名单、索引联动、layer1/2/3 同步、预翻译刷新
  - delete()：highlight 保护（默认拒删 / force 强删）、清理范围、total_moments 不减
  - highlight()：标记 + 返回 bool
  - 层间晋升：同场景判定（layer1 滑出 → layer2/layer3）、layer2 超限滑入 layer3
  - 跨天清理：today 变化时清空窗口（数据保留在 layer7.moments）

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python test_crud_graduation.py
"""

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


def section(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


_counter = {"n": 0}


def make_mem(tmpdir):
    _counter["n"] += 1
    d = tmpdir / f"case_{_counter['n']}"
    d.mkdir(parents=True, exist_ok=True)
    return PersonMemory(memory_dir=str(d))


def T1_update_whitelist(tmpdir):
    print("\n[T1] update()：字段白名单 + 同步 + 返回 bool")
    m = make_mem(tmpdir)
    mid = m.add(scene="old scene", user_action="walking", needs=[], solutions=[],
                location="park", activity="walk")
    # 合法字段更新
    ok = m.update(mid, {"scene": "new scene", "user_action": "jogging"})
    moment = m.memory["layer7"]["moments"][mid]
    check("update 返回 True", ok)
    check("scene 已更新", moment["scene"] == "new scene")
    check("user_action 已更新", moment["user_action"] == "jogging")
    # 越权字段被忽略（不改 id/timestamp）
    old_ts = moment["timestamp"]
    m.update(mid, {"id": "hacked", "timestamp": "1999-01-01"})
    check("越权字段 id 被忽略", moment["id"] == mid)
    check("越权字段 timestamp 被忽略", moment["timestamp"] == old_ts)
    # layer1 同步（重载后引用断开，防御同步）
    layer1_ids = [x.get("id") for x in m.memory["layer1"]]
    check("layer1 同 id 条目同步更新",
          mid in layer1_ids and m.memory["layer1"][0]["scene"] == "new scene")
    # moment 不存在 / 全越权 → False
    check("moment 不存在返回 False", m.update("m_nonexist", {"scene": "x"}) is False)
    check("全越权字段返回 False", m.update(mid, {"layer": 9}) is False)


def T2_update_index_sync(tmpdir):
    print("\n[T2] update()：people/location/activity 变更联动索引")
    m = make_mem(tmpdir)
    mid = m.add(scene="airport", user_action="", needs=[], solutions=[],
                people=["Alice"], location="LAX", activity="boarding")
    # 改 people/location/activity
    m.update(mid, {"people": ["Bob"], "location": "JFK", "activity": "waiting"})
    l7 = m.memory["layer7"]
    check("旧 people 索引引用已清除", mid not in l7["people"].get("Alice", []))
    check("新 people 索引引用已建立", mid in l7["people"].get("Bob", []))
    check("旧 location 索引引用已清除", mid not in l7["locations"].get("LAX", []))
    check("新 location 索引引用已建立", mid in l7["locations"].get("JFK", []))
    check("旧 activity 索引引用已清除", mid not in l7["activity_events"].get("boarding", []))
    check("新 activity 索引引用已建立", mid in l7["activity_events"].get("waiting", []))
    # 非法 person 不进索引
    m.update(mid, {"people": ["Multiple travelers visible"]})
    check("非法 person 不建索引",
          "Multiple travelers visible" not in l7["people"])


def T3_delete_highlight_protect(tmpdir):
    print("\n[T3] delete()：highlight 保护 + force + 清理范围 + total 不减")
    m = make_mem(tmpdir)
    mid = m.add(scene="x", user_action="", needs=[], solutions=[],
                people=["Alice"], location="park", activity="walk")
    total_before = m.memory["metadata"]["total_moments"]

    # 未 highlight：正常删除
    ok = m.delete(mid)
    check("普通 moment 删除成功", ok)
    check("主存储已删除", mid not in m.memory["layer7"]["moments"])
    check("索引引用已清除", all(mid not in ids
          for idx in m.memory["layer7"].values()
          if isinstance(idx, dict) for ids in idx.values() if isinstance(ids, list)))
    check("total_moments 不减（防 ID 复用）",
          m.memory["metadata"]["total_moments"] == total_before)

    # highlight 保护
    mid2 = m.add(scene="y", user_action="", needs=[], solutions=[])
    check("highlight 返回 True", m.highlight(mid2))
    check("highlight 不存在 moment 返回 False", m.highlight("m_nope") is False)
    ok2 = m.delete(mid2)
    check("highlighted moment 默认拒删", ok2 is False)
    check("拒删后数据仍在", mid2 in m.memory["layer7"]["moments"])
    ok3 = m.delete(mid2, force=True)
    check("force=True 强制删除成功", ok3)
    check("强删后数据移除", mid2 not in m.memory["layer7"]["moments"])


def T4_graduation_same_env(tmpdir):
    print("\n[T4] 层间晋升：layer1 滑出的同场景判定")
    m = make_mem(tmpdir)
    # 6 条不同场景 → 第 6 条触发 layer1 滑出（最旧与当前不同场景 → 直接进 layer3）
    for i in range(6):
        m.add(scene=f"scene {i}", user_action="", needs=[], solutions=[],
              location=f"loc {i}", activity=f"act {i}")
    layer2_ids = [x.get("id") for x in m.memory["layer2"]]
    check("不同场景滑出不进 layer2", layer2_ids == [], f"got {layer2_ids}")
    check("layer3 含滑出的 moment（数据不丢）",
          len([x for x in m.memory["layer3"] if x.get("scene") == "scene 0"]) == 1)

    # 同场景：连续 6 条同 location+activity → 滑出进 layer2
    m2 = make_mem(tmpdir)
    for i in range(6):
        m2.add(scene=f"same scene {i}", user_action="", needs=[], solutions=[],
               location="home", activity="cooking")
    check("同场景滑出进 layer2", len(m2.memory["layer2"]) >= 1,
          f"got {len(m2.memory['layer2'])}")
    check("layer2 中 moment.layer 标记为 2",
          all(x.get("layer") == 2 for x in m2.memory["layer2"]))


def T5_layer2_overflow_to_layer3(tmpdir):
    print("\n[T5] layer2 超限：最旧滑入 layer3（不丢弃）")
    m = make_mem(tmpdir)
    # 同场景持续写入，超出 MAX_LAYER2=20 后最旧应从 layer2 滑入 layer3
    for i in range(30):
        m.add(scene=f"same scene {i}", user_action="", needs=[], solutions=[],
              location="home", activity="cooking")
    total_kept = (len(m.memory["layer1"]) + len(m.memory["layer2"]) + len(m.memory["layer3"]))
    check("layer2 窗口受容量约束", len(m.memory["layer2"]) <= 20,
          f"got {len(m.memory['layer2'])}")
    check("滑出数据保留在 layer3（不丢弃）",
          any(x.get("scene") == "same scene 0" for x in m.memory["layer3"]))
    # 数据真身都在 layer7.moments
    check("真身全部在 layer7.moments", len(m.memory["layer7"]["moments"]) == 30,
          f"got {len(m.memory['layer7']['moments'])}")


def T6_daily_rollover(tmpdir):
    print("\n[T6] 跨天清理：today 变化清空窗口，数据保留在 layer7")
    m = make_mem(tmpdir)
    m.add(scene="yesterday", user_action="", needs=[], solutions=[])
    m.add(scene="yesterday 2", user_action="", needs=[], solutions=[])
    check("跨天前窗口有数据", len(m.memory["layer3"]) == 2)

    # 模拟跨天：把 metadata.today 改到昨天，再 add 触发清理
    m.memory["metadata"]["today"] = "2000-01-01"
    m._save()
    m.add(scene="today new", user_action="", needs=[], solutions=[])
    check("跨天后窗口清空重建", len(m.memory["layer1"]) == 1 and len(m.memory["layer3"]) == 1,
          f"L1={len(m.memory['layer1'])} L3={len(m.memory['layer3'])}")
    check("today 已更新", m.memory["metadata"]["today"] != "2000-01-01")
    check("历史数据保留在 layer7.moments", len(m.memory["layer7"]["moments"]) == 3)


def T7_pretranslate_normalized(tmpdir):
    print("\n[T7] 预翻译：写入时中文文本翻译存 normalized")
    mock = {"在图书馆学习": "studying in the library"}
    m = make_mem(tmpdir)
    m.translate_fn = lambda t: mock.get(t, t)
    mid = m.add(scene="在图书馆学习", user_action="", needs=[], solutions=[])
    moment = m.memory["layer7"]["moments"][mid]
    check("moment 带 normalized 字段", "normalized" in moment, f"got {list(moment.keys())}")
    check("normalized.scene 为英文翻译",
          moment.get("normalized", {}).get("scene") == "studying in the library")
    # 无翻译函数：不存 normalized
    m2 = make_mem(tmpdir)
    mid2 = m2.add(scene="在图书馆学习", user_action="", needs=[], solutions=[])
    check("无翻译函数时不存 normalized",
          "normalized" not in m2.memory["layer7"]["moments"][mid2])


def T8_six_dimension_indices(tmpdir):
    print("\n[T8] 六维索引：environments/objects 写入 + 联动 + 排序（09-22 规范）")
    m = make_mem(tmpdir)
    mid = m.add(scene="在厨房做饭", user_action="拿杯子喝水",
                needs=[], solutions=[],
                people=["张三"], location="厨房", activity="做饭",
                environments=["厨房", "餐桌"], objects=["杯子", "手机"])
    l7 = m.memory["layer7"]
    check("environments 索引写入",
          "厨房" in l7["environments"] and "餐桌" in l7["environments"],
          f"got {list(l7['environments'].keys())}")
    check("objects 索引写入",
          "杯子" in l7["objects"] and "手机" in l7["objects"],
          f"got {list(l7['objects'].keys())}")
    moment = l7["moments"][mid]
    check("moment 存 environments 字段", "厨房" in moment.get("environments", []))
    check("moment 存 objects 字段", "杯子" in moment.get("objects", []))

    # update 联动：改 environments/objects，旧键清理 + 新键建立
    m.update(mid, {"environments": ["卧室"], "objects": ["眼镜"]})
    check("update 联动 environments（旧键清理）",
          "厨房" not in l7["environments"] and "卧室" in l7["environments"])
    check("update 联动 objects（旧键清理）",
          "杯子" not in l7["objects"] and "眼镜" in l7["objects"])

    # 排序：objects 从高频到低频（"眼镜"出现 2 次，"手机"1 次）
    m.add(scene="", user_action="", needs=[], solutions=[], objects=["眼镜"])
    sorted_tags = m._sorted_index_tags("objects")
    check("objects 高频优先排序", sorted_tags[:1] == ["眼镜"], f"got {sorted_tags}")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="crud_graduation_"))
    try:
        T1_update_whitelist(tmp)
        T2_update_index_sync(tmp)
        T3_delete_highlight_protect(tmp)
        T4_graduation_same_env(tmp)
        T5_layer2_overflow_to_layer3(tmp)
        T6_daily_rollover(tmp)
        T7_pretranslate_normalized(tmp)
        T8_six_dimension_indices(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 50}\n结果：{PASS} 通过 / {FAIL} 失败")
    sys.exit(0 if FAIL == 0 else 1)
