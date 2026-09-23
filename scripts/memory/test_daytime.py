"""
test_daytime.py — 白天实时处理模块测试（09-24 阶段 D）
======================================================================
验证短期记忆（闪存）的三条职责：
1. 实时写入：Layer1 → 按场景分流 Layer2/3 → Layer7
2. 场景结束：清理 Layer1/Layer2 短期记忆，Layer3/Layer7 保留
3. 后台翻译：非英文文本异步翻译回填 normalized，不阻塞写入与检索

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent
    .../envs/agent/bin/python scripts/memory/test_daytime.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import PersonMemory                    # noqa: E402
from daytime_processor import DaytimeProcessor     # noqa: E402

PASS = FAIL = 0


def check(cond, label, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        print(f"  ❌ {label}  {extra}")


def fake_translate(text: str) -> str:
    """假的翻译函数：中文 → 固定的英文串，用于验证后台翻译链路。"""
    table = {"我在厨房切菜": "I am cutting vegetables in the kitchen",
             "切菜": "cut vegetables",
             "厨房": "kitchen",
             "做饭": "cooking"}
    return table.get(text, f"[EN] {text}")


def section(title):
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="daytime_test_"))
    try:
        # --------------------------------------------------------------
        section("T1 · 场景写入与实时检索")
        mem = PersonMemory(memory_dir=str(tmp / "m1"))
        day = DaytimeProcessor(mem, translate_fn=fake_translate)

        sid = day.start_scene("morning_kitchen")
        check(bool(sid), "start_scene 返回场景 id", f"sid={sid}")

        mid1 = day.ingest(scene="我在厨房切菜", user_action="切菜",
                          location="厨房", activity="做饭")
        check(bool(mid1), "ingest 返回 moment_id", f"mid={mid1}")
        check(len(mem.memory["layer1"]) >= 1, "写入后 Layer1 有内容",
              f"layer1={len(mem.memory['layer1'])}")
        check(len(mem.memory["layer3"]) >= 1, "写入后 Layer3（当日）有内容",
              f"layer3={len(mem.memory['layer3'])}")
        check(mid1 in mem.memory["layer7"]["moments"], "Layer7 主存储保存了完整内容")

        # 再写两条，累积当日记忆
        for i in range(2):
            day.ingest(scene=f"我在厨房切菜 {i}", user_action="切菜",
                       location="厨房", activity="做饭")
        check(len(mem.memory["layer3"]) == 3, "Layer3 累积到 3 条",
              f"layer3={len(mem.memory['layer3'])}")

        hits = day.query("切菜", top_k=3)
        check(len(hits) > 0, "实时检索能命中（中文查询词）",
              f"hits={len(hits)}")
        check(len(day.context(top_k=3)) > 0, "context 返回短期上下文")

        # --------------------------------------------------------------
        section("T2 · 场景结束：清理短期记忆，保留长期")
        before_l3 = len(mem.memory["layer3"])
        before_l7 = len(mem.memory["layer7"]["moments"])

        cleared = day.end_scene()
        check(cleared > 0, "end_scene 返回清理条数", f"cleared={cleared}")
        check(len(mem.memory["layer1"]) == 0, "场景结束后 Layer1 已清空",
              f"layer1={len(mem.memory['layer1'])}")
        check(len(mem.memory["layer2"]) == 0, "场景结束后 Layer2 已清空",
              f"layer2={len(mem.memory['layer2'])}")
        check(len(mem.memory["layer3"]) == before_l3, "Layer3（当日窗口）保留，未丢数据",
              f"{before_l3} → {len(mem.memory['layer3'])}")
        check(len(mem.memory["layer7"]["moments"]) == before_l7, "Layer7（永久）保留，未丢数据")
        check(day.scene_id is None, "场景状态已重置")

        # 清空短期后仍能检索到（数据在 Layer7）
        hits2 = day.query("切菜", top_k=3)
        check(len(hits2) > 0, "清理短期记忆后仍能检索到（真身在 Layer7）",
              f"hits={len(hits2)}")

        # --------------------------------------------------------------
        section("T3 · 后台离线翻译")
        ok = day.flush_translations(timeout=5)
        check(ok, "后台翻译队列已清空", f"pending={day._queue.qsize()}")

        translated_any = False
        for m in mem.memory["layer7"]["moments"].values():
            if m.get("normalized"):
                translated_any = True
                break
        check(translated_any, "后台翻译回填了 moment['normalized']")
        check(day.stats["translated"] > 0, "统计到翻译条数",
              f"translated={day.stats['translated']}")

        # --------------------------------------------------------------
        section("T4 · 无翻译函数时降级（不阻塞、不报错）")
        mem2 = PersonMemory(memory_dir=str(tmp / "m2"))
        day2 = DaytimeProcessor(mem2, translate_fn=None)     # 不注入翻译
        day2.start_scene("no_translate")
        mid = day2.ingest(scene="我在客厅看书", user_action="看书",
                          location="客厅", activity="阅读")
        check(bool(mid), "无翻译函数时仍能正常写入")
        check(day2._queue.qsize() == 0, "无翻译函数时不入队（不翻译）")
        hits3 = day2.query("看书", top_k=3)
        check(len(hits3) > 0, "无翻译时检索仍可用（中英混合分词兜底）",
              f"hits={len(hits3)}")

        # --------------------------------------------------------------
        section("T5 · 多场景连续切换")
        mem3 = PersonMemory(memory_dir=str(tmp / "m3"))
        day3 = DaytimeProcessor(mem3, translate_fn=fake_translate)
        total_cleared = 0
        for name in ("scene_a", "scene_b", "scene_c"):
            day3.start_scene(name)
            for i in range(3):
                day3.ingest(scene=f"{name} 的活动 {i}", user_action=f"动作{i}",
                            location="厨房", activity="做饭")
            total_cleared += day3.end_scene()
        check(day3.stats["scenes"] == 3, "场景计数为 3", f"scenes={day3.stats['scenes']}")
        check(day3.stats["ingested"] == 9, "写入计数为 9", f"ingested={day3.stats['ingested']}")
        check(len(mem3.memory["layer3"]) == 9, "Layer3 保留全部 9 条（跨场景累积）",
              f"layer3={len(mem3.memory['layer3'])}")
        check(len(mem3.memory["layer7"]["moments"]) == 9, "Layer7 保留全部 9 条")

        # --------------------------------------------------------------
        section("T6 · 快照与统计")
        snap = day3.snapshot()
        check(snap["layer7_moments"] == 9, "快照 Layer7 数量为 9", f"{snap['layer7_moments']}")
        check(snap["stats"]["ingested"] == 9, "快照统计写入数正确")
        check(snap["layer1"] == 0 and snap["layer2"] == 0, "快照显示短期记忆已清空")

        day.stop()
        day2.stop()
        day3.stop()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 66)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 66)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
