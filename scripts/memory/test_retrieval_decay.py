"""
test_retrieval_decay.py — Step 07+08 单元测试
===============================================
验证对象：
  - tokenize（中英混合分词：中文单字 + 英文单词，09-03 中文 0 命中回归）
  - query / retrieve（中英混合分词 + 相关性排序 + 排除 stale）
  - get_context_for_analysis（early-stop 路径 + 画像注入 + 冷启动）
  - effective_confidence（时间衰减）
  - _apply_decay_and_stale（陈旧淘汰 stale 标记）

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python test_retrieval_decay.py
"""

import sys
import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import memory as M  # noqa: E402

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


def make_mem(tmpdir, translate_fn=None):
    _counter["n"] += 1
    d = tmpdir / f"case_{_counter['n']}"
    d.mkdir(parents=True, exist_ok=True)
    return M.PersonMemory(memory_dir=str(d), translate_fn=translate_fn)


# 模拟翻译（统一英文分词策略：非英文先翻译为英文，测试用 mock 不走 LLM）
MOCK_TRANSLATE = {
    "在图书馆学习": "studying in the library",
    "在图书馆学习，准备考试": "studying in the library preparing for the exam",
    "用户在图书馆学习，准备考试": "user studying in the library preparing for the exam",
}
mock_translate = lambda text: MOCK_TRANSLATE.get(text, text)  # noqa: E731


def T1_tokenize_en():
    print("\n[T1] tokenize 统一英文分词")
    en = M.tokenize("Studying in the LIBRARY, preparing exam!")
    check("英文切出 library/studying/preparing/exam",
          all(t in en for t in ("library", "studying", "preparing", "exam")), f"got {en}")
    check("大小写归一（全部小写）", "LIBRARY" not in en and "studying" in en, f"got {en}")
    check("标点被清理", "!" not in "".join(en) and "," not in "".join(en), f"got {en}")
    cn = M.tokenize("在图书馆学习")
    check("中文切单字 token", cn == ['在', '图', '书', '馆', '学', '习'], f"got {cn}")
    check("空串返回空", M.tokenize("") == [])


def T2_query_cn(tmpdir):
    print("\n[T2] 中文查询单字命中（中英混合分词，零 LLM）")
    m = make_mem(tmpdir, translate_fn=mock_translate)
    m.add(scene="用户在图书馆学习，准备考试", user_action="看书", needs=[], solutions=[])
    # 中文查询词按单字分词，直接命中中文原文，不依赖查询时翻译
    res = m.query("在图书馆学习")
    check("中文 query 单字命中", len(res) >= 1, f"got {len(res)} 条")
    check("命中的是图书馆 moment",
          any("图书馆" in r.get("scene", "") for r in res), f"got {[r.get('scene') for r in res]}")
    check("写入预翻译 normalized 仍存英文",
          "library" in (res[0].get("normalized", {}).get("scene", "")).lower(), f"got {res[0].get('normalized')}")
    # 无翻译函数时中文 query 仍命中（零 LLM 依赖）
    m2 = make_mem(tmpdir)
    m2.add(scene="用户在图书馆学习", user_action="", needs=[], solutions=[])
    check("无翻译函数时中文 query 仍命中（零 LLM）", len(m2.query("在图书馆学习")) >= 1)


def T3_query_en(tmpdir):
    print("\n[T3] query 英文不回归")
    m = make_mem(tmpdir)
    m.add(scene="User is studying in the library", user_action="reading", needs=[], solutions=[])
    res = m.query("library study")
    check("英文 query 命中", len(res) >= 1, f"got {len(res)} 条")


def T4_retrieve_rank_topk(tmpdir):
    print("\n[T4] retrieve 相关性排序 + Top-K")
    m = make_mem(tmpdir)
    m.add(scene="airport security check", user_action="", needs=[], solutions=[])
    m.add(scene="airport gate boarding area", user_action="", needs=[], solutions=[])
    m.add(scene="hotel lobby", user_action="", needs=[], solutions=[])
    out = m.retrieve(location="airport", top_k=2)
    check("返回内容非空", "airport" in out.lower(), f"got {out[:80]}")
    check("不含 hotel（相关性排序，airport 优先）", "hotel" not in out.lower(), f"got {out}")
    lines = [l for l in out.splitlines() if l.startswith("[Layer7/")]
    check("Top-K 截断为 2", len(lines) <= 2, f"got {len(lines)} 条 Layer7")


def T5_context_profile(tmpdir):
    print("\n[T5] get_context_for_analysis 画像注入（修复旧 schema bug）")
    m = make_mem(tmpdir)
    m.update_profile(
        {"demographics": {"name": {"value": "张三", "confidence": 0.9}}},
        moment_id="m1", timestamp="2026-09-10T10:00:00",
    )
    ctx = m.get_context_for_analysis()
    check("含 [Profile] 段", "[Profile]" in ctx, f"got {ctx[:100]}")
    check("画像 name 注入", "张三" in ctx, f"got {ctx[:200]}")


def T6_stale_excluded(tmpdir):
    print("\n[T6] 排除 stale（query / retrieve 不返回 stale moment）")
    m = make_mem(tmpdir)
    mid = m.add(scene="old stale moment about library", user_action="", needs=[], solutions=[])
    m.memory["layer7"]["moments"][mid]["stale"] = True
    res = m.query("library")
    check("stale moment 被排除", all(r.get("id") != mid for r in res), f"got {[r.get('id') for r in res]}")


def T7_effective_confidence():
    print("\n[T7] effective_confidence 时间衰减")
    attr = {"confidence": 0.8, "last_seen": "2026-01-01T00:00:00"}
    now = "2026-09-10T00:00:00"  # 约 252 天
    eff = M.effective_confidence(attr, daily_decay=0.005, now=now)
    check("有效置信度低于原始（衰减生效）", eff < 0.8, f"got {eff:.4f}")
    check("有效置信度 > 0", eff > 0, f"got {eff:.4f}")
    # 无 last_seen 时不衰减
    check("无 last_seen 不衰减", M.effective_confidence({"confidence": 0.8}) == 0.8)


def T8_decay_stale_mark(tmpdir):
    print("\n[T8] update_profile 后陈旧淘汰（stale 标记，按三类曲线）")
    m = make_mem(tmpdir)
    # 渐变型（decaying）字段 + 很旧 timestamp → 衰减至 stale
    m.update_profile(
        {"behavior_patterns": {"with_agents": [
            {"value": "旧观察", "confidence": 0.4, "decay_type": M.DECAY_DECAYING}]}},
        moment_id="m1", timestamp="2025-01-01T00:00:00",
    )
    bhv = m.get_profile()["behavior_patterns"]["with_agents"][0]
    check("attr 带 effective_confidence", "effective_confidence" in bhv, f"got {bhv.keys()}")
    check("attr 带 stale 标记", "stale" in bhv, f"got {bhv.keys()}")
    check("低置信 + 久远（decaying）→ stale=True", bhv.get("stale") is True,
          f"stale={bhv.get('stale')} eff={bhv.get('effective_confidence')}")
    # 永久字段（name）同样久远但不 stale（三类曲线：硬身份永久有效）
    m.update_profile(
        {"demographics": {"name": {"value": "永久", "confidence": 0.4}}},
        moment_id="m2", timestamp="2025-01-01T00:00:00",
    )
    name = m.get_profile()["demographics"]["name"]
    check("永久字段（name）久远不 stale", name.get("stale") is False,
          f"stale={name.get('stale')} eff={name.get('effective_confidence')}")


def T9_cold_start(tmpdir):
    print("\n[T9] 冷启动：空画像不注入 [Profile]")
    m = make_mem(tmpdir)
    ctx = m.get_context_for_analysis()
    check("空画像不注入 [Profile]", "[Profile]" not in ctx, f"got {ctx}")
    check("返回 No previous memory found", "No previous memory found" in ctx, f"got {ctx}")


def T10_context_no_stale_profile(tmpdir):
    print("\n[T10] get_context_for_analysis 排除 stale 画像")
    m = make_mem(tmpdir)
    m.update_profile(
        {
            "demographics": {"name": {"value": "活跃用户", "confidence": 0.9}},
            "behavior_patterns": {"with_agents": [
                {"value": "过期信息", "confidence": 0.9, "decay_type": M.DECAY_DECAYING}]},
        },
        moment_id="m1", timestamp="2025-01-01T00:00:00",  # 很旧 → decaying 字段会 stale
    )
    # 重新 update 用新 timestamp 激活 name，但 decaying 字段保持旧（不再佐证 → 衰减至 stale）
    m.update_profile(
        {"demographics": {"name": {"value": "活跃用户", "confidence": 0.9}}},
        moment_id="m2", timestamp="2026-09-10T10:00:00",
    )
    ctx = m.get_context_for_analysis()
    check("活跃 name 注入", "活跃用户" in ctx, f"got {ctx[:200]}")
    check("stale 画像被排除", "过期信息" not in ctx, f"got {ctx[:300]}")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="retrieval_decay_"))
    try:
        T1_tokenize_en()
        T2_query_cn(tmp)
        T3_query_en(tmp)
        T4_retrieve_rank_topk(tmp)
        T5_context_profile(tmp)
        T6_stale_excluded(tmp)
        T7_effective_confidence()
        T8_decay_stale_mark(tmp)
        T9_cold_start(tmp)
        T10_context_no_stale_profile(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 50}\n结果：{PASS} 通过 / {FAIL} 失败")
    sys.exit(0 if FAIL == 0 else 1)
