"""
run_scenarios.py — 用两套 EgoLife 测试集跑通 7 层记忆模型，产出可视化快照
======================================================================
短期场景（不连续片段）：按 crud_plan 依次执行 add → query → update → highlight → delete，
    每一步记录各层状态，观察 Layer1~7 的完整 CRUD 流程。

长期场景（完整一天）：按时间顺序写入一整天数据（白天累积），每小时记录一次状态；
    全部写完后执行夜间离线整理 sort → combine → compress → update_profile，
    对比整理前后，观察长期记忆如何提炼并更新画像。

产出（供可视化网站读取）：
  data/egolife/runs/short_term_snapshots.json
  data/egolife/runs/long_term_snapshots.json
  data/egolife/runs/*_memory.json          （最终完整记忆状态）

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent
    .../envs/agent/bin/python scripts/memory/run_scenarios.py
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from memory import PersonMemory, LAYER7_INDICES      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "data" / "egolife" / "testset"
RUNS = ROOT / "data" / "egolife" / "runs"

SHORT_FILE = TESTSET / "short_term" / "moments_scattered.jsonl"
SHORT_PLAN = TESTSET / "short_term" / "crud_plan.json"
LONG_FILE = TESTSET / "long_term" / "full_day.jsonl"
LONG_META = TESTSET / "long_term" / "timeline_meta.json"


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------

def snapshot(mem, label, action=""):
    # Capture structural quality signals at this exact replay step.
    m = mem.memory
    l7 = m["layer7"]
    moments = l7.get("moments", {}) or {}
    live_ids = {str(mid) for mid, doc in moments.items()
                if isinstance(doc, dict) and not doc.get("stale")}
    all_ids = {str(mid) for mid in moments}
    indices = {}
    index_top = {}
    refs_by_index = {}
    dangling_refs = 0
    indexed_ids = set()
    for key in LAYER7_INDICES:
        idx = l7.get(key, {}) or {}
        refs = set()
        for ids in idx.values():
            if isinstance(ids, (list, tuple, set)):
                refs.update(str(mid) for mid in ids)
        dangling_refs += len(refs - all_ids)
        valid_live_refs = refs & live_ids
        refs_by_index[key] = len(valid_live_refs)
        indexed_ids.update(valid_live_refs)
        indices[key] = len(idx)
        index_top[key] = sorted(
            ((tag, len(ids)) for tag, ids in idx.items()),
            key=lambda x: -x[1],
        )[:8]

    profile = m["layer6"].get("profile", {})
    profile_fields = {
        k: (len(v) if isinstance(v, (list, dict)) else (1 if v else 0))
        for k, v in profile.items()
    }
    active_docs = [doc for mid, doc in moments.items() if str(mid) in live_ids]
    core_complete = sum(
        bool(str(doc.get("scene") or "").strip()) and
        bool(str(doc.get("user_action") or "").strip())
        for doc in active_docs if isinstance(doc, dict)
    )
    timestamped = sum(bool(str(doc.get("timestamp") or "").strip())
                      for doc in active_docs if isinstance(doc, dict))
    layer4 = m.get("layer4", {}) or {}
    layer5 = m.get("layer5", {}) or {}
    l4_summary = str(layer4.get("summary") or "")
    l5_summary = str(layer5.get("summary") or "")
    denominator = len(live_ids)
    dimension_coverage = {
        key: {
            "count": refs_by_index.get(key, 0),
            "rate": round(100 * refs_by_index.get(key, 0) / denominator, 1)
                    if denominator else 0.0,
        }
        for key in LAYER7_INDICES
    }
    episode_ids = {
        str(doc.get("id"))
        for i in range(1, 4)
        for doc in m.get(f"layer{i}", [])
        if isinstance(doc, dict) and doc.get("id")
    }
    step_metrics = {
        "episodic_moments": len(episode_ids),
        "archive_moments": len(live_ids),
        "core_fields_complete": core_complete,
        "core_fields_rate": round(100 * core_complete / denominator, 1) if denominator else 0.0,
        "timestamped_moments": timestamped,
        "timestamp_rate": round(100 * timestamped / denominator, 1) if denominator else 0.0,
        "indexed_moments": len(indexed_ids),
        "index_coverage_rate": round(100 * len(indexed_ids) / denominator, 1) if denominator else 0.0,
        "index_coverage_by_dimension": dimension_coverage,
        "dangling_index_references": dangling_refs,
        "stale_moments": len(all_ids - live_ids),
        "highlighted_moments": sum(bool(doc.get("highlighted")) for doc in active_docs
                                   if isinstance(doc, dict)),
        "layer4_summary_chars": len(l4_summary),
        "layer5_summary_chars": len(l5_summary),
        "layer4_source_moments": int(layer4.get("source_moment_count") or 0),
        "layer5_source_moments": int(layer5.get("source_moment_count") or 0),
        "profile_slots": sum(profile_fields.values()),
    }
    return {
        "label": label,
        "action": action,
        "layers": {
            "layer1": len(m["layer1"]),
            "layer2": len(m["layer2"]),
            "layer3": len(m["layer3"]),
        },
        "layer7_moments": len(l7.get("moments", {})),
        "indices": indices,
        "index_top": index_top,
        "profile_fields": profile_fields,
        "layer4_summary": l4_summary[:120],
        "layer5_summary": l5_summary[:120],
        "total_moments": m["metadata"].get("total_moments", 0),
        "step_metrics": step_metrics,
    }

def add_moment(mem, m):
    """Reject the retired annotation-backed writer instead of faking memory."""
    raise RuntimeError(
        "The legacy snapshot writer is disabled because it writes benchmark labels "
        "as if they were video perception. Use the interactive video-only pipeline."
    )


# ---------------------------------------------------------------------------
# 短期场景
# ---------------------------------------------------------------------------

def run_short_term():
    raise RuntimeError(
        "Legacy snapshot generation is disabled: it used benchmark labels as memory. "
        "Replay clips through the interactive video-only pipeline instead."
    )
    print("=== 短期记忆场景（不连续视频片段）===")
    moments = [json.loads(l) for l in open(SHORT_FILE, encoding="utf-8") if l.strip()]
    plan = json.load(open(SHORT_PLAN, encoding="utf-8"))

    mem_dir = RUNS / "short_term_mem"
    mem = PersonMemory(memory_dir=str(mem_dir))
    snaps = [snapshot(mem, "初始状态")]

    # 1) add：全部写入
    id_map = {}                       # 数据集 id → 记忆模型 moment_id
    for i, m in enumerate(moments, 1):
        mid = add_moment(mem, m)
        id_map[m["id"]] = mid
        if i % 10 == 0 or i == len(moments):
            snaps.append(snapshot(mem, f"写入 {i} 条", action=f"add #{i}"))
    print(f"  add: {len(moments)} 条")

    # 2) query：用计划里的查询词检索
    query_results = []
    for q in plan.get("query", []):
        hits = mem.query(q["query"], top_k=3)
        query_results.append({"query": q["query"], "hits": len(hits)})
        snaps.append(snapshot(mem, f"检索「{q['query']}」", action="query"))
    print(f"  query: {len(query_results)} 个查询词 → "
          + ", ".join(f"{r['query']}({r['hits']})" for r in query_results))

    # 3) update：按计划修改字段
    for case in plan.get("update", []):
        mid = id_map.get(case["id"])
        if not mid:
            continue
        ok = mem.update(mid, {case["field"]: case["new_value"]})
        snaps.append(snapshot(mem, f"更新 {case['field']}", action="update"))
        print(f"  update: {case['field']} → {case['new_value']} ({ok})")

    # 4) highlight：标记重要数据
    for sid in plan.get("highlight", []):
        mid = id_map.get(sid)
        if mid:
            mem.highlight(mid)
    snaps.append(snapshot(mem, "标记重要记忆", action="highlight"))
    print(f"  highlight: {len(plan.get('highlight', []))} 条")

    # 5) delete：删除（含 highlight 保护验证）
    deleted, protected = [], []
    for case in plan.get("delete", []):
        mid = id_map.get(case["id"])
        if not mid:
            continue
        ok = mem.delete(mid, reason=case.get("reason", "manual"))
        (deleted if ok else protected).append(case["id"])
    snaps.append(snapshot(mem, "清理短期记忆", action="delete"))
    print(f"  delete: 成功 {len(deleted)} 条")

    # 验证 highlight 保护：尝试删除一条被标记的数据
    hl_ids = [id_map[s] for s in plan.get("highlight", []) if id_map.get(s)]
    if hl_ids:
        blocked = mem.delete(hl_ids[0], reason="验证保护")
        forced = mem.delete(hl_ids[0], reason="验证强制删除", force=True)
        print(f"  highlight 保护：默认删除 {'被拒绝 ✅' if not blocked else '未生效 ❌'}；"
              f"force 强删 {'成功 ✅' if forced else '失败 ❌'}")

    out = {
        "scenario": "short_term",
        "desc": "不连续视频片段 · 验证 Layer1~7 完整 CRUD",
        "moments_total": len(moments),
        "crud_summary": {
            "add": len(id_map),
            "query": len(query_results),
            "update": len(plan.get("update", [])),
            "highlight": len(plan.get("highlight", [])),
            "delete": len(deleted),
        },
        "snapshots": snaps,
        "final": snapshot(mem, "最终状态"),
    }

    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RUNS / "short_term_snapshots.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(mem.memory, open(RUNS / "short_term_memory.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  → {RUNS / 'short_term_snapshots.json'}")
    return out


# ---------------------------------------------------------------------------
# 长期场景
# ---------------------------------------------------------------------------

def build_sort_analysis(moments, id_map):
    """The old helper encoded reference answers as model output; fail closed."""
    raise RuntimeError("Annotation-derived index generation is disabled.")


def build_canonical_map(mem):
    """找出近义索引键（仅大小写/单复数差异），构造合并映射。"""
    l7 = mem.memory["layer7"]
    canonical_map = {}
    for key in LAYER7_INDICES:
        idx = l7.get(key, {})
        groups = defaultdict(list)
        for tag in idx:
            groups[tag.strip().lower().rstrip("s")].append(tag)
        mapping = {}
        for base, tags in groups.items():
            if len(tags) > 1:
                # 以出现最早的为主键（稳定），其余并入
                canonical = sorted(tags)[0]
                for t in tags:
                    if t != canonical:
                        mapping[t] = canonical
        if mapping:
            canonical_map[key] = mapping
    return canonical_map


def build_compress_summary(moments, participant):
    """The old helper encoded reference answers as memory summaries; fail closed."""
    raise RuntimeError("Annotation-derived summaries are disabled.")


def run_long_term():
    raise RuntimeError(
        "Legacy snapshot generation is disabled: it used benchmark labels as memory. "
        "Replay clips through the interactive video-only pipeline instead."
    )
    print("\n=== 长期记忆场景（完整一天）===")
    moments = [json.loads(l) for l in open(LONG_FILE, encoding="utf-8") if l.strip()]
    meta = json.load(open(LONG_META, encoding="utf-8"))
    participant = meta["participant"]
    print(f"  参与者 {participant}，共 {len(moments)} 条，覆盖 {meta['hours_covered']}")

    mem_dir = RUNS / "long_term_mem"
    mem = PersonMemory(memory_dir=str(mem_dir))
    snaps = [snapshot(mem, "初始状态")]

    # 白天：按时间顺序写入，每个小时记录一次
    id_map = {}
    cur_hour = None
    t0 = time.time()
    for i, m in enumerate(moments, 1):
        mid = add_moment(mem, m)
        id_map[m["id"]] = mid
        h = m.get("_hour") or (m["id"].split("_")[3][:2] if len(m["id"].split("_")) > 3 else "")
        if h != cur_hour:
            if cur_hour is not None:
                snaps.append(snapshot(mem, f"{cur_hour}:00 结束", action="hour_end"))
            cur_hour = h
    snaps.append(snapshot(mem, f"{cur_hour}:00 结束（白天结束）", action="day_end"))
    print(f"  白天写入 {len(id_map)} 条，用时 {time.time() - t0:.1f}s")

    before = snapshot(mem, "夜间整理前")

    # 夜间整理 1：sort 归类
    analysis = build_sort_analysis(moments, id_map)
    mem.sort(analysis)
    snaps.append(snapshot(mem, "夜间整理 · 归类 sort", action="sort"))
    print(f"  sort: 归类 {sum(len(v) for v in analysis.values())} 个索引键")

    # 夜间整理 2：combine 近义合并
    canonical_map = build_canonical_map(mem)
    if canonical_map:
        mem.combine(canonical_map)
        merged = sum(len(v) for v in canonical_map.values())
        print(f"  combine: 合并 {merged} 个近义键")
    else:
        print("  combine: 无近义键可合并")
    snaps.append(snapshot(mem, "夜间整理 · 合并 combine", action="combine"))

    # 夜间整理 3：compress 压缩摘要
    summary = build_compress_summary(moments, participant)
    mem.compress(summary)
    snaps.append(snapshot(mem, "夜间整理 · 压缩 compress", action="compress"))
    print("  compress: 写入 layer4/layer5 摘要")

    # 夜间整理 4：update_profile 画像更新
    gt_profile = json.load(open(TESTSET / "long_term" / "gt_profile.json", encoding="utf-8"))
    prof = gt_profile.get(participant, {})
    extracted = {
        "demographics": prof.get("demographics", {}),
        "frequent_locations": prof.get("frequent_locations", []),
        "behavior_patterns": prof.get("behavior_patterns", []),
    }
    mem.update_profile(extracted)
    snaps.append(snapshot(mem, "夜间整理 · 画像更新", action="update_profile"))
    print(f"  update_profile: 更新画像字段 {list(extracted.keys())}")

    after = snapshot(mem, "夜间整理后")

    out = {
        "scenario": "long_term",
        "desc": f"{participant} 完整一天 · 验证记忆流转与夜间整理",
        "participant": participant,
        "moments_total": len(moments),
        "hours_covered": meta["hours_covered"],
        "timeline_segments": meta["segments"],
        "before_consolidation": before,
        "after_consolidation": after,
        "snapshots": snaps,
        "final": after,
    }

    RUNS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RUNS / "long_term_snapshots.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(mem.memory, open(RUNS / "long_term_memory.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  → {RUNS / 'long_term_snapshots.json'}")
    return out


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    run_short_term()
    run_long_term()
    print("\n=== 完成 ===")
    print(f"  快照输出目录: {RUNS}")


if __name__ == "__main__":
    main()
