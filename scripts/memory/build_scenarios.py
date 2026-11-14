"""
build_scenarios.py — 构建两套场景测试集（09-23 会议要求）
======================================================================
最终目标：跑通 EgoLife 数据集 + 搭建可视化网站，分短期记忆 / 长期记忆两个场景。

测试集一 · 短期记忆（不连续视频片段）
  - 从 6 个参与者各跳跃抽取若干条，片段之间时间不连续（中间有跳过的片段）
  - 用于验证白天实时处理下 Layer1~7 的完整 CRUD 流程
  - 附带 crud_plan.json：规定哪些片段做 update / highlight / delete，以及用什么查询词 query

测试集二 · 长期记忆（完整一天）
  - 取一个参与者 DAY1 的完整数据（按时间排序），尽可能还原一整天
  - 用于验证一天过程中的记忆流转，以及一天结束后夜间离线整理
    （合并 combine / 归类 sort / 压缩 compress / 画像更新 update_profile）

输出：
  data/egolife/testset/short_term/moments_scattered.jsonl   不连续片段
  data/egolife/testset/short_term/crud_plan.json            CRUD 操作计划
  data/egolife/testset/long_term/full_day.jsonl             完整一天
  data/egolife/testset/long_term/timeline_meta.json         时间轴元数据

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent
    .../envs/agent/bin/python scripts/memory/build_scenarios.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import build_testset as bt           # 复用已有 ground-truth 标注逻辑

ROOT = bt.ROOT
IN_FILE = bt.IN_FILE
OUT_DIR = bt.OUT_DIR

SHORT_DIR = OUT_DIR / "short_term"
LONG_DIR = OUT_DIR / "long_term"

# 短期场景：每个参与者抽取条数（片段之间刻意不连续）
SHORT_PER_PARTICIPANT = 10
# 长期场景：取哪个参与者的完整一天
LONG_PARTICIPANT = "A1_JAKE"

random.seed(20260923)                # 固定随机种子，保证可复现


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def load_moments():
    with open(IN_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hour_of(m):
    """id = DAY1_A1_JAKE_13000000_en → 13（小时）"""
    parts = m["id"].split("_")
    ts = parts[3] if len(parts) > 3 else "00000000"
    return ts[:2] if len(ts) >= 2 else "00"


def sort_key(m):
    """按（小时, 段内起始秒）排序，还原一天的时间顺序。"""
    return (hour_of(m), float(m.get("start_time", 0)))


def annotate(m):
    """给一条 moment 打上六维归类 ground-truth。"""
    gt = bt.annotate_classification(m)
    m = dict(m)
    m["ground_truth"] = gt
    m["_gt"] = gt                 # build_profile_gt 读取的字段名
    m["_hour"] = hour_of(m)
    return m


# ---------------------------------------------------------------------------
# 测试集一：短期记忆（不连续片段）
# ---------------------------------------------------------------------------

def build_short_term(moments):
    """跳跃采样：片段之间时间不连续，模拟白天零散发生的场景。"""
    by_participant = defaultdict(list)
    for m in moments:
        by_participant[bt.participant_of(m["id"])].append(m)

    picked = []
    for p, items in sorted(by_participant.items()):
        items = sorted(items, key=sort_key)
        n = len(items)
        k = min(SHORT_PER_PARTICIPANT, n)
        if k == 0:
            continue

        # 跳跃采样：步长尽量大，保证相邻两条之间隔着若干未选取的片段（不连续）
        step = max(1, n // k)
        idxs = []
        i = 0
        while len(idxs) < k and i < n:
            idxs.append(i)
            i += step

        # 记录与前一条选中片段的间隔（体现"不连续"）
        prev = None
        for idx in idxs:
            m = annotate(items[idx])
            gap = None if prev is None else idx - prev
            m["_gap_before"] = gap
            picked.append(m)
            prev = idx

        hours = sorted({hour_of(m) for m in picked if bt.participant_of(m["id"]) == p})
        print(f"  {p}: 从 {n} 条跳跃取 {len(idxs)} 条（步长 {step}），覆盖小时 {hours}")

    picked.sort(key=sort_key)
    return picked


def build_crud_plan(scattered):
    """短期场景的 CRUD 操作计划：规定 update / highlight / delete / query 用哪些数据。"""
    ids = [m["id"] for m in scattered]

    # update：挑 5 条，把 location 改成 ground-truth 中的某个值（模拟纠错/补全）
    # 字段优先级：location → environments → objects → activity（取 ground-truth 中非空的一项）
    update_cases = []
    for m in scattered:
        if len(update_cases) >= 5:
            break
        gt = m["ground_truth"]
        for field in ("location", "environments", "objects", "activity"):
            values = gt.get(f"{field}_gt")
            if isinstance(values, str):
                values = [values] if values else []
            if values:
                update_cases.append({
                    "id": m["id"],
                    "field": field,
                    "new_value": values[0],
                    "reason": f"按 ground-truth 补全{field}（模拟用户纠错）",
                })
                break

    # highlight：挑 5 条（与 update 不重叠），标记为用户确认正确
    used = {c["id"] for c in update_cases}
    highlight_ids = [i for i in ids if i not in used][:5]

    # delete：挑 3 条（与 update / highlight 均不重叠），模拟场景结束后清理短期记忆
    used |= set(highlight_ids)
    delete_cases = [
        {"id": i, "reason": "场景结束后清理短期记忆"}
        for i in ids if i not in used
    ][:3]

    # query：用参与者名 / 地点 / 活动构造查询词，验证检索链路
    query_terms = []
    for term in ["Jake", "Alice", "kitchen", "living room", "phone"]:
        query_terms.append({"query": term, "type": "keyword"})

    return {
        "add": ids,
        "update": update_cases,
        "highlight": highlight_ids,
        "delete": delete_cases,
        "query": query_terms,
        "_说明": "短期记忆场景：按此计划依次执行 add→query→update→highlight→delete，"
                 "观察 Layer1~7 各层的变化",
    }


# ---------------------------------------------------------------------------
# 测试集二：长期记忆（完整一天）
# ---------------------------------------------------------------------------

def build_long_term(moments):
    """取一个参与者 DAY1 的完整数据，按时间排序还原一整天。"""
    items = [m for m in moments if bt.participant_of(m["id"]) == LONG_PARTICIPANT]
    items.sort(key=sort_key)
    return [annotate(m) for m in items]


def build_timeline_meta(full_day):
    """时间轴元数据：按小时分段统计，供可视化网站做回放。"""
    by_hour = defaultdict(list)
    for m in full_day:
        by_hour[m["_hour"]].append(m)

    segments = []
    for hour in sorted(by_hour):
        items = by_hour[hour]
        people = Counter()
        locations = Counter()
        activities = Counter()
        for m in items:
            gt = m["ground_truth"]
            for p in gt["people_gt"]:
                people[p] += 1
            for loc in gt["location_gt"]:
                locations[loc] += 1
            if gt["activity_gt"]:
                activities[gt["activity_gt"]] += 1
        segments.append({
            "hour": f"{hour}:00",
            "count": len(items),
            "people": people.most_common(3),
            "locations": locations.most_common(3),
            "activities": activities.most_common(3),
            "first_id": items[0]["id"],
            "last_id": items[-1]["id"],
        })

    return {
        "participant": LONG_PARTICIPANT,
        "day": "DAY1",
        "total": len(full_day),
        "hours_covered": [s["hour"] for s in segments],
        "segments": segments,
        "_说明": "长期记忆场景：按 hour 顺序回放 full_day.jsonl，"
                 "白天累积写入；全部写完后执行夜间整理（sort→combine→compress→update_profile）",
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            r = {k: v for k, v in r.items() if k != "_gt"}   # _gt 与 ground_truth 重复
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {path}  ({len(rows)} 条)")


def dump_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  → {path}")


def main():
    print("=== 读取预处理数据 ===")
    moments = load_moments()
    print(f"  总条数: {len(moments)}")

    print("\n=== 测试集一：短期记忆（不连续视频片段）===")
    scattered = build_short_term(moments)
    dump_jsonl(SHORT_DIR / "moments_scattered.jsonl", scattered)

    plan = build_crud_plan(scattered)
    dump_json(SHORT_DIR / "crud_plan.json", plan)
    print(f"  CRUD 计划：add {len(plan['add'])} / update {len(plan['update'])} / "
          f"highlight {len(plan['highlight'])} / delete {len(plan['delete'])} / "
          f"query {len(plan['query'])}")

    # 短期场景的画像 ground-truth（按参与者聚合）
    short_profile = bt.build_profile_gt(scattered)
    dump_json(SHORT_DIR / "gt_profile.json", short_profile)

    print("\n=== 测试集二：长期记忆（完整一天）===")
    full_day = build_long_term(moments)
    dump_jsonl(LONG_DIR / "full_day.jsonl", full_day)
    print(f"  参与者: {LONG_PARTICIPANT}，条数: {len(full_day)}")

    meta = build_timeline_meta(full_day)
    dump_json(LONG_DIR / "timeline_meta.json", meta)
    print(f"  覆盖小时: {meta['hours_covered']}")

    long_profile = bt.build_profile_gt(full_day)
    dump_json(LONG_DIR / "gt_profile.json", long_profile)

    print("\n=== 完成 ===")
    print(f"  短期（不连续片段）: {SHORT_DIR}")
    print(f"  长期（完整一天）  : {LONG_DIR}")


if __name__ == "__main__":
    main()
