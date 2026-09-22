"""
build_testset.py — EgoLife 测试集构建（09-22 阶段 C）
======================================================
采样方案 + ground-truth 标注（画像字段 / 记忆归类 / 检索相关）口径定稿并自动标注。

输入：data/egolife/structured/moments.jsonl（9002 条，DAY1，6 参与者 × 30 秒片段）

采样方案（分层等距）：
  按 6 参与者分层，每层按时间顺序等距采样 N 条（默认 150）→ 900 条主测试集（约 10%）。
  保证：参与者均衡、时间均匀覆盖、可复现（固定随机种子/等距）。

ground-truth 三类标注口径：
  1. 记忆归类（四维索引）：people / location / activity / time
     - people_gt  ：已知 6 参与者白名单 ∩ Transcript 提取的说话人
     - location_gt：地点词典命中（30 秒窗口的 scene+dense_caption+transcript）
     - activity_gt：动作关键词 → 活动类别（取命中次数最多的主活动）
     - time_gt    ：DAY + 小时节点（如 "DAY1_14"）
  2. 画像字段（按参与者聚合，从长期统计推导）：
     - frequent_locations：location 频次 top3
     - behavior_patterns ：activity 频次 top3
     - demographics.name ：参与者名
  3. 检索相关（评测 Precision@K / Recall）：
     - 查询词 = 高频 location / people / activity
     - 每个查询词的正例 = 精确包含该关键词的 moment 集合

产出（data/egolife/testset/）：
  - moments_sampled.jsonl   采样后的 moment（带 gt 字段）
  - gt_profile.json         画像字段 ground-truth（按参与者）
  - queries.json            检索评测查询词 + 正例 moment 集合

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python \
        scripts/memory/build_testset.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]          # Proactive_AI_Agent
IN_FILE = ROOT / "data" / "egolife" / "structured" / "moments.jsonl"
OUT_DIR = ROOT / "data" / "egolife" / "testset"

# 采样规模：每参与者采样条数
SAMPLE_PER_PARTICIPANT = 150

# 已知 6 参与者（ground-truth people 白名单）
KNOWN_PEOPLE = ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"]

# 活动类别词典：类别 → 触发关键词（中文，来自 DenseCaption 动作描述）
ACTIVITY_RULES = [
    ("饮食", ["吃饭", "吃东西", "喝水", "喝了", "餐具", "餐桌", "水果", "做饭", "煮", "吃"]),
    ("移动", ["走到", "走过去", "走向", "上楼", "下楼", "楼梯", "站起来", "转身", "后退", "走动"]),
    ("社交", ["说话", "聊天", "讨论", "开会", "回答", "交流", "招呼", "问她", "问他", "笑"]),
    ("观察", ["看着", "看向", "盯着", "注视", "观察", "转头", "抬头", "低头", "望"]),
    ("操作物品", ["拿起", "放到", "放在", "打开", "关上", "整理", "收拾", "递给", "拼图", "摆放"]),
    ("清洁", ["洗碗", "刷", "打扫", "擦", "垃圾", "清洁", "洗"]),
    ("电子设备", ["手机", "电脑", "打字", "屏幕", "充电", "耳机", "拍照", "录像"]),
]


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def participant_of(moment_id: str) -> str:
    """id = DAY1_A6_SHURE_14000000_en → A6_SHURE"""
    parts = moment_id.split("_")
    return f"{parts[1]}_{parts[2]}"


def stratified_sample(moments, per_participant=SAMPLE_PER_PARTICIPANT):
    """分层等距采样：按参与者分层，每层按时间顺序等距取 N 条。"""
    by_participant = defaultdict(list)
    for m in moments:
        by_participant[participant_of(m["id"])].append(m)

    sampled = []
    for p, items in sorted(by_participant.items()):
        # 按时间顺序排序，保证时间均匀覆盖
        items = sorted(items, key=lambda x: x["start_time"])
        n = len(items)
        k = min(per_participant, n)
        if k == 0:
            continue
        # 等距采样（含首不含尾，保证覆盖全程）
        step = n / k
        idxs = sorted(set(int(i * step) for i in range(k)))
        for i in idxs:
            if i < n:
                sampled.append(items[i])
        print(f"  {p}: 从 {n} 条采样 {len(idxs)} 条")
    return sampled


# ---------------------------------------------------------------------------
# ground-truth 标注
# ---------------------------------------------------------------------------

def annotate_classification(m):
    """记忆归类 ground-truth：people / location / activity / time 四维。"""
    # people：白名单 ∩ 提取的说话人
    people_gt = [p for p in m["people"] if p in KNOWN_PEOPLE]

    # location：已提取的地点词列表
    location_gt = m["location"].split("/") if m["location"] else []

    # activity：动作关键词 → 类别（取命中次数最多的主活动）
    text = f"{m['user_action']} {m['dense_caption']} {m['transcript']}"
    hits = Counter()
    for category, keywords in ACTIVITY_RULES:
        for kw in keywords:
            if kw in text:
                hits[category] += text.count(kw)
    activity_gt = hits.most_common(1)[0][0] if hits else ""

    # time：DAY + 小时节点
    parts = m["id"].split("_")
    day = parts[0]                      # DAY1
    ts = parts[3] if len(parts) > 3 else "00000000"
    hour = ts[:2] if len(ts) >= 2 else "00"
    time_gt = f"{day}_{hour}"

    return {
        "id": m["id"],
        "people_gt": people_gt,
        "location_gt": location_gt,
        "activity_gt": activity_gt,
        "time_gt": time_gt,
    }


def build_profile_gt(sampled):
    """画像字段 ground-truth：按参与者聚合长期统计。"""
    by_participant = defaultdict(lambda: {"locations": Counter(), "activities": Counter()})
    for m in sampled:
        p = participant_of(m["id"])
        gt = m["_gt"]
        for loc in gt["location_gt"]:
            by_participant[p]["locations"][loc] += 1
        if gt["activity_gt"]:
            by_participant[p]["activities"][gt["activity_gt"]] += 1

    profile = {}
    for p, data in sorted(by_participant.items()):
        profile[p] = {
            "demographics": {"name": p.split("_")[1].capitalize()},
            "frequent_locations": [l for l, _ in data["locations"].most_common(3)],
            "behavior_patterns": [a for a, _ in data["activities"].most_common(3)],
        }
    return profile


def build_queries(sampled, min_support=10):
    """检索评测查询词 + 正例 moment 集合（精确关键词匹配）。"""
    # 高频 location / activity 作为查询词
    loc_counter = Counter()
    act_counter = Counter()
    people_counter = Counter()
    for m in sampled:
        gt = m["_gt"]
        for loc in gt["location_gt"]:
            loc_counter[loc] += 1
        if gt["activity_gt"]:
            act_counter[gt["activity_gt"]] += 1
        for p in gt["people_gt"]:
            people_counter[p] += 1

    queries = []
    # 地点查询
    for loc, c in loc_counter.items():
        if c >= min_support:
            positives = [m["id"] for m in sampled if loc in m["_gt"]["location_gt"]]
            queries.append({"query": loc, "type": "location", "support": c,
                            "positives": positives})
    # 人物查询
    for p, c in people_counter.items():
        if c >= min_support:
            positives = [m["id"] for m in sampled if p in m["_gt"]["people_gt"]]
            queries.append({"query": p, "type": "people", "support": c,
                            "positives": positives})
    # 活动查询
    for act, c in act_counter.items():
        if c >= min_support:
            positives = [m["id"] for m in sampled if m["_gt"]["activity_gt"] == act]
            queries.append({"query": act, "type": "activity", "support": c,
                            "positives": positives})
    return queries


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    moments = []
    with IN_FILE.open(encoding="utf-8") as f:
        for line in f:
            moments.append(json.loads(line))
    print(f"输入 moment 总数: {len(moments)}")

    print("\n=== 分层采样 ===")
    sampled = stratified_sample(moments)
    print(f"采样后总数: {len(sampled)}")

    print("\n=== 标注记忆归类 ground-truth ===")
    for m in sampled:
        m["_gt"] = annotate_classification(m)

    # 统计标注覆盖率
    has_people = sum(1 for m in sampled if m["_gt"]["people_gt"])
    has_loc = sum(1 for m in sampled if m["_gt"]["location_gt"])
    has_act = sum(1 for m in sampled if m["_gt"]["activity_gt"])
    print(f"  people_gt 覆盖率: {has_people}/{len(sampled)}")
    print(f"  location_gt 覆盖率: {has_loc}/{len(sampled)}")
    print(f"  activity_gt 覆盖率: {has_act}/{len(sampled)}")

    # 画像字段 ground-truth（在 _gt 改名之前构建）
    profile_gt = build_profile_gt(sampled)
    profile_file = OUT_DIR / "gt_profile.json"
    profile_file.write_text(json.dumps(profile_gt, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"产出: {profile_file}（{len(profile_gt)} 参与者）")

    # 检索查询 + 正例
    queries = build_queries(sampled)
    queries_file = OUT_DIR / "queries.json"
    queries_file.write_text(json.dumps(queries, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"产出: {queries_file}（{len(queries)} 个查询词）")

    # 最后写采样 moment（_gt → ground_truth）
    sampled_file = OUT_DIR / "moments_sampled.jsonl"
    with sampled_file.open("w", encoding="utf-8") as f:
        for m in sampled:
            gt = m.pop("_gt")
            m["ground_truth"] = gt
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"产出: {sampled_file}")

    # 打印样例
    print("\n=== 样例：画像 ground-truth ===")
    for p, v in list(profile_gt.items())[:2]:
        print(f"  {p}: {json.dumps(v, ensure_ascii=False)}")
    print("\n=== 样例：检索查询 top5 ===")
    for q in queries[:5]:
        print(f"  [{q['type']}] '{q['query']}' support={q['support']} "
              f"正例={len(q['positives'])} 条")


if __name__ == "__main__":
    main()
