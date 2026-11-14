"""
bench_retrieval_baseline.py — 09-03 检索性能基线压测
=====================================================
目的：在无法使用 GPU（8 卡全占满）的情况下，测量 PersonMemory 检索路径的
      纯 CPU 耗时基线，为 DDL3「检索 ≤2s」目标提供对照数据。

测量对象：
  1. PersonMemory.query(text)              — Part2 需求分析的关键词检索
  2. PersonMemory.retrieve(people,loc,act) — 分层索引查找
  3. PersonMemory.get_context_for_analysis() — 组装 prompt 上下文
  4. PersonMemory._load()                  — 冷启动加载（JSON 解析）

用法：
  python bench_retrieval_baseline.py
输出：
  打印各规模下的 P50 / P95 耗时；结果写入同目录 bench_retrieval_baseline.json

⚠️ 使用 /tmp 下的临时记忆目录，绝不污染 code/memory/memory.json。
"""

import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from memory import PersonMemory  # noqa: E402

TMP_DIR = "/tmp/bench_mem_0903"
SCALES = [100, 500, 1000, 5000]
REPEATS = 20

PEOPLE = ["Alice", "Bob", "姥姥", "Richard", "Mom", "David", "同事小李"]
LOCS = ["LAX", "Sun Noodle House", "library", "home", "classroom", "hotel_lobby"]
ACTS = ["dining", "travel", "studying", "shopping", "meeting", "photo_taking"]


def build_memory(n: int) -> PersonMemory:
    """构造含 n 条 moment 的临时记忆库（合成数据）。"""
    if Path(TMP_DIR).exists():
        shutil.rmtree(TMP_DIR)
    mem = PersonMemory(memory_dir=TMP_DIR)

    for i in range(n):
        mid = f"m_{i}"
        loc = random.choice(LOCS)
        act = random.choice(ACTS)
        moment = {
            "id": mid,
            "timestamp": f"2026-09-{(i % 28) + 1:02d}T10:00:00",
            "scene": f"At {loc}, the user is {act} with others. scene#{i}",
            "user_action": "walking, looking around and talking",
            "needs": [
                {"need": f"needs help with {act} at {loc} #{i}", "confidence": 0.8}
            ],
            "solutions": [
                {"need": f"needs help with {act} #{i}", "solution": "show AR guidance"}
            ],
            "feedback": {"confirmed": False, "corrections": {}, "user_rating": None},
            "highlighted": False,
            "layer": 3,
        }
        mem.memory["layer7"]["moments"][mid] = moment
        mem.memory["layer7"]["people"].setdefault(random.choice(PEOPLE), []).append(mid)
        mem.memory["layer7"]["locations"].setdefault(loc, []).append(mid)
        mem.memory["layer7"]["activity_events"].setdefault(act, []).append(mid)
        mem.memory["layer3"].append(moment)

    mem.memory["metadata"]["total_moments"] = n
    mem._save()
    return mem


def pct(values, p):
    """计算百分位数（线性插值）。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def measure(fn, repeats=REPEATS):
    """重复执行 fn，返回 (p50, p95, mean) 毫秒。"""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return pct(samples, 50), pct(samples, 95), statistics.mean(samples)


def main():
    random.seed(42)
    results = {}

    for n in SCALES:
        print(f"\n{'=' * 60}\n构建记忆库: n = {n} 条 moment\n{'=' * 60}")
        mem = build_memory(n)

        # 冷启动加载耗时（重新从磁盘 load）
        t0 = time.perf_counter()
        mem2 = PersonMemory(memory_dir=TMP_DIR)
        load_ms = (time.perf_counter() - t0) * 1000

        q_p50, q_p95, q_mean = measure(
            lambda: mem.query("dining library Alice", top_k=5)
        )
        r_p50, r_p95, r_mean = measure(
            lambda: mem.retrieve(people=["Alice"], location="library", activity="dining")
        )
        c_p50, c_p95, c_mean = measure(
            lambda: mem.get_context_for_analysis(people=["Alice"], location="library")
        )

        # 全量 context（当前 Phase A 实际在用的 get_all_memory）
        a_p50, a_p95, a_mean = measure(lambda: mem.get_all_memory())

        results[n] = {
            "load_ms": round(load_ms, 3),
            "query": {"p50": round(q_p50, 3), "p95": round(q_p95, 3), "mean": round(q_mean, 3)},
            "retrieve": {"p50": round(r_p50, 3), "p95": round(r_p95, 3), "mean": round(r_mean, 3)},
            "context": {"p50": round(c_p50, 3), "p95": round(c_p95, 3), "mean": round(c_mean, 3)},
            "all_memory": {"p50": round(a_p50, 3), "p95": round(a_p95, 3), "mean": round(a_mean, 3)},
        }

        print(f"  _load()               : {load_ms:8.2f} ms")
        print(f"  query()               : p50={q_p50:7.2f} ms  p95={q_p95:7.2f} ms")
        print(f"  retrieve()            : p50={r_p50:7.2f} ms  p95={r_p95:7.2f} ms")
        print(f"  get_context_for_...() : p50={c_p50:7.2f} ms  p95={c_p95:7.2f} ms")
        print(f"  get_all_memory()      : p50={a_p50:7.2f} ms  p95={a_p95:7.2f} ms")

    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✅ 结果已写入: {out}")

    # 清理临时目录
    if Path(TMP_DIR).exists():
        shutil.rmtree(TMP_DIR)
        print(f"🧹 已清理临时目录 {TMP_DIR}")


if __name__ == "__main__":
    main()
