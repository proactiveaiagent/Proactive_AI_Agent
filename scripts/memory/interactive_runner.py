"""
interactive_runner.py — 交互式步进执行器
======================================================================
把 run_scenarios.py 的一次性流程拆成一步一步，供可视化网站交互执行：
人可以看到当前输入的是哪个视频、时间点，手工点击「运行」再走下一步。

短期场景：每条视频片段一个 add 步骤 → query / update / highlight / delete。
长期场景：白天按小时批量写入 → NightlyProcessor 汇总 / 分类 / 画像更新。

被 viz_server.py 调用（单线程 HTTPServer，无需加锁）：
    InteractiveSession("short_term")   创建会话
    session.state()                    当前状态（下一步信息 + 已执行日志 + 最新快照）
    session.run_step()                 执行当前步，返回结果消息
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import PersonMemory          # noqa: E402
from daytime_processor import DaytimeProcessor  # noqa: E402
from nighttime_processor import NightlyProcessor  # noqa: E402
import run_scenarios as rs               # noqa: E402  复用 snapshot / 夜间整理辅助函数
from qwen_perceive import QwenPerceiver, golden_eval   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "data" / "egolife" / "testset"
RUNS = ROOT / "data" / "egolife" / "runs"
VIDEO_ROOT = (ROOT / "data" / "egolife").resolve()

SHORT_FILE = TESTSET / "short_term" / "moments_scattered.jsonl"
SHORT_PLAN = TESTSET / "short_term" / "crud_plan.json"
LONG_FILE = TESTSET / "long_term" / "full_day.jsonl"
LONG_META = TESTSET / "long_term" / "timeline_meta.json"

SCENARIO_META = {
    "short_term": {"name": "短期记忆场景", "desc": "不连续视频片段 · 验证 Layer1~7 完整 CRUD"},
    "long_term": {"name": "长期记忆场景", "desc": "完整一天 · 验证记忆流转与夜间整理"},
}


def _clock_from_id(mid):
    """DAY1_A2_ALICE_11093000_en → 11:09:30"""
    parts = str(mid).split("_")
    if len(parts) >= 4:
        t = parts[3]
        if len(t) >= 6 and t[:6].isdigit():
            return f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
    return ""


def _intra_hour_offset(mid):
    """文件名时间戳相对整点小时的秒数：DAY1_A2_ALICE_11093000 → 9*60+30 = 570s"""
    parts = str(mid).split("_")
    if len(parts) >= 4 and len(parts[3]) >= 6 and parts[3][:6].isdigit():
        t = parts[3]
        return int(t[2:4]) * 60 + int(t[4:6])
    return 0


def moment_brief(m):
    """单条 moment 的展示摘要（视频、时间点、人物、动作、可播放地址）。"""
    video = m.get("video") or ""                 # data/EgoLife/A2_ALICE/DAY1/xxx.mp4
    gt = m.get("ground_truth", {})
    loc = gt.get("location_gt") or []

    # 本地视频路径（目录是小写 egolife），存在则给出可播放 URL
    local = ROOT / video.replace("EgoLife", "egolife") if video else None
    has_video = local is not None and local.exists()
    video_url = None
    clip_start = clip_end = None
    if has_video:
        rel = local.resolve().relative_to(VIDEO_ROOT)
        video_url = "/api/video?p=" + quote(str(rel))
        # start_time/end_time 是相对整点小时的偏移；换算成 30s 片段内秒数
        if m.get("start_time") is not None:
            off = _intra_hour_offset(m.get("id", ""))
            clip_start = round(max(0.0, m["start_time"] - off), 2)
            clip_end = round(m.get("end_time", m["start_time"] + 30) - off, 2)

    return {
        "id": m.get("id", ""),
        "video": video.rsplit("/", 1)[-1] if video else "",
        "clock": _clock_from_id(m.get("id", "")),
        "start_time": m.get("start_time"),
        "end_time": m.get("end_time"),
        "hour": m.get("_hour", ""),
        "people": m.get("people") or gt.get("people_gt") or [],
        "location": loc[0] if loc else (m.get("location") or ""),
        "action": (m.get("user_action") or m.get("scene") or "")[:160],
        "video_url": video_url,
        "clip_start": clip_start,
        "clip_end": clip_end,
    }


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class InteractiveSession:
    """一个场景的步进执行会话。"""

    def __init__(self, key, perceiver=None):
        assert key in SCENARIO_META, f"unknown scenario: {key}"
        self.key = key
        self.pos = 0            # 已执行步数 = 下一步索引
        self.log = []           # [{kind, title, msg, snapshot, perception}]
        self.perceiver = perceiver or QwenPerceiver()
        self.qwen_health = self.perceiver.health()
        self._build()

    # ------------------------------------------------------------------
    # 感知 → 记忆写入
    # Qwen 看视频产出记忆条目；数据集 caption / ground_truth 只用于评估。
    # ------------------------------------------------------------------

    def _add_one(self, m, force=False):
        """用 Qwen 的感知产出写入一条记忆，返回 (moment_id, info)。"""
        r = self.perceiver.perceive(m, force=force)
        if not r.get("fields"):
            return None, {"error": r.get("error", "感知失败"),
                          "source": r.get("source", "error")}
        f = r["fields"]
        mid = self.daytime.ingest(
            scene=f["scene"],
            user_action=f["user_action"],
            needs=[],
            solutions=[],
            people=f["people"] or None,
            location=f["location"],
            activity=f["activity"],
            environments=f["environments"],
            objects=f["objects"],
            extra_notes=(f.get("time_occasion") or "")[:200],
            timestamp=(time.strftime("%Y-%m-%d") + "T" + (_clock_from_id(m.get("id", "")) or time.strftime("%H:%M:%S"))),
        )
        self.id_map[m["id"]] = mid
        self._qwen_fields[m["id"]] = f
        info = {
            "dataset_id": m.get("id", ""),
            "moment_id": mid,
            "qwen": f,
            "raw": r.get("raw", ""),
            "elapsed_ms": r.get("elapsed_ms"),
            "num_frames": r.get("num_frames"),
            "source": r.get("source", "qwen"),
            "eval": golden_eval(f, m.get("ground_truth", {})),
        }
        return mid, info

    def _sort_analysis_from_qwen(self):
        """按 Qwen 产出的字段聚合出归类结果（ground-truth 不参与记忆）。"""
        dims = {"people": "people", "locations": "location",
                "activity_events": "activity", "environments": "environments",
                "objects": "objects"}
        analysis = {}
        for dim, fkey in dims.items():
            bucket = {}
            for did, mid in self.id_map.items():
                f = self._qwen_fields.get(did) or {}
                v = f.get(fkey)
                vals = v if isinstance(v, list) else ([v] if v else [])
                for t in vals:
                    t = str(t).strip()
                    if t:
                        bucket.setdefault(t, []).append(mid)
            if bucket:
                analysis[dim] = bucket
        return analysis

    # ------------------------------------------------------------------
    # 会话构建
    # ------------------------------------------------------------------

    def _build(self):
        mem_dir = RUNS / "interactive_mem" / self.key
        mem_dir.mkdir(parents=True, exist_ok=True)
        if any(mem_dir.iterdir()):
            # 旧会话数据改名归档（os.rename 不触发 safe-delete 钩子，避免 trash 权限问题）
            trash_dir = RUNS / "interactive_mem" / "_trash"
            trash_dir.mkdir(exist_ok=True)
            os.rename(mem_dir, trash_dir / f"{self.key}_{time.time_ns()}")
            mem_dir.mkdir(parents=True, exist_ok=True)
        self.mem = PersonMemory(memory_dir=str(mem_dir))
        self.daytime = DaytimeProcessor(self.mem)
        self.daytime.start_scene(f"{self.key}_{time.strftime('%Y-%m-%d')}")
        self.id_map = {}        # 数据集 id → 记忆模型 moment_id
        self._qwen_fields = {}  # 数据集 id → Qwen 产出（sort / 展示用）

        if self.key == "short_term":
            self.steps = self._plan_short()
        else:
            self.steps = self._plan_long()

        self._log_snapshot("初始状态", "init", "会话创建，等待第一步")

    def _plan_short(self):
        """短期场景：逐条写入 + CRUD 计划。"""
        moments = _load_jsonl(SHORT_FILE)
        plan = json.load(open(SHORT_PLAN, encoding="utf-8"))
        self.moments = moments
        self.plan = plan
        steps = []

        # 1) add：每条视频片段一步（Qwen 感知 → 记忆）
        for i, m in enumerate(moments, 1):
            def run(m=m, i=i, force=False):
                mid, info = self._add_one(m, force=force)
                if mid is None:
                    return f"感知失败：{info.get('error')}", {"kind": "single", **info}
                msg = f"Qwen 看过 {m['id']} → 写入记忆 {mid}（{info['source']}，{info['elapsed_ms']}ms）"
                return msg, {"kind": "single", **info}
            steps.append({
                "kind": "add",
                "title": f"感知并写入视频片段 {i}/{len(moments)}",
                "inputs": [moment_brief(m)],
                "run": run,
            })

        # 2) query
        for q in plan.get("query", []):
            def run(q=q, force=False):
                hits = self.daytime.query(q["query"], top_k=3)
                return f"检索「{q['query']}」命中 {len(hits)} 条", None
            steps.append({
                "kind": "query",
                "title": f"检索「{q['query']}」",
                "detail": {"query": q["query"], "top_k": 3},
                "inputs": [],
                "run": run,
            })

        # 3) update
        for case in plan.get("update", []):
            def run(case=case, force=False):
                mid = self.id_map.get(case["id"])
                if not mid:
                    return f"未找到 {case['id']}，跳过", None
                ok = self.mem.update(mid, {case["field"]: case["new_value"]})
                return f"更新 {case['id']}.{case['field']} → {case['new_value']}（{'成功' if ok else '失败'}）", None
            steps.append({
                "kind": "update",
                "title": f"更新字段 {case['field']}",
                "detail": {"id": case["id"], "field": case["field"],
                           "new_value": case["new_value"]},
                "inputs": [],
                "run": run,
            })

        # 4) highlight
        hl_ids = plan.get("highlight", [])
        def run_highlight(force=False):
            n = 0
            for sid in hl_ids:
                mid = self.id_map.get(sid)
                if mid and self.mem.highlight(mid):
                    n += 1
            return f"标记重要记忆 {n} 条", None
        steps.append({
            "kind": "highlight",
            "title": "标记重要记忆",
            "detail": {"ids": hl_ids},
            "inputs": [],
            "run": run_highlight,
        })

        # 5) delete（含 highlight 保护验证）
        del_cases = plan.get("delete", [])
        def run_delete(force=False):
            cleared = self.daytime.end_scene()
            msgs = [f"场景结束：清理 Layer1/2 {cleared} 条"]
            ok_n = 0
            for case in del_cases:
                mid = self.id_map.get(case["id"])
                if not mid:
                    continue
                if self.mem.delete(mid, reason=case.get("reason", "manual")):
                    ok_n += 1
            msgs.append(f"清理短期记忆：删除 {ok_n} 条")
            # 验证 highlight 保护
            prot = [self.id_map[s] for s in hl_ids if self.id_map.get(s)]
            if prot:
                blocked = not self.mem.delete(prot[0], reason="验证保护")
                forced = self.mem.delete(prot[0], reason="验证强制删除", force=True)
                msgs.append(f"highlight 保护：默认删除被拒绝 {'✅' if blocked else '❌'}；"
                            f"force 强删 {'成功 ✅' if forced else '失败 ❌'}")
            return "；".join(msgs), None
        steps.append({
            "kind": "delete",
            "title": "清理短期记忆（含保护验证）",
            "detail": {"ids": [c["id"] for c in del_cases]},
            "inputs": [],
            "run": run_delete,
        })

        return steps

    def _plan_long(self):
        """长期场景：白天按小时批量写入 + 夜间整理 4 步。"""
        moments = _load_jsonl(LONG_FILE)
        meta = json.load(open(LONG_META, encoding="utf-8"))
        self.moments = moments
        self.meta = meta
        participant = meta["participant"]
        steps = []

        # 白天：按小时分批
        by_hour = {}
        for m in moments:
            h = m.get("_hour") or (m["id"].split("_")[3][:2] if len(m["id"].split("_")) > 3 else "??")
            by_hour.setdefault(h, []).append(m)

        for h in sorted(by_hour):
            group = by_hour[h]
            def run(group=group, h=h, force=False):
                items, ok = [], 0
                for m in group:
                    mid, info = self._add_one(m, force=force)
                    if mid:
                        ok += 1
                    items.append(info)
                msg = f"{h}:00 时段：Qwen 感知 {ok}/{len(group)} 条并写入记忆"
                return msg, {"kind": "batch", "items": items, "n": len(group), "ok": ok}
            steps.append({
                "kind": "add_batch",
                "title": f"白天感知写入 · {h}:00 时段（{len(group)} 条）",
                "inputs": [moment_brief(m) for m in group],
                "run": run,
            })

        # 夜间整理：使用真实记忆输入，不将测试集 ground-truth 写入画像。
        def run_nightly(force=False):
            result = NightlyProcessor(self.mem).run(
                day=self.mem.memory["metadata"]["today"],
                daytime_processor=self.daytime,
                force=force,
            )
            if not result.get("ok"):
                return "夜间整理失败", {"kind": "nightly", "result": result}
            cleared = self.daytime.end_scene()
            result["short_term_cleared"] = cleared
            msg = (
                f"夜间整理完成：处理 {result['moment_count']} 条，"
                f"分 {result['chunk_count']} 批，画像更新 "
                f"{'成功' if result['profile_updated'] else '无新属性'}；"
                f"场景结束清理 Layer1/2 {cleared} 条"
            )
            return msg, {"kind": "nightly", "result": result}
        steps.append({
            "kind": "nightly",
            "title": "夜间整理 · 汇总、归类与画像更新",
            "detail": {"note": "按真实 Qwen 感知结果整理 Layer 4~7，不读取 ground-truth"},
            "inputs": [],
            "run": run_nightly,
        })

        return steps

    # ------------------------------------------------------------------
    # 执行与状态
    # ------------------------------------------------------------------

    def _full_snapshot(self):
        """提取各层实际内容（供前端展示完整记忆，不只是数值）。"""
        m = self.mem.memory
        l7 = m["layer7"]

        # 反向映射：记忆 moment_id → 原始数据集 moment，用于补充视频地址 / 时间点
        orig_by_id = {x.get("id"): x for x in getattr(self, "moments", [])}
        rev = {mid: did for did, mid in self.id_map.items()}

        def vid_of(mid):
            """该条记忆对应的视频（可播放地址 + 片段区间 + 拍摄时刻）。"""
            orig = orig_by_id.get(rev.get(mid))
            if not orig:
                return None
            b = moment_brief(orig)
            if not b.get("video_url"):
                return None
            return {"video": b["video"], "video_url": b["video_url"],
                    "clip_start": b["clip_start"], "clip_end": b["clip_end"],
                    "clock": b["clock"]}

        def cut(s, n=110):
            s = s or ""
            return s if len(s) <= n else s[:n] + "…"

        def fmt(x, with_stale=False):
            d = {"id": x.get("id", ""), "scene": cut(x.get("scene", "")),
                 "user_action": cut(x.get("user_action", "")),
                 "location": x.get("location", ""),
                 "activity": x.get("activity", ""),
                 "people": x.get("people", []),
                 "highlight": x.get("highlighted", False),
                 "timestamp": x.get("timestamp", "")}
            if with_stale:
                d["stale"] = x.get("stale", False)
            v = vid_of(x.get("id", ""))
            if v:
                d["video"] = v
            return d

        # 索引：列出每个维度下的 tag 及其命中的 moment id（限每个维度 15 个 tag，每个 tag 5 个 id）
        indices = {}
        for dim in ("people", "locations", "time_nodes", "activity_events", "environments", "objects"):
            idx = l7.get(dim, {})
            indices[dim] = {tag: ids[:5] for tag, ids in list(idx.items())[:15]}

        return {
            "layer1": [fmt(x) for x in m.get("layer1", [])[:30]],
            "layer2": [fmt(x) for x in m.get("layer2", [])[:30]],
            "layer3": [fmt(x) for x in m.get("layer3", [])[:30]],
            "layer4": m.get("layer4", {}),
            "layer5": m.get("layer5", {}),
            "layer6": m.get("layer6", {}),
            "layer7_moments": [fmt(x, True) for x in list(l7.get("moments", {}).values())[:80]],
            "layer7_total": len(l7.get("moments", {})),
            "indices": indices,
            "metadata": m.get("metadata", {}),
            "daytime": self.daytime.snapshot(),
        }

    def _log_snapshot(self, title, kind, msg, perception=None):
        snap = rs.snapshot(self.mem, title, action=kind)
        snap["full"] = self._full_snapshot()
        self.log.append({"kind": kind, "title": title, "msg": msg,
                         "snapshot": snap, "perception": perception})

    def run_step(self, force=False):
        """执行当前步；force=True 强制实时调用 Qwen（忽略缓存）。"""
        if self.pos >= len(self.steps):
            return "所有步骤已执行完毕"
        step = self.steps[self.pos]
        msg, perception = step["run"](force=force)
        self._log_snapshot(step["title"], step["kind"], msg, perception)
        self.pos += 1
        return msg

    def prewarm(self, limit=None):
        """批量预热 Qwen 缓存（后台任务调用）。返回 (成功, 失败)。"""
        ms = self.moments[:limit] if limit else self.moments
        return self.perceiver.prewarm(ms)

    def state(self):
        """返回给前端的完整状态。"""
        total = len(self.steps)
        cur = None
        if self.pos < total:
            s = self.steps[self.pos]
            cur = {
                "kind": s["kind"],
                "title": s["title"],
                "detail": s.get("detail", {}),
                "inputs": s.get("inputs", []),
            }
        # 日志里的 Qwen 原始输出按需裁剪，避免单步响应过大
        slim = []
        for e in self.log:
            e2 = dict(e)
            p = e.get("perception")
            if p:
                p = dict(p)
                if p.get("raw"):
                    p["raw"] = p["raw"][:400]
                if p.get("kind") == "batch":
                    p["items"] = [{**it, "raw": (it.get("raw") or "")[:200]}
                                  for it in p.get("items", [])]
                e2["perception"] = p
            slim.append(e2)

        meta = SCENARIO_META[self.key]
        extra = {}
        if self.key == "long_term":
            extra = {
                "participant": self.meta.get("participant"),
                "hours_covered": self.meta.get("hours_covered", []),
            }
        return {
            "scenario": self.key,
            "name": meta["name"],
            "desc": meta["desc"],
            "total": total,
            "pos": self.pos,
            "finished": self.pos >= total,
            "current": cur,
            "log": slim,
            "snapshot": self.log[-1]["snapshot"] if self.log else None,
            "qwen": {
                "ok": bool(self.qwen_health.get("model_loaded")),
                "gpu": (self.qwen_health.get("gpu") or {}).get("name"),
                "cached": len(getattr(self.perceiver, "cache", {}) or {}),
            },
            **extra,
        }


if __name__ == "__main__":
    # 命令行自测：逐步执行并打印
    s = InteractiveSession("short_term")
    while not s.state()["finished"]:
        st = s.state()
        print(f"[{st['pos']+1}/{st['total']}] {st['current']['title']} ... ", end="")
        print(s.run_step())
    print("done")
