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

import copy
import json
import os
import sys
import threading
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
SHORT_EXPANDED_FILE = TESTSET / "short_term" / "moments_scattered_v2_120.jsonl"
SHORT_EXPANDED_PLAN = TESTSET / "short_term" / "crud_plan_v2.json"
LONG_FILE = TESTSET / "long_term" / "full_day.jsonl"
LONG_META = TESTSET / "long_term" / "timeline_meta.json"
LONG_PERSON_DIR = TESTSET / "long_term" / "person_sets"


def site_perceiver():
    """Use the memory-eval V3 cache for this visualization, with an env override."""
    configured = os.environ.get("EGOLIFE_QWEN_CACHE")
    if configured:
        cache_path = Path(configured)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
    else:
        cache_path = RUNS / "qwen_cache_memory_prompt_v3.json"
        if not cache_path.exists():
            cache_path = RUNS / "qwen_cache.json"
    return QwenPerceiver(cache_file=cache_path)


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
        "scene": m.get("scene") or "",
        "action": (m.get("user_action") or m.get("scene") or ""),
        "video_url": video_url,
        "clip_start": clip_start,
        "clip_end": clip_end,
    }


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


PARTICIPANT_NAMES = {
    "A1_JAKE": "Jake",
    "A2_ALICE": "Alice",
    "A3_TASHA": "Tasha",
    "A4_LUCIA": "Lucia",
    "A5_KATRINA": "Katrina",
    "A6_SHURE": "Shure",
}


def interactive_dataset_options():
    """Datasets available in the step-by-step runner."""
    options = []
    for dataset, path, plan_path, title in (
        ("base", SHORT_FILE, SHORT_PLAN, "短期 CRUD 演示集"),
        ("expanded", SHORT_EXPANDED_FILE, SHORT_EXPANDED_PLAN, "短期扩展集"),
    ):
        rows = _load_jsonl(path)
        counts = {}
        for row in rows:
            parts = str(row.get("id", "")).split("_")
            participant = "_".join(parts[1:3]) if len(parts) >= 3 else ""
            if participant:
                counts[participant] = counts.get(participant, 0) + 1
        per_person = next(iter(counts.values()), 0)
        purpose = (
            f"{len(rows)} 段，6 人各 {per_person} 段；原 CRUD 步进演示集。"
            if dataset == "base" else
            f"{len(rows)} 段，6 人各 {per_person} 段；扩展短期测评集，可逐段写入并执行 CRUD。"
        )
        options.append({
            "value": f"short_term|{dataset}|all",
            "scenario": "short_term",
            "dataset": dataset,
            "participant": None,
            "label": f"{title} · 全部人物（{len(rows)} 段）",
            "note": purpose,
        })
        for participant in PARTICIPANT_NAMES:
            if participant not in counts:
                continue
            name = PARTICIPANT_NAMES[participant]
            options.append({
                "value": f"short_term|{dataset}|{participant}",
                "scenario": "short_term",
                "dataset": dataset,
                "participant": participant,
                "label": f"{title} · {name}（{counts[participant]} 段）",
                "note": f"{purpose} 当前只加载 {name} 的片段。",
            })

    long_index = json.load(open(LONG_PERSON_DIR / "index.json", encoding="utf-8"))
    for meta in long_index.get("datasets", []):
        participant = meta["participant"]
        name = PARTICIPANT_NAMES.get(participant, participant)
        complete = meta.get("complete", False)
        coverage = round(float(meta.get("coverage", 0)) * 100, 1)
        if complete:
            label = f"长期完整日集 · {name}（{meta['available_count']} 段）"
            note = (
                f"DAY1 完整视频集：{meta['available_count']} 段，覆盖 "
                f"{len(meta.get('hours_covered', []))} 个有数据时段。"
            )
        else:
            label = (
                f"长期部分集 · {name}（{meta['available_count']}/"
                f"{meta['total_candidates']} 段 · {coverage}%）"
            )
            note = (
                f"DAY1 视频不完整：仅 {meta['available_count']}/"
                f"{meta['total_candidates']} 段（{coverage}%）可运行；"
                "结果仅代表本地已下载片段，不是完整日测评。"
            )
        options.append({
            "value": f"long_term|long_day|{participant}",
            "scenario": "long_term",
            "dataset": "long_day",
            "participant": participant,
            "label": label,
            "note": note,
        })
    return options


class InteractiveSession:
    """一个场景的步进执行会话。"""

    def __init__(self, key, participant=None, dataset="base", perceiver=None):
        assert key in SCENARIO_META, f"unknown scenario: {key}"
        self.key = key
        self.participant = participant
        self.dataset_key = dataset
        self.pos = 0            # 已执行步数 = 下一步索引
        self.auto_paused = False  # 连续运行只允许执行写入步骤
        self.step_cancelled = False
        self.log = []           # [{kind, title, msg, snapshot, perception}]
        self._cancel_lock = threading.Lock()
        self._active_step_cancel = None
        self._active_step_force = False
        self._pending_cancel = False
        self.perceiver = perceiver or site_perceiver()
        self.qwen_health = self.perceiver.health()
        self._build()

    def _install_operation_audit(self):
        """Record actual memory API calls made by this interactive session."""
        self.operation_log = []
        self._active_step_no = None
        self._active_step_title = None
        tracked = (
            "add", "query", "query_hierarchical", "update", "update_feedback",
            "highlight", "delete", "sort", "combine", "compress", "update_profile",
            "find_merge_candidates", "merge_moments",
        )
        for name in tracked:
            original = getattr(self.mem, name, None)
            if not callable(original):
                continue

            def audited(*args, __name=name, __original=original, **kwargs):
                started = time.perf_counter()
                before_moment = None
                before_profile = None
                if __name in ("update", "update_feedback", "highlight", "delete"):
                    mid = str(args[0] if args else kwargs.get("moment_id", ""))
                    item = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid)
                    before_moment = copy.deepcopy(item) if isinstance(item, dict) else None
                if __name == "merge_moments":
                    archive = self.mem.memory.get("layer7", {}).get("moments", {})
                    source_id = str(args[0] if args else kwargs.get("source_id", ""))
                    target_id = str(args[1] if len(args) > 1 else kwargs.get("target_id", ""))
                    before_moment = {
                        "source": copy.deepcopy(archive.get(source_id, {})),
                        "target": copy.deepcopy(archive.get(target_id, {})),
                    }
                if __name == "update_profile":
                    before_profile = copy.deepcopy(
                        self.mem.memory.get("layer6", {}).get("profile", {})
                    )
                try:
                    result = __original(*args, **kwargs)
                except Exception as exc:
                    self._append_operation(__name, args, kwargs, None,
                                           started, error=exc,
                                           before_moment=before_moment,
                                           before_profile=before_profile)
                    raise
                self._append_operation(
                    __name, args, kwargs, result, started,
                    before_moment=before_moment,
                    before_profile=before_profile,
                )
                return result

            setattr(self.mem, name, audited)

    def _append_operation(self, name, args, kwargs, result, started,
                          error=None, before_moment=None, before_profile=None):
        """Append a factual audit event with the affected memory content."""
        status = "failed" if error else (
            "blocked" if result is False or
            (isinstance(result, dict) and result.get("ok") is False)
            else "success"
        )
        if name == "update" and result is False:
            mid = str(args[0] if args else kwargs.get("moment_id", ""))
            exists = mid in self.mem.memory.get("layer7", {}).get("moments", {})
            status = "no_change" if exists else "blocked"
        summary = self._operation_summary(
            name, args, kwargs, result, error, before_moment, before_profile
        )
        self.operation_log.append({
            "seq": len(self.operation_log) + 1,
            "time": time.strftime("%H:%M:%S"),
            "step": self._active_step_no,
            "step_title": self._active_step_title or "",
            "operation": name,
            "status": status,
            "summary": summary,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        })

    @staticmethod
    def _brief_value(value, limit=110):
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value if value is not None else "—")
        return " ".join(text.split())[:limit]

    def _moment_detail(self, moment):
        """Return the visible, factual fields of a memory without mutating it."""
        if not isinstance(moment, dict):
            return {}
        return {
            "id": str(moment.get("id") or ""),
            "timestamp": moment.get("timestamp") or "",
            "scene": moment.get("scene") or "",
            "action": moment.get("user_action") or "",
            "people": moment.get("people") or [],
            "location": moment.get("location") or "",
            "activity": moment.get("activity") or "",
            "objects": moment.get("objects") or [],
            "notes": moment.get("notes") or "",
            "highlighted": bool(moment.get("highlighted")),
            "highlight_source": moment.get("highlight_source") or "",
            "highlighted_at": moment.get("highlighted_at") or "",
            "merged_into": moment.get("merged_into") or "",
            "merged_sources": moment.get("merged_sources") or [],
            "merge_history": moment.get("merge_history") or [],
        }

    def _operation_summary(self, name, args, kwargs, result, error,
                           before_moment=None, before_profile=None):
        if error:
            return f"{name} 失败：{str(error)[:140]}"
        arg0 = args[0] if args else kwargs.get("moment_id")
        if name == "add":
            mid = str(result or "")
            item = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            text = str(item.get("scene") or item.get("user_action") or "").strip()
            text = text.replace("\n", " ")[:100]
            return f"写入记忆 {mid}" + (f" · {text}" if text else "")
        if name in ("query", "query_hierarchical"):
            query = str(arg0 if arg0 is not None else kwargs.get("query_text", ""))[:80]
            hits = result if isinstance(result, list) else []
            examples = []
            for hit in hits[:3]:
                if not isinstance(hit, dict):
                    continue
                moment = hit.get("moment") if isinstance(hit.get("moment"), dict) else hit
                mid = str(moment.get("id") or "无 ID")
                content = str(moment.get("scene") or moment.get("user_action") or "").replace("\n", " ")
                layer = f"L{hit['layer']} · " if hit.get("layer") else ""
                examples.append(f"{layer}{mid}: {content[:100]}")
            detail = "；命中：" + "；".join(examples) if examples else ""
            return f"检索「{query}」· 命中 {len(hits)} 条{detail}"
        if name == "find_merge_candidates":
            candidates = result if isinstance(result, list) else []
            examples = [
                f"{row.get('source_id')}→{row.get('target_id')} "
                f"({row.get('similarity', 0):.0%})"
                for row in candidates[:3]
            ]
            suffix = " · " + "；".join(examples) if examples else ""
            return f"相似事件候选 {len(candidates)} 组（需人工确认）{suffix}"
        if name == "merge_moments":
            details = result if isinstance(result, dict) else {}
            if not details.get("ok"):
                return str(details.get("message") or "合并未执行")
            if details.get("status") == "already_merged":
                return f"来源 {details.get('source_id')} 已归并到 {details.get('target_id')}（无重复改动）"
            source_id = details.get("source_id") or "?"
            target_id = details.get("target_id") or "?"
            fields = "、".join(details.get("merged_fields") or []) or "没有新增列表字段"
            conflicts = len(details.get("scalar_conflicts") or [])
            media = details.get("media_preserved") or []
            retained = sum(bool(item.get("visual") or item.get("audio")) for item in media)
            return (f"合并事件 {source_id} → {target_id} · 列表字段：{fields}"
                    f" · 标量冲突留在来源并记录 {conflicts} 项"
                    f" · {retained}/{len(media)} 条来源保留多模态引用")
        if name == "update":
            mid = str(arg0 or "")
            updates = args[1] if len(args) > 1 else kwargs.get("updates", {})
            exists = mid in self.mem.memory.get("layer7", {}).get("moments", {})
            if result is False:
                return f"更新记忆 {mid} · 字段未变化" if exists else f"更新记忆 {mid} · 未找到"
            after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            changes = []
            if isinstance(updates, dict):
                for field, new_value in updates.items():
                    old_value = (before_moment or {}).get(field)
                    actual_value = after.get(field)
                    changes.append(
                        f"{field}：{self._brief_value(old_value)} → {self._brief_value(actual_value)}"
                    )
            return f"更新记忆 {mid}" + (" · " + "；".join(changes) if changes else "")
        if name == "update_feedback":
            mid = str(arg0 or "")
            corrections = kwargs.get("corrections", {})
            if len(args) > 1 and isinstance(args[1], dict):
                corrections = args[1]
            detail = "；".join(
                f"{key}：{self._brief_value((before_moment or {}).get(key))} → {self._brief_value(value)}"
                for key, value in corrections.items()
            ) if isinstance(corrections, dict) else ""
            confirmed = kwargs.get("confirmed", False)
            rating = kwargs.get("user_rating")
            return f"反馈修正 {mid}" + (f" · {detail}" if detail else "") + (
                f" · 确认={'是' if confirmed else '否'} · 评分={rating if rating is not None else '未评分'}"
            )
        if name == "highlight":
            mid = str(arg0 or "")
            item = before_moment or self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            content = str(item.get("scene") or item.get("user_action") or "").replace("\n", " ")[:100]
            result_text = "已标记重点" if result else "标记失败"
            return f"{result_text} {mid}" + (f" · {content}" if content else "")
        if name == "delete":
            mid = str(arg0 or "")
            reason = kwargs.get("reason", "manual")
            item = before_moment or {}
            content = str(item.get("scene") or item.get("user_action") or "").replace("\n", " ")[:100]
            result_text = "已删除" if result else "被保护/未找到"
            return f"{result_text} {mid} · 原因：{reason}" + (f" · 内容：{content}" if content else "")
        if name == "sort":
            analysis = arg0 if isinstance(arg0, dict) else {}
            tags = sum(len(v) for v in analysis.values() if isinstance(v, dict))
            refs = sum(len(ids) for dim in analysis.values() if isinstance(dim, dict)
                       for ids in dim.values() if isinstance(ids, list))
            return f"整理索引 · {len(analysis)} 个维度、{tags} 个标签、{refs} 条引用"
        if name == "combine":
            details = result if isinstance(result, dict) else {}
            aliases = details.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            examples = []
            for alias in aliases[:4]:
                if not isinstance(alias, dict):
                    continue
                dimension = str(alias.get("dimension") or "索引")
                old = self._brief_value(alias.get("from"), 48)
                new = self._brief_value(alias.get("to"), 48)
                affected = int(alias.get("affected_moments") or 0)
                item = f"{dimension}「{old}」→「{new}」·影响 {affected} 条"
                sample_ids = alias.get("sample_moment_ids")
                if isinstance(sample_ids, list) and sample_ids:
                    item += "·例 " + ", ".join(str(mid) for mid in sample_ids[:3])
                    if affected > len(sample_ids[:3]):
                        item += f" 等 {affected} 条"
                if alias.get("overlap_moments"):
                    item += f"·两索引共有 {alias['overlap_moments']} 条"
                if alias.get("field_updates"):
                    item += f"·字段更新 {alias['field_updates']} 条"
                if alias.get("list_duplicates_removed"):
                    item += f"·列表去重 {alias['list_duplicates_removed']} 项"
                examples.append(item)
            summary = f"规范化标签 {len(aliases)} 组"
            if examples:
                summary += " · " + "；".join(examples)
                if len(aliases) > len(examples):
                    summary += f"；另有 {len(aliases) - len(examples)} 组"
            elif not aliases:
                summary += "（无有效别名需要合并）"
            if details.get("updated_moments"):
                summary += f" · 记忆记录更新 {details['updated_moments']} 条"
            return summary
        if name == "compress":
            payload = arg0 if isinstance(arg0, dict) else {}
            layers = "、".join(k.upper() for k in payload if k in ("layer4", "layer5", "layer6"))
            return f"压缩整理摘要 · 更新 {layers or '长期摘要'}"
        if name == "update_profile":
            fields = list(arg0) if isinstance(arg0, dict) else []
            profile = result if isinstance(result, (dict, list)) else self.mem.memory.get("layer6", {}).get("profile", {})
            def count_attrs(node):
                if isinstance(node, dict):
                    if "value" in node and "stale" in node:
                        return (1, int(bool(node.get("stale"))))
                    totals = [count_attrs(value) for value in node.values()]
                elif isinstance(node, list):
                    totals = [count_attrs(value) for value in node]
                else:
                    return (0, 0)
                return (sum(x[0] for x in totals), sum(x[1] for x in totals))
            total, stale = count_attrs(profile)
            detail = f"画像 {total} 项 · 过时 {stale} 项" if total else "暂无画像字段 · 衰减状态已检查"

            def conflicts(node, path=""):
                found = []
                if isinstance(node, dict) and "value" in node:
                    for old_value in node.get("history", []) or []:
                        if not isinstance(old_value, dict):
                            continue
                        found.append({
                            "path": path,
                            "old": old_value.get("value"),
                            "new": node.get("value"),
                            "resolution": old_value.get("resolution") or {},
                        })
                elif isinstance(node, dict):
                    for key, value in node.items():
                        found.extend(conflicts(value, f"{path}.{key}" if path else str(key)))
                elif isinstance(node, list):
                    for index, value in enumerate(node):
                        found.extend(conflicts(value, f"{path}[{index + 1}]"))
                return found

            before_conflicts = conflicts(before_profile or {})
            before_signatures = {
                (item["path"], json.dumps(item["old"], ensure_ascii=False, sort_keys=True, default=str),
                 item["resolution"].get("resolved_at", ""))
                for item in before_conflicts
            }
            new_conflicts = [
                item for item in conflicts(profile)
                if (item["path"], json.dumps(item["old"], ensure_ascii=False, sort_keys=True, default=str),
                    item["resolution"].get("resolved_at", "")) not in before_signatures
            ]
            basis_labels = {
                "observations": "观测次数更多",
                "recency": "最近观察",
                "confidence": "置信度更高",
                "existing_value_tiebreak": "条件相同时保留旧值",
            }
            conflict_note = ""
            if new_conflicts:
                examples = []
                for item in new_conflicts[:3]:
                    resolution = item["resolution"]
                    examples.append(
                        f"{item['path']}「{self._brief_value(item['old'], 32)}」→"
                        f"「{self._brief_value(item['new'], 32)}」"
                        f"（{basis_labels.get(resolution.get('basis'), '裁决依据未记录')}）"
                    )
                conflict_note = f" · 新增冲突裁决 {len(new_conflicts)} 项：{'；'.join(examples)}"
                if len(new_conflicts) > len(examples):
                    conflict_note += f"；另有 {len(new_conflicts) - len(examples)} 项"
                conflict_note += " · 旧值保留在画像历史"
            names = f" · 字段 {chr(12289).join(map(str, fields[:8]))}" if fields else ""
            return f"更新用户画像 · {detail}{conflict_note}{names}"
        return name

    # ------------------------------------------------------------------
    # 感知 → 记忆写入
    # Qwen 看视频产出记忆条目；数据集 caption / ground_truth 只用于评估。
    # ------------------------------------------------------------------

    def cancel_current(self):
        """Interrupt the active real-time step; completed items remain available for retry."""
        with self._cancel_lock:
            event = self._active_step_cancel
            if event is None:
                if not self._active_step_force:
                    return {"requested": False, "model_cancel_sent": False}
                self._pending_cancel = True
            else:
                event.set()
        cancel_sent = bool(getattr(self.perceiver, "cancel_active", lambda: False)())
        return {"requested": True, "model_cancel_sent": cancel_sent}

    def _add_one(self, m, force=False):
        """用 Qwen 的感知产出写入一条记忆，返回 (moment_id, info)。"""
        dataset_id = m.get("id", "")
        existing_mid = self.id_map.get(dataset_id)
        if existing_mid:
            cached = self.perceiver.cache.get(dataset_id) or {}
            fields = self._qwen_fields.get(dataset_id) or cached.get("fields") or {}
            return existing_mid, {
                "dataset_id": dataset_id,
                "moment_id": existing_mid,
                "qwen": fields,
                "raw": cached.get("raw", ""),
                "elapsed_ms": cached.get("elapsed_ms"),
                "num_frames": cached.get("num_frames"),
                "source": "cache",
                "eval": golden_eval(fields, m.get("ground_truth", {})),
            }
        cancel_event = self._active_step_cancel
        r = self.perceiver.perceive(m, force=force, cancel_event=cancel_event)
        if not r.get("fields"):
            return None, {"error": r.get("error", "感知失败"),
                          "source": r.get("source", "error"),
                          "cancelled": bool(r.get("cancelled"))}
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
            source_evidence=f.get("source_evidence") or "",
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
        self._install_operation_audit()
        self.daytime = DaytimeProcessor(self.mem)
        self.daytime.start_scene(f"{self.key}_{time.strftime('%Y-%m-%d')}")
        self.id_map = {}        # 数据集 id → 记忆模型 moment_id
        self._qwen_fields = {}  # 数据集 id → Qwen 产出（sort / 展示用）

        if self.key == "short_term":
            self.steps = self._plan_short()
        else:
            self.steps = self._plan_long()

        self._log_snapshot("初始状态", "init", "会话创建，等待第一步")

    def _display_brief(self, m):
        """Current-step preview uses the same cached English fields that feed memory."""
        brief = moment_brief(m)
        cached = (self.perceiver.cache.get(m.get("id", "")) or {}).get("fields") or {}
        if cached:
            if cached.get("scene"):
                brief["scene"] = cached["scene"]
            if cached.get("user_action"):
                brief["action"] = cached["user_action"]
            for key in ("people", "location", "activity"):
                if cached.get(key):
                    brief[key] = cached[key]
        if not brief.get("action") or any("\u3400" <= ch <= "\u9fff" for ch in str(brief.get("action", ""))):
            brief["action"] = "English summary will appear after this step runs."
        return brief

    def _plan_short(self):
        """短期场景：逐条写入 + CRUD 计划，可按人物筛选当前演示集。"""
        if self.dataset_key not in ("base", "expanded"):
            raise ValueError(f"未知短期数据集：{self.dataset_key}")
        source_file, plan_file = (
            (SHORT_EXPANDED_FILE, SHORT_EXPANDED_PLAN)
            if self.dataset_key == "expanded" else (SHORT_FILE, SHORT_PLAN)
        )
        all_moments = _load_jsonl(source_file)
        all_plan = json.load(open(plan_file, encoding="utf-8"))
        if self.participant:
            moments = [
                m for m in all_moments
                if "_".join(str(m.get("id", "")).split("_")[1:3]) == self.participant
            ]
            if not moments:
                raise ValueError(f"短期交互集没有参与者 {self.participant}")
            def for_participant(item_id):
                return "_".join(str(item_id).split("_")[1:3]) == self.participant

            plan = {
                **all_plan,
                "add": [m.get("id") for m in moments],
                "update": [x for x in all_plan.get("update", [])
                           if for_participant(x.get("id"))],
                "highlight": [x for x in all_plan.get("highlight", [])
                              if for_participant(x)],
                "delete": [x for x in all_plan.get("delete", [])
                           if for_participant(x.get("id"))],
            }
            # 每个人物都保留完整的 CRUD 演示路径；没有预设操作时，
            # 从该人物自己的片段生成一条可追溯的演示操作。
            if not plan["update"]:
                for m in moments:
                    gt = m.get("ground_truth", {})
                    for field, gt_key in (
                        ("environments", "environments_gt"),
                        ("objects", "objects_gt"),
                        ("location", "location_gt"),
                        ("people", "people_gt"),
                        ("activity", "activity_gt"),
                    ):
                        value = gt.get(gt_key)
                        if isinstance(value, list) and value:
                            value = value[0]
                        if value:
                            plan["update"] = [{
                                "id": m["id"], "field": field,
                                "new_value": value,
                                "reason": "按该片段标注生成的 CRUD 演示操作",
                            }]
                            break
                    if plan["update"]:
                        break
            if not plan["highlight"]:
                plan["highlight"] = [moments[0]["id"]]
            if not plan["delete"]:
                delete_target = next(
                    (m["id"] for m in reversed(moments)
                     if m["id"] not in plan["highlight"]),
                    moments[-1]["id"],
                )
                plan["delete"] = [{
                    "id": delete_target,
                    "reason": "短期场景结束后清理该人物的演示记忆",
                }]
        else:
            moments = all_moments
            plan = all_plan

        # CRUD 演示只写入明确标注的演示备注，不把官方参考标签写回记忆。
        plan = dict(plan)
        plan["update"] = [
            {**case, "field": "notes", "new_value": "CRUD field update demo"}
            for case in plan.get("update", [])
        ]
        self.moments = moments
        self.plan = plan
        self.dataset_options = interactive_dataset_options()
        self.dataset_id = f"short_term|{self.dataset_key}|{self.participant or 'all'}"
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")
        self.participant_label = PARTICIPANT_NAMES.get(
            self.participant, "全部人物" if not self.participant else self.participant)
        steps = []

        # 1) add：每条视频片段一步（Qwen 感知 → 记忆）
        for i, m in enumerate(moments, 1):
            def run(m=m, i=i, force=False):
                mid, info = self._add_one(m, force=force)
                if mid is None:
                    msg = ("实时推理已停止" if info.get("cancelled") else "感知失败")
                    return f"{msg}：{info.get('error')}", {"kind": "single", **info}
                msg = f"Qwen 看过 {m['id']} → 写入记忆 {mid}（{info['source']}，{info['elapsed_ms']}ms）"
                return msg, {"kind": "single", **info}
            steps.append({
                "kind": "add",
                "title": f"感知并写入视频片段 {i}/{len(moments)}",
                "inputs": [self._display_brief(m)],
                "run": run,
            })

        # 相似事件候选只进入待审阅队列，不会自动合并。
        def run_merge_candidates(force=False):
            candidates = self.mem.find_merge_candidates(limit=100)
            result = {
                "kind": "operation_result", "operation": "merge_candidates",
                "title": "相似事件候选 · 逐条确认后才会合并",
                "candidates": candidates,
            }
            return f"发现 {len(candidates)} 组相似事件候选；没有自动合并", result
        steps.append({
            "kind": "merge_candidates",
            "title": "检查相似事件候选（需要逐条确认）",
            "detail": {"note": "候选依据：同源片段重叠≥80%，或120秒内Scene与Action合并文本相似度≥92%且共享人物/物品。画面/音频只供核对；扫描不改数据，逐条确认后才归并。"},
            "inputs": [],
            "run": run_merge_candidates,
        })

        # 2) query
        for q in plan.get("query", []):
            def run(q=q, force=False):
                hits = self.daytime.query(q["query"], top_k=3)
                items = [self._moment_detail(hit) for hit in hits]
                result = {
                    "kind": "operation_result", "operation": "query",
                    "title": "检索命中内容", "query": q["query"],
                    "count": len(hits), "items": items,
                }
                return f"检索「{q['query']}」命中 {len(hits)} 条", result
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
                    return f"未找到 {case['id']}，跳过", {
                        "kind": "operation_result", "operation": "update",
                        "title": "更新对象未找到", "items": [],
                    }
                before = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
                field = case["field"]
                old_value = copy.deepcopy(before.get(field))
                ok = self.mem.update(mid, {field: case["new_value"]})
                after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
                item = self._moment_detail(after)
                item["changes"] = [{
                    "field": field, "before": old_value,
                    "after": copy.deepcopy(after.get(field)),
                }]
                result = {
                    "kind": "operation_result", "operation": "update",
                    "title": "字段更新前后", "items": [item],
                }
                return f"更新 {case['id']}.{field}（{'成功' if ok else '字段未变化'}）", result
            steps.append({
                "kind": "update",
                "title": f"更新字段 {case['field']}",
                "detail": {"id": case["id"], "field": case["field"],
                           "new_value": case["new_value"]},
                "inputs": [],
                "run": run,
            })

        # 4) update_feedback：在独立交互演示库模拟纠错、确认与评分，不使用评测标签。
        update_ids = {case.get("id") for case in plan.get("update", [])}
        protected_ids = set(plan.get("highlight", [])) | {
            case.get("id") for case in plan.get("delete", [])
        }
        feedback_moment = next(
            (m for m in moments if m.get("id") not in update_ids | protected_ids),
            moments[0] if moments else None,
        )
        def run_feedback(force=False, m=feedback_moment):
            if not m:
                return "没有可反馈的记忆", None
            mid = self.id_map.get(m.get("id"))
            if not mid:
                return f"未找到 {m.get('id')}，反馈未保存", None
            corrections = {"notes": "CRUD feedback correction demo"}
            before = copy.deepcopy(
                self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            )
            ok = self.mem.update_feedback(
                mid, corrections=corrections, confirmed=True, user_rating=5,
            )
            after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            item = self._moment_detail(after)
            item["corrections"] = [
                {"field": field, "before": copy.deepcopy(before.get(field)), "after": value}
                for field, value in corrections.items()
            ]
            item["confirmed"] = after.get("feedback", {}).get("confirmed")
            item["user_rating"] = after.get("feedback", {}).get("user_rating")
            result_detail = {
                "kind": "operation_result", "operation": "update_feedback",
                "title": "模拟反馈修正内容", "simulated": True,
                "items": [item],
            }
            result = "已保存模拟纠错、确认和 5 星评分" if ok else "模拟反馈保存失败"
            return result, result_detail
        steps.append({
            "kind": "update_feedback",
            "title": "模拟反馈 · 纠错、确认与评分",
            "detail": {
                "id": feedback_moment.get("id", "") if feedback_moment else "",
                "note": "仅写入交互演示库的演示备注；不读取官方参考标签。确认和评分为模拟输入，不代表真实用户评价。",
            },
            "inputs": [],
            "run": run_feedback,
        })

        # 5) highlight
        hl_ids = plan.get("highlight", [])
        def run_highlight(force=False):
            items, n = [], 0
            moments = self.mem.memory.get("layer7", {}).get("moments", {})
            for sid in hl_ids:
                mid = self.id_map.get(sid)
                before = copy.deepcopy(moments.get(mid, {})) if mid else {}
                ok = bool(mid and self.mem.highlight(mid))
                after = moments.get(mid, before) if mid else before
                if ok:
                    n += 1
                item = self._moment_detail(after)
                item.update({"dataset_id": sid, "marked": ok,
                             "was_highlighted": bool(before.get("highlighted"))})
                items.append(item)
            result = {
                "kind": "operation_result", "operation": "highlight",
                "title": "已标记为重点的记忆", "items": items,
            }
            return f"标记重要记忆 {n}/{len(hl_ids)} 条", result
        steps.append({
            "kind": "highlight",
            "title": "标记重要记忆",
            "detail": {"ids": hl_ids},
            "inputs": [],
            "run": run_highlight,
        })

        # 6) delete（含 highlight 保护验证）
        del_cases = plan.get("delete", [])
        def run_delete(force=False):
            targets = []
            for case in del_cases:
                mid = self.id_map.get(case["id"])
                moment = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid) if mid else None
                targets.append((case, mid, copy.deepcopy(moment) if isinstance(moment, dict) else None))
            cleared = self.daytime.end_scene()
            msgs = [f"场景结束：清理 Layer1/2 {cleared} 条"]
            items, ok_n = [], 0
            for case, mid, before in targets:
                if not mid:
                    items.append({"dataset_id": case.get("id"), "deleted": False,
                                  "reason": case.get("reason", "manual"), "memory": {}})
                    continue
                ok = self.mem.delete(mid, reason=case.get("reason", "manual"))
                if ok:
                    ok_n += 1
                items.append({
                    **self._moment_detail(before or {}),
                    "dataset_id": case.get("id"),
                    "deleted": bool(ok),
                    "reason": case.get("reason", "manual"),
                })
            msgs.append(f"清理短期记忆：删除 {ok_n} 条")
            protection = []
            # 验证 highlight 保护，并把被验证的目标也明确展示出来。
            prot = [self.id_map[s] for s in hl_ids if self.id_map.get(s)]
            if prot:
                protected_moment = self.mem.memory.get("layer7", {}).get("moments", {}).get(prot[0], {})
                protected_detail = self._moment_detail(copy.deepcopy(protected_moment))
                blocked = not self.mem.delete(prot[0], reason="验证保护")
                forced = self.mem.delete(prot[0], reason="验证强制删除", force=True)
                protection.append({
                    **protected_detail, "default_delete_blocked": bool(blocked),
                    "force_deleted": bool(forced),
                })
                msgs.append(f"highlight 保护：默认删除被拒绝 {'✅' if blocked else '❌'}；"
                            f"force 强删 {'成功 ✅' if forced else '失败 ❌'}")
            result = {
                "kind": "operation_result", "operation": "delete",
                "title": "删除对象及保护验证", "items": items,
                "protection_checks": protection,
            }
            return "；".join(msgs), result
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
        self.dataset_key = "long_day"
        default_meta = json.load(open(LONG_META, encoding="utf-8"))
        participant = self.participant or default_meta.get("participant", "A1_JAKE")
        meta_path = LONG_PERSON_DIR / f"{participant}.meta.json"
        data_path = LONG_PERSON_DIR / f"{participant}.jsonl"
        if not meta_path.exists() or not data_path.exists():
            raise ValueError(f"没有 {participant} 的长期数据集")
        meta = json.load(open(meta_path, encoding="utf-8"))
        moments = _load_jsonl(data_path)
        self.participant = participant
        self.moments = moments
        self.meta = meta
        self.dataset_options = interactive_dataset_options()
        self.dataset_id = f"long_term|long_day|{participant}"
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")
        self.participant_label = PARTICIPANT_NAMES.get(participant, participant)
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
                cancelled = False
                for m in group:
                    mid, info = self._add_one(m, force=force)
                    if mid:
                        ok += 1
                    items.append(info)
                    if info.get("cancelled"):
                        cancelled = True
                        break
                msg = f"{h}:00 时段：Qwen 感知 {ok}/{len(group)} 条并写入记忆"
                if cancelled:
                    msg = f"实时推理已停止；本时段已完成 {ok}/{len(group)} 条"
                return msg, {
                    "kind": "batch", "items": items, "n": len(group),
                    "ok": ok, "cancelled": cancelled,
                }
            steps.append({
                "kind": "add_batch",
                "title": f"白天感知写入 · {h}:00 时段（{len(group)} 条）",
                "inputs": [self._display_brief(m) for m in group],
                "run": run,
            })

        # 相似事件候选只进入待审阅队列，不会自动合并。
        def run_merge_candidates(force=False):
            candidates = self.mem.find_merge_candidates(limit=100)
            result = {
                "kind": "operation_result", "operation": "merge_candidates",
                "title": "相似事件候选 · 逐条确认后才会合并",
                "candidates": candidates,
            }
            return f"发现 {len(candidates)} 组相似事件候选；没有自动合并", result
        steps.append({
            "kind": "merge_candidates",
            "title": "检查相似事件候选（需要逐条确认）",
            "detail": {"note": "候选依据：同源片段重叠≥80%，或120秒内Scene与Action合并文本相似度≥92%且共享人物/物品。画面/音频只供核对；扫描不改数据，逐条确认后才归并。"},
            "inputs": [],
            "run": run_merge_candidates,
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
            "title": "夜间整理 · sort 归类、combine 合并、compress 压缩、update_profile 画像更新",
            "detail": {"note": "按真实 Qwen 感知结果执行归类、同义项合并、L4/L5 摘要压缩和画像更新；画像更新同时重算置信度并标记过时属性。不读取 ground-truth。"},
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

        def fmt(x, with_stale=False):
            d = {"id": x.get("id", ""), "scene": x.get("scene", ""),
                 "user_action": x.get("user_action", ""),
                 "location": x.get("location", ""),
                 "activity": x.get("activity", ""),
                 "people": x.get("people", []),
                 "highlight": x.get("highlighted", False),
                 "timestamp": x.get("timestamp", "")}
            if with_stale:
                d["stale"] = x.get("stale", False)
                d["merged_into"] = x.get("merged_into", "")
                d["merged_sources"] = copy.deepcopy(x.get("merged_sources", []))
                d["merge_history"] = copy.deepcopy(x.get("merge_history", [])[-5:])
            v = vid_of(x.get("id", ""))
            if v:
                d["video"] = v
            return d

        layer7_moments = list(l7.get("moments", {}).values())
        merged_count = sum(bool(item.get("merged_into")) for item in layer7_moments)

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
            "layer7_active_total": len(layer7_moments) - merged_count,
            "layer7_merged_total": merged_count,
            "indices": indices,
            "metadata": m.get("metadata", {}),
            "daytime": self.daytime.snapshot(),
        }

    def _log_snapshot(self, title, kind, msg, perception=None):
        snap = rs.snapshot(self.mem, title, action=kind)
        snap["full"] = self._full_snapshot()
        snap["layer7_active_moments"] = snap["full"]["layer7_active_total"]
        snap["layer7_merged_moments"] = snap["full"]["layer7_merged_total"]
        self.log.append({"kind": kind, "title": title, "msg": msg,
                         "snapshot": snap, "perception": perception})

    def run_step(self, force=False, automatic=False, continuous_kind=None):
        """执行当前步；连续模式仅跨越相邻的同类步骤。"""
        self.auto_paused = False
        self.step_cancelled = False
        if self.pos >= len(self.steps):
            return "所有步骤已执行完毕"
        step = self.steps[self.pos]
        if automatic:
            expected_kind = continuous_kind or step["kind"]
            if step["kind"] != expected_kind:
                self.auto_paused = True
                return (f"连续运行已暂停：当前操作类型已从“{expected_kind}”切换为“{step['kind']}”。"
                        f"下一步“{step['title']}”未执行；可继续运行该类操作。")
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._active_step_cancel = cancel_event
            self._active_step_force = bool(force)
            if force and self._pending_cancel:
                cancel_event.set()
                self._pending_cancel = False
        self._active_step_no = self.pos + 1
        self._active_step_title = step["title"]
        try:
            msg, perception = step["run"](force=force)
        finally:
            self._active_step_no = None
            self._active_step_title = None
            with self._cancel_lock:
                if self._active_step_cancel is cancel_event:
                    self._active_step_cancel = None
                    self._active_step_force = False
        if cancel_event.is_set():
            self.step_cancelled = True
            if isinstance(perception, dict):
                perception["cancelled"] = True
            msg = "实时推理已停止；已完成的记忆会保留，当前步骤未标为完成，可再次运行。"
            self._log_snapshot(step["title"], step["kind"], msg, perception)
            return msg
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
        # 原始输出在前端折叠显示；这里保留完整文本，避免用户看到人为截断的字段。
        slim = []
        for e in self.log:
            e2 = dict(e)
            p = e.get("perception")
            if p:
                e2["perception"] = dict(p)
            slim.append(e2)

        meta = SCENARIO_META[self.key]
        extra = {
            "participant": self.participant,
            "participant_label": self.participant_label,
            "dataset_id": self.dataset_id,
            "dataset_key": self.dataset_key,
            "dataset_note": self.dataset_note,
            "dataset_options": self.dataset_options,
        }
        if self.key == "long_term":
            extra["hours_covered"] = self.meta.get("hours_covered", [])
        display_name = meta["name"]
        description = meta["desc"]
        if self.key == "short_term" and self.participant:
            display_name += f" · {self.participant_label}"
        elif self.key == "long_term":
            display_name += f" · {self.participant_label}"
            if not self.meta.get("complete", False):
                display_name += "（部分视频）"
            extra["available_count"] = self.meta.get("available_count", len(self.moments))
            extra["total_candidates"] = self.meta.get("total_candidates", len(self.moments))
            extra["coverage"] = self.meta.get("coverage", 1.0)
            extra["complete_day"] = self.meta.get("complete", False)
            description = self.dataset_note
        return {
            "scenario": self.key,
            "name": display_name,
            "desc": description,
            "total": total,
            "pos": self.pos,
            "finished": self.pos >= total,
            "current": cur,
            "auto_paused": self.auto_paused,
            "step_cancelled": self.step_cancelled,
            "log": slim,
            "operations": [dict(event) for event in self.operation_log[-2000:]],
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
