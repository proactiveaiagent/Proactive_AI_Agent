"""
viz_server.py — 7 层记忆模型可视化网站后端
======================================================================
读取 run_scenarios.py 产出的快照，提供 API 给前端展示：

    GET  /                                  前端页面
    GET  /api/scenarios                     场景列表（短期 / 长期）
    GET  /api/snapshot?scenario=short_term  某场景的时间轴快照序列
    GET  /api/memory?scenario=long_term     某场景的最终完整记忆状态

技术选型说明：用 Python 标准库 http.server，不引入 Flask/FastAPI 等依赖，
    "跑通 + 可视化"优先，后续需要再换框架。

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent
    .../envs/agent/bin/python scripts/memory/viz_server.py [port]
    然后浏览器打开 http://localhost:8765
"""

import hashlib
import math
import json
import re
import shutil
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request as URLRequest, urlopen

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "egolife" / "runs"
VIZ_DIR = Path(__file__).resolve().parent / "viz"
VIDEO_ROOT = (ROOT / "data" / "egolife").resolve()
PREVIEW_ROOT = RUNS / "viz_video_cache"

# 交互式步进执行会话（ThreadingHTTPServer 多线程，步骤执行需互斥：
# 实时推理可能耗时数分钟，期间 state / 视频流照常服务，重复 step 返回 409）
SESSION = None
STEP_LOCK = threading.Lock()
# Qwen 缓存预热任务状态
PREWARM = {"running": False, "done": 0, "total": 0, "cache_hits": 0, "inferred": 0, "failed": 0, "error": None, "scenario": None}
PREVIEW_JOBS = {}
PREVIEW_LOCK = threading.Lock()
PREVIEW_SEMAPHORE = threading.BoundedSemaphore(2)
ACTIVE_STEP_FORCE = False
ACTIVE_REASON = {"request_id": None, "event": None}
REASON_LOCK = threading.Lock()

try:
    from interactive_runner import InteractiveSession
    from qwen_perceive import QwenPerceiver
    import interactive_runner as ir
except ImportError:      # 兼容从其他 cwd 启动
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from interactive_runner import InteractiveSession
    from qwen_perceive import QwenPerceiver
    import interactive_runner as ir

SCENARIOS = {
    "short_term": {
        "name": "短期记忆场景",
        "desc": "不连续视频片段 · 验证 Layer1~7 完整 CRUD",
        "snapshots": "short_term_snapshots.json",
        "memory": "short_term_memory.json",
    },
    "long_term": {
        "name": "长期记忆场景",
        "desc": "完整一天 · 验证记忆流转与夜间整理",
        "snapshots": "long_term_snapshots.json",
        "memory": "long_term_memory.json",
    },
}


def load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _preview_clip(qs):
    try:
        start = float((qs.get("start") or [""])[0])
        end = float((qs.get("end") or [""])[0])
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return None, None
    return start, min(end, start + 180.0)


def _preview_cache_path(source, start=None, end=None):
    stat = source.stat()
    rel = source.relative_to(VIDEO_ROOT).as_posix()
    clip = "full" if start is None or end is None else f"{start:.3f}:{end:.3f}"
    key = hashlib.sha256(
        f"{rel}|{stat.st_size}|{stat.st_mtime_ns}|{clip}".encode("utf-8")
    ).hexdigest()
    return PREVIEW_ROOT / f"{key}.mp4", key


def _build_preview(source, target, key, start=None, end=None):
    temp = target.with_name(f"{target.stem}.{threading.get_ident()}.tmp.mp4")
    try:
        PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("服务器没有安装 ffmpeg")
        with PREVIEW_SEMAPHORE:
            with PREVIEW_LOCK:
                PREVIEW_JOBS[key] = {"state": "building"}
            cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
            if start is not None and end is not None:
                cmd.extend(["-ss", f"{start:.3f}"])
            cmd.extend(["-i", str(source)])
            if start is not None and end is not None:
                cmd.extend(["-t", f"{end - start:.3f}"])
            cmd.extend([
                "-vf", "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart", "-threads", "2", str(temp),
            ])
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=240, check=False,
            )
            if result.returncode:
                detail = (result.stderr or b"").decode("utf-8", "replace")[-500:]
                raise RuntimeError(detail or f"ffmpeg exited {result.returncode}")
        if not temp.is_file() or temp.stat().st_size == 0:
            raise RuntimeError("ffmpeg 未生成有效预览文件")
        temp.replace(target)
        with PREVIEW_LOCK:
            PREVIEW_JOBS[key] = {"state": "ready"}
    except Exception as exc:
        with PREVIEW_LOCK:
            PREVIEW_JOBS[key] = {"state": "error", "message": str(exc)[:300]}
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path, ctype):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------
    # 视频流（支持 Range，浏览器 <video> 拖动进度条必需）
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_video(rel):
        """校验相对路径在 data/egolife 下且为 mp4，返回绝对 Path 或 None。"""
        try:
            p = (VIDEO_ROOT / unquote(rel)).resolve()
        except Exception:
            return None
        if p.suffix.lower() != ".mp4" or VIDEO_ROOT not in p.parents:
            return None
        return p if p.exists() else None

    def _preview_status(self):
        qs = parse_qs(urlparse(self.path).query)
        source = self._safe_video((qs.get("p") or [""])[0])
        if not source or not source.is_file():
            return self._send(404, {"state": "error", "message": "video not found"})
        start, end = _preview_clip(qs)
        target, key = _preview_cache_path(source, start, end)
        if target.is_file():
            return self._send(200, {"state": "ready", "bytes": target.stat().st_size})
        if not shutil.which("ffmpeg"):
            return self._send(200, {"state": "error", "message": "服务器没有安装 ffmpeg"})
        with PREVIEW_LOCK:
            job = PREVIEW_JOBS.get(key)
            if job is None or job.get("state") == "error":
                PREVIEW_JOBS[key] = {"state": "queued"}
                threading.Thread(
                    target=_build_preview,
                    args=(source, target, key, start, end), daemon=True,
                ).start()
                job = PREVIEW_JOBS[key]
        return self._send(200, dict(job))

    def _stream_video(self, head_only=False):
        qs = parse_qs(urlparse(self.path).query)
        p = self._safe_video((qs.get("p") or [""])[0])
        if not p:
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"error": "video not found"}')
            return
        quality = (qs.get("quality") or ["original"])[0]
        if quality == "preview":
            preview_start, preview_end = _preview_clip(qs)
            preview, _ = _preview_cache_path(p, preview_start, preview_end)
            if not preview.is_file():
                return self._send(425, {"error": "preview is not ready"})
            p = preview
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                start = int(m.group(1) or 0)
                end = int(m.group(2)) if m.group(2) else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=86400")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with open(p, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_HEAD(self):
        if urlparse(self.path).path == "/api/video":
            return self._stream_video(head_only=True)
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        global SESSION, ACTIVE_STEP_FORCE
        url = urlparse(self.path)
        qs = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            idx = VIZ_DIR / "index.html"
            if idx.exists():
                return self._send_file(idx, "text/html; charset=utf-8")
            return self._send(404, {"error": "index.html not found"})

        if url.path == "/api/memory_asset":
            if SESSION is None:
                return self._send(404, {"error": "no interactive session"})
            rel = (qs.get("p") or [""])[0]
            root = SESSION.mem.memory_dir.resolve()
            asset = (root / rel).resolve()
            if not rel.startswith("assets/") or not asset.is_relative_to(root / "assets"):
                return self._send(400, {"error": "invalid asset path"})
            if not asset.is_file():
                return self._send(404, {"error": "memory asset missing"})
            ctype = {".webp": "image/webp", ".ogg": "audio/ogg"}.get(asset.suffix)
            if not ctype:
                return self._send(400, {"error": "invalid asset type"})
            return self._send_file(asset, ctype)

        if url.path == "/api/interactive/progress":
            try:
                after = float((qs.get("after") or ["0"])[0])
            except (TypeError, ValueError):
                after = 0
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            last = None
            while True:
                nightly = SESSION.nightly_progress
                step_progress = getattr(SESSION, "step_progress", None)
                progress = (dict(nightly) if nightly and nightly.get("running")
                            else dict(step_progress) if step_progress else
                            dict(nightly) if nightly else None)
                fresh = progress and progress.get("updated_at", 0) >= after
                payload = json.dumps(progress, ensure_ascii=False)
                if fresh and payload != last:
                    try:
                        self.wfile.write(f"event: progress\ndata: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                    last = payload
                if fresh and not progress.get("running"):
                    return
                threading.Event().wait(0.5)

        if url.path == "/api/video_preview/status":
            return self._preview_status()

        if url.path == "/api/video":
            return self._stream_video()

        if url.path == "/api/evaluation-data":
            evaluation = load_json(RUNS / "evaluation_dashboard_data_20260928.json")
            if evaluation is None:
                return self._send(503, {"error": "evaluation data is not ready"})
            return self._send(200, evaluation)

        if url.path == "/api/multimodal-capacity":
            temporal_long = load_json(RUNS / "multimodal_full_day_v8_20261007" / "capacity_report.json")
            if temporal_long is not None:
                temporal_long["retrieval_latency"] = load_json(
                    RUNS / "multimodal_decision_review_v40_20261007" / "retrieval_latency_report.json")
            return self._send(200, {
                "short_120": load_json(RUNS / "multimodal_baseline_v5_20260929" / "capacity_report.json"),
                "long_725": load_json(RUNS / "multimodal_full_day_v5_20260929" / "capacity_report.json"),
                "temporal_sample": load_json(RUNS / "multimodal_aligned_v8_20261007" / "capacity_report.json"),
                "temporal_long": temporal_long,
            })

        if url.path == "/api/decision-benchmark":
            benchmark = load_json(ROOT / "data/egolife/testset/multimodal_decision_benchmark_v1.json")
            if benchmark is None:
                return self._send(503, {"error": "decision benchmark is not ready"})
            return self._send(200, benchmark)

        if url.path == "/api/semantic-scenario-review":
            report_path = RUNS / "multimodal_decision_benchmark_v3_20261008" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_benchmark_v2_20261008" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_benchmark_v1_20261008" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v50_20261007" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v45_20261007" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v41_20261007" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v39_20261007" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v36_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v35_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v34_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v33_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v32_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v31_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v30_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v29_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "multimodal_decision_review_v28_20260930" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v27" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v26" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v22" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v21" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v20" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v19" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v18" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v17" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v13" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v12" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v11" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v9" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260930_v7" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260929_v6" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260929_v5" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260929_v4" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260929_v3" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260929_v2" / "scenario_review.json"
            if not report_path.exists():
                report_path = RUNS / "semantic_scenario_review_20260929" / "scenario_review.json"
            report = load_json(report_path)
            if report is None:
                return self._send(404, {"error": "场景审阅仍在生成"})
            return self._send(200, report)

        if url.path == "/api/scenarios":
            out = []
            for key, cfg in SCENARIOS.items():
                snap = load_json(RUNS / cfg["snapshots"])
                out.append({
                    "key": key,
                    "name": cfg["name"],
                    "desc": cfg["desc"],
                    "moments_total": (snap or {}).get("moments_total", 0),
                    "snapshot_count": len((snap or {}).get("snapshots", [])),
                    "available": snap is not None,
                })
            return self._send(200, out)

        if url.path == "/api/snapshot":
            key = (qs.get("scenario") or ["short_term"])[0]
            cfg = SCENARIOS.get(key)
            if not cfg:
                return self._send(404, {"error": f"unknown scenario: {key}"})
            snap = load_json(RUNS / cfg["snapshots"])
            if snap is None:
                return self._send(404, {"error": "请先运行 run_scenarios.py 生成快照"})
            return self._send(200, snap)

        if url.path == "/api/memory":
            key = (qs.get("scenario") or ["long_term"])[0]
            cfg = SCENARIOS.get(key)
            if not cfg:
                return self._send(404, {"error": f"unknown scenario: {key}"})
            mem = load_json(RUNS / cfg["memory"])
            if mem is None:
                return self._send(404, {"error": "memory state not found"})
            from memory import entity_graph_summary
            # 按六维高频索引挑少量正文样例，避免把全部归档重复发给浏览器。
            l7 = mem.get("layer7", {})
            moments = l7.get("moments", {})
            dimensions = ("people", "locations", "time_nodes",
                          "activity_events", "environments", "objects")
            indices = {k: l7.get(k, {}) for k in dimensions}
            archive_samples = {}
            for dimension in dimensions:
                ranked = sorted(indices[dimension].items(),
                                key=lambda item: (-len(item[1]), str(item[0])))[:2]
                groups = []
                for tag, ids in ranked:
                    examples = []
                    for mid in ids[:1]:
                        moment = moments.get(mid)
                        if moment:
                            examples.append({
                                key: moment.get(key)
                                for key in ("id", "timestamp", "scene", "user_action",
                                            "people", "location", "activity",
                                            "environments", "objects", "notes",
                                            "source_provenance",
                                            "highlighted", "highlight_source",
                                            "highlighted_at", "merged_into",
                                            "merged_sources", "merge_history")
                                if moment.get(key) is not None
                            })
                    groups.append({"tag": tag, "count": len(ids), "moments": examples})
                archive_samples[dimension] = groups
            return self._send(200, {
                "layer1": mem.get("layer1", []),
                "layer2": mem.get("layer2", []),
                "layer3": mem.get("layer3", []),
                "layer4": mem.get("layer4", {}),
                "layer5": mem.get("layer5", {}),
                "layer6": mem.get("layer6", {}),
                "indices": indices,
                "archive_samples": archive_samples,
                "entity_relationships": entity_graph_summary(l7, limit=12),
                "moments_count": len(moments),
                "active_moments_count": sum(not item.get("merged_into") for item in moments.values()),
                "merged_moments_count": sum(bool(item.get("merged_into")) for item in moments.values()),
                "metadata": mem.get("metadata", {}),
            })

        # ---------------- 交互式步进执行 ----------------
        if url.path == "/api/interactive/options":
            return self._send(200, ir.interactive_dataset_options())

        if url.path == "/api/interactive/reason_cancel":
            with REASON_LOCK:
                request_id = ACTIVE_REASON.get("request_id")
                cancel_event = ACTIVE_REASON.get("event")
                if not request_id or cancel_event is None:
                    return self._send(409, {"error": "当前没有正在运行的语义建议"})
                cancel_event.set()
            try:
                payload = json.dumps({"request_id": request_id}).encode("utf-8")
                req = URLRequest(
                    "http://127.0.0.1:8000/cancel_inference",
                    data=payload, headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=3):
                    pass
            except Exception:
                # The in-process event still prevents later reasoning rounds.
                pass
            return self._send(200, {"cancel_requested": True})

        if url.path == "/api/interactive/reason":
            if SESSION is None:
                return self._send(404, {"error": "请先创建交互会话"})
            mode = (qs.get("mode") or [""])[0]
            situation = (qs.get("situation") or [""])[0].strip()[:1200]
            reason_context = {}
            if mode == "latest":
                archive = SESSION.mem.memory.get("layer7", {}).get("moments", {})
                active = [m for m in archive.values() if not m.get("merged_into")]
                if not active:
                    return self._send(409, {"error": "当前会话还没有写入记忆，请先运行一个视频步骤"})
                latest = max(active, key=lambda m: str(m.get("timestamp") or ""))
                reason_context = {"source_moment_id": latest.get("id"), "proactive": True}
            elif not situation:
                return self._send(400, {"error": "请填写当前事件"})
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "当前步骤正在运行，请完成后再分析"})
            request_id = uuid.uuid4().hex
            cancel_event = threading.Event()
            reason_context.update({
                "reasoning_request_id": request_id,
                "cancellation_event": cancel_event,
            })
            with REASON_LOCK:
                ACTIVE_REASON["request_id"] = request_id
                ACTIVE_REASON["event"] = cancel_event
            try:
                result = SESSION.daytime.advise(situation, context=reason_context)
                log_path = SESSION.mem.memory_dir / "reasoning_log.jsonl"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                return self._send(200, result)
            except Exception as exc:
                if cancel_event.is_set() or exc.__class__.__name__ == "MemoryReasoningCancelled":
                    return self._send(409, {"error": "主动建议已停止", "cancelled": True})
                return self._send(503, {"error": f"语义推理未完成：{exc}"})
            finally:
                with REASON_LOCK:
                    if ACTIVE_REASON.get("request_id") == request_id:
                        ACTIVE_REASON["request_id"] = None
                        ACTIVE_REASON["event"] = None
                STEP_LOCK.release()

        if url.path == "/api/interactive/new":
            key = (qs.get("scenario") or ["short_term"])[0]
            participant = (qs.get("participant") or [""])[0] or None
            if participant == "all":
                participant = None
            dataset = (qs.get("dataset") or ["base"])[0]
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "busy",
                                        "message": "有步骤正在执行（实时推理可能需要数分钟），请等它完成后再新建会话"})
            try:
                try:
                    SESSION = InteractiveSession(
                        key, participant=participant, dataset=dataset
                    )
                except Exception as e:
                    return self._send(500, {"error": f"创建会话失败: {e}"})
                return self._send(200, SESSION.state())
            finally:
                STEP_LOCK.release()

        if url.path == "/api/interactive/state":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            return self._send(200, SESSION.state())

        if url.path == "/api/interactive/cancel":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            if not STEP_LOCK.locked() or not ACTIVE_STEP_FORCE:
                return self._send(409, {
                    "error": "没有正在运行的实时推理可停止",
                    "message": "只有实时推理运行时才能使用停止按钮",
                })
            result = SESSION.cancel_current()
            if not result.get("requested"):
                return self._send(409, {
                    "error": "当前实时推理已结束或尚未开始",
                    "message": "请在实时推理进行中点击停止",
                })
            return self._send(200, result)

        if url.path == "/api/interactive/step":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            force = (qs.get("force") or ["0"])[0] in ("1", "true", "True")
            automatic = (qs.get("automatic") or ["0"])[0] in ("1", "true", "True")
            continuous_kind = (qs.get("continuous_kind") or [None])[0]
            use_cache = (qs.get("use_cache") or ["1"])[0] not in ("0", "false", "False")
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "busy",
                                        "message": "上一步仍在执行（实时推理可能需要数分钟），请稍候"})
            ACTIVE_STEP_FORCE = force
            try:
                try:
                    msg = SESSION.run_step(force=force, automatic=automatic,
                                            continuous_kind=continuous_kind,
                                            use_cache=use_cache)
                except Exception as e:
                    return self._send(500, {"error": f"执行失败: {e}"})
                st = SESSION.state()
                st["last_result"] = msg
                st["last_force"] = force
                st["auto_paused"] = SESSION.auto_paused
                return self._send(200, st)
            finally:
                ACTIVE_STEP_FORCE = False
                STEP_LOCK.release()

        # Qwen 缓存预热（后台跑，前端轮询进度）
        if url.path == "/api/interactive/prewarm":
            key = (qs.get("scenario") or ["short_term"])[0]
            if PREWARM["running"]:
                return self._send(200, {"started": False, **PREWARM})

            def work():
                PREWARM.update(running=True, done=0, total=0, cache_hits=0, inferred=0, failed=0, error=None, scenario=key)
                try:
                    dataset = (qs.get("dataset") or ["base"])[0]
                    participant = (qs.get("participant") or [""])[0] or None
                    dataset_id = (
                        f"{key}|{dataset}|{participant or 'all'}"
                    )
                    same_session = SESSION is not None and SESSION.dataset_id == dataset_id
                    if same_session:
                        ms = SESSION.moments
                        p = SESSION.perceiver
                    else:
                        if key == "short_term":
                            src = (ir.SHORT_EXPANDED_FILE if dataset == "expanded"
                                   else ir.SHORT_FILE)
                        else:
                            participant = participant or "A1_JAKE"
                            src = ir.LONG_PERSON_DIR / f"{participant}.jsonl"
                        ms = ir._load_jsonl(src)
                        if key == "long_term":
                            pass  # participant datasets contain only local English clips
                        elif participant:
                            ms = [m for m in ms
                                  if "_".join(str(m.get("id", "")).split("_")[1:3]) == participant]
                        p = ir.site_perceiver()
                    PREWARM["total"] = len(ms)
                    for m in ms:
                        try:
                            result = p.perceive(m)
                            if result.get("fields"):
                                if result.get("source") == "cache":
                                    PREWARM["cache_hits"] += 1
                                elif result.get("source") == "qwen":
                                    PREWARM["inferred"] += 1
                                else:
                                    PREWARM["failed"] += 1
                            else:
                                PREWARM["failed"] += 1
                        except Exception:
                            PREWARM["failed"] += 1
                        finally:
                            PREWARM["done"] += 1
                except Exception as e:
                    PREWARM["error"] = str(e)
                finally:
                    PREWARM["running"] = False

            threading.Thread(target=work, daemon=True).start()
            return self._send(200, {"started": True})

        if url.path == "/api/interactive/prewarm_status":
            return self._send(200, dict(PREWARM))

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        global SESSION
        url = urlparse(self.path)
        if url.path not in ("/api/interactive/merge", "/api/interactive/unmerge"):
            return self._send(404, {"error": "not found"})
        if SESSION is None:
            return self._send(404, {"error": "请先创建交互会话"})
        if not STEP_LOCK.acquire(blocking=False):
            return self._send(409, {"error": "当前步骤正在运行，请稍后再执行归并操作"})
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32768:
                    return self._send(400, {"error": "请求内容长度无效"})
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                return self._send(400, {"error": "请求内容必须是有效 JSON"})
            source_id = str(payload.get("source_id") or "")[:160]
            target_id = str(payload.get("target_id") or "")[:160]
            if url.path.endswith("/merge"):
                reason = str(payload.get("reason") or "网页人工确认重复事件")[:240]
                result = SESSION.mem.merge_moments(
                    source_id, target_id,
                    confirmed_duplicate=payload.get("confirmed_duplicate") is True,
                    reason=reason)
                operation, label = "merge_moments", "用户确认合并相似事件"
                message = (
                    f"已合并 {source_id} → {target_id}；原始来源与多模态资产保留"
                    if result.get("ok") and result.get("status") == "merged"
                    else result.get("message") or
                         ("两条记录此前已合并" if result.get("status") == "already_merged"
                          else "合并未执行")
                )
            else:
                reason = str(payload.get("reason") or "网页撤销最近一次事件归并")[:240]
                result = SESSION.mem.unmerge_moments(source_id, target_id, reason=reason)
                operation, label = "unmerge_moments", "用户撤销事件归并"
                message = (
                    f"已撤销 {source_id} → {target_id}；来源记忆和多模态资产已恢复"
                    if result.get("ok") and result.get("status") == "unmerged"
                    else result.get("message") or "撤销未执行"
                )
            archive = SESSION.mem.memory.get("layer7", {}).get("moments", {})
            source = SESSION._moment_detail(archive.get(source_id, {}))
            target = SESSION._moment_detail(archive.get(target_id, {}))
            candidates = SESSION.mem.find_merge_candidates(limit=20)
            perception = {
                "kind": "operation_result", "operation": operation,
                "title": "事件合并记录" if operation == "merge_moments" else "撤销归并记录",
                "merge": result, "source": source, "target": target,
                "candidates": candidates,
            }
            SESSION._log_snapshot(label, operation, message, perception)
            state = SESSION.state()
            state["last_result"] = message
            return self._send(200, state)
        finally:
            STEP_LOCK.release()

    def log_message(self, fmt, *args):
        # 静默常规请求日志，只打印错误
        if args and str(args[0]).startswith(("GET /api", "GET / ")):
            return
        super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("=" * 60)
    print("7 层记忆模型可视化网站")
    print("=" * 60)
    print(f"  快照目录: {RUNS}")
    for key, cfg in SCENARIOS.items():
        ok = (RUNS / cfg["snapshots"]).exists()
        print(f"  {cfg['name']:<8} {'✅ 就绪' if ok else '⚠️ 未生成'}")
    print(f"\n  服务地址: http://localhost:{port}")
    print("  页面内可切换「交互执行」模式：逐步查看输入的视频/时间点，手工点击运行")
    print("  按 Ctrl+C 停止\n")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
