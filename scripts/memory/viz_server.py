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
import copy
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request as URLRequest, urlopen

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "egolife" / "runs"
SESSION_RESUME_CHECKPOINT = RUNS / "interactive_mem" / "resume_after_restart.json"
SESSION_INTEGRITY_VERSION = "video-only-perception-v1"
VIZ_DIR = Path(__file__).resolve().parent / "viz"
VIDEO_ROOT = (ROOT / "data" / "egolife").resolve()
FEISHU_VIDEO_ROOT = (ROOT / "test_data" / "feishu_ar_agent_memory_retrieval_20261013").resolve()
PREVIEW_ROOT = RUNS / "viz_video_cache"

# 交互式步进执行会话（ThreadingHTTPServer 多线程，步骤执行需互斥：
# 实时推理可能耗时数分钟，期间 state / 视频流照常服务，重复 step 返回 409）
SESSION = None
RESTORED_SESSION_ACTIVE = False
STEP_LOCK = threading.Lock()
ENTITY_READ_LOCK = threading.Lock()  # Queue concurrent read-only entity queries; writers still use STEP_LOCK.
ENTITY_ARCHIVE_INDEX_CACHE = {}  # scenario -> (path signature, EntityRecordIndex)
# Qwen 缓存预热任务状态
PREWARM = {"running": False, "done": 0, "total": 0, "cache_hits": 0, "inferred": 0,
           "failed": 0, "failed_items": [], "current_id": None,
           "current_status": None, "error": None, "scenario": None}
PREVIEW_JOBS = {}
PREVIEW_LOCK = threading.Lock()
MEMORY_ASSET_CACHE = {}
MEMORY_ASSET_CACHE_LOCK = threading.Lock()
PREVIEW_SEMAPHORE = threading.BoundedSemaphore(2)


def _with_memory_asset_urls(value):
    """Add browser URLs to nested image assets returned from saved memory records."""
    if isinstance(value, list):
        return [_with_memory_asset_urls(item) for item in value]
    if not isinstance(value, dict):
        return value
    enriched = {key: _with_memory_asset_urls(item) for key, item in value.items()}
    asset = str(value.get("asset") or "")
    source_asset = str(value.get("source_asset") or "")
    if asset.startswith("assets/"):
        enriched["url"] = "/api/memory_asset?p=" + quote(asset)
    if source_asset.startswith("assets/"):
        enriched["source_url"] = "/api/memory_asset?p=" + quote(source_asset)

    # Entity summaries and visual-grounding results are stored separately.
    # Join them here so the detail UI does not hide valid crops just because
    # entity_visuals only contains the semantic card text.
    entities = enriched.get("entity_visuals")
    person_group = enriched.get("person_visuals") or {}
    object_group = enriched.get("object_visuals") or {}
    person_rows = person_group.get("items") or [] if isinstance(person_group, dict) else []
    object_rows = object_group.get("items") or [] if isinstance(object_group, dict) else []
    def allowed_roles(entity_type):
        return {"person": {"face_crop", "person_crop"},
                "object": {"object_crop"}}.get(entity_type, set())

    # Do not attach arbitrary source frames to entity cards. A card gets only
    # its own grounded crop; the crop's source_asset remains available for toggle.
    for entity_type, rows in (("person", person_rows), ("object", object_rows)):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["frame_refs"] = [ref for ref in row.get("frame_refs") or []
                                 if isinstance(ref, dict)
                                 and ref.get("visual_role") in allowed_roles(entity_type)]

    if isinstance(entities, dict) and isinstance(entities.get("items"), list):
        entity_assets = entities.get("assets") or []
        for item in entities["items"]:
            if not isinstance(item, dict) or item.get("frame_refs"):
                continue
            refs = []
            for raw_index in item.get("asset_indices") or []:
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(entity_assets):
                    ref = entity_assets[index]
                    if (isinstance(ref, dict) and ref.get("asset")
                            and ref.get("visual_role") in allowed_roles(item.get("entity_type"))):
                        refs.append(ref)
            if refs:
                item["frame_refs"] = refs
        def label_key(text):
            return " ".join(str(text or "").casefold().split())
        people_by_name = {label_key(row.get("person")): row for row in person_rows
                          if isinstance(row, dict) and label_key(row.get("person"))}
        objects_by_name = {label_key(row.get("object")): row for row in object_rows
                           if isinstance(row, dict) and label_key(row.get("object"))}
        for item in entities["items"]:
            if not isinstance(item, dict):
                continue
            entity_type = str(item.get("entity_type") or "").casefold()
            name = label_key(item.get("entity"))
            grounding = (people_by_name.get(name) if entity_type == "person"
                         else objects_by_name.get(name) if entity_type == "object" else None)
            if grounding:
                if not item.get("frame_refs") and grounding.get("frame_refs"):
                    item["frame_refs"] = grounding["frame_refs"]
                if grounding.get("status"):
                    item["status"] = grounding["status"]
                if not item.get("description") and grounding.get("appearance"):
                    item["description"] = grounding.get("appearance")
                if grounding.get("evidence_fallback_reason"):
                    item["evidence_fallback_reason"] = grounding["evidence_fallback_reason"]
            elif entity_type in {"person", "object"} and not item.get("frame_refs"):
                reason = ("人物未与定位结果匹配，已保留完整来源画面供核对"
                          if entity_type == "person" else
                          "物品未与定位结果匹配，已保留完整来源画面供核对")
                item["frame_refs"] = fallback_refs(source_frames, reason)
                if item["frame_refs"]:
                    item["evidence_fallback_reason"] = reason
    if isinstance(entities, dict) and isinstance(entities.get("items"), list):
        # Older records may contain scene/context cards. Keep only person/object
        # evidence in the multimodal entity gallery; their text remains in memory.
        kept_items, kept_assets, remap = [], [], {}
        for item in entities["items"]:
            if not isinstance(item, dict) or item.get("entity_type") not in {"person", "object"}:
                continue
            new_indices = []
            for raw_index in item.get("asset_indices") or []:
                try:
                    old_index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if old_index < 0 or old_index >= len(entity_assets):
                    continue
                asset_row = entity_assets[old_index]
                if (not isinstance(asset_row, dict)
                        or asset_row.get("visual_role") not in allowed_roles(item.get("entity_type"))):
                    continue
                if old_index not in remap:
                    remap[old_index] = len(kept_assets)
                    kept_assets.append(entity_assets[old_index])
                new_indices.append(remap[old_index])
            item["asset_indices"] = new_indices
            kept_items.append(item)
        entities["items"] = kept_items
        entities["assets"] = kept_assets
    return enriched
ACTIVE_STEP_FORCE = False
ACTIVE_STEP_RUNNING = False
ACTIVE_STEP_PHASE = None
ACTIVE_REASON = {"request_id": None, "event": None}
REASON_LOCK = threading.Lock()

try:
    from interactive_runner import InteractiveSession
    from qwen_perceive import QwenPerceiver
    from prewarm_status import multimodal_prewarm_status
    import interactive_runner as ir
except ImportError:      # 兼容从其他 cwd 启动
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from interactive_runner import InteractiveSession
    from qwen_perceive import QwenPerceiver
    from prewarm_status import multimodal_prewarm_status
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


def save_session_checkpoint(state):
    """Persist only idle states; atomic replacement prevents a partial resume file."""
    if (state.get('step_progress') or {}).get('running'):
        return
    SESSION_RESUME_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = SESSION_RESUME_CHECKPOINT.with_suffix('.tmp')
    payload = dict(state)
    payload["integrity_version"] = SESSION_INTEGRITY_VERSION
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, SESSION_RESUME_CHECKPOINT)


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
    source = source.resolve()
    rel = None
    for corpus, root in (("egolife", VIDEO_ROOT), ("feishu", FEISHU_VIDEO_ROOT)):
        try:
            rel = f"{corpus}/{source.relative_to(root).as_posix()}"
            break
        except ValueError:
            continue
    if rel is None:
        raise ValueError("preview source is outside the configured video corpora")
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


def _compact_interactive_state(state):
    """Keep interactive responses small while retaining current memory and video navigation."""
    compact = dict(state)
    rows = state.get("log") or []
    compact_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        snapshot = item.pop("snapshot", None)
        if item.get("kind") == "add" and SESSION is not None:
            perception = item.get("perception")
            if isinstance(perception, dict):
                moment_id = str(perception.get("moment_id") or "")
                stored = ((SESSION.mem.memory.get("layer7") or {}).get("moments") or {}).get(moment_id)
                if isinstance(stored, dict) and isinstance(stored.get("multimodal"), dict):
                    perception["multimodal"] = copy.deepcopy(stored["multimodal"])
        if item.get("kind") == "add" and isinstance(snapshot, dict):
            full = snapshot.get("full") or {}
            moments = full.get("layer7_moments") or []
            perception = item.get("perception") or {}
            moment_id = perception.get("moment_id")
            moment = next((entry for entry in moments
                           if isinstance(entry, dict) and entry.get("id") == moment_id),
                          moments[-1] if moments else {})
            video = (moment or {}).get("video") or {}
            source = (moment or {}).get("source_provenance") or {}
            if video.get("video_url"):
                title_match = re.search(r"(\d+)\s*/\s*\d+", str(item.get("title") or ""))
                item["video_nav"] = {
                    "pos": max(0, int(title_match.group(1)) - 1) if title_match else len(compact_rows),
                    "moment_id": (moment or {}).get("id"),
                    "segment_id": source.get("source_segment_id") or perception.get("dataset_id") or (moment or {}).get("id"),
                    "video_url": video.get("video_url"),
                    "video": video.get("video") or source.get("source_video_id") or "",
                    "clock": video.get("clock") or "",
                    "clip_start": video.get("clip_start") if video.get("clip_start") is not None else source.get("clip_start_s", 0),
                    "clip_end": video.get("clip_end") if video.get("clip_end") is not None else source.get("clip_end_s"),
                    "people": (moment or {}).get("people") or [],
                    "location": (moment or {}).get("location") or "",
                    "scene": (moment or {}).get("scene") or "",
                    "action": (moment or {}).get("user_action") or (moment or {}).get("activity") or "",
                }
        compact_rows.append(item)
    compact["log"] = compact_rows
    compact["snapshots_compact"] = True
    # Step cards and memory details must resolve the same saved crop/source assets.
    return _with_memory_asset_urls(compact)


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
        """Allow videos only under the EgoLife corpus or copied Feishu test corpus."""
        decoded = unquote(rel)
        for allowed_root in (VIDEO_ROOT, FEISHU_VIDEO_ROOT):
            try:
                p = (allowed_root / decoded).resolve()
            except Exception:
                continue
            if p.suffix.lower() == ".mp4" and allowed_root in p.parents and p.is_file():
                return p
        return None

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
        global SESSION, ACTIVE_STEP_FORCE, ACTIVE_STEP_RUNNING, ACTIVE_STEP_PHASE, RESTORED_SESSION_ACTIVE
        url = urlparse(self.path)
        qs = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            idx = VIZ_DIR / "index.html"
            if idx.exists():
                return self._send_file(idx, "text/html; charset=utf-8")
            return self._send(404, {"error": "index.html not found"})

        if url.path == "/api/memory_asset":
            rel = unquote((qs.get("p") or [""])[0])
            if not rel.startswith("assets/"):
                return self._send(400, {"error": "invalid asset path"})
            asset = None
            roots = []
            if SESSION is not None:
                roots.append(Path(SESSION.mem.memory_dir))
            memory_root = RUNS / "interactive_mem"
            roots.extend(memory_root / name for name in ("current", "short_term", "long_term", "custom_test"))
            # Tabs can retain media URLs after a session is archived. Search only
            # the application's own archived memory folders so those thumbnails
            # remain available after switching scenarios or restarting a session.
            trash_root = memory_root / "_trash"
            try:
                roots.extend(sorted((p for p in trash_root.iterdir() if p.is_dir()), reverse=True))
            except OSError:
                pass
            cache = globals().get("MEMORY_ASSET_CACHE", {})
            cached = cache.get(rel)
            if cached is not None and Path(cached).is_file():
                asset = Path(cached)
            if asset is None:
                for root in roots:
                    try:
                        root = root.resolve()
                        candidate = (root / rel).resolve()
                        if candidate.is_relative_to(root / "assets") and candidate.is_file():
                            asset = candidate
                            break
                    except (OSError, RuntimeError):
                        continue
            if asset is None:
                return self._send(404, {"error": "memory asset missing"})
            ctype = {".webp": "image/webp", ".jpg": "image/jpeg",
                     ".jpeg": "image/jpeg", ".png": "image/png"}.get(asset.suffix.lower())
            if not ctype:
                return self._send(400, {"error": "invalid asset type"})
            cache_lock = globals().get("MEMORY_ASSET_CACHE_LOCK")
            if cache_lock is not None:
                with cache_lock:
                    cache[rel] = asset
            else:
                cache[rel] = asset
            return self._send_file(asset, ctype)

        if url.path == "/api/interactive/progress":
            try:
                after = float((qs.get("after") or ["0"])[0])
            except (TypeError, ValueError):
                after = 0
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            requested_run_id = (qs.get("run_id") or [""])[0]
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
                if progress and nightly and nightly.get("running") and step_progress:
                    progress.setdefault("run_id", step_progress.get("run_id"))
                fresh = (progress and progress.get("run_id") == requested_run_id
                         if requested_run_id else
                         progress and progress.get("updated_at", 0) >= after)
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

        if url.path == "/api/object-grounding":
            run_dir = RUNS / "multimodal_full_day_v9_object_grounded_20261011"
            report = load_json(run_dir / "object_grounding_report.json")
            progress_path = run_dir / "object_grounding_progress.jsonl"
            processed = checked = linked = not_verified = 0
            if progress_path.is_file():
                with progress_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        grounding = row.get("object_visuals") or {}
                        processed += 1
                        items = grounding.get("items") or []
                        checked += len(items)
                        for item in items:
                            if item.get("status") in {"grounded", "visible_unlabeled"} and item.get("frame_refs"):
                                linked += 1
                            elif item.get("status") == "not_verified":
                                not_verified += 1
            expected = int((report or {}).get("moments_total") or 725)
            capacity = load_json(run_dir / "capacity_report.json")
            if report is not None and capacity is not None and processed >= expected:
                return self._send(200, {
                    "complete": True, "report": report, "capacity": capacity,
                })
            return self._send(200, {
                "complete": False, "processed_moments": processed,
                "total_moments": 725, "checked_objects": checked,
                "total_objects": 2871, "reported_linked": linked,
                "not_verified": not_verified,
            })

        if url.path == "/api/archive_asset":
            allowed_runs = {
                "multimodal_full_day_v9_object_grounded_20261011",
                "multimodal_full_day_v10_object_crops_20261012",
            }
            run_name = (qs.get("run") or ["multimodal_full_day_v9_object_grounded_20261011"])[0]
            if run_name not in allowed_runs:
                return self._send(400, {"error": "invalid archive run"})
            run_root = (RUNS / run_name).resolve()
            rel = (qs.get("p") or [""])[0]
            asset = (run_root / rel).resolve()
            if (not rel.startswith("assets/") or
                    not asset.is_relative_to(run_root / "assets") or not asset.is_file()):
                return self._send(404, {"error": "archive asset not found"})
            ctype = {".webp": "image/webp"}.get(asset.suffix.lower())
            if not ctype:
                return self._send(400, {"error": "invalid asset type"})
            return self._send_file(asset, ctype)

        if url.path == "/api/multimodal-capacity":
            temporal_long = load_json(RUNS / "multimodal_full_day_v8_20261007" / "capacity_report.json")
            grounding_dir = RUNS / "multimodal_full_day_v9_object_grounded_20261011"
            grounding_report = load_json(grounding_dir / "object_grounding_report.json")
            progress_path = grounding_dir / "object_grounding_progress.jsonl"
            progress_count = sum(1 for line in progress_path.open(encoding="utf-8")) if progress_path.is_file() else 0
            if (grounding_report is not None and
                    progress_count >= int(grounding_report.get("moments_total") or 725)):
                temporal_long = load_json(grounding_dir / "capacity_report.json") or temporal_long
            sample_dir = RUNS / "multimodal_aligned_v8_20261007"
            return self._send(200, {
                "short_120": load_json(RUNS / "multimodal_baseline_v5_20260929" / "capacity_report.json"),
                "long_725": load_json(RUNS / "multimodal_full_day_v5_20260929" / "capacity_report.json"),
                "temporal_sample": load_json(sample_dir / "capacity_report.json"),
                "temporal_long": temporal_long,
                "all_people_capacity": load_json(sample_dir / "all_people_capacity_report.json"),
                "indexed_retrieval": load_json(sample_dir / "all_people_index_retrieval_report.json"),
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
            l7 = mem.get("layer7", {})
            moments = l7.get("moments", {})
            dimensions = ("people", "locations", "time_nodes",
                          "activity_events", "environments", "objects")
            indices = {k: l7.get(k, {}) for k in dimensions}
            # Long-term UI and the full-day multimodal run use the same A1_JAKE
            # DAY1 time slots with different synthetic dates. Join by time-of-day,
            # then require an exact normalized object label before attaching a frame.
            object_index_samples = {}
            if key == "long_term":
                v9_dir = RUNS / "multimodal_full_day_v9_object_grounded_20261011"
                v10_dir = RUNS / "multimodal_full_day_v10_object_crops_20261012"
                def progress_rows_for(directory):
                    path = directory / "object_grounding_progress.jsonl"
                    rows = []
                    if path.is_file():
                        with path.open(encoding="utf-8") as handle:
                            for line in handle:
                                try:
                                    row = json.loads(line)
                                    if row.get("id") and isinstance(row.get("object_visuals"), dict):
                                        rows.append(row)
                                except ValueError:
                                    continue
                    return rows
                v9_rows = progress_rows_for(v9_dir)
                v10_rows = progress_rows_for(v10_dir)
                v9_report = load_json(v9_dir / "object_grounding_report.json")
                v10_report = load_json(v10_dir / "object_grounding_report.json")
                v9_expected = int((v9_report or {}).get("moments_total") or 725)
                v10_expected = int((v10_report or {}).get("moments_total") or 725)
                v9_complete = v9_report is not None and len(v9_rows) >= v9_expected
                v10_complete = v10_report is not None and len(v10_rows) >= v10_expected
                if v10_complete:
                    grounded_db = load_json(v10_dir / "memory.json")
                    base_run = v10_dir.name
                elif v9_complete:
                    grounded_db = load_json(v9_dir / "memory.json")
                    base_run = v9_dir.name
                else:
                    grounded_db = load_json(RUNS / "multimodal_full_day_v8_20261007" / "memory.json")
                    base_run = "multimodal_full_day_v8_20261007"
                source_archive = ((grounded_db or {}).get("layer7") or {}).get("moments") or {}
                source_run_by_id = {str(mid): base_run for mid in source_archive}
                # While v10 is being rebuilt, keep v9 for not-yet-processed moments
                # and overlay each completed v10 row as soon as its crop assets exist.
                if not v10_complete:
                    for rows, run_name in ((v9_rows, v9_dir.name), (v10_rows, v10_dir.name)):
                        for row in rows:
                            source_moment = source_archive.get(str(row.get("id") or ""))
                            if source_moment is not None:
                                source_moment.setdefault("multimodal", {})["object_visuals"] = row.get("object_visuals") or {}
                                source_run_by_id[str(row.get("id"))] = run_name
                grounded_archive = source_archive
                by_clock = {
                    str(item.get("timestamp") or "")[11:19]: item
                    for item in grounded_archive.values()
                    if len(str(item.get("timestamp") or "")) >= 19
                }
                def object_key(value):
                    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
                for moment in moments.values():
                    clock = str(moment.get("timestamp") or "")[11:19]
                    source = by_clock.get(clock)
                    if not source:
                        continue
                    source_mm = source.get("multimodal") or {}
                    grounding = source_mm.get("object_visuals") or {}
                    labels = {
                        object_key(label): str(label).strip()
                        for label in moment.get("objects") or []
                        if str(label or "").strip()
                    }
                    matched = []
                    for item in grounding.get("items") or []:
                        label = labels.get(object_key(item.get("object")))
                        if not label:
                            continue
                        copied = dict(item)
                        copied["object"] = label
                        source_run = source_run_by_id.get(str(source.get("id") or ""), base_run)
                        copied["frame_refs"] = [
                            {**ref,
                             "url": "/api/archive_asset?run=" + quote(source_run) + "&p=" + quote(str(ref["asset"])),
                             **({"source_url": "/api/archive_asset?run=" + quote(source_run) + "&p=" + quote(str(ref["source_asset"]))}
                                if ref.get("source_asset") else {})}
                            for ref in item.get("frame_refs") or []
                            if ref.get("asset")
                        ]
                        matched.append(copied)
                    if matched:
                        moment.setdefault("multimodal", {})["object_visuals"] = {
                            "schema": 1, "status": "completed",
                            "model": grounding.get("model") or "Qwen3-VL",
                            "prompt_version": grounding.get("prompt_version"),
                            "source_run": source_run,
                            "items": matched,
                        }
                for tag, ids in (indices.get("objects") or {}).items():
                    matches = []
                    for mid in ids:
                        moment = moments.get(mid)
                        if not moment:
                            continue
                        grounding = ((moment.get("multimodal") or {}).get("object_visuals") or {})
                        exact = [
                            item for item in grounding.get("items") or []
                            if object_key(item.get("object")) == object_key(tag)
                            and item.get("frame_refs")
                        ]
                        if exact:
                            matches.append(moment)
                    matches.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
                    object_index_samples[tag] = [
                        {
                            field: moment.get(field)
                            for field in ("id", "timestamp", "scene", "user_action", "objects", "multimodal")
                            if moment.get(field) is not None
                        }
                        for moment in matches[:3]
                    ]
            # 按六维高频索引挑少量正文样例，避免把全部归档重复发给浏览器。
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
                            sample = {
                                key: moment.get(key)
                                for key in ("id", "timestamp", "scene", "user_action",
                                            "interaction_details", "person_observations",
                                            "temporal_facts", "source_time",
                                            "people", "location", "activity",
                                            "environments", "objects", "notes",
                                            "source_provenance",
                                            "highlighted", "highlight_source",
                                            "highlighted_at", "merged_into",
                                            "merged_sources", "merge_history")
                                if moment.get(key) is not None
                            }
                            if moment.get("multimodal"):
                                sample["multimodal"] = moment["multimodal"]
                            examples.append(sample)
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
                "object_index_samples": object_index_samples,
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
                    return self._send(409, {"error": "当前没有正在运行的场景检索"})
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
                reason_context = {"source_moment_id": latest.get("id"), "retrieval_only": True, "memory_modalities": "text"}
            else:
                reason_context = {"retrieval_only": True, "memory_modalities": "text"}
                if not situation:
                    return self._send(400, {"error": "请填写当前场景"})
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
                result = SESSION.daytime.retrieve_relevant_memories(situation, context=reason_context)
                log_path = SESSION.mem.memory_dir / "scene_retrieval_log.jsonl"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                return self._send(200, result)
            except Exception as exc:
                if cancel_event.is_set() or exc.__class__.__name__ == "MemoryReasoningCancelled":
                    return self._send(409, {"error": "场景检索已停止", "cancelled": True})
                return self._send(503, {"error": f"场景检索未完成：{exc}"})
            finally:
                with REASON_LOCK:
                    if ACTIVE_REASON.get("request_id") == request_id:
                        ACTIVE_REASON["request_id"] = None
                        ACTIVE_REASON["event"] = None
                STEP_LOCK.release()

        if url.path == "/api/interactive/new":
            reset_requested = (qs.get("reset") or ["0"])[0] in ("1", "true", "True")
            if SESSION is not None and not reset_requested:
                return self._send(200, _compact_interactive_state(SESSION.state()))
            if reset_requested:
                RESTORED_SESSION_ACTIVE = False
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
                    resume_state = None
                    if SESSION_RESUME_CHECKPOINT.is_file() and not reset_requested:
                        try:
                            fresh = True  # Saved sessions remain valid across overnight restarts.
                            if fresh:
                                candidate = load_json(SESSION_RESUME_CHECKPOINT)
                                if isinstance(candidate, dict):
                                    saved_progress = candidate.get("step_progress") or {}
                                    if (candidate.get("integrity_version") == SESSION_INTEGRITY_VERSION
                                            and candidate.get("scenario") in ir.SCENARIO_META
                                            and not saved_progress.get("running")):
                                        resume_state = candidate
                                        key = str(candidate["scenario"])
                                        participant = candidate.get("participant") or None
                                        dataset = candidate.get("dataset_key") or "base"
                                    else:
                                        rejected = SESSION_RESUME_CHECKPOINT.with_name(
                                            f"resume_rejected_pre_video_only_{time.time_ns()}.json"
                                        )
                                        shutil.copy2(SESSION_RESUME_CHECKPOINT, rejected)
                                        SESSION_RESUME_CHECKPOINT.unlink()
                        except (OSError, ValueError, TypeError, KeyError):
                            resume_state = None
                    SESSION = InteractiveSession(
                        key, participant=participant, dataset=dataset,
                        resume_state=resume_state,
                    )
                    if resume_state is not None:
                        RESTORED_SESSION_ACTIVE = True
                        archived = SESSION_RESUME_CHECKPOINT.with_name(
                            f"resume_restored_{time.time_ns()}.json"
                        )
                        shutil.copy2(SESSION_RESUME_CHECKPOINT, archived)
                except Exception as e:
                    return self._send(500, {"error": f"创建会话失败: {e}"})
                state = SESSION.state()
                save_session_checkpoint(state)
                return self._send(200, _compact_interactive_state(state))
            finally:
                STEP_LOCK.release()

        if url.path == '/api/interactive/entities':
            from entity_records import EntityRecordIndex
            if not ENTITY_READ_LOCK.acquire(timeout=15):
                return self._send(503, {'error': '查询队列繁忙，请稍后重试'})
            try:
                try:
                    offset = max(0, int((qs.get('offset') or ['0'])[0]))
                    limit = max(1, min(50, int((qs.get('limit') or ['12'])[0])))
                except ValueError:
                    return self._send(400, {'error': 'offset 和 limit 必须是整数'})
                scenario = (qs.get('scenario') or
                            [SESSION.key if SESSION is not None else 'custom_test'])[0]
                if scenario not in ('short_term', 'long_term', 'custom_test'):
                    return self._send(400, {'error': 'unknown scenario'})
                if SESSION is not None:
                    if not STEP_LOCK.acquire(blocking=False):
                        return self._send(409, {'error': '记忆正在写入，请稍后刷新'})
                    try:
                        layer7 = SESSION.mem.memory.get('layer7') or {}
                        moments = layer7.get('moments') or {}
                        identity_root = Path(SESSION.media_store.root)
                        face_model_root = Path(os.environ.get(
                            "MEMORY_FACEID_ROOT", str(Path.home() / ".insightface")
                        )) / "models" / "buffalo_l"
                        face_model_signature = []
                        worker_path = ROOT / "scripts" / "memory" / "face_identity_worker.py"
                        face_model_paths = [
                            face_model_root / "det_10g.onnx",
                            face_model_root / "w600k_r50.onnx",
                            worker_path,
                        ]
                        for model_path in face_model_paths:
                            model_name = model_path.name
                            try:
                                model_stat = model_path.stat()
                                face_model_signature.append((model_name, model_stat.st_mtime_ns,
                                                             model_stat.st_size))
                            except OSError:
                                face_model_signature.append((model_name, 0, 0))
                        embedding_cache = identity_root / "person_face_embedding_cache.json"
                        correction_file = identity_root / "person_identity_corrections.json"
                        identity_state_signature = []
                        for state_file in (embedding_cache, correction_file):
                            try:
                                state_stat = state_file.stat()
                                identity_state_signature.append(
                                    (state_file.name, state_stat.st_mtime_ns, state_stat.st_size)
                                )
                            except OSError:
                                identity_state_signature.append((state_file.name, 0, 0))
                        revision = (
                            getattr(SESSION, 'extraction_batch_id', id(SESSION)),
                            int(getattr(SESSION, 'pos', 0) or 0),
                            len(getattr(SESSION, 'operation_log', []) or []),
                            len(moments),
                            tuple(face_model_signature),
                            tuple(identity_state_signature),
                        )
                        index = getattr(SESSION, '_entity_record_index', None)
                        if (index is None or
                                getattr(SESSION, '_entity_record_index_revision', None) != revision):
                            index = EntityRecordIndex(
                                layer7, asset_root=identity_root,
                                identity_correction_path=identity_root / "person_identity_corrections.json",
                            )
                            SESSION._entity_record_index = index
                            SESSION._entity_record_index_revision = revision
                        result = index.query(kind=(qs.get('kind') or ['all'])[0],
                                             query=(qs.get('q') or [''])[0], offset=offset, limit=limit)
                    finally:
                        STEP_LOCK.release()
                else:
                    memory_path = RUNS / 'interactive_mem' / 'current' / 'memory.json'
                    if not memory_path.is_file() and scenario in SCENARIOS:
                        memory_path = RUNS / SCENARIOS[scenario]['memory']
                    try:
                        stat = memory_path.stat()
                    except OSError:
                        return self._send(404, {'error': '该场景还没有已保存的记忆'})
                    identity_root = memory_path.parent
                    correction_path = identity_root / "person_identity_corrections.json"
                    embedding_path = identity_root / "person_face_embedding_cache.json"
                    def _identity_stat(path):
                        try:
                            current = path.stat()
                            return current.st_mtime_ns, current.st_size
                        except OSError:
                            return 0, 0
                    signature = (str(memory_path.resolve()), stat.st_mtime_ns, stat.st_size,
                                 _identity_stat(correction_path), _identity_stat(embedding_path))
                    cached = ENTITY_ARCHIVE_INDEX_CACHE.get(scenario)
                    if not cached or cached[0] != signature:
                        memory = load_json(memory_path)
                        if not isinstance(memory, dict):
                            return self._send(404, {'error': '该场景还没有已保存的记忆'})
                        index = EntityRecordIndex(
                            memory.get('layer7') or {}, asset_root=identity_root,
                            identity_correction_path=correction_path,
                        )
                        ENTITY_ARCHIVE_INDEX_CACHE[scenario] = (signature, index)
                    else:
                        index = cached[1]
                    result = index.query(
                        kind=(qs.get('kind') or ['all'])[0],
                        query=(qs.get('q') or [''])[0], offset=offset, limit=limit)
                for item in result['items']:
                    for event in item['events']:
                        for ref in event['images']:
                            if ref.get('asset'):
                                ref['url'] = '/api/memory_asset?p=' + quote(ref['asset'])
                            if ref.get('source_asset'):
                                ref['source_url'] = '/api/memory_asset?p=' + quote(ref['source_asset'])
                            if ref.get('at_s') is None:
                                ref['at_s'] = ref.get('frame_time_s', 0)
                result['scenario'] = scenario
                return self._send(200, result)
            finally:
                ENTITY_READ_LOCK.release()

        if url.path == "/api/interactive/memory_record":
            moment_id = (qs.get("id") or [""])[0].strip()
            if not moment_id:
                return self._send(400, {"error": "missing memory id"})

            # Prefer the live session. When the page is displaying persisted memory
            # after a web-server restart, SESSION can be empty; read the selected
            # scenario's saved memory file instead of returning "no session".
            if SESSION is not None:
                if not STEP_LOCK.acquire(blocking=False):
                    return self._send(409, {"error": "memory is being written"})
                try:
                    layer7 = SESSION.mem.memory.get("layer7") or {}
                    record = (layer7.get("moments") or {}).get(moment_id)
                    if isinstance(record, dict):
                        return self._send(200, {
                            "moment_id": moment_id,
                            "scenario": SESSION.key,
                            "record": _with_memory_asset_urls(record),
                        })
                finally:
                    STEP_LOCK.release()

            requested_scenario = (qs.get("scenario") or [getattr(SESSION, "key", "")])[0]
            allowed_scenarios = ("short_term", "long_term", "custom_test")
            scenarios = ([requested_scenario] if requested_scenario in allowed_scenarios
                         else list(allowed_scenarios) if not requested_scenario else [])
            for scenario in scenarios:
                memory_path = RUNS / "interactive_mem" / "current" / "memory.json"
                try:
                    memory = load_json(memory_path)
                except (OSError, ValueError, TypeError):
                    continue
                if not isinstance(memory, dict):
                    continue
                layer7 = memory.get("layer7") or {}
                record = (layer7.get("moments") or {}).get(moment_id)
                if isinstance(record, dict):
                    return self._send(200, {
                        "moment_id": moment_id,
                        "scenario": scenario,
                        "record": _with_memory_asset_urls(record),
                    })
            return self._send(404, {"error": "memory record not found"})

        if url.path == "/api/interactive/state":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            return self._send(200, _compact_interactive_state(SESSION.state()))

        if url.path == "/api/interactive/retry_failed":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "busy", "message": "当前步骤仍在运行"})
            cancel_event = threading.Event()
            retry_results = []
            try:
                if SESSION.pos < 1 or not SESSION.log:
                    return self._send(409, {"error": "没有可重试的批量写入结果"})
                last = SESSION.log[-1]
                perception = last.get("perception") or {}
                items = perception.get("items") or []
                previous_step = SESSION.steps[SESSION.pos - 1]
                if (last.get("kind") != "add_batch"
                        or previous_step.get("kind") != "add_batch"
                        or len(items) != len(previous_step.get("inputs") or [])):
                    return self._send(409, {"error": "最近一步不是可核对的批量写入"})
                failed = [index for index, info in enumerate(items)
                          if isinstance(info, dict) and not info.get("moment_id")
                          and not info.get("cancelled")]
                if not failed:
                    return self._send(409, {"error": "没有待重试的失败片段"})
                inputs = previous_step.get("inputs") or []
                moment_by_id = {str(moment.get("id") or ""): moment
                                for moment in SESSION.moments}
                retry_inputs = []
                for index in failed:
                    source_id = str(inputs[index].get("id") or "")
                    moment = moment_by_id.get(source_id)
                    if not moment:
                        return self._send(409, {
                            "error": f"无法定位失败片段：{source_id or index + 1}"
                        })
                    retry_inputs.append((index, source_id, moment))

                ACTIVE_STEP_FORCE = True
                with SESSION._cancel_lock:
                    SESSION._active_step_cancel = cancel_event
                    SESSION._active_step_force = True
                    SESSION._active_step_no = SESSION.pos
                    SESSION._active_step_title = "重试失败片段 · " + last.get("title", "批量写入")
                SESSION._set_step_progress(
                    running=True, stage="retry_failed", label="重试失败片段",
                    step_index=SESSION.pos, step_total=len(SESSION.steps),
                    item_index=0, item_total=len(retry_inputs), message="准备重试未写入的片段",
                    output=None, completed=[], run_id=uuid.uuid4().hex,
                )
                for retry_index, (item_index, source_id, moment) in enumerate(retry_inputs, 1):
                    progress = {
                        "label": "重试失败片段", "item_index": retry_index,
                        "item_total": len(retry_inputs),
                    }
                    SESSION._set_step_progress(
                        **progress, stage="retry_failed",
                        message=f"正在重新感知第 {retry_index}/{len(retry_inputs)} 个失败片段：{source_id}",
                    )
                    moment_id, info = SESSION._add_one(moment, force=True, progress=progress)
                    info = dict(info or {})
                    info.setdefault("dataset_id", source_id)
                    items[item_index] = info
                    retry_results.append({
                        "dataset_id": source_id,
                        "moment_id": moment_id,
                        "error": info.get("error"),
                        "cancelled": bool(info.get("cancelled")),
                    })
                    if info.get("cancelled") or cancel_event.is_set():
                        break

                perception["ok"] = sum(bool(info.get("moment_id"))
                                       for info in items if isinstance(info, dict))
                perception["cancelled"] = bool(cancel_event.is_set())
                remaining = sum(not (isinstance(info, dict) and info.get("moment_id"))
                                and not (isinstance(info, dict) and info.get("cancelled"))
                                for info in items)
                last["msg"] = (
                    f"12:00 时段：Qwen 感知 {perception['ok']}/{perception.get('n', len(items))} 条并写入记忆"
                    if not remaining else
                    f"失败片段重试后仍有 {remaining} 条未写入，其他已完成内容保留"
                )
                snap = ir.rs.snapshot(SESSION.mem, last.get("title", "批量写入"),
                                      action=last.get("kind", "add_batch"))
                snap["full"] = SESSION._full_snapshot()
                snap["layer7_active_moments"] = snap["full"]["layer7_active_total"]
                snap["layer7_merged_moments"] = snap["full"]["layer7_merged_total"]
                last["snapshot"] = snap
                SESSION._set_step_progress(
                    running=False, stage="done" if not remaining else "retry_failed_partial",
                    message=last["msg"], output=None,
                )
                response = SESSION.state()
                response["retry_results"] = retry_results
                response["remaining_failed"] = remaining
                return self._send(200, _compact_interactive_state(response))
            except Exception as exc:
                SESSION._set_step_progress(running=False, stage="error",
                                           message=f"失败片段重试出错：{str(exc)[:240]}")
                return self._send(500, {"error": f"失败片段重试出错：{exc}"})
            finally:
                ACTIVE_STEP_FORCE = False
                with SESSION._cancel_lock:
                    if SESSION._active_step_cancel is cancel_event:
                        SESSION._active_step_cancel = None
                        SESSION._active_step_force = False
                    SESSION._active_step_no = None
                    SESSION._active_step_title = None
                STEP_LOCK.release()

        if url.path == "/api/interactive/cancel":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            with SESSION._cancel_lock:
                progress = SESSION.step_progress or {}
                has_step_event = SESSION._active_step_cancel is not None
                step_running = bool(STEP_LOCK.locked() and (
                    (progress.get("running") and has_step_event)
                    or (ACTIVE_STEP_RUNNING and ACTIVE_STEP_PHASE == "starting")
                ))
            if not step_running:
                return self._send(409, {
                    "error": "当前没有正在运行的步骤",
                    "message": "当前没有需要终止的运行步骤",
                })
            result = SESSION.cancel_current()
            if not result.get("requested"):
                return self._send(409, {
                    "error": "当前处理已结束或尚未开始",
                    "message": "当前步骤已结束，请刷新进度后再操作",
                })
            progress = SESSION.step_progress or {}
            SESSION._set_step_progress(
                running=True, stage="cancelling",
                message="已收到终止请求，正在停止当前处理并释放推理资源",
                output=progress.get("output"),
            )
            request_id = getattr(SESSION, "_active_reasoning_request_id", None)
            if request_id:
                def cancel_reasoning(request_id=request_id):
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
                        pass
                threading.Thread(target=cancel_reasoning, daemon=True,
                                 name="cancel-active-reasoning").start()
                result["reasoning_cancel_pending"] = True
            return self._send(200, result)

        if url.path == "/api/interactive/step":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            force = (qs.get("force") or ["0"])[0] in ("1", "true", "True")
            automatic = (qs.get("automatic") or ["0"])[0] in ("1", "true", "True")
            continuous_kind = (qs.get("continuous_kind") or [None])[0]
            use_cache = (qs.get("use_cache") or ["1"])[0] not in ("0", "false", "False")
            progress_id = (qs.get("progress_id") or [None])[0]
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "busy",
                                        "message": "上一步仍在执行（实时推理可能需要数分钟），请稍候"})
            ACTIVE_STEP_RUNNING = True
            ACTIVE_STEP_PHASE = "starting"
            ACTIVE_STEP_FORCE = force
            try:
                try:
                    msg = SESSION.run_step(force=force, automatic=automatic,
                                            continuous_kind=continuous_kind,
                                            use_cache=use_cache,
                                            progress_id=progress_id)
                    ACTIVE_STEP_PHASE = "finishing"
                except Exception as e:
                    return self._send(500, {"error": f"执行失败: {e}"})
                st = SESSION.state()
                st["last_result"] = msg
                st["last_force"] = force
                st["auto_paused"] = SESSION.auto_paused
                save_session_checkpoint(st)
                return self._send(200, _compact_interactive_state(st))
            finally:
                ACTIVE_STEP_FORCE = False
                ACTIVE_STEP_RUNNING = False
                ACTIVE_STEP_PHASE = None
                with SESSION._cancel_lock:
                    SESSION._pending_cancel = False
                STEP_LOCK.release()

        # Qwen 缓存预热（后台跑，前端轮询进度）
        if url.path == "/api/interactive/prewarm":
            key = (qs.get("scenario") or ["short_term"])[0]
            external_status = multimodal_prewarm_status(ROOT)
            if external_status and external_status.get("running"):
                return self._send(200, {"started": False, **external_status})
            if PREWARM["running"]:
                return self._send(200, {"started": False, **PREWARM})

            def work():
                PREWARM.update(running=True, done=0, total=0, cache_hits=0,
                               inferred=0, failed=0, failed_items=[], current_id=None,
                               current_status="starting", error=None, scenario=key)
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
                            ms = ir._load_jsonl(src)
                            if participant:
                                ms = [m for m in ms
                                      if "_".join(str(m.get("id", "")).split("_")[1:3]) == participant]
                        elif dataset.startswith("feishu:"):
                            ms = ir.feishu_test_moments(dataset)
                        else:
                            participant = participant or "A1_JAKE"
                            src = ir.LONG_PERSON_DIR / f"{participant}.jsonl"
                            ms = ir._load_jsonl(src)
                        p = ir.site_perceiver()
                    PREWARM["total"] = len(ms)
                    worker_count = max(1, min(4, len(getattr(p, "api_urls", []) or [p.api_url]),
                                              len(ms) or 1))
                    PREWARM["worker_count"] = worker_count
                    with ThreadPoolExecutor(max_workers=worker_count,
                                            thread_name_prefix="qwen-prewarm") as executor:
                        futures = {executor.submit(p.perceive, moment): moment
                                   for moment in ms}
                        for future in as_completed(futures):
                            moment = futures[future]
                            moment_id = str(moment.get("id") or "")
                            try:
                                result = future.result()
                                source = result.get("source")
                                if result.get("fields") and source == "cache":
                                    PREWARM["cache_hits"] += 1
                                elif result.get("fields") and source == "qwen":
                                    PREWARM["inferred"] += 1
                                else:
                                    PREWARM["failed"] += 1
                                    PREWARM["failed_items"].append({
                                        "id": moment_id,
                                        "source": str(source or ""),
                                        "error": str(result.get("error") or result.get("message") or
                                                     "perception returned no fields")[:500],
                                    })
                            except Exception as exc:
                                PREWARM["failed"] += 1
                                PREWARM["failed_items"].append({
                                    "id": moment_id, "source": "exception",
                                    "error": str(exc)[:500],
                                })
                            finally:
                                PREWARM["done"] += 1
                                PREWARM["current_id"] = moment_id
                                PREWARM["current_status"] = "processing"
                    PREWARM.update(current_id=None, current_status="complete")
                except Exception as e:
                    PREWARM["error"] = str(e)
                finally:
                    PREWARM["running"] = False

            threading.Thread(target=work, daemon=True).start()
            return self._send(200, {"started": True})

        if url.path == "/api/interactive/prewarm_status":
            external_status = multimodal_prewarm_status(ROOT)
            if external_status and (
                    external_status.get("running") or not PREWARM["running"]):
                return self._send(200, external_status)
            return self._send(200, dict(PREWARM))

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        global SESSION, ACTIVE_STEP_FORCE, ACTIVE_STEP_RUNNING, ACTIVE_STEP_PHASE
        url = urlparse(self.path)
        if url.path not in ("/api/interactive/merge", "/api/interactive/unmerge",
                            "/api/interactive/person_identity", "/api/interactive/clear_memory",
                            "/api/interactive/nightly"):
            return self._send(404, {"error": "not found"})
        if url.path == "/api/interactive/nightly":
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "busy", "message": "当前步骤正在运行，请稍后再整理"})
            ACTIVE_STEP_RUNNING = True
            ACTIVE_STEP_PHASE = "starting"
            ACTIVE_STEP_FORCE = False
            try:
                if SESSION is None:
                    return self._send(409, {"error": "请先创建交互会话"})
                if not SESSION.has_accumulated_memory():
                    return self._send(409, {"error": "empty_memory", "message": "当前还没有可整理的记忆"})
                params = parse_qs(url.query)
                use_cache = (params.get("use_cache") or ["1"])[0] not in ("0", "false", "False")
                force = (params.get("force") or ["0"])[0] not in ("0", "false", "False")
                ACTIVE_STEP_FORCE = force
                progress_id = (params.get("progress_id") or [None])[0]
                msg = SESSION.run_nightly_now(
                    force=force, use_cache=use_cache, progress_id=progress_id)
                ACTIVE_STEP_PHASE = "finishing"
                state = SESSION.state()
                state["last_result"] = msg
                state["manual_nightly"] = True
                save_session_checkpoint(state)
                return self._send(200, _compact_interactive_state(state))
            except Exception as exc:
                return self._send(500, {"error": f"手动夜间整理失败：{exc}"})
            finally:
                ACTIVE_STEP_FORCE = False
                ACTIVE_STEP_RUNNING = False
                ACTIVE_STEP_PHASE = None
                if SESSION is not None:
                    with SESSION._cancel_lock:
                        SESSION._pending_cancel = False
                STEP_LOCK.release()
        if url.path == "/api/interactive/clear_memory":
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "记忆正在写入，无法清空"})
            try:
                if SESSION is None:
                    return self._send(409, {"error": "请先创建交互会话"})
                old_dir = Path(SESSION.mem.memory_dir)
                try:
                    SESSION.daytime.end_scene()
                except Exception:
                    pass
                trash_dir = RUNS / "interactive_mem" / "_trash"
                trash_dir.mkdir(parents=True, exist_ok=True)
                archived = trash_dir / f"current_cleared_{time.time_ns()}"
                if old_dir.exists():
                    os.rename(old_dir, archived)
                old_dir.mkdir(parents=True, exist_ok=True)
                SESSION.mem = ir.PersonMemory(memory_dir=str(old_dir))
                SESSION.media_store = ir.MultimodalAssetStore(old_dir)
                SESSION.daytime = ir.DaytimeProcessor(SESSION.mem)
                SESSION.daytime.start_scene(f"accumulation_{time.strftime('%Y-%m-%d')}")
                SESSION.id_map = {}
                SESSION.log = []
                SESSION._entity_record_index = None
                SESSION._entity_record_index_revision = None
                SESSION.nightly_progress = None
                SESSION._install_operation_audit()
                save_session_checkpoint(SESSION.state())
                MEMORY_ASSET_CACHE.clear()
                ENTITY_ARCHIVE_INDEX_CACHE.clear()
                return self._send(200, {"ok": True, "archived": archived.name,
                                        "memory_count": 0,
                                        "message": "当前累积记忆已清空；原记忆已保留在可恢复归档中"})
            except Exception as exc:
                return self._send(500, {"error": f"清空累积记忆失败：{exc}"})
            finally:
                STEP_LOCK.release()
        if url.path == "/api/interactive/person_identity":
            if not STEP_LOCK.acquire(blocking=False):
                return self._send(409, {"error": "记忆正在写入，请稍后纠正"})
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 32768:
                        return self._send(400, {"error": "请求内容长度无效"})
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                    return self._send(400, {"error": "请求内容必须是有效 JSON"})
                scenario = str(payload.get("scenario") or
                               (SESSION.key if SESSION is not None else "custom_test"))
                if scenario not in ("short_term", "long_term", "custom_test"):
                    return self._send(400, {"error": "unknown scenario"})
                if SESSION is not None and scenario == SESSION.key:
                    identity_root = Path(SESSION.media_store.root)
                    layer7 = SESSION.mem.memory.get("layer7") or {}
                else:
                    memory_path = RUNS / "interactive_mem" / "current" / "memory.json"
                    if not memory_path.is_file():
                        return self._send(404, {"error": "该场景还没有已保存的记忆"})
                    identity_root = memory_path.parent
                    memory = load_json(memory_path) or {}
                    layer7 = memory.get("layer7") or {}
                from entity_records import EntityRecordIndex
                from face_identity import write_correction
                index = EntityRecordIndex(
                    layer7, asset_root=identity_root,
                    identity_correction_path=identity_root / "person_identity_corrections.json",
                )
                valid_ids = {str(member_id) for row in index.records if row.get("kind") == "person"
                             for member_id in (row.get("member_ids") or [row.get("id")])}
                source_member = str(payload.get("source_member_id") or "")
                target_member = str(payload.get("target_member_id") or "")
                if not source_member or not target_member:
                    return self._send(400, {"error": "请指定两条具体人物成员记录"})
                if (source_member not in valid_ids or target_member not in valid_ids
                        or source_member == target_member):
                    return self._send(404, {"error": "人物成员记录已变化，请刷新人物记忆后重试"})
                action = str(payload.get("action") or "")
                result = write_correction(
                    identity_root / "person_identity_corrections.json",
                    action, source_member, target_member,
                    reason=str(payload.get("reason") or "人物页人工纠错"),
                )
                if not result.get("ok"):
                    return self._send(400, result)
                ENTITY_ARCHIVE_INDEX_CACHE.pop(scenario, None)
                if SESSION is not None and scenario == SESSION.key:
                    SESSION._entity_record_index = None
                    SESSION._entity_record_index_revision = None
                return self._send(200, {"ok": True, "action": action,
                                        "message": "身份纠错已保存，人物记忆已按纠错记录刷新"})
            finally:
                STEP_LOCK.release()
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
            return self._send(200, _compact_interactive_state(state))
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
    # Keep local crop and face models resident on their explicitly assigned GPUs.
    for module_name, function_name in (
        ("clip_crop_grounding", "warmup"),
        ("face_detection", "warmup"),
    ):
        try:
            module = __import__(module_name, fromlist=[function_name])
            result = getattr(module, function_name)()
            print(f"  本地 GPU 模型就绪: {result}")
        except Exception as exc:
            print(f"  GPU 模型预热失败（网页仍可用）: {module_name}: {str(exc)[:240]}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
