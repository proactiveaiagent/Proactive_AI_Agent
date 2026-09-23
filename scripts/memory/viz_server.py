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

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "egolife" / "runs"
VIZ_DIR = Path(__file__).resolve().parent / "viz"
VIDEO_ROOT = (ROOT / "data" / "egolife").resolve()

# 交互式步进执行会话（单线程 HTTPServer，全局单例即可）
SESSION = None
# Qwen 缓存预热任务状态
PREWARM = {"running": False, "done": 0, "total": 0, "error": None, "scenario": None}

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

    def _stream_video(self, head_only=False):
        qs = parse_qs(urlparse(self.path).query)
        p = self._safe_video((qs.get("p") or [""])[0])
        if not p:
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"error": "video not found"}')
            return
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
        url = urlparse(self.path)
        qs = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            idx = VIZ_DIR / "index.html"
            if idx.exists():
                return self._send_file(idx, "text/html; charset=utf-8")
            return self._send(404, {"error": "index.html not found"})

        if url.path == "/api/video":
            return self._stream_video()

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
            # 只回传可视化需要的部分，避免全量 moment 过大
            l7 = mem.get("layer7", {})
            return self._send(200, {
                "layer1": mem.get("layer1", []),
                "layer2": mem.get("layer2", []),
                "layer3": mem.get("layer3", []),
                "layer4": mem.get("layer4", {}),
                "layer5": mem.get("layer5", {}),
                "layer6": mem.get("layer6", {}),
                "indices": {k: l7.get(k, {}) for k in
                            ("people", "locations", "time_nodes",
                             "activity_events", "environments", "objects")},
                "moments_count": len(l7.get("moments", {})),
                "metadata": mem.get("metadata", {}),
            })

        # ---------------- 交互式步进执行 ----------------
        global SESSION

        if url.path == "/api/interactive/new":
            key = (qs.get("scenario") or ["short_term"])[0]
            try:
                SESSION = InteractiveSession(key)
            except Exception as e:
                return self._send(500, {"error": f"创建会话失败: {e}"})
            return self._send(200, SESSION.state())

        if url.path == "/api/interactive/state":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            return self._send(200, SESSION.state())

        if url.path == "/api/interactive/step":
            if SESSION is None:
                return self._send(404, {"error": "no session"})
            force = (qs.get("force") or ["0"])[0] in ("1", "true", "True")
            try:
                msg = SESSION.run_step(force=force)
            except Exception as e:
                return self._send(500, {"error": f"执行失败: {e}"})
            st = SESSION.state()
            st["last_result"] = msg
            st["last_force"] = force
            return self._send(200, st)

        # Qwen 缓存预热（后台跑，前端轮询进度）
        if url.path == "/api/interactive/prewarm":
            key = (qs.get("scenario") or ["short_term"])[0]
            if PREWARM["running"]:
                return self._send(200, {"started": False, **PREWARM})

            def work():
                PREWARM.update(running=True, done=0, total=0, error=None, scenario=key)
                try:
                    src = ir.SHORT_FILE if key == "short_term" else ir.LONG_FILE
                    ms = ir._load_jsonl(src)
                    PREWARM["total"] = len(ms)
                    p = QwenPerceiver()
                    for m in ms:
                        try:
                            p.perceive(m)
                        except Exception:
                            pass
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
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
