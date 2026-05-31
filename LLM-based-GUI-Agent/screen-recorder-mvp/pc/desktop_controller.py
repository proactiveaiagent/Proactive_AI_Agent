# -*- coding: utf-8 -*-
"""桌面 Web 壳与 API 共用的业务状态：录屏、接收服务、分析、记忆。"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recorder import ScreenRecorder
from receiver import app as flask_receiver_app
from receiver import run_receiver
from config import (
    DASHSCOPE_API_KEY,
    FRAME_SAMPLE_INTERVAL_SEC,
    GUI_AGENT_BACKEND,
    LOCAL_GUI_OWL_MODEL_ID,
    SSIM_THRESHOLD,
)

RECEIVER_PORT = 8765
WEB_UI_PORT = 8776


def get_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def local_ip() -> str:
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class DesktopController:
    """单例：线程安全的状态与核心操作。"""

    _singleton_lock = threading.Lock()
    _instance: Optional["DesktopController"] = None

    @classmethod
    def instance(cls) -> "DesktopController":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = DesktopController()
            return cls._instance

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.base_dir = get_base_dir()
        recordings_dir = os.path.join(self.base_dir, "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        self.recorder = ScreenRecorder(fps=15, output_dir=recordings_dir)
        self.receiver_thread: Optional[threading.Thread] = None
        self.receiver_running = False
        self._analysis_running = False
        self.selected_video: Optional[str] = None
        self._analysis_result_dir: Optional[str] = None
        self._logs: deque = deque(maxlen=400)
        self._progress: Dict[str, Any] = {"pct": 0.0, "message": ""}
        self.backend_default = "local" if GUI_AGENT_BACKEND != "api" else "api"
        self._deploy_running = False

    def log(self, msg: str) -> None:
        with self._lock:
            self._logs.append({"t": time.time(), "msg": msg})

    def logs_count(self) -> int:
        with self._lock:
            return len(self._logs)

    def logs_since(self, start_index: int) -> List[Dict[str, Any]]:
        with self._lock:
            logs = list(self._logs)
        out = []
        for i in range(max(0, start_index), len(logs)):
            out.append({"i": i, "t": logs[i]["t"], "msg": logs[i]["msg"]})
        return out

    def set_progress(self, pct: float, message: str) -> None:
        with self._lock:
            self._progress = {"pct": float(pct), "message": message}

    def get_progress(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._progress)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "recording": self.recorder.is_recording(),
                "receiver_running": self.receiver_running,
                "receiver_port": RECEIVER_PORT,
                "receiver_upload_url": (
                    f"http://{local_ip()}:{RECEIVER_PORT}/upload"
                    if self.receiver_running
                    else ""
                ),
                "analysis_running": self._analysis_running,
                "deploy_running": self._deploy_running,
                "selected_video": self.selected_video,
                "analysis_result_dir": self._analysis_result_dir,
                "progress": dict(self._progress),
                "backend_default": self.backend_default,
                "default_model_id": LOCAL_GUI_OWL_MODEL_ID,
                "base_dir": self.base_dir,
            }

    def toggle_record(self) -> Dict[str, Any]:
        if self.recorder.is_recording():
            path = self.recorder.stop()
            self.log(f"录制已停止: {path}")
            return {"ok": True, "recording": False, "path": path}
        ok, info = self.recorder.start()
        if ok:
            self.log(f"录制已开始: {info}")
            return {"ok": True, "recording": True, "path": info}
        self.log(f"录制失败: {info}")
        return {"ok": False, "error": str(info)}

    def start_receiver(self) -> Dict[str, Any]:
        if self.receiver_running:
            return {
                "ok": True,
                "already": True,
                "url": f"http://{local_ip()}:{RECEIVER_PORT}/upload",
            }
        upload_dir = os.path.join(self.base_dir, "recordings")
        os.makedirs(upload_dir, exist_ok=True)
        flask_receiver_app.config["UPLOAD_FOLDER"] = upload_dir

        def run() -> None:
            try:
                run_receiver(host="0.0.0.0", port=RECEIVER_PORT)
            except Exception as e:
                self.log(f"接收服务异常退出: {e}")

        with self._lock:
            self.receiver_running = True
        self.receiver_thread = threading.Thread(target=run, daemon=True)
        self.receiver_thread.start()
        url = f"http://{local_ip()}:{RECEIVER_PORT}/upload"
        self.log(f"上传接收服务已启动: {url}")
        time.sleep(0.25)
        return {"ok": True, "url": url}

    def stop_receiver(self) -> Dict[str, Any]:
        """与 Tk 版一致：仅更新 UI 状态；Flask 线程可能仍占用端口。"""
        with self._lock:
            self.receiver_running = False
        self.log("上传接收服务状态已标记为停止（后台线程可能仍占用端口，完整释放请重启应用）")
        return {"ok": True, "warning": "flask_may_still_hold_port"}

    def set_selected_video(self, path: str) -> Dict[str, Any]:
        path = (path or "").strip()
        if not path:
            with self._lock:
                self.selected_video = None
            return {"ok": True, "cleared": True}
        if not os.path.isfile(path):
            return {"ok": False, "error": "文件不存在"}
        with self._lock:
            self.selected_video = path
        self.log(f"已选择视频: {path}")
        return {"ok": True, "path": path}

    def list_recordings(self) -> List[Dict[str, Any]]:
        d = os.path.join(self.base_dir, "recordings")
        if not os.path.isdir(d):
            return []
        names = sorted(os.listdir(d), key=lambda n: os.path.getmtime(os.path.join(d, n)), reverse=True)
        out: List[Dict[str, Any]] = []
        for name in names:
            if not name.lower().endswith((".mp4", ".webm", ".mkv")):
                continue
            p = os.path.join(d, name)
            try:
                st = os.stat(p)
                out.append(
                    {
                        "name": name,
                        "path": p,
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                    }
                )
            except OSError:
                continue
        return out

    def open_folder(self, kind: str) -> Dict[str, Any]:
        """kind: recordings | analysis"""
        if kind == "recordings":
            folder = os.path.join(self.base_dir, "recordings")
        elif kind == "analysis":
            with self._lock:
                last = self._analysis_result_dir
            folder = last or os.path.join(self.base_dir, "analysis_output")
        else:
            return {"ok": False, "error": "unknown kind"}
        os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]
        else:
            import subprocess

            subprocess.run(["xdg-open", folder], check=False)
        return {"ok": True, "path": folder}

    def start_analysis(
        self,
        backend: str,
        api_key: str,
        model_id: str,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._analysis_running:
                return {"ok": False, "error": "分析进行中"}
            path = self.selected_video
        if not path or not os.path.isfile(path):
            return {"ok": False, "error": "请先选择有效视频"}
        backend = (backend or self.backend_default).strip().lower()
        model_id = (model_id or LOCAL_GUI_OWL_MODEL_ID).strip()
        key = (api_key or DASHSCOPE_API_KEY or "").strip()
        if backend == "api" and not key:
            return {"ok": False, "error": "api 模式需要 API Key"}

        with self._lock:
            self._analysis_running = True
        self.set_progress(0, "准备分析...")
        self.log(
            f"开始分析: {path} | backend={backend} | model={model_id}",
        )

        def worker() -> None:
            try:
                from analyzer import analyze_video

                if backend == "local":
                    from gui_agent_api import ensure_local_model_ready

                    try:

                        def pcb(m: str) -> None:
                            self.set_progress(0, f"[检查] {m}")

                        ensure_local_model_ready(
                            base_dir=self.base_dir,
                            model_id=model_id,
                            progress_cb=pcb,
                            allow_download=False,
                        )
                    except Exception as e:
                        self._finish_analysis(None, str(e))
                        return

                def on_progress(current: int, total: int, message: str) -> None:
                    if total > 0:
                        pct = min(100.0, current / total * 100)
                    else:
                        pct = 0.0
                    self.set_progress(pct, message)

                output_base = os.path.join(self.base_dir, "analysis_output")
                result_dir = analyze_video(
                    video_path=path,
                    api_key=key,
                    backend=backend,
                    app_base_dir=self.base_dir,
                    model=model_id,
                    interval_sec=FRAME_SAMPLE_INTERVAL_SEC,
                    ssim_threshold=SSIM_THRESHOLD,
                    output_dir=output_base,
                    progress_cb=on_progress,
                )
                self._finish_analysis(result_dir, None)
            except Exception as e:
                self._finish_analysis(None, str(e))

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "started": True}

    def _finish_analysis(self, result_dir: Optional[str], error: Optional[str]) -> None:
        with self._lock:
            self._analysis_running = False
        if error:
            self.set_progress(0, f"分析失败: {error}")
            self.log(f"分析失败: {error}")
        else:
            with self._lock:
                self._analysis_result_dir = result_dir
            self.set_progress(100, "分析完成!")
            self.log(f"分析完成: {result_dir}")
            if result_dir:
                self._store_memory(result_dir)

    def _store_memory(self, result_dir: str) -> None:
        try:
            from memory import store_report_into_memory

            report_path = os.path.join(result_dir, "report.json")
            if not os.path.isfile(report_path):
                return
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            memory_dir = os.path.join(self.base_dir, "memory")
            os.makedirs(memory_dir, exist_ok=True)
            stats = store_report_into_memory(
                report, source_path=result_dir, memory_dir=memory_dir
            )
            self.log(
                f"已写入记忆: +{stats.get('added', 0)} 条, 压缩后 {stats.get('compressed', 0)} 条",
            )
        except Exception as e:
            self.log(f"写入记忆失败: {e}")

    def memory_stats(self) -> Dict[str, Any]:
        try:
            from memory import MemoryStore

            mem_dir = os.path.join(self.base_dir, "memory")
            os.makedirs(mem_dir, exist_ok=True)
            store = MemoryStore(os.path.join(mem_dir, "memory.db"))
            return {"ok": True, "stats": store.get_stats()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def memory_search(self, query: str, limit: int = 30) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"ok": False, "error": "空查询"}
        try:
            from memory import MemoryStore

            mem_dir = os.path.join(self.base_dir, "memory")
            store = MemoryStore(os.path.join(mem_dir, "memory.db"))
            results = store.search_natural_language(query, limit=limit)
            return {"ok": True, "results": results}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def memory_profile(self) -> Dict[str, Any]:
        try:
            from memory import MemoryStore

            mem_dir = os.path.join(self.base_dir, "memory")
            store = MemoryStore(os.path.join(mem_dir, "memory.db"))
            profile = store.get_user_profile()
            if not profile:
                profile = store.build_user_profile()
            return {"ok": True, "profile": profile}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def deploy_model(self, model_id: str) -> Dict[str, Any]:
        with self._lock:
            if self._deploy_running:
                return {"ok": False, "error": "部署进行中"}
            if self._analysis_running:
                return {"ok": False, "error": "分析进行中，请稍后再部署"}
            self._deploy_running = True
        model_id = (model_id or LOCAL_GUI_OWL_MODEL_ID).strip()
        self.log(f"开始部署本地模型: {model_id}")

        def worker() -> None:
            try:
                from gui_agent_api import deploy_local_model

                def progress(msg: str) -> None:
                    self.set_progress(0, msg)
                    self.log(msg)

                info = deploy_local_model(
                    base_dir=self.base_dir,
                    model_id=model_id,
                    progress_cb=progress,
                )
                self.log(f"本地模型部署完成: {info.get('model_id', model_id)}")
                self.set_progress(100, "部署完成")
            except Exception as e:
                self.log(f"部署失败: {e}")
                self.set_progress(0, f"部署失败: {e}")
            finally:
                with self._lock:
                    self._deploy_running = False

        threading.Thread(target=worker, daemon=True).start()
        return {"ok": True, "started": True}

    def get_latest_recording_meta(self) -> Dict[str, Any]:
        items = self.list_recordings()
        if not items:
            return {"ok": False, "error": "no recordings found"}
        return {"ok": True, "recording": items[0]}

    def find_report_path_for_video(self, video_path: str) -> Optional[str]:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        candidates = [
            os.path.join(self.base_dir, "analysis_output", stem, "report.json"),
        ]
        output_base = os.path.join(self.base_dir, "analysis_output")
        if os.path.isdir(output_base):
            for name in os.listdir(output_base):
                candidates.append(os.path.join(output_base, name, "report.json"))
        target_name = os.path.basename(video_path)
        for report_path in candidates:
            if not os.path.isfile(report_path):
                continue
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                if report.get("video") == target_name:
                    return report_path
            except Exception:
                continue
        return None

    def load_gui_report(self, video_path: str = "") -> Dict[str, Any]:
        path = (video_path or "").strip()
        if not path:
            latest = self.get_latest_recording_meta()
            if not latest.get("ok"):
                return latest
            path = latest["recording"]["path"]
        if not os.path.isfile(path):
            return {"ok": False, "error": f"video not found: {path}"}
        report_path = self.find_report_path_for_video(path)
        if not report_path:
            return {"ok": False, "error": "report not found", "video_path": path}
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return {
            "ok": True,
            "video_path": path,
            "report_path": report_path,
            "report": report,
        }

    def prepare_for_proactive(
        self,
        video_path: str = "",
        auto_analyze: bool = True,
        backend: str = "local",
        api_key: str = "",
        model_id: str = "",
        wait_timeout_sec: float = 900.0,
    ) -> Dict[str, Any]:
        """Resolve a screen recording and optionally run GUI-Owl analysis for Proactive Agent."""
        path = (video_path or "").strip()
        if not path:
            latest = self.get_latest_recording_meta()
            if not latest.get("ok"):
                return latest
            path = latest["recording"]["path"]
            recording = latest["recording"]
        else:
            if not os.path.isfile(path):
                return {"ok": False, "error": f"video not found: {path}"}
            recording = {"path": path, "name": os.path.basename(path)}

        cached = self.load_gui_report(path)
        if cached.get("ok"):
            return {
                "ok": True,
                "video_path": path,
                "recording": recording,
                "report": cached["report"],
                "report_path": cached["report_path"],
                "analyzed": False,
                "cached": True,
            }

        if not auto_analyze:
            return {
                "ok": True,
                "video_path": path,
                "recording": recording,
                "report": None,
                "report_path": None,
                "analyzed": False,
                "cached": False,
            }

        select_result = self.set_selected_video(path)
        if not select_result.get("ok"):
            return select_result

        start_result = self.start_analysis(backend, api_key, model_id)
        if not start_result.get("ok"):
            return start_result

        deadline = time.time() + wait_timeout_sec
        while time.time() < deadline:
            with self._lock:
                running = self._analysis_running
            if not running:
                break
            time.sleep(1.0)
        else:
            return {"ok": False, "error": "analysis timed out", "video_path": path}

        report_result = self.load_gui_report(path)
        if not report_result.get("ok"):
            return {
                "ok": False,
                "error": report_result.get("error", "report missing after analysis"),
                "video_path": path,
            }

        return {
            "ok": True,
            "video_path": path,
            "recording": recording,
            "report": report_result["report"],
            "report_path": report_result["report_path"],
            "analyzed": True,
            "cached": False,
        }
