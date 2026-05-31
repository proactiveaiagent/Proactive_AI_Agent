"""
Client for LLM-based-GUI-Agent (screen-recorder PC Web API).

Connects Proactive Agent to the GUI Agent service that records phone/PC screens,
receives mobile uploads, and runs GUI-Owl screen recognition analysis.

Default GUI Agent URL: http://localhost:8776  (main_web.py)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class GUIAgentClient:
    """HTTP client for LLM-based-GUI-Agent PC bridge API."""

    def __init__(self, base_url: str = "http://localhost:8776", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **params) -> dict:
        resp = requests.get(
            f"{self.base_url}{path}", params=params or None, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict | None = None) -> dict:
        resp = requests.post(
            f"{self.base_url}{path}", json=payload or {}, timeout=self.timeout
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"GUI Agent API error ({resp.status_code}): {detail}")
        return resp.json()

    def health(self) -> bool:
        try:
            data = self._get("/api/health")
            return bool(data.get("ok"))
        except Exception:
            return False

    def get_status(self) -> dict:
        data = self._get("/api/status")
        return data.get("status") or data

    def list_recordings(self) -> List[dict]:
        data = self._get("/api/recordings")
        return data.get("items", [])

    def get_latest_recording(self) -> Optional[dict]:
        data = self._get("/api/proactive/latest-recording")
        if not data.get("ok"):
            return None
        return data.get("recording")

    def select_video(self, path: str) -> dict:
        return self._post("/api/video/select", {"path": path})

    def start_analysis(
        self,
        backend: str = "local",
        api_key: str = "",
        model_id: str = "",
    ) -> dict:
        return self._post(
            "/api/analyze/start",
            {"backend": backend, "api_key": api_key, "model_id": model_id},
        )

    def get_gui_report(self, video_path: str = "") -> Optional[dict]:
        """Fetch structured GUI-Owl analysis report (report.json) for a recording."""
        params = {"video_path": video_path} if video_path else {}
        try:
            data = self._get("/api/proactive/gui-report", **params)
        except Exception:
            return None
        if not data.get("ok"):
            return None
        return data.get("report")

    def memory_search(self, query: str, limit: int = 10) -> List[dict]:
        try:
            data = self._post("/api/memory/search", {"query": query, "limit": limit})
            if data.get("ok"):
                return data.get("results", [])
        except Exception:
            pass
        return []

    def wait_for_analysis(
        self,
        poll_interval: float = 2.0,
        max_wait: float = 900.0,
    ) -> dict:
        """Poll GUI Agent until analysis finishes or times out."""
        deadline = time.time() + max_wait
        last_pct = -1.0
        while time.time() < deadline:
            status = self.get_status()
            running = status.get("analysis_running", False)
            progress = status.get("progress") or {}
            pct = float(progress.get("pct", 0))
            msg = progress.get("message", "")
            if pct != last_pct:
                print(f"🖥️  GUI Agent analysis: {pct:.0f}% — {msg}")
                last_pct = pct
            if not running:
                result_dir = status.get("analysis_result_dir")
                if result_dir and pct >= 99:
                    return {"ok": True, "result_dir": result_dir}
                if result_dir:
                    return {"ok": True, "result_dir": result_dir}
                if "失败" in msg or "error" in msg.lower():
                    return {"ok": False, "error": msg}
                return {"ok": True, "result_dir": result_dir}
            time.sleep(poll_interval)
        return {"ok": False, "error": "GUI Agent analysis timed out"}

    def prepare_screen_input(
        self,
        video_path: str = "",
        auto_analyze: bool = True,
        analysis_backend: str = "local",
        analysis_model_id: str = "",
        analysis_api_key: str = "",
        wait_for_analysis: bool = True,
    ) -> dict:
        """
        Resolve a screen recording and optionally run GUI-Owl analysis.

        Returns:
            {
              "video_path": str,
              "recording": dict,
              "report": dict | None,
              "report_path": str | None,
            }
        """
        payload = {
            "video_path": video_path,
            "auto_analyze": auto_analyze,
            "backend": analysis_backend,
            "model_id": analysis_model_id,
            "api_key": analysis_api_key,
            "wait": wait_for_analysis,
        }
        data = self._post("/api/proactive/prepare", payload)
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "GUI Agent prepare failed"))
        return data

    @staticmethod
    def format_report_for_prompt(report: dict, max_frames: int = 8) -> str:
        """Turn GUI-Owl report.json into text for Proactive Agent LLM prompts."""
        if not report:
            return ""

        lines = [
            "GUI Agent screen-recording analysis (GUI-Owl):",
            f"- Video: {report.get('video', 'unknown')}",
            f"- Frames analyzed: {report.get('total_frames_analyzed', 0)}",
        ]

        frames = report.get("frames") or []
        for frame in frames[:max_frames]:
            ts = frame.get("timestamp_sec", 0)
            lines.append(f"\n[Screen @ {ts:.1f}s]")
            if frame.get("error"):
                lines.append(f"  Error: {frame['error']}")
                continue
            lines.append(f"  App: {frame.get('app_name', 'N/A')}")
            lines.append(f"  Page: {frame.get('page_name', 'N/A')}")
            lines.append(f"  User Action: {frame.get('user_action', 'N/A')}")
            lines.append(f"  Description: {frame.get('description', 'N/A')}")
            texts = frame.get("visible_text") or []
            if texts:
                preview = ", ".join(str(t) for t in texts[:6])
                lines.append(f"  Visible Text: {preview}")
            elements = frame.get("elements") or []
            if elements:
                el_preview = "; ".join(
                    f"{e.get('type', '?')}:{e.get('label') or e.get('description', '')}"
                    for e in elements[:5]
                )
                lines.append(f"  UI Elements: {el_preview}")

        if len(frames) > max_frames:
            lines.append(f"\n... ({len(frames) - max_frames} more frames omitted)")

        return "\n".join(lines)

    @staticmethod
    def load_report_from_path(report_path: str) -> Optional[dict]:
        path = Path(report_path)
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
