# -*- coding: utf-8 -*-
"""Smoke test for PC web shell bridge APIs.

Usage:
  cd screen-recorder-mvp/pc
  python scripts/smoke_web_bridge.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _get_json(url: str, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return int(e.code), parsed


def main() -> int:
    pc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    port = int(os.environ.get("SCREEN_RECORDER_WEB_PORT", "18776"))
    env = os.environ.copy()
    env["SCREEN_RECORDER_WEB_HOST"] = "127.0.0.1"
    env["SCREEN_RECORDER_WEB_PORT"] = str(port)
    env["SCREEN_RECORDER_NO_BROWSER"] = "1"

    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "main_web.py"],
        cwd=pc_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        start = time.time()
        while time.time() - start < 15:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                print("Server exited unexpectedly:")
                print(out)
                return 1
            try:
                health = _get_json(f"{base}/api/health")
                if health.get("ok"):
                    break
            except Exception:
                time.sleep(0.4)
        else:
            print("Timeout waiting for /api/health")
            return 1

        status = _get_json(f"{base}/api/status")
        print("health:", health)
        print("status keys:", sorted(status.keys()))

        code, bad_select = _post_json(f"{base}/api/video/select", {"path": "Z:/not_exists.mp4"})
        print("select invalid status:", code, bad_select)
        if code != 400:
            print("Expected 400 for invalid path")
            return 1

        logs = _get_json(f"{base}/api/logs?since=0")
        print("logs count:", len(logs.get("logs", [])))
        print("OK: web bridge smoke test passed")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
