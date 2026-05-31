# -*- coding: utf-8 -*-
"""PC Web 桌面壳入口：FastAPI + 静态前端，复用现有 Python 核心模块。"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from desktop_controller import WEB_UI_PORT
from app_server import app


def main() -> None:
    host = os.environ.get("SCREEN_RECORDER_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("SCREEN_RECORDER_WEB_PORT", str(WEB_UI_PORT)))
    url = f"http://{host}:{port}/"

    def open_browser() -> None:
        time.sleep(0.8)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if os.environ.get("SCREEN_RECORDER_NO_BROWSER") != "1":
        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
