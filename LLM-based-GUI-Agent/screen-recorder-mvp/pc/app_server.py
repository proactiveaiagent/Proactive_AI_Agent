# -*- coding: utf-8 -*-
"""FastAPI：静态 Web UI + 桌面桥接 REST API。"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from desktop_controller import WEB_UI_PORT, DesktopController

_controller = DesktopController.instance()

app = FastAPI(title="Screen Recorder Desktop API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PC_DIR = os.path.dirname(os.path.abspath(__file__))
_WEBUI_DIST = os.path.join(_PC_DIR, "webui", "dist")
_INDEX_HTML = os.path.join(_WEBUI_DIST, "index.html")
_LOGO_SVG = os.path.normpath(os.path.join(_PC_DIR, "..", "design", "logo.svg"))
_APP_ASSETS = os.path.join(_PC_DIR, "assets")


class AnalyzeRequest(BaseModel):
    backend: str = "local"
    api_key: str = ""
    model_id: str = ""


class SelectVideoRequest(BaseModel):
    path: str = ""


class DeployRequest(BaseModel):
    model_id: str = ""


class MemorySearchRequest(BaseModel):
    query: str = ""
    limit: int = Field(default=30, ge=1, le=200)


class OpenFolderRequest(BaseModel):
    kind: str = "recordings"  # recordings | analysis


class ProactivePrepareRequest(BaseModel):
    video_path: str = ""
    auto_analyze: bool = True
    backend: str = "local"
    model_id: str = ""
    api_key: str = ""
    wait: bool = True


@app.get("/api/health")
def api_health() -> dict:
    return {"ok": True, "service": "screen-recorder-pc-web"}


@app.get("/api/status")
def api_status() -> dict:
    return {"ok": True, **_controller.get_status()}


@app.get("/api/logs")
def api_logs(since: int = 0) -> dict:
    return {"ok": True, "logs": _controller.logs_since(since)}


@app.post("/api/record/toggle")
def api_record_toggle() -> dict:
    return _controller.toggle_record()


@app.post("/api/receiver/start")
def api_receiver_start() -> dict:
    return _controller.start_receiver()


@app.post("/api/receiver/stop")
def api_receiver_stop() -> dict:
    return _controller.stop_receiver()


@app.get("/api/recordings")
def api_recordings() -> dict:
    return {"ok": True, "items": _controller.list_recordings()}


@app.post("/api/video/select")
def api_video_select(body: SelectVideoRequest) -> dict:
    r = _controller.set_selected_video(body.path)
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error", "bad request"))
    return r


@app.post("/api/analyze/start")
def api_analyze_start(body: AnalyzeRequest) -> dict:
    r = _controller.start_analysis(body.backend, body.api_key, body.model_id)
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error", "bad request"))
    return r


@app.post("/api/model/deploy")
def api_model_deploy(body: DeployRequest) -> dict:
    r = _controller.deploy_model(body.model_id)
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error", "bad request"))
    return r


@app.get("/api/memory/stats")
def api_memory_stats() -> dict:
    return _controller.memory_stats()


@app.post("/api/memory/search")
def api_memory_search(body: MemorySearchRequest) -> dict:
    r = _controller.memory_search(body.query, limit=body.limit)
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error", "bad request"))
    return r


@app.get("/api/memory/profile")
def api_memory_profile() -> dict:
    return _controller.memory_profile()


@app.post("/api/system/open-folder")
def api_open_folder(body: OpenFolderRequest) -> dict:
    r = _controller.open_folder(body.kind)
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error", "bad request"))
    return r


# ── Proactive Agent integration ─────────────────────────────────────────────

@app.get("/api/proactive/latest-recording")
def api_proactive_latest_recording() -> dict:
    return _controller.get_latest_recording_meta()


@app.get("/api/proactive/gui-report")
def api_proactive_gui_report(video_path: str = "") -> dict:
    r = _controller.load_gui_report(video_path)
    if not r.get("ok"):
        raise HTTPException(status_code=404, detail=r.get("error", "report not found"))
    return r


@app.post("/api/proactive/prepare")
def api_proactive_prepare(body: ProactivePrepareRequest) -> dict:
    if body.wait:
        r = _controller.prepare_for_proactive(
            video_path=body.video_path,
            auto_analyze=body.auto_analyze,
            backend=body.backend,
            api_key=body.api_key,
            model_id=body.model_id,
        )
    else:
        latest = _controller.get_latest_recording_meta()
        if not latest.get("ok") and not body.video_path.strip():
            raise HTTPException(status_code=404, detail=latest.get("error", "no recording"))
        path = body.video_path.strip() or latest["recording"]["path"]
        _controller.set_selected_video(path)
        if body.auto_analyze:
            start = _controller.start_analysis(body.backend, body.api_key, body.model_id)
            if not start.get("ok"):
                raise HTTPException(status_code=400, detail=start.get("error", "analyze failed"))
        r = {
            "ok": True,
            "video_path": path,
            "recording": latest.get("recording") if latest.get("ok") else {"path": path},
            "report": None,
            "started_analysis": body.auto_analyze,
        }
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error", "prepare failed"))
    return r


def _mount_static() -> None:
    if os.path.isdir(_APP_ASSETS):
        app.mount("/app-assets", StaticFiles(directory=_APP_ASSETS), name="app-assets")
    built = os.path.join(_WEBUI_DIST, "assets")
    if os.path.isdir(built):
        app.mount("/assets", StaticFiles(directory=built), name="assets")


_mount_static()


@app.get("/", response_class=HTMLResponse)
def spa_index() -> HTMLResponse:
    if os.path.isfile(_INDEX_HTML):
        with open(_INDEX_HTML, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse(
        "<html><body style='font-family:sans-serif;padding:24px'>"
        "<h2>Web UI 未构建</h2>"
        "<p>请在 <code>screen-recorder-mvp/pc/webui</code> 执行 "
        "<code>npm install && npm run build</code> 后重启。</p>"
        "<p>API 仍可用：<a href='/docs'>/docs</a></p>"
        "</body></html>",
        status_code=200,
    )


@app.get("/logo.svg")
def logo_svg() -> FileResponse:
    if os.path.isfile(_LOGO_SVG):
        return FileResponse(_LOGO_SVG, media_type="image/svg+xml")
    raise HTTPException(status_code=404)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    ico = os.path.join(_PC_DIR, "icon.ico")
    if os.path.isfile(ico):
        return FileResponse(ico)
    raise HTTPException(status_code=404)


def run_server(host: str = "127.0.0.1", port: int | None = None) -> None:
    import uvicorn

    p = port if port is not None else WEB_UI_PORT
    uvicorn.run(app, host=host, port=p, log_level="info")
