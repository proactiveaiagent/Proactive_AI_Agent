# -*- coding: utf-8 -*-
"""接收来自手机上传的录制视频 - Flask 简单 HTTP 服务"""
import os
from flask import Flask, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

UPLOAD_FOLDER = "recordings"
ALLOWED_EXTENSIONS = {"mp4", "webm", "mkv"}

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def upload_file():
    app.logger.info(
        "upload request: content_length=%s content_type=%s files=%s form=%s",
        request.content_length,
        request.content_type,
        list(request.files.keys()),
        list(request.form.keys()),
    )
    if "file" not in request.files and "video" not in request.files:
        return {"ok": False, "error": "没有文件"}, 400
    file = request.files.get("file") or request.files.get("video")
    if file.filename == "":
        return {"ok": False, "error": "未选择文件"}, 400
    if file and allowed_file(file.filename):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        return {"ok": True, "path": filepath}
    return {"ok": False, "error": "不允许的文件类型"}, 400


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_err):
    return {"ok": False, "error": "文件过大，超过 500MB 限制"}, 413


@app.errorhandler(BadRequest)
def handle_bad_request(err):
    # Typical cases: malformed multipart body / body truncated before boundary.
    app.logger.warning("bad request while upload: %s", err.description)
    return {"ok": False, "error": f"请求格式错误: {err.description}"}, 400


@app.route("/health", methods=["GET"])
def health():
    return {"ok": True, "service": "screen-recorder-receiver"}


def run_receiver(host="0.0.0.0", port=8765):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host=host, port=port, threaded=True, use_reloader=False)
