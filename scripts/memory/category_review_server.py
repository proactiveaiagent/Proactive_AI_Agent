#!/usr/bin/env python3
"""Serve a localhost-only blind annotation UI for one reviewer CSV packet."""
import argparse
import csv
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
EGOLIFE_ROOT = (ROOT / "data/egolife").resolve()
LABEL_COLUMNS = {
    "people": "people_json",
    "locations": "locations_json",
    "activity_events": "activity_events_json",
    "environments": "environments_json",
    "objects": "objects_json",
}
REQUIRED_COLUMNS = {
    "sample_id", "video_path", "transcripts_json", "annotation_status",
    "reviewer_id", "confidence", "reviewer_notes", *LABEL_COLUMNS.values(),
}
MAX_BODY_BYTES = 128 * 1024

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EgoLife 盲审工作台</title>
<style>
:root{font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif;color:#182230;background:#f4f6f9}
*{box-sizing:border-box}body{margin:0}header{position:sticky;top:0;background:#16263c;color:white;padding:12px 20px;z-index:2;display:flex;gap:14px;align-items:center}
header strong{font-size:18px}#progress{margin-left:auto}.wrap{max-width:1500px;margin:auto;padding:16px;display:grid;grid-template-columns:minmax(380px,1.05fr) minmax(360px,.95fr);gap:16px}
.card{background:white;border:1px solid #d9e0e8;border-radius:10px;padding:16px;box-shadow:0 2px 8px #16263c0b}
.row{display:flex;gap:8px;align-items:center}.row>*{min-width:0}.grow{flex:1}button,select,textarea{font:inherit}button,select{padding:8px 10px;border:1px solid #bbc5d2;border-radius:6px;background:white}button{cursor:pointer}button.primary{background:#145dcc;color:white;border-color:#145dcc}button.done{background:#14804a;color:white;border-color:#14804a}
video{display:block;width:100%;max-height:55vh;background:#111;border-radius:7px;margin:12px 0}.transcript{white-space:pre-wrap;max-height:180px;overflow:auto;background:#f6f8fa;border-radius:6px;padding:10px}
.field{margin:12px 0}.field label{font-weight:650;display:block;margin-bottom:4px}.field small{color:#566579}.field textarea{display:block;width:100%;min-height:58px;resize:vertical;border:1px solid #bbc5d2;border-radius:6px;padding:8px}
#notice{min-height:24px;color:#145dcc}.muted{color:#59697c}.warning{color:#a44500}
@media(max-width:850px){.wrap{grid-template-columns:1fr;padding:8px}video{max-height:40vh}header{padding:10px}}
</style></head><body>
<header><strong>EgoLife 盲审标注工作台</strong><span id="reviewer"></span><span id="progress"></span></header>
<div class="wrap">
<section class="card">
<div class="row"><button id="prev">上一个</button><button id="nextPending">下一个待标</button><button id="next">下一个</button>
<select id="sample" class="grow" aria-label="选择片段"></select></div>
<div id="meta" class="muted" style="margin-top:8px"></div>
<video id="video" controls preload="metadata"></video>
<h3>语音转写（允许参考）</h3><div id="transcript" class="transcript">加载中…</div>
<p class="muted">请依据视频内容标注；不要参考旧自动标签或模型预测。多标签每行一个英文标签；确认无可信标签时留空。</p>
</section>
<section class="card">
<div class="field"><label>人物 people_json</label><small>可识别的人名；无法确认身份时不要猜。</small><textarea id="people" placeholder="每行一个名字；无标签留空"></textarea></div>
<div class="field"><label>地点 locations_json</label><small>功能性地点，如 kitchen、office、yard。</small><textarea id="locations" placeholder="每行一个地点；无标签留空"></textarea></div>
<div class="field"><label>主要活动 activity_events_json（恰好一个）</label><small>选持续时间最长或最关键的活动；词表见标注规范，无法匹配时填 other:简短英文短语。</small><textarea id="activity_events" placeholder="例如：meeting"></textarea></div>
<div class="field"><label>环境 environments_json</label><small>例如 indoor、outdoor、natural_light、dim、quiet；不要填地点名或物品。</small><textarea id="environments" placeholder="每行一个环境标签；无标签留空"></textarea></div>
<div class="field"><label>物品 objects_json</label><small>与活动显著相关的具体物品，使用单数英文名词。</small><textarea id="objects" placeholder="例如：phone"></textarea></div>
<div class="row"><label for="confidence">置信度</label><select id="confidence"><option value="high">high</option><option value="medium" selected>medium</option><option value="low">low</option></select>
<span class="grow"></span><span class="muted">reviewer_id 由服务端设置</span></div>
<div class="field"><label>备注 / 不确定原因</label><textarea id="notes" placeholder="可选；不可判定或片段问题请说明"></textarea></div>
<div id="notice"></div><div class="row"><button id="save">保存草稿</button><button id="mark" class="done">保存并标为 reviewed</button><button id="exclude">标记不可判定</button></div>
</section></div>
<script>
const fieldNames=["people","locations","activity_events","environments","objects"];
let state=null,current=null;
const $=id=>document.getElementById(id);
async function req(url,options){const r=await fetch(url,options);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
function labels(text){return text.split(/\r?\n/).map(x=>x.trim()).filter(Boolean)}
function setTextarea(key,values){$(key).value=(values||[]).join("\n")}
function updateProgress(){let done=state.items.filter(x=>x.annotation_status==="reviewed").length;let excluded=state.items.filter(x=>x.annotation_status==="excluded").length;$("progress").textContent="已完成 "+done+"/"+state.total_samples+" · 排除 "+excluded+" · 当前 "+(current?.sample_id||"")}
async function loadState(){
 state=await req("/api/state");$("reviewer").textContent="标注者："+state.reviewer_id;
 const select=$("sample");select.innerHTML="";
 for(const item of state.items){const op=document.createElement("option");op.value=item.sample_id;op.textContent=(item.annotation_status==="reviewed"?"✓":item.annotation_status==="excluded"?"×":"○")+" "+item.sample_id;select.appendChild(op)}
 if(!state.items.length)throw Error("标注包没有样本");
 await show(state.items[0].sample_id);
}
async function show(id){
 const item=await req("/api/sample?sample_id="+encodeURIComponent(id));current=item;
 $("sample").value=id;$("meta").textContent=item.sample_id+" · "+item.start_seconds+"s–"+item.end_seconds+"s · "+item.annotation_status;
 $("video").src="/api/video?sample_id="+encodeURIComponent(id);$("video").load();
 $("transcript").textContent=Array.isArray(item.transcripts)?item.transcripts.join("\n"):"";
 for(const key of fieldNames)setTextarea(key,item.labels[key]);
 $("confidence").value=item.confidence||"medium";$("notes").value=item.reviewer_notes||"";
 $("notice").textContent="";updateProgress();
}
function draft(status){
 const fields={};for(const k of fieldNames)fields[k]=labels($(k).value);
 return {sample_id:current.sample_id,status,confidence:$("confidence").value,reviewer_notes:$("notes").value,fields};
}
async function save(status){
 try{
  $("notice").textContent="保存中…";
  await req("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(draft(status))});
  state=await req("/api/state");
  for(const op of $("sample").options){const row=state.items.find(x=>x.sample_id===op.value);op.textContent=(row.annotation_status==="reviewed"?"✓":row.annotation_status==="excluded"?"×":"○")+" "+row.sample_id}
  $("notice").textContent=status==="reviewed"?"已保存并标记 reviewed。":status==="excluded"?"已标记 excluded；请保留具体原因。":"草稿已保存。";
  updateProgress();
 }catch(e){$("notice").textContent=e.message;$("notice").className="warning"}
}
function move(offset,onlyPending=false){
 const items=state.items;let i=items.findIndex(x=>x.sample_id===current.sample_id);
 for(let step=1;step<=items.length;step++){const j=(i+offset*step+items.length)%items.length;if(!onlyPending||!["reviewed","excluded"].includes(items[j].annotation_status)){show(items[j].sample_id);return}}
 $("notice").textContent="没有其他待标样本。";
}
$("sample").addEventListener("change",e=>show(e.target.value));
$("prev").onclick=()=>move(-1);$("next").onclick=()=>move(1);$("nextPending").onclick=()=>move(1,true);
$("save").onclick=()=>save("pending");$("mark").onclick=()=>save("reviewed");$("exclude").onclick=()=>save("excluded");
loadState().catch(e=>{$("notice").textContent=e.message;$("notice").className="warning"});
</script></body></html>"""


class ReviewStore:
    def __init__(self, packet, reviewer_id):
        self.packet = packet.resolve()
        self.reviewer_id = reviewer_id
        self.lock = threading.RLock()
        with self.packet.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.columns = reader.fieldnames or []
            self.rows = list(reader)
        missing = REQUIRED_COLUMNS - set(self.columns)
        if missing:
            raise ValueError(f"Packet missing columns: {sorted(missing)}")
        self.by_id = {row.get("sample_id", ""): row for row in self.rows}
        if not self.rows or "" in self.by_id or len(self.by_id) != len(self.rows):
            raise ValueError("Packet must have unique, non-empty sample_id values")

    def state(self):
        with self.lock:
            items = [{"sample_id": row["sample_id"], "annotation_status": row.get("annotation_status", "pending")}
                     for row in self.rows]
            completed = sum(item["annotation_status"] == "reviewed" for item in items)
            excluded = sum(item["annotation_status"] == "excluded" for item in items)
            return {"reviewer_id": self.reviewer_id, "total_samples": len(items),
                    "reviewed_samples": completed, "excluded_samples": excluded, "items": items}

    def sample(self, sample_id):
        with self.lock:
            row = self.by_id.get(sample_id)
            if row is None:
                raise KeyError("Unknown sample_id")
            try:
                transcripts = json.loads(row.get("transcripts_json") or "[]")
            except json.JSONDecodeError:
                transcripts = [row.get("transcripts_json", "")]
            labels = {}
            for name, column in LABEL_COLUMNS.items():
                try:
                    labels[name] = json.loads(row.get(column) or "[]")
                except json.JSONDecodeError:
                    labels[name] = []
            return {
                "sample_id": sample_id,
                "start_seconds": row.get("start_seconds", ""),
                "end_seconds": row.get("end_seconds", ""),
                "transcripts": transcripts if isinstance(transcripts, list) else [str(transcripts)],
                "labels": labels,
                "annotation_status": row.get("annotation_status", "pending"),
                "confidence": row.get("confidence", ""),
                "reviewer_notes": row.get("reviewer_notes", ""),
            }

    def video_path(self, sample_id):
        with self.lock:
            row = self.by_id.get(sample_id)
            if row is None:
                raise KeyError("Unknown sample_id")
            value = Path(row.get("video_path", ""))
            path = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
        if EGOLIFE_ROOT not in path.parents or not path.is_file():
            raise FileNotFoundError("Video is not available in the EgoLife data directory")
        return path

    def save(self, payload):
        sample_id = payload.get("sample_id")
        status = payload.get("status")
        if not isinstance(sample_id, str) or sample_id not in self.by_id:
            raise ValueError("Unknown sample_id")
        if status not in {"pending", "reviewed", "excluded"}:
            raise ValueError("status must be pending, reviewed, or excluded")
        fields = payload.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("fields must be a JSON object")
        parsed = {}
        for name in LABEL_COLUMNS:
            values = fields.get(name)
            if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{name} must be an array of non-empty strings")
            parsed[name] = [v.strip() for v in values]
        confidence = payload.get("confidence", "")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")
        if status == "reviewed" and len(parsed["activity_events"]) != 1:
            raise ValueError("activity_events must contain exactly one primary label before review is complete")
        notes = payload.get("reviewer_notes", "")
        if not isinstance(notes, str):
            raise ValueError("reviewer_notes must be text")
        if status == "excluded" and not notes.strip():
            raise ValueError("An exclusion reason is required in reviewer_notes")

        with self.lock:
            row = self.by_id[sample_id]
            row["annotation_status"] = status
            row["reviewer_id"] = self.reviewer_id
            row["confidence"] = confidence
            row["reviewer_notes"] = notes
            for name, column in LABEL_COLUMNS.items():
                row[column] = json.dumps(parsed[name], ensure_ascii=False)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8-sig", newline="", dir=self.packet.parent,
                    prefix=self.packet.name + ".", suffix=".tmp", delete=False
                ) as stream:
                    temp_path = Path(stream.name)
                    writer = csv.DictWriter(stream, fieldnames=self.columns, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(self.rows)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_path, self.packet)
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()
        return self.state()


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    data = PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(data)
                elif parsed.path == "/api/state":
                    self.send_json(200, store.state())
                elif parsed.path == "/api/sample":
                    self.send_json(200, store.sample(query.get("sample_id", [""])[0]))
                elif parsed.path == "/api/video":
                    self.send_video(store.video_path(query.get("sample_id", [""])[0]))
                else:
                    self.send_json(404, {"error": "Not found"})
            except KeyError as exc:
                self.send_json(404, {"error": str(exc).strip("'")})
            except FileNotFoundError as exc:
                self.send_json(404, {"error": str(exc)})
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})

        def do_HEAD(self):
            self.do_GET()

        def send_video(self, path):
            size = path.stat().st_size
            start, end, partial = 0, size - 1, False
            header = self.headers.get("Range", "")
            if header:
                if not header.startswith("bytes=") or "," in header:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                try:
                    left, right = header[6:].split("-", 1)
                    if left:
                        start = int(left)
                        end = int(right) if right else size - 1
                    else:
                        suffix = int(right)
                        start = max(0, size - suffix)
                    if start < 0 or end < start or start >= size:
                        raise ValueError
                    end = min(end, size - 1)
                    partial = True
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(64 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            if urlparse(self.path).path != "/api/save":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError("Request body must be between 1 byte and 128 KiB")
                payload = json.loads(self.rfile.read(length))
                result = store.save(payload)
                self.send_json(200, result)
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.log_date_time_string(), fmt % args), flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True, help="One reviewer CSV packet")
    parser.add_argument("--reviewer-id", required=True, help="Stable reviewer identifier stored with labels")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host; defaults to localhost only")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("For safety, bind only to loopback; expose access through an SSH tunnel")
    store = ReviewStore(args.packet, args.reviewer_id)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    server.daemon_threads = True
    print(f"Reviewer {args.reviewer_id}: http://{args.host}:{args.port} "
          f"({store.state()['total_samples']} samples; packet {store.packet})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping review server", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
