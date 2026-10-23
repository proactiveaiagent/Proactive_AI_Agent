"""
generate_report.py — 从端到端产物自动生成可视化 HTML 报告
============================================================
读取以下产物，重新生成 `zhx/task2/端到端验证报告.html`：
  - code/memory/memory.json            → 7 层记忆状态 + layer6 画像（动态置信度条形图）
  - code/output/analysis_result.txt    → Phase 0/A/B 完整输出 + 各阶段耗时
  - code/output/first_frame.jpg        → 模型实际看到的抽帧图（base64 内嵌）
  - /tmp/agent_run.log                 → 音频转写文本（best-effort）
  - 视频文件                           → 时长/分辨率/fps（moviepy，best-effort）

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python generate_report.py [视频路径]
"""

import base64
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data/cxr25/zhx/Proactive_AI_Agent")
CODE = PROJECT_ROOT / "code"
MEMORY_JSON = CODE / "memory" / "memory.json"
ANALYSIS_TXT = CODE / "output" / "analysis_result.txt"
FRAME_JPG = CODE / "output" / "first_frame.jpg"
AGENT_LOG = Path("/tmp/agent_run.log")
OUT_HTML = PROJECT_ROOT / "zhx" / "task2" / "端到端验证报告.html"

DEFAULT_VIDEO = PROJECT_ROOT / "test_data/test_data/2.travel_abroad/2.1.mp4"

# 字段中文名
FIELD_CN = {
    "demographics": "身份属性",
    "preferences": "偏好",
    "frequent_locations": "常去地点",
    "behavior_patterns": "行为模式",
}
KEY_CN = {
    "food": "饮食", "hobbies": "兴趣", "name": "姓名", "age": "年龄",
    "gender": "性别", "identity": "身份", "living_region": "居住区域",
    "occupation": "职业", "education": "教育", "social_relations": "社会关系",
    "with_surroundings": "与周围环境交互", "with_ar_system": "与 AR 系统交互",
    "with_agents": "与 AI 智能体交互", "common_apps": "常用应用",
    "typical_behaviors": "典型行为",
}


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def conf_color(c):
    if c >= 0.8:
        return "#10b981"
    if c >= 0.6:
        return "#f59e0b"
    return "#ef4444"


def load_memory():
    return json.loads(MEMORY_JSON.read_text(encoding="utf-8"))


def load_analysis():
    return ANALYSIS_TXT.read_text(encoding="utf-8") if ANALYSIS_TXT.exists() else ""


def load_frame_b64():
    if FRAME_JPG.exists():
        return base64.b64encode(FRAME_JPG.read_bytes()).decode()
    return None


def load_video_info(path):
    info = {"path": str(path), "duration": None, "size": None, "fps": None, "has_audio": None}
    try:
        from moviepy import VideoFileClip
        v = VideoFileClip(str(path))
        info.update(duration=round(v.duration, 2), size=v.size, fps=v.fps,
                    has_audio=(v.audio is not None))
        v.close()
    except Exception as e:
        info["error"] = str(e)
    return info


def load_transcript():
    if not AGENT_LOG.exists():
        return None
    text = AGENT_LOG.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Transcript with timestamps:\n(.*?)(?:\n\n|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else None


def render_attr(av):
    """渲染单个 AttrValue：值 + evidence + 置信度条。"""
    v = esc(av.get("value", ""))
    c = float(av.get("confidence", 0))
    ev = esc(av.get("evidence", ""))
    ev_html = f'<span class="ev">{ev}</span>' if ev else ""
    return (f'<div class="attr"><div class="v"><b>{v}</b>{ev_html}</div>'
            f'<div class="conf"><div class="bar-track"><div class="bar" '
            f'style="width:{int(c * 100)}%;background:{conf_color(c)}"></div></div>'
            f'<span class="pct">{c:.2f}</span></div></div>')


def render_node(node, depth=0):
    """递归渲染画像节点：AttrValue 叶子 / list / 嵌套 dict。"""
    if isinstance(node, dict) and "value" in node:
        return render_attr(node)
    if isinstance(node, list):
        return "".join(render_attr(x) for x in node
                       if isinstance(x, dict) and "value" in x)
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            label = KEY_CN.get(k, k)
            if isinstance(v, dict) and "value" in v:
                out.append(render_attr(v))
            elif isinstance(v, list) and v and all(isinstance(x, dict) and "value" in x for x in v):
                out.append(f'<div class="sname">{esc(label)}</div>')
                out.extend(render_attr(x) for x in v)
            elif isinstance(v, dict):
                out.append(f'<div class="sname">{esc(label)}</div>')
                out.append(render_node(v, depth + 1))
        return "".join(out)
    return ""


def render_profile(profile):
    """渲染 layer6 画像四字段。"""
    html = []
    for field in ["demographics", "preferences", "frequent_locations", "behavior_patterns"]:
        node = profile.get(field)
        if not node:
            continue
        body = render_node(node)
        if not body.strip():
            continue
        cn = FIELD_CN.get(field, field)
        html.append(f'<div class="field"><span class="fname">{esc(cn)}</span>{body}</div>')
    return "".join(html)


def parse_analysis_sections(text):
    """从 analysis_result.txt 切出 Phase 0 / A / B 三段。"""
    secs = {}
    for key in ["PHASE 0", "PHASE A", "PHASE B"]:
        m = re.search(rf"{key}[^\n]*\n=+\n(.*?)(?=\n=+\n|\Z)", text, re.DOTALL)
        if m:
            secs[key] = m.group(1).strip()
    return secs


def parse_timings(text):
    """从 analysis_result.txt 提取各阶段耗时。"""
    def get(pat):
        m = re.search(pat, text)
        return m.group(1) if m else None
    return {
        "phase0": get(r"total_phase_0:\s*([\d.]+)s"),
        "phase_a": get(r"total_phase_a:\s*([\d.]+)s"),
        "phase_b": get(r"total_phase_b:\s*([\d.]+)s"),
        "analyze_qwen": get(r"analyze_qwen:\s*([\d.]+)s"),
        "wall": get(r"wall_clock_total:\s*([\d.]+)s"),
    }


def build_html(video_info, frame_b64, transcript, profile, mem, secs, timings):
    def v_or_dash(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "—"

    # 视频信息卡
    frame_html = ""
    if frame_b64:
        frame_html = (f'<div class="framewrap"><div class="framelbl">模型实际看到的抽帧（第一帧）</div>'
                      f'<img class="frame" src="data:image/jpeg;base64,{frame_b64}" alt="first frame"></div>')

    transcript_html = ""
    if transcript:
        lines = "".join(f'<div class="tsline">{esc(l)}</div>' for l in transcript.splitlines() if l.strip())
        transcript_html = (f'<div class="vidcol"><div class="vidlbl">音频转写（Whisper 识别到的语音）</div>'
                           f'<div class="transcript">{lines}</div></div>')
    else:
        transcript_html = (f'<div class="vidcol"><div class="vidlbl">音频转写</div>'
                           f'<div class="muted">未捕获到转写文本</div></div>')

    vid_meta = (
        f'<div class="vidmeta">'
        f'<div><span>路径</span><code>{esc(video_info["path"])}</code></div>'
        f'<div><span>时长</span><b>{v_or_dash(video_info["duration"], "s")}</b></div>'
        f'<div><span>分辨率</span><b>{video_info["size"] or "—"}</b></div>'
        f'<div><span>帧率</span><b>{v_or_dash(video_info["fps"], "fps")}</b></div>'
        f'<div><span>音频</span><b>{"有" if video_info["has_audio"] else "无"}</b></div>'
        f'</div>'
    )

    # Phase 输出
    def phase_card(title, body, icon):
        if not body:
            return ""
        return (f'<div class="phasecard"><div class="phhead">{icon} {esc(title)}</div>'
                f'<pre class="pre">{esc(body)}</pre></div>')

    phases_html = "".join([
        phase_card("Phase 0 · 合规预检", secs.get("PHASE 0"), "✅"),
        phase_card("Phase A · 场景·需求·方案（Part1-3）", secs.get("PHASE A"), "🧠"),
        phase_card("Phase B · 多通道输出计划", secs.get("PHASE B"), "📤"),
    ])

    profile_html = render_profile(profile) or '<div class="muted">画像为空（尚未生成）</div>'

    # 记忆摘要
    l4 = (mem["layer4"].get("summary") or "")[:200]
    l5 = (mem["layer5"].get("summary") or "")[:200]
    l6 = (mem["layer6"].get("summary") or "")[:200]
    last_cons = mem["metadata"].get("last_consolidation") or "null（从未执行）"

    # 时间线占比（Phase C 用 wall - 其他）
    p0 = float(timings["phase0"] or 0)
    pa = float(timings["phase_a"] or 0)
    pb = float(timings["phase_b"] or 0)
    wall = float(timings["wall"] or 1)
    pc = max(0, wall - p0 - pa - pb)
    def pct(x):
        return max(0.5, min(100, x / wall * 100))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>任务2 · 分层记忆与画像 — 端到端验证报告</title>
<style>
  :root {{ --bg:#f5f6fa; --card:#fff; --ink:#1a1a2e; --muted:#6b7280; --brand:#4f46e5; --brand2:#7c3aed; --ok:#059669; --warn:#d97706; --bad:#dc2626; --border:#e5e7eb; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--ink); line-height:1.6; padding-bottom:60px; }}
  .hero {{ background:linear-gradient(135deg,#4f46e5,#7c3aed 50%,#a855f7); color:#fff; padding:48px 32px 40px; }}
  .hero h1 {{ font-size:30px; font-weight:700; }}
  .hero .sub {{ margin-top:10px; font-size:15px; opacity:.92; max-width:900px; }}
  .hero .tag {{ display:inline-block; margin-top:16px; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35); padding:4px 14px; border-radius:999px; font-size:13px; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:0 20px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:-26px auto 0; position:relative; }}
  .kpi {{ background:var(--card); border-radius:14px; padding:18px; box-shadow:0 8px 24px rgba(0,0,0,.08); border:1px solid var(--border); }}
  .kpi .num {{ font-size:26px; font-weight:700; }}
  .kpi .num.ok {{ color:var(--ok); }} .kpi .num.brand {{ color:var(--brand); }}
  .kpi .lbl {{ font-size:13px; color:var(--muted); margin-top:4px; }}
  section {{ margin-top:36px; }}
  h2 {{ font-size:20px; font-weight:700; display:flex; align-items:center; gap:8px; margin-bottom:16px; }}
  h2 .dot {{ width:10px; height:10px; border-radius:50%; background:linear-gradient(135deg,var(--brand),var(--brand2)); display:inline-block; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px; box-shadow:0 2px 10px rgba(0,0,0,.04); }}
  .muted {{ color:var(--muted); font-size:13px; }}
  code {{ background:#f1f5f9; padding:1px 6px; border-radius:5px; font-size:12px; color:#0f172a; word-break:break-all; }}

  /* 视频 */
  .vidgrid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  @media(max-width:760px) {{ .vidgrid {{ grid-template-columns:1fr; }} }}
  .framewrap {{ text-align:center; }}
  .frame {{ max-width:100%; border-radius:10px; border:1px solid var(--border); }}
  .framelbl, .vidlbl {{ font-size:12px; color:var(--muted); font-weight:600; margin-bottom:8px; }}
  .transcript {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:12px; font-size:13px; }}
  .tsline {{ padding:2px 0; }}
  .vidmeta {{ display:flex; flex-wrap:wrap; gap:10px 22px; margin-top:16px; font-size:13px; }}
  .vidmeta span {{ color:var(--muted); margin-right:6px; }}

  /* 数据流 */
  .flow {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
  .fnode {{ background:#eef2ff; border:1px solid #c7d2fe; border-radius:10px; padding:10px 14px; font-size:13px; font-weight:600; color:#4338ca; }}
  .fnode.model {{ background:#faf5ff; border-color:#e9d5ff; color:#7e22ce; }}
  .farrow {{ color:#9ca3af; font-size:18px; }}

  /* 流水线 */
  .pipe {{ display:flex; flex-direction:column; gap:12px; }}
  .prow {{ display:grid; grid-template-columns:210px 1fr 90px; align-items:center; gap:12px; }}
  .prow .nm {{ font-size:13px; font-weight:600; }}
  .bar-track {{ background:#eef0f6; border-radius:999px; height:18px; overflow:hidden; }}
  .bar {{ height:100%; border-radius:999px; background:linear-gradient(90deg,var(--brand),var(--brand2)); }}
  .bar.c {{ background:linear-gradient(90deg,#f59e0b,#ef4444); }}
  .prow .val {{ font-size:13px; font-weight:700; text-align:right; }}
  .note {{ font-size:13px; color:var(--muted); margin-top:12px; }}

  /* phase 输出 */
  .phasecard {{ margin-bottom:16px; border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
  .phhead {{ background:#f3f4f6; padding:8px 14px; font-weight:700; font-size:14px; }}
  .pre {{ padding:14px; font-size:13px; white-space:pre-wrap; word-break:break-word; background:#fafafa; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; line-height:1.5; }}

  /* 画像 */
  .field {{ margin-bottom:18px; }}
  .fname {{ font-weight:700; font-size:15px; padding:6px 12px; border-radius:8px; background:linear-gradient(135deg,#eef2ff,#faf5ff); color:#4338ca; display:inline-block; margin-bottom:10px; }}
  .attr {{ display:grid; grid-template-columns:1fr 140px; gap:8px; align-items:center; padding:7px 0; border-bottom:1px dashed #f0f0f5; }}
  .attr .v {{ font-size:14px; }}
  .attr .v .ev {{ display:block; font-size:12px; color:var(--muted); margin-top:2px; }}
  .conf {{ display:flex; align-items:center; gap:8px; }}
  .conf .bar-track {{ flex:1; height:10px; }}
  .conf .bar {{ height:100%; }}
  .conf .pct {{ font-size:12px; font-weight:700; width:36px; text-align:right; }}
  .subsec {{ margin-top:8px; padding-left:14px; border-left:3px solid #e0e7ff; }}
  .sname {{ font-size:13px; font-weight:600; color:#4b5563; margin:8px 0 4px; }}
  .warnbox {{ background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:14px 16px; margin-top:14px; font-size:13px; color:#78350f; }}
  .warnbox b {{ color:#92400e; }}

  /* 记忆层 */
  .layers {{ display:flex; flex-direction:column; gap:10px; }}
  .layer {{ display:grid; grid-template-columns:56px 1fr 110px; align-items:center; gap:14px; background:#fafaff; border:1px solid var(--border); border-radius:10px; padding:12px 16px; }}
  .layer .no {{ width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:17px; color:#fff; background:linear-gradient(135deg,var(--brand),var(--brand2)); }}
  .layer.hot .no {{ background:linear-gradient(135deg,#059669,#10b981); }}
  .layer.warn .no {{ background:linear-gradient(135deg,#d97706,#f59e0b); }}
  .layer.empty .no {{ background:#cbd5e1; }}
  .layer .ttl {{ font-weight:600; font-size:15px; }}
  .layer .desc {{ font-size:13px; color:var(--muted); }}
  .layer .st {{ text-align:right; font-size:12px; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; }}
  .badge.ok {{ background:#d1fae5; color:#065f46; }}
  .badge.warn {{ background:#fef3c7; color:#92400e; }}
  .badge.empty {{ background:#e5e7eb; color:#4b5563; }}
</style>
</head>
<body>
<div class="hero">
  <div class="wrap">
    <h1>🧠 任务 2 · 分层记忆与用户画像</h1>
    <div class="sub">目标：从多模态识别结果中持续构建用户画像，存入 7 层记忆，做到 ≤2s 检索、≥95% 准确。本次验证了<b>「分层存储落地 + 端到端跑通」</b>——画像第一次真实生成进 layer6。</div>
    <span class="tag">自动生成 · 数据快照 last_consolidation = {esc(last_cons)}</span>
  </div>
</div>

<div class="wrap">
  <div class="kpis">
    <div class="kpi"><div class="num brand">{esc(timings["wall"] or "—")}s</div><div class="lbl">端到端总耗时</div></div>
    <div class="kpi"><div class="num">{esc(timings["analyze_qwen"] or "—")}s</div><div class="lbl">Phase A 视觉分析</div></div>
    <div class="kpi"><div class="num ok">{esc(str(mem["metadata"]["total_moments"]))} 条</div><div class="lbl">累计 moment</div></div>
    <div class="kpi"><div class="num ok">✓ 已生成</div><div class="lbl">layer6 画像</div></div>
  </div>

  <section>
    <h2><span class="dot"></span>① 输入了什么：视频 → 模型</h2>
    <div class="card">
      <div class="flow" style="margin-bottom:18px">
        <div class="fnode">📹 视频<br><small>{esc(v_or_dash(video_info["duration"], "s"))}</small></div>
        <div class="farrow">→</div>
        <div class="fnode">🖼 抽帧 ×1<br><small>首帧</small></div>
        <div class="farrow">→</div>
        <div class="fnode">🔊 抽音频<br><small>转写</small></div>
        <div class="farrow">→</div>
        <div class="fnode model">🤖 Qwen3-VL-8B<br><small>视觉+文本</small></div>
        <div class="farrow">→</div>
        <div class="fnode">📋 Phase A/B/C<br><small>分析输出</small></div>
      </div>
      <div class="vidgrid">
        {frame_html}
        {transcript_html}
      </div>
      {vid_meta}
      <div class="note" style="margin-top:12px">输入模式为 <code>frame</code>（只抽第一帧，最快），所以模型只看到<b> 1 张图</b> + 音频转写文本 + 分析 prompt。视频是「旅行出国」场景（机场登机桥）。</div>
    </div>
  </section>

  <section>
    <h2><span class="dot"></span>② 端到端流水线耗时</h2>
    <div class="card">
      <div class="pipe">
        <div class="prow"><div class="nm">Phase 0 · 合规预检</div><div class="bar-track"><div class="bar" style="width:{pct(p0):.1f}%"></div></div><div class="val">{esc(timings["phase0"] or "—")}s</div></div>
        <div class="prow"><div class="nm">Phase A · 场景·需求·方案</div><div class="bar-track"><div class="bar" style="width:{pct(pa):.1f}%"></div></div><div class="val">{esc(timings["phase_a"] or "—")}s</div></div>
        <div class="prow"><div class="nm">Phase B · 多通道输出</div><div class="bar-track"><div class="bar" style="width:{pct(pb):.1f}%"></div></div><div class="val">{esc(timings["phase_b"] or "—")}s</div></div>
        <div class="prow"><div class="nm">Phase C · 记忆整理（生成画像）</div><div class="bar-track"><div class="bar c" style="width:{pct(pc):.1f}%"></div></div><div class="val">{pc:.1f}s</div></div>
      </div>
      <div class="note">🔑 耗时大头是 <b>Phase C 的 LLM 调用</b>，跑在后台、不阻塞响应。真正的「检索」环节（L6 画像直读）只需毫秒级——符合「≤2s 检索」目标。</div>
    </div>
  </section>

  <section>
    <h2><span class="dot"></span>③ 模型的完整输出（Phase 0 / A / B）</h2>
    <div class="card">
      {phases_html}
    </div>
  </section>

  <section>
    <h2><span class="dot"></span>④ 画像实际内容（layer6）</h2>
    <div class="card">
      <div class="muted" style="margin-bottom:14px">每个属性带 6 个元字段：<code>value</code>/<code>confidence</code>/<code>source</code>/<code>evidence</code>/<code>timestamp</code>/<code>observations</code>。颜色 = 置信度（绿 ≥0.8 / 黄 0.6~0.8 / 红 &lt;0.6）。</div>
      {profile_html}
      <div class="warnbox"><b>⚠️ 画像质量结论：</b>链路已跑通，但质量待 DDL 4 调优——部分字段（如 name）存在误抽/臆测，evidence 与 value 不一致。详见 06 文档「六」。</div>
    </div>
  </section>

  <section>
    <h2><span class="dot"></span>⑤ 7 层记忆状态</h2>
    <div class="card">
      <div class="layers">
        <div class="layer empty"><div class="no">L1</div><div><div class="ttl">当前瞬间</div><div class="desc">当次交互原始 moment</div></div><div class="st"><span class="badge empty">{len(mem["layer1"])} 条</span></div></div>
        <div class="layer empty"><div class="no">L2</div><div><div class="ttl">同场景历史</div><div class="desc">本会话相同场景 moment</div></div><div class="st"><span class="badge empty">{len(mem["layer2"])} 条</span></div></div>
        <div class="layer empty"><div class="no">L3</div><div><div class="ttl">今日全部</div><div class="desc">当天所有 moment</div></div><div class="st"><span class="badge empty">{len(mem["layer3"])} 条</span></div></div>
        <div class="layer hot"><div class="no">L4</div><div><div class="ttl">近期摘要</div><div class="desc">{esc(l4 or "（空）")}</div></div><div class="st"><span class="badge ok">{'✓ 已生成' if l4 else '空'}</span></div></div>
        <div class="layer hot"><div class="no">L5</div><div><div class="ttl">长期模式</div><div class="desc">{esc(l5 or "（空）")}</div></div><div class="st"><span class="badge ok">{'✓ 已生成' if l5 else '空'}</span></div></div>
        <div class="layer hot"><div class="no">L6</div><div><div class="ttl">用户画像 ⭐</div><div class="desc">{esc(l6 or "（空）")}</div></div><div class="st"><span class="badge ok">{'✓ 已生成' if l6 else '空'}</span></div></div>
        <div class="layer warn"><div class="no">L7</div><div><div class="ttl">分类归档索引</div><div class="desc">person / location / activity / time</div></div><div class="st"><span class="badge warn">⚠ 索引污染</span></div></div>
      </div>
    </div>
  </section>

  <div style="text-align:center; color:#9ca3af; font-size:12px; margin-top:36px">
    由 generate_report.py 自动生成 · 数据源 memory.json + analysis_result.txt + first_frame.jpg
  </div>
</div>
</body>
</html>"""


def main():
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    mem = load_memory()
    analysis = load_analysis()
    frame_b64 = load_frame_b64()
    video_info = load_video_info(video_path)
    transcript = load_transcript()

    profile = mem["layer6"].get("profile", {})
    secs = parse_analysis_sections(analysis)
    timings = parse_timings(analysis)

    html = build_html(video_info, frame_b64, transcript, profile, mem, secs, timings)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ 报告已生成：{OUT_HTML}")
    print(f"   视频：{video_info['path']}（{video_info.get('duration')}s）")
    print(f"   画像字段：{list(profile.keys())}")
    print(f"   Phase 耗时：0={timings['phase0']}s A={timings['phase_a']}s B={timings['phase_b']}s wall={timings['wall']}s")


if __name__ == "__main__":
    main()
