"""
generate_presentation.py — 生成 9.1~9.11 任务 2 工作汇报 HTML（翻页式）
======================================================================
输出：zhx/task2/汇报展示-0901-0911.html（单文件，帧图 base64 内嵌，视频相对路径引用）
用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python generate_presentation.py
"""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # Proactive_AI_Agent
TASK2 = ROOT / "zhx" / "task2"
ASSETS = TASK2 / "report_assets"
OUT = TASK2 / "汇报展示-0901-0911.html"

# ---------------------------------------------------------------------------
# 1. 读帧图 → base64
# ---------------------------------------------------------------------------
def b64(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()

FRAMES = {
    "FRAME_2_1": b64("frame_2_1.jpg"),
    "FRAME_9_2": b64("frame_9_2.jpg"),
    "FRAME_1_1": b64("frame_1_1.jpg"),
    "FRAME_6_1": b64("frame_6_1.jpg"),
    "FRAME_12_1": b64("frame_12_1.jpg"),
    "FRAME_9_1": b64("frame_9_1.jpg"),
    "FRAME_4_1": b64("frame_4_1.jpg"),
    "FRAME_7_1": b64("frame_7_1.jpg"),
}

# ---------------------------------------------------------------------------
# 2. 画像数据（memory.json 当前状态）
# ---------------------------------------------------------------------------
mem = json.loads((ROOT / "code" / "memory" / "memory.json").read_text(encoding="utf-8"))
PROFILE = mem["layer6"]["profile"]

def conf_color(c: float) -> str:
    if c >= 0.8:
        return "#10b981"
    if c >= 0.6:
        return "#f59e0b"
    return "#ef4444"

def attr_row(v, c, ev=""):
    pct = int(round(c * 100))
    return (f'<div class="attr"><div class="v"><b>{v}</b>'
            f'<span class="ev">{ev}</span></div>'
            f'<div class="conf"><div class="bar-track"><div class="bar" '
            f'style="width:{pct}%;background:{conf_color(c)}"></div></div>'
            f'<span class="pct" style="color:{conf_color(c)}">{pct}%</span></div></div>')

def render_profile_html() -> str:
    labels = {
        "demographics": "身份属性 demographics",
        "preferences": "偏好 preferences",
        "frequent_locations": "常去地点 frequent_locations",
        "behavior_patterns": "行为模式 behavior_patterns",
    }
    subs = {
        "food": "饮食", "hobbies": "兴趣", "social_relations": "社会关系",
        "with_surroundings": "与周围环境交互", "with_agents": "与 AI 智能体交互",
        "with_ar_system": "与 AR 系统交互", "common_apps": "常用应用",
        "typical_behaviors": "典型行为",
    }
    def emit_attr(a):
        if isinstance(a, dict) and not a.get("stale"):
            html.append(attr_row(a.get("value"), a.get("confidence", 0), a.get("evidence", "")))
    html = []
    for field in ("demographics", "preferences", "frequent_locations", "behavior_patterns"):
        node = PROFILE.get(field)
        if not node:
            continue
        html.append(f'<div class="field"><span class="fname">{labels[field]}</span>')
        if field == "frequent_locations":
            for a in node:
                emit_attr(a)
            html.append("</div>")
            continue
        for k, v in node.items():
            if k in subs:
                html.append(f'<div class="sname">{subs[k]}</div>')
            if isinstance(v, list):
                # 子键的值是 AttrValue 列表（如 food / hobbies / social_relations / with_surroundings / with_agents）
                for a in v:
                    emit_attr(a)
            elif isinstance(v, dict) and "value" in v:
                # 单值 AttrValue（demographics.name/age/gender/... 规格书骨架的稳定字段）
                if not v.get("stale"):
                    html.append(attr_row(v.get("value"), v.get("confidence", 0), v.get("evidence", "")))
            elif isinstance(v, dict):
                # 嵌套 dict（behavior_patterns.with_ar_system.{common_apps, typical_behaviors}）
                for k2, v2 in v.items():
                    if k2 in subs:
                        html.append(f'<div class="sname" style="padding-left:12px">{subs[k2]}</div>')
                    if isinstance(v2, list):
                        for a in v2:
                            emit_attr(a)
                    elif isinstance(v2, dict) and "value" in v2:
                        if not v2.get("stale"):
                            html.append(attr_row(v2.get("value"), v2.get("confidence", 0), v2.get("evidence", "")))
        html.append("</div>")
    return "\n".join(html)

PROFILE_HTML = render_profile_html()

# ---------------------------------------------------------------------------
# 3. e2e 回归数据（35 视频）
# ---------------------------------------------------------------------------
e2e = json.loads((ROOT / "scripts" / "memory" / "e2e_regression_0911.json").read_text(encoding="utf-8"))
E2E_POINTS = ",".join(
    f'{i + 1},{r["elapsed_s"]}' for i, r in enumerate(e2e["results"])
)
E2E_BARS = "\n".join(
    f'<div class="e2ebar" style="height:{max(2, r["elapsed_s"] / 90 * 120):.0f}px" '
    f'title="{r["video"]}  {r["elapsed_s"]}s"></div>' for r in e2e["results"]
)

# ---------------------------------------------------------------------------
# 4. HTML 模板
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>任务 2 · 终身用户建模与画像精化 — 工作汇报（09-01 ~ 09-11）</title>
<style>
:root{--bg:#f5f6fb;--card:#ffffff;--card2:#f0f2fa;--ink:#1f2430;--muted:#6b7189;--brand:#5b6cff;--brand2:#a855f7;--ok:#0ca678;--warn:#e8930c;--bad:#e5484d;--border:#e3e6f0;--code:#eef0f8}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink);line-height:1.65;padding-bottom:92px}
.hero{background:linear-gradient(135deg,#e9edff,#dfe6ff 55%,#e9dfff);color:#1f2430;padding:36px 28px 30px;border-bottom:1px solid var(--border)}
.hero h1{font-size:26px;font-weight:800;letter-spacing:.5px}
.hero .sub{margin-top:8px;font-size:14px;opacity:.9;max-width:1000px}
.hero .tags{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.hero .tag{background:rgba(31,36,48,.08);border:1px solid rgba(31,36,48,.18);padding:3px 13px;border-radius:999px;font-size:12.5px}
.wrap{max-width:1120px;margin:0 auto;padding:0 22px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:22px 0 6px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:13px;padding:14px 16px}
.kpi .num{font-size:23px;font-weight:800}
.kpi .num.ok{color:var(--ok)}.kpi .num.brand{color:var(--brand2)}.kpi .num.warn{color:var(--warn)}
.kpi .lbl{font-size:12px;color:var(--muted);margin-top:3px}
.page{display:none;animation:fade .25s ease}
.page.active{display:block}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
section{margin-top:26px}
h2{font-size:19px;font-weight:800;display:flex;align-items:center;gap:9px;margin-bottom:14px}
h2 .dot{width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,var(--brand),var(--brand2));display:inline-block;flex:none}
h3{font-size:15px;font-weight:700;margin:16px 0 8px;color:#3a4058}
.card{background:var(--card);border:1px solid var(--border);border-radius:13px;padding:20px}
.muted{color:var(--muted);font-size:12.5px}
code{background:#eef0f8;padding:1px 6px;border-radius:5px;font-size:12px;color:#3d4bd1;word-break:break-all}
pre{background:var(--code);border:1px solid var(--border);border-radius:11px;padding:14px 16px;font-size:12.5px;line-height:1.55;overflow-x:auto;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#2b3147}
pre .cm{color:#6b7399}.pre .k{color:#7dd3fc}.pre .s{color:#86efac}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#eef0f8;color:#3a4058;text-align:left;padding:8px 12px;font-weight:600}
td{padding:8px 12px;border-top:1px solid var(--border);vertical-align:top}
tr:hover td{background:rgba(91,108,255,.06)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11.5px;font-weight:700;white-space:nowrap}
.badge.ok{background:rgba(12,166,120,.12);color:var(--ok)}
.badge.warn{background:rgba(232,147,12,.12);color:var(--warn)}
.badge.bad{background:rgba(229,72,77,.12);color:var(--bad)}
.badge.gray{background:#e3e6f0;color:#6b7189}
.badge.brand{background:rgba(91,108,255,.12);color:#5b6cff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:820px){.grid2,.grid3{grid-template-columns:1fr}}
/* 流程节点 */
.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap}
.fnode{background:#eef0f8;border:1px solid #c9d1f2;border-radius:11px;padding:10px 14px;font-size:12.5px;font-weight:600;color:#3a4058;text-align:center}
.fnode.model{background:#ece6ff;border-color:#a879d8;color:#5b3b9e}
.fnode.hot{background:#e0f5ec;border-color:#7fd4b0;color:#0c8a5f}
.fnode.bad{background:#fde8e8;border-color:#f0a8a8;color:#c0392b}
.farrow{color:#9aa0c3;font-size:19px;align-self:center}
.fnode small{display:block;font-size:10.5px;color:#6b7189;font-weight:400;margin-top:2px}
/* 分层记忆 */
.layers{display:flex;flex-direction:column;gap:8px}
.layer{display:grid;grid-template-columns:54px 1fr auto;align-items:center;gap:14px;background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:10px 16px}
.layer .no{width:42px;height:42px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px;color:#fff;background:linear-gradient(135deg,#5b6cff,#a855f7)}
.layer.hot .no{background:linear-gradient(135deg,#059669,#10b981)}
.layer.warn .no{background:linear-gradient(135deg,#d97706,#f59e0b)}
.layer.empty .no{background:#c3c9e0}
.layer .ttl{font-weight:700;font-size:14px}
.layer .desc{font-size:12px;color:var(--muted);margin-top:1px}
/* 柱状图 */
.bar-track{background:#e3e6f0;border-radius:999px;height:14px;overflow:hidden;flex:1}
.bar{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--brand),var(--brand2))}
.bar.c{background:linear-gradient(90deg,#f59e0b,#ef4444)}
.bar.ok{background:linear-gradient(90deg,#059669,#10b981)}
.prow{display:grid;grid-template-columns:230px 1fr 96px;align-items:center;gap:12px;margin:7px 0}
.prow .nm{font-size:13px}
.prow .val{font-size:13px;font-weight:700;text-align:right}
/* 帧/视频 */
.vidgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:820px){.vidgrid{grid-template-columns:1fr}}
.framewrap{text-align:center}
.frame{max-width:100%;border-radius:11px;border:1px solid var(--border)}
.framelbl,.vidlbl{font-size:12px;color:var(--muted);font-weight:700;margin-bottom:8px}
video{width:100%;border-radius:11px;border:1px solid var(--border);background:#000}
.transcript{background:#f0f2f8;border:1px solid var(--border);border-radius:10px;padding:12px;font-size:12.5px;color:#3a4058}
.tsline{padding:2px 0}
.note{font-size:12.5px;color:var(--muted);margin-top:11px;line-height:1.7}
.note b{color:#c77400}
/* 画像 */
.field{margin-bottom:14px}
.fname{font-weight:800;font-size:13.5px;padding:4px 12px;border-radius:8px;background:linear-gradient(135deg,rgba(91,108,255,.12),rgba(168,85,247,.15));color:#b9c2ff;display:inline-block;margin-bottom:8px}
.attr{display:grid;grid-template-columns:1fr 150px;gap:10px;align-items:center;padding:6px 0;border-bottom:1px dashed #e3e6f0}
.attr .v{font-size:13px}
.attr .v .ev{display:block;font-size:11px;color:var(--muted);margin-top:2px}
.conf{display:flex;align-items:center;gap:8px}
.conf .bar-track{height:9px}
.conf .pct{font-size:11.5px;font-weight:700;width:36px;text-align:right}
.sname{font-size:12px;font-weight:700;color:#6b7189;margin:8px 0 3px}
.warnbox{background:rgba(232,147,12,.08);border:1px solid rgba(251,191,36,.35);border-radius:10px;padding:12px 15px;margin-top:12px;font-size:12.5px;color:#c77400}
.okbox{background:rgba(12,166,120,.08);border:1px solid rgba(52,211,153,.35);border-radius:10px;padding:12px 15px;margin-top:12px;font-size:12.5px;color:#0c8a5f}
/* 图例+统计 */
.statrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:14px}
.stat{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:10px 12px;text-align:center}
.stat .n{font-size:19px;font-weight:800;color:#3a4058}
.stat .l{font-size:11px;color:var(--muted);margin-top:2px}
/* e2e 柱群 */
.e2echart{display:flex;align-items:flex-end;gap:3px;height:150px;padding:10px;background:var(--card2);border:1px solid var(--border);border-radius:10px;overflow-x:auto}
.e2ebar{flex:none;width:12px;border-radius:3px 3px 0 0;background:linear-gradient(180deg,var(--brand),#a855f7);opacity:.85}
.e2ebar:hover{opacity:1}
/* 翻页条 */
.pager{position:fixed;left:0;right:0;bottom:0;background:rgba(245,246,251,.92);backdrop-filter:blur(10px);border-top:1px solid var(--border);padding:10px 16px;display:flex;align-items:center;justify-content:center;gap:8px;z-index:99;flex-wrap:wrap}
.pbtn{min-width:44px;height:38px;border-radius:10px;border:1px solid var(--border);background:var(--card);color:#3a4058;font-size:13px;font-weight:700;cursor:pointer;padding:0 10px;transition:.15s}
.pbtn:hover{border-color:#5b6cff}
.pbtn.on{background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;border-color:transparent}
.pbtn.nav{background:transparent;font-size:17px}
.pbtn .d{display:block;font-size:9px;font-weight:400;color:#6b7189;margin-top:-2px}
.pbtn.on .d{color:rgba(255,255,255,.75)}
.phint{position:fixed;left:16px;bottom:64px;font-size:11px;color:#9aa0c3;z-index:99}
.svglbl{font-size:11px;fill:#6b7189}
.svggrid{stroke:#e3e6f0}
</style>
</head>
<body>

<div class="hero">
  <h1>🧠 任务 2 · 终身用户建模与画像精化</h1>
  <div class="sub">多模态识别结果 → 持续构建与精化用户画像（偏好 / 性格 / 兴趣 / 日常规律 / 行为模式）→ 分层 RAG 记忆系统，目标 <b>检索 ≤2s · 准确率 ≥95%</b>。本汇报覆盖 09-01 ~ 09-11（DDL 1 + DDL 2 两个里程碑）。</div>
  <div class="tags">
    <span class="tag">DDL 1 · 09-04 设计定稿 ✅</span>
    <span class="tag">DDL 2 · 09-11 落地实施 ✅</span>
    <span class="tag">单测 153/153 ✅</span>
    <span class="tag">端到端 35/35 视频 ✅</span>
    <span class="tag">Qwen3-VL-8B · 4090</span>
  </div>
</div>

<div class="wrap">

<div class="kpis">
  <div class="kpi"><div class="num brand">8 天</div><div class="lbl">汇报跨度（09-01 ~ 09-11）</div></div>
  <div class="kpi"><div class="num brand">4 套</div><div class="lbl">单元测试脚本</div></div>
  <div class="kpi"><div class="num ok">153</div><div class="lbl">测试断言全部通过</div></div>
  <div class="kpi"><div class="num ok">35/35</div><div class="lbl">测试视频端到端跑通</div></div>
  <div class="kpi"><div class="num ok">7 个</div><div class="lbl">09-03 缺陷全部闭环</div></div>
  <div class="kpi"><div class="num warn">14.38ms</div><div class="lbl">检索 p95 @5000条（远低于2s）</div></div>
</div>

<!-- ============================== PAGE 1 · 09-01 ============================== -->
<div class="page" id="p1">
  <section>
    <h2><span class="dot"></span>09-01 · 环境搭建 — 打通开发环境</h2>
    <div class="card">
      <div class="grid2">
        <div>
          <h3>① 工作空间与目录梳理</h3>
          <table>
            <tr><th>目录</th><th>说明</th></tr>
            <tr><td><code>Proactive_AI_Agent/zhx/task2/</code></td><td>任务 2 专属工作空间</td></tr>
            <tr><td><code>vflow-backend/</code></td><td>后端服务代码</td></tr>
            <tr><td><code>cxr25/</code></td><td>共享代码与依赖项</td></tr>
          </table>
          <h3>② 基础环境校验</h3>
          <p style="font-size:13px">网络 ping + 端口连接测试：底层网络及端口访问正常，多进程/多节点通信 OK。</p>
          <h3>③ 运行 pipeline.sh</h3>
          <pre>cd /data/cxr25/zhx/Proactive_AI_Agent/zhx/task2
bash pipeline.sh</pre>
          <p style="font-size:13px">✅ 各模块数据传递顺畅、无报错 —— Baseline 在当前服务器环境可正常编译与运行，依赖包、环境变量、底层通信配置完毕。</p>
        </div>
        <div>
          <h3>④ 运行环境（后续统一使用）</h3>
          <table>
            <tr><th>项</th><th>值</th></tr>
            <tr><td>GPU</td><td>RTX 4090 × 8（24GB），驱动 570.169，CUDA 12.8</td></tr>
            <tr><td>conda</td><td>Miniforge3，环境 <code>agent</code>（Python 3.11.16）</td></tr>
            <tr><td>模型</td><td>Qwen3-VL-8B-Instruct（17GB）+ faster-whisper</td></tr>
            <tr><td>推理服务</td><td>FastAPI <code>api_server.py</code>（8000 端口，可用 <code>PORT</code> 覆盖）</td></tr>
          </table>
          <h3>⑤ 验证链路</h3>
          <div class="flow">
            <div class="fnode">📹 测试视频<br><small>test_data 35 个</small></div>
            <div class="farrow">→</div>
            <div class="fnode">🖥 agent.py<br><small>主流程</small></div>
            <div class="farrow">→</div>
            <div class="fnode model">🤖 api_server.py<br><small>Qwen3-VL + Whisper</small></div>
            <div class="farrow">→</div>
            <div class="fnode hot">🗂 7 层记忆<br><small>memory.json</small></div>
          </div>
          <div class="okbox">✅ <b>结论</b>：环境打通，具备运行 Proactive AI Agent 的基础条件，后续所有开发/测试命令统一在 <code>agent</code> 环境下执行。</div>
        </div>
      </div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 2 · 09-02 ============================== -->
<div class="page" id="p2">
  <section>
    <h2><span class="dot"></span>09-02 · 代码理解 — 架构与数据流</h2>
    <div class="card">
      <h3>系统拓扑</h3>
      <div class="flow">
        <div class="fnode">📹 用户视频/屏幕录制</div>
        <div class="farrow">→</div>
        <div class="fnode">🖥 agent.py<br><small>VRAssistant 主流程</small></div>
        <div class="farrow">⇄</div>
        <div class="fnode model">🤖 api_server.py :8000<br><small>Qwen3-VL 视觉 + Whisper 语音</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">🗂 memory.py<br><small>PersonMemory 7 层 + HintMemory</small></div>
        <div class="farrow">→</div>
        <div class="fnode">📤 三通道输出<br><small>digital_info / device_control / hardware_robot</small></div>
      </div>
      <h3>主流程：Phase 0 / A / B / C</h3>
      <div class="flow">
        <div class="fnode">📦 共享提取<br><small>抽帧+抽音频+转写<br>ThreadPool 并行</small></div>
        <div class="farrow">→</div>
        <div class="fnode">✅ Phase 0<br><small>合规预检<br>上次建议是否照做</small></div>
        <div class="farrow">→</div>
        <div class="fnode">🧠 Phase A<br><small>场景→需求→方案<br>Part1-3 + memory.add</small></div>
        <div class="farrow">→</div>
        <div class="fnode">📤 Phase B<br><small>多通道输出+反馈<br>Part4</small></div>
        <div class="farrow">→</div>
        <div class="fnode model">🗜 Phase C<br><small>记忆压缩/归类/合并<br>后台线程</small></div>
      </div>
      <h3>7 层记忆结构（任务 2 核心落点 = layer6 画像）</h3>
      <div class="layers">
        <div class="layer empty"><div class="no">L1</div><div><div class="ttl">当前时刻原始记忆</div><div class="desc">本次交互的原始 moment</div></div><div><span class="badge gray">瞬时</span></div></div>
        <div class="layer empty"><div class="no">L2</div><div><div class="ttl">同场景历史记忆</div><div class="desc">同一 scene 的近期 moment</div></div><div><span class="badge gray">瞬时</span></div></div>
        <div class="layer empty"><div class="no">L3</div><div><div class="ttl">当日全部记忆</div><div class="desc">当天所有 moment</div></div><div><span class="badge gray">瞬时</span></div></div>
        <div class="layer"><div class="no">L4</div><div><div class="ttl">近期摘要</div><div class="desc">当前任务 / 生活动线（压缩生成）</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer"><div class="no">L5</div><div><div class="ttl">远期摘要</div><div class="desc">关键事件 / 长期模式</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer hot"><div class="no">L6</div><div><div class="ttl">用户画像 ⭐ 任务 2 落点</div><div class="desc">身份 / 偏好 / 习惯（静态稳定量）</div></div><div><span class="badge warn">改造重点</span></div></div>
        <div class="layer warn"><div class="no">L7</div><div><div class="ttl">分类索引归档</div><div class="desc">时间 / 活动 / 人物 / 地点 索引</div></div><div><span class="badge warn">索引被污染</span></div></div>
      </div>
      <h3>关键发现（7 条缺口，按影响排序）</h3>
      <table>
        <tr><th>#</th><th>发现</th><th>影响</th></tr>
        <tr><td>1</td><td>画像维度不全：layer6.profile 只有 name/basic_info/preferences/habits</td><td>缺 personality / hobbies / routines，无 schema 约束</td></tr>
        <tr><td>2</td><td>画像无来源与置信度</td><td>无法增量加权与冲突消解，画像质量不可控</td></tr>
        <tr><td>3</td><td>检索未接入：query()/retrieve() 零调用</td><td>Phase A 走 get_context_for_analysis，画像难以按需命中</td></tr>
        <tr><td>4</td><td>无延迟/准确率埋点</td><td>无法证明 ≤2s / ≥95%</td></tr>
        <tr><td>5</td><td>单 JSON 全量读写</td><td>记忆规模增长后检索与写入变慢</td></tr>
        <tr><td>6</td><td>反馈是模拟的（_collect_feedback_simulated）</td><td>画像更新缺真实用户信号</td></tr>
        <tr><td>7</td><td>_decide_output_plan() 规则硬编码</td><td>未用画像做个性化输出</td></tr>
      </table>
      <div class="note">📌 为 09-03 的「画像现状梳理 + 缺陷实证」划定目标。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 3 · 09-03 ============================== -->
<div class="page" id="p3">
  <section>
    <h2><span class="dot"></span>09-03 · 画像现状梳理 — 基线实测与缺陷实证</h2>
    <div class="card">
      <h3>① 历史端到端基线（HPC A800-80GB，旧版代码）</h3>
      <div class="prow"><div class="nm">Phase 0 · 合规预检 llm_check</div><div class="bar-track"><div class="bar" style="width:16%"></div></div><div class="val">2.32s</div></div>
      <div class="prow"><div class="nm">Phase A · 分析 analyze_qwen（瓶颈）</div><div class="bar-track"><div class="bar c" style="width:76%"></div></div><div class="val">10.68s</div></div>
      <div class="prow"><div class="nm">Phase A · 检索 memory_retrieval</div><div class="bar-track"><div class="bar ok" style="width:0.1%"></div></div><div class="val">0.00s</div></div>
      <div class="prow"><div class="nm">Phase B · 多通道输出</div><div class="bar-track"><div class="bar ok" style="width:0.1%"></div></div><div class="val">0.00s</div></div>
      <div class="note"><b>🔑 关键读数</b>：① <code>analyze_qwen</code> 10.68s 占端到端 <b>76%</b>，LLM 才是瓶颈；② <code>memory_retrieval</code> 0.00s —— 因为 <code>get_all_memory()</code> 没真正检索；③ 无 Phase C 数据，画像从未生成。</div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>② 检索压测（本次新增，纯 CPU，合成数据 seed=42，每档 20 次取 P50/P95）</h3>
      <div class="grid2">
        <div>
          <svg viewBox="0 0 430 240" width="100%">
            <line x1="45" y1="200" x2="410" y2="200" class="svggrid" stroke-width="1"/>
            <text x="12" y="204" class="svglbl">0</text>
            <text x="2" y="138" class="svglbl">5</text>
            <line x1="45" y1="134" x2="410" y2="134" class="svggrid" stroke-dasharray="4 4"/>
            <text x="0" y="72" class="svglbl">10</text>
            <line x1="45" y1="68" x2="410" y2="68" class="svggrid" stroke-dasharray="4 4"/>
            <text x="0" y="10" class="svglbl">15</text>
            <line x1="45" y1="6" x2="410" y2="6" class="svggrid" stroke-dasharray="4 4"/>
            <rect x="58"  y="196" width="52" height="4"  fill="#5b6cff" rx="2"><title>n=100: p95=0.26ms</title></rect>
            <rect x="152" y="183" width="52" height="17" fill="#5b6cff" rx="2"><title>n=500: p95=1.27ms</title></rect>
            <rect x="246" y="158" width="52" height="42" fill="#a855f7" rx="2"><title>n=1000: p95=3.09ms</title></rect>
            <rect x="340" y="8"   width="52" height="192" fill="#a855f7" rx="2"><title>n=5000: p95=14.38ms</title></rect>
            <text x="84" y="215" class="svglbl">100</text>
            <text x="178" y="215" class="svglbl">500</text>
            <text x="272" y="215" class="svglbl">1000</text>
            <text x="360" y="215" class="svglbl">5000</text>
            <text x="215" y="232" class="svglbl" text-anchor="middle">记忆规模 n（条）· query() p95 耗时（ms）</text>
          </svg>
        </div>
        <div>
          <table>
            <tr><th>记忆规模 n</th><th>_load()</th><th>query() p50</th><th>query() p95</th></tr>
            <tr><td>100</td><td>1.83 ms</td><td>0.24 ms</td><td>0.26 ms</td></tr>
            <tr><td>500</td><td>6.20 ms</td><td>1.22 ms</td><td>1.27 ms</td></tr>
            <tr><td>1000</td><td>11.47 ms</td><td>2.49 ms</td><td>3.09 ms</td></tr>
            <tr><td>5000</td><td>59.22 ms</td><td>13.69 ms</td><td>14.38 ms</td></tr>
          </table>
          <div class="okbox">✅ <b>检索不是瓶颈</b>：5000 条时 query() p95 仅 14.38ms，距 2s 目标有 <b>两个数量级</b> 余量。真正的瓶颈是 LLM（10.68s）。<br>⚠️ <b>隐患</b>：query() 随规模线性劣化（G3：逐条 json.dumps 全文扫描）；_load() 5000 条时 59.22ms（G4：无缓存）。</div>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>③ 三大实证缺陷（脚本 verify_profile_write.py 实测）</h3>
      <div class="grid3">
        <div>
          <h3 style="color:#e5484d">P0 · 画像从未生成</h3>
          <pre>layer4/5/6: (empty)
metadata.last_consolidation = null

调用链：
process(consolidation_blocking=False)   ← 默认非阻塞
  └─ run_phase_c → Thread(daemon=True)
       └─ t.start()  立即返回
            ↓ 主进程结束
       daemon 线程被强制杀死 ❌</pre>
          <div class="note">根因已定位：<b>运行配置问题</b>，非代码 bug。改 <code>consolidation_blocking=True</code> 即可。</div>
        </div>
        <div>
          <h3 style="color:#e5484d">P0 · 画像增量更新丢历史</h3>
          <table>
            <tr><th>字段</th><th>合并前</th><th>合并后</th></tr>
            <tr><td>hobbies</td><td>['photography','travel']</td><td style="color:var(--bad)">['hiking'] ❌</td></tr>
            <tr><td>traits</td><td>['efficient','independent']</td><td style="color:var(--bad)">['patient'] ❌</td></tr>
            <tr><td>food</td><td>'spicy'</td><td style="color:var(--bad)">'not spicy' 覆盖</td></tr>
          </table>
          <div class="note">根因：<code>compress()</code> 对 profile 内部是 <code>dict.update()</code> 浅合并，列表被整体覆盖。<b>必须重写合并逻辑</b>。</div>
        </div>
        <div>
          <h3 style="color:#e5484d">P1 · 中文检索 0 命中</h3>
          <pre>'在图书馆学习'.split()
→ ['在图书馆学习']    ← 整句 1 个 token
→ query() 返回 0 条 ❌

'library study'.split()
→ ['library','study'] ← 2 个 token
→ query() 返回 1 条 ✅</pre>
          <div class="note">G1 缺陷：按空格分词，中文整句失效。DDL 2 需替换分词策略。</div>
        </div>
      </div>
      <h3>④ 检索缺陷清单 G1~G7 + 缺口汇总</h3>
      <table>
        <tr><th>#</th><th>缺陷</th><th>证据</th><th>归属</th></tr>
        <tr><td>G1</td><td>query() 按空格分词，中文失效</td><td>实测 0 命中</td><td>DDL 2</td></tr>
        <tr><td>G2</td><td>word in text 纯子串匹配，无语义</td><td>「吃饭」≠「用餐」</td><td>DDL 2 字面 / DDL 3 语义</td></tr>
        <tr><td>G3</td><td>逐条 json.dumps 全文扫描 O(n)</td><td>0.24→13.69ms 线性劣化</td><td>DDL 2 倒排</td></tr>
        <tr><td>G4</td><td>无索引缓存，每次全量 _load()</td><td>59.22ms @5000</td><td>DDL 3</td></tr>
        <tr><td>G5</td><td>retrieve() 硬截断 [:10]，无排序</td><td>代码</td><td>DDL 2</td></tr>
        <tr><td>G6</td><td>上下文按字符截断 [:300]</td><td>中文语义破碎</td><td>DDL 2</td></tr>
        <tr><td>G7</td><td>layer7 索引污染（people 存整段描述）</td><td>实测 "Multiple travelers..."</td><td>DDL 2</td></tr>
      </table>
      <div class="note">另调研文献 14 篇（Generative Agents / MemGPT / MemoryBank / DEEPER / MemInsight / HippoRAG 等），结论：分层是共识，真正难点在<b>画像持续精化不抖</b>与<b>多模态观察中稳定抽取属性</b>。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 4 · 09-04 ============================== -->
<div class="page" id="p4">
  <section>
    <h2><span class="dot"></span>09-04 · 记忆模型与分层检索定稿（DDL 1）</h2>
    <div class="card">
      <h3>① 画像骨架：规格书 Part 2 字段 → 7 层映射（MWM 方法论支撑）</h3>
      <table>
        <tr><th>来源</th><th>字段</th><th>落层</th><th>性质</th></tr>
        <tr><td>个人账号信息（L102）</td><td><code>demographics</code>（姓名/年龄/性别/身份/居住区域/职业/受教育/社会关系）</td><td>layer6</td><td>静态稳定</td></tr>
        <tr><td>个人账号信息（L102）</td><td><code>preferences</code>（food 饮食 / hobbies 兴趣）</td><td>layer6</td><td>静态稳定</td></tr>
        <tr><td>个人账号信息（L102）</td><td><code>frequent_locations</code>（常去地点）</td><td>layer6</td><td>静态稳定</td></tr>
        <tr><td>用户交互习惯（L97）</td><td><code>behavior_patterns</code>（with_surroundings / with_ar_system / with_agents）</td><td>layer6</td><td>静态稳定</td></tr>
        <tr><td>用户状态（L96）</td><td><code>status_inference</code>（情绪/专注度）、<code>gaze_target</code>、<code>observable_behaviors</code></td><td>layer1~4</td><td>动态，不进 layer6</td></tr>
        <tr><td>7 层第 4 层（L121）</td><td>用户目前在做/即将要做的事</td><td>layer4</td><td>动态</td></tr>
      </table>
      <div class="note">🔑 三条硬约束：① 静态画像与动态状态<b>必须分存</b>；② 画像字段必须带 <code>confidence</code> 且允许被新证据修正；③ 检索是按需部分渲染，<b>否定全量塞</b>。</div>
      <h3>② 统一属性值结构 AttrValue（每个属性都带 6 个元字段）</h3>
      <pre>{
  "value":         "spicy",                // 属性值
  "confidence":    0.85,                   // 置信度 0~1
  "source":        "moment_000123",        // 来源（可溯源审计）
  "timestamp":     "2026-09-04T10:00:00",  // 首次观察时间
  "last_seen":     "2026-09-10T18:30:00",  // 最近佐证时间（时间衰减依据）
  "observations":  5                       // 佐证次数（冲突裁决"频次"依据）
}</pre>
      <h3>③ 分层检索路径（early-stop，取够即停）</h3>
      <div class="flow">
        <div class="fnode hot">① layer6 稳定画像<br><small>直读 + 缓存 &lt;1ms</small></div>
        <div class="farrow">↓不足</div>
        <div class="fnode">② layer4/5 摘要<br><small>时间范围过滤 &lt;5ms</small></div>
        <div class="farrow">↓不足</div>
        <div class="fnode">③ layer7 索引<br><small>person/location/activity &lt;10ms</small></div>
        <div class="farrow">↓不足</div>
        <div class="fnode">④ layer1/2/3 瞬时<br><small>bigram 全文兜底 &lt;30ms</small></div>
      </div>
      <div class="note"><b>硬约束</b>：检索同步链路（①→④）<b>不含任何 LLM 调用</b>——实测 5000 条 query() p95 仅 14.38ms，而 analyze_qwen 10.68s（相差 740 倍）。画像抽取/摘要生成放 Phase C 异步执行。总预算 &lt;50ms，距 2s 目标 40 倍余量。</div>
      <h3>④ G1~G7 缺陷处置定稿</h3>
      <table>
        <tr><th>#</th><th>缺陷</th><th>定稿处置</th><th>归属</th></tr>
        <tr><td>G1</td><td>中文分词失效</td><td>字符 bigram（零依赖，jieba 不采用，向量检索 DDL 3 评估）</td><td>DDL 2</td></tr>
        <tr><td>G2</td><td>无语义</td><td>先落 bigram 字面保底，DDL 3 引向量检索融合排序</td><td>DDL 2/3</td></tr>
        <tr><td>G3</td><td>O(n) 全量扫描</td><td>索引字段倒排，命中后才取正文</td><td>DDL 2</td></tr>
        <tr><td>G4</td><td>无索引缓存</td><td>进程内索引缓存 + 写时增量更新</td><td>DDL 3</td></tr>
        <tr><td>G5</td><td>硬截断 [:10]</td><td>相关性打分排序 + Top-K</td><td>DDL 2</td></tr>
        <tr><td>G6</td><td>字符截断 [:300]</td><td>按语义单元截断（early-stop 上下文）</td><td>DDL 2</td></tr>
        <tr><td>G7</td><td>索引污染</td><td>修 sort/combine 抽取 prompt 只抽人名 + 存量清洗</td><td>DDL 2</td></tr>
      </table>
      <h3>⑤ 合并语义定稿（针对 09-03 浅覆盖缺陷）</h3>
      <table>
        <tr><th>场景</th><th>语义</th></tr>
        <tr><td>列表型维度（hobbies/food/地点/行为）</td><td><b>追加去重</b>：相似度 ≥0.85（bigram Jaccard）→ 佐证合并，否则追加</td></tr>
        <tr><td>单值冲突（如 age）</td><td><b>裁决顺序</b>：频次 &gt; 最近 &gt; 置信度；败者入 history 留痕</td></tr>
        <tr><td>同值重复观察</td><td>佐证合并：observations+1、confidence 增强、last_seen 更新</td></tr>
        <tr><td>低置信度 &lt;0.3</td><td>不入库</td></tr>
        <tr><td>陈旧值</td><td>时间衰减（0.5%/天）→ effective_confidence &lt;0.3 标 stale，不物理删除</td></tr>
      </table>
      <div class="note">📦 数据集选型：首选 <b>EgoLife</b>（CVPR 2025，6 人共居一周 300h，与任务高度一致），退路 Aria Everyday Activities + 自有 test_data 12 场景。评测口径参考 EgoMemory：Precision@K / MRR / 延迟 P50/P95。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 5 · 09-07 ============================== -->
<div class="page" id="p5">
  <section>
    <h2><span class="dot"></span>09-07 · Phase C 修复 + 画像抽取模块</h2>
    <div class="card">
      <h3>① P0 根因：consolidation 跑在 daemon 线程，主进程退出即被杀</h3>
      <div class="flow">
        <div class="fnode">process()<br><small>consolidation_blocking=False</small></div>
        <div class="farrow">→</div>
        <div class="fnode">run_phase_c()<br><small>Thread(daemon=True)</small></div>
        <div class="farrow">→</div>
        <div class="fnode">t.start()<br><small>立即返回</small></div>
        <div class="farrow">→</div>
        <div class="fnode bad">主进程 return<br><small>daemon 线程被杀 ❌</small></div>
        <div class="farrow">→</div>
        <div class="fnode">LLM 请求<br><small>timeout=90s 永远等不到</small></div>
      </div>
      <h3>② 三处最小修复</h3>
      <div class="grid3">
        <div>
          <h3>修复 1 · 线程非 daemon</h3>
          <pre>t = threading.Thread(
    target=self._consolidation_worker,
    args=(phase_b_result,),
    daemon=False   # ← 解释器退出前隐式 join
)</pre>
        </div>
        <div>
          <h3>修复 2 · 脚本入口阻塞等待</h3>
          <pre># 09-07：脚本是一次性进程，必须 blocking=True
results = assistant.process(
    consolidation_blocking=True
)</pre>
        </div>
        <div>
          <h3>修复 3 · 触发条件澄清</h3>
          <pre>def _should_consolidate(self):
    if last is None:
        return total >= 3   # 首次 >=3 触发
    return total % 3 == 0   # 之后每 3 条</pre>
        </div>
      </div>
      <div class="note">原逻辑 <code>(total % 3 == 0) or (last is None and total &gt; 1)</code> 在 total=2 就过早触发。设计权衡：原「不阻塞用户响应」对生产正确，问题只在一次性脚本。</div>
      <h3>③ 画像抽取模块 code/profile_extractor.py（对齐规格书骨架）</h3>
      <table>
        <tr><th>抽取字段</th><th>规格书出处</th><th>落层</th></tr>
        <tr><td><code>demographics</code></td><td>L102 个人账号信息</td><td>layer6</td></tr>
        <tr><td><code>preferences</code></td><td>L102（food / hobbies）</td><td>layer6</td></tr>
        <tr><td><code>frequent_locations</code></td><td>L102</td><td>layer6</td></tr>
        <tr><td><code>behavior_patterns</code></td><td>L97 用户交互习惯</td><td>layer6</td></tr>
      </table>
      <pre>def build_profile_extraction_prompt(analysis_text)   # Part1-3 文本 → 抽取 prompt
def parse_profile_output(raw_llm_text)                # LLM 输出 → 结构化画像（JSON 多策略兜底）
def validate_profile(profile, min_confidence)         # 低置信剔除 + 动态量拒绝
def extract_profile_from_raw(raw_llm_text)            # 一步：解析 → 校验</pre>
      <div class="note">关键设计：① 静态 vs 动态分存（情绪/专注度/视线不进画像）；② 只抽有证据字段、不得臆造；③ 低置信 &lt;0.3 剔除；④ LLM 调用解耦，离线可单测。</div>
      <div class="okbox">✅ 单元测试 <b>42/42</b> 通过（含变异测试验证断言有效性：故意删「低置信过滤」后 49→47 变红，抓到 2 处）。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 6 · 09-08 ============================== -->
<div class="page" id="p6">
  <section>
    <h2><span class="dot"></span>09-08 · 分层存储落地 + 端到端验证（画像第一次真实生成）</h2>
    <div class="card">
      <h3>① 分层存储实现</h3>
      <div class="grid2">
        <div>
          <pre># layer6.profile 改为规格书骨架
"profile": {
    "demographics": {},          # 姓名/年龄/性别/身份/...
    "preferences": {},           # food / hobbies
    "frequent_locations": [],    # 常去地点（list 型）
    "behavior_patterns": {}      # with_surroundings / with_ar_system / with_agents
}

# 合并语义（修复 09-03 浅覆盖）
def _merge_attr_list(old, new)    # 列表追加去重（bigram Jaccard ≥0.85 佐证合并）
def _resolve_conflict(old, new)   # 频次>最近>置信度，败者入 history
def _corroborate_attr(old, new)   # observations+1、confidence 增强
def _filter_low_confidence(node)  # <0.3 剔除

# 独立画像 API（画像写入唯一入口）
def update_profile(self, extracted, moment_id=None, timestamp=None)
def get_profile(self) -> dict
# compress() 跳过 layer6.profile（职责边界）
# _save() 原子落盘：临时文件 + os.replace</pre>
        </div>
        <div>
          <h3>② 端到端验证（环境修复 7 项）</h3>
          <table>
            <tr><th>#</th><th>问题</th><th>修复</th></tr>
            <tr><td>1</td><td>transformers 5.16 移除 Vision2Seq</td><td>AutoModelForImageTextToText</td></tr>
            <tr><td>2</td><td>model_path 硬编码 HPC 路径</td><td>QWEN_MODEL_PATH → 8B 权重</td></tr>
            <tr><td>3</td><td>8000 端口被占用</td><td>PORT 环境变量可覆盖</td></tr>
            <tr><td>4</td><td>whisper safe-delete 失败</td><td>WHISPER_MODEL_PATH 本地缓存</td></tr>
            <tr><td>5</td><td>moviepy verbose 参数移除</td><td>去 verbose、保留 logger</td></tr>
            <tr><td>6</td><td>f-string 花括号 NameError</td><td>转义 {{...}}</td></tr>
            <tr><td>7</td><td>gui_agent 示例覆盖 camera</td><td>注释示例</td></tr>
          </table>
          <div class="note">另发现：本机只有完整 <b>Qwen3-VL-8B</b>（17GB），8B 在 24GB 4090 加载后占 16.33GB、空闲 7.2GB，单帧推理可跑。</div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>③ 端到端实验：输入了什么（原视频 + 抽帧 + 转写）</h3>
      <div class="vidgrid">
        <div class="framewrap">
          <div class="framelbl">原视频 · test_data/2.travel_abroad/2.1.mp4（15.05s，1280×720@30fps）</div>
          <video controls preload="metadata" src="videos/2.travel_abroad/2.1.mp4"></video>
          <div class="muted" style="margin-top:6px">视频经 task2/videos 软链接引用；若浏览器限制本地 file:// 视频，可用 python3 -m http.server 起服务后访问。</div>
        </div>
        <div>
          <div class="framelbl">模型实际看到的抽帧（第一帧）</div>
          <img class="frame" src="{{FRAME_2_1}}" alt="LAX jet bridge first frame">
          <div class="vidlbl" style="margin-top:12px">音频转写（Whisper 识别到的语音）</div>
          <div class="transcript">
            <div class="tsline">[0.00s - 5.04s]: Welcome to Los Angeles International Airport. Please proceed to immigration and customs.</div>
            <div class="tsline">[5.68s - 9.92s]: Passengers arriving on international flights, please follow signs for immigration,</div>
            <div class="tsline">[9.92s - 15.92s]: then baggage reclaim. Thank you for flying with us. Welcome to Los Angeles.</div>
          </div>
          <div class="note">输入模式 <code>frame</code>（只抽第一帧，最快）：模型只看到 <b>1 张图</b> + 音频转写文本 + 分析 prompt。</div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>④ 端到端耗时（总计 61.80s）与模型输出</h3>
      <div class="prow"><div class="nm">Phase 0 · 合规预检</div><div class="bar-track"><div class="bar" style="width:3%"></div></div><div class="val">1.89s</div></div>
      <div class="prow"><div class="nm">Phase A · 场景·需求·方案</div><div class="bar-track"><div class="bar" style="width:16%"></div></div><div class="val">9.71s</div></div>
      <div class="prow"><div class="nm">Phase B · 多通道输出</div><div class="bar-track"><div class="bar ok" style="width:0.3%"></div></div><div class="val">0.02s</div></div>
      <div class="prow"><div class="nm">Phase C · 记忆整理（生成画像）</div><div class="bar-track"><div class="bar c" style="width:81%"></div></div><div class="val">46.64s</div></div>
      <div class="note">🔑 耗时大头是 Phase C 的 LLM 调用（后台，不阻塞响应）；真正「检索」环节毫秒级，符合 ≤2s 目标。</div>
      <h3>⑤ 模型的完整输出（Phase A，节选）</h3>
      <pre>## PART 1 — Scene Recognition
- Location: Jet bridge corridor at Los Angeles International Airport (LAX), en route to boarding gate
- People: None visible (user is alone)
- User Action: Walking through jet bridge, visually following directional signage

## PART 2 — Need Analysis
Need [1]: Navigate efficiently to boarding gate using AR-assisted guidance (Confidence: 0.95)
Need [2]: Confirm flight status and gate assignment (Confidence: 0.90)
Need [3]: Minimize physical interaction with airport staff or systems (Confidence: 0.85)

## PART 3 — Solutions
Solution [1]: Overlay AR directional arrows and gate number on live camera feed.
  Output Type: digital_info   Action: Display "Gate 4B →" with animated arrow path
Solution [2]: Fetch and display real-time flight status via AR overlay.
  Output Type: digital_info   Action: Show "Flight UA 123: Boarding in 5 mins"
Solution [3]: Enable "Do Not Disturb" mode to suppress non-essential notifications.
  Output Type: device_control</pre>
    </div>

    <div class="card" style="margin-top:16px">
      <h3>⑥ 画像实际内容（layer6，规格书骨架 + 置信度）</h3>
      <div class="muted" style="margin-bottom:10px">每个属性带元字段 value / confidence / source / evidence / timestamp / observations。颜色 = 置信度（绿 ≥0.8 / 黄 0.6~0.8 / 红 &lt;0.6）。</div>
      {{PROFILE_HTML}}
      <div class="warnbox"><b>⚠️ 画像质量结论：</b>链路已跑通，但质量待 DDL 4 调优——name="Richard" 是误抽（evidence 来自污染的 layer7.people 索引），age/education 为臆测字段。印证「索引污染会反过来污染画像」。</div>
      <div class="okbox">✅ <b>关键证据</b>：<code>metadata.last_consolidation</code> null → 2026-09-08T15:59:12（Phase C 真正执行，09-03 的 P0 根因解决）；日志出现「compress() 忽略 layer6.profile」职责边界生效；单测 28/28，05 的 42/42 无回归。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 7 · 09-10 ============================== -->
<div class="page" id="p7">
  <section>
    <h2><span class="dot"></span>09-10 · 增量更新 + 时间衰减 + 分层检索接口</h2>
    <div class="card">
      <h3>① 中文分词修复（G1）：字符 bigram tokenize</h3>
      <div class="grid2">
        <div>
          <pre>def tokenize(text: str) -> list[str]:
    '''中文按字符 bigram 切分，英文/数字按空格与边界切分。零第三方依赖。'''
    tokens, buf = [], ""
    for ch in text.lower():
        if '\u4e00' &lt;= ch &lt;= '\u9fff':      # 中文字符
            if buf: tokens.append(buf); buf = ""
            tokens.append(ch)                # 单字
        elif ch.isalnum():                   # 英数累积成词
            buf += ch
        else:
            if buf: tokens.append(buf); buf = ""
    if buf: tokens.append(buf)
    # 中文部分再生成 bigram
    out = []
    for i, t in enumerate(tokens):
        out.append(t)
        if len(t) == 1 and '\u4e00' &lt;= t &lt;= '\u9fff':
            if i + 1 &lt; len(tokens) and len(tokens[i+1]) == 1 \
               and '\u4e00' &lt;= tokens[i+1] &lt;= '\u9fff':
                out.append(t + tokens[i+1])  # bigram
    return out</pre>
        </div>
        <div>
          <table>
            <tr><th>输入</th><th>修复前（split）</th><th>修复后（tokenize）</th></tr>
            <tr><td>'在图书馆学习'</td><td style="color:var(--bad)">1 个 token → 0 命中</td><td style="color:var(--ok)">['在','在图','图','图书','书','书馆','馆','馆学','学','学习','习'] → 命中 ✅</td></tr>
            <tr><td>'library study'</td><td>2 个 token → 命中</td><td style="color:var(--ok)">['library','study'] → 命中（不回归）</td></tr>
          </table>
          <div class="okbox">✅ 09-03 实证的「中文 0 命中」修复，英文保持兼容。选型理由：jieba 需词典且新词差；向量检索违反「同步链路不含 LLM」硬约束。</div>
        </div>
      </div>
      <h3>② 时间衰减 + 陈旧淘汰 + 冷启动</h3>
      <div class="grid2">
        <div>
          <pre>STABLE_DAILY_DECAY = 0.005   # 画像稳定量每天衰减 0.5%（半衰期约 139 天）

def effective_confidence(attr, daily_decay=0.005, now=None) -> float:
    days = max(0, (now - last_seen).days)
    return max(attr["confidence"] * (1 - daily_decay) ** days, 0.0)

# effective_confidence &lt; 0.3 → 标记 stale
# stale 数据：不参与检索、不注入上下文、不物理删除（保留审计）</pre>
        </div>
        <div>
          <table>
            <tr><th>机制</th><th>行为</th></tr>
            <tr><td>时间衰减</td><td>eff = conf × (1-0.005)^days，按 last_seen 起算</td></tr>
            <tr><td>陈旧淘汰</td><td>eff &lt; 0.3 → stale=True（递归重算所有 AttrValue）</td></tr>
            <tr><td>冷启动</td><td>空画像不注入 [Profile] 段，Phase A 自然退化为无画像基线</td></tr>
          </table>
        </div>
      </div>
      <h3>③ 分层检索落地：query / retrieve / get_context_for_analysis</h3>
      <table>
        <tr><th>接口</th><th>改造</th></tr>
        <tr><td><code>query()</code></td><td>split() → tokenize()；对 layer7.moments 做 token 命中数打分排序，取 Top-K，跳过 stale</td></tr>
        <tr><td><code>retrieve()</code></td><td>person/location/activity 用 tokenize 打分排序取 Top-K（不再硬截断 [:10]），跳过 stale</td></tr>
        <tr><td><code>get_context_for_analysis()</code></td><td>early-stop 路径（layer6 画像 → layer5 → layer4 → layer7 索引）；修复旧 schema 检查 bug（画像曾从未注入）</td></tr>
        <tr><td>Phase A（agent.py）</td><td><code>get_all_memory()</code> 全量 dump → <code>get_context_for_analysis()</code> 最小够用上下文</td></tr>
      </table>
      <div class="okbox">✅ 单元测试 <b>24/24</b> 通过；05 的 42/42、06 的 28/28 无回归。遗留（DDL 3）：倒排索引 G3、索引缓存 G4、语义检索 G2。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 8 · 09-11 ============================== -->
<div class="page" id="p8">
  <section>
    <h2><span class="dot"></span>09-11 · 全量单元测试 + 35 视频回归 + 周报 2（DDL 2 交付）</h2>
    <div class="card">
      <h3>① 全量单元测试：4 套 153 断言全部通过</h3>
      <div class="grid2">
        <div>
          <table>
            <tr><th>测试脚本</th><th>覆盖</th><th>结果</th></tr>
            <tr><td><code>test_profile_extractor.py</code></td><td>画像抽取（骨架字段/低置信剔除/静态动态分存/解析兜底/触发条件/prompt）</td><td><span class="badge ok">42/42</span></td></tr>
            <tr><td><code>test_profile_storage.py</code></td><td>分层存储（读写/追加去重/冲突裁决/佐证合并/旧数据迁移/职责边界/原子落盘）</td><td><span class="badge ok">28/28</span></td></tr>
            <tr><td><code>test_retrieval_decay.py</code></td><td>增量更新+检索（bigram/排序/early-stop/衰减/stale/冷启动）</td><td><span class="badge ok">24/24</span></td></tr>
            <tr><td><code>test_ddl2_regression.py</code></td><td>09-03 缺陷专项回归（D1/D2/D3 + G1/G5/G6/G7）</td><td><span class="badge ok">59/59</span></td></tr>
          </table>
          <div class="statrow">
            <div class="stat"><div class="n" style="color:#0ca678">153</div><div class="l">断言总数</div></div>
            <div class="stat"><div class="n" style="color:#0ca678">0</div><div class="l">失败</div></div>
            <div class="stat"><div class="n">4</div><div class="l">测试脚本</div></div>
          </div>
        </div>
        <div>
          <h3>② 09-03 缺陷清单闭环核对</h3>
          <table>
            <tr><th>缺陷</th><th>内容</th><th>状态</th></tr>
            <tr><td>D1</td><td>Phase C daemon 线程被杀，画像从未生成</td><td><span class="badge ok">已闭环</span></td></tr>
            <tr><td>D2</td><td>Phase A 假检索（get_all_memory 全量塞）</td><td><span class="badge ok">已闭环</span></td></tr>
            <tr><td>D3</td><td>compress() 浅覆盖，画像丢历史</td><td><span class="badge ok">已闭环</span></td></tr>
            <tr><td>G1</td><td>中文检索 0 命中</td><td><span class="badge ok">已闭环</span></td></tr>
            <tr><td>G5</td><td>retrieve 硬截断无排序</td><td><span class="badge ok">已闭环</span></td></tr>
            <tr><td>G6</td><td>上下文字符截断 / 画像不注入</td><td><span class="badge ok">已闭环</span></td></tr>
            <tr><td>G7</td><td>layer7 索引污染</td><td><span class="badge ok">已闭环</span></td></tr>
            <tr><td>G2/G3/G4</td><td>语义检索 / 倒排索引 / 索引缓存</td><td><span class="badge warn">顺延 DDL 3</span></td></tr>
          </table>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>③ 全量测试视频端到端回归：35/35 通过（12 场景 × 35 视频连续输入）</h3>
      <div class="grid2">
        <div>
          <div class="e2echart">{{E2E_BARS}}</div>
          <div class="muted" style="margin-top:6px">每个柱子 = 1 个视频的端到端耗时（9.0s ~ 89.4s，均值 25.3s）。峰值出现在 Phase C consolidation 触发的视频（每 3 条 moment 触发一次）。</div>
          <div class="statrow">
            <div class="stat"><div class="n" style="color:#0ca678">35/35</div><div class="l">视频全部跑通</div></div>
            <div class="stat"><div class="n" style="color:#0ca678">34</div><div class="l">画像生成（首条冷启动不触发）</div></div>
            <div class="stat"><div class="n">0</div><div class="l">崩溃 / 异常</div></div>
            <div class="stat"><div class="n">36</div><div class="l">累计 moments</div></div>
          </div>
          <div class="okbox">✅ 关键观察：① 第 1 条视频 moments=1 按设计不触发 Phase C（首次 ≥3 才触发），第 2 条起 <code>consolidated=True</code>、<code>profile=True</code>，画像随连续输入<b>持续精化</b>；② 每个视频 Phase A 的 [Profile] 段正确注入画像；③ 无崩溃、无 JSON 解析异常、无索引污染。</div>
        </div>
        <div>
          <h3>④ 代表场景帧（模型实际看到的抽帧）</h3>
          <div class="grid3" style="grid-template-columns:1fr 1fr">
            <div class="framewrap"><div class="framelbl">1.1 拜年 · social_hint</div><img class="frame" src="{{FRAME_1_1}}"></div>
            <div class="framewrap"><div class="framelbl">4.1 急救 · first_aid</div><img class="frame" src="{{FRAME_4_1}}"></div>
            <div class="framewrap"><div class="framelbl">6.1 酒店 · hotel</div><img class="frame" src="{{FRAME_6_1}}"></div>
            <div class="framewrap"><div class="framelbl">7.1 拍照 · photo_taking</div><img class="frame" src="{{FRAME_7_1}}"></div>
            <div class="framewrap"><div class="framelbl">12.1 购物 · shopping</div><img class="frame" src="{{FRAME_12_1}}"></div>
            <div class="framewrap"><div class="framelbl">9.2 安防 · security</div><img class="frame" src="{{FRAME_9_2}}"></div>
          </div>
          <div class="note">回归脚本 <code>run_all_videos.py</code>：备份原记忆 → 冷启动 → 连续跑 35 视频（Phase 0/A/B/C 全链路）→ 汇总 → 恢复原记忆。结果存 <code>e2e_regression_0911.json</code>。</div>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3>⑤ 周报 2 总结（09-07 ~ 09-11，DDL 2 交付）</h3>
      <table>
        <tr><th>Step</th><th>主题</th><th>状态</th></tr>
        <tr><td>05 · 09-07</td><td>Phase C 修复 + 画像抽取模块（42 单测）</td><td><span class="badge ok">完成</span></td></tr>
        <tr><td>06 · 09-08</td><td>分层存储落地 + 端到端跑通（28 单测）</td><td><span class="badge ok">完成</span></td></tr>
        <tr><td>07+08 · 09-10</td><td>增量更新/时间衰减 + 分层检索接口（24 单测）</td><td><span class="badge ok">完成</span></td></tr>
        <tr><td>09 · 09-11</td><td>单元测试 + 全量视频回归 + 周报 2</td><td><span class="badge ok">完成</span></td></tr>
      </table>
      <div class="grid2" style="margin-top:14px">
        <div class="okbox" style="margin:0"><b>本周关键成果</b>：画像从「从未生成」到「真实生成并持续精化」；静态画像/动态状态分层存储；时间维度（衰减/stale/冷启动）齐备；中文检索修复；Phase A 从全量 dump 改最小够用上下文；09-03 缺陷全闭环。</div>
        <div class="warnbox" style="margin:0"><b>下周计划（DDL 3 · 09-18 中期里程碑）</b>：索引与缓存层（倒排 G3 + 热画像 G4）→ 检索延迟 ≤2s 实测 → 画像接入 Phase A/B（A/B 对照）→ 中期评测 + 周报 3。需导师确认：向量检索（G2）选型。</div>
      </div>
    </div>
  </section>
</div>

</div>

<!-- 翻页条 -->
<div class="phint">← → 方向键翻页 · 点击下方按钮跳转</div>
<div class="pager" id="pager">
  <button class="pbtn nav" onclick="go(-1)" title="上一页">‹</button>
  <button class="pbtn" onclick="jump(1)">09-01<span class="d">环境搭建</span></button>
  <button class="pbtn" onclick="jump(2)">09-02<span class="d">代码理解</span></button>
  <button class="pbtn" onclick="jump(3)">09-03<span class="d">画像现状</span></button>
  <button class="pbtn" onclick="jump(4)">09-04<span class="d">设计定稿</span></button>
  <button class="pbtn" onclick="jump(5)">09-07<span class="d">PhaseC修复</span></button>
  <button class="pbtn" onclick="jump(6)">09-08<span class="d">存储落地</span></button>
  <button class="pbtn" onclick="jump(7)">09-10<span class="d">增量检索</span></button>
  <button class="pbtn" onclick="jump(8)">09-11<span class="d">测试周报</span></button>
  <button class="pbtn nav" onclick="go(1)" title="下一页">›</button>
</div>

<script>
const N = 8;
let cur = 1;
function jump(i) { cur = Math.min(Math.max(i, 1), N); render(); }
function go(d) { jump(cur + d); }
function render() {
  document.querySelectorAll('.page').forEach((p, idx) => {
    p.classList.toggle('active', idx + 1 === cur);
  });
  document.querySelectorAll('#pager .pbtn:not(.nav)').forEach((b, idx) => {
    b.classList.toggle('on', idx + 1 === cur);
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowRight') go(1);
  if (e.key === 'ArrowLeft') go(-1);
});
render();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# 5. 替换占位符写文件
# ---------------------------------------------------------------------------
html = HTML
for key, val in FRAMES.items():
    html = html.replace("{{" + key + "}}", val)
html = html.replace("{{PROFILE_HTML}}", PROFILE_HTML)
html = html.replace("{{E2E_BARS}}", E2E_BARS)

OUT.write_text(html, encoding="utf-8")
print(f"✅ 已生成 {OUT}  （{OUT.stat().st_size // 1024} KB）")
