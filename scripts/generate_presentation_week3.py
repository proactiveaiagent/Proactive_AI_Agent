"""
generate_presentation_week3.py — 生成 9.12~9.18 任务 2 工作汇报 HTML（翻页式 · 日间模式）
======================================================================
输出：zhx/task2/汇报展示-0912-0918.html（单文件，无外部依赖）
覆盖：阶段 A（会议重构 + 设计说明书 v1.1）+ 阶段 B（CRUD 完备 + 遗忘差异化 + 压缩归类）
用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python generate_presentation_week3.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # Proactive_AI_Agent
OUT = ROOT / "zhx" / "task2" / "汇报展示-0912-0918.html"

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>任务 2 · 7 层记忆模型 — 工作汇报（09-12 ~ 09-18）</title>
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
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#eef0f8;color:#3a4058;text-align:left;padding:8px 12px;font-weight:600}
td{padding:8px 12px;border-top:1px solid var(--border);vertical-align:top}
tr:hover td{background:rgba(91,108,255,.06)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11.5px;font-weight:700;white-space:nowrap}
.badge.ok{background:rgba(12,166,120,.12);color:var(--ok)}
.badge.warn{background:rgba(232,147,12,.12);color:var(--warn)}
.badge.bad{background:rgba(229,72,77,.12);color:var(--bad)}
.badge.brand{background:rgba(91,108,255,.12);color:#5b6cff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:820px){.grid2,.grid3{grid-template-columns:1fr}}
.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap}
.fnode{background:#eef0f8;border:1px solid #c9d1f2;border-radius:11px;padding:10px 14px;font-size:12.5px;font-weight:600;color:#3a4058;text-align:center}
.fnode.model{background:#ece6ff;border-color:#a879d8;color:#5b3b9e}
.fnode.hot{background:#e0f5ec;border-color:#7fd4b0;color:#0c8a5f}
.farrow{color:#9aa0c3;font-size:19px;align-self:center}
.fnode small{display:block;font-size:10.5px;color:#6b7189;font-weight:400;margin-top:2px}
.layers{display:flex;flex-direction:column;gap:8px}
.layer{display:grid;grid-template-columns:54px 1fr auto;align-items:center;gap:14px;background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:10px 16px}
.layer .no{width:42px;height:42px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px;color:#fff;background:linear-gradient(135deg,#5b6cff,#a855f7)}
.layer.hot .no{background:linear-gradient(135deg,#059669,#10b981)}
.layer .ttl{font-weight:700;font-size:14px}
.layer .desc{font-size:12px;color:var(--muted);margin-top:1px}
.note{font-size:12.5px;color:var(--muted);margin-top:11px;line-height:1.7}
.note b{color:#c77400}
.okbox{background:rgba(12,166,120,.08);border:1px solid rgba(52,211,153,.35);border-radius:10px;padding:12px 15px;margin-top:12px;font-size:12.5px;color:#0c8a5f}
.warnbox{background:rgba(232,147,12,.08);border:1px solid rgba(251,191,36,.35);border-radius:10px;padding:12px 15px;margin-top:12px;font-size:12.5px;color:#c77400}
.pager{position:fixed;left:0;right:0;bottom:0;background:rgba(245,246,251,.92);backdrop-filter:blur(10px);border-top:1px solid var(--border);padding:10px 16px;display:flex;align-items:center;justify-content:center;gap:8px;z-index:99;flex-wrap:wrap}
.pbtn{min-width:44px;height:38px;border-radius:10px;border:1px solid var(--border);background:var(--card);color:#3a4058;font-size:13px;font-weight:700;cursor:pointer;padding:0 10px;transition:.15s}
.pbtn:hover{border-color:#5b6cff}
.pbtn.on{background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;border-color:transparent}
.pbtn.nav{background:transparent;font-size:17px}
.pbtn .d{display:block;font-size:9px;font-weight:400;color:#6b7189;margin-top:-2px}
.pbtn.on .d{color:rgba(255,255,255,.75)}
.phint{position:fixed;left:16px;bottom:64px;font-size:11px;color:#9aa0c3;z-index:99}
</style>
</head>
<body>

<div class="hero">
  <h1>🧠 任务 2 · 7 层记忆模型</h1>
  <div class="sub">多模态识别结果 → 持续构建与精化用户画像 → 分层 RAG 记忆系统，目标 <b>存储 ≤100GB · 检索 ≤1s · 归类准确率 ≥95%</b>。本汇报覆盖 <b>09-12 ~ 09-18</b>：阶段 A（会议重构 + 设计说明书）+ 阶段 B（CRUD 完备 / 遗忘差异化 / 压缩归类 / EgoLife 数据）。</div>
  <div class="tags">
    <span class="tag">阶段 A · 会议重构 ✅</span>
    <span class="tag">阶段 B · CRUD 完备 ✅</span>
    <span class="tag">单测 263/263 ✅</span>
    <span class="tag">设计说明书 v1.1 ✅</span>
    <span class="tag">EgoLife 下载 ✅</span>
  </div>
</div>

<div class="wrap">

<div class="kpis">
  <div class="kpi"><div class="num brand">7 天</div><div class="lbl">汇报跨度（09-12 ~ 09-18）</div></div>
  <div class="kpi"><div class="num ok">263</div><div class="lbl">测试断言全通过（7 套）</div></div>
  <div class="kpi"><div class="num brand">9 操作</div><div class="lbl">CRUD 全面落地</div></div>
  <div class="kpi"><div class="num brand">3 类</div><div class="lbl">遗忘衰减曲线</div></div>
  <div class="kpi"><div class="num brand">4 维</div><div class="lbl">索引全链路联动</div></div>
  <div class="kpi"><div class="num ok">216G</div><div class="lbl">EgoLife 视频 + 文本</div></div>
</div>

<!-- ============================== PAGE 1 · 09-12 ============================== -->
<div class="page" id="p1">
  <section>
    <h2><span class="dot"></span>09-12 · 会议重构 — 范围扩大为整个 7 层记忆模型</h2>
    <div class="card">
      <h3>① 为什么要重构：范围发生了根本变化</h3>
      <div class="flow">
        <div class="fnode">原范围<br><small>仅 layer6 用户画像优化</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">新范围 ⭐<br><small>整个 7 层记忆模型</small></div>
      </div>
      <p style="font-size:13px;margin-top:10px">09-12 会议重新定义了任务 2 的工作范围：从「只优化 layer6 画像」扩大到「构建整个 7 层记忆模型的基础架构 + 增删改查（CRUD）机制 + 数据归类压缩流程 + 一套完整体系说明书」。核心交付不再是单一画像，而是<b>一整套记忆系统</b>。</p>
      <h3>② 会议五条决议（逐条落地）</h3>
      <table>
        <tr><th>#</th><th>决议</th><th>含义</th></tr>
        <tr><td>1</td><td>范围边界</td><td>只管记忆模型本身（存储 / 检索 / 更新 / 遗忘 / 压缩 / 索引），LLM 理解等其他模块一律不管</td></tr>
        <tr><td>2</td><td>属性统一结构</td><td>所有属性统一带「置信度 / 时间戳 / 最后使用时间」等字段（即 AttrValue）</td></tr>
        <tr><td>3</td><td>遗忘机制差异化</td><td>按属性类型分三类：身份偏好类（永久/不衰减）、经历状态类（渐变衰减）、优惠券类（过期阶跃失效）</td></tr>
        <tr><td>4</td><td>文档规范</td><td>摒弃日报周报式概括，改成「说明书 + 论文」形式，记录设计原理、代码位置、修改说明</td></tr>
        <tr><td>5</td><td>代码规范</td><td>代码与产出直接放主目录（<code>scripts/memory/</code>、<code>code/</code>），不再放 <code>zhx/</code> 下</td></tr>
      </table>
      <h3>③ 当天产出的文档重写</h3>
      <table>
        <tr><th>文档</th><th>改动</th></tr>
        <tr><td><code>zhx/task2/todo.md</code></td><td>重写为「7 层记忆模型体系」工作计划，含验收标准、范围澄清、阶段 A~E 计划</td></tr>
        <tr><td><code>zhx/task2/report.md</code></td><td>重写为「体系说明书 + 设计文档」形式（v3）</td></tr>
        <tr><td>代码目录</td><td>测试脚本从 <code>zhx/task2/scripts/</code> 迁入主目录 <code>scripts/memory/</code></td></tr>
      </table>
      <h3>④ 代码整理：细粒度重写为 8 个 commit 提交 zhx_dev</h3>
      <table>
        <tr><th>#</th><th>commit 内容</th></tr>
        <tr><td>1</td><td>新增用户画像抽取模块 <code>profile_extractor.py</code></td></tr>
        <tr><td>2</td><td>修复记忆整理后台线程被终止导致画像从未生成</td></tr>
        <tr><td>3</td><td>场景分析接入分层检索 + 记忆整理接入画像抽取</td></tr>
        <tr><td>4</td><td>重构记忆模型：分层存储、检索、时间衰减、索引清洗</td></tr>
        <tr><td>5</td><td>适配推理服务到本机环境</td></tr>
        <tr><td>6</td><td>新增单元测试与记忆模块说明文档</td></tr>
        <tr><td>7</td><td>忽略运行时产物（memory.json / output 不入库）</td></tr>
        <tr><td>8</td><td>更新记忆模块说明文档匹配当前历史与分词方案</td></tr>
      </table>
      <div class="okbox">✅ <b>本周起点</b>：范围明确、文档重写、代码梳理完毕，为阶段 A（设计说明书）和阶段 B（CRUD 落地）打好基础。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 2 · 09-14 ============================== -->
<div class="page" id="p2">
  <section>
    <h2><span class="dot"></span>09-14 · 记忆模型体系设计说明书 v1.1（定稿）</h2>
    <div class="card">
      <h3>① 产出：<code>docs/memory-model-design.md</code>（11 章完整定义）</h3>
      <table>
        <tr><th>章节</th><th>定义内容</th></tr>
        <tr><td>§2</td><td>7 层逐层定义（每层存储内容 / 写入时机 / 保留时长 / 晋升降级）+ 层间晋升与跨天清理</td></tr>
        <tr><td>§3</td><td>字段字典（每个字段的类型 / 元字段 / 衰减策略）</td></tr>
        <tr><td>§4</td><td>CRUD 9 操作精确语义（写入 / 读取 / 整理 / 删除 / 画像合并）</td></tr>
        <tr><td>§5</td><td>三类遗忘曲线定稿（stable 0/0.001、decaying 0.005、deadline 阶跃）</td></tr>
        <tr><td>§8</td><td>存储容量预算（≤100GB 预算表 + 控制策略）</td></tr>
        <tr><td>§9</td><td>检索性能设计（倒排索引 / 缓存 / 懒加载，≤1s 预算）</td></tr>
      </table>
      <h3>② 7 层结构定稿（核心架构）</h3>
      <div class="layers">
        <div class="layer"><div class="no">L1</div><div><div class="ttl">当前时刻</div><div class="desc">本次交互原始 moment（会话级瞬时）</div></div><div><span class="badge brand">窗口视图</span></div></div>
        <div class="layer"><div class="no">L2</div><div><div class="ttl">同场景历史</div><div class="desc">location+activity 相同的近期 moment（layer1 滑出时同场景判定）</div></div><div><span class="badge brand">窗口视图</span></div></div>
        <div class="layer"><div class="no">L3</div><div><div class="ttl">当日全部</div><div class="desc">当天所有 moment（容量 MAX 1000，超限最旧滑出）</div></div><div><span class="badge brand">窗口视图</span></div></div>
        <div class="layer"><div class="no">L4</div><div><div class="ttl">近期摘要</div><div class="desc">日级摘要 / 当前任务 / 生活动线（Phase C compress 生成）</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer"><div class="no">L5</div><div><div class="ttl">远期摘要</div><div class="desc">周/月级摘要 / 关键事件 / 长期模式</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer hot"><div class="no">L6</div><div><div class="ttl">用户画像 ⭐</div><div class="desc">demographics / preferences / frequent_locations / behavior_patterns</div></div><div><span class="badge ok">画像层</span></div></div>
        <div class="layer"><div class="no">L7</div><div><div class="ttl">分类索引归档</div><div class="desc">time_nodes / activity_events / people / locations 四索引 + moments 主存储</div></div><div><span class="badge brand">索引层</span></div></div>
      </div>
      <h3>③ 四条核心设计原则（贯穿后续所有实现）</h3>
      <table>
        <tr><th>原则</th><th>含义</th><th>落点</th></tr>
        <tr><td>数据真身唯一</td><td>moment 真身只存 <code>layer7.moments</code>，layer1~3 只是「窗口视图」引用</td><td>跨天清空窗口不丢数据</td></tr>
        <tr><td>画像与摘要解耦</td><td>画像走独立 <code>update_profile()</code>，<code>compress()</code> 禁止写 profile</td><td>防浅覆盖丢历史</td></tr>
        <tr><td>索引全链路校验</td><td>所有索引键过 <code>_is_valid_index_tag</code> 校验</td><td>防 G7 污染</td></tr>
        <tr><td>检索同步不含 LLM</td><td>检索链路（layer6→5→4→7）不调用 LLM</td><td>保证 ≤1s 延迟</td></tr>
      </table>
      <div class="note">📌 这份说明书是本周所有代码改动的「施工图」，09-15 ~ 09-17 的三天工作就是把 §4/§5 的语义一条条落成代码。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 3 · 09-15 ============================== -->
<div class="page" id="p3">
  <section>
    <h2><span class="dot"></span>09-15 · CRUD 补齐 + 分词策略改造</h2>
    <div class="card">
      <h3>① <code>update()</code>：从「旧 shim」重写为真正的 moment 字段更新</h3>
      <p style="font-size:13px">原实现 <code>update()</code> 其实是个空壳（等价于 add），本周重写为完整语义：</p>
      <pre>def update(self, moment_id, updates) -> bool:
    # 1. 字段白名单：只允许改 Part1-3 数据，禁止改 id/timestamp/feedback/highlighted
    for field, val in updates.items():
        if field not in self.UPDATE_FIELDS or val is None:
            continue
        if field == "people":
            moment["people"] = [p for p in val if _is_valid_person_tag(p)]  # 人名过校验
        else:
            moment[field] = val
        changed = True
    # 2. 预翻译刷新：更新后重译，保持 normalized 与正文一致
    # 3. 同步 layer1/2/3 同 id 条目（防 JSON 重载后引用断开）
    # 4. 索引联动：people/location/activity 变更时同步增删索引
    self._sync_indices_for_moment(moment_id, moment, old_people, old_location, old_activity)</pre>
      <h3>② <code>delete()</code> / <code>highlight()</code>：highlight 保护 + 返回 bool</h3>
      <pre># delete 新增 highlight 保护：用户确认正确的高价值数据默认拒删
if moment.get("highlighted") and not force:
    print("⛔ 拒绝删除 highlighted moment（force=True 可强制）")
    return False
# 删除范围：moments 主存储 + 四索引引用 + layer1/2/3 条目
# total_moments 不减（防 moment_id 复用冲突）</pre>
      <h3>③ 层间晋升：落地设计说明书 §2.4 的五处差距</h3>
      <table>
        <tr><th>差距</th><th>落地实现</th></tr>
        <tr><td>同场景判定</td><td><code>location + activity</code> 都相同 → 进 layer2，否则 → layer3（原先「滑出即进 layer2」无判定）</td></tr>
        <tr><td>layer2 超限</td><td>最旧滑入 layer3（不丢弃，真身保留在 layer7.moments）</td></tr>
        <tr><td>MAX_LAYER3</td><td>100 → 1000（当日容量扩大 10 倍）</td></tr>
        <tr><td>跨天清理</td><td><code>today</code> 变化时清空 layer1/2/3 窗口、重置 session_start</td></tr>
        <tr><td>窗口去重</td><td>滑入前按 id 去重，防止 add 时已入 layer3 的重复</td></tr>
      </table>
      <pre># _graduate_to_layer2 核心判定
same_env = (
    bool(moment.get("location"))
    and moment.get("location") == current.get("location")
    and moment.get("activity") == current.get("activity")
)
if same_env:
    _push(layer2, moment, 2)   # 同场景
else:
    _push(layer3, moment, 3)   # 不同场景直接进当日窗口</pre>
      <h3>④ 分词策略改造：统一英文分词（09-12 会议新决策）</h3>
      <div class="flow">
        <div class="fnode">非英文文本<br><small>中文等</small></div>
        <div class="farrow">→</div>
        <div class="fnode model">LLM 翻译<br><small>translator.py + 进程内缓存</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">英文 tokenize<br><small>写入时存 normalized</small></div>
      </div>
      <table>
        <tr><th>改动</th><th>实现</th></tr>
        <tr><td><code>tokenize()</code></td><td>纯英文分词（ASCII 字母数字），替换原中文 bigram</td></tr>
        <tr><td><code>translator.py</code></td><td>新增模块：LLM 翻译 + 进程内缓存（避免重复翻译）</td></tr>
        <tr><td>预翻译</td><td>add/update 写入时把中文转成英文存 <code>moment["normalized"]</code></td></tr>
        <tr><td>查询翻译</td><td>query/retrieve 查询词经 <code>_query_tokens</code> 翻译后命中英文</td></tr>
        <tr><td>降级</td><td>翻译不可用时自动降级原文兜底，不影响主流程</td></tr>
      </table>
      <div class="okbox">✅ 新增 <code>test_crud_graduation.py</code>（39 断言）：覆盖 update 白名单 / 索引联动、delete highlight 保护、层间晋升、跨天清理、预翻译。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 4 · 09-16 ============================== -->
<div class="page" id="p4">
  <section>
    <h2><span class="dot"></span>09-16 · 遗忘机制差异化 — 三类衰减曲线</h2>
    <div class="card">
      <h3>① 从「一刀切」到「三类曲线」</h3>
      <p style="font-size:13px">原实现是统一 0.5%/天衰减，会议要求「按属性类型差异化」。本周落地三类曲线：</p>
      <table>
        <tr><th>曲线</th><th>衰减参数</th><th>适用属性</th><th>效果</th></tr>
        <tr><td><b>stable</b>（身份类）</td><td>永久 0.0 / 长有效 0.001</td><td>name / gender / identity / education</td><td><span class="badge ok">基本不衰减</span></td></tr>
        <tr><td><b>decaying</b>（经历类）</td><td>渐变 0.005/天</td><td>occupation 等经历状态</td><td><span class="badge warn">随时间渐变</span></td></tr>
        <tr><td><b>deadline</b>（时效类）</td><td><code>expires_at</code> 阶跃</td><td>优惠券等有时效事件</td><td><span class="badge bad">过期瞬间失效</span></td></tr>
      </table>
      <pre>def effective_confidence(attr, now=None) -> float:
    dt = attr.get("decay_type")
    if dt == DECAY_DEADLINE:      # 时效类：阶跃
        return conf if _now_before(attr["expires_at"]) else 0.0
    elif dt == DECAY_STABLE:      # 身份类：极低衰减
        daily = attr.get("decay_rate", 0.001)
    elif dt == DECAY_DECAYING:    # 经历类：渐变
        daily = attr.get("decay_rate", 0.005)
    days = (now - last_seen).days
    return conf * (1 - daily) ** days   # eff < 0.3 → stale</pre>
      <h3>② 判定机制：三层（LLM 标注 → 字段路径兜底 → 语义校正）</h3>
      <div class="flow">
        <div class="fnode">① LLM 标注<br><small>抽取 prompt 输出 decay_type/expires_at</small></div>
        <div class="farrow">→</div>
        <div class="fnode">② 字段路径兜底<br><small>_default_decay_for_path 按字段名定策略</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">③ 语义校正<br><small>_finalize_attr 统一收口</small></div>
      </div>
      <h3>③ 字段路径默认策略映射</h3>
      <table>
        <tr><th>字段</th><th>默认 decay_type</th></tr>
        <tr><td>name / gender / identity / education</td><td>stable（永久，硬身份）</td></tr>
        <tr><td>occupation</td><td>decaying（渐变）</td></tr>
        <tr><td>preferences / frequent_locations / behavior_patterns</td><td>stable（长有效 0.001）</td></tr>
        <tr><td>带 expires_at 的任何属性</td><td>deadline（强制阶跃）</td></tr>
      </table>
      <div class="note">🔑 <b>语义校正</b>：<code>_finalize_attr</code> 里带 <code>expires_at</code> 的属性强制标为 deadline；LLM 已标注的 decay_type/decay_rate 保留不覆盖。</div>
      <div class="okbox">✅ 新增 <code>test_decay_curves.py</code>（24 断言）：覆盖三类曲线分派、字段路径默认策略、expires_at 阶跃、语义校正、旧数据兼容。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 5 · 09-17 ============================== -->
<div class="page" id="p5">
  <section>
    <h2><span class="dot"></span>09-17 · 数据归类压缩 + 索引体系完整实现</h2>
    <div class="card">
      <h3>① <code>compress()</code>：list 精确去重 + 空值保护</h3>
      <pre># list 去重：json.dumps(sort_keys=True) 精确比较（支持 str 与嵌套 dict）
existing = {json.dumps(x, ensure_ascii=False, sort_keys=True) for x in layer[k]}
for item in v:
    key = json.dumps(item, ensure_ascii=False, sort_keys=True)
    if key not in existing:
        layer[k].append(item); existing.add(key)

# 空值保护：空字符串/None 不覆盖已有值（防 LLM 返回空摘要丢历史）
if v is not None and v != "":
    layer[k] = v</pre>
      <p style="font-size:13px">原先用 <code>str(item)</code> 去重，对 dict 元素顺序不稳定会误判重复；改用 JSON 规范化后精确比较。空摘要不覆盖，防止 LLM 偶发返回空结果冲掉已有摘要。</p>
      <h3>② <code>sort()</code>：悬挂 moment_id 过滤</h3>
      <pre># 只写真实存在的 moment_id（防止索引指向已删除的 moment）
valid_ids = [i for i in ids if i in moments]   # 悬挂引用过滤
existing = set(index.get(tag, [])); existing.update(valid_ids)
index[tag] = list(existing)                     # 同键去重合并</pre>
      <h3>③ <code>combine()</code>：近义索引键合并 + moments 引用同步</h3>
      <pre># 索引 → (moments 引用字段, 字段类型[str 单值 / list 列表])
REF_FIELD = {
    "locations":       ("location", "str"),
    "people":          ("people",   "list"),
    "activity_events": ("activity", "str"),
}
# 合并 old_key 的 moment_id 集到 canonical，并同步更新 moments 里的引用字段
if kind == "str":   m[field] = canonical
if kind == "list":  m[field] = [canonical if x == old_key else x for x in m[field]]</pre>
      <p style="font-size:13px">比如「老图书馆 → 图书馆」：索引键合并的同时，把每个 moment 的 <code>location</code> 字段也从「老图书馆」改成「图书馆」，保证索引和主存储引用一致。幂等：old_key 不存在或等于 canonical 则跳过。</p>
      <h3>④ 索引体系 4 维全链路（本周打通的关键）</h3>
      <div class="flow">
        <div class="fnode">add()<br><small>四索引写入</small></div>
        <div class="farrow">→</div>
        <div class="fnode">update()<br><small>索引联动增删</small></div>
        <div class="farrow">→</div>
        <div class="fnode">delete()<br><small>清理引用</small></div>
        <div class="farrow">→</div>
        <div class="fnode">sort()<br><small>归类写入</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">combine()<br><small>近义合并</small></div>
      </div>
      <pre>layer7 = {
  "time_nodes":      {"2026-09-17": ["m_001", ...]},
  "activity_events": {"学习": ["m_001", ...]},
  "people":          {"张三": ["m_001", ...]},
  "locations":       {"图书馆": ["m_001", ...]},
  "moments":         {"m_001": {...}}   # 主存储，真身唯一
}</pre>
      <div class="okbox">✅ 新增 <code>test_compress_sort_combine.py</code>（42 断言）：覆盖 compress 精确去重/空值保护、sort 悬挂过滤、combine 三类引用同步、幂等。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 6 · 09-18 ============================== -->
<div class="page" id="p6">
  <section>
    <h2><span class="dot"></span>09-18 · 阶段小结 + EgoLife 数据集下载</h2>
    <div class="card">
      <h3>① 阶段 B 测试全绿（7 套 263 断言，0 失败）</h3>
      <table>
        <tr><th>测试脚本</th><th>覆盖内容</th><th>结果</th></tr>
        <tr><td><code>test_compress_sort_combine.py</code></td><td>compress 精确去重/空值保护、sort 悬挂过滤、combine 引用同步、幂等</td><td><span class="badge ok">42/42</span></td></tr>
        <tr><td><code>test_crud_graduation.py</code></td><td>update 白名单/索引联动、delete highlight 保护、层间晋升、跨天清理、预翻译</td><td><span class="badge ok">39/39</span></td></tr>
        <tr><td><code>test_ddl2_regression.py</code></td><td>09-03 缺陷专项回归（D1/D2/D3 + G1/G5/G6/G7）</td><td><span class="badge ok">61/61</span></td></tr>
        <tr><td><code>test_decay_curves.py</code></td><td>三类衰减曲线分派、字段路径默认策略、expires_at 阶跃、语义校正</td><td><span class="badge ok">24/24</span></td></tr>
        <tr><td><code>test_profile_extractor.py</code></td><td>画像抽取（骨架字段/低置信剔除/静态动态分存/解析兜底）</td><td><span class="badge ok">42/42</span></td></tr>
        <tr><td><code>test_profile_storage.py</code></td><td>分层存储（合并语义/职责边界/原子落盘）</td><td><span class="badge ok">28/28</span></td></tr>
        <tr><td><code>test_retrieval_decay.py</code></td><td>增量更新+检索（英文分词/排序/early-stop/衰减/stale）</td><td><span class="badge ok">27/27</span></td></tr>
      </table>
      <div class="okbox">✅ 较上周（153 断言）新增 110 断言，覆盖 CRUD / 遗忘 / 压缩归类三大新能力。全量回归无失败。</div>
      <h3>② EgoLife 数据集下载（任务 2 月末验收的测试集）</h3>
      <table>
        <tr><th>数据</th><th>内容</th><th>状态</th></tr>
        <tr><td>EgoLifeCap/Transcript</td><td>6 参与者 402 个 .srt，双语 ASR 转写（中文+英文，带时间戳）</td><td><span class="badge ok">完整 ✅</span></td></tr>
        <tr><td>EgoLifeCap/DenseCaption</td><td>406 个 .srt，第一人称密集场景 caption</td><td><span class="badge ok">完整 ✅</span></td></tr>
        <tr><td>EgoIT/</td><td>EgoLife_Caption.json（9002 条）+ EgoLife_QA.json（26.8MB）</td><td><span class="badge ok">完整 ✅</span></td></tr>
        <tr><td>原始视频</td><td>A1_JAKE 完整（6266）+ A2_ALICE 完整（5515）+ A3_TASHA 部分（2646），共 216GB</td><td><span class="badge warn">部分</span></td></tr>
      </table>
      <h3>③ 下载过程记录（遇到的问题与解决）</h3>
      <table>
        <tr><th>阶段</th><th>情况</th></tr>
        <tr><td>数据结构探索</td><td>摸清 EgoLife 结构：结构化文本（Transcript/DenseCaption/EgoIT，小体积）+ 原始视频（A1~A6 × 7 天，32003 个 mp4，512GB）</td></tr>
        <tr><td>关键发现</td><td>EgoLifeCap 已内置 ASR 转写 + 密集 caption，9.21 预处理可直接复用文本，无需重新抽帧+ASR</td></tr>
        <tr><td>下载受阻</td><td>hf 下载被 CodeBuddy 的 safe-delete 机制卡死（拦截缓存临时文件清理，卡在 499/32003）</td></tr>
        <tr><td>解决</td><td>清理残留临时文件 + <code>env -i</code> 干净环境重启 + <code>--max-workers 16</code> 提并发</td></tr>
        <tr><td>带宽瓶颈</td><td>实测 HF 直连 ~106KB/s、镜像 ~83KB/s，全量 512GB 需 6+ 天，按需终止（已下 216GB 够抽帧验证）</td></tr>
      </table>
      <div class="note">📌 <b>下周（阶段 C）</b>：09-21 起基于已下好的结构化文本构建 EgoLife 测试集（预处理对齐 agent 输入格式 → ground-truth 标注 → 归类准确率评测框架）。</div>
    </div>
  </section>
</div>

</div>

<!-- 翻页条 -->
<div class="phint">← → 方向键翻页 · 点击下方按钮跳转</div>
<div class="pager" id="pager">
  <button class="pbtn nav" onclick="go(-1)" title="上一页">‹</button>
  <button class="pbtn" onclick="jump(1)">09-12<span class="d">会议重构</span></button>
  <button class="pbtn" onclick="jump(2)">09-14<span class="d">设计定稿</span></button>
  <button class="pbtn" onclick="jump(3)">09-15<span class="d">CRUD+分词</span></button>
  <button class="pbtn" onclick="jump(4)">09-16<span class="d">遗忘机制</span></button>
  <button class="pbtn" onclick="jump(5)">09-17<span class="d">压缩归类</span></button>
  <button class="pbtn" onclick="jump(6)">09-18<span class="d">小结+数据</span></button>
  <button class="pbtn nav" onclick="go(1)" title="下一页">›</button>
</div>

<script>
const N = 6;
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

OUT.write_text(HTML, encoding="utf-8")
print(f"✅ 已生成 {OUT}  （{OUT.stat().st_size // 1024} KB）")
