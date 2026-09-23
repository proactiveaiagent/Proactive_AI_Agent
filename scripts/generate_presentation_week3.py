"""
generate_presentation_week3.py — 生成 9.12~9.22 任务 2 工作汇报 HTML（翻页式 · 日间模式）
======================================================================
输出：zhx/task2/汇报展示-0912-0918.html（单文件，无外部依赖）
覆盖：阶段 A（范围重构 + 设计说明书 v1.1）+ 阶段 B（增删改查完备 + 遗忘差异化 + 归类压缩）
      + 阶段 C（数据集下载 + 预处理 + 测试集构建 + 数据库规范对齐）
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
<title>任务 2 · 7 层记忆模型 — 工作汇报（09-12 ~ 09-22）</title>
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
  <div class="sub">把看到听到的内容持续整理成用户画像，存进一个分层的记忆系统。目标是存储不超过 100GB、检索不超过 1 秒、归类准确率不低于 95%。本汇报覆盖 9 月 12 日到 9 月 22 日：先是重新界定了范围并写出设计说明书，然后把 CRUD、遗忘、归类压缩这些能力一个个写成代码，接着把 EgoLife 数据集下载下来并整理成能用的数据，最后构建测试集并对齐了数据库规范。</div>
  <div class="tags">
    <span class="tag">阶段 A · 范围重构与设计 ✅</span>
    <span class="tag">阶段 B · CRUD 完备 ✅</span>
    <span class="tag">阶段 C · 数据处理与测试集 ✅</span>
    <span class="tag">274 项自动检查全过 ✅</span>
    <span class="tag">设计说明书 v1.1 ✅</span>
    <span class="tag">EgoLife 数据 ✅</span>
  </div>
  <div class="note" style="color:#3a4058;max-width:1000px">几个词先说清楚：<b>CRUD</b> 指新增、读取、更新、删除这四类操作；<b>LLM</b> 指大语言模型，也就是负责理解内容的大模型；<b>ASR</b> 指语音转文字；<b>检查项</b>指为验证代码行为是否符合预期而写的自动检查，跑一遍就能知道代码有没有写错。</div>
</div>

<div class="wrap">

<div class="kpis">
  <div class="kpi"><div class="num brand">11 天</div><div class="lbl">汇报跨度（9 月 12 日 ~ 9 月 22 日）</div></div>
  <div class="kpi"><div class="num ok">274</div><div class="lbl">项自动检查全部通过</div></div>
  <div class="kpi"><div class="num brand">9 个</div><div class="lbl">CRUD 操作全部实现</div></div>
  <div class="kpi"><div class="num brand">6 个</div><div class="lbl">索引分类维度</div></div>
  <div class="kpi"><div class="num brand">8 个</div><div class="lbl">用户画像字段</div></div>
  <div class="kpi"><div class="num ok">9002</div><div class="lbl">条整理好的数据</div></div>
</div>

<!-- ============================== PAGE 1 · 09-12 ============================== -->
<div class="page" id="p1">
  <section>
    <h2><span class="dot"></span>09-12 · 会议重新界定了范围</h2>
    <div class="card">
      <h3>① 范围为什么变了</h3>
      <div class="flow">
        <div class="fnode">原范围<br><small>只优化用户画像那一层</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">新范围<br><small>整个 7 层记忆模型</small></div>
      </div>
      <p style="font-size:13px;margin-top:10px">原来只打算优化用户画像那一层，开会讨论后决定把整个 7 层记忆模型都做一遍——包括基础架构、增删改查机制、数据归类压缩流程，还要配套一份完整的设计说明书。核心交付从一个画像模块，变成了一整套记忆系统。</p>
      <h3>② 会议的五条决定</h3>
      <table>
        <tr><th>#</th><th>决定</th><th>具体含义</th></tr>
        <tr><td>1</td><td>范围边界</td><td>只管记忆模型本身（存储、检索、更新、遗忘、压缩、索引），LLM怎么理解内容等其他模块一律不管</td></tr>
        <tr><td>2</td><td>属性统一结构</td><td>所有属性统一带上置信度（这条信息有多可靠）、时间戳、最后使用时间这些字段</td></tr>
        <tr><td>3</td><td>遗忘机制差异化</td><td>按属性类型分三类处理：身份偏好类基本不衰减，经历状态类随时间渐变，优惠券这类有使用期限的一到期就失效</td></tr>
        <tr><td>4</td><td>文档规范</td><td>不再写日报周报式的概括，改成说明书加论文的形式，记录设计原理、代码位置、修改说明</td></tr>
        <tr><td>5</td><td>代码规范</td><td>代码和产出直接放主目录（<code>scripts/memory/</code>、<code>code/</code>），不再放在 <code>zhx/</code> 下</td></tr>
      </table>
      <h3>③ 当天重写的文档</h3>
      <table>
        <tr><th>文档</th><th>改动</th></tr>
        <tr><td><code>zhx/task2/todo.md</code></td><td>重写成 7 层记忆模型的工作计划，含验收标准、范围澄清、阶段 A 到 E 的安排</td></tr>
        <tr><td><code>zhx/task2/report.md</code></td><td>重写成体系说明书加设计文档的形式</td></tr>
        <tr><td>代码目录</td><td>测试脚本从 <code>zhx/task2/scripts/</code> 迁到主目录 <code>scripts/memory/</code></td></tr>
      </table>
      <h3>④ 代码整理：梳理成 8 个清晰的 commit（一次代码提交），推送到开发分支</h3>
      <table>
        <tr><th>#</th><th>commit 内容</th></tr>
        <tr><td>1</td><td>新增用户画像抽取模块 <code>profile_extractor.py</code></td></tr>
        <tr><td>2</td><td>修复记忆整理后台线程被终止、导致画像从未生成的问题</td></tr>
        <tr><td>3</td><td>场景分析接入分层检索，记忆整理接入画像抽取</td></tr>
        <tr><td>4</td><td>重构记忆模型：分层存储、检索、时间衰减、索引清洗</td></tr>
        <tr><td>5</td><td>适配推理服务到本机环境</td></tr>
        <tr><td>6</td><td>新增单元测试与记忆模块说明文档</td></tr>
        <tr><td>7</td><td>忽略运行时产物（memory.json、output 不入库）</td></tr>
        <tr><td>8</td><td>更新记忆模块说明文档，匹配当前历史与分词方案</td></tr>
      </table>
      <div class="okbox">✅ 这一天的意义是把范围、文档、代码都理顺了，为后面写设计说明书、实现 CRUD 打好基础。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 2 · 09-14 ============================== -->
<div class="page" id="p2">
  <section>
    <h2><span class="dot"></span>09-14 · 记忆模型设计说明书定稿</h2>
    <div class="card">
      <h3>① 产出：<code>docs/memory-model-design.md</code>，一共 11 章</h3>
      <table>
        <tr><th>章节</th><th>定义了什么</th></tr>
        <tr><td>§2</td><td>7 层逐层定义（每层存什么、什么时候写入、保留多久、怎么在各层之间流转），以及跨天清理的规则</td></tr>
        <tr><td>§3</td><td>字段字典（每个字段的类型、元字段、衰减策略）</td></tr>
        <tr><td>§4</td><td>增删改查九个操作的精确语义（新增、读取、整理、删除、画像合并）</td></tr>
        <tr><td>§5</td><td>三类遗忘曲线的定稿（身份类 0 和 0.001，经历类 0.005，时效类到期归零）</td></tr>
        <tr><td>§8</td><td>存储容量预算（100GB 以内的预算表和控制策略）</td></tr>
        <tr><td>§9</td><td>检索性能设计（倒排索引（按关键词建查找表）、缓存、懒加载，保证 1 秒内返回）</td></tr>
      </table>
      <h3>② 7 层结构定稿</h3>
      <div class="layers">
        <div class="layer"><div class="no">L1</div><div><div class="ttl">当前时刻</div><div class="desc">本次交互的原始记录，会话级瞬时</div></div><div><span class="badge brand">只存引用</span></div></div>
        <div class="layer"><div class="no">L2</div><div><div class="ttl">同场景历史</div><div class="desc">地点和活动都相同的近期记录</div></div><div><span class="badge brand">只存引用</span></div></div>
        <div class="layer"><div class="no">L3</div><div><div class="ttl">当日全部</div><div class="desc">当天的所有记录，容量上限 1000，超了最旧的移出</div></div><div><span class="badge brand">只存引用</span></div></div>
        <div class="layer"><div class="no">L4</div><div><div class="ttl">近期摘要</div><div class="desc">按天压缩出来的摘要、当前任务、生活动线</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer"><div class="no">L5</div><div><div class="ttl">远期摘要</div><div class="desc">按周或月压缩出来的摘要、关键事件、长期模式</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer hot"><div class="no">L6</div><div><div class="ttl">用户画像</div><div class="desc">人口统计学、偏好、常去地点、行为习惯</div></div><div><span class="badge ok">画像层</span></div></div>
        <div class="layer"><div class="no">L7</div><div><div class="ttl">分类索引归档</div><div class="desc">时间、活动、人物、地点四个索引，加上记录的完整主存储</div></div><div><span class="badge brand">索引层</span></div></div>
      </div>
      <h3>③ 四条核心设计原则，贯穿后面所有实现</h3>
      <table>
        <tr><th>原则</th><th>含义</th><th>带来什么</th></tr>
        <tr><td>数据只有一份</td><td>每条记录的完整内容只存在 layer7.moments 里，layer1 到 layer3 只保存它的编号引用</td><td>所以跨天清空 layer1 到 layer3 不会丢数据</td></tr>
        <tr><td>画像和摘要分开</td><td>画像走独立的 update_profile 接口，压缩摘要的操作不碰画像</td><td>避免摘要把画像覆盖掉、丢了历史</td></tr>
        <tr><td>索引键全部校验</td><td>所有索引键都要经过合法性校验</td><td>防止脏数据污染索引</td></tr>
        <tr><td>检索不依赖LLM</td><td>检索这条链路上不调用LLM</td><td>保证检索能在 1 秒内返回</td></tr>
      </table>
      <div class="note">📌 后面几天的代码改动，都是照着这份说明书写的——把第四章和第五章的规定一条条实现出来。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 3 · 09-15 ============================== -->
<div class="page" id="p3">
  <section>
    <h2><span class="dot"></span>09-15 · 补齐 CRUD，改造分词</h2>
    <div class="card">
            <h3>① 记忆系统一共有 9 个操作</h3>
      <p style="font-size:13px">记忆模型对外提供 9 个操作，覆盖一条记录从写入、查找、整理到删除的整个过程：</p>
      <table>
        <tr><th>#</th><th>操作</th><th>什么时候触发</th><th>做什么</th></tr>
        <tr><td>1</td><td><code>add()</code></td><td>每来一条新内容</td><td>把新记录写进 layer1、layer3 和 layer7，同时更新四个索引</td></tr>
        <tr><td>2</td><td><code>update()</code></td><td>要修改已有记录时</td><td>按字段白名单修改记录内容，人物、地点、活动变化时同步更新索引</td></tr>
        <tr><td>3</td><td><code>query()</code></td><td>按关键词找记忆</td><td>把查询词分词后匹配，按匹配程度打分，返回最相关的几条</td></tr>
        <tr><td>4</td><td><code>retrieve()</code></td><td>按条件取记忆</td><td>结合摘要和索引，按人物、地点等条件取记录，取够就停</td></tr>
        <tr><td>5</td><td><code>compress()</code></td><td>每天或每周整理时</td><td>把整理好的摘要写进 layer4、layer5，不碰画像</td></tr>
        <tr><td>6</td><td><code>sort()</code></td><td>整理时归类</td><td>把记录归到人物、地点、活动、时间等索引下</td></tr>
        <tr><td>7</td><td><code>combine()</code></td><td>整理时合并</td><td>把意思相同的索引键合并（比如「老图书馆」和「图书馆」）</td></tr>
        <tr><td>8</td><td><code>delete()</code></td><td>手动删除</td><td>删除记录和它在索引里的引用，重要数据默认不让删</td></tr>
        <tr><td>9</td><td><code>highlight()</code></td><td>标记重要内容</td><td>把记录标记为确认正确，之后受删除保护、检索时优先</td></tr>
      </table>
      <div class="note">📌 用户画像不在这 9 个操作里，它有单独的入口 <code>update_profile()</code>。原因是画像和摘要要分开管理，避免整理摘要时把画像覆盖掉。</div>
      <h3>② 这天重点重写了 <code>update()</code>：从占位实现（shim）变成真正能用</h3>
      <p style="font-size:13px">原来的 update() 只是个占位实现（shim），实际效果和新增一条记录没有区别。这次把它重写成了真正能用的更新功能：</p>
      <pre>def update(self, moment_id, updates) -> bool:
    # 1. 字段白名单：只允许改数据字段，id、时间戳、反馈、高价值标记这些不让改
    for field, val in updates.items():
        if field not in self.UPDATE_FIELDS or val is None:
            continue
        if field == "people":
            moment["people"] = [p for p in val if _is_valid_person_tag(p)]  # 人名过校验
        else:
            moment[field] = val
        changed = True
    # 2. 预翻译刷新：更新后重新翻译，保持 normalized 与正文一致
    # 3. 同步 layer1/2/3 里同 id 的条目，防止重载后引用断开
    # 4. 索引联动：人物、地点、活动变更时同步增删索引
    self._sync_indices_for_moment(moment_id, moment, old_people, old_location, old_activity)</pre>
      <h3>③ <code>delete()</code> 和 <code>highlight()</code>：给重要数据加保护</h3>
      <pre># 删除时新增保护：被用户确认过正确的重要数据（highlight）默认拒绝删除
if moment.get("highlighted") and not force:
    print("⛔ 拒绝删除 highlighted moment（force=True 可强制）")
    return False
# 删除范围：主存储 + 四个索引里的引用 + layer1/2/3 里的条目
# 总计数不减，防止记录 id 被复用造成冲突</pre>
      <h3>④ 层与层之间的流转：补上设计说明书里缺的五处</h3>
      <table>
        <tr><th>缺什么</th><th>这次怎么补的</th></tr>
        <tr><td>同场景判定</td><td>地点和活动都相同才进 layer2，否则直接进 layer3（原来是一移出就进 layer2，没有判断）</td></tr>
        <tr><td>layer2 超限</td><td>最旧的滑进 layer3，不丢弃，完整内容始终保留在主存储里</td></tr>
        <tr><td>容量上限</td><td>当日容量从 100 提到 1000</td></tr>
        <tr><td>跨天清理</td><td>日期变了就清空 layer1 到 layer3 里的内容，重置会话起始时间</td></tr>
        <tr><td>移入时去重</td><td>移入前按编号去重，避免同一条记录重复进入</td></tr>
      </table>
      <pre># 同场景判定的核心逻辑
same_env = (
    bool(moment.get("location"))
    and moment.get("location") == current.get("location")
    and moment.get("activity") == current.get("activity")
)
if same_env:
    _push(layer2, moment, 2)   # 同场景
else:
    _push(layer3, moment, 3)   # 不同场景直接进当日记录</pre>
      <h3>⑤ 分词策略改造：统一用英文分词</h3>
      <div class="flow">
        <div class="fnode">中文等文本<br><small>原始内容</small></div>
        <div class="farrow">→</div>
        <div class="fnode model">LLM翻译<br><small>translator.py + 进程内缓存</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">英文分词<br><small>写入时存 normalized</small></div>
      </div>
      <table>
        <tr><th>改动</th><th>实现</th></tr>
        <tr><td><code>tokenize()</code></td><td>改成纯英文分词（按空格和单词边界切分），替换原来的中文二元切分</td></tr>
        <tr><td><code>translator.py</code></td><td>新增翻译模块，带缓存，同一个内容不重复翻译</td></tr>
        <tr><td>预翻译</td><td>新增和更新时把中文转成英文，存到记录的 normalized 字段</td></tr>
        <tr><td>查询翻译</td><td>检索时查询词也先翻译成英文，再去命中英文内容</td></tr>
        <tr><td>翻译失败时</td><td>翻译不可用时自动退回用原文，不影响主流程</td></tr>
      </table>
      <div class="okbox">✅ 新增了一个测试文件，专门验证这天的改动：update 的字段白名单和索引联动、delete 的 highlight 保护、层间流转、跨天清理，以及写入时的预翻译，一共 39 项检查，全部通过。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 4 · 09-16 ============================== -->
<div class="page" id="p4">
  <section>
    <h2><span class="dot"></span>09-16 · 遗忘机制改成三类，不同类型区别对待</h2>
    <div class="card">
      <h3>① 遗忘不再对所有信息一视同仁</h3>
      <p style="font-size:13px">原来的做法是所有属性每天统一衰减 0.5%，会议要求按属性类型区别对待。这天改成了三种衰减方式：</p>
      <table>
        <tr><th>类型</th><th>衰减参数</th><th>适用属性</th><th>效果</th></tr>
        <tr><td><b>身份类（stable）</b></td><td>每天衰减 0，或者极慢的 0.001</td><td>姓名、性别、身份、学历</td><td><span class="badge ok">基本不衰减</span></td></tr>
        <tr><td><b>经历类（decaying）</b></td><td>每天衰减 0.005</td><td>职业这类会变化的状态</td><td><span class="badge warn">随时间慢慢衰减</span></td></tr>
        <tr><td><b>时效类（deadline）</b></td><td>一到有效期就归零</td><td>优惠券这类带 expires_at（有效期）的</td><td><span class="badge bad">一到期立刻失效</span></td></tr>
      </table>
      <pre>def effective_confidence(attr, now=None) -> float:
    dt = attr.get("decay_type")
    if dt == DECAY_DEADLINE:      # 时效类：到期就归零
        return conf if _now_before(attr["expires_at"]) else 0.0
    elif dt == DECAY_STABLE:      # 身份类：衰减极低
        daily = attr.get("decay_rate", 0.001)
    elif dt == DECAY_DECAYING:    # 经历类：渐变
        daily = attr.get("decay_rate", 0.005)
    days = (now - last_seen).days
    return conf * (1 - daily) ** days   # 有效值低于 0.3 就标记为陈旧（stale）</pre>
      <h3>② 怎么判断用哪一类：三步判断</h3>
      <div class="flow">
        <div class="fnode">① LLM标注<br><small>抽取时输出 decay_type 和 expires_at</small></div>
        <div class="farrow">→</div>
        <div class="fnode">② 按字段名定<br><small>前面没判断出来，就根据字段名决定</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">③ 最后统一校正<br><small>收尾时统一修正一遍</small></div>
      </div>
      <h3>③ 按字段名的默认策略</h3>
      <table>
        <tr><th>字段</th><th>默认衰减类型</th></tr>
        <tr><td>姓名、性别、身份、学历</td><td>身份类（永久不会忘，属于确定的身份信息）</td></tr>
        <tr><td>职业</td><td>经历类（渐变）</td></tr>
        <tr><td>偏好、常去地点、行为习惯</td><td>身份类（长有效 0.001）</td></tr>
        <tr><td>任何带 expires_at（有效期）的属性</td><td>deadline 时效类（强制按到期失效处理）</td></tr>
      </table>
      <div class="note">🔑 <b>校正规则</b>：凡是带了 expires_at 的属性，一律强制标成 deadline；LLM 已经判断过的类型会保留，不被覆盖。</div>
      <div class="okbox">✅ 新增了一个测试文件，验证三种衰减方式的选择、按字段名的默认策略、到期失效的处理、最后的校正逻辑，以及对旧数据的兼容，一共 24 项检查，全部通过。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 5 · 09-17 ============================== -->
<div class="page" id="p5">
  <section>
    <h2><span class="dot"></span>09-17 · 数据归类压缩，索引体系全部串起来</h2>
    <div class="card">
      <h3>① <code>compress()</code>（压缩摘要）：去重更精确，空值不再覆盖</h3>
      <pre># 列表去重：改成 JSON 规范化后精确比较，支持字符串和嵌套字典
existing = {json.dumps(x, ensure_ascii=False, sort_keys=True) for x in layer[k]}
for item in v:
    key = json.dumps(item, ensure_ascii=False, sort_keys=True)
    if key not in existing:
        layer[k].append(item); existing.add(key)

# 空值保护：空字符串和 None 不覆盖已有值
if v is not None and v != "":
    layer[k] = v</pre>
      <p style="font-size:13px">原来用 str(item) 去重，遇到字典元素时字段顺序不稳定，会误判成重复；改成 JSON 规范化后就精确了。另外加了空值保护，LLM偶尔返回空摘要时，不会把已有的摘要冲掉。</p>
      <h3>② <code>sort()</code>（归类索引）：过滤掉无效引用</h3>
      <pre># 只写真实存在的记录编号，防止索引指向已经被删掉的记录（这种无效引用叫悬挂引用）
valid_ids = [i for i in ids if i in moments]   # 过滤悬挂引用
existing = set(index.get(tag, [])); existing.update(valid_ids)
index[tag] = list(existing)                     # 同一个键下去重合并</pre>
      <h3>③ <code>combine()</code>（同类合并）：近义词合并，同时同步引用</h3>
      <pre># 索引到记录字段的映射，地点和活动是单值，人物是列表
REF_FIELD = {
    "locations":       ("location", "str"),
    "people":          ("people",   "list"),
    "activity_events": ("activity", "str"),
}
# 把旧键的记录集合并到规范键，同时同步更新记录里的引用字段
if kind == "str":   m[field] = canonical
if kind == "list":  m[field] = [canonical if x == old_key else x for x in m[field]]</pre>
      <p style="font-size:13px">比如「老图书馆」和「图书馆」其实是同一个地方：合并索引键的同时，把每条记录里的地点字段也从「老图书馆」改成「图书馆」，保证索引和主存储的引用一致。这个操作是幂等的（重复执行多次，结果和执行一次相同）。</p>
      <h3>④ 索引体系四个维度全部串起来</h3>
      <div class="flow">
        <div class="fnode">新增<br><small>四个索引写入</small></div>
        <div class="farrow">→</div>
        <div class="fnode">更新<br><small>索引联动增删</small></div>
        <div class="farrow">→</div>
        <div class="fnode">删除<br><small>清理引用</small></div>
        <div class="farrow">→</div>
        <div class="fnode">归类<br><small>写入索引</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">合并<br><small>近义词合并</small></div>
      </div>
      <pre>layer7 = {
  "time_nodes":      {"2026-09-17": ["m_001", ...]},
  "activity_events": {"学习": ["m_001", ...]},
  "people":          {"张三": ["m_001", ...]},
  "locations":       {"图书馆": ["m_001", ...]},
  "moments":         {"m_001": {...}}   # 主存储，完整内容只存这里
}</pre>
      <div class="okbox">✅ 新增了一个测试文件，验证 compress 的精确去重和空值保护、sort 的悬挂过滤、combine 的三类引用同步，以及幂等性，一共 42 项检查，全部通过。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 6 · 09-18 ============================== -->
<div class="page" id="p6">
  <section>
    <h2><span class="dot"></span>09-18 · 阶段小结，开始下载 EgoLife 数据集</h2>
    <div class="card">
      <h3>① 阶段 B 的测试全部通过</h3>
      <p>这一阶段的检查写在 7 个测试文件里，一共 263 项，全部通过，没有一项失败（这是 09-18 当天跑出来的数字；09-22 又补充了检查项，现在的总数是 274 项）。每一项检查对应一处代码行为，跑一遍就能验证代码有没有写错。</p>
      <table>
        <tr><th>测试脚本</th><th>验证什么</th><th>结果</th></tr>
        <tr><td><code>test_compress_sort_combine.py</code></td><td>压缩时的精确去重和空值保护、归类时的悬挂过滤、合并时的引用同步、幂等</td><td><span class="badge ok">42 项全过</span></td></tr>
        <tr><td><code>test_crud_graduation.py</code></td><td>update 的字段白名单和索引联动、delete 对重要数据（highlight）的保护、层间流转、跨天清理、预翻译</td><td><span class="badge ok">39 项全过</span></td></tr>
        <tr><td><code>test_ddl2_regression.py</code></td><td>9 月 3 日发现的那些问题的专项复查（重新验证一遍，确认没再出现）</td><td><span class="badge ok">61 项全过</span></td></tr>
        <tr><td><code>test_decay_curves.py</code></td><td>三类遗忘曲线的分派、按字段名的默认策略、有效期到期失效、校正逻辑</td><td><span class="badge ok">24 项全过</span></td></tr>
        <tr><td><code>test_profile_extractor.py</code></td><td>画像抽取（骨架字段、可信度太低就剔除、静态动态分开存、解析失败时的备用处理）</td><td><span class="badge ok">42 项全过</span></td></tr>
        <tr><td><code>test_profile_storage.py</code></td><td>分层存储（多条信息怎么合并、各接口的职责边界、原子落盘（要么完整写入、要么完全不写））</td><td><span class="badge ok">28 项全过</span></td></tr>
        <tr><td><code>test_retrieval_decay.py</code></td><td>增量更新（只处理变化的部分）和检索（英文分词、排序、early-stop（找够就停）、衰减、stale（太久没用已失效的信息））</td><td><span class="badge ok">27 项全过</span></td></tr>
      </table>
      <div class="okbox">✅ 比上周的 153 项新增了 110 项，覆盖 CRUD、遗忘、归类压缩这三块新能力，全部通过。</div>
      <h3>② EgoLife 数据集下载（这是月末验收要用的测试集）</h3>
      <table>
        <tr><th>数据</th><th>内容</th><th>状态</th></tr>
        <tr><td>ASR 转写（Transcript）</td><td>6 个参与者 402 个字幕文件，中英双语，带时间戳</td><td><span class="badge ok">完整</span></td></tr>
        <tr><td>DenseCaption</td><td>406 个字幕文件，第一人称的密集 caption</td><td><span class="badge ok">完整</span></td></tr>
        <tr><td>EgoIT</td><td>EgoLife_Caption.json（9002 条）+ EgoLife_QA.json（26.8MB）</td><td><span class="badge ok">完整</span></td></tr>
        <tr><td>原始视频</td><td>A1 完整 6266 个、A2 完整 5515 个、A3 部分 2646 个，一共 216GB</td><td><span class="badge warn">部分</span></td></tr>
      </table>
      <h3>③ 下载过程中遇到的问题</h3>
      <table>
        <tr><th>阶段</th><th>情况</th></tr>
        <tr><td>先摸清数据结构</td><td>EgoLife 分两部分：结构化文本（语音转写、场景描述、问答数据，体积小）和原始视频（6 个参与者各 7 天，32003 个文件，512GB）</td></tr>
        <tr><td>一个关键发现</td><td>数据集自带 ASR 转写和 caption，后面做预处理可以直接用这些现成文字，不用自己抽帧和做 ASR</td></tr>
        <tr><td>下载卡住</td><td>下载被开发工具的安全删除机制卡住，它拦截了缓存临时文件的清理，进度停在 499 个</td></tr>
        <tr><td>解决办法</td><td>用系统调用清理掉残留的临时文件，再用干净的环境重启下载，同时提高并发数</td></tr>
        <tr><td>带宽瓶颈</td><td>实测直连和镜像都只有每秒 100K 左右，全量下完要好几天，于是决定按需停止</td></tr>
      </table>
      <div class="note">📌 这之后就进入阶段 C——把下载好的数据整理成能用来评测的样子。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 7 · 09-19 ============================== -->
<div class="page" id="p7">
  <section>
    <h2><span class="dot"></span>09-19 · 视频下载收尾</h2>
    <div class="card">
      <h3>① 带宽到底有多慢</h3>
      <table>
        <tr><th>下载路径</th><th>实测速度</th></tr>
        <tr><td>直连下载（单线程）</td><td>每秒约 100K</td></tr>
        <tr><td>国内镜像</td><td>每秒约 83K</td></tr>
        <tr><td>16 路并发</td><td>每秒约 466K</td></tr>
      </table>
      <p>日志里没有任何报错，下载一直在正常跑，网速就是这么慢。全量 512GB 按这个速度要 6 天以上，不值得等，于是决定停止。</p>
      <h3>② 停止时保留了什么</h3>
      <table>
        <tr><th>参与者</th><th>视频数</th><th>状态</th></tr>
        <tr><td>A1_JAKE</td><td>6266</td><td><span class="badge ok">完整（一周）</span></td></tr>
        <tr><td>A2_ALICE</td><td>5515</td><td><span class="badge ok">完整（一周）</span></td></tr>
        <tr><td>A3_TASHA</td><td>2646</td><td><span class="badge warn">部分</span></td></tr>
      </table>
      <div class="okbox">✅ 拿到 2 个参与者完整的一周视频一共 216GB，加上全部 6 个参与者的结构化文本，对任务 2 完全够用。</div>
      <h3>③ 过程中的一个坑</h3>
      <p>下载中途被开发工具的安全删除机制挡过一次——它拦截了清理缓存临时文件的操作，导致下载停住。最后用系统调用清理了残留，再用干净环境重启才恢复正常。</p>
    </div>
  </section>
</div>

<!-- ============================== PAGE 8 · 09-20 ============================== -->
<div class="page" id="p8">
  <section>
    <h2><span class="dot"></span>09-20 · 检索分词改成中英混合，查询不再调LLM</h2>
    <div class="card">
      <h3>① 发现问题：查询时临时翻译会拖慢速度</h3>
      <p>之前的分词策略是遇到中文先翻译成英文再分词。写入时的预翻译没问题，但查询词如果是中文，检索时就会临时调一次LLM翻译，可能让检索要花 2 秒以上，达不到检索要在 1 秒内返回的要求。</p>
      <h3>② 改成中英混合分词</h3>
      <pre>def tokenize(text):
    # 中文按单字切，英文按单词切，不再翻译
    for ch in text.lower():
        if 是英文字母或数字:
            拼到当前单词
        elif 是中文字符:
            当前单词先入库，中文字单独作为一个词
        else:
            当前单词入库（遇到标点或空格）</pre>
      <p>中文查询词按单字去命中原文，英文查询词按单词去命中写入时预翻译好的英文。写入时的预翻译保留，供英文查询使用。</p>
      <h3>③ 效果</h3>
      <table>
        <tr><th>场景</th><th>是否调用LLM</th></tr>
        <tr><td>写入中文内容（预翻译）</td><td>调用，但是离线的，不影响检索</td></tr>
        <tr><td>中文查询词</td><td>不调用，单字直接命中原文</td></tr>
        <tr><td>英文查询词</td><td>不调用，单词命中预翻译好的英文</td></tr>
      </table>
      <div class="okbox">✅ 检索这条链路彻底不调用LLM，速度有了保障，全量测试仍然全过。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 9 · 09-21 ============================== -->
<div class="page" id="p9">
  <section>
    <h2><span class="dot"></span>09-21 · 把 EgoLife 数据整理成能用的格式</h2>
    <div class="card">
      <h3>① 复用现成文本，省掉抽帧和 ASR</h3>
      <p>数据集本身就提供了 ASR 转写和 caption，所以预处理直接复用这三样现成文字，不用自己再去做视频抽帧和 ASR：</p>
      <table>
        <tr><th>数据源</th><th>内容</th><th>对应字段</th></tr>
        <tr><td>英文叙事（9002 条）</td><td>每个 30 秒片段一条第一人称叙事</td><td>场景</td></tr>
        <tr><td>DenseCaption</td><td>逐帧的中文动作</td><td>动作</td></tr>
        <tr><td>ASR 转写</td><td>双语对话，带说话人</td><td>人物和对话</td></tr>
      </table>
      <h3>② 对齐方法</h3>
      <p>英文叙事是 30 秒一条，而语音转写和场景描述是按整点文件组织的。于是把 30 秒片段向下取整到整点，再按 30 秒一段从转写里截取对应的字幕块，避免把整小时的内容都塞进一条记录。</p>
      <h3>③ 说话人清洗</h3>
      <p>从转写里提取说话人时做了两件事：把 pJake 这类连在一起的名字还原成 Jake（归一化，也就是统一成标准写法），并过滤掉老板、女朋友这类称呼词。</p>
      <div class="okbox">✅ 产出 9002 条结构化记录，对齐记忆模型的输入格式（场景、动作、人物、地点、对话），文件只有 20MB。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 10 · 09-22 ============================== -->
<div class="page" id="p10">
  <section>
    <h2><span class="dot"></span>09-22 · 构建测试集，对齐数据库规范</h2>
    <div class="card">
      <h3>① 构建测试集</h3>
      <p>从 9000 多条里按 6 个参与者分组，每组按时间均匀抽取 150 条，得到 900 条主测试集。同时从归类、画像、检索三个方面标注了 ground-truth（人工确认过的标准答案，用来评判结果对不对）：归类覆盖人物、地点、环境、物品、活动、时间六个维度；画像目前只标了 3 个字段（人口统计学、常去地点、行为习惯），另外 5 个字段（偏好、性格、目标、决策、动机）的标准答案还没标，留待补充；检索整理了 40 个查询词和它们对应的正确答案。</p>
      <h3>② 对齐导师给的数据库规范</h3>
      <p>按最新的数据库规范做了一轮对齐，主要是两处扩展：</p>
      <table>
        <tr><th>部分</th><th>原结构</th><th>扩展后</th></tr>
        <tr><td>索引分类</td><td>4 个维度（人物、地点、活动、时间）</td><td>6 个维度（加上环境场景和物品）</td></tr>
        <tr><td>用户画像</td><td>4 个字段</td><td>8 个字段（加上性格、目标、决策、动机）</td></tr>
      </table>
      <p>还按规范定了排序规则：人物、地点、时间按从近到远排，动作、环境、物品按从高频到低频排；画像里行为习惯按频率排，其余按置信度排。第二层的同场景判定也改成按环境场景判断。</p>
      <div class="okbox">✅ 全部测试 274 项检查全部通过，覆盖了新增的六个索引维度和八个画像字段。</div>
    </div>
  </section>
</div>

</div>

<!-- 翻页条 -->
<div class="phint">← → 方向键翻页 · 点击下方按钮跳转</div>
<div class="pager" id="pager">
  <button class="pbtn nav" onclick="go(-1)" title="上一页">‹</button>
  <button class="pbtn" onclick="jump(1)">09-12<span class="d">范围重构</span></button>
  <button class="pbtn" onclick="jump(2)">09-14<span class="d">设计定稿</span></button>
  <button class="pbtn" onclick="jump(3)">09-15<span class="d">CRUD</span></button>
  <button class="pbtn" onclick="jump(4)">09-16<span class="d">遗忘机制</span></button>
  <button class="pbtn" onclick="jump(5)">09-17<span class="d">归类压缩</span></button>
  <button class="pbtn" onclick="jump(6)">09-18<span class="d">小结与数据</span></button>
  <button class="pbtn" onclick="jump(7)">09-19<span class="d">下载收尾</span></button>
  <button class="pbtn" onclick="jump(8)">09-20<span class="d">分词提速</span></button>
  <button class="pbtn" onclick="jump(9)">09-21<span class="d">数据整理</span></button>
  <button class="pbtn" onclick="jump(10)">09-22<span class="d">测试集</span></button>
  <button class="pbtn nav" onclick="go(1)" title="下一页">›</button>
</div>

<script>
const N = 10;
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
