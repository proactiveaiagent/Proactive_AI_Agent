"""
generate_presentation_week3.py — 生成 9.12~9.18 任务 2 工作汇报 HTML（翻页式）
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
:root{--bg:#0f1020;--card:#171931;--card2:#1e2040;--ink:#e8e9f5;--muted:#9aa0c3;--brand:#6d7cff;--brand2:#a855f7;--ok:#34d399;--warn:#fbbf24;--bad:#f87171;--border:#2a2d52;--code:#0b0d1f}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink);line-height:1.65;padding-bottom:92px}
.hero{background:linear-gradient(135deg,#1b1d3f,#2b1e52 55%,#3b1e63);color:#fff;padding:36px 28px 30px;border-bottom:1px solid var(--border)}
.hero h1{font-size:26px;font-weight:800;letter-spacing:.5px}
.hero .sub{margin-top:8px;font-size:14px;opacity:.9;max-width:1000px}
.hero .tags{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px}
.hero .tag{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.28);padding:3px 13px;border-radius:999px;font-size:12.5px}
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
h3{font-size:15px;font-weight:700;margin:16px 0 8px;color:#c9cdf2}
.card{background:var(--card);border:1px solid var(--border);border-radius:13px;padding:20px}
.muted{color:var(--muted);font-size:12.5px}
code{background:#262a55;padding:1px 6px;border-radius:5px;font-size:12px;color:#d6dbff;word-break:break-all}
pre{background:var(--code);border:1px solid var(--border);border-radius:11px;padding:14px 16px;font-size:12.5px;line-height:1.55;overflow-x:auto;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#cdd3f5}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#232552;color:#c9cdf2;text-align:left;padding:8px 12px;font-weight:600}
td{padding:8px 12px;border-top:1px solid var(--border);vertical-align:top}
tr:hover td{background:rgba(109,124,255,.05)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11.5px;font-weight:700;white-space:nowrap}
.badge.ok{background:rgba(52,211,153,.15);color:var(--ok)}
.badge.warn{background:rgba(251,191,36,.15);color:var(--warn)}
.badge.bad{background:rgba(248,113,113,.15);color:var(--bad)}
.badge.brand{background:rgba(109,124,255,.15);color:#a5b0ff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:820px){.grid2,.grid3{grid-template-columns:1fr}}
.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap}
.fnode{background:#232552;border:1px solid #3b4180;border-radius:11px;padding:10px 14px;font-size:12.5px;font-weight:600;color:#c9cdf2;text-align:center}
.fnode.model{background:#2c1f4d;border-color:#5b3b9e;color:#d8c6ff}
.fnode.hot{background:#12302a;border-color:#1f5c4c;color:#9ff0d3}
.farrow{color:#5b618f;font-size:19px;align-self:center}
.fnode small{display:block;font-size:10.5px;color:#8b91bd;font-weight:400;margin-top:2px}
.layers{display:flex;flex-direction:column;gap:8px}
.layer{display:grid;grid-template-columns:54px 1fr auto;align-items:center;gap:14px;background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:10px 16px}
.layer .no{width:42px;height:42px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:15px;color:#fff;background:linear-gradient(135deg,#4c5be0,#7c3aed)}
.layer.hot .no{background:linear-gradient(135deg,#059669,#10b981)}
.layer .ttl{font-weight:700;font-size:14px}
.layer .desc{font-size:12px;color:var(--muted);margin-top:1px}
.note{font-size:12.5px;color:var(--muted);margin-top:11px;line-height:1.7}
.note b{color:#ffd58a}
.okbox{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.35);border-radius:10px;padding:12px 15px;margin-top:12px;font-size:12.5px;color:#9ff0d3}
.warnbox{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.35);border-radius:10px;padding:12px 15px;margin-top:12px;font-size:12.5px;color:#ffd58a}
.pager{position:fixed;left:0;right:0;bottom:0;background:rgba(15,16,32,.92);backdrop-filter:blur(10px);border-top:1px solid var(--border);padding:10px 16px;display:flex;align-items:center;justify-content:center;gap:8px;z-index:99;flex-wrap:wrap}
.pbtn{min-width:44px;height:38px;border-radius:10px;border:1px solid var(--border);background:var(--card);color:#c9cdf2;font-size:13px;font-weight:700;cursor:pointer;padding:0 10px;transition:.15s}
.pbtn:hover{border-color:#6d7cff}
.pbtn.on{background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;border-color:transparent}
.pbtn.nav{background:transparent;font-size:17px}
.pbtn .d{display:block;font-size:9px;font-weight:400;color:#8b91bd;margin-top:-2px}
.pbtn.on .d{color:rgba(255,255,255,.75)}
.phint{position:fixed;left:16px;bottom:64px;font-size:11px;color:#5b618f;z-index:99}
.curve{display:flex;align-items:flex-end;gap:4px;height:120px;padding:12px;background:var(--card2);border:1px solid var(--border);border-radius:10px}
</style>
</head>
<body>

<div class="hero">
  <h1>🧠 任务 2 · 7 层记忆模型</h1>
  <div class="sub">多模态识别结果 → 持续构建与精化用户画像 → 分层 RAG 记忆系统，目标 <b>存储 ≤100GB · 检索 ≤1s · 归类准确率 ≥95%</b>。本汇报覆盖 09-12 ~ 09-18（阶段 A 会议重构 + 阶段 B CRUD 完备/遗忘差异化/压缩归类）。</div>
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
  <div class="kpi"><div class="num ok">263</div><div class="lbl">测试断言全部通过（7 套）</div></div>
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
      <h3>① 范围重定义</h3>
      <div class="flow">
        <div class="fnode">原范围<br><small>仅 layer6 画像优化</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">新范围 ⭐<br><small>整个 7 层记忆模型</small></div>
      </div>
      <table style="margin-top:14px">
        <tr><th>项</th><th>会议定稿</th></tr>
        <tr><td>范围边界</td><td>只管记忆模型本身（存储/检索/更新/遗忘/压缩/索引），LLM 理解等其他模块一律不管</td></tr>
        <tr><td>交付核心</td><td>记忆模型基础架构 + <b>CRUD 机制</b> + 数据归类压缩流程 + 一套完整体系说明书</td></tr>
        <tr><td>文档规范</td><td>摒弃日报周报式概括，改成「说明书 + 论文」形式</td></tr>
        <tr><td>代码规范</td><td>产出直接放主目录（<code>scripts/memory/</code>、<code>code/</code>），不再放 <code>zhx/</code> 下</td></tr>
        <tr><td>遗忘机制</td><td>按属性类型差异化，不得一刀切（身份偏好类/经历状态类/优惠券类）</td></tr>
      </table>
      <div class="okbox">✅ <b>产出</b>：重写 <code>todo.md</code>、<code>report.md</code>（说明书+论文）；memory 模块代码细粒度重写为 8 个 commit 提交 <b>zhx_dev</b> 分支供审查。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 2 · 09-14 ============================== -->
<div class="page" id="p2">
  <section>
    <h2><span class="dot"></span>09-14 · 记忆模型体系设计说明书 v1.1（定稿）</h2>
    <div class="card">
      <h3>① 7 层结构定稿</h3>
      <div class="layers">
        <div class="layer"><div class="no">L1</div><div><div class="ttl">当前时刻</div><div class="desc">本次交互原始 moment（会话级）</div></div><div><span class="badge brand">窗口视图</span></div></div>
        <div class="layer"><div class="no">L2</div><div><div class="ttl">同场景历史</div><div class="desc">location+activity 相同才进（09-15 落地同场景判定）</div></div><div><span class="badge brand">窗口视图</span></div></div>
        <div class="layer"><div class="no">L3</div><div><div class="ttl">当日全部</div><div class="desc">当天所有 moment（MAX 1000，超限最旧滑出）</div></div><div><span class="badge brand">窗口视图</span></div></div>
        <div class="layer"><div class="no">L4</div><div><div class="ttl">近期摘要</div><div class="desc">日级摘要 / 当前任务 / 生活动线</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer"><div class="no">L5</div><div><div class="ttl">远期摘要</div><div class="desc">周/月摘要 / 关键事件 / 长期模式</div></div><div><span class="badge brand">压缩层</span></div></div>
        <div class="layer hot"><div class="no">L6</div><div><div class="ttl">用户画像 ⭐</div><div class="desc">demographics/preferences/frequent_locations/behavior_patterns</div></div><div><span class="badge ok">画像层</span></div></div>
        <div class="layer"><div class="no">L7</div><div><div class="ttl">分类索引归档</div><div class="desc">time_nodes/activity_events/people/locations + moments 主存储</div></div><div><span class="badge brand">索引层</span></div></div>
      </div>
      <div class="note">🔑 核心设计：<b>数据真身唯一</b>（<code>layer7.moments</code> 主存储），layer1~3 只是窗口视图，跨天清空不丢数据。</div>
      <h3>② 说明书 11 章结构</h3>
      <table>
        <tr><th>章节</th><th>内容</th></tr>
        <tr><td>§2</td><td>7 层逐层定义 + 层间晋升与跨天清理</td></tr>
        <tr><td>§3</td><td>字段字典（每字段类型/元字段/衰减策略）</td></tr>
        <tr><td>§4</td><td>CRUD 9 操作精确语义</td></tr>
        <tr><td>§5</td><td>三类遗忘曲线定稿</td></tr>
        <tr><td>§8</td><td>存储容量预算（≤100GB）</td></tr>
        <tr><td>§9</td><td>检索性能设计（≤1s）</td></tr>
      </table>
    </div>
  </section>
</div>

<!-- ============================== PAGE 3 · 09-15 ============================== -->
<div class="page" id="p3">
  <section>
    <h2><span class="dot"></span>09-15 · CRUD 补齐 + 分词策略改造</h2>
    <div class="card">
      <h3>① CRUD 补齐</h3>
      <table>
        <tr><th>操作</th><th>改造内容</th></tr>
        <tr><td><code>update()</code></td><td>旧 shim → 真字段更新：白名单（禁改 id/timestamp/feedback/highlighted）、索引联动、layer 同步、预翻译刷新</td></tr>
        <tr><td><code>delete()</code></td><td>highlight 保护（<code>highlighted=True</code> 默认拒删、<code>force=True</code> 强删）+ 返回 bool</td></tr>
        <tr><td><code>highlight()</code></td><td>返回 bool</td></tr>
      </table>
      <h3>② 层间晋升（§2.4 五处差距落地）</h3>
      <table>
        <tr><th>差距</th><th>落地</th></tr>
        <tr><td>同场景判定</td><td>location+activity 相同 → layer2，否则 → layer3</td></tr>
        <tr><td>layer2 超限</td><td>最旧滑入 layer3（不丢弃）</td></tr>
        <tr><td>MAX_LAYER3</td><td>100 → 1000</td></tr>
        <tr><td>跨天清理</td><td>today 变化清空窗口</td></tr>
        <tr><td>窗口去重</td><td>滑入前按 id 去重</td></tr>
      </table>
      <h3>③ 分词改造（统一英文分词）</h3>
      <div class="flow">
        <div class="fnode">非英文文本<br><small>中文等</small></div>
        <div class="farrow">→</div>
        <div class="fnode model">LLM 翻译<br><small>translator.py + 缓存</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">英文 tokenize<br><small>写入存 normalized</small></div>
      </div>
      <div class="note">写入时预翻译存 <code>moment["normalized"]</code>；检索时查询词经 <code>_query_tokens</code> 翻译后命中；翻译不可用自动降级原文兜底。</div>
      <div class="okbox">✅ 新增 <code>test_crud_graduation.py</code>（39 断言），覆盖 update 白名单/索引联动、delete highlight 保护、层间晋升、跨天清理。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 4 · 09-16 ============================== -->
<div class="page" id="p4">
  <section>
    <h2><span class="dot"></span>09-16 · 遗忘机制差异化 — 三类衰减曲线</h2>
    <div class="card">
      <h3>① 从「一刀切」到「三类曲线」</h3>
      <table>
        <tr><th>曲线</th><th>参数</th><th>适用</th><th>示意</th></tr>
        <tr>
          <td><b>stable</b> 身份类</td><td>永久 0.0 / 长有效 0.001</td><td>name / gender / identity</td>
          <td><span class="badge ok">水平不衰减</span></td>
        </tr>
        <tr>
          <td><b>decaying</b> 经历类</td><td>渐变 0.005/天</td><td>occupation 等经历状态</td>
          <td><span class="badge warn">缓慢渐变</span></td>
        </tr>
        <tr>
          <td><b>deadline</b> 时效类</td><td><code>expires_at</code> 阶跃</td><td>优惠券等时效事件</td>
          <td><span class="badge bad">过期瞬间失效</span></td>
        </tr>
      </table>
      <h3>② 判定机制（三层）</h3>
      <div class="flow">
        <div class="fnode">LLM 标注<br><small>decay_type/expires_at</small></div>
        <div class="farrow">→</div>
        <div class="fnode">字段路径兜底<br><small>_default_decay_for_path</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">语义校正<br><small>_finalize_attr</small></div>
      </div>
      <div class="note">字段路径默认策略：硬身份（name/gender/identity/education）→ 永久；occupation → 渐变；preferences/frequent_locations/behavior_patterns → 长有效。</div>
      <div class="okbox">✅ 新增 <code>test_decay_curves.py</code>（24 断言），覆盖三类曲线分派、字段路径默认策略、expires_at 阶跃、语义校正。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 5 · 09-17 ============================== -->
<div class="page" id="p5">
  <section>
    <h2><span class="dot"></span>09-17 · 数据归类压缩 + 索引体系完整实现</h2>
    <div class="card">
      <h3>① compress / sort / combine 补全</h3>
      <table>
        <tr><th>操作</th><th>补全内容</th></tr>
        <tr><td><code>compress()</code></td><td>list 精确去重（json.dumps sort_keys，支持 str/嵌套 dict）+ 空值保护（空摘要不覆盖）+ 跳过 layer6.profile</td></tr>
        <tr><td><code>sort()</code></td><td>悬挂 moment_id 过滤（不存在的 id 不写索引）+ 索引键校验 + 同键去重</td></tr>
        <tr><td><code>combine()</code></td><td>people/locations/activity 三类索引合并 + moments 引用同步（location/activity str、people list 统一处理）+ 幂等</td></tr>
      </table>
      <h3>② 索引体系 4 维全链路</h3>
      <div class="flow">
        <div class="fnode">add()<br><small>四索引写入</small></div>
        <div class="farrow">→</div>
        <div class="fnode">update()<br><small>索引联动</small></div>
        <div class="farrow">→</div>
        <div class="fnode">delete()<br><small>清理引用</small></div>
        <div class="farrow">→</div>
        <div class="fnode">sort()<br><small>归类写入</small></div>
        <div class="farrow">→</div>
        <div class="fnode hot">combine()<br><small>近义合并</small></div>
      </div>
      <pre>layer7 = {
  "time_nodes":      {"2026-09-17": ["m_xxx", ...]},
  "activity_events": {"学习": ["m_xxx", ...]},
  "people":          {"张三": ["m_xxx", ...]},
  "locations":       {"图书馆": ["m_xxx", ...]},
  "moments":         {"m_xxx": {...}}  # 主存储，真身唯一
}</pre>
      <div class="okbox">✅ 新增 <code>test_compress_sort_combine.py</code>（42 断言），覆盖三操作补全 + 索引 4 维全链路 + 幂等。</div>
    </div>
  </section>
</div>

<!-- ============================== PAGE 6 · 09-18 ============================== -->
<div class="page" id="p6">
  <section>
    <h2><span class="dot"></span>09-18 · 阶段小结 + EgoLife 下载启动</h2>
    <div class="card">
      <h3>① 阶段 B 测试全绿（7 套 263 断言）</h3>
      <table>
        <tr><th>测试脚本</th><th>覆盖</th><th>结果</th></tr>
        <tr><td><code>test_compress_sort_combine.py</code></td><td>压缩/归类/合并 + 索引体系</td><td><span class="badge ok">42/42</span></td></tr>
        <tr><td><code>test_crud_graduation.py</code></td><td>CRUD + 层间晋升</td><td><span class="badge ok">39/39</span></td></tr>
        <tr><td><code>test_ddl2_regression.py</code></td><td>09-03 缺陷回归</td><td><span class="badge ok">61/61</span></td></tr>
        <tr><td><code>test_decay_curves.py</code></td><td>三类遗忘曲线</td><td><span class="badge ok">24/24</span></td></tr>
        <tr><td><code>test_profile_extractor.py</code></td><td>画像抽取</td><td><span class="badge ok">42/42</span></td></tr>
        <tr><td><code>test_profile_storage.py</code></td><td>分层存储</td><td><span class="badge ok">28/28</span></td></tr>
        <tr><td><code>test_retrieval_decay.py</code></td><td>增量更新+检索</td><td><span class="badge ok">27/27</span></td></tr>
      </table>
      <div class="okbox">✅ 较上周（153 断言）新增 110 断言，覆盖 CRUD / 遗忘 / 压缩归类三大新能力。</div>
      <h3>② EgoLife 数据集下载</h3>
      <table>
        <tr><th>数据</th><th>内容</th><th>状态</th></tr>
        <tr><td>结构化文本</td><td>Transcript（402 .srt 双语 ASR）+ DenseCaption（406 .srt 密集 caption）+ EgoIT（9002 条 QA）</td><td><span class="badge ok">完整 ✅</span></td></tr>
        <tr><td>原始视频</td><td>A1/A2 完整 + A3 部分，共 216GB（HF 带宽瓶颈 ~100KB/s，全量 512GB 需 6+ 天，按需终止）</td><td><span class="badge warn">部分</span></td></tr>
      </table>
      <div class="note">🔑 <b>关键发现</b>：EgoLifeCap 已内置 ASR 转写 + 密集 caption，9.21 预处理可直接复用现成文本，无需重新抽帧+ASR。</div>
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
