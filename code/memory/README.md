# Memory 模块 · 7 层记忆模型说明书（供代码审查）

> 本文件是 `code/memory.py` 所属 memory 模块的**架构说明 + 设计文档 + 变更记录**，用于代码审查。
> 核心代码：`code/memory.py`（`PersonMemory` / `HintMemory`）、`code/profile_extractor.py`（画像抽取）。
> 相关调用方：`code/agent.py`（主流程 `VRAssistant`，Phase A/C 读写记忆）。
> 任务范围（09-12 会议确认）：只管记忆模型本身（存储 / 检索 / 更新 / 遗忘 / 压缩 / 索引），不管 LLM 理解等其他模块。

---

## 一、模块概述

`PersonMemory` 是一个**分层 RAG 记忆系统**：把多模态识别结果（场景 / 需求 / 交互）持续写入 7 层记忆，
并从中精化出用户画像（layer6），支撑按需检索（≤2s）与个性化。

数据落盘为一个 JSON 文件 `code/memory/memory.json`，顶层 8 个键：`layer1~7` + `metadata`。
库结构与单条 moment 结构分别由 `_empty_db()`（`memory.py` L423）与 `_empty_moment()`（L401）定义。

### 依赖关系

```
code/agent.py（主流程 VRAssistant）
   ├─ 读：memory.get_context_for_analysis() / query() / retrieve()
   ├─ 写：memory.add() / update_feedback() / compress() / sort() / combine()
   └─ 画像：memory.update_profile() / get_profile()
code/profile_extractor.py（画像抽取，Phase C 调用）
   └─ build_profile_extraction_prompt / parse_profile_output / validate_profile
```

---

## 二、架构设计

### 2.1 7 层记忆结构

| 层 | 名称 | 存储内容 | 写入时机 | 触发者 |
|---|---|---|---|---|
| layer1 | 当前时刻 | 本次交互原始 moment | 每次交互 | Phase A `add()` |
| layer2 | 同场景历史 | 同 scene 的历史 moment | layer1 溢出毕业 | `_graduate_to_layer2()` |
| layer3 | 当日全部 | 当天所有 moment | 每次交互 | Phase A `add()` |
| layer4 | 近期摘要 | 日级摘要 / 当前任务 / 生活动线 | Phase C | `compress()` |
| layer5 | 远期摘要 | 周/月摘要 / 关键事件 / 长期模式 | Phase C | `compress()` |
| layer6 | 用户画像 ⭐ | 静态画像（四字段，见 2.3） | Phase C | `update_profile()` |
| layer7 | 分类索引归档 | time_nodes / activity_events / people / locations 倒排索引 + moments 主存储 | 每次交互 + Phase C | `add()` / `sort()` / `combine()` |

### 2.2 单条 moment 结构（layer1~3 的基本单元）

```jsonc
{
  "timestamp": "...", "scene": "...", "user_action": "...",
  "needs": [{"need": str, "confidence": float}],     // Part2
  "solutions": [{"need": str, "solution": str}],     // Part3
  "feedback": {"confirmed": false, "corrections": {}, "user_rating": null},  // Part4
  "highlighted": false, "layer": 1
}
```

### 2.3 画像 schema（layer6.profile，规格书 Part 2 骨架）

```python
"profile": {
    "demographics": {},       # name/age/gender/identity/living_region/occupation/education/social_relations
    "preferences": {},        # food / hobbies（AttrValue 列表）
    "frequent_locations": [], # 常去地点（AttrValue 列表）
    "behavior_patterns": {}   # with_surroundings / with_ar_system{common_apps,typical_behaviors} / with_agents
}
```

静态画像（以上四字段）落 layer6；动态状态（`status_inference` 情绪/专注度、`gaze_target`）**不进 layer6**，
由 moment 在 layer1~4 承载。白名单见 `PROFILE_FIELDS`（L49）。

### 2.4 AttrValue 统一属性结构

画像中**每个属性**都是统一结构（`_finalize_attr()` L201）：

```jsonc
{ "value": "...", "confidence": 0.85, "source": "moment_xxx", "evidence": "...",
  "timestamp": "...", "last_seen": "...", "observations": 3,
  "effective_confidence": 0.83, "stale": false }
```

### 2.5 9 种数据库操作（CRUD）

`PersonMemory`（L481）对外提供：

| # | 方法（行号） | 触发点 | 语义 |
|---|---|---|---|
| 1 | `add()` L559 | Phase A | 写新 moment（layer1/3/7 + 索引） |
| 2 | `update()` L1009 | 手动 | 更新 moment 的 people/location/notes |
| 3 | `query()` L745 | 检索 | tokenize 分词 + 命中数排序 + Top-K + 排除 stale |
| 4 | `retrieve()` L767 | 检索 | 按 person/location/activity 打分取 Top-K |
| 5 | `compress()` L813 | Phase C | 写 layer4/5/6.summary；**跳过 profile**（职责边界） |
| 6 | `sort()` L856 | Phase C | layer7 索引归类（带索引键校验） |
| 7 | `combine()` L882 | Phase C | 同类索引合并 |
| 8 | `delete()` L916 | 手动 | 删除 moment 及索引残留 |
| 9 | `highlight()` L938 | Phase B 确认 | 标记 confirmed-correct |

画像专用 API：`update_profile()` L709 / `get_profile()` L737（画像写入唯一入口，与 `compress()` 解耦）。
上下文组装：`get_context_for_analysis()` L951 / `get_all_memory()` L986。

---

## 三、核心机制

### 3.1 分层检索（early-stop，同步链路不含 LLM）

`get_context_for_analysis()`（L951）自顶向下取够即停：
`layer6 画像 → layer5 长期模式 → layer4 近期任务 → layer7 索引细节`。
硬约束：此链路不含 LLM 调用（实测 5000 条 query() p95=14.38ms vs LLM 10.68s）。

### 3.2 画像合并语义（修复浅覆盖丢历史）

| 场景 | 函数 | 语义 |
|---|---|---|
| 列表型维度 | `_merge_attr_list()` L250 | 追加去重（bigram Jaccard ≥0.85 佐证合并），绝不整体覆盖 |
| 单值冲突 | `_resolve_conflict()` L234 | 频次 > 最近 > 置信度，败者入 history |
| 同值重复 | `_corroborate_attr()` L224 | observations+1、confidence 增强、last_seen 更新 |
| 低置信 <0.3 | `_filter_low_confidence()` L266 | 不入库 |

### 3.3 时间衰减与遗忘

`effective_confidence()` L332 / `_apply_decay_and_stale()` L348：
`eff = confidence × (1 − 0.005)^days`；`eff < 0.3` → `stale=True`（不检索、不物理删除）。
当前为统一 0.5%/天，**待按属性类型差异化**（身份偏好类永久 / 经历状态类渐变 / 优惠券类阶跃，见 09-12 会议）。

### 3.4 索引清洗（G7）

`_is_valid_person_tag()` L98 / `_is_valid_index_tag()` L121 / `_clean_layer7_indices()` L146。
拒绝占位符、否定短语、>30 字符长句、动词分句；加载时自动清洗存量（`_load` L507）。

### 3.5 原子落盘

`_save()` L529：临时文件 + `os.replace` 原子替换，防并发写坏 / 写一半崩溃。

---

## 四、目录结构

```
code/
├── memory.py              # 本模块核心：PersonMemory（7 层）+ HintMemory
├── profile_extractor.py   # 画像抽取模块（Phase C 调用）
├── agent.py               # 主流程 VRAssistant（记忆读写调用方）
├── api_server.py          # 推理服务（FastAPI，/consolidate 供画像抽取）
└── memory/
    ├── memory.json        # 运行时记忆数据（不应提交，见 .gitignore）
    └── hints.json         # HintMemory 用户自定义规则

scripts/memory/            # 测试与工具（09-12 从 zhx/task2/scripts 迁入）
├── test_profile_extractor.py    # 画像抽取单测 42
├── test_profile_storage.py      # 分层存储单测 28
├── test_retrieval_decay.py      # 增量更新+检索单测 24
├── test_ddl2_regression.py      # 缺陷回归 59
├── bench_retrieval_baseline.py  # 检索压测
└── run_all_videos.py            # 全量视频端到端回归
```

---

## 五、zhx_dev 分支历次更改记录（供审查）

`zhx_dev` 相对 `main`（分叉点 `2fd7b2e`）共 4 个提交，净改动集中在 memory 相关模块。逐条说明：

### commit `361ec2a` — feat：新增用户画像抽取模块

- **文件**：`code/profile_extractor.py`（+283 行，全新文件）
- **改动**：按规格书 Part 2 骨架（demographics/preferences/frequent_locations/behavior_patterns）
  新增画像抽取模块，含 `build_profile_extraction_prompt`（抽取 prompt 构建）、
  `parse_profile_output`（JSON 多策略兜底解析）、`validate_profile`（低置信剔除 + 动态量拒绝）。

### commit `4cdc247` — fix：修复记忆整理线程被主进程杀死

- **文件**：`code/agent.py`（+22 / -11）
- **改动**：Phase C 的 `run_phase_c` 后台线程 `daemon=True → False`，脚本入口 `consolidation_blocking=True`，
  `_should_consolidate()` 触发条件澄清（首次 ≥3 才触发）。根因：一次性脚本主进程退出时 daemon 线程被杀，
  导致 layer6 画像从未生成。

### commit `1923ba2` — chore：gitignore 忽略个人工作目录

- **文件**：`.gitignore`（+1 行）
- **改动**：新增 `zhx/*` 忽略规则。

### commit `843ec57` — feat：画像分层存储与检索落地（Step 06-07）⚠️ 大提交

- **文件**：8 个文件，+1599 / -422。此提交体量较大，按功能拆解如下（审查时可对照下表）：

| 子功能 | 文件 | 关键改动 |
|---|---|---|
| 分层存储（画像 API + 合并语义） | `code/memory.py` | 新增 `_bigram_jaccard` / `_merge_attr_list` / `_resolve_conflict` / `_corroborate_attr` / `_filter_low_confidence` / `_finalize_attr` / `_migrate_profile` / `update_profile` / `get_profile`；`_empty_db` 的 layer6.profile 改为规格书骨架四字段 |
| 时间衰减与遗忘 | `code/memory.py` | 新增 `effective_confidence` / `_apply_decay_and_stale` / `_has_profile_content` / `_collect_attrs` |
| 中文分词 | `code/memory.py` | 新增 `tokenize`（字符 bigram，修复中文 0 命中） |
| 分层检索 | `code/memory.py` | `query` / `retrieve` 改相关性打分 + Top-K + 排除 stale；`get_context_for_analysis` early-stop |
| 索引清洗 | `code/memory.py` | 新增 `_is_valid_person_tag` / `_is_valid_index_tag` / `_clean_layer7_indices` / `clean_layer7_indices` |
| 画像抽取集成 | `code/agent.py` | consolidation prompt 换规格书骨架；`_consolidation_worker` 抽 profile → `update_profile()` 独立落库 |
| Phase A 检索接入 | `code/agent.py` | `get_all_memory()` 全量 dump → `get_context_for_analysis()` 最小够用上下文 |
| 环境适配 | `code/api_server.py` | transformers 类名 / QWEN_MODEL_PATH / PORT / whisper 本地缓存（7 处） |
| 运行时产物 | `code/memory/memory.json`、`code/output/*` | ⚠️ 运行数据不应入库（09-12 已在 .gitignore 清理，见下） |

### 未提交工作区（09-12 目录重组，待提交）

- `.gitignore`：新增 `scripts/memory/archive/`、`*.log`、`_archive/`、`history/`、`logs/`、`code/output` 二进制产物等忽略规则。
- `scripts/memory/`：4 套测试脚本 + 压测 + 回归脚本（从 `zhx/task2/scripts/` 迁入）。
- `code/memory/README.md`：本文档。

---

## 六、测试

| 测试 | 脚本 | 覆盖 | 结果 |
|---|---|---|---|
| 画像抽取 | `scripts/memory/test_profile_extractor.py` | 骨架字段/低置信剔除/静态动态分存/解析兜底/触发条件/prompt | 42/42 |
| 分层存储 | `scripts/memory/test_profile_storage.py` | 读写/追加去重/冲突裁决/佐证合并/旧数据迁移/职责边界/原子落盘 | 28/28 |
| 增量更新+检索 | `scripts/memory/test_retrieval_decay.py` | bigram 分词/排序/early-stop/衰减/stale/冷启动 | 24/24 |
| 缺陷回归 | `scripts/memory/test_ddl2_regression.py` | D1/D2/D3 + G1/G5/G6/G7 专项 | 59/59 |
| 端到端 | `scripts/memory/run_all_videos.py` | 35 视频全链路 | 35/35 |

运行：

```bash
cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
/data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python test_profile_storage.py
```

---

## 七、已修复缺陷清单（09-03 实证）

| 编号 | 缺陷 | 修复点 |
|---|---|---|
| D1 | Phase C daemon 线程被杀 → 画像从未生成 | `agent.py` run_phase_c / process |
| D2 | Phase A 假检索（全量 dump） | `memory.py` get_context_for_analysis + agent.py 接入 |
| D3 | compress 浅覆盖 → 画像丢历史 | `memory.py` update_profile 独立 API + 合并语义 |
| G1 | 中文检索 0 命中 | `memory.py` tokenize 字符 bigram |
| G5 | retrieve 硬截断无排序 | `memory.py` retrieve 相关性打分 + Top-K |
| G6 | 上下文字符截断/画像不注入 | `memory.py` early-stop + `_collect_attrs` 展平 |
| G7 | layer7 索引污染 | `memory.py` 三重校验 + agent.py parse_analysis 源头过滤 |

## 八、下一步（09-12 会议）

1. CRUD 完备化（update/combine/delete 按新 schema 完善 + 补测试）。
2. 遗忘机制差异化（三类衰减曲线：身份偏好类 / 经历状态类 / 优惠券类）。
3. EgoLife 测试集 + 三指标验收（存储 ≤100GB / 提取 ≤1s / 准确率 ≥95%）。
