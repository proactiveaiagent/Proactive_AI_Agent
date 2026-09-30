# 记忆模型体系设计说明书（v1）

> 版本：v1.1 ｜ 日期：2026-09-14 ｜ 状态：定稿（供开发与代码审查使用）
>
> v1.1 补全：§2.4 层间晋升与跨天清理定稿、§4.2 update 完整语义、§4.4 combine 完整语义、
> §4.5 delete/highlight 完整语义、§8 存储容量预算（≤100GB）、§9 检索性能设计（≤1s）。
>
> 范围：7 层记忆模型的**体系设计**——分层定义、字段字典、CRUD 语义、遗忘机制（三类衰减曲线），以及依赖记忆证据的语义检索与行动建议接口。
> 记忆基础查询保持无 LLM 的快速路径；语义规划、关联检索和建议生成属于显式的上层决策路径，不替代记忆 CRUD，也不自动执行行动。
>
> 配套文档：`code/memory/README.md`（架构说明 + 变更记录，供代码审查）、
> `zhx/task2/report.md`（任务说明书）、`zhx/task2/todo.md`（工作计划）。
> 代码位置：`code/memory.py`（核心实现）、`code/profile_extractor.py`（画像抽取）。

---

## 一、设计总纲

### 1.1 目标与验收（月末，EgoLife 测试集）

| 指标 | 目标 |
|---|---|
| 存储容量 | ≤ 100GB（硬上限 200GB） |
| 提取速度 | ≤ 1s（检索/提取关键路径） |
| 准确率 | ≥ 95%（归类准确，无类别混淆） |

> 评测口径复核（2026-09-23）：≥95% 的基础类别归类准确性与记忆检索 P@5 是不同指标。EgoLife 检索 P@5=97.5% 只证明检索相关性达标，不能替代类别归类验收。现有自动标签的类别本体和覆盖存在差异；Qwen 评测的人物 F1 为 97.1%，地点/活动/环境/物品 F1 分别为 14.5%/0%/5.5%/12.4%，这些分数也不能作为可靠的真实归类准确率。人工复核 schema 与 725 个片段的盲审模板现已准备（详见 docs/task2-category-annotation-schema.md），但人工标签尚未产出，须另行验收。

### 1.2 四条设计原则

1. **分层压缩**：原始 moment（layer1~3）经 Phase C 压缩成摘要（layer4/5）与画像（layer6），
   layer7 只做分类索引。热数据小、冷数据归档，控制存储与检索成本。
2. **静态 / 动态分存**：静态稳定量（demographics / preferences / frequent_locations /
   behavior_patterns）落 layer6；动态状态（情绪 / 专注度 / 视线）由 moment 在 layer1~4 承载，
   不进画像。
3. **按需检索，取够即停**：检索自顶向下（画像 → 摘要 → 索引）逐层取，同步链路**不含 LLM 调用**；
   画像生成/精化走 Phase C 异步通路。
4. **一切属性可溯源、会遗忘**：每个属性带置信度、来源、时间戳（AttrValue）；按属性类型
   差异化衰减（三类曲线，见第五节），陈旧数据标记后退出检索但不物理删除。

### 1.3 与规格书 / 方法论的关系

- 画像字段骨架唯一来源：规格书 `proactive_ar_agent.md` Part 2（L96-102）。
- 方法论支撑：MWM（心理世界模型）三条论断——「心理状态 = 相关历史的压缩摘要」支撑分层压缩；
  「推断变量不必等于真值」支撑置信度元字段；「第一人称部分观察」支撑按需检索、否定全量塞。
- 规格书没有的维度（信念 / 社会规范 / 行为约束）一律不引入。

---

## 二、7 层分层定义（定稿）

实现：`code/memory.py::_empty_db()`（L423）。单 JSON 文件 `code/memory/memory.json`，顶层 8 个键。

### 2.1 总览

| 层 | 名称 | 存储内容 | 写入时机 | 触发者 |
|---|---|---|---|---|
| layer1 | 当前时刻 | 本次交互原始 moment | 每次交互 | Phase A `add()` |
| layer2 | 同场景历史 | 同 scene 的历史 moment | layer1 溢出毕业 | `add()` 内部 |
| layer3 | 当日全部 | 当天所有 moment | 每次交互 | Phase A `add()` |
| layer4 | 近期摘要 | 日级摘要 / 当前任务 / 生活动线 | 整理时 | Phase C `compress()` |
| layer5 | 远期摘要 | 周/月级摘要 / 关键事件 / 长期模式 | 整理时 | Phase C `compress()` |
| layer6 | 用户画像 ⭐ | 静态画像（8 字段，见 §3.3） | 整理时 | Phase C → `update_profile()` |
| layer7 | 分类索引归档 | 6 个倒排索引 + moments 主存储 | 每次交互 + 整理时 | `add()` / `sort()` / `combine()` |

### 2.1.1 层级规范（09-22 导师文档定稿）

1. **层级 1~5 由近到远**：短期（当前）→ 中期（近期）→ 长期（远期）。
2. **每一层内的记忆按时间远近顺序排列**（检索/渲染时按时间倒序：最近优先）。
3. **查询按层级先后 1→2→3→4→5 提取**（见 §4.3 读取类）。
4. **特别标注两类记忆**（`highlight`）：
   - 用户在 Part4 确认无误的数据；
   - 用户明确标记为重点的数据。标签归并（`combine`）只规范索引，不代表用户确认，也不会自动标记重点。
5. **layer2 归类依据是「环境场景」**：按 Part1 识别的 `environments` 判定同场景（有交集即同场景）；
   无 `environments` 标注时降级为 `location + activity` 判定（兼容旧数据）。

### 2.2 逐层定义

**Layer 1 · 当前时刻（瞬时）**

| 项 | 定稿 |
|---|---|
| 存储内容 | 当前时刻原始观察（scene / user_action / needs / solutions = 一个 moment） |
| 写入时机 | 每次交互（Phase A `memory.add`） |
| 保留时长【推导】 | 1 小时 |
| 窗口容量 | 5 条（`MAX_LAYER1`，超出时最旧滑出） |
| 滑出去向 | 与当前 scene 同场景（location+activity 相同）→ layer2；否则 → layer3 |
| 淘汰 | 不丢数据（真身在 layer7.moments，滑出只是移出窗口） |

**Layer 2 · 同场景历史（瞬时）**

| 项 | 定稿 |
|---|---|
| 存储内容 | 当前场景（location + activity 相同）的最近历史 moment |
| 写入时机 | layer1 滑出（同场景判定） |
| 窗口容量 | 20 条（`MAX_LAYER2`，超出时最旧滑入 layer3） |
| 淘汰 | 不丢数据（滑入 layer3 窗口，真身仍在 layer7.moments） |

> 注：09-03 实测 layer2 为 0 条（旧实现「滑出即进 layer2、无同场景判定」）。09-15 已在
> `_graduate_to_layer2` 落地同场景判定（location+activity 相同才进 layer2，否则直接进 layer3）。

**Layer 3 · 当日窗口（瞬时）**

| 项 | 定稿 |
|---|---|
| 存储内容 | 当日最近的 moment 窗口 |
| 写入时机 | 每次 add 同步追加；layer2 滑出 |
| 窗口容量 | 1000 条（`MAX_LAYER3`；Layer 7 永久归档不受此窗口上限影响） |
| 跨天清理 | 新一天首次 add 时清空窗口（数据仍在 layer7.moments，见 §2.4） |
| 压缩 | 不依赖 layer3：Phase C 从 layer7.moments 生成 layer4/5/6 |

**Layer 4 · 近期摘要**

| 项 | 定稿 |
|---|---|
| 存储内容 | 日级摘要：`summary` + `current_tasks`（当前任务）+ `life_trajectory`（生活动线） |
| 写入时机 | Phase C `compress()`（每 3 条 moment 触发一次整理） |
| 保留时长【推导】 | 30 天 |
| 晋升 | 满 30 天 → 归并入 layer5 |
| 淘汰 | 归并后原层清空 |

**Layer 5 · 远期摘要**

| 项 | 定稿 |
|---|---|
| 存储内容 | 周/月级摘要：`summary` + `key_events`（关键事件）+ `long_term_patterns`（长期模式） |
| 写入时机 | Phase C `compress()` |
| 保留时长【推导】 | 90 天 |
| 晋升 | 其中的稳定量经裁决后晋升 layer6（走画像合并语义） |
| 降级 | 稳定量被更高置信新证据推翻 → 降回 layer5 重估 |
| 淘汰 | 非稳定量随摘要过期淘汰 |

**Layer 6 · 用户画像（核心层）**

| 项 | 定稿 |
|---|---|
| 存储内容 | 静态画像四字段（`demographics` / `preferences` / `frequent_locations` / `behavior_patterns`），字段字典见 §3.3 |
| 写入时机 | Phase C 整理时，**必须**走 `update_profile()`（画像唯一写入入口，禁止 `compress()` 直写） |
| 保留时长 | 永久 |
| 降级 | 有效置信度衰减至阈值以下 → 标记 `stale`，退出检索（见 §5 遗忘机制） |
| 淘汰 | 不物理删除；stale 数据保留供审计与历史分析 |

**Layer 7 · 分类索引归档**

| 子结构 | 键 | 值 | 检索键 |
|---|---|---|---|
| `moments` | moment_id | moment dict | moment_id（主存储） |
| `people` | 人名 | [moment_id] | 人名 |
| `locations` | 地点名 | [moment_id] | 地点名 |
| `activity_events` | 活动类型 | [moment_id] | 活动类型 |
| `time_nodes` | 日期 | [moment_id] | 日期 |
| `environments` | 环境场景 | [moment_id] | 环境标签 |
| `objects` | 物品 | [moment_id] | 物品标签 |

写入时机：每次交互（`add()` 同步更新）+ 整理时（`sort()` / `combine()`）。
索引键写入前必须过校验（防污染，见 §4.4）；加载时自动清洗存量污染。

### 2.3 层间数据流

```
moment 到达 ──add()──► layer1 ──溢出──► layer2 ──晋升──► layer3
                                                        │ Phase C 触发（每 3 条）
                       compress() ─────► layer4（近期摘要）─► layer5（远期摘要）
                       update_profile() ► layer6（画像，走合并语义 + 遗忘）
                       sort()/combine() ► layer7（索引）
检索：layer6 画像 → layer5 → layer4 → layer7 索引（取够即停）
```

### 2.4 层间晋升与跨天清理（定稿）

**核心原则（三条）**：

1. **窗口视图模型**：layer1/2/3 是「窗口视图」而非数据容器——数据真身唯一存放在
   layer7.moments（主存储），窗口只是对最近 N 条的视图。晋升 = 窗口滑动，**不是数据的物理移动**，
   因此任何窗口滑出都不丢数据。
2. **惰性执行**：记忆系统是被动存储（单 JSON 文件、单进程），**不引入后台定时器**。
   所有晋升/清理在写操作（add / compress）时顺带执行，避免与 ≤1s 检索竞争资源。
3. **容量上限为硬约束**：每个窗口有容量上限（MAX），超出即滑动；时间语义（当日/近期）
   由「跨天清理」和「Phase C 压缩」承载，不靠定时器逐条判定。

**晋升规则表**：

| 事件 | 触发时机 | 动作 |
|---|---|---|
| layer1 超限（>5） | add() 内 | 最旧滑出：同场景（location+activity 相同）→ layer2；否则 → layer3 |
| layer2 超限（>20） | add() 内 | 最旧滑入 layer3 窗口 |
| layer3 超限（>1000） | add() 内 | 最旧滑出窗口（真身保留在 layer7.moments） |
| 跨天（today 变化） | add() 内 | 清空 layer1/2/3 窗口，更新 metadata.today（layer7.time_nodes 历史保留） |
| 压缩（Phase C） | 每 3 条 moment | 从 **layer7.moments** 生成 layer4/5/6，不依赖 layer3 窗口 |

**与现状代码的差距**（9.15 CRUD 完备化落地）：

| # | 现状 | 定稿 |
|---|---|---|
| 1 | `_graduate_to_layer2` 无同场景判定（所有滑出都进 layer2） | 补 location+activity 判定，不同场景直接进 layer3 |
| 2 | layer2 超限 pop 丢弃 | 改为滑入 layer3 |
| 3 | MAX_LAYER3=100 | 上调 1000（EgoLife 实测校准） |
| 4 | 跨天无清理（layer3 持续累积） | add 时检测 today 变化，清空窗口 |
| 5 | update() 是旧版 shim（实为 add） | 重定义为 moment 字段更新操作（见 §4.2） |

---

## 三、字段字典

### 3.1 moment 字段（layer1~3 基本单元）

实现：`_empty_moment()`（L401）。

| 字段 | 类型 | 含义 | 备注 |
|---|---|---|---|
| `timestamp` | str(ISO) | 记录时间 | |
| `scene` | str | 外部环境描述（Part1） | |
| `user_action` | str | 用户行为（Part1） | |
| `source_evidence` | str | 与本片段 ID 精确匹配的英文、带说话人标签转写证据 | 独立于摘要；缺少精确来源时留空；默认不拼入回答上下文 |
| `source_provenance` | dict | 来源视频相对 ID、片段 ID、片段起止秒数、抽取批次 ID | 不记录服务器绝对路径；文本来源字段与多模态资产使用同一来源范围 |
| `multimodal` | dict | 受预算约束的画面、音频特征与实体字段 | 资产按内容哈希去重；缺少媒体时保留文本 moment |
| `needs` | list | 需求 `[{"need": str, "confidence": float}]`（Part2） | |
| `solutions` | list | 方案 `[{"need": str, "solution": str}]`（Part3） | |
| `feedback` | dict | `{confirmed, corrections, user_rating}`（Part4，后填） | 不阻塞 Part1-3 |
| `highlighted` | bool | 记忆受重点保护并在检索排序中提权 | 仅显式 `highlight()` 或 `update_feedback(confirmed=True)` 置位 |
| `highlight_source` | str | 最近一次显式标记入口 | `highlight_api` / `feedback_confirmed`；旧记录可能缺失 |
| `highlighted_at` | str(ISO) | 最近一次显式标记时间 | 旧记录可能缺失 |
| `layer` | int | 所在层 | 1~3 |

### 3.2 layer4 / layer5 字段

| 层 | 字段 | 类型 | 含义 |
|---|---|---|---|
| layer4 | `summary` | str | 近期状态摘要 |
| | `current_tasks` | list[str] | 用户当前任务 |
| | `life_trajectory` | str | 近期生活动线 |
| | `last_updated` / `source_moment_count` | str / int | 元信息 |
| layer5 | `summary` | str | 远期状态摘要 |
| | `key_events` | list | 关键事件 |
| | `long_term_patterns` | str | 长期稳定模式 |
| | `last_updated` / `source_moment_count` | str / int | 元信息 |

### 3.3 layer6 画像字段字典（含衰减策略）

| 字段路径 | 类型 | 含义 | 衰减策略（§5） |
|---|---|---|---|
| `demographics.name` | AttrValue | 姓名 | 稳定-永久（decay=0） |
| `demographics.gender` | AttrValue | 性别 | 稳定-永久（decay=0） |
| `demographics.identity` | AttrValue | 身份 | 稳定-永久（decay=0） |
| `demographics.education` | AttrValue | 受教育 | 稳定-永久（decay=0） |
| `demographics.age` | AttrValue | 年龄 | 稳定-长有效（decay=0.001） |
| `demographics.living_region` | AttrValue | 居住区域 | 稳定-长有效（decay=0.001） |
| `demographics.occupation` | AttrValue | 职业 | 渐变（decay=0.005，可能随时间变化） |
| `demographics.social_relations` | AttrValue[] | 社会关系 | 稳定-长有效（decay=0.001） |
| `preferences.food` | AttrValue[] | 饮食偏好 | 稳定-长有效（decay=0.001） |
| `preferences.hobbies` | AttrValue[] | 兴趣爱好 | 稳定-长有效（decay=0.001） |
| `frequent_locations` | AttrValue[] | 常去地点 | 稳定-长有效（decay=0.001） |
| `behavior_patterns.with_surroundings` | AttrValue[] | 与环境交互习惯 | 稳定-长有效（decay=0.001） |
| `behavior_patterns.with_ar_system.common_apps` | AttrValue[] | 常用应用 | 稳定-长有效（decay=0.001） |
| `behavior_patterns.with_ar_system.typical_behaviors` | AttrValue[] | 典型行为 | 稳定-长有效（decay=0.001） |
| `behavior_patterns.with_agents` | AttrValue[] | 与智能体交互习惯 | 稳定-长有效（decay=0.001） |
| `personality.traits` | AttrValue[] | **性格特点**（09-22 新增） | 稳定-永久（decay=0） |
| `goals` | AttrValue[] | **规划目标**（09-22 新增） | 渐变（decay=0.005，目标会变化） |
| `decisions.patterns` / `decisions.key_factors` | AttrValue[] | **选择决策**模式与影响因素（09-22 新增） | 稳定-长有效（decay=0.001） |
| `motivations` | AttrValue[] | **动机**（09-22 新增） | 稳定-长有效（decay=0.001） |
| （新）经历状态类字段 | AttrValue | 上学阶段等阶段类属性 | 渐变（decay=0.005） |
| （新）时效事件类字段 | AttrValue | 优惠券/会员等 | 截止（expires_at 阶跃） |

**画像排序规则（09-22 规范「用户的自身画像」6 类）**：渲染/检索时按字段语义排序——

| 类别 | 字段 | 排序规则 |
|---|---|---|
| 行为习惯 | `behavior_patterns` | 频率从高到低（`observations` 降序） |
| 性格特点 | `personality` | 确定性从高到低（`confidence` 降序） |
| 规划目标 | `goals` | 重要性从高到低（`confidence` 代理） |
| 偏好爱好 | `preferences` | 从强到弱（`confidence` 降序） |
| 选择决策 | `decisions` | 影响因素从高到低（`confidence` 降序） |
| 动机 | `motivations` | 从强到弱（`confidence` 降序） |

> 动态状态（`status_inference` 情绪/专注度、`gaze_target`、`observable_behaviors`）**不进 layer6**，
> 由 moment 在 layer1~4 承载（规格书 L96 本就将它们定义在「用户状态行动」里）。

### 3.4 AttrValue 统一属性结构（v1 扩展）

每个画像属性都是统一结构。**v1 新增 `decay_type` / `expires_at` 两个字段**（§5 遗忘机制用），
旧数据无这两字段时按字段路径默认值兼容。

```jsonc
{
  "value":              "spicy",             // 属性值
  "confidence":         0.85,                // 置信度 0~1
  "source":             "moment_000123",     // 来源（moment_id / "consolidation" / "legacy"）
  "evidence":           "餐后要求加辣",        // 抽取依据（供人工核对）
  "timestamp":          "2026-09-04T10:00",  // 首次观察时间
  "last_seen":          "2026-09-10T18:30",  // 最近佐证时间（渐变型衰减依据）
  "observations":       5,                   // 佐证次数（冲突裁决"频次"依据）
  "decay_type":         "stable",            // v1 新增：stable / decaying / deadline
  "expires_at":         null,                // v1 新增：deadline 型专用，过期时间 ISO
  "effective_confidence": 0.83,              // 有效置信度（按 decay_type 计算）
  "stale":              false                // 是否陈旧（不参与检索，不物理删除）
}
```

### 3.5 layer7 索引与 metadata

| 键 | 类型 | 约束 |
|---|---|---|
| `moments` | dict | moment_id → moment dict（主存储） |
| `people` / `locations` / `time_nodes` | dict | **外在内容（从近到远排序）**：人物 / 空间位置 / 时间节点（日周月年）；索引键 → [moment_id]，键写入前过 `_is_valid_index_tag` 校验 |
| `activity_events` / `environments` / `objects` | dict | **外在内容（从高频到低频排序）**：动作事件 / 环境场景 / 物品；索引键 → [moment_id] |
| `metadata.total_moments` | int | 累计 moment 数（ID 生成与整理触发依据） |

**索引维度定义（09-22 规范「存储的外在内容」6 类）**：

| 维度 | 含义 | 排序规则 |
|---|---|---|
| `people` | 人物（亲朋好友等） | 从近到远（最近出现优先） |
| `locations` | 空间位置（家/办公室/学校等） | 从近到远 |
| `time_nodes` | 时间节点（日/周/月/年） | 从近到远 |
| `activity_events` | 活动事件（吃喝/学习/工作/生活等） | 从高频到低频 |
| `environments` | 环境场景（在什么情况下） | 从高频到低频 |
| `objects` | 物品 | 从高频到低频 |

排序在**检索/渲染时动态计算**（`_sorted_index_tags`），存储保持简单（索引存 moment_id 集合）。
| `metadata.total_moments` | int | 累计 moment 数（ID 生成与整理触发依据） |
| `metadata.last_consolidation` | str/None | 最近整理时间（触发条件依据） |
| `metadata.session_start` / `today` | str | 会话元信息 |

实体关系不在每条 moment 外重复保存一份常驻图结构。读取与可视化时由 L7 原文、六维索引和多模态实体字段即时构造类型化关系，并为每条关系保留来源 `moment_id`。例如，“Alice 参与事件”“事件发生在厨房”“事件包含水杯”；两个人同一片段出现只表示共同参与该事件，不推断私人关系。这样可以按实体关联检索，同时避免图节点和索引重复占用长期容量。

---

## 四、CRUD 语义定义（定稿）

`PersonMemory` 对外 9 操作 + 画像 API。每个操作给出：签名、触发者、精确语义、落盘行为。

### 4.1 操作总览

| # | 操作 | 方法 | 触发者 | 状态 |
|---|---|---|---|---|
| 1 | add | `add()` L559 | Phase A | ✅ |
| 2 | update | `update()` L1009 | 手动 | ✅（09-15 落地：moment 字段更新 + 索引联动） |
| 3 | query | `query()` L745 | 检索 | ✅ |
| 4 | retrieve | `retrieve()` L767 | 检索 | ✅ |
| 5 | compress | `compress()` L813 | Phase C | ✅ |
| 6 | sort | `sort()` L856 | Phase C | ✅ |
| 7 | combine | combine() | Phase C | ✅ 索引标签归一化 |
| 8 | find_merge_candidates | find_merge_candidates() | 交互页扫描 | ✅ 只生成候选，不改写记忆 |
| 9 | merge_moments | merge_moments() | 用户逐条确认 | ✅ 非破坏性事件归并，来源留档 |
| 10 | delete | delete() | 手动 | ✅（重点记忆默认保护） |
| 11 | highlight | highlight() | Phase B 确认 | ✅ |
| — | update_profile | L709 | Phase C | ✅（画像唯一写入口） |
| — | get_profile | L737 | 任意 | ✅ |
| — | get_context_for_analysis | L951 | Phase A | ✅（按需检索入口） |

### 4.2 写入类

**add(scene, user_action, needs, solutions, people, location, activity, extra_notes, source_evidence) → moment_id**
1. 生成 moment_id（`_new_moment_id`，时间戳 + 累计序号）；
2. 写 layer1（当前窗口）；
3. layer1 溢出 → `_graduate_to_layer2` 晋升 layer2（同 scene 聚合）；
4. 写 layer3（当日全部）；
5. 写 layer7：moments 主存储 + people/locations/activity_events/time_nodes 四个索引
   （索引键过 `_is_valid_index_tag` 校验，防污染）；
6. metadata.total_moments + 1；原子落盘。

**update_feedback(moment_id, corrections, rating, confirmed)**：把 Part4 反馈写回
layer7.moments 对应 moment 的 `feedback` 字段；corrections 中允许的字段直接改正 moment 本体
（Part1-3 数据），并同步 layer1/2/3 中的同 id 条目。

**update(moment_id, updates) → bool**（定稿，9.15 落地）
- **语义**：改正指定 moment 的 Part1-3 数据（与 `update_feedback` 的分工：update 改数据，
  update_feedback 改反馈）。
- **允许字段白名单**：`scene` / `user_action` / `needs` / `solutions` / `people` / `location` /
  `activity` / `notes` / `source_evidence`；禁止改 `id` / `timestamp` / `feedback` / `highlighted` / `layer`（越权字段忽略）。
- **同步**：修改 layer7.moments 主存储后，同步 layer1/2/3 中同 id 条目（JSON 落盘重载后引用断开，
  需防御同步）。
- **索引联动**：若 `people` / `location` / `activity` 变更，同步增删 layer7 对应索引的 moment_id
  （新键过 `_is_valid_index_tag` 校验）。
- **返回**：moment 存在且至少一个合法字段被更新 → True；否则 False。
- **旧签名**：`update(people, location, notes)` 无调用方（已确认 `agent.py` 零调用），直接替换，
  不做兼容层。

**update_profile(extracted, moment_id, timestamp) → profile**（画像写入唯一入口）
1. **白名单过滤**：只接受 `PROFILE_FIELDS` 四字段；
2. **低置信丢弃**：confidence < 0.3 不入库；
3. **递归合并**：逐字段按合并语义（§4.6）并入现有画像；
4. **遗忘处理**：重算有效置信度 + 陈旧标记（§5）；
5. **原子落盘**。
约束：`compress()` 禁止写 layer6.profile（职责边界，违反时打印警告并跳过）。

### 4.3 读取类

**query(query_text, top_k=5) → List[moment]**
统一英文分词（§6.2）→ 对 layer7.moments 全文做 token 命中数打分 → 按分数降序取 Top-K →
跳过 stale。返回 moment dict 列表。

**retrieve(people, location, activity, layer, top_k=5) → str**
拼 layer4/5/6 摘要（可选按 layer 过滤）+ 对 layer7.moments 按 person/location/activity
关键词打分排序取 Top-K，输出格式化文本；无结果返回 "No relevant memory found."。

**get_context_for_analysis(people, location, ...) → str**（Phase A 检索入口）
按 §6.1 的 early-stop 路径组装最小够用上下文；冷启动（空画像）不注入画像段。

### 4.4 整理类（Phase C）

**compress(llm_summary)**：把 LLM 整理的 layer4/5/6.summary 写入对应层；
list 型字段追加去重（`json.dumps(sort_keys=True)` 精确比较，支持 str 与嵌套 dict 元素）；
**跳过 layer6.profile**（打印警告，画像走 update_profile）；**空值保护**（空字符串/None 不覆盖已有值，
防 LLM 返回空摘要丢历史）；更新 last_consolidation；落盘。

**sort(sort_analysis)**：把 LLM 归类的 layer7 索引（people/locations/activity_events/time_nodes）
写入；每个索引键过 `_is_valid_index_tag` 校验（修复 G7 污染）；**悬挂 moment_id 过滤**（不存在的 id
不写索引）；同键 moment_id 集合去重合并；落盘。

**combine(canonical_map)**（定稿补全，9.17 落地 + 补测试）
- **canonical_map 来源（双通道）**：① Phase C LLM 输出——识别近义索引键对（如
  "LAX" → "Los Angeles International Airport"）；② 规则兜底——大小写/空白规范化后的
  精确重复检测（不引入语义推断）。
- **合并规则**：对每个 `old_key → canonical`：校验 canonical 过 `_is_valid_index_tag`
  （非法则跳过）；old_key ≠ canonical 且存在 → 把 old_key 的 moment_id 集并入 canonical，
  删除 old_key；同步更新 moments 主存储中引用字段：
  `locations → location`（str 单值）、`activity_events → activity`（str 单值）、
  `people → people`、`environments → environments`、`objects → objects`（列表元素替换并按首次出现顺序去重）。
- **稳定与可追溯**：索引 moment_id 合并保留规范键原有顺序，再追加旧键中尚未出现的 ID；不依赖无序集合生成输出。
  每条受影响的 moment 追加 `canonicalization_history`（维度、原标签、规范标签、时间），并同步仍在 L1–L3 的窗口副本。
  Layer 7 moment 原文和多模态引用不会因标签归并而删除；操作日志显示每组映射、受影响记录示例 ID、总数、两索引共有条数和字段更新数。
- **重点标记边界**：`combine` 不修改 `highlighted` 或其来源字段。只有显式调用 `highlight()` 或 `update_feedback(confirmed=True)` 才会设置重点，并记录触发入口和时间；标签相似或索引归并不等同于用户确认。旧记录缺少来源时不得推断或自动撤销。
- **幂等**：重复执行不产生副作用（old_key 已不存在或等于 canonical 则跳过）。
- **不处理**：画像内的属性合并（那是 `update_profile` 的合并语义），combine 只管 layer7 索引键。

### 4.5 相似事件候选与显式归并

find_merge_candidates(limit=30, max_time_delta_s=120) 只生成待审阅候选，不自动更改数据。候选需满足以下路径之一：① 时间相差不超过 120 秒、共享人物或物品，且 Scene 与 Action 合并文本的相似度至少 0.92；② 同一来源视频的片段重叠比例至少 0.80。画面/音频相似度只作为人工核对证据，不单独触发候选，因为相似背景声或静止画面不能证明事件重复。候选显示时间差、共享实体、片段重叠和可用模态分值。该筛选只覆盖共享实体/时间桶或相同视频来源的候选，不是全库语义聚类；分数是相似度，不是“重复概率”。

merge_moments(source_id, target_id, confirmed_duplicate=True, reason=...) 只允许用户明确确认后执行。交互页的“相似事件候选”步骤提供两条记忆对照，点击“确认是同一事件并合并”才会调用该操作。confirmed_duplicate 未设置时拒绝修改；规范记录不得是另一个合并组的来源。

- 不删除来源：source 的 L7 原文、时间、标签、来源证据和多模态引用都保留；source 标记 merged_into，规范记录登记 merged_sources。归并关系、理由、时间、字段冲突、归并前索引/窗口归属和多模态保留情况写入 merge_history。
- 撤销归并：页面提供“撤销本次归并”入口，只能按目标记录的最近归并顺序撤销。操作会恢复来源原有的归并关系、索引成员与 L1–L3 窗口位置，并恢复归并前列表字段；来源记录及其多模态资产始终保留。撤销会在原审计事件上追加撤销时间。若归并后目标字段又被修改、关系链发生变化，或旧归并缺少完整撤销快照，系统拒绝撤销并说明原因，不覆盖后续改动。
- 字段规则：人物、物品、环境、needs、solutions 列表按稳定顺序精确去重并入规范记录；Scene、Action、地点、活动和时间等单值字段不覆盖。冲突两侧原文留在 source，并在规范记录的审计记录中并列展示。
- 索引与窗口：六维 L7 索引把 source ID 替换为规范 ID 并去重；L1–L3 移除重复 source 副本，必要时保留规范记录。L7 总存储条数不因归并减少，索引与可检索事件数可以减少。
- 检索与多模态：默认结果只返回规范事件一次，但会把来源事件正文作为附属证据用于检索和推理；重点标记不会被移到规范记录，只作为组级排序信号。source 仍持有原画面/音频资产，因此事件归并不会腾出多模态容量。
- 评测边界：候选门限与误合并率尚未通过全量评测；容量、检索延迟和问答结果不能因归并而被推断达标。

### 4.6 删除与标记

**delete(moment_id, reason="manual", force=False) → bool**（定稿补全，9.15 落地 + 测试覆盖）
- **清理范围**：moments 主存储条目 + 四个索引（people/locations/activity_events/time_nodes）中
  的 moment_id 引用 + layer1/2/3 中的该 moment 条目。
- **highlight 保护（定稿新增）**：`highlighted=True` 的 moment 是用户确认正确的高价值数据，
  默认**拒绝删除**并返回 False；`force=True` 才允许（对应"错误标记需强制纠正"场景）。
- **metadata 语义**：`total_moments` **不减**——moment_id 由累计序号生成，保留计数可防止
  新 moment 与已删除 moment 的 ID 复用冲突；删除仅影响存储与索引。
- **返回**：删除成功 → True；moment 不存在或被 highlight 保护拒绝 → False。
- **语义边界**：delete 是「单条移除」，批量清扫（如压缩后去重）走 v2 的 sweep 接口。

**highlight(moment_id) → bool**：把 moment 标记为 confirmed-correct（`highlighted=True`）；
效果：① 删除清扫中受保护（见上）；② 检索排序中提权（v2：highlight 条目在同等命中数下优先）；
③ 画像更新的高权重佐证来源（v2）。moment 不存在返回 False。

### 4.7 画像合并语义（精确规则）

实现：`_merge_profile_node`（L678）及配套工具。

| 场景 | 规则 |
|---|---|
| 列表型维度（food/hobbies/地点/行为/社会关系） | **追加去重**：新值与旧值相似度 ≥ 0.85（英文单词字符 bigram Jaccard）→ 佐证合并；否则追加。绝不整体覆盖（09-03 缺陷 D3 的修复） |
| 单值 AttrValue 冲突（如 age） | **裁决顺序：频次 > 最近 > 置信度**；败者入 history 留痕 |
| 同值重复观察 | **佐证合并**：observations +1、confidence 增强（上限 0.999）、last_seen 更新 |
| 低置信（< 0.3） | 不入库 |
| 新值到达时 | 先过 `_finalize_attr` 规范化（补 source/timestamp/last_seen、observations=1），再过置信度门槛 |


单值冲突的 history 项保留败者原值，并附 resolution：胜出值、胜出来源、裁决时间及依据（观测次数、最近观察、置信度或平局保留旧值）。若观测次数和观察时间相同，比较置信度；仍相同则保留已有值。交互日志只报告本次新增的裁决，L6 详情可展开查看旧值与当前值。旧格式历史没有裁决字段时，界面会注明依据未记录，不补造历史结论。

---

## 五、遗忘机制：三类衰减曲线（v1 定稿）

### 5.1 设计动机（09-12 会议）

统一 0.5%/天的一刀切衰减不合理：性别不该衰减、上学阶段该慢慢淡出、优惠券过期应瞬间失效。
定稿为**按属性类型差异化**的三类曲线，目标演进为「Agent 根据属性类型自主判断衰减策略」。

### 5.2 三类曲线定义

| 曲线 | 名称 | 语义 | 有效置信度计算 | 参数 |
|---|---|---|---|---|
| ① | 稳定型 `stable` | 身份偏好类：较长有效期或**永久有效** | `eff = confidence`（硬身份，decay=0）或 `eff = confidence × (1−0.001)^days`（偏好行为，半衰期 ≈ 1.9 年） | decay ∈ {0, 0.001} |
| ② | 渐变型 `decaying` | 经历状态类：随时间流逝逐渐衰减 | `eff = confidence × (1−0.005)^days`（半衰期 ≈ 139 天），days 按 `last_seen` 起算 | decay = 0.005 |
| ③ | 截止型 `deadline` | 优惠券类：过期前完全有效，过期后瞬间失效 | `now < expires_at` → `eff = confidence`；`now ≥ expires_at` → `eff = 0`（阶跃） | `expires_at`（ISO 时间） |

统一失效规则：`effective_confidence < 0.3` → `stale = True`（不参与检索、不注入上下文、
不物理删除，保留审计）。三类曲线仅改变"如何计算有效置信度"。

```python
def effective_confidence(attr, now=None):
    dt = attr.get("decay_type", _default_decay_type(attr))   # 默认按字段路径推断
    if dt == "deadline":
        return attr["confidence"] if now < attr["expires_at"] else 0.0
    daily = 0.0 if dt == "stable" else DAILY_DECAY[dt]       # stable 细分见字段字典
    days = max(0, (now - attr["last_seen"]).days)
    return attr["confidence"] * (1 - daily) ** days
```

### 5.3 属性分类与判定机制（v1）

**分类来源（两级，后者兜底）**：
1. **LLM 标注优先**：画像抽取 prompt 要求对每个属性输出 `decay_type`
   （stable / decaying / deadline），deadline 型必须附 `expires_at`；
2. **字段路径默认兜底**：LLM 未标注时按 §3.3 字段字典的衰减策略列取值。

**语义校正规则（写入时二次判定，向"自主判断"演进）**：
- 属性带明确过期时间（如优惠券到期、会员截止）→ 强制 `deadline` + `expires_at`；
- `demographics` 硬身份字段（name/gender/identity/education）→ 强制 `stable`（decay=0），
  防止被误标为 decaying；
- 其余按 LLM 标注或字段默认值。

### 5.4 落地计划（代码改动点）

| 改动 | 位置 |
|---|---|
| AttrValue 增加 `decay_type` / `expires_at`（可选字段，旧数据兼容） | `code/memory.py::_finalize_attr` |
| `effective_confidence` 按 decay_type 分派三类曲线 | `code/memory.py::effective_confidence` |
| 字段路径 → 默认 decay_type 映射表 + 语义校正 | `code/memory.py`（新函数） |
| 抽取 prompt 要求标注 decay_type / expires_at | `code/profile_extractor.py::build_profile_extraction_prompt` |
| 三类曲线 + 语义校正的单元测试 | `scripts/memory/test_decay_curves.py`（新增） |
| 旧数据迁移（无 decay_type 的 AttrValue 按字段路径补默认值） | `code/memory.py::_migrate_profile` |


### 5.5 读取与更新中的陈旧属性处理

L6 原始画像保留陈旧属性及冲突历史，供页面审计；供模型推理的画像使用只读副本，在读取时按当前时间重算有效置信度，并排除 stale 属性及其历史值。这样即使夜间整理尚未运行，已过期属性也不会继续进入行动建议上下文。

新观察到来时，若同字段单值属性已过期或 stale，新观察优先成为当前值；旧值连同来源和替换依据写入 history，不做物理删除。列表型属性也不会把新观察并入已陈旧条目，而是保留旧记录并追加当前观察。更新时用本次观察的事件时间判断旧值是否已陈旧，历史回放不会拿服务器当前时间误判当时仍有效的属性。尚未制定一般性的跨字段冲突裁决和用户主动遗忘流程；这些规则应在明确字段语义、删除范围及多模态引用保护后再实现。

临时隔离库检查覆盖陈旧值读取排除、按事件时间回放、长间隔后单值替换和列表型 stale 项不吸收新观察，均通过。该检查不修改真实记忆，也不替代固定短期、长期和跨日题集评测。

---

## 六、分层检索语义

### 6.1 early-stop 检索路径

`get_context_for_analysis()`（L951）自顶向下取够即停：

```
① layer6 稳定画像   → [Profile] 段（展平注入，排除 stale）  预算 <1ms
② layer5 长期模式   → 摘要段                           预算 <5ms
③ layer4 近期任务   → 摘要段                           预算 <5ms
④ layer7 索引细节   → 按 person/location/activity 检索  预算 <10ms
（预算合计 <50ms，相对 ≤1s 目标 20 倍余量；实测 5000 条 query() p95 = 14.38ms）
```

**硬约束**：①→④ 同步链路**不含任何 LLM 调用**（画像抽取/摘要生成走 Phase C 异步）。

**实时查询与 Agent 上下文是两个用途不同的入口（09-23 对齐）**：

- `PersonMemory.query_hierarchical()` / `DaytimeProcessor.query()`：查询相关记忆时按 Layer 1→2→3→4→5 逐层取结果，短期层优先，取够 `top_k` 即停；Layer 1~3 返回 moment，Layer 4/5 摘要作为结果项。同步查询不调用 LLM。
- `get_context_for_analysis()`：为 Agent 需求分析准备个性化背景，按 Layer 6→5→4→相关 Layer 7 事实组装。该顺序继续遵循本节既有规格。
- `query()` / `retrieve()`：直接查 Layer 7 永久归档，服务归档检索和 Phase C 整理，不等同于当前场景的 Layer 1→5 查询。

`PersonMemory.add(..., timestamp=ISO)` 保留事件时间，`time_nodes` 使用事件日期；省略 timestamp 则使用服务器当前时间。`NightlyProcessor` 在日间写入结束后读取指定日期的 Layer 7 全量 moments（而非已截断的窗口），分块总结、索引修复与画像更新。详见 `code/nighttime_processor.py` 和 `docs/task2-operations.md`。

### 6.2 分词策略（09-20 修订：中英混合分词，零 LLM）

**中英混合分词**：`tokenize()` 同时处理中英文——中文按单字切分，英文/数字按单词边界切分。

- **写入端（离线预翻译）**：`add()`/`update()` 写入时把非英文文本翻译为英文，存
  `moment["normalized"]`（走 `translator.py` 的 LLM 翻译 + 进程内缓存）。翻译在写入路径执行，
  **不计入检索时延**。
- **查询端（零 LLM）**：`_query_tokens()` 直接 `tokenize()` 分词，**不实时调用 LLM**：
  - 中文查询词 → 单字 token → 命中 moment 的中文原文（scene/location/activity 等）；
  - 英文查询词 → 单词 token → 命中写入时预翻译的 `normalized`。

**演进动机**：09-12 曾定稿「统一英文分词 + 非英文先翻译为英文」，但该方案在查询端
（`_query_tokens`）需实时调用 LLM 翻译，违背 §9.3「检索同步链路不含 LLM」硬约束，存在
突破 ≤1s/2s 的时延风险。09-20 修订为中英混合分词：`tokenize` 直接切中文单字，保留写入
预翻译（供英文查询命中），彻底移除查询路径的 LLM 依赖。

### 6.3 排序与截断

- `query`：按场景、动作、人物、地点、活动、物品等字段独立计算 BM25，并返回 Top-K；`query_ranked()` 同时返回排序分，供语义决策记录候选依据；
- `retrieve`：按当前归档接口对候选记忆排序并输出完整事实；
- 上下文注入：按语义单元展平（属性条目级），不再按字符数硬截断；
- 排除：stale 数据一律不参与检索与注入。

### 6.4 多轮语义检索与行动建议

记忆模块另提供显式的语义决策路径（`code/memory_reasoner.py`，由 `DaytimeProcessor.advise()` 调用），与不含模型调用的快速 `query()` 分开：

1. LLM 根据当前事件规划人物/实体及 2–4 条英文关联查询；
2. 先运行结构化 BM25 查询，再按确有需要时执行一轮补充检索；
3. 以最高文本相关度作为主排序。实体索引命中、用户重点标记、事件时间仅作为同分排序信号，避免弱标签挤掉明确的文本匹配；
4. 对命中实体读取其所在的来源事件，并把该事件中明确记录的其他人物、活动、地点、物品与环境一并作为有出处的关系线索；人物共现只表示共同参与同一事件；
5. 通过实体重叠约束视觉 dHash / 音频频谱相似扩展关联事件；逐轮保留候选 ID 和模态相似度；
6. 建议只引用本轮召回的记忆 ID，并分开展示个人记忆、一般知识、不确定性与风险；模块只给建议，不执行行动。

**多模态决策输入**：完成文本与实体召回后，最终决策会从当前事件及候选记忆读取最多 4 张实际保存的关键帧，并将图像文件送入 Qwen3-VL；最多读取 2 段保存的短音频，经 Whisper 转写后与事件文字一起交给决策模型。画面 dHash 和音频频谱只用于候选相似扩展，不代替模型看到画面或听到转写。没有可用画面时走文本模型降级，并在结果中报告实际帧数、转写数及来源 moment ID。音频在此链路中由 Whisper 转成文本，原始波形没有直接送入 Qwen3-VL。

“基于最新记忆主动建议”会把最新未归并事件的 Scene、Action、人物、地点、活动、物品和语音证据作为当前情境，不需要用户先写问题；它由用户点击按钮触发一次，不是常驻后台监控，也不会自动执行建议。输出分别保留模型原始建议与特定场景规则校正后的最终建议，方便识别哪些内容来自模型、哪些由规则补足。无可用媒体、LLM 无法访问媒体或模型调用失败时，不得把文本降级结果描述为多模态推断。

**M3-Agent 的适配范围**：本实现采用“以事件为证据锚点”的多模态记忆组织方式，并让 LLM 先规划实体与查询、再按结果决定是否补充检索；来源事件和显式字段关系始终可追溯。为兼容既有 L1—L7 与容量目标，实体关系在读取时从 moment 和多模态实体字段派生，不另存一份重复图索引。排序仍以本项目 BM25 为主，并以实体重叠约束轻量画面/音频特征候选扩展；最终决策模型会在媒体可用时接收实际关键帧及音频转写。这不等同于 M3-Agent 的模型、训练数据或完整多模态推理实现。语义嵌入、跨模态检索的误关联评测仍是后续工作。

候选“文本相关度”是本轮查询内按最高 BM25 分归一化的排序值，用来解释排序，**不是置信度、答案正确率或验收指标**。09-30 在 725 条完整日记忆的临时副本中验证了 Alice 洗澡事件：决策使用 4 张关键帧及 2 段 Whisper 转写，端到端约 12.3 秒；模型原始输出为“无需提醒”，场景规则再依据事件记忆改为“先询问 Alice 当前是否仍需要帮助并尊重隐私”，引用来源 moment。该单例证明输入与结果链路接通，不代表准确率或模型独立完成了最终建议。个人护理、晕厥和美国餐厅小费规则只覆盖明确场景；模态消融、更多场景的建议质量与误关联率仍待固定短期、长期、跨日题集评测。

---

## 七、一致性与并发

| 机制 | 定稿 |
|---|---|
| 原子落盘 | `_save()` 写 `.tmp` 临时文件 + `os.replace` 原子替换（防写一半崩溃 / 并发写坏） |
| 单写者模型 | 同一进程内 PersonMemory 为唯一写者；跨进程并发写不在 v1 支持范围（常驻服务化时再加锁） |
| 损坏恢复 | 加载失败时重建空库并告警（避免崩溃扩散） |
| 画像写入约束 | layer6.profile 只允许 `update_profile()` 写入；`compress()` 触碰即警告跳过 |

---

## 八、存储容量预算设计（≤100GB 目标）

### 8.1 存储构成与单条开销估算

记忆库保存结构化事件文本、来源定位和按预算抽取的轻量证据；完整原视频不复制进记忆库。

| 构成 | 单条估算 | 说明 |
|---|---|---|
| moment 本体（scene/user_action/needs/solutions/feedback） | 以实测为准 | 结构化文本、来源视频/片段 ID 和时间范围；不保存主机绝对路径 |
| layer7 索引（六维 + 时间节点） | 以实测为准 | 索引只保存 moment_id，不复制事件正文；实体关系读取时从事实字段派生 |
| 多模态轻量证据 | 画面 ≤12 KB、音频 ≤6 KB / 条（当前抽取上限） | 内容哈希去重；资产预算默认 80 GiB，超预算时优先退场低优先级媒体，不删文字记忆 |
| 画像 / 摘要（layer4/5/6） | 恒定 ~几十 KB | 与 moment 数量无关 |

### 8.2 容量预算表（按 moment 总量 N）

| N（moment 总量） | 无图 | 每 moment 1 帧（50KB） |
|---|---|---|
| 10 万 | ~0.1 GB | ~5.1 GB |
| 100 万 | ~1.1 GB | ~51 GB |
| 500 万 | ~5.5 GB | ~255 GB ❌ 超预算 |

**结论**：结构化事件与轻量证据分开管理，容量结论按实际工作负载测量并保留外推边界。
当前完整日样本 725 条使用 10.58 MiB 去重后的媒体资产；线性外推十年约 37.72 GiB，
不包含原始视频，也不是十年实测。**容量策略**：

1. 完整原视频保留在数据集存储；moment 只记数据集相对视频 ID、片段 ID、片段时间范围和抽取批次；
2. 画面与音频只存有上限的摘要证据，内容相同的资产复用；低优先级媒体可以退出，原始事件文本和来源 ID 保留；
3. 图关系由 moment 字段和现有六维索引即时派生，不额外复制整套长期关系索引；
4. 容量监控记录实际媒体字节数和记忆 JSON 字节数；容量逼近上限时，layer7 中已被压缩进 layer4/5 的旧 moment 可降级为
   摘要引用（保留 moment_id + 摘要指针，正文移出热库）——v2 实现。

### 8.3 容量控制分层职责

| 层 | 容量控制手段 |
|---|---|
| layer1/2/3 | 窗口容量上限（5 / 20 / 1000），恒定小 |
| layer4/5 | 摘要覆盖式更新（不累积），恒定小 |
| layer6 | 画像合并去重 + stale 标记（不物理膨胀） |
| layer7 | 唯一随 N 线性增长的层：索引 + 主存储；由 8.2 预算约束 |

---

## 九、检索性能设计（≤1s 目标）

### 9.1 现状与瓶颈

| 环节 | 现状 | 100 万条外推 |
|---|---|---|
| `_load()` 全量解析 | 59.22ms @5000 条 | ~12s ❌ |
| `query()` O(n) 全量 json.dumps 扫描 | 13.69ms p50 @5000 条 | ~2.7s ❌ |
| layer6 画像直读 | 毫秒级 | 恒定 ✅ |

外推超出 ≤1s 目标，必须按定稿方案优化。

### 9.2 定稿优化方案（三件套）

| # | 方案 | 内容 | 预算 |
|---|---|---|---|
| 1 | 倒排索引（G3） | `token → [moment_id]` 倒排表，写入时增量维护；查询 O(命中数) 而非 O(n) | <100ms @100 万条 |
| 2 | 索引缓存 + 懒加载（G4） | 进程内缓存索引与热数据（layer6 画像常驻内存）；`_load` 只读 metadata + 画像，layer7 按需加载；写时增量更新缓存 | 冷启动 ~100ms |
| 3 | 上下文截断 | 已落地：early-stop + Top-K + 按语义单元展平（不按字符硬截断） | 恒定 |

**检索路径预算（定稿）**：① 画像直读 <1ms → ②/③ 摘要 <5ms → ④ 倒排检索 <100ms，
同步链路合计 <150ms，相对 ≤1s 目标有约 7 倍余量。

### 9.3 与「同步链路不含 LLM」的兼容

倒排索引与缓存都是纯本地数据结构操作，不引入 LLM 调用，不违反硬约束。
分词（统一英文）翻译环节若走 LLM，必须**离线预翻译并缓存**（写入时翻译，查询时查缓存），
检索同步链路上不得实时调用 LLM。

---

## 十、演进路线（v1 → v2）

| 项 | v1（本版） | v2 计划 |
|---|---|---|
| 遗忘判定 | 字段路径默认 + 写入时语义校正 | Agent 完全自主判断衰减策略（会议目标） |
| 分词 | 统一英文分词（非英文先翻译） | 翻译质量与延迟优化（缓存翻译结果） |
| 索引 | layer7 字典式索引 | 倒排索引 + 热画像缓存（≤1s 达标关键） |
| CRUD | 9 操作骨架齐备 + 语义定稿（§4） | 已完成 update/combine/delete 落地与回归；11 个测试模块全通过 |
| 晋升机制 | 窗口视图 + 惰性执行（§2.4 定稿） | 已落地；Layer 3 上限按 EgoLife 校准，跨日和多日容量仍需扩集复核 |
| 存储 | 容量预算定稿（§8） | storage_bytes 监控 + 降级压缩兜底 |
| 评测 | 待 EgoLife 测试集 | 已完成容量、检索速度、P@5 检索相关性、Qwen 感知与单日夜间联调；类别 schema 和 725 片段盲审模板已准备，双人标注、分歧裁决与准确率验收仍待完成 |

---

## 十一、附录：代码位置索引

| 主题 | 位置 |
|---|---|
| 库结构定义 | `code/memory.py::_empty_db` L423 |
| moment 结构定义 | `code/memory.py::_empty_moment` L401 |
| 画像白名单 / 阈值常量 | `code/memory.py` L49-56、L327 |
| AttrValue 规范化 | `code/memory.py::_finalize_attr` L201 |
| 合并语义 | `code/memory.py::_merge_profile_node` L678 及 `_merge_attr_list` L250 / `_resolve_conflict` L234 / `_corroborate_attr` L224 |
| 遗忘机制（当前统一衰减） | `code/memory.py::effective_confidence` L332 / `_apply_decay_and_stale` L348 |
| 分词（当前 bigram，待改英文） | `code/memory.py::tokenize` L287 |
| 索引清洗 | `code/memory.py::_is_valid_person_tag` L98 / `_is_valid_index_tag` L121 / `_clean_layer7_indices` L146 |
| 9 操作 | `code/memory.py::PersonMemory` L481（add L559 / query L745 / retrieve L767 / compress L813 / sort L856 / combine L882 / delete L916 / highlight L938 / update L1009） |
| 画像 API | `code/memory.py::update_profile` L709 / `get_profile` L737 |
| 按需检索入口 | `code/memory.py::get_context_for_analysis` L951 |
| 画像抽取 | `code/profile_extractor.py`（prompt L90 / 解析 L163 / 校验 L205） |
