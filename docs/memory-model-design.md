# 记忆模型体系设计说明书（v1）

> 版本：v1 ｜ 日期：2026-09-14 ｜ 状态：定稿（供开发与代码审查使用）
>
> 范围：7 层记忆模型的**体系设计**——分层定义、字段字典、CRUD 语义、遗忘机制（三类衰减曲线）。
> 只管记忆模型本身（存储 / 检索 / 更新 / 遗忘 / 压缩 / 索引），LLM 理解等其他模块不在本设计范围。
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
| layer6 | 用户画像 ⭐ | 静态画像四字段（见 §3.3） | 整理时 | Phase C → `update_profile()` |
| layer7 | 分类索引归档 | 4 个倒排索引 + moments 主存储 | 每次交互 + 整理时 | `add()` / `sort()` / `combine()` |

### 2.2 逐层定义

**Layer 1 · 当前时刻（瞬时）**

| 项 | 定稿 |
|---|---|
| 存储内容 | 当前时刻原始观察（scene / user_action / needs / solutions = 一个 moment） |
| 写入时机 | 每次交互（Phase A `memory.add`） |
| 保留时长【推导】 | 1 小时 |
| 晋升 | 满 1 小时 → 晋升 layer2 |
| 淘汰 | 不淘汰，晋升后原层清空 |

**Layer 2 · 同场景历史（瞬时）**

| 项 | 定稿 |
|---|---|
| 存储内容 | 同一场景（location + activity 相同）内历史 moment，按时间排序 |
| 写入时机 | layer1 晋升时；同场景新 moment 到达时追加 |
| 保留时长【推导】 | 24 小时 |
| 晋升 | 满 24 小时 → 晋升 layer3 |
| 淘汰 | 不淘汰，晋升后原层清空 |

> 注：09-03 实测 layer2 为 0 条（同场景聚合逻辑未生效），晋升机制待 CRUD 完备化时修复。

**Layer 3 · 短期记忆（瞬时）**

| 项 | 定稿 |
|---|---|
| 存储内容 | 最近 24h 内全部 moment，按时间排序 |
| 写入时机 | layer2 晋升时 |
| 保留时长【推导】 | 7 天 |
| 晋升 | 满 7 天 → 由 Phase C 压缩生成 layer4 |
| 淘汰 | 不淘汰，压缩后原层清空 |

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

---

## 三、字段字典

### 3.1 moment 字段（layer1~3 基本单元）

实现：`_empty_moment()`（L401）。

| 字段 | 类型 | 含义 | 备注 |
|---|---|---|---|
| `timestamp` | str(ISO) | 记录时间 | |
| `scene` | str | 外部环境描述（Part1） | |
| `user_action` | str | 用户行为（Part1） | |
| `needs` | list | 需求 `[{"need": str, "confidence": float}]`（Part2） | |
| `solutions` | list | 方案 `[{"need": str, "solution": str}]`（Part3） | |
| `feedback` | dict | `{confirmed, corrections, user_rating}`（Part4，后填） | 不阻塞 Part1-3 |
| `highlighted` | bool | 用户确认正确 | `highlight()` 置位 |
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
| （新）经历状态类字段 | AttrValue | 上学阶段等阶段类属性 | 渐变（decay=0.005） |
| （新）时效事件类字段 | AttrValue | 优惠券/会员等 | 截止（expires_at 阶跃） |

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
| `people` / `locations` / `activity_events` / `time_nodes` | dict | 索引键 → [moment_id]，键写入前过 `_is_valid_index_tag` 校验 |
| `metadata.total_moments` | int | 累计 moment 数（ID 生成与整理触发依据） |
| `metadata.last_consolidation` | str/None | 最近整理时间（触发条件依据） |
| `metadata.session_start` / `today` | str | 会话元信息 |

---

## 四、CRUD 语义定义（定稿）

`PersonMemory` 对外 9 操作 + 画像 API。每个操作给出：签名、触发者、精确语义、落盘行为。

### 4.1 操作总览

| # | 操作 | 方法 | 触发者 | 状态 |
|---|---|---|---|---|
| 1 | add | `add()` L559 | Phase A | ✅ |
| 2 | update | `update()` L1009 | 手动 | ⚠️ 仅局部字段，待补全 |
| 3 | query | `query()` L745 | 检索 | ✅ |
| 4 | retrieve | `retrieve()` L767 | 检索 | ✅ |
| 5 | compress | `compress()` L813 | Phase C | ✅ |
| 6 | sort | `sort()` L856 | Phase C | ✅ |
| 7 | combine | `combine()` L882 | Phase C | ⚠️ 待补测试 |
| 8 | delete | `delete()` L916 | 手动 | ⚠️ 待补测试 |
| 9 | highlight | `highlight()` L938 | Phase B 确认 | ✅ |
| — | update_profile | L709 | Phase C | ✅（画像唯一写入口） |
| — | get_profile | L737 | 任意 | ✅ |
| — | get_context_for_analysis | L951 | Phase A | ✅（按需检索入口） |

### 4.2 写入类

**add(scene, user_action, needs, solutions, people, location, activity, extra_notes) → moment_id**
1. 生成 moment_id（`_new_moment_id`，时间戳 + 累计序号）；
2. 写 layer1（当前窗口）；
3. layer1 溢出 → `_graduate_to_layer2` 晋升 layer2（同 scene 聚合）；
4. 写 layer3（当日全部）；
5. 写 layer7：moments 主存储 + people/locations/activity_events/time_nodes 四个索引
   （索引键过 `_is_valid_index_tag` 校验，防污染）；
6. metadata.total_moments + 1；原子落盘。

**update_feedback(moment_id, corrections, rating, confirmed)**：把 Part4 反馈写回
layer7.moments 对应 moment 的 `feedback` 字段。

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
list 型字段追加去重；**跳过 layer6.profile**（打印警告）；更新 last_consolidation；落盘。

**sort(sort_analysis)**：把 LLM 归类的 layer7 索引（people/locations/activity_events/time_nodes）
写入；每个索引键过 `_is_valid_index_tag` 校验（修复 G7 污染）；同键 moment_id 集合去重合并；落盘。

**combine(canonical_map)**：把 layer7 索引中近义旧键合并到规范键（如 "Old fuzzy name" →
canonical）；合并 moment_id 集合并同步更新 moments 主存储里的引用字段；规范键同样过校验；落盘。

### 4.5 删除与标记

**delete(moment_id, reason)**：删除 moments 主存储条目 + 四个索引中的 moment_id 引用 +
layer1/2/3 中的该 moment；落盘。语义：一次性/错误/压缩后重复条目的移除。

**highlight(moment_id)**：把 moment 标记为 confirmed-correct（`highlighted=True`）；
highlight 条目在删除清扫中保留、检索中提权。

### 4.6 画像合并语义（精确规则）

实现：`_merge_profile_node`（L678）及配套工具。

| 场景 | 规则 |
|---|---|
| 列表型维度（food/hobbies/地点/行为/社会关系） | **追加去重**：新值与旧值相似度 ≥ 0.85（英文单词字符 bigram Jaccard）→ 佐证合并；否则追加。绝不整体覆盖（09-03 缺陷 D3 的修复） |
| 单值 AttrValue 冲突（如 age） | **裁决顺序：频次 > 最近 > 置信度**；败者入 history 留痕 |
| 同值重复观察 | **佐证合并**：observations +1、confidence 增强（上限 0.999）、last_seen 更新 |
| 低置信（< 0.3） | 不入库 |
| 新值到达时 | 先过 `_finalize_attr` 规范化（补 source/timestamp/last_seen、observations=1），再过置信度门槛 |

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

### 6.2 分词策略（09-12 定稿）

**统一英文分词**：英文按空格/单词边界分词；**非英文语种（中文等）先翻译为英文再分词**。
翻译在写入与查询两端一致执行（写入时 moment 文本即转为英文索引，查询时查询词同样翻译）。
09-12 前的中文 bigram 实现将被替换。

### 6.3 排序与截断

- `query` / `retrieve`：token 命中数打分降序 + Top-K（K 可配置，默认 5）；
- 上下文注入：按语义单元展平（属性条目级），不再按字符数硬截断；
- 排除：stale 数据一律不参与检索与注入。

---

## 七、一致性与并发

| 机制 | 定稿 |
|---|---|
| 原子落盘 | `_save()` 写 `.tmp` 临时文件 + `os.replace` 原子替换（防写一半崩溃 / 并发写坏） |
| 单写者模型 | 同一进程内 PersonMemory 为唯一写者；跨进程并发写不在 v1 支持范围（常驻服务化时再加锁） |
| 损坏恢复 | 加载失败时重建空库并告警（避免崩溃扩散） |
| 画像写入约束 | layer6.profile 只允许 `update_profile()` 写入；`compress()` 触碰即警告跳过 |

---

## 八、演进路线（v1 → v2）

| 项 | v1（本版） | v2 计划 |
|---|---|---|
| 遗忘判定 | 字段路径默认 + 写入时语义校正 | Agent 完全自主判断衰减策略（会议目标） |
| 分词 | 统一英文分词（非英文先翻译） | 翻译质量与延迟优化（缓存翻译结果） |
| 索引 | layer7 字典式索引 | 倒排索引 + 热画像缓存（≤1s 达标关键） |
| CRUD | 9 操作骨架齐备 | update/combine/delete 补全语义 + 全覆盖测试 |
| 评测 | 待 EgoLife 测试集 | 三指标（容量/速度/准确率）月末验收 |

---

## 九、附录：代码位置索引

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
