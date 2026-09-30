# Task 2 服务器操作手册

适用仓库：`/data/cxr25/zhx/Proactive_AI_Agent`（`zhx_dev`）
Python：`/data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python`
推理 API：`http://127.0.0.1:8000`（Qwen3-VL + Whisper；模型默认读取 `/data/cxr25/zhx/HF_HOME/qwen/Qwen3-VL-8B-Instruct`）

以下命令在 `cxr25` 上执行。先连接并进入仓库：

```bash
ssh cxr25
cd /data/cxr25/zhx/Proactive_AI_Agent
PY=/data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python
```

## 1. 启动和检查推理服务

如果 `/health` 已返回 `model_loaded: true`，复用现有服务。否则启动：

```bash
nohup env PORT=8000 "$PY" code/api_server.py > /tmp/cxr25-qwen-api.log 2>&1 &
curl --retry 30 --retry-delay 2 --retry-connrefused -s http://127.0.0.1:8000/health
```

日志：`tail -f /tmp/cxr25-qwen-api.log`。`PORT` 可换成空闲端口；评测器和夜间整理器需同步传入 `--api-url`。

## 2. 创建记忆和 CRUD 操作

`PersonMemory` 将记忆写入 `memory_dir/memory.json`。生产用户数据请给每位用户指定独立、持久的目录；测试代码使用临时目录或 `data/egolife/runs/`，不要改写测试集或仓库内的 `code/memory/memory.json`。

```python
import sys
sys.path.insert(0, "code")
from memory import PersonMemory

mem = PersonMemory(memory_dir="/data/cxr25/zhx/Proactive_AI_Agent/data/egolife/runs/manual_memory")
mid = mem.add(
    scene="Preparing lunch in the kitchen",
    user_action="Chopping vegetables",
    needs=[], solutions=[],
    people=["Alex"], location="kitchen", activity="cooking",
    environments=["home kitchen"], objects=["knife", "vegetables"],
)

# 查找：Layer 1 → 2 → 3 → 4 → 5，优先新近层；同步查询不调用模型。
hits = mem.query_hierarchical("cooking kitchen", top_k=5)

# 改正事实；支持 people/location/activity/environments/objects 等白名单字段。
mem.update(mid, {"location": "home kitchen"})

# 用户确认正确；highlight 条目受到普通 delete 保护。
mem.highlight(mid)
mem.delete(mid, reason="user correction")  # highlighted 时返回 False
# 只有明确需要强删时才传 force=True。
# mem.delete(mid, reason="user requested removal", force=True)
```

完整方法和参数以 `code/memory.py` 的 `PersonMemory` 为准。反馈确认可调用 `update_feedback(mid, confirmed=True, corrections={...})`；画像只通过 `update_profile()` 写入，不要直接编辑 `layer6.profile`。

画像单值属性冲突按“观测次数更多 → 最近观察 → 置信度更高”裁决；全部相同时保留已有值。败者写入该属性的 history，记录胜出值、双方来源、依据和时间。交互执行的操作日志会列出本次新增裁决；逐层详情与完整记忆中的 L6 画像可以展开查看历史。既有历史若无依据字段会标注“旧记录未保存裁决依据”。


## 3. 分层查看与备份

记忆文件是 JSON。可用下列脚本查看每层数量、索引和画像，不打印完整个人内容：

```bash
"$PY" - <<'PY'
import json
p = "data/egolife/runs/manual_memory/memory.json"
d = json.load(open(p, encoding="utf-8"))
for layer in range(1, 6):
    value = d[f"layer{layer}"]
    count = len(value) if isinstance(value, list) else bool(value.get("summary"))
    print(f"layer{layer}: {count}")
print("layer6 profile fields:", list(d["layer6"].get("profile", {})))
print("layer7 moments:", len(d["layer7"].get("moments", {})))
for key in ("people", "locations", "time_nodes", "activity_events", "environments", "objects"):
    print(f"layer7.{key}: {len(d['layer7'].get(key, {}))} index keys")
PY
```

变更或升级前备份：

```bash
cp -a data/egolife/runs/manual_memory/memory.json \
  "data/egolife/runs/manual_memory/memory.$(date +%Y%m%d_%H%M%S).json.bak"
```

## 4. 白天写入与夜间整理

实时流程由 `code/daytime_processor.py` 的 `DaytimeProcessor` 管理场景；`ingest()` 写入 Layer 1/3/7 和索引，`query()` 走 Layer 1→5。传入可解析的 ISO `timestamp` 时，事件时间保存在 moment，日期索引使用事件日期；省略时取服务器当前时间。场景结束调用 `end_scene()` 清空 Layer 1/2 窗口，Layer 3/7 档案保留。

完成当日写入后运行夜间整理：

```bash
"$PY" scripts/memory/run_nightly.py \
  --memory-dir data/egolife/runs/manual_memory \
  --api-url http://127.0.0.1:8000
```

处理指定日期：追加 `--day YYYY-MM-DD`。对相同源数据重复运行会按 digest 跳过；确需重跑时追加 `--force`。整理会生成 Layer 4/5 摘要、验证画像候选并通过 `update_profile()` 写入 Layer 6，修复 Layer 7 六维索引。原始 Layer 7 moments 保留供溯源。

## 5. 测试集评测

```bash
# 两套数据；默认复用 Qwen 结果缓存，未缓存的片段调用本机服务
"$PY" scripts/memory/evaluate_qwen.py --scenario both

# 只评估短期集，或限制每个场景的样本数
"$PY" scripts/memory/evaluate_qwen.py --scenario short_term
"$PY" scripts/memory/evaluate_qwen.py --scenario both --limit 20

# 忽略缓存、重新调用模型
"$PY" scripts/memory/evaluate_qwen.py --scenario short_term --force
```

JSON 报告默认写入 `data/egolife/runs/qwen_eval_*.json`，包括按维度汇总的 precision / recall / F1、失败样本和延迟。`time_index_check` 是输入文件时间标签与 moment ID 的一致性检查，不是视觉模型准确率。

定向测试入口：`scripts/memory/test_daytime.py`、`test_hierarchical_retrieval.py`、`test_nighttime.py`、`test_day_night_integration.py`。测试只操作临时目录。

## 6. 查看可视化与逐步执行

服务器终端启动：

```bash
"$PY" scripts/memory/viz_server.py 8765
```

从开发机建立 SSH 隧道：

```bash
ssh -N -L 8765:127.0.0.1:8765 cxr25
```

然后打开 `http://127.0.0.1:8765`。页面采用固定回放栏与内容标签（记忆流动、交互执行、逐层详情、索引与画像、完整记忆、操作日志），切换视图时无需整页上下翻找；逐层详情、完整记忆和 Qwen 原始输出默认展开，较长内容只在当前视图区滚动。结果回放支持上一步/下一步、0.5×/1×/2×/4×倍速和拖动时间轴；键盘空格播放/暂停、←/→切换步骤。切换场景会并行加载快照与记忆数据。页面可查看已保存快照，也可新建短期或长期交互会话，逐步让 Qwen 感知视频、执行记忆读写并查看快照。视频默认使用轻量预览，可继续切换原始画质。层级说明：L1–L3 共用 moment schema，分别表示当前事件、同环境历史和当日滚动窗口（容量 5/20/1000）；L4 是近期日摘要，L5 面向周/月的长期摘要，L6 保存稳定属性，L7 保存 canonical moments 并提供人物、地点、时间、活动、环境、物品六类索引。完整记忆视图按 moment ID 去重。单日回放不能验证跨日长期模式；当前样例的 L6 没有有效画像值。`/api/interactive/prewarm?scenario=short_term|long_term` 可预热视频感知缓存，进度见 `/api/interactive/prewarm_status`。

可视化服务目前是无认证的内部开发工具，应通过 SSH 隧道或可信网络访问。

夜间整理的 `combine` 只归一化人物、地点、活动、环境和物品等索引标签，不合并或删除 Layer 7 事件。归并按稳定顺序处理，列表属性替换后去重；每条受影响记忆会记录原标签、规范标签和时间。交互页“操作日志”会列出具体的 `旧标签 → 新标签`、受影响记忆示例 ID 和总数、两索引共有条数及字段更新数，便于检查模型给出的归并是否合理。

事件归并与 combine 的索引标签归一化是两项独立操作。交互执行新增“检查相似事件候选”步骤：候选只按两种证据筛选：同源视频片段重叠比例至少 0.80；或时间相差不超过 120 秒、Scene 与 Action 合并文本相似度至少 0.92，且共享人物或物品。画面/音频相似度只供核对，不单独触发候选，因为相似背景声或静止画面不能证明事件重复。候选只供检查，不会自动合并；用户在页面逐条确认后，merge_moments 才会把来源事件归到规范事件下。

合并会将人物、物品、环境、需求和方案列表稳定去重后并入规范记录；Scene、Action、地点、活动和时间冲突保持在来源记录，并在操作日志及 L7 合并历史中列出。来源事件的 L7 正文及视觉/音频资产继续保留；检索默认只返回规范事件一次，并携带来源事件作补充证据。因此它减少重复结果和索引引用，不会直接节省多模态资产存储。网页现支持撤销目标记录最近一次归并：会恢复来源归并关系、归并前索引成员及 L1–L3 窗口位置；合并审计保留撤销时间。若归并后目标字段已变化或缺少撤销快照，会拒绝执行。误候选率与撤销功能仍需在隔离样本上评测；真实缓存不会由扫描自动合并。

09-29 对当前样本做只读候选扫描：长期完整日库 725 条在收紧门限后没有候选；短期 Qwen 缓存中 112 条产生 56 对 Scene、Action、事件时间及现存标签字段完全一致、仅 ID 不同的候选。缓存未记录原视频片段 ID，能确认结构化内容重复，不能据此确认重复写入的上游原因。扫描没有实际归并任何记录；该结果不代表准确率或误合并率验收。

## 7. 检索接口的用途

- `query_hierarchical()` / `DaytimeProcessor.query()`：面向当前交互，按 Layer 1→5 顺序检索，较近层优先。
- `get_context_for_analysis()`：面向 `VRAssistant` 场景分析，按规格书组装稳定画像（Layer 6）→ 长期摘要（Layer 5）→ 近期摘要（Layer 4）→ 相关 Layer 7 事实。
- `query()` / `retrieve()`：Layer 7 归档的全文关键词或人物/地点/活动查找。

### 7.1 语义决策与候选排序说明

可视化页面的“语义决策”使用 `DaytimeProcessor.advise()`：先由模型规划关联查询，再调用 `MemoryReasoner` 检索并给出行动建议。它不等同于 `query()` 的毫秒级检索延迟，也不会执行建议。

- 候选先按本轮 BM25 文本相关度排序；精确实体索引、用户重点标记和事件时间只用于同分排序。
- 展开“候选为什么排在这里”可查看归一化文本排序值、命中的实体、重点标记和多模态关联。归一化分值只解释候选顺序，不代表准确率或置信度。
- 多模态相似扩展要求视觉/音频相似度达到门限并有实体重叠；页面会列出来源事件、目标事件及各模态分数。它是候选补充机制，尚未通过完整误关联率评测。
- 最终建议引用召回候选中的记忆 ID，并单列不确定性和一般知识。无个人证据时应明确说无记忆依据；通用急救或当地习俗不伪装成个人记忆。
- 当前语义场景审阅为三例定性检查，不构成通用准确率。显式规则只覆盖文档列明的特定场景，扩展边界案例仍在计划中。

这些入口承载不同上下文任务。增加新调用方时，应按调用目的选接口，不要把 Layer 7 全库 dump 到实时 prompt。

## 8. 多模态来源与事件实体关联

每条新写入的片段记忆会保留数据集相对视频 ID、源片段 ID、片段起止秒数和抽取批次 ID。服务器绝对路径不写入记忆，也不参加关键词打分；画面/音频资产与文本 moment 共用来源范围。可视化的 L7 详情可查看来源片段、多模态证据和来源 moment。

事件实体关系在读取时由 L7 事实字段及多模态实体字段生成，不另外保存重复图索引。语义决策遇到精确实体命中时，会读取支撑该命中的来源事件，并展示同一事件中明示的人物、地点、活动、环境与物品。每个关联都带来源 moment ID；“同一事件出现”只表示共同出现在该片段，不能据此推断人际关系。若相关事实没有写入记忆，系统仍应明确说明证据缺失，不补造个人病史、偏好或关系。

本链路借鉴 M3-Agent 的实体中心情景记忆与迭代检索组织方式，但保留 L1—L7 和当前模型接口；它不是对 M3-Agent 模型或完整多模态推理框架的复刻。

图关系派生和片段溯源本身不会触发新的模型调用。快速 `query()` 仍以索引和字段加权 BM25 为主；LLM 仅用于用户显式发起的语义决策路径，多模态相似度仍按视觉/音频门限和实体重叠扩展候选。新增关联召回需要继续用隔离评测集检查误关联率和延迟，不能用关系数量替代准确率。


## 9. 画像过期值与检索边界

Layer 6 的“完整记忆”保留过期属性和冲突历史，便于追溯；模型生成语义建议时会在读取时重算衰减，只把仍有效的属性传入推理，并排除历史败选值。新观察若遇到已 stale 的单值，会替换当前值并把旧值、来源和替换依据记入历史。列表型 stale 记录也不再吸收新观察。更新判断使用本次观察时间，因此按时间顺序回放旧数据时不会用服务器当前日期提前淘汰当时有效的属性。

页面将 stale 属性标为“已陈旧 · 当前检索已排除”，并在有效画像摘要中单独计数；这是检索降级，不等于物理删除。跨字段冲突和用户主动遗忘/删除策略仍需单独定义，执行删除时继续保护仍被引用的视觉、音频资产。

临时隔离库已核对陈旧值读取隔离、历史事件时间回放、单值替换和列表更新行为，结果通过；这一 smoke 不代表真实短期、长期或跨日数据的准确率验收。
