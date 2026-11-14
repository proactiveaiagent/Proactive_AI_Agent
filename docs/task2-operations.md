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

可视化页面的“语义决策”使用 `DaytimeProcessor.advise()`：先由模型规划关联查询，再由 `MemoryReasoner` 检索、读取可用多模态证据并给出行动建议。它不等同于 `query()` 的毫秒级检索延迟，也不会执行建议。

- 候选先按本轮 BM25 文本相关度排序；精确实体索引、用户重点标记和事件时间只用于同分排序。
- 展开“候选为什么排在这里”可查看归一化文本排序值、命中的实体、重点标记和多模态关联。归一化分值只解释候选顺序，不代表准确率或置信度。
- 多模态相似扩展要求视觉/音频相似度达到门限并有实体重叠；页面会列出来源事件、目标事件及各模态分数。它是候选补充机制，尚未通过完整误关联率评测。
- 最终决策最多将 4 张候选关键帧实际送入 Qwen3-VL；最多 2 段短音频先经 Whisper 转写。感知模型只看图像，不接收场景提示；按图像序号输出的观察才会被采用。决策模型只能引用校验后的观察与已接受转写。缺媒体或媒体未通过相关性检查时显示文本降级。
- 手动点击“基于最新记忆主动建议”时，系统读取最新未归并事件；真实 Qwen `add`/`add_batch` 步骤写入成功后，页面也会自动触发一次可中断建议。缓存回放和连续执行不触发额外建议；这不是常驻后台监控。模型最多检索三轮（含首轮），页面展示实际帧数、转写数及降级状态。当前事件媒体始终保留；其他媒体只按当前情境与事件摘要/结构化字段的词面关联筛选，只有转录命中时至少要求两个主题词。过滤依据显示源视频、画面/音频秒数和命中词。
- 急症规则先给通用安全处置，但会保留本轮检索到的相关健康记忆引用；“本轮未命中”不等于“完整记忆中不存在”。个人护理中，“一起走”等口头回应不等于允许触碰或进入私人空间。小费按当前地点与一般知识处理：美国餐厅提供可选提醒，先核对服务费，不说成强制支付。
- 最终建议引用召回候选中的记忆 ID，并单列不确定性和一般知识。无个人证据时应明确说无记忆依据；通用急救或当地习俗不伪装成个人记忆。建议不会自动执行。
- 09-30 v22 隔离副本检查使用 725 条完整日记忆及其真实媒体资产。Alice 事件在 9.97 秒抽到的帧显示电子设备而非浴室；8.97 秒音频被转写为与事件不符的短句并被拒绝，最终建议明确区分画面与事件文字，询问陪同意愿并尊重隐私。晕倒与美国小费情境没有相关个人媒体，走文本降级；前者给出意识/呼吸检查及条件性急救建议，后者给出核对服务费、金额自选的可选提醒。 v19 检查还发现 “let / one” 等填充词误命中其他片段，当前已加入填充词与宽泛服务账单词过滤。此三例只用于发现链路缺陷，不代表准确率；跨日/多人物数据、真实语音对齐、模态消融及误关联率仍待固定评测。

- 09-30 v27 复核发现检索规划原先要求输出 2—4 个检索问题，与“只查可能改变下一步的事实”相矛盾，已改为允许空查询并限制到当前情境明示的个人事实；程序会过滤通用急救流程、地点习俗及未具名餐厅账单问题，页面展开项显示模型提议、实际执行和过滤的查询。
- 同轮复核发现 “use / request / reply” 等提示词会把户外和聚餐帧误判为相关，已加入媒体词面过滤。Alice 示例现只送入同一事件的 9.97 秒画面（电子设备，与洗澡无关）；8.97 秒音频转写不匹配并被拒绝。该例不能证明建议由多模态证据支持。
- Qwen 原始输出仍可能越过个人护理同意边界，提出陪同前往浴室。已加人物无关的边界校正：没有明确同意时，最终建议必须先询问偏好并留在私人空间外；网页展示原始建议和校正原因。
- v27 晕倒例只检索个人健康史，通用急救步骤由通用知识提供；小费例仅检索个人偏好，不把习俗或当前账单说成个人记忆。以上是三例定性排查，不是准确率结果；仍需建立具有可靠音视频对齐的正例集与模态消融评测。

这些入口承载不同上下文任务。增加新调用方时，应按调用目的选接口，不要把 Layer 7 全库 dump 到实时 prompt。

## 8. 多模态来源与事件实体关联

每条新写入的片段记忆会保留数据集相对视频 ID、源片段 ID、片段起止秒数和抽取批次 ID。服务器绝对路径不写入记忆，也不参加关键词打分；画面/音频资产与文本 moment 共用来源范围。可视化的 L7 详情可查看来源片段、多模态证据和来源 moment。

事件实体关系在读取时由 L7 事实字段及多模态实体字段生成，不另外保存重复图索引。语义决策遇到精确实体命中时，会读取支撑该命中的来源事件，并展示同一事件中明示的人物、地点、活动、环境与物品。每个关联都带来源 moment ID；“同一事件出现”只表示共同出现在该片段，不能据此推断人际关系。若相关事实没有写入记忆，系统仍应明确说明证据缺失，不补造个人病史、偏好或关系。

本链路借鉴 M3-Agent 的实体中心情景记忆与迭代检索组织方式，但保留 L1—L7 和当前模型接口；它不是对 M3-Agent 模型或完整多模态推理框架的复刻。

图关系派生和片段溯源本身不会触发新的模型调用。快速 `query()` 仍以索引和字段加权 BM25 为主；LLM 仅用于用户显式发起的语义决策路径，多模态相似度仍按视觉/音频门限和实体重叠扩展候选；进入语义决策后，模型会在资产可用时接收实际关键帧和音频转写。新增关联召回需要继续用隔离评测集检查误关联率和延迟，不能用关系数量替代准确率。


## 9. 画像过期值与检索边界

Layer 6 的“完整记忆”保留过期属性和冲突历史，便于追溯；模型生成语义建议时会在读取时重算衰减，只把仍有效的属性传入推理，并排除历史败选值。新观察若遇到已 stale 的单值，会替换当前值并把旧值、来源和替换依据记入历史。列表型 stale 记录也不再吸收新观察。更新判断使用本次观察时间，因此按时间顺序回放旧数据时不会用服务器当前日期提前淘汰当时有效的属性。

页面将 stale 属性标为“已陈旧 · 当前检索已排除”，并在有效画像摘要中单独计数；这是检索降级，不等于物理删除。跨字段冲突和用户主动遗忘/删除策略仍需单独定义，执行删除时继续保护仍被引用的视觉、音频资产。

临时隔离库已核对陈旧值读取隔离、历史事件时间回放、单值替换和列表更新行为，结果通过；这一 smoke 不代表真实短期、长期或跨日数据的准确率验收。

## 10. 时序证据与多模态对照

新版写入使用 `visual_frames` 和 `audio_clips` 保留多时点，仍保留 `visual`、`audio` 供旧调用方读取。可视化在完整记忆中显示三帧各自的秒数和音频起止区间，不增加重复视频播放器。建议详情显示逐段原文、译文、置信度过滤和语义采用原因。每段证据都来自自己的时间点；最后结果不能重新套用中间帧位置。

`scripts/memory/build_multimodal_sample.py` 支持 `--long-day`、`--dataset`、`--output-dir`、`--limit` 和重复 `--source-id`，按来源 ID 断点续做。使用新目录保留旧版结果。当前六人物及四条日记忆样本在 `data/egolife/runs/multimodal_aligned_v7_20260930`，完整日新版在 `multimodal_full_day_v7_20260930`。媒体封顶 30 KB/事件；文本文件紧凑落盘，所有事实字段不变。

`code/memory_audio.py` 只读取已准备的本地模型，不在交互时下载。默认模型位置是 `data/models/memory-whisper-small`；可用 `MEMORY_WHISPER_MODEL_PATH` 指向另一份本地目录。模型不可用时记录共享 ASR 降级来源。语言提示来自输入对话，人物名来自文字标注，不通过声音猜测身份。模型相容性判断只检查能否佐证当前事件，不证明 ASR 原文一定正确。

`scripts/memory/evaluate_multimodal_decisions.py` 在隔离目录生成五个真实主动扫描事件、两个“真实媒体＋构造当前目标”场景各两种输入的报告，每个条件单独保存 `result.json`。模型没有读取其他相机人物或未来事件、摘要、画像。网页“语义决策”可展开“与仅文本比较”；来源校验列检查引用和输入是否一致，不能解释为建议正确率。

语义决策先读当前媒体，再规划检索；一次请求内复用相同输入批次，不反复识别已经处理的同一批画面和声音。构造目标场景必须保留标识，不作为官方题目或无需问题的主动触发正例。

建议提交前另有模型证据复核。程序核对来源原句和明确指令覆盖，漏句时最多补查一次；修订建议还需一次后续复核。原始建议保存在 `model_recommendation`，最终建议在 `recommendation`，过程保存在 `evidence_review`。复核失败或修订仍有问题时暂缓建议，界面保留原因。该机制不能当作独立标注答案或建议准确率。

当前固定对照的输出目录为 `data/egolife/runs/multimodal_decision_review_v37_20260930`；v28—v36 目录保留用于追踪问题。报告包含各源文件及输入记忆的 SHA256；同目录断点续做时实现必须一致，否则需换新输出目录。模型本体放在 `data/models/memory-whisper-small`，约 483.5 MB，不计入每用户事件资产；下载临时块在校验后清理。

本轮 14 条结果及每条件 `result.json` 已全部保存。`asset_integrity_report.json` 记录 124/725 两批资产的来源引用检查，`retrieval_latency_report.json` 记录新版库 100 次快速检索，`speech_model_provision_report.json` 记录官方模型哈希及临时下载清理；`qualitative_review.json` 是对两例构造目标的输出和来源文字核对，没有重新逐帧审阅视频，不能当作官方行动答案。

过程检查 76/77，不计为正确率。1 条建议因复核未解决而暂缓；包装例已保留盒子和镜布，硬盘盒例仍有未经支持的 USB 供电假设，网页旁注明示。后续优先改进语音有效片段选择和未知设备推断边界，再扩充多日与模态消融。


## 11. 三点音频采样与 v8 容量复测（2026-10-07）

写入记忆时，片长不超过 10 秒的事件保留完整音频；更长事件从开头、中间和结尾各摘录约 3.33 秒，单事件音频不超过 10 秒。语义建议最多消费三段音频，网页仍按原始起点和时长展示。

完整日 725 条样本生成 2,175 帧、1,943 段音频，合计 21,813,239 字节；按当前样本密度十年线性估算为 74.2 GiB（79.7 GB），不含原视频。4,118 个多模态引用的文件、大小、哈希、时间边界和单事件预算检查均通过。100 次 Top5 快速文本检索 P95 为 164.36 ms，最大 165.51 ms；不包含媒体处理和大模型调用。

固定场景对照只检查链路来源一致性并用于发现错误，不是官方行动答案集，也不能由通过项推导建议正确率。当前 21 个音频摘录中 3 段通过语义相容性检查、18 段未识别出可靠语音，语音转写准确率仍需带标注数据测量。

v40 的 14 条对照共有 74/77 项来源与流程检查通过；3 个未通过条件按规则暂缓建议。该数字只描述引用、媒体输入和复核流程，不表示建议正确率。Tasha 纯文本对照中的“询问咖啡馆能否借冰块”可在原始对话找到依据，不能记为编造。v41 增加了区分第三方提议和用户已确认意图的复核提示，但该建议仍被保留；后续需要独立标注的行动标准，评估提议是否已被采纳、是否对佩戴者相关、是否应主动提醒。Alice 的硬盘盒规格回查存在于事件标注中，因此不属于无来源设备型号推断。v41 在同一 14 条对照上仍为 74/77 项来源与流程检查通过；Tasha 提议在原始对话中有据，但当前没有官方答案判断是否应主动推荐。

对 v41 14 条多模态/纯文本条件追加 Codex AI 行动质量定性审阅，维度包括来源支持、佩戴者意图、当前相关性和安全边界。审阅指出两条“保留盒子/镜头布”提醒遗漏、Tasha 场景多模态输入未利用明确语音计划、硬盘盒接电理由需核实 USB 供电要求。Tasha 纯文本建议有对话依据，不能记为幻觉。无官方行动答案，因此不汇总准确率；逐条判断和引文见 data/egolife/runs/multimodal_decision_review_v41_20261007/scenario_review.json，评测页面比较表增加 Codex AI 定性审阅列。


## 2026-10-07 行动建议复核 v45

以同一份 EgoLife v8 记忆样本重新运行 5 个真实事件和 2 个评测构造目标的多模态/纯文本配对条件，共 14 条。结果保存在 data/egolife/runs/multimodal_decision_review_v45_20261007/scenario_review.json。代码和输入哈希随结果记录。

77 项来源、模态隔离和复核流程检查通过 76 项；唯一未通过是硬盘盒目标的纯文本条件，复核仍暂缓建议。此比例只表示流程检查完成度，不是建议准确率。EgoLife 给出事件/语音标注，没有下一步行动官方答案。

Codex AI 逐条定性复核指出：拆箱场景可能漏掉“保留盒子/镜头布”的提醒；Tasha 场景没有把 Katrina 的咖啡馆借冰提议作为带说话人归属的可选建议；硬盘盒接电目标仍把 USB 接口带入建议，需核对设备标识或说明书。花束与洗澡场景不提醒是合适的。完整引文和逐条件审阅写在结果 JSON，网页比较表同步显示。该复核不是盲审金标准，也不能作为准确率。

v45 提示明确区分同一任务内的保留例外、第三方可选提议和普通讨论，并要求不推断设备连接/供电事实。完整重跑中没有消除拆箱提醒与语音提议遗漏，说明仅改提示不足以证明行动质量改进；这两类应留作下一轮边界测试。v44/v45 旧输出独立保留，不拼接成同一评测轮。


## 2026-10-07 来源核验与行动线索 v50

v49 的完整对照发现：检索规划器已经识别出 Katrina 关于向附近咖啡馆询问冰块的多句提议，但来源核验要求整段引文逐字连续匹配，因说话人标签和句间分隔符而错误拒绝。v50 将说话人标签及标点归一化，同时要求引文正文完整、连续地出现在单条来源字段内；模型提出的线索仍须是高相关、未完成或未明确状态，并绑定记忆 ID、提议人和原文。该修正解决的是引用格式兼容，不放宽内容来源要求。

完整配对评测仍使用 EgoLife v8 同人、截至当前事件的历史，5 个真实事件和 2 个构造目标，共 14 个多模态/纯文本条件。结果在 data/egolife/runs/multimodal_decision_review_v50_20261007/scenario_review.json。过程检查 75/77；两项失败均为硬盘盒构造目标的建议证据复核，输出暂缓，没有给出具体接线操作。该计数只衡量流程检查，不是建议准确率。

定性复核显示：Jake 拆箱场景依据原话保留镜头布并将丢弃范围限定在盒内其他物品；Tasha 场景将 Katrina 的咖啡馆建议明确标为尚未采纳的可选项；花束与洗澡场景继续无提醒。构造目标仅用于边界检查。EgoLife 没有官方下一步行动答案，Codex AI 的逐条意见也不能作为盲审准确率或验收金标准。

剩余工作：解决硬盘盒目标的来源复核失败；扩展跨日、多人物、真实健康记录正例和更多独立决策案例；评估分别移除画面/音频的影响与错误关联率；补充真实 ASR 误纳/误拒标注，以及语义向量召回和存储/延迟评测。M3-Agent 仍是设计参考，当前系统不是其复刻。
