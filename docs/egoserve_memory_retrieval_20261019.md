# EgoServe 记忆检索子集评测（2026-10-19）

## 评测范围

[EgoServe 官方基准](https://sitonggong.github.io/EgoServe-page/#benchmark)评估连续第一视角视频中的主动服务，覆盖 4 个时间跨度和 10 类服务，官方成绩是服务检测/生成的 F1 与 LLM 评分。当前项目按会议约定负责记忆提取和检索，不负责生成提醒或建议，因此本次只评测可对应的历史记忆检索子集，不把检索命中率当作官方 Overall F1。官方代码与数据说明见 [EgoMemo 评测代码](https://github.com/SitongGong/EgoMemo) 和 [EgoServe 数据集](https://huggingface.co/datasets/SitongGong/EgoServe)。

选取 EgoServe EgoLife 标注里明确指向历史片段的三类任务：Memory Recall、Task Reminder、Memory Link Contextual。覆盖 Jake、Lucia、Katrina 共 231 条标注。候选记忆由已有 EgoLife 视频密集描述与文字转录整理为 908 个十分钟窗口，并通过项目实际的 `PersonMemory.query_ranked`（Layer 7 BM25）检索。

查询只使用标注中的当前场景描述和局部上下文；历史候选只允许来自当前查询时间窗之前已结束的窗口。86 条样本因没有任何严格早于当前查询窗口的目标而整条排除；多目标样本中同窗或未来目标也不计入相关目标。剩余 145 条进入评分。这样做避免同一十分钟记忆同时含有查询之后的信息。运行不调用 GPU，也不生成建议；测的是文字记忆检索，不包含原始图片/音频召回。

## 结果

| 任务类型 | 有效查询 | BM25 Top‑5 命中 | 最近窗口 Top‑5 对照 |
|---|---:|---:|---:|
| Memory Recall | 50 | 12/50 · 24.0% | 49/50 · 98.0% |
| Task Reminder | 15 | 5/15 · 33.3% | 15/15 · 100% |
| Memory Link Contextual | 80 | 28/80 · 35.0% | 9/80 · 11.3% |
| **合计** | **145** | **45/145 · 31.0%** | **73/145 · 50.3%** |

BM25 的总体 MRR 为 0.200，相关历史目标片段 Recall@5 为 28.9%（46/159）。单次 BM25 查询耗时中位数约 290 ms，P95 约 826 ms。

## 观察

同一种排序策略不适合所有记忆跨度。近期回忆和任务提醒的目标通常离当前场景很近，按时间近邻排序远好于纯 BM25；跨日记忆关联则相反，纯最近窗口几乎找不到目标，BM25 Top‑5 命中率为 35.0%。这说明检索应保留时间顺序，并按记忆跨度结合语义和实体关系检索；不宜把所有历史记忆压成一个平面关键词排名。

## 限制

- 这是 EgoServe 官方数据上的**检索子集适配评测**，不是论文官方端到端 F1，也不能和论文中的模型总分直接比较。
- 只评估三个 EgoLife 人物和三类带明确历史片段标注的服务；没有运行 HoloAssist、CaptainCook4D、主动介入判断或建议生成。
- 候选文本来自预先生成的视频描述和文字转录，并按时间窗送入项目检索器；没有重新运行视频感知模型，也没有检查原图/音频附件的召回质量。
- 86 条样本因时间因果约束被整条排除；这是当前十分钟评测粒度的限制，不代表标注无效。

## 可复现产物

- 评测脚本：`scripts/memory/evaluate_egoserve_memory_retrieval.py`
- 逐条结果 JSON：`data/egolife/runs/egoserve_memory_retrieval_20261019.json`（数据目录被 .gitignore 忽略）
- 结果摘要 SHA-256：`95ec0506770b12da032691cfe621b54ce61b8f843cd1f93a3236f75e578a900e`
- 脚本支持官方 Hugging Face 目录结构。下载标注后可运行：
  `huggingface-cli download SitongGong/EgoServe --repo-type dataset --local-dir data/EgoServe_release`
  `python3 scripts/memory/evaluate_egoserve_memory_retrieval.py --annotations-dir data/EgoServe_release`
