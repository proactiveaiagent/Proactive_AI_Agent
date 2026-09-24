# 类别人工复核标注方案

## 目的与范围

本方案用于验证“基础类别归类准确性 ≥95%”这一验收项。检索 P@5、Qwen 调用成功率和人工标注正确率是不同指标。当前 EgoLife 自动标签由词典与规则生成，字段边界与模型输出不一致，因此不能直接充当类别准确率的金标准。

复核单位是一个视频片段（video + start/end 秒），不是一行双语 JSONL。完整 DAY1 数据有 1450 行，英文/中文配对后对应 725 个唯一片段。标注模板由 scripts/memory/build_category_review_set.py 生成，位于 data/egolife/testset/long_term/category_review_template.csv。模板不含旧自动标签和 Qwen 预测，避免标注者被答案提示。

## 标注流程

1. 至少两位标注者分别复制空白模板，独立查看视频片段，并可参考对应语音转写。不要查看 Qwen 输出、旧 ground_truth、自动生成的 scene 或 dense caption。
2. 每个字段填 JSON 数组。字段空白表示尚未标注；复核完成且确实没有可信标签时填 []。填写 annotation_status=reviewed、reviewer_id 与 confidence（high / medium / low）。
3. 对类别字段和实体存在分歧的样本，由第三位标注者或任务负责人查看原片并写入 adjudicated 版本。保留两份原始标注和裁决记录，不覆盖原始文件。
4. 视频打不开、片段缺失、内容不足以判断时，在 reviewer_notes 说明原因；不要把不可判定强行记成负例。最终评测报告单独披露排除样本和原因。

## 字段定义

| 字段 | 含义 | 标注边界 |
|---|---|---|
| people_json | 片段中可由语音或上下文可靠识别的参与者 | 只写有依据的名字；不能确认身份时不猜名字，并在备注说明 |
| locations_json | 事件发生的功能性地点 | 例如 kitchen、bedroom、office、yard；table、window、door 等物体或局部构件不算地点 |
| activity_events_json | 片段中的主要用户活动 | 使用下方活动词表中的一个主标签；确实不匹配时使用 other:简短英文短语 |
| environments_json | 可观察的环境/场景属性 | 可填 indoor、outdoor、natural_light、artificial_light、dim、crowded、quiet；不把地点名或物体名放入此字段；其他属性用 other:简短英文短语 |
| objects_json | 与活动显著相关的具体物品 | 使用单数英文名词并合并同义词，例如 mobile phone / smartphone 统一为 phone；不要把同一物品的外观描述拆成多个物品 |

主活动词表：meeting、socializing、device_use、object_transfer、cooking、eating_drinking、walking_mobility、cleaning、exercise、personal_care、work_study、resting、other:短语。若一个片段包含多个动作，选持续时间最长或对片段最关键的活动作为主标签；补充活动可写在 reviewer_notes，本轮主指标不据此扩展标签数。

## 评测与验收口径

复制 docs/task2-category-aliases.template.json 为 category_review_aliases.json，并为裁决集中的每个 human label 和 Qwen prediction 建立精确的“观测别名 → 规范标签”映射；规范标签本身也要有显式自映射。评测器拒绝任何未映射标签，不会把未知标签原样兜底为规范标签。默认使用完整的 725 样本模板；子集试跑必须显式传 --allow-partial，报告会保持 complete=false。只有模板中的 725 个 sample_id 全部出现、状态均为 reviewed 且没有 pending/excluded 时，报告才会标记 complete=true。命令示例：python3 scripts/memory/evaluate_category_review.py --annotations adjudicated.csv --qwen-report data/egolife/runs/long_term_eval_20260923.json --aliases category_review_aliases.json --output data/egolife/runs/category_review.json。评测逐维输出集合准确率、micro P/R/F1 或活动 top-1 准确率及混淆矩阵；不得使用模糊子串匹配。对 people、locations、environments、objects 报告逐片段 exact-set accuracy 和 micro precision/recall/F1；对 activity_events 报告 top-1 accuracy 与混淆矩阵。逐维列出结果，并保留总体汇总，不能只报告一个总体平均值掩盖弱项。

原始项目标准写的是“≥95%，无明显类别混淆”，但未明确 95% 是每个维度、宏平均还是总体。正式宣称通过前，需由验收负责人确认主指标口径；同时报告逐维结果和混淆矩阵，说明误差类别。模型输出需先用同一套规范标签映射，再与裁决集比较。人工复核集完成之前，验收状态保持“尚未证实”。
