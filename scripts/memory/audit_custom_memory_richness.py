#!/usr/bin/env python3
"""Audit cached structural coverage for the 35-clip custom video set."""
import argparse, json, re, sys
from PIL import Image
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_perceive import _person_observation_coverage_error, _temporal_fact_coverage_error
CACHE=Path("data/egolife/runs/qwen_cache_memory_prompt_v3.json")
OUTPUT=Path("docs/feishu_custom_memory_richness_audit_20261019.md")
HUMAN=re.compile(r"\b(person|people|woman|man|child|adult|camera wearer|employee|server|waiter|doctor|traveler|traveller|customer|worker|individual)\b",re.I)
NUMBERED_PERSON=re.compile(r"\b(?:unidentified\s+)?(?:person|individual)\s*\d+\b",re.I)

def txt(v):
    if isinstance(v,str): return v.strip()
    if isinstance(v,list): return "; ".join(txt(x) for x in v if x)
    if isinstance(v,dict): return "; ".join(f"{k}: {txt(x)}" for k,x in v.items() if x)
    return ""
def short(v,n=140):
    s=re.sub(r"\s+"," ",txt(v))
    return s if len(s)<=n else s[:n-1]+"…"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,default=CACHE)
    ap.add_argument("--output",type=Path,default=OUTPUT)
    a=ap.parse_args()
    data=json.loads(a.cache.read_text(encoding="utf-8"))
    rows=sorted((str(k),v) for k,v in data.items() if str(k).startswith("FEISHU_"))
    mem_path=Path("data/egolife/runs/interactive_mem/custom_test/memory.json")
    memory=json.loads(mem_path.read_text(encoding="utf-8"))
    moments=list((memory.get("layer7") or {}).get("moments",{}).values())
    roles=Counter(); entity_types=Counter(); entity_statuses=Counter()
    modality=Counter(); speech_predicates=Counter(); speech_clips=set()
    speech_actor_count=Counter(); speech_affected_count=Counter(); speech_timed_count=Counter()
    for moment in moments:
        segment_id=(moment.get("source_provenance") or {}).get("source_segment_id") or ""
        for fact in moment.get("temporal_facts") or []:
            if isinstance(fact,dict) and fact.get("modality")=="speech":
                speech_predicates[str(fact.get("predicate") or "unknown")]+=1
                speech_clips.add(segment_id)
                speech_actor_count["has_actor"] += bool(fact.get("actor"))
                speech_actor_count["identified_actor"] += bool(
                    fact.get("actor") and fact.get("actor") not in
                    {"unidentified speaker", "unidentified addressee"})
                speech_actor_count["unknown_speaker"] += fact.get("actor") == "unidentified speaker"
                speech_actor_count["public_address"] += fact.get("actor") == "public-address announcement"
                speech_affected_count["linked"] += bool(fact.get("affected_people"))
                ref=fact.get("evidence_ref") or {}
                speech_timed_count["source_time"] += bool(
                    ref.get("source_segment_id") and ref.get("audio_start_s") is not None
                    and ref.get("audio_end_s") is not None)
        mm=moment.get("multimodal") or {}
        modality["source_frames"]+=len(mm.get("visual_frames") or [])
        ev=mm.get("entity_visuals") or {}
        assets=ev.get("assets") or []
        modality["entity_assets"]+=len(assets)
        modality["entity_items"]+=len(ev.get("items") or [])
        for asset in assets:
            roles[asset.get("visual_role") or "missing"]+=1
        for item in ev.get("items") or []:
            entity_types[item.get("entity_type") or "missing"]+=1
            entity_statuses[(item.get("entity_type") or "missing",item.get("status") or "missing")]+=1
        ov=mm.get("object_visuals") or {}
        pv=mm.get("person_visuals") or {}
        modality["object_crops"]+=sum(int(item.get("crop_count") or 0) for item in ov.get("items") or [])
        modality["person_crops"]+=sum(int(item.get("crop_count") or 0) for item in pv.get("items") or [])
        if sum(int(item.get("crop_count") or 0) for item in ov.get("items") or []):
            modality["object_crop_clips"]+=1
        if sum(int(item.get("crop_count") or 0) for item in pv.get("items") or []):
            modality["person_crop_clips"]+=1
        for visual_key in ("object_visuals", "person_visuals"):
            for item in (mm.get(visual_key) or {}).get("items") or []:
                for ref in item.get("frame_refs") or []:
                    if ref.get("visual_role") not in {"object_crop", "person_crop"}:
                        continue
                    modality["crop_refs"] += 1
                    asset_path = (mem_path.parent / str(ref.get("asset") or "")).resolve()
                    try:
                        if not asset_path.is_relative_to(mem_path.parent.resolve() / "assets"):
                            raise ValueError("crop path escapes asset directory")
                        with Image.open(asset_path) as image:
                            image.verify()
                        modality["valid_crop_refs"] += 1
                    except Exception:
                        modality["invalid_crop_refs"] += 1
    versions=Counter(); capture_versions=Counter(); totals=Counter(); bins=Counter(); table=[]
    quality_gaps=Counter(); visual_name_count=0; visual_age_count=0
    for mid,e in rows:
        f=e.get("fields") or {}
        facts=[x for x in f.get("temporal_facts") or [] if isinstance(x,dict)]
        persons=[x for x in f.get("person_observations") or [] if isinstance(x,dict)]
        if not persons: totals["clips_without_person_observations"] += 1
        for person in persons:
            totals["numbered_person_aliases"] += bool(NUMBERED_PERSON.search(
                str(person.get("entity") or person.get("subject") or "")))
        visual_name_count += sum(bool(str(x.get("readable_name") or "").strip())
                                 and bool(str(x.get("name_evidence") or "").strip()) for x in persons)
        visual_age_count += sum(bool(str(x.get("age") or "").strip())
                                and x.get("age_basis")=="visible_text" for x in persons)
        objects=f.get("objects") or []; env=f.get("environments") or []
        ver=str(e.get("prompt_version") or "(missing)")
        versions[ver]+=1; bins[len(facts)]+=1
        provenance=moment.get("source_provenance") or {}
        capture_versions[str((moment.get("multimodal") or {}).get("capture_version") or "(missing)")]+=1
        totals["persons"]+=len(persons); totals["objects"]+=len(objects); totals["env"]+=len(env); totals["facts"]+=len(facts)
        totals["locations"]+=bool(f.get("location")); totals["activities"]+=bool(f.get("activity")); totals["times"]+=bool(f.get("time_occasion"))
        fp=e.get("source_input_fingerprint") or {}
        totals["fingerprinted"]+=bool(fp.get("video_sha256") and fp.get("clip_sha256"))
        keys=[(txt(x.get("subject")).casefold(),txt(x.get("predicate")).casefold(),txt(x.get("value")).casefold()) for x in facts]
        repeated=len(keys)-len(set(keys))
        flags=[]
        temporal_error=_temporal_fact_coverage_error(f)
        person_error=_person_observation_coverage_error(f)
        if temporal_error:
            quality_gaps["temporal"]+=1
            flags.append("动作事实覆盖不足")
        if person_error:
            quality_gaps["person"]+=1
            flags.append("人物观察覆盖不足")
        if ver!="memory-v8.0-visible-text-atomic-evidence-2026-10-19": flags.append("待新版复核")
        if len(facts)==0: flags.append("无事件事实")
        elif len(facts)<=2: flags.append("事件事实偏少")
        if HUMAN.search(txt(f.get("scene"))) and not persons: flags.append("场景提到人物但无外观观察")
        if repeated: flags.append(f"{repeated}条完全重复事实")
        if not objects: flags.append("无物品记录")
        if not fp.get("video_sha256") or not fp.get("clip_sha256"): flags.append("来源指纹不完整")
        preview=" / ".join(f"{short(x.get('subject'),22)}:{short(x.get('predicate'),14)}:{short(x.get('value'),38)}" for x in facts[:3])
        table.append([mid,ver,len(persons),len(objects),len(env),len(facts),short(f.get("location"),34),short(f.get("activity"),48),"; ".join(flags) or "初看齐全",short(f.get("scene"),110),preview])
    md=["# 自定义测试集 35 段记忆质量审阅","",
        "范围：Qwen 多模态感知缓存中的全部 FEISHU 自定义测试片段。报告统计已保存的结构字段；当前 GPU 推理故障期间没有把旧缓存冒充成新模型复核结果。缓存文本核查不等于人工金标准。","",
        "## 总览","",
        f"- 片段数：{len(rows)}；视频与片段 SHA 指纹齐全：{totals['fingerprinted']}/{len(rows)}。",
        f"- 当前详细提示词版本：{versions['memory-v8.0-visible-text-atomic-evidence-2026-10-19']} 段；待新版复核标记：{versions['review-required-crosschecked-20261019']} 段。",
        f"- 多模态采集版本：{', '.join(f'{k}={v}' for k,v in sorted(capture_versions.items()))}；当前版本为 temporal-v7-context-entity-crops-2026-10-19。",
        f"- 人物观察 {totals['persons']} 条；物品 {totals['objects']} 条；环境 {totals['env']} 条；视觉事件事实 {totals['facts']} 条。",
        f"- 人物观察平均 {totals['persons']/max(1,len(rows)):.1f} 条/段；{totals['clips_without_person_observations']} 段没有人物观察；编号占位人物别名 {totals['numbered_person_aliases']} 条。无观察的片段需要复核画面，不能用“人物1/人物2”替代整理。",
        f"- 可核验的画面文字姓名 {visual_name_count} 条、画面文字年龄 {visual_age_count} 条。记忆中另有语音转写事实 {sum(speech_predicates.values())} 条，覆盖 {len(speech_clips)} 段；类型：{', '.join(f'{k}={v}' for k,v in sorted(speech_predicates.items()))}。其中明确计划/年龄陈述为 {speech_predicates['stated_plan']}/{speech_predicates['stated_age']} 条；空缺不能靠画面外观推测。",
        f"- 地点、活动字段有值：{totals['locations']}/{len(rows)}、{totals['activities']}/{len(rows)}；明确可读的时间/场合：{totals['times']}/{len(rows)}。没有画面时间文字时留空是正确行为，视频时间另由来源元数据保存。",
        f"- 语音事实来源与关联：片段和音频起止时间齐全 {speech_timed_count['source_time']}/{sum(speech_predicates.values())}；说话来源字段齐全 {speech_actor_count['has_actor']}/{sum(speech_predicates.values())}，其中公共广播来源 {speech_actor_count['public_address']} 条、人物说话者身份仍未知 {speech_actor_count['unknown_speaker']} 条；明确受话人/文字指代关联 {speech_affected_count['linked']}/{sum(speech_predicates.values())}。未做声音身份识别，未知说话者仍保持未知。",
        f"- 事件事实分布：{', '.join(f'{k}条={bins[k]}' for k in sorted(bins))}；{sum(v for k,v in bins.items() if k<=2)} 段只有 0–2 条，{bins[0]} 段没有事件事实。",
        f"- 当前质量门槛未通过：动作事实覆盖不足 {quality_gaps['temporal']}/{len(rows)} 段；人物观察覆盖不足 {quality_gaps['person']}/{len(rows)} 段。普通交互仍即时显示同源缓存并带质量提示；预热负责完整补强。","",
        "## 逐片段检查","",
        "| 片段 | 感知版本 | 人物观察 | 物品 | 环境 | 事实 | 地点 | 活动 | 审阅标记 | 场景摘要 | 前三条事实 |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for row in table: md.append("| "+" | ".join(str(x).replace("|","/").replace("\n"," ") for x in row)+" |")
    md += ["","## 已确认的高优先级缺口","",
      "1. FEISHU_2_1（机场）：0 条事件事实、0 条人物外观观察；空间迁移和人物活动没有分解成可检索事实。",
      "2. FEISHU_4_3（急救场景）：0 条事实、0 条人物观察；应记录可见的人物外观、靠近/跪下/接触等顺序，不输出诊断或行动建议。",
      "3. FEISHU_3_1（钱包放入抽屉）：仅 1 条事实，未表达开抽屉、放钱包、关抽屉和手机仍在桌面等可见状态。",
      "4. FEISHU_4_1（医生与纸张）：存在把持纸动作归给 camera wearer 的风险；需重核主体。可读姓名、日期、病名须逐字按画面记录，不能转成未经核实的诊断或错误关联人物。",
      "5. FEISHU_4_2（红色液体/转场）：仅 1 条事实；把画面转场与真实环境变化分开。",
      "6. FEISHU_5_4（健身）：2 条事实，动作阶段和背景人物活动不足。",
      "7. FEISHU_5_1、FEISHU_6_2、FEISHU_8_1：各 1 条事实；分别核对秤读数/身体部位、走廊和门牌、人物与折叠椅的关系。",
      "8. 冗余：FEISHU_7_2 描述重复 small/yellow onesie 且鸭子 held 事实重复；FEISHU_6_1 重复服饰短语；FEISHU_11_3 的 tissue 多次出现，应按时序区分状态回归和重复噪声。","",
      "## 多模态附件与失败运行","",
      f"- 当前 35 段记忆文件已有 {modality['source_frames']} 张片段级原始帧、{modality['entity_assets']} 个实体证据附件，关联 {modality['entity_items']} 条人物/物品/场景/地点/环境/活动/事件记录；其中 scene/location/environment/activity 各 {entity_types['scene']}/{entity_types['location']}/{entity_types['environment']}/{entity_types['activity']} 条，event {entity_types['event']} 条。环境与活动多模态已经有代表帧/事件帧，不是零附件。",
      f"- 人物已生成 {modality['person_crops']} 个 crop，覆盖 {modality['person_crop_clips']} 段含人物视频；物品只生成 {modality['object_crops']} 个 crop，且仅覆盖 {modality['object_crop_clips']} 段。其余物品虽匹配到源帧，仍不是物品局部 crop，这是当前多模态最明显的附件缺口。",
      f"- 裁剪附件完整性：{modality['valid_crop_refs']}/{modality['crop_refs']} 个裁剪引用都能从 assets 目录解码，损坏或缺失 {modality['invalid_crop_refs']} 个。",
      "- 2026-10-19 复用与源视频、物品清单、抽样帧内容完全匹配的旧检测缓存，为 22 段增加 153 张物品 crop，约 1.74 MB；另有 1 张原有 crop 保留。35 段中 12 段没有完全匹配的检测框，未猜测裁剪位置，需等可靠检测可用后补齐。",
      "- 最近一次 35 段缓存运行：感知命中 17、更新/新建 0、失败 18；物品画面对照新建 0、失败 21、无需对照 14；文字转写缓存命中 35。失败返回 CUDA unspecified launch failure，旧缓存未被覆盖。",
      "- 实体附件已带有来源片段和视频时间；下一步应优先补齐物品框与 crop，并核对人物 crop 是否与稳定描述 alias 对应。仅保留必要的文字转写，不存声音片段。","",
      "## 已落实的提取逻辑改进","",
      "- 事件事实覆盖门槛已从约一半动作线索提高到 75%（最多 8 条不同事实）；现在以 Scene、User Action、Interaction Details、Activity 中动作线索最完整的一栏检查，避免 Scene 简短时漏检，也避免跨字段重复计数；相同 subject/predicate/value 重复项不能抵数。",
      "- 普通交互步骤遇到同源旧提示词缓存时立即返回并标记“旧缓存临时回退 · 待复核”；当前提示词缓存若人物/事件覆盖不足，则立即返回并标记“质量待补强”。两种快速回退都要求源视频与片段 SHA 一致、内容可用且没有未经验证的物品移动；不会覆盖缓存或冒充实时推理。预热流程仍按完整质量门槛强制复核。",
      "- 同源旧记忆重跑时已加入多模态版本检查：使用当前缓存成功刷新文本后，会重新采集场景/地点/环境/活动/事件的来源帧；人物/物品 crop 仅在新旧实体标签精确匹配时复用，不猜测框。当前落盘记忆仍是旧采集版本，需逐段运行后刷新。",
      "- 网页服务已安全重启并返回 HTTP 200；语音关联代码已加载。2026-10-19 04:31 检查 8 张 RTX 4090 全部报告 GPU Recovery Action: Reset，因此未启动视觉推理，也未重置任何共享 GPU。",
      "- 07:00 CST 释放额外 GPU 的既有任务保留；释放后会启动最多 4 个既有 GPU 端点的恢复监视器，每 5 分钟检查，GPU 故障时不回退 CPU，重试窗口到 12:30 CST；只处理短期两套数据各前 10 段、自定义全部、长期 Jake。","",
      "## 后续验收顺序","",
      "1. GPU 驱动恢复后重跑短期前 10 个、自定义全部、长期 Jake；GPU 不可用时不回退 CPU 长跑。",
      "2. 优先复核 2_1、4_3、3_1、4_1、4_2、5_4、5_1、6_2、8_1，逐条确认人物归属、状态转移、地点环境和帧证据。",
      "3. 对 35 段逐条核对场景、人物、物品去向、活动时间顺序、环境和 crop 可打开性。AI 抽查作为质检辅助，不当成人工金标准。",""]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("\n".join(md),encoding="utf-8")
    print(a.output)
    print(f"clips={len(rows)} v8={versions['memory-v8.0-visible-text-atomic-evidence-2026-10-19']} review={versions['review-required-crosschecked-20261019']} facts={totals['facts']} <=2={sum(v for k,v in bins.items() if k<=2)}")
if __name__=="__main__": main()
