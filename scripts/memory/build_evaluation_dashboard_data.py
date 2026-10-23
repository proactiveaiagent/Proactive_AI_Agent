#!/usr/bin/env python3
import argparse
import json,re
from pathlib import Path

ROOT=Path("/data/cxr25/zhx/Proactive_AI_Agent")
DATA=ROOT/"data/egolife"
RUNS=DATA/"runs"

def parse_args():
    parser=argparse.ArgumentParser(description="Build the visualization payload from saved memory evaluation reports.")
    parser.add_argument("--short-long-report",type=Path,default=RUNS/"short_long_official_qa_all_20260928.json")
    parser.add_argument("--cache",type=Path,default=RUNS/"qwen_cache_expanded_eval.json")
    parser.add_argument("--qa",type=Path,default=DATA/"testset/long_term/egolifeqa_official_a1_jake.json")
    parser.add_argument("--retrieval-report",type=Path,default=RUNS/"egolifeqa_memory_retrieval_qwen_multiday_20260928_115121.json")
    parser.add_argument("--answer-report",type=Path,default=RUNS/"egolifeqa_answer_accuracy_forced_choice_multiday_760_20260928_fieldaware.json")
    parser.add_argument("--output",type=Path,default=RUNS/"evaluation_dashboard_data_20260928.json")
    return parser.parse_args()

args=parse_args()
CACHE_MANIFEST_PATH=args.cache.with_suffix(".manifest.json")
CACHE_MANIFEST=json.load(open(CACHE_MANIFEST_PATH,encoding="utf-8")) if CACHE_MANIFEST_PATH.exists() else {}
REPORT=json.load(open(args.short_long_report,encoding="utf-8"))
CACHE=json.load(open(args.cache,encoding="utf-8"))
QA=json.load(open(args.qa,encoding="utf-8"))
RET=json.load(open(args.retrieval_report,encoding="utf-8"))
ANS=json.load(open(args.answer_report,encoding="utf-8"))
QA_BY={str(x.get("ID","")):x for x in QA}
ANS_BY={str(x.get("id","")):x for x in ANS.get("details",[])}
RET_BY={str(x.get("id","")):x for x in RET.get("details",[])}
NAMES={"A1_JAKE":"Jake","A2_ALICE":"Alice","A3_TASHA":"Tasha","A4_LUCIA":"Lucia","A5_KATRINA":"Katrina","A6_SHURE":"Shure"}

def video_rel(sid):
    m=re.fullmatch(r"DAY([1-7])_(A[1-6]_[A-Z]+)_(\d{8})_en",str(sid or ""))
    if not m:return ""
    d,person,stamp=m.groups()
    base=str(sid)[:-3]
    rel=f"{person}/DAY{d}/{base}.mp4"
    return rel if (DATA/rel).is_file() else ""

def parse_id(sid):
    m=re.fullmatch(r"DAY([1-7])_(A[1-6]_[A-Z]+)_(\d{8})_en",str(sid or ""))
    if not m:return None
    return int(m.group(1)),m.group(2),m.group(3)

def stamp_seconds(stamp):
    return int(stamp[:2])*3600+int(stamp[2:4])*60+int(stamp[4:6])+int(stamp[6:8])/30.0

SHORT_IDS={json.loads(line)["id"] for line in (DATA/"testset/short_term/moments_scattered_v2_120.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}

def causal_memory_count(sid,scope):
    parsed=parse_id(sid)
    if not parsed:return 0
    qday,person,qstamp=parsed
    cutoff=stamp_seconds(qstamp)+30.0
    ids=SHORT_IDS if scope=="short_term" else CACHE.keys()
    count=0
    for mid in ids:
        m=parse_id(mid)
        if not m or m[1]!=person:continue
        day,_,stamp=m
        if day<qday or (day==qday and stamp_seconds(stamp)+30.0<=cutoff):count+=1
    return count

def fields_for(sid):
    row=CACHE.get(sid) or {}
    return row.get("fields") if isinstance(row.get("fields"),dict) else {}

def segment(sid,rank=None,targets=None):
    sid=str(sid or "")
    targets=targets or []
    return {
      "id":sid,"rank":rank,"is_target":sid in targets,
      "video":video_rel(sid),"fields":fields_for(sid)
    }

def enrich_common(row):
    targets=[str(x) for x in row.get("expected_ids",[])]
    retrieved=[str(x) for x in row.get("retrieved_ids",[])[:5]]
    return {
      "id":row.get("id"),"question_id":row.get("question_id",row.get("id")),"source_question_ordinal":row.get("source_question_ordinal"),"participant":row.get("participant"),
      "question":row.get("question",""),
      "official_answer":row.get("official_answer",""),
      "model_answer":row.get("model_answer",""),
      "answer_correct":row.get("answer_correct") if isinstance(row.get("answer_correct"),bool) else None,
      "top5_hit":bool(set(targets)&set(retrieved)),
      "target_ids":targets,"target_points":row.get("target_points",[]),
      "retrieved_ids":retrieved,
      "target_segments":[segment(sid,targets=targets) for sid in targets],
      "retrieved_segments":[segment(sid,i+1,targets) for i,sid in enumerate(retrieved)],
      "video":video_rel(row.get("id")),
      "causal_memory_count":causal_memory_count(row.get("id"),row.get("scope","")),"memory_clips_available":row.get("memory_clips_available"),
      "answer_latency_ms":row.get("answer_latency_ms"),
      "judge_latency_ms":row.get("judge_latency_ms"),
    }

scopes={}
for scope in ("short_term","long_term"):
    src=REPORT[scope]
    scopes[scope]={
      "overall":src.get("overall",{}),
      "by_participant":src.get("by_participant",{}),
      "details":[enrich_common(r) for r in src.get("details",[])],
      "source":REPORT.get("method",{}),
    }

multi=[]
for r in RET.get("details",[]):
    qid=str(r.get("id",""))
    qa=QA_BY.get(qid,{})
    a=ANS_BY.get(qid,{})
    targets=[str(x) for x in r.get("expected_ids",[])]
    retrieved=[str(x) for x in (a.get("retrieved_ids") or r.get("retrieved_ids") or [])[:5]]
    choices={letter:qa.get("choice_"+letter.lower(),"") for letter in "ABCD"}
    multi.append({
      "id":qid,"participant":"A1_JAKE","type":r.get("type",""),
      "question":qa.get("question") or r.get("query",""),
      "choices":choices,
      "official_choice":str(qa.get("answer") or r.get("answer_choice") or "").upper(),
      "model_choice":str(a.get("predicted_choice") or "").upper(),
      "answer_correct":a.get("correct") is True,
      "top5_hit":bool(set(targets)&set(retrieved)),
      "target_ids":targets,"target_points":r.get("target_points",[]),
      "question_day":r.get("question_day"),"question_time":r.get("question_time"),
      "temporal_relation":r.get("temporal_relation"),
      "retrieved_ids":retrieved,
      "target_segments":[segment(sid,targets=targets) for sid in targets],
      "retrieved_segments":[segment(sid,i+1,targets) for i,sid in enumerate(retrieved)],
    })
payload={
  "created_at":REPORT.get("completed_at"),
  "sample_coverage":REPORT.get("sample_coverage",{}),
  "evaluation_sources":{"short_long_report":str(args.short_long_report),"memory_cache":str(args.cache),"memory_cache_manifest":CACHE_MANIFEST,"retrieval_report":str(args.retrieval_report),"answer_report":str(args.answer_report)},
  "participant_names":NAMES,
  "short_term":scopes["short_term"],
  "long_term":scopes["long_term"],
  "multi_day":{
    "overall":ANS.get("overall",{}),
    "retrieval_overall":RET.get("overall",{}),
    "by_question_day":RET.get("by_question_day",{}),
    "details":multi,
    "scope":"official A1_JAKE multi-day 500-question multiple-choice benchmark"
  },
}
out=args.output
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(payload,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
print("wrote",out,"bytes",out.stat().st_size,"short",len(payload["short_term"]["details"]),"long",len(payload["long_term"]["details"]),"multiday",len(multi))
print("available videos short",sum(bool(s["video"]) for r in payload["short_term"]["details"] for s in r["target_segments"]))
