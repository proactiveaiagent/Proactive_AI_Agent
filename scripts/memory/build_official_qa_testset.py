#!/usr/bin/env python3
"""Build the labeled official QA subset from the 120-clip short-term sample."""
import json, re
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/egolife"
SOURCE=DATA/"testset/short_term/moments_scattered_v2_120.jsonl"
QA=DATA/"EgoIT/EgoLife_QA.json"
OUT=DATA/"testset/short_term"
CLIPS_OUT=OUT/"official_video_qa_v1_clips_84.jsonl"
QUESTIONS_OUT=OUT/"official_video_qa_v1_questions_810.jsonl"
MANIFEST_OUT=OUT/"official_video_qa_v1_manifest.json"
def clean(s): return re.sub(r"<[^>]+>"," ",str(s or "")).strip()
def parse_id(sid):
 m=re.fullmatch(r"DAY([1-7])_(A[1-6]_[A-Z]+)_(\d{8})_en",str(sid or ""))
 return (int(m.group(1)),m.group(2),m.group(3)) if m else (None,None,None)
def main():
 samples=[json.loads(x) for x in SOURCE.read_text(encoding="utf-8").splitlines() if x.strip()]
 official=json.loads(QA.read_text(encoding="utf-8"))
 by_clip=defaultdict(list)
 for item in official: by_clip[str(item.get("id") or "")+"_en"].append(item)
 clips=[]; questions=[]; excluded=[]
 for sample in samples:
  sid=sample["id"]; day,person,stamp=parse_id(sid); ordinal=0
  for item in by_clip.get(sid,[]):
   conv=item.get("conversations") or []
   if len(conv)<2 or len(clean(conv[0].get("value")))<=8 or not clean(conv[1].get("value")): continue
   ordinal+=1
   questions.append({"question_id":f"{sid}::qa{ordinal:02d}","source_question_ordinal":ordinal,"id":sid,"participant":person,"question":clean(conv[0].get("value")),"official_answer":clean(conv[1].get("value")),"expected_ids":[sid],"day":day,"stamp":stamp,"video":f"{person}/DAY{day}/{sid[:-3]}.mp4"})
  if ordinal:
   clips.append({"id":sid,"participant":person,"day":day,"stamp":stamp,"video":f"{person}/DAY{day}/{sid[:-3]}.mp4","question_count":ordinal})
  else: excluded.append(sid)
 OUT.mkdir(parents=True,exist_ok=True)
 def write_jsonl(path,rows): path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
 write_jsonl(CLIPS_OUT,clips); write_jsonl(QUESTIONS_OUT,questions)
 manifest={"version":"official_video_qa_v1","source_sample_file":str(SOURCE.relative_to(ROOT)),"official_answers_source":str(QA.relative_to(ROOT)),"clips_file":str(CLIPS_OUT.relative_to(ROOT)),"questions_file":str(QUESTIONS_OUT.relative_to(ROOT)),"source_sample_clip_count":len(samples),"test_clip_count":len(clips),"official_question_count":len(questions),"excluded_unlabeled_clip_count":len(excluded),"excluded_unlabeled_clip_ids":excluded,"selection":"Keep only source sample clips with at least one valid official EgoLifeQA question and include every valid question for each retained clip.","note":"This is the labeled QA evaluation set. The 120-clip short-term memory corpus remains unchanged for retrieval context and other memory evaluations."}
 MANIFEST_OUT.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"manifest":str(MANIFEST_OUT),"clips":len(clips),"questions":len(questions),"excluded":len(excluded)},ensure_ascii=False))
if __name__=="__main__":main()
