#!/usr/bin/env python3
"""EgoServe historical-memory retrieval slice; not the official end-to-end service F1."""
import argparse, hashlib, json, re, sys, time, tempfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from memory import PersonMemory
PEOPLE=("A1_JAKE","A4_LUCIA","A5_KATRINA")
TIME_RE=re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})")
FILE_RE=re.compile(r"(A[1-6]_[A-Z]+)_DAY([1-7])_(\d{8})$")
SEG_RE=re.compile(r"(A[1-6]_[A-Z]+)_DAY([1-7])_(\d{8})$")
WIN_RE=re.compile(r"DAY(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?\s*-")
def clock(s):
 m=re.match(r"(\d{1,2}):(\d{2}):(\d{2})(?:[,.](\d{3}))?$",s)
 return int(m[1])*3600+int(m[2])*60+int(m[3])+int(m[4] or 0)/1000 if m else None
def abs_start(s):
 m=WIN_RE.search(s or "")
 return (int(m[1])-1)*86400+int(m[2])*3600+int(m[3])*60+int(m[4]) if m else None
def segkey(s):
 m=SEG_RE.fullmatch(s or "")
 if not m: return None
 day,stamp=int(m[2]),m[3]
 t=int(stamp[:2])*3600+int(stamp[2:4])*60
 return m[1],(day-1)*86400+(t//600)*600
def cues(path):
 text=path.read_text(encoding="utf-8-sig",errors="replace").replace("\r","")
 for block in re.split(r"\n\s*\n",text):
  lines=block.splitlines()
  ix=next((i for i,x in enumerate(lines) if TIME_RE.search(x)),None)
  if ix is None: continue
  m=TIME_RE.search(lines[ix]); start=clock(m[1])
  val=" ".join(x.strip() for x in lines[ix+1:] if x.strip())
  val=re.sub(r"<[^>]*>"," ",val).strip()
  if start is not None and val: yield start,val
def memory_docs():
 grouped=defaultdict(lambda:{"scene":[],"transcript":[]}); files=defaultdict(int)
 for kind,field in (("DenseCaption","scene"),("Transcript","transcript")):
  base=ROOT/"data/egolife/EgoLifeCap"/kind
  for person in PEOPLE:
   for path in sorted((base/person).glob("DAY*/*.srt")):
    m=FILE_RE.fullmatch(path.stem)
    if not m: continue
    day,stamp=int(m[2]),m[3]; base_s=int(stamp[:2])*3600+int(stamp[2:4])*60+int(stamp[4:6]); files[kind]+=1
    for offset,val in cues(path):
     absolute=(day-1)*86400+base_s+int(offset)
     key=(person,(absolute//600)*600); grouped[key][field].append(val)
 docs=[]
 for (person,start),v in grouped.items():
  day0,secs=divmod(start,86400); hour,rem=divmod(secs,3600); minute=rem//60
  docs.append({"person":person,"day":day0+1,"start":start,
   "source_id":f"{person}_DAY{day0+1}_{hour:02d}{minute:02d}0000",
   "scene":"\n".join(v["scene"]).strip(),"transcript":"\n".join(v["transcript"]).strip()})
 return sorted(docs,key=lambda d:(d["person"],d["start"])),dict(files)
def annotation_file(ann_dir,person,kind):
 for candidate in (ann_dir/f"{person}_{kind}.json",ann_dir/"EgoLife"/person/f"{kind}.json",ann_dir/person/f"{kind}.json"):
  if candidate.is_file(): return candidate
 raise FileNotFoundError(f"Missing EgoServe annotation: {person}/{kind}.json under {ann_dir}")
def query_rows(ann_dir):
 out=[]
 for person in PEOPLE:
  for kind in ("episodic","long_term"):
   for e in json.loads(annotation_file(ann_dir,person,kind).read_text(encoding="utf-8")):
    subtype=e.get("sub_type")
    if subtype in ("memory_recall","task_reminder"):
     targets=[e.get("past_segment_id")]
    elif subtype=="memory_link_contextual":
     targets=[x.get("past_segment_id") for x in e.get("linked_past_events",[]) if x.get("past_segment_id")]
    else: continue
    current=abs_start(e.get("current_time_window",""))
    text=" ".join(x.strip() for x in (e.get("current_observation"),e.get("current_local_context")) if x and x.strip())
    if current is None or not text or not targets: continue
    qbin=(current//600)*600; causal=[]; noncausal=0
    for t in targets:
     key=segkey(t)
     if key and key[0]==person:
      if key[1]<qbin: causal.append(key)
      else: noncausal+=1
    out.append({"person":person,"subtype":subtype,"query":text,"current":current,"qbin":qbin,
     "targets":sorted(set(causal)),"noncausal":noncausal,"n_targets":len(targets),
     "event_id":e.get("recall_id") or e.get("reminder_id") or e.get("link_id"),
     "current_segment_id":e.get("current_segment_id")})
 return out
def ts_for(d):
 secs=d["start"]-(d["day"]-1)*86400; h,rem=divmod(secs,3600); m=rem//60
 day=(date(2026,9,1)+timedelta(days=d["day"]-1)).isoformat()
 return f"{day}T{h:02d}:{m:02d}:00"
def add(mem,d):
 return mem.add(scene=d["scene"],user_action="",needs=[],solutions=[],people=None,location=None,
  activity=None,environments=None,objects=None,extra_notes=d["transcript"],timestamp=ts_for(d))
def metrics(rows,k):
 if not rows: return {"queries":0,"hit_at_1":None,"hit_at_5":None,"recall_at_5":None,"mrr":None}
 first=[min((x for x in r["target_ranks"] if x is not None),default=None) for r in rows]
 hit1=sum(x==1 for x in first); hitk=sum(x is not None and x<=k for x in first)
 nrel=sum(len(r["relevant_memory_ids"]) for r in rows); got=sum(sum(x is not None and x<=k for x in r["target_ranks"]) for r in rows)
 return {"queries":len(rows),"hit_at_1":round(hit1/len(rows),4),"hit_at_5":round(hitk/len(rows),4),
  "recall_at_5":round(got/nrel,4) if nrel else None,"mrr":round(sum(1/x if x else 0 for x in first)/len(rows),4),
  "hit_at_1_count":hit1,"hit_at_5_count":hitk,"relevant_targets":nrel,"relevant_targets_retrieved_at_5":got}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--annotations-dir",type=Path,default=ROOT/"data/EgoServe_release")
 p.add_argument("--output",type=Path,default=ROOT/"data/egolife/runs/egoserve_memory_retrieval_20261019.json")
 p.add_argument("--top-k",type=int,default=5); a=p.parse_args()
 docs,files=memory_docs(); bykey={(d["person"],d["start"]):d for d in docs}; queries=query_rows(a.annotations_dir)
 eligible=[]; excluded=defaultdict(int)
 for q in queries:
  if not q["targets"]: excluded["same_or_future_window"]+=1; continue
  available=[x for x in q["targets"] if x in bykey]
  if not available: excluded["target_window_has_no_caption_or_transcript"]+=1; continue
  q["targets"]=available; eligible.append(q)
 results=[]; latencies=[]
 for person in PEOPLE:
  ds=[d for d in docs if d["person"]==person]
  qs=sorted((q for q in eligible if q["person"]==person),key=lambda q:(q["current"],q["subtype"],q["event_id"] or ""))
  with tempfile.TemporaryDirectory(prefix="egoserve_retrieval_") as td:
   mem=PersonMemory(memory_dir=td); mem._save=lambda:None; idmap={}; i=0; indexed_count=-1
   for q in qs:
    while i<len(ds) and ds[i]["start"]<q["qbin"]:
     d=ds[i]; mid=add(mem,d); idmap[mid]=(person,d["start"]); i+=1
    if len(idmap) != indexed_count:
     mem._lexical_search_cache=None
     indexed_count=len(idmap)
    t=time.perf_counter(); ranked=mem.query_ranked(q["query"],top_k=a.top_k); ms=(time.perf_counter()-t)*1000
    latencies.append(ms); ranked_keys=[idmap.get(x.get("moment",{}).get("id")) for x in ranked]
    ranks=[]
    for key in q["targets"]:
     try:ranks.append(ranked_keys.index(key)+1)
     except ValueError:ranks.append(None)
    results.append({"person":person,"subtype":q["subtype"],"event_id":q["event_id"],
     "current_segment_id":q["current_segment_id"],"candidate_memories":len(idmap),
     "relevant_memory_ids":[bykey[x]["source_id"] for x in q["targets"]],
     "retrieved_memory_ids":[bykey[x]["source_id"] for x in ranked_keys if x in bykey],
     "target_ranks":ranks,"query_ms":round(ms,3)})
 k=a.top_k; subs=("memory_recall","task_reminder","memory_link_contextual")
 bysub={s:metrics([r for r in results if r["subtype"]==s],k) for s in subs}
 recent_rows=[]
 for q in eligible:
  candidates=sorted((d for d in docs if d["person"]==q["person"] and d["start"]<q["qbin"]),key=lambda d:d["start"],reverse=True)[:k]
  ranked_keys=[(d["person"],d["start"]) for d in candidates]
  ranks=[]
  for key in q["targets"]:
   try:ranks.append(ranked_keys.index(key)+1)
   except ValueError:ranks.append(None)
  recent_rows.append({"subtype":q["subtype"],"relevant_memory_ids":[bykey[x]["source_id"] for x in q["targets"]],"target_ranks":ranks})
 recent_baseline={"overall":metrics(recent_rows,k),"by_subtype":{s:metrics([r for r in recent_rows if r["subtype"]==s],k) for s in subs}}
 lat=sorted(latencies)
 report={"benchmark":"EgoServe released EgoLife historical-memory retrieval slice",
  "created_at":time.strftime("%Y-%m-%d %H:%M:%S"),
  "scope":"Layer-7 text retrieval only; not the official proactive-service F1.",
  "dataset":{"source":"SitongGong/EgoServe annotations + local EgoLifeCap DenseCaption/Transcript",
   "participants":list(PEOPLE),"service_instances_considered":len(queries),"eligible_queries":len(eligible),
   "retrieval_queries_scored":len(results),
   "query_count_by_subtype":{s:sum(q["subtype"]==s for q in queries) for s in subs},
   "eligible_query_count_by_subtype":{s:sum(q["subtype"]==s for q in eligible) for s in subs},
   "excluded":dict(excluded),"source_srt_files":files,"text_memory_windows":len(docs),
   "memory_windows_by_person":{p:sum(d["person"]==p for d in docs) for p in PEOPLE}},
  "method":{"retriever":"project PersonMemory.query_ranked (Layer-7 BM25)",
   "top_k":k,"query":"current_observation + current_local_context only",
   "memory_document":"one causal 10-minute window; dense captions in scene, text transcript in extra_notes",
   "causal_cutoff":"only completed 10-minute windows before current query window; same-window references excluded",
   "multimodal_scope":"pre-captioned video text only; no raw image/audio retrieval or VLM inference"},
  "overall":metrics(results,k),"by_subtype":bysub,"recent_memory_baseline":recent_baseline,
  "latency_ms":{"p50":round(lat[int((len(lat)-1)*.5)],3) if lat else None,
   "p95":round(lat[int((len(lat)-1)*.95)],3) if lat else None,
   "mean":round(sum(lat)/len(lat),3) if lat else None},
  "details":results}
 digest=hashlib.sha256(json.dumps(report,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 report["report_sha256"]=digest; a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"dataset":report["dataset"],"overall":report["overall"],"by_subtype":bysub,
  "latency_ms":report["latency_ms"],"report_sha256":digest,"saved":str(a.output)},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
