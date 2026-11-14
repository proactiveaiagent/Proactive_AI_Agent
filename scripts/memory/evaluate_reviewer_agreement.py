#!/usr/bin/env python3
"""Measure agreement between two independent blind category reviewers."""
import argparse,csv,json,math,re
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/egolife/testset/long_term"
COLS={"people":"people_json","locations":"locations_json","activity_events":"activity_events_json","environments":"environments_json","objects":"objects_json"}
LISTS={"people","locations","environments","objects"}
def norm(x): return re.sub(r"\\s+"," ",str(x or "").casefold()).strip()
def load(p):
 with p.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
 out={r.get("sample_id",""):r for r in rows}
 if not rows or "" in out or len(out)!=len(rows): raise ValueError(f"invalid/duplicate sample ids: {p}")
 return out
def kappa(a,b):
 n=len(a)
 if not n:return {"n":0,"agreement":None,"cohen_kappa":None}
 obs=sum(x==y for x,y in zip(a,b))/n; labels=set(a)|set(b)
 exp=sum(sum(x==z for x in a)/n*sum(y==z for y in b)/n for z in labels)
 return {"n":n,"agreement":round(obs,4),"cohen_kappa":round((obs-exp)/(1-exp),4) if not math.isclose(exp,1) else None}
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--reviewer-1",type=Path,default=BASE/"review_packets/reviewer_1.csv"); p.add_argument("--reviewer-2",type=Path,default=BASE/"review_packets/reviewer_2.csv"); p.add_argument("--template",type=Path,default=BASE/"category_review_template.csv"); p.add_argument("--output",type=Path); a=p.parse_args()
 r1,r2,expected=load(a.reviewer_1),load(a.reviewer_2),load(a.template); ids=set(expected)
 if set(r1)!=ids or set(r2)!=ids: raise ValueError("reviewer packets must match all authoritative template sample ids")
 paired=[]; pending=[]; excluded=[]
 for sid in sorted(ids):
  pair=(r1[sid],r2[sid]); status=[norm(x.get("annotation_status")) for x in pair]
  if any(s not in {"pending","reviewed","excluded"} for s in status): raise ValueError(f"{sid}: invalid annotation_status")
  if "excluded" in status: excluded.append(sid); continue
  if "pending" in status: pending.append(sid); continue
  if any(not x.get("reviewer_id","").strip() for x in pair): raise ValueError(f"{sid}: reviewer_id required")
  vals=[]
  for row in pair:
   rec={}
   for dim,col in COLS.items():
    raw=(row.get(col) or "").strip()
    if not raw: raise ValueError(f"{sid}: {col} blank; use [] for reviewed empty")
    v=json.loads(raw)
    if not isinstance(v,list) or any(not isinstance(t,str) or not norm(t) for t in v): raise ValueError(f"{sid}: invalid {col}")
    v=[norm(t) for t in v]
    if dim=="activity_events" and len(v)!=1: raise ValueError(f"{sid}: activity_events requires one primary label")
    rec[dim]=set(v) if dim in LISTS else v[0]
   vals.append(rec)
  paired.append((sid,*vals))
 metrics={}
 for dim in COLS:
  left=[x[1][dim] for x in paired]; right=[x[2][dim] for x in paired]
  if dim not in LISTS: metrics[dim]=kappa(left,right); continue
  exact=sum(x==y for x,y in zip(left,right))/len(left) if left else None
  jac=[len(x&y)/len(x|y) if x|y else 1.0 for x,y in zip(left,right)]
  labels=set().union(*left,*right) if left else set()
  aa=[label in sample for label in labels for sample in left]; bb=[label in sample for label in labels for sample in right]; micro=kappa(aa,bb)
  metrics[dim]={"n":len(left),"exact_set_agreement":round(exact,4) if exact is not None else None,"mean_jaccard":round(sum(jac)/len(jac),4) if jac else None,"labelwise_micro_agreement":micro["agreement"],"labelwise_micro_cohen_kappa":micro["cohen_kappa"],"labels":len(labels)}
 result={"created_at":datetime.now().isoformat(timespec="seconds"),"expected_samples":len(ids),"reviewed_by_both":len(paired),"pending_any_reviewer":len(pending),"excluded_any_reviewer":excluded,"complete":len(paired)==len(ids) and not pending and not excluded,"per_dimension_agreement":metrics,"note":"Human reviewer agreement is a quality check for gold labels, not model accuracy."}
 out=a.output or ROOT/"data/egolife/runs"/f"reviewer_agreement_{datetime.now():%Y%m%d_%H%M%S}.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False,indent=2)); print("report:",out)
if __name__=="__main__":main()