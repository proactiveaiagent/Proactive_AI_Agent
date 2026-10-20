"""Source-backed local face matching and reversible identity corrections."""
import hashlib,json,os,subprocess,re
from pathlib import Path
THRESHOLD=float(os.environ.get("MEMORY_FACE_SIMILARITY_THRESHOLD","0.64"))
MODEL="InsightFace buffalo_l / ArcFace"
def read(path,default):
 try:
  x=json.loads(Path(path).read_text(encoding="utf-8")); return x if isinstance(x,type(default)) else default
 except (OSError,ValueError,TypeError): return default
def corrections(path):
 x=read(path,{})
 if not isinstance(x,dict): x={}
 x.setdefault("schema",1); x.setdefault("merge_groups",[]); x.setdefault("merge_links",[]); x.setdefault("separated_pairs",[]); x.setdefault("history",[])
 return x
def write_correction(path,action,left,right,reason="user correction"):
 path=Path(path); left,right=str(left or ""),str(right or "")
 if not left or not right or left==right:return {"ok":False,"error":"需要两条不同的人物记录"}
 d=corrections(path); pair=sorted((left,right))
 if action=="merge":
  d["merge_links"]=[p for p in d["merge_links"] if not(isinstance(p,list) and set(map(str,p))==set(pair))]
  d["merge_links"].append(pair)
  d["separated_pairs"]=[p for p in d["separated_pairs"] if not(isinstance(p,list) and set(map(str,p))==set(pair))]
 elif action=="separate":
  d["merge_links"]=[p for p in d["merge_links"] if not(isinstance(p,list) and set(map(str,p))==set(pair))]
  d["separated_pairs"]=[p for p in d["separated_pairs"] if not(isinstance(p,list) and set(map(str,p))==set(pair))]
  d["separated_pairs"].append(pair)
 else:return {"ok":False,"error":"action must be merge or separate"}
 d["history"].append({"action":action,"source_id":left,"target_id":right,"reason":str(reason)[:240]}); d["history"]=d["history"][-500:]
 path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(path.name+"."+str(os.getpid())+".tmp")
 t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); os.replace(t,path)
 return {"ok":True,"action":action,"file":path.name}
def link_records(records,asset_root=None,correction_path=None):
 people=[r for r in records if r.get("kind")=="person"]
 if not people:return {"model":MODEL,"status":"no_people","matches":0}
 root=Path(asset_root).resolve() if asset_root else None; cp=root/"person_face_embedding_cache.json" if root else None
 cache=read(cp,{}) if cp else {}; cache=cache if isinstance(cache,dict) else {}; crops={}; assets={}
 if root:
  for r in people:
   for ev in r.get("events",[]):
    seg=str((ev.get("source_provenance") or {}).get("source_segment_id") or ev.get("moment_id") or "")
    image_refs=[ref for ref in ev.get("images",[]) if ref.get("asset") and ref.get("visual_role") in {"face_crop","person_crop"}]
    # A dedicated face crop retains facial detail; use the full person crop only as fallback.
    selected=[ref for ref in image_refs if ref.get("visual_role")=="face_crop"] or image_refs
    for ref in selected:
     p=(root/str(ref["asset"])).resolve()
     try:p.relative_to(root); raw=p.read_bytes()
     except (OSError,ValueError):continue
     k=hashlib.sha256(raw).hexdigest()
     assets.setdefault(r["id"],[]).append((k,seg,str(ref["asset"]),None))
     if k not in cache:crops[k]={"path":str(p)}
     source_asset=str(ref.get("source_asset") or "")
     box=ref.get("crop_bbox")
     if source_asset and isinstance(box,list) and len(box)==4:
      try:
       bbox=[max(0.0,min(1.0,float(v))) for v in box]
       if bbox[2]<=bbox[0] or bbox[3]<=bbox[1]:continue
       source_path=(root/source_asset).resolve(); source_path.relative_to(root)
       source_raw=source_path.read_bytes()
      except (OSError,ValueError,TypeError):continue
      source_key=hashlib.sha256(source_raw+json.dumps(bbox,separators=(",",":")).encode()).hexdigest()
      assets.setdefault(r["id"],[]).append((source_key,seg,source_asset,bbox))
      if source_key not in cache:crops[source_key]={"path":str(source_path),"region":bbox}
 py=Path(os.environ.get("MEMORY_FACEID_PYTHON",str(Path(__file__).resolve().parents[1].parent/"runtime/faceid_py311/bin/python")))
 worker=Path(__file__).resolve().parents[1]/"scripts/memory/face_identity_worker.py"
 model_root=Path(os.environ.get("MEMORY_FACEID_ROOT",str(Path.home()/".insightface")))
 model_dir=model_root/"models"/"buffalo_l"
 model_ready=(model_dir/"det_10g.onnx").is_file() and (model_dir/"w600k_r50.onnx").is_file()
 if crops and py.is_file() and worker.is_file() and model_ready:
  try:
   inp=json.dumps([{"key":k,**task} for k,task in crops.items()])
   run=subprocess.run([str(py),str(worker)],input=inp,text=True,capture_output=True,timeout=120,check=True,env={**os.environ,"MEMORY_FACEID_ROOT":str(model_root)})
   for row in json.loads(run.stdout):
    if row.get("key"):cache[row["key"]]=row
   t=cp.with_name(cp.name+"."+str(os.getpid())+".tmp"); t.write_text(json.dumps(cache,separators=(",",":")),encoding="utf-8"); os.replace(t,cp)
  except (OSError,ValueError,subprocess.SubprocessError) as e:
   for r in people:r["identity_link_status"]="face_worker_unavailable: "+str(e)[:120]
 vec={}; segs={}; speech_aliases={}
 for r in people:
  found=[(cache.get(k,{}).get("embedding"),s,a,bbox) for k,s,a,bbox in assets.get(r["id"],[]) if cache.get(k,{}).get("status")=="face_found"]
  found=[x for x in found if isinstance(x[0],list) and x[0]]
  if found:
   vec[r["id"]]=[v for v,_,_,_ in found]; segs[r["id"]]={s for _,s,_,_ in found}
   r["face_evidence"]=[{"asset":a,"source_segment_id":s,**({"region_bbox":bbox} if bbox else {})} for _,s,a,bbox in found]
   r["identity_link_status"]="face_embedding_ready"
  else:r["identity_link_status"]="no_reliable_face_crop"
  aliases=set()
  for event in r.get("events",[]):
   for observation in event.get("person_observations",[]):
    for alias in observation.get("spoken_aliases",[]):
     ref=alias.get("evidence_ref") or {}
     if alias.get("status")=="probable_visual_match" and ref.get("modality")=="speech" and alias.get("basis"):
      value=str(alias.get("alias") or "").strip().casefold()
      if value:aliases.add(value)
  speech_aliases[r["id"]]=aliases
 d=corrections(correction_path); split={frozenset(map(str,p)) for p in d["separated_pairs"] if isinstance(p,list) and len(p)==2}; ids=[r["id"] for r in people]; parent={x:x for x in ids}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]]; x=parent[x]
  return x
 def union(a,b):
  aroot,broot=find(a),find(b)
  if aroot==broot:return True
  left={x for x in ids if find(x)==aroot}; right={x for x in ids if find(x)==broot}
  if any((pair & left and pair & right) for pair in split):return False
  parent[max(aroot,broot)]=min(aroot,broot); return True
 auto=[]
 for i,a in enumerate(ids):
  for b in ids[i+1:]:
   if a not in vec or b not in vec or frozenset((a,b)) in split or segs[a]&segs[b]:continue
   scores=[sum(float(x)*float(y) for x,y in zip(va,vb)) for va in vec[a] for vb in vec[b] if len(va)==len(vb)]; score=max(scores,default=-1)
   shared=speech_aliases.get(a,set()) & speech_aliases.get(b,set())
   if score>=THRESHOLD or (score>=0.52 and shared):
    if union(a,b):
     auto.append({"left":a,"right":b,"score":round(score,5),
                   "basis":"face_plus_explicit_speech" if shared and score<THRESHOLD else "insightface_face_embedding",
                   "speech_aliases":sorted(shared)})
 for pair in d["merge_links"]:
  if isinstance(pair,list) and len(pair)==2 and all(str(x) in parent for x in pair):
   union(str(pair[0]),str(pair[1]))
 for g in d["merge_groups"]:
  valid=[str(x) for x in g if str(x) in parent] if isinstance(g,list) else []
  for x in valid[1:]:union(valid[0],x)
 comps={}
 for r in people:comps.setdefault(find(r["id"]),[]).append(r)
 merged=[]
 for rows in comps.values():
  if len(rows)==1:
   row=rows[0]; row["member_ids"]=[row["id"]]
   first_event=(row.get("events") or [{}])[0]
   provenance=first_event.get("source_provenance") or {}
   row["identity_members"]=[{"id":row["id"],"name":row.get("name",""),
    "source_segment_id":provenance.get("source_segment_id") or first_event.get("moment_id","")}]
   row["identity_matches"]=[]; merged.append(row); continue
  rows.sort(key=lambda r:r["id"]); members={r["id"] for r in rows}
  names=sorted({r["name"] for r in rows},key=lambda n:(bool(re.search(r"(?:person|woman|man|speaker)\s*\d+|unidentified|unknown|人物\s*\d+",n,re.I)),n.casefold()))
  r=dict(rows[0]); r.update({"id":rows[0]["id"],"member_ids":sorted(members),
   "identity_members":[{"id":x["id"],"name":x.get("name",""),
    "source_segment_id":((x.get("events") or [{}])[0].get("source_provenance") or {}).get("source_segment_id")
      or (x.get("events") or [{}])[0].get("moment_id","")} for x in rows],
   "name":names[0],"aliases":names,"identity_scope":"cross_clip_face_match","source_scope":"",
   "events":sorted([e for x in rows for e in x.get("events",[])],key=lambda e:(str(e.get("source_provenance",{}).get("source_video_id") or ""),float(e.get("source_time",{}).get("clip_start_s") or 0),e.get("moment_id",""))),
   "fact_count":sum(int(x.get("fact_count") or 0) for x in rows),
   "identity_matches":[m for m in auto if m["left"] in members and m["right"] in members],
   "identity_correction_available":True})
  r["event_count"]=len(r["events"]); merged.append(r)
 records[:]=[r for r in records if r.get("kind")!="person"]+merged
 records.sort(key=lambda r:(-r.get("fact_count",0),-r.get("event_count",0),r.get("name","").casefold(),r["id"]))
 cached_no_face = any(
  cache.get(k, {}).get("status") == "no_reliable_face"
  for rows in assets.values() for k, _, _, _ in rows
 )
 if vec:
  status="active"
 elif assets and not model_ready:
  status="model_missing"
 elif cached_no_face or (assets and not crops):
  status="no_reliable_faces"
 elif assets:
  status="worker_missing"
 else:
  status="no_face_crops"
 return {"model":MODEL,"status":status,"threshold":THRESHOLD,
         "embedded_people":len(vec),"matches":len(auto),
         "corrections":len(d["history"])}
