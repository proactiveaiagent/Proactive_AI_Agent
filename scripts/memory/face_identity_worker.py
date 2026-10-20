import contextlib,json,sys,cv2,os
from pathlib import Path
from insightface.app import FaceAnalysis
rows=json.load(sys.stdin); model_root=Path(os.environ.get("MEMORY_FACEID_ROOT",str(Path.home()/".insightface")))
out=[]
# InsightFace writes model initialization diagnostics to stdout; keep the worker
# protocol stream as strict JSON by redirecting those diagnostics to stderr.
with contextlib.redirect_stdout(sys.stderr):
 app=FaceAnalysis(name="buffalo_l",root=str(model_root),providers=["CPUExecutionProvider"])
 app.prepare(ctx_id=-1,det_size=(640,640))
 for row in rows:
  im=cv2.imread(str(Path(row["path"]))); fs=app.get(im) if im is not None else []
  region=row.get("region")
  if im is not None and isinstance(region,list) and len(region)==4:
   h,w=im.shape[:2]; rx1,ry1,rx2,ry2=region
   fs=[f for f in fs if rx1*w <= (float(f.bbox[0])+float(f.bbox[2]))/2 <= rx2*w
       and ry1*h <= (float(f.bbox[1])+float(f.bbox[3]))/2 <= ry2*h]
  fs=[f for f in fs if float(getattr(f,"det_score",0) or 0)>=.75 and min(float(f.bbox[2]-f.bbox[0]),float(f.bbox[3]-f.bbox[1]))>=24]
  if not fs: out.append({"key":row["key"],"status":"no_reliable_face"}); continue
  f=max(fs,key=lambda z:(z.bbox[2]-z.bbox[0])*(z.bbox[3]-z.bbox[1])*z.det_score)
  out.append({"key":row["key"],"status":"face_found","embedding":[round(float(v),7) for v in f.normed_embedding],"det_score":round(float(f.det_score),5)})
json.dump(out,sys.stdout,separators=(",",":"))
