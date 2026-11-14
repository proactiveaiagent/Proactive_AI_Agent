import contextlib,json,sys,cv2,os
from pathlib import Path
import onnxruntime as ort
from insightface.app import FaceAnalysis

rows=json.load(sys.stdin)
model_root=Path(os.environ.get("MEMORY_FACEID_ROOT",str(Path.home()/".insightface")))
prefer_cuda=os.environ.get("MEMORY_FACEID_DEVICE","cuda").casefold() != "cpu"
available=ort.get_available_providers()
use_cuda=prefer_cuda and "CUDAExecutionProvider" in available
providers=["CUDAExecutionProvider","CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
startup_error=""
with contextlib.redirect_stdout(sys.stderr):
 app=FaceAnalysis(name="buffalo_l",root=str(model_root),providers=providers)
 try:
  app.prepare(ctx_id=0 if use_cuda else -1,det_size=(640,640))
 except Exception as exc:
  if not use_cuda: raise
  startup_error=str(exc)[:300]
  app=FaceAnalysis(name="buffalo_l",root=str(model_root),providers=["CPUExecutionProvider"])
  app.prepare(ctx_id=-1,det_size=(640,640))
  use_cuda=False
actual_provider=("CUDAExecutionProvider" if use_cuda and any(
 "CUDAExecutionProvider" in getattr(getattr(model,"session",None),"get_providers",lambda:[])()
 for model in app.models.values()) else "CPUExecutionProvider")
out=[]
with contextlib.redirect_stdout(sys.stderr):
 for row in rows:
  im=cv2.imread(str(Path(row["path"])))
  fs=app.get(im) if im is not None else []
  region=row.get("region")
  if im is not None and isinstance(region,list) and len(region)==4:
   h,w=im.shape[:2]; rx1,ry1,rx2,ry2=region
   fs=[f for f in fs if rx1*w <= (float(f.bbox[0])+float(f.bbox[2]))/2 <= rx2*w
       and ry1*h <= (float(f.bbox[1])+float(f.bbox[3]))/2 <= ry2*h]
  fs=[f for f in fs if float(getattr(f,"det_score",0) or 0)>=.75 and min(float(f.bbox[2]-f.bbox[0]),float(f.bbox[3]-f.bbox[1]))>=24]
  base={"key":row["key"],"provider":actual_provider}
  if startup_error: base["cuda_fallback_reason"]=startup_error
  if not fs:
   out.append({**base,"status":"no_reliable_face"}); continue
  f=max(fs,key=lambda z:(z.bbox[2]-z.bbox[0])*(z.bbox[3]-z.bbox[1])*z.det_score)
  out.append({**base,"status":"face_found","embedding":[round(float(v),7) for v in f.normed_embedding],"det_score":round(float(f.det_score),5)})
json.dump(out,sys.stdout,separators=(",",":"))
