"""GPU YOLOv8-Face detector for source-resolution person evidence."""
from __future__ import annotations
import os
import threading
from pathlib import Path
_MODEL = None
_MODEL_LOCK = threading.Lock()
MODEL_NAME = "YOLOv8n-Face (lindevs, WIDERFace)"
def _model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU face detection unavailable: PyTorch has no CUDA")
        from ultralytics import YOLO
        root = Path(__file__).resolve().parents[2]
        weights = Path(os.environ.get("MEMORY_FACE_DETECTOR_WEIGHTS", str(root / "models" / "face_detection" / "yolov8n-face-lindevs.pt")))
        if not weights.is_file():
            raise FileNotFoundError(f"Face detector weights missing: {weights}")
        _MODEL = YOLO(str(weights))
    return _MODEL
def detect_faces(frames: list[dict], device: str | None = None) -> list[list[dict]]:
    """Detect faces on full-resolution frame bytes. CUDA is mandatory."""
    import io
    import numpy as np
    from PIL import Image
    model = _model()
    device = str(device if device is not None else os.environ.get("MEMORY_FACE_GPU", "0"))
    output = []
    for frame in frames:
        raw = frame.get("bytes") or b""
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        width, height = image.size
        result = model.predict(source=np.asarray(image), imgsz=1280, conf=0.08, iou=0.55, device=device, max_det=40, verbose=False)[0]
        faces = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                faces.append({"bbox": [max(0.0, min(1.0, x1 / width)), max(0.0, min(1.0, y1 / height)), max(0.0, min(1.0, x2 / width)), max(0.0, min(1.0, y2 / height))], "confidence": float(box.conf[0])})
        output.append(faces)
    return output

def select_face_for_person(person_bbox: list, faces: list[dict]) -> dict | None:
    """Associate a detected face with a person proposal by coarse spatial proximity."""
    if not isinstance(person_bbox, list) or len(person_bbox) != 4 or not faces:
        return None
    x1,y1,x2,y2=[float(x) for x in person_bbox]
    w=max(0.02,x2-x1); h=max(0.04,y2-y1)
    pcx=(x1+x2)/2; expected_y=y1+0.18*h
    ranked=[]
    for face in faces:
        bbox=face.get("bbox")
        if not isinstance(bbox,list) or len(bbox)!=4:
            continue
        fx1,fy1,fx2,fy2=[float(x) for x in bbox]
        cx=(fx1+fx2)/2; cy=(fy1+fy2)/2
        # Allow a face slightly above a torso-biased person proposal.
        if cx < x1-0.45*w or cx > x2+0.45*w or cy < y1-0.55*h or cy > y2+0.1*h:
            continue
        score=((cx-pcx)/w)**2+((cy-expected_y)/h)**2
        ranked.append((score,face))
    return min(ranked,key=lambda row:row[0])[1] if ranked else None

def warmup(device: str | None = None) -> dict:
    """Load weights and execute one tiny CUDA inference at web-server startup."""
    import io
    from PIL import Image
    image = Image.new("RGB", (128, 128), color=(127, 127, 127))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    rows = detect_faces([{"bytes": buffer.getvalue()}], device=device)
    return {"model": MODEL_NAME, "device": str(device or os.environ.get("MEMORY_FACE_GPU", "0")),
            "ready": True, "warmup_detections": len(rows[0]) if rows else 0}
