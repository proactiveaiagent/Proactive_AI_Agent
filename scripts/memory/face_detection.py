"""GPU YOLOv8-Face detector for source-resolution person evidence."""
from __future__ import annotations
import os
import threading
from pathlib import Path
_MODEL = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
MODEL_NAME = "YOLOv8n-Face (lindevs, WIDERFace)"
def expanded_face_sample_times(start_s: float, end_s: float,
                               anchor_times: list[float] | None,
                               max_samples: int = 12) -> list[float]:
    """Sample face evidence between Qwen anchors so brief close-up faces are not skipped."""
    start = max(0.0, float(start_s or 0.0))
    last = max(start, float(end_s if end_s is not None else start + 30.0) - 0.1)
    anchors = sorted({round(float(x), 3) for x in (anchor_times or [])
                      if start <= float(x) <= last})
    if len(anchors) < 2:
        count = min(max(2, int(max_samples)), 7)
        anchors = [round(start + (last - start) * i / (count - 1), 3)
                   for i in range(count)]
    mids = [round((left + right) / 2, 3)
            for left, right in zip(anchors, anchors[1:]) if right - left > 0.25]
    extras = sorted(set(mids + [round(start, 3), round(last, 3)]) - set(anchors))
    limit = max(1, int(max_samples))
    if len(anchors) >= limit:
        if len(anchors) <= limit:
            return anchors
        indices = [round(i * (len(anchors) - 1) / (limit - 1))
                   for i in range(limit)] if limit > 1 else [len(anchors) - 1]
        return sorted({anchors[index] for index in indices})
    slots = limit - len(anchors)
    if len(extras) > slots:
        indices = [round(i * (len(extras) - 1) / (slots - 1))
                   for i in range(slots)] if slots > 1 else [len(extras) - 1]
        extras = [extras[index] for index in indices]
    return sorted(set(anchors + extras))
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
    # The detector is shared by the interactive runner and parallel cache warmer.
    # Serialize CUDA inference so several clips cannot race the same YOLO instance.
    with _INFERENCE_LOCK:
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

def assign_faces_to_people(person_rows: list[dict], faces: list[dict],
                           min_confidence: float = 0.25) -> dict:
    """Assign faces only when person geometry gives a unique, mutual match.

    Ambiguous group-frame associations are dropped instead of guessed. Callers
    may then preserve the source frame without adding it to an identity gallery.
    """
    by_face = {}
    by_person = {}
    for person in person_rows or []:
        key = person.get("key")
        bbox = person.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width, height = max(0.02, x2 - x1), max(0.04, y2 - y1)
        center_x = (x1 + x2) / 2
        expected_y = y1 + 0.14 * height
        for face_index, face in enumerate(faces or []):
            face_box = face.get("bbox") if isinstance(face, dict) else None
            confidence = float((face or {}).get("confidence") or 0)
            if confidence < float(min_confidence) or not isinstance(face_box, list) or len(face_box) != 4:
                continue
            fx1, fy1, fx2, fy2 = [float(value) for value in face_box]
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            # Keep the face close to the head area of the proposed person.
            if cx < x1 - 0.08 * width or cx > x2 + 0.08 * width:
                continue
            if cy < y1 - 0.18 * height or cy > y1 + 0.42 * height:
                continue
            distance = ((cx - center_x) / width) ** 2 + 0.7 * ((cy - expected_y) / height) ** 2
            score = distance - 0.015 * confidence
            by_face.setdefault(face_index, []).append((score, key, face))
            by_person.setdefault(key, []).append((score, face_index, face))

    def unique_best(rows):
        rows = sorted(rows, key=lambda row: (row[0], str(row[1])))
        if not rows:
            return None
        if len(rows) > 1:
            gap = rows[1][0] - rows[0][0]
            if gap < max(0.045, abs(rows[0][0]) * 0.30):
                return None
        return rows[0]

    best_person_for_face = {}
    for face_index, rows in by_face.items():
        best = unique_best(rows)
        if best is not None:
            best_person_for_face[face_index] = best[1]
    best_face_for_person = {}
    for key, rows in by_person.items():
        best = unique_best(rows)
        if best is not None:
            best_face_for_person[key] = best[1]

    result = {}
    for face_index, key in best_person_for_face.items():
        if best_face_for_person.get(key) != face_index:
            continue
        row = next((face for score, candidate_key, face in by_face[face_index]
                    if candidate_key == key), None)
        if row is not None:
            result[key] = row
    return result


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
