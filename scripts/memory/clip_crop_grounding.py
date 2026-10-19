"""CPU-only entity crop localization using HOG proposals and cached CLIP ranking.

The main perception model supplies entity labels and appearance hints. This module
never calls a generative vision service: OpenCV proposes human regions and a small
multiscale grid proposes generic regions; locally cached CLIP ranks them.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
from collections import OrderedDict
from pathlib import Path

_MODEL_ID = "openai/clip-vit-large-patch14"
_METHOD = "opencv-hog+clip-grid-v1"
_MODEL = None
_PROCESSOR = None
_DEVICE = None
_MODEL_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_FEATURE_CACHE = OrderedDict()
_FEATURE_CACHE_LIMIT = 4096


def _model():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    with _MODEL_LOCK:
        if _MODEL is None:
            import torch
            from transformers import (CLIPImageProcessor, CLIPModel, CLIPProcessor,
                                     CLIPTokenizer)
            cache = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
            os.environ.setdefault("HF_HOME", cache)
            # The cached CLIP snapshot contains weights and tokenizer files but may
            # omit preprocessor_config.json. Use the model family's canonical image
            # preprocessing defaults locally instead of reaching the network.
            tokenizer = CLIPTokenizer.from_pretrained(_MODEL_ID, local_files_only=True)
            image_processor = CLIPImageProcessor()
            _PROCESSOR = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
            # Keep local crop ranking on CPU so it cannot exhaust VRAM reserved
            # for the perception endpoints.
            device_name = "cpu"
            try:
                _MODEL = CLIPModel.from_pretrained(
                    _MODEL_ID, local_files_only=True).to(device_name).eval()
                _DEVICE = torch.device(device_name)
            except RuntimeError:
                if not device_name.startswith("cuda"):
                    raise
                torch.cuda.empty_cache()
                _MODEL = CLIPModel.from_pretrained(
                    _MODEL_ID, local_files_only=True).to("cpu").eval()
                _DEVICE = torch.device("cpu")
            torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    return _MODEL, _PROCESSOR


def _feature_tensor(output):
    """Support both tensor-returning and ModelOutput-returning Transformers releases."""
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def _grid_boxes(width, height):
    # Fixed low-count image pyramid: keeps corpus prewarming bounded on CPU.
    specs = ((.16,.24),(.28,.44),(.46,.72),(.62,.34))
    centers = ((.25,.30),(.75,.30),(.50,.50),(.25,.70),(.75,.70))
    rows = []
    for wf, hf in specs:
        bw, bh = min(.88, wf), min(.90, hf)
        for cx, cy in centers:
            x1, y1 = max(0.0,cx-bw/2), max(0.0,cy-bh/2)
            x2, y2 = min(1.0,cx+bw/2), min(1.0,cy+bh/2)
            if (x2-x1) * width < 44 or (y2-y1) * height < 36:
                continue
            box = (x1,y1,x2,y2)
            if not any(_iou(box, old) > .93 for old in rows):
                rows.append(box)
    return rows


def _person_boxes(image):
    try:
        import cv2
        import numpy as np
        detector = cv2.HOGDescriptor()
        detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = min(1.8, 900 / max(1, bgr.shape[1]))
        if scale > 1:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
        found, weights = detector.detectMultiScale(
            bgr, winStride=(8,8), padding=(8,8), scale=1.05, finalThreshold=1.5)
        height, width = rgb.shape[:2]
        output = []
        for (x,y,w,h), weight in zip(found, weights):
            confidence = float(weight[0]) if hasattr(weight, "__len__") else float(weight)
            box = (max(0,x/scale/width), max(0,y/scale/height),
                   min(1,(x+w)/scale/width), min(1,(y+h)/scale/height))
            if box[2] > box[0] and box[3] > box[1]:
                output.append((box, confidence))
        output.sort(key=lambda row: row[1], reverse=True)
        kept = []
        for row in output:
            if not any(_iou(row[0], old[0]) > .65 for old in kept):
                kept.append(row)
            if len(kept) >= 12:
                break
        return kept
    except Exception:
        return []


def _load_frame(frame):
    from PIL import Image, ImageOps
    raw = frame.get("bytes") or b""
    if not raw:
        return None
    try:
        return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        return None


def _crop(image, box):
    width, height = image.size
    x1,y1,x2,y2 = box
    pad_x, pad_y = max(.025,(x2-x1)*.10), max(.025,(y2-y1)*.10)
    bounds = (round(max(0,x1-pad_x)*width), round(max(0,y1-pad_y)*height),
              round(min(1,x2+pad_x)*width), round(min(1,y2+pad_y)*height))
    return image.crop(bounds)


def _text_for(row, kind):
    if kind == "person":
        label = str(row.get("entity") or row.get("person") or "person").strip()
        appearance = str(row.get("appearance") or "").strip()
        text = f"a person, {appearance}" if appearance else f"a person described as {label}"
        return label, [f"a photo of {text}", f"a close view of {text}"]
    label = str(row or "").strip()
    return label, [f"a photo of {label}", f"a close-up photo of {label}"]


def _ground(rows, frames, kind, progress_callback=None):
    label_key = "person" if kind == "person" else "object"
    limit = 8 if kind == "person" else 12
    candidates, seen = [], set()
    for value in rows or []:
        label, prompts = _text_for(value, kind)
        if not label or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        candidates.append({"value": value, "label": label[:100], "prompts": prompts})
        if len(candidates) >= limit:
            break
    base = {"schema": 1, "status": "completed", "model": _METHOD,
            "method": _METHOD, "items": []}
    if not candidates:
        base["status"] = "not_needed"
        return base
    if not frames:
        base["status"] = "no_frames"
        base["items"] = [{label_key: row["label"], "status": "uncertain",
                           "appearance": "", "frame_refs": [], "frame_boxes": [],
                           "reason": "no_sampled_frames"} for row in candidates]
        return base
    if progress_callback:
        try:
            progress_callback({"stage": f"{kind}_visual_grounding",
                               "message": "本地 CPU 正在生成候选区域并匹配实体"})
        except Exception:
            pass
    try:
        import torch
        from PIL import Image
        model, processor = _model()
        proposals, proposal_images = [], []
        frame_rows = list(enumerate(frames[:7], 1))
        if len(frame_rows) > 4:
            last = len(frame_rows) - 1
            anchor_positions = sorted({0, round(last / 3), round(2 * last / 3), last})
        else:
            anchor_positions = list(range(len(frame_rows)))
        anchor_set = set(anchor_positions)
        anchor_images = {}
        hog_proposals = []
        for position, (frame_index, frame) in enumerate(frame_rows):
            image = _load_frame(frame)
            if image is None:
                continue
            if position in anchor_set:
                anchor_images[frame_index] = image
            if kind == "person":
                hog_proposals.extend((frame_index, box)
                                     for box, _score in _person_boxes(image))
        if kind == "person" and hog_proposals:
            proposals.extend(hog_proposals)
        else:
            for frame_index, image in anchor_images.items():
                proposals.extend((frame_index, box)
                                 for box in _grid_boxes(*image.size))
        for frame_index, bbox in proposals:
            image = anchor_images.get(frame_index)
            if image is None:
                image = _load_frame(frames[frame_index - 1])
            if image is not None:
                proposal_images.append(_crop(image, bbox))
        proposals = [proposal for proposal, image in zip(proposals, proposal_images)]
        if not proposal_images:
            raise RuntimeError("no_candidate_regions")
        prompt_texts = []
        for candidate in candidates:
            prompt_texts.append(candidate["prompts"])
        with _RUN_LOCK, torch.inference_mode():
            text_features = []
            for prompts in prompt_texts:
                inputs = processor(text=prompts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=77).to(_DEVICE)
                variants = _feature_tensor(model.get_text_features(**inputs))
                variants = variants / variants.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                text_features.append(variants.mean(dim=0))
            text_features = torch.stack(text_features, dim=0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            crop_keys = [hashlib.sha1(image.tobytes()).hexdigest() for image in proposal_images]
            image_vectors = [None] * len(proposal_images)
            missing = []
            for index, key in enumerate(crop_keys):
                cached_feature = _FEATURE_CACHE.get(key)
                if cached_feature is None:
                    missing.append(index)
                else:
                    _FEATURE_CACHE.move_to_end(key)
                    image_vectors[index] = cached_feature
            for offset in range(0, len(missing), 24):
                batch_indices = missing[offset:offset+24]
                batch = processor(images=[proposal_images[i] for i in batch_indices],
                                  return_tensors="pt")
                features = _feature_tensor(model.get_image_features(
                    pixel_values=batch["pixel_values"].to(_DEVICE)))
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                for row, index in enumerate(batch_indices):
                    feature = features[row].detach().cpu()
                    image_vectors[index] = feature
                    key = crop_keys[index]
                    _FEATURE_CACHE[key] = feature
                    _FEATURE_CACHE.move_to_end(key)
                    while len(_FEATURE_CACHE) > _FEATURE_CACHE_LIMIT:
                        _FEATURE_CACHE.popitem(last=False)
            image_features = torch.stack(image_vectors, dim=0).to(_DEVICE)
            scores = image_features @ text_features.T
            scores = scores.cpu().tolist()
        output_items = []
        assigned_people = []
        for candidate_index, candidate in enumerate(candidates):
            ranked = sorted(((float(score[candidate_index]), index)
                             for index, score in enumerate(scores)
                             if kind != "person" or not any(
                                 proposals[index][0] == used_frame
                                 and _iou(proposals[index][1], used_box) > .50
                                 for used_frame, used_box in assigned_people)), reverse=True)
            if not ranked:
                ranked = [(0.0, 0)]
            best_score, best_index = ranked[0]
            frame_index, bbox = proposals[best_index]
            # Reject weak semantics and same-frame competing regions; uncertainty is
            # stored as a terminal outcome for this exact source fingerprint.
            rivals = [score for score, idx in ranked[1:]
                      if proposals[idx][0] == frame_index
                      and _iou(proposals[idx][1], bbox) < .50]
            gap = best_score - rivals[0] if rivals else 1.0
            threshold = .205 if kind == "person" else .195
            accepted = best_score >= threshold and gap >= .010
            if accepted and kind == "person":
                assigned_people.append((frame_index, bbox))
            item = {label_key: candidate["label"],
                    "status": "visible_unlabeled" if accepted else "uncertain",
                    "appearance": (str(candidate["value"].get("appearance") or "")
                                   if kind == "person" and accepted else ""),
                    "frame_indices": [frame_index] if accepted else [],
                    "frame_boxes": ([{"frame_index": frame_index,
                                      "bbox": [round(float(v),5) for v in bbox],
                                      "score": round(best_score,5),
                                      "method": _METHOD}] if accepted else []),
                    "score": round(best_score,5),
                    "reason": "matched" if accepted else "low_similarity_or_ambiguous"}
            output_items.append(item)
        base["items"] = output_items
        base["prompt_version"] = _METHOD
        return base
    except Exception as exc:
        # This is a local cropper error, never a Qwen-service error. Preserve the
        # source frame in callers and expose an explicit retryable diagnostic.
        base["status"] = "unavailable"
        base["message"] = f"local crop grounding failed: {str(exc)[:220]}"
        base["items"] = [{label_key: row["label"], "status": "uncertain",
                          "appearance": "", "frame_refs": [], "frame_boxes": [],
                          "reason": "local_crop_grounding_failed"} for row in candidates]
        return base


def ground_objects(objects, frames, cancel_event=None, progress_callback=None):
    if cancel_event is not None and cancel_event.is_set():
        return {"schema":1,"status":"cancelled","model":_METHOD,"items":[]}
    return _ground(objects, frames, "object", progress_callback)


def ground_people(people, frames, cancel_event=None, progress_callback=None):
    if cancel_event is not None and cancel_event.is_set():
        return {"schema":1,"status":"cancelled","model":_METHOD,"items":[]}
    return _ground(people, frames, "person", progress_callback)
