"""GPU entity crop localization using CUDA HOG proposals and cached CLIP ranking.

The main perception model supplies entity labels and appearance hints. This module
never calls a generative vision service: CUDA HOG and a small
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
_METHOD = "torch-cuda-hog+clip-grid-v6-qwen-frame-anchored-face-identity-no-score-threshold"
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
            # Cropping must use a GPU; CUDA failures are surfaced instead of hidden by CPU fallback.
            device_name = os.environ.get("MEMORY_CROP_DEVICE", "cuda:1").strip() or "cuda:1"
            if not device_name.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError(f"local crop ranking requires CUDA, configured device={device_name}")
            device = torch.device(device_name)
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
            if free_bytes < 3 * 1024**3:
                raise RuntimeError(f"insufficient free CUDA memory for CLIP on {device_name}: "
                                   f"{free_bytes / 1024**3:.2f} GiB free")
            _MODEL = CLIPModel.from_pretrained(_MODEL_ID, local_files_only=True).to(device).eval()
            _DEVICE = device
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
    # Fixed low-count image pyramid: bounds proposal generation work.
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


def _person_boxes(image, device):
    """Run HOG feature extraction and people-window ranking entirely on CUDA."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"person HOG requires CUDA, got {device}")
    rgb = np.array(image.convert("RGB"), copy=True)
    height, width = rgb.shape[:2]
    pixels = torch.as_tensor(rgb, device=device).permute(2, 0, 1).unsqueeze(0).float()
    gray = (pixels[:, 0:1] * 0.299 + pixels[:, 1:2] * 0.587
            + pixels[:, 2:3] * 0.114) / 255.0
    max_width = 900
    if width > max_width:
        gray = F.interpolate(gray, size=(max(128, round(height * max_width / width)), max_width),
                             mode="bilinear", align_corners=False)
    base_h, base_w = gray.shape[-2:]
    weights = np.load(Path(__file__).parent / "assets" / "opencv_default_people_detector.npy",
                      allow_pickle=False).astype(np.float32, copy=False).reshape(-1)
    descriptor_size = 15 * 7 * 36
    if weights.size != descriptor_size + 1:
        raise RuntimeError(f"unexpected OpenCV HOG weight count: {weights.size}")
    svm = torch.as_tensor(weights[:descriptor_size], device=device)
    bias = float(weights[descriptor_size])
    found = []
    for scale in (1.0, 0.82, 0.67):
        sh, sw = max(128, round(base_h * scale)), max(64, round(base_w * scale))
        frame = gray if (sh, sw) == (base_h, base_w) else F.interpolate(
            gray, size=(sh, sw), mode="bilinear", align_corners=False)
        h, w = (frame.shape[-2] // 8) * 8, (frame.shape[-1] // 8) * 8
        if h < 128 or w < 64:
            continue
        frame = frame[:, :, :h, :w]
        gx, gy = torch.zeros_like(frame), torch.zeros_like(frame)
        gx[:, :, :, 1:-1] = frame[:, :, :, 2:] - frame[:, :, :, :-2]
        gx[:, :, :, 0] = frame[:, :, :, 1] - frame[:, :, :, 0]
        gx[:, :, :, -1] = frame[:, :, :, -1] - frame[:, :, :, -2]
        gy[:, :, 1:-1, :] = frame[:, :, 2:, :] - frame[:, :, :-2, :]
        gy[:, :, 0, :] = frame[:, :, 1, :] - frame[:, :, 0, :]
        gy[:, :, -1, :] = frame[:, :, -1, :] - frame[:, :, -2, :]
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12).reshape(-1)
        angle = torch.remainder(torch.atan2(gy, gx).reshape(-1) * (9.0 / np.pi), 9.0).clamp_(0.0, 8.999999)
        bin0 = torch.floor(angle).long()
        frac, bin1 = angle - bin0, (bin0 + 1) % 9
        hc, wc = h // 8, w // 8
        yy = torch.arange(h, device=device).repeat_interleave(w) // 8
        xx = torch.arange(w, device=device).repeat(h) // 8
        cell, ncells = yy * wc + xx, hc * wc
        hist_flat = torch.zeros(9 * ncells, dtype=frame.dtype, device=device)
        hist_flat.scatter_add_(0, bin0 * ncells + cell, magnitude * (1.0 - frac))
        hist_flat.scatter_add_(0, bin1 * ncells + cell, magnitude * frac)
        hist = hist_flat.view(9, hc, wc).permute(1, 2, 0)
        blocks = torch.cat((hist[:-1, :-1], hist[:-1, 1:],
                            hist[1:, :-1], hist[1:, 1:]), dim=-1)
        blocks = (blocks / torch.sqrt(blocks.square().sum(-1, keepdim=True) + 1e-6)).clamp(max=0.2)
        blocks = blocks / torch.sqrt(blocks.square().sum(-1, keepdim=True) + 1e-6)
        if blocks.shape[0] < 15 or blocks.shape[1] < 7:
            continue
        windows = blocks.unfold(0, 15, 1).unfold(1, 7, 1)
        windows = windows.permute(0, 1, 3, 4, 2).contiguous()
        windows = windows.view(windows.shape[0], windows.shape[1], descriptor_size)
        scores = torch.matmul(windows, svm) + bias
        flat = scores.reshape(-1)
        top_scores, indices = torch.topk(flat, min(48, flat.numel()))
        nx = scores.shape[1]
        ys = torch.div(indices, nx, rounding_mode="floor")
        xs = indices % nx
        for score, y, x in zip(top_scores.tolist(), ys.tolist(), xs.tolist()):
            box = (max(0.0, x * 8 / (base_w * scale)),
                   max(0.0, y * 8 / (base_h * scale)),
                   min(1.0, (x * 8 + 64) / (base_w * scale)),
                   min(1.0, (y * 8 + 128) / (base_h * scale)))
            if box[2] > box[0] and box[3] > box[1]:
                found.append((box, float(score)))
    found.sort(key=lambda row: row[1], reverse=True)
    kept = []
    for row in found:
        if not any(_iou(row[0], old[0]) > 0.55 for old in kept):
            kept.append(row)
        if len(kept) >= 12:
            break
    return kept


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


def _person_visual_anchor_indices(value, frames):
    """Return only frame indices explicitly linked to this person's visual evidence."""
    if not isinstance(value, dict):
        return []
    indices = set()
    direct = value.get("frame_index")
    try:
        direct = int(direct)
        if 1 <= direct <= len(frames):
            indices.add(direct)
    except (TypeError, ValueError):
        pass
    refs = [value.get("evidence_ref")]
    refs.extend(value.get("evidence_refs") or [])
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("modality") != "visual":
            continue
        raw_index = ref.get("frame_index")
        try:
            index = int(raw_index)
            if 1 <= index <= len(frames):
                indices.add(index)
                continue
        except (TypeError, ValueError):
            pass
        raw_time = ref.get("frame_time_s")
        try:
            at = float(raw_time)
        except (TypeError, ValueError):
            continue
        nearest = min(range(len(frames)), key=lambda i: abs(
            float(frames[i].get("at_s") or 0) - at)) if frames else -1
        if nearest >= 0 and abs(float(frames[nearest].get("at_s") or 0) - at) <= 0.35:
            indices.add(nearest + 1)
    if indices and value.get("frame_time_s") is not None:
        try:
            at = float(value["frame_time_s"])
            nearest = min(range(len(frames)), key=lambda i: abs(
                float(frames[i].get("at_s") or 0) - at)) if frames else -1
            if nearest >= 0 and abs(float(frames[nearest].get("at_s") or 0) - at) <= 0.35:
                indices.add(nearest + 1)
        except (TypeError, ValueError):
            pass
    return sorted(indices)


def _ground(rows, frames, kind, progress_callback=None):
    label_key = "person" if kind == "person" else "object"
    limit = 8 if kind == "person" else 12
    candidates, seen = [], set()
    for value in rows or []:
        label, prompts = _text_for(value, kind)
        if not label or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        candidate = {"value": value, "label": label[:100], "prompts": prompts}
        if kind == "person":
            candidate["anchor_frames"] = _person_visual_anchor_indices(value, frames)
        candidates.append(candidate)
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
            requested_device = os.environ.get("MEMORY_CROP_DEVICE", "cuda:1").strip() or "cuda:1"
            progress_callback({"stage": f"{kind}_visual_grounding",
                               "message": f"本地 HOG + CLIP（配置设备 {requested_device}）正在匹配实体；不调用 Qwen"})
        except Exception:
            pass
    try:
        import torch
        from PIL import Image
        model, processor = _model()
        if progress_callback:
            try:
                progress_callback({"stage": f"{kind}_visual_grounding",
                                   "message": f"本地 CLIP 已就绪（实际设备 {_DEVICE or 'CPU'}），正在排序候选区域；不调用 Qwen"})
            except Exception:
                pass
        proposals, proposal_images = [], []
        frame_rows = list(enumerate(frames[:7], 1))
        if kind == "person":
            # Face clarity and pose can change sharply between input moments;
            # compare every supplied frame instead of four temporal anchors.
            anchor_positions = list(range(len(frame_rows)))
        elif len(frame_rows) > 4:
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
                                     for box, _score in _person_boxes(image, _DEVICE))
        if kind == "person":
            proposals.extend(hog_proposals)
        # Grid candidates cover partial/occluded people; CLIP ranks all proposals on CUDA.
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
                             if (kind != "person" or
                                 (proposals[index][0] in candidate["anchor_frames"] and not any(
                                  proposals[index][0] == used_frame
                                  and _iou(proposals[index][1], used_box) > .20
                                  for used_frame, used_box in assigned_people)))), reverse=True)
            if not ranked:
                ranked = [(0.0, 0)]
            selected = []
            if kind == "person":
                # One best region per sampled moment; retain all available
                # moments without a similarity threshold.
                used_frames = set()
                for score, index in ranked:
                    frame_index, bbox = proposals[index]
                    if frame_index in used_frames:
                        continue
                    selected.append((score, index))
                    used_frames.add(frame_index)
                    assigned_people.append((frame_index, bbox))
                    if len(selected) >= min(6, len(candidate["anchor_frames"])):
                        break
            else:
                selected = ranked[:1]
            if not selected:
                output_items.append({label_key: candidate["label"],
                                     "status": "uncertain",
                                     "appearance": (str(candidate["value"].get("appearance") or "")
                                                    if kind == "person" else ""),
                                     "frame_indices": [], "frame_boxes": [],
                                     "reason": "no_direct_visual_anchor_for_person"
                                               if kind == "person" else "no_candidate_region"})
                continue
            best_score, best_index = selected[0]
            frame_index, bbox = proposals[best_index]
            item = {label_key: candidate["label"],
                    "status": "visible_unlabeled",
                    "appearance": (str(candidate["value"].get("appearance") or "")
                                   if kind == "person" else ""),
                    "frame_indices": list(dict.fromkeys(
                        proposals[index][0] for _score, index in selected)),
                     "anchor_frame_indices": list(candidate["anchor_frames"]) if kind == "person" else [],
                    "frame_boxes": [{"frame_index": proposals[index][0],
                                     "bbox": [round(float(v),5) for v in proposals[index][1]],
                                     "score": round(score,5),
                                     "method": _METHOD}
                                    for score, index in selected],
                    "score": round(best_score,5),
                    "reason": "Qwen_visual_evidence_anchored_frames_no_crop_quality_threshold"
                              if kind == "person" else "best_available_region_no_score_gate"}
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


def warmup():
    """Load the local CLIP ranker on its configured CUDA device."""
    model, _processor = _model()
    return {"model": _MODEL_ID, "device": str(_DEVICE), "ready": model is not None}
