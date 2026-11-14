"""GPU entity crop localization using CUDA HOG proposals and cached CLIP ranking.

The main perception model supplies entity labels and appearance hints. This module
never calls a generative vision service: CUDA HOG and a small
multiscale grid proposes generic regions; locally cached CLIP ranks them.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

_MODEL_ID = "openai/clip-vit-large-patch14"
_METHOD = "torch-cuda-hog+clip-grid-v18-confidence-gated-context-fallback"
_OBJECT_LOCAL_SCORE_MIN = 0.28
_OBJECT_LOCAL_MARGIN_MIN = 0.035
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
            # Keep ranking on CUDA, but do not pin it to a busy GPU. Prefer an
            # explicitly configured device when it has room; otherwise select the
            # visible CUDA device with the most free memory. This avoids silently
            # losing all image grounding when another model occupies cuda:1.
            if not torch.cuda.is_available():
                raise RuntimeError("local crop ranking requires CUDA; no CUDA device is available")
            preferred = os.environ.get("MEMORY_CROP_DEVICE", "").strip()
            device_indices = list(range(torch.cuda.device_count()))
            if preferred and preferred.lower() != "auto":
                try:
                    preferred_index = torch.device(preferred).index
                    if preferred_index is None:
                        preferred_index = torch.cuda.current_device()
                    if preferred_index in device_indices:
                        device_indices.remove(preferred_index)
                        device_indices.insert(0, preferred_index)
                except (RuntimeError, ValueError):
                    pass
            available = []
            for index in device_indices:
                try:
                    free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
                except Exception:
                    continue
                if free_bytes >= 3 * 1024**3:
                    available.append((index, free_bytes))
            if not available:
                raise RuntimeError("local crop ranking requires at least 3 GiB free on a CUDA device")
            # Keep a valid explicit preference first; otherwise favor headroom.
            preferred_index = None
            if preferred and preferred.lower() != "auto":
                try:
                    preferred_index = torch.device(preferred).index
                    if preferred_index is None:
                        preferred_index = torch.cuda.current_device()
                except (RuntimeError, ValueError):
                    preferred_index = None
            available.sort(key=lambda row: (row[0] == preferred_index, row[1]), reverse=True)
            last_oom = None
            for index, _free_bytes in available:
                device = torch.device(f"cuda:{index}")
                try:
                    _MODEL = CLIPModel.from_pretrained(
                        _MODEL_ID, local_files_only=True).to(device).eval()
                    _DEVICE = device
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    last_oom = exc
                    _MODEL = None
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            if _MODEL is None:
                raise RuntimeError(f"CLIP could not fit on available CUDA devices: {last_oom}")
            torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    return _MODEL, _PROCESSOR


def _feature_tensor(output):
    """Support both tensor-returning and ModelOutput-returning Transformers releases."""
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _face_image_features(images):
    """Return normalized CLIP vectors for face crops, reusing the GPU feature cache."""
    import io
    import torch
    from PIL import Image

    model, processor = _model()
    image_rows = []
    keys = []
    for raw in images:
        raw = bytes(raw or b"")
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image_rows.append(image)
        keys.append(hashlib.sha1(raw).hexdigest())
    vectors = [None] * len(image_rows)
    missing = []
    for index, key in enumerate(keys):
        cached = _FEATURE_CACHE.get(key)
        if cached is None:
            missing.append(index)
        else:
            _FEATURE_CACHE.move_to_end(key)
            vectors[index] = cached
    with _RUN_LOCK, torch.inference_mode():
        for offset in range(0, len(missing), 24):
            batch_indices = missing[offset:offset + 24]
            batch = processor(images=[image_rows[index] for index in batch_indices],
                              return_tensors="pt")
            features = _feature_tensor(model.get_image_features(
                pixel_values=batch["pixel_values"].to(_DEVICE)))
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for row, index in enumerate(batch_indices):
                key = keys[index]
                feature = features[row].detach().cpu()
                vectors[index] = feature
                _FEATURE_CACHE[key] = feature
                _FEATURE_CACHE.move_to_end(key)
                while len(_FEATURE_CACHE) > _FEATURE_CACHE_LIMIT:
                    _FEATURE_CACHE.popitem(last=False)
    return vectors


def resolve_face_gallery_candidates(items, candidates,
                                    min_similarity=0.82, min_margin=0.05):
    """Keep or relabel extra face crops only when they agree with Qwen face anchors.

    Candidate text/geometry proposes where a face might be. For clips with more
    than one anchored person, the face's visual embedding must identify the same
    anchored profile; a strong match to another profile relabels the crop rather
    than contaminating the proposed card. Single-person clips retain the existing
    one-face/text-grounded fallback, but ambiguous group frames do not.
    """
    rows = [row for row in candidates or []
            if isinstance(row, dict) and isinstance(row.get("face_bytes"), bytes)
            and row.get("face_bytes")]
    if not rows:
        return []
    vectors = _face_image_features([row["face_bytes"] for row in rows])
    anchors = {}
    qwen_people_by_frame = {}
    for row in rows:
        if row.get("is_anchor"):
            frame_index = int(row.get("highres_index", -1))
            qwen_people_by_frame.setdefault(frame_index, set()).add(
                int(row.get("item_index", -1)))
    trusted_anchor_indices = set()
    for index, row in enumerate(rows):
        person_index = int(row.get("item_index", -1))
        evidence = str(row.get("identity_evidence") or "")
        score = float(row.get("person_box_score") or 0.0)
        margin = float(row.get("identity_margin") or 0.0)
        visually_unique = ("unique_appearance" in evidence
                           and score >= 0.23 and margin >= 0.06)
        face_count_residual = "unique_face_count_residual_assignment" in evidence
        sole_qwen_person = (row.get("is_anchor") and len(
            qwen_people_by_frame.get(int(row.get("highres_index", -1)), set())) == 1)
        if (person_index >= 0 and vectors[index] is not None
                and (visually_unique or face_count_residual or sole_qwen_person)):
            trusted_anchor_indices.add(index)
            anchors.setdefault(person_index, []).append(vectors[index])
    anchored_people = sorted(anchors)
    if not anchored_people:
        return [dict(row, accepted=bool(row.get("is_anchor")),
                     target_item_index=int(row.get("item_index", -1)),
                     identity_method="direct_qwen_anchor" if row.get("is_anchor") else "no_face_anchor")
                for index, row in enumerate(rows) if index in trusted_anchor_indices]

    accepted = []
    for index, row in enumerate(rows):
        source_index = int(row.get("item_index", -1))
        if index in trusted_anchor_indices:
            accepted.append(dict(row, accepted=True, target_item_index=source_index,
                                 identity_method=("unique_face_count_residual_assignment"
                                                  if "unique_face_count_residual_assignment" in str(row.get("identity_evidence") or "")
                                                  else "unique_visual_identity_seed"
                                                  if "unique_appearance" in str(row.get("identity_evidence") or "")
                                                  else "direct_qwen_anchor"),
                                 identity_score=1.0, identity_margin=1.0))
            continue
        vector = vectors[index]
        scored = []
        if vector is not None:
            for person_index in anchored_people:
                score = max(float(vector @ anchor) for anchor in anchors[person_index])
                scored.append((score, person_index))
        scored.sort(reverse=True)
        best_score, best_person = scored[0] if scored else (-1.0, -1)
        next_score = scored[1][0] if len(scored) > 1 else -1.0
        margin = best_score - next_score if next_score >= 0 else 1.0
        if best_score >= float(min_similarity) and margin >= float(min_margin):
            accepted.append(dict(row, accepted=True, target_item_index=best_person,
                                 identity_method="face_similarity_clip",
                                 identity_score=best_score, identity_margin=margin))
            continue

        # With one known person in a clip, a lone detected face can still be
        # grounded by the existing appearance match. This fallback is disabled
        # for group frames and whenever multiple identities need disambiguation.
        if (len(anchored_people) == 1 and source_index in anchors
                and int(row.get("frame_face_count") or 0) == 1):
            accepted.append(dict(row, accepted=True, target_item_index=source_index,
                                 identity_method="single_face_text_grounded",
                                 identity_score=best_score, identity_margin=margin))

    # A frame may be proposed under more than one text label. Keep one binding
    # for each detected face, preferring a direct anchor and then the strongest
    # face-identity margin.
    deduped = {}
    for row in accepted:
        bbox = tuple(round(float(value), 4) for value in row.get("face_bbox") or [])
        key = (int(row.get("highres_index", -1)), bbox,
               int(row.get("target_item_index", -1)))
        previous = deduped.get(key)
        rank = (bool(row.get("is_anchor")), float(row.get("identity_margin") or 0),
                float(row.get("identity_score") or 0))
        previous_rank = ((bool(previous.get("is_anchor")),
                          float(previous.get("identity_margin") or 0),
                          float(previous.get("identity_score") or 0))
                         if previous else None)
        if previous is None or rank > previous_rank:
            deduped[key] = row
    return list(deduped.values())


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def _grid_boxes(width, height):
    # Multi-scale coarse proposals plus a denser middle/bottom sweep. This
    # improves localization of small hand-held items and documents without
    # assuming a particular clip layout or target position.
    specs = ((.16,.24),(.28,.44),(.46,.72),(.62,.34),(.20,.16),(.32,.24),(.42,.30))
    centers = ((.25,.30),(.75,.30),(.50,.50),(.25,.70),(.75,.70),
               (.25,.90),(.50,.90),(.75,.90),
               (.375,.50),(.625,.50),(.375,.68),(.625,.68),
               (.375,.84),(.625,.84),(.50,.34),(.50,.68))
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
        face_features = str(row.get("face_features") or "").strip()
        details = ", ".join(value for value in (appearance, face_features) if value)
        text = f"a person, {details}" if details else f"a person described as {label}"
        return label, [f"a photo of {text}", f"a close view of {text}"]
    if isinstance(row, dict):
        label = str(row.get("name") or row.get("object") or row.get("entity") or "").strip()
    else:
        label = str(row or "").strip()
    prompts = [
        f"a photo of {label}",
        f"a close-up photo of {label}",
        f"the exact object or product package labeled {label}",
        f"the specific item named {label}, with identifying details visible",
    ]
    # CLIP often reads a generic object name too broadly. Add visual synonyms
    # for common paper artifacts and wipes so the selected local region must
    # resemble the physical item, not just nearby text or a face.
    folded = " ".join(label.casefold().replace("_", " ").split())
    visual_synonyms = {
        "tissue": ("a white facial tissue paper used to wipe a face",
                   "a hand holding a thin paper tissue",
                   "a white tissue held in front of a person's face"),
        "receipt": ("a printed paper receipt with lines of text and a total",
                    "a restaurant receipt on a table or held in a hand"),
        "document": ("a printed paper document or medical form",
                     "a sheet of paper with readable printed text"),
        "price tag": ("a small retail shelf price label",
                      "a price label attached to a store shelf"),
    }
    for synonym in visual_synonyms.get(folded, ()):
        prompts.extend((f"a close-up photo of {synonym}",
                        f"the exact object is {synonym}, with its edges visible"))
    return label, prompts


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


def _object_anchor_frame_indices(label, temporal_facts, frames, source_frame_times=(),
                                 document_evidence=(), object_observations=()):
    """Map an object to direct source frames, preferring explicit object evidence."""
    def normalize(value):
        return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ",
                      str(value or "").casefold()).strip()
    target = normalize(label)
    target_stem = re.sub(r"\s+\d+$", "", target)
    if not target_stem:
        return []
    ranked = {}
    for fact in temporal_facts or ():
        if not isinstance(fact, dict):
            continue
        if str(fact.get("confidence") or "").casefold() in {
                "low", "uncertain", "unknown", "very_low"}:
            continue
        if str(fact.get("modality") or "visual").casefold() not in {"visual", "multimodal"}:
            continue
        subject_type = str(fact.get("subject_type") or "").casefold()
        if subject_type not in {"", "object", "document", "event", "person"}:
            continue
        subject = normalize(fact.get("subject"))
        subject_stem = re.sub(r"\s+\d+$", "", subject)
        evidence_text = normalize(" ".join(str(fact.get(key) or "")
                                            for key in ("value", "evidence", "predicate")))
        direct_subject = bool(subject_stem and
                              (subject_stem == target_stem or
                               (len(min(subject_stem, target_stem, key=len)) >= 4 and
                                (subject_stem in target_stem or target_stem in subject_stem))))
        direct_mention = bool(target_stem and
                              (" " + target_stem + " ") in (" " + evidence_text + " "))
        if not direct_subject and not direct_mention:
            continue
        predicate = normalize(fact.get("predicate"))
        priority = 0 if (direct_subject and predicate in {
            "documented", "held", "holds", "used", "wearing", "placed", "stored",
            "opened", "closed", "received", "purchased", "owned by", "belongs to"}) else 1
        times = []
        refs = [fact.get("evidence_ref")] + list(fact.get("evidence_refs") or [])
        # Qwen frame_index refers to its original input frame list. Map it through
        # timestamps because the cropper may use an expanded candidate list.
        try:
            fact_index = int(fact.get("frame_index"))
        except (TypeError, ValueError):
            fact_index = 0
        if 1 <= fact_index <= len(source_frame_times):
            try:
                times.append(float(source_frame_times[fact_index - 1]))
            except (TypeError, ValueError):
                pass
        elif not source_frame_times and 1 <= fact_index <= len(frames):
            ranked[fact_index] = min(priority, ranked.get(fact_index, priority))
        try:
            if fact.get("frame_time_s") is not None:
                times.append(float(fact["frame_time_s"]))
        except (TypeError, ValueError):
            pass
        for ref in refs:
            if not isinstance(ref, dict) or str(ref.get("modality") or "visual").casefold() != "visual":
                continue
            try:
                if ref.get("frame_time_s") is not None:
                    times.append(float(ref["frame_time_s"]))
            except (TypeError, ValueError):
                pass
            try:
                index = int(ref.get("frame_index"))
                if 1 <= index <= len(source_frame_times):
                    times.append(float(source_frame_times[index - 1]))
                elif not source_frame_times and 1 <= index <= len(frames):
                    ranked[index] = min(priority, ranked.get(index, priority))
            except (TypeError, ValueError):
                pass
        for at in times:
            if not frames:
                continue
            nearest = min(range(len(frames)), key=lambda i:
                          abs(float(frames[i].get("at_s") or 0) - at))
            if abs(float(frames[nearest].get("at_s") or 0) - at) <= 0.8:
                index = nearest + 1
                ranked[index] = min(priority, ranked.get(index, priority))
    # Object observations often carry direct frame references even when the
    # fact extractor attached the action to the user (e.g. “user holds wallet”).
    # Bind those source-grounded refs to the named object before considering
    # document/OCR anchors; this keeps cache warming and runtime crop selection
    # from silently dropping otherwise explicit visual evidence.
    for observation in object_observations or ():
        if not isinstance(observation, dict):
            continue
        observed_label = normalize(observation.get("object") or observation.get("entity"))
        observed_stem = re.sub(r"\s+\d+$", "", observed_label)
        if not observed_stem or not (observed_stem == target_stem or
                (len(min(observed_stem, target_stem, key=len)) >= 4 and
                 (observed_stem in target_stem or target_stem in observed_stem))):
            continue
        refs = [observation.get("evidence_ref")]
        refs.extend(observation.get("evidence_refs") or [])
        for ref in refs:
            if not isinstance(ref, dict) or str(ref.get("modality") or "visual").casefold() != "visual":
                continue
            ref_source = str(ref.get("source_segment_id") or "")
            frame_source = next((str(frame.get("source_id") or "") for frame in frames
                                 if isinstance(frame, dict) and frame.get("source_id")), "")
            if ref_source and frame_source and ref_source != frame_source:
                continue
            at_values = []
            try:
                if ref.get("frame_time_s") is not None:
                    at_values.append(float(ref["frame_time_s"]))
            except (TypeError, ValueError):
                pass
            try:
                ref_index = int(ref.get("frame_index"))
            except (TypeError, ValueError):
                ref_index = 0
            if 1 <= ref_index <= len(source_frame_times):
                try:
                    at_values.append(float(source_frame_times[ref_index - 1]))
                except (TypeError, ValueError):
                    pass
            elif not source_frame_times and 1 <= ref_index <= len(frames):
                ranked[ref_index] = min(-1, ranked.get(ref_index, -1))
            for at in at_values:
                if not frames:
                    continue
                nearest = min(range(len(frames)), key=lambda i:
                              abs(float(frames[i].get("at_s") or 0) - at))
                if abs(float(frames[nearest].get("at_s") or 0) - at) <= 0.8:
                    index = nearest + 1
                    ranked[index] = min(-1, ranked.get(index, -1))
    # OCR/document evidence carries an exact Qwen input-frame index. Use it as a
    # first-class anchor for document-like objects; temporal relation facts may
    # describe the holder/action without citing the frame containing readable text.
    document_terms = {"document", "receipt", "bill", "invoice", "paper", "prescription",
                      "diagnosis", "card", "ticket", "menu", "label"}
    target_terms = set(target.split())
    for row in document_evidence or ():
        if not isinstance(row, dict):
            continue
        doc_text = normalize(" ".join(str(row.get(key) or "") for key in
                                     ("document", "object", "type", "text", "evidence")))
        doc_terms = set(doc_text.split())
        if not (target_terms & doc_terms or
                (target_terms & document_terms and doc_terms & document_terms)):
            continue
        try:
            index = int(row.get("frame_index"))
        except (TypeError, ValueError):
            index = 0
        if 1 <= index <= len(source_frame_times):
            try:
                at = float(source_frame_times[index - 1])
                nearest = min(range(len(frames)), key=lambda i:
                              abs(float(frames[i].get("at_s") or 0) - at))
                if abs(float(frames[nearest].get("at_s") or 0) - at) <= 0.8:
                    ranked[nearest + 1] = -1
            except (TypeError, ValueError):
                pass
        elif not source_frame_times and 1 <= index <= len(frames):
            ranked[index] = -1
    return [index for index, _priority in sorted(ranked.items(), key=lambda row: (row[1], row[0]))]

def _ground(rows, frames, kind, progress_callback=None, temporal_facts=(), face_detections=None,
            source_frame_times=(), document_evidence=(), object_observations=()):
    label_key = "person" if kind == "person" else "object"
    limit = 8 if kind == "person" else 12
    candidates, seen = [], set()
    for value in rows or []:
        label, prompts = _text_for(value, kind)
        if not label:
            continue
        if kind == "person":
            # A model may use the same generic display label (for example,
            # "woman") for several distinct visual observations. Deduplicate
            # only identical anchored observations, never by label alone.
            anchors = tuple(_person_visual_anchor_indices(value, frames))
            appearance = " ".join(str(value.get("appearance") or "").casefold().split())
            face = " ".join(str(value.get("face_features") or "").casefold().split())
            observation_id = str(value.get("observation_id") or "").strip()
            duplicate_key = (label.casefold(), observation_id, anchors, appearance, face)
        else:
            duplicate_key = label.casefold()
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        candidate = {"value": value, "label": label[:100], "prompts": prompts}
        if kind == "person":
            candidate["anchor_frames"] = _person_visual_anchor_indices(value, frames)
        else:
            candidate["anchor_frames"] = _object_anchor_frame_indices(
                label, temporal_facts, frames, source_frame_times, document_evidence,
                object_observations)
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
            requested_device = os.environ.get("MEMORY_CROP_DEVICE", "auto").strip() or "auto"
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
        frame_limit = 18 if kind == "person" else 14
        frame_rows = list(enumerate(frames[:frame_limit], 1))
        if kind == "person":
            # Face clarity and pose can change sharply between input moments;
            # compare every supplied frame instead of four temporal anchors.
            anchor_positions = list(range(len(frame_rows)))
        elif len(frame_rows) > 4:
            last = len(frame_rows) - 1
            anchor_positions = sorted({0, round(last / 3), round(2 * last / 3), last})
            anchor_positions = sorted(set(anchor_positions) | {
                index - 1 for candidate in candidates
                for index in candidate.get("anchor_frames", [])
                if 1 <= index <= len(frame_rows)
            })
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
        hog_keys = {(frame_index, tuple(box)) for frame_index, box in hog_proposals}
        hog_indices_by_frame = {}
        if kind == "person":
            proposals.extend(hog_proposals)
        # Grid candidates cover partial/occluded people; CLIP ranks all proposals on CUDA.
        for frame_index, image in anchor_images.items():
            proposals.extend((frame_index, box)
                             for box in _grid_boxes(*image.size))
        if kind == "object":
            event_anchor_frames = {index for candidate in candidates
                                   for index in candidate.get("anchor_frames", [])
                                   if index in anchor_images}
            # Keep a source-context option for cluttered scenes where similar
            # products make a tight CLIP crop ambiguous.
            proposals.extend((frame_index, (0.0, 0.0, 1.0, 1.0))
                             for frame_index in sorted(event_anchor_frames))
        for frame_index, bbox in proposals:
            image = anchor_images.get(frame_index)
            if image is None:
                image = _load_frame(frames[frame_index - 1])
            if image is not None:
                proposal_images.append(_crop(image, bbox))
        proposals = [proposal for proposal, image in zip(proposals, proposal_images)]
        for proposal_index, (frame_index, bbox) in enumerate(proposals):
            if (frame_index, tuple(bbox)) in hog_keys:
                hog_indices_by_frame.setdefault(frame_index, []).append(proposal_index)
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
        if kind == "person":
            # A Qwen-linked frame establishes identity. Other input frames may
            # contribute gallery crops only when CLIP can distinguish the
            # person's described appearance from every other person in that
            # frame. This restores useful multi-frame portraits without
            # treating every face in a group shot as the same person.
            selected_by_person = [dict() for _ in candidates]
            assigned_by_frame = {}
            frame_proposals = {}
            for proposal_index, (frame_index, _bbox) in enumerate(proposals):
                frame_proposals.setdefault(frame_index, []).append(proposal_index)

            # A frame-level Qwen anchor can contain several people and does not
            # by itself prove which detected face belongs to each label. Resolve
            # group-frame anchors against all appearance prompts jointly, and
            # keep only a clearly discriminative region. This prevents the same
            # visible face (or a nearby person's clothing) from being attached
            # to another person's profile when Qwen's frame anchor is shared.
            anchor_candidates_by_frame = {}
            for candidate_index, candidate in enumerate(candidates):
                for frame_index in candidate["anchor_frames"]:
                    anchor_candidates_by_frame.setdefault(frame_index, []).append(candidate_index)
            for frame_index, anchored_indices in sorted(anchor_candidates_by_frame.items()):
                rows_for_frame = frame_proposals.get(frame_index) or []
                if not rows_for_frame:
                    continue
                proposed = []
                for candidate_index in anchored_indices:
                    ranked = sorted(
                        ((float(scores[index][candidate_index]), index)
                         for index in rows_for_frame), reverse=True)
                    if not ranked:
                        continue
                    score, proposal_index = ranked[0]
                    other_scores = [float(scores[proposal_index][other_index])
                                    for other_index in anchored_indices
                                    if other_index != candidate_index]
                    margin = score - max(other_scores) if other_scores else 1.0
                    # Single-person close-ups keep the direct Qwen anchor path.
                    # In group frames a weak/ambiguous CLIP match is not enough
                    # to publish a face crop; the source frame remains available.
                    if len(anchored_indices) > 1 and (score < 0.23 or margin < 0.06):
                        continue
                    proposed.append((margin, score, candidate_index, proposal_index))
                used = []
                for margin, score, candidate_index, proposal_index in sorted(
                        proposed, reverse=True):
                    bbox = proposals[proposal_index][1]
                    if any(_iou(bbox, prior_bbox) > .15 for prior_bbox in used):
                        continue
                    selected_by_person[candidate_index][frame_index] = (
                        score, proposal_index, "qwen_anchor" if len(anchored_indices) == 1
                        else "qwen_anchor+unique_appearance_margin", margin)
                    used.append(bbox)
                    assigned_by_frame.setdefault(frame_index, []).append(bbox)

            # Qwen frame links are candidate evidence, not a guarantee that a
            # particular person box is the named person. Seed identities from any
            # frame only when CLIP separates both the description from competing
            # profiles and the chosen box from another plausible box. This lets a
            # clear later view establish the profile when an initial group frame
            # is occluded, while refusing ambiguous shared-frame bindings.
            seed_floor = 0.23
            seed_margin = 0.06
            for frame_index, proposal_indices in sorted(frame_proposals.items()):
                if len(candidates) < 1 or frame_index in anchor_candidates_by_frame:
                    continue
                proposed = []
                for candidate_index in range(len(candidates)):
                    ranked = sorted(((float(scores[index][candidate_index]), index)
                                     for index in proposal_indices), reverse=True)
                    if not ranked:
                        continue
                    score, proposal_index = ranked[0]
                    other_scores = [float(scores[proposal_index][other_index])
                                    for other_index in range(len(candidates))
                                    if other_index != candidate_index]
                    identity_margin = score - max(other_scores) if other_scores else 1.0
                    second_boxes = [value for value in ranked[1:]
                                    if _iou(proposals[proposal_index][1],
                                            proposals[value[1]][1]) < 0.20]
                    proposal_margin = score - second_boxes[0][0] if second_boxes else 1.0
                    if score >= seed_floor and identity_margin >= seed_margin and proposal_margin >= 0.015:
                        proposed.append((identity_margin, proposal_margin, score,
                                         candidate_index, proposal_index))
                used = []
                for identity_margin, proposal_margin, score, candidate_index, proposal_index in sorted(
                        proposed, reverse=True):
                    bbox = proposals[proposal_index][1]
                    if any(_iou(bbox, previous) > 0.20 for previous in used):
                        continue
                    selected_by_person[candidate_index][frame_index] = (
                        score, proposal_index, "unique_appearance_seed", identity_margin)
                    used.append(bbox)

            # In a frame with exactly one detected face per clip-level person,
            # allow a single unmatched identity to take the sole remaining face
            # only after all other identities have a high-margin appearance match.
            # This is a generic one-to-one residual assignment; it never runs when
            # the face/person counts differ or more than one identity is unknown.
            for frame_index, proposal_indices in sorted(frame_proposals.items()):
                if not (isinstance(face_detections, list)
                        and 1 <= frame_index <= len(face_detections)
                        and len(candidates) >= 2):
                    continue
                faces = [row for row in (face_detections[frame_index - 1] or [])
                         if isinstance(row, dict)
                         and float(row.get("confidence") or 0) >= 0.25
                         and isinstance(row.get("bbox"), list)
                         and len(row["bbox"]) == 4]
                if len(faces) != len(candidates):
                    continue
                used_faces = {}
                reliable_people = set()
                for candidate_index in range(len(candidates)):
                    current = selected_by_person[candidate_index].get(frame_index)
                    if not current:
                        continue
                    score, _proposal_index, evidence, margin = current
                    if "unique_appearance" not in evidence or score < 0.23 or (margin or 0) < 0.06:
                        continue
                    bbox = proposals[_proposal_index][1]
                    x1, y1, x2, y2 = bbox
                    width, height = max(0.02, x2-x1), max(0.04, y2-y1)
                    possible = []
                    for face_index, face in enumerate(faces):
                        fx1, fy1, fx2, fy2 = face["bbox"]
                        cx, cy = (fx1+fx2)/2, (fy1+fy2)/2
                        if (x1-0.10*width <= cx <= x2+0.10*width
                                and y1-0.20*height <= cy <= y1+0.58*height):
                            possible.append(face_index)
                    if len(possible) == 1 and possible[0] not in used_faces:
                        used_faces[possible[0]] = candidate_index
                        reliable_people.add(candidate_index)
                missing_people = [index for index in range(len(candidates))
                                  if index not in reliable_people]
                missing_faces = [index for index in range(len(faces))
                                 if index not in used_faces]
                if len(missing_people) != 1 or len(missing_faces) != 1:
                    continue
                missing_person, missing_face = missing_people[0], missing_faces[0]
                fx1, fy1, fx2, fy2 = faces[missing_face]["bbox"]
                face_cx, face_cy = (fx1+fx2)/2, (fy1+fy2)/2
                options = []
                for proposal_index in hog_indices_by_frame.get(frame_index, []):
                    bbox = proposals[proposal_index][1]
                    x1, y1, x2, y2 = bbox
                    width, height = max(0.02, x2-x1), max(0.04, y2-y1)
                    if not (x1-0.10*width <= face_cx <= x2+0.10*width
                            and y1-0.20*height <= face_cy <= y1+0.58*height):
                        continue
                    if any(_iou(bbox, proposals[used_index][1]) > 0.20
                           for _face_index, used_person in used_faces.items()
                           for _score, used_index, _evidence, _margin
                           in [selected_by_person[used_person][frame_index]]):
                        continue
                    expected_y = y1 + 0.14*height
                    geometry = abs(face_cx-(x1+x2)/2)/width + 0.7*abs(face_cy-expected_y)/height
                    ratio = height/width
                    shape_penalty = abs(ratio-2.0)*0.08 if ratio < 1.0 or ratio > 4.5 else 0.0
                    options.append((geometry+shape_penalty,
                                    -float(scores[proposal_index][missing_person]), proposal_index))
                if options:
                    _geometry, _negative_score, proposal_index = min(options)
                    score = float(scores[proposal_index][missing_person])
                    selected_by_person[missing_person][frame_index] = (
                        score, proposal_index, "unique_face_count_residual_assignment", 0.0)

            # Calibrate further gallery matches from all visually grounded seeds,
            # rather than requiring the seed to originate in a Qwen anchor frame.
            anchor_scores = {}
            seeded_candidates = set()
            for candidate_index, candidate in enumerate(candidates):
                selected = selected_by_person[candidate_index]
                if selected:
                    seeded_candidates.add(candidate_index)
                anchor_scores[candidate_index] = max(
                    (score for score, _proposal, _source, _margin in selected.values()),
                    default=0.0)

            for frame_index, proposal_indices in sorted(frame_proposals.items()):
                candidate_matches = []
                for candidate_index, candidate in enumerate(candidates):
                    if (candidate_index not in seeded_candidates
                            or not candidate.get("anchor_frames")
                            or frame_index in candidate["anchor_frames"]):
                        continue
                    floor = max(0.18, anchor_scores.get(candidate_index, 0.0) - 0.07)
                    for proposal_index in proposal_indices:
                        score = float(scores[proposal_index][candidate_index])
                        if score < floor:
                            continue
                        other_scores = [float(scores[proposal_index][other_index])
                                        for other_index in range(len(candidates))
                                        if other_index != candidate_index]
                        margin = (score - max(other_scores)) if other_scores else 1.0
                        if other_scores and margin < (0.06 if len(candidates) > 1 else 0.025):
                            continue
                        candidate_matches.append((margin, score, candidate_index,
                                                  proposal_index))

                # The same proposal cannot seed two profiles. Rank by identity
                # margin first, then appearance similarity, and reject spatially
                # overlapping boxes before binding them to detected faces.
                accepted = []
                for margin, score, candidate_index, proposal_index in sorted(
                        candidate_matches, reverse=True):
                    if frame_index in selected_by_person[candidate_index]:
                        continue
                    bbox = proposals[proposal_index][1]
                    if any(_iou(bbox, prior_bbox) > .20 for prior_bbox in
                           assigned_by_frame.get(frame_index, [])):
                        continue
                    if any(_iou(bbox, proposals[prior_index][1]) > .20
                           for _prior_margin, _prior_score, _prior_candidate,
                           prior_index in accepted):
                        continue
                    selected_by_person[candidate_index][frame_index] = (
                        score, proposal_index, "unique_appearance_match", margin)
                    accepted.append((margin, score, candidate_index, proposal_index))
                    assigned_by_frame.setdefault(frame_index, []).append(bbox)

            for candidate_index, candidate in enumerate(candidates):
                selected = selected_by_person[candidate_index]
                if not selected:
                    output_items.append({label_key: candidate["label"],
                                         "status": "uncertain",
                                         "appearance": str(candidate["value"].get("appearance") or ""),
                                         "frame_indices": [], "frame_boxes": [],
                                         "anchor_frame_indices": list(candidate["anchor_frames"]),
                                         "identity_candidate_frame_indices": [],
                                         "reason": "no_direct_visual_anchor_for_person"})
                    continue
                rows = []
                for frame_index, (score, proposal_index, evidence, margin) in sorted(selected.items()):
                    row = {"frame_index": frame_index,
                           "bbox": [round(float(v), 5) for v in proposals[proposal_index][1]],
                           "score": round(score, 5), "method": _METHOD,
                           "identity_evidence": evidence}
                    if margin is not None:
                        row["identity_margin"] = round(float(margin), 5)
                    rows.append(row)
                extras = sorted(frame_index for frame_index, value in selected.items()
                                if value[2] == "unique_appearance_match")
                best_score = max(value[0] for value in selected.values())
                output_items.append({label_key: candidate["label"],
                                     "status": "visible_unlabeled",
                                     "appearance": str(candidate["value"].get("appearance") or ""),
                                     "frame_indices": [row["frame_index"] for row in rows],
                                     "anchor_frame_indices": list(candidate["anchor_frames"]),
                                     "identity_candidate_frame_indices": extras,
                                     "frame_boxes": rows, "score": round(best_score, 5),
                                     "reason": "visually_grounded_unique_appearance_matches"})
        else:
            for candidate_index, candidate in enumerate(candidates):
                ranked = sorted(((float(score[candidate_index]), index)
                                 for index, score in enumerate(scores)), reverse=True)
                if not ranked:
                    ranked = [(0.0, 0)]
                anchors = set(candidate.get("anchor_frames") or [])
                document_anchors = set(_object_anchor_frame_indices(
                    candidate["label"], (), frames, source_frame_times, document_evidence))
                anchored = {}
                ambiguous_frames = set()
                for frame_index in sorted(anchors):
                    rows_for_frame = [index for index, (proposal_frame, _bbox) in enumerate(proposals)
                                      if proposal_frame == frame_index]
                    frame_ranked = sorted(((float(scores[index][candidate_index]), index)
                                           for index in rows_for_frame), reverse=True)
                    if not frame_ranked:
                        continue
                    best_score, best_index = frame_ranked[0]
                    full_frame = next((index for index in rows_for_frame
                                       if proposals[index][1] == (0.0, 0.0, 1.0, 1.0)), None)
                    # Repeated, visually similar products are a common source of
                    # false identity crops. If another non-overlapping region is
                    # nearly tied, retain the linked source frame instead of
                    # presenting an incorrect tight crop as the named object.
                    competitors = [(score_value, proposal_index) for score_value, proposal_index
                                   in frame_ranked[1:]
                                   if _iou(proposals[best_index][1],
                                           proposals[proposal_index][1]) < 0.15]
                    full_score = (float(scores[full_frame][candidate_index])
                                  if full_frame is not None else -1.0)
                    weak_localization = (
                        proposals[best_index][1] != (0.0, 0.0, 1.0, 1.0)
                        and (best_score < _OBJECT_LOCAL_SCORE_MIN
                             or best_score - full_score < _OBJECT_LOCAL_MARGIN_MIN))
                    near_tie = bool(competitors and best_score - competitors[0][0] < 0.025)
                    if (full_frame is not None and (weak_localization or near_tie)
                            and frame_index not in document_anchors):
                        anchored[frame_index] = (full_score, full_frame)
                        ambiguous_frames.add(frame_index)
                        continue
                    anchored[frame_index] = (best_score, best_index)
                if anchored:
                    preferred = [index for index in candidate.get("anchor_frames") or []
                                 if index in anchored]
                    selected = [(frame_index, *anchored[frame_index])
                                for frame_index in preferred[:3]]
                else:
                    output_items.append({label_key: candidate["label"],
                                         "status": "uncertain",
                                         "appearance": "",
                                         "frame_indices": [],
                                         "anchor_frame_indices": [],
                                         "frame_boxes": [],
                                         "score": 0.0,
                                         "reason": "no_source_evidence_anchor"})
                    continue
                frame_boxes = []
                for frame_index, score_value, proposal_index in selected:
                    _frame, bbox = proposals[proposal_index]
                    frame_boxes.append({"frame_index": frame_index,
                                        "bbox": [round(float(v), 5) for v in bbox],
                                        "score": round(score_value, 5), "method": _METHOD,
                                        "identity_evidence": ("qwen_event_anchor_context"
                                                              if frame_index in ambiguous_frames
                                                              else "qwen_event_anchor")
                                        if frame_index in anchors else "clip_best_region"})
                output_items.append({label_key: candidate["label"],
                                     "status": "visible_unlabeled",
                                     "appearance": "",
                                     "frame_indices": [row["frame_index"] for row in frame_boxes],
                                     "anchor_frame_indices": sorted(anchors),
                                     "frame_boxes": frame_boxes,
                                     "score": round(max(row["score"] for row in frame_boxes), 5),
                                     "reason": ("ambiguous_object_region_use_context_frame"
                                                if ambiguous_frames else "source_evidence_anchor_best_regions")
                                     if anchors else "no_source_evidence_anchor"})
        if kind == "object":
            # Collapse numbered aliases only when independent grounding lands on
            # the exact same evidence frame and region. This prevents one object
            # from appearing as several cards without merging distinct instances.
            canonical_items = []
            collapsed_aliases = []
            for item in output_items:
                label = str(item.get("object") or "").strip()
                match = re.fullmatch(r"(.+?)\s+\d+", label)
                if match:
                    stem = match.group(1).strip().casefold()
                    signature = (
                        tuple(item.get("anchor_frame_indices") or []),
                        tuple((int(box.get("frame_index") or 0),
                               tuple(round(float(value), 4) for value in box.get("bbox") or []))
                              for box in item.get("frame_boxes") or []
                              if isinstance(box, dict)),
                    )
                    prior = next((row for row in canonical_items
                                  if str(row.get("object") or "").strip().casefold() == stem), None)
                    if prior is not None:
                        prior_signature = (
                            tuple(prior.get("anchor_frame_indices") or []),
                            tuple((int(box.get("frame_index") or 0),
                                   tuple(round(float(value), 4) for value in box.get("bbox") or []))
                                  for box in prior.get("frame_boxes") or []
                                  if isinstance(box, dict)),
                        )
                        if signature[0] and signature == prior_signature:
                            collapsed_aliases.append({"alias": label,
                                                      "canonical": prior.get("object"),
                                                      "reason": "same_source_anchor_and_region"})
                            continue
                canonical_items.append(item)
            output_items = canonical_items
            if collapsed_aliases:
                base["collapsed_object_aliases"] = collapsed_aliases
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


def ground_objects(objects, frames, cancel_event=None, progress_callback=None,
                   temporal_facts=(), source_frame_times=(), document_evidence=(),
                   object_observations=()):
    if cancel_event is not None and cancel_event.is_set():
        return {"schema":1,"status":"cancelled","model":_METHOD,"items":[]}
    return _ground(objects, frames, "object", progress_callback, temporal_facts,
                   source_frame_times=source_frame_times, document_evidence=document_evidence,
                   object_observations=object_observations)


def ground_people(people, frames, cancel_event=None, progress_callback=None,
                  face_detections=None):
    if cancel_event is not None and cancel_event.is_set():
        return {"schema":1,"status":"cancelled","model":_METHOD,"items":[]}
    return _ground(people, frames, "person", progress_callback,
                   face_detections=face_detections)


def warmup():
    """Load the local CLIP ranker on its configured CUDA device."""
    model, _processor = _model()
    return {"model": _MODEL_ID, "device": str(_DEVICE), "ready": model is not None}
