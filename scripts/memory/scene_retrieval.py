"""Write-after-scene retrieval orchestration for the interactive memory runner."""

import json
import threading
import time
import uuid
from urllib.parse import quote

from entity_records import EntityRecordIndex, memory_facts
from memory_search import _retrieval_tokens


_GENERIC_ANCHORS = {
    "man", "woman", "person", "people", "user", "speaker", "viewer", "camera",
    "unknown", "unidentified", "visible", "adult", "someone", "anyone", "one",
    "thing", "item", "object", "stuff", "background", "scene", "setting",
    "indoor", "indoors", "outdoor", "outdoors", "place", "area", "space",
    "audio", "speech", "transcript", "video", "frame", "en", "zh",
    "table", "door", "phone", "document", "paper", "box", "bag",
    "clothing", "clothes", "shirt", "coat", "jacket", "camera",
    "speaker", "customer", "client", "employee", "staff", "doctor",
    "patient", "boss", "manager", "stranger", "neighbor", "neighbour",
    "son", "daughter", "mother", "father", "mom", "dad",
    # Common appearance and setting words are not identity evidence by themselves.
    "expression", "light", "short", "wearing", "standing", "sitting", "seated",
    "hair", "face", "eyes", "mouth", "smile", "smiling", "gray", "grey",
    "dark", "black", "blue", "white", "green", "red", "pink", "brown",
    "color", "coloured", "colored", "clothing", "body", "build", "adult",
    "near", "behind", "front", "left", "right", "side", "inside",
    "office", "room", "home", "house", "restaurant", "street", "city",
    # Common home furnishings and size words are weak background context, not episode links.
    "apartment", "living", "livingroom", "furniture", "sofa", "couch", "curtain", "curtains", "coffee", "large", "small", "big",
    "desk", "chair", "table", "window", "door", "building", "floor",
    "jacket", "coat", "shirt", "tie", "pants", "dress", "glasses",
}


def _anchor_terms(*values):
    tokens = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            nested = value
        else:
            nested = [value]
        for item in nested:
            if not item:
                continue
            for token in _retrieval_tokens(str(item).casefold()):
                if (len(token) > 1 and token.isalpha()
                        and token not in _GENERIC_ANCHORS):
                    tokens.add(token)
    return tokens


def _moment_anchor_sets(moment):
    moment = moment if isinstance(moment, dict) else {}
    people = list(moment.get("people") or [])
    person_rows = [row for row in moment.get("person_observations") or []
                   if isinstance(row, dict)]
    person_features = []
    for row in person_rows:
        # Entity labels and source-linked spoken aliases can identify a recurring
        # person. Episode roles (speaker, doctor, customer, etc.) describe function,
        # not identity, so they must not act as person-match anchors.
        people.append(row.get("entity"))
        for alias in row.get("spoken_aliases") or []:
            if isinstance(alias, dict) and alias.get("status") == "probable_visual_match":
                people.append(alias.get("alias"))
        people.extend(row.get("relationships") or [])
        person_features.extend([row.get("appearance"), row.get("face_features")])
    objects = list(moment.get("objects") or [])
    objects.extend(row.get("object") for row in moment.get("object_observations") or []
                   if isinstance(row, dict))
    return {
        "people": _anchor_terms(people),
        "person_features": _anchor_terms(person_features),
        "objects": _anchor_terms(objects),
        # Keep static scene context separate from event/topic wording. This
        # lets a recurring place/environment retrieve across different actions
        # without turning generic same-day language into a match.
        "context": _anchor_terms(moment.get("location"), moment.get("activity")),
        "environment": _anchor_terms(moment.get("environments")),
        "event": _anchor_terms(
            moment.get("scene"), moment.get("content_summary"),
            moment.get("user_action"), moment.get("interaction_details"),
            moment.get("source_evidence"),
        ),
    }


_IDENTITY_WARMUP_LOCK = threading.Lock()
_IDENTITY_WARMUP_ACTIVE = False
_IDENTITY_WARMUP_PENDING = None


def _identity_warmup_worker():
    """Fill missing face embeddings off the interactive retrieval path."""
    global _IDENTITY_WARMUP_ACTIVE, _IDENTITY_WARMUP_PENDING
    while True:
        with _IDENTITY_WARMUP_LOCK:
            job = _IDENTITY_WARMUP_PENDING
            _IDENTITY_WARMUP_PENDING = None
            if job is None:
                _IDENTITY_WARMUP_ACTIVE = False
                return
        moments, asset_root, correction_path = job
        try:
            EntityRecordIndex(
                {"moments": moments}, asset_root=asset_root,
                identity_correction_path=correction_path,
                compute_missing_face_embeddings=True)
        except Exception:
            # Retrieval is already served from text and cached identity evidence.
            pass


def _schedule_identity_warmup(archive, asset_root, correction_path):
    """Coalesce face-index rebuild requests and process only the latest snapshot."""
    global _IDENTITY_WARMUP_ACTIVE, _IDENTITY_WARMUP_PENDING
    with _IDENTITY_WARMUP_LOCK:
        _IDENTITY_WARMUP_PENDING = (dict(archive), asset_root, correction_path)
        if _IDENTITY_WARMUP_ACTIVE:
            return
        _IDENTITY_WARMUP_ACTIVE = True
    try:
        threading.Thread(
            target=_identity_warmup_worker, daemon=True,
            name="memory-face-identity-warmup").start()
    except RuntimeError:
        with _IDENTITY_WARMUP_LOCK:
            _IDENTITY_WARMUP_ACTIVE = False


_SPOKEN_ROLE_GROUPS = {
    "maternal_grandmother": ("姥姥", "外婆", "外祖母"),
    "paternal_grandmother": ("奶奶", "祖母", "祖奶奶"),
    "grandparent_unspecified": ("grandma", "grandmother", "grandparent"),
    "maternal_grandfather": ("外公", "外祖父"),
    "paternal_grandfather": ("爷爷", "祖父", "祖爷爷"),
    "grandfather_unspecified": ("grandpa", "grandfather"),
    "mom": ("妈妈", "母亲", "妈", "mom", "mother", "mum"),
    "dad": ("爸爸", "父亲", "爸", "dad", "father"),
    "neighbor": ("邻居", "neighbor", "neighbour"),
    "friend": ("朋友", "friend"),
    "sibling": ("哥哥", "姐姐", "弟弟", "妹妹", "brother", "sister", "sibling"),
}


def _spoken_role_evidence(moment):
    """Return directly spoken role anchors, never treating a role as a face identity."""
    found = {}
    for row in moment.get("person_observations") or []:
        if not isinstance(row, dict):
            continue
        refs = list(row.get("evidence_refs") or [])
        if isinstance(row.get("evidence_ref"), dict):
            refs.append(row["evidence_ref"])
        speech_ref = next((ref for ref in refs if isinstance(ref, dict)
                           and ref.get("modality") == "speech"), None)
        if not speech_ref:
            continue
        claims = [str(row.get("entity") or "")]
        claims.extend(str(value) for value in row.get("relationships") or [])
        claims.extend(str(alias.get("alias") or "") for alias in row.get("spoken_aliases") or []
                      if isinstance(alias, dict)
                      and alias.get("status") == "probable_visual_match"
                      and isinstance(alias.get("evidence_ref"), dict)
                      and alias["evidence_ref"].get("modality") == "speech")
        for role, aliases in _SPOKEN_ROLE_GROUPS.items():
            if any(alias.casefold() in claim.casefold()
                   for alias in aliases for claim in claims if alias):
                evidence = str(row.get("evidence") or "")[:280]
                found.setdefault(role, {
                    "role": role, "evidence_ref": dict(speech_ref),
                    "evidence": evidence,
                    "confidence": row.get("confidence") or row.get("profile_confidence") or "medium",
                })
    return found


class PostWriteRetrievalMixin:
    """Retrieve historical memories before and after a new moment is stored."""

    def _retrieve_before_write(self, source_segment_id, fields, dataset_id="", progress=None):
        """Retrieve source-linked history from the observed scene before ingesting it."""
        started = time.perf_counter()
        fields = fields if isinstance(fields, dict) else {}
        archive = self.mem.memory.get("layer7", {}).get("moments", {})
        people = [row for row in fields.get("person_observations") or [] if isinstance(row, dict)]
        people_text = [" ".join(str(row.get(key) or "") for key in
                                 ("entity", "appearance", "face_features", "relationships")).strip()
                       for row in people[:8]]
        object_rows = [row for row in fields.get("object_observations") or [] if isinstance(row, dict)]
        object_text = [" ".join(str(row.get(key) or "") for key in
                                 ("object", "appearance", "evidence")).strip()
                       for row in object_rows[:8]]
        observed = [
            "Scene: " + str(fields.get("scene") or ""),
            "Content summary: " + str(fields.get("content_summary") or ""),
            "Action: " + str(fields.get("user_action") or ""),
            "People: " + "; ".join(value for value in people_text if value),
            "Location: " + str(fields.get("location") or ""),
            "Activity: " + str(fields.get("activity") or ""),
            "Objects: " + ", ".join(str(value) for value in fields.get("objects") or []),
            "Object observations: " + "; ".join(value for value in object_text if value),
            "Environment: " + ", ".join(str(value) for value in fields.get("environments") or []),
            "Speech: " + str(fields.get("source_evidence") or ""),
        ]
        situation = "Current observed video scene: " + " ".join(
            item for item in observed if item.split(": ", 1)[-1].strip())
        situation = situation[:1600].strip()
        if not situation.strip(" :"):
            return [], {"status": "skipped", "message": "当前片段没有可检索的场景信息"}
        cache_context = {
            "source_segment_id": str(source_segment_id or ""),
            "retrieval_only": True,
            "memory_modalities": "text",
            "retrieval_policy_version": "scene-memory-prewrite-v7-specific-environment-anchors",
            "observed_scene": situation,
            "people_observations": people_text[:8],
            "object_observations": object_text[:8],
            "source_video_speech": str(fields.get("source_evidence") or "")[:1000],
        }
        def set_progress(stage, message, text):
            if progress:
                self._set_step_progress(**progress, stage=stage, message=message,
                                        output={"id": dataset_id, "text": text})
        set_progress("prewrite_retrieval", "正在写入前检索相关历史记忆",
                     "用当前视频的人物、场景、物品和语音查询已有记忆")
        try:
            cache_key, stable_ids = self.retrieval_cache.key(
                self.mem.memory, self.dataset_id, "scene_retrieval_before_write_v2",
                situation, cache_context)
            cached, lookup_ms = self.retrieval_cache.get(cache_key, stable_ids)
        except Exception:
            cache_key, stable_ids, cached, lookup_ms = None, {}, None, 0
        cache_meta = {"status": "miss", "lookup_ms": lookup_ms}
        if cached:
            result, record = cached
            cache_meta = {"status": "hit", "lookup_ms": lookup_ms,
                          "duration_ms": record.get("duration_ms", record.get("inference_ms", 0)),
                          "duration_kind": record.get("duration_kind", "legacy_unknown")}
        else:
            cancel_event = self._active_step_cancel
            if cancel_event is not None and cancel_event.is_set():
                return [], {"status": "cancelled", "message": "写入前检索已停止"}
            request_id = f"prewrite-retrieval-{uuid.uuid4().hex}"
            with self._cancel_lock:
                self._active_reasoning_request_id = request_id
            try:
                set_progress("prewrite_retrieval", "正在写入前检索相关历史记忆",
                             "先召回历史候选；当前片段尚未写入记忆")
                # Use the already-built lexical/semantic memory index here. This is a
                # retrieval step, not a second Qwen reasoning call; it keeps the write
                # path responsive and gives profile enrichment source-grounded history.
                # Retrieve a broad lexical pool, then apply corpus-wide rare-anchor
                # checks. Ranking the top five before grounding was the source of
                # unrelated matches when generic words dominated the query.
                ranked = self.mem.query_hierarchical(situation, top_k=max(35, len(archive)))
                candidates = []
                candidate_details = []
                # Old evaluation runs can leave several records for the same
                # source clip. Count one source episode once so duplicates do
                # not make its genuinely shared setting/topic appear common.
                canonical = {}
                for mid, moment in archive.items():
                    if not isinstance(moment, dict) or moment.get("stale") or moment.get("merged_into"):
                        continue
                    provenance = moment.get("source_provenance") or {}
                    source_time = moment.get("source_time") or {}
                    source_id = str(provenance.get("source_segment_id")
                                    or source_time.get("source_segment_id") or "").strip()
                    source_key = source_id or "moment:" + str(mid)
                    quality = (
                        3 * len(moment.get("person_observations") or [])
                        + 2 * len(moment.get("object_observations") or [])
                        + len(str(moment.get("source_evidence") or ""))
                        + len(str(moment.get("scene") or ""))
                        + len(str(moment.get("content_summary") or "")),
                        str(moment.get("timestamp") or ""), str(mid),
                    )
                    previous = canonical.get(source_key)
                    if previous is None or quality > previous[0]:
                        canonical[source_key] = (quality, str(mid), moment)
                canonical_moments = {mid: moment for _, mid, moment in canonical.values()}
                canonical_source_ids = {key: mid for key, (_, mid, _) in canonical.items()}
                all_anchors = {mid: _moment_anchor_sets(moment)
                               for mid, moment in canonical_moments.items()}
                anchor_kinds = ("people", "person_features", "objects", "context",
                                "environment", "event")
                document_frequency = {kind: {} for kind in anchor_kinds}
                for anchor_sets in all_anchors.values():
                    for kind, terms in anchor_sets.items():
                        for term in terms:
                            document_frequency[kind][term] = document_frequency[kind].get(term, 0) + 1
                # Terms appearing in more than four distinct source episodes
                # are too generic to establish a cross-scene relation alone.
                def rare(anchors, kind):
                    return {term for term in anchors.get(kind, set())
                            if document_frequency[kind].get(term, 0) <= 4}
                current_raw = _moment_anchor_sets(fields)
                current_anchors = {kind: rare(current_raw, kind) for kind in document_frequency}
                scored = []
                seen_canonical_hits = set()
                for hit in ranked:
                    hit_moment = hit.get("moment") or {}
                    hit_mid = str(hit_moment.get("id") or "")
                    hit_provenance = hit_moment.get("source_provenance") or {}
                    hit_source_time = hit_moment.get("source_time") or {}
                    hit_source_id = str(hit_provenance.get("source_segment_id")
                                        or hit_source_time.get("source_segment_id") or "").strip()
                    mid = canonical_source_ids.get(hit_source_id, hit_mid)
                    # Ignore noncanonical duplicate copies of a source clip.
                    if mid not in canonical_moments or mid in seen_canonical_hits:
                        continue
                    seen_canonical_hits.add(mid)
                    moment = canonical_moments[mid]
                    provenance = moment.get("source_provenance") or {}
                    stored_segment = str(provenance.get("source_segment_id") or "")
                    if (not mid or mid == str(source_segment_id or "")
                            or (source_segment_id and stored_segment == str(source_segment_id))):
                        continue
                    historical_raw = all_anchors.get(mid) or _moment_anchor_sets(moment)
                    historical_anchors = {kind: rare(historical_raw, kind)
                                          for kind in document_frequency}
                    shared_people = sorted(current_anchors["people"] & historical_anchors["people"])
                    shared_features = sorted(current_anchors["person_features"] & historical_anchors["person_features"])
                    shared_objects = sorted(current_anchors["objects"] & historical_anchors["objects"])
                    shared_context = sorted(current_anchors["context"] & historical_anchors["context"])
                    shared_environment = sorted(current_anchors["environment"] & historical_anchors["environment"])
                    shared_event = sorted(current_anchors["event"] & historical_anchors["event"])
                    reasons = []
                    if shared_people:
                        reasons.append("共享具名人物 / 关系线索：" + "、".join(shared_people[:6]))
                    if len(shared_features) >= 2:
                        reasons.append("共享低频人物外观线索：" + "、".join(shared_features[:6]))
                    if shared_objects:
                        reasons.append("共享低频物品线索：" + "、".join(shared_objects[:6]))
                    if len(shared_context) >= 2:
                        reasons.append("共享低频地点 / 活动线索：" + "、".join(shared_context[:6]))
                    if len(shared_environment) >= 2:
                        reasons.append("共享场景环境线索：" + "、".join(shared_environment[:6]))
                    if len(shared_event) >= 3:
                        reasons.append("共享事件主题线索：" + "、".join(shared_event[:6]))
                    # A single highly-specific object or direct person anchor is
                    # sufficient. Context-only matches need two rare terms; clothing
                    # resemblance needs two rare features to avoid false identity links.
                    lexical_score = float(hit.get("score", 0) or 0)
                    strong_match = bool(shared_people or shared_objects)
                    visually_supported_match = (
                        lexical_score >= 55 and
                        (len(shared_features) >= 2 or len(shared_context) >= 2))
                    # A setting can recur across adjacent clips even when the
                    # action and wording differ. Three rare environment anchors
                    # are sufficient without requiring lexical-rank inflation.
                    environment_match = len(shared_environment) >= 3
                    event_topic_match = lexical_score >= 60 and len(shared_event) >= 3
                    blended_context_match = (
                        lexical_score >= 60
                        and len(set(shared_context + shared_environment + shared_event)) >= 3
                        and bool(shared_context or shared_environment)
                    )
                    if not (strong_match or visually_supported_match or environment_match
                            or event_topic_match or blended_context_match):
                        continue
                    anchor_score = (5.0 * bool(shared_people) + 4.0 * bool(shared_objects)
                                    + 1.5 * min(len(shared_features), 4)
                                    + 1.0 * min(len(shared_context), 4)
                                    + 1.1 * min(len(shared_environment), 4)
                                    + 0.65 * min(len(shared_event), 5)
                                    + float(hit.get("score", 0) or 0))
                    candidate = {
                        "id": mid, "time": moment.get("timestamp", ""),
                        "scene": moment.get("scene", ""),
                        "action": moment.get("user_action", ""),
                        "people": moment.get("people") or [],
                        "location": moment.get("location", ""),
                        "activity": moment.get("activity", ""),
                        "objects": moment.get("objects") or [],
                        "score": hit.get("score", 0),
                    }
                    detail = {
                        "id": mid, "text_relevance": hit.get("score", 0),
                        "anchor_score": round(anchor_score, 3),
                        "matched_terms": sorted(set(
                            shared_people + shared_features + shared_objects + shared_context
                            + shared_environment + shared_event))[:16],
                        # Retrieval method is diagnostic metadata, not a semantic match reason.
                        "reasons": reasons,
                        "retrieval_method": hit.get("retrieval_method"),
                    }
                    scored.append((anchor_score, candidate, detail))
                scored.sort(key=lambda item: item[0], reverse=True)
                for _, candidate, detail in scored[:5]:
                    candidates.append(candidate)
                    candidate_details.append(detail)
                result = {"mode": "memory_retrieval", "candidates": candidates,
                          "retrieval_rounds": [{"candidate_details": candidate_details}],
                          "retrieval_method": "indexed_prewrite_retrieval"}
                elapsed = round((time.perf_counter() - started) * 1000)
                stored = bool(cache_key) and self.retrieval_cache.put_retrieval(
                    cache_key, result, stable_ids, elapsed)
                cache_meta = {"status": "stored" if stored else "uncached",
                              "lookup_ms": lookup_ms, "duration_ms": elapsed,
                              "duration_kind": "prewrite_retrieval"}
            except Exception as exc:
                cancelled = bool(cancel_event is not None and cancel_event.is_set())
                return [], {"status": "cancelled" if cancelled else "error",
                            "message": "写入前检索已停止" if cancelled else str(exc)[:350],
                            "duration_ms": round((time.perf_counter() - started) * 1000)}
            finally:
                with self._cancel_lock:
                    if self._active_reasoning_request_id == request_id:
                        self._active_reasoning_request_id = None
        details = {}
        for round_result in result.get("retrieval_rounds") or []:
            for detail in round_result.get("candidate_details") or []:
                if isinstance(detail, dict) and detail.get("id"):
                    details[str(detail["id"])] = detail
        output = []
        for candidate in result.get("candidates") or []:
            mid = str(candidate.get("id") or "")
            if not mid:
                continue
            moment = archive.get(mid) or {}
            if not moment:
                continue
            detail = details.get(mid) or {}
            mm = moment.get("multimodal") or {}
            output.append({
                "id": mid, "layer": 7, "matched_layers": [7], **memory_facts(moment),
                "score": detail.get("text_relevance"),
                "match_reasons": detail.get("reasons") or [],
                "matched_terms": detail.get("matched_terms") or [],
                "entity_matches": detail.get("entity_matches") or [],
                "multimodal_link": detail.get("multimodal_link") or {},
                "timestamp": moment.get("timestamp") or candidate.get("time", ""),
                "scene": moment.get("scene") or candidate.get("scene", ""),
                "action": moment.get("user_action") or candidate.get("action", ""),
                "people": moment.get("people") or candidate.get("people") or [],
                "person_observations": moment.get("person_observations") or [],
                "object_observations": moment.get("object_observations") or [],
                "location": moment.get("location") or candidate.get("location", ""),
                "activity": moment.get("activity") or candidate.get("activity", ""),
                "objects": moment.get("objects") or candidate.get("objects") or [],
                "environments": moment.get("environments") or [],
                "summary": moment.get("content_summary") or moment.get("summary", ""),
                "source_evidence": moment.get("source_evidence") or "",
                "source_provenance": moment.get("source_provenance") or {},
                "multimodal": mm,
            })
            if len(output) >= 5:
                break
        total_ms = round((time.perf_counter() - started) * 1000)
        cache_meta["wrapper_elapsed_ms"] = total_ms
        return output, {"status": "hit" if cached else cache_meta.get("status"),
                        "duration_ms": cache_meta.get("duration_ms", 0),
                        "cache": cache_meta, "wrapper_elapsed_ms": total_ms,
                        "retrieval_method": result.get("retrieval_method", "cached_indexed_retrieval"),
                        "used_for_current_memory": bool(output)}

    def _retrieve_after_write(self, current_moment_id, fields, dataset_id="", progress=None):
        """Understand the just-observed scene and retrieve source-linked history only."""
        wrapper_started = time.perf_counter()
        wrapper_timings = {}
        fields = fields if isinstance(fields, dict) else {}
        current_moment_id = str(current_moment_id or "")
        archive = self.mem.memory.get("layer7", {}).get("moments", {})
        current = archive.get(current_moment_id) or {}
        if not current_moment_id or not current:
            return [], {"status": "skipped", "message": "当前片段尚无可检索的已写入记忆"}

        def values(value):
            if isinstance(value, (list, tuple, set)):
                return [str(item) for item in value if item]
            return [str(value)] if value else []

        multimodal = current.get("multimodal") or {}
        object_visuals = multimodal.get("object_visuals") or {}
        visual_observations = []
        if isinstance(object_visuals, dict):
            for item in object_visuals.get("items") or []:
                if not isinstance(item, dict):
                    continue
                appearance = str(item.get("appearance") or "").strip()
                label = str(item.get("object") or "").strip()
                if appearance:
                    visual_observations.append(
                        f"{label}: {appearance}" if label else appearance)
        observed = [
            "Scene: " + str(fields.get("scene") or current.get("scene") or ""),
            "Action: " + str(fields.get("user_action") or current.get("user_action") or ""),
            "People: " + ", ".join(values(fields.get("people") or current.get("people"))),
            "Location: " + str(fields.get("location") or current.get("location") or ""),
            "Activity: " + str(fields.get("activity") or current.get("activity") or ""),
            "Objects: " + ", ".join(values(fields.get("objects") or current.get("objects"))),
            "Environment: " + ", ".join(values(fields.get("environments") or current.get("environments"))),
        ]
        spoken_evidence = str(current.get("source_evidence") or "").strip()
        situation = "Current observed scene: " + " ".join(
            part for part in observed if part.split(": ", 1)[-1].strip())
        if visual_observations:
            situation += " Visual object observations: " + "; ".join(visual_observations[:8])
        if spoken_evidence:
            situation += " Source-video speech transcript: " + spoken_evidence[:700]
        situation = situation[:1000]
        if not situation.strip(" :"):
            return [], {"status": "skipped", "message": "当前片段没有可用的场景描述"}

        cache_context = {
            "source_facts": memory_facts(current),
            "source_moment_id": current_moment_id,
            "retrieval_only": True,
            "memory_modalities": "text",
            "retrieval_policy_version": "scene-memory-retrieval-v8-identity-link-object-state",
            "observed_scene": situation,
            "visual_object_observations": visual_observations[:8],
            "source_video_speech": spoken_evidence[:700],
        }

        def set_progress(stage, message, text):
            if progress:
                self._set_step_progress(
                    **progress, stage=stage, message=message,
                    output={"id": dataset_id, "text": text},
                )

        set_progress("retrieval_cache_lookup", "正在检查场景检索缓存",
                     "校验当前片段、完整记忆上下文和检索版本")
        try:
            cache_key, stable_ids = self.retrieval_cache.key(
                self.mem.memory, self.dataset_id, "scene_retrieval_after_write_v6",
                situation, cache_context,
            )
            cached, lookup_ms = self.retrieval_cache.get(cache_key, stable_ids)
        except Exception:
            cache_key, stable_ids, cached, lookup_ms = None, {}, None, 0

        if cached:
            result, cache_record = cached
            cache_status = {
                "status": "hit", "lookup_ms": lookup_ms,
                "inference_ms": cache_record.get("inference_ms", 0),
                "duration_ms": cache_record.get("duration_ms", cache_record.get("inference_ms", 0)),
                "duration_kind": cache_record.get("duration_kind", "legacy_unknown"),
                "scene_plan": {"status": "not_needed"},
                "created_at": cache_record.get("created_at"),
            }
            set_progress("retrieval_cache_hit", "场景检索命中持久缓存，跳过模型调用",
                         "已复用相同场景、记忆上下文与模型版本的结果")
        else:
            set_progress("scene_plan_cache_lookup", "正在检查场景理解计划缓存",
                         "该缓存只复用当前片段的理解与查询计划，历史候选会按当前记忆重新检索")
            try:
                plan_key, plan_stable_ids = self.scene_plan_cache.key_scene_plan(
                    self.dataset_id, situation, cache_context,
                )
                plan_cached, plan_lookup_ms = self.scene_plan_cache.get(
                    plan_key, plan_stable_ids)
            except Exception:
                plan_key, plan_stable_ids = None, {}
                plan_cached, plan_lookup_ms = None, 0
            cached_scene_plan = None
            if plan_cached:
                plan_payload, plan_record = plan_cached
                cached_scene_plan = plan_payload.get("scene_plan")
                plan_cache_status = {
                    "status": "hit", "lookup_ms": plan_lookup_ms,
                    "inference_ms": plan_record.get("inference_ms", 0),
                    "duration_ms": plan_record.get("duration_ms", plan_record.get("inference_ms", 0)),
                    "duration_kind": plan_record.get("duration_kind", "legacy_unknown"),
                }
            else:
                plan_record = {}
                plan_cache_status = {"status": "miss", "lookup_ms": plan_lookup_ms}
            cancel_event = self._active_step_cancel
            if cancel_event is not None and cancel_event.is_set():
                return [], {"status": "cancelled", "message": "场景检索已停止"}
            request_id = f"retrieval-{uuid.uuid4().hex}"
            with self._cancel_lock:
                self._active_reasoning_request_id = request_id
            if cached_scene_plan:
                set_progress("scene_retrieval", "结构化场景计划命中，正在刷新检索结果",
                             "按当前记忆重新检索历史记录")
            else:
                set_progress("scene_retrieval", "正在按当前场景检索相关历史记忆",
                             "使用已写入的人物、地点、事件和物品事实，不调用建议推理")
            started = time.perf_counter()
            try:
                result = self.daytime.retrieve_relevant_memories(situation, context={
                    "source_moment_id": current_moment_id,
                    "retrieval_only": True,
                    "memory_modalities": "text",
                    "reasoning_request_id": request_id,
                    "cancellation_event": cancel_event,
                    "current_scene_observations": {
                        "visual_objects": visual_observations[:8],
                    },
                    **({"cached_scene_plan": cached_scene_plan}
                       if isinstance(cached_scene_plan, dict) else {}),
                })
                retrieval_elapsed_ms = int(result.get("elapsed_ms") or
                                           (time.perf_counter() - started) * 1000)
                if not cached_scene_plan:
                    planner = result.get("search_plan") or {}
                    scene_plan = {
                        "situation": planner.get("situation") or situation,
                        "entities": planner.get("entities") or [],
                        "searches": planner.get("proposed_searches") or planner.get("searches") or [],
                        "scene_dimensions": planner.get("scene_dimensions") or {},
                        "planner_method": result.get("scene_plan_method") or
                                          planner.get("planner_method") or "unknown",
                    }
                    plan_payload = {"mode": "scene_plan", "scene_plan": scene_plan}
                    plan_stored = bool(plan_key) and self.scene_plan_cache.put_scene_plan(
                        plan_key, plan_payload, plan_stable_ids,
                        result.get("scene_plan_elapsed_ms") or 0,
                        duration_kind=result.get("scene_plan_method") or "unknown_scene_plan",
                    )
                    plan_cache_status = {
                        "status": "stored" if plan_stored else "uncached",
                        "lookup_ms": plan_lookup_ms,
                        "inference_ms": result.get("scene_plan_elapsed_ms", 0),
                        "duration_ms": result.get("scene_plan_elapsed_ms", 0),
                        "duration_kind": result.get("scene_plan_method") or "unknown_scene_plan",
                    }
                stored = bool(cache_key) and self.retrieval_cache.put_retrieval(
                    cache_key, result, stable_ids, retrieval_elapsed_ms)
                cache_status = {
                    "status": "stored" if stored else "uncached",
                    "lookup_ms": lookup_ms,
                    "inference_ms": retrieval_elapsed_ms,  # Backward-compatible alias.
                    "duration_ms": retrieval_elapsed_ms,
                    "duration_kind": "retrieval_total",
                    "scene_plan": plan_cache_status,
                }
            except Exception as exc:
                cancelled = bool(cancel_event is not None and cancel_event.is_set())
                return [], {
                    "status": "cancelled" if cancelled else "error",
                    "message": "场景检索已停止；已写入的记忆仍保留" if cancelled
                               else str(exc)[:400],
                }
            finally:
                with self._cancel_lock:
                    if self._active_reasoning_request_id == request_id:
                        self._active_reasoning_request_id = None

        details = {}
        for round_result in result.get("retrieval_rounds") or []:
            for detail in round_result.get("candidate_details") or []:
                if isinstance(detail, dict) and detail.get("id"):
                    details[str(detail["id"])] = detail
        output = []
        for candidate in result.get("candidates") or []:
            mid = str(candidate.get("id") or "")
            if not mid or mid == current_moment_id:
                continue
            moment = archive.get(mid) or {}
            detail = details.get(mid) or {}
            mm = moment.get("multimodal") or {}
            output.append({
                "id": mid, "layer": 7, "matched_layers": [7],
                **memory_facts(moment),
                "score": detail.get("text_relevance"),
                "match_reasons": detail.get("reasons") or [],
                "entity_matches": detail.get("entity_matches") or [],
                "multimodal_link": detail.get("multimodal_link") or {},
                "timestamp": moment.get("timestamp") or candidate.get("time", ""),
                "scene": moment.get("scene") or candidate.get("scene", ""),
                "action": moment.get("user_action") or candidate.get("action", ""),
                "people": moment.get("people") or candidate.get("people") or [],
                "location": moment.get("location") or candidate.get("location", ""),
                "activity": moment.get("activity") or candidate.get("activity", ""),
                "objects": moment.get("objects") or candidate.get("objects") or [],
                "environments": moment.get("environments") or [],
                "summary": moment.get("summary", ""),
                "source_evidence": moment.get("source_evidence") or "",
                "source_moment_ids": moment.get("source_moment_ids") or [],
                "source_link_coverage": moment.get("source_link_coverage", "unknown"),
                "source_provenance": moment.get("source_provenance") or {},
                "modalities": {
                    "visual": bool(mm.get("visual") or mm.get("visual_frames")),
                    "transcript_text": bool(moment.get("source_evidence")),
                },
            })
            if len(output) >= 5:
                break
        # Face-linked history is resolved against stored person crops and explicit
        # user corrections. It supplements lexical retrieval; it never invents a
        # relationship from appearance alone.
        identity_link_status = {"status": "not_checked", "linked_people": 0, "added_events": 0}
        identity_rows = []
        identity_started = time.perf_counter()
        try:
            identity_index = EntityRecordIndex(
                {"moments": archive}, asset_root=self.mem.memory_dir,
                identity_correction_path=self.mem.memory_dir / "person_identity_corrections.json",
                compute_missing_face_embeddings=False)
            identity_link_status = dict(identity_index.identity_link_status or {})
            if identity_link_status.get("pending_embeddings"):
                _schedule_identity_warmup(
                    archive, self.mem.memory_dir,
                    self.mem.memory_dir / "person_identity_corrections.json")
                identity_link_status["enrichment"] = "background_queued"
            for record in identity_index.records:
                members = set(record.get("member_ids") or [])
                current_member = next((member for member in record.get("identity_members") or []
                                       if member.get("id") in members
                                       and any(event.get("moment_id") == current_moment_id
                                               and event.get("source_provenance", {}).get("source_segment_id")
                                               == member.get("source_segment_id")
                                               for event in record.get("events") or [])), None)
                if record.get("kind") != "person" or current_member is None:
                    continue
                prior_events = [event for event in record.get("events") or []
                                if event.get("moment_id") != current_moment_id]
                for event in prior_events:
                    mid = str(event.get("moment_id") or "")
                    moment = archive.get(mid) or {}
                    if not mid or not moment or mid in {row.get("id") for row in output}:
                        continue
                    links = [link for link in record.get("identity_matches") or []
                             if link.get("left") in members and link.get("right") in members]
                    best_score = max((float(link.get("score") or 0) for link in links), default=None)
                    reason = {
                        "type": "face_identity_match" if links else "identity_correction",
                        "entity": record.get("name") or "人物",
                        "evidence": "复用已存人物裁剪帧的人脸特征跨片段关联" if links
                                    else "使用已保存的人物身份纠错关联",
                    }
                    if best_score is not None:
                        reason["similarity"] = round(best_score, 5)
                    row = {
                        "id": mid, "layer": 7, "matched_layers": [7],
                        **memory_facts(moment),
                        "score": 1.0, "match_reasons": [reason],
                        "entity_matches": [{"entity": record.get("name"),
                                            "aliases": record.get("aliases") or [],
                                            "identity_members": record.get("identity_members") or []}],
                        "timestamp": moment.get("timestamp", ""),
                        "scene": moment.get("scene", ""),
                        "action": moment.get("user_action", ""),
                        "people": moment.get("people") or [],
                        "location": moment.get("location", ""),
                        "activity": moment.get("activity", ""),
                        "objects": moment.get("objects") or [],
                        "environments": moment.get("environments") or [],
                        "summary": moment.get("summary", ""),
                        "source_evidence": moment.get("source_evidence") or "",
                        "source_moment_ids": moment.get("source_moment_ids") or [],
                        "source_link_coverage": moment.get("source_link_coverage", "unknown"),
                        "source_provenance": moment.get("source_provenance") or {},
                        "multimodal_link": {"basis": reason["type"],
                                             "identity_members": record.get("identity_members") or []},
                        "modalities": {"visual": True,
                                       "transcript_text": bool(moment.get("source_evidence"))},
                    }
                    identity_rows.append(row)
                    if len(identity_rows) >= 5:
                        break
                if identity_rows:
                    identity_link_status["linked_people"] = int(identity_link_status.get("linked_people", 0) or 0) + 1
                if len(identity_rows) >= 5:
                    break
        except Exception as exc:
            identity_link_status = {"status": "error", "message": str(exc)[:240]}
        finally:
            wrapper_timings["cached_identity_index_ms"] = round(
                (time.perf_counter() - identity_started) * 1000)
        role_started = time.perf_counter()
        current_roles = _spoken_role_evidence(current)
        role_rows = []
        if current_roles:
            for mid, prior in archive.items():
                mid = str(mid)
                if mid == current_moment_id or not isinstance(prior, dict) or prior.get("merged_into"):
                    continue
                shared_roles = sorted(set(current_roles) & set(_spoken_role_evidence(prior)))
                if not shared_roles or mid in {row.get("id") for row in output + identity_rows}:
                    continue
                role = shared_roles[0]
                old_role = _spoken_role_evidence(prior)[role]
                current_role = current_roles[role]
                reason = {
                    "type": "shared_spoken_role", "entity": role,
                    "confidence": "medium",
                    "evidence": "兩段記憶都由語音直接標記了相同關係稱呼；這表示角色關聯，不等同於人臉已確認同一人",
                    "evidence_refs": [current_role.get("evidence_ref"), old_role.get("evidence_ref")],
                }
                role_rows.append({
                    "id": mid, "layer": 7, "matched_layers": [7],
                    **memory_facts(prior), "score": 0.9, "match_reasons": [reason],
                    "entity_matches": [{"entity": role, "match_type": "shared_spoken_role"}],
                    "timestamp": prior.get("timestamp", ""),
                    "scene": prior.get("scene", ""), "action": prior.get("user_action", ""),
                    "people": prior.get("people") or [], "location": prior.get("location", ""),
                    "activity": prior.get("activity", ""), "objects": prior.get("objects") or [],
                    "environments": prior.get("environments") or [], "summary": prior.get("summary", ""),
                    "source_evidence": prior.get("source_evidence") or "",
                    "source_moment_ids": prior.get("source_moment_ids") or [],
                    "source_link_coverage": prior.get("source_link_coverage", "unknown"),
                    "source_provenance": prior.get("source_provenance") or {},
                    "multimodal_link": {"basis": "shared_spoken_role",
                                         "role": role,
                                         "face_identity_confirmed": False},
                    "modalities": {"visual": bool((prior.get("multimodal") or {}).get("visual_frames")),
                                   "transcript_text": bool(prior.get("source_evidence"))},
                })
        wrapper_timings["speech_role_linking_ms"] = round(
            (time.perf_counter() - role_started) * 1000)
        if identity_rows or role_rows:
            identity_rows.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
            role_rows.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
            combined_rows = identity_rows + role_rows
            seen_ids = set()
            combined_rows = [row for row in combined_rows
                             if row.get("id") and not (row["id"] in seen_ids or seen_ids.add(row["id"]))]
            output = combined_rows + output
            output = output[:5]
            identity_link_status["added_events"] = len(identity_rows)
            identity_link_status["role_related_events"] = len(role_rows)

        for summary in result.get("summary_layers") or []:
            output.append({
                "id": f"L{summary.get('layer')}_summary",
                "layer": summary.get("layer"),
                "matched_layers": [summary.get("layer")],
                "score": summary.get("score"),
                "summary": summary.get("summary", ""),
                "source_moment_ids": summary.get("source_moment_ids") or [],
                "source_link_coverage": summary.get("source_link_coverage", "unknown"),
            })
        # Return only the stored portrait/object references for retrieved moments.
        # The current moment is excluded above, so these are historical profiles.
        for row in output:
            prior = archive.get(str(row.get("id") or ""))
            if not isinstance(prior, dict):
                continue
            row["person_observations"] = prior.get("person_observations") or []
            row["object_observations"] = prior.get("object_observations") or []
            media = prior.get("multimodal") or {}
            profile_media = {}
            for key in ("entity_visuals", "person_visuals", "object_visuals"):
                group = media.get(key) or {}
                items = []
                for item in group.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    entry = dict(item)
                    refs = []
                    for ref in item.get("frame_refs") or []:
                        if not isinstance(ref, dict):
                            continue
                        image = dict(ref)
                        if image.get("asset") and not image.get("url"):
                            image["url"] = "/api/memory_asset?p=" + quote(str(image["asset"]), safe="/")
                        refs.append(image)
                    entry["frame_refs"] = refs
                    items.append(entry)
                if items:
                    profile_media[key] = {"items": items}
            row["multimodal"] = profile_media

        searches = ((result.get("search_plan") or {}).get("searches") or [])
        metadata = {
            "status": cache_status["status"], "cache": cache_status,
            "elapsed_ms": result.get("elapsed_ms"),
            "scene_plan_elapsed_ms": result.get("scene_plan_elapsed_ms", 0),
            "scene_plan_cache_used": bool(result.get("scene_plan_cache_used")),
            "scene_plan_method": result.get("scene_plan_method") or "unknown",
            "stage_timings_ms": result.get("stage_timings_ms") or {},
            "model_calls": int(result.get("model_calls", 0) or 0),
            "scene_understanding": result.get("scene_understanding") or {},
            "searches": searches[:5],
            "candidate_count": len(result.get("candidates") or []) + len(identity_rows) + len(role_rows),
            "identity_link": identity_link_status,
            "summary_count": len(result.get("summary_layers") or []),
            "scope": "当前片段沿用已完成的画面/语音感知，不重复分析原视频；仅检索并回传可回查的历史记忆，不生成提醒、诊断或行动建议",
            "wrapper_stage_timings_ms": wrapper_timings,
            "wrapper_elapsed_ms": round((time.perf_counter() - wrapper_started) * 1000),
        }
        try:
            log_path = self.mem.memory_dir / "scene_retrieval_log.jsonl"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "dataset_id": dataset_id, "source_moment_id": current_moment_id,
                    "cache": cache_status, "elapsed_ms": result.get("elapsed_ms"),
                    "wrapper_stage_timings_ms": wrapper_timings,
                    "wrapper_elapsed_ms": metadata["wrapper_elapsed_ms"],
                    "identity_link": identity_link_status,
                    "searches": searches[:5],
                    "candidate_ids": [row.get("id") for row in output],
                }, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            pass
        return output, metadata
