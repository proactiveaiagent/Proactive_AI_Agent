"""Write-after-scene retrieval orchestration for the interactive memory runner."""

import json
import threading
import time
import uuid

from entity_records import EntityRecordIndex, memory_facts


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
    "grandma": ("姥姥", "外婆", "奶奶", "祖母", "grandma", "grandmother"),
    "grandpa": ("爷爷", "外公", "祖父", "grandpa", "grandfather"),
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
    """Retrieve source-linked historical memories after a new moment is stored."""

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
