"""Write-after-scene retrieval orchestration for the interactive memory runner."""

import json
import time
import uuid

from entity_records import memory_facts


class PostWriteRetrievalMixin:
    """Retrieve source-linked historical memories after a new moment is stored."""

    def _retrieve_after_write(self, current_moment_id, fields, dataset_id="", progress=None):
        """Understand the just-observed scene and retrieve source-linked history only."""
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
            "retrieval_policy_version": "scene-memory-retrieval-v6-structured-fields",
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
                self.mem.memory, self.dataset_id, "scene_retrieval_after_write_v4",
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
            "candidate_count": len(result.get("candidates") or []),
            "summary_count": len(result.get("summary_layers") or []),
            "scope": "当前片段沿用已完成的画面/语音感知，不重复分析原视频；仅检索并回传可回查的历史记忆，不生成提醒、诊断或行动建议",
        }
        try:
            log_path = self.mem.memory_dir / "scene_retrieval_log.jsonl"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "dataset_id": dataset_id, "source_moment_id": current_moment_id,
                    "cache": cache_status, "elapsed_ms": result.get("elapsed_ms"),
                    "searches": searches[:5],
                    "candidate_ids": [row.get("id") for row in output],
                }, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            pass
        return output, metadata
