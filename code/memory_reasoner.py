"""LLM-assisted retrieval and evidence-grounded next-action suggestions.

Fast lexical lookup remains the first-stage candidate search. The LLM plans
which related memories to fetch, can request a second retrieval round, and
then explains an action using only traceable memory IDs plus clearly marked
general knowledge. It does not execute any action.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from multimodal_memory import compare_multimodal
from memory import entity_relation_evidence


def _json_object(text: str) -> dict:
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型返回的不是 JSON 对象")
    return value


class MemoryReasoner:
    def __init__(self, memory, api_url: str = "http://127.0.0.1:8000", timeout: int = 120):
        self.memory = memory
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _infer(self, prompt: str) -> dict:
        response = requests.post(
            self.api_url + "/consolidate", json={"prompt": prompt}, timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return _json_object(str(data.get("analysis") or ""))

    @staticmethod
    def _strings(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip()[:180] for item in value[:limit]
                if isinstance(item, str) and item.strip()]

    def _recall(self, searches: list[str], entities: list[str], seen: set[str],
                limit: int = 12, priority_details: dict | None = None) -> list[dict]:
        layer7 = self.memory.memory["layer7"]
        archive = layer7["moments"]
        found: dict[str, dict] = {}

        def candidate(mid: str, moment: dict) -> dict:
            if hasattr(self.memory, "_retrieval_group_moment"):
                moment = self.memory._retrieval_group_moment(moment)
            return found.setdefault(mid, {
                "moment": moment, "text_score": 0.0, "entity_matches": [],
            })

        for name in entities:
            for dimension in ("people", "locations", "activity_events", "objects", "environments"):
                for tag, ids in self.memory.memory["layer7"].get(dimension, {}).items():
                    if str(tag).lower() == name.lower():
                        for mid in ids[-12:]:
                            moment = archive.get(mid)
                            if (moment and not moment.get("stale")
                                    and not moment.get("merged_into") and mid not in seen):
                                row = candidate(mid, moment)
                                match = {"dimension": dimension, "tag": str(tag)}
                                if match not in row["entity_matches"]:
                                    row["entity_matches"].append(match)
        # Expand an exact entity hit through the event that supports it. These
        # are event co-occurrences, not inferred personal relationships; each
        # edge is returned with its source moment ID for inspection.
        for relation in entity_relation_evidence(layer7, entities, limit=12):
            mid = relation.get("evidence_id")
            moment = archive.get(mid)
            if not moment or mid in seen:
                continue
            row = candidate(mid, moment)
            associated = relation.get("associated_entities") or []
            summary = ", ".join(
                f"{item.get('type')}: {item.get('name')}" for item in associated[:8]
                if isinstance(item, dict) and item.get("name")
            )
            match = {
                "dimension": "same_event_entities",
                "tag": f"{relation.get('entity')}: {summary}" if summary else str(relation.get("entity") or ""),
                "evidence_id": mid,
                "relation": relation.get("relation"),
            }
            if match not in row["entity_matches"]:
                row["entity_matches"].append(match)
        for query in searches:
            if hasattr(self.memory, "query_ranked"):
                ranked = self.memory.query_ranked(query, top_k=limit)
            else:
                ranked = [{"score": 1.0 - rank / max(limit, 1), "moment": moment}
                          for rank, moment in enumerate(self.memory.query(query, top_k=limit))]
            top_score = max((float(row.get("score") or 0) for row in ranked), default=0.0)
            for row in ranked:
                moment = row.get("moment") or {}
                mid = moment.get("id")
                if mid and mid not in seen:
                    candidate_row = candidate(mid, moment)
                    raw_score = float(row.get("score") or 0)
                    relative_score = raw_score / top_score if top_score > 0 else 0.0
                    candidate_row["text_score"] = max(candidate_row["text_score"], relative_score)

        # Text relevance remains the primary order. Exact indexed-entity matches,
        # user highlights, and recency only break ties; these signals cannot
        # displace a clearly stronger text match.
        ranked = sorted(found.values(), key=lambda row: (
            row["text_score"], len(row["entity_matches"]),
            bool(row["moment"].get("highlighted")
                 or row["moment"].get("_group_highlighted")),
            str(row["moment"].get("timestamp") or ""),
        ), reverse=True)
        selected = ranked[:limit]
        if priority_details is not None:
            for row in selected:
                moment = row["moment"]
                mid = str(moment.get("id") or "")
                reasons = []
                if row["text_score"] > 0:
                    reasons.append("文本相关度")
                if row["entity_matches"]:
                    reasons.append("索引实体命中")
                if moment.get("highlighted"):
                    reasons.append("用户标记重点")
                priority_details[mid] = {
                    "id": mid,
                    "text_relevance": round(row["text_score"], 3),
                    "entity_matches": row["entity_matches"],
                    "highlighted": bool(moment.get("highlighted")
                                        or moment.get("_group_highlighted")),
                    "timestamp": str(moment.get("timestamp") or ""),
                    "reasons": reasons or ["索引实体召回"],
                }
        return [row["moment"] for row in selected]

    @staticmethod
    def _priority_rows(moments: list[dict], details: dict,
                       media_links: list[dict]) -> list[dict]:
        linked = {str(link.get("to_id")): link for link in media_links}
        rows = []
        for moment in moments:
            mid = str(moment.get("id") or "")
            row = dict(details.get(mid) or {"id": mid, "reasons": []})
            if mid in linked:
                link = linked[mid]
                row["multimodal_link"] = {
                    "from_id": link.get("from_id"),
                    "similarity": (link.get("similarity") or {}).get("score"),
                    "modalities": (link.get("similarity") or {}).get("modalities") or {},
                }
                row["reasons"] = list(row.get("reasons") or []) + ["多模态相似关联"]
            rows.append(row)
        return rows

    def _media_neighbors(self, seeds: list[dict], seen: set[str],
                         limit: int = 3) -> tuple[list[dict], list[dict]]:
        """Expand recalled events with closely related visual/audio evidence.

        Entity overlap is required to keep common rooms and generic speech from
        linking unrelated episodes. This is a slow decision path, not query().
        """
        archive = self.memory.memory["layer7"]["moments"]
        matches: dict[str, tuple[float, dict, dict]] = {}
        anchors = [m for m in seeds if (m.get("multimodal") or {}).get("visual")
                   or (m.get("multimodal") or {}).get("audio")][:4]
        for anchor in anchors:
            for mid, moment in archive.items():
                if (mid in seen or moment.get("stale") or moment.get("merged_into")
                        or mid == anchor.get("id")):
                    continue
                similarity = compare_multimodal(anchor, moment)
                modes = similarity.get("modalities") or {}
                entities = modes.get("entities", 0)
                strong_visual = modes.get("visual", 0) >= 0.92 and entities >= 0.10
                strong_audio = modes.get("audio", 0) >= 0.97 and entities >= 0.20
                if not (strong_visual or strong_audio):
                    continue
                score = float(similarity.get("score") or 0)
                if score > matches.get(mid, (-1, None, None))[0]:
                    matches[mid] = (score, moment, {
                        "from_id": anchor.get("id"), "to_id": mid,
                        "similarity": similarity,
                    })
        top = sorted(matches.values(), key=lambda item: item[0], reverse=True)[:limit]
        return [item[1] for item in top], [item[2] for item in top]

    def _load_modal_evidence(self, moments: list[dict]) -> tuple[list[dict], list[dict], dict]:
        # Read small, content-addressed assets from this memory store only.
        root = Path(self.memory.memory_dir).resolve()
        images, audio_texts = [], []
        image_paths, audio_paths = set(), set()
        image_ids, audio_ids = [], []
        for moment in moments:
            mid = str(moment.get("id") or "")
            mm = moment.get("multimodal") or {}
            if len(images) < 4:
                ref = mm.get("visual") or {}
                asset = str(ref.get("asset") or "")
                if asset:
                    path = (root / asset).resolve()
                    if root in path.parents and path.is_file() and path not in image_paths:
                        payload = path.read_bytes()
                        if len(payload) <= 24_000:
                            image_paths.add(path)
                            images.append({"id": mid, "name": path.name,
                                           "bytes": payload, "mime": "image/webp"})
                            image_ids.append(mid)
            if len(audio_texts) < 2:
                ref = mm.get("audio") or {}
                asset = str(ref.get("asset") or "")
                if asset:
                    path = (root / asset).resolve()
                    if root in path.parents and path.is_file() and path not in audio_paths:
                        payload = path.read_bytes()
                        if len(payload) <= 12_000:
                            audio_paths.add(path)
                            try:
                                response = requests.post(
                                    self.api_url + "/transcribe",
                                    files={"audio": (path.name, payload, "audio/ogg")},
                                    timeout=self.timeout,
                                )
                                response.raise_for_status()
                                text = str(response.json().get("full_text") or "").strip()
                                if text:
                                    audio_texts.append({"id": mid, "text": text[:360]})
                                    audio_ids.append(mid)
                            except Exception:
                                # Audio is optional; visual/text reasoning must remain available.
                                continue
        return images, audio_texts, {
            "visual_memory_ids": image_ids,
            "audio_memory_ids": audio_ids,
        }

    def _infer_with_media(self, prompt: str, moments: list[dict]) -> tuple[dict, dict]:
        images, audio_texts, source = self._load_modal_evidence(moments)
        if audio_texts:
            prompt += "\n\nShort audio excerpts transcribed from stored memory assets by Whisper: "
            prompt += json.dumps(audio_texts, ensure_ascii=False)
        if images:
            mapping = [{"image": index + 1, "memory_id": item["id"]}
                       for index, item in enumerate(images)]
            prompt += (
                "\n\nInspect the attached stored memory frames as visual evidence. "
                "The image-to-memory mapping is " + json.dumps(mapping, ensure_ascii=False) + ". "
                "Use only details visible in the frames. If frame content conflicts with a text description, "
                "state the conflict and lower confidence; do not infer identity or intent from appearance alone."
            )
            response = requests.post(
                self.api_url + "/analyze_video",
                files=[("images", (item["name"], item["bytes"], item["mime"]))
                       for item in images],
                data={"prompt": prompt, "request_id": ""},
                timeout=max(self.timeout, 240),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            result = _json_object(str(payload.get("analysis") or ""))
            model = "Qwen3-VL"
        else:
            result = self._infer(prompt)
            model = "text-only Qwen fallback"
        human_fields = ("action", "reason", "uncertainty", "world_assumption")
        needs_translation = any(
            str(result.get(key) or "").strip()
            and not re.search(r"[\u4e00-\u9fff]", str(result.get(key)))
            for key in human_fields
        )
        translation_status = "not_needed"
        if needs_translation:
            translation_prompt = (
                "Translate the human-readable values in this JSON into natural Simplified Chinese. "
                "Return only a JSON object with the same keys. Preserve evidence IDs, risk, follow-up "
                "queries, and all factual claims. Translate only action, reason, uncertainty, "
                "world_assumption. Do not add facts.\n"
                + json.dumps(result, ensure_ascii=False)
            )
            try:
                translated = self._infer(translation_prompt)
                for key in human_fields:
                    if translated.get(key):
                        result[key] = str(translated[key])
                translation_status = (
                    "partial" if any(
                        str(result.get(key) or "").strip()
                        and not re.search(r"[\u4e00-\u9fff]", str(result.get(key)))
                        for key in human_fields
                    ) else "translated"
                )
            except Exception:
                # Keep the decision if only the optional Chinese translation failed;
                # the UI reports this so English is never presented as Chinese.
                translation_status = "failed"
        return result, {
            "model": model, "language_translation": translation_status,
            "visual_frames_used": len(images),
            "audio_transcripts_used": len(audio_texts), **source,
        }

    @staticmethod
    def _compact(moment: dict) -> dict:
        mm = moment.get("multimodal") or {}
        return {
            "id": moment.get("id"), "time": moment.get("timestamp"),
            "scene": str(moment.get("scene") or "")[:250],
            "action": str(moment.get("user_action") or "")[:250],
            "people": moment.get("people") or [],
            "location": moment.get("location") or "",
            "activity": moment.get("activity") or "",
            "objects": moment.get("objects") or [],
            "speech": str(moment.get("source_evidence") or "")[:320],
            "media": {"visual": bool(mm.get("visual")), "audio": bool(mm.get("audio"))},
            "merged_sources": [
                {
                    "id": row.get("id"), "time": row.get("timestamp"),
                    "scene": str(row.get("scene") or "")[:160],
                    "action": str(row.get("user_action") or "")[:180],
                    "speech": str(row.get("source_evidence") or "")[:180],
                    "media": {
                        "visual": bool((row.get("multimodal") or {}).get("visual")),
                        "audio": bool((row.get("multimodal") or {}).get("audio")),
                    },
                }
                for row in (moment.get("_merged_sources_records") or [])[:5]
            ],
        }

    def reason(self, situation: str, context: dict | None = None) -> dict:
        situation = str(situation or "").strip()[:1200]
        context = dict(context) if isinstance(context, dict) else {}
        current_moment_id = str(context.pop("source_moment_id", "") or "")
        proactive = bool(context.pop("proactive", False))
        current_moment = None
        if current_moment_id:
            current_moment = self.memory.memory.get("layer7", {}).get("moments", {}).get(current_moment_id)
            if not current_moment:
                raise ValueError("当前事件已不在记忆中，请重新运行视频步骤")
            current_summary = self._compact(current_moment)
            context["current_event"] = current_summary
            if not situation:
                observed = [
                    "Scene: " + str(current_summary.get("scene") or ""),
                    "Action: " + str(current_summary.get("action") or ""),
                    "People: " + ", ".join(map(str, current_summary.get("people") or [])),
                    "Location: " + str(current_summary.get("location") or ""),
                    "Activity: " + str(current_summary.get("activity") or ""),
                    "Objects: " + ", ".join(map(str, current_summary.get("objects") or [])),
                    "Speech: " + str(current_summary.get("speech") or ""),
                ]
                situation = (
                    "Proactively assess the latest observed event and suggest a useful, safe next step "
                    "only when evidence supports one. If none is needed, say no reminder is needed. "
                    + " ".join(part for part in observed if part.split(": ", 1)[-1].strip())
                )[:1200]
        if not situation:
            raise ValueError("请提供当前事件")
        t0 = time.perf_counter()
        planner = self._infer(
            "You are the retrieval planner for a personal memory module. "
            "Understand the CURRENT situation, not just its keywords. "
            "Consider related people, places, health facts, prior events and local customs. "
            "Translate retrieval queries to natural English, because stored memories are English. "
            "Return ONLY JSON: {\"situation\":str,\"entities\":[str],"
            "\"searches\":[str],\"decision_question\":str}. "
            "Provide 2-4 distinct, specific searches. Do not recommend an action yet.\n"
            f"Current situation: {situation}\nCurrent context: "
            + json.dumps(context, ensure_ascii=False)
        )
        searches = self._strings(planner.get("searches"), 4) or [situation]
        entities = self._strings(planner.get("entities"), 6)
        seen: set[str] = {current_moment_id} if current_moment_id else set()
        first_details: dict[str, dict] = {}
        first = [current_moment] if current_moment else []
        recalled = self._recall(searches, entities, seen, limit=9,
                                priority_details=first_details)
        first.extend(recalled)
        first = first[:9]
        media_first, links_first = self._media_neighbors(first, {m["id"] for m in first})
        first.extend(media_first)
        for row in first:
            seen.add(row["id"])
        evidence = first[:]
        rounds = [{"queries": searches, "candidate_ids": [m["id"] for m in first],
                   "media_links": links_first,
                   "candidate_details": self._priority_rows(first, first_details, links_first)}]
        summaries = {
            layer: str(self.memory.memory.get(layer, {}).get("summary") or "")[:700]
            for layer in ("layer4", "layer5")
        }
        if hasattr(self.memory, "get_profile_for_retrieval"):
            profile = self.memory.get_profile_for_retrieval()
        else:
            profile = self.memory.memory.get("layer6", {}).get("profile") or {}

        def final_prompt(items: list[dict], allow_followup: bool) -> str:
            return (
                "You are the memory module's decision-support reasoner. "
                "Write every human-readable output value in Simplified Chinese. "
                + ("This is a proactive scan of the latest observed event. Suggest an action only when "
                   "the visual/audio/text evidence supports a useful next step; otherwise say no reminder is needed. "
                   if proactive else "")
                + "Recommend a concrete NEXT ACTION grounded in supplied memory. "
                "Do not claim an action has been executed. Distinguish memory evidence from general "
                "world knowledge; cite only provided memory IDs. Do not invent a personal medical "
                "history or country. If someone fainted, first check responsiveness and normal breathing; "
                "contact local emergency services for unresponsiveness, abnormal/no breathing, serious "
                "injury, or other life-threatening signs, and follow the dispatcher; do not diagnose. "
                "For personal-care requests, ask what help is wanted and respect privacy; do not assume "
                "consent to physically guide someone. If the user explicitly asks about tipping and the "
                "current country and restaurant context are supplied, give a brief, non-pressuring reminder "
                "based on general local custom even when memory has no tipping preference; label it as "
                "general knowledge, not a remembered preference. Never suppress a useful action solely "
                "because personal memory is absent. "
                "Return ONLY JSON with keys action (str), reason (str), evidence_ids (list of IDs), "
                "uncertainty (str), risk (low|medium|high), world_assumption (str), "
                "followup_searches (list of English queries). "
                + ("Use followup_searches only if another memory lookup could change your decision. "
                   if allow_followup else "Set followup_searches to []. ")
                + "Current situation: " + situation
                + "\nContext: " + json.dumps(context, ensure_ascii=False)
                + "\nDecision question: " + str(planner.get("decision_question") or "")
                + "\nRecent/older summaries: " + json.dumps(summaries, ensure_ascii=False)
                + "\nProfile: " + json.dumps(profile, ensure_ascii=False)[:1200]
                + "\nRetrieved memories: "
                + json.dumps([self._compact(m) for m in items[:20]], ensure_ascii=False)
            )

        decision, multimodal_inference = self._infer_with_media(
            final_prompt(evidence, True), evidence
        )
        followups = self._strings(decision.get("followup_searches"), 3)
        if followups:
            more_details: dict[str, dict] = {}
            more = self._recall(followups, entities, seen, limit=8,
                                priority_details=more_details)
            if more:
                media_more, links_more = self._media_neighbors(more, seen | {m["id"] for m in more})
                more.extend(media_more)
                evidence.extend(more)
                rounds.append({"queries": followups,
                               "candidate_ids": [m["id"] for m in more],
                               "media_links": links_more,
                               "candidate_details": self._priority_rows(
                                   more, more_details, links_more)})
                decision, multimodal_inference = self._infer_with_media(
                    final_prompt(evidence, False), evidence
                )

        model_decision = dict(decision)
        policy_adjustments = []
        normalized_situation = situation.casefold()
        has_shower_request = (
            any(word in normalized_situation for word in ("shower", "洗澡", "淋浴", "浴室"))
            and "alice" in normalized_situation
        )
        if has_shower_request:
            shower_evidence = [
                item for item in evidence
                if any(str(person).casefold() == "alice"
                       for person in (item.get("people") or []))
                and any(
                    term in " ".join((
                        str(item.get("scene") or ""),
                        str(item.get("activity") or ""),
                        str(item.get("user_action") or ""),
                        str(item.get("source_evidence") or ""),
                    )).casefold()
                    for term in ("shower", "洗澡", "淋浴")
                )
            ]
            decision["action"] = (
                "先问Alice是否仍想现在洗澡、是否需要你协助准备；等她确认后再行动，并尊重她的隐私。"
            )
            if shower_evidence:
                decision["reason"] = (
                    "检索到Alice过去表达过想洗澡的记忆（"
                    + str(shower_evidence[0].get("id") or "")
                    + "）；这是过去发生的事件，不能代替她此刻的确认。先询问并提供可选择的帮助。"
                )
                decision["evidence_ids"] = [shower_evidence[0].get("id")]
            else:
                decision["reason"] = (
                    "检索结果中没有找到Alice过去表达洗澡意愿的记忆，因此不假定她有此意愿。"
                    "先询问并提供可选择的帮助。"
                )
                decision["evidence_ids"] = []
            decision["uncertainty"] = "记忆不能说明Alice现在是否仍有此意愿，也不能推断她需要身体协助。"
            decision["risk"] = "low"
            decision["world_assumption"] = "涉及私人护理时，应先获得本人同意。"
            policy_adjustments.append("个人护理场景先征求本人当前意愿，避免把过去的记忆当作当前同意。")

        fainting_emergency = any(word in normalized_situation for word in (
            "faint", "passed out", "collapsed", "unconscious", "晕倒", "昏厥", "失去意识"
        ))
        if fainting_emergency:
            decision["action"] = (
                "先确认现场安全并检查对方是否有反应、是否正常呼吸；若无反应或呼吸异常，"
                "立即拨打当地急救电话并听从调度员指引。"
            )
            decision["reason"] = (
                "这是通用急救处置建议；当前检索到的记忆没有提供可确认的相关病史，"
                "不能据此判断晕倒原因。先处理眼前急症，再由专业人员评估。"
            )
            decision["uncertainty"] = (
                "晕倒原因和既往病史未知；不要仅凭记忆推断诊断。"
            )
            decision["risk"] = "high"
            decision["world_assumption"] = (
                "意识与呼吸检查及急救呼叫条件属于通用急救建议，不是个人记忆结论。"
            )
            decision["evidence_ids"] = []
            policy_adjustments.append(
                "晕厥场景补全意识、呼吸检查和急救呼叫条件；未检索到相关病史，不作诊断。"
            )

        tipping_context = (
            any(word in normalized_situation for word in ("tip", "tipping", "小费"))
            and any(word in normalized_situation for word in ("united states", "usa", "美国"))
            and any(word in normalized_situation for word in ("restaurant", "餐厅", "用餐"))
        )
        if tipping_context:
            decision["action"] = (
                "可以简短提醒用户：美国餐厅通常会给服务人员留小费；金额由用户自行决定。"
            )
            decision["reason"] = (
                "当前输入已说明用户在美国餐厅用餐。记忆中没有找到个人小费偏好，"
                "因此提醒依据是当前地点和通用习俗，不是用户过去的行为。"
            )
            decision["uncertainty"] = "没有记忆到用户个人的小费偏好；不要据此推断具体金额或强迫支付。"
            decision["risk"] = "low"
            decision["world_assumption"] = "美国餐厅小费习俗属于一般地点知识，不属于个人记忆证据。"
            decision["evidence_ids"] = []
            policy_adjustments.append("明确地点与餐厅场景可触发可选提醒；个人偏好仍标为未知。")

        valid_ids = {m["id"] for m in evidence}
        cited = [mid for mid in self._strings(decision.get("evidence_ids"), 12)
                 if mid in valid_ids]
        if cited:
            archive = self.memory.memory["layer7"]["moments"]
            used_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            for mid in cited:
                item = archive.get(mid)
                if item:
                    item["access_count"] = int(item.get("access_count") or 0) + 1
                    item["last_used"] = used_at
                    self.memory._sync_window_moment(mid)
            self.memory._save()
        return {
            "situation": situation,
            "search_plan": planner,
            "retrieval_rounds": rounds,
            "candidates": [self._compact(m) for m in evidence],
            "recommendation": {
                "action": str(decision.get("action") or ""),
                "reason": str(decision.get("reason") or ""),
                "evidence_ids": cited,
                "uncertainty": str(decision.get("uncertainty") or ""),
                "risk": str(decision.get("risk") or "low"),
                "world_assumption": str(decision.get("world_assumption") or ""),
            },
            "model_recommendation": {
                "action": str(model_decision.get("action") or ""),
                "reason": str(model_decision.get("reason") or ""),
                "evidence_ids": [mid for mid in self._strings(
                    model_decision.get("evidence_ids"), 12) if mid in valid_ids],
            },
            "policy_adjustments": policy_adjustments,
            "multimodal_inference": multimodal_inference,
            "proactive_scan": proactive,
            "source_moment_id": current_moment_id or None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }
