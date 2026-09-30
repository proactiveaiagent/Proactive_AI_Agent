"""LLM-assisted retrieval and evidence-grounded next-action suggestions.

Fast lexical lookup remains the first-stage candidate search. The LLM plans
which related memories to fetch, can request up to two follow-up retrieval rounds, and
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

MAX_REASONING_ROUNDS = 3


class MemoryReasoningCancelled(RuntimeError):
    """Raised when the UI asks the current reasoning pass to stop."""


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise MemoryReasoningCancelled("主动建议已停止")


def _json_object(text: str) -> dict:
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型返回的不是 JSON 对象")
    return value


_MEDIA_STOP_WORDS = {
    "what", "are", "the", "when", "where", "why", "how", "should", "would", "could",
    "does", "did", "do", "has", "have", "had", "someone", "anyone", "user", "current",
    "currently", "memory", "memories", "stored", "record", "records", "relevant", "related",
    "personal", "preference", "preferences", "history", "context", "available", "there", "this",
    "that", "with", "from", "into", "about", "after", "before", "first", "steps", "take",
    "action", "actions", "help", "nearby", "local", "custom", "customs", "general", "known",
    "person", "people", "around", "just", "then", "says", "said", "want", "wants", "wanted",
    "need", "needs", "needed", "helpful", "useful", "next", "remember", "remembered",
    "remembering", "specific", "detail", "details", "provide", "provided", "situation", "whether",
    "their", "these", "those", "must", "can", "will", "been", "being", "were", "was", "are",
    "is", "and", "for", "not", "none", "check", "checking", "state", "clearly", "without",
    "she", "her", "he", "him", "they", "them", "environment", "environments",
    "restaurant", "restaurants", "dining", "meal", "meals", "country", "countries",
    "location", "located", "china", "united", "states", "moved", "moving", "finish",
    "finishing", "forget", "forgot", "remind", "reminder", "call", "stor", "preferenc",
    "let", "lets", "one", "ones", "get", "got", "go", "going", "went", "come", "came",
    "use", "uses", "using", "request", "requests", "requested",
    "reply", "replies", "replied", "repli",
    "yeah", "yes", "okay", "well", "like", "just", "really", "right", "thing", "things",
    "someone", "anybody", "everybody", "people", "person", "charge", "charges",
    "service", "services", "bill", "bills",
}


def _semantic_tokens(value: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z][a-z'-]{2,}|[\u4e00-\u9fff]{2,}", str(value or "").casefold()):
        if re.fullmatch(r"[a-z][a-z'-]{2,}", token):
            token = token.replace("’", "'").strip("'-")
            token = re.sub(r"'(?:s|re|ve|ll|d|m|t)$", "", token).replace("'", "")
            if token.endswith("ing") and len(token) > 6:
                token = token[:-3]
                if len(token) > 3 and token[-1] == token[-2]:
                    token = token[:-1]
            else:
                for suffix in ("ed", "es", "s"):
                    if len(token) > len(suffix) + 3 and token.endswith(suffix):
                        token = token[:-len(suffix)]
                        break
        if len(token) > 2 and token not in _MEDIA_STOP_WORDS:
            tokens.add(token)
    return tokens


class MemoryReasoner:
    def __init__(self, memory, api_url: str = "http://127.0.0.1:8000", timeout: int = 120):
        self.memory = memory
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _infer(self, prompt: str, request_id: str = "", cancel_event=None) -> dict:
        _check_cancelled(cancel_event)
        response = requests.post(
            self.api_url + "/consolidate",
            json={"prompt": prompt, "request_id": request_id},
            timeout=self.timeout,
        )
        if response.status_code == 409 and cancel_event is not None and cancel_event.is_set():
            raise MemoryReasoningCancelled("主动建议已停止")
        response.raise_for_status()
        data = response.json()
        if data.get("cancelled"):
            raise MemoryReasoningCancelled("主动建议已停止")
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return _json_object(str(data.get("analysis") or ""))

    @staticmethod
    def _strings(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip()[:180] for item in value[:limit]
                if isinstance(item, str) and item.strip()]

    @classmethod
    def _memory_searches(cls, value: Any, limit: int) -> tuple[list[str], list[str]]:
        """Keep personal-memory lookups separate from generic world knowledge."""
        queries = cls._strings(value, limit)
        generic_knowledge = re.compile(
            r"\b(?:first[- ]aid|emergency response|safety)\s+"
            r"(?:protocol|procedure|guideline|steps?)\b"
            r"|\b(?:local|location[- ]specific|restaurant)\s+"
            r"(?:tipping\s+)?customs?\b"
            r"|\b(?:general|standard)\s+(?:advice|guidance|procedure|protocol)\b"
            r"|\b(?:service|surcharge)\s+charge\b.{0,50}\b(?:this|the|current)\s+(?:restaurant|venue|place)\b",
            re.IGNORECASE,
        )
        accepted, filtered = [], []
        for query in queries:
            (filtered if generic_knowledge.search(query) else accepted).append(query)
        return accepted, filtered

    @staticmethod
    def _enforce_personal_care_boundary(situation: str, decision: dict) -> bool:
        care_context = re.search(
            r"\b(?:bathroom|shower|bath|toilet|undress|personal care|hygiene)\b"
            r"|洗澡|淋浴|浴室|卫生间|厕所|换衣|更衣|个人护理",
            situation, re.IGNORECASE,
        )
        if not care_context:
            return False
        explicit_permission = re.search(
            r"(?:\b(?:asked|requested|explicitly asked)\b.{0,80}"
            r"\b(?:enter|accompany|help|guide)\b"
            r"|\bplease\b.{0,60}\b(?:enter|accompany|help|guide)\b"
            r"|明确(?:要求|请求|同意).{0,20}(?:陪|进入|协助|搀扶)"
            r"|(?:要求|请求).{0,20}(?:陪同|进入|协助|搀扶))",
            situation, re.IGNORECASE,
        )
        action = str(decision.get("action") or "")
        proposes_entry = re.search(
            r"\b(?:accompany|escort|guide|lead|join)\b.{0,50}"
            r"\b(?:into|inside|bathroom|shower|bath|toilet)\b"
            r"|\b(?:into|inside)\b.{0,50}\b(?:accompany|escort|guide|join)\b"
            r"|陪同.{0,12}(?:浴室|洗澡|浴|卫生间|厕所|进入|进去)"
            r"|(?:浴室|卫生间|厕所).{0,12}陪同"
            r"|(?:进入|进去|进到).{0,12}(?:浴室|卫生间|厕所)"
            r"|(?:搀扶|触碰|脱衣)",
            action, re.IGNORECASE,
        )
        if not proposes_entry or explicit_permission:
            return False
        decision["action"] = (
            "先询问对方希望获得哪种帮助，或是否需要陪伴或保持隐私；"
            "在明确同意前留在私人空间外等待，不触碰或引导。"
        )
        decision["reason"] = (
            "当前情境涉及个人护理，但没有明确授权他人进入私人空间或提供身体协助。"
            "应先确认对方的偏好，并在明确同意前留在空间外。"
        )
        decision["uncertainty"] = "是否需要陪伴或身体协助尚未明确，需要先询问。"
        decision["risk"] = "low"
        return True

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

    def _load_modal_evidence(self, moments: list[dict], relevance_queries: list[str] | None = None,
                             entity_terms: list[str] | None = None,
                             active_moment_id: str = "") -> tuple[list[dict], list[dict], dict]:
        # Gate assets on the current situation/decision, never on broad planner
        # search strings that can make unrelated clips look relevant.
        root = Path(self.memory.memory_dir).resolve()
        # The first query is the user's actual situation. Planner follow-ups and
        # generated decision questions are intentionally excluded from this gate.
        query_tokens = _semantic_tokens((relevance_queries or [""])[0])
        query_tokens -= _semantic_tokens(" ".join(entity_terms or []))
        people_index = self.memory.memory.get("layer7", {}).get("people", {})
        if isinstance(people_index, dict):
            query_tokens -= _semantic_tokens(" ".join(map(str, people_index.keys())))
        images, audio_texts = [], []
        image_paths, audio_paths = set(), set()
        image_ids, audio_ids, relevant_ids, filtered_ids, relevance_rows = [], [], [], [], []
        audio_rejected_ids = []
        for moment in moments:
            mid = str(moment.get("id") or "")
            mm = moment.get("multimodal") or {}
            # Use the concise event description and structured evidence, but
            # not free-form interaction detail, which often repeats dialogue.
            description = " ".join(str(moment.get(key) or "") for key in (
                "scene", "activity", "location"))
            structured = " ".join(
                str(value) for key in ("objects", "environments", "needs", "solutions")
                for value in (moment.get(key) if isinstance(moment.get(key), list)
                              else [moment.get(key)])
                if value not in (None, "")
            )
            description += " " + structured
            transcript = str(moment.get("source_evidence") or "")
            description_hits = query_tokens & _semantic_tokens(description)
            transcript_hits = query_tokens & _semantic_tokens(transcript)
            # A topic hit in the structured event description is a useful anchor.
            # A transcript-only mention needs two distinct topic hits, so an
            # incidental word or joke cannot pull an unrelated clip into a decision.
            is_active = mid == active_moment_id
            is_relevant = is_active or bool(description_hits) or len(transcript_hits) >= 2
            has_assets = bool((mm.get("visual") or {}).get("asset") or
                              (mm.get("audio") or {}).get("asset"))
            if has_assets:
                relevance_rows.append({
                    "memory_id": mid,
                    "used": is_relevant,
                    "basis": "current_event" if is_active else (
                        "event_description" if description_hits else (
                            "transcript_topic_match" if len(transcript_hits) >= 2 else "no_topic_match"
                        )
                    ),
                    "description_terms": sorted(description_hits)[:8],
                    "transcript_terms": sorted(transcript_hits)[:8],
                    "source_video_id": str(mm.get("source_id") or ""),
                    "visual_at_s": (mm.get("visual") or {}).get("at_s"),
                    "audio_at_s": (mm.get("audio") or {}).get("at_s"),
                })
            if has_assets and not is_relevant:
                filtered_ids.append(mid)
                continue
            if has_assets:
                relevant_ids.append(mid)
            if len(images) < 4:
                ref = mm.get("visual") or {}
                asset = str(ref.get("asset") or "")
                if asset:
                    path = (root / asset).resolve()
                    if root in path.parents and path.is_file() and path not in image_paths:
                        payload = path.read_bytes()
                        if len(payload) <= 24_000:
                            image_paths.add(path)
                            images.append({
                                "id": mid, "name": path.name, "bytes": payload,
                                "mime": "image/webp", "source_video_id": str(mm.get("source_id") or ""),
                                "frame_at_s": ref.get("at_s"),
                                "clip_start_s": mm.get("start_s"), "clip_end_s": mm.get("end_s"),
                            })
                            image_ids.append(mid)
                            row = next((item for item in relevance_rows
                                        if item.get("memory_id") == mid), None)
                            if row is not None:
                                row["visual_status"] = "used"
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
                                    # Reject a clearly unrelated ASR result instead of treating
                                    # generic transcription hallucinations as event evidence.
                                    asr_tokens = _semantic_tokens(text)
                                    anchor_tokens = (_semantic_tokens(transcript) |
                                                     _semantic_tokens(description) |
                                                     query_tokens)
                                    row = next((item for item in relevance_rows
                                                if item.get("memory_id") == mid), None)
                                    if not (asr_tokens & anchor_tokens):
                                        audio_rejected_ids.append(mid)
                                        if row is not None:
                                            row["audio_status"] = (
                                                "rejected_topic_mismatch" if asr_tokens else "rejected_no_content_terms")
                                            row["asr_terms"] = sorted(asr_tokens)[:8]
                                    else:
                                        audio_texts.append({
                                            "id": mid, "text": text[:360],
                                            "source_video_id": str(mm.get("source_id") or ""),
                                            "audio_at_s": ref.get("at_s"),
                                            "duration_s": ref.get("duration_s"),
                                        })
                                        audio_ids.append(mid)
                                        if row is not None:
                                            row["audio_status"] = "used_topic_match"
                            except Exception:
                                row = next((item for item in relevance_rows
                                            if item.get("memory_id") == mid), None)
                                if row is not None:
                                    row["audio_status"] = "transcription_failed"
                                # Audio is optional; visual/text reasoning must remain available.
                                continue
        return images, audio_texts, {
            "visual_memory_ids": image_ids,
            "audio_memory_ids": audio_ids,
            "relevant_media_memory_ids": relevant_ids,
            "filtered_irrelevant_media_count": len(filtered_ids),
            "audio_transcription_rejected_ids": audio_rejected_ids,
            "audio_transcription_rejected_count": len(audio_rejected_ids),
            "media_relevance": relevance_rows,
        }

    def _infer_with_media(self, prompt: str, moments: list[dict],
                          relevance_queries: list[str] | None = None,
                          entity_terms: list[str] | None = None,
                          active_moment_id: str = "", request_id: str = "",
                          cancel_event=None) -> tuple[dict, dict]:
        _check_cancelled(cancel_event)
        images, audio_texts, source = self._load_modal_evidence(
            moments, relevance_queries, entity_terms, active_moment_id)

        # Preserve exact Whisper text with its source ID; it is direct audio
        # evidence and does not need a second model to restate it.
        observations = [
            {
                "memory_id": str(item.get("id") or ""),
                "modality": "audio",
                "observation": "Whisper transcription: “" + str(item.get("text") or "")[:240] + "”",
                "source_video_id": item.get("source_video_id") or "",
                "audio_at_s": item.get("audio_at_s"),
                "duration_s": item.get("duration_s"),
            }
            for item in audio_texts if item.get("id") and item.get("text")
        ]

        if images:
            mapping = [{"image": index + 1, "memory_id": item["id"]}
                       for index, item in enumerate(images)]
            perception_prompt = (
                "Inspect only the attached image pixels; do not use or infer any event context. "
                "Return ONLY JSON in this schema: {\"observations\":[{\"image\":1,\"observation\":\"one concise Simplified Chinese description of visible facts\"}]}. "
                "Create one row per image. Describe only the sampled instant, not the whole clip. "
                "Report visible people count, objects, visible actions, and environment. Do not infer names, intent, consent, emotion, diagnosis, or time of day from lighting. "
                "Do not claim that an object or activity is absent from the whole event because it is not visible in one frame. "
                "Do not explain, answer a question, or recommend an action. If image content is unclear, say so briefly. "
                "Image-to-source mapping: " + json.dumps(mapping, ensure_ascii=False)
            )
            response = requests.post(
                self.api_url + "/analyze_video",
                files=[("images", (item["name"], item["bytes"], item["mime"]))
                       for item in images],
                data={"prompt": perception_prompt, "request_id": request_id},
                timeout=max(self.timeout, 240),
            )
            if response.status_code == 409 and cancel_event is not None and cancel_event.is_set():
                raise MemoryReasoningCancelled("主动建议已停止")
            response.raise_for_status()
            payload = response.json()
            if payload.get("cancelled"):
                raise MemoryReasoningCancelled("主动建议已停止")
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            visual_text = str(payload.get("analysis") or "").strip()
            source["visual_observation_output"] = visual_text[:600]
            valid_visual_ids = set(source.get("visual_memory_ids") or [])
            visual_result = {}
            try:
                visual_result = _json_object(visual_text)
            except (ValueError, TypeError):
                visual_result = {}
            visual_rows = visual_result.get("observations")
            if isinstance(visual_rows, dict):
                visual_rows = [visual_rows]
            if isinstance(visual_rows, str):
                visual_rows = [visual_rows]
            if isinstance(visual_rows, list):
                for index, row in enumerate(visual_rows[:6]):
                    if isinstance(row, str):
                        ids = ([images[index]["id"]] if index < len(images)
                               else ([images[0]["id"]] if len(images) == 1 else []))
                        text = row.strip()
                    elif isinstance(row, dict):
                        image_index = row.get("image")
                        if isinstance(image_index, int) and 1 <= image_index <= len(images):
                            ids = [images[image_index - 1]["id"]]
                        else:
                            ids = row.get("memory_ids")
                            if not isinstance(ids, list):
                                ids = [row.get("memory_id")] if row.get("memory_id") else []
                        ids = [str(mid) for mid in ids if str(mid) in valid_visual_ids]
                        text = str(row.get("observation") or row.get("description") or "").strip()
                    else:
                        continue
                    if any(marker in text for marker in (
                        "建议", "下一步", "有用的一步", "可以引导", "应该", "应当", "最好", "提醒用户",
                        "当前情况：", "recommend", "next step", "should remind", "helpful next",
                    )):
                        continue
                    if ids and text and all(mid in valid_visual_ids for mid in ids):
                        image_by_id = {image["id"]: image for image in images}
                        sources = [{
                            "memory_id": mid,
                            "source_video_id": image_by_id[mid].get("source_video_id") or "",
                            "frame_at_s": image_by_id[mid].get("frame_at_s"),
                            "clip_start_s": image_by_id[mid].get("clip_start_s"),
                            "clip_end_s": image_by_id[mid].get("clip_end_s"),
                        } for mid in ids if mid in image_by_id]
                        observations.append({
                            **({"memory_id": ids[0]} if len(ids) == 1 else {"memory_ids": ids}),
                            "sources": sources,
                            "modality": "visual", "observation": text[:320],
                        })
            if not observations and visual_text and "observations" not in visual_result and len(images) == 1:
                # Only accept a plain caption for one frame. Advice-like prose is
                # rejected above; an unindexed multi-frame answer cannot be traced.
                blocked_markers = (
                    "建议", "下一步", "有用的一步", "可以引导", "应该", "应当", "最好", "提醒用户",
                    "当前情况：", "recommend", "next step", "should remind", "helpful next",
                )
                if not any(marker in visual_text for marker in blocked_markers):
                    image = images[0]
                    observations.append({
                        "memory_id": image["id"],
                        "sources": [{
                            "memory_id": image["id"],
                            "source_video_id": image.get("source_video_id") or "",
                            "frame_at_s": image.get("frame_at_s"),
                            "clip_start_s": image.get("clip_start_s"),
                            "clip_end_s": image.get("clip_end_s"),
                        }],
                        "modality": "visual", "observation": visual_text[:320],
                    })
                    source["visual_observation_status"] = "single_frame_caption_fallback"
            if observations:
                source["visual_observation_status"] = source.get(
                    "visual_observation_status") or "structured_or_parsed"
            else:
                source["visual_observation_status"] = "no_verifiable_observation_returned"
            if observations:
                prompt += (
                    "\n\nVisual/audio evidence extracted from the actual attached assets. "
                    "Use these source-linked observations with the event text and retrieved memory. "
                    "If a visual observation conflicts with event text, state the mismatch and lower confidence; "
                    "do not treat a still frame as proof of intent: " +
                    json.dumps(observations, ensure_ascii=False)
                )
            else:
                prompt += (
                    "\n\nThe relevant frame was sent to Qwen3-VL, but it returned no usable observation. "
                    "Do not claim a specific visual fact; ground any suggestion in event text and retrieved memories only."
                )
            model = "Qwen3-VL visual reading + Qwen reasoning"
        else:
            if audio_texts:
                prompt += "\n\nWhisper transcribed stored audio: " + json.dumps(audio_texts, ensure_ascii=False)
            model = "Whisper transcription + Qwen reasoning" if audio_texts else "text-only Qwen fallback"

        _check_cancelled(cancel_event)
        result = self._infer(prompt, request_id=request_id, cancel_event=cancel_event)
        # This list is populated from the VLM observation pass and exact ASR
        # output, then checked against the source IDs actually sent to inference.
        result["multimodal_observations"] = observations
        source["source_observations"] = observations
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
                translated = self._infer(
                    translation_prompt, request_id=request_id, cancel_event=cancel_event)
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
            except MemoryReasoningCancelled:
                raise
            except Exception:
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
        request_id = str(context.pop("reasoning_request_id", "") or "")
        cancel_event = context.pop("cancellation_event", None)
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
            "Search only for facts that could change the next action. Do not invent speculative problems, medical concerns, preferences, or routines that the current event did not raise; use one direct query for the expressed intent and at most a few clearly relevant prior preferences. "
            "Return ONLY JSON: {\"situation\":str,\"entities\":[str],"
            "\"searches\":[str],\"decision_question\":str,"
            "\"current_country\":str|null}. Set current_country only when the present country is explicit or clear; otherwise null. "
            "Return at most 3 searches, and use an empty list when no prior fact could change the next action. Every search must be anchored to a person, event, action, place, symptom, or stated preference explicitly present in the situation. Never add hypothetical allergies, plumbing, maintenance, temperature, schedules, or other concerns not raised by the situation. Do not recommend an action yet.\n"
            f"Current situation: {situation}\nCurrent context: "
            + json.dumps(context, ensure_ascii=False),
            request_id=request_id, cancel_event=cancel_event,
        )
        proposed_searches = self._strings(planner.get("searches"), 4)
        searches, filtered_searches = self._memory_searches(proposed_searches, 3)
        planner = dict(planner)
        planner["proposed_searches"] = proposed_searches
        planner["searches"] = searches or [situation]
        planner["filtered_general_knowledge_searches"] = filtered_searches
        planner["used_situation_fallback"] = not bool(searches)
        searches = planner["searches"]
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
                "Inspect the supplied current scene, retrieved episodes, and source-linked visual/audio descriptions extracted from actual media. "
                "Before suggesting an action, connect a concrete supplied observation or spoken intent to the relevant memory or stated need. "
                "When multimodal_observations contains a visual row, the reason MUST mention one concrete visual fact from that row and explain whether it supports, contradicts, or cannot verify the current event. Include its source moment ID; for a mismatch, explicitly say the sampled frame cannot verify progress and do not claim the activity did not happen before or after it. Never attribute an action or identity to the frame unless its observation says so. "
                "A frame is a timestamped sample, not proof of what happened across the whole clip. If no audio observation is supplied, do not claim direct audio support; stored speech text is a prior transcript, not a new verified audio perception. "
                "Do not treat keyword overlap as proof of intent. Clearly separate direct observation, remembered fact, and general knowledge. "
                "Choose one specific, low-risk next step that responds to this episode; do not answer only with generic advice when a clear request or offer is present. "
                "If another person offers help, use the user's explicit intent and the offer together. Ask only about the unresolved part (for example, whether the offered company is wanted now); do not repeat a question the user already answered or invent requested supplies. "
                "Hard consent boundary for personal care: never tell someone to physically guide, touch, undress, or enter a private space with another person unless that person explicitly asked for that specific help. An offer such as ‘Let's go’ is not permission to touch, guide, or enter a bathroom. Treat it as a verbal offer; if accompaniment is unresolved, suggest asking whether the person wants company or privacy, then wait outside the private space. Do not suggest accompanying them inside. "
                "Include the supplied multimodal_observations records {memory_id, modality, observation} unchanged in the JSON; "
                "do not invent additional sensory observations. "
                "Do not claim multimodal evidence was used when no media was supplied. "
                + ("This is a proactive scan of the latest observed event. Inspect visual, audio, and text evidence together. "
                   "Check whether the event reveals an urgent safety need, an unresolved expressed intent, a consent-sensitive request, "
                   "or a location change that makes an optional reminder useful. Compare current and retrieved locations when available. "
                   "Suggest an action only when evidence supports it; if the country or preference is unclear, say so rather than guess. "
                   "Otherwise say no reminder is needed. " if proactive else "")
                + "Recommend a concrete NEXT ACTION grounded in supplied memory. "
                "Do not claim an action has been executed. Distinguish memory evidence from general "
                "world knowledge; cite only provided memory IDs. Do not invent a personal medical "
                "history or country. If someone fainted, first check responsiveness and normal breathing; "
                "contact local emergency services for unresponsiveness, abnormal/no breathing, serious "
                "injury, or other life-threatening signs, and follow the dispatcher; do not diagnose. "
                "For personal-care requests, ask what help is wanted and respect privacy; do not assume "
                "consent to physically guide someone. The current situation is valid evidence for where the "
                "user is now; do not require the location or general custom to have been stored in memory. "
                "If the current situation explicitly says the user is in a United States restaurant, the country is known from the current situation even if it is absent from memory. Give an optional reminder to check whether the bill includes a service charge; explain that tipping is customary, not compulsory, and the amount is the user's choice. Do not withhold this general reminder solely because no personal preference is stored. Never say the user 'needs' or 'must' tip, and do not prescribe a percentage. In other places, use only general knowledge and avoid overclaiming. "
                "If you are unsure, advise checking the bill/local custom; "
                "never abstain only because personal tipping preference is absent. Label customs as general "
                "knowledge, not remembered preference. Never suppress a useful action solely "
                "because personal memory is absent. "
                "Return ONLY JSON with keys action (str), reason (str), evidence_ids (list of IDs), "
                "multimodal_observations (list of {memory_id, modality, observation}), "
                "uncertainty (str), risk (low|medium|high), world_assumption (str), "
                "followup_searches (list of English queries). "
                "Searches are queries against the user's personal memory only: request user-specific past facts, "
                "health history, explicit preferences, or prior events. Do not search personal memory for "
                "general first-aid procedures, safety protocols, local customs, or general advice; use general "
                "knowledge for those. "
                + ("Use followup_searches only if another memory lookup could change your decision. "
                   if allow_followup else "Set followup_searches to []. ")
                + "Current situation: " + situation
                + "\nContext: " + json.dumps(context, ensure_ascii=False)
                + "\nDecision question: " + str(planner.get("decision_question") or "")
                + "\nRecent/older summaries: " + json.dumps(summaries, ensure_ascii=False)
                + "\nProfile: " + json.dumps(profile, ensure_ascii=False)[:1200]
                + "\nRetrieved memories: "
                + json.dumps([self._compact(m) for m in prompt_items(items)], ensure_ascii=False)
            )

        def prompt_items(items: list[dict]) -> list[dict]:
            # Preserve the active event, then favor newly retrieved evidence so a
            # later reasoning round is not crowded out by the first lexical hits.
            if current_moment:
                active_id = str(current_moment.get("id") or "")
                remaining = [row for row in items if str(row.get("id") or "") != active_id]
                return [current_moment, *remaining[-19:]]
            return items[-20:]

        # Search strings remain available to text retrieval. Media relevance is
        # gated only by the current situation and the planner's decision question.
        modality_queries = [situation]
        decision, multimodal_inference = self._infer_with_media(
            final_prompt(evidence, True), prompt_items(evidence), modality_queries,
            entities, current_moment_id, request_id, cancel_event,
        )
        for round_number in range(2, MAX_REASONING_ROUNDS + 1):
            _check_cancelled(cancel_event)
            proposed_followups = self._strings(decision.get("followup_searches"), 3)
            followups, filtered_followups = self._memory_searches(proposed_followups, 3)
            if filtered_followups:
                rounds[-1]["filtered_general_knowledge_followups"] = filtered_followups
            if not followups:
                break
            more_details: dict[str, dict] = {}
            more = self._recall(followups, entities, seen, limit=8,
                                priority_details=more_details)
            if not more:
                break
            media_more, links_more = self._media_neighbors(
                more, seen | {m["id"] for m in more})
            more.extend(media_more)
            evidence.extend(more)
            seen.update(str(row.get("id") or "") for row in more)
            rounds.append({"queries": followups,
                           "candidate_ids": [m["id"] for m in more],
                           "media_links": links_more,
                           "candidate_details": self._priority_rows(
                               more, more_details, links_more)})
            selected = prompt_items(evidence)
            decision, multimodal_inference = self._infer_with_media(
                final_prompt(evidence, round_number < MAX_REASONING_ROUNDS), selected,
                modality_queries, entities, current_moment_id, request_id, cancel_event,
            )

        # Keep the model's recommendation intact. Safety and evidence limits are
        # instructions to the reasoner, not keyword-based scenario answer rewrites.
        model_decision = dict(decision)
        policy_adjustments = []
        if self._enforce_personal_care_boundary(situation, decision):
            policy_adjustments.append(
                "个人护理同意边界：模型建议涉及未经明确同意的陪同或进入私人空间，"
                "已改为先询问偏好并在空间外等待。"
            )
        # Validate sensory claims against the exact media passed to the model.
        allowed_observations = {
            "visual": set(multimodal_inference.get("visual_memory_ids") or []),
            "audio": set(multimodal_inference.get("audio_memory_ids") or []),
        }
        observations = []
        media_rows = {
            str(row.get("memory_id") or ""): row
            for row in (multimodal_inference.get("media_relevance") or [])
            if row.get("memory_id")
        }
        # Only show observations emitted by the perception/ASR pass itself.
        # The decision model may cite these, but cannot manufacture sensory facts.
        raw_observations = multimodal_inference.get("source_observations")
        if isinstance(raw_observations, list):
            for row in raw_observations[:8]:
                if not isinstance(row, dict):
                    continue
                modality = str(row.get("modality") or "").strip().lower()
                if modality in ("image", "frame", "画面", "视觉"):
                    modality = "visual"
                elif modality in ("sound", "speech", "transcript", "语音", "声音"):
                    modality = "audio"
                mids = row.get("memory_ids")
                if not isinstance(mids, list):
                    mids = [row.get("memory_id")] if row.get("memory_id") else []
                mids = [str(mid) for mid in mids]
                text = str(row.get("observation") or "").strip()[:320]
                allowed = allowed_observations.get(modality, set())
                if mids and all(mid in allowed for mid in mids) and text:
                    sources = [{
                        "memory_id": mid,
                        "source_video_id": media_rows.get(mid, {}).get("source_video_id", ""),
                        "frame_at_s": media_rows.get(mid, {}).get("visual_at_s") if modality == "visual" else None,
                        "audio_at_s": media_rows.get(mid, {}).get("audio_at_s") if modality == "audio" else None,
                    } for mid in mids]
                    observations.append({
                        **({"memory_id": mids[0]} if len(mids) == 1 else {"memory_ids": mids}),
                        "sources": sources,
                        "modality": modality, "observation": text,
                    })
        decision["multimodal_observations"] = observations
        model_decision["multimodal_observations"] = observations
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
                "multimodal_observations": observations,
            },
            "model_recommendation": {
                "action": str(model_decision.get("action") or ""),
                "reason": str(model_decision.get("reason") or ""),
                "evidence_ids": [mid for mid in self._strings(
                    model_decision.get("evidence_ids"), 12) if mid in valid_ids],
                "multimodal_observations": observations,
            },
            "policy_adjustments": policy_adjustments,
            "multimodal_inference": multimodal_inference,
            "proactive_scan": proactive,
            "source_moment_id": current_moment_id or None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }
