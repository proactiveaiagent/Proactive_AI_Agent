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
from typing import Any

import requests

from multimodal_memory import compare_multimodal


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
                limit: int = 12) -> list[dict]:
        archive = self.memory.memory["layer7"]["moments"]
        found: dict[str, tuple[float, dict]] = {}
        for name in entities:
            for dimension in ("people", "locations", "activity_events", "objects", "environments"):
                for tag, ids in self.memory.memory["layer7"].get(dimension, {}).items():
                    if str(tag).lower() == name.lower():
                        for mid in ids[-12:]:
                            moment = archive.get(mid)
                            if moment and not moment.get("stale") and mid not in seen:
                                found[mid] = (max(found.get(mid, (0, None))[0], 1.0), moment)
        for query in searches:
            for rank, moment in enumerate(self.memory.query(query, top_k=limit)):
                mid = moment.get("id")
                if mid and mid not in seen:
                    lexical_score = 3.0 - rank / max(limit, 1)
                    found[mid] = (max(found.get(mid, (0, None))[0], lexical_score), moment)
        ranked = sorted(found.values(), key=lambda item: (
            item[0] + (0.3 if item[1].get("highlighted") else 0),
            item[1].get("timestamp", "")), reverse=True)
        return [moment for _, moment in ranked[:limit]]

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
                if mid in seen or moment.get("stale") or mid == anchor.get("id"):
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
        }

    def reason(self, situation: str, context: dict | None = None) -> dict:
        situation = str(situation or "").strip()[:1200]
        if not situation:
            raise ValueError("请提供当前事件")
        context = context if isinstance(context, dict) else {}
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
        seen: set[str] = set()
        first = self._recall(searches, entities, seen, limit=9)
        media_first, links_first = self._media_neighbors(first, {m["id"] for m in first})
        first.extend(media_first)
        for row in first:
            seen.add(row["id"])
        evidence = first[:]
        rounds = [{"queries": searches, "candidate_ids": [m["id"] for m in first],
                   "media_links": links_first}]
        summaries = {
            layer: str(self.memory.memory.get(layer, {}).get("summary") or "")[:700]
            for layer in ("layer4", "layer5")
        }
        profile = self.memory.memory.get("layer6", {}).get("profile") or {}

        def final_prompt(items: list[dict], allow_followup: bool) -> str:
            return (
                "You are the memory module's decision-support reasoner. "
                "Recommend a concrete NEXT ACTION in Chinese, grounded in supplied memory. "
                "Do not claim an action has been executed. Distinguish memory evidence from general "
                "world knowledge; cite only provided memory IDs. Do not invent a personal medical "
                "history or country. If someone fainted, recommend immediate safety checks and "
                "contacting local emergency services when needed; do not diagnose. If a country "
                "changes, state any tipping custom as general knowledge unless explicitly in memory. "
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

        decision = self._infer(final_prompt(evidence, True))
        followups = self._strings(decision.get("followup_searches"), 3)
        if followups:
            more = self._recall(followups, entities, seen, limit=8)
            if more:
                media_more, links_more = self._media_neighbors(more, seen | {m["id"] for m in more})
                more.extend(media_more)
                evidence.extend(more)
                rounds.append({"queries": followups,
                               "candidate_ids": [m["id"] for m in more],
                               "media_links": links_more})
                decision = self._infer(final_prompt(evidence, False))

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
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }
