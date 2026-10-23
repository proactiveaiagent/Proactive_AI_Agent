"""Search and retrieve methods for the seven-layer memory store."""

import json
import os
from typing import Dict, List

from memory_search import (
    _build_memory_search_index, _rank_memory_docs, _rank_memory_search_index,
    _retrieval_tokens, tokenize,
)


class MemoryRetrievalMixin:
    """Layered lexical/semantic retrieval shared by PersonMemory."""

    def _query_tokens(self, *texts: str) -> list:
        """查询词 → tokens：中英混合分词，不实时调用 LLM（对齐 §9.3 硬约束）。

        统一分词策略：tokenize 同时处理中文（单字）与英文（单词）——
        中文查询词按单字命中 moment 中文原文，英文查询词按单词命中写入时
        预翻译的 normalized。检索同步链路零 LLM 调用。
        """
        tokens = []
        for text in texts:
            if text:
                tokens += tokenize(str(text))
        return tokens

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search the Layer 7 archive with stopword filtering and BM25 ranking."""
        return [row["moment"] for row in self.query_ranked(query_text, top_k)]

    def query_ranked(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Return active Layer 7 memories with their field-aware BM25 scores."""
        if top_k <= 0:
            return []
        revision = self.memory.get("metadata", {}).get("semantic_index_revision", 0)
        cache = self._lexical_search_cache
        if cache is None or cache.get("revision") != revision:
            archive = self.memory["layer7"]["moments"]
            # Keep one best source-backed record per video segment. Legacy runs
            # can contain duplicate writes; retrieval should not count them as
            # separate history or let them crowd out distinct episodes.
            best_by_source = {}
            for moment_id, moment in archive.items():
                if (not isinstance(moment, dict) or moment.get("merged_into")
                        or moment.get("stale")):
                    continue
                provenance = moment.get("source_provenance") or {}
                source_time = moment.get("source_time") or {}
                multimodal = moment.get("multimodal") or {}
                source_id = str(provenance.get("source_segment_id")
                                or source_time.get("source_segment_id")
                                or multimodal.get("source_id") or "")
                if not source_id:
                    continue
                quality = (
                    240 * len(moment.get("temporal_facts") or [])
                    + 160 * len(moment.get("person_observations") or [])
                    + 100 * len(moment.get("object_observations") or [])
                    + 80 * bool(moment.get("source_evidence"))
                    + sum(len(str(moment.get(key) or "")) for key in
                          ("scene", "user_action", "interaction_details", "activity", "location"))
                )
                rank = (quality, str(moment.get("timestamp") or ""), str(moment_id))
                previous = best_by_source.get(source_id)
                if previous is None or rank > previous[0]:
                    best_by_source[source_id] = (rank, str(moment_id))
            canonical_ids = {value[1] for value in best_by_source.values()}
            documents = [
                self._retrieval_group_moment(moment)
                for moment_id, moment in archive.items()
                if (isinstance(moment, dict) and not moment.get("merged_into")
                    and not moment.get("stale")
                    and (not (
                        (moment.get("source_provenance") or {}).get("source_segment_id")
                        or (moment.get("source_time") or {}).get("source_segment_id")
                        or (moment.get("multimodal") or {}).get("source_id")
                    ) or str(moment_id) in canonical_ids))
            ]
            cache = _build_memory_search_index(documents)
            cache["revision"] = revision
            cache["all_documents"] = documents
            self._lexical_search_cache = cache
        documents = cache["all_documents"]
        lexical = _rank_memory_search_index(query_text, cache)
        if self._semantic_index_enabled:
            try:
                from semantic_retrieval import PersistentSemanticIndex, loaded_encoder
                encoder = loaded_encoder()
                if encoder is not None:
                    archive = self.memory.get("layer7", {}).get("moments", {})
                    index = PersistentSemanticIndex(self.memory_dir)
                    revision = self.memory.get("metadata", {}).get("semantic_index_revision", 0)
                    if index.ready(archive, encoder, revision=revision):
                        depth = min(len(documents), max(50, top_k * 10))
                        semantic = index.search(query_text, encoder, top_k=depth)
                        lexical_ids = [str(moment.get("id") or "") for _, moment in lexical[:depth]]
                        fused = {}
                        for ranked_ids in (lexical_ids, [row["moment_id"] for row in semantic]):
                            for rank, moment_id in enumerate(ranked_ids, 1):
                                if moment_id:
                                    fused[moment_id] = fused.get(moment_id, 0.0) + 1.0 / (60 + rank)
                        by_id = {str(moment.get("id") or ""): moment for moment in documents}
                        ordered = sorted(fused, key=lambda mid: (-fused[mid], mid))
                        return [
                            {"score": round(fused[mid], 6), "moment": by_id[mid],
                             "retrieval_method": "bm25_gte_rrf"}
                            for mid in ordered[:top_k] if mid in by_id
                        ]
            except Exception as exc:
                # A stale/missing/corrupt sidecar must not block ordinary retrieval.
                print(f"[semantic-index] query fallback to BM25: {exc}", flush=True)
        return [
            {"score": round(score, 6), "moment": moment,
             "retrieval_method": "bm25"}
            for score, moment in lexical[:top_k]
        ]

    def query_summary_layers(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Search L4-L6 summaries and return explicit source lineage."""
        if top_k <= 0 or not _retrieval_tokens(str(query_text or "")[:1200]):
            return []
        query_tokens = set(_retrieval_tokens(str(query_text or "")[:1200]))
        documents, originals = [], {}

        def active_profile_facts(node, path=""):
            if isinstance(node, dict) and "value" in node:
                if node.get("stale") or not node.get("value"):
                    return []
                return [f"{path}: {node['value']}" if path else str(node["value"])]
            if isinstance(node, dict):
                result = []
                for key, value in node.items():
                    result.extend(active_profile_facts(value, f"{path}.{key}" if path else key))
                return result
            if isinstance(node, list):
                return [fact for value in node for fact in active_profile_facts(value, path)]
            return []

        for layer_no in (4, 5, 6):
            layer = self.memory.get(f"layer{layer_no}", {})
            if not isinstance(layer, dict):
                continue
            document = {key: value for key, value in layer.items() if key not in {
                "source_moment_ids", "source_link_coverage", "source_moment_count",
                "last_updated", "profile",
            }}
            profile_facts = active_profile_facts(layer.get("profile") or {}) if layer_no == 6 else []
            matched_facts = [fact for fact in profile_facts
                             if query_tokens & set(_retrieval_tokens(fact))]
            if matched_facts:
                document["profile_facts"] = matched_facts
            document["id"] = f"layer{layer_no}_summary"
            documents.append(document)
            originals[document["id"]] = (layer_no, layer, matched_facts)

        rows = []
        for score, document in _rank_memory_docs(query_text, documents)[:top_k]:
            layer_no, layer, matched_facts = originals[document["id"]]
            summary = str(layer.get("summary") or "").strip()
            if layer_no == 6 and matched_facts:
                summary = (summary + "\n" if summary else "") + "相关稳定事实：" + "；".join(matched_facts[:12])
            if not summary:
                keys = ("current_tasks", "life_trajectory") if layer_no == 4 else ("key_events", "long_term_patterns")
                summary = "；".join(str(value) for key in keys
                                    for value in (layer.get(key) if isinstance(layer.get(key), list)
                                                  else [layer.get(key)]) if value)
            if not summary:
                continue
            source_ids = list(layer.get("source_moment_ids") or [])
            rows.append({
                "layer": layer_no, "score": round(score, 4), "summary": summary[:1600],
                "source_moment_ids": source_ids[:40],
                "source_moment_count": max(int(layer.get("source_moment_count") or 0), len(source_ids)),
                "source_link_coverage": layer.get("source_link_coverage", "unknown"),
            })
        return rows

    def query_hierarchical(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Rank unique recent/archive episodes together; summaries supplement source records."""
        query_text = str(query_text or "")[:1200]
        if top_k <= 0 or not _retrieval_tokens(query_text):
            return []

        memberships = {}
        window_only = {}
        for layer_no in (1, 2, 3):
            for moment in self._sorted_layer_moments(f"layer{layer_no}"):
                moment_id = str(moment.get("id") or "")
                if not moment_id or moment.get("merged_into"):
                    continue
                memberships.setdefault(moment_id, []).append(layer_no)
                window_only.setdefault(moment_id, moment)

        archive = self.memory.get("layer7", {}).get("moments", {})
        for moment_id, moment in archive.items() if isinstance(archive, dict) else []:
            if not isinstance(moment, dict) or moment.get("merged_into"):
                continue
            memberships.setdefault(str(moment_id), []).append(7)
            window_only.setdefault(str(moment_id), self._retrieval_group_moment(moment))

        # query_ranked preserves the optional BM25 + semantic-index fusion.
        archive_rows = self.query_ranked(query_text, top_k=max(50, top_k * 10))
        ranked_moments = []
        ranked_ids = set()
        for row in archive_rows:
            moment = row.get("moment") or {}
            moment_id = str(moment.get("id") or "")
            if not moment_id or moment_id in ranked_ids:
                continue
            ranked_ids.add(moment_id)
            ranked_moments.append((float(row.get("score") or 0), moment))

        # Preserve any valid rolling-window row missing from the canonical archive.
        orphan_docs = [moment for mid, moment in window_only.items()
                       if mid not in archive and mid not in ranked_ids]
        ranked_moments.extend(_rank_memory_docs(query_text, orphan_docs))

        results = []
        for score, moment in ranked_moments:
            moment_id = str(moment.get("id") or "")
            layers = sorted(set(memberships.get(moment_id, [])))
            display_layer = next((number for number in (1, 2, 3) if number in layers), 7)
            results.append({"layer": display_layer, "matched_layers": layers,
                            "score": round(float(score), 6), "moment": moment})
            if len(results) >= top_k:
                break

        if len(results) < top_k:
            for row in self.query_summary_layers(query_text, top_k=top_k - len(results)):
                moment = {"id": f"layer{row['layer']}_summary", "summary": row["summary"],
                          "source_moment_ids": row["source_moment_ids"],
                          "source_moment_count": row["source_moment_count"],
                          "source_link_coverage": row["source_link_coverage"]}
                results.append({"layer": row["layer"], "matched_layers": [row["layer"]],
                                "score": row["score"], "moment": moment})
        return results[:top_k]

    def retrieve(self, people: List[str] = None, location: str = None,
                 activity: str = None, layer: int = None, top_k: int = 5,
                 query_text: str = None, include_summaries: bool = True,
                 exclude_ids: set = None,
                 include_source_evidence: bool = False) -> str:
        """按标签和当前查询检索 L7，并返回完整的 Part1–4 事实。"""
        if top_k <= 0:
            return "No relevant memory found."
        parts = []
        if include_summaries:
            for lnum, lkey in [(4, "layer4"), (5, "layer5"), (6, "layer6")]:
                if layer and layer != lnum:
                    continue
                summary = self.memory.get(lkey, {}).get("summary")
                if summary:
                    parts.append(f"[Layer {lnum}] {summary}")
                if lnum == 4:
                    plans = self.memory.get(lkey, {}).get("stated_plans") or []
                    for plan in plans[:8] if isinstance(plans, list) else []:
                        if not isinstance(plan, dict):
                            continue
                        source = str(plan.get("source_moment_id") or "")
                        when = plan.get("source_time") or {}
                        scope = plan.get("time_scope") or ""
                        description = "; ".join(str(value) for value in (
                            plan.get("actor"), plan.get("value"), scope,
                            when.get("day_label"), when.get("time_of_day"), source)
                            if value)
                        if description:
                            parts.append(f"[Layer 4 明确说出的计划（尚不代表已完成）] {description}")

        query_text = str(query_text or "")[:1200]
        query_tokens = set(self._query_tokens(
            query_text, *(str(x) for x in (people or [])),
            str(location or ""), str(activity or "")))
        excluded = set(exclude_ids or ())
        scored = []
        for moment_id, stored in self.memory["layer7"]["moments"].items():
            if (moment_id in excluded or stored.get("stale")
                    or stored.get("merged_into")):
                continue
            moment = self._retrieval_group_moment(stored)
            searchable = json.dumps({key: value for key, value in moment.items()
                                     if key not in {"source_provenance", "multimodal", "needs", "solutions"}},
                                    ensure_ascii=False).lower()
            score = sum(1 for token in query_tokens if token in searchable)
            if score:
                scored.append((score, bool(moment.get("highlighted")
                                           or moment.get("_group_highlighted")),
                               moment.get("timestamp", ""), moment))

        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        for _, _, _, moment in scored[:top_k]:
            def values(items):
                if not isinstance(items, list):
                    return str(items or "")
                rendered = []
                for item in items:
                    if isinstance(item, dict):
                        value = (item.get("need") or item.get("solution")
                                 or item.get("value") or item.get("action"))
                    else:
                        value = item
                    if value:
                        rendered.append(str(value))
                return "；".join(rendered)

            feedback = moment.get("feedback") or {}
            corrections = feedback.get("corrections") or {}
            feedback_text = "；".join(
                f"{key}={value}" for key, value in corrections.items()
            ) if isinstance(corrections, dict) else ""
            if feedback.get("confirmed"):
                feedback_text = (feedback_text + "；" if feedback_text else "") + "用户已确认"
            if moment.get("highlighted") and not feedback_text:
                feedback_text = "已标记为重点/重复记忆"

            line = f"[Layer7/{moment.get('id','')}] {moment.get('timestamp','')[:19]}"
            source_time = moment.get("source_time") or {}
            if source_time:
                source_clock = " · ".join(str(value) for value in (
                    source_time.get("day_label"), source_time.get("time_of_day")) if value)
                if source_clock:
                    line += f" | 来源时间: {source_clock}"
            source_id = (moment.get("source_provenance") or {}).get("source_segment_id")
            if source_id:
                line += f" | 来源片段: {str(source_id)[:128]}"
            fields = [
                ("场景", moment.get("scene")),
                ("动作", moment.get("user_action")),
                ("人物交互与事件顺序", moment.get("interaction_details")),
                ("人物", values(moment.get("people"))),
                ("地点", moment.get("location")),
                ("活动", moment.get("activity")),
                ("物品", json.dumps(moment.get("objects") or [], ensure_ascii=False)),
                ("物品观察", json.dumps(moment.get("object_observations") or [], ensure_ascii=False)),
                ("物品状态与位置", json.dumps(moment.get("object_states") or [], ensure_ascii=False)),
                ("环境", json.dumps(moment.get("environments") or [], ensure_ascii=False)),
                ("人物观察", json.dumps(moment.get("person_observations") or [], ensure_ascii=False)),
                ("时间事实", json.dumps(moment.get("temporal_facts") or [], ensure_ascii=False)),
                ("来源时间", json.dumps(moment.get("source_time") or {}, ensure_ascii=False)),
                ("反馈修正", feedback_text),
            ]
            if include_source_evidence and moment.get("source_evidence"):
                evidence = str(moment["source_evidence"]).replace(chr(10), " ").strip()
                line += f" | Transcript evidence: {evidence[:1200]}"
            for label, value in fields:
                if value:
                    limit = 3200 if label == "时间事实" else 1800 if label == "物品观察" else 1600 if label == "人物观察" else 500 if label in {"人物交互与事件顺序", "来源时间"} else 300 if label in {"环境", "物品"} else 180
                    line += f" | {label}: {str(value).replace(chr(10), ' ').strip()[:limit]}"
            parts.append(line)
            for source in moment.get("_merged_sources_records", []):
                source_line = f"  └ 合并来源 [{source.get('id','')}] {str(source.get('timestamp',''))[:19]}"
                for label, key in (("Scene", "scene"), ("Action", "user_action"),
                                   ("People", "people"), ("Location", "location"),
                                   ("Activity", "activity")):
                    value = source.get(key)
                    if value:
                        source_line += f" | {label}: {str(value).replace(chr(10), ' ').strip()[:180]}"
                media = source.get("multimodal") or {}
                if media:
                    source_line += " | retained media: " + ("visual " if (media.get("visual") or {}).get("asset") else "") + ("audio" if (media.get("audio") or {}).get("asset") else "")
                parts.append(source_line)

        return chr(10).join(parts) if parts else "No relevant memory found."
