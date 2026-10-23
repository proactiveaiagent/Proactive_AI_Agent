"""Tokenization and field-aware BM25 helpers for the memory archive."""

import math
import os
from collections import Counter


# ---------------------------------------------------------------------------
# 检索分词（09-12 定稿：统一英文分词——非英文语种先翻译为英文再分词）
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """中英混合分词：英文/数字按单词边界切分，中文按单字切分。零 LLM 依赖。

    设计依据（设计说明书 §6.2，09-20 修订）：统一分词不再依赖「先翻译为英文」，
    而是让 tokenize 同时处理中英文——中文输出单字 token（命中中文原文），
    英文输出单词 token（命中写入时预翻译的 normalized）。检索同步链路零 LLM。
    """
    if not text:
        return []
    tokens, buf = [], ""
    for ch in text.lower():
        if ('a' <= ch <= 'z') or ('0' <= ch <= '9') or ch == "'":
            buf += ch
        elif '\u4e00' <= ch <= '\u9fff':   # 中文字符 → 单字 token
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    return tokens



_SEARCH_EN_STOPWORDS = frozenset({
    "a", "about", "after", "all", "also", "am", "an", "and", "are", "as", "at",
    "be", "because", "been", "being", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "for", "from", "had", "has",
    "have", "having", "he", "her", "here", "hers", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "more",
    "most", "my", "no", "nor", "of", "off", "on", "once", "only", "or", "our",
    "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "now", "currently"
})
_SEARCH_ZH_STOPWORDS = frozenset("的了在是有我你他她它和与及就也都很过把被给让而或又吗呢吧对从到这那一个些谁什么如何哪怎么现在刚才")

def _retrieval_tokens(text: str) -> list:
    """Return content-bearing terms for lexical retrieval; preserve Chinese character tokens."""
    return [token for token in tokenize(text)
            if token not in _SEARCH_EN_STOPWORDS
            and token not in _SEARCH_ZH_STOPWORDS
            and (len(token) > 1 or '\u4e00' <= token <= '\u9fff')]

def _memory_search_field_weights():
    return {
        "scene": 1.0,
        "user_action": 1.0,
        "interaction_details": 1.2,
        "person_observations": 1.1,
        "temporal_facts": 1.3,
        "object_observations": 1.2,
        "object_states": 1.4,
        "source_time": 0.8,
        "activity": 1.0,
        "people": 1.0,
        "location": 1.0,
        "objects": 1.0,
        "environments": 1.0,
        "activity_events": 1.0,
        "time_occasion": 0.7,
        "extra_notes": 0.6,
        "normalized": 0.5,
        "source_evidence": float(os.environ.get("MEMORY_SOURCE_EVIDENCE_WEIGHT", "0.25")),
        "multimodal_entities": 0.8,
        "merged_source_text": 0.6,
    }


_MEMORY_SEARCH_METADATA = {
    "id", "timestamp", "created_at", "updated_at", "last_used",
    "access_count", "stale", "decay_type", "decay_rate", "expires_at",
    "confidence", "highlighted", "feedback", "source_ids", "layer",
    "source_provenance", "canonicalization_history", "entity_graph",
    "needs", "solutions", "multimodal", "merged_into", "merged_sources",
    "merge_history", "merge_event", "source_refresh_history",
    "_merged_sources_records",
    "_merged_search_text", "_group_highlighted",
}


def _as_memory_search_text(value):
    if isinstance(value, dict):
        return " ".join(_as_memory_search_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_as_memory_search_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _build_memory_search_index(documents: list) -> dict:
    """Build field-aware BM25 statistics once; callers own cache invalidation."""
    active_documents = [doc for doc in documents
                        if isinstance(doc, dict) and not doc.get("stale")
                        and not doc.get("merged_into")]
    field_names = set(_memory_search_field_weights()) | {"other"}
    field_counts = []
    field_lengths = []
    document_frequency = Counter()

    for document in active_documents:
        values = {}
        for key, value in document.items():
            if key == "_merged_search_text":
                if value:
                    values["merged_source_text"] = _as_memory_search_text(value)
                continue
            if key in _MEMORY_SEARCH_METADATA:
                continue
            if key == "multimodal":
                entities = value.get("entities") if isinstance(value, dict) else None
                if entities:
                    values["multimodal_entities"] = _as_memory_search_text(entities)
                continue
            field = key if key in field_names else "other"
            values[field] = (values.get(field, "") + " " +
                             _as_memory_search_text(value)).strip()

        counts = {
            field: Counter(_retrieval_tokens(values.get(field, "").lower()))
            for field in field_names
        }
        lengths = {field: sum(counter.values()) for field, counter in counts.items()}
        field_counts.append(counts)
        field_lengths.append(lengths)
        for field, counter in counts.items():
            for term in counter:
                document_frequency[(field, term)] += 1

    averages = {}
    for field in field_names:
        nonempty = [lengths[field] for lengths in field_lengths if lengths[field] > 0]
        averages[field] = sum(nonempty) / len(nonempty) if nonempty else 1.0

    postings = {}
    for doc_index, counts in enumerate(field_counts):
        for field, counter in counts.items():
            for term, tf in counter.items():
                postings.setdefault(term, {}).setdefault(field, []).append((doc_index, tf))

    return {
        "documents": active_documents,
        "field_lengths": field_lengths,
        "document_frequency": document_frequency,
        "average_lengths": averages,
        "postings": postings,
        "live_count": len(active_documents),
    }


def _rank_memory_search_index(query_text: str, index_data: dict) -> list:
    """Score a prebuilt BM25 index while preserving the established tie order."""
    documents = index_data["documents"]
    if not documents:
        return []
    query_terms = list(dict.fromkeys(_retrieval_tokens(str(query_text or "")[:1200])))
    if not query_terms:
        return []

    field_weights = _memory_search_field_weights()
    df = index_data["document_frequency"]
    averages = index_data["average_lengths"]
    lengths = index_data["field_lengths"]
    scores = {}
    k1, b = 1.2, 0.75

    for term in query_terms:
        for field, rows in index_data["postings"].get(term, {}).items():
            idf = math.log(1.0 + (index_data["live_count"] - df[(field, term)] + 0.5)
                           / (df[(field, term)] + 0.5))
            average_length = averages[field]
            weight = field_weights.get(field, 0.75)
            for doc_index, tf in rows:
                length = lengths[doc_index][field]
                norm_tf = (tf * (k1 + 1.0)) / (
                    tf + k1 * (1.0 - b + b * length / average_length)
                )
                scores[doc_index] = scores.get(doc_index, 0.0) + weight * idf * norm_tf

    ranked = [(score, doc_index, documents[doc_index])
              for doc_index, score in scores.items() if score > 0]
    ranked.sort(key=lambda item: (
        -item[0], not bool(item[2].get("highlighted") or
                           item[2].get("_group_highlighted")), item[1]
    ))
    return [(score, document) for score, _, document in ranked]


def _rank_memory_docs(query_text: str, documents: list) -> list:
    """Compatibility entry point for one-off BM25 queries."""
    return _rank_memory_search_index(query_text, _build_memory_search_index(documents))
