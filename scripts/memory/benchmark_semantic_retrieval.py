#!/usr/bin/env python3
"""Compare BM25 and local GTE semantic retrieval on official EgoLife clip QA.

This is an isolated retrieval benchmark. It does not modify production memory,
does not use answers to rank results, and does not call the Qwen answer API.
"""
import hashlib
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_short_long_official_qa as qa
from memory import PersonMemory


def model_path():
    base = Path(os.environ.get(
        "MEMORY_GTE_MODEL_ROOT",
        "/data/cxr25/zhx/HF_HOME/sentence_transformers/models--thenlper--gte-base/snapshots",
    ))
    if base.name == "snapshots":
        snapshots = sorted(p for p in base.iterdir() if p.is_dir())
        if not snapshots:
            raise FileNotFoundError(f"No local GTE snapshot in {base}")
        return snapshots[-1]
    return base


class GTE:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(max(1, int(os.environ.get("MEMORY_GTE_CPU_THREADS", "4"))))
        self.torch = torch
        path = str(model_path())
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval()
        self.device = "cpu"

    def encode(self, texts, batch_size=32):
        torch = self.torch
        vectors = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.cpu().numpy())
        return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=qa.CACHE)
    parser.add_argument("--output", type=Path, default=qa.RUNS / "gte_semantic_retrieval_v2_20261008.json")
    args = parser.parse_args()
    cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
    test = [json.loads(line) for line in qa.TEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [json.loads(line) for line in qa.QA_TESTSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    clips = [json.loads(line) for line in qa.QA_CLIP_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if not questions or len({q["question_id"] for q in questions}) != len(questions):
        raise SystemExit("Official QA set is empty or contains duplicate question IDs")
    by_person = {}
    for q in questions:
        by_person.setdefault(q["participant"], []).append(q)

    model_started = time.perf_counter()
    encoder = GTE()
    model_load_ms = round((time.perf_counter() - model_started) * 1000, 1)
    all_clip_ids = sorted({str(row["id"]) for row in test if row.get("id") in cache})
    doc_text = [qa.compact(cache[sid].get("fields") or {}) for sid in all_clip_ids]
    doc_started = time.perf_counter()
    doc_vectors = encoder.encode(doc_text)
    document_encode_ms = round((time.perf_counter() - doc_started) * 1000, 1)
    query_started = time.perf_counter()
    query_vectors = encoder.encode([qa.clean_question(q["question"]) for q in questions])
    query_encode_all_ms = round((time.perf_counter() - query_started) * 1000, 1)
    vector_by_id = {sid: vec for sid, vec in zip(all_clip_ids, doc_vectors)}
    sample_by_person = {}
    for row in test:
        sid = str(row.get("id") or "")
        if sid in vector_by_id:
            sample_by_person.setdefault(qa.participant_of(sid), []).append((sid, cache[sid].get("fields") or {}))
    for rows in sample_by_person.values():
        rows.sort(key=lambda x: (qa.parse_id(x[0])[0], qa.sec(qa.parse_id(x[0])[2]), x[0]))

    rows, query_times = [], []
    for q_index, q in enumerate(questions):
        person = q["participant"]
        qday, qsec = q["day"], qa.sec(q["stamp"]) + qa.CLIP_SECONDS
        past = [(sid, fields) for sid, fields in sample_by_person.get(person, [])
                if (lambda parsed: parsed[0] < qday or
                    (parsed[0] == qday and qa.sec(parsed[2]) + qa.CLIP_SECONDS <= qsec))(
                        qa.parse_id(sid))]
        if not past:
            rows.append({"question_id": q["question_id"], "participant": person,
                         "target_id": q["id"], "bm25_hit": False,
                         "vector_hit": False, "candidate_count": 0,
                         "error": "no causal memories"})
            continue
        with tempfile.TemporaryDirectory(prefix="gte_retrieval_") as td:
            mem = PersonMemory(memory_dir=td)
            mem._save = lambda: None
            id_map = {}
            for sid, fields in past:
                mid = qa.make_mem_add(mem, fields, sid)
                id_map[mid] = sid
            lexical_started = time.perf_counter()
            lexical_ranked = mem.query_ranked(q["question"], top_k=len(past))
            lexical_ms = (time.perf_counter() - lexical_started) * 1000
        query_started = time.perf_counter()
        query_vector = query_vectors[q_index]
        import numpy as np
        scores = np.asarray([float(query_vector @ vector_by_id[sid]) for sid, _ in past])
        top_indices = np.argsort(-scores)
        vector_ranked_ids = [past[int(i)][0] for i in top_indices]
        vector_ids = vector_ranked_ids[:5]
        vector_ms = (time.perf_counter() - query_started) * 1000
        query_times.append(vector_ms)
        lexical_ranked_ids = [id_map.get(item.get("moment", {}).get("id"))
                              for item in lexical_ranked]
        lexical_ids = lexical_ranked_ids[:5]
        rrf = {}
        for ranked_ids in (lexical_ranked_ids, vector_ranked_ids):
            for rank, sid in enumerate(ranked_ids, 1):
                if sid:
                    rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (60 + rank)
        hybrid_ids = sorted(rrf, key=lambda sid: (-rrf[sid], sid))[:5]
        target = str(q["id"])
        rows.append({
            "question_id": q["question_id"], "participant": person,
            "target_id": target, "candidate_count": len(past),
            "bm25_hit": target in lexical_ids, "bm25_ids": lexical_ids,
            "vector_hit": target in vector_ids, "vector_ids": vector_ids,
            "hybrid_hit": target in hybrid_ids, "hybrid_ids": hybrid_ids,
            "bm25_query_ms": round(lexical_ms, 3),
            "vector_query_ms": round(vector_ms, 3),
        })

    def summary(selected):
        n = len(selected)
        return {"questions": n,
                "bm25_top5_hits": sum(bool(x["bm25_hit"]) for x in selected),
                "bm25_top5_recall": round(sum(bool(x["bm25_hit"]) for x in selected) / n, 4) if n else None,
                "gte_top5_hits": sum(bool(x["vector_hit"]) for x in selected),
                "gte_top5_recall": round(sum(bool(x["vector_hit"]) for x in selected) / n, 4) if n else None,
                "rrf_hybrid_top5_hits": sum(bool(x.get("hybrid_hit")) for x in selected),
                "rrf_hybrid_top5_recall": round(sum(bool(x.get("hybrid_hit")) for x in selected) / n, 4) if n else None,
                "either_top5_hits": sum(bool(x["bm25_hit"] or x["vector_hit"]) for x in selected),
                "either_top5_recall": round(sum(bool(x["bm25_hit"] or x["vector_hit"]) for x in selected) / n, 4) if n else None}

    all_query_times = sorted(query_times)
    report = {
        "schema_version": 1,
        "benchmark": "official EgoLife QA target-clip retrieval; causal cutoff after source clip",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "thenlper/gte-base", "model_path": str(model_path()),
        "method": {
            "corpus": "Qwen-derived structured English memory fields for the same participant, cutoff before query time",
            "bm25": "PersonMemory.query top 5",
            "gte": "mean-pool last hidden state, L2 normalize, cosine top 5; no answer or target ID used in ranking",
            "scope_limit": "Text embedding of multimodal-derived Scene/Action/entities, not raw image/audio embedding. Retrieval benchmark only; no answer accuracy measured.",
            "source_files": [str(qa.TEST.relative_to(ROOT)), str(qa.QA_TESTSET.relative_to(ROOT)), str(qa.QA_CLIP_SET.relative_to(ROOT)), str(cache_path.relative_to(ROOT))],
        },
        "sample_coverage": {"source_clips": len(test), "clips_with_cache": len(all_clip_ids),
                            "official_target_clips": len(clips), "questions": len(questions)},
        "performance": {"model_load_ms": model_load_ms,
                        "document_encode_ms": document_encode_ms,
                        "all_query_embeddings_ms": query_encode_all_ms,
                        "cosine_scan_p95_ms": round(all_query_times[min(len(all_query_times)-1, int(len(all_query_times)*0.95))], 3) if all_query_times else None,
                        "cpu_threads": int(os.environ.get("MEMORY_GTE_CPU_THREADS", "4"))},
        "overall": summary(rows),
        "by_participant": {person: summary([r for r in rows if r["participant"] == person])
                           for person in sorted({r["participant"] for r in rows})},
        "details": rows,
    }
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    report["report_sha256"] = digest
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sample_coverage", "performance", "overall", "by_participant", "report_sha256")}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
