#!/usr/bin/env python3
"""Score end-to-end EgoLifeQA multiple-choice accuracy from top-5 memory retrieval.

Only the retrieved Qwen memory snippets are given to the answer model. Official
answer keys are used solely after generation for exact-choice scoring.
"""
import argparse
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/egolife/runs"
DEFAULT_CACHE = RUNS / "qwen_cache_memory_prompt_v3.json"
DEFAULT_QA = ROOT / "data/egolife/testset/long_term/egolifeqa_official_a1_jake.json"
DEFAULT_RETRIEVAL = RUNS / "egolifeqa_memory_retrieval_qwen_20260927_144433.json"


_QUERY_STOPWORDS = {
    "who", "what", "when", "where", "why", "which", "how", "does", "did",
    "do", "is", "are", "was", "were", "the", "a", "an", "to", "of", "in",
    "on", "at", "for", "from", "with", "and", "or", "we", "i", "you", "they",
    "it", "this", "that", "then", "before", "after", "already", "just", "now",
}
_WORD_RE = re.compile(r"[a-z][a-z'-]{1,}", re.I)


def _word_forms(word):
    word = word.lower().strip("'-")
    forms = {word}
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            forms.add(word[:-len(suffix)])
    return forms


def _query_matched_evidence(evidence, question, max_segments=3, max_chars=600):
    query_words = {
        form for word in _WORD_RE.findall(question or "")
        if word.lower() not in _QUERY_STOPWORDS
        for form in _word_forms(word)
    }
    if not query_words:
        return ""
    segments = re.split(r"\s+;\s+", str(evidence or "").strip())
    scored = []
    for index, segment in enumerate(segments):
        words = {
            form for word in _WORD_RE.findall(segment)
            for form in _word_forms(word)
        }
        overlap = len(query_words & words)
        if overlap:
            scored.append((overlap, index, segment.strip()))
    scored.sort(key=lambda row: (-row[0], row[1]))
    selected = []
    length = 0
    for _, _, segment in scored[:max_segments]:
        remaining = max_chars - length - (3 if selected else 0)
        if remaining <= 0:
            break
        segment = segment[:remaining].rstrip()
        if segment:
            selected.append(segment)
            length += len(segment)
    return " | ".join(selected)


def compact_memory(fields, question="", evidence_mode="relevant"):
    labels = {
        "scene": "Scene", "user_action": "Action", "people": "People",
        "location": "Location", "activity": "Activity", "environments": "Environment",
        "objects": "Objects", "needs": "Needs", "solutions": "Solutions",
        "time_occasion": "Time",
    }
    parts = []
    for key, label in labels.items():
        value = fields.get(key)
        if isinstance(value, list):
            value = ", ".join(str(x) for x in value if x)
        if value:
            parts.append(f"{label}: {value}")
    evidence = str(fields.get("source_evidence") or "")
    if evidence_mode == "all":
        evidence = evidence[:2000]
    elif evidence_mode == "relevant":
        evidence = _query_matched_evidence(evidence, question)
    else:
        evidence = ""
    if evidence:
        parts.append(f"Transcript evidence: {evidence}")
    return "\n".join(parts)


def make_prompt(question, choices, memories):
    options = "\n".join(f"{letter}. {choices.get(letter.lower(), '')}" for letter in "ABCD")
    context = "\n\n".join(f"[{i}] {body}" for i, body in enumerate(memories, 1)) or "(No memories retrieved.)"
    return (
        "Answer this multiple-choice question using ONLY the retrieved memory snippets. "
        "Do not use outside knowledge. Choose exactly one option A, B, C, or D. "
        "If evidence is incomplete or ambiguous, still choose the option best supported by the retrieved snippets; never abstain. "
        "Return exactly one JSON object and nothing else: {\"choice\":\"A\"}, "
        "where choice is A, B, C, or D.\n\n"
        f"Question: {question}\n\nChoices:\n{options}\n\nRetrieved memory snippets:\n{context}"
    )


def parse_choice(text, choices=None):
    raw = str(text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            value = str(obj.get("choice", "")).strip().upper()
            if value in {"A", "B", "C", "D"}:
                return value
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.fullmatch(r"[\s*]*(A|B|C|D)[\s*.]*", raw, re.I)
    if match:
        return match.group(1).upper()
    # Accept an option written verbatim, but only when it uniquely matches a choice.
    if choices:
        def normalize(value):
            return re.sub(r"\s+", " ", str(value or "").strip()).casefold()
        candidate = raw
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                candidate = str(obj.get("choice", "")).strip()
        except (json.JSONDecodeError, TypeError):
            pass
        normalized = normalize(candidate)
        matches = [
            letter.upper() for letter in "abcd"
            if normalize(choices.get(letter, choices.get(letter.upper(), ""))) == normalized
        ]
        if normalized and len(matches) == 1:
            return matches[0]
    return None

def ask_model(api_url, prompt, timeout):
    body = json.dumps({"prompt": prompt}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(api_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            if data.get("error"):
                raise RuntimeError(data["error"])
            return str(data.get("analysis", ""))
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Qwen request failed after retries: {last_error}")


def summarize(results):
    total = len(results)
    answered = [row for row in results if row.get("predicted_choice") in {"A", "B", "C", "D"}]
    hit_rows = [row for row in results if row.get("evidence_hit_at_5") is True]
    correct = sum(row.get("correct") is True for row in results)
    correct_on_hit = sum(row.get("correct") is True for row in hit_rows)
    return {
        "questions": total,
        "correct": correct,
        "exact_accuracy": round(correct / total, 4) if total else None,
        "answered": len(answered),
        "answer_coverage": round(len(answered) / total, 4) if total else None,
        "accuracy_when_answered": round(sum(row.get("correct") is True for row in answered) / len(answered), 4) if answered else None,
        "abstained_or_unparsed": total - len(answered),
        "evidence_hit_questions": len(hit_rows),
        "answer_correct_on_evidence_hit": correct_on_hit,
        "accuracy_given_evidence_hit": round(correct_on_hit / len(hit_rows), 4) if hit_rows else None,
    }

def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--api-url", default=os.environ.get("EGOLIFE_QWEN_API", "http://127.0.0.1:8000/consolidate"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume an interrupted report file")
    parser.add_argument("--limit", type=int, default=0, help="Optional small run; 0 means all eligible questions")
    parser.add_argument("--evidence-mode", choices=("relevant", "top1-relevant", "all", "none"), default="none",
                        help="How source transcript evidence is passed to the answer model")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    retrieval = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    qa_by_id = {str(item.get("ID", "")): item for item in qa}
    details = retrieval.get("details", [])
    total_eligible = len(details)
    if not details:
        raise SystemExit("Retrieval report has no eligible question details")

    if args.resume:
        output = args.resume
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("retrieval_report") != str(args.retrieval_report):
            raise SystemExit("Resume file was created from a different retrieval report")
        if report.get("cache") != str(args.cache):
            raise SystemExit("Resume file was created from a different memory cache")
        if report.get("method", {}).get("transcript_evidence_mode", "none") != args.evidence_mode:
            raise SystemExit("Resume file was created with a different transcript evidence mode")
        done = {str(row.get("id")) for row in report.get("details", [])}
    else:
        output = args.output or RUNS / f"egolifeqa_answer_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "benchmark": "official EgoLifeQA A1_JAKE",
            "method": {
                "memory_source": "Qwen memory cache; top-5 results from the causal retrieval benchmark",
                "transcript_evidence_mode": args.evidence_mode,
                "answer_model": "Qwen3-VL-8B-Instruct via /consolidate",
                "scoring": "Forced-choice exact match against the official A/B/C/D answer; a unique exact option-text response is normalized to its letter, while malformed or missing choices count as incorrect.",
                "blindness": "Official answer and target evidence are not included in the generation prompt",
                "scope": "End-to-end retrieval-plus-answer accuracy for every eligible query in the supplied retrieval report; day coverage and cache limits are documented in that report.",
                "temperature": 0.2,
            },
            "retrieval_report": str(args.retrieval_report),
            "eligible_question_count": total_eligible,
            "requested_limit": args.limit or total_eligible,
            "cache": str(args.cache),
            "api_url": args.api_url,
            "details": [],
        }
        done = set()
        write_report(output, report)

    results = report.setdefault("details", [])
    if args.limit:
        details = details[:args.limit]
    for idx, item in enumerate(details, 1):
        qid = str(item.get("id", ""))
        if qid in done:
            continue
        source = qa_by_id.get(qid)
        if source is None:
            report.setdefault("errors", []).append({"id": qid, "error": "official question ID missing"})
            continue
        memories = []
        for rank, sample_id in enumerate(item.get("retrieved_ids", [])[:5]):
            fields = (cache.get(sample_id) or {}).get("fields") or {}
            evidence_mode = args.evidence_mode
            if evidence_mode == "top1-relevant":
                evidence_mode = "relevant" if rank == 0 else "none"
            body = compact_memory(fields, item.get("query", ""), evidence_mode)
            if body:
                memories.append(body)
        choices = {letter: source.get("choice_" + letter.lower(), "") for letter in "ABCD"}
        prompt = make_prompt(item.get("query", ""), choices, memories)
        start = time.perf_counter()
        raw = ask_model(args.api_url, prompt, args.timeout)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        predicted = parse_choice(raw, choices)
        expected = str(source.get("answer", "")).strip().upper()
        row = {
            "id": qid,
            "type": item.get("type", "unknown"),
            "question": item.get("query", ""),
            "retrieved_ids": item.get("retrieved_ids", [])[:5],
            "predicted_choice": predicted,
            "official_choice": expected,
            "correct": predicted == expected,
            "evidence_hit_at_5": bool(set(item.get("retrieved_ids", [])[:5]) & set(item.get("expected_ids", []))),
            "raw_output": raw,
            "latency_ms": elapsed_ms,
        }
        results.append(row)
        done.add(qid)
        report["overall"] = summarize(results)
        per_type = defaultdict(list)
        for result in results:
            per_type[result.get("type", "unknown")].append(result)
        report["by_question_type"] = {key: summarize(value) for key, value in sorted(per_type.items())}
        report["completed_at"] = datetime.now().isoformat(timespec="seconds")
        write_report(output, report)
        print(f"answered {len(results)}/{len(details)}: {qid} -> {predicted or 'UNPARSED'} ({'correct' if row['correct'] else 'incorrect'})", flush=True)

    report["status"] = "complete" if len(results) >= total_eligible else "partial"
    report["overall"] = summarize(results)
    per_type = defaultdict(list)
    for result in results:
        per_type[result.get("type", "unknown")].append(result)
    report["by_question_type"] = {key: summarize(value) for key, value in sorted(per_type.items())}
    report["completed_at"] = datetime.now().isoformat(timespec="seconds")
    write_report(output, report)
    print(json.dumps({"status": report["status"], "overall": report["overall"], "by_question_type": report["by_question_type"], "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
