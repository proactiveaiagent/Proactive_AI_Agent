#!/usr/bin/env python3
"""Compare short-term and currently cached memory on official EgoLife clip QA.

All valid official EgoLifeQA questions for the 120 short-term reference clips
are evaluated. The source video clip is the official target segment. Answer
accuracy is judged semantically by a separate blind Qwen request.
"""
import json, re, sys, time, tempfile, urllib.request
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory
DATA = ROOT / "data/egolife"
RUNS = DATA / "runs"
TEST = DATA / "testset/short_term/moments_scattered_v2_120.jsonl"
QA_TESTSET = DATA / "testset/short_term/official_video_qa_v1_questions_810.jsonl"
QA_CLIP_SET = DATA / "testset/short_term/official_video_qa_v1_clips_84.jsonl"
CACHE = RUNS / "qwen_cache_expanded_eval.json"
OUT = RUNS / "short_long_official_qa_all_20260928.json"
API = "http://127.0.0.1:8000/consolidate"
FPS = 30.0
CLIP_SECONDS = 30.0


def participant_of(sid):
    p = sid.split("_")
    return p[1] + "_" + p[2]


def parse_id(sid):
    m = re.fullmatch(r"DAY([1-7])_(A[1-6]_[A-Z]+)_(\d{8})_en", sid)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3)


def sec(stamp):
    return int(stamp[:2])*3600 + int(stamp[2:4])*60 + int(stamp[4:6]) + int(stamp[6:8])/FPS


def timestamp(day, stamp):
    dt = date(2026, 9, 1) + timedelta(days=day-1)
    return f"{dt.isoformat()}T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}"


def video_rel(sid):
    parsed = parse_id(sid)
    if not parsed: return ""
    day, person, stamp = parsed
    base = sid[:-3]
    return f"{person}/DAY{day}/{base}.mp4"


def clean_question(s):
    return re.sub(r"<[^>]+>", " ", str(s or "")).strip()


def as_list(x):
    if isinstance(x, list): return [str(v).strip() for v in x if str(v).strip()]
    if x in (None, ""): return []
    return [str(x).strip()]


def compact(fields):
    labels = {"scene":"Scene", "user_action":"Action", "people":"People", "location":"Location", "activity":"Activity", "environments":"Environment", "objects":"Objects", "needs":"Needs", "solutions":"Solutions", "time_occasion":"Time"}
    out=[]
    for k,label in labels.items():
        v=fields.get(k)
        if isinstance(v,list): v=", ".join(str(x) for x in v if x)
        if v: out.append(f"{label}: {v}")
    return "\n".join(out)


def api(prompt):
    body=json.dumps({"prompt":prompt}, ensure_ascii=False).encode("utf-8")
    req=urllib.request.Request(API, data=body, headers={"Content-Type":"application/json"}, method="POST")
    last=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data=json.loads(resp.read().decode("utf-8"))
            if data.get("error"): raise RuntimeError(data["error"])
            return str(data.get("analysis", ""))
        except Exception as exc:
            last=exc
            if attempt < 2: time.sleep(2*(attempt+1))
    raise RuntimeError(f"Qwen API failed: {last}")


def parse_verdict(text):
    try:
        x=json.loads(text)
        if isinstance(x,dict) and isinstance(x.get("correct"), bool): return x["correct"]
    except Exception: pass
    m=re.search(r"(?i)\b(true|false)\b", text or "")
    return m.group(1).lower()=="true" if m else False


def query_prompt(question, memories):
    context="\n\n".join(f"[{i}] {m}" for i,m in enumerate(memories,1)) or "(No memories retrieved.)"
    return ("Answer this question using only the memory snippets below. Do not use outside knowledge. "
            "If the snippets do not contain the answer, say 'Unknown'. Give a concise answer.\n\n"
            f"Question: {question}\n\nMemory snippets:\n{context}")


def judge_prompt(question, official, predicted):
    return ("You are a strict but fair semantic answer judge. Compare the model answer to the official reference answer for the question. "
            "Ignore harmless paraphrase and wording differences. Mark correct only when the answer conveys the key facts needed by the reference; "
            "mark false if it omits a necessary fact, contradicts the reference, answers a different question, or says Unknown. "
            'Do not use outside knowledge. Return only JSON: {"correct":true} or {"correct":false}.\n\n'
            f"Question: {question}\nOfficial reference answer: {official}\nModel answer: {predicted}")


def make_mem_add(mem, fields, sid):
    day, person, stamp = parse_id(sid)
    return mem.add(
        scene=str(fields.get("scene") or ""), user_action=str(fields.get("user_action") or ""),
        needs=as_list(fields.get("needs")), solutions=as_list(fields.get("solutions")),
        people=as_list(fields.get("people")) or None,
        location=str(fields.get("location") or "") or None,
        activity=str(fields.get("activity") or "") or None,
        environments=as_list(fields.get("environments")) or None,
        objects=as_list(fields.get("objects")) or None,
        extra_notes=str(fields.get("time_occasion") or "")[:200],
        timestamp=timestamp(day,stamp))


def summarize(rows):
    n=len(rows)
    if not n: return {"questions":0,"top5_hit":0,"top5_hit_rate":None,"answer_correct":0,"answer_accuracy":None}
    hits=sum(r.get("top5_hit") is True for r in rows)
    correct=sum(r.get("answer_correct") is True for r in rows)
    return {"questions":n,"top5_hit":hits,"top5_hit_rate":round(hits/n,4),"answer_correct":correct,"answer_accuracy":round(correct/n,4)}


def write_report(path, report):
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    samples=[json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    qa_test_clips=[json.loads(l) for l in QA_CLIP_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions=[json.loads(l) for l in QA_TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    cache=json.loads(CACHE.read_text(encoding="utf-8"))
    if not questions:
        raise SystemExit("Official QA test set is empty")
    matched_clip_ids={q["id"] for q in questions}
    test_clip_ids={c["id"] for c in qa_test_clips}
    question_ids=[q["question_id"] for q in questions]
    if matched_clip_ids != test_clip_ids or len(question_ids) != len(set(question_ids)):
        raise SystemExit("Official QA test set files are inconsistent or contain duplicate question IDs")
    qa_coverage = {
        "short_term_clips": len(qa_test_clips),
        "source_short_term_clips": len(samples),
        "official_qa_clips": len(test_clip_ids),
        "official_qa_questions": len(questions),
        "excluded_no_official_qa_clips": len(samples) - len(test_clip_ids),
        "by_participant": {person: sum(q["participant"] == person for q in questions) for person in sorted({participant_of(r["id"]) for r in samples})},
        "clips_by_participant": {person: len({q["id"] for q in questions if q["participant"] == person}) for person in sorted({participant_of(r["id"]) for r in samples})},
        "testset_file": str(QA_TESTSET.relative_to(ROOT)),
        "testset_clip_file": str(QA_CLIP_SET.relative_to(ROOT)),
    }
    report={"created_at":time.strftime("%Y-%m-%d %H:%M:%S"),"status":"running","benchmark":"all valid official EgoLifeQA questions matched to short-term clips","method":{"question_source":"The persisted official_video_qa_v1_questions_810.jsonl set; all valid official questions for the 84 retained clips, with clips lacking official QA removed from this QA test set.","target":"The official QA source video clip itself; each top-5 hit requires exact retrieval of that clip ID.","short_term_memory":"The 120 Qwen-cached short-term snippets remain the retrieval corpus in six isolated participant stores; only the 84 labeled clips are QA test targets. Causal cutoff is after the source 30-second clip ends.","long_term_memory":"All currently cached Qwen memory for the same participant; future clips are excluded by the same causal cutoff. Multi-day memory is available only for A1_JAKE in the current cache.","answer_scoring":"Qwen generates an answer without seeing the reference; a separate Qwen call judges semantic agreement against the official reference answer. This is AI-judged, not human gold.","scope":"This is a same-question, single-day comparison across the 84 labeled target clips (810 questions). The 36 source clips without official QA are excluded from this QA test set. It does not replace the separate 500-question A1_JAKE cross-day multiple-choice benchmark."},"participant_names":{"A1_JAKE":"Jake","A2_ALICE":"Alice","A3_TASHA":"Tasha","A4_LUCIA":"Lucia","A5_KATRINA":"Katrina","A6_SHURE":"Shure"},"question_count":len(questions),"sample_coverage":qa_coverage,"short_term":{"overall":{},"by_participant":{},"details":[]},"long_term":{"overall":{},"by_participant":{},"details":[]}}
    if OUT.exists():
        prior=json.loads(OUT.read_text(encoding="utf-8"))
        if prior.get("benchmark")==report["benchmark"]:
            report=prior
            report["status"]="running"
            report["question_count"]=len(questions)
            report["sample_coverage"]=qa_coverage
    for scope in ("short_term","long_term"):
        done={r.get("question_id",r["id"]) for r in report[scope].get("details",[])}
        for q in questions:
            key=(scope,q["id"])
            if q["question_id"] in done: continue
            person=q["participant"]
            clips=[]
            if scope=="short_term":
                clips=[(r["id"], cache[r["id"]].get("fields") or {}) for r in samples if participant_of(r["id"])==person]
            else:
                for sid,val in cache.items():
                    parsed=parse_id(sid)
                    if parsed and parsed[1]==person and isinstance(val,dict) and isinstance(val.get("fields"),dict): clips.append((sid,val["fields"]))
            clips.sort(key=lambda x:(parse_id(x[0])[0],sec(parse_id(x[0])[2]),x[0]))
            qday,qsec=q["day"],sec(q["stamp"])+CLIP_SECONDS
            with tempfile.TemporaryDirectory(prefix=f"officialqa_{scope}_",dir=RUNS) as td:
                mem=PersonMemory(memory_dir=td); mem._save=lambda:None
                id_map={}; idx=0
                while idx<len(clips):
                    sid,fields=clips[idx]; d,_,stamp=parse_id(sid)
                    clip_end=sec(stamp)+CLIP_SECONDS
                    if not (d<qday or (d==qday and clip_end<=qsec)): break
                    mid=make_mem_add(mem,fields,sid);id_map[mid]=sid;idx+=1
                found=mem.query(q["question"],top_k=5)
                retrieved=[]; mem_text=[]
                for m in found:
                    sid=id_map.get(m.get("id"))
                    if not sid: continue
                    retrieved.append(sid)
                    fields=next((f for x,f in clips if x==sid),{})
                    text=compact(fields)
                    if text:mem_text.append(text)
                started=time.perf_counter()
                predicted=api(query_prompt(q["question"],mem_text))
                answer_ms=round((time.perf_counter()-started)*1000,1)
                started=time.perf_counter()
                judge=api(judge_prompt(q["question"],q["official_answer"],predicted))
                judge_ms=round((time.perf_counter()-started)*1000,1)
                row={**q,"scope":scope,"retrieved_ids":retrieved,"top5_hit":bool(set(retrieved[:5])&set(q["expected_ids"])),"model_answer":predicted,"answer_correct":parse_verdict(judge),"judge_output":judge,"answer_latency_ms":answer_ms,"judge_latency_ms":judge_ms,"memory_clips_available":len(clips)}
                report[scope]["details"].append(row)
                group=report[scope]["details"]
                report[scope]["overall"]=summarize(group)
                by=defaultdict(list)
                for r in group:by[r["participant"]].append(r)
                report[scope]["by_participant"]={k:summarize(v) for k,v in sorted(by.items())}
                report["updated_at"]=time.strftime("%Y-%m-%d %H:%M:%S")
                write_report(OUT,report)
                print(f"{scope} {len(group)}/{len(questions)} {person} {q['question_id']} hit={row['top5_hit']} answer={row['answer_correct']}",flush=True)
    report["status"]="complete"
    report["completed_at"]=time.strftime("%Y-%m-%d %H:%M:%S")
    write_report(OUT,report)
    print(json.dumps({"status":report["status"],"short_term":report["short_term"]["overall"],"long_term":report["long_term"]["overall"],"by_participant_short":report["short_term"]["by_participant"],"by_participant_long":report["long_term"]["by_participant"],"output":str(OUT)},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
