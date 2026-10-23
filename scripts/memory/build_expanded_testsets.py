"""Build auditable expanded short-term and multi-day EgoLife testsets."""
import json
import math
import os
import re
import shutil
from functools import lru_cache
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EGO = ROOT / "data/egolife"
TEST = EGO / "testset"
RUNS = EGO / "runs"
FPS = 30.0
QA_PATH = TEST / "long_term/egolifeqa_official_a1_jake.json"
CACHE_PATH = RUNS / "qwen_cache_memory_prompt_v3.json"
DAY_RE = re.compile(r"DAY\s*([1-7])", re.I)
STAMP_RE = re.compile(r"(?:(DAY\s*([1-7])[_-]))?(\d{8})", re.I)
SRT_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*\n(.*)", re.S)


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def day_number(value):
    match = DAY_RE.search(str(value or ""))
    return int(match.group(1)) if match else None


def parse_stamp(stamp):
    if not re.fullmatch(r"\d{8}", str(stamp or "")):
        return None
    stamp = str(stamp)
    return (int(stamp[:2]) * 3600 + int(stamp[2:4]) * 60 +
            int(stamp[4:6]) + int(stamp[6:8]) / FPS)


def extract_target_points(target):
    default = day_number(target.get("date")) if isinstance(target, dict) else None
    raw_values = target.get("time_list") if isinstance(target, dict) else None
    if not isinstance(raw_values, list):
        raw_values = [target.get("time", "") if isinstance(target, dict) else ""]
    points, invalid = [], 0
    for raw in raw_values:
        matches = list(STAMP_RE.finditer(str(raw or "")))
        if not matches and re.fullmatch(r"\d{8}", str(raw or "").strip()):
            stamp = str(raw).strip()
            if default is not None:
                points.append((default, stamp))
            else:
                invalid += 1
            continue
        if not matches:
            invalid += 1
            continue
        for match in matches:
            day = int(match.group(2)) if match.group(2) else default
            stamp = match.group(3)
            if day is None or parse_stamp(stamp) is None:
                invalid += 1
            else:
                points.append((day, stamp))
    return sorted(set(points)), invalid


@lru_cache(maxsize=None)
def parse_srt(path):
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    cues = []
    for block in re.split(r"\n\s*\n", raw):
        match = SRT_TIME_RE.search(block)
        if not match:
            continue
        a = list(map(int, match.groups()[:4]))
        b = list(map(int, match.groups()[4:8]))
        start = a[0]*3600 + a[1]*60 + a[2] + a[3]/1000
        end = b[0]*3600 + b[1]*60 + b[2] + b[3]/1000
        text = " ".join(line.strip() for line in match.group(9).splitlines() if line.strip())
        if text:
            cues.append((start, end, text))
    return cues


def srt_path(kind, day, hour):
    return EGO / "EgoLifeCap" / kind / "A1_JAKE" / f"DAY{day}" / f"A1_JAKE_DAY{day}_{hour:02d}000000.srt"


def row_from_video(path, day, stamp, source_kind, qa_ref_count=0):
    stem = path.stem
    hour = int(stamp[:2])
    minute = int(stamp[2:4])
    second = int(stamp[4:6])
    frames = int(stamp[6:8])
    offset = minute*60 + second + frames/FPS
    end = offset + 30.0
    dense_cues = [c for c in parse_srt(srt_path("DenseCaption", day, hour))
                  if c[0] < end and c[1] > offset]
    speech_cues = [c for c in parse_srt(srt_path("Transcript", day, hour))
                   if c[0] < end and c[1] > offset]
    dense = " ".join(c[2] for c in dense_cues)
    transcript = " ".join(c[2] for c in speech_cues)
    return {
        "id": stem + "_en",
        "video": str(path.relative_to(ROOT)).replace("egolife/", "EgoLife/"),
        "scene": dense,
        "user_action": dense,
        "people": [],
        "location": "",
        "transcript": transcript,
        "dense_caption": dense,
        "start_time": minute*60 + second,
        "end_time": minute*60 + second + 30.0,
        "_hour": stamp[:2],
        "_day": day,
        "_source_kind": source_kind,
        "_qa_reference_count": qa_ref_count,
    }


def build_short_term():
    candidates = load_jsonl(TEST / "moments_sampled.jsonl")
    by_person = defaultdict(list)
    for row in candidates:
        if str(row.get("id", "")).endswith("_en"):
            parts = row["id"].split("_")
            by_person[f"{parts[1]}_{parts[2]}"].append(row)
    picked = []
    for person in sorted(by_person):
        rows = sorted(by_person[person], key=lambda r: (float(r.get("start_time", 0)), r["id"]))
        if len(rows) < 20:
            raise RuntimeError(f"{person}: only {len(rows)} English rows available, need 20")
        indices = [round(i*(len(rows)-1)/19) for i in range(20)]
        for index, row_index in enumerate(indices):
            row = dict(rows[row_index])
            row["_hour"] = row["id"].split("_")[3][:2]
            row["_day"] = 1
            row["_gap_before"] = None if index == 0 else row_index - indices[index-1]
            picked.append(row)
    picked.sort(key=lambda r: (r["_day"], r["_hour"], float(r.get("start_time", 0)), r["id"]))
    for row in picked:
        source = row.get("video", "").replace("EgoLife", "egolife")
        if not (ROOT / source).is_file():
            raise RuntimeError(f"Short-term sample video missing: {row.get('video')}")
    return picked


def build_crud_plan(rows):
    ids = [r["id"] for r in rows]
    update, used = [], set()
    for row in rows:
        if len(update) >= 12:
            break
        gt = row.get("ground_truth") or {}
        for field in ("location", "environments", "objects", "activity"):
            values = gt.get(f"{field}_gt")
            if isinstance(values, str):
                values = [values] if values else []
            if values:
                new_value = values[0]
                if field in ("people", "environments", "objects", "needs", "solutions"):
                    new_value = [new_value]
                update.append({"id": row["id"], "field": field, "new_value": new_value,
                               "reason": f"Ground-truth correction case for {field}."})
                used.add(row["id"])
                break
    highlight = [i for i in ids if i not in used][:12]
    used.update(highlight)
    delete = [{"id": i, "reason": "Short-term lifecycle cleanup case."}
              for i in ids if i not in used][:8]
    return {
        "add": ids,
        "update": update,
        "highlight": highlight,
        "delete": delete,
        "query": [{"query": q, "type": "keyword"}
                  for q in ["Jake", "Alice", "kitchen", "living room", "phone", "meeting"]],
        "note": "Expanded balanced short-term set: 20 English narrative clips per participant, selected at separated times from DAY1.",
    }


def build_multiday_clips():
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    videos_by_day = defaultdict(list)
    for day in range(1, 8):
        day_dir = EGO / f"A1_JAKE/DAY{day}"
        for path in day_dir.glob(f"DAY{day}_A1_JAKE_*.mp4"):
            stamp = path.stem.rsplit("_", 1)[-1]
            sec = parse_stamp(stamp)
            if sec is not None:
                videos_by_day[day].append((sec, stamp, path))
        videos_by_day[day].sort()

    refs = Counter()
    qid_per_clip = defaultdict(set)
    question_count_with_media = 0
    unmapped_points = []
    for item in qa:
        qid = str(item.get("ID", ""))
        qtime = item.get("query_time") or {}
        qday = day_number(qtime.get("date"))
        qsec = parse_stamp(qtime.get("time"))
        points, _ = extract_target_points(item.get("target_time") or {})
        q_has_media = False
        for target_day, target_stamp in points:
            target_sec = parse_stamp(target_stamp)
            if qday is None or qsec is None or target_sec is None:
                continue
            if target_day > qday or (target_day == qday and target_sec >= qsec):
                continue
            candidates = [v for v in videos_by_day.get(target_day, [])
                          if v[0] <= target_sec < v[0] + 30.2]
            if not candidates:
                unmapped_points.append({"question_id": qid, "day": target_day, "stamp": target_stamp})
                continue
            _, clip_stamp, clip_path = candidates[-1]
            sample_id = clip_path.stem + "_en"
            refs[sample_id] += 1
            qid_per_clip[sample_id].add(qid)
            q_has_media = True
        question_count_with_media += int(q_has_media)

    selected = {}
    for day in range(1, 8):
        for _, stamp, path in videos_by_day[day]:
            sample_id = path.stem + "_en"
            if sample_id in refs:
                selected[sample_id] = (day, stamp, path, "official_qa_target", refs[sample_id])
    target_ids = set(selected)

    # Add deterministic temporal distractors from each later day. DAY1 already
    # has a cached full-day corpus; days 2-7 need non-target candidate memories.
    decoys_per_day = 50
    decoy_counts = {}
    for day in range(2, 8):
        pool = [v for v in videos_by_day[day] if v[2].stem + "_en" not in target_ids]
        k = min(decoys_per_day, len(pool))
        indices = [round(i*(len(pool)-1)/(k-1)) for i in range(k)] if k > 1 else ([0] if k else [])
        for ix in indices:
            _, stamp, path = pool[ix]
            sample_id = path.stem + "_en"
            selected[sample_id] = (day, stamp, path, "uniform_day_distractor", 0)
        decoy_counts[f"DAY{day}"] = k

    rows = []
    for sample_id, (day, stamp, path, kind, ref_count) in selected.items():
        rows.append(row_from_video(path, day, stamp, kind, ref_count))
    rows.sort(key=lambda r: (r["_day"], parse_stamp(r["id"].split("_")[3]),
                             r["id"]))
    return rows, {
        "official_qa_questions": len(qa),
        "questions_with_local_causal_target_video": question_count_with_media,
        "unique_official_target_clips": len(target_ids),
        "target_clips_by_day": dict(sorted(Counter(f"DAY{v[0]}" for v in selected.values()
                                                  if v[3] == "official_qa_target").items())),
        "uniform_distractors_by_day": decoy_counts,
        "unmapped_target_points": unmapped_points,
    }


def main():
    TEST.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    short_path = TEST / "short_term/moments_scattered_v2_120.jsonl"
    plan_path = TEST / "short_term/crud_plan_v2.json"
    long_path = TEST / "long_term/multiday_qa_clipset.jsonl"
    report_path = TEST / "data_coverage_report.json"

    short_rows = build_short_term()
    dump_jsonl(short_path, short_rows)
    plan = build_crud_plan(short_rows)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    long_rows, qa_meta = build_multiday_clips()
    dump_jsonl(long_path, long_rows)

    mp4_by_group = {}
    all_mp4 = list(EGO.glob("A[1-6]_*/*/*.mp4"))
    for path in all_mp4:
        key = f"{path.parent.parent.name}/{path.parent.name}"
        item = mp4_by_group.setdefault(key, {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += path.stat().st_size
    annotation_coverage = {}
    for kind in ("DenseCaption", "Transcript"):
        annotation_coverage[kind] = {}
        for person in sorted((EGO / f"EgoLifeCap/{kind}").glob("A[1-6]_*")):
            for day_dir in sorted(person.glob("DAY*")):
                annotation_coverage[kind][f"{person.name}/{day_dir.name}"] = len(list(day_dir.glob("*.srt")))

    v3_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    short_cached = sum(r["id"] in v3_cache for r in short_rows)
    long_cached = sum(r["id"] in v3_cache for r in long_rows)
    existing_short = load_jsonl(TEST / "short_term/moments_scattered.jsonl")
    existing_long = load_jsonl(TEST / "long_term/full_day.jsonl")
    old_short_present = sum((ROOT / r["video"].replace("EgoLife", "egolife")).is_file()
                            for r in existing_short)
    old_long_en = [r for r in existing_long if str(r.get("id", "")).endswith("_en")]
    old_long_present = sum((ROOT / r["video"].replace("EgoLife", "egolife")).is_file()
                           for r in old_long_en)
    usage = shutil.disk_usage(ROOT)
    mp4_bytes = sum(v["bytes"] for v in mp4_by_group.values())
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(EGO),
        "local_video_inventory": {
            "mp4_files": len(all_mp4),
            "mp4_bytes": mp4_bytes,
            "mp4_gib": round(mp4_bytes / 1024**3, 2),
            "mp4_by_participant_day": dict(sorted(mp4_by_group.items())),
            "filesystem_total_gib": round(usage.total / 1024**3, 1),
            "filesystem_used_gib": round((usage.total-usage.free) / 1024**3, 1),
            "filesystem_free_gib": round(usage.free / 1024**3, 1),
            "filesystem_used_percent": round(100*(usage.total-usage.free)/usage.total, 1),
        },
        "annotation_srt_files_by_participant_day": annotation_coverage,
        "existing_testsets": {
            "short_term": {"rows": len(existing_short), "videos_present": old_short_present},
            "long_term_day1": {"rows_bilingual": len(existing_long),
                               "unique_english_clips": len(old_long_en),
                               "videos_present": old_long_present},
            "main_sampled": {"rows": len(load_jsonl(TEST / "moments_sampled.jsonl")),
                             "videos_present": sum((ROOT / r["video"].replace("EgoLife", "egolife")).is_file()
                                                   for r in load_jsonl(TEST / "moments_sampled.jsonl"))},
        },
        "expanded_short_term": {
            "file": str(short_path), "rows": len(short_rows),
            "rows_by_participant": dict(sorted(Counter("_".join(r["id"].split("_")[1:3]) for r in short_rows).items())),
            "videos_present": len(short_rows), "qwen_v3_cache_hits_before_run": short_cached,
            "qwen_missing_before_run": len(short_rows)-short_cached,
            "crud_plan": str(plan_path), "crud_cases": {k: len(v) for k, v in plan.items()
                                                       if isinstance(v, list)},
        },
        "expanded_long_term": {
            "file": str(long_path), "rows": len(long_rows),
            "rows_by_day": dict(sorted(Counter(f"DAY{r['_day']}" for r in long_rows).items())),
            "official_qa_target_clips": sum(r["_source_kind"] == "official_qa_target" for r in long_rows),
            "uniform_distractors": sum(r["_source_kind"] == "uniform_day_distractor" for r in long_rows),
            "videos_present": len(long_rows), "qwen_v3_cache_hits_before_run": long_cached,
            "qwen_missing_before_run": len(long_rows)-long_cached,
            "scope_note": "Sparse, official-QA-referenced 7-day evidence corpus plus evenly sampled DAY2-DAY7 distractors; not a full 7-day ingest.",
            **qa_meta,
        },
        "qwen_memory_prompt_v3_cache_before_run": {"file": str(CACHE_PATH), "entries": len(v3_cache)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("coverage report:", report_path)


if __name__ == "__main__":
    main()
