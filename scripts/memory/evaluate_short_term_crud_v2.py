"""Execute the expanded short-term CRUD plan against isolated temporary memory stores."""
import json
import re
import sys
import tempfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory

TEST = ROOT / "data/egolife/testset"
RUNS = ROOT / "data/egolife/runs"
ROWS = [json.loads(line) for line in (TEST / "short_term/moments_scattered_v2_120.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
PLAN = json.loads((TEST / "short_term/crud_plan_v2.json").read_text(encoding="utf-8"))
CACHE = json.loads((RUNS / "qwen_cache_expanded_eval.json").read_text(encoding="utf-8"))

def person_of(sample_id):
    parts = sample_id.split("_")
    return parts[1] + "_" + parts[2]

def timestamp_of(sample_id):
    parts = sample_id.split("_")
    day = int(parts[0][3:])
    stamp = parts[3]
    event_date = date(2026, 9, 1) + timedelta(days=day - 1)
    return f"{event_date.isoformat()}T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}"

def as_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]

def fields_for(row):
    fields = (CACHE.get(row["id"]) or {}).get("fields") or {}
    return {
        "scene": str(fields.get("scene") or row.get("scene") or ""),
        "user_action": str(fields.get("user_action") or row.get("user_action") or ""),
        "needs": as_list(fields.get("needs")),
        "solutions": as_list(fields.get("solutions")),
        "people": as_list(fields.get("people")),
        "location": str(fields.get("location") or "") or None,
        "activity": str(fields.get("activity") or "") or None,
        "environments": as_list(fields.get("environments")),
        "objects": as_list(fields.get("objects")),
        "timestamp": timestamp_of(row["id"]),
    }

def main():
    by_id = {row["id"]: row for row in ROWS}
    by_person = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="short_term_crud_v2_") as tmp:
        memories = {}
        for sample_id in PLAN["add"]:
            by_person[person_of(sample_id)].append(sample_id)
        id_map = {}
        add_errors = []
        for person, sample_ids in sorted(by_person.items()):
            mem = PersonMemory(memory_dir=str(Path(tmp) / person))
            mem._save = lambda: None
            memories[person] = mem
            for sample_id in sorted(sample_ids, key=lambda sid: timestamp_of(sid)):
                try:
                    data = fields_for(by_id[sample_id])
                    if not data["scene"] and not data["user_action"]:
                        raise ValueError("no scene or action in Qwen cache")
                    mid = mem.add(**data)
                    id_map[sample_id] = (person, mid)
                    if mid not in mem.memory["layer7"]["moments"]:
                        raise AssertionError("added moment missing from Layer 7")
                except Exception as exc:
                    add_errors.append({"id": sample_id, "error": str(exc)})

        updates = {"applied": 0, "already_equal": 0, "rejected": []}
        for case in PLAN["update"]:
            person, mid = id_map[case["id"]]
            mem = memories[person]
            field, value = case["field"], case["new_value"]
            before = mem.memory["layer7"]["moments"][mid].get(field)
            if before == value:
                ok = mem.update(mid, {field: value})
                if not ok and mem.memory["layer7"]["moments"][mid].get(field) == value:
                    updates["already_equal"] += 1
                    continue
            else:
                ok = mem.update(mid, {field: value})
            after = mem.memory["layer7"]["moments"][mid].get(field)
            if ok and after == value:
                updates["applied"] += 1
            else:
                updates["rejected"].append({"id": case["id"], "field": field, "before": before, "requested": value, "after": after})

        highlights = []
        highlight_errors = []
        for sample_id in PLAN["highlight"]:
            person, mid = id_map[sample_id]
            mem = memories[person]
            if mem.highlight(mid) and mem.memory["layer7"]["moments"][mid].get("highlighted"):
                highlights.append(sample_id)
            else:
                highlight_errors.append(sample_id)

        protected_id = PLAN["highlight"][0]
        person, mid = id_map[protected_id]
        protected_delete_refused = not memories[person].delete(mid, reason="verify highlight protection")

        deletes = []
        delete_errors = []
        for case in PLAN["delete"]:
            person, mid = id_map[case["id"]]
            mem = memories[person]
            if mem.delete(mid, reason=case.get("reason", "short-term benchmark")) and mid not in mem.memory["layer7"]["moments"]:
                deletes.append(case["id"])
            else:
                delete_errors.append(case["id"])

        query_results = []
        for case in PLAN["query"]:
            matched = 0
            try:
                for mem in memories.values():
                    matched += len(mem.query(case["query"], top_k=5))
                query_results.append({"query": case["query"], "completed": True, "matches_across_people": matched})
            except Exception as exc:
                query_results.append({"query": case["query"], "completed": False, "error": str(exc)})

        remaining = sum(len(mem.memory["layer7"]["moments"]) for mem in memories.values())
        report = {
            "scope": "isolated temporary PersonMemory stores; validates CRUD execution and index-safe mutation, not semantic truth of automatically generated labels",
            "short_term_samples": len(ROWS),
            "participant_count": len(memories),
            "add": {"passed": len(id_map), "expected": len(PLAN["add"]), "errors": add_errors},
            "update": {"applied": updates["applied"], "already_equal": updates["already_equal"], "expected": len(PLAN["update"]), "rejected": updates["rejected"]},
            "highlight": {"passed": len(highlights), "expected": len(PLAN["highlight"]), "errors": highlight_errors},
            "highlight_delete_protection": {"passed": protected_delete_refused, "expected": True},
            "delete": {"passed": len(deletes), "expected": len(PLAN["delete"]), "errors": delete_errors},
            "query": {"completed": sum(row["completed"] for row in query_results), "expected": len(PLAN["query"]), "details": query_results},
            "remaining_after_delete": remaining,
        }
        report["ok"] = (
            report["add"]["passed"] == report["add"]["expected"]
            and not updates["rejected"]
            and report["update"]["applied"] + report["update"]["already_equal"] == report["update"]["expected"]
            and report["highlight"]["passed"] == report["highlight"]["expected"]
            and report["highlight_delete_protection"]["passed"]
            and report["delete"]["passed"] == report["delete"]["expected"]
            and report["query"]["completed"] == report["query"]["expected"]
            and remaining == len(PLAN["add"]) - len(PLAN["delete"])
        )
        RUNS.mkdir(parents=True, exist_ok=True)
        out = RUNS / "short_term_crud_v2_result.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("report:", out)
        if not report["ok"]:
            raise SystemExit(1)

if __name__ == "__main__":
    main()
