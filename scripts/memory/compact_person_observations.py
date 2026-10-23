#!/usr/bin/env python3
"""Compact repeated same-person/same-appearance frame rows in an interactive memory.

Dry-run is the default. --apply creates a one-time backup before changing the
explicit memory JSON. Other memory files are never selected automatically.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import _clean_source_fact_rows
MEMORY_ROOT = (ROOT / "data/egolife/runs/interactive_mem").resolve()
DEFAULT_MEMORY = MEMORY_ROOT / "custom_test/memory.json"


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--source-segment", default=None)
    parser.add_argument("--apply", action="store_true", help="write changes after creating a backup")
    args = parser.parse_args()
    path = args.memory.resolve()
    if not inside(path, MEMORY_ROOT):
        raise SystemExit(f"refusing to edit a memory outside {MEMORY_ROOT}")
    if not path.is_file():
        raise SystemExit(f"memory file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    moments = (data.get("layer7") or {}).get("moments") or {}
    changed = []
    for moment_id, moment in moments.items():
        if not isinstance(moment, dict):
            continue
        provenance = moment.get("source_provenance") or {}
        segment = str(provenance.get("source_segment_id") or "")
        if args.source_segment and segment != args.source_segment:
            continue
        old_rows = moment.get("person_observations") or []
        if not old_rows:
            continue
        new_rows = _clean_source_fact_rows(old_rows, "person", 8)
        if len(new_rows) < len(old_rows):
            changed.append({
                "moment_id": moment_id,
                "source_segment_id": segment,
                "before": len(old_rows),
                "after": len(new_rows),
                "preserved_evidence_refs": sum(len(row.get("evidence_refs") or []) for row in new_rows),
            })
            moment["person_observations"] = new_rows
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "memory": str(path),
        "changed_moments": changed,
        "changed_count": len(changed),
    }
    if not args.apply or not changed:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    backup_dir = ROOT / "deploy_backups/20261017_person_observation_compaction"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / "custom_test_memory.before.json"
    if backup.exists():
        raise SystemExit(f"backup already exists; refusing to overwrite: {backup}")
    shutil.copy2(path, backup)
    tmp = path.with_name(f".{path.name}.compact.tmp")
    with tmp.open("w", encoding="utf-8") as target:
        json.dump(data, target, ensure_ascii=False, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.chmod(tmp, path.stat().st_mode)
    os.replace(tmp, path)
    summary["backup"] = str(backup)
    summary["written"] = True
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
