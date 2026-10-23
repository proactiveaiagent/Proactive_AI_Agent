#!/usr/bin/env python3
"""Prepare a blank, video-only annotation template for Feishu retrieval evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "test_data/feishu_ar_agent_memory_retrieval_20261013"
MANIFEST = DATASET / "manifest.json"
DEFAULT_OUTPUT = DATASET / "retrieval_annotations.template.jsonl"
SPLIT_VERSION = "feishu-retrieval-split-v1"
SCHEMA_VERSION = 2


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(verify_sha256=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != manifest.get("video_count"):
        raise ValueError("manifest files/count mismatch")
    seen = set()
    for entry in entries:
        clip_id = str(entry.get("clip_id", "")).strip()
        rel = PurePosixPath(str(entry.get("relative_path", "")))
        if not clip_id or clip_id in seen:
            raise ValueError(f"missing or duplicate clip_id: {clip_id!r}")
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe video path: {rel}")
        path = DATASET.joinpath(*rel.parts)
        if not path.is_file() or path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ValueError(f"missing video or size mismatch: {path}")
        if verify_sha256:
            expected = str(entry.get("sha256", "")).lower()
            actual = sha256_file(path)
            if not expected or actual != expected:
                raise ValueError(f"SHA-256 mismatch for {path}")
        seen.add(clip_id)
    return manifest, entries


def split_by_scenario(entries):
    """Freeze one deterministic holdout clip per multi-clip scenario group."""
    groups = {}
    for entry in entries:
        scenario = str(entry.get("scenario", "")).strip()
        clip_id = str(entry.get("clip_id", "")).strip()
        if not scenario or not clip_id:
            raise ValueError("every clip needs a scenario and clip_id for a frozen split")
        groups.setdefault(scenario, []).append(entry)
    assignments = {}
    for scenario, rows in sorted(groups.items()):
        ranked = sorted(
            rows,
            key=lambda entry: hashlib.sha256(
                f"{SPLIT_VERSION}|{scenario}|{entry['clip_id']}".encode("utf-8")
            ).hexdigest(),
        )
        holdout_ids = {str(ranked[0]["clip_id"])} if len(ranked) > 1 else set()
        for entry in rows:
            clip_id = str(entry["clip_id"])
            assignments[clip_id] = "holdout" if clip_id in holdout_ids else "dev"
    return assignments


def row_for(entry, split):
    clip_id = str(entry["clip_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": f"feishu:{clip_id}",
        "query_clip_id": clip_id,
        "category": str(entry["scenario"]),
        "input_video": str(PurePosixPath(entry["relative_path"])),
        "split": split,
        "split_scheme": SPLIT_VERSION,
        "history_clip_ids": None,
        "relevant_memory_clip_ids": None,
        "hard_negative_clip_ids": None,
        "facts_to_retrieve": [],
        "source_evidence": [],
        "review_status": "unreviewed",
        "annotator": None,
        "adjudicator": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()
    manifest, entries = load_manifest(verify_sha256=args.verify_sha256)
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; use --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments = split_by_scenario(entries)
    rows = [row_for(entry, assignments[str(entry["clip_id"])]) for entry in entries]
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "video_count": len(rows),
        "manifest_count": manifest["video_count"], "template_only": True,
        "annotated_count": 0, "source_captions_copied": False,
        "sha256_verified_count": len(rows) if args.verify_sha256 else 0,
        "split_scheme": SPLIT_VERSION,
        "split_counts": {name: sum(row["split"] == name for row in rows) for name in ("dev", "holdout")},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
