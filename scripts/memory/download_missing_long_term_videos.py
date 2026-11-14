"""Safely fetch only missing DAY1 clips needed by the three partial person sets.

The command is a dry run unless ``--download`` is given. It pins the Hugging
Face snapshot, checks exact remote sizes and local free space, and verifies every
download before writing a completion manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/egolife/structured/moments.jsonl"
DATASET_DIR = ROOT / "data/egolife"
TARGET_PARTICIPANTS = {"A4_LUCIA", "A5_KATRINA", "A6_SHURE"}
DEFAULT_REVISION = "143fb319be7aa5ae210c936bf4f0f3a86092afb0"
GIB = 1024**3


def participant_of(row: dict) -> str:
    parts = str(row.get("id", "")).split("_")
    return "_".join(parts[1:3]) if len(parts) >= 3 else ""


def remote_path(video: str) -> str:
    value = str(video or "").replace("\\", "/")
    for prefix in ("data/EgoLife/", "data/egolife/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe dataset video path: {video!r}")
    if path.parts[0] not in TARGET_PARTICIPANTS or len(path.parts) < 3 or path.parts[1] != "DAY1":
        raise ValueError(f"Unexpected long-term video path: {video!r}")
    return path.as_posix()


def source_rows() -> tuple[dict[str, set[str]], str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    digest = hashlib.sha256()
    with SOURCE.open("rb") as raw:
        for line in raw:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("id", "")).endswith("_en"):
                continue
            participant = participant_of(row)
            if participant not in TARGET_PARTICIPANTS:
                continue
            if "DAY1" not in str(row.get("id", "")).split("_"):
                continue
            video = str(row.get("video") or "")
            if video:
                grouped[participant].add(remote_path(video))
    return grouped, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Pinned EgoLife dataset commit (default: audited revision)")
    parser.add_argument("--min-free-gib", type=float, default=100.0,
                        help="Require at least this much disk space after download")
    parser.add_argument("--max-download-gib", type=float, default=40.0,
                        help="Refuse plans larger than this cap")
    parser.add_argument("--download", action="store_true",
                        help="Perform the download; without this flag only inspect the plan")
    args = parser.parse_args()

    grouped, source_sha256 = source_rows()
    api = HfApi()
    info = api.repo_info("lmms-lab/EgoLife", repo_type="dataset",
                         revision=args.revision, files_metadata=True)
    siblings = {item.rfilename: item for item in (info.siblings or [])}
    missing_by_person: dict[str, list[tuple[str, int]]] = {}
    unknown = []
    for participant in sorted(TARGET_PARTICIPANTS):
        missing = []
        for path in sorted(grouped.get(participant, set())):
            local = DATASET_DIR / Path(*PurePosixPath(path).parts)
            if local.is_file():
                continue
            sibling = siblings.get(path)
            size = getattr(sibling, "size", None) if sibling else None
            if size is None:
                unknown.append(path)
            else:
                missing.append((path, int(size)))
        missing_by_person[participant] = missing

    if unknown:
        raise RuntimeError("Remote size/path missing; refusing download: " + ", ".join(unknown[:20]))

    files = [(participant, path, size)
             for participant, entries in missing_by_person.items()
             for path, size in entries]
    total_bytes = sum(size for _, _, size in files)
    free_bytes = shutil.disk_usage(DATASET_DIR).free
    projected_free = free_bytes - total_bytes
    summary = {
        "dataset": "lmms-lab/EgoLife",
        "revision": info.sha,
        "source_sha256": source_sha256,
        "participants": {
            person: {"missing_files": len(entries), "bytes": sum(size for _, size in entries)}
            for person, entries in missing_by_person.items()
        },
        "unique_missing_files": len(files),
        "bytes": total_bytes,
        "free_bytes": free_bytes,
        "projected_free_bytes": projected_free,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.download:
        print("DRY RUN: no files changed; pass --download to fetch this exact missing set.")
        return 0
    if info.sha != args.revision:
        raise RuntimeError(f"Resolved revision mismatch: requested {args.revision}, got {info.sha}")
    if total_bytes > args.max_download_gib * GIB:
        raise RuntimeError(f"Plan exceeds --max-download-gib={args.max_download_gib}.")
    if projected_free < args.min_free_gib * GIB:
        raise RuntimeError(
            f"Projected free space {projected_free / GIB:.1f} GiB is below "
            f"the required {args.min_free_gib:.1f} GiB."
        )
    if not files:
        print("All target DAY1 video files are already present.")
        return 0

    manifest_dir = ROOT / "data/egolife/testset/long_term/download_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"missing_day1_videos_{info.sha[:12]}.json"
    manifest = {**summary, "status": "downloading",
                "files": [{"participant": person, "path": path, "bytes": size}
                          for person, path, size in files]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    snapshot_download(
        repo_id="lmms-lab/EgoLife",
        repo_type="dataset",
        revision=info.sha,
        local_dir=str(DATASET_DIR),
        allow_patterns=[path for _, path, _ in files],
        max_workers=4,
    )

    failures = []
    for participant, path, expected_size in files:
        local = DATASET_DIR / Path(*PurePosixPath(path).parts)
        if not local.is_file() or local.stat().st_size != expected_size:
            failures.append({"path": path, "expected_bytes": expected_size,
                             "actual_bytes": local.stat().st_size if local.exists() else None})
    if failures:
        manifest["status"] = "verification_failed"
        manifest["failures"] = failures
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        raise RuntimeError(f"Downloaded files failed size verification ({len(failures)}).")

    manifest["status"] = "complete"
    manifest["verified_files"] = len(files)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"Verified {len(files)} files. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
