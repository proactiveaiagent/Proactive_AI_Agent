"""Live status reader for the standalone all-profile multimodal cache warmer."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

_STAGE_LABELS = {
    "starting": "准备中",
    "perception": "视频感知缓存更新",
    "grounding": "物品画面对照",
    "audio": "语音转写文本",
    "complete": "全部完成",
    "reconciling": "核对剩余缓存",
    "stopped": "已停止",
}
_PERCEPTION = re.compile(
    r"perception cache pass (\d+)/(\d+): hit (\d+), refreshed (\d+), "
    r"new (\d+), repaired (\d+), failed (\d+)"
)
_PERCEPTION_LEGACY = re.compile(
    r"perception quality pass (\d+)/(\d+): (\d+) repaired, (\d+) failed"
)
_GROUNDING = re.compile(
    r"object-frame cache (\d+)/(\d+): hit (\d+)"
    r"(?: \(source fast-path (\d+)\))?, new (\d+), failed (\d+)"
)
_AUDIO = re.compile(
    r"audio transcript cache (\d+)/(\d+): hit (\d+), new (\d+), "
    r"failed (\d+), silent (\d+)"
)
_COMPLETE = re.compile(r"multimodal cache pass complete: (\{.*\})")
_QUEUE_COMPLETE = re.compile(r"v7\.\d+ cache refresh complete and audited: (\d+) clips")


def parse_multimodal_prewarm_log(text: str, *, running: bool = False,
                                  pid: int | None = None,
                                  log_path: str | None = None) -> dict:
    """Parse the progress lines emitted by prewarm_multimodal_evidence.py."""
    out = {
        "source": "multimodal_cache",
        "task": "all_profiles",
        "running": bool(running),
        "status": "running" if running else "stopped",
        "pid": pid,
        "log_path": log_path,
        "stage": "starting",
        "stage_label": _STAGE_LABELS["starting"],
        "done": 0,
        "total": 0,
        "cache_hits": None,
        "refreshed": None,
        "new": None,
        "inferred": None,
        "failed": 0,
        "updated_at": None,
        "perception": {"done": 0, "total": 0, "cache_hits": None,
                       "refreshed": None, "new": None, "unknown_success": 0,
                       "repaired": 0, "failed": 0},
        "grounding": {"done": 0, "total": 0, "hit": 0, "source_hit": 0,
                      "new": 0, "failed": 0},
        "audio": {"done": 0, "total": 0, "hit": 0, "new": 0,
                  "failed": 0, "silent": 0},
        "final_totals": None,
        "error": None,
    }
    for line in text.splitlines():
        timestamp = re.match(r"^\[([^]]+)\]", line)
        if timestamp:
            out["updated_at"] = timestamp.group(1)
        match = _PERCEPTION.search(line)
        if match:
            done, total, hit, refreshed, new, repaired, failed = map(int, match.groups())
            out["perception"] = {
                "done": done, "total": total, "cache_hits": hit,
                "refreshed": refreshed, "new": new, "unknown_success": 0,
                "repaired": repaired, "failed": failed,
            }
            out.update(stage="perception", done=done, total=total,
                       cache_hits=hit, refreshed=refreshed, new=new,
                       inferred=new, failed=failed)
            continue
        match = _PERCEPTION_LEGACY.search(line)
        if match:
            done, total, repaired, failed = map(int, match.groups())
            unknown_success = max(0, done - repaired - failed)
            out["perception"] = {
                "done": done, "total": total, "cache_hits": None,
                "refreshed": None, "new": None,
                "unknown_success": unknown_success,
                "repaired": repaired, "failed": failed,
            }
            out.update(stage="perception", done=done, total=total,
                       cache_hits=None, refreshed=None, new=None, inferred=None,
                       failed=failed)
            continue
        match = _GROUNDING.search(line)
        if match:
            done, total, hit, source_hit, new, failed = match.groups()
            done, total, hit, new, failed = map(int, (done, total, hit, new, failed))
            source_hit = int(source_hit or 0)
            out["grounding"] = {"done": done, "total": total, "hit": hit,
                                "source_hit": source_hit, "new": new,
                                "failed": failed}
            out.update(stage="grounding", done=done, total=total,
                       cache_hits=hit, new=new, failed=failed)
            continue
        match = _AUDIO.search(line)
        if match:
            done, total, hit, new, failed, silent = map(int, match.groups())
            out["audio"] = {"done": done, "total": total, "hit": hit,
                            "new": new, "failed": failed, "silent": silent}
            out.update(stage="audio", done=done, total=total,
                       cache_hits=hit, new=new, failed=failed)
            continue
        match = _COMPLETE.search(line)
        if match:
            try:
                totals = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                totals = None
            if isinstance(totals, dict):
                out["final_totals"] = totals
                out["status"] = "complete"
                out["stage"] = "complete"
                out["stage_label"] = _STAGE_LABELS["complete"]
                out["running"] = bool(running)
                out["error"] = None
    if _QUEUE_COMPLETE.search(text):
        match = _QUEUE_COMPLETE.search(text)
        out["final_totals"] = {"clips": int(match.group(1))}
        out["status"] = "complete"
        out["stage"] = "complete"
        out["stage_label"] = _STAGE_LABELS["complete"]
        out["running"] = False
        out["error"] = None
    stage = out["stage"]
    out["stage_label"] = _STAGE_LABELS.get(stage, stage)
    if out["status"] != "complete":
        if running:
            out["status"] = "running"
            out["error"] = None
        else:
            out["status"] = "stopped"
            if text.strip():
                out["error"] = "预热进程已退出，日志没有完成标记"
    return out


def _active_all_profiles_process(proc_root: Path, root: Path):
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
            script_match = any(
                Path(arg.decode("utf-8", "ignore")).name == "prewarm_multimodal_evidence.py"
                for arg in args if arg
            )
            if not script_match or b"--all-profiles" not in args:
                continue
            raw_log = os.readlink(str(entry / "fd" / "1"))
            raw_log = raw_log.removesuffix(" (deleted)")
            log_path = Path(raw_log)
            if not log_path.is_absolute():
                log_path = root / log_path
            return int(entry.name), (log_path if log_path.is_file() else None)
        except (OSError, ValueError, PermissionError):
            continue
    return None


def _active_prewarm_queue_process(proc_root: Path):
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            args = [part.decode("utf-8", "ignore")
                    for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
            if not any(Path(arg).name == "wait_then_prewarm_v72.py" for arg in args):
                continue
            raw_log = os.readlink(str(entry / "fd" / "1"))
            raw_log = raw_log.removesuffix(" (deleted)")
            log_path = Path(raw_log) if raw_log else None
            return int(entry.name), log_path
        except (OSError, ValueError, PermissionError):
            continue
    return None


def _read_tail(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes), os.SEEK_SET)
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def multimodal_prewarm_status(root: Path, proc_root: Path = Path("/proc")):
    """Return live or most recent all-profile prewarm status, if available."""
    root = Path(root)
    active = _active_all_profiles_process(Path(proc_root), root)
    if active:
        pid, log_path = active
        if log_path is None:
            candidates = sorted((root / "logs").glob("prewarm_multimodal_*.log"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = candidates[0] if candidates else None
        text = _read_tail(log_path) if log_path else ""
        return parse_multimodal_prewarm_log(
            text, running=True, pid=pid,
            log_path=str(log_path) if log_path else None,
        )
    queue = _active_prewarm_queue_process(Path(proc_root))
    if queue:
        pid, log_path = queue
        if log_path is None or not log_path.is_absolute():
            candidates = sorted((root / "logs").glob("prewarm_memory_v72_*.log"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = candidates[0] if candidates else None
        elif not log_path.is_file():
            log_path = None
        text = _read_tail(log_path) if log_path else ""
        out = parse_multimodal_prewarm_log(
            text, running=True, pid=pid,
            log_path=str(log_path) if log_path else None,
        )
        if _QUEUE_COMPLETE.search(text):
            out.update(status="complete", running=False, stage="complete",
                       stage_label=_STAGE_LABELS["complete"], error=None)
        else:
            out.update(status="running", running=True, stage="reconciling",
                       stage_label=_STAGE_LABELS["reconciling"], error=None)
        return out
    candidates = sorted((root / "logs").glob("prewarm_multimodal_*.log"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    log_path = candidates[0]
    text = _read_tail(log_path)
    if not text:
        return None
    return parse_multimodal_prewarm_log(text, running=False, log_path=str(log_path))
