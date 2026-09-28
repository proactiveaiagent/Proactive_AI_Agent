"""
qwen_perceive.py — Qwen3-VL 感知客户端（记忆的上游）
======================================================================
职责边界：本模块只做「感知产出」——
    视频片段 → 抽帧 → Qwen3-VL → 结构化记忆条目
7 层记忆系统只负责存储 / 组织 / 检索 / 生命周期。

重要：数据集自带的 caption 与 ground_truth **不是记忆内容**，
只用 golden_eval() 对 Qwen 的产出打分。

用法：
    p = QwenPerceiver()
    out = p.perceive(moment)              # 走缓存；无缓存则调真模型
    out = p.perceive(moment, force=True)  # 强制实时调用真模型
    p.prewarm(moments)                    # 批量预热缓存
"""

import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "egolife" / "runs"
CACHE_FILE = RUNS / "qwen_cache.json"

DEFAULT_API = "http://localhost:8000"
DEFAULT_FRAMES = 4
PROMPT_VERSION = "memory-v2-2026-09-28"


# ---------------------------------------------------------------------------
# Prompt：对齐 code/agent.py 的 PART 1 — Scene Recognition，
# 并补全六维索引所需的 activity / objects / environments。
# ---------------------------------------------------------------------------

def build_prompt(transcript: str, n_frames: int) -> str:
    ts = transcript.strip() if transcript and transcript.strip() else \
        "[No audio transcript — infer from visual only]"
    return f"""Create a concise, factual episodic memory from {n_frames} ordered first-person video frames and this transcript:

{ts}

Return exactly these English fields, one line each. Translate speech into natural English; never copy source-language text. Keep names in Latin script.
Scene: one sentence stating the main event and key fact
Location: broad or specific place supported by the frames; use a broad label if the exact room or store is uncertain
Time/Occasion: only supported time or occasion
People: only named speakers or people clearly identified in the frames; never use generic labels such as visitors, group, others, or people. Use None if no one can be identified
User Action: concrete camera-wearer actions in order; do not assign another person's action to the camera wearer
Activity: the main activity. If speech explains the event, preserve the named person, action, object, and key fact instead of reducing it to generic conversation
Objects: only tangible items clearly visible or explicitly handled/discussed; prefer the few important items. Do not guess furniture, devices, or objects from a room type
Environment: concise physical setting supported by frames only. Do not guess room names, lighting, clutter, or multiple rooms; use a broad description when uncertain
Interaction Details: key supported who-did-what-to-whom/object facts, transfers, state changes, or specific spoken facts

Attribute speech using transcript speaker labels. Separate explicit transcript facts from visual inference. If uncertain, say so briefly in English instead of guessing. Keep the answer under 150 words. Add no extra text."""

def _field(text: str, label: str) -> str:
    """抽取 '- Label: value' 形式的字段值。"""
    for line in text.split("\n"):
        ll = line.lower().strip().lstrip("-").strip()
        if ll.startswith(label.lower() + ":"):
            return line.split(":", 1)[-1].strip()
    return ""


def _split_list(raw: str, person: bool = False):
    if not raw:
        return []
    parts = re.split(r"[,，、;；]", raw)
    out = []
    person_placeholders = {"other", "others", "other people", "people", "group",
                           "crowd", "someone", "somebody", "unknown person",
                           "unidentified person", "camera wearer",
                           "none identified", "none clearly identifiable",
                           "no one identifiable", "no one clearly identifiable",
                           "no identifiable people", "no people identified"}
    for p in parts:
        if re.search(r"\b(implied|possibly|maybe|unclear|uncertain|inferred|guess(?:ed)?)\b", p, re.I):
            continue
        p = re.sub(r"（[^）]*）|\([^)]*\)", "", p).strip().strip(".，。, ;；: ")
        if not p or p.lower() in ("none", "n/a", "unknown", "null", "-"):
            continue
        if person and (p.lower() in person_placeholders or
                       re.fullmatch(r"(?:one|two|three|several|many|a few) others?", p.lower())):
            continue
        out.append(p)
    return out

def parse_analysis(text: str) -> dict:
    """Qwen 输出 → 结构化记忆条目（对齐 memory.add 的参数）。"""
    location = _field(text, "Location")
    people = _split_list(_field(text, "People"), person=True)
    action = _field(text, "User Action")
    interaction = _field(text, "Interaction Details")
    if interaction and interaction.lower() not in action.lower():
        action = (action.rstrip(" .") + ". Interaction details: " +
                  interaction.strip().rstrip(" .")) if action else interaction
    activity = _field(text, "Activity")
    objects = _split_list(_field(text, "Objects"))
    envs = _split_list(_field(text, "Environment"))
    return {
        # Store the event narrative separately from the location so search can find facts.
        "scene": _field(text, "Scene") or location,
        "user_action": action,
        "people": people,
        "location": location or None,
        "activity": activity or None,
        "environments": envs or None,
        "objects": objects or None,
        "time_occasion": _field(text, "Time/Occasion"),
        "needs": [],
        "solutions": [],
    }


def _contains_cjk(text: str) -> bool:
    """Return True when model output still contains untranslated CJK text."""
    return bool(re.search(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]", str(text or "")))


# ---------------------------------------------------------------------------
# 视频 → 抽帧
# ---------------------------------------------------------------------------

def _intra_hour_offset(mid: str) -> int:
    """文件名时间戳相对整点小时的秒数：..._11093000 → 9*60+30 = 570"""
    parts = str(mid).split("_")
    if len(parts) >= 4 and len(parts[3]) >= 6 and parts[3][:6].isdigit():
        t = parts[3]
        return int(t[2:4]) * 60 + int(t[4:6])
    return 0


def clip_range(m: dict):
    """片段在视频文件内的 [start, end] 秒。"""
    st = m.get("start_time")
    if st is None:
        return 0.0, None
    off = _intra_hour_offset(m.get("id", ""))
    return max(0.0, st - off), m.get("end_time", st + 30) - off


def video_path_of(m: dict):
    """数据集 video 字段 → 本地视频绝对路径；不存在返回 None。"""
    v = m.get("video") or ""
    if not v:
        return None
    p = ROOT / v.replace("EgoLife", "egolife")
    return p if p.exists() else None


def extract_frames(vp: Path, start: float, end, n: int = DEFAULT_FRAMES):
    """在片段区间内均匀抽 n 帧，返回 JPEG bytes 列表。"""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {vp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = (total / fps) if fps > 0 else 0.0

    if end is None or end <= start:
        end = (dur or (start + 30))
    end = min(end, dur) if dur else end

    times = [start + (end - start) * i / max(n - 1, 1) for i in range(n)] if n > 1 else [start]
    bufs = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
            ok, fr = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok2:
                bufs.append(buf.tobytes())
    finally:
        cap.release()
    if not bufs:
        raise RuntimeError(f"抽帧失败: {vp}")
    return bufs


# ---------------------------------------------------------------------------
# 感知客户端
# ---------------------------------------------------------------------------

class QwenPerceiver:
    def __init__(self, api_url: str = DEFAULT_API, num_frames: int = DEFAULT_FRAMES,
                 cache_file: Path = CACHE_FILE):
        self.api_url = api_url.rstrip("/")
        self.num_frames = num_frames
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache = None

    # -- 缓存 ---------------------------------------------------------------
    @property
    def cache(self):
        if self._cache is None:
            self._cache = json.loads(self.cache_file.read_text(encoding="utf-8")) \
                if self.cache_file.exists() else {}
        return self._cache

    def _save_cache(self):
        self.cache_file.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- 健康检查 -----------------------------------------------------------
    def health(self):
        try:
            r = requests.get(f"{self.api_url}/health", timeout=3)
            return r.json() if r.status_code == 200 else {"model_loaded": False}
        except Exception as e:
            return {"model_loaded": False, "error": str(e)}

    # -- 核心 ---------------------------------------------------------------
    def perceive(self, m: dict, force: bool = False):
        """
        产出一条记忆条目。force=True 强制实时调用真模型（忽略缓存）。

        返回 {source, fields, raw, elapsed_ms, num_frames, model, error}
        source: 'cache' | 'qwen'
        """
        mid = m.get("id", "")
        if not force and mid in self.cache:
            hit = dict(self.cache[mid])
            fields_text = json.dumps(hit.get("fields") or {}, ensure_ascii=False)
            if not _contains_cjk(fields_text) and not _contains_cjk(hit.get("raw", "")):
                hit["source"] = "cache"
                return hit
            # Do not let an old non-English cache entry re-enter memory.
            self.cache.pop(mid, None)
            self._save_cache()

        vp = video_path_of(m)
        if vp is None:
            return {"source": "error", "error": "视频文件不存在", "fields": None}

        start, end = clip_range(m)
        t0 = time.time()
        try:
            bufs = extract_frames(vp, start, end, self.num_frames)
            base_prompt = build_prompt(m.get("transcript", ""), len(bufs))
            files = [("images", (f"f{i}.jpg", b, "image/jpeg"))
                     for i, b in enumerate(bufs)]
            raw = ""
            for attempt in range(3):
                prompt = base_prompt
                if attempt == 1:
                    prompt += ("\n\nRewrite the previous answer. It violated the English-only rule. "
                               "Translate or paraphrase every phrase into English; never include CJK characters. "
                               "For an idiom whose meaning is uncertain, say it is an unclear colloquial phrase.")
                elif attempt == 2:
                    prompt += ("\n\nFinal rewrite: do not quote any speech. Summarize dialogue only when you can "
                               "state it naturally in English; otherwise write that the colloquial speech was unclear. "
                               "Every field value must contain English text only and zero CJK characters.")
                resp = requests.post(f"{self.api_url}/analyze_video",
                                     files=files, data={"prompt": prompt}, timeout=240)
                if resp.status_code != 200:
                    return {"source": "error",
                            "error": f"API {resp.status_code}: {resp.text[:200]}",
                            "fields": None}
                raw = resp.json().get("analysis", "")
                if not _contains_cjk(raw):
                    break
            if _contains_cjk(raw):
                return {"source": "error", "error": "Qwen output remained non-English after retry; rejected from memory", "fields": None}
        except Exception as e:
            return {"source": "error", "error": str(e), "fields": None}

        elapsed = int((time.time() - t0) * 1000)
        out = {
            "fields": parse_analysis(raw),
            "raw": raw,
            "elapsed_ms": elapsed,
            "num_frames": len(bufs) if bufs else 0,
            "model": self.health().get("model", "Qwen3-VL"),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_version": PROMPT_VERSION,
        }
        self.cache[mid] = out
        self._save_cache()
        out["source"] = "qwen"
        return out

    def prewarm(self, moments, force: bool = False):
        """批量预热缓存，返回 (成功数, 失败数)。"""
        ok = fail = 0
        for m in moments:
            r = self.perceive(m, force=force)
            if r.get("fields"):
                ok += 1
            else:
                fail += 1
        return ok, fail


# ---------------------------------------------------------------------------
# Golden label 评估（数据集标注只在这里用，绝不进记忆）
# ---------------------------------------------------------------------------

def _norm(s):
    return re.sub(r"\s+", "", str(s or "")).lower()


def _match(q, g):
    """单项匹配：相等或互为子串。"""
    qn, gn = _norm(q), _norm(g)
    if not qn or not gn:
        return False
    return qn == gn or qn in gn or gn in qn


def golden_eval(fields: dict, gt: dict):
    """
    对比 Qwen 产出 vs 数据集 golden 标注。

    返回 {dim: {golden:[], qwen:[], hit:[], miss:[], extra:[]}, ...}
    """
    if not fields:
        return {}

    dims = [
        ("people", fields.get("people") or [], gt.get("people_gt") or []),
        ("locations", [fields.get("location")] if fields.get("location") else [],
         gt.get("location_gt") or []),
        ("activity_events", [fields.get("activity")] if fields.get("activity") else [],
         [gt.get("activity_gt")] if gt.get("activity_gt") else []),
        ("objects", fields.get("objects") or [], gt.get("objects_gt") or []),
        ("environments", fields.get("environments") or [], gt.get("environments_gt") or []),
    ]

    out = {}
    for dim, qwen_items, golden_items in dims:
        golden_items = [g for g in golden_items if g]
        hit, miss = [], []
        for g in golden_items:
            (hit if any(_match(q, g) for q in qwen_items) else miss).append(g)
        extra = [q for q in qwen_items if not any(_match(q, g) for g in golden_items)]
        out[dim] = {
            "golden": golden_items,
            "qwen": qwen_items,
            "hit": hit,
            "miss": miss,
            "extra": extra,
            "recall": (len(hit) / len(golden_items)) if golden_items else None,
        }
    return out


if __name__ == "__main__":
    import sys
    from interactive_runner import _load_jsonl, SHORT_FILE

    moments = _load_jsonl(SHORT_FILE)
    p = QwenPerceiver()
    print("Qwen 服务:", p.health())
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for m in moments[:n]:
        r = p.perceive(m, force=True)
        print("\n" + "=" * 60)
        print(m["id"], "| source:", r["source"], "| 耗时:", r.get("elapsed_ms"), "ms")
        if r.get("fields"):
            print("Qwen 产出:", json.dumps(r["fields"], ensure_ascii=False))
            print("Golden   :", json.dumps(m.get("ground_truth", {}), ensure_ascii=False))
            ev = golden_eval(r["fields"], m.get("ground_truth", {}))
            for dim, e in ev.items():
                if e["golden"]:
                    print(f"  {dim}: 命中 {len(e['hit'])}/{len(e['golden'])} "
                          f"miss={e['miss']} extra={e['extra']}")
