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
DEFAULT_FRAMES = 6
PROMPT_VERSION = "memory-v5-visible-actions-6frames-2026-09-29"


# ---------------------------------------------------------------------------
# Prompt: turn ordered video frames and speech into searchable episodic facts.
# ---------------------------------------------------------------------------

def build_prompt(transcript: str, n_frames: int) -> str:
    ts = transcript.strip() if transcript and transcript.strip() else \
        "[No audio transcript - infer from visual only]"
    return f"""Create an accurate, searchable episodic memory from {n_frames} ordered first-person video frames and this transcript:

{ts}

Compare the frames in time order. Record what changes and who performs each visible action. Do not treat an object's presence as proof that someone used or moved it.

Return exactly these English fields, each on its own line with the exact field name and one concise value:
Scene: one sentence naming the main event, the key visible action sequence, and its outcome when supported; avoid a generic room or group description
Location: one broad place supported by the frames; do not guess an exact room
Time/Occasion: only supported time or occasion
People: names explicitly used as transcript speaker labels or clearly identified by name; never list unnamed people, clothing, or generic groups
User Action: concrete actions performed by the camera wearer only; never assign another person's action to the wearer; leave blank if no action is visible
Activity: the specific main activity and purpose, not a generic label such as "discussion"
Objects: at most five important tangible items clearly visible or explicitly named
Environment: one short physical setting supported by frames; omit guesses and background detail
Interaction Details: one concise line of semicolon-separated facts. Preserve meaningful spoken who-said-what claims, plans, questions and replies, decisions, transfers, named options, quantities, reasons, negation, preferences, and event order. Also record important visible non-speech actions: who did what to which object, in what order, and any transfer or state change. Name an actor only when identity is supported; otherwise say "unidentified person" or "camera wearer" as appropriate. Keep visual observations separate from speech facts. Do not merge nearby speakers' statements or quantities. Preserve each named item in an explicit list; do not replace a concrete list with a broad topic. If a speaker or detail is unclear, say unclear instead of guessing.

For first/last/sequence questions, preserve the order of relevant actions and the person-object relationship. Translate speech into natural English. Use a provided English translation where available; never copy source-language text. Do not infer actions from a person's presence or speech alone. Keep routine visual details concise. Target at most 260 words, but preserve a clearly enumerated list or action sequence even if it needs more space. Do not write "None" or "None specified"; leave unsupported fields blank. Add no extra text."""

_FIELD_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(Scene|Location|Time/Occasion|People|User Action|"
    r"Activity|Objects|Environment|Interaction Details)\s*:\s*(.*)$",
    re.I,
)


def _field(text: str, label: str) -> str:
    """Extract a field value, including continuation and bullet lines."""
    target = label.casefold()
    active = False
    parts = []
    for line in str(text or "").splitlines():
        match = _FIELD_LINE_RE.match(line)
        if match:
            if active:
                break
            if match.group(1).casefold() == target:
                active = True
                value = match.group(2).strip()
                if value:
                    parts.append(value)
            continue
        if active:
            value = line.strip()
            if value.startswith(("- ", "* ")):
                value = value[2:].strip()
            if value:
                parts.append(value)
    return " ".join(parts)


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

_TRANSCRIPT_SPEAKERS = (
    "Camera wearer", "Speaker 1", "Speaker 2", "Consumer",
    "Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure", "Unknown", "Me",
    "Shure's GF", "Shuna", "Nina", "Violet", "Jack", "Nicous", "Sister", "Others",
)
_TRANSCRIPT_SPEAKER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" +
    "|".join(re.escape(x) for x in sorted(_TRANSCRIPT_SPEAKERS, key=len, reverse=True)) +
    r")\s*:",
    re.I,
)


def extract_english_transcript(transcript: str, max_chars: int = 4000) -> str:
    """Keep speaker-labelled English source evidence without copying CJK text."""
    raw = str(transcript or "").strip()
    if not raw:
        return ""
    speakers = list(_TRANSCRIPT_SPEAKER_RE.finditer(raw))
    if not speakers:
        return ""
    facts = []
    previous_signature = None
    for index, match in enumerate(speakers):
        end = speakers[index + 1].start() if index + 1 < len(speakers) else len(raw)
        segment = re.sub(r"\s+", " ", raw[match.end():end]).strip(" ,;")
        if not segment or _contains_cjk(segment):
            continue
        signature = (match.group(1).casefold(),
                     re.sub(r"[^a-z0-9]+", "", segment.casefold()))
        if signature == previous_signature:
            continue
        previous_signature = signature
        fact = f"{match.group(1)}: {segment}"
        proposed = " ; ".join(facts + [fact])
        if len(proposed) > max_chars:
            break
        facts.append(fact)
    return " ; ".join(facts)


def parse_analysis(text: str, transcript: str = "") -> dict:
    """Qwen 输出 → 结构化记忆条目（对齐 memory.add 的参数）。"""
    location = _field(text, "Location")
    people = _split_list(_field(text, "People"), person=True)
    action = _field(text, "User Action")
    for marker in ("none specified", "none", "not specified", "n/a", "unknown"):
        if action.casefold().startswith(marker):
            action = action[len(marker):].lstrip(" .,:;")
            break
    prefix = "interaction details:"
    if action.casefold().startswith(prefix):
        action = action[len(prefix):].lstrip()
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
        "source_evidence": extract_english_transcript(transcript),
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

    # Video intervals are end-exclusive. Sampling exactly at end often asks
    # OpenCV for a frame past EOF, silently dropping the final (often decisive) frame.
    last_valid_time = max(start, end - 1.0 / fps)
    times = [start + (last_valid_time - start) * i / max(n - 1, 1)
             for i in range(n)] if n > 1 else [start]
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
            cached_fields = dict(hit.get("fields") or {})
            if not cached_fields.get("source_evidence"):
                cached_fields["source_evidence"] = extract_english_transcript(
                    m.get("transcript", ""))
            hit["fields"] = cached_fields
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
            "fields": parse_analysis(raw, m.get("transcript", "")),
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
