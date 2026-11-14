"""
preprocess_egolife.py — EgoLife 预处理管线（09-21）
====================================================
把 EgoLife 原始数据解析并对齐，产出结构化 moment 数据（对齐 memory.add 的输入格式）。

关键发现（09-18）：EgoLifeCap 已内置 ASR 转写（Transcript）+ 密集 caption（DenseCaption），
无需重新抽帧 + ASR。本脚本复用现成文本做结构化对齐。

数据源（`data/egolife/`）：
  1. `EgoIT/EgoLife_Caption.json`     — 9002 条英文第一人称叙事（每个 30s 视频片段一条）
  2. `EgoLifeCap/Transcript/*.srt`    — 双语 ASR 转写（说话人 + 对话，带时间戳）
  3. `EgoLifeCap/DenseCaption/*.srt`  — 中文第一人称密集 caption（逐帧动作描述）

产出：`data/egolife/structured/moments.jsonl`（每行一个 moment，对齐 agent 输入）
  字段：id / video / scene / user_action / people / location / transcript / dense_caption /
        start_time / end_time

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python \
        scripts/memory/preprocess_egolife.py
"""

import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]          # Proactive_AI_Agent
EGOLIFE = ROOT / "data" / "egolife"
CAPTION_FILE = EGOLIFE / "EgoIT" / "EgoLife_Caption.json"
TRANSCRIPT_DIR = EGOLIFE / "EgoLifeCap" / "Transcript"
DENSECAP_DIR = EGOLIFE / "EgoLifeCap" / "DenseCaption"
OUT_DIR = EGOLIFE / "structured"
OUT_FILE = OUT_DIR / "moments.jsonl"

# 室内地点词典（EgoLife 为 6 人共居居家场景，规则提取 location 用）
LOCATION_VOCAB = [
    "living room", "kitchen", "bedroom", "bathroom", "dining room",
    "dining table", "table", "staircase", "stairs", "hallway", "corridor",
    "balcony", "garden", "yard", "fridge", "refrigerator", "window", "sofa",
    "couch", "door", "entrance", "office", "meeting room", "counter",
    "sink", "countertop", "closet", "garage", "basement",
]

SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


# ---------------------------------------------------------------------------
# 解析函数
# ---------------------------------------------------------------------------

def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path):
    """解析 .srt → list[(start_sec, end_sec, text)]。text 为该字幕块的文本（多行拼接）。"""
    if not path.exists():
        return []
    blocks = []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # 按空行切分字幕块
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        if not lines:
            continue
        m = SRT_TIME_RE.search(chunk)
        if not m:
            continue
        start = _ts_to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
        # 时间戳行之后的所有行是文本
        text_lines = [ln for ln in lines if " --> " not in ln and not ln.isdigit()]
        text = " ".join(t.strip() for t in text_lines if t.strip())
        if text:
            blocks.append((start, end, text))
    return blocks


def load_captions(path: Path) -> list:
    """加载 EgoIT Caption json → list[{id, video, caption}]。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in data:
        conv = item.get("conversations") or []
        caption = ""
        for c in conv:
            if c.get("from") == "gpt":
                caption = c.get("value", "")
                break
        out.append({
            "id": item.get("id", ""),
            "video": item.get("video", ""),
            "caption": caption.strip(),
        })
    return out


# ---------------------------------------------------------------------------
# 字段提取
# ---------------------------------------------------------------------------

KNOWN_PEOPLE = ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"]
NOISE_NAMES = {
    "speaker", "boss", "gf", "sister", "brother", "mom", "dad", "mother",
    "father", "guy", "girl", "man", "woman", "friend", "colleague", "everyone",
    "everybody", "okay", "yes", "no", "so", "then", "right", "come", "mark",
    "look", "here", "hey", "oh", "well", "now", "let",
}


def normalize_name(name: str) -> str:
    """归一化说话人：若含已知参与者名（如 'pJake' → 'Jake'），归一到标准名。"""
    for k in KNOWN_PEOPLE:
        if k.lower() in name.lower():
            return k
    return name


def extract_people(transcript_lines):
    """从 Transcript 提取说话人（英文名归一化 + 去重 + 噪声过滤）。"""
    people = []
    seen = set()
    for _, _, text in transcript_lines:
        # 每段文本形如 "Jake: 中文 Jake: English"，提取 "Name:" 里的英文名
        for m in re.finditer(r"([A-Za-z]+):\s*", text):
            name = normalize_name(m.group(1))
            if name.lower() in NOISE_NAMES:
                continue
            if name not in seen:
                seen.add(name)
                people.append(name)
    return people


def extract_location(*texts) -> str:
    """从文本里匹配室内地点词典，返回命中地点（多个用 / 连接）。"""
    found = []
    joined = " ".join(t for t in texts if t).lower()
    for loc in LOCATION_VOCAB:
        if loc in joined and loc not in found:
            found.append(loc)
    return "/".join(found) if found else ""


def srt_stem_and_offset(video: str):
    """从 EgoIT video 路径解析 (srt_stem, offset_sec)。

    EgoIT video 是 30 秒片段（DAY1_A6_SHURE_14003000.mp4），srt 文件按整点组织
    （A6_SHURE_DAY1_14000000.srt 覆盖 14:00~15:00）。offset 是片段相对整点的秒数。
    """
    parts = video.split("/")
    person = parts[2]      # A6_SHURE
    day = parts[3]         # DAY1
    filename = parts[4]    # DAY1_A6_SHURE_14003000.mp4
    ts = filename[:-4].split("_")[-1]   # 14003000
    hh = int(ts[0:2])
    mm = int(ts[2:4])
    ss = int(ts[4:6])
    offset = mm * 60 + ss              # 相对整点的秒数（如 30）
    stem = f"{person}_{day}_{hh:02d}000000"
    return stem, offset


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    captions = load_captions(CAPTION_FILE)
    print(f"EgoIT caption 条数: {len(captions)}")

    stats = Counter()
    written = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for cap in captions:
            stem, offset = srt_stem_and_offset(cap["video"])
            person = cap["video"].split("/")[2]
            day = cap["video"].split("/")[3]
            transcript_path = TRANSCRIPT_DIR / person / day / f"{stem}.srt"
            densecap_path = DENSECAP_DIR / person / day / f"{stem}.srt"

            all_transcript = parse_srt(transcript_path)
            all_densecap = parse_srt(densecap_path)

            # 30 秒窗口：只取该片段 [offset, offset+30) 时间范围内的字幕块
            win_start, win_end = offset, offset + 30
            transcript_lines = [(s, e, t) for (s, e, t) in all_transcript
                                if s >= win_start and s < win_end]
            densecap_lines = [(s, e, t) for (s, e, t) in all_densecap
                              if s >= win_start and s < win_end]

            people = extract_people(transcript_lines)
            transcript_text = " ".join(t for _, _, t in transcript_lines)
            dense_caption = " ".join(t for _, _, t in densecap_lines)
            location = extract_location(cap["caption"], transcript_text, dense_caption)

            start_time = densecap_lines[0][0] if densecap_lines else float(win_start)
            end_time = densecap_lines[-1][1] if densecap_lines else float(win_end)

            moment = {
                "id": cap["id"],
                "video": cap["video"],
                "scene": cap["caption"],          # 英文第一人称叙事 → 场景上下文
                "user_action": dense_caption,     # 中文密集 caption → 动作（30 秒窗口）
                "people": people,                 # 说话人
                "location": location,             # 规则提取地点
                "transcript": transcript_text,    # 对话转写（30 秒窗口）
                "dense_caption": dense_caption,   # 原始密集 caption（30 秒窗口）
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
            }
            f.write(json.dumps(moment, ensure_ascii=False) + "\n")
            written += 1
            stats["total"] += 1
            if not transcript_lines:
                stats["no_transcript"] += 1
            if not densecap_lines:
                stats["no_densecap"] += 1
            if people:
                stats["with_people"] += 1
            if location:
                stats["with_location"] += 1

    print(f"\n产出: {OUT_FILE}")
    print(f"共写出 {written} 条 moment")
    print(f"统计: 无 transcript={stats['no_transcript']}, "
          f"无 densecap={stats['no_densecap']}, "
          f"含 people={stats['with_people']}, 含 location={stats['with_location']}")

    # 打印样例
    with OUT_FILE.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("\n样例 moment:")
    for k in ("id", "scene", "user_action", "people", "location"):
        v = first[k]
        if isinstance(v, str) and len(v) > 120:
            v = v[:120] + "..."
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
