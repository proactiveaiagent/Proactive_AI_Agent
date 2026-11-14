"""补齐短期场景缺失的视频（A4_LUCIA / A5_KATRINA / A6_SHURE 的 DAY1 片段）。"""
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
missing = []
for l in open(ROOT / "data/egolife/testset/short_term/moments_scattered.jsonl", encoding="utf-8"):
    if l.strip():
        v = json.loads(l)["video"]                       # data/EgoLife/A4_LUCIA/DAY1/xxx.mp4
        if not (ROOT / v.replace("EgoLife", "egolife")).exists():
            missing.append(v.replace("data/EgoLife/", ""))

print(f"缺失 {len(missing)} 个视频，开始下载…")
snapshot_download(
    "lmms-lab/EgoLife", repo_type="dataset",
    local_dir=str(ROOT / "data/egolife"),
    allow_patterns=missing,
    max_workers=8,
)
print("下载完成")
