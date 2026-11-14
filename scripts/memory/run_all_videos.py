"""
run_all_videos.py — Step 09 全量测试视频端到端回归（DDL 2 收尾）
==================================================================
验证对象：
  - 全量 test_data 视频逐个跑通 agent.py 主流程（抽帧 + 转写 + Phase 0/A/B/C）
  - 确认 05/06/07 的分层存储、画像 API、分层检索改动未破坏端到端
  - 连续输入下 Phase C 周期触发、layer6 画像持续精化、无异常/崩溃

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python run_all_videos.py [--limit N]

说明：
  - 运行前备份 code/memory/memory.json，冷启动后连续跑，跑完恢复原记忆文件。
  - 每跑完一个视频打印一条进度 + 当前画像生成状态。
  - 汇总结果写到 scripts/memory/e2e_regression_0911.json。
  - 跑完的累积记忆自动存档为 scripts/memory/archive/memory_<时间戳>.json（对比不同轮次用）。
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
MEMORY_DIR = CODE_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / "memory.json"
TEST_DATA = PROJECT_ROOT / "test_data" / "test_data"

# 关键：agent.py 内部 PersonMemory 用相对路径 "memory"（相对 cwd），
# 必须切到 code/ 目录跑，否则记忆会写到 scripts/memory/ 而不是 code/memory/，
# 导致脚本的备份/恢复/存档路径与实际写入位置不一致。
os.chdir(CODE_DIR)

sys.path.insert(0, str(CODE_DIR))

from agent import VRAssistant  # noqa: E402

BACKUP_FILE = MEMORY_FILE.with_name("memory.json.bak_0911")


def collect_videos(limit: int = None):
    vids = sorted(
        p for p in TEST_DATA.rglob("*")
        if p.suffix.lower() in (".mp4", ".mov")
    )
    if limit:
        vids = vids[:limit]
    return vids


def profile_nonempty(assistant) -> bool:
    """判断 layer6 画像是否已有任意稳定字段值。"""
    p = assistant.memory.get_profile()
    for field, val in p.items():
        if isinstance(val, dict):
            if any(v for v in val.values()):
                return True
        elif isinstance(val, list) and val:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个视频（冒烟用）")
    args = ap.parse_args()

    videos = collect_videos(args.limit)
    print(f"共 {len(videos)} 个测试视频")

    # 1. 备份当前记忆文件（若存在）
    had_backup = False
    if MEMORY_FILE.exists():
        shutil.copy2(MEMORY_FILE, BACKUP_FILE)
        had_backup = True
        print(f"已备份记忆文件 → {BACKUP_FILE.name}")

    # 2. 冷启动（删掉记忆文件，PersonMemory 会重建空库）
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
        print("已冷启动（清空记忆）")

    results = []
    try:
        for i, v in enumerate(videos, 1):
            rel = v.relative_to(PROJECT_ROOT)
            t0 = time.time()
            entry = {"video": str(rel), "ok": False}
            try:
                a = VRAssistant(str(v), input_source="camera")
                a.process(consolidation_blocking=True)
                total = a.memory.memory["metadata"]["total_moments"]
                last_cons = a.memory.memory["metadata"].get("last_consolidation")
                has_profile = profile_nonempty(a)
                entry.update({
                    "ok": True,
                    "elapsed_s": round(time.time() - t0, 1),
                    "total_moments": total,
                    "consolidated": last_cons is not None,
                    "has_profile": has_profile,
                })
                print(f"[{i}/{len(videos)}] ✅ {rel}  "
                      f"{entry['elapsed_s']}s  moments={total}  "
                      f"consolidated={entry['consolidated']}  profile={has_profile}")
            except Exception as e:  # noqa: BLE001
                entry["error"] = repr(e)
                results.append(entry)
                print(f"[{i}/{len(videos)}] ❌ {rel}  异常: {e!r}")
                continue
            results.append(entry)
    finally:
        # 3. 先把本次累积的记忆存档为带时间戳的快照（便于对比不同轮次）
        if MEMORY_FILE.exists():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_dir = Path(__file__).resolve().parent / "archive"
            snapshot_dir.mkdir(exist_ok=True)
            snapshot = snapshot_dir / f"memory_{stamp}.json"
            shutil.copy2(MEMORY_FILE, snapshot)
            print(f"已存档本次回归记忆 → {snapshot}")

        # 4. 恢复原记忆文件
        if had_backup:
            if MEMORY_FILE.exists():
                MEMORY_FILE.unlink()
            shutil.copy2(BACKUP_FILE, MEMORY_FILE)
            BACKUP_FILE.unlink()
            print("已恢复原记忆文件")
        else:
            # 本来就没有记忆文件，跑完清理掉测试产生的
            if MEMORY_FILE.exists():
                MEMORY_FILE.unlink()
            print("已清理测试产生的记忆文件")

    # 4. 汇总
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print("\n" + "=" * 60)
    print(f"端到端回归汇总：{len(ok)}/{len(results)} 通过，{len(fail)} 失败")
    if fail:
        for r in fail:
            print(f"  ❌ {r['video']}: {r.get('error')}")
    print("=" * 60)

    out = Path(__file__).resolve().parent / "e2e_regression_0911.json"
    out.write_text(json.dumps({
        "total": len(results),
        "passed": len(ok),
        "failed": len(fail),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"汇总已写入 → {out.name}")

    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
