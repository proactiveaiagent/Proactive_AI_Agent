"""Fill the persistent Qwen perception cache for every selectable dataset.

This is an operational helper for the visualization server's existing prewarm
API. It only reads video datasets and writes perception-cache entries; it does
not create or mutate memory sessions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8768"
STATUS = BASE + "/api/interactive/prewarm_status"
START = BASE + "/api/interactive/prewarm?"
PROFILES = [
    # Put the user's active evaluation set first. Long-term full-day datasets
    # contain thousands of clips and can otherwise keep this queue occupied
    # for hours before the 35 custom clips are reached.
    {"scenario": "custom_test", "dataset": "feishu:all", "label": "自定义·飞书全集"},
    {"scenario": "short_term", "dataset": "base", "label": "短期·标准集"},
    {"scenario": "short_term", "dataset": "expanded", "label": "短期·扩展集"},
    {"scenario": "long_term", "dataset": "long_day", "participant": "A1_JAKE", "label": "长期·Jake"},
    {"scenario": "long_term", "dataset": "long_day", "participant": "A2_ALICE", "label": "长期·Alice"},
    {"scenario": "long_term", "dataset": "long_day", "participant": "A3_TASHA", "label": "长期·Tasha"},
    {"scenario": "long_term", "dataset": "long_day", "participant": "A4_LUCIA", "label": "长期·Lucia"},
    {"scenario": "long_term", "dataset": "long_day", "participant": "A5_KATRINA", "label": "长期·Katrina"},
    {"scenario": "long_term", "dataset": "long_day", "participant": "A6_SHURE", "label": "长期·Shure"},
]


def get_json(url: str, timeout: int = 10) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def wait_for_server() -> dict:
    while True:
        try:
            health = get_json("http://127.0.0.1:8000/health", timeout=5)
            if health.get("model_loaded"):
                return health
            log("Qwen API 未就绪；60 秒后重试")
        except (URLError, TimeoutError, OSError) as exc:
            log(f"等待 Qwen API：{exc}")
        time.sleep(60)


def main() -> None:
    health = wait_for_server()
    gpu = health.get("gpu") or {}
    log(f"开始补齐感知缓存；模型已就绪，GPU={gpu.get('name')}，剩余显存={gpu.get('free_gb')} GB")
    totals = {"cache_hits": 0, "inferred": 0, "failed": 0}

    for profile in PROFILES:
        label = profile["label"]
        while True:
            try:
                status = get_json(STATUS)
                if not status.get("running"):
                    break
                log(f"等待当前预热任务结束：{status.get('scenario')} {status.get('done')}/{status.get('total')}")
            except (URLError, TimeoutError, OSError) as exc:
                log(f"网页服务暂不可达，60 秒后重试：{exc}")
                time.sleep(60)
                wait_for_server()
            time.sleep(15)

        params = {key: value for key, value in profile.items() if key != "label"}
        started = get_json(START + urlencode(params))
        if not started.get("started"):
            log(f"未能启动 {label}：{started}")
            continue
        log(f"开始 {label} 缓存检查")

        last_report = 0.0
        while True:
            time.sleep(10)
            try:
                status = get_json(STATUS)
            except (URLError, TimeoutError, OSError) as exc:
                log(f"读取进度失败，等待服务恢复：{exc}")
                wait_for_server()
                continue

            now = time.monotonic()
            if status.get("running") and now - last_report >= 60:
                log(
                    f"{label}：{status.get('done', 0)}/{status.get('total', 0)}；"
                    f"缓存命中 {status.get('cache_hits', 0)}，新推理 {status.get('inferred', 0)}，"
                    f"失败 {status.get('failed', 0)}"
                )
                last_report = now
            if not status.get("running"):
                for key in totals:
                    totals[key] += int(status.get(key, 0) or 0)
                log(
                    f"完成 {label}：{status.get('done', 0)}/{status.get('total', 0)}；"
                    f"缓存命中 {status.get('cache_hits', 0)}，新推理 {status.get('inferred', 0)}，"
                    f"失败 {status.get('failed', 0)}；错误={status.get('error')}"
                )
                break

    log(f"所有数据集处理结束：{totals}")
    if os.environ.get("PREWARM_CUSTOM_MULTIMODAL", "1") != "0":
        auxiliary = ROOT / "scripts" / "memory" / "prewarm_multimodal_evidence.py"
        log("开始补齐飞书测试视频的物品画面与原声转写缓存")
        try:
            result = subprocess.run([sys.executable, str(auxiliary), "--all-profiles"], cwd=ROOT,
                                    check=False)
            if result.returncode:
                log(f"多模态证据缓存任务结束，退出码={result.returncode}")
        except OSError as exc:
            log(f"无法启动多模态证据缓存任务：{exc}")


if __name__ == "__main__":
    main()
