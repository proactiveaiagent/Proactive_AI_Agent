# -*- coding: utf-8 -*-
"""Video analyzer: extract key frames, deduplicate, call GUI-Owl API, export reports."""
import json
import os
import time
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from config import (
    ANALYSIS_OUTPUT_DIR,
    FRAME_SAMPLE_INTERVAL_SEC,
    SSIM_THRESHOLD,
)
from gui_agent_api import analyze_screenshot


def _compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Simplified SSIM between two grayscale images (no skimage dependency)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


def _to_gray_small(frame: np.ndarray, width: int = 320) -> np.ndarray:
    """Convert frame to small grayscale for SSIM comparison."""
    h, w = frame.shape[:2]
    scale = width / w
    small = cv2.resize(frame, (width, int(h * scale)))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def extract_keyframes(
    video_path: str,
    output_dir: str,
    interval_sec: float = FRAME_SAMPLE_INTERVAL_SEC,
    ssim_threshold: float = SSIM_THRESHOLD,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """Extract key frames from video, deduplicate by SSIM.

    Returns list of dicts: {"frame_index": int, "timestamp_sec": float, "path": str}
    progress_cb(current_step, total_steps, message) is called for UI updates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    frame_interval = int(fps * interval_sec)

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    keyframes = []
    prev_gray = None
    frame_idx = 0
    sample_count = 0
    total_samples = max(1, int(total_frames / frame_interval))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            gray = _to_gray_small(frame)

            is_duplicate = False
            if prev_gray is not None:
                ssim = _compute_ssim(prev_gray, gray)
                if ssim > ssim_threshold:
                    is_duplicate = True

            if not is_duplicate:
                ts_str = f"{int(timestamp)}s"
                filename = f"frame_{len(keyframes):04d}_{ts_str}.png"
                filepath = os.path.join(frames_dir, filename)
                cv2.imwrite(filepath, frame)
                keyframes.append({
                    "frame_index": len(keyframes),
                    "timestamp_sec": round(timestamp, 1),
                    "path": filepath,
                    "relative_path": f"frames/{filename}",
                })
                prev_gray = gray

            sample_count += 1
            if progress_cb:
                progress_cb(
                    sample_count,
                    total_samples,
                    f"Extracting frames: {sample_count}/{total_samples} "
                    f"({len(keyframes)} keyframes kept)",
                )

        frame_idx += 1

    cap.release()
    return keyframes


def analyze_video(
    video_path: str,
    api_key: str,
    backend: str = "local",
    app_base_dir: str = "",
    model: str = "",
    interval_sec: float = FRAME_SAMPLE_INTERVAL_SEC,
    ssim_threshold: float = SSIM_THRESHOLD,
    output_dir: str = "",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Full pipeline: extract frames -> analyze each -> export reports.

    Returns the path to the output directory.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if not output_dir:
        output_dir = os.path.join(ANALYSIS_OUTPUT_DIR, video_name)
    else:
        output_dir = os.path.join(output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    if progress_cb:
        progress_cb(0, 100, "Extracting key frames from video...")

    keyframes = extract_keyframes(
        video_path, output_dir, interval_sec, ssim_threshold,
        progress_cb=lambda cur, tot, msg: (
            progress_cb(cur, tot, f"[1/2] {msg}") if progress_cb else None
        ),
    )

    if not keyframes:
        raise RuntimeError("No key frames extracted from video")

    # Prepare local model only once before per-frame inference.
    if (backend or "").lower() == "local":
        from gui_agent_api import ensure_local_model_ready

        ensure_local_model_ready(
            base_dir=app_base_dir or os.getcwd(),
            model_id=model,
            progress_cb=lambda msg: (
                progress_cb(0, 100, f"[2/2] {msg}") if progress_cb else None
            ),
            allow_download=False,
        )

    total_steps = len(keyframes)
    results = []

    for i, kf in enumerate(keyframes):
        if progress_cb:
            progress_cb(
                i + 1,
                total_steps,
                f"[2/2] Analyzing frame {i + 1}/{total_steps} "
                f"(t={kf['timestamp_sec']:.1f}s)...",
            )

        analysis = analyze_screenshot(
            image_path=kf["path"],
            api_key=api_key,
            timestamp_sec=kf["timestamp_sec"],
            backend=backend,
            app_base_dir=app_base_dir,
            model=model,
            progress_cb=lambda msg: (
                progress_cb(i + 1, total_steps, f"[2/2] {msg}") if progress_cb else None
            ),
        )
        analysis["screenshot"] = kf["relative_path"]
        analysis["frame_index"] = kf["frame_index"]
        results.append(analysis)

        if i < total_steps - 1:
            time.sleep(0.5)

    report = {
        "video": os.path.basename(video_path),
        "analyzed_at": datetime.now().isoformat(),
        "total_frames_analyzed": len(results),
        "frames": results,
    }

    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(output_dir, "report.txt")
    _write_text_report(report, txt_path)

    if progress_cb:
        progress_cb(total_steps, total_steps, "Analysis complete!")

    return output_dir


def _write_text_report(report: dict, path: str):
    lines = [
        f"Video Analysis Report",
        f"=====================",
        f"Video: {report['video']}",
        f"Analyzed at: {report['analyzed_at']}",
        f"Total frames analyzed: {report['total_frames_analyzed']}",
        f"",
        f"{'=' * 60}",
    ]

    for frame in report["frames"]:
        ts = frame.get("timestamp_sec", 0)
        lines.append(f"")
        lines.append(f"--- Frame at {ts:.1f}s ---")
        lines.append(f"Screenshot: {frame.get('screenshot', '')}")

        if frame.get("error"):
            lines.append(f"ERROR: {frame['error']}")
            continue

        lines.append(f"App: {frame.get('app_name', 'N/A')}")
        lines.append(f"Page: {frame.get('page_name', 'N/A')}")
        lines.append(f"User Action: {frame.get('user_action', 'N/A')}")
        lines.append(f"Description: {frame.get('description', 'N/A')}")

        texts = frame.get("visible_text", [])
        if texts:
            lines.append(f"Visible Text: {', '.join(str(t) for t in texts)}")

        elements = frame.get("elements", [])
        if elements:
            lines.append(f"UI Elements ({len(elements)}):")
            for el in elements:
                etype = el.get("type", "?")
                label = el.get("label", "")
                desc = el.get("description", "")
                lines.append(f"  [{etype}] {label} - {desc}")

    lines.append(f"")
    lines.append(f"{'=' * 60}")
    lines.append(f"End of report.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
