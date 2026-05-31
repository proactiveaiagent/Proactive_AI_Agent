# -*- coding: utf-8 -*-
"""屏幕录制模块 - 使用 mss 截屏 + OpenCV 写入视频"""
import threading
import time
import os
from datetime import datetime

try:
    import mss
    import cv2
    import numpy as np
except ImportError:
    print("请先安装依赖: pip install -r requirements.txt")
    raise


class ScreenRecorder:
    def __init__(self, fps=15, output_dir="recordings"):
        self.fps = fps
        self.output_dir = output_dir
        self._running = False
        self._thread = None
        self._writer = None
        self._output_path = None
        os.makedirs(output_dir, exist_ok=True)

    def _generate_filename(self):
        return os.path.join(
            self.output_dir,
            f"pc_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )

    def start(self):
        if self._running:
            return False, "已在录制中"
        self._running = True
        self._output_path = self._generate_filename()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        return True, self._output_path

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._output_path

    def is_recording(self):
        return self._running

    def _record_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # 全屏
            width = monitor["width"]
            height = monitor["height"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._output_path, fourcc, self.fps, (width, height)
            )
            frame_interval = 1.0 / self.fps
            while self._running:
                t0 = time.perf_counter()
                img = sct.grab(monitor)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self._writer.write(frame)
                elapsed = time.perf_counter() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if self._writer:
                self._writer.release()
                self._writer = None
