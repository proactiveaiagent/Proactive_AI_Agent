import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import wait_then_prewarm_v72 as waiter


class WaiterCoordinationTests(unittest.TestCase):
    def test_detects_in_process_web_prewarm(self):
        payload = {"running": True, "stage_label": "正在请求 Qwen3-VL"}
        response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch.object(waiter.urllib.request, "urlopen", return_value=response):
            self.assertEqual(
                waiter.active_ui_prewarms(),
                [("viz-server", "正在请求 Qwen3-VL")],
            )

    def test_ignores_its_own_queue_status(self):
        payload = {
            "running": True, "pid": 12345, "stage": "reconciling",
            "stage_label": "核对剩余缓存",
        }
        response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch.object(waiter, "SELF", 12345), patch.object(
                waiter.urllib.request, "urlopen", return_value=response):
            self.assertEqual(waiter.active_ui_prewarms(), [])

    def test_other_queue_status_still_blocks(self):
        payload = {
            "running": True, "pid": 23456, "stage": "reconciling",
            "stage_label": "核对剩余缓存",
        }
        response = io.BytesIO(json.dumps(payload).encode("utf-8"))
        with patch.object(waiter, "SELF", 12345), patch.object(
                waiter.urllib.request, "urlopen", return_value=response):
            self.assertEqual(
                waiter.active_ui_prewarms(), [("viz-server", "核对剩余缓存")]
            )

    def test_idle_web_service_does_not_block_queue(self):
        response = io.BytesIO(b'{"running": false}')
        with patch.object(waiter.urllib.request, "urlopen", return_value=response):
            self.assertEqual(waiter.active_ui_prewarms(), [])

    def test_unreachable_web_service_does_not_block_queue(self):
        with patch.object(waiter.urllib.request, "urlopen", side_effect=OSError("offline")):
            self.assertEqual(waiter.active_ui_prewarms(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
