import unittest

from prewarm_status import parse_multimodal_prewarm_log


class MultimodalPrewarmStatusTests(unittest.TestCase):
    def test_reports_current_grounding_progress(self):
        status = parse_multimodal_prewarm_log(
            "[2026-10-17 21:28:30] object-frame cache 450/2793: "
            "hit 355 (source fast-path 12), new 93, failed 2",
            running=True, pid=139177, log_path="/tmp/prewarm.log",
        )
        self.assertTrue(status["running"])
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["stage"], "grounding")
        self.assertEqual(status["done"], 450)
        self.assertEqual(status["total"], 2793)
        self.assertEqual(status["grounding"]["hit"], 355)
        self.assertEqual(status["grounding"]["source_hit"], 12)
        self.assertEqual(status["grounding"]["new"], 93)
        self.assertEqual(status["grounding"]["failed"], 2)

    def test_reports_all_three_cache_stages_and_completion(self):
        status = parse_multimodal_prewarm_log(
            "[2026-10-17 20:00:00] perception quality pass 100/100: 3 repaired, 1 failed\n"
            "[2026-10-17 20:01:00] object-frame cache 100/100: hit 60, new 39, failed 1\n"
            "[2026-10-17 20:02:00] audio transcript cache 100/100: "
            "hit 80, new 10, failed 0, silent 10\n"
            "[2026-10-17 20:03:00] all selectable profiles multimodal cache pass complete: "
            "{'perception_repaired': 3, 'grounding_hit': 60, 'audio_new': 10}",
            running=False,
        )
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["stage"], "complete")
        self.assertEqual(status["perception"]["done"], 100)
        self.assertIsNone(status["perception"]["cache_hits"])
        self.assertEqual(status["perception"]["unknown_success"], 96)
        self.assertEqual(status["grounding"]["hit"], 60)
        self.assertEqual(status["audio"]["silent"], 10)
        self.assertEqual(status["final_totals"]["audio_new"], 10)

    def test_reports_perception_hits_updates_and_new_results_separately(self):
        status = parse_multimodal_prewarm_log(
            "[2026-10-17 23:40:00] perception cache pass 100/2793: "
            "hit 24, refreshed 70, new 2, repaired 3, failed 1",
            running=True,
        )
        self.assertEqual(status["stage"], "perception")
        self.assertEqual(status["perception"]["cache_hits"], 24)
        self.assertEqual(status["perception"]["refreshed"], 70)
        self.assertEqual(status["perception"]["new"], 2)
        self.assertEqual(status["perception"]["repaired"], 3)
        self.assertEqual(status["perception"]["failed"], 1)

    def test_old_perception_log_does_not_mislabel_success_as_cache_hit(self):
        status = parse_multimodal_prewarm_log(
            "[2026-10-17 23:30:19] perception quality pass 50/2793: 0 repaired, 3 failed",
            running=True,
        )
        self.assertIsNone(status["cache_hits"])
        self.assertEqual(status["perception"]["unknown_success"], 47)
        self.assertEqual(status["perception"]["failed"], 3)

    def test_marks_an_exited_incomplete_run_as_stopped(self):
        status = parse_multimodal_prewarm_log(
            "[2026-10-17 21:28:30] object-frame cache 450/2793: "
            "hit 355, new 93, failed 2",
            running=False,
        )
        self.assertEqual(status["status"], "stopped")
        self.assertFalse(status["running"])
        self.assertIn("没有完成标记", status["error"])


if __name__ == "__main__":
    unittest.main()
