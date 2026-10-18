"""Explicit spoken memory facts stay source-linked and never infer identities."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "code"))

from interactive_runner import _extract_explicit_speech_facts
from memory import PersonMemory, _clean_source_fact_rows
from nighttime_processor import NightlyProcessor


class ExplicitSpeechFactTests(unittest.TestCase):
    def extract(self, transcript):
        return _extract_explicit_speech_facts(transcript, "synthetic-clip")

    def test_named_and_relationship_referents_keep_explicit_age_plan_and_health(self):
        facts = self.extract(
            "[audio 1.00–1.80 s; English] Jake is 42 years old.\n"
            "[audio 2.00–3.50 s; English] My mother has diabetes.\n"
            "[audio 4.00–5.50 s; English] She is planning to visit next week."
        )
        ages = [fact for fact in facts if fact["predicate"] == "stated_age"]
        plans = [fact for fact in facts if fact["predicate"] == "stated_plan"]
        health = [fact for fact in facts if fact["predicate"] == "stated_health_history"]
        self.assertEqual((ages[0]["subject"], ages[0]["value"]), ("Jake", "42 years old"))
        self.assertEqual(health[0]["subject"], "My mother")
        self.assertEqual(plans[0]["subject"], "unidentified person referred to as she")
        self.assertEqual(plans[0]["time_scope"], "next week")
        self.assertEqual(plans[0]["evidence_ref"]["audio_start_s"], 4.0)
        self.assertEqual(plans[0]["evidence_ref"]["source_segment_id"], "synthetic-clip")

    def test_chinese_age_plan_and_health_are_explicit_and_timed(self):
        facts = self.extract(
            "[audio 1.00–2.00 s; Chinese] 我妈妈今年67岁。\n"
            "[audio 3.00–4.20 s; Chinese] 张鸿兴这几天打算去医院。\n"
            "[audio 5.00–6.20 s; Chinese] 他有高血压。"
        )
        ages = [fact for fact in facts if fact["predicate"] == "stated_age"]
        plans = [fact for fact in facts if fact["predicate"] == "stated_plan"]
        health = [fact for fact in facts if fact["predicate"] == "stated_health_history"]
        self.assertEqual((ages[0]["subject"], ages[0]["value"]), ("我妈妈", "67岁"))
        self.assertEqual(plans[0]["subject"], "张鸿兴")
        self.assertEqual(plans[0]["time_scope"], "这几天")
        self.assertEqual(health[0]["subject"], "unidentified person referred to as 他")

    def test_first_person_and_uncertain_or_negated_claims_are_not_misattributed(self):
        facts = self.extract(
            "[audio 1.00–2.00 s; English] I am 33 years old.\n"
            "[audio 3.00–4.00 s; English] I'm going to see the doctor tomorrow.\n"
            "[audio 5.00–6.00 s; English] Jake might visit tomorrow.\n"
            "[audio 7.00–8.00 s; English] Jake has no diabetes.\\n"
            "[audio 9.00–10.00 s; English] I have no diabetes.\\n"
            "[audio 11.00–12.00 s; English] Jake will not travel tomorrow.\\n"
            "[audio 13.00–14.00 s; English] I will not travel tomorrow."
        )
        ages = [fact for fact in facts if fact["predicate"] == "stated_age"]
        plans = [fact for fact in facts if fact["predicate"] == "stated_plan"]
        health = [fact for fact in facts if fact["predicate"] == "stated_health_history"]
        self.assertEqual(ages[0]["subject"], "unidentified speaker")
        self.assertEqual(plans[0]["subject"], "unidentified speaker")
        self.assertEqual(plans[0]["time_scope"], "tomorrow")
        self.assertEqual(len(plans), 1)
        self.assertEqual(health, [])
        self.assertEqual(len(plans), 1)

    def test_chinese_negated_health_and_plans_are_not_recorded(self):
        facts = self.extract(
            "[audio 1.00–2.00 s; Chinese] 他没有高血压。\\n"
            "[audio 3.00–4.00 s; Chinese] 她对花生不过敏。\\n"
            "[audio 5.00–6.00 s; Chinese] 张鸿兴不打算出门。"
        )
        self.assertEqual(facts, [])

    def test_plan_round_trips_through_search_and_l4(self):
        fact = self.extract("[audio 1.00–2.00 s; English] Jake plans to visit next week.")[0]
        with tempfile.TemporaryDirectory(prefix="speech-plan-memory-") as directory:
            memory = PersonMemory(memory_dir=directory)
            moment_id = memory.add(
                scene="A conversation mentions Jake's next-week visit.",
                user_action="", needs=[], solutions=[], people=["Jake"],
                temporal_facts=[fact],
                source_time={"timestamp": "2026-10-17T10:00:00", "basis": "recorded"},
                source_provenance={"source_video_id": "synthetic.mp4",
                                   "source_segment_id": "synthetic-clip"},
            )
            restored = PersonMemory(memory_dir=directory)
            hits = restored.query_hierarchical("Jake visit next week", top_k=3)
            hit = next(row for row in hits if row["moment"]["id"] == moment_id)
            persisted = hit["moment"]["temporal_facts"][0]
            self.assertEqual(persisted["time_scope"], "next week")
            self.assertEqual(persisted["evidence_ref"]["modality"], "speech")
            l4_plans = NightlyProcessor._stated_plans([hit["moment"]])
            self.assertEqual(l4_plans[0]["subject"], "Jake")
            self.assertEqual(l4_plans[0]["time_scope"], "next week")
            self.assertEqual(l4_plans[0]["evidence_ref"]["source_segment_id"],
                             "synthetic-clip")

    def test_time_scope_survives_memory_fact_validation(self):
        fact = self.extract("[audio 1.00–2.00 s; Chinese] 她下周准备出门。")[0]
        cleaned = _clean_source_fact_rows([fact], "fact", 16)
        self.assertEqual(cleaned[0]["time_scope"], "下周")
        self.assertEqual(cleaned[0]["evidence_ref"]["modality"], "speech")


if __name__ == "__main__":
    unittest.main(verbosity=2)
