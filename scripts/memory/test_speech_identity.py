"""Spoken relationship terms only become person identities when the clip singles
out exactly one visible candidate; otherwise they stay speech evidence."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "code").is_dir() else HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code"))

from speech_identity import (attach_spoken_identity, person_labels_from_observations,
                             terms_in_text)
from memory import _clean_source_fact_rows

REF = {"source_segment_id": "clip-1", "modality": "speech",
       "audio_start_s": 0.5, "audio_end_s": 2.0}


def speech_fact(term, quote):
    return {"subject_type": "person", "subject": term, "predicate": "spoken_family_reference",
            "value": quote, "modality": "speech", "status": "quoted_relationship_term",
            "confidence": "medium", "evidence": quote, "evidence_ref": dict(REF),
            "actor": "unidentified speaker", "affected_people": [term]}


class SpeechIdentityTests(unittest.TestCase):
    def test_term_vocabulary_covers_chinese_and_english_addresses(self):
        self.assertEqual(terms_in_text("这是姥姥,快跟姥姥说过年好。"), ["姥姥"])
        self.assertEqual(terms_in_text("谢谢妈妈,姥姥好。"), ["妈妈", "姥姥"])
        self.assertEqual(terms_in_text("My neighbour helped me carry it."), ["neighbor"])

    def test_unique_camera_facing_person_gains_evidence_backed_alias(self):
        rows = [{"entity": "headscarf wearer",
                 "appearance": "walking toward the camera in a snowy gate",
                 "face_features": "smiling at the camera"},
                {"entity": "delivery person", "appearance": "standing behind the gate"}]
        result = attach_spoken_identity(rows, [speech_fact("姥姥", "这是姥姥,快跟姥姥说过年好。")])
        self.assertEqual([item["alias"] for item in rows[0]["spoken_aliases"]], ["姥姥"])
        self.assertEqual(rows[0]["spoken_aliases"][0]["status"], "probable_visual_match")
        self.assertIn("这是姥姥", rows[0]["spoken_aliases"][0]["basis"])
        self.assertEqual(rows[0]["entity"], "headscarf wearer")
        self.assertNotIn("spoken_aliases", rows[1])
        self.assertEqual(result["linked"][0]["entity"], "headscarf wearer")

    def test_approached_person_is_the_greeted_one_when_two_face_the_camera(self):
        rows = [{"entity": "person in pink", "appearance": "smiling at the camera"},
                {"entity": "headscarf wearer", "appearance": "smiling at the camera"}]
        visual_facts = [{"actor": "person in pink", "subject": "person in pink",
                         "predicate": "approached", "value": "headscarf wearer",
                         "modality": "visual"}]
        attach_spoken_identity(rows, [speech_fact("姥姥", "姥姥好,给您拜年了。")],
                               visual_facts)
        self.assertEqual([row.get("spoken_aliases") for row in rows][1][0]["alias"], "姥姥")
        self.assertIsNone([row.get("spoken_aliases") for row in rows][0])

    def test_ambiguous_people_keep_the_term_unlinked(self):
        rows = [{"entity": "person in blue", "appearance": "smiling at the camera"},
                {"entity": "person in red", "appearance": "smiling at the camera"}]
        result = attach_spoken_identity(rows, [speech_fact("姥姥", "这是姥姥,快跟姥姥说过年好。")])
        self.assertNotIn("spoken_aliases", rows[0])
        self.assertNotIn("spoken_aliases", rows[1])
        self.assertEqual(result["linked"], [])
        self.assertEqual([item["term"] for item in result["unlinked"]], ["姥姥"])

    def test_self_identified_speaker_never_binds_to_a_second_visible_person(self):
        rows = [{"entity": "person in blue", "appearance": "sitting"},
                {"entity": "person in red", "appearance": "standing"}]
        result = attach_spoken_identity(
            rows, [speech_fact("邻居", "我是邻居,来帮你搬东西。")])
        self.assertNotIn("spoken_aliases", rows[0])
        self.assertNotIn("spoken_aliases", rows[1])
        self.assertEqual(result["linked"], [])

    def test_self_identified_speaker_is_not_assigned_to_the_only_visible_person(self):
        rows = [{"entity": "person in blue", "appearance": "standing"}]
        result = attach_spoken_identity(
            rows, [speech_fact("邻居", "我是邻居,来帮你搬东西。")])
        self.assertNotIn("spoken_aliases", rows[0])
        self.assertEqual(result["linked"], [])
        self.assertIn("身份保持未知", result["unlinked"][0]["reason"])

    def test_event_people_are_filled_from_observations_using_supported_aliases(self):
        rows = [
            {"entity": "headscarf wearer", "appearance": "wearing a gray headscarf"},
            {"entity": "person in pink", "appearance": "wearing a pink beanie"},
        ]
        visual_facts = [{"actor": "person in pink", "subject": "person in pink",
                         "predicate": "approached", "value": "headscarf wearer",
                         "modality": "visual"}]
        attach_spoken_identity(
            rows, [speech_fact("姥姥", "这是姥姥,快跟姥姥说过年好。")], visual_facts)
        self.assertEqual(
            person_labels_from_observations([], rows), ["姥姥", "person in pink"])
        self.assertEqual(rows[0]["entity"], "headscarf wearer")

    def test_unlinked_or_unproven_speech_does_not_rename_event_people(self):
        rows = [{"entity": "person in blue", "appearance": "standing"}]
        attach_spoken_identity(rows, [speech_fact("邻居", "我是邻居。")])
        self.assertEqual(
            person_labels_from_observations(["person in blue"], rows),
            ["person in blue"])

    def test_english_vocative_is_case_insensitive(self):
        from speech_identity import _usage_of
        self.assertEqual(_usage_of("dad", "Dad, you are too easy to trust others."),
                         "address")

    def test_written_aliases_survive_the_persistence_sanitizer(self):
        rows = [{"entity": "headscarf wearer", "appearance": "walking toward the camera"}]
        attach_spoken_identity(rows, [speech_fact("姥姥", "这是姥姥,快跟姥姥说过年好。")])
        kept = _clean_source_fact_rows(rows, "person", 8)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["spoken_aliases"][0]["alias"], "姥姥")
        self.assertEqual(kept[0]["spoken_aliases"][0]["evidence_ref"]["audio_start_s"], 0.5)
        self.assertTrue(kept[0]["relationships"][0].startswith("语音中被介绍为姥姥"))

    def test_relationships_record_the_quote_for_later_verification(self):
        rows = [{"entity": "person in gray", "appearance": "facing the camera and smiling"}]
        attach_spoken_identity(rows, [speech_fact("妈妈", "谢谢妈妈。")])
        self.assertIn("谢谢妈妈", rows[0]["relationships"][0])

    def test_one_visible_person_cannot_answer_to_several_spoken_terms(self):
        rows = [{"entity": "person in pink", "appearance": "smiling at the camera"},
                {"entity": "headscarf wearer", "appearance": "smiling at the camera"}]
        visual_facts = [{"actor": "person in pink", "subject": "person in pink",
                         "predicate": "approached", "value": "headscarf wearer",
                         "modality": "visual"}]
        facts = [speech_fact("姥姥", "这是姥姥,快跟姥姥说过年好。"),
                 speech_fact("妈妈", "谢谢妈妈,姥姥好,给您拜年了。"),
                 speech_fact("孙子", "谢谢孙子。")]
        for fact, start in zip(facts, (0.0, 5.0, 11.0)):
            fact["evidence_ref"]["audio_start_s"] = start
            fact["evidence_ref"]["audio_end_s"] = start + 4.0
        result = attach_spoken_identity(rows, facts, visual_facts)
        self.assertEqual([item["alias"] for item in rows[1]["spoken_aliases"]], ["姥姥"])
        self.assertNotIn("spoken_aliases", rows[0])
        self.assertEqual([item["term"] for item in result["unlinked"]], ["妈妈", "孙子"])
        self.assertTrue(all("已按更早的语音线索关联为姥姥" in item["reason"]
                            for item in result["unlinked"]))

    def test_placeholder_speakers_do_not_become_person_cards(self):
        from entity_records import _speech_placeholder
        self.assertTrue(_speech_placeholder("unidentified speaker"))
        self.assertTrue(_speech_placeholder("unidentified person referred to as 他"))
        self.assertFalse(_speech_placeholder("camera wearer"))
        self.assertFalse(_speech_placeholder("unidentified person 1"))
        self.assertFalse(_speech_placeholder("姥姥"))


if __name__ == "__main__":
    unittest.main()
