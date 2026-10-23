"""CPU-only regression tests using synthetic records, never evaluation evidence."""
import copy
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from entity_records import entity_records, memory_facts


def event(mid, video='clip-a'):
    return {'id': mid, 'scene': 'Synthetic test event', 'people': ['Alice'],
            'objects': ['phone'], 'location': 'room', 'timestamp': '2026-10-15T12:00:00',
            'source_provenance': {'source_video_id': video, 'source_segment_id': mid},
            'source_time': {'clip_start_s': 1, 'basis': 'video_relative'},
            'person_observations': [{'entity': 'unidentified person 1', 'appearance': 'blue shirt',
                                     'age': '30', 'age_basis': 'unknown'}],
            'temporal_facts': [{'subject_type': 'object', 'subject': 'phone', 'predicate': 'moved',
                                'actor': 'Alice', 'affected_people': ['Bob'], 'from_location': 'desk',
                                'to_location': 'bag', 'status': 'completed', 'evidence': 'visible transfer'},
                               {'subject_type': 'person', 'subject': 'Alice', 'predicate': 'plan',
                                'value': 'visit tomorrow', 'status': 'stated_plan', 'evidence': 'spoken plan'}]}


class EntityRecordTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = event('a'), event('b', 'clip-b')
        self.archive = {'moments': {'a': self.a, 'b': self.b}}

    def test_named_person_keeps_events_plans_and_changes(self):
        data = entity_records(self.archive, 'person', 'Alice')
        alice = next(x for x in data['items'] if x['name'] == 'Alice')
        self.assertEqual(alice['event_count'], 2)
        self.assertEqual(alice['events'][0]['temporal_facts'][0]['to_location'], 'bag')
        self.assertEqual(alice['events'][0]['temporal_facts'][1]['status'], 'stated_plan')

    def test_generic_objects_and_anonymous_people_are_separate(self):
        self.assertEqual(entity_records(self.archive, 'object')['total'], 2)
        rows = entity_records(self.archive, 'person')['items']
        self.assertEqual(sum(r['name'] == 'unidentified person 1' for r in rows), 2)

    def test_affected_people_are_searchable(self):
        bob = next(r for r in entity_records(self.archive, 'person')['items'] if r['name'] == 'Bob')
        self.assertEqual(bob['events'][0]['temporal_facts'][0]['affected_people'], ['Bob'])

    def test_no_inferred_age_or_fake_occurrence_time(self):
        row = entity_records(self.archive, 'person', 'unidentified')['items'][0]['events'][0]
        self.assertNotIn('age', row['person_observations'][0])
        self.assertEqual(row['source_time']['basis'], 'video_relative')
        self.assertIn('recorded_at', row)

    def test_deleted_stale_and_merged_events_absent(self):
        self.a['stale'] = True
        self.b['merged_into'] = 'a'
        self.assertEqual(entity_records(self.archive)['total'], 0)

    def test_payload_is_detached_and_pagination_works(self):
        original = copy.deepcopy(self.archive)
        data = entity_records(self.archive, limit=1)
        data['items'][0]['events'][0]['temporal_facts'].clear()
        self.assertEqual(self.archive, original)
        self.assertEqual(len(data['items']), 1)
        self.assertNotEqual(data['items'][0]['id'], entity_records(self.archive, offset=1, limit=1)['items'][0]['id'])

    def test_legacy_data_does_not_fabricate_new_facts(self):
        for m in self.archive['moments'].values():
            m.pop('person_observations')
            m.pop('temporal_facts')
        self.assertEqual(entity_records(self.archive)['structured_moments'], 0)

    def test_downstream_payload_preserves_evidence(self):
        self.assertEqual(memory_facts(self.a)['temporal_facts'], self.a['temporal_facts'])
        from memory_reasoner import MemoryReasoner
        self.assertEqual(MemoryReasoner._compact(self.a)['source_time'], self.a['source_time'])

    def test_partial_structured_output_is_rejected(self):
        from qwen_perceive import parse_analysis
        with self.assertRaises(ValueError):
            parse_analysis('Scene: example\nPerson Observations: []\nTemporal Facts: [{', strict=True)
        parsed = parse_analysis('Scene: example\nPerson Observations: []\nTemporal Facts: []', strict=True)
        self.assertEqual(parsed['temporal_facts'], [])

    def test_empty_people_json_array_is_not_a_person_name(self):
        from qwen_perceive import parse_analysis
        parsed = parse_analysis('Scene: example\nPeople: []\nPerson Observations: []\nTemporal Facts: []', strict=True)
        self.assertEqual(parsed['people'], [])

    def test_cjk_output_switches_to_english_fallback_retry(self):
        import qwen_perceive as qp
        from types import SimpleNamespace
        valid = ('Scene: A person walks past a sign with 中文文字.\nLocation: street\n'
                 'People: []\nPerson Observations: []\nUser Action: \nActivity: walking\n'
                 'Objects: sign\nEnvironment: street\nInteraction Details: A person walks past a sign.\n'
                 'Temporal Facts: []')
        responses = [
            SimpleNamespace(status_code=200, json=lambda: {"analysis": "Scene: 门口有中文标志"}),
            SimpleNamespace(status_code=200, json=lambda: {"analysis": "Scene: 中文文字"}),
            SimpleNamespace(status_code=200, json=lambda: {"analysis": valid}),
        ]
        get_response = SimpleNamespace(status_code=200, json=lambda: {"model": "Qwen3-VL"})
        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / 'cache.json')
            with patch.object(qp, 'video_path_of', return_value=Path('synthetic.mp4')), \
                 patch.object(qp, 'extract_frames', return_value=([b'frame'], [0.0])), \
                 patch.object(qp.requests, 'post', side_effect=responses) as post, \
                 patch.object(qp.requests, 'get', return_value=get_response):
                result = perceiver.perceive({'id': 'language-retry-case', 'video': 'synthetic.mp4',
                                             'start_time': 0, 'end_time': 1})
        self.assertEqual(result['source'], 'qwen')
        self.assertEqual(post.call_count, 3)
        self.assertIn('Final rewrite using only visible frame pixels', post.call_args_list[2].kwargs['data']['prompt'])
        self.assertIn('unreadable text', result['fields']['scene'])

    def test_incomplete_json_retries_before_a_result_is_cached(self):
        import qwen_perceive as qp
        from types import SimpleNamespace
        valid = ('Scene: A person places a phone in a bag.\nLocation: room\nPeople: Alice\n'
                 'Person Observations: [{"entity":"Alice","appearance":"blue shirt",'
                 '"age_basis":"unknown","frame_index":1,"evidence":"visible","confidence":"high"}]\n'
                 'User Action: \nActivity: putting a phone away\nObjects: phone, bag\nEnvironment: room\n'
                 'Interaction Details: Alice places the phone in the bag.\n'
                 'Temporal Facts: [{"subject_type":"object","subject":"phone","predicate":"moved",'
                 '"actor":"Alice","from_location":"desk","to_location":"bag","frame_index":1,'
                 '"status":"observed","evidence":"visible transfer","confidence":"high"}]')
        responses = [
            SimpleNamespace(status_code=200, json=lambda: {"analysis":
                'Scene: example\nPerson Observations: []\nTemporal Facts: [{'}),
            SimpleNamespace(status_code=200, json=lambda: {"analysis": valid}),
        ]
        get_response = SimpleNamespace(status_code=200, json=lambda: {"model": "Qwen3-VL"})
        with tempfile.TemporaryDirectory() as directory:
            perceiver = qp.QwenPerceiver(cache_file=Path(directory) / 'cache.json')
            with patch.object(qp, 'video_path_of', return_value=Path('synthetic.mp4')), \
                 patch.object(qp, 'extract_frames', return_value=([b'frame'], [0.0])), \
                 patch.object(qp.requests, 'post', side_effect=responses) as post, \
                 patch.object(qp.requests, 'get', return_value=get_response):
                result = perceiver.perceive({'id': 'retry-case', 'video': 'synthetic.mp4',
                                             'start_time': 0, 'end_time': 1})
            self.assertEqual(result['source'], 'qwen')
            self.assertEqual(post.call_count, 2)
            self.assertIn('retry-case', perceiver.cache)

    def test_custom_clip_hash_is_not_a_clock(self):
        from qwen_perceive import _intra_hour_offset, clip_range
        self.assertEqual(_intra_hour_offset('FEISHU_1_1_123456abcd'), 0)
        self.assertEqual(clip_range({'id': 'FEISHU_1_1_123456abcd', 'start_time': 0, 'end_time': 15}), (0, 15))
        self.assertEqual(_intra_hour_offset('DAY1_A2_ALICE_11093000_en'), 570)

    def test_structured_fact_is_persisted_and_retrieved_across_layers(self):
        from memory import PersonMemory
        with tempfile.TemporaryDirectory() as directory:
            memory = PersonMemory(directory)
            mid = memory.add(scene='Synthetic scene', user_action='', needs=[], solutions=[], people=['Alice'],
                             temporal_facts=self.a['temporal_facts'],
                             person_observations=self.a['person_observations'],
                             source_time=self.a['source_time'],
                             source_provenance=self.a['source_provenance'])
            restored = PersonMemory(directory)
            hits = restored.query_hierarchical('phone moved bag', top_k=3)
            hit = next(x for x in hits if x['moment']['id'] == mid)
            self.assertIn(7, hit['matched_layers'])
            self.assertEqual(hit['moment']['temporal_facts'][0]['to_location'], 'bag')
            self.assertEqual(hit['moment']['temporal_facts'][1]['status'], 'stated_plan')


if __name__ == '__main__':
    unittest.main()
