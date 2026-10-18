"""Source-backed entity records for browsing and downstream retrieval.

This view is derived from the canonical archive. It never infers identity,
age, completed plans, or an object's present location from missing evidence.
"""
from copy import deepcopy
import hashlib
import re

from memory import normalize_object_transition_facts, object_visual_observations


def memory_facts(moment):
    """Stable payload shared by the retrieval response and entity browser."""
    view = deepcopy(moment or {})
    normalize_object_transition_facts(view)
    object_rows = object_visual_observations(view)
    return {key: deepcopy(view.get(key) or default) for key, default in (
        ('scene', ''), ('user_action', ''), ('interaction_details', ''),
        ('people', []), ('location', ''), ('activity', ''),
        ('objects', []), ('environments', []), ('person_observations', []),
        ('temporal_facts', []), ('object_observations', object_rows),
        ('source_time', {}), ('source_provenance', {}), ('source_evidence', ''),
    )}


def _name(value):
    return ' '.join(str(value or '').split())[:180]


def _anonymous(name):
    return bool(re.search(r'unknown|unidentified|unnamed|camera wearer|speaker\s*\d|未知|匿名|未识别', name, re.I))


class EntityRecordIndex:
    def __init__(self, layer7):
        groups = {}
        active = [m for m in (layer7.get('moments') or {}).values()
                  if isinstance(m, dict) and not m.get('merged_into') and not m.get('stale')]
        self.source_moments = len(active)
        self.structured_moments = 0
        for original_moment in active:
            # Only normalization inputs are copied; attached media can be large.
            moment = dict(original_moment)
            temporal_facts = moment.get('temporal_facts')
            if isinstance(temporal_facts, list):
                moment['temporal_facts'] = [
                    dict(row) if isinstance(row, dict) else row for row in temporal_facts
                ]
            objects = moment.get('objects')
            if isinstance(objects, list):
                moment['objects'] = list(objects)
            normalize_object_transition_facts(moment)
            object_rows = object_visual_observations(moment)
            observations = [x for x in moment.get('person_observations') or [] if isinstance(x, dict)]
            facts = [x for x in moment.get('temporal_facts') or [] if isinstance(x, dict)]
            if observations or facts or object_rows:
                self.structured_moments += 1

            mid = str(moment.get('id') or '')
            provenance = moment.get('source_provenance') or {}
            segment = str(provenance.get('source_segment_id') or mid)
            video = str(provenance.get('source_video_id') or segment)
            labels = set()
            for entity_kind, values in (
                ('person', moment.get('people') or []), ('object', moment.get('objects') or []),
                ('place', [moment.get('location')]), ('environment', moment.get('environments') or []),
            ):
                for value in values:
                    if _name(value):
                        labels.add((entity_kind, _name(value)))
            labels.update(('object', _name(x.get('object'))) for x in object_rows if x.get('object'))
            labels.update(('person', _name(x.get('entity'))) for x in observations if x.get('entity'))
            for fact in facts:
                fact_kind = fact.get('subject_type')
                if fact_kind in {'person', 'object', 'place', 'environment'} and fact.get('subject'):
                    labels.add((fact_kind, _name(fact['subject'])))
                for person in [fact.get('actor')] + (fact.get('affected_people') or []):
                    if _name(person):
                        labels.add(('person', _name(person)))

            media = moment.get('multimodal') or {}
            visual = media.get('visual_frames') or ([media['visual']] if media.get('visual') else [])
            visual_objects = (media.get('object_visuals') or {}).get('items') or []
            base_event = {
                'moment_id': mid, 'scene': moment.get('scene') or '',
                'interaction_details': moment.get('interaction_details') or '',
                'source_evidence': str(moment.get('source_evidence') or '')[:4000],
                'source_time': deepcopy(moment.get('source_time') or {}),
                'source_provenance': deepcopy(provenance),
                'recorded_at': moment.get('timestamp'),
                'location': moment.get('location') or '',
                'people': deepcopy(moment.get('people') or []),
            }
            for entity_kind, label in sorted(labels):
                # Repeated generic objects and anonymous people are source-scoped.
                scope = segment if ((entity_kind == 'person' and _anonymous(label)) or entity_kind == 'object') else (
                    '' if entity_kind == 'person' else video)
                identity = (entity_kind, label.casefold(), scope)
                record = groups.setdefault(identity, {
                    'id': hashlib.sha256(repr(identity).encode()).hexdigest()[:20],
                    'kind': entity_kind, 'name': label,
                    'identity_scope': 'named_mentions' if not scope else 'source_observations',
                    'source_scope': scope, 'events': [], 'fact_count': 0,
                })
                selected_people = [deepcopy(x) for x in observations
                                   if _name(x.get('entity')).casefold() == label.casefold()]
                for person in selected_people:
                    if person.get('age_basis') not in {'stated', 'visible_text'}:
                        person.pop('age', None)
                    if (person.get('apparent_age_basis') != 'visual_estimate'
                            or person.get('apparent_age_range') not in {
                                'child', 'teenager', '18-29', '30-44', '45-59', '60-74', '75+'
                            }):
                        person.pop('apparent_age_range', None)
                        person.pop('apparent_age_basis', None)
                selected_facts = [deepcopy(x) for x in facts if label.casefold() in {
                    _name(x.get('subject')).casefold(), _name(x.get('actor')).casefold(),
                    *(_name(p).casefold() for p in x.get('affected_people') or []),
                }]
                selected_objects = [deepcopy(x) for x in object_rows
                                    if _name(x.get('object')).casefold() == label.casefold()]
                visual_people = (media.get('person_visuals') or {}).get('items') or []
                person_crops = [deepcopy(ref) for item in visual_people
                                if _name(item.get('person')).casefold() == label.casefold()
                                for ref in item.get('frame_refs') or []
                                if ref.get('visual_role') == 'person_crop']
                object_crops = [deepcopy(ref) for item in visual_objects
                                if _name(item.get('object')).casefold() == label.casefold()
                                for ref in item.get('frame_refs') or []
                                if ref.get('visual_role') == 'object_crop']
                crops = person_crops or object_crops
                event = dict(base_event)
                event.update({
                    'person_observations': selected_people,
                    'temporal_facts': selected_facts,
                    'object_observations': selected_objects,
                    'images': crops[:3],
                    'image_kind': 'person_crop' if person_crops else ('object_crop' if object_crops else 'none'),
                })
                record['events'].append(event)
                record['fact_count'] += len(selected_people) + len(selected_facts) + len(selected_objects)

        records = list(groups.values())
        for record in records:
            record['events'].sort(key=lambda event: (
                str(event['source_provenance'].get('source_video_id') or ''),
                float(event['source_time'].get('clip_start_s') or 0), event['moment_id']))
            record['event_count'] = len(record['events'])
            record['_search_text'] = None
        records.sort(key=lambda record: (
            -record['fact_count'], -record['event_count'], record['name'].casefold(), record['id']))
        self.records = records

    def query(self, kind='all', query='', offset=0, limit=12):
        kind = str(kind or 'all')
        needle = _name(query).casefold()
        selected = []
        for record in self.records:
            if kind != 'all' and record['kind'] != kind:
                continue
            if needle:
                search_text = record.get('_search_text')
                if search_text is None:
                    search_text = (record['name'] + ' ' + str(record['events'])).casefold()
                    record['_search_text'] = search_text
                if needle not in search_text:
                    continue
            selected.append(record)
        offset, limit = max(0, int(offset)), max(1, min(50, int(limit)))
        items = deepcopy(selected[offset:offset + limit])
        for item in items:
            item.pop('_search_text', None)
        return {'items': items, 'total': len(selected), 'offset': offset, 'limit': limit,
                'source_moments': self.source_moments,
                'structured_moments': self.structured_moments}


def entity_records(layer7, kind='all', query='', offset=0, limit=12):
    return EntityRecordIndex(layer7).query(kind=kind, query=query, offset=offset, limit=limit)
