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


def _speech_placeholder(name):
    """Whisper cannot attribute a voice, so speaker placeholders are not scene people.

    A visually observed 'unidentified person 1' stays a card; an 'unidentified
    speaker' is only ever a slot in a quoted utterance.
    """
    return bool(re.search(
        r'^(?:unidentified|unknown)\s+(?:speaker|addressee|voice)\b'
        r'|^(?:unidentified|unknown)\s+person referred to as\b'
        r'|^public-address announcement$|未识别说话人|未知说话', name, re.I))


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
            # Existing records may have been written before audio evidence was
            # attached. Prefer an explicitly supported spoken alias or a readable
            # on-screen name when presenting the person, while keeping all evidence.
            person_aliases = {}
            for row in observations:
                original = _name(row.get('entity'))
                readable = _name(row.get('readable_name'))
                if readable and str(row.get('name_evidence') or '').strip():
                    person_aliases[original.casefold()] = readable
                    continue
                spoken = next((item for item in row.get('spoken_aliases') or []
                               if isinstance(item, dict) and item.get('alias')
                               and item.get('status') == 'probable_visual_match'), None)
                if original and spoken:
                    person_aliases[original.casefold()] = _name(spoken.get('alias'))

            def person_label(value):
                label = _name(value)
                return person_aliases.get(label.casefold(), label)

            if observations or facts or object_rows:
                self.structured_moments += 1

            mid = str(moment.get('id') or '')
            provenance = moment.get('source_provenance') or {}
            segment = str(provenance.get('source_segment_id') or mid)
            video = str(provenance.get('source_video_id') or segment)
            def selected_person_rows(rows, wanted, label_fn):
                return [row for row in rows
                        if label_fn(row.get('entity')).casefold() == wanted.casefold()]

            labels = set()
            for entity_kind, values in (
                ('person', moment.get('people') or []), ('object', moment.get('objects') or []),
                ('place', [moment.get('location')]), ('environment', moment.get('environments') or []),
            ):
                for value in values:
                    label = person_label(value) if entity_kind == 'person' else _name(value)
                    if label:
                        labels.add((entity_kind, label))
            labels.update(('object', _name(x.get('object'))) for x in object_rows if x.get('object'))
            labels.update(('person', person_label(x.get('entity')))
                          for x in observations if x.get('entity'))
            for fact in facts:
                fact_kind = fact.get('subject_type')
                if fact_kind in {'person', 'object', 'place', 'environment'} and fact.get('subject'):
                    subject = person_label(fact['subject']) if fact_kind == 'person' else _name(fact['subject'])
                    labels.add((fact_kind, subject))
                for person in [fact.get('actor')] + (fact.get('affected_people') or []):
                    if _name(person):
                        labels.add(('person', person_label(person)))

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
                'people': [person_label(value) for value in moment.get('people') or []],
            }
            for entity_kind, label in sorted(labels):
                if entity_kind == 'person' and _speech_placeholder(label):
                    continue
                # Textual similarity is not proof that two visible people are the same
                # person. Keep person labels local to their source clip unless the
                # observation carries a directly linked readable name, or the label is
                # one of the fixed identities in the long-term participant dataset.
                named_in_frame = any(
                    _name(row.get('readable_name')).casefold() == label.casefold()
                    and bool(str(row.get('name_evidence') or '').strip())
                    for row in selected_person_rows(observations, label, person_label)
                )
                known_participant = label.casefold() in {
                    'jake', 'alice', 'tasha', 'lucia', 'katrina', 'shure'
                }
                if entity_kind == 'person':
                    scope = '' if named_in_frame or known_participant else segment
                elif entity_kind == 'object':
                    scope = segment
                else:
                    scope = video
                identity = (entity_kind, label.casefold(), scope)
                record = groups.setdefault(identity, {
                    'id': hashlib.sha256(repr(identity).encode()).hexdigest()[:20],
                    'kind': entity_kind, 'name': label,
                    'identity_scope': 'named_mentions' if not scope else 'source_observations',
                    'source_scope': scope, 'events': [], 'fact_count': 0,
                })
                selected_people = []
                for original_person in selected_person_rows(observations, label, person_label):
                    person = deepcopy(original_person)
                    person['entity'] = person_label(person.get('entity'))
                    selected_people.append(person)
                for person in selected_people:
                    if person.get('age_basis') not in {'stated', 'visible_text'}:
                        person.pop('age', None)
                    if (person.get('apparent_age_basis') != 'visual_estimate'
                            or person.get('apparent_age_range') not in {
                                'child', 'teenager', '18-29', '30-44', '45-59', '60-74', '75+'
                            }):
                        person.pop('apparent_age_range', None)
                        person.pop('apparent_age_basis', None)
                selected_facts = []
                for original_fact in facts:
                    subject = (person_label(original_fact.get('subject'))
                               if original_fact.get('subject_type') == 'person'
                               else _name(original_fact.get('subject')))
                    linked = {
                        subject.casefold(), person_label(original_fact.get('actor')).casefold(),
                        *(person_label(p).casefold()
                          for p in original_fact.get('affected_people') or []),
                    }
                    if label.casefold() in linked:
                        selected_facts.append(deepcopy(original_fact))
                selected_objects = [deepcopy(x) for x in object_rows
                                    if _name(x.get('object')).casefold() == label.casefold()]
                visual_people = (media.get('person_visuals') or {}).get('items') or []
                person_crops = [deepcopy(ref) for item in visual_people
                                if person_label(item.get('person')).casefold() == label.casefold()
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
