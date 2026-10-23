"""Source-backed entity records for browsing and downstream retrieval.

This view is derived from the canonical archive. It never infers identity,
age, completed plans, or an object's present location from missing evidence.
"""
from copy import deepcopy
import hashlib
import re

from memory import derive_object_states, normalize_object_transition_facts, object_visual_observations
from face_identity import link_records


def memory_facts(moment):
    """Stable payload shared by the retrieval response and entity browser."""
    view = deepcopy(moment or {})
    normalize_object_transition_facts(view)
    object_rows = object_visual_observations(view)
    object_states = derive_object_states(view)
    return {key: deepcopy(view.get(key) or default) for key, default in (
        ('scene', ''), ('user_action', ''), ('interaction_details', ''),
        ('people', []), ('location', ''), ('activity', ''),
        ('objects', []), ('environments', []), ('person_observations', []),
        ('temporal_facts', []), ('object_observations', object_rows),
        ('object_states', object_states), ('source_time', {}), ('source_provenance', {}), ('source_evidence', ''),
    )}


def _name(value):
    return ' '.join(str(value or '').split())[:180]


def _anonymous(name):
    return bool(re.search(r'unknown|unidentified|unnamed|camera wearer|speaker\s*\d|未知|匿名|未识别', name, re.I))


def _clip_local_person(name):
    return bool(re.fullmatch(
        r"(?:woman|man|person|people|人物|人物编号)\s*#?\s*\d+",
        str(name or "").strip(), re.I))


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
    def __init__(self, layer7, asset_root=None, identity_correction_path=None,
                 compute_missing_face_embeddings=True):
        groups = {}
        active = [m for m in (layer7.get('moments') or {}).values()
                  if isinstance(m, dict) and not m.get('merged_into') and not m.get('stale')]
        # A crop assigned to multiple labels is ambiguous, even if Qwen marked
        # each assignment verified. Hide it from every profile until corrected.
        asset_labels = {}
        for moment in active:
            for item in ((moment.get("multimodal") or {}).get("person_visuals") or {}).get("items") or []:
                if not isinstance(item, dict):
                    continue
                label = _name(item.get("person")).casefold()
                if not label:
                    continue
                for ref in item.get("frame_refs") or []:
                    if (isinstance(ref, dict) and ref.get("visual_role") == "face_crop"
                            and ref.get("identity_verified") is True and ref.get("asset")):
                        asset_labels.setdefault(str(ref["asset"]), set()).add(label)
        ambiguous_face_assets = set()
        for asset, labels in asset_labels.items():
            if len(labels) <= 1:
                continue
            named = {label for label in labels if not _clip_local_person(label)}
            local = {label for label in labels if _clip_local_person(label)}
            # A single canonical person plus temporary clip label is a duplicate
            # assignment of the same verified crop, not a two-person ambiguity.
            if not (len(named) == 1 and local):
                ambiguous_face_assets.add(asset)

        # Nightly identity links are source-scoped: only the cited target and
        # evidence moments join the stable person gallery.
        stable_person_members = set()
        source_identity_aliases = {}
        archive = layer7.get('moments') or {}
        for target in active:
            target_id = str(target.get('id') or '')
            for link in target.get('nightly_identity_resolutions') or []:
                if not isinstance(link, dict):
                    continue
                canonical = _name(link.get('canonical_entity')).casefold()
                observed = _name(link.get('observed_entity'))
                if not canonical:
                    continue
                stable_person_members.add((target_id, canonical))
                for source_id in link.get('evidence_moment_ids') or []:
                    source_id = str(source_id)
                    source = archive.get(source_id)
                    source_people = set()
                    if isinstance(source, dict):
                        source_people = {
                            _name(value).casefold() for value in source.get('people') or []
                        } | {
                            _name(row.get('entity')).casefold()
                            for row in source.get('person_observations') or []
                            if isinstance(row, dict)
                        }
                    if canonical in source_people:
                        stable_person_members.add((source_id, canonical))
                        # Propagate an audited local label only to cited moments
                        # that independently contain the canonical identity. This
                        # eliminates duplicate clip-local cards without applying a
                        # global name heuristic across unrelated videos.
                        if observed and _clip_local_person(observed):
                            source_identity_aliases.setdefault(source_id, {}).setdefault(
                                observed.casefold(), set()).add(canonical)
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
            media = moment.get("multimodal") or {}
            # Existing records may have been written before audio evidence was
            # attached. Prefer an explicitly supported spoken alias or a readable
            # on-screen name when presenting the person, while keeping all evidence.
            # Apply the source-scoped nightly identity audit to every structured
            # mention, including temporal-fact actors that are not observation rows.
            identity_aliases = {
                alias: next(iter(targets))
                for alias, targets in source_identity_aliases.get(
                    str(moment.get('id') or ''), {}).items()
                if len(targets) == 1
            }
            for link in moment.get("nightly_identity_resolutions") or []:
                if not isinstance(link, dict):
                    continue
                old_name = _name(link.get("observed_entity"))
                canonical = _name(link.get("canonical_entity"))
                if old_name and canonical and old_name.casefold() != canonical.casefold():
                    identity_aliases[old_name.casefold()] = canonical
            for row in observations:
                audit = row.get("identity_resolution") or {}
                if not isinstance(audit, dict) or audit.get("source") != "nightly_evidence_reconciliation":
                    continue
                old_name = _name(audit.get("observed_entity") or row.get("source_entity"))
                canonical = _name(audit.get("canonical_entity") or row.get("entity"))
                if old_name and canonical and old_name.casefold() != canonical.casefold():
                    identity_aliases[old_name.casefold()] = canonical

            # A single verified crop may be repeated under a stable spoken/name
            # label and a clip-local detector label. Treat that exact crop as a
            # source-scoped alias only when there is exactly one non-local label;
            # if two named identities share it, retain the ambiguity.
            same_frame_labels = {}
            for item in ((media.get("person_visuals") or {}).get("items") or []):
                if not isinstance(item, dict):
                    continue
                label = _name(item.get("person"))
                if not label:
                    continue
                for ref in item.get("frame_refs") or []:
                    if (isinstance(ref, dict) and ref.get("visual_role") == "face_crop"
                            and ref.get("identity_verified") is True
                            and str(ref.get("identity_assignment") or "").startswith("qwen_visual_anchor+")
                            and ref.get("asset")):
                        same_frame_labels.setdefault(str(ref["asset"]), set()).add(label)
            visual_aliases = {}
            for labels_for_crop in same_frame_labels.values():
                named = {label for label in labels_for_crop if not _clip_local_person(label)}
                local = {label for label in labels_for_crop if _clip_local_person(label)}
                if len(named) == 1:
                    canonical = next(iter(named))
                    for alias in local:
                        visual_aliases.setdefault(alias.casefold(), set()).add(canonical)
            for alias, targets in visual_aliases.items():
                if len(targets) == 1:
                    identity_aliases.setdefault(alias, next(iter(targets)))

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
                return identity_aliases.get(
                    label.casefold(), person_aliases.get(label.casefold(), label))

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
                # Name similarity alone is not identity evidence. Candidate records
                # remain clip-scoped here; face_identity later links only reliable
                # person-crop embeddings or explicit user corrections.
                named_in_frame = any(
                    _name(row.get('readable_name')).casefold() == label.casefold()
                    and bool(str(row.get('name_evidence') or '').strip())
                    for row in selected_person_rows(observations, label, person_label)
                )
                known_participant = label.casefold() in {
                    'jake', 'alice', 'tasha', 'lucia', 'katrina', 'shure'
                }
                if entity_kind == 'person':
                    resolved_identity = (mid, label.casefold()) in stable_person_members
                    scope = '' if named_in_frame or known_participant or resolved_identity else segment
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
                person_crops = []
                seen_person_crops = set()
                for item in visual_people:
                    if person_label(item.get('person')).casefold() != label.casefold():
                        continue
                    for ref in item.get('frame_refs') or []:
                        if (ref.get('visual_role') != 'face_crop'
                                or ref.get('identity_verified') is not True
                                or not str(ref.get('identity_assignment') or '').startswith('qwen_visual_anchor+')
                                or str(ref.get('asset') or '') in ambiguous_face_assets):
                            continue
                        crop_key = (str(ref.get('asset') or ''),
                                    str(ref.get('at_s', ref.get('frame_time_s', ''))))
                        if crop_key in seen_person_crops:
                            continue
                        seen_person_crops.add(crop_key)
                        person_crops.append(deepcopy(ref))
                person_crops.sort(key=lambda ref: (
                    0 if isinstance(ref.get('nightly_importance'), dict) else 1,
                    -float((ref.get('nightly_importance') or {}).get('score') or 0),
                    0 if ref.get('quality_tier') == 'high' else 1,
                    float(ref.get('at_s') or 0)))
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
                    'images': crops[:12],
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
        self.identity_link_status = link_records(
            records, asset_root=asset_root, correction_path=identity_correction_path,
            compute_missing=compute_missing_face_embeddings)
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
                'structured_moments': self.structured_moments,
                'identity_link_status': self.identity_link_status}


def entity_records(layer7, kind='all', query='', offset=0, limit=12):
    return EntityRecordIndex(layer7).query(kind=kind, query=query, offset=offset, limit=limit)
