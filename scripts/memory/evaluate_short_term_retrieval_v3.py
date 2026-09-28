"""Short-term retrieval diagnostic using automatic EgoLife reference tags."""
import json, sys, tempfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from memory import PersonMemory
TEST = ROOT / 'data/egolife/testset/short_term/moments_scattered_v2_120.jsonl'
CACHE = ROOT / 'data/egolife/runs/qwen_cache_expanded_eval.json'
OUT = ROOT / 'data/egolife/runs/short_term_retrieval_v3_mapped_result.json'
QUERYSET = ROOT / 'data/egolife/testset/short_term/retrieval_queries_v3.jsonl'

def person_of(sample_id):
    parts = sample_id.split('_')
    return parts[1] + '_' + parts[2]

def timestamp_of(sample_id):
    parts = sample_id.split('_')
    day, stamp = int(parts[0][3:]), parts[3]
    event_date = date(2026, 9, 1) + timedelta(days=day - 1)
    return f'{event_date.isoformat()}T{stamp[:2]}:{stamp[2:4]}:{stamp[4:6]}'

def as_list(value):
    if isinstance(value, list): return [str(x).strip() for x in value if str(x).strip()]
    if value in (None, ''): return []
    return [str(value).strip()]

def fields_for(row, cache):
    fields = (cache.get(row['id']) or {}).get('fields') or {}
    return {
        'scene': str(fields.get('scene') or row.get('scene') or ''),
        'user_action': str(fields.get('user_action') or row.get('user_action') or ''),
        'needs': as_list(fields.get('needs')), 'solutions': as_list(fields.get('solutions')),
        'people': as_list(fields.get('people')),
        'location': str(fields.get('location') or '') or None,
        'activity': str(fields.get('activity') or '') or None,
        'environments': as_list(fields.get('environments')),
        'objects': as_list(fields.get('objects')),
        'timestamp': timestamp_of(row['id']),
    }

ACTIVITY_QUERY = {
    '饮食': 'eating drinking food cooking meal',
    '移动': 'walking moving stairs standing',
    '社交': 'talking chatting discussion conversation group',
    '观察': 'watching looking observing',
    '操作物品': 'handling objects assembling opening placing organizing',
    '清洁': 'cleaning washing wiping trash',
    '电子设备': 'electronics electronic device phone computer charging',
}
def mean(xs): return round(sum(xs) / len(xs), 4) if xs else None

def main():
    rows = [json.loads(line) for line in TEST.read_text(encoding='utf-8').splitlines() if line.strip()]
    cache = json.loads(CACHE.read_text(encoding='utf-8'))
    mems, id_maps = {}, {}
    ref_groups = defaultdict(set)
    tag_fields = [('people', 'people_gt'), ('locations', 'location_gt'),
                  ('environments', 'environments_gt'), ('objects', 'objects_gt'),
                  ('activity_events', 'activity_gt')]
    with tempfile.TemporaryDirectory(prefix='short_term_retrieval_v3_') as tmp:
        for row in rows:
            sid = row['id']; person = person_of(sid)
            if person not in mems:
                mem = PersonMemory(memory_dir=str(Path(tmp) / person)); mem._save = lambda: None
                mems[person] = mem; id_maps[person] = {}
            data = fields_for(row, cache)
            mid = mems[person].add(**data)
            id_maps[person][mid] = sid
        for row in rows:
            person, sid = person_of(row['id']), row['id']
            gt = row.get('ground_truth') or {}
            for dim, key in tag_fields:
                for label in as_list(gt.get(key)):
                    ref_groups[(person, dim, label)].add(sid)
        details = []
        queryset_rows = []
        min_support = 2
        for (person, dim, label), expected in sorted(ref_groups.items()):
            if len(expected) < min_support: continue
            query_text = ACTIVITY_QUERY.get(label, label) if dim == 'activity_events' else label
            queryset_rows.append({'query_id': f'st-v3-{len(queryset_rows)+1:03d}', 'participant': person, 'dimension': dim, 'reference_tag': label, 'query_text': query_text, 'expected_ids': sorted(expected)})
            found = mems[person].query(query_text, top_k=5)
            retrieved_ids = [id_maps[person].get(m.get('id')) for m in found]
            retrieved_ids = [x for x in retrieved_ids if x]
            relevant = [sid for sid in retrieved_ids if sid in expected]
            details.append({
                'participant': person, 'dimension': dim, 'reference_tag': label, 'query_text': query_text,
                'expected_count': len(expected), 'relevant_in_top5': len(set(relevant)),
                'precision_at_5': round(len(set(relevant)) / 5, 4),
                'recall_at_5': round(len(set(relevant)) / len(expected), 4),
                'hit_at_5': bool(relevant), 'top5_ids': retrieved_ids,
            })
    by_dimension = {}
    for dim in sorted({x['dimension'] for x in details}):
        group = [x for x in details if x['dimension'] == dim]
        by_dimension[dim] = {
            'queries': len(group), 'hit_at_5_count': sum(x['hit_at_5'] for x in group),
            'hit_at_5': mean([float(x['hit_at_5']) for x in group]),
            'precision_at_5': mean([x['precision_at_5'] for x in group]),
            'recall_at_5': mean([x['recall_at_5'] for x in group]),
        }
    n = len(details)
    report = {
        'benchmark': 'short-term v3 reference-tag retrieval diagnostic',
        'samples': len(rows), 'participants': len(mems), 'participant_rows': {k: sum(person_of(r['id']) == k for r in rows) for k in sorted(mems)},
        'minimum_reference_support': min_support, 'query_count': n,
        'overall': {
            'hit_at_5_count': sum(x['hit_at_5'] for x in details),
            'hit_at_5': mean([float(x['hit_at_5']) for x in details]),
            'precision_at_5': mean([x['precision_at_5'] for x in details]),
            'recall_at_5': mean([x['recall_at_5'] for x in details]),
        },
        'by_dimension': by_dimension,
        'method': {
            'memory': '120 Qwen-generated moments loaded into six isolated participant stores; query() top 5.',
            'reference': 'EgoLife testset ground_truth tags generated by transcript/caption vocabulary rules; not human verified.',
            'query': 'Each frequent tag is used as a concise lookup query; activity categories use a fixed English synonym map because the reference categories are Chinese while stored Qwen memories are English.',
            'limitation': 'This diagnoses tag lookup only; it is not natural-language QA, human gold, or the 95% category accuracy acceptance metric.',
        },
        'details': details,
    }
    QUERYSET.parent.mkdir(parents=True, exist_ok=True)
    QUERYSET.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in queryset_rows), encoding='utf-8')
    report['queryset_file'] = str(QUERYSET)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'overall': report['overall'], 'by_dimension': by_dimension, 'query_count': n, 'report': str(OUT)}, ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
