#!/usr/bin/env python3
"""Repair source-grounded people and event gaps in the 35 custom video memories."""
from __future__ import annotations
import copy, json, os, re, sys, tempfile, time, fcntl
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'/'memory'), str(ROOT/'code')]
from interactive_runner import feishu_test_moments
from qwen_perceive import (_build_person_repair_prompt, _parse_person_repair,
    _apply_person_repair, _person_reference_labels, _build_temporal_fact_repair_prompt,
    _parse_json_array, _parse_json_array_prefix, _apply_temporal_fact_repair,
    _object_move_evidence_error, _structured_fact_quality_error, clip_range,
    extract_frames, video_path_of, QwenPerceiver)
from grounding_cache import PersistentObjectGroundingCache
from multimodal_memory import MultimodalAssetStore
CACHE=ROOT/'data/egolife/runs/qwen_cache_memory_prompt_v3.json'
GROUNDING=ROOT/'data/egolife/runs/interactive_object_grounding_cache.sqlite3'
API=os.environ.get('QWEN_API_URL','http://127.0.0.1:8007').rstrip('/')
LOW_FACT_IDS={'FEISHU_5_4_c21aee7538','FEISHU_7_1_41aeb05dab','FEISHU_7_2_325c12353f','FEISHU_9_1_fabdd91b9a'}
PERSON_REVIEW_IDS={'FEISHU_2_1_b4d69d0493','FEISHU_5_3_570dc77996','FEISHU_7_1_41aeb05dab','FEISHU_7_2_325c12353f','FEISHU_10_1_1d077083a6','FEISHU_12_2_3ce4e03600','FEISHU_12_3_d781904f40','FEISHU_3_1_3adf23f5bb','FEISHU_3_3_6f73e3eba3','FEISHU_5_1_90383d197b','FEISHU_9_1_fabdd91b9a'}
AGE_PATTERNS=[
 (r'\b(?:elderly|older|younger|young|old)\s+(?:woman|man|adult|person|people)\b','person'),
 (r'\b(?:elderly|older|younger|young|old)\s+(?:women|men|adults|persons)\b','people'),
 (r'\b(?:baby|infant|toddler|newborn|child|boy|girl)\b','person'),
 (r'\b(?:babies|infants|toddlers|newborns|children|boys|girls)\b','people'),
]
def deage(text):
    out=str(text or '')
    out=re.sub(r'\bchildren\s+and\s+adults\b','people',out,flags=re.I)
    out=re.sub(r'\bchildren\s+or\s+adults\b','people',out,flags=re.I)
    for pat,repl in AGE_PATTERNS: out=re.sub(pat,repl,out,flags=re.I)
    out=re.sub(r'\bperson\s+in\s+(?=yellow onesie|blue outfit|brown jacket|dark clothing)','person in ',out,flags=re.I)
    out=re.sub(r'\s+',' ',out).strip()
    out=re.sub(r'\bperson\s+person\b','person',out,flags=re.I)
    return out

def normalize_fields(fields):
    f=copy.deepcopy(fields)
    alias_map={}
    observations=[]
    for old in f.get('person_observations') or []:
        if not isinstance(old,dict): continue
        row=dict(old)
        old_entity=str(row.get('entity') or '')
        row['entity']=deage(old_entity)
        alias_map[old_entity]=row['entity']
        for key in ('appearance','face_features','evidence'):
            if row.get(key): row[key]=deage(row[key])
        if row.get('age_basis')!='visible_text' or not str(row.get('age') or '').isdigit():
            row['age']=''; row['age_basis']='unknown'
        observations.append(row)
    f['person_observations']=observations
    def remap(text):
        out=str(text or '')
        for old,new in sorted(alias_map.items(),key=lambda x:len(x[0]),reverse=True):
            if old and new!=old: out=re.sub(r'(?<![\w])'+re.escape(old)+r'(?![\w])',lambda _m,n=new:n,out,flags=re.I)
        return deage(out)
    for key in ('scene','location','activity','interaction_details','user_action','time_occasion'):
        if f.get(key): f[key]=remap(f[key])
    for key in ('people','objects','environments'):
        vals=f.get(key) or []
        if isinstance(vals,list):
            cleaned=[]
            for value in vals:
                label=remap(value)
                if key=='objects' and re.match(r'^(?:a|an|the\s+)?(?:person|people)\b',label,re.I): continue
                if label and label not in cleaned: cleaned.append(label)
            f[key]=cleaned
    facts=[]
    for fact in f.get('temporal_facts') or []:
        if not isinstance(fact,dict): continue
        row=dict(fact)
        for key in ('subject','predicate','value','actor','evidence','from_location','to_location','from_state','to_state','time_scope'):
            if row.get(key): row[key]=remap(row[key])
        if isinstance(row.get('affected_people'),list): row['affected_people']=[remap(x) for x in row['affected_people']]
        facts.append(row)
    f['temporal_facts']=facts
    return f

def request_images(prompt, images, label):
    last=None
    for attempt in range(3):
        try:
            resp=requests.post(API+'/analyze_video',files=[('images',(f'{label}_{n+1}.jpg',b,'image/jpeg')) for n,b in enumerate(images)],data={'prompt':prompt,'request_id':f'memory-gap-{label}-{attempt}'},timeout=240)
            if resp.status_code!=200: raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:200]}')
            return str(resp.json().get('analysis') or '')
        except Exception as exc:
            last=exc
            if attempt<2: time.sleep(2*(attempt+1))
    raise RuntimeError(str(last))

def write_entry(mid, entry):
    lock_path=CACHE.with_name(CACHE.name+'.lock')
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock_path,os.O_CREAT|os.O_RDWR,0o600)
    with os.fdopen(fd,'a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        disk=json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
        current=disk.get(mid,{})
        if isinstance(current,dict) and current.get('prompt_version')==entry.get('prompt_version'):
            # Preserve newer perception source metadata while keeping this explicit quality repair.
            merged=dict(current); merged.update(entry); entry=merged
        entry['ts']=time.strftime('%Y-%m-%d %H:%M:%S')
        disk[mid]=entry
        fd,tmp=tempfile.mkstemp(prefix=CACHE.name+'.',suffix='.tmp',dir=str(CACHE.parent))
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as out:
                json.dump(disk,out,ensure_ascii=False,indent=2); out.flush(); os.fsync(out.fileno())
            os.replace(tmp,CACHE)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

def main():
    moments=feishu_test_moments('feishu:all')
    if len(moments)!=35: raise RuntimeError(f'expected 35 custom clips, got {len(moments)}')
    health=requests.get(API+'/health',timeout=8).json()
    if not health.get('model_loaded'): raise RuntimeError('Qwen endpoint is not ready')
    media=object.__new__(MultimodalAssetStore)
    grounding=PersistentObjectGroundingCache(GROUNDING,ROOT)
    moment_by_id={m.get('id'):m for m in moments}
    cache=json.loads(CACHE.read_text(encoding='utf-8'))
    updates=[]
    only_ids={value.strip() for value in os.environ.get('MEMORY_REPAIR_IDS','').split(',') if value.strip()}
    selected_ids=only_ids or (PERSON_REVIEW_IDS|LOW_FACT_IDS)
    for mid in sorted(selected_ids):
        m=moment_by_id.get(mid); entry=copy.deepcopy(cache.get(mid) or {})
        fields=normalize_fields(entry.get('fields') or {})
        if not m or not fields: print(mid,'SKIP missing moment/cache'); continue
        video=video_path_of(m); start,end=clip_range(m)
        frames,_times=extract_frames(video,start,end,6,return_times=True)
        if not frames: print(mid,'FAIL no source frames'); continue
        changed=[]
        if mid in PERSON_REVIEW_IDS:
            refs=_person_reference_labels(fields)
            context={k:fields.get(k) for k in ('scene','location','activity','interaction_details','user_action','people','person_observations','temporal_facts') if fields.get(k)}
            prompt=_build_person_repair_prompt(context,refs)
            prompt+='\nSTRICT ADDITION: do not use any age-category terms such as baby, child, young, elderly, adult, boy, or girl in an alias or description. Describe only visible details. In first-person clips, if hands, forearms, or feet are visible, record the local subject with a descriptive alias such as camera wearer with a gray-gloved hand (never the role alone), describe only those visible parts, and state that the face or full body is outside the view. Age must remain blank unless readable text directly identifies a numeric age for this person. A human figure is a person, never an object. '
            if mid=='FEISHU_7_1_41aeb05dab':
                prompt+='The frames visibly contain one seated human figure with short dark hair, a light blue short-sleeved shirt, gray shorts, sitting on a light floor mat and holding a yellow duck-shaped toy. This is one of the people that must appear in person_observations, not objects. Describe the visible figure from pixels only; do not call it a baby or infer age. '
            if mid=='FEISHU_2_1_b4d69d0493':
                prompt+='In the last sampled frame, several travelers are visibly present around the baggage carousel. Inspect the whole frame carefully. Return separate observations for at least two clearly visible figures when their clothing or position distinguishes them, even when faces are too small to resolve; state that facial details are not discernible. Do not return an empty array for visible distant people. Keep aliases descriptive and grounded in clothing/position, never numbered. '
            prompt+='Return a complete JSON object with the required person_observations array; no prose.'
            raw=request_images(prompt,frames,'person_'+mid)
            payload=_parse_person_repair(raw)
            repaired=_apply_person_repair(fields,payload,refs,len(frames))
            observations=repaired.get('person_observations') or []
            if mid=='FEISHU_7_1_41aeb05dab' and not observations:
                retry=prompt+' The visible seated figure is plainly human; returning an empty array would be incorrect. Retry from the six attached frames.'
                raw=request_images(retry,frames,'person_retry_'+mid)
                payload=_parse_person_repair(raw)
                repaired=_apply_person_repair(fields,payload,refs,len(frames))
                observations=repaired.get('person_observations') or []
            if mid=='FEISHU_2_1_b4d69d0493' and len(observations)<2:
                retry=prompt+' The final frame clearly contains more than one human figure; inspect it again and return each visibly distinguishable figure with clothing/position only. Never infer age, gender, or identity. '
                raw=request_images(retry,frames,'person_retry_'+mid)
                payload=_parse_person_repair(raw)
                repaired=_apply_person_repair(fields,payload,refs,len(frames))
                observations=repaired.get('person_observations') or []
            if not observations:
                raise RuntimeError(f'{mid}: visual person pass returned no valid observations')
            repaired['people']=list(dict.fromkeys(list(repaired.get('people') or [])+[r['entity'] for r in observations if r.get('entity')]))
            # A visible human accidentally listed as an object is removed from the object inventory.
            if mid=='FEISHU_7_1_41aeb05dab':
                repaired['objects']=[x for x in repaired.get('objects') or [] if not re.search(r'\b(?:baby|infant|child|person|human)\b',str(x),re.I)]
            fields=normalize_fields(repaired)
            entry['person_frame_review']={'status':'repaired','method':'targeted_source_frame_repair_v2','repaired_people':len(fields.get('person_observations') or []),'repair_error':'','removed_count':0,'removed_items':[],'raw_model_output_preserved':True}
            entry['person_frame_review_raw']=raw
            changed.append('people')
        if mid in LOW_FACT_IDS:
            existing=fields.get('temporal_facts') or []
            max_add=max(1,min(5,8-len(existing)))
            aliases=[r.get('entity') for r in fields.get('person_observations') or [] if r.get('entity')]
            prompt=_build_temporal_fact_repair_prompt(len(frames))
            prompt+=f'\nSTRICT OUTPUT: return no more than {max_add} short JSON rows (maximum 8 total after merging), all JSON on one complete array, no markdown or commentary. Use exact person aliases only from this list: {json.dumps(aliases,ensure_ascii=True)}. Do not infer a person who is not visually distinguishable; do not treat an off-screen camera wearer as visible. Age-category words and exact ages are prohibited unless numeric age is readable in linked on-screen text. Prioritize distinct observed actions and before/after states, visible object locations, setting and activity. Existing facts (do not duplicate): {json.dumps(existing,ensure_ascii=True)}. Scene context for frame navigation only, not ground truth: {json.dumps({k:fields.get(k) for k in ("scene","location","activity","interaction_details","user_action","objects","environments")},ensure_ascii=True)}'
            raw=request_images(prompt,frames,'facts_'+mid)
            try: raw_rows=_parse_json_array(raw)
            except Exception:
                raw_rows,complete=_parse_json_array_prefix(raw)
                if not raw_rows or not complete: raise
            rows=_apply_temporal_fact_repair(raw_rows[:max_add],len(frames))
            merged=list(existing); seen={(str(x.get('subject_type','')).casefold(),str(x.get('subject','')).casefold(),str(x.get('predicate','')).casefold(),re.sub(r'\s+',' ',str(x.get('value','')).strip()).casefold()) for x in merged if isinstance(x,dict)}
            added=0
            for row in rows:
                key=(str(row.get('subject_type','')).casefold(),str(row.get('subject','')).casefold(),str(row.get('predicate','')).casefold(),re.sub(r'\s+',' ',str(row.get('value','')).strip()).casefold())
                if key not in seen and len(merged)<16: merged.append(row); seen.add(key); added+=1
            fields['temporal_facts']=merged
            err=_object_move_evidence_error(fields)
            if err: raise ValueError(err)
            entry['temporal_fact_repair']={'status':'added' if added else 'no_additional_facts','method':'targeted_source_frame_fact_repair_v2','facts_added':added,'total_facts':len(merged),'model_rows_returned':len(raw_rows)}
            entry['temporal_fact_repair_raw']=raw
            changed.append('facts')
        # Normalize age guesses throughout all structured content and preserve raw model output separately.
        fields=normalize_fields(fields)
        entry['fields']=fields
        entry['source']='cache_frame_quality_repair_v2'
        entry['custom_frame_review_version']='2026-10-18-v2'
        entry['custom_frame_review_at']=time.strftime('%Y-%m-%dT%H:%M:%S')
        # Persist the model output and corrected fields before crop grounding so a later crop failure cannot lose them.
        write_entry(mid,entry)
        person_rows=[r for r in fields.get('person_observations') or [] if r.get('entity') and r.get('appearance')]
        crop_result={'status':'no_people','people':len(person_rows),'grounded':0}
        if person_rows:
            candidate_frames=media.sample_visual_candidates(video,start,end,source_id=mid,count=7)
            if candidate_frames:
                labels=['person-crop-v1: '+r['entity']+' | '+r['appearance'] for r in person_rows[:8]]
                key=grounding.key(mid,labels,candidate_frames)
                cached,latency=grounding.get(key)
                lookup='hit'
                if cached is None:
                    cached=QwenPerceiver(api_url=API,cache_file=CACHE).ground_people(person_rows,candidate_frames)
                    grounding.put(key,cached); lookup='created'
                items=cached.get('items') or []
                crop_result={'status':lookup,'people':len(person_rows),'grounded':sum(i.get('status')=='grounded' for i in items if isinstance(i,dict)),'crop_candidates':sum(bool(i.get('frame_boxes')) for i in items if isinstance(i,dict)),'lookup_ms':latency}
        current=json.loads(CACHE.read_text(encoding='utf-8')).get(mid,{})
        current['person_crop_preheat']=crop_result
        write_entry(mid,current)
        print(mid,'updated='+','.join(changed),'people='+str(len(fields.get('person_observations') or [])),'facts='+str(len(fields.get('temporal_facts') or [])),'crop='+json.dumps(crop_result,ensure_ascii=False),flush=True)
    # Perform an age-language sweep on every reviewed custom memory, including rows outside targeted repair.
    snapshot=json.loads(CACHE.read_text(encoding='utf-8'))
    total=0
    for mid,e in snapshot.items():
        if not mid.startswith('FEISHU_') or not e.get('custom_frame_review_version'): continue
        before=e.get('fields') or {}; after=normalize_fields(before)
        if after!=before:
            e['fields']=after; e['source']='cache_frame_quality_repair_v2'; e['custom_frame_review_version']='2026-10-18-v2'; e['custom_frame_review_at']=time.strftime('%Y-%m-%dT%H:%M:%S'); write_entry(mid,e); total+=1
    print('AGE_LANGUAGE_SWEEP changed='+str(total),flush=True)
if __name__=='__main__': main()
