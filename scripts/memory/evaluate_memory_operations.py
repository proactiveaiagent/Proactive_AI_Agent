"""Run memory-only CRUD/consolidation regressions and emit a compact acceptance report."""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUNS=ROOT/'data/egolife/runs'

CASES=[
    ('crud_graduation','test_crud_graduation.py',r'结果：(\d+) 通过 / (\d+) 失败'),
    ('compress_sort_combine','test_compress_sort_combine.py',r'测试完成：(\d+) 通过，(\d+) 失败'),
    ('hierarchical_retrieval','test_hierarchical_retrieval.py',r'结果：(\d+) 组通过 / (\d+) 失败'),
]
INTEGRATION_MARKERS=[
    'Daytime asynchronous translation completed before nightly summary',
    'Nightly generated summaries, indices, and profile from archived moments',
    'Scene close cleared Layer 1/2 while Layer 3/7 stayed queryable',
]

def main():
    results=[]
    for name,filename,pattern in CASES:
        proc=subprocess.run([sys.executable,str(HERE/filename)],cwd=HERE,
                            text=True,encoding='utf-8',errors='replace',
                            capture_output=True,timeout=180)
        match=re.search(pattern,proc.stdout)
        results.append({
            'name':name,'script':filename,'return_code':proc.returncode,
            'passed':int(match.group(1)) if match else None,
            'failed':int(match.group(2)) if match else None,
            'ok':proc.returncode==0 and bool(match) and int(match.group(2))==0,
            'output_tail':proc.stdout[-1200:] if proc.returncode or not match else '',
            'error_tail':proc.stderr[-1200:] if proc.stderr else '',
        })
    proc=subprocess.run([sys.executable,str(HERE/'test_day_night_integration.py')],
                        cwd=HERE,text=True,encoding='utf-8',errors='replace',
                        capture_output=True,timeout=180)
    markers=[item in proc.stdout for item in INTEGRATION_MARKERS]
    results.append({
        'name':'day_night_integration','script':'test_day_night_integration.py',
        'return_code':proc.returncode,'passed':sum(markers),'failed':len(markers)-sum(markers),
        'ok':proc.returncode==0 and all(markers),
        'output_tail':proc.stdout[-1200:] if proc.returncode or not all(markers) else '',
        'error_tail':proc.stderr[-1200:] if proc.stderr else '',
    })
    assertion_total=sum((r.get('passed') or 0) for r in results[:2])
    failure_total=sum((r.get('failed') or 0) for r in results[:2])
    report={
        'created_at':datetime.now().isoformat(timespec='seconds'),
        'scope':'isolated temporary memory stores; operation semantics, not semantic correctness of generated summaries/profile',
        'ok':all(r['ok'] for r in results),
        'operation_assertions_passed':assertion_total,
        'operation_assertions_failed':failure_total,
        'hierarchical_retrieval_cases_passed':results[2]['passed'],
        'day_night_integration_checks_passed':results[-1]['passed'],
        'cases':results,
    }
    RUNS.mkdir(parents=True,exist_ok=True)
    out=RUNS/f'memory_operations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('created_at','ok','operation_assertions_passed','operation_assertions_failed','hierarchical_retrieval_cases_passed','day_night_integration_checks_passed')},ensure_ascii=False,indent=2))
    print('report:',out)
    if not report['ok']:
        raise SystemExit(1)

if __name__=='__main__':
    main()
