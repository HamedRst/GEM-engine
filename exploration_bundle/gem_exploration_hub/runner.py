"""Dispatch saved unified queries into unchanged native scientific engines."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import uuid
import yaml
from .pool import QueryPool


def run_one(pool_root,problem_id,*,landscape_override=None):
    pool=QueryPool(pool_root);q=pool.load(problem_id)
    name=q['problem_id'];out=pool.root/'results'/f'{name}_{uuid.uuid4().hex[:10]}'
    out.mkdir(parents=True,exist_ok=False)
    try:
        if q['problem_type']=='exploration.calculator':
            from gem_exploration.config import load_problem
            from gem_exploration.engine import plan_problem,run_problem
            # If the original landscape was not saved, fail closed rather than inventing FBA evidence.
            raw=q['payload']['internal_problem'];native=out/'native_calculator_problem.yaml'
            native.write_text(yaml.safe_dump(raw,sort_keys=False),encoding='utf-8')
            loaded=load_problem(native)
            src=landscape_override or (loaded.get('landscape') or {}).get('path')
            acquisition=None
            policy=(q.get('execution') or {}).get('new_fba_policy','never')
            if policy=='when_needed':
                from .acquisition import extend_archive
                a=(q['execution'].get('acquisition') or {})
                for required in ('source_csv','source_manifest','model_path','evaluator','query_group','min_exact_rows','max_new_designs'):
                    if required not in a:raise ValueError('Acquisition requires verified '+required)
                if str(Path(a['source_csv']).resolve())!=str(Path(src).resolve()):
                    raise ValueError('Acquisition archive differs from calculator scientific problem')
                acquisition=extend_archive(a['source_csv'],a['source_manifest'],a['model_path'],a['evaluator'],
                    pool.root,a['query_group'],int(a['min_exact_rows']),int(a['max_new_designs']),int(a.get('seed',42)))
                if not acquisition['sufficient_exact_rows']:
                    raise ValueError('FBA acquisition budget exhausted without sufficient exact evidence')
                src=acquisition['working_landscape']
            elif policy=='adaptive_qc':
                from .adaptive import run_adaptive, InsufficientEvidence
                from .acquisition import evaluator_identity, digest
                a=(q['execution'].get('acquisition') or {})
                for required in ('source_csv','source_manifest','model_path','evaluator','query_group',
                                 'max_new_designs_total','batch_size','max_rounds'):
                    if required not in a:raise ValueError('Adaptive acquisition requires '+required)
                if str(Path(a['source_csv']).expanduser().resolve())!=str(Path(src).expanduser().resolve()):
                    raise ValueError('Adaptive acquisition source differs from scientific problem')
                try:
                    acquisition=run_adaptive(loaded,str(src),a,pool.root)
                except InsufficientEvidence as exc:
                    (out/'ADAPTIVE_QC_UNRESOLVED.json').write_text(json.dumps(exc.record,indent=2,default=str))
                    raise
                src=acquisition['working_landscape']
            elif policy!='never':raise ValueError('Unsupported exact FBA acquisition policy')
            if not src or not Path(src).exists():raise FileNotFoundError('Matching exact landscape is required; no synthetic replacement allowed')
            plan=plan_problem(loaded,str(src),20260822)
            (out/'preflight.json').write_text(json.dumps({k:str(v) for k,v in plan.items() if k not in {'spec','df','scoped'}},indent=2))
            result=run_problem(loaded,str(src),str(out/'native_results'),False,20260822)
            summary={'problem_type':q['problem_type'],'problem_id':name,'result_directory':str(result),
                     'new_fba_calls':acquisition['new_fba_solves'] if acquisition else 0,
                     'new_designs':acquisition['new_designs'] if acquisition else 0,
                     'acquisition':acquisition,
                     'evidence_policy':'Original Calculator v0.5.0 evidence QC with optional verified exact FBA on copied source',
                     'surrogate_qc_status':acquisition.get('final_surrogate_qc') if acquisition else None}
        elif q['problem_type']=='exploration.activity':
            from .activity_queries import run_activity
            base=Path(__file__).resolve().parents[1]
            summary=run_activity(q['payload'],out,base)
            summary['problem_id']=name
        else:
            from gem_atlas_network.problems import Problem
            from gem_atlas_network.engine import execute
            p=Problem.model_validate(q['payload'])
            # Native engine resolves original registered models and preserves native render style.
            base=Path(__file__).resolve().parents[1]
            result=execute(p,base,out)
            summary={'problem_type':q['problem_type'],'problem_id':name,'result':result}
        summary.setdefault('result_directory',str(out))
        (out/'query.yaml').write_text(yaml.safe_dump(q,sort_keys=False),encoding='utf-8')
        (out/'run_summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
        return summary
    except Exception as exc:
        (out/'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n',encoding='utf-8')
        raise
