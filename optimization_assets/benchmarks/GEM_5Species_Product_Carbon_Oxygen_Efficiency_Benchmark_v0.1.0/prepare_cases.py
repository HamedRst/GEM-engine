from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd,yaml
from benchmark_paths import resolve_from_config,resolve_model_path

def run_stream(cmd,log_path:Path):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8') as log:
        log.write('\nCOMMAND: '+' '.join(map(str,cmd))+'\n'); p=subprocess.Popen(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout: print(line.rstrip(),flush=True); log.write(line)
        rc=p.wait()
    if rc: raise RuntimeError(f'Command failed ({rc}); see {log_path}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--study-config',default='study_config.yaml'); ap.add_argument('--cases',default=None); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
    cfgp=Path(a.study_config).expanduser().resolve(); cfg=yaml.safe_load(cfgp.read_text()) or {}; pkg=Path(__file__).resolve().parent
    prepared=resolve_from_config(cfgp,cfg['paths']['prepared_root']); prepared.mkdir(parents=True,exist_ok=True); names=list(cfg['cases']) if not a.cases else [x.strip() for x in a.cases.split(',') if x.strip()]
    required=int(cfg['growth_constraint'].get('required_archive_rows',50000)); rows=[]; errors={}
    print('='*105); print('5-SPECIES PRODUCT × CARBON EFFICIENCY × OXYGEN EFFICIENCY — PRIOR PREPARATION'); print('='*105)
    print('Archive carbon/oxygen relabeling: DISABLED (0 solves). Both efficiency objectives are deterministic from named frozen design coordinates; growth viability is frozen from the exact 50k archive.')
    for name in names:
        case=cfg['cases'][name]; out=prepared/name; pf=out/'landscape_preflight.json'; print('\n'+'-'*105); print(name,'—',case.get('label','')); print('-'*105)
        try:
            model=resolve_model_path(cfgp,cfg,name,case); run=resolve_from_config(cfgp,case['run_dir']); reuse=False
            if pf.exists() and not a.force:
                try:
                    old=json.loads(pf.read_text()); reuse=(old.get('status')=='PASS' and old.get('exact_replay_qc_pass') and int(old.get('rows',-1))==required and int(old.get('archive_additional_fba_solves_for_efficiency',-1))==0)
                except Exception: reuse=False
            if not reuse:
                out.mkdir(parents=True,exist_ok=True); run_stream([sys.executable,pkg/'engine_bridge.py','inspect','--run-dir',run,'--model-path',model,'--output-dir',out,'--study-config',cfgp,'--case-key',name,'--required-archive-rows',str(required)],out/'prepare.log')
            rep=json.loads(pf.read_text())
            if rep.get('status')!='PASS' or not rep.get('exact_replay_qc_pass'): raise RuntimeError('landscape_preflight is not PASS')
            problem=yaml.safe_load((out/'problem_config.yaml').read_text()) or {}; problem['problem']['qd_feature']=case.get('qd_feature'); problem['problem']['case_key']=name; problem['problem']['case_label']=case.get('label',name); (out/'problem_config.yaml').write_text(yaml.safe_dump(problem,sort_keys=False))
            rows.append({'case':name,'status':'PASS','rows':rep['rows'],'feasible':rep['feasible'],'dimension':rep['dimension'],'archive_maximum_growth':rep['archive_maximum_growth'],'growth_viability_floor':rep['growth_viability_floor'],'product_scale':rep['product_scale'],'archive_additional_fba_solves_for_efficiency':0,'model_path':rep['model_path']})
            print(f"PASS | {rep['rows']:,} rows | {rep['dimension']}D | growth floor={rep['growth_viability_floor']:.6g} | product scale={rep['product_scale']:.6g} | carbon={rep['carbon_feature']} | oxygen={rep['oxygen_feature']} | archive efficiency FBA solves=0")
        except Exception as exc:
            errors[name]=f'{type(exc).__name__}: {exc}'; rows.append({'case':name,'status':'FAIL','error':errors[name]}); print('FAIL:',errors[name])
    pd.DataFrame(rows).to_csv(prepared/'preflight_summary.csv',index=False); (prepared/'preflight_summary.json').write_text(json.dumps({'cases':rows,'errors':errors},indent=2))
    if errors: raise SystemExit(2)
    print('\nALL REQUESTED CASES PREPARED: PASS')
if __name__=='__main__': main()
