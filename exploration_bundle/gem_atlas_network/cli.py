import argparse,json
from pathlib import Path
from .problems import parse_yaml,ROOT,import_bundle
from .engine import execute
from .batch import BatchRunner
from .ui_helpers import new_run_root

def main():
    ap=argparse.ArgumentParser(description='GEM Atlas: Network, Genes, Routes or FBA jobs')
    ap.add_argument('problems',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,default=Path('outputs/cli'))
    ap.add_argument('--model-root',type=Path)
    ap.add_argument('--batch',action='store_true')
    ap.add_argument('--rerun',action='store_true')
    args=ap.parse_args();jobs=[]
    for file in args.problems:
        if file.suffix=='.zip':
            for yaml,p in import_bundle(file,args.output/'imported'/file.stem):jobs.append((p,yaml.parent,args.output/'imported'/file.stem,None))
        else:
            file=file.resolve();p=parse_yaml(file.read_bytes());allowed=args.model_root or (ROOT if file.is_relative_to(ROOT) else file.parent)
            jobs.append((p,file.parent,allowed,None))
    if args.batch or len(jobs)>1:
        rows=BatchRunner(args.output).run(jobs,resume=not args.rerun,progress=lambda i,n,r:print(f"{i}/{n}: {r['problem_id']} {r['status']}"))
        if any(r['status']=='failed' for r in rows.values()):raise SystemExit(1)
    else:
        p,base,allowed,supplied=jobs[0];run_root=new_run_root(args.output,p.problem_id);r=execute(p,base,run_root,allowed,supplied)
        print(json.dumps({'problem_id':r['problem_id'],'solver_calls':r['solver_calls'],'output':str(run_root/r['output_directory'])},indent=2))
        if any(s['status']!='optimal' for s in r['solutions'].values()):raise SystemExit(1)
if __name__=='__main__':main()
