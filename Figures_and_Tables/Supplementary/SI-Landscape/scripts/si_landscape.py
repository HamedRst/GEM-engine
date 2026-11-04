#!/usr/bin/env python3
"""SI-Landscape: READ ONLY original 50K landscape-generation chunk adapter.

NO optimization results, NO FBA, NO timing interpolation. Seed 42/43/44,
5 species. The source directory _CANDIDATE_UNVERIFIED retains its provenance
flag even when a coherent 50K content dataset has been validated.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPECIES=('cglutamicum','cho','ecoli_antiepex','pputida','yeast')
SEEDS=(42,43,44)
SIZES=(10,25,50,100,250,500,1000,2500,5000,10000,25000,50000)
LABELS={'cglutamicum':r'$C.\ glutamicum$','cho':'CHO','ecoli_antiepex':r'$E.\ coli$','pputida':r'$P.\ putida$','yeast':'Yeast'}
COLORS={'cglutamicum':'#4477AA','cho':'#EE6677','ecoli_antiepex':'#228833','pputida':'#AA3377','yeast':'#AD921E'}
SOURCE_COLS=['seed','species','run_relpath','result_chunk_count','log_chunk_count','performance_relpath','provenance_status','start_time_utc']
CASE_COLS=['species','seed','evaluations','feasible_points','hypervolume_normalized','reference_50k_hv','knowledge_percent',
           'cumulative_evaluation_minutes','generation_wall_minutes','wall_evidence','worker_evidence','source_run','source_provenance',
           'results_chunks','logs_chunks','results_manifest_sha256']
AUDIT_COLS=['species','seed','status','source_path','provenance','n_chunks','n_event_logs','n_sample_ids','sample_id_start','sample_id_end',
            'feasible','failed','n_valid_worker_durations','worker_time_points','wall_time_points','endpoint_measured_wall_minutes',
            'event_timestamp_count','event_timestamp_distinct_ids','result_files_sha256_manifest','warnings']

def save_csv(path,items,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(items)

def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f: yield from csv.DictReader(f)

def sha256_manifest(files,archive):
    h=hashlib.sha256()
    for p in files:
        rp=str(p.relative_to(archive))
        h.update(rp.encode('utf-8')+b'\0')
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        h.update(b'\n')
    return h.hexdigest()

def checked_path(archive,rel):
    p=(archive/rel).resolve()
    if not p.is_relative_to(archive):raise ValueError('Path escapes the archive: '+str(rel))
    if any(s in str(p).lower() for s in ('/optimizations/','/optimizer/')):raise ValueError('Optimization path forbidden')
    return p

def native_roots(archive):
    base=archive/'Results'/'Landscape_Generation'
    if not base.exists():raise ValueError('Landscape_Generation folder missing')
    roots=[]
    for p in base.rglob('outputs_multiseed_50k'):
        if not p.is_dir():continue
        for seed in SEEDS:
            for sp in SPECIES:
                root=p/f'seed{seed}'/f'{sp}_50k'
                if not (root/'chunks'/'results').is_dir():continue
                results=sorted((root/'chunks'/'results').glob('results_chunk_*.parquet'))
                if not results:continue
                logs=sorted((root/'chunks'/'logs').glob('simulation_log_chunk_*.parquet'))
                perfparent=p.parent/'performance_multiseed_50k'/f'{sp}_50k_seed{seed}'
                perf=perfparent/'performance_macos_time.txt'
                roots.append({'seed':seed,'species':sp,'run_relpath':str(root.relative_to(archive)),
                    'result_chunk_count':len(results),'log_chunk_count':len(logs),
                    'performance_relpath':str(perf.relative_to(archive)) if perf.is_file() else '',
                    'provenance_status':'PROVISIONAL_CANDIDATE_DIRECTORY' if '_CANDIDATE_UNVERIFIED' in root.parts else 'NOT_YET_VERIFIED',
                    'start_time_utc':parse_verified_start(perfparent/'run_metadata.txt')})
    return roots



def parse_verified_start(metadata_file):
    """Only explicit timezone-aware ISO timestamps with a run-start field label."""
    if not metadata_file.is_file():return ''
    data=metadata_file.read_text(encoding='utf-8',errors='replace')[:200000]
    for line in data.splitlines():
        m=re.search(r'(?i)\b(?:run_start_utc|start_time_utc|started_at_utc|run_started_at|start_timestamp_utc)\b[\"\s]*[:=]\s*[\"\s]*([0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.\d+)?(?:Z|[+-][0-9]{2}:[0-9]{2}))',line)
        if m and iso_utc(m.group(1)):return m.group(1)
    return ''

def discover(archive,figroot):
    inp=figroot/'inputs';inp.mkdir(parents=True,exist_ok=True)
    allrows=native_roots(archive)
    save_csv(inp/'DISCOVERED_NATIVE_RUNS.csv',allrows,SOURCE_COLS)
    grouped=defaultdict(list)
    for r in allrows:grouped[(int(r['seed']),r['species'])].append(r)
    selection=[]
    for seed in SEEDS:
        for sp in SPECIES:
            opts=grouped.get((seed,sp),[])
            # Select only a *unique* native group; seed/model content will be validated in load_run.
            r=opts[0].copy() if len(opts)==1 else dict.fromkeys(SOURCE_COLS,'')
            r['seed']=seed;r['species']=sp
            if len(opts)>1:r['provenance_status']='AMBIGUOUS_MULTIPLE_NATIVE_RUNS'
            if not opts:r['provenance_status']='MISSING_NATIVE_RUN'
            selection.append(r)
    p=inp/'SOURCE_MAP.csv'
    if not p.exists() or 'run_relpath' not in p.read_text(encoding='utf-8-sig',errors='replace').splitlines()[0].split(','):
        if p.exists():p.rename(inp/'PREVIOUS_SOURCE_MAP_V2.csv')
        save_csv(p,selection,SOURCE_COLS)
    print('DISCOVERED:',len(allrows),'native run folders; expected 15; mapped:',sum(bool(x['run_relpath']) for x in selection))
    print('Source map:',p)
    if len(allrows)!=15:print('WARNING: missing/ambiguous native runs; see DISCOVERED_NATIVE_RUNS.csv')
    return p

def parse_sample_id(value):
    if value is None:raise ValueError('Missing sample_id')
    if isinstance(value,(int,)) and not isinstance(value,bool):return int(value)
    if isinstance(value,float) and math.isfinite(value) and value.is_integer():return int(value)
    s=str(value).strip()
    if re.fullmatch(r'[0-9]+',s):return int(s)
    m=re.fullmatch(r'(?:sample[_-]?|s[_-]?)([0-9]+)',s,re.IGNORECASE)
    if m:return int(m.group(1))
    raise ValueError('sample_id is not a sequential numeric/native sample ID: '+repr(s[:80]))

def valid_bool(value):
    if isinstance(value,bool):return value
    return str(value).strip().lower() in {'true','t','yes','1'}

def finite_float(value):
    try:v=float(value)
    except (ValueError,TypeError):return math.nan
    return v if math.isfinite(v) else math.nan

def iso_utc(value):
    if not value:return None
    try:
        t=datetime.fromisoformat(str(value).strip().replace('Z','+00:00'))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except (ValueError,OverflowError):return None

def exact_hv2d(points):
    """Exact area dominated by finite 2D maximization points in [0,1]^2 relative to (0,0)."""
    clean=[]
    for x,y in points:
        if not(math.isfinite(x) and math.isfinite(y)):continue
        if not(-1e-10<=x<=1+1e-10 and -1e-10<=y<=1+1e-10):raise ValueError('Point outside normalized range')
        clean.append((max(0.,min(1.,x)),max(0.,min(1.,y))))
    clean.sort(key=lambda z:z[0],reverse=True)
    best_y=0.;upper_x=1.;volume=0.
    for x,y in clean:
        if x<upper_x:
            volume+=(upper_x-x)*best_y
            upper_x=x
        if y>best_y:best_y=y
    return volume+upper_x*best_y

def parse_measured_wall_endpoint(path):
    """BSD /usr/bin/time -l or GNU time real endpoint only; never infer from CPU work."""
    if not path or not path.is_file():return None
    text=path.read_text(encoding='utf-8',errors='replace')[:150000]
    pats=[r'(?m)^\s*(\d+(?:\.\d+)?)\s+real\s+\d',
          r'(?mi)^\s*real\s+(\d+)m(\d+(?:\.\d+)?)s',
          r'(?mi)^\s*elapsed[_ ](?:wall[_ ]?)?(?:seconds|time)\s*[:=]\s*(\d+(?:\.\d+)?)\b',
          r'(?mi)^\s*(\d+(?:\.\d+)?)\s+seconds\s+elapsed\b']
    for j,pat in enumerate(pats):
        m=re.search(pat,text)
        if not m:continue
        val=(float(m.group(1))*60+float(m.group(2))) if j==1 else float(m.group(1))
        if math.isfinite(val) and val>=0:return val
    # GNU wall clock formats: 0:12.34, 1:02:12.34
    m=re.search(r'(?mi)^\s*elapsed \(wall clock\).*?:\s*(\d+(?::\d+){1,2}(?:\.\d+)?)\s*$',text)
    if m:
        bits=[float(x) for x in m.group(1).split(':')];seconds=0
        for v in bits:seconds=seconds*60+v
        return seconds
    return None

def load_run(row,archive,verify_logs=True):
    import pyarrow.parquet as pq
    sp=row['species'];seed=int(row['seed'])
    root=checked_path(archive,row['run_relpath'])
    if root.name!=f'{sp}_50k' or root.parent.name!=f'seed{seed}':
        raise ValueError('Seed/species folder inconsistent with map')
    result_dir=root/'chunks'/'results';log_dir=root/'chunks'/'logs'
    files=sorted(result_dir.glob('results_chunk_*.parquet'))
    if not files:raise ValueError('No native result chunks')
    if any(p.is_symlink() for p in files):raise ValueError('Symlinked result chunks not accepted')
    records={};worker={};warnings=[];total_runs_values=set();invalid=0
    for i,p in enumerate(files):
        table=pq.read_table(p,columns=['sample_id','biomass_flux','product_flux','feasible','runtime_seconds','total_runs'])
        columns=table.to_pydict();length=len(columns['sample_id'])
        for j in range(length):
            sid=parse_sample_id(columns['sample_id'][j])
            if sid in records:raise ValueError(f'Duplicate native sample_id {sid} in {p.name}')
            if sid<0:raise ValueError('Negative sample_id')
            a=finite_float(columns['biomass_flux'][j]);b=finite_float(columns['product_flux'][j])
            feasible=valid_bool(columns['feasible'][j]) and math.isfinite(a) and math.isfinite(b)
            if not feasible:invalid+=1
            records[sid]=(a,b,feasible)
            duration=finite_float(columns['runtime_seconds'][j])
            worker[sid]=duration if duration>=0 else math.nan
            tt=columns['total_runs'][j]
            if tt is not None:
                fv=finite_float(tt)
                if math.isfinite(fv):total_runs_values.add(fv)
        if (i+1)%80==0:print(f'  seed{seed} {sp}: {i+1}/{len(files)} result chunks',flush=True)
    ordered=sorted(records)
    if len(ordered)!=50000:
        raise ValueError(f'Expected exactly 50,000 native sample IDs; got {len(ordered)}')
    if ordered!=list(range(ordered[0],ordered[0]+50000)) or ordered[0] not in (0,1):
        raise ValueError('Native sample_id is not a contiguous 0-49999 or 1-50000 sequence')
    if total_runs_values and total_runs_values!={50000.0}:
        warnings.append('total_runs values not uniformly 50000: '+repr(sorted(total_runs_values)[:10]))
    if len(files)!=196:warnings.append(f'Unexpected native result chunk count: {len(files)} (196 in prior audit)')
    if len(records)!=len(ordered):raise ValueError('Internal ID mismatch')
    logfile=sorted(log_dir.glob('simulation_log_chunk_*.parquet')) if log_dir.is_dir() else []
    event_map={};duplicate_event=0
    if verify_logs and logfile:
        # 196 lightweight logging tables; read only required fields.
        for p in logfile:
            t=pq.read_table(p,columns=['sample_id','event_time_utc'])
            cols=t.to_pydict()
            for key,val in zip(cols['sample_id'],cols['event_time_utc']):
                sid=parse_sample_id(key)
                dt=iso_utc(val)
                if dt is None:continue
                if sid in event_map:
                    duplicate_event+=1
                    # Keep latest logged event, but don't use for timing if events are duplicated.
                    if dt>event_map[sid]:event_map[sid]=dt
                else:event_map[sid]=dt
        if set(event_map)-set(records):raise ValueError('Event-log sample IDs not in result chunks')
        if duplicate_event:warnings.append('Multiple event timestamps per sample: '+str(duplicate_event))
    timestamps_count=len(event_map)
    manifest_digest=sha256_manifest(files,archive)
    wall_file=checked_path(archive,row['performance_relpath']) if row.get('performance_relpath','').strip() else None
    wall_final=parse_measured_wall_endpoint(wall_file)
    return {'species':sp,'seed':seed,'root':root,'ids':ordered,'records':records,'worker':worker,
            'files':files,'logfiles':logfile,'event_map':event_map,'event_n':timestamps_count,
            'wall_final':wall_final,'start_time_utc':iso_utc(row.get('start_time_utc')),
            'digest':manifest_digest,'warnings':warnings,'invalid':invalid,'duplicate_event_count':duplicate_event,
            'provenance':'PROVISIONAL_CANDIDATE_DIRECTORY' if '_CANDIDATE_UNVERIFIED' in root.parts else 'NEEDS_NATIVE_METADATA_VERIFICATION'}

def compute_run(data,archive):
    import numpy as np
    ids=data['ids'];obs=[data['records'][i] for i in ids]
    valid=np.array([[a,b] for a,b,f in obs if f],dtype=float)
    if valid.shape[0]==0:raise ValueError('No valid feasible objectives')
    # Freeze objective scaling over full 50K, once for each species/seed.
    vmin=valid.min(axis=0);vmax=valid.max(axis=0);span=vmax-vmin
    if np.any(span<=0):raise ValueError('Zero objective dynamic range in full 50K')
    xy=[tuple((np.array([a,b])-vmin)/span) if f else (math.nan,math.nan) for a,b,f in obs]
    hv={n:exact_hv2d(xy[:n]) for n in SIZES}
    if not hv[50000]>0:raise ValueError('Full 50K hypervolume = 0')
    if any(v>hv[50000]+1e-10 for v in hv.values()):raise ValueError('Non-monotonic 50K hypervolume')
    if any(hv[SIZES[i]]>hv[SIZES[i+1]]+1e-10 for i in range(len(SIZES)-1)):
        raise ValueError('Prefix HV is not nondecreasing')
    feasible=np.cumsum([f for _,_,f in obs])
    dur=np.array([data['worker'][i] for i in ids],dtype=float)
    cum=np.cumsum(np.where(np.isfinite(dur),dur,0))
    missing=np.cumsum(~np.isfinite(dur))
    # Actual wall-clock trajectory is supported ONLY by start ISO + event timestamps.
    start=data['start_time_utc'];evt=data['event_map']
    eligible_wall=(start is not None and len(evt)==50000 and len(data['logfiles'])>0 and data['duplicate_event_count']==0)
    times=[]
    if eligible_wall:
        elapsed=np.array([(evt[i]-start).total_seconds() for i in ids],dtype=float)
        if np.any(elapsed < -1):eligible_wall=False
        else:times=np.maximum.accumulate(elapsed)
    wall_final=data['wall_final'];out=[]
    if eligible_wall and wall_final is not None:
        if abs(float(times[-1])-float(wall_final))>max(6.,float(wall_final)*.03):
            data['warnings'].append('Run start/event timestamps conflict with 50K /usr/bin/time elapsed; intermediate wall time withheld')
            eligible_wall=False
    for n in SIZES:
        t_worker=(float(cum[n-1])/60) if missing[n-1]==0 else ''
        t_wall=(float(times[n-1])/60) if eligible_wall else ''
        evidence='VERIFIED_START_AND_NATIVE_COMPLETION_EVENTS' if eligible_wall else 'MISSING_WALL_START_OR_EVENTS'
        if n==50000 and wall_final is not None:
            t_wall=float(wall_final)/60;evidence='BSD_TIME_MEASURED_50K_ENDPOINT' if not eligible_wall else 'START_EVENTS_AND_ENDPOINT_AGREE'
        out.append(dict(species=data['species'],seed=data['seed'],evaluations=n,
             feasible_points=int(feasible[n-1]),hypervolume_normalized=hv[n],reference_50k_hv=hv[50000],
             knowledge_percent=100.0*hv[n]/hv[50000],
             cumulative_evaluation_minutes=t_worker,generation_wall_minutes=t_wall,wall_evidence=evidence,
             worker_evidence='SUM_OF_MEASURED_PER_SAMPLE_RUNTIME' if t_worker!='' else 'INCOMPLETE_PER_SAMPLE_RUNTIME',
             source_run=str(data['root'].relative_to(archive)),source_provenance=data['provenance'],
             results_chunks=len(data['files']),logs_chunks=len(data['logfiles']),results_manifest_sha256=data['digest']))
    audit=dict(species=data['species'],seed=data['seed'],status='CONTENT_PASS_50000',
       source_path=str(data['root'].relative_to(archive)),provenance=data['provenance'],n_chunks=len(data['files']),
       n_event_logs=len(data['logfiles']),n_sample_ids=len(ids),sample_id_start=ids[0],sample_id_end=ids[-1],
       feasible=int(feasible[-1]),failed=data['invalid'],n_valid_worker_durations=int((~np.isnan(dur)).sum()),
       worker_time_points=sum(isinstance(v['cumulative_evaluation_minutes'],float) for v in out),
       wall_time_points=sum(isinstance(v['generation_wall_minutes'],float) for v in out),
       endpoint_measured_wall_minutes=wall_final/60 if wall_final is not None else '',
       event_timestamp_count=data['event_n'],event_timestamp_distinct_ids=len(evt),
       result_files_sha256_manifest=data['digest'],warnings=' | '.join(data['warnings']))
    return out,audit

def aggregate(case):
    import numpy as np
    grouped=defaultdict(list)
    for r in case:grouped[(r['species'],r['evaluations'])].append(r)
    out=[]
    for (species,size),rows in sorted(grouped.items()):
        hv=[float(r['knowledge_percent']) for r in rows]
        worker=[float(r['cumulative_evaluation_minutes']) for r in rows if r['cumulative_evaluation_minutes']!='']
        wall=[float(r['generation_wall_minutes']) for r in rows if r['generation_wall_minutes']!='']
        def avg_sd(values):
            return (float(np.mean(values)) if values else '',float(np.std(values,ddof=1)) if len(values)>1 else '')
        k,k_sd=avg_sd(hv);w,w_sd=avg_sd(worker);e,e_sd=avg_sd(wall)
        out.append(dict(species=species,evaluations=size,knowledge_mean_percent=k,knowledge_sd_percent=k_sd,
                        n_knowledge=len(hv),worker_mean_minutes=w,worker_sd_minutes=w_sd,n_worker=len(worker),
                        wall_mean_minutes=e,wall_sd_minutes=e_sd,n_wall=len(wall)))
    return out

def figure(records,out,provisional,time_mode='auto'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    if time_mode=='auto':
        # Need >=2 actual checkpoints across all 5 species to draw trustworthy wall-time curves.
        n_species=sum(any(x['species']==sp and x['evaluations']<50000 and x['wall_mean_minutes']!='' for x in records) for sp in SPECIES)
        time_mode='wall' if n_species==5 else 'worker'
    key='wall_mean_minutes' if time_mode=='wall' else 'worker_mean_minutes'
    sdkey='wall_sd_minutes' if time_mode=='wall' else 'worker_sd_minutes'
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.8,5.5))
    fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.26,wspace=.28)
    for sp in SPECIES:
        rr=sorted([r for r in records if r['species']==sp],key=lambda x:x['evaluations'])
        if not rr:continue
        x=[r['evaluations'] for r in rr];y=[r['knowledge_mean_percent'] for r in rr]
        ax.plot(x,y,marker='o',ms=3.4,lw=1.9,color=COLORS[sp],label=LABELS[sp])
        if all(r['knowledge_sd_percent']!='' for r in rr):
            err=np.array([r['knowledge_sd_percent'] for r in rr])
            ax.fill_between(x,np.maximum(0,np.array(y)-err),np.minimum(105,np.array(y)+err),color=COLORS[sp],alpha=.10,lw=0)
        rt=[r for r in rr if r[key]!='']
        if rt:
            bx.scatter([r['evaluations'] for r in rt],[r[key] for r in rt],s=13,color=COLORS[sp],zorder=3)
            # Only contiguous checkpoints connected; if missing internal checkpoints, leave gaps.
            seq=[]
            for r in rr+[dict(evaluations=50001,**{key:''})]:
                if r[key]=='':
                    if len(seq)>1:bx.plot([v['evaluations'] for v in seq],[v[key] for v in seq],color=COLORS[sp],lw=1.9)
                    seq=[]
                else:seq.append(r)
            # First plot line above only if no intermediate missing checkpoints.
            err=[r for r in rt if r[sdkey]!='']
            if err:bx.errorbar([r['evaluations'] for r in err],[r[key] for r in err],yerr=[r[sdkey] for r in err],
                               fmt='none',ecolor=COLORS[sp],capsize=2,lw=1,alpha=.7)
    for a in (ax,bx):
        a.set_xscale('log');a.set_xlim(9,58000)
        a.set_xticks([10,100,1000,5000,10000,50000],['10','100','1K','5K','10K','50K'])
        a.set_xlabel('Evaluated landscape points')
        a.grid(axis='y',alpha=.17)
        a.spines[['top','right']].set_visible(False)
    ax.set_ylim(0,105)
    ax.set_ylabel('Hypervolume relative to 50K (%)')
    ax.set_title('A   Landscape knowledge acquired',loc='left',fontweight='bold')
    bx.set_title('B   Generation computational cost' if time_mode=='worker' else 'B   Measured generation wall time',loc='left',fontweight='bold')
    bx.set_ylabel('Cumulative measured evaluation runtime (worker min)' if time_mode=='worker' else 'Elapsed generation wall time (min)')
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.50,.028))
    # Retain all timing/provenance explanations in data/metadata only.
    # The plot itself has neither an extra time disclaimer nor a provenance banner.
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'SI-Landscape.{ext}',dpi=300,bbox_inches='tight',pad_inches=.16,facecolor='white')
    plt.close(fig)
    return time_mode

def build(archive,figroot,time_mode='auto'):
    src=figroot/'inputs'/'SOURCE_MAP.csv'
    if not src.exists():discover(archive,figroot)
    rows=list(read_csv(src));case=[];audits=[];errors=[]
    expected={(seed,sp) for seed in SEEDS for sp in SPECIES}
    mapped={(int(r['seed']),r['species']) for r in rows if r.get('seed') and r.get('species')}
    for k in sorted(expected-mapped):errors.append(f'Missing SOURCE_MAP row for seed{k[0]} {k[1]}')
    if len(rows)!=15:errors.append(f'Expected 15 source rows, got {len(rows)}')
    for r in rows:
        if not r.get('run_relpath','').strip():
            errors.append(f'Missing native run: seed{r["seed"]} {r["species"]}');continue
        try:
            data=load_run(r,archive)
            output,audit=compute_run(data,archive)
            case.extend(output);audits.append(audit)
            print('  PASS native: seed%s %-14s %d samples; %d result chunks; worker %d/12; wall %d/12' %
                  (r['seed'],r['species'],len(data['ids']),len(data['files']),audit['worker_time_points'],audit['wall_time_points']),flush=True)
        except Exception as e:
            err=f'seed{r["seed"]} {r["species"]}: {type(e).__name__}: {e}'
            errors.append(err);print('  FAILED:',err,flush=True)
    m=figroot/'metadata';m.mkdir(parents=True,exist_ok=True)
    save_csv(m/'SOURCE_AND_METRIC_AUDIT.csv',audits,AUDIT_COLS)
    if errors:
        (m/'INCOMPLETE_SOURCES.txt').write_text('\n'.join(errors)+'\n')
        print('INCOMPLETE: '+' | '.join(errors))
        print('No publication-ready figure saved; input data not complete across 15 source runs.')
        return False
    (m/'INCOMPLETE_SOURCES.txt').unlink(missing_ok=True)
    if len(case)!=15*len(SIZES):raise ValueError('Unexpected case/size coverage')
    out=figroot/'outputs';out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'plot_data_casewise.csv',case,CASE_COLS)
    agg=aggregate(case)
    save_csv(out/'plot_data_aggregate.csv',agg,list(agg[0]))
    provisional=any(a['provenance']=='PROVISIONAL_CANDIDATE_DIRECTORY' for a in audits)
    mode=figure(agg,out,provisional,time_mode=time_mode)
    metadata={'figure':'SI-Landscape','scientific_source':'ONLY_ORIGINAL_LANDSCAPE_GENERATION',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'all_15_runs_content_validated':True,
        'original_run_provenance_certified':not provisional,'provisional_reason':'Source path labeled _CANDIDATE_UNVERIFIED; independent model/seed provenance evidence not supplied' if provisional else '',
        'species':SPECIES,'seeds':SEEDS,'evaluations':SIZES,'hypervolume':'Exact dominated 2D HV, same seed 50K min-max bounds, maximize biomass_flux and product_flux, fixed reference 0/0',
        'time_panel_mode':mode,'wall_time_source':'Original measured macOS /usr/bin/time endpoints; intermediate wall time only if original logged start and completion timestamps verified',
        'worker_time_source':'Sum of original sample-level runtime_seconds, not parallel wall time',
        'results':str(out.relative_to(figroot)),'source_integrity_manifest':str((m/'SOURCE_AND_METRIC_AUDIT.csv').relative_to(figroot))}
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('SUCCESS: figure outputs at',out)
    print('Right panel time basis:',mode)
    if provisional:print('PROVENANCE NOTE: original source status retained in metadata only (no plot banner).')
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,default=Path.home()/'Desktop'/'GEM_Publication_Archive')
    ap.add_argument('--figure-folder',type=Path)
    ap.add_argument('--mode',choices=['discover','generate'],default='generate')
    ap.add_argument('--time-mode',choices=['auto','worker','wall'],default='auto')
    a=ap.parse_args()
    archive=a.archive.expanduser().resolve()
    folder=(a.figure_folder or archive/'Figures_and_Tables'/'Supplementary'/'SI-Landscape').resolve()
    folder.mkdir(parents=True,exist_ok=True)
    if a.mode=='discover':discover(archive,folder);return
    if not build(archive,folder,a.time_mode):sys.exit(2)

if __name__=='__main__':main()
